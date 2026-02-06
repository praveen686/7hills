//! Binance Perp Session Capture - Spot + Perp + Funding orchestrator.
//!
//! This module captures all data needed for funding rate arbitrage:
//! - Spot bookTicker (reference price)
//! - Perp bookTicker or depth (trading price / L2)
//! - Perp depth (optional, for slippage modeling)
//! - Funding rate stream (funding signals)
//!
//! The combined capture enables basis calculation:
//! ```text
//! Basis = (Perp_Price - Spot_Price) / Spot_Price
//! ```
//!
//! ## Manifest Features (Phase 2A)
//! - Per-stream event counts
//! - Per-stream first/last timestamps
//! - File SHA256 digests for integrity verification
//!
//! Directory structure:
//! ```text
//! data/perp_sessions/{tag}/
//! ├── session_manifest.json
//! ├── BTCUSDT/
//! │   ├── spot_quotes.jsonl      # Spot bookTicker
//! │   ├── perp_quotes.jsonl      # Perp bookTicker (or perp_depth.jsonl)
//! │   ├── perp_depth.jsonl       # Perp L2 depth (if --with-depth)
//! │   ├── funding.jsonl          # Funding rate stream
//! │   └── manifest.json          # Per-symbol manifest
//! └── ETHUSDT/
//!     └── ...
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::binance_capture;
use crate::binance_funding_capture;
use crate::binance_perp_capture;

// =============================================================================
// Stream Digest (Phase 2A: per-file integrity + timestamps)
// =============================================================================

/// Digest information for a captured stream file.
/// Enables integrity verification and temporal bounds checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDigest {
    /// Relative path to the file (e.g., "BTCUSDT/funding.jsonl")
    pub file_path: String,
    /// Number of events/lines in the file
    pub event_count: usize,
    /// SHA256 hash of the file contents (hex-encoded)
    pub sha256: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// First event timestamp (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_event_ts: Option<String>,
    /// Last event timestamp (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_ts: Option<String>,
}

impl StreamDigest {
    /// Compute digest for a JSONL file, extracting timestamps from "ts" field.
    pub fn from_jsonl_file(path: &Path, relative_path: &str) -> Result<Self> {
        use std::fs::File;

        let metadata = std::fs::metadata(path).with_context(|| format!("stat file: {:?}", path))?;
        let size_bytes = metadata.len();

        let mut event_count = 0;
        let mut first_ts: Option<String> = None;
        let mut last_ts: Option<String> = None;

        // Read file for timestamp extraction
        let file_for_parse = File::open(path).with_context(|| format!("open file: {:?}", path))?;
        let parse_reader = BufReader::new(file_for_parse);

        for line_result in parse_reader.lines() {
            let line = line_result?;
            event_count += 1;

            // Extract timestamp from JSON line
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line)
                && let Some(ts) = json.get("ts").and_then(|v| v.as_str())
            {
                if first_ts.is_none() {
                    first_ts = Some(ts.to_string());
                }
                last_ts = Some(ts.to_string());
            }
        }

        // Compute SHA256 hash (single pass over file bytes)
        let file_bytes = std::fs::read(path)?;
        let hash_result = Sha256::digest(&file_bytes);
        let sha256 = hex::encode(hash_result);

        Ok(StreamDigest {
            file_path: relative_path.to_string(),
            event_count,
            sha256,
            size_bytes,
            first_event_ts: first_ts,
            last_event_ts: last_ts,
        })
    }

    /// Compute digest for an existing file without timestamp extraction (binary files).
    pub fn from_file_no_timestamps(path: &Path, relative_path: &str) -> Result<Self> {
        let metadata = std::fs::metadata(path).with_context(|| format!("stat file: {:?}", path))?;
        let size_bytes = metadata.len();

        let file_bytes = std::fs::read(path)?;
        let hash_result = Sha256::digest(&file_bytes);
        let sha256 = hex::encode(hash_result);

        // Count lines for event_count
        let event_count = file_bytes.iter().filter(|&&b| b == b'\n').count();

        Ok(StreamDigest {
            file_path: relative_path.to_string(),
            event_count,
            sha256,
            size_bytes,
            first_event_ts: None,
            last_event_ts: None,
        })
    }
}

/// Configuration for perp session capture.
#[derive(Clone)]
pub struct PerpSessionConfig {
    /// Symbols to capture (e.g., BTCUSDT, ETHUSDT)
    pub symbols: Vec<String>,
    /// Output directory for session data
    pub out_dir: PathBuf,
    /// Capture duration in seconds
    pub duration_secs: u64,
    /// Include spot capture (for basis calculation)
    pub include_spot: bool,
    /// Include perp depth (L2 order book)
    pub include_depth: bool,
    /// Include aggTrades capture (for SLRT toxicity/VPIN)
    pub include_trades: bool,
    /// Price exponent for depth capture
    pub price_exponent: i8,
    /// Quantity exponent for depth capture
    pub qty_exponent: i8,
    /// Binance API key (required for SBE streams if using include_trades)
    pub api_key: Option<String>,
}

// Custom Debug impl to redact api_key (avoid accidental logging of secrets)
impl std::fmt::Debug for PerpSessionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerpSessionConfig")
            .field("symbols", &self.symbols)
            .field("out_dir", &self.out_dir)
            .field("duration_secs", &self.duration_secs)
            .field("include_spot", &self.include_spot)
            .field("include_depth", &self.include_depth)
            .field("include_trades", &self.include_trades)
            .field("price_exponent", &self.price_exponent)
            .field("qty_exponent", &self.qty_exponent)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

impl Default for PerpSessionConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_string()],
            out_dir: PathBuf::from("data/perp_sessions"),
            duration_secs: 3600, // 1 hour default
            include_spot: true,
            include_depth: false,  // bookTicker is sufficient for MVP
            include_trades: false, // Trades capture disabled by default
            price_exponent: -2,
            qty_exponent: -8,
            api_key: None,
        }
    }
}

/// Per-symbol capture statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolCaptureStats {
    pub symbol: String,
    pub spot_events: usize,
    pub perp_events: usize,
    pub depth_events: usize,
    pub funding_events: usize,
    pub trade_events: usize,
    pub last_spot_bid: f64,
    pub last_spot_ask: f64,
    pub last_perp_bid: f64,
    pub last_perp_ask: f64,
    pub last_funding_rate: f64,
    pub funding_settlements: usize,
    pub basis_bps: f64,
    /// Output directory for this symbol (for digest computation)
    #[serde(skip)]
    pub out_dir: Option<PathBuf>,
    /// Whether depth was captured (vs bookticker)
    #[serde(skip)]
    pub depth_captured: bool,
    /// Whether trades were captured
    #[serde(skip)]
    pub trades_captured: bool,
}

impl SymbolCaptureStats {
    /// Calculate basis in basis points.
    pub fn calculate_basis_bps(&mut self) {
        if self.last_spot_ask > 0.0 && self.last_perp_bid > 0.0 {
            let spot_mid = (self.last_spot_bid + self.last_spot_ask) / 2.0;
            let perp_mid = (self.last_perp_bid + self.last_perp_ask) / 2.0;
            if spot_mid > 0.0 {
                self.basis_bps = ((perp_mid - spot_mid) / spot_mid) * 10000.0;
            }
        }
    }
}

/// Session-level statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionStats {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_secs: f64,
    pub symbols: Vec<SymbolCaptureStats>,
    pub total_spot_events: usize,
    pub total_perp_events: usize,
    pub total_funding_events: usize,
    pub all_symbols_complete: bool,
}

/// Canonical quote schema version identifier.
/// Any manifest without this exact value is considered legacy and must be rejected.
pub const CANONICAL_QUOTE_SCHEMA: &str = "canonical_v1";

/// =============================================================================
/// PerpSessionManifest – Compatibility Matrix
/// =============================================================================
///
/// Purpose
/// - Defines the on-disk contract for perp capture sessions (manifest + stream files).
/// - The loader MUST enforce schema_version and quote_schema so downstream
///   pipelines never silently ingest ambiguous/legacy layouts.
///
/// Versioning rules
/// - `schema_version` is a hard compatibility boundary:
///     - If manifest.schema_version != PERP_SESSION_MANIFEST_SCHEMA_VERSION → HARD FAIL.
/// - `quote_schema` is a hard compatibility boundary:
///     - Must be Some("canonical_v1") → otherwise HARD FAIL.
/// - Backward compatibility is NOT supported across breaking versions.
///   Recapture or run an explicit migration tool (if/when provided).
///
/// Schema versions
/// - v1 (legacy; unsupported)
///     - Pre-canonical quote fields and/or legacy naming.
///     - May lack quote_schema entirely.
///     - Rejected by loader.
///
/// - v2 (legacy; unsupported)
///     - Transitional format used before canonical quote stabilization.
///     - Still incompatible with canonical mantissa-based QuoteEvent assumptions.
///     - Rejected by loader.
///
/// - v3 (current; supported)
///     - Canonical quote schema enforced via quote_schema == "canonical_v1".
///     - Manifest includes capture configuration (include_spot/include_depth,
///       price_exponent/qty_exponent).
///     - Symbol entries enumerate stream files and event counts.
///     - Optional `digests` enables Phase 2A integrity + temporal bounds.
///     - Loader guarantees:
///         - schema_version == 3
///         - quote_schema == Some("canonical_v1")
///         - Downstream can assume canonical mantissa-based quoting semantics.
///
/// Planned changes (when bumping beyond v3)
/// - Any rename of fields, semantic change to file paths, changes to units/meaning
///   of exponents, or digest layout changes MUST bump schema_version.
/// - When bumping, update:
///     - PERP_SESSION_MANIFEST_SCHEMA_VERSION
///     - load_perp_session_manifest() enforcement message
///     - fixtures + tests asserting against the constant
///
/// - v4 (current; supported)
///     - Adds include_trades option and capabilities field
///     - Adds trades_file and trade_events to SymbolManifestEntry
///     - Backward compatible: v3 manifests still load if capabilities is absent
pub const PERP_SESSION_MANIFEST_SCHEMA_VERSION: u32 = 4;

/// Session manifest for perp capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionManifest {
    pub schema_version: u32,
    /// Quote schema identifier. Must be "canonical_v1" for valid captures.
    pub quote_schema: String,
    pub created_at_utc: String,
    pub session_id: String,
    pub capture_mode: String,
    pub duration_secs: f64,
    pub symbols: Vec<SymbolManifestEntry>,
    pub config: PerpSessionManifestConfig,
    /// Session capabilities (for SLRT feature requirements)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub capabilities: Option<SessionCapabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolManifestEntry {
    pub symbol: String,
    pub spot_file: Option<String>,
    pub perp_file: String,
    pub depth_file: Option<String>,
    pub funding_file: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub trades_file: Option<String>,
    pub spot_events: usize,
    pub perp_events: usize,
    pub depth_events: usize,
    pub funding_events: usize,
    #[serde(default)]
    pub trade_events: usize,
    /// Per-stream digests (Phase 2A: integrity + temporal bounds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digests: Option<SymbolDigests>,
}

/// Per-stream file digests for a symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDigests {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<StreamDigest>,
    pub perp: StreamDigest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<StreamDigest>,
    pub funding: StreamDigest,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trades: Option<StreamDigest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionManifestConfig {
    pub include_spot: bool,
    pub include_depth: bool,
    #[serde(default)]
    pub include_trades: bool,
    pub price_exponent: i8,
    pub qty_exponent: i8,
}

/// Session capabilities (for SLRT feature requirements).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCapabilities {
    /// Has depth data (L2 order book)
    pub has_depth: bool,
    /// Has aggTrades data (for toxicity/VPIN)
    pub has_trades: bool,
    /// Has spot quotes (for basis calculation)
    pub has_spot: bool,
    /// Has funding rate stream
    pub has_funding: bool,
}

/// Compute digests for all captured files for a symbol.
/// Returns None if any required file is missing or unreadable.
fn compute_symbol_digests(
    sym_dir: &Path,
    sym_upper: &str,
    include_spot: bool,
    include_depth: bool,
    include_trades: bool,
) -> Option<SymbolDigests> {
    // Perp file (required)
    let perp_file = if include_depth {
        "perp_depth.jsonl"
    } else {
        "perp_quotes.jsonl"
    };
    let perp_path = sym_dir.join(perp_file);
    let perp_digest =
        match StreamDigest::from_jsonl_file(&perp_path, &format!("{}/{}", sym_upper, perp_file)) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to compute perp digest for {}: {}", sym_upper, e);
                return None;
            }
        };

    // Funding file (required)
    let funding_path = sym_dir.join("funding.jsonl");
    let funding_digest =
        match StreamDigest::from_jsonl_file(&funding_path, &format!("{}/funding.jsonl", sym_upper))
        {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to compute funding digest for {}: {}", sym_upper, e);
                return None;
            }
        };

    // Spot file (optional)
    let spot_digest = if include_spot {
        let spot_path = sym_dir.join("spot_quotes.jsonl");
        match StreamDigest::from_jsonl_file(&spot_path, &format!("{}/spot_quotes.jsonl", sym_upper))
        {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::warn!("Failed to compute spot digest for {}: {}", sym_upper, e);
                None
            }
        }
    } else {
        None
    };

    // Depth file (optional, only if include_depth)
    let depth_digest = if include_depth {
        // Depth is same as perp when include_depth=true
        Some(perp_digest.clone())
    } else {
        None
    };

    // Trades file (optional, only if include_trades)
    let trades_digest = if include_trades {
        let trades_path = sym_dir.join("agg_trades.jsonl");
        match StreamDigest::from_jsonl_file(
            &trades_path,
            &format!("{}/agg_trades.jsonl", sym_upper),
        ) {
            Ok(d) => Some(d),
            Err(e) => {
                tracing::warn!("Failed to compute trades digest for {}: {}", sym_upper, e);
                None
            }
        }
    } else {
        None
    };

    Some(SymbolDigests {
        spot: spot_digest,
        perp: perp_digest,
        depth: depth_digest,
        funding: funding_digest,
        trades: trades_digest,
    })
}

/// Result of trades validation.
#[derive(Debug)]
pub struct TradesValidationResult {
    /// Number of trades validated
    pub trade_count: usize,
    /// Whether timestamps are monotonically non-decreasing
    pub is_monotonic: bool,
    /// Number of out-of-order timestamps found
    pub violations: usize,
    /// First violation details (if any)
    pub first_violation: Option<String>,
}

/// Validate that trades file has monotonically non-decreasing timestamps.
///
/// Returns validation result with details about any violations.
/// This is a post-capture integrity check - if violated, has_trades should be false.
fn validate_trades_monotonic(trades_path: &Path) -> TradesValidationResult {
    use std::io::{BufRead, BufReader};

    let mut result = TradesValidationResult {
        trade_count: 0,
        is_monotonic: true,
        violations: 0,
        first_violation: None,
    };

    let file = match std::fs::File::open(trades_path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                "Cannot open trades file for validation: {:?} - {}",
                trades_path,
                e
            );
            result.is_monotonic = false;
            result.first_violation = Some(format!("Cannot open file: {}", e));
            return result;
        }
    };

    let reader = BufReader::new(file);
    let mut prev_ts_ms: Option<i64> = None;

    for (line_num, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!("Read error at line {}: {}", line_num + 1, e);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        result.trade_count += 1;

        // Extract timestamp from JSON (look for "ts" field)
        // Format: "ts":"2026-01-25T13:34:27.123456Z"
        let ts_ms = if let Some(ts_start) = line.find("\"ts\":\"") {
            let ts_str_start = ts_start + 6;
            if let Some(ts_end) = line[ts_str_start..].find('"') {
                let ts_str = &line[ts_str_start..ts_str_start + ts_end];
                // Parse ISO8601 timestamp
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts_str) {
                    dt.timestamp_millis()
                } else {
                    continue; // Skip unparseable timestamps
                }
            } else {
                continue;
            }
        } else {
            continue;
        };

        // Check monotonicity
        if let Some(prev) = prev_ts_ms
            && ts_ms < prev
        {
            result.violations += 1;
            if result.first_violation.is_none() {
                result.first_violation = Some(format!(
                    "Line {}: ts {} < prev ts {} (delta = {}ms)",
                    line_num + 1,
                    ts_ms,
                    prev,
                    prev - ts_ms
                ));
            }
            result.is_monotonic = false;
        }
        prev_ts_ms = Some(ts_ms);
    }

    result
}

/// Capture a perp session with Spot + Perp + Funding data.
///
/// NOTE: This function creates its own timestamped subdirectory.
/// For managed segments (Phase 2B), use `capture_to_segment()` instead.
pub async fn capture_perp_session(config: PerpSessionConfig) -> Result<PerpSessionStats> {
    let start_time = Utc::now();

    // Create session directory
    let session_tag = format!("perp_{}", start_time.format("%Y%m%d_%H%M%S"));
    let session_dir = config.out_dir.join(&session_tag);
    std::fs::create_dir_all(&session_dir)
        .with_context(|| format!("create session dir: {:?}", session_dir))?;

    capture_to_segment_inner(&session_dir, &config, start_time).await
}

/// Capture directly into an existing segment directory (Phase 2B).
///
/// Unlike `capture_perp_session`, this does NOT create a subdirectory.
/// The caller (ManagedSegment) is responsible for directory creation.
pub async fn capture_to_segment(
    segment_dir: &Path,
    config: &PerpSessionConfig,
) -> Result<PerpSessionStats> {
    let start_time = Utc::now();
    capture_to_segment_inner(segment_dir, config, start_time).await
}

/// Early manifest for session start (Safeguard A).
/// Written before capture begins to prevent "capture ran but no manifest".
#[derive(Debug, Serialize)]
struct EarlyManifest {
    session_id: String,
    start_time_utc: String,
    intended_duration_secs: u64,
    include_trades: bool,
    include_depth: bool,
    include_spot: bool,
    symbols: Vec<String>,
    status: String,
    api_key_present: bool,
}

/// Inner capture implementation shared by both entry points.
async fn capture_to_segment_inner(
    session_dir: &Path,
    config: &PerpSessionConfig,
    start_time: DateTime<Utc>,
) -> Result<PerpSessionStats> {
    let session_id = uuid::Uuid::new_v4().to_string();

    tracing::info!("=== Perp Session Capture ===");
    tracing::info!("Session ID: {}", &session_id[..8]);
    tracing::info!("Symbols: {:?}", config.symbols);
    tracing::info!("Duration: {}s", config.duration_secs);
    tracing::info!("Output: {:?}", session_dir);
    tracing::info!("Include trades: {}", config.include_trades);
    tracing::info!("API key present: {}", config.api_key.is_some());

    // Safeguard A: Write early manifest before capture begins
    let early_manifest = EarlyManifest {
        session_id: session_id.clone(),
        start_time_utc: start_time.to_rfc3339(),
        intended_duration_secs: config.duration_secs,
        include_trades: config.include_trades,
        include_depth: config.include_depth,
        include_spot: config.include_spot,
        symbols: config.symbols.clone(),
        status: "in_progress".to_string(),
        api_key_present: config.api_key.is_some(),
    };
    let early_manifest_path = session_dir.join("session_manifest_early.json");
    std::fs::write(
        &early_manifest_path,
        serde_json::to_string_pretty(&early_manifest)?,
    )?;
    tracing::info!("Early manifest written: {:?}", early_manifest_path);

    // Spawn capture tasks for each symbol
    let mut handles = Vec::new();
    for symbol in &config.symbols {
        let sym = symbol.clone();
        let sym_dir = session_dir.join(symbol.to_uppercase());
        std::fs::create_dir_all(&sym_dir)?;

        let duration = config.duration_secs;
        let include_spot = config.include_spot;
        let include_depth = config.include_depth;
        let include_trades = config.include_trades;
        let price_exp = config.price_exponent;
        let qty_exp = config.qty_exponent;
        let api_key = config.api_key.clone();

        let handle = tokio::spawn(async move {
            capture_symbol(
                &sym,
                &sym_dir,
                duration,
                include_spot,
                include_depth,
                include_trades,
                price_exp,
                qty_exp,
                api_key.as_deref(),
            )
            .await
        });
        handles.push((symbol.clone(), handle));
    }

    // Wait for all captures to complete
    let mut symbol_stats = Vec::new();
    let mut total_spot = 0;
    let mut total_perp = 0;
    let mut total_funding = 0;
    let mut total_trades = 0;
    let mut all_complete = true;

    for (symbol, handle) in handles {
        match handle.await {
            Ok(Ok(stats)) => {
                total_spot += stats.spot_events;
                total_perp += stats.perp_events;
                total_funding += stats.funding_events;
                total_trades += stats.trade_events;
                symbol_stats.push(stats);
            }
            Ok(Err(e)) => {
                tracing::error!("Capture failed for {}: {}", symbol, e);
                all_complete = false;
            }
            Err(e) => {
                tracing::error!("Task panicked for {}: {}", symbol, e);
                all_complete = false;
            }
        }
    }

    let end_time = Utc::now();
    let duration_secs = (end_time - start_time).num_milliseconds() as f64 / 1000.0;

    // Build manifest entries with digests
    let mut manifest_entries = Vec::new();
    for s in &symbol_stats {
        let sym_upper = s.symbol.to_uppercase();
        let sym_dir = session_dir.join(&sym_upper);

        // Compute digests for each captured file
        let digests = compute_symbol_digests(
            &sym_dir,
            &sym_upper,
            config.include_spot,
            s.depth_captured,
            s.trades_captured,
        );

        let entry = SymbolManifestEntry {
            symbol: s.symbol.clone(),
            spot_file: if config.include_spot {
                Some(format!("{}/spot_quotes.jsonl", sym_upper))
            } else {
                None
            },
            perp_file: if s.depth_captured {
                format!("{}/perp_depth.jsonl", sym_upper)
            } else {
                format!("{}/perp_quotes.jsonl", sym_upper)
            },
            depth_file: if s.depth_captured {
                Some(format!("{}/perp_depth.jsonl", sym_upper))
            } else {
                None
            },
            funding_file: format!("{}/funding.jsonl", sym_upper),
            trades_file: if s.trades_captured {
                Some(format!("{}/agg_trades.jsonl", sym_upper))
            } else {
                None
            },
            spot_events: s.spot_events,
            perp_events: s.perp_events,
            depth_events: s.depth_events,
            funding_events: s.funding_events,
            trade_events: s.trade_events,
            digests,
        };
        manifest_entries.push(entry);
    }

    // Build capabilities based on OBSERVED OUTPUT (not config)
    // has_trades requires: file exists + trades_captured > 0 + digest present + monotonic timestamps
    let mut observed_has_trades = false;
    let mut trades_validation_failed = false;

    if total_trades > 0 {
        // Check all symbols for valid trades
        let mut all_trades_valid = true;
        for entry in &manifest_entries {
            if entry.trade_events > 0 {
                // Check if digest is present
                let has_digest = entry
                    .digests
                    .as_ref()
                    .map(|d| d.trades.is_some())
                    .unwrap_or(false);

                if !has_digest {
                    tracing::warn!(
                        "{}: trades captured but digest missing, marking has_trades=false",
                        entry.symbol
                    );
                    all_trades_valid = false;
                    continue;
                }

                // Validate monotonic timestamps
                let sym_upper = entry.symbol.to_uppercase();
                let trades_path = session_dir.join(&sym_upper).join("agg_trades.jsonl");
                let validation = validate_trades_monotonic(&trades_path);

                if !validation.is_monotonic {
                    tracing::error!(
                        "{}: TRADES TIMESTAMP VALIDATION FAILED - {} violations in {} trades",
                        entry.symbol,
                        validation.violations,
                        validation.trade_count
                    );
                    if let Some(ref detail) = validation.first_violation {
                        tracing::error!("  First violation: {}", detail);
                    }
                    all_trades_valid = false;
                    trades_validation_failed = true;
                } else {
                    tracing::info!(
                        "{}: trades timestamp validation passed ({} trades, monotonic)",
                        entry.symbol,
                        validation.trade_count
                    );
                }
            }
        }

        observed_has_trades = all_trades_valid && !trades_validation_failed;
    }

    if trades_validation_failed {
        tracing::warn!(
            "SESSION has_trades=false due to timestamp validation failure. \
             Trades file may still be useful for analysis but not for certified replay."
        );
    }

    let capabilities = SessionCapabilities {
        has_depth: manifest_entries.iter().any(|e| e.depth_events > 0),
        has_trades: observed_has_trades,
        has_spot: manifest_entries.iter().any(|e| e.spot_events > 0),
        has_funding: manifest_entries.iter().any(|e| e.funding_events > 0),
    };

    // Write session manifest
    let manifest = PerpSessionManifest {
        schema_version: PERP_SESSION_MANIFEST_SCHEMA_VERSION,
        quote_schema: CANONICAL_QUOTE_SCHEMA.to_string(),
        created_at_utc: start_time.to_rfc3339(),
        session_id: session_id.clone(),
        capture_mode: "perp_session".to_string(),
        duration_secs,
        symbols: manifest_entries,
        config: PerpSessionManifestConfig {
            include_spot: config.include_spot,
            include_depth: config.include_depth,
            include_trades: config.include_trades,
            price_exponent: config.price_exponent,
            qty_exponent: config.qty_exponent,
        },
        capabilities: Some(capabilities),
    };

    let manifest_path = session_dir.join("session_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;
    tracing::info!("Session manifest written: {:?}", manifest_path);

    let stats = PerpSessionStats {
        session_id,
        start_time,
        end_time,
        duration_secs,
        symbols: symbol_stats,
        total_spot_events: total_spot,
        total_perp_events: total_perp,
        total_funding_events: total_funding,
        all_symbols_complete: all_complete,
    };

    // Print summary
    tracing::info!("=== Session Summary ===");
    tracing::info!("Duration: {:.1}s", stats.duration_secs);
    tracing::info!("Spot events: {}", stats.total_spot_events);
    tracing::info!("Perp events: {}", stats.total_perp_events);
    tracing::info!("Funding events: {}", stats.total_funding_events);
    for sym_stat in &stats.symbols {
        tracing::info!(
            "  {}: basis={:.2}bps, funding={:.4}%",
            sym_stat.symbol,
            sym_stat.basis_bps,
            sym_stat.last_funding_rate * 100.0
        );
    }

    Ok(stats)
}

/// Capture result enum for unified task handling.
enum CaptureResult {
    Spot {
        events: usize,
    },
    Perp {
        events: usize,
        bid: f64,
        ask: f64,
    },
    Funding {
        events: usize,
        rate: f64,
        settlements: usize,
    },
    Trades {
        events: usize,
    },
}

/// Capture all streams for a single symbol.
#[allow(clippy::too_many_arguments)]
async fn capture_symbol(
    symbol: &str,
    out_dir: &Path,
    duration_secs: u64,
    include_spot: bool,
    include_depth: bool,
    include_trades: bool,
    price_exponent: i8,
    qty_exponent: i8,
    api_key: Option<&str>,
) -> Result<SymbolCaptureStats> {
    let mut stats = SymbolCaptureStats {
        symbol: symbol.to_string(),
        spot_events: 0,
        perp_events: 0,
        depth_events: 0,
        funding_events: 0,
        trade_events: 0,
        last_spot_bid: 0.0,
        last_spot_ask: 0.0,
        last_perp_bid: 0.0,
        last_perp_ask: 0.0,
        last_funding_rate: 0.0,
        funding_settlements: 0,
        basis_bps: 0.0,
        out_dir: Some(out_dir.to_path_buf()),
        depth_captured: include_depth,
        trades_captured: include_trades,
    };

    // Spawn parallel capture tasks with unified return type
    let mut handles: Vec<tokio::task::JoinHandle<Result<CaptureResult>>> = Vec::new();

    // 1. Spot bookTicker (optional)
    if include_spot {
        let sym = symbol.to_string();
        let path = out_dir.join("spot_quotes.jsonl");
        handles.push(tokio::spawn(async move {
            let result =
                binance_capture::capture_book_ticker_jsonl(&sym, &path, duration_secs).await?;
            Ok(CaptureResult::Spot {
                events: result.stats.events_written,
            })
        }));
    }

    // 2. Perp bookTicker (or depth)
    let sym = symbol.to_string();
    let path = if include_depth {
        out_dir.join("perp_depth.jsonl")
    } else {
        out_dir.join("perp_quotes.jsonl")
    };
    let dur = duration_secs;
    if include_depth {
        handles.push(tokio::spawn(async move {
            let result = binance_perp_capture::capture_perp_depth_jsonl(
                &sym,
                &path,
                dur,
                price_exponent,
                qty_exponent,
            )
            .await?;
            Ok(CaptureResult::Perp {
                events: result.stats.events_written,
                bid: result.stats.last_bid,
                ask: result.stats.last_ask,
            })
        }));
    } else {
        handles.push(tokio::spawn(async move {
            let result =
                binance_perp_capture::capture_perp_bookticker_jsonl(&sym, &path, dur).await?;
            Ok(CaptureResult::Perp {
                events: result.stats.events_written,
                bid: result.stats.last_bid,
                ask: result.stats.last_ask,
            })
        }));
    }

    // 3. Funding rate stream
    let sym = symbol.to_string();
    let path = out_dir.join("funding.jsonl");
    let dur = duration_secs;
    handles.push(tokio::spawn(async move {
        let result = binance_funding_capture::capture_funding_jsonl(&sym, &path, dur).await?;
        Ok(CaptureResult::Funding {
            events: result.stats.events_written,
            rate: result.stats.last_funding_rate_f64(),
            settlements: result.stats.funding_settlements,
        })
    }));

    // 4. AggTrades stream (optional)
    // Strategy: Try SBE first (if API key provided), fall back to public WS
    if include_trades {
        let sym = symbol.to_string();
        let path = out_dir.join("agg_trades.jsonl");
        let dur = duration_secs;
        let key = api_key.map(|s| s.to_string());
        handles.push(tokio::spawn(async move {
            // Try SBE first if API key is available
            if let Some(api_key) = &key {
                tracing::info!("{}: Attempting SBE trades capture...", sym);
                match crate::binance_trades_capture::capture_sbe_trades_jsonl(
                    &sym,
                    &path,
                    dur,
                    price_exponent,
                    qty_exponent,
                    api_key,
                )
                .await
                {
                    Ok(result) => {
                        tracing::info!(
                            "{}: SBE trades capture succeeded ({} trades)",
                            sym,
                            result.trades_written
                        );
                        return Ok(CaptureResult::Trades {
                            events: result.trades_written,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            "{}: SBE trades capture failed: {}. Falling back to public WS...",
                            sym,
                            e
                        );
                    }
                }
            } else {
                tracing::info!("{}: No API key, using public aggTrades stream", sym);
            }

            // Fallback: Use public aggTrades stream (no API key required)
            let result = crate::binance_trades_capture::capture_public_aggtrades_jsonl(
                &sym,
                &path,
                dur,
                price_exponent,
                qty_exponent,
            )
            .await?;
            Ok(CaptureResult::Trades {
                events: result.trades_written,
            })
        }));
    }

    // Collect results
    for handle in handles {
        match handle.await {
            Ok(Ok(CaptureResult::Spot { events })) => {
                stats.spot_events = events;
            }
            Ok(Ok(CaptureResult::Perp { events, bid, ask })) => {
                stats.perp_events = events;
                stats.last_perp_bid = bid;
                stats.last_perp_ask = ask;
            }
            Ok(Ok(CaptureResult::Funding {
                events,
                rate,
                settlements,
            })) => {
                stats.funding_events = events;
                stats.last_funding_rate = rate;
                stats.funding_settlements = settlements;
            }
            Ok(Ok(CaptureResult::Trades { events })) => {
                stats.trade_events = events;
                tracing::info!("{}: captured {} trades", symbol, events);
            }
            Ok(Err(e)) => {
                tracing::warn!("Capture for {} failed: {}", symbol, e);
            }
            Err(e) => {
                tracing::warn!("Task panicked for {}: {}", symbol, e);
            }
        }
    }

    // Calculate final basis
    stats.calculate_basis_bps();

    Ok(stats)
}

/// Load a perp session manifest from disk.
///
/// # Schema Validation (Hard Fail)
/// This function enforces that the manifest contains `quote_schema: "canonical_v1"`.
/// Legacy captures (pre-2026-01-24) used different quote field names and are fundamentally
/// incompatible with the current pipeline. They MUST be recaptured.
///
/// If you encounter a schema validation error:
/// 1. Move the legacy capture to `data/legacy_pre_canonical_2026_01_24/`
/// 2. Recapture with the current tooling
pub fn load_perp_session_manifest(session_dir: &Path) -> Result<PerpSessionManifest> {
    let manifest_path = session_dir.join("session_manifest.json");
    let content = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read manifest: {:?}", manifest_path))?;
    let manifest: PerpSessionManifest = serde_json::from_str(&content)
        .with_context(|| format!("parse manifest: {:?}", manifest_path))?;

    // HARD FAIL: Reject unsupported schema versions
    // Accept both v3 (depth-only) and v4 (depth + trades) for backward compatibility
    let supported_versions = [3, 4];
    if !supported_versions.contains(&manifest.schema_version) {
        anyhow::bail!(
            "FATAL: Unsupported manifest schema_version {} in {:?}. \
             Expected one of {:?}. This capture may need to be migrated or recaptured.",
            manifest.schema_version,
            manifest_path,
            supported_versions
        );
    }

    // HARD FAIL: Reject captures that don't have canonical quote schema
    if manifest.quote_schema != CANONICAL_QUOTE_SCHEMA {
        anyhow::bail!(
            "FATAL: Invalid quote schema '{}' in {:?}. \
             Expected '{}'. This capture uses an incompatible schema and must be recaptured.",
            manifest.quote_schema,
            manifest_path,
            CANONICAL_QUOTE_SCHEMA
        );
    }

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_calculation() {
        let mut stats = SymbolCaptureStats {
            symbol: "BTCUSDT".to_string(),
            spot_events: 100,
            perp_events: 100,
            depth_events: 0,
            funding_events: 10,
            trade_events: 0,
            last_spot_bid: 99990.0,
            last_spot_ask: 100010.0,
            last_perp_bid: 100040.0,
            last_perp_ask: 100060.0,
            last_funding_rate: 0.0001,
            funding_settlements: 0,
            basis_bps: 0.0,
            out_dir: None,
            depth_captured: false,
            trades_captured: false,
        };

        stats.calculate_basis_bps();

        // Spot mid = 100000, Perp mid = 100050
        // Basis = (100050 - 100000) / 100000 = 0.0005 = 5 bps
        assert!((stats.basis_bps - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_stream_digest_sha256_and_timestamps() {
        use std::io::Write;

        // Create a temp JSONL file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_digest.jsonl");

        let mut file = std::fs::File::create(&test_file).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00Z","price":100}}"#).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:01Z","price":101}}"#).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:02Z","price":102}}"#).unwrap();
        drop(file);

        // Compute digest
        let digest = StreamDigest::from_jsonl_file(&test_file, "test/test.jsonl").unwrap();

        assert_eq!(digest.event_count, 3);
        assert_eq!(digest.file_path, "test/test.jsonl");
        assert_eq!(
            digest.first_event_ts.as_deref(),
            Some("2026-01-24T10:00:00Z")
        );
        assert_eq!(
            digest.last_event_ts.as_deref(),
            Some("2026-01-24T10:00:02Z")
        );
        assert_eq!(digest.sha256.len(), 64); // SHA256 hex is 64 chars
        assert!(digest.size_bytes > 0);

        // Verify SHA256 is deterministic
        let digest2 = StreamDigest::from_jsonl_file(&test_file, "test/test.jsonl").unwrap();
        assert_eq!(digest.sha256, digest2.sha256);

        // Cleanup
        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_trades_monotonic_validation_passes() {
        use std::io::Write;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_trades_monotonic.jsonl");

        let mut file = std::fs::File::create(&test_file).unwrap();
        // Write trades with monotonically increasing timestamps
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.000Z","price":100}}"#).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.001Z","price":101}}"#).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.002Z","price":102}}"#).unwrap();
        drop(file);

        let result = validate_trades_monotonic(&test_file);

        assert_eq!(result.trade_count, 3);
        assert!(result.is_monotonic);
        assert_eq!(result.violations, 0);
        assert!(result.first_violation.is_none());

        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_trades_monotonic_validation_fails_on_out_of_order() {
        use std::io::Write;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_trades_non_monotonic.jsonl");

        let mut file = std::fs::File::create(&test_file).unwrap();
        // Write trades with out-of-order timestamp (second is earlier than first)
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.100Z","price":100}}"#).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.050Z","price":101}}"#).unwrap(); // Out of order!
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.200Z","price":102}}"#).unwrap();
        drop(file);

        let result = validate_trades_monotonic(&test_file);

        assert_eq!(result.trade_count, 3);
        assert!(!result.is_monotonic);
        assert_eq!(result.violations, 1);
        assert!(result.first_violation.is_some());
        // The violation should indicate line 2
        assert!(result.first_violation.unwrap().contains("Line 2"));

        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_trades_monotonic_validation_allows_equal_timestamps() {
        use std::io::Write;

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_trades_equal_ts.jsonl");

        let mut file = std::fs::File::create(&test_file).unwrap();
        // Write trades with equal timestamps (allowed - non-decreasing)
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.100Z","price":100}}"#).unwrap();
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.100Z","price":101}}"#).unwrap(); // Equal is OK
        writeln!(file, r#"{{"ts":"2026-01-24T10:00:00.200Z","price":102}}"#).unwrap();
        drop(file);

        let result = validate_trades_monotonic(&test_file);

        assert_eq!(result.trade_count, 3);
        assert!(result.is_monotonic);
        assert_eq!(result.violations, 0);

        std::fs::remove_file(&test_file).ok();
    }
}
