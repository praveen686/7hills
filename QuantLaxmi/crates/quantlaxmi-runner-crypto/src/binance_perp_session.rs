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
#[derive(Debug, Clone)]
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
    /// Price exponent for depth capture
    pub price_exponent: i8,
    /// Quantity exponent for depth capture
    pub qty_exponent: i8,
}

impl Default for PerpSessionConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_string()],
            out_dir: PathBuf::from("data/perp_sessions"),
            duration_secs: 3600, // 1 hour default
            include_spot: true,
            include_depth: false, // bookTicker is sufficient for MVP
            price_exponent: -2,
            qty_exponent: -8,
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

/// Session manifest for perp capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionManifest {
    pub schema_version: u32,
    /// Quote schema identifier. Must be "canonical_v1" for valid captures.
    /// Legacy captures (pre-2026-01-24) used different field names and are invalid.
    #[serde(default)]
    pub quote_schema: Option<String>,
    pub created_at_utc: String,
    pub session_id: String,
    pub capture_mode: String,
    pub duration_secs: f64,
    pub symbols: Vec<SymbolManifestEntry>,
    pub config: PerpSessionManifestConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolManifestEntry {
    pub symbol: String,
    pub spot_file: Option<String>,
    pub perp_file: String,
    pub depth_file: Option<String>,
    pub funding_file: String,
    pub spot_events: usize,
    pub perp_events: usize,
    pub depth_events: usize,
    pub funding_events: usize,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpSessionManifestConfig {
    pub include_spot: bool,
    pub include_depth: bool,
    pub price_exponent: i8,
    pub qty_exponent: i8,
}

/// Compute digests for all captured files for a symbol.
/// Returns None if any required file is missing or unreadable.
fn compute_symbol_digests(
    sym_dir: &Path,
    sym_upper: &str,
    include_spot: bool,
    include_depth: bool,
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

    Some(SymbolDigests {
        spot: spot_digest,
        perp: perp_digest,
        depth: depth_digest,
        funding: funding_digest,
    })
}

/// Capture a perp session with Spot + Perp + Funding data.
pub async fn capture_perp_session(config: PerpSessionConfig) -> Result<PerpSessionStats> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let start_time = Utc::now();

    // Create session directory
    let session_tag = format!("perp_{}", start_time.format("%Y%m%d_%H%M%S"));
    let session_dir = config.out_dir.join(&session_tag);
    std::fs::create_dir_all(&session_dir)
        .with_context(|| format!("create session dir: {:?}", session_dir))?;

    tracing::info!("=== Perp Session Capture ===");
    tracing::info!("Session ID: {}", &session_id[..8]);
    tracing::info!("Symbols: {:?}", config.symbols);
    tracing::info!("Duration: {}s", config.duration_secs);
    tracing::info!("Output: {:?}", session_dir);

    // Spawn capture tasks for each symbol
    let mut handles = Vec::new();
    for symbol in &config.symbols {
        let sym = symbol.clone();
        let sym_dir = session_dir.join(symbol.to_uppercase());
        std::fs::create_dir_all(&sym_dir)?;

        let duration = config.duration_secs;
        let include_spot = config.include_spot;
        let include_depth = config.include_depth;
        let price_exp = config.price_exponent;
        let qty_exp = config.qty_exponent;

        let handle = tokio::spawn(async move {
            capture_symbol(
                &sym,
                &sym_dir,
                duration,
                include_spot,
                include_depth,
                price_exp,
                qty_exp,
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
    let mut all_complete = true;

    for (symbol, handle) in handles {
        match handle.await {
            Ok(Ok(stats)) => {
                total_spot += stats.spot_events;
                total_perp += stats.perp_events;
                total_funding += stats.funding_events;
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
        let digests =
            compute_symbol_digests(&sym_dir, &sym_upper, config.include_spot, s.depth_captured);

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
            spot_events: s.spot_events,
            perp_events: s.perp_events,
            depth_events: s.depth_events,
            funding_events: s.funding_events,
            digests,
        };
        manifest_entries.push(entry);
    }

    // Write session manifest
    let manifest = PerpSessionManifest {
        schema_version: 3, // Bump for canonical quote schema
        quote_schema: Some(CANONICAL_QUOTE_SCHEMA.to_string()),
        created_at_utc: start_time.to_rfc3339(),
        session_id: session_id.clone(),
        capture_mode: "perp_session".to_string(),
        duration_secs,
        symbols: manifest_entries,
        config: PerpSessionManifestConfig {
            include_spot: config.include_spot,
            include_depth: config.include_depth,
            price_exponent: config.price_exponent,
            qty_exponent: config.qty_exponent,
        },
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
}

/// Capture all streams for a single symbol.
async fn capture_symbol(
    symbol: &str,
    out_dir: &Path,
    duration_secs: u64,
    include_spot: bool,
    include_depth: bool,
    price_exponent: i8,
    qty_exponent: i8,
) -> Result<SymbolCaptureStats> {
    let mut stats = SymbolCaptureStats {
        symbol: symbol.to_string(),
        spot_events: 0,
        perp_events: 0,
        depth_events: 0,
        funding_events: 0,
        last_spot_bid: 0.0,
        last_spot_ask: 0.0,
        last_perp_bid: 0.0,
        last_perp_ask: 0.0,
        last_funding_rate: 0.0,
        funding_settlements: 0,
        basis_bps: 0.0,
        out_dir: Some(out_dir.to_path_buf()),
        depth_captured: include_depth,
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
                events: result.events_written,
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
                events: result.events_written,
                bid: result.last_bid,
                ask: result.last_ask,
            })
        }));
    } else {
        handles.push(tokio::spawn(async move {
            let result =
                binance_perp_capture::capture_perp_bookticker_jsonl(&sym, &path, dur).await?;
            Ok(CaptureResult::Perp {
                events: result.events_written,
                bid: result.last_bid,
                ask: result.last_ask,
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
            events: result.events_written,
            rate: result.last_funding_rate_f64(),
            settlements: result.funding_settlements,
        })
    }));

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

    // HARD FAIL: Reject legacy captures that don't have canonical quote schema
    match &manifest.quote_schema {
        Some(schema) if schema == CANONICAL_QUOTE_SCHEMA => {
            // Valid canonical capture
        }
        Some(schema) => {
            anyhow::bail!(
                "FATAL: Invalid quote schema '{}' in {:?}. \
                 Expected '{}'. This capture uses an incompatible schema and must be recaptured.",
                schema,
                manifest_path,
                CANONICAL_QUOTE_SCHEMA
            );
        }
        None => {
            anyhow::bail!(
                "FATAL: Legacy capture detected — missing 'quote_schema' field in {:?}. \
                 This capture predates the canonical schema (2026-01-24) and is incompatible. \
                 Move to data/legacy_pre_canonical_2026_01_24/ and recapture with current tooling.",
                manifest_path
            );
        }
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
            last_spot_bid: 99990.0,
            last_spot_ask: 100010.0,
            last_perp_bid: 100040.0,
            last_perp_ask: 100060.0,
            last_funding_rate: 0.0001,
            funding_settlements: 0,
            basis_bps: 0.0,
            out_dir: None,
            depth_captured: false,
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
}
