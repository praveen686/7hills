//! Segment Manifest Management
//!
//! Provides segment-aware capture with:
//! - Manifest written at segment start (with "running" status)
//! - Heartbeat updates every 60 seconds
//! - Finalization on graceful shutdown or signal
//! - Automatic inventory maintenance for session families
//!
//! ## Segment Lifecycle
//! ```text
//! 1. Start capture → write segment_manifest.json (status: "running")
//! 2. Every 60s → update heartbeat_ts in manifest
//! 3. On signal/completion → finalize manifest with end_ts and stop_reason
//! 4. Auto-append to family inventory
//! ```

use crate::backtest::{PNL_EXPONENT, PnlAccumulatorFixed};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

/// Reason the segment stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StopReason {
    /// Segment is still running
    Running,
    /// Normal completion (duration elapsed)
    NormalCompletion,
    /// User interrupt (Ctrl+C / SIGINT)
    UserInterrupt,
    /// External kill (SIGTERM)
    ExternalKillSigterm,
    /// Terminal disconnect (SIGHUP)
    ExternalKillSighup,
    /// Panic or crash
    Panic,
    /// Network error
    NetworkError,
    /// Unknown (for recovery from hard stops)
    Unknown,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Running => write!(f, "RUNNING"),
            StopReason::NormalCompletion => write!(f, "NORMAL_COMPLETION"),
            StopReason::UserInterrupt => write!(f, "USER_INTERRUPT"),
            StopReason::ExternalKillSigterm => write!(f, "EXTERNAL_KILL_SIGTERM"),
            StopReason::ExternalKillSighup => write!(f, "EXTERNAL_KILL_SIGHUP"),
            StopReason::Panic => write!(f, "PANIC"),
            StopReason::NetworkError => write!(f, "NETWORK_ERROR"),
            StopReason::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Event counts per stream type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCounts {
    pub spot_quotes: usize,
    pub perp_quotes: usize,
    pub funding: usize,
    pub depth: usize,
}

impl EventCounts {
    pub fn total(&self) -> usize {
        self.spot_quotes + self.perp_quotes + self.funding + self.depth
    }
}

/// Gap detected between segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapInfo {
    pub previous_segment_id: String,
    pub gap_seconds: f64,
    pub reason: String,
}

/// Segment lifecycle state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SegmentState {
    /// Manifest written at start, capture in progress
    #[default]
    Bootstrap,
    /// Graceful shutdown with digests computed
    Finalized,
    /// Retroactively finalized from crashed segment
    FinalizedRetro,
}

/// Per-stream digest for integrity verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDigest {
    pub file_path: String,
    pub sha256: String,
    pub event_count: usize,
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_event_ts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_ts: Option<String>,
}

/// All stream digests for a segment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SegmentDigests {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<StreamDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perp: Option<StreamDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub funding: Option<StreamDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<StreamDigest>,
}

/// Capture configuration snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub include_spot: bool,
    pub include_depth: bool,
    pub price_exponent: i32,
    pub qty_exponent: i32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            include_spot: true,
            include_depth: false,
            price_exponent: -2,
            qty_exponent: -8,
        }
    }
}

/// Decision trace binding for replay parity verification.
///
/// Binds the decision trace artifact to the manifest, enabling:
/// - Discovery of trace file location
/// - Integrity verification via SHA-256
/// - Encoding version compatibility checking
/// - Decision count audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceBinding {
    /// Relative path to decision trace JSON file (from segment directory)
    pub decision_trace_path: String,
    /// SHA-256 hash of the trace file contents
    pub decision_trace_sha256: String,
    /// Encoding version used for canonical bytes (for compatibility checking)
    pub decision_trace_encoding_version: u8,
    /// Total number of decisions in the trace
    pub total_decisions: usize,
    /// Fixed-point realized PnL (mantissa with pnl_exponent)
    pub realized_pnl_mantissa: i128,
    /// Fixed-point total fees (mantissa with pnl_exponent)
    pub total_fees_mantissa: i128,
    /// PnL exponent (typically price_exponent + qty_exponent, normalized to -8 for crypto)
    pub pnl_exponent: i8,
}

/// Strategy identity binding for manifest.
///
/// Records which strategy (and with what config) produced the trace.
/// Enables reproducibility verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyBinding {
    /// Strategy name (e.g., "funding_bias")
    pub strategy_name: String,
    /// Strategy version (e.g., "2.0.0")
    pub strategy_version: String,
    /// Full SHA-256 hash of canonical_bytes() (NOT JSON)
    pub config_hash: String,
    /// Full strategy_id: "{name}:{version}:{config_hash}"
    pub strategy_id: String,
    /// Short ID for display (first 8 chars of config_hash)
    pub short_id: String,
    /// Original config file path (if loaded from file)
    pub config_path: Option<String>,
    /// JSON snapshot for human inspection ONLY (not used in hashing)
    pub config_snapshot: Option<serde_json::Value>,
}

/// Attribution artifact binding for manifest (Phase 3).
///
/// Records the trade attribution artifact location and integrity.
/// Enables PnL verification and audit trails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionBinding {
    /// Relative path to attribution JSONL file (from segment directory)
    pub attribution_path: String,
    /// SHA-256 hash of the attribution file contents
    pub attribution_sha256: String,
    /// Number of attribution events in the file
    pub num_attribution_events: usize,

    // === Summary statistics (for manifest self-description) ===
    /// Total net PnL across all attribution events (mantissa)
    pub total_net_pnl_mantissa: i128,
    /// Total fees across all attribution events (mantissa)
    pub total_fees_mantissa: i128,
    /// PnL exponent (typically -8 for crypto)
    pub pnl_exponent: i8,
}

/// Attribution summary binding for manifest (Phase 4).
///
/// Records the aggregated attribution summary artifact for strategy evaluation.
/// Enables alpha scoring and G1 promotion gating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionSummaryBinding {
    /// Relative path to attribution_summary.json (from segment directory)
    pub summary_path: String,
    /// SHA-256 hash of the summary file contents
    pub summary_sha256: String,
    /// Strategy ID this summary is for
    pub strategy_id: String,

    // === Summary statistics (for manifest self-description) ===
    /// Total decisions aggregated
    pub total_decisions: u32,
    /// Total fills executed
    pub total_fills: u32,
    /// Win rate in basis points (10000 = 100%)
    pub win_rate_bps: u32,
    /// Total net PnL (mantissa)
    pub total_net_pnl_mantissa: i128,
    /// Max loss (mantissa, positive value)
    pub max_loss_mantissa: i128,
    /// PnL exponent (typically -8)
    pub pnl_exponent: i8,

    // === Alpha Score ===
    /// Alpha score mantissa (computed using AlphaScoreV1 formula)
    pub alpha_score_mantissa: i128,
    /// Alpha score exponent
    pub alpha_score_exponent: i8,
    /// Alpha score formula version
    pub alpha_score_formula: String,
}

/// Router binding for manifest (Phase 5).
///
/// Records the router configuration and decisions artifact for replay parity.
/// Enables reproducible regime classification and strategy routing.
///
/// # Phase 6 Correlation Note
/// RouterDecisionEvent.decision_id should be linked to downstream DecisionEvent
/// via a `router_decision_id` field or shared CorrelationContext. This enables
/// full traceability: Router → Decision → Intent → Fill → Attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterBinding {
    /// Relative path to router_decisions.jsonl (from segment directory)
    pub decisions_path: String,
    /// SHA-256 hash of the decisions file contents
    pub decisions_sha256: String,
    /// Number of routing decisions in the file
    pub num_decisions: usize,

    // === Router Identity ===
    /// Router config hash (SHA-256 of canonical config bytes)
    pub router_config_hash: String,
    /// Router version
    pub router_version: String,

    // === Summary Statistics ===
    /// Distribution of regimes classified (regime label -> count).
    /// Uses BTreeMap for deterministic serialization order (replay parity).
    pub regime_distribution: std::collections::BTreeMap<String, usize>,
    /// Number of unique strategies selected
    pub unique_strategies_used: usize,
}

/// G2 robustness report binding for manifest (Phase 6).
///
/// Records the G2 robustness test results artifact.
/// Enables verification that strategy passed anti-overfit checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2Binding {
    /// Relative path to g2_report.json (from segment directory)
    pub report_path: String,
    /// SHA-256 hash of the report file contents
    pub report_sha256: String,
    /// Whether G2 gate passed
    pub passed: bool,
    /// G2 gate version used
    pub version: String,
    /// Base alpha score (mantissa) for reference
    pub base_score_mantissa: i128,
    /// Number of time-shift tests run
    pub num_shift_tests: u32,
    /// Number of cost sensitivity tests run
    pub num_cost_tests: u32,
}

/// G3 walk-forward report binding for manifest (Phase 6).
///
/// Records the G3 walk-forward stability test results artifact.
/// Enables verification that strategy is stable across time periods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3Binding {
    /// Relative path to g3_walkforward.json (from segment directory)
    pub report_path: String,
    /// SHA-256 hash of the report file contents
    pub report_sha256: String,
    /// Whether G3 gate passed
    pub passed: bool,
    /// G3 gate version used
    pub version: String,
    /// Number of folds used
    pub num_folds: u32,
    /// Median alpha score across folds (mantissa)
    pub median_score_mantissa: i128,
    /// Consistency ratio (basis points)
    pub consistency_ratio_bps: u32,
}

/// Current segment manifest schema version.
/// Bump this when manifest structure changes.
/// Schema version 4: Added trace_binding for decision trace artifact binding.
/// Schema version 5: Added strategy_binding for strategy identity (Phase 2).
/// Schema version 6: Added attribution_binding for trade attribution (Phase 3).
/// Schema version 7: Added attribution_summary_binding for strategy evaluation (Phase 4).
/// Schema version 8: Added router_binding for regime routing (Phase 5).
/// Schema version 9: Added g2_binding and g3_binding for anti-overfit validation (Phase 6).
pub const SEGMENT_MANIFEST_SCHEMA_VERSION: u32 = 9;

/// Segment manifest - written to segment_manifest.json in each segment directory.
///
/// ## Lifecycle
/// 1. BOOTSTRAP: Written immediately at segment start with schema assertions
/// 2. FINALIZED: Updated on graceful shutdown with digests and counts
/// 3. FINALIZED_RETRO: Created by retro-finalize for crashed segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManifest {
    /// Schema version for this manifest format (must match SEGMENT_MANIFEST_SCHEMA_VERSION)
    pub schema_version: u32,
    /// Quote schema identifier (must be "canonical_v1" for valid captures)
    pub quote_schema: String,
    /// Segment lifecycle state
    pub state: SegmentState,
    /// Family ID (e.g., "perp_BTCUSDT_20260125")
    pub session_family_id: String,
    /// Segment ID (typically the folder name, e.g., "perp_20260125_051437")
    pub segment_id: String,
    /// Symbol(s) being captured
    pub symbols: Vec<String>,
    /// Capture mode (e.g., "capture-perp-session")
    pub capture_mode: String,
    /// Start timestamp
    pub start_ts: DateTime<Utc>,
    /// End timestamp (None if still running)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ts: Option<DateTime<Utc>>,
    /// Last heartbeat timestamp (updated every 60s while running)
    pub heartbeat_ts: DateTime<Utc>,
    /// Stop reason
    pub stop_reason: StopReason,
    /// Event counts
    pub events: EventCounts,
    /// Binary hash (SHA256 of the capture binary)
    pub binary_hash: String,
    /// Capture configuration snapshot
    pub config: CaptureConfig,
    /// Per-stream digests (populated on finalization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digests: Option<SegmentDigests>,
    /// Gap info if this segment follows a prior segment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap_from_prior: Option<GapInfo>,
    /// Duration in seconds (None if still running)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
    /// Decision trace binding (populated after backtest/replay)
    pub trace_binding: Option<TraceBinding>,
    /// Strategy identity binding (populated after backtest with strategy SDK)
    pub strategy_binding: Option<StrategyBinding>,
    /// Attribution artifact binding (populated after backtest with attribution)
    pub attribution_binding: Option<AttributionBinding>,
    /// Attribution summary binding (populated after strategy evaluation, Phase 4)
    pub attribution_summary_binding: Option<AttributionSummaryBinding>,
    /// Router binding (populated after routing decisions, Phase 5)
    pub router_binding: Option<RouterBinding>,
    /// G2 robustness report binding (populated after anti-overfit validation, Phase 6)
    pub g2_binding: Option<G2Binding>,
    /// G3 walk-forward report binding (populated after stability validation, Phase 6)
    pub g3_binding: Option<G3Binding>,
}

impl SegmentManifest {
    /// Create a new bootstrap manifest at segment start.
    ///
    /// This is written immediately when capture begins, ensuring every segment
    /// has a manifest with schema assertions even if killed ungracefully.
    pub fn new(
        session_family_id: String,
        segment_id: String,
        symbols: Vec<String>,
        capture_mode: String,
        binary_hash: String,
        config: CaptureConfig,
    ) -> Self {
        let now = Utc::now();
        Self {
            schema_version: SEGMENT_MANIFEST_SCHEMA_VERSION,
            quote_schema: "canonical_v1".to_string(),
            state: SegmentState::Bootstrap,
            session_family_id,
            segment_id,
            symbols,
            capture_mode,
            start_ts: now,
            end_ts: None,
            heartbeat_ts: now,
            stop_reason: StopReason::Running,
            events: EventCounts::default(),
            binary_hash,
            config,
            digests: None,
            gap_from_prior: None,
            duration_secs: None,
            trace_binding: None,
            strategy_binding: None,
            attribution_binding: None,
            attribution_summary_binding: None,
            router_binding: None,
            g2_binding: None,
            g3_binding: None,
        }
    }

    /// Update heartbeat timestamp.
    pub fn heartbeat(&mut self) {
        self.heartbeat_ts = Utc::now();
    }

    /// Finalize the manifest with stop reason, counts, and digests.
    ///
    /// Called on graceful shutdown. Sets state to FINALIZED.
    pub fn finalize(
        &mut self,
        stop_reason: StopReason,
        events: EventCounts,
        digests: Option<SegmentDigests>,
    ) {
        let now = Utc::now();
        self.end_ts = Some(now);
        self.stop_reason = stop_reason;
        self.events = events;
        self.digests = digests;
        self.state = SegmentState::Finalized;
        self.duration_secs = Some((now - self.start_ts).num_milliseconds() as f64 / 1000.0);
    }

    /// Retroactively finalize a crashed segment.
    ///
    /// Called by the retro-finalize command. Sets state to FINALIZED_RETRO.
    /// Infers start/end timestamps and duration from actual data in digests.
    pub fn retro_finalize(&mut self, events: EventCounts, digests: SegmentDigests) {
        // Infer actual data time range from digests
        let mut first_ts: Option<DateTime<Utc>> = None;
        let mut last_ts: Option<DateTime<Utc>> = None;

        // Check all digest sources for earliest/latest timestamps
        for d in [
            &digests.spot,
            &digests.perp,
            &digests.funding,
            &digests.depth,
        ]
        .into_iter()
        .flatten()
        {
            if let (Some(first_str), Some(last_str)) = (&d.first_event_ts, &d.last_event_ts) {
                // Parse timestamp strings
                if let (Ok(first), Ok(last)) = (
                    DateTime::parse_from_rfc3339(first_str).map(|t| t.with_timezone(&Utc)),
                    DateTime::parse_from_rfc3339(last_str).map(|t| t.with_timezone(&Utc)),
                ) {
                    first_ts = Some(first_ts.map_or(first, |t| t.min(first)));
                    last_ts = Some(last_ts.map_or(last, |t| t.max(last)));
                }
            }
        }

        // Update timestamps from actual data range
        if let Some(first) = first_ts {
            self.start_ts = first;
        }
        if let Some(last) = last_ts {
            self.end_ts = Some(last);
        } else if self.end_ts.is_none() {
            self.end_ts = Some(self.heartbeat_ts);
        }

        if self.stop_reason == StopReason::Running {
            self.stop_reason = StopReason::Unknown;
        }
        self.events = events;
        self.digests = Some(digests);
        self.state = SegmentState::FinalizedRetro;

        // Compute duration from actual data time range
        if let Some(end) = self.end_ts {
            self.duration_secs = Some((end - self.start_ts).num_milliseconds() as f64 / 1000.0);
        }
    }

    /// Check if this manifest is finalized (either gracefully or retro).
    pub fn is_finalized(&self) -> bool {
        matches!(
            self.state,
            SegmentState::Finalized | SegmentState::FinalizedRetro
        )
    }

    /// Bind a decision trace artifact to this manifest.
    ///
    /// This should be called after a backtest/replay produces a trace artifact.
    /// The trace file must exist at the specified path.
    ///
    /// # Arguments
    /// * `trace_path` - Path to the decision trace JSON file
    /// * `segment_dir` - Base segment directory (for computing relative path)
    /// * `encoding_version` - Encoding version used in the trace
    /// * `total_decisions` - Number of decisions in the trace
    /// * `pnl` - Fixed-point PnL accumulator with realized PnL and fees
    pub fn bind_trace(
        &mut self,
        trace_path: &Path,
        segment_dir: &Path,
        encoding_version: u8,
        total_decisions: usize,
        pnl: &PnlAccumulatorFixed,
    ) -> Result<()> {
        // Compute relative path from segment directory
        let relative_path = trace_path
            .strip_prefix(segment_dir)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| trace_path.to_string_lossy().to_string());

        // Compute SHA-256 of trace file
        let trace_sha256 = compute_file_sha256(trace_path)?;

        self.trace_binding = Some(TraceBinding {
            decision_trace_path: relative_path,
            decision_trace_sha256: trace_sha256,
            decision_trace_encoding_version: encoding_version,
            total_decisions,
            realized_pnl_mantissa: pnl.realized_pnl_mantissa,
            total_fees_mantissa: pnl.total_fees_mantissa,
            pnl_exponent: PNL_EXPONENT,
        });

        Ok(())
    }

    /// Bind a decision trace from BacktestResult.
    ///
    /// Convenience method that extracts trace info from a backtest result.
    /// The trace file must have been saved (i.e., trace_path must be Some).
    ///
    /// # Arguments
    /// * `trace_path` - Path to the decision trace JSON file
    /// * `segment_dir` - Base segment directory (for computing relative path)
    /// * `trace_hash` - Content hash from the trace (for debug logging)
    /// * `encoding_version` - Encoding version used in the trace
    /// * `total_decisions` - Number of decisions in the trace
    /// * `pnl` - Fixed-point PnL accumulator with realized PnL and fees
    pub fn bind_trace_from_result(
        &mut self,
        trace_path: &Path,
        segment_dir: &Path,
        trace_hash: &str,
        encoding_version: u8,
        total_decisions: usize,
        pnl: &PnlAccumulatorFixed,
    ) -> Result<()> {
        // Compute relative path from segment directory
        let relative_path = trace_path
            .strip_prefix(segment_dir)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| trace_path.to_string_lossy().to_string());

        // Verify file exists and compute SHA-256
        let file_sha256 = compute_file_sha256(trace_path)?;

        // Log for debugging (content hash vs file hash may differ due to formatting)
        tracing::debug!(
            "Binding trace: content_hash={}, file_sha256={}",
            trace_hash,
            file_sha256
        );

        self.trace_binding = Some(TraceBinding {
            decision_trace_path: relative_path,
            decision_trace_sha256: file_sha256,
            decision_trace_encoding_version: encoding_version,
            total_decisions,
            realized_pnl_mantissa: pnl.realized_pnl_mantissa,
            total_fees_mantissa: pnl.total_fees_mantissa,
            pnl_exponent: PNL_EXPONENT,
        });

        Ok(())
    }

    /// Bind strategy identity to manifest.
    ///
    /// Records the strategy that produced the decision trace for reproducibility.
    ///
    /// # Arguments
    /// * `name` - Strategy name
    /// * `version` - Strategy version
    /// * `config_hash` - SHA-256 of canonical config bytes (NOT JSON)
    /// * `config_path` - Original config file path (if loaded from file)
    /// * `config_snapshot` - JSON snapshot for human inspection (optional)
    pub fn bind_strategy(
        &mut self,
        name: &str,
        version: &str,
        config_hash: &str,
        config_path: Option<&str>,
        config_snapshot: Option<serde_json::Value>,
    ) {
        let strategy_id = format!("{}:{}:{}", name, version, config_hash);
        let short_hash = &config_hash[..8.min(config_hash.len())];
        let short_id = format!("{}:{}:{}", name, version, short_hash);

        self.strategy_binding = Some(StrategyBinding {
            strategy_name: name.to_string(),
            strategy_version: version.to_string(),
            config_hash: config_hash.to_string(),
            strategy_id,
            short_id,
            config_path: config_path.map(|s| s.to_string()),
            config_snapshot,
        });
    }

    /// Bind an attribution artifact to this manifest (Phase 3).
    ///
    /// This should be called after a backtest/replay produces an attribution artifact.
    /// The attribution file must exist at the specified path.
    ///
    /// # Arguments
    /// * `attribution_path` - Path to the attribution JSONL file
    /// * `segment_dir` - Base segment directory (for computing relative path)
    /// * `num_events` - Number of attribution events
    /// * `total_net_pnl_mantissa` - Sum of net PnL across all events
    /// * `total_fees_mantissa` - Sum of fees across all events
    /// * `pnl_exponent` - PnL exponent (typically -8)
    pub fn bind_attribution(
        &mut self,
        attribution_path: &Path,
        segment_dir: &Path,
        num_events: usize,
        total_net_pnl_mantissa: i128,
        total_fees_mantissa: i128,
        pnl_exponent: i8,
    ) -> Result<()> {
        // Compute relative path from segment directory
        let relative_path = attribution_path
            .strip_prefix(segment_dir)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| attribution_path.to_string_lossy().to_string());

        // Compute SHA-256 of attribution file
        let attribution_sha256 = compute_file_sha256(attribution_path)?;

        self.attribution_binding = Some(AttributionBinding {
            attribution_path: relative_path,
            attribution_sha256,
            num_attribution_events: num_events,
            total_net_pnl_mantissa,
            total_fees_mantissa,
            pnl_exponent,
        });

        Ok(())
    }

    /// Bind an attribution summary artifact to this manifest (Phase 4).
    ///
    /// This should be called after aggregating attribution events into a summary.
    /// Writes the summary JSON file and binds it with SHA-256 hash.
    ///
    /// # Arguments
    /// * `summary` - The attribution summary to bind
    /// * `alpha_score` - The computed alpha score for this strategy
    /// * `segment_dir` - Base segment directory (for writing file and computing relative path)
    pub fn bind_attribution_summary(
        &mut self,
        summary: &quantlaxmi_models::AttributionSummary,
        alpha_score: &quantlaxmi_models::AlphaScoreV1,
        segment_dir: &Path,
    ) -> Result<()> {
        // Write summary to JSON file
        let summary_path = segment_dir.join("attribution_summary.json");
        let json = serde_json::to_string_pretty(summary)?;
        std::fs::write(&summary_path, &json)
            .with_context(|| format!("write attribution summary: {:?}", summary_path))?;

        // Compute SHA-256 of summary file
        let summary_sha256 = compute_file_sha256(&summary_path)?;

        // Compute relative path from segment directory
        let relative_path = "attribution_summary.json".to_string();

        self.attribution_summary_binding = Some(AttributionSummaryBinding {
            summary_path: relative_path,
            summary_sha256,
            strategy_id: summary.strategy_id.clone(),
            total_decisions: summary.total_decisions,
            total_fills: summary.total_fills,
            win_rate_bps: summary.win_rate_bps,
            total_net_pnl_mantissa: summary.total_net_pnl_mantissa,
            max_loss_mantissa: summary.max_loss_mantissa,
            pnl_exponent: summary.pnl_exponent,
            alpha_score_mantissa: alpha_score.score_mantissa,
            alpha_score_exponent: alpha_score.score_exponent,
            alpha_score_formula: alpha_score.formula_version.to_string(),
        });

        Ok(())
    }

    /// Bind router decisions artifact to this manifest (Phase 5).
    ///
    /// This should be called after a backtest/replay produces router decisions.
    /// Writes the decisions JSONL file and binds it with SHA-256 hash.
    ///
    /// # Arguments
    /// * `decisions` - The list of routing decisions
    /// * `router_config_hash` - Hash of the router configuration
    /// * `router_version` - Router version string
    /// * `segment_dir` - Base segment directory (for writing file)
    pub fn bind_router_decisions(
        &mut self,
        decisions: &[quantlaxmi_models::RouterDecisionEvent],
        router_config_hash: &str,
        router_version: &str,
        segment_dir: &Path,
    ) -> Result<()> {
        use std::io::Write;

        // Write decisions to JSONL file
        let decisions_path = segment_dir.join("router_decisions.jsonl");
        let mut file = std::fs::File::create(&decisions_path)
            .with_context(|| format!("create router decisions file: {:?}", decisions_path))?;

        // Collect regime distribution and unique strategies
        // Using BTreeMap for deterministic serialization order (replay parity)
        let mut regime_distribution: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut unique_strategies: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for decision in decisions {
            // Write JSONL line
            let json = serde_json::to_string(decision)?;
            writeln!(file, "{}", json)?;

            // Track regime distribution
            *regime_distribution
                .entry(decision.regime.as_str().to_string())
                .or_insert(0) += 1;

            // Track unique strategies
            unique_strategies.insert(decision.selected_strategy_id.clone());
        }

        drop(file);

        // Compute SHA-256 of decisions file
        let decisions_sha256 = compute_file_sha256(&decisions_path)?;

        self.router_binding = Some(RouterBinding {
            decisions_path: "router_decisions.jsonl".to_string(),
            decisions_sha256,
            num_decisions: decisions.len(),
            router_config_hash: router_config_hash.to_string(),
            router_version: router_version.to_string(),
            regime_distribution,
            unique_strategies_used: unique_strategies.len(),
        });

        Ok(())
    }

    /// Bind a G2 robustness report to this manifest (Phase 6).
    ///
    /// This should be called after generating the G2 anti-overfit report.
    /// The report file must exist at the specified path.
    ///
    /// # Arguments
    /// * `report` - The G2Report to bind
    /// * `segment_dir` - Base segment directory (for writing file and computing relative path)
    pub fn bind_g2_report(
        &mut self,
        report: &quantlaxmi_models::G2Report,
        segment_dir: &Path,
    ) -> Result<()> {
        use std::io::Write;

        // Write report to JSON file
        let report_path = segment_dir.join("g2_report.json");
        let json = serde_json::to_string_pretty(report)?;
        let mut file = std::fs::File::create(&report_path)
            .with_context(|| format!("create G2 report file: {:?}", report_path))?;
        file.write_all(json.as_bytes())?;
        drop(file);

        // Compute SHA-256 of report file
        let report_sha256 = compute_file_sha256(&report_path)?;

        self.g2_binding = Some(G2Binding {
            report_path: "g2_report.json".to_string(),
            report_sha256,
            passed: report.passed,
            version: report.version.clone(),
            base_score_mantissa: report.base_score_mantissa,
            num_shift_tests: report.time_shift_results.len() as u32,
            num_cost_tests: report.cost_sensitivity_results.len() as u32,
        });

        Ok(())
    }

    /// Bind a G3 walk-forward report to this manifest (Phase 6).
    ///
    /// This should be called after generating the G3 stability report.
    /// The report file must exist at the specified path.
    ///
    /// # Arguments
    /// * `report` - The G3Report to bind
    /// * `segment_dir` - Base segment directory (for writing file and computing relative path)
    pub fn bind_g3_report(
        &mut self,
        report: &quantlaxmi_models::G3Report,
        segment_dir: &Path,
    ) -> Result<()> {
        use std::io::Write;

        // Write report to JSON file
        let report_path = segment_dir.join("g3_walkforward.json");
        let json = serde_json::to_string_pretty(report)?;
        let mut file = std::fs::File::create(&report_path)
            .with_context(|| format!("create G3 report file: {:?}", report_path))?;
        file.write_all(json.as_bytes())?;
        drop(file);

        // Compute SHA-256 of report file
        let report_sha256 = compute_file_sha256(&report_path)?;

        self.g3_binding = Some(G3Binding {
            report_path: "g3_walkforward.json".to_string(),
            report_sha256,
            passed: report.passed,
            version: report.version.clone(),
            num_folds: report.folds.len() as u32,
            median_score_mantissa: report.stability_metrics.median_score_mantissa,
            consistency_ratio_bps: report.stability_metrics.consistency_ratio_bps,
        });

        Ok(())
    }

    /// Write manifest to disk atomically (write temp + rename).
    ///
    /// This ensures crash safety - the manifest is either fully written or not at all.
    pub fn write(&self, segment_dir: &Path) -> Result<()> {
        let manifest_path = segment_dir.join("segment_manifest.json");
        let temp_path = segment_dir.join(".segment_manifest.json.tmp");
        let json = serde_json::to_string_pretty(self)?;

        // Write to temp file
        std::fs::write(&temp_path, &json)
            .with_context(|| format!("write temp manifest: {:?}", temp_path))?;

        // Atomic rename
        std::fs::rename(&temp_path, &manifest_path)
            .with_context(|| format!("rename manifest: {:?} -> {:?}", temp_path, manifest_path))?;

        Ok(())
    }

    /// Load manifest from disk.
    pub fn load(segment_dir: &Path) -> Result<Self> {
        let manifest_path = segment_dir.join("segment_manifest.json");
        let content = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read segment manifest: {:?}", manifest_path))?;
        serde_json::from_str(&content)
            .with_context(|| format!("parse segment manifest: {:?}", manifest_path))
    }
}

/// Inventory entry for a segment within a session family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventorySegment {
    pub segment_id: String,
    pub path: String,
    pub start_ts: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
    pub stop_reason: StopReason,
    pub events: EventCounts,
    pub usable: bool,
}

/// Gap analysis between segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryGap {
    pub from_segment: String,
    pub to_segment: String,
    pub from_ts: String,
    pub to_ts: String,
    pub duration_secs: f64,
}

/// Session family inventory - auto-maintained across segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInventory {
    /// Family ID (e.g., "perp_BTCUSDT_20260125")
    pub session_family: String,
    /// Primary symbol
    pub symbol: String,
    /// Capture mode
    pub mode: String,
    /// Binary hash (shared across segments, or "MIXED" if different)
    pub binary_hash: String,
    /// Schema version
    pub schema_version: String,
    /// Ordered list of segments
    pub segments: Vec<InventorySegment>,
    /// Gaps between segments
    pub gaps: Vec<InventoryGap>,
    /// Notes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl SessionInventory {
    /// Create a new inventory for a session family.
    pub fn new(session_family: String, symbol: String, mode: String, binary_hash: String) -> Self {
        Self {
            session_family,
            symbol,
            mode,
            binary_hash,
            schema_version: "fixed_point_v1".to_string(),
            segments: Vec::new(),
            gaps: Vec::new(),
            notes: None,
        }
    }

    /// Add a segment to the inventory, computing gap from prior if applicable.
    pub fn add_segment(&mut self, manifest: &SegmentManifest) {
        // Check for gap from prior segment
        if let Some(last) = self.segments.last()
            && let Some(ref last_end) = last.end_ts
            && let Ok(last_end_dt) = DateTime::parse_from_rfc3339(last_end)
        {
            let start_ts = manifest.start_ts.to_rfc3339();
            let gap_secs = (manifest.start_ts - last_end_dt.with_timezone(&Utc)).num_milliseconds()
                as f64
                / 1000.0;
            if gap_secs > 1.0 {
                // Only record gaps > 1 second
                self.gaps.push(InventoryGap {
                    from_segment: last.segment_id.clone(),
                    to_segment: manifest.segment_id.clone(),
                    from_ts: last_end.clone(),
                    to_ts: start_ts,
                    duration_secs: gap_secs,
                });
            }
        }

        // Add segment entry
        let entry = InventorySegment {
            segment_id: manifest.segment_id.clone(),
            path: manifest.segment_id.clone(), // Relative path
            start_ts: manifest.start_ts.to_rfc3339(),
            end_ts: manifest.end_ts.map(|t| t.to_rfc3339()),
            duration_secs: manifest.duration_secs,
            stop_reason: manifest.stop_reason,
            events: manifest.events.clone(),
            usable: true,
        };
        self.segments.push(entry);

        // Check for binary hash consistency
        if self.binary_hash != manifest.binary_hash && self.binary_hash != "MIXED" {
            self.binary_hash = "MIXED".to_string();
        }
    }

    /// Update the last segment with finalized manifest data.
    pub fn update_last_segment(&mut self, manifest: &SegmentManifest) {
        if let Some(last) = self.segments.last_mut()
            && last.segment_id == manifest.segment_id
        {
            last.end_ts = manifest.end_ts.map(|t| t.to_rfc3339());
            last.duration_secs = manifest.duration_secs;
            last.stop_reason = manifest.stop_reason;
            last.events = manifest.events.clone();
        }
    }

    /// Write inventory to disk.
    pub fn write(&self, out_dir: &Path) -> Result<()> {
        // Extract date from session_family (e.g., "perp_BTCUSDT_20260125" -> "20260125")
        let date_part = self
            .session_family
            .split('_')
            .next_back()
            .unwrap_or("unknown");
        let filename = format!("perp_{}_inventory.json", date_part);
        let inventory_path = out_dir.join(filename);

        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&inventory_path, json)
            .with_context(|| format!("write inventory: {:?}", inventory_path))?;
        tracing::info!("Inventory updated: {:?}", inventory_path);
        Ok(())
    }

    /// Load inventory from disk, or create new if not found.
    pub fn load_or_create(
        out_dir: &Path,
        session_family: &str,
        symbol: &str,
        mode: &str,
        binary_hash: &str,
    ) -> Self {
        // Extract date from session_family
        let date_part = session_family.split('_').next_back().unwrap_or("unknown");
        let filename = format!("perp_{}_inventory.json", date_part);
        let inventory_path = out_dir.join(filename);

        if inventory_path.exists() {
            match std::fs::read_to_string(&inventory_path) {
                Ok(content) => match serde_json::from_str(&content) {
                    Ok(inv) => return inv,
                    Err(e) => {
                        tracing::warn!("Failed to parse inventory {:?}: {}", inventory_path, e);
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read inventory {:?}: {}", inventory_path, e);
                }
            }
        }

        Self::new(
            session_family.to_string(),
            symbol.to_string(),
            mode.to_string(),
            binary_hash.to_string(),
        )
    }
}

/// Compute SHA256 hash of the current binary.
pub fn compute_binary_hash() -> Result<String> {
    let exe_path = std::env::current_exe().context("get current exe path")?;
    let bytes = std::fs::read(&exe_path).context("read binary")?;
    let hash = Sha256::digest(&bytes);
    Ok(hex::encode(hash))
}

/// Compute SHA256 hash of a file.
pub fn compute_file_sha256(file_path: &Path) -> Result<String> {
    let bytes = std::fs::read(file_path)
        .with_context(|| format!("read file for sha256: {:?}", file_path))?;
    let hash = Sha256::digest(&bytes);
    Ok(hex::encode(hash))
}

/// Compute digest for a stream file (JSONL format).
///
/// Returns None if the file doesn't exist or is empty.
pub fn compute_stream_digest(file_path: &Path) -> Result<Option<StreamDigest>> {
    use std::io::{BufRead, BufReader};

    if !file_path.exists() {
        return Ok(None);
    }

    let metadata = std::fs::metadata(file_path)?;
    let size_bytes = metadata.len();
    if size_bytes == 0 {
        return Ok(None);
    }

    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut hasher = Sha256::new();
    let mut event_count = 0usize;
    let mut first_ts: Option<String> = None;
    let mut last_ts: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
        event_count += 1;

        // Try to extract timestamp from JSONL (look for "ts" field)
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line)
            && let Some(ts) = json.get("ts").and_then(|v| v.as_str())
        {
            if first_ts.is_none() {
                first_ts = Some(ts.to_string());
            }
            last_ts = Some(ts.to_string());
        }
    }

    if event_count == 0 {
        return Ok(None);
    }

    Ok(Some(StreamDigest {
        file_path: file_path.display().to_string(),
        sha256: hex::encode(hasher.finalize()),
        event_count,
        size_bytes,
        first_event_ts: first_ts,
        last_event_ts: last_ts,
    }))
}

/// Compute all stream digests for a segment directory.
///
/// Handles the actual directory structure where files are in symbol subdirectories:
/// ```text
/// segment_dir/
/// └── BTCUSDT/
///     ├── spot_quotes.jsonl
///     ├── perp_quotes.jsonl (or perp_depth.jsonl)
///     ├── funding.jsonl
///     └── perp_depth.jsonl (optional)
/// ```
pub fn compute_segment_digests(segment_dir: &Path) -> Result<SegmentDigests> {
    let mut spot_digest: Option<StreamDigest> = None;
    let mut perp_digest: Option<StreamDigest> = None;
    let mut funding_digest: Option<StreamDigest> = None;
    let mut depth_digest: Option<StreamDigest> = None;

    // Iterate through symbol subdirectories
    for entry in std::fs::read_dir(segment_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        // Skip non-symbol directories (like .tmp files)
        let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !dir_name.chars().all(|c| c.is_alphanumeric()) {
            continue;
        }

        // Check for spot quotes
        let spot_path = path.join("spot_quotes.jsonl");
        if spot_path.exists()
            && let Some(digest) = compute_stream_digest(&spot_path)?
        {
            spot_digest = Some(digest);
        }

        // Check for perp quotes (prefer perp_depth if exists)
        let perp_depth_path = path.join("perp_depth.jsonl");
        let perp_quotes_path = path.join("perp_quotes.jsonl");
        if perp_depth_path.exists()
            && let Some(digest) = compute_stream_digest(&perp_depth_path)?
        {
            perp_digest = Some(digest.clone());
            depth_digest = Some(digest);
        } else if perp_quotes_path.exists()
            && let Some(digest) = compute_stream_digest(&perp_quotes_path)?
        {
            perp_digest = Some(digest);
        }

        // Check for funding
        let funding_path = path.join("funding.jsonl");
        if funding_path.exists()
            && let Some(digest) = compute_stream_digest(&funding_path)?
        {
            funding_digest = Some(digest);
        }
    }

    Ok(SegmentDigests {
        spot: spot_digest,
        perp: perp_digest,
        funding: funding_digest,
        depth: depth_digest,
    })
}

/// Managed segment capture with automatic manifest handling.
pub struct ManagedSegment {
    pub manifest: Arc<Mutex<SegmentManifest>>,
    pub segment_dir: PathBuf,
    pub out_dir: PathBuf,
    shutdown_flag: Arc<AtomicBool>,
    heartbeat_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ManagedSegment {
    /// Start a new managed segment.
    pub async fn start(
        out_dir: &Path,
        symbols: &[String],
        capture_mode: &str,
        config: CaptureConfig,
    ) -> Result<Self> {
        let binary_hash = compute_binary_hash().unwrap_or_else(|_| "UNKNOWN".to_string());
        let start_time = Utc::now();

        // Generate segment ID and family ID
        let segment_tag = format!("perp_{}", start_time.format("%Y%m%d_%H%M%S"));
        let date_str = start_time.format("%Y%m%d").to_string();
        let primary_symbol = symbols.first().map(|s| s.as_str()).unwrap_or("UNKNOWN");
        let session_family_id = format!("perp_{}_{}", primary_symbol, date_str);

        // Create segment directory
        let segment_dir = out_dir.join(&segment_tag);
        std::fs::create_dir_all(&segment_dir)
            .with_context(|| format!("create segment dir: {:?}", segment_dir))?;

        // Create manifest
        let manifest = SegmentManifest::new(
            session_family_id.clone(),
            segment_tag.clone(),
            symbols.to_vec(),
            capture_mode.to_string(),
            binary_hash.clone(),
            config,
        );

        // Check for prior segment and compute gap
        let mut inventory = SessionInventory::load_or_create(
            out_dir,
            &session_family_id,
            primary_symbol,
            capture_mode,
            &binary_hash,
        );

        let mut manifest = manifest;
        if let Some(last) = inventory.segments.last()
            && let Some(ref last_end) = last.end_ts
            && let Ok(last_end_dt) = DateTime::parse_from_rfc3339(last_end)
        {
            let gap_secs =
                (start_time - last_end_dt.with_timezone(&Utc)).num_milliseconds() as f64 / 1000.0;
            manifest.gap_from_prior = Some(GapInfo {
                previous_segment_id: last.segment_id.clone(),
                gap_seconds: gap_secs,
                reason: "restart".to_string(),
            });
            tracing::info!(
                "Gap detected from prior segment {}: {:.1}s",
                last.segment_id,
                gap_secs
            );
        }

        // Write initial manifest
        manifest.write(&segment_dir)?;
        tracing::info!(
            "Segment manifest created: {:?}",
            segment_dir.join("segment_manifest.json")
        );

        // Add to inventory (will be updated on finalize)
        inventory.add_segment(&manifest);
        inventory.write(out_dir)?;

        let manifest = Arc::new(Mutex::new(manifest));
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        // Start heartbeat task
        let heartbeat_manifest = Arc::clone(&manifest);
        let heartbeat_dir = segment_dir.clone();
        let heartbeat_shutdown = Arc::clone(&shutdown_flag);
        let heartbeat_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                if heartbeat_shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let mut m = heartbeat_manifest.lock().await;
                m.heartbeat();
                if let Err(e) = m.write(&heartbeat_dir) {
                    tracing::warn!("Failed to write heartbeat: {}", e);
                }
            }
        });

        Ok(Self {
            manifest,
            segment_dir,
            out_dir: out_dir.to_path_buf(),
            shutdown_flag,
            heartbeat_handle: Some(heartbeat_handle),
        })
    }

    /// Get the segment directory path.
    pub fn segment_dir(&self) -> &Path {
        &self.segment_dir
    }

    /// Get the segment ID.
    pub async fn segment_id(&self) -> String {
        self.manifest.lock().await.segment_id.clone()
    }

    /// Finalize the segment with stop reason and event counts.
    pub async fn finalize(&mut self, stop_reason: StopReason, events: EventCounts) -> Result<()> {
        // Stop heartbeat
        self.shutdown_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.heartbeat_handle.take() {
            handle.abort();
        }

        // Compute stream digests
        let digests = compute_segment_digests(&self.segment_dir).ok();

        // Update manifest
        {
            let mut m = self.manifest.lock().await;
            m.finalize(stop_reason, events, digests);
            m.write(&self.segment_dir)?;
        }

        // Update inventory
        let manifest = self.manifest.lock().await;
        let mut inventory = SessionInventory::load_or_create(
            &self.out_dir,
            &manifest.session_family_id,
            manifest
                .symbols
                .first()
                .map(|s| s.as_str())
                .unwrap_or("UNKNOWN"),
            &manifest.capture_mode,
            &manifest.binary_hash,
        );
        inventory.update_last_segment(&manifest);
        inventory.write(&self.out_dir)?;

        tracing::info!(
            "Segment finalized: {} ({})",
            manifest.segment_id,
            stop_reason
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_reason_serialization() {
        let reason = StopReason::ExternalKillSighup;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"EXTERNAL_KILL_SIGHUP\"");

        let parsed: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, reason);
    }

    #[test]
    fn test_event_counts_total() {
        let counts = EventCounts {
            spot_quotes: 100,
            perp_quotes: 200,
            funding: 10,
            depth: 0,
        };
        assert_eq!(counts.total(), 310);
    }

    #[test]
    fn test_manifest_lifecycle() {
        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260125".to_string(),
            "perp_20260125_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "capture-perp-session".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        assert_eq!(manifest.stop_reason, StopReason::Running);
        assert!(manifest.end_ts.is_none());
        assert_eq!(manifest.state, SegmentState::Bootstrap);
        assert_eq!(manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.quote_schema, "canonical_v1");

        manifest.heartbeat();
        assert!(manifest.heartbeat_ts >= manifest.start_ts);

        let events = EventCounts {
            spot_quotes: 1000,
            perp_quotes: 2000,
            funding: 50,
            depth: 0,
        };
        manifest.finalize(StopReason::NormalCompletion, events, None);

        assert_eq!(manifest.stop_reason, StopReason::NormalCompletion);
        assert!(manifest.end_ts.is_some());
        assert!(manifest.duration_secs.is_some());
        assert_eq!(manifest.events.total(), 3050);
        assert_eq!(manifest.state, SegmentState::Finalized);
    }

    #[test]
    fn test_retro_finalize() {
        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260125".to_string(),
            "perp_20260125_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "capture-perp-session".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        assert_eq!(manifest.state, SegmentState::Bootstrap);
        assert_eq!(manifest.stop_reason, StopReason::Running);

        let events = EventCounts {
            spot_quotes: 500,
            perp_quotes: 1000,
            funding: 25,
            depth: 0,
        };
        let digests = SegmentDigests::default();
        manifest.retro_finalize(events, digests);

        assert_eq!(manifest.state, SegmentState::FinalizedRetro);
        assert_eq!(manifest.stop_reason, StopReason::Unknown);
        assert!(manifest.end_ts.is_some());
        assert!(manifest.digests.is_some());
    }

    #[test]
    fn test_attribution_binding() {
        use std::io::Write;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let segment_dir = temp_dir.path();

        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260125".to_string(),
            "perp_20260125_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "capture-perp-session".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        // Schema version should be 8 (Phase 5)
        assert_eq!(manifest.schema_version, 9);
        assert!(manifest.attribution_binding.is_none());

        // Create a test attribution file
        let attribution_path = segment_dir.join("attribution.jsonl");
        let mut file = std::fs::File::create(&attribution_path).unwrap();
        writeln!(
            file,
            r#"{{"ts_ns":1000,"symbol":"BTCUSDT","net_pnl_mantissa":1000000000}}"#
        )
        .unwrap();

        // Bind the attribution
        manifest
            .bind_attribution(
                &attribution_path,
                segment_dir,
                1,          // num_events
                1000000000, // total_net_pnl_mantissa (~$10)
                20000,      // total_fees_mantissa
                -8,         // pnl_exponent
            )
            .unwrap();

        // Verify binding
        let binding = manifest.attribution_binding.as_ref().unwrap();
        assert_eq!(binding.attribution_path, "attribution.jsonl");
        assert_eq!(binding.num_attribution_events, 1);
        assert_eq!(binding.total_net_pnl_mantissa, 1000000000);
        assert_eq!(binding.total_fees_mantissa, 20000);
        assert_eq!(binding.pnl_exponent, -8);
        assert!(!binding.attribution_sha256.is_empty());
    }

    #[test]
    fn test_attribution_summary_binding() {
        use quantlaxmi_models::{AlphaScoreV1, AttributionSummary};
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let segment_dir = temp_dir.path();

        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260125".to_string(),
            "perp_20260125_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "backtest".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        // Schema version should be 8 (Phase 5)
        assert_eq!(manifest.schema_version, 9);
        assert!(manifest.attribution_summary_binding.is_none());

        // Create a test attribution summary
        let summary = AttributionSummary {
            strategy_id: "funding_bias:2.0.0:abc123def".to_string(),
            run_id: "run_001".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            generated_ts_ns: 1_706_180_400_000_000_000,
            total_decisions: 100,
            total_fills: 200,
            winning_decisions: 60,
            losing_decisions: 40,
            round_trips: 30,
            total_gross_pnl_mantissa: 5_000_000_000, // $50
            total_fees_mantissa: 200_000,            // $0.002
            total_net_pnl_mantissa: 4_999_800_000,   // $49.998
            pnl_exponent: -8,
            win_rate_bps: 6000,                        // 60%
            avg_pnl_per_decision_mantissa: 49_998_000, // ~$0.50
            total_slippage_mantissa: 10_000_000,
            slippage_exponent: -8,
            max_loss_mantissa: 500_000_000, // $5 max loss
            total_holding_time_ns: 3_600_000_000_000,
        };

        // Compute alpha score
        let alpha_score = AlphaScoreV1::from_summary(&summary);

        // Bind the summary
        manifest
            .bind_attribution_summary(&summary, &alpha_score, segment_dir)
            .unwrap();

        // Verify binding
        let binding = manifest.attribution_summary_binding.as_ref().unwrap();
        assert_eq!(binding.summary_path, "attribution_summary.json");
        assert_eq!(binding.strategy_id, "funding_bias:2.0.0:abc123def");
        assert_eq!(binding.total_decisions, 100);
        assert_eq!(binding.total_fills, 200);
        assert_eq!(binding.win_rate_bps, 6000);
        assert_eq!(binding.total_net_pnl_mantissa, 4_999_800_000);
        assert_eq!(binding.max_loss_mantissa, 500_000_000);
        assert_eq!(binding.pnl_exponent, -8);
        assert_eq!(binding.alpha_score_formula, "alpha_score_v1.0");
        assert!(!binding.summary_sha256.is_empty());

        // Verify the file was written
        let summary_path = segment_dir.join("attribution_summary.json");
        assert!(summary_path.exists());

        // Verify file can be parsed back
        let content = std::fs::read_to_string(&summary_path).unwrap();
        let loaded: AttributionSummary = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.strategy_id, summary.strategy_id);
        assert_eq!(loaded.total_decisions, summary.total_decisions);

        // Verify manifest round-trip
        manifest.write(segment_dir).unwrap();
        let loaded_manifest = SegmentManifest::load(segment_dir).unwrap();
        assert!(loaded_manifest.attribution_summary_binding.is_some());
        let loaded_binding = loaded_manifest.attribution_summary_binding.unwrap();
        assert_eq!(loaded_binding.strategy_id, "funding_bias:2.0.0:abc123def");
        assert_eq!(
            loaded_binding.alpha_score_mantissa,
            alpha_score.score_mantissa
        );
    }

    #[test]
    fn test_router_binding() {
        use quantlaxmi_models::{RegimeInputs, RegimeLabel, RouterDecisionEvent};
        use tempfile::TempDir;
        use uuid::Uuid;

        let temp_dir = TempDir::new().unwrap();
        let segment_dir = temp_dir.path();

        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260126".to_string(),
            "perp_20260126_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "backtest".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        // Schema version should be 8 (Phase 5)
        assert_eq!(manifest.schema_version, 9);
        assert!(manifest.router_binding.is_none());

        // Create test regime inputs
        let make_inputs = |ts_ns: i64, vol_bps: i32, vol_tier: u8| RegimeInputs {
            ts_ns,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 10,
            spread_tier: 0,
            volatility_bps: vol_bps,
            volatility_tier: vol_tier,
            depth_mantissa: 100_000_000,
            depth_exponent: -2,
            liquidity_tier: 0,
            funding_rate_bps: 5,
            funding_tier: 0,
            trend_strength: 0,
        };

        // Create test router decisions
        let decisions = vec![
            RouterDecisionEvent {
                ts_ns: 1_000_000_000,
                decision_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440001").unwrap(),
                symbols: vec!["BTCUSDT".to_string()],
                inputs: make_inputs(1_000_000_000, 50, 0),
                regime: RegimeLabel::Normal,
                confidence_bps: 8000,
                selected_strategy_id: "funding_bias:2.0.0:abc123".to_string(),
                alternatives: vec![],
                selection_reason: "Best match for Normal regime".to_string(),
                router_config_hash: "router_hash_001".to_string(),
                router_version: "router_v1.0".to_string(),
            },
            RouterDecisionEvent {
                ts_ns: 2_000_000_000,
                decision_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440002").unwrap(),
                symbols: vec!["BTCUSDT".to_string()],
                inputs: make_inputs(2_000_000_000, 300, 2),
                regime: RegimeLabel::HighVol,
                confidence_bps: 9000,
                selected_strategy_id: "momentum:1.0.0:xyz456".to_string(),
                alternatives: vec!["funding_bias:2.0.0:abc123".to_string()],
                selection_reason: "Best match for HighVol regime".to_string(),
                router_config_hash: "router_hash_001".to_string(),
                router_version: "router_v1.0".to_string(),
            },
            RouterDecisionEvent {
                ts_ns: 3_000_000_000,
                decision_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440003").unwrap(),
                symbols: vec!["BTCUSDT".to_string()],
                inputs: make_inputs(3_000_000_000, 60, 0),
                regime: RegimeLabel::Normal,
                confidence_bps: 7500,
                selected_strategy_id: "funding_bias:2.0.0:abc123".to_string(),
                alternatives: vec![],
                selection_reason: "Best match for Normal regime".to_string(),
                router_config_hash: "router_hash_001".to_string(),
                router_version: "router_v1.0".to_string(),
            },
        ];

        // Bind router decisions
        manifest
            .bind_router_decisions(&decisions, "router_hash_001", "router_v1.0", segment_dir)
            .unwrap();

        // Verify binding
        let binding = manifest.router_binding.as_ref().unwrap();
        assert_eq!(binding.decisions_path, "router_decisions.jsonl");
        assert_eq!(binding.num_decisions, 3);
        assert_eq!(binding.router_config_hash, "router_hash_001");
        assert_eq!(binding.router_version, "router_v1.0");
        assert_eq!(binding.unique_strategies_used, 2); // funding_bias and momentum
        assert!(!binding.decisions_sha256.is_empty());

        // Verify regime distribution (as_str returns uppercase like "NORMAL", "HIGH_VOL")
        assert_eq!(*binding.regime_distribution.get("NORMAL").unwrap(), 2);
        assert_eq!(*binding.regime_distribution.get("HIGH_VOL").unwrap(), 1);

        // Verify the decisions file was written
        let decisions_path = segment_dir.join("router_decisions.jsonl");
        assert!(decisions_path.exists());

        // Verify decisions file can be parsed back
        let content = std::fs::read_to_string(&decisions_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);

        // Parse first decision and verify
        let parsed: RouterDecisionEvent = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed.regime, RegimeLabel::Normal);
        assert_eq!(parsed.selected_strategy_id, "funding_bias:2.0.0:abc123");

        // Verify manifest round-trip
        manifest.write(segment_dir).unwrap();
        let loaded_manifest = SegmentManifest::load(segment_dir).unwrap();
        assert!(loaded_manifest.router_binding.is_some());

        let loaded_binding = loaded_manifest.router_binding.unwrap();
        assert_eq!(loaded_binding.router_config_hash, "router_hash_001");
        assert_eq!(loaded_binding.num_decisions, 3);
        assert_eq!(loaded_binding.unique_strategies_used, 2);
    }
}
