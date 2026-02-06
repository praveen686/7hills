//! Tournament Types for Alpha Discovery
//!
//! Phase 12.2: Pure schema types for tournament evaluation and ranking.
//!
//! ## Determinism Contract
//! - Tournament ID derived from inputs (no wall-clock)
//! - Stable ordering: segments by ID, strategies by name
//! - All timestamps derived from segment data
//! - Canonical JSON serialization (typed structs only)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

// =============================================================================
// Schema Versions
// =============================================================================

pub const TOURNAMENT_MANIFEST_SCHEMA: &str = "tournament_manifest_v1.0";
pub const LEADERBOARD_SCHEMA: &str = "leaderboard_v1.0";
pub const RUN_MANIFEST_SCHEMA: &str = "run_manifest_v1.0";

// =============================================================================
// Artifact Digest
// =============================================================================

/// Artifact digest with hash and size.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactDigest {
    pub sha256: String,
    pub bytes: u64,
}

impl ArtifactDigest {
    /// Compute digest from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let hash = Sha256::digest(data);
        Self {
            sha256: hex::encode(hash),
            bytes: data.len() as u64,
        }
    }
}

// =============================================================================
// Input Segment Record
// =============================================================================

/// Reference to an input segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSegment {
    /// Segment ID (deterministic, derived from content)
    pub segment_id: String,
    /// Relative path to segment directory
    pub path: String,
    /// SHA-256 of segment_manifest.json
    pub manifest_sha256: String,
}

// =============================================================================
// Run Record
// =============================================================================

/// Record for a single strategy/segment evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Deterministic run ID: sha256("tournament_run_v1:" + segment_id + ":" + strategy_id)
    pub run_id: String,
    /// Run key for filesystem: "{segment_id}__{strategy_id}"
    pub run_key: String,
    /// Segment ID
    pub segment_id: String,
    /// Full strategy ID (name:version:hash)
    pub strategy_id: String,
    /// Strategy name only
    pub strategy_name: String,
    /// Relative paths to result artifacts
    pub result_paths: RunResultPaths,

    // === Alpha Score Summary ===
    /// Alpha score mantissa
    pub alpha_score_mantissa: i64,
    /// Alpha score exponent
    pub alpha_score_exponent: i8,

    // === Activity Counts ===
    pub decisions: u32,
    pub fills: u32,
    pub round_trips: u32,
    pub win_rate_bps: u32,

    // === PnL ===
    pub net_pnl_mantissa: i128,
    pub pnl_exponent: i8,

    // === Gate Results ===
    pub g1_passed: bool,
    pub g1_reasons: Vec<String>,
    pub g2_passed: Option<bool>,
    pub g2_reasons: Vec<String>,
    pub g3_passed: Option<bool>,
    pub g3_reasons: Vec<String>,

    /// Run manifest digest
    pub run_manifest_sha256: String,
}

/// Relative paths to run result artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResultPaths {
    pub decision_trace: String,
    pub attribution_summary: String,
    pub alpha_score: String,
    pub g1_result: String,
    pub g2_result: Option<String>,
    pub g3_result: Option<String>,
    pub run_manifest: String,
}

// =============================================================================
// Tournament Manifest
// =============================================================================

/// Tournament manifest - deterministic index of all inputs and outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentManifestV1 {
    /// Schema version
    pub schema_version: String,
    /// Deterministic tournament ID
    pub tournament_id: String,
    /// Created timestamp (derived from max segment end time)
    pub created_ts_ns: i64,
    /// Preset name
    pub preset: String,
    /// Input segments (sorted by segment_id)
    pub input_segments: Vec<InputSegment>,
    /// Strategy IDs evaluated (sorted)
    pub strategies: Vec<String>,
    /// Symbols filter
    pub symbols: Vec<String>,
    /// Run records (sorted by run_key)
    pub runs: Vec<RunRecord>,
    /// Artifact digests (filename -> digest)
    pub artifact_digests: BTreeMap<String, ArtifactDigest>,
    /// Bundle digest (sha256 of sorted artifact digests)
    pub bundle_digest: String,
}

impl TournamentManifestV1 {
    pub const SCHEMA_VERSION: &'static str = TOURNAMENT_MANIFEST_SCHEMA;
}

// =============================================================================
// Leaderboard
// =============================================================================

/// Leaderboard row for ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardRow {
    /// Rank (1-indexed)
    pub rank: u32,
    /// Strategy ID (name:version:hash)
    pub strategy_id: String,
    /// Strategy name only
    pub strategy_name: String,
    /// Segment ID
    pub segment_id: String,
    /// Run key
    pub run_key: String,
    /// Alpha score mantissa
    pub alpha_score_mantissa: i64,
    /// Alpha score exponent
    pub alpha_score_exponent: i8,
    /// Alpha score as f64 (for display)
    pub alpha_score_f64: f64,
    /// G1 passed
    pub g1_passed: bool,
    /// G2 passed
    pub g2_passed: Option<bool>,
    /// G3 passed
    pub g3_passed: Option<bool>,
    /// Total decisions
    pub decisions: u32,
    /// Total fills
    pub fills: u32,
    /// Round-trips
    pub round_trips: u32,
    /// Win rate (bps)
    pub win_rate_bps: u32,
    /// Net PnL mantissa
    pub net_pnl_mantissa: i128,
    /// PnL exponent
    pub pnl_exponent: i8,
}

/// Leaderboard - ranked list of strategy runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardV1 {
    /// Schema version
    pub schema_version: String,
    /// Tournament ID
    pub tournament_id: String,
    /// Rows sorted by rank
    pub rows: Vec<LeaderboardRow>,
    /// Total runs
    pub total_runs: u32,
    /// Meaningful runs (met activity threshold)
    pub meaningful_runs: u32,
    /// Created timestamp
    pub created_ts_ns: i64,
}

impl LeaderboardV1 {
    pub const SCHEMA_VERSION: &'static str = LEADERBOARD_SCHEMA;
}

// =============================================================================
// Run Manifest (per-run index)
// =============================================================================

/// Per-run manifest with artifact digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifestV1 {
    /// Schema version
    pub schema_version: String,
    /// Run ID
    pub run_id: String,
    /// Run key
    pub run_key: String,
    /// Strategy ID
    pub strategy_id: String,
    /// Segment ID
    pub segment_id: String,
    /// Segment path
    pub segment_path: String,
    /// Symbols
    pub symbols: Vec<String>,

    // === Metrics ===
    pub decisions: u32,
    pub fills: u32,
    pub round_trips: u32,
    pub win_rate_bps: u32,
    pub net_pnl_mantissa: i128,
    pub pnl_exponent: i8,
    pub alpha_score_mantissa: i64,
    pub alpha_score_exponent: i8,

    // === Gate Results ===
    pub g1_passed: bool,
    pub g1_reasons: Vec<String>,
    pub g2_passed: Option<bool>,
    pub g2_reasons: Vec<String>,
    pub g3_passed: Option<bool>,
    pub g3_reasons: Vec<String>,

    // === Artifact Digests ===
    pub decision_trace_sha256: String,
    pub attribution_summary_sha256: String,
    pub alpha_score_sha256: String,
    pub g1_result_sha256: String,
    pub g2_result_sha256: Option<String>,
    pub g3_result_sha256: Option<String>,

    /// Derived timestamp
    pub derived_ts_ns: i64,
}

impl RunManifestV1 {
    pub const SCHEMA_VERSION: &'static str = RUN_MANIFEST_SCHEMA;
}

// =============================================================================
// Tournament Preset
// =============================================================================

/// Tournament preset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentPreset {
    pub name: String,
    pub description: String,
    /// Minimum decisions for meaningful run
    pub min_decisions: u32,
    /// Minimum fills for meaningful run
    pub min_fills: u32,
    /// Minimum round-trips for meaningful run
    pub min_round_trips: u32,
}

impl TournamentPreset {
    /// Baseline v1 preset.
    pub fn baseline_v1() -> Self {
        Self {
            name: "baseline_v1".to_string(),
            description: "Single run per strategy/segment, default configs only".to_string(),
            min_decisions: 1,
            min_fills: 1,
            min_round_trips: 0,
        }
    }
}

impl Default for TournamentPreset {
    fn default() -> Self {
        Self::baseline_v1()
    }
}

// =============================================================================
// Ranking Functions
// =============================================================================

/// Check if a run is meaningful (meets activity threshold).
pub fn is_meaningful_run(
    decisions: u32,
    fills: u32,
    round_trips: u32,
    preset: &TournamentPreset,
) -> bool {
    decisions >= preset.min_decisions
        && fills >= preset.min_fills
        && round_trips >= preset.min_round_trips
}

/// Compare two leaderboard rows for ranking.
/// Returns Ordering::Less if a ranks higher than b.
///
/// Sort order:
/// 1. total_score desc
/// 2. g3_pass desc
/// 3. g2_pass desc
/// 4. round_trips desc
/// 5. run_key asc (deterministic tie-break)
pub fn compare_rows(a: &LeaderboardRow, b: &LeaderboardRow) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    // 1. Alpha score desc (higher is better)
    match b.alpha_score_mantissa.cmp(&a.alpha_score_mantissa) {
        Ordering::Equal => {}
        ord => return ord,
    }

    // 2. G3 pass desc (pass > fail > None)
    match (a.g3_passed, b.g3_passed) {
        (Some(true), Some(false)) => return Ordering::Less,
        (Some(false), Some(true)) => return Ordering::Greater,
        (Some(true), None) => return Ordering::Less,
        (None, Some(true)) => return Ordering::Greater,
        _ => {}
    }

    // 3. G2 pass desc
    match (a.g2_passed, b.g2_passed) {
        (Some(true), Some(false)) => return Ordering::Less,
        (Some(false), Some(true)) => return Ordering::Greater,
        (Some(true), None) => return Ordering::Less,
        (None, Some(true)) => return Ordering::Greater,
        _ => {}
    }

    // 4. Round-trips desc
    match b.round_trips.cmp(&a.round_trips) {
        Ordering::Equal => {}
        ord => return ord,
    }

    // 5. Run key asc (deterministic tie-break)
    a.run_key.cmp(&b.run_key)
}

// =============================================================================
// ID Generation
// =============================================================================

/// Generate deterministic run ID.
pub fn generate_run_id(segment_id: &str, strategy_id: &str) -> String {
    let input = format!("tournament_run_v1:{}:{}", segment_id, strategy_id);
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(hash)
}

/// Generate deterministic run key (filesystem-safe).
pub fn generate_run_key(segment_id: &str, strategy_name: &str) -> String {
    format!("{}__{}", segment_id, strategy_name)
}

/// Generate deterministic tournament ID.
pub fn generate_tournament_id(
    preset: &str,
    segment_digests: &[String],
    strategies: &[String],
) -> String {
    let mut input = format!("tournament_v1:{}:", preset);
    for digest in segment_digests {
        input.push_str(digest);
        input.push(':');
    }
    for strategy in strategies {
        input.push_str(strategy);
        input.push(':');
    }
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(hash)
}

/// Compute bundle digest from artifact digests.
pub fn compute_bundle_digest(digests: &BTreeMap<String, ArtifactDigest>) -> String {
    let mut input = String::new();
    for (name, digest) in digests {
        input.push_str(&format!("{}:{}:{}\n", name, digest.sha256, digest.bytes));
    }
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(hash)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_run_id_deterministic() {
        let id1 = generate_run_id("seg123", "funding_bias:1.0.0:abc123");
        let id2 = generate_run_id("seg123", "funding_bias:1.0.0:abc123");
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_generate_run_key() {
        let key = generate_run_key("seg123", "funding_bias");
        assert_eq!(key, "seg123__funding_bias");
    }

    #[test]
    fn test_generate_tournament_id_deterministic() {
        let id1 = generate_tournament_id(
            "baseline_v1",
            &["digest1".to_string(), "digest2".to_string()],
            &["strategy_a".to_string(), "strategy_b".to_string()],
        );
        let id2 = generate_tournament_id(
            "baseline_v1",
            &["digest1".to_string(), "digest2".to_string()],
            &["strategy_a".to_string(), "strategy_b".to_string()],
        );
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_is_meaningful_run() {
        let preset = TournamentPreset::baseline_v1();
        assert!(is_meaningful_run(5, 5, 2, &preset));
        assert!(!is_meaningful_run(0, 5, 2, &preset));
        assert!(!is_meaningful_run(5, 0, 2, &preset));
    }

    #[test]
    fn test_compare_rows_by_alpha() {
        let base = LeaderboardRow {
            rank: 0,
            strategy_id: "test:1.0.0:abc".to_string(),
            strategy_name: "test".to_string(),
            segment_id: "seg1".to_string(),
            run_key: "seg1__test".to_string(),
            alpha_score_mantissa: 1000,
            alpha_score_exponent: -4,
            alpha_score_f64: 0.1,
            g1_passed: true,
            g2_passed: None,
            g3_passed: None,
            decisions: 10,
            fills: 10,
            round_trips: 5,
            win_rate_bps: 5000,
            net_pnl_mantissa: 1000,
            pnl_exponent: -2,
        };

        let higher = LeaderboardRow {
            alpha_score_mantissa: 2000,
            run_key: "seg1__higher".to_string(),
            ..base.clone()
        };

        // higher alpha should rank better (Less in sort)
        assert_eq!(compare_rows(&higher, &base), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_compare_rows_g3_tiebreaker() {
        let base = LeaderboardRow {
            rank: 0,
            strategy_id: "test:1.0.0:abc".to_string(),
            strategy_name: "test".to_string(),
            segment_id: "seg1".to_string(),
            run_key: "seg1__test_a".to_string(),
            alpha_score_mantissa: 1000,
            alpha_score_exponent: -4,
            alpha_score_f64: 0.1,
            g1_passed: true,
            g2_passed: Some(true),
            g3_passed: Some(true),
            decisions: 10,
            fills: 10,
            round_trips: 5,
            win_rate_bps: 5000,
            net_pnl_mantissa: 1000,
            pnl_exponent: -2,
        };

        let g3_fail = LeaderboardRow {
            g3_passed: Some(false),
            run_key: "seg1__test_b".to_string(),
            ..base.clone()
        };

        // G3 pass should rank better
        assert_eq!(compare_rows(&base, &g3_fail), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_compute_bundle_digest_deterministic() {
        let mut digests = BTreeMap::new();
        digests.insert(
            "file1.json".to_string(),
            ArtifactDigest {
                sha256: "abc123".to_string(),
                bytes: 100,
            },
        );
        digests.insert(
            "file2.json".to_string(),
            ArtifactDigest {
                sha256: "def456".to_string(),
                bytes: 200,
            },
        );

        let digest1 = compute_bundle_digest(&digests);
        let digest2 = compute_bundle_digest(&digests);
        assert_eq!(digest1, digest2);
    }
}
