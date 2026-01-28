//! G4 Admission Determinism Gate — Strategy admission WAL parity check.
//!
//! Phase 23A: Compares two `strategy_admission.jsonl` WALs (live vs replay)
//! to verify deterministic admission decisions.
//!
//! ## Key Structure
//! Decisions are keyed by: `(correlation_id, strategy_id, signal_id)`
//!
//! ## Comparison Rules (Frozen)
//! For each key present in either WAL:
//! 1. If missing in replay → `MissingInReplay`
//! 2. If missing in live → `MissingInLive`
//! 3. If outcome differs → `OutcomeMismatch`
//! 4. If refuse_reasons differ → `ReasonsMismatch`
//! 5. If digest differs → `DigestMismatch`
//! 6. Else → Match (counted, not stored)
//!
//! ## Usage
//! ```ignore
//! use quantlaxmi_gates::g4_admission_determinism::G4AdmissionDeterminismGate;
//!
//! let result = G4AdmissionDeterminismGate::compare(
//!     Path::new("live/wal/strategy_admission.jsonl"),
//!     Path::new("replay/wal/strategy_admission.jsonl"),
//! )?;
//!
//! if result.passed {
//!     println!("G4 PASSED: {} decisions match", result.matched_count);
//! } else {
//!     println!("G4 FAILED: {} mismatches", result.mismatches.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use quantlaxmi_models::{StrategyAdmissionDecision, StrategyAdmissionOutcome, StrategyRefuseReason};

use crate::CheckResult;

// =============================================================================
// Check Names (Frozen)
// =============================================================================

pub mod check_names {
    pub const G4_PARSE_LIVE: &str = "g4_parse_live";
    pub const G4_PARSE_REPLAY: &str = "g4_parse_replay";
    pub const G4_DECISION_PARITY: &str = "g4_decision_parity";
}

// =============================================================================
// G4DecisionKey — Composite key for matching decisions
// =============================================================================

/// Key for matching strategy admission decisions across WALs.
///
/// Tuple: (correlation_id, strategy_id, signal_id)
pub type G4DecisionKey = (String, String, String);

/// Extract key from a decision.
fn decision_key(d: &StrategyAdmissionDecision) -> G4DecisionKey {
    (
        d.correlation_id.trim().to_string(),
        d.strategy_id.clone(),
        d.signal_id.clone(),
    )
}

// =============================================================================
// G4MismatchKind — Why two decisions don't match
// =============================================================================

/// Mismatch types for G4 comparison (frozen 6-way taxonomy).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum G4MismatchKind {
    /// Decision exists in live but not in replay
    MissingInReplay,

    /// Decision exists in replay but not in live
    MissingInLive,

    /// Outcome differs (Admit vs Refuse)
    OutcomeMismatch {
        live_outcome: String,
        replay_outcome: String,
    },

    /// Refuse reasons differ (order matters)
    ReasonsMismatch {
        live_reasons: Vec<String>,
        replay_reasons: Vec<String>,
    },

    /// Digest differs (canonical bytes changed)
    DigestMismatch {
        live_digest: String,
        replay_digest: String,
    },
}

impl G4MismatchKind {
    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            G4MismatchKind::MissingInReplay => "Decision missing in replay WAL".to_string(),
            G4MismatchKind::MissingInLive => "Decision missing in live WAL".to_string(),
            G4MismatchKind::OutcomeMismatch {
                live_outcome,
                replay_outcome,
            } => format!(
                "Outcome mismatch: live={}, replay={}",
                live_outcome, replay_outcome
            ),
            G4MismatchKind::ReasonsMismatch {
                live_reasons,
                replay_reasons,
            } => format!(
                "Reasons mismatch: live={:?}, replay={:?}",
                live_reasons, replay_reasons
            ),
            G4MismatchKind::DigestMismatch {
                live_digest,
                replay_digest,
            } => format!(
                "Digest mismatch: live={}..., replay={}...",
                &live_digest[..16.min(live_digest.len())],
                &replay_digest[..16.min(replay_digest.len())]
            ),
        }
    }
}

// =============================================================================
// G4Mismatch — A single mismatch entry
// =============================================================================

/// A single mismatch between live and replay WALs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G4Mismatch {
    /// Key that identifies this decision
    pub correlation_id: String,
    pub strategy_id: String,
    pub signal_id: String,

    /// Type of mismatch
    pub kind: G4MismatchKind,
}

impl G4Mismatch {
    /// Create a new mismatch.
    pub fn new(key: &G4DecisionKey, kind: G4MismatchKind) -> Self {
        Self {
            correlation_id: key.0.clone(),
            strategy_id: key.1.clone(),
            signal_id: key.2.clone(),
            kind,
        }
    }

    /// Human-readable description including key.
    pub fn description(&self) -> String {
        format!(
            "[cid={}, strat={}, sig={}] {}",
            &self.correlation_id[..8.min(self.correlation_id.len())],
            self.strategy_id,
            self.signal_id,
            self.kind.description()
        )
    }
}

// =============================================================================
// G4Result — Gate result summary
// =============================================================================

/// Result of G4 admission determinism gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G4Result {
    /// Check name identifier
    pub check_name: String,

    /// Overall pass/fail
    pub passed: bool,

    /// Summary message
    pub message: String,

    /// Path to live WAL (for traceability)
    pub live_wal_path: String,

    /// Path to replay WAL (for traceability)
    pub replay_wal_path: String,

    /// Number of entries in live WAL
    pub live_entry_count: usize,

    /// Number of entries in replay WAL
    pub replay_entry_count: usize,

    /// Number of decisions that matched exactly
    pub matched_count: usize,

    /// Individual check results (parse + compare)
    pub checks: Vec<CheckResult>,

    /// List of mismatches (empty if passed)
    pub mismatches: Vec<G4Mismatch>,

    /// Parse errors encountered (non-fatal warnings)
    pub parse_errors: Vec<String>,

    /// Error message if gate failed to run
    pub error: Option<String>,
}

impl G4Result {
    /// Create a new result in error state.
    pub fn error(live_path: &str, replay_path: &str, error: impl Into<String>) -> Self {
        Self {
            check_name: check_names::G4_DECISION_PARITY.to_string(),
            passed: false,
            message: "G4 gate error".to_string(),
            live_wal_path: live_path.to_string(),
            replay_wal_path: replay_path.to_string(),
            live_entry_count: 0,
            replay_entry_count: 0,
            matched_count: 0,
            checks: vec![],
            mismatches: vec![],
            parse_errors: vec![],
            error: Some(error.into()),
        }
    }
}

// =============================================================================
// G4AdmissionDeterminismGate — The gate implementation
// =============================================================================

/// G4 Admission Determinism Gate.
///
/// Compares two strategy_admission.jsonl WALs to verify deterministic
/// admission decisions between live and replay runs.
pub struct G4AdmissionDeterminismGate;

impl G4AdmissionDeterminismGate {
    /// Compare two strategy admission WALs.
    ///
    /// Returns `G4Result` with pass/fail status and any mismatches.
    pub fn compare(live_wal: &Path, replay_wal: &Path) -> Result<G4Result, String> {
        let live_path_str = live_wal.display().to_string();
        let replay_path_str = replay_wal.display().to_string();

        let mut checks = Vec::new();
        let mut parse_errors = Vec::new();

        // Parse live WAL
        let (live_map, live_count, live_errors) = match Self::parse_wal(live_wal) {
            Ok(result) => result,
            Err(e) => {
                return Ok(G4Result::error(
                    &live_path_str,
                    &replay_path_str,
                    format!("Failed to parse live WAL: {}", e),
                ));
            }
        };

        checks.push(if live_errors.is_empty() {
            CheckResult::pass(
                check_names::G4_PARSE_LIVE,
                format!("Parsed {} entries from live WAL", live_count),
            )
        } else {
            CheckResult::fail(
                check_names::G4_PARSE_LIVE,
                format!(
                    "Parsed {} entries with {} errors",
                    live_count,
                    live_errors.len()
                ),
            )
        });
        parse_errors.extend(live_errors.into_iter().map(|e| format!("live: {}", e)));

        // Parse replay WAL
        let (replay_map, replay_count, replay_errors) = match Self::parse_wal(replay_wal) {
            Ok(result) => result,
            Err(e) => {
                return Ok(G4Result::error(
                    &live_path_str,
                    &replay_path_str,
                    format!("Failed to parse replay WAL: {}", e),
                ));
            }
        };

        checks.push(if replay_errors.is_empty() {
            CheckResult::pass(
                check_names::G4_PARSE_REPLAY,
                format!("Parsed {} entries from replay WAL", replay_count),
            )
        } else {
            CheckResult::fail(
                check_names::G4_PARSE_REPLAY,
                format!(
                    "Parsed {} entries with {} errors",
                    replay_count,
                    replay_errors.len()
                ),
            )
        });
        parse_errors.extend(replay_errors.into_iter().map(|e| format!("replay: {}", e)));

        // Compare decisions
        let (mismatches, matched_count) = Self::compare_maps(&live_map, &replay_map)?;

        let passed = mismatches.is_empty();
        let message = if passed {
            format!(
                "All {} decisions match between live and replay",
                matched_count
            )
        } else {
            format!(
                "{} mismatches found ({} matched)",
                mismatches.len(),
                matched_count
            )
        };

        checks.push(if passed {
            CheckResult::pass(check_names::G4_DECISION_PARITY, message.clone())
        } else {
            CheckResult::fail(check_names::G4_DECISION_PARITY, message.clone())
        });

        Ok(G4Result {
            check_name: check_names::G4_DECISION_PARITY.to_string(),
            passed,
            message,
            live_wal_path: live_path_str,
            replay_wal_path: replay_path_str,
            live_entry_count: live_count,
            replay_entry_count: replay_count,
            matched_count,
            checks,
            mismatches,
            parse_errors,
            error: None,
        })
    }

    /// Parse a WAL file into a map keyed by (correlation_id, strategy_id, signal_id).
    ///
    /// Returns (map, total_count, parse_errors).
    fn parse_wal(
        path: &Path,
    ) -> Result<(HashMap<G4DecisionKey, StrategyAdmissionDecision>, usize, Vec<String>), String>
    {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
        let reader = BufReader::new(file);

        let mut map = HashMap::new();
        let mut errors = Vec::new();
        let mut line_num = 0;

        for line in reader.lines() {
            line_num += 1;
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    errors.push(format!("Line {}: read error: {}", line_num, e));
                    continue;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<StrategyAdmissionDecision>(&line) {
                Ok(decision) => {
                    // Validate correlation_id is non-empty
                    let cid = decision.correlation_id.trim();
                    if cid.is_empty() {
                        errors.push(format!(
                            "Line {}: empty correlation_id for strategy={}, signal={}",
                            line_num, decision.strategy_id, decision.signal_id
                        ));
                        continue;
                    }

                    let key = decision_key(&decision);
                    map.insert(key, decision);
                }
                Err(e) => {
                    errors.push(format!("Line {}: parse error: {}", line_num, e));
                }
            }
        }

        Ok((map, line_num, errors))
    }

    /// Compare two maps and return (mismatches, matched_count).
    fn compare_maps(
        live: &HashMap<G4DecisionKey, StrategyAdmissionDecision>,
        replay: &HashMap<G4DecisionKey, StrategyAdmissionDecision>,
    ) -> Result<(Vec<G4Mismatch>, usize), String> {
        let mut mismatches = Vec::new();
        let mut matched = 0;

        // Check all keys in live
        for (key, live_decision) in live {
            match replay.get(key) {
                None => {
                    mismatches.push(G4Mismatch::new(key, G4MismatchKind::MissingInReplay));
                }
                Some(replay_decision) => {
                    if let Some(mismatch) = Self::compare_decisions(key, live_decision, replay_decision) {
                        mismatches.push(mismatch);
                    } else {
                        matched += 1;
                    }
                }
            }
        }

        // Check keys only in replay (missing in live)
        for key in replay.keys() {
            if !live.contains_key(key) {
                mismatches.push(G4Mismatch::new(key, G4MismatchKind::MissingInLive));
            }
        }

        Ok((mismatches, matched))
    }

    /// Compare two decisions with the same key.
    ///
    /// Returns Some(mismatch) if they differ, None if they match.
    /// Comparison order (frozen):
    /// 1. Outcome
    /// 2. Refuse reasons
    /// 3. Digest
    fn compare_decisions(
        key: &G4DecisionKey,
        live: &StrategyAdmissionDecision,
        replay: &StrategyAdmissionDecision,
    ) -> Option<G4Mismatch> {
        // 1. Outcome mismatch
        if live.outcome != replay.outcome {
            return Some(G4Mismatch::new(
                key,
                G4MismatchKind::OutcomeMismatch {
                    live_outcome: outcome_to_string(live.outcome),
                    replay_outcome: outcome_to_string(replay.outcome),
                },
            ));
        }

        // 2. Refuse reasons mismatch (exact order matters)
        if live.refuse_reasons != replay.refuse_reasons {
            return Some(G4Mismatch::new(
                key,
                G4MismatchKind::ReasonsMismatch {
                    live_reasons: live.refuse_reasons.iter().map(reason_to_string).collect(),
                    replay_reasons: replay.refuse_reasons.iter().map(reason_to_string).collect(),
                },
            ));
        }

        // 3. Digest mismatch
        if live.digest != replay.digest {
            return Some(G4Mismatch::new(
                key,
                G4MismatchKind::DigestMismatch {
                    live_digest: live.digest.clone(),
                    replay_digest: replay.digest.clone(),
                },
            ));
        }

        // Match
        None
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn outcome_to_string(outcome: StrategyAdmissionOutcome) -> String {
    match outcome {
        StrategyAdmissionOutcome::Admit => "admit".to_string(),
        StrategyAdmissionOutcome::Refuse => "refuse".to_string(),
    }
}

fn reason_to_string(reason: &StrategyRefuseReason) -> String {
    match reason {
        StrategyRefuseReason::SignalNotAdmitted { signal_id } => {
            format!("signal_not_admitted:{}", signal_id)
        }
        StrategyRefuseReason::StrategyNotFound { strategy_id } => {
            format!("strategy_not_found:{}", strategy_id)
        }
        StrategyRefuseReason::SignalNotBound {
            signal_id,
            strategy_id,
        } => {
            format!("signal_not_bound:{}:{}", signal_id, strategy_id)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_decision(
        correlation_id: &str,
        strategy_id: &str,
        signal_id: &str,
        outcome: StrategyAdmissionOutcome,
        refuse_reasons: Vec<StrategyRefuseReason>,
    ) -> StrategyAdmissionDecision {
        let builder = StrategyAdmissionDecision::builder(strategy_id, signal_id)
            .correlation_id(correlation_id)
            .session_id("test_session")
            .ts_ns(1234567890)
            .strategies_manifest_hash([0u8; 32])
            .signals_manifest_hash([0u8; 32]);

        match outcome {
            StrategyAdmissionOutcome::Admit => builder.build_admit(),
            StrategyAdmissionOutcome::Refuse => builder.build_refuse(refuse_reasons),
        }
    }

    fn write_wal(dir: &TempDir, name: &str, decisions: &[StrategyAdmissionDecision]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut file = File::create(&path).unwrap();
        for d in decisions {
            writeln!(file, "{}", serde_json::to_string(d).unwrap()).unwrap();
        }
        path
    }

    // =========================================================================
    // Test: Identical WALs pass
    // =========================================================================

    #[test]
    fn test_identical_wals_pass() {
        let dir = TempDir::new().unwrap();

        let decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Admit, vec![]),
            make_decision("cid2", "strat1", "sig2", StrategyAdmissionOutcome::Refuse, vec![
                StrategyRefuseReason::SignalNotAdmitted { signal_id: "sig2".to_string() }
            ]),
        ];

        let live_path = write_wal(&dir, "live.jsonl", &decisions);
        let replay_path = write_wal(&dir, "replay.jsonl", &decisions);

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(result.passed);
        assert_eq!(result.matched_count, 2);
        assert!(result.mismatches.is_empty());
        assert_eq!(result.live_entry_count, 2);
        assert_eq!(result.replay_entry_count, 2);
    }

    // =========================================================================
    // Test: Missing in replay
    // =========================================================================

    #[test]
    fn test_missing_in_replay() {
        let dir = TempDir::new().unwrap();

        let live_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Admit, vec![]),
            make_decision("cid2", "strat1", "sig2", StrategyAdmissionOutcome::Admit, vec![]),
        ];
        let replay_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Admit, vec![]),
            // cid2 missing
        ];

        let live_path = write_wal(&dir, "live.jsonl", &live_decisions);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_decisions);

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.matched_count, 1);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G4MismatchKind::MissingInReplay
        ));
    }

    // =========================================================================
    // Test: Missing in live
    // =========================================================================

    #[test]
    fn test_missing_in_live() {
        let dir = TempDir::new().unwrap();

        let live_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Admit, vec![]),
        ];
        let replay_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Admit, vec![]),
            make_decision("cid2", "strat1", "sig2", StrategyAdmissionOutcome::Admit, vec![]),
        ];

        let live_path = write_wal(&dir, "live.jsonl", &live_decisions);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_decisions);

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.matched_count, 1);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G4MismatchKind::MissingInLive
        ));
    }

    // =========================================================================
    // Test: Outcome mismatch
    // =========================================================================

    #[test]
    fn test_outcome_mismatch() {
        let dir = TempDir::new().unwrap();

        let live_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Admit, vec![]),
        ];
        let replay_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Refuse, vec![
                StrategyRefuseReason::SignalNotAdmitted { signal_id: "sig1".to_string() }
            ]),
        ];

        let live_path = write_wal(&dir, "live.jsonl", &live_decisions);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_decisions);

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        match &result.mismatches[0].kind {
            G4MismatchKind::OutcomeMismatch { live_outcome, replay_outcome } => {
                assert_eq!(live_outcome, "admit");
                assert_eq!(replay_outcome, "refuse");
            }
            _ => panic!("Expected OutcomeMismatch"),
        }
    }

    // =========================================================================
    // Test: Reasons mismatch
    // =========================================================================

    #[test]
    fn test_reasons_mismatch() {
        let dir = TempDir::new().unwrap();

        let live_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Refuse, vec![
                StrategyRefuseReason::SignalNotAdmitted { signal_id: "sig1".to_string() }
            ]),
        ];
        let replay_decisions = vec![
            make_decision("cid1", "strat1", "sig1", StrategyAdmissionOutcome::Refuse, vec![
                StrategyRefuseReason::StrategyNotFound { strategy_id: "strat1".to_string() }
            ]),
        ];

        let live_path = write_wal(&dir, "live.jsonl", &live_decisions);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_decisions);

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G4MismatchKind::ReasonsMismatch { .. }
        ));
    }

    // =========================================================================
    // Test: Digest mismatch (different manifest hashes → different digest)
    // =========================================================================

    #[test]
    fn test_digest_mismatch() {
        let dir = TempDir::new().unwrap();

        // Create decisions manually to get different digests
        let live_decision = StrategyAdmissionDecision::builder("strat1", "sig1")
            .correlation_id("cid1")
            .session_id("test_session")
            .ts_ns(1234567890)
            .strategies_manifest_hash([1u8; 32]) // Different hash
            .signals_manifest_hash([0u8; 32])
            .build_admit();

        let replay_decision = StrategyAdmissionDecision::builder("strat1", "sig1")
            .correlation_id("cid1")
            .session_id("test_session")
            .ts_ns(1234567890)
            .strategies_manifest_hash([2u8; 32]) // Different hash
            .signals_manifest_hash([0u8; 32])
            .build_admit();

        // Digests should differ because manifest hashes differ
        assert_ne!(live_decision.digest, replay_decision.digest);

        let live_path = write_wal(&dir, "live.jsonl", &[live_decision]);
        let replay_path = write_wal(&dir, "replay.jsonl", &[replay_decision]);

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G4MismatchKind::DigestMismatch { .. }
        ));
    }

    // =========================================================================
    // Test: Empty correlation_id is rejected
    // =========================================================================

    #[test]
    fn test_empty_correlation_id_rejected() {
        let dir = TempDir::new().unwrap();

        // Write raw JSON with empty correlation_id
        let live_path = dir.path().join("live.jsonl");
        let mut file = File::create(&live_path).unwrap();
        writeln!(file, r#"{{"schema_version":"1.0.0","ts_ns":0,"session_id":"s","strategy_id":"strat","signal_id":"sig","outcome":"admit","refuse_reasons":[],"correlation_id":"","strategies_manifest_hash":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"signals_manifest_hash":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"digest":"abc"}}"#).unwrap();

        let replay_path = dir.path().join("replay.jsonl");
        File::create(&replay_path).unwrap(); // Empty file

        let result = G4AdmissionDeterminismGate::compare(&live_path, &replay_path).unwrap();

        // Should have parse error for empty correlation_id
        assert!(!result.parse_errors.is_empty());
        assert!(result.parse_errors[0].contains("empty correlation_id"));
    }

    // =========================================================================
    // Test: Check names are correct
    // =========================================================================

    #[test]
    fn test_check_names() {
        assert_eq!(check_names::G4_PARSE_LIVE, "g4_parse_live");
        assert_eq!(check_names::G4_PARSE_REPLAY, "g4_parse_replay");
        assert_eq!(check_names::G4_DECISION_PARITY, "g4_decision_parity");
    }

    // =========================================================================
    // Test: Mismatch description formatting
    // =========================================================================

    #[test]
    fn test_mismatch_descriptions() {
        let key = ("correlation123".to_string(), "strategy1".to_string(), "signal1".to_string());

        let m1 = G4Mismatch::new(&key, G4MismatchKind::MissingInReplay);
        assert!(m1.description().contains("missing in replay"));

        let m2 = G4Mismatch::new(&key, G4MismatchKind::MissingInLive);
        assert!(m2.description().contains("missing in live"));

        let m3 = G4Mismatch::new(&key, G4MismatchKind::OutcomeMismatch {
            live_outcome: "admit".to_string(),
            replay_outcome: "refuse".to_string(),
        });
        assert!(m3.description().contains("admit"));
        assert!(m3.description().contains("refuse"));
    }
}
