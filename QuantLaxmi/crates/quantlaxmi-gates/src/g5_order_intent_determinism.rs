//! G5 Order Intent Determinism Gate — Order intent WAL parity check.
//!
//! Phase 23C: Compares two `order_intent.jsonl` WALs (live vs replay)
//! to verify deterministic order intent emission and permission outcomes.
//!
//! ## Key Structure
//! Decisions are keyed by: `(session_id, seq)`
//!
//! ## Comparison Rules (Frozen v1)
//! For each key present in either WAL:
//! 1. If missing in replay → `MissingInReplay`
//! 2. If missing in live → `MissingInLive`
//! 3. If permission differs → `PermissionMismatch`
//! 4. If both refuse and reason differs → `RefuseReasonMismatch`
//! 5. If payload fields differ → `PayloadMismatch`
//! 6. If digest differs → `DigestMismatch`
//! 7. Else → Match (counted, not stored)
//!
//! ## Payload Fields Compared (Frozen v1)
//! - strategy_id
//! - symbol
//! - side
//! - order_type
//! - qty_mantissa
//! - qty_exponent
//! - limit_price_mantissa
//! - price_exponent
//! - correlation_id
//! - parent_admission_digest
//!
//! Note: ts_ns is NOT compared in v1 to avoid false failures from timestamp semantics.
//!
//! ## Usage
//! ```ignore
//! use quantlaxmi_gates::g5_order_intent_determinism::G5OrderIntentDeterminismGate;
//!
//! let result = G5OrderIntentDeterminismGate::compare(
//!     Path::new("live/wal/order_intent.jsonl"),
//!     Path::new("replay/wal/order_intent.jsonl"),
//! )?;
//!
//! if result.passed {
//!     println!("G5 PASSED: {} intents match", result.matched_count);
//! } else {
//!     println!("G5 FAILED: {} mismatches", result.mismatches.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use quantlaxmi_models::{OrderIntentPermission, OrderIntentRecord, OrderRefuseReason};

use crate::CheckResult;

// =============================================================================
// Check Names (Frozen)
// =============================================================================

pub mod check_names {
    pub const G5_PARSE_LIVE: &str = "g5_parse_live";
    pub const G5_PARSE_REPLAY: &str = "g5_parse_replay";
    pub const G5_ORDER_INTENT_PARITY: &str = "g5_order_intent_parity";
}

// =============================================================================
// G5OrderIntentKey — Composite key for matching intents
// =============================================================================

/// Key for matching order intents across WALs.
///
/// Tuple: (session_id, seq)
pub type G5OrderIntentKey = (String, u64);

/// Extract key from an intent record.
fn intent_key(r: &OrderIntentRecord) -> G5OrderIntentKey {
    (r.session_id.clone(), r.seq)
}

// =============================================================================
// G5MismatchKind — Why two intents don't match
// =============================================================================

/// Mismatch types for G5 comparison (frozen 6-way taxonomy).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum G5MismatchKind {
    /// Intent exists in live but not in replay
    MissingInReplay,

    /// Intent exists in replay but not in live
    MissingInLive,

    /// Permission outcome differs (Permit vs Refuse)
    PermissionMismatch {
        live_permission: String,
        replay_permission: String,
    },

    /// Both refuse but reason differs
    RefuseReasonMismatch {
        live_reason: String,
        replay_reason: String,
    },

    /// Core payload field differs
    PayloadMismatch {
        field: String,
        live_value: serde_json::Value,
        replay_value: serde_json::Value,
    },

    /// Digest differs (canonical bytes changed)
    DigestMismatch {
        live_digest: String,
        replay_digest: String,
    },
}

impl G5MismatchKind {
    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            G5MismatchKind::MissingInReplay => "Intent missing in replay WAL".to_string(),
            G5MismatchKind::MissingInLive => "Intent missing in live WAL".to_string(),
            G5MismatchKind::PermissionMismatch {
                live_permission,
                replay_permission,
            } => format!(
                "Permission mismatch: live={}, replay={}",
                live_permission, replay_permission
            ),
            G5MismatchKind::RefuseReasonMismatch {
                live_reason,
                replay_reason,
            } => format!(
                "Refuse reason mismatch: live={}, replay={}",
                truncate(live_reason, 40),
                truncate(replay_reason, 40)
            ),
            G5MismatchKind::PayloadMismatch {
                field,
                live_value,
                replay_value,
            } => format!(
                "Payload mismatch: field={}, live={}, replay={}",
                field, live_value, replay_value
            ),
            G5MismatchKind::DigestMismatch {
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

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

// =============================================================================
// G5Mismatch — A single mismatch entry
// =============================================================================

/// A single mismatch between live and replay WALs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G5Mismatch {
    /// Session ID component of the key
    pub session_id: String,
    /// Sequence number component of the key
    pub seq: u64,
    /// Type of mismatch
    pub kind: G5MismatchKind,
}

impl G5Mismatch {
    /// Create a new mismatch.
    pub fn new(key: &G5OrderIntentKey, kind: G5MismatchKind) -> Self {
        Self {
            session_id: key.0.clone(),
            seq: key.1,
            kind,
        }
    }

    /// Human-readable description including key.
    pub fn description(&self) -> String {
        format!(
            "[session={}, seq={}] {}",
            truncate(&self.session_id, 16),
            self.seq,
            self.kind.description()
        )
    }
}

// =============================================================================
// G5Result — Gate result summary
// =============================================================================

/// Result of G5 order intent determinism gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G5Result {
    /// Overall pass/fail
    pub passed: bool,

    /// Path to live WAL (for traceability)
    pub live_wal_path: String,

    /// Path to replay WAL (for traceability)
    pub replay_wal_path: String,

    /// Number of entries in live WAL
    pub live_entry_count: usize,

    /// Number of entries in replay WAL
    pub replay_entry_count: usize,

    /// Number of intents that matched exactly
    pub matched_count: usize,

    /// Individual check results (parse + compare)
    pub checks: Vec<CheckResult>,

    /// List of mismatches (empty if passed)
    pub mismatches: Vec<G5Mismatch>,

    /// Error message if gate failed to run
    pub error: Option<String>,
}

impl G5Result {
    /// Create a new result in error state.
    pub fn error(live_path: &str, replay_path: &str, error: impl Into<String>) -> Self {
        Self {
            passed: false,
            live_wal_path: live_path.to_string(),
            replay_wal_path: replay_path.to_string(),
            live_entry_count: 0,
            replay_entry_count: 0,
            matched_count: 0,
            checks: vec![],
            mismatches: vec![],
            error: Some(error.into()),
        }
    }

    /// Summary message for display.
    pub fn message(&self) -> String {
        if let Some(ref err) = self.error {
            format!("G5 gate error: {}", err)
        } else if self.passed {
            format!(
                "All {} intents match between live and replay",
                self.matched_count
            )
        } else {
            format!(
                "{} mismatches found ({} matched)",
                self.mismatches.len(),
                self.matched_count
            )
        }
    }
}

// =============================================================================
// G5OrderIntentDeterminismGate — The gate implementation
// =============================================================================

/// G5 Order Intent Determinism Gate.
///
/// Compares two order_intent.jsonl WALs to verify deterministic
/// order intent emission between live and replay runs.
pub struct G5OrderIntentDeterminismGate;

impl G5OrderIntentDeterminismGate {
    /// Compare two order intent WALs.
    ///
    /// Returns `G5Result` with pass/fail status and any mismatches.
    ///
    /// # Errors
    /// Returns error string if:
    /// - Files cannot be opened
    /// - JSON parse errors occur (hard failure per spec)
    /// - Duplicate keys detected in same WAL
    pub fn compare(live_wal: &Path, replay_wal: &Path) -> Result<G5Result, String> {
        let live_path_str = live_wal.display().to_string();
        let replay_path_str = replay_wal.display().to_string();

        let mut checks = Vec::new();

        // Parse live WAL
        let (live_map, live_count) = match Self::parse_wal(live_wal) {
            Ok(result) => result,
            Err(e) => {
                return Err(format!("Failed to parse live WAL: {}", e));
            }
        };

        checks.push(CheckResult::pass(
            check_names::G5_PARSE_LIVE,
            format!("Parsed {} entries from live WAL", live_count),
        ));

        // Parse replay WAL
        let (replay_map, replay_count) = match Self::parse_wal(replay_wal) {
            Ok(result) => result,
            Err(e) => {
                return Err(format!("Failed to parse replay WAL: {}", e));
            }
        };

        checks.push(CheckResult::pass(
            check_names::G5_PARSE_REPLAY,
            format!("Parsed {} entries from replay WAL", replay_count),
        ));

        // Compare intents
        let (mismatches, matched_count) = Self::compare_maps(&live_map, &replay_map);

        let passed = mismatches.is_empty();
        let message = if passed {
            format!(
                "All {} intents match between live and replay",
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
            CheckResult::pass(check_names::G5_ORDER_INTENT_PARITY, message)
        } else {
            CheckResult::fail(check_names::G5_ORDER_INTENT_PARITY, message)
        });

        Ok(G5Result {
            passed,
            live_wal_path: live_path_str,
            replay_wal_path: replay_path_str,
            live_entry_count: live_count,
            replay_entry_count: replay_count,
            matched_count,
            checks,
            mismatches,
            error: None,
        })
    }

    /// Parse a WAL file into a map keyed by (session_id, seq).
    ///
    /// Returns (map, total_count).
    ///
    /// # Errors
    /// - File cannot be opened
    /// - JSON parse error on any line
    /// - Empty session_id
    /// - Duplicate key (same session_id + seq)
    fn parse_wal(
        path: &Path,
    ) -> Result<(HashMap<G5OrderIntentKey, OrderIntentRecord>, usize), String> {
        let file =
            File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
        let reader = BufReader::new(file);

        let mut map = HashMap::new();
        let mut line_num = 0;

        for line in reader.lines() {
            line_num += 1;
            let line = line.map_err(|e| format!("Line {}: read error: {}", line_num, e))?;

            if line.trim().is_empty() {
                continue;
            }

            let record: OrderIntentRecord = serde_json::from_str(&line)
                .map_err(|e| format!("Line {}: parse error: {}", line_num, e))?;

            // Validate session_id is non-empty
            if record.session_id.trim().is_empty() {
                return Err(format!(
                    "Line {}: empty session_id for seq={}",
                    line_num, record.seq
                ));
            }

            let key = intent_key(&record);

            // Check for duplicate key (hard error per spec)
            if map.contains_key(&key) {
                return Err(format!(
                    "Line {}: duplicate key (session_id={}, seq={}) - broken seq monotonicity",
                    line_num, key.0, key.1
                ));
            }

            map.insert(key, record);
        }

        let count = map.len();
        Ok((map, count))
    }

    /// Compare two maps and return (mismatches, matched_count).
    fn compare_maps(
        live: &HashMap<G5OrderIntentKey, OrderIntentRecord>,
        replay: &HashMap<G5OrderIntentKey, OrderIntentRecord>,
    ) -> (Vec<G5Mismatch>, usize) {
        let mut mismatches = Vec::new();
        let mut matched = 0;

        // Check all keys in live
        for (key, live_record) in live {
            match replay.get(key) {
                None => {
                    mismatches.push(G5Mismatch::new(key, G5MismatchKind::MissingInReplay));
                }
                Some(replay_record) => {
                    if let Some(mismatch) = Self::compare_records(key, live_record, replay_record) {
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
                mismatches.push(G5Mismatch::new(key, G5MismatchKind::MissingInLive));
            }
        }

        (mismatches, matched)
    }

    /// Compare two records with the same key.
    ///
    /// Returns Some(mismatch) if they differ, None if they match.
    /// Comparison order (frozen v1):
    /// 1. Permission
    /// 2. Refuse reason (if both refuse)
    /// 3. Payload fields
    /// 4. Digest
    fn compare_records(
        key: &G5OrderIntentKey,
        live: &OrderIntentRecord,
        replay: &OrderIntentRecord,
    ) -> Option<G5Mismatch> {
        // 1. Permission mismatch
        if live.permission != replay.permission {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PermissionMismatch {
                    live_permission: permission_to_string(&live.permission),
                    replay_permission: permission_to_string(&replay.permission),
                },
            ));
        }

        // 2. Refuse reason mismatch (only if both refuse)
        if live.permission == OrderIntentPermission::Refuse {
            let live_reason = refuse_reason_to_string(&live.refuse_reason);
            let replay_reason = refuse_reason_to_string(&replay.refuse_reason);
            if live_reason != replay_reason {
                return Some(G5Mismatch::new(
                    key,
                    G5MismatchKind::RefuseReasonMismatch {
                        live_reason,
                        replay_reason,
                    },
                ));
            }
        }

        // 3. Payload fields (frozen order)
        // strategy_id
        if live.strategy_id != replay.strategy_id {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "strategy_id".to_string(),
                    live_value: serde_json::Value::String(live.strategy_id.clone()),
                    replay_value: serde_json::Value::String(replay.strategy_id.clone()),
                },
            ));
        }

        // symbol
        if live.symbol != replay.symbol {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "symbol".to_string(),
                    live_value: serde_json::Value::String(live.symbol.clone()),
                    replay_value: serde_json::Value::String(replay.symbol.clone()),
                },
            ));
        }

        // side
        if live.side != replay.side {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "side".to_string(),
                    live_value: serde_json::json!(live.side),
                    replay_value: serde_json::json!(replay.side),
                },
            ));
        }

        // order_type
        if live.order_type != replay.order_type {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "order_type".to_string(),
                    live_value: serde_json::json!(live.order_type),
                    replay_value: serde_json::json!(replay.order_type),
                },
            ));
        }

        // qty_mantissa
        if live.qty_mantissa != replay.qty_mantissa {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "qty_mantissa".to_string(),
                    live_value: serde_json::Value::Number(live.qty_mantissa.into()),
                    replay_value: serde_json::Value::Number(replay.qty_mantissa.into()),
                },
            ));
        }

        // qty_exponent
        if live.qty_exponent != replay.qty_exponent {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "qty_exponent".to_string(),
                    live_value: serde_json::Value::Number(live.qty_exponent.into()),
                    replay_value: serde_json::Value::Number(replay.qty_exponent.into()),
                },
            ));
        }

        // limit_price_mantissa
        if live.limit_price_mantissa != replay.limit_price_mantissa {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "limit_price_mantissa".to_string(),
                    live_value: option_i64_to_json(&live.limit_price_mantissa),
                    replay_value: option_i64_to_json(&replay.limit_price_mantissa),
                },
            ));
        }

        // price_exponent
        if live.price_exponent != replay.price_exponent {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "price_exponent".to_string(),
                    live_value: serde_json::Value::Number(live.price_exponent.into()),
                    replay_value: serde_json::Value::Number(replay.price_exponent.into()),
                },
            ));
        }

        // correlation_id
        if live.correlation_id != replay.correlation_id {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "correlation_id".to_string(),
                    live_value: serde_json::Value::String(live.correlation_id.clone()),
                    replay_value: serde_json::Value::String(replay.correlation_id.clone()),
                },
            ));
        }

        // parent_admission_digest
        if live.parent_admission_digest != replay.parent_admission_digest {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::PayloadMismatch {
                    field: "parent_admission_digest".to_string(),
                    live_value: serde_json::Value::String(live.parent_admission_digest.clone()),
                    replay_value: serde_json::Value::String(replay.parent_admission_digest.clone()),
                },
            ));
        }

        // 4. Digest mismatch
        if live.digest != replay.digest {
            return Some(G5Mismatch::new(
                key,
                G5MismatchKind::DigestMismatch {
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

fn permission_to_string(permission: &OrderIntentPermission) -> String {
    match permission {
        OrderIntentPermission::Permit => "permit".to_string(),
        OrderIntentPermission::Refuse => "refuse".to_string(),
    }
}

fn refuse_reason_to_string(reason: &Option<OrderRefuseReason>) -> String {
    match reason {
        None => "none".to_string(),
        Some(r) => r.description(),
    }
}

fn option_i64_to_json(opt: &Option<i64>) -> serde_json::Value {
    match opt {
        Some(v) => serde_json::Value::Number((*v).into()),
        None => serde_json::Value::Null,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::{OrderIntentRecord, OrderIntentSide, OrderIntentType, OrderRefuseReason};
    use std::io::Write;
    use tempfile::TempDir;

    fn make_permit_record(session_id: &str, seq: u64, strategy_id: &str) -> OrderIntentRecord {
        OrderIntentRecord::builder(strategy_id, "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id(session_id)
            .seq(seq)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Limit)
            .qty(100000000, -8)
            .limit_price(5000000, -2)
            .correlation_id(&format!("corr_{}", seq))
            .parent_admission_digest("test_digest")
            .build_permit()
    }

    fn make_refuse_record(
        session_id: &str,
        seq: u64,
        strategy_id: &str,
        reason: OrderRefuseReason,
    ) -> OrderIntentRecord {
        OrderIntentRecord::builder(strategy_id, "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id(session_id)
            .seq(seq)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Market)
            .qty(100000000, -8)
            .price_exponent(-2)
            .correlation_id(&format!("corr_{}", seq))
            .parent_admission_digest("test_digest")
            .build_refuse(reason)
    }

    fn write_wal(dir: &TempDir, name: &str, records: &[OrderIntentRecord]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut file = File::create(&path).unwrap();
        for r in records {
            writeln!(file, "{}", serde_json::to_string(r).unwrap()).unwrap();
        }
        path
    }

    // =========================================================================
    // Test: Identical WALs pass
    // =========================================================================

    #[test]
    fn test_g5_identical_wals_pass() {
        let dir = TempDir::new().unwrap();

        let records = vec![
            make_permit_record("sess1", 1, "strat1"),
            make_permit_record("sess1", 2, "strat1"),
            make_refuse_record(
                "sess1",
                3,
                "strat1",
                OrderRefuseReason::Custom {
                    reason: "test reason".to_string(),
                },
            ),
        ];

        let live_path = write_wal(&dir, "live.jsonl", &records);
        let replay_path = write_wal(&dir, "replay.jsonl", &records);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(result.passed);
        assert_eq!(result.matched_count, 3);
        assert!(result.mismatches.is_empty());
        assert_eq!(result.live_entry_count, 3);
        assert_eq!(result.replay_entry_count, 3);
    }

    // =========================================================================
    // Test: Missing in replay
    // =========================================================================

    #[test]
    fn test_g5_missing_in_replay() {
        let dir = TempDir::new().unwrap();

        let live_records = vec![
            make_permit_record("sess1", 1, "strat1"),
            make_permit_record("sess1", 2, "strat1"),
        ];
        let replay_records = vec![make_permit_record("sess1", 1, "strat1")];

        let live_path = write_wal(&dir, "live.jsonl", &live_records);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_records);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.matched_count, 1);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G5MismatchKind::MissingInReplay
        ));
        assert_eq!(result.mismatches[0].seq, 2);
    }

    // =========================================================================
    // Test: Missing in live
    // =========================================================================

    #[test]
    fn test_g5_missing_in_live() {
        let dir = TempDir::new().unwrap();

        let live_records = vec![make_permit_record("sess1", 1, "strat1")];
        let replay_records = vec![
            make_permit_record("sess1", 1, "strat1"),
            make_permit_record("sess1", 2, "strat1"),
        ];

        let live_path = write_wal(&dir, "live.jsonl", &live_records);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_records);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.matched_count, 1);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G5MismatchKind::MissingInLive
        ));
    }

    // =========================================================================
    // Test: Permission mismatch
    // =========================================================================

    #[test]
    fn test_g5_permission_mismatch() {
        let dir = TempDir::new().unwrap();

        let live_records = vec![make_permit_record("sess1", 1, "strat1")];
        let replay_records = vec![make_refuse_record(
            "sess1",
            1,
            "strat1",
            OrderRefuseReason::Custom {
                reason: "test".to_string(),
            },
        )];

        let live_path = write_wal(&dir, "live.jsonl", &live_records);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_records);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        match &result.mismatches[0].kind {
            G5MismatchKind::PermissionMismatch {
                live_permission,
                replay_permission,
            } => {
                assert_eq!(live_permission, "permit");
                assert_eq!(replay_permission, "refuse");
            }
            _ => panic!("Expected PermissionMismatch"),
        }
    }

    // =========================================================================
    // Test: Refuse reason mismatch
    // =========================================================================

    #[test]
    fn test_g5_refuse_reason_mismatch() {
        let dir = TempDir::new().unwrap();

        let live_records = vec![make_refuse_record(
            "sess1",
            1,
            "strat1",
            OrderRefuseReason::Custom {
                reason: "reason A".to_string(),
            },
        )];
        let replay_records = vec![make_refuse_record(
            "sess1",
            1,
            "strat1",
            OrderRefuseReason::Custom {
                reason: "reason B".to_string(),
            },
        )];

        let live_path = write_wal(&dir, "live.jsonl", &live_records);
        let replay_path = write_wal(&dir, "replay.jsonl", &replay_records);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G5MismatchKind::RefuseReasonMismatch { .. }
        ));
    }

    // =========================================================================
    // Test: Payload mismatch - qty_mantissa
    // =========================================================================

    #[test]
    fn test_g5_payload_mismatch_qty_mantissa() {
        let dir = TempDir::new().unwrap();

        let live = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(100, -8)
            .correlation_id("corr")
            .parent_admission_digest("digest")
            .build_permit();

        let replay = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(200, -8) // Different qty!
            .correlation_id("corr")
            .parent_admission_digest("digest")
            .build_permit();

        let live_path = write_wal(&dir, "live.jsonl", &[live]);
        let replay_path = write_wal(&dir, "replay.jsonl", &[replay]);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        match &result.mismatches[0].kind {
            G5MismatchKind::PayloadMismatch { field, .. } => {
                assert_eq!(field, "qty_mantissa");
            }
            _ => panic!("Expected PayloadMismatch"),
        }
    }

    // =========================================================================
    // Test: Payload mismatch - side
    // =========================================================================

    #[test]
    fn test_g5_payload_mismatch_side() {
        let dir = TempDir::new().unwrap();

        let live = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(100, -8)
            .correlation_id("corr")
            .parent_admission_digest("digest")
            .build_permit();

        let replay = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Sell) // Different side!
            .qty(100, -8)
            .correlation_id("corr")
            .parent_admission_digest("digest")
            .build_permit();

        let live_path = write_wal(&dir, "live.jsonl", &[live]);
        let replay_path = write_wal(&dir, "replay.jsonl", &[replay]);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        match &result.mismatches[0].kind {
            G5MismatchKind::PayloadMismatch { field, .. } => {
                assert_eq!(field, "side");
            }
            _ => panic!("Expected PayloadMismatch"),
        }
    }

    // =========================================================================
    // Test: Payload mismatch - order_type
    // =========================================================================

    #[test]
    fn test_g5_payload_mismatch_order_type() {
        let dir = TempDir::new().unwrap();

        let live = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Market)
            .qty(100, -8)
            .correlation_id("corr")
            .parent_admission_digest("digest")
            .build_permit();

        let replay = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Limit) // Different type!
            .qty(100, -8)
            .limit_price(5000, -2)
            .correlation_id("corr")
            .parent_admission_digest("digest")
            .build_permit();

        let live_path = write_wal(&dir, "live.jsonl", &[live]);
        let replay_path = write_wal(&dir, "replay.jsonl", &[replay]);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        match &result.mismatches[0].kind {
            G5MismatchKind::PayloadMismatch { field, .. } => {
                assert_eq!(field, "order_type");
            }
            _ => panic!("Expected PayloadMismatch"),
        }
    }

    // =========================================================================
    // Test: Digest mismatch (same payload but different correlation_id)
    // =========================================================================

    #[test]
    fn test_g5_digest_mismatch() {
        let dir = TempDir::new().unwrap();

        // Create two records that will have different digests due to correlation_id
        let live = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(100, -8)
            .correlation_id("corr_live")
            .parent_admission_digest("digest")
            .build_permit();

        let replay = OrderIntentRecord::builder("strat1", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess1")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(100, -8)
            .correlation_id("corr_replay") // Different → will cause PayloadMismatch first
            .parent_admission_digest("digest")
            .build_permit();

        // This test actually catches PayloadMismatch on correlation_id
        // because that's compared before digest
        let live_path = write_wal(&dir, "live.jsonl", &[live]);
        let replay_path = write_wal(&dir, "replay.jsonl", &[replay]);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(!result.passed);
        // Should be PayloadMismatch on correlation_id, not DigestMismatch
        match &result.mismatches[0].kind {
            G5MismatchKind::PayloadMismatch { field, .. } => {
                assert_eq!(field, "correlation_id");
            }
            _ => panic!("Expected PayloadMismatch on correlation_id"),
        }
    }

    // =========================================================================
    // Test: Empty WALs pass
    // =========================================================================

    #[test]
    fn test_g5_empty_wals_pass() {
        let dir = TempDir::new().unwrap();

        let live_path = write_wal(&dir, "live.jsonl", &[]);
        let replay_path = write_wal(&dir, "replay.jsonl", &[]);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path).unwrap();

        assert!(result.passed);
        assert_eq!(result.matched_count, 0);
        assert_eq!(result.live_entry_count, 0);
        assert_eq!(result.replay_entry_count, 0);
    }

    // =========================================================================
    // Test: Duplicate key is error
    // =========================================================================

    #[test]
    fn test_g5_duplicate_key_is_error() {
        let dir = TempDir::new().unwrap();

        // Write WAL with duplicate key manually
        let live_path = dir.path().join("live.jsonl");
        let record = make_permit_record("sess1", 1, "strat1");
        let mut file = File::create(&live_path).unwrap();
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        writeln!(file, "{}", serde_json::to_string(&record).unwrap()).unwrap(); // Duplicate!

        let replay_path = write_wal(&dir, "replay.jsonl", &[]);

        let result = G5OrderIntentDeterminismGate::compare(&live_path, &replay_path);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("duplicate key"));
        assert!(err.contains("broken seq monotonicity"));
    }

    // =========================================================================
    // Test: Check names are correct
    // =========================================================================

    #[test]
    fn test_g5_check_names() {
        assert_eq!(check_names::G5_PARSE_LIVE, "g5_parse_live");
        assert_eq!(check_names::G5_PARSE_REPLAY, "g5_parse_replay");
        assert_eq!(check_names::G5_ORDER_INTENT_PARITY, "g5_order_intent_parity");
    }

    // =========================================================================
    // Test: Mismatch description formatting
    // =========================================================================

    #[test]
    fn test_g5_mismatch_descriptions() {
        let key = ("session123".to_string(), 42u64);

        let m1 = G5Mismatch::new(&key, G5MismatchKind::MissingInReplay);
        assert!(m1.description().contains("missing in replay"));
        assert!(m1.description().contains("seq=42"));

        let m2 = G5Mismatch::new(&key, G5MismatchKind::MissingInLive);
        assert!(m2.description().contains("missing in live"));

        let m3 = G5Mismatch::new(
            &key,
            G5MismatchKind::PermissionMismatch {
                live_permission: "permit".to_string(),
                replay_permission: "refuse".to_string(),
            },
        );
        assert!(m3.description().contains("permit"));
        assert!(m3.description().contains("refuse"));
    }
}
