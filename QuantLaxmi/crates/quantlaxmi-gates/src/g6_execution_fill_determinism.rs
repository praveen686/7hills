//! G6 Execution Fill Determinism Gate — Execution fills WAL parity check.
//!
//! Phase 24C: Compares two `execution_fills.jsonl` WALs (live vs replay)
//! to verify deterministic fill emission.
//!
//! ## Key Structure
//! Fills are keyed by: `(session_id, seq)`
//!
//! ## Comparison Rules (Frozen v1)
//! For each key present in either WAL:
//! 1. If missing in replay → `MissingInReplay`
//! 2. If missing in live → `MissingInLive`
//! 3. If fill_type differs → `FillTypeMismatch`
//! 4. If side differs → `SideMismatch`
//! 5. If payload fields differ → `PayloadMismatch`
//! 6. If digest differs → `DigestMismatch`
//! 7. Else → Match (counted, not stored)
//!
//! ## Payload Fields Compared (Frozen v1)
//! - parent_intent_seq
//! - parent_intent_digest
//! - strategy_id
//! - symbol
//! - qty_mantissa
//! - qty_exponent
//! - price_mantissa
//! - price_exponent
//! - fee_mantissa (Option)
//! - fee_exponent (always compared, even if fee_mantissa=None)
//! - venue
//! - correlation_id
//!
//! Note: ts_ns is NOT compared in v1 to avoid false failures from timestamp jitter.
//!
//! ## Parsing Strictness (Frozen)
//! - Blank/empty lines → parse error (exit code 2)
//! - Unknown enum variants → parse error
//! - Duplicate key → hard error
//!
//! ## Usage
//! ```ignore
//! use quantlaxmi_gates::g6_execution_fill_determinism::G6ExecutionFillDeterminismGate;
//!
//! let result = G6ExecutionFillDeterminismGate::compare(
//!     Path::new("live/wal/execution_fills.jsonl"),
//!     Path::new("replay/wal/execution_fills.jsonl"),
//! )?;
//!
//! if result.passed {
//!     println!("G6 PASSED: {} fills match", result.matched_count);
//! } else {
//!     println!("G6 FAILED: {} mismatches", result.mismatches.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use quantlaxmi_models::{ExecutionFillRecord, FillSide, FillType};

use crate::CheckResult;

// =============================================================================
// Check Names (Frozen)
// =============================================================================

pub mod check_names {
    pub const G6_PARSE_LIVE: &str = "g6_parse_live";
    pub const G6_PARSE_REPLAY: &str = "g6_parse_replay";
    pub const G6_EXECUTION_FILL_PARITY: &str = "g6_execution_fill_parity";
}

// =============================================================================
// Payload Field Names (Frozen)
// =============================================================================

pub mod field_names {
    pub const PARENT_INTENT_SEQ: &str = "parent_intent_seq";
    pub const PARENT_INTENT_DIGEST: &str = "parent_intent_digest";
    pub const STRATEGY_ID: &str = "strategy_id";
    pub const SYMBOL: &str = "symbol";
    pub const QTY_MANTISSA: &str = "qty_mantissa";
    pub const QTY_EXPONENT: &str = "qty_exponent";
    pub const PRICE_MANTISSA: &str = "price_mantissa";
    pub const PRICE_EXPONENT: &str = "price_exponent";
    pub const FEE_MANTISSA: &str = "fee_mantissa";
    pub const FEE_EXPONENT: &str = "fee_exponent";
    pub const VENUE: &str = "venue";
    pub const CORRELATION_ID: &str = "correlation_id";
}

// =============================================================================
// G6ExecutionFillKey — Composite key for matching fills
// =============================================================================

/// Key for matching execution fills across WALs.
///
/// Tuple: (session_id, seq)
pub type G6ExecutionFillKey = (String, u64);

/// Extract key from a fill record.
fn fill_key(r: &ExecutionFillRecord) -> G6ExecutionFillKey {
    (r.session_id.clone(), r.seq)
}

// =============================================================================
// G6MismatchKind — Why two fills don't match
// =============================================================================

/// Mismatch types for G6 comparison (frozen 6-way taxonomy).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum G6MismatchKind {
    /// Fill exists in live but not in replay
    MissingInReplay,

    /// Fill exists in replay but not in live
    MissingInLive,

    /// Fill type differs (full vs partial)
    FillTypeMismatch {
        live_fill_type: String,
        replay_fill_type: String,
    },

    /// Side differs (buy vs sell)
    SideMismatch {
        live_side: String,
        replay_side: String,
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

impl G6MismatchKind {
    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            G6MismatchKind::MissingInReplay => "Fill missing in replay WAL".to_string(),
            G6MismatchKind::MissingInLive => "Fill missing in live WAL".to_string(),
            G6MismatchKind::FillTypeMismatch {
                live_fill_type,
                replay_fill_type,
            } => format!(
                "Fill type mismatch: live={}, replay={}",
                live_fill_type, replay_fill_type
            ),
            G6MismatchKind::SideMismatch {
                live_side,
                replay_side,
            } => format!("Side mismatch: live={}, replay={}", live_side, replay_side),
            G6MismatchKind::PayloadMismatch {
                field,
                live_value,
                replay_value,
            } => format!(
                "Payload mismatch: field={}, live={}, replay={}",
                field, live_value, replay_value
            ),
            G6MismatchKind::DigestMismatch {
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

fn fill_type_to_string(ft: &FillType) -> String {
    match ft {
        FillType::Full => "full".to_string(),
        FillType::Partial => "partial".to_string(),
    }
}

fn side_to_string(side: &FillSide) -> String {
    match side {
        FillSide::Buy => "buy".to_string(),
        FillSide::Sell => "sell".to_string(),
    }
}

// =============================================================================
// G6Mismatch — A single mismatch entry
// =============================================================================

/// A single mismatch between live and replay WALs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G6Mismatch {
    /// Session ID component of the key
    pub session_id: String,
    /// Sequence number component of the key
    pub seq: u64,
    /// Type of mismatch
    pub kind: G6MismatchKind,
}

impl G6Mismatch {
    /// Create a new mismatch.
    pub fn new(key: &G6ExecutionFillKey, kind: G6MismatchKind) -> Self {
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
// G6Result — Gate result summary
// =============================================================================

/// Result of G6 execution fill determinism gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G6Result {
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

    /// Number of fills that matched exactly
    pub matched_count: usize,

    /// Individual check results (parse + compare)
    pub checks: Vec<CheckResult>,

    /// List of mismatches (empty if passed)
    pub mismatches: Vec<G6Mismatch>,

    /// Error message if gate failed to run
    pub error: Option<String>,
}

impl G6Result {
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
            format!("G6 gate error: {}", err)
        } else if self.passed {
            format!(
                "All {} fills match between live and replay",
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
// G6ExecutionFillDeterminismGate — The gate implementation
// =============================================================================

/// G6 Execution Fill Determinism Gate.
///
/// Compares two execution_fills.jsonl WALs to verify deterministic
/// fill emission between live and replay runs.
pub struct G6ExecutionFillDeterminismGate;

impl G6ExecutionFillDeterminismGate {
    /// Compare two execution fill WALs.
    ///
    /// Returns `G6Result` with pass/fail status and any mismatches.
    ///
    /// # Errors
    /// Returns error string if:
    /// - Files cannot be opened
    /// - JSON parse errors occur (hard failure per spec)
    /// - Blank/empty lines found (parse error per spec)
    /// - Duplicate keys detected in same WAL
    pub fn compare(live_wal: &Path, replay_wal: &Path) -> Result<G6Result, String> {
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
            check_names::G6_PARSE_LIVE,
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
            check_names::G6_PARSE_REPLAY,
            format!("Parsed {} entries from replay WAL", replay_count),
        ));

        // Compare fills
        let (mismatches, matched_count) = Self::compare_maps(&live_map, &replay_map);

        let passed = mismatches.is_empty();
        let message = if passed {
            format!("All {} fills match between live and replay", matched_count)
        } else {
            format!(
                "{} mismatches found ({} matched)",
                mismatches.len(),
                matched_count
            )
        };

        checks.push(if passed {
            CheckResult::pass(check_names::G6_EXECUTION_FILL_PARITY, message)
        } else {
            CheckResult::fail(check_names::G6_EXECUTION_FILL_PARITY, message)
        });

        Ok(G6Result {
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
    /// - Blank/empty line (strictness: reject)
    /// - Empty session_id
    /// - Duplicate key (same session_id + seq)
    fn parse_wal(
        path: &Path,
    ) -> Result<(HashMap<G6ExecutionFillKey, ExecutionFillRecord>, usize), String> {
        let file =
            File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
        let reader = BufReader::new(file);

        let mut map = HashMap::new();
        let mut line_num = 0;

        for line in reader.lines() {
            line_num += 1;
            let line = line.map_err(|e| format!("Line {}: read error: {}", line_num, e))?;

            // Strictness: blank lines are parse errors (not skipped)
            if line.trim().is_empty() {
                return Err(format!(
                    "Line {}: blank/empty line (WAL corruption or malformed)",
                    line_num
                ));
            }

            let record: ExecutionFillRecord = serde_json::from_str(&line)
                .map_err(|e| format!("Line {}: parse error: {}", line_num, e))?;

            // Validate session_id is non-empty
            if record.session_id.trim().is_empty() {
                return Err(format!(
                    "Line {}: empty session_id for seq={}",
                    line_num, record.seq
                ));
            }

            let key = fill_key(&record);

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
        live: &HashMap<G6ExecutionFillKey, ExecutionFillRecord>,
        replay: &HashMap<G6ExecutionFillKey, ExecutionFillRecord>,
    ) -> (Vec<G6Mismatch>, usize) {
        let mut mismatches = Vec::new();
        let mut matched = 0;

        // Check all keys in live
        for (key, live_record) in live {
            match replay.get(key) {
                None => {
                    mismatches.push(G6Mismatch::new(key, G6MismatchKind::MissingInReplay));
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
                mismatches.push(G6Mismatch::new(key, G6MismatchKind::MissingInLive));
            }
        }

        (mismatches, matched)
    }

    /// Compare two records with the same key.
    ///
    /// Returns Some(mismatch) if they differ, None if they match.
    /// Comparison order (frozen v1):
    /// 1. FillType
    /// 2. Side
    /// 3. Payload fields
    /// 4. Digest
    ///
    /// Note: ts_ns is NOT compared (v1).
    fn compare_records(
        key: &G6ExecutionFillKey,
        live: &ExecutionFillRecord,
        replay: &ExecutionFillRecord,
    ) -> Option<G6Mismatch> {
        // 1. FillType mismatch
        if live.fill_type != replay.fill_type {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::FillTypeMismatch {
                    live_fill_type: fill_type_to_string(&live.fill_type),
                    replay_fill_type: fill_type_to_string(&replay.fill_type),
                },
            ));
        }

        // 2. Side mismatch
        if live.side != replay.side {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::SideMismatch {
                    live_side: side_to_string(&live.side),
                    replay_side: side_to_string(&replay.side),
                },
            ));
        }

        // 3. Payload fields (frozen order, frozen field names)

        // parent_intent_seq
        if live.parent_intent_seq != replay.parent_intent_seq {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::PARENT_INTENT_SEQ.to_string(),
                    live_value: serde_json::json!(live.parent_intent_seq),
                    replay_value: serde_json::json!(replay.parent_intent_seq),
                },
            ));
        }

        // parent_intent_digest
        if live.parent_intent_digest != replay.parent_intent_digest {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::PARENT_INTENT_DIGEST.to_string(),
                    live_value: serde_json::json!(live.parent_intent_digest),
                    replay_value: serde_json::json!(replay.parent_intent_digest),
                },
            ));
        }

        // strategy_id
        if live.strategy_id != replay.strategy_id {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::STRATEGY_ID.to_string(),
                    live_value: serde_json::Value::String(live.strategy_id.clone()),
                    replay_value: serde_json::Value::String(replay.strategy_id.clone()),
                },
            ));
        }

        // symbol
        if live.symbol != replay.symbol {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::SYMBOL.to_string(),
                    live_value: serde_json::Value::String(live.symbol.clone()),
                    replay_value: serde_json::Value::String(replay.symbol.clone()),
                },
            ));
        }

        // qty_mantissa
        if live.qty_mantissa != replay.qty_mantissa {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::QTY_MANTISSA.to_string(),
                    live_value: serde_json::Value::Number(live.qty_mantissa.into()),
                    replay_value: serde_json::Value::Number(replay.qty_mantissa.into()),
                },
            ));
        }

        // qty_exponent
        if live.qty_exponent != replay.qty_exponent {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::QTY_EXPONENT.to_string(),
                    live_value: serde_json::Value::Number(live.qty_exponent.into()),
                    replay_value: serde_json::Value::Number(replay.qty_exponent.into()),
                },
            ));
        }

        // price_mantissa
        if live.price_mantissa != replay.price_mantissa {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::PRICE_MANTISSA.to_string(),
                    live_value: serde_json::Value::Number(live.price_mantissa.into()),
                    replay_value: serde_json::Value::Number(replay.price_mantissa.into()),
                },
            ));
        }

        // price_exponent
        if live.price_exponent != replay.price_exponent {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::PRICE_EXPONENT.to_string(),
                    live_value: serde_json::Value::Number(live.price_exponent.into()),
                    replay_value: serde_json::Value::Number(replay.price_exponent.into()),
                },
            ));
        }

        // fee_mantissa (Option<i64>)
        if live.fee_mantissa != replay.fee_mantissa {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::FEE_MANTISSA.to_string(),
                    live_value: serde_json::json!(live.fee_mantissa),
                    replay_value: serde_json::json!(replay.fee_mantissa),
                },
            ));
        }

        // fee_exponent (always compared, even if fee_mantissa=None)
        if live.fee_exponent != replay.fee_exponent {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::FEE_EXPONENT.to_string(),
                    live_value: serde_json::Value::Number(live.fee_exponent.into()),
                    replay_value: serde_json::Value::Number(replay.fee_exponent.into()),
                },
            ));
        }

        // venue
        if live.venue != replay.venue {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::VENUE.to_string(),
                    live_value: serde_json::Value::String(live.venue.clone()),
                    replay_value: serde_json::Value::String(replay.venue.clone()),
                },
            ));
        }

        // correlation_id
        if live.correlation_id != replay.correlation_id {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::PayloadMismatch {
                    field: field_names::CORRELATION_ID.to_string(),
                    live_value: serde_json::Value::String(live.correlation_id.clone()),
                    replay_value: serde_json::Value::String(replay.correlation_id.clone()),
                },
            ));
        }

        // 4. Digest mismatch (catches any canonical bytes difference)
        if live.digest != replay.digest {
            return Some(G6Mismatch::new(
                key,
                G6MismatchKind::DigestMismatch {
                    live_digest: live.digest.clone(),
                    replay_digest: replay.digest.clone(),
                },
            ));
        }

        // All checks passed
        None
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::ExecutionFillRecord;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: create a fill record with minimal fields
    fn make_fill(session_id: &str, seq: u64) -> ExecutionFillRecord {
        ExecutionFillRecord::builder("test_strat", "BTCUSDT")
            .ts_ns(1000 + seq as i64 * 1000)
            .session_id(session_id)
            .seq(seq)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id(&format!("corr_{}", seq))
            .fill_type(FillType::Full)
            .build()
    }

    /// Helper: write fills to temp file, return path
    fn write_temp_wal(fills: &[ExecutionFillRecord]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for fill in fills {
            let line = serde_json::to_string(fill).unwrap();
            writeln!(file, "{}", line).unwrap();
        }
        file.flush().unwrap();
        file
    }

    // -------------------------------------------------------------------------
    // Test 1: Identical WALs pass
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_identical_wals_pass() {
        let fills = vec![
            make_fill("sess1", 1),
            make_fill("sess1", 2),
            make_fill("sess1", 3),
        ];

        let live_file = write_temp_wal(&fills);
        let replay_file = write_temp_wal(&fills);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(result.passed, "Identical WALs should pass");
        assert_eq!(result.matched_count, 3);
        assert!(result.mismatches.is_empty());
        assert_eq!(result.live_entry_count, 3);
        assert_eq!(result.replay_entry_count, 3);
    }

    // -------------------------------------------------------------------------
    // Test 2: Missing in replay
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_missing_in_replay() {
        let live_fills = vec![make_fill("sess1", 1), make_fill("sess1", 2)];
        let replay_fills = vec![make_fill("sess1", 1)]; // Missing seq=2

        let live_file = write_temp_wal(&live_fills);
        let replay_file = write_temp_wal(&replay_fills);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G6MismatchKind::MissingInReplay
        ));
        assert_eq!(result.mismatches[0].seq, 2);
    }

    // -------------------------------------------------------------------------
    // Test 3: Missing in live
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_missing_in_live() {
        let live_fills = vec![make_fill("sess1", 1)];
        let replay_fills = vec![make_fill("sess1", 1), make_fill("sess1", 2)]; // Extra seq=2

        let live_file = write_temp_wal(&live_fills);
        let replay_file = write_temp_wal(&replay_fills);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G6MismatchKind::MissingInLive
        ));
        assert_eq!(result.mismatches[0].seq, 2);
    }

    // -------------------------------------------------------------------------
    // Test 4: Fill type mismatch
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_fill_type_mismatch() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Partial) // Different
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G6MismatchKind::FillTypeMismatch { .. }
        ));
    }

    // -------------------------------------------------------------------------
    // Test 5: Side mismatch
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_side_mismatch() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Sell) // Different
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G6MismatchKind::SideMismatch { .. }
        ));
    }

    // -------------------------------------------------------------------------
    // Test 6: Payload mismatch - qty
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_payload_mismatch_qty() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(200, -8) // Different
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        if let G6MismatchKind::PayloadMismatch { field, .. } = &result.mismatches[0].kind {
            assert_eq!(field, field_names::QTY_MANTISSA);
        } else {
            panic!("Expected PayloadMismatch");
        }
    }

    // -------------------------------------------------------------------------
    // Test 7: Payload mismatch - price
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_payload_mismatch_price() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(6000, -2) // Different
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        if let G6MismatchKind::PayloadMismatch { field, .. } = &result.mismatches[0].kind {
            assert_eq!(field, field_names::PRICE_MANTISSA);
        } else {
            panic!("Expected PayloadMismatch");
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: Payload mismatch - fee option
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_payload_mismatch_fee_option() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2) // Has fee
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            // No fee
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        if let G6MismatchKind::PayloadMismatch { field, .. } = &result.mismatches[0].kind {
            assert_eq!(field, field_names::FEE_MANTISSA);
        } else {
            panic!("Expected PayloadMismatch for fee_mantissa");
        }
    }

    // -------------------------------------------------------------------------
    // Test 9: Payload mismatch - venue
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_payload_mismatch_venue() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("binance") // Different
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        if let G6MismatchKind::PayloadMismatch { field, .. } = &result.mismatches[0].kind {
            assert_eq!(field, field_names::VENUE);
        } else {
            panic!("Expected PayloadMismatch for venue");
        }
    }

    // -------------------------------------------------------------------------
    // Test 10: Payload mismatch - parent_intent_digest
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_payload_mismatch_parent_intent_digest() {
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .parent_intent_seq(5)
            .parent_intent_digest("abc123")
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let replay_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .parent_intent_seq(5)
            .parent_intent_digest("xyz789") // Different
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        if let G6MismatchKind::PayloadMismatch { field, .. } = &result.mismatches[0].kind {
            assert_eq!(field, field_names::PARENT_INTENT_DIGEST);
        } else {
            panic!("Expected PayloadMismatch for parent_intent_digest");
        }
    }

    // -------------------------------------------------------------------------
    // Test 11: Digest mismatch (same payload but different digest - should not happen)
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_digest_mismatch() {
        // Create two fills with same payload but manually tamper with digest
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let mut replay_fill = live_fill.clone();
        // Tamper with digest (in practice this means canonical bytes differ)
        replay_fill.digest = "tampered_digest_value_0000000000000000".to_string();

        let live_file = write_temp_wal(&[live_fill]);
        let replay_file = write_temp_wal(&[replay_fill]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            result.mismatches[0].kind,
            G6MismatchKind::DigestMismatch { .. }
        ));
    }

    // -------------------------------------------------------------------------
    // Test 12: Empty WALs pass
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_empty_wals_pass() {
        let live_file = write_temp_wal(&[]);
        let replay_file = write_temp_wal(&[]);

        let result =
            G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path()).unwrap();

        assert!(result.passed, "Empty WALs should pass");
        assert_eq!(result.matched_count, 0);
        assert!(result.mismatches.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test 13: Duplicate key error
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_duplicate_key_error() {
        let fill1 = make_fill("sess1", 1);
        let fill2 = make_fill("sess1", 1); // Duplicate key!

        // Write manually to create duplicate
        let mut live_file = NamedTempFile::new().unwrap();
        writeln!(live_file, "{}", serde_json::to_string(&fill1).unwrap()).unwrap();
        writeln!(live_file, "{}", serde_json::to_string(&fill2).unwrap()).unwrap();
        live_file.flush().unwrap();

        let replay_file = write_temp_wal(&[make_fill("sess1", 1)]);

        let result = G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path());

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("duplicate key"),
            "Error should mention duplicate key: {}",
            err
        );
    }

    // -------------------------------------------------------------------------
    // Test 14: Check names are stable (frozen)
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_check_names_stable() {
        assert_eq!(check_names::G6_PARSE_LIVE, "g6_parse_live");
        assert_eq!(check_names::G6_PARSE_REPLAY, "g6_parse_replay");
        assert_eq!(
            check_names::G6_EXECUTION_FILL_PARITY,
            "g6_execution_fill_parity"
        );
    }

    // -------------------------------------------------------------------------
    // Test 15: Field names are stable (frozen)
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_field_names_stable() {
        assert_eq!(field_names::PARENT_INTENT_SEQ, "parent_intent_seq");
        assert_eq!(field_names::PARENT_INTENT_DIGEST, "parent_intent_digest");
        assert_eq!(field_names::STRATEGY_ID, "strategy_id");
        assert_eq!(field_names::SYMBOL, "symbol");
        assert_eq!(field_names::QTY_MANTISSA, "qty_mantissa");
        assert_eq!(field_names::QTY_EXPONENT, "qty_exponent");
        assert_eq!(field_names::PRICE_MANTISSA, "price_mantissa");
        assert_eq!(field_names::PRICE_EXPONENT, "price_exponent");
        assert_eq!(field_names::FEE_MANTISSA, "fee_mantissa");
        assert_eq!(field_names::FEE_EXPONENT, "fee_exponent");
        assert_eq!(field_names::VENUE, "venue");
        assert_eq!(field_names::CORRELATION_ID, "correlation_id");
    }

    // -------------------------------------------------------------------------
    // Test 16: Mismatch descriptions are stable
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_mismatch_descriptions() {
        let key = ("sess1".to_string(), 42u64);

        // MissingInReplay
        let m1 = G6Mismatch::new(&key, G6MismatchKind::MissingInReplay);
        assert!(m1.description().contains("missing in replay"));

        // MissingInLive
        let m2 = G6Mismatch::new(&key, G6MismatchKind::MissingInLive);
        assert!(m2.description().contains("missing in live"));

        // FillTypeMismatch
        let m3 = G6Mismatch::new(
            &key,
            G6MismatchKind::FillTypeMismatch {
                live_fill_type: "full".to_string(),
                replay_fill_type: "partial".to_string(),
            },
        );
        assert!(m3.description().contains("Fill type mismatch"));

        // SideMismatch
        let m4 = G6Mismatch::new(
            &key,
            G6MismatchKind::SideMismatch {
                live_side: "buy".to_string(),
                replay_side: "sell".to_string(),
            },
        );
        assert!(m4.description().contains("Side mismatch"));

        // PayloadMismatch
        let m5 = G6Mismatch::new(
            &key,
            G6MismatchKind::PayloadMismatch {
                field: "qty_mantissa".to_string(),
                live_value: serde_json::json!(100),
                replay_value: serde_json::json!(200),
            },
        );
        assert!(m5.description().contains("Payload mismatch"));
        assert!(m5.description().contains("qty_mantissa"));

        // DigestMismatch
        let m6 = G6Mismatch::new(
            &key,
            G6MismatchKind::DigestMismatch {
                live_digest: "abc123def456789012345678".to_string(),
                replay_digest: "xyz789000000000012345678".to_string(),
            },
        );
        assert!(m6.description().contains("Digest mismatch"));
    }

    // -------------------------------------------------------------------------
    // Test 17: Blank line is parse error (strictness)
    // -------------------------------------------------------------------------

    #[test]
    fn test_g6_blank_line_parse_error() {
        let fill = make_fill("sess1", 1);

        // Write with blank line
        let mut live_file = NamedTempFile::new().unwrap();
        writeln!(live_file, "{}", serde_json::to_string(&fill).unwrap()).unwrap();
        writeln!(live_file, "").unwrap(); // Blank line
        writeln!(
            live_file,
            "{}",
            serde_json::to_string(&make_fill("sess1", 2)).unwrap()
        )
        .unwrap();
        live_file.flush().unwrap();

        let replay_file = write_temp_wal(&[fill]);

        let result = G6ExecutionFillDeterminismGate::compare(live_file.path(), replay_file.path());

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("blank") || err.contains("empty"),
            "Error should mention blank/empty line: {}",
            err
        );
    }
}
