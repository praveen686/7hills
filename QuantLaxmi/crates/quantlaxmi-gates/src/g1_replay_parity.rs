//! # G1 ReplayParity Gate
//!
//! Ensures deterministic replay matches live execution.
//!
//! ## Checks
//! - **Integrity**: WAL integrity verified against manifest (live + replay)
//! - **Decision Parity**: Replay decision trace hash matches live decision trace hash
//!
//! ## Implementation Notes
//! This gate compares WAL records from a live session against a replay
//! of the same session using the v2 decision trace hash:
//! - Domain-separated (`quantlaxmi:decision_trace:v2`)
//! - Record-delimited (ASCII RS `\x1e`)
//! - Field-explicit (no serde, explicit byte encoding)
//! - Float-immune (excludes confidence, spread_bps, metadata)
//!
//! ## Usage
//! ```ignore
//! let g1 = G1ReplayParity::new(config);
//! let result = g1.compare_sessions(&live_session, &replay_session)?;
//! ```

use crate::{CheckResult, GateError, GateResult};
use chrono::{DateTime, Utc};
use quantlaxmi_models::events::{CorrelationContext, DecisionEvent};
use quantlaxmi_wal::{WalManifest, WalReader};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use uuid::Uuid;

/// G1 ReplayParity configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct G1Config {
    /// Maximum allowed price deviation in basis points
    #[serde(default = "default_price_tolerance_bps")]
    pub price_tolerance_bps: f64,

    /// Maximum allowed quantity deviation as fraction
    #[serde(default = "default_qty_tolerance")]
    pub qty_tolerance: f64,

    /// Require exact decision sequence match
    #[serde(default)]
    pub require_exact_decisions: bool,

    /// Allow timing slack in milliseconds
    #[serde(default = "default_timing_slack_ms")]
    pub timing_slack_ms: i64,
}

fn default_price_tolerance_bps() -> f64 {
    10.0
} // 0.1%
fn default_qty_tolerance() -> f64 {
    0.001
} // 0.1%
fn default_timing_slack_ms() -> i64 {
    100
}

/// G1 ReplayParity gate validator.
pub struct G1ReplayParity {
    #[allow(dead_code)]
    config: G1Config,
}

impl G1ReplayParity {
    /// Create a new G1 validator.
    pub fn new(config: G1Config) -> Self {
        Self { config }
    }

    /// Compare live session against replay session.
    ///
    /// # Arguments
    /// * `live` - Path to live session WAL directory
    /// * `replay` - Path to replay session WAL directory
    ///
    /// # Returns
    /// Gate result with integrity and parity checks
    pub fn compare_sessions(&self, live: &Path, replay: &Path) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G1_ReplayParity");

        // 1) Load manifests
        let live_manifest = WalManifest::load(&live.join("manifest.json"))
            .map_err(|e| GateError::Validation(format!("live manifest load failed: {e}")))?;
        let replay_manifest = WalManifest::load(&replay.join("manifest.json"))
            .map_err(|e| GateError::Validation(format!("replay manifest load failed: {e}")))?;

        // 2) Open WAL readers
        let live_reader = WalReader::open(live)
            .map_err(|e| GateError::Validation(format!("live WalReader::open failed: {e}")))?;
        let replay_reader = WalReader::open(replay)
            .map_err(|e| GateError::Validation(format!("replay WalReader::open failed: {e}")))?;

        // 3) Verify integrity against manifests
        if let Err(e) = live_reader.verify_integrity(&live_manifest) {
            result.add_check(CheckResult::fail(
                "live_integrity",
                format!("Live session integrity check failed: {e}"),
            ));
            result.duration_ms = start.elapsed().as_millis() as u64;
            result.summary = format!(
                "{}/{} checks passed",
                result.passed_count(),
                result.checks.len()
            );
            return Ok(result);
        }
        result.add_check(CheckResult::pass(
            "live_integrity",
            "Live WAL integrity verified",
        ));

        if let Err(e) = replay_reader.verify_integrity(&replay_manifest) {
            result.add_check(CheckResult::fail(
                "replay_integrity",
                format!("Replay session integrity check failed: {e}"),
            ));
            result.duration_ms = start.elapsed().as_millis() as u64;
            result.summary = format!(
                "{}/{} checks passed",
                result.passed_count(),
                result.checks.len()
            );
            return Ok(result);
        }
        result.add_check(CheckResult::pass(
            "replay_integrity",
            "Replay WAL integrity verified",
        ));

        // 3b) Verify stream digests (streaming hash, hard-fail on mismatch)
        if let Err(e) = live_reader.verify_stream_digests(&live_manifest) {
            result.add_check(CheckResult::fail(
                "live_digests",
                format!("Live stream digest verification failed: {e}"),
            ));
            result.duration_ms = start.elapsed().as_millis() as u64;
            result.summary = format!(
                "{}/{} checks passed",
                result.passed_count(),
                result.checks.len()
            );
            return Ok(result);
        }
        result.add_check(CheckResult::pass(
            "live_digests",
            "Live stream digests verified (or none present)",
        ));

        if let Err(e) = replay_reader.verify_stream_digests(&replay_manifest) {
            result.add_check(CheckResult::fail(
                "replay_digests",
                format!("Replay stream digest verification failed: {e}"),
            ));
            result.duration_ms = start.elapsed().as_millis() as u64;
            result.summary = format!(
                "{}/{} checks passed",
                result.passed_count(),
                result.checks.len()
            );
            return Ok(result);
        }
        result.add_check(CheckResult::pass(
            "replay_digests",
            "Replay stream digests verified (or none present)",
        ));

        // 4) Load decisions (WAL line order; DO NOT sort)
        let live_decisions = live_reader
            .read_decisions()
            .map_err(|e| GateError::Validation(format!("live read_decisions failed: {e}")))?;
        let replay_decisions = replay_reader
            .read_decisions()
            .map_err(|e| GateError::Validation(format!("replay read_decisions failed: {e}")))?;

        // 5) Compute v2 trace hashes (domain-separated, record-separated, field-explicit)
        let live_hash = compute_decision_trace_hash_v2(&live_decisions);
        let replay_hash = compute_decision_trace_hash_v2(&replay_decisions);

        // 6) Compare
        if live_hash == replay_hash {
            result.add_check(CheckResult::pass(
                "decision_parity",
                format!("Decision trace hashes match: {}", &live_hash[..16]),
            ));
        } else {
            let divergence_idx = find_first_divergence_v2(&live_decisions, &replay_decisions);
            let live_len = live_decisions.len();
            let replay_len = replay_decisions.len();

            // Compute per-record hashes at divergence point
            let live_record_hash = live_decisions.get(divergence_idx).map(|d| {
                let h = compute_decision_trace_hash_v2(std::slice::from_ref(d));
                h[..16].to_string()
            });
            let replay_record_hash = replay_decisions.get(divergence_idx).map(|d| {
                let h = compute_decision_trace_hash_v2(std::slice::from_ref(d));
                h[..16].to_string()
            });

            // Enrich message with decision_ids if available at divergence index
            let live_id = live_decisions
                .get(divergence_idx)
                .map(|d| d.decision_id.to_string());
            let replay_id = replay_decisions
                .get(divergence_idx)
                .map(|d| d.decision_id.to_string());

            let id_hint = match (live_id, replay_id) {
                (Some(l), Some(r)) => format!(
                    "live_id={}, replay_id={}",
                    l.get(..8).unwrap_or(&l),
                    r.get(..8).unwrap_or(&r)
                ),
                (Some(l), None) => {
                    format!("live_id={}, replay=MISSING", l.get(..8).unwrap_or(&l))
                }
                (None, Some(r)) => {
                    format!("live=MISSING, replay_id={}", r.get(..8).unwrap_or(&r))
                }
                (None, None) => "no_ids".to_string(),
            };

            let record_hint = match (live_record_hash, replay_record_hash) {
                (Some(l), Some(r)) => format!("record_hash: {}... vs {}...", l, r),
                (Some(l), None) => format!("record_hash: {}... vs MISSING", l),
                (None, Some(r)) => format!("record_hash: MISSING vs {}...", r),
                (None, None) => "record_hash: both MISSING".to_string(),
            };

            result.add_check(CheckResult::fail(
                "decision_parity",
                format!(
                    "MISMATCH@{} (live_len={}, replay_len={}). {}. {}. trace: {}...vs {}...",
                    divergence_idx,
                    live_len,
                    replay_len,
                    id_hint,
                    record_hint,
                    &live_hash[..16],
                    &replay_hash[..16],
                ),
            ));
        }

        result.duration_ms = start.elapsed().as_millis() as u64;
        result.summary = format!(
            "{}/{} checks passed",
            result.passed_count(),
            result.checks.len()
        );
        Ok(result)
    }
}

/// Find the first divergence index between live and replay decision sequences.
///
/// Returns the index of the first mismatching decision, or `min(live.len(), replay.len())`
/// if one sequence is a prefix of the other.
fn find_first_divergence_v2(live: &[DecisionEvent], replay: &[DecisionEvent]) -> usize {
    let n = std::cmp::min(live.len(), replay.len());
    for i in 0..n {
        // Fast path: identical domain-relevant identity often catches mismatch early
        if live[i].decision_id != replay[i].decision_id {
            return i;
        }

        // Strict path: compare the exact v2-encoded bytes by hashing each record alone
        // This avoids needing to manually compare every field here.
        let lh = compute_decision_trace_hash_v2(std::slice::from_ref(&live[i]));
        let rh = compute_decision_trace_hash_v2(std::slice::from_ref(&replay[i]));
        if lh != rh {
            return i;
        }
    }
    n // either one ended earlier, or both identical up to min length
}

// =============================================================================
// DECISION TRACE HASH V2
// =============================================================================

/// Decision trace hash v2
///
/// Properties:
/// - Domain separated with prefix `b"quantlaxmi:decision_trace:v2\0"`
/// - WAL line order preserved (caller must pass decisions in WAL order; DO NOT sort)
/// - Explicitly EXCLUDES:
///   - `DecisionEvent.confidence` (f64)
///   - `DecisionEvent.metadata` (serde_json::Value)
///   - `MarketSnapshot.spread_bps` (f64)
///
/// Encoding rules:
/// - Strings: u32_le(len) + utf8 bytes (no normalization)
/// - UUID: 16 raw bytes (Uuid::as_bytes())
/// - i64: little-endian 8 bytes
/// - i8: single byte as u8
/// - ts: i64 nanoseconds since Unix epoch, little-endian
/// - Record separator: b"\x1e" between decision records
pub fn compute_decision_trace_hash_v2(decisions: &[DecisionEvent]) -> String {
    let mut hasher = Sha256::new();

    // (1) Domain prefix
    hasher.update(b"quantlaxmi:decision_trace:v2\0");

    for d in decisions {
        // (2) Record separator
        hasher.update(b"\x1e");

        // (3) Core identity + decision parameters
        hash_ts_ns(&mut hasher, &d.ts);

        hash_uuid(&mut hasher, &d.decision_id);

        hash_string(&mut hasher, &d.strategy_id);
        hash_string(&mut hasher, &d.symbol);
        hash_string(&mut hasher, &d.decision_type);

        // direction: i8 -> single byte
        hasher.update([d.direction as u8]);

        // qty (mantissa + exponent)
        hasher.update(d.target_qty_mantissa.to_le_bytes());
        hasher.update([d.qty_exponent as u8]);

        // reference price (mantissa + exponent)
        hasher.update(d.reference_price_mantissa.to_le_bytes());
        hasher.update([d.price_exponent as u8]);

        // (4) MarketSnapshot (EXCLUDING spread_bps)
        //
        // Keep explicit field order. Do not serde.
        hasher.update(d.market_snapshot.bid_price_mantissa.to_le_bytes());
        hasher.update(d.market_snapshot.ask_price_mantissa.to_le_bytes());
        hasher.update(d.market_snapshot.bid_qty_mantissa.to_le_bytes());
        hasher.update(d.market_snapshot.ask_qty_mantissa.to_le_bytes());
        hasher.update([d.market_snapshot.price_exponent as u8]);
        hasher.update([d.market_snapshot.qty_exponent as u8]);

        // book_ts_ns: i64
        hasher.update(d.market_snapshot.book_ts_ns.to_le_bytes());

        // (5) CorrelationContext (locked helper)
        hash_correlation_context(&mut hasher, &d.ctx);

        // (6) Explicit exclusions are by omission:
        // - d.confidence
        // - d.metadata
        // - d.market_snapshot.spread_bps
    }

    hex::encode(hasher.finalize())
}

/* ------------------------- Helpers (encoding) ------------------------- */

fn hash_string<H: Digest>(hasher: &mut H, s: &str) {
    let b = s.as_bytes();
    let len = b.len() as u32;
    hasher.update(len.to_le_bytes());
    hasher.update(b);
}

fn hash_uuid<H: Digest>(hasher: &mut H, u: &Uuid) {
    hasher.update(u.as_bytes());
}

fn hash_ts_ns<H: Digest>(hasher: &mut H, ts: &DateTime<Utc>) {
    // i64 nanoseconds since unix epoch
    // (chrono returns i64 seconds + u32 nanos; keep stable, avoid floats)
    let secs = ts.timestamp(); // i64
    let nanos = ts.timestamp_subsec_nanos() as i64; // 0..999,999,999
    let ns = secs.saturating_mul(1_000_000_000).saturating_add(nanos);
    hasher.update(ns.to_le_bytes());
}

/* ------------------ Locked CorrelationContext encoding ------------------ */

fn hash_opt_string<H: Digest>(hasher: &mut H, v: &Option<String>) {
    match v {
        None => hasher.update([0x00]),
        Some(s) => {
            hasher.update([0x01]);
            let b = s.as_bytes();
            let len = b.len() as u32;
            hasher.update(len.to_le_bytes());
            hasher.update(b);
        }
    }
}

fn hash_opt_uuid<H: Digest>(hasher: &mut H, v: &Option<Uuid>) {
    match v {
        None => hasher.update([0x00]),
        Some(u) => {
            hasher.update([0x01]);
            hasher.update(u.as_bytes());
        }
    }
}

fn hash_correlation_context<H: Digest>(hasher: &mut H, ctx: &CorrelationContext) {
    // Field order MUST match declared order in events.rs
    hash_opt_string(hasher, &ctx.session_id);
    hash_opt_string(hasher, &ctx.run_id);
    hash_opt_string(hasher, &ctx.symbol);
    hash_opt_string(hasher, &ctx.venue);
    hash_opt_string(hasher, &ctx.strategy_id);
    hash_opt_uuid(hasher, &ctx.decision_id);
    hash_opt_uuid(hasher, &ctx.order_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::events::MarketSnapshot;
    use tempfile::tempdir;

    #[test]
    fn test_g1_missing_manifest_returns_error() {
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();

        let g1 = G1ReplayParity::new(G1Config::default());
        let err = g1.compare_sessions(dir1.path(), dir2.path()).unwrap_err();

        // Empty directories have no manifest.json - should fail with Validation error
        match err {
            GateError::Validation(msg) => {
                assert!(msg.contains("manifest"), "Expected manifest error: {msg}")
            }
            other => panic!("Expected GateError::Validation, got: {other:?}"),
        }
    }

    // =========================================================================
    // Decision Trace Hash v2 - Context Correctness Tests
    // =========================================================================

    /// Create a minimal DecisionEvent for testing hash behavior.
    fn make_test_decision(ctx: CorrelationContext) -> DecisionEvent {
        DecisionEvent {
            ts: chrono::DateTime::from_timestamp(1700000000, 0).unwrap(),
            decision_id: Uuid::nil(),
            strategy_id: "test_strategy".to_string(),
            symbol: "BTCUSDT".to_string(),
            decision_type: "ENTRY".to_string(),
            direction: 1,
            target_qty_mantissa: 100_000,
            qty_exponent: -4,
            reference_price_mantissa: 5000000,
            price_exponent: -2,
            market_snapshot: MarketSnapshot {
                bid_price_mantissa: 4999900,
                ask_price_mantissa: 5000100,
                bid_qty_mantissa: 1000,
                ask_qty_mantissa: 1000,
                price_exponent: -2,
                qty_exponent: -2,
                spread_bps: 4.0, // EXCLUDED from hash
                book_ts_ns: 1_700_000_000_000_000_000,
            },
            confidence: 0.95,                  // EXCLUDED from hash
            metadata: serde_json::Value::Null, // EXCLUDED from hash
            ctx,
        }
    }

    #[test]
    fn test_ctx_none_vs_empty_string() {
        // None vs Some("") must produce different hashes
        let ctx_none = CorrelationContext {
            session_id: None,
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let ctx_empty = CorrelationContext {
            session_id: Some("".to_string()),
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let hash_none = compute_decision_trace_hash_v2(&[make_test_decision(ctx_none)]);
        let hash_empty = compute_decision_trace_hash_v2(&[make_test_decision(ctx_empty)]);

        assert_ne!(
            hash_none, hash_empty,
            "None vs Some(\"\") must produce different hashes"
        );
    }

    #[test]
    fn test_ctx_case_sensitivity() {
        // Some("binance") vs Some("BINANCE") must produce different hashes
        let ctx_lower = CorrelationContext {
            session_id: None,
            run_id: None,
            symbol: None,
            venue: Some("binance".to_string()),
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let ctx_upper = CorrelationContext {
            session_id: None,
            run_id: None,
            symbol: None,
            venue: Some("BINANCE".to_string()),
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let hash_lower = compute_decision_trace_hash_v2(&[make_test_decision(ctx_lower)]);
        let hash_upper = compute_decision_trace_hash_v2(&[make_test_decision(ctx_upper)]);

        assert_ne!(
            hash_lower, hash_upper,
            "Case must matter: 'binance' vs 'BINANCE'"
        );
    }

    #[test]
    fn test_ctx_decision_id_none_vs_some() {
        // decision_id None vs Some(uuid) must produce different hashes
        let test_uuid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

        let ctx_none = CorrelationContext {
            session_id: None,
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let ctx_some = CorrelationContext {
            session_id: None,
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: Some(test_uuid),
            order_id: None,
        };

        let hash_none = compute_decision_trace_hash_v2(&[make_test_decision(ctx_none)]);
        let hash_some = compute_decision_trace_hash_v2(&[make_test_decision(ctx_some)]);

        assert_ne!(
            hash_none, hash_some,
            "decision_id None vs Some(uuid) must produce different hashes"
        );
    }

    #[test]
    fn test_ctx_each_field_matters() {
        // Verify that changing each ctx field produces a different hash
        let base_ctx = CorrelationContext {
            session_id: None,
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };
        let base_hash = compute_decision_trace_hash_v2(&[make_test_decision(base_ctx.clone())]);

        // Test each field independently
        let test_cases = [
            (
                "session_id",
                CorrelationContext {
                    session_id: Some("sess".to_string()),
                    ..base_ctx.clone()
                },
            ),
            (
                "run_id",
                CorrelationContext {
                    run_id: Some("run".to_string()),
                    ..base_ctx.clone()
                },
            ),
            (
                "symbol",
                CorrelationContext {
                    symbol: Some("ETHUSDT".to_string()),
                    ..base_ctx.clone()
                },
            ),
            (
                "venue",
                CorrelationContext {
                    venue: Some("binance".to_string()),
                    ..base_ctx.clone()
                },
            ),
            (
                "strategy_id",
                CorrelationContext {
                    strategy_id: Some("strat".to_string()),
                    ..base_ctx.clone()
                },
            ),
            (
                "decision_id",
                CorrelationContext {
                    decision_id: Some(
                        Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
                    ),
                    ..base_ctx.clone()
                },
            ),
            (
                "order_id",
                CorrelationContext {
                    order_id: Some(
                        Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap(),
                    ),
                    ..base_ctx.clone()
                },
            ),
        ];

        for (field_name, modified_ctx) in test_cases {
            let modified_hash = compute_decision_trace_hash_v2(&[make_test_decision(modified_ctx)]);
            assert_ne!(
                base_hash, modified_hash,
                "Changing ctx.{} must produce a different hash",
                field_name
            );
        }
    }

    #[test]
    fn test_excluded_fields_do_not_affect_hash() {
        // confidence, metadata, spread_bps are excluded - changing them must NOT affect hash
        let ctx = CorrelationContext {
            session_id: Some("test".to_string()),
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let d1 = make_test_decision(ctx.clone());
        let mut d2 = make_test_decision(ctx);

        // Change excluded fields
        d2.confidence = 0.50; // different confidence
        d2.metadata = serde_json::json!({"key": "value"}); // different metadata
        d2.market_snapshot.spread_bps = 999.0; // different spread_bps

        let hash1 = compute_decision_trace_hash_v2(&[d1]);
        let hash2 = compute_decision_trace_hash_v2(&[d2]);

        assert_eq!(
            hash1, hash2,
            "Excluded fields (confidence, metadata, spread_bps) must NOT affect hash"
        );
    }

    #[test]
    fn test_ctx_field_order_drift() {
        // Detects accidental reordering in hash_correlation_context()
        // If someone swaps the order of session_id/run_id hashing, this fails
        let ctx_a = CorrelationContext {
            session_id: Some("A".to_string()),
            run_id: Some("B".to_string()),
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let ctx_b = CorrelationContext {
            session_id: Some("B".to_string()),
            run_id: Some("A".to_string()),
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };

        let hash_a = compute_decision_trace_hash_v2(&[make_test_decision(ctx_a)]);
        let hash_b = compute_decision_trace_hash_v2(&[make_test_decision(ctx_b)]);

        assert_ne!(
            hash_a, hash_b,
            "Field order must matter: (session_id=A,run_id=B) vs (session_id=B,run_id=A)"
        );
    }

    #[test]
    fn test_deterministic_hash() {
        // Same input must always produce same output
        let ctx = CorrelationContext {
            session_id: Some("session_123".to_string()),
            run_id: Some("run_456".to_string()),
            symbol: Some("BTCUSDT".to_string()),
            venue: Some("binance".to_string()),
            strategy_id: Some("funding_arb".to_string()),
            decision_id: Some(Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap()),
            order_id: Some(Uuid::parse_str("660e8400-e29b-41d4-a716-446655440000").unwrap()),
        };

        let decisions = vec![make_test_decision(ctx.clone()), make_test_decision(ctx)];

        let hash1 = compute_decision_trace_hash_v2(&decisions);
        let hash2 = compute_decision_trace_hash_v2(&decisions);

        assert_eq!(hash1, hash2, "Hash must be deterministic");
        assert_eq!(hash1.len(), 64, "SHA256 hex must be 64 characters");
    }

    // =========================================================================
    // G1 Integration Test - compare_sessions with valid manifests and digests
    // =========================================================================

    /// Helper to create a valid WAL session directory with manifest and stream files.
    fn create_test_session(
        session_dir: &std::path::Path,
        decision: &DecisionEvent,
    ) -> quantlaxmi_wal::WalManifest {
        use std::collections::HashMap;

        // Create wal subdirectory
        let wal_dir = session_dir.join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Write decisions file
        let decisions_content = serde_json::to_string(decision).unwrap() + "\n";
        let decisions_path = wal_dir.join("decisions.jsonl");
        std::fs::write(&decisions_path, &decisions_content).unwrap();

        // Write empty market file (required by reader)
        let market_path = wal_dir.join("market.jsonl");
        std::fs::write(&market_path, "").unwrap();

        // Compute digests
        let decisions_hash = quantlaxmi_wal::sha256_hex(decisions_content.as_bytes());
        let market_hash = quantlaxmi_wal::sha256_hex(b"");

        // Build manifest
        let mut files = HashMap::new();
        files.insert(
            "decisions".to_string(),
            quantlaxmi_wal::WalFileInfo {
                path: "wal/decisions.jsonl".to_string(),
                sha256: decisions_hash,
                record_count: 1,
                bytes_len: decisions_content.len(),
            },
        );
        files.insert(
            "market".to_string(),
            quantlaxmi_wal::WalFileInfo {
                path: "wal/market.jsonl".to_string(),
                sha256: market_hash,
                record_count: 0,
                bytes_len: 0,
            },
        );

        let manifest = quantlaxmi_wal::WalManifest {
            created_at: chrono::Utc::now(),
            counts: quantlaxmi_wal::WalCounts {
                market_events: 0,
                decision_events: 1,
                order_events: 0,
                fill_events: 0,
                risk_events: 0,
            },
            files,
        };

        // Write manifest
        let manifest_json = serde_json::to_string_pretty(&manifest).unwrap();
        std::fs::write(session_dir.join("manifest.json"), manifest_json).unwrap();

        manifest
    }

    #[test]
    fn test_g1_compare_sessions_with_digests_pass() {
        let temp = tempdir().unwrap();
        let live_dir = temp.path().join("live");
        let replay_dir = temp.path().join("replay");
        std::fs::create_dir_all(&live_dir).unwrap();
        std::fs::create_dir_all(&replay_dir).unwrap();

        // Create identical decision for both sessions
        let ctx = CorrelationContext {
            session_id: Some("test-session".to_string()),
            run_id: Some("test-run".to_string()),
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };
        let decision = make_test_decision(ctx);

        // Create valid sessions
        create_test_session(&live_dir, &decision);
        create_test_session(&replay_dir, &decision);

        // Run G1 comparison
        let g1 = G1ReplayParity::new(G1Config::default());
        let result = g1.compare_sessions(&live_dir, &replay_dir).unwrap();

        // All checks should pass
        assert!(
            result.passed,
            "Expected all checks to pass. Summary: {}. Checks: {:?}",
            result.summary, result.checks
        );

        // Should have 5 checks: live_integrity, replay_integrity, live_digests, replay_digests, decision_parity
        assert_eq!(
            result.checks.len(),
            5,
            "Expected 5 checks, got {}",
            result.checks.len()
        );

        // Verify specific checks passed
        let check_names: Vec<_> = result.checks.iter().map(|c| c.name.as_str()).collect();
        assert!(
            check_names.contains(&"live_integrity"),
            "Missing live_integrity check"
        );
        assert!(
            check_names.contains(&"replay_integrity"),
            "Missing replay_integrity check"
        );
        assert!(
            check_names.contains(&"live_digests"),
            "Missing live_digests check"
        );
        assert!(
            check_names.contains(&"replay_digests"),
            "Missing replay_digests check"
        );
        assert!(
            check_names.contains(&"decision_parity"),
            "Missing decision_parity check"
        );
    }

    #[test]
    fn test_g1_compare_sessions_digest_mismatch_fails() {
        let temp = tempdir().unwrap();
        let live_dir = temp.path().join("live");
        let replay_dir = temp.path().join("replay");
        std::fs::create_dir_all(&live_dir).unwrap();
        std::fs::create_dir_all(&replay_dir).unwrap();

        let ctx = CorrelationContext {
            session_id: Some("test-session".to_string()),
            run_id: None,
            symbol: None,
            venue: None,
            strategy_id: None,
            decision_id: None,
            order_id: None,
        };
        let decision = make_test_decision(ctx);

        // Create valid sessions
        create_test_session(&live_dir, &decision);
        create_test_session(&replay_dir, &decision);

        // Corrupt the live decisions file after manifest was created
        let live_decisions_path = live_dir.join("wal/decisions.jsonl");
        std::fs::write(&live_decisions_path, "CORRUPTED CONTENT").unwrap();

        // Run G1 comparison
        let g1 = G1ReplayParity::new(G1Config::default());
        let result = g1.compare_sessions(&live_dir, &replay_dir).unwrap();

        // Should fail due to digest mismatch
        assert!(
            !result.passed,
            "Expected checks to fail due to digest mismatch"
        );

        // Find the failing check
        let failing_check = result.checks.iter().find(|c| !c.passed);
        assert!(
            failing_check.is_some(),
            "Expected at least one failing check"
        );
        let failing = failing_check.unwrap();
        assert!(
            failing.message.contains("digest") || failing.name.contains("digest"),
            "Expected digest-related failure, got: {} - {}",
            failing.name,
            failing.message
        );
    }
}
