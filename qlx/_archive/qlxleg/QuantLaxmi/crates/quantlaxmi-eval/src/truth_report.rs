//! Phase 26.2: Truth Report Builder
//!
//! Pure builder that converts aggregator output + session metadata into:
//! - Deterministic JSON report struct
//! - Deterministic text summary string
//!
//! ## Invariants (Frozen v1)
//! - No file I/O (caller writes files in Phase 26.3)
//! - Deterministic: BTreeMap ordering, canonical JSON
//! - All i128 values serialized as decimal strings
//! - Digest = SHA-256 of canonical JSON with digest field empty

use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::strategy_aggregator::StrategyAccumulator;

// =============================================================================
// Constants
// =============================================================================

/// Schema version for truth reports (frozen v1).
pub const TRUTH_REPORT_SCHEMA_VERSION: &str = "1";

// =============================================================================
// Session Metadata
// =============================================================================

/// Metadata about the session being reported.
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    pub session_id: String,
    pub instrument: String,
    pub start_ts_ns: i64,
    pub end_ts_ns: i64,
    pub latency_ticks: u32,
    pub cost_model_digest: Option<String>,
    pub unified_exponent: i8,
}

// =============================================================================
// Report Structs
// =============================================================================

/// The complete strategy truth report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrategyTruthReport {
    pub schema_version: String,
    pub session_id: String,
    pub instrument: String,
    pub latency_ticks: u32,
    pub cost_model_digest: Option<String>,
    pub unified_exponent: i8,
    pub period: ReportPeriod,
    pub strategies: BTreeMap<String, StrategyMetrics>,
    pub digest: String,
}

/// Time period covered by the report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportPeriod {
    pub start_ts_ns: i64,
    pub end_ts_ns: i64,
}

/// Per-strategy metrics derived from accumulator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StrategyMetrics {
    // Trade stats
    pub trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub win_rate: String, // "0.53" format (2 decimal places)

    // PnL (i128 as decimal strings)
    pub gross_pnl_mantissa: String,
    pub net_pnl_mantissa: String,
    pub fees_mantissa: String,

    // Risk
    pub max_drawdown_mantissa: String,
    pub avg_trade_pnl_mantissa: String,

    // Exposure
    pub exposure_updates: u64,

    // Time bounds
    pub first_trade_ts_ns: Option<i64>,
    pub last_trade_ts_ns: Option<i64>,
}

// =============================================================================
// StrategyMetrics Implementation
// =============================================================================

impl StrategyMetrics {
    /// Build metrics from a StrategyAccumulator.
    pub fn from_accumulator(acc: &StrategyAccumulator) -> Self {
        let win_rate = Self::format_win_rate(acc.winning_trades, acc.losing_trades);

        Self {
            trades: acc.trade_count,
            winning_trades: acc.winning_trades,
            losing_trades: acc.losing_trades,
            win_rate,
            gross_pnl_mantissa: acc.gross_pnl_mantissa().to_string(),
            net_pnl_mantissa: acc.net_pnl_mantissa().to_string(),
            fees_mantissa: acc.fees_mantissa.to_string(),
            max_drawdown_mantissa: acc.max_drawdown_mantissa.to_string(),
            avg_trade_pnl_mantissa: acc.avg_trade_pnl_mantissa().to_string(),
            exposure_updates: acc.exposure_updates,
            first_trade_ts_ns: acc.first_ts_ns,
            last_trade_ts_ns: acc.last_ts_ns,
        }
    }

    /// Format win rate as "0.XX" (2 decimal places).
    fn format_win_rate(wins: u64, losses: u64) -> String {
        let denom = wins + losses;
        if denom == 0 {
            "0.00".to_string()
        } else {
            let rate = wins as f64 / denom as f64;
            format!("{:.2}", rate)
        }
    }
}

// =============================================================================
// StrategyTruthReport Implementation
// =============================================================================

impl StrategyTruthReport {
    /// Build a complete report from session metadata and accumulators.
    ///
    /// Build procedure:
    /// 1. Convert accumulators → metrics map
    /// 2. Create report with digest = ""
    /// 3. Compute digest using canonical JSON with empty digest
    /// 4. Set digest
    /// 5. Return report
    pub fn build(
        metadata: SessionMetadata,
        accumulators: BTreeMap<String, StrategyAccumulator>,
    ) -> Self {
        // Step 1: Convert accumulators to metrics
        let strategies: BTreeMap<String, StrategyMetrics> = accumulators
            .iter()
            .map(|(id, acc)| (id.clone(), StrategyMetrics::from_accumulator(acc)))
            .collect();

        // Step 2: Create report with empty digest
        let mut report = Self {
            schema_version: TRUTH_REPORT_SCHEMA_VERSION.to_string(),
            session_id: metadata.session_id,
            instrument: metadata.instrument,
            latency_ticks: metadata.latency_ticks,
            cost_model_digest: metadata.cost_model_digest,
            unified_exponent: metadata.unified_exponent,
            period: ReportPeriod {
                start_ts_ns: metadata.start_ts_ns,
                end_ts_ns: metadata.end_ts_ns,
            },
            strategies,
            digest: String::new(),
        };

        // Step 3 & 4: Compute and set digest
        report.digest = report.compute_digest_hex();

        report
    }

    /// Serialize to canonical JSON with digest field set to empty string.
    ///
    /// Used for digest computation.
    pub fn to_canonical_json_with_empty_digest(&self) -> String {
        // Create a clone with empty digest for canonical serialization
        let mut canonical = self.clone();
        canonical.digest = String::new();

        // serde_json with BTreeMap produces stable key order
        serde_json::to_string(&canonical).expect("serialization should not fail")
    }

    /// Compute SHA-256 digest of canonical JSON (with empty digest field).
    pub fn compute_digest_hex(&self) -> String {
        let canonical_json = self.to_canonical_json_with_empty_digest();
        let canonical_bytes = canonical_json.as_bytes();

        let mut hasher = Sha256::new();
        hasher.update(canonical_bytes);
        let result = hasher.finalize();

        hex::encode(result)
    }

    /// Serialize to JSON (includes computed digest).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self).expect("serialization should not fail")
    }

    /// Generate deterministic text summary.
    ///
    /// Uses raw ts_ns values (no datetime formatting in v1).
    /// PnL shown as mantissa + exponent.
    pub fn to_text_summary(&self) -> String {
        let mut out = String::new();

        // Header
        writeln!(
            out,
            "================================================================================"
        )
        .unwrap();
        writeln!(out, "STRATEGY TRUTH REPORT").unwrap();
        writeln!(
            out,
            "================================================================================"
        )
        .unwrap();
        writeln!(out, "Session:    {}", self.session_id).unwrap();
        writeln!(out, "Instrument: {}", self.instrument).unwrap();
        writeln!(
            out,
            "Period:     start_ts_ns={} end_ts_ns={}",
            self.period.start_ts_ns, self.period.end_ts_ns
        )
        .unwrap();
        writeln!(out, "Latency:    {} tick(s)", self.latency_ticks).unwrap();
        writeln!(
            out,
            "Cost Model: {}",
            self.cost_model_digest.as_deref().unwrap_or("none")
        )
        .unwrap();
        writeln!(out, "Exponent:   {}", self.unified_exponent).unwrap();
        writeln!(out).unwrap();

        // Per-strategy sections
        for (strategy_id, metrics) in &self.strategies {
            writeln!(
                out,
                "--------------------------------------------------------------------------------"
            )
            .unwrap();
            writeln!(out, "STRATEGY: {}", strategy_id).unwrap();
            writeln!(
                out,
                "--------------------------------------------------------------------------------"
            )
            .unwrap();
            writeln!(
                out,
                "  Trades:       {} ({} wins / {} losses)",
                metrics.trades, metrics.winning_trades, metrics.losing_trades
            )
            .unwrap();
            writeln!(out, "  Win Rate:     {}", metrics.win_rate).unwrap();
            writeln!(
                out,
                "  Gross PnL:    mantissa={} exponent={}",
                metrics.gross_pnl_mantissa, self.unified_exponent
            )
            .unwrap();
            writeln!(
                out,
                "  Net PnL:      mantissa={} exponent={}",
                metrics.net_pnl_mantissa, self.unified_exponent
            )
            .unwrap();
            writeln!(
                out,
                "  Fees:         mantissa={} exponent={}",
                metrics.fees_mantissa, self.unified_exponent
            )
            .unwrap();
            writeln!(
                out,
                "  Max Drawdown: mantissa={} exponent={}",
                metrics.max_drawdown_mantissa, self.unified_exponent
            )
            .unwrap();
            writeln!(
                out,
                "  Avg Trade:    mantissa={} exponent={}",
                metrics.avg_trade_pnl_mantissa, self.unified_exponent
            )
            .unwrap();
            writeln!(out, "  Exposure:     {} updates", metrics.exposure_updates).unwrap();
            writeln!(
                out,
                "  First Trade:  ts_ns={}",
                metrics
                    .first_trade_ts_ns
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "none".to_string())
            )
            .unwrap();
            writeln!(
                out,
                "  Last Trade:   ts_ns={}",
                metrics
                    .last_trade_ts_ns
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "none".to_string())
            )
            .unwrap();
            writeln!(out).unwrap();
        }

        // Footer
        writeln!(
            out,
            "--------------------------------------------------------------------------------"
        )
        .unwrap();
        writeln!(out, "Report Digest: {}", self.digest).unwrap();
        writeln!(
            out,
            "================================================================================"
        )
        .unwrap();

        out
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy_aggregator::StrategyAccumulator;

    fn make_test_metadata() -> SessionMetadata {
        SessionMetadata {
            session_id: "test_session_123".to_string(),
            instrument: "BTCUSDT".to_string(),
            start_ts_ns: 1706500000000000000,
            end_ts_ns: 1706600000000000000,
            latency_ticks: 1,
            cost_model_digest: Some("abc123def456".to_string()),
            unified_exponent: -8,
        }
    }

    fn make_test_accumulator(
        strategy_id: &str,
        trades: u64,
        wins: u64,
        pnl: i128,
    ) -> StrategyAccumulator {
        let mut acc = StrategyAccumulator::new(strategy_id.to_string());
        acc.trade_count = trades;
        acc.winning_trades = wins;
        acc.losing_trades = trades - wins;
        acc.gross_realized_pnl_mantissa = pnl;
        acc.equity_mantissa = pnl - 100; // net = gross - some costs
        acc.fees_mantissa = 50;
        acc.max_drawdown_mantissa = 200;
        acc.exposure_updates = trades * 2;
        acc.first_ts_ns = Some(1706500100000000000);
        acc.last_ts_ns = Some(1706599900000000000);
        acc
    }

    #[test]
    fn test_build_single_strategy() {
        let metadata = make_test_metadata();
        let mut accumulators = BTreeMap::new();
        accumulators.insert(
            "strat_a".to_string(),
            make_test_accumulator("strat_a", 10, 6, 1000),
        );

        let report = StrategyTruthReport::build(metadata, accumulators);

        assert_eq!(report.schema_version, "1");
        assert_eq!(report.session_id, "test_session_123");
        assert_eq!(report.strategies.len(), 1);

        let metrics = report.strategies.get("strat_a").unwrap();
        assert_eq!(metrics.trades, 10);
        assert_eq!(metrics.winning_trades, 6);
        assert_eq!(metrics.win_rate, "0.60");
        assert_eq!(metrics.gross_pnl_mantissa, "1000");
        assert_eq!(metrics.net_pnl_mantissa, "900"); // 1000 - 100

        // Digest should be non-empty
        assert!(!report.digest.is_empty());
        assert_eq!(report.digest.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_build_multi_strategy_sorted_order() {
        let metadata = make_test_metadata();
        let mut accumulators = BTreeMap::new();
        accumulators.insert(
            "zebra_strat".to_string(),
            make_test_accumulator("zebra_strat", 5, 3, 500),
        );
        accumulators.insert(
            "alpha_strat".to_string(),
            make_test_accumulator("alpha_strat", 8, 4, 800),
        );
        accumulators.insert(
            "beta_strat".to_string(),
            make_test_accumulator("beta_strat", 12, 7, 1200),
        );

        let report = StrategyTruthReport::build(metadata, accumulators);

        // Verify BTreeMap order (lexicographic)
        let keys: Vec<&String> = report.strategies.keys().collect();
        assert_eq!(keys, vec!["alpha_strat", "beta_strat", "zebra_strat"]);
    }

    #[test]
    fn test_digest_stable_same_inputs() {
        let metadata = make_test_metadata();
        let mut accumulators = BTreeMap::new();
        accumulators.insert(
            "strat_a".to_string(),
            make_test_accumulator("strat_a", 10, 6, 1000),
        );

        // Build twice with same inputs
        let report1 = StrategyTruthReport::build(metadata.clone(), accumulators.clone());
        let report2 = StrategyTruthReport::build(metadata, accumulators);

        assert_eq!(report1.digest, report2.digest);
        assert_eq!(report1, report2);
    }

    #[test]
    fn test_digest_changes_with_content() {
        let metadata = make_test_metadata();

        let mut acc1 = BTreeMap::new();
        acc1.insert(
            "strat_a".to_string(),
            make_test_accumulator("strat_a", 10, 6, 1000),
        );

        let mut acc2 = BTreeMap::new();
        acc2.insert(
            "strat_a".to_string(),
            make_test_accumulator("strat_a", 10, 6, 1001), // Different PnL
        );

        let report1 = StrategyTruthReport::build(metadata.clone(), acc1);
        let report2 = StrategyTruthReport::build(metadata, acc2);

        assert_ne!(report1.digest, report2.digest);
    }

    #[test]
    fn test_win_rate_formatting() {
        // 0 trades
        assert_eq!(StrategyMetrics::format_win_rate(0, 0), "0.00");

        // 50%
        assert_eq!(StrategyMetrics::format_win_rate(5, 5), "0.50");

        // 66.67%
        assert_eq!(StrategyMetrics::format_win_rate(2, 1), "0.67");

        // 100%
        assert_eq!(StrategyMetrics::format_win_rate(10, 0), "1.00");

        // 0%
        assert_eq!(StrategyMetrics::format_win_rate(0, 10), "0.00");
    }

    #[test]
    fn test_i128_string_fields() {
        let mut acc = StrategyAccumulator::new("test".to_string());
        acc.gross_realized_pnl_mantissa = 123456789012345678901234567890_i128;
        acc.equity_mantissa = -98765432109876543210_i128;
        acc.fees_mantissa = 999999999999_i128;

        let metrics = StrategyMetrics::from_accumulator(&acc);

        // Should be decimal strings, no scientific notation
        assert_eq!(metrics.gross_pnl_mantissa, "123456789012345678901234567890");
        assert_eq!(metrics.net_pnl_mantissa, "-98765432109876543210");
        assert_eq!(metrics.fees_mantissa, "999999999999");
    }

    #[test]
    fn test_text_summary_contains_expected_fields() {
        let metadata = make_test_metadata();
        let mut accumulators = BTreeMap::new();
        accumulators.insert(
            "my_strategy".to_string(),
            make_test_accumulator("my_strategy", 10, 6, 1000),
        );

        let report = StrategyTruthReport::build(metadata, accumulators);
        let summary = report.to_text_summary();

        // Check header
        assert!(summary.contains("STRATEGY TRUTH REPORT"));
        assert!(summary.contains("Session:    test_session_123"));
        assert!(summary.contains("Instrument: BTCUSDT"));
        assert!(summary.contains("Latency:    1 tick(s)"));
        assert!(summary.contains("Cost Model: abc123def456"));
        assert!(summary.contains("Exponent:   -8"));

        // Check period (raw ts_ns)
        assert!(summary.contains("start_ts_ns=1706500000000000000"));
        assert!(summary.contains("end_ts_ns=1706600000000000000"));

        // Check strategy section
        assert!(summary.contains("STRATEGY: my_strategy"));
        assert!(summary.contains("Trades:       10 (6 wins / 4 losses)"));
        assert!(summary.contains("Win Rate:     0.60"));
        assert!(summary.contains("Gross PnL:    mantissa=1000 exponent=-8"));
        assert!(summary.contains("Net PnL:      mantissa=900 exponent=-8"));
        assert!(summary.contains("Fees:         mantissa=50 exponent=-8"));
        assert!(summary.contains("Exposure:     20 updates"));

        // Check footer
        assert!(summary.contains("Report Digest:"));
    }

    #[test]
    fn test_json_roundtrip() {
        let metadata = make_test_metadata();
        let mut accumulators = BTreeMap::new();
        accumulators.insert(
            "strat_a".to_string(),
            make_test_accumulator("strat_a", 10, 6, 1000),
        );

        let report = StrategyTruthReport::build(metadata, accumulators);
        let json = report.to_json();

        // Parse back
        let parsed: StrategyTruthReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, parsed);
    }

    #[test]
    fn test_empty_strategies() {
        let metadata = make_test_metadata();
        let accumulators = BTreeMap::new();

        let report = StrategyTruthReport::build(metadata, accumulators);

        assert!(report.strategies.is_empty());
        assert!(!report.digest.is_empty()); // Still has a digest
    }
}
