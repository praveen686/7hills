//! # G2 BacktestCorrectness Gate (Scaffold)
//!
//! Validates backtest assumptions and methodology.
//!
//! ## Checks (To Be Implemented)
//! - **No Lookahead**: Decisions use only past data
//! - **Fill Realism**: Fills respect liquidity and slippage
//! - **Transaction Costs**: Fees and spread are modeled
//! - **Market Impact**: Large orders affect price
//!
//! ## Implementation Notes
//! This gate validates that backtests don't have common biases:
//! - Lookahead bias (using future data)
//! - Survivorship bias (only testing on surviving assets)
//! - Overfitting (too many parameters for data)
//!
//! ## Usage (Future)
//! ```ignore
//! let g2 = G2BacktestCorrectness::new(config);
//! let result = g2.validate_backtest(&backtest_run)?;
//! ```

use crate::{CheckResult, GateError, GateResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

/// G2 BacktestCorrectness configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct G2Config {
    /// Require explicit transaction cost modeling
    #[serde(default = "default_true")]
    pub require_transaction_costs: bool,

    /// Minimum slippage in basis points
    #[serde(default = "default_min_slippage_bps")]
    pub min_slippage_bps: f64,

    /// Maximum fill rate as fraction of volume
    #[serde(default = "default_max_fill_rate")]
    pub max_fill_rate: f64,

    /// Require market impact modeling for large orders
    #[serde(default)]
    pub require_market_impact: bool,

    /// Large order threshold as fraction of ADV
    #[serde(default = "default_large_order_threshold")]
    pub large_order_threshold: f64,
}

fn default_true() -> bool { true }
fn default_min_slippage_bps() -> f64 { 1.0 } // 0.01%
fn default_max_fill_rate() -> f64 { 0.10 } // 10% of volume
fn default_large_order_threshold() -> f64 { 0.01 } // 1% of ADV

/// G2 BacktestCorrectness gate validator (scaffold).
pub struct G2BacktestCorrectness {
    #[allow(dead_code)]
    config: G2Config,
}

impl G2BacktestCorrectness {
    /// Create a new G2 validator.
    pub fn new(config: G2Config) -> Self {
        Self { config }
    }

    /// Validate a backtest run.
    ///
    /// # Arguments
    /// * `backtest_dir` - Path to backtest output directory
    ///
    /// # Returns
    /// Gate result with correctness checks
    pub fn validate_backtest(&self, backtest_dir: &Path) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G2_BacktestCorrectness");

        info!(
            backtest = %backtest_dir.display(),
            "Starting G2 BacktestCorrectness validation (scaffold)"
        );

        // Scaffold: Return placeholder checks
        result.add_check(CheckResult::pass(
            "no_lookahead",
            "SCAFFOLD: Lookahead bias check not yet implemented",
        ));

        result.add_check(CheckResult::pass(
            "fill_realism",
            "SCAFFOLD: Fill realism check not yet implemented",
        ));

        result.add_check(CheckResult::pass(
            "transaction_costs",
            "SCAFFOLD: Transaction cost check not yet implemented",
        ));

        result.add_check(CheckResult::pass(
            "market_impact",
            "SCAFFOLD: Market impact check not yet implemented",
        ));

        result.add_check(CheckResult::pass(
            "data_quality",
            "SCAFFOLD: Data quality check not yet implemented",
        ));

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "SCAFFOLD: {}/{} checks passed (not implemented)",
            result.passed_count(),
            result.checks.len()
        );

        Ok(result)
    }

    /// Placeholder for lookahead bias detection.
    #[allow(dead_code)]
    fn check_lookahead(&self, _decisions: &[quantlaxmi_models::events::DecisionEvent]) -> CheckResult {
        // TODO: Implement lookahead detection
        // - For each decision, verify all referenced data has ts < decision.ts
        // - Check that market_snapshot.book_ts_ns < decision.ts
        // - Flag any decisions that reference future prices
        CheckResult::pass("no_lookahead", "SCAFFOLD")
    }

    /// Placeholder for fill realism validation.
    #[allow(dead_code)]
    fn check_fill_realism(&self, _fills: &[quantlaxmi_models::FillEvent]) -> CheckResult {
        // TODO: Implement fill realism
        // - Verify fill prices are within bid-ask at fill time
        // - Verify fill quantities don't exceed available liquidity
        // - Check fill rate against volume
        CheckResult::pass("fill_realism", "SCAFFOLD")
    }

    /// Placeholder for transaction cost validation.
    #[allow(dead_code)]
    fn check_transaction_costs(&self, _fills: &[quantlaxmi_models::FillEvent]) -> CheckResult {
        // TODO: Implement transaction cost check
        // - Verify commission is non-zero
        // - Verify slippage is at least min_slippage_bps
        // - Check spread crossing costs
        CheckResult::pass("transaction_costs", "SCAFFOLD")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_g2_scaffold() {
        let dir = tempdir().unwrap();

        let g2 = G2BacktestCorrectness::new(G2Config::default());
        let result = g2.validate_backtest(dir.path()).unwrap();

        // Scaffold should pass with placeholder checks
        assert!(result.passed);
        assert!(result.summary.contains("SCAFFOLD"));
    }
}
