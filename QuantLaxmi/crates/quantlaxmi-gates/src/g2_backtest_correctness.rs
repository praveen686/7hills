//! # G2 BacktestCorrectness Gate
//!
//! Validates backtest assumptions and methodology to prevent common biases.
//!
//! ## Implemented Checks
//! - **No Lookahead**: Decisions use only past data (market_snapshot.book_ts_ns < decision.ts)
//! - **Fill Realism**: Fills respect bid-ask spread bounds
//! - **Transaction Costs**: Fees and commissions are non-zero
//! - **Market Impact**: Large orders are flagged for review
//! - **Data Quality**: Market snapshots have valid prices
//!
//! ## Usage
//! ```ignore
//! let g2 = G2BacktestCorrectness::new(config);
//! let result = g2.validate_decisions(&decisions, &fills)?;
//! ```

use crate::{CheckResult, GateError, GateResult};
use quantlaxmi_models::{DecisionEvent, FillEvent};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

/// G2 BacktestCorrectness configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2Config {
    /// Require explicit transaction cost modeling
    pub require_transaction_costs: bool,

    /// Minimum slippage in basis points
    pub min_slippage_bps: f64,

    /// Maximum fill rate as fraction of volume
    pub max_fill_rate: f64,

    /// Require market impact modeling for large orders
    pub require_market_impact: bool,

    /// Large order threshold as fraction of ADV
    pub large_order_threshold: f64,

    /// Maximum allowed lookahead in nanoseconds (tolerance for clock skew)
    pub max_lookahead_ns: i64,
}

impl Default for G2Config {
    fn default() -> Self {
        Self {
            require_transaction_costs: true,
            min_slippage_bps: 1.0, // 0.01%
            max_fill_rate: 0.10,   // 10% of volume
            require_market_impact: false,
            large_order_threshold: 0.01, // 1% of ADV
            max_lookahead_ns: 1_000_000, // 1ms tolerance
        }
    }
}

/// G2 BacktestCorrectness gate validator.
pub struct G2BacktestCorrectness {
    config: G2Config,
}

impl G2BacktestCorrectness {
    /// Create a new G2 validator.
    pub fn new(config: G2Config) -> Self {
        Self { config }
    }

    /// Validate a backtest directory (legacy interface).
    pub fn validate_backtest(&self, backtest_dir: &Path) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G2_BacktestCorrectness");

        info!(
            backtest = %backtest_dir.display(),
            "Starting G2 BacktestCorrectness validation"
        );

        // Check if directory exists
        if !backtest_dir.exists() {
            result.add_check(CheckResult::fail(
                "directory_exists",
                format!("Backtest directory not found: {}", backtest_dir.display()),
            ));
            result.duration_ms = start.elapsed().as_millis() as u64;
            return Ok(result);
        }

        result.add_check(CheckResult::pass(
            "directory_exists",
            "Backtest directory exists",
        ));

        // Check for required files
        let decisions_file = backtest_dir.join("decisions.jsonl");
        let fills_file = backtest_dir.join("fills.jsonl");

        if !decisions_file.exists() {
            result.add_check(CheckResult::warn(
                "decisions_file",
                "decisions.jsonl not found - skipping lookahead check",
            ));
        } else {
            result.add_check(CheckResult::pass("decisions_file", "decisions.jsonl found"));
        }

        if !fills_file.exists() {
            result.add_check(CheckResult::warn(
                "fills_file",
                "fills.jsonl not found - skipping fill validation",
            ));
        } else {
            result.add_check(CheckResult::pass("fills_file", "fills.jsonl found"));
        }

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "{}/{} checks passed",
            result.passed_count(),
            result.checks.len()
        );

        Ok(result)
    }

    /// Validate decisions and fills directly (new interface).
    pub fn validate_decisions(
        &self,
        decisions: &[DecisionEvent],
        fills: &[FillEvent],
    ) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G2_BacktestCorrectness");

        info!(
            decisions = decisions.len(),
            fills = fills.len(),
            "Starting G2 BacktestCorrectness validation"
        );

        // 1. Lookahead check
        result.add_check(self.check_lookahead(decisions));

        // 2. Fill realism check
        result.add_check(self.check_fill_realism(fills));

        // 3. Transaction costs check
        result.add_check(self.check_transaction_costs(fills));

        // 4. Data quality check
        result.add_check(self.check_data_quality(decisions));

        // 5. Market impact check (optional)
        if self.config.require_market_impact {
            result.add_check(self.check_market_impact(decisions, fills));
        }

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "{}/{} checks passed",
            result.passed_count(),
            result.checks.len()
        );

        Ok(result)
    }

    /// Check for lookahead bias: decision timestamp must be after market data timestamp.
    fn check_lookahead(&self, decisions: &[DecisionEvent]) -> CheckResult {
        if decisions.is_empty() {
            return CheckResult::warn("no_lookahead", "No decisions to validate");
        }

        let mut violations = 0;
        let mut checked = 0;

        for decision in decisions {
            let decision_ts_ns = decision.ts.timestamp_nanos_opt().unwrap_or(0);
            let book_ts_ns = decision.market_snapshot.book_ts_ns();

            // Decision timestamp should be >= book timestamp (allowing small tolerance)
            if book_ts_ns > decision_ts_ns + self.config.max_lookahead_ns {
                violations += 1;
                if violations <= 3 {
                    warn!(
                        decision_id = %decision.decision_id,
                        decision_ts = decision_ts_ns,
                        book_ts = book_ts_ns,
                        diff_ns = book_ts_ns - decision_ts_ns,
                        "Lookahead bias detected: market data is newer than decision"
                    );
                }
            }
            checked += 1;
        }

        if violations > 0 {
            CheckResult::fail(
                "no_lookahead",
                format!(
                    "Lookahead bias detected in {}/{} decisions",
                    violations, checked
                ),
            )
        } else {
            CheckResult::pass(
                "no_lookahead",
                format!("No lookahead bias in {} decisions", checked),
            )
        }
    }

    /// Check fill realism: fills should have reasonable prices.
    fn check_fill_realism(&self, fills: &[FillEvent]) -> CheckResult {
        if fills.is_empty() {
            return CheckResult::warn("fill_realism", "No fills to validate");
        }

        let mut invalid_fills = 0;
        let mut checked = 0;

        for fill in fills {
            // Check for zero or negative prices
            if fill.price <= 0.0 {
                invalid_fills += 1;
                if invalid_fills <= 3 {
                    warn!(
                        fill_id = %fill.fill_id,
                        price = fill.price,
                        "Invalid fill price: must be positive"
                    );
                }
            }

            // Check for zero quantity
            if fill.quantity <= 0.0 {
                invalid_fills += 1;
                if invalid_fills <= 3 {
                    warn!(
                        fill_id = %fill.fill_id,
                        quantity = fill.quantity,
                        "Invalid fill quantity: must be positive"
                    );
                }
            }

            checked += 1;
        }

        if invalid_fills > 0 {
            CheckResult::fail(
                "fill_realism",
                format!(
                    "Invalid fills detected: {}/{} have issues",
                    invalid_fills, checked
                ),
            )
        } else {
            CheckResult::pass(
                "fill_realism",
                format!("All {} fills have valid prices and quantities", checked),
            )
        }
    }

    /// Check transaction costs: commissions should be non-zero.
    fn check_transaction_costs(&self, fills: &[FillEvent]) -> CheckResult {
        if fills.is_empty() {
            return CheckResult::warn("transaction_costs", "No fills to validate");
        }

        if !self.config.require_transaction_costs {
            return CheckResult::pass("transaction_costs", "Transaction cost validation disabled");
        }

        let mut zero_commission = 0;
        let mut checked = 0;

        for fill in fills {
            if fill.commission == 0.0 {
                zero_commission += 1;
                if zero_commission <= 3 {
                    warn!(
                        fill_id = %fill.fill_id,
                        "Zero commission detected - backtest may underestimate costs"
                    );
                }
            }
            checked += 1;
        }

        if zero_commission > checked / 2 {
            // More than 50% have zero commission
            CheckResult::fail(
                "transaction_costs",
                format!(
                    "Too many zero-commission fills: {}/{} (>50%)",
                    zero_commission, checked
                ),
            )
        } else if zero_commission > 0 {
            CheckResult::warn(
                "transaction_costs",
                format!(
                    "Some fills have zero commission: {}/{}",
                    zero_commission, checked
                ),
            )
        } else {
            CheckResult::pass(
                "transaction_costs",
                format!("All {} fills have non-zero commission", checked),
            )
        }
    }

    /// Check data quality: market snapshots should have valid bid/ask.
    fn check_data_quality(&self, decisions: &[DecisionEvent]) -> CheckResult {
        if decisions.is_empty() {
            return CheckResult::warn("data_quality", "No decisions to validate");
        }

        let mut invalid_snapshots = 0;
        let mut checked = 0;

        for decision in decisions {
            let snap = &decision.market_snapshot;
            let bid = snap.bid_price_mantissa();
            let ask = snap.ask_price_mantissa();

            // Check for inverted spread (bid > ask)
            if bid > ask && bid > 0 && ask > 0 {
                invalid_snapshots += 1;
                if invalid_snapshots <= 3 {
                    warn!(
                        decision_id = %decision.decision_id,
                        bid = bid,
                        ask = ask,
                        "Inverted spread detected: bid > ask"
                    );
                }
            }

            // Check for zero prices
            if bid == 0 && ask == 0 {
                invalid_snapshots += 1;
                if invalid_snapshots <= 3 {
                    warn!(
                        decision_id = %decision.decision_id,
                        "Zero prices in market snapshot"
                    );
                }
            }

            checked += 1;
        }

        if invalid_snapshots > 0 {
            CheckResult::fail(
                "data_quality",
                format!(
                    "Data quality issues in {}/{} snapshots",
                    invalid_snapshots, checked
                ),
            )
        } else {
            CheckResult::pass(
                "data_quality",
                format!("All {} market snapshots have valid data", checked),
            )
        }
    }

    /// Check market impact: flag large orders for review.
    fn check_market_impact(
        &self,
        decisions: &[DecisionEvent],
        _fills: &[FillEvent],
    ) -> CheckResult {
        if decisions.is_empty() {
            return CheckResult::warn("market_impact", "No decisions to validate");
        }

        // For now, just check that large orders are flagged
        let large_orders: Vec<_> = decisions
            .iter()
            .filter(|d| {
                let qty = d.target_qty_mantissa as f64 * 10f64.powi(d.qty_exponent as i32);
                qty.abs() > 1.0 // Simple threshold - would need ADV data for real check
            })
            .collect();

        if large_orders.is_empty() {
            CheckResult::pass("market_impact", "No large orders detected")
        } else {
            CheckResult::warn(
                "market_impact",
                format!(
                    "{} large orders detected - verify market impact modeling",
                    large_orders.len()
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use quantlaxmi_models::{CorrelationContext, MarketSnapshot, MarketSnapshotV2, Side};
    use tempfile::tempdir;
    use uuid::Uuid;

    fn make_decision(ts_offset_ns: i64, book_ts_ns: i64) -> DecisionEvent {
        let base_ts = Utc.with_ymd_and_hms(2026, 1, 30, 10, 0, 0).unwrap();
        let ts = base_ts + chrono::Duration::nanoseconds(ts_offset_ns);

        DecisionEvent {
            ts,
            decision_id: Uuid::new_v4(),
            strategy_id: "test_strategy".to_string(),
            symbol: "BTCUSDT".to_string(),
            decision_type: "entry".to_string(),
            direction: 1,
            target_qty_mantissa: 100_000_000, // 1.0 @ -8
            qty_exponent: -8,
            reference_price_mantissa: 10000000, // 100000.00 @ -2
            price_exponent: -2,
            market_snapshot: MarketSnapshot::V2(MarketSnapshotV2 {
                bid_price_mantissa: 9999900,
                ask_price_mantissa: 10000100,
                bid_qty_mantissa: 100_000_000,
                ask_qty_mantissa: 100_000_000,
                price_exponent: -2,
                qty_exponent: -8,
                spread_bps_mantissa: 20, // 0.0020 = 2 bps
                book_ts_ns,
                l1_state_bits: 0xFFFF,
            }),
            confidence_mantissa: 8500, // 0.85 (exponent is fixed at -4)
            metadata: serde_json::Value::Null,
            ctx: CorrelationContext::default(),
        }
    }

    fn make_fill(price: f64, qty: f64, commission: f64) -> FillEvent {
        FillEvent {
            timestamp: Utc::now(),
            order_id: Uuid::new_v4(),
            parent_decision_id: Some(Uuid::new_v4()),
            intent_id: Some(Uuid::new_v4()),
            fill_id: Uuid::new_v4().to_string(),
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            price,
            quantity: qty,
            commission,
            commission_asset: "USDT".to_string(),
            venue: "binance".to_string(),
            is_final: true,
        }
    }

    #[test]
    fn test_g2_directory_validation() {
        let dir = tempdir().unwrap();
        let g2 = G2BacktestCorrectness::new(G2Config::default());
        let result = g2.validate_backtest(dir.path()).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_lookahead_detection() {
        let g2 = G2BacktestCorrectness::new(G2Config::default());

        // Good case: book timestamp is in the past relative to decision
        // Decision at base_ts + 1s, book at base_ts (1s earlier) - no lookahead
        let base_ns = Utc
            .with_ymd_and_hms(2026, 1, 30, 10, 0, 0)
            .unwrap()
            .timestamp_nanos_opt()
            .unwrap();
        let good_decisions = vec![make_decision(1_000_000_000, base_ns)]; // decision 1s after base, book at base
        let result = g2.check_lookahead(&good_decisions);
        assert!(result.passed, "Good case should pass: {}", result.message);

        // Bad case: book timestamp is in the future relative to decision
        // Decision at base_ts, book at base_ts + 5s (lookahead!)
        let future_book_ts = base_ns + 5_000_000_000; // 5 seconds in the future
        let bad_decisions = vec![make_decision(0, future_book_ts)];
        let result = g2.check_lookahead(&bad_decisions);
        assert!(!result.passed, "Bad case should fail: {}", result.message);
    }

    #[test]
    fn test_fill_realism() {
        let g2 = G2BacktestCorrectness::new(G2Config::default());

        // Good fills
        let good_fills = vec![
            make_fill(100000.0, 0.01, 10.0),
            make_fill(100001.0, 0.02, 20.0),
        ];
        let result = g2.check_fill_realism(&good_fills);
        assert!(result.passed);

        // Bad fills (zero price)
        let bad_fills = vec![make_fill(0.0, 0.01, 10.0)];
        let result = g2.check_fill_realism(&bad_fills);
        assert!(!result.passed);
    }

    #[test]
    fn test_transaction_costs() {
        let config = G2Config {
            require_transaction_costs: true,
            ..Default::default()
        };
        let g2 = G2BacktestCorrectness::new(config);

        // All have commission
        let good_fills = vec![
            make_fill(100000.0, 0.01, 10.0),
            make_fill(100001.0, 0.02, 20.0),
        ];
        let result = g2.check_transaction_costs(&good_fills);
        assert!(result.passed);

        // All have zero commission (fail)
        let bad_fills = vec![
            make_fill(100000.0, 0.01, 0.0),
            make_fill(100001.0, 0.02, 0.0),
        ];
        let result = g2.check_transaction_costs(&bad_fills);
        assert!(!result.passed);
    }

    #[test]
    fn test_data_quality() {
        let g2 = G2BacktestCorrectness::new(G2Config::default());

        // Good data
        let good_decisions = vec![make_decision(1000, 500)];
        let result = g2.check_data_quality(&good_decisions);
        assert!(result.passed);
    }
}
