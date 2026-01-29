//! Phase 26.1: Strategy Aggregator
//!
//! Pure computation module that consumes PositionUpdateRecord stream
//! and produces per-strategy metric accumulators.
//!
//! ## Invariants (Frozen v1)
//! - Pure computation: no I/O, no WAL writes
//! - Deterministic: BTreeMap for ordering, all math in i128
//! - Single unified exponent enforced across all records
//! - No invented components (no fake slippage/spread attribution)
//!
//! ## PnL Definitions (v1)
//! - gross_pnl = sum(realized_pnl_delta_mantissa)
//! - net_pnl = equity_mantissa (from cash_delta accumulation)
//! - fees = sum(fee_mantissa)

use std::collections::BTreeMap;

use quantlaxmi_models::PositionUpdateRecord;
use thiserror::Error;

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Error)]
pub enum AggregatorError {
    #[error("Exponent mismatch: expected {expected}, got {actual} for {field}")]
    ExponentMismatch {
        field: String,
        expected: i8,
        actual: i8,
    },

    #[error("Missing strategy_id in record")]
    MissingStrategyId,
}

// =============================================================================
// StrategyAccumulator
// =============================================================================

/// Per-strategy accumulator for evaluation metrics.
///
/// All monetary values stored as i128 mantissas with a unified exponent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrategyAccumulator {
    pub strategy_id: String,

    // Time bounds
    pub first_ts_ns: Option<i64>,
    pub last_ts_ns: Option<i64>,

    // Trade counting (based on position transitions)
    pub trade_count: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,

    // Position state tracking
    pub last_position_qty_mantissa: i64,

    // PnL components (all in unified exponent)
    /// sum(realized_pnl_delta_mantissa)
    pub gross_realized_pnl_mantissa: i128,
    /// sum(fee_mantissa)
    pub fees_mantissa: i128,
    /// Cumulative cash_delta (equity curve proxy)
    pub equity_mantissa: i128,

    // Drawdown (on equity curve)
    pub peak_equity_mantissa: i128,
    pub max_drawdown_mantissa: i128,

    // Exposure: counted per PositionUpdateRecord where position != 0
    pub exposure_updates: u64,
}

impl StrategyAccumulator {
    /// Create a new accumulator for the given strategy.
    pub fn new(strategy_id: String) -> Self {
        Self {
            strategy_id,
            first_ts_ns: None,
            last_ts_ns: None,
            trade_count: 0,
            winning_trades: 0,
            losing_trades: 0,
            last_position_qty_mantissa: 0,
            gross_realized_pnl_mantissa: 0,
            fees_mantissa: 0,
            equity_mantissa: 0,
            peak_equity_mantissa: 0,
            max_drawdown_mantissa: 0,
            exposure_updates: 0,
        }
    }

    /// Process a position update record.
    ///
    /// Updates:
    /// - Time bounds
    /// - Trade counting (on position close/flip)
    /// - Win/loss tracking
    /// - PnL components
    /// - Equity curve and drawdown
    /// - Exposure updates
    pub fn process_position_update(
        &mut self,
        ts_ns: i64,
        position_qty_mantissa: i64,
        cash_delta_mantissa: i64,
        realized_pnl_delta_mantissa: i64,
        fee_mantissa: Option<i64>,
    ) {
        // Update time bounds
        if self.first_ts_ns.is_none() {
            self.first_ts_ns = Some(ts_ns);
        }
        self.last_ts_ns = Some(ts_ns);

        // Trade counting: detect close or flip
        let prev_position = self.last_position_qty_mantissa;
        let new_position = position_qty_mantissa;

        if Self::is_trade_close(prev_position, new_position) {
            self.trade_count += 1;

            // Win/loss classification based on realized_pnl_delta
            if realized_pnl_delta_mantissa > 0 {
                self.winning_trades += 1;
            } else if realized_pnl_delta_mantissa < 0 {
                self.losing_trades += 1;
            }
            // == 0: neither winning nor losing
        }

        // Update position state
        self.last_position_qty_mantissa = new_position;

        // Accumulate PnL components
        self.gross_realized_pnl_mantissa += realized_pnl_delta_mantissa as i128;
        if let Some(fee) = fee_mantissa {
            self.fees_mantissa += fee as i128;
        }

        // Equity curve: accumulate cash_delta
        self.equity_mantissa += cash_delta_mantissa as i128;

        // Drawdown computation
        if self.equity_mantissa > self.peak_equity_mantissa {
            self.peak_equity_mantissa = self.equity_mantissa;
        }
        let current_drawdown = self.peak_equity_mantissa - self.equity_mantissa;
        if current_drawdown > self.max_drawdown_mantissa {
            self.max_drawdown_mantissa = current_drawdown;
        }

        // Exposure: count updates where position != 0
        if new_position != 0 {
            self.exposure_updates += 1;
        }
    }

    /// Determine if this position transition constitutes a "trade close".
    ///
    /// A trade close occurs when:
    /// - prev_position != 0 AND (new_position == 0 OR sign changes)
    fn is_trade_close(prev_position: i64, new_position: i64) -> bool {
        if prev_position == 0 {
            // Opening position, not closing
            false
        } else if new_position == 0 {
            // Full close
            true
        } else if prev_position.signum() != new_position.signum() {
            // Flip: close + open (counted as one close in v1)
            true
        } else {
            // Partial add/reduce, not a close
            false
        }
    }

    // =========================================================================
    // Derived Metrics
    // =========================================================================

    /// Net PnL = equity_mantissa (cash_delta accumulation including all costs).
    ///
    /// This is the safest invariant: reflects actual cash evolution.
    #[inline]
    pub fn net_pnl_mantissa(&self) -> i128 {
        self.equity_mantissa
    }

    /// Gross PnL = sum(realized_pnl_delta_mantissa).
    #[inline]
    pub fn gross_pnl_mantissa(&self) -> i128 {
        self.gross_realized_pnl_mantissa
    }

    /// Win rate = winning_trades / (winning_trades + losing_trades).
    ///
    /// Returns 0.0 if no trades closed.
    /// This is the only float return (convenience only).
    pub fn win_rate(&self) -> f64 {
        let total = self.winning_trades + self.losing_trades;
        if total == 0 {
            0.0
        } else {
            self.winning_trades as f64 / total as f64
        }
    }

    /// Average trade PnL = gross_realized_pnl / trade_count.
    ///
    /// Returns 0 if no trades.
    pub fn avg_trade_pnl_mantissa(&self) -> i128 {
        if self.trade_count == 0 {
            0
        } else {
            self.gross_realized_pnl_mantissa / self.trade_count as i128
        }
    }
}

// =============================================================================
// StrategyAggregatorRegistry
// =============================================================================

/// Registry of per-strategy accumulators.
///
/// Uses BTreeMap for deterministic iteration order.
pub struct StrategyAggregatorRegistry {
    accumulators: BTreeMap<String, StrategyAccumulator>,
    /// Unified exponent for all monetary values. Set on first record.
    unified_exponent: Option<i8>,
}

impl StrategyAggregatorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            accumulators: BTreeMap::new(),
            unified_exponent: None,
        }
    }

    /// Get the unified exponent (if set).
    pub fn unified_exponent(&self) -> Option<i8> {
        self.unified_exponent
    }

    /// Process a position update record.
    ///
    /// Routes to the appropriate strategy accumulator (creates if needed).
    /// Enforces exponent consistency.
    pub fn process_position_update(
        &mut self,
        record: &PositionUpdateRecord,
    ) -> Result<(), AggregatorError> {
        // Validate strategy_id
        if record.strategy_id.is_empty() {
            return Err(AggregatorError::MissingStrategyId);
        }

        // Enforce unified exponent
        let record_pnl_exp = record.pnl_exponent;
        let record_cash_exp = record.cash_exponent;
        let record_fee_exp = record.fee_exponent;

        // All exponents must match
        if record_cash_exp != record_pnl_exp {
            return Err(AggregatorError::ExponentMismatch {
                field: "cash_exponent vs pnl_exponent".to_string(),
                expected: record_pnl_exp,
                actual: record_cash_exp,
            });
        }
        if record_fee_exp != record_pnl_exp {
            return Err(AggregatorError::ExponentMismatch {
                field: "fee_exponent vs pnl_exponent".to_string(),
                expected: record_pnl_exp,
                actual: record_fee_exp,
            });
        }

        // Set or validate unified exponent
        match self.unified_exponent {
            None => {
                self.unified_exponent = Some(record_pnl_exp);
            }
            Some(expected) => {
                if record_pnl_exp != expected {
                    return Err(AggregatorError::ExponentMismatch {
                        field: "pnl_exponent".to_string(),
                        expected,
                        actual: record_pnl_exp,
                    });
                }
            }
        }

        // Get or create accumulator
        let accumulator = self
            .accumulators
            .entry(record.strategy_id.clone())
            .or_insert_with(|| StrategyAccumulator::new(record.strategy_id.clone()));

        // Process the update
        accumulator.process_position_update(
            record.ts_ns,
            record.position_qty_mantissa,
            record.cash_delta_mantissa,
            record.realized_pnl_delta_mantissa,
            record.fee_mantissa,
        );

        Ok(())
    }

    /// Finalize and return all accumulators.
    ///
    /// Consumes the registry.
    pub fn finalize(self) -> BTreeMap<String, StrategyAccumulator> {
        self.accumulators
    }

    /// Get a reference to an accumulator by strategy_id.
    pub fn get(&self, strategy_id: &str) -> Option<&StrategyAccumulator> {
        self.accumulators.get(strategy_id)
    }

    /// Get the number of strategies tracked.
    pub fn len(&self) -> usize {
        self.accumulators.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.accumulators.is_empty()
    }
}

impl Default for StrategyAggregatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_position_update(
        strategy_id: &str,
        ts_ns: i64,
        position_qty: i64,
        cash_delta: i64,
        realized_pnl_delta: i64,
        fee: Option<i64>,
    ) -> PositionUpdateRecord {
        PositionUpdateRecord {
            schema_version: "1".to_string(),
            ts_ns,
            session_id: "test_session".to_string(),
            seq: 1,
            correlation_id: "corr_1".to_string(),
            strategy_id: strategy_id.to_string(),
            symbol: "BTCUSDT".to_string(),
            fill_seq: 1,
            position_qty_mantissa: position_qty,
            qty_exponent: -8,
            avg_price_mantissa: Some(5000000),
            price_exponent: -2,
            cash_delta_mantissa: cash_delta,
            cash_exponent: -8,
            realized_pnl_delta_mantissa: realized_pnl_delta,
            pnl_exponent: -8,
            fee_mantissa: fee,
            fee_exponent: -8,
            venue: "sim".to_string(),
            digest: "test_digest".to_string(),
        }
    }

    #[test]
    fn test_empty_registry_defaults() {
        let registry = StrategyAggregatorRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.unified_exponent().is_none());
    }

    #[test]
    fn test_single_close_trade_counts_trade_and_winloss() {
        let mut registry = StrategyAggregatorRegistry::new();

        // Open position
        let open = make_position_update("strat_a", 1000, 100, -5000, 0, Some(10));
        registry.process_position_update(&open).unwrap();

        // Close position with profit
        let close = make_position_update("strat_a", 2000, 0, 5500, 500, Some(10));
        registry.process_position_update(&close).unwrap();

        let acc = registry.get("strat_a").unwrap();
        assert_eq!(acc.trade_count, 1);
        assert_eq!(acc.winning_trades, 1);
        assert_eq!(acc.losing_trades, 0);
        assert_eq!(acc.gross_realized_pnl_mantissa, 500);
    }

    #[test]
    fn test_flip_counts_trade_close() {
        let mut registry = StrategyAggregatorRegistry::new();

        // Open long
        let open = make_position_update("strat_a", 1000, 100, -5000, 0, None);
        registry.process_position_update(&open).unwrap();

        // Flip to short (close long + open short)
        let flip = make_position_update("strat_a", 2000, -50, 2500, -200, None);
        registry.process_position_update(&flip).unwrap();

        let acc = registry.get("strat_a").unwrap();
        assert_eq!(acc.trade_count, 1); // Flip counts as one close
        assert_eq!(acc.losing_trades, 1); // Loss on the close
        assert_eq!(acc.last_position_qty_mantissa, -50);
    }

    #[test]
    fn test_drawdown_computation_on_equity() {
        let mut registry = StrategyAggregatorRegistry::new();

        // Start with gain
        let u1 = make_position_update("strat_a", 1000, 100, 1000, 0, None);
        registry.process_position_update(&u1).unwrap();

        // Peak
        let u2 = make_position_update("strat_a", 2000, 100, 500, 0, None);
        registry.process_position_update(&u2).unwrap();

        // Drawdown
        let u3 = make_position_update("strat_a", 3000, 100, -800, 0, None);
        registry.process_position_update(&u3).unwrap();

        let acc = registry.get("strat_a").unwrap();
        assert_eq!(acc.equity_mantissa, 700); // 1000 + 500 - 800
        assert_eq!(acc.peak_equity_mantissa, 1500); // 1000 + 500
        assert_eq!(acc.max_drawdown_mantissa, 800); // 1500 - 700
    }

    #[test]
    fn test_fee_accumulation_optional() {
        let mut registry = StrategyAggregatorRegistry::new();

        // With fee
        let u1 = make_position_update("strat_a", 1000, 100, -5000, 0, Some(25));
        registry.process_position_update(&u1).unwrap();

        // Without fee
        let u2 = make_position_update("strat_a", 2000, 0, 5100, 100, None);
        registry.process_position_update(&u2).unwrap();

        let acc = registry.get("strat_a").unwrap();
        assert_eq!(acc.fees_mantissa, 25); // Only first update had fee
    }

    #[test]
    fn test_exponent_mismatch_errors() {
        let mut registry = StrategyAggregatorRegistry::new();

        // First record sets exponent
        let u1 = make_position_update("strat_a", 1000, 100, -5000, 0, None);
        registry.process_position_update(&u1).unwrap();

        // Second record with different exponent
        let mut u2 = make_position_update("strat_a", 2000, 0, 5000, 0, None);
        u2.pnl_exponent = -6; // Different!
        u2.cash_exponent = -6;
        u2.fee_exponent = -6;

        let result = registry.process_position_update(&u2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AggregatorError::ExponentMismatch { .. }
        ));
    }

    #[test]
    fn test_multiple_strategies_isolated() {
        let mut registry = StrategyAggregatorRegistry::new();

        // Strategy A
        let a1 = make_position_update("strat_a", 1000, 100, -5000, 0, None);
        let a2 = make_position_update("strat_a", 2000, 0, 5500, 500, None);
        registry.process_position_update(&a1).unwrap();
        registry.process_position_update(&a2).unwrap();

        // Strategy B
        let b1 = make_position_update("strat_b", 1500, 200, -10000, 0, None);
        let b2 = make_position_update("strat_b", 2500, 0, 9000, -1000, None);
        registry.process_position_update(&b1).unwrap();
        registry.process_position_update(&b2).unwrap();

        assert_eq!(registry.len(), 2);

        let acc_a = registry.get("strat_a").unwrap();
        assert_eq!(acc_a.trade_count, 1);
        assert_eq!(acc_a.winning_trades, 1);
        assert_eq!(acc_a.gross_realized_pnl_mantissa, 500);

        let acc_b = registry.get("strat_b").unwrap();
        assert_eq!(acc_b.trade_count, 1);
        assert_eq!(acc_b.losing_trades, 1);
        assert_eq!(acc_b.gross_realized_pnl_mantissa, -1000);
    }

    #[test]
    fn test_determinism_same_inputs_same_output() {
        let records = vec![
            make_position_update("strat_a", 1000, 100, -5000, 0, Some(10)),
            make_position_update("strat_a", 2000, 200, -5000, 0, Some(10)),
            make_position_update("strat_a", 3000, 0, 10500, 500, Some(10)),
            make_position_update("strat_b", 1500, -50, 2500, 0, None),
            make_position_update("strat_b", 2500, 0, -2400, -100, None),
        ];

        // Run twice
        let mut registry1 = StrategyAggregatorRegistry::new();
        let mut registry2 = StrategyAggregatorRegistry::new();

        for r in &records {
            registry1.process_position_update(r).unwrap();
            registry2.process_position_update(r).unwrap();
        }

        let result1 = registry1.finalize();
        let result2 = registry2.finalize();

        // Results must be identical
        assert_eq!(result1.len(), result2.len());
        for (id, acc1) in &result1 {
            let acc2 = result2.get(id).unwrap();
            assert_eq!(acc1, acc2, "Determinism violation for strategy {}", id);
        }
    }

    #[test]
    fn test_exposure_updates_counting() {
        let mut registry = StrategyAggregatorRegistry::new();

        // Open (position != 0)
        let u1 = make_position_update("strat_a", 1000, 100, -5000, 0, None);
        registry.process_position_update(&u1).unwrap();

        // Add more (position != 0)
        let u2 = make_position_update("strat_a", 2000, 200, -5000, 0, None);
        registry.process_position_update(&u2).unwrap();

        // Close (position == 0)
        let u3 = make_position_update("strat_a", 3000, 0, 10500, 500, None);
        registry.process_position_update(&u3).unwrap();

        let acc = registry.get("strat_a").unwrap();
        assert_eq!(acc.exposure_updates, 2); // Only u1 and u2 had non-zero position
    }

    #[test]
    fn test_time_bounds() {
        let mut registry = StrategyAggregatorRegistry::new();

        let u1 = make_position_update("strat_a", 1000, 100, -5000, 0, None);
        let u2 = make_position_update("strat_a", 5000, 0, 5000, 0, None);
        registry.process_position_update(&u1).unwrap();
        registry.process_position_update(&u2).unwrap();

        let acc = registry.get("strat_a").unwrap();
        assert_eq!(acc.first_ts_ns, Some(1000));
        assert_eq!(acc.last_ts_ns, Some(5000));
    }

    #[test]
    fn test_derived_metrics() {
        let mut acc = StrategyAccumulator::new("test".to_string());

        // Simulate: 3 trades, 2 wins, 1 loss
        // Win 1: +100
        acc.trade_count = 3;
        acc.winning_trades = 2;
        acc.losing_trades = 1;
        acc.gross_realized_pnl_mantissa = 150; // 100 + 100 - 50
        acc.equity_mantissa = 120; // After costs

        assert_eq!(acc.gross_pnl_mantissa(), 150);
        assert_eq!(acc.net_pnl_mantissa(), 120);
        assert!((acc.win_rate() - 0.6666).abs() < 0.01);
        assert_eq!(acc.avg_trade_pnl_mantissa(), 50); // 150 / 3
    }

    #[test]
    fn test_zero_trades_defaults() {
        let acc = StrategyAccumulator::new("empty".to_string());

        assert_eq!(acc.trade_count, 0);
        assert_eq!(acc.win_rate(), 0.0);
        assert_eq!(acc.avg_trade_pnl_mantissa(), 0);
        assert_eq!(acc.gross_pnl_mantissa(), 0);
        assert_eq!(acc.net_pnl_mantissa(), 0);
    }
}
