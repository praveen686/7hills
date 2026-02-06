//! Deterministic Cost Model for backtest/paper trading economic honesty.
//!
//! Applied at fill processing time, captured in WAL, strategy-agnostic.
//! All costs are deterministic and parameterized - no RNG, no clocks.
//!
//! ## Cost Components (v1)
//! - **Slippage**: Deterministic penalty on notional (bps)
//! - **Spread**: Half-spread cost fallback if fills aren't at bid/ask (bps)
//! - **Fee override**: Optional per-venue fee that replaces fill.fee
//!
//! ## Determinism Rules
//! - All computations use i128 internally
//! - Negative bps inputs are clamped to 0
//! - Results are clamped to i64 bounds at return

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema version for cost model config files.
pub const COST_MODEL_SCHEMA_VERSION: &str = "1";

/// Deterministic cost model configuration (v1).
///
/// Loaded once per session, included in manifest hashing for audit trail.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CostModelV1 {
    /// Schema version (must be "1")
    pub schema_version: String,
    /// Per-venue cost parameters
    pub venues: BTreeMap<String, CostVenueParamsV1>,
}

impl Default for CostModelV1 {
    fn default() -> Self {
        Self {
            schema_version: COST_MODEL_SCHEMA_VERSION.to_string(),
            venues: BTreeMap::new(),
        }
    }
}

/// Per-venue cost parameters.
///
/// All values in basis points (bps). 1 bps = 0.01% = 1/10_000.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CostVenueParamsV1 {
    /// Slippage penalty on notional (bps). Clamped to >= 0.
    #[serde(default)]
    pub slippage_bps: i32,

    /// Half-spread cost fallback (bps). Used when fills aren't at bid/ask.
    /// Clamped to >= 0.
    #[serde(default)]
    pub spread_half_bps: i32,

    /// Optional fee override (bps). If Some, replaces fill.fee entirely.
    /// If None, fill.fee is used as-is.
    #[serde(default)]
    pub fee_bps_override: Option<i32>,
}

/// Computed cost adjustments from a single fill.
///
/// Pure computed struct - no serde needed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CostAdjustments {
    /// Additional fee delta: (effective_fee - fill_fee).
    /// Can be negative if override reduces fee (rebate).
    pub extra_fee_mantissa: i64,

    /// Slippage cost (always >= 0).
    pub slippage_cost_mantissa: i64,

    /// Spread cost (always >= 0).
    pub spread_cost_mantissa: i64,
}

impl CostAdjustments {
    /// Total additional cost beyond the fill fee.
    /// slippage + spread (extra_fee is already in fee_delta path)
    pub fn total_cash_cost(&self) -> i64 {
        self.slippage_cost_mantissa
            .saturating_add(self.spread_cost_mantissa)
    }
}

/// Pure deterministic cost computation.
///
/// # Arguments
/// * `venue` - Venue identifier (must match PositionUpdateRecord.venue)
/// * `symbol` - Symbol (reserved for future per-symbol overrides)
/// * `notional_abs_mantissa` - Absolute notional value (must be >= 0)
/// * `fill_fee_mantissa` - Fee from the fill (0 if none)
/// * `model` - Cost model configuration
///
/// # Determinism Guarantees
/// - All operations in i128 internally
/// - Negative bps clamped to 0
/// - Results clamped to i64 bounds
/// - Same inputs always produce same outputs
pub fn compute_costs_v1(
    venue: &str,
    _symbol: &str, // reserved for future per-symbol overrides
    notional_abs_mantissa: i128,
    fill_fee_mantissa: i64,
    model: &CostModelV1,
) -> CostAdjustments {
    const BPS_DIVISOR: i128 = 10_000;

    // Get venue params, default to zeros if unknown
    let params = model.venues.get(venue).cloned().unwrap_or_default();

    // Clamp negative bps to 0 (no negative "costs")
    let slippage_bps = params.slippage_bps.max(0) as i128;
    let spread_half_bps = params.spread_half_bps.max(0) as i128;

    // Ensure notional is non-negative
    let n = notional_abs_mantissa.max(0);

    // slippage_cost = floor(N * slippage_bps / 10_000)
    let slippage_cost_i128 = (n * slippage_bps) / BPS_DIVISOR;

    // spread_cost = floor(N * spread_half_bps / 10_000)
    let spread_cost_i128 = (n * spread_half_bps) / BPS_DIVISOR;

    // Fee override: if present, compute from notional and replace fill_fee
    let extra_fee_i128 = if let Some(override_bps) = params.fee_bps_override {
        // Clamp negative fee override to 0
        let override_bps_clamped = override_bps.max(0) as i128;
        let override_fee = (n * override_bps_clamped) / BPS_DIVISOR;
        override_fee - fill_fee_mantissa as i128
    } else {
        0
    };

    CostAdjustments {
        extra_fee_mantissa: extra_fee_i128.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
        slippage_cost_mantissa: slippage_cost_i128.clamp(0, i64::MAX as i128) as i64,
        spread_cost_mantissa: spread_cost_i128.clamp(0, i64::MAX as i128) as i64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_costs() {
        let model = CostModelV1::default();
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 0, &model);
        assert_eq!(costs.slippage_cost_mantissa, 0);
        assert_eq!(costs.spread_cost_mantissa, 0);
        assert_eq!(costs.extra_fee_mantissa, 0);
    }

    #[test]
    fn test_slippage_10bps() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                slippage_bps: 10,
                ..Default::default()
            },
        );
        // notional = 100_000_000 (exp -8 = 1.0 in base units)
        // slippage = 100_000_000 * 10 / 10_000 = 100_000
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 0, &model);
        assert_eq!(costs.slippage_cost_mantissa, 100_000);
        assert_eq!(costs.spread_cost_mantissa, 0);
        assert_eq!(costs.extra_fee_mantissa, 0);
    }

    #[test]
    fn test_spread_half_5bps() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                spread_half_bps: 5,
                ..Default::default()
            },
        );
        // spread = 100_000_000 * 5 / 10_000 = 50_000
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 0, &model);
        assert_eq!(costs.slippage_cost_mantissa, 0);
        assert_eq!(costs.spread_cost_mantissa, 50_000);
        assert_eq!(costs.extra_fee_mantissa, 0);
    }

    #[test]
    fn test_fee_override_replaces() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                fee_bps_override: Some(10), // 10 bps
                ..Default::default()
            },
        );
        // override_fee = 100_000_000 * 10 / 10_000 = 100_000
        // fill_fee = 50_000
        // extra_fee = 100_000 - 50_000 = 50_000
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 50_000, &model);
        assert_eq!(costs.extra_fee_mantissa, 50_000);
    }

    #[test]
    fn test_fee_override_can_reduce_rebate() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                fee_bps_override: Some(5),
                ..Default::default()
            },
        );
        // override_fee = 100_000_000 * 5 / 10_000 = 50_000
        // fill_fee = 100_000
        // extra_fee = 50_000 - 100_000 = -50_000 (rebate)
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 100_000, &model);
        assert_eq!(costs.extra_fee_mantissa, -50_000);
    }

    #[test]
    fn test_combined_costs() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "binance".to_string(),
            CostVenueParamsV1 {
                slippage_bps: 10,
                spread_half_bps: 5,
                fee_bps_override: Some(10),
            },
        );
        let costs = compute_costs_v1("binance", "BTCUSDT", 100_000_000, 0, &model);
        assert_eq!(costs.slippage_cost_mantissa, 100_000);
        assert_eq!(costs.spread_cost_mantissa, 50_000);
        assert_eq!(costs.extra_fee_mantissa, 100_000);
        assert_eq!(costs.total_cash_cost(), 150_000);
    }

    #[test]
    fn test_unknown_venue_returns_zero() {
        let model = CostModelV1::default();
        let costs = compute_costs_v1("unknown", "BTCUSDT", 100_000_000, 50_000, &model);
        assert_eq!(costs.slippage_cost_mantissa, 0);
        assert_eq!(costs.spread_cost_mantissa, 0);
        assert_eq!(costs.extra_fee_mantissa, 0);
    }

    #[test]
    fn test_negative_slippage_bps_clamped_to_zero() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                slippage_bps: -10, // negative should be clamped
                spread_half_bps: -5,
                ..Default::default()
            },
        );
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 0, &model);
        assert_eq!(costs.slippage_cost_mantissa, 0);
        assert_eq!(costs.spread_cost_mantissa, 0);
    }

    #[test]
    fn test_negative_fee_override_bps_clamped_to_zero() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                fee_bps_override: Some(-10), // negative override clamped
                ..Default::default()
            },
        );
        // override_fee = 0 (clamped)
        // fill_fee = 50_000
        // extra_fee = 0 - 50_000 = -50_000
        let costs = compute_costs_v1("sim", "BTCUSDT", 100_000_000, 50_000, &model);
        assert_eq!(costs.extra_fee_mantissa, -50_000);
    }

    #[test]
    fn test_determinism_same_inputs_same_outputs() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "binance".to_string(),
            CostVenueParamsV1 {
                slippage_bps: 15,
                spread_half_bps: 7,
                fee_bps_override: Some(12),
            },
        );

        let costs1 = compute_costs_v1("binance", "ETHUSDT", 987_654_321, 12345, &model);
        let costs2 = compute_costs_v1("binance", "ETHUSDT", 987_654_321, 12345, &model);

        assert_eq!(costs1, costs2);
    }

    #[test]
    fn test_large_notional_no_overflow() {
        let mut model = CostModelV1::default();
        model.venues.insert(
            "sim".to_string(),
            CostVenueParamsV1 {
                slippage_bps: 100,   // 1%
                spread_half_bps: 50, // 0.5%
                fee_bps_override: Some(30),
            },
        );
        // Large notional that would overflow i64 if not careful
        let large_notional: i128 = i64::MAX as i128;
        let costs = compute_costs_v1("sim", "BTCUSDT", large_notional, 0, &model);

        // Results should be clamped to i64::MAX
        assert!(costs.slippage_cost_mantissa >= 0);
        assert!(costs.spread_cost_mantissa >= 0);
    }

    #[test]
    fn test_schema_version() {
        let model = CostModelV1::default();
        assert_eq!(model.schema_version, "1");
        assert_eq!(COST_MODEL_SCHEMA_VERSION, "1");
    }
}
