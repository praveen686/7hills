//! Phase 25A Integration Tests: Deterministic Cost Model
//!
//! Tests verify:
//! 1. Zero-cost model produces identical cash_delta vs baseline
//! 2. Nonzero slippage reduces cash_delta deterministically
//! 3. Replay with same cost model passes G7

use quantlaxmi_models::{CostAdjustments, CostModelV1, CostVenueParamsV1, compute_costs_v1};

// =============================================================================
// Unit Tests for Cost Computation
// =============================================================================

#[test]
fn test_zero_cost_model_no_adjustment() {
    let model = CostModelV1::default();

    // Simulate a fill: notional = 1 BTC at $50,000 = $50,000
    // In fixed-point: price = 5_000_000 (exp -2), qty = 100_000_000 (exp -8)
    // notional_mantissa = (5_000_000 * 100_000_000) / 100 = 5_000_000_000_000 (exp -8)
    let notional_abs = 5_000_000_000_000i128;
    let fill_fee = 0i64;

    let costs = compute_costs_v1("sim", "BTCUSDT", notional_abs, fill_fee, &model);

    assert_eq!(costs.slippage_cost_mantissa, 0);
    assert_eq!(costs.spread_cost_mantissa, 0);
    assert_eq!(costs.extra_fee_mantissa, 0);
    assert_eq!(costs.total_cash_cost(), 0);
}

#[test]
fn test_slippage_reduces_cash_deterministically() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 10, // 10 bps = 0.1%
            spread_half_bps: 0,
            fee_bps_override: None,
        },
    );

    // notional = $50,000 in fixed-point = 5_000_000_000_000 (exp -8)
    let notional_abs = 5_000_000_000_000i128;
    let fill_fee = 0i64;

    let costs = compute_costs_v1("sim", "BTCUSDT", notional_abs, fill_fee, &model);

    // slippage = 5_000_000_000_000 * 10 / 10_000 = 5_000_000_000 (= $50)
    assert_eq!(costs.slippage_cost_mantissa, 5_000_000_000);
    assert_eq!(costs.spread_cost_mantissa, 0);
    assert_eq!(costs.extra_fee_mantissa, 0);
}

#[test]
fn test_spread_cost_deterministic() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 0,
            spread_half_bps: 5, // 5 bps = 0.05%
            fee_bps_override: None,
        },
    );

    let notional_abs = 5_000_000_000_000i128;
    let fill_fee = 0i64;

    let costs = compute_costs_v1("sim", "BTCUSDT", notional_abs, fill_fee, &model);

    // spread = 5_000_000_000_000 * 5 / 10_000 = 2_500_000_000 (= $25)
    assert_eq!(costs.slippage_cost_mantissa, 0);
    assert_eq!(costs.spread_cost_mantissa, 2_500_000_000);
}

#[test]
fn test_fee_override_deterministic() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "binance".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 0,
            spread_half_bps: 0,
            fee_bps_override: Some(10), // 10 bps override
        },
    );

    let notional_abs = 5_000_000_000_000i128;
    let fill_fee = 2_500_000_000i64; // Original fill fee = $25

    let costs = compute_costs_v1("binance", "BTCUSDT", notional_abs, fill_fee, &model);

    // override_fee = 5_000_000_000_000 * 10 / 10_000 = 5_000_000_000 (= $50)
    // extra_fee = 5_000_000_000 - 2_500_000_000 = 2_500_000_000 (= $25 extra)
    assert_eq!(costs.extra_fee_mantissa, 2_500_000_000);
}

#[test]
fn test_combined_costs_deterministic() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 10,
            spread_half_bps: 5,
            fee_bps_override: Some(10),
        },
    );

    let notional_abs = 5_000_000_000_000i128;
    let fill_fee = 0i64;

    let costs = compute_costs_v1("sim", "BTCUSDT", notional_abs, fill_fee, &model);

    // slippage = $50, spread = $25, fee_override = $50
    assert_eq!(costs.slippage_cost_mantissa, 5_000_000_000);
    assert_eq!(costs.spread_cost_mantissa, 2_500_000_000);
    assert_eq!(costs.extra_fee_mantissa, 5_000_000_000);

    // total_cash_cost = slippage + spread = $75 (fee is separate accounting)
    assert_eq!(costs.total_cash_cost(), 7_500_000_000);
}

#[test]
fn test_determinism_same_inputs_same_outputs() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 15,
            spread_half_bps: 7,
            fee_bps_override: Some(12),
        },
    );

    // Run 100 times with same inputs
    let results: Vec<CostAdjustments> = (0..100)
        .map(|_| compute_costs_v1("sim", "ETHUSDT", 987_654_321_000, 12345, &model))
        .collect();

    // All results must be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            result, first,
            "Determinism violation at iteration {}: {:?} != {:?}",
            i, result, first
        );
    }
}

#[test]
fn test_unknown_venue_defaults_to_zero() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "binance".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 100,
            spread_half_bps: 50,
            fee_bps_override: Some(30),
        },
    );

    // Query with unknown venue
    let costs = compute_costs_v1("unknown_venue", "BTCUSDT", 5_000_000_000_000, 0, &model);

    // Should return zero costs (no adjustment)
    assert_eq!(costs.slippage_cost_mantissa, 0);
    assert_eq!(costs.spread_cost_mantissa, 0);
    assert_eq!(costs.extra_fee_mantissa, 0);
}

#[test]
fn test_negative_bps_clamped() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: -10, // Invalid: should be clamped to 0
            spread_half_bps: -5,
            fee_bps_override: None,
        },
    );

    let costs = compute_costs_v1("sim", "BTCUSDT", 5_000_000_000_000, 0, &model);

    // Negative bps should be clamped to 0
    assert_eq!(costs.slippage_cost_mantissa, 0);
    assert_eq!(costs.spread_cost_mantissa, 0);
}

#[test]
fn test_fee_rebate_negative_extra_fee() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 0,
            spread_half_bps: 0,
            fee_bps_override: Some(5), // 5 bps override
        },
    );

    let notional_abs = 5_000_000_000_000i128;
    let fill_fee = 5_000_000_000i64; // Original fill fee = $50 (10 bps)

    let costs = compute_costs_v1("sim", "BTCUSDT", notional_abs, fill_fee, &model);

    // override_fee = 5_000_000_000_000 * 5 / 10_000 = 2_500_000_000 (= $25)
    // extra_fee = 2_500_000_000 - 5_000_000_000 = -2_500_000_000 (rebate of $25)
    assert_eq!(costs.extra_fee_mantissa, -2_500_000_000);
}

// =============================================================================
// Cash Delta Computation Tests (simulating runner logic)
// =============================================================================

/// Simulate the runner's cash delta computation with cost model
fn compute_cash_delta_with_costs(
    is_buy: bool,
    notional_mantissa: i128,
    fee_delta: i128,
    costs: &CostAdjustments,
) -> i128 {
    let effective_fee_delta = fee_delta + costs.extra_fee_mantissa as i128;

    let mut cash_delta = if is_buy {
        -notional_mantissa - effective_fee_delta
    } else {
        notional_mantissa - effective_fee_delta
    };

    // Apply slippage + spread (always reduces cash)
    cash_delta -= costs.slippage_cost_mantissa as i128;
    cash_delta -= costs.spread_cost_mantissa as i128;

    cash_delta
}

#[test]
fn test_buy_cash_delta_with_costs() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 10,
            spread_half_bps: 5,
            fee_bps_override: None,
        },
    );

    // Buy 1 BTC at $50,000
    let notional = 5_000_000_000_000i128;
    let fee_delta = 2_500_000_000i128; // $25 fee

    let costs = compute_costs_v1("sim", "BTCUSDT", notional, fee_delta as i64, &model);
    let cash_delta = compute_cash_delta_with_costs(true, notional, fee_delta, &costs);

    // Without costs: -50000 - 25 = -50025
    // With costs: -50025 - 50 (slippage) - 25 (spread) = -50100
    // In fixed-point: -5_000_000_000_000 - 2_500_000_000 - 5_000_000_000 - 2_500_000_000
    //               = -5_010_000_000_000
    assert_eq!(cash_delta, -5_010_000_000_000);
}

#[test]
fn test_sell_cash_delta_with_costs() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 10,
            spread_half_bps: 5,
            fee_bps_override: None,
        },
    );

    // Sell 1 BTC at $50,000
    let notional = 5_000_000_000_000i128;
    let fee_delta = 2_500_000_000i128; // $25 fee

    let costs = compute_costs_v1("sim", "BTCUSDT", notional, fee_delta as i64, &model);
    let cash_delta = compute_cash_delta_with_costs(false, notional, fee_delta, &costs);

    // Without costs: +50000 - 25 = +49975
    // With costs: +49975 - 50 (slippage) - 25 (spread) = +49900
    // In fixed-point: +5_000_000_000_000 - 2_500_000_000 - 5_000_000_000 - 2_500_000_000
    //               = +4_990_000_000_000
    assert_eq!(cash_delta, 4_990_000_000_000);
}

#[test]
fn test_zero_cost_model_preserves_baseline() {
    let model = CostModelV1::default();

    let notional = 5_000_000_000_000i128;
    let fee_delta = 2_500_000_000i128;

    let costs = compute_costs_v1("sim", "BTCUSDT", notional, fee_delta as i64, &model);

    // Zero-cost model should produce zero adjustments
    assert_eq!(costs.slippage_cost_mantissa, 0);
    assert_eq!(costs.spread_cost_mantissa, 0);
    assert_eq!(costs.extra_fee_mantissa, 0);

    // Cash delta should equal baseline (no cost adjustment)
    let cash_delta_buy = compute_cash_delta_with_costs(true, notional, fee_delta, &costs);
    let cash_delta_sell = compute_cash_delta_with_costs(false, notional, fee_delta, &costs);

    // Baseline: buy = -notional - fee, sell = +notional - fee
    assert_eq!(cash_delta_buy, -notional - fee_delta);
    assert_eq!(cash_delta_sell, notional - fee_delta);
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_cost_model_json_roundtrip() {
    let mut model = CostModelV1::default();
    model.venues.insert(
        "sim".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 10,
            spread_half_bps: 5,
            fee_bps_override: Some(15),
        },
    );
    model.venues.insert(
        "binance".to_string(),
        CostVenueParamsV1 {
            slippage_bps: 5,
            spread_half_bps: 3,
            fee_bps_override: None,
        },
    );

    let json = serde_json::to_string_pretty(&model).unwrap();
    let parsed: CostModelV1 = serde_json::from_str(&json).unwrap();

    assert_eq!(model, parsed);
}

#[test]
fn test_cost_model_schema_version() {
    let model = CostModelV1::default();
    assert_eq!(model.schema_version, "1");
}

#[test]
fn test_cost_venue_params_defaults() {
    let params = CostVenueParamsV1::default();
    assert_eq!(params.slippage_bps, 0);
    assert_eq!(params.spread_half_bps, 0);
    assert_eq!(params.fee_bps_override, None);
}
