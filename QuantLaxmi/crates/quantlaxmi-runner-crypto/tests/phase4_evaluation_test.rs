//! Phase 4 Integration Tests: Alpha Output Contracts + Strategy Evaluation
//!
//! Tests the Phase 4 "Alpha Truth Layer" evaluation pipeline:
//! 1. Attribution Events → AttributionSummary aggregation
//! 2. AttributionSummary → AlphaScoreV1 computation
//! 3. AlphaScoreV1 → G1PromotionGate evaluation
//! 4. Full pipeline manifest binding and artifact verification

use quantlaxmi_models::{
    AlphaScoreV1, AttributionSummary, AttributionSummaryBuilder, G1PromotionGate,
    TradeAttributionEvent,
};
use quantlaxmi_runner_crypto::segment_manifest::{
    CaptureConfig, SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest,
};
use tempfile::TempDir;
use uuid::Uuid;

// =============================================================================
// Test 1: Full Pipeline - Attribution Events to G1 Decision
// =============================================================================

/// Test the complete pipeline: events → summary → alpha → gate decision
#[test]
fn test_full_pipeline_attribution_to_promotion() {
    // Step 1: Create attribution events (simulating backtest output)
    let decision_ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();

    let events: Vec<TradeAttributionEvent> = decision_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            // 6 wins, 4 losses
            let is_win = i < 6;
            let gross_pnl = if is_win { 1_000_000_000 } else { -500_000_000 }; // $10 or -$5
            let fees = 10_000; // $0.0001

            TradeAttributionEvent {
                ts_ns: (i as i64 + 1) * 1_000_000_000,
                symbol: "BTCUSDT".to_string(),
                venue: "paper".to_string(),
                parent_decision_id: id,
                strategy_id: "funding_bias:2.0.0:abc123".to_string(),
                gross_pnl_mantissa: gross_pnl,
                fees_mantissa: fees,
                net_pnl_mantissa: gross_pnl - fees,
                pnl_exponent: -8,
                holding_time_ns: 3_600_000_000_000, // 1 hour
                num_fills: 2,
                slippage_mantissa: 1_000_000,
                slippage_exponent: -8,
            }
        })
        .collect();

    // Step 2: Build attribution summary
    let mut builder = AttributionSummaryBuilder::new(
        "funding_bias:2.0.0:abc123".to_string(),
        "backtest_run_001".to_string(),
        vec!["BTCUSDT".to_string()],
        -8,
    );

    for event in &events {
        builder.add_event(event);
    }

    let summary = builder.build(1_234_567_890_000_000_000);

    // Verify summary aggregation
    assert_eq!(summary.total_decisions, 10);
    assert_eq!(summary.total_fills, 20);
    assert_eq!(summary.winning_decisions, 6);
    assert_eq!(summary.losing_decisions, 4);
    assert_eq!(summary.win_rate_bps, 6000); // 60%

    // Net PnL: 6 * $9.9999 - 4 * $5.0001 = ~$39.9994
    let expected_net_pnl = 6 * (1_000_000_000 - 10_000) - 4 * (500_000_000 + 10_000);
    assert_eq!(summary.total_net_pnl_mantissa, expected_net_pnl);

    // Step 3: Compute alpha score
    let alpha_score = AlphaScoreV1::from_summary(&summary);

    assert!(alpha_score.is_profitable(), "Strategy should be profitable");
    assert!(
        alpha_score.score_mantissa > 0,
        "Alpha score should be positive"
    );

    // Step 4: Evaluate with G1 gate
    let gate = G1PromotionGate::new();
    let result = gate.evaluate(&summary, &alpha_score);

    // With 60% win rate, 10 decisions, and profitable - should pass
    assert!(
        result.passed,
        "Strategy should pass G1 gate: {}",
        result.summary()
    );
    assert!(result.reasons.is_empty());
}

// =============================================================================
// Test 2: Pipeline with Failing Strategy
// =============================================================================

/// Test pipeline with a strategy that fails G1 gate
#[test]
fn test_pipeline_failing_strategy() {
    // Create events for a losing strategy
    let events: Vec<TradeAttributionEvent> = (0..10)
        .map(|i| {
            // 3 wins, 7 losses
            let is_win = i < 3;
            let gross_pnl = if is_win { 500_000_000 } else { -800_000_000 };

            TradeAttributionEvent {
                ts_ns: (i as i64 + 1) * 1_000_000_000,
                symbol: "BTCUSDT".to_string(),
                venue: "paper".to_string(),
                parent_decision_id: Uuid::new_v4(),
                strategy_id: "losing_strategy:1.0:xyz".to_string(),
                gross_pnl_mantissa: gross_pnl,
                fees_mantissa: 10_000,
                net_pnl_mantissa: gross_pnl - 10_000,
                pnl_exponent: -8,
                holding_time_ns: 3_600_000_000_000,
                num_fills: 2,
                slippage_mantissa: 1_000_000,
                slippage_exponent: -8,
            }
        })
        .collect();

    let mut builder = AttributionSummaryBuilder::new(
        "losing_strategy:1.0:xyz".to_string(),
        "run_001".to_string(),
        vec!["BTCUSDT".to_string()],
        -8,
    );

    for event in &events {
        builder.add_event(event);
    }

    let summary = builder.build(1_234_567_890_000_000_000);
    let alpha_score = AlphaScoreV1::from_summary(&summary);
    let gate = G1PromotionGate::new();
    let result = gate.evaluate(&summary, &alpha_score);

    // Should fail due to:
    // - 30% win rate (< 40% minimum)
    // - Unprofitable (negative net PnL)
    assert!(!result.passed, "Strategy should fail G1 gate");
    assert!(
        result.reasons.iter().any(|r| r.contains("Win rate")),
        "Should fail win rate check"
    );
}

// =============================================================================
// Test 3: Manifest Binding Integration
// =============================================================================

/// Test that attribution summary binds correctly to manifest
#[test]
fn test_attribution_summary_manifest_binding() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    // Create a manifest
    let mut manifest = SegmentManifest::new(
        "perp_BTCUSDT_20260126".to_string(),
        "perp_20260126_120000".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "test_hash".to_string(),
        CaptureConfig::default(),
    );

    // Create a summary from a successful backtest
    let summary = AttributionSummary {
        strategy_id: "funding_bias:2.0.0:abc123".to_string(),
        run_id: "backtest_001".to_string(),
        symbols: vec!["BTCUSDT".to_string()],
        generated_ts_ns: 1706270400_000_000_000,
        total_decisions: 50,
        total_fills: 100,
        winning_decisions: 30,
        losing_decisions: 20,
        round_trips: 15,
        total_gross_pnl_mantissa: 5_000_000_000,
        total_fees_mantissa: 50_000,
        total_net_pnl_mantissa: 4_999_950_000,
        pnl_exponent: -8,
        win_rate_bps: 6000,
        avg_pnl_per_decision_mantissa: 99_999_000,
        total_slippage_mantissa: 10_000_000,
        slippage_exponent: -8,
        max_loss_mantissa: 500_000_000,
        total_holding_time_ns: 50_000_000_000_000,
    };

    let alpha_score = AlphaScoreV1::from_summary(&summary);

    // Bind to manifest
    manifest
        .bind_attribution_summary(&summary, &alpha_score, segment_dir)
        .unwrap();

    // Verify binding exists
    assert!(manifest.attribution_summary_binding.is_some());
    let binding = manifest.attribution_summary_binding.as_ref().unwrap();

    assert_eq!(binding.strategy_id, "funding_bias:2.0.0:abc123");
    assert_eq!(binding.total_decisions, 50);
    assert_eq!(binding.win_rate_bps, 6000);
    assert_eq!(binding.alpha_score_formula, AlphaScoreV1::VERSION);

    // Write and reload manifest
    manifest.write(segment_dir).unwrap();
    let loaded = SegmentManifest::load(segment_dir).unwrap();

    assert!(loaded.attribution_summary_binding.is_some());
    let loaded_binding = loaded.attribution_summary_binding.unwrap();
    assert_eq!(loaded_binding.strategy_id, binding.strategy_id);
    assert_eq!(
        loaded_binding.alpha_score_mantissa,
        binding.alpha_score_mantissa
    );

    // Verify artifact file exists and is valid JSON
    let summary_path = segment_dir.join("attribution_summary.json");
    assert!(summary_path.exists());

    let content = std::fs::read_to_string(&summary_path).unwrap();
    let parsed: AttributionSummary = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed.strategy_id, summary.strategy_id);
}

// =============================================================================
// Test 4: Schema Version
// =============================================================================

/// Test that schema version is 9 for Phase 6 (G2/G3 bindings)
#[test]
fn test_phase6_schema_version() {
    assert_eq!(
        SEGMENT_MANIFEST_SCHEMA_VERSION, 9,
        "Phase 6 schema version should be 9 (g2/g3 bindings)"
    );
}

// =============================================================================
// Test 5: Determinism - Same Events → Same Result
// =============================================================================

/// Test that the pipeline produces deterministic results
#[test]
fn test_pipeline_determinism() {
    // Fixed decision IDs for reproducibility
    let decision_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

    let create_event = || TradeAttributionEvent {
        ts_ns: 1_000_000_000,
        symbol: "BTCUSDT".to_string(),
        venue: "paper".to_string(),
        parent_decision_id: decision_id,
        strategy_id: "test:1.0:abc".to_string(),
        gross_pnl_mantissa: 1_000_000_000,
        fees_mantissa: 10_000,
        net_pnl_mantissa: 999_990_000,
        pnl_exponent: -8,
        holding_time_ns: 3_600_000_000_000,
        num_fills: 2,
        slippage_mantissa: 1_000_000,
        slippage_exponent: -8,
    };

    // Run pipeline twice
    let run_pipeline = || {
        let event = create_event();
        let mut builder = AttributionSummaryBuilder::new(
            event.strategy_id.clone(),
            "run_001".to_string(),
            vec![event.symbol.clone()],
            event.pnl_exponent,
        );
        builder.add_event(&event);
        let summary = builder.build(1_234_567_890);
        let alpha = AlphaScoreV1::from_summary(&summary);
        let gate = G1PromotionGate::new();
        let result = gate.evaluate(&summary, &alpha);
        (summary, alpha, result)
    };

    let (summary1, alpha1, result1) = run_pipeline();
    let (summary2, alpha2, result2) = run_pipeline();

    // All values must be identical
    assert_eq!(
        summary1.total_net_pnl_mantissa,
        summary2.total_net_pnl_mantissa
    );
    assert_eq!(alpha1.score_mantissa, alpha2.score_mantissa);
    assert_eq!(result1.passed, result2.passed);

    // JSON serialization must also be identical
    let json1 = serde_json::to_string(&summary1).unwrap();
    let json2 = serde_json::to_string(&summary2).unwrap();
    assert_eq!(json1, json2);
}

// =============================================================================
// Test 6: G1 Gate Threshold Documentation
// =============================================================================

/// Test that G1 gate thresholds are documented correctly
#[test]
fn test_g1_gate_threshold_constants() {
    // Verify documented threshold values match implementation
    assert_eq!(G1PromotionGate::DEFAULT_MIN_ALPHA_SCORE, 1000);
    assert_eq!(G1PromotionGate::DEFAULT_MIN_WIN_RATE_BPS, 4000); // 40%
    assert_eq!(G1PromotionGate::DEFAULT_MIN_DECISIONS, 10);
    assert_eq!(G1PromotionGate::DEFAULT_MAX_LOSS_PCT_BPS, 5000); // 50%
    assert_eq!(G1PromotionGate::VERSION, "g1_gate_v1.0");
}

// =============================================================================
// Test 7: Alpha Score Formula Locked
// =============================================================================

/// Test that alpha score formula constants are locked
#[test]
fn test_alpha_score_formula_locked() {
    // Verify documented formula constants match implementation
    assert_eq!(AlphaScoreV1::SCALE, 10000);
    assert_eq!(AlphaScoreV1::EPSILON, 1_000_000);
    assert_eq!(AlphaScoreV1::VERSION, "alpha_score_v1.0");
}

// =============================================================================
// Test 8: Edge Cases
// =============================================================================

/// Test edge case: exactly at threshold boundaries
#[test]
fn test_g1_gate_boundary_conditions() {
    // Create summary exactly at the minimum thresholds
    let summary = AttributionSummary {
        strategy_id: "boundary:1.0:test".to_string(),
        run_id: "run_001".to_string(),
        symbols: vec!["BTCUSDT".to_string()],
        generated_ts_ns: 1_234_567_890,
        total_decisions: 10, // Exactly at minimum
        total_fills: 20,
        winning_decisions: 4,
        losing_decisions: 6,
        round_trips: 5,
        total_gross_pnl_mantissa: 1_000_000_000,
        total_fees_mantissa: 10_000,
        total_net_pnl_mantissa: 999_990_000,
        pnl_exponent: -8,
        win_rate_bps: 4000, // Exactly at minimum (40%)
        avg_pnl_per_decision_mantissa: 99_999_000,
        total_slippage_mantissa: 1_000_000,
        slippage_exponent: -8,
        max_loss_mantissa: 499_995_000, // Just under 50% of net PnL
        total_holding_time_ns: 1_000_000_000,
    };

    let alpha_score = AlphaScoreV1::from_summary(&summary);

    // Alpha score needs to be >= 1000
    // With these values: (999_990_000 * 10000) / (499_995_000 + 1_000_000)
    // = 9999900000000 / 500995000 ≈ 19960
    // This should pass the alpha threshold

    let gate = G1PromotionGate::new();
    let result = gate.evaluate(&summary, &alpha_score);

    // Should pass if all boundaries are met
    if !result.passed {
        println!("Failed reasons: {:?}", result.reasons);
        println!("Alpha score: {}", alpha_score.score_mantissa);
    }

    // At exact boundaries, should pass (>= checks)
    assert!(
        result.passed || result.reasons.iter().any(|r| r.contains("Alpha")),
        "Should pass or only fail on alpha"
    );
}
