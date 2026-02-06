//! Phase 6: G2/G3 Anti-Overfit Harness Integration Tests
//!
//! Tests for deterministic anti-overfit validation:
//! - G2 robustness gate (time-shift, cost sensitivity, random baseline)
//! - G3 walk-forward stability (chronological folds, stability metrics)
//! - Manifest binding for G2/G3 reports
//! - Replay parity for G2/G3 artifacts

use quantlaxmi_models::{
    CostSensitivityResult, G2Gate, G2Report, G2Thresholds, G3Gate, G3Report, G3Thresholds,
    RandomBaselineResult, TimeShiftResult, WalkForwardFold,
};
use quantlaxmi_runner_crypto::segment_manifest::{
    CaptureConfig, SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

// =============================================================================
// Test 1: G2 Report Determinism
// =============================================================================

/// Test that G2Report produces deterministic canonical bytes
#[test]
fn test_g2_report_canonical_bytes_determinism() {
    let report = create_test_g2_report(true);

    let bytes1 = report.canonical_bytes();
    let bytes2 = report.canonical_bytes();

    assert_eq!(
        bytes1, bytes2,
        "G2Report canonical bytes must be deterministic"
    );

    // Hash must be stable
    let hash1 = hex::encode(Sha256::digest(&bytes1));
    let hash2 = hex::encode(Sha256::digest(&bytes2));
    assert_eq!(hash1, hash2, "G2Report hash must be stable");
}

/// Test that G2Report JSON serialization is deterministic
#[test]
fn test_g2_report_json_determinism() {
    let report = create_test_g2_report(true);

    let json1 = serde_json::to_string(&report).unwrap();
    let json2 = serde_json::to_string(&report).unwrap();

    assert_eq!(json1, json2, "G2Report JSON must be deterministic");
}

/// Test that different G2Reports produce different hashes
#[test]
fn test_g2_report_different_content_different_hash() {
    let report1 = create_test_g2_report(true);
    let report2 = create_test_g2_report(false);

    let hash1 = report1.compute_hash();
    let hash2 = report2.compute_hash();

    assert_ne!(
        hash1, hash2,
        "Different G2Reports must produce different hashes"
    );
}

// =============================================================================
// Test 2: G2 Gate Evaluation - Time Shift
// =============================================================================

/// Test time-shift gate evaluation with degrading strategy (PASS)
#[test]
fn test_time_shift_gate_appropriate_degradation() {
    let gate = G2Gate::default();
    let base_score = 5000i128;

    // Strategy degrades appropriately: k=1 keeps 85% (below 90% max threshold)
    let results = vec![
        create_time_shift_result(1, 4250, base_score), // 85% retention
        create_time_shift_result(3, 3500, base_score), // 70% retention
        create_time_shift_result(5, 2500, base_score), // 50% retention
    ];

    let (passed, reasons) = gate.evaluate_time_shift(base_score, &results);
    assert!(
        passed,
        "Should pass: score degrades appropriately. Reasons: {:?}",
        reasons
    );
}

/// Test time-shift gate evaluation with overfit strategy (FAIL)
#[test]
fn test_time_shift_gate_insufficient_degradation() {
    let gate = G2Gate::default();
    let base_score = 5000i128;

    // Strategy does NOT degrade: k=1 still has 95% (above 90% max threshold)
    let results = vec![
        create_time_shift_result(1, 4750, base_score), // 95% retention > 90% threshold = FAIL
    ];

    let (passed, reasons) = gate.evaluate_time_shift(base_score, &results);
    assert!(!passed, "Should fail: insufficient degradation at k=1");
    assert!(!reasons.is_empty(), "Should have failure reason");
    assert!(
        reasons[0].contains("insufficient degradation"),
        "Reason should mention insufficient degradation"
    );
}

// =============================================================================
// Test 3: G2 Gate Evaluation - Cost Sensitivity
// =============================================================================

/// Test cost sensitivity gate with robust strategy (PASS)
#[test]
fn test_cost_sensitivity_gate_robust_strategy() {
    let gate = G2Gate::default();
    let base_score = 5000i128;

    // Strategy survives costs: 60% retention at 2x (above 50% min threshold)
    let results = vec![
        create_cost_sensitivity_result(1, 1, 5000, base_score), // 100% at baseline
        create_cost_sensitivity_result(2, 2, 3000, base_score), // 60% at 2x > 50% min
        create_cost_sensitivity_result(5, 5, 1500, base_score), // 30% at 5x > 20% min
    ];

    let (passed, reasons) = gate.evaluate_cost_sensitivity(base_score, &results);
    assert!(
        passed,
        "Should pass: survives costs. Reasons: {:?}",
        reasons
    );
}

/// Test cost sensitivity gate with fragile strategy (FAIL)
#[test]
fn test_cost_sensitivity_gate_fragile_strategy() {
    let gate = G2Gate::default();
    let base_score = 5000i128;

    // Strategy collapses at 2x: only 40% retention (below 50% min threshold)
    let results = vec![
        create_cost_sensitivity_result(2, 2, 2000, base_score), // 40% < 50% = FAIL
    ];

    let (passed, reasons) = gate.evaluate_cost_sensitivity(base_score, &results);
    assert!(!passed, "Should fail: fragile at 2x costs");
    assert!(!reasons.is_empty(), "Should have failure reason");
}

// =============================================================================
// Test 4: G2 Gate Evaluation - Random Baseline
// =============================================================================

/// Test baseline gate with edge over random (PASS)
#[test]
fn test_baseline_gate_has_edge() {
    let gate = G2Gate::default();

    // Strategy has 150% of baseline (above 125% min threshold)
    let result = create_baseline_result(3000, 4500); // edge_ratio = 15000 bps = 150%

    let (passed, reasons) = gate.evaluate_baseline(&result);
    assert!(
        passed,
        "Should pass: has edge over baseline. Reasons: {:?}",
        reasons
    );
}

/// Test baseline gate without edge (FAIL)
#[test]
fn test_baseline_gate_no_edge() {
    let gate = G2Gate::default();

    // Strategy only has 110% of baseline (below 125% min threshold)
    let result = create_baseline_result(3000, 3300); // edge_ratio = 11000 bps = 110%

    let (passed, reasons) = gate.evaluate_baseline(&result);
    assert!(!passed, "Should fail: no edge over baseline");
    assert!(!reasons.is_empty(), "Should have failure reason");
}

// =============================================================================
// Test 5: G3 Stability Metrics Computation
// =============================================================================

/// Test stability metrics computation with all positive folds
#[test]
fn test_stability_metrics_all_positive_folds() {
    let gate = G3Gate::default();

    // 5 folds with scores: 4000, 4500, 5000, 5500, 6000
    let folds = create_test_folds(&[4000, 4500, 5000, 5500, 6000]);

    let metrics = gate.compute_stability_metrics(&folds);

    // Median should be 5000 (middle value)
    assert_eq!(metrics.median_score_mantissa, 5000, "Median should be 5000");
    // Min should be 4000
    assert_eq!(metrics.min_score_mantissa, 4000, "Min should be 4000");
    // Max should be 6000
    assert_eq!(metrics.max_score_mantissa, 6000, "Max should be 6000");
    // Dispersion = 6000 - 4000 = 2000
    assert_eq!(
        metrics.score_dispersion_mantissa, 2000,
        "Dispersion should be 2000"
    );
    // All folds profitable
    assert_eq!(metrics.profitable_folds, 5, "All 5 folds profitable");
    assert_eq!(metrics.total_folds, 5, "Total 5 folds");
    // Consistency = 100%
    assert_eq!(
        metrics.consistency_ratio_bps, 10000,
        "Consistency should be 100%"
    );
}

/// Test stability metrics with some negative folds
#[test]
fn test_stability_metrics_with_negative_fold() {
    let gate = G3Gate::default();

    // 5 folds with one negative: -1000, 2000, 3000, 4000, 5000
    let folds = create_test_folds(&[-1000, 2000, 3000, 4000, 5000]);

    let metrics = gate.compute_stability_metrics(&folds);

    // 4 of 5 folds profitable
    assert_eq!(metrics.profitable_folds, 4, "4 folds profitable");
    // Consistency = 80%
    assert_eq!(
        metrics.consistency_ratio_bps, 8000,
        "Consistency should be 80%"
    );
    // Min should be negative
    assert_eq!(metrics.min_score_mantissa, -1000, "Min should be -1000");
}

// =============================================================================
// Test 6: G3 Gate Evaluation
// =============================================================================

/// Test G3 gate with stable strategy (PASS)
#[test]
fn test_g3_gate_stable_strategy() {
    let gate = G3Gate::default();

    // All positive folds above threshold
    let folds = create_test_folds(&[2000, 2500, 3000, 3500, 4000]);
    let metrics = gate.compute_stability_metrics(&folds);

    let (passed, reasons) = gate.evaluate(&metrics);
    assert!(
        passed,
        "Should pass: stable strategy. Reasons: {:?}",
        reasons
    );
}

/// Test G3 gate with negative fold (FAIL)
#[test]
fn test_g3_gate_negative_fold() {
    let gate = G3Gate::default();

    // One negative fold violates min_fold_score_mantissa = 0
    let folds = create_test_folds(&[-500, 2000, 3000, 4000, 5000]);
    let metrics = gate.compute_stability_metrics(&folds);

    let (passed, reasons) = gate.evaluate(&metrics);
    assert!(!passed, "Should fail: negative fold");
    assert!(!reasons.is_empty(), "Should have failure reason");
}

/// Test G3 gate with low consistency (FAIL)
#[test]
fn test_g3_gate_low_consistency() {
    let gate = G3Gate::default();

    // 2 of 5 folds negative = 60% consistency (at min threshold, should pass)
    let folds = create_test_folds(&[-100, -200, 3000, 4000, 5000]);
    let metrics = gate.compute_stability_metrics(&folds);

    // With default threshold of 60%, exactly 60% should pass
    assert_eq!(metrics.consistency_ratio_bps, 6000);
    let (passed, _) = gate.evaluate(&metrics);
    // But min_fold_score check will fail first (threshold is 0)
    assert!(
        !passed,
        "Should fail: negative folds violate min_fold_score"
    );
}

// =============================================================================
// Test 7: G3 Report Determinism
// =============================================================================

/// Test that G3Report produces deterministic canonical bytes
#[test]
fn test_g3_report_canonical_bytes_determinism() {
    let report = create_test_g3_report(true);

    let bytes1 = report.canonical_bytes();
    let bytes2 = report.canonical_bytes();

    assert_eq!(
        bytes1, bytes2,
        "G3Report canonical bytes must be deterministic"
    );

    let hash1 = hex::encode(Sha256::digest(&bytes1));
    let hash2 = hex::encode(Sha256::digest(&bytes2));
    assert_eq!(hash1, hash2, "G3Report hash must be stable");
}

/// Test that G3Report JSON serialization is deterministic
#[test]
fn test_g3_report_json_determinism() {
    let report = create_test_g3_report(true);

    let json1 = serde_json::to_string(&report).unwrap();
    let json2 = serde_json::to_string(&report).unwrap();

    assert_eq!(json1, json2, "G3Report JSON must be deterministic");
}

// =============================================================================
// Test 8: G2 Manifest Binding
// =============================================================================

/// Test G2 report binding to manifest
#[test]
fn test_g2_manifest_binding() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    let mut manifest = SegmentManifest::new(
        "perp_BTCUSDT_20260126".to_string(),
        "perp_20260126_120000".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "test_hash".to_string(),
        CaptureConfig::default(),
    );

    assert!(
        manifest.g2_binding.is_none(),
        "G2 binding should be None initially"
    );

    // Create and bind G2 report
    let report = create_test_g2_report(true);
    manifest.bind_g2_report(&report, segment_dir).unwrap();

    // Verify binding exists
    let binding = manifest.g2_binding.as_ref().unwrap();
    assert_eq!(binding.report_path, "g2_report.json");
    assert!(binding.passed);
    assert_eq!(binding.version, G2Report::VERSION);
    assert_eq!(binding.base_score_mantissa, 5000);
    assert_eq!(binding.num_shift_tests, 3); // k=1,3,5
    assert_eq!(binding.num_cost_tests, 3); // 1x, 2x, 5x
    assert!(!binding.report_sha256.is_empty());

    // Verify file was written
    let report_path = segment_dir.join("g2_report.json");
    assert!(report_path.exists(), "G2 report file should exist");

    // Verify file can be parsed back
    let content = std::fs::read_to_string(&report_path).unwrap();
    let loaded: G2Report = serde_json::from_str(&content).unwrap();
    assert_eq!(loaded.passed, report.passed);
    assert_eq!(loaded.strategy_id, report.strategy_id);
}

// =============================================================================
// Test 9: G3 Manifest Binding
// =============================================================================

/// Test G3 report binding to manifest
#[test]
fn test_g3_manifest_binding() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    let mut manifest = SegmentManifest::new(
        "perp_BTCUSDT_20260126".to_string(),
        "perp_20260126_120000".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "test_hash".to_string(),
        CaptureConfig::default(),
    );

    assert!(
        manifest.g3_binding.is_none(),
        "G3 binding should be None initially"
    );

    // Create and bind G3 report
    let report = create_test_g3_report(true);
    manifest.bind_g3_report(&report, segment_dir).unwrap();

    // Verify binding exists
    let binding = manifest.g3_binding.as_ref().unwrap();
    assert_eq!(binding.report_path, "g3_walkforward.json");
    assert!(binding.passed);
    assert_eq!(binding.version, G3Report::VERSION);
    assert_eq!(binding.num_folds, 5);
    assert!(binding.median_score_mantissa > 0);
    assert!(binding.consistency_ratio_bps > 0);
    assert!(!binding.report_sha256.is_empty());

    // Verify file was written
    let report_path = segment_dir.join("g3_walkforward.json");
    assert!(report_path.exists(), "G3 report file should exist");

    // Verify file can be parsed back
    let content = std::fs::read_to_string(&report_path).unwrap();
    let loaded: G3Report = serde_json::from_str(&content).unwrap();
    assert_eq!(loaded.passed, report.passed);
    assert_eq!(loaded.strategy_id, report.strategy_id);
}

// =============================================================================
// Test 10: Manifest Round-Trip with G2/G3 Bindings
// =============================================================================

/// Test manifest round-trip preserves G2/G3 bindings
#[test]
fn test_manifest_roundtrip_with_g2_g3_bindings() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    let mut manifest = SegmentManifest::new(
        "perp_BTCUSDT_20260126".to_string(),
        "perp_20260126_120000".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "test_hash".to_string(),
        CaptureConfig::default(),
    );

    // Bind both G2 and G3 reports
    let g2_report = create_test_g2_report(true);
    let g3_report = create_test_g3_report(true);

    manifest.bind_g2_report(&g2_report, segment_dir).unwrap();
    manifest.bind_g3_report(&g3_report, segment_dir).unwrap();

    // Write manifest
    manifest.write(segment_dir).unwrap();

    // Load and verify
    let loaded = SegmentManifest::load(segment_dir).unwrap();

    // Verify G2 binding preserved
    let g2_binding = loaded.g2_binding.as_ref().unwrap();
    assert_eq!(g2_binding.report_path, "g2_report.json");
    assert!(g2_binding.passed);
    assert_eq!(g2_binding.num_shift_tests, 3);

    // Verify G3 binding preserved
    let g3_binding = loaded.g3_binding.as_ref().unwrap();
    assert_eq!(g3_binding.report_path, "g3_walkforward.json");
    assert!(g3_binding.passed);
    assert_eq!(g3_binding.num_folds, 5);
}

// =============================================================================
// Test 11: Schema Version
// =============================================================================

/// Test that schema version is 9 for Phase 6
#[test]
fn test_phase6_schema_version() {
    assert_eq!(
        SEGMENT_MANIFEST_SCHEMA_VERSION, 9,
        "Phase 6 schema version should be 9 (g2/g3 bindings)"
    );
}

// =============================================================================
// Test 12: G2/G3 Gate Version Constants
// =============================================================================

/// Test G2 gate version constant
#[test]
fn test_g2_gate_version() {
    assert_eq!(G2Gate::VERSION, "g2_gate_v1.0");
    let gate = G2Gate::default();
    assert_eq!(gate.version, G2Gate::VERSION);
}

/// Test G3 gate version constant
#[test]
fn test_g3_gate_version() {
    assert_eq!(G3Gate::VERSION, "g3_gate_v1.0");
    let gate = G3Gate::default();
    assert_eq!(gate.version, G3Gate::VERSION);
}

/// Test G2 Report version constant
#[test]
fn test_g2_report_version() {
    assert_eq!(G2Report::VERSION, "g2_report_v1.0");
}

/// Test G3 Report version constant
#[test]
fn test_g3_report_version() {
    assert_eq!(G3Report::VERSION, "g3_report_v1.0");
}

// =============================================================================
// Test 13: G2 Thresholds
// =============================================================================

/// Test G2 default thresholds
#[test]
fn test_g2_default_thresholds() {
    let thresholds = G2Thresholds::default();

    assert_eq!(
        thresholds.max_shift_1_retention_bps, 9000,
        "k=1 max retention should be 90%"
    );
    assert_eq!(
        thresholds.max_shift_3_retention_bps, 7500,
        "k=3 max retention should be 75%"
    );
    assert_eq!(
        thresholds.max_shift_5_retention_bps, 6000,
        "k=5 max retention should be 60%"
    );
    assert_eq!(
        thresholds.min_cost_2x_retention_bps, 5000,
        "2x min retention should be 50%"
    );
    assert_eq!(
        thresholds.min_cost_5x_retention_bps, 2000,
        "5x min retention should be 20%"
    );
    assert_eq!(
        thresholds.min_baseline_edge_ratio_bps, 12500,
        "Min baseline edge should be 125%"
    );
}

// =============================================================================
// Test 14: G3 Thresholds
// =============================================================================

/// Test G3 default thresholds
#[test]
fn test_g3_default_thresholds() {
    let thresholds = G3Thresholds::default();

    assert_eq!(
        thresholds.min_median_score_mantissa, 1000,
        "Min median score"
    );
    assert_eq!(
        thresholds.min_fold_score_mantissa, 0,
        "No fold should be negative"
    );
    assert_eq!(
        thresholds.max_dispersion_ratio_bps, 20000,
        "Max 200% dispersion"
    );
    assert_eq!(
        thresholds.min_consistency_ratio_bps, 6000,
        "Min 60% consistency"
    );
    assert_eq!(thresholds.num_folds, 5, "Default K=5 folds");
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_time_shift_result(shift_k: u32, score: i128, base_score: i128) -> TimeShiftResult {
    let degradation_ratio_bps = ((score * 10000) / base_score) as u32;
    TimeShiftResult {
        shift_k,
        score_mantissa: score,
        score_exponent: -4,
        win_rate_bps: 5500,
        total_decisions: 100,
        net_pnl_mantissa: score * 1000,
        pnl_exponent: -8,
        degradation_ratio_bps,
        summary_sha256: format!("shift_{}_hash", shift_k),
    }
}

fn create_cost_sensitivity_result(
    fee_mult: u32,
    slip_mult: u32,
    score: i128,
    base_score: i128,
) -> CostSensitivityResult {
    let retention_ratio_bps = if base_score > 0 {
        ((score * 10000) / base_score) as u32
    } else {
        0
    };
    CostSensitivityResult {
        fee_multiplier: fee_mult,
        slippage_multiplier: slip_mult,
        score_mantissa: score,
        score_exponent: -4,
        win_rate_bps: 5500,
        net_pnl_mantissa: score * 1000,
        pnl_exponent: -8,
        retention_ratio_bps,
        summary_sha256: format!("cost_{}x{}x_hash", fee_mult, slip_mult),
    }
}

fn create_baseline_result(baseline_score: i128, strategy_score: i128) -> RandomBaselineResult {
    let edge_ratio_bps = if baseline_score > 0 {
        ((strategy_score * 10000) / baseline_score) as u32
    } else {
        0
    };
    RandomBaselineResult {
        seed_hex: "test_seed_abc123".to_string(),
        baseline_score_mantissa: baseline_score,
        baseline_score_exponent: -4,
        baseline_win_rate_bps: 5000,
        baseline_net_pnl_mantissa: baseline_score * 1000,
        strategy_score_mantissa: strategy_score,
        edge_ratio_bps,
        absolute_edge_mantissa: strategy_score - baseline_score,
        baseline_summary_sha256: "baseline_hash".to_string(),
    }
}

fn create_test_folds(scores: &[i128]) -> Vec<WalkForwardFold> {
    scores
        .iter()
        .enumerate()
        .map(|(i, &score)| WalkForwardFold {
            fold_index: i as u32,
            start_ts_ns: (i as i64 + 1) * 1_000_000_000,
            end_ts_ns: (i as i64 + 2) * 1_000_000_000,
            num_decisions: 100,
            score_mantissa: score,
            score_exponent: -4,
            win_rate_bps: if score > 0 { 5500 } else { 4500 },
            net_pnl_mantissa: score * 1000,
            pnl_exponent: -8,
            max_loss_mantissa: if score > 0 { 100 } else { score.abs() },
            summary_sha256: format!("fold_{}_hash", i),
        })
        .collect()
}

fn create_test_g2_report(passed: bool) -> G2Report {
    let base_score = 5000i128;
    G2Report {
        version: G2Report::VERSION.to_string(),
        generated_ts_ns: 1_706_180_400_000_000_000,
        strategy_id: "funding_bias:2.0.0:abc123".to_string(),
        run_id: "run_001".to_string(),
        base_summary_sha256: "base_summary_hash".to_string(),
        base_score_mantissa: base_score,
        base_score_exponent: -4,
        time_shift_results: vec![
            create_time_shift_result(1, 4250, base_score), // 85%
            create_time_shift_result(3, 3500, base_score), // 70%
            create_time_shift_result(5, 2500, base_score), // 50%
        ],
        time_shift_passed: passed,
        time_shift_reasons: vec![],
        cost_sensitivity_results: vec![
            create_cost_sensitivity_result(1, 1, 5000, base_score), // 100%
            create_cost_sensitivity_result(2, 2, 3000, base_score), // 60%
            create_cost_sensitivity_result(5, 5, 1500, base_score), // 30%
        ],
        cost_sensitivity_passed: passed,
        cost_sensitivity_reasons: vec![],
        baseline_result: create_baseline_result(3000, 5000),
        baseline_passed: passed,
        baseline_reasons: vec![],
        thresholds: G2Thresholds::default(),
        passed,
        all_reasons: if passed {
            vec![]
        } else {
            vec!["Test failure reason".to_string()]
        },
    }
}

fn create_test_g3_report(passed: bool) -> G3Report {
    let folds = create_test_folds(&[4000, 4500, 5000, 5500, 6000]);
    let gate = G3Gate::default();
    let metrics = gate.compute_stability_metrics(&folds);

    G3Report {
        version: G3Report::VERSION.to_string(),
        generated_ts_ns: 1_706_180_400_000_000_000,
        strategy_id: "funding_bias:2.0.0:abc123".to_string(),
        run_id: "run_001".to_string(),
        segment_start_ts_ns: 1_000_000_000,
        segment_end_ts_ns: 6_000_000_000,
        folds,
        stability_metrics: metrics,
        thresholds: G3Thresholds::default(),
        passed,
        reasons: if passed {
            vec![]
        } else {
            vec!["Test failure reason".to_string()]
        },
    }
}
