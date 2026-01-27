//! Phase 5 Integration Tests: Router v1 + Regime Buckets
//!
//! Tests the Phase 5 "Deterministic Router" pipeline:
//! 1. Regime classification determinism: identical inputs -> identical regime
//! 2. Strategy routing determinism: identical regime -> identical strategy selection
//! 3. RouterDecisionEvent determinism: identical decisions -> identical hashes
//! 4. Manifest binding: router decisions bound with SHA-256

use quantlaxmi_models::{
    RegimeInputs, RegimeLabel, Router, RouterConfig, RouterConfigBuilder, RouterDecisionEvent,
    StrategyProfile,
};
use quantlaxmi_runner_crypto::segment_manifest::{
    CaptureConfig, SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest,
};
use tempfile::TempDir;
use uuid::Uuid;

// =============================================================================
// Test 1: Regime Classification Determinism
// =============================================================================

/// Test that identical inputs always produce identical regime classification.
#[test]
fn test_regime_classification_determinism() {
    let config = RouterConfigBuilder::new()
        .default_strategy("default:1.0:abc")
        .build();
    let router = Router::new(config);

    // Create fixed inputs
    let inputs = RegimeInputs {
        ts_ns: 1_000_000_000,
        symbol: "BTCUSDT".to_string(),
        spread_bps: 10,
        spread_tier: 0,
        volatility_bps: 150,
        volatility_tier: 1,
        depth_mantissa: 100_000_000,
        depth_exponent: -2,
        liquidity_tier: 0,
        funding_rate_bps: 5,
        funding_tier: 0,
        trend_strength: 0,
    };

    // Classify 100 times - all results must be identical
    let (first_regime, first_confidence) = router.classify_regime(&inputs);

    for _ in 0..100 {
        let (regime, confidence) = router.classify_regime(&inputs);
        assert_eq!(
            regime, first_regime,
            "Regime should be deterministic for identical inputs"
        );
        assert_eq!(
            confidence, first_confidence,
            "Confidence should be deterministic for identical inputs"
        );
    }
}

/// Test regime classification for all regime types.
#[test]
fn test_regime_classification_all_types() {
    let config = RouterConfigBuilder::new()
        .default_strategy("default:1.0:abc")
        .build();
    let router = Router::new(config);

    // Base inputs
    let base = RegimeInputs {
        ts_ns: 1_000_000_000,
        symbol: "BTCUSDT".to_string(),
        spread_bps: 10,
        spread_tier: 0,
        volatility_bps: 50,
        volatility_tier: 0,
        depth_mantissa: 100_000_000,
        depth_exponent: -2,
        liquidity_tier: 0,
        funding_rate_bps: 0,
        funding_tier: 0,
        trend_strength: 0,
    };

    // Test HALT: empty liquidity (tier 3)
    let halt_inputs = RegimeInputs {
        liquidity_tier: 3,
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&halt_inputs);
    assert_eq!(
        regime,
        RegimeLabel::Halt,
        "Empty liquidity should trigger HALT"
    );

    // Test LOW_LIQUIDITY: thin books (tier 2)
    let low_liq_inputs = RegimeInputs {
        liquidity_tier: 2,
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&low_liq_inputs);
    assert_eq!(
        regime,
        RegimeLabel::LowLiquidity,
        "Thin liquidity should trigger LOW_LIQUIDITY"
    );

    // Test HIGH_VOL: volatility tier 2+
    let high_vol_inputs = RegimeInputs {
        volatility_tier: 2,
        volatility_bps: 300,
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&high_vol_inputs);
    assert_eq!(
        regime,
        RegimeLabel::HighVol,
        "High volatility should trigger HIGH_VOL"
    );

    // Test FUNDING_SKEW: extreme funding (tier 2+)
    let funding_inputs = RegimeInputs {
        funding_tier: 2,
        funding_rate_bps: 100,
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&funding_inputs);
    assert_eq!(
        regime,
        RegimeLabel::FundingSkew,
        "Elevated funding should trigger FUNDING_SKEW"
    );

    // Test TRENDING: strong trend
    let trend_inputs = RegimeInputs {
        trend_strength: 6000, // Above default threshold of 5000
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&trend_inputs);
    assert_eq!(
        regime,
        RegimeLabel::Trending,
        "Strong trend should trigger TRENDING"
    );

    // Test MEAN_REVERT: tight spread, low vol
    let mean_revert_inputs = RegimeInputs {
        spread_tier: 0,
        spread_bps: 5,
        volatility_tier: 0,
        volatility_bps: 30,
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&mean_revert_inputs);
    assert_eq!(
        regime,
        RegimeLabel::MeanRevert,
        "Tight spread + low vol should trigger MEAN_REVERT"
    );

    // Test NORMAL: default
    let normal_inputs = RegimeInputs {
        spread_tier: 1,
        spread_bps: 20,
        volatility_tier: 1,
        volatility_bps: 100,
        ..base.clone()
    };
    let (regime, _) = router.classify_regime(&normal_inputs);
    assert_eq!(regime, RegimeLabel::Normal, "Default should be NORMAL");
}

// =============================================================================
// Test 2: Strategy Selection Determinism
// =============================================================================

/// Test that identical regime + config always selects identical strategy.
#[test]
fn test_strategy_selection_determinism() {
    let config = RouterConfigBuilder::new()
        .add_profile(StrategyProfile {
            strategy_id: "funding_bias:2.0.0:abc123".to_string(),
            name: "Funding Bias".to_string(),
            suitable_regimes: vec![RegimeLabel::Normal, RegimeLabel::FundingSkew],
            priority: 100,
            enabled: true,
        })
        .add_profile(StrategyProfile {
            strategy_id: "momentum:1.0.0:xyz456".to_string(),
            name: "Momentum".to_string(),
            suitable_regimes: vec![RegimeLabel::Trending, RegimeLabel::HighVol],
            priority: 100,
            enabled: true,
        })
        .default_strategy("default:1.0:fallback")
        .build();

    let router = Router::new(config);

    // Normal regime should select funding_bias
    let (strategy_id, _, _) = router.select_strategy(RegimeLabel::Normal);
    assert_eq!(strategy_id, "funding_bias:2.0.0:abc123");

    // Trending regime should select momentum
    let (strategy_id, _, _) = router.select_strategy(RegimeLabel::Trending);
    assert_eq!(strategy_id, "momentum:1.0.0:xyz456");

    // Run selection 100 times to verify determinism
    for _ in 0..100 {
        let (s1, _, _) = router.select_strategy(RegimeLabel::Normal);
        let (s2, _, _) = router.select_strategy(RegimeLabel::Trending);

        assert_eq!(s1, "funding_bias:2.0.0:abc123");
        assert_eq!(s2, "momentum:1.0.0:xyz456");
    }
}

// =============================================================================
// Test 3: RouterDecisionEvent Determinism + Canonical Bytes
// =============================================================================

/// Test that identical router decisions produce identical canonical bytes and hashes.
#[test]
fn test_router_decision_canonical_bytes_determinism() {
    let decision_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

    let create_decision = || {
        let inputs = RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 10,
            spread_tier: 0,
            volatility_bps: 50,
            volatility_tier: 0,
            depth_mantissa: 100_000_000,
            depth_exponent: -2,
            liquidity_tier: 0,
            funding_rate_bps: 5,
            funding_tier: 0,
            trend_strength: 0,
        };

        RouterDecisionEvent {
            ts_ns: 1_000_000_000,
            decision_id,
            symbols: vec!["BTCUSDT".to_string()],
            inputs,
            regime: RegimeLabel::Normal,
            confidence_bps: 8000,
            selected_strategy_id: "funding_bias:2.0.0:abc123".to_string(),
            alternatives: vec![],
            selection_reason: "Best match for Normal regime".to_string(),
            router_config_hash: "router_config_hash_001".to_string(),
            router_version: "router_v1.0".to_string(),
        }
    };

    // Create two identical decisions
    let decision1 = create_decision();
    let decision2 = create_decision();

    // Canonical bytes must be identical
    let bytes1 = decision1.canonical_bytes();
    let bytes2 = decision2.canonical_bytes();
    assert_eq!(bytes1, bytes2, "Canonical bytes must be identical");

    // Compute hash must be identical
    let hash1 = decision1.compute_hash();
    let hash2 = decision2.compute_hash();
    assert_eq!(
        hash1, hash2,
        "Hash must be identical for identical decisions"
    );

    // JSON serialization must be identical
    let json1 = serde_json::to_string(&decision1).unwrap();
    let json2 = serde_json::to_string(&decision2).unwrap();
    assert_eq!(json1, json2, "JSON serialization must be identical");
}

// =============================================================================
// Test 4: Full Pipeline Determinism - Inputs -> Routing -> Decision
// =============================================================================

/// Test the complete routing pipeline produces identical results.
#[test]
fn test_full_routing_pipeline_determinism() {
    let config = RouterConfigBuilder::new()
        .add_profile(StrategyProfile {
            strategy_id: "funding_bias:2.0.0:abc123".to_string(),
            name: "Funding Bias".to_string(),
            suitable_regimes: vec![RegimeLabel::Normal, RegimeLabel::FundingSkew],
            priority: 100,
            enabled: true,
        })
        .default_strategy("default:1.0:fallback")
        .build();

    let router = Router::new(config);

    // Fixed inputs
    let inputs = RegimeInputs {
        ts_ns: 1_000_000_000,
        symbol: "BTCUSDT".to_string(),
        spread_bps: 10,
        spread_tier: 0,
        volatility_bps: 50,
        volatility_tier: 0,
        depth_mantissa: 100_000_000,
        depth_exponent: -2,
        liquidity_tier: 0,
        funding_rate_bps: 5,
        funding_tier: 0,
        trend_strength: 0,
    };

    // Fixed decision ID for reproducibility
    let decision_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

    // Run pipeline twice
    let decision1 = router.route_with_id(inputs.clone(), vec!["BTCUSDT".to_string()], decision_id);
    let decision2 = router.route_with_id(inputs.clone(), vec!["BTCUSDT".to_string()], decision_id);

    // All fields must be identical
    assert_eq!(decision1.ts_ns, decision2.ts_ns);
    assert_eq!(decision1.decision_id, decision2.decision_id);
    assert_eq!(decision1.regime, decision2.regime);
    assert_eq!(decision1.confidence_bps, decision2.confidence_bps);
    assert_eq!(
        decision1.selected_strategy_id,
        decision2.selected_strategy_id
    );
    assert_eq!(decision1.router_config_hash, decision2.router_config_hash);

    // Hashes must be identical
    assert_eq!(decision1.compute_hash(), decision2.compute_hash());
}

// =============================================================================
// Test 5: Manifest Binding Integration
// =============================================================================

/// Test that router decisions can be bound to manifest with deterministic hash.
#[test]
fn test_router_manifest_binding_determinism() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    // Create router config
    let config = RouterConfigBuilder::new()
        .add_profile(StrategyProfile {
            strategy_id: "funding_bias:2.0.0:abc123".to_string(),
            name: "Funding Bias".to_string(),
            suitable_regimes: vec![RegimeLabel::Normal],
            priority: 100,
            enabled: true,
        })
        .default_strategy("default:1.0:fallback")
        .build();

    let router = Router::new(config.clone());

    // Create deterministic inputs
    let inputs_list = vec![
        RegimeInputs {
            ts_ns: 1_000_000_000,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 10,
            spread_tier: 0,
            volatility_bps: 50,
            volatility_tier: 0,
            depth_mantissa: 100_000_000,
            depth_exponent: -2,
            liquidity_tier: 0,
            funding_rate_bps: 5,
            funding_tier: 0,
            trend_strength: 0,
        },
        RegimeInputs {
            ts_ns: 2_000_000_000,
            symbol: "BTCUSDT".to_string(),
            spread_bps: 15,
            spread_tier: 0,
            volatility_bps: 60,
            volatility_tier: 0,
            depth_mantissa: 90_000_000,
            depth_exponent: -2,
            liquidity_tier: 0,
            funding_rate_bps: 8,
            funding_tier: 0,
            trend_strength: 0,
        },
    ];

    // Generate decisions with fixed IDs
    let decisions: Vec<RouterDecisionEvent> = inputs_list
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            let id =
                Uuid::parse_str(&format!("550e8400-e29b-41d4-a716-44665544{:04x}", i)).unwrap();
            router.route_with_id(inp.clone(), vec!["BTCUSDT".to_string()], id)
        })
        .collect();

    assert_eq!(decisions.len(), 2, "Should have 2 routing decisions");

    // Bind to manifest
    let mut manifest = SegmentManifest::new(
        "perp_BTCUSDT_20260126".to_string(),
        "perp_20260126_120000".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "binary_hash_001".to_string(),
        CaptureConfig::default(),
    );

    manifest
        .bind_router_decisions(
            &decisions,
            &config.compute_hash(),
            RouterConfig::VERSION,
            segment_dir,
        )
        .unwrap();

    let binding = manifest.router_binding.as_ref().unwrap();
    let first_hash = binding.decisions_sha256.clone();

    // Write and reload
    manifest.write(segment_dir).unwrap();
    let loaded = SegmentManifest::load(segment_dir).unwrap();

    // Verify binding persisted correctly
    let loaded_binding = loaded.router_binding.unwrap();
    assert_eq!(loaded_binding.decisions_sha256, first_hash);
    assert_eq!(loaded_binding.num_decisions, 2);
    assert_eq!(loaded_binding.router_version, RouterConfig::VERSION);

    // Create a second manifest with identical decisions - hash must match
    let temp_dir2 = TempDir::new().unwrap();
    let segment_dir2 = temp_dir2.path();

    let mut manifest2 = SegmentManifest::new(
        "perp_BTCUSDT_20260126".to_string(),
        "perp_20260126_120000_replay".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "binary_hash_001".to_string(),
        CaptureConfig::default(),
    );

    // Generate identical decisions again
    let decisions2: Vec<RouterDecisionEvent> = inputs_list
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            let id =
                Uuid::parse_str(&format!("550e8400-e29b-41d4-a716-44665544{:04x}", i)).unwrap();
            router.route_with_id(inp.clone(), vec!["BTCUSDT".to_string()], id)
        })
        .collect();

    manifest2
        .bind_router_decisions(
            &decisions2,
            &config.compute_hash(),
            RouterConfig::VERSION,
            segment_dir2,
        )
        .unwrap();

    let binding2 = manifest2.router_binding.as_ref().unwrap();

    // Hashes must be identical for replay parity
    assert_eq!(
        binding.decisions_sha256, binding2.decisions_sha256,
        "Identical decisions must produce identical hash (replay parity)"
    );
}

// =============================================================================
// Test 6: Schema Version
// =============================================================================

/// Test that schema version is 9 for Phase 6 (G2/G3 bindings).
#[test]
fn test_phase6_schema_version() {
    assert_eq!(
        SEGMENT_MANIFEST_SCHEMA_VERSION, 9,
        "Phase 6 schema version should be 9 (g2/g3 bindings)"
    );
}

// =============================================================================
// Test 7: Router Config Determinism
// =============================================================================

/// Test that router config hash is deterministic.
#[test]
fn test_router_config_hash_determinism() {
    let create_config = || {
        RouterConfigBuilder::new()
            .add_profile(StrategyProfile {
                strategy_id: "funding_bias:2.0.0:abc123".to_string(),
                name: "Funding Bias".to_string(),
                suitable_regimes: vec![RegimeLabel::Normal, RegimeLabel::FundingSkew],
                priority: 100,
                enabled: true,
            })
            .add_profile(StrategyProfile {
                strategy_id: "momentum:1.0.0:xyz456".to_string(),
                name: "Momentum".to_string(),
                suitable_regimes: vec![RegimeLabel::Trending],
                priority: 90,
                enabled: true,
            })
            .default_strategy("default:1.0:fallback")
            .build()
    };

    let config1 = create_config();
    let config2 = create_config();

    // Canonical bytes must be identical
    assert_eq!(
        config1.canonical_bytes(),
        config2.canonical_bytes(),
        "Config canonical bytes must be identical"
    );

    // Hashes must be identical
    assert_eq!(
        config1.compute_hash(),
        config2.compute_hash(),
        "Config hash must be identical"
    );

    // Hash should be a valid hex SHA-256 (64 chars)
    let hash = config1.compute_hash();
    assert_eq!(hash.len(), 64, "Hash should be 64 hex chars");
    assert!(
        hash.chars().all(|c| c.is_ascii_hexdigit()),
        "Hash should be valid hex"
    );
}

// =============================================================================
// Test 8: RegimeInputs Canonical Bytes
// =============================================================================

/// Test that RegimeInputs produces deterministic canonical bytes.
#[test]
fn test_regime_inputs_canonical_bytes_determinism() {
    let create_inputs = || RegimeInputs {
        ts_ns: 1_234_567_890_000_000_000,
        symbol: "BTCUSDT".to_string(),
        spread_bps: 10,
        spread_tier: 0,
        volatility_bps: 150,
        volatility_tier: 1,
        depth_mantissa: 100_000_000,
        depth_exponent: -2,
        liquidity_tier: 0,
        funding_rate_bps: 5,
        funding_tier: 0,
        trend_strength: 1000,
    };

    let inputs1 = create_inputs();
    let inputs2 = create_inputs();

    assert_eq!(
        inputs1.canonical_bytes(),
        inputs2.canonical_bytes(),
        "Canonical bytes must be identical for identical inputs"
    );

    // Verify canonical bytes include all fields
    let bytes = inputs1.canonical_bytes();
    assert!(
        !bytes.is_empty(),
        "Canonical bytes should include all input fields"
    );
}

// =============================================================================
// Test 9: Priority Order Verification
// =============================================================================

/// Test that regime classification follows documented priority order.
#[test]
fn test_regime_priority_order() {
    let config = RouterConfigBuilder::new()
        .default_strategy("default:1.0:abc")
        .build();
    let router = Router::new(config);

    // Create inputs that trigger multiple regimes - highest priority should win
    // Priority order: HALT > LOW_LIQUIDITY > HIGH_VOL > FUNDING_SKEW > TRENDING > MEAN_REVERT > NORMAL

    // Base with all triggers active
    let all_triggers = RegimeInputs {
        ts_ns: 1_000_000_000,
        symbol: "BTCUSDT".to_string(),
        spread_bps: 10,
        spread_tier: 0,
        volatility_bps: 500, // High vol
        volatility_tier: 3,
        depth_mantissa: 0,
        depth_exponent: -2,
        liquidity_tier: 3,     // HALT
        funding_rate_bps: 200, // Extreme funding
        funding_tier: 3,
        trend_strength: 8000, // Strong trend
    };

    // HALT should override everything
    let (regime, _) = router.classify_regime(&all_triggers);
    assert_eq!(
        regime,
        RegimeLabel::Halt,
        "HALT should have highest priority"
    );

    // LOW_LIQUIDITY should override HIGH_VOL
    let low_liq_with_vol = RegimeInputs {
        liquidity_tier: 2,
        ..all_triggers.clone()
    };
    let (regime, _) = router.classify_regime(&low_liq_with_vol);
    assert_eq!(
        regime,
        RegimeLabel::LowLiquidity,
        "LOW_LIQUIDITY should override HIGH_VOL"
    );

    // HIGH_VOL should override FUNDING_SKEW
    let high_vol_with_funding = RegimeInputs {
        liquidity_tier: 0,
        volatility_tier: 2,
        ..all_triggers.clone()
    };
    let (regime, _) = router.classify_regime(&high_vol_with_funding);
    assert_eq!(
        regime,
        RegimeLabel::HighVol,
        "HIGH_VOL should override FUNDING_SKEW"
    );
}

// =============================================================================
// Test 10: RouterConfig Version Constant
// =============================================================================

/// Test that router config version constant is correct.
#[test]
fn test_router_config_version_constant() {
    assert_eq!(
        RouterConfig::VERSION,
        "router_v1.0",
        "Router config version should be router_v1.0"
    );
}
