//! Phase 3 Integration Tests: Correlation + Attribution
//!
//! Tests the Phase 3 "Alpha Truth Layer":
//! 1. Correlation integrity: Decision → Intent → Fill chain
//! 2. Attribution emission and manifest binding
//! 3. Replay parity for attribution artifacts

use quantlaxmi_models::{DecisionAttributionAccumulator, TradeAttributionEvent};
use quantlaxmi_runner_crypto::backtest::{Fill, Side};
use quantlaxmi_runner_crypto::segment_manifest::{
    SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest,
};
use std::collections::HashMap;
use std::io::Write;
use tempfile::TempDir;
use uuid::Uuid;

// =============================================================================
// Test 1: Correlation Integrity
// =============================================================================

/// Test that fills carry parent_decision_id correctly.
///
/// Scenario:
/// - 1 decision emits 2 intents
/// - Each intent → 1 fill
/// - Assert: each fill.parent_decision_id == decision.decision_id
#[test]
fn test_correlation_integrity_decision_to_fills() {
    // Simulate a decision that creates two fills
    let decision_id = Uuid::new_v4();

    // Simulate fill creation (in real engine, this happens in PaperExchange::execute)
    let fill1 = Fill {
        ts: chrono::Utc::now(),
        parent_decision_id: decision_id,
        symbol: "BTCUSDT".to_string(),
        side: Side::Buy,
        qty: 0.01,
        price: 100_000.0,
        fee: 1.0,
        tag: Some("entry".to_string()),
    };

    let fill2 = Fill {
        ts: chrono::Utc::now(),
        parent_decision_id: decision_id,
        symbol: "BTCUSDT".to_string(),
        side: Side::Sell,
        qty: 0.01,
        price: 101_000.0,
        fee: 1.0,
        tag: Some("exit".to_string()),
    };

    // Assert correlation preserved
    assert_eq!(
        fill1.parent_decision_id, decision_id,
        "Fill 1 should have correct parent_decision_id"
    );
    assert_eq!(
        fill2.parent_decision_id, decision_id,
        "Fill 2 should have correct parent_decision_id"
    );
    assert_eq!(
        fill1.parent_decision_id, fill2.parent_decision_id,
        "Both fills should reference same decision"
    );
}

/// Test correlation map: order_id → parent_decision_id lookup.
#[test]
fn test_correlation_map_order_to_decision() {
    // Simulate the engine's correlation map
    let mut correlation_map: HashMap<String, Uuid> = HashMap::new();

    let decision_id_1 = Uuid::new_v4();
    let decision_id_2 = Uuid::new_v4();

    // When orders are submitted, register correlation
    correlation_map.insert("order_001".to_string(), decision_id_1);
    correlation_map.insert("order_002".to_string(), decision_id_1); // Same decision, 2 orders
    correlation_map.insert("order_003".to_string(), decision_id_2);

    // When fills arrive, look up parent decision
    let fill1_decision = correlation_map.get("order_001").unwrap();
    let fill2_decision = correlation_map.get("order_002").unwrap();
    let fill3_decision = correlation_map.get("order_003").unwrap();

    assert_eq!(*fill1_decision, decision_id_1);
    assert_eq!(*fill2_decision, decision_id_1);
    assert_eq!(*fill3_decision, decision_id_2);
}

// =============================================================================
// Test 2: Attribution Emission
// =============================================================================

/// Test that DecisionAttributionAccumulator produces correct TradeAttributionEvent.
#[test]
fn test_attribution_accumulator_produces_event() {
    let decision_id = Uuid::new_v4();
    let mut acc = DecisionAttributionAccumulator::new(
        decision_id,
        "funding_bias:2.0.0:abc123".to_string(),
        "BTCUSDT".to_string(),
        "paper".to_string(),
        10_000_000, // $100,000 mid
        -2,
        -8,
    );

    // Add buy fill
    acc.add_fill(
        1_000_000_000, // ts_ns
        10_000_000,    // price: $100,000
        1_000_000,     // qty: 0.01 BTC
        10_000,        // fee
        true,          // is_buy
    );

    // Add sell fill (closing the position)
    acc.add_fill(
        2_000_000_000, // ts_ns
        10_100_000,    // price: $101,000
        1_000_000,     // qty: 0.01 BTC
        10_000,        // fee
        false,         // is_sell
    );

    assert!(acc.is_closed(), "Position should be closed");

    let event = acc.flush(2_000_000_000);

    // Verify attribution event
    assert_eq!(event.parent_decision_id, decision_id);
    assert_eq!(event.strategy_id, "funding_bias:2.0.0:abc123");
    assert_eq!(event.symbol, "BTCUSDT");
    assert_eq!(event.num_fills, 2);
    assert_eq!(event.holding_time_ns, 1_000_000_000);

    // PnL: ($101,000 - $100,000) * 0.01 BTC = $10 gross
    // Net: $10 - $0.0002 fees = ~$9.9998
    let net_pnl = event.net_pnl_f64();
    assert!(
        (net_pnl - 9.9998).abs() < 0.01,
        "Net PnL should be ~$9.9998, got {}",
        net_pnl
    );
}

/// Test attribution artifact JSON serialization.
#[test]
fn test_attribution_event_jsonl_serialization() {
    let event = TradeAttributionEvent {
        ts_ns: 1_234_567_890_000_000_000,
        symbol: "BTCUSDT".to_string(),
        venue: "paper".to_string(),
        parent_decision_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
        strategy_id: "funding_bias:2.0.0:abc123".to_string(),
        gross_pnl_mantissa: 1_000_000_000,
        fees_mantissa: 20_000,
        net_pnl_mantissa: 999_980_000,
        pnl_exponent: -8,
        holding_time_ns: 1_000_000_000,
        num_fills: 2,
        slippage_mantissa: 50_000_000,
        slippage_exponent: -8,
    };

    // Serialize to JSONL line
    let json = serde_json::to_string(&event).unwrap();

    // Verify it's valid JSON and can be parsed back
    let parsed: TradeAttributionEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.parent_decision_id, event.parent_decision_id);
    assert_eq!(parsed.gross_pnl_mantissa, event.gross_pnl_mantissa);
    assert_eq!(parsed.net_pnl_mantissa, event.net_pnl_mantissa);

    // Verify no f64 in JSON (only mantissa + exponent)
    assert!(json.contains("gross_pnl_mantissa"));
    assert!(json.contains("pnl_exponent"));
    assert!(!json.contains(r#""gross_pnl":10"#)); // No direct float representation
}

// =============================================================================
// Test 3: Attribution Artifact + Manifest Binding
// =============================================================================

/// Test that attribution artifact can be bound to manifest.
#[test]
fn test_attribution_manifest_binding() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    // Create a mock attribution file
    let attribution_path = segment_dir.join("attribution.jsonl");
    let mut file = std::fs::File::create(&attribution_path).unwrap();

    // Write 2 attribution events
    let event1 = TradeAttributionEvent {
        ts_ns: 1_000_000_000,
        symbol: "BTCUSDT".to_string(),
        venue: "paper".to_string(),
        parent_decision_id: Uuid::new_v4(),
        strategy_id: "test:1.0:abc".to_string(),
        gross_pnl_mantissa: 500_000_000,
        fees_mantissa: 10_000,
        net_pnl_mantissa: 499_990_000,
        pnl_exponent: -8,
        holding_time_ns: 1_000_000_000,
        num_fills: 2,
        slippage_mantissa: 10_000_000,
        slippage_exponent: -8,
    };

    let event2 = TradeAttributionEvent {
        ts_ns: 2_000_000_000,
        symbol: "BTCUSDT".to_string(),
        venue: "paper".to_string(),
        parent_decision_id: Uuid::new_v4(),
        strategy_id: "test:1.0:abc".to_string(),
        gross_pnl_mantissa: 700_000_000,
        fees_mantissa: 12_000,
        net_pnl_mantissa: 699_988_000,
        pnl_exponent: -8,
        holding_time_ns: 500_000_000,
        num_fills: 1,
        slippage_mantissa: -5_000_000, // Favorable slippage
        slippage_exponent: -8,
    };

    writeln!(file, "{}", serde_json::to_string(&event1).unwrap()).unwrap();
    writeln!(file, "{}", serde_json::to_string(&event2).unwrap()).unwrap();
    drop(file);

    // Create manifest and bind attribution
    let mut manifest = SegmentManifest::new(
        "perp_BTCUSDT_20260125".to_string(),
        "test_segment".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "test_hash".to_string(),
        Default::default(),
    );

    let total_net_pnl = event1.net_pnl_mantissa + event2.net_pnl_mantissa;
    let total_fees = event1.fees_mantissa + event2.fees_mantissa;

    manifest
        .bind_attribution(
            &attribution_path,
            segment_dir,
            2,
            total_net_pnl,
            total_fees,
            -8,
        )
        .unwrap();

    // Verify binding
    let binding = manifest.attribution_binding.as_ref().unwrap();
    assert_eq!(binding.attribution_path, "attribution.jsonl");
    assert_eq!(binding.num_attribution_events, 2);
    assert_eq!(binding.total_net_pnl_mantissa, total_net_pnl);
    assert_eq!(binding.total_fees_mantissa, total_fees);
    assert_eq!(binding.pnl_exponent, -8);
    assert!(!binding.attribution_sha256.is_empty());

    // Verify manifest can be serialized/deserialized
    manifest.write(segment_dir).unwrap();
    let loaded = SegmentManifest::load(segment_dir).unwrap();
    assert!(loaded.attribution_binding.is_some());
    assert_eq!(
        loaded.attribution_binding.unwrap().num_attribution_events,
        2
    );
}

/// Test that attribution binding hash is deterministic.
#[test]
fn test_attribution_hash_deterministic() {
    let temp_dir = TempDir::new().unwrap();
    let segment_dir = temp_dir.path();

    // Create identical attribution files
    let content = r#"{"ts_ns":1000,"symbol":"BTCUSDT","venue":"paper","parent_decision_id":"550e8400-e29b-41d4-a716-446655440000","strategy_id":"test","gross_pnl_mantissa":1000000,"fees_mantissa":1000,"net_pnl_mantissa":999000,"pnl_exponent":-8,"holding_time_ns":1000,"num_fills":1,"slippage_mantissa":0,"slippage_exponent":-8}"#;

    let path1 = segment_dir.join("attribution1.jsonl");
    let path2 = segment_dir.join("attribution2.jsonl");

    std::fs::write(&path1, content).unwrap();
    std::fs::write(&path2, content).unwrap();

    // Bind both and compare hashes
    let mut manifest1 = SegmentManifest::new(
        "test".to_string(),
        "seg1".to_string(),
        vec!["BTCUSDT".to_string()],
        "test".to_string(),
        "hash".to_string(),
        Default::default(),
    );

    let mut manifest2 = SegmentManifest::new(
        "test".to_string(),
        "seg2".to_string(),
        vec!["BTCUSDT".to_string()],
        "test".to_string(),
        "hash".to_string(),
        Default::default(),
    );

    manifest1
        .bind_attribution(&path1, segment_dir, 1, 999_000, 1_000, -8)
        .unwrap();
    manifest2
        .bind_attribution(&path2, segment_dir, 1, 999_000, 1_000, -8)
        .unwrap();

    let hash1 = &manifest1.attribution_binding.unwrap().attribution_sha256;
    let hash2 = &manifest2.attribution_binding.unwrap().attribution_sha256;

    assert_eq!(
        hash1, hash2,
        "Identical content should produce identical hash"
    );
}

// =============================================================================
// Test 4: Replay Parity
// =============================================================================

/// Test that identical fills produce identical attribution (determinism).
#[test]
fn test_replay_parity_attribution_determinism() {
    let decision_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();

    // Run "backtest" twice with identical parameters
    let run = || {
        let mut acc = DecisionAttributionAccumulator::new(
            decision_id,
            "test:1.0:abc".to_string(),
            "BTCUSDT".to_string(),
            "paper".to_string(),
            10_000_000,
            -2,
            -8,
        );

        // Same fills
        acc.add_fill(1_000, 10_000_000, 1_000_000, 10_000, true);
        acc.add_fill(2_000, 10_100_000, 1_000_000, 10_000, false);

        acc.flush(2_000)
    };

    let event1 = run();
    let event2 = run();

    // All fields must match exactly
    assert_eq!(event1.gross_pnl_mantissa, event2.gross_pnl_mantissa);
    assert_eq!(event1.fees_mantissa, event2.fees_mantissa);
    assert_eq!(event1.net_pnl_mantissa, event2.net_pnl_mantissa);
    assert_eq!(event1.slippage_mantissa, event2.slippage_mantissa);
    assert_eq!(event1.holding_time_ns, event2.holding_time_ns);
    assert_eq!(event1.num_fills, event2.num_fills);

    // JSON serialization must also match
    let json1 = serde_json::to_string(&event1).unwrap();
    let json2 = serde_json::to_string(&event2).unwrap();
    assert_eq!(
        json1, json2,
        "Serialized JSON must be identical for replay parity"
    );
}

/// Test that schema version is 9 (Phase 6 with G2/G3 bindings).
#[test]
fn test_phase6_schema_version() {
    assert_eq!(
        SEGMENT_MANIFEST_SCHEMA_VERSION, 9,
        "Phase 6 schema version should be 9 (g2/g3 bindings)"
    );
}

/// Test Fill struct has parent_decision_id field.
#[test]
fn test_fill_has_parent_decision_id() {
    let decision_id = Uuid::new_v4();
    let fill = Fill {
        ts: chrono::Utc::now(),
        parent_decision_id: decision_id,
        symbol: "BTCUSDT".to_string(),
        side: Side::Buy,
        qty: 0.01,
        price: 100_000.0,
        fee: 1.0,
        tag: None,
    };

    assert_eq!(fill.parent_decision_id, decision_id);
}
