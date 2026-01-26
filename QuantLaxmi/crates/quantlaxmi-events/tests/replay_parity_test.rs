//! Integration tests for replay parity verification.
//!
//! These tests verify that:
//! 1. Identical decision sequences produce identical trace hashes
//! 2. Divergences are detected at the correct index with proper reasons
//! 3. Trace serialization/deserialization preserves hashes

use chrono::{TimeZone, Utc};
use quantlaxmi_events::{
    CorrelationContext, DecisionEvent, DecisionTrace, DecisionTraceBuilder, MarketSnapshot,
    ReplayParityResult, verify_replay_parity,
};
use uuid::Uuid;

/// Create a deterministic test decision.
fn make_decision(seq: u8, direction: i8, tag: &str) -> DecisionEvent {
    // Use fixed timestamps for determinism
    let ts = Utc
        .with_ymd_and_hms(2026, 1, 25, 12, 0, seq as u32)
        .unwrap();

    // Use predictable UUIDs
    let decision_id =
        Uuid::parse_str(&format!("00000000-0000-0000-0000-00000000{:04x}", seq)).unwrap();

    DecisionEvent {
        ts,
        decision_id,
        strategy_id: "test_strategy".to_string(),
        symbol: "BTCUSDT".to_string(),
        decision_type: tag.to_string(),
        direction,
        target_qty_mantissa: 1_000_000, // 0.01 BTC
        qty_exponent: -8,
        reference_price_mantissa: 8_871_660, // $88,716.60
        price_exponent: -2,
        market_snapshot: MarketSnapshot {
            bid_price_mantissa: 8_871_650,
            ask_price_mantissa: 8_871_670,
            bid_qty_mantissa: 10_000_000,
            ask_qty_mantissa: 10_000_000,
            price_exponent: -2,
            qty_exponent: -8,
            // Fixed-point: 23 with exponent -2 = 0.23 bps
            spread_bps_mantissa: 23,
            book_ts_ns: 1737799200000000000 + (seq as i64 * 1_000_000_000),
        },
        // Fixed-point: 10000 with exponent -4 = 1.0
        confidence_mantissa: 10_000,
        metadata: serde_json::json!({"tag": tag}),
        // Note: CorrelationContext is flattened, so fields like symbol, strategy_id,
        // and decision_id would conflict with top-level fields. Set them to None.
        ctx: CorrelationContext {
            session_id: Some("test-session".to_string()),
            run_id: Some("test-run".to_string()),
            symbol: None, // Conflicts with top-level symbol
            venue: Some("binance".to_string()),
            strategy_id: None, // Conflicts with top-level strategy_id
            decision_id: None, // Conflicts with top-level decision_id
            order_id: None,
        },
    }
}

#[test]
fn test_identical_sequences_produce_identical_hashes() {
    // Create two identical sequences
    let decisions = vec![
        make_decision(1, 1, "entry"),
        make_decision(2, 0, "hold"),
        make_decision(3, -1, "exit"),
        make_decision(4, 1, "entry"),
        make_decision(5, -1, "exit"),
    ];

    // Build original trace
    let mut original_builder = DecisionTraceBuilder::new();
    for d in &decisions {
        original_builder.record(d);
    }
    let original = original_builder.finalize();

    // Build replay trace with same decisions
    let mut replay_builder = DecisionTraceBuilder::new();
    for d in &decisions {
        replay_builder.record(d);
    }
    let replay = replay_builder.finalize();

    // Verify hashes match
    assert_eq!(
        original.trace_hash, replay.trace_hash,
        "Identical sequences must produce identical hashes"
    );

    // Verify using verify_replay_parity
    let result = verify_replay_parity(&original, &replay);
    assert!(
        matches!(result, ReplayParityResult::Match),
        "Expected Match, got {:?}",
        result
    );
}

#[test]
fn test_divergence_detected_at_correct_index() {
    // Create original sequence
    let original_decisions = vec![
        make_decision(1, 1, "entry"),
        make_decision(2, 0, "hold"),
        make_decision(3, -1, "exit"),
    ];

    // Create replay sequence with difference at index 1
    let mut replay_decisions = original_decisions.clone();
    replay_decisions[1] = make_decision(2, 1, "hold"); // Changed direction from 0 to 1

    // Build traces
    let mut original_builder = DecisionTraceBuilder::new();
    for d in &original_decisions {
        original_builder.record(d);
    }
    let original = original_builder.finalize();

    let mut replay_builder = DecisionTraceBuilder::new();
    for d in &replay_decisions {
        replay_builder.record(d);
    }
    let replay = replay_builder.finalize();

    // Verify divergence detected
    let result = verify_replay_parity(&original, &replay);
    match result {
        ReplayParityResult::Divergence { index, reason, .. } => {
            assert_eq!(index, 1, "Divergence should be at index 1");
            assert!(
                reason.contains("direction"),
                "Reason should mention direction: {}",
                reason
            );
        }
        _ => panic!("Expected Divergence, got {:?}", result),
    }
}

#[test]
fn test_length_mismatch_detected() {
    let decisions = vec![make_decision(1, 1, "entry"), make_decision(2, -1, "exit")];

    // Original has 2 decisions
    let mut original_builder = DecisionTraceBuilder::new();
    for d in &decisions {
        original_builder.record(d);
    }
    let original = original_builder.finalize();

    // Replay has 1 decision
    let mut replay_builder = DecisionTraceBuilder::new();
    replay_builder.record(&decisions[0]);
    let replay = replay_builder.finalize();

    let result = verify_replay_parity(&original, &replay);
    match result {
        ReplayParityResult::LengthMismatch {
            original_len,
            replay_len,
        } => {
            assert_eq!(original_len, 2);
            assert_eq!(replay_len, 1);
        }
        _ => panic!("Expected LengthMismatch, got {:?}", result),
    }
}

#[test]
fn test_trace_serialization_preserves_hash() {
    let decisions = vec![
        make_decision(1, 1, "entry"),
        make_decision(2, 0, "hold"),
        make_decision(3, -1, "exit"),
    ];

    let mut builder = DecisionTraceBuilder::new();
    for d in &decisions {
        builder.record(d);
    }
    let original = builder.finalize();
    let original_hash = original.trace_hash;

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&original).expect("Failed to serialize");

    // Deserialize
    let deserialized: DecisionTrace = serde_json::from_str(&json).expect("Failed to deserialize");

    // Hash should be preserved
    assert_eq!(
        original_hash, deserialized.trace_hash,
        "Hash should be preserved through serialization"
    );

    // Verify parity still works
    let result = verify_replay_parity(&original, &deserialized);
    assert!(
        matches!(result, ReplayParityResult::Match),
        "Deserialized trace should match original"
    );
}

#[test]
fn test_file_round_trip() {
    let decisions = vec![make_decision(1, 1, "entry"), make_decision(2, -1, "exit")];

    let mut builder = DecisionTraceBuilder::new();
    for d in &decisions {
        builder.record(d);
    }
    let original = builder.finalize();
    let original_hash = original.trace_hash;

    // Save to temp file
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let trace_path = temp_dir.path().join("test_trace.json");

    original.save(&trace_path).expect("Failed to save trace");

    // Load and verify
    let loaded = DecisionTrace::load(&trace_path).expect("Failed to load trace");

    assert_eq!(
        original_hash, loaded.trace_hash,
        "Hash should be preserved through file round-trip"
    );

    // Verify parity
    let result = verify_replay_parity(&original, &loaded);
    assert!(matches!(result, ReplayParityResult::Match));
}

#[test]
fn test_metadata_order_does_not_affect_hash() {
    // Test that JSON object key order doesn't affect hash
    // (canonicalize_json should sort keys)

    let mut d1 = make_decision(1, 1, "entry");
    #[allow(unused_mut)]
    let mut d2 = make_decision(1, 1, "entry");

    // Create metadata with keys in different order
    d1.metadata = serde_json::json!({"a": 1, "b": 2, "c": 3});
    d2.metadata = serde_json::json!({"c": 3, "a": 1, "b": 2});

    let mut builder1 = DecisionTraceBuilder::new();
    builder1.record(&d1);
    let trace1 = builder1.finalize();

    let mut builder2 = DecisionTraceBuilder::new();
    builder2.record(&d2);
    let trace2 = builder2.finalize();

    // Hashes should match because keys are sorted before encoding
    assert_eq!(
        trace1.trace_hash, trace2.trace_hash,
        "Metadata key order should not affect hash"
    );
}

#[test]
fn test_encoding_version_in_trace() {
    let mut builder = DecisionTraceBuilder::new();
    builder.record(&make_decision(1, 1, "entry"));
    let trace = builder.finalize();

    assert_eq!(
        trace.encoding_version,
        quantlaxmi_events::ENCODING_VERSION,
        "Encoding version should be set"
    );
}

#[test]
fn test_divergence_at_price_detected() {
    let d1 = make_decision(1, 1, "entry");
    let mut d2 = make_decision(1, 1, "entry");

    // Change reference price
    d2.reference_price_mantissa = 8_871_670; // Different from 8_871_660

    let mut builder1 = DecisionTraceBuilder::new();
    builder1.record(&d1);
    let trace1 = builder1.finalize();

    let mut builder2 = DecisionTraceBuilder::new();
    builder2.record(&d2);
    let trace2 = builder2.finalize();

    assert_ne!(trace1.trace_hash, trace2.trace_hash);

    let result = verify_replay_parity(&trace1, &trace2);
    match result {
        ReplayParityResult::Divergence { index, reason, .. } => {
            assert_eq!(index, 0);
            assert!(
                reason.contains("reference_price"),
                "Reason should mention price: {}",
                reason
            );
        }
        _ => panic!("Expected Divergence"),
    }
}

#[test]
fn test_empty_traces_match() {
    let trace1 = DecisionTraceBuilder::new().finalize();
    let trace2 = DecisionTraceBuilder::new().finalize();

    assert_eq!(trace1.trace_hash, trace2.trace_hash);

    let result = verify_replay_parity(&trace1, &trace2);
    assert!(matches!(result, ReplayParityResult::Match));
}
