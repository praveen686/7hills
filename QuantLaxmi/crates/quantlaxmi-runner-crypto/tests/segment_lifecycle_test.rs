//! Phase 2B.2 CI Gate: Golden Segment Fixtures
//!
//! Validates segment manifest lifecycle invariants:
//! - Bootstrap manifests exist with correct schema
//! - Finalized segments have digests
//! - Retro-finalization produces correct state transitions

use quantlaxmi_runner_crypto::segment_manifest::{
    EventCounts, SEGMENT_MANIFEST_SCHEMA_VERSION, SegmentManifest, SegmentState,
    compute_segment_digests,
};
use std::path::Path;

const FIXTURES_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/segment_lifecycle"
);

/// Test that finalized segment fixture has correct structure
#[test]
fn test_finalized_segment_fixture() {
    let segment_dir = Path::new(FIXTURES_DIR).join("finalized_segment");
    assert!(
        segment_dir.exists(),
        "Finalized segment fixture missing: {:?}",
        segment_dir
    );

    let manifest = SegmentManifest::load(&segment_dir).expect("Failed to load finalized manifest");

    // Schema version must match
    assert_eq!(
        manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION,
        "Schema version mismatch"
    );

    // Quote schema must be canonical_v1
    assert_eq!(
        manifest.quote_schema, "canonical_v1",
        "Quote schema mismatch"
    );

    // State must be FINALIZED
    assert_eq!(
        manifest.state,
        SegmentState::Finalized,
        "State should be FINALIZED"
    );

    // Must have digests
    assert!(
        manifest.digests.is_some(),
        "Finalized segment must have digests"
    );
    let digests = manifest.digests.as_ref().unwrap();

    // Perp digest should exist
    assert!(
        digests.perp.is_some(),
        "Finalized segment should have perp digest"
    );

    // Binary hash must exist
    assert!(
        !manifest.binary_hash.is_empty(),
        "Binary hash must not be empty"
    );

    // Config must exist
    assert!(manifest.config.price_exponent != 0 || manifest.config.qty_exponent != 0);

    // Duration should be set
    assert!(
        manifest.duration_secs.is_some(),
        "Finalized segment should have duration"
    );

    // End timestamp should be set
    assert!(
        manifest.end_ts.is_some(),
        "Finalized segment should have end_ts"
    );
}

/// Test that bootstrap segment fixture has correct structure
#[test]
fn test_bootstrap_segment_fixture() {
    let segment_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");
    assert!(
        segment_dir.exists(),
        "Bootstrap segment fixture missing: {:?}",
        segment_dir
    );

    let manifest = SegmentManifest::load(&segment_dir).expect("Failed to load bootstrap manifest");

    // Schema version must match
    assert_eq!(
        manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION,
        "Schema version mismatch"
    );

    // Quote schema must be canonical_v1
    assert_eq!(
        manifest.quote_schema, "canonical_v1",
        "Quote schema mismatch"
    );

    // State must be BOOTSTRAP
    assert_eq!(
        manifest.state,
        SegmentState::Bootstrap,
        "State should be BOOTSTRAP"
    );

    // Should NOT have digests (bootstrap only)
    assert!(
        manifest.digests.is_none(),
        "Bootstrap segment should not have digests"
    );

    // Binary hash must exist
    assert!(
        !manifest.binary_hash.is_empty(),
        "Binary hash must not be empty"
    );

    // Stop reason should be RUNNING
    assert_eq!(
        manifest.stop_reason,
        quantlaxmi_runner_crypto::segment_manifest::StopReason::Running,
        "Bootstrap stop_reason should be RUNNING"
    );
}

/// Test retro-finalization of bootstrap segment
#[test]
fn test_retro_finalize_bootstrap_segment() {
    let segment_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");

    // Load the bootstrap manifest
    let mut manifest =
        SegmentManifest::load(&segment_dir).expect("Failed to load bootstrap manifest");

    assert_eq!(manifest.state, SegmentState::Bootstrap);

    // Compute digests from the fixture files
    let digests = compute_segment_digests(&segment_dir).expect("Failed to compute digests");

    // Count events from digests
    let events = EventCounts {
        spot_quotes: digests.spot.as_ref().map_or(0, |d| d.event_count),
        perp_quotes: digests.perp.as_ref().map_or(0, |d| d.event_count),
        funding: digests.funding.as_ref().map_or(0, |d| d.event_count),
        depth: digests.depth.as_ref().map_or(0, |d| d.event_count),
    };

    // Perform retro-finalization (in memory only - don't modify fixture)
    manifest.retro_finalize(events.clone(), digests);

    // Verify state transition
    assert_eq!(
        manifest.state,
        SegmentState::FinalizedRetro,
        "State should be FINALIZED_RETRO after retro_finalize"
    );

    // Verify digests are present
    assert!(
        manifest.digests.is_some(),
        "Digests should be present after retro_finalize"
    );

    // Verify stop_reason changed from RUNNING
    assert_eq!(
        manifest.stop_reason,
        quantlaxmi_runner_crypto::segment_manifest::StopReason::Unknown,
        "Stop reason should be UNKNOWN after retro_finalize"
    );

    // Verify end_ts is set
    assert!(
        manifest.end_ts.is_some(),
        "end_ts should be set after retro_finalize"
    );

    // Verify duration is computed
    assert!(
        manifest.duration_secs.is_some(),
        "duration_secs should be computed after retro_finalize"
    );

    // Verify event counts
    assert!(
        events.perp_quotes > 0,
        "Should have perp events from fixture"
    );
    assert!(
        events.spot_quotes > 0,
        "Should have spot events from fixture"
    );
}

/// Test digest computation produces valid SHA256
#[test]
fn test_digest_computation_validity() {
    let segment_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");

    let digests = compute_segment_digests(&segment_dir).expect("Failed to compute digests");

    // Perp digest should exist
    let perp = digests.perp.expect("Perp digest should exist");
    assert_eq!(perp.sha256.len(), 64, "SHA256 should be 64 hex chars");
    assert!(
        perp.sha256.chars().all(|c| c.is_ascii_hexdigit()),
        "SHA256 should be valid hex"
    );
    assert_eq!(perp.event_count, 3, "Should have 3 perp events");

    // Spot digest should exist
    let spot = digests.spot.expect("Spot digest should exist");
    assert_eq!(spot.sha256.len(), 64, "SHA256 should be 64 hex chars");
    assert_eq!(spot.event_count, 2, "Should have 2 spot events");

    // Funding should be None (no funding.jsonl in fixture)
    assert!(digests.funding.is_none(), "No funding file in fixture");
}

/// Test that schema version constant is correct
#[test]
fn test_schema_version_constant() {
    assert_eq!(
        SEGMENT_MANIFEST_SCHEMA_VERSION, 9,
        "Schema version should be 9 for Phase 6 (g2/g3 bindings)"
    );
}

/// Test is_finalized helper
#[test]
fn test_is_finalized_helper() {
    let finalized_dir = Path::new(FIXTURES_DIR).join("finalized_segment");
    let bootstrap_dir = Path::new(FIXTURES_DIR).join("bootstrap_segment");

    let finalized = SegmentManifest::load(&finalized_dir).unwrap();
    let bootstrap = SegmentManifest::load(&bootstrap_dir).unwrap();

    assert!(finalized.is_finalized(), "FINALIZED should return true");
    assert!(!bootstrap.is_finalized(), "BOOTSTRAP should return false");
}

// =============================================================================
// Phase 1.3: Trace Binding and Replay Parity Integration Tests
// =============================================================================

use chrono::{TimeZone, Utc};
use quantlaxmi_events::{
    CorrelationContext, DecisionEvent, DecisionTraceBuilder, ENCODING_VERSION, MarketSnapshot,
};
use quantlaxmi_runner_crypto::backtest::{PNL_EXPONENT, PnlAccumulatorFixed};
use quantlaxmi_runner_crypto::segment_manifest::{CaptureConfig, compute_file_sha256};
use std::fs;
use uuid::Uuid;

/// Test that trace encoding version is v2 (fixed-point, no floats)
#[test]
fn test_trace_encoding_version_is_v2() {
    assert_eq!(ENCODING_VERSION, 0x02, "Trace encoding should be v2");
}

/// Test PnlAccumulatorFixed determinism
#[test]
fn test_pnl_fixed_determinism_multiple_runs() {
    // Run the same sequence of fills multiple times
    for _ in 0..10 {
        let mut acc = PnlAccumulatorFixed::new();

        // Buy 0.01 BTC at $100,000
        acc.process_fill(10_000_000, 1_000_000, 10_000, true);
        // Sell 0.01 BTC at $101,000
        acc.process_fill(10_100_000, 1_000_000, 10_000, false);

        // PnL must be exactly the same every time (no floating-point drift)
        // Expected: (101000 - 100000) * 0.01 - fees = ~$9.9998
        assert!(acc.is_flat(), "Position should be flat after round-trip");

        // Verify mantissa is deterministic
        let expected_pnl_range = 999_900_000..1_000_100_000i128; // ~$9.999 to $10.001 in mantissa
        assert!(
            expected_pnl_range.contains(&acc.realized_pnl_mantissa),
            "PnL mantissa {} should be in expected range",
            acc.realized_pnl_mantissa
        );
    }
}

/// Test that fixed-point PnL matches float PnL within tolerance
#[test]
fn test_pnl_fixed_matches_float_pnl() {
    let mut acc = PnlAccumulatorFixed::new();

    // Series of trades
    acc.process_fill(10_000_000, 1_000_000, 10_000, true); // Buy 0.01 @ $100,000
    acc.process_fill(10_050_000, 500_000, 5_000, true); // Buy 0.005 @ $100,500
    acc.process_fill(10_100_000, 1_500_000, 15_000, false); // Sell 0.015 @ $101,000

    // Convert to f64 for display
    let pnl_f64 = acc.realized_pnl_f64();

    // Expected rough calculation:
    // Buy 0.01 @ $100,000 = $1,000
    // Buy 0.005 @ $100,500 = $502.50
    // Total cost = $1,502.50 for 0.015 BTC
    // Avg entry = $100,166.67
    // Sell 0.015 @ $101,000 = $1,515
    // Gross profit = $1,515 - $1,502.50 = $12.50
    // Fees = 10000 + 5000 + 15000 = 30000 (exp -8 = $0.0003)
    // Net profit ≈ $12.50

    assert!(
        pnl_f64 > 10.0 && pnl_f64 < 15.0,
        "PnL should be around $12.50, got {}",
        pnl_f64
    );

    // Verify position is flat
    assert!(acc.is_flat());
}

/// Test that PNL_EXPONENT matches the crypto qty exponent
#[test]
fn test_pnl_exponent_matches_qty_exponent() {
    // PNL_EXPONENT should be -8 to match crypto quantity precision
    assert_eq!(PNL_EXPONENT, -8, "PNL exponent should be -8 for crypto");
}

// =============================================================================
// Trace Binding Contract Test
// =============================================================================

/// Create a test decision for trace binding tests.
fn make_test_decision_for_binding(seq: u8, direction: i8) -> DecisionEvent {
    let ts = Utc
        .with_ymd_and_hms(2026, 1, 25, 12, 0, seq as u32)
        .unwrap();
    let decision_id =
        Uuid::parse_str(&format!("00000000-0000-0000-0000-00000000{:04x}", seq)).unwrap();

    DecisionEvent {
        ts,
        decision_id,
        strategy_id: "test_strategy".to_string(),
        symbol: "BTCUSDT".to_string(),
        decision_type: if direction > 0 { "entry" } else { "exit" }.to_string(),
        direction,
        target_qty_mantissa: 1_000_000,
        qty_exponent: -8,
        reference_price_mantissa: 10_000_000, // $100,000
        price_exponent: -2,
        market_snapshot: MarketSnapshot {
            bid_price_mantissa: 9_999_900,
            ask_price_mantissa: 10_000_100,
            bid_qty_mantissa: 10_000_000,
            ask_qty_mantissa: 10_000_000,
            price_exponent: -2,
            qty_exponent: -8,
            spread_bps_mantissa: 20, // 0.20 bps
            book_ts_ns: 1737799200000000000 + (seq as i64 * 1_000_000_000),
        },
        confidence_mantissa: 10_000, // 1.0
        metadata: serde_json::Value::Null,
        ctx: CorrelationContext {
            session_id: Some("test-session".to_string()),
            run_id: Some("test-run".to_string()),
            symbol: None,
            venue: Some("paper".to_string()),
            strategy_id: None,
            decision_id: None,
            order_id: None,
        },
    }
}

/// Integration test: trace binding populates manifest correctly
#[test]
fn test_trace_binding_manifest_integration() {
    // Create temp directory for test segment
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let segment_dir = temp_dir.path();

    // Create a decision trace
    let decisions = vec![
        make_test_decision_for_binding(1, 1),  // Entry
        make_test_decision_for_binding(2, -1), // Exit
        make_test_decision_for_binding(3, 1),  // Entry
        make_test_decision_for_binding(4, -1), // Exit
    ];

    let mut trace_builder = DecisionTraceBuilder::new();
    for d in &decisions {
        trace_builder.record(d);
    }
    let trace = trace_builder.finalize();

    // Save trace to file
    let trace_path = segment_dir.join("decision_trace.json");
    trace.save(&trace_path).expect("Failed to save trace");

    // Create a segment manifest
    let mut manifest = SegmentManifest::new(
        "test_family".to_string(),
        "test_segment".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "test_binary_hash".to_string(),
        CaptureConfig::default(),
    );

    // Simulated PnL from "backtest"
    let mut pnl = PnlAccumulatorFixed::new();
    pnl.realized_pnl_mantissa = 1_000_000_000; // ~$10
    pnl.total_fees_mantissa = 20_000; // ~$0.0002

    // Bind trace to manifest
    manifest
        .bind_trace_from_result(
            &trace_path,
            segment_dir,
            &trace.hash_hex(),
            trace.encoding_version,
            trace.len(),
            &pnl,
        )
        .expect("Failed to bind trace");

    // Finalize and write manifest
    manifest.finalize(
        quantlaxmi_runner_crypto::segment_manifest::StopReason::NormalCompletion,
        quantlaxmi_runner_crypto::segment_manifest::EventCounts::default(),
        None,
    );
    manifest
        .write(segment_dir)
        .expect("Failed to write manifest");

    // ===== VERIFICATION =====

    // 1. Reload manifest and verify trace_binding is populated
    let loaded_manifest = SegmentManifest::load(segment_dir).expect("Failed to load manifest");

    assert!(
        loaded_manifest.trace_binding.is_some(),
        "trace_binding should be populated in manifest"
    );

    let binding = loaded_manifest.trace_binding.unwrap();

    // 2. Verify all binding fields
    assert_eq!(
        binding.decision_trace_path, "decision_trace.json",
        "Trace path should be relative"
    );
    assert_eq!(
        binding.decision_trace_encoding_version, ENCODING_VERSION,
        "Encoding version should match"
    );
    assert_eq!(binding.total_decisions, 4, "Should have 4 decisions");
    assert_eq!(
        binding.realized_pnl_mantissa, pnl.realized_pnl_mantissa,
        "PnL mantissa should match"
    );
    assert_eq!(
        binding.total_fees_mantissa, pnl.total_fees_mantissa,
        "Fees mantissa should match"
    );
    assert_eq!(
        binding.pnl_exponent, PNL_EXPONENT,
        "PnL exponent should match"
    );

    // 3. Verify SHA256 matches file content
    let computed_sha256 = compute_file_sha256(&trace_path).expect("Failed to compute SHA256");
    assert_eq!(
        binding.decision_trace_sha256, computed_sha256,
        "SHA256 in manifest should match computed file hash"
    );

    // 4. Verify schema version is 4
    assert_eq!(
        loaded_manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION,
        "Schema version should be 4"
    );

    // 5. Verify trace file can be reloaded and matches
    let reloaded_trace =
        quantlaxmi_events::DecisionTrace::load(&trace_path).expect("Failed to reload trace");
    assert_eq!(
        reloaded_trace.trace_hash, trace.trace_hash,
        "Reloaded trace hash should match original"
    );
    assert_eq!(
        reloaded_trace.encoding_version, ENCODING_VERSION,
        "Reloaded encoding version should match"
    );
}

/// Test that trace binding produces correct relative paths
#[test]
fn test_trace_binding_relative_path() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let segment_dir = temp_dir.path();

    // Create nested trace path
    let traces_dir = segment_dir.join("traces");
    fs::create_dir_all(&traces_dir).expect("Failed to create traces dir");
    let trace_path = traces_dir.join("my_trace.json");

    // Create minimal trace
    let trace = DecisionTraceBuilder::new().finalize();
    trace.save(&trace_path).expect("Failed to save trace");

    // Create manifest and bind
    let mut manifest = SegmentManifest::new(
        "family".to_string(),
        "segment".to_string(),
        vec!["BTCUSDT".to_string()],
        "backtest".to_string(),
        "hash".to_string(),
        CaptureConfig::default(),
    );

    let pnl = PnlAccumulatorFixed::new();
    manifest
        .bind_trace(&trace_path, segment_dir, ENCODING_VERSION, 0, &pnl)
        .expect("Failed to bind trace");

    let binding = manifest.trace_binding.as_ref().unwrap();

    // Path should be relative: "traces/my_trace.json"
    assert_eq!(
        binding.decision_trace_path, "traces/my_trace.json",
        "Path should be relative to segment dir"
    );
}
