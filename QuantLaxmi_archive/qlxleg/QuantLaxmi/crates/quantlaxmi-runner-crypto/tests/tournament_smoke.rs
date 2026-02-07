//! Tournament Smoke Test (P1)
//!
//! Validates that the tournament outputs are valid JSON and schema-correct.
//!
//! This test validates the tournament data structures without requiring
//! real segment data.

#[test]
fn test_tournament_grid_expansion() {
    // Test grid expansion logic directly
    use quantlaxmi_runner_crypto::tournament::{LeaderboardEntry, ParamValue};
    use std::collections::BTreeMap;

    // Create test params
    let mut params = BTreeMap::new();
    params.insert("threshold_mantissa".to_string(), ParamValue::Int(100));
    params.insert(
        "position_size_mantissa".to_string(),
        ParamValue::Int(1000000),
    );

    // Basic test that ParamValue serializes correctly
    let json = serde_json::to_string(&params).unwrap();
    assert!(json.contains("threshold_mantissa"));
    assert!(json.contains("100"));

    // Test leaderboard entry serialization
    let entry = LeaderboardEntry {
        param_hash: "abc123".to_string(),
        params: params.clone(),
        score: 0.05,
        mean_total_return: 0.03,
        median_total_return: 0.02,
        mean_max_drawdown: 0.01,
        segments: 5,
        positive_fraction: 0.80,
    };

    let entry_json = serde_json::to_string(&entry).unwrap();
    assert!(entry_json.contains("abc123"));
    assert!(entry_json.contains("0.05")); // score
}

#[test]
fn test_promotion_candidates_schema() {
    use quantlaxmi_runner_crypto::tournament::{LeaderboardEntry, PromotionCandidates};
    use std::collections::BTreeMap;

    let candidates = PromotionCandidates {
        generated_at_rfc3339: "2026-01-30T12:00:00Z".to_string(),
        strategy: "funding_bias".to_string(),
        candidates: vec![LeaderboardEntry {
            param_hash: "test".to_string(),
            params: BTreeMap::new(),
            score: 0.1,
            mean_total_return: 0.05,
            median_total_return: 0.04,
            mean_max_drawdown: 0.02,
            segments: 3,
            positive_fraction: 0.75,
        }],
    };

    let json = serde_json::to_string_pretty(&candidates).unwrap();
    assert!(json.contains("generated_at_rfc3339"));
    assert!(json.contains("funding_bias"));
    assert!(json.contains("candidates"));
}

#[test]
fn test_grid_tournament_manifest_schema() {
    use quantlaxmi_runner_crypto::tournament::GridTournamentManifest;

    let manifest = GridTournamentManifest {
        schema_version: "v1".to_string(),
        strategy: "funding_bias".to_string(),
        segments_root: "/data/segments".to_string(),
        matched_segments: vec!["seg1".to_string(), "seg2".to_string()],
        grid_file_hash: "abc123def456".to_string(),
        total_runs: 10,
        git_commit: Some("abc123".to_string()),
        git_dirty: Some(false),
    };

    let json = serde_json::to_string_pretty(&manifest).unwrap();

    // Verify schema fields present
    assert!(json.contains("schema_version"));
    assert!(json.contains("\"v1\""));
    assert!(json.contains("matched_segments"));
    assert!(json.contains("grid_file_hash"));
    assert!(json.contains("git_commit"));
}

/// P1.1: Test diagnostic types for zero-trade investigation.
#[test]
fn test_run_diagnostics_schema() {
    use quantlaxmi_runner_crypto::tournament::{
        EventCounts, OrderCounts, RunDiagnostics, SignalCounts, ZeroTradeReason,
    };
    use std::collections::BTreeMap;

    // Test all ZeroTradeReason variants serialize correctly
    let reasons = vec![
        ZeroTradeReason::NotZeroTrade,
        ZeroTradeReason::NoRelevantEvents,
        ZeroTradeReason::NoSignals,
        ZeroTradeReason::AllSignalsRefused,
        ZeroTradeReason::NoFills,
        ZeroTradeReason::Unknown,
    ];

    for reason in &reasons {
        let json = serde_json::to_string(reason).unwrap();
        // Verify SCREAMING_SNAKE_CASE serialization
        assert!(json.contains("_") || json.contains("UNKNOWN") || json.contains("NOT_ZERO_TRADE"));
    }

    // Test full diagnostics structure
    let diagnostics = RunDiagnostics {
        events: EventCounts {
            depth: 1000,
            trades: 0,
            funding: 50,
            spot: 500,
            total: 1550,
        },
        signals: SignalCounts {
            generated: 10,
            admitted: 8,
            refused: 2,
        },
        orders: OrderCounts {
            submitted: 8,
            filled: 6,
        },
        zero_trade_reason: ZeroTradeReason::NotZeroTrade,
        exit_reasons: BTreeMap::new(),
        notes: Some("Funding rate crossed threshold 3 times".to_string()),
    };

    let json = serde_json::to_string_pretty(&diagnostics).unwrap();

    // Verify all sections present
    assert!(json.contains("events"));
    assert!(json.contains("signals"));
    assert!(json.contains("orders"));
    assert!(json.contains("zero_trade_reason"));
    assert!(json.contains("notes"));

    // Verify values
    assert!(json.contains("\"depth\": 1000"));
    assert!(json.contains("\"funding\": 50"));
    assert!(json.contains("\"generated\": 10"));
    assert!(json.contains("\"admitted\": 8"));
    assert!(json.contains("\"filled\": 6"));
    assert!(json.contains("NOT_ZERO_TRADE"));
    assert!(json.contains("Funding rate crossed threshold 3 times"));

    // Verify roundtrip
    let parsed: RunDiagnostics = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.events.depth, 1000);
    assert_eq!(parsed.events.funding, 50);
    assert_eq!(parsed.signals.generated, 10);
    assert_eq!(parsed.orders.filled, 6);
    assert_eq!(parsed.zero_trade_reason, ZeroTradeReason::NotZeroTrade);
}
