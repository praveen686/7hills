//! Phase 26.3: Truth Report Integration Tests
//!
//! Tests verify:
//! 1. Report files are written (strategy_truth_report.json, strategy_truth_summary.txt)
//! 2. JSON digest is stable across identical inputs
//! 3. Summary contains expected strategy_id
//! 4. Report structure matches Phase 26.2 spec

use quantlaxmi_eval::{SessionMetadata, StrategyAggregatorRegistry, StrategyTruthReport};
use quantlaxmi_models::PositionUpdateRecord;
use quantlaxmi_wal::{WalReader, WalWriter};
use tempfile::tempdir;

// =============================================================================
// Test 1: Truth report can be built from WAL position updates
// =============================================================================

#[test]
fn test_truth_report_from_position_updates() {
    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write position updates simulating a buy then sell (one round-trip trade)
        // Buy: position goes to +100
        let buy_update = PositionUpdateRecord::builder("funding_bias", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("test_session_001")
            .seq(1)
            .correlation_id("event_1")
            .fill_seq(1)
            .position_qty(10000000, -8) // +0.1 BTC
            .avg_price(4200000, -2)
            .cash_delta(-4200000, -2) // -$42,000
            .realized_pnl_delta(0, -2)
            .fee(420, -2) // $4.20 fee
            .venue("sim")
            .build();
        writer.write_position_update(buy_update).await.unwrap();

        // Sell: position goes to 0 (close trade with profit)
        let sell_update = PositionUpdateRecord::builder("funding_bias", "BTCUSDT")
            .ts_ns(1706400001000000000)
            .session_id("test_session_001")
            .seq(2)
            .correlation_id("event_2")
            .fill_seq(2)
            .position_qty(0, -8) // Back to 0
            .avg_price_flat(-2)
            .cash_delta(4300000, -2) // +$43,000 (sold at higher price)
            .realized_pnl_delta(10000, -2) // $100 profit
            .fee(430, -2) // $4.30 fee
            .venue("sim")
            .build();
        writer.write_position_update(sell_update).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read position updates from WAL
    let reader = WalReader::open(&session_dir).unwrap();
    let position_updates = reader.read_position_updates().unwrap();
    assert_eq!(position_updates.len(), 2, "Should have 2 position updates");

    // Build aggregator
    let mut registry = StrategyAggregatorRegistry::new();
    for record in &position_updates {
        registry.process_position_update(record).unwrap();
    }

    // Verify aggregator state
    let acc = registry.get("funding_bias").unwrap();
    assert_eq!(acc.trade_count, 1, "Should have 1 completed trade");
    assert_eq!(acc.winning_trades, 1, "Trade should be winning (pnl > 0)");
    assert_eq!(acc.losing_trades, 0);

    // Build truth report
    let metadata = SessionMetadata {
        session_id: "test_session_001".to_string(),
        instrument: "BTCUSDT".to_string(),
        start_ts_ns: 1706400000000000000,
        end_ts_ns: 1706400001000000000,
        latency_ticks: 0,
        cost_model_digest: None,
        unified_exponent: -2,
    };

    let accumulators = registry.finalize();
    let report = StrategyTruthReport::build(metadata, accumulators);

    // Verify report structure
    assert_eq!(report.schema_version, "1");
    assert_eq!(report.session_id, "test_session_001");
    assert_eq!(report.instrument, "BTCUSDT");
    assert!(report.strategies.contains_key("funding_bias"));

    let metrics = report.strategies.get("funding_bias").unwrap();
    assert_eq!(metrics.trades, 1);
    assert_eq!(metrics.winning_trades, 1);
    assert_eq!(metrics.win_rate, "1.00"); // 100% win rate
}

// =============================================================================
// Test 2: Digest is stable across identical inputs
// =============================================================================

#[test]
fn test_truth_report_digest_stability() {
    // Create identical inputs twice and verify same digest
    let build_report = || {
        let metadata = SessionMetadata {
            session_id: "digest_test_session".to_string(),
            instrument: "ETHUSDT".to_string(),
            start_ts_ns: 1000000000,
            end_ts_ns: 2000000000,
            latency_ticks: 3,
            cost_model_digest: Some("abc123".to_string()),
            unified_exponent: -2,
        };

        // Create accumulator with known state
        let mut registry = StrategyAggregatorRegistry::new();

        // Manually create a record to process
        let record = PositionUpdateRecord::builder("test_strat", "ETHUSDT")
            .ts_ns(1500000000)
            .session_id("digest_test_session")
            .seq(1)
            .correlation_id("event_1")
            .fill_seq(1)
            .position_qty(0, -2) // Close position
            .avg_price_flat(-2)
            .cash_delta(5000, -2)
            .realized_pnl_delta(500, -2)
            .fee(10, -2)
            .venue("sim")
            .build();

        registry.process_position_update(&record).unwrap();

        StrategyTruthReport::build(metadata, registry.finalize())
    };

    let report1 = build_report();
    let report2 = build_report();

    assert_eq!(
        report1.digest, report2.digest,
        "Identical inputs must produce identical digest"
    );

    // Verify digest is non-empty
    assert!(!report1.digest.is_empty());
    assert_eq!(report1.digest.len(), 64, "SHA-256 hex should be 64 chars");
}

// =============================================================================
// Test 3: Text summary contains expected fields
// =============================================================================

#[test]
fn test_truth_report_text_summary() {
    let metadata = SessionMetadata {
        session_id: "summary_test".to_string(),
        instrument: "BTCUSDT".to_string(),
        start_ts_ns: 1000000000,
        end_ts_ns: 2000000000,
        latency_ticks: 5,
        cost_model_digest: None,
        unified_exponent: -2,
    };

    let mut registry = StrategyAggregatorRegistry::new();

    // Add some data
    let record = PositionUpdateRecord::builder("my_strategy", "BTCUSDT")
        .ts_ns(1500000000)
        .session_id("summary_test")
        .seq(1)
        .correlation_id("event_1")
        .fill_seq(1)
        .position_qty(0, -2)
        .avg_price_flat(-2)
        .cash_delta(10000, -2)
        .realized_pnl_delta(1000, -2)
        .fee(50, -2)
        .venue("sim")
        .build();

    registry.process_position_update(&record).unwrap();

    let report = StrategyTruthReport::build(metadata, registry.finalize());
    let summary = report.to_text_summary();

    // Verify summary contains expected fields
    assert!(summary.contains("STRATEGY TRUTH REPORT"));
    assert!(summary.contains("summary_test"));
    assert!(summary.contains("BTCUSDT"));
    assert!(summary.contains("5 tick(s)"));
    assert!(summary.contains("my_strategy"));
    assert!(summary.contains("Trades:"));
    assert!(summary.contains("Win Rate:"));
    assert!(summary.contains("Gross PnL:"));
    assert!(summary.contains("Net PnL:"));
    assert!(summary.contains("Max Drawdown:"));
}

// =============================================================================
// Test 4: JSON roundtrip preserves data
// =============================================================================

#[test]
fn test_truth_report_json_roundtrip() {
    let metadata = SessionMetadata {
        session_id: "json_test".to_string(),
        instrument: "SOLUSDT".to_string(),
        start_ts_ns: 100,
        end_ts_ns: 200,
        latency_ticks: 1,
        cost_model_digest: Some("deadbeef".to_string()),
        unified_exponent: -8,
    };

    let mut registry = StrategyAggregatorRegistry::new();

    let record = PositionUpdateRecord::builder("sol_strat", "SOLUSDT")
        .ts_ns(150)
        .session_id("json_test")
        .seq(1)
        .correlation_id("evt")
        .fill_seq(1)
        .position_qty(0, -8)
        .avg_price_flat(-8)
        .cash_delta(1000000, -8)
        .realized_pnl_delta(500000, -8)
        .fee(1000, -8)
        .venue("sim")
        .build();

    registry.process_position_update(&record).unwrap();

    let report = StrategyTruthReport::build(metadata, registry.finalize());
    let json = report.to_json();

    // Parse back
    let parsed: StrategyTruthReport = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.session_id, report.session_id);
    assert_eq!(parsed.instrument, report.instrument);
    assert_eq!(parsed.latency_ticks, report.latency_ticks);
    assert_eq!(parsed.digest, report.digest);
    assert_eq!(parsed.strategies.len(), 1);
    assert!(parsed.strategies.contains_key("sol_strat"));
}

// =============================================================================
// Test 5: Empty strategies produces valid report
// =============================================================================

#[test]
fn test_truth_report_empty_strategies() {
    let metadata = SessionMetadata {
        session_id: "empty_test".to_string(),
        instrument: "UNKNOWN".to_string(),
        start_ts_ns: 0,
        end_ts_ns: 0,
        latency_ticks: 0,
        cost_model_digest: None,
        unified_exponent: -2,
    };

    let registry = StrategyAggregatorRegistry::new();
    let report = StrategyTruthReport::build(metadata, registry.finalize());

    assert!(report.strategies.is_empty());
    assert!(!report.digest.is_empty());

    // Should still produce valid JSON
    let json = report.to_json();
    assert!(
        json.contains("\"strategies\": {}") || json.contains("\"strategies\":{}"),
        "Empty strategies should be in JSON"
    );
}

// =============================================================================
// Test 6: Report files written to correct location
// =============================================================================

#[test]
fn test_truth_report_file_paths() {
    let temp_dir = tempdir().unwrap();
    let reports_dir = temp_dir.path().join("reports");

    // Simulate what Phase 26.3 runner wiring does
    std::fs::create_dir_all(&reports_dir).unwrap();

    let metadata = SessionMetadata {
        session_id: "file_test".to_string(),
        instrument: "BTCUSDT".to_string(),
        start_ts_ns: 0,
        end_ts_ns: 0,
        latency_ticks: 0,
        cost_model_digest: None,
        unified_exponent: -2,
    };

    let registry = StrategyAggregatorRegistry::new();
    let report = StrategyTruthReport::build(metadata, registry.finalize());

    // Write files
    let json_path = reports_dir.join("strategy_truth_report.json");
    let summary_path = reports_dir.join("strategy_truth_summary.txt");

    std::fs::write(&json_path, report.to_json()).unwrap();
    std::fs::write(&summary_path, report.to_text_summary()).unwrap();

    // Verify files exist
    assert!(json_path.exists(), "JSON report should exist");
    assert!(summary_path.exists(), "Summary should exist");

    // Verify contents are non-empty
    let json_content = std::fs::read_to_string(&json_path).unwrap();
    let summary_content = std::fs::read_to_string(&summary_path).unwrap();

    assert!(!json_content.is_empty());
    assert!(!summary_content.is_empty());
    assert!(json_content.contains("file_test"));
    assert!(summary_content.contains("file_test"));
}

// =============================================================================
// Test 7: Multi-strategy report
// =============================================================================

#[test]
fn test_truth_report_multi_strategy() {
    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("multi_strat");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Strategy A: 2 trades (open/close each), 1 win 1 loss
        // Trade 1: Open (no realized pnl)
        let a1_open = PositionUpdateRecord::builder("strat_A", "BTCUSDT")
            .ts_ns(1000)
            .session_id("multi")
            .seq(1)
            .correlation_id("a1_open")
            .fill_seq(1)
            .position_qty(100, -2) // Open position
            .avg_price(4200, -2)
            .cash_delta(-4200, -2)
            .realized_pnl_delta(0, -2)
            .fee_exponent(-2)
            .venue("sim")
            .build();
        writer.write_position_update(a1_open).await.unwrap();

        // Trade 1: Close (win)
        let a1_close = PositionUpdateRecord::builder("strat_A", "BTCUSDT")
            .ts_ns(1001)
            .session_id("multi")
            .seq(2)
            .correlation_id("a1_close")
            .fill_seq(2)
            .position_qty(0, -2) // Close position
            .avg_price_flat(-2)
            .cash_delta(4300, -2)
            .realized_pnl_delta(100, -2) // Win: +100
            .fee_exponent(-2)
            .venue("sim")
            .build();
        writer.write_position_update(a1_close).await.unwrap();

        // Trade 2: Open (no realized pnl)
        let a2_open = PositionUpdateRecord::builder("strat_A", "BTCUSDT")
            .ts_ns(2000)
            .session_id("multi")
            .seq(3)
            .correlation_id("a2_open")
            .fill_seq(3)
            .position_qty(100, -2) // Open position
            .avg_price(4500, -2)
            .cash_delta(-4500, -2)
            .realized_pnl_delta(0, -2)
            .fee_exponent(-2)
            .venue("sim")
            .build();
        writer.write_position_update(a2_open).await.unwrap();

        // Trade 2: Close (loss)
        let a2_close = PositionUpdateRecord::builder("strat_A", "BTCUSDT")
            .ts_ns(2001)
            .session_id("multi")
            .seq(4)
            .correlation_id("a2_close")
            .fill_seq(4)
            .position_qty(0, -2) // Close position
            .avg_price_flat(-2)
            .cash_delta(4450, -2)
            .realized_pnl_delta(-50, -2) // Loss: -50
            .fee_exponent(-2)
            .venue("sim")
            .build();
        writer.write_position_update(a2_close).await.unwrap();

        // Strategy B: 1 trade (open/close), win
        let b1_open = PositionUpdateRecord::builder("strat_B", "BTCUSDT")
            .ts_ns(1500)
            .session_id("multi")
            .seq(5)
            .correlation_id("b1_open")
            .fill_seq(5)
            .position_qty(200, -2)
            .avg_price(4100, -2)
            .cash_delta(-8200, -2)
            .realized_pnl_delta(0, -2)
            .fee_exponent(-2)
            .venue("sim")
            .build();
        writer.write_position_update(b1_open).await.unwrap();

        let b1_close = PositionUpdateRecord::builder("strat_B", "BTCUSDT")
            .ts_ns(1501)
            .session_id("multi")
            .seq(6)
            .correlation_id("b1_close")
            .fill_seq(6)
            .position_qty(0, -2)
            .avg_price_flat(-2)
            .cash_delta(8400, -2)
            .realized_pnl_delta(200, -2) // Win: +200
            .fee_exponent(-2)
            .venue("sim")
            .build();
        writer.write_position_update(b1_close).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Build report
    let reader = WalReader::open(&session_dir).unwrap();
    let position_updates = reader.read_position_updates().unwrap();

    let mut registry = StrategyAggregatorRegistry::new();
    for record in &position_updates {
        registry.process_position_update(record).unwrap();
    }

    let metadata = SessionMetadata {
        session_id: "multi".to_string(),
        instrument: "BTCUSDT".to_string(),
        start_ts_ns: 1000,
        end_ts_ns: 2000,
        latency_ticks: 0,
        cost_model_digest: None,
        unified_exponent: -2,
    };

    let report = StrategyTruthReport::build(metadata, registry.finalize());

    // Verify both strategies are in report (BTreeMap sorted order)
    assert_eq!(report.strategies.len(), 2);

    let keys: Vec<&String> = report.strategies.keys().collect();
    assert_eq!(keys, vec!["strat_A", "strat_B"]); // Sorted order

    let a_metrics = report.strategies.get("strat_A").unwrap();
    assert_eq!(a_metrics.trades, 2);
    assert_eq!(a_metrics.winning_trades, 1);
    assert_eq!(a_metrics.losing_trades, 1);
    assert_eq!(a_metrics.win_rate, "0.50"); // 50%

    let b_metrics = report.strategies.get("strat_B").unwrap();
    assert_eq!(b_metrics.trades, 1);
    assert_eq!(b_metrics.winning_trades, 1);
    assert_eq!(b_metrics.losing_trades, 0);
    assert_eq!(b_metrics.win_rate, "1.00"); // 100%
}
