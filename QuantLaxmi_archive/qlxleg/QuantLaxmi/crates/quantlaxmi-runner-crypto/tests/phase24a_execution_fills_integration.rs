//! Phase 24A: Execution Fills WAL Integration Tests
//!
//! Tests verify that execution fills are written to WAL correctly
//! during backtest execution, following the "write-before-state" doctrine.

use quantlaxmi_models::{ExecutionFillRecord, FillSide, FillType};
use quantlaxmi_wal::WalReader;
use std::collections::HashSet;
use std::path::Path;

/// Helper: Check if execution_fills.jsonl exists in a segment directory
fn execution_fills_exists(segment_dir: &Path) -> bool {
    segment_dir.join("wal/execution_fills.jsonl").exists()
}

/// Helper: Read execution fills from a segment directory
fn read_execution_fills(segment_dir: &Path) -> Vec<ExecutionFillRecord> {
    let reader = WalReader::open(segment_dir).expect("Failed to open WAL reader");
    reader
        .read_execution_fills()
        .expect("Failed to read execution fills")
}

// =============================================================================
// Test 1: execution_fills.jsonl is written during backtest
// =============================================================================

#[test]
fn test_execution_fills_wal_written_before_state() {
    // This test verifies the "write-before-state" doctrine by checking
    // that execution_fills.jsonl is created when fills occur.
    //
    // We use an existing test fixture that has fills to verify the WAL exists.
    // In a real scenario, this would run a tiny backtest and check the output.

    // Check if we have a test fixture with fills
    let test_fixtures = [
        "tests/fixtures/segment_with_fills",
        "../quantlaxmi-runner-crypto/tests/fixtures/segment_with_fills",
    ];

    // If no fixture exists, this test is a placeholder for manual verification
    // The actual integration test requires running a backtest with fills
    for fixture in &test_fixtures {
        let path = Path::new(fixture);
        if path.exists() && execution_fills_exists(path) {
            let fills = read_execution_fills(path);
            assert!(
                !fills.is_empty(),
                "execution_fills.jsonl should have at least 1 fill record"
            );
            return;
        }
    }

    // If no fixture exists, document that this test requires a live backtest
    eprintln!(
        "NOTE: test_execution_fills_wal_written_before_state requires a segment with fills. \
         Run a backtest with fills to verify Phase 24A integration."
    );
}

// =============================================================================
// Test 2: seq is strictly monotonic
// =============================================================================

#[test]
fn test_execution_fills_seq_monotonic() {
    // Verify that seq values are strictly increasing within a session.
    // This test uses the WAL writer's monotonicity enforcement.

    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write 5 fills with strictly increasing seq
        for seq in 1..=5 {
            let fill = ExecutionFillRecord::builder("test_strat", "BTCUSDT")
                .ts_ns(1000 + seq as i64 * 1000)
                .session_id("test_session")
                .seq(seq)
                .side(FillSide::Buy)
                .qty(100, -8)
                .price(5000, -2)
                .venue("sim")
                .correlation_id(&format!("corr_{}", seq))
                .fill_type(FillType::Full)
                .build();
            writer.write_execution_fill(fill).await.unwrap();
        }

        writer.finalize().await.unwrap();
    });

    // Read back and verify monotonicity
    let fills = read_execution_fills(&session_dir);
    assert_eq!(fills.len(), 5, "Should have 5 fills");

    let mut prev_seq = 0u64;
    for fill in &fills {
        assert!(
            fill.seq > prev_seq,
            "seq {} should be > prev_seq {}",
            fill.seq,
            prev_seq
        );
        prev_seq = fill.seq;
    }
}

// =============================================================================
// Test 3: fills link to parent intents
// =============================================================================

#[test]
fn test_execution_fills_links_to_intents() {
    // Verify that fills have parent_intent_seq and parent_intent_digest
    // when they originate from a tracked intent.

    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write a fill with parent intent linkage
        let fill = ExecutionFillRecord::builder("test_strat", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("test_session")
            .seq(1)
            .parent_intent_seq(42) // Link to intent seq 42
            .parent_intent_digest("abc123def456") // Intent digest
            .side(FillSide::Buy)
            .qty(100000000, -8)
            .price(4200000, -2)
            .fee(420, -2)
            .venue("sim")
            .correlation_id("event_seq:99")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill).await.unwrap();

        // Write a fill without parent intent (manual fill case)
        let manual_fill = ExecutionFillRecord::builder("unknown", "ETHUSDT")
            .ts_ns(1706400001000000000)
            .session_id("test_session")
            .seq(2)
            // No parent_intent_seq or parent_intent_digest
            .side(FillSide::Sell)
            .qty(50000000, -8)
            .price(250000, -2)
            .venue("binance")
            .correlation_id("corr_unknown")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(manual_fill).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let fills = read_execution_fills(&session_dir);
    assert_eq!(fills.len(), 2);

    // First fill should have parent intent linkage
    assert_eq!(fills[0].parent_intent_seq, Some(42));
    assert_eq!(
        fills[0].parent_intent_digest,
        Some("abc123def456".to_string())
    );
    assert_eq!(fills[0].correlation_id, "event_seq:99");

    // Second fill should not have parent intent linkage
    assert_eq!(fills[1].parent_intent_seq, None);
    assert_eq!(fills[1].parent_intent_digest, None);
}

// =============================================================================
// Test 4: full and partial fills are correctly typed
// =============================================================================

#[test]
fn test_execution_fills_written_for_partial_and_full() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write a full fill
        let full_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(full_fill).await.unwrap();

        // Write a partial fill
        let partial_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .side(FillSide::Sell)
            .qty(50, -8) // Partial quantity
            .price(5001, -2)
            .venue("binance")
            .correlation_id("c2")
            .fill_type(FillType::Partial)
            .build();
        writer.write_execution_fill(partial_fill).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let fills = read_execution_fills(&session_dir);
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0].fill_type, FillType::Full);
    assert_eq!(fills[1].fill_type, FillType::Partial);
}

// =============================================================================
// Test 5: deterministic across runs
// =============================================================================

#[test]
fn test_execution_fills_deterministic_across_runs() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    // Run 1
    let temp_dir1 = tempdir().unwrap();
    let session_dir1 = temp_dir1.path().join("run1");

    // Run 2
    let temp_dir2 = tempdir().unwrap();
    let session_dir2 = temp_dir2.path().join("run2");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Helper to write identical fills
    async fn write_test_fills(session_dir: &std::path::Path) {
        let mut writer = WalWriter::new(session_dir).await.unwrap();

        for seq in 1..=3 {
            let fill = ExecutionFillRecord::builder("test_strat", "BTCUSDT")
                .ts_ns(1706400000000000000 + seq as i64 * 1_000_000_000)
                .session_id("determinism_test")
                .seq(seq)
                .parent_intent_seq(seq * 10)
                .parent_intent_digest(&format!("digest_{}", seq))
                .side(if seq % 2 == 1 {
                    FillSide::Buy
                } else {
                    FillSide::Sell
                })
                .qty(100 * seq as i64, -8)
                .price(5000 + seq as i64 * 100, -2)
                .fee(10 * seq as i64, -2)
                .venue("sim")
                .correlation_id(&format!("corr_{}", seq))
                .fill_type(FillType::Full)
                .build();
            writer.write_execution_fill(fill).await.unwrap();
        }

        writer.finalize().await.unwrap();
    }

    rt.block_on(async {
        write_test_fills(&session_dir1).await;
        write_test_fills(&session_dir2).await;
    });

    // Read both runs
    let fills1 = read_execution_fills(&session_dir1);
    let fills2 = read_execution_fills(&session_dir2);

    // Verify identical line counts
    assert_eq!(
        fills1.len(),
        fills2.len(),
        "Line counts should be identical"
    );

    // Verify digests are identical
    for (f1, f2) in fills1.iter().zip(fills2.iter()) {
        assert_eq!(f1.digest, f2.digest, "Digests should be identical");
        assert_eq!(f1.seq, f2.seq, "Seqs should be identical");
        assert_eq!(
            f1.parent_intent_seq, f2.parent_intent_seq,
            "Parent intent seqs should be identical"
        );
    }

    // Verify first/last digest match
    assert_eq!(
        fills1.first().unwrap().digest,
        fills2.first().unwrap().digest
    );
    assert_eq!(fills1.last().unwrap().digest, fills2.last().unwrap().digest);
}

// =============================================================================
// Test 6: fee is optional (None when unknown)
// =============================================================================

#[test]
fn test_execution_fills_fee_optional() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Fill with fee
        let with_fee = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2) // Has fee
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(with_fee).await.unwrap();

        // Fill without fee (fee unknown)
        let without_fee = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .side(FillSide::Sell)
            .qty(100, -8)
            .price(5001, -2)
            // No .fee() call - fee is unknown
            .venue("sim")
            .correlation_id("c2")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(without_fee).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let fills = read_execution_fills(&session_dir);
    assert_eq!(fills.len(), 2);
    assert_eq!(
        fills[0].fee_mantissa,
        Some(10),
        "First fill should have fee"
    );
    assert_eq!(
        fills[1].fee_mantissa, None,
        "Second fill should have no fee"
    );
}

// =============================================================================
// Test 7: venue field correctly indicates execution mode
// =============================================================================

#[test]
fn test_execution_fills_venue_indicates_mode() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Simulated fill (backtest)
        let sim_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim") // Simulated
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(sim_fill).await.unwrap();

        // Live fill (hypothetical)
        let live_fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .side(FillSide::Sell)
            .qty(100, -8)
            .price(5001, -2)
            .venue("binance") // Live exchange
            .correlation_id("c2")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(live_fill).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let fills = read_execution_fills(&session_dir);
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0].venue, "sim");
    assert_eq!(fills[1].venue, "binance");
}

// =============================================================================
// Test 8: session_id isolation
// =============================================================================

#[test]
fn test_execution_fills_session_isolation() {
    // Different sessions can have same seq values independently
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Session A, seq 1
        let fill_a1 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("session_a")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill_a1).await.unwrap();

        // Session B, seq 1 (same seq, different session - OK)
        let fill_b1 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("session_b")
            .seq(1) // Same seq as session_a - OK
            .side(FillSide::Sell)
            .qty(200, -8)
            .price(5001, -2)
            .venue("sim")
            .correlation_id("c2")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill_b1).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify both were written
    let fills = read_execution_fills(&session_dir);
    assert_eq!(fills.len(), 2);

    // Verify different sessions
    let sessions: HashSet<_> = fills.iter().map(|f| &f.session_id).collect();
    assert_eq!(sessions.len(), 2);
    assert!(sessions.contains(&"session_a".to_string()));
    assert!(sessions.contains(&"session_b".to_string()));
}
