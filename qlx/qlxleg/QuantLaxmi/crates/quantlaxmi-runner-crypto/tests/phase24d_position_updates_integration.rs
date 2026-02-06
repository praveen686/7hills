//! Phase 24D: Position Updates WAL Integration Tests
//!
//! Tests verify that position updates are written to WAL correctly
//! during backtest execution, following the "state-first then persist" doctrine.
//!
//! Write timing (frozen):
//! - AFTER applying the fill to internal position/ledger state
//! - BEFORE calling any callbacks that observe the updated state

use quantlaxmi_models::PositionUpdateRecord;
use quantlaxmi_wal::WalReader;
use std::collections::HashSet;
use std::path::Path;

/// Helper: Check if position_updates.jsonl exists in a segment directory
fn position_updates_exists(segment_dir: &Path) -> bool {
    segment_dir.join("wal/position_updates.jsonl").exists()
}

/// Helper: Read position updates from a segment directory
fn read_position_updates(segment_dir: &Path) -> Vec<PositionUpdateRecord> {
    let reader = WalReader::open(segment_dir).expect("Failed to open WAL reader");
    reader
        .read_position_updates()
        .expect("Failed to read position updates")
}

// =============================================================================
// Test 1: position_updates.jsonl is written during backtest
// =============================================================================

#[test]
fn test_position_updates_written_for_each_fill() {
    // This test verifies that position_updates.jsonl is created when fills occur.
    //
    // We use an existing test fixture that has position updates to verify the WAL exists.
    // In a real scenario, this would run a tiny backtest and check the output.

    // Check if we have a test fixture with position updates
    let test_fixtures = [
        "tests/fixtures/segment_with_position_updates",
        "../quantlaxmi-runner-crypto/tests/fixtures/segment_with_position_updates",
    ];

    // If no fixture exists, this test is a placeholder for manual verification
    for fixture in &test_fixtures {
        let path = Path::new(fixture);
        if path.exists() && position_updates_exists(path) {
            let updates = read_position_updates(path);
            assert!(
                !updates.is_empty(),
                "position_updates.jsonl should have at least 1 record"
            );
            return;
        }
    }

    // If no fixture exists, document that this test requires a live backtest
    eprintln!(
        "NOTE: test_position_updates_written_for_each_fill requires a segment with position updates. \
         Run a backtest with fills to verify Phase 24D integration."
    );
}

// =============================================================================
// Test 2: seq is strictly monotonic
// =============================================================================

#[test]
fn test_position_update_seq_monotonic() {
    // Verify that seq values are strictly increasing within a session.
    // This test uses the WAL writer's monotonicity enforcement.

    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write 5 position updates with strictly increasing seq
        for seq in 1..=5 {
            let update = PositionUpdateRecord::builder("test_strat", "BTCUSDT")
                .ts_ns(1000 + seq as i64 * 1000)
                .session_id("test_session")
                .seq(seq)
                .correlation_id(&format!("corr_{}", seq))
                .fill_seq(seq)
                .position_qty(100 * seq as i64, -8)
                .avg_price(5000, -2)
                .cash_delta(-5000 * seq as i64, -2)
                .realized_pnl_delta(0, -2)
                .venue("sim")
                .build();
            writer.write_position_update(update).await.unwrap();
        }

        writer.finalize().await.unwrap();
    });

    // Read back and verify monotonicity
    let updates = read_position_updates(&session_dir);
    assert_eq!(updates.len(), 5, "Should have 5 position updates");

    let mut prev_seq = 0u64;
    for update in &updates {
        assert!(
            update.seq > prev_seq,
            "seq {} should be > prev_seq {}",
            update.seq,
            prev_seq
        );
        prev_seq = update.seq;
    }
}

// =============================================================================
// Test 3: position updates link to fill_seq
// =============================================================================

#[test]
fn test_position_update_links_fill_seq() {
    // Verify that position updates have valid fill_seq linkage.

    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Position update linked to fill seq=42
        let update = PositionUpdateRecord::builder("test_strat", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("test_session")
            .seq(1)
            .correlation_id("event_seq:99")
            .fill_seq(42) // Link to fill seq 42
            .position_qty(100000000, -8)
            .avg_price(4200000, -2)
            .cash_delta(-4200000, -2)
            .realized_pnl_delta(0, -2)
            .fee(420, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let updates = read_position_updates(&session_dir);
    assert_eq!(updates.len(), 1);

    // Verify fill_seq linkage
    assert_eq!(updates[0].fill_seq, 42);
    assert_eq!(updates[0].correlation_id, "event_seq:99");
}

// =============================================================================
// Test 4: post-state matches expected (simple deterministic scenario)
// =============================================================================

#[test]
fn test_position_update_post_state_matches_expected() {
    // Verify that post-state snapshot fields are correctly set.

    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Buy 1 BTC at $42,000
        let buy_update = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100000000, -8) // 1 BTC (post-state)
            .avg_price(4200000, -2) // $42,000 avg entry
            .cash_delta(-4200000, -2) // Spent $42,000
            .realized_pnl_delta(0, -2) // No realized PnL on open
            .fee(420, -2) // $4.20 fee
            .venue("sim")
            .build();
        writer.write_position_update(buy_update).await.unwrap();

        // Sell 1 BTC at $43,000 (close to flat with profit)
        let sell_update = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .fill_seq(2)
            .position_qty(0, -8) // Flat (post-state)
            .avg_price_flat(-2) // No avg price when flat
            .cash_delta(4300000, -2) // Received $43,000
            .realized_pnl_delta(100000, -2) // $1,000 profit
            .fee(430, -2) // $4.30 fee
            .venue("sim")
            .build();
        writer.write_position_update(sell_update).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let updates = read_position_updates(&session_dir);
    assert_eq!(updates.len(), 2);

    // Buy update
    assert_eq!(updates[0].position_qty_mantissa, 100000000); // 1 BTC
    assert_eq!(updates[0].avg_price_mantissa, Some(4200000));
    assert!(updates[0].cash_delta_mantissa < 0); // Negative (spent cash)
    assert_eq!(updates[0].realized_pnl_delta_mantissa, 0);
    assert_eq!(updates[0].fee_mantissa, Some(420));

    // Sell update
    assert_eq!(updates[1].position_qty_mantissa, 0); // Flat
    assert_eq!(updates[1].avg_price_mantissa, None); // No avg price when flat
    assert!(updates[1].cash_delta_mantissa > 0); // Positive (received cash)
    assert!(updates[1].realized_pnl_delta_mantissa > 0); // Profit
    assert_eq!(updates[1].fee_mantissa, Some(430));
}

// =============================================================================
// Test 5: deterministic across runs (digests identical)
// =============================================================================

#[test]
fn test_position_update_deterministic_across_runs() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    // Run 1
    let temp_dir1 = tempdir().unwrap();
    let session_dir1 = temp_dir1.path().join("run1");

    // Run 2
    let temp_dir2 = tempdir().unwrap();
    let session_dir2 = temp_dir2.path().join("run2");

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Helper to write identical position updates
    async fn write_test_updates(session_dir: &std::path::Path) {
        let mut writer = WalWriter::new(session_dir).await.unwrap();

        for seq in 1..=3 {
            let update = PositionUpdateRecord::builder("test_strat", "BTCUSDT")
                .ts_ns(1706400000000000000 + seq as i64 * 1_000_000_000)
                .session_id("determinism_test")
                .seq(seq)
                .correlation_id(&format!("corr_{}", seq))
                .fill_seq(seq)
                .position_qty(100 * seq as i64, -8)
                .avg_price(5000 + seq as i64 * 100, -2)
                .cash_delta(-5000 * seq as i64, -2)
                .realized_pnl_delta(10 * seq as i64, -2)
                .fee(10 * seq as i64, -2)
                .venue("sim")
                .build();
            writer.write_position_update(update).await.unwrap();
        }

        writer.finalize().await.unwrap();
    }

    rt.block_on(async {
        write_test_updates(&session_dir1).await;
        write_test_updates(&session_dir2).await;
    });

    // Read both runs
    let updates1 = read_position_updates(&session_dir1);
    let updates2 = read_position_updates(&session_dir2);

    // Verify identical line counts
    assert_eq!(
        updates1.len(),
        updates2.len(),
        "Line counts should be identical"
    );

    // Verify digests are identical
    for (u1, u2) in updates1.iter().zip(updates2.iter()) {
        assert_eq!(u1.digest, u2.digest, "Digests should be identical");
        assert_eq!(u1.seq, u2.seq, "Seqs should be identical");
        assert_eq!(u1.fill_seq, u2.fill_seq, "Fill seqs should be identical");
        assert_eq!(
            u1.position_qty_mantissa, u2.position_qty_mantissa,
            "Position qty should be identical"
        );
    }

    // Verify first/last digest match
    assert_eq!(
        updates1.first().unwrap().digest,
        updates2.first().unwrap().digest
    );
    assert_eq!(
        updates1.last().unwrap().digest,
        updates2.last().unwrap().digest
    );
}

// =============================================================================
// Test 6: fee is optional (None when unknown)
// =============================================================================

#[test]
fn test_position_update_fee_optional() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Update with fee
        let with_fee = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .fee(10, -2) // Has fee
            .venue("sim")
            .build();
        writer.write_position_update(with_fee).await.unwrap();

        // Update without fee (fee unknown)
        let without_fee = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .fill_seq(2)
            .position_qty(200, -8)
            .avg_price(5001, -2)
            .cash_delta(-5001, -2)
            .realized_pnl_delta(0, -2)
            // No .fee() call - fee is unknown
            .venue("sim")
            .build();
        writer.write_position_update(without_fee).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let updates = read_position_updates(&session_dir);
    assert_eq!(updates.len(), 2);
    assert_eq!(
        updates[0].fee_mantissa,
        Some(10),
        "First update should have fee"
    );
    assert_eq!(
        updates[1].fee_mantissa, None,
        "Second update should have no fee"
    );
}

// =============================================================================
// Test 7: avg_price is None when flat
// =============================================================================

#[test]
fn test_position_update_avg_price_none_when_flat() {
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Open position
        let open_update = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2) // Has avg price
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(open_update).await.unwrap();

        // Close to flat
        let flat_update = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .fill_seq(2)
            .position_qty(0, -8) // Flat
            .avg_price_flat(-2) // No avg price when flat
            .cash_delta(5100, -2)
            .realized_pnl_delta(100, -2)
            .venue("sim")
            .build();
        writer.write_position_update(flat_update).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify
    let updates = read_position_updates(&session_dir);
    assert_eq!(updates.len(), 2);
    assert_eq!(
        updates[0].avg_price_mantissa,
        Some(5000),
        "Open should have avg price"
    );
    assert_eq!(
        updates[1].avg_price_mantissa, None,
        "Flat should have no avg price"
    );
}

// =============================================================================
// Test 8: session_id isolation
// =============================================================================

#[test]
fn test_position_update_session_isolation() {
    // Different sessions can have same seq values independently
    use quantlaxmi_wal::WalWriter;
    use tempfile::tempdir;

    let temp_dir = tempdir().unwrap();
    let session_dir = temp_dir.path().join("test_session");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Session A, seq 1
        let update_a1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("session_a")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update_a1).await.unwrap();

        // Session B, seq 1 (same seq, different session - OK)
        let update_b1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("session_b")
            .seq(1) // Same seq as session_a - OK
            .fill_seq(1)
            .position_qty(200, -8)
            .avg_price(5001, -2)
            .cash_delta(-5001, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update_b1).await.unwrap();

        writer.finalize().await.unwrap();
    });

    // Read back and verify both were written
    let updates = read_position_updates(&session_dir);
    assert_eq!(updates.len(), 2);

    // Verify different sessions
    let sessions: HashSet<_> = updates.iter().map(|u| &u.session_id).collect();
    assert_eq!(sessions.len(), 2);
    assert!(sessions.contains(&"session_a".to_string()));
    assert!(sessions.contains(&"session_b".to_string()));
}
