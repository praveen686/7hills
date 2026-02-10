//! Paper ↔ Backtest Parity Test
//!
//! This test proves that paper trading and backtest produce IDENTICAL results
//! when given the same depth stream and order sequence.
//!
//! ## What This Tests
//! - Same depth events → same order book state
//! - Same order submission → same fills
//! - Same fills → same position/PnL
//!
//! ## Why This Matters
//! If paper != backtest, then:
//! - Backtest PnL is meaningless
//! - Live trading will diverge from expectations
//! - Trust in the entire profitability pipeline breaks

use chrono::Utc;
use quantlaxmi_models::depth::{DepthEvent, DepthLevel, IntegrityTier};
use quantlaxmi_runner_crypto::paper_trading::PaperVenue;
use quantlaxmi_runner_crypto::sim::{Order, Side, SimConfig, Simulator};

/// Create a test depth event.
fn make_depth(symbol: &str, bid: i64, ask: i64, update_id: u64, is_snapshot: bool) -> DepthEvent {
    DepthEvent {
        ts: Utc::now(),
        tradingsymbol: symbol.to_string(),
        first_update_id: update_id,
        last_update_id: update_id,
        price_exponent: -2,
        qty_exponent: -8,
        bids: vec![DepthLevel {
            price: bid,
            qty: 100_000_000, // 1.0 BTC
        }],
        asks: vec![DepthLevel {
            price: ask,
            qty: 100_000_000,
        }],
        is_snapshot,
        integrity_tier: IntegrityTier::Certified,
        source: None,
    }
}

/// Test: Paper and backtest produce identical fills from same depth stream.
#[test]
fn test_parity_same_depth_same_fills() {
    let cfg = SimConfig {
        fee_bps_maker: 2.0,
        fee_bps_taker: 10.0,
        latency_ticks: 0,
        allow_partial_fills: true,
        initial_cash: 100_000.0,
    };

    // Create backtest simulator (direct)
    let mut backtest_sim = Simulator::new(cfg.clone());

    // Create paper venue (wraps Simulator)
    let mut paper_venue = PaperVenue::with_config(cfg);

    // Same depth stream
    let depth_events = vec![
        make_depth("BTCUSDT", 8999000, 9000000, 1, true), // bid=89990, ask=90000
        make_depth("BTCUSDT", 8999500, 9000500, 2, false), // bid=89995, ask=90005
        make_depth("BTCUSDT", 9000000, 9001000, 3, false), // bid=90000, ask=90010
    ];

    // Apply same depth to both
    for ev in &depth_events {
        backtest_sim.on_depth("BTCUSDT", ev);
        paper_venue.apply_depth_event(ev).unwrap();
    }

    // Same order: market buy
    let order_id = 1;
    let backtest_order = Order::market(order_id, "BTCUSDT", Side::Buy, 0.5);
    let paper_fills = {
        paper_venue.submit_market_order("BTCUSDT".to_string(), Side::Buy, 0.5);
        paper_venue.fills().to_vec()
    };

    let ts = Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64;
    let backtest_fills = backtest_sim.submit(ts, backtest_order);

    // PARITY CHECK: Same number of fills
    assert_eq!(
        paper_fills.len(),
        backtest_fills.len(),
        "Fill count mismatch: paper={}, backtest={}",
        paper_fills.len(),
        backtest_fills.len()
    );

    // PARITY CHECK: Same fill prices
    for (p, b) in paper_fills.iter().zip(backtest_fills.iter()) {
        assert!(
            (p.price - b.price).abs() < 0.01,
            "Fill price mismatch: paper={}, backtest={}",
            p.price,
            b.price
        );
        assert!(
            (p.qty - b.qty).abs() < 0.0001,
            "Fill qty mismatch: paper={}, backtest={}",
            p.qty,
            b.qty
        );
        assert!(
            (p.fee - b.fee).abs() < 0.01,
            "Fill fee mismatch: paper={}, backtest={}",
            p.fee,
            b.fee
        );
    }

    // PARITY CHECK: Same position
    let paper_pos = paper_venue.position();
    let backtest_pos = backtest_sim.position("BTCUSDT");
    assert!(
        (paper_pos - backtest_pos).abs() < 0.0001,
        "Position mismatch: paper={}, backtest={}",
        paper_pos,
        backtest_pos
    );

    // PARITY CHECK: Same realized PnL (should be 0 since no round trip yet)
    let paper_pnl = paper_venue.realized_pnl();
    let backtest_pnl = backtest_sim.realized_pnl();
    assert!(
        (paper_pnl - backtest_pnl).abs() < 0.01,
        "Realized PnL mismatch: paper={}, backtest={}",
        paper_pnl,
        backtest_pnl
    );

    println!("PARITY_CHECK_PASSED: paper == backtest");
    println!("  Fill count: {}", paper_fills.len());
    println!("  Position: {:.4}", paper_pos);
    println!("  Realized PnL: {:.2}", paper_pnl);
}

/// Test: Limit orders behave identically in paper and backtest.
#[test]
fn test_parity_limit_orders() {
    let cfg = SimConfig {
        fee_bps_maker: 2.0,
        fee_bps_taker: 10.0,
        latency_ticks: 0,
        allow_partial_fills: true,
        initial_cash: 100_000.0,
    };

    let mut backtest_sim = Simulator::new(cfg.clone());
    let mut paper_venue = PaperVenue::with_config(cfg);

    // Initial depth: bid=89990, ask=90010
    let ev1 = make_depth("BTCUSDT", 8999000, 9001000, 1, true);
    backtest_sim.on_depth("BTCUSDT", &ev1);
    paper_venue.apply_depth_event(&ev1).unwrap();

    // Submit limit buy at 90005 (doesn't cross ask of 90010)
    let ts = Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64;
    let order = Order::limit(1, "BTCUSDT", Side::Buy, 0.3, 90005.0);
    let backtest_fills = backtest_sim.submit(ts, order);
    paper_venue.submit_order("BTCUSDT".to_string(), Side::Buy, 90005.0, 0.3);

    // PARITY: Both should have 0 fills (order is pending)
    assert_eq!(
        backtest_fills.len(),
        0,
        "Backtest should have pending order"
    );
    assert_eq!(
        paper_venue.fills().len(),
        0,
        "Paper should have pending order"
    );

    // PARITY: Both should have 1 pending order
    assert_eq!(backtest_sim.pending_order_count(), 1);
    assert_eq!(paper_venue.pending_orders().len(), 1);

    // Price drops: ask moves to 90003 (below our limit of 90005)
    let ev2 = make_depth("BTCUSDT", 8998000, 9000300, 2, false);
    let backtest_fills = backtest_sim.on_depth("BTCUSDT", &ev2);
    let paper_fills = paper_venue.apply_depth_event(&ev2).unwrap();

    // PARITY: Both should fill as maker
    assert_eq!(
        paper_fills.len(),
        backtest_fills.len(),
        "Fill count mismatch after price drop"
    );
    assert_eq!(paper_fills.len(), 1, "Should have 1 fill");

    // PARITY: Both should use maker fee
    for (p, b) in paper_fills.iter().zip(backtest_fills.iter()) {
        assert!(
            (p.price - b.price).abs() < 0.01,
            "Fill price mismatch: paper={}, backtest={}",
            p.price,
            b.price
        );
        // Maker fee = 2 bps = 0.02%
        // Notional = 90003 * 0.3 = 27000.9
        // Fee = 27000.9 * 0.0002 = 5.40
        assert!(
            (p.fee - b.fee).abs() < 0.01,
            "Fee mismatch: paper={}, backtest={}",
            p.fee,
            b.fee
        );
    }

    // PARITY: No more pending orders
    assert_eq!(backtest_sim.pending_order_count(), 0);
    assert_eq!(paper_venue.pending_orders().len(), 0);

    println!("PARITY_CHECK_PASSED: limit orders behave identically");
}

/// Test: Round-trip PnL is identical in paper and backtest.
#[test]
fn test_parity_round_trip_pnl() {
    let cfg = SimConfig {
        fee_bps_maker: 0.0, // No fees for clean PnL test
        fee_bps_taker: 0.0,
        latency_ticks: 0,
        allow_partial_fills: true,
        initial_cash: 100_000.0,
    };

    let mut backtest_sim = Simulator::new(cfg.clone());
    let mut paper_venue = PaperVenue::with_config(cfg);

    // Buy at 90000
    let ev1 = make_depth("BTCUSDT", 8999900, 9000000, 1, true);
    backtest_sim.on_depth("BTCUSDT", &ev1);
    paper_venue.apply_depth_event(&ev1).unwrap();

    let ts = Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64;
    backtest_sim.submit(ts, Order::market(1, "BTCUSDT", Side::Buy, 1.0));
    paper_venue.submit_market_order("BTCUSDT".to_string(), Side::Buy, 1.0);

    // Sell at 91000 (profit of $1000)
    let ev2 = make_depth("BTCUSDT", 9100000, 9100100, 2, false);
    backtest_sim.on_depth("BTCUSDT", &ev2);
    paper_venue.apply_depth_event(&ev2).unwrap();

    backtest_sim.submit(ts + 1, Order::market(2, "BTCUSDT", Side::Sell, 1.0));
    paper_venue.submit_market_order("BTCUSDT".to_string(), Side::Sell, 1.0);

    // PARITY: Realized PnL
    let paper_pnl = paper_venue.realized_pnl();
    let backtest_pnl = backtest_sim.realized_pnl();

    assert!(
        (paper_pnl - backtest_pnl).abs() < 0.01,
        "PnL mismatch: paper={:.2}, backtest={:.2}",
        paper_pnl,
        backtest_pnl
    );

    // Should be ~1000 profit (sold at 91000, bought at 90000)
    assert!(
        (paper_pnl - 1000.0).abs() < 1.0,
        "Expected ~1000 PnL, got {:.2}",
        paper_pnl
    );

    // PARITY: Position should be 0
    assert!(
        paper_venue.position().abs() < 0.0001,
        "Paper position should be 0"
    );
    assert!(
        backtest_sim.position("BTCUSDT").abs() < 0.0001,
        "Backtest position should be 0"
    );

    println!("PARITY_CHECK_PASSED: round-trip PnL identical");
    println!("  Realized PnL: {:.2}", paper_pnl);
}
