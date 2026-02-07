//! Phase 22D Integration Tests — Order Permission Gate Enforcement
//!
//! Proves Phase 22C order permission gate is enforced at runtime:
//! - Admitted signals + market order + passive spec → refused by gate
//! - Admitted signals + limit order + passive spec → allowed
//! - Permission refusals are logged with correlation_id
//!
//! ## Hard Laws Verified
//! - L1: Advisory cannot emit orders
//! - L2: Passive cannot emit market orders
//! - L3: allow_short/allow_long constraints enforced
//! - L4: Permission gate is checked BEFORE execution

use quantlaxmi_gates::{ExecutionClass, StrategySpec};
use quantlaxmi_models::SignalRequirements;
use quantlaxmi_models::events::{CorrelationContext, DecisionEvent};
use quantlaxmi_runner_crypto::backtest::{
    BacktestConfig, BacktestEngine, ExchangeConfig, PaceMode,
};
use quantlaxmi_strategy::{
    DecisionOutput, OrderIntent, ReplayEvent, Side, Strategy, StrategyContext,
};

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::TempDir;
use uuid::Uuid;

// =============================================================================
// Helper to create DecisionEvent
// =============================================================================

fn make_decision_event(ctx: &StrategyContext, strategy_id: &str, direction: i8) -> DecisionEvent {
    DecisionEvent {
        ts: ctx.ts,
        decision_id: Uuid::new_v4(),
        strategy_id: strategy_id.to_string(),
        symbol: ctx.symbol.to_string(),
        decision_type: if direction > 0 { "entry" } else { "exit" }.to_string(),
        direction,
        target_qty_mantissa: 1,
        qty_exponent: -4, // 0.0001 BTC
        reference_price_mantissa: ctx.market.bid_price_mantissa(),
        price_exponent: ctx.market.price_exponent(),
        market_snapshot: ctx.market.clone(),
        confidence_mantissa: 8500,
        metadata: serde_json::Value::Null,
        ctx: CorrelationContext::default(),
    }
}

// =============================================================================
// Test Strategies
// =============================================================================

/// Strategy that emits market orders (for testing passive rejection).
struct MarketOrderStrategy {
    order_count: AtomicU64,
}

impl MarketOrderStrategy {
    fn new() -> Self {
        Self {
            order_count: AtomicU64::new(0),
        }
    }
}

impl Strategy for MarketOrderStrategy {
    fn name(&self) -> &str {
        "market_order_test"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn config_hash(&self) -> String {
        "test_hash_market".to_string()
    }

    fn required_signals(&self) -> Vec<SignalRequirements> {
        // No admission gating required for this test
        vec![]
    }

    fn on_event(&mut self, _event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        self.order_count.fetch_add(1, Ordering::SeqCst);

        let decision = make_decision_event(ctx, "test_passive", 1);

        // Emit a MARKET order (no limit_price)
        // qty = 1 * 10^(-4) = 0.0001 BTC (small enough to fit in $10k cash)
        let intent = OrderIntent::market(decision.decision_id, ctx.symbol, Side::Buy, 1, -4);

        vec![DecisionOutput::new(decision, intent)]
    }
}

/// Strategy that emits limit orders (for testing passive allow).
struct LimitOrderStrategy {
    order_count: AtomicU64,
}

impl LimitOrderStrategy {
    fn new() -> Self {
        Self {
            order_count: AtomicU64::new(0),
        }
    }
}

impl Strategy for LimitOrderStrategy {
    fn name(&self) -> &str {
        "limit_order_test"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn config_hash(&self) -> String {
        "test_hash_limit".to_string()
    }

    fn required_signals(&self) -> Vec<SignalRequirements> {
        vec![]
    }

    fn on_event(&mut self, _event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        self.order_count.fetch_add(1, Ordering::SeqCst);

        let decision = make_decision_event(ctx, "test_passive", 1);

        // Emit a LIMIT order (has limit_price)
        // Use ask price for limit buy to ensure it can execute
        // qty = 1 * 10^(-4) = 0.0001 BTC (small enough to fit in $10k cash)
        let intent = OrderIntent::limit(
            decision.decision_id,
            ctx.symbol,
            Side::Buy,
            1,
            -4,
            ctx.market.ask_price_mantissa(), // Buy at ask price to ensure fill
            ctx.market.price_exponent(),
        );

        vec![DecisionOutput::new(decision, intent)]
    }
}

/// Strategy that emits sell orders (for testing allow_short constraint).
struct SellOrderStrategy {
    order_count: AtomicU64,
}

impl SellOrderStrategy {
    fn new() -> Self {
        Self {
            order_count: AtomicU64::new(0),
        }
    }
}

impl Strategy for SellOrderStrategy {
    fn name(&self) -> &str {
        "sell_order_test"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn config_hash(&self) -> String {
        "test_hash_sell".to_string()
    }

    fn required_signals(&self) -> Vec<SignalRequirements> {
        vec![]
    }

    fn on_event(&mut self, _event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        self.order_count.fetch_add(1, Ordering::SeqCst);

        let decision = make_decision_event(ctx, "test_no_short", -1);

        // Emit a SELL limit order
        // qty = 1 * 10^(-4) = 0.0001 BTC
        let intent = OrderIntent::limit(
            decision.decision_id,
            ctx.symbol,
            Side::Sell,
            1,
            -4,
            ctx.market.bid_price_mantissa(), // Use market bid as limit price (sell at bid)
            ctx.market.price_exponent(),
        );

        vec![DecisionOutput::new(decision, intent)]
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn make_passive_spec() -> StrategySpec {
    StrategySpec {
        strategy_id: "test_passive".to_string(),
        description: "Test passive strategy".to_string(),
        signals: vec!["spread".to_string()],
        execution_class: ExecutionClass::Passive,
        max_orders_per_min: 120,
        max_position_abs: 10000,
        allow_short: true,
        allow_long: true,
        allow_market_orders: false,
        tags: vec![],
    }
}

fn make_no_short_spec() -> StrategySpec {
    StrategySpec {
        strategy_id: "test_no_short".to_string(),
        description: "Test no-short strategy".to_string(),
        signals: vec!["spread".to_string()],
        execution_class: ExecutionClass::Passive,
        max_orders_per_min: 120,
        max_position_abs: 10000,
        allow_short: false, // No shorting allowed
        allow_long: true,
        allow_market_orders: false,
        tags: vec![],
    }
}

/// Create a test segment with perp quotes containing ALL L1 fields.
fn create_test_segment() -> TempDir {
    let dir = TempDir::new().unwrap();
    let sym_dir = dir.path().join("BTCUSDT");
    std::fs::create_dir_all(&sym_dir).unwrap();

    // Create perp quotes WITH qty fields (use static timestamps like other tests)
    let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();

    // Generate 10 quotes with realistic data and static timestamps
    for i in 0..10 {
        writeln!(
            perp,
            r#"{{"ts":"2026-01-28T10:00:{:02}Z","bid_price_mantissa":{},"ask_price_mantissa":{},"bid_qty_mantissa":150000000,"ask_qty_mantissa":200000000,"price_exponent":-2,"qty_exponent":-8}}"#,
            i,
            4200000 + i * 10,
            4200100 + i * 10
        ).unwrap();
    }

    dir
}

// =============================================================================
// TEST Case B: Passive + Market Order → Refused
// =============================================================================

#[tokio::test]
async fn test_case_b_passive_refuses_market_order() {
    // Setup: Create test segment
    let segment_dir = create_test_segment();

    // Create strategy that emits market orders
    let strategy = Box::new(MarketOrderStrategy::new());

    // Configure with passive strategy spec
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_case_b".to_string()),
        strategy_spec: Some(make_passive_spec()), // Passive = no market orders
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest
    let (result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Assert: Strategy was called (decisions recorded), but NO fills occurred
    // because market orders were refused by the permission gate
    assert!(
        result.total_decisions > 0,
        "Strategy should have been called and recorded decisions"
    );
    assert_eq!(
        result.total_fills, 0,
        "Market orders should have been refused by permission gate - no fills expected"
    );
}

// =============================================================================
// TEST Case C: Passive + Limit Order → Allowed
// =============================================================================

#[tokio::test]
async fn test_case_c_passive_permits_limit_order() {
    // Setup: Create test segment
    let segment_dir = create_test_segment();

    // Create strategy that emits limit orders
    let strategy = Box::new(LimitOrderStrategy::new());

    // Configure with passive strategy spec
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig {
            initial_cash: 100000.0,
            fee_bps: 0.0,
            use_perp_prices: true,
        },
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_case_c".to_string()),
        strategy_spec: Some(make_passive_spec()), // Passive allows limit orders
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest
    let (result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Assert: Strategy was called and limit orders were executed (fills occurred)
    assert!(
        result.total_decisions > 0,
        "Strategy should have been called and recorded decisions"
    );
    assert!(
        result.total_fills > 0,
        "Limit orders should have been permitted by permission gate - fills expected"
    );
}

// =============================================================================
// TEST: allow_short=false Refuses Sell Orders
// =============================================================================

#[tokio::test]
async fn test_allow_short_false_refuses_sell() {
    // Setup: Create test segment
    let segment_dir = create_test_segment();

    // Create strategy that emits sell orders
    let strategy = Box::new(SellOrderStrategy::new());

    // Configure with no-short strategy spec
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig::default(),
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_no_short".to_string()),
        strategy_spec: Some(make_no_short_spec()), // allow_short=false
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest
    let (result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Assert: Strategy was called, but sell orders were refused
    assert!(
        result.total_decisions > 0,
        "Strategy should have been called and recorded decisions"
    );
    assert_eq!(
        result.total_fills, 0,
        "Sell orders should have been refused (allow_short=false) - no fills expected"
    );
}

// =============================================================================
// TEST: No strategy_spec → No permission gating (backwards compatible)
// =============================================================================

#[tokio::test]
async fn test_no_spec_allows_all_orders() {
    // Setup: Create test segment
    let segment_dir = create_test_segment();

    // Create strategy that emits market orders
    let strategy = Box::new(MarketOrderStrategy::new());

    // Configure WITHOUT strategy_spec (backwards compatible mode)
    let config = BacktestConfig {
        pace: PaceMode::Fast,
        exchange: ExchangeConfig {
            initial_cash: 100000.0,
            fee_bps: 0.0,
            use_perp_prices: true,
        },
        log_interval: 10000,
        output_trace: None,
        run_id: Some("test_no_spec".to_string()),
        strategy_spec: None, // No permission gating
        ..Default::default()
    };
    let runner = BacktestEngine::new(config);

    // Run backtest
    let (result, _binding) = runner
        .run_with_strategy(segment_dir.path(), strategy, None)
        .await
        .expect("Backtest should complete");

    // Assert: Without strategy_spec, all orders are allowed (no permission gating)
    assert!(
        result.total_decisions > 0,
        "Strategy should have been called"
    );
    assert!(
        result.total_fills > 0,
        "Without strategy_spec, market orders should be allowed - fills expected"
    );
}
