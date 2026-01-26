//! Backtest Engine for Crypto Strategies
//!
//! Provides:
//! - `Strategy` trait for pluggable strategy logic
//! - `PaperExchange` for simulated order execution
//! - `BacktestEngine` to orchestrate replay → strategy → exchange
//!
//! ## Usage
//! ```ignore
//! let engine = BacktestEngine::new(config);
//! let result = engine.run(segment_dir, strategy)?;
//! println!("PnL: {:.2}", result.realized_pnl);
//! ```

use crate::replay::{EventKind, ReplayEvent, SegmentReplayAdapter};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Order and Fill Types
// =============================================================================

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Order intent from strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntent {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    /// Optional limit price. If None, treated as market order.
    pub limit_price: Option<f64>,
    /// Strategy-defined tag for tracking.
    pub tag: Option<String>,
}

impl OrderIntent {
    pub fn market(symbol: impl Into<String>, side: Side, qty: f64) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            qty,
            limit_price: None,
            tag: None,
        }
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }
}

/// Execution fill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    pub tag: Option<String>,
}

// =============================================================================
// Strategy Trait
// =============================================================================

/// Strategy interface for backtest and paper trading.
///
/// Implement this trait to create a trading strategy.
pub trait Strategy: Send {
    /// Called for each replay event. Return order intents to execute.
    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent>;

    /// Called when an order is filled.
    fn on_fill(&mut self, fill: &Fill) {
        let _ = fill; // Default: ignore
    }

    /// Strategy name for logging.
    fn name(&self) -> &str {
        "unnamed_strategy"
    }
}

// =============================================================================
// Paper Exchange
// =============================================================================

/// Per-symbol market state.
#[derive(Debug, Clone, Default)]
struct MarketState {
    bid: f64,
    ask: f64,
    last_update: Option<DateTime<Utc>>,
}

/// Per-symbol position.
#[derive(Debug, Clone, Default)]
struct Position {
    qty: f64,       // Positive = long, negative = short
    avg_price: f64, // Average entry price
}

/// Paper exchange configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    /// Fee in basis points (e.g., 10 = 0.1%)
    pub fee_bps: f64,
    /// Initial cash balance
    pub initial_cash: f64,
    /// Use perp prices for execution (vs spot)
    pub use_perp_prices: bool,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            fee_bps: 10.0, // 0.1% taker fee
            initial_cash: 10_000.0,
            use_perp_prices: true,
        }
    }
}

/// Extract bid/ask prices from event payload.
/// Handles both float format (bid/ask) and mantissa format (bid_price_mantissa + price_exponent).
fn extract_bid_ask(payload: &serde_json::Value, kind: EventKind) -> (f64, f64) {
    // Try float format first (simple quotes)
    if let (Some(bid), Some(ask)) = (
        payload.get("bid").and_then(|v| v.as_f64()),
        payload.get("ask").and_then(|v| v.as_f64()),
    ) {
        return (bid, ask);
    }

    // Try mantissa format (canonical quote schema)
    // price = mantissa * 10^price_exponent
    // e.g., mantissa=8874152, exponent=-2 → price = 8874152 * 0.01 = 88741.52
    let price_exp = payload
        .get("price_exponent")
        .and_then(|v| v.as_i64())
        .unwrap_or(-2) as i32;
    let scale = 10f64.powi(price_exp);

    match kind {
        EventKind::SpotQuote | EventKind::PerpQuote => {
            // Format: bid_price_mantissa, ask_price_mantissa
            let bid_mantissa = payload
                .get("bid_price_mantissa")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let ask_mantissa = payload
                .get("ask_price_mantissa")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let bid = bid_mantissa as f64 * scale;
            let ask = ask_mantissa as f64 * scale;
            (bid, ask)
        }
        EventKind::PerpDepth => {
            // Depth events come in two forms:
            // - is_snapshot=true: full book, bids sorted highest-first, asks sorted lowest-first
            // - is_snapshot=false: delta updates containing random book level changes
            //
            // ONLY use snapshots for price discovery. Deltas don't contain best bid/ask.
            // Return (0,0) for deltas so the strategy skips them.
            let is_snapshot = payload
                .get("is_snapshot")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !is_snapshot {
                return (0.0, 0.0); // Skip deltas
            }

            // Snapshot: first bid is best bid, first ask is best ask
            let bid_price = payload
                .get("bids")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("price"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let ask_price = payload
                .get("asks")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("price"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let bid = bid_price as f64 * scale;
            let ask = ask_price as f64 * scale;

            (bid, ask)
        }
        _ => (0.0, 0.0),
    }
}

/// Paper exchange for simulated execution.
pub struct PaperExchange {
    config: ExchangeConfig,
    markets: HashMap<String, MarketState>,
    positions: HashMap<String, Position>,
    cash: f64,
    realized_pnl: f64,
    fills: Vec<Fill>,
}

impl PaperExchange {
    pub fn new(config: ExchangeConfig) -> Self {
        Self {
            cash: config.initial_cash,
            config,
            markets: HashMap::new(),
            positions: HashMap::new(),
            realized_pnl: 0.0,
            fills: Vec::new(),
        }
    }

    /// Update market state from a replay event.
    pub fn update_market(&mut self, event: &ReplayEvent) {
        // Only update from quote events
        let dominated = match event.kind {
            EventKind::PerpQuote | EventKind::PerpDepth if self.config.use_perp_prices => false,
            EventKind::SpotQuote if !self.config.use_perp_prices => false,
            _ => true,
        };

        if dominated {
            return;
        }

        // Extract bid/ask from payload (handle both float and mantissa formats)
        let (bid, ask) = extract_bid_ask(&event.payload, event.kind);

        if bid > 0.0 && ask > 0.0 {
            let state = self.markets.entry(event.symbol.clone()).or_default();
            state.bid = bid;
            state.ask = ask;
            state.last_update = Some(event.ts);
        }
    }

    /// Execute an order intent. Returns fill if executed.
    pub fn execute(&mut self, intent: &OrderIntent, ts: DateTime<Utc>) -> Option<Fill> {
        let market = self.markets.get(&intent.symbol)?;

        // Determine execution price (cross the spread)
        let exec_price = match intent.side {
            Side::Buy => market.ask,  // Buy at ask
            Side::Sell => market.bid, // Sell at bid
        };

        if exec_price <= 0.0 {
            return None;
        }

        // Check limit price
        if let Some(limit) = intent.limit_price {
            match intent.side {
                Side::Buy if exec_price > limit => return None,
                Side::Sell if exec_price < limit => return None,
                _ => {}
            }
        }

        // Calculate notional and fee
        let notional = exec_price * intent.qty;
        let fee = notional * (self.config.fee_bps / 10_000.0);

        // Update position
        let position = self.positions.entry(intent.symbol.clone()).or_default();
        let old_qty = position.qty;

        match intent.side {
            Side::Buy => {
                // Check cash
                if self.cash < notional + fee {
                    tracing::warn!(
                        "Insufficient cash: need {:.2}, have {:.2}",
                        notional + fee,
                        self.cash
                    );
                    return None;
                }
                self.cash -= notional + fee;

                // Update position
                if position.qty >= 0.0 {
                    // Adding to long or opening long
                    let total_cost = position.avg_price * position.qty + exec_price * intent.qty;
                    position.qty += intent.qty;
                    position.avg_price = if position.qty > 0.0 {
                        total_cost / position.qty
                    } else {
                        0.0
                    };
                } else {
                    // Covering short
                    let cover_qty = intent.qty.min(-position.qty);
                    let pnl = (position.avg_price - exec_price) * cover_qty;
                    self.realized_pnl += pnl;
                    self.cash += pnl; // Return PnL to cash

                    position.qty += intent.qty;
                    if position.qty > 0.0 {
                        position.avg_price = exec_price;
                    }
                }
            }
            Side::Sell => {
                // Update position
                if position.qty <= 0.0 {
                    // Adding to short or opening short
                    let total_cost =
                        position.avg_price * (-position.qty) + exec_price * intent.qty;
                    position.qty -= intent.qty;
                    position.avg_price = if position.qty < 0.0 {
                        total_cost / (-position.qty)
                    } else {
                        0.0
                    };
                } else {
                    // Closing long
                    let close_qty = intent.qty.min(position.qty);
                    let pnl = (exec_price - position.avg_price) * close_qty;
                    self.realized_pnl += pnl;

                    position.qty -= intent.qty;
                    if position.qty < 0.0 {
                        position.avg_price = exec_price;
                    }
                }

                self.cash += notional - fee;
            }
        }

        let fill = Fill {
            ts,
            symbol: intent.symbol.clone(),
            side: intent.side,
            qty: intent.qty,
            price: exec_price,
            fee,
            tag: intent.tag.clone(),
        };

        tracing::debug!(
            "FILL: {} {} {:.4} @ {:.2} (fee={:.4}, pos: {:.4} -> {:.4})",
            fill.side,
            fill.symbol,
            fill.qty,
            fill.price,
            fill.fee,
            old_qty,
            position.qty
        );

        self.fills.push(fill.clone());
        Some(fill)
    }

    /// Get current position for a symbol.
    pub fn position(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map(|p| p.qty).unwrap_or(0.0)
    }

    /// Get current cash balance.
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get unrealized PnL across all positions.
    pub fn unrealized_pnl(&self) -> f64 {
        let mut pnl = 0.0;
        for (symbol, position) in &self.positions {
            if let Some(market) = self.markets.get(symbol) {
                let mid = (market.bid + market.ask) / 2.0;
                if position.qty > 0.0 {
                    pnl += (mid - position.avg_price) * position.qty;
                } else if position.qty < 0.0 {
                    pnl += (position.avg_price - mid) * (-position.qty);
                }
            }
        }
        pnl
    }

    /// Get total fills.
    pub fn fills(&self) -> &[Fill] {
        &self.fills
    }
}

// =============================================================================
// Backtest Engine
// =============================================================================

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub exchange: ExchangeConfig,
    /// Log progress every N events
    pub log_interval: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            exchange: ExchangeConfig::default(),
            log_interval: 100_000,
        }
    }
}

/// Backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub strategy_name: String,
    pub segment_path: String,
    pub total_events: usize,
    pub total_fills: usize,
    pub initial_cash: f64,
    pub final_cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub return_pct: f64,
    pub start_ts: Option<DateTime<Utc>>,
    pub end_ts: Option<DateTime<Utc>>,
    pub duration_secs: f64,
    pub fills: Vec<Fill>,
}

/// Backtest engine.
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on a segment with a strategy.
    pub fn run<S: Strategy>(
        &self,
        segment_dir: &Path,
        mut strategy: S,
    ) -> Result<BacktestResult> {
        let mut adapter = SegmentReplayAdapter::open(segment_dir)?;
        let mut exchange = PaperExchange::new(self.config.exchange.clone());

        let mut total_events = 0usize;
        let mut start_ts: Option<DateTime<Utc>> = None;
        let mut end_ts: Option<DateTime<Utc>> = None;

        tracing::info!(
            "Starting backtest: strategy={}, segment={:?}",
            strategy.name(),
            segment_dir
        );

        while let Some(event) = adapter.next_event()? {
            total_events += 1;

            if start_ts.is_none() {
                start_ts = Some(event.ts);
            }
            end_ts = Some(event.ts);

            // Update market state
            exchange.update_market(&event);

            // Get strategy orders
            let orders = strategy.on_event(&event);

            // Execute orders
            for order in orders {
                if let Some(fill) = exchange.execute(&order, event.ts) {
                    strategy.on_fill(&fill);
                }
            }

            // Progress logging
            if total_events % self.config.log_interval == 0 {
                tracing::info!(
                    "Progress: {} events, {} fills, PnL: {:.2}",
                    total_events,
                    exchange.fills().len(),
                    exchange.realized_pnl() + exchange.unrealized_pnl()
                );
            }
        }

        let realized = exchange.realized_pnl();
        let unrealized = exchange.unrealized_pnl();
        let total_pnl = realized + unrealized;
        let return_pct = (total_pnl / self.config.exchange.initial_cash) * 100.0;

        let duration_secs = match (start_ts, end_ts) {
            (Some(s), Some(e)) => (e - s).num_milliseconds() as f64 / 1000.0,
            _ => 0.0,
        };

        let result = BacktestResult {
            strategy_name: strategy.name().to_string(),
            segment_path: segment_dir.display().to_string(),
            total_events,
            total_fills: exchange.fills().len(),
            initial_cash: self.config.exchange.initial_cash,
            final_cash: exchange.cash(),
            realized_pnl: realized,
            unrealized_pnl: unrealized,
            total_pnl,
            return_pct,
            start_ts,
            end_ts,
            duration_secs,
            fills: exchange.fills().to_vec(),
        };

        tracing::info!("=== Backtest Complete ===");
        tracing::info!("Strategy: {}", result.strategy_name);
        tracing::info!("Events: {}", result.total_events);
        tracing::info!("Fills: {}", result.total_fills);
        tracing::info!("Duration: {:.1}s", result.duration_secs);
        tracing::info!("Realized PnL: ${:.2}", result.realized_pnl);
        tracing::info!("Unrealized PnL: ${:.2}", result.unrealized_pnl);
        tracing::info!("Total PnL: ${:.2} ({:.2}%)", result.total_pnl, result.return_pct);

        Ok(result)
    }
}

// =============================================================================
// Example Strategies
// =============================================================================

/// Simple funding rate strategy.
///
/// Goes short perp when funding rate is positive (longs pay shorts).
/// Goes flat when funding is negative.
pub struct FundingBiasStrategy {
    target_position: f64,
    current_funding_rate: f64,
    threshold: f64,
}

impl FundingBiasStrategy {
    pub fn new(target_position: f64, threshold: f64) -> Self {
        Self {
            target_position,
            current_funding_rate: 0.0,
            threshold,
        }
    }
}

impl Strategy for FundingBiasStrategy {
    fn name(&self) -> &str {
        "funding_bias"
    }

    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        // Update funding rate
        if event.kind == EventKind::Funding {
            if let Some(rate) = event.payload.get("rate").and_then(|v| v.as_f64()) {
                self.current_funding_rate = rate;
            }
        }

        // Only trade on perp quotes (to have price)
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth) {
            return vec![];
        }

        // Strategy logic: short when funding > threshold
        let mut orders = vec![];

        if self.current_funding_rate > self.threshold {
            // Funding positive, go short
            orders.push(
                OrderIntent::market(&event.symbol, Side::Sell, self.target_position)
                    .with_tag("funding_short"),
            );
        } else if self.current_funding_rate < -self.threshold {
            // Funding negative, go long
            orders.push(
                OrderIntent::market(&event.symbol, Side::Buy, self.target_position)
                    .with_tag("funding_long"),
            );
        }

        orders
    }
}

/// Spread capture strategy.
///
/// Trades when basis (perp - spot) exceeds threshold.
/// Uses actual position tracking from exchange fills.
pub struct BasisCaptureStrategy {
    // Price state
    spot_mid: f64,
    perp_mid: f64,

    // Parameters
    threshold_bps: f64,
    exit_threshold_bps: f64, // Exit when basis reverts to this level
    position_size: f64,

    // Position tracking (from fills, not assumptions)
    position_qty: f64,  // Positive = long, negative = short

    // Cooldown to prevent churn
    last_trade_ts: Option<DateTime<Utc>>,
    min_hold_ms: i64,

    // Debug counters
    events_seen: usize,
    signals_seen: usize,
    entries_emitted: usize,
    exits_emitted: usize,
    fills_received: usize,
}

impl BasisCaptureStrategy {
    pub fn new(threshold_bps: f64, position_size: f64) -> Self {
        Self {
            spot_mid: 0.0,
            perp_mid: 0.0,
            threshold_bps,
            exit_threshold_bps: threshold_bps / 2.0, // Exit at 1/2 of entry threshold
            position_size,
            position_qty: 0.0,
            last_trade_ts: None,
            min_hold_ms: 5_000, // 5 second minimum hold (validation mode)
            events_seen: 0,
            signals_seen: 0,
            entries_emitted: 0,
            exits_emitted: 0,
            fills_received: 0,
        }
    }

    fn basis_bps(&self) -> f64 {
        if self.spot_mid > 0.0 && self.perp_mid > 0.0 {
            ((self.perp_mid - self.spot_mid) / self.spot_mid) * 10_000.0
        } else {
            0.0
        }
    }

    fn is_flat(&self) -> bool {
        self.position_qty.abs() < 1e-9
    }

    fn is_long(&self) -> bool {
        self.position_qty > 1e-9
    }

    fn is_short(&self) -> bool {
        self.position_qty < -1e-9
    }

    fn can_trade(&self, ts: DateTime<Utc>) -> bool {
        match self.last_trade_ts {
            Some(last) => (ts - last).num_milliseconds() >= self.min_hold_ms,
            None => true,
        }
    }
}

impl Strategy for BasisCaptureStrategy {
    fn name(&self) -> &str {
        "basis_capture"
    }

    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        self.events_seen += 1;

        // Update prices using mantissa-aware extraction
        match event.kind {
            EventKind::SpotQuote => {
                let (bid, ask) = extract_bid_ask(&event.payload, event.kind);
                if bid > 0.0 && ask > 0.0 {
                    self.spot_mid = (bid + ask) / 2.0;
                }
            }
            EventKind::PerpQuote | EventKind::PerpDepth => {
                let (bid, ask) = extract_bid_ask(&event.payload, event.kind);
                if bid > 0.0 && ask > 0.0 {
                    self.perp_mid = (bid + ask) / 2.0;
                }
            }
            _ => {}
        }

        // Only act on perp updates (when we have fresh perp price)
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth) {
            return vec![];
        }

        // Need both prices to calculate basis
        if self.spot_mid <= 0.0 || self.perp_mid <= 0.0 {
            return vec![];
        }

        let basis = self.basis_bps();
        let mut orders = vec![];

        // Progress logging every 1M events
        if self.events_seen % 1_000_000 == 0 {
            eprintln!(
                "[STRATEGY] events={}M, signals={}, fills={}, pos={:.4}, basis={:.2}bps",
                self.events_seen / 1_000_000, self.signals_seen, self.fills_received,
                self.position_qty, basis
            );
        }

        // Check cooldown
        if !self.can_trade(event.ts) {
            return vec![];
        }

        // Entry signals (only when flat)
        if self.is_flat() {
            if basis > self.threshold_bps {
                // Basis too high: short perp (expect convergence down)
                self.signals_seen += 1;
                orders.push(
                    OrderIntent::market(&event.symbol, Side::Sell, self.position_size)
                        .with_tag("basis_short_entry"),
                );
                self.entries_emitted += 1;
                tracing::debug!(
                    "ENTRY SIGNAL: SHORT at basis={:.2}bps, threshold={:.2}bps",
                    basis, self.threshold_bps
                );
            } else if basis < -self.threshold_bps {
                // Basis too low: long perp (expect convergence up)
                self.signals_seen += 1;
                orders.push(
                    OrderIntent::market(&event.symbol, Side::Buy, self.position_size)
                        .with_tag("basis_long_entry"),
                );
                self.entries_emitted += 1;
                tracing::debug!(
                    "ENTRY SIGNAL: LONG at basis={:.2}bps, threshold={:.2}bps",
                    basis, self.threshold_bps
                );
            }
        }
        // Exit signals (when in position)
        else if self.is_short() && basis < self.exit_threshold_bps {
            // Was short, basis has converged down enough - cover
            self.signals_seen += 1;
            orders.push(
                OrderIntent::market(&event.symbol, Side::Buy, self.position_qty.abs())
                    .with_tag("basis_short_exit"),
            );
            self.exits_emitted += 1;
            tracing::debug!(
                "EXIT SIGNAL: COVER SHORT at basis={:.2}bps, exit_threshold={:.2}bps",
                basis, self.exit_threshold_bps
            );
        } else if self.is_long() && basis > -self.exit_threshold_bps {
            // Was long, basis has converged up enough - sell
            self.signals_seen += 1;
            orders.push(
                OrderIntent::market(&event.symbol, Side::Sell, self.position_qty.abs())
                    .with_tag("basis_long_exit"),
            );
            self.exits_emitted += 1;
            tracing::debug!(
                "EXIT SIGNAL: SELL LONG at basis={:.2}bps, exit_threshold={:.2}bps",
                basis, self.exit_threshold_bps
            );
        }

        orders
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.fills_received += 1;
        self.last_trade_ts = Some(fill.ts);

        // Update position from actual fill
        match fill.side {
            Side::Buy => self.position_qty += fill.qty,
            Side::Sell => self.position_qty -= fill.qty,
        }

        tracing::debug!(
            "FILL RECEIVED: {} {:.6} @ {:.2}, new_pos={:.6}",
            fill.side, fill.qty, fill.price, self.position_qty
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_exchange_basic() {
        let mut exchange = PaperExchange::new(ExchangeConfig {
            fee_bps: 10.0,
            initial_cash: 10_000.0,
            use_perp_prices: true,
        });

        // Simulate market update
        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({"bid": 100000.0, "ask": 100010.0}),
        };
        exchange.update_market(&event);

        // Execute buy
        let order = OrderIntent::market("BTCUSDT", Side::Buy, 0.1);
        let fill = exchange.execute(&order, Utc::now());

        assert!(fill.is_some());
        let fill = fill.unwrap();
        assert_eq!(fill.price, 100010.0); // Bought at ask
        assert_eq!(fill.qty, 0.1);
        assert!(fill.fee > 0.0);

        assert_eq!(exchange.position("BTCUSDT"), 0.1);
    }

    #[test]
    fn test_pnl_calculation() {
        let mut exchange = PaperExchange::new(ExchangeConfig {
            fee_bps: 0.0, // No fees for simple test
            initial_cash: 10_000.0,
            use_perp_prices: true,
        });

        // Buy at 100
        let event1 = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({"bid": 100.0, "ask": 100.0}),
        };
        exchange.update_market(&event1);
        exchange.execute(&OrderIntent::market("BTCUSDT", Side::Buy, 1.0), Utc::now());

        // Price moves to 110
        let event2 = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({"bid": 110.0, "ask": 110.0}),
        };
        exchange.update_market(&event2);

        // Check unrealized PnL
        assert!((exchange.unrealized_pnl() - 10.0).abs() < 0.01);

        // Sell to realize
        exchange.execute(&OrderIntent::market("BTCUSDT", Side::Sell, 1.0), Utc::now());

        // Check realized PnL
        assert!((exchange.realized_pnl() - 10.0).abs() < 0.01);
        assert_eq!(exchange.position("BTCUSDT"), 0.0);
    }
}
