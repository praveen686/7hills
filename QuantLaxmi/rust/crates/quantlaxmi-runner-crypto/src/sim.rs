//! Unified Execution Simulator
//!
//! **SINGLE SOURCE OF TRUTH** for simulated order execution in:
//! - Backtest (replay mode)
//! - Paper trading (live mode with WAL)
//!
//! ## Design Goals
//! - Determinism: Mantissa-based order book for exact replay
//! - Unified matching: Same fill logic for backtest AND paper trading
//! - Pending orders: Limit orders that wait for price to cross
//!
//! ## Architecture
//! ```text
//! DepthEvent → Simulator.on_depth() → check_pending_fills()
//!                   ↓                        ↓
//!              OrderBook update         Fill events
//!                   ↓
//!            SimState (inventory, cash, positions)
//! ```
//!
//! ## Matching Rules (maker/taker)
//! - Market orders: ALWAYS taker (fill at best opposite side)
//! - Limit orders at submit:
//!   - If crosses spread → taker (immediate fill)
//!   - Else → queued as pending (maker when filled)
//! - Pending limit orders: maker fee when filled

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

use quantlaxmi_models::depth::DepthEvent;

// =============================================================================
// Configuration
// =============================================================================

/// Simulator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Maker fee in basis points (limit orders that add liquidity)
    pub fee_bps_maker: f64,
    /// Taker fee in basis points (market orders / crossing spread)
    pub fee_bps_taker: f64,
    /// Latency in ticks before orders can be filled (0 = immediate)
    pub latency_ticks: u64,
    /// Allow partial fills (if false, orders fill completely or not at all)
    pub allow_partial_fills: bool,
    /// Initial cash balance
    pub initial_cash: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            fee_bps_maker: 2.0,  // 0.02% maker fee (typical for VIP)
            fee_bps_taker: 10.0, // 0.1% taker fee
            latency_ticks: 0,
            allow_partial_fills: true,
            initial_cash: 10_000.0,
        }
    }
}

// =============================================================================
// Order Types
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

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    /// Market order - crosses the spread immediately
    Market,
    /// Limit order - only fills at specified price or better
    Limit,
}

/// Whether a fill was maker or taker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillType {
    Maker,
    Taker,
}

/// Order to be submitted to the simulator.
#[derive(Debug, Clone)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub qty: f64,
    /// Limit price (required for Limit orders)
    pub limit_price: Option<f64>,
    /// Optional tag for tracking
    pub tag: Option<String>,
    /// Timestamp when order was created
    pub created_at: DateTime<Utc>,
}

impl Order {
    /// Create a market order.
    pub fn market(id: u64, symbol: impl Into<String>, side: Side, qty: f64) -> Self {
        Self {
            id,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            qty,
            limit_price: None,
            tag: None,
            created_at: Utc::now(),
        }
    }

    /// Create a limit order.
    pub fn limit(
        id: u64,
        symbol: impl Into<String>,
        side: Side,
        qty: f64,
        limit_price: f64,
    ) -> Self {
        Self {
            id,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Limit,
            qty,
            limit_price: Some(limit_price),
            tag: None,
            created_at: Utc::now(),
        }
    }

    /// Add a tag for tracking.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }
}

// =============================================================================
// Pending Order (internal)
// =============================================================================

/// Pending limit order waiting for price to cross.
#[derive(Debug, Clone)]
#[allow(dead_code)] // qty_mantissa preserved for future order fill quantity tracking
struct PendingOrder {
    order: Order,
    /// Price as mantissa (for deterministic matching)
    price_mantissa: i64,
    /// Quantity as mantissa
    qty_mantissa: i64,
    /// Tick when order was submitted (for latency)
    submitted_tick: u64,
}

// =============================================================================
// Fill
// =============================================================================

/// Execution fill from the simulator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub order_id: u64,
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    pub fill_type: FillType,
    pub ts_ns: u64,
    /// Optional tag copied from the order
    pub tag: Option<String>,
}

// =============================================================================
// State
// =============================================================================

/// Position for a single symbol.
#[derive(Debug, Clone, Default)]
pub struct Position {
    /// Position quantity (positive = long, negative = short)
    pub qty: f64,
    /// Average entry price
    pub avg_price: f64,
}

/// Simulator state (portfolio).
#[derive(Debug, Default)]
pub struct SimState {
    /// Cash balance
    pub cash: f64,
    /// Inventory per symbol
    pub inventory: HashMap<String, f64>,
    /// Full position tracking with avg price
    pub positions: HashMap<String, Position>,
    /// Realized PnL
    pub realized_pnl: f64,
}

impl SimState {
    /// Create a new state with initial cash.
    pub fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            inventory: HashMap::new(),
            positions: HashMap::new(),
            realized_pnl: 0.0,
        }
    }

    /// Get position for a symbol (0 if not present).
    pub fn position(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map(|p| p.qty).unwrap_or(0.0)
    }
}

// =============================================================================
// Order Book State (per symbol)
// =============================================================================

/// Per-symbol order book state (mantissa-based for determinism).
#[derive(Debug, Clone, Default)]
pub struct OrderBook {
    /// Bids: price (mantissa) -> qty (mantissa), sorted descending
    bids: BTreeMap<i64, i64>,
    /// Asks: price (mantissa) -> qty (mantissa), sorted ascending
    asks: BTreeMap<i64, i64>,
    /// Last update ID for sequence validation
    last_update_id: u64,
    /// Price exponent
    price_exponent: i8,
    /// Quantity exponent
    qty_exponent: i8,
}

impl OrderBook {
    /// Create a new order book.
    pub fn new(price_exponent: i8, qty_exponent: i8) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            price_exponent,
            qty_exponent,
        }
    }

    /// Convert mantissa to float.
    #[inline]
    pub fn mantissa_to_f64(&self, mantissa: i64, is_price: bool) -> f64 {
        let exp = if is_price {
            self.price_exponent
        } else {
            self.qty_exponent
        };
        mantissa as f64 * 10f64.powi(exp as i32)
    }

    /// Convert float to mantissa.
    #[inline]
    pub fn f64_to_mantissa(&self, value: f64, is_price: bool) -> i64 {
        let exp = if is_price {
            self.price_exponent
        } else {
            self.qty_exponent
        };
        (value / 10f64.powi(exp as i32)).round() as i64
    }

    /// Apply a depth event to update the order book.
    pub fn apply_depth(&mut self, event: &DepthEvent) {
        // Handle snapshot vs diff
        if event.is_snapshot {
            self.bids.clear();
            self.asks.clear();
        }

        // Update exponents (should be consistent)
        self.price_exponent = event.price_exponent;
        self.qty_exponent = event.qty_exponent;

        // Apply bid updates
        for level in &event.bids {
            if level.qty == 0 {
                self.bids.remove(&level.price);
            } else {
                self.bids.insert(level.price, level.qty);
            }
        }

        // Apply ask updates
        for level in &event.asks {
            if level.qty == 0 {
                self.asks.remove(&level.price);
            } else {
                self.asks.insert(level.price, level.qty);
            }
        }

        self.last_update_id = event.last_update_id;
    }

    /// Update from float bid/ask (for backward compat with ReplayEvent).
    pub fn update_quote(&mut self, bid: f64, ask: f64) {
        // Clear and set single level at each side
        self.bids.clear();
        self.asks.clear();

        if bid > 0.0 {
            let bid_m = self.f64_to_mantissa(bid, true);
            let qty_m = self.f64_to_mantissa(1.0, false); // Assume liquidity
            self.bids.insert(bid_m, qty_m);
        }
        if ask > 0.0 {
            let ask_m = self.f64_to_mantissa(ask, true);
            let qty_m = self.f64_to_mantissa(1.0, false);
            self.asks.insert(ask_m, qty_m);
        }
    }

    /// Get best bid as (price_mantissa, qty_mantissa).
    pub fn best_bid(&self) -> Option<(i64, i64)> {
        self.bids.iter().next_back().map(|(&p, &q)| (p, q))
    }

    /// Get best ask as (price_mantissa, qty_mantissa).
    pub fn best_ask(&self) -> Option<(i64, i64)> {
        self.asks.iter().next().map(|(&p, &q)| (p, q))
    }

    /// Get best bid as f64.
    pub fn best_bid_f64(&self) -> Option<f64> {
        self.best_bid().map(|(p, _)| self.mantissa_to_f64(p, true))
    }

    /// Get best ask as f64.
    pub fn best_ask_f64(&self) -> Option<f64> {
        self.best_ask().map(|(p, _)| self.mantissa_to_f64(p, true))
    }

    /// Get last update ID.
    pub fn last_update_id(&self) -> u64 {
        self.last_update_id
    }

    /// Get price exponent.
    pub fn price_exponent(&self) -> i8 {
        self.price_exponent
    }

    /// Get qty exponent.
    pub fn qty_exponent(&self) -> i8 {
        self.qty_exponent
    }
}

// =============================================================================
// Simulator
// =============================================================================

/// Unified execution simulator.
///
/// **SINGLE SOURCE OF TRUTH** for order matching in both backtest and paper trading.
///
/// All order matching, position tracking, and PnL calculation happens here.
pub struct Simulator {
    cfg: SimConfig,
    state: SimState,
    /// Order books per symbol
    books: HashMap<String, OrderBook>,
    /// Pending limit orders (FIFO queue per symbol)
    pending_orders: HashMap<String, VecDeque<PendingOrder>>,
    /// All fills
    fills: Vec<Fill>,
    /// Current tick (for latency simulation)
    current_tick: u64,
    /// Next order ID
    next_order_id: u64,
}

impl Simulator {
    /// Create a new simulator.
    pub fn new(cfg: SimConfig) -> Self {
        let initial_cash = cfg.initial_cash;
        Self {
            cfg,
            state: SimState::new(initial_cash),
            books: HashMap::new(),
            pending_orders: HashMap::new(),
            fills: Vec::new(),
            current_tick: 0,
            next_order_id: 1,
        }
    }

    /// Process a depth event (mantissa-based).
    ///
    /// This updates the order book AND checks if any pending orders can fill.
    /// Returns fills generated from pending orders.
    pub fn on_depth(&mut self, symbol: &str, ev: &DepthEvent) -> Vec<Fill> {
        // Sequence validation
        if let Some(book) = self.books.get(symbol)
            && !ev.is_snapshot
            && ev.first_update_id != book.last_update_id() + 1
        {
            tracing::warn!(
                "Sequence gap: expected {}, got {}",
                book.last_update_id() + 1,
                ev.first_update_id
            );
        }

        // Update order book
        let book = self
            .books
            .entry(symbol.to_string())
            .or_insert_with(|| OrderBook::new(ev.price_exponent, ev.qty_exponent));
        book.apply_depth(ev);

        self.current_tick += 1;

        // Check pending orders for fills
        self.check_pending_fills(symbol, ev.ts.timestamp_nanos_opt().unwrap_or(0) as u64)
    }

    /// Update market state from float bid/ask (for backtest compatibility).
    pub fn update_market(&mut self, symbol: &str, bid: f64, ask: f64) {
        let book = self
            .books
            .entry(symbol.to_string())
            .or_insert_with(|| OrderBook::new(-2, -8)); // Default exponents
        book.update_quote(bid, ask);
        self.current_tick += 1;
    }

    /// Get the order book for a symbol.
    pub fn book(&self, symbol: &str) -> Option<&OrderBook> {
        self.books.get(symbol)
    }

    /// Get best bid for a symbol.
    pub fn best_bid(&self, symbol: &str) -> Option<f64> {
        self.books.get(symbol).and_then(|b| b.best_bid_f64())
    }

    /// Get best ask for a symbol.
    pub fn best_ask(&self, symbol: &str) -> Option<f64> {
        self.books.get(symbol).and_then(|b| b.best_ask_f64())
    }

    /// Generate a new order ID.
    pub fn next_order_id(&mut self) -> u64 {
        let id = self.next_order_id;
        self.next_order_id += 1;
        id
    }

    /// Submit an order.
    ///
    /// # Matching Rules
    /// - Market orders: immediate fill at best opposite side (taker)
    /// - Limit orders:
    ///   - If crosses spread → immediate fill (taker)
    ///   - Else → queued as pending (maker when filled later)
    ///
    /// # Returns
    /// A vector of fills (may be empty if order is queued as pending).
    pub fn submit(&mut self, ts_ns: u64, order: Order) -> Vec<Fill> {
        let book = match self.books.get(&order.symbol) {
            Some(b) => b,
            None => return Vec::new(),
        };

        // Get best opposite side price
        let (opposite_price, _opposite_qty) = match order.side {
            Side::Buy => match book.best_ask() {
                Some(p) => p,
                None => return self.queue_pending(order),
            },
            Side::Sell => match book.best_bid() {
                Some(p) => p,
                None => return self.queue_pending(order),
            },
        };

        let opposite_price_f64 = book.mantissa_to_f64(opposite_price, true);

        // Determine if order crosses the spread
        let crosses = match order.order_type {
            OrderType::Market => true, // Always crosses
            OrderType::Limit => {
                let limit = order.limit_price.unwrap_or(0.0);
                match order.side {
                    Side::Buy => limit >= opposite_price_f64, // Buy limit >= best ask
                    Side::Sell => limit <= opposite_price_f64, // Sell limit <= best bid
                }
            }
        };

        if crosses {
            // Immediate fill (taker)
            self.execute_fill(ts_ns, order, opposite_price_f64, FillType::Taker)
        } else {
            // Queue as pending (will be maker when filled)
            self.queue_pending(order)
        }
    }

    /// Queue a limit order as pending.
    fn queue_pending(&mut self, order: Order) -> Vec<Fill> {
        let book = match self.books.get(&order.symbol) {
            Some(b) => b,
            None => return Vec::new(),
        };

        let price_mantissa = book.f64_to_mantissa(order.limit_price.unwrap_or(0.0), true);
        let qty_mantissa = book.f64_to_mantissa(order.qty, false);

        let pending = PendingOrder {
            order,
            price_mantissa,
            qty_mantissa,
            submitted_tick: self.current_tick,
        };

        let symbol = pending.order.symbol.clone();
        self.pending_orders
            .entry(symbol)
            .or_default()
            .push_back(pending);

        Vec::new() // No immediate fill
    }

    /// Check pending orders for fills after a depth update.
    fn check_pending_fills(&mut self, symbol: &str, ts_ns: u64) -> Vec<Fill> {
        let book = match self.books.get(symbol) {
            Some(b) => b.clone(), // Clone to avoid borrow issues
            None => return Vec::new(),
        };

        // First pass: collect orders that can fill (to avoid borrow issues)
        let orders_to_fill: Vec<(usize, Order, f64)> = {
            let pending = match self.pending_orders.get(symbol) {
                Some(p) => p,
                None => return Vec::new(),
            };

            let mut to_fill = Vec::new();

            for (i, pending_order) in pending.iter().enumerate() {
                // Check latency constraint
                if self.cfg.latency_ticks > 0
                    && self.current_tick < pending_order.submitted_tick + self.cfg.latency_ticks
                {
                    continue;
                }

                let can_fill = match pending_order.order.side {
                    Side::Buy => {
                        // Buy fills if best ask <= limit price
                        if let Some((best_ask, _)) = book.best_ask() {
                            best_ask <= pending_order.price_mantissa
                        } else {
                            false
                        }
                    }
                    Side::Sell => {
                        // Sell fills if best bid >= limit price
                        if let Some((best_bid, _)) = book.best_bid() {
                            best_bid >= pending_order.price_mantissa
                        } else {
                            false
                        }
                    }
                };

                if can_fill {
                    // Get fill price (best opposite side)
                    let fill_price = match pending_order.order.side {
                        Side::Buy => book.best_ask().map(|(p, _)| book.mantissa_to_f64(p, true)),
                        Side::Sell => book.best_bid().map(|(p, _)| book.mantissa_to_f64(p, true)),
                    };

                    if let Some(price) = fill_price {
                        to_fill.push((i, pending_order.order.clone(), price));
                    }
                }
            }

            to_fill
        };

        // Second pass: execute fills (now we can borrow self mutably)
        let mut fills = Vec::new();
        let mut filled_indices = Vec::new();

        for (i, order, price) in orders_to_fill {
            let new_fills = self.execute_fill(ts_ns, order, price, FillType::Maker);
            fills.extend(new_fills);
            filled_indices.push(i);
        }

        // Remove filled orders (in reverse to maintain indices)
        if let Some(pending) = self.pending_orders.get_mut(symbol) {
            for i in filled_indices.into_iter().rev() {
                pending.remove(i);
            }
        }

        fills
    }

    /// Execute a fill and update state.
    fn execute_fill(
        &mut self,
        ts_ns: u64,
        order: Order,
        exec_price: f64,
        fill_type: FillType,
    ) -> Vec<Fill> {
        // Calculate notional and fee
        let notional = exec_price * order.qty;
        let fee_bps = match fill_type {
            FillType::Maker => self.cfg.fee_bps_maker,
            FillType::Taker => self.cfg.fee_bps_taker,
        };
        let fee = notional * (fee_bps / 10_000.0);

        // Check cash for buys
        if matches!(order.side, Side::Buy) && self.state.cash < notional + fee {
            tracing::warn!(
                "Insufficient cash: need {:.2}, have {:.2}",
                notional + fee,
                self.state.cash
            );
            return Vec::new();
        }

        // Update position and cash
        let position = self
            .state
            .positions
            .entry(order.symbol.clone())
            .or_default();
        let old_qty = position.qty;

        match order.side {
            Side::Buy => {
                self.state.cash -= notional + fee;

                if position.qty >= 0.0 {
                    // Adding to long or opening long
                    let total_cost = position.avg_price * position.qty + exec_price * order.qty;
                    position.qty += order.qty;
                    position.avg_price = if position.qty > 0.0 {
                        total_cost / position.qty
                    } else {
                        0.0
                    };
                } else {
                    // Covering short
                    let cover_qty = order.qty.min(-position.qty);
                    let pnl = (position.avg_price - exec_price) * cover_qty;
                    self.state.realized_pnl += pnl;
                    self.state.cash += pnl;

                    position.qty += order.qty;
                    if position.qty > 0.0 {
                        position.avg_price = exec_price;
                    }
                }
            }
            Side::Sell => {
                if position.qty <= 0.0 {
                    // Adding to short or opening short
                    let total_cost = position.avg_price * (-position.qty) + exec_price * order.qty;
                    position.qty -= order.qty;
                    position.avg_price = if position.qty < 0.0 {
                        total_cost / (-position.qty)
                    } else {
                        0.0
                    };
                } else {
                    // Closing long
                    let close_qty = order.qty.min(position.qty);
                    let pnl = (exec_price - position.avg_price) * close_qty;
                    self.state.realized_pnl += pnl;

                    position.qty -= order.qty;
                    if position.qty < 0.0 {
                        position.avg_price = exec_price;
                    }
                }

                self.state.cash += notional - fee;
            }
        }

        // Update inventory
        *self
            .state
            .inventory
            .entry(order.symbol.clone())
            .or_default() = position.qty;

        tracing::debug!(
            "FILL: {} {} {:.4} @ {:.2} ({:?}, fee={:.4}, pos: {:.4} -> {:.4})",
            order.side,
            order.symbol,
            order.qty,
            exec_price,
            fill_type,
            fee,
            old_qty,
            position.qty
        );

        let fill = Fill {
            order_id: order.id,
            symbol: order.symbol,
            side: order.side,
            qty: order.qty,
            price: exec_price,
            fee,
            fill_type,
            ts_ns,
            tag: order.tag,
        };

        self.fills.push(fill.clone());
        vec![fill]
    }

    /// Get pending orders for a symbol.
    pub fn pending_orders(&self, symbol: &str) -> Vec<&Order> {
        self.pending_orders
            .get(symbol)
            .map(|q| q.iter().map(|p| &p.order).collect())
            .unwrap_or_default()
    }

    /// Get all pending order count.
    pub fn pending_order_count(&self) -> usize {
        self.pending_orders.values().map(|q| q.len()).sum()
    }

    /// Get current simulator state.
    pub fn state(&self) -> &SimState {
        &self.state
    }

    /// Get cash balance.
    pub fn cash(&self) -> f64 {
        self.state.cash
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.state.realized_pnl
    }

    /// Get position for a symbol.
    pub fn position(&self, symbol: &str) -> f64 {
        self.state.position(symbol)
    }

    /// Get unrealized PnL across all positions.
    pub fn unrealized_pnl(&self) -> f64 {
        let mut pnl = 0.0;
        for (symbol, position) in &self.state.positions {
            if let Some(book) = self.books.get(symbol)
                && let (Some(bid), Some(ask)) = (book.best_bid_f64(), book.best_ask_f64())
            {
                let mid = (bid + ask) / 2.0;
                if position.qty > 0.0 {
                    pnl += (mid - position.avg_price) * position.qty;
                } else if position.qty < 0.0 {
                    pnl += (position.avg_price - mid) * (-position.qty);
                }
            }
        }
        pnl
    }

    /// Get all fills.
    pub fn fills(&self) -> &[Fill] {
        &self.fills
    }

    /// Get current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Get config.
    pub fn config(&self) -> &SimConfig {
        &self.cfg
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::depth::{DepthLevel, IntegrityTier};

    fn make_depth_event(symbol: &str, bid: i64, ask: i64) -> DepthEvent {
        DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: symbol.to_string(),
            first_update_id: 1,
            last_update_id: 1,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![DepthLevel {
                price: bid,
                qty: 100_000_000,
            }],
            asks: vec![DepthLevel {
                price: ask,
                qty: 100_000_000,
            }],
            is_snapshot: true,
            integrity_tier: IntegrityTier::Certified,
            source: None,
        }
    }

    #[test]
    fn test_market_buy_taker() {
        let cfg = SimConfig {
            initial_cash: 100_000.0,
            fee_bps_taker: 10.0,
            fee_bps_maker: 2.0,
            ..Default::default()
        };
        let mut sim = Simulator::new(cfg);

        // Price: 90000.00 (mantissa 9000000 with exp -2)
        let ev = make_depth_event("BTCUSDT", 8999900, 9000000);
        sim.on_depth("BTCUSDT", &ev);

        let order = Order::market(1, "BTCUSDT", Side::Buy, 0.1);
        let fills = sim.submit(0, order);

        assert_eq!(fills.len(), 1);
        assert!((fills[0].price - 90000.0).abs() < 0.01);
        assert_eq!(fills[0].fill_type, FillType::Taker);

        // Check position
        assert!((sim.position("BTCUSDT") - 0.1).abs() < 0.0001);
    }

    #[test]
    fn test_limit_order_queued_then_filled_as_maker() {
        let cfg = SimConfig {
            initial_cash: 100_000.0,
            fee_bps_taker: 10.0,
            fee_bps_maker: 2.0,
            latency_ticks: 0,
            ..Default::default()
        };
        let mut sim = Simulator::new(cfg);

        // Initial: bid=89990, ask=90010
        let ev = make_depth_event("BTCUSDT", 8999000, 9001000);
        sim.on_depth("BTCUSDT", &ev);

        // Submit limit buy at 90000 (doesn't cross ask of 90010)
        let order = Order::limit(1, "BTCUSDT", Side::Buy, 0.1, 90000.0);
        let fills = sim.submit(0, order);

        // Should be queued, not filled
        assert!(fills.is_empty());
        assert_eq!(sim.pending_order_count(), 1);

        // Price drops: ask moves to 89995 (below our limit of 90000)
        let ev2 = make_depth_event("BTCUSDT", 8998000, 8999500);
        let fills = sim.on_depth("BTCUSDT", &ev2);

        // Should fill as maker
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill_type, FillType::Maker);
        assert!((fills[0].price - 89995.0).abs() < 0.01);
        assert_eq!(sim.pending_order_count(), 0);
    }

    #[test]
    fn test_limit_order_crosses_immediately_taker() {
        let cfg = SimConfig::default();
        let mut sim = Simulator::new(cfg);

        // Ask at 90000
        let ev = make_depth_event("BTCUSDT", 8999000, 9000000);
        sim.on_depth("BTCUSDT", &ev);

        // Submit limit buy at 90010 (crosses ask of 90000)
        let order = Order::limit(1, "BTCUSDT", Side::Buy, 0.1, 90010.0);
        let fills = sim.submit(0, order);

        // Should fill immediately as taker
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].fill_type, FillType::Taker);
        assert!((fills[0].price - 90000.0).abs() < 0.01);
    }

    #[test]
    fn test_round_trip_pnl() {
        let cfg = SimConfig {
            initial_cash: 100_000.0,
            fee_bps_taker: 0.0,
            fee_bps_maker: 0.0,
            ..Default::default()
        };
        let mut sim = Simulator::new(cfg);

        // Buy at 90000
        let ev = make_depth_event("BTCUSDT", 8999900, 9000000);
        sim.on_depth("BTCUSDT", &ev);

        let order = Order::market(1, "BTCUSDT", Side::Buy, 1.0);
        sim.submit(0, order);

        // Sell at 91000 (profit)
        let ev = make_depth_event("BTCUSDT", 9100000, 9100100);
        sim.on_depth("BTCUSDT", &ev);

        let order = Order::market(2, "BTCUSDT", Side::Sell, 1.0);
        sim.submit(1, order);

        // Should have realized ~1000 profit
        assert!((sim.realized_pnl() - 1000.0).abs() < 1.0);
        assert!((sim.position("BTCUSDT")).abs() < 0.0001);
    }

    #[test]
    fn test_insufficient_cash() {
        let cfg = SimConfig {
            initial_cash: 1000.0,
            ..Default::default()
        };
        let mut sim = Simulator::new(cfg);

        let ev = make_depth_event("BTCUSDT", 8999900, 9000000);
        sim.on_depth("BTCUSDT", &ev);

        // Try to buy 1 BTC at 90000 (need ~90000, only have 1000)
        let order = Order::market(1, "BTCUSDT", Side::Buy, 1.0);
        let fills = sim.submit(0, order);

        assert!(fills.is_empty());
    }

    #[test]
    fn test_update_market_float() {
        let cfg = SimConfig::default();
        let mut sim = Simulator::new(cfg);

        sim.update_market("BTCUSDT", 89999.0, 90000.0);

        assert!((sim.best_bid("BTCUSDT").unwrap() - 89999.0).abs() < 0.01);
        assert!((sim.best_ask("BTCUSDT").unwrap() - 90000.0).abs() < 0.01);
    }
}
