//! Unified Execution Simulator
//!
//! **SINGLE SOURCE OF TRUTH** for simulated order execution in:
//! - Backtest (replay mode)
//! - Paper trading (live mode with WAL)

use std::collections::{HashMap, VecDeque};

use quantlaxmi_models::depth::DepthEvent;

use super::book::L2Book;
use super::ledger::Ledger;
use super::types::{Fill, FillType, Order, OrderType, Side, SimConfig};

/// Pending limit order waiting for price to cross.
#[derive(Debug, Clone)]
struct PendingOrder {
    order: Order,
    /// Price as mantissa (for deterministic matching)
    price_mantissa: i64,
    /// Tick when order was submitted (for latency)
    submitted_tick: u64,
}

/// Unified execution simulator.
///
/// **SINGLE SOURCE OF TRUTH** for order matching in both backtest and paper trading.
///
/// All order matching, position tracking, and PnL calculation happens here.
pub struct Simulator {
    cfg: SimConfig,
    ledger: Ledger,
    /// Order books per symbol
    books: HashMap<String, L2Book>,
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
        let ledger = Ledger::new(cfg.initial_cash);
        Self {
            cfg,
            ledger,
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
            .or_insert_with(|| L2Book::new(ev.price_exponent, ev.qty_exponent));
        book.apply_depth(ev);

        self.current_tick += 1;

        // Check pending orders for fills
        let ts_ns = ev.ts.timestamp_nanos_opt().unwrap_or(0) as u64;
        self.check_pending_fills(symbol, ts_ns)
    }

    /// Update market state from float bid/ask (for backtest compatibility).
    pub fn update_market(&mut self, symbol: &str, bid: f64, ask: f64) {
        let book = self
            .books
            .entry(symbol.to_string())
            .or_insert_with(|| L2Book::new(-2, -8)); // Default exponents
        book.update_quote(bid, ask);
        self.current_tick += 1;
    }

    /// Get the order book for a symbol.
    pub fn book(&self, symbol: &str) -> Option<&L2Book> {
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
    ///   - If crosses spread -> immediate fill (taker)
    ///   - Else -> queued as pending (maker when filled later)
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

        let pending = PendingOrder {
            order,
            price_mantissa,
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
        // Calculate fee
        let notional = exec_price * order.qty;
        let fee_bps = match fill_type {
            FillType::Maker => self.cfg.fee_bps_maker,
            FillType::Taker => self.cfg.fee_bps_taker,
        };
        let fee = notional * (fee_bps / 10_000.0);

        // Update ledger (handles cash, positions, PnL)
        if !self
            .ledger
            .on_fill(&order.symbol, order.side, order.qty, exec_price, fee)
        {
            return Vec::new(); // Insufficient funds
        }

        tracing::debug!(
            "FILL: {} {} {:.4} @ {:.2} ({:?}, fee={:.4})",
            order.side,
            order.symbol,
            order.qty,
            exec_price,
            fill_type,
            fee
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

    /// Get cash balance.
    pub fn cash(&self) -> f64 {
        self.ledger.cash
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.ledger.realized_pnl
    }

    /// Get position for a symbol.
    pub fn position(&self, symbol: &str) -> f64 {
        self.ledger.position(symbol)
    }

    /// Get unrealized PnL across all positions.
    pub fn unrealized_pnl(&self) -> f64 {
        let mut pnl = 0.0;
        for (symbol, position) in &self.ledger.positions {
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
}
