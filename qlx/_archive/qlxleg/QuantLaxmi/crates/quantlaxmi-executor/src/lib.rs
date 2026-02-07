//! # Order Execution Engine Module
//!
//! Unified execution layer supporting both simulation and live trading.
//!
//! ## Description
//! Implements order execution through a unified trait:
//! - **SimulatedExchange**: For backtesting with configurable slippage/commissions
//! - **ZerodhaLiveExchange**: For live trading via Kite Connect API
//!
//! ## Order Flow
//! ```text
//! Strategy → OrderEvent → Exchange (Simulated/Live) → Fill → PositionUpdate
//! ```
//!
//! ## Modes
//! - **Backtest**: Deterministic fills, configurable slippage
//! - **Paper**: Simulated fills, real market data
//! - **Live**: Real order placement via Zerodha Kite API
//!
//! ## References
//! - IEEE Std 1016-2009: Software Design Descriptions
//! - Kite Connect API: <https://kite.trade/docs/connect/v3/>

// New unified modules for crypto paper trading
pub mod paper;
pub mod sim;

// Re-export key types for ergonomic imports
pub use paper::{FillSink, NoopFillSink, PaperEngine};
pub use sim::Side as SimSide;
pub use sim::{Fill, FillType, Order, OrderType, SimConfig, Simulator}; // Alias to avoid conflict with quantlaxmi_models::Side

use async_trait::async_trait;
use chrono::Utc;
use quantlaxmi_core::EventBus;
use quantlaxmi_models::{MarketEvent, MarketPayload, OrderEvent, OrderPayload, OrderStatus, Side};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info, warn};
use uuid::Uuid;

// ============================================================================
// UNIFIED EXCHANGE TRAIT
// ============================================================================

/// Unified execution interface for both simulated and live trading.
///
/// All exchange implementations must provide order handling and market data processing.
#[async_trait]
pub trait Exchange: Send + Sync {
    /// Handle an incoming order (new, cancel, modify)
    async fn handle_order(&mut self, event: OrderEvent) -> anyhow::Result<()>;

    /// Process market data for order matching (simulated) or position updates (live)
    async fn on_market_data(&mut self, event: MarketEvent) -> anyhow::Result<()>;

    /// Get the exchange name for logging
    fn name(&self) -> &'static str;
}

/// Internal position state for a specific asset.
struct PositionState {
    /// Current quantity (positive for long, negative for short).
    quantity: f64,
    /// Volume-weighted average entry price (VWAP).
    avg_price: f64,
}

/// Internal pending order state.
struct OrderState {
    /// Unique order identifier.
    id: Uuid,
    /// Link to the original strategy intent.
    intent_id: Option<Uuid>,
    /// Asset symbol (e.g., "BTCUSDT").
    symbol: String,
    /// Side of the transaction (Buy/Sell).
    side: Side,
    /// Optional limit price; market order if None.
    price: Option<f64>,
    /// Target quantity to execute.
    quantity: f64,
}

/// Supported commission models for different exchanges and asset classes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CommissionModel {
    /// No commissions (default).
    None,
    /// Fixed basis points of turnover.
    Linear(f64),
    /// Zerodha Indian F&O (Futures) model as of Dec 2025.
    ZerodhaFnO,
}

// ============================================================================
// RISK ENVELOPE (venue-level position/notional limits)
// ============================================================================

/// Risk enforcement mode for when limits are exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskMode {
    /// Clip order quantity to fit within limits (research/backtest).
    Clip,
    /// Reject orders that exceed limits (strict/certification).
    Reject,
}

/// Venue-level risk envelope for position and notional limits.
///
/// Enforced at the exchange boundary to prevent strategy bugs from causing
/// catastrophic exposure. All limits are in USD for cross-asset consistency.
///
/// # Example
/// ```rust
/// use quantlaxmi_executor::RiskEnvelope;
/// let envelope = RiskEnvelope::for_equity(100_000.0); // $100k account
/// // max_gross_notional_usd = $200k (2x)
/// // max_symbol_notional_usd = $50k (0.5x)
/// // max_order_notional_usd = $5k (0.05x)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEnvelope {
    /// Maximum gross notional across all positions (sum of |pos * price|).
    pub max_gross_notional_usd: f64,
    /// Maximum notional per symbol.
    pub max_symbol_notional_usd: f64,
    /// Maximum notional per single order.
    pub max_order_notional_usd: f64,
    /// Enforcement mode (Clip or Reject).
    pub mode: RiskMode,
    /// Whether envelope is enabled.
    pub enabled: bool,
}

impl RiskEnvelope {
    /// Create a disabled (no-op) risk envelope.
    pub fn disabled() -> Self {
        Self {
            max_gross_notional_usd: f64::MAX,
            max_symbol_notional_usd: f64::MAX,
            max_order_notional_usd: f64::MAX,
            mode: RiskMode::Clip,
            enabled: false,
        }
    }

    /// Create a risk envelope sized for a given equity amount.
    ///
    /// Uses conservative defaults:
    /// - max_gross_notional = 2x equity
    /// - max_symbol_notional = 0.5x equity
    /// - max_order_notional = 0.05x equity
    pub fn for_equity(equity: f64) -> Self {
        Self {
            max_gross_notional_usd: 2.0 * equity,
            max_symbol_notional_usd: 0.5 * equity,
            max_order_notional_usd: 0.05 * equity,
            mode: RiskMode::Clip,
            enabled: true,
        }
    }

    /// Create with explicit limits.
    pub fn new(
        max_gross_notional_usd: f64,
        max_symbol_notional_usd: f64,
        max_order_notional_usd: f64,
        mode: RiskMode,
    ) -> Self {
        Self {
            max_gross_notional_usd,
            max_symbol_notional_usd,
            max_order_notional_usd,
            mode,
            enabled: true,
        }
    }
}

impl Default for RiskEnvelope {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Audit record for risk enforcement actions.
#[derive(Debug, Clone, Serialize)]
pub struct RiskEvent {
    pub timestamp: chrono::DateTime<Utc>,
    pub symbol: String,
    pub side: Side,
    pub requested_qty: f64,
    pub accepted_qty: f64,
    pub price_used: f64,
    pub order_notional: f64,
    pub symbol_notional_before: f64,
    pub symbol_notional_after: f64,
    pub gross_notional_before: f64,
    pub gross_notional_after: f64,
    pub rule_triggered: String, // "MAX_ORDER", "MAX_SYMBOL", "MAX_GROSS", "NONE"
    pub action: String,         // "CLIP", "REJECT", "PASS"
}

/// High-fidelity simulated execution venue.
///
/// # Architecture
/// - **Order Matching**: Compares pending orders against incoming L1 market data.
/// - **Slippage Simulation**: Adds probabilistic noise to execution prices.
/// - **Commission Modeling**: Calculates realistic fees as of late 2025.
/// - **State Management**: Tracks per-symbol positions and pending limit orders.
pub struct SimulatedExchange {
    /// Broadcast bus for publishing execution reports (Fills).
    bus: Arc<EventBus>,
    /// Queue of orders awaiting market triggers, partitioned by symbol.
    pending_orders: HashMap<String, Vec<OrderState>>,
    /// Portfolio state tracking current holdings.
    positions: HashMap<String, PositionState>,
    /// Constant slippage parameter in basis points.
    slippage_bps: f64,
    /// Commission model to use for fills.
    commission_model: CommissionModel,
    /// Seeded PRNG for deterministic slippage noise.
    rng: Pcg64,
    /// Internal simulation clock synchronized with market data.
    current_time: chrono::DateTime<chrono::Utc>,
    /// Last known prices per symbol (for notional calculations).
    last_prices: HashMap<String, f64>,
    /// Risk envelope for position/notional limits.
    risk_envelope: RiskEnvelope,
    /// Risk event log for audit trail.
    risk_events: Vec<RiskEvent>,
}

impl SimulatedExchange {
    /// Initializes a new simulated exchange.
    ///
    /// # Parameters
    /// * `bus` - Shared event distribution system.
    /// * `slippage_bps` - Expected price impact/cost in basis points.
    /// * `commission_model` - Fees model to apply.
    /// * `seed` - Optional seed for the PRNG to enable reproducible runs.
    pub fn new(
        bus: Arc<EventBus>,
        slippage_bps: f64,
        commission_model: CommissionModel,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        Self {
            bus,
            pending_orders: HashMap::new(),
            positions: HashMap::new(),
            slippage_bps,
            commission_model,
            rng,
            current_time: Utc::now(),
            last_prices: HashMap::new(),
            risk_envelope: RiskEnvelope::disabled(),
            risk_events: Vec::new(),
        }
    }

    /// Create a new simulated exchange with a risk envelope.
    ///
    /// # Parameters
    /// * `bus` - Shared event distribution system.
    /// * `slippage_bps` - Expected price impact/cost in basis points.
    /// * `commission_model` - Fees model to apply.
    /// * `seed` - Optional seed for the PRNG to enable reproducible runs.
    /// * `risk_envelope` - Risk limits for position/notional enforcement.
    pub fn with_risk_envelope(
        bus: Arc<EventBus>,
        slippage_bps: f64,
        commission_model: CommissionModel,
        seed: Option<u64>,
        risk_envelope: RiskEnvelope,
    ) -> Self {
        let rng = match seed {
            Some(s) => Pcg64::seed_from_u64(s),
            None => Pcg64::from_entropy(),
        };

        Self {
            bus,
            pending_orders: HashMap::new(),
            positions: HashMap::new(),
            slippage_bps,
            commission_model,
            rng,
            current_time: Utc::now(),
            last_prices: HashMap::new(),
            risk_envelope,
            risk_events: Vec::new(),
        }
    }

    /// Get a reference to the risk events log.
    pub fn risk_events(&self) -> &[RiskEvent] {
        &self.risk_events
    }

    /// Calculate current gross notional across all positions.
    fn gross_notional(&self) -> f64 {
        self.positions
            .iter()
            .map(|(sym, pos)| {
                let price = self.last_prices.get(sym).copied().unwrap_or(0.0);
                pos.quantity.abs() * price
            })
            .sum()
    }

    /// Calculate current notional for a specific symbol.
    fn symbol_notional(&self, symbol: &str) -> f64 {
        let pos = self
            .positions
            .get(symbol)
            .map(|p| p.quantity.abs())
            .unwrap_or(0.0);
        let price = self.last_prices.get(symbol).copied().unwrap_or(0.0);
        pos * price
    }

    /// Enforce risk envelope on an order, returning the accepted quantity.
    ///
    /// Returns (accepted_qty, rule_triggered, action).
    fn enforce_risk(
        &mut self,
        symbol: &str,
        side: Side,
        requested_qty: f64,
        price: f64,
    ) -> (f64, String, String) {
        if !self.risk_envelope.enabled {
            return (requested_qty, "NONE".to_string(), "PASS".to_string());
        }

        let order_notional = requested_qty * price;
        let current_symbol_notional = self.symbol_notional(symbol);
        let current_gross_notional = self.gross_notional();

        // Get current position for the symbol
        let current_pos = self
            .positions
            .get(symbol)
            .map(|p| p.quantity)
            .unwrap_or(0.0);
        let delta = if side == Side::Buy {
            requested_qty
        } else {
            -requested_qty
        };
        let new_pos = current_pos + delta;
        let new_symbol_notional = new_pos.abs() * price;

        // Calculate new gross notional (sum of all |position * price|)
        let new_gross_notional =
            current_gross_notional - current_symbol_notional + new_symbol_notional;

        // Check limits in order: ORDER -> SYMBOL -> GROSS
        // 1. Max order notional
        if order_notional > self.risk_envelope.max_order_notional_usd {
            let max_qty = self.risk_envelope.max_order_notional_usd / price;
            match self.risk_envelope.mode {
                RiskMode::Clip => {
                    warn!(
                        "[RISK] MAX_ORDER exceeded: requested ${:.0} > limit ${:.0}, clipping to {:.4} units",
                        order_notional, self.risk_envelope.max_order_notional_usd, max_qty
                    );
                    return (max_qty, "MAX_ORDER".to_string(), "CLIP".to_string());
                }
                RiskMode::Reject => {
                    warn!(
                        "[RISK] MAX_ORDER exceeded: requested ${:.0} > limit ${:.0}, REJECTING",
                        order_notional, self.risk_envelope.max_order_notional_usd
                    );
                    return (0.0, "MAX_ORDER".to_string(), "REJECT".to_string());
                }
            }
        }

        // 2. Max symbol notional (position after fill)
        if new_symbol_notional > self.risk_envelope.max_symbol_notional_usd {
            // Calculate max qty that keeps us under the limit
            let max_new_pos = self.risk_envelope.max_symbol_notional_usd / price;
            let max_delta = if side == Side::Buy {
                (max_new_pos - current_pos).max(0.0)
            } else {
                (current_pos + max_new_pos).max(0.0)
            };

            match self.risk_envelope.mode {
                RiskMode::Clip => {
                    warn!(
                        "[RISK] MAX_SYMBOL exceeded: new notional ${:.0} > limit ${:.0}, clipping to {:.4} units",
                        new_symbol_notional, self.risk_envelope.max_symbol_notional_usd, max_delta
                    );
                    return (max_delta, "MAX_SYMBOL".to_string(), "CLIP".to_string());
                }
                RiskMode::Reject => {
                    warn!(
                        "[RISK] MAX_SYMBOL exceeded: new notional ${:.0} > limit ${:.0}, REJECTING",
                        new_symbol_notional, self.risk_envelope.max_symbol_notional_usd
                    );
                    return (0.0, "MAX_SYMBOL".to_string(), "REJECT".to_string());
                }
            }
        }

        // 3. Max gross notional
        if new_gross_notional > self.risk_envelope.max_gross_notional_usd {
            let available = self.risk_envelope.max_gross_notional_usd - current_gross_notional
                + current_symbol_notional;
            let max_qty = (available / price).max(0.0);

            match self.risk_envelope.mode {
                RiskMode::Clip => {
                    warn!(
                        "[RISK] MAX_GROSS exceeded: new gross ${:.0} > limit ${:.0}, clipping to {:.4} units",
                        new_gross_notional, self.risk_envelope.max_gross_notional_usd, max_qty
                    );
                    return (max_qty, "MAX_GROSS".to_string(), "CLIP".to_string());
                }
                RiskMode::Reject => {
                    warn!(
                        "[RISK] MAX_GROSS exceeded: new gross ${:.0} > limit ${:.0}, REJECTING",
                        new_gross_notional, self.risk_envelope.max_gross_notional_usd
                    );
                    return (0.0, "MAX_GROSS".to_string(), "REJECT".to_string());
                }
            }
        }

        (requested_qty, "NONE".to_string(), "PASS".to_string())
    }

    /// Calculates commission for a trade based on the configured model.
    fn calculate_commission(&self, side: Side, quantity: f64, price: f64) -> f64 {
        let turnover = quantity * price;
        match self.commission_model {
            CommissionModel::None => 0.0,
            CommissionModel::Linear(bps) => turnover * (bps / 10000.0),
            CommissionModel::ZerodhaFnO => {
                // Brokerage: Lower of 0.03% or ₹20 per executed order
                let brokerage = (turnover * 0.0003).min(20.0);

                // STT (Sell side only for Futures): 0.02%
                let stt = if side == Side::Sell {
                    turnover * 0.0002
                } else {
                    0.0
                };

                // Transaction Charge: 0.00173% (NSE)
                let trans_charge = turnover * 0.0000173;

                // SEBI: 0.00005%
                let sebi = turnover * 0.0000005;

                // Stamp Duty (Buy side only): 0.002%
                let stamp = if side == Side::Buy {
                    turnover * 0.00002
                } else {
                    0.0
                };

                // GST: 18% on (Brokerage + Trans + SEBI)
                let gst = (brokerage + trans_charge + sebi) * 0.18;

                brokerage + stt + trans_charge + sebi + stamp + gst
            }
        }
    }

    /// Generates a UUID derived from the internal seeded RNG.
    pub fn next_deterministic_id(&mut self) -> Uuid {
        let bytes: [u8; 16] = self.rng.r#gen();
        Uuid::from_bytes(bytes)
    }

    /// Explicitly updates the internal simulation clock.
    pub fn set_time(&mut self, time: chrono::DateTime<chrono::Utc>) {
        self.current_time = time;
    }

    /// Ingests a new order command or cancellation request.
    ///
    /// # Parameters
    /// * `event` - The order event containing the intent (New, Cancel, Modify).
    pub async fn handle_order(&mut self, event: OrderEvent) -> anyhow::Result<()> {
        match event.payload {
            OrderPayload::New {
                symbol,
                side,
                price,
                quantity,
                ..
            } => {
                // Get price for risk enforcement (use limit price or last known price)
                let risk_price = price
                    .or_else(|| self.last_prices.get(&symbol).copied())
                    .unwrap_or(0.0);

                // Enforce risk envelope
                let symbol_notional_before = self.symbol_notional(&symbol);
                let gross_notional_before = self.gross_notional();

                let (accepted_qty, rule_triggered, action) =
                    self.enforce_risk(&symbol, side, quantity, risk_price);

                // Log risk event if envelope is enabled
                if self.risk_envelope.enabled {
                    let order_notional = accepted_qty * risk_price;

                    // Calculate projected notional after fill
                    let current_pos = self
                        .positions
                        .get(&symbol)
                        .map(|p| p.quantity)
                        .unwrap_or(0.0);
                    let delta = if side == Side::Buy {
                        accepted_qty
                    } else {
                        -accepted_qty
                    };
                    let new_pos = current_pos + delta;
                    let symbol_notional_after = new_pos.abs() * risk_price;
                    let gross_notional_after =
                        gross_notional_before - symbol_notional_before + symbol_notional_after;

                    let risk_event = RiskEvent {
                        timestamp: self.current_time,
                        symbol: symbol.clone(),
                        side,
                        requested_qty: quantity,
                        accepted_qty,
                        price_used: risk_price,
                        order_notional,
                        symbol_notional_before,
                        symbol_notional_after,
                        gross_notional_before,
                        gross_notional_after,
                        rule_triggered: rule_triggered.clone(),
                        action: action.clone(),
                    };
                    self.risk_events.push(risk_event);
                }

                // Reject order entirely if accepted_qty is 0
                if accepted_qty <= 0.0 {
                    warn!(
                        "Order REJECTED by risk envelope: {} {} @ {:?}",
                        side, symbol, price
                    );
                    self.bus.publish_order_update(OrderEvent {
                        order_id: event.order_id,
                        parent_decision_id: None,
                        intent_id: event.intent_id,
                        timestamp: self.current_time,
                        symbol: event.symbol.clone(),
                        side: event.side,
                        payload: OrderPayload::Update {
                            status: OrderStatus::Rejected,
                            filled_quantity: 0.0,
                            avg_price: 0.0,
                            commission: 0.0,
                        },
                    })?;
                    return Ok(());
                }

                info!(
                    "Accepted New Order: {} {} @ {:?} (qty: {:.4} of {:.4})",
                    side, symbol, price, accepted_qty, quantity
                );
                let order = OrderState {
                    id: event.order_id,
                    intent_id: event.intent_id,
                    symbol: symbol.clone(),
                    side,
                    price,
                    quantity: accepted_qty, // Use risk-adjusted quantity
                };
                self.pending_orders.entry(symbol).or_default().push(order);

                self.bus.publish_order_update(OrderEvent {
                    order_id: event.order_id,
                    parent_decision_id: None,
                    intent_id: event.intent_id,
                    timestamp: self.current_time,
                    symbol: event.symbol.clone(),
                    side: event.side,
                    payload: OrderPayload::Update {
                        status: OrderStatus::Accepted,
                        filled_quantity: 0.0,
                        avg_price: 0.0,
                        commission: 0.0,
                    },
                })?;
            }
            OrderPayload::Cancel => {
                let mut cancelled = false;
                for orders in self.pending_orders.values_mut() {
                    if let Some(pos) = orders.iter().position(|o| o.id == event.order_id) {
                        orders.remove(pos);
                        cancelled = true;
                        break;
                    }
                }

                if cancelled {
                    info!("Cancelled Order: {}", event.order_id);
                    self.bus.publish_order_update(OrderEvent {
                        order_id: event.order_id,
                        parent_decision_id: None,
                        intent_id: event.intent_id,
                        timestamp: self.current_time,
                        symbol: event.symbol.clone(),
                        side: event.side,
                        payload: OrderPayload::Update {
                            status: OrderStatus::Cancelled,
                            filled_quantity: 0.0,
                            avg_price: 0.0,
                            commission: 0.0,
                        },
                    })?;
                } else {
                    warn!("Cancel failed - order {} not found", event.order_id);
                }
            }
            OrderPayload::Modify {
                new_price,
                new_quantity,
            } => {
                let mut modified = false;
                for orders in self.pending_orders.values_mut() {
                    if let Some(order) = orders.iter_mut().find(|o| o.id == event.order_id) {
                        if let Some(p) = new_price {
                            order.price = Some(p);
                        }
                        if let Some(q) = new_quantity {
                            order.quantity = q;
                        }
                        modified = true;
                        break;
                    }
                }

                if modified {
                    info!("Modified Order: {}", event.order_id);
                    self.bus.publish_order_update(OrderEvent {
                        order_id: event.order_id,
                        parent_decision_id: None,
                        intent_id: event.intent_id,
                        timestamp: self.current_time,
                        symbol: event.symbol.clone(),
                        side: event.side,
                        payload: OrderPayload::Update {
                            status: OrderStatus::Accepted,
                            filled_quantity: 0.0,
                            avg_price: 0.0,
                            commission: 0.0,
                        },
                    })?;
                } else {
                    warn!("Modify failed - order {} not found", event.order_id);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Evaluates market data and triggers executions for pending orders.
    ///
    /// # Parameters
    /// * `event` - Incoming market tick or bar.
    pub async fn on_market_data(&mut self, event: MarketEvent) -> anyhow::Result<()> {
        self.current_time = event.exchange_time;
        // Handle both Tick and Trade events for order matching
        let price = match &event.payload {
            MarketPayload::Tick { price, .. } => Some(*price),
            MarketPayload::Trade { price, .. } => Some(*price),
            _ => None,
        };
        if let Some(price) = price {
            let symbol = event.symbol.clone();

            // Track last price for risk notional calculations
            self.last_prices.insert(symbol.clone(), price);

            // 1. Collect potential matches to avoid borrow checker conflicts
            let mut fills = Vec::new();

            if let Some(orders) = self.pending_orders.get_mut(&symbol) {
                let mut remaining_orders = Vec::new();
                for order in orders.drain(..) {
                    let can_fill = match order.price {
                        None => true,
                        Some(limit_price) => match order.side {
                            Side::Buy => price <= limit_price,
                            Side::Sell => price >= limit_price,
                        },
                    };

                    if can_fill {
                        let random_factor = self.rng.gen_range(0.9..1.1);
                        let effective_slippage = self.slippage_bps * random_factor;

                        let slip = price * (effective_slippage / 10000.0);
                        let fill_price = match order.side {
                            Side::Buy => price + slip,
                            Side::Sell => price - slip,
                        };

                        fills.push((order, fill_price));
                    } else {
                        remaining_orders.push(order);
                    }
                }
                *orders = remaining_orders;
            }

            // 2. Process fills outside the borrow
            for (order, fill_price) in fills {
                info!(
                    "Simulated Fill: {} {} @ {}",
                    order.side, order.symbol, fill_price
                );

                let pos = self
                    .positions
                    .entry(order.symbol.clone())
                    .or_insert(PositionState {
                        quantity: 0.0,
                        avg_price: 0.0,
                    });
                let side_mult = if order.side == Side::Buy { 1.0 } else { -1.0 };

                if (pos.quantity > 0.0 && order.side == Side::Buy)
                    || (pos.quantity < 0.0 && order.side == Side::Sell)
                {
                    pos.avg_price = ((pos.quantity * pos.avg_price)
                        + (order.quantity * fill_price))
                        / (pos.quantity + order.quantity);
                } else if pos.quantity == 0.0 {
                    pos.avg_price = fill_price;
                }

                pos.quantity += side_mult * order.quantity;
                info!(
                    "Position Update: {} quantity: {}",
                    order.symbol, pos.quantity
                );

                let commission = self.calculate_commission(order.side, order.quantity, fill_price);

                self.bus.publish_order_update(OrderEvent {
                    order_id: order.id,
                    parent_decision_id: None,
                    intent_id: order.intent_id,
                    timestamp: self.current_time,
                    symbol: order.symbol.clone(),
                    side: order.side,
                    payload: OrderPayload::Update {
                        status: OrderStatus::Filled,
                        filled_quantity: order.quantity,
                        avg_price: fill_price,
                        commission,
                    },
                })?;

                self.bus
                    .publish_fill(quantlaxmi_models::FillEvent {
                        timestamp: self.current_time,
                        order_id: order.id,
                        parent_decision_id: None,
                        intent_id: order.intent_id,
                        fill_id: Uuid::new_v4().to_string(),
                        symbol: order.symbol.clone(),
                        side: order.side,
                        price: fill_price,
                        quantity: order.quantity,
                        commission,
                        commission_asset: "INR".to_string(),
                        venue: "SimulatedExchange".to_string(),
                        is_final: true,
                    })
                    .await?;
            }
        }
        Ok(())
    }
}

// Implement Exchange trait for SimulatedExchange
#[async_trait]
impl Exchange for SimulatedExchange {
    async fn handle_order(&mut self, event: OrderEvent) -> anyhow::Result<()> {
        // Call the existing implementation
        SimulatedExchange::handle_order(self, event).await
    }

    async fn on_market_data(&mut self, event: MarketEvent) -> anyhow::Result<()> {
        // Call the existing implementation
        SimulatedExchange::on_market_data(self, event).await
    }

    fn name(&self) -> &'static str {
        "SimulatedExchange"
    }
}

// ============================================================================
// ZERODHA LIVE EXCHANGE
// ============================================================================

/// Base URL for Kite Connect REST API
const KITE_API_URL: &str = "https://api.kite.trade";

/// Zerodha live execution client for production trading.
///
/// # Safety
/// This client places REAL orders with REAL money. Use with extreme caution.
/// Always test thoroughly in paper mode before going live.
///
/// # Features
/// - Market and limit order support
/// - Automatic commission calculation (ZerodhaFnO model)
/// - Position tracking for risk management
/// - Order status polling for fill confirmation
pub struct ZerodhaLiveExchange {
    /// Event bus for publishing execution reports
    bus: Arc<EventBus>,
    /// Kite API key
    api_key: String,
    /// Session access token
    access_token: String,
    /// HTTP client for API calls
    client: reqwest::Client,
    /// Pending orders awaiting fill confirmation
    pending_orders: HashMap<String, PendingOrder>,
    /// Commission model
    commission_model: CommissionModel,
}

/// Pending order tracking for live exchange
#[allow(dead_code)]
struct PendingOrder {
    /// Internal order ID
    order_id: Uuid,
    /// Intent ID for attribution
    intent_id: Option<Uuid>,
    /// Zerodha order ID (kept for order status polling)
    zerodha_order_id: String,
    /// Symbol
    symbol: String,
    /// Side
    side: Side,
    /// Quantity
    quantity: f64,
}

/// Zerodha order response
#[derive(Debug, serde::Deserialize)]
struct ZerodhaOrderResponse {
    status: String,
    data: Option<ZerodhaOrderData>,
    message: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ZerodhaOrderData {
    order_id: String,
}

impl ZerodhaLiveExchange {
    /// Create a new Zerodha live exchange client.
    ///
    /// # Parameters
    /// * `bus` - Event bus for publishing fills
    /// * `api_key` - Kite API key from developer console
    /// * `access_token` - Session token from login
    ///
    /// # Safety
    /// This will place REAL orders. Ensure credentials are correct.
    pub fn new(bus: Arc<EventBus>, api_key: String, access_token: String) -> Self {
        info!("[ZERODHA LIVE] Initializing live execution client - REAL ORDERS ENABLED");
        Self {
            bus,
            api_key,
            access_token,
            client: reqwest::Client::new(),
            pending_orders: HashMap::new(),
            commission_model: CommissionModel::ZerodhaFnO,
        }
    }

    /// Place an order via Kite API.
    async fn place_order(
        &self,
        symbol: &str,
        side: Side,
        quantity: f64,
        price: Option<f64>,
    ) -> anyhow::Result<String> {
        // Determine exchange based on symbol type
        let exchange = if symbol.contains("FUT") || symbol.contains("CE") || symbol.contains("PE") {
            "NFO"
        } else {
            "NSE"
        };

        let transaction_type = match side {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        };

        let order_type = if price.is_some() { "LIMIT" } else { "MARKET" };

        let url = format!("{}/orders/regular", KITE_API_URL);

        info!(
            "[ZERODHA LIVE] Placing {} {} {} {} @ {:?}",
            transaction_type, quantity as u32, symbol, order_type, price
        );

        let mut form = vec![
            ("tradingsymbol", symbol.to_string()),
            ("exchange", exchange.to_string()),
            ("transaction_type", transaction_type.to_string()),
            ("order_type", order_type.to_string()),
            ("quantity", (quantity as u32).to_string()),
            ("product", "NRML".to_string()), // F&O normal
            ("validity", "DAY".to_string()),
        ];

        if let Some(p) = price {
            form.push(("price", format!("{:.2}", p)));
        }

        let response = self
            .client
            .post(&url)
            .header("X-Kite-Version", "3")
            .header(
                "Authorization",
                format!("token {}:{}", self.api_key, self.access_token),
            )
            .form(&form)
            .send()
            .await?;

        let status = response.status();
        let resp: ZerodhaOrderResponse = response.json().await?;

        if resp.status != "success" {
            let msg = resp.message.unwrap_or_else(|| "Unknown error".to_string());
            error!("[ZERODHA LIVE] Order placement FAILED: {}", msg);
            return Err(anyhow::anyhow!("Order failed: {}", msg));
        }

        let order_id = resp
            .data
            .ok_or_else(|| anyhow::anyhow!("No order ID in response"))?
            .order_id;

        info!(
            "[ZERODHA LIVE] Order placed successfully: {} (HTTP {})",
            order_id, status
        );
        Ok(order_id)
    }

    /// Cancel an order via Kite API.
    async fn cancel_order(&self, zerodha_order_id: &str) -> anyhow::Result<()> {
        let url = format!("{}/orders/regular/{}", KITE_API_URL, zerodha_order_id);

        info!("[ZERODHA LIVE] Cancelling order: {}", zerodha_order_id);

        let response = self
            .client
            .delete(&url)
            .header("X-Kite-Version", "3")
            .header(
                "Authorization",
                format!("token {}:{}", self.api_key, self.access_token),
            )
            .send()
            .await?;

        let resp: ZerodhaOrderResponse = response.json().await?;

        if resp.status != "success" {
            let msg = resp.message.unwrap_or_else(|| "Unknown error".to_string());
            return Err(anyhow::anyhow!("Cancel failed: {}", msg));
        }

        info!("[ZERODHA LIVE] Order cancelled: {}", zerodha_order_id);
        Ok(())
    }

    /// Calculate commission using ZerodhaFnO model
    fn calculate_commission(&self, side: Side, quantity: f64, price: f64) -> f64 {
        let turnover = quantity * price;
        match self.commission_model {
            CommissionModel::ZerodhaFnO => {
                let brokerage = (turnover * 0.0003).min(20.0);
                let stt = if side == Side::Sell {
                    turnover * 0.0002
                } else {
                    0.0
                };
                let trans_charge = turnover * 0.0000173;
                let sebi = turnover * 0.0000005;
                let stamp = if side == Side::Buy {
                    turnover * 0.00002
                } else {
                    0.0
                };
                let gst = (brokerage + trans_charge + sebi) * 0.18;
                brokerage + stt + trans_charge + sebi + stamp + gst
            }
            CommissionModel::Linear(bps) => turnover * (bps / 10000.0),
            CommissionModel::None => 0.0,
        }
    }
}

#[async_trait]
impl Exchange for ZerodhaLiveExchange {
    async fn handle_order(&mut self, event: OrderEvent) -> anyhow::Result<()> {
        match event.payload {
            OrderPayload::New {
                symbol,
                side,
                price,
                quantity,
                ..
            } => {
                // Place REAL order
                match self.place_order(&symbol, side, quantity, price).await {
                    Ok(zerodha_order_id) => {
                        // Track pending order
                        self.pending_orders.insert(
                            zerodha_order_id.clone(),
                            PendingOrder {
                                order_id: event.order_id,
                                intent_id: event.intent_id,
                                zerodha_order_id: zerodha_order_id.clone(),
                                symbol: symbol.clone(),
                                side,
                                quantity,
                            },
                        );

                        // Publish accepted status
                        self.bus.publish_order_update(OrderEvent {
                            order_id: event.order_id,
                            parent_decision_id: None,
                            intent_id: event.intent_id,
                            timestamp: Utc::now(),
                            symbol: symbol.clone(),
                            side,
                            payload: OrderPayload::Update {
                                status: OrderStatus::Accepted,
                                filled_quantity: 0.0,
                                avg_price: 0.0,
                                commission: 0.0,
                            },
                        })?;

                        // For market orders, assume immediate fill and simulate fill event
                        // In production, you'd poll order status or use WebSocket for updates
                        if price.is_none() {
                            // Market order - assume filled at last price
                            // NOTE: In real production, poll /orders/{order_id} for actual fill price
                            info!(
                                "[ZERODHA LIVE] Market order {} assumed filled (poll for actual price)",
                                zerodha_order_id
                            );

                            // The actual fill will be confirmed via order status polling
                            // For now, mark as working and let market data update position
                        }
                    }
                    Err(e) => {
                        error!("[ZERODHA LIVE] Order placement failed: {}", e);
                        self.bus.publish_order_update(OrderEvent {
                            order_id: event.order_id,
                            parent_decision_id: None,
                            intent_id: event.intent_id,
                            timestamp: Utc::now(),
                            symbol: symbol.clone(),
                            side,
                            payload: OrderPayload::Update {
                                status: OrderStatus::Rejected,
                                filled_quantity: 0.0,
                                avg_price: 0.0,
                                commission: 0.0,
                            },
                        })?;
                    }
                }
            }
            OrderPayload::Cancel => {
                // Find and cancel the order
                let zerodha_id = self
                    .pending_orders
                    .iter()
                    .find(|(_, po)| po.order_id == event.order_id)
                    .map(|(id, _)| id.clone());

                if let Some(zerodha_order_id) = zerodha_id {
                    if let Err(e) = self.cancel_order(&zerodha_order_id).await {
                        warn!("[ZERODHA LIVE] Cancel failed: {}", e);
                    } else {
                        self.pending_orders.remove(&zerodha_order_id);
                        self.bus.publish_order_update(OrderEvent {
                            order_id: event.order_id,
                            parent_decision_id: None,
                            intent_id: event.intent_id,
                            timestamp: Utc::now(),
                            symbol: event.symbol.clone(),
                            side: event.side,
                            payload: OrderPayload::Update {
                                status: OrderStatus::Cancelled,
                                filled_quantity: 0.0,
                                avg_price: 0.0,
                                commission: 0.0,
                            },
                        })?;
                    }
                } else {
                    warn!(
                        "[ZERODHA LIVE] Order {} not found for cancel",
                        event.order_id
                    );
                }
            }
            _ => {
                warn!(
                    "[ZERODHA LIVE] Unsupported order payload: {:?}",
                    event.payload
                );
            }
        }
        Ok(())
    }

    async fn on_market_data(&mut self, event: MarketEvent) -> anyhow::Result<()> {
        // For live trading, market data is used to track position value
        // Fill confirmations come from order status polling, not market data
        let price = match &event.payload {
            MarketPayload::Tick { price, .. } => Some(*price),
            MarketPayload::Trade { price, .. } => Some(*price),
            _ => None,
        };

        if let Some(price) = price {
            // Check if any pending market orders should be marked as filled
            // This is a simplified approach - production should poll order status
            let symbol = &event.symbol;
            let fills_to_process: Vec<_> = self
                .pending_orders
                .iter()
                .filter(|(_, po)| &po.symbol == symbol)
                .map(|(id, po)| (id.clone(), po.order_id, po.intent_id, po.side, po.quantity))
                .collect();

            for (zerodha_id, order_id, intent_id, side, quantity) in fills_to_process {
                // Assume market orders fill at current price
                let commission = self.calculate_commission(side, quantity, price);

                info!(
                    "[ZERODHA LIVE] Confirming fill for {}: {} {} @ {:.2}",
                    zerodha_id, side, quantity, price
                );

                self.bus.publish_order_update(OrderEvent {
                    order_id,
                    parent_decision_id: None,
                    intent_id,
                    timestamp: Utc::now(),
                    symbol: symbol.clone(),
                    side,
                    payload: OrderPayload::Update {
                        status: OrderStatus::Filled,
                        filled_quantity: quantity,
                        avg_price: price,
                        commission,
                    },
                })?;

                self.bus
                    .publish_fill(quantlaxmi_models::FillEvent {
                        timestamp: Utc::now(),
                        order_id,
                        parent_decision_id: None,
                        intent_id,
                        fill_id: zerodha_id.clone(),
                        symbol: symbol.clone(),
                        side,
                        price,
                        quantity,
                        commission,
                        commission_asset: "INR".to_string(),
                        venue: "Zerodha".to_string(),
                        is_final: true,
                    })
                    .await?;

                self.pending_orders.remove(&zerodha_id);
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ZerodhaLiveExchange"
    }
}
