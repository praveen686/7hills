//! Simulator types for paper trading and backtest.
//!
//! These types are the canonical representation for order execution simulation.

use serde::{Deserialize, Serialize};

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
    /// Timestamp when order was created (nanoseconds)
    pub created_ts_ns: u64,
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
            created_ts_ns: 0,
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
            created_ts_ns: 0,
        }
    }

    /// Add a tag for tracking.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Set creation timestamp.
    pub fn with_created_ts_ns(mut self, ts_ns: u64) -> Self {
        self.created_ts_ns = ts_ns;
        self
    }
}

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
