//! Core data structures for SLRT reference implementation.
//!
//! All vendor fields are tri-state per QuantLaxmi Doctrine (spec: Section 1.2):
//! - Absent
//! - PresentNull
//! - Present(value)

use serde::{Deserialize, Serialize};

/// Tri-state field wrapper per QuantLaxmi Doctrine.
/// Any required field not in Present(value) triggers RefuseSignal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum TriState<T> {
    /// Field was not present in the source data
    #[default]
    Absent,
    /// Field was present but null/empty
    PresentNull,
    /// Field was present with a value
    Present(T),
}

impl<T> TriState<T> {
    /// Returns true if this is Present(value)
    pub fn is_present(&self) -> bool {
        matches!(self, TriState::Present(_))
    }

    /// Returns the inner value if Present, None otherwise
    pub fn as_option(&self) -> Option<&T> {
        match self {
            TriState::Present(v) => Some(v),
            _ => None,
        }
    }

    /// Unwrap or return default (for optional fields only)
    pub fn unwrap_or(&self, default: T) -> T
    where
        T: Clone,
    {
        match self {
            TriState::Present(v) => v.clone(),
            _ => default,
        }
    }
}

/// Single price level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price as fixed-point mantissa
    pub price_mantissa: i64,
    /// Price exponent (e.g., -2 for cents)
    pub price_exponent: i8,
    /// Quantity as fixed-point mantissa
    pub qty_mantissa: i64,
    /// Quantity exponent
    pub qty_exponent: i8,
}

impl PriceLevel {
    /// Convert price to f64 for computation
    pub fn price_f64(&self) -> f64 {
        self.price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert quantity to f64 for computation
    pub fn qty_f64(&self) -> f64 {
        self.qty_mantissa as f64 * 10f64.powi(self.qty_exponent as i32)
    }
}

/// Limit Order Book snapshot (top N levels per side).
/// Spec: Section 3.1 - L2 order book (top N levels, default N=20)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Timestamp in nanoseconds (canonical time = receive timestamp)
    pub ts_ns: i64,
    /// Symbol identifier
    pub symbol: String,
    /// Bid levels (best bid first, index 0 = top of book)
    pub bids: Vec<PriceLevel>,
    /// Ask levels (best ask first, index 0 = top of book)
    pub asks: Vec<PriceLevel>,
}

impl OrderBook {
    /// Check if book is crossed (invalid state)
    pub fn is_crossed(&self) -> bool {
        if self.bids.is_empty() || self.asks.is_empty() {
            return false;
        }
        self.bids[0].price_f64() >= self.asks[0].price_f64()
    }

    /// Best bid price (None if empty)
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price_f64())
    }

    /// Best ask price (None if empty)
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price_f64())
    }

    /// Best bid quantity (None if empty)
    pub fn best_bid_qty(&self) -> Option<f64> {
        self.bids.first().map(|l| l.qty_f64())
    }

    /// Best ask quantity (None if empty)
    pub fn best_ask_qty(&self) -> Option<f64> {
        self.asks.first().map(|l| l.qty_f64())
    }

    /// Spread in price units
    pub fn spread(&self) -> Option<f64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        }
    }
}

/// Trade side (aggressor)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
    Unknown,
}

/// Single trade event.
/// Spec: Section 3.1 - Trades (price, quantity, timestamp)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Timestamp in nanoseconds
    pub ts_ns: i64,
    /// Symbol identifier
    pub symbol: String,
    /// Trade price (fixed-point mantissa)
    pub price_mantissa: i64,
    /// Price exponent
    pub price_exponent: i8,
    /// Trade quantity (fixed-point mantissa)
    pub qty_mantissa: i64,
    /// Quantity exponent
    pub qty_exponent: i8,
    /// Aggressor side (tri-state per doctrine)
    pub side: TriState<TradeSide>,
}

impl Trade {
    /// Convert price to f64
    pub fn price_f64(&self) -> f64 {
        self.price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert quantity to f64
    pub fn qty_f64(&self) -> f64 {
        self.qty_mantissa as f64 * 10f64.powi(self.qty_exponent as i32)
    }
}

/// Venue metadata (static configuration).
/// Spec: Section 3.1 - Static venue metadata (tick size, lot size, limits)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueMetadata {
    pub symbol: String,
    /// Tick size (minimum price increment)
    pub tick_size: f64,
    /// Lot size (minimum quantity increment)
    pub lot_size: f64,
    /// Maximum order size
    pub max_order_size: Option<f64>,
    /// Maximum position size
    pub max_position_size: Option<f64>,
}

/// Funding rate update (perpetuals only).
/// Spec: Section 3.1 - Funding stream (optional for perpetuals)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingUpdate {
    pub ts_ns: i64,
    pub symbol: String,
    /// Funding rate as decimal (e.g., 0.0001 = 0.01%)
    pub rate: TriState<f64>,
    /// Next funding timestamp
    pub next_funding_ts_ns: TriState<i64>,
}

/// Basis update (perpetuals only).
/// Spec: Section 3.1 - Basis stream (optional for perpetuals)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisUpdate {
    pub ts_ns: i64,
    pub symbol: String,
    /// Basis in price units (perp - spot)
    pub basis: TriState<f64>,
}

/// Market event enum for unified event stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MarketEvent {
    Book(OrderBook),
    Trade(Trade),
    Funding(FundingUpdate),
    Basis(BasisUpdate),
}

impl MarketEvent {
    /// Get timestamp in nanoseconds
    pub fn ts_ns(&self) -> i64 {
        match self {
            MarketEvent::Book(b) => b.ts_ns,
            MarketEvent::Trade(t) => t.ts_ns,
            MarketEvent::Funding(f) => f.ts_ns,
            MarketEvent::Basis(b) => b.ts_ns,
        }
    }

    /// Get symbol
    pub fn symbol(&self) -> &str {
        match self {
            MarketEvent::Book(b) => &b.symbol,
            MarketEvent::Trade(t) => &t.symbol,
            MarketEvent::Funding(f) => &f.symbol,
            MarketEvent::Basis(b) => &b.symbol,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_parse() {
        let json = r#"{"type":"Trade","ts_ns":1769097411822776000,"symbol":"BTCUSDT","price_mantissa":8903490,"price_exponent":-2,"qty_mantissa":22000,"qty_exponent":-8,"side":{"Present":"Sell"}}"#;
        let result = serde_json::from_str::<MarketEvent>(json);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        match result.unwrap() {
            MarketEvent::Trade(t) => {
                assert_eq!(t.symbol, "BTCUSDT");
                assert!(matches!(t.side, TriState::Present(TradeSide::Sell)));
            }
            _ => panic!("Expected Trade variant"),
        }
    }

    #[test]
    fn test_book_parse() {
        let json = r#"{"type":"Book","ts_ns":1769097412164524000,"symbol":"BTCUSDT","bids":[{"price_mantissa":8903574,"price_exponent":-2,"qty_mantissa":500000,"qty_exponent":-8}],"asks":[{"price_mantissa":8903575,"price_exponent":-2,"qty_mantissa":400000,"qty_exponent":-8}]}"#;
        let result = serde_json::from_str::<MarketEvent>(json);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
        match result.unwrap() {
            MarketEvent::Book(b) => {
                assert_eq!(b.symbol, "BTCUSDT");
                assert!(!b.bids.is_empty());
            }
            _ => panic!("Expected Book variant"),
        }
    }
}
