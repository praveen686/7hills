//! Canonical quote schema used across capture + strategy pipelines.
//!
//! This module re-exports the canonical QuoteEvent from quantlaxmi-models
//! and provides backward-compatible conversion utilities.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// Re-export the canonical event from models
pub use quantlaxmi_models::events::{
    CorrelationContext, ParseMantissaError, QuoteEvent as CanonicalQuoteEvent,
    parse_to_mantissa_pure,
};

/// Legacy QuoteEvent for backward compatibility with existing capture files.
/// New code should use `CanonicalQuoteEvent` from `quantlaxmi_models::events`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteEvent {
    pub ts: DateTime<Utc>,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,

    /// Best bid/ask prices as integer mantissas.
    pub bid_price_mantissa: i64,
    pub ask_price_mantissa: i64,

    /// Best bid/ask quantities as integer mantissas.
    pub bid_qty_mantissa: i64,
    pub ask_qty_mantissa: i64,

    /// Exponent for price mantissas (e.g., -2 means cents).
    pub price_exponent: i8,
    /// Exponent for quantity mantissas (e.g., -8 for BTC size precision).
    pub qty_exponent: i8,
}

impl QuoteEvent {
    pub fn bid_f64(&self) -> f64 {
        self.bid_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    pub fn ask_f64(&self) -> f64 {
        self.ask_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    pub fn mid_f64(&self) -> f64 {
        (self.bid_f64() + self.ask_f64()) / 2.0
    }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid_f64();
        if mid > 0.0 {
            (self.ask_f64() - self.bid_f64()) / mid * 10000.0
        } else {
            f64::MAX
        }
    }

    /// Convert to canonical QuoteEvent.
    pub fn to_canonical(&self, venue: impl Into<String>) -> CanonicalQuoteEvent {
        CanonicalQuoteEvent {
            ts: self.ts,
            symbol: self.symbol.clone(),
            bid_price_mantissa: self.bid_price_mantissa,
            ask_price_mantissa: self.ask_price_mantissa,
            bid_qty_mantissa: self.bid_qty_mantissa,
            ask_qty_mantissa: self.ask_qty_mantissa,
            price_exponent: self.price_exponent,
            qty_exponent: self.qty_exponent,
            venue: venue.into(),
            ctx: CorrelationContext::default(),
        }
    }
}

impl From<CanonicalQuoteEvent> for QuoteEvent {
    fn from(canonical: CanonicalQuoteEvent) -> Self {
        Self {
            ts: canonical.ts,
            symbol: canonical.symbol,
            bid_price_mantissa: canonical.bid_price_mantissa,
            ask_price_mantissa: canonical.ask_price_mantissa,
            bid_qty_mantissa: canonical.bid_qty_mantissa,
            ask_qty_mantissa: canonical.ask_qty_mantissa,
            price_exponent: canonical.price_exponent,
            qty_exponent: canonical.qty_exponent,
        }
    }
}
