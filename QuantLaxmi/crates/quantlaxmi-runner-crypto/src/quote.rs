//! Canonical quote schema used across capture + strategy pipelines.
//!
//! This exists to prevent schema drift between live capture outputs and
//! downstream signal generation / scoring.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Canonical quote event from spot or perp bookTicker stream.
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
}
