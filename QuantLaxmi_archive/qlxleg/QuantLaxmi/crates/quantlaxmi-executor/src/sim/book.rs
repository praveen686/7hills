//! Order book state for deterministic matching.
//!
//! Uses mantissa-based representation for determinism.

use std::collections::BTreeMap;

use quantlaxmi_models::depth::DepthEvent;

/// Per-symbol order book state (mantissa-based for determinism).
#[derive(Debug, Clone, Default)]
pub struct L2Book {
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
    /// Last timestamp
    last_ts_ns: u64,
}

impl L2Book {
    /// Create a new order book.
    pub fn new(price_exponent: i8, qty_exponent: i8) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            price_exponent,
            qty_exponent,
            last_ts_ns: 0,
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
        self.last_ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0) as u64;
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

    /// Get last timestamp.
    pub fn last_ts_ns(&self) -> u64 {
        self.last_ts_ns
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
