//! Market Data Hub — wraps quantlaxmi-data for terminal access.
//!
//! Manages L2 orderbooks, bar aggregation, and VPIN calculation
//! for all subscribed symbols.

use quantlaxmi_data::{BarAggregator, Level2Book, VpinCalculator};
use quantlaxmi_models::MarketEvent;
use std::collections::HashMap;

/// Central hub for all market data processing in the terminal.
pub struct MarketDataHub {
    /// Per-symbol L2 orderbooks.
    pub books: HashMap<String, Level2Book>,
    /// Per-symbol bar aggregators (1-minute default).
    pub bar_aggs: HashMap<String, BarAggregator>,
    /// Per-symbol VPIN calculators.
    pub vpins: HashMap<String, VpinCalculator>,
    /// Last known prices.
    pub last_prices: HashMap<String, f64>,
}

impl MarketDataHub {
    pub fn new() -> Self {
        Self {
            books: HashMap::new(),
            bar_aggs: HashMap::new(),
            vpins: HashMap::new(),
            last_prices: HashMap::new(),
        }
    }

    /// Subscribe to a symbol — creates book, bar agg, and VPIN calculator.
    pub fn subscribe(&mut self, symbol: &str) {
        if !self.books.contains_key(symbol) {
            self.books
                .insert(symbol.to_string(), Level2Book::new(symbol.to_string()));
        }
        if !self.bar_aggs.contains_key(symbol) {
            self.bar_aggs
                .insert(symbol.to_string(), BarAggregator::new(60_000));
        }
        if !self.vpins.contains_key(symbol) {
            self.vpins.insert(
                symbol.to_string(),
                VpinCalculator::new(symbol.to_string(), 1000.0, 50),
            );
        }
    }

    /// Process an incoming market event — updates book, bars, VPIN.
    pub fn on_event(&mut self, event: &MarketEvent) -> Option<MarketEvent> {
        let symbol = &event.symbol;

        // Update last price from tick
        if let quantlaxmi_models::MarketPayload::Tick { price, .. } = &event.payload {
            self.last_prices.insert(symbol.clone(), *price);
        }

        // Aggregate bars
        if let Some(agg) = self.bar_aggs.get_mut(symbol) {
            return agg.handle_tick(event);
        }

        None
    }

    /// Get a snapshot of the orderbook for a symbol.
    pub fn get_book(&self, symbol: &str) -> Option<&Level2Book> {
        self.books.get(symbol)
    }

    /// Get the latest VPIN value for a symbol.
    pub fn get_vpin(&self, symbol: &str) -> Option<f64> {
        self.vpins.get(symbol).and_then(|calc| {
            calc.vpin_history.last().copied()
        })
    }

    /// Get the last known price for a symbol.
    pub fn get_last_price(&self, symbol: &str) -> Option<f64> {
        self.last_prices.get(symbol).copied()
    }
}

impl Default for MarketDataHub {
    fn default() -> Self {
        Self::new()
    }
}
