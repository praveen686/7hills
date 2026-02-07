//! Basis snapshot: per-symbol spot/perp state (no funding fields).
//!
//! `BasisSnapshot` is the `TSnapshot` for the generic `PaperEngine`.
//! It aggregates data from 2 WebSocket streams:
//! - Spot @bookTicker  (best bid/ask)
//! - Perp @bookTicker  (best bid/ask)

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use quantlaxmi_paper::TopOfBookProvider;

/// Per-symbol market state for one basis pair.
#[derive(Debug, Clone, Default)]
pub struct SymbolState {
    pub symbol: String,
    /// Spot best bid
    pub spot_bid: f64,
    /// Spot best ask
    pub spot_ask: f64,
    /// Spot quote timestamp
    pub spot_ts: Option<DateTime<Utc>>,

    /// Perp best bid
    pub perp_bid: f64,
    /// Perp best ask
    pub perp_ask: f64,
    /// Perp quote timestamp
    pub perp_ts: Option<DateTime<Utc>>,
}

impl SymbolState {
    /// Spot mid price.
    pub fn spot_mid(&self) -> f64 {
        (self.spot_bid + self.spot_ask) / 2.0
    }

    /// Perp mid price.
    pub fn perp_mid(&self) -> f64 {
        (self.perp_bid + self.perp_ask) / 2.0
    }

    /// Basis in bps: (perp_mid - spot_mid) / spot_mid * 10000.
    pub fn basis_bps(&self) -> f64 {
        let spot = self.spot_mid();
        if spot <= 0.0 {
            return 0.0;
        }
        (self.perp_mid() - spot) / spot * 10_000.0
    }

    /// Whether both spot and perp quotes are present.
    pub fn has_quotes(&self) -> bool {
        self.spot_bid > 0.0 && self.spot_ask > 0.0 && self.perp_bid > 0.0 && self.perp_ask > 0.0
    }

    /// Combined spread in bps (spot spread + perp spread).
    pub fn combined_spread_bps(&self) -> f64 {
        let spot_spread = if self.spot_mid() > 0.0 {
            (self.spot_ask - self.spot_bid) / self.spot_mid() * 10_000.0
        } else {
            f64::MAX
        };
        let perp_spread = if self.perp_mid() > 0.0 {
            (self.perp_ask - self.perp_bid) / self.perp_mid() * 10_000.0
        } else {
            f64::MAX
        };
        spot_spread + perp_spread
    }

    /// Quote staleness in ms (max of spot/perp age).
    pub fn max_quote_age_ms(&self, now: DateTime<Utc>) -> i64 {
        let spot_age = self
            .spot_ts
            .map(|t| (now - t).num_milliseconds())
            .unwrap_or(i64::MAX);
        let perp_age = self
            .perp_ts
            .map(|t| (now - t).num_milliseconds())
            .unwrap_or(i64::MAX);
        spot_age.max(perp_age)
    }
}

/// Composite snapshot across all tracked symbols.
#[derive(Debug, Clone, Default)]
pub struct BasisSnapshot {
    pub ts: DateTime<Utc>,
    pub symbols: HashMap<String, SymbolState>,
}

impl BasisSnapshot {
    /// Get or create a symbol's state.
    pub fn get_or_insert(&mut self, symbol: &str) -> &mut SymbolState {
        self.symbols
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolState {
                symbol: symbol.to_string(),
                ..Default::default()
            })
    }
}

/// Deterministic hash of symbol string to u32 token for Ledger compatibility.
pub fn symbol_token(symbol: &str) -> u32 {
    let mut hash: u32 = 5381;
    for byte in symbol.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u32);
    }
    hash
}

/// Compute a token for a specific leg (spot vs perp).
pub fn leg_token(symbol: &str, is_perp: bool) -> u32 {
    let base = symbol_token(symbol);
    if is_perp {
        base.wrapping_add(1)
    } else {
        base
    }
}

impl TopOfBookProvider for BasisSnapshot {
    fn best_bid_ask(&self, token: u32) -> Option<(f64, f64)> {
        for (sym, state) in &self.symbols {
            let spot_tok = leg_token(sym, false);
            let perp_tok = leg_token(sym, true);

            if token == spot_tok && state.spot_bid > 0.0 && state.spot_ask > 0.0 {
                return Some((state.spot_bid, state.spot_ask));
            }
            if token == perp_tok && state.perp_bid > 0.0 && state.perp_ask > 0.0 {
                return Some((state.perp_bid, state.perp_ask));
            }
        }
        None
    }
}
