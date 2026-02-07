//! Multi-pair capital allocation and funding payment tracking.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

/// Per-position funding tracker.
#[derive(Debug, Clone, Default)]
pub struct PositionFunding {
    /// Notional value in USDT
    pub notional_usd: f64,
    /// Spot leg quantity (for exact exit sizing)
    pub spot_qty: f64,
    /// Perp leg quantity (for exact exit sizing)
    pub perp_qty: f64,
    /// Total funding payments collected (positive = received)
    pub total_funding_usd: f64,
    /// Number of funding settlements collected
    pub settlements: u32,
    /// When the position was opened
    pub entry_ts: Option<DateTime<Utc>>,
    /// Round-trip cost paid to enter (for breakeven tracking)
    pub entry_cost_usd: f64,
}

impl PositionFunding {
    /// Funding collected minus entry cost = net profit from this position so far.
    pub fn net_profit(&self) -> f64 {
        self.total_funding_usd - self.entry_cost_usd
    }

    /// Settlements needed to break even: entry_cost / avg_funding_per_settlement.
    /// Returns None if no settlements yet.
    pub fn settlements_to_breakeven(&self) -> Option<f64> {
        if self.settlements == 0 {
            return None;
        }
        let avg = self.total_funding_usd / self.settlements as f64;
        if avg <= 0.0 {
            return None;
        }
        // Total cost to recover = entry cost + exit cost (assume same as entry)
        let total_cost = self.entry_cost_usd * 2.0;
        Some(total_cost / avg)
    }

    /// Whether this position has collected enough funding to cover round-trip cost.
    pub fn is_breakeven(&self) -> bool {
        // funding collected >= entry cost * 2 (entry + exit cost)
        self.total_funding_usd >= self.entry_cost_usd * 2.0
    }

    /// Hold duration in hours.
    pub fn hold_hours(&self, now: DateTime<Utc>) -> f64 {
        self.entry_ts
            .map(|t| (now - t).num_minutes() as f64 / 60.0)
            .unwrap_or(0.0)
    }
}

/// Portfolio: tracks capital allocation across multiple pairs.
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub initial_capital: f64,
    pub allocated_usd: f64,
    pub positions: HashMap<String, PositionFunding>,
    /// Total funding collected across all positions (lifetime)
    pub total_funding_collected: f64,
}

impl Portfolio {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            allocated_usd: 0.0,
            positions: HashMap::new(),
            total_funding_collected: 0.0,
        }
    }

    /// Available capital for new positions.
    pub fn available_capital(&self) -> f64 {
        self.initial_capital - self.allocated_usd
    }

    /// Number of active positions.
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get notional for a symbol (if position exists).
    pub fn position_notional(&self, symbol: &str) -> Option<f64> {
        self.positions.get(symbol).map(|p| p.notional_usd)
    }

    /// Get position funding tracker (if exists).
    pub fn get_position(&self, symbol: &str) -> Option<&PositionFunding> {
        self.positions.get(symbol)
    }

    /// Add a new position allocation.
    pub fn add_position(
        &mut self,
        symbol: &str,
        notional_usd: f64,
        spot_qty: f64,
        perp_qty: f64,
        entry_cost_usd: f64,
        ts: DateTime<Utc>,
    ) {
        self.allocated_usd += notional_usd;
        self.positions.insert(
            symbol.to_string(),
            PositionFunding {
                notional_usd,
                spot_qty,
                perp_qty,
                entry_cost_usd,
                entry_ts: Some(ts),
                ..Default::default()
            },
        );
    }

    /// Remove a position (release capital).
    pub fn remove_position(&mut self, symbol: &str) {
        if let Some(pos) = self.positions.remove(symbol) {
            self.allocated_usd -= pos.notional_usd;
        }
    }

    /// Record a funding payment for a symbol.
    pub fn record_funding(&mut self, symbol: &str, payment_usd: f64) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.total_funding_usd += payment_usd;
            pos.settlements += 1;
            self.total_funding_collected += payment_usd;
        }
    }

    /// Annualized yield from collected funding.
    pub fn funding_yield_annualized(&self) -> f64 {
        if self.allocated_usd <= 0.0 {
            return 0.0;
        }
        let total_settlements: u32 = self.positions.values().map(|p| p.settlements).sum();
        if total_settlements == 0 {
            return 0.0;
        }
        let yield_per_settlement = self.total_funding_collected / self.allocated_usd;
        let settlements_per_year = 1095.0;
        yield_per_settlement / total_settlements as f64 * settlements_per_year * 100.0
    }
}
