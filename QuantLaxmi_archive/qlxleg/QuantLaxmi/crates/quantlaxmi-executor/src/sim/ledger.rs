//! Ledger for tracking cash, positions, and realized PnL.

use std::collections::HashMap;

use super::types::Side;

/// Position for a single symbol.
#[derive(Debug, Clone, Default)]
pub struct Position {
    /// Position quantity (positive = long, negative = short)
    pub qty: f64,
    /// Average entry price
    pub avg_price: f64,
}

/// Simulator state (portfolio).
#[derive(Debug, Default)]
pub struct Ledger {
    /// Cash balance
    pub cash: f64,
    /// Full position tracking with avg price
    pub positions: HashMap<String, Position>,
    /// Realized PnL
    pub realized_pnl: f64,
}

impl Ledger {
    /// Create a new ledger with initial cash.
    pub fn new(initial_cash: f64) -> Self {
        Self {
            cash: initial_cash,
            positions: HashMap::new(),
            realized_pnl: 0.0,
        }
    }

    /// Get position for a symbol (0 if not present).
    pub fn position(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map(|p| p.qty).unwrap_or(0.0)
    }

    /// Process a fill and update ledger state.
    ///
    /// Returns true if the fill was processed successfully.
    pub fn on_fill(&mut self, symbol: &str, side: Side, qty: f64, price: f64, fee: f64) -> bool {
        let notional = price * qty;

        // Check cash for buys
        if matches!(side, Side::Buy) && self.cash < notional + fee {
            tracing::warn!(
                "Insufficient cash: need {:.2}, have {:.2}",
                notional + fee,
                self.cash
            );
            return false;
        }

        // Update position and cash
        let position = self.positions.entry(symbol.to_string()).or_default();
        let old_qty = position.qty;

        match side {
            Side::Buy => {
                self.cash -= notional + fee;

                if position.qty >= 0.0 {
                    // Adding to long or opening long
                    let total_cost = position.avg_price * position.qty + price * qty;
                    position.qty += qty;
                    position.avg_price = if position.qty > 0.0 {
                        total_cost / position.qty
                    } else {
                        0.0
                    };
                } else {
                    // Covering short
                    let cover_qty = qty.min(-position.qty);
                    let pnl = (position.avg_price - price) * cover_qty;
                    self.realized_pnl += pnl;
                    self.cash += pnl;

                    position.qty += qty;
                    if position.qty > 0.0 {
                        position.avg_price = price;
                    }
                }
            }
            Side::Sell => {
                if position.qty <= 0.0 {
                    // Adding to short or opening short
                    let total_cost = position.avg_price * (-position.qty) + price * qty;
                    position.qty -= qty;
                    position.avg_price = if position.qty < 0.0 {
                        total_cost / (-position.qty)
                    } else {
                        0.0
                    };
                } else {
                    // Closing long
                    let close_qty = qty.min(position.qty);
                    let pnl = (price - position.avg_price) * close_qty;
                    self.realized_pnl += pnl;

                    position.qty -= qty;
                    if position.qty < 0.0 {
                        position.avg_price = price;
                    }
                }

                self.cash += notional - fee;
            }
        }

        tracing::debug!(
            "LEDGER: {} {} {:.4} @ {:.2} (fee={:.4}, pos: {:.4} -> {:.4})",
            side,
            symbol,
            qty,
            price,
            fee,
            old_qty,
            position.qty
        );

        true
    }
}
