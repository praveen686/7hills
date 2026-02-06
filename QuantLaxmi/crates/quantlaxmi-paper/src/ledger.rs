//! Paper Trading Ledger
//!
//! Tracks cash, positions, PnL, and fees for paper trading.
//!
//! ## Conservative MTM
//!
//! - Long positions marked at bid (what we could sell for)
//! - Short positions marked at ask (what we'd pay to cover)
//! - This prevents paper trading illusion of profitability

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Fill, FillSide};

// =============================================================================
// POSITION
// =============================================================================

/// A position in a single instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Instrument symbol
    pub symbol: String,
    /// Instrument token (for India) or symbol hash (for Crypto)
    pub token: u32,
    /// Position quantity (positive = long, negative = short)
    pub qty: f64,
    /// Average entry price
    pub avg_price: f64,
    /// Realized PnL for this position (from partial closes)
    pub realized_pnl: f64,
    /// Total fees paid for this position
    pub fees_paid: f64,
    /// Entry timestamp
    pub entry_ts: Option<DateTime<Utc>>,
    /// Last update timestamp
    pub last_update: Option<DateTime<Utc>>,
}

impl Position {
    /// Create a new position from a fill.
    pub fn from_fill(fill: &Fill, token: u32) -> Self {
        let qty = match fill.side {
            FillSide::Buy => fill.qty,
            FillSide::Sell => -fill.qty,
        };
        Self {
            symbol: fill.symbol.clone(),
            token,
            qty,
            avg_price: fill.price,
            realized_pnl: 0.0,
            fees_paid: fill.fees.total,
            entry_ts: Some(fill.ts),
            last_update: Some(fill.ts),
        }
    }

    /// Apply a fill to this position.
    ///
    /// Returns realized PnL if the fill reduces the position.
    pub fn apply_fill(&mut self, fill: &Fill) -> f64 {
        let fill_qty = match fill.side {
            FillSide::Buy => fill.qty,
            FillSide::Sell => -fill.qty,
        };

        let mut realized = 0.0;

        // Check if this reduces or adds to position
        let same_direction =
            (self.qty > 0.0 && fill_qty > 0.0) || (self.qty < 0.0 && fill_qty < 0.0);

        if same_direction || self.qty == 0.0 {
            // Adding to position - update average price
            let total_cost = self.qty * self.avg_price + fill_qty * fill.price;
            let new_qty = self.qty + fill_qty;
            if new_qty.abs() > 1e-9 {
                self.avg_price = total_cost / new_qty;
            }
            self.qty = new_qty;
        } else {
            // Reducing or flipping position
            let close_qty = fill_qty.abs().min(self.qty.abs());

            // Calculate realized PnL
            if self.qty > 0.0 {
                // Was long, selling
                realized = close_qty * (fill.price - self.avg_price);
            } else {
                // Was short, buying to cover
                realized = close_qty * (self.avg_price - fill.price);
            }

            self.realized_pnl += realized;
            let remaining = self.qty.abs() - close_qty;

            if fill_qty.abs() > self.qty.abs() {
                // Flip position
                let flip_qty = fill_qty.abs() - self.qty.abs();
                self.qty = fill_qty.signum() * flip_qty;
                self.avg_price = fill.price;
            } else {
                // Just reduce
                self.qty = self.qty.signum() * remaining;
            }
        }

        self.fees_paid += fill.fees.total;
        self.last_update = Some(fill.ts);

        realized
    }

    /// Check if position is flat (zero quantity).
    pub fn is_flat(&self) -> bool {
        self.qty.abs() < 1e-9
    }

    /// Calculate unrealized PnL with conservative MTM.
    ///
    /// - Long positions use bid price
    /// - Short positions use ask price
    pub fn unrealized_pnl(&self, bid: f64, ask: f64) -> f64 {
        if self.qty > 0.0 {
            // Long: would sell at bid
            self.qty * (bid - self.avg_price)
        } else if self.qty < 0.0 {
            // Short: would buy at ask
            self.qty.abs() * (self.avg_price - ask)
        } else {
            0.0
        }
    }
}

// =============================================================================
// FILL LOG ENTRY
// =============================================================================

/// Log entry for a fill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillLogEntry {
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub token: u32,
    pub side: String,
    pub qty: f64,
    pub price: f64,
    pub fees: f64,
    pub realized_pnl: f64,
    pub tag: String,
}

// =============================================================================
// FEES BREAKDOWN
// =============================================================================

/// Aggregated fees tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeesAggregate {
    /// Total fees paid
    pub total: f64,
    /// Number of fills
    pub fill_count: u64,
    /// Number of rejections
    pub reject_count: u64,
}

// =============================================================================
// LEDGER
// =============================================================================

/// Paper trading ledger with position tracking and conservative MTM.
#[derive(Debug, Default)]
pub struct Ledger {
    /// Positions by token
    positions: HashMap<u32, Position>,
    /// Starting cash
    pub initial_capital: f64,
    /// Current cash balance
    pub cash: f64,
    /// Total realized PnL (sum of all position realized PnL)
    pub realized_pnl: f64,
    /// Aggregated fees
    pub fees: FeesAggregate,
    /// Fill log
    pub fill_log: Vec<FillLogEntry>,
    /// Last update timestamp
    pub last_update: Option<DateTime<Utc>>,
}

impl Ledger {
    /// Create a new ledger with initial capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            cash: initial_capital,
            ..Default::default()
        }
    }

    /// Apply a fill to the ledger.
    ///
    /// Returns the realized PnL from this fill (if position was reduced).
    pub fn apply_fill(&mut self, fill: &Fill, token: u32, tag: &str) -> f64 {
        let realized = if let Some(pos) = self.positions.get_mut(&token) {
            pos.apply_fill(fill)
        } else {
            // New position
            self.positions
                .insert(token, Position::from_fill(fill, token));
            0.0
        };

        // Update cash
        let cash_impact = match fill.side {
            FillSide::Buy => -fill.qty * fill.price - fill.fees.total,
            FillSide::Sell => fill.qty * fill.price - fill.fees.total,
        };
        self.cash += cash_impact;

        // Update aggregates
        self.realized_pnl += realized;
        self.fees.total += fill.fees.total;
        self.fees.fill_count += 1;
        self.last_update = Some(fill.ts);

        // Log the fill
        self.fill_log.push(FillLogEntry {
            ts: fill.ts,
            symbol: fill.symbol.clone(),
            token,
            side: format!("{:?}", fill.side),
            qty: fill.qty,
            price: fill.price,
            fees: fill.fees.total,
            realized_pnl: realized,
            tag: tag.to_string(),
        });

        realized
    }

    /// Record a fill rejection.
    pub fn record_rejection(&mut self) {
        self.fees.reject_count += 1;
    }

    /// Get position by token.
    pub fn get_position(&self, token: u32) -> Option<&Position> {
        self.positions.get(&token)
    }

    /// Get all positions.
    pub fn positions(&self) -> impl Iterator<Item = &Position> {
        self.positions.values()
    }

    /// Get number of open positions (non-zero qty).
    pub fn open_position_count(&self) -> usize {
        self.positions.values().filter(|p| !p.is_flat()).count()
    }

    /// Calculate total unrealized PnL with conservative MTM.
    ///
    /// Takes a price provider closure that returns (bid, ask) for each token.
    pub fn unrealized_pnl<F>(&self, price_provider: F) -> f64
    where
        F: Fn(u32) -> Option<(f64, f64)>,
    {
        self.positions
            .iter()
            .filter(|(_, pos)| !pos.is_flat())
            .map(|(token, pos)| {
                if let Some((bid, ask)) = price_provider(*token) {
                    pos.unrealized_pnl(bid, ask)
                } else {
                    // If no price, use last known (zero unrealized)
                    0.0
                }
            })
            .sum()
    }

    /// Calculate total equity with conservative MTM.
    pub fn equity<F>(&self, price_provider: F) -> f64
    where
        F: Fn(u32) -> Option<(f64, f64)>,
    {
        self.cash + self.unrealized_pnl(price_provider)
    }

    /// Get a summary of the ledger state.
    pub fn summary<F>(&self, price_provider: F) -> LedgerSummary
    where
        F: Fn(u32) -> Option<(f64, f64)>,
    {
        let unrealized = self.unrealized_pnl(&price_provider);
        LedgerSummary {
            initial_capital: self.initial_capital,
            cash: self.cash,
            realized_pnl: self.realized_pnl,
            unrealized_pnl: unrealized,
            equity: self.cash + unrealized,
            total_pnl: (self.cash + unrealized) - self.initial_capital,
            fees_paid: self.fees.total,
            fill_count: self.fees.fill_count,
            reject_count: self.fees.reject_count,
            open_positions: self.open_position_count(),
        }
    }
}

/// Ledger summary for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerSummary {
    pub initial_capital: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub equity: f64,
    pub fees_paid: f64,
    pub fill_count: u64,
    pub reject_count: u64,
    pub open_positions: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fees;

    fn make_fill(symbol: &str, side: FillSide, qty: f64, price: f64, fees: f64) -> Fill {
        Fill {
            ts: Utc::now(),
            symbol: symbol.to_string(),
            side,
            qty,
            price,
            fees: Fees { total: fees },
        }
    }

    #[test]
    fn test_position_from_buy_fill() {
        let fill = make_fill("NIFTY", FillSide::Buy, 1.0, 100.0, 5.0);
        let pos = Position::from_fill(&fill, 123);

        assert_eq!(pos.qty, 1.0);
        assert_eq!(pos.avg_price, 100.0);
        assert_eq!(pos.fees_paid, 5.0);
    }

    #[test]
    fn test_position_from_sell_fill() {
        let fill = make_fill("NIFTY", FillSide::Sell, 1.0, 100.0, 5.0);
        let pos = Position::from_fill(&fill, 123);

        assert_eq!(pos.qty, -1.0); // Negative = short
        assert_eq!(pos.avg_price, 100.0);
    }

    #[test]
    fn test_position_add_to_long() {
        let fill1 = make_fill("NIFTY", FillSide::Buy, 1.0, 100.0, 5.0);
        let mut pos = Position::from_fill(&fill1, 123);

        let fill2 = make_fill("NIFTY", FillSide::Buy, 1.0, 110.0, 5.0);
        let realized = pos.apply_fill(&fill2);

        assert_eq!(realized, 0.0); // No realized PnL when adding
        assert_eq!(pos.qty, 2.0);
        assert_eq!(pos.avg_price, 105.0); // Average of 100 and 110
    }

    #[test]
    fn test_position_close_long_profit() {
        let fill1 = make_fill("NIFTY", FillSide::Buy, 1.0, 100.0, 5.0);
        let mut pos = Position::from_fill(&fill1, 123);

        let fill2 = make_fill("NIFTY", FillSide::Sell, 1.0, 120.0, 5.0);
        let realized = pos.apply_fill(&fill2);

        assert_eq!(realized, 20.0); // Profit: sold at 120, bought at 100
        assert!(pos.is_flat());
    }

    #[test]
    fn test_position_close_short_profit() {
        let fill1 = make_fill("NIFTY", FillSide::Sell, 1.0, 120.0, 5.0);
        let mut pos = Position::from_fill(&fill1, 123);

        let fill2 = make_fill("NIFTY", FillSide::Buy, 1.0, 100.0, 5.0);
        let realized = pos.apply_fill(&fill2);

        assert_eq!(realized, 20.0); // Profit: sold at 120, covered at 100
        assert!(pos.is_flat());
    }

    #[test]
    fn test_conservative_mtm_long() {
        let fill = make_fill("NIFTY", FillSide::Buy, 1.0, 100.0, 5.0);
        let pos = Position::from_fill(&fill, 123);

        // Long position uses bid for MTM
        let unrealized = pos.unrealized_pnl(105.0, 110.0);
        assert_eq!(unrealized, 5.0); // bid - avg = 105 - 100
    }

    #[test]
    fn test_conservative_mtm_short() {
        let fill = make_fill("NIFTY", FillSide::Sell, 1.0, 100.0, 5.0);
        let pos = Position::from_fill(&fill, 123);

        // Short position uses ask for MTM
        let unrealized = pos.unrealized_pnl(85.0, 90.0);
        assert_eq!(unrealized, 10.0); // avg - ask = 100 - 90
    }

    #[test]
    fn test_ledger_apply_fill() {
        let mut ledger = Ledger::new(100000.0);

        let fill = make_fill("NIFTY", FillSide::Buy, 10.0, 100.0, 20.0);
        ledger.apply_fill(&fill, 123, "test");

        assert_eq!(ledger.cash, 100000.0 - 10.0 * 100.0 - 20.0); // Initial - cost - fees
        assert_eq!(ledger.fees.total, 20.0);
        assert_eq!(ledger.fees.fill_count, 1);
        assert_eq!(ledger.open_position_count(), 1);
    }

    #[test]
    fn test_ledger_equity_with_conservative_mtm() {
        let mut ledger = Ledger::new(100000.0);

        let fill = make_fill("NIFTY", FillSide::Buy, 10.0, 100.0, 20.0);
        ledger.apply_fill(&fill, 123, "test");

        // Cash after buy: 100000 - 1000 - 20 = 98980
        // Position: 10 @ 100
        // MTM at bid=105: unrealized = 10 * (105 - 100) = 50
        // Equity = 98980 + 50 = 99030

        let equity = ledger.equity(|_| Some((105.0, 110.0)));
        assert_eq!(equity, 99030.0);
    }

    #[test]
    fn test_ledger_summary() {
        let mut ledger = Ledger::new(100000.0);

        let fill = make_fill("NIFTY", FillSide::Buy, 10.0, 100.0, 20.0);
        ledger.apply_fill(&fill, 123, "test");

        let summary = ledger.summary(|_| Some((105.0, 110.0)));

        assert_eq!(summary.initial_capital, 100000.0);
        assert_eq!(summary.fees_paid, 20.0);
        assert_eq!(summary.fill_count, 1);
        assert_eq!(summary.open_positions, 1);
        assert_eq!(summary.unrealized_pnl, 50.0);
    }
}
