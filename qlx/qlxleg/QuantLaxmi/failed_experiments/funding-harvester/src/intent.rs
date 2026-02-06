//! Funding arbitrage intents.
//!
//! Each trade is two intents: one spot leg + one perp leg.
//! The engine iterates intents and fills each independently via the fill model.
//!
//! ## Dollar-neutral hedging (Lever 7)
//! The perp leg qty is adjusted so that dollar notional matches the spot leg:
//!   perp_qty = spot_qty * spot_price / perp_price
//! This prevents basis drift from creating unhedged P&L.

use quantlaxmi_paper::InstrumentIdentity;

use crate::snapshot::leg_token;

/// Which market the leg executes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Venue {
    Spot,
    Perp,
}

/// Buy or sell direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// A single-leg intent for the funding arbitrage strategy.
///
/// Entry: [SpotBuy, PerpSell]  — buy spot, short perp
/// Exit:  [SpotSell, PerpBuy]  — sell spot, cover perp
#[derive(Debug, Clone)]
pub struct FundingArbIntent {
    pub symbol: String,
    pub venue: Venue,
    pub side: Side,
    pub qty: f64,
}

impl FundingArbIntent {
    /// Create dollar-neutral entry pair: buy spot + short perp.
    ///
    /// `spot_qty` is the base quantity for the spot leg.
    /// `spot_price` / `perp_price` are used to compute the perp qty
    /// so that both legs have equal dollar notional.
    pub fn entry_pair(
        symbol: &str,
        spot_qty: f64,
        spot_price: f64,
        perp_price: f64,
    ) -> Vec<Self> {
        // Dollar-neutral: perp_notional = spot_notional
        // perp_qty * perp_price = spot_qty * spot_price
        let perp_qty = if perp_price > 0.0 {
            spot_qty * spot_price / perp_price
        } else {
            spot_qty
        };
        vec![
            FundingArbIntent {
                symbol: symbol.to_string(),
                venue: Venue::Spot,
                side: Side::Buy,
                qty: spot_qty,
            },
            FundingArbIntent {
                symbol: symbol.to_string(),
                venue: Venue::Perp,
                side: Side::Sell,
                qty: perp_qty,
            },
        ]
    }

    /// Create dollar-neutral exit pair: sell spot + cover perp.
    pub fn exit_pair(
        symbol: &str,
        spot_qty: f64,
        spot_price: f64,
        perp_price: f64,
    ) -> Vec<Self> {
        let perp_qty = if perp_price > 0.0 {
            spot_qty * spot_price / perp_price
        } else {
            spot_qty
        };
        vec![
            FundingArbIntent {
                symbol: symbol.to_string(),
                venue: Venue::Spot,
                side: Side::Sell,
                qty: spot_qty,
            },
            FundingArbIntent {
                symbol: symbol.to_string(),
                venue: Venue::Perp,
                side: Side::Buy,
                qty: perp_qty,
            },
        ]
    }
}

impl InstrumentIdentity for FundingArbIntent {
    type Key = u32;

    fn instrument_key(&self) -> u32 {
        let is_perp = self.venue == Venue::Perp;
        leg_token(&self.symbol, is_perp)
    }
}
