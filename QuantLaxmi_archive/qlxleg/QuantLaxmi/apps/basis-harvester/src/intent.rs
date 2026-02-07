//! Basis trading intents with bidirectional support.
//!
//! Each trade is two intents: one spot leg + one perp leg.
//! Unlike funding-harvester (always buy-spot/sell-perp), basis mean-reversion
//! supports both directions:
//!
//! - **ShortBasis**: basis too wide → buy spot, sell perp (expect narrowing)
//! - **LongBasis**: basis too narrow → sell spot, buy perp (expect widening)
//!
//! ## Dollar-neutral hedging
//! The perp leg qty is adjusted so that dollar notional matches the spot leg:
//!   perp_qty = spot_qty * spot_price / perp_price

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

/// Direction of the basis trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisDirection {
    /// Basis too wide (z > +threshold) → buy spot, sell perp → expect narrowing.
    ShortBasis,
    /// Basis too narrow (z < -threshold) → sell spot, buy perp → expect widening.
    LongBasis,
}

impl std::fmt::Display for BasisDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisDirection::ShortBasis => write!(f, "SB"),
            BasisDirection::LongBasis => write!(f, "LB"),
        }
    }
}

/// A single-leg intent for the basis mean-reversion strategy.
#[derive(Debug, Clone)]
pub struct BasisIntent {
    pub symbol: String,
    pub venue: Venue,
    pub side: Side,
    pub qty: f64,
    pub direction: BasisDirection,
}

impl BasisIntent {
    /// Create dollar-neutral entry pair based on basis direction.
    ///
    /// ShortBasis: buy spot + sell perp (basis too wide, expect narrowing)
    /// LongBasis:  sell spot + buy perp (basis too narrow, expect widening)
    pub fn entry_pair(
        symbol: &str,
        spot_qty: f64,
        spot_price: f64,
        perp_price: f64,
        direction: BasisDirection,
    ) -> Vec<Self> {
        let perp_qty = if perp_price > 0.0 {
            spot_qty * spot_price / perp_price
        } else {
            spot_qty
        };

        let (spot_side, perp_side) = match direction {
            BasisDirection::ShortBasis => (Side::Buy, Side::Sell),
            BasisDirection::LongBasis => (Side::Sell, Side::Buy),
        };

        vec![
            BasisIntent {
                symbol: symbol.to_string(),
                venue: Venue::Spot,
                side: spot_side,
                qty: spot_qty,
                direction,
            },
            BasisIntent {
                symbol: symbol.to_string(),
                venue: Venue::Perp,
                side: perp_side,
                qty: perp_qty,
                direction,
            },
        ]
    }

    /// Create dollar-neutral exit pair (reverses the entry direction).
    ///
    /// ShortBasis exit: sell spot + buy perp
    /// LongBasis exit:  buy spot + sell perp
    pub fn exit_pair(
        symbol: &str,
        spot_qty: f64,
        spot_price: f64,
        perp_price: f64,
        direction: BasisDirection,
    ) -> Vec<Self> {
        let perp_qty = if perp_price > 0.0 {
            spot_qty * spot_price / perp_price
        } else {
            spot_qty
        };

        let (spot_side, perp_side) = match direction {
            BasisDirection::ShortBasis => (Side::Sell, Side::Buy),
            BasisDirection::LongBasis => (Side::Buy, Side::Sell),
        };

        vec![
            BasisIntent {
                symbol: symbol.to_string(),
                venue: Venue::Spot,
                side: spot_side,
                qty: spot_qty,
                direction,
            },
            BasisIntent {
                symbol: symbol.to_string(),
                venue: Venue::Perp,
                side: perp_side,
                qty: perp_qty,
                direction,
            },
        ]
    }
}

impl InstrumentIdentity for BasisIntent {
    type Key = u32;

    fn instrument_key(&self) -> u32 {
        let is_perp = self.venue == Venue::Perp;
        leg_token(&self.symbol, is_perp)
    }
}
