//! Binance-specific fill model for basis mean-reversion.
//!
//! Default mode is **Maker** (limit orders) — lower fees compensate for queue risk.
//! Round-trip cost: Maker+BNB = ~18 bps vs Taker+BNB = ~21 bps.
//!
//! Fee schedule:
//! - **Taker**: spot 0.1%, perp 0.04%
//! - **Maker**: spot 0.1%, perp 0.02%
//! - **BNB discount**: 25% off all fees (opt-in)

use chrono::{DateTime, Utc};
use quantlaxmi_paper::{Fees, Fill, FillRejection, FillSide, PaperFillModel};

use crate::intent::{BasisIntent, Side, Venue};
use crate::snapshot::BasisSnapshot;

// ---------------------------------------------------------------------------
// Fee schedule
// ---------------------------------------------------------------------------

const SPOT_TAKER_FEE: f64 = 0.001; // 0.10%
const SPOT_MAKER_FEE: f64 = 0.001; // 0.10%
const PERP_TAKER_FEE: f64 = 0.0004; // 0.04%
const PERP_MAKER_FEE: f64 = 0.0002; // 0.02%

/// BNB discount factor (25% off).
const BNB_DISCOUNT: f64 = 0.75;

/// Max quote age before rejection (ms). 5s for mid-cap feed latency.
const MAX_QUOTE_AGE_MS: i64 = 5_000;

// ---------------------------------------------------------------------------
// Fee mode configuration
// ---------------------------------------------------------------------------

/// Order execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeeMode {
    Taker,
    Maker,
}

/// Fill model configuration.
#[derive(Debug, Clone)]
pub struct FillModelConfig {
    pub fee_mode: FeeMode,
    pub bnb_discount: bool,
}

impl Default for FillModelConfig {
    fn default() -> Self {
        Self {
            fee_mode: FeeMode::Maker, // Lower fees: 18 bps RT vs 21 bps taker
            bnb_discount: true,
        }
    }
}

impl FillModelConfig {
    /// Compute the fee rate for a given venue.
    pub fn fee_rate(&self, venue: Venue) -> f64 {
        let base = match (venue, self.fee_mode) {
            (Venue::Spot, FeeMode::Taker) => SPOT_TAKER_FEE,
            (Venue::Spot, FeeMode::Maker) => SPOT_MAKER_FEE,
            (Venue::Perp, FeeMode::Taker) => PERP_TAKER_FEE,
            (Venue::Perp, FeeMode::Maker) => PERP_MAKER_FEE,
        };
        if self.bnb_discount { base * BNB_DISCOUNT } else { base }
    }

    /// Round-trip cost in bps (entry + exit on both legs).
    pub fn round_trip_cost_bps(&self) -> f64 {
        let spot_rt = self.fee_rate(Venue::Spot) * 2.0;
        let perp_rt = self.fee_rate(Venue::Perp) * 2.0;
        (spot_rt + perp_rt) * 10_000.0
    }
}

// ---------------------------------------------------------------------------
// Fill model
// ---------------------------------------------------------------------------

/// Fill model implementing Binance fee schedule. Default: Maker mode.
#[derive(Default)]
pub struct BinanceFillModel {
    pub config: FillModelConfig,
}

impl BinanceFillModel {
    pub fn new(config: FillModelConfig) -> Self {
        Self { config }
    }
}

impl PaperFillModel<BasisSnapshot, BasisIntent> for BinanceFillModel {
    fn try_fill(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &BasisSnapshot,
        intent: &BasisIntent,
    ) -> Result<Fill, FillRejection> {
        let state = snapshot.symbols.get(&intent.symbol).ok_or_else(|| {
            FillRejection::NoExecutableQuote {
                reason: format!("No data for {}", intent.symbol),
            }
        })?;

        let (bid, ask, quote_ts) = match intent.venue {
            Venue::Spot => (state.spot_bid, state.spot_ask, state.spot_ts),
            Venue::Perp => (state.perp_bid, state.perp_ask, state.perp_ts),
        };

        if bid <= 0.0 || ask <= 0.0 {
            return Err(FillRejection::NoExecutableQuote {
                reason: format!(
                    "{} {:?} bid={:.6} ask={:.6}",
                    intent.symbol, intent.venue, bid, ask
                ),
            });
        }

        if let Some(qt) = quote_ts {
            let age_ms = (ts - qt).num_milliseconds();
            if age_ms > MAX_QUOTE_AGE_MS {
                return Err(FillRejection::StaleQuote {
                    age_ms: age_ms as u32,
                    threshold_ms: MAX_QUOTE_AGE_MS as u32,
                });
            }
        }

        // Taker: buy at ask (cross spread), sell at bid
        // Maker: buy at bid (join queue), sell at ask
        let price = match (self.config.fee_mode, intent.side) {
            (FeeMode::Taker, Side::Buy) => ask,
            (FeeMode::Taker, Side::Sell) => bid,
            (FeeMode::Maker, Side::Buy) => bid,
            (FeeMode::Maker, Side::Sell) => ask,
        };

        let fee_rate = self.config.fee_rate(intent.venue);
        let notional = price * intent.qty;
        let fees = notional * fee_rate;

        let side = match intent.side {
            Side::Buy => FillSide::Buy,
            Side::Sell => FillSide::Sell,
        };

        Ok(Fill {
            ts,
            symbol: intent.symbol.clone(),
            side,
            qty: intent.qty,
            price,
            fees: Fees { total: fees },
        })
    }
}
