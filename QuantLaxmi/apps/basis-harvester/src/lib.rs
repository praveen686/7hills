//! Basis Harvester
//!
//! Crypto spot-perp basis mean-reversion: the basis `(perp_mid - spot_mid) / spot_mid`
//! oscillates around a rolling mean. Enter delta-neutral when z-score deviates beyond
//! threshold, exit on reversion.
//!
//! Edge is structural: basis oscillation from arbitrage/funding pressure generates
//! many trades/day in any market condition.

pub mod feed;
pub mod fill_model;
pub mod intent;
pub mod risk;
pub mod scanner;
pub mod snapshot;
pub mod state;
pub mod stats;
pub mod strategy;
