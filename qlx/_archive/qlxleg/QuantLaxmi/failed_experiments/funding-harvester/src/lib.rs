//! Funding Harvester
//!
//! Live crypto funding rate arbitrage: BUY spot + SHORT perp = delta-neutral,
//! collect funding payments every 8h.
//!
//! Edge is structural (retail net long), typical yield 15-40% APY.

pub mod feed;
pub mod fill_model;
pub mod intent;
pub mod portfolio;
pub mod risk;
pub mod scanner;
pub mod snapshot;
pub mod state;
pub mod strategy;
