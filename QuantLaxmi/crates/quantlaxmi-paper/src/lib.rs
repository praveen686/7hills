//! quantlaxmi-paper
//!
//! Venue-agnostic paper trading spine.
//!
//! This crate is intentionally minimal: it defines the core traits and the
//! orchestration engine that runners (India/Crypto) can adapt to their feeds,
//! fee models, and strategies.

pub mod engine;
pub mod feed;
pub mod fill_model;
pub mod identity;
pub mod ledger;
pub mod state;
pub mod strategy;

pub use engine::{EngineConfig, PaperEngine};
pub use feed::{MarketEvent, MarketFeed};
pub use fill_model::{Fees, Fill, FillRejection, FillSide, PaperFillModel, TopOfBookProvider};
pub use identity::InstrumentIdentity;
pub use ledger::{FeesAggregate, FillLogEntry, Ledger, LedgerSummary, Position};
pub use state::{OptionLegView, PaperState, PositionView, StrategyView};
pub use strategy::{DecisionMetrics, DecisionType, FillOutcome, Strategy, StrategyDecision};
