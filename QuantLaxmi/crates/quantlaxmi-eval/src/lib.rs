//! QuantLaxmi Evaluation Module
//!
//! Phase 26: Strategy Truth Report
//!
//! This crate provides pure computation modules for evaluating strategy
//! performance from WAL-derived state. No I/O, no side effects, deterministic.
//!
//! ## Modules
//! - `strategy_aggregator` (Phase 26.1): Per-strategy metric accumulators

pub mod strategy_aggregator;

pub use strategy_aggregator::{AggregatorError, StrategyAccumulator, StrategyAggregatorRegistry};
