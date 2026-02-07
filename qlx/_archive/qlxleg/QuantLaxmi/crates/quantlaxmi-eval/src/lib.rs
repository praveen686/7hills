//! QuantLaxmi Evaluation Module
//!
//! Phase 26: Strategy Truth Report
//!
//! This crate provides pure computation modules for evaluating strategy
//! performance from WAL-derived state. No I/O, no side effects, deterministic.
//!
//! ## Modules
//! - `strategy_aggregator` (Phase 26.1): Per-strategy metric accumulators
//! - `truth_report` (Phase 26.2): Report builder (JSON + text summary)

pub mod strategy_aggregator;
pub mod truth_report;

pub use strategy_aggregator::{AggregatorError, StrategyAccumulator, StrategyAggregatorRegistry};
pub use truth_report::{
    ReportPeriod, SessionMetadata, StrategyMetrics, StrategyTruthReport,
    TRUTH_REPORT_SCHEMA_VERSION,
};
