//! Paper trading engine module.
//!
//! Combines simulator, risk envelope, and fill sink for paper trading.

mod engine;
mod risk;
mod sink;

pub use engine::PaperEngine;
pub use sink::{FillSink, LoggingFillSink, NoopFillSink, VecFillSink};
