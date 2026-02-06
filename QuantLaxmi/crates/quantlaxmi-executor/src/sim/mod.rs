//! Deterministic simulator for paper trading and backtesting.
//!
//! **SINGLE SOURCE OF TRUTH** for simulated order execution.
//! Fixed-point refactor can replace the numeric types later without changing public APIs.

mod book;
mod engine;
mod ledger;
mod types;

pub use book::L2Book;
pub use engine::Simulator;
pub use ledger::{Ledger, Position};
pub use types::{Fill, FillType, Order, OrderType, Side, SimConfig};
