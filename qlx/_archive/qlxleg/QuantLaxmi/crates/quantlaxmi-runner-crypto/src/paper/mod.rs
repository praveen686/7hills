//! Paper trading runner wiring + telemetry snapshot for the crypto binary/TUI.
//!
//! Design goals:
//! - Minimal surface area to integrate with "real" strategy/admission logic later.
//! - One shared UiSnapshot that the TUI can render without knowing internals.
//! - Deterministic, auditable values (no "fabrication"): unknown stays unknown.

pub mod admission_bridge;
pub mod decision_log;
pub mod engine;
pub mod intent;
pub mod position_manager;
pub mod slrt;
pub mod sniper;
pub mod state;
pub mod telemetry;
pub mod trade_log;

pub use admission_bridge::*;
pub use decision_log::*;
pub use engine::*;
pub use intent::*;
pub use position_manager::*;
pub use slrt::*;
pub use sniper::*;
pub use state::*;
pub use telemetry::*;
pub use trade_log::*;
