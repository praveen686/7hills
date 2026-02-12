//! # QuantLaxmi Terminal Library
//!
//! Tauri IPC command and event wrappers for the BRAHMASTRA desktop trading terminal.
//! Wraps quantlaxmi-data, quantlaxmi-risk, and quantlaxmi-executor for frontend access.

pub mod market_hub;
pub mod exec_bridge;
pub mod config;

pub use market_hub::MarketDataHub;
pub use exec_bridge::ExecutionBridge;
pub use config::TerminalConfig;
