//! # QuantLaxmi Zerodha Connector
//!
//! Zerodha Kite Connect integration for Indian markets (NSE/BSE F&O).
//!
//! ## Features
//! - WebSocket streaming with 184-byte full mode L2 parsing
//! - Auto-discovery of ATM options for NIFTY/BANKNIFTY
//! - Live order placement via Kite API
//! - TOTP + OAuth authentication via Python sidecar
//!
//! ## Isolation
//! This crate has NO dependency on Binance or any crypto-related code.
//! It is exclusively for Indian equity derivatives markets.

pub mod vendor_fields;
mod zerodha;

pub use zerodha::*;
