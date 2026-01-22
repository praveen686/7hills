//! # QuantLaxmi Binance Connector
//!
//! Binance integration for crypto markets (spot & derivatives).
//!
//! ## Features
//! - SBE binary protocol decoder (zero-copy, ultra-low latency)
//! - WebSocket streaming for trades and L2 depth
//! - Update-ID sequencing for certified determinism
//! - REST API for order placement
//!
//! ## Isolation
//! This crate has NO dependency on Zerodha or any India-specific code.
//! It is exclusively for cryptocurrency markets.

pub mod binance;
pub mod sbe;

pub use binance::{BinanceConnector, ConnectorStats, DepthResponse, fetch_depth_snapshot, depth_to_snapshot};
pub use sbe::{SbeHeader, BinanceSbeDecoder, AggTrade, DepthUpdate, TradeEntry, SBE_HEADER_SIZE};
