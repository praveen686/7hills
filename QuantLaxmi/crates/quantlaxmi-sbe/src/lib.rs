//! QuantLaxmi SBE - shim layer re-exporting kubera-sbe
//!
//! This crate provides the migration path from kubera-sbe to quantlaxmi-sbe.
//! All types are re-exported wholesale; no behavior changes.
//! Note: This is crypto-only (Binance SBE codec).

pub use kubera_sbe::*;
