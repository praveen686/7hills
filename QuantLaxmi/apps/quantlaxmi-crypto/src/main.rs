//! # QuantLaxmi Crypto
//!
//! Cryptocurrency trading application for Binance (and future exchanges).
//!
//! ## Isolation Guarantee
//! This binary has NO dependency on Zerodha or any India-specific code.
//! It exclusively targets cryptocurrency markets.

fn main() -> anyhow::Result<()> {
    quantlaxmi_runner_crypto::run()
}
