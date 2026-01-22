//! # QuantLaxmi Crypto
//!
//! Cryptocurrency trading application for Binance (and future exchanges).
//!
//! ## Isolation Guarantee
//! This binary has NO dependency on Zerodha or any India-specific code.
//! It exclusively targets cryptocurrency markets.

fn main() -> anyhow::Result<()> {
    // Load .env file for API keys (BINANCE_API_KEY_ED25519, etc.)
    dotenv::dotenv().ok();

    quantlaxmi_runner_crypto::run()
}
