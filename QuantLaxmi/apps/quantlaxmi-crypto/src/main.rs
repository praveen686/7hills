//! # QuantLaxmi Crypto
//!
//! Cryptocurrency trading application for Binance (and future exchanges).
//!
//! ## Isolation Guarantee
//! This binary has NO dependency on Zerodha or any India-specific code.
//! It exclusively targets cryptocurrency markets.

fn main() -> anyhow::Result<()> {
    // Load .env file for API keys (BINANCE_API_KEY_ED25519, etc.)
    dotenvy::dotenv().ok();

    // Log API key presence (NEVER log the actual key)
    let api_key_present = std::env::var("BINANCE_API_KEY_ED25519").is_ok();
    eprintln!(
        "[quantlaxmi-crypto] BINANCE_API_KEY_ED25519 present: {}",
        api_key_present
    );

    quantlaxmi_runner_crypto::run()
}
