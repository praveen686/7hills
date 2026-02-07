//! # QuantLaxmi India
//!
//! Indian F&O trading application for NSE/BSE markets via Zerodha.
//!
//! ## Isolation Guarantee
//! This binary has NO dependency on Binance or any crypto-related code.
//! It exclusively targets Indian equity derivatives markets.

fn main() -> anyhow::Result<()> {
    quantlaxmi_runner_india::run()
}
