//! QuantLaxmi_Crypto runner (Crypto / Binance family).
//!
//! Current implementation reuses the shared CLI. Next step is to restrict commands
//! and dependencies to Crypto-family only (see docs/QUANTLAXMI_SPLIT_SPEC.md).

fn main() -> anyhow::Result<()> {
    kubera_runner::run_cli()
}
