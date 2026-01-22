//! QuantLaxmi_India runner (India F&O / Zerodha family).
//!
//! Current implementation reuses the shared CLI. Next step is to restrict commands
//! and dependencies to India-family only (see docs/QUANTLAXMI_SPLIT_SPEC.md).

fn main() -> anyhow::Result<()> {
    kubera_runner::run_cli()
}
