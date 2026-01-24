//! QuantLaxmi Crypto Runner - CLI entry point.
//!
//! Binary target for running crypto capture and trading commands.

fn main() -> anyhow::Result<()> {
    quantlaxmi_runner_crypto::run()
}
