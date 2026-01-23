//! QuantLaxmi India Runner - CLI Entry Point
//!
//! This is the main binary for the India trading runner.
//!
//! ## Usage
//! ```bash
//! # Discover NIFTY options
//! cargo run -p quantlaxmi-runner-india -- discover-zerodha --underlying NIFTY --strikes 5
//!
//! # Capture session
//! cargo run -p quantlaxmi-runner-india -- capture-session \
//!     --instruments NIFTY26JAN25300CE,NIFTY26JAN25300PE \
//!     --out-dir data/sessions/nifty_nfo_20260123 \
//!     --duration-secs 300
//! ```

fn main() -> anyhow::Result<()> {
    quantlaxmi_runner_india::run()
}
