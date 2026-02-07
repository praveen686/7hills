//! Triangle Arbitrage Data Capture
//!
//! Captures depth data for BTC-ETH, BNB-BTC, and SOL-BTC triangles with
//! correct price exponents automatically fetched from Binance.
//!
//! Usage:
//!   cargo run --bin capture_triangles -- --duration 7200 --out-dir data/sessions/triangles_$(date +%Y%m%d_%H%M%S)

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use quantlaxmi_runner_crypto::binance_exchange_info::fetch_price_exponents;
use quantlaxmi_runner_crypto::session_capture::{capture_session, SessionCaptureConfig};

/// Triangle Arbitrage Data Capture
#[derive(Parser, Debug)]
#[command(name = "capture_triangles")]
#[command(about = "Capture depth data for triangle arbitrage analysis")]
struct Args {
    /// Duration in seconds (default: 2 hours)
    #[arg(long, default_value = "7200")]
    duration: u64,

    /// Output directory
    #[arg(long, default_value = "data/sessions/triangles")]
    out_dir: PathBuf,

    /// Include trades capture (default: true)
    #[arg(long, default_value = "true")]
    include_trades: bool,

    /// Strict mode - fail on gaps (default: true)
    #[arg(long, default_value = "true")]
    strict: bool,

    /// Binance API key (reads BINANCE_API_KEY env if not specified)
    #[arg(long, default_value = "")]
    api_key: String,
}

/// Triangle arbitrage symbols:
/// - USDT pairs: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT
/// - BTC cross pairs: ETHBTC, BNBBTC, SOLBTC
const TRIANGLE_SYMBOLS: &[&str] = &[
    // USDT pairs (tick_size = 0.01, exp = -2)
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    // BTC cross pairs (need finer precision)
    "ETHBTC",  // tick_size = 0.00001, exp = -5
    "BNBBTC",  // tick_size = 0.000001, exp = -6
    "SOLBTC",  // tick_size = 0.0000001, exp = -7
];

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file
    let _ = dotenvy::dotenv();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();

    // Get API key from arg or env (try both naming conventions)
    let api_key = if args.api_key.is_empty() {
        std::env::var("BINANCE_API_KEY_ED25519")
            .or_else(|_| std::env::var("BINANCE_API_KEY"))
            .unwrap_or_default()
    } else {
        args.api_key.clone()
    };

    // Generate timestamped output directory
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let out_dir = if args.out_dir.to_string_lossy().contains("triangles") {
        PathBuf::from(format!("{}/triangles_{}", args.out_dir.display(), timestamp))
    } else {
        args.out_dir.clone()
    };

    tracing::info!("=== Triangle Arbitrage Data Capture ===");
    tracing::info!("Symbols: {:?}", TRIANGLE_SYMBOLS);
    tracing::info!("Duration: {} seconds ({:.1} hours)", args.duration, args.duration as f64 / 3600.0);
    tracing::info!("Output: {:?}", out_dir);

    // Fetch correct price exponents from Binance
    let symbols: Vec<String> = TRIANGLE_SYMBOLS.iter().map(|s| s.to_string()).collect();

    tracing::info!("\nFetching price exponents from Binance...");
    let price_exponents = fetch_price_exponents(&symbols)?;

    tracing::info!("\nPrice exponents:");
    for sym in &symbols {
        if let Some(exp) = price_exponents.get(sym) {
            tracing::info!("  {} → {}", sym, exp);
        }
    }

    // Create capture config
    let config = SessionCaptureConfig {
        symbols,
        out_dir: out_dir.clone(),
        duration_secs: args.duration,
        price_exponent: -2, // Default (overridden by per-symbol)
        qty_exponent: -8,
        include_trades: args.include_trades,
        strict: args.strict,
        api_key,
        price_exponents: Some(price_exponents),
    };

    tracing::info!("\nStarting capture...");
    let stats = capture_session(config).await?;

    // Print summary
    tracing::info!("\n=== Capture Complete ===");
    tracing::info!("Session ID: {}", stats.session_id);
    tracing::info!("Duration: {:.1}s", stats.duration_secs);
    tracing::info!("Total depth events: {}", stats.total_depth_events);
    tracing::info!("Total trades: {}", stats.total_trades);
    tracing::info!("Total gaps: {}", stats.total_gaps);
    tracing::info!("All symbols clean: {}", stats.all_symbols_clean);
    tracing::info!("\nOutput directory: {:?}", out_dir);

    if stats.total_gaps > 0 {
        tracing::warn!("WARNING: Gaps detected - session may not be certified");
    }

    if stats.all_symbols_clean {
        tracing::info!("\n✓ Session is CERTIFIED - ready for triangle analysis");
    }

    Ok(())
}
