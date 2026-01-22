//! # QuantLaxmi Runner Crypto
//!
//! Crypto-specific trading runner for Binance (and future exchanges).
//!
//! ## Isolation Guarantee
//! This crate has NO dependency on Zerodha or any India-specific code:
//! - NO Zerodha imports
//! - NO KiteSim imports
//! - NO NSE specs imports
//! - NO options ledger imports
//!
//! ## Commands
//! - `capture-binance` - Capture bookTicker stream
//! - `capture-sbe-depth` - Capture SBE depth stream (certified)
//! - `exchange-info` - Fetch exchange info for symbols
//! - `paper` - Run paper trading
//! - `live` - Run live trading

pub mod binance_capture;
pub mod binance_exchange_info;
pub mod binance_sbe_depth_capture;

use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tracing::{error, info};

pub use quantlaxmi_runner_common::{
    AppState, RunnerConfig, SymbolState,
    circuit_breakers::TradingCircuitBreakers,
    create_runtime, init_observability, report, tui,
    web_server::{ServerState, start_server},
};

use kubera_core::ExecutionMode;

#[derive(Parser, Debug)]
#[command(name = "quantlaxmi-crypto")]
#[command(about = "QuantLaxmi Crypto - Cryptocurrency Trading via Binance")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Capture Binance Spot bookTicker stream into QuoteEvent JSONL
    CaptureBinance {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/BINANCE/BTCUSDT/quotes.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,
    },

    /// Capture Binance SBE depth stream into DepthEvent JSONL (Phase-2A authoritative)
    CaptureSbeDepth {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_depth.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,
    },

    /// Fetch exchange info for symbols
    ExchangeInfo {
        /// Comma-separated symbols, e.g. BTCUSDT,ETHUSDT
        #[arg(long)]
        symbols: String,
    },

    /// Run paper trading mode
    Paper {
        /// Path to configuration file
        #[arg(short, long, default_value = "configs/paper_crypto.toml")]
        config: String,

        /// Run in headless mode (no TUI)
        #[arg(long, default_value_t = false)]
        headless: bool,

        /// Initial capital (USD)
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,
    },

    /// Run live trading mode
    Live {
        /// Path to configuration file
        #[arg(short, long, default_value = "configs/live.toml")]
        config: String,

        /// Initial capital (USD)
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,
    },
}

/// Main entry point for the Crypto runner
pub fn run() -> anyhow::Result<()> {
    let rt = create_runtime()?;
    rt.block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize observability
    init_observability("quantlaxmi-crypto");

    match cli.command {
        Commands::CaptureBinance {
            symbol,
            out,
            duration_secs,
        } => run_capture_binance(&symbol, &out, duration_secs).await,
        Commands::CaptureSbeDepth {
            symbol,
            out,
            duration_secs,
            price_exponent,
            qty_exponent,
        } => {
            run_capture_sbe_depth(&symbol, &out, duration_secs, price_exponent, qty_exponent).await
        }
        Commands::ExchangeInfo { symbols } => run_exchange_info(&symbols).await,
        Commands::Paper {
            config,
            headless,
            initial_capital,
        } => run_paper_mode(&config, headless, initial_capital).await,
        Commands::Live {
            config,
            initial_capital,
        } => run_live_mode(&config, initial_capital).await,
    }
}

async fn run_capture_binance(symbol: &str, out: &str, duration_secs: u64) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!(
        "Capturing Binance {} bookTicker for {} seconds...",
        symbol, duration_secs
    );
    binance_capture::capture_book_ticker_jsonl(symbol, out_path, duration_secs).await?;
    println!("Capture complete: {}", out);

    Ok(())
}

async fn run_capture_sbe_depth(
    symbol: &str,
    out: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    println!(
        "Capturing Binance SBE {} depth stream for {} seconds (price_exp={}, qty_exp={})...",
        symbol, duration_secs, price_exponent, qty_exponent
    );

    let stats = binance_sbe_depth_capture::capture_sbe_depth_jsonl(
        symbol,
        out_path,
        duration_secs,
        price_exponent,
        qty_exponent,
        &api_key,
    )
    .await?;

    println!("Capture complete: {} ({})", out, stats);
    Ok(())
}

async fn run_exchange_info(symbols: &str) -> anyhow::Result<()> {
    let symbol_list: std::collections::HashSet<String> = symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if symbol_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No symbols provided. Use --symbols BTCUSDT,ETHUSDT"
        ));
    }

    println!(
        "Fetching exchange info for {} symbols...",
        symbol_list.len()
    );

    match binance_exchange_info::fetch_spot_specs(&symbol_list) {
        Ok(specs) => {
            println!("\n=== Binance Exchange Info ===");
            for (sym, (tick_size, qty_scale)) in specs {
                println!(
                    "  {}: tick_size={}, qty_scale={}",
                    sym, tick_size, qty_scale
                );
            }
        }
        Err(e) => {
            error!("Failed to fetch exchange info: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

async fn run_paper_mode(
    config_path: &str,
    headless: bool,
    initial_capital: f64,
) -> anyhow::Result<()> {
    info!("QuantLaxmi Crypto - Paper Trading Mode");
    info!("Loading configuration from: {}", config_path);

    let config = RunnerConfig::load(config_path)?;

    let kill_switch = Arc::new(AtomicBool::new(false));
    let is_indian_fno = false; // Always false for Crypto runner

    let app_state = Arc::new(std::sync::Mutex::new(AppState::new(
        config.mode.symbols.clone(),
        initial_capital,
        ExecutionMode::Paper,
        headless,
        kill_switch.clone(),
        is_indian_fno,
    )));

    if headless {
        tui::print_headless_banner("QuantLaxmi Crypto Paper Trading", initial_capital);
    }

    // Start web server
    let (web_tx, _) = tokio::sync::broadcast::channel(100);
    let web_state = Arc::new(ServerState { tx: web_tx.clone() });
    tokio::spawn(start_server(web_state, 8081)); // Different port from India

    info!("Paper trading mode initialized");
    info!("Symbols: {:?}", config.mode.symbols);
    info!("Initial capital: ${:.2}", initial_capital);
    info!("Press Ctrl+C to stop");

    tokio::signal::ctrl_c().await?;

    // Log final state before shutdown
    let state = app_state.lock().unwrap();
    info!("Final equity: ${:.2}", state.equity);
    info!("Shutting down...");

    Ok(())
}

async fn run_live_mode(config_path: &str, initial_capital: f64) -> anyhow::Result<()> {
    info!("QuantLaxmi Crypto - LIVE Trading Mode");
    info!("Loading configuration from: {}", config_path);

    let config = RunnerConfig::load(config_path)?;

    info!("Live trading mode initialized");
    info!("Symbols: {:?}", config.mode.symbols);
    info!("Initial capital: ${:.2}", initial_capital);
    info!("⚠️  LIVE MODE - Real orders will be placed!");
    info!("Press Ctrl+C to stop");

    tokio::signal::ctrl_c().await?;
    info!("Shutting down...");

    Ok(())
}
