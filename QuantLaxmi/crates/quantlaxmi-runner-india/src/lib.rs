//! # QuantLaxmi Runner India
//!
//! India-specific trading runner for NSE/BSE via Zerodha.
//!
//! ## Isolation Guarantee
//! This crate has NO dependency on Binance or any crypto-related code:
//! - NO Binance imports
//! - NO SBE imports
//! - NO crypto exchange info
//!
//! ## Commands
//! - `discover-zerodha` - Discover ATM options for NIFTY/BANKNIFTY
//! - `capture-zerodha` - Capture market data from Zerodha
//! - `backtest-kitesim` - Run KiteSim backtest
//! - `paper` - Run paper trading
//! - `live` - Run live trading

pub mod kitesim_backtest;
pub mod session_capture;
pub mod zerodha_capture;

use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tracing::info;

pub use quantlaxmi_runner_common::{
    AppState, RunnerConfig, SymbolState,
    circuit_breakers::TradingCircuitBreakers,
    create_runtime, init_observability, report, tui,
    web_server::{ServerState, start_server},
};

use kubera_core::ExecutionMode;
use quantlaxmi_connectors_zerodha::{AutoDiscoveryConfig, ZerodhaAutoDiscovery};

#[derive(Parser, Debug)]
#[command(name = "quantlaxmi-india")]
#[command(about = "QuantLaxmi India - NSE/BSE F&O Trading via Zerodha")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Discover BANKNIFTY/NIFTY option symbols for nearest expiry
    DiscoverZerodha {
        /// Underlying: NIFTY, BANKNIFTY, FINNIFTY
        #[arg(long, default_value = "BANKNIFTY")]
        underlying: String,

        /// Strikes around ATM (e.g., 2 = ATM ± 2 strikes = 5 strikes * 2 option types = 10 symbols)
        #[arg(long, default_value_t = 2)]
        strikes: u32,
    },

    /// Capture Zerodha quotes from Kite WebSocket into QuoteEvent JSONL
    CaptureZerodha {
        /// Comma-separated symbols, e.g. BANKNIFTY26JAN48000CE,BANKNIFTY26JAN48000PE
        #[arg(long)]
        symbols: String,

        /// Output path, e.g. data/replay/BANKNIFTY/2026-01-22/quotes.jsonl
        #[arg(long)]
        out: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,
    },

    /// Capture multi-instrument session with TickEvent JSONL (mantissa-based pricing)
    CaptureSession {
        /// Comma-separated instruments, e.g. BANKNIFTY26JAN48000CE,BANKNIFTY26JAN48000PE
        /// If not provided, use --underlying and --strikes for auto-discovery
        #[arg(long)]
        instruments: Option<String>,

        /// Underlying(s) for auto-discovery: NIFTY, BANKNIFTY, FINNIFTY (comma-separated)
        /// Use with --strikes to auto-discover ATM options
        #[arg(long, value_delimiter = ',')]
        underlying: Option<Vec<String>>,

        /// Strikes around ATM for auto-discovery (e.g., 5 = ATM ± 5 strikes = 11 strikes * 2 = 22 symbols)
        #[arg(long, default_value_t = 5)]
        strikes: u32,

        /// Output directory, e.g. data/sessions/banknifty_20260122
        #[arg(long)]
        out_dir: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 300)]
        duration_secs: u64,

        /// Price exponent (Kite uses -2, meaning divide by 100)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,
    },

    /// Offline KiteSim backtest runner (India/Zerodha only)
    BacktestKitesim {
        /// Fixed-point quantity scale
        #[arg(long, default_value_t = 1)]
        qty_scale: u32,

        /// Strategy label (for report metadata)
        #[arg(long, default_value = "")]
        strategy: String,

        /// Path to replay quotes (JSONL of QuoteEvent)
        #[arg(long)]
        replay: String,

        /// Path to orders JSON
        #[arg(long)]
        orders: String,

        /// Path to intents JSON (scheduled timestamps)
        #[arg(long)]
        intents: Option<String>,

        /// Path to depth replay file (DepthEvent JSONL)
        #[arg(long)]
        depth: Option<String>,

        /// Output directory for report.json
        #[arg(long, default_value = "artifacts/kitesim")]
        out: String,

        /// Atomic execution timeout (milliseconds)
        #[arg(long, default_value_t = 5000)]
        timeout_ms: i64,

        /// Simulated placement latency (milliseconds)
        #[arg(long, default_value_t = 150)]
        latency_ms: i64,

        /// Taker slippage (bps)
        #[arg(long, default_value_t = 0.0)]
        slippage_bps: f64,

        /// Adverse selection penalty cap (bps)
        #[arg(long, default_value_t = 0.0)]
        adverse_bps: f64,

        /// Reject if last quote older than this (milliseconds)
        #[arg(long, default_value_t = 10000)]
        stale_quote_ms: i64,

        /// Hedge on failure (rollback neutralization)
        #[arg(long, default_value_t = true)]
        hedge_on_failure: bool,
    },

    /// Run paper trading mode
    Paper {
        /// Path to configuration file
        #[arg(short, long, default_value = "configs/paper.toml")]
        config: String,

        /// Run in headless mode (no TUI)
        #[arg(long, default_value_t = false)]
        headless: bool,

        /// Initial capital
        #[arg(long, default_value_t = 1000000.0)]
        initial_capital: f64,
    },

    /// Run live trading mode
    Live {
        /// Path to configuration file
        #[arg(short, long, default_value = "configs/live.toml")]
        config: String,

        /// Initial capital
        #[arg(long, default_value_t = 1000000.0)]
        initial_capital: f64,
    },
}

/// Main entry point for the India runner
pub fn run() -> anyhow::Result<()> {
    let rt = create_runtime()?;
    rt.block_on(async_main())
}

async fn async_main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize observability
    init_observability("quantlaxmi-india");

    match cli.command {
        Commands::DiscoverZerodha {
            underlying,
            strikes,
        } => run_discover_zerodha(&underlying, strikes).await,
        Commands::CaptureZerodha {
            symbols,
            out,
            duration_secs,
        } => run_capture_zerodha(&symbols, &out, duration_secs).await,
        Commands::CaptureSession {
            instruments,
            underlying,
            strikes,
            out_dir,
            duration_secs,
            price_exponent,
        } => run_capture_session(instruments.as_deref(), underlying.as_deref(), strikes, &out_dir, duration_secs, price_exponent).await,
        Commands::BacktestKitesim {
            qty_scale,
            strategy,
            replay,
            orders,
            intents,
            depth,
            out,
            timeout_ms,
            latency_ms,
            slippage_bps,
            adverse_bps,
            stale_quote_ms,
            hedge_on_failure,
        } => {
            kitesim_backtest::run_kitesim_backtest_cli(kitesim_backtest::KiteSimCliConfig {
                qty_scale,
                strategy_name: strategy,
                replay_path: replay,
                orders_path: orders,
                intents_path: intents,
                depth_path: depth,
                out_dir: out,
                timeout_ms,
                latency_ms,
                slippage_bps,
                adverse_bps,
                stale_quote_ms,
                hedge_on_failure,
            })
            .await
        }
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

async fn run_discover_zerodha(underlying: &str, strikes: u32) -> anyhow::Result<()> {
    println!(
        "Discovering {} options (ATM ± {} strikes)...",
        underlying, strikes
    );

    let discovery = ZerodhaAutoDiscovery::from_sidecar()?;
    let config = AutoDiscoveryConfig {
        underlying: underlying.to_uppercase(),
        strikes_around_atm: strikes,
        strike_interval: if underlying.to_uppercase() == "BANKNIFTY" {
            100.0
        } else {
            50.0
        },
        ..Default::default()
    };

    let symbols = discovery.discover_symbols(&config).await?;

    println!("\n✅ Found {} symbols for {}:\n", symbols.len(), underlying);
    let symbol_names: Vec<&str> = symbols.iter().map(|(s, _)| s.as_str()).collect();
    for (sym, token) in &symbols {
        println!("  {} (token: {})", sym, token);
    }

    println!("\n📋 Copy this for --symbols:");
    println!("{}", symbol_names.join(","));

    Ok(())
}

async fn run_capture_zerodha(symbols: &str, out: &str, duration_secs: u64) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let symbol_list: Vec<String> = symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if symbol_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No symbols provided. Use --symbols BANKNIFTY26JAN48000CE,BANKNIFTY26JAN48000PE"
        ));
    }

    println!(
        "Capturing Zerodha quotes for {} symbols for {} seconds...",
        symbol_list.len(),
        duration_secs
    );

    let stats =
        zerodha_capture::capture_zerodha_quotes(&symbol_list, out_path, duration_secs).await?;
    println!("Capture complete: {} ({})", out, stats);

    Ok(())
}

async fn run_capture_session(
    instruments: Option<&str>,
    underlyings: Option<&[String]>,
    strikes: u32,
    out_dir: &str,
    duration_secs: u64,
    price_exponent: i8,
) -> anyhow::Result<()> {
    // Determine instrument list: either from --instruments or auto-discovery via --underlying
    let instrument_list: Vec<String> = if let Some(instr) = instruments {
        instr
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else if let Some(underlying_list) = underlyings {
        // Auto-discover ATM options for all underlyings
        let mut all_symbols = Vec::new();
        let discovery = ZerodhaAutoDiscovery::from_sidecar()?;

        for underlying_sym in underlying_list {
            println!(
                "Auto-discovering {} options (ATM ± {} strikes)...",
                underlying_sym, strikes
            );

            let config = AutoDiscoveryConfig {
                underlying: underlying_sym.to_uppercase(),
                strikes_around_atm: strikes,
                strike_interval: if underlying_sym.to_uppercase() == "BANKNIFTY" {
                    100.0
                } else {
                    50.0
                },
                ..Default::default()
            };

            let symbols = discovery.discover_symbols(&config).await?;
            println!("✅ Discovered {} instruments for {}:", symbols.len(), underlying_sym);
            for (sym, token) in &symbols {
                println!("  {} (token: {})", sym, token);
            }

            all_symbols.extend(symbols.into_iter().map(|(sym, _)| sym));
        }

        all_symbols
    } else {
        return Err(anyhow::anyhow!(
            "Must provide either --instruments or --underlying for auto-discovery"
        ));
    };

    if instrument_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No instruments found. Check --instruments or --underlying + --strikes"
        ));
    }

    let config = session_capture::SessionCaptureConfig {
        instruments: instrument_list,
        out_dir: std::path::PathBuf::from(out_dir),
        duration_secs,
        price_exponent,
    };

    let stats = session_capture::capture_session(config).await?;

    println!("\n=== Session Summary ===");
    println!("Session ID: {}", stats.session_id);
    println!("Duration: {:.1}s", stats.duration_secs);
    println!("Total ticks: {}", stats.total_ticks);
    println!("Certified: {}", if stats.all_certified { "YES" } else { "NO" });

    Ok(())
}

async fn run_paper_mode(
    config_path: &str,
    headless: bool,
    initial_capital: f64,
) -> anyhow::Result<()> {
    info!("QuantLaxmi India - Paper Trading Mode");
    info!("Loading configuration from: {}", config_path);

    let config = RunnerConfig::load(config_path)?;

    let kill_switch = Arc::new(AtomicBool::new(false));
    let is_indian_fno = true; // Always true for India runner

    let app_state = Arc::new(std::sync::Mutex::new(AppState::new(
        config.mode.symbols.clone(),
        initial_capital,
        ExecutionMode::Paper,
        headless,
        kill_switch.clone(),
        is_indian_fno,
    )));

    if headless {
        tui::print_headless_banner("QuantLaxmi India Paper Trading", initial_capital);
    }

    // Start web server
    let (web_tx, _) = tokio::sync::broadcast::channel(100);
    let web_state = Arc::new(ServerState { tx: web_tx.clone() });
    tokio::spawn(start_server(web_state, 8080));

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
    info!("QuantLaxmi India - LIVE Trading Mode");
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
