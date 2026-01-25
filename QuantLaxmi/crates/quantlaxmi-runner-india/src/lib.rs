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
pub mod sanos_io;
pub mod session_capture;
pub mod zerodha_capture;

use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tracing::{debug, info, warn};

pub use quantlaxmi_runner_common::{
    AppState, IntegritySummary, RunnerConfig, SessionManifest, SymbolState, TickOutputEntry,
    UnderlyingEntry,
    circuit_breakers::TradingCircuitBreakers,
    create_runtime, init_observability, persist_session_manifest_atomic, report, tui,
    web_server::{ServerState, start_server},
};

use quantlaxmi_connectors_zerodha::{AutoDiscoveryConfig, ZerodhaAutoDiscovery};
use quantlaxmi_core::ExecutionMode;

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

    /// [DEPRECATED] Legacy capture - use `capture-session` instead.
    ///
    /// WARNING: This command generates QuoteEvent with f64 prices and synthetic
    /// spreads when depth is unavailable. This corrupts research integrity.
    /// Use `capture-session` for audit-grade data with integrity_tier tracking.
    #[command(hide = true)] // Hide from --help, but still functional for migration
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

    /// [RECOMMENDED] Capture multi-instrument session with TickEvent JSONL.
    ///
    /// This is the canonical India capture command for research-grade data:
    /// - Mantissa-based pricing (no float drift)
    /// - IntegrityTier tracking (L2Present vs L1Only)
    /// - Multi-expiry auto-discovery (t1/t2/t3)
    /// - Session manifest for audit trail
    ///
    /// Synthetic quotes (L1Only) are labeled and rejected by default in scoring.
    CaptureSession {
        /// Comma-separated instruments, e.g. BANKNIFTY26JAN48000CE,BANKNIFTY26JAN48000PE
        /// If not provided, use --underlying and --strike-band for auto-discovery
        #[arg(long)]
        instruments: Option<String>,

        /// Underlying(s) for auto-discovery: NIFTY, BANKNIFTY, FINNIFTY (comma-separated)
        /// Use with --strike-band to auto-discover options
        #[arg(long, value_delimiter = ',')]
        underlying: Option<Vec<String>>,

        /// Strikes around ATM for auto-discovery (e.g., 20 = ATM ± 20 strikes = 41 strikes * 2 = 82 symbols per expiry)
        #[arg(long, default_value_t = 20)]
        strike_band: u32,

        /// Expiry policy: "t1" (nearest only), "t1t2t3" (nearest + next + front monthly)
        #[arg(long, default_value = "t1t2t3")]
        expiry_policy: String,

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

    // Initialize observability - guard must be held for process lifetime
    let _tracing_guards = init_observability("quantlaxmi-india");

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
            strike_band,
            expiry_policy,
            out_dir,
            duration_secs,
            price_exponent,
        } => {
            run_capture_session(
                instruments.as_deref(),
                underlying.as_deref(),
                strike_band,
                &expiry_policy,
                &out_dir,
                duration_secs,
                price_exponent,
            )
            .await
        }
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
    tracing::info!(
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

    tracing::info!("\n✅ Found {} symbols for {}:\n", symbols.len(), underlying);
    let symbol_names: Vec<&str> = symbols.iter().map(|(s, _)| s.as_str()).collect();
    for (sym, token) in &symbols {
        tracing::info!("  {} (token: {})", sym, token);
    }

    tracing::info!("\n📋 Copy this for --symbols:");
    tracing::info!("{}", symbol_names.join(","));

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

    tracing::info!(
        "Capturing Zerodha quotes for {} symbols for {} seconds...",
        symbol_list.len(),
        duration_secs
    );

    let stats =
        zerodha_capture::capture_zerodha_quotes(&symbol_list, out_path, duration_secs).await?;
    tracing::info!("Capture complete: {} ({})", out, stats);

    Ok(())
}

async fn run_capture_session(
    instruments: Option<&str>,
    underlyings: Option<&[String]>,
    strike_band: u32,
    expiry_policy: &str,
    out_dir: &str,
    duration_secs: u64,
    price_exponent: i8,
) -> anyhow::Result<()> {
    use quantlaxmi_connectors_zerodha::{ExpiryPolicy, MultiExpiryDiscoveryConfig};

    // Determine instrument list: either from --instruments or auto-discovery via --underlying
    // Commit B: Track both symbols and manifest tokens for subscription
    // Commit C: Also track underlying entries for aggregate session manifest
    let (instrument_list, manifest_tokens, underlying_entries) = if let Some(instr) = instruments {
        // Manual instrument list - no manifest tokens (legacy path)
        let symbols: Vec<String> = instr
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        (symbols, None, Vec::new())
    } else if let Some(underlying_list) = underlyings {
        // Auto-discover multi-expiry universe for all underlyings
        let mut all_symbols = Vec::new();
        let mut all_tokens: Vec<(String, u32)> = Vec::new();
        let mut all_underlying_entries: Vec<UnderlyingEntry> = Vec::new();
        let discovery = ZerodhaAutoDiscovery::from_sidecar()?;

        // Parse expiry policy
        let policy = match expiry_policy.to_lowercase().as_str() {
            "t1" => ExpiryPolicy::Count {
                expiry_count: 1,
                include_front_monthly: false,
            },
            "t1t2" => ExpiryPolicy::Count {
                expiry_count: 2,
                include_front_monthly: false,
            },
            _ => ExpiryPolicy::T1T2T3,
        };

        let today = chrono::Local::now().date_naive();

        for underlying_sym in underlying_list {
            tracing::info!(
                "Auto-discovering {} options (±{} strikes, policy={})...",
                underlying_sym, strike_band, expiry_policy
            );

            let config = MultiExpiryDiscoveryConfig {
                underlying: underlying_sym.to_uppercase(),
                strike_band,
                expiry_policy: policy.clone(),
                today,
                prefer_futures_for_atm: false,
                override_strike_step: None,
            };

            let manifest = discovery.discover_universe(&config).await?;

            tracing::info!("✅ {} Universe Manifest:", underlying_sym);
            tracing::info!("   Spot: {:.2}", manifest.spot);
            tracing::info!("   ATM: {:.0}", manifest.atm);
            tracing::info!("   Strike step: {:.0}", manifest.strike_step);
            tracing::info!(
                "   Expiries: T1={}, T2={:?}, T3={:?}",
                manifest.expiry_selection.t1,
                manifest.expiry_selection.t2,
                manifest.expiry_selection.t3
            );
            tracing::info!(
                "   Strikes: {} ({:.0} to {:.0})",
                manifest.target_strikes.len(),
                manifest.target_strikes.first().unwrap_or(&0.0),
                manifest.target_strikes.last().unwrap_or(&0.0)
            );
            tracing::info!("   Instruments: {}", manifest.instruments.len());

            // Log missing
            for (exp, miss) in &manifest.missing {
                if !miss.is_empty() {
                    tracing::info!("   Missing for {}: {} instruments", exp, miss.len());
                }
            }

            // Persist manifest with cryptographic integrity (Commit A: Phase 5.0)
            // Use per-underlying subdirectory to avoid overwrite when capturing multiple underlyings
            let underlying_dir = std::path::Path::new(out_dir).join(underlying_sym.to_lowercase());
            std::fs::create_dir_all(&underlying_dir)?;
            let persist_result = quantlaxmi_runner_common::manifest_io::persist_universe_manifest(
                &underlying_dir,
                &manifest,
            )?;

            // Pre-Commit B gates: validate manifest before subscription
            if manifest.instruments.is_empty() {
                return Err(anyhow::anyhow!(
                    "UniverseManifest for {} has no instruments - cannot proceed",
                    underlying_sym
                ));
            }

            // Validate token uniqueness
            let token_set: std::collections::HashSet<u32> = manifest
                .instruments
                .iter()
                .map(|i| i.instrument_token)
                .collect();
            if token_set.len() != manifest.instruments.len() {
                return Err(anyhow::anyhow!(
                    "UniverseManifest for {} has duplicate instrument tokens",
                    underlying_sym
                ));
            }

            info!(
                underlying_dir = %underlying_dir.display(),
                underlying = %underlying_sym,
                universe_manifest_sha256 = %persist_result.sha256,
                universe_manifest_path = %persist_result.manifest_path.display(),
                instrument_count = manifest.instruments.len(),
                t1_expiry = %manifest.expiry_selection.t1,
                t2_expiry = ?manifest.expiry_selection.t2,
                t3_expiry = ?manifest.expiry_selection.t3,
                strike_step = manifest.strike_step,
                bytes_len = persist_result.bytes_len,
                "UniverseManifest persisted"
            );

            tracing::info!("   Manifest saved: {:?}", persist_result.manifest_path);
            tracing::info!("   SHA-256: {}", persist_result.sha256);

            // Commit B: Collect tokens from manifest for subscription (use iter, not into_iter)
            for instr in &manifest.instruments {
                all_symbols.push(instr.tradingsymbol.clone());
                all_tokens.push((instr.tradingsymbol.clone(), instr.instrument_token));
            }

            // Commit C: Build UnderlyingEntry for aggregate session manifest
            let underlying_entry = UnderlyingEntry {
                underlying: underlying_sym.to_uppercase(),
                subdir: format!("{}/", underlying_sym.to_lowercase()),
                universe_manifest_path: format!(
                    "{}/universe_manifest.json",
                    underlying_sym.to_lowercase()
                ),
                universe_manifest_sha256: persist_result.sha256.clone(),
                instrument_count: manifest.instruments.len(),
                t1_expiry: manifest.expiry_selection.t1.to_string(),
                t2_expiry: manifest.expiry_selection.t2.map(|d| d.to_string()),
                t3_expiry: manifest.expiry_selection.t3.map(|d| d.to_string()),
                strike_step: manifest.strike_step,
            };
            all_underlying_entries.push(underlying_entry);

            debug!(
                underlying = %underlying_sym,
                token_count = all_tokens.len(),
                sample_tokens = ?all_tokens.iter().take(5).map(|(_, t)| t).collect::<Vec<_>>(),
                "Universe tokens prepared for subscription"
            );
        }

        (all_symbols, Some(all_tokens), all_underlying_entries)
    } else {
        return Err(anyhow::anyhow!(
            "Must provide either --instruments or --underlying for auto-discovery"
        ));
    };

    if instrument_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No instruments found. Check --instruments or --underlying + --strike-band"
        ));
    }

    tracing::info!(
        "\nStarting capture with {} instruments...",
        instrument_list.len()
    );

    let config = session_capture::SessionCaptureConfig {
        instruments: instrument_list,
        out_dir: std::path::PathBuf::from(out_dir),
        duration_secs,
        price_exponent,
        manifest_tokens,
    };

    let stats = session_capture::capture_session(config).await?;

    // Commit C: Build and persist aggregate session manifest
    if !underlying_entries.is_empty() {
        let mut session_manifest = SessionManifest::new(
            stats.session_id.clone(),
            "india_capture".to_string(),
            out_dir.to_string(),
            duration_secs as f64,
            price_exponent,
        );

        // Add underlying entries
        for entry in underlying_entries {
            session_manifest.add_underlying(entry);
        }

        // Add tick outputs from capture stats
        for (symbol, path, ticks_written, has_depth) in &stats.tick_outputs {
            session_manifest.add_tick_output(TickOutputEntry {
                symbol: symbol.clone(),
                path: path.clone(),
                ticks_written: *ticks_written,
                has_depth: *has_depth,
            });
        }

        // Set integrity summary
        session_manifest.set_integrity(IntegritySummary {
            out_of_universe_ticks_dropped: stats.out_of_universe_ticks_dropped,
            subscribe_mode: stats.subscribe_mode.clone(),
            notes: Vec::new(),
        });

        // Persist atomically
        let out_path = std::path::Path::new(out_dir);
        let persist_result = persist_session_manifest_atomic(out_path, &session_manifest)?;

        info!(
            session_manifest_path = %persist_result.manifest_path.display(),
            bytes_len = persist_result.bytes_len,
            underlying_count = session_manifest.underlyings.len(),
            tick_output_count = session_manifest.tick_outputs.len(),
            subscribe_mode = %stats.subscribe_mode,
            out_of_universe_ticks_dropped = stats.out_of_universe_ticks_dropped,
            "Aggregate session manifest persisted"
        );
        tracing::info!(
            "Session manifest saved: {:?} ({} bytes)",
            persist_result.manifest_path, persist_result.bytes_len
        );
    }

    tracing::info!("\n=== Session Summary ===");
    tracing::info!("Session ID: {}", stats.session_id);
    tracing::info!("Duration: {:.1}s", stats.duration_secs);
    tracing::info!("Total ticks: {}", stats.total_ticks);
    tracing::info!("Subscribe mode: {}", stats.subscribe_mode);
    if stats.out_of_universe_ticks_dropped > 0 {
        tracing::info!(
            "Out-of-universe ticks dropped: {}",
            stats.out_of_universe_ticks_dropped
        );
    }
    tracing::info!(
        "Certified: {}",
        if stats.all_certified { "YES" } else { "NO" }
    );

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
    // P3: Handle potential mutex poison gracefully on shutdown path
    match app_state.lock() {
        Ok(state) => info!("Final equity: ${:.2}", state.equity),
        Err(poisoned) => {
            warn!("Mutex was poisoned, recovering state for final log");
            let state = poisoned.into_inner();
            info!("Final equity: ${:.2}", state.equity);
        }
    }
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
