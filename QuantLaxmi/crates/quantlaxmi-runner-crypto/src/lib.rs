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
pub mod binance_trades_capture;
pub mod paper_trading;
pub mod session_capture;

use anyhow::Context;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use tracing::{error, info};

pub use quantlaxmi_runner_common::{
    AppState, RunnerConfig, SymbolState,
    artifact::{ArtifactBuilder, ArtifactFamily, FileHash, RunManifest, RunProfile},
    circuit_breakers::TradingCircuitBreakers,
    create_runtime, init_observability, report, tui,
    web_server::{ServerState, start_server},
};

// ExecutionMode used by paper_trading module

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

        /// Strict sequencing mode (default: true). Fails capture if sequence gaps detected.
        /// Gaps break deterministic replay, so certified captures must have strict=true.
        #[arg(long, default_value_t = true)]
        strict: bool,
    },

    /// Capture Binance SBE trades stream into TradeEvent JSONL (certified)
    CaptureTrades {
        /// Symbol, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output path, e.g. data/replay/binance/BTCUSDT_trades.jsonl
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

    /// Capture multi-symbol session (depth + trades) for arbitrage research
    CaptureSession {
        /// Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,BNBUSDT
        #[arg(long)]
        symbols: String,

        /// Output directory for session data
        #[arg(long)]
        out_dir: String,

        /// Capture duration in seconds
        #[arg(long, default_value_t = 7200)]
        duration_secs: u64,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,

        /// Include trades capture (in addition to depth)
        #[arg(long, default_value_t = true)]
        include_trades: bool,

        /// Strict sequencing mode (default: true). Fails capture if sequence gaps detected.
        #[arg(long, default_value_t = true)]
        strict: bool,
    },

    /// Fetch exchange info for symbols
    ExchangeInfo {
        /// Comma-separated symbols, e.g. BTCUSDT,ETHUSDT
        #[arg(long)]
        symbols: String,
    },

    /// Run paper trading mode with live capture + WAL + PaperVenue
    Paper {
        /// Symbol to trade, e.g. BTCUSDT
        #[arg(long)]
        symbol: String,

        /// Output directory for WAL and session data
        #[arg(long, default_value = "data/paper")]
        out_dir: String,

        /// Price exponent for scaled integers (e.g., -2 for 2 decimal places)
        #[arg(long, default_value_t = -2)]
        price_exponent: i8,

        /// Quantity exponent for scaled integers (e.g., -8 for 8 decimal places)
        #[arg(long, default_value_t = -8)]
        qty_exponent: i8,

        /// Initial capital (USD)
        #[arg(long, default_value_t = 10000.0)]
        initial_capital: f64,

        /// Run in headless mode (no TUI)
        #[arg(long, default_value_t = true)]
        headless: bool,
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
            strict,
        } => {
            run_capture_sbe_depth(
                &symbol,
                &out,
                duration_secs,
                price_exponent,
                qty_exponent,
                strict,
            )
            .await
        }
        Commands::CaptureTrades {
            symbol,
            out,
            duration_secs,
            price_exponent,
            qty_exponent,
        } => run_capture_trades(&symbol, &out, duration_secs, price_exponent, qty_exponent).await,
        Commands::CaptureSession {
            symbols,
            out_dir,
            duration_secs,
            price_exponent,
            qty_exponent,
            include_trades,
            strict,
        } => {
            run_capture_session(
                &symbols,
                &out_dir,
                duration_secs,
                price_exponent,
                qty_exponent,
                include_trades,
                strict,
            )
            .await
        }
        Commands::ExchangeInfo { symbols } => run_exchange_info(&symbols).await,
        Commands::Paper {
            symbol,
            out_dir,
            price_exponent,
            qty_exponent,
            initial_capital,
            headless,
        } => {
            run_paper_mode(
                &symbol,
                &out_dir,
                price_exponent,
                qty_exponent,
                initial_capital,
                headless,
            )
            .await
        }
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
    let stats = binance_capture::capture_book_ticker_jsonl(symbol, out_path, duration_secs).await?;
    println!("Capture complete: {} ({})", out, stats);

    // Emit RunManifest (Research profile - bookTicker is not the authoritative path)
    let manifest_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    emit_bookticker_manifest(manifest_dir, out_path, symbol, &stats)?;

    Ok(())
}

/// Emit RunManifest for bookTicker capture.
/// BookTicker uses Research profile (not Certified - SBE depth is authoritative).
fn emit_bookticker_manifest(
    manifest_dir: &std::path::Path,
    quotes_path: &std::path::Path,
    symbol: &str,
    stats: &binance_capture::CaptureStats,
) -> anyhow::Result<()> {
    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, RunProfile::Research);

    // Record input hash (the captured quotes file)
    manifest.inputs.quotes = Some(FileHash::from_file(quotes_path)?);

    // NOT certified - bookTicker is JSON WebSocket, not SBE
    manifest.determinism.certified = false;

    // Compute input hash
    manifest.compute_input_hash();

    // Add capture metadata
    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: chrono::Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol,
                "events_written": stats.events_written,
                "integrity_tier": "Uncertified",
                "source": "binance_bookticker_capture",
                "note": "Use capture-sbe-depth for certified replay"
            }),
        },
    );

    // Finish and write manifest
    manifest.finish();
    let manifest_path = manifest_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    println!("Manifest written: {:?}", manifest_path);
    println!("  Run ID: {}", manifest.run_id);
    println!(
        "  ⚠️  Profile: Research (not Certified - use capture-sbe-depth for deterministic replay)"
    );

    Ok(())
}

async fn run_capture_sbe_depth(
    symbol: &str,
    out: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
    strict: bool,
) -> anyhow::Result<()> {
    let out_path = std::path::Path::new(out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    println!(
        "Capturing Binance SBE {} depth stream for {} seconds (price_exp={}, qty_exp={}, strict={})...",
        symbol, duration_secs, price_exponent, qty_exponent, strict
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

    // In strict mode (default), fail if any sequence gaps detected
    // Gaps break deterministic replay, so certified captures must have no gaps
    if strict && stats.gaps_detected > 0 {
        // Clean up the output file since it's not usable for certified replay
        let _ = std::fs::remove_file(out_path);
        return Err(anyhow::anyhow!(
            "STRICT MODE FAILURE: {} sequence gaps detected. \
             Deterministic replay requires continuous sequencing. \
             Re-run capture or use --strict=false for research-only data.",
            stats.gaps_detected
        ));
    }

    // Always emit RunManifest for audit-grade reproducibility
    let manifest_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    emit_capture_manifest(
        manifest_dir,
        out_path,
        symbol,
        &stats,
        price_exponent,
        qty_exponent,
        strict,
    )?;

    Ok(())
}

/// Emit RunManifest for a capture operation.
/// SBE depth captures with strict=true are Certified profile.
fn emit_capture_manifest(
    manifest_dir: &std::path::Path,
    depth_events_path: &std::path::Path,
    symbol: &str,
    stats: &binance_sbe_depth_capture::CaptureStats,
    price_exponent: i8,
    qty_exponent: i8,
    strict: bool,
) -> anyhow::Result<()> {
    // Profile depends on strict mode: Certified if strict, Research otherwise
    let profile = if strict {
        RunProfile::Certified
    } else {
        RunProfile::Research
    };
    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, profile);

    // Record input hash (the captured depth events file)
    manifest.inputs.depth_events = Some(FileHash::from_file(depth_events_path)?);

    // Mark as certified only if strict mode (no gaps allowed)
    manifest.determinism.certified = strict;

    // Compute input hash for reproducibility tracking
    manifest.compute_input_hash();

    // Add capture metadata to diagnostics
    manifest.diagnostics.context_validation.validation_passed = stats.snapshot_written;
    manifest.diagnostics.context_validation.validation_errors = if stats.gaps_detected > 0 {
        vec![format!(
            "Sequence gaps detected: {} (replay may fail)",
            stats.gaps_detected
        )]
    } else {
        vec![]
    };

    // Add custom metadata as JSON in regime_transitions (repurposed for capture metadata)
    let integrity_tier = if strict { "Certified" } else { "Research" };
    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: chrono::Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol,
                "events_written": stats.events_written,
                "snapshot_written": stats.snapshot_written,
                "gaps_detected": stats.gaps_detected,
                "price_exponent": price_exponent,
                "qty_exponent": qty_exponent,
                "strict_mode": strict,
                "integrity_tier": integrity_tier,
                "source": "binance_sbe_depth_capture"
            }),
        },
    );

    // Finish and write manifest
    manifest.finish();
    let manifest_path = manifest_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    println!("Manifest written: {:?}", manifest_path);
    println!("  Run ID: {}", manifest.run_id);
    println!("  Watermark: {}", manifest.watermark);
    if let Some(ref hash) = manifest.determinism.input_hash {
        println!("  Input hash: {}", &hash[..16]);
    }

    Ok(())
}

async fn run_capture_trades(
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
        "Capturing Binance SBE {} trades stream for {} seconds (price_exp={}, qty_exp={})...",
        symbol, duration_secs, price_exponent, qty_exponent
    );

    let stats = binance_trades_capture::capture_sbe_trades_jsonl(
        symbol,
        out_path,
        duration_secs,
        price_exponent,
        qty_exponent,
        &api_key,
    )
    .await?;

    println!("Capture complete: {} ({})", out, stats);

    // Emit manifest for trades capture
    let manifest_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    emit_trades_manifest(
        manifest_dir,
        out_path,
        symbol,
        &stats,
        price_exponent,
        qty_exponent,
    )?;

    Ok(())
}

/// Emit RunManifest for trades capture.
fn emit_trades_manifest(
    manifest_dir: &std::path::Path,
    trades_path: &std::path::Path,
    symbol: &str,
    stats: &binance_trades_capture::TradesCaptureStats,
    price_exponent: i8,
    qty_exponent: i8,
) -> anyhow::Result<()> {
    let mut manifest = RunManifest::new(ArtifactFamily::Crypto, RunProfile::Certified);

    // Record trades file hash (using quotes field as generic input)
    manifest.inputs.quotes = Some(FileHash::from_file(trades_path)?);

    manifest.determinism.certified = true;
    manifest.compute_input_hash();

    manifest.diagnostics.regime_transitions.push(
        quantlaxmi_runner_common::artifact::RegimeTransition {
            timestamp: chrono::Utc::now(),
            previous_regime: "capture_start".to_string(),
            new_regime: "capture_complete".to_string(),
            confidence: 1.0,
            features: serde_json::json!({
                "symbol": symbol,
                "trades_written": stats.trades_written,
                "buy_count": stats.buy_count,
                "sell_count": stats.sell_count,
                "total_volume_mantissa": stats.total_volume_mantissa,
                "price_exponent": price_exponent,
                "qty_exponent": qty_exponent,
                "integrity_tier": "Certified",
                "source": "binance_sbe_trades_capture"
            }),
        },
    );

    manifest.finish();
    let manifest_path = manifest_dir.join("trades_manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, json)?;

    println!("Trades manifest written: {:?}", manifest_path);
    println!("  Run ID: {}", manifest.run_id);

    Ok(())
}

async fn run_capture_session(
    symbols: &str,
    out_dir: &str,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
    include_trades: bool,
    strict: bool,
) -> anyhow::Result<()> {
    // Parse symbols
    let symbol_list: Vec<String> = symbols
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect();

    if symbol_list.is_empty() {
        return Err(anyhow::anyhow!(
            "No symbols provided. Use --symbols BTCUSDT,ETHUSDT,BNBUSDT"
        ));
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    println!("=== Session Capture ===");
    println!("Symbols: {:?}", symbol_list);
    println!("Duration: {} seconds", duration_secs);
    println!("Include trades: {}", include_trades);
    println!("Strict mode: {}", strict);
    println!("Output: {}", out_dir);

    let config = session_capture::SessionCaptureConfig {
        symbols: symbol_list,
        out_dir: std::path::PathBuf::from(out_dir),
        duration_secs,
        price_exponent,
        qty_exponent,
        include_trades,
        strict,
        api_key,
    };

    let stats = session_capture::capture_session(config).await?;

    println!("\n=== Session Summary ===");
    println!("Session ID: {}", stats.session_id);
    println!("Duration: {:.1}s", stats.duration_secs);
    println!("Total depth events: {}", stats.total_depth_events);
    println!("Total trades: {}", stats.total_trades);
    println!("Total gaps: {}", stats.total_gaps);
    println!(
        "Certified: {}",
        if stats.all_symbols_clean { "YES" } else { "NO" }
    );

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
    symbol: &str,
    out_dir: &str,
    price_exponent: i8,
    qty_exponent: i8,
    initial_capital: f64,
    headless: bool,
) -> anyhow::Result<()> {
    use futures_util::{SinkExt, StreamExt};
    use kubera_models::IntegrityTier;
    use kubera_options::replay::DepthEvent;
    use kubera_sbe::{BinanceSbeDecoder, SBE_HEADER_SIZE, SbeHeader};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message;
    use url::Url;

    info!("QuantLaxmi Crypto - Paper Trading Mode");
    info!("Symbol: {}", symbol);
    info!("Initial capital: ${:.2}", initial_capital);

    if headless {
        tui::print_headless_banner("QuantLaxmi Crypto Paper Trading", initial_capital);
    }

    // Requires BINANCE_API_KEY_ED25519 env var for SBE stream access
    let api_key = std::env::var("BINANCE_API_KEY_ED25519")
        .map_err(|_| anyhow::anyhow!("BINANCE_API_KEY_ED25519 env var required for SBE stream"))?;

    // Create paper session with WAL
    let out_path = std::path::Path::new(out_dir);
    let mut session =
        paper_trading::PaperSession::new(out_path, symbol, price_exponent, qty_exponent).await?;

    info!("Paper session created: {:?}", session.wal.session_dir());

    // Connect to SBE stream
    let sym_lower = symbol.to_lowercase();
    let url_str = "wss://stream-sbe.binance.com:9443/stream";
    info!("Connecting to Binance SBE stream: {}", url_str);

    let url = Url::parse(url_str)?;
    let mut request = url.into_client_request()?;
    request
        .headers_mut()
        .insert("X-MBX-APIKEY", api_key.parse()?);
    request
        .headers_mut()
        .insert("Sec-WebSocket-Protocol", "binance-sbe".parse()?);

    let (ws_stream, _) = tokio_tungstenite::connect_async(request)
        .await
        .context("connect SBE websocket")?;

    info!("Connected to SBE stream");

    let (mut write, mut read) = ws_stream.split();

    // Subscribe to depth stream
    let sub = serde_json::json!({
        "method": "SUBSCRIBE",
        "params": [format!("{}@depth", sym_lower)],
        "id": 1
    });
    write.send(Message::Text(sub.to_string())).await?;

    // Fetch initial snapshot
    info!("Fetching depth snapshot...");
    let snapshot_url = format!(
        "https://api.binance.com/api/v3/depth?symbol={}&limit=1000",
        symbol.to_uppercase()
    );
    let snapshot_resp: serde_json::Value = reqwest::get(&snapshot_url).await?.json().await?;
    let snapshot_last_id = snapshot_resp["lastUpdateId"].as_u64().unwrap_or(0);
    info!("Snapshot lastUpdateId: {}", snapshot_last_id);

    // Create snapshot event
    let snapshot_bids: Vec<kubera_options::replay::DepthLevel> = snapshot_resp["bids"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|b| {
            let price_str = b[0].as_str()?;
            let qty_str = b[1].as_str()?;
            let price =
                binance_sbe_depth_capture::parse_to_mantissa_from_str(price_str, price_exponent)
                    .ok()?;
            let qty = binance_sbe_depth_capture::parse_to_mantissa_from_str(qty_str, qty_exponent)
                .ok()?;
            Some(kubera_options::replay::DepthLevel { price, qty })
        })
        .collect();

    let snapshot_asks: Vec<kubera_options::replay::DepthLevel> = snapshot_resp["asks"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|a| {
            let price_str = a[0].as_str()?;
            let qty_str = a[1].as_str()?;
            let price =
                binance_sbe_depth_capture::parse_to_mantissa_from_str(price_str, price_exponent)
                    .ok()?;
            let qty = binance_sbe_depth_capture::parse_to_mantissa_from_str(qty_str, qty_exponent)
                .ok()?;
            Some(kubera_options::replay::DepthLevel { price, qty })
        })
        .collect();

    let snapshot_event = DepthEvent {
        ts: chrono::Utc::now(),
        tradingsymbol: symbol.to_uppercase(),
        first_update_id: snapshot_last_id,
        last_update_id: snapshot_last_id,
        price_exponent,
        qty_exponent,
        bids: snapshot_bids,
        asks: snapshot_asks,
        is_snapshot: true,
        integrity_tier: IntegrityTier::Certified,
        source: Some("paper_trading".to_string()),
    };

    // Process snapshot through session
    session.process_depth_event(snapshot_event).await?;
    info!("Snapshot applied to paper venue");

    // Start web server
    let (web_tx, _) = tokio::sync::broadcast::channel(100);
    let web_state = Arc::new(ServerState { tx: web_tx.clone() });
    tokio::spawn(start_server(web_state, 8081));

    info!("Paper trading active. Press Ctrl+C to stop.");

    let mut last_status_time = std::time::Instant::now();

    // Process messages until Ctrl+C
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Shutdown signal received");
                break;
            }
            msg = tokio::time::timeout(std::time::Duration::from_secs(30), read.next()) => {
                match msg {
                    Ok(Some(Ok(Message::Binary(bin)))) => {
                        if bin.len() >= SBE_HEADER_SIZE
                            && let Ok(header) = SbeHeader::decode(&bin[..SBE_HEADER_SIZE])
                            && header.template_id == 10003
                            && let Ok(update) = BinanceSbeDecoder::decode_depth_update(
                                &header,
                                &bin[SBE_HEADER_SIZE..],
                            )
                        {
                            let event = binance_sbe_depth_capture::sbe_depth_to_event_pub(
                                &update,
                                symbol,
                                price_exponent,
                                qty_exponent,
                            );
                            let fills = session.process_depth_event(event).await?;

                            for fill in fills {
                                info!("FILL: {} {} @ {:.2}",
                                    fill.side, fill.fill_qty, fill.fill_price);
                            }
                        }
                    }
                    Ok(Some(Ok(Message::Ping(p)))) => {
                        let _ = write.send(Message::Pong(p)).await;
                    }
                    Ok(Some(Ok(_))) => {}
                    Ok(Some(Err(e))) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    Ok(None) => {
                        info!("WebSocket closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - print status
                        info!("Waiting for market data...");
                    }
                }
            }
        }

        // Print status every 10 seconds
        if last_status_time.elapsed() > std::time::Duration::from_secs(10) {
            let summary = session.summary().await;
            info!(
                "Status: events={}, fills={}, position={:.4}, pnl=${:.2}, bid={:?}, ask={:?}",
                summary.events_processed,
                summary.fills_count,
                summary.position,
                summary.realized_pnl,
                summary.best_bid,
                summary.best_ask,
            );
            last_status_time = std::time::Instant::now();
        }
    }

    // Final flush and summary
    session.flush().await?;
    let summary = session.summary().await;

    info!("=== Paper Trading Session Complete ===");
    info!("  Events processed: {}", summary.events_processed);
    info!("  Fills: {}", summary.fills_count);
    info!("  Final position: {:.4}", summary.position);
    info!("  Realized PnL: ${:.2}", summary.realized_pnl);
    info!("  WAL saved to: {:?}", session.wal.session_dir());

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
