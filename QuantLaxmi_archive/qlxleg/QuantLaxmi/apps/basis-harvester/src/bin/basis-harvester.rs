//! Headless paper engine for basis mean-reversion.
//!
//! Wires: scanner → feed → strategy → PaperEngine.
//! Background scanner re-scans every 5s and rotates symbols.
//!
//! Usage: cargo run -p basis-harvester --bin basis-harvester [INITIAL_CAPITAL] [TOP_N]

use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use basis_harvester::feed::BasisFeed;
use basis_harvester::fill_model::BinanceFillModel;
use basis_harvester::scanner;
use basis_harvester::strategy::{BasisMeanRevStrategy, StrategyConfig};
use quantlaxmi_paper::{EngineConfig, PaperEngine};

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let initial_capital: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000.0);

    let top_n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    info!(capital = initial_capital, top_n = top_n, "Basis Harvester starting");

    // Scan for liquid symbols with meaningful basis
    info!("Fetching 24h volumes...");
    let volumes = scanner::fetch_24h_volumes().await.unwrap_or_default();
    info!(symbols_with_volume = volumes.len(), "Volume data loaded");

    info!("Scanning basis...");
    let entries = scanner::fetch_premium_index().await?;
    let all_opps = scanner::rank_basis_opportunities(&entries, &volumes);

    info!("Fetching spot exchange info...");
    let spot_symbols = scanner::fetch_spot_symbols().await?;
    let opps = scanner::filter_spot_available(&all_opps, &spot_symbols);
    info!(
        liquid_perps = all_opps.len(),
        spot_available = opps.len(),
        "Filtered: vol>$2M, |basis|<=200bps, spot+perp"
    );

    let symbols: Vec<String> = opps.iter().take(top_n).map(|o| o.symbol.clone()).collect();

    if symbols.is_empty() {
        info!("No liquid basis pairs found. Exiting.");
        return Ok(());
    }

    info!(symbols = ?symbols, "Tracking {} symbols", symbols.len());

    // Set up channels
    let (symbol_tx, symbol_rx) = tokio::sync::watch::channel(symbols.clone());
    let (pinned_tx, pinned_rx) = tokio::sync::watch::channel(Vec::<String>::new());

    // Connect feed (2 WS streams)
    let feed = BasisFeed::connect(symbols)
        .await?
        .with_symbol_updates(symbol_rx)
        .with_pinned_symbols(pinned_rx);

    // Create strategy (window from env or default)
    let window: usize = std::env::var("WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let config = StrategyConfig {
        window_size: window,
        ..Default::default()
    };
    let strategy =
        BasisMeanRevStrategy::new(config, initial_capital).with_pinned_channel(pinned_tx);
    let fill_model = BinanceFillModel::default();

    // Run engine
    let engine_config = EngineConfig::new(initial_capital);
    let mut engine = PaperEngine::with_config(feed, strategy, fill_model, engine_config);

    // Background scanner for symbol rotation
    tokio::spawn(async move {
        let mut cached_spot = spot_symbols;
        let mut cached_volumes = volumes;
        let mut spot_refresh = Instant::now();
        let mut vol_refresh = Instant::now();

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;

            // Refresh spot symbols every 5 minutes
            if spot_refresh.elapsed() >= Duration::from_secs(300) {
                if let Ok(new_spot) = scanner::fetch_spot_symbols().await {
                    cached_spot = new_spot;
                }
                spot_refresh = Instant::now();
            }

            // Refresh volumes every 60s
            if vol_refresh.elapsed() >= Duration::from_secs(60) {
                if let Ok(v) = scanner::fetch_24h_volumes().await {
                    cached_volumes = v;
                }
                vol_refresh = Instant::now();
            }

            if let Ok(entries) = scanner::fetch_premium_index().await {
                let all = scanner::rank_basis_opportunities(&entries, &cached_volumes);
                let filtered = scanner::filter_spot_available(&all, &cached_spot);
                let new_symbols: Vec<String> =
                    filtered.iter().take(top_n).map(|o| o.symbol.clone()).collect();
                let current = symbol_tx.borrow().clone();
                let current_set: std::collections::HashSet<&String> = current.iter().collect();
                let has_new = new_symbols.iter().any(|s| !current_set.contains(s));
                if has_new {
                    let _ = symbol_tx.send(new_symbols);
                }
            }
        }
    });

    info!("Engine running (Maker fees, liquid universe). Press Ctrl+C to stop.");

    tokio::select! {
        result = engine.run() => {
            match result {
                Ok(()) => info!("Engine finished normally"),
                Err(e) => info!(error = %e, "Engine stopped"),
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Ctrl+C received, shutting down");
        }
    }

    let state = engine.state();
    println!("\n--- Basis Harvester Summary ---");
    println!("Capital:     ${:.2}", state.cash);
    println!("Equity:      ${:.2}", state.equity);
    println!("Realized:    ${:.2}", state.realized_pnl);
    println!("Unrealized:  ${:.2}", state.unrealized_pnl);
    println!("Fees paid:   ${:.2}", state.fees_paid);
    println!("Fills:       {}", state.fills);
    println!("Rejections:  {}", state.rejections);

    Ok(())
}
