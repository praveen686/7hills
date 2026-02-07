//! Phase 2: Paper trade via generic PaperEngine.
//!
//! Wires: scanner → feed → strategy → PaperEngine.
//! Background scanner re-scans every 5s and rotates symbols dynamically.
//!
//! Usage: cargo run -p funding-harvester --bin funding-harvester

use std::time::{Duration, Instant};

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use funding_harvester::feed::FundingArbFeed;
use funding_harvester::fill_model::BinanceFillModel;
use funding_harvester::scanner;
use funding_harvester::strategy::{FundingArbStrategy, StrategyConfig};
use quantlaxmi_paper::{EngineConfig, PaperEngine};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
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

    info!(capital = initial_capital, top_n = top_n, "Funding Harvester starting");

    // Phase 1: Scan for top opportunities (must exist on both spot AND perp)
    info!("Scanning funding rates...");
    let entries = scanner::fetch_premium_index().await?;
    let all_opps = scanner::rank_opportunities(&entries);

    info!("Fetching spot exchange info (filter perp-only symbols)...");
    let spot_symbols = scanner::fetch_spot_symbols().await?;
    let opps = scanner::filter_spot_available(&all_opps, &spot_symbols);
    info!(
        perp_total = all_opps.len(),
        spot_available = opps.len(),
        "Filtered to symbols with both spot + perp"
    );

    let symbols: Vec<String> = opps
        .iter()
        .take(top_n)
        .filter(|o| o.funding_rate >= 0.0)
        .map(|o| o.symbol.clone())
        .collect();

    if symbols.is_empty() {
        info!("No positive funding rates with spot pairs found. Exiting.");
        return Ok(());
    }

    info!(symbols = ?symbols, "Tracking {} symbols", symbols.len());

    // Set up symbol rotation channel
    let (symbol_tx, symbol_rx) = tokio::sync::watch::channel(symbols.clone());
    // Set up pinned symbols channel (strategy → feed: don't rotate out open positions)
    let (pinned_tx, pinned_rx) = tokio::sync::watch::channel(Vec::<String>::new());

    // Phase 2: Connect feed with dynamic rotation + pinned symbols
    let feed = FundingArbFeed::connect(symbols)
        .await?
        .with_symbol_updates(symbol_rx)
        .with_pinned_symbols(pinned_rx);

    // Phase 3: Create strategy
    let config = StrategyConfig {
        base_position_usd: (initial_capital / 5.0).min(5_000.0),
        ..Default::default()
    };
    let strategy = FundingArbStrategy::new(config, initial_capital).with_pinned_channel(pinned_tx);
    let fill_model = BinanceFillModel::default();

    // Phase 4: Run engine
    let engine_config = EngineConfig::new(initial_capital);
    let mut engine = PaperEngine::with_config(feed, strategy, fill_model, engine_config);

    // Spawn background scanner for symbol rotation (every 5s)
    tokio::spawn(async move {
        let mut cached_spot = spot_symbols;
        let mut spot_refresh = Instant::now();

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;

            // Refresh spot symbols every 5 minutes
            if spot_refresh.elapsed() >= Duration::from_secs(300) {
                if let Ok(new_spot) = scanner::fetch_spot_symbols().await {
                    cached_spot = new_spot;
                }
                spot_refresh = Instant::now();
            }

            // Refresh funding rates and pick top symbols
            if let Ok(entries) = scanner::fetch_premium_index().await {
                let all = scanner::rank_opportunities(&entries);
                let filtered = scanner::filter_spot_available(&all, &cached_spot);
                // Only rotate if a new above-threshold symbol appeared
                let entry_threshold = 0.00015; // must match strategy config
                let above_threshold: Vec<String> = filtered
                    .iter()
                    .filter(|o| o.funding_rate >= entry_threshold)
                    .map(|o| o.symbol.clone())
                    .collect();
                let current = symbol_tx.borrow().clone();
                let current_set: std::collections::HashSet<&String> =
                    current.iter().collect();
                let has_new_opportunity = above_threshold
                    .iter()
                    .any(|s| !current_set.contains(s));
                if has_new_opportunity {
                    let mut new_symbols = above_threshold.clone();
                    for opp in filtered.iter().take(top_n) {
                        if opp.funding_rate >= 0.0
                            && !new_symbols.contains(&opp.symbol)
                        {
                            new_symbols.push(opp.symbol.clone());
                        }
                    }
                    let _ = symbol_tx.send(new_symbols);
                }
            }
        }
    });

    info!("Engine running with dynamic symbol rotation. Press Ctrl+C to stop.");

    // Run until interrupted
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

    // Print summary
    let state = engine.state();
    println!("\n--- Funding Harvester Summary ---");
    println!("Capital:     ${:.2}", state.cash);
    println!("Equity:      ${:.2}", state.equity);
    println!("Realized:    ${:.2}", state.realized_pnl);
    println!("Unrealized:  ${:.2}", state.unrealized_pnl);
    println!("Fees paid:   ${:.2}", state.fees_paid);
    println!("Fills:       {}", state.fills);
    println!("Rejections:  {}", state.rejections);

    Ok(())
}
