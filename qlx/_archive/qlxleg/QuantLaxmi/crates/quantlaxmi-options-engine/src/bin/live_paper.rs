//! Live Paper Trading with Zerodha WebSocket
//!
//! Connects to Zerodha Kite WebSocket for real-time data and runs
//! the Options Engine in paper trading mode.
//!
//! Usage:
//!   live-paper --symbols NIFTY-ATM          # Auto-discover ATM options
//!   live-paper --symbols NIFTY2620325300CE,NIFTY2620325300PE  # Explicit symbols
//!
//! Requirements:
//!   - scripts/zerodha_auth.py configured with API credentials
//!   - TOTP secret set in environment or config

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::Serialize;

use quantlaxmi_connectors_zerodha::{ZerodhaAutoDiscovery, ZerodhaConnector};
use quantlaxmi_core::{EventBus, MarketPayload};
use quantlaxmi_options_engine::{EngineConfig, OptionsEngine, StrategyType, TradingAction};
use quantlaxmi_regime::FeatureVector;

#[derive(Parser)]
#[command(name = "live-paper")]
#[command(about = "Live paper trading with Zerodha WebSocket")]
struct Args {
    /// Symbols to trade (comma-separated, or NIFTY-ATM for auto-discovery)
    #[arg(long, default_value = "NIFTY-ATM")]
    symbols: String,

    /// Duration in seconds (0 = run forever)
    #[arg(long, default_value = "0")]
    duration: u64,

    /// Initial capital (INR)
    #[arg(long, default_value = "1000000")]
    capital: f64,

    /// Max positions
    #[arg(long, default_value = "3")]
    max_positions: u32,

    /// Minimum strategy score to trade
    #[arg(long, default_value = "60")]
    min_score: f64,

    /// Output directory for logs
    #[arg(long, default_value = "/tmp/live_paper")]
    out: String,

    /// Stop loss per position (INR)
    #[arg(long, default_value = "5000")]
    stop_loss: f64,

    /// Profit target per position (INR)
    #[arg(long, default_value = "2500")]
    profit_target: f64,
}

/// Paper position
#[derive(Debug, Clone, Serialize)]
struct PaperPosition {
    symbol: String,
    strategy: StrategyType,
    entry_ts: DateTime<Utc>,
    entry_price: f64,
    quantity: i32,
    current_price: f64,
    unrealized_pnl: f64,
}

/// Paper trade record
#[derive(Debug, Clone, Serialize)]
struct PaperTrade {
    ts: DateTime<Utc>,
    symbol: String,
    strategy: StrategyType,
    action: String,
    price: f64,
    quantity: i32,
    pnl: f64,
    score: f64,
    regime: String,
    hft_detected: bool,
}

/// Session stats
struct SessionStats {
    capital: f64,
    initial_capital: f64,
    decisions: u32,
    entry_signals: u32,
    hft_blocks: u32,
    winning_trades: u32,
    losing_trades: u32,
    peak_equity: f64,
    max_drawdown: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     QuantLaxmi Options Engine - Live Paper Trading               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Create output directory
    std::fs::create_dir_all(&args.out)?;

    // Parse symbols
    let symbol_list: Vec<String> = args
        .symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Auto-discover symbols if needed
    println!("Authenticating with Zerodha...");
    let discovery =
        ZerodhaAutoDiscovery::from_sidecar().context("Failed to authenticate with Zerodha")?;

    println!("Resolving symbols...");
    let resolved_symbols = discovery.resolve_symbols(&symbol_list).await?;

    if resolved_symbols.is_empty() {
        anyhow::bail!("No symbols resolved. Check your symbol list.");
    }

    let symbols: Vec<String> = resolved_symbols.iter().map(|(s, _)| s.clone()).collect();
    let _tokens: Vec<u32> = resolved_symbols.iter().map(|(_, t)| *t).collect();

    println!("\nResolved {} symbols:", symbols.len());
    for (sym, tok) in &resolved_symbols {
        println!("  {} (token: {})", sym, tok);
    }
    println!();

    println!("Configuration:");
    println!("  Capital:       ₹{:.0}", args.capital);
    println!("  Max Positions: {}", args.max_positions);
    println!("  Min Score:     {}", args.min_score);
    println!("  Stop Loss:     ₹{:.0}", args.stop_loss);
    println!("  Profit Target: ₹{:.0}", args.profit_target);
    println!(
        "  Duration:      {}",
        if args.duration == 0 {
            "Unlimited".to_string()
        } else {
            format!("{} seconds", args.duration)
        }
    );
    println!();

    // Setup signal handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n\nReceived Ctrl+C, shutting down gracefully...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Create event bus for market data (capacity of 10000 events)
    let bus = EventBus::new(10000);

    // Create Zerodha connector
    let connector = ZerodhaConnector::new(bus.clone(), symbols.clone());

    // Initialize options engine
    let config = EngineConfig {
        symbol: "NIFTY".into(),
        lot_size: 50,
        risk_free_rate: 0.065,
        dividend_yield: 0.012,
        max_positions: args.max_positions,
        max_loss_per_position: args.stop_loss,
        max_portfolio_delta: 500.0,
        min_iv_percentile_sell: 60.0,
        max_iv_percentile_buy: 40.0,
        min_strategy_score: args.min_score,
        ramanujan_enabled: true,
        block_on_hft: true,
        pcr_enabled: true,
        pcr_lookback: 100,
    };

    let mut engine = OptionsEngine::new(config);

    // Session state
    let mut positions: HashMap<String, PaperPosition> = HashMap::new();
    let mut trades: Vec<PaperTrade> = Vec::new();
    let mut prev_prices: HashMap<String, f64> = HashMap::new();
    let mut stats = SessionStats {
        capital: args.capital,
        initial_capital: args.capital,
        decisions: 0,
        entry_signals: 0,
        hft_blocks: 0,
        winning_trades: 0,
        losing_trades: 0,
        peak_equity: args.capital,
        max_drawdown: 0.0,
    };

    // Open trade log
    let trade_log_path = format!("{}/trades.jsonl", args.out);
    let mut trade_log = std::fs::File::create(&trade_log_path)?;

    println!("Starting live paper trading session...");
    println!("Press Ctrl+C to stop\n");

    // Subscribe to market data
    let mut market_rx = bus.subscribe_market();

    // Start connector in background
    let connector_running = running.clone();
    let connector_handle = tokio::spawn(async move {
        use quantlaxmi_core::connector::MarketConnector;

        if let Err(e) = connector.run().await {
            if connector_running.load(Ordering::SeqCst) {
                eprintln!("Connector error: {}", e);
            }
        }
    });

    // Main processing loop
    let start_time = std::time::Instant::now();
    let duration = if args.duration == 0 {
        Duration::MAX
    } else {
        Duration::from_secs(args.duration)
    };

    while running.load(Ordering::SeqCst) {
        // Check duration
        if start_time.elapsed() > duration {
            println!("\nDuration limit reached.");
            break;
        }

        // Receive market data with timeout
        match tokio::time::timeout(Duration::from_secs(1), market_rx.recv()).await {
            Ok(Ok(record)) => {
                let symbol = record.symbol.clone();
                let ts = record.ts;

                // Extract price from payload - use Trade (LTP) only for now
                // Depth ticks have parsing issues with the bid/ask levels
                let price = match &record.payload {
                    MarketPayload::Trade { price_mantissa, .. } => {
                        *price_mantissa as f64 / 100.0 // Convert from paisa to rupees
                    }
                    MarketPayload::Depth { .. } => {
                        // Skip depth ticks - use LTP from Trade for pricing
                        continue;
                    }
                    MarketPayload::Quote {
                        bid_price_mantissa,
                        ask_price_mantissa,
                        ..
                    } => {
                        // Use mid price from quote
                        if *bid_price_mantissa > 0 && *ask_price_mantissa > 0 {
                            (*bid_price_mantissa as f64 + *ask_price_mantissa as f64) / 2.0 / 100.0
                        } else {
                            continue;
                        }
                    }
                };

                if price <= 0.0 {
                    continue;
                }

                // Calculate features
                let prev_price = prev_prices.get(&symbol).copied().unwrap_or(price);
                let mid_return = if prev_price > 0.0 {
                    (((price - prev_price) / prev_price) * 10000.0) as i64
                } else {
                    0
                };

                let features = FeatureVector::new(mid_return, 0, 50, 50, 50, 0);

                // Extract spot from symbol
                let spot = extract_strike_from_symbol(&symbol).unwrap_or(25000.0);
                engine.on_tick(ts, spot, &features);

                // Get decision
                let decision = engine.decide(ts);
                let status = engine.status();
                stats.decisions += 1;

                // Track stats
                let is_entry = matches!(decision.action, TradingAction::Enter);
                if is_entry {
                    stats.entry_signals += 1;
                }
                if status.hft_detected {
                    stats.hft_blocks += 1;
                }

                // Process decision
                if !status.hft_detected {
                    if let TradingAction::Enter = decision.action {
                        if let Some(rec) = &decision.strategy {
                            let has_position = positions.contains_key(&symbol);
                            if !has_position && positions.len() < args.max_positions as usize {
                                // Paper entry
                                let quantity = 50;

                                positions.insert(
                                    symbol.clone(),
                                    PaperPosition {
                                        symbol: symbol.clone(),
                                        strategy: rec.strategy,
                                        entry_ts: ts,
                                        entry_price: price,
                                        quantity,
                                        current_price: price,
                                        unrealized_pnl: 0.0,
                                    },
                                );

                                let trade = PaperTrade {
                                    ts,
                                    symbol: symbol.clone(),
                                    strategy: rec.strategy,
                                    action: "ENTER".into(),
                                    price,
                                    quantity,
                                    pnl: 0.0,
                                    score: rec.score,
                                    regime: format!("{:?}", status.regime),
                                    hft_detected: status.hft_detected,
                                };

                                writeln!(trade_log, "{}", serde_json::to_string(&trade)?)?;
                                trades.push(trade);

                                println!(
                                    "\x1b[32m[{}] PAPER ENTER {:?} {} @ ₹{:.2} | Score: {:.1}\x1b[0m",
                                    ts.format("%H:%M:%S"),
                                    rec.strategy,
                                    symbol,
                                    price,
                                    rec.score
                                );
                            }
                        }
                    }
                }

                // Update position MTM
                if let Some(pos) = positions.get_mut(&symbol) {
                    pos.current_price = price;
                    pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity as f64;
                }

                // Check stop loss / profit target
                let symbols_to_close: Vec<String> = positions
                    .iter()
                    .filter(|(_, pos)| {
                        pos.unrealized_pnl < -args.stop_loss
                            || pos.unrealized_pnl > args.profit_target
                    })
                    .map(|(sym, _)| sym.clone())
                    .collect();

                for sym in symbols_to_close {
                    if let Some(pos) = positions.remove(&sym) {
                        let action = if pos.unrealized_pnl > 0.0 {
                            "EXIT_PROFIT"
                        } else {
                            "EXIT_STOP"
                        };

                        stats.capital += pos.unrealized_pnl;

                        if pos.unrealized_pnl > 0.0 {
                            stats.winning_trades += 1;
                        } else {
                            stats.losing_trades += 1;
                        }

                        let trade = PaperTrade {
                            ts,
                            symbol: pos.symbol.clone(),
                            strategy: pos.strategy,
                            action: action.into(),
                            price: pos.current_price,
                            quantity: pos.quantity,
                            pnl: pos.unrealized_pnl,
                            score: 0.0,
                            regime: format!("{:?}", status.regime),
                            hft_detected: status.hft_detected,
                        };

                        writeln!(trade_log, "{}", serde_json::to_string(&trade)?)?;
                        trades.push(trade);

                        let color = if pos.unrealized_pnl > 0.0 {
                            "\x1b[32m"
                        } else {
                            "\x1b[31m"
                        };
                        println!(
                            "{}[{}] PAPER {} {} @ ₹{:.2} | P&L: {:+.2}\x1b[0m",
                            color,
                            ts.format("%H:%M:%S"),
                            action,
                            pos.symbol,
                            pos.current_price,
                            pos.unrealized_pnl
                        );
                    }
                }

                // Track drawdown
                let unrealized: f64 = positions.values().map(|p| p.unrealized_pnl).sum();
                let equity = stats.capital + unrealized;
                if equity > stats.peak_equity {
                    stats.peak_equity = equity;
                }
                let drawdown = (stats.peak_equity - equity) / stats.peak_equity * 100.0;
                if drawdown > stats.max_drawdown {
                    stats.max_drawdown = drawdown;
                }

                // Update prev price
                prev_prices.insert(symbol, price);

                // Periodic status
                if stats.decisions.is_multiple_of(100) {
                    print!(
                        "\r[{}] Ticks: {} | Signals: {} | HFT Blocks: {} | Positions: {} | Equity: ₹{:.2}    ",
                        ts.format("%H:%M:%S"),
                        stats.decisions,
                        stats.entry_signals,
                        stats.hft_blocks,
                        positions.len(),
                        equity
                    );
                    std::io::stdout().flush()?;
                }
            }
            Ok(Err(_)) => {
                // Channel closed
                break;
            }
            Err(_) => {
                // Timeout - continue
            }
        }
    }

    // Shutdown
    running.store(false, Ordering::SeqCst);
    connector_handle.abort();

    // Print summary
    print_summary(&stats, &positions, &trades, &args);

    // Save results
    save_results(&stats, &positions, &trades, &args)?;

    Ok(())
}

fn extract_strike_from_symbol(symbol: &str) -> Option<f64> {
    if symbol.len() < 12 {
        return None;
    }

    let stripped = if symbol.ends_with("CE") || symbol.ends_with("PE") {
        &symbol[..symbol.len() - 2]
    } else {
        return None;
    };

    if stripped.len() > 10 {
        let strike_str = &stripped[10..];
        strike_str.parse::<f64>().ok()
    } else {
        None
    }
}

fn print_summary(
    stats: &SessionStats,
    positions: &HashMap<String, PaperPosition>,
    trades: &[PaperTrade],
    args: &Args,
) {
    let unrealized: f64 = positions.values().map(|p| p.unrealized_pnl).sum();
    let realized = stats.capital - stats.initial_capital;
    let total_pnl = realized + unrealized;
    let return_pct = total_pnl / stats.initial_capital * 100.0;

    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                 LIVE PAPER TRADING SESSION SUMMARY               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("CAPITAL:");
    println!("  Initial:         ₹{:.2}", stats.initial_capital);
    println!("  Current:         ₹{:.2}", stats.capital + unrealized);
    println!("  Realized P&L:    ₹{:.2}", realized);
    println!("  Unrealized P&L:  ₹{:.2}", unrealized);
    println!("  Total P&L:       ₹{:.2} ({:+.2}%)", total_pnl, return_pct);
    println!("  Max Drawdown:    {:.2}%", stats.max_drawdown);
    println!();

    println!("ACTIVITY:");
    println!("  Market Ticks:    {}", stats.decisions);
    println!("  Entry Signals:   {}", stats.entry_signals);
    println!("  HFT Blocks:      {}", stats.hft_blocks);
    println!();

    let trades_opened = trades.iter().filter(|t| t.action == "ENTER").count();
    let trades_closed = stats.winning_trades + stats.losing_trades;
    let win_rate = if trades_closed > 0 {
        stats.winning_trades as f64 / trades_closed as f64 * 100.0
    } else {
        0.0
    };

    println!("TRADES:");
    println!("  Opened:          {}", trades_opened);
    println!("  Closed:          {}", trades_closed);
    println!(
        "  Winners:         {} ({:.1}%)",
        stats.winning_trades, win_rate
    );
    println!("  Losers:          {}", stats.losing_trades);
    println!("  Still Open:      {}", positions.len());
    println!();

    if !positions.is_empty() {
        println!("OPEN POSITIONS:");
        for pos in positions.values() {
            let color = if pos.unrealized_pnl >= 0.0 {
                "\x1b[32m"
            } else {
                "\x1b[31m"
            };
            println!(
                "  {}{:?} {} @ ₹{:.2} → ₹{:.2} | P&L: {:+.2}\x1b[0m",
                color,
                pos.strategy,
                pos.symbol,
                pos.entry_price,
                pos.current_price,
                pos.unrealized_pnl
            );
        }
        println!();
    }

    println!("Output saved to: {}/", args.out);
}

fn save_results(
    stats: &SessionStats,
    positions: &HashMap<String, PaperPosition>,
    trades: &[PaperTrade],
    args: &Args,
) -> Result<()> {
    let unrealized: f64 = positions.values().map(|p| p.unrealized_pnl).sum();
    let realized = stats.capital - stats.initial_capital;

    let summary = serde_json::json!({
        "initial_capital": stats.initial_capital,
        "final_equity": stats.capital + unrealized,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "return_pct": (realized + unrealized) / stats.initial_capital * 100.0,
        "max_drawdown_pct": stats.max_drawdown,
        "market_ticks": stats.decisions,
        "entry_signals": stats.entry_signals,
        "hft_blocks": stats.hft_blocks,
        "trades_opened": trades.iter().filter(|t| t.action == "ENTER").count(),
        "trades_closed": stats.winning_trades + stats.losing_trades,
        "winning_trades": stats.winning_trades,
        "losing_trades": stats.losing_trades,
        "win_rate_pct": if stats.winning_trades + stats.losing_trades > 0 {
            stats.winning_trades as f64 / (stats.winning_trades + stats.losing_trades) as f64 * 100.0
        } else { 0.0 },
        "open_positions": positions.len(),
    });

    std::fs::write(
        format!("{}/summary.json", args.out),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}
