//! Options Engine Paper Trading
//!
//! Live paper trading with the QuantLaxmi Options Engine:
//! - Connects to Zerodha WebSocket for real-time data
//! - Runs Ramanujan + Grassmann gates in real-time
//! - Simulates trades without real execution
//! - Logs all decisions and paper P&L
//!
//! Usage:
//!   paper-trading --symbols NIFTY2620325300CE,NIFTY2620325300PE --duration 3600

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};

use quantlaxmi_options_engine::{EngineConfig, OptionsEngine, StrategyType, TradingAction};
use quantlaxmi_regime::FeatureVector;

#[derive(Parser)]
#[command(name = "paper-trading")]
#[command(about = "Live paper trading with the Options Engine")]
struct Args {
    /// Symbols to trade (comma-separated)
    #[arg(long)]
    symbols: String,

    /// Duration in seconds (0 = run forever)
    #[arg(long, default_value = "3600")]
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
    #[arg(long, default_value = "/tmp/paper_trading")]
    out: String,

    /// Kite WebSocket URL (or use mock for testing)
    #[arg(long, default_value = "mock")]
    data_source: String,

    /// Mock data file (for testing without live connection)
    #[arg(long)]
    mock_file: Option<String>,

    /// Print verbose decision logs
    #[arg(long, default_value = "false")]
    verbose: bool,
}

/// Live quote from data source.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct LiveQuote {
    ts: DateTime<Utc>,
    symbol: String,
    bid: f64,
    ask: f64,
    bid_qty: u32,
    ask_qty: u32,
    last_price: f64,
    volume: u64,
}

impl LiveQuote {
    fn mid(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
}

/// Paper position.
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

/// Paper trade record.
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
    reasoning: Vec<String>,
}

/// Paper trading session state.
struct PaperSession {
    capital: f64,
    initial_capital: f64,
    positions: HashMap<String, PaperPosition>,
    trades: Vec<PaperTrade>,
    decisions_made: u32,
    hft_blocks: u32,
    entry_signals: u32,
    signals_blocked: u32,
    winning_trades: u32,
    losing_trades: u32,
    peak_equity: f64,
    max_drawdown: f64,
}

impl PaperSession {
    fn new(capital: f64) -> Self {
        Self {
            capital,
            initial_capital: capital,
            positions: HashMap::new(),
            trades: Vec::new(),
            decisions_made: 0,
            hft_blocks: 0,
            entry_signals: 0,
            signals_blocked: 0,
            winning_trades: 0,
            losing_trades: 0,
            peak_equity: capital,
            max_drawdown: 0.0,
        }
    }

    fn total_equity(&self) -> f64 {
        let unrealized: f64 = self.positions.values().map(|p| p.unrealized_pnl).sum();
        self.capital + unrealized
    }

    fn realized_pnl(&self) -> f64 {
        self.capital - self.initial_capital
    }

    fn unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let symbols: Vec<String> = args
        .symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        QuantLaxmi Options Engine - Paper Trading                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Configuration:");
    println!("  Symbols:       {:?}", symbols);
    println!(
        "  Duration:      {} seconds",
        if args.duration == 0 {
            "unlimited".to_string()
        } else {
            args.duration.to_string()
        }
    );
    println!("  Capital:       ₹{:.0}", args.capital);
    println!("  Max Positions: {}", args.max_positions);
    println!("  Min Score:     {}", args.min_score);
    println!("  Data Source:   {}", args.data_source);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.out)?;

    // Initialize engine
    let config = EngineConfig {
        symbol: "NIFTY".into(),
        lot_size: 50,
        risk_free_rate: 0.065,
        dividend_yield: 0.012,
        max_positions: args.max_positions,
        max_loss_per_position: 25000.0,
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
    let mut session = PaperSession::new(args.capital);
    let mut prev_quotes: HashMap<String, LiveQuote> = HashMap::new();

    // Setup signal handler for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n\nReceived Ctrl+C, shutting down...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let start_time = Instant::now();
    let duration = if args.duration == 0 {
        Duration::MAX
    } else {
        Duration::from_secs(args.duration)
    };

    println!("Starting paper trading session...\n");
    println!("Press Ctrl+C to stop\n");

    // Open trade log file
    let trade_log_path = format!("{}/trades.jsonl", args.out);
    let mut trade_log = std::fs::File::create(&trade_log_path)?;

    // Main loop - get quotes from data source
    match args.data_source.as_str() {
        "mock" => {
            // Use mock data from file or generate synthetic quotes
            if let Some(mock_file) = &args.mock_file {
                run_from_file(
                    &mut engine,
                    &mut session,
                    &mut prev_quotes,
                    mock_file,
                    &args,
                    &mut trade_log,
                    running.clone(),
                )?;
            } else {
                println!("Mock mode: Use --mock-file to provide test data, or use --data-source kite for live data");
                println!("\nTo capture live data first, run:");
                println!("  cargo run --bin capture-zerodha -- --symbols {} --duration-secs 300 --out /tmp/quotes.jsonl", args.symbols);
                return Ok(());
            }
        }
        "kite" => {
            println!("Kite WebSocket mode requires authentication.");
            println!("Ensure KITE_API_KEY and KITE_ACCESS_TOKEN are set.");
            println!("\nFor now, use mock mode with captured data:");
            println!("  1. Capture: cargo run --bin capture-zerodha -- --symbols {} --duration-secs 300 --out /tmp/quotes.jsonl", args.symbols);
            println!("  2. Paper:  cargo run --bin paper-trading -- --symbols {} --mock-file /tmp/quotes.jsonl", args.symbols);
            return Ok(());
        }
        "stdin" => {
            // Read quotes from stdin (for piping from capture)
            run_from_stdin(
                &mut engine,
                &mut session,
                &mut prev_quotes,
                &args,
                &mut trade_log,
                running.clone(),
                duration,
                start_time,
            )?;
        }
        _ => {
            bail!(
                "Unknown data source: {}. Use 'mock', 'kite', or 'stdin'",
                args.data_source
            );
        }
    }

    // Print final summary
    print_session_summary(&session, &args);

    // Save session results
    save_session_results(&session, &args)?;

    Ok(())
}

fn run_from_file(
    engine: &mut OptionsEngine,
    session: &mut PaperSession,
    prev_quotes: &mut HashMap<String, LiveQuote>,
    file_path: &str,
    args: &Args,
    trade_log: &mut std::fs::File,
    running: Arc<AtomicBool>,
) -> Result<()> {
    let file = std::fs::File::open(file_path).context("open mock file")?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as LiveQuote or as the capture format
        let quote: LiveQuote = match serde_json::from_str(&line) {
            Ok(q) => q,
            Err(_) => {
                // Try alternate format from capture tool
                if let Ok(capture_quote) = parse_capture_format(&line) {
                    capture_quote
                } else {
                    continue;
                }
            }
        };

        process_quote(engine, session, prev_quotes, &quote, args, trade_log)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_from_stdin(
    engine: &mut OptionsEngine,
    session: &mut PaperSession,
    prev_quotes: &mut HashMap<String, LiveQuote>,
    args: &Args,
    trade_log: &mut std::fs::File,
    running: Arc<AtomicBool>,
    duration: Duration,
    start_time: Instant,
) -> Result<()> {
    let stdin = std::io::stdin();
    let reader = stdin.lock();

    for line in reader.lines() {
        if !running.load(Ordering::SeqCst) {
            break;
        }
        if start_time.elapsed() > duration {
            println!("\nDuration limit reached.");
            break;
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let quote: LiveQuote = match serde_json::from_str(&line) {
            Ok(q) => q,
            Err(_) => {
                if let Ok(capture_quote) = parse_capture_format(&line) {
                    capture_quote
                } else {
                    continue;
                }
            }
        };

        process_quote(engine, session, prev_quotes, &quote, args, trade_log)?;
    }

    Ok(())
}

fn parse_capture_format(line: &str) -> Result<LiveQuote> {
    // Parse the capture format from zerodha_capture
    #[derive(Deserialize)]
    struct CaptureQuote {
        ts: DateTime<Utc>,
        tradingsymbol: String,
        bid: i64,
        ask: i64,
        bid_qty: u32,
        ask_qty: u32,
        price_exponent: i8,
        #[serde(default)]
        last_price: Option<i64>,
        #[serde(default)]
        volume: Option<u64>,
    }

    let cq: CaptureQuote = serde_json::from_str(line)?;
    let scale = 10f64.powi(cq.price_exponent as i32);

    Ok(LiveQuote {
        ts: cq.ts,
        symbol: cq.tradingsymbol,
        bid: cq.bid as f64 * scale,
        ask: cq.ask as f64 * scale,
        bid_qty: cq.bid_qty,
        ask_qty: cq.ask_qty,
        last_price: cq
            .last_price
            .map(|p| p as f64 * scale)
            .unwrap_or_else(|| (cq.bid + cq.ask) as f64 * scale / 2.0),
        volume: cq.volume.unwrap_or(0),
    })
}

fn process_quote(
    engine: &mut OptionsEngine,
    session: &mut PaperSession,
    prev_quotes: &mut HashMap<String, LiveQuote>,
    quote: &LiveQuote,
    args: &Args,
    trade_log: &mut std::fs::File,
) -> Result<()> {
    // Calculate features
    let features = calculate_features(quote, prev_quotes.get(&quote.symbol));

    // Extract spot from symbol
    let spot = extract_strike_from_symbol(&quote.symbol).unwrap_or(25000.0);
    engine.on_tick(quote.ts, spot, &features);

    // Get decision
    let decision = engine.decide(quote.ts);
    let status = engine.status();
    session.decisions_made += 1;

    // Track entry signals and HFT blocks
    let is_entry = matches!(decision.action, TradingAction::Enter);
    if is_entry {
        session.entry_signals += 1;
    }
    if status.hft_detected {
        session.hft_blocks += 1;
        if is_entry {
            session.signals_blocked += 1;
        }
    }

    // Process decision
    if !status.hft_detected && decision.action == TradingAction::Enter {
        if let Some(rec) = &decision.strategy {
            let has_position = session.positions.contains_key(&quote.symbol);
            if !has_position && session.positions.len() < args.max_positions as usize {
                // Paper entry
                let entry_price = quote.mid();
                let quantity = 50; // 1 lot

                session.positions.insert(
                    quote.symbol.clone(),
                    PaperPosition {
                        symbol: quote.symbol.clone(),
                        strategy: rec.strategy,
                        entry_ts: quote.ts,
                        entry_price,
                        quantity,
                        current_price: entry_price,
                        unrealized_pnl: 0.0,
                    },
                );

                let trade = PaperTrade {
                    ts: quote.ts,
                    symbol: quote.symbol.clone(),
                    strategy: rec.strategy,
                    action: "ENTER".into(),
                    price: entry_price,
                    quantity,
                    pnl: 0.0,
                    score: rec.score,
                    regime: format!("{:?}", status.regime),
                    hft_detected: status.hft_detected,
                    reasoning: rec.reasoning.clone(),
                };

                // Log to file
                writeln!(trade_log, "{}", serde_json::to_string(&trade)?)?;
                session.trades.push(trade);

                println!(
                    "\x1b[32m[{}] PAPER ENTER {:?} on {} @ ₹{:.2} | Score: {:.1} | Regime: {:?}\x1b[0m",
                    quote.ts.format("%H:%M:%S"),
                    rec.strategy,
                    quote.symbol,
                    entry_price,
                    rec.score,
                    status.regime
                );
            }
        }
    }

    // Update position MTM
    for pos in session.positions.values_mut() {
        if pos.symbol == quote.symbol {
            pos.current_price = quote.mid();
            pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity as f64;
        }
    }

    // Check stop loss / profit target
    let symbols_to_close: Vec<String> = session
        .positions
        .iter()
        .filter(|(_, pos)| pos.unrealized_pnl < -5000.0 || pos.unrealized_pnl > 2500.0)
        .map(|(sym, _)| sym.clone())
        .collect();

    for sym in symbols_to_close {
        if let Some(pos) = session.positions.remove(&sym) {
            let action = if pos.unrealized_pnl > 0.0 {
                "EXIT_PROFIT"
            } else {
                "EXIT_STOP"
            };

            session.capital += pos.unrealized_pnl;

            if pos.unrealized_pnl > 0.0 {
                session.winning_trades += 1;
            } else {
                session.losing_trades += 1;
            }

            let trade = PaperTrade {
                ts: quote.ts,
                symbol: pos.symbol.clone(),
                strategy: pos.strategy,
                action: action.into(),
                price: pos.current_price,
                quantity: pos.quantity,
                pnl: pos.unrealized_pnl,
                score: 0.0,
                regime: format!("{:?}", status.regime),
                hft_detected: status.hft_detected,
                reasoning: vec![format!("Closed at {} target", action)],
            };

            writeln!(trade_log, "{}", serde_json::to_string(&trade)?)?;
            session.trades.push(trade);

            let color = if pos.unrealized_pnl > 0.0 {
                "\x1b[32m"
            } else {
                "\x1b[31m"
            };
            println!(
                "{}[{}] PAPER {} {} @ ₹{:.2} | P&L: {:+.2}\x1b[0m",
                color,
                quote.ts.format("%H:%M:%S"),
                action,
                pos.symbol,
                pos.current_price,
                pos.unrealized_pnl
            );
        }
    }

    // Track drawdown
    let equity = session.total_equity();
    if equity > session.peak_equity {
        session.peak_equity = equity;
    }
    let drawdown = (session.peak_equity - equity) / session.peak_equity * 100.0;
    if drawdown > session.max_drawdown {
        session.max_drawdown = drawdown;
    }

    // Periodic status update
    if session.decisions_made.is_multiple_of(1000) {
        print!(
            "\r[{}] Decisions: {} | Signals: {} (blocked: {}) | Positions: {} | Equity: ₹{:.2}     ",
            quote.ts.format("%H:%M:%S"),
            session.decisions_made,
            session.entry_signals,
            session.signals_blocked,
            session.positions.len(),
            equity
        );
        std::io::stdout().flush()?;
    }

    prev_quotes.insert(quote.symbol.clone(), quote.clone());
    Ok(())
}

fn calculate_features(quote: &LiveQuote, prev: Option<&LiveQuote>) -> FeatureVector {
    let mid = quote.mid();

    let mid_return = if let Some(prev) = prev {
        let prev_mid = prev.mid();
        if prev_mid > 0.0 {
            (((mid - prev_mid) / prev_mid) * 10000.0) as i64
        } else {
            0
        }
    } else {
        0
    };

    let total_qty = quote.bid_qty as i64 + quote.ask_qty as i64;
    let imbalance = if total_qty > 0 {
        ((quote.bid_qty as i64 - quote.ask_qty as i64) * 10000) / total_qty
    } else {
        0
    };

    let spread_bps = if mid > 0.0 {
        (((quote.ask - quote.bid) / mid) * 10000.0) as i64
    } else {
        0
    };

    let vol_proxy = spread_bps;

    let pressure = if total_qty > 0 {
        (quote.bid_qty as i64 * 100) / total_qty
    } else {
        50
    };

    let vpin = imbalance.abs();

    FeatureVector::new(mid_return, imbalance, spread_bps, vol_proxy, pressure, vpin)
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

fn print_session_summary(session: &PaperSession, args: &Args) {
    println!("\n\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    PAPER TRADING SESSION SUMMARY                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let equity = session.total_equity();
    let total_pnl = session.realized_pnl() + session.unrealized_pnl();
    let return_pct = total_pnl / session.initial_capital * 100.0;

    println!("CAPITAL:");
    println!("  Initial:         ₹{:.2}", session.initial_capital);
    println!("  Current Equity:  ₹{:.2}", equity);
    println!("  Realized P&L:    ₹{:.2}", session.realized_pnl());
    println!("  Unrealized P&L:  ₹{:.2}", session.unrealized_pnl());
    println!("  Total P&L:       ₹{:.2} ({:+.2}%)", total_pnl, return_pct);
    println!("  Max Drawdown:    {:.2}%", session.max_drawdown);
    println!();

    println!("ACTIVITY:");
    println!("  Decisions:       {}", session.decisions_made);
    println!("  Entry Signals:   {}", session.entry_signals);
    println!(
        "  Signals Blocked: {} ({:.1}%)",
        session.signals_blocked,
        if session.entry_signals > 0 {
            session.signals_blocked as f64 / session.entry_signals as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("  HFT Blocks:      {}", session.hft_blocks);
    println!();

    let trades_opened = session
        .trades
        .iter()
        .filter(|t| t.action == "ENTER")
        .count();
    let trades_closed = session.winning_trades + session.losing_trades;
    let win_rate = if trades_closed > 0 {
        session.winning_trades as f64 / trades_closed as f64 * 100.0
    } else {
        0.0
    };

    println!("TRADES:");
    println!("  Opened:          {}", trades_opened);
    println!("  Closed:          {}", trades_closed);
    println!(
        "  Winners:         {} ({:.1}%)",
        session.winning_trades, win_rate
    );
    println!("  Losers:          {}", session.losing_trades);
    println!("  Still Open:      {}", session.positions.len());
    println!();

    if !session.positions.is_empty() {
        println!("OPEN POSITIONS:");
        for pos in session.positions.values() {
            let color = if pos.unrealized_pnl >= 0.0 {
                "\x1b[32m"
            } else {
                "\x1b[31m"
            };
            println!(
                "  {} {:?} {} @ ₹{:.2} → ₹{:.2} | P&L: {:+.2}\x1b[0m",
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

fn save_session_results(session: &PaperSession, args: &Args) -> Result<()> {
    let summary = serde_json::json!({
        "initial_capital": session.initial_capital,
        "final_equity": session.total_equity(),
        "realized_pnl": session.realized_pnl(),
        "unrealized_pnl": session.unrealized_pnl(),
        "total_pnl": session.realized_pnl() + session.unrealized_pnl(),
        "return_pct": (session.realized_pnl() + session.unrealized_pnl()) / session.initial_capital * 100.0,
        "max_drawdown_pct": session.max_drawdown,
        "decisions_made": session.decisions_made,
        "entry_signals": session.entry_signals,
        "signals_blocked": session.signals_blocked,
        "hft_blocks": session.hft_blocks,
        "trades_opened": session.trades.iter().filter(|t| t.action == "ENTER").count(),
        "trades_closed": session.winning_trades + session.losing_trades,
        "winning_trades": session.winning_trades,
        "losing_trades": session.losing_trades,
        "win_rate_pct": if session.winning_trades + session.losing_trades > 0 {
            session.winning_trades as f64 / (session.winning_trades + session.losing_trades) as f64 * 100.0
        } else { 0.0 },
        "open_positions": session.positions.len(),
    });

    std::fs::write(
        format!("{}/summary.json", args.out),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}
