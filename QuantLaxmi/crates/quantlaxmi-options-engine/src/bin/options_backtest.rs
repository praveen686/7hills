//! Options Engine Backtest Runner
//!
//! Demonstrates the full power of the QuantLaxmi Options Engine:
//! - Grassmann regime detection
//! - Ramanujan periodicity filtering
//! - Multi-factor strategy selection
//! - Greeks-aware position management

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};

use quantlaxmi_options_engine::{
    EngineConfig, EngineStatus, Greeks, OptionData, OptionDataType, OptionParams, OptionType,
    OptionsEngine, PCRMetrics, PortfolioGreeks, StrategyType, TradingAction, TradingDecision,
};
use quantlaxmi_regime::{FeatureVector, RegimeLabel};

#[derive(Parser)]
#[command(name = "options-backtest")]
#[command(about = "Run options engine backtest on India FNO data")]
struct Args {
    /// Path to quotes JSONL file
    #[arg(long)]
    quotes: String,

    /// Output directory for results
    #[arg(long, default_value = "/tmp/options_backtest")]
    out: String,

    /// Initial capital (INR)
    #[arg(long, default_value = "1000000")]
    capital: f64,

    /// Max positions
    #[arg(long, default_value = "3")]
    max_positions: u32,

    /// Minimum strategy score to trade
    #[arg(long, default_value = "55")]
    min_score: f64,

    /// Enable Ramanujan HFT blocking
    #[arg(long, default_value = "true")]
    block_hft: bool,
}

/// Quote event from replay file.
#[derive(Debug, Clone, Deserialize)]
struct QuoteEvent {
    ts: DateTime<Utc>,
    tradingsymbol: String,
    bid: i64,
    ask: i64,
    bid_qty: u32,
    ask_qty: u32,
    price_exponent: i8,
}

impl QuoteEvent {
    fn bid_f64(&self) -> f64 {
        self.bid as f64 * 10f64.powi(self.price_exponent as i32)
    }

    fn ask_f64(&self) -> f64 {
        self.ask as f64 * 10f64.powi(self.price_exponent as i32)
    }

    fn mid(&self) -> f64 {
        (self.bid_f64() + self.ask_f64()) / 2.0
    }
}

/// Backtest state.
struct BacktestState {
    /// Capital remaining
    capital: f64,
    /// Positions by symbol
    positions: HashMap<String, Position>,
    /// Trade log
    trades: Vec<Trade>,
    /// Decision log
    decisions: Vec<DecisionLog>,
    /// Equity curve
    equity_curve: Vec<EquityPoint>,
    /// Statistics
    stats: BacktestStats,
}

#[derive(Debug, Clone, Serialize)]
struct Position {
    symbol: String,
    strategy: StrategyType,
    entry_ts: DateTime<Utc>,
    entry_price: f64,
    quantity: i32,
    current_price: f64,
    pnl: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Trade {
    ts: DateTime<Utc>,
    symbol: String,
    strategy: StrategyType,
    action: String,
    price: f64,
    quantity: i32,
    pnl: f64,
    score: f64,
    regime: String,
    reasoning: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DecisionLog {
    ts: DateTime<Utc>,
    action: String,
    strategy: Option<StrategyType>,
    score: f64,
    regime: String,
    iv_percentile: f64,
    pcr: f64,
    hft_blocked: bool,
    reasoning: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EquityPoint {
    ts: DateTime<Utc>,
    equity: f64,
    drawdown: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct BacktestStats {
    total_trades: u32,
    winning_trades: u32,
    losing_trades: u32,
    gross_pnl: f64,
    net_pnl: f64,
    max_drawdown: f64,
    sharpe_ratio: f64,
    win_rate: f64,
    profit_factor: f64,
    avg_trade_pnl: f64,
    decisions_made: u32,
    hft_blocks: u32,
    regime_blocks: u32,
    score_blocks: u32,
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("=== QuantLaxmi Options Engine Backtest ===\n");
    println!("Quotes: {}", args.quotes);
    println!("Output: {}", args.out);
    println!("Capital: ₹{:.0}", args.capital);
    println!("Max Positions: {}", args.max_positions);
    println!("Min Score: {}", args.min_score);
    println!("Block HFT: {}", args.block_hft);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.out)?;

    // Load quotes
    let quotes = load_quotes(&args.quotes)?;
    println!("Loaded {} quotes", quotes.len());

    // Initialize engine
    let config = EngineConfig {
        symbol: "NIFTY".into(),
        lot_size: 50, // NIFTY lot size
        risk_free_rate: 0.065,
        dividend_yield: 0.012,
        max_positions: args.max_positions,
        max_loss_per_position: 25000.0,
        max_portfolio_delta: 500.0,
        min_iv_percentile_sell: 60.0,
        max_iv_percentile_buy: 40.0,
        min_strategy_score: args.min_score,
        ramanujan_enabled: true,
        block_on_hft: args.block_hft,
        pcr_enabled: true,
        pcr_lookback: 100,
    };

    let mut engine = OptionsEngine::new(config);
    let mut state = BacktestState {
        capital: args.capital,
        positions: HashMap::new(),
        trades: Vec::new(),
        decisions: Vec::new(),
        equity_curve: Vec::new(),
        stats: BacktestStats::default(),
    };

    // Track previous quote for feature calculation
    let mut prev_quotes: HashMap<String, QuoteEvent> = HashMap::new();
    let mut peak_equity = args.capital;

    println!("\nRunning backtest...\n");

    // Process quotes
    for (i, quote) in quotes.iter().enumerate() {
        // Calculate features
        let features = calculate_features(quote, prev_quotes.get(&quote.tradingsymbol));

        // Extract underlying spot from symbol (e.g., NIFTY2620325300CE -> 25300)
        let spot = extract_strike_from_symbol(&quote.tradingsymbol).unwrap_or(25000.0);
        engine.on_tick(quote.ts, spot, &features);

        // Get trading decision
        let decision = engine.decide(quote.ts);
        state.stats.decisions_made += 1;

        // Log decision
        let status = engine.status();
        let hft_blocked = status.hft_detected && args.block_hft;

        if hft_blocked {
            state.stats.hft_blocks += 1;
        }

        let decision_log = DecisionLog {
            ts: quote.ts,
            action: format!("{:?}", decision.action),
            strategy: decision.strategy.as_ref().map(|s| s.strategy),
            score: decision.confidence * 100.0,
            regime: format!("{:?}", status.regime),
            iv_percentile: status.iv_percentile,
            pcr: status.pcr,
            hft_blocked,
            reasoning: decision.reasoning.clone(),
        };
        state.decisions.push(decision_log);

        // Process decision
        match decision.action {
            TradingAction::Enter => {
                if let Some(rec) = &decision.strategy {
                    // Only enter if we don't already have a position and have room
                    let has_position = state.positions.contains_key(&quote.tradingsymbol);
                    if !has_position && state.positions.len() < args.max_positions as usize {
                        // Enter position
                        let quantity = 50; // 1 lot
                        let entry_price = quote.mid();

                        state.positions.insert(
                            quote.tradingsymbol.clone(),
                            Position {
                                symbol: quote.tradingsymbol.clone(),
                                strategy: rec.strategy,
                                entry_ts: quote.ts,
                                entry_price,
                                quantity,
                                current_price: entry_price,
                                pnl: 0.0,
                            },
                        );

                        state.trades.push(Trade {
                            ts: quote.ts,
                            symbol: quote.tradingsymbol.clone(),
                            strategy: rec.strategy,
                            action: "ENTER".into(),
                            price: entry_price,
                            quantity,
                            pnl: 0.0,
                            score: rec.score,
                            regime: format!("{:?}", status.regime),
                            reasoning: rec.reasoning.clone(),
                        });

                        println!(
                            "[{}] ENTER {:?} on {} @ {:.2} | Score: {:.1} | Regime: {:?}",
                            quote.ts.format("%H:%M:%S"),
                            rec.strategy,
                            quote.tradingsymbol,
                            entry_price,
                            rec.score,
                            status.regime
                        );
                    }
                }
            }
            TradingAction::Wait => {
                // Track blocking reason
                if decision.reasoning.iter().any(|r| r.contains("Score")) {
                    state.stats.score_blocks += 1;
                }
            }
            _ => {}
        }

        // Update position MTM
        for pos in state.positions.values_mut() {
            if pos.symbol == quote.tradingsymbol {
                pos.current_price = quote.mid();
                pos.pnl = (pos.current_price - pos.entry_price) * pos.quantity as f64;
            }
        }

        // Exit logic: close positions on stop loss (-5000) or profit target (+2500)
        let symbols_to_close: Vec<String> = state
            .positions
            .iter()
            .filter(|(_, pos)| pos.pnl < -5000.0 || pos.pnl > 2500.0)
            .map(|(sym, _)| sym.clone())
            .collect();

        for sym in symbols_to_close {
            if let Some(pos) = state.positions.remove(&sym) {
                let action = if pos.pnl > 0.0 { "PROFIT" } else { "STOP" };
                state.trades.push(Trade {
                    ts: quote.ts,
                    symbol: pos.symbol.clone(),
                    strategy: pos.strategy,
                    action: format!("EXIT_{}", action),
                    price: pos.current_price,
                    quantity: pos.quantity,
                    pnl: pos.pnl,
                    score: 0.0,
                    regime: format!("{:?}", status.regime),
                    reasoning: vec![format!("Closed at {} target", action)],
                });

                // Credit/debit capital
                state.capital += pos.pnl;

                if pos.pnl > 0.0 {
                    state.stats.winning_trades += 1;
                } else {
                    state.stats.losing_trades += 1;
                }

                println!(
                    "[{}] {} {} @ {:.2} | P&L: {:+.2}",
                    quote.ts.format("%H:%M:%S"),
                    action,
                    pos.symbol,
                    pos.current_price,
                    pos.pnl
                );
            }
        }

        // Calculate equity
        let positions_value: f64 = state.positions.values().map(|p| p.pnl).sum();
        let equity = state.capital + positions_value;

        if equity > peak_equity {
            peak_equity = equity;
        }
        let drawdown = (peak_equity - equity) / peak_equity * 100.0;

        if i % 1000 == 0 {
            state.equity_curve.push(EquityPoint {
                ts: quote.ts,
                equity,
                drawdown,
            });
        }

        if drawdown > state.stats.max_drawdown {
            state.stats.max_drawdown = drawdown;
        }

        // Store previous quote
        prev_quotes.insert(quote.tradingsymbol.clone(), quote.clone());

        // Progress
        if i % 5000 == 0 {
            print!(
                "\rProcessed {}/{} quotes ({:.1}%)",
                i,
                quotes.len(),
                i as f64 / quotes.len() as f64 * 100.0
            );
        }
    }

    println!("\n");

    // Close remaining positions at last price (mark to market)
    let open_pnl: f64 = state.positions.values().map(|p| p.pnl).sum();

    // Calculate realized P&L from closed trades
    let realized_pnl: f64 = state
        .trades
        .iter()
        .filter(|t| t.action.starts_with("EXIT"))
        .map(|t| t.pnl)
        .sum();

    state.stats.gross_pnl = realized_pnl + open_pnl;
    state.stats.net_pnl = realized_pnl + open_pnl;
    state.stats.total_trades = state.trades.len() as u32;

    // Calculate win rate
    let closed_trades = state.stats.winning_trades + state.stats.losing_trades;
    state.stats.win_rate = if closed_trades > 0 {
        state.stats.winning_trades as f64 / closed_trades as f64 * 100.0
    } else {
        0.0
    };

    // Calculate final stats
    let final_equity = state.capital + open_pnl;

    // Print results
    println!("=== BACKTEST RESULTS ===\n");
    println!("Initial Capital:    ₹{:.2}", args.capital);
    println!("Final Equity:       ₹{:.2}", final_equity);
    println!("Realized P&L:       ₹{:.2}", realized_pnl);
    println!("Open P&L:           ₹{:.2}", open_pnl);
    println!(
        "Total P&L:          ₹{:.2} ({:+.2}%)",
        state.stats.net_pnl,
        state.stats.net_pnl / args.capital * 100.0
    );
    println!("Max Drawdown:       {:.2}%", state.stats.max_drawdown);
    println!();
    println!("Total Decisions:    {}", state.stats.decisions_made);
    println!(
        "Trades Opened:      {}",
        state.trades.iter().filter(|t| t.action == "ENTER").count()
    );
    println!("Trades Closed:      {}", closed_trades);
    println!(
        "  - Winners:        {} ({:.1}%)",
        state.stats.winning_trades, state.stats.win_rate
    );
    println!("  - Losers:         {}", state.stats.losing_trades);
    println!("Open Positions:     {}", state.positions.len());
    println!();
    println!("HFT Blocks:         {}", state.stats.hft_blocks);
    println!("Score Blocks:       {}", state.stats.score_blocks);
    println!();

    // Strategy breakdown
    println!("=== STRATEGY BREAKDOWN ===\n");
    let mut strategy_counts: HashMap<StrategyType, u32> = HashMap::new();
    for trade in &state.trades {
        *strategy_counts.entry(trade.strategy).or_default() += 1;
    }
    for (strat, count) in &strategy_counts {
        println!("{:?}: {} trades", strat, count);
    }
    println!();

    // Regime breakdown
    println!("=== REGIME BREAKDOWN ===\n");
    let mut regime_counts: HashMap<String, u32> = HashMap::new();
    for trade in &state.trades {
        *regime_counts.entry(trade.regime.clone()).or_default() += 1;
    }
    for (regime, count) in &regime_counts {
        println!("{}: {} trades", regime, count);
    }
    println!();

    // Save results
    let summary = serde_json::json!({
        "initial_capital": args.capital,
        "final_equity": final_equity,
        "realized_pnl": realized_pnl,
        "open_pnl": open_pnl,
        "total_pnl": state.stats.net_pnl,
        "return_pct": state.stats.net_pnl / args.capital * 100.0,
        "max_drawdown_pct": state.stats.max_drawdown,
        "total_decisions": state.stats.decisions_made,
        "trades_opened": state.trades.iter().filter(|t| t.action == "ENTER").count(),
        "trades_closed": closed_trades,
        "winning_trades": state.stats.winning_trades,
        "losing_trades": state.stats.losing_trades,
        "win_rate_pct": state.stats.win_rate,
        "open_positions": state.positions.len(),
        "hft_blocks": state.stats.hft_blocks,
        "score_blocks": state.stats.score_blocks,
        "strategy_counts": strategy_counts.iter().map(|(k, v)| (format!("{:?}", k), v)).collect::<HashMap<_, _>>(),
        "regime_counts": regime_counts,
    });

    std::fs::write(
        format!("{}/summary.json", args.out),
        serde_json::to_string_pretty(&summary)?,
    )?;

    std::fs::write(
        format!("{}/trades.json", args.out),
        serde_json::to_string_pretty(&state.trades)?,
    )?;

    std::fs::write(
        format!("{}/equity_curve.json", args.out),
        serde_json::to_string_pretty(&state.equity_curve)?,
    )?;

    println!("Results saved to {}/", args.out);

    Ok(())
}

fn load_quotes(path: &str) -> Result<Vec<QuoteEvent>> {
    let file = std::fs::File::open(path).context("open quotes file")?;
    let reader = BufReader::new(file);
    let mut quotes = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.context("read line")?;
        if line.trim().is_empty() {
            continue;
        }
        let quote: QuoteEvent = serde_json::from_str(&line)
            .with_context(|| format!("parse quote at line {}", i + 1))?;
        quotes.push(quote);
    }

    Ok(quotes)
}

fn calculate_features(quote: &QuoteEvent, prev: Option<&QuoteEvent>) -> FeatureVector {
    let mid = quote.mid();

    // Mid return in bps
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

    // Book imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) * 10000
    let total_qty = quote.bid_qty as i64 + quote.ask_qty as i64;
    let imbalance = if total_qty > 0 {
        ((quote.bid_qty as i64 - quote.ask_qty as i64) * 10000) / total_qty
    } else {
        0
    };

    // Spread in bps
    let spread_bps = if mid > 0.0 {
        (((quote.ask_f64() - quote.bid_f64()) / mid) * 10000.0) as i64
    } else {
        0
    };

    // Volatility proxy (spread)
    let vol_proxy = spread_bps;

    // Pressure: bid_qty * 100 / total
    let pressure = if total_qty > 0 {
        (quote.bid_qty as i64 * 100) / total_qty
    } else {
        50
    };

    // VPIN proxy
    let vpin = imbalance.abs();

    FeatureVector::new(mid_return, imbalance, spread_bps, vol_proxy, pressure, vpin)
}

/// Extract strike price from symbol like NIFTY2620325300CE -> 25300.0
fn extract_strike_from_symbol(symbol: &str) -> Option<f64> {
    // Pattern: NIFTY + YYMMDD + STRIKE + CE/PE
    // e.g., NIFTY2620325300CE = NIFTY + 26203 + 25300 + CE
    if symbol.len() < 12 {
        return None;
    }

    // Strip trailing CE/PE
    let stripped = if symbol.ends_with("CE") || symbol.ends_with("PE") {
        &symbol[..symbol.len() - 2]
    } else {
        return None;
    };

    // Find where NIFTY ends (5 chars) + YYMMDD (5 chars) = 10 chars
    // The rest is the strike
    if stripped.len() > 10 {
        let strike_str = &stripped[10..];
        strike_str.parse::<f64>().ok()
    } else {
        None
    }
}
