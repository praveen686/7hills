//! Gate Comparison Backtest
//!
//! Runs multiple backtests to compare the effect of each gate:
//! - Baseline: No gates
//! - Ramanujan only: HFT detection
//! - Grassmann only: Regime detection
//! - Combined: Both gates
//!
//! Produces a comprehensive comparison report.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};

use quantlaxmi_options_engine::{EngineConfig, OptionsEngine, StrategyType, TradingAction};
use quantlaxmi_regime::FeatureVector;

#[derive(Parser)]
#[command(name = "gate-comparison")]
#[command(about = "Compare the effect of different trading gates")]
struct Args {
    /// Path to quotes JSONL file
    #[arg(long)]
    quotes: String,

    /// Output directory for results
    #[arg(long, default_value = "/tmp/gate_comparison")]
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
    #[allow(dead_code)]
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

/// Gate configuration for a backtest run.
#[derive(Debug, Clone, Copy)]
struct GateConfig {
    /// Name of this configuration
    name: &'static str,
    /// Enable Ramanujan HFT detection
    ramanujan_enabled: bool,
    /// Block on HFT detection
    block_on_hft: bool,
    /// Minimum score to trade (0 = disabled)
    min_score: f64,
    /// Use regime-aware strategy selection
    regime_aware: bool,
}

/// Results from a single backtest run.
#[derive(Debug, Clone, Serialize)]
struct BacktestResult {
    name: String,
    trades_opened: u32,
    trades_closed: u32,
    winning_trades: u32,
    losing_trades: u32,
    realized_pnl: f64,
    open_pnl: f64,
    total_pnl: f64,
    return_pct: f64,
    max_drawdown_pct: f64,
    win_rate_pct: f64,
    hft_blocks: u32,
    regime_blocks: u32,
    score_blocks: u32,
    total_decisions: u32,
    /// Entry signals generated (before position limit)
    entry_signals: u32,
    /// Entry signals that were blocked by gates
    signals_blocked: u32,
    strategies_used: HashMap<String, u32>,
    regimes_seen: HashMap<String, u32>,
}

/// Position in the portfolio.
struct Position {
    symbol: String,
    strategy: StrategyType,
    entry_price: f64,
    quantity: i32,
    current_price: f64,
    pnl: f64,
}

/// Trade record.
#[derive(Clone)]
struct Trade {
    strategy: StrategyType,
    action: String,
    pnl: f64,
    regime: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        QuantLaxmi Gate Comparison Analysis                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Configuration:");
    println!("  Quotes:        {}", args.quotes);
    println!("  Capital:       ₹{:.0}", args.capital);
    println!("  Max Positions: {}", args.max_positions);
    println!("  Min Score:     {}", args.min_score);
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.out)?;

    // Load quotes once
    let quotes = load_quotes(&args.quotes)?;
    println!("Loaded {} quotes\n", quotes.len());

    // Define gate configurations to test
    let configs = vec![
        GateConfig {
            name: "1. Baseline (No Gates)",
            ramanujan_enabled: false,
            block_on_hft: false,
            min_score: 0.0, // Accept any score
            regime_aware: false,
        },
        GateConfig {
            name: "2. Score Gate Only",
            ramanujan_enabled: false,
            block_on_hft: false,
            min_score: args.min_score,
            regime_aware: false,
        },
        GateConfig {
            name: "3. Ramanujan HFT Gate",
            ramanujan_enabled: true,
            block_on_hft: true,
            min_score: 0.0,
            regime_aware: false,
        },
        GateConfig {
            name: "4. Grassmann Regime Gate",
            ramanujan_enabled: false,
            block_on_hft: false,
            min_score: 0.0,
            regime_aware: true,
        },
        GateConfig {
            name: "5. Ramanujan + Score",
            ramanujan_enabled: true,
            block_on_hft: true,
            min_score: args.min_score,
            regime_aware: false,
        },
        GateConfig {
            name: "6. Grassmann + Score",
            ramanujan_enabled: false,
            block_on_hft: false,
            min_score: args.min_score,
            regime_aware: true,
        },
        GateConfig {
            name: "7. Ramanujan + Grassmann",
            ramanujan_enabled: true,
            block_on_hft: true,
            min_score: 0.0,
            regime_aware: true,
        },
        GateConfig {
            name: "8. ALL GATES (Full System)",
            ramanujan_enabled: true,
            block_on_hft: true,
            min_score: args.min_score,
            regime_aware: true,
        },
    ];

    // Run each configuration
    let mut results = Vec::new();
    for (i, config) in configs.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Running {} ({}/{})", config.name, i + 1, configs.len());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let result = run_backtest(&quotes, &args, *config)?;
        results.push(result);
        println!();
    }

    // Print comparison table
    print_comparison_table(&results);

    // Print gate effectiveness analysis
    print_gate_analysis(&results);

    // Save detailed results
    let summary = serde_json::json!({
        "configurations": configs.iter().map(|c| c.name).collect::<Vec<_>>(),
        "results": results,
    });

    std::fs::write(
        format!("{}/comparison.json", args.out),
        serde_json::to_string_pretty(&summary)?,
    )?;

    println!("\nResults saved to {}/", args.out);

    Ok(())
}

fn run_backtest(quotes: &[QuoteEvent], args: &Args, config: GateConfig) -> Result<BacktestResult> {
    // Create engine with this configuration
    let engine_config = EngineConfig {
        symbol: "NIFTY".into(),
        lot_size: 50,
        risk_free_rate: 0.065,
        dividend_yield: 0.012,
        max_positions: args.max_positions,
        max_loss_per_position: 25000.0,
        max_portfolio_delta: 500.0,
        min_iv_percentile_sell: 60.0,
        max_iv_percentile_buy: 40.0,
        min_strategy_score: if config.min_score > 0.0 {
            config.min_score
        } else {
            0.0
        },
        ramanujan_enabled: config.ramanujan_enabled,
        block_on_hft: config.block_on_hft,
        pcr_enabled: true,
        pcr_lookback: 100,
    };

    let mut engine = OptionsEngine::new(engine_config);

    // Backtest state
    let mut capital = args.capital;
    let mut positions: HashMap<String, Position> = HashMap::new();
    let mut trades: Vec<Trade> = Vec::new();
    let mut prev_quotes: HashMap<String, QuoteEvent> = HashMap::new();

    // Stats
    let mut hft_blocks = 0u32;
    let mut regime_blocks = 0u32;
    let mut score_blocks = 0u32;
    let mut total_decisions = 0u32;
    let mut entry_signals = 0u32; // Total entry signals from engine
    let mut signals_blocked = 0u32; // Entry signals blocked by gates
    let mut winning_trades = 0u32;
    let mut losing_trades = 0u32;
    let mut peak_equity = args.capital;
    let mut max_drawdown = 0.0f64;
    let mut strategies_used: HashMap<String, u32> = HashMap::new();
    let mut regimes_seen: HashMap<String, u32> = HashMap::new();

    for quote in quotes {
        let features = calculate_features(quote, prev_quotes.get(&quote.tradingsymbol));
        let spot = extract_strike_from_symbol(&quote.tradingsymbol).unwrap_or(25000.0);

        engine.on_tick(quote.ts, spot, &features);

        let decision = engine.decide(quote.ts);
        let status = engine.status();
        total_decisions += 1;

        // Track regime
        let regime_str = format!("{:?}", status.regime);
        *regimes_seen.entry(regime_str.clone()).or_default() += 1;

        // Count entry signals (engine wants to enter)
        let is_entry_signal = matches!(decision.action, TradingAction::Enter);
        if is_entry_signal {
            entry_signals += 1;
        }

        // Check if blocked by HFT
        let hft_blocked = config.block_on_hft && status.hft_detected;
        if hft_blocked && is_entry_signal {
            hft_blocks += 1;
            signals_blocked += 1;
        }

        // Check if blocked by regime (for regime-aware mode)
        let regime_blocked = config.regime_aware && !is_tradeable_regime(&regime_str);
        if regime_blocked && !hft_blocked && is_entry_signal {
            regime_blocks += 1;
            signals_blocked += 1;
        }

        // Check if blocked by score
        let score_blocked =
            config.min_score > 0.0 && decision.confidence * 100.0 < config.min_score;
        if score_blocked && !hft_blocked && !regime_blocked && is_entry_signal {
            score_blocks += 1;
            signals_blocked += 1;
        }

        // Process decision
        let should_trade = !hft_blocked
            && !regime_blocked
            && (config.min_score == 0.0 || decision.confidence * 100.0 >= config.min_score);

        if should_trade {
            match decision.action {
                TradingAction::Enter => {
                    if let Some(rec) = &decision.strategy {
                        let has_position = positions.contains_key(&quote.tradingsymbol);
                        if !has_position && positions.len() < args.max_positions as usize {
                            let entry_price = quote.mid();

                            positions.insert(
                                quote.tradingsymbol.clone(),
                                Position {
                                    symbol: quote.tradingsymbol.clone(),
                                    strategy: rec.strategy,
                                    entry_price,
                                    quantity: 50,
                                    current_price: entry_price,
                                    pnl: 0.0,
                                },
                            );

                            trades.push(Trade {
                                strategy: rec.strategy,
                                action: "ENTER".into(),
                                pnl: 0.0,
                                regime: regime_str.clone(),
                            });

                            *strategies_used
                                .entry(format!("{:?}", rec.strategy))
                                .or_default() += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        // Update MTM
        for pos in positions.values_mut() {
            if pos.symbol == quote.tradingsymbol {
                pos.current_price = quote.mid();
                pos.pnl = (pos.current_price - pos.entry_price) * pos.quantity as f64;
            }
        }

        // Exit logic
        let symbols_to_close: Vec<String> = positions
            .iter()
            .filter(|(_, pos)| pos.pnl < -5000.0 || pos.pnl > 2500.0)
            .map(|(sym, _)| sym.clone())
            .collect();

        for sym in symbols_to_close {
            if let Some(pos) = positions.remove(&sym) {
                trades.push(Trade {
                    strategy: pos.strategy,
                    action: if pos.pnl > 0.0 {
                        "EXIT_PROFIT".into()
                    } else {
                        "EXIT_STOP".into()
                    },
                    pnl: pos.pnl,
                    regime: regime_str.clone(),
                });

                capital += pos.pnl;

                if pos.pnl > 0.0 {
                    winning_trades += 1;
                } else {
                    losing_trades += 1;
                }
            }
        }

        // Track drawdown
        let positions_value: f64 = positions.values().map(|p| p.pnl).sum();
        let equity = capital + positions_value;
        if equity > peak_equity {
            peak_equity = equity;
        }
        let drawdown = (peak_equity - equity) / peak_equity * 100.0;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }

        prev_quotes.insert(quote.tradingsymbol.clone(), quote.clone());
    }

    // Calculate final stats
    let open_pnl: f64 = positions.values().map(|p| p.pnl).sum();
    let realized_pnl: f64 = trades
        .iter()
        .filter(|t| t.action.starts_with("EXIT"))
        .map(|t| t.pnl)
        .sum();
    let total_pnl = realized_pnl + open_pnl;
    let trades_opened = trades.iter().filter(|t| t.action == "ENTER").count() as u32;
    let trades_closed = winning_trades + losing_trades;
    let win_rate = if trades_closed > 0 {
        winning_trades as f64 / trades_closed as f64 * 100.0
    } else {
        0.0
    };

    // Print summary for this config
    let block_rate = if entry_signals > 0 {
        signals_blocked as f64 / entry_signals as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  Entry signals: {} total, {} blocked ({:.1}%)",
        entry_signals, signals_blocked, block_rate
    );
    println!(
        "  Trades: {} opened, {} closed",
        trades_opened, trades_closed
    );
    println!(
        "  P&L: ₹{:.2} (realized) + ₹{:.2} (open) = ₹{:.2}",
        realized_pnl, open_pnl, total_pnl
    );
    println!(
        "  Blocks: {} HFT, {} regime, {} score",
        hft_blocks, regime_blocks, score_blocks
    );

    Ok(BacktestResult {
        name: config.name.to_string(),
        trades_opened,
        trades_closed,
        winning_trades,
        losing_trades,
        realized_pnl,
        open_pnl,
        total_pnl,
        return_pct: total_pnl / args.capital * 100.0,
        max_drawdown_pct: max_drawdown,
        win_rate_pct: win_rate,
        hft_blocks,
        regime_blocks,
        score_blocks,
        total_decisions,
        entry_signals,
        signals_blocked,
        strategies_used,
        regimes_seen,
    })
}

fn is_tradeable_regime(regime: &str) -> bool {
    // Only trade in favorable regimes
    matches!(
        regime,
        "Quiet" | "MeanReversionChop" | "Trending" | "Unknown"
    )
}

fn print_comparison_table(results: &[BacktestResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                           GATE COMPARISON RESULTS                                                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Configuration              │ Signals │ Blocked │ Block% │ Trades │ Total P&L      │ Return │ MaxDD  │ HFT   │ Regime ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for r in results {
        let name_short = if r.name.len() > 26 {
            &r.name[..26]
        } else {
            &r.name
        };
        let block_pct = if r.entry_signals > 0 {
            r.signals_blocked as f64 / r.entry_signals as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "║ {:<26} │ {:>7} │ {:>7} │ {:>5.1}% │ {:>6} │ ₹{:>12.2} │ {:>5.2}% │ {:>5.2}% │ {:>5} │ {:>6} ║",
            name_short,
            r.entry_signals,
            r.signals_blocked,
            block_pct,
            r.trades_opened,
            r.total_pnl,
            r.return_pct,
            r.max_drawdown_pct,
            r.hft_blocks,
            r.regime_blocks
        );
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");
}

fn print_gate_analysis(results: &[BacktestResult]) {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    GATE EFFECTIVENESS ANALYSIS                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Baseline is index 0
    let baseline = &results[0];

    println!("BASELINE (No Gates):");
    println!("  Entry signals:  {}", baseline.entry_signals);
    println!("  Trades opened:  {}", baseline.trades_opened);
    println!("  Total P&L:      ₹{:.2}", baseline.total_pnl);
    println!();

    // Calculate gate effects
    for r in results.iter().skip(1) {
        let signal_block_rate = if baseline.entry_signals > 0 {
            r.signals_blocked as f64 / baseline.entry_signals as f64 * 100.0
        } else {
            0.0
        };

        let pnl_change = r.total_pnl - baseline.total_pnl;
        let pnl_change_str = if pnl_change >= 0.0 {
            format!("+₹{:.2}", pnl_change)
        } else {
            format!("-₹{:.2}", pnl_change.abs())
        };

        let dd_change = baseline.max_drawdown_pct - r.max_drawdown_pct;
        let dd_str = if dd_change >= 0.0 {
            format!("-{:.2}% (better)", dd_change)
        } else {
            format!("+{:.2}% (worse)", dd_change.abs())
        };

        println!("{}:", r.name);
        println!(
            "  Signals blocked:  {} of {} ({:.1}%)",
            r.signals_blocked, baseline.entry_signals, signal_block_rate
        );
        println!("    - HFT blocked:    {}", r.hft_blocks);
        println!("    - Regime blocked: {}", r.regime_blocks);
        println!("    - Score blocked:  {}", r.score_blocks);
        println!(
            "  Trades opened:    {} (vs {} baseline)",
            r.trades_opened, baseline.trades_opened
        );
        println!(
            "  P&L change:       {} (₹{:.2} → ₹{:.2})",
            pnl_change_str, baseline.total_pnl, r.total_pnl
        );
        println!("  Drawdown change:  {}", dd_str);
        println!();
    }

    // Summary table - show signal reduction (gates work inside engine)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("GATE CONTRIBUTION SUMMARY (Signal Generation Impact):");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Gates work INSIDE the engine, reducing signal generation:");
    println!();

    // Find the configs for individual gates
    let ramanujan_only = results
        .iter()
        .find(|r| r.name.contains("Ramanujan HFT Gate"));
    let grassmann_only = results
        .iter()
        .find(|r| r.name.contains("Grassmann Regime Gate"));
    let score_only = results.iter().find(|r| r.name.contains("Score Gate Only"));

    if let Some(r) = ramanujan_only {
        let reduction = if baseline.entry_signals > 0 {
            (1.0 - r.entry_signals as f64 / baseline.entry_signals as f64) * 100.0
        } else {
            0.0
        };
        println!("  Ramanujan (HFT Detection):");
        println!(
            "    Signals: {} → {} ({:.1}% REDUCTION)",
            baseline.entry_signals, r.entry_signals, reduction
        );
        println!("    Effect:  Blocks trading when HFT/market-maker activity detected");
    }

    if let Some(r) = grassmann_only {
        let reduction = if baseline.entry_signals > 0 {
            (1.0 - r.entry_signals as f64 / baseline.entry_signals as f64) * 100.0
        } else {
            0.0
        };
        println!();
        println!("  Grassmann (Regime Detection):");
        println!(
            "    Signals: {} → {} ({:.1}% reduction)",
            baseline.entry_signals, r.entry_signals, reduction
        );
        println!("    Effect:  Filters based on market regime (Quiet, Trending, etc.)");
    }

    if let Some(r) = score_only {
        let reduction = if baseline.entry_signals > 0 {
            (1.0 - r.entry_signals as f64 / baseline.entry_signals as f64) * 100.0
        } else {
            0.0
        };
        println!();
        println!("  Score Threshold:");
        println!(
            "    Signals: {} → {} ({:.1}% reduction)",
            baseline.entry_signals, r.entry_signals, reduction
        );
        println!("    Effect:  Only allows high-confidence signals (>{})", 55); // TODO: pass actual min_score
    }

    // Best configuration
    let best = results
        .iter()
        .max_by(|a, b| a.total_pnl.partial_cmp(&b.total_pnl).unwrap())
        .unwrap();

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BEST CONFIGURATION: {}", best.name);
    println!(
        "  Total P&L:     ₹{:.2} ({:+.2}%)",
        best.total_pnl, best.return_pct
    );
    println!(
        "  Entry signals: {} ({} blocked)",
        best.entry_signals, best.signals_blocked
    );
    println!("  Trades opened: {}", best.trades_opened);
    println!("  Max Drawdown:  {:.2}%", best.max_drawdown_pct);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
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
        (((quote.ask_f64() - quote.bid_f64()) / mid) * 10000.0) as i64
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
