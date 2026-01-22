//! # Session Replay Runner for HYDRA Backtest
//!
//! Replays captured JSONL session data through the Hydra strategy.
//!
//! ## Usage
//! ```bash
//! cargo run --bin replay_session -- --session data/sessions/profile1_2h_20260122_2224
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use kubera_core::{EventBus, HydraStrategy, Portfolio, Strategy};
use kubera_executor::{CommissionModel, RiskEnvelope, SimulatedExchange};
use kubera_models::{DepthEvent, L2Level, L2Snapshot, L2Update, MarketEvent, MarketPayload, OrderEvent, OrderPayload, Side};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct SessionManifest {
    session_id: String,
    symbols: Vec<String>,
    captures: Vec<CaptureInfo>,
}

#[derive(Debug, Deserialize)]
struct CaptureInfo {
    symbol: String,
    depth_file: String,
    trades_file: Option<String>,
    events_written: u64,
    trades_written: Option<u64>,
}

/// Trade event from JSONL (matches binance_sbe_trades_capture format)
#[derive(Debug, Deserialize)]
struct TradeEvent {
    ts: chrono::DateTime<Utc>,
    tradingsymbol: String,
    #[serde(default)]
    trade_id: i64,
    price: i64,
    qty: i64,
    price_exponent: i8,
    qty_exponent: i8,
    is_buyer_maker: bool,
}

/// Wrapper for sorting events by timestamp (min-heap)
struct TimestampedEvent {
    event: MarketEvent,
}

impl PartialEq for TimestampedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.event.exchange_time == other.event.exchange_time
    }
}

impl Eq for TimestampedEvent {}

impl PartialOrd for TimestampedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimestampedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (earliest first)
        other.event.exchange_time.cmp(&self.event.exchange_time)
    }
}

/// Convert DepthEvent to MarketEvent
fn depth_to_market_event(depth: &DepthEvent) -> MarketEvent {
    let price_exp = depth.price_exponent;
    let qty_exp = depth.qty_exponent;

    let convert_levels = |levels: &[kubera_models::DepthLevel]| -> Vec<L2Level> {
        levels
            .iter()
            .map(|l| {
                let (price, size) = l.to_f64(price_exp, qty_exp);
                L2Level { price, size }
            })
            .collect()
    };

    let payload = if depth.is_snapshot {
        MarketPayload::L2Snapshot(L2Snapshot {
            bids: convert_levels(&depth.bids),
            asks: convert_levels(&depth.asks),
            update_id: depth.last_update_id,
        })
    } else {
        MarketPayload::L2Update(L2Update {
            bids: convert_levels(&depth.bids),
            asks: convert_levels(&depth.asks),
            first_update_id: depth.first_update_id,
            last_update_id: depth.last_update_id,
        })
    };

    MarketEvent {
        exchange_time: depth.ts,
        local_time: depth.ts, // Use exchange time for replay
        symbol: depth.tradingsymbol.clone(),
        payload,
    }
}

/// Convert TradeEvent to MarketEvent
fn trade_to_market_event(trade: &TradeEvent) -> MarketEvent {
    let price = trade.price as f64 * 10f64.powi(trade.price_exponent as i32);
    let quantity = trade.qty as f64 * 10f64.powi(trade.qty_exponent as i32);

    MarketEvent {
        exchange_time: trade.ts,
        local_time: trade.ts,
        symbol: trade.tradingsymbol.clone(),
        payload: MarketPayload::Trade {
            trade_id: trade.trade_id,
            price,
            quantity,
            is_buyer_maker: trade.is_buyer_maker,
        },
    }
}

/// Load all depth events from a JSONL file
fn load_depth_events(path: &PathBuf) -> Result<Vec<DepthEvent>> {
    let file = File::open(path).context(format!("Failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context(format!("Failed to read line {}", line_num))?;
        if line.trim().is_empty() {
            continue;
        }
        let event: DepthEvent = serde_json::from_str(&line)
            .context(format!("Failed to parse line {}: {}", line_num, &line[..line.len().min(100)]))?;
        events.push(event);
    }

    Ok(events)
}

/// Load all trade events from a JSONL file
fn load_trade_events(path: &PathBuf) -> Result<Vec<TradeEvent>> {
    let file = File::open(path).context(format!("Failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context(format!("Failed to read line {}", line_num))?;
        if line.trim().is_empty() {
            continue;
        }
        let event: TradeEvent = serde_json::from_str(&line)
            .context(format!("Failed to parse trade line {}: {}", line_num, &line[..line.len().min(100)]))?;
        events.push(event);
    }

    Ok(events)
}

/// Fill record for CSV export (VectorBT Pro compatible)
#[derive(Debug, Clone, Serialize)]
struct FillRecord {
    timestamp: DateTime<Utc>,
    symbol: String,
    side: String,
    quantity: f64,
    price: f64,
    commission: f64,
    position_after: f64,
    unrealized_pnl: f64,
}

/// Export fills to CSV for VectorBT Pro analysis
fn export_fills_csv(fills: &[FillRecord], path: &PathBuf) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "timestamp,symbol,side,quantity,price,commission,position_after,unrealized_pnl")?;
    for f in fills {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            f.timestamp.to_rfc3339(),
            f.symbol,
            f.side,
            f.quantity,
            f.price,
            f.commission,
            f.position_after,
            f.unrealized_pnl
        )?;
    }
    Ok(())
}

/// Equity snapshot for time-indexed MTM curve (VectorBT compatible)
#[derive(Debug, Clone, Serialize)]
struct EquitySample {
    timestamp: DateTime<Utc>,
    equity: f64,
    cash: f64,
    gross_notional: f64,
    unrealized_pnl: f64,
}

/// Export equity curve to CSV for VectorBT Pro analysis
fn export_equity_csv(samples: &[EquitySample], path: &PathBuf) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "timestamp,equity,cash,gross_notional,unrealized_pnl")?;
    for s in samples {
        writeln!(
            file,
            "{},{},{},{},{}",
            s.timestamp.to_rfc3339(),
            s.equity,
            s.cash,
            s.gross_notional,
            s.unrealized_pnl
        )?;
    }
    Ok(())
}

#[derive(Default)]
struct BacktestMetrics {
    market_events: u64,
    signals_generated: u64,
    fills_executed: u64,
    equity_curve: Vec<f64>,
    trade_pnls: Vec<f64>,
    peak_equity: f64,
    max_drawdown: f64,
}

impl BacktestMetrics {
    fn new(initial_capital: f64) -> Self {
        Self {
            equity_curve: vec![initial_capital],
            peak_equity: initial_capital,
            ..Default::default()
        }
    }

    fn update_equity(&mut self, current_equity: f64) {
        self.equity_curve.push(current_equity);
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }
        let drawdown = (self.peak_equity - current_equity) / self.peak_equity;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }
    }

    fn calculate_sharpe(&self) -> f64 {
        if self.equity_curve.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;
        let std_return = variance.sqrt();

        if std_return > 0.0 {
            // Annualize assuming ~500k events per day (crypto 24/7)
            (mean_return / std_return) * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }

    fn win_rate(&self) -> f64 {
        if self.trade_pnls.is_empty() {
            return 0.0;
        }
        let winners = self.trade_pnls.iter().filter(|&&p| p > 0.0).count();
        winners as f64 / self.trade_pnls.len() as f64 * 100.0
    }

    fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.trade_pnls.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = self.trade_pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
        if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let session_path = if args.len() > 2 && args[1] == "--session" {
        PathBuf::from(&args[2])
    } else {
        // Default to our captured session
        PathBuf::from("data/sessions/profile1_2h_20260122_2224")
    };

    info!(">>> HYDRA Session Replay <<<");
    info!("Session: {:?}", session_path);

    // Load manifest
    let manifest_path = session_path.join("session_manifest.json");
    let manifest: SessionManifest = serde_json::from_reader(
        File::open(&manifest_path).context("Failed to open session manifest")?,
    )?;

    info!("Session ID: {}", manifest.session_id);
    info!("Symbols: {:?}", manifest.symbols);

    // Load all events into a priority queue (min-heap by timestamp)
    let mut event_heap: BinaryHeap<TimestampedEvent> = BinaryHeap::new();
    let mut total_events = 0u64;

    let mut total_trades = 0u64;

    for capture in &manifest.captures {
        // Load depth events
        let depth_path = session_path.join(&capture.depth_file);
        info!("Loading {} depth ({} events)...", capture.symbol, capture.events_written);

        let depth_events = load_depth_events(&depth_path)?;
        for depth in depth_events {
            let market_event = depth_to_market_event(&depth);
            event_heap.push(TimestampedEvent { event: market_event });
            total_events += 1;
        }

        // Load trade events (these trigger Hydra signals)
        if let Some(trades_file) = &capture.trades_file {
            let trades_path = session_path.join(trades_file);
            let trade_count = capture.trades_written.unwrap_or(0);
            info!("Loading {} trades ({} events)...", capture.symbol, trade_count);

            let trade_events = load_trade_events(&trades_path)?;
            for trade in trade_events {
                let market_event = trade_to_market_event(&trade);
                event_heap.push(TimestampedEvent { event: market_event });
                total_trades += 1;
            }
        }
    }

    info!("Loaded {} depth + {} trades = {} total events across {} symbols",
          total_events, total_trades, total_events + total_trades, manifest.symbols.len());
    total_events += total_trades;

    // Initialize backtest components
    let initial_capital = 100_000.0;
    let bus = EventBus::new(100_000);
    let mut strategy = HydraStrategy::new();
    strategy.on_start(bus.clone());

    // Create exchange with risk envelope sized for our capital
    // - max_gross_notional = 2x equity ($200k)
    // - max_symbol_notional = 0.5x equity ($50k)
    // - max_order_notional = 0.05x equity ($5k)
    let risk_envelope = RiskEnvelope::for_equity(initial_capital);
    info!("Risk Envelope: max_gross=${:.0}, max_symbol=${:.0}, max_order=${:.0}",
          risk_envelope.max_gross_notional_usd,
          risk_envelope.max_symbol_notional_usd,
          risk_envelope.max_order_notional_usd);

    let mut exchange = SimulatedExchange::with_risk_envelope(
        bus.clone(),
        0.0,
        CommissionModel::None,
        None,
        risk_envelope,
    );
    let mut portfolio = Portfolio::new();
    portfolio.balance = initial_capital;

    let mut signal_rx = bus.subscribe_signal();
    let mut fill_rx = bus.subscribe_fill();

    let mut metrics = BacktestMetrics::new(initial_capital);
    let mut last_prices: HashMap<String, f64> = HashMap::new();
    let mut last_fill_info: Option<(Side, f64)> = None;
    let mut fill_records: Vec<FillRecord> = Vec::new();
    let mut positions: HashMap<String, f64> = HashMap::new();

    // Equity curve sampling (every 1 second of market time)
    let mut equity_samples: Vec<EquitySample> = Vec::new();
    let mut last_sample_ts: Option<DateTime<Utc>> = None;
    let sample_interval = chrono::Duration::seconds(1);

    // Progress tracking
    let progress_interval = total_events / 20; // 5% intervals
    let start_time = std::time::Instant::now();

    // Replay loop
    while let Some(timestamped) = event_heap.pop() {
        let event = timestamped.event;
        metrics.market_events += 1;

        // Track last price for equity calculation
        match &event.payload {
            MarketPayload::L2Snapshot(snap) => {
                if let Some(best_bid) = snap.bids.first() {
                    last_prices.insert(event.symbol.clone(), best_bid.price);
                }
            }
            MarketPayload::L2Update(update) => {
                if let Some(best_bid) = update.bids.iter().find(|l| l.size > 0.0) {
                    last_prices.insert(event.symbol.clone(), best_bid.price);
                }
            }
            MarketPayload::Tick { price, .. } | MarketPayload::Trade { price, .. } => {
                last_prices.insert(event.symbol.clone(), *price);
            }
            _ => {}
        }

        // Sample equity every 1 second of market time
        let should_sample = match last_sample_ts {
            None => true,
            Some(last_ts) => event.exchange_time >= last_ts + sample_interval,
        };

        if should_sample && !last_prices.is_empty() {
            // Calculate current equity (cash + MTM positions)
            let gross_notional: f64 = positions.iter().map(|(sym, pos)| {
                let price = last_prices.get(sym).copied().unwrap_or(0.0);
                pos.abs() * price
            }).sum();

            let unrealized_pnl: f64 = positions.iter().map(|(sym, pos)| {
                let price = last_prices.get(sym).copied().unwrap_or(0.0);
                // Simplified: assume entry at current price for now
                // Real implementation would track avg entry price per symbol
                pos * price
            }).sum();

            let equity = portfolio.calculate_total_value(&last_prices);

            equity_samples.push(EquitySample {
                timestamp: event.exchange_time,
                equity,
                cash: portfolio.balance,
                gross_notional,
                unrealized_pnl,
            });

            last_sample_ts = Some(event.exchange_time);
        }

        // 1. Feed strategy
        strategy.on_tick(&event);

        // 2. Feed exchange
        exchange.on_market_data(event.clone()).await?;

        // 3. Process signals
        while let Ok(signal) = signal_rx.try_recv() {
            metrics.signals_generated += 1;

            exchange
                .handle_order(OrderEvent {
                    order_id: Uuid::new_v4(),
                    intent_id: signal.intent_id,
                    timestamp: Utc::now(),
                    symbol: signal.symbol.clone(),
                    side: signal.side,
                    payload: OrderPayload::New {
                        symbol: signal.symbol.clone(),
                        side: signal.side,
                        quantity: signal.quantity,
                        price: None,
                        order_type: kubera_models::OrderType::Market,
                    },
                })
                .await?;
        }

        // 4. Process fills
        while let Ok(fill) = fill_rx.try_recv() {
            metrics.fills_executed += 1;
            portfolio.apply_fill(&fill.symbol, fill.side, fill.quantity, fill.price, fill.commission);

            // Track position per symbol
            let pos_delta = match fill.side {
                Side::Buy => fill.quantity,
                Side::Sell => -fill.quantity,
            };
            let pos = positions.entry(fill.symbol.clone()).or_insert(0.0);
            *pos += pos_delta;

            // Calculate unrealized PnL
            let current_price = last_prices.get(&fill.symbol).copied().unwrap_or(fill.price);
            let unrealized = *pos * (current_price - fill.price);

            // Record fill for CSV export
            fill_records.push(FillRecord {
                timestamp: event.exchange_time,
                symbol: fill.symbol.clone(),
                side: if fill.side == Side::Buy { "BUY".to_string() } else { "SELL".to_string() },
                quantity: fill.quantity,
                price: fill.price,
                commission: fill.commission,
                position_after: *pos,
                unrealized_pnl: unrealized,
            });

            // Calculate trade PnL on position reversal
            if let Some((prev_side, prev_price)) = last_fill_info {
                if prev_side != fill.side {
                    let trade_pnl = match prev_side {
                        Side::Buy => (fill.price - prev_price) * fill.quantity,
                        Side::Sell => (prev_price - fill.price) * fill.quantity,
                    };
                    metrics.trade_pnls.push(trade_pnl);
                }
            }
            last_fill_info = Some((fill.side, fill.price));

            // Update equity
            let current_equity = portfolio.calculate_total_value(&last_prices);
            metrics.update_equity(current_equity);

            // Feed back to strategy
            strategy.on_fill(&OrderEvent {
                order_id: Uuid::new_v4(),
                intent_id: fill.intent_id,
                timestamp: Utc::now(),
                symbol: fill.symbol.clone(),
                side: fill.side,
                payload: OrderPayload::Update {
                    status: kubera_models::OrderStatus::Filled,
                    filled_quantity: fill.quantity,
                    avg_price: fill.price,
                    commission: fill.commission,
                },
            });
        }

        // Progress logging
        if progress_interval > 0 && metrics.market_events % progress_interval == 0 {
            let pct = (metrics.market_events as f64 / total_events as f64) * 100.0;
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = metrics.market_events as f64 / elapsed;
            info!(
                "Progress: {:.0}% ({} events, {:.0} ev/s, {} signals)",
                pct, metrics.market_events, rate, metrics.signals_generated
            );
        }
    }

    // Final results
    let final_equity = portfolio.calculate_total_value(&last_prices);
    let total_return = (final_equity - initial_capital) / initial_capital * 100.0;
    let elapsed = start_time.elapsed();

    println!("\n{}", "=".repeat(70));
    println!("         KUBERA HYDRA - SESSION REPLAY RESULTS");
    println!("{}", "=".repeat(70));
    println!();
    println!("Session: {}", manifest.session_id);
    println!("Symbols: {:?}", manifest.symbols);
    println!();
    println!("REPLAY STATS:");
    println!("  Market Events:    {:>12}", metrics.market_events);
    println!("  Signals Generated:{:>12}", metrics.signals_generated);
    println!("  Fills Executed:   {:>12}", metrics.fills_executed);
    println!("  Replay Time:      {:>12.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput:       {:>12.0} events/sec",
        metrics.market_events as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("PERFORMANCE:");
    println!("  Initial Capital:  ${:>11.2}", initial_capital);
    println!("  Final Equity:     ${:>11.2}", final_equity);
    println!("  Total Return:     {:>11.2}%", total_return);
    println!("  Max Drawdown:     {:>11.2}%", metrics.max_drawdown * 100.0);
    println!();
    println!("TRADE METRICS:");
    println!("  Total Trades:     {:>12}", metrics.trade_pnls.len());
    println!("  Win Rate:         {:>11.2}%", metrics.win_rate());
    println!("  Profit Factor:    {:>12.2}", metrics.profit_factor());
    println!("  Sharpe Ratio:     {:>12.2}", metrics.calculate_sharpe());
    println!();

    // Show final positions (the source of unrealized PnL)
    println!("FINAL POSITIONS:");
    for (symbol, pos) in &positions {
        if pos.abs() > 1e-10 {
            let price = last_prices.get(symbol).copied().unwrap_or(0.0);
            let notional = pos * price;
            println!("  {}: {:>12.6} (notional: ${:.2})", symbol, pos, notional);
        }
    }
    println!();

    // Risk envelope summary
    let risk_events = exchange.risk_events();
    let clipped_count = risk_events.iter().filter(|e| e.action == "CLIP").count();
    let rejected_count = risk_events.iter().filter(|e| e.action == "REJECT").count();

    println!("RISK ENVELOPE:");
    println!("  Total Risk Events:{:>12}", risk_events.len());
    println!("  Orders Clipped:   {:>12}", clipped_count);
    println!("  Orders Rejected:  {:>12}", rejected_count);

    if clipped_count > 0 || rejected_count > 0 {
        println!();
        println!("  Risk Events by Rule:");
        let max_order_events = risk_events.iter().filter(|e| e.rule_triggered == "MAX_ORDER").count();
        let max_symbol_events = risk_events.iter().filter(|e| e.rule_triggered == "MAX_SYMBOL").count();
        let max_gross_events = risk_events.iter().filter(|e| e.rule_triggered == "MAX_GROSS").count();
        if max_order_events > 0 { println!("    MAX_ORDER:      {:>12}", max_order_events); }
        if max_symbol_events > 0 { println!("    MAX_SYMBOL:     {:>12}", max_symbol_events); }
        if max_gross_events > 0 { println!("    MAX_GROSS:      {:>12}", max_gross_events); }
    }
    println!();

    // Export fills to CSV for VectorBT Pro analysis
    let fills_csv_path = session_path.join("fills_backtest.csv");
    export_fills_csv(&fill_records, &fills_csv_path)?;
    println!("Fills exported to: {:?}", fills_csv_path);

    // Export equity curve to CSV for VectorBT Pro analysis
    let equity_csv_path = session_path.join("equity_curve.csv");
    export_equity_csv(&equity_samples, &equity_csv_path)?;
    println!("Equity curve exported to: {:?} ({} samples at 1s intervals)", equity_csv_path, equity_samples.len());

    println!();
    println!("{}", "=".repeat(70));

    Ok(())
}
