//! # India Session Replay Runner for HYDRA Backtest
//!
//! Replays captured India/Zerodha TickEvent JSONL through the Hydra strategy.
//!
//! ## Usage
//! ```bash
//! cargo run --release -p quantlaxmi-runner-india --bin replay_india_session -- \
//!     --session data/sessions/nifty_banknifty_20260123_1002
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use kubera_core::{EventBus, NullObserverStrategy, Portfolio, Strategy};
use kubera_executor::{CommissionModel, RiskEnvelope, SimulatedExchange};
use kubera_models::{L2Level, L2Snapshot, MarketEvent, MarketPayload, OrderEvent, OrderPayload, Side};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;

/// India session manifest format
#[derive(Debug, Deserialize)]
struct IndiaSessionManifest {
    session_id: String,
    instruments: Vec<String>,
    captures: Vec<IndiaCaptureInfo>,
    #[serde(default)]
    duration_secs: f64,
}

#[derive(Debug, Deserialize)]
struct IndiaCaptureInfo {
    tradingsymbol: String,
    #[serde(default)]
    instrument_token: u32,
    ticks_file: String,
    ticks_written: usize,
    #[serde(default)]
    has_depth: bool,
}

/// TickEvent from India session capture (L1 bid/ask with LTP)
#[derive(Debug, Deserialize)]
struct TickEvent {
    ts: DateTime<Utc>,
    tradingsymbol: String,
    #[serde(default)]
    instrument_token: u32,
    bid_price: i64,
    ask_price: i64,
    bid_qty: u32,
    ask_qty: u32,
    ltp: i64,
    #[serde(default)]
    ltq: u32,
    #[serde(default)]
    volume: u64,
    #[serde(default = "default_price_exponent")]
    price_exponent: i8,
    #[serde(default)]
    integrity_tier: String,
}

fn default_price_exponent() -> i8 {
    -2
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

/// Convert TickEvent to MarketEvent
fn tick_to_market_event(tick: &TickEvent) -> MarketEvent {
    let price_mult = 10f64.powi(tick.price_exponent as i32);

    let bid_price = tick.bid_price as f64 * price_mult;
    let ask_price = tick.ask_price as f64 * price_mult;
    let _ltp = tick.ltp as f64 * price_mult; // Available for future use

    // Create L2 snapshot with single level (L1 data)
    let payload = MarketPayload::L2Snapshot(L2Snapshot {
        bids: vec![L2Level {
            price: bid_price,
            size: tick.bid_qty as f64,
        }],
        asks: vec![L2Level {
            price: ask_price,
            size: tick.ask_qty as f64,
        }],
        update_id: tick.volume, // Use volume as update_id
    });

    MarketEvent {
        exchange_time: tick.ts,
        local_time: tick.ts,
        symbol: tick.tradingsymbol.clone(),
        payload,
    }
}

/// Load all tick events from a JSONL file
fn load_tick_events(path: &PathBuf) -> Result<Vec<TickEvent>> {
    let file = File::open(path).context(format!("Failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.context(format!("Failed to read line {}", line_num))?;
        if line.trim().is_empty() {
            continue;
        }
        let event: TickEvent = serde_json::from_str(&line)
            .context(format!("Failed to parse line {}: {}", line_num, &line[..line.len().min(100)]))?;
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
            / (returns.len() - 1).max(1) as f64;
        let std_return = variance.sqrt();

        if std_return > 0.0 {
            // Annualize assuming ~375 trading days for India (accounting for holidays)
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
        eprintln!("Usage: replay_india_session --session <path>");
        eprintln!("Example: replay_india_session --session data/sessions/nifty_banknifty_20260123_1002");
        std::process::exit(1);
    };

    info!(">>> NullObserver India Session Replay <<<");
    info!("Session: {:?}", session_path);

    // Load manifest
    let manifest_path = session_path.join("session_manifest.json");
    let manifest: IndiaSessionManifest = serde_json::from_reader(
        File::open(&manifest_path).context("Failed to open session manifest")?,
    )?;

    info!("Session ID: {}", manifest.session_id);
    info!("Instruments: {} total", manifest.instruments.len());
    info!("Duration: {:.1}s", manifest.duration_secs);

    // Load all events into a priority queue (min-heap by timestamp)
    let mut event_heap: BinaryHeap<TimestampedEvent> = BinaryHeap::new();
    let mut total_events = 0u64;

    for capture in &manifest.captures {
        let ticks_path = session_path.join(&capture.ticks_file);
        info!("Loading {} ({} ticks)...", capture.tradingsymbol, capture.ticks_written);

        let tick_events = load_tick_events(&ticks_path)?;
        for tick in tick_events {
            let market_event = tick_to_market_event(&tick);
            event_heap.push(TimestampedEvent { event: market_event });
            total_events += 1;
        }
    }

    info!("Loaded {} total tick events across {} instruments",
          total_events, manifest.instruments.len());

    // Initialize backtest components
    // Use INR capital (₹10 lakh = ~$12k USD)
    let initial_capital = 1_000_000.0; // ₹10 lakh
    let bus = EventBus::new(100_000);
    let mut strategy = NullObserverStrategy::new();
    strategy.on_start(bus.clone());

    // Create exchange with risk envelope sized for our capital
    let risk_envelope = RiskEnvelope::for_equity(initial_capital);
    info!("Risk Envelope: max_gross=₹{:.0}, max_symbol=₹{:.0}, max_order=₹{:.0}",
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
            let gross_notional: f64 = positions.iter().map(|(sym, pos)| {
                let price = last_prices.get(sym).copied().unwrap_or(0.0);
                pos.abs() * price
            }).sum();

            let unrealized_pnl: f64 = positions.iter().map(|(sym, pos)| {
                let price = last_prices.get(sym).copied().unwrap_or(0.0);
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

            let pos_delta = match fill.side {
                Side::Buy => fill.quantity,
                Side::Sell => -fill.quantity,
            };
            let pos = positions.entry(fill.symbol.clone()).or_insert(0.0);
            *pos += pos_delta;

            let current_price = last_prices.get(&fill.symbol).copied().unwrap_or(fill.price);
            let unrealized = *pos * (current_price - fill.price);

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

            let current_equity = portfolio.calculate_total_value(&last_prices);
            metrics.update_equity(current_equity);

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
    let elapsed = start_time.elapsed();

    println!("\n{}", "=".repeat(70));
    println!("     NULLOBSERVER - INDIA SESSION OPTION CHAIN ANALYSIS");
    println!("{}", "=".repeat(70));
    println!();
    println!("Session: {}", manifest.session_id);
    println!("Instruments: {} (NIFTY + BANKNIFTY options)", manifest.instruments.len());
    println!();
    println!("REPLAY STATS:");
    println!("  Market Events:    {:>12}", metrics.market_events);
    println!("  Signals Generated:{:>12} (expected: 0)", metrics.signals_generated);
    println!("  Fills Executed:   {:>12} (expected: 0)", metrics.fills_executed);
    println!("  Replay Time:      {:>12.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput:       {:>12.0} events/sec",
        metrics.market_events as f64 / elapsed.as_secs_f64()
    );
    println!();

    // Generate option chain report from NullObserverStrategy
    let chain_report = strategy.generate_report();
    println!("{}", chain_report);

    // Export fills to CSV for VectorBT Pro analysis (will be empty for NullObserver)
    let fills_csv_path = session_path.join("fills_backtest.csv");
    export_fills_csv(&fill_records, &fills_csv_path)?;
    println!("Fills exported to: {:?}", fills_csv_path);

    // Export equity curve to CSV for VectorBT Pro analysis
    let equity_csv_path = session_path.join("equity_curve.csv");
    export_equity_csv(&equity_samples, &equity_csv_path)?;
    println!("Equity curve exported to: {:?} ({} samples at 1s intervals)", equity_csv_path, equity_samples.len());

    // Export option chain report to JSON
    let chain_report_path = session_path.join("option_chain_report.json");
    let chain_report_json = serde_json::json!({
        "total_events": chain_report.total_events,
        "unique_symbols": chain_report.unique_symbols,
        "expiries": chain_report.expiries.iter().map(|e| {
            serde_json::json!({
                "expiry": e.expiry,
                "total_strikes": e.total_strikes,
                "call_strikes": e.call_strikes,
                "put_strikes": e.put_strikes,
                "missing_calls": e.missing_calls,
                "missing_puts": e.missing_puts,
                "strike_range": [e.strike_range.0, e.strike_range.1]
            })
        }).collect::<Vec<_>>(),
        "atm_spreads": chain_report.atm_spreads.iter().map(|a| {
            serde_json::json!({
                "symbol": a.symbol,
                "strike": a.strike,
                "option_type": format!("{}", a.option_type),
                "avg_spread_bps": a.avg_spread_bps,
                "min_spread_bps": a.min_spread_bps,
                "max_spread_bps": a.max_spread_bps,
                "tick_count": a.tick_count
            })
        }).collect::<Vec<_>>(),
        "underlying_estimates": chain_report.underlying_estimates
    });
    std::fs::write(&chain_report_path, serde_json::to_string_pretty(&chain_report_json)?)?;
    println!("Option chain report exported to: {:?}", chain_report_path);

    println!();
    println!("{}", "=".repeat(70));

    Ok(())
}
