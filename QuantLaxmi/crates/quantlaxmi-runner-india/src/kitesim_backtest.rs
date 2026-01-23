//! KiteSim Offline Backtest Runner (India/Zerodha)
//!
//! Provides a CLI-friendly entrypoint for NSE/BSE backtesting:
//! - load replay events (JSONL of QuoteEvent)
//! - load orders (JSON) or intents (JSON with timestamps)
//! - execute sequentially through MultiLegCoordinator
//! - emit report.json, fills.jsonl, pnl.json
//!
//! ## Isolation
//! This module has NO Binance dependencies. It only supports Zerodha/NSE venues.

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use quantlaxmi_options::execution::{LegSide, LegStatus, MultiLegOrder};
use quantlaxmi_options::kitesim::{
    AtomicExecPolicy, KiteSim, KiteSimConfig, MultiLegCoordinator, SimExecutionMode,
};
use quantlaxmi_options::replay::{DepthEvent, QuoteEvent, ReplayEvent, ReplayFeed};
use quantlaxmi_options::report::{BacktestReport, FillMetrics};
use quantlaxmi_options::specs::SpecStore;

/// Scheduled order intent with timestamp
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OrderIntent {
    pub ts: String,
    pub order: MultiLegOrder,
}

/// File format for scheduled intents
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OrderIntentFile {
    pub strategy_name: String,
    pub intents: Vec<OrderIntent>,
}

/// File format for bulk orders
#[derive(Debug, Clone, serde::Deserialize)]
pub struct OrderFile {
    pub strategy_name: String,
    pub orders: Vec<MultiLegOrder>,
}

pub struct KiteSimCliConfig {
    pub qty_scale: u32,
    pub strategy_name: String,
    pub replay_path: String,
    pub orders_path: String,
    pub intents_path: Option<String>,
    pub depth_path: Option<String>,
    pub out_dir: String,
    pub timeout_ms: i64,
    pub latency_ms: i64,
    pub slippage_bps: f64,
    pub adverse_bps: f64,
    pub stale_quote_ms: i64,
    pub hedge_on_failure: bool,
}

fn parse_rfc3339_utc(s: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(s)?.with_timezone(&Utc))
}

/// Load JSONL quotes. Each line must be a QuoteEvent JSON object.
pub fn load_quotes_jsonl(path: &Path) -> Result<Vec<ReplayEvent>> {
    let f = File::open(path).with_context(|| format!("open replay file: {:?}", path))?;
    let br = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in br.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let q: QuoteEvent = serde_json::from_str(&line)
            .with_context(|| format!("parse QuoteEvent JSON on line {}", i + 1))?;
        out.push(ReplayEvent::Quote(q));
    }
    Ok(out)
}

/// Load JSONL depth events.
pub fn load_depth_jsonl(path: &Path) -> Result<Vec<ReplayEvent>> {
    let f = File::open(path).with_context(|| format!("open depth file: {:?}", path))?;
    let br = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in br.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let d: DepthEvent = serde_json::from_str(&line)
            .with_context(|| format!("parse DepthEvent JSON on line {}", i + 1))?;
        out.push(ReplayEvent::Depth(d));
    }
    Ok(out)
}

pub fn load_orders_json(path: &Path) -> Result<OrderFile> {
    let s =
        std::fs::read_to_string(path).with_context(|| format!("read orders file: {:?}", path))?;
    let of: OrderFile = serde_json::from_str(&s).with_context(|| "parse OrderFile JSON")?;
    Ok(of)
}

pub fn load_intents_json(path: &Path) -> Result<OrderIntentFile> {
    let s =
        std::fs::read_to_string(path).with_context(|| format!("read intents file: {:?}", path))?;
    let intf: OrderIntentFile =
        serde_json::from_str(&s).with_context(|| "parse OrderIntentFile JSON")?;
    Ok(intf)
}

/// Get NSE lot size for a symbol
pub fn get_nse_lot_size(symbol: &str) -> u32 {
    let symbol_upper = symbol.to_uppercase();
    if symbol_upper.starts_with("NIFTY") && !symbol_upper.starts_with("NIFTYBANK") {
        return 65;
    }
    if symbol_upper.starts_with("BANKNIFTY") {
        return 30;
    }
    if symbol_upper.starts_with("FINNIFTY") {
        return 25;
    }
    if symbol_upper.starts_with("MIDCPNIFTY") {
        return 50;
    }
    if symbol_upper.starts_with("SENSEX") {
        return 10;
    }
    if symbol_upper.starts_with("BANKEX") {
        return 15;
    }
    1 // Default for unknown symbols
}

/// Get tick size for NSE symbols
pub fn get_nse_tick_size(_symbol: &str) -> f64 {
    // NSE options and futures have 0.05 tick size
    0.05
}

pub async fn run_kitesim_backtest_cli(cfg: KiteSimCliConfig) -> Result<()> {
    let replay_path = Path::new(&cfg.replay_path);
    let orders_path = Path::new(&cfg.orders_path);
    let out_dir = Path::new(&cfg.out_dir);

    let use_l2_mode = cfg.depth_path.is_some();
    let execution_mode = if use_l2_mode {
        SimExecutionMode::L2Book
    } else {
        SimExecutionMode::L1Quote
    };

    let mut replay_events = load_quotes_jsonl(replay_path)?;

    if let Some(ref depth_path) = cfg.depth_path {
        let depth_events = load_depth_jsonl(Path::new(depth_path))?;
        println!("Loaded {} depth events for L2Book mode", depth_events.len());
        replay_events.extend(depth_events);
    }

    let use_intents = cfg.intents_path.is_some();
    let intents_file = if let Some(ref ip) = cfg.intents_path {
        Some(load_intents_json(Path::new(ip))?)
    } else {
        None
    };
    let order_file = load_orders_json(orders_path)?;

    let strategy_name = if cfg.strategy_name.trim().is_empty() {
        if use_intents {
            intents_file
                .as_ref()
                .map(|f| f.strategy_name.clone())
                .unwrap_or_else(|| order_file.strategy_name.clone())
        } else {
            order_file.strategy_name.clone()
        }
    } else {
        cfg.strategy_name.clone()
    };

    let mut sim = KiteSim::new(KiteSimConfig {
        latency: Duration::milliseconds(cfg.latency_ms),
        allow_partial: true,
        taker_slippage_bps: cfg.slippage_bps,
        adverse_selection_max_bps: cfg.adverse_bps,
        reject_if_no_quote_after: Duration::milliseconds(cfg.stale_quote_ms),
        execution_mode,
    });

    // Build SpecStore with NSE specs (lot sizes + tick sizes)
    let mut specs = SpecStore::new();

    let symbols: std::collections::HashSet<String> = if use_intents {
        intents_file
            .as_ref()
            .map(|f| {
                f.intents
                    .iter()
                    .flat_map(|i| i.order.legs.iter().map(|l| l.tradingsymbol.clone()))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        order_file
            .orders
            .iter()
            .flat_map(|o| o.legs.iter().map(|l| l.tradingsymbol.clone()))
            .collect()
    };

    // Use NSE specs for all symbols
    for sym in &symbols {
        let lot_size = get_nse_lot_size(sym) as i64;
        let tick_size = get_nse_tick_size(sym);
        specs.insert_with_scale(sym, lot_size, tick_size, cfg.qty_scale);
    }
    sim = sim.with_specs(specs);

    let policy = AtomicExecPolicy {
        timeout: Duration::milliseconds(cfg.timeout_ms),
        hedge_on_failure: cfg.hedge_on_failure,
    };

    let mut feed = ReplayFeed::new(replay_events);
    // Store (order, result) pairs to track side info for PnL computation
    let mut all_results: Vec<(MultiLegOrder, quantlaxmi_options::execution::MultiLegResult)> =
        Vec::new();

    if use_intents {
        let intf = intents_file.as_ref().unwrap();
        let mut intents: Vec<_> = intf.intents.iter().collect();
        intents.sort_by_key(|i| parse_rfc3339_utc(&i.ts).ok());

        for intent in intents {
            let target_ts = parse_rfc3339_utc(&intent.ts)?;

            loop {
                let should_consume = feed.peek().map(|ev| ev.ts() < target_ts).unwrap_or(false);
                if !should_consume {
                    break;
                }
                if let Some(ev) = feed.next() {
                    sim.ingest_event(&ev)?;
                }
            }

            sim.set_now(target_ts);
            let mut coord = MultiLegCoordinator::new(&mut sim, policy.clone());
            let res = coord.execute_with_feed(&intent.order, &mut feed).await?;
            all_results.push((intent.order.clone(), res));
        }
    } else {
        for order in order_file.orders.iter() {
            let mut coord = MultiLegCoordinator::new(&mut sim, policy.clone());
            let res = coord.execute_with_feed(order, &mut feed).await?;
            all_results.push((order.clone(), res));
        }
    }

    let stats = sim.stats();

    let fill = FillMetrics {
        orders_total: all_results.len() as u64,
        legs_total: all_results
            .iter()
            .map(|(_, r)| r.leg_results.len() as u64)
            .sum(),
        legs_filled: all_results
            .iter()
            .flat_map(|(_, r)| r.leg_results.iter())
            .filter(|lr| lr.status == LegStatus::Filled)
            .count() as u64,
        legs_partially_filled: all_results
            .iter()
            .flat_map(|(_, r)| r.leg_results.iter())
            .filter(|lr| lr.status == LegStatus::PartiallyFilled)
            .count() as u64,
        legs_rejected: all_results
            .iter()
            .flat_map(|(_, r)| r.leg_results.iter())
            .filter(|lr| lr.status == LegStatus::Rejected)
            .count() as u64,
        legs_cancelled: all_results
            .iter()
            .flat_map(|(_, r)| r.leg_results.iter())
            .filter(|lr| lr.status == LegStatus::Cancelled)
            .count() as u64,
        rollbacks: stats.rollbacks,
        timeouts: stats.timeouts,
        hedges_attempted: stats.hedges_attempted,
        hedges_filled: stats.hedges_filled,
        slippage_bps_p50: quantile(&stats.slippage_samples_bps, 0.50),
        slippage_bps_p90: quantile(&stats.slippage_samples_bps, 0.90),
        slippage_bps_p99: quantile(&stats.slippage_samples_bps, 0.99),
    };

    let mut initial_notes = vec![format!("strategy={}", strategy_name)];
    if use_intents {
        initial_notes.push(format!(
            "intents_file={}",
            cfg.intents_path.as_ref().unwrap()
        ));
    } else {
        initial_notes.push(format!("orders_file={}", orders_path.to_string_lossy()));
    }
    initial_notes.push(format!("qty_scale={}", cfg.qty_scale));

    let mut report = BacktestReport {
        created_at: Utc::now(),
        engine: "KiteSim".to_string(),
        venue: "NSE-Zerodha-Sim".to_string(),
        dataset: replay_path.to_string_lossy().to_string(),
        fill: fill.clone(),
        notes: initial_notes,
    };

    // Compute PnL from fills
    // Side info comes from the order legs (matched by index)
    let mut pnl: f64 = 0.0;
    for (order, res) in &all_results {
        for (i, lr) in res.leg_results.iter().enumerate() {
            if lr.status == LegStatus::Filled || lr.status == LegStatus::PartiallyFilled {
                // Get side from original order leg (legs match by index)
                let side = order.legs.get(i).map(|leg| &leg.side);
                let side_mult = match side {
                    Some(LegSide::Buy) => -1.0,
                    Some(LegSide::Sell) => 1.0,
                    None => 0.0, // Safety: if no matching leg, skip
                };
                let fill_price = lr.fill_price.unwrap_or(0.0);
                pnl += side_mult * fill_price * lr.filled_qty as f64;
            }
        }
    }
    // Note: BacktestReport doesn't have a pnl field, so we store it in notes
    report.notes.push(format!("total_pnl={:.2}", pnl));

    // Write outputs
    std::fs::create_dir_all(out_dir)?;
    let report_path = out_dir.join("report.json");
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&report_path, report_json)?;
    println!("Report written to: {}", report_path.display());

    // Write PnL summary
    let pnl_path = out_dir.join("pnl.json");
    let pnl_json = serde_json::json!({
        "total_pnl": pnl,
        "orders": all_results.len(),
        "legs_filled": fill.legs_filled,
    });
    std::fs::write(&pnl_path, serde_json::to_string_pretty(&pnl_json)?)?;
    println!("PnL summary written to: {}", pnl_path.display());

    println!("\n=== KiteSim Backtest Complete (India/Zerodha) ===");
    println!("Strategy: {}", strategy_name);
    println!(
        "Orders: {}, Legs filled: {}/{}",
        fill.orders_total, fill.legs_filled, fill.legs_total
    );
    println!("PnL: ₹{:.2}", pnl);

    Ok(())
}

fn quantile(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
