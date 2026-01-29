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

use crate::equity_curve::{
    EquityCurveConfig, EquityPoint, MaxDrawdownTracker, OnlineReturnsTracker, floor_to_bar,
};
use crate::india_fees::{
    DailyPnlReport, FeeLedgerRecord, FillFeeBreakdown, IndiaFeeModel, aggregate_fees_for_report,
};
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

/// Extract last quote per symbol from replay events (for MTM valuation)
fn extract_last_quotes(events: &[ReplayEvent]) -> std::collections::HashMap<String, QuoteEvent> {
    let mut last_quotes = std::collections::HashMap::new();
    for ev in events {
        if let ReplayEvent::Quote(q) = ev {
            last_quotes.insert(q.tradingsymbol.clone(), q.clone());
        }
    }
    last_quotes
}

pub async fn run_kitesim_backtest_cli(cfg: KiteSimCliConfig) -> Result<()> {
    let replay_path = Path::new(&cfg.replay_path);
    let orders_path = Path::new(&cfg.orders_path);
    let out_dir = Path::new(&cfg.out_dir);

    // Ensure output directory exists (prevents "No such file or directory" errors)
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("create output dir: {}", out_dir.display()))?;

    let use_l2_mode = cfg.depth_path.is_some();
    let execution_mode = if use_l2_mode {
        SimExecutionMode::L2Book
    } else {
        SimExecutionMode::L1Quote
    };

    let mut replay_events = load_quotes_jsonl(replay_path)?;

    // Gate B0.3: Extract last quotes per symbol for MTM valuation
    let last_quotes = extract_last_quotes(&replay_events);

    if let Some(ref depth_path) = cfg.depth_path {
        let depth_events = load_depth_jsonl(Path::new(depth_path))?;
        tracing::info!("Loaded {} depth events for L2Book mode", depth_events.len());
        replay_events.extend(depth_events);
    }

    // Defensive: ensure replay events are strictly time-ordered
    replay_events.sort_by_key(|ev| ev.ts());

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

    // Extract first quote timestamp for baseline (before creating feed)
    let first_quote_ts: Option<DateTime<Utc>> = replay_events.first().map(|ev| ev.ts());

    let mut feed = ReplayFeed::new(replay_events);
    // Store (order, result) pairs to track side info for PnL computation
    let mut all_results: Vec<(MultiLegOrder, quantlaxmi_options::execution::MultiLegResult)> =
        Vec::new();

    // Equity curve tracking: emit points after each order execution
    let equity_curve_path = out_dir.join("equity_curve.jsonl");
    let mut equity_curve_file = std::io::BufWriter::new(
        File::create(&equity_curve_path)
            .with_context(|| format!("create equity curve file: {:?}", equity_curve_path))?,
    );
    let eq_cfg = EquityCurveConfig { interval_secs: 1 };
    let mut last_bar_ts: Option<DateTime<Utc>> = None;
    let mut mdd_tracker: Option<MaxDrawdownTracker> = None;
    let mut returns_tracker = OnlineReturnsTracker::new();
    let mut equity_bar_count = 0u32;

    // Track first/last equity for run_summary verification
    let mut equity_first_inr: Option<f64> = None;
    let mut equity_last_inr: Option<f64> = None;

    // Emit baseline point (equity=0 before any trades)
    if let Some(first_ts) = first_quote_ts {
        let baseline_ts = first_ts - Duration::milliseconds(1); // Strictly before first quote
        let baseline = EquityPoint::with_components(baseline_ts, 0.0, 0.0);
        use std::io::Write;
        writeln!(equity_curve_file, "{}", serde_json::to_string(&baseline)?)?;
        equity_bar_count += 1;
        equity_first_inr = Some(0.0);
        equity_last_inr = Some(0.0);
        // Initialize MDD tracker at baseline
        mdd_tracker = Some(MaxDrawdownTracker::new(0.0, baseline_ts));
        returns_tracker.update(0.0);
        last_bar_ts = Some(floor_to_bar(baseline_ts, eq_cfg.interval_secs));
    }

    // Track positions and cashflow incrementally during execution
    let mut running_cashflow: f64 = 0.0;
    let mut running_positions: std::collections::BTreeMap<String, i64> =
        std::collections::BTreeMap::new();

    // Helper to compute MTM from positions using latest quotes
    let compute_running_mtm = |positions: &std::collections::BTreeMap<String, i64>,
                               quotes: &std::collections::HashMap<String, QuoteEvent>|
     -> f64 {
        let mut mtm = 0.0;
        for (sym, &qty) in positions {
            if qty == 0 {
                continue;
            }
            if let Some(q) = quotes.get(sym) {
                let mark = if qty > 0 { q.bid_f64() } else { q.ask_f64() };
                if mark.is_finite() && mark > 0.0 {
                    mtm += mark * qty as f64;
                }
            }
        }
        mtm
    };

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

            // Update running positions and cashflow from this execution
            for (i, lr) in res.leg_results.iter().enumerate() {
                if lr.status == LegStatus::Filled || lr.status == LegStatus::PartiallyFilled {
                    if let Some(leg) = intent.order.legs.get(i) {
                        let fill_price = lr.fill_price.unwrap_or(0.0);
                        let fill_qty = lr.filled_qty as i64;
                        match leg.side {
                            LegSide::Buy => {
                                running_cashflow -= fill_price * fill_qty as f64;
                                *running_positions
                                    .entry(leg.tradingsymbol.clone())
                                    .or_insert(0) += fill_qty;
                            }
                            LegSide::Sell => {
                                running_cashflow += fill_price * fill_qty as f64;
                                *running_positions
                                    .entry(leg.tradingsymbol.clone())
                                    .or_insert(0) -= fill_qty;
                            }
                        }
                    }
                }
            }

            // Emit equity point at this timestamp
            let running_mtm = compute_running_mtm(&running_positions, &last_quotes);
            let equity_inr = running_cashflow + running_mtm;
            let bar_ts = floor_to_bar(target_ts, eq_cfg.interval_secs);

            let should_emit = match last_bar_ts {
                None => true,
                Some(prev) => bar_ts > prev,
            };

            if should_emit {
                let mut pt = EquityPoint::with_components(bar_ts, running_cashflow, running_mtm);
                if equity_bar_count == 0 {
                    mdd_tracker = Some(MaxDrawdownTracker::new(equity_inr, bar_ts));
                }
                pt.pnl_inr = Some(equity_inr); // For first point, this is just current equity

                writeln!(equity_curve_file, "{}", serde_json::to_string(&pt)?)?;
                equity_bar_count += 1;
                equity_last_inr = Some(equity_inr);

                if let Some(tr) = mdd_tracker.as_mut() {
                    tr.update(equity_inr, bar_ts);
                }
                returns_tracker.update(equity_inr);

                last_bar_ts = Some(bar_ts);
            }

            all_results.push((intent.order.clone(), res));
        }
    } else {
        let mut order_idx = 0u64;
        for order in order_file.orders.iter() {
            let mut coord = MultiLegCoordinator::new(&mut sim, policy.clone());
            let res = coord.execute_with_feed(order, &mut feed).await?;

            // Update running positions and cashflow
            for (i, lr) in res.leg_results.iter().enumerate() {
                if lr.status == LegStatus::Filled || lr.status == LegStatus::PartiallyFilled {
                    if let Some(leg) = order.legs.get(i) {
                        let fill_price = lr.fill_price.unwrap_or(0.0);
                        let fill_qty = lr.filled_qty as i64;
                        match leg.side {
                            LegSide::Buy => {
                                running_cashflow -= fill_price * fill_qty as f64;
                                *running_positions
                                    .entry(leg.tradingsymbol.clone())
                                    .or_insert(0) += fill_qty;
                            }
                            LegSide::Sell => {
                                running_cashflow += fill_price * fill_qty as f64;
                                *running_positions
                                    .entry(leg.tradingsymbol.clone())
                                    .or_insert(0) -= fill_qty;
                            }
                        }
                    }
                }
            }

            // Emit equity point (use synthetic timestamp for bulk orders)
            let running_mtm = compute_running_mtm(&running_positions, &last_quotes);
            let equity_inr = running_cashflow + running_mtm;
            // For bulk orders, use a synthetic timestamp based on order index
            let synthetic_ts = Utc::now() + Duration::seconds(order_idx as i64);
            let bar_ts = floor_to_bar(synthetic_ts, eq_cfg.interval_secs);

            if equity_bar_count == 0 {
                mdd_tracker = Some(MaxDrawdownTracker::new(equity_inr, bar_ts));
            }

            let mut pt = EquityPoint::with_components(bar_ts, running_cashflow, running_mtm);
            pt.pnl_inr = Some(equity_inr);

            writeln!(equity_curve_file, "{}", serde_json::to_string(&pt)?)?;
            equity_bar_count += 1;
            equity_last_inr = Some(equity_inr);

            if let Some(tr) = mdd_tracker.as_mut() {
                tr.update(equity_inr, bar_ts);
            }
            returns_tracker.update(equity_inr);

            all_results.push((order.clone(), res));
            order_idx += 1;
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

    // Gate B0.3: Compute cashflow, positions, and MTM PnL
    // cashflow = net premium (buys negative, sells positive)
    // positions = net qty per symbol (buys positive, sells negative)
    // Using BTreeMap for deterministic JSON output (gate-grade)
    let mut cashflow: f64 = 0.0;
    let mut positions: std::collections::BTreeMap<String, i64> = std::collections::BTreeMap::new();

    for (order, res) in &all_results {
        for (i, lr) in res.leg_results.iter().enumerate() {
            if lr.status == LegStatus::Filled || lr.status == LegStatus::PartiallyFilled {
                let leg = match order.legs.get(i) {
                    Some(l) => l,
                    None => continue,
                };
                // Use leg.tradingsymbol (ground truth), not lr.tradingsymbol
                let sym = leg.tradingsymbol.clone();
                let fill_price = lr.fill_price.unwrap_or(0.0);
                let fill_qty = lr.filled_qty as i64;

                match leg.side {
                    LegSide::Buy => {
                        cashflow -= fill_price * fill_qty as f64; // Pay premium
                        *positions.entry(sym).or_insert(0) += fill_qty;
                    }
                    LegSide::Sell => {
                        cashflow += fill_price * fill_qty as f64; // Receive premium
                        *positions.entry(sym).or_insert(0) -= fill_qty;
                    }
                }
            }
        }
    }

    // MTM valuation using conservative pricing:
    // - Long positions (pos > 0): value at bid (worst-case liquidation)
    // - Short positions (pos < 0): value at ask (worst-case buyback)
    // Using BTreeMap for deterministic JSON output
    let mut mtm_value: f64 = 0.0;
    let mut eod_marks: std::collections::BTreeMap<String, serde_json::Value> =
        std::collections::BTreeMap::new();
    let mut mtm_warnings: Vec<String> = Vec::new();

    for (sym, &qty) in &positions {
        if qty == 0 {
            continue;
        }
        match last_quotes.get(sym) {
            Some(q) => {
                let bid = q.bid_f64();
                let ask = q.ask_f64();

                // Mark sanity + NaN guard (gate-grade validation)
                if !bid.is_finite() || !ask.is_finite() || ask < bid || bid <= 0.0 {
                    mtm_warnings.push(format!(
                        "invalid last quote for {}: bid={}, ask={}",
                        sym, bid, ask
                    ));
                    continue;
                }

                let mark_price = if qty > 0 { bid } else { ask };
                mtm_value += mark_price * qty as f64;

                eod_marks.insert(
                    sym.clone(),
                    serde_json::json!({
                        "bid": bid,
                        "ask": ask,
                        "mark": mark_price,
                        "qty": qty
                    }),
                );
            }
            None => {
                mtm_warnings.push(format!("missing last quote for {}", sym));
            }
        }
    }

    let mtm_pnl = cashflow + mtm_value;
    let open_positions = positions.values().filter(|&&q| q != 0).count();

    // Gate B0.4 + B0.4.1 + B0.4.2 + M2.1: Maker quality scorecard
    // Slippage convention: (fill_px - mid) for buys, (mid - fill_px) for sells
    // Negative = filled better than mid (maker edge), Positive = worse than mid
    // B0.4.1: Invalid book fills tracked separately, excluded from edge calculation
    // B0.4.2: slip_quote_mode documents quote timing for edge (should be "current")
    // M2.1: queue_priority_blocked tracks fills blocked by queue-ahead
    let slippage_samples = &stats.slippage_samples_bps;
    let invalid_book_fills = stats.invalid_book_fills;
    let queue_consumption_fills = stats.queue_consumption_fills;
    let queue_priority_blocked = stats.queue_priority_blocked;
    let slip_quote_mode = stats.slip_quote_mode;
    let total_fills = slippage_samples.len() as u64 + invalid_book_fills;
    let invalid_fill_pct = if total_fills > 0 {
        (invalid_book_fills as f64 / total_fills as f64) * 100.0
    } else {
        0.0
    };

    let maker_scorecard = if !slippage_samples.is_empty()
        || invalid_book_fills > 0
        || queue_priority_blocked > 0
    {
        let (mean_edge_bps, edge_p50, edge_p90, favorable_pct) = if !slippage_samples.is_empty() {
            let mean_slip: f64 =
                slippage_samples.iter().sum::<f64>() / slippage_samples.len() as f64;
            let favorable_count = slippage_samples.iter().filter(|&&s| s <= 0.0).count();
            let fav_pct = (favorable_count as f64 / slippage_samples.len() as f64) * 100.0;

            // Edge is negative slippage (we got better price than mid)
            // Flip sign: positive edge = good
            (
                -mean_slip,
                -quantile(slippage_samples, 0.50),
                -quantile(slippage_samples, 0.10), // p10 of slippage = p90 of edge
                fav_pct,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        Some(serde_json::json!({
            "valid_book_fills": slippage_samples.len(),
            "invalid_book_fills": invalid_book_fills,
            "invalid_fill_pct": invalid_fill_pct,
            "queue_consumption_fills": queue_consumption_fills,
            "queue_priority_blocked": queue_priority_blocked,
            "slip_quote_mode": slip_quote_mode,
            "mean_edge_bps": mean_edge_bps,
            "edge_p50_bps": edge_p50,
            "edge_p90_bps": edge_p90,
            "favorable_fill_pct": favorable_pct,
            "note": "M2.1: Queue priority model. Edge vs mid at fill time (slip_quote_mode=current)."
        }))
    } else {
        None
    };

    // Note: BacktestReport doesn't have a pnl field, so we store it in notes
    report.notes.push(format!("cashflow={:.2}", cashflow));
    report.notes.push(format!("mtm_value={:.2}", mtm_value));
    report.notes.push(format!("mtm_pnl={:.2}", mtm_pnl));

    // Write outputs
    let report_path = out_dir.join("report.json");
    let report_json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&report_path, report_json)?;
    tracing::info!("Report written to: {}", report_path.display());

    // Write comprehensive PnL summary with MTM and maker scorecard
    let pnl_path = out_dir.join("pnl.json");
    // Compute edge-adjusted PnL: separates directional alpha from execution quality
    // edge_adjusted_pnl = mtm_pnl + (mean_edge_bps/10000 * notional_filled)
    // If edge_adjusted < mtm_pnl: "direction saved me"
    // If edge_adjusted > mtm_pnl: "signal was actually good"
    let notional_filled: f64 = all_results
        .iter()
        .flat_map(|(_, r)| r.leg_results.iter())
        .filter(|lr| lr.status == LegStatus::Filled || lr.status == LegStatus::PartiallyFilled)
        .filter_map(|lr| lr.fill_price.map(|px| px * lr.filled_qty as f64))
        .sum();

    let mean_edge_bps = if !slippage_samples.is_empty() {
        -(slippage_samples.iter().sum::<f64>() / slippage_samples.len() as f64)
    } else {
        0.0
    };
    let edge_cost = (mean_edge_bps / 10000.0) * notional_filled;
    let edge_adjusted_pnl = mtm_pnl - edge_cost; // Subtract cost (negative edge = positive cost)

    // Interpretation: edge_adjusted > mtm means "direction saved me"
    // (directional alpha exceeded execution cost)
    let edge_interpretation = if edge_adjusted_pnl > mtm_pnl {
        "direction_saved_me"
    } else {
        "execution_alpha_contributed"
    };

    // Normalized execution tax: bps penalty per rupee of notional (comparable across runs)
    // execution_tax_bps = (edge_adjusted_pnl - mtm_pnl) / notional_filled * 10000
    // Positive = execution helped, Negative = execution hurt
    let execution_tax_bps = if notional_filled > 0.0 {
        ((mtm_pnl - edge_adjusted_pnl) / notional_filled) * 10000.0
    } else {
        0.0
    };

    let mut pnl_json = serde_json::json!({
        "orders": all_results.len(),
        "legs_filled": fill.legs_filled,
        "cashflow": cashflow,
        "mtm_value": mtm_value,
        "mtm_pnl": mtm_pnl,
        "notional_filled": notional_filled,
        "edge_cost": edge_cost,
        "edge_adjusted_pnl": edge_adjusted_pnl,
        "execution_tax_bps": execution_tax_bps,
        "edge_interpretation": edge_interpretation,
        "open_positions": open_positions,
        "positions": positions,
        "eod_marks": eod_marks,
    });
    if let Some(scorecard) = maker_scorecard {
        pnl_json["maker_scorecard"] = scorecard;
    }
    if !mtm_warnings.is_empty() {
        pnl_json["mtm_warnings"] = serde_json::json!(mtm_warnings);
    }
    std::fs::write(&pnl_path, serde_json::to_string_pretty(&pnl_json)?)?;
    tracing::info!("PnL summary written to: {}", pnl_path.display());

    // B1.2: Write per-fill records for routing→fill join
    let fills_path = out_dir.join("fills.jsonl");
    let mut fills_file = std::io::BufWriter::new(
        File::create(&fills_path)
            .with_context(|| format!("create fills file: {:?}", fills_path))?,
    );
    use std::io::Write;
    let mut fill_count = 0u64;
    for (order, res) in &all_results {
        for (i, lr) in res.leg_results.iter().enumerate() {
            let leg = match order.legs.get(i) {
                Some(l) => l,
                None => continue,
            };
            // Only emit fills (including partial fills)
            if lr.status != LegStatus::Filled && lr.status != LegStatus::PartiallyFilled {
                continue;
            }
            let fill_record = serde_json::json!({
                "intent_id": lr.intent_id,
                "order_id": lr.order_id,
                "tradingsymbol": lr.tradingsymbol,
                "side": format!("{:?}", leg.side),
                "order_type": format!("{:?}", leg.order_type),
                "status": format!("{:?}", lr.status),
                "fill_price": lr.fill_price,
                "filled_qty": lr.filled_qty,
                "requested_qty": leg.quantity,
            });
            writeln!(fills_file, "{}", serde_json::to_string(&fill_record)?)?;
            fill_count += 1;
        }
    }
    fills_file.flush()?;
    tracing::info!(
        "Fills written to: {} ({} records)",
        fills_path.display(),
        fill_count
    );

    // India Fee Ledger: compute and emit per-fill fees
    let fee_ledger_path = out_dir.join("fee_ledger.jsonl");
    let mut fee_ledger_file = std::io::BufWriter::new(
        File::create(&fee_ledger_path)
            .with_context(|| format!("create fee ledger file: {:?}", fee_ledger_path))?,
    );

    let mut all_fees: Vec<FillFeeBreakdown> = Vec::new();
    let mut fee_count = 0u64;

    for (order, res) in &all_results {
        for (i, lr) in res.leg_results.iter().enumerate() {
            let leg = match order.legs.get(i) {
                Some(l) => l,
                None => continue,
            };
            // Only compute fees for fills
            if lr.status != LegStatus::Filled && lr.status != LegStatus::PartiallyFilled {
                continue;
            }

            let fill_price = lr.fill_price.unwrap_or(0.0);
            let filled_qty = lr.filled_qty;
            let side = match leg.side {
                LegSide::Buy => "Buy",
                LegSide::Sell => "Sell",
            };

            // Note: filled_qty is already in shares (qty × lot_size), not lots.
            // So we set lot_size=1 to avoid double-counting.
            let fee_model = IndiaFeeModel {
                lot_size: 1, // filled_qty already includes lot multiplier
                ..IndiaFeeModel::zerodha_nse()
            };

            let fees = fee_model.estimate_cost(fill_price, filled_qty, side);
            all_fees.push(fees.clone());

            // Create fee ledger record
            // Note: LegExecutionResult doesn't have fill_ts, use current time for backtest
            let ts_utc = chrono::Utc::now().to_rfc3339();

            let fee_record = FeeLedgerRecord::new(
                lr.intent_id.clone(),
                lr.order_id.clone(),
                ts_utc,
                lr.tradingsymbol.clone(),
                fees,
            );

            writeln!(fee_ledger_file, "{}", serde_json::to_string(&fee_record)?)?;
            fee_count += 1;
        }
    }
    fee_ledger_file.flush()?;
    tracing::info!(
        "Fee ledger written to: {} ({} records)",
        fee_ledger_path.display(),
        fee_count
    );

    // Flush equity curve
    equity_curve_file.flush()?;
    tracing::info!(
        "Equity curve written to: {} ({} bars)",
        equity_curve_path.display(),
        equity_bar_count
    );

    // Aggregate fees for daily PnL report
    let fee_summary = aggregate_fees_for_report(&all_fees);
    let gross_mtm = mtm_pnl;
    let net_mtm = gross_mtm - fee_summary.fees_total_inr;

    // Update pnl.json with fee breakdown
    pnl_json["fees"] = serde_json::json!({
        "total_inr": fee_summary.fees_total_inr,
        "brokerage_inr": fee_summary.fees_brokerage_inr,
        "stt_inr": fee_summary.fees_stt_inr,
        "exchange_txn_inr": fee_summary.fees_exchange_txn_inr,
        "sebi_inr": fee_summary.fees_sebi_inr,
        "stamp_inr": fee_summary.fees_stamp_inr,
        "gst_inr": fee_summary.fees_gst_inr,
    });
    pnl_json["gross_mtm_inr"] = serde_json::json!(gross_mtm);
    pnl_json["net_mtm_inr"] = serde_json::json!(net_mtm);

    // Rewrite pnl.json with fee info
    std::fs::write(&pnl_path, serde_json::to_string_pretty(&pnl_json)?)?;
    tracing::info!("PnL updated with fees: {}", pnl_path.display());

    // Generate daily_pnl.json report (quantlaxmi.reports.daily_pnl.v1)
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let mut daily_report = DailyPnlReport::new(&today, &strategy_name, "NSE-Zerodha");

    // Counts
    daily_report.counts.orders_total = all_results.len() as u32;
    daily_report.counts.legs_total = fill.legs_total as u32;
    daily_report.counts.legs_filled = fill.legs_filled as u32;
    daily_report.counts.legs_partial = fill.legs_partially_filled as u32;
    daily_report.counts.legs_rejected = fill.legs_rejected as u32;
    daily_report.counts.legs_cancelled = fill.legs_cancelled as u32;

    // PnL breakdown
    daily_report.pnl.gross_mtm_inr = gross_mtm;
    daily_report.pnl.fees_total_inr = fee_summary.fees_total_inr;
    daily_report.pnl.fees_brokerage_inr = fee_summary.fees_brokerage_inr;
    daily_report.pnl.fees_stt_inr = fee_summary.fees_stt_inr;
    daily_report.pnl.fees_exchange_txn_inr = fee_summary.fees_exchange_txn_inr;
    daily_report.pnl.fees_sebi_inr = fee_summary.fees_sebi_inr;
    daily_report.pnl.fees_stamp_inr = fee_summary.fees_stamp_inr;
    daily_report.pnl.fees_gst_inr = fee_summary.fees_gst_inr;
    daily_report.pnl.net_mtm_inr = net_mtm;

    // Execution metrics
    daily_report.execution.turnover_inr = notional_filled;
    daily_report.execution.slippage_bps_p50 = fill.slippage_bps_p50;
    daily_report.execution.slippage_bps_p90 = fill.slippage_bps_p90;
    daily_report.execution.slippage_bps_p99 = fill.slippage_bps_p99;
    daily_report.execution.fill_rate_pct = if fill.legs_total > 0 {
        (fill.legs_filled as f64 / fill.legs_total as f64) * 100.0
    } else {
        0.0
    };
    daily_report.execution.avg_execution_tax_bps = execution_tax_bps;

    // Risk metrics
    daily_report.risk.max_position_contracts = positions
        .values()
        .map(|q| q.abs() as u32)
        .max()
        .unwrap_or(0);

    // Populate equity curve performance from trackers
    daily_report.performance.equity_curve.bar_interval = "1s".to_string();
    daily_report.performance.equity_curve.bars = equity_bar_count;
    daily_report.performance.equity_curve.gross_pnl_inr = gross_mtm;
    daily_report.performance.equity_curve.fees_inr = fee_summary.fees_total_inr;
    daily_report.performance.equity_curve.net_pnl_inr = net_mtm;

    if let Some(ref tr) = mdd_tracker {
        daily_report.performance.equity_curve.max_drawdown_inr = tr.max_drawdown_inr;
        daily_report.performance.equity_curve.max_drawdown_pct = tr.max_drawdown_pct;
        daily_report.performance.equity_curve.dd_peak_ts_utc = Some(tr.peak_ts_utc());
        daily_report.performance.equity_curve.dd_trough_ts_utc = Some(tr.trough_ts_utc());
        // Also populate risk metrics
        daily_report.risk.peak_drawdown_inr = tr.max_drawdown_inr;
    }

    daily_report.performance.equity_curve.sharpe_equity_curve = returns_tracker.sharpe();
    daily_report.performance.equity_curve.sortino = returns_tracker.sortino();
    daily_report.performance.equity_curve.mean_return = returns_tracker.mean;
    daily_report.performance.equity_curve.std_return = returns_tracker.std_dev();

    // Note: intent_edge performance should be populated from intent_pnl.jsonl
    // For now, leave it at defaults - will be filled by generate-intent-pnl

    // Notes
    daily_report
        .notes
        .push(format!("strategy={}", strategy_name));
    daily_report
        .notes
        .push(format!("replay={}", cfg.replay_path));
    daily_report
        .notes
        .push(format!("open_positions={}", open_positions));
    daily_report
        .notes
        .push(format!("equity_bars={}", equity_bar_count));

    let daily_pnl_path = out_dir.join("daily_pnl.json");
    std::fs::write(
        &daily_pnl_path,
        serde_json::to_string_pretty(&daily_report)?,
    )?;
    tracing::info!("Daily PnL report written to: {}", daily_pnl_path.display());

    // Emit run_summary.json for offline verification: equity_last == gross_mtm_inr
    let run_summary = serde_json::json!({
        "schema": "quantlaxmi.reports.run_summary.v1",
        "equity_first_inr": equity_first_inr,
        "equity_last_inr": equity_last_inr,
        "gross_mtm_inr": gross_mtm,
        "fees_total_inr": fee_summary.fees_total_inr,
        "net_mtm_inr": net_mtm,
        "equity_bars": equity_bar_count,
        "verification": {
            "equity_last_equals_gross_mtm": equity_last_inr.map(|el| (el - gross_mtm).abs() < 0.01).unwrap_or(false),
            "note": "With baseline, equity_first=0 and equity_last==gross_mtm_inr"
        }
    });
    let run_summary_path = out_dir.join("run_summary.json");
    std::fs::write(
        &run_summary_path,
        serde_json::to_string_pretty(&run_summary)?,
    )?;
    tracing::info!("Run summary written to: {}", run_summary_path.display());

    tracing::info!("\n=== KiteSim Backtest Complete (India/Zerodha) ===");
    tracing::info!("Strategy: {}", strategy_name);
    tracing::info!(
        "Orders: {}, Legs filled: {}/{}",
        fill.orders_total,
        fill.legs_filled,
        fill.legs_total
    );
    tracing::info!("Cashflow: ₹{:.2}", cashflow);
    tracing::info!("MTM Value: ₹{:.2}", mtm_value);
    tracing::info!("Gross MTM PnL: ₹{:.2}", gross_mtm);
    tracing::info!(
        "Fees: ₹{:.2} (brokerage={:.2}, STT={:.2}, txn={:.2}, GST={:.2})",
        fee_summary.fees_total_inr,
        fee_summary.fees_brokerage_inr,
        fee_summary.fees_stt_inr,
        fee_summary.fees_exchange_txn_inr,
        fee_summary.fees_gst_inr
    );
    tracing::info!("Net MTM PnL: ₹{:.2}", net_mtm);
    tracing::info!("Open positions: {}", open_positions);

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
