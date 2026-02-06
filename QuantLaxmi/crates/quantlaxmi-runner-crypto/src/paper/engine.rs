use crate::paper::decision_log::{
    BookLog, DecisionLogger, DecisionRecordBuilder, EdgeLog, FTILog, GateCounters, IntentLog,
    ProposedIntent, SniperLog, StateLog, extract_gates_from_refusals,
};
use crate::paper::intent::Side;
use crate::paper::position_manager::{EntrySnapshot, ExitConfig, PositionManager};
use crate::paper::slrt::{BinanceDepth, BinanceTrade, SlrtConfig, SlrtPipeline};
use crate::paper::sniper::{SniperConfig, SniperInput, SniperState, sniper_admission};
use crate::paper::state::{
    LastTrade, MarketData, PaperPosition, SniperStats, TradeTape, UiSnapshot,
};
use crate::paper::telemetry::TelemetryBus;
use crate::paper::trade_log::{RunConfig, TradeLogger};

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, mpsc};
use tracing::{error, info, warn};

/// Shared state for paper runner and TUI.
#[derive(Clone)]
pub struct SharedState {
    inner: Arc<RwLock<UiSnapshot>>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(UiSnapshot::default())),
        }
    }

    pub async fn get(&self) -> UiSnapshot {
        self.inner.read().await.clone()
    }

    pub async fn set_snapshot(&self, snap: UiSnapshot) {
        *self.inner.write().await = snap;
    }

    pub fn arc(&self) -> Arc<RwLock<UiSnapshot>> {
        self.inner.clone()
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

/// Local order book maintained from diff updates.
#[derive(Debug, Clone, Default)]
pub struct LocalOrderBook {
    pub bids: BTreeMap<i64, f64>, // price_mantissa -> qty (descending for bids)
    pub asks: BTreeMap<i64, f64>, // price_mantissa -> qty (ascending for asks)
    pub last_update_id: u64,
    pub initialized: bool,
}

impl LocalOrderBook {
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize from REST API snapshot.
    pub fn init_from_snapshot(
        &mut self,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
        last_update_id: u64,
    ) {
        self.bids.clear();
        self.asks.clear();

        for (price, qty) in bids {
            let mantissa = (*price * 100.0) as i64;
            if *qty > 0.0 {
                self.bids.insert(mantissa, *qty);
            }
        }

        for (price, qty) in asks {
            let mantissa = (*price * 100.0) as i64;
            if *qty > 0.0 {
                self.asks.insert(mantissa, *qty);
            }
        }

        self.last_update_id = last_update_id;
        self.initialized = true;
    }

    /// Apply diff update from WebSocket.
    pub fn apply_diff(
        &mut self,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
        first_update_id: u64,
        final_update_id: u64,
    ) -> bool {
        // Validate sequence
        if !self.initialized {
            return false;
        }

        // Drop outdated updates
        if final_update_id <= self.last_update_id {
            return false;
        }

        // Check for gap (should be first_update_id <= last_update_id + 1)
        if first_update_id > self.last_update_id + 1 {
            warn!(
                "[BOOK] Sequence gap detected: expected <= {}, got {}",
                self.last_update_id + 1,
                first_update_id
            );
            self.initialized = false;
            return false;
        }

        // Apply bid updates
        for (price, qty) in bids {
            let mantissa = (*price * 100.0) as i64;
            if *qty == 0.0 {
                self.bids.remove(&mantissa);
            } else {
                self.bids.insert(mantissa, *qty);
            }
        }

        // Apply ask updates
        for (price, qty) in asks {
            let mantissa = (*price * 100.0) as i64;
            if *qty == 0.0 {
                self.asks.remove(&mantissa);
            } else {
                self.asks.insert(mantissa, *qty);
            }
        }

        self.last_update_id = final_update_id;
        true
    }

    /// Convert to BinanceDepth for SLRT pipeline.
    pub fn to_binance_depth(&self, ts_ns: i64) -> BinanceDepth {
        // Bids: highest price first (reverse order)
        let bids: Vec<(f64, f64)> = self
            .bids
            .iter()
            .rev()
            .take(20)
            .map(|(m, q)| (*m as f64 / 100.0, *q))
            .collect();

        // Asks: lowest price first (natural order)
        let asks: Vec<(f64, f64)> = self
            .asks
            .iter()
            .take(20)
            .map(|(m, q)| (*m as f64 / 100.0, *q))
            .collect();

        BinanceDepth { ts_ns, bids, asks }
    }

    pub fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|m| *m as f64 / 100.0)
    }

    pub fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|m| *m as f64 / 100.0)
    }
}

/// Trade statistics tracker.
#[derive(Debug, Clone)]
pub struct TradeStats {
    trades: Vec<(Instant, f64, bool)>, // (time, qty, is_buy)
    window: Duration,
}

impl TradeStats {
    pub fn new(window: Duration) -> Self {
        Self {
            trades: Vec::with_capacity(1000),
            window,
        }
    }

    pub fn add_trade(&mut self, qty: f64, is_buy: bool) {
        let now = Instant::now();
        self.trades.push((now, qty, is_buy));

        // Prune old trades
        let cutoff = now - self.window;
        self.trades.retain(|(t, _, _)| *t > cutoff);
    }

    pub fn trades_per_sec(&self) -> f64 {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(1);
        self.trades.iter().filter(|(t, _, _)| *t > cutoff).count() as f64
    }

    pub fn volume_in_window(&self) -> (f64, f64) {
        let now = Instant::now();
        let cutoff = now - self.window;

        let mut buy_vol = 0.0;
        let mut sell_vol = 0.0;

        for (t, qty, is_buy) in &self.trades {
            if *t > cutoff {
                if *is_buy {
                    buy_vol += qty;
                } else {
                    sell_vol += qty;
                }
            }
        }

        (buy_vol, sell_vol)
    }
}

impl Default for TradeStats {
    fn default() -> Self {
        Self::new(Duration::from_secs(5))
    }
}

/// Message types for internal channels.
#[derive(Debug)]
enum BookEvent {
    Diff {
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        first_update_id: u64,
        final_update_id: u64,
        ts_ns: i64,
    },
    Trade(BinanceTrade),
}

/// PaperEngine runs the SLRT pipeline and emits telemetry.
pub struct PaperEngine {
    pub symbol: String,
    pub state: SharedState,
    pub telemetry: TelemetryBus,
    pub log_dir: Option<PathBuf>,
}

impl PaperEngine {
    pub fn new(symbol: impl Into<String>, state: SharedState, telemetry: TelemetryBus) -> Self {
        Self {
            symbol: symbol.into(),
            state,
            telemetry,
            log_dir: None,
        }
    }

    /// Enable decision logging to the specified directory.
    pub fn with_log_dir(mut self, log_dir: impl Into<PathBuf>) -> Self {
        self.log_dir = Some(log_dir.into());
        self
    }

    /// Get gate counters (for TUI display).
    pub fn gate_counters(&self) -> Option<Arc<GateCounters>> {
        // This is set during run(), so we can't return it here
        // The counters are exposed via the logger in run()
        None
    }

    /// Run the paper engine with real-time Binance WebSocket data.
    pub async fn run(self) {
        info!("[PAPER] Starting SLRT paper engine for {}", self.symbol);

        // Create SLRT pipeline
        let slrt_config = SlrtConfig {
            symbol: self.symbol.clone(),
            tick_size: 0.01,
            bucket_size: 100.0,
        };

        let mut pipeline = match SlrtPipeline::new(slrt_config) {
            Ok(p) => p,
            Err(e) => {
                error!("[PAPER] Failed to create SLRT pipeline: {}", e);
                return;
            }
        };

        // Initialize local order book from REST snapshot
        let mut book = LocalOrderBook::new();
        let symbol_lower = self.symbol.to_lowercase();

        info!("[PAPER] Fetching initial order book snapshot...");
        if let Err(e) = fetch_snapshot(&mut book, &symbol_lower).await {
            error!("[PAPER] Failed to fetch initial snapshot: {}", e);
            return;
        }
        info!(
            "[PAPER] Order book initialized with {} bids, {} asks",
            book.bids.len(),
            book.asks.len()
        );

        // Channel for events
        let (tx, mut rx) = mpsc::channel::<BookEvent>(10000);
        let tx_depth = tx.clone();
        let tx_trade = tx;

        // Real-time diff depth stream
        let depth_url = format!(
            "wss://stream.binance.com:9443/ws/{}@depth@100ms",
            symbol_lower
        );

        // Trade stream
        let trade_url = format!("wss://stream.binance.com:9443/ws/{}@trade", symbol_lower);

        info!(
            "[PAPER] Connecting to real-time depth stream: {}",
            depth_url
        );
        info!("[PAPER] Connecting to trade stream: {}", trade_url);

        use futures::StreamExt;
        use tokio_tungstenite::tungstenite::Message;

        // Spawn depth stream handler
        let depth_handle = tokio::spawn(async move {
            loop {
                let ws = match tokio_tungstenite::connect_async(&depth_url).await {
                    Ok((ws, _)) => ws,
                    Err(e) => {
                        error!("[PAPER] Failed to connect to depth stream: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };

                let (_, mut read) = ws.split();

                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Some(diff) = parse_depth_diff(&text)
                                && tx_depth.send(diff).await.is_err()
                            {
                                return;
                            }
                        }
                        Ok(Message::Close(_)) => {
                            warn!("[PAPER] Depth stream closed, reconnecting...");
                            break;
                        }
                        Err(e) => {
                            warn!("[PAPER] Depth stream error: {}, reconnecting...", e);
                            break;
                        }
                        _ => {}
                    }
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        // Spawn trade stream handler
        let trade_handle = tokio::spawn(async move {
            loop {
                let ws = match tokio_tungstenite::connect_async(&trade_url).await {
                    Ok((ws, _)) => ws,
                    Err(e) => {
                        error!("[PAPER] Failed to connect to trade stream: {}", e);
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        continue;
                    }
                };

                let (_, mut read) = ws.split();

                while let Some(msg) = read.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Some(trade) = parse_trade(&text)
                                && tx_trade.send(BookEvent::Trade(trade)).await.is_err()
                            {
                                return;
                            }
                        }
                        Ok(Message::Close(_)) => {
                            warn!("[PAPER] Trade stream closed, reconnecting...");
                            break;
                        }
                        Err(e) => {
                            warn!("[PAPER] Trade stream error: {}, reconnecting...", e);
                            break;
                        }
                        _ => {}
                    }
                }

                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });

        info!("[PAPER] Connected to Binance streams");

        // Initialize sniper system
        // Use CANARY_MODE=1 env var to enable canary mode for validation
        let sniper_config = if std::env::var("CANARY_MODE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            info!("[PAPER] *** CANARY MODE ENABLED *** - permissive thresholds for validation");
            SniperConfig::canary()
        } else {
            SniperConfig::sniper()
        };
        let mut sniper_state = SniperState::new(now_ms());
        info!(
            "[PAPER] Sniper config: regime={:?}, conf>={}, tox<={}, fti_confirm={}, cooldown={}s",
            sniper_config.allowed_regimes,
            sniper_config.confidence_min,
            sniper_config.toxicity_max,
            sniper_config.require_fti_confirm,
            sniper_config.cooldown_seconds
        );

        // Exit config for canary mode (TP/SL/time-stop)
        let exit_config = if std::env::var("CANARY_MODE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            ExitConfig::canary()
        } else {
            ExitConfig::default()
        };
        info!(
            "[PAPER] Exit config: time_stop={}s, TP={}bps, SL={}bps",
            exit_config.time_stop_seconds, exit_config.take_profit_bps, exit_config.stop_loss_bps
        );

        // Generate run_id for all loggers
        let run_id = format!("slrt_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"));

        // Initialize decision logger
        let decision_logger = if let Some(log_dir) = &self.log_dir {
            match DecisionLogger::new(log_dir.clone(), &run_id, 10000) {
                Ok(logger) => {
                    info!(
                        "[PAPER] Decision logging enabled: {}/{}_decisions.jsonl",
                        log_dir.display(),
                        run_id
                    );
                    Some(logger)
                }
                Err(e) => {
                    error!("[PAPER] Failed to create decision logger: {}", e);
                    None
                }
            }
        } else {
            info!("[PAPER] Decision logging disabled (no log_dir set)");
            None
        };

        // Initialize trade logger (entry/exit pairs for profitability analysis)
        let mut trade_logger = if let Some(log_dir) = &self.log_dir {
            let run_config = RunConfig::from_sniper_config(
                &run_id,
                &self.symbol,
                if std::env::var("CANARY_MODE")
                    .map(|v| v == "1")
                    .unwrap_or(false)
                {
                    "canary"
                } else {
                    "sniper"
                },
                &sniper_config,
                exit_config.time_stop_seconds,
                exit_config.take_profit_bps,
                exit_config.stop_loss_bps,
            );
            match TradeLogger::new(log_dir.clone(), &run_id, &run_config) {
                Ok(logger) => {
                    info!(
                        "[PAPER] Trade logging enabled: {}/{}_trades.jsonl",
                        log_dir.display(),
                        run_id
                    );
                    Some(logger)
                }
                Err(e) => {
                    error!("[PAPER] Failed to create trade logger: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize position manager with exit logic
        let mut position_mgr = PositionManager::new(exit_config.clone());

        // Process events
        let mut tick: u64 = 0;
        let mut trade_stats = TradeStats::default();
        let mut last_trade: Option<LastTrade> = None;
        let mut trade_tape = TradeTape::new(20); // Keep last 20 trades
        let mut position = PaperPosition::default();

        while let Some(event) = rx.recv().await {
            match event {
                BookEvent::Diff {
                    bids,
                    asks,
                    first_update_id,
                    final_update_id,
                    ts_ns,
                } => {
                    // Re-fetch snapshot if needed
                    if !book.initialized {
                        info!("[PAPER] Re-fetching snapshot after sequence gap...");
                        if let Err(e) = fetch_snapshot(&mut book, &symbol_lower).await {
                            error!("[PAPER] Failed to re-fetch snapshot: {}", e);
                            continue;
                        }
                    }

                    // Apply diff
                    if !book.apply_diff(&bids, &asks, first_update_id, final_update_id) {
                        continue;
                    }

                    tick += 1;
                    let now = now_ms();

                    // Convert to BinanceDepth for pipeline
                    let depth = book.to_binance_depth(ts_ns);

                    // Build market data
                    let (buy_vol, sell_vol) = trade_stats.volume_in_window();
                    let mut market_data = build_market_data(&depth, &self.symbol, tick);
                    market_data.last_trade = last_trade.clone();
                    market_data.trade_tape = trade_tape.clone();
                    market_data.trades_per_sec = trade_stats.trades_per_sec();
                    market_data.buy_volume = buy_vol;
                    market_data.sell_volume = sell_vol;

                    // Process through SLRT pipeline
                    let admission = match pipeline.process_depth(&depth) {
                        Some(a) => a,
                        None => continue,
                    };

                    // === SNIPER MULTI-GATE ADMISSION ===
                    let best_bid = depth.bids.first().map(|(p, _)| *p).unwrap_or(0.0);
                    let best_ask = depth.asks.first().map(|(p, _)| *p).unwrap_or(0.0);
                    let bid_qty = depth.bids.first().map(|(_, q)| *q).unwrap_or(0.0);
                    let ask_qty = depth.asks.first().map(|(_, q)| *q).unwrap_or(0.0);

                    let sniper_input = SniperInput {
                        symbol: self.symbol.clone(),
                        tick,
                        now_ms: now,

                        // From SLRT pipeline
                        eligibility: admission.eligibility,
                        regime: admission.regime.clone(),
                        metrics: admission.metrics.clone(),
                        slrt_refusal_reasons: admission.refusal_reasons.clone(),

                        // Market data
                        best_bid,
                        best_ask,
                        bid_qty,
                        ask_qty,
                        spread_bps: market_data.spread_bps.unwrap_or(f64::MAX),
                        imbalance: market_data.imbalance.unwrap_or(0.0),

                        // Position - use position_mgr to check if we have open position
                        position_size: if position_mgr.has_open_position() {
                            match position_mgr.position_side() {
                                Some(Side::Buy) => sniper_config.qty,
                                Some(Side::Sell) => -sniper_config.qty,
                                None => 0.0,
                            }
                        } else {
                            0.0
                        },
                    };

                    let sniper_output =
                        sniper_admission(&sniper_config, &mut sniper_state, &sniper_input);

                    // === DECISION LOGGING ===
                    // Compute edge values for logging
                    let notional = sniper_config.qty * best_ask.max(best_bid);
                    let fee_cost = 2.0 * notional * sniper_config.fee_rate;
                    let spread_cost = (best_ask - best_bid).max(0.0) * sniper_config.qty;
                    let buffer_cost = (sniper_config.sniper_buffer_bps / 10000.0) * notional;
                    let edge_req = fee_cost + spread_cost + buffer_cost;
                    let imb_abs = market_data.imbalance.unwrap_or(0.0).abs();
                    let edge_est = imb_abs * 0.5 * notional * 0.01;

                    if let Some(ref logger) = decision_logger {
                        let metrics = &sniper_output.metrics;
                        let mid = (best_bid + best_ask) / 2.0;

                        let book_log = BookLog {
                            bid: best_bid,
                            ask: best_ask,
                            mid,
                            spread_bps: market_data.spread_bps.unwrap_or(0.0),
                            bid_qty,
                            ask_qty,
                            imb: market_data.imbalance.unwrap_or(0.0),
                        };

                        // No silent poisoning: pass Options directly, log null when absent
                        let state_log = StateLog {
                            regime: admission.regime.clone(),
                            confidence: metrics.confidence,
                            d_perp: metrics.d_perp,
                            fragility: metrics.fragility,
                            toxicity: metrics.toxicity,
                            tox_persist: metrics.toxicity_persist,
                            fti: FTILog {
                                level: metrics.fti_level,
                                thresh: metrics.fti_thresh,
                                elev: metrics.fti_elevated,
                                persist: metrics.fti_persist,
                                calibrated: metrics.fti_calibrated,
                            },
                        };

                        let gates = extract_gates_from_refusals(
                            &sniper_output.refusal_reasons,
                            &sniper_config,
                            metrics,
                            edge_est,
                            edge_req,
                        );

                        let cooldown_ms = sniper_state
                            .last_entry_ms
                            .map(|last| {
                                let elapsed = now.saturating_sub(last);
                                let required = sniper_config.cooldown_seconds * 1000;
                                required.saturating_sub(elapsed)
                            })
                            .unwrap_or(0);

                        let sniper_log = SniperLog {
                            eligible: sniper_output.accepted.is_some(),
                            final_gate: sniper_output.final_gate.as_str().to_string(),
                            gates,
                            edge: EdgeLog {
                                est: edge_est,
                                req: edge_req,
                                fees: fee_cost,
                                spread: spread_cost,
                                buffer: buffer_cost,
                            },
                            cooldown_ms,
                            entries_hour: sniper_state.entries_in_last_hour(now),
                            entries_session: sniper_state.session_entries,
                        };

                        let intent_log = IntentLog {
                            proposed: sniper_output.proposed.as_ref().map(|p| ProposedIntent {
                                side: format!("{:?}", p.side),
                                qty: p.qty,
                                order_type: "MKT".to_string(),
                                id: p.intent_id.clone(),
                            }),
                            accepted: sniper_output.accepted.as_ref().map(|a| ProposedIntent {
                                side: format!("{:?}", a.side),
                                qty: a.qty,
                                order_type: "MKT".to_string(),
                                id: a.intent_id.clone(),
                            }),
                        };

                        let record = DecisionRecordBuilder::new(tick, &self.symbol)
                            .build(book_log, state_log, sniper_log, intent_log, None);

                        logger.log(record);
                    }

                    // === EXIT CHECK (before new entries) ===
                    // Check for TP/SL/time-stop exits on open position
                    if let Some(completed) = position_mgr.check_exit(tick, now, best_bid, best_ask)
                    {
                        info!(
                            "[EXIT] {} trade #{} @ ${:.2} | reason={} | gross=${:.4} | fees=${:.4} | net=${:.4} | hold={}ms",
                            completed.entry_side,
                            completed.trade_id,
                            completed.exit_price,
                            completed.exit_reason,
                            completed.gross_pnl,
                            completed.fees_total,
                            completed.net_pnl,
                            completed.hold_time_ms
                        );

                        // Update paper position for TUI display
                        position.fill(
                            completed.entry_qty,
                            completed.exit_price,
                            completed.entry_side == "Sell", // Opposite side to close
                        );

                        // Log completed trade
                        if let Some(ref mut logger) = trade_logger {
                            logger.log_trade(completed);
                        }
                    }

                    // === ENTRY (only when flat) ===
                    // Execute fill if accepted AND no current position
                    if let Some(ref accepted) = sniper_output.accepted
                        && !position_mgr.has_open_position()
                    {
                        let fill_price = match accepted.side {
                            Side::Buy => best_ask,
                            Side::Sell => best_bid,
                        };
                        if fill_price > 0.0 {
                            // Create entry snapshot for post-mortem
                            let entry_snapshot = EntrySnapshot::from_metrics(
                                &sniper_output.metrics,
                                &admission.regime,
                                market_data.imbalance.unwrap_or(0.0),
                                edge_est,
                                edge_req,
                                market_data.spread_bps.unwrap_or(0.0),
                            );

                            // Open trade via position manager
                            let trade_id = position_mgr.open_trade(
                                tick,
                                now,
                                accepted.side,
                                sniper_config.qty,
                                fill_price,
                                market_data.spread_bps.unwrap_or(0.0),
                                entry_snapshot,
                            );

                            // Update paper position for TUI display
                            position.fill(
                                sniper_config.qty,
                                fill_price,
                                matches!(accepted.side, Side::Buy),
                            );

                            // Record entry for trade logger
                            if let Some(ref mut logger) = trade_logger {
                                logger.record_entry();
                            }

                            info!(
                                "[ENTRY] #{} {} @ ${:.2} | conf={:.2} | tox={:.2} | imb={:.2} | Trades: {} | Net: ${:.4}",
                                trade_id,
                                accepted.side,
                                fill_price,
                                sniper_output.metrics.confidence.unwrap_or(0.0),
                                sniper_output.metrics.toxicity.unwrap_or(0.0),
                                market_data.imbalance.unwrap_or(0.0),
                                position_mgr.accepted_trades,
                                position_mgr.net_pnl_total
                            );
                        }
                    }

                    // Mark position to market
                    if let Some(mid) = market_data.mid_price {
                        position.mark_to_market(mid);
                    }

                    // === UPDATE SNAPSHOT WITH SNIPER OUTPUT ===
                    // Build last_decision from sniper output (merges SLRT + sniper refusals)
                    let mut snap = self.state.get().await;

                    // Update last_decision with merged refusal reasons
                    let sniper_eligibility = if sniper_output.accepted.is_some() {
                        crate::paper::state::R3Eligibility::Eligible
                    } else if !sniper_output.refusal_reasons.is_empty() {
                        crate::paper::state::R3Eligibility::Refused
                    } else {
                        admission.eligibility
                    };

                    snap.last_decision = crate::paper::state::LastDecision {
                        eligibility: sniper_eligibility,
                        refusal_reasons: sniper_output.refusal_reasons.clone(),
                        metrics: sniper_output.metrics.clone(),
                        decided_at_unix_ms: now,
                    };

                    snap.market_data = market_data;
                    snap.position = position.clone();

                    // proposed_this_tick: from sniper (None if no signal)
                    snap.proposed_this_tick = sniper_output.proposed.clone();

                    // accepted_this_tick: from sniper (None if any gate failed)
                    snap.accepted_this_tick = sniper_output.accepted.clone();

                    // last_accepted_historical: only update when sniper actually fires
                    if let Some(a) = sniper_output.accepted {
                        snap.last_accepted_historical = Some(a);
                    }

                    // Sniper statistics for TUI
                    let secs_since_last = sniper_state
                        .last_entry_ms
                        .map(|last| now.saturating_sub(last) / 1000);
                    snap.sniper_stats = SniperStats {
                        entries_last_hour: sniper_state.entries_in_last_hour(now),
                        max_per_hour: sniper_config.max_entries_per_hour,
                        session_entries: sniper_state.session_entries,
                        max_per_session: sniper_config.max_entries_per_session,
                        secs_since_last,
                        cooldown_secs: sniper_config.cooldown_seconds,
                        cooldown_ok: sniper_state.cooldown_ok(now, sniper_config.cooldown_seconds),
                        regime: admission.regime.clone(),
                        r3_cause: sniper_output.r3_cause.as_str().to_string(),
                    };

                    self.state.set_snapshot(snap.clone()).await;
                    self.telemetry.publish(snap);

                    // Log progress periodically with trade stats
                    if tick.is_multiple_of(1000) {
                        let avg_net = position_mgr.avg_net_per_trade();
                        let expectancy = if avg_net > 0.0 { "+" } else { "-" };
                        info!(
                            "[STATS] tick={} | trades: {}/{} | net=${:.4} | avg=${:.6} ({}) | fee_share={:.1}%",
                            tick,
                            position_mgr.completed_trades,
                            position_mgr.accepted_trades,
                            position_mgr.net_pnl_total,
                            avg_net,
                            expectancy,
                            position_mgr.fee_share() * 100.0
                        );
                    }
                }
                BookEvent::Trade(trade) => {
                    // Update pipeline
                    pipeline.process_trade(&trade);

                    // Track stats
                    let is_buy = !trade.is_buyer_maker;
                    trade_stats.add_trade(trade.qty, is_buy);

                    // Update last trade and tape
                    let lt = LastTrade {
                        price: trade.price,
                        qty: trade.qty,
                        is_buy,
                        ts_ms: (trade.ts_ns / 1_000_000) as u64,
                    };
                    trade_tape.add(lt.clone());
                    last_trade = Some(lt);
                }
            }
        }

        // Cleanup
        depth_handle.abort();
        trade_handle.abort();

        // Force close any open position at last price
        if position_mgr.has_open_position() {
            let exit_price = book.best_bid().unwrap_or(0.0);
            if let Some(completed) = position_mgr.force_close(tick, now_ms(), exit_price) {
                info!(
                    "[EXIT] FORCED {} trade #{} @ ${:.2} | net=${:.4}",
                    completed.entry_side,
                    completed.trade_id,
                    completed.exit_price,
                    completed.net_pnl
                );
                if let Some(ref mut logger) = trade_logger {
                    logger.log_trade(completed);
                }
            }
        }

        // Finalize trade logger and print summary
        if let Some(logger) = trade_logger {
            let summary = logger.finalize();
            info!("============================================================");
            info!("[SUMMARY] Run: {}", summary.run_id);
            info!(
                "[SUMMARY] Duration: {}s | Ticks: {}",
                summary.duration_secs, tick
            );
            info!(
                "[SUMMARY] Trades: {} accepted, {} completed",
                summary.accepted_trades, summary.completed_trades
            );
            info!(
                "[SUMMARY] Gross PnL: ${:.4} | Fees: ${:.4} | Net PnL: ${:.4}",
                summary.gross_pnl_total, summary.fees_total, summary.net_pnl_total
            );
            info!(
                "[SUMMARY] Avg net/trade: ${:.6} | Win rate: {:.1}%",
                summary.avg_net_per_trade,
                summary.win_rate * 100.0
            );
            info!(
                "[SUMMARY] Exits: TP={}, SL={}, TIME={}, MANUAL={}",
                summary.exits_tp, summary.exits_sl, summary.exits_time, summary.exits_manual
            );
            info!("[SUMMARY] Fee share: {:.1}%", summary.fee_share * 100.0);
            info!("============================================================");
            if summary.expectancy_positive {
                info!("[VERDICT] EXPECTANCY POSITIVE - Signal has edge after fees");
            } else {
                info!("[VERDICT] EXPECTANCY NEGATIVE - No edge detected");
            }
            info!("============================================================");
        }

        info!("[PAPER] Engine stopped after {} ticks", tick);
    }
}

/// Fetch order book snapshot from REST API.
async fn fetch_snapshot(book: &mut LocalOrderBook, symbol: &str) -> Result<(), String> {
    let url = format!(
        "https://api.binance.com/api/v3/depth?symbol={}&limit=1000",
        symbol.to_uppercase()
    );

    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("HTTP error: {}", e))?;

    let text = resp
        .text()
        .await
        .map_err(|e| format!("Read error: {}", e))?;

    #[derive(serde::Deserialize)]
    struct SnapshotResponse {
        #[serde(rename = "lastUpdateId")]
        last_update_id: u64,
        bids: Vec<[String; 2]>,
        asks: Vec<[String; 2]>,
    }

    let snap: SnapshotResponse =
        serde_json::from_str(&text).map_err(|e| format!("Parse error: {}", e))?;

    let bids: Vec<(f64, f64)> = snap
        .bids
        .iter()
        .filter_map(|[p, q]| Some((p.parse().ok()?, q.parse().ok()?)))
        .collect();

    let asks: Vec<(f64, f64)> = snap
        .asks
        .iter()
        .filter_map(|[p, q]| Some((p.parse().ok()?, q.parse().ok()?)))
        .collect();

    book.init_from_snapshot(&bids, &asks, snap.last_update_id);
    Ok(())
}

/// Build market data from depth snapshot.
fn build_market_data(depth: &BinanceDepth, symbol: &str, tick: u64) -> MarketData {
    let best_bid = depth.bids.first().map(|(p, _)| *p);
    let best_ask = depth.asks.first().map(|(p, _)| *p);
    let bid_qty = depth.bids.first().map(|(_, q)| *q);
    let ask_qty = depth.asks.first().map(|(_, q)| *q);

    let mid_price = match (best_bid, best_ask) {
        (Some(b), Some(a)) => Some((b + a) / 2.0),
        _ => None,
    };

    let spread_bps = match (best_bid, best_ask) {
        (Some(b), Some(a)) if b > 0.0 => Some((a - b) / ((a + b) / 2.0) * 10000.0),
        _ => None,
    };

    let imbalance = match (bid_qty, ask_qty) {
        (Some(bq), Some(aq)) if bq + aq > 0.0 => Some((bq - aq) / (bq + aq)),
        _ => None,
    };

    MarketData {
        symbol: symbol.to_string(),
        best_bid,
        best_ask,
        mid_price,
        spread_bps,
        bid_qty,
        ask_qty,
        imbalance,
        tick_count: tick,
        last_trade: None,
        trade_tape: TradeTape::default(),
        trades_per_sec: 0.0,
        buy_volume: 0.0,
        sell_volume: 0.0,
    }
}

/// Parse Binance depth diff message.
fn parse_depth_diff(text: &str) -> Option<BookEvent> {
    #[derive(serde::Deserialize)]
    struct DepthDiff {
        #[serde(rename = "U")]
        first_update_id: u64,
        #[serde(rename = "u")]
        final_update_id: u64,
        #[serde(rename = "b")]
        bids: Vec<[String; 2]>,
        #[serde(rename = "a")]
        asks: Vec<[String; 2]>,
    }

    let msg: DepthDiff = serde_json::from_str(text).ok()?;

    let ts_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64;

    let bids: Vec<(f64, f64)> = msg
        .bids
        .iter()
        .filter_map(|[p, q]| Some((p.parse().ok()?, q.parse().ok()?)))
        .collect();

    let asks: Vec<(f64, f64)> = msg
        .asks
        .iter()
        .filter_map(|[p, q]| Some((p.parse().ok()?, q.parse().ok()?)))
        .collect();

    Some(BookEvent::Diff {
        bids,
        asks,
        first_update_id: msg.first_update_id,
        final_update_id: msg.final_update_id,
        ts_ns,
    })
}

/// Parse Binance trade message.
fn parse_trade(text: &str) -> Option<BinanceTrade> {
    #[derive(serde::Deserialize)]
    struct BinanceTradeMsg {
        #[serde(rename = "T")]
        trade_time: u64,
        #[serde(rename = "p")]
        price: String,
        #[serde(rename = "q")]
        qty: String,
        #[serde(rename = "m")]
        is_buyer_maker: bool,
    }

    let msg: BinanceTradeMsg = serde_json::from_str(text).ok()?;

    Some(BinanceTrade {
        ts_ns: (msg.trade_time as i64) * 1_000_000,
        price: msg.price.parse().ok()?,
        qty: msg.qty.parse().ok()?,
        is_buyer_maker: msg.is_buyer_maker,
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}
