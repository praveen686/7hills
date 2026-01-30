//! Live Paper Trading TUI
//!
//! Real-time terminal UI for paper trading with Zerodha WebSocket data.
//! Shows positions, trades, market data, and gate status in a dashboard.
//!
//! Features regime warmup from historical data to ensure the Grassmann
//! detector has context before any trades are taken.

use std::collections::HashMap;
use std::io::{self, Stdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Local, Utc};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table},
    Frame, Terminal,
};

use quantlaxmi_connectors_zerodha::{ZerodhaAutoDiscovery, ZerodhaConnector};
use quantlaxmi_core::{EventBus, MarketPayload};
use quantlaxmi_options_engine::{
    aggregate_warmup_data,
    pcr::{OptionData, OptionDataType},
    EngineConfig, OptionsEngine, StrategyType, TradingAction, WarmupState,
};
use quantlaxmi_regime::FeatureVector;

#[derive(Parser)]
#[command(name = "live-paper-tui")]
#[command(about = "Live paper trading with TUI dashboard")]
struct Args {
    /// Symbols to trade (comma-separated, or NIFTY-ATM for auto-discovery)
    #[arg(long, default_value = "NIFTY-ATM")]
    symbols: String,

    /// Initial capital (INR)
    #[arg(long, default_value = "1000000")]
    capital: f64,

    /// Max positions
    #[arg(long, default_value = "3")]
    max_positions: u32,

    /// Minimum strategy score to trade
    #[arg(long, default_value = "60")]
    min_score: f64,

    /// Stop loss per position (INR)
    #[arg(long, default_value = "5000")]
    stop_loss: f64,

    /// Profit target per position (INR)
    #[arg(long, default_value = "2500")]
    profit_target: f64,

    /// Warmup lookback in minutes (0 to disable)
    #[arg(long, default_value = "60")]
    warmup_minutes: i64,

    /// Block trading until warmup complete
    #[arg(long, default_value = "true")]
    block_until_warm: bool,
}

/// Paper position
#[derive(Debug, Clone)]
struct Position {
    symbol: String,
    strategy: StrategyType,
    entry_ts: DateTime<Utc>,
    entry_price: f64,
    quantity: i32,
    current_price: f64,
    unrealized_pnl: f64,
}

/// Trade record for activity log
#[derive(Debug, Clone)]
struct TradeRecord {
    ts: DateTime<Utc>,
    symbol: String,
    action: String,
    price: f64,
    pnl: f64,
}

/// Market quote for a symbol
#[derive(Debug, Clone)]
struct Quote {
    symbol: String,
    price: f64,
    ts: DateTime<Utc>,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
struct Stats {
    initial_capital: f64,
    capital: f64,
    ticks: u64,
    entry_signals: u64,
    hft_blocks: u64,
    regime_blocks: u64,
    winning_trades: u32,
    losing_trades: u32,
    peak_equity: f64,
    max_drawdown: f64,
    start_time: Option<DateTime<Utc>>,
}

/// Gate status for display
#[derive(Debug, Clone, Default)]
struct GateStatus {
    ramanujan_active: bool,
    hft_detected: bool,
    hft_confidence: f64,
    regime: String,
    regime_confidence: f64,
    // Market data
    spot: f64,
    futures: f64,
    basis: f64, // futures - spot
    atm_iv: f64,
    iv_percentile: f64,
    realized_vol: f64,
    pcr: f64,
    pcr_signal: String,
    vol_regime: String,
}

/// Strategy info for display when strategy changes
#[derive(Debug, Clone, Default)]
struct StrategyInfo {
    current_strategy: String,
    score: f64,
    min_score_threshold: f64,
    regime_score: f64,
    vol_score: f64,
    pcr_score: f64,
    risk_score: f64,
    edge_score: f64,
    reasoning: Vec<String>,
    last_change_ts: Option<DateTime<Utc>>,
}

/// Warmup progress for display
#[derive(Debug, Clone, Default)]
struct WarmupDisplay {
    state: WarmupState,
    symbols_fetched: usize,
    symbols_total: usize,
    candles_processed: usize,
    candles_total: usize,
}

impl WarmupDisplay {
    fn progress_pct(&self) -> f64 {
        match self.state {
            WarmupState::Fetching => {
                if self.symbols_total == 0 {
                    0.0
                } else {
                    (self.symbols_fetched as f64 / self.symbols_total as f64) * 50.0
                }
            }
            WarmupState::Processing => {
                if self.candles_total == 0 {
                    50.0
                } else {
                    50.0 + (self.candles_processed as f64 / self.candles_total as f64) * 50.0
                }
            }
            WarmupState::Ready => 100.0,
            WarmupState::Failed => 0.0,
        }
    }
}

/// Shared application state
struct AppState {
    positions: HashMap<String, Position>,
    trades: Vec<TradeRecord>,
    quotes: HashMap<String, Quote>,
    stats: Stats,
    gates: GateStatus,
    strategy: StrategyInfo,
    warmup: WarmupDisplay,
    logs: Vec<String>,
    running: bool,
    block_until_warm: bool,
}

impl AppState {
    fn new(capital: f64, block_until_warm: bool, min_score: f64) -> Self {
        Self {
            positions: HashMap::new(),
            trades: Vec::new(),
            quotes: HashMap::new(),
            stats: Stats {
                initial_capital: capital,
                capital,
                peak_equity: capital,
                start_time: Some(Utc::now()),
                ..Default::default()
            },
            gates: GateStatus::default(),
            strategy: StrategyInfo {
                min_score_threshold: min_score,
                ..Default::default()
            },
            warmup: WarmupDisplay::default(),
            logs: vec!["Session started".to_string()],
            running: true,
            block_until_warm,
        }
    }

    fn can_trade(&self) -> bool {
        if self.block_until_warm {
            self.warmup.state == WarmupState::Ready
        } else {
            true
        }
    }

    fn equity(&self) -> f64 {
        let unrealized: f64 = self.positions.values().map(|p| p.unrealized_pnl).sum();
        self.stats.capital + unrealized
    }

    fn total_pnl(&self) -> f64 {
        self.equity() - self.stats.initial_capital
    }

    fn return_pct(&self) -> f64 {
        self.total_pnl() / self.stats.initial_capital * 100.0
    }

    fn add_log(&mut self, msg: String) {
        self.logs.push(msg);
        if self.logs.len() > 100 {
            self.logs.remove(0);
        }
    }
}

type SharedState = Arc<Mutex<AppState>>;

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the app
    let result = run_app(&mut terminal, args);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>, args: Args) -> Result<()> {
    // Show loading screen
    terminal.draw(|f| {
        let area = f.area();
        let block = Block::default()
            .title(" QuantLaxmi Options Engine ")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Cyan));
        let text = Paragraph::new("Connecting to Zerodha...")
            .block(block)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(text, area);
    })?;

    // Initialize Zerodha connection
    let mut symbol_list: Vec<String> = args
        .symbols
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Add NIFTY 50 index for spot price tracking
    symbol_list.push("NIFTY 50".to_string());

    // Add current month and next month futures - construct from current date
    // Zerodha futures format: NIFTY + YY + MON + FUT (e.g., NIFTY26JANFUT)
    let now = chrono::Utc::now();
    let current_month_futures = format!(
        "NIFTY{}{}FUT",
        now.format("%y"),                            // YY
        now.format("%b").to_string().to_uppercase()  // MON (e.g., JAN, FEB)
    );
    symbol_list.push(current_month_futures.clone());

    // Also add next month in case current month has expired
    let next_month = now + chrono::Duration::days(32);
    let next_month_futures = format!(
        "NIFTY{}{}FUT",
        next_month.format("%y"),
        next_month.format("%b").to_string().to_uppercase()
    );
    if next_month_futures != current_month_futures {
        symbol_list.push(next_month_futures.clone());
    }

    // Log what we're subscribing to
    eprintln!(
        "[TUI] Subscribing to: NIFTY 50, {}, {}",
        current_month_futures, next_month_futures
    );

    let discovery =
        ZerodhaAutoDiscovery::from_sidecar().context("Failed to authenticate with Zerodha")?;

    // Create tokio runtime for async operations
    let rt = tokio::runtime::Runtime::new()?;

    // Resolve ALL symbols including index and futures
    let resolved_symbols = rt.block_on(async { discovery.resolve_symbols(&symbol_list).await })?;

    if resolved_symbols.is_empty() {
        anyhow::bail!("No symbols resolved");
    }

    // Extract just the symbol names for the connector
    let symbols: Vec<String> = resolved_symbols.iter().map(|(s, _)| s.clone()).collect();

    // Log resolved symbols
    eprintln!("[TUI] Resolved {} symbols", resolved_symbols.len());
    for (sym, token) in &resolved_symbols {
        if sym.contains("NIFTY 50") || sym.contains("FUT") {
            eprintln!("[TUI]   {} -> token {}", sym, token);
        }
    }

    // Create shared state
    let state = Arc::new(Mutex::new(AppState::new(
        args.capital,
        args.block_until_warm,
        args.min_score,
    )));

    // Add resolved symbols to log
    {
        let mut s = state.lock().unwrap();
        let option_count = symbols
            .iter()
            .filter(|s| s.ends_with("CE") || s.ends_with("PE"))
            .count();
        s.add_log(format!(
            "Resolved {} symbols ({} options, index, futures)",
            symbols.len(),
            option_count
        ));
        // Warmup uses NIFTY index, not options (1 symbol)
        s.warmup.symbols_total = 1;
        if args.warmup_minutes > 0 {
            s.add_log(format!(
                "Warmup: fetching {} min of NIFTY index data...",
                args.warmup_minutes
            ));
        } else {
            s.warmup.state = WarmupState::Ready;
            s.add_log("Warmup disabled, trading immediately".to_string());
        }
    }

    // Setup signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    // Perform warmup if enabled
    let warmup_vectors = if args.warmup_minutes > 0 {
        terminal.draw(|f| {
            let area = f.area();
            let block = Block::default()
                .title(" QuantLaxmi Options Engine ")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Cyan));
            let text = Paragraph::new(format!(
                "Fetching {} minutes of NIFTY index historical data for warmup...",
                args.warmup_minutes
            ))
            .block(block)
            .style(Style::default().fg(Color::Yellow));
            f.render_widget(text, area);
        })?;

        // Fetch historical data for warmup from NIFTY INDEX (not options!)
        // NIFTY 50 index token is 256265 in Kite Connect
        let nifty_index_token = 256265u32;

        // Log to file during warmup (TUI can't show tracing output)
        use std::io::Write;
        let log_path = "/tmp/quantlaxmi_warmup.log";
        if let Ok(mut log) = std::fs::File::create(log_path) {
            let _ = writeln!(log, "Starting warmup fetch at {}", chrono::Utc::now());
            let _ = writeln!(log, "NIFTY 50 token: {}", nifty_index_token);
            let _ = writeln!(log, "Lookback: {} minutes", args.warmup_minutes);
        }

        let warmup_data = rt.block_on(async {
            let result = discovery
                .fetch_warmup_data(
                    &[("NIFTY 50".to_string(), nifty_index_token)],
                    args.warmup_minutes,
                )
                .await;

            // Log result
            if let Ok(mut log) = std::fs::OpenOptions::new().append(true).open(log_path) {
                match &result {
                    Ok(data) => {
                        let _ = writeln!(log, "Warmup fetch succeeded:");
                        for (sym, candles) in data {
                            let _ = writeln!(log, "  {}: {} candles", sym, candles.len());
                            if let Some(c) = candles.first() {
                                let _ =
                                    writeln!(log, "    First: {} close={}", c.timestamp, c.close);
                            }
                            if let Some(c) = candles.last() {
                                let _ =
                                    writeln!(log, "    Last: {} close={}", c.timestamp, c.close);
                            }
                        }
                    }
                    Err(e) => {
                        let _ = writeln!(log, "Warmup fetch failed: {}", e);
                    }
                }
            }

            result
        })?;

        {
            let mut s = state.lock().unwrap();
            s.warmup.symbols_fetched = warmup_data.len();
            s.warmup.state = WarmupState::Processing;
            let total_candles: usize = warmup_data.values().map(|v| v.len()).sum();
            s.warmup.candles_total = total_candles;
            s.add_log(format!(
                "Fetched {} candles for {} symbols",
                total_candles,
                warmup_data.len()
            ));
        }

        // Convert candles to feature vectors
        let vectors = aggregate_warmup_data(&warmup_data);

        {
            let mut s = state.lock().unwrap();
            s.warmup.candles_processed = vectors.len();
            if vectors.is_empty() {
                s.warmup.state = WarmupState::Failed;
                s.add_log("Warmup failed: no historical data available".to_string());
            } else {
                s.add_log(format!(
                    "Prepared {} warmup ticks for regime detector",
                    vectors.len()
                ));
            }
        }

        vectors
    } else {
        {
            let mut s = state.lock().unwrap();
            s.warmup.state = WarmupState::Ready;
            s.add_log("Warmup disabled".to_string());
        }
        Vec::new()
    };

    // Start market data processing in background
    let state_clone = state.clone();
    let running_clone = running.clone();
    let args_clone = Args {
        symbols: args.symbols.clone(),
        capital: args.capital,
        max_positions: args.max_positions,
        min_score: args.min_score,
        stop_loss: args.stop_loss,
        profit_target: args.profit_target,
        warmup_minutes: args.warmup_minutes,
        block_until_warm: args.block_until_warm,
    };

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            if let Err(e) = run_trading_loop(
                state_clone,
                running_clone,
                symbols,
                args_clone,
                warmup_vectors,
            )
            .await
            {
                eprintln!("Trading loop error: {}", e);
            }
        });
    });

    // Main UI loop
    loop {
        // Check if still running
        if !running.load(Ordering::SeqCst) {
            let mut s = state.lock().unwrap();
            s.running = false;
            break;
        }

        // Draw UI
        {
            let s = state.lock().unwrap();
            terminal.draw(|f| draw_ui(f, &s))?;
        }

        // Handle input with timeout
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            running.store(false, Ordering::SeqCst);
                            let mut s = state.lock().unwrap();
                            s.running = false;
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Show final summary
    {
        let s = state.lock().unwrap();
        terminal.draw(|f| draw_summary(f, &s))?;
    }

    // Wait for key press before exiting
    loop {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(_) = event::read()? {
                break;
            }
        }
    }

    Ok(())
}

async fn run_trading_loop(
    state: SharedState,
    running: Arc<AtomicBool>,
    symbols: Vec<String>,
    args: Args,
    warmup_vectors: Vec<(DateTime<Utc>, f64, FeatureVector)>,
) -> Result<()> {
    // Create event bus
    let bus = EventBus::new(10000);

    // Create connector
    let connector = ZerodhaConnector::new(bus.clone(), symbols.clone());

    // Initialize engine
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
    let mut prev_prices: HashMap<String, f64> = HashMap::new();
    let mut current_spot: f64 = 0.0; // NIFTY 50 index price
    let mut option_chain: HashMap<String, OptionData> = HashMap::new(); // Current option chain
    let mut last_chain_update = std::time::Instant::now();

    // Process warmup vectors through engine for regime detection
    if !warmup_vectors.is_empty() {
        {
            let mut s = state.lock().unwrap();
            s.add_log(format!(
                "Processing {} warmup ticks through engine...",
                warmup_vectors.len()
            ));
        }

        for (ts, spot, features) in &warmup_vectors {
            engine.on_tick(*ts, *spot, features);
        }

        // Check regime after warmup
        let status = engine.status();
        {
            let mut s = state.lock().unwrap();
            s.warmup.state = WarmupState::Ready;
            s.gates.regime = format!("{:?}", status.regime);
            s.add_log(format!("Warmup complete: regime = {:?}", status.regime));
        }
    } else {
        // No warmup data, mark as ready anyway but note it
        let mut s = state.lock().unwrap();
        s.warmup.state = WarmupState::Ready;
        s.add_log("No warmup data available, starting with Unknown regime".to_string());
    }

    // Subscribe to market data
    let mut market_rx = bus.subscribe_market();

    // Start connector
    let connector_running = running.clone();
    tokio::spawn(async move {
        use quantlaxmi_core::connector::MarketConnector;
        if let Err(e) = connector.run().await {
            if connector_running.load(Ordering::SeqCst) {
                eprintln!("Connector error: {}", e);
            }
        }
    });

    // Processing loop
    let mut current_futures: f64 = 0.0; // NIFTY futures price

    while running.load(Ordering::SeqCst) {
        match tokio::time::timeout(Duration::from_secs(1), market_rx.recv()).await {
            Ok(Ok(record)) => {
                let symbol = record.symbol.clone();
                let ts = record.ts;

                // Extract price (LTP only)
                let price = match &record.payload {
                    MarketPayload::Trade { price_mantissa, .. } => *price_mantissa as f64 / 100.0,
                    _ => continue,
                };

                if price <= 0.0 {
                    continue;
                }

                // Determine symbol type and update appropriate price
                let is_index = symbol == "NIFTY 50";
                let is_futures = symbol.contains("FUT");
                let _is_option = symbol.ends_with("CE") || symbol.ends_with("PE");

                // Update spot/futures prices and process regime ONLY on index ticks
                if is_index {
                    let prev_spot = current_spot;
                    current_spot = price;

                    // Calculate features from INDEX price changes
                    let mid_return = if prev_spot > 0.0 {
                        (((price - prev_spot) / prev_spot) * 10000.0) as i64
                    } else {
                        0
                    };

                    // Only feed regime engine when we have meaningful data
                    if prev_spot > 0.0 {
                        let features = FeatureVector::new(mid_return, 0, 50, 50, 50, 0);
                        engine.on_tick(ts, price, &features);
                    }

                    // Update state
                    let mut s = state.lock().unwrap();
                    s.gates.spot = price;
                    s.gates.basis = current_futures - price;
                    s.stats.ticks += 1;

                    // Get updated status after processing index tick
                    drop(s); // Release lock before calling engine
                    let status = engine.status();
                    let decision = engine.decide(ts);

                    let mut s = state.lock().unwrap();
                    s.gates.hft_detected = status.hft_detected;
                    s.gates.ramanujan_active = true;
                    s.gates.regime = format!("{:?}", status.regime);
                    s.gates.atm_iv = status.atm_iv * 100.0;
                    s.gates.iv_percentile = status.iv_percentile;
                    s.gates.realized_vol = status.realized_vol * 100.0;
                    s.gates.pcr = status.pcr;
                    s.gates.pcr_signal = format!("{:?}", status.pcr_signal);
                    s.gates.vol_regime = format!("{:?}", status.vol_regime);

                    // Update strategy info
                    if let Some(rec) = &decision.strategy {
                        let prev_strategy = s.strategy.current_strategy.clone();
                        let new_strategy = format!("{:?}", rec.strategy);
                        s.strategy.current_strategy = new_strategy.clone();
                        s.strategy.score = rec.score;
                        s.strategy.regime_score = rec.component_scores.regime_score;
                        s.strategy.vol_score = rec.component_scores.vol_score;
                        s.strategy.pcr_score = rec.component_scores.pcr_score;
                        s.strategy.risk_score = rec.component_scores.risk_score;
                        s.strategy.edge_score = rec.component_scores.edge_score;
                        s.strategy.reasoning = rec.reasoning.clone();

                        if !prev_strategy.is_empty() && prev_strategy != new_strategy {
                            s.strategy.last_change_ts = Some(ts);
                            let min_threshold = s.strategy.min_score_threshold;
                            s.add_log(format!(
                                "STRATEGY: {} -> {} (score: {:.1} >= {:.1})",
                                prev_strategy, new_strategy, rec.score, min_threshold
                            ));
                        }
                    }

                    continue; // Don't process index as tradeable
                }

                if is_futures {
                    current_futures = price;
                    let mut s = state.lock().unwrap();
                    s.gates.futures = price;
                    s.gates.basis = price - current_spot;
                    s.stats.ticks += 1;
                    continue; // Don't process futures as tradeable option
                }

                // Update quote for options only
                {
                    let mut s = state.lock().unwrap();
                    s.quotes.insert(
                        symbol.clone(),
                        Quote {
                            symbol: symbol.clone(),
                            price,
                            ts,
                        },
                    );
                    s.stats.ticks += 1;
                }

                // Parse option data and add to chain
                if let Some(opt_data) = parse_option_symbol(&symbol, price) {
                    let is_new = !option_chain.contains_key(&symbol);
                    let strike = opt_data.strike; // Copy before move
                    option_chain.insert(symbol.clone(), opt_data);

                    // Log first few options added
                    if is_new && option_chain.len() <= 6 {
                        let mut s = state.lock().unwrap();
                        s.add_log(format!(
                            "OPT: {} strike={:.0} price={:.2}",
                            if symbol.ends_with("CE") { "CE" } else { "PE" },
                            strike,
                            price
                        ));
                    }
                }

                // Periodically update engine with option chain (every 1 second)
                if current_spot > 0.0 && last_chain_update.elapsed() > Duration::from_secs(1) {
                    let options: Vec<OptionData> = option_chain.values().cloned().collect();
                    if !options.is_empty() {
                        // Count ATM options for debugging
                        let atm_count = options
                            .iter()
                            .filter(|o| (o.strike - current_spot).abs() < current_spot * 0.02)
                            .count();

                        engine.on_chain_update(ts, &options, current_spot);

                        // Get updated status AFTER chain update and refresh display
                        let status = engine.status();
                        let mut s = state.lock().unwrap();
                        s.gates.atm_iv = status.atm_iv * 100.0;
                        s.gates.iv_percentile = status.iv_percentile;
                        s.gates.realized_vol = status.realized_vol * 100.0;
                        s.gates.pcr = status.pcr;
                        s.gates.pcr_signal = format!("{:?}", status.pcr_signal);
                        s.gates.vol_regime = format!("{:?}", status.vol_regime);

                        // Log chain update results periodically
                        if s.stats.ticks % 100 == 0 {
                            s.add_log(format!(
                                "Chain: {} opts ({} ATM), IV={:.1}%, PCR={:.2}",
                                options.len(),
                                atm_count,
                                status.atm_iv * 100.0,
                                status.pcr
                            ));
                        }
                    }
                    last_chain_update = std::time::Instant::now();
                }

                // Skip if we don't have a valid spot price yet
                if current_spot <= 0.0 {
                    prev_prices.insert(symbol, price);
                    continue;
                }

                // For option ticks: check if we should trade (using current regime state)
                let decision = engine.decide(ts);
                let status = engine.status();

                {
                    let mut s = state.lock().unwrap();
                    if matches!(decision.action, TradingAction::Enter) {
                        s.stats.entry_signals += 1;
                    }
                    if status.hft_detected {
                        s.stats.hft_blocks += 1;
                    }
                }

                // Process trading decision (only if warmup complete and no HFT)
                let can_trade = {
                    let s = state.lock().unwrap();
                    s.can_trade()
                };

                if !status.hft_detected && can_trade {
                    if let TradingAction::Enter = decision.action {
                        if let Some(rec) = &decision.strategy {
                            let mut s = state.lock().unwrap();
                            let has_position = s.positions.contains_key(&symbol);
                            if !has_position && s.positions.len() < args.max_positions as usize {
                                let quantity = 50;

                                s.positions.insert(
                                    symbol.clone(),
                                    Position {
                                        symbol: symbol.clone(),
                                        strategy: rec.strategy,
                                        entry_ts: ts,
                                        entry_price: price,
                                        quantity,
                                        current_price: price,
                                        unrealized_pnl: 0.0,
                                    },
                                );

                                s.trades.push(TradeRecord {
                                    ts,
                                    symbol: symbol.clone(),
                                    action: format!("ENTER {:?}", rec.strategy),
                                    price,
                                    pnl: 0.0,
                                });

                                s.add_log(format!(
                                    "ENTER {:?} {} @ {:.2}",
                                    rec.strategy, symbol, price
                                ));
                            }
                        }
                    }
                }

                // Update position MTM
                {
                    let mut s = state.lock().unwrap();
                    if let Some(pos) = s.positions.get_mut(&symbol) {
                        pos.current_price = price;
                        pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity as f64;
                    }

                    // Check stop loss / profit target
                    let symbols_to_close: Vec<String> = s
                        .positions
                        .iter()
                        .filter(|(_, pos)| {
                            pos.unrealized_pnl < -args.stop_loss
                                || pos.unrealized_pnl > args.profit_target
                        })
                        .map(|(sym, _)| sym.clone())
                        .collect();

                    for sym in symbols_to_close {
                        if let Some(pos) = s.positions.remove(&sym) {
                            let action = if pos.unrealized_pnl > 0.0 {
                                "EXIT_PROFIT"
                            } else {
                                "EXIT_STOP"
                            };

                            s.stats.capital += pos.unrealized_pnl;

                            if pos.unrealized_pnl > 0.0 {
                                s.stats.winning_trades += 1;
                            } else {
                                s.stats.losing_trades += 1;
                            }

                            s.trades.push(TradeRecord {
                                ts,
                                symbol: sym.clone(),
                                action: action.to_string(),
                                price: pos.current_price,
                                pnl: pos.unrealized_pnl,
                            });

                            s.add_log(format!(
                                "{} {} @ {:.2} P&L: {:+.2}",
                                action, sym, pos.current_price, pos.unrealized_pnl
                            ));
                        }
                    }

                    // Track drawdown
                    let equity = s.equity();
                    if equity > s.stats.peak_equity {
                        s.stats.peak_equity = equity;
                    }
                    let drawdown = (s.stats.peak_equity - equity) / s.stats.peak_equity * 100.0;
                    if drawdown > s.stats.max_drawdown {
                        s.stats.max_drawdown = drawdown;
                    }
                }

                prev_prices.insert(symbol, price);
            }
            Ok(Err(_)) => break,
            Err(_) => {} // Timeout
        }
    }

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

/// Parse option symbol to OptionData
fn parse_option_symbol(symbol: &str, price: f64) -> Option<OptionData> {
    // Symbol format: NIFTY2620325300CE (NIFTY + YYMM + DD + strike + CE/PE)
    if symbol.len() < 14 {
        return None;
    }

    let option_type = if symbol.ends_with("CE") {
        OptionDataType::Call
    } else if symbol.ends_with("PE") {
        OptionDataType::Put
    } else {
        return None;
    };

    let strike = extract_strike_from_symbol(symbol)?;

    // Estimate DTE from symbol (YYMM format at position 5-9)
    // For simplicity, assume weekly expiry ~7 DTE
    let expiry_dte = 7u32;

    Some(OptionData {
        strike,
        expiry_dte,
        option_type,
        volume: 0,        // Not available from tick data
        open_interest: 0, // Not available from tick data
        last_price: price,
        delta: 0.0, // Will be calculated by engine
    })
}

fn draw_ui(f: &mut Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Footer
        ])
        .split(f.area());

    draw_header(f, chunks[0], state);
    draw_main_content(f, chunks[1], state);
    draw_footer(f, chunks[2], state);
}

fn draw_header(f: &mut Frame, area: Rect, state: &AppState) {
    let elapsed = state
        .stats
        .start_time
        .map(|st| Utc::now().signed_duration_since(st))
        .map(|d| {
            format!(
                "{}:{:02}:{:02}",
                d.num_hours(),
                d.num_minutes() % 60,
                d.num_seconds() % 60
            )
        })
        .unwrap_or_else(|| "00:00:00".to_string());

    let pnl = state.total_pnl();
    let pnl_color = if pnl >= 0.0 { Color::Green } else { Color::Red };

    let title = Line::from(vec![
        Span::styled(
            " QuantLaxmi ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("Options Engine | "),
        Span::styled(
            "PAPER TRADING",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(" | Elapsed: {} | ", elapsed)),
        Span::styled(
            format!("P&L: {:+.2} ({:+.2}%)", pnl, state.return_pct()),
            Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
        ),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    let paragraph = Paragraph::new(title).block(block);
    f.render_widget(paragraph, area);
}

fn draw_main_content(f: &mut Frame, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(area);

    // Left column: positions and quotes
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);

    // Middle column: market data and gates
    let middle_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Right column: strategy and activity
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(30),
            Constraint::Percentage(30),
        ])
        .split(chunks[2]);

    draw_positions_panel(f, left_chunks[0], state);
    draw_quotes_panel(f, left_chunks[1], state);
    draw_market_data_panel(f, middle_chunks[0], state);
    draw_gates_panel(f, middle_chunks[1], state);
    draw_stats_panel(f, right_chunks[0], state);
    draw_strategy_panel(f, right_chunks[1], state);
    draw_activity_panel(f, right_chunks[2], state);
}

fn draw_positions_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Positions ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Green));

    if state.positions.is_empty() {
        let text = Paragraph::new("No open positions")
            .block(block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, area);
        return;
    }

    let header = Row::new(vec!["Symbol", "Strategy", "Entry", "Current", "P&L"])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let rows: Vec<Row> = state
        .positions
        .values()
        .map(|pos| {
            let pnl_style = if pos.unrealized_pnl >= 0.0 {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            };

            Row::new(vec![
                Cell::from(pos.symbol.chars().skip(5).take(15).collect::<String>()),
                Cell::from(format!("{:?}", pos.strategy)),
                Cell::from(format!("{:.2}", pos.entry_price)),
                Cell::from(format!("{:.2}", pos.current_price)),
                Cell::from(format!("{:+.2}", pos.unrealized_pnl)).style(pnl_style),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(25),
            Constraint::Percentage(20),
            Constraint::Percentage(18),
            Constraint::Percentage(18),
            Constraint::Percentage(19),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn draw_quotes_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Market Data ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Blue));

    if state.quotes.is_empty() {
        let text = Paragraph::new("Waiting for market data...")
            .block(block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, area);
        return;
    }

    let header = Row::new(vec!["Symbol", "LTP", "Time"])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let mut quotes: Vec<_> = state.quotes.values().collect();
    quotes.sort_by(|a, b| a.symbol.cmp(&b.symbol));

    let rows: Vec<Row> = quotes
        .iter()
        .map(|q| {
            Row::new(vec![
                Cell::from(q.symbol.chars().skip(5).take(15).collect::<String>()),
                Cell::from(format!("{:.2}", q.price)),
                Cell::from(q.ts.with_timezone(&Local).format("%H:%M:%S").to_string()),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Percentage(50),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ],
    )
    .header(header)
    .block(block);

    f.render_widget(table, area);
}

fn draw_stats_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Statistics ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Magenta));

    let total_trades = state.stats.winning_trades + state.stats.losing_trades;
    let win_rate = if total_trades > 0 {
        state.stats.winning_trades as f64 / total_trades as f64 * 100.0
    } else {
        0.0
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("Capital:     "),
            Span::styled(
                format!("{:.2}", state.stats.initial_capital),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Equity:      "),
            Span::styled(
                format!("{:.2}", state.equity()),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("Max DD:      "),
            Span::styled(
                format!("{:.2}%", state.stats.max_drawdown),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::raw("Ticks:       "),
            Span::styled(
                format!("{}", state.stats.ticks),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Signals:     "),
            Span::styled(
                format!("{}", state.stats.entry_signals),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("Trades:      "),
            Span::styled(
                format!(
                    "{} (W:{} L:{})",
                    total_trades, state.stats.winning_trades, state.stats.losing_trades
                ),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Win Rate:    "),
            Span::styled(
                format!("{:.1}%", win_rate),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];

    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

fn draw_market_data_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Market Data ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    // Color coding for vol regime
    let vol_color = if state.gates.vol_regime.contains("High") {
        Color::Red
    } else if state.gates.vol_regime.contains("Low") {
        Color::Green
    } else {
        Color::Yellow
    };

    // Color for PCR signal
    let pcr_color = if state.gates.pcr_signal.contains("Bullish") {
        Color::Green
    } else if state.gates.pcr_signal.contains("Bearish") {
        Color::Red
    } else {
        Color::Yellow
    };

    // Regime color based on type
    let regime_color = if state.gates.regime.contains("Unknown") {
        Color::Yellow
    } else if state.gates.regime.contains("Quiet") {
        Color::Green
    } else if state.gates.regime.contains("Trend") {
        Color::Cyan
    } else if state.gates.regime.contains("Chop") || state.gates.regime.contains("MeanReversion") {
        Color::Magenta
    } else if state.gates.regime.contains("Drought") || state.gates.regime.contains("Shock") {
        Color::Red
    } else {
        Color::White
    };

    // Basis color: green for contango (futures > spot), red for backwardation
    let basis_color = if state.gates.basis > 0.0 {
        Color::Green
    } else if state.gates.basis < 0.0 {
        Color::Red
    } else {
        Color::White
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("Spot:       "),
            Span::styled(
                format!("{:.2}", state.gates.spot),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Futures:    "),
            Span::styled(
                format!("{:.2}", state.gates.futures),
                Style::default().fg(Color::White),
            ),
            Span::raw(" ("),
            Span::styled(
                format!("{:+.1}", state.gates.basis),
                Style::default().fg(basis_color),
            ),
            Span::raw(")"),
        ]),
        Line::from(vec![
            Span::raw("Regime:     "),
            Span::styled(&state.gates.regime, Style::default().fg(regime_color)),
        ]),
        Line::from(vec![
            Span::raw("ATM IV:     "),
            Span::styled(
                format!("{:.1}%", state.gates.atm_iv),
                Style::default().fg(Color::White),
            ),
            Span::raw(format!(" ({}th pctl)", state.gates.iv_percentile as u32)),
        ]),
        Line::from(vec![
            Span::raw("Vol Regime: "),
            Span::styled(&state.gates.vol_regime, Style::default().fg(vol_color)),
        ]),
        Line::from(vec![
            Span::raw("PCR:        "),
            Span::styled(
                format!("{:.2}", state.gates.pcr),
                Style::default().fg(Color::White),
            ),
            Span::raw(" "),
            Span::styled(&state.gates.pcr_signal, Style::default().fg(pcr_color)),
        ]),
    ];

    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

fn draw_gates_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Gates & Warmup ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Yellow));

    // Warmup status
    let (warmup_text, warmup_color) = match state.warmup.state {
        WarmupState::Fetching => ("Fetching...", Color::Yellow),
        WarmupState::Processing => ("Processing...", Color::Yellow),
        WarmupState::Ready => ("Ready", Color::Green),
        WarmupState::Failed => ("Failed", Color::Red),
    };

    let hft_status = if state.gates.hft_detected {
        Span::styled(
            "DETECTED",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled("Clear", Style::default().fg(Color::Green))
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("Warmup:     "),
            Span::styled(
                format!("{} ({:.0}%)", warmup_text, state.warmup.progress_pct()),
                Style::default().fg(warmup_color),
            ),
        ]),
        Line::from(vec![
            Span::raw("Ramanujan:  "),
            Span::styled("Active", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![Span::raw("HFT Status: "), hft_status]),
        Line::from(vec![
            Span::raw("HFT Blocks: "),
            Span::styled(
                format!("{}", state.stats.hft_blocks),
                Style::default().fg(Color::Yellow),
            ),
        ]),
    ];

    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

fn draw_strategy_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Strategy ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Magenta));

    if state.strategy.current_strategy.is_empty() {
        let text = Paragraph::new("Waiting for strategy selection...")
            .block(block)
            .style(Style::default().fg(Color::DarkGray));
        f.render_widget(text, area);
        return;
    }

    // Color based on whether score meets threshold
    let score_color = if state.strategy.score >= state.strategy.min_score_threshold {
        Color::Green
    } else {
        Color::Red
    };

    let lines = vec![
        Line::from(vec![
            Span::raw("Strategy:  "),
            Span::styled(
                &state.strategy.current_strategy,
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::raw("Score:     "),
            Span::styled(
                format!("{:.1}", state.strategy.score),
                Style::default().fg(score_color),
            ),
            Span::raw(format!(" / {:.1}", state.strategy.min_score_threshold)),
        ]),
        Line::from(vec![Span::raw("Components:")]),
        Line::from(vec![
            Span::raw("  Regime:  "),
            Span::styled(
                format!("{:.1}", state.strategy.regime_score),
                Style::default().fg(Color::White),
            ),
            Span::raw("  Vol: "),
            Span::styled(
                format!("{:.1}", state.strategy.vol_score),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("  PCR:     "),
            Span::styled(
                format!("{:.1}", state.strategy.pcr_score),
                Style::default().fg(Color::White),
            ),
            Span::raw("  Edge: "),
            Span::styled(
                format!("{:.1}", state.strategy.edge_score),
                Style::default().fg(Color::White),
            ),
        ]),
    ];

    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

fn draw_activity_panel(f: &mut Frame, area: Rect, state: &AppState) {
    let block = Block::default()
        .title(" Activity ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::White));

    let items: Vec<ListItem> = state
        .logs
        .iter()
        .rev()
        .take(5)
        .map(|log| {
            let style = if log.contains("ENTER") {
                Style::default().fg(Color::Green)
            } else if log.contains("EXIT_PROFIT") {
                Style::default().fg(Color::Cyan)
            } else if log.contains("EXIT_STOP") {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            ListItem::new(log.as_str()).style(style)
        })
        .collect();

    let list = List::new(items).block(block);
    f.render_widget(list, area);
}

fn draw_footer(f: &mut Frame, area: Rect, state: &AppState) {
    let status = if state.running { "LIVE" } else { "STOPPED" };
    let status_color = if state.running {
        Color::Green
    } else {
        Color::Red
    };

    let text = Line::from(vec![
        Span::raw(" Status: "),
        Span::styled(
            status,
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | Press "),
        Span::styled(
            "Q",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" to quit | "),
        Span::raw(format!("Positions: {}/{} | ", state.positions.len(), 3)),
        Span::raw(Local::now().format("%Y-%m-%d %H:%M:%S").to_string()),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

fn draw_summary(f: &mut Frame, state: &AppState) {
    let block = Block::default()
        .title(" Session Summary - Press any key to exit ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));

    let pnl = state.total_pnl();
    let pnl_color = if pnl >= 0.0 { Color::Green } else { Color::Red };

    let total_trades = state.stats.winning_trades + state.stats.losing_trades;
    let win_rate = if total_trades > 0 {
        state.stats.winning_trades as f64 / total_trades as f64 * 100.0
    } else {
        0.0
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            "  CAPITAL",
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(format!(
            "    Initial:         {:.2}",
            state.stats.initial_capital
        )),
        Line::from(format!("    Final:           {:.2}", state.equity())),
        Line::from(vec![
            Span::raw("    Total P&L:       "),
            Span::styled(
                format!("{:+.2} ({:+.2}%)", pnl, state.return_pct()),
                Style::default().fg(pnl_color),
            ),
        ]),
        Line::from(format!(
            "    Max Drawdown:    {:.2}%",
            state.stats.max_drawdown
        )),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  ACTIVITY",
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(format!("    Market Ticks:    {}", state.stats.ticks)),
        Line::from(format!(
            "    Entry Signals:   {}",
            state.stats.entry_signals
        )),
        Line::from(format!("    HFT Blocks:      {}", state.stats.hft_blocks)),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  TRADES",
            Style::default().add_modifier(Modifier::BOLD),
        )]),
        Line::from(format!("    Total:           {}", total_trades)),
        Line::from(format!(
            "    Winners:         {} ({:.1}%)",
            state.stats.winning_trades, win_rate
        )),
        Line::from(format!(
            "    Losers:          {}",
            state.stats.losing_trades
        )),
        Line::from(format!("    Still Open:      {}", state.positions.len())),
        Line::from(""),
    ];

    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, f.area());
}
