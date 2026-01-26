//! Backtest Engine for Crypto Strategies
//!
//! Provides:
//! - `Strategy` trait for pluggable strategy logic
//! - `PaperExchange` for simulated order execution
//! - `BacktestEngine` to orchestrate replay → strategy → exchange
//!
//! ## Usage
//! ```ignore
//! let engine = BacktestEngine::new(config);
//! let result = engine.run(segment_dir, strategy)?;
//! println!("PnL: {:.2}", result.realized_pnl);
//! ```

use crate::replay::{EventKind, ReplayEvent, SegmentReplayAdapter};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Pacing Mode
// =============================================================================

/// Pacing mode for replay.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PaceMode {
    /// No delays - run as fast as possible (default).
    #[default]
    Fast,
    /// Real-time pacing - sleep between events to match original timestamps.
    #[serde(rename = "real")]
    RealTime,
}

// =============================================================================
// Order and Fill Types
// =============================================================================

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Order intent from strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntent {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    /// Optional limit price. If None, treated as market order.
    pub limit_price: Option<f64>,
    /// Strategy-defined tag for tracking.
    pub tag: Option<String>,
}

impl OrderIntent {
    pub fn market(symbol: impl Into<String>, side: Side, qty: f64) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            qty,
            limit_price: None,
            tag: None,
        }
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }
}

/// Execution fill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    pub tag: Option<String>,
}

// =============================================================================
// Trade Metrics
// =============================================================================

/// A completed round-trip trade (entry + exit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundTrip {
    pub symbol: String,
    pub side: Side, // Entry side (Buy = long trade, Sell = short trade)
    pub entry_ts: DateTime<Utc>,
    pub exit_ts: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub qty: f64,
    pub entry_fee: f64,
    pub exit_fee: f64,
    pub pnl: f64,     // Net PnL after fees
    pub pnl_pct: f64, // PnL as percentage of entry notional
    pub duration_secs: f64,
}

/// Point on the equity curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub ts: DateTime<Utc>,
    pub equity: f64,
    pub drawdown: f64,     // Current drawdown from peak
    pub drawdown_pct: f64, // Drawdown as percentage
}

/// Aggregated trade metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradeMetrics {
    // Trade counts
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,

    // PnL stats
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub net_pnl: f64,
    pub profit_factor: f64, // gross_profit / |gross_loss|
    pub expectancy: f64,    // Average PnL per trade

    // Win/loss averages
    pub avg_win: f64,
    pub avg_loss: f64,
    pub avg_win_loss_ratio: f64, // avg_win / |avg_loss|
    pub largest_win: f64,
    pub largest_loss: f64,

    // Risk metrics
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,  // Annualized, assumes 365 days
    pub sortino_ratio: f64, // Downside deviation only

    // Trade timing
    pub avg_trade_duration_secs: f64,
    pub total_fees: f64,
}

impl TradeMetrics {
    /// Compute metrics from round-trip trades and equity curve.
    pub fn compute(
        trades: &[RoundTrip],
        equity_curve: &[EquityPoint],
        _initial_capital: f64,
        duration_secs: f64,
    ) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        let total_trades = trades.len();
        let mut winning_trades = 0usize;
        let mut losing_trades = 0usize;
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;
        let mut largest_win = 0.0f64;
        let mut largest_loss = 0.0f64;
        let mut total_duration = 0.0;
        let mut total_fees = 0.0;
        let mut returns: Vec<f64> = Vec::with_capacity(trades.len());

        for trade in trades {
            let pnl = trade.pnl;
            returns.push(trade.pnl_pct);
            total_duration += trade.duration_secs;
            total_fees += trade.entry_fee + trade.exit_fee;

            if pnl >= 0.0 {
                winning_trades += 1;
                gross_profit += pnl;
                largest_win = largest_win.max(pnl);
            } else {
                losing_trades += 1;
                gross_loss += pnl.abs();
                largest_loss = largest_loss.max(pnl.abs());
            }
        }

        let net_pnl = gross_profit - gross_loss;
        let win_rate = if total_trades > 0 {
            (winning_trades as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        let avg_win = if winning_trades > 0 {
            gross_profit / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            gross_loss / losing_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let expectancy = net_pnl / total_trades as f64;

        let avg_win_loss_ratio = if avg_loss > 0.0 {
            avg_win / avg_loss
        } else if avg_win > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Max drawdown from equity curve
        let (max_drawdown, max_drawdown_pct) = equity_curve
            .iter()
            .map(|p| (p.drawdown, p.drawdown_pct))
            .fold((0.0f64, 0.0f64), |(max_dd, max_pct), (dd, pct)| {
                (max_dd.max(dd), max_pct.max(pct))
            });

        // Sharpe ratio (annualized)
        // Minimum std_dev threshold to avoid blow-up from near-identical returns
        const MIN_STD_DEV: f64 = 0.01; // 0.01% minimum standard deviation
        const MAX_RATIO: f64 = 99.0; // Clamp to reasonable bounds

        let (sharpe_ratio, sortino_ratio) = if returns.len() > 1 && duration_secs > 0.0 {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (returns.len() - 1) as f64;
            let std_dev = variance.sqrt();

            // Downside deviation (only negative returns)
            let downside_variance: f64 = returns
                .iter()
                .filter(|&&r| r < 0.0)
                .map(|r| r.powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let downside_dev = downside_variance.sqrt();

            // Annualize: assume trades are spread over duration_secs
            // trades_per_year = total_trades * (365 * 24 * 3600 / duration_secs)
            let annual_factor = (365.0 * 24.0 * 3600.0 / duration_secs).sqrt();

            let sharpe = if std_dev >= MIN_STD_DEV {
                ((mean_return / std_dev) * annual_factor).clamp(-MAX_RATIO, MAX_RATIO)
            } else {
                0.0 // Insufficient variance for meaningful Sharpe
            };

            let sortino = if downside_dev >= MIN_STD_DEV {
                ((mean_return / downside_dev) * annual_factor).clamp(-MAX_RATIO, MAX_RATIO)
            } else if mean_return > 0.0 {
                MAX_RATIO // All positive returns, cap at max
            } else {
                0.0
            };

            (sharpe, sortino)
        } else {
            (0.0, 0.0)
        };

        Self {
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            gross_profit,
            gross_loss,
            net_pnl,
            profit_factor,
            expectancy,
            avg_win,
            avg_loss,
            avg_win_loss_ratio,
            largest_win,
            largest_loss,
            max_drawdown,
            max_drawdown_pct,
            sharpe_ratio,
            sortino_ratio,
            avg_trade_duration_secs: total_duration / total_trades as f64,
            total_fees,
        }
    }
}

/// Extract round-trip trades from a list of fills.
///
/// Matches entry fills to exit fills based on tags (entry/exit pairs).
pub fn extract_round_trips(fills: &[Fill]) -> Vec<RoundTrip> {
    let mut trades = Vec::new();
    let mut pending_entries: Vec<&Fill> = Vec::new();

    for fill in fills {
        let tag = fill.tag.as_deref().unwrap_or("");

        if tag.contains("entry") {
            pending_entries.push(fill);
        } else if tag.contains("exit") && !pending_entries.is_empty() {
            // Match with the oldest pending entry for the same symbol
            if let Some(idx) = pending_entries.iter().position(|e| e.symbol == fill.symbol) {
                let entry = pending_entries.remove(idx);
                let entry_notional = entry.price * entry.qty;
                let exit_notional = fill.price * fill.qty;

                // Calculate PnL based on trade direction
                let gross_pnl = match entry.side {
                    Side::Buy => exit_notional - entry_notional, // Long: sell high - buy low
                    Side::Sell => entry_notional - exit_notional, // Short: sell high - buy low
                };
                let net_pnl = gross_pnl - entry.fee - fill.fee;
                let pnl_pct = (net_pnl / entry_notional) * 100.0;

                let duration_secs = (fill.ts - entry.ts).num_milliseconds() as f64 / 1000.0;

                trades.push(RoundTrip {
                    symbol: entry.symbol.clone(),
                    side: entry.side,
                    entry_ts: entry.ts,
                    exit_ts: fill.ts,
                    entry_price: entry.price,
                    exit_price: fill.price,
                    qty: entry.qty,
                    entry_fee: entry.fee,
                    exit_fee: fill.fee,
                    pnl: net_pnl,
                    pnl_pct,
                    duration_secs,
                });
            }
        }
    }

    trades
}

// =============================================================================
// Strategy Trait
// =============================================================================

/// Strategy interface for backtest and paper trading.
///
/// Implement this trait to create a trading strategy.
pub trait Strategy: Send {
    /// Called for each replay event. Return order intents to execute.
    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent>;

    /// Called when an order is filled.
    fn on_fill(&mut self, fill: &Fill) {
        let _ = fill; // Default: ignore
    }

    /// Strategy name for logging.
    fn name(&self) -> &str {
        "unnamed_strategy"
    }
}

// =============================================================================
// Paper Exchange
// =============================================================================

/// Per-symbol market state.
#[derive(Debug, Clone, Default)]
struct MarketState {
    bid: f64,
    ask: f64,
    last_update: Option<DateTime<Utc>>,
}

/// Per-symbol position.
#[derive(Debug, Clone, Default)]
struct Position {
    qty: f64,       // Positive = long, negative = short
    avg_price: f64, // Average entry price
}

/// Paper exchange configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    /// Fee in basis points (e.g., 10 = 0.1%)
    pub fee_bps: f64,
    /// Initial cash balance
    pub initial_cash: f64,
    /// Use perp prices for execution (vs spot)
    pub use_perp_prices: bool,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            fee_bps: 10.0, // 0.1% taker fee
            initial_cash: 10_000.0,
            use_perp_prices: true,
        }
    }
}

/// Extract bid/ask prices from event payload.
/// Handles both float format (bid/ask) and mantissa format (bid_price_mantissa + price_exponent).
fn extract_bid_ask(payload: &serde_json::Value, kind: EventKind) -> (f64, f64) {
    // Try float format first (simple quotes)
    if let (Some(bid), Some(ask)) = (
        payload.get("bid").and_then(|v| v.as_f64()),
        payload.get("ask").and_then(|v| v.as_f64()),
    ) {
        return (bid, ask);
    }

    // Try mantissa format (canonical quote schema)
    // price = mantissa * 10^price_exponent
    // e.g., mantissa=8874152, exponent=-2 → price = 8874152 * 0.01 = 88741.52
    let price_exp = payload
        .get("price_exponent")
        .and_then(|v| v.as_i64())
        .unwrap_or(-2) as i32;
    let scale = 10f64.powi(price_exp);

    match kind {
        EventKind::SpotQuote | EventKind::PerpQuote => {
            // Format: bid_price_mantissa, ask_price_mantissa
            let bid_mantissa = payload
                .get("bid_price_mantissa")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let ask_mantissa = payload
                .get("ask_price_mantissa")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let bid = bid_mantissa as f64 * scale;
            let ask = ask_mantissa as f64 * scale;
            (bid, ask)
        }
        EventKind::PerpDepth => {
            // Depth events come in two forms:
            // - is_snapshot=true: full book, bids sorted highest-first, asks sorted lowest-first
            // - is_snapshot=false: delta updates containing random book level changes
            //
            // ONLY use snapshots for price discovery. Deltas don't contain best bid/ask.
            // Return (0,0) for deltas so the strategy skips them.
            let is_snapshot = payload
                .get("is_snapshot")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !is_snapshot {
                return (0.0, 0.0); // Skip deltas
            }

            // Snapshot: first bid is best bid, first ask is best ask
            let bid_price = payload
                .get("bids")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("price"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let ask_price = payload
                .get("asks")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("price"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let bid = bid_price as f64 * scale;
            let ask = ask_price as f64 * scale;

            (bid, ask)
        }
        _ => (0.0, 0.0),
    }
}

/// Paper exchange for simulated execution.
pub struct PaperExchange {
    config: ExchangeConfig,
    markets: HashMap<String, MarketState>,
    positions: HashMap<String, Position>,
    cash: f64,
    realized_pnl: f64,
    fills: Vec<Fill>,
}

impl PaperExchange {
    pub fn new(config: ExchangeConfig) -> Self {
        Self {
            cash: config.initial_cash,
            config,
            markets: HashMap::new(),
            positions: HashMap::new(),
            realized_pnl: 0.0,
            fills: Vec::new(),
        }
    }

    /// Update market state from a replay event.
    pub fn update_market(&mut self, event: &ReplayEvent) {
        // Only update from quote events
        let dominated = match event.kind {
            EventKind::PerpQuote | EventKind::PerpDepth if self.config.use_perp_prices => false,
            EventKind::SpotQuote if !self.config.use_perp_prices => false,
            _ => true,
        };

        if dominated {
            return;
        }

        // Extract bid/ask from payload (handle both float and mantissa formats)
        let (bid, ask) = extract_bid_ask(&event.payload, event.kind);

        if bid > 0.0 && ask > 0.0 {
            let state = self.markets.entry(event.symbol.clone()).or_default();
            state.bid = bid;
            state.ask = ask;
            state.last_update = Some(event.ts);
        }
    }

    /// Execute an order intent. Returns fill if executed.
    pub fn execute(&mut self, intent: &OrderIntent, ts: DateTime<Utc>) -> Option<Fill> {
        let market = self.markets.get(&intent.symbol)?;

        // Determine execution price (cross the spread)
        let exec_price = match intent.side {
            Side::Buy => market.ask,  // Buy at ask
            Side::Sell => market.bid, // Sell at bid
        };

        if exec_price <= 0.0 {
            return None;
        }

        // Check limit price
        if let Some(limit) = intent.limit_price {
            match intent.side {
                Side::Buy if exec_price > limit => return None,
                Side::Sell if exec_price < limit => return None,
                _ => {}
            }
        }

        // Calculate notional and fee
        let notional = exec_price * intent.qty;
        let fee = notional * (self.config.fee_bps / 10_000.0);

        // Update position
        let position = self.positions.entry(intent.symbol.clone()).or_default();
        let old_qty = position.qty;

        match intent.side {
            Side::Buy => {
                // Check cash
                if self.cash < notional + fee {
                    tracing::warn!(
                        "Insufficient cash: need {:.2}, have {:.2}",
                        notional + fee,
                        self.cash
                    );
                    return None;
                }
                self.cash -= notional + fee;

                // Update position
                if position.qty >= 0.0 {
                    // Adding to long or opening long
                    let total_cost = position.avg_price * position.qty + exec_price * intent.qty;
                    position.qty += intent.qty;
                    position.avg_price = if position.qty > 0.0 {
                        total_cost / position.qty
                    } else {
                        0.0
                    };
                } else {
                    // Covering short
                    let cover_qty = intent.qty.min(-position.qty);
                    let pnl = (position.avg_price - exec_price) * cover_qty;
                    self.realized_pnl += pnl;
                    self.cash += pnl; // Return PnL to cash

                    position.qty += intent.qty;
                    if position.qty > 0.0 {
                        position.avg_price = exec_price;
                    }
                }
            }
            Side::Sell => {
                // Update position
                if position.qty <= 0.0 {
                    // Adding to short or opening short
                    let total_cost = position.avg_price * (-position.qty) + exec_price * intent.qty;
                    position.qty -= intent.qty;
                    position.avg_price = if position.qty < 0.0 {
                        total_cost / (-position.qty)
                    } else {
                        0.0
                    };
                } else {
                    // Closing long
                    let close_qty = intent.qty.min(position.qty);
                    let pnl = (exec_price - position.avg_price) * close_qty;
                    self.realized_pnl += pnl;

                    position.qty -= intent.qty;
                    if position.qty < 0.0 {
                        position.avg_price = exec_price;
                    }
                }

                self.cash += notional - fee;
            }
        }

        let fill = Fill {
            ts,
            symbol: intent.symbol.clone(),
            side: intent.side,
            qty: intent.qty,
            price: exec_price,
            fee,
            tag: intent.tag.clone(),
        };

        tracing::debug!(
            "FILL: {} {} {:.4} @ {:.2} (fee={:.4}, pos: {:.4} -> {:.4})",
            fill.side,
            fill.symbol,
            fill.qty,
            fill.price,
            fill.fee,
            old_qty,
            position.qty
        );

        self.fills.push(fill.clone());
        Some(fill)
    }

    /// Get current position for a symbol.
    pub fn position(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map(|p| p.qty).unwrap_or(0.0)
    }

    /// Get current cash balance.
    pub fn cash(&self) -> f64 {
        self.cash
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get unrealized PnL across all positions.
    pub fn unrealized_pnl(&self) -> f64 {
        let mut pnl = 0.0;
        for (symbol, position) in &self.positions {
            if let Some(market) = self.markets.get(symbol) {
                let mid = (market.bid + market.ask) / 2.0;
                if position.qty > 0.0 {
                    pnl += (mid - position.avg_price) * position.qty;
                } else if position.qty < 0.0 {
                    pnl += (position.avg_price - mid) * (-position.qty);
                }
            }
        }
        pnl
    }

    /// Get total fills.
    pub fn fills(&self) -> &[Fill] {
        &self.fills
    }
}

// =============================================================================
// Backtest Engine
// =============================================================================

/// Backtest configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub exchange: ExchangeConfig,
    /// Log progress every N events
    pub log_interval: usize,
    /// Pacing mode (fast or real-time)
    pub pace: PaceMode,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            exchange: ExchangeConfig::default(),
            log_interval: 100_000,
            pace: PaceMode::Fast,
        }
    }
}

/// Backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub strategy_name: String,
    pub segment_path: String,
    pub total_events: usize,
    pub total_fills: usize,
    pub initial_cash: f64,
    pub final_cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub return_pct: f64,
    pub start_ts: Option<DateTime<Utc>>,
    pub end_ts: Option<DateTime<Utc>>,
    pub duration_secs: f64,
    pub fills: Vec<Fill>,
    pub trades: Vec<RoundTrip>,
    pub metrics: TradeMetrics,
    pub equity_curve: Vec<EquityPoint>,
}

/// Backtest engine.
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on a segment with a strategy.
    ///
    /// This is an async method to support real-time pacing mode.
    pub async fn run<S: Strategy>(
        &self,
        segment_dir: &Path,
        mut strategy: S,
    ) -> Result<BacktestResult> {
        let mut adapter = SegmentReplayAdapter::open(segment_dir)?;
        let mut exchange = PaperExchange::new(self.config.exchange.clone());

        let mut total_events = 0usize;
        let mut start_ts: Option<DateTime<Utc>> = None;
        let mut end_ts: Option<DateTime<Utc>> = None;
        let mut last_event_ts: Option<DateTime<Utc>> = None;

        // Equity curve tracking
        let mut equity_curve: Vec<EquityPoint> = Vec::new();
        let mut peak_equity = self.config.exchange.initial_cash;
        let mut last_fill_count = 0usize;

        let real_time = self.config.pace == PaceMode::RealTime;
        let wall_clock_start = std::time::Instant::now();

        tracing::info!(
            "Starting backtest: strategy={}, segment={:?}, pace={:?}",
            strategy.name(),
            segment_dir,
            self.config.pace
        );

        while let Some(event) = adapter.next_event()? {
            total_events += 1;

            if start_ts.is_none() {
                start_ts = Some(event.ts);
            }

            // Real-time pacing: sleep to match original event timing
            if real_time && let Some(last_ts) = last_event_ts {
                let event_delta_ms = (event.ts - last_ts).num_milliseconds();
                if event_delta_ms > 0 && event_delta_ms < 60_000 {
                    // Cap at 1 minute to avoid long waits on gaps
                    tokio::time::sleep(std::time::Duration::from_millis(event_delta_ms as u64))
                        .await;
                }
            }
            last_event_ts = Some(event.ts);
            end_ts = Some(event.ts);

            // Update market state
            exchange.update_market(&event);

            // Get strategy orders
            let orders = strategy.on_event(&event);

            // Execute orders
            for order in orders {
                if let Some(fill) = exchange.execute(&order, event.ts) {
                    strategy.on_fill(&fill);
                }
            }

            // Track equity curve after fills
            let current_fill_count = exchange.fills().len();
            if current_fill_count > last_fill_count {
                let equity = exchange.cash() + exchange.realized_pnl() + exchange.unrealized_pnl();
                peak_equity = peak_equity.max(equity);
                let drawdown = peak_equity - equity;
                let drawdown_pct = if peak_equity > 0.0 {
                    (drawdown / peak_equity) * 100.0
                } else {
                    0.0
                };

                equity_curve.push(EquityPoint {
                    ts: event.ts,
                    equity,
                    drawdown,
                    drawdown_pct,
                });
                last_fill_count = current_fill_count;
            }

            // Progress logging
            if total_events.is_multiple_of(self.config.log_interval) {
                let elapsed = wall_clock_start.elapsed().as_secs_f64();
                tracing::info!(
                    "Progress: {} events, {} fills, PnL: {:.2}, elapsed: {:.1}s",
                    total_events,
                    exchange.fills().len(),
                    exchange.realized_pnl() + exchange.unrealized_pnl(),
                    elapsed
                );
            }
        }

        let realized = exchange.realized_pnl();
        let unrealized = exchange.unrealized_pnl();
        let total_pnl = realized + unrealized;
        let return_pct = (total_pnl / self.config.exchange.initial_cash) * 100.0;

        let duration_secs = match (start_ts, end_ts) {
            (Some(s), Some(e)) => (e - s).num_milliseconds() as f64 / 1000.0,
            _ => 0.0,
        };

        // Extract round-trip trades and compute metrics
        let fills = exchange.fills().to_vec();
        let trades = extract_round_trips(&fills);
        let metrics = TradeMetrics::compute(
            &trades,
            &equity_curve,
            self.config.exchange.initial_cash,
            duration_secs,
        );

        let result = BacktestResult {
            strategy_name: strategy.name().to_string(),
            segment_path: segment_dir.display().to_string(),
            total_events,
            total_fills: fills.len(),
            initial_cash: self.config.exchange.initial_cash,
            final_cash: exchange.cash(),
            realized_pnl: realized,
            unrealized_pnl: unrealized,
            total_pnl,
            return_pct,
            start_ts,
            end_ts,
            duration_secs,
            fills,
            trades,
            metrics,
            equity_curve,
        };

        tracing::info!("=== Backtest Complete ===");
        tracing::info!("Strategy: {}", result.strategy_name);
        tracing::info!("Events: {}", result.total_events);
        tracing::info!("Fills: {}", result.total_fills);
        tracing::info!("Duration: {:.1}s", result.duration_secs);
        tracing::info!("Realized PnL: ${:.2}", result.realized_pnl);
        tracing::info!("Unrealized PnL: ${:.2}", result.unrealized_pnl);
        tracing::info!(
            "Total PnL: ${:.2} ({:.2}%)",
            result.total_pnl,
            result.return_pct
        );

        Ok(result)
    }
}

// =============================================================================
// Example Strategies
// =============================================================================

/// Simple funding rate strategy.
///
/// Goes short perp when funding rate is positive (longs pay shorts).
/// Goes flat when funding is negative.
pub struct FundingBiasStrategy {
    target_position: f64,
    current_funding_rate: f64,
    threshold: f64,
}

impl FundingBiasStrategy {
    pub fn new(target_position: f64, threshold: f64) -> Self {
        Self {
            target_position,
            current_funding_rate: 0.0,
            threshold,
        }
    }
}

impl Strategy for FundingBiasStrategy {
    fn name(&self) -> &str {
        "funding_bias"
    }

    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        // Update funding rate
        if event.kind == EventKind::Funding
            && let Some(rate) = event.payload.get("rate").and_then(|v| v.as_f64())
        {
            self.current_funding_rate = rate;
        }

        // Only trade on perp quotes (to have price)
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth) {
            return vec![];
        }

        // Strategy logic: short when funding > threshold
        let mut orders = vec![];

        if self.current_funding_rate > self.threshold {
            // Funding positive, go short
            orders.push(
                OrderIntent::market(&event.symbol, Side::Sell, self.target_position)
                    .with_tag("funding_short"),
            );
        } else if self.current_funding_rate < -self.threshold {
            // Funding negative, go long
            orders.push(
                OrderIntent::market(&event.symbol, Side::Buy, self.target_position)
                    .with_tag("funding_long"),
            );
        }

        orders
    }
}

/// Spread capture strategy.
///
/// Trades when basis (perp - spot) exceeds threshold.
/// Uses actual position tracking from exchange fills.
pub struct BasisCaptureStrategy {
    // Price state
    spot_mid: f64,
    perp_mid: f64,

    // Parameters
    threshold_bps: f64,
    exit_threshold_bps: f64, // Exit when basis reverts to this level
    position_size: f64,

    // Position tracking (from fills, not assumptions)
    position_qty: f64, // Positive = long, negative = short

    // Cooldown to prevent churn
    last_trade_ts: Option<DateTime<Utc>>,
    min_hold_ms: i64,

    // Debug counters
    events_seen: usize,
    signals_seen: usize,
    entries_emitted: usize,
    exits_emitted: usize,
    fills_received: usize,
}

impl BasisCaptureStrategy {
    pub fn new(threshold_bps: f64, position_size: f64) -> Self {
        Self {
            spot_mid: 0.0,
            perp_mid: 0.0,
            threshold_bps,
            exit_threshold_bps: threshold_bps / 2.0, // Exit at 1/2 of entry threshold
            position_size,
            position_qty: 0.0,
            last_trade_ts: None,
            min_hold_ms: 5_000, // 5 second minimum hold (validation mode)
            events_seen: 0,
            signals_seen: 0,
            entries_emitted: 0,
            exits_emitted: 0,
            fills_received: 0,
        }
    }

    fn basis_bps(&self) -> f64 {
        if self.spot_mid > 0.0 && self.perp_mid > 0.0 {
            ((self.perp_mid - self.spot_mid) / self.spot_mid) * 10_000.0
        } else {
            0.0
        }
    }

    fn is_flat(&self) -> bool {
        self.position_qty.abs() < 1e-9
    }

    fn is_long(&self) -> bool {
        self.position_qty > 1e-9
    }

    fn is_short(&self) -> bool {
        self.position_qty < -1e-9
    }

    fn can_trade(&self, ts: DateTime<Utc>) -> bool {
        match self.last_trade_ts {
            Some(last) => (ts - last).num_milliseconds() >= self.min_hold_ms,
            None => true,
        }
    }
}

impl Strategy for BasisCaptureStrategy {
    fn name(&self) -> &str {
        "basis_capture"
    }

    fn on_event(&mut self, event: &ReplayEvent) -> Vec<OrderIntent> {
        self.events_seen += 1;

        // Update prices using mantissa-aware extraction
        match event.kind {
            EventKind::SpotQuote => {
                let (bid, ask) = extract_bid_ask(&event.payload, event.kind);
                if bid > 0.0 && ask > 0.0 {
                    self.spot_mid = (bid + ask) / 2.0;
                }
            }
            EventKind::PerpQuote | EventKind::PerpDepth => {
                let (bid, ask) = extract_bid_ask(&event.payload, event.kind);
                if bid > 0.0 && ask > 0.0 {
                    self.perp_mid = (bid + ask) / 2.0;
                }
            }
            _ => {}
        }

        // Only act on perp updates (when we have fresh perp price)
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth) {
            return vec![];
        }

        // Need both prices to calculate basis
        if self.spot_mid <= 0.0 || self.perp_mid <= 0.0 {
            return vec![];
        }

        let basis = self.basis_bps();
        let mut orders = vec![];

        // Progress logging every 1M events
        if self.events_seen.is_multiple_of(1_000_000) {
            eprintln!(
                "[STRATEGY] events={}M, signals={}, fills={}, pos={:.4}, basis={:.2}bps",
                self.events_seen / 1_000_000,
                self.signals_seen,
                self.fills_received,
                self.position_qty,
                basis
            );
        }

        // Check cooldown
        if !self.can_trade(event.ts) {
            return vec![];
        }

        // Entry signals (only when flat)
        if self.is_flat() {
            if basis > self.threshold_bps {
                // Basis too high: short perp (expect convergence down)
                self.signals_seen += 1;
                orders.push(
                    OrderIntent::market(&event.symbol, Side::Sell, self.position_size)
                        .with_tag("basis_short_entry"),
                );
                self.entries_emitted += 1;
                tracing::debug!(
                    "ENTRY SIGNAL: SHORT at basis={:.2}bps, threshold={:.2}bps",
                    basis,
                    self.threshold_bps
                );
            } else if basis < -self.threshold_bps {
                // Basis too low: long perp (expect convergence up)
                self.signals_seen += 1;
                orders.push(
                    OrderIntent::market(&event.symbol, Side::Buy, self.position_size)
                        .with_tag("basis_long_entry"),
                );
                self.entries_emitted += 1;
                tracing::debug!(
                    "ENTRY SIGNAL: LONG at basis={:.2}bps, threshold={:.2}bps",
                    basis,
                    self.threshold_bps
                );
            }
        }
        // Exit signals (when in position)
        else if self.is_short() && basis < self.exit_threshold_bps {
            // Was short, basis has converged down enough - cover
            self.signals_seen += 1;
            orders.push(
                OrderIntent::market(&event.symbol, Side::Buy, self.position_qty.abs())
                    .with_tag("basis_short_exit"),
            );
            self.exits_emitted += 1;
            tracing::debug!(
                "EXIT SIGNAL: COVER SHORT at basis={:.2}bps, exit_threshold={:.2}bps",
                basis,
                self.exit_threshold_bps
            );
        } else if self.is_long() && basis > -self.exit_threshold_bps {
            // Was long, basis has converged up enough - sell
            self.signals_seen += 1;
            orders.push(
                OrderIntent::market(&event.symbol, Side::Sell, self.position_qty.abs())
                    .with_tag("basis_long_exit"),
            );
            self.exits_emitted += 1;
            tracing::debug!(
                "EXIT SIGNAL: SELL LONG at basis={:.2}bps, exit_threshold={:.2}bps",
                basis,
                self.exit_threshold_bps
            );
        }

        orders
    }

    fn on_fill(&mut self, fill: &Fill) {
        self.fills_received += 1;
        self.last_trade_ts = Some(fill.ts);

        // Update position from actual fill
        match fill.side {
            Side::Buy => self.position_qty += fill.qty,
            Side::Sell => self.position_qty -= fill.qty,
        }

        tracing::debug!(
            "FILL RECEIVED: {} {:.6} @ {:.2}, new_pos={:.6}",
            fill.side,
            fill.qty,
            fill.price,
            self.position_qty
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bid_ask_float_format() {
        // Test that extract_bid_ask handles simple float format
        let payload = serde_json::json!({"bid": 100000.0, "ask": 100010.0});
        let (bid, ask) = extract_bid_ask(&payload, EventKind::PerpQuote);
        assert_eq!(bid, 100000.0);
        assert_eq!(ask, 100010.0);
    }

    #[test]
    fn test_paper_exchange_basic() {
        let mut exchange = PaperExchange::new(ExchangeConfig {
            fee_bps: 10.0,
            initial_cash: 100_000.0, // Enough for 0.1 BTC at $100k
            use_perp_prices: true,
        });

        // Simulate market update
        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({"bid": 100000.0, "ask": 100010.0}),
        };
        exchange.update_market(&event);

        // Verify market state was set
        assert!(
            exchange.markets.contains_key("BTCUSDT"),
            "Market state should be set"
        );
        let market = exchange.markets.get("BTCUSDT").unwrap();
        assert_eq!(market.bid, 100000.0, "Bid should be 100000.0");
        assert_eq!(market.ask, 100010.0, "Ask should be 100010.0");

        // Execute buy (0.1 BTC @ $100,010 = $10,001 + ~$1 fee)
        let order = OrderIntent::market("BTCUSDT", Side::Buy, 0.1);
        let fill = exchange.execute(&order, Utc::now());

        assert!(fill.is_some(), "Fill should not be None");
        let fill = fill.unwrap();
        assert_eq!(fill.price, 100010.0); // Bought at ask
        assert_eq!(fill.qty, 0.1);
        assert!(fill.fee > 0.0);

        assert_eq!(exchange.position("BTCUSDT"), 0.1);
    }

    #[test]
    fn test_pnl_calculation() {
        let mut exchange = PaperExchange::new(ExchangeConfig {
            fee_bps: 0.0, // No fees for simple test
            initial_cash: 10_000.0,
            use_perp_prices: true,
        });

        // Buy at 100
        let event1 = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({"bid": 100.0, "ask": 100.0}),
        };
        exchange.update_market(&event1);
        exchange.execute(&OrderIntent::market("BTCUSDT", Side::Buy, 1.0), Utc::now());

        // Price moves to 110
        let event2 = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({"bid": 110.0, "ask": 110.0}),
        };
        exchange.update_market(&event2);

        // Check unrealized PnL
        assert!((exchange.unrealized_pnl() - 10.0).abs() < 0.01);

        // Sell to realize
        exchange.execute(&OrderIntent::market("BTCUSDT", Side::Sell, 1.0), Utc::now());

        // Check realized PnL
        assert!((exchange.realized_pnl() - 10.0).abs() < 0.01);
        assert_eq!(exchange.position("BTCUSDT"), 0.0);
    }
}
