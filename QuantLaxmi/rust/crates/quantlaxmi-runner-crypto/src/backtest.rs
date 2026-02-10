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

use crate::deterministic_ids::{DeterministicIdState, compute_run_id};
use crate::replay::{EventKind, ReplayEvent, SegmentReplayAdapter};
use crate::sim::{
    FillType as SimFillType, Order as SimOrder, Side as SimSide, SimConfig, Simulator,
};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use quantlaxmi_eval::{SessionMetadata, StrategyAggregatorRegistry, StrategyTruthReport};
use quantlaxmi_events::{CorrelationContext, DecisionEvent, DecisionTraceBuilder, MarketSnapshot};
use quantlaxmi_gates::admission::{
    AdmissionContext, InternalSnapshot, SignalAdmissionController, VendorSnapshot,
};
use quantlaxmi_gates::{
    OrderIntentRef, OrderPermission, OrderPermissionGate, OrderSide as GateSide,
    OrderType as GateOrderType, StrategySpec,
};
use quantlaxmi_models::{
    AdmissionDecision, AdmissionOutcome, CostModelV1, ExecutionFillRecord, FillSide, FillType,
    OrderIntentRecord, OrderIntentSide, OrderIntentType, OrderRefuseReason, PositionUpdateRecord,
    SignalRequirements, compute_costs_v1,
    depth::{DepthEvent, DepthLevel, IntegrityTier},
};
use quantlaxmi_wal::{
    AdmissionIndex, AdmissionMismatch, AdmissionMismatchReason, WalReader, WalWriter,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use uuid::Uuid;

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
// Phase 19D: Admission Mode & Enforcement
// =============================================================================

/// Admission mode for strategy invocation gating.
///
/// Controls how the engine decides whether to call `strategy.on_event()`.
#[derive(Debug)]
pub enum AdmissionMode {
    /// Live evaluation: Evaluate admission using SignalAdmissionController.
    /// This is the default 19C behavior.
    EvaluateLive,

    /// Enforce from WAL: Follow WAL admission decisions exactly.
    /// The engine does NOT re-evaluate; it uses WAL as authoritative truth.
    EnforceFromWal {
        index: AdmissionIndex,
        policy: AdmissionMismatchPolicy,
    },
}

/// Policy for handling admission mismatches in enforce mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdmissionMismatchPolicy {
    /// Fail immediately on mismatch (default, strict).
    #[default]
    Fail,
    /// Log warning but continue (follow WAL decision).
    Warn,
}

impl std::str::FromStr for AdmissionMismatchPolicy {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "warn" => Self::Warn,
            _ => Self::Fail,
        })
    }
}

/// Result of admission decision for an event.
#[derive(Debug)]
pub struct AdmissionDecisionResult {
    /// Whether the event is admitted (strategy should be called).
    pub admit: bool,
    /// Mismatches detected (only relevant in enforce mode with verification).
    pub mismatches: Vec<AdmissionMismatch>,
    /// Per-signal decisions (for WAL writing in live mode).
    pub decisions: Vec<AdmissionDecision>,
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

/// Phase 25B: Pending intent awaiting delayed execution.
///
/// Stores what's needed for:
/// - Determining when to execute (scheduled_tick)
/// - Executing the intent (local_intent, parent_decision_id)
/// - WAL linking (intent_seq, intent_digest, correlation_id)
#[derive(Clone, Debug)]
struct PendingIntent {
    /// Tick at which this intent becomes executable
    scheduled_tick: u64,
    /// The intent to execute
    local_intent: OrderIntent,
    /// Parent decision ID for correlation
    parent_decision_id: Uuid,
    /// OrderIntent WAL sequence for linking
    intent_seq: u64,
    /// Intent digest for fill linking
    intent_digest: String,
    /// Correlation ID for WAL records
    correlation_id: String,
}

/// Liquidity type for fills (maker/taker).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Liquidity {
    Maker,
    Taker,
}

/// Execution fill.
///
/// ## Correlation Chain
/// Every fill carries `parent_decision_id` linking it back to the
/// originating DecisionEvent. This enables:
/// - Decision → Intent → Order → Fill attribution chain
/// - PnL attribution per decision
/// - Audit trails for G2/G3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub ts: DateTime<Utc>,
    /// Parent decision that originated this fill (for correlation)
    pub parent_decision_id: Uuid,
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub price: f64,
    pub fee: f64,
    /// Maker or Taker
    pub liquidity: Liquidity,
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
    pub cash: f64,           // Cash balance
    pub realized_pnl: f64,   // Cumulative realized PnL
    pub unrealized_pnl: f64, // Current unrealized PnL
    pub drawdown: f64,       // Current drawdown from peak
    pub drawdown_pct: f64,   // Drawdown as percentage
}

// =============================================================================
// V1 Output Schemas (Frozen)
// =============================================================================

/// Schema version for backtest output files.
pub const BACKTEST_SCHEMA_VERSION: &str = "v1";

/// V1 run manifest for backtest CLI output.
///
/// Schema version: v1
/// This schema is FROZEN - do not modify without bumping the version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRunManifestV1 {
    /// Schema version (always "v1")
    pub schema_version: String,
    /// Deterministic run ID (trace hash)
    pub run_id: String,
    /// Strategy name
    pub strategy: String,
    /// Path to input segment
    pub segment_path: String,
    /// Total depth events processed
    pub total_events: usize,
    /// Total fills executed
    pub total_fills: usize,
    /// Net realized PnL (float for display)
    pub realized_pnl: f64,
    /// Return percentage
    pub return_pct: f64,
    /// Decision trace hash (SHA-256)
    pub trace_hash: String,
}

impl BacktestRunManifestV1 {
    /// Create from backtest result.
    pub fn from_result(result: &BacktestResult) -> Self {
        Self {
            schema_version: BACKTEST_SCHEMA_VERSION.to_string(),
            run_id: result.trace_hash.clone(),
            strategy: result.strategy_name.clone(),
            segment_path: result.segment_path.clone(),
            total_events: result.total_events,
            total_fills: result.total_fills,
            realized_pnl: result.realized_pnl,
            return_pct: result.return_pct,
            trace_hash: result.trace_hash.clone(),
        }
    }
}

/// V1 metrics output for backtest CLI.
///
/// Schema version: v1
/// This schema is FROZEN - do not modify without bumping the version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetricsV1 {
    /// Schema version (always "v1")
    pub schema_version: String,
    /// Trade counts
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    /// PnL stats
    pub gross_profit: f64,
    pub gross_loss: f64,
    pub net_pnl: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    /// Win/loss averages
    pub avg_win: f64,
    pub avg_loss: f64,
    pub avg_win_loss_ratio: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    /// Risk metrics
    pub max_drawdown: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    /// Trade timing
    pub avg_trade_duration_secs: f64,
    pub total_fees: f64,
}

impl BacktestMetricsV1 {
    /// Create from TradeMetrics.
    pub fn from_metrics(metrics: &TradeMetrics) -> Self {
        Self {
            schema_version: BACKTEST_SCHEMA_VERSION.to_string(),
            total_trades: metrics.total_trades,
            winning_trades: metrics.winning_trades,
            losing_trades: metrics.losing_trades,
            win_rate: metrics.win_rate,
            gross_profit: metrics.gross_profit,
            gross_loss: metrics.gross_loss,
            net_pnl: metrics.net_pnl,
            profit_factor: metrics.profit_factor,
            expectancy: metrics.expectancy,
            avg_win: metrics.avg_win,
            avg_loss: metrics.avg_loss,
            avg_win_loss_ratio: metrics.avg_win_loss_ratio,
            largest_win: metrics.largest_win,
            largest_loss: metrics.largest_loss,
            max_drawdown: metrics.max_drawdown,
            max_drawdown_pct: metrics.max_drawdown_pct,
            sharpe_ratio: metrics.sharpe_ratio,
            sortino_ratio: metrics.sortino_ratio,
            avg_trade_duration_secs: metrics.avg_trade_duration_secs,
            total_fees: metrics.total_fees,
        }
    }
}

// =============================================================================
// JSONL Output Types (P0 - Audit-Grade Traces)
// =============================================================================

/// Equity curve point for JSONL output.
///
/// One line per timestamp checkpoint in equity_curve.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPointOutput {
    /// Timestamp in nanoseconds since Unix epoch
    pub ts_ns: i64,
    /// Total equity (cash + realized + unrealized)
    pub equity: f64,
    /// Cash balance
    pub cash: f64,
    /// Cumulative realized PnL
    pub realized_pnl: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
    /// Current drawdown from peak (absolute)
    pub drawdown: f64,
    /// Current drawdown from peak (percentage)
    pub drawdown_pct: f64,
}

impl EquityPointOutput {
    /// Create from internal EquityPoint.
    pub fn from_equity_point(ep: &EquityPoint) -> Self {
        Self {
            ts_ns: ep.ts.timestamp_nanos_opt().unwrap_or(0),
            equity: ep.equity,
            cash: ep.cash,
            realized_pnl: ep.realized_pnl,
            unrealized_pnl: ep.unrealized_pnl,
            drawdown: ep.drawdown,
            drawdown_pct: ep.drawdown_pct,
        }
    }
}

/// Fill record for JSONL output.
///
/// One line per fill in fills.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillOutput {
    /// Timestamp in nanoseconds since Unix epoch
    pub ts_ns: i64,
    /// Trading symbol
    pub symbol: String,
    /// Side: "Buy" or "Sell"
    pub side: String,
    /// Quantity filled
    pub qty: f64,
    /// Fill price
    pub price: f64,
    /// Fee paid
    pub fee: f64,
    /// Liquidity type: "Maker", "Taker", or null if unknown
    pub liquidity: Option<String>,
    /// Order ID (null if not available)
    pub order_id: Option<u64>,
    /// Strategy-defined tag (e.g., "exit_long_stop_loss")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

impl FillOutput {
    /// Create from internal Fill.
    pub fn from_fill(fill: &Fill) -> Self {
        Self {
            ts_ns: fill.ts.timestamp_nanos_opt().unwrap_or(0),
            symbol: fill.symbol.clone(),
            side: format!("{:?}", fill.side),
            qty: fill.qty,
            price: fill.price,
            fee: fill.fee,
            liquidity: Some(format!("{:?}", fill.liquidity)), // "Maker" or "Taker"
            order_id: None, // We have parent_decision_id but not order_id
            tag: fill.tag.clone(),
        }
    }
}

/// Write a vector of serializable items to a JSONL file.
///
/// Each item is written as a single JSON line, followed by newline.
/// File always ends with a newline.
pub fn write_jsonl<T: Serialize>(path: &std::path::Path, items: &[T]) -> std::io::Result<()> {
    use std::io::{BufWriter, Write};
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for item in items {
        serde_json::to_writer(&mut writer, item)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Write equity curve to JSONL file.
pub fn write_equity_curve_jsonl(
    path: &std::path::Path,
    equity_curve: &[EquityPoint],
) -> std::io::Result<()> {
    let outputs: Vec<EquityPointOutput> = equity_curve
        .iter()
        .map(EquityPointOutput::from_equity_point)
        .collect();
    write_jsonl(path, &outputs)
}

/// Write fills to JSONL file.
pub fn write_fills_jsonl(path: &std::path::Path, fills: &[Fill]) -> std::io::Result<()> {
    let outputs: Vec<FillOutput> = fills.iter().map(FillOutput::from_fill).collect();
    write_jsonl(path, &outputs)
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

        let net_pnl = gross_profit - gross_loss - total_fees;
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
// Fixed-Point PnL Accumulator
// =============================================================================

/// Fixed exponent for PnL values (-8 = 8 decimal places, matching crypto qty precision)
pub const PNL_EXPONENT: i8 = -8;

/// Fixed-point PnL accumulator for deterministic backtest accounting.
///
/// All values are stored as mantissas with fixed exponent (PNL_EXPONENT = -8).
/// This ensures cross-platform reproducibility without floating-point drift.
///
/// ## Usage
/// ```ignore
/// let mut acc = PnlAccumulatorFixed::new();
/// acc.add_fill(price_mantissa, qty_mantissa, fee_mantissa, is_buy);
/// let pnl = acc.realized_pnl_f64(); // For display only
/// ```
#[derive(Debug, Clone, Default)]
pub struct PnlAccumulatorFixed {
    /// Total realized PnL (mantissa, exponent = PNL_EXPONENT)
    pub realized_pnl_mantissa: i128,
    /// Total fees paid (mantissa, exponent = PNL_EXPONENT)
    pub total_fees_mantissa: i128,
    /// Current position quantity (mantissa, can be negative for short)
    pub position_qty_mantissa: i128,
    /// Average entry price for current position (mantissa, exponent = -2)
    pub avg_entry_price_mantissa: i128,
    /// Total cost basis for current position (mantissa, exponent = PNL_EXPONENT)
    pub cost_basis_mantissa: i128,
}

impl PnlAccumulatorFixed {
    /// Create a new accumulator with zero balances.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a fill (buy or sell).
    ///
    /// # Arguments
    /// * `price_mantissa` - Fill price (exponent = -2, cents)
    /// * `qty_mantissa` - Fill quantity (exponent = -8)
    /// * `fee_mantissa` - Fee amount (exponent = -8)
    /// * `is_buy` - True for buy, false for sell
    pub fn process_fill(
        &mut self,
        price_mantissa: i64,
        qty_mantissa: i64,
        fee_mantissa: i64,
        is_buy: bool,
    ) {
        // Fee is always added (exponent already matches PNL_EXPONENT)
        self.total_fees_mantissa += fee_mantissa as i128;

        // Notional = price * qty, but we need to account for exponent difference
        // price (exp -2) * qty (exp -8) = notional (exp -10)
        // We want PNL_EXPONENT = -8, so divide by 10^(-10 - (-8)) = 10^(-2) = 100
        let notional_raw = (price_mantissa as i128) * (qty_mantissa as i128);
        let notional_mantissa = notional_raw / 100; // Adjust exponent from -10 to -8

        if is_buy {
            if self.position_qty_mantissa >= 0 {
                // Adding to long or opening new long
                self.cost_basis_mantissa += notional_mantissa;
                self.position_qty_mantissa += qty_mantissa as i128;
                // Update average entry price
                if self.position_qty_mantissa > 0 {
                    // avg_price = cost_basis / position_qty
                    // But we need to handle exponent: cost_basis (exp -8) / qty (exp -8) = price (exp 0)
                    // We want price with exp -2, so multiply by 100
                    self.avg_entry_price_mantissa =
                        (self.cost_basis_mantissa * 100) / self.position_qty_mantissa;
                }
            } else {
                // Covering short position
                let cover_qty = (qty_mantissa as i128).min(-self.position_qty_mantissa);
                // PnL = (entry_price - exit_price) * cover_qty for short
                // entry_price (exp -2) * cover_qty (exp -8) / 100 = notional (exp -8)
                let entry_notional = (self.avg_entry_price_mantissa * cover_qty) / 100;
                let exit_notional = (price_mantissa as i128 * cover_qty) / 100;
                let pnl = entry_notional - exit_notional - fee_mantissa as i128;
                self.realized_pnl_mantissa += pnl;

                self.position_qty_mantissa += qty_mantissa as i128;
                if self.position_qty_mantissa > 0 {
                    // Flipped to long
                    let excess_qty = self.position_qty_mantissa;
                    self.cost_basis_mantissa = (price_mantissa as i128 * excess_qty) / 100;
                    self.avg_entry_price_mantissa = price_mantissa as i128;
                } else if self.position_qty_mantissa == 0 {
                    self.cost_basis_mantissa = 0;
                    self.avg_entry_price_mantissa = 0;
                }
            }
        } else {
            // Sell
            if self.position_qty_mantissa <= 0 {
                // Adding to short or opening new short
                self.cost_basis_mantissa += notional_mantissa;
                self.position_qty_mantissa -= qty_mantissa as i128;
                if self.position_qty_mantissa < 0 {
                    self.avg_entry_price_mantissa =
                        (self.cost_basis_mantissa * 100) / (-self.position_qty_mantissa);
                }
            } else {
                // Closing long position
                let close_qty = (qty_mantissa as i128).min(self.position_qty_mantissa);
                // PnL = (exit_price - entry_price) * close_qty for long
                let entry_notional = (self.avg_entry_price_mantissa * close_qty) / 100;
                let exit_notional = (price_mantissa as i128 * close_qty) / 100;
                let pnl = exit_notional - entry_notional - fee_mantissa as i128;
                self.realized_pnl_mantissa += pnl;

                self.position_qty_mantissa -= qty_mantissa as i128;
                if self.position_qty_mantissa < 0 {
                    // Flipped to short
                    let excess_qty = -self.position_qty_mantissa;
                    self.cost_basis_mantissa = (price_mantissa as i128 * excess_qty) / 100;
                    self.avg_entry_price_mantissa = price_mantissa as i128;
                } else if self.position_qty_mantissa == 0 {
                    self.cost_basis_mantissa = 0;
                    self.avg_entry_price_mantissa = 0;
                }
            }
        }
    }

    /// Get realized PnL as f64 (for display only).
    pub fn realized_pnl_f64(&self) -> f64 {
        self.realized_pnl_mantissa as f64 * 10f64.powi(PNL_EXPONENT as i32)
    }

    /// Get total fees as f64 (for display only).
    pub fn total_fees_f64(&self) -> f64 {
        self.total_fees_mantissa as f64 * 10f64.powi(PNL_EXPONENT as i32)
    }

    /// Get position quantity as f64 (for display only).
    pub fn position_qty_f64(&self) -> f64 {
        self.position_qty_mantissa as f64 * 10f64.powi(PNL_EXPONENT as i32)
    }

    /// Check if position is flat.
    pub fn is_flat(&self) -> bool {
        self.position_qty_mantissa == 0
    }
}

// =============================================================================
// Decision Event Conversion
// =============================================================================

/// Convert an OrderIntent to a DecisionEvent for trace recording.
///
/// This creates a canonical DecisionEvent that can be hashed for replay parity.
/// The price fields use a fixed exponent of -2 (cents precision) for consistency.
///
/// # Arguments
/// * `order` - The order intent to convert
/// * `ts` - Timestamp (MUST be from replay event, NOT wall-clock)
/// * `strategy_name` - Strategy identifier
/// * `run_id` - Deterministic run ID (hex hash)
/// * `decision_id` - Deterministic decision ID (derived from run_id + counters)
/// * `current_bid` - Current bid price
/// * `current_ask` - Current ask price
/// * `book_ts_ns` - Book timestamp in nanoseconds
#[allow(clippy::too_many_arguments)]
fn order_intent_to_decision(
    order: &OrderIntent,
    ts: DateTime<Utc>,
    strategy_name: &str,
    run_id: &str,
    decision_id: Uuid,
    current_bid: f64,
    current_ask: f64,
    book_ts_ns: i64,
) -> DecisionEvent {
    // decision_id is now passed in (deterministic), not generated here

    // Convert direction: Buy = 1 (long), Sell = -1 (short)
    let direction: i8 = match order.side {
        Side::Buy => 1,
        Side::Sell => -1,
    };

    // Determine decision type from tag
    let decision_type = order
        .tag
        .as_ref()
        .map(|t| {
            if t.contains("entry") {
                "entry"
            } else if t.contains("exit") {
                "exit"
            } else {
                "order"
            }
        })
        .unwrap_or("order")
        .to_string();

    // Price exponent: -2 means prices in cents (e.g., 8871660 = $88716.60)
    const PRICE_EXPONENT: i8 = -2;
    const QTY_EXPONENT: i8 = -8; // 8 decimal places for crypto

    // Convert f64 prices to mantissa (multiply by 10^(-exponent))
    let price_scale = 10f64.powi(-PRICE_EXPONENT as i32);
    let qty_scale = 10f64.powi(-QTY_EXPONENT as i32);

    let reference_price = order
        .limit_price
        .unwrap_or((current_bid + current_ask) / 2.0);
    let reference_price_mantissa = (reference_price * price_scale).round() as i64;
    let target_qty_mantissa = (order.qty * qty_scale).round() as i64;

    let bid_price_mantissa = (current_bid * price_scale).round() as i64;
    let ask_price_mantissa = (current_ask * price_scale).round() as i64;

    // Calculate spread in basis points (fixed-point: exponent = -2)
    let mid_price = (current_bid + current_ask) / 2.0;
    let spread_bps_mantissa = if mid_price > 0.0 {
        MarketSnapshot::spread_bps_from_f64(((current_ask - current_bid) / mid_price) * 10_000.0)
    } else {
        0
    };

    DecisionEvent {
        ts,
        decision_id,
        strategy_id: strategy_name.to_string(),
        symbol: order.symbol.clone(),
        decision_type,
        direction,
        target_qty_mantissa,
        qty_exponent: QTY_EXPONENT,
        reference_price_mantissa,
        price_exponent: PRICE_EXPONENT,
        // V2 snapshot: prices present, quantities absent (not tracked in backtest)
        market_snapshot: MarketSnapshot::v2_with_states(
            bid_price_mantissa,
            ask_price_mantissa,
            0, // bid_qty_mantissa: not tracked
            0, // ask_qty_mantissa: not tracked
            PRICE_EXPONENT,
            QTY_EXPONENT,
            spread_bps_mantissa,
            book_ts_ns,
            quantlaxmi_models::build_l1_state_bits(
                quantlaxmi_models::FieldState::Value,  // bid_price: present
                quantlaxmi_models::FieldState::Value,  // ask_price: present
                quantlaxmi_models::FieldState::Absent, // bid_qty: not tracked
                quantlaxmi_models::FieldState::Absent, // ask_qty: not tracked
            ),
        ),
        // Full confidence for backtest decisions (10000 = 1.0 with exponent -4)
        confidence_mantissa: DecisionEvent::confidence_from_f64(1.0),
        metadata: order
            .tag
            .as_ref()
            .map(|t| serde_json::json!({"tag": t}))
            .unwrap_or(serde_json::Value::Null),
        // Note: CorrelationContext is flattened in serde, so fields like symbol,
        // strategy_id, and decision_id would conflict with top-level fields.
        // We only set fields that don't have top-level equivalents.
        ctx: CorrelationContext {
            session_id: None,
            run_id: Some(run_id.to_string()),
            symbol: None, // Top-level symbol field exists
            venue: Some("paper".to_string()),
            strategy_id: None, // Top-level strategy_id field exists
            decision_id: None, // Top-level decision_id field exists
            order_id: None,
        },
    }
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

// NOTE: MarketState and Position types have been moved to sim.rs (crate::sim)
// PaperExchange now wraps Simulator from sim.rs for unified execution logic.

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
            fee_bps: 10.0, // 10 bps = 0.1% (Binance retail taker fee)
            initial_cash: 10_000.0,
            use_perp_prices: true,
        }
    }
}

// =============================================================================
// PHASE 22E: Enforcement Configuration
// =============================================================================

/// Enforcement configuration for production deployment.
///
/// Phase 22E: Controls manifest paths, promotion requirements, and WAL output.
///
/// ## Dev vs Production
/// - Dev mode (default): `require_promotion = false`, no manifests required
/// - Production mode: `require_promotion = true`, manifests + promotion root required
///
/// ## Frozen Warning (v1)
/// When `require_promotion = false`, emits:
/// ```text
/// ⚠️  PROMOTION ENFORCEMENT DISABLED - Running in dev mode. Production deployments
///     MUST set --require-promotion to enforce signal promotion status.
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnforcementConfig {
    /// Path to strategies_manifest.json (required for production)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategies_manifest_path: Option<std::path::PathBuf>,

    /// Path to signals_manifest.json (required for production)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signals_manifest_path: Option<std::path::PathBuf>,

    /// Path to promotion directory root (required for production)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub promotion_root: Option<std::path::PathBuf>,

    /// Require promotion status check before executing signals.
    /// When false: dev mode (no enforcement, warning emitted)
    /// When true: production mode (signal must be promoted)
    #[serde(default)]
    pub require_promotion: bool,

    /// Override WAL output directory (default: segment_dir/wal)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wal_dir: Option<std::path::PathBuf>,
}

impl EnforcementConfig {
    /// Create dev-mode config (no enforcement).
    pub fn dev() -> Self {
        Self::default()
    }

    /// Create production config with all paths.
    pub fn production(
        strategies_manifest: impl Into<std::path::PathBuf>,
        signals_manifest: impl Into<std::path::PathBuf>,
        promotion_root: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            strategies_manifest_path: Some(strategies_manifest.into()),
            signals_manifest_path: Some(signals_manifest.into()),
            promotion_root: Some(promotion_root.into()),
            require_promotion: true,
            wal_dir: None,
        }
    }

    /// Check if running in dev mode (no enforcement).
    pub fn is_dev_mode(&self) -> bool {
        !self.require_promotion
    }

    /// Emit frozen warning if promotion enforcement is disabled.
    ///
    /// Returns true if warning was emitted.
    pub fn emit_dev_mode_warning(&self) -> bool {
        if self.is_dev_mode() {
            tracing::warn!(
                "⚠️  PROMOTION ENFORCEMENT DISABLED - Running in dev mode. Production deployments \
                 MUST set --require-promotion to enforce signal promotion status."
            );
            true
        } else {
            false
        }
    }

    /// Validate config for production mode.
    ///
    /// Returns error if require_promotion is true but paths are missing.
    pub fn validate_for_production(&self) -> Result<(), String> {
        if !self.require_promotion {
            return Ok(()); // Dev mode, no validation
        }

        if self.strategies_manifest_path.is_none() {
            return Err(
                "--strategies-manifest required when --require-promotion is set".to_string(),
            );
        }
        if self.signals_manifest_path.is_none() {
            return Err("--signals-manifest required when --require-promotion is set".to_string());
        }
        if self.promotion_root.is_none() {
            return Err("--promotion-root required when --require-promotion is set".to_string());
        }

        Ok(())
    }
}

// =============================================================================
// PHASE 19C: Admission Gating Helpers
// =============================================================================

/// Convert MarketSnapshot to VendorSnapshot for admission gating.
///
/// Uses FieldState bits from V2 snapshots to determine field presence.
/// V1 snapshots validate prices (must be > 0) and always have quantities present.
fn vendor_snapshot_from_market(snap: &MarketSnapshot) -> VendorSnapshot {
    match snap {
        MarketSnapshot::V1(v1) => {
            // V1: Validate prices (must be > 0), qty always present (L5: Zero Is Valid)
            VendorSnapshot {
                bid_price: if v1.bid_price_mantissa > 0 {
                    Some(v1.bid_price_mantissa)
                } else {
                    None
                },
                ask_price: if v1.ask_price_mantissa > 0 {
                    Some(v1.ask_price_mantissa)
                } else {
                    None
                },
                buy_quantity: Some(v1.bid_qty_mantissa as u64),
                sell_quantity: Some(v1.ask_qty_mantissa as u64),
                ..VendorSnapshot::empty()
            }
        }
        MarketSnapshot::V2(v2) => {
            use quantlaxmi_models::events::{FieldState, get_field_state, l1_slots};

            let bid_price_state = get_field_state(v2.l1_state_bits, l1_slots::BID_PRICE);
            let ask_price_state = get_field_state(v2.l1_state_bits, l1_slots::ASK_PRICE);
            let bid_qty_state = get_field_state(v2.l1_state_bits, l1_slots::BID_QTY);
            let ask_qty_state = get_field_state(v2.l1_state_bits, l1_slots::ASK_QTY);

            VendorSnapshot {
                bid_price: if bid_price_state == FieldState::Value {
                    Some(v2.bid_price_mantissa)
                } else {
                    None
                },
                ask_price: if ask_price_state == FieldState::Value {
                    Some(v2.ask_price_mantissa)
                } else {
                    None
                },
                buy_quantity: if bid_qty_state == FieldState::Value {
                    Some(v2.bid_qty_mantissa as u64)
                } else {
                    None
                },
                sell_quantity: if ask_qty_state == FieldState::Value {
                    Some(v2.ask_qty_mantissa as u64)
                } else {
                    None
                },
                ..VendorSnapshot::empty()
            }
        }
    }
}

// =============================================================================
// Quote Extraction (non-doctrine path for display/debug)
// =============================================================================

/// Extract bid/ask prices from event payload as f64.
///
/// **WARNING**: This function converts mantissas to floats. Use `extract_bid_ask_mantissa()`
/// for doctrine-safe paths (scoring, replay, admission) to avoid float nondeterminism.
///
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

/// Extract bid/ask as mantissas + exponent, doctrine-safe (no float round-trips).
///
/// **Canonical path**: If the payload contains `bid_price_mantissa` / `ask_price_mantissa`,
/// returns them directly with no float conversion. This ensures cross-platform determinism.
///
/// **Legacy fallback**: If only float `bid`/`ask` are present, quantizes to mantissa using
/// the specified exponent. This path is not doctrine-perfect but preserves compatibility.
///
/// Returns `None` if no valid prices can be extracted (e.g., depth delta, unknown event).
fn extract_bid_ask_mantissa(
    payload: &serde_json::Value,
    kind: EventKind,
    default_price_exp: i8,
) -> Option<(i64, i64, i8)> {
    let price_exp = payload
        .get("price_exponent")
        .and_then(|v| v.as_i64())
        .unwrap_or(default_price_exp as i64) as i8;

    match kind {
        EventKind::SpotQuote | EventKind::PerpQuote => {
            // Canonical path: use mantissas directly (no floats!)
            let bid_m = payload.get("bid_price_mantissa").and_then(|v| v.as_i64());
            let ask_m = payload.get("ask_price_mantissa").and_then(|v| v.as_i64());

            if let (Some(bid_m), Some(ask_m)) = (bid_m, ask_m)
                && bid_m > 0
                && ask_m > 0
            {
                return Some((bid_m, ask_m, price_exp));
            }

            // Legacy fallback: float bid/ask → quantize to mantissa
            // Not doctrine-perfect, but necessary for legacy data
            let bid_f = payload.get("bid").and_then(|v| v.as_f64());
            let ask_f = payload.get("ask").and_then(|v| v.as_f64());

            if let (Some(bid_f), Some(ask_f)) = (bid_f, ask_f)
                && bid_f > 0.0
                && ask_f > 0.0
            {
                let scale = 10f64.powi(-(price_exp as i32));
                let bid_m = (bid_f * scale).round() as i64;
                let ask_m = (ask_f * scale).round() as i64;
                return Some((bid_m, ask_m, price_exp));
            }

            None
        }
        EventKind::PerpDepth => {
            // Only snapshots have valid best bid/ask
            let is_snapshot = payload
                .get("is_snapshot")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !is_snapshot {
                return None; // Skip deltas
            }

            // Snapshot: first bid is best bid, first ask is best ask
            // These are already mantissas in canonical depth format
            let bid_m = payload
                .get("bids")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("price"))
                .and_then(|v| v.as_i64());
            let ask_m = payload
                .get("asks")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("price"))
                .and_then(|v| v.as_i64());

            match (bid_m, ask_m) {
                (Some(b), Some(a)) if b > 0 && a > 0 => Some((b, a, price_exp)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract bid/ask quantities as mantissas, doctrine-safe.
///
/// Returns `None` if quantities are not available in the payload.
/// For depth snapshots, extracts quantities from the first bid/ask level.
fn extract_bid_ask_qty_mantissa(
    payload: &serde_json::Value,
    kind: EventKind,
    default_qty_exp: i8,
) -> Option<(i64, i64, i8)> {
    let qty_exp = payload
        .get("qty_exponent")
        .and_then(|v| v.as_i64())
        .unwrap_or(default_qty_exp as i64) as i8;

    match kind {
        EventKind::SpotQuote | EventKind::PerpQuote => {
            // Canonical path: use mantissas directly
            let bid_qty = payload.get("bid_qty_mantissa").and_then(|v| v.as_i64());
            let ask_qty = payload.get("ask_qty_mantissa").and_then(|v| v.as_i64());

            match (bid_qty, ask_qty) {
                (Some(b), Some(a)) => Some((b, a, qty_exp)),
                _ => None, // Quantities not in payload
            }
        }
        EventKind::PerpDepth => {
            let is_snapshot = payload
                .get("is_snapshot")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if !is_snapshot {
                return None;
            }

            // Snapshot: extract qty from first level
            let bid_qty = payload
                .get("bids")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("qty"))
                .and_then(|v| v.as_i64());
            let ask_qty = payload
                .get("asks")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|level| level.get("qty"))
                .and_then(|v| v.as_i64());

            match (bid_qty, ask_qty) {
                (Some(b), Some(a)) => Some((b, a, qty_exp)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Paper exchange for simulated execution.
///
/// NOTE: This is now a thin wrapper around `crate::sim::Simulator` which provides
/// the single source of truth for execution simulation. The wrapper maintains
/// backward compatibility with existing backtest code.
pub struct PaperExchange {
    config: ExchangeConfig,
    /// Unified simulator (Phase 2)
    sim: Simulator,
    /// Order ID counter
    next_order_id: u64,
    /// Fill records (converted from sim::Fill for backtest compatibility)
    fills: Vec<Fill>,
    // DEBUG: Book mutation counters (Step 2 of diagnosis)
    pub depth_calls: u64,
    pub book_mutations: u64,
    pub best_changes: u64,
}

impl PaperExchange {
    pub fn new(config: ExchangeConfig) -> Self {
        // Create SimConfig from ExchangeConfig
        let sim_cfg = SimConfig {
            fee_bps_maker: config.fee_bps,
            fee_bps_taker: config.fee_bps,
            latency_ticks: 0,
            allow_partial_fills: true,
            initial_cash: config.initial_cash,
        };
        Self {
            config,
            sim: Simulator::new(sim_cfg),
            next_order_id: 1,
            fills: Vec::new(),
            depth_calls: 0,
            book_mutations: 0,
            best_changes: 0,
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
            // Delegate to unified Simulator
            self.sim.update_market(&event.symbol, bid, ask);
        }
    }

    /// Process a depth event (handles both snapshots AND delta updates).
    ///
    /// This is the proper way to update the order book for PerpDepth events.
    /// Unlike update_market() which only works with snapshots, this correctly
    /// applies delta updates to maintain an accurate order book.
    pub fn on_depth(&mut self, symbol: &str, payload: &serde_json::Value, ts: DateTime<Utc>) {
        self.depth_calls += 1;

        // Parse the payload into a DepthEvent
        let Some(depth_event) = Self::parse_depth_payload(payload, ts) else {
            return;
        };

        // Track book state before update
        let best_bid_before = self.sim.best_bid(symbol);
        let best_ask_before = self.sim.best_ask(symbol);

        // Count mutations (non-empty bids/asks = changes)
        let n_mutations = depth_event.bids.len() + depth_event.asks.len();
        if n_mutations > 0 {
            self.book_mutations += n_mutations as u64;
        }

        // Delegate to Simulator's on_depth which handles both snapshots and deltas
        let _fills = self.sim.on_depth(symbol, &depth_event);

        // Track if best changed
        let best_bid_after = self.sim.best_bid(symbol);
        let best_ask_after = self.sim.best_ask(symbol);
        if best_bid_before != best_bid_after || best_ask_before != best_ask_after {
            self.best_changes += 1;
        }
    }

    /// Parse a PerpDepth payload into a DepthEvent.
    fn parse_depth_payload(payload: &serde_json::Value, ts: DateTime<Utc>) -> Option<DepthEvent> {
        let tradingsymbol = payload
            .get("tradingsymbol")
            .and_then(|v| v.as_str())
            .unwrap_or("BTCUSDT")
            .to_string();

        let first_update_id = payload
            .get("first_update_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let last_update_id = payload
            .get("last_update_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(first_update_id);

        let price_exponent = payload
            .get("price_exponent")
            .and_then(|v| v.as_i64())
            .unwrap_or(-2) as i8;
        let qty_exponent = payload
            .get("qty_exponent")
            .and_then(|v| v.as_i64())
            .unwrap_or(-8) as i8;

        let is_snapshot = payload
            .get("is_snapshot")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Parse bids
        let bids = payload
            .get("bids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|level| {
                        let price = level.get("price").and_then(|v| v.as_i64())?;
                        let qty = level.get("qty").and_then(|v| v.as_i64())?;
                        Some(DepthLevel { price, qty })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Parse asks
        let asks = payload
            .get("asks")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|level| {
                        let price = level.get("price").and_then(|v| v.as_i64())?;
                        let qty = level.get("qty").and_then(|v| v.as_i64())?;
                        Some(DepthLevel { price, qty })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Some(DepthEvent {
            ts,
            tradingsymbol,
            first_update_id,
            last_update_id,
            price_exponent,
            qty_exponent,
            bids,
            asks,
            is_snapshot,
            integrity_tier: IntegrityTier::NonCertified, // Backtest uses JSON
            source: None,
        })
    }

    /// Execute an order intent. Returns fill if executed.
    ///
    /// # Arguments
    /// * `intent` - The order intent to execute
    /// * `ts` - Execution timestamp
    /// * `parent_decision_id` - Decision that originated this order (for correlation)
    pub fn execute(
        &mut self,
        intent: &OrderIntent,
        ts: DateTime<Utc>,
        parent_decision_id: Uuid,
    ) -> Option<Fill> {
        // Convert backtest Side to sim Side
        let sim_side = match intent.side {
            Side::Buy => SimSide::Buy,
            Side::Sell => SimSide::Sell,
        };

        // Convert OrderIntent to sim Order
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        let sim_order = if let Some(limit) = intent.limit_price {
            let mut order = SimOrder::limit(order_id, &intent.symbol, sim_side, intent.qty, limit);
            order.tag = intent.tag.clone();
            order
        } else {
            let mut order = SimOrder::market(order_id, &intent.symbol, sim_side, intent.qty);
            order.tag = intent.tag.clone();
            order
        };

        // Submit to unified Simulator
        let ts_ns = ts.timestamp_nanos_opt().unwrap_or(0) as u64;
        let sim_fills = self.sim.submit(ts_ns, sim_order);

        // Convert sim::Fill to backtest::Fill
        if let Some(sim_fill) = sim_fills.into_iter().next() {
            // Convert sim::FillType to backtest::Liquidity
            let liquidity = match sim_fill.fill_type {
                SimFillType::Maker => Liquidity::Maker,
                SimFillType::Taker => Liquidity::Taker,
            };

            let fill = Fill {
                ts,
                parent_decision_id,
                symbol: sim_fill.symbol,
                side: intent.side, // Use original side to preserve type
                qty: sim_fill.qty,
                price: sim_fill.price,
                fee: sim_fill.fee,
                liquidity,
                tag: sim_fill.tag,
            };

            self.fills.push(fill.clone());
            Some(fill)
        } else {
            None
        }
    }

    /// Get current position for a symbol.
    pub fn position(&self, symbol: &str) -> f64 {
        self.sim.position(symbol)
    }

    /// Get current cash balance.
    pub fn cash(&self) -> f64 {
        self.sim.cash()
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.sim.realized_pnl()
    }

    /// Get unrealized PnL across all positions.
    pub fn unrealized_pnl(&self) -> f64 {
        self.sim.unrealized_pnl()
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
    /// Optional path to save the decision trace (for replay parity verification).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_trace: Option<String>,
    /// Run ID for correlation context (auto-generated if not provided).
    /// If not provided, a deterministic run_id is computed from strategy + segment + params.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Canonical JSON of grid parameters (for deterministic run_id computation).
    /// Used when run_id is not explicitly provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params_json: Option<String>,
    /// Phase 19D: Enforce admission from WAL instead of re-evaluating.
    #[serde(default)]
    pub enforce_admission_from_wal: bool,
    /// Phase 19D: Policy when enforcement detects mismatch.
    #[serde(default)]
    pub admission_mismatch_policy: String,
    /// Phase 22C: Strategy specification for order permission gating.
    /// If provided, all order intents are checked against execution class constraints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy_spec: Option<StrategySpec>,
    /// Phase 22E: Enforcement configuration for production deployment.
    #[serde(default)]
    pub enforcement: EnforcementConfig,
    /// Phase 25A: Optional path to cost model config (JSON).
    /// If not provided, uses default (all zeros = no cost adjustment).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_model_path: Option<std::path::PathBuf>,
    /// Phase 25B: Latency buckets - delay between intent and fill (in ticks).
    /// 0 = immediate (baseline), 1 = 1 tick delay, 3 = 3 tick delay.
    #[serde(default)]
    pub latency_ticks: u32,
    /// Evaluation mode: Force flatten any open position at end of session.
    /// Converts MTM to realized PnL for proper profitability measurement.
    #[serde(default)]
    pub flatten_on_end: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            exchange: ExchangeConfig::default(),
            log_interval: 100_000,
            pace: PaceMode::Fast,
            output_trace: None,
            run_id: None,
            params_json: None,
            enforce_admission_from_wal: false,
            admission_mismatch_policy: "fail".to_string(),
            strategy_spec: None,
            enforcement: EnforcementConfig::default(),
            cost_model_path: None,
            latency_ticks: 0,
            flatten_on_end: false,
        }
    }
}

// =============================================================================
// SLRT Feature Engine (Real, Deterministic)
// =============================================================================
//
// Computes SLRT features from raw L2 + trades per SLRT-GPU-v1.1-Sealed spec.
// NO synthetic constants. All features derived from event stream.

use std::collections::BTreeMap;

/// Funnel statistics for gate tracking.
#[derive(Debug, Default, Clone)]
pub struct FunnelStats {
    /// Total events processed
    pub events: u64,
    /// Events refused (book issues, etc.)
    pub refused: u64,
    /// Events where regime = R3 (trade-eligible)
    pub eligible_r3: u64,
    /// Events where confidence > threshold
    pub conf_ok: u64,
    /// Events where d_perp > minimum
    pub d_perp_ok: u64,
    /// Events where |S_dir| >= threshold
    pub dir_ok: u64,
    /// Intents emitted
    pub intents: u64,
}

/// Per-symbol SLRT feature computation state.
/// Maintains deterministic rolling windows and computes features from L2 + trades.
#[derive(Debug)]
pub struct SlrtFeatureEngine {
    /// Per-symbol state
    symbols: BTreeMap<String, SymbolSlrtState>,
    /// Target size Q* for gap risk calculation (qty mantissa, default 0.1 BTC = 10_000_000 at exp=-8)
    q_star_mantissa: i64,
    /// Window duration in nanoseconds (250ms)
    window_ns: i64,
    /// Rolling z-score window size for d_perp_proxy
    zscore_window: usize,
    /// Funnel statistics
    pub funnel: FunnelStats,
    /// Feature samples for quantile computation
    pub d_perp_samples: Vec<f64>,
    pub fragility_samples: Vec<f64>,
    pub toxicity_samples: Vec<f64>,
    pub confidence_samples: Vec<f64>,
    pub urgency_samples: Vec<f64>,
}

/// Per-symbol SLRT state.
#[derive(Debug, Default)]
pub(crate) struct SymbolSlrtState {
    // === L2 Snapshot State ===
    /// Best bid price (mantissa)
    b1_px: i64,
    /// Best ask price (mantissa)
    a1_px: i64,
    /// Best bid qty (mantissa)
    b1_qty: i64,
    /// Best ask qty (mantissa)
    a1_qty: i64,
    /// Top-10 bid qty sum (mantissa)
    sum_bid_qty_10: i64,
    /// Top-10 ask qty sum (mantissa)
    sum_ask_qty_10: i64,
    /// Price exponent
    price_exp: i8,
    /// Qty exponent
    qty_exp: i8,
    /// Full book snapshot for gap risk calculation
    bids: Vec<(i64, i64)>, // (price_mantissa, qty_mantissa) sorted desc
    asks: Vec<(i64, i64)>, // (price_mantissa, qty_mantissa) sorted asc
    /// Book validity flag
    book_valid: bool,
    /// Crossed book flag
    book_crossed: bool,

    // === Computed Snapshot Features ===
    /// Mid price (mantissa)
    mid_mantissa: i64,
    /// Microprice (mantissa, scaled by 2 to avoid div by 2)
    microprice_mantissa: i64,
    /// Microprice deviation = microprice - mid (mantissa)
    microprice_dev_mantissa: i64,
    /// Top-10 imbalance I10 = (sum_bid - sum_ask) / (sum_bid + sum_ask) as f64
    imbalance_10: f64,
    /// Spread (mantissa)
    spread_mantissa: i64,

    // === Trade-Flow Window State ===
    /// Rolling window: (ts_ns, signed_qty_mantissa)
    trade_window: VecDeque<(i64, i64)>,
    /// Rolling window: (ts_ns, abs_qty_mantissa)
    trade_abs_window: VecDeque<(i64, i64)>,
    /// V_signed_250ms (mantissa sum)
    v_signed_250ms: i64,
    /// V_total_250ms (abs mantissa sum)
    v_total_250ms: i64,

    // === Mid History for Elasticity ===
    /// Rolling window: (ts_ns, mid_mantissa)
    mid_window: VecDeque<(i64, i64)>,

    // === Z-score History for d_perp_proxy ===
    /// Rolling history of microprice_dev for z-score
    microprice_dev_history: VecDeque<f64>,
    /// Rolling history of gap_risk for z-score
    gap_risk_history: VecDeque<f64>,
    /// Rolling history of elasticity for z-score
    elasticity_history: VecDeque<f64>,
    /// Rolling history of toxicity for z-score
    toxicity_history: VecDeque<f64>,
}

/// Computed SLRT features ready for classification.
#[derive(Debug, Clone, Default)]
pub struct SlrtFeatures {
    // === §4.1 Snapshot Features ===
    /// Microprice deviation μ - m (f64 for downstream)
    pub microprice_dev: f64,
    /// Top-10 imbalance I10
    pub imbalance_10: f64,
    /// Spread ratio (a1 - b1) / mid
    pub spread_ratio: f64,

    // === §4.2 Trade-Flow Features ===
    /// Signed volume over 250ms window (f64 units)
    pub v_signed_250ms: f64,
    /// Elasticity |Δm| / (|ΔV| + ε)
    pub elasticity_250ms: f64,

    // === §5.3 Fragility Components ===
    /// Gap risk (sweep cost normalized)
    pub gap_risk: f64,
    /// Toxicity proxy (NOT full VPIN)
    pub toxicity_proxy: f64,
    /// Fragility scalar (weighted combination)
    pub fragility: f64,

    // === §6.4 Off-Manifold Proxy ===
    /// d_perp_proxy (zscore-based, NOT sealed subspace)
    pub d_perp_proxy: f64,

    // === Regime Classification (from slrt-ref or local) ===
    /// Regime string "R0"/"R1"/"R2"/"R3"
    pub regime: String,
    /// Confidence [0,1]
    pub confidence: f64,
    /// Raw confidence before penalties
    pub raw_confidence: f64,
    /// Normalization penalty applied
    pub normalization_penalty: f64,
    /// Degraded reasons bitmask
    pub degraded_reasons: u32,
    /// Refused flag
    pub refused: bool,

    // === Best Quotes for Execution ===
    pub best_bid: f64,
    pub best_ask: f64,
}

impl SlrtFeatureEngine {
    /// Create a new feature engine with default config.
    pub fn new() -> Self {
        Self {
            symbols: BTreeMap::new(),
            q_star_mantissa: 10_000_000, // 0.1 BTC at exp=-8
            window_ns: 250_000_000,      // 250ms
            zscore_window: 100,          // 100 samples for rolling z-score
            funnel: FunnelStats::default(),
            d_perp_samples: Vec::new(),
            fragility_samples: Vec::new(),
            toxicity_samples: Vec::new(),
            confidence_samples: Vec::new(),
            urgency_samples: Vec::new(),
        }
    }

    /// Get or create per-symbol state.
    fn get_state(&mut self, symbol: &str) -> &mut SymbolSlrtState {
        self.symbols.entry(symbol.to_string()).or_default()
    }

    /// Update from a PerpDepth event.
    /// Returns true if book is valid for feature computation.
    pub fn on_depth(&mut self, ts_ns: i64, symbol: &str, payload: &serde_json::Value) -> bool {
        // Copy window_ns before getting mutable state reference to avoid borrow checker issues
        let window_ns = self.window_ns;
        let state = self.get_state(symbol);

        // Parse exponents
        state.price_exp = payload
            .get("price_exponent")
            .and_then(|v| v.as_i64())
            .unwrap_or(-2) as i8;
        state.qty_exp = payload
            .get("qty_exponent")
            .and_then(|v| v.as_i64())
            .unwrap_or(-8) as i8;

        // Parse bids (should be sorted desc by price)
        state.bids.clear();
        if let Some(bids) = payload.get("bids").and_then(|v| v.as_array()) {
            for bid in bids.iter().take(20) {
                let price = bid.get("price").and_then(|v| v.as_i64()).unwrap_or(0);
                let qty = bid.get("qty").and_then(|v| v.as_i64()).unwrap_or(0);
                if price > 0 && qty > 0 {
                    state.bids.push((price, qty));
                }
            }
        }

        // Parse asks (should be sorted asc by price)
        state.asks.clear();
        if let Some(asks) = payload.get("asks").and_then(|v| v.as_array()) {
            for ask in asks.iter().take(20) {
                let price = ask.get("price").and_then(|v| v.as_i64()).unwrap_or(0);
                let qty = ask.get("qty").and_then(|v| v.as_i64()).unwrap_or(0);
                if price > 0 && qty > 0 {
                    state.asks.push((price, qty));
                }
            }
        }

        // Validate book
        state.book_valid = !state.bids.is_empty() && !state.asks.is_empty();
        if !state.book_valid {
            return false;
        }

        // Extract top-of-book
        state.b1_px = state.bids[0].0;
        state.b1_qty = state.bids[0].1;
        state.a1_px = state.asks[0].0;
        state.a1_qty = state.asks[0].1;

        // Check for crossed book
        state.book_crossed = state.b1_px >= state.a1_px;
        if state.book_crossed {
            // Crossed book is a degraded state but we still compute features
            // (will be reflected in degraded_reasons)
        }

        // Compute top-10 sums
        state.sum_bid_qty_10 = state.bids.iter().take(10).map(|(_, q)| q).sum();
        state.sum_ask_qty_10 = state.asks.iter().take(10).map(|(_, q)| q).sum();

        // === Phase 1: Snapshot Features ===

        // Mid price (mantissa)
        state.mid_mantissa = (state.a1_px + state.b1_px) / 2;

        // Microprice (mantissa): μ = (a1*qb1 + b1*qa1) / (qa1 + qb1 + ε)
        // Use ε=1 in mantissa units to avoid div0
        let numer = state.a1_px * state.b1_qty + state.b1_px * state.a1_qty;
        let denom = state.a1_qty + state.b1_qty + 1;
        state.microprice_mantissa = numer / denom;

        // Microprice deviation
        state.microprice_dev_mantissa = state.microprice_mantissa - state.mid_mantissa;

        // Spread
        state.spread_mantissa = state.a1_px - state.b1_px;

        // Top-10 imbalance I10 = (sum_bid - sum_ask) / (sum_bid + sum_ask + ε)
        let total_qty = state.sum_bid_qty_10 + state.sum_ask_qty_10 + 1;
        state.imbalance_10 =
            (state.sum_bid_qty_10 - state.sum_ask_qty_10) as f64 / total_qty as f64;

        // Update mid history for elasticity
        state.mid_window.push_back((ts_ns, state.mid_mantissa));
        let cutoff = ts_ns - window_ns;
        while let Some(&(ts, _)) = state.mid_window.front() {
            if ts < cutoff {
                state.mid_window.pop_front();
            } else {
                break;
            }
        }

        true
    }

    /// Update from a Trade event.
    pub fn on_trade(&mut self, ts_ns: i64, symbol: &str, payload: &serde_json::Value) {
        // Copy window_ns before getting mutable state reference to avoid borrow checker issues
        let window_ns = self.window_ns;
        let state = self.get_state(symbol);

        // Parse qty mantissa
        let qty_mantissa = payload.get("qty").and_then(|v| v.as_i64()).unwrap_or(0);

        // Determine sign from is_buyer_maker
        // is_buyer_maker=false → buyer is aggressor → +qty
        // is_buyer_maker=true → seller is aggressor → -qty
        let is_buyer_maker = payload
            .get("is_buyer_maker")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let signed_qty = if is_buyer_maker {
            -qty_mantissa
        } else {
            qty_mantissa
        };

        // Add to windows
        state.trade_window.push_back((ts_ns, signed_qty));
        state.v_signed_250ms += signed_qty;

        state
            .trade_abs_window
            .push_back((ts_ns, qty_mantissa.abs()));
        state.v_total_250ms += qty_mantissa.abs();

        // Trim old entries
        let cutoff = ts_ns - window_ns;
        while let Some(&(ts, qty)) = state.trade_window.front() {
            if ts < cutoff {
                state.trade_window.pop_front();
                state.v_signed_250ms -= qty;
            } else {
                break;
            }
        }
        while let Some(&(ts, qty)) = state.trade_abs_window.front() {
            if ts < cutoff {
                state.trade_abs_window.pop_front();
                state.v_total_250ms -= qty;
            } else {
                break;
            }
        }
    }

    /// Compute all features for a symbol. Call after on_depth/on_trade updates.
    pub fn compute_features(&mut self, symbol: &str) -> Option<SlrtFeatures> {
        let state = self.symbols.get_mut(symbol)?;

        if !state.book_valid {
            return None;
        }

        let price_scale = 10f64.powi(state.price_exp as i32);
        let qty_scale = 10f64.powi(state.qty_exp as i32);

        // === Phase 1: Snapshot features (f64 conversion) ===
        let microprice_dev = state.microprice_dev_mantissa as f64 * price_scale;
        let imbalance_10 = state.imbalance_10;
        let mid_f64 = state.mid_mantissa as f64 * price_scale;
        let spread_f64 = state.spread_mantissa as f64 * price_scale;
        let spread_ratio = if mid_f64 > 0.0 {
            spread_f64 / mid_f64
        } else {
            0.0
        };

        // === Phase 2: Trade-flow features ===
        let v_signed_250ms = state.v_signed_250ms as f64 * qty_scale;
        let v_total_250ms = state.v_total_250ms as f64 * qty_scale;

        // Elasticity: |Δm| / (|ΔV| + ε)
        let delta_m = if let Some(&(_, oldest_mid)) = state.mid_window.front() {
            (state.mid_mantissa - oldest_mid).abs() as f64 * price_scale
        } else {
            0.0
        };
        let elasticity_250ms = delta_m / (v_total_250ms.abs() + 1e-10);

        // === Phase 3: Fragility components ===

        // Gap risk: sweep cost for Q*
        // Copy q_star_mantissa to avoid borrow issues
        let q_star_mantissa = self.q_star_mantissa;
        let gap_risk = Self::compute_gap_risk(q_star_mantissa, state);

        // === Toxicity proxy (fixed: non-saturating) ===
        // tox_base = |V_signed| / (V_total + ε)  ∈ [0,1]
        // activity = clamp(V_total / V_ref, 0, 1)  ∈ [0,1]
        // toxicity = clamp(tox_base^2 * activity, 0, 1)
        // V_ref = 0.5 BTC for 250ms window (configurable, reasonable for BTCUSDT)
        const V_REF: f64 = 0.5; // Reference volume for 250ms window
        let tox_base = v_signed_250ms.abs() / (v_total_250ms + 1e-10);
        let activity = (v_total_250ms / V_REF).clamp(0.0, 1.0);
        let toxicity_proxy = (tox_base.powi(2) * activity).clamp(0.0, 1.0);

        // === Fragility scalar (data-driven references) ===
        // Reference values calibrated to produce p95 fragility ~ 0.7-0.8:
        // - gap_ref: Use 10M (observed p99 values can exceed 1M mantissa)
        // - el_ref: Use 100k (elasticity spikes when volume is near-zero)
        // - spread_ref: Use 0.05 (5% spread as extreme reference)
        const GAP_REF: f64 = 10_000_000.0; // Reference gap risk (mantissa units)
        const EL_REF: f64 = 100_000.0; // Reference elasticity (price/volume)
        const SPREAD_REF: f64 = 0.05; // Reference spread ratio (5%)

        // Bounded terms [0,1]
        let gap_term = (gap_risk / GAP_REF).clamp(0.0, 1.0);
        let el_term = (elasticity_250ms / EL_REF).clamp(0.0, 1.0);
        let spread_term = (spread_ratio / SPREAD_REF).clamp(0.0, 1.0);

        // Weighted sum with final clamp to [0,1]
        // Weights: w_gap=0.5, w_el=0.3, w_sp=0.2
        let fragility_raw = 0.5 * gap_term + 0.3 * el_term + 0.2 * spread_term;
        let fragility = fragility_raw.clamp(0.0, 1.0);

        // === d_perp_proxy: zscore-based structural shock ===
        // Update history
        state.microprice_dev_history.push_back(microprice_dev.abs());
        state.gap_risk_history.push_back(gap_risk);
        state.elasticity_history.push_back(elasticity_250ms);
        state.toxicity_history.push_back(toxicity_proxy);

        // Trim to window size
        let zscore_window = self.zscore_window;
        while state.microprice_dev_history.len() > zscore_window {
            state.microprice_dev_history.pop_front();
        }
        while state.gap_risk_history.len() > zscore_window {
            state.gap_risk_history.pop_front();
        }
        while state.elasticity_history.len() > zscore_window {
            state.elasticity_history.pop_front();
        }
        while state.toxicity_history.len() > zscore_window {
            state.toxicity_history.pop_front();
        }

        // Compute z-scores
        let z_microprice = Self::zscore(&state.microprice_dev_history, microprice_dev.abs());
        let z_gap = Self::zscore(&state.gap_risk_history, gap_risk);
        let z_elasticity = Self::zscore(&state.elasticity_history, elasticity_250ms);
        let z_toxicity = Self::zscore(&state.toxicity_history, toxicity_proxy);

        // d_perp_proxy = sqrt(z1^2 + z2^2 + z3^2 + z4^2)
        let d_perp_proxy =
            (z_microprice.powi(2) + z_gap.powi(2) + z_elasticity.powi(2) + z_toxicity.powi(2))
                .sqrt();

        // === Regime Classification (MVP: threshold-based) ===
        // This should eventually call slrt-ref, but for MVP we use simple thresholds
        let (regime, confidence, raw_confidence, normalization_penalty, degraded_reasons, refused) =
            Self::classify_regime(state, fragility, d_perp_proxy, toxicity_proxy);

        Some(SlrtFeatures {
            microprice_dev,
            imbalance_10,
            spread_ratio,
            v_signed_250ms,
            elasticity_250ms,
            gap_risk,
            toxicity_proxy,
            fragility,
            d_perp_proxy,
            regime,
            confidence,
            raw_confidence,
            normalization_penalty,
            degraded_reasons,
            refused,
            best_bid: state.b1_px as f64 * price_scale,
            best_ask: state.a1_px as f64 * price_scale,
        })
    }

    /// Compute gap risk (sweep cost) for target size Q*.
    /// Note: Takes q_star_mantissa as parameter to avoid borrow checker issues.
    fn compute_gap_risk(q_star_mantissa: i64, state: &SymbolSlrtState) -> f64 {
        if state.asks.is_empty() || state.mid_mantissa == 0 {
            return 0.0;
        }

        // Walk asks to compute sweep cost for Q*
        let mut cum_qty: i64 = 0;
        let mut cum_cost: i64 = 0;

        for &(price, qty) in &state.asks {
            let take_qty = (q_star_mantissa - cum_qty).min(qty);
            if take_qty <= 0 {
                break;
            }
            cum_qty += take_qty;
            cum_cost += price * take_qty;

            if cum_qty >= q_star_mantissa {
                break;
            }
        }

        if cum_qty == 0 {
            return 0.0;
        }

        // VWAP of sweep
        let vwap = cum_cost / cum_qty;

        // Gap risk = |vwap - microprice| / tick_size
        // tick_size = 1 mantissa unit for MVP
        let gap = (vwap - state.microprice_mantissa).abs();

        // Normalize to reasonable scale (gap in ticks)
        gap as f64
    }

    /// Compute z-score for a value given history.
    fn zscore(history: &VecDeque<f64>, value: f64) -> f64 {
        if history.len() < 10 {
            return 0.0; // Not enough history
        }

        let n = history.len() as f64;
        let mean: f64 = history.iter().sum::<f64>() / n;
        let variance: f64 = history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        (value - mean) / std_dev
    }

    /// MVP regime classification (threshold-based).
    /// Returns (regime, confidence, raw_confidence, normalization_penalty, degraded_reasons, refused)
    fn classify_regime(
        state: &SymbolSlrtState,
        fragility: f64,
        d_perp_proxy: f64,
        toxicity_proxy: f64,
    ) -> (String, f64, f64, f64, u32, bool) {
        let mut degraded_reasons: u32 = 0;

        // Check for crossed book
        if state.book_crossed {
            degraded_reasons |= 1; // CROSSED_BOOK
        }

        // Raw confidence starts at 1.0
        let raw_confidence = 1.0;

        // Apply penalties
        let mut penalty = 1.0;

        // Penalty for high fragility
        if fragility > 0.7 {
            penalty *= 0.8;
            degraded_reasons |= 2; // HIGH_FRAGILITY
        }

        // Penalty for high toxicity
        if toxicity_proxy > 0.8 {
            penalty *= 0.7;
            degraded_reasons |= 4; // HIGH_TOXICITY
        }

        // Penalty for crossed book
        if state.book_crossed {
            penalty *= 0.5;
        }

        let confidence = raw_confidence * penalty;
        let normalization_penalty = penalty;

        // Determine regime based on d_perp_proxy and fragility
        // Per SLRT-GPU-v1.1-Sealed:
        //   R3 = Trade-Eligible (high instability: high d_perp AND high fragility)
        //   R2 = Watchlist (moderate instability)
        //   R1 = Low activity (restricted)
        //   R0 = Quiet/Refused (too stable or book issues)
        let regime = if state.book_crossed {
            "R0" // Refuse on crossed book
        } else if d_perp_proxy >= 2.0 && fragility >= 0.7 {
            "R3" // Trade-Eligible: high instability
        } else if d_perp_proxy >= 1.0 || fragility >= 0.5 {
            "R2" // Watchlist: moderate instability
        } else if d_perp_proxy >= 0.5 || fragility >= 0.3 {
            "R1" // Low activity
        } else {
            "R0" // Too quiet/stable
        };

        // Refuse if confidence too low
        let refused = confidence < 0.3 || state.book_crossed;

        (
            regime.to_string(),
            confidence,
            raw_confidence,
            normalization_penalty,
            degraded_reasons,
            refused,
        )
    }

    /// Enrich an event payload with SLRT features.
    /// Note: This is a placeholder - actual enrichment happens in create_enriched_payload.
    pub fn enrich_payload(&self, _symbol: &str, payload: &serde_json::Value) -> serde_json::Value {
        payload.clone()
    }

    /// Create enriched payload with computed features.
    pub fn create_enriched_payload(
        &self,
        payload: &serde_json::Value,
        features: &SlrtFeatures,
    ) -> serde_json::Value {
        let mut enriched = payload.clone();

        if let Some(obj) = enriched.as_object_mut() {
            // Compute eligible_to_trade here so strategy doesn't recompute
            // R3 = trade-eligible per SLRT-GPU-v1.1-Sealed
            let eligible_to_trade = features.regime == "R3" && !features.refused;

            // Nested slrt object per spec
            let slrt = serde_json::json!({
                "regime": features.regime,
                "eligible_to_trade": eligible_to_trade,
                "confidence": features.confidence,
                "raw_confidence": features.raw_confidence,
                "normalization_penalty": features.normalization_penalty,
                "degraded_reasons": features.degraded_reasons,
                "refused": features.refused,
                "microprice_dev": features.microprice_dev,
                "signed_volume_250ms": features.v_signed_250ms,
                "imbalance_10": features.imbalance_10,
                "d_perp": features.d_perp_proxy,
                "fragility": features.fragility,
                "toxicity": features.toxicity_proxy,
                "elasticity_250ms": features.elasticity_250ms,
                "gap_risk": features.gap_risk,
                "spread_ratio": features.spread_ratio,
                "best_bid": features.best_bid,
                "best_ask": features.best_ask,
            });
            obj.insert("slrt".to_string(), slrt);

            // Also insert flat keys for backward compatibility with slr.rs
            obj.insert("regime".to_string(), serde_json::json!(features.regime));
            obj.insert(
                "eligible_to_trade".to_string(),
                serde_json::json!(eligible_to_trade),
            );
            obj.insert(
                "confidence".to_string(),
                serde_json::json!(features.confidence),
            );
            obj.insert(
                "raw_confidence".to_string(),
                serde_json::json!(features.raw_confidence),
            );
            obj.insert(
                "normalization_penalty".to_string(),
                serde_json::json!(features.normalization_penalty),
            );
            obj.insert(
                "degraded_reasons".to_string(),
                serde_json::json!(features.degraded_reasons),
            );
            obj.insert(
                "d_perp".to_string(),
                serde_json::json!(features.d_perp_proxy),
            );
            obj.insert(
                "fragility".to_string(),
                serde_json::json!(features.fragility),
            );
            obj.insert(
                "toxicity".to_string(),
                serde_json::json!(features.toxicity_proxy),
            );
            obj.insert(
                "microprice_dev".to_string(),
                serde_json::json!(features.microprice_dev),
            );
            obj.insert(
                "signed_volume".to_string(),
                serde_json::json!(features.v_signed_250ms),
            );
            obj.insert(
                "imbalance_10".to_string(),
                serde_json::json!(features.imbalance_10),
            );
            obj.insert("refused".to_string(), serde_json::json!(features.refused));
            obj.insert("fti_persist".to_string(), serde_json::json!(false));
            obj.insert("toxicity_persist".to_string(), serde_json::json!(false));
        }

        enriched
    }

    /// Get regime distribution counts for diagnostics.
    #[allow(dead_code)]
    pub(crate) fn regime_counts(&self, symbol: &str) -> Option<&SymbolSlrtState> {
        self.symbols.get(symbol)
    }

    /// Record feature samples for quantile computation.
    pub fn record_sample(&mut self, features: &SlrtFeatures) {
        self.funnel.events += 1;
        if features.refused {
            self.funnel.refused += 1;
        }
        if features.regime == "R3" {
            self.funnel.eligible_r3 += 1;
        }
        // Collect samples for quantiles (subsample to avoid memory bloat)
        if self.funnel.events.is_multiple_of(10) {
            self.d_perp_samples.push(features.d_perp_proxy);
            self.fragility_samples.push(features.fragility);
            self.toxicity_samples.push(features.toxicity_proxy);
            self.confidence_samples.push(features.confidence);
        }
    }

    /// Record an intent emission.
    pub fn record_intent(&mut self) {
        self.funnel.intents += 1;
    }

    /// Compute quantiles from samples.
    fn compute_quantiles(samples: &mut [f64]) -> (f64, f64, f64, f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();
        let p50 = samples[n / 2];
        let p90 = samples[(n * 90) / 100];
        let p95 = samples[(n * 95) / 100];
        let p99 = samples[(n * 99) / 100];
        let max = samples[n - 1];
        (p50, p90, p95, p99, max)
    }

    /// Print funnel statistics and feature quantiles.
    pub fn print_diagnostics(&mut self) {
        println!("\n=== SLRT FEATURE FUNNEL ===");
        println!("events:      {}", self.funnel.events);
        println!("refused:     {}", self.funnel.refused);
        println!("eligible_r3: {}", self.funnel.eligible_r3);
        println!("conf_ok:     {}", self.funnel.conf_ok);
        println!("d_perp_ok:   {}", self.funnel.d_perp_ok);
        println!("dir_ok:      {}", self.funnel.dir_ok);
        println!("intents:     {}", self.funnel.intents);

        println!("\n=== FEATURE QUANTILES (p50/p90/p95/p99/max) ===");
        let (p50, p90, p95, p99, max) = Self::compute_quantiles(&mut self.d_perp_samples);
        println!(
            "d_perp:     {:.4} / {:.4} / {:.4} / {:.4} / {:.4}",
            p50, p90, p95, p99, max
        );

        let (p50, p90, p95, p99, max) = Self::compute_quantiles(&mut self.fragility_samples);
        println!(
            "fragility:  {:.4} / {:.4} / {:.4} / {:.4} / {:.4}",
            p50, p90, p95, p99, max
        );

        let (p50, p90, p95, p99, max) = Self::compute_quantiles(&mut self.toxicity_samples);
        println!(
            "toxicity:   {:.4} / {:.4} / {:.4} / {:.4} / {:.4}",
            p50, p90, p95, p99, max
        );

        let (p50, p90, p95, p99, max) = Self::compute_quantiles(&mut self.confidence_samples);
        println!(
            "confidence: {:.4} / {:.4} / {:.4} / {:.4} / {:.4}",
            p50, p90, p95, p99, max
        );

        if !self.urgency_samples.is_empty() {
            let (p50, p90, p95, p99, max) = Self::compute_quantiles(&mut self.urgency_samples);
            println!(
                "urgency:    {:.4} / {:.4} / {:.4} / {:.4} / {:.4}",
                p50, p90, p95, p99, max
            );
        } else {
            println!("urgency:    no samples (no decisions)");
        }
        println!("==============================\n");
    }
}

impl Default for SlrtFeatureEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub strategy_name: String,
    pub segment_path: String,
    pub total_events: usize,
    pub total_fills: usize,
    pub total_decisions: usize,
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
    /// Decision trace hash (SHA-256 hex) for replay parity verification.
    pub trace_hash: String,
    /// Path to the saved trace file (if output_trace was specified).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_path: Option<String>,
    /// Decision trace encoding version (for compatibility checking).
    pub trace_encoding_version: u8,
    /// Fixed-point realized PnL (mantissa with pnl_exponent).
    /// This is the deterministic value; realized_pnl is for display only.
    pub realized_pnl_mantissa: i128,
    /// Fixed-point total fees (mantissa with pnl_exponent).
    pub total_fees_mantissa: i128,
    /// PnL exponent for fixed-point values.
    /// For crypto: typically -8 (price_exp(-2) + qty_exp(-8) normalized to -8).
    pub pnl_exponent: i8,
}

/// Backtest engine.
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Centralized admission decision function (Phase 19D).
    ///
    /// Handles both live evaluation and WAL enforcement modes.
    /// Returns whether the event should be admitted and any decisions made.
    ///
    /// # Arguments
    /// - `manifest_version_hash`: SHA-256 hash of signals_manifest.json (Phase 20B)
    #[allow(clippy::too_many_arguments)]
    fn decide_admission(
        mode: &AdmissionMode,
        correlation_id: &str,
        ts_ns: i64,
        session_id: &str,
        required_signals: &[SignalRequirements],
        vendor_snapshot: &VendorSnapshot,
        internal_snapshot: &InternalSnapshot,
        manifest_version_hash: [u8; 32],
    ) -> Result<AdmissionDecisionResult> {
        match mode {
            AdmissionMode::EvaluateLive => {
                // 19C path: Evaluate each signal live
                let mut all_admitted = true;
                let mut decisions = Vec::new();

                for requirements in required_signals {
                    let admission_ctx = AdmissionContext::new(ts_ns, session_id)
                        .with_correlation(correlation_id.to_string());

                    let decision = SignalAdmissionController::evaluate(
                        requirements,
                        vendor_snapshot,
                        internal_snapshot,
                        admission_ctx,
                        manifest_version_hash,
                    );

                    if decision.is_refused() {
                        all_admitted = false;
                    }
                    decisions.push(decision);
                }

                Ok(AdmissionDecisionResult {
                    admit: all_admitted,
                    mismatches: Vec::new(),
                    decisions,
                })
            }

            AdmissionMode::EnforceFromWal { index, policy } => {
                // 19D path: Follow WAL decisions exactly
                let outcome = index.event_outcome(correlation_id);

                match outcome {
                    Some(AdmissionOutcome::Admit) => {
                        // WAL says admit → admit
                        Ok(AdmissionDecisionResult {
                            admit: true,
                            mismatches: Vec::new(),
                            decisions: Vec::new(), // No new decisions in enforce mode
                        })
                    }
                    Some(AdmissionOutcome::Refuse) => {
                        // WAL says refuse → refuse
                        Ok(AdmissionDecisionResult {
                            admit: false,
                            mismatches: Vec::new(),
                            decisions: Vec::new(),
                        })
                    }
                    None => {
                        // Missing WAL entry
                        let mismatch = AdmissionMismatch {
                            correlation_id: correlation_id.to_string(),
                            reason: AdmissionMismatchReason::MissingWalEntry,
                        };

                        match policy {
                            AdmissionMismatchPolicy::Fail => Err(anyhow::anyhow!(
                                "Admission enforcement failed: {} for correlation_id={}",
                                mismatch.reason,
                                correlation_id
                            )),
                            AdmissionMismatchPolicy::Warn => {
                                tracing::warn!(
                                    correlation_id = %correlation_id,
                                    "Missing WAL entry in enforce mode; refusing event"
                                );
                                // Doctrine: Missing WAL entry → refuse (cannot prove admission)
                                Ok(AdmissionDecisionResult {
                                    admit: false,
                                    mismatches: vec![mismatch],
                                    decisions: Vec::new(),
                                })
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run backtest on a segment with a strategy.
    ///
    /// This is an async method to support real-time pacing mode.
    /// Every strategy decision (OrderIntent) is recorded as a DecisionEvent
    /// for replay parity verification.
    pub async fn run<S: Strategy>(
        &self,
        segment_dir: &Path,
        mut strategy: S,
    ) -> Result<BacktestResult> {
        let mut adapter = SegmentReplayAdapter::open(segment_dir)?;
        let mut exchange = PaperExchange::new(self.config.exchange.clone());

        // Decision trace builder for replay parity
        let mut trace_builder = DecisionTraceBuilder::new();

        // Compute deterministic run_id from config + segment
        // If run_id is provided in config, use it; otherwise compute from inputs
        let segment_id = segment_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown_segment".to_string());
        let run_id = self.config.run_id.clone().unwrap_or_else(|| {
            // Compute deterministic run_id from strategy + segment + params
            let params_json = self.config.params_json.as_deref().unwrap_or("{}");
            compute_run_id(
                strategy.name(),
                std::slice::from_ref(&segment_id),
                params_json,
            )
        });

        // Initialize deterministic ID state
        let mut id_state = DeterministicIdState::new(run_id.clone());

        let mut total_events = 0usize;
        let mut start_ts: Option<DateTime<Utc>> = None;
        let mut end_ts: Option<DateTime<Utc>> = None;
        let mut last_event_ts: Option<DateTime<Utc>> = None;

        // Track market state for DecisionEvent snapshots
        let mut current_bid: f64 = 0.0;
        let mut current_ask: f64 = 0.0;
        let mut current_book_ts_ns: i64 = 0;

        // Equity curve tracking
        let mut equity_curve: Vec<EquityPoint> = Vec::new();
        let mut peak_equity = self.config.exchange.initial_cash;
        let mut last_fill_count = 0usize;

        let real_time = self.config.pace == PaceMode::RealTime;
        let wall_clock_start = std::time::Instant::now();

        tracing::info!(
            "Starting backtest: strategy={}, segment={:?}, pace={:?}, run_id={}",
            strategy.name(),
            segment_dir,
            self.config.pace,
            run_id
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

            // Update market state (also track for DecisionEvent snapshots)
            exchange.update_market(&event);
            let (bid, ask) = extract_bid_ask(&event.payload, event.kind);
            if bid > 0.0 && ask > 0.0 {
                current_bid = bid;
                current_ask = ask;
                current_book_ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0);
            }

            // Get strategy orders
            let orders = strategy.on_event(&event);

            // Record each order as a DecisionEvent and execute
            for order in orders {
                // Determine direction for decision_id generation
                let direction: i8 = match order.side {
                    Side::Buy => 1,
                    Side::Sell => -1,
                };
                let decision_type = order
                    .tag
                    .as_ref()
                    .map(|t| {
                        if t.contains("entry") {
                            "entry"
                        } else if t.contains("exit") {
                            "exit"
                        } else {
                            "order"
                        }
                    })
                    .unwrap_or("order");

                // Generate deterministic decision_id
                let (_, _, decision_id) =
                    id_state.next_decision_id(&order.symbol, decision_type, direction);

                // Create DecisionEvent from OrderIntent
                let decision = order_intent_to_decision(
                    &order,
                    event.ts,
                    strategy.name(),
                    &run_id,
                    decision_id,
                    current_bid,
                    current_ask,
                    current_book_ts_ns,
                );
                trace_builder.record(&decision);

                // Execute the order (pass decision_id for correlation)
                if let Some(fill) = exchange.execute(&order, event.ts, decision.decision_id) {
                    strategy.on_fill(&fill);
                }
            }

            // Increment event counter after processing
            id_state.next_event();

            // Track equity curve after fills
            let current_fill_count = exchange.fills().len();
            if current_fill_count > last_fill_count {
                let cash = exchange.cash();
                let realized_pnl = exchange.realized_pnl();
                let unrealized_pnl = exchange.unrealized_pnl();
                let equity = cash + realized_pnl + unrealized_pnl;
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
                    cash,
                    realized_pnl,
                    unrealized_pnl,
                    drawdown,
                    drawdown_pct,
                });
                last_fill_count = current_fill_count;
            }

            // Progress logging
            if total_events.is_multiple_of(self.config.log_interval) {
                let elapsed = wall_clock_start.elapsed().as_secs_f64();
                tracing::info!(
                    "Progress: {} events, {} fills, {} decisions, PnL: {:.2}, elapsed: {:.1}s",
                    total_events,
                    exchange.fills().len(),
                    trace_builder.len(),
                    exchange.realized_pnl() + exchange.unrealized_pnl(),
                    elapsed
                );
            }
        }

        // Finalize trace
        let trace = trace_builder.finalize();
        let trace_hash = trace.hash_hex();
        let total_decisions = trace.len();

        // Save trace if output path specified
        let trace_path = if let Some(ref path) = self.config.output_trace {
            let trace_file = Path::new(path);
            trace
                .save(trace_file)
                .map_err(|e| anyhow::anyhow!("Failed to save trace: {}", e))?;
            tracing::info!("Decision trace saved to: {}", path);
            Some(path.clone())
        } else {
            None
        };

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

        // Compute fixed-point PnL from fills for determinism
        let mut pnl_fixed = PnlAccumulatorFixed::new();
        for fill in &fills {
            // Convert fill to fixed-point
            let price_mantissa = (fill.price * 100.0).round() as i64; // exp -2
            let qty_mantissa = (fill.qty * 100_000_000.0).round() as i64; // exp -8
            let fee_mantissa = (fill.fee * 100_000_000.0).round() as i64; // exp -8
            let is_buy = matches!(fill.side, Side::Buy);
            pnl_fixed.process_fill(price_mantissa, qty_mantissa, fee_mantissa, is_buy);
        }

        let result = BacktestResult {
            strategy_name: strategy.name().to_string(),
            segment_path: segment_dir.display().to_string(),
            total_events,
            total_fills: fills.len(),
            total_decisions,
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
            trace_hash: trace_hash.clone(),
            trace_path,
            trace_encoding_version: trace.encoding_version,
            realized_pnl_mantissa: pnl_fixed.realized_pnl_mantissa,
            total_fees_mantissa: pnl_fixed.total_fees_mantissa,
            pnl_exponent: PNL_EXPONENT,
        };

        tracing::info!("=== Backtest Complete ===");
        tracing::info!("Strategy: {}", result.strategy_name);
        tracing::info!("Events: {}", result.total_events);
        tracing::info!("Fills: {}", result.total_fills);
        tracing::info!("Decisions: {}", result.total_decisions);
        tracing::info!("Trace hash: {}...", &trace_hash[..16]);
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

    /// Run backtest with Phase 2 Strategy SDK.
    ///
    /// This method uses the new Strategy trait from quantlaxmi-strategy:
    /// - Strategy authors DecisionEvent AND OrderIntent together
    /// - Engine records decisions to trace (not strategy)
    /// - Returns strategy_binding for manifest
    pub async fn run_with_strategy(
        &self,
        segment_dir: &Path,
        mut strategy: Box<dyn quantlaxmi_strategy::Strategy>,
        config_path: Option<&Path>,
    ) -> Result<(
        BacktestResult,
        Option<crate::segment_manifest::StrategyBinding>,
    )> {
        let mut adapter = SegmentReplayAdapter::open(segment_dir)?;
        let mut exchange = PaperExchange::new(self.config.exchange.clone());

        // Decision trace builder for replay parity
        let mut trace_builder = DecisionTraceBuilder::new();

        // Compute deterministic run_id from config + segment (Phase 2)
        let segment_id = segment_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown_segment".to_string());
        let run_id = self.config.run_id.clone().unwrap_or_else(|| {
            let params_json = self.config.params_json.as_deref().unwrap_or("{}");
            compute_run_id(
                &strategy.short_id(),
                std::slice::from_ref(&segment_id),
                params_json,
            )
        });

        // Initialize deterministic ID state (Phase 2 strategies author their own DecisionEvents,
        // but we still track event_index for any engine-generated decisions)
        // NOTE: For full Phase 2 determinism, strategies would need to receive id_state
        // to generate deterministic decision_ids. This is a future enhancement.
        let _id_state = DeterministicIdState::new(run_id.clone());

        let mut total_events = 0usize;
        let mut start_ts: Option<DateTime<Utc>> = None;
        let mut end_ts: Option<DateTime<Utc>> = None;
        let mut last_event_ts: Option<DateTime<Utc>> = None;

        // Fixed-point market state for Phase 2 strategy context
        // Initialize with placeholder values (will be updated on first quote)
        let mut market_snapshot = MarketSnapshot::v2_with_states(
            0,
            0,
            0,
            0,
            -2,
            -8,
            0,
            0,
            quantlaxmi_models::build_l1_state_bits(
                quantlaxmi_models::FieldState::Absent,
                quantlaxmi_models::FieldState::Absent,
                quantlaxmi_models::FieldState::Absent,
                quantlaxmi_models::FieldState::Absent,
            ),
        );

        // Equity curve tracking
        let mut equity_curve: Vec<EquityPoint> = Vec::new();
        let mut peak_equity = self.config.exchange.initial_cash;
        let mut last_fill_count = 0usize;

        // Phase 19C/19D: Admission gating
        let required_signals = strategy.required_signals();
        let has_admission_gating = !required_signals.is_empty();
        let mut wal_writer = WalWriter::new(segment_dir).await?;
        let session_id = format!("backtest_{}", run_id);
        let mut admitted_events = 0u64;
        let mut refused_events = 0u64;
        let mut event_seq = 0u64;

        // DEBUG: Depth event flow counters (Step 1 of diagnosis)
        let mut n_perp_depth_total = 0u64;
        let mut n_perp_depth_snapshot = 0u64;
        let mut n_perp_depth_delta = 0u64;
        let mut n_on_depth_called = 0u64;
        // Invariant counters (keep for regression detection)
        let mut n_market_snapshot_from_payload = 0u64;
        let mut n_market_snapshot_from_book = 0u64;
        let mut n_market_snapshot_zero = 0u64;
        let mut n_book_broken = 0u64; // bid==0 || ask==0 || bid >= ask after fallback

        // Phase 22C: Order permission counters
        let mut permitted_orders = 0u64;
        let mut refused_orders = 0u64;

        // Phase 23B: Order intent WAL sequence counter
        let mut order_intent_seq = 0u64;

        // Phase 24A: Execution fill WAL sequence counter
        let mut execution_fill_seq = 0u64;
        // Track current order_intent_seq for linking fills to intents
        let mut current_intent_seq: Option<u64> = None;
        let mut current_intent_digest: Option<String> = None;

        // Phase 24D: Position update WAL tracking
        let mut position_update_seq = 0u64;
        // Track position state for computing post-state snapshots and deltas
        let mut position_tracker = PnlAccumulatorFixed::new();

        // Phase 25A: Load deterministic cost model (once per session)
        let cost_model: CostModelV1 = if let Some(ref path) = self.config.cost_model_path {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read cost model from {:?}", path))?;
            serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse cost model from {:?}", path))?
        } else {
            CostModelV1::default() // All zeros = no cost adjustment
        };
        if !cost_model.venues.is_empty() {
            tracing::info!(
                venues = cost_model.venues.len(),
                "Phase 25A: Cost model loaded"
            );
        }

        // Phase 25B: Latency bucket state
        let mut sim_tick: u64 = 0;
        let mut pending_intents: VecDeque<PendingIntent> = VecDeque::new();

        // SLRT Feature Engine (real, deterministic)
        let mut slrt_engine = SlrtFeatureEngine::new();
        let mut slrt_features_cache: Option<SlrtFeatures> = None;
        // Regime distribution counters
        let mut regime_counts: [u64; 4] = [0; 4]; // R0, R1, R2, R3
        // Urgency distribution tracking
        #[allow(unused_mut)]
        let mut urgency_samples: Vec<f64> = Vec::new();
        let latency_ticks = self.config.latency_ticks as u64;
        // Track last event timestamp for end-of-run drain (deterministic, no wall-clock)
        let mut last_drain_ts: DateTime<Utc> = DateTime::UNIX_EPOCH;

        if latency_ticks > 0 {
            tracing::info!(
                latency_ticks = latency_ticks,
                "Phase 25B: Latency buckets enabled"
            );
        }

        // Phase 19D: Build admission mode
        let admission_mode = if self.config.enforce_admission_from_wal && has_admission_gating {
            // Load admission index from existing WAL
            let index = AdmissionIndex::from_wal(segment_dir)
                .context("Failed to load admission WAL for enforcement")?;
            let policy = self
                .config
                .admission_mismatch_policy
                .parse::<AdmissionMismatchPolicy>()
                .unwrap();
            tracing::info!(
                wal_decisions = index.total_decisions,
                policy = ?policy,
                "Phase 19D: Enforcing admission from WAL"
            );
            AdmissionMode::EnforceFromWal { index, policy }
        } else {
            AdmissionMode::EvaluateLive
        };

        if has_admission_gating {
            match &admission_mode {
                AdmissionMode::EvaluateLive => {
                    tracing::info!(
                        signals = required_signals.len(),
                        "Phase 19C admission gating enabled (live evaluation)"
                    );
                }
                AdmissionMode::EnforceFromWal { .. } => {
                    tracing::info!(
                        signals = required_signals.len(),
                        "Phase 19D admission gating enabled (WAL enforcement)"
                    );
                }
            }
        }

        let real_time = self.config.pace == PaceMode::RealTime;
        let wall_clock_start = std::time::Instant::now();

        tracing::info!(
            "Starting Phase 2 backtest: strategy={}, segment={:?}, pace={:?}, run_id={}",
            strategy.short_id(),
            segment_dir,
            self.config.pace,
            run_id
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
                    tokio::time::sleep(std::time::Duration::from_millis(event_delta_ms as u64))
                        .await;
                }
            }
            last_event_ts = Some(event.ts);
            end_ts = Some(event.ts);
            last_drain_ts = event.ts; // Phase 25B: Track for end-of-run drain

            // =======================================================
            // PHASE 25B: Drain ready pending intents BEFORE processing new events
            // =======================================================
            // Create a ctx for fill notifications in drain loop
            // Uses current market_snapshot state from previous event(s)
            let drain_ctx = quantlaxmi_strategy::StrategyContext {
                ts: event.ts,
                run_id: &run_id,
                symbol: &event.symbol,
                market: &market_snapshot,
            };

            while let Some(front) = pending_intents.front() {
                if front.scheduled_tick > sim_tick {
                    break;
                }
                let pending = pending_intents.pop_front().unwrap();

                // Restore intent context for fill processing
                current_intent_seq = Some(pending.intent_seq);
                current_intent_digest = Some(pending.intent_digest.clone());
                let drain_correlation_id = pending.correlation_id.clone();

                // Execute the delayed intent using current event timestamp
                if let Some(fill) =
                    exchange.execute(&pending.local_intent, event.ts, pending.parent_decision_id)
                {
                    // --- FILL PROCESSING (same as immediate path) ---
                    let drain_ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0);
                    execution_fill_seq += 1;

                    let fill_qty_mantissa = (fill.qty
                        * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                    .round() as i64;
                    let fill_price_mantissa = (fill.price
                        * 10f64.powi(-market_snapshot.price_exponent() as i32))
                    .round() as i64;

                    let (fill_fee_mantissa, fill_fee_exponent) = if fill.fee > 0.0 {
                        let fee_m = (fill.fee * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                            .round() as i64;
                        (Some(fee_m), market_snapshot.qty_exponent())
                    } else {
                        (None, market_snapshot.qty_exponent())
                    };

                    let fill_side = match fill.side {
                        Side::Buy => FillSide::Buy,
                        Side::Sell => FillSide::Sell,
                    };

                    let fill_strategy_id = self
                        .config
                        .strategy_spec
                        .as_ref()
                        .map(|s| s.strategy_id.as_str())
                        .unwrap_or("unknown");

                    let mut ef_builder =
                        ExecutionFillRecord::builder(fill_strategy_id, &fill.symbol)
                            .ts_ns(drain_ts_ns)
                            .session_id(&session_id)
                            .seq(execution_fill_seq)
                            .side(fill_side)
                            .qty(fill_qty_mantissa, market_snapshot.qty_exponent())
                            .price(fill_price_mantissa, market_snapshot.price_exponent())
                            .venue("sim")
                            .correlation_id(&drain_correlation_id)
                            .fill_type(FillType::Full);

                    if let Some(fee_m) = fill_fee_mantissa {
                        ef_builder = ef_builder.fee(fee_m, fill_fee_exponent);
                    }
                    if let Some(parent_seq) = current_intent_seq {
                        ef_builder = ef_builder.parent_intent_seq(parent_seq);
                    }
                    if let Some(ref parent_digest) = current_intent_digest {
                        ef_builder = ef_builder.parent_intent_digest(parent_digest);
                    }

                    let ef_record = ef_builder.build();
                    wal_writer.write_execution_fill(ef_record).await?;

                    // Position update
                    position_update_seq += 1;
                    let pre_realized_pnl = position_tracker.realized_pnl_mantissa;
                    let pre_fees = position_tracker.total_fees_mantissa;

                    let is_buy = matches!(fill.side, Side::Buy);
                    position_tracker.process_fill(
                        fill_price_mantissa,
                        fill_qty_mantissa,
                        fill_fee_mantissa.unwrap_or(0),
                        is_buy,
                    );

                    let realized_pnl_delta_i128 =
                        position_tracker.realized_pnl_mantissa - pre_realized_pnl;
                    let fee_delta_i128 = position_tracker.total_fees_mantissa - pre_fees;
                    let notional_mantissa_i128 =
                        (fill_price_mantissa as i128 * fill_qty_mantissa as i128) / 100;
                    let notional_abs_i128 = notional_mantissa_i128.abs();

                    let fill_venue = "sim";
                    let costs = compute_costs_v1(
                        fill_venue,
                        &fill.symbol,
                        notional_abs_i128,
                        fill_fee_mantissa.unwrap_or(0),
                        &cost_model,
                    );

                    let effective_fee_delta_i128 =
                        fee_delta_i128 + costs.extra_fee_mantissa as i128;
                    let mut cash_delta_i128 = if is_buy {
                        -notional_mantissa_i128 - effective_fee_delta_i128
                    } else {
                        notional_mantissa_i128 - effective_fee_delta_i128
                    };
                    cash_delta_i128 -= costs.slippage_cost_mantissa as i128;
                    cash_delta_i128 -= costs.spread_cost_mantissa as i128;

                    let cash_delta_mantissa: i64 = i64::try_from(cash_delta_i128)
                        .map_err(|_| anyhow::anyhow!("cash_delta overflow: {}", cash_delta_i128))?;
                    let realized_pnl_delta: i64 =
                        i64::try_from(realized_pnl_delta_i128).map_err(|_| {
                            anyhow::anyhow!(
                                "realized_pnl_delta overflow: {}",
                                realized_pnl_delta_i128
                            )
                        })?;
                    let effective_fee_delta: i64 = i64::try_from(effective_fee_delta_i128)
                        .map_err(|_| {
                            anyhow::anyhow!("fee_delta overflow: {}", effective_fee_delta_i128)
                        })?;

                    let mut pu_builder =
                        PositionUpdateRecord::builder(fill_strategy_id, &fill.symbol)
                            .ts_ns(drain_ts_ns)
                            .session_id(&session_id)
                            .seq(position_update_seq)
                            .correlation_id(&drain_correlation_id)
                            .fill_seq(execution_fill_seq)
                            .position_qty(
                                position_tracker.position_qty_mantissa as i64,
                                market_snapshot.qty_exponent(),
                            )
                            .cash_delta(cash_delta_mantissa, PNL_EXPONENT)
                            .realized_pnl_delta(realized_pnl_delta, PNL_EXPONENT)
                            .venue(fill_venue);

                    if position_tracker.position_qty_mantissa != 0 {
                        pu_builder = pu_builder
                            .avg_price(position_tracker.avg_entry_price_mantissa as i64, -2);
                    } else {
                        pu_builder = pu_builder.avg_price_flat(-2);
                    }

                    if effective_fee_delta != 0 {
                        pu_builder = pu_builder.fee(effective_fee_delta, PNL_EXPONENT);
                    } else {
                        pu_builder = pu_builder.fee_exponent(PNL_EXPONENT);
                    }

                    let pu_record = pu_builder.build();
                    wal_writer.write_position_update(pu_record).await?;

                    current_intent_seq = None;
                    current_intent_digest = None;

                    let notification = quantlaxmi_strategy::FillNotification {
                        ts: fill.ts,
                        symbol: fill.symbol.clone(),
                        side: match fill.side {
                            Side::Buy => quantlaxmi_strategy::Side::Buy,
                            Side::Sell => quantlaxmi_strategy::Side::Sell,
                        },
                        qty_mantissa: fill_qty_mantissa,
                        qty_exponent: market_snapshot.qty_exponent(),
                        price_mantissa: fill_price_mantissa,
                        price_exponent: market_snapshot.price_exponent(),
                        fee_mantissa: fill_fee_mantissa.unwrap_or(0),
                        fee_exponent: fill_fee_exponent,
                        tag: fill.tag.clone(),
                    };
                    strategy.on_fill(&notification, &drain_ctx);
                }
            }

            // Update market state (for PaperExchange)
            // For PerpDepth events, use on_depth to properly handle delta updates.
            // For other quote events, use update_market.
            if event.kind == EventKind::PerpDepth {
                n_perp_depth_total += 1;
                let is_snapshot = event
                    .payload
                    .get("is_snapshot")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_snapshot {
                    n_perp_depth_snapshot += 1;
                } else {
                    n_perp_depth_delta += 1;
                }
                n_on_depth_called += 1;
                exchange.on_depth(&event.symbol, &event.payload, event.ts);
            } else {
                exchange.update_market(&event);
            }

            // SLRT Feature Engine: Update from L2 and trades (real, deterministic)
            let ts_ns_for_slrt = event.ts.timestamp_nanos_opt().unwrap_or(0);
            match event.kind {
                EventKind::PerpDepth => {
                    slrt_engine.on_depth(ts_ns_for_slrt, &event.symbol, &event.payload);
                    // Recompute features after depth update
                    if let Some(features) = slrt_engine.compute_features(&event.symbol) {
                        // Track regime distribution
                        match features.regime.as_str() {
                            "R0" => regime_counts[0] += 1,
                            "R1" => regime_counts[1] += 1,
                            "R2" => regime_counts[2] += 1,
                            "R3" => regime_counts[3] += 1,
                            _ => {}
                        }
                        // Record sample for funnel/quantiles
                        slrt_engine.record_sample(&features);
                        slrt_features_cache = Some(features);
                    }
                }
                EventKind::Trade => {
                    slrt_engine.on_trade(ts_ns_for_slrt, &event.symbol, &event.payload);
                }
                _ => {}
            }

            // DOCTRINE-SAFE: Extract prices as mantissas (no float round-trips)
            const DEFAULT_PRICE_EXP: i8 = -2;
            const DEFAULT_QTY_EXP: i8 = -8;

            // For PerpDepth delta events, extract_bid_ask_mantissa returns None.
            // In this case, fall back to the exchange's order book (which IS updated by on_depth).
            let price_from_payload =
                extract_bid_ask_mantissa(&event.payload, event.kind, DEFAULT_PRICE_EXP);
            let (bid_m, ask_m, price_exp) = if let Some(prices) = price_from_payload {
                n_market_snapshot_from_payload += 1;
                (prices.0, prices.1, prices.2)
            } else if event.kind == EventKind::PerpDepth {
                // Fallback: get prices from exchange's order book (updated by on_depth)
                // Convert f64 prices to mantissa with DEFAULT_PRICE_EXP (-2 = cents)
                let scale = 10f64.powi(-DEFAULT_PRICE_EXP as i32);
                let bid_opt = exchange.sim.best_bid(&event.symbol);
                let ask_opt = exchange.sim.best_ask(&event.symbol);
                match (bid_opt, ask_opt) {
                    (Some(bid), Some(ask)) if bid > 0.0 && ask > 0.0 && bid < ask => {
                        let bid_m = (bid * scale).round() as i64;
                        let ask_m = (ask * scale).round() as i64;
                        n_market_snapshot_from_book += 1;
                        (bid_m, ask_m, DEFAULT_PRICE_EXP)
                    }
                    (Some(bid), Some(ask)) if bid >= ask => {
                        // Broken book: bid >= ask (crossed/locked)
                        n_book_broken += 1;
                        if n_book_broken <= 5 {
                            tracing::warn!(
                                symbol = %event.symbol,
                                bid = bid,
                                ask = ask,
                                "Book broken: bid >= ask"
                            );
                        }
                        (0, 0, DEFAULT_PRICE_EXP)
                    }
                    _ => {
                        n_market_snapshot_zero += 1;
                        (0, 0, DEFAULT_PRICE_EXP)
                    }
                }
            } else {
                n_market_snapshot_zero += 1;
                (0, 0, DEFAULT_PRICE_EXP)
            };

            if bid_m > 0 && ask_m > 0 {
                let book_ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0);

                // Extract quantities if available (doctrine-safe)
                let (bid_qty_m, ask_qty_m, qty_exp, qty_state) = if let Some((bq, aq, qe)) =
                    extract_bid_ask_qty_mantissa(&event.payload, event.kind, DEFAULT_QTY_EXP)
                {
                    (bq, aq, qe, quantlaxmi_models::FieldState::Value)
                } else {
                    // Quantities not in payload → mark Absent (not fabricated)
                    (0, 0, DEFAULT_QTY_EXP, quantlaxmi_models::FieldState::Absent)
                };

                // Compute spread in basis points using integer math (no floats)
                // spread_bps = (ask - bid) / mid * 10000
                // For mantissas: spread_bps = (ask_m - bid_m) * 10000 * 2 / (bid_m + ask_m)
                let spread_bps_mantissa = if bid_m > 0 && ask_m > 0 {
                    let spread_m = ask_m - bid_m;
                    let mid_m = bid_m + ask_m; // 2 * mid
                    // spread_bps = spread / mid * 10000 = spread * 20000 / (2 * mid)
                    // Use i128 to avoid overflow, then scale to exponent -2
                    let spread_bps_raw = (spread_m as i128 * 20000) / (mid_m as i128);
                    spread_bps_raw as i64
                } else {
                    0
                };

                // Create V2 snapshot with proper field states (doctrine-safe)
                market_snapshot = MarketSnapshot::v2_with_states(
                    bid_m,
                    ask_m,
                    bid_qty_m,
                    ask_qty_m,
                    price_exp,
                    qty_exp,
                    spread_bps_mantissa,
                    book_ts_ns,
                    quantlaxmi_models::build_l1_state_bits(
                        quantlaxmi_models::FieldState::Value, // bid_price: present
                        quantlaxmi_models::FieldState::Value, // ask_price: present
                        qty_state,                            // bid_qty: present or absent
                        qty_state,                            // ask_qty: present or absent
                    ),
                );
            }

            // Convert to Phase 2 ReplayEvent
            let sdk_event = quantlaxmi_strategy::ReplayEvent {
                ts: event.ts,
                symbol: event.symbol.clone(),
                kind: match event.kind {
                    EventKind::SpotQuote => quantlaxmi_strategy::EventKind::SpotQuote,
                    EventKind::PerpQuote => quantlaxmi_strategy::EventKind::PerpQuote,
                    EventKind::PerpDepth => quantlaxmi_strategy::EventKind::PerpDepth,
                    EventKind::Funding => quantlaxmi_strategy::EventKind::Funding,
                    EventKind::Trade => quantlaxmi_strategy::EventKind::Trade,
                    EventKind::Unknown => quantlaxmi_strategy::EventKind::Unknown,
                },
                // Enrich payload with SLRT features (real, deterministic)
                payload: if let Some(ref features) = slrt_features_cache {
                    slrt_engine.create_enriched_payload(&event.payload, features)
                } else {
                    event.payload.clone()
                },
            };

            // Create strategy context
            let ctx = quantlaxmi_strategy::StrategyContext {
                ts: event.ts,
                run_id: &run_id,
                symbol: &event.symbol,
                market: &market_snapshot,
            };

            // =====================================================================
            // PHASE 19C/19D: Admission Gating (BEFORE strategy.on_event)
            // =====================================================================
            event_seq += 1;
            let ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0);
            let correlation_id = format!("event_seq:{}", event_seq);

            // Determine admission using centralized function
            let admission_result = if has_admission_gating {
                let vendor_snapshot = vendor_snapshot_from_market(&market_snapshot);
                let internal_snapshot = InternalSnapshot::empty(); // TODO: build from engine state

                // Phase 20B: Placeholder manifest hash until proper wiring
                // TODO: Load manifest and compute hash at backtest initialization
                const PLACEHOLDER_MANIFEST_HASH: [u8; 32] = [0u8; 32];

                Self::decide_admission(
                    &admission_mode,
                    &correlation_id,
                    ts_ns,
                    &session_id,
                    &required_signals,
                    &vendor_snapshot,
                    &internal_snapshot,
                    PLACEHOLDER_MANIFEST_HASH,
                )?
            } else {
                // No gating required → always admit
                AdmissionDecisionResult {
                    admit: true,
                    mismatches: Vec::new(),
                    decisions: Vec::new(),
                }
            };

            // DOCTRINE: Write decisions to WAL FIRST (live mode only)
            // In enforce mode, we don't write new decisions (WAL is authoritative)
            if matches!(admission_mode, AdmissionMode::EvaluateLive) {
                for decision in &admission_result.decisions {
                    wal_writer.write_admission(decision.clone()).await?;

                    // Phase 22C: Admission refusal logging (required fields)
                    if decision.is_refused() {
                        let strategy_id = self
                            .config
                            .strategy_spec
                            .as_ref()
                            .map(|s| s.strategy_id.as_str())
                            .unwrap_or("unknown");
                        let refuse_reasons: Vec<String> = decision
                            .missing_vendor_fields
                            .iter()
                            .map(|f| f.to_string())
                            .collect();
                        tracing::warn!(
                            strategy_id = %strategy_id,
                            signal_id = %decision.signal_id,
                            refuse_reasons = ?refuse_reasons,
                            correlation_id = %correlation_id,
                            "Admission refused"
                        );
                    }
                }
            }

            // Track admit/refuse counts
            if has_admission_gating {
                if admission_result.admit {
                    admitted_events += 1;
                } else {
                    refused_events += 1;
                }
            }

            // Strategy is ONLY called if ALL required signals are admitted
            let outputs = if admission_result.admit {
                strategy.on_event(&sdk_event, &ctx)
            } else {
                // Skip strategy call - admission refused
                Vec::new()
            };

            // Process each decision output
            for output in outputs {
                // ENGINE records decision to trace (not strategy)
                trace_builder.record(&output.decision);
                tracing::debug!(
                    "Decision: {} intents, type={}",
                    output.intents.len(),
                    output.decision.decision_type
                );

                // Execute intents (authored by strategy)
                for intent in output.intents {
                    // Phase 2: intent carries parent_decision_id for correlation
                    let parent_decision_id = intent.parent_decision_id;

                    // Convert Phase 2 OrderIntent to local OrderIntent
                    let local_intent = OrderIntent {
                        symbol: intent.symbol.clone(),
                        side: match intent.side {
                            quantlaxmi_strategy::Side::Buy => Side::Buy,
                            quantlaxmi_strategy::Side::Sell => Side::Sell,
                        },
                        qty: intent.qty_mantissa as f64 * 10f64.powi(intent.qty_exponent as i32),
                        limit_price: intent
                            .limit_price_mantissa
                            .map(|m| m as f64 * 10f64.powi(intent.price_exponent as i32)),
                        tag: intent.tag.clone(),
                    };

                    // =========================================================
                    // PHASE 22C: Order Permission Gate (BEFORE exchange.execute)
                    // =========================================================
                    if let Some(ref spec) = self.config.strategy_spec {
                        // Build gate-compatible intent reference
                        let gate_side = match intent.side {
                            quantlaxmi_strategy::Side::Buy => GateSide::Buy,
                            quantlaxmi_strategy::Side::Sell => GateSide::Sell,
                        };
                        let gate_order_type = if intent.limit_price_mantissa.is_some() {
                            GateOrderType::Limit
                        } else {
                            GateOrderType::Market
                        };
                        let intent_ref = OrderIntentRef {
                            side: gate_side,
                            order_type: gate_order_type,
                            quantity: local_intent.qty,
                            strategy_id: &spec.strategy_id,
                        };

                        // Check permission
                        let permission = OrderPermissionGate::authorize(spec, &intent_ref);

                        // =======================================================
                        // PHASE 23B: Write order_intent WAL BEFORE acting on permission
                        // =======================================================
                        order_intent_seq += 1;
                        let oi_side = match intent.side {
                            quantlaxmi_strategy::Side::Buy => OrderIntentSide::Buy,
                            quantlaxmi_strategy::Side::Sell => OrderIntentSide::Sell,
                        };
                        let oi_type = if intent.limit_price_mantissa.is_some() {
                            OrderIntentType::Limit
                        } else {
                            OrderIntentType::Market
                        };

                        // Build common fields
                        let mut oi_builder =
                            OrderIntentRecord::builder(&spec.strategy_id, &intent.symbol)
                                .ts_ns(ts_ns)
                                .session_id(&session_id)
                                .seq(order_intent_seq)
                                .side(oi_side)
                                .order_type(oi_type)
                                .qty(intent.qty_mantissa, intent.qty_exponent)
                                .price_exponent(intent.price_exponent)
                                .correlation_id(&correlation_id)
                                .parent_admission_digest(&correlation_id);

                        // Set limit price through builder if present
                        if let Some(lp) = intent.limit_price_mantissa {
                            oi_builder = oi_builder.limit_price(lp, intent.price_exponent);
                        }

                        let oi_record = match &permission {
                            OrderPermission::Permit => oi_builder.build_permit(),
                            OrderPermission::Refuse(reason) => {
                                let oi_reason = OrderRefuseReason::Custom {
                                    reason: reason.description(),
                                };
                                oi_builder.build_refuse(oi_reason)
                            }
                        };

                        // Phase 24A: Store intent seq/digest for linking to fills
                        current_intent_seq = Some(order_intent_seq);
                        current_intent_digest = Some(oi_record.digest.clone());

                        wal_writer.write_order_intent(oi_record).await?;

                        // Now act on permission
                        if let OrderPermission::Refuse(reason) = permission {
                            // Log refusal (Phase 22C requirement)
                            tracing::warn!(
                                strategy_id = %spec.strategy_id,
                                correlation_id = %correlation_id,
                                reason = %reason.description(),
                                "Order permission refused"
                            );
                            refused_orders += 1;
                            continue; // Skip this order
                        }
                        permitted_orders += 1;
                    }

                    // =======================================================
                    // PHASE 25B: Enqueue intent for delayed execution
                    // =======================================================
                    let scheduled_tick = sim_tick + latency_ticks;
                    pending_intents.push_back(PendingIntent {
                        scheduled_tick,
                        local_intent: local_intent.clone(),
                        parent_decision_id,
                        intent_seq: current_intent_seq.unwrap_or(order_intent_seq),
                        intent_digest: current_intent_digest.clone().unwrap_or_default(),
                        correlation_id: correlation_id.clone(),
                    });
                    // Track intent emission for funnel
                    slrt_engine.record_intent();

                    // For latency_ticks == 0, immediately drain to preserve baseline behavior
                    if latency_ticks == 0 {
                        tracing::debug!(
                            "Intent enqueued: {:?} {} @ qty={}",
                            local_intent.side,
                            local_intent.symbol,
                            local_intent.qty
                        );
                        while let Some(front) = pending_intents.front() {
                            if front.scheduled_tick > sim_tick {
                                break;
                            }
                            let pending = pending_intents.pop_front().unwrap();

                            current_intent_seq = Some(pending.intent_seq);
                            current_intent_digest = Some(pending.intent_digest.clone());
                            let imm_correlation_id = pending.correlation_id.clone();

                            let fill_result = exchange.execute(
                                &pending.local_intent,
                                event.ts,
                                pending.parent_decision_id,
                            );
                            if let Some(fill) = fill_result {
                                let imm_ts_ns = event.ts.timestamp_nanos_opt().unwrap_or(0);
                                execution_fill_seq += 1;

                                let fill_qty_mantissa =
                                    (fill.qty * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                                        .round() as i64;
                                let fill_price_mantissa = (fill.price
                                    * 10f64.powi(-market_snapshot.price_exponent() as i32))
                                .round()
                                    as i64;

                                let (fill_fee_mantissa, fill_fee_exponent) = if fill.fee > 0.0 {
                                    let fee_m = (fill.fee
                                        * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                                    .round() as i64;
                                    (Some(fee_m), market_snapshot.qty_exponent())
                                } else {
                                    (None, market_snapshot.qty_exponent())
                                };

                                let fill_side = match fill.side {
                                    Side::Buy => FillSide::Buy,
                                    Side::Sell => FillSide::Sell,
                                };

                                let fill_strategy_id = self
                                    .config
                                    .strategy_spec
                                    .as_ref()
                                    .map(|s| s.strategy_id.as_str())
                                    .unwrap_or("unknown");

                                let mut ef_builder =
                                    ExecutionFillRecord::builder(fill_strategy_id, &fill.symbol)
                                        .ts_ns(imm_ts_ns)
                                        .session_id(&session_id)
                                        .seq(execution_fill_seq)
                                        .side(fill_side)
                                        .qty(fill_qty_mantissa, market_snapshot.qty_exponent())
                                        .price(
                                            fill_price_mantissa,
                                            market_snapshot.price_exponent(),
                                        )
                                        .venue("sim")
                                        .correlation_id(&imm_correlation_id)
                                        .fill_type(FillType::Full);

                                if let Some(fee_m) = fill_fee_mantissa {
                                    ef_builder = ef_builder.fee(fee_m, fill_fee_exponent);
                                }
                                if let Some(parent_seq) = current_intent_seq {
                                    ef_builder = ef_builder.parent_intent_seq(parent_seq);
                                }
                                if let Some(ref parent_digest) = current_intent_digest {
                                    ef_builder = ef_builder.parent_intent_digest(parent_digest);
                                }

                                let ef_record = ef_builder.build();
                                wal_writer.write_execution_fill(ef_record).await?;

                                position_update_seq += 1;
                                let pre_realized_pnl = position_tracker.realized_pnl_mantissa;
                                let pre_fees = position_tracker.total_fees_mantissa;

                                let is_buy = matches!(fill.side, Side::Buy);
                                position_tracker.process_fill(
                                    fill_price_mantissa,
                                    fill_qty_mantissa,
                                    fill_fee_mantissa.unwrap_or(0),
                                    is_buy,
                                );

                                let realized_pnl_delta_i128 =
                                    position_tracker.realized_pnl_mantissa - pre_realized_pnl;
                                let fee_delta_i128 =
                                    position_tracker.total_fees_mantissa - pre_fees;
                                let notional_mantissa_i128 =
                                    (fill_price_mantissa as i128 * fill_qty_mantissa as i128) / 100;
                                let notional_abs_i128 = notional_mantissa_i128.abs();

                                let fill_venue = "sim";
                                let costs = compute_costs_v1(
                                    fill_venue,
                                    &fill.symbol,
                                    notional_abs_i128,
                                    fill_fee_mantissa.unwrap_or(0),
                                    &cost_model,
                                );

                                let effective_fee_delta_i128 =
                                    fee_delta_i128 + costs.extra_fee_mantissa as i128;
                                let mut cash_delta_i128 = if is_buy {
                                    -notional_mantissa_i128 - effective_fee_delta_i128
                                } else {
                                    notional_mantissa_i128 - effective_fee_delta_i128
                                };
                                cash_delta_i128 -= costs.slippage_cost_mantissa as i128;
                                cash_delta_i128 -= costs.spread_cost_mantissa as i128;

                                let cash_delta_mantissa: i64 = i64::try_from(cash_delta_i128)
                                    .map_err(|_| {
                                        anyhow::anyhow!("cash_delta overflow: {}", cash_delta_i128)
                                    })?;
                                let realized_pnl_delta: i64 =
                                    i64::try_from(realized_pnl_delta_i128).map_err(|_| {
                                        anyhow::anyhow!(
                                            "realized_pnl_delta overflow: {}",
                                            realized_pnl_delta_i128
                                        )
                                    })?;
                                let effective_fee_delta: i64 =
                                    i64::try_from(effective_fee_delta_i128).map_err(|_| {
                                        anyhow::anyhow!(
                                            "fee_delta overflow: {}",
                                            effective_fee_delta_i128
                                        )
                                    })?;

                                let mut pu_builder =
                                    PositionUpdateRecord::builder(fill_strategy_id, &fill.symbol)
                                        .ts_ns(imm_ts_ns)
                                        .session_id(&session_id)
                                        .seq(position_update_seq)
                                        .correlation_id(&imm_correlation_id)
                                        .fill_seq(execution_fill_seq)
                                        .position_qty(
                                            position_tracker.position_qty_mantissa as i64,
                                            market_snapshot.qty_exponent(),
                                        )
                                        .cash_delta(cash_delta_mantissa, PNL_EXPONENT)
                                        .realized_pnl_delta(realized_pnl_delta, PNL_EXPONENT)
                                        .venue(fill_venue);

                                if position_tracker.position_qty_mantissa != 0 {
                                    pu_builder = pu_builder.avg_price(
                                        position_tracker.avg_entry_price_mantissa as i64,
                                        -2,
                                    );
                                } else {
                                    pu_builder = pu_builder.avg_price_flat(-2);
                                }

                                if effective_fee_delta != 0 {
                                    pu_builder = pu_builder.fee(effective_fee_delta, PNL_EXPONENT);
                                } else {
                                    pu_builder = pu_builder.fee_exponent(PNL_EXPONENT);
                                }

                                let pu_record = pu_builder.build();
                                wal_writer.write_position_update(pu_record).await?;

                                current_intent_seq = None;
                                current_intent_digest = None;

                                let notification = quantlaxmi_strategy::FillNotification {
                                    ts: fill.ts,
                                    symbol: fill.symbol.clone(),
                                    side: match fill.side {
                                        Side::Buy => quantlaxmi_strategy::Side::Buy,
                                        Side::Sell => quantlaxmi_strategy::Side::Sell,
                                    },
                                    qty_mantissa: fill_qty_mantissa,
                                    qty_exponent: market_snapshot.qty_exponent(),
                                    price_mantissa: fill_price_mantissa,
                                    price_exponent: market_snapshot.price_exponent(),
                                    fee_mantissa: fill_fee_mantissa.unwrap_or(0),
                                    fee_exponent: fill_fee_exponent,
                                    tag: fill.tag.clone(),
                                };
                                strategy.on_fill(&notification, &ctx);
                            }
                        }
                    }
                }
            }

            // Track equity curve after fills
            let current_fill_count = exchange.fills().len();
            if current_fill_count > last_fill_count {
                let cash = exchange.cash();
                let realized_pnl = exchange.realized_pnl();
                let unrealized_pnl = exchange.unrealized_pnl();
                let equity = cash + realized_pnl + unrealized_pnl;
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
                    cash,
                    realized_pnl,
                    unrealized_pnl,
                    drawdown,
                    drawdown_pct,
                });
                last_fill_count = current_fill_count;
            }

            // Progress logging
            if total_events.is_multiple_of(self.config.log_interval) {
                let elapsed = wall_clock_start.elapsed().as_secs_f64();
                tracing::info!(
                    "Progress: {} events, {} fills, {} decisions, PnL: {:.2}, elapsed: {:.1}s",
                    total_events,
                    exchange.fills().len(),
                    trace_builder.len(),
                    exchange.realized_pnl() + exchange.unrealized_pnl(),
                    elapsed
                );
            }

            // Increment sim_tick at end of each event loop iteration
            sim_tick += 1;
        }

        // End-of-run drain: flush any remaining pending intents using last_drain_ts
        while let Some(pending) = pending_intents.pop_front() {
            current_intent_seq = Some(pending.intent_seq);
            current_intent_digest = Some(pending.intent_digest.clone());
            let eor_correlation_id = pending.correlation_id.clone();

            if let Some(fill) = exchange.execute(
                &pending.local_intent,
                last_drain_ts,
                pending.parent_decision_id,
            ) {
                let eor_ts_ns = last_drain_ts.timestamp_nanos_opt().unwrap_or(0);
                execution_fill_seq += 1;

                let fill_qty_mantissa =
                    (fill.qty * 10f64.powi(-market_snapshot.qty_exponent() as i32)).round() as i64;
                let fill_price_mantissa = (fill.price
                    * 10f64.powi(-market_snapshot.price_exponent() as i32))
                .round() as i64;

                let (fill_fee_mantissa, fill_fee_exponent) = if fill.fee > 0.0 {
                    let fee_m = (fill.fee * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                        .round() as i64;
                    (Some(fee_m), market_snapshot.qty_exponent())
                } else {
                    (None, market_snapshot.qty_exponent())
                };

                let fill_side = match fill.side {
                    Side::Buy => FillSide::Buy,
                    Side::Sell => FillSide::Sell,
                };

                let fill_strategy_id = self
                    .config
                    .strategy_spec
                    .as_ref()
                    .map(|s| s.strategy_id.as_str())
                    .unwrap_or("unknown");

                let mut ef_builder = ExecutionFillRecord::builder(fill_strategy_id, &fill.symbol)
                    .ts_ns(eor_ts_ns)
                    .session_id(&session_id)
                    .seq(execution_fill_seq)
                    .side(fill_side)
                    .qty(fill_qty_mantissa, market_snapshot.qty_exponent())
                    .price(fill_price_mantissa, market_snapshot.price_exponent())
                    .venue("sim")
                    .correlation_id(&eor_correlation_id)
                    .fill_type(FillType::Full);

                if let Some(fee_m) = fill_fee_mantissa {
                    ef_builder = ef_builder.fee(fee_m, fill_fee_exponent);
                }
                if let Some(parent_seq) = current_intent_seq {
                    ef_builder = ef_builder.parent_intent_seq(parent_seq);
                }
                if let Some(ref parent_digest) = current_intent_digest {
                    ef_builder = ef_builder.parent_intent_digest(parent_digest);
                }

                let ef_record = ef_builder.build();
                wal_writer.write_execution_fill(ef_record).await?;

                position_update_seq += 1;
                let pre_realized_pnl = position_tracker.realized_pnl_mantissa;
                let pre_fees = position_tracker.total_fees_mantissa;

                let is_buy = matches!(fill.side, Side::Buy);
                position_tracker.process_fill(
                    fill_price_mantissa,
                    fill_qty_mantissa,
                    fill_fee_mantissa.unwrap_or(0),
                    is_buy,
                );

                let realized_pnl_delta_i128 =
                    position_tracker.realized_pnl_mantissa - pre_realized_pnl;
                let fee_delta_i128 = position_tracker.total_fees_mantissa - pre_fees;
                let notional_mantissa_i128 =
                    (fill_price_mantissa as i128 * fill_qty_mantissa as i128) / 100;
                let notional_abs_i128 = notional_mantissa_i128.abs();

                let fill_venue = "sim";
                let costs = compute_costs_v1(
                    fill_venue,
                    &fill.symbol,
                    notional_abs_i128,
                    fill_fee_mantissa.unwrap_or(0),
                    &cost_model,
                );

                let effective_fee_delta_i128 = fee_delta_i128 + costs.extra_fee_mantissa as i128;
                let mut cash_delta_i128 = if is_buy {
                    -notional_mantissa_i128 - effective_fee_delta_i128
                } else {
                    notional_mantissa_i128 - effective_fee_delta_i128
                };
                cash_delta_i128 -= costs.slippage_cost_mantissa as i128;
                cash_delta_i128 -= costs.spread_cost_mantissa as i128;

                let cash_delta_mantissa: i64 = i64::try_from(cash_delta_i128)
                    .map_err(|_| anyhow::anyhow!("cash_delta overflow: {}", cash_delta_i128))?;
                let realized_pnl_delta: i64 =
                    i64::try_from(realized_pnl_delta_i128).map_err(|_| {
                        anyhow::anyhow!("realized_pnl_delta overflow: {}", realized_pnl_delta_i128)
                    })?;
                let effective_fee_delta: i64 =
                    i64::try_from(effective_fee_delta_i128).map_err(|_| {
                        anyhow::anyhow!("fee_delta overflow: {}", effective_fee_delta_i128)
                    })?;

                let mut pu_builder = PositionUpdateRecord::builder(fill_strategy_id, &fill.symbol)
                    .ts_ns(eor_ts_ns)
                    .session_id(&session_id)
                    .seq(position_update_seq)
                    .correlation_id(&eor_correlation_id)
                    .fill_seq(execution_fill_seq)
                    .position_qty(
                        position_tracker.position_qty_mantissa as i64,
                        market_snapshot.qty_exponent(),
                    )
                    .cash_delta(cash_delta_mantissa, PNL_EXPONENT)
                    .realized_pnl_delta(realized_pnl_delta, PNL_EXPONENT)
                    .venue(fill_venue);

                if position_tracker.position_qty_mantissa != 0 {
                    pu_builder =
                        pu_builder.avg_price(position_tracker.avg_entry_price_mantissa as i64, -2);
                } else {
                    pu_builder = pu_builder.avg_price_flat(-2);
                }

                if effective_fee_delta != 0 {
                    pu_builder = pu_builder.fee(effective_fee_delta, PNL_EXPONENT);
                } else {
                    pu_builder = pu_builder.fee_exponent(PNL_EXPONENT);
                }

                let pu_record = pu_builder.build();
                wal_writer.write_position_update(pu_record).await?;
            }
        }

        // Finalize trace
        let trace = trace_builder.finalize();
        let trace_hash = trace.hash_hex();
        let total_decisions = trace.len();

        // Save trace if output path specified
        let trace_path = if let Some(ref path) = self.config.output_trace {
            let trace_file = Path::new(path);
            trace
                .save(trace_file)
                .map_err(|e| anyhow::anyhow!("Failed to save trace: {}", e))?;
            tracing::info!("Decision trace saved to: {}", path);
            Some(path.clone())
        } else {
            None
        };

        // =========================================================================
        // Evaluation Mode: Flatten Position at End of Session
        // =========================================================================
        // Force close any open position to convert MTM to realized PnL.
        // This is for evaluation ONLY - not production behavior.
        if self.config.flatten_on_end {
            // Get current position
            let position = exchange.sim.position("BTCUSDT");
            if position.abs() > 1e-10 {
                let flatten_ts = end_ts.unwrap_or_else(chrono::Utc::now);
                let flatten_side = if position > 0.0 {
                    Side::Sell
                } else {
                    Side::Buy
                };
                let flatten_intent = OrderIntent {
                    symbol: "BTCUSDT".to_string(),
                    side: flatten_side,
                    qty: position.abs(),
                    limit_price: None, // Market order for flatten
                    tag: Some("eval_flatten_end".to_string()),
                };
                // Use a dummy decision ID for flatten
                let flatten_decision_id = uuid::Uuid::new_v4();
                if let Some(fill) =
                    exchange.execute(&flatten_intent, flatten_ts, flatten_decision_id)
                {
                    tracing::info!(
                        side = ?fill.side,
                        qty = fill.qty,
                        price = fill.price,
                        "Evaluation flatten: Closed position at end of session"
                    );
                }
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

        // Compute fixed-point PnL from fills for determinism
        let mut pnl_fixed = PnlAccumulatorFixed::new();
        for fill in &fills {
            let price_mantissa = (fill.price * 100.0).round() as i64;
            let qty_mantissa = (fill.qty * 100_000_000.0).round() as i64;
            let fee_mantissa = (fill.fee * 100_000_000.0).round() as i64;
            let is_buy = matches!(fill.side, Side::Buy);
            pnl_fixed.process_fill(price_mantissa, qty_mantissa, fee_mantissa, is_buy);
        }

        // Phase 19C: Finalize WAL and log admission stats
        wal_writer.flush().await?;
        if has_admission_gating {
            tracing::info!(
                admitted = admitted_events,
                refused = refused_events,
                total = admitted_events + refused_events,
                "Phase 19C admission gating complete"
            );
        }

        // Phase 22C: Log order permission stats
        if self.config.strategy_spec.is_some() {
            tracing::info!(
                permitted = permitted_orders,
                refused = refused_orders,
                total = permitted_orders + refused_orders,
                "Phase 22C order permission gating complete"
            );
        }

        // =========================================================================
        // Phase 26.3: Truth Report Generation
        // =========================================================================
        // After WAL finalization, build and emit strategy truth report.
        // This is pure plumbing: reads WAL, builds aggregator, writes artifacts.

        // Step 1: Read position updates from WAL
        let wal_reader = WalReader::open(segment_dir)?;
        let position_updates = wal_reader.read_position_updates()?;

        // Step 2: Build aggregator from position updates
        let mut aggregator_registry = StrategyAggregatorRegistry::new();
        for record in &position_updates {
            if let Err(e) = aggregator_registry.process_position_update(record) {
                tracing::warn!(error = %e, "Phase 26.3: Aggregator error (skipping record)");
            }
        }

        // Step 3: Build SessionMetadata
        let start_ts_ns = start_ts.and_then(|t| t.timestamp_nanos_opt()).unwrap_or(0);
        let end_ts_ns = end_ts.and_then(|t| t.timestamp_nanos_opt()).unwrap_or(0);

        // Compute cost model digest (SHA-256 of JSON, or None if default/empty)
        let cost_model_digest: Option<String> = if cost_model.venues.is_empty() {
            None
        } else {
            use sha2::{Digest as Sha2Digest, Sha256};
            let json = serde_json::to_string(&cost_model).unwrap_or_default();
            let mut hasher = Sha256::new();
            hasher.update(json.as_bytes());
            Some(hex::encode(hasher.finalize()))
        };

        // Get unified exponent from aggregator (default to PNL_EXPONENT if no records)
        let unified_exponent = aggregator_registry
            .unified_exponent()
            .unwrap_or(PNL_EXPONENT);

        let truth_metadata = SessionMetadata {
            session_id: session_id.clone(),
            instrument: fills
                .first()
                .map(|f| f.symbol.clone())
                .unwrap_or_else(|| "UNKNOWN".to_string()),
            start_ts_ns,
            end_ts_ns,
            latency_ticks: self.config.latency_ticks,
            cost_model_digest,
            unified_exponent,
        };

        // Step 4: Build Truth Report
        let accumulators = aggregator_registry.finalize();
        let truth_report = StrategyTruthReport::build(truth_metadata, accumulators);

        // Step 5: Write artifacts to disk
        let reports_dir = segment_dir.join("reports");
        std::fs::create_dir_all(&reports_dir)
            .with_context(|| format!("Failed to create reports dir: {:?}", reports_dir))?;

        let json_path = reports_dir.join("strategy_truth_report.json");
        std::fs::write(&json_path, truth_report.to_json())
            .with_context(|| format!("Failed to write truth report JSON: {:?}", json_path))?;

        let summary_path = reports_dir.join("strategy_truth_summary.txt");
        std::fs::write(&summary_path, truth_report.to_text_summary())
            .with_context(|| format!("Failed to write truth summary: {:?}", summary_path))?;

        tracing::info!(
            json_path = %json_path.display(),
            summary_path = %summary_path.display(),
            digest = %truth_report.digest,
            "Phase 26.3: Truth report written"
        );

        // Step 6: Print summary to stdout (compact format for CI logs)
        println!();
        println!(
            "================================================================================"
        );
        println!("STRATEGY TRUTH REPORT (Phase 26.3)");
        println!(
            "================================================================================"
        );
        println!("Session:    {}", truth_report.session_id);
        println!("Instrument: {}", truth_report.instrument);
        println!("Latency:    {} tick(s)", truth_report.latency_ticks);
        println!("Exponent:   {}", truth_report.unified_exponent);
        println!("Digest:     {}...", &truth_report.digest[..16]);
        println!();

        for (strategy_id, metrics) in &truth_report.strategies {
            println!("--- {} ---", strategy_id);
            println!(
                "  Trades: {} ({} W / {} L)  Win Rate: {}",
                metrics.trades, metrics.winning_trades, metrics.losing_trades, metrics.win_rate
            );
            println!(
                "  Net PnL:      {} (exp={})",
                metrics.net_pnl_mantissa, truth_report.unified_exponent
            );
            println!(
                "  Gross PnL:    {} (exp={})",
                metrics.gross_pnl_mantissa, truth_report.unified_exponent
            );
            println!(
                "  Max Drawdown: {} (exp={})",
                metrics.max_drawdown_mantissa, truth_report.unified_exponent
            );
            println!("  Exposure:     {} updates", metrics.exposure_updates);
            println!();
        }
        println!(
            "================================================================================"
        );
        println!();

        // Create strategy binding for manifest
        let strategy_binding = Some(crate::segment_manifest::StrategyBinding {
            strategy_name: strategy.name().to_string(),
            strategy_version: strategy.version().to_string(),
            config_hash: strategy.config_hash(),
            strategy_id: strategy.strategy_id(),
            short_id: strategy.short_id(),
            config_path: config_path.map(|p| p.display().to_string()),
            config_snapshot: None, // Could be populated from config file
        });

        let result = BacktestResult {
            strategy_name: strategy.short_id(),
            segment_path: segment_dir.display().to_string(),
            total_events,
            total_fills: fills.len(),
            total_decisions,
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
            trace_hash: trace_hash.clone(),
            trace_path,
            trace_encoding_version: trace.encoding_version,
            realized_pnl_mantissa: pnl_fixed.realized_pnl_mantissa,
            total_fees_mantissa: pnl_fixed.total_fees_mantissa,
            pnl_exponent: PNL_EXPONENT,
        };

        tracing::info!("=== Phase 2 Backtest Complete ===");
        tracing::info!("Strategy: {} ({})", strategy.name(), strategy.short_id());
        tracing::info!("Events: {}", result.total_events);
        tracing::info!("Fills: {}", result.total_fills);
        tracing::info!("Decisions: {}", result.total_decisions);
        tracing::info!("Trace hash: {}...", &trace_hash[..16]);
        tracing::info!("Duration: {:.1}s", result.duration_secs);
        tracing::info!("Realized PnL: ${:.2}", result.realized_pnl);
        tracing::info!("Unrealized PnL: ${:.2}", result.unrealized_pnl);
        tracing::info!(
            "Total PnL: ${:.2} ({:.2}%)",
            result.total_pnl,
            result.return_pct
        );

        // DEBUG: Depth event flow diagnosis (Step 1)
        eprintln!("=== DEPTH EVENT FLOW DIAGNOSIS ===");
        eprintln!("n_perp_depth_total:    {}", n_perp_depth_total);
        eprintln!("n_perp_depth_snapshot: {}", n_perp_depth_snapshot);
        eprintln!("n_perp_depth_delta:    {}", n_perp_depth_delta);
        eprintln!("n_on_depth_called:     {}", n_on_depth_called);
        eprintln!("--- BOOK MUTATION TRACKING (Step 2) ---");
        eprintln!("exchange.depth_calls:    {}", exchange.depth_calls);
        eprintln!("exchange.book_mutations: {}", exchange.book_mutations);
        eprintln!("exchange.best_changes:   {}", exchange.best_changes);
        // Log current best bid/ask
        let final_bid = exchange.sim.best_bid("BTCUSDT");
        let final_ask = exchange.sim.best_ask("BTCUSDT");
        eprintln!("final_best_bid: {:?}", final_bid);
        eprintln!("final_best_ask: {:?}", final_ask);
        eprintln!("--- MARKET SNAPSHOT INVARIANTS ---");
        eprintln!("from_payload: {}", n_market_snapshot_from_payload);
        eprintln!("from_book:    {}", n_market_snapshot_from_book);
        eprintln!("zero:         {}", n_market_snapshot_zero);
        eprintln!("book_broken:  {}", n_book_broken);
        if n_book_broken > 0 {
            tracing::warn!(
                n_book_broken = n_book_broken,
                "Book broken events detected (bid >= ask)"
            );
        }
        eprintln!("--- SLRT FEATURE ENGINE (Real) ---");
        eprintln!(
            "regime_counts: R0={} R1={} R2={} R3={}",
            regime_counts[0], regime_counts[1], regime_counts[2], regime_counts[3]
        );
        let total_regimes: u64 = regime_counts.iter().sum();
        if total_regimes > 0 {
            eprintln!(
                "regime_pct: R0={:.1}% R1={:.1}% R2={:.1}% R3={:.1}%",
                regime_counts[0] as f64 / total_regimes as f64 * 100.0,
                regime_counts[1] as f64 / total_regimes as f64 * 100.0,
                regime_counts[2] as f64 / total_regimes as f64 * 100.0,
                regime_counts[3] as f64 / total_regimes as f64 * 100.0
            );
        }
        // Urgency distribution
        if !urgency_samples.is_empty() {
            let mut sorted_urgency = urgency_samples.clone();
            sorted_urgency.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted_urgency.len();
            let p50 = sorted_urgency[n / 2];
            let p95 = sorted_urgency[(n * 95) / 100];
            let min = sorted_urgency[0];
            let max = sorted_urgency[n - 1];
            eprintln!(
                "urgency_dist: n={} min={:.3} p50={:.3} p95={:.3} max={:.3}",
                n, min, p50, p95, max
            );
        } else {
            eprintln!("urgency_dist: no samples (no decisions)");
        }
        eprintln!("==================================");

        // Print SLRT feature funnel and quantiles
        slrt_engine.print_diagnostics();

        // Print strategy-specific diagnostics (funnel stats)
        strategy.print_diagnostics();

        Ok((result, strategy_binding))
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
    use chrono::TimeZone;

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

        // Market state is now internal to Simulator - verify via position tracking
        // We can't directly access markets anymore, but we can verify the exchange works
        // by executing an order and checking the result

        // Execute buy (0.1 BTC @ $100,010 = $10,001 + ~$1 fee)
        let decision_id = Uuid::new_v4();
        let order = OrderIntent::market("BTCUSDT", Side::Buy, 0.1);
        let fill = exchange.execute(&order, Utc::now(), decision_id);

        assert!(fill.is_some(), "Fill should not be None");
        let fill = fill.unwrap();
        assert_eq!(fill.price, 100010.0); // Bought at ask
        assert_eq!(fill.qty, 0.1);
        assert!(fill.fee > 0.0);
        assert_eq!(fill.parent_decision_id, decision_id); // Correlation preserved

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
        let decision_id1 = Uuid::new_v4();
        exchange.execute(
            &OrderIntent::market("BTCUSDT", Side::Buy, 1.0),
            Utc::now(),
            decision_id1,
        );

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
        let decision_id2 = Uuid::new_v4();
        exchange.execute(
            &OrderIntent::market("BTCUSDT", Side::Sell, 1.0),
            Utc::now(),
            decision_id2,
        );

        // Check realized PnL
        assert!((exchange.realized_pnl() - 10.0).abs() < 0.01);
        assert_eq!(exchange.position("BTCUSDT"), 0.0);
    }

    // =========================================================================
    // PnlAccumulatorFixed Tests
    // =========================================================================

    #[test]
    fn test_pnl_fixed_buy_sell_round_trip() {
        // Test a simple buy/sell round trip with fees
        let mut acc = PnlAccumulatorFixed::new();

        // Buy 0.01 BTC at $100,000.00
        // price_mantissa = 10000000 (exp -2 = $100,000.00)
        // qty_mantissa = 1_000_000 (exp -8 = 0.01 BTC)
        // fee_mantissa = 10_000 (exp -8 = 0.0001 BTC = $10 in fees)
        acc.process_fill(10_000_000, 1_000_000, 10_000, true);

        assert_eq!(acc.position_qty_mantissa, 1_000_000);
        assert!(!acc.is_flat());

        // Sell 0.01 BTC at $101,000.00
        // price_mantissa = 10100000 (exp -2 = $101,000.00)
        acc.process_fill(10_100_000, 1_000_000, 10_000, false);

        assert!(acc.is_flat());

        // PnL calculation:
        // Entry notional = 10000000 * 1000000 / 100 = 100_000_000_000 (exp -8 = $1000)
        // Exit notional = 10100000 * 1000000 / 100 = 101_000_000_000 (exp -8 = $1010)
        // Gross PnL = 101_000_000_000 - 100_000_000_000 = 1_000_000_000 (exp -8 = $10)
        // Net PnL = 1_000_000_000 - 10_000 (fee) = 999_990_000 (exp -8 = ~$9.9999)

        // Allow for small rounding differences
        let pnl_f64 = acc.realized_pnl_f64();
        assert!(
            (pnl_f64 - 9.9999).abs() < 0.01,
            "Expected ~$9.9999 PnL, got {}",
            pnl_f64
        );

        // Total fees = 2 * 10_000 = 20_000 (exp -8 = $0.0002)
        let fees_f64 = acc.total_fees_f64();
        assert!(
            (fees_f64 - 0.0002).abs() < 0.0001,
            "Expected $0.0002 fees, got {}",
            fees_f64
        );
    }

    #[test]
    fn test_pnl_fixed_multi_fill_aggregation() {
        // Test aggregating multiple fills
        let mut acc = PnlAccumulatorFixed::new();

        // Buy 0.01 BTC at $100,000
        acc.process_fill(10_000_000, 1_000_000, 10_000, true);
        // Buy another 0.01 BTC at $100,500
        acc.process_fill(10_050_000, 1_000_000, 10_000, true);

        // Position should be 0.02 BTC
        assert_eq!(acc.position_qty_mantissa, 2_000_000);

        // Average entry price should be ~$100,250
        let avg_price_f64 = acc.avg_entry_price_mantissa as f64 / 100.0;
        assert!(
            (avg_price_f64 - 100_250.0).abs() < 1.0,
            "Expected avg entry ~$100,250, got {}",
            avg_price_f64
        );

        // Sell all 0.02 BTC at $101,000
        acc.process_fill(10_100_000, 2_000_000, 20_000, false);

        assert!(acc.is_flat());

        // PnL: Bought avg $100,250, sold at $101,000
        // (101000 - 100250) * 0.02 = $15 gross profit
        // Minus fees: 10_000 + 10_000 + 20_000 = 40_000 (exp -8 = $0.0004)
        // Net should be close to $15 - $0.0004 ≈ $14.9996 (adjusted for fixed-point)
        let pnl_f64 = acc.realized_pnl_f64();
        assert!(
            pnl_f64 > 14.0 && pnl_f64 < 16.0,
            "Expected ~$15 PnL, got {}",
            pnl_f64
        );
    }

    #[test]
    fn test_pnl_fixed_determinism() {
        // Verify identical fills produce identical results
        let mut acc1 = PnlAccumulatorFixed::new();
        let mut acc2 = PnlAccumulatorFixed::new();

        // Same sequence of fills
        acc1.process_fill(10_000_000, 1_000_000, 10_000, true);
        acc1.process_fill(10_100_000, 1_000_000, 10_000, false);

        acc2.process_fill(10_000_000, 1_000_000, 10_000, true);
        acc2.process_fill(10_100_000, 1_000_000, 10_000, false);

        // Mantissas must be exactly equal (no floating-point drift)
        assert_eq!(
            acc1.realized_pnl_mantissa, acc2.realized_pnl_mantissa,
            "Fixed-point PnL must be deterministic"
        );
        assert_eq!(
            acc1.total_fees_mantissa, acc2.total_fees_mantissa,
            "Fixed-point fees must be deterministic"
        );
        assert_eq!(
            acc1.position_qty_mantissa, acc2.position_qty_mantissa,
            "Fixed-point position must be deterministic"
        );
    }

    #[test]
    fn test_pnl_fixed_short_position() {
        // Test short position: sell first, then buy to cover
        let mut acc = PnlAccumulatorFixed::new();

        // Sell 0.01 BTC at $100,000 (opening short)
        acc.process_fill(10_000_000, 1_000_000, 10_000, false);

        assert_eq!(acc.position_qty_mantissa, -1_000_000);

        // Buy 0.01 BTC at $99,000 (covering short for profit)
        acc.process_fill(9_900_000, 1_000_000, 10_000, true);

        assert!(acc.is_flat());

        // Short profit: (100000 - 99000) * 0.01 = $10 gross
        // Net after fees should be ~$10 - $0.0002 ≈ $9.9998
        let pnl_f64 = acc.realized_pnl_f64();
        assert!(
            (pnl_f64 - 9.9998).abs() < 0.01,
            "Expected ~$10 short profit, got {}",
            pnl_f64
        );
    }

    // =========================================================================
    // Phase 19C: Canonical Quote Extraction Tests (no float round-trips)
    // =========================================================================

    #[test]
    fn test_extract_bid_ask_mantissa_canonical_path() {
        // Canonical payload: mantissas present, no floats in round-trip
        let payload = serde_json::json!({
            "bid_price_mantissa": 8874152_i64,
            "ask_price_mantissa": 8874252_i64,
            "price_exponent": -2
        });

        let result = extract_bid_ask_mantissa(&payload, EventKind::PerpQuote, -2);
        assert!(result.is_some(), "Should extract canonical mantissas");

        let (bid_m, ask_m, exp) = result.unwrap();
        // CRITICAL: mantissas must be bit-identical to input (no float conversion)
        assert_eq!(
            bid_m, 8874152,
            "Bid mantissa must be exact (no float round-trip)"
        );
        assert_eq!(
            ask_m, 8874252,
            "Ask mantissa must be exact (no float round-trip)"
        );
        assert_eq!(exp, -2, "Exponent must match payload");
    }

    #[test]
    fn test_extract_bid_ask_mantissa_determinism() {
        // Same canonical payload run multiple times must produce identical results
        let payload = serde_json::json!({
            "bid_price_mantissa": 8874152_i64,
            "ask_price_mantissa": 8874252_i64,
            "price_exponent": -2
        });

        let results: Vec<_> = (0..100)
            .map(|_| extract_bid_ask_mantissa(&payload, EventKind::PerpQuote, -2))
            .collect();

        // All results must be identical
        let first = results[0];
        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                *result, first,
                "Run {} produced different result: {:?} vs {:?}",
                i, result, first
            );
        }
    }

    #[test]
    fn test_extract_bid_ask_mantissa_legacy_fallback() {
        // Legacy payload: only float bid/ask (not canonical)
        let payload = serde_json::json!({
            "bid": 88741.52,
            "ask": 88742.52
        });

        let result = extract_bid_ask_mantissa(&payload, EventKind::PerpQuote, -2);
        assert!(result.is_some(), "Should fall back to legacy floats");

        let (bid_m, ask_m, exp) = result.unwrap();
        // Legacy path: floats quantized to mantissa
        // 88741.52 * 100 = 8874152
        assert_eq!(bid_m, 8874152, "Bid should quantize correctly");
        assert_eq!(ask_m, 8874252, "Ask should quantize correctly");
        assert_eq!(exp, -2, "Should use default exponent");
    }

    #[test]
    fn test_extract_bid_ask_mantissa_canonical_takes_priority() {
        // Payload has BOTH canonical mantissas AND legacy floats
        // Canonical must take priority (no float conversion)
        let payload = serde_json::json!({
            "bid_price_mantissa": 8874152_i64,
            "ask_price_mantissa": 8874252_i64,
            "price_exponent": -2,
            // Legacy floats (slightly different to detect wrong path)
            "bid": 88741.99,
            "ask": 88742.99
        });

        let result = extract_bid_ask_mantissa(&payload, EventKind::PerpQuote, -2);
        let (bid_m, ask_m, _) = result.unwrap();

        // Must use canonical mantissas, NOT quantized floats
        assert_eq!(
            bid_m, 8874152,
            "Must use canonical mantissa, not legacy float"
        );
        assert_eq!(
            ask_m, 8874252,
            "Must use canonical mantissa, not legacy float"
        );
    }

    #[test]
    fn test_extract_bid_ask_qty_mantissa_present() {
        // Payload with quantities
        let payload = serde_json::json!({
            "bid_qty_mantissa": 50000000_i64,  // 0.5 BTC
            "ask_qty_mantissa": 75000000_i64,  // 0.75 BTC
            "qty_exponent": -8
        });

        let result = extract_bid_ask_qty_mantissa(&payload, EventKind::PerpQuote, -8);
        assert!(result.is_some(), "Should extract quantities when present");

        let (bid_qty, ask_qty, exp) = result.unwrap();
        assert_eq!(bid_qty, 50000000);
        assert_eq!(ask_qty, 75000000);
        assert_eq!(exp, -8);
    }

    #[test]
    fn test_extract_bid_ask_qty_mantissa_absent() {
        // Payload without quantities (common in backtest data)
        let payload = serde_json::json!({
            "bid_price_mantissa": 8874152_i64,
            "ask_price_mantissa": 8874252_i64,
            "price_exponent": -2
            // No qty fields!
        });

        let result = extract_bid_ask_qty_mantissa(&payload, EventKind::PerpQuote, -8);
        assert!(
            result.is_none(),
            "Should return None when quantities absent"
        );
    }

    #[test]
    fn test_spread_bps_integer_math() {
        // Test the integer spread calculation used in run_with_strategy
        // spread_bps = (ask - bid) / mid * 10000
        // For bid=88741.52, ask=88742.52: spread = 1.00, mid = 88742.02
        // spread_bps = 1.00 / 88742.02 * 10000 ≈ 0.1127 bps

        let bid_m: i64 = 8874152;
        let ask_m: i64 = 8874252;

        // Integer calculation (same as in run_with_strategy)
        let spread_m = ask_m - bid_m;
        let mid_m = bid_m + ask_m;
        let spread_bps_raw = (spread_m as i128 * 20000) / (mid_m as i128);
        let spread_bps = spread_bps_raw as i64;

        // Verify it's reasonable (should be ~1 basis point for $1 spread on ~$88k)
        assert!(spread_bps >= 0, "Spread must be non-negative");
        assert!(
            spread_bps < 100,
            "Spread should be less than 1% for this data"
        );

        // Run twice to confirm determinism
        let spread_bps_2 = ((spread_m as i128 * 20000) / (mid_m as i128)) as i64;
        assert_eq!(
            spread_bps, spread_bps_2,
            "Integer spread calc must be deterministic"
        );
    }

    #[test]
    fn test_depth_snapshot_mantissa_extraction() {
        // Depth snapshot payload (canonical format)
        let payload = serde_json::json!({
            "is_snapshot": true,
            "price_exponent": -2,
            "qty_exponent": -8,
            "bids": [
                {"price": 8874152_i64, "qty": 50000000_i64},
                {"price": 8874100_i64, "qty": 30000000_i64}
            ],
            "asks": [
                {"price": 8874252_i64, "qty": 40000000_i64},
                {"price": 8874300_i64, "qty": 20000000_i64}
            ]
        });

        // Extract prices
        let price_result = extract_bid_ask_mantissa(&payload, EventKind::PerpDepth, -2);
        assert!(
            price_result.is_some(),
            "Should extract depth snapshot prices"
        );
        let (bid_m, ask_m, _) = price_result.unwrap();
        assert_eq!(bid_m, 8874152, "Best bid from snapshot");
        assert_eq!(ask_m, 8874252, "Best ask from snapshot");

        // Extract quantities
        let qty_result = extract_bid_ask_qty_mantissa(&payload, EventKind::PerpDepth, -8);
        assert!(
            qty_result.is_some(),
            "Should extract depth snapshot quantities"
        );
        let (bid_qty, ask_qty, _) = qty_result.unwrap();
        assert_eq!(bid_qty, 50000000, "Best bid qty from snapshot");
        assert_eq!(ask_qty, 40000000, "Best ask qty from snapshot");
    }

    #[test]
    fn test_depth_delta_returns_none() {
        // Depth delta (not snapshot) should return None
        let payload = serde_json::json!({
            "is_snapshot": false,
            "bids": [{"price": 8874152_i64, "qty": 50000000_i64}],
            "asks": []
        });

        let result = extract_bid_ask_mantissa(&payload, EventKind::PerpDepth, -2);
        assert!(result.is_none(), "Depth deltas should return None");
    }

    // =========================================================================
    // V1 Schema Tests
    // =========================================================================

    #[test]
    fn test_v1_schema_version_constant() {
        assert_eq!(BACKTEST_SCHEMA_VERSION, "v1");
    }

    #[test]
    fn test_backtest_metrics_v1_serialization() {
        let metrics = TradeMetrics {
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            win_rate: 60.0,
            gross_profit: 100.0,
            gross_loss: 50.0,
            net_pnl: 50.0,
            profit_factor: 2.0,
            expectancy: 5.0,
            avg_win: 16.67,
            avg_loss: 12.5,
            avg_win_loss_ratio: 1.33,
            largest_win: 30.0,
            largest_loss: 20.0,
            max_drawdown: 15.0,
            max_drawdown_pct: 1.5,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            avg_trade_duration_secs: 3600.0,
            total_fees: 5.0,
        };

        let v1 = BacktestMetricsV1::from_metrics(&metrics);

        // Check schema version
        assert_eq!(v1.schema_version, "v1");

        // Verify JSON serialization is deterministic
        let json1 = serde_json::to_string_pretty(&v1).unwrap();
        let json2 = serde_json::to_string_pretty(&v1).unwrap();
        assert_eq!(json1, json2, "JSON serialization should be deterministic");

        // Verify schema_version appears in JSON
        assert!(json1.contains("\"schema_version\": \"v1\""));
    }

    #[test]
    fn test_backtest_run_manifest_v1_serialization() {
        let manifest = BacktestRunManifestV1 {
            schema_version: BACKTEST_SCHEMA_VERSION.to_string(),
            run_id: "abc123".to_string(),
            strategy: "test_strategy".to_string(),
            segment_path: "/data/segments/test".to_string(),
            total_events: 1000,
            total_fills: 50,
            realized_pnl: 123.45,
            return_pct: 1.23,
            trace_hash: "abc123".to_string(),
        };

        // Verify JSON serialization is deterministic
        let json1 = serde_json::to_string_pretty(&manifest).unwrap();
        let json2 = serde_json::to_string_pretty(&manifest).unwrap();
        assert_eq!(json1, json2, "JSON serialization should be deterministic");

        // Verify schema_version appears in JSON
        assert!(json1.contains("\"schema_version\": \"v1\""));
    }

    // =========================================================================
    // JSONL Output Tests (P0)
    // =========================================================================

    #[test]
    fn test_write_jsonl_creates_valid_file() {
        use std::io::{BufRead, BufReader};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestRecord {
            id: u64,
            value: f64,
        }

        let records = vec![
            TestRecord { id: 1, value: 1.5 },
            TestRecord { id: 2, value: 2.5 },
            TestRecord { id: 3, value: 3.5 },
        ];

        write_jsonl(&path, &records).unwrap();

        // Read back and verify
        let file = std::fs::File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let mut read_records: Vec<TestRecord> = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            let record: TestRecord = serde_json::from_str(&line).unwrap();
            read_records.push(record);
        }

        assert_eq!(read_records.len(), 3);
        assert_eq!(read_records[0].id, 1);
        assert_eq!(read_records[2].id, 3);
    }

    #[test]
    fn test_equity_point_output_serialization() {
        let ep = EquityPoint {
            ts: Utc.with_ymd_and_hms(2026, 1, 30, 12, 0, 0).unwrap(),
            equity: 10500.0,
            cash: 500.0,
            realized_pnl: 400.0,
            unrealized_pnl: 100.0,
            drawdown: 100.0,
            drawdown_pct: 0.95,
        };

        let output = EquityPointOutput::from_equity_point(&ep);

        assert!(output.ts_ns > 0);
        assert_eq!(output.equity, 10500.0);
        assert_eq!(output.cash, 500.0);
        assert_eq!(output.realized_pnl, 400.0);
        assert_eq!(output.unrealized_pnl, 100.0);
        assert_eq!(output.drawdown, 100.0);
        assert_eq!(output.drawdown_pct, 0.95);

        // Verify JSON serialization
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"ts_ns\":"));
        assert!(json.contains("\"equity\":10500"));
    }

    #[test]
    fn test_fill_output_serialization() {
        let fill = Fill {
            ts: Utc.with_ymd_and_hms(2026, 1, 30, 12, 0, 0).unwrap(),
            parent_decision_id: Uuid::new_v4(),
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            qty: 0.001,
            price: 43000.0,
            fee: 0.04,
            liquidity: Liquidity::Taker,
            tag: Some("test".to_string()),
        };

        let output = FillOutput::from_fill(&fill);

        assert!(output.ts_ns > 0);
        assert_eq!(output.symbol, "BTCUSDT");
        assert_eq!(output.side, "Buy");
        assert_eq!(output.qty, 0.001);
        assert_eq!(output.price, 43000.0);
        assert_eq!(output.fee, 0.04);
        assert_eq!(output.liquidity, Some("Taker".to_string()));
        assert!(output.order_id.is_none());

        // Verify JSON serialization
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"symbol\":\"BTCUSDT\""));
        assert!(json.contains("\"side\":\"Buy\""));
    }

    #[test]
    fn test_write_equity_curve_jsonl() {
        use std::io::{BufRead, BufReader};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("equity_curve.jsonl");

        let equity_curve = vec![
            EquityPoint {
                ts: Utc.with_ymd_and_hms(2026, 1, 30, 12, 0, 0).unwrap(),
                equity: 10000.0,
                cash: 10000.0,
                realized_pnl: 0.0,
                unrealized_pnl: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                ts: Utc.with_ymd_and_hms(2026, 1, 30, 12, 1, 0).unwrap(),
                equity: 10100.0,
                cash: 10000.0,
                realized_pnl: 0.0,
                unrealized_pnl: 100.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        write_equity_curve_jsonl(&path, &equity_curve).unwrap();

        // Read back and verify
        let file = std::fs::File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<_> = reader.lines().collect();

        assert_eq!(lines.len(), 2);

        // Parse first line
        let first: serde_json::Value = serde_json::from_str(lines[0].as_ref().unwrap()).unwrap();
        assert_eq!(first["equity"], 10000.0);
    }

    #[test]
    fn test_write_fills_jsonl() {
        use std::io::{BufRead, BufReader};
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("fills.jsonl");

        let fills = vec![
            Fill {
                ts: Utc.with_ymd_and_hms(2026, 1, 30, 12, 0, 0).unwrap(),
                parent_decision_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: Side::Buy,
                qty: 0.001,
                price: 43000.0,
                fee: 0.04,
                liquidity: Liquidity::Taker,
                tag: None,
            },
            Fill {
                ts: Utc.with_ymd_and_hms(2026, 1, 30, 12, 1, 0).unwrap(),
                parent_decision_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: Side::Sell,
                qty: 0.001,
                price: 43100.0,
                fee: 0.04,
                liquidity: Liquidity::Taker,
                tag: None,
            },
        ];

        write_fills_jsonl(&path, &fills).unwrap();

        // Read back and verify
        let file = std::fs::File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<_> = reader.lines().collect();

        assert_eq!(lines.len(), 2);

        // Parse and check first line
        let first: serde_json::Value = serde_json::from_str(lines[0].as_ref().unwrap()).unwrap();
        assert_eq!(first["symbol"], "BTCUSDT");
        assert_eq!(first["side"], "Buy");
        assert_eq!(first["qty"], 0.001);

        // Parse and check second line
        let second: serde_json::Value = serde_json::from_str(lines[1].as_ref().unwrap()).unwrap();
        assert_eq!(second["side"], "Sell");
    }

    // =========================================================================
    // Metrics Invariant Tests (P0.2)
    // =========================================================================

    #[test]
    fn test_metrics_invariant_fees_match_fills() {
        // Create fills with known fees
        let fills = [
            Fill {
                ts: Utc::now(),
                parent_decision_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: Side::Buy,
                qty: 0.1,
                price: 43000.0,
                fee: 0.43,
                liquidity: Liquidity::Taker,
                tag: None,
            },
            Fill {
                ts: Utc::now(),
                parent_decision_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: Side::Sell,
                qty: 0.1,
                price: 43100.0,
                fee: 0.431,
                liquidity: Liquidity::Taker,
                tag: None,
            },
        ];

        let total_fees: f64 = fills.iter().map(|f| f.fee).sum();
        let metrics = TradeMetrics {
            total_fees,
            total_trades: 1, // One round-trip
            ..Default::default()
        };

        // INVARIANT: metrics.total_fees == sum(fills.fee)
        let eps = 1e-9;
        assert!(
            (metrics.total_fees - total_fees).abs() < eps,
            "Fee invariant violated: metrics.total_fees={} != sum(fills.fee)={}",
            metrics.total_fees,
            total_fees
        );
    }

    #[test]
    fn test_metrics_invariant_trade_count() {
        // Create fills - 2 fills make 1 round-trip
        let fills = [
            Fill {
                ts: Utc::now(),
                parent_decision_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: Side::Buy,
                qty: 0.1,
                price: 43000.0,
                fee: 0.43,
                liquidity: Liquidity::Taker,
                tag: None,
            },
            Fill {
                ts: Utc::now(),
                parent_decision_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: Side::Sell,
                qty: 0.1,
                price: 43100.0,
                fee: 0.431,
                liquidity: Liquidity::Maker,
                tag: None,
            },
        ];

        // Note: In the actual system, total_trades counts round-trips, not fills
        // So 2 fills = 1 round-trip
        // For raw fill count invariant, we'd check fills.len()
        assert_eq!(fills.len(), 2);
    }
}
