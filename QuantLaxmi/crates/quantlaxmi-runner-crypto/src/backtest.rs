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
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use quantlaxmi_events::{CorrelationContext, DecisionEvent, DecisionTraceBuilder, MarketSnapshot};
use quantlaxmi_gates::admission::{
    AdmissionContext, InternalSnapshot, SignalAdmissionController, VendorSnapshot,
};
use quantlaxmi_models::{AdmissionDecision, AdmissionOutcome, SignalRequirements};
use quantlaxmi_wal::{AdmissionIndex, AdmissionMismatch, AdmissionMismatchReason, WalWriter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

impl AdmissionMismatchPolicy {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "warn" => Self::Warn,
            _ => Self::Fail,
        }
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
fn order_intent_to_decision(
    order: &OrderIntent,
    ts: DateTime<Utc>,
    strategy_name: &str,
    run_id: &str,
    current_bid: f64,
    current_ask: f64,
    book_ts_ns: i64,
) -> DecisionEvent {
    let decision_id = Uuid::new_v4();

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

            if let (Some(bid_m), Some(ask_m)) = (bid_m, ask_m) {
                if bid_m > 0 && ask_m > 0 {
                    return Some((bid_m, ask_m, price_exp));
                }
            }

            // Legacy fallback: float bid/ask → quantize to mantissa
            // Not doctrine-perfect, but necessary for legacy data
            let bid_f = payload.get("bid").and_then(|v| v.as_f64());
            let ask_f = payload.get("ask").and_then(|v| v.as_f64());

            if let (Some(bid_f), Some(ask_f)) = (bid_f, ask_f) {
                if bid_f > 0.0 && ask_f > 0.0 {
                    let scale = 10f64.powi(-(price_exp as i32));
                    let bid_m = (bid_f * scale).round() as i64;
                    let ask_m = (ask_f * scale).round() as i64;
                    return Some((bid_m, ask_m, price_exp));
                }
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
            parent_decision_id,
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
    /// Optional path to save the decision trace (for replay parity verification).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_trace: Option<String>,
    /// Run ID for correlation context (auto-generated if not provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Phase 19D: Enforce admission from WAL instead of re-evaluating.
    #[serde(default)]
    pub enforce_admission_from_wal: bool,
    /// Phase 19D: Policy when enforcement detects mismatch.
    #[serde(default)]
    pub admission_mismatch_policy: String,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            exchange: ExchangeConfig::default(),
            log_interval: 100_000,
            pace: PaceMode::Fast,
            output_trace: None,
            run_id: None,
            enforce_admission_from_wal: false,
            admission_mismatch_policy: "fail".to_string(),
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
    fn decide_admission(
        mode: &AdmissionMode,
        correlation_id: &str,
        ts_ns: i64,
        session_id: &str,
        required_signals: &[SignalRequirements],
        vendor_snapshot: &VendorSnapshot,
        internal_snapshot: &InternalSnapshot,
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
        let run_id = self
            .config
            .run_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

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
                // Create DecisionEvent from OrderIntent
                let decision = order_intent_to_decision(
                    &order,
                    event.ts,
                    strategy.name(),
                    &run_id,
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
        let run_id = self
            .config
            .run_id
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

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

        // Phase 19D: Build admission mode
        let admission_mode = if self.config.enforce_admission_from_wal && has_admission_gating {
            // Load admission index from existing WAL
            let index = AdmissionIndex::from_wal(segment_dir)
                .context("Failed to load admission WAL for enforcement")?;
            let policy = AdmissionMismatchPolicy::from_str(&self.config.admission_mismatch_policy);
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

            // Update market state (for PaperExchange - still uses floats internally)
            exchange.update_market(&event);

            // DOCTRINE-SAFE: Extract prices as mantissas (no float round-trips)
            const DEFAULT_PRICE_EXP: i8 = -2;
            const DEFAULT_QTY_EXP: i8 = -8;

            if let Some((bid_m, ask_m, price_exp)) =
                extract_bid_ask_mantissa(&event.payload, event.kind, DEFAULT_PRICE_EXP)
            {
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
                payload: event.payload.clone(),
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

                Self::decide_admission(
                    &admission_mode,
                    &correlation_id,
                    ts_ns,
                    &session_id,
                    &required_signals,
                    &vendor_snapshot,
                    &internal_snapshot,
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

                    if decision.is_refused() {
                        tracing::trace!(
                            signal = %decision.signal_id,
                            missing = ?decision.missing_vendor_fields,
                            "Admission refused for signal"
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

                    if let Some(fill) =
                        exchange.execute(&local_intent, event.ts, parent_decision_id)
                    {
                        // Convert fill to Phase 2 FillNotification
                        let notification = quantlaxmi_strategy::FillNotification {
                            ts: fill.ts,
                            symbol: fill.symbol.clone(),
                            side: match fill.side {
                                Side::Buy => quantlaxmi_strategy::Side::Buy,
                                Side::Sell => quantlaxmi_strategy::Side::Sell,
                            },
                            qty_mantissa: (fill.qty
                                * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                            .round() as i64,
                            qty_exponent: market_snapshot.qty_exponent(),
                            price_mantissa: (fill.price
                                * 10f64.powi(-market_snapshot.price_exponent() as i32))
                            .round() as i64,
                            price_exponent: market_snapshot.price_exponent(),
                            fee_mantissa: (fill.fee
                                * 10f64.powi(-market_snapshot.qty_exponent() as i32))
                            .round() as i64,
                            fee_exponent: market_snapshot.qty_exponent(),
                            tag: fill.tag.clone(),
                        };
                        strategy.on_fill(&notification, &ctx);
                    }
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
}
