//! Position Manager: Tracks open trades with TP/SL/time-stop exits.
//!
//! Design:
//! - Each entry creates an OpenTrade with entry details
//! - On each tick, check exit conditions (TP, SL, time-stop)
//! - When exit triggers, emit CompletedTrade with full entry/exit details

use crate::paper::intent::Side;
use crate::paper::state::DecisionMetrics;
use serde::{Deserialize, Serialize};

/// Exit configuration for canary/sniper modes.
#[derive(Debug, Clone)]
pub struct ExitConfig {
    /// Force exit after this many seconds (0 = disabled)
    pub time_stop_seconds: u64,
    /// Take profit in basis points (0 = disabled)
    pub take_profit_bps: f64,
    /// Stop loss in basis points (0 = disabled)
    pub stop_loss_bps: f64,
    /// Fee rate for exit cost calculation
    pub fee_rate: f64,
}

impl Default for ExitConfig {
    fn default() -> Self {
        Self {
            time_stop_seconds: 60,
            take_profit_bps: 8.0,
            stop_loss_bps: 8.0,
            fee_rate: 0.001,
        }
    }
}

impl ExitConfig {
    /// Canary mode exits - extended horizon for realistic TP/SL achievement.
    /// Round-trip taker fees ~20 bps. With 5-min horizon, 40 bps targets are achievable.
    /// Prior run showed 60s horizon had 91% TIME exits - price didn't move enough.
    pub fn canary() -> Self {
        Self {
            time_stop_seconds: 300, // 5 minutes - gives price time to move
            take_profit_bps: 40.0,  // 0.40% - clears ~20 bps fees with margin
            stop_loss_bps: 40.0,    // 0.40% - symmetric
            fee_rate: 0.001,
        }
    }
}

/// Snapshot of features at trade entry (for post-mortem analysis).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrySnapshot {
    pub regime: String,
    pub confidence: Option<f64>,
    pub toxicity: Option<f64>,
    pub imbalance: f64,
    pub fti_level: Option<f64>,
    pub fti_persist: Option<f64>,
    pub edge_est: f64,
    pub edge_req: f64,
    pub spread_bps: f64,
}

impl EntrySnapshot {
    pub fn from_metrics(
        metrics: &DecisionMetrics,
        regime: &str,
        imbalance: f64,
        edge_est: f64,
        edge_req: f64,
        spread_bps: f64,
    ) -> Self {
        Self {
            regime: regime.to_string(),
            confidence: metrics.confidence,
            toxicity: metrics.toxicity,
            imbalance,
            fti_level: metrics.fti_level,
            fti_persist: metrics.fti_persist,
            edge_est,
            edge_req,
            spread_bps,
        }
    }
}

/// An open trade being managed.
#[derive(Debug, Clone)]
pub struct OpenTrade {
    pub trade_id: u64,
    pub entry_tick: u64,
    pub entry_ts_ms: u64,
    pub side: Side,
    pub qty: f64,
    pub entry_price: f64,
    pub entry_fee: f64,
    pub entry_spread_cost: f64,
    pub entry_snapshot: EntrySnapshot,
}

impl OpenTrade {
    /// Calculate current PnL given market price (before exit fee).
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        match self.side {
            Side::Buy => (current_price - self.entry_price) * self.qty,
            Side::Sell => (self.entry_price - current_price) * self.qty,
        }
    }

    /// Calculate current return in basis points.
    pub fn unrealized_bps(&self, current_price: f64) -> f64 {
        let pnl_pct = match self.side {
            Side::Buy => (current_price - self.entry_price) / self.entry_price,
            Side::Sell => (self.entry_price - current_price) / self.entry_price,
        };
        pnl_pct * 10000.0
    }
}

/// Exit reason for logging.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExitReason {
    TakeProfit,
    StopLoss,
    TimeStop,
    Manual,
}

impl ExitReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExitReason::TakeProfit => "TP",
            ExitReason::StopLoss => "SL",
            ExitReason::TimeStop => "TIME",
            ExitReason::Manual => "MANUAL",
        }
    }
}

/// A completed trade (entry + exit) for logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTrade {
    pub trade_id: u64,
    // Entry
    pub entry_tick: u64,
    pub entry_ts_ms: u64,
    pub entry_side: String,
    pub entry_qty: f64,
    pub entry_price: f64,
    pub entry_fee: f64,
    pub entry_spread: f64,
    // Exit
    pub exit_tick: u64,
    pub exit_ts_ms: u64,
    pub exit_price: f64,
    pub exit_fee: f64,
    pub exit_reason: String,
    // PnL
    pub gross_pnl: f64,
    pub fees_total: f64,
    pub net_pnl: f64,
    pub hold_time_ms: u64,
    pub return_bps: f64,
    // Entry snapshot
    pub snapshot: EntrySnapshot,
}

/// Position manager with exit logic.
pub struct PositionManager {
    config: ExitConfig,
    open_trade: Option<OpenTrade>,
    next_trade_id: u64,
    // Counters
    pub accepted_trades: u64,
    pub completed_trades: u64,
    pub gross_pnl_total: f64,
    pub fees_total: f64,
    pub net_pnl_total: f64,
    // For fee_share calculation
    pub gross_abs_total: f64,
}

impl PositionManager {
    pub fn new(config: ExitConfig) -> Self {
        Self {
            config,
            open_trade: None,
            next_trade_id: 1,
            accepted_trades: 0,
            completed_trades: 0,
            gross_pnl_total: 0.0,
            fees_total: 0.0,
            net_pnl_total: 0.0,
            gross_abs_total: 0.0,
        }
    }

    /// Is there currently an open position?
    pub fn has_open_position(&self) -> bool {
        self.open_trade.is_some()
    }

    /// Get current position side (for no_flip check).
    pub fn position_side(&self) -> Option<Side> {
        self.open_trade.as_ref().map(|t| t.side)
    }

    /// Open a new trade. Returns the trade_id.
    #[allow(clippy::too_many_arguments)]
    pub fn open_trade(
        &mut self,
        tick: u64,
        ts_ms: u64,
        side: Side,
        qty: f64,
        fill_price: f64,
        spread_bps: f64,
        snapshot: EntrySnapshot,
    ) -> u64 {
        let notional = qty * fill_price;
        let entry_fee = notional * self.config.fee_rate;
        let spread_cost = (spread_bps / 10000.0) * notional;

        let trade_id = self.next_trade_id;
        self.next_trade_id += 1;

        self.open_trade = Some(OpenTrade {
            trade_id,
            entry_tick: tick,
            entry_ts_ms: ts_ms,
            side,
            qty,
            entry_price: fill_price,
            entry_fee,
            entry_spread_cost: spread_cost,
            entry_snapshot: snapshot,
        });

        self.accepted_trades += 1;
        trade_id
    }

    /// Check exit conditions and close if triggered.
    /// Returns Some(CompletedTrade) if closed, None if still open.
    pub fn check_exit(
        &mut self,
        tick: u64,
        ts_ms: u64,
        best_bid: f64,
        best_ask: f64,
    ) -> Option<CompletedTrade> {
        let trade = self.open_trade.as_ref()?;

        // Calculate exit price (opposite side)
        let exit_price = match trade.side {
            Side::Buy => best_bid,  // Sell to close long
            Side::Sell => best_ask, // Buy to close short
        };

        // Check time stop
        let hold_time_ms = ts_ms.saturating_sub(trade.entry_ts_ms);
        let time_stop_triggered = self.config.time_stop_seconds > 0
            && hold_time_ms >= self.config.time_stop_seconds * 1000;

        // Check TP/SL
        let current_bps = trade.unrealized_bps(exit_price);
        let tp_triggered =
            self.config.take_profit_bps > 0.0 && current_bps >= self.config.take_profit_bps;
        let sl_triggered =
            self.config.stop_loss_bps > 0.0 && current_bps <= -self.config.stop_loss_bps;

        let exit_reason = if tp_triggered {
            Some(ExitReason::TakeProfit)
        } else if sl_triggered {
            Some(ExitReason::StopLoss)
        } else if time_stop_triggered {
            Some(ExitReason::TimeStop)
        } else {
            None
        };

        if let Some(reason) = exit_reason {
            self.close_trade(tick, ts_ms, exit_price, reason)
        } else {
            None
        }
    }

    /// Force close current position (for manual exits).
    pub fn force_close(
        &mut self,
        tick: u64,
        ts_ms: u64,
        exit_price: f64,
    ) -> Option<CompletedTrade> {
        if self.open_trade.is_some() {
            self.close_trade(tick, ts_ms, exit_price, ExitReason::Manual)
        } else {
            None
        }
    }

    fn close_trade(
        &mut self,
        tick: u64,
        ts_ms: u64,
        exit_price: f64,
        reason: ExitReason,
    ) -> Option<CompletedTrade> {
        let trade = self.open_trade.take()?;

        let notional = trade.qty * exit_price;
        let exit_fee = notional * self.config.fee_rate;

        let gross_pnl = trade.unrealized_pnl(exit_price);
        let fees_total = trade.entry_fee + exit_fee;
        let net_pnl = gross_pnl - fees_total;
        let return_bps = trade.unrealized_bps(exit_price);
        let hold_time_ms = ts_ms.saturating_sub(trade.entry_ts_ms);

        // Update counters
        self.completed_trades += 1;
        self.gross_pnl_total += gross_pnl;
        self.fees_total += fees_total;
        self.net_pnl_total += net_pnl;
        self.gross_abs_total += gross_pnl.abs();

        Some(CompletedTrade {
            trade_id: trade.trade_id,
            entry_tick: trade.entry_tick,
            entry_ts_ms: trade.entry_ts_ms,
            entry_side: format!("{:?}", trade.side),
            entry_qty: trade.qty,
            entry_price: trade.entry_price,
            entry_fee: trade.entry_fee,
            entry_spread: trade.entry_spread_cost,
            exit_tick: tick,
            exit_ts_ms: ts_ms,
            exit_price,
            exit_fee,
            exit_reason: reason.as_str().to_string(),
            gross_pnl,
            fees_total,
            net_pnl,
            hold_time_ms,
            return_bps,
            snapshot: trade.entry_snapshot,
        })
    }

    /// Average net PnL per completed trade.
    pub fn avg_net_per_trade(&self) -> f64 {
        if self.completed_trades == 0 {
            0.0
        } else {
            self.net_pnl_total / self.completed_trades as f64
        }
    }

    /// Fee share = fees_total / gross_abs_total
    pub fn fee_share(&self) -> f64 {
        if self.gross_abs_total < 1e-12 {
            0.0
        } else {
            self.fees_total / self.gross_abs_total
        }
    }
}
