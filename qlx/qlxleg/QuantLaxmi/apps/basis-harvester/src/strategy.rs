//! Basis mean-reversion strategy with z-score entry/exit.
//!
//! ## Signal
//! ```text
//! basis_bps = (perp_mid - spot_mid) / spot_mid * 10000
//! z_score = (basis_bps - rolling_mean) / rolling_std
//! ```
//!
//! ## Entry (|z| > 2.0)
//! - z > +2.0: basis too wide → ShortBasis (buy spot, sell perp)
//! - z < -2.0: basis too narrow → LongBasis (sell spot, buy perp)
//!
//! ## Exit (any of)
//! 1. |z| < 0.5 (reverted to mean)
//! 2. P&L > +8 bps (take profit)
//! 3. P&L < -15 bps (stop loss)
//! 4. Hold time > 300s (time stop)

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use quantlaxmi_paper::{
    DecisionMetrics, DecisionType, FillOutcome, Strategy, StrategyDecision, StrategyView,
};
use std::collections::HashMap;
use tracing::info;

use crate::fill_model::FillModelConfig;
use crate::intent::{BasisDirection, BasisIntent, Venue};
use crate::risk::RiskLimits;
use crate::snapshot::BasisSnapshot;
use crate::stats::BasisStats;

/// Strategy configuration.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Rolling window size for mean/std computation
    pub window_size: usize,
    /// Z-score threshold for entry (default 2.0)
    pub entry_z: f64,
    /// Z-score threshold for exit — reversion complete (default 0.5)
    pub exit_z: f64,
    /// Stop loss in bps (default 15.0)
    pub stop_loss_bps: f64,
    /// Take profit in bps (default 8.0)
    pub take_profit_bps: f64,
    /// Maximum hold time in seconds (default 300)
    pub max_hold_secs: u64,
    /// Position size in USDT per trade (default 2000)
    pub position_size_usd: f64,
    /// Maximum concurrent positions (default 5)
    pub max_positions: usize,
    /// Max combined spread in bps to allow entry (default 15.0)
    pub max_spread_bps: f64,
    /// Max quote age in ms (default 3000)
    pub max_quote_age_ms: i64,
    /// Minimum rolling std in bps to allow entry (default 1.0).
    /// Prevents z-score spikes on noise when std is tiny.
    pub min_std_bps: f64,
    /// Fill model config (for fee calculations)
    pub fill_config: FillModelConfig,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            window_size: 300,  // ~30s lookback at 10 obs/s
            entry_z: 2.0,     // proven: triggers on mid-cap volatile pairs
            exit_z: 0.3,      // capture most of reversion
            stop_loss_bps: 30.0,  // must be > combined spread (~20-25 bps)
            take_profit_bps: 20.0, // lock in winners
            max_hold_secs: 300,   // 5 min: balance between patience and capital lockup
            position_size_usd: 1_500.0, // moderate size
            max_positions: 6,     // diversified
            max_spread_bps: 25.0, // mid-cap pairs
            max_quote_age_ms: 5_000, // 5s: mid-cap feed latency
            min_std_bps: 1.0,     // minimal filter, let G4 handle profitability
            fill_config: FillModelConfig::default(),
        }
    }
}

/// Active trade state tracked by the strategy.
#[derive(Debug, Clone)]
struct ActiveTrade {
    direction: BasisDirection,
    entry_basis_bps: f64,
    entry_z: f64,
    entry_ts: DateTime<Utc>,
    spot_qty: f64,
    notional_usd: f64,
}

/// Pending entry data stored between on_snapshot and on_outcome.
#[derive(Debug, Clone)]
struct PendingEntry {
    direction: BasisDirection,
    entry_basis_bps: f64,
    entry_z: f64,
    spot_qty: f64,
    notional_usd: f64,
}

/// Closed trade record for statistics.
#[derive(Debug, Clone)]
struct ClosedTrade {
    pnl_bps: f64,
    hold_secs: u64,
}

/// Basis mean-reversion strategy.
pub struct BasisMeanRevStrategy {
    pub config: StrategyConfig,
    pub risk: RiskLimits,
    stats: BasisStats,
    /// Active trades: symbol → trade state
    active_trades: HashMap<String, ActiveTrade>,
    /// Pending entries: symbol → pending (consumed in on_outcome)
    pending_entries: HashMap<String, PendingEntry>,
    /// Closed trade history (for statistics)
    closed_trades: Vec<ClosedTrade>,
    /// Allocated capital in active trades
    allocated_usd: f64,
    initial_capital: f64,
    /// Channel to tell the feed which symbols have open positions.
    pinned_tx: Option<tokio::sync::watch::Sender<Vec<String>>>,
}

impl BasisMeanRevStrategy {
    pub fn new(config: StrategyConfig, initial_capital: f64) -> Self {
        let risk = RiskLimits {
            max_total_exposure_usd: initial_capital,
            max_per_pair_usd: config.position_size_usd,
            max_drawdown_pct: 5.0,
        };
        let stats = BasisStats::new(config.window_size);
        Self {
            config,
            risk,
            stats,
            active_trades: HashMap::new(),
            pending_entries: HashMap::new(),
            closed_trades: Vec::new(),
            allocated_usd: 0.0,
            initial_capital,
            pinned_tx: None,
        }
    }

    /// Set the watch channel for broadcasting pinned symbols to the feed.
    pub fn with_pinned_channel(mut self, tx: tokio::sync::watch::Sender<Vec<String>>) -> Self {
        self.pinned_tx = Some(tx);
        self
    }

    fn broadcast_pinned(&self) {
        if let Some(ref tx) = self.pinned_tx {
            let pinned: Vec<String> = self.active_trades.keys().cloned().collect();
            let _ = tx.send(pinned);
        }
    }

    fn available_capital(&self) -> f64 {
        self.initial_capital - self.allocated_usd
    }

    /// Check all entry gates for a symbol.
    fn check_entry_gates(
        &self,
        sym: &str,
        state: &crate::snapshot::SymbolState,
        z: f64,
        now: DateTime<Utc>,
    ) -> Result<BasisDirection, String> {
        // Gate 1: |z-score| > entry_z
        if z.abs() < self.config.entry_z {
            return Err(format!("G1: |z|={:.2} < {:.1}", z.abs(), self.config.entry_z));
        }

        // Gate 1b: Rolling std large enough (prevents z-score spikes on noise)
        let rolling_std = self.stats.std(sym);
        if rolling_std < self.config.min_std_bps {
            return Err(format!(
                "G1b: std {:.2}bps < {:.1}bps (noise)",
                rolling_std, self.config.min_std_bps
            ));
        }

        // Gate 2: Combined spread < threshold
        let spread = state.combined_spread_bps();
        if spread > self.config.max_spread_bps {
            return Err(format!("G2: spread {:.1}bps > {:.1}bps", spread, self.config.max_spread_bps));
        }

        // Gate 3: Quote age < threshold
        let age = state.max_quote_age_ms(now);
        if age > self.config.max_quote_age_ms {
            return Err(format!("G3: quote age {}ms > {}ms", age, self.config.max_quote_age_ms));
        }

        // Gate 4: Expected move > round-trip fees
        let rt_cost_bps = self.config.fill_config.round_trip_cost_bps();
        let expected_move_bps = (z.abs() - self.config.exit_z) * self.stats.std(sym);
        if expected_move_bps < rt_cost_bps {
            return Err(format!(
                "G4: expected {:.1}bps < cost {:.1}bps",
                expected_move_bps, rt_cost_bps
            ));
        }

        // Gate 5: Not already in position
        if self.active_trades.contains_key(sym) {
            return Err(format!("G5: already in {}", sym));
        }

        // Gate 6: Max positions not reached
        if self.active_trades.len() >= self.config.max_positions {
            return Err(format!(
                "G6: {} >= max {}",
                self.active_trades.len(),
                self.config.max_positions
            ));
        }

        // Gate 7: Capital available
        if self.available_capital() < self.config.position_size_usd {
            return Err(format!(
                "G7: capital {:.0} < {:.0}",
                self.available_capital(),
                self.config.position_size_usd
            ));
        }

        // Gate 8: Risk limits
        if !self.risk.can_add_exposure(self.allocated_usd, self.config.position_size_usd) {
            return Err("G8: exposure limit".into());
        }

        // Determine direction from z-score sign
        let direction = if z > 0.0 {
            BasisDirection::ShortBasis // basis too wide → expect narrowing
        } else {
            BasisDirection::LongBasis // basis too narrow → expect widening
        };

        Ok(direction)
    }

    /// Check exit conditions for an active trade.
    fn check_exit(
        &self,
        _sym: &str,
        state: &crate::snapshot::SymbolState,
        trade: &ActiveTrade,
        z: f64,
        now: DateTime<Utc>,
    ) -> Option<String> {
        // Current basis P&L in bps
        let current_basis = state.basis_bps();
        let pnl_bps = match trade.direction {
            // ShortBasis entered when basis was wide, profits when it narrows
            BasisDirection::ShortBasis => trade.entry_basis_bps - current_basis,
            // LongBasis entered when basis was narrow, profits when it widens
            BasisDirection::LongBasis => current_basis - trade.entry_basis_bps,
        };

        // Exit 1: Z-score reverted to mean
        if z.abs() < self.config.exit_z {
            return Some(format!(
                "E1: |z|={:.2} < {:.1} (reversion) pnl={:.1}bps",
                z.abs(),
                self.config.exit_z,
                pnl_bps
            ));
        }

        // Exit 2: Take profit
        if pnl_bps > self.config.take_profit_bps {
            return Some(format!("E2: pnl {:.1}bps > +{:.1}bps (TP)", pnl_bps, self.config.take_profit_bps));
        }

        // Exit 3: Stop loss
        if pnl_bps < -self.config.stop_loss_bps {
            return Some(format!("E3: pnl {:.1}bps < -{:.1}bps (SL)", pnl_bps, self.config.stop_loss_bps));
        }

        // Exit 4: Time stop
        let hold_secs = (now - trade.entry_ts).num_seconds() as u64;
        if hold_secs > self.config.max_hold_secs {
            return Some(format!(
                "E4: hold {}s > {}s (time) pnl={:.1}bps",
                hold_secs, self.config.max_hold_secs, pnl_bps
            ));
        }

        None
    }

    /// Compute basis P&L for a trade.
    fn trade_pnl_bps(trade: &ActiveTrade, current_basis: f64) -> f64 {
        match trade.direction {
            BasisDirection::ShortBasis => trade.entry_basis_bps - current_basis,
            BasisDirection::LongBasis => current_basis - trade.entry_basis_bps,
        }
    }

    /// Get trade statistics for TUI display.
    pub fn trade_stats(&self) -> crate::state::TradeStatistics {
        let total = self.closed_trades.len() as u64;
        let wins = self.closed_trades.iter().filter(|t| t.pnl_bps > 0.0).count() as u64;
        let losses = total - wins;
        let avg_hold = if total > 0 {
            self.closed_trades.iter().map(|t| t.hold_secs as f64).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let total_pnl: f64 = self.closed_trades.iter().map(|t| t.pnl_bps).sum();
        let avg_pnl = if total > 0 { total_pnl / total as f64 } else { 0.0 };

        crate::state::TradeStatistics {
            total_trades: total,
            wins,
            losses,
            avg_hold_secs: avg_hold,
            avg_pnl_bps: avg_pnl,
            total_pnl_bps: total_pnl,
        }
    }

    /// Build z-score displays for TUI.
    pub fn z_score_displays(&self, snapshot: &BasisSnapshot) -> Vec<crate::state::ZScoreDisplay> {
        let mut displays: Vec<crate::state::ZScoreDisplay> = snapshot
            .symbols
            .iter()
            .filter(|(_, s)| s.has_quotes())
            .map(|(sym, s)| {
                let basis = s.basis_bps();
                crate::state::ZScoreDisplay {
                    symbol: sym.clone(),
                    basis_bps: basis,
                    z_score: self.stats.z_score(sym, basis),
                    rolling_std: self.stats.std(sym),
                    rolling_mean: self.stats.mean(sym),
                    observations: self.stats.count(sym),
                }
            })
            .collect();
        displays.sort_by(|a, b| {
            b.z_score
                .abs()
                .partial_cmp(&a.z_score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        displays
    }

    /// Build active trade displays for TUI.
    pub fn trade_displays(&self, snapshot: &BasisSnapshot) -> Vec<crate::state::TradeDisplay> {
        let now = Utc::now();
        self.active_trades
            .iter()
            .map(|(sym, trade)| {
                let current_basis = snapshot
                    .symbols
                    .get(sym)
                    .map(|s| s.basis_bps())
                    .unwrap_or(trade.entry_basis_bps);
                let current_z = self.stats.z_score(sym, current_basis);
                let pnl_bps = Self::trade_pnl_bps(trade, current_basis);
                let hold_secs = (now - trade.entry_ts).num_seconds().max(0) as u64;
                crate::state::TradeDisplay {
                    symbol: sym.clone(),
                    direction: trade.direction,
                    entry_basis_bps: trade.entry_basis_bps,
                    entry_z: trade.entry_z,
                    current_basis_bps: current_basis,
                    current_z,
                    pnl_bps,
                    hold_secs,
                }
            })
            .collect()
    }
}

#[async_trait]
impl Strategy<BasisSnapshot, BasisIntent> for BasisMeanRevStrategy {
    async fn on_snapshot(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &BasisSnapshot,
    ) -> Result<StrategyDecision<BasisIntent>> {
        // Update rolling stats for all symbols
        for (sym, state) in &snapshot.symbols {
            if state.has_quotes() {
                self.stats.push(sym, state.basis_bps());
            }
        }

        let mut intents = Vec::new();
        let mut reasons = Vec::new();

        // Check exits first (higher priority)
        let active_syms: Vec<String> = self.active_trades.keys().cloned().collect();
        for sym in &active_syms {
            if let Some(state) = snapshot.symbols.get(sym) {
                let basis = state.basis_bps();
                let z = self.stats.z_score(sym, basis);
                let trade = &self.active_trades[sym];

                if let Some(exit_reason) = self.check_exit(sym, state, trade, z, ts) {
                    // Exit: use exact entry quantities
                    let spot_qty = trade.spot_qty;
                    let direction = trade.direction;

                    intents.extend(BasisIntent::exit_pair(
                        sym,
                        spot_qty,
                        state.spot_mid(),
                        state.perp_mid(),
                        direction,
                    ));
                    reasons.push(format!("EXIT {}: {}", sym, exit_reason));
                }
            }
        }

        // Check entries (only if no exits pending — one action per tick)
        if intents.is_empty() {
            // Build candidates sorted by |z-score| descending
            let mut candidates: Vec<(&String, &crate::snapshot::SymbolState, f64)> = snapshot
                .symbols
                .iter()
                .filter(|(sym, s)| s.has_quotes() && self.stats.is_ready(sym))
                .map(|(sym, s)| {
                    let basis = s.basis_bps();
                    let z = self.stats.z_score(sym, basis);
                    (sym, s, z)
                })
                .collect();

            candidates.sort_by(|a, b| {
                b.2.abs()
                    .partial_cmp(&a.2.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (sym, state, z) in &candidates {
                match self.check_entry_gates(sym, state, *z, ts) {
                    Ok(direction) => {
                        let size_usd = self.config.position_size_usd;
                        let spot_qty = size_usd / state.spot_ask;

                        intents.extend(BasisIntent::entry_pair(
                            sym,
                            spot_qty,
                            state.spot_ask,
                            state.perp_bid,
                            direction,
                        ));

                        let basis = state.basis_bps();
                        self.pending_entries.insert(
                            sym.to_string(),
                            PendingEntry {
                                direction,
                                entry_basis_bps: basis,
                                entry_z: *z,
                                spot_qty,
                                notional_usd: size_usd,
                            },
                        );

                        reasons.push(format!(
                            "ENTER {} {:?} z={:.2} basis={:.1}bps size=${:.0}",
                            sym, direction, z, basis, size_usd
                        ));
                        break; // one entry per tick
                    }
                    Err(reason) => {
                        reasons.push(format!("{}: {}", sym, reason));
                    }
                }
            }
        }

        let accepted = !intents.is_empty();
        let decision_type = if accepted {
            DecisionType::Accept
        } else {
            DecisionType::Hold
        };

        let reason = if reasons.is_empty() {
            "No symbols with data".into()
        } else {
            reasons.first().cloned().unwrap_or_default()
        };

        if accepted {
            info!(target: "basis.strategy", "{}", reason);
        } else {
            static HOLD_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = HOLD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(100) {
                info!(target: "basis.strategy", hold_count = n, "{}", reason);
            }
        }

        let strategy_view = Some(StrategyView {
            name: "BasisMeanRev".into(),
            underlying: String::new(),
            decision_type: format!("{}", decision_type),
            decision_reason: reason.clone(),
            ..Default::default()
        });

        let metrics = if accepted {
            Some(DecisionMetrics {
                strategy_name: "BasisMeanRev".into(),
                ..Default::default()
            })
        } else {
            None
        };

        Ok(StrategyDecision {
            ts,
            accepted,
            reason,
            intents,
            decision_type,
            metrics,
            strategy_view,
        })
    }

    async fn on_outcome(
        &mut self,
        decision: &StrategyDecision<BasisIntent>,
        outcome: FillOutcome,
    ) -> Result<()> {
        if outcome == FillOutcome::AllFilled {
            for intent in &decision.intents {
                // Detect entries by checking pending_entries
                if intent.venue == Venue::Spot {
                    if let Some(pending) = self.pending_entries.remove(&intent.symbol) {
                        // Entry filled
                        self.active_trades.insert(
                            intent.symbol.clone(),
                            ActiveTrade {
                                direction: pending.direction,
                                entry_basis_bps: pending.entry_basis_bps,
                                entry_z: pending.entry_z,
                                entry_ts: decision.ts,
                                spot_qty: pending.spot_qty,
                                notional_usd: pending.notional_usd,
                            },
                        );
                        self.allocated_usd += pending.notional_usd;
                    } else if let Some(trade) = self.active_trades.remove(&intent.symbol) {
                        // Exit filled — record closed trade
                        self.allocated_usd -= trade.notional_usd;
                        let hold_secs =
                            (decision.ts - trade.entry_ts).num_seconds().max(0) as u64;
                        // Approximate P&L from basis move
                        // In reality the fill prices determine P&L, but for stats
                        // we track the basis move that triggered the trade
                        let exit_reason = &decision.reason;
                        let pnl_bps = if exit_reason.contains("pnl=") {
                            // Parse pnl from exit reason string
                            exit_reason
                                .split("pnl=")
                                .nth(1)
                                .and_then(|s| s.split("bps").next())
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.0)
                        } else if exit_reason.contains("pnl ") {
                            exit_reason
                                .split("pnl ")
                                .nth(1)
                                .and_then(|s| s.split("bps").next())
                                .and_then(|s| s.parse::<f64>().ok())
                                .unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        self.closed_trades.push(ClosedTrade { pnl_bps, hold_secs });
                    }
                }
            }
            self.broadcast_pinned();
        }
        Ok(())
    }
}
