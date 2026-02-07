//! Funding arbitrage strategy with tuned entry/exit gates.
//!
//! ## Entry gates (all must pass):
//! 1. Funding rate > 5 bps (cost-covering threshold)
//! 2. Spread acceptable (combined < max_spread_bps)
//! 3. Quotes fresh (age < max_quote_age_ms)
//! 4. Capital available (unused capital > min_position_usd)
//! 5. Edge > fees (annualized funding > round-trip cost)
//! 6. Not already in position for this symbol
//! 7. Max positions not reached
//! 8. Basis not diverged (|basis| < max_basis_bps)
//!
//! ## Exit conditions:
//! 0. Force exit if rate goes negative (paying funding)
//! 1. Funding rate drops below exit threshold AND (breakeven OR min hold elapsed)
//! 2. Basis diverges beyond limit
//! 3. Max settlements reached
//!
//! ## Sizing (Kelly-inspired):
//! Position size scales with edge: more funding → bigger position.
//! Capped at max_position_usd per pair.
//!
//! ## Settlement proximity:
//! Candidates near next settlement get priority (funding payment imminent).

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use quantlaxmi_paper::{
    DecisionMetrics, DecisionType, FillOutcome, Strategy, StrategyDecision, StrategyView,
};
use std::collections::{HashMap, HashSet};
use tracing::info;

use crate::fill_model::FillModelConfig;
use crate::intent::FundingArbIntent;
use crate::portfolio::Portfolio;
use crate::risk::RiskLimits;
use crate::snapshot::FundingArbSnapshot;

/// Strategy configuration.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Minimum funding rate to enter (default 0.0005 = 5 bps)
    pub entry_rate_threshold: f64,
    /// Exit when funding rate drops below this (AND breakeven or min-hold met)
    pub exit_rate_threshold: f64,
    /// Force exit when rate goes negative
    pub force_exit_rate: f64,
    /// Max combined spread in bps (spot spread + perp spread)
    pub max_spread_bps: f64,
    /// Max quote age in ms
    pub max_quote_age_ms: i64,
    /// Minimum position size in USDT
    pub min_position_usd: f64,
    /// Base position size in USDT (Kelly scales this up/down)
    pub base_position_usd: f64,
    /// Maximum position size in USDT per pair
    pub max_position_usd: f64,
    /// Max basis divergence in bps before exit
    pub max_basis_bps: f64,
    /// Max funding settlements to collect before exit (0 = unlimited)
    pub max_settlements: u32,
    /// Max number of concurrent positions
    pub max_positions: usize,
    /// Minimum hold duration in hours before soft exit allowed
    pub min_hold_hours: f64,
    /// Settlement proximity window in minutes (prefer entries within this window)
    pub settlement_window_min: i64,
    /// Fill model config (for fee calculations)
    pub fill_config: FillModelConfig,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            entry_rate_threshold: 0.00015, // 1.5 bps — aggressive, enters above baseline
            exit_rate_threshold: 0.00005, // 0.5 bps — exit only when edge truly gone
            force_exit_rate: -0.0001,     // force exit if rate goes negative
            max_spread_bps: 30.0,
            max_quote_age_ms: 5_000,
            min_position_usd: 100.0,
            base_position_usd: 1_500.0,
            max_position_usd: 5_000.0,
            max_basis_bps: 300.0,
            max_settlements: 0, // unlimited
            max_positions: 10,
            min_hold_hours: 2.0,       // shorter hold — rotate capital faster
            settlement_window_min: 60, // wider window — more settlement captures
            fill_config: FillModelConfig::default(),
        }
    }
}

/// Pending entry info stored between on_snapshot and on_outcome.
#[derive(Debug, Clone)]
struct PendingEntry {
    notional_usd: f64,
    spot_qty: f64,
    perp_qty: f64,
    entry_cost_usd: f64,
}

/// Funding arbitrage strategy.
pub struct FundingArbStrategy {
    pub config: StrategyConfig,
    pub portfolio: Portfolio,
    pub risk: RiskLimits,
    /// Symbols with active positions
    active_symbols: HashSet<String>,
    /// Last seen next_funding_time_ms per symbol (for settlement detection)
    last_funding_times: HashMap<String, i64>,
    /// Pending entry data: symbol → (notional, cost) — consumed in on_outcome
    pending_entries: HashMap<String, PendingEntry>,
    /// Channel to tell the feed which symbols have open positions (must not rotate out).
    pinned_tx: Option<tokio::sync::watch::Sender<Vec<String>>>,
}

impl FundingArbStrategy {
    pub fn new(config: StrategyConfig, initial_capital: f64) -> Self {
        let risk = RiskLimits {
            max_total_exposure_usd: initial_capital * 0.8,
            max_per_pair_usd: config.max_position_usd * 2.0,
            max_drawdown_pct: 5.0,
            max_basis_divergence_bps: config.max_basis_bps,
        };
        Self {
            config,
            portfolio: Portfolio::new(initial_capital),
            risk,
            active_symbols: HashSet::new(),
            last_funding_times: HashMap::new(),
            pending_entries: HashMap::new(),
            pinned_tx: None,
        }
    }

    /// Set the watch channel for broadcasting pinned symbols to the feed.
    pub fn with_pinned_channel(mut self, tx: tokio::sync::watch::Sender<Vec<String>>) -> Self {
        self.pinned_tx = Some(tx);
        self
    }

    /// Broadcast current active symbols to the feed so they don't get rotated out.
    fn broadcast_pinned(&self) {
        if let Some(ref tx) = self.pinned_tx {
            let pinned: Vec<String> = self.active_symbols.iter().cloned().collect();
            let _ = tx.send(pinned);
        }
    }

    /// Detect funding settlements by watching next_funding_time_ms transitions.
    fn detect_settlements(&mut self, snapshot: &FundingArbSnapshot) {
        for (sym, state) in &snapshot.symbols {
            if state.next_funding_time_ms == 0 {
                continue;
            }
            let last = self.last_funding_times.get(sym).copied().unwrap_or(0);
            if last != 0 && state.next_funding_time_ms != last {
                // Settlement occurred — record funding payment if we hold this symbol
                if self.active_symbols.contains(sym) {
                    let rate = state.funding_rate;
                    let notional = self
                        .portfolio
                        .position_notional(sym)
                        .unwrap_or(0.0);
                    // Short perp receives funding when rate > 0
                    let payment = notional * rate;
                    self.portfolio.record_funding(sym, payment);
                    info!(
                        target: "harvester.funding",
                        symbol = sym,
                        rate_bps = format!("{:.2}", rate * 10_000.0),
                        payment_usd = format!("{:.4}", payment),
                        "Settlement collected"
                    );
                }
            }
            self.last_funding_times
                .insert(sym.clone(), state.next_funding_time_ms);
        }
    }

    /// Kelly-inspired position sizing: scale with edge.
    fn compute_position_size(&self, state: &crate::snapshot::SymbolState) -> f64 {
        let annualized = state.annualized_pct();
        let cost = self.config.fill_config.round_trip_cost_pct();

        // edge_multiple: how many times does annualized funding cover round-trip cost?
        let edge_multiple = if cost > 0.0 { annualized / cost } else { 1.0 };

        // Scale: base_size * clamp(edge_multiple / 2, 0.5, 2.0)
        //   edge_multiple=2 → 1x base
        //   edge_multiple=4 → 2x base
        //   edge_multiple=1 → 0.5x base
        let scale = (edge_multiple / 2.0).clamp(0.5, 2.0);
        let size = self.config.base_position_usd * scale;

        size.clamp(self.config.min_position_usd, self.config.max_position_usd)
            .min(self.portfolio.available_capital())
    }

    /// Minutes until next funding settlement.
    fn minutes_to_settlement(state: &crate::snapshot::SymbolState, now: DateTime<Utc>) -> i64 {
        if state.next_funding_time_ms == 0 {
            return i64::MAX;
        }
        let now_ms = now.timestamp_millis();
        (state.next_funding_time_ms - now_ms) / 60_000
    }

    /// Check all entry gates for a symbol.
    fn check_entry_gates(
        &self,
        sym: &str,
        state: &crate::snapshot::SymbolState,
        now: DateTime<Utc>,
    ) -> Result<(), String> {
        // Gate 1: Funding rate > threshold (5 bps default)
        if state.funding_rate < self.config.entry_rate_threshold {
            return Err(format!(
                "G1: rate {:.2}bps < {:.2}bps",
                state.funding_rate * 10_000.0,
                self.config.entry_rate_threshold * 10_000.0
            ));
        }

        // Gate 2: Spread acceptable
        let spot_spread_bps = if state.spot_mid() > 0.0 {
            (state.spot_ask - state.spot_bid) / state.spot_mid() * 10_000.0
        } else {
            f64::MAX
        };
        let perp_spread_bps = if state.perp_mid() > 0.0 {
            (state.perp_ask - state.perp_bid) / state.perp_mid() * 10_000.0
        } else {
            f64::MAX
        };
        let combined_spread = spot_spread_bps + perp_spread_bps;
        if combined_spread > self.config.max_spread_bps {
            return Err(format!(
                "G2: spread {:.1}bps > {:.1}bps",
                combined_spread, self.config.max_spread_bps
            ));
        }

        // Gate 3: Quotes fresh
        let age = state.max_quote_age_ms(now);
        if age > self.config.max_quote_age_ms {
            return Err(format!(
                "G3: quote age {}ms > {}ms",
                age, self.config.max_quote_age_ms
            ));
        }

        // Gate 4: Capital available
        if self.portfolio.available_capital() < self.config.min_position_usd {
            return Err(format!(
                "G4: capital {:.0} < {:.0}",
                self.portfolio.available_capital(),
                self.config.min_position_usd
            ));
        }

        // Gate 5: Edge > fees (annualized funding must exceed round-trip cost)
        let annualized = state.annualized_pct();
        let round_trip_cost_pct = self.config.fill_config.round_trip_cost_pct();
        if annualized < round_trip_cost_pct {
            return Err(format!(
                "G5: ann {:.2}% < cost {:.2}%",
                annualized, round_trip_cost_pct
            ));
        }

        // Gate 6: Not already in position
        if self.active_symbols.contains(sym) {
            return Err(format!("G6: already in {}", sym));
        }

        // Gate 7: Max positions not reached
        if self.active_symbols.len() >= self.config.max_positions {
            return Err(format!(
                "G7: {} positions >= max {}",
                self.active_symbols.len(),
                self.config.max_positions
            ));
        }

        // Gate 8: Basis not diverged
        let basis = state.basis_bps().abs();
        if basis > self.config.max_basis_bps {
            return Err(format!(
                "G8: |basis| {:.1}bps > {:.1}bps",
                basis, self.config.max_basis_bps
            ));
        }

        Ok(())
    }

    /// Check exit conditions for an active position.
    fn check_exit(
        &self,
        sym: &str,
        state: &crate::snapshot::SymbolState,
        now: DateTime<Utc>,
    ) -> Option<String> {
        // Exit 0: Force exit if rate went negative (we'd be PAYING funding)
        if state.funding_rate < self.config.force_exit_rate {
            return Some(format!(
                "E0: rate {:.2}bps NEGATIVE (FORCE)",
                state.funding_rate * 10_000.0,
            ));
        }

        // Exit 2: Basis diverged (risk limit — always enforced)
        let basis = state.basis_bps().abs();
        if basis > self.risk.max_basis_divergence_bps {
            return Some(format!(
                "E2: |basis| {:.1}bps > {:.1}bps",
                basis, self.risk.max_basis_divergence_bps
            ));
        }

        // Exit 3: Max settlements reached
        if self.config.max_settlements > 0
            && let Some(pos) = self.portfolio.get_position(sym)
            && pos.settlements >= self.config.max_settlements
        {
            return Some(format!(
                "E3: {} settlements >= max {}",
                pos.settlements, self.config.max_settlements
            ));
        }

        // Exit 1: Funding rate dropped — but only if breakeven OR min hold elapsed
        if state.funding_rate < self.config.exit_rate_threshold
            && let Some(pos) = self.portfolio.get_position(sym)
        {
            let held_hours = pos.hold_hours(now);
            if pos.is_breakeven() {
                return Some(format!(
                    "E1: rate {:.2}bps < {:.2}bps (breakeven reached)",
                    state.funding_rate * 10_000.0,
                    self.config.exit_rate_threshold * 10_000.0
                ));
            }
            if held_hours >= self.config.min_hold_hours {
                return Some(format!(
                    "E1: rate {:.2}bps < {:.2}bps (held {:.1}h >= {:.1}h min)",
                    state.funding_rate * 10_000.0,
                    self.config.exit_rate_threshold * 10_000.0,
                    held_hours,
                    self.config.min_hold_hours
                ));
            }
            // Rate is low but not breakeven and not past min hold — keep holding
        }

        None
    }
}

#[async_trait]
impl Strategy<FundingArbSnapshot, FundingArbIntent> for FundingArbStrategy {
    async fn on_snapshot(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &FundingArbSnapshot,
    ) -> Result<StrategyDecision<FundingArbIntent>> {
        // Detect funding settlements
        self.detect_settlements(snapshot);

        let mut intents = Vec::new();
        let mut reasons = Vec::new();

        // Check exits first (higher priority)
        let active: Vec<String> = self.active_symbols.iter().cloned().collect();
        for sym in &active {
            if let Some(state) = snapshot.symbols.get(sym)
                && let Some(exit_reason) = self.check_exit(sym, state, ts)
            {
                // Exit: use EXACT entry quantities (not recalculated from notional/price)
                // Recalculating from notional/current_price causes qty overshoot when price moves
                if let Some(pos) = self.portfolio.get_position(sym) {
                    let spot_qty = pos.spot_qty;
                    let perp_qty = pos.perp_qty;
                    if spot_qty > 0.0 {
                        intents.push(FundingArbIntent {
                            symbol: sym.to_string(),
                            venue: crate::intent::Venue::Spot,
                            side: crate::intent::Side::Sell,
                            qty: spot_qty,
                        });
                        intents.push(FundingArbIntent {
                            symbol: sym.to_string(),
                            venue: crate::intent::Venue::Perp,
                            side: crate::intent::Side::Buy,
                            qty: perp_qty,
                        });
                        reasons.push(format!("EXIT {}: {}", sym, exit_reason));
                    }
                }
            }
        }

        // Check entries (only if no exits pending)
        if intents.is_empty() {
            let with_funding = snapshot
                .symbols
                .values()
                .filter(|s| s.has_funding())
                .count();

            // Build candidates with edge score + settlement proximity for sorting
            let mut candidates: Vec<(&String, &crate::snapshot::SymbolState, f64)> = snapshot
                .symbols
                .iter()
                .filter(|(_, s)| s.has_quotes() && s.has_funding())
                .map(|(sym, s)| {
                    let rate_bps = s.funding_rate * 10_000.0;
                    let basis_bps = s.basis_bps().abs();
                    let edge_score = if rate_bps > 0.0 {
                        rate_bps / basis_bps.max(1.0)
                    } else {
                        0.0
                    };
                    // Boost score if near settlement (within window)
                    let min_to_settle = Self::minutes_to_settlement(s, ts);
                    let proximity_boost = if min_to_settle > 0
                        && min_to_settle <= self.config.settlement_window_min
                    {
                        1.5 // 50% boost for imminent settlement
                    } else {
                        1.0
                    };
                    (sym, s, edge_score * proximity_boost)
                })
                .collect();

            if candidates.is_empty() && with_funding == 0 {
                reasons.push("No symbols with funding data yet".into());
            }

            // Sort by score descending (best risk-adjusted + settlement-boosted first)
            candidates.sort_by(|a, b| {
                b.2.partial_cmp(&a.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (sym, state, _score) in &candidates {
                match self.check_entry_gates(sym, state, ts) {
                    Ok(()) => {
                        let size_usd = self.compute_position_size(state);
                        let spot_qty = size_usd / state.spot_ask;
                        let perp_qty = if state.perp_bid > 0.0 {
                            spot_qty * state.spot_ask / state.perp_bid
                        } else {
                            spot_qty
                        };

                        intents.extend(FundingArbIntent::entry_pair(
                            sym,
                            spot_qty,
                            state.spot_ask,
                            state.perp_bid,
                        ));

                        // Store pending entry for on_outcome (with exact quantities)
                        let entry_cost = size_usd
                            * (self
                                .config
                                .fill_config
                                .fee_rate(crate::intent::Venue::Spot)
                                + self
                                    .config
                                    .fill_config
                                    .fee_rate(crate::intent::Venue::Perp));
                        self.pending_entries.insert(
                            sym.to_string(),
                            PendingEntry {
                                notional_usd: size_usd,
                                spot_qty,
                                perp_qty,
                                entry_cost_usd: entry_cost,
                            },
                        );

                        let min_to_settle = Self::minutes_to_settlement(state, ts);
                        reasons.push(format!(
                            "ENTER {}: rate={:.1}bps ann={:.1}% size=${:.0} settle={}min",
                            sym,
                            state.funding_rate * 10_000.0,
                            state.annualized_pct(),
                            size_usd,
                            if min_to_settle < i64::MAX {
                                min_to_settle
                            } else {
                                -1
                            },
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

        // Log decision
        if accepted {
            info!(target: "harvester.strategy", "{}", reason);
        } else {
            static HOLD_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = HOLD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(50) {
                info!(target: "harvester.strategy", hold_count = n, "{}", reason);
            }
        }

        let strategy_view = Some(StrategyView {
            name: "FundingArb".into(),
            underlying: String::new(),
            decision_type: format!("{}", decision_type),
            decision_reason: reason.clone(),
            ..Default::default()
        });

        let metrics = if accepted {
            Some(DecisionMetrics {
                strategy_name: "FundingArb".into(),
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
        decision: &StrategyDecision<FundingArbIntent>,
        outcome: FillOutcome,
    ) -> Result<()> {
        if outcome == FillOutcome::AllFilled {
            for intent in &decision.intents {
                if intent.venue == crate::intent::Venue::Spot
                    && intent.side == crate::intent::Side::Buy
                {
                    // Entry: use pending entry data for notional + cost + exact quantities
                    let pending = self
                        .pending_entries
                        .remove(&intent.symbol)
                        .unwrap_or(PendingEntry {
                            notional_usd: self.config.base_position_usd,
                            spot_qty: intent.qty,
                            perp_qty: intent.qty,
                            entry_cost_usd: self.config.base_position_usd * 0.002,
                        });

                    self.active_symbols.insert(intent.symbol.clone());
                    self.portfolio.add_position(
                        &intent.symbol,
                        pending.notional_usd,
                        pending.spot_qty,
                        pending.perp_qty,
                        pending.entry_cost_usd,
                        decision.ts,
                    );
                } else if intent.venue == crate::intent::Venue::Spot
                    && intent.side == crate::intent::Side::Sell
                {
                    // Exit: remove position
                    self.active_symbols.remove(&intent.symbol);
                    self.portfolio.remove_position(&intent.symbol);
                }
            }
            // Broadcast updated pinned symbols to feed
            self.broadcast_pinned();
        }
        Ok(())
    }
}
