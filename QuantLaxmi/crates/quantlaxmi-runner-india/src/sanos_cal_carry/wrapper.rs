//! SANOS Strategy Wrapper
//!
//! Implements `Strategy<OptionsSnapshot, TradeIntent>` for the PaperEngine.
//! Wraps `SanosCalendarCarryAdapter` and handles:
//! - Margin checking via `MarginGate`
//! - Token resolution from snapshot
//! - TUI state broadcasting via `watch` channel

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, watch};
use tracing::{info, warn};

use quantlaxmi_options::strategies::{GateCheckResult, StrategyDecision as SanosStrategyDecision};
use quantlaxmi_paper::Strategy;

use crate::paper::fees_india::{InstrumentKind, Side};
use crate::paper::mapping::InstrumentMap;
use crate::paper::margin::{MarginGate, MarginOrderParams, MarginRejectReason};
use crate::paper::snapshot::{OptionsSnapshot, Right};
use crate::paper::strategy_adapter::{IntentTag, TradeIntent};

use super::adapter::SanosCalendarCarryAdapter;
use super::tui_state::{SanosGateView, SanosTuiState};

/// Pending margin action (deferred until AllFilled).
#[derive(Debug, Clone, Copy)]
enum PendingMarginAction {
    Reserve(f64),
    Release,
}

/// SANOS strategy wrapper implementing the PaperEngine Strategy trait.
pub struct SanosStrategyWrapper {
    /// Core adapter (SANOS calibration + strategy evaluation).
    adapter: SanosCalendarCarryAdapter,
    /// Margin gate for SPAN margin checking.
    margin_gate: Option<Arc<Mutex<MarginGate>>>,
    /// Instrument map for canonical symbol lookup.
    instrument_map: Option<InstrumentMap>,
    /// Lot size for the underlying.
    lot_size: i32,
    /// Last margin reserved.
    last_margin_reserved: f64,
    /// Pending margin action.
    pending_margin_action: Option<PendingMarginAction>,
    /// TUI state sender.
    tui_tx: Option<watch::Sender<SanosTuiState>>,
    /// Token resolution cache: (expiry, strike, right) → instrument_token.
    token_cache: HashMap<(String, i32, Right), u32>,
    /// Whether we have an active position.
    has_position: bool,
}

impl SanosStrategyWrapper {
    /// Create a new wrapper.
    pub fn new(
        underlying: &str,
        lot_size: i32,
        calibration_interval_secs: u64,
        relax_e_gates: bool,
    ) -> Self {
        Self {
            adapter: SanosCalendarCarryAdapter::new(
                underlying,
                lot_size as u32,
                calibration_interval_secs,
                relax_e_gates,
            ),
            margin_gate: None,
            instrument_map: None,
            lot_size,
            last_margin_reserved: 0.0,
            pending_margin_action: None,
            tui_tx: None,
            token_cache: HashMap::new(),
            has_position: false,
        }
    }

    /// Set the margin gate.
    pub fn with_margin_gate(mut self, gate: Arc<Mutex<MarginGate>>) -> Self {
        self.margin_gate = Some(gate);
        self
    }

    /// Set the instrument map.
    pub fn with_instrument_map(mut self, map: InstrumentMap) -> Self {
        self.instrument_map = Some(map);
        self
    }

    /// Set the TUI state sender.
    pub fn with_tui_sender(mut self, tx: watch::Sender<SanosTuiState>) -> Self {
        self.tui_tx = Some(tx);
        self
    }

    /// Update token cache from snapshot quotes.
    fn update_token_cache(&mut self, snapshot: &OptionsSnapshot) {
        self.token_cache.clear();
        for q in &snapshot.quotes {
            self.token_cache
                .insert((q.expiry.clone(), q.strike, q.right), q.instrument_token);
        }
    }

    /// Resolve token for (expiry, strike, right).
    fn resolve_token(&self, expiry: &str, strike: i32, right: Right) -> Option<u32> {
        self.token_cache
            .get(&(expiry.to_string(), strike, right))
            .copied()
    }

    /// Build trade intents from SANOS Enter decision.
    fn build_entry_intents(
        &self,
        front_expiry: &str,
        back_expiry: &str,
        front_strike: f64,
        back_strike: f64,
        front_lots: i32,
        back_lots: i32,
    ) -> Option<Vec<TradeIntent>> {
        let front_strike_i32 = front_strike as i32;
        let back_strike_i32 = back_strike as i32;

        let front_ce_token = self.resolve_token(front_expiry, front_strike_i32, Right::Call)?;
        let front_pe_token = self.resolve_token(front_expiry, front_strike_i32, Right::Put)?;
        let back_ce_token = self.resolve_token(back_expiry, back_strike_i32, Right::Call)?;
        let back_pe_token = self.resolve_token(back_expiry, back_strike_i32, Right::Put)?;

        let contracts_front = front_lots.unsigned_abs() as i32 * self.lot_size;
        let contracts_back = back_lots.unsigned_abs() as i32 * self.lot_size;

        // Front leg: short straddle (sell CE + sell PE)
        let front_ce = TradeIntent::new(
            front_ce_token,
            Side::Sell,
            contracts_front,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            self.lot_size,
        );
        let front_pe = TradeIntent::new(
            front_pe_token,
            Side::Sell,
            contracts_front,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            self.lot_size,
        );

        // Back leg: long straddle (buy CE + buy PE)
        let back_ce = TradeIntent::new(
            back_ce_token,
            Side::Buy,
            contracts_back,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            self.lot_size,
        );
        let back_pe = TradeIntent::new(
            back_pe_token,
            Side::Buy,
            contracts_back,
            IntentTag::CalendarCarryEntry,
            InstrumentKind::IndexOption,
            self.lot_size,
        );

        Some(vec![front_ce, front_pe, back_ce, back_pe])
    }

    /// Check margin for a set of intents using Zerodha SPAN API.
    async fn check_margin(&self, intents: &[TradeIntent]) -> Result<f64, MarginRejectReason> {
        let gate = match &self.margin_gate {
            Some(g) => g,
            None => return Ok(0.0),
        };

        let mut orders: Vec<MarginOrderParams> = Vec::new();
        for intent in intents {
            let symbol = self
                .instrument_map
                .as_ref()
                .and_then(|m| m.get(intent.instrument_token))
                .map(|meta| meta.tradingsymbol.clone())
                .unwrap_or_else(|| format!("TOKEN_{}", intent.instrument_token));

            let transaction_type = match intent.side {
                Side::Buy => "BUY",
                Side::Sell => "SELL",
            };

            orders.push(MarginOrderParams {
                exchange: "NFO".to_string(),
                tradingsymbol: symbol,
                transaction_type: transaction_type.to_string(),
                variety: "regular".to_string(),
                product: "NRML".to_string(),
                order_type: "MARKET".to_string(),
                quantity: intent.qty,
                price: None,
            });
        }

        let mut gate_lock = gate.lock().await;
        let result = gate_lock.check_basket_entry(orders).await;

        match result {
            Ok(req) => {
                info!(
                    total = req.total,
                    from_cache = req.from_cache,
                    "[SANOS-MARGIN] SPAN check passed"
                );
                Ok(req.total)
            }
            Err(e) => {
                warn!(reason = %e, "[SANOS-MARGIN] SPAN check failed");
                Err(e)
            }
        }
    }

    /// Reserve margin (engine-confirmed AllFilled).
    async fn reserve_margin_confirmed(&mut self, margin: f64) {
        if let Some(gate) = &self.margin_gate {
            let mut gate_lock = gate.lock().await;
            gate_lock.reserve_margin(margin);
            self.last_margin_reserved = margin;
        }
    }

    /// Release margin (engine-confirmed AllFilled).
    async fn release_margin_confirmed(&mut self) {
        if let Some(gate) = &self.margin_gate
            && self.last_margin_reserved > 0.0
        {
            let mut gate_lock = gate.lock().await;
            gate_lock.release_margin(self.last_margin_reserved);
            self.last_margin_reserved = 0.0;
        }
    }

    /// Convert GateCheckResult to TUI gate views.
    fn gates_to_views(gates: &GateCheckResult) -> Vec<SanosGateView> {
        let gate_list = [
            &gates.h1_surface,
            &gates.h2_calendar,
            &gates.h3_quote_front,
            &gates.h3_quote_back,
            &gates.h4_liquidity_front,
            &gates.h4_liquidity_back,
            &gates.carry,
            &gates.r1_inversion,
            &gates.r2_skew,
            &gates.e1_premium_gap,
            &gates.e2_friction_dominance,
            &gates.e3_friction_floor,
        ];

        gate_list
            .iter()
            .map(|g| SanosGateView {
                name: g.name.clone(),
                passed: g.passed,
                value: g.value,
                threshold: g.threshold,
                reason: g.reason.clone(),
            })
            .collect()
    }

    /// Broadcast TUI state.
    fn broadcast_tui_state(&self, result: &super::adapter::AdapterResult) {
        let Some(tx) = &self.tui_tx else { return };

        let gate_views = match &result.gates {
            Some(g) => Self::gates_to_views(g),
            None => Vec::new(),
        };

        let secs_since = result
            .last_calibration_ts
            .map(|last| (Utc::now() - last).num_milliseconds() as f64 / 1000.0);

        let state = SanosTuiState {
            surfaces: result.surfaces.clone(),
            gates: gate_views,
            features: result.feature_view.clone(),
            last_calibration_ts: result.last_calibration_ts.map(|t| t.to_rfc3339()),
            secs_since_calibration: secs_since,
            warmup: result.last_calibration_ts.is_none(),
            last_decision: Some(format!("{:?}", result.decision)),
        };

        let _ = tx.send(state);
    }
}

/// Convert Rationale-style metrics for the engine.
fn sanos_to_metrics(decision: &str) -> quantlaxmi_paper::DecisionMetrics {
    quantlaxmi_paper::DecisionMetrics {
        edge_estimate: 0,
        friction_estimate: 0,
        spread_cost: 0,
        stale_quotes_ratio_bps: 0,
        strategy_name: format!("SanosCalendarCarry({})", decision),
    }
}

/// Build a StrategyView for the engine/TUI.
fn build_strategy_view(
    snapshot: &OptionsSnapshot,
    decision_type: &str,
    decision_reason: &str,
) -> quantlaxmi_paper::StrategyView {
    quantlaxmi_paper::StrategyView {
        name: "SANOS Calendar Carry".to_string(),
        underlying: snapshot.underlying.clone(),
        spot: snapshot.spot,
        futures: None,
        edge_rupees: 0.0,
        friction_rupees: 0.0,
        net_edge_rupees: 0.0,
        entry_threshold_rupees: 8.0,
        exit_threshold_rupees: 2.0,
        front_leg: None,
        back_leg: None,
        positions: vec![],
        decision_type: decision_type.to_string(),
        decision_reason: decision_reason.to_string(),
    }
}

#[async_trait]
impl Strategy<OptionsSnapshot, TradeIntent> for SanosStrategyWrapper {
    async fn on_snapshot(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &OptionsSnapshot,
    ) -> Result<quantlaxmi_paper::StrategyDecision<TradeIntent>> {
        // Update token cache from snapshot
        self.update_token_cache(snapshot);

        // Run SANOS adapter
        let result = self.adapter.evaluate(ts, snapshot);

        // Broadcast TUI state
        self.broadcast_tui_state(&result);

        // Convert SANOS decision to PaperEngine decision
        match result.decision {
            SanosStrategyDecision::Enter { intent, gates: _ } => {
                // Build trade intents from SANOS enter intent
                let intents = match self.build_entry_intents(
                    &intent.front_expiry,
                    &intent.back_expiry,
                    intent.front_strike,
                    intent.back_strike,
                    intent.front_lots,
                    intent.back_lots,
                ) {
                    Some(intents) => intents,
                    None => {
                        let reason = "Cannot resolve tokens for entry".to_string();
                        let strategy_view = build_strategy_view(snapshot, "Refuse", &reason);
                        return Ok(quantlaxmi_paper::StrategyDecision {
                            ts,
                            accepted: false,
                            reason,
                            intents: vec![],
                            decision_type: quantlaxmi_paper::DecisionType::Refuse,
                            metrics: Some(sanos_to_metrics("TokenResolveFailure")),
                            strategy_view: Some(strategy_view),
                        });
                    }
                };

                // Margin check
                match self.check_margin(&intents).await {
                    Ok(margin_required) => {
                        self.pending_margin_action =
                            Some(PendingMarginAction::Reserve(margin_required));
                        self.has_position = true;

                        let reason = format!(
                            "SANOS_ENTER cal={:.2} friction={:.2} h={:.2} margin={:.0}",
                            intent.cal_value,
                            intent.friction_estimate,
                            intent.hedge_ratio,
                            margin_required,
                        );

                        let strategy_view = build_strategy_view(snapshot, "Accept", &reason);

                        Ok(quantlaxmi_paper::StrategyDecision {
                            ts,
                            accepted: true,
                            reason,
                            intents,
                            decision_type: quantlaxmi_paper::DecisionType::Accept,
                            metrics: Some(sanos_to_metrics("Enter")),
                            strategy_view: Some(strategy_view),
                        })
                    }
                    Err(margin_err) => {
                        let reason = format!("SANOS_MARGIN_REJECTED: {}", margin_err);
                        let strategy_view = build_strategy_view(snapshot, "Refuse", &reason);

                        Ok(quantlaxmi_paper::StrategyDecision {
                            ts,
                            accepted: false,
                            reason,
                            intents: vec![],
                            decision_type: quantlaxmi_paper::DecisionType::Refuse,
                            metrics: Some(sanos_to_metrics("MarginReject")),
                            strategy_view: Some(strategy_view),
                        })
                    }
                }
            }

            SanosStrategyDecision::NoTrade { reason, .. } => {
                let strategy_view = build_strategy_view(snapshot, "Refuse", &reason);

                Ok(quantlaxmi_paper::StrategyDecision {
                    ts,
                    accepted: false,
                    reason,
                    intents: vec![],
                    decision_type: quantlaxmi_paper::DecisionType::Refuse,
                    metrics: Some(sanos_to_metrics("NoTrade")),
                    strategy_view: Some(strategy_view),
                })
            }

            SanosStrategyDecision::Hold => {
                let reason = "SANOS_HOLD".to_string();
                let strategy_view = build_strategy_view(snapshot, "Hold", &reason);

                Ok(quantlaxmi_paper::StrategyDecision {
                    ts,
                    accepted: true,
                    reason,
                    intents: vec![],
                    decision_type: quantlaxmi_paper::DecisionType::Hold,
                    metrics: Some(sanos_to_metrics("Hold")),
                    strategy_view: Some(strategy_view),
                })
            }

            SanosStrategyDecision::Exit { intent } => {
                // For now, exit handling is deferred (no position tracking in v1)
                self.pending_margin_action = Some(PendingMarginAction::Release);
                self.has_position = false;

                let reason = format!("SANOS_EXIT: {}", intent.reason);
                let strategy_view = build_strategy_view(snapshot, "Accept", &reason);

                Ok(quantlaxmi_paper::StrategyDecision {
                    ts,
                    accepted: true,
                    reason,
                    intents: vec![], // Exit intents would require position tracking
                    decision_type: quantlaxmi_paper::DecisionType::Accept,
                    metrics: Some(sanos_to_metrics("Exit")),
                    strategy_view: Some(strategy_view),
                })
            }
        }
    }

    async fn on_outcome(
        &mut self,
        _decision: &quantlaxmi_paper::StrategyDecision<TradeIntent>,
        outcome: quantlaxmi_paper::FillOutcome,
    ) -> Result<()> {
        let action = self.pending_margin_action.take();
        if action.is_none() {
            return Ok(());
        }

        match (outcome, action) {
            (quantlaxmi_paper::FillOutcome::AllFilled, Some(PendingMarginAction::Reserve(m))) => {
                self.reserve_margin_confirmed(m).await;
            }
            (quantlaxmi_paper::FillOutcome::AllFilled, Some(PendingMarginAction::Release)) => {
                self.release_margin_confirmed().await;
            }
            _ => {}
        }

        Ok(())
    }
}
