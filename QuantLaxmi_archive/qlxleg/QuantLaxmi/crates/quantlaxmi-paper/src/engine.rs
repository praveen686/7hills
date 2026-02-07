//! Paper Trading Engine
//!
//! Venue-agnostic orchestration of:
//! - Market feed (snapshots)
//! - Strategy (decisions)
//! - Fill model (execution simulation)
//! - Ledger (position tracking)
//!
//! ## Identity Contract
//!
//! The engine uses `InstrumentIdentity` to extract instrument keys from intents.
//! This keeps the engine market-agnostic while enabling proper position tracking.
//!
//! ## Decision Summary Logging
//!
//! When `config.log_decisions` is enabled, the engine logs a per-decision summary
//! for edge vs friction analysis. This enables profitability tuning without code changes.

use anyhow::Result;
use std::fmt::Debug;
use tokio::sync::watch;
use tracing::{debug, info, warn};

use crate::{
    FillOutcome, FillRejection, InstrumentIdentity, Ledger, MarketEvent, MarketFeed,
    PaperFillModel, PaperState, PositionView, Strategy, StrategyDecision, TopOfBookProvider,
};

// =============================================================================
// ENGINE CONFIG
// =============================================================================

/// Engine configuration.
#[derive(Debug, Clone, Default)]
pub struct EngineConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Enable per-decision summary logging (edge vs friction).
    /// Disabled by default for high-frequency scenarios.
    pub log_decisions: bool,
    /// Optional watch channel sender for state updates (for TUI).
    pub state_tx: Option<watch::Sender<PaperState>>,
}

impl EngineConfig {
    /// Create a new EngineConfig with specified capital.
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            log_decisions: true, // Default on for paper trading
            state_tx: None,
        }
    }

    /// Enable state broadcasting via watch channel.
    pub fn with_state_channel(mut self, tx: watch::Sender<PaperState>) -> Self {
        self.state_tx = Some(tx);
        self
    }
}

/// Venue-agnostic paper engine.
///
/// Generic over:
/// - `F`: Market feed type
/// - `S`: Strategy type
/// - `M`: Fill model type
/// - `TSnapshot`: Market snapshot type (must implement `TopOfBookProvider` for MTM)
/// - `TIntent`: Trade intent type (must implement `InstrumentIdentity`)
pub struct PaperEngine<F, S, M, TSnapshot, TIntent>
where
    F: MarketFeed<TSnapshot>,
    S: Strategy<TSnapshot, TIntent>,
    M: PaperFillModel<TSnapshot, TIntent>,
    TSnapshot: TopOfBookProvider,
    TIntent: InstrumentIdentity,
{
    feed: F,
    strategy: S,
    fill_model: M,
    ledger: Ledger,
    state: PaperState,
    config: EngineConfig,
    _phantom: std::marker::PhantomData<(TSnapshot, TIntent)>,
}

impl<F, S, M, TSnapshot, TIntent> PaperEngine<F, S, M, TSnapshot, TIntent>
where
    F: MarketFeed<TSnapshot>,
    S: Strategy<TSnapshot, TIntent>,
    M: PaperFillModel<TSnapshot, TIntent>,
    TSnapshot: TopOfBookProvider,
    TIntent: InstrumentIdentity,
    TIntent::Key: Into<u32>, // For ledger compatibility (uses u32 tokens)
{
    /// Create a new paper engine with default config.
    pub fn new(feed: F, strategy: S, fill_model: M, initial_capital: f64) -> Self {
        Self::with_config(
            feed,
            strategy,
            fill_model,
            EngineConfig {
                initial_capital,
                ..Default::default()
            },
        )
    }

    /// Create a new paper engine with custom config.
    pub fn with_config(feed: F, strategy: S, fill_model: M, config: EngineConfig) -> Self {
        let ledger = Ledger::new(config.initial_capital);
        Self {
            feed,
            strategy,
            fill_model,
            ledger,
            state: PaperState {
                cash: config.initial_capital,
                equity: config.initial_capital,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
                total_pnl: 0.0,
                fees_paid: 0.0,
                open_positions: 0,
                ..Default::default()
            },
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get current state.
    pub fn state(&self) -> &PaperState {
        &self.state
    }

    /// Get ledger.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Get mutable ledger (for direct access if needed).
    pub fn ledger_mut(&mut self) -> &mut Ledger {
        &mut self.ledger
    }

    /// Run a single step of the engine loop.
    ///
    /// ## Invariants
    ///
    /// - Per snapshot: strategy sees snapshot, emits decision
    /// - Accepted intents go to fill model
    /// - Fills update ledger with intent's identity
    /// - Rejections are logged with intent's identity
    /// - Errors are explicit events, no hidden skipping
    /// - One decision summary log per snapshot (when enabled)
    pub async fn step(&mut self) -> Result<()> {
        match self.feed.next().await? {
            MarketEvent::Heartbeat { ts } => {
                self.state.ts = Some(ts);
                debug!(ts = %ts, "[ENGINE] Heartbeat");
            }
            MarketEvent::Snapshot { ts, snapshot } => {
                // 1. Get strategy decision
                let decision = self.strategy.on_snapshot(ts, &snapshot).await?;
                self.state.ts = Some(ts);
                self.state.last_decision = Some(decision.reason.clone());
                self.state.strategy_view = decision.strategy_view.clone();

                // 2. Process decision and track outcome
                let outcome = self.process_decision(ts, &snapshot, &decision);

                // 2b. Allow strategy to react to the realized outcome (side-effects)
                // This is used for things like reserving/releasing margin only after `AllFilled`.
                self.strategy.on_outcome(&decision, outcome).await?;

                // 3. Update state with conservative MTM (after every snapshot)
                self.update_state_mtm(&snapshot);

                // 4. Publish state via watch channel (for TUI)
                if let Some(tx) = &self.config.state_tx {
                    let _ = tx.send(self.state.clone());
                }

                // 5. Log decision summary (feature-gated)
                if self.config.log_decisions {
                    log_decision_summary(ts, &decision, outcome);
                }
            }
        }
        Ok(())
    }

    /// Process a strategy decision and return the fill outcome.
    fn process_decision(
        &mut self,
        ts: chrono::DateTime<chrono::Utc>,
        snapshot: &TSnapshot,
        decision: &StrategyDecision<TIntent>,
    ) -> FillOutcome {
        // Handle non-accepted decisions
        if !decision.accepted {
            debug!(reason = %decision.reason, "[ENGINE] Strategy refused");
            return FillOutcome::None;
        }

        // No intents = Hold
        if decision.intents.is_empty() {
            return FillOutcome::None;
        }

        // Process each intent and track fills/rejections
        let mut fills = 0usize;
        let mut rejections = 0usize;

        for intent in &decision.intents {
            let key: u32 = intent.instrument_key().into();

            match self.fill_model.try_fill(ts, snapshot, intent) {
                Ok(fill) => {
                    // Apply fill to ledger using intent's identity
                    let realized = self.ledger.apply_fill(&fill, key, &decision.reason);

                    // Update state
                    self.state.fees_paid = self.ledger.fees.total;
                    self.state.realized_pnl = self.ledger.realized_pnl;
                    self.state.open_positions = self.ledger.open_position_count();

                    info!(
                        symbol = %fill.symbol,
                        side = ?fill.side,
                        qty = fill.qty,
                        price = fill.price,
                        fees = fill.fees.total,
                        realized = realized,
                        "[ENGINE] Fill executed"
                    );

                    fills += 1;
                    self.state.fills += 1;
                }
                Err(rejection) => {
                    // Log rejection with intent's identity
                    self.ledger.record_rejection();
                    log_rejection(key, &rejection);
                    rejections += 1;
                    self.state.rejections += 1;
                }
            }
        }

        // Determine outcome
        match (fills, rejections) {
            (f, 0) if f > 0 => FillOutcome::AllFilled,
            (0, r) if r > 0 => FillOutcome::AllRejected,
            (f, r) if f > 0 && r > 0 => FillOutcome::PartialFill,
            _ => FillOutcome::None,
        }
    }

    /// Run the engine until the feed is exhausted or an error occurs.
    pub async fn run(&mut self) -> Result<()> {
        loop {
            if let Err(e) = self.step().await {
                warn!(error = %e, "[ENGINE] Step error, stopping");
                return Err(e);
            }
        }
    }

    /// Run for a specified number of steps.
    pub async fn run_steps(&mut self, max_steps: usize) -> Result<usize> {
        let mut steps = 0;
        while steps < max_steps {
            self.step().await?;
            steps += 1;
        }
        Ok(steps)
    }

    /// Update state with conservative MTM from current snapshot.
    ///
    /// Called after every snapshot to ensure equity is always current.
    /// Uses TopOfBookProvider to get bid/ask for each position.
    fn update_state_mtm(&mut self, snapshot: &TSnapshot) {
        // Update cash from ledger
        self.state.cash = self.ledger.cash;
        self.state.realized_pnl = self.ledger.realized_pnl;
        self.state.fees_paid = self.ledger.fees.total;
        self.state.open_positions = self.ledger.open_position_count();

        // Compute unrealized PnL with conservative MTM and build position views
        let mut unrealized_total = 0.0;
        let mut position_views = Vec::new();

        for pos in self.ledger.positions() {
            if pos.is_flat() {
                continue;
            }
            let (bid, ask) = snapshot.best_bid_ask(pos.token).unwrap_or((0.0, 0.0));
            let pnl = pos.unrealized_pnl(bid, ask);
            let mtm = if pos.qty > 0.0 { bid } else { ask };
            unrealized_total += pnl;

            position_views.push(PositionView {
                symbol: pos.symbol.clone(),
                qty: pos.qty as i32,
                avg_price: pos.avg_price,
                mtm,
                unrealized_pnl: pnl,
            });
        }

        self.state.unrealized_pnl = unrealized_total;

        // Inject positions into strategy_view so the TUI can render them
        if let Some(ref mut view) = self.state.strategy_view {
            view.positions = position_views;
        }

        // Compute equity and total PnL
        // equity = cash + unrealized (cash already has fees deducted)
        self.state.equity = self.state.cash + self.state.unrealized_pnl;
        // total_pnl = equity - initial_capital (the true net P&L)
        self.state.total_pnl = self.state.equity - self.config.initial_capital;
    }

    /// Get a summary of the ledger with current MTM.
    ///
    /// Uses the last known snapshot for MTM computation.
    pub fn summary(
        &self,
        price_provider: impl Fn(u32) -> Option<(f64, f64)>,
    ) -> crate::LedgerSummary {
        self.ledger.summary(price_provider)
    }

    /// Mark the engine as finished and broadcast final state.
    ///
    /// Call this after the trading loop ends to signal the TUI to exit.
    pub fn mark_finished(&mut self) {
        self.state.is_finished = true;
        if let Some(tx) = &self.config.state_tx {
            let _ = tx.send(self.state.clone());
        }
    }
}

/// Log a fill rejection.
fn log_rejection<K: Debug>(key: K, rejection: &FillRejection) {
    match rejection {
        FillRejection::NoExecutableQuote { reason } => {
            warn!(key = ?key, reason = %reason, "[ENGINE] Rejected: no executable quote");
        }
        FillRejection::StaleQuote {
            age_ms,
            threshold_ms,
        } => {
            warn!(
                key = ?key,
                age_ms = age_ms,
                threshold_ms = threshold_ms,
                "[ENGINE] Rejected: stale quote"
            );
        }
        FillRejection::InsufficientQuantity {
            requested,
            available,
        } => {
            warn!(
                key = ?key,
                requested = requested,
                available = available,
                "[ENGINE] Rejected: insufficient quantity"
            );
        }
        FillRejection::Other { reason } => {
            warn!(key = ?key, reason = %reason, "[ENGINE] Rejected: other");
        }
    }
}

/// Log per-decision summary for edge vs friction analysis.
///
/// One log per decision, not per intent. Enables:
/// - Edge vs friction scatter plots
/// - Regime-wise analysis
/// - Threshold tuning
/// - Time-of-day filtering
fn log_decision_summary<TIntent>(
    ts: chrono::DateTime<chrono::Utc>,
    decision: &StrategyDecision<TIntent>,
    outcome: FillOutcome,
) {
    // Extract metrics or use defaults
    let (edge, friction, spread, stale_bps, strategy) = match &decision.metrics {
        Some(m) => (
            m.edge_estimate,
            m.friction_estimate,
            m.spread_cost,
            m.stale_quotes_ratio_bps,
            m.strategy_name.as_str(),
        ),
        None => (0, 0, 0, 0, "unknown"),
    };

    let edge_minus_friction = edge - friction;

    // Use target for independent routing/filtering
    info!(
        target: "paper.decision",
        edge_estimate = edge,
        friction_estimate = friction,
        edge_minus_friction = edge_minus_friction,
        decision = %decision.decision_type,
        fill_outcome = ?outcome,
        spread_cost = spread,
        stale_quotes_ratio_bps = stale_bps,
        snapshot_ts_ns = ts.timestamp_nanos_opt().unwrap_or(0),
        strategy = %strategy,
        "[DECISION] summary"
    );
}
