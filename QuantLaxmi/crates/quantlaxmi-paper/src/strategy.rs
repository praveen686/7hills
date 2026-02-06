use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::state::StrategyView;

// =============================================================================
// FILL OUTCOME (Reported by Engine)
// =============================================================================

/// Outcome of processing intents for a single decision.
///
/// Reported by the engine back to the strategy via `Strategy::on_outcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillOutcome {
    /// All intents filled successfully.
    AllFilled,
    /// Some intents filled, some rejected.
    PartialFill,
    /// All intents rejected.
    AllRejected,
    /// No intents to process (Refuse/Hold).
    None,
}

// =============================================================================
// DECISION METRICS (Venue-Agnostic)
// =============================================================================

/// Venue-agnostic decision metrics for edge vs friction analysis.
///
/// All monetary values in smallest unit (paise for India, satoshis for crypto).
/// This enables per-decision summary logging for profitability tuning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionMetrics {
    /// Edge estimate (expected profit from position)
    pub edge_estimate: i64,
    /// Friction estimate (expected round-trip costs)
    pub friction_estimate: i64,
    /// Spread cost (half-spread on entry)
    pub spread_cost: i64,
    /// Ratio of stale quotes (basis points, 0-10000)
    pub stale_quotes_ratio_bps: u32,
    /// Strategy name for attribution
    pub strategy_name: String,
}

impl DecisionMetrics {
    /// Edge minus friction (net expected value).
    #[inline]
    pub fn edge_minus_friction(&self) -> i64 {
        self.edge_estimate - self.friction_estimate
    }
}

// =============================================================================
// DECISION TYPE (For Logging)
// =============================================================================

/// Decision type for logging (Accept/Refuse/Hold).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionType {
    Accept,
    Refuse,
    Hold,
}

impl std::fmt::Display for DecisionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecisionType::Accept => write!(f, "Accept"),
            DecisionType::Refuse => write!(f, "Refuse"),
            DecisionType::Hold => write!(f, "Hold"),
        }
    }
}

// =============================================================================
// STRATEGY DECISION
// =============================================================================

/// Strategy decision emitted per evaluation step.
#[derive(Debug, Clone)]
pub struct StrategyDecision<TIntent> {
    pub ts: DateTime<Utc>,
    /// Whether the strategy is allowed to trade (gates/risk/session).
    pub accepted: bool,
    /// Human/audit reason (refusal rationale, confidence, etc.)
    pub reason: String,
    /// Optional intents (orders) to submit to paper executor.
    pub intents: Vec<TIntent>,
    /// Decision type (Accept/Refuse/Hold) for summary logging.
    pub decision_type: DecisionType,
    /// Optional metrics for per-decision edge vs friction analysis.
    pub metrics: Option<DecisionMetrics>,
    /// Optional strategy view for TUI display.
    pub strategy_view: Option<StrategyView>,
}

#[async_trait]
pub trait Strategy<TSnapshot, TIntent>: Send + Sync {
    async fn on_snapshot(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &TSnapshot,
    ) -> Result<StrategyDecision<TIntent>>;

    /// Optional callback invoked after the engine processes a decision.
    ///
    /// This is the correct place to perform side effects that must depend on
    /// actual fill outcomes (e.g., reserve/release margin only after `AllFilled`).
    ///
    /// Default implementation is a no-op.
    async fn on_outcome(
        &mut self,
        _decision: &StrategyDecision<TIntent>,
        _outcome: FillOutcome,
    ) -> Result<()> {
        Ok(())
    }
}
