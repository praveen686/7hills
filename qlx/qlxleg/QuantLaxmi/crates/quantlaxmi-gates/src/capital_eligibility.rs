//! # Capital Eligibility Layer (Phase 13.1)
//!
//! Translates G3 promotion into capital eligibility signals,
//! without assigning size, leverage, or portfolio weight.
//!
//! ## Core Question
//! "Is this G3 strategy allowed to touch capital under what constraints?"
//!
//! ## NOT in Scope
//! - Position sizing
//! - Capital math
//! - Portfolio optimization
//! - Cross-strategy arbitration
//! - Runtime enforcement
//!
//! ## Required Inputs
//! - `PromotionDecision` (must be Accepted)
//! - `PaperEvidence` (validated in Phase 12.3)
//! - Gate summaries (G3 robustness)
//! - Strategy metadata (venue, symbols)
//!
//! ## Mandatory Invariants
//! - No G3 → No eligibility
//! - No paper evidence → No eligibility
//! - Promotion rejected → No eligibility
//! - Drawdown breach → Ineligible
//! - Alpha below threshold → Ineligible

use crate::GateResult;
use crate::promotion::PromotionDecision;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

pub const ELIGIBILITY_DECISION_SCHEMA: &str = "eligibility_decision_v1.0";

// =============================================================================
// Venue & Symbol Types
// =============================================================================

/// Trading venue identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Venue {
    /// Binance perpetual futures
    BinancePerp,
    /// Binance spot
    BinanceSpot,
    /// NSE India futures
    NseF,
    /// NSE India options
    NseO,
    /// Paper trading (any venue simulation)
    Paper,
}

impl std::fmt::Display for Venue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Venue::BinancePerp => write!(f, "binance_perp"),
            Venue::BinanceSpot => write!(f, "binance_spot"),
            Venue::NseF => write!(f, "nse_f"),
            Venue::NseO => write!(f, "nse_o"),
            Venue::Paper => write!(f, "paper"),
        }
    }
}

/// Time window for trading restrictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start hour (0-23, UTC)
    pub start_hour: u8,
    /// End hour (0-23, UTC)
    pub end_hour: u8,
    /// Days of week (0=Sunday, 6=Saturday)
    pub days: Vec<u8>,
}

// =============================================================================
// Eligibility Status
// =============================================================================

/// Eligibility outcome — explicit, not boolean.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EligibilityStatus {
    /// Strategy is eligible for capital
    Eligible,
    /// Strategy is not eligible (hard rejection)
    Ineligible { reasons: Vec<String> },
    /// Strategy is conditionally eligible (soft constraints)
    Conditional {
        conditions: Vec<EligibilityCondition>,
    },
}

impl EligibilityStatus {
    /// Check if status is eligible (unconditional).
    pub fn is_eligible(&self) -> bool {
        matches!(self, EligibilityStatus::Eligible)
    }

    /// Check if status allows any capital access.
    pub fn allows_capital(&self) -> bool {
        !matches!(self, EligibilityStatus::Ineligible { .. })
    }
}

/// Condition for conditional eligibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EligibilityCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Human-readable description
    pub description: String,
}

/// Types of eligibility conditions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConditionType {
    /// Restricted to specific symbols
    SymbolRestriction,
    /// Restricted to specific time windows
    TimeWindowRestriction,
    /// Restricted to specific venue
    VenueRestriction,
    /// Notional cap applies
    NotionalCap,
    /// Requires additional monitoring
    EnhancedMonitoring,
}

// =============================================================================
// Capital Constraints
// =============================================================================

/// Capital constraints — bounds, not instructions.
///
/// These restrict future allocators; they do not decide for them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapitalConstraints {
    /// Target venue
    pub venue: Option<Venue>,
    /// Maximum notional exposure (mantissa, exponent -8)
    pub max_notional_mantissa: Option<i128>,
    /// Maximum daily loss (mantissa, exponent -8)
    pub max_daily_loss_mantissa: Option<i128>,
    /// Maximum drawdown from peak (mantissa, exponent -8)
    pub max_drawdown_mantissa: Option<i128>,
    /// Exponent for all monetary values
    pub monetary_exponent: i8,
    /// Allowed symbols (None = all)
    pub allowed_symbols: Option<Vec<String>>,
    /// Allowed time windows (None = all)
    pub allowed_time_windows: Option<Vec<TimeWindow>>,
}

impl CapitalConstraints {
    /// Create new constraints with default exponent.
    pub fn new() -> Self {
        Self {
            monetary_exponent: -8,
            ..Default::default()
        }
    }

    /// Set venue constraint.
    pub fn with_venue(mut self, venue: Venue) -> Self {
        self.venue = Some(venue);
        self
    }

    /// Set max notional (as f64, converted to mantissa).
    pub fn with_max_notional(mut self, max_notional: f64) -> Self {
        self.max_notional_mantissa = Some((max_notional * 100_000_000.0) as i128);
        self
    }

    /// Set max daily loss (as f64, converted to mantissa).
    pub fn with_max_daily_loss(mut self, max_loss: f64) -> Self {
        self.max_daily_loss_mantissa = Some((max_loss * 100_000_000.0) as i128);
        self
    }

    /// Set max drawdown (as f64, converted to mantissa).
    pub fn with_max_drawdown(mut self, max_drawdown: f64) -> Self {
        self.max_drawdown_mantissa = Some((max_drawdown * 100_000_000.0) as i128);
        self
    }

    /// Set allowed symbols.
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.allowed_symbols = Some(symbols);
        self
    }
}

// =============================================================================
// Eligibility Policy
// =============================================================================

/// Policy configuration for eligibility evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityPolicy {
    /// Require G3 gate pass
    pub require_g3: bool,
    /// Minimum paper session duration
    pub min_paper_duration_ns: i64,
    /// Maximum historical drawdown (mantissa, exponent -4 = basis points)
    /// e.g., 2000 = 20% max drawdown
    pub max_historical_drawdown_bps: i64,
    /// Minimum alpha score (mantissa, exponent -4)
    pub min_alpha_score_mantissa: i64,
    /// Alpha score exponent
    pub alpha_exponent: i8,
    /// Maximum correlation with existing strategies (placeholder for Phase 13.2)
    pub max_correlation: Option<f64>,
    /// Minimum number of paper trades
    pub min_paper_trades: u32,
    /// Minimum win rate (basis points, 10000 = 100%)
    pub min_win_rate_bps: u32,
}

impl Default for EligibilityPolicy {
    fn default() -> Self {
        Self {
            require_g3: true,
            min_paper_duration_ns: 3_600_000_000_000, // 1 hour
            max_historical_drawdown_bps: 2000,        // 20%
            min_alpha_score_mantissa: 1000,           // 0.1 at exp -4
            alpha_exponent: -4,
            max_correlation: None, // Not enforced yet
            min_paper_trades: 10,
            min_win_rate_bps: 4000, // 40%
        }
    }
}

impl EligibilityPolicy {
    /// Strict policy for production.
    pub fn strict() -> Self {
        Self {
            require_g3: true,
            min_paper_duration_ns: 7_200_000_000_000, // 2 hours
            max_historical_drawdown_bps: 1500,        // 15%
            min_alpha_score_mantissa: 2000,           // 0.2 at exp -4
            alpha_exponent: -4,
            max_correlation: Some(0.7),
            min_paper_trades: 25,
            min_win_rate_bps: 4500, // 45%
        }
    }

    /// Lenient policy for testing.
    pub fn lenient() -> Self {
        Self {
            require_g3: false,
            min_paper_duration_ns: 0,
            max_historical_drawdown_bps: 5000, // 50%
            min_alpha_score_mantissa: 0,
            alpha_exponent: -4,
            max_correlation: None,
            min_paper_trades: 1,
            min_win_rate_bps: 0,
        }
    }
}

// =============================================================================
// Eligibility Check
// =============================================================================

/// Individual eligibility check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityCheck {
    /// Check name
    pub name: String,
    /// Pass/fail
    pub passed: bool,
    /// Reason/details
    pub reason: String,
    /// Is this a hard requirement (fail = ineligible) or soft (fail = conditional)?
    pub is_hard: bool,
}

impl EligibilityCheck {
    /// Create a passing hard check.
    pub fn pass_hard(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            reason: reason.to_string(),
            is_hard: true,
        }
    }

    /// Create a failing hard check.
    pub fn fail_hard(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            reason: reason.to_string(),
            is_hard: true,
        }
    }

    /// Create a passing soft check.
    pub fn pass_soft(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            reason: reason.to_string(),
            is_hard: false,
        }
    }

    /// Create a failing soft check (leads to conditional eligibility).
    pub fn fail_soft(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            reason: reason.to_string(),
            is_hard: false,
        }
    }
}

// =============================================================================
// Eligibility Decision
// =============================================================================

/// Decision on capital eligibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityDecision {
    /// Schema version
    pub schema_version: String,
    /// Strategy ID
    pub strategy_id: String,
    /// Eligibility status
    pub eligibility: EligibilityStatus,
    /// Capital constraints (if eligible/conditional)
    pub constraints: CapitalConstraints,
    /// Individual checks
    pub checks: Vec<EligibilityCheck>,
    /// Decision timestamp
    pub decided_at: DateTime<Utc>,
    /// Promotion decision ID (link to Phase 12.3)
    pub promotion_decision_id: String,
    /// Decision digest (for audit trail)
    pub decision_digest: String,
}

impl EligibilityDecision {
    /// Compute decision digest.
    pub fn compute_digest(
        strategy_id: &str,
        eligibility: &EligibilityStatus,
        checks: &[EligibilityCheck],
    ) -> String {
        let status_str = match eligibility {
            EligibilityStatus::Eligible => "eligible",
            EligibilityStatus::Ineligible { .. } => "ineligible",
            EligibilityStatus::Conditional { .. } => "conditional",
        };
        let checks_str: String = checks
            .iter()
            .map(|c| format!("{}:{}", c.name, c.passed))
            .collect::<Vec<_>>()
            .join(";");

        let input = format!(
            "eligibility_decision_v1:{}:{}:{}",
            strategy_id, status_str, checks_str
        );
        let hash = Sha256::digest(input.as_bytes());
        hex::encode(hash)
    }
}

// =============================================================================
// Capital Eligibility (Output Object)
// =============================================================================

/// Capital eligibility result — the primary output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapitalEligibility {
    /// Strategy ID
    pub strategy_id: String,
    /// Eligibility status
    pub eligibility: EligibilityStatus,
    /// Constraints (bounds for future allocators)
    pub constraints: CapitalConstraints,
    /// Decision digest
    pub decision_digest: String,
    /// Decision timestamp
    pub decided_at: DateTime<Utc>,
}

impl CapitalEligibility {
    /// Create from eligibility decision.
    pub fn from_decision(decision: EligibilityDecision) -> Self {
        Self {
            strategy_id: decision.strategy_id,
            eligibility: decision.eligibility,
            constraints: decision.constraints,
            decision_digest: decision.decision_digest,
            decided_at: decision.decided_at,
        }
    }
}

// =============================================================================
// Eligibility Input (Strategy Metrics)
// =============================================================================

/// Strategy metrics for eligibility evaluation.
///
/// These come from paper evidence and attribution summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    /// Strategy ID
    pub strategy_id: String,
    /// Alpha score (mantissa)
    pub alpha_score_mantissa: i64,
    /// Alpha score exponent
    pub alpha_exponent: i8,
    /// Win rate (basis points)
    pub win_rate_bps: u32,
    /// Maximum drawdown (basis points)
    pub max_drawdown_bps: i64,
    /// Total paper trades
    pub paper_trades: u32,
    /// Paper session duration (nanoseconds)
    pub paper_duration_ns: i64,
    /// Target venue
    pub venue: Venue,
    /// Symbols traded
    pub symbols: Vec<String>,
}

// =============================================================================
// Eligibility Validator
// =============================================================================

/// Validator for capital eligibility.
pub struct EligibilityValidator {
    policy: EligibilityPolicy,
}

impl EligibilityValidator {
    /// Create a new validator with the given policy.
    pub fn new(policy: EligibilityPolicy) -> Self {
        Self { policy }
    }

    /// Create a validator with default policy.
    pub fn default_policy() -> Self {
        Self::new(EligibilityPolicy::default())
    }

    /// Create a validator with strict policy.
    pub fn strict() -> Self {
        Self::new(EligibilityPolicy::strict())
    }

    /// Evaluate capital eligibility.
    ///
    /// ## Required Inputs
    /// - `promotion`: Must be an accepted promotion decision
    /// - `metrics`: Strategy metrics from paper evidence
    /// - `g3_result`: G3 gate result (if `require_g3` is true)
    ///
    /// ## Mandatory Invariants
    /// - Promotion must be accepted
    /// - Paper evidence must exist
    /// - G3 must pass (if required)
    /// - Drawdown within limits
    /// - Alpha above threshold
    pub fn evaluate(
        &self,
        promotion: &PromotionDecision,
        metrics: &StrategyMetrics,
        g3_result: Option<&GateResult>,
    ) -> EligibilityDecision {
        let mut checks = Vec::new();
        let mut hard_failures = Vec::new();

        // === HARD CHECKS (failure = Ineligible) ===

        // Check 1: Promotion must be accepted
        if !promotion.accepted {
            let check = EligibilityCheck::fail_hard(
                "promotion_accepted",
                &format!(
                    "Promotion was rejected: {}",
                    promotion.rejection_reasons.join(", ")
                ),
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        } else {
            checks.push(EligibilityCheck::pass_hard(
                "promotion_accepted",
                "Promotion decision accepted",
            ));
        }

        // Check 2: Paper evidence must exist
        if let Some(paper_session_id) = &promotion.paper_session_id {
            checks.push(EligibilityCheck::pass_hard(
                "paper_evidence_exists",
                &format!("Paper evidence present: {}", paper_session_id),
            ));
        } else {
            let check = EligibilityCheck::fail_hard(
                "paper_evidence_exists",
                "No paper evidence linked to promotion",
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        }

        // Check 3: G3 gate (if required)
        if self.policy.require_g3 {
            match g3_result {
                Some(result) if result.passed => {
                    checks.push(EligibilityCheck::pass_hard(
                        "g3_gate_passed",
                        "G3 Robustness gate passed",
                    ));
                }
                Some(result) => {
                    let check = EligibilityCheck::fail_hard(
                        "g3_gate_passed",
                        &format!("G3 gate failed: {}", result.summary),
                    );
                    hard_failures.push(check.reason.clone());
                    checks.push(check);
                }
                None => {
                    let check = EligibilityCheck::fail_hard(
                        "g3_gate_passed",
                        "G3 gate result required but not provided",
                    );
                    hard_failures.push(check.reason.clone());
                    checks.push(check);
                }
            }
        }

        // Check 4: Paper duration
        if metrics.paper_duration_ns < self.policy.min_paper_duration_ns {
            let check = EligibilityCheck::fail_hard(
                "paper_duration",
                &format!(
                    "Paper duration {}ns < required {}ns",
                    metrics.paper_duration_ns, self.policy.min_paper_duration_ns
                ),
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        } else {
            checks.push(EligibilityCheck::pass_hard(
                "paper_duration",
                &format!(
                    "Paper duration {}ns meets requirement",
                    metrics.paper_duration_ns
                ),
            ));
        }

        // Check 5: Drawdown limit
        if metrics.max_drawdown_bps > self.policy.max_historical_drawdown_bps {
            let check = EligibilityCheck::fail_hard(
                "drawdown_limit",
                &format!(
                    "Max drawdown {}bps exceeds limit {}bps",
                    metrics.max_drawdown_bps, self.policy.max_historical_drawdown_bps
                ),
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        } else {
            checks.push(EligibilityCheck::pass_hard(
                "drawdown_limit",
                &format!("Max drawdown {}bps within limit", metrics.max_drawdown_bps),
            ));
        }

        // Check 6: Alpha score threshold
        if metrics.alpha_score_mantissa < self.policy.min_alpha_score_mantissa {
            let check = EligibilityCheck::fail_hard(
                "alpha_threshold",
                &format!(
                    "Alpha score {} < required {}",
                    metrics.alpha_score_mantissa, self.policy.min_alpha_score_mantissa
                ),
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        } else {
            checks.push(EligibilityCheck::pass_hard(
                "alpha_threshold",
                &format!(
                    "Alpha score {} meets threshold",
                    metrics.alpha_score_mantissa
                ),
            ));
        }

        // Check 7: Minimum paper trades
        if metrics.paper_trades < self.policy.min_paper_trades {
            let check = EligibilityCheck::fail_hard(
                "min_paper_trades",
                &format!(
                    "Paper trades {} < required {}",
                    metrics.paper_trades, self.policy.min_paper_trades
                ),
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        } else {
            checks.push(EligibilityCheck::pass_hard(
                "min_paper_trades",
                &format!("Paper trades {} meets requirement", metrics.paper_trades),
            ));
        }

        // Check 8: Win rate threshold
        if metrics.win_rate_bps < self.policy.min_win_rate_bps {
            let check = EligibilityCheck::fail_hard(
                "win_rate_threshold",
                &format!(
                    "Win rate {}bps < required {}bps",
                    metrics.win_rate_bps, self.policy.min_win_rate_bps
                ),
            );
            hard_failures.push(check.reason.clone());
            checks.push(check);
        } else {
            checks.push(EligibilityCheck::pass_hard(
                "win_rate_threshold",
                &format!("Win rate {}bps meets threshold", metrics.win_rate_bps),
            ));
        }

        // === SOFT CHECKS (failure = Conditional) ===

        // Check 9: Correlation (placeholder for Phase 13.2)
        if let Some(max_corr) = self.policy.max_correlation {
            // For now, we just note that correlation will be checked
            checks.push(EligibilityCheck::pass_soft(
                "correlation_check",
                &format!(
                    "Correlation limit {:.2} declared (enforcement in Phase 13.2)",
                    max_corr
                ),
            ));
        }

        // === BUILD DECISION ===
        // Note: Conditional status via soft_failures will be added in Phase 13.2
        // when correlation checks and other soft constraints are implemented.

        let eligibility = if !hard_failures.is_empty() {
            EligibilityStatus::Ineligible {
                reasons: hard_failures,
            }
        } else {
            EligibilityStatus::Eligible
        };

        // Build default constraints based on venue
        let constraints = CapitalConstraints::new().with_venue(metrics.venue.clone());

        let decision_digest =
            EligibilityDecision::compute_digest(&metrics.strategy_id, &eligibility, &checks);

        EligibilityDecision {
            schema_version: ELIGIBILITY_DECISION_SCHEMA.to_string(),
            strategy_id: metrics.strategy_id.clone(),
            eligibility,
            constraints,
            checks,
            decided_at: Utc::now(),
            promotion_decision_id: promotion.request_id.clone(),
            decision_digest,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::promotion::PROMOTION_DECISION_SCHEMA;

    fn make_accepted_promotion() -> PromotionDecision {
        PromotionDecision {
            schema_version: PROMOTION_DECISION_SCHEMA.to_string(),
            request_id: "req_001".to_string(),
            strategy_id: "funding_bias:1.0.0:abc123".to_string(),
            accepted: true,
            rejection_reasons: Vec::new(),
            checks: Vec::new(),
            decided_at: Utc::now(),
            tournament_id: Some("tournament_001".to_string()),
            paper_session_id: Some("paper_001".to_string()),
            decision_digest: "digest123".to_string(),
        }
    }

    fn make_rejected_promotion() -> PromotionDecision {
        PromotionDecision {
            schema_version: PROMOTION_DECISION_SCHEMA.to_string(),
            request_id: "req_002".to_string(),
            strategy_id: "bad_strategy:1.0.0:def456".to_string(),
            accepted: false,
            rejection_reasons: vec!["No paper evidence".to_string()],
            checks: Vec::new(),
            decided_at: Utc::now(),
            tournament_id: Some("tournament_001".to_string()),
            paper_session_id: None,
            decision_digest: "digest456".to_string(),
        }
    }

    fn make_good_metrics() -> StrategyMetrics {
        StrategyMetrics {
            strategy_id: "funding_bias:1.0.0:abc123".to_string(),
            alpha_score_mantissa: 2500, // 0.25 at exp -4
            alpha_exponent: -4,
            win_rate_bps: 5500,     // 55%
            max_drawdown_bps: 1000, // 10%
            paper_trades: 50,
            paper_duration_ns: 7_200_000_000_000, // 2 hours
            venue: Venue::BinancePerp,
            symbols: vec!["BTCUSDT".to_string()],
        }
    }

    fn make_g3_pass() -> GateResult {
        GateResult::new("G3").with_summary("All robustness checks passed")
    }

    fn make_g3_fail() -> GateResult {
        let mut result = GateResult::new("G3");
        result.passed = false;
        result.summary = "Connection loss test failed".to_string();
        result
    }

    #[test]
    fn test_eligible_g3_strategy() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let metrics = make_good_metrics();
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            decision.eligibility.is_eligible(),
            "Should be eligible: {:?}",
            decision.eligibility
        );
        assert!(decision.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn test_ineligible_without_g3() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let metrics = make_good_metrics();

        // No G3 result provided
        let decision = validator.evaluate(&promotion, &metrics, None);

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible without G3"
        );
        assert!(
            decision
                .checks
                .iter()
                .any(|c| c.name == "g3_gate_passed" && !c.passed)
        );
    }

    #[test]
    fn test_ineligible_g3_failed() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let metrics = make_good_metrics();
        let g3 = make_g3_fail();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible when G3 fails"
        );
    }

    #[test]
    fn test_ineligible_promotion_rejected() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_rejected_promotion();
        let metrics = make_good_metrics();
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible when promotion rejected"
        );
        assert!(
            decision
                .checks
                .iter()
                .any(|c| c.name == "promotion_accepted" && !c.passed)
        );
    }

    #[test]
    fn test_ineligible_drawdown_breach() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let mut metrics = make_good_metrics();
        metrics.max_drawdown_bps = 3000; // 30% > 20% limit
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible when drawdown exceeds limit"
        );
        assert!(
            decision
                .checks
                .iter()
                .any(|c| c.name == "drawdown_limit" && !c.passed)
        );
    }

    #[test]
    fn test_ineligible_low_alpha() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let mut metrics = make_good_metrics();
        metrics.alpha_score_mantissa = 500; // Below 1000 threshold
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible when alpha below threshold"
        );
        assert!(
            decision
                .checks
                .iter()
                .any(|c| c.name == "alpha_threshold" && !c.passed)
        );
    }

    #[test]
    fn test_ineligible_low_win_rate() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let mut metrics = make_good_metrics();
        metrics.win_rate_bps = 3000; // 30% < 40% threshold
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible when win rate below threshold"
        );
        assert!(
            decision
                .checks
                .iter()
                .any(|c| c.name == "win_rate_threshold" && !c.passed)
        );
    }

    #[test]
    fn test_ineligible_insufficient_trades() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let mut metrics = make_good_metrics();
        metrics.paper_trades = 5; // Below 10 minimum
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Should be ineligible with insufficient trades"
        );
        assert!(
            decision
                .checks
                .iter()
                .any(|c| c.name == "min_paper_trades" && !c.passed)
        );
    }

    #[test]
    fn test_decision_digest_deterministic() {
        let checks = vec![
            EligibilityCheck::pass_hard("check1", "reason1"),
            EligibilityCheck::fail_hard("check2", "reason2"),
        ];
        let eligibility = EligibilityStatus::Ineligible {
            reasons: vec!["reason2".to_string()],
        };

        let digest1 = EligibilityDecision::compute_digest("strategy_001", &eligibility, &checks);
        let digest2 = EligibilityDecision::compute_digest("strategy_001", &eligibility, &checks);

        assert_eq!(digest1, digest2, "Digest should be deterministic");
    }

    #[test]
    fn test_lenient_policy_allows_without_g3() {
        let validator = EligibilityValidator::new(EligibilityPolicy::lenient());
        let promotion = make_accepted_promotion();
        let metrics = make_good_metrics();

        // No G3 result, but lenient policy doesn't require it
        let decision = validator.evaluate(&promotion, &metrics, None);

        assert!(
            decision.eligibility.is_eligible(),
            "Lenient policy should allow without G3: {:?}",
            decision.eligibility
        );
    }

    #[test]
    fn test_strict_policy_higher_thresholds() {
        let validator = EligibilityValidator::strict();
        let promotion = make_accepted_promotion();
        let mut metrics = make_good_metrics();
        metrics.alpha_score_mantissa = 1500; // Passes default (1000) but fails strict (2000)
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));

        assert!(
            !decision.eligibility.allows_capital(),
            "Strict policy should reject alpha=1500"
        );
    }

    #[test]
    fn test_capital_eligibility_from_decision() {
        let validator = EligibilityValidator::default_policy();
        let promotion = make_accepted_promotion();
        let metrics = make_good_metrics();
        let g3 = make_g3_pass();

        let decision = validator.evaluate(&promotion, &metrics, Some(&g3));
        let eligibility = CapitalEligibility::from_decision(decision.clone());

        assert_eq!(eligibility.strategy_id, decision.strategy_id);
        assert_eq!(eligibility.decision_digest, decision.decision_digest);
        assert!(eligibility.eligibility.is_eligible());
    }

    #[test]
    fn test_constraints_builder() {
        let constraints = CapitalConstraints::new()
            .with_venue(Venue::BinancePerp)
            .with_max_notional(100_000.0)
            .with_max_daily_loss(5_000.0)
            .with_max_drawdown(10_000.0)
            .with_symbols(vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()]);

        assert_eq!(constraints.venue, Some(Venue::BinancePerp));
        assert_eq!(constraints.max_notional_mantissa, Some(10_000_000_000_000));
        assert_eq!(constraints.allowed_symbols.as_ref().unwrap().len(), 2);
    }
}
