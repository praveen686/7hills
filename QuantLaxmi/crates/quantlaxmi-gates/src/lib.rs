//! # QuantLaxmi Gates
//!
//! Production gates for validating trading system integrity across the pipeline.
//!
//! ## Gate Hierarchy
//! - **G0 DataTruth**: Validate capture data integrity
//! - **G1 ReplayParity**: Ensure deterministic replay matches live
//! - **G2 BacktestCorrectness**: Validate backtest assumptions
//! - **G3 Robustness**: Stress testing and edge cases
//! - **G4 Deployability**: Pre-production readiness checks
//!
//! ## Usage
//! ```ignore
//! use quantlaxmi_gates::{G0DataTruth, G4Deployability, GateResult};
//!
//! // Validate capture data
//! let g0 = G0DataTruth::new(config);
//! let result = g0.validate_session(&session_dir)?;
//!
//! // Pre-deployment checks
//! let g4 = G4Deployability::new(config);
//! let result = g4.validate(&deployment_config)?;
//! ```

pub mod capital_allocation;
pub mod capital_buckets;
pub mod capital_eligibility;
pub mod execution_budget;
pub mod g0_data_truth;
pub mod g1_replay_parity;
pub mod g2_backtest_correctness;
pub mod g3_robustness;
pub mod g4_deployability;
pub mod portfolio_selector;
pub mod promotion;

pub use capital_allocation::{
    ALLOCATION_PLAN_SCHEMA, AllocationCheck, AllocationDecision, AllocationError, AllocationMode,
    AllocationPlan, AllocationPolicy, Allocator, PlanId, RebalancePolicy, SkipReason,
    StrategyAllocation, validate_plan,
};
pub use capital_buckets::{
    BUCKET_SCHEMA_VERSION, BucketBindingDecision, BucketConstraints, BucketEligibilityBinding,
    BucketError, BucketId, BucketRegistry, BucketSnapshot, CapitalBucket, Currency, FixedPoint,
    RiskClass, SnapshotId, StrategyId, Symbol,
};
pub use capital_eligibility::{
    CapitalConstraints as EligibilityConstraints, CapitalEligibility, ConditionType,
    ELIGIBILITY_DECISION_SCHEMA, EligibilityCheck, EligibilityCondition, EligibilityDecision,
    EligibilityPolicy, EligibilityStatus, EligibilityValidator, StrategyMetrics, TimeWindow, Venue,
};
pub use g0_data_truth::{G0Config, G0DataTruth};
pub use g1_replay_parity::{G1Config, G1ReplayParity};
pub use g2_backtest_correctness::{G2BacktestCorrectness, G2Config};
pub use g3_robustness::{G3Config, G3Robustness};
pub use g4_deployability::{G4Config, G4Deployability};
pub use portfolio_selector::{
    BucketSelectionResult, IntentId, OrderingRule, PORTFOLIO_INTENT_SCHEMA, PortfolioIntent,
    PortfolioPolicy, PortfolioRejection, PortfolioSelectionResult, PortfolioSelector,
    StrategyIntent, StrategyOrderingMetrics,
};
pub use promotion::{
    PROMOTION_DECISION_SCHEMA, PaperEvidence, PromotionCheck, PromotionDecision, PromotionPolicy,
    PromotionRequest, PromotionSource, PromotionValidator,
};
pub use execution_budget::{
    BudgetDelta, BudgetError, BudgetId, BudgetManager, BudgetPolicy, BudgetSnapshot, BudgetStatus,
    DeltaId, DeltaType, EXECUTION_BUDGET_SCHEMA, ExecutionBudget, OrderCheckResult,
    OrderConstraints, RateLimitTracker,
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Gate validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Gate identifier (G0, G1, G2, G3, G4)
    pub gate: String,
    /// Overall pass/fail
    pub passed: bool,
    /// Timestamp of validation
    pub timestamp: DateTime<Utc>,
    /// Individual check results
    pub checks: Vec<CheckResult>,
    /// Summary message
    pub summary: String,
    /// Validation duration in milliseconds
    pub duration_ms: u64,
}

impl GateResult {
    /// Create a new gate result.
    pub fn new(gate: impl Into<String>) -> Self {
        Self {
            gate: gate.into(),
            passed: true,
            timestamp: Utc::now(),
            checks: Vec::new(),
            summary: String::new(),
            duration_ms: 0,
        }
    }

    /// Add a check result.
    pub fn add_check(&mut self, check: CheckResult) {
        if !check.passed {
            self.passed = false;
        }
        self.checks.push(check);
    }

    /// Set the summary message.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }

    /// Count passed checks.
    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    /// Count failed checks.
    pub fn failed_count(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }
}

/// Individual check result within a gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    /// Check name
    pub name: String,
    /// Pass/fail
    pub passed: bool,
    /// Details/reason
    pub message: String,
    /// Optional metrics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
}

impl CheckResult {
    /// Create a passing check.
    pub fn pass(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            message: message.into(),
            metrics: None,
        }
    }

    /// Create a failing check.
    pub fn fail(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            message: message.into(),
            metrics: None,
        }
    }

    /// Add metrics to the check.
    pub fn with_metrics(mut self, metrics: serde_json::Value) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

/// Gate validation error.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Missing file: {0}")]
    MissingFile(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
