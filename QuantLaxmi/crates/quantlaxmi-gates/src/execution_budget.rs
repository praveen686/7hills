//! # Execution Budget (Phase 14.1)
//!
//! Bridge allocation plans to execution constraints without market interaction.
//! Capital math becomes capital enforcement.
//!
//! ## Core Question
//! "How do allocation plans become runtime-enforceable budgets?"
//!
//! ## Hard Laws
//! 1. Budgets are derived from AllocationPlan — never invented
//! 2. Budget enforcement is deterministic and auditable
//! 3. No market interaction in Phase 14.1 (paper simulation only)
//! 4. All budget state changes are WAL events
//! 5. Budget violations are hard rejections (no soft limits)
//! 6. Every artifact has deterministic SHA-256 digest
//! 7. No wall-clock affects identity or digests
//!
//! ## NOT in Scope (Phase 14.1)
//! - Venue connections
//! - Order submission
//! - Fill processing from venues
//! - PnL accounting

use crate::capital_allocation::{AllocationPlan, StrategyAllocation};
use crate::capital_buckets::{BucketId, FixedPoint, SnapshotId, StrategyId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

// =============================================================================
// Schema Version
// =============================================================================

pub const EXECUTION_BUDGET_SCHEMA: &str = "execution_budget_v1.0";

// =============================================================================
// Identifier Types
// =============================================================================

/// Unique identifier for an execution budget.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BudgetId(pub String);

impl BudgetId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive budget ID deterministically from strategy and bucket.
    pub fn derive(strategy_id: &StrategyId, bucket_id: &BucketId, plan_digest: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"budget_id:");
        hasher.update(strategy_id.0.as_bytes());
        hasher.update(b":");
        hasher.update(bucket_id.0.as_bytes());
        hasher.update(b":");
        hasher.update(plan_digest.as_bytes());
        Self(format!("budget_{:x}", hasher.finalize())[..24].to_string())
    }
}

impl std::fmt::Display for BudgetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a budget delta (deterministic, not UUID).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeltaId(pub String);

impl DeltaId {
    /// Derive delta ID deterministically from budget, type, order, and timestamp.
    pub fn derive(
        budget_id: &BudgetId,
        delta_type: &DeltaType,
        order_id: Option<&str>,
        sequence: u64,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"delta_id:");
        hasher.update(budget_id.0.as_bytes());
        hasher.update(b":");
        hasher.update(delta_type.as_str().as_bytes());
        hasher.update(b":");
        if let Some(oid) = order_id {
            hasher.update(oid.as_bytes());
        }
        hasher.update(b":");
        hasher.update(sequence.to_le_bytes());
        Self(format!("delta_{:x}", hasher.finalize())[..24].to_string())
    }
}

impl std::fmt::Display for DeltaId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Budget Status
// =============================================================================

/// Current state of an execution budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetStatus {
    /// Normal operation
    Active,
    /// No available capital
    Exhausted,
    /// Manually paused
    Suspended,
    /// Plan superseded by new allocation
    Expired,
}

impl BudgetStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            BudgetStatus::Active => "Active",
            BudgetStatus::Exhausted => "Exhausted",
            BudgetStatus::Suspended => "Suspended",
            BudgetStatus::Expired => "Expired",
        }
    }
}

// =============================================================================
// Order Constraints
// =============================================================================

/// Per-order constraints derived from policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderConstraints {
    /// Max notional per order (derived from fraction of allocated).
    pub max_order_notional: FixedPoint,

    /// Max orders per time window.
    pub max_orders_per_window: u32,

    /// Window size in seconds for rate limiting.
    pub window_seconds: u64,

    /// Max open positions allowed.
    pub max_open_positions: u32,
}

// =============================================================================
// Execution Budget
// =============================================================================

/// Execution budget derived from AllocationPlan.
/// Enforces capital constraints at order time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionBudget {
    /// Unique budget identifier (deterministic).
    pub budget_id: BudgetId,

    /// Strategy this budget belongs to.
    pub strategy_id: StrategyId,

    /// Bucket this budget draws from.
    pub bucket_id: BucketId,

    /// Total allocated capital (from AllocationPlan).
    pub allocated_capital: FixedPoint,

    /// Capital reserved for open orders (not yet filled).
    pub reserved_capital: FixedPoint,

    /// Capital committed to open positions (filled orders).
    pub committed_capital: FixedPoint,

    /// Available for new orders: allocated - reserved - committed.
    pub available_capital: FixedPoint,

    /// Source allocation plan digest (provenance).
    pub allocation_plan_digest: String,

    /// Per-order constraints.
    pub order_constraints: OrderConstraints,

    /// Current budget state.
    pub status: BudgetStatus,

    /// Current delta sequence number.
    pub delta_sequence: u64,

    /// Timestamp of last update (event time, not wall-clock).
    pub last_update_ts_ns: i64,

    /// Deterministic digest.
    pub digest: String,

    /// Creation timestamp (event time).
    pub created_ts_ns: i64,
}

impl ExecutionBudget {
    /// Compute deterministic digest for this budget.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_digest(
        budget_id: &BudgetId,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
        allocated: &FixedPoint,
        reserved: &FixedPoint,
        committed: &FixedPoint,
        available: &FixedPoint,
        plan_digest: &str,
        status: BudgetStatus,
        delta_sequence: u64,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(EXECUTION_BUDGET_SCHEMA.as_bytes());
        hasher.update(budget_id.0.as_bytes());
        hasher.update(strategy_id.0.as_bytes());
        hasher.update(bucket_id.0.as_bytes());

        hasher.update(allocated.mantissa.to_le_bytes());
        hasher.update([allocated.exponent as u8]);
        hasher.update(reserved.mantissa.to_le_bytes());
        hasher.update([reserved.exponent as u8]);
        hasher.update(committed.mantissa.to_le_bytes());
        hasher.update([committed.exponent as u8]);
        hasher.update(available.mantissa.to_le_bytes());
        hasher.update([available.exponent as u8]);

        hasher.update(plan_digest.as_bytes());
        hasher.update(status.as_str().as_bytes());
        hasher.update(delta_sequence.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Check if budget has capacity for the given notional.
    pub fn has_capacity(&self, notional: &FixedPoint) -> bool {
        self.status == BudgetStatus::Active
            && notional.exponent == self.available_capital.exponent
            && notional.mantissa <= self.available_capital.mantissa
    }

    /// Recalculate available capital from components.
    fn recalculate_available(&mut self) {
        self.available_capital.mantissa = self.allocated_capital.mantissa
            - self.reserved_capital.mantissa
            - self.committed_capital.mantissa;

        // Update status based on availability
        if self.available_capital.mantissa <= 0 && self.status == BudgetStatus::Active {
            self.status = BudgetStatus::Exhausted;
        } else if self.available_capital.mantissa > 0 && self.status == BudgetStatus::Exhausted {
            self.status = BudgetStatus::Active;
        }
    }

    /// Update the digest after state change.
    fn update_digest(&mut self) {
        self.digest = Self::compute_digest(
            &self.budget_id,
            &self.strategy_id,
            &self.bucket_id,
            &self.allocated_capital,
            &self.reserved_capital,
            &self.committed_capital,
            &self.available_capital,
            &self.allocation_plan_digest,
            self.status,
            self.delta_sequence,
        );
    }
}

// =============================================================================
// Delta Type
// =============================================================================

/// Type of budget change event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaType {
    /// Initial allocation from plan.
    Allocation,
    /// Capital reserved for order.
    OrderOpen,
    /// Order filled (moves from reserved to committed).
    OrderFill,
    /// Order cancelled, capital returned to available.
    OrderCancel,
    /// Position closed, capital returned to available.
    PositionClose,
    /// New allocation plan applied.
    Rebalance,
    /// Budget expired (superseded by new plan).
    Expiration,
}

impl DeltaType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeltaType::Allocation => "Allocation",
            DeltaType::OrderOpen => "OrderOpen",
            DeltaType::OrderFill => "OrderFill",
            DeltaType::OrderCancel => "OrderCancel",
            DeltaType::PositionClose => "PositionClose",
            DeltaType::Rebalance => "Rebalance",
            DeltaType::Expiration => "Expiration",
        }
    }
}

// =============================================================================
// Budget Delta
// =============================================================================

/// Budget change event (WAL-bound, deterministic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetDelta {
    /// Deterministic delta identifier.
    pub delta_id: DeltaId,

    /// Budget this delta applies to.
    pub budget_id: BudgetId,

    /// Type of change.
    pub delta_type: DeltaType,

    /// Amount changed (positive = increase in utilized, negative = decrease).
    pub amount: FixedPoint,

    /// Human-readable reason.
    pub reason: String,

    /// Associated order ID (if applicable).
    pub order_id: Option<String>,

    /// Sequence number within this budget.
    pub sequence: u64,

    /// Event timestamp (nanoseconds since epoch).
    pub ts_ns: i64,

    /// Deterministic digest.
    pub digest: String,
}

impl BudgetDelta {
    /// Compute deterministic digest for this delta.
    pub fn compute_digest(
        delta_id: &DeltaId,
        budget_id: &BudgetId,
        delta_type: &DeltaType,
        amount: &FixedPoint,
        order_id: Option<&str>,
        sequence: u64,
        ts_ns: i64,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(delta_id.0.as_bytes());
        hasher.update(budget_id.0.as_bytes());
        hasher.update(delta_type.as_str().as_bytes());
        hasher.update(amount.mantissa.to_le_bytes());
        hasher.update([amount.exponent as u8]);
        if let Some(oid) = order_id {
            hasher.update(oid.as_bytes());
        }
        hasher.update(sequence.to_le_bytes());
        hasher.update(ts_ns.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Create a delta for a budget state change.
    pub fn new(
        budget_id: &BudgetId,
        delta_type: DeltaType,
        amount: FixedPoint,
        reason: String,
        order_id: Option<String>,
        sequence: u64,
        ts_ns: i64,
    ) -> Self {
        let delta_id = DeltaId::derive(budget_id, &delta_type, order_id.as_deref(), sequence);
        let delta_digest = Self::compute_digest(
            &delta_id,
            budget_id,
            &delta_type,
            &amount,
            order_id.as_deref(),
            sequence,
            ts_ns,
        );

        Self {
            delta_id,
            budget_id: budget_id.clone(),
            delta_type,
            amount,
            reason,
            order_id,
            sequence,
            ts_ns,
            digest: delta_digest,
        }
    }
}

// =============================================================================
// Budget Policy
// =============================================================================

/// Policy controlling budget derivation and constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPolicy {
    /// Fraction of allocation for max single order (mantissa, exp).
    /// e.g., 10_000 with exp -5 = 0.10 = 10%
    pub max_order_fraction_mantissa: i64,
    pub max_order_fraction_exponent: i8,

    /// Default max orders per window.
    pub default_max_orders_per_window: u32,

    /// Default window size in seconds.
    pub default_window_seconds: u64,

    /// Default max open positions.
    pub default_max_open_positions: u32,

    /// Policy version for audit.
    pub policy_version: String,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self {
            // 10% of allocation per order
            max_order_fraction_mantissa: 10_000,
            max_order_fraction_exponent: -5,
            // 100 orders per 5-minute window
            default_max_orders_per_window: 100,
            default_window_seconds: 300,
            // Max 10 open positions
            default_max_open_positions: 10,
            policy_version: "budget_policy_v1.0".to_string(),
        }
    }
}

impl BudgetPolicy {
    /// Create a conservative policy.
    pub fn conservative() -> Self {
        Self {
            // 5% per order
            max_order_fraction_mantissa: 5_000,
            max_order_fraction_exponent: -5,
            default_max_orders_per_window: 50,
            default_window_seconds: 300,
            default_max_open_positions: 5,
            policy_version: "budget_policy_conservative_v1.0".to_string(),
        }
    }

    /// Compute policy fingerprint.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.max_order_fraction_mantissa.to_le_bytes());
        hasher.update([self.max_order_fraction_exponent as u8]);
        hasher.update(self.default_max_orders_per_window.to_le_bytes());
        hasher.update(self.default_window_seconds.to_le_bytes());
        hasher.update(self.default_max_open_positions.to_le_bytes());
        hasher.update(self.policy_version.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Compute max order notional from allocated capital.
    pub fn compute_max_order_notional(&self, allocated: &FixedPoint) -> FixedPoint {
        // max_order = allocated * fraction
        // fraction = mantissa * 10^exponent
        let denom = 10i128.pow((-self.max_order_fraction_exponent) as u32);
        let max_mantissa = (allocated.mantissa * self.max_order_fraction_mantissa as i128) / denom;
        FixedPoint::new(max_mantissa, allocated.exponent)
    }
}

// =============================================================================
// Order Check Result
// =============================================================================

/// Result of pre-trade order check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderCheckResult {
    /// Whether the order is allowed.
    pub allowed: bool,

    /// Rejection reason if not allowed.
    pub rejection_reason: Option<String>,

    /// Current available capital.
    pub available_capital: FixedPoint,

    /// Applicable order constraints.
    pub order_constraints: OrderConstraints,
}

// =============================================================================
// Budget Error
// =============================================================================

/// Errors that can occur in budget operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BudgetError {
    #[error("Budget not found: strategy={0}, bucket={1}")]
    BudgetNotFound(String, String),

    #[error("Budget not active: status={0}")]
    BudgetNotActive(String),

    #[error("Insufficient capital: required={required}, available={available}")]
    InsufficientCapital { required: i128, available: i128 },

    #[error("Order exceeds max notional: order={order}, max={max}")]
    ExceedsMaxNotional { order: i128, max: i128 },

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Max positions exceeded: current={current}, max={max}")]
    MaxPositionsExceeded { current: u32, max: u32 },

    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Exponent mismatch: expected={expected}, got={got}")]
    ExponentMismatch { expected: i8, got: i8 },
}

// =============================================================================
// Rate Limit Tracker
// =============================================================================

/// Tracks order counts per deterministic time window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitTracker {
    /// Window size in nanoseconds.
    window_ns: i64,
    /// Orders per window bucket: window_start_ns -> count.
    windows: BTreeMap<i64, u32>,
}

impl RateLimitTracker {
    pub fn new(window_seconds: u64) -> Self {
        Self {
            window_ns: (window_seconds as i64) * 1_000_000_000,
            windows: BTreeMap::new(),
        }
    }

    /// Get the window bucket for a timestamp (floor division).
    fn window_bucket(&self, ts_ns: i64) -> i64 {
        (ts_ns / self.window_ns) * self.window_ns
    }

    /// Record an order at the given timestamp.
    pub fn record_order(&mut self, ts_ns: i64) {
        let bucket = self.window_bucket(ts_ns);
        *self.windows.entry(bucket).or_insert(0) += 1;
    }

    /// Get order count in the window containing ts_ns.
    pub fn get_count(&self, ts_ns: i64) -> u32 {
        let bucket = self.window_bucket(ts_ns);
        self.windows.get(&bucket).copied().unwrap_or(0)
    }

    /// Clean up old windows (keep only recent).
    pub fn cleanup(&mut self, current_ts_ns: i64, keep_windows: usize) {
        let current_bucket = self.window_bucket(current_ts_ns);
        let min_bucket = current_bucket - (keep_windows as i64 * self.window_ns);
        self.windows.retain(|&bucket, _| bucket >= min_bucket);
    }
}

// =============================================================================
// Budget Manager
// =============================================================================

/// Manages execution budgets derived from allocation plans.
pub struct BudgetManager {
    /// Budgets keyed by (strategy_id, bucket_id).
    budgets: BTreeMap<(StrategyId, BucketId), ExecutionBudget>,

    /// Policy governing budget creation and constraints.
    policy: BudgetPolicy,

    /// Rate limiters per budget.
    rate_limiters: BTreeMap<BudgetId, RateLimitTracker>,

    /// Open orders per budget (order_id -> reserved_notional).
    open_orders: BTreeMap<BudgetId, BTreeMap<String, FixedPoint>>,

    /// Open position count per budget.
    open_positions: BTreeMap<BudgetId, u32>,
}

impl BudgetManager {
    /// Create a new budget manager with the given policy.
    pub fn new(policy: BudgetPolicy) -> Self {
        Self {
            budgets: BTreeMap::new(),
            policy,
            rate_limiters: BTreeMap::new(),
            open_orders: BTreeMap::new(),
            open_positions: BTreeMap::new(),
        }
    }

    /// Create with default policy.
    pub fn with_default_policy() -> Self {
        Self::new(BudgetPolicy::default())
    }

    /// Get the policy.
    pub fn policy(&self) -> &BudgetPolicy {
        &self.policy
    }

    /// Apply an allocation plan, creating or updating budgets.
    /// Returns deltas describing all changes.
    pub fn apply_allocation_plan(
        &mut self,
        plan: &AllocationPlan,
        event_ts_ns: i64,
    ) -> Result<Vec<BudgetDelta>, BudgetError> {
        let mut deltas = vec![];

        for allocation in &plan.allocations {
            if allocation.is_zero() || allocation.skipped {
                continue;
            }

            let key = (allocation.strategy_id.clone(), plan.bucket_id.clone());

            // Check if budget already exists
            if let Some(existing) = self.budgets.get_mut(&key) {
                // Plan changed - expire old budget and create new
                if existing.allocation_plan_digest != plan.digest {
                    // Emit expiration delta for old budget
                    existing.delta_sequence += 1;
                    let exp_delta = BudgetDelta::new(
                        &existing.budget_id,
                        DeltaType::Expiration,
                        FixedPoint::zero(existing.allocated_capital.exponent),
                        "superseded_by_new_plan".to_string(),
                        None,
                        existing.delta_sequence,
                        event_ts_ns,
                    );
                    deltas.push(exp_delta);

                    existing.status = BudgetStatus::Expired;
                    existing.last_update_ts_ns = event_ts_ns;
                    existing.update_digest();
                }
            }

            // Handle new budget creation or replacement outside the mutable borrow
            let needs_new_budget = if let Some(existing) = self.budgets.get(&key) {
                existing.allocation_plan_digest != plan.digest
            } else {
                true
            };

            let needs_rebalance = if let Some(existing) = self.budgets.get(&key) {
                existing.allocation_plan_digest == plan.digest
                    && existing.allocated_capital.mantissa != allocation.assigned_capital.mantissa
            } else {
                false
            };

            if needs_new_budget {
                // Create new budget (will replace in map if exists)
                let (new_budget, alloc_delta) =
                    self.create_budget_from_allocation(plan, allocation, event_ts_ns)?;
                deltas.push(alloc_delta);

                // Update tracking structures
                self.rate_limiters.insert(
                    new_budget.budget_id.clone(),
                    RateLimitTracker::new(self.policy.default_window_seconds),
                );
                self.open_orders
                    .insert(new_budget.budget_id.clone(), BTreeMap::new());
                self.open_positions.insert(new_budget.budget_id.clone(), 0);

                self.budgets.insert(key, new_budget);
            } else if needs_rebalance {
                // Same plan - emit rebalance delta if amounts differ
                let existing = self.budgets.get_mut(&key).unwrap();
                let diff = FixedPoint::new(
                    allocation.assigned_capital.mantissa - existing.allocated_capital.mantissa,
                    allocation.assigned_capital.exponent,
                );

                existing.delta_sequence += 1;
                let rebal_delta = BudgetDelta::new(
                    &existing.budget_id,
                    DeltaType::Rebalance,
                    diff,
                    "allocation_amount_changed".to_string(),
                    None,
                    existing.delta_sequence,
                    event_ts_ns,
                );
                deltas.push(rebal_delta);

                existing.allocated_capital = allocation.assigned_capital;
                existing.recalculate_available();
                existing.last_update_ts_ns = event_ts_ns;
                existing.update_digest();
            }
        }

        Ok(deltas)
    }

    /// Create a new budget from an allocation.
    fn create_budget_from_allocation(
        &self,
        plan: &AllocationPlan,
        allocation: &StrategyAllocation,
        event_ts_ns: i64,
    ) -> Result<(ExecutionBudget, BudgetDelta), BudgetError> {
        let budget_id = BudgetId::derive(&allocation.strategy_id, &plan.bucket_id, &plan.digest);

        let exponent = allocation.assigned_capital.exponent;
        let max_order_notional = self
            .policy
            .compute_max_order_notional(&allocation.assigned_capital);

        let constraints = OrderConstraints {
            max_order_notional,
            max_orders_per_window: self.policy.default_max_orders_per_window,
            window_seconds: self.policy.default_window_seconds,
            max_open_positions: self.policy.default_max_open_positions,
        };

        let digest = ExecutionBudget::compute_digest(
            &budget_id,
            &allocation.strategy_id,
            &plan.bucket_id,
            &allocation.assigned_capital,
            &FixedPoint::zero(exponent),
            &FixedPoint::zero(exponent),
            &allocation.assigned_capital,
            &plan.digest,
            BudgetStatus::Active,
            1, // First delta
        );

        let budget = ExecutionBudget {
            budget_id: budget_id.clone(),
            strategy_id: allocation.strategy_id.clone(),
            bucket_id: plan.bucket_id.clone(),
            allocated_capital: allocation.assigned_capital,
            reserved_capital: FixedPoint::zero(exponent),
            committed_capital: FixedPoint::zero(exponent),
            available_capital: allocation.assigned_capital,
            allocation_plan_digest: plan.digest.clone(),
            order_constraints: constraints,
            status: BudgetStatus::Active,
            delta_sequence: 1,
            last_update_ts_ns: event_ts_ns,
            digest,
            created_ts_ns: event_ts_ns,
        };

        // Create allocation delta
        let delta_id = DeltaId::derive(&budget_id, &DeltaType::Allocation, None, 1);
        let delta_digest = BudgetDelta::compute_digest(
            &delta_id,
            &budget_id,
            &DeltaType::Allocation,
            &allocation.assigned_capital,
            None,
            1,
            event_ts_ns,
        );

        let delta = BudgetDelta {
            delta_id,
            budget_id,
            delta_type: DeltaType::Allocation,
            amount: allocation.assigned_capital,
            reason: "initial_allocation_from_plan".to_string(),
            order_id: None,
            sequence: 1,
            ts_ns: event_ts_ns,
            digest: delta_digest,
        };

        Ok((budget, delta))
    }

    /// Check if an order is allowed (pre-trade).
    pub fn check_order(
        &self,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
        order_notional: &FixedPoint,
        current_ts_ns: i64,
    ) -> OrderCheckResult {
        let key = (strategy_id.clone(), bucket_id.clone());

        let Some(budget) = self.budgets.get(&key) else {
            return OrderCheckResult {
                allowed: false,
                rejection_reason: Some(format!(
                    "Budget not found: strategy={}, bucket={}",
                    strategy_id, bucket_id
                )),
                available_capital: FixedPoint::zero(-2),
                order_constraints: OrderConstraints {
                    max_order_notional: FixedPoint::zero(-2),
                    max_orders_per_window: 0,
                    window_seconds: 0,
                    max_open_positions: 0,
                },
            };
        };

        // Check budget status
        if budget.status != BudgetStatus::Active {
            return OrderCheckResult {
                allowed: false,
                rejection_reason: Some(format!("Budget not active: {}", budget.status.as_str())),
                available_capital: budget.available_capital,
                order_constraints: budget.order_constraints.clone(),
            };
        }

        // Check exponent match
        if order_notional.exponent != budget.available_capital.exponent {
            return OrderCheckResult {
                allowed: false,
                rejection_reason: Some(format!(
                    "Exponent mismatch: order={}, budget={}",
                    order_notional.exponent, budget.available_capital.exponent
                )),
                available_capital: budget.available_capital,
                order_constraints: budget.order_constraints.clone(),
            };
        }

        // Check available capital
        if order_notional.mantissa > budget.available_capital.mantissa {
            return OrderCheckResult {
                allowed: false,
                rejection_reason: Some(format!(
                    "Insufficient capital: required={}, available={}",
                    order_notional.to_f64_display(),
                    budget.available_capital.to_f64_display()
                )),
                available_capital: budget.available_capital,
                order_constraints: budget.order_constraints.clone(),
            };
        }

        // Check max order notional
        if order_notional.mantissa > budget.order_constraints.max_order_notional.mantissa {
            return OrderCheckResult {
                allowed: false,
                rejection_reason: Some(format!(
                    "Exceeds max order notional: order={}, max={}",
                    order_notional.to_f64_display(),
                    budget.order_constraints.max_order_notional.to_f64_display()
                )),
                available_capital: budget.available_capital,
                order_constraints: budget.order_constraints.clone(),
            };
        }

        // Check rate limit
        if let Some(tracker) = self.rate_limiters.get(&budget.budget_id) {
            let count = tracker.get_count(current_ts_ns);
            if count >= budget.order_constraints.max_orders_per_window {
                return OrderCheckResult {
                    allowed: false,
                    rejection_reason: Some(format!(
                        "Rate limit exceeded: {} orders in window (max {})",
                        count, budget.order_constraints.max_orders_per_window
                    )),
                    available_capital: budget.available_capital,
                    order_constraints: budget.order_constraints.clone(),
                };
            }
        }

        // Check max positions
        let positions = self
            .open_positions
            .get(&budget.budget_id)
            .copied()
            .unwrap_or(0);
        if positions >= budget.order_constraints.max_open_positions {
            return OrderCheckResult {
                allowed: false,
                rejection_reason: Some(format!(
                    "Max positions exceeded: {} (max {})",
                    positions, budget.order_constraints.max_open_positions
                )),
                available_capital: budget.available_capital,
                order_constraints: budget.order_constraints.clone(),
            };
        }

        OrderCheckResult {
            allowed: true,
            rejection_reason: None,
            available_capital: budget.available_capital,
            order_constraints: budget.order_constraints.clone(),
        }
    }

    /// Reserve capital for an order.
    pub fn reserve_for_order(
        &mut self,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
        order_id: &str,
        order_notional: &FixedPoint,
        ts_ns: i64,
    ) -> Result<BudgetDelta, BudgetError> {
        let key = (strategy_id.clone(), bucket_id.clone());

        let budget = self.budgets.get_mut(&key).ok_or_else(|| {
            BudgetError::BudgetNotFound(strategy_id.0.clone(), bucket_id.0.clone())
        })?;

        if budget.status != BudgetStatus::Active {
            return Err(BudgetError::BudgetNotActive(
                budget.status.as_str().to_string(),
            ));
        }

        if order_notional.exponent != budget.available_capital.exponent {
            return Err(BudgetError::ExponentMismatch {
                expected: budget.available_capital.exponent,
                got: order_notional.exponent,
            });
        }

        if order_notional.mantissa > budget.available_capital.mantissa {
            return Err(BudgetError::InsufficientCapital {
                required: order_notional.mantissa,
                available: budget.available_capital.mantissa,
            });
        }

        // Reserve the capital
        budget.reserved_capital.mantissa += order_notional.mantissa;
        budget.recalculate_available();
        budget.delta_sequence += 1;
        budget.last_update_ts_ns = ts_ns;

        // Extract data for delta creation before other operations
        let budget_id = budget.budget_id.clone();
        let delta_sequence = budget.delta_sequence;

        // Track the order
        self.open_orders
            .entry(budget_id.clone())
            .or_default()
            .insert(order_id.to_string(), *order_notional);

        // Update rate limiter
        if let Some(tracker) = self.rate_limiters.get_mut(&budget_id) {
            tracker.record_order(ts_ns);
        }

        // Create delta
        let delta = BudgetDelta::new(
            &budget_id,
            DeltaType::OrderOpen,
            *order_notional,
            format!("order_reserved:{}", order_id),
            Some(order_id.to_string()),
            delta_sequence,
            ts_ns,
        );

        // Update budget digest
        let budget = self.budgets.get_mut(&key).unwrap();
        budget.update_digest();

        Ok(delta)
    }

    /// Release reserved capital when order is cancelled.
    pub fn release_order(
        &mut self,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
        order_id: &str,
        ts_ns: i64,
    ) -> Result<BudgetDelta, BudgetError> {
        let key = (strategy_id.clone(), bucket_id.clone());

        // First get the budget_id
        let budget_id = {
            let budget = self.budgets.get(&key).ok_or_else(|| {
                BudgetError::BudgetNotFound(strategy_id.0.clone(), bucket_id.0.clone())
            })?;
            budget.budget_id.clone()
        };

        // Find and remove the reserved amount from open_orders
        let notional = {
            let orders = self
                .open_orders
                .get_mut(&budget_id)
                .ok_or_else(|| BudgetError::OrderNotFound(order_id.to_string()))?;

            orders
                .remove(order_id)
                .ok_or_else(|| BudgetError::OrderNotFound(order_id.to_string()))?
        };

        // Now get mutable budget and update it
        let budget = self.budgets.get_mut(&key).unwrap();

        // Release the capital
        budget.reserved_capital.mantissa -= notional.mantissa;
        budget.recalculate_available();
        budget.delta_sequence += 1;
        budget.last_update_ts_ns = ts_ns;

        let delta_sequence = budget.delta_sequence;

        let delta = BudgetDelta::new(
            &budget_id,
            DeltaType::OrderCancel,
            FixedPoint::new(-notional.mantissa, notional.exponent),
            format!("order_cancelled:{}", order_id),
            Some(order_id.to_string()),
            delta_sequence,
            ts_ns,
        );

        budget.update_digest();

        Ok(delta)
    }

    /// Process order fill (move from reserved to committed).
    pub fn process_fill(
        &mut self,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
        order_id: &str,
        fill_notional: &FixedPoint,
        ts_ns: i64,
    ) -> Result<BudgetDelta, BudgetError> {
        let key = (strategy_id.clone(), bucket_id.clone());

        let budget = self.budgets.get_mut(&key).ok_or_else(|| {
            BudgetError::BudgetNotFound(strategy_id.0.clone(), bucket_id.0.clone())
        })?;

        // Move from reserved to committed
        budget.reserved_capital.mantissa -= fill_notional.mantissa;
        budget.committed_capital.mantissa += fill_notional.mantissa;
        budget.recalculate_available();
        budget.delta_sequence += 1;
        budget.last_update_ts_ns = ts_ns;

        // Extract data needed for delta and tracking operations
        let budget_id = budget.budget_id.clone();
        let delta_sequence = budget.delta_sequence;

        // Increment position count
        *self.open_positions.entry(budget_id.clone()).or_insert(0) += 1;

        // Remove from open orders if fully filled
        if let Some(orders) = self.open_orders.get_mut(&budget_id) {
            orders.remove(order_id);
        }

        let delta = BudgetDelta::new(
            &budget_id,
            DeltaType::OrderFill,
            *fill_notional,
            format!("order_filled:{}", order_id),
            Some(order_id.to_string()),
            delta_sequence,
            ts_ns,
        );

        // Update budget digest
        let budget = self.budgets.get_mut(&key).unwrap();
        budget.update_digest();

        Ok(delta)
    }

    /// Release capital when position is closed.
    pub fn release_position(
        &mut self,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
        position_notional: &FixedPoint,
        ts_ns: i64,
    ) -> Result<BudgetDelta, BudgetError> {
        let key = (strategy_id.clone(), bucket_id.clone());

        let budget = self.budgets.get_mut(&key).ok_or_else(|| {
            BudgetError::BudgetNotFound(strategy_id.0.clone(), bucket_id.0.clone())
        })?;

        // Release committed capital
        budget.committed_capital.mantissa -= position_notional.mantissa;
        budget.recalculate_available();
        budget.delta_sequence += 1;
        budget.last_update_ts_ns = ts_ns;

        // Extract data needed for operations
        let budget_id = budget.budget_id.clone();
        let delta_sequence = budget.delta_sequence;

        // Decrement position count
        if let Some(count) = self.open_positions.get_mut(&budget_id) {
            *count = count.saturating_sub(1);
        }

        let delta = BudgetDelta::new(
            &budget_id,
            DeltaType::PositionClose,
            FixedPoint::new(-position_notional.mantissa, position_notional.exponent),
            "position_closed".to_string(),
            None,
            delta_sequence,
            ts_ns,
        );

        // Update budget digest
        let budget = self.budgets.get_mut(&key).unwrap();
        budget.update_digest();

        Ok(delta)
    }

    /// Get a budget by strategy and bucket.
    pub fn get_budget(
        &self,
        strategy_id: &StrategyId,
        bucket_id: &BucketId,
    ) -> Option<&ExecutionBudget> {
        self.budgets.get(&(strategy_id.clone(), bucket_id.clone()))
    }

    /// Create a snapshot of all budgets.
    pub fn snapshot(&self, snapshot_id: SnapshotId, ts_ns: i64) -> BudgetSnapshot {
        let mut budgets: Vec<ExecutionBudget> = self.budgets.values().cloned().collect();
        budgets.sort_by(|a, b| (&a.strategy_id, &a.bucket_id).cmp(&(&b.strategy_id, &b.bucket_id)));

        let exponent = budgets
            .first()
            .map(|b| b.allocated_capital.exponent)
            .unwrap_or(-2);

        let total_allocated = FixedPoint::new(
            budgets.iter().map(|b| b.allocated_capital.mantissa).sum(),
            exponent,
        );
        let total_reserved = FixedPoint::new(
            budgets.iter().map(|b| b.reserved_capital.mantissa).sum(),
            exponent,
        );
        let total_committed = FixedPoint::new(
            budgets.iter().map(|b| b.committed_capital.mantissa).sum(),
            exponent,
        );
        let total_available = FixedPoint::new(
            budgets.iter().map(|b| b.available_capital.mantissa).sum(),
            exponent,
        );

        let digest = BudgetSnapshot::compute_digest(
            &snapshot_id,
            &budgets,
            &total_allocated,
            &total_reserved,
            &total_committed,
            &total_available,
        );

        BudgetSnapshot {
            snapshot_id,
            budgets,
            total_allocated,
            total_reserved,
            total_committed,
            total_available,
            digest,
            taken_at_ts_ns: ts_ns,
        }
    }
}

// =============================================================================
// Budget Snapshot
// =============================================================================

/// Audit snapshot of all budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    /// Snapshot identifier.
    pub snapshot_id: SnapshotId,

    /// All budgets at snapshot time.
    pub budgets: Vec<ExecutionBudget>,

    /// Total allocated across all budgets.
    pub total_allocated: FixedPoint,

    /// Total reserved across all budgets.
    pub total_reserved: FixedPoint,

    /// Total committed across all budgets.
    pub total_committed: FixedPoint,

    /// Total available across all budgets.
    pub total_available: FixedPoint,

    /// Deterministic digest.
    pub digest: String,

    /// Event timestamp when snapshot was taken.
    pub taken_at_ts_ns: i64,
}

impl BudgetSnapshot {
    /// Compute deterministic digest.
    pub fn compute_digest(
        snapshot_id: &SnapshotId,
        budgets: &[ExecutionBudget],
        total_allocated: &FixedPoint,
        total_reserved: &FixedPoint,
        total_committed: &FixedPoint,
        total_available: &FixedPoint,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(snapshot_id.0.as_bytes());

        for budget in budgets {
            hasher.update(budget.digest.as_bytes());
        }

        hasher.update(total_allocated.mantissa.to_le_bytes());
        hasher.update([total_allocated.exponent as u8]);
        hasher.update(total_reserved.mantissa.to_le_bytes());
        hasher.update([total_reserved.exponent as u8]);
        hasher.update(total_committed.mantissa.to_le_bytes());
        hasher.update([total_committed.exponent as u8]);
        hasher.update(total_available.mantissa.to_le_bytes());
        hasher.update([total_available.exponent as u8]);

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capital_allocation::{
        ALLOCATION_PLAN_SCHEMA, AllocationPlan, PlanId, StrategyAllocation,
    };
    use crate::portfolio_selector::IntentId;
    use chrono::Utc;

    fn create_test_allocation_plan(bucket_id: &str, strategies: &[(&str, i128)]) -> AllocationPlan {
        let exponent = -2;
        let allocations: Vec<StrategyAllocation> = strategies
            .iter()
            .map(|(s, amount)| StrategyAllocation {
                strategy_id: StrategyId::new(*s),
                assigned_capital: FixedPoint::new(*amount, exponent),
                max_notional: None,
                reasons: vec!["test".to_string()],
                skipped: false,
                skip_reason: None,
            })
            .collect();

        let total: i128 = strategies.iter().map(|(_, a)| a).sum();
        let bucket = BucketId::new(bucket_id);
        let snapshot_id = SnapshotId::new("snap_001");
        let intent_id = IntentId::new("intent_001");
        let policy_fp = "test_policy_fp";

        let digest = AllocationPlan::compute_digest(
            &bucket,
            &snapshot_id,
            &intent_id,
            &allocations,
            &FixedPoint::new(total, exponent),
            &FixedPoint::zero(exponent),
            &FixedPoint::zero(exponent),
            policy_fp,
        );

        AllocationPlan {
            plan_id: PlanId::new("plan_001"),
            schema_version: ALLOCATION_PLAN_SCHEMA.to_string(),
            bucket_id: bucket,
            snapshot_id,
            intent_id,
            allocations,
            total_allocated: FixedPoint::new(total, exponent),
            reserved: FixedPoint::zero(exponent),
            unallocated: FixedPoint::zero(exponent),
            unallocated_reasons: vec![],
            policy_fingerprint: policy_fp.to_string(),
            digest,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_budget_from_allocation_plan() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);
        let mut manager = BudgetManager::with_default_policy();

        let deltas = manager.apply_allocation_plan(&plan, 1_000_000_000).unwrap();

        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].delta_type, DeltaType::Allocation);

        let budget = manager
            .get_budget(&StrategyId::new("strat_a"), &BucketId::new("bucket_1"))
            .unwrap();

        assert_eq!(budget.allocated_capital.mantissa, 100_000_00);
        assert_eq!(budget.available_capital.mantissa, 100_000_00);
        assert_eq!(budget.reserved_capital.mantissa, 0);
        assert_eq!(budget.committed_capital.mantissa, 0);
        assert_eq!(budget.status, BudgetStatus::Active);
    }

    #[test]
    fn test_order_check_respects_constraints() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);
        let mut manager = BudgetManager::with_default_policy();
        manager.apply_allocation_plan(&plan, 1_000_000_000).unwrap();

        let strat = StrategyId::new("strat_a");
        let bucket = BucketId::new("bucket_1");

        // Order within limits - should be allowed
        let result = manager.check_order(
            &strat,
            &bucket,
            &FixedPoint::new(5_000_00, -2),
            1_000_000_000,
        );
        assert!(result.allowed);

        // Order exceeding max notional (10% of 100,000 = 10,000)
        let result = manager.check_order(
            &strat,
            &bucket,
            &FixedPoint::new(15_000_00, -2),
            1_000_000_000,
        );
        assert!(!result.allowed);
        assert!(
            result
                .rejection_reason
                .unwrap()
                .contains("max order notional")
        );

        // Order exceeding available
        let result = manager.check_order(
            &strat,
            &bucket,
            &FixedPoint::new(200_000_00, -2),
            1_000_000_000,
        );
        assert!(!result.allowed);
        assert!(result.rejection_reason.unwrap().contains("Insufficient"));
    }

    #[test]
    fn test_reserve_reduces_available() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);
        let mut manager = BudgetManager::with_default_policy();
        manager.apply_allocation_plan(&plan, 1_000_000_000).unwrap();

        let strat = StrategyId::new("strat_a");
        let bucket = BucketId::new("bucket_1");

        // Reserve 10,000
        let delta = manager
            .reserve_for_order(
                &strat,
                &bucket,
                "order_1",
                &FixedPoint::new(10_000_00, -2),
                2_000_000_000,
            )
            .unwrap();

        assert_eq!(delta.delta_type, DeltaType::OrderOpen);
        assert_eq!(delta.amount.mantissa, 10_000_00);

        let budget = manager.get_budget(&strat, &bucket).unwrap();
        assert_eq!(budget.reserved_capital.mantissa, 10_000_00);
        assert_eq!(budget.available_capital.mantissa, 90_000_00);
    }

    #[test]
    fn test_release_restores_available() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);
        let mut manager = BudgetManager::with_default_policy();
        manager.apply_allocation_plan(&plan, 1_000_000_000).unwrap();

        let strat = StrategyId::new("strat_a");
        let bucket = BucketId::new("bucket_1");

        // Reserve then release
        manager
            .reserve_for_order(
                &strat,
                &bucket,
                "order_1",
                &FixedPoint::new(10_000_00, -2),
                2_000_000_000,
            )
            .unwrap();

        let delta = manager
            .release_order(&strat, &bucket, "order_1", 3_000_000_000)
            .unwrap();

        assert_eq!(delta.delta_type, DeltaType::OrderCancel);

        let budget = manager.get_budget(&strat, &bucket).unwrap();
        assert_eq!(budget.reserved_capital.mantissa, 0);
        assert_eq!(budget.available_capital.mantissa, 100_000_00);
    }

    #[test]
    fn test_budget_exhausted_rejects_orders() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 20_000_00)]); // Small allocation
        let mut manager = BudgetManager::with_default_policy();
        manager.apply_allocation_plan(&plan, 1_000_000_000).unwrap();

        let strat = StrategyId::new("strat_a");
        let bucket = BucketId::new("bucket_1");

        // Reserve most of the budget (max order = 10% = 2,000, so we reserve twice)
        manager
            .reserve_for_order(
                &strat,
                &bucket,
                "order_1",
                &FixedPoint::new(2_000_00, -2),
                2_000_000_000,
            )
            .unwrap();
        manager
            .reserve_for_order(
                &strat,
                &bucket,
                "order_2",
                &FixedPoint::new(2_000_00, -2),
                3_000_000_000,
            )
            .unwrap();

        // Now only 16,000 available
        let result = manager.check_order(
            &strat,
            &bucket,
            &FixedPoint::new(20_000_00, -2),
            4_000_000_000,
        );
        assert!(!result.allowed);
        assert!(result.rejection_reason.unwrap().contains("Insufficient"));
    }

    #[test]
    fn test_rebalance_updates_budgets() {
        let plan1 = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);
        let mut manager = BudgetManager::with_default_policy();
        manager
            .apply_allocation_plan(&plan1, 1_000_000_000)
            .unwrap();

        // Create a new plan with different digest (different allocation)
        let plan2 = create_test_allocation_plan("bucket_1", &[("strat_a", 150_000_00)]);
        let deltas = manager
            .apply_allocation_plan(&plan2, 2_000_000_000)
            .unwrap();

        // Should have expiration + new allocation
        assert_eq!(deltas.len(), 2);
        assert!(deltas.iter().any(|d| d.delta_type == DeltaType::Expiration));
        assert!(deltas.iter().any(|d| d.delta_type == DeltaType::Allocation));

        let budget = manager
            .get_budget(&StrategyId::new("strat_a"), &BucketId::new("bucket_1"))
            .unwrap();
        assert_eq!(budget.allocated_capital.mantissa, 150_000_00);
    }

    #[test]
    fn test_budget_digest_deterministic() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);

        let mut manager1 = BudgetManager::with_default_policy();
        let mut manager2 = BudgetManager::with_default_policy();

        manager1
            .apply_allocation_plan(&plan, 1_000_000_000)
            .unwrap();
        manager2
            .apply_allocation_plan(&plan, 1_000_000_000)
            .unwrap();

        let budget1 = manager1
            .get_budget(&StrategyId::new("strat_a"), &BucketId::new("bucket_1"))
            .unwrap();
        let budget2 = manager2
            .get_budget(&StrategyId::new("strat_a"), &BucketId::new("bucket_1"))
            .unwrap();

        assert_eq!(budget1.digest, budget2.digest);
    }

    #[test]
    fn test_snapshot_digest_deterministic() {
        let plan = create_test_allocation_plan(
            "bucket_1",
            &[("strat_a", 100_000_00), ("strat_b", 50_000_00)],
        );

        let mut manager1 = BudgetManager::with_default_policy();
        let mut manager2 = BudgetManager::with_default_policy();

        manager1
            .apply_allocation_plan(&plan, 1_000_000_000)
            .unwrap();
        manager2
            .apply_allocation_plan(&plan, 1_000_000_000)
            .unwrap();

        let snap1 = manager1.snapshot(SnapshotId::new("snap_test"), 2_000_000_000);
        let snap2 = manager2.snapshot(SnapshotId::new("snap_test"), 2_000_000_000);

        assert_eq!(snap1.digest, snap2.digest);
    }

    #[test]
    fn test_delta_id_deterministic() {
        let budget_id = BudgetId::new("budget_test");
        let delta_type = DeltaType::OrderOpen;
        let order_id = Some("order_123");
        let sequence = 5u64;

        let id1 = DeltaId::derive(&budget_id, &delta_type, order_id, sequence);
        let id2 = DeltaId::derive(&budget_id, &delta_type, order_id, sequence);

        assert_eq!(id1.0, id2.0);

        // Different sequence should produce different ID
        let id3 = DeltaId::derive(&budget_id, &delta_type, order_id, sequence + 1);
        assert_ne!(id1.0, id3.0);
    }

    #[test]
    fn test_rate_limit_windowing_deterministic() {
        let mut tracker = RateLimitTracker::new(300); // 5 min window

        // Orders in same window
        let ts1 = 1_000_000_000_000i64; // ~1000 seconds
        let ts2 = 1_000_100_000_000i64; // ~1000.1 seconds (same 300s window)

        tracker.record_order(ts1);
        tracker.record_order(ts2);

        assert_eq!(tracker.get_count(ts1), 2);
        assert_eq!(tracker.get_count(ts2), 2);

        // Order in different window
        let ts3 = 1_300_000_000_000i64; // ~1300 seconds (different window)
        tracker.record_order(ts3);

        assert_eq!(tracker.get_count(ts3), 1);
        assert_eq!(tracker.get_count(ts1), 2); // Old window still has 2
    }

    #[test]
    fn test_fill_moves_reserved_to_committed() {
        let plan = create_test_allocation_plan("bucket_1", &[("strat_a", 100_000_00)]);
        let mut manager = BudgetManager::with_default_policy();
        manager.apply_allocation_plan(&plan, 1_000_000_000).unwrap();

        let strat = StrategyId::new("strat_a");
        let bucket = BucketId::new("bucket_1");

        // Reserve for order
        manager
            .reserve_for_order(
                &strat,
                &bucket,
                "order_1",
                &FixedPoint::new(10_000_00, -2),
                2_000_000_000,
            )
            .unwrap();

        // Process fill
        let delta = manager
            .process_fill(
                &strat,
                &bucket,
                "order_1",
                &FixedPoint::new(10_000_00, -2),
                3_000_000_000,
            )
            .unwrap();

        assert_eq!(delta.delta_type, DeltaType::OrderFill);

        let budget = manager.get_budget(&strat, &bucket).unwrap();
        assert_eq!(budget.reserved_capital.mantissa, 0);
        assert_eq!(budget.committed_capital.mantissa, 10_000_00);
        assert_eq!(budget.available_capital.mantissa, 90_000_00);
    }
}
