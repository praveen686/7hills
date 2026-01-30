//! # Capital Allocation (Phase 13.3)
//!
//! Convert portfolio intents into quantified allocation plans per bucket,
//! using deterministic policies and strict constraints, producing hashed decision artifacts.
//!
//! ## Core Question
//! "How much capital does each strategy receive?"
//!
//! ## NOT in Scope
//! - Trade sizing / order sizing
//! - Execution logic
//! - PnL accounting
//! - Market interaction
//!
//! ## Hard Architectural Laws
//! 1. Allocation is a pure function of inputs + policy (same inputs → identical digest)
//! 2. No re-evaluation of eligibility (consumes digests, does not recompute)
//! 3. Bucket constraints are absolute (never violated)
//! 4. Ordering is respected (cannot skip higher priority unless policy allows)
//! 5. No hidden state (all parameters in AllocationPolicy)
//! 6. Audit artifacts are first-class (deterministic SHA-256 digests)

use crate::capital_buckets::{
    BucketId, BucketSnapshot, CapitalBucket, FixedPoint, SnapshotId, StrategyId,
};
use crate::portfolio_selector::{IntentId, PortfolioIntent, StrategyIntent};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

// =============================================================================
// Schema Version
// =============================================================================

pub const ALLOCATION_PLAN_SCHEMA: &str = "allocation_plan_v1.0";

// =============================================================================
// Identifier Types
// =============================================================================

/// Unique identifier for an allocation plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlanId(pub String);

impl PlanId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for PlanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Allocation Mode
// =============================================================================

/// Mode determining how capital is distributed among strategies.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationMode {
    /// Split equally among eligible strategies (baseline_v1)
    /// Most auditable, least gameable.
    #[default]
    EqualSplit,

    /// Fill strategies in priority order up to their caps.
    /// Higher priority strategies get allocated first.
    PriorityFill,

    /// Proportional to normalized alpha score.
    /// Still deterministic given same inputs.
    ScoreProportional,
}

// =============================================================================
// Skip Reason
// =============================================================================

/// Explicit reasons why a strategy may be skipped during allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipReason {
    /// Strategy's cap is below minimum allocation threshold
    CapBelowMinimum,
    /// Insufficient remaining capital to meet minimum
    InsufficientCapital,
    /// Strategy explicitly excluded by policy
    PolicyExclusion,
}

// =============================================================================
// Allocation Policy
// =============================================================================

/// Policy controlling how capital quantities are assigned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPolicy {
    /// Mode determining distribution algorithm.
    pub mode: AllocationMode,

    /// Reserve ratio (mantissa, exponent).
    /// e.g., 10_000 with exp -5 = 0.10000 = 10%
    pub reserve_ratio_mantissa: i64,
    pub reserve_exponent: i8,

    /// Minimum allocation per strategy to be considered "allocated".
    /// Below this, assign 0 and record reason.
    pub min_allocation: FixedPoint,

    /// Maximum allocation per strategy (policy-level cap).
    /// Applied on top of bucket/eligibility caps.
    pub max_allocation_per_strategy: Option<FixedPoint>,

    /// Whether to allow skipping higher-priority strategies.
    /// If false, lower priority gets 0 if higher priority was skipped.
    pub allow_skip: bool,

    /// Explicit skip reasons allowed (only these are valid).
    pub skip_reasons_allowed: Vec<SkipReason>,

    /// Policy version for audit trails.
    pub policy_version: String,
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        Self {
            mode: AllocationMode::EqualSplit,
            // 10% reserve (10000 * 10^-5 = 0.10)
            reserve_ratio_mantissa: 10_000,
            reserve_exponent: -5,
            // Minimum 100.00 units (10000 * 10^-2)
            min_allocation: FixedPoint::new(10_000, -2),
            max_allocation_per_strategy: None,
            allow_skip: false,
            skip_reasons_allowed: vec![SkipReason::CapBelowMinimum],
            policy_version: "baseline_v1.0".to_string(),
        }
    }
}

impl AllocationPolicy {
    /// Create a conservative policy with higher reserves.
    pub fn conservative() -> Self {
        Self {
            mode: AllocationMode::EqualSplit,
            // 20% reserve
            reserve_ratio_mantissa: 20_000,
            reserve_exponent: -5,
            // Higher minimum
            min_allocation: FixedPoint::new(100_000, -2),
            max_allocation_per_strategy: Some(FixedPoint::new(50_000_000, -2)), // 500,000.00
            allow_skip: false,
            skip_reasons_allowed: vec![SkipReason::CapBelowMinimum],
            policy_version: "conservative_v1.0".to_string(),
        }
    }

    /// Create an aggressive policy with lower reserves.
    pub fn aggressive() -> Self {
        Self {
            mode: AllocationMode::PriorityFill,
            // 5% reserve
            reserve_ratio_mantissa: 5_000,
            reserve_exponent: -5,
            min_allocation: FixedPoint::new(5_000, -2),
            max_allocation_per_strategy: None,
            allow_skip: true,
            skip_reasons_allowed: vec![
                SkipReason::CapBelowMinimum,
                SkipReason::InsufficientCapital,
            ],
            policy_version: "aggressive_v1.0".to_string(),
        }
    }

    /// Compute deterministic fingerprint of this policy.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();

        // Mode
        let mode_str = match self.mode {
            AllocationMode::EqualSplit => "EqualSplit",
            AllocationMode::PriorityFill => "PriorityFill",
            AllocationMode::ScoreProportional => "ScoreProportional",
        };
        hasher.update(mode_str.as_bytes());

        // Reserve ratio
        hasher.update(self.reserve_ratio_mantissa.to_le_bytes());
        hasher.update([self.reserve_exponent as u8]);

        // Min allocation
        hasher.update(self.min_allocation.mantissa.to_le_bytes());
        hasher.update([self.min_allocation.exponent as u8]);

        // Max allocation (optional)
        if let Some(max) = &self.max_allocation_per_strategy {
            hasher.update([1u8]); // present marker
            hasher.update(max.mantissa.to_le_bytes());
            hasher.update([max.exponent as u8]);
        } else {
            hasher.update([0u8]); // absent marker
        }

        // Allow skip
        hasher.update([self.allow_skip as u8]);

        // Skip reasons (sorted for determinism)
        let mut reasons: Vec<&str> = self
            .skip_reasons_allowed
            .iter()
            .map(|r| match r {
                SkipReason::CapBelowMinimum => "CapBelowMinimum",
                SkipReason::InsufficientCapital => "InsufficientCapital",
                SkipReason::PolicyExclusion => "PolicyExclusion",
            })
            .collect();
        reasons.sort();
        for reason in reasons {
            hasher.update(reason.as_bytes());
        }

        // Version
        hasher.update(self.policy_version.as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Get reserve ratio as a fraction (for computation).
    /// Returns (numerator, denominator) where ratio = num/denom.
    pub fn reserve_fraction(&self) -> (i128, i128) {
        // If exponent is -5, then mantissa represents mantissa * 10^-5
        // e.g., 10000 with exp -5 = 0.10 = 10000/100000 = 1/10
        let denom = 10i128.pow((-self.reserve_exponent) as u32);
        (self.reserve_ratio_mantissa as i128, denom)
    }
}

// =============================================================================
// Strategy Allocation
// =============================================================================

/// Capital assigned to a single strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAllocation {
    /// Strategy identifier.
    pub strategy_id: StrategyId,

    /// Assigned capital (the quantity).
    pub assigned_capital: FixedPoint,

    /// Maximum notional allowed (derived from bucket + eligibility constraints).
    pub max_notional: Option<FixedPoint>,

    /// Reasons explaining assignment/cap/zero.
    pub reasons: Vec<String>,

    /// Whether this strategy was skipped.
    pub skipped: bool,

    /// Skip reason if skipped.
    pub skip_reason: Option<SkipReason>,
}

impl StrategyAllocation {
    /// Check if this allocation is effectively zero.
    pub fn is_zero(&self) -> bool {
        self.assigned_capital.is_zero()
    }
}

// =============================================================================
// Allocation Plan
// =============================================================================

/// The sole output of Phase 13.3.
/// One plan per bucket, containing quantified allocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPlan {
    /// Unique plan identifier.
    pub plan_id: PlanId,

    /// Schema version.
    pub schema_version: String,

    /// The bucket this plan applies to.
    pub bucket_id: BucketId,

    /// The snapshot this plan was derived from.
    pub snapshot_id: SnapshotId,

    /// The intent this plan fulfills.
    pub intent_id: IntentId,

    /// Per-strategy allocations (in priority order).
    pub allocations: Vec<StrategyAllocation>,

    /// Total allocated capital.
    pub total_allocated: FixedPoint,

    /// Reserved capital (held back by reserve ratio).
    pub reserved: FixedPoint,

    /// Unallocated remainder (after allocations + reserve).
    pub unallocated: FixedPoint,

    /// Unallocated reasons.
    pub unallocated_reasons: Vec<String>,

    /// Policy fingerprint (hash of policy parameters).
    pub policy_fingerprint: String,

    /// Deterministic digest of this plan.
    pub digest: String,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl AllocationPlan {
    /// Compute the deterministic digest for this plan.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_digest(
        bucket_id: &BucketId,
        snapshot_id: &SnapshotId,
        intent_id: &IntentId,
        allocations: &[StrategyAllocation],
        total_allocated: &FixedPoint,
        reserved: &FixedPoint,
        unallocated: &FixedPoint,
        policy_fingerprint: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        // Schema
        hasher.update(ALLOCATION_PLAN_SCHEMA.as_bytes());

        // Identifiers
        hasher.update(bucket_id.0.as_bytes());
        hasher.update(snapshot_id.0.as_bytes());
        hasher.update(intent_id.0.as_bytes());

        // Allocations (in order)
        for alloc in allocations {
            hasher.update(alloc.strategy_id.0.as_bytes());
            hasher.update(alloc.assigned_capital.mantissa.to_le_bytes());
            hasher.update([alloc.assigned_capital.exponent as u8]);
            hasher.update([alloc.skipped as u8]);
        }

        // Totals
        hasher.update(total_allocated.mantissa.to_le_bytes());
        hasher.update([total_allocated.exponent as u8]);
        hasher.update(reserved.mantissa.to_le_bytes());
        hasher.update([reserved.exponent as u8]);
        hasher.update(unallocated.mantissa.to_le_bytes());
        hasher.update([unallocated.exponent as u8]);

        // Policy fingerprint
        hasher.update(policy_fingerprint.as_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Allocation Check
// =============================================================================

/// Individual check result within allocation validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationCheck {
    /// Check name.
    pub name: String,

    /// Pass/fail.
    pub passed: bool,

    /// Explanation.
    pub reason: String,
}

impl AllocationCheck {
    /// Create a passing check.
    pub fn pass(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            reason: reason.into(),
        }
    }

    /// Create a failing check.
    pub fn fail(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            reason: reason.into(),
        }
    }
}

// =============================================================================
// Allocation Decision
// =============================================================================

/// Audit artifact for allocation validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationDecision {
    /// Overall acceptance.
    pub accepted: bool,

    /// Individual checks.
    pub checks: Vec<AllocationCheck>,

    /// Digest of the plan being validated.
    pub plan_digest: String,

    /// Deterministic digest of this decision.
    pub digest: String,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
}

impl AllocationDecision {
    /// Compute the deterministic digest.
    pub fn compute_digest(accepted: bool, checks: &[AllocationCheck], plan_digest: &str) -> String {
        let mut hasher = Sha256::new();

        hasher.update([accepted as u8]);

        for check in checks {
            hasher.update(check.name.as_bytes());
            hasher.update([check.passed as u8]);
            hasher.update(check.reason.as_bytes());
        }

        hasher.update(plan_digest.as_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Rebalance Policy
// =============================================================================

/// When to recompute allocation plans.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalancePolicy {
    /// Recompute at fixed intervals.
    FixedInterval { seconds: u64 },

    /// Recompute when a new bucket snapshot is created.
    #[default]
    OnNewSnapshot,

    /// Recompute when portfolio intent is updated.
    OnNewIntent,
}

// =============================================================================
// Allocation Error
// =============================================================================

/// Errors that can occur during allocation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AllocationError {
    #[error("Bucket not found in snapshot: {0}")]
    BucketNotFound(String),

    #[error("Intent bucket mismatch: intent={0}, expected={1}")]
    BucketMismatch(String, String),

    #[error("Exponent mismatch: expected {expected}, got {got}")]
    ExponentMismatch { expected: i8, got: i8 },

    #[error("Invalid reserve ratio: {0}")]
    InvalidReserveRatio(String),

    #[error("Ordering violation: {0}")]
    OrderingViolation(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}

// =============================================================================
// Allocator
// =============================================================================

/// The capital allocator.
/// Pure function: same inputs → identical plan digest.
pub struct Allocator {
    policy: AllocationPolicy,
}

impl Allocator {
    /// Create a new allocator with the given policy.
    pub fn new(policy: AllocationPolicy) -> Self {
        Self { policy }
    }

    /// Create an allocator with default policy.
    pub fn with_default_policy() -> Self {
        Self::new(AllocationPolicy::default())
    }

    /// Get the policy.
    pub fn policy(&self) -> &AllocationPolicy {
        &self.policy
    }

    /// Allocate capital for a single bucket given its intent.
    ///
    /// # Arguments
    /// * `snapshot` - Current registry snapshot (contains all buckets)
    /// * `intent` - Portfolio intent for the specific bucket
    /// * `strategy_caps` - Per-strategy caps from eligibility constraints
    ///
    /// # Returns
    /// `AllocationPlan` with deterministic digest
    pub fn allocate(
        &self,
        snapshot: &BucketSnapshot,
        intent: &PortfolioIntent,
        strategy_caps: &BTreeMap<StrategyId, Option<FixedPoint>>,
    ) -> Result<AllocationPlan, AllocationError> {
        // Find the bucket in the snapshot
        let bucket = snapshot
            .buckets
            .iter()
            .find(|b| b.bucket_id == intent.bucket_id)
            .ok_or_else(|| AllocationError::BucketNotFound(intent.bucket_id.0.clone()))?;

        self.allocate_for_bucket(snapshot, bucket, intent, strategy_caps)
    }

    /// Allocate capital for a specific bucket.
    fn allocate_for_bucket(
        &self,
        snapshot: &BucketSnapshot,
        bucket: &CapitalBucket,
        intent: &PortfolioIntent,
        strategy_caps: &BTreeMap<StrategyId, Option<FixedPoint>>,
    ) -> Result<AllocationPlan, AllocationError> {
        let available = &bucket.available_capital;
        let exponent = available.exponent;

        // Ensure min_allocation uses same exponent
        if self.policy.min_allocation.exponent != exponent {
            return Err(AllocationError::ExponentMismatch {
                expected: exponent,
                got: self.policy.min_allocation.exponent,
            });
        }

        // Step 1: Compute usable capital (apply reserve ratio)
        let (reserve_num, reserve_denom) = self.policy.reserve_fraction();
        let reserved_mantissa = (available.mantissa * reserve_num) / reserve_denom;
        let reserved = FixedPoint::new(reserved_mantissa, exponent);
        let usable_mantissa = available.mantissa - reserved_mantissa;

        // Step 2: Get strategies in priority order (from intent)
        let strategies: Vec<&StrategyIntent> = intent.strategy_order.iter().collect();
        let n = strategies.len();

        if n == 0 {
            // No strategies to allocate
            let plan_id = PlanId::new(uuid::Uuid::new_v4().to_string());
            let total_allocated = FixedPoint::zero(exponent);
            let unallocated = FixedPoint::new(usable_mantissa, exponent);
            let policy_fingerprint = self.policy.fingerprint();

            let digest = AllocationPlan::compute_digest(
                &bucket.bucket_id,
                &snapshot.snapshot_id,
                &intent.intent_id,
                &[],
                &total_allocated,
                &reserved,
                &unallocated,
                &policy_fingerprint,
            );

            return Ok(AllocationPlan {
                plan_id,
                schema_version: ALLOCATION_PLAN_SCHEMA.to_string(),
                bucket_id: bucket.bucket_id.clone(),
                snapshot_id: snapshot.snapshot_id.clone(),
                intent_id: intent.intent_id.clone(),
                allocations: vec![],
                total_allocated,
                reserved,
                unallocated,
                unallocated_reasons: vec!["no_strategies_in_intent".to_string()],
                policy_fingerprint,
                digest,
                created_at: Utc::now(),
            });
        }

        // Step 3: Allocate based on mode
        let allocations = match self.policy.mode {
            AllocationMode::EqualSplit => self.allocate_equal_split(
                usable_mantissa,
                exponent,
                &strategies,
                strategy_caps,
                bucket,
            )?,
            AllocationMode::PriorityFill => self.allocate_priority_fill(
                usable_mantissa,
                exponent,
                &strategies,
                strategy_caps,
                bucket,
            )?,
            AllocationMode::ScoreProportional => {
                // For v1, fall back to equal split (score proportional needs alpha scores)
                self.allocate_equal_split(
                    usable_mantissa,
                    exponent,
                    &strategies,
                    strategy_caps,
                    bucket,
                )?
            }
        };

        // Step 4: Compute totals
        let total_allocated_mantissa: i128 = allocations
            .iter()
            .map(|a| a.assigned_capital.mantissa)
            .sum();
        let total_allocated = FixedPoint::new(total_allocated_mantissa, exponent);

        let unallocated_mantissa = usable_mantissa - total_allocated_mantissa;
        let unallocated = FixedPoint::new(unallocated_mantissa, exponent);

        // Determine unallocated reasons
        let mut unallocated_reasons = vec![];
        if reserved_mantissa > 0 {
            unallocated_reasons.push("reserve_held".to_string());
        }
        if allocations.iter().any(|a| a.skipped) {
            unallocated_reasons.push("strategies_skipped".to_string());
        }
        if allocations
            .iter()
            .any(|a| a.max_notional.is_some() && !a.is_zero())
        {
            unallocated_reasons.push("caps_applied".to_string());
        }
        if unallocated_mantissa > 0 && unallocated_reasons.is_empty() {
            unallocated_reasons.push("rounding".to_string());
        }

        // Step 5: Compute plan digest
        let plan_id = PlanId::new(uuid::Uuid::new_v4().to_string());
        let policy_fingerprint = self.policy.fingerprint();

        let digest = AllocationPlan::compute_digest(
            &bucket.bucket_id,
            &snapshot.snapshot_id,
            &intent.intent_id,
            &allocations,
            &total_allocated,
            &reserved,
            &unallocated,
            &policy_fingerprint,
        );

        Ok(AllocationPlan {
            plan_id,
            schema_version: ALLOCATION_PLAN_SCHEMA.to_string(),
            bucket_id: bucket.bucket_id.clone(),
            snapshot_id: snapshot.snapshot_id.clone(),
            intent_id: intent.intent_id.clone(),
            allocations,
            total_allocated,
            reserved,
            unallocated,
            unallocated_reasons,
            policy_fingerprint,
            digest,
            created_at: Utc::now(),
        })
    }

    /// Equal split allocation.
    fn allocate_equal_split(
        &self,
        usable_mantissa: i128,
        exponent: i8,
        strategies: &[&StrategyIntent],
        strategy_caps: &BTreeMap<StrategyId, Option<FixedPoint>>,
        bucket: &CapitalBucket,
    ) -> Result<Vec<StrategyAllocation>, AllocationError> {
        let n = strategies.len() as i128;
        let target_per_strategy = usable_mantissa / n;

        let mut allocations = vec![];
        let mut any_higher_skipped = false;

        for strategy in strategies {
            // Get cap for this strategy
            let cap = self.get_strategy_cap(&strategy.strategy_id, strategy_caps, bucket);

            // Compute assigned amount
            let mut assigned_mantissa = target_per_strategy;
            let mut reasons = vec![];
            let mut skipped = false;
            let mut skip_reason = None;

            // Apply cap
            if let Some(cap_fp) = &cap
                && cap_fp.exponent == exponent
                && cap_fp.mantissa < assigned_mantissa
            {
                assigned_mantissa = cap_fp.mantissa;
                reasons.push(format!("capped_at_{}", cap_fp.to_f64_display()));
            }

            // Check minimum allocation threshold
            if assigned_mantissa < self.policy.min_allocation.mantissa {
                if self
                    .policy
                    .skip_reasons_allowed
                    .contains(&SkipReason::CapBelowMinimum)
                {
                    assigned_mantissa = 0;
                    skipped = true;
                    skip_reason = Some(SkipReason::CapBelowMinimum);
                    reasons.push("below_min_allocation_threshold".to_string());
                } else {
                    // Keep the small allocation
                    reasons.push("below_min_but_skip_not_allowed".to_string());
                }
            }

            // Check ordering constraint
            if !self.policy.allow_skip && any_higher_skipped && !skipped {
                // Higher priority was skipped, this one must also be skipped
                assigned_mantissa = 0;
                skipped = true;
                skip_reason = Some(SkipReason::PolicyExclusion);
                reasons.push("higher_priority_skipped_no_skip_allowed".to_string());
            }

            if skipped {
                any_higher_skipped = true;
            }

            if reasons.is_empty() {
                reasons.push("equal_split".to_string());
            }

            allocations.push(StrategyAllocation {
                strategy_id: strategy.strategy_id.clone(),
                assigned_capital: FixedPoint::new(assigned_mantissa, exponent),
                max_notional: cap,
                reasons,
                skipped,
                skip_reason,
            });
        }

        Ok(allocations)
    }

    /// Priority fill allocation.
    fn allocate_priority_fill(
        &self,
        usable_mantissa: i128,
        exponent: i8,
        strategies: &[&StrategyIntent],
        strategy_caps: &BTreeMap<StrategyId, Option<FixedPoint>>,
        bucket: &CapitalBucket,
    ) -> Result<Vec<StrategyAllocation>, AllocationError> {
        let mut remaining = usable_mantissa;
        let mut allocations = vec![];
        let mut any_higher_skipped = false;

        for strategy in strategies {
            // Get cap for this strategy
            let cap = self.get_strategy_cap(&strategy.strategy_id, strategy_caps, bucket);

            let mut reasons = vec![];
            let mut skipped = false;
            let mut skip_reason = None;

            // Determine max we can assign
            let max_assignable = if let Some(cap_fp) = &cap {
                if cap_fp.exponent == exponent {
                    cap_fp.mantissa.min(remaining)
                } else {
                    remaining
                }
            } else {
                remaining
            };

            let mut assigned_mantissa = max_assignable;

            // Check minimum
            if assigned_mantissa < self.policy.min_allocation.mantissa
                && (self
                    .policy
                    .skip_reasons_allowed
                    .contains(&SkipReason::InsufficientCapital)
                    || self
                        .policy
                        .skip_reasons_allowed
                        .contains(&SkipReason::CapBelowMinimum))
            {
                assigned_mantissa = 0;
                skipped = true;
                skip_reason = Some(SkipReason::InsufficientCapital);
                reasons.push("insufficient_for_min_allocation".to_string());
            }

            // Check ordering constraint
            if !self.policy.allow_skip && any_higher_skipped && !skipped {
                assigned_mantissa = 0;
                skipped = true;
                skip_reason = Some(SkipReason::PolicyExclusion);
                reasons.push("higher_priority_skipped_no_skip_allowed".to_string());
            }

            if skipped {
                any_higher_skipped = true;
            }

            if cap.is_some() && !skipped {
                reasons.push("priority_fill_capped".to_string());
            } else if !skipped {
                reasons.push("priority_fill".to_string());
            }

            remaining -= assigned_mantissa;

            allocations.push(StrategyAllocation {
                strategy_id: strategy.strategy_id.clone(),
                assigned_capital: FixedPoint::new(assigned_mantissa, exponent),
                max_notional: cap,
                reasons,
                skipped,
                skip_reason,
            });
        }

        Ok(allocations)
    }

    /// Get the effective cap for a strategy.
    fn get_strategy_cap(
        &self,
        strategy_id: &StrategyId,
        strategy_caps: &BTreeMap<StrategyId, Option<FixedPoint>>,
        bucket: &CapitalBucket,
    ) -> Option<FixedPoint> {
        // Start with eligibility cap
        let eligibility_cap = strategy_caps.get(strategy_id).cloned().flatten();

        // Get bucket cap
        let bucket_cap = bucket.constraints.max_notional_per_strategy;

        // Combine all caps: policy, eligibility, bucket
        let caps: Vec<FixedPoint> = [
            eligibility_cap,
            bucket_cap,
            self.policy.max_allocation_per_strategy,
        ]
        .into_iter()
        .flatten()
        .collect();

        if caps.is_empty() {
            return None;
        }

        // Find minimum cap (assuming same exponent for simplicity)
        // In production, would need proper exponent handling
        caps.into_iter().min_by_key(|fp| fp.mantissa)
    }
}

// =============================================================================
// Validation
// =============================================================================

/// Validate an allocation plan against constraints.
pub fn validate_plan(
    snapshot: &BucketSnapshot,
    intent: &PortfolioIntent,
    policy: &AllocationPolicy,
    plan: &AllocationPlan,
) -> AllocationDecision {
    let mut checks = vec![];
    let mut all_passed = true;

    // Find the bucket in snapshot
    let bucket = snapshot
        .buckets
        .iter()
        .find(|b| b.bucket_id == plan.bucket_id);

    // Check 1: Bucket exists and matches
    if let Some(bucket) = bucket {
        if plan.bucket_id == intent.bucket_id {
            checks.push(AllocationCheck::pass(
                "bucket_match",
                "Plan bucket matches snapshot and intent",
            ));
        } else {
            checks.push(AllocationCheck::fail(
                "bucket_match",
                format!(
                    "Bucket mismatch: plan={}, intent={}",
                    plan.bucket_id, intent.bucket_id
                ),
            ));
            all_passed = false;
        }

        // Check 2: Snapshot match
        if plan.snapshot_id == snapshot.snapshot_id {
            checks.push(AllocationCheck::pass(
                "snapshot_match",
                "Plan derived from correct snapshot",
            ));
        } else {
            checks.push(AllocationCheck::fail(
                "snapshot_match",
                format!(
                    "Snapshot mismatch: plan={}, expected={}",
                    plan.snapshot_id.0, snapshot.snapshot_id.0
                ),
            ));
            all_passed = false;
        }

        // Check 3: Intent match
        if plan.intent_id == intent.intent_id {
            checks.push(AllocationCheck::pass(
                "intent_match",
                "Plan fulfills correct intent",
            ));
        } else {
            checks.push(AllocationCheck::fail(
                "intent_match",
                format!(
                    "Intent mismatch: plan={}, expected={}",
                    plan.intent_id, intent.intent_id
                ),
            ));
            all_passed = false;
        }

        // Check 4: Respects max_concurrent_strategies
        let active_count = plan.allocations.iter().filter(|a| !a.is_zero()).count() as u32;
        let max_concurrent = bucket
            .constraints
            .max_concurrent_strategies
            .unwrap_or(u32::MAX);
        if active_count <= max_concurrent {
            checks.push(AllocationCheck::pass(
                "respects_max_concurrent_strategies",
                format!(
                    "Active strategies ({}) within limit ({})",
                    active_count,
                    if max_concurrent == u32::MAX {
                        "unlimited".to_string()
                    } else {
                        max_concurrent.to_string()
                    }
                ),
            ));
        } else {
            checks.push(AllocationCheck::fail(
                "respects_max_concurrent_strategies",
                format!(
                    "Active strategies ({}) exceeds limit ({})",
                    active_count, max_concurrent
                ),
            ));
            all_passed = false;
        }

        // Check 5: Total allocated does not exceed available
        let available = &bucket.available_capital;
        if plan.total_allocated.exponent == available.exponent
            && plan.total_allocated.mantissa <= available.mantissa
        {
            checks.push(AllocationCheck::pass(
                "within_available_capital",
                format!(
                    "Allocated {} <= available {}",
                    plan.total_allocated.to_f64_display(),
                    available.to_f64_display()
                ),
            ));
        } else {
            checks.push(AllocationCheck::fail(
                "within_available_capital",
                format!(
                    "Allocated {} exceeds available {}",
                    plan.total_allocated.to_f64_display(),
                    available.to_f64_display()
                ),
            ));
            all_passed = false;
        }

        // Check 6: Reserve applied correctly
        let (reserve_num, reserve_denom) = policy.reserve_fraction();
        let expected_reserve = (available.mantissa * reserve_num) / reserve_denom;
        if plan.reserved.mantissa == expected_reserve {
            checks.push(AllocationCheck::pass(
                "reserve_applied",
                format!(
                    "Reserve {} correctly applied",
                    plan.reserved.to_f64_display()
                ),
            ));
        } else {
            checks.push(AllocationCheck::fail(
                "reserve_applied",
                format!(
                    "Reserve mismatch: got {}, expected {}",
                    plan.reserved.mantissa, expected_reserve
                ),
            ));
            all_passed = false;
        }
    } else {
        checks.push(AllocationCheck::fail(
            "bucket_exists",
            format!("Bucket {} not found in snapshot", plan.bucket_id),
        ));
        all_passed = false;
    }

    // Check 7: Respects ordering (no lower priority allocated if higher skipped, unless allowed)
    let mut ordering_ok = true;
    let mut saw_skip = false;
    for alloc in &plan.allocations {
        if alloc.skipped {
            saw_skip = true;
        } else if saw_skip && !policy.allow_skip && !alloc.is_zero() {
            ordering_ok = false;
            break;
        }
    }
    if ordering_ok {
        checks.push(AllocationCheck::pass(
            "respects_ordering",
            "Allocation respects priority ordering",
        ));
    } else {
        checks.push(AllocationCheck::fail(
            "respects_ordering",
            "Lower priority allocated while higher priority skipped (allow_skip=false)",
        ));
        all_passed = false;
    }

    // Check 8: Policy fingerprint matches
    let expected_fingerprint = policy.fingerprint();
    if plan.policy_fingerprint == expected_fingerprint {
        checks.push(AllocationCheck::pass(
            "policy_fingerprint",
            "Policy fingerprint matches",
        ));
    } else {
        checks.push(AllocationCheck::fail(
            "policy_fingerprint",
            "Policy fingerprint mismatch",
        ));
        all_passed = false;
    }

    // Check 9: Deterministic digest
    let expected_digest = AllocationPlan::compute_digest(
        &plan.bucket_id,
        &plan.snapshot_id,
        &plan.intent_id,
        &plan.allocations,
        &plan.total_allocated,
        &plan.reserved,
        &plan.unallocated,
        &plan.policy_fingerprint,
    );
    if plan.digest == expected_digest {
        checks.push(AllocationCheck::pass(
            "deterministic_digest",
            "Plan digest is deterministic",
        ));
    } else {
        checks.push(AllocationCheck::fail(
            "deterministic_digest",
            "Plan digest does not match recomputed value",
        ));
        all_passed = false;
    }

    // Compute decision digest
    let decision_digest = AllocationDecision::compute_digest(all_passed, &checks, &plan.digest);

    AllocationDecision {
        accepted: all_passed,
        checks,
        plan_digest: plan.digest.clone(),
        digest: decision_digest,
        created_at: Utc::now(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capital_buckets::{BucketConstraints, Currency, RiskClass};
    use crate::capital_eligibility::Venue;

    fn create_test_bucket(id: &str, available: i128) -> CapitalBucket {
        CapitalBucket::new(
            BucketId::new(id),
            Venue::BinancePerp,
            Currency::USDT,
            FixedPoint::new(available, -2),
            BucketConstraints::new(RiskClass::Moderate),
        )
    }

    fn create_test_snapshot(buckets: Vec<CapitalBucket>) -> BucketSnapshot {
        BucketSnapshot::new(SnapshotId::new("snap_001"), buckets, vec![])
    }

    fn create_test_intent(bucket_id: &BucketId, strategies: Vec<&str>) -> PortfolioIntent {
        let strategy_order: Vec<StrategyIntent> = strategies
            .into_iter()
            .enumerate()
            .map(|(i, s)| StrategyIntent {
                strategy_id: StrategyId::new(s),
                priority: (i + 1) as u32,
                eligibility_digest: format!("elig_digest_{}", s),
                binding_digest: format!("bind_digest_{}", s),
            })
            .collect();

        PortfolioIntent {
            intent_id: IntentId::new("intent_001"),
            schema_version: "portfolio_intent_v1.0".to_string(),
            bucket_id: bucket_id.clone(),
            strategy_order,
            constraints: BucketConstraints::new(RiskClass::Moderate),
            policy_version: "test_v1.0".to_string(),
            digest: "test_intent_digest".to_string(),
            created_at: Utc::now(),
        }
    }

    #[test]
    fn test_allocation_digest_deterministic() {
        let bucket = create_test_bucket("bucket_1", 100_000_000); // 1,000,000.00
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);
        let policy = AllocationPolicy::default();
        let caps = BTreeMap::new();

        let allocator = Allocator::new(policy);

        // Allocate twice
        let plan1 = allocator.allocate(&snapshot, &intent, &caps).unwrap();
        let plan2 = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // Digests must match (deterministic)
        assert_eq!(plan1.digest, plan2.digest);
    }

    #[test]
    fn test_policy_fingerprint_deterministic() {
        let policy1 = AllocationPolicy::default();
        let policy2 = AllocationPolicy::default();

        assert_eq!(policy1.fingerprint(), policy2.fingerprint());

        let policy3 = AllocationPolicy::conservative();
        assert_ne!(policy1.fingerprint(), policy3.fingerprint());
    }

    #[test]
    fn test_respects_bucket_max_concurrent_strategies() {
        let mut bucket = create_test_bucket("bucket_1", 100_000_000);
        bucket.constraints = BucketConstraints::new(RiskClass::Moderate).with_max_strategies(2);

        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(
            &bucket.bucket_id,
            vec!["strat_a", "strat_b", "strat_c", "strat_d"],
        );
        let policy = AllocationPolicy::default();
        let caps = BTreeMap::new();

        let allocator = Allocator::new(policy.clone());
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // Validation should check if active exceeds limit
        let decision = validate_plan(&snapshot, &intent, &policy, &plan);

        // The allocator allocated all 4, but validator checks the constraint
        let active_count = plan.allocations.iter().filter(|a| !a.is_zero()).count();
        if active_count > 2 {
            assert!(!decision.accepted);
        }
    }

    #[test]
    fn test_respects_caps_bucket_and_eligibility() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);

        let mut caps = BTreeMap::new();
        // Cap strat_a at 100,000.00
        caps.insert(
            StrategyId::new("strat_a"),
            Some(FixedPoint::new(10_000_000, -2)),
        );

        let policy = AllocationPolicy::default();
        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // strat_a should be capped at 100,000
        let alloc_a = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_a")
            .unwrap();
        assert!(alloc_a.assigned_capital.mantissa <= 10_000_000);
        assert!(alloc_a.reasons.iter().any(|r| r.contains("capped")));
    }

    #[test]
    fn test_ordering_no_skip_when_disallowed() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b", "strat_c"]);

        let mut caps = BTreeMap::new();
        // Cap strat_a below minimum so it gets skipped
        caps.insert(
            StrategyId::new("strat_a"),
            Some(FixedPoint::new(50, -2)), // 0.50 - below default min of 100.00
        );

        let policy = AllocationPolicy {
            allow_skip: false,
            ..Default::default()
        };

        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // strat_a is skipped due to cap below minimum
        let alloc_a = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_a")
            .unwrap();
        assert!(alloc_a.skipped);

        // Since allow_skip=false, strat_b and strat_c should also be 0
        let alloc_b = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_b")
            .unwrap();
        let alloc_c = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_c")
            .unwrap();

        assert!(alloc_b.is_zero());
        assert!(alloc_c.is_zero());
    }

    #[test]
    fn test_skip_allowed_records_reason() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);

        let mut caps = BTreeMap::new();
        caps.insert(
            StrategyId::new("strat_a"),
            Some(FixedPoint::new(50, -2)), // Below minimum
        );

        let policy = AllocationPolicy {
            allow_skip: true,
            ..Default::default()
        };

        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // strat_a should be skipped with reason
        let alloc_a = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_a")
            .unwrap();
        assert!(alloc_a.skipped);
        assert!(alloc_a.skip_reason.is_some());
        assert!(alloc_a.reasons.iter().any(|r| r.contains("min_allocation")));

        // strat_b should still get allocation (allow_skip=true)
        let alloc_b = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_b")
            .unwrap();
        assert!(!alloc_b.is_zero());
    }

    #[test]
    fn test_equal_split_basic() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);
        let policy = AllocationPolicy::default();
        let caps = BTreeMap::new();

        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // With default 10% reserve, usable = 900,000.00
        // Split between 2 strategies = 450,000.00 each
        let alloc_a = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_a")
            .unwrap();
        let alloc_b = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_b")
            .unwrap();

        assert_eq!(alloc_a.assigned_capital.mantissa, 45_000_000);
        assert_eq!(alloc_b.assigned_capital.mantissa, 45_000_000);
    }

    #[test]
    fn test_min_allocation_threshold() {
        let bucket = create_test_bucket("bucket_1", 15_000); // 150.00
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b", "strat_c"]);
        let policy = AllocationPolicy::default(); // min = 100.00

        let caps = BTreeMap::new();
        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // Usable = 150 * 0.9 = 135.00
        // Split 3 ways = 45.00 each, below min of 100.00
        // All should be skipped
        for alloc in &plan.allocations {
            assert!(alloc.skipped || alloc.is_zero());
        }
    }

    #[test]
    fn test_reserve_ratio_applied() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a"]);
        let policy = AllocationPolicy::default(); // 10% reserve
        let caps = BTreeMap::new();

        let allocator = Allocator::new(policy.clone());
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // Reserve should be 10% of 1,000,000 = 100,000.00
        assert_eq!(plan.reserved.mantissa, 10_000_000);

        // Validate reserve is correct
        let decision = validate_plan(&snapshot, &intent, &policy, &plan);
        let reserve_check = decision
            .checks
            .iter()
            .find(|c| c.name == "reserve_applied")
            .unwrap();
        assert!(reserve_check.passed);
    }

    #[test]
    fn test_unallocated_computed_correctly() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);

        let mut caps = BTreeMap::new();
        // Cap both strategies low
        caps.insert(
            StrategyId::new("strat_a"),
            Some(FixedPoint::new(20_000_000, -2)), // 200,000.00
        );
        caps.insert(
            StrategyId::new("strat_b"),
            Some(FixedPoint::new(20_000_000, -2)), // 200,000.00
        );

        let policy = AllocationPolicy::default();
        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // Total available = 1,000,000
        // Reserve = 100,000
        // Usable = 900,000
        // Each capped at 200,000, so max allocated = 400,000
        // Unallocated = 900,000 - 400,000 = 500,000

        let total =
            plan.total_allocated.mantissa + plan.unallocated.mantissa + plan.reserved.mantissa;
        assert_eq!(total, bucket.available_capital.mantissa);

        assert!(plan.unallocated_reasons.iter().any(|r| r.contains("caps")));
    }

    #[test]
    fn test_priority_fill_mode() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);

        let policy = AllocationPolicy {
            mode: AllocationMode::PriorityFill,
            ..Default::default()
        };

        let mut caps = BTreeMap::new();
        // Cap strat_a at 600,000
        caps.insert(
            StrategyId::new("strat_a"),
            Some(FixedPoint::new(60_000_000, -2)),
        );

        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        // Usable = 900,000
        // strat_a gets min(900,000, 600,000) = 600,000
        // strat_b gets remaining 300,000
        let alloc_a = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_a")
            .unwrap();
        let alloc_b = plan
            .allocations
            .iter()
            .find(|a| a.strategy_id.0 == "strat_b")
            .unwrap();

        assert_eq!(alloc_a.assigned_capital.mantissa, 60_000_000);
        assert_eq!(alloc_b.assigned_capital.mantissa, 30_000_000);
    }

    #[test]
    fn test_validation_all_checks_pass() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec!["strat_a", "strat_b"]);
        let policy = AllocationPolicy::default();
        let caps = BTreeMap::new();

        let allocator = Allocator::new(policy.clone());
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        let decision = validate_plan(&snapshot, &intent, &policy, &plan);

        assert!(decision.accepted);
        for check in &decision.checks {
            assert!(
                check.passed,
                "Check {} failed: {}",
                check.name, check.reason
            );
        }
    }

    #[test]
    fn test_empty_intent_handled() {
        let bucket = create_test_bucket("bucket_1", 100_000_000);
        let snapshot = create_test_snapshot(vec![bucket.clone()]);
        let intent = create_test_intent(&bucket.bucket_id, vec![]);
        let policy = AllocationPolicy::default();
        let caps = BTreeMap::new();

        let allocator = Allocator::new(policy);
        let plan = allocator.allocate(&snapshot, &intent, &caps).unwrap();

        assert!(plan.allocations.is_empty());
        assert!(plan.total_allocated.is_zero());
        assert!(
            plan.unallocated_reasons
                .contains(&"no_strategies_in_intent".to_string())
        );
    }
}
