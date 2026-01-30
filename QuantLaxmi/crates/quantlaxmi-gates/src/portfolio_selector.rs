//! # Portfolio Selector (Phase 13.2b)
//!
//! Select which eligible strategies may draw from which capital buckets,
//! in what priority order, under explicit constraints — without assigning quantities.
//!
//! ## Core Question
//! "Given governed capital buckets and eligible strategies,
//! what is the admissible portfolio structure?"
//!
//! ## NOT in Scope
//! - Capital quantities
//! - Allocation math
//! - Rebalancing
//! - Trade-level logic
//! - Optimizer feedback loops
//!
//! ## Hard Architectural Laws
//! - Selector is read-only (consumes snapshots, never mutates)
//! - No capital quantities (only ordering and admissibility)
//! - Bucket constraints are absolute
//! - Eligibility is never re-evaluated (trusts Phase 13.1 + 13.2a artifacts)
//! - All outcomes are auditable artifacts (deterministic, hashed, replayable)

use crate::capital_buckets::{
    BucketConstraints, BucketEligibilityBinding, BucketId, BucketSnapshot, StrategyId,
};
use crate::capital_eligibility::EligibilityStatus;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

pub const PORTFOLIO_INTENT_SCHEMA: &str = "portfolio_intent_v1.0";

// =============================================================================
// Identifier Types
// =============================================================================

/// Unique identifier for a portfolio intent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentId(pub String);

impl IntentId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for IntentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Ordering Rules
// =============================================================================

/// Rule for ordering strategies within a bucket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderingRule {
    /// Higher eligibility tier first (Eligible > Conditional)
    EligibilityTier,
    /// Higher alpha score first
    AlphaScoreDescending,
    /// Lower historical drawdown first
    DrawdownAscending,
    /// Older promotion timestamp first
    PromotionTimestampAscending,
    /// Deterministic tiebreaker by strategy ID
    StrategyIdAscending,
}

// =============================================================================
// Portfolio Policy
// =============================================================================

/// Policy driving the portfolio selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPolicy {
    /// Maximum strategies per bucket (overrides bucket constraint if lower)
    pub max_strategies_per_bucket: Option<u32>,

    /// Whether to include conditional eligibility strategies
    pub allow_conditional: bool,

    /// Ordering rules applied in sequence
    pub ordering_rules: Vec<OrderingRule>,

    /// Policy version for audit trails
    pub policy_version: String,
}

impl Default for PortfolioPolicy {
    fn default() -> Self {
        Self {
            max_strategies_per_bucket: None,
            allow_conditional: true,
            ordering_rules: vec![
                OrderingRule::EligibilityTier,
                OrderingRule::AlphaScoreDescending,
                OrderingRule::DrawdownAscending,
                OrderingRule::PromotionTimestampAscending,
                OrderingRule::StrategyIdAscending,
            ],
            policy_version: "default_v1.0".to_string(),
        }
    }
}

impl PortfolioPolicy {
    /// Create a strict policy that excludes conditional strategies.
    pub fn strict() -> Self {
        Self {
            max_strategies_per_bucket: Some(3),
            allow_conditional: false,
            ordering_rules: vec![
                OrderingRule::AlphaScoreDescending,
                OrderingRule::DrawdownAscending,
                OrderingRule::StrategyIdAscending,
            ],
            policy_version: "strict_v1.0".to_string(),
        }
    }

    /// Create a lenient policy that allows more strategies.
    pub fn lenient() -> Self {
        Self {
            max_strategies_per_bucket: Some(10),
            allow_conditional: true,
            ordering_rules: vec![
                OrderingRule::EligibilityTier,
                OrderingRule::AlphaScoreDescending,
                OrderingRule::StrategyIdAscending,
            ],
            policy_version: "lenient_v1.0".to_string(),
        }
    }
}

// =============================================================================
// Strategy Metrics (for ordering)
// =============================================================================

/// Metrics used for ordering strategies.
/// Provided externally — selector does not compute these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyOrderingMetrics {
    pub strategy_id: StrategyId,
    pub eligibility_status: EligibilityStatus,
    pub alpha_score_mantissa: i64,
    pub alpha_exponent: i8,
    pub max_drawdown_bps: i64,
    pub promotion_ts_ns: i64,
    pub eligibility_digest: String,
    pub binding_digest: String,
}

// =============================================================================
// Strategy Intent
// =============================================================================

/// A strategy's position in the portfolio intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyIntent {
    /// Strategy identifier
    pub strategy_id: StrategyId,

    /// Ordinal priority (1 = highest)
    pub priority: u32,

    /// Digest of eligibility decision that authorized this
    pub eligibility_digest: String,

    /// Digest of bucket binding decision
    pub binding_digest: String,
}

// =============================================================================
// Portfolio Intent
// =============================================================================

/// The sole output of Phase 13.2b.
/// One intent per bucket, containing ordered strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioIntent {
    /// Unique intent identifier
    pub intent_id: IntentId,

    /// Schema version
    pub schema_version: String,

    /// The bucket this intent applies to
    pub bucket_id: BucketId,

    /// Ordered list of strategies (by priority)
    pub strategy_order: Vec<StrategyIntent>,

    /// Bucket constraints (copied for audit)
    pub constraints: BucketConstraints,

    /// Policy version used
    pub policy_version: String,

    /// Deterministic digest
    pub digest: String,

    /// When intent was created
    pub created_at: DateTime<Utc>,
}

impl PortfolioIntent {
    /// Compute deterministic digest.
    fn compute_digest(
        bucket_id: &BucketId,
        strategy_order: &[StrategyIntent],
        policy_version: &str,
        snapshot_digest: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(PORTFOLIO_INTENT_SCHEMA.as_bytes());
        hasher.update(bucket_id.0.as_bytes());
        hasher.update(policy_version.as_bytes());
        hasher.update(snapshot_digest.as_bytes());

        // Hash strategy ordering
        for intent in strategy_order {
            hasher.update(intent.strategy_id.0.as_bytes());
            hasher.update(intent.priority.to_le_bytes());
            hasher.update(intent.eligibility_digest.as_bytes());
            hasher.update(intent.binding_digest.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Get strategy count.
    pub fn strategy_count(&self) -> usize {
        self.strategy_order.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.strategy_order.is_empty()
    }
}

// =============================================================================
// Portfolio Rejection
// =============================================================================

/// Rejection artifact for strategies that failed selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioRejection {
    /// Strategy that was rejected
    pub strategy_id: StrategyId,

    /// Bucket it was rejected from
    pub bucket_id: BucketId,

    /// Reason for rejection
    pub reason: String,

    /// Deterministic digest
    pub digest: String,

    /// When rejection was recorded
    pub rejected_at: DateTime<Utc>,
}

impl PortfolioRejection {
    /// Create a new rejection.
    pub fn new(strategy_id: StrategyId, bucket_id: BucketId, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        let digest = Self::compute_digest(&strategy_id, &bucket_id, &reason);

        Self {
            strategy_id,
            bucket_id,
            reason,
            digest,
            rejected_at: Utc::now(),
        }
    }

    fn compute_digest(strategy_id: &StrategyId, bucket_id: &BucketId, reason: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(strategy_id.0.as_bytes());
        hasher.update(bucket_id.0.as_bytes());
        hasher.update(reason.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Selection Result
// =============================================================================

/// Result of portfolio selection for a single bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketSelectionResult {
    /// The portfolio intent (selected strategies)
    pub intent: PortfolioIntent,

    /// Strategies that were rejected
    pub rejections: Vec<PortfolioRejection>,
}

/// Result of full portfolio selection across all buckets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSelectionResult {
    /// Schema version
    pub schema_version: String,

    /// Results per bucket
    pub bucket_results: Vec<BucketSelectionResult>,

    /// Overall digest
    pub digest: String,

    /// When selection was performed
    pub selected_at: DateTime<Utc>,
}

impl PortfolioSelectionResult {
    /// Total intents across all buckets.
    pub fn total_intents(&self) -> usize {
        self.bucket_results
            .iter()
            .map(|r| r.intent.strategy_count())
            .sum()
    }

    /// Total rejections across all buckets.
    pub fn total_rejections(&self) -> usize {
        self.bucket_results.iter().map(|r| r.rejections.len()).sum()
    }
}

// =============================================================================
// Portfolio Selector
// =============================================================================

/// The portfolio selector — read-only, policy-driven, deterministic.
#[derive(Debug, Clone)]
pub struct PortfolioSelector {
    policy: PortfolioPolicy,
}

impl PortfolioSelector {
    /// Create a new selector with given policy.
    pub fn new(policy: PortfolioPolicy) -> Self {
        Self { policy }
    }

    /// Create with default policy.
    pub fn with_default_policy() -> Self {
        Self::new(PortfolioPolicy::default())
    }

    /// Select portfolio from a bucket snapshot and strategy metrics.
    ///
    /// This is the main entry point. It:
    /// 1. Collects candidates from bindings
    /// 2. Filters by constraints
    /// 3. Orders by policy rules
    /// 4. Produces intents and rejections
    pub fn select(
        &self,
        snapshot: &BucketSnapshot,
        metrics: &[StrategyOrderingMetrics],
    ) -> PortfolioSelectionResult {
        let mut bucket_results = Vec::new();

        // Process each bucket in the snapshot
        for bucket in &snapshot.buckets {
            let result =
                self.select_for_bucket(&bucket.bucket_id, &bucket.constraints, snapshot, metrics);
            bucket_results.push(result);
        }

        // Compute overall digest
        let digest = Self::compute_result_digest(&bucket_results, &snapshot.digest);

        PortfolioSelectionResult {
            schema_version: PORTFOLIO_INTENT_SCHEMA.to_string(),
            bucket_results,
            digest,
            selected_at: Utc::now(),
        }
    }

    /// Select for a single bucket.
    fn select_for_bucket(
        &self,
        bucket_id: &BucketId,
        constraints: &BucketConstraints,
        snapshot: &BucketSnapshot,
        metrics: &[StrategyOrderingMetrics],
    ) -> BucketSelectionResult {
        let mut rejections = Vec::new();

        // Step 1: Collect candidates from bindings
        let bindings: Vec<&BucketEligibilityBinding> = snapshot
            .bindings
            .iter()
            .filter(|b| &b.bucket_id == bucket_id)
            .collect();

        // Step 2: Match bindings with metrics and filter
        let mut candidates: Vec<&StrategyOrderingMetrics> = Vec::new();

        for binding in &bindings {
            // Find metrics for this strategy
            let strategy_metrics = metrics
                .iter()
                .find(|m| m.strategy_id == binding.strategy_id);

            match strategy_metrics {
                None => {
                    rejections.push(PortfolioRejection::new(
                        binding.strategy_id.clone(),
                        bucket_id.clone(),
                        "No metrics provided for strategy",
                    ));
                }
                Some(m) => {
                    // Check eligibility status
                    match &m.eligibility_status {
                        EligibilityStatus::Eligible => {
                            candidates.push(m);
                        }
                        EligibilityStatus::Conditional { .. } => {
                            if self.policy.allow_conditional {
                                candidates.push(m);
                            } else {
                                rejections.push(PortfolioRejection::new(
                                    binding.strategy_id.clone(),
                                    bucket_id.clone(),
                                    "Conditional eligibility not allowed by policy",
                                ));
                            }
                        }
                        EligibilityStatus::Ineligible { reasons } => {
                            rejections.push(PortfolioRejection::new(
                                binding.strategy_id.clone(),
                                bucket_id.clone(),
                                format!("Strategy is ineligible: {}", reasons.join(", ")),
                            ));
                        }
                    }
                }
            }
        }

        // Step 3: Sort candidates by policy rules
        let mut sorted_candidates = candidates;
        self.sort_by_policy(&mut sorted_candidates);

        // Step 4: Apply limits
        let max_strategies = self
            .policy
            .max_strategies_per_bucket
            .or(constraints.max_concurrent_strategies)
            .unwrap_or(u32::MAX) as usize;

        // Reject candidates beyond limit
        if sorted_candidates.len() > max_strategies {
            for rejected in sorted_candidates.drain(max_strategies..) {
                rejections.push(PortfolioRejection::new(
                    rejected.strategy_id.clone(),
                    bucket_id.clone(),
                    format!("Exceeded max strategies limit ({})", max_strategies),
                ));
            }
        }

        // Step 5: Build strategy intents with priorities
        let strategy_order: Vec<StrategyIntent> = sorted_candidates
            .iter()
            .enumerate()
            .map(|(idx, m)| StrategyIntent {
                strategy_id: m.strategy_id.clone(),
                priority: (idx + 1) as u32, // 1-indexed
                eligibility_digest: m.eligibility_digest.clone(),
                binding_digest: m.binding_digest.clone(),
            })
            .collect();

        // Step 6: Build intent
        let digest = PortfolioIntent::compute_digest(
            bucket_id,
            &strategy_order,
            &self.policy.policy_version,
            &snapshot.digest,
        );

        let intent = PortfolioIntent {
            intent_id: IntentId::new(format!("intent_{}_{}", bucket_id.0, Utc::now().timestamp())),
            schema_version: PORTFOLIO_INTENT_SCHEMA.to_string(),
            bucket_id: bucket_id.clone(),
            strategy_order,
            constraints: constraints.clone(),
            policy_version: self.policy.policy_version.clone(),
            digest,
            created_at: Utc::now(),
        };

        BucketSelectionResult { intent, rejections }
    }

    /// Sort candidates by policy ordering rules.
    fn sort_by_policy(&self, candidates: &mut [&StrategyOrderingMetrics]) {
        candidates.sort_by(|a, b| {
            for rule in &self.policy.ordering_rules {
                let cmp = match rule {
                    OrderingRule::EligibilityTier => {
                        let tier_a = Self::eligibility_tier(&a.eligibility_status);
                        let tier_b = Self::eligibility_tier(&b.eligibility_status);
                        tier_a.cmp(&tier_b) // Lower tier number = higher priority
                    }
                    OrderingRule::AlphaScoreDescending => {
                        // Higher alpha = better, so reverse comparison
                        b.alpha_score_mantissa.cmp(&a.alpha_score_mantissa)
                    }
                    OrderingRule::DrawdownAscending => {
                        // Lower drawdown = better
                        a.max_drawdown_bps.cmp(&b.max_drawdown_bps)
                    }
                    OrderingRule::PromotionTimestampAscending => {
                        // Older = better (lower timestamp)
                        a.promotion_ts_ns.cmp(&b.promotion_ts_ns)
                    }
                    OrderingRule::StrategyIdAscending => {
                        // Deterministic tiebreaker
                        a.strategy_id.0.cmp(&b.strategy_id.0)
                    }
                };

                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    /// Map eligibility status to tier (lower = higher priority).
    fn eligibility_tier(status: &EligibilityStatus) -> u8 {
        match status {
            EligibilityStatus::Eligible => 0,
            EligibilityStatus::Conditional { .. } => 1,
            EligibilityStatus::Ineligible { .. } => 2,
        }
    }

    /// Compute digest for full result.
    fn compute_result_digest(
        bucket_results: &[BucketSelectionResult],
        snapshot_digest: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(PORTFOLIO_INTENT_SCHEMA.as_bytes());
        hasher.update(snapshot_digest.as_bytes());

        for result in bucket_results {
            hasher.update(result.intent.digest.as_bytes());
            for rejection in &result.rejections {
                hasher.update(rejection.digest.as_bytes());
            }
        }

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capital_buckets::{CapitalBucket, Currency, FixedPoint, RiskClass, SnapshotId};
    use crate::capital_eligibility::{EligibilityCondition, Venue};

    fn make_bucket(id: &str) -> CapitalBucket {
        CapitalBucket::new(
            BucketId::new(id),
            Venue::BinancePerp,
            Currency::USDT,
            FixedPoint::new(1_000_000_000, -2),
            BucketConstraints::new(RiskClass::Moderate),
        )
    }

    fn make_binding(bucket_id: &str, strategy_id: &str) -> BucketEligibilityBinding {
        BucketEligibilityBinding {
            bucket_id: BucketId::new(bucket_id),
            strategy_id: StrategyId::new(strategy_id),
            eligibility_digest: format!("elig_{}", strategy_id),
            bound_at: Utc::now(),
        }
    }

    fn make_metrics(
        strategy_id: &str,
        alpha: i64,
        drawdown: i64,
        eligible: bool,
    ) -> StrategyOrderingMetrics {
        StrategyOrderingMetrics {
            strategy_id: StrategyId::new(strategy_id),
            eligibility_status: if eligible {
                EligibilityStatus::Eligible
            } else {
                EligibilityStatus::Ineligible {
                    reasons: vec!["test".to_string()],
                }
            },
            alpha_score_mantissa: alpha,
            alpha_exponent: -4,
            max_drawdown_bps: drawdown,
            promotion_ts_ns: 1_000_000_000,
            eligibility_digest: format!("elig_{}", strategy_id),
            binding_digest: format!("bind_{}", strategy_id),
        }
    }

    fn make_snapshot(
        buckets: Vec<CapitalBucket>,
        bindings: Vec<BucketEligibilityBinding>,
    ) -> BucketSnapshot {
        BucketSnapshot::new(SnapshotId::new("test_snapshot"), buckets, bindings)
    }

    // =========================================================================
    // Test 1: Selector respects bucket limits
    // =========================================================================
    #[test]
    fn test_selector_respects_bucket_limits() {
        let bucket = CapitalBucket::new(
            BucketId::new("limited_bucket"),
            Venue::BinancePerp,
            Currency::USDT,
            FixedPoint::new(1_000_000_000, -2),
            BucketConstraints::new(RiskClass::Moderate).with_max_strategies(2),
        );

        let bindings = vec![
            make_binding("limited_bucket", "strategy_1"),
            make_binding("limited_bucket", "strategy_2"),
            make_binding("limited_bucket", "strategy_3"),
        ];

        let metrics = vec![
            make_metrics("strategy_1", 1000, 100, true),
            make_metrics("strategy_2", 900, 150, true),
            make_metrics("strategy_3", 800, 200, true),
        ];

        let snapshot = make_snapshot(vec![bucket], bindings);
        let selector = PortfolioSelector::with_default_policy();
        let result = selector.select(&snapshot, &metrics);

        assert_eq!(result.bucket_results.len(), 1);

        let bucket_result = &result.bucket_results[0];
        // Only 2 strategies should be selected (bucket limit)
        assert_eq!(bucket_result.intent.strategy_count(), 2);
        // 1 rejection
        assert_eq!(bucket_result.rejections.len(), 1);
        assert!(
            bucket_result.rejections[0]
                .reason
                .contains("max strategies")
        );
    }

    // =========================================================================
    // Test 2: Selector orders by priority
    // =========================================================================
    #[test]
    fn test_selector_orders_by_priority() {
        let bucket = make_bucket("bucket_001");
        let bindings = vec![
            make_binding("bucket_001", "low_alpha"),
            make_binding("bucket_001", "high_alpha"),
            make_binding("bucket_001", "mid_alpha"),
        ];

        let metrics = vec![
            make_metrics("low_alpha", 500, 100, true),
            make_metrics("high_alpha", 2000, 100, true),
            make_metrics("mid_alpha", 1000, 100, true),
        ];

        let snapshot = make_snapshot(vec![bucket], bindings);
        let selector = PortfolioSelector::with_default_policy();
        let result = selector.select(&snapshot, &metrics);

        let intent = &result.bucket_results[0].intent;
        assert_eq!(intent.strategy_count(), 3);

        // Should be ordered by alpha (descending)
        assert_eq!(intent.strategy_order[0].strategy_id.0, "high_alpha");
        assert_eq!(intent.strategy_order[0].priority, 1);
        assert_eq!(intent.strategy_order[1].strategy_id.0, "mid_alpha");
        assert_eq!(intent.strategy_order[1].priority, 2);
        assert_eq!(intent.strategy_order[2].strategy_id.0, "low_alpha");
        assert_eq!(intent.strategy_order[2].priority, 3);
    }

    // =========================================================================
    // Test 3: Conditional strategies handled
    // =========================================================================
    #[test]
    fn test_conditional_strategies_handled() {
        let bucket = make_bucket("bucket_001");
        let bindings = vec![
            make_binding("bucket_001", "eligible_strategy"),
            make_binding("bucket_001", "conditional_strategy"),
        ];

        let metrics = vec![
            StrategyOrderingMetrics {
                strategy_id: StrategyId::new("eligible_strategy"),
                eligibility_status: EligibilityStatus::Eligible,
                alpha_score_mantissa: 1000,
                alpha_exponent: -4,
                max_drawdown_bps: 100,
                promotion_ts_ns: 1_000_000_000,
                eligibility_digest: "elig_1".to_string(),
                binding_digest: "bind_1".to_string(),
            },
            StrategyOrderingMetrics {
                strategy_id: StrategyId::new("conditional_strategy"),
                eligibility_status: EligibilityStatus::Conditional {
                    conditions: vec![EligibilityCondition {
                        condition_type:
                            crate::capital_eligibility::ConditionType::EnhancedMonitoring,
                        description: "Needs monitoring".to_string(),
                    }],
                },
                alpha_score_mantissa: 2000, // Higher alpha
                alpha_exponent: -4,
                max_drawdown_bps: 100,
                promotion_ts_ns: 1_000_000_000,
                eligibility_digest: "elig_2".to_string(),
                binding_digest: "bind_2".to_string(),
            },
        ];

        let snapshot = make_snapshot(vec![bucket.clone()], bindings.clone());

        // Default policy allows conditional
        let default_selector = PortfolioSelector::with_default_policy();
        let result = default_selector.select(&snapshot, &metrics);
        assert_eq!(result.bucket_results[0].intent.strategy_count(), 2);
        // Eligible comes first due to EligibilityTier rule
        assert_eq!(
            result.bucket_results[0].intent.strategy_order[0]
                .strategy_id
                .0,
            "eligible_strategy"
        );

        // Strict policy rejects conditional
        let strict_selector = PortfolioSelector::new(PortfolioPolicy::strict());
        let snapshot2 = make_snapshot(vec![bucket], bindings);
        let result2 = strict_selector.select(&snapshot2, &metrics);
        assert_eq!(result2.bucket_results[0].intent.strategy_count(), 1);
        assert_eq!(result2.bucket_results[0].rejections.len(), 1);
        assert!(
            result2.bucket_results[0].rejections[0]
                .reason
                .contains("Conditional")
        );
    }

    // =========================================================================
    // Test 4: Selector produces deterministic digest
    // =========================================================================
    #[test]
    fn test_selector_deterministic_digest() {
        let bucket = make_bucket("bucket_001");
        let bindings = vec![
            make_binding("bucket_001", "strategy_1"),
            make_binding("bucket_001", "strategy_2"),
        ];

        let metrics = vec![
            make_metrics("strategy_1", 1000, 100, true),
            make_metrics("strategy_2", 900, 150, true),
        ];

        let snapshot = make_snapshot(vec![bucket], bindings);
        let selector = PortfolioSelector::with_default_policy();

        // Run twice
        let result1 = selector.select(&snapshot, &metrics);
        let result2 = selector.select(&snapshot, &metrics);

        // Intent digests should match
        assert_eq!(
            result1.bucket_results[0].intent.digest,
            result2.bucket_results[0].intent.digest
        );

        // Overall result digests should match
        assert_eq!(result1.digest, result2.digest);
    }

    // =========================================================================
    // Test 5: Rejection artifacts created
    // =========================================================================
    #[test]
    fn test_rejection_artifacts_created() {
        let bucket = make_bucket("bucket_001");
        let bindings = vec![
            make_binding("bucket_001", "eligible_strategy"),
            make_binding("bucket_001", "ineligible_strategy"),
            make_binding("bucket_001", "missing_metrics_strategy"),
        ];

        let metrics = vec![
            make_metrics("eligible_strategy", 1000, 100, true),
            make_metrics("ineligible_strategy", 1000, 100, false), // Ineligible
                                                                   // No metrics for missing_metrics_strategy
        ];

        let snapshot = make_snapshot(vec![bucket], bindings);
        let selector = PortfolioSelector::with_default_policy();
        let result = selector.select(&snapshot, &metrics);

        let bucket_result = &result.bucket_results[0];

        // 1 strategy selected
        assert_eq!(bucket_result.intent.strategy_count(), 1);

        // 2 rejections
        assert_eq!(bucket_result.rejections.len(), 2);

        // Verify rejection reasons
        let reasons: Vec<&str> = bucket_result
            .rejections
            .iter()
            .map(|r| r.reason.as_str())
            .collect();

        assert!(reasons.iter().any(|r| r.contains("ineligible")));
        assert!(reasons.iter().any(|r| r.contains("No metrics")));

        // Rejections have digests
        for rejection in &bucket_result.rejections {
            assert!(!rejection.digest.is_empty());
        }
    }

    // =========================================================================
    // Test 6: Multiple buckets processed
    // =========================================================================
    #[test]
    fn test_multiple_buckets_processed() {
        let bucket1 = make_bucket("bucket_1");
        let bucket2 = make_bucket("bucket_2");

        let bindings = vec![
            make_binding("bucket_1", "strategy_a"),
            make_binding("bucket_1", "strategy_b"),
            make_binding("bucket_2", "strategy_c"),
        ];

        let metrics = vec![
            make_metrics("strategy_a", 1000, 100, true),
            make_metrics("strategy_b", 900, 150, true),
            make_metrics("strategy_c", 800, 200, true),
        ];

        let snapshot = make_snapshot(vec![bucket1, bucket2], bindings);
        let selector = PortfolioSelector::with_default_policy();
        let result = selector.select(&snapshot, &metrics);

        // Two bucket results
        assert_eq!(result.bucket_results.len(), 2);

        // Bucket 1 has 2 strategies
        assert_eq!(result.bucket_results[0].intent.strategy_count(), 2);

        // Bucket 2 has 1 strategy
        assert_eq!(result.bucket_results[1].intent.strategy_count(), 1);

        // Total intents
        assert_eq!(result.total_intents(), 3);
    }

    // =========================================================================
    // Test 7: Empty bucket handled
    // =========================================================================
    #[test]
    fn test_empty_bucket_handled() {
        let bucket = make_bucket("empty_bucket");
        let bindings: Vec<BucketEligibilityBinding> = vec![];
        let metrics: Vec<StrategyOrderingMetrics> = vec![];

        let snapshot = make_snapshot(vec![bucket], bindings);
        let selector = PortfolioSelector::with_default_policy();
        let result = selector.select(&snapshot, &metrics);

        assert_eq!(result.bucket_results.len(), 1);
        assert!(result.bucket_results[0].intent.is_empty());
        assert!(result.bucket_results[0].rejections.is_empty());
    }

    // =========================================================================
    // Test 8: Drawdown tiebreaker works
    // =========================================================================
    #[test]
    fn test_drawdown_tiebreaker() {
        let bucket = make_bucket("bucket_001");
        let bindings = vec![
            make_binding("bucket_001", "high_dd"),
            make_binding("bucket_001", "low_dd"),
        ];

        // Same alpha, different drawdown
        let metrics = vec![
            make_metrics("high_dd", 1000, 500, true), // Higher drawdown
            make_metrics("low_dd", 1000, 100, true),  // Lower drawdown
        ];

        let snapshot = make_snapshot(vec![bucket], bindings);
        let selector = PortfolioSelector::with_default_policy();
        let result = selector.select(&snapshot, &metrics);

        let intent = &result.bucket_results[0].intent;

        // Lower drawdown should come first
        assert_eq!(intent.strategy_order[0].strategy_id.0, "low_dd");
        assert_eq!(intent.strategy_order[1].strategy_id.0, "high_dd");
    }

    // =========================================================================
    // Test 9: Policy limit overrides bucket constraint
    // =========================================================================
    #[test]
    fn test_policy_limit_overrides_bucket() {
        let bucket = CapitalBucket::new(
            BucketId::new("bucket_001"),
            Venue::BinancePerp,
            Currency::USDT,
            FixedPoint::new(1_000_000_000, -2),
            BucketConstraints::new(RiskClass::Moderate).with_max_strategies(10), // Bucket allows 10
        );

        let bindings = vec![
            make_binding("bucket_001", "s1"),
            make_binding("bucket_001", "s2"),
            make_binding("bucket_001", "s3"),
            make_binding("bucket_001", "s4"),
        ];

        let metrics = vec![
            make_metrics("s1", 1000, 100, true),
            make_metrics("s2", 900, 100, true),
            make_metrics("s3", 800, 100, true),
            make_metrics("s4", 700, 100, true),
        ];

        let snapshot = make_snapshot(vec![bucket], bindings);

        // Strict policy limits to 3
        let selector = PortfolioSelector::new(PortfolioPolicy::strict());
        let result = selector.select(&snapshot, &metrics);

        // Only 3 selected despite bucket allowing 10
        assert_eq!(result.bucket_results[0].intent.strategy_count(), 3);
        assert_eq!(result.bucket_results[0].rejections.len(), 1);
    }
}
