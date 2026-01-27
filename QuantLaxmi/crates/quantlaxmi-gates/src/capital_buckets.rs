//! # Capital Buckets (Phase 13.2a)
//!
//! Define venue- and risk-isolated pools of capital as first-class, auditable entities,
//! independent of strategy selection or allocation math.
//!
//! ## Core Question
//! "What capital exists, where is it allowed to operate, and under what constraints?"
//!
//! ## NOT in Scope
//! - Capital allocation
//! - Position sizing
//! - Drawdown tracking
//! - Optimizer hooks
//! - Rebalancing
//!
//! ## Core Invariants
//! - Buckets are venue-isolated (Crypto ≠ India, Perps ≠ Spot ≠ Options)
//! - Strategies do not own capital — they are granted access to buckets
//! - Buckets do not optimize — they expose capacity and constraints only
//! - All bucket state is deterministic and auditable
//! - No bucket can be implicitly shared — every binding is explicit

use crate::capital_eligibility::{EligibilityStatus, TimeWindow, Venue};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

// =============================================================================
// Schema Version
// =============================================================================

pub const BUCKET_SCHEMA_VERSION: &str = "capital_bucket_v1.0";

// =============================================================================
// Identifier Types
// =============================================================================

/// Unique identifier for a capital bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BucketId(pub String);

impl BucketId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for BucketId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a strategy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StrategyId(pub String);

impl StrategyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SnapshotId(pub String);

impl SnapshotId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Trading symbol identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self(symbol.into())
    }
}

// =============================================================================
// Fixed-Point Capital Representation
// =============================================================================

/// Fixed-point representation for capital amounts.
/// All capital values use mantissa + exponent to avoid floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedPoint {
    /// The mantissa (significant digits).
    pub mantissa: i128,
    /// The exponent (power of 10).
    /// e.g., mantissa=100000000, exponent=-2 means 1,000,000.00
    pub exponent: i8,
}

impl FixedPoint {
    /// Create a new fixed-point value.
    pub fn new(mantissa: i128, exponent: i8) -> Self {
        Self { mantissa, exponent }
    }

    /// Create zero with given exponent.
    pub fn zero(exponent: i8) -> Self {
        Self {
            mantissa: 0,
            exponent,
        }
    }

    /// Check if value is zero.
    pub fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// Check if value is positive.
    pub fn is_positive(&self) -> bool {
        self.mantissa > 0
    }

    /// Check if value is negative.
    pub fn is_negative(&self) -> bool {
        self.mantissa < 0
    }

    /// Convert to f64 for display purposes only.
    /// NOT for computation.
    pub fn to_f64_display(&self) -> f64 {
        self.mantissa as f64 * 10f64.powi(self.exponent as i32)
    }

    /// Subtract another fixed-point value (must have same exponent).
    /// Returns None if exponents don't match.
    pub fn checked_sub(&self, other: &FixedPoint) -> Option<FixedPoint> {
        if self.exponent != other.exponent {
            return None;
        }
        Some(FixedPoint {
            mantissa: self.mantissa - other.mantissa,
            exponent: self.exponent,
        })
    }

    /// Add another fixed-point value (must have same exponent).
    /// Returns None if exponents don't match.
    pub fn checked_add(&self, other: &FixedPoint) -> Option<FixedPoint> {
        if self.exponent != other.exponent {
            return None;
        }
        Some(FixedPoint {
            mantissa: self.mantissa + other.mantissa,
            exponent: self.exponent,
        })
    }

    /// Compare with another fixed-point value (must have same exponent).
    pub fn cmp_same_exp(&self, other: &FixedPoint) -> Option<std::cmp::Ordering> {
        if self.exponent != other.exponent {
            return None;
        }
        Some(self.mantissa.cmp(&other.mantissa))
    }
}

impl std::fmt::Display for FixedPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}e{}", self.mantissa, self.exponent)
    }
}

// =============================================================================
// Currency
// =============================================================================

/// Currency for capital denomination.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Currency {
    /// US Dollar
    USD,
    /// Tether (USDT)
    USDT,
    /// Indian Rupee
    INR,
}

impl std::fmt::Display for Currency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Currency::USD => write!(f, "USD"),
            Currency::USDT => write!(f, "USDT"),
            Currency::INR => write!(f, "INR"),
        }
    }
}

// =============================================================================
// Risk Class
// =============================================================================

/// Risk class categorization — categorical, not numeric.
/// Later phases may interpret this, but Phase 13.2a only declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskClass {
    /// Low-risk, capital preservation focus
    Conservative,
    /// Balanced risk/reward
    Moderate,
    /// Higher risk tolerance
    Aggressive,
    /// Research/testing, not for production capital
    Experimental,
}

impl std::fmt::Display for RiskClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskClass::Conservative => write!(f, "conservative"),
            RiskClass::Moderate => write!(f, "moderate"),
            RiskClass::Aggressive => write!(f, "aggressive"),
            RiskClass::Experimental => write!(f, "experimental"),
        }
    }
}

// =============================================================================
// Bucket Constraints
// =============================================================================

/// Hard bounds on bucket usage — not preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketConstraints {
    /// Maximum notional per strategy (optional)
    pub max_notional_per_strategy: Option<FixedPoint>,

    /// Maximum concurrent strategies using this bucket (optional)
    pub max_concurrent_strategies: Option<u32>,

    /// Allowed symbols for this bucket (None = all allowed)
    pub allowed_symbols: Option<Vec<Symbol>>,

    /// Allowed time windows for trading (None = all times allowed)
    pub allowed_time_windows: Option<Vec<TimeWindow>>,

    /// Risk class declaration
    pub risk_class: RiskClass,
}

impl BucketConstraints {
    /// Create new constraints with required risk class.
    pub fn new(risk_class: RiskClass) -> Self {
        Self {
            max_notional_per_strategy: None,
            max_concurrent_strategies: None,
            allowed_symbols: None,
            allowed_time_windows: None,
            risk_class,
        }
    }

    /// Set maximum notional per strategy.
    pub fn with_max_notional(mut self, max: FixedPoint) -> Self {
        self.max_notional_per_strategy = Some(max);
        self
    }

    /// Set maximum concurrent strategies.
    pub fn with_max_strategies(mut self, max: u32) -> Self {
        self.max_concurrent_strategies = Some(max);
        self
    }

    /// Set allowed symbols.
    pub fn with_symbols(mut self, symbols: Vec<Symbol>) -> Self {
        self.allowed_symbols = Some(symbols);
        self
    }

    /// Set allowed time windows.
    pub fn with_time_windows(mut self, windows: Vec<TimeWindow>) -> Self {
        self.allowed_time_windows = Some(windows);
        self
    }
}

// =============================================================================
// Capital Bucket
// =============================================================================

/// The atomic unit of capital — venue-isolated and constraint-bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapitalBucket {
    /// Unique bucket identifier
    pub bucket_id: BucketId,

    /// Trading venue (enforces isolation)
    pub venue: Venue,

    /// Currency denomination
    pub currency: Currency,

    /// Total capital in bucket (immutable except via explicit events)
    pub total_capital: FixedPoint,

    /// Available capital (changes only through allocations in Phase 13.2+)
    pub available_capital: FixedPoint,

    /// Hard constraints on bucket usage
    pub constraints: BucketConstraints,

    /// Bucket creation timestamp
    pub created_at: DateTime<Utc>,
}

impl CapitalBucket {
    /// Create a new capital bucket.
    pub fn new(
        bucket_id: BucketId,
        venue: Venue,
        currency: Currency,
        capital: FixedPoint,
        constraints: BucketConstraints,
    ) -> Self {
        Self {
            bucket_id,
            venue,
            currency,
            total_capital: capital,
            available_capital: capital, // Initially all available
            constraints,
            created_at: Utc::now(),
        }
    }

    /// Check if bucket has capacity (available > 0).
    pub fn has_capacity(&self) -> bool {
        self.available_capital.is_positive()
    }

    /// Get utilization ratio as basis points (0-10000).
    /// Returns 0 if total is zero.
    pub fn utilization_bps(&self) -> u32 {
        if self.total_capital.is_zero() {
            return 0;
        }
        if self.total_capital.exponent != self.available_capital.exponent {
            return 0; // Exponent mismatch, can't compute
        }
        let used = self.total_capital.mantissa - self.available_capital.mantissa;
        let ratio = (used * 10000) / self.total_capital.mantissa;
        ratio.clamp(0, 10000) as u32
    }
}

// =============================================================================
// Bucket Registry
// =============================================================================

/// Error type for bucket operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BucketError {
    #[error("Bucket already exists: {0}")]
    AlreadyExists(BucketId),

    #[error("Bucket not found: {0}")]
    NotFound(BucketId),

    #[error("Venue mismatch: bucket is {bucket_venue}, strategy requires {strategy_venue}")]
    VenueMismatch {
        bucket_venue: String,
        strategy_venue: String,
    },

    #[error("Binding rejected: {0}")]
    BindingRejected(String),
}

/// Registry for managing capital buckets.
/// Buckets are not ad-hoc objects — they live in a governed registry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BucketRegistry {
    /// All registered buckets, ordered by ID for determinism
    buckets: BTreeMap<BucketId, CapitalBucket>,

    /// All active bindings
    bindings: Vec<BucketEligibilityBinding>,
}

impl BucketRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
            bindings: Vec::new(),
        }
    }

    /// Create a new bucket and register it.
    pub fn create_bucket(&mut self, bucket: CapitalBucket) -> Result<(), BucketError> {
        if self.buckets.contains_key(&bucket.bucket_id) {
            return Err(BucketError::AlreadyExists(bucket.bucket_id.clone()));
        }
        self.buckets.insert(bucket.bucket_id.clone(), bucket);
        Ok(())
    }

    /// Get a bucket by ID (immutable reference).
    pub fn get_bucket(&self, id: &BucketId) -> Option<&CapitalBucket> {
        self.buckets.get(id)
    }

    /// List all bucket IDs.
    pub fn list_bucket_ids(&self) -> Vec<&BucketId> {
        self.buckets.keys().collect()
    }

    /// List all buckets.
    pub fn list_buckets(&self) -> Vec<&CapitalBucket> {
        self.buckets.values().collect()
    }

    /// Get buckets by venue.
    pub fn buckets_by_venue(&self, venue: &Venue) -> Vec<&CapitalBucket> {
        self.buckets
            .values()
            .filter(|b| &b.venue == venue)
            .collect()
    }

    /// Count of registered buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Get all bindings.
    pub fn bindings(&self) -> &[BucketEligibilityBinding] {
        &self.bindings
    }

    /// Get bindings for a specific bucket.
    pub fn bindings_for_bucket(&self, bucket_id: &BucketId) -> Vec<&BucketEligibilityBinding> {
        self.bindings
            .iter()
            .filter(|b| &b.bucket_id == bucket_id)
            .collect()
    }

    /// Get bindings for a specific strategy.
    pub fn bindings_for_strategy(
        &self,
        strategy_id: &StrategyId,
    ) -> Vec<&BucketEligibilityBinding> {
        self.bindings
            .iter()
            .filter(|b| &b.strategy_id == strategy_id)
            .collect()
    }

    /// Bind an eligible strategy to a bucket.
    /// Returns a decision artifact recording acceptance or rejection.
    pub fn bind_strategy(
        &mut self,
        bucket_id: &BucketId,
        strategy_id: StrategyId,
        strategy_venue: &Venue,
        eligibility_status: &EligibilityStatus,
        eligibility_digest: String,
    ) -> BucketBindingDecision {
        let mut reasons = Vec::new();

        // Check 1: Bucket must exist
        let bucket = match self.buckets.get(bucket_id) {
            Some(b) => b,
            None => {
                reasons.push(format!("Bucket not found: {}", bucket_id));
                return BucketBindingDecision::rejected(bucket_id.clone(), strategy_id, reasons);
            }
        };

        // Check 2: Eligibility must be Eligible or Conditional
        match eligibility_status {
            EligibilityStatus::Eligible => {}
            EligibilityStatus::Conditional { .. } => {}
            EligibilityStatus::Ineligible {
                reasons: inel_reasons,
            } => {
                reasons.push(format!(
                    "Strategy is ineligible: {}",
                    inel_reasons.join(", ")
                ));
                return BucketBindingDecision::rejected(bucket_id.clone(), strategy_id, reasons);
            }
        }

        // Check 3: Venue must match
        if &bucket.venue != strategy_venue {
            reasons.push(format!(
                "Venue mismatch: bucket={}, strategy={}",
                bucket.venue, strategy_venue
            ));
            return BucketBindingDecision::rejected(bucket_id.clone(), strategy_id, reasons);
        }

        // Check 4: Max concurrent strategies constraint
        if let Some(max) = bucket.constraints.max_concurrent_strategies {
            let current = self.bindings_for_bucket(bucket_id).len() as u32;
            if current >= max {
                reasons.push(format!(
                    "Bucket at max concurrent strategies: {}/{}",
                    current, max
                ));
                return BucketBindingDecision::rejected(bucket_id.clone(), strategy_id, reasons);
            }
        }

        // Check 5: No duplicate bindings
        let already_bound = self
            .bindings
            .iter()
            .any(|b| &b.bucket_id == bucket_id && b.strategy_id == strategy_id);
        if already_bound {
            reasons.push("Strategy already bound to this bucket".to_string());
            return BucketBindingDecision::rejected(bucket_id.clone(), strategy_id, reasons);
        }

        // All checks passed — create binding
        let binding = BucketEligibilityBinding {
            bucket_id: bucket_id.clone(),
            strategy_id: strategy_id.clone(),
            eligibility_digest,
            bound_at: Utc::now(),
        };
        self.bindings.push(binding);

        BucketBindingDecision::accepted(bucket_id.clone(), strategy_id)
    }

    /// Take a deterministic snapshot of current state.
    pub fn snapshot(&self, snapshot_id: SnapshotId) -> BucketSnapshot {
        BucketSnapshot::new(
            snapshot_id,
            self.buckets.values().cloned().collect(),
            self.bindings.clone(),
        )
    }
}

// =============================================================================
// Bucket-Eligibility Binding
// =============================================================================

/// Explicit binding between a strategy and a bucket.
/// A strategy may bind to multiple buckets.
/// A bucket may serve multiple strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketEligibilityBinding {
    /// The bucket being bound to
    pub bucket_id: BucketId,

    /// The strategy being granted access
    pub strategy_id: StrategyId,

    /// Digest of the eligibility decision that authorized this binding
    pub eligibility_digest: String,

    /// When the binding was created
    pub bound_at: DateTime<Utc>,
}

// =============================================================================
// Bucket Binding Decision
// =============================================================================

/// Decision artifact for binding attempts.
/// Every binding attempt produces an explicit decision — no silent acceptance or rejection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketBindingDecision {
    /// Schema version for forward compatibility
    pub schema_version: String,

    /// Whether the binding was accepted
    pub accepted: bool,

    /// The bucket involved
    pub bucket_id: BucketId,

    /// The strategy involved
    pub strategy_id: StrategyId,

    /// Reasons for rejection (empty if accepted)
    pub reasons: Vec<String>,

    /// Deterministic digest of this decision
    pub digest: String,

    /// When the decision was made
    pub decided_at: DateTime<Utc>,
}

impl BucketBindingDecision {
    /// Create an accepted binding decision.
    pub fn accepted(bucket_id: BucketId, strategy_id: StrategyId) -> Self {
        let mut decision = Self {
            schema_version: BUCKET_SCHEMA_VERSION.to_string(),
            accepted: true,
            bucket_id,
            strategy_id,
            reasons: Vec::new(),
            digest: String::new(),
            decided_at: Utc::now(),
        };
        decision.digest = decision.compute_digest();
        decision
    }

    /// Create a rejected binding decision.
    pub fn rejected(bucket_id: BucketId, strategy_id: StrategyId, reasons: Vec<String>) -> Self {
        let mut decision = Self {
            schema_version: BUCKET_SCHEMA_VERSION.to_string(),
            accepted: false,
            bucket_id,
            strategy_id,
            reasons,
            digest: String::new(),
            decided_at: Utc::now(),
        };
        decision.digest = decision.compute_digest();
        decision
    }

    /// Compute deterministic digest.
    fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.schema_version.as_bytes());
        hasher.update(if self.accepted { b"1" } else { b"0" });
        hasher.update(self.bucket_id.0.as_bytes());
        hasher.update(self.strategy_id.0.as_bytes());
        for reason in &self.reasons {
            hasher.update(reason.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Bucket Snapshot
// =============================================================================

/// Deterministic, read-only snapshot of bucket state.
/// Used for audit trails, replay, and compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketSnapshot {
    /// Unique snapshot identifier
    pub snapshot_id: SnapshotId,

    /// Schema version
    pub schema_version: String,

    /// All buckets at snapshot time
    pub buckets: Vec<CapitalBucket>,

    /// All bindings at snapshot time
    pub bindings: Vec<BucketEligibilityBinding>,

    /// Deterministic digest of snapshot contents
    pub digest: String,

    /// When snapshot was taken
    pub taken_at: DateTime<Utc>,
}

impl BucketSnapshot {
    /// Create a new snapshot with computed digest.
    pub fn new(
        snapshot_id: SnapshotId,
        buckets: Vec<CapitalBucket>,
        bindings: Vec<BucketEligibilityBinding>,
    ) -> Self {
        let mut snapshot = Self {
            snapshot_id,
            schema_version: BUCKET_SCHEMA_VERSION.to_string(),
            buckets,
            bindings,
            digest: String::new(),
            taken_at: Utc::now(),
        };
        snapshot.digest = snapshot.compute_digest();
        snapshot
    }

    /// Compute deterministic digest of snapshot contents.
    fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.snapshot_id.0.as_bytes());
        hasher.update(self.schema_version.as_bytes());

        // Hash buckets in order
        for bucket in &self.buckets {
            hasher.update(bucket.bucket_id.0.as_bytes());
            hasher.update(bucket.venue.to_string().as_bytes());
            hasher.update(bucket.currency.to_string().as_bytes());
            hasher.update(bucket.total_capital.mantissa.to_le_bytes());
            hasher.update([bucket.total_capital.exponent as u8]);
            hasher.update(bucket.available_capital.mantissa.to_le_bytes());
            hasher.update([bucket.available_capital.exponent as u8]);
            hasher.update(bucket.constraints.risk_class.to_string().as_bytes());
        }

        // Hash bindings in order
        for binding in &self.bindings {
            hasher.update(binding.bucket_id.0.as_bytes());
            hasher.update(binding.strategy_id.0.as_bytes());
            hasher.update(binding.eligibility_digest.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Get bucket count.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Get binding count.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bucket(id: &str, venue: Venue) -> CapitalBucket {
        CapitalBucket::new(
            BucketId::new(id),
            venue,
            Currency::USDT,
            FixedPoint::new(10_000_000_00, -2), // $100,000.00
            BucketConstraints::new(RiskClass::Moderate),
        )
    }

    // =========================================================================
    // Test 1: Create bucket
    // =========================================================================
    #[test]
    fn test_create_bucket() {
        let mut registry = BucketRegistry::new();

        let bucket = make_bucket("bucket_001", Venue::BinancePerp);
        assert!(registry.create_bucket(bucket).is_ok());
        assert_eq!(registry.bucket_count(), 1);

        // Verify bucket is retrievable
        let retrieved = registry.get_bucket(&BucketId::new("bucket_001"));
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().venue, Venue::BinancePerp);
    }

    // =========================================================================
    // Test 2: Bucket venue isolation
    // =========================================================================
    #[test]
    fn test_bucket_venue_isolation() {
        let mut registry = BucketRegistry::new();

        // Create buckets for different venues
        registry
            .create_bucket(make_bucket("crypto_perp", Venue::BinancePerp))
            .unwrap();
        registry
            .create_bucket(make_bucket("crypto_spot", Venue::BinanceSpot))
            .unwrap();
        registry
            .create_bucket(make_bucket("india_futures", Venue::NseF))
            .unwrap();

        // Verify venue filtering
        let perp_buckets = registry.buckets_by_venue(&Venue::BinancePerp);
        assert_eq!(perp_buckets.len(), 1);
        assert_eq!(perp_buckets[0].bucket_id.0, "crypto_perp");

        let india_buckets = registry.buckets_by_venue(&Venue::NseF);
        assert_eq!(india_buckets.len(), 1);
        assert_eq!(india_buckets[0].bucket_id.0, "india_futures");

        // No buckets for unused venue
        let options_buckets = registry.buckets_by_venue(&Venue::NseO);
        assert!(options_buckets.is_empty());
    }

    // =========================================================================
    // Test 3: Bind eligible strategy
    // =========================================================================
    #[test]
    fn test_bind_eligible_strategy() {
        let mut registry = BucketRegistry::new();
        registry
            .create_bucket(make_bucket("bucket_001", Venue::BinancePerp))
            .unwrap();

        let decision = registry.bind_strategy(
            &BucketId::new("bucket_001"),
            StrategyId::new("strategy_001"),
            &Venue::BinancePerp,
            &EligibilityStatus::Eligible,
            "eligibility_digest_abc".to_string(),
        );

        assert!(decision.accepted);
        assert!(decision.reasons.is_empty());
        assert_eq!(registry.bindings().len(), 1);

        let binding = &registry.bindings()[0];
        assert_eq!(binding.bucket_id.0, "bucket_001");
        assert_eq!(binding.strategy_id.0, "strategy_001");
        assert_eq!(binding.eligibility_digest, "eligibility_digest_abc");
    }

    // =========================================================================
    // Test 4: Reject ineligible strategy binding
    // =========================================================================
    #[test]
    fn test_bind_ineligible_strategy_rejected() {
        let mut registry = BucketRegistry::new();
        registry
            .create_bucket(make_bucket("bucket_001", Venue::BinancePerp))
            .unwrap();

        let decision = registry.bind_strategy(
            &BucketId::new("bucket_001"),
            StrategyId::new("bad_strategy"),
            &Venue::BinancePerp,
            &EligibilityStatus::Ineligible {
                reasons: vec!["Failed G3".to_string()],
            },
            "ineligible_digest".to_string(),
        );

        assert!(!decision.accepted);
        assert!(decision.reasons.iter().any(|r| r.contains("ineligible")));
        assert!(registry.bindings().is_empty());
    }

    // =========================================================================
    // Test 5: Binding digest is deterministic
    // =========================================================================
    #[test]
    fn test_binding_digest_deterministic() {
        // Create two identical decisions
        let decision1 = BucketBindingDecision {
            schema_version: BUCKET_SCHEMA_VERSION.to_string(),
            accepted: true,
            bucket_id: BucketId::new("bucket_001"),
            strategy_id: StrategyId::new("strategy_001"),
            reasons: Vec::new(),
            digest: String::new(),
            decided_at: Utc::now(), // Different timestamp
        };

        let decision2 = BucketBindingDecision {
            schema_version: BUCKET_SCHEMA_VERSION.to_string(),
            accepted: true,
            bucket_id: BucketId::new("bucket_001"),
            strategy_id: StrategyId::new("strategy_001"),
            reasons: Vec::new(),
            digest: String::new(),
            decided_at: Utc::now(), // Different timestamp
        };

        // Digests should be identical (timestamp excluded from hash)
        assert_eq!(decision1.compute_digest(), decision2.compute_digest());

        // Different bucket ID → different digest
        let decision3 = BucketBindingDecision {
            schema_version: BUCKET_SCHEMA_VERSION.to_string(),
            accepted: true,
            bucket_id: BucketId::new("bucket_002"),
            strategy_id: StrategyId::new("strategy_001"),
            reasons: Vec::new(),
            digest: String::new(),
            decided_at: Utc::now(),
        };
        assert_ne!(decision1.compute_digest(), decision3.compute_digest());
    }

    // =========================================================================
    // Test 6: Snapshot digest is deterministic
    // =========================================================================
    #[test]
    fn test_snapshot_digest_deterministic() {
        let mut registry = BucketRegistry::new();
        registry
            .create_bucket(make_bucket("bucket_001", Venue::BinancePerp))
            .unwrap();
        registry.bind_strategy(
            &BucketId::new("bucket_001"),
            StrategyId::new("strategy_001"),
            &Venue::BinancePerp,
            &EligibilityStatus::Eligible,
            "digest_abc".to_string(),
        );

        // Take two snapshots with same ID
        let snapshot1 = registry.snapshot(SnapshotId::new("snap_001"));
        let snapshot2 = registry.snapshot(SnapshotId::new("snap_001"));

        // Digests should match (same content, same ID)
        assert_eq!(snapshot1.digest, snapshot2.digest);

        // Different ID → different digest
        let snapshot3 = registry.snapshot(SnapshotId::new("snap_002"));
        assert_ne!(snapshot1.digest, snapshot3.digest);
    }

    // =========================================================================
    // Test 7: Venue mismatch rejected
    // =========================================================================
    #[test]
    fn test_venue_mismatch_rejected() {
        let mut registry = BucketRegistry::new();
        registry
            .create_bucket(make_bucket("crypto_bucket", Venue::BinancePerp))
            .unwrap();

        // Try to bind India strategy to crypto bucket
        let decision = registry.bind_strategy(
            &BucketId::new("crypto_bucket"),
            StrategyId::new("india_strategy"),
            &Venue::NseF, // Mismatch!
            &EligibilityStatus::Eligible,
            "digest".to_string(),
        );

        assert!(!decision.accepted);
        assert!(
            decision
                .reasons
                .iter()
                .any(|r| r.contains("Venue mismatch"))
        );
    }

    // =========================================================================
    // Test 8: Bucket not found rejected
    // =========================================================================
    #[test]
    fn test_bind_nonexistent_bucket_rejected() {
        let mut registry = BucketRegistry::new();

        let decision = registry.bind_strategy(
            &BucketId::new("nonexistent"),
            StrategyId::new("strategy_001"),
            &Venue::BinancePerp,
            &EligibilityStatus::Eligible,
            "digest".to_string(),
        );

        assert!(!decision.accepted);
        assert!(decision.reasons.iter().any(|r| r.contains("not found")));
    }

    // =========================================================================
    // Test 9: Max concurrent strategies enforced
    // =========================================================================
    #[test]
    fn test_max_concurrent_strategies_enforced() {
        let mut registry = BucketRegistry::new();

        // Create bucket with max 2 strategies
        let bucket = CapitalBucket::new(
            BucketId::new("limited_bucket"),
            Venue::BinancePerp,
            Currency::USDT,
            FixedPoint::new(10_000_000_00, -2),
            BucketConstraints::new(RiskClass::Moderate).with_max_strategies(2),
        );
        registry.create_bucket(bucket).unwrap();

        // Bind first two strategies
        assert!(
            registry
                .bind_strategy(
                    &BucketId::new("limited_bucket"),
                    StrategyId::new("strategy_1"),
                    &Venue::BinancePerp,
                    &EligibilityStatus::Eligible,
                    "d1".to_string(),
                )
                .accepted
        );

        assert!(
            registry
                .bind_strategy(
                    &BucketId::new("limited_bucket"),
                    StrategyId::new("strategy_2"),
                    &Venue::BinancePerp,
                    &EligibilityStatus::Eligible,
                    "d2".to_string(),
                )
                .accepted
        );

        // Third should be rejected
        let decision = registry.bind_strategy(
            &BucketId::new("limited_bucket"),
            StrategyId::new("strategy_3"),
            &Venue::BinancePerp,
            &EligibilityStatus::Eligible,
            "d3".to_string(),
        );

        assert!(!decision.accepted);
        assert!(
            decision
                .reasons
                .iter()
                .any(|r| r.contains("max concurrent"))
        );
    }

    // =========================================================================
    // Test 10: Duplicate binding rejected
    // =========================================================================
    #[test]
    fn test_duplicate_binding_rejected() {
        let mut registry = BucketRegistry::new();
        registry
            .create_bucket(make_bucket("bucket_001", Venue::BinancePerp))
            .unwrap();

        // First binding succeeds
        assert!(
            registry
                .bind_strategy(
                    &BucketId::new("bucket_001"),
                    StrategyId::new("strategy_001"),
                    &Venue::BinancePerp,
                    &EligibilityStatus::Eligible,
                    "d1".to_string(),
                )
                .accepted
        );

        // Same binding again rejected
        let decision = registry.bind_strategy(
            &BucketId::new("bucket_001"),
            StrategyId::new("strategy_001"),
            &Venue::BinancePerp,
            &EligibilityStatus::Eligible,
            "d2".to_string(),
        );

        assert!(!decision.accepted);
        assert!(decision.reasons.iter().any(|r| r.contains("already bound")));
    }

    // =========================================================================
    // Test 11: Conditional eligibility allowed
    // =========================================================================
    #[test]
    fn test_conditional_eligibility_allowed() {
        use crate::capital_eligibility::{ConditionType, EligibilityCondition};

        let mut registry = BucketRegistry::new();
        registry
            .create_bucket(make_bucket("bucket_001", Venue::BinancePerp))
            .unwrap();

        let decision = registry.bind_strategy(
            &BucketId::new("bucket_001"),
            StrategyId::new("conditional_strategy"),
            &Venue::BinancePerp,
            &EligibilityStatus::Conditional {
                conditions: vec![EligibilityCondition {
                    condition_type: ConditionType::EnhancedMonitoring,
                    description: "Requires extra monitoring".to_string(),
                }],
            },
            "conditional_digest".to_string(),
        );

        // Conditional strategies can bind (with conditions noted)
        assert!(decision.accepted);
    }

    // =========================================================================
    // Test 12: Fixed point arithmetic
    // =========================================================================
    #[test]
    fn test_fixed_point_arithmetic() {
        let a = FixedPoint::new(100_00, -2); // 100.00
        let b = FixedPoint::new(25_00, -2); // 25.00

        let sum = a.checked_add(&b).unwrap();
        assert_eq!(sum.mantissa, 125_00);
        assert_eq!(sum.exponent, -2);

        let diff = a.checked_sub(&b).unwrap();
        assert_eq!(diff.mantissa, 75_00);
        assert_eq!(diff.exponent, -2);

        // Mismatched exponents fail
        let c = FixedPoint::new(100, -4);
        assert!(a.checked_add(&c).is_none());
    }

    // =========================================================================
    // Test 13: Bucket utilization
    // =========================================================================
    #[test]
    fn test_bucket_utilization() {
        let mut bucket = make_bucket("test", Venue::BinancePerp);

        // Initially 0% utilized
        assert_eq!(bucket.utilization_bps(), 0);

        // Simulate 50% used
        bucket.available_capital = FixedPoint::new(5_000_000_00, -2);
        assert_eq!(bucket.utilization_bps(), 5000); // 50%

        // 100% used
        bucket.available_capital = FixedPoint::zero(-2);
        assert_eq!(bucket.utilization_bps(), 10000); // 100%
    }
}
