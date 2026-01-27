//! # Risk & Exposure Layer (Phase 15.1)
//!
//! Deterministic, read-only risk evaluation from portfolio truth.
//!
//! ## Core Question
//! "Given portfolio truth, what exposures exist and should execution proceed?"
//!
//! ## Hard Laws
//! - L1: Read-Only — Risk layer NEVER mutates positions, budgets, or orders
//! - L2: Deterministic — Same inputs → identical outputs and digests
//! - L3: Fixed-Point — All arithmetic uses i128 mantissa + i8 exponent (no f64)
//! - L4: Audit-First — Every decision has deterministic digest + explicit reason codes
//! - L5: Gating Order — RiskDecision → BudgetCheck → Reserve → Submit (frozen)
//!
//! ## NOT in Scope (Phase 15.2+)
//! - Mark-to-market unrealized PnL
//! - Drawdown tracking
//! - Volatility-based sizing
//! - Correlation risk
//! - Margin models

use crate::execution_budget::ExecutionBudget;
use quantlaxmi_models::{OrderIntentEvent, PositionVenue};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::position_keeper::PortfolioSnapshot;

// =============================================================================
// Schema Versions
// =============================================================================

pub const RISK_POLICY_SCHEMA_VERSION: &str = "risk_policy_v1.0";
pub const RISK_SNAPSHOT_SCHEMA_VERSION: &str = "risk_snapshot_v1.0";
pub const RISK_DECISION_SCHEMA_VERSION: &str = "risk_decision_v1.0";

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum RiskError {
    #[error("Risk evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Invalid policy: {0}")]
    InvalidPolicy(String),

    #[error("Arithmetic overflow: {0}")]
    ArithmeticOverflow(String),
}

// =============================================================================
// Notional Normalization Helper (Canonical, Deterministic)
// =============================================================================

/// Normalize notional from source exponent to target exponent.
/// Uses floor division for determinism.
///
/// Formula: result = value * 10^(source_exp - target_exp)
/// If source_exp > target_exp: multiply (shift left)
/// If source_exp < target_exp: divide (shift right, floor)
///
/// Example:
/// - value=1000, source_exp=-8, target_exp=-6
/// - shift = -8 - (-6) = -2
/// - result = 1000 / 10^2 = 10 (floor division)
#[inline]
pub fn normalize_notional(value: i128, source_exp: i8, target_exp: i8) -> i128 {
    let shift = source_exp as i32 - target_exp as i32;
    if shift == 0 {
        value
    } else if shift > 0 {
        // source_exp > target_exp: multiply
        let factor = 10i128.pow(shift as u32);
        value.saturating_mul(factor)
    } else {
        // source_exp < target_exp: divide (floor)
        let factor = 10i128.pow((-shift) as u32);
        value / factor // floor division for negative values too
    }
}

/// Compute notional = abs(quantity * price), normalized to target exponent.
/// Combined exponent = qty_exp + price_exp, then normalize to target.
#[inline]
pub fn compute_notional(
    quantity_mantissa: i128,
    quantity_exponent: i8,
    price_mantissa: i128,
    price_exponent: i8,
    target_exponent: i8,
) -> i128 {
    let raw_notional = quantity_mantissa.abs().saturating_mul(price_mantissa.abs());
    let combined_exp = quantity_exponent as i32 + price_exponent as i32;
    normalize_notional(raw_notional, combined_exp as i8, target_exponent)
}

// =============================================================================
// Risk Policy
// =============================================================================

/// Risk policy configuration.
/// Preset-driven, deterministic fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPolicy {
    /// Schema version.
    pub schema_version: String,

    /// Policy name (e.g., "conservative", "moderate", "aggressive").
    pub name: String,

    /// Policy version for tracking changes.
    pub version: String,

    // === Notional Limits (mantissa, exponent) ===
    /// Max notional exposure per symbol (mantissa).
    pub max_symbol_notional_mantissa: i128,

    /// Max notional exposure per strategy (mantissa).
    pub max_strategy_notional_mantissa: i128,

    /// Max notional exposure per bucket (mantissa).
    pub max_bucket_notional_mantissa: i128,

    /// Max total notional across portfolio (mantissa).
    pub max_portfolio_notional_mantissa: i128,

    /// Notional exponent (shared).
    pub notional_exponent: i8,

    // === Position Count Limits ===
    /// Max open positions per strategy.
    pub max_positions_per_strategy: u32,

    /// Max open positions per bucket.
    pub max_positions_per_bucket: u32,

    /// Max open positions across portfolio.
    pub max_portfolio_positions: u32,

    // === Venue Restrictions ===
    /// Allowed venues (empty = all allowed).
    pub allowed_venues: Vec<PositionVenue>,

    // === Budget Coupling ===
    /// Max ratio of committed to allocated (mantissa, exp -4).
    /// e.g., 8000 with exp -4 = 0.8000 = 80%
    pub max_committed_ratio_mantissa: i64,
    pub committed_ratio_exponent: i8,
}

impl RiskPolicy {
    /// Compute deterministic policy fingerprint (SHA-256).
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();

        // schema_version
        hasher.update((self.schema_version.len() as u32).to_le_bytes());
        hasher.update(self.schema_version.as_bytes());

        // name
        hasher.update((self.name.len() as u32).to_le_bytes());
        hasher.update(self.name.as_bytes());

        // version
        hasher.update((self.version.len() as u32).to_le_bytes());
        hasher.update(self.version.as_bytes());

        // notional limits
        hasher.update(self.max_symbol_notional_mantissa.to_le_bytes());
        hasher.update(self.max_strategy_notional_mantissa.to_le_bytes());
        hasher.update(self.max_bucket_notional_mantissa.to_le_bytes());
        hasher.update(self.max_portfolio_notional_mantissa.to_le_bytes());
        hasher.update([self.notional_exponent as u8]);

        // position count limits
        hasher.update(self.max_positions_per_strategy.to_le_bytes());
        hasher.update(self.max_positions_per_bucket.to_le_bytes());
        hasher.update(self.max_portfolio_positions.to_le_bytes());

        // allowed venues
        hasher.update((self.allowed_venues.len() as u32).to_le_bytes());
        for venue in &self.allowed_venues {
            hasher.update([venue_to_byte(venue)]);
        }

        // budget coupling
        hasher.update(self.max_committed_ratio_mantissa.to_le_bytes());
        hasher.update([self.committed_ratio_exponent as u8]);

        format!("{:x}", hasher.finalize())
    }

    /// Preset: Conservative (tight limits).
    pub fn conservative() -> Self {
        Self {
            schema_version: RISK_POLICY_SCHEMA_VERSION.to_string(),
            name: "conservative".to_string(),
            version: "1.0.0".to_string(),
            // 10,000 USDT per symbol (exp -2)
            max_symbol_notional_mantissa: 1_000_000,
            // 25,000 USDT per strategy
            max_strategy_notional_mantissa: 2_500_000,
            // 50,000 USDT per bucket
            max_bucket_notional_mantissa: 5_000_000,
            // 100,000 USDT portfolio total
            max_portfolio_notional_mantissa: 10_000_000,
            notional_exponent: -2,
            max_positions_per_strategy: 3,
            max_positions_per_bucket: 5,
            max_portfolio_positions: 10,
            allowed_venues: vec![],
            // 50% max committed ratio
            max_committed_ratio_mantissa: 5000,
            committed_ratio_exponent: -4,
        }
    }

    /// Preset: Moderate (balanced).
    pub fn moderate() -> Self {
        Self {
            schema_version: RISK_POLICY_SCHEMA_VERSION.to_string(),
            name: "moderate".to_string(),
            version: "1.0.0".to_string(),
            // 50,000 USDT per symbol
            max_symbol_notional_mantissa: 5_000_000,
            // 100,000 USDT per strategy
            max_strategy_notional_mantissa: 10_000_000,
            // 200,000 USDT per bucket
            max_bucket_notional_mantissa: 20_000_000,
            // 500,000 USDT portfolio total
            max_portfolio_notional_mantissa: 50_000_000,
            notional_exponent: -2,
            max_positions_per_strategy: 5,
            max_positions_per_bucket: 10,
            max_portfolio_positions: 25,
            allowed_venues: vec![],
            // 70% max committed ratio
            max_committed_ratio_mantissa: 7000,
            committed_ratio_exponent: -4,
        }
    }

    /// Preset: Aggressive (loose limits).
    pub fn aggressive() -> Self {
        Self {
            schema_version: RISK_POLICY_SCHEMA_VERSION.to_string(),
            name: "aggressive".to_string(),
            version: "1.0.0".to_string(),
            // 200,000 USDT per symbol
            max_symbol_notional_mantissa: 20_000_000,
            // 500,000 USDT per strategy
            max_strategy_notional_mantissa: 50_000_000,
            // 1,000,000 USDT per bucket
            max_bucket_notional_mantissa: 100_000_000,
            // 2,000,000 USDT portfolio total
            max_portfolio_notional_mantissa: 200_000_000,
            notional_exponent: -2,
            max_positions_per_strategy: 10,
            max_positions_per_bucket: 20,
            max_portfolio_positions: 50,
            allowed_venues: vec![],
            // 90% max committed ratio
            max_committed_ratio_mantissa: 9000,
            committed_ratio_exponent: -4,
        }
    }
}

fn venue_to_byte(venue: &PositionVenue) -> u8 {
    match venue {
        PositionVenue::BinancePerp => 0,
        PositionVenue::BinanceSpot => 1,
        PositionVenue::NseF => 2,
        PositionVenue::NseO => 3,
        PositionVenue::Paper => 4,
    }
}

// =============================================================================
// Exposure Metrics
// =============================================================================

/// Exposure aggregations computed from PortfolioSnapshot.
/// All values are notional = abs(quantity * price).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExposureMetrics {
    /// Notional by symbol: symbol -> mantissa.
    pub notional_by_symbol: BTreeMap<String, i128>,

    /// Notional by strategy: strategy_id -> mantissa.
    pub notional_by_strategy: BTreeMap<String, i128>,

    /// Notional by bucket: bucket_id -> mantissa.
    pub notional_by_bucket: BTreeMap<String, i128>,

    /// Notional by venue: venue -> mantissa.
    pub notional_by_venue: BTreeMap<String, i128>,

    /// Total portfolio notional (mantissa).
    pub total_notional_mantissa: i128,

    /// Notional exponent (shared).
    pub notional_exponent: i8,

    /// Position count by strategy.
    pub position_count_by_strategy: BTreeMap<String, u32>,

    /// Position count by bucket.
    pub position_count_by_bucket: BTreeMap<String, u32>,

    /// Total position count.
    pub total_position_count: u32,
}

impl ExposureMetrics {
    /// Compute from PortfolioSnapshot.
    /// Notional = abs(quantity_mantissa * avg_entry_price_mantissa).
    /// Uses canonical normalize_notional helper.
    pub fn from_portfolio_snapshot(snapshot: &PortfolioSnapshot, target_exponent: i8) -> Self {
        let mut metrics = ExposureMetrics {
            notional_exponent: target_exponent,
            ..Default::default()
        };

        for pos_snap in &snapshot.positions {
            let state = &pos_snap.state;

            // Compute notional using canonical helper
            let notional = compute_notional(
                state.quantity_mantissa,
                state.quantity_exponent,
                state.avg_entry_price_mantissa,
                state.price_exponent,
                target_exponent,
            );

            // Aggregate by symbol
            *metrics
                .notional_by_symbol
                .entry(state.key.symbol.clone())
                .or_insert(0) += notional;

            // Aggregate by strategy
            *metrics
                .notional_by_strategy
                .entry(state.key.strategy_id.clone())
                .or_insert(0) += notional;

            // Aggregate by bucket
            *metrics
                .notional_by_bucket
                .entry(state.key.bucket_id.clone())
                .or_insert(0) += notional;

            // Aggregate by venue
            *metrics
                .notional_by_venue
                .entry(format!("{}", state.key.venue))
                .or_insert(0) += notional;

            // Total notional
            metrics.total_notional_mantissa += notional;

            // Position counts
            *metrics
                .position_count_by_strategy
                .entry(state.key.strategy_id.clone())
                .or_insert(0) += 1;

            *metrics
                .position_count_by_bucket
                .entry(state.key.bucket_id.clone())
                .or_insert(0) += 1;

            metrics.total_position_count += 1;
        }

        metrics
    }
}

// =============================================================================
// Violation Types
// =============================================================================

/// Risk violation type with explicit reason codes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ViolationType {
    /// Symbol notional exceeds limit.
    SymbolNotionalExceeded {
        symbol: String,
        current_mantissa: i128,
        limit_mantissa: i128,
    },

    /// Strategy notional exceeds limit.
    StrategyNotionalExceeded {
        strategy_id: String,
        current_mantissa: i128,
        limit_mantissa: i128,
    },

    /// Bucket notional exceeds limit.
    BucketNotionalExceeded {
        bucket_id: String,
        current_mantissa: i128,
        limit_mantissa: i128,
    },

    /// Portfolio notional exceeds limit.
    PortfolioNotionalExceeded {
        current_mantissa: i128,
        limit_mantissa: i128,
    },

    /// Strategy position count exceeds limit.
    StrategyPositionCountExceeded {
        strategy_id: String,
        current: u32,
        limit: u32,
    },

    /// Bucket position count exceeds limit.
    BucketPositionCountExceeded {
        bucket_id: String,
        current: u32,
        limit: u32,
    },

    /// Portfolio position count exceeds limit.
    PortfolioPositionCountExceeded { current: u32, limit: u32 },

    /// Venue not allowed by policy.
    VenueNotAllowed { venue: String },

    /// Committed capital ratio exceeds limit.
    CommittedRatioExceeded {
        strategy_id: String,
        bucket_id: String,
        current_ratio_mantissa: i64,
        limit_ratio_mantissa: i64,
    },
}

impl ViolationType {
    /// Get violation code for WAL/logging.
    pub fn code(&self) -> &'static str {
        match self {
            ViolationType::SymbolNotionalExceeded { .. } => "SYMBOL_NOTIONAL",
            ViolationType::StrategyNotionalExceeded { .. } => "STRATEGY_NOTIONAL",
            ViolationType::BucketNotionalExceeded { .. } => "BUCKET_NOTIONAL",
            ViolationType::PortfolioNotionalExceeded { .. } => "PORTFOLIO_NOTIONAL",
            ViolationType::StrategyPositionCountExceeded { .. } => "STRATEGY_POSITIONS",
            ViolationType::BucketPositionCountExceeded { .. } => "BUCKET_POSITIONS",
            ViolationType::PortfolioPositionCountExceeded { .. } => "PORTFOLIO_POSITIONS",
            ViolationType::VenueNotAllowed { .. } => "VENUE_NOT_ALLOWED",
            ViolationType::CommittedRatioExceeded { .. } => "COMMITTED_RATIO",
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            ViolationType::SymbolNotionalExceeded {
                symbol,
                current_mantissa,
                limit_mantissa,
            } => {
                format!(
                    "Symbol {} notional {} exceeds limit {}",
                    symbol, current_mantissa, limit_mantissa
                )
            }
            ViolationType::StrategyNotionalExceeded {
                strategy_id,
                current_mantissa,
                limit_mantissa,
            } => {
                format!(
                    "Strategy {} notional {} exceeds limit {}",
                    strategy_id, current_mantissa, limit_mantissa
                )
            }
            ViolationType::BucketNotionalExceeded {
                bucket_id,
                current_mantissa,
                limit_mantissa,
            } => {
                format!(
                    "Bucket {} notional {} exceeds limit {}",
                    bucket_id, current_mantissa, limit_mantissa
                )
            }
            ViolationType::PortfolioNotionalExceeded {
                current_mantissa,
                limit_mantissa,
            } => {
                format!(
                    "Portfolio notional {} exceeds limit {}",
                    current_mantissa, limit_mantissa
                )
            }
            ViolationType::StrategyPositionCountExceeded {
                strategy_id,
                current,
                limit,
            } => {
                format!(
                    "Strategy {} has {} positions, exceeds limit {}",
                    strategy_id, current, limit
                )
            }
            ViolationType::BucketPositionCountExceeded {
                bucket_id,
                current,
                limit,
            } => {
                format!(
                    "Bucket {} has {} positions, exceeds limit {}",
                    bucket_id, current, limit
                )
            }
            ViolationType::PortfolioPositionCountExceeded { current, limit } => {
                format!(
                    "Portfolio has {} positions, exceeds limit {}",
                    current, limit
                )
            }
            ViolationType::VenueNotAllowed { venue } => {
                format!("Venue {} is not allowed by policy", venue)
            }
            ViolationType::CommittedRatioExceeded {
                strategy_id,
                bucket_id,
                current_ratio_mantissa,
                limit_ratio_mantissa,
            } => {
                format!(
                    "Strategy {} bucket {} committed ratio {} exceeds limit {}",
                    strategy_id, bucket_id, current_ratio_mantissa, limit_ratio_mantissa
                )
            }
        }
    }

    /// Check if this violation triggers HALT.
    pub fn is_halt(&self) -> bool {
        matches!(
            self,
            ViolationType::PortfolioNotionalExceeded { .. }
                | ViolationType::PortfolioPositionCountExceeded { .. }
        )
    }

    /// Canonical bytes for hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.code().as_bytes());
        bytes.push(b':');

        match self {
            ViolationType::SymbolNotionalExceeded {
                symbol,
                current_mantissa,
                limit_mantissa,
            } => {
                bytes.extend_from_slice(&(symbol.len() as u32).to_le_bytes());
                bytes.extend_from_slice(symbol.as_bytes());
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&limit_mantissa.to_le_bytes());
            }
            ViolationType::StrategyNotionalExceeded {
                strategy_id,
                current_mantissa,
                limit_mantissa,
            } => {
                bytes.extend_from_slice(&(strategy_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(strategy_id.as_bytes());
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&limit_mantissa.to_le_bytes());
            }
            ViolationType::BucketNotionalExceeded {
                bucket_id,
                current_mantissa,
                limit_mantissa,
            } => {
                bytes.extend_from_slice(&(bucket_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bucket_id.as_bytes());
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&limit_mantissa.to_le_bytes());
            }
            ViolationType::PortfolioNotionalExceeded {
                current_mantissa,
                limit_mantissa,
            } => {
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&limit_mantissa.to_le_bytes());
            }
            ViolationType::StrategyPositionCountExceeded {
                strategy_id,
                current,
                limit,
            } => {
                bytes.extend_from_slice(&(strategy_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(strategy_id.as_bytes());
                bytes.extend_from_slice(&current.to_le_bytes());
                bytes.extend_from_slice(&limit.to_le_bytes());
            }
            ViolationType::BucketPositionCountExceeded {
                bucket_id,
                current,
                limit,
            } => {
                bytes.extend_from_slice(&(bucket_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bucket_id.as_bytes());
                bytes.extend_from_slice(&current.to_le_bytes());
                bytes.extend_from_slice(&limit.to_le_bytes());
            }
            ViolationType::PortfolioPositionCountExceeded { current, limit } => {
                bytes.extend_from_slice(&current.to_le_bytes());
                bytes.extend_from_slice(&limit.to_le_bytes());
            }
            ViolationType::VenueNotAllowed { venue } => {
                bytes.extend_from_slice(&(venue.len() as u32).to_le_bytes());
                bytes.extend_from_slice(venue.as_bytes());
            }
            ViolationType::CommittedRatioExceeded {
                strategy_id,
                bucket_id,
                current_ratio_mantissa,
                limit_ratio_mantissa,
            } => {
                bytes.extend_from_slice(&(strategy_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(strategy_id.as_bytes());
                bytes.extend_from_slice(&(bucket_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bucket_id.as_bytes());
                bytes.extend_from_slice(&current_ratio_mantissa.to_le_bytes());
                bytes.extend_from_slice(&limit_ratio_mantissa.to_le_bytes());
            }
        }

        bytes
    }
}

// =============================================================================
// Risk Snapshot ID
// =============================================================================

/// Unique identifier for a risk snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RiskSnapshotId(pub String);

impl RiskSnapshotId {
    /// Derive from portfolio digest + policy fingerprint + ts_ns.
    pub fn derive(portfolio_digest: &str, policy_fingerprint: &str, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"risk_snapshot:");
        hasher.update(portfolio_digest.as_bytes());
        hasher.update(b":");
        hasher.update(policy_fingerprint.as_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for RiskSnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Risk Snapshot
// =============================================================================

/// Deterministic snapshot of risk state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSnapshot {
    /// Schema version.
    pub schema_version: String,

    /// Unique snapshot ID (deterministic).
    pub snapshot_id: RiskSnapshotId,

    /// Timestamp (nanoseconds).
    pub ts_ns: i64,

    /// Source portfolio snapshot digest (links to 14.3).
    pub portfolio_snapshot_digest: String,

    /// Policy fingerprint used for evaluation.
    pub policy_fingerprint: String,

    /// Computed exposures.
    pub exposures: ExposureMetrics,

    /// Active violations (empty = compliant).
    pub violations: Vec<ViolationType>,

    /// Is portfolio currently compliant?
    pub is_compliant: bool,

    /// Deterministic digest.
    pub digest: String,
}

impl RiskSnapshot {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.snapshot_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.portfolio_snapshot_digest.as_bytes());
        hasher.update(self.policy_fingerprint.as_bytes());

        // Exposures
        hasher.update(self.exposures.total_notional_mantissa.to_le_bytes());
        hasher.update([self.exposures.notional_exponent as u8]);
        hasher.update(self.exposures.total_position_count.to_le_bytes());

        // Violations
        hasher.update((self.violations.len() as u32).to_le_bytes());
        for violation in &self.violations {
            hasher.update(violation.canonical_bytes());
        }

        hasher.update([self.is_compliant as u8]);

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Risk Decision Types
// =============================================================================

/// Risk decision outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskDecisionStatus {
    /// Execution may proceed.
    Allow,

    /// This specific request is rejected.
    Reject,

    /// Global halt — no execution until cleared.
    Halt,
}

impl std::fmt::Display for RiskDecisionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskDecisionStatus::Allow => write!(f, "ALLOW"),
            RiskDecisionStatus::Reject => write!(f, "REJECT"),
            RiskDecisionStatus::Halt => write!(f, "HALT"),
        }
    }
}

/// Scope of risk decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskDecisionScope {
    /// Decision applies to specific order intent.
    Order { intent_id: String },

    /// Decision applies to specific strategy.
    Strategy { strategy_id: String },

    /// Decision applies to specific bucket.
    Bucket { bucket_id: String },

    /// Decision applies globally (halt all execution).
    Global,
}

impl RiskDecisionScope {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        match self {
            RiskDecisionScope::Order { intent_id } => {
                bytes.extend_from_slice(b"order:");
                bytes.extend_from_slice(intent_id.as_bytes());
            }
            RiskDecisionScope::Strategy { strategy_id } => {
                bytes.extend_from_slice(b"strategy:");
                bytes.extend_from_slice(strategy_id.as_bytes());
            }
            RiskDecisionScope::Bucket { bucket_id } => {
                bytes.extend_from_slice(b"bucket:");
                bytes.extend_from_slice(bucket_id.as_bytes());
            }
            RiskDecisionScope::Global => {
                bytes.extend_from_slice(b"global");
            }
        }
        bytes
    }
}

// =============================================================================
// Risk Decision ID
// =============================================================================

/// Unique identifier for a risk decision.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RiskDecisionId(pub String);

impl RiskDecisionId {
    /// Derive from scope + ts_ns + seq.
    pub fn derive(scope: &RiskDecisionScope, ts_ns: i64, seq: u64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"risk_decision:");
        hasher.update(scope.canonical_bytes());
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        hasher.update(b":");
        hasher.update(seq.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for RiskDecisionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Risk Decision
// =============================================================================

/// Runtime risk gate decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDecision {
    /// Schema version.
    pub schema_version: String,

    /// Unique decision ID (deterministic).
    pub decision_id: RiskDecisionId,

    /// Timestamp (nanoseconds).
    pub ts_ns: i64,

    /// Decision scope.
    pub scope: RiskDecisionScope,

    /// Decision status.
    pub status: RiskDecisionStatus,

    /// Violation reasons (empty for Allow).
    pub reasons: Vec<ViolationType>,

    /// Source risk snapshot digest.
    pub risk_snapshot_digest: String,

    /// Deterministic digest.
    pub digest: String,
}

impl RiskDecision {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.decision_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());
        hasher.update(self.scope.canonical_bytes());
        hasher.update(format!("{}", self.status).as_bytes());

        hasher.update((self.reasons.len() as u32).to_le_bytes());
        for reason in &self.reasons {
            hasher.update(reason.canonical_bytes());
        }

        hasher.update(self.risk_snapshot_digest.as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Is this decision allowing execution?
    pub fn is_allowed(&self) -> bool {
        self.status == RiskDecisionStatus::Allow
    }
}

// =============================================================================
// Budget View (for committed ratio check)
// =============================================================================

/// Read-only view of budget state for risk evaluation.
/// Does not mutate budget - just provides allocated/committed values.
#[derive(Debug, Clone)]
pub struct BudgetView {
    pub strategy_id: String,
    pub bucket_id: String,
    pub allocated_mantissa: i128,
    pub committed_mantissa: i128,
    pub reserved_mantissa: i128,
    pub exponent: i8,
}

impl BudgetView {
    /// Create from ExecutionBudget.
    pub fn from_budget(budget: &ExecutionBudget) -> Self {
        Self {
            strategy_id: budget.strategy_id.0.clone(),
            bucket_id: budget.bucket_id.0.clone(),
            allocated_mantissa: budget.allocated_capital.mantissa,
            committed_mantissa: budget.committed_capital.mantissa,
            reserved_mantissa: budget.reserved_capital.mantissa,
            exponent: budget.allocated_capital.exponent,
        }
    }

    /// Compute committed ratio (mantissa with exp -4).
    /// ratio = committed / allocated
    /// Returns None if allocated is zero.
    pub fn committed_ratio_mantissa(&self) -> Option<i64> {
        if self.allocated_mantissa == 0 {
            return None;
        }
        // ratio = (committed * 10000) / allocated
        let ratio = (self.committed_mantissa * 10000) / self.allocated_mantissa;
        Some(ratio as i64)
    }
}

// =============================================================================
// Risk Evaluator
// =============================================================================

/// Risk evaluation engine (read-only, deterministic).
pub struct RiskEvaluator {
    policy: RiskPolicy,
    decision_seq: u64,
}

impl RiskEvaluator {
    /// Create with policy.
    pub fn new(policy: RiskPolicy) -> Self {
        Self {
            policy,
            decision_seq: 0,
        }
    }

    /// Get policy fingerprint.
    pub fn policy_fingerprint(&self) -> String {
        self.policy.fingerprint()
    }

    /// Get policy reference.
    pub fn policy(&self) -> &RiskPolicy {
        &self.policy
    }

    /// Evaluate risk snapshot from portfolio state.
    /// Pure function: same inputs → same snapshot.
    pub fn evaluate_snapshot(
        &self,
        portfolio_snapshot: &PortfolioSnapshot,
        ts_ns: i64,
    ) -> RiskSnapshot {
        // 1. Compute exposures
        let exposures = ExposureMetrics::from_portfolio_snapshot(
            portfolio_snapshot,
            self.policy.notional_exponent,
        );

        // 2. Check all limits, collect violations
        let mut violations = Vec::new();

        // 2a. Symbol notional checks
        for (symbol, &notional) in &exposures.notional_by_symbol {
            if notional > self.policy.max_symbol_notional_mantissa {
                violations.push(ViolationType::SymbolNotionalExceeded {
                    symbol: symbol.clone(),
                    current_mantissa: notional,
                    limit_mantissa: self.policy.max_symbol_notional_mantissa,
                });
            }
        }

        // 2b. Strategy notional checks
        for (strategy_id, &notional) in &exposures.notional_by_strategy {
            if notional > self.policy.max_strategy_notional_mantissa {
                violations.push(ViolationType::StrategyNotionalExceeded {
                    strategy_id: strategy_id.clone(),
                    current_mantissa: notional,
                    limit_mantissa: self.policy.max_strategy_notional_mantissa,
                });
            }
        }

        // 2c. Bucket notional checks
        for (bucket_id, &notional) in &exposures.notional_by_bucket {
            if notional > self.policy.max_bucket_notional_mantissa {
                violations.push(ViolationType::BucketNotionalExceeded {
                    bucket_id: bucket_id.clone(),
                    current_mantissa: notional,
                    limit_mantissa: self.policy.max_bucket_notional_mantissa,
                });
            }
        }

        // 2d. Portfolio notional check
        if exposures.total_notional_mantissa > self.policy.max_portfolio_notional_mantissa {
            violations.push(ViolationType::PortfolioNotionalExceeded {
                current_mantissa: exposures.total_notional_mantissa,
                limit_mantissa: self.policy.max_portfolio_notional_mantissa,
            });
        }

        // 2e. Strategy position count checks
        for (strategy_id, &count) in &exposures.position_count_by_strategy {
            if count > self.policy.max_positions_per_strategy {
                violations.push(ViolationType::StrategyPositionCountExceeded {
                    strategy_id: strategy_id.clone(),
                    current: count,
                    limit: self.policy.max_positions_per_strategy,
                });
            }
        }

        // 2f. Bucket position count checks
        for (bucket_id, &count) in &exposures.position_count_by_bucket {
            if count > self.policy.max_positions_per_bucket {
                violations.push(ViolationType::BucketPositionCountExceeded {
                    bucket_id: bucket_id.clone(),
                    current: count,
                    limit: self.policy.max_positions_per_bucket,
                });
            }
        }

        // 2g. Portfolio position count check
        if exposures.total_position_count > self.policy.max_portfolio_positions {
            violations.push(ViolationType::PortfolioPositionCountExceeded {
                current: exposures.total_position_count,
                limit: self.policy.max_portfolio_positions,
            });
        }

        // 2h. Venue checks (if allowed_venues is non-empty)
        if !self.policy.allowed_venues.is_empty() {
            for venue_str in exposures.notional_by_venue.keys() {
                let venue_allowed = self
                    .policy
                    .allowed_venues
                    .iter()
                    .any(|v| format!("{}", v) == *venue_str);
                if !venue_allowed {
                    violations.push(ViolationType::VenueNotAllowed {
                        venue: venue_str.clone(),
                    });
                }
            }
        }

        // 3. Build snapshot
        let is_compliant = violations.is_empty();
        let policy_fingerprint = self.policy.fingerprint();
        let snapshot_id =
            RiskSnapshotId::derive(&portfolio_snapshot.digest, &policy_fingerprint, ts_ns);

        let mut snapshot = RiskSnapshot {
            schema_version: RISK_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id,
            ts_ns,
            portfolio_snapshot_digest: portfolio_snapshot.digest.clone(),
            policy_fingerprint,
            exposures,
            violations,
            is_compliant,
            digest: String::new(),
        };
        snapshot.digest = snapshot.compute_digest();
        snapshot
    }

    /// Make risk decision for order intent.
    /// Checks: current exposure + budget coupling vs limits.
    pub fn evaluate_order(
        &mut self,
        risk_snapshot: &RiskSnapshot,
        intent: &OrderIntentEvent,
        budget_view: Option<&BudgetView>,
        ts_ns: i64,
    ) -> RiskDecision {
        self.decision_seq += 1;

        let scope = RiskDecisionScope::Order {
            intent_id: intent.intent_id.0.clone(),
        };

        let mut reasons = Vec::new();

        // Check 1: Is snapshot already non-compliant?
        if !risk_snapshot.is_compliant {
            // Copy existing violations as reasons
            reasons.extend(risk_snapshot.violations.clone());
        }

        // Check 2: Budget coupling (committed ratio)
        if let Some(bv) = budget_view
            && let Some(ratio) = bv.committed_ratio_mantissa()
            && ratio > self.policy.max_committed_ratio_mantissa
        {
            reasons.push(ViolationType::CommittedRatioExceeded {
                strategy_id: bv.strategy_id.clone(),
                bucket_id: bv.bucket_id.clone(),
                current_ratio_mantissa: ratio,
                limit_ratio_mantissa: self.policy.max_committed_ratio_mantissa,
            });
        }

        // Determine status
        let status = if reasons.is_empty() {
            RiskDecisionStatus::Allow
        } else {
            // Check for portfolio-level breach → Halt
            let has_portfolio_breach = reasons.iter().any(|r| {
                matches!(
                    r,
                    ViolationType::PortfolioNotionalExceeded { .. }
                        | ViolationType::PortfolioPositionCountExceeded { .. }
                )
            });

            if has_portfolio_breach {
                RiskDecisionStatus::Halt
            } else {
                RiskDecisionStatus::Reject
            }
        };

        let decision_id = RiskDecisionId::derive(&scope, ts_ns, self.decision_seq);

        let mut decision = RiskDecision {
            schema_version: RISK_DECISION_SCHEMA_VERSION.to_string(),
            decision_id,
            ts_ns,
            scope,
            status,
            reasons,
            risk_snapshot_digest: risk_snapshot.digest.clone(),
            digest: String::new(),
        };
        decision.digest = decision.compute_digest();
        decision
    }

    /// Check if global halt should be active based on risk snapshot.
    pub fn is_halted(&self, risk_snapshot: &RiskSnapshot) -> bool {
        risk_snapshot.violations.iter().any(|v| {
            matches!(
                v,
                ViolationType::PortfolioNotionalExceeded { .. }
                    | ViolationType::PortfolioPositionCountExceeded { .. }
            )
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position_keeper::{PositionKeeper, PositionState, SnapshotId};
    use quantlaxmi_models::{IntentId, PositionKey, PositionSide};

    fn make_portfolio_with_position(
        strategy_id: &str,
        bucket_id: &str,
        symbol: &str,
        qty_mantissa: i128,
        price_mantissa: i128,
    ) -> PortfolioSnapshot {
        let mut keeper = PositionKeeper::new(-8);
        let key = PositionKey::new(strategy_id, bucket_id, symbol, PositionVenue::BinancePerp);
        let state = PositionState::new(
            key.clone(),
            PositionSide::Long,
            qty_mantissa,
            -8,
            price_mantissa,
            -8,
            -8,
            0,
            -8,
            1000,
        );
        keeper.ledger.upsert_position(state);
        keeper.snapshot(SnapshotId::new("test_snap"), 1000)
    }

    fn make_intent(intent_id: &str) -> OrderIntentEvent {
        OrderIntentEvent {
            schema_version: "test".to_string(),
            ts_ns: 1000,
            intent_id: IntentId(intent_id.to_string()),
            strategy_id: "strategy_001".to_string(),
            bucket_id: "bucket_001".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: quantlaxmi_models::ExecutionSide::Buy,
            order_type: quantlaxmi_models::ExecutionOrderType::Limit,
            quantity_mantissa: 100_000_000,
            quantity_exponent: -8,
            limit_price_mantissa: Some(50000_00000000),
            price_exponent: -8,
            reference_price_mantissa: 50000_00000000,
            parent_decision_id: "decision_001".to_string(),
            state: quantlaxmi_models::LiveOrderState::IntentCreated,
            digest: "test".to_string(),
        }
    }

    #[test]
    fn test_normalize_notional_same_exponent() {
        assert_eq!(normalize_notional(1000, -8, -8), 1000);
    }

    #[test]
    fn test_normalize_notional_shift_up() {
        // source=-8, target=-6: multiply by 100
        assert_eq!(normalize_notional(1000, -6, -8), 100_000);
    }

    #[test]
    fn test_normalize_notional_shift_down() {
        // source=-8, target=-6: divide by 100
        assert_eq!(normalize_notional(1000, -8, -6), 10);
    }

    #[test]
    fn test_compute_notional() {
        // qty=1 BTC (100_000_000 with exp -8), price=50000 USDT (50000_00000000 with exp -8)
        // raw notional = 100_000_000 * 50000_00000000 = 5_000_000_000_000_000_000
        // combined exp = -8 + -8 = -16
        // target exp = -2 (cents)
        // shift = -16 - (-2) = -14, divide by 10^14
        // result = 5_000_000_000_000_000_000 / 100_000_000_000_000 = 50_000_00 = 5000000
        let notional = compute_notional(100_000_000, -8, 50000_00000000, -8, -2);
        assert_eq!(notional, 5_000_000); // 50,000.00 USDT
    }

    #[test]
    fn test_exposure_aggregation_deterministic() {
        let snap1 = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000,
            50000_00000000,
        );
        let snap2 = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000,
            50000_00000000,
        );

        let metrics1 = ExposureMetrics::from_portfolio_snapshot(&snap1, -2);
        let metrics2 = ExposureMetrics::from_portfolio_snapshot(&snap2, -2);

        assert_eq!(
            metrics1.total_notional_mantissa,
            metrics2.total_notional_mantissa
        );
        assert_eq!(metrics1.total_position_count, metrics2.total_position_count);
    }

    #[test]
    fn test_policy_fingerprint_deterministic() {
        let policy1 = RiskPolicy::conservative();
        let policy2 = RiskPolicy::conservative();

        assert_eq!(policy1.fingerprint(), policy2.fingerprint());
    }

    #[test]
    fn test_policy_fingerprint_different_for_different_policies() {
        let conservative = RiskPolicy::conservative();
        let aggressive = RiskPolicy::aggressive();

        assert_ne!(conservative.fingerprint(), aggressive.fingerprint());
    }

    #[test]
    fn test_risk_snapshot_digest_deterministic() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000,
            50000_00000000,
        );
        let policy = RiskPolicy::aggressive(); // Use aggressive so we don't get violations
        let evaluator = RiskEvaluator::new(policy);

        let snap1 = evaluator.evaluate_snapshot(&portfolio, 1000);
        let snap2 = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert_eq!(snap1.digest, snap2.digest);
    }

    #[test]
    fn test_risk_decision_digest_deterministic() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000,
            50000_00000000,
        );
        let policy = RiskPolicy::aggressive();
        let mut evaluator1 = RiskEvaluator::new(policy.clone());
        let mut evaluator2 = RiskEvaluator::new(policy);

        let snap = evaluator1.evaluate_snapshot(&portfolio, 1000);
        let intent = make_intent("intent_001");

        let decision1 = evaluator1.evaluate_order(&snap, &intent, None, 1000);
        let decision2 = evaluator2.evaluate_order(&snap, &intent, None, 1000);

        assert_eq!(decision1.digest, decision2.digest);
    }

    #[test]
    fn test_symbol_notional_violation() {
        // Create position with 50,000 USDT notional
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000,
            50000_00000000,
        );

        // Conservative policy: max 10,000 USDT per symbol
        let policy = RiskPolicy::conservative();
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(!snapshot.is_compliant);
        assert!(snapshot.violations.iter().any(|v| matches!(
            v,
            ViolationType::SymbolNotionalExceeded { symbol, .. } if symbol == "BTCUSDT"
        )));
    }

    #[test]
    fn test_strategy_notional_violation() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000,
            50000_00000000,
        );

        let policy = RiskPolicy::conservative();
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(snapshot.violations.iter().any(|v| matches!(
            v,
            ViolationType::StrategyNotionalExceeded { strategy_id, .. } if strategy_id == "strategy_001"
        )));
    }

    #[test]
    fn test_bucket_notional_violation() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            200_000_000, // 2 BTC
            50000_00000000,
        );

        let policy = RiskPolicy::conservative();
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(snapshot.violations.iter().any(|v| matches!(
            v,
            ViolationType::BucketNotionalExceeded { bucket_id, .. } if bucket_id == "bucket_001"
        )));
    }

    #[test]
    fn test_portfolio_notional_violation() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            500_000_000, // 5 BTC = 250,000 USDT
            50000_00000000,
        );

        let policy = RiskPolicy::conservative(); // max 100,000 USDT portfolio
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(
            snapshot
                .violations
                .iter()
                .any(|v| matches!(v, ViolationType::PortfolioNotionalExceeded { .. }))
        );
    }

    #[test]
    fn test_strategy_position_count_violation() {
        let mut keeper = PositionKeeper::new(-8);

        // Add 5 positions for same strategy (conservative limit is 3)
        for i in 0..5 {
            let key = PositionKey::new(
                "strategy_001",
                "bucket_001",
                &format!("SYMBOL{}", i),
                PositionVenue::BinancePerp,
            );
            let state = PositionState::new(
                key,
                PositionSide::Long,
                10_000_000,
                -8,
                1000_00000000,
                -8,
                -8,
                0,
                -8,
                1000,
            );
            keeper.ledger.upsert_position(state);
        }

        let portfolio = keeper.snapshot(SnapshotId::new("test"), 1000);
        let policy = RiskPolicy::conservative();
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(snapshot.violations.iter().any(|v| matches!(
            v,
            ViolationType::StrategyPositionCountExceeded { strategy_id, current, limit }
                if strategy_id == "strategy_001" && *current == 5 && *limit == 3
        )));
    }

    #[test]
    fn test_bucket_position_count_violation() {
        let mut keeper = PositionKeeper::new(-8);

        // Add 7 positions for same bucket (conservative limit is 5)
        for i in 0..7 {
            let key = PositionKey::new(
                &format!("strategy_{}", i),
                "bucket_001",
                &format!("SYMBOL{}", i),
                PositionVenue::BinancePerp,
            );
            let state = PositionState::new(
                key,
                PositionSide::Long,
                10_000_000,
                -8,
                1000_00000000,
                -8,
                -8,
                0,
                -8,
                1000,
            );
            keeper.ledger.upsert_position(state);
        }

        let portfolio = keeper.snapshot(SnapshotId::new("test"), 1000);
        let policy = RiskPolicy::conservative();
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(snapshot.violations.iter().any(|v| matches!(
            v,
            ViolationType::BucketPositionCountExceeded { bucket_id, current, limit }
                if bucket_id == "bucket_001" && *current == 7 && *limit == 5
        )));
    }

    #[test]
    fn test_portfolio_position_count_violation() {
        let mut keeper = PositionKeeper::new(-8);

        // Add 12 positions (conservative limit is 10)
        for i in 0..12 {
            let key = PositionKey::new(
                &format!("strategy_{}", i),
                &format!("bucket_{}", i),
                &format!("SYMBOL{}", i),
                PositionVenue::BinancePerp,
            );
            let state = PositionState::new(
                key,
                PositionSide::Long,
                10_000_000,
                -8,
                1000_00000000,
                -8,
                -8,
                0,
                -8,
                1000,
            );
            keeper.ledger.upsert_position(state);
        }

        let portfolio = keeper.snapshot(SnapshotId::new("test"), 1000);
        let policy = RiskPolicy::conservative();
        let evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(snapshot.violations.iter().any(|v| matches!(
            v,
            ViolationType::PortfolioPositionCountExceeded { current, limit }
                if *current == 12 && *limit == 10
        )));
    }

    #[test]
    fn test_venue_not_allowed_violation() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            10_000_000,
            1000_00000000,
        );

        // Policy that only allows Paper venue
        let mut policy = RiskPolicy::aggressive();
        policy.allowed_venues = vec![PositionVenue::Paper];

        let evaluator = RiskEvaluator::new(policy);
        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        assert!(snapshot.violations.iter().any(
            |v| matches!(v, ViolationType::VenueNotAllowed { venue } if venue == "BinancePerp")
        ));
    }

    #[test]
    fn test_allow_when_compliant() {
        // Small position that doesn't violate conservative limits
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            1_000_000, // 0.01 BTC = 500 USDT
            50000_00000000,
        );

        let policy = RiskPolicy::conservative();
        let mut evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);
        assert!(snapshot.is_compliant);

        let intent = make_intent("intent_001");
        let decision = evaluator.evaluate_order(&snapshot, &intent, None, 1000);

        assert_eq!(decision.status, RiskDecisionStatus::Allow);
        assert!(decision.reasons.is_empty());
    }

    #[test]
    fn test_reject_when_violated() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            100_000_000, // 1 BTC = 50,000 USDT
            50000_00000000,
        );

        let policy = RiskPolicy::conservative();
        let mut evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);
        assert!(!snapshot.is_compliant);

        let intent = make_intent("intent_001");
        let decision = evaluator.evaluate_order(&snapshot, &intent, None, 1000);

        assert_eq!(decision.status, RiskDecisionStatus::Reject);
        assert!(!decision.reasons.is_empty());
    }

    #[test]
    fn test_halt_on_portfolio_breach() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            500_000_000, // 5 BTC = 250,000 USDT (exceeds 100k portfolio limit)
            50000_00000000,
        );

        let policy = RiskPolicy::conservative();
        let mut evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);
        assert!(evaluator.is_halted(&snapshot));

        let intent = make_intent("intent_001");
        let decision = evaluator.evaluate_order(&snapshot, &intent, None, 1000);

        assert_eq!(decision.status, RiskDecisionStatus::Halt);
    }

    #[test]
    fn test_risk_snapshot_id_derivation() {
        let id1 = RiskSnapshotId::derive("portfolio_digest", "policy_fp", 1000);
        let id2 = RiskSnapshotId::derive("portfolio_digest", "policy_fp", 1000);

        assert_eq!(id1, id2);
        assert_eq!(id1.0.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_risk_decision_id_derivation() {
        let scope = RiskDecisionScope::Order {
            intent_id: "intent_001".to_string(),
        };
        let id1 = RiskDecisionId::derive(&scope, 1000, 1);
        let id2 = RiskDecisionId::derive(&scope, 1000, 1);

        assert_eq!(id1, id2);
        assert_eq!(id1.0.len(), 64);
    }

    #[test]
    fn test_committed_ratio_exceeded() {
        let portfolio = make_portfolio_with_position(
            "strategy_001",
            "bucket_001",
            "BTCUSDT",
            1_000_000, // Small position
            50000_00000000,
        );

        let policy = RiskPolicy::conservative(); // 50% max committed ratio
        let mut evaluator = RiskEvaluator::new(policy);

        let snapshot = evaluator.evaluate_snapshot(&portfolio, 1000);

        // Create budget view with 90% committed ratio
        let budget_view = BudgetView {
            strategy_id: "strategy_001".to_string(),
            bucket_id: "bucket_001".to_string(),
            allocated_mantissa: 100_000_000,
            committed_mantissa: 90_000_000, // 90%
            reserved_mantissa: 0,
            exponent: -2,
        };

        let intent = make_intent("intent_001");
        let decision = evaluator.evaluate_order(&snapshot, &intent, Some(&budget_view), 1000);

        assert_eq!(decision.status, RiskDecisionStatus::Reject);
        assert!(decision.reasons.iter().any(|r| matches!(
            r,
            ViolationType::CommittedRatioExceeded { current_ratio_mantissa, limit_ratio_mantissa, .. }
                if *current_ratio_mantissa == 9000 && *limit_ratio_mantissa == 5000
        )));
    }

    #[test]
    fn test_violation_codes() {
        assert_eq!(
            ViolationType::SymbolNotionalExceeded {
                symbol: "X".to_string(),
                current_mantissa: 0,
                limit_mantissa: 0
            }
            .code(),
            "SYMBOL_NOTIONAL"
        );
        assert_eq!(
            ViolationType::PortfolioPositionCountExceeded {
                current: 0,
                limit: 0
            }
            .code(),
            "PORTFOLIO_POSITIONS"
        );
        assert_eq!(
            ViolationType::VenueNotAllowed {
                venue: "X".to_string()
            }
            .code(),
            "VENUE_NOT_ALLOWED"
        );
    }
}
