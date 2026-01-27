//! # Intent Shaping Layer (Phase 15.3)
//!
//! Deterministic intent transformation based on equity state and risk policy.
//!
//! ## Core Question
//! "Given equity state and risk policy, how should an order intent be transformed
//! (blocked, mode-restricted, or capped) before budget reservation?"
//!
//! ## Hard Laws
//! - L1: Deterministic Replay — Same inputs → identical transform
//! - L2: Fixed-Point Only — All caps, thresholds use i128 mantissa + i8 exponent
//! - L3: No Sizing Alpha — Shaping is policy enforcement, not optimization
//! - L4: Audit-First — Every transform produces deterministic OrderIntentTransform
//! - L5: Non-Mutating — Original intent preserved; transform describes delta
//!
//! ## Shaping Rules (Frozen)
//! 1. Block if any equity/risk violation is_halt()
//! 2. Reduce-only mode if DD/leverage/stale exceed thresholds AND intent = Increase
//! 3. Notional caps (smallest wins, floor division, Increase only)
//! 4. Reduce/Close exempt from caps

use crate::mtm_drawdown::{DrawdownSnapshot, EquityViolationType, MtmSnapshot};
use crate::risk_exposure::{RiskSnapshot, ViolationType};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

pub const INTENT_TRANSFORM_SCHEMA_VERSION: &str = "intent_transform_v1.0";
pub const SHAPING_POLICY_SCHEMA_VERSION: &str = "shaping_policy_v1.0";

// =============================================================================
// Intent Type
// =============================================================================

/// Intent types for mode restriction.
/// Discriminant values are frozen for canonical encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
pub enum IntentType {
    /// Open new position or increase existing.
    Increase = 0,
    /// Reduce existing position.
    Reduce = 1,
    /// Close entire position.
    Close = 2,
    /// Any direction (unrestricted).
    Any = 3,
}

impl IntentType {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            IntentType::Increase => 0,
            IntentType::Reduce => 1,
            IntentType::Close => 2,
            IntentType::Any => 3,
        }
    }

    /// Check if this is an increase intent.
    pub fn is_increase(&self) -> bool {
        matches!(self, IntentType::Increase)
    }

    /// Check if this is a reduce or close intent.
    pub fn is_reduce_or_close(&self) -> bool {
        matches!(self, IntentType::Reduce | IntentType::Close)
    }
}

// =============================================================================
// Block Reason
// =============================================================================

/// Why intent was blocked.
/// Discriminant values are frozen for canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum BlockReason {
    /// Equity violations triggered HALT (discriminant 0).
    EquityHalt { violations: Vec<EquityViolationType> },
    /// Risk violations triggered REJECT/HALT (discriminant 1).
    RiskReject { violations: Vec<ViolationType> },
    /// Reduce-only mode active but intent is increase (discriminant 2).
    ReduceOnlyViolation,
}

impl BlockReason {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            BlockReason::EquityHalt { .. } => 0,
            BlockReason::RiskReject { .. } => 1,
            BlockReason::ReduceOnlyViolation => 2,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            BlockReason::EquityHalt { violations } => {
                // Sort violations for determinism
                let mut sorted = violations.clone();
                sorted.sort();
                bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
                for v in &sorted {
                    let vb = v.canonical_bytes();
                    bytes.extend_from_slice(&(vb.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(&vb);
                }
            }
            BlockReason::RiskReject { violations } => {
                // Sort violations for determinism
                let mut sorted = violations.clone();
                sorted.sort();
                bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
                for v in &sorted {
                    let vb = v.canonical_bytes();
                    bytes.extend_from_slice(&(vb.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(&vb);
                }
            }
            BlockReason::ReduceOnlyViolation => {
                // No additional data
            }
        }

        bytes
    }
}

// =============================================================================
// Mode Restrict Reason
// =============================================================================

/// Why intent mode was restricted.
/// Discriminant values are frozen for canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ModeRestrictReason {
    /// Drawdown above warning threshold → reduce-only mode (discriminant 0).
    DrawdownWarningActive {
        current_pct_mantissa: i64,
        threshold_pct_mantissa: i64,
    },
    /// Leverage above soft limit → reduce-only mode (discriminant 1).
    LeverageSoftLimitActive {
        current_mantissa: i64,
        soft_limit_mantissa: i64,
    },
    /// Stale prices detected → reduce-only mode (discriminant 2).
    StalePriceActive { stale_count: u32 },
}

impl ModeRestrictReason {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            ModeRestrictReason::DrawdownWarningActive { .. } => 0,
            ModeRestrictReason::LeverageSoftLimitActive { .. } => 1,
            ModeRestrictReason::StalePriceActive { .. } => 2,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            ModeRestrictReason::DrawdownWarningActive {
                current_pct_mantissa,
                threshold_pct_mantissa,
            } => {
                bytes.extend_from_slice(&current_pct_mantissa.to_le_bytes());
                bytes.extend_from_slice(&threshold_pct_mantissa.to_le_bytes());
            }
            ModeRestrictReason::LeverageSoftLimitActive {
                current_mantissa,
                soft_limit_mantissa,
            } => {
                bytes.extend_from_slice(&current_mantissa.to_le_bytes());
                bytes.extend_from_slice(&soft_limit_mantissa.to_le_bytes());
            }
            ModeRestrictReason::StalePriceActive { stale_count } => {
                bytes.extend_from_slice(&stale_count.to_le_bytes());
            }
        }

        bytes
    }
}

// =============================================================================
// Cap Reason
// =============================================================================

/// Why notional was capped.
/// Discriminant values are frozen for canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum CapReason {
    /// Leverage headroom cap (discriminant 0).
    LeverageHeadroom {
        max_leverage_mantissa: i64,
        current_leverage_mantissa: i64,
        headroom_mantissa: i128,
    },
    /// Equity-proportional cap (discriminant 1).
    EquityProportional {
        max_pct_mantissa: i64,
        equity_mantissa: i128,
        cap_mantissa: i128,
    },
    /// Drawdown recovery cap (discriminant 2).
    DrawdownRecoveryCap {
        drawdown_pct_mantissa: i64,
        scale_factor_mantissa: i64,
    },
}

impl CapReason {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            CapReason::LeverageHeadroom { .. } => 0,
            CapReason::EquityProportional { .. } => 1,
            CapReason::DrawdownRecoveryCap { .. } => 2,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            CapReason::LeverageHeadroom {
                max_leverage_mantissa,
                current_leverage_mantissa,
                headroom_mantissa,
            } => {
                bytes.extend_from_slice(&max_leverage_mantissa.to_le_bytes());
                bytes.extend_from_slice(&current_leverage_mantissa.to_le_bytes());
                bytes.extend_from_slice(&headroom_mantissa.to_le_bytes());
            }
            CapReason::EquityProportional {
                max_pct_mantissa,
                equity_mantissa,
                cap_mantissa,
            } => {
                bytes.extend_from_slice(&max_pct_mantissa.to_le_bytes());
                bytes.extend_from_slice(&equity_mantissa.to_le_bytes());
                bytes.extend_from_slice(&cap_mantissa.to_le_bytes());
            }
            CapReason::DrawdownRecoveryCap {
                drawdown_pct_mantissa,
                scale_factor_mantissa,
            } => {
                bytes.extend_from_slice(&drawdown_pct_mantissa.to_le_bytes());
                bytes.extend_from_slice(&scale_factor_mantissa.to_le_bytes());
            }
        }

        bytes
    }
}

// =============================================================================
// Transform Action
// =============================================================================

/// Transform action applied to intent.
/// Discriminant values are frozen for canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum TransformAction {
    /// Intent proceeds unchanged (discriminant 0).
    PassThrough,

    /// Intent blocked (discriminant 1).
    Block { reason: BlockReason },

    /// Intent type downgraded (discriminant 2).
    ModeRestrict {
        original_intent_type: IntentType,
        allowed_intent_type: IntentType,
        reason: ModeRestrictReason,
    },

    /// Notional capped (discriminant 3).
    NotionalCap {
        original_notional_mantissa: i128,
        capped_notional_mantissa: i128,
        notional_exponent: i8,
        cap_reason: CapReason,
    },

    /// Multiple transforms applied (discriminant 4).
    Composite { actions: Vec<TransformAction> },
}

impl TransformAction {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            TransformAction::PassThrough => 0,
            TransformAction::Block { .. } => 1,
            TransformAction::ModeRestrict { .. } => 2,
            TransformAction::NotionalCap { .. } => 3,
            TransformAction::Composite { .. } => 4,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            TransformAction::PassThrough => {
                // No additional data
            }
            TransformAction::Block { reason } => {
                let rb = reason.canonical_bytes();
                bytes.extend_from_slice(&(rb.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&rb);
            }
            TransformAction::ModeRestrict {
                original_intent_type,
                allowed_intent_type,
                reason,
            } => {
                bytes.push(original_intent_type.discriminant());
                bytes.push(allowed_intent_type.discriminant());
                let rb = reason.canonical_bytes();
                bytes.extend_from_slice(&(rb.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&rb);
            }
            TransformAction::NotionalCap {
                original_notional_mantissa,
                capped_notional_mantissa,
                notional_exponent,
                cap_reason,
            } => {
                bytes.extend_from_slice(&original_notional_mantissa.to_le_bytes());
                bytes.extend_from_slice(&capped_notional_mantissa.to_le_bytes());
                bytes.push(*notional_exponent as u8);
                let crb = cap_reason.canonical_bytes();
                bytes.extend_from_slice(&(crb.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&crb);
            }
            TransformAction::Composite { actions } => {
                // Sort actions for determinism
                let mut sorted = actions.clone();
                sorted.sort();
                bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
                for action in &sorted {
                    let ab = action.canonical_bytes();
                    bytes.extend_from_slice(&(ab.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(&ab);
                }
            }
        }

        bytes
    }

    /// Check if this action blocks the intent.
    pub fn is_block(&self) -> bool {
        matches!(self, TransformAction::Block { .. })
    }
}

// =============================================================================
// Transform ID
// =============================================================================

/// Unique transform ID (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransformId(pub String);

impl TransformId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from intent_id + snapshot digests + policy fingerprint.
    pub fn derive(
        intent_id: &str,
        risk_digest: &str,
        mtm_digest: &str,
        drawdown_digest: &str,
        policy_fingerprint: &str,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"intent_transform:");
        hasher.update((intent_id.len() as u32).to_le_bytes());
        hasher.update(intent_id.as_bytes());
        hasher.update(b":");
        hasher.update(risk_digest.as_bytes());
        hasher.update(b":");
        hasher.update(mtm_digest.as_bytes());
        hasher.update(b":");
        hasher.update(drawdown_digest.as_bytes());
        hasher.update(b":");
        hasher.update(policy_fingerprint.as_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for TransformId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Order Intent Transform
// =============================================================================

/// Complete transform artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderIntentTransform {
    /// Schema version.
    pub schema_version: String,

    /// Unique transform ID (deterministic).
    pub transform_id: TransformId,

    /// Timestamp of transform (nanoseconds).
    pub ts_ns: i64,

    /// Original intent ID being transformed.
    pub intent_id: String,

    /// Original intent type.
    pub original_intent_type: IntentType,

    /// Original notional (mantissa).
    pub original_notional_mantissa: i128,
    pub notional_exponent: i8,

    /// Transform action applied.
    pub action: TransformAction,

    /// Final intent type after transform (may be same or restricted).
    pub final_intent_type: IntentType,

    /// Final notional after transform (may be same or capped).
    pub final_notional_mantissa: i128,

    /// Is intent allowed to proceed to BudgetManager?
    pub allowed: bool,

    /// Source snapshot digests (for replay verification).
    pub risk_snapshot_digest: String,
    pub mtm_snapshot_digest: String,
    pub drawdown_snapshot_digest: String,

    /// Policy fingerprint used.
    pub policy_fingerprint: String,

    /// Deterministic digest.
    pub digest: String,
}

impl OrderIntentTransform {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.transform_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        hasher.update((self.intent_id.len() as u32).to_le_bytes());
        hasher.update(self.intent_id.as_bytes());

        hasher.update([self.original_intent_type.discriminant()]);
        hasher.update(self.original_notional_mantissa.to_le_bytes());
        hasher.update([self.notional_exponent as u8]);

        let action_bytes = self.action.canonical_bytes();
        hasher.update((action_bytes.len() as u32).to_le_bytes());
        hasher.update(&action_bytes);

        hasher.update([self.final_intent_type.discriminant()]);
        hasher.update(self.final_notional_mantissa.to_le_bytes());
        hasher.update([self.allowed as u8]);

        hasher.update(self.risk_snapshot_digest.as_bytes());
        hasher.update(self.mtm_snapshot_digest.as_bytes());
        hasher.update(self.drawdown_snapshot_digest.as_bytes());
        hasher.update(self.policy_fingerprint.as_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Shaping Policy
// =============================================================================

/// Policy for intent shaping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapingPolicy {
    /// Schema version.
    pub schema_version: String,

    // === Reduce-Only Mode Triggers ===
    /// Drawdown % that triggers reduce-only mode (mantissa, exp -4).
    /// e.g., 500 = 5.00% → at 5% DD, only reduce/close allowed.
    pub reduce_only_drawdown_pct_mantissa: i64,

    /// Leverage that triggers reduce-only mode (mantissa, exp -4).
    /// e.g., 25000 = 2.5x → at 2.5x leverage, only reduce/close allowed.
    pub reduce_only_leverage_mantissa: i64,

    /// Stale position count that triggers reduce-only mode.
    /// e.g., 1 → any stale price triggers reduce-only.
    pub reduce_only_stale_count: u32,

    // === Notional Caps ===
    /// Enable leverage headroom cap (cap intent to stay under max leverage).
    pub enable_leverage_headroom_cap: bool,

    /// Max leverage for headroom calculation (mantissa, exp -4).
    /// e.g., 30000 = 3.0x max leverage.
    pub max_leverage_mantissa: i64,

    /// Enable equity-proportional cap (cap intent to % of equity).
    pub enable_equity_proportional_cap: bool,

    /// Max % of equity per intent (mantissa, exp -4).
    /// e.g., 1000 = 10.00% → single intent can use max 10% of equity.
    pub max_equity_pct_per_intent_mantissa: i64,

    /// Enable drawdown recovery scaling (reduce size as DD increases).
    pub enable_drawdown_recovery_cap: bool,

    /// Drawdown recovery curve: min scale at max drawdown (mantissa, exp -4).
    /// e.g., 2500 = 25% at max DD.
    pub recovery_min_scale_mantissa: i64,

    /// DD % at which min_scale applies (mantissa, exp -4).
    /// e.g., 1000 = 10.00% → at 10% DD, scale = min_scale.
    pub recovery_max_drawdown_pct_mantissa: i64,

    /// Shared exponent for percentages (frozen at -4).
    pub pct_exponent: i8,

    /// Policy fingerprint (derived, never user-supplied).
    pub fingerprint: String,
}

impl ShapingPolicy {
    /// Compute deterministic fingerprint (SHA-256).
    pub fn compute_fingerprint(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.reduce_only_drawdown_pct_mantissa.to_le_bytes());
        hasher.update(self.reduce_only_leverage_mantissa.to_le_bytes());
        hasher.update(self.reduce_only_stale_count.to_le_bytes());
        hasher.update([self.enable_leverage_headroom_cap as u8]);
        hasher.update(self.max_leverage_mantissa.to_le_bytes());
        hasher.update([self.enable_equity_proportional_cap as u8]);
        hasher.update(self.max_equity_pct_per_intent_mantissa.to_le_bytes());
        hasher.update([self.enable_drawdown_recovery_cap as u8]);
        hasher.update(self.recovery_min_scale_mantissa.to_le_bytes());
        hasher.update(self.recovery_max_drawdown_pct_mantissa.to_le_bytes());
        hasher.update([self.pct_exponent as u8]);

        format!("{:x}", hasher.finalize())
    }

    /// Conservative preset: strict reduce-only triggers, aggressive caps.
    pub fn conservative() -> Self {
        let mut policy = Self {
            schema_version: SHAPING_POLICY_SCHEMA_VERSION.to_string(),
            reduce_only_drawdown_pct_mantissa: 300, // 3% DD triggers reduce-only
            reduce_only_leverage_mantissa: 15000,   // 1.5x triggers reduce-only
            reduce_only_stale_count: 1,             // Any stale triggers reduce-only
            enable_leverage_headroom_cap: true,
            max_leverage_mantissa: 20000, // 2.0x max leverage
            enable_equity_proportional_cap: true,
            max_equity_pct_per_intent_mantissa: 500, // 5% of equity per intent
            enable_drawdown_recovery_cap: true,
            recovery_min_scale_mantissa: 2500, // 25% at max DD
            recovery_max_drawdown_pct_mantissa: 500, // At 5% DD
            pct_exponent: -4,
            fingerprint: String::new(),
        };
        policy.fingerprint = policy.compute_fingerprint();
        policy
    }

    /// Moderate preset: balanced triggers and caps.
    pub fn moderate() -> Self {
        let mut policy = Self {
            schema_version: SHAPING_POLICY_SCHEMA_VERSION.to_string(),
            reduce_only_drawdown_pct_mantissa: 700, // 7% DD triggers reduce-only
            reduce_only_leverage_mantissa: 25000,   // 2.5x triggers reduce-only
            reduce_only_stale_count: 3,             // 3 stale triggers reduce-only
            enable_leverage_headroom_cap: true,
            max_leverage_mantissa: 30000, // 3.0x max leverage
            enable_equity_proportional_cap: true,
            max_equity_pct_per_intent_mantissa: 1000, // 10% of equity per intent
            enable_drawdown_recovery_cap: true,
            recovery_min_scale_mantissa: 5000,        // 50% at max DD
            recovery_max_drawdown_pct_mantissa: 1000, // At 10% DD
            pct_exponent: -4,
            fingerprint: String::new(),
        };
        policy.fingerprint = policy.compute_fingerprint();
        policy
    }

    /// Permissive preset: minimal intervention.
    pub fn permissive() -> Self {
        let mut policy = Self {
            schema_version: SHAPING_POLICY_SCHEMA_VERSION.to_string(),
            reduce_only_drawdown_pct_mantissa: 1500, // 15% DD triggers reduce-only
            reduce_only_leverage_mantissa: 40000,    // 4.0x triggers reduce-only
            reduce_only_stale_count: 10,             // 10 stale triggers reduce-only
            enable_leverage_headroom_cap: true,
            max_leverage_mantissa: 50000, // 5.0x max leverage
            enable_equity_proportional_cap: false,
            max_equity_pct_per_intent_mantissa: 2000, // 20% (disabled anyway)
            enable_drawdown_recovery_cap: false,
            recovery_min_scale_mantissa: 7500,        // 75% at max DD (disabled)
            recovery_max_drawdown_pct_mantissa: 2000, // At 20% DD
            pct_exponent: -4,
            fingerprint: String::new(),
        };
        policy.fingerprint = policy.compute_fingerprint();
        policy
    }
}

// =============================================================================
// Intent Shaper
// =============================================================================

/// Deterministic intent shaper (read-only evaluator).
pub struct IntentShaper {
    /// Shaping policy.
    policy: ShapingPolicy,
}

impl IntentShaper {
    /// Create new intent shaper.
    pub fn new(policy: ShapingPolicy) -> Self {
        Self { policy }
    }

    /// Get current policy.
    pub fn policy(&self) -> &ShapingPolicy {
        &self.policy
    }

    /// Evaluate and transform an intent.
    /// DETERMINISTIC: same inputs → identical transform.
    #[allow(clippy::too_many_arguments)]
    pub fn shape(
        &self,
        intent_id: &str,
        intent_type: IntentType,
        intent_notional_mantissa: i128,
        notional_exponent: i8,
        risk_snapshot: &RiskSnapshot,
        mtm_snapshot: &MtmSnapshot,
        drawdown_snapshot: &DrawdownSnapshot,
        equity_violations: &[EquityViolationType],
        risk_violations: &[ViolationType],
        ts_ns: i64,
    ) -> OrderIntentTransform {
        let transform_id = TransformId::derive(
            intent_id,
            &risk_snapshot.digest,
            &mtm_snapshot.digest,
            &drawdown_snapshot.digest,
            &self.policy.fingerprint,
        );

        // Rule 1: Block precedence
        // Check for halt violations
        let halt_equity_violations: Vec<_> =
            equity_violations.iter().filter(|v| v.is_halt()).cloned().collect();
        let halt_risk_violations: Vec<_> =
            risk_violations.iter().filter(|v| v.is_halt()).cloned().collect();

        if !halt_equity_violations.is_empty() {
            return self.build_transform(
                transform_id,
                ts_ns,
                intent_id,
                intent_type,
                intent_notional_mantissa,
                notional_exponent,
                TransformAction::Block {
                    reason: BlockReason::EquityHalt {
                        violations: halt_equity_violations,
                    },
                },
                intent_type,
                intent_notional_mantissa,
                false,
                risk_snapshot,
                mtm_snapshot,
                drawdown_snapshot,
            );
        }

        if !halt_risk_violations.is_empty() {
            return self.build_transform(
                transform_id,
                ts_ns,
                intent_id,
                intent_type,
                intent_notional_mantissa,
                notional_exponent,
                TransformAction::Block {
                    reason: BlockReason::RiskReject {
                        violations: halt_risk_violations,
                    },
                },
                intent_type,
                intent_notional_mantissa,
                false,
                risk_snapshot,
                mtm_snapshot,
                drawdown_snapshot,
            );
        }

        // Rule 2: Reduce-only mode
        if let Some(restrict_reason) =
            self.is_reduce_only_mode(drawdown_snapshot, mtm_snapshot)
        {
            if intent_type.is_increase() {
                return self.build_transform(
                    transform_id,
                    ts_ns,
                    intent_id,
                    intent_type,
                    intent_notional_mantissa,
                    notional_exponent,
                    TransformAction::Block {
                        reason: BlockReason::ReduceOnlyViolation,
                    },
                    intent_type,
                    intent_notional_mantissa,
                    false,
                    risk_snapshot,
                    mtm_snapshot,
                    drawdown_snapshot,
                );
            }
            // Reduce/Close pass through in reduce-only mode (Rule 4: exempt from caps)
            if intent_type.is_reduce_or_close() {
                return self.build_transform(
                    transform_id,
                    ts_ns,
                    intent_id,
                    intent_type,
                    intent_notional_mantissa,
                    notional_exponent,
                    TransformAction::ModeRestrict {
                        original_intent_type: intent_type,
                        allowed_intent_type: intent_type,
                        reason: restrict_reason,
                    },
                    intent_type,
                    intent_notional_mantissa,
                    true,
                    risk_snapshot,
                    mtm_snapshot,
                    drawdown_snapshot,
                );
            }
        }

        // Rule 4: Reduce/Close exempt from caps
        if intent_type.is_reduce_or_close() {
            return self.build_transform(
                transform_id,
                ts_ns,
                intent_id,
                intent_type,
                intent_notional_mantissa,
                notional_exponent,
                TransformAction::PassThrough,
                intent_type,
                intent_notional_mantissa,
                true,
                risk_snapshot,
                mtm_snapshot,
                drawdown_snapshot,
            );
        }

        // Rule 3: Notional caps (Increase only)
        let mut caps: Vec<(i128, CapReason)> = Vec::new();

        // Leverage headroom cap
        if self.policy.enable_leverage_headroom_cap
            && let Some((headroom, reason)) = self.compute_leverage_headroom(mtm_snapshot)
            && headroom < intent_notional_mantissa
        {
            caps.push((headroom, reason));
        }

        // Equity proportional cap
        if self.policy.enable_equity_proportional_cap
            && let Some((cap, reason)) = self.compute_equity_cap(mtm_snapshot)
            && cap < intent_notional_mantissa
        {
            caps.push((cap, reason));
        }

        // Drawdown recovery cap
        if self.policy.enable_drawdown_recovery_cap
            && let Some((cap, reason)) =
                self.compute_recovery_cap(intent_notional_mantissa, drawdown_snapshot)
            && cap < intent_notional_mantissa
        {
            caps.push((cap, reason));
        }

        // Smallest cap wins
        if let Some((smallest_cap, reason)) = caps.into_iter().min_by_key(|(cap, _)| *cap) {
            let capped_notional = smallest_cap.max(0);
            return self.build_transform(
                transform_id,
                ts_ns,
                intent_id,
                intent_type,
                intent_notional_mantissa,
                notional_exponent,
                TransformAction::NotionalCap {
                    original_notional_mantissa: intent_notional_mantissa,
                    capped_notional_mantissa: capped_notional,
                    notional_exponent,
                    cap_reason: reason,
                },
                intent_type,
                capped_notional,
                true,
                risk_snapshot,
                mtm_snapshot,
                drawdown_snapshot,
            );
        }

        // No transformation needed
        self.build_transform(
            transform_id,
            ts_ns,
            intent_id,
            intent_type,
            intent_notional_mantissa,
            notional_exponent,
            TransformAction::PassThrough,
            intent_type,
            intent_notional_mantissa,
            true,
            risk_snapshot,
            mtm_snapshot,
            drawdown_snapshot,
        )
    }

    /// Check if reduce-only mode is active.
    fn is_reduce_only_mode(
        &self,
        drawdown: &DrawdownSnapshot,
        mtm: &MtmSnapshot,
    ) -> Option<ModeRestrictReason> {
        // Check drawdown threshold (strict >)
        if drawdown.drawdown_pct_mantissa > self.policy.reduce_only_drawdown_pct_mantissa {
            return Some(ModeRestrictReason::DrawdownWarningActive {
                current_pct_mantissa: drawdown.drawdown_pct_mantissa,
                threshold_pct_mantissa: self.policy.reduce_only_drawdown_pct_mantissa,
            });
        }

        // Check leverage threshold (strict >)
        if mtm.metrics.leverage_mantissa > self.policy.reduce_only_leverage_mantissa {
            return Some(ModeRestrictReason::LeverageSoftLimitActive {
                current_mantissa: mtm.metrics.leverage_mantissa,
                soft_limit_mantissa: self.policy.reduce_only_leverage_mantissa,
            });
        }

        // Check stale count (>=)
        if mtm.metrics.stale_price_count >= self.policy.reduce_only_stale_count {
            return Some(ModeRestrictReason::StalePriceActive {
                stale_count: mtm.metrics.stale_price_count,
            });
        }

        None
    }

    /// Compute leverage headroom cap.
    /// Formula: max_notional = floor(equity * max_leverage / 10000)
    ///          headroom = max(0, max_notional - current_notional)
    fn compute_leverage_headroom(&self, mtm: &MtmSnapshot) -> Option<(i128, CapReason)> {
        let equity = mtm.metrics.equity_mantissa;
        if equity <= 0 {
            return None;
        }

        // max_notional = floor(equity * max_leverage / 10000)
        // leverage is exp -4, so divide by 10000
        let max_notional =
            (equity * self.policy.max_leverage_mantissa as i128) / 10000;
        let current_notional = mtm.metrics.total_notional_mantissa;
        let headroom = (max_notional - current_notional).max(0);

        Some((
            headroom,
            CapReason::LeverageHeadroom {
                max_leverage_mantissa: self.policy.max_leverage_mantissa,
                current_leverage_mantissa: mtm.metrics.leverage_mantissa,
                headroom_mantissa: headroom,
            },
        ))
    }

    /// Compute equity-proportional cap.
    /// Formula: cap = floor(equity * max_pct / 10000)
    fn compute_equity_cap(&self, mtm: &MtmSnapshot) -> Option<(i128, CapReason)> {
        let equity = mtm.metrics.equity_mantissa;
        if equity <= 0 {
            return None;
        }

        // cap = floor(equity * max_pct / 10000)
        let cap = (equity * self.policy.max_equity_pct_per_intent_mantissa as i128) / 10000;

        Some((
            cap,
            CapReason::EquityProportional {
                max_pct_mantissa: self.policy.max_equity_pct_per_intent_mantissa,
                equity_mantissa: equity,
                cap_mantissa: cap,
            },
        ))
    }

    /// Compute drawdown recovery cap.
    /// Linear interpolation: at DD=0 scale=100%, at DD=max_dd scale=min_scale
    /// Formula: scale = 10000 - ((10000 - min_scale) * dd_pct / max_dd_pct)
    ///          cap = floor(original * scale / 10000)
    fn compute_recovery_cap(
        &self,
        original_notional: i128,
        drawdown: &DrawdownSnapshot,
    ) -> Option<(i128, CapReason)> {
        let dd_pct = drawdown.drawdown_pct_mantissa;
        if dd_pct <= 0 {
            // No drawdown, no scaling needed
            return None;
        }

        let max_dd_pct = self.policy.recovery_max_drawdown_pct_mantissa;
        let min_scale = self.policy.recovery_min_scale_mantissa;

        // Clamp dd_pct to max_dd_pct
        let effective_dd = dd_pct.min(max_dd_pct);

        // scale = 10000 - ((10000 - min_scale) * effective_dd / max_dd_pct)
        let scale_reduction = if max_dd_pct > 0 {
            ((10000 - min_scale) * effective_dd) / max_dd_pct
        } else {
            0
        };
        let scale = (10000 - scale_reduction).max(min_scale);

        // cap = floor(original * scale / 10000)
        let cap = (original_notional * scale as i128) / 10000;

        Some((
            cap,
            CapReason::DrawdownRecoveryCap {
                drawdown_pct_mantissa: dd_pct,
                scale_factor_mantissa: scale,
            },
        ))
    }

    /// Build the transform artifact.
    #[allow(clippy::too_many_arguments)]
    fn build_transform(
        &self,
        transform_id: TransformId,
        ts_ns: i64,
        intent_id: &str,
        original_intent_type: IntentType,
        original_notional_mantissa: i128,
        notional_exponent: i8,
        action: TransformAction,
        final_intent_type: IntentType,
        final_notional_mantissa: i128,
        allowed: bool,
        risk_snapshot: &RiskSnapshot,
        mtm_snapshot: &MtmSnapshot,
        drawdown_snapshot: &DrawdownSnapshot,
    ) -> OrderIntentTransform {
        let mut transform = OrderIntentTransform {
            schema_version: INTENT_TRANSFORM_SCHEMA_VERSION.to_string(),
            transform_id,
            ts_ns,
            intent_id: intent_id.to_string(),
            original_intent_type,
            original_notional_mantissa,
            notional_exponent,
            action,
            final_intent_type,
            final_notional_mantissa,
            allowed,
            risk_snapshot_digest: risk_snapshot.digest.clone(),
            mtm_snapshot_digest: mtm_snapshot.digest.clone(),
            drawdown_snapshot_digest: drawdown_snapshot.digest.clone(),
            policy_fingerprint: self.policy.fingerprint.clone(),
            digest: String::new(),
        };
        transform.digest = transform.compute_digest();
        transform
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtm_drawdown::{
        DrawdownSnapshotId, MtmMetrics, MtmSnapshotId, DRAWDOWN_SNAPSHOT_SCHEMA_VERSION,
        MTM_SNAPSHOT_SCHEMA_VERSION,
    };
    use crate::risk_exposure::{
        ExposureMetrics, RiskSnapshotId, RISK_SNAPSHOT_SCHEMA_VERSION,
    };
    use std::collections::BTreeMap;

    fn create_healthy_mtm() -> MtmSnapshot {
        let mut mtm = MtmSnapshot {
            schema_version: MTM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: MtmSnapshotId("mtm_001".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "portfolio_digest".to_string(),
            price_source_digest: "price_digest".to_string(),
            position_valuations: BTreeMap::new(),
            metrics: MtmMetrics {
                total_unrealized_pnl_mantissa: 1000_00000000,
                total_realized_pnl_mantissa: 0,
                total_pnl_mantissa: 1000_00000000,
                pnl_exponent: -8,
                starting_capital_mantissa: 100000_00000000,
                equity_mantissa: 101000_00000000, // $101,000
                equity_exponent: -8,
                total_notional_mantissa: 50000_00000000, // $50,000 notional
                notional_exponent: -8,
                leverage_mantissa: 4950, // ~0.5x leverage
                leverage_exponent: -4,
                stale_price_count: 0,
                staleness_threshold_ns: 60_000_000_000,
            },
            digest: String::new(),
        };
        mtm.digest = mtm.compute_digest();
        mtm
    }

    fn create_healthy_drawdown() -> DrawdownSnapshot {
        let mut dd = DrawdownSnapshot {
            schema_version: DRAWDOWN_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: DrawdownSnapshotId("dd_001".to_string()),
            ts_ns: 1000,
            mtm_snapshot_digest: "mtm_digest".to_string(),
            peak_equity_mantissa: 101000_00000000,
            peak_ts_ns: 1000,
            current_equity_mantissa: 101000_00000000,
            equity_exponent: -8,
            drawdown_mantissa: 0,
            drawdown_pct_mantissa: 0,
            drawdown_pct_exponent: -4,
            max_drawdown_mantissa: 0,
            max_drawdown_pct_mantissa: 0,
            max_drawdown_ts_ns: 0,
            digest: String::new(),
        };
        dd.digest = dd.compute_digest();
        dd
    }

    fn create_healthy_risk() -> RiskSnapshot {
        let mut risk = RiskSnapshot {
            schema_version: RISK_SNAPSHOT_SCHEMA_VERSION.to_string(),
            snapshot_id: RiskSnapshotId("risk_001".to_string()),
            ts_ns: 1000,
            portfolio_snapshot_digest: "portfolio_digest".to_string(),
            policy_fingerprint: "policy_fp".to_string(),
            exposures: ExposureMetrics {
                notional_by_symbol: BTreeMap::new(),
                notional_by_strategy: BTreeMap::new(),
                notional_by_bucket: BTreeMap::new(),
                notional_by_venue: BTreeMap::new(),
                total_notional_mantissa: 50000_00000000,
                notional_exponent: -8,
                position_count_by_strategy: BTreeMap::new(),
                position_count_by_bucket: BTreeMap::new(),
                total_position_count: 1,
            },
            violations: Vec::new(),
            is_compliant: true,
            digest: String::new(),
        };
        risk.digest = risk.compute_digest();
        risk
    }

    #[test]
    fn test_passthrough_healthy() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let mtm = create_healthy_mtm();
        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000, // $1,000 intent
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(transform.action, TransformAction::PassThrough));
        assert_eq!(transform.final_notional_mantissa, 1000_00000000);
    }

    #[test]
    fn test_block_on_equity_halt() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let mtm = create_healthy_mtm();
        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        let violations = vec![EquityViolationType::DrawdownBreach {
            current_pct_mantissa: 1500,
            max_pct_mantissa: 1000,
        }];

        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &violations,
            &[],
            1000,
        );

        assert!(!transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::Block {
                reason: BlockReason::EquityHalt { .. }
            }
        ));
    }

    #[test]
    fn test_block_on_risk_halt() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let mtm = create_healthy_mtm();
        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        let violations = vec![ViolationType::PortfolioNotionalExceeded {
            current_mantissa: 200000_00000000,
            limit_mantissa: 100000_00000000,
        }];

        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &violations,
            1000,
        );

        assert!(!transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::Block {
                reason: BlockReason::RiskReject { .. }
            }
        ));
    }

    #[test]
    fn test_reduce_only_on_drawdown() {
        let policy = ShapingPolicy::moderate(); // 7% DD threshold
        let shaper = IntentShaper::new(policy);

        let mtm = create_healthy_mtm();
        let risk = create_healthy_risk();

        // 8% drawdown > 7% threshold
        let mut dd = create_healthy_drawdown();
        dd.drawdown_pct_mantissa = 800;
        dd.digest = dd.compute_digest();

        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(!transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::Block {
                reason: BlockReason::ReduceOnlyViolation
            }
        ));
    }

    #[test]
    fn test_reduce_only_on_leverage() {
        let policy = ShapingPolicy::moderate(); // 2.5x leverage threshold
        let shaper = IntentShaper::new(policy);

        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        // 3x leverage > 2.5x threshold
        let mut mtm = create_healthy_mtm();
        mtm.metrics.leverage_mantissa = 30000;
        mtm.digest = mtm.compute_digest();

        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(!transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::Block {
                reason: BlockReason::ReduceOnlyViolation
            }
        ));
    }

    #[test]
    fn test_reduce_only_on_stale() {
        let policy = ShapingPolicy::moderate(); // 3 stale threshold
        let shaper = IntentShaper::new(policy);

        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        // 3 stale >= 3 threshold
        let mut mtm = create_healthy_mtm();
        mtm.metrics.stale_price_count = 3;
        mtm.digest = mtm.compute_digest();

        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(!transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::Block {
                reason: BlockReason::ReduceOnlyViolation
            }
        ));
    }

    #[test]
    fn test_reduce_intent_passes_in_reduce_only() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let risk = create_healthy_risk();

        // High drawdown triggers reduce-only
        let mut dd = create_healthy_drawdown();
        dd.drawdown_pct_mantissa = 800;
        dd.digest = dd.compute_digest();

        let mtm = create_healthy_mtm();

        let transform = shaper.shape(
            "intent_001",
            IntentType::Reduce,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::ModeRestrict { .. }
        ));
    }

    #[test]
    fn test_close_intent_passes_in_reduce_only() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let risk = create_healthy_risk();

        // High leverage triggers reduce-only
        let mut mtm = create_healthy_mtm();
        mtm.metrics.leverage_mantissa = 30000;
        mtm.digest = mtm.compute_digest();

        let dd = create_healthy_drawdown();

        let transform = shaper.shape(
            "intent_001",
            IntentType::Close,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::ModeRestrict { .. }
        ));
    }

    #[test]
    fn test_leverage_headroom_cap() {
        let policy = ShapingPolicy::moderate(); // 3.0x max leverage
        let shaper = IntentShaper::new(policy);

        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        // equity = $101,000, max_leverage = 3.0x
        // max_notional = 101000 * 30000 / 10000 = 303,000
        // current_notional = $50,000
        // headroom = 303,000 - 50,000 = 253,000
        let mtm = create_healthy_mtm();

        // Intent for $300,000 (exceeds headroom)
        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            300000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::NotionalCap { .. }
        ));
        // Capped to headroom ~253,000
        assert!(transform.final_notional_mantissa < 300000_00000000);
    }

    #[test]
    fn test_equity_proportional_cap() {
        let policy = ShapingPolicy::moderate(); // 10% of equity per intent
        let shaper = IntentShaper::new(policy);

        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        // equity = $101,000, max 10% = $10,100
        let mtm = create_healthy_mtm();

        // Intent for $20,000 (exceeds 10% cap)
        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            20000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::NotionalCap { .. }
        ));
        // Capped to ~$10,100
        assert!(transform.final_notional_mantissa <= 10100_00000000);
    }

    #[test]
    fn test_drawdown_recovery_scale() {
        let policy = ShapingPolicy::moderate(); // min 50% at 10% DD
        let shaper = IntentShaper::new(policy);

        let risk = create_healthy_risk();
        let mtm = create_healthy_mtm();

        // 5% drawdown (halfway to max 10%)
        // scale = 10000 - ((10000 - 5000) * 500 / 1000) = 10000 - 2500 = 7500 (75%)
        let mut dd = create_healthy_drawdown();
        dd.drawdown_pct_mantissa = 500;
        dd.digest = dd.compute_digest();

        // Intent for $10,000
        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            10000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        // Should be capped to ~$7,500 (75% of $10,000)
        if let TransformAction::NotionalCap {
            capped_notional_mantissa,
            ..
        } = &transform.action
        {
            assert!(*capped_notional_mantissa <= 7500_00000000);
        }
    }

    #[test]
    fn test_smallest_cap_wins() {
        // Use conservative policy for tighter caps
        let policy = ShapingPolicy::conservative();
        let shaper = IntentShaper::new(policy);

        let risk = create_healthy_risk();

        // Lower equity for tighter proportional cap
        let mut mtm = create_healthy_mtm();
        mtm.metrics.equity_mantissa = 10000_00000000; // $10,000 equity
        mtm.metrics.total_notional_mantissa = 5000_00000000; // $5,000 current
        mtm.digest = mtm.compute_digest();

        // Some drawdown for recovery cap
        let mut dd = create_healthy_drawdown();
        dd.drawdown_pct_mantissa = 200; // 2% DD
        dd.digest = dd.compute_digest();

        // Conservative: 5% of equity = $500, leverage headroom = 10000*2 - 5000 = 15000
        // Intent for $10,000
        let transform = shaper.shape(
            "intent_001",
            IntentType::Increase,
            10000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(
            transform.action,
            TransformAction::NotionalCap { .. }
        ));
        // Should be capped to smallest (~$500)
        assert!(transform.final_notional_mantissa <= 500_00000000);
    }

    #[test]
    fn test_caps_not_applied_to_reduce() {
        let policy = ShapingPolicy::conservative();
        let shaper = IntentShaper::new(policy);

        let risk = create_healthy_risk();
        let dd = create_healthy_drawdown();

        // Very low equity to ensure caps would trigger
        let mut mtm = create_healthy_mtm();
        mtm.metrics.equity_mantissa = 1000_00000000; // $1,000 equity
        mtm.digest = mtm.compute_digest();

        // Large reduce intent
        let transform = shaper.shape(
            "intent_001",
            IntentType::Reduce,
            50000_00000000, // $50,000 reduce
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert!(transform.allowed);
        assert!(matches!(transform.action, TransformAction::PassThrough));
        assert_eq!(transform.final_notional_mantissa, 50000_00000000);
    }

    #[test]
    fn test_transform_digest_deterministic() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let mtm = create_healthy_mtm();
        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        let transform1 = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        let transform2 = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert_eq!(transform1.digest, transform2.digest);
    }

    #[test]
    fn test_transform_id_deterministic() {
        let policy = ShapingPolicy::moderate();
        let shaper = IntentShaper::new(policy);

        let mtm = create_healthy_mtm();
        let dd = create_healthy_drawdown();
        let risk = create_healthy_risk();

        let transform1 = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        let transform2 = shaper.shape(
            "intent_001",
            IntentType::Increase,
            1000_00000000,
            -8,
            &risk,
            &mtm,
            &dd,
            &[],
            &[],
            1000,
        );

        assert_eq!(transform1.transform_id.0, transform2.transform_id.0);
    }

    #[test]
    fn test_policy_fingerprint_deterministic() {
        let policy1 = ShapingPolicy::moderate();
        let policy2 = ShapingPolicy::moderate();
        let policy3 = ShapingPolicy::conservative();

        assert_eq!(policy1.fingerprint, policy2.fingerprint);
        assert_ne!(policy1.fingerprint, policy3.fingerprint);
    }
}
