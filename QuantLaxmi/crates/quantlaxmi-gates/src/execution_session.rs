//! # Execution Session & Control Plane (Phase 16)
//!
//! Session-level state machine, kill-switch hierarchy, and manual override system.
//!
//! ## Core Question
//! "When the system detects danger or an operator intervenes, how does execution
//! state transition deterministically, and how do we recover?"
//!
//! ## Hard Laws
//! - L1: Deterministic State Transitions — Same inputs → identical session state
//! - L2: Hierarchy Propagation — Global halt implies all children halted
//! - L3: WAL-Bound — Every state transition logged before effect
//! - L4: No Silent Failures — All halts have explicit reason codes
//! - L5: Manual Override Audit — Human interventions require operator ID + reason
//! - L6: Reduce-Only Before Recovery — Halted sessions must pass through ReduceOnly
//!
//! ## Precedence Rules (Frozen)
//! - SessionState.Halted always overrides KillSwitchRegistry scope checks
//! - KillSwitchRegistry is evaluated only if session state allows execution
//! - EmergencyFlatten forces Halted → flatten → ReduceOnly (partial) or Terminated (complete)

use crate::intent_shaping::IntentType;
use crate::mtm_drawdown::EquityViolationType;
use crate::risk_exposure::ViolationType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

// =============================================================================
// Schema Versions
// =============================================================================

pub const EXECUTION_SESSION_SCHEMA_VERSION: &str = "execution_session_v1.0";
pub const KILL_SWITCH_SCHEMA_VERSION: &str = "kill_switch_v1.0";
pub const MANUAL_OVERRIDE_SCHEMA_VERSION: &str = "manual_override_v1.0";
pub const EMERGENCY_FLATTEN_SCHEMA_VERSION: &str = "emergency_flatten_v1.0";

// =============================================================================
// Session State
// =============================================================================

/// Session-level state machine.
/// State transitions are deterministic and WAL-logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
pub enum SessionState {
    /// Normal operation — all intent types allowed.
    Active = 0,

    /// Hard stop — no execution of any kind.
    /// Triggered by: HALT violations, manual kill, circuit breaker trip.
    Halted = 1,

    /// Recovery mode — only Reduce/Close intents allowed.
    /// Entered from Halted after operator acknowledgment.
    ReduceOnly = 2,

    /// Graceful shutdown — completing open orders, no new orders.
    Draining = 3,

    /// Session complete — terminal state.
    Terminated = 4,
}

impl SessionState {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            SessionState::Active => 0,
            SessionState::Halted => 1,
            SessionState::ReduceOnly => 2,
            SessionState::Draining => 3,
            SessionState::Terminated => 4,
        }
    }

    /// Check if this state allows any execution.
    pub fn allows_execution(&self) -> bool {
        matches!(self, SessionState::Active | SessionState::ReduceOnly)
    }

    /// Check if this state is terminal.
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionState::Terminated)
    }
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Active => write!(f, "ACTIVE"),
            SessionState::Halted => write!(f, "HALTED"),
            SessionState::ReduceOnly => write!(f, "REDUCE_ONLY"),
            SessionState::Draining => write!(f, "DRAINING"),
            SessionState::Terminated => write!(f, "TERMINATED"),
        }
    }
}

// =============================================================================
// Circuit Breaker Type
// =============================================================================

/// Circuit breaker type (integration with circuit_breakers.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
pub enum CircuitBreakerType {
    GlobalKillSwitch = 0,
    LatencyBreaker = 1,
    OrderFlowBreaker = 2,
    DrawdownBreaker = 3,
}

impl CircuitBreakerType {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            CircuitBreakerType::GlobalKillSwitch => 0,
            CircuitBreakerType::LatencyBreaker => 1,
            CircuitBreakerType::OrderFlowBreaker => 2,
            CircuitBreakerType::DrawdownBreaker => 3,
        }
    }

    /// Get code string.
    pub fn code(&self) -> &'static str {
        match self {
            CircuitBreakerType::GlobalKillSwitch => "GLOBAL_KILL",
            CircuitBreakerType::LatencyBreaker => "LATENCY",
            CircuitBreakerType::OrderFlowBreaker => "ORDER_FLOW",
            CircuitBreakerType::DrawdownBreaker => "DRAWDOWN",
        }
    }
}

// =============================================================================
// Session Transition Reason
// =============================================================================

/// Reason for session state transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SessionTransitionReason {
    // === Automatic Triggers (discriminant 0-3) ===
    /// Risk violation triggered HALT (from Phase 15.1).
    RiskViolation { violations: Vec<ViolationType> },

    /// Equity violation triggered HALT (from Phase 15.2).
    EquityViolation {
        violations: Vec<EquityViolationType>,
    },

    /// Circuit breaker tripped (from circuit_breakers.rs).
    CircuitBreakerTrip { breaker_type: CircuitBreakerType },

    /// Connectivity loss to exchange.
    ConnectivityLoss { exchange: String, duration_ms: u64 },

    // === Manual Triggers (discriminant 4-9) ===
    /// Operator initiated halt.
    ManualHalt { operator_id: String, reason: String },

    /// Operator initiated reduce-only mode.
    ManualReduceOnly { operator_id: String, reason: String },

    /// Operator cleared halt and entered reduce-only.
    ManualRecoveryStart { operator_id: String },

    /// Operator restored full operation.
    ManualRecoveryComplete { operator_id: String },

    /// Operator initiated graceful shutdown.
    ManualDrain { operator_id: String, reason: String },

    /// Session ended normally.
    SessionComplete,
}

impl SessionTransitionReason {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            SessionTransitionReason::RiskViolation { .. } => 0,
            SessionTransitionReason::EquityViolation { .. } => 1,
            SessionTransitionReason::CircuitBreakerTrip { .. } => 2,
            SessionTransitionReason::ConnectivityLoss { .. } => 3,
            SessionTransitionReason::ManualHalt { .. } => 4,
            SessionTransitionReason::ManualReduceOnly { .. } => 5,
            SessionTransitionReason::ManualRecoveryStart { .. } => 6,
            SessionTransitionReason::ManualRecoveryComplete { .. } => 7,
            SessionTransitionReason::ManualDrain { .. } => 8,
            SessionTransitionReason::SessionComplete => 9,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            SessionTransitionReason::RiskViolation { violations } => {
                let mut sorted = violations.clone();
                sorted.sort();
                bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
                for v in &sorted {
                    let vb = v.canonical_bytes();
                    bytes.extend_from_slice(&(vb.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(&vb);
                }
            }
            SessionTransitionReason::EquityViolation { violations } => {
                let mut sorted = violations.clone();
                sorted.sort();
                bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
                for v in &sorted {
                    let vb = v.canonical_bytes();
                    bytes.extend_from_slice(&(vb.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(&vb);
                }
            }
            SessionTransitionReason::CircuitBreakerTrip { breaker_type } => {
                bytes.push(breaker_type.discriminant());
            }
            SessionTransitionReason::ConnectivityLoss {
                exchange,
                duration_ms,
            } => {
                bytes.extend_from_slice(&(exchange.len() as u32).to_le_bytes());
                bytes.extend_from_slice(exchange.as_bytes());
                bytes.extend_from_slice(&duration_ms.to_le_bytes());
            }
            SessionTransitionReason::ManualHalt {
                operator_id,
                reason,
            }
            | SessionTransitionReason::ManualReduceOnly {
                operator_id,
                reason,
            }
            | SessionTransitionReason::ManualDrain {
                operator_id,
                reason,
            } => {
                bytes.extend_from_slice(&(operator_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(operator_id.as_bytes());
                bytes.extend_from_slice(&(reason.len() as u32).to_le_bytes());
                bytes.extend_from_slice(reason.as_bytes());
            }
            SessionTransitionReason::ManualRecoveryStart { operator_id }
            | SessionTransitionReason::ManualRecoveryComplete { operator_id } => {
                bytes.extend_from_slice(&(operator_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(operator_id.as_bytes());
            }
            SessionTransitionReason::SessionComplete => {
                // No additional data
            }
        }

        bytes
    }
}

// =============================================================================
// Transition ID
// =============================================================================

/// Unique transition ID (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransitionId(pub String);

impl TransitionId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from session_id + from_state + to_state + ts_ns.
    pub fn derive(session_id: &str, from: SessionState, to: SessionState, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"session_transition:");
        hasher.update((session_id.len() as u32).to_le_bytes());
        hasher.update(session_id.as_bytes());
        hasher.update(b":");
        hasher.update([from.discriminant()]);
        hasher.update(b":");
        hasher.update([to.discriminant()]);
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for TransitionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Session Transition Event
// =============================================================================

/// Session state transition event (WAL artifact).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTransitionEvent {
    pub schema_version: String,
    pub transition_id: TransitionId,
    pub ts_ns: i64,
    pub session_id: String,
    pub from_state: SessionState,
    pub to_state: SessionState,
    pub reason: SessionTransitionReason,

    /// Snapshot digests at transition time (for replay verification).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub risk_snapshot_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtm_snapshot_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drawdown_snapshot_digest: Option<String>,

    /// Deterministic digest.
    pub digest: String,
}

impl SessionTransitionEvent {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.transition_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        hasher.update((self.session_id.len() as u32).to_le_bytes());
        hasher.update(self.session_id.as_bytes());

        hasher.update([self.from_state.discriminant()]);
        hasher.update([self.to_state.discriminant()]);

        let reason_bytes = self.reason.canonical_bytes();
        hasher.update((reason_bytes.len() as u32).to_le_bytes());
        hasher.update(&reason_bytes);

        // Optional digests
        if let Some(ref d) = self.risk_snapshot_digest {
            hasher.update([1u8]);
            hasher.update(d.as_bytes());
        } else {
            hasher.update([0u8]);
        }

        if let Some(ref d) = self.mtm_snapshot_digest {
            hasher.update([1u8]);
            hasher.update(d.as_bytes());
        } else {
            hasher.update([0u8]);
        }

        if let Some(ref d) = self.drawdown_snapshot_digest {
            hasher.update([1u8]);
            hasher.update(d.as_bytes());
        } else {
            hasher.update([0u8]);
        }

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Kill-Switch Scope
// =============================================================================

/// Kill-switch scope hierarchy.
/// Global encompasses all; Symbol is most granular.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord, Hash)]
pub enum KillSwitchScope {
    /// Entire session halted.
    Global,

    /// Specific bucket halted (all strategies within).
    Bucket { bucket_id: String },

    /// Specific strategy halted (all symbols within).
    Strategy { strategy_id: String },

    /// Specific symbol halted.
    Symbol { symbol: String },
}

impl KillSwitchScope {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            KillSwitchScope::Global => 0,
            KillSwitchScope::Bucket { .. } => 1,
            KillSwitchScope::Strategy { .. } => 2,
            KillSwitchScope::Symbol { .. } => 3,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            KillSwitchScope::Global => {}
            KillSwitchScope::Bucket { bucket_id } => {
                bytes.extend_from_slice(&(bucket_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bucket_id.as_bytes());
            }
            KillSwitchScope::Strategy { strategy_id } => {
                bytes.extend_from_slice(&(strategy_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(strategy_id.as_bytes());
            }
            KillSwitchScope::Symbol { symbol } => {
                bytes.extend_from_slice(&(symbol.len() as u32).to_le_bytes());
                bytes.extend_from_slice(symbol.as_bytes());
            }
        }

        bytes
    }
}

impl std::fmt::Display for KillSwitchScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KillSwitchScope::Global => write!(f, "GLOBAL"),
            KillSwitchScope::Bucket { bucket_id } => write!(f, "BUCKET:{}", bucket_id),
            KillSwitchScope::Strategy { strategy_id } => write!(f, "STRATEGY:{}", strategy_id),
            KillSwitchScope::Symbol { symbol } => write!(f, "SYMBOL:{}", symbol),
        }
    }
}

// =============================================================================
// Kill-Switch Reason
// =============================================================================

/// Kill-switch activation reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum KillSwitchReason {
    /// Automatic from risk violation.
    RiskTriggered { violation_codes: Vec<String> },

    /// Automatic from equity violation.
    EquityTriggered { violation_codes: Vec<String> },

    /// Automatic from circuit breaker.
    CircuitBreakerTriggered { breaker: CircuitBreakerType },

    /// Manual operator intervention.
    ManualActivation { operator_id: String, reason: String },

    /// Manual deactivation.
    ManualDeactivation { operator_id: String },
}

impl KillSwitchReason {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            KillSwitchReason::RiskTriggered { .. } => 0,
            KillSwitchReason::EquityTriggered { .. } => 1,
            KillSwitchReason::CircuitBreakerTriggered { .. } => 2,
            KillSwitchReason::ManualActivation { .. } => 3,
            KillSwitchReason::ManualDeactivation { .. } => 4,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            KillSwitchReason::RiskTriggered { violation_codes }
            | KillSwitchReason::EquityTriggered { violation_codes } => {
                let mut sorted = violation_codes.clone();
                sorted.sort();
                bytes.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
                for code in &sorted {
                    bytes.extend_from_slice(&(code.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(code.as_bytes());
                }
            }
            KillSwitchReason::CircuitBreakerTriggered { breaker } => {
                bytes.push(breaker.discriminant());
            }
            KillSwitchReason::ManualActivation {
                operator_id,
                reason,
            } => {
                bytes.extend_from_slice(&(operator_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(operator_id.as_bytes());
                bytes.extend_from_slice(&(reason.len() as u32).to_le_bytes());
                bytes.extend_from_slice(reason.as_bytes());
            }
            KillSwitchReason::ManualDeactivation { operator_id } => {
                bytes.extend_from_slice(&(operator_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(operator_id.as_bytes());
            }
        }

        bytes
    }
}

// =============================================================================
// Kill-Switch Event ID
// =============================================================================

/// Unique kill-switch event ID (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KillSwitchEventId(pub String);

impl KillSwitchEventId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from scope + activated + ts_ns.
    pub fn derive(scope: &KillSwitchScope, activated: bool, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"kill_switch:");
        let scope_bytes = scope.canonical_bytes();
        hasher.update(&scope_bytes);
        hasher.update(b":");
        hasher.update([activated as u8]);
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for KillSwitchEventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Kill-Switch Event
// =============================================================================

/// Kill-switch event (WAL artifact).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchEvent {
    pub schema_version: String,
    pub event_id: KillSwitchEventId,
    pub ts_ns: i64,
    pub scope: KillSwitchScope,
    pub activated: bool, // true = activated, false = deactivated
    pub reason: KillSwitchReason,
    pub digest: String,
}

impl KillSwitchEvent {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.event_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        let scope_bytes = self.scope.canonical_bytes();
        hasher.update((scope_bytes.len() as u32).to_le_bytes());
        hasher.update(&scope_bytes);

        hasher.update([self.activated as u8]);

        let reason_bytes = self.reason.canonical_bytes();
        hasher.update((reason_bytes.len() as u32).to_le_bytes());
        hasher.update(&reason_bytes);

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Kill-Switch Registry
// =============================================================================

/// Kill-switch state tracker.
#[derive(Debug, Clone, Default)]
pub struct KillSwitchRegistry {
    /// Global switch (if true, everything halted).
    global_halted: bool,

    /// Halted buckets.
    halted_buckets: BTreeSet<String>,

    /// Halted strategies.
    halted_strategies: BTreeSet<String>,

    /// Halted symbols.
    halted_symbols: BTreeSet<String>,

    /// Audit log of all switch activations.
    activation_log: Vec<KillSwitchEvent>,
}

impl KillSwitchRegistry {
    /// Create new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if execution is allowed for given context.
    /// Hierarchy: Global > Bucket > Strategy > Symbol
    pub fn is_halted(&self, bucket_id: &str, strategy_id: &str, symbol: &str) -> bool {
        if self.global_halted {
            return true;
        }
        if self.halted_buckets.contains(bucket_id) {
            return true;
        }
        if self.halted_strategies.contains(strategy_id) {
            return true;
        }
        if self.halted_symbols.contains(symbol) {
            return true;
        }
        false
    }

    /// Check if global halt is active.
    pub fn is_global_halted(&self) -> bool {
        self.global_halted
    }

    /// Activate kill-switch at scope.
    pub fn activate(
        &mut self,
        scope: KillSwitchScope,
        reason: KillSwitchReason,
        ts_ns: i64,
    ) -> KillSwitchEvent {
        match &scope {
            KillSwitchScope::Global => {
                self.global_halted = true;
            }
            KillSwitchScope::Bucket { bucket_id } => {
                self.halted_buckets.insert(bucket_id.clone());
            }
            KillSwitchScope::Strategy { strategy_id } => {
                self.halted_strategies.insert(strategy_id.clone());
            }
            KillSwitchScope::Symbol { symbol } => {
                self.halted_symbols.insert(symbol.clone());
            }
        }

        let event_id = KillSwitchEventId::derive(&scope, true, ts_ns);
        let mut event = KillSwitchEvent {
            schema_version: KILL_SWITCH_SCHEMA_VERSION.to_string(),
            event_id,
            ts_ns,
            scope,
            activated: true,
            reason,
            digest: String::new(),
        };
        event.digest = event.compute_digest();
        self.activation_log.push(event.clone());
        event
    }

    /// Deactivate kill-switch at scope (manual only).
    pub fn deactivate(
        &mut self,
        scope: KillSwitchScope,
        operator_id: &str,
        ts_ns: i64,
    ) -> KillSwitchEvent {
        match &scope {
            KillSwitchScope::Global => {
                self.global_halted = false;
            }
            KillSwitchScope::Bucket { bucket_id } => {
                self.halted_buckets.remove(bucket_id);
            }
            KillSwitchScope::Strategy { strategy_id } => {
                self.halted_strategies.remove(strategy_id);
            }
            KillSwitchScope::Symbol { symbol } => {
                self.halted_symbols.remove(symbol);
            }
        }

        let event_id = KillSwitchEventId::derive(&scope, false, ts_ns);
        let mut event = KillSwitchEvent {
            schema_version: KILL_SWITCH_SCHEMA_VERSION.to_string(),
            event_id,
            ts_ns,
            scope,
            activated: false,
            reason: KillSwitchReason::ManualDeactivation {
                operator_id: operator_id.to_string(),
            },
            digest: String::new(),
        };
        event.digest = event.compute_digest();
        self.activation_log.push(event.clone());
        event
    }

    /// Get all active kill-switches.
    pub fn active_switches(&self) -> Vec<KillSwitchScope> {
        let mut switches = Vec::new();
        if self.global_halted {
            switches.push(KillSwitchScope::Global);
        }
        for bucket_id in &self.halted_buckets {
            switches.push(KillSwitchScope::Bucket {
                bucket_id: bucket_id.clone(),
            });
        }
        for strategy_id in &self.halted_strategies {
            switches.push(KillSwitchScope::Strategy {
                strategy_id: strategy_id.clone(),
            });
        }
        for symbol in &self.halted_symbols {
            switches.push(KillSwitchScope::Symbol {
                symbol: symbol.clone(),
            });
        }
        switches
    }

    /// Get activation log.
    pub fn activation_log(&self) -> &[KillSwitchEvent] {
        &self.activation_log
    }

    /// Clear all kill-switches (for testing only).
    #[cfg(test)]
    pub fn clear_all(&mut self) {
        self.global_halted = false;
        self.halted_buckets.clear();
        self.halted_strategies.clear();
        self.halted_symbols.clear();
    }
}

// =============================================================================
// Override Type
// =============================================================================

/// Manual override types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum OverrideType {
    /// Force-halt execution.
    ForceHalt,

    /// Force reduce-only mode.
    ForceReduceOnly,

    /// Clear halt, enter reduce-only recovery.
    ClearHalt,

    /// Restore full operation from reduce-only.
    RestoreFull,

    /// Emergency flatten all positions.
    EmergencyFlatten,

    /// Cancel all open orders.
    CancelAllOrders,

    /// Adjust position limit (temporary).
    AdjustLimit {
        scope: KillSwitchScope,
        new_limit_mantissa: i128,
    },
}

impl OverrideType {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            OverrideType::ForceHalt => 0,
            OverrideType::ForceReduceOnly => 1,
            OverrideType::ClearHalt => 2,
            OverrideType::RestoreFull => 3,
            OverrideType::EmergencyFlatten => 4,
            OverrideType::CancelAllOrders => 5,
            OverrideType::AdjustLimit { .. } => 6,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        if let OverrideType::AdjustLimit {
            scope,
            new_limit_mantissa,
        } = self
        {
            let scope_bytes = scope.canonical_bytes();
            bytes.extend_from_slice(&(scope_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&scope_bytes);
            bytes.extend_from_slice(&new_limit_mantissa.to_le_bytes());
        }

        bytes
    }
}

// =============================================================================
// Override ID
// =============================================================================

/// Unique override ID (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OverrideId(pub String);

impl OverrideId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from session_id + override_type + ts_ns.
    pub fn derive(session_id: &str, override_type: &OverrideType, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"manual_override:");
        hasher.update((session_id.len() as u32).to_le_bytes());
        hasher.update(session_id.as_bytes());
        hasher.update(b":");
        let type_bytes = override_type.canonical_bytes();
        hasher.update(&type_bytes);
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for OverrideId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Manual Override Event
// =============================================================================

/// Manual override event (WAL artifact).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualOverrideEvent {
    pub schema_version: String,
    pub override_id: OverrideId,
    pub ts_ns: i64,
    pub session_id: String,

    /// Operator identification (required).
    pub operator_id: String,

    /// Override type.
    pub override_type: OverrideType,

    /// Human-readable reason (required, min 10 chars).
    pub reason: String,

    /// Scope of override (for scoped overrides).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<KillSwitchScope>,

    /// Prior session state (for audit).
    pub prior_session_state: SessionState,

    /// Resulting session state.
    pub resulting_session_state: SessionState,

    /// Snapshot digests at override time.
    pub portfolio_snapshot_digest: String,
    pub risk_snapshot_digest: String,

    /// Deterministic digest.
    pub digest: String,
}

impl ManualOverrideEvent {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.override_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        hasher.update((self.session_id.len() as u32).to_le_bytes());
        hasher.update(self.session_id.as_bytes());

        hasher.update((self.operator_id.len() as u32).to_le_bytes());
        hasher.update(self.operator_id.as_bytes());

        let type_bytes = self.override_type.canonical_bytes();
        hasher.update((type_bytes.len() as u32).to_le_bytes());
        hasher.update(&type_bytes);

        hasher.update((self.reason.len() as u32).to_le_bytes());
        hasher.update(self.reason.as_bytes());

        if let Some(ref s) = self.scope {
            hasher.update([1u8]);
            let scope_bytes = s.canonical_bytes();
            hasher.update(&scope_bytes);
        } else {
            hasher.update([0u8]);
        }

        hasher.update([self.prior_session_state.discriminant()]);
        hasher.update([self.resulting_session_state.discriminant()]);

        hasher.update(self.portfolio_snapshot_digest.as_bytes());
        hasher.update(self.risk_snapshot_digest.as_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Override Policy
// =============================================================================

/// Override validation policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverridePolicy {
    /// Minimum reason length (chars).
    pub min_reason_length: usize,

    /// Require 2FA confirmation for these override types.
    pub require_2fa: Vec<OverrideType>,

    /// Allowed operator IDs (empty = all allowed).
    pub allowed_operators: Vec<String>,

    /// Cool-down between repeated overrides (ns).
    pub cooldown_ns: i64,
}

impl Default for OverridePolicy {
    fn default() -> Self {
        Self {
            min_reason_length: 10,
            require_2fa: vec![OverrideType::EmergencyFlatten],
            allowed_operators: vec![],   // Empty = all allowed
            cooldown_ns: 60_000_000_000, // 60 seconds
        }
    }
}

impl OverridePolicy {
    /// Create strict policy (for production).
    pub fn strict() -> Self {
        Self {
            min_reason_length: 20,
            require_2fa: vec![
                OverrideType::EmergencyFlatten,
                OverrideType::ForceHalt,
                OverrideType::RestoreFull,
            ],
            allowed_operators: vec![],
            cooldown_ns: 120_000_000_000, // 120 seconds
        }
    }

    /// Create permissive policy (for testing).
    pub fn permissive() -> Self {
        Self {
            min_reason_length: 1,
            require_2fa: vec![],
            allowed_operators: vec![],
            cooldown_ns: 0,
        }
    }
}

// =============================================================================
// Override Error
// =============================================================================

/// Override validation error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OverrideError {
    #[error("Reason too short: {actual} chars (minimum {required})")]
    ReasonTooShort { actual: usize, required: usize },

    #[error("Operator not authorized: {operator_id}")]
    OperatorNotAuthorized { operator_id: String },

    #[error("Cooldown active: {remaining_ns} ns remaining")]
    CooldownActive { remaining_ns: i64 },

    #[error("Invalid state transition: {from} -> {to}")]
    InvalidTransition {
        from: SessionState,
        to: SessionState,
    },

    #[error("2FA required for override type")]
    Requires2FA,

    #[error("Session is terminal")]
    SessionTerminal,
}

// =============================================================================
// Flatten Scope
// =============================================================================

/// Flatten scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum FlattenScope {
    /// Flatten entire portfolio.
    All,

    /// Flatten specific bucket.
    Bucket { bucket_id: String },

    /// Flatten specific strategy.
    Strategy { strategy_id: String },

    /// Flatten specific symbol.
    Symbol { symbol: String },
}

impl FlattenScope {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            FlattenScope::All => 0,
            FlattenScope::Bucket { .. } => 1,
            FlattenScope::Strategy { .. } => 2,
            FlattenScope::Symbol { .. } => 3,
        }
    }

    /// Canonical bytes for deterministic hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.discriminant());

        match self {
            FlattenScope::All => {}
            FlattenScope::Bucket { bucket_id } => {
                bytes.extend_from_slice(&(bucket_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(bucket_id.as_bytes());
            }
            FlattenScope::Strategy { strategy_id } => {
                bytes.extend_from_slice(&(strategy_id.len() as u32).to_le_bytes());
                bytes.extend_from_slice(strategy_id.as_bytes());
            }
            FlattenScope::Symbol { symbol } => {
                bytes.extend_from_slice(&(symbol.len() as u32).to_le_bytes());
                bytes.extend_from_slice(symbol.as_bytes());
            }
        }

        bytes
    }
}

// =============================================================================
// Flatten Request ID
// =============================================================================

/// Unique flatten request ID (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlattenRequestId(pub String);

impl FlattenRequestId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from operator_id + scope + ts_ns.
    pub fn derive(operator_id: &str, scope: &FlattenScope, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"flatten_request:");
        hasher.update((operator_id.len() as u32).to_le_bytes());
        hasher.update(operator_id.as_bytes());
        hasher.update(b":");
        let scope_bytes = scope.canonical_bytes();
        hasher.update(&scope_bytes);
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for FlattenRequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Emergency Flatten Request
// =============================================================================

/// Emergency flatten request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyFlattenRequest {
    pub schema_version: String,
    pub request_id: FlattenRequestId,
    pub ts_ns: i64,
    pub operator_id: String,
    pub reason: String,

    /// Scope of flatten.
    pub scope: FlattenScope,

    /// Price tolerance for market orders (bps).
    pub price_tolerance_bps: u32,

    /// Max time to complete (ns). After this, report failure.
    pub timeout_ns: i64,

    /// Deterministic digest.
    pub digest: String,
}

impl EmergencyFlattenRequest {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.request_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        hasher.update((self.operator_id.len() as u32).to_le_bytes());
        hasher.update(self.operator_id.as_bytes());

        hasher.update((self.reason.len() as u32).to_le_bytes());
        hasher.update(self.reason.as_bytes());

        let scope_bytes = self.scope.canonical_bytes();
        hasher.update((scope_bytes.len() as u32).to_le_bytes());
        hasher.update(&scope_bytes);

        hasher.update(self.price_tolerance_bps.to_le_bytes());
        hasher.update(self.timeout_ns.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Flatten Result ID
// =============================================================================

/// Unique flatten result ID (deterministic).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlattenResultId(pub String);

impl FlattenResultId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Derive from request_id + outcome + ts_ns.
    pub fn derive(request_id: &FlattenRequestId, outcome: &FlattenOutcome, ts_ns: i64) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"flatten_result:");
        hasher.update(request_id.0.as_bytes());
        hasher.update(b":");
        hasher.update([outcome.discriminant()]);
        hasher.update(b":");
        hasher.update(ts_ns.to_le_bytes());
        Self(format!("{:x}", hasher.finalize()))
    }
}

impl std::fmt::Display for FlattenResultId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Flatten Outcome
// =============================================================================

/// Flatten outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
pub enum FlattenOutcome {
    /// All positions successfully closed.
    Success = 0,

    /// Some positions closed, some failed.
    PartialSuccess = 1,

    /// Timed out before completion.
    Timeout = 2,

    /// Failed due to exchange connectivity.
    ConnectivityFailure = 3,

    /// Failed due to insufficient liquidity.
    LiquidityFailure = 4,
}

impl FlattenOutcome {
    /// Get discriminant for canonical encoding.
    pub fn discriminant(&self) -> u8 {
        match self {
            FlattenOutcome::Success => 0,
            FlattenOutcome::PartialSuccess => 1,
            FlattenOutcome::Timeout => 2,
            FlattenOutcome::ConnectivityFailure => 3,
            FlattenOutcome::LiquidityFailure => 4,
        }
    }

    /// Check if flatten completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(self, FlattenOutcome::Success)
    }
}

// =============================================================================
// Emergency Flatten Result
// =============================================================================

/// Emergency flatten result (WAL artifact).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyFlattenResult {
    pub schema_version: String,
    pub result_id: FlattenResultId,
    pub request_id: FlattenRequestId,
    pub ts_ns: i64,

    /// Flatten outcome.
    pub outcome: FlattenOutcome,

    /// Positions closed.
    pub positions_closed: u32,

    /// Positions failed to close.
    pub positions_failed: u32,

    /// Orders generated.
    pub orders_generated: Vec<String>,

    /// Total realized PnL from flatten (mantissa).
    pub realized_pnl_mantissa: i128,
    pub pnl_exponent: i8,

    /// Duration of flatten operation (ns).
    pub duration_ns: i64,

    /// Digest.
    pub digest: String,
}

impl EmergencyFlattenResult {
    /// Compute deterministic digest (SHA-256).
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();

        hasher.update(self.schema_version.as_bytes());
        hasher.update(self.result_id.0.as_bytes());
        hasher.update(self.request_id.0.as_bytes());
        hasher.update(self.ts_ns.to_le_bytes());

        hasher.update([self.outcome.discriminant()]);
        hasher.update(self.positions_closed.to_le_bytes());
        hasher.update(self.positions_failed.to_le_bytes());

        hasher.update((self.orders_generated.len() as u32).to_le_bytes());
        for order_id in &self.orders_generated {
            hasher.update((order_id.len() as u32).to_le_bytes());
            hasher.update(order_id.as_bytes());
        }

        hasher.update(self.realized_pnl_mantissa.to_le_bytes());
        hasher.update([self.pnl_exponent as u8]);
        hasher.update(self.duration_ns.to_le_bytes());

        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Session Status Summary
// =============================================================================

/// Session status summary (for TUI/API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatusSummary {
    pub session_id: String,
    pub state: SessionState,
    pub state_since_ts_ns: i64,
    pub active_kill_switches: Vec<KillSwitchScope>,
    pub last_transition_reason: Option<SessionTransitionReason>,
    pub transition_count: u32,
    pub override_count: u32,
}

// =============================================================================
// Session Controller
// =============================================================================

/// Execution session controller.
/// Integrates: RiskEvaluator, MtmEvaluator, IntentShaper, KillSwitchRegistry.
pub struct SessionController {
    session_id: String,
    state: SessionState,
    state_since_ts_ns: i64,
    last_override_ts_ns: i64,
    kill_switches: KillSwitchRegistry,
    override_policy: OverridePolicy,

    /// Transition log (WAL-bound).
    transition_log: Vec<SessionTransitionEvent>,

    /// Override log (WAL-bound).
    override_log: Vec<ManualOverrideEvent>,
}

impl SessionController {
    /// Create new session controller.
    pub fn new(session_id: &str, override_policy: OverridePolicy) -> Self {
        Self {
            session_id: session_id.to_string(),
            state: SessionState::Active,
            state_since_ts_ns: 0,
            last_override_ts_ns: 0,
            kill_switches: KillSwitchRegistry::new(),
            override_policy,
            transition_log: Vec::new(),
            override_log: Vec::new(),
        }
    }

    /// Get session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get current session state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Get kill-switch registry reference.
    pub fn kill_switches(&self) -> &KillSwitchRegistry {
        &self.kill_switches
    }

    /// Get mutable kill-switch registry reference.
    pub fn kill_switches_mut(&mut self) -> &mut KillSwitchRegistry {
        &mut self.kill_switches
    }

    /// Check if intent type is allowed given current state.
    /// PRECEDENCE: SessionState is checked BEFORE KillSwitchRegistry.
    pub fn is_intent_allowed(&self, intent_type: IntentType) -> bool {
        match self.state {
            SessionState::Active => true,
            SessionState::ReduceOnly => intent_type.is_reduce_or_close(),
            SessionState::Halted | SessionState::Draining | SessionState::Terminated => false,
        }
    }

    /// Check if execution is allowed for given context.
    /// PRECEDENCE: SessionState.Halted overrides KillSwitchRegistry.
    pub fn is_execution_allowed(
        &self,
        intent_type: IntentType,
        bucket_id: &str,
        strategy_id: &str,
        symbol: &str,
    ) -> bool {
        // First check session state
        if !self.is_intent_allowed(intent_type) {
            return false;
        }

        // Then check kill-switches (only if session state allows execution)
        !self.kill_switches.is_halted(bucket_id, strategy_id, symbol)
    }

    /// Validate state transition.
    fn is_valid_transition(&self, to: SessionState) -> bool {
        match (self.state, to) {
            // From Active
            (SessionState::Active, SessionState::Halted) => true,
            (SessionState::Active, SessionState::ReduceOnly) => true,
            (SessionState::Active, SessionState::Draining) => true,

            // From Halted
            (SessionState::Halted, SessionState::ReduceOnly) => true, // Manual ack required
            (SessionState::Halted, SessionState::Terminated) => true,

            // From ReduceOnly
            (SessionState::ReduceOnly, SessionState::Active) => true, // Manual clear
            (SessionState::ReduceOnly, SessionState::Halted) => true, // Re-halt
            (SessionState::ReduceOnly, SessionState::Draining) => true,

            // From Draining
            (SessionState::Draining, SessionState::Halted) => true, // Emergency
            (SessionState::Draining, SessionState::Terminated) => true,

            // Same state (no-op)
            (from, to) if from == to => true,

            // All other transitions invalid
            _ => false,
        }
    }

    /// Transition session state.
    fn transition(
        &mut self,
        to: SessionState,
        reason: SessionTransitionReason,
        ts_ns: i64,
        risk_digest: Option<String>,
        mtm_digest: Option<String>,
        drawdown_digest: Option<String>,
    ) -> Result<SessionTransitionEvent, OverrideError> {
        if !self.is_valid_transition(to) {
            return Err(OverrideError::InvalidTransition {
                from: self.state,
                to,
            });
        }

        if self.state.is_terminal() {
            return Err(OverrideError::SessionTerminal);
        }

        let from = self.state;
        let transition_id = TransitionId::derive(&self.session_id, from, to, ts_ns);

        let mut event = SessionTransitionEvent {
            schema_version: EXECUTION_SESSION_SCHEMA_VERSION.to_string(),
            transition_id,
            ts_ns,
            session_id: self.session_id.clone(),
            from_state: from,
            to_state: to,
            reason,
            risk_snapshot_digest: risk_digest,
            mtm_snapshot_digest: mtm_digest,
            drawdown_snapshot_digest: drawdown_digest,
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        // Apply transition
        self.state = to;
        self.state_since_ts_ns = ts_ns;
        self.transition_log.push(event.clone());

        Ok(event)
    }

    /// Process risk decision and potentially trigger state transition.
    pub fn process_risk_decision(
        &mut self,
        violations: &[ViolationType],
        ts_ns: i64,
        risk_digest: Option<String>,
    ) -> Option<SessionTransitionEvent> {
        // Check for halt violations
        let halt_violations: Vec<_> = violations.iter().filter(|v| v.is_halt()).cloned().collect();

        if !halt_violations.is_empty() && self.state != SessionState::Halted {
            let reason = SessionTransitionReason::RiskViolation {
                violations: halt_violations,
            };
            self.transition(SessionState::Halted, reason, ts_ns, risk_digest, None, None)
                .ok()
        } else {
            None
        }
    }

    /// Process equity violations and potentially trigger state transition.
    pub fn process_equity_violations(
        &mut self,
        violations: &[EquityViolationType],
        ts_ns: i64,
        mtm_digest: Option<String>,
        drawdown_digest: Option<String>,
    ) -> Option<SessionTransitionEvent> {
        // Check for halt violations
        let halt_violations: Vec<_> = violations.iter().filter(|v| v.is_halt()).cloned().collect();

        if !halt_violations.is_empty() && self.state != SessionState::Halted {
            let reason = SessionTransitionReason::EquityViolation {
                violations: halt_violations,
            };
            self.transition(
                SessionState::Halted,
                reason,
                ts_ns,
                None,
                mtm_digest,
                drawdown_digest,
            )
            .ok()
        } else {
            None
        }
    }

    /// Process circuit breaker trip.
    pub fn process_circuit_breaker(
        &mut self,
        breaker_type: CircuitBreakerType,
        ts_ns: i64,
    ) -> Option<SessionTransitionEvent> {
        if self.state != SessionState::Halted {
            let reason = SessionTransitionReason::CircuitBreakerTrip { breaker_type };
            self.transition(SessionState::Halted, reason, ts_ns, None, None, None)
                .ok()
        } else {
            None
        }
    }

    /// Apply manual override.
    pub fn apply_override(
        &mut self,
        override_type: OverrideType,
        operator_id: &str,
        reason: &str,
        ts_ns: i64,
        portfolio_digest: &str,
        risk_digest: &str,
    ) -> Result<ManualOverrideEvent, OverrideError> {
        // Validate reason length
        if reason.len() < self.override_policy.min_reason_length {
            return Err(OverrideError::ReasonTooShort {
                actual: reason.len(),
                required: self.override_policy.min_reason_length,
            });
        }

        // Validate operator authorization
        if !self.override_policy.allowed_operators.is_empty()
            && !self
                .override_policy
                .allowed_operators
                .contains(&operator_id.to_string())
        {
            return Err(OverrideError::OperatorNotAuthorized {
                operator_id: operator_id.to_string(),
            });
        }

        // Validate cooldown
        let time_since_last = ts_ns - self.last_override_ts_ns;
        if time_since_last < self.override_policy.cooldown_ns && self.last_override_ts_ns > 0 {
            return Err(OverrideError::CooldownActive {
                remaining_ns: self.override_policy.cooldown_ns - time_since_last,
            });
        }

        // Check 2FA requirement (flag only - actual 2FA implementation external)
        if self.override_policy.require_2fa.contains(&override_type) {
            // In production, this would check against a 2FA token
            // For now, we just document the requirement
        }

        // Check terminal state
        if self.state.is_terminal() {
            return Err(OverrideError::SessionTerminal);
        }

        let prior_state = self.state;
        let resulting_state = match &override_type {
            OverrideType::ForceHalt => SessionState::Halted,
            OverrideType::ForceReduceOnly => SessionState::ReduceOnly,
            OverrideType::ClearHalt => SessionState::ReduceOnly, // Must go through ReduceOnly
            OverrideType::RestoreFull => SessionState::Active,
            OverrideType::EmergencyFlatten => SessionState::Halted, // Flatten starts from Halted
            OverrideType::CancelAllOrders => self.state,            // No state change
            OverrideType::AdjustLimit { .. } => self.state,         // No state change
        };

        // Validate the transition
        if !self.is_valid_transition(resulting_state) {
            return Err(OverrideError::InvalidTransition {
                from: prior_state,
                to: resulting_state,
            });
        }

        // Create transition event
        let transition_reason = match &override_type {
            OverrideType::ForceHalt => SessionTransitionReason::ManualHalt {
                operator_id: operator_id.to_string(),
                reason: reason.to_string(),
            },
            OverrideType::ForceReduceOnly => SessionTransitionReason::ManualReduceOnly {
                operator_id: operator_id.to_string(),
                reason: reason.to_string(),
            },
            OverrideType::ClearHalt => SessionTransitionReason::ManualRecoveryStart {
                operator_id: operator_id.to_string(),
            },
            OverrideType::RestoreFull => SessionTransitionReason::ManualRecoveryComplete {
                operator_id: operator_id.to_string(),
            },
            OverrideType::EmergencyFlatten => SessionTransitionReason::ManualHalt {
                operator_id: operator_id.to_string(),
                reason: format!("Emergency flatten: {}", reason),
            },
            _ => SessionTransitionReason::ManualHalt {
                operator_id: operator_id.to_string(),
                reason: reason.to_string(),
            },
        };

        // Apply state change if needed
        if prior_state != resulting_state {
            let _ = self.transition(
                resulting_state,
                transition_reason,
                ts_ns,
                Some(risk_digest.to_string()),
                None,
                None,
            )?;
        }

        // Create override event
        let override_id = OverrideId::derive(&self.session_id, &override_type, ts_ns);
        let mut event = ManualOverrideEvent {
            schema_version: MANUAL_OVERRIDE_SCHEMA_VERSION.to_string(),
            override_id,
            ts_ns,
            session_id: self.session_id.clone(),
            operator_id: operator_id.to_string(),
            override_type,
            reason: reason.to_string(),
            scope: None,
            prior_session_state: prior_state,
            resulting_session_state: resulting_state,
            portfolio_snapshot_digest: portfolio_digest.to_string(),
            risk_snapshot_digest: risk_digest.to_string(),
            digest: String::new(),
        };
        event.digest = event.compute_digest();

        self.last_override_ts_ns = ts_ns;
        self.override_log.push(event.clone());

        Ok(event)
    }

    /// Get session state summary (for TUI/API).
    pub fn status_summary(&self) -> SessionStatusSummary {
        SessionStatusSummary {
            session_id: self.session_id.clone(),
            state: self.state,
            state_since_ts_ns: self.state_since_ts_ns,
            active_kill_switches: self.kill_switches.active_switches(),
            last_transition_reason: self.transition_log.last().map(|e| e.reason.clone()),
            transition_count: self.transition_log.len() as u32,
            override_count: self.override_log.len() as u32,
        }
    }

    /// Get transition log.
    pub fn transition_log(&self) -> &[SessionTransitionEvent] {
        &self.transition_log
    }

    /// Get override log.
    pub fn override_log(&self) -> &[ManualOverrideEvent] {
        &self.override_log
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> SessionController {
        SessionController::new("test_session", OverridePolicy::permissive())
    }

    fn make_strict_controller() -> SessionController {
        SessionController::new("test_session", OverridePolicy::default())
    }

    // -------------------------------------------------------------------------
    // State Machine Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_session_state_machine_valid_transitions() {
        let mut ctrl = make_controller();

        // Active -> Halted
        assert!(ctrl.is_valid_transition(SessionState::Halted));
        let event = ctrl
            .transition(
                SessionState::Halted,
                SessionTransitionReason::ManualHalt {
                    operator_id: "op1".to_string(),
                    reason: "test halt".to_string(),
                },
                1000,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(event.from_state, SessionState::Active);
        assert_eq!(event.to_state, SessionState::Halted);
        assert_eq!(ctrl.state(), SessionState::Halted);

        // Halted -> ReduceOnly
        assert!(ctrl.is_valid_transition(SessionState::ReduceOnly));
        ctrl.transition(
            SessionState::ReduceOnly,
            SessionTransitionReason::ManualRecoveryStart {
                operator_id: "op1".to_string(),
            },
            2000,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(ctrl.state(), SessionState::ReduceOnly);

        // ReduceOnly -> Active
        assert!(ctrl.is_valid_transition(SessionState::Active));
        ctrl.transition(
            SessionState::Active,
            SessionTransitionReason::ManualRecoveryComplete {
                operator_id: "op1".to_string(),
            },
            3000,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(ctrl.state(), SessionState::Active);
    }

    #[test]
    fn test_session_state_machine_invalid_transitions() {
        let mut ctrl = make_controller();

        // Active -> Terminated (invalid)
        assert!(!ctrl.is_valid_transition(SessionState::Terminated));
        let result = ctrl.transition(
            SessionState::Terminated,
            SessionTransitionReason::SessionComplete,
            1000,
            None,
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(OverrideError::InvalidTransition { .. })
        ));

        // Go to Halted first
        ctrl.transition(
            SessionState::Halted,
            SessionTransitionReason::ManualHalt {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
            None,
            None,
            None,
        )
        .unwrap();

        // Halted -> Active (invalid, must go through ReduceOnly)
        assert!(!ctrl.is_valid_transition(SessionState::Active));
    }

    // -------------------------------------------------------------------------
    // Kill-Switch Hierarchy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_kill_switch_hierarchy_global() {
        let mut registry = KillSwitchRegistry::new();

        // Activate global
        registry.activate(
            KillSwitchScope::Global,
            KillSwitchReason::ManualActivation {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
        );

        // Everything should be halted
        assert!(registry.is_halted("bucket1", "strategy1", "BTCUSDT"));
        assert!(registry.is_halted("bucket2", "strategy2", "ETHUSDT"));
        assert!(registry.is_global_halted());
    }

    #[test]
    fn test_kill_switch_hierarchy_bucket() {
        let mut registry = KillSwitchRegistry::new();

        // Activate bucket
        registry.activate(
            KillSwitchScope::Bucket {
                bucket_id: "bucket1".to_string(),
            },
            KillSwitchReason::ManualActivation {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
        );

        // bucket1 should be halted
        assert!(registry.is_halted("bucket1", "strategy1", "BTCUSDT"));
        assert!(registry.is_halted("bucket1", "strategy2", "ETHUSDT"));

        // bucket2 should NOT be halted
        assert!(!registry.is_halted("bucket2", "strategy1", "BTCUSDT"));
    }

    #[test]
    fn test_kill_switch_hierarchy_strategy() {
        let mut registry = KillSwitchRegistry::new();

        // Activate strategy
        registry.activate(
            KillSwitchScope::Strategy {
                strategy_id: "strategy1".to_string(),
            },
            KillSwitchReason::ManualActivation {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
        );

        // strategy1 should be halted
        assert!(registry.is_halted("bucket1", "strategy1", "BTCUSDT"));
        assert!(registry.is_halted("bucket2", "strategy1", "ETHUSDT"));

        // strategy2 should NOT be halted
        assert!(!registry.is_halted("bucket1", "strategy2", "BTCUSDT"));
    }

    #[test]
    fn test_kill_switch_hierarchy_symbol() {
        let mut registry = KillSwitchRegistry::new();

        // Activate symbol
        registry.activate(
            KillSwitchScope::Symbol {
                symbol: "BTCUSDT".to_string(),
            },
            KillSwitchReason::ManualActivation {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
        );

        // BTCUSDT should be halted
        assert!(registry.is_halted("bucket1", "strategy1", "BTCUSDT"));
        assert!(registry.is_halted("bucket2", "strategy2", "BTCUSDT"));

        // ETHUSDT should NOT be halted
        assert!(!registry.is_halted("bucket1", "strategy1", "ETHUSDT"));
    }

    // -------------------------------------------------------------------------
    // Manual Override Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_manual_halt_requires_reason() {
        let mut ctrl = make_strict_controller();

        // Short reason should fail
        let result = ctrl.apply_override(
            OverrideType::ForceHalt,
            "op1",
            "short", // < 10 chars
            1000,
            "portfolio_digest",
            "risk_digest",
        );
        assert!(matches!(result, Err(OverrideError::ReasonTooShort { .. })));
    }

    #[test]
    fn test_manual_halt_cooldown() {
        let mut ctrl = make_strict_controller();

        // First override should succeed
        ctrl.apply_override(
            OverrideType::ForceHalt,
            "op1",
            "First halt for testing purposes",
            1000,
            "portfolio_digest",
            "risk_digest",
        )
        .unwrap();

        // Need to go through ReduceOnly first
        ctrl.apply_override(
            OverrideType::ClearHalt,
            "op1",
            "Clear halt for testing purposes",
            2000 + ctrl.override_policy.cooldown_ns, // After cooldown
            "portfolio_digest",
            "risk_digest",
        )
        .unwrap();

        // Try immediate second override (should fail due to cooldown)
        let result = ctrl.apply_override(
            OverrideType::RestoreFull,
            "op1",
            "Restore for testing purposes",
            2000 + ctrl.override_policy.cooldown_ns + 1000, // Within cooldown
            "portfolio_digest",
            "risk_digest",
        );
        assert!(matches!(result, Err(OverrideError::CooldownActive { .. })));
    }

    // -------------------------------------------------------------------------
    // Reduce-Only Mode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reduce_only_allows_close() {
        let mut ctrl = make_controller();

        // Go to ReduceOnly
        ctrl.transition(
            SessionState::Halted,
            SessionTransitionReason::ManualHalt {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
            None,
            None,
            None,
        )
        .unwrap();
        ctrl.transition(
            SessionState::ReduceOnly,
            SessionTransitionReason::ManualRecoveryStart {
                operator_id: "op1".to_string(),
            },
            2000,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(ctrl.is_intent_allowed(IntentType::Close));
        assert!(ctrl.is_intent_allowed(IntentType::Reduce));
    }

    #[test]
    fn test_reduce_only_blocks_increase() {
        let mut ctrl = make_controller();

        // Go to ReduceOnly
        ctrl.transition(
            SessionState::Halted,
            SessionTransitionReason::ManualHalt {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            1000,
            None,
            None,
            None,
        )
        .unwrap();
        ctrl.transition(
            SessionState::ReduceOnly,
            SessionTransitionReason::ManualRecoveryStart {
                operator_id: "op1".to_string(),
            },
            2000,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(!ctrl.is_intent_allowed(IntentType::Increase));
    }

    // -------------------------------------------------------------------------
    // Emergency Flatten Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emergency_flatten_request_digest() {
        let scope = FlattenScope::All;
        let request_id = FlattenRequestId::derive("op1", &scope, 1000);

        let request = EmergencyFlattenRequest {
            schema_version: EMERGENCY_FLATTEN_SCHEMA_VERSION.to_string(),
            request_id,
            ts_ns: 1000,
            operator_id: "op1".to_string(),
            reason: "Market crash - flatten all".to_string(),
            scope,
            price_tolerance_bps: 50,
            timeout_ns: 30_000_000_000,
            digest: String::new(),
        };
        let digest = request.compute_digest();
        assert_eq!(digest.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_emergency_flatten_result_digest() {
        let request_id = FlattenRequestId::new("req_001");
        let outcome = FlattenOutcome::Success;
        let result_id = FlattenResultId::derive(&request_id, &outcome, 2000);

        let result = EmergencyFlattenResult {
            schema_version: EMERGENCY_FLATTEN_SCHEMA_VERSION.to_string(),
            result_id,
            request_id,
            ts_ns: 2000,
            outcome,
            positions_closed: 5,
            positions_failed: 0,
            orders_generated: vec!["order1".to_string(), "order2".to_string()],
            realized_pnl_mantissa: -1000_00000000,
            pnl_exponent: -8,
            duration_ns: 5_000_000_000,
            digest: String::new(),
        };
        let digest = result.compute_digest();
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn test_emergency_flatten_partial_result() {
        let request_id = FlattenRequestId::new("req_001");
        let outcome = FlattenOutcome::PartialSuccess;

        let result = EmergencyFlattenResult {
            schema_version: EMERGENCY_FLATTEN_SCHEMA_VERSION.to_string(),
            result_id: FlattenResultId::derive(&request_id, &outcome, 2000),
            request_id,
            ts_ns: 2000,
            outcome,
            positions_closed: 3,
            positions_failed: 2,
            orders_generated: vec!["order1".to_string()],
            realized_pnl_mantissa: -500_00000000,
            pnl_exponent: -8,
            duration_ns: 30_000_000_000,
            digest: String::new(),
        };

        assert!(!result.outcome.is_success());
        assert_eq!(result.positions_failed, 2);
    }

    #[test]
    fn test_emergency_flatten_timeout_result() {
        let request_id = FlattenRequestId::new("req_001");
        let outcome = FlattenOutcome::Timeout;

        let result = EmergencyFlattenResult {
            schema_version: EMERGENCY_FLATTEN_SCHEMA_VERSION.to_string(),
            result_id: FlattenResultId::derive(&request_id, &outcome, 2000),
            request_id,
            ts_ns: 2000,
            outcome,
            positions_closed: 2,
            positions_failed: 3,
            orders_generated: vec![],
            realized_pnl_mantissa: -200_00000000,
            pnl_exponent: -8,
            duration_ns: 30_000_000_000,
            digest: String::new(),
        };

        assert!(!result.outcome.is_success());
        assert_eq!(result.outcome, FlattenOutcome::Timeout);
    }

    // -------------------------------------------------------------------------
    // Digest Determinism Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_session_transition_digest_deterministic() {
        let event1 = SessionTransitionEvent {
            schema_version: EXECUTION_SESSION_SCHEMA_VERSION.to_string(),
            transition_id: TransitionId::derive(
                "session1",
                SessionState::Active,
                SessionState::Halted,
                1000,
            ),
            ts_ns: 1000,
            session_id: "session1".to_string(),
            from_state: SessionState::Active,
            to_state: SessionState::Halted,
            reason: SessionTransitionReason::ManualHalt {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            risk_snapshot_digest: Some("risk_digest".to_string()),
            mtm_snapshot_digest: None,
            drawdown_snapshot_digest: None,
            digest: String::new(),
        };

        let event2 = SessionTransitionEvent {
            schema_version: EXECUTION_SESSION_SCHEMA_VERSION.to_string(),
            transition_id: TransitionId::derive(
                "session1",
                SessionState::Active,
                SessionState::Halted,
                1000,
            ),
            ts_ns: 1000,
            session_id: "session1".to_string(),
            from_state: SessionState::Active,
            to_state: SessionState::Halted,
            reason: SessionTransitionReason::ManualHalt {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            risk_snapshot_digest: Some("risk_digest".to_string()),
            mtm_snapshot_digest: None,
            drawdown_snapshot_digest: None,
            digest: String::new(),
        };

        assert_eq!(event1.compute_digest(), event2.compute_digest());
    }

    #[test]
    fn test_kill_switch_event_digest_deterministic() {
        let event1 = KillSwitchEvent {
            schema_version: KILL_SWITCH_SCHEMA_VERSION.to_string(),
            event_id: KillSwitchEventId::derive(&KillSwitchScope::Global, true, 1000),
            ts_ns: 1000,
            scope: KillSwitchScope::Global,
            activated: true,
            reason: KillSwitchReason::ManualActivation {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            digest: String::new(),
        };

        let event2 = KillSwitchEvent {
            schema_version: KILL_SWITCH_SCHEMA_VERSION.to_string(),
            event_id: KillSwitchEventId::derive(&KillSwitchScope::Global, true, 1000),
            ts_ns: 1000,
            scope: KillSwitchScope::Global,
            activated: true,
            reason: KillSwitchReason::ManualActivation {
                operator_id: "op1".to_string(),
                reason: "test".to_string(),
            },
            digest: String::new(),
        };

        assert_eq!(event1.compute_digest(), event2.compute_digest());
    }

    #[test]
    fn test_override_event_digest_deterministic() {
        let event1 = ManualOverrideEvent {
            schema_version: MANUAL_OVERRIDE_SCHEMA_VERSION.to_string(),
            override_id: OverrideId::derive("session1", &OverrideType::ForceHalt, 1000),
            ts_ns: 1000,
            session_id: "session1".to_string(),
            operator_id: "op1".to_string(),
            override_type: OverrideType::ForceHalt,
            reason: "test override".to_string(),
            scope: None,
            prior_session_state: SessionState::Active,
            resulting_session_state: SessionState::Halted,
            portfolio_snapshot_digest: "portfolio_digest".to_string(),
            risk_snapshot_digest: "risk_digest".to_string(),
            digest: String::new(),
        };

        let event2 = ManualOverrideEvent {
            schema_version: MANUAL_OVERRIDE_SCHEMA_VERSION.to_string(),
            override_id: OverrideId::derive("session1", &OverrideType::ForceHalt, 1000),
            ts_ns: 1000,
            session_id: "session1".to_string(),
            operator_id: "op1".to_string(),
            override_type: OverrideType::ForceHalt,
            reason: "test override".to_string(),
            scope: None,
            prior_session_state: SessionState::Active,
            resulting_session_state: SessionState::Halted,
            portfolio_snapshot_digest: "portfolio_digest".to_string(),
            risk_snapshot_digest: "risk_digest".to_string(),
            digest: String::new(),
        };

        assert_eq!(event1.compute_digest(), event2.compute_digest());
    }
}
