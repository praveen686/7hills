//! # Execution Control View (Phase 17A)
//!
//! Read-only snapshot view of the Phase 16 control plane for observability.
//!
//! ## Design Principle
//! This module provides a **read-only, cloneable, serialization-friendly** view
//! into the execution control plane. It does NOT embed `SessionController` or
//! any mutable state - it only captures snapshots for display and streaming.
//!
//! ## Usage
//! The runner core owns `SessionController`. Phase 17A adds this snapshot view
//! for the web server and TUI layer to consume without coupling to control logic.

use quantlaxmi_gates::{
    EmergencyFlattenResult, FlattenScope, KillSwitchScope, ManualOverrideEvent, OverrideType,
    SessionState, SessionStatusSummary, SessionTransitionEvent, SessionTransitionReason,
};
use serde::{Deserialize, Serialize};

/// Schema version for control view serialization.
pub const CONTROL_VIEW_SCHEMA_VERSION: &str = "control_view_v1.0";

/// Maximum number of transitions to retain in history.
pub const MAX_TRANSITION_HISTORY: usize = 50;

/// Maximum number of overrides to retain in history.
pub const MAX_OVERRIDE_HISTORY: usize = 50;

// =============================================================================
// Execution Control View
// =============================================================================

/// Read-only snapshot view of the execution control plane.
///
/// This struct is designed to be:
/// - Cloneable (for broadcast)
/// - Serializable (for WebSocket/JSON)
/// - Bounded (fixed-size history rings)
/// - Pure "view" (no computation, no mutation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionControlView {
    /// Schema version for compatibility.
    pub schema_version: String,

    /// Session status summary (from Phase 16).
    pub session: SessionStatusSummary,

    /// Recent session transitions (bounded ring, newest first).
    pub recent_transitions: Vec<SessionTransitionEvent>,

    /// Active kill-switches (from Phase 16 registry).
    pub active_kill_switches: Vec<KillSwitchScope>,

    /// Recent manual overrides (bounded ring, newest first).
    pub recent_overrides: Vec<ManualOverrideEvent>,

    /// Last emergency flatten result (if any).
    pub last_flatten: Option<EmergencyFlattenResult>,

    /// Snapshot timestamp (nanoseconds).
    pub ts_ns: i64,
}

impl ExecutionControlView {
    /// Create an empty control view for a new session.
    pub fn new(session_id: &str) -> Self {
        Self {
            schema_version: CONTROL_VIEW_SCHEMA_VERSION.to_string(),
            session: SessionStatusSummary {
                session_id: session_id.to_string(),
                state: SessionState::Active,
                state_since_ts_ns: 0,
                active_kill_switches: Vec::new(),
                last_transition_reason: None,
                transition_count: 0,
                override_count: 0,
            },
            recent_transitions: Vec::new(),
            active_kill_switches: Vec::new(),
            recent_overrides: Vec::new(),
            last_flatten: None,
            ts_ns: 0,
        }
    }

    /// Update from session status summary.
    pub fn update_session(&mut self, summary: SessionStatusSummary, ts_ns: i64) {
        self.session = summary;
        self.ts_ns = ts_ns;
    }

    /// Add a transition event to history (maintains bounded size).
    pub fn add_transition(&mut self, event: SessionTransitionEvent) {
        self.recent_transitions.insert(0, event);
        if self.recent_transitions.len() > MAX_TRANSITION_HISTORY {
            self.recent_transitions.pop();
        }
    }

    /// Add an override event to history (maintains bounded size).
    pub fn add_override(&mut self, event: ManualOverrideEvent) {
        self.recent_overrides.insert(0, event);
        if self.recent_overrides.len() > MAX_OVERRIDE_HISTORY {
            self.recent_overrides.pop();
        }
    }

    /// Update active kill-switches.
    pub fn update_kill_switches(&mut self, switches: Vec<KillSwitchScope>) {
        self.active_kill_switches = switches;
    }

    /// Set last flatten result.
    pub fn set_flatten_result(&mut self, result: EmergencyFlattenResult) {
        self.last_flatten = Some(result);
    }

    /// Check if session is in a halted state.
    pub fn is_halted(&self) -> bool {
        self.session.state == SessionState::Halted
    }

    /// Check if session is in reduce-only mode.
    pub fn is_reduce_only(&self) -> bool {
        self.session.state == SessionState::ReduceOnly
    }

    /// Check if any kill-switch is active.
    pub fn has_active_kill_switches(&self) -> bool {
        !self.active_kill_switches.is_empty()
    }
}

impl Default for ExecutionControlView {
    fn default() -> Self {
        Self::new("unknown")
    }
}

// =============================================================================
// Operator Request / Response
// =============================================================================

/// Operator request types (maps exactly to Phase 16 operations).
///
/// Every request type requires `operator_id` for audit trail.
/// Requests that modify state require `reason` (minimum length enforced by Phase 16).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", content = "params")]
pub enum OperatorRequest {
    /// Force session halt.
    ForceHalt { operator_id: String, reason: String },

    /// Force reduce-only mode.
    ForceReduceOnly { operator_id: String, reason: String },

    /// Clear halt, enter reduce-only recovery.
    ClearHalt { operator_id: String },

    /// Restore full operation from reduce-only.
    RestoreFull { operator_id: String },

    /// Activate kill-switch at scope.
    ActivateKillSwitch {
        operator_id: String,
        reason: String,
        scope: KillSwitchScope,
    },

    /// Deactivate kill-switch at scope.
    DeactivateKillSwitch {
        operator_id: String,
        scope: KillSwitchScope,
    },

    /// Emergency flatten positions.
    EmergencyFlatten {
        operator_id: String,
        reason: String,
        scope: FlattenScope,
        price_tolerance_bps: u32,
        timeout_ns: i64,
    },

    /// Cancel all open orders.
    CancelAllOrders { operator_id: String, reason: String },

    /// Adjust position limit (temporary).
    AdjustLimit {
        operator_id: String,
        reason: String,
        scope: KillSwitchScope,
        new_limit_mantissa: i128,
    },
}

impl OperatorRequest {
    /// Get the operator ID from any request.
    pub fn operator_id(&self) -> &str {
        match self {
            OperatorRequest::ForceHalt { operator_id, .. } => operator_id,
            OperatorRequest::ForceReduceOnly { operator_id, .. } => operator_id,
            OperatorRequest::ClearHalt { operator_id } => operator_id,
            OperatorRequest::RestoreFull { operator_id } => operator_id,
            OperatorRequest::ActivateKillSwitch { operator_id, .. } => operator_id,
            OperatorRequest::DeactivateKillSwitch { operator_id, .. } => operator_id,
            OperatorRequest::EmergencyFlatten { operator_id, .. } => operator_id,
            OperatorRequest::CancelAllOrders { operator_id, .. } => operator_id,
            OperatorRequest::AdjustLimit { operator_id, .. } => operator_id,
        }
    }

    /// Get the reason if present.
    pub fn reason(&self) -> Option<&str> {
        match self {
            OperatorRequest::ForceHalt { reason, .. } => Some(reason),
            OperatorRequest::ForceReduceOnly { reason, .. } => Some(reason),
            OperatorRequest::ClearHalt { .. } => None,
            OperatorRequest::RestoreFull { .. } => None,
            OperatorRequest::ActivateKillSwitch { reason, .. } => Some(reason),
            OperatorRequest::DeactivateKillSwitch { .. } => None,
            OperatorRequest::EmergencyFlatten { reason, .. } => Some(reason),
            OperatorRequest::CancelAllOrders { reason, .. } => Some(reason),
            OperatorRequest::AdjustLimit { reason, .. } => Some(reason),
        }
    }

    /// Map to Phase 16 OverrideType (for those that map directly).
    pub fn to_override_type(&self) -> Option<OverrideType> {
        match self {
            OperatorRequest::ForceHalt { .. } => Some(OverrideType::ForceHalt),
            OperatorRequest::ForceReduceOnly { .. } => Some(OverrideType::ForceReduceOnly),
            OperatorRequest::ClearHalt { .. } => Some(OverrideType::ClearHalt),
            OperatorRequest::RestoreFull { .. } => Some(OverrideType::RestoreFull),
            OperatorRequest::EmergencyFlatten { .. } => Some(OverrideType::EmergencyFlatten),
            OperatorRequest::CancelAllOrders { .. } => Some(OverrideType::CancelAllOrders),
            OperatorRequest::AdjustLimit {
                scope,
                new_limit_mantissa,
                ..
            } => Some(OverrideType::AdjustLimit {
                scope: scope.clone(),
                new_limit_mantissa: *new_limit_mantissa,
            }),
            OperatorRequest::ActivateKillSwitch { .. } => None, // Not an override
            OperatorRequest::DeactivateKillSwitch { .. } => None, // Not an override
        }
    }
}

/// Operator response outcome.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "status", content = "details")]
pub enum OperatorOutcome {
    /// Request accepted and processed.
    Accepted { resulting_state: SessionState },

    /// Request rejected with error.
    Rejected { error_code: String, message: String },
}

impl OperatorOutcome {
    /// Create an accepted outcome.
    pub fn accepted(resulting_state: SessionState) -> Self {
        OperatorOutcome::Accepted { resulting_state }
    }

    /// Create a rejected outcome.
    pub fn rejected(error_code: impl Into<String>, message: impl Into<String>) -> Self {
        OperatorOutcome::Rejected {
            error_code: error_code.into(),
            message: message.into(),
        }
    }

    /// Check if outcome is accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, OperatorOutcome::Accepted { .. })
    }
}

/// Operator response (sent back to client).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorResponse {
    /// Correlation ID (echoed from request if provided).
    pub correlation_id: Option<String>,

    /// Outcome of the operation.
    pub outcome: OperatorOutcome,

    /// Timestamp of response.
    pub ts_ns: i64,
}

impl OperatorResponse {
    /// Create a new response.
    pub fn new(correlation_id: Option<String>, outcome: OperatorOutcome, ts_ns: i64) -> Self {
        Self {
            correlation_id,
            outcome,
            ts_ns,
        }
    }
}

// =============================================================================
// Display Helpers
// =============================================================================

/// Format session state for display with color hint.
pub fn format_session_state(state: SessionState) -> (&'static str, &'static str) {
    match state {
        SessionState::Active => ("ACTIVE", "green"),
        SessionState::Halted => ("HALTED", "red"),
        SessionState::ReduceOnly => ("REDUCE_ONLY", "yellow"),
        SessionState::Draining => ("DRAINING", "blue"),
        SessionState::Terminated => ("TERMINATED", "gray"),
    }
}

/// Format kill-switch scope for display.
pub fn format_kill_switch_scope(scope: &KillSwitchScope) -> String {
    match scope {
        KillSwitchScope::Global => "GLOBAL".to_string(),
        KillSwitchScope::Bucket { bucket_id } => format!("BUCKET:{}", bucket_id),
        KillSwitchScope::Strategy { strategy_id } => format!("STRATEGY:{}", strategy_id),
        KillSwitchScope::Symbol { symbol } => format!("SYMBOL:{}", symbol),
    }
}

/// Format transition reason for display (human-readable).
pub fn format_transition_reason(reason: &SessionTransitionReason) -> String {
    match reason {
        SessionTransitionReason::RiskViolation { violations } => {
            format!("Risk violation: {} violations", violations.len())
        }
        SessionTransitionReason::EquityViolation { violations } => {
            format!("Equity violation: {} violations", violations.len())
        }
        SessionTransitionReason::CircuitBreakerTrip { breaker_type } => {
            format!("Circuit breaker: {}", breaker_type.code())
        }
        SessionTransitionReason::ConnectivityLoss {
            exchange,
            duration_ms,
        } => {
            format!("Connectivity loss: {} ({}ms)", exchange, duration_ms)
        }
        SessionTransitionReason::ManualHalt {
            operator_id,
            reason,
        } => {
            format!("Manual halt by {}: {}", operator_id, reason)
        }
        SessionTransitionReason::ManualReduceOnly {
            operator_id,
            reason,
        } => {
            format!("Manual reduce-only by {}: {}", operator_id, reason)
        }
        SessionTransitionReason::ManualRecoveryStart { operator_id } => {
            format!("Recovery started by {}", operator_id)
        }
        SessionTransitionReason::ManualRecoveryComplete { operator_id } => {
            format!("Recovery completed by {}", operator_id)
        }
        SessionTransitionReason::ManualDrain {
            operator_id,
            reason,
        } => {
            format!("Manual drain by {}: {}", operator_id, reason)
        }
        SessionTransitionReason::SessionComplete => "Session complete".to_string(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_view_bounded_history() {
        let mut view = ExecutionControlView::new("test_session");

        // Add more than MAX_TRANSITION_HISTORY transitions
        for i in 0..60 {
            let event = SessionTransitionEvent {
                schema_version: "test".to_string(),
                transition_id: quantlaxmi_gates::TransitionId::new(format!("t_{}", i)),
                ts_ns: i as i64,
                session_id: "test_session".to_string(),
                from_state: SessionState::Active,
                to_state: SessionState::Halted,
                reason: SessionTransitionReason::ManualHalt {
                    operator_id: "op".to_string(),
                    reason: "test".to_string(),
                },
                risk_snapshot_digest: None,
                mtm_snapshot_digest: None,
                drawdown_snapshot_digest: None,
                digest: String::new(),
            };
            view.add_transition(event);
        }

        assert_eq!(view.recent_transitions.len(), MAX_TRANSITION_HISTORY);
        // Newest should be first
        assert_eq!(view.recent_transitions[0].ts_ns, 59);
    }

    #[test]
    fn test_operator_request_roundtrip() {
        let request = OperatorRequest::ForceHalt {
            operator_id: "op1".to_string(),
            reason: "Market volatility too high".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: OperatorRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request, parsed);
    }

    #[test]
    fn test_operator_response_roundtrip() {
        let response = OperatorResponse::new(
            Some("corr_123".to_string()),
            OperatorOutcome::accepted(SessionState::Halted),
            1000,
        );

        let json = serde_json::to_string(&response).unwrap();
        let parsed: OperatorResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(response.correlation_id, parsed.correlation_id);
        assert!(parsed.outcome.is_accepted());
    }

    #[test]
    fn test_operator_outcome_rejected() {
        let outcome = OperatorOutcome::rejected("COOLDOWN", "Cooldown active: 30s remaining");

        assert!(!outcome.is_accepted());

        if let OperatorOutcome::Rejected { error_code, .. } = outcome {
            assert_eq!(error_code, "COOLDOWN");
        }
    }

    #[test]
    fn test_format_session_state() {
        let (text, color) = format_session_state(SessionState::Halted);
        assert_eq!(text, "HALTED");
        assert_eq!(color, "red");
    }

    #[test]
    fn test_format_kill_switch_scope() {
        let scope = KillSwitchScope::Strategy {
            strategy_id: "funding_arb".to_string(),
        };
        assert_eq!(format_kill_switch_scope(&scope), "STRATEGY:funding_arb");
    }
}
