//! Strategy Admission Control types.
//!
//! Phase 21B: Extends signal admission to strategy-level binding.
//! A signal cannot influence execution unless:
//! 1. It is promoted (20D), AND
//! 2. It is bound to a strategy in strategies_manifest.json
//!
//! ## Core Question
//! "Is this signal allowed to influence this strategy's execution?"
//!
//! ## Hard Laws (Frozen)
//! - STRATEGY_ADMISSION_SCHEMA_VERSION: "1.0.0"
//! - StrategyAdmissionOutcome variants + canonical bytes
//! - StrategyRefuseReason variants + canonical bytes
//! - StrategyAdmissionDecision canonical bytes order
//! - correlation_id required (not Option)
//!
//! See: Phase 21 spec for full details.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

/// Schema version for strategy admission decision serialization.
pub const STRATEGY_ADMISSION_SCHEMA_VERSION: &str = "1.0.0";

// =============================================================================
// StrategyAdmissionOutcome — The binary decision
// =============================================================================

/// Outcome of strategy admission evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyAdmissionOutcome {
    /// Strategy may use this signal for execution
    Admit,
    /// Strategy MUST NOT use this signal for execution
    Refuse,
}

impl StrategyAdmissionOutcome {
    /// Canonical byte representation (frozen).
    pub fn canonical_byte(self) -> u8 {
        match self {
            StrategyAdmissionOutcome::Admit => 0x01,
            StrategyAdmissionOutcome::Refuse => 0x02,
        }
    }
}

impl std::fmt::Display for StrategyAdmissionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StrategyAdmissionOutcome::Admit => write!(f, "Admit"),
            StrategyAdmissionOutcome::Refuse => write!(f, "Refuse"),
        }
    }
}

// =============================================================================
// StrategyRefuseReason — Why was the signal refused?
// =============================================================================

/// Typed reasons for refusal (eligibility-only, NOT enforcement).
///
/// These are static eligibility checks, not runtime enforcement.
/// Runtime enforcement (rate limits, position limits) is Phase 21D/22.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StrategyRefuseReason {
    /// Signal was not admitted at L1 layer
    SignalNotAdmitted { signal_id: String },

    /// Strategy not in strategies_manifest
    StrategyNotFound { strategy_id: String },

    /// Signal is not bound to this strategy in manifest
    SignalNotBound {
        signal_id: String,
        strategy_id: String,
    },
}

impl StrategyRefuseReason {
    /// Canonical bytes for hashing (frozen field order).
    ///
    /// Format:
    /// - SignalNotAdmitted: 0x01 + signal_id (u32 len + UTF-8)
    /// - StrategyNotFound: 0x02 + strategy_id (u32 len + UTF-8)
    /// - SignalNotBound: 0x03 + signal_id (u32 len + UTF-8) + strategy_id (u32 len + UTF-8)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        match self {
            StrategyRefuseReason::SignalNotAdmitted { signal_id } => {
                bytes.push(0x01);
                write_string(&mut bytes, signal_id);
            }
            StrategyRefuseReason::StrategyNotFound { strategy_id } => {
                bytes.push(0x02);
                write_string(&mut bytes, strategy_id);
            }
            StrategyRefuseReason::SignalNotBound {
                signal_id,
                strategy_id,
            } => {
                bytes.push(0x03);
                write_string(&mut bytes, signal_id);
                write_string(&mut bytes, strategy_id);
            }
        }

        bytes
    }

    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            StrategyRefuseReason::SignalNotAdmitted { signal_id } => {
                format!("Signal '{}' was not admitted at L1 layer", signal_id)
            }
            StrategyRefuseReason::StrategyNotFound { strategy_id } => {
                format!(
                    "Strategy '{}' not found in strategies_manifest",
                    strategy_id
                )
            }
            StrategyRefuseReason::SignalNotBound {
                signal_id,
                strategy_id,
            } => {
                format!(
                    "Signal '{}' is not bound to strategy '{}' in manifest",
                    signal_id, strategy_id
                )
            }
        }
    }
}

// =============================================================================
// StrategyAdmissionDecision — The WAL audit artifact
// =============================================================================

/// WAL event for strategy admission decisions.
///
/// Written to `wal/strategy_admission.jsonl` before strategy execution.
/// Provides audit trail for strategy-signal binding decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAdmissionDecision {
    /// Schema version for forward compatibility
    pub schema_version: String,

    /// Timestamp in nanoseconds since epoch (i64, matches AdmissionDecision)
    pub ts_ns: i64,

    /// Session identifier (String, matches AdmissionDecision)
    pub session_id: String,

    /// Strategy being evaluated
    pub strategy_id: String,

    /// Signal that triggered evaluation
    pub signal_id: String,

    /// Admission outcome
    pub outcome: StrategyAdmissionOutcome,

    /// Refusal reasons (empty if Admit)
    pub refuse_reasons: Vec<StrategyRefuseReason>,

    /// Linkage to upstream correlation context (REQUIRED)
    pub correlation_id: String,

    /// SHA-256 of strategies_manifest.json at evaluation time
    pub strategies_manifest_hash: [u8; 32],

    /// SHA-256 of signals_manifest.json at evaluation time
    pub signals_manifest_hash: [u8; 32],

    /// SHA-256 digest of canonical representation (hex string)
    pub digest: String,
}

impl StrategyAdmissionDecision {
    /// Check if this decision allows strategy execution.
    pub fn is_admitted(&self) -> bool {
        self.outcome == StrategyAdmissionOutcome::Admit
    }

    /// Check if this decision refuses strategy execution.
    pub fn is_refused(&self) -> bool {
        self.outcome == StrategyAdmissionOutcome::Refuse
    }

    /// Compute canonical bytes for hashing (frozen field order).
    ///
    /// Field order:
    /// 1. schema_version (u32 LE len + UTF-8)
    /// 2. ts_ns (i64 LE)
    /// 3. session_id (u32 LE len + UTF-8)
    /// 4. strategy_id (u32 LE len + UTF-8)
    /// 5. signal_id (u32 LE len + UTF-8)
    /// 6. outcome (0x01=Admit, 0x02=Refuse)
    /// 7. refuse_reasons (u32 LE count + each reason's canonical bytes)
    /// 8. correlation_id (u32 LE len + UTF-8) — always present, no Option tag
    /// 9. strategies_manifest_hash (32 raw bytes)
    /// 10. signals_manifest_hash (32 raw bytes)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. schema_version
        write_string(&mut bytes, &self.schema_version);

        // 2. ts_ns
        bytes.extend_from_slice(&self.ts_ns.to_le_bytes());

        // 3. session_id
        write_string(&mut bytes, &self.session_id);

        // 4. strategy_id
        write_string(&mut bytes, &self.strategy_id);

        // 5. signal_id
        write_string(&mut bytes, &self.signal_id);

        // 6. outcome
        bytes.push(self.outcome.canonical_byte());

        // 7. refuse_reasons
        bytes.extend_from_slice(&(self.refuse_reasons.len() as u32).to_le_bytes());
        for reason in &self.refuse_reasons {
            bytes.extend_from_slice(&reason.canonical_bytes());
        }

        // 8. correlation_id (always present, no Option tag)
        write_string(&mut bytes, &self.correlation_id);

        // 9. strategies_manifest_hash
        bytes.extend_from_slice(&self.strategies_manifest_hash);

        // 10. signals_manifest_hash
        bytes.extend_from_slice(&self.signals_manifest_hash);

        bytes
    }

    /// Compute SHA-256 digest of canonical bytes.
    pub fn compute_digest(&self) -> String {
        let bytes = self.canonical_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Create a new decision with computed digest.
    pub fn new(
        ts_ns: i64,
        session_id: String,
        strategy_id: String,
        signal_id: String,
        outcome: StrategyAdmissionOutcome,
        refuse_reasons: Vec<StrategyRefuseReason>,
        correlation_id: String,
        strategies_manifest_hash: [u8; 32],
        signals_manifest_hash: [u8; 32],
    ) -> Self {
        let mut decision = Self {
            schema_version: STRATEGY_ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns,
            session_id,
            strategy_id,
            signal_id,
            outcome,
            refuse_reasons,
            correlation_id,
            strategies_manifest_hash,
            signals_manifest_hash,
            digest: String::new(),
        };
        decision.digest = decision.compute_digest();
        decision
    }

    /// Create an Admit decision.
    pub fn admit(
        ts_ns: i64,
        session_id: String,
        strategy_id: String,
        signal_id: String,
        correlation_id: String,
        strategies_manifest_hash: [u8; 32],
        signals_manifest_hash: [u8; 32],
    ) -> Self {
        Self::new(
            ts_ns,
            session_id,
            strategy_id,
            signal_id,
            StrategyAdmissionOutcome::Admit,
            vec![],
            correlation_id,
            strategies_manifest_hash,
            signals_manifest_hash,
        )
    }

    /// Create a Refuse decision.
    pub fn refuse(
        ts_ns: i64,
        session_id: String,
        strategy_id: String,
        signal_id: String,
        refuse_reasons: Vec<StrategyRefuseReason>,
        correlation_id: String,
        strategies_manifest_hash: [u8; 32],
        signals_manifest_hash: [u8; 32],
    ) -> Self {
        Self::new(
            ts_ns,
            session_id,
            strategy_id,
            signal_id,
            StrategyAdmissionOutcome::Refuse,
            refuse_reasons,
            correlation_id,
            strategies_manifest_hash,
            signals_manifest_hash,
        )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Write a string with length prefix (u32 LE len + UTF-8 bytes).
fn write_string(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_canonical_bytes() {
        assert_eq!(StrategyAdmissionOutcome::Admit.canonical_byte(), 0x01);
        assert_eq!(StrategyAdmissionOutcome::Refuse.canonical_byte(), 0x02);
    }

    #[test]
    fn test_refuse_reason_canonical_bytes() {
        let reason1 = StrategyRefuseReason::SignalNotAdmitted {
            signal_id: "spread".to_string(),
        };
        let bytes1 = reason1.canonical_bytes();
        assert_eq!(bytes1[0], 0x01); // SignalNotAdmitted tag

        let reason2 = StrategyRefuseReason::StrategyNotFound {
            strategy_id: "test".to_string(),
        };
        let bytes2 = reason2.canonical_bytes();
        assert_eq!(bytes2[0], 0x02); // StrategyNotFound tag

        let reason3 = StrategyRefuseReason::SignalNotBound {
            signal_id: "spread".to_string(),
            strategy_id: "test".to_string(),
        };
        let bytes3 = reason3.canonical_bytes();
        assert_eq!(bytes3[0], 0x03); // SignalNotBound tag
    }

    #[test]
    fn test_decision_digest_deterministic() {
        let decision1 = StrategyAdmissionDecision::admit(
            1000000000,
            "session_1".to_string(),
            "spread_passive".to_string(),
            "spread".to_string(),
            "corr_123".to_string(),
            [1u8; 32],
            [2u8; 32],
        );

        let decision2 = StrategyAdmissionDecision::admit(
            1000000000,
            "session_1".to_string(),
            "spread_passive".to_string(),
            "spread".to_string(),
            "corr_123".to_string(),
            [1u8; 32],
            [2u8; 32],
        );

        assert_eq!(decision1.digest, decision2.digest);

        // Run 100 times to prove determinism
        for _ in 0..100 {
            let d = StrategyAdmissionDecision::admit(
                1000000000,
                "session_1".to_string(),
                "spread_passive".to_string(),
                "spread".to_string(),
                "corr_123".to_string(),
                [1u8; 32],
                [2u8; 32],
            );
            assert_eq!(d.digest, decision1.digest);
        }
    }

    #[test]
    fn test_decision_digest_changes_with_content() {
        let decision1 = StrategyAdmissionDecision::admit(
            1000000000,
            "session_1".to_string(),
            "spread_passive".to_string(),
            "spread".to_string(),
            "corr_123".to_string(),
            [1u8; 32],
            [2u8; 32],
        );

        // Change timestamp
        let decision2 = StrategyAdmissionDecision::admit(
            1000000001,
            "session_1".to_string(),
            "spread_passive".to_string(),
            "spread".to_string(),
            "corr_123".to_string(),
            [1u8; 32],
            [2u8; 32],
        );

        assert_ne!(decision1.digest, decision2.digest);

        // Change outcome
        let decision3 = StrategyAdmissionDecision::refuse(
            1000000000,
            "session_1".to_string(),
            "spread_passive".to_string(),
            "spread".to_string(),
            vec![StrategyRefuseReason::SignalNotAdmitted {
                signal_id: "spread".to_string(),
            }],
            "corr_123".to_string(),
            [1u8; 32],
            [2u8; 32],
        );

        assert_ne!(decision1.digest, decision3.digest);
    }

    #[test]
    fn test_admit_refuse_helpers() {
        let admit = StrategyAdmissionDecision::admit(
            1000000000,
            "session".to_string(),
            "strategy".to_string(),
            "signal".to_string(),
            "corr".to_string(),
            [0u8; 32],
            [0u8; 32],
        );
        assert!(admit.is_admitted());
        assert!(!admit.is_refused());
        assert!(admit.refuse_reasons.is_empty());

        let refuse = StrategyAdmissionDecision::refuse(
            1000000000,
            "session".to_string(),
            "strategy".to_string(),
            "signal".to_string(),
            vec![StrategyRefuseReason::StrategyNotFound {
                strategy_id: "strategy".to_string(),
            }],
            "corr".to_string(),
            [0u8; 32],
            [0u8; 32],
        );
        assert!(!refuse.is_admitted());
        assert!(refuse.is_refused());
        assert_eq!(refuse.refuse_reasons.len(), 1);
    }

    #[test]
    fn test_refuse_reason_descriptions() {
        let r1 = StrategyRefuseReason::SignalNotAdmitted {
            signal_id: "spread".to_string(),
        };
        assert!(r1.description().contains("spread"));
        assert!(r1.description().contains("L1"));

        let r2 = StrategyRefuseReason::StrategyNotFound {
            strategy_id: "test".to_string(),
        };
        assert!(r2.description().contains("test"));
        assert!(r2.description().contains("not found"));

        let r3 = StrategyRefuseReason::SignalNotBound {
            signal_id: "spread".to_string(),
            strategy_id: "test".to_string(),
        };
        assert!(r3.description().contains("spread"));
        assert!(r3.description().contains("test"));
        assert!(r3.description().contains("not bound"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let decision = StrategyAdmissionDecision::refuse(
            1000000000,
            "session_1".to_string(),
            "spread_passive".to_string(),
            "spread".to_string(),
            vec![StrategyRefuseReason::SignalNotBound {
                signal_id: "spread".to_string(),
                strategy_id: "other_strategy".to_string(),
            }],
            "corr_123".to_string(),
            [1u8; 32],
            [2u8; 32],
        );

        let json = serde_json::to_string(&decision).unwrap();
        let parsed: StrategyAdmissionDecision = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.schema_version, decision.schema_version);
        assert_eq!(parsed.ts_ns, decision.ts_ns);
        assert_eq!(parsed.session_id, decision.session_id);
        assert_eq!(parsed.strategy_id, decision.strategy_id);
        assert_eq!(parsed.signal_id, decision.signal_id);
        assert_eq!(parsed.outcome, decision.outcome);
        assert_eq!(parsed.refuse_reasons.len(), 1);
        assert_eq!(parsed.correlation_id, decision.correlation_id);
        assert_eq!(
            parsed.strategies_manifest_hash,
            decision.strategies_manifest_hash
        );
        assert_eq!(parsed.signals_manifest_hash, decision.signals_manifest_hash);
        assert_eq!(parsed.digest, decision.digest);
    }

    #[test]
    fn test_schema_version() {
        let decision = StrategyAdmissionDecision::admit(
            1000000000,
            "session".to_string(),
            "strategy".to_string(),
            "signal".to_string(),
            "corr".to_string(),
            [0u8; 32],
            [0u8; 32],
        );
        assert_eq!(decision.schema_version, "1.0.0");
    }
}
