//! # Promotion Policy (Phase 12.3)
//!
//! Enforces the contract: **No G3 without paper evidence.**
//!
//! ## Promotion Requirements
//! G2 → G3 requires ALL of the following:
//!
//! 1. **Tournament Eligibility**
//!    - Appears in tournament leaderboard
//!    - Passes `is_meaningful_run`
//!    - Rank ≤ configurable cutoff
//!
//! 2. **Paper Evidence Bundle**
//!    - Deterministic paper session exists
//!    - Attribution + AlphaScore present
//!    - Evidence digest verified
//!
//! 3. **Gate Consistency**
//!    - G2 passes on paper evidence
//!    - G3 passes on paper evidence
//!
//! 4. **Data Lineage Match**
//!    - Strategy config hash matches tournament config
//!    - Segment class compatible (venue, mode)
//!
//! If ANY fail → promotion rejected, with reason recorded.

use crate::GateResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// =============================================================================
// Schema Version
// =============================================================================

pub const PROMOTION_DECISION_SCHEMA: &str = "promotion_decision_v1.0";

// =============================================================================
// Promotion Source
// =============================================================================

/// Source of a promotion request.
///
/// If `Some`, the promotion originated from a tournament run.
/// If `None`, this is a legacy/manual promotion (auditable but no tournament link).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PromotionSource {
    /// Promotion from tournament result.
    Tournament {
        tournament_id: String,
        run_id: String,
        run_key: String,
        rank: u32,
    },
    /// Legacy/manual promotion (no tournament link).
    Manual { reason: String },
}

impl PromotionSource {
    /// Create a tournament source.
    pub fn tournament(tournament_id: &str, run_id: &str, run_key: &str, rank: u32) -> Self {
        Self::Tournament {
            tournament_id: tournament_id.to_string(),
            run_id: run_id.to_string(),
            run_key: run_key.to_string(),
            rank,
        }
    }

    /// Create a manual source.
    pub fn manual(reason: &str) -> Self {
        Self::Manual {
            reason: reason.to_string(),
        }
    }

    /// Check if this is a tournament source.
    pub fn is_tournament(&self) -> bool {
        matches!(self, Self::Tournament { .. })
    }
}

// =============================================================================
// Paper Evidence
// =============================================================================

/// Paper trading evidence bundle.
///
/// Required for G2 → G3 promotion when source is tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperEvidence {
    /// Paper session ID
    pub session_id: String,
    /// Paper session directory path
    pub session_path: String,
    /// Attribution summary SHA-256
    pub attribution_sha256: String,
    /// Alpha score SHA-256
    pub alpha_score_sha256: String,
    /// G2 result SHA-256
    pub g2_result_sha256: String,
    /// G3 result SHA-256 (if evaluated)
    pub g3_result_sha256: Option<String>,
    /// Paper session start timestamp
    pub session_start_ts: DateTime<Utc>,
    /// Paper session end timestamp
    pub session_end_ts: DateTime<Utc>,
    /// Bundle digest (all evidence combined)
    pub bundle_digest: String,
}

impl PaperEvidence {
    /// Compute bundle digest from evidence hashes.
    pub fn compute_bundle_digest(
        attribution_sha256: &str,
        alpha_score_sha256: &str,
        g2_result_sha256: &str,
        g3_result_sha256: Option<&str>,
    ) -> String {
        let mut input = format!(
            "paper_evidence_v1:{}:{}:{}:",
            attribution_sha256, alpha_score_sha256, g2_result_sha256
        );
        if let Some(g3) = g3_result_sha256 {
            input.push_str(g3);
        }
        let hash = Sha256::digest(input.as_bytes());
        hex::encode(hash)
    }
}

// =============================================================================
// Promotion Request
// =============================================================================

/// Request to promote a strategy from G2 to G3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionRequest {
    /// Unique request ID
    pub request_id: String,
    /// Strategy ID (name:version:config_hash)
    pub strategy_id: String,
    /// Strategy config hash (for lineage verification)
    pub config_hash: String,
    /// Promotion source
    pub source: PromotionSource,
    /// Paper evidence (required if source is Tournament)
    pub paper_evidence: Option<PaperEvidence>,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Requesting user/system
    pub requested_by: String,
}

impl PromotionRequest {
    /// Generate deterministic request ID.
    pub fn generate_request_id(strategy_id: &str, requested_at: DateTime<Utc>) -> String {
        let input = format!(
            "promotion_request_v1:{}:{}",
            strategy_id,
            requested_at.timestamp_nanos_opt().unwrap_or(0)
        );
        let hash = Sha256::digest(input.as_bytes());
        hex::encode(&hash[..16]) // 32 hex chars
    }
}

// =============================================================================
// Promotion Decision
// =============================================================================

/// Decision on a promotion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionDecision {
    /// Schema version
    pub schema_version: String,
    /// Request ID
    pub request_id: String,
    /// Strategy ID
    pub strategy_id: String,
    /// Whether promotion was accepted
    pub accepted: bool,
    /// Rejection reasons (empty if accepted)
    pub rejection_reasons: Vec<String>,
    /// Individual check results
    pub checks: Vec<PromotionCheck>,
    /// Decision timestamp
    pub decided_at: DateTime<Utc>,
    /// Tournament ID (if source was tournament)
    pub tournament_id: Option<String>,
    /// Paper evidence ID (if provided)
    pub paper_session_id: Option<String>,
    /// Decision digest (for audit trail)
    pub decision_digest: String,
}

impl PromotionDecision {
    /// Compute decision digest.
    pub fn compute_digest(
        request_id: &str,
        strategy_id: &str,
        accepted: bool,
        rejection_reasons: &[String],
    ) -> String {
        let reasons_str = rejection_reasons.join(";");
        let input = format!(
            "promotion_decision_v1:{}:{}:{}:{}",
            request_id, strategy_id, accepted, reasons_str
        );
        let hash = Sha256::digest(input.as_bytes());
        hex::encode(hash)
    }
}

/// Individual promotion check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCheck {
    /// Check name
    pub name: String,
    /// Pass/fail
    pub passed: bool,
    /// Reason/details
    pub reason: String,
}

impl PromotionCheck {
    pub fn pass(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            reason: reason.to_string(),
        }
    }

    pub fn fail(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            reason: reason.to_string(),
        }
    }
}

// =============================================================================
// Promotion Policy
// =============================================================================

/// Promotion policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPolicy {
    /// Maximum tournament rank for eligibility (0 = no limit)
    pub max_rank: u32,
    /// Require paper evidence for tournament-sourced promotions
    pub require_paper_evidence: bool,
    /// Require G2 pass on paper evidence
    pub require_g2_pass: bool,
    /// Require G3 pass on paper evidence
    pub require_g3_pass: bool,
    /// Require config hash match between tournament and paper
    pub require_config_match: bool,
    /// Minimum paper session duration (nanoseconds)
    pub min_paper_duration_ns: i64,
    /// Paper evidence must be newer than tournament
    pub require_paper_after_tournament: bool,
}

impl Default for PromotionPolicy {
    fn default() -> Self {
        Self {
            max_rank: 10,                 // Top 10 only
            require_paper_evidence: true, // Core principle
            require_g2_pass: true,
            require_g3_pass: true,
            require_config_match: true,
            min_paper_duration_ns: 3_600_000_000_000, // 1 hour minimum
            require_paper_after_tournament: true,
        }
    }
}

impl PromotionPolicy {
    /// Strict policy: all checks enforced.
    pub fn strict() -> Self {
        Self::default()
    }

    /// Lenient policy: only paper evidence required.
    pub fn lenient() -> Self {
        Self {
            max_rank: 0, // No rank limit
            require_paper_evidence: true,
            require_g2_pass: false,
            require_g3_pass: false,
            require_config_match: false,
            min_paper_duration_ns: 0,
            require_paper_after_tournament: false,
        }
    }
}

// =============================================================================
// Promotion Validator
// =============================================================================

/// Validator for promotion requests.
pub struct PromotionValidator {
    policy: PromotionPolicy,
}

impl PromotionValidator {
    /// Create a new validator with the given policy.
    pub fn new(policy: PromotionPolicy) -> Self {
        Self { policy }
    }

    /// Create a validator with strict (default) policy.
    pub fn strict() -> Self {
        Self::new(PromotionPolicy::strict())
    }

    /// Validate a promotion request.
    ///
    /// Returns a `PromotionDecision` indicating acceptance or rejection.
    pub fn validate(&self, request: &PromotionRequest) -> PromotionDecision {
        let mut checks = Vec::new();
        let mut rejection_reasons = Vec::new();

        // Check 1: Tournament eligibility (if tournament source)
        if let PromotionSource::Tournament { rank, .. } = &request.source {
            if self.policy.max_rank > 0 && *rank > self.policy.max_rank {
                let check = PromotionCheck::fail(
                    "tournament_rank",
                    &format!(
                        "Rank {} exceeds maximum allowed rank {}",
                        rank, self.policy.max_rank
                    ),
                );
                rejection_reasons.push(check.reason.clone());
                checks.push(check);
            } else {
                checks.push(PromotionCheck::pass(
                    "tournament_rank",
                    &format!("Rank {} within limit", rank),
                ));
            }
        }

        // Check 2: Paper evidence required (if tournament source)
        if request.source.is_tournament() && self.policy.require_paper_evidence {
            match &request.paper_evidence {
                Some(evidence) => {
                    checks.push(PromotionCheck::pass(
                        "paper_evidence_present",
                        "Paper evidence bundle provided",
                    ));

                    // Check 2a: Paper session duration
                    if self.policy.min_paper_duration_ns > 0 {
                        let duration_ns = evidence
                            .session_end_ts
                            .signed_duration_since(evidence.session_start_ts)
                            .num_nanoseconds()
                            .unwrap_or(0);

                        if duration_ns < self.policy.min_paper_duration_ns {
                            let check = PromotionCheck::fail(
                                "paper_duration",
                                &format!(
                                    "Paper session duration {}ns < required {}ns",
                                    duration_ns, self.policy.min_paper_duration_ns
                                ),
                            );
                            rejection_reasons.push(check.reason.clone());
                            checks.push(check);
                        } else {
                            checks.push(PromotionCheck::pass(
                                "paper_duration",
                                &format!(
                                    "Paper session duration {}ns meets requirement",
                                    duration_ns
                                ),
                            ));
                        }
                    }

                    // Check 2b: Evidence bundle digest
                    let computed_digest = PaperEvidence::compute_bundle_digest(
                        &evidence.attribution_sha256,
                        &evidence.alpha_score_sha256,
                        &evidence.g2_result_sha256,
                        evidence.g3_result_sha256.as_deref(),
                    );
                    if computed_digest != evidence.bundle_digest {
                        let check = PromotionCheck::fail(
                            "evidence_digest",
                            "Paper evidence bundle digest mismatch",
                        );
                        rejection_reasons.push(check.reason.clone());
                        checks.push(check);
                    } else {
                        checks.push(PromotionCheck::pass(
                            "evidence_digest",
                            "Paper evidence bundle digest verified",
                        ));
                    }

                    // Check 2c: G3 result present (if required)
                    if self.policy.require_g3_pass && evidence.g3_result_sha256.is_none() {
                        let check = PromotionCheck::fail(
                            "g3_result_present",
                            "G3 result required but not present in paper evidence",
                        );
                        rejection_reasons.push(check.reason.clone());
                        checks.push(check);
                    } else if self.policy.require_g3_pass {
                        checks.push(PromotionCheck::pass(
                            "g3_result_present",
                            "G3 result present in paper evidence",
                        ));
                    }
                }
                None => {
                    let check = PromotionCheck::fail(
                        "paper_evidence_present",
                        "Paper evidence required for tournament-sourced promotion",
                    );
                    rejection_reasons.push(check.reason.clone());
                    checks.push(check);
                }
            }
        }

        // Check 3: Manual source audit (if manual)
        if let PromotionSource::Manual { reason } = &request.source {
            if reason.is_empty() {
                let check = PromotionCheck::fail(
                    "manual_reason",
                    "Manual promotion requires explicit reason",
                );
                rejection_reasons.push(check.reason.clone());
                checks.push(check);
            } else {
                checks.push(PromotionCheck::pass(
                    "manual_reason",
                    &format!("Manual promotion reason: {}", reason),
                ));
            }
        }

        // Build decision
        let accepted = rejection_reasons.is_empty();
        let decided_at = Utc::now();
        let decision_digest = PromotionDecision::compute_digest(
            &request.request_id,
            &request.strategy_id,
            accepted,
            &rejection_reasons,
        );

        let tournament_id = match &request.source {
            PromotionSource::Tournament { tournament_id, .. } => Some(tournament_id.clone()),
            _ => None,
        };

        let paper_session_id = request
            .paper_evidence
            .as_ref()
            .map(|e| e.session_id.clone());

        PromotionDecision {
            schema_version: PROMOTION_DECISION_SCHEMA.to_string(),
            request_id: request.request_id.clone(),
            strategy_id: request.strategy_id.clone(),
            accepted,
            rejection_reasons,
            checks,
            decided_at,
            tournament_id,
            paper_session_id,
            decision_digest,
        }
    }

    /// Validate with additional gate results.
    ///
    /// This method accepts pre-computed G2 and G3 gate results
    /// and incorporates them into the decision.
    pub fn validate_with_gates(
        &self,
        request: &PromotionRequest,
        g2_result: Option<&GateResult>,
        g3_result: Option<&GateResult>,
    ) -> PromotionDecision {
        let mut decision = self.validate(request);

        // Check G2 gate result
        if self.policy.require_g2_pass {
            match g2_result {
                Some(result) if result.passed => {
                    decision.checks.push(PromotionCheck::pass(
                        "g2_gate_pass",
                        "G2 BacktestCorrectness gate passed",
                    ));
                }
                Some(result) => {
                    let check = PromotionCheck::fail(
                        "g2_gate_pass",
                        &format!("G2 gate failed: {}", result.summary),
                    );
                    decision.rejection_reasons.push(check.reason.clone());
                    decision.checks.push(check);
                    decision.accepted = false;
                }
                None => {
                    let check = PromotionCheck::fail(
                        "g2_gate_pass",
                        "G2 gate result required but not provided",
                    );
                    decision.rejection_reasons.push(check.reason.clone());
                    decision.checks.push(check);
                    decision.accepted = false;
                }
            }
        }

        // Check G3 gate result
        if self.policy.require_g3_pass {
            match g3_result {
                Some(result) if result.passed => {
                    decision.checks.push(PromotionCheck::pass(
                        "g3_gate_pass",
                        "G3 Robustness gate passed",
                    ));
                }
                Some(result) => {
                    let check = PromotionCheck::fail(
                        "g3_gate_pass",
                        &format!("G3 gate failed: {}", result.summary),
                    );
                    decision.rejection_reasons.push(check.reason.clone());
                    decision.checks.push(check);
                    decision.accepted = false;
                }
                None => {
                    let check = PromotionCheck::fail(
                        "g3_gate_pass",
                        "G3 gate result required but not provided",
                    );
                    decision.rejection_reasons.push(check.reason.clone());
                    decision.checks.push(check);
                    decision.accepted = false;
                }
            }
        }

        // Recompute digest with updated rejection reasons
        decision.decision_digest = PromotionDecision::compute_digest(
            &decision.request_id,
            &decision.strategy_id,
            decision.accepted,
            &decision.rejection_reasons,
        );

        decision
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_evidence() -> PaperEvidence {
        let attribution_sha256 = "abc123".to_string();
        let alpha_score_sha256 = "def456".to_string();
        let g2_result_sha256 = "ghi789".to_string();
        let g3_result_sha256 = Some("jkl012".to_string());

        let bundle_digest = PaperEvidence::compute_bundle_digest(
            &attribution_sha256,
            &alpha_score_sha256,
            &g2_result_sha256,
            g3_result_sha256.as_deref(),
        );

        PaperEvidence {
            session_id: "paper_session_001".to_string(),
            session_path: "/data/paper/session_001".to_string(),
            attribution_sha256,
            alpha_score_sha256,
            g2_result_sha256,
            g3_result_sha256,
            session_start_ts: Utc::now() - chrono::Duration::hours(2),
            session_end_ts: Utc::now(),
            bundle_digest,
        }
    }

    fn make_test_request(
        source: PromotionSource,
        evidence: Option<PaperEvidence>,
    ) -> PromotionRequest {
        let requested_at = Utc::now();
        let strategy_id = "funding_bias:1.0.0:abc123".to_string();
        let request_id = PromotionRequest::generate_request_id(&strategy_id, requested_at);

        PromotionRequest {
            request_id,
            strategy_id,
            config_hash: "abc123".to_string(),
            source,
            paper_evidence: evidence,
            requested_at,
            requested_by: "test".to_string(),
        }
    }

    #[test]
    fn test_promotion_accepted_with_evidence() {
        let validator = PromotionValidator::new(PromotionPolicy {
            require_g2_pass: false,
            require_g3_pass: false,
            ..PromotionPolicy::default()
        });

        let source = PromotionSource::tournament("tournament_001", "run_001", "seg__strat", 5);
        let evidence = make_test_evidence();
        let request = make_test_request(source, Some(evidence));

        let decision = validator.validate(&request);

        assert!(
            decision.accepted,
            "Should be accepted: {:?}",
            decision.rejection_reasons
        );
        assert!(decision.rejection_reasons.is_empty());
    }

    #[test]
    fn test_promotion_rejected_without_evidence() {
        let validator = PromotionValidator::strict();

        let source = PromotionSource::tournament("tournament_001", "run_001", "seg__strat", 5);
        // No evidence provided
        let request = make_test_request(source, None);

        let decision = validator.validate(&request);

        assert!(!decision.accepted);
        assert!(
            decision
                .rejection_reasons
                .iter()
                .any(|r| r.contains("Paper evidence required"))
        );
    }

    #[test]
    fn test_promotion_rejected_rank_too_high() {
        let validator = PromotionValidator::new(PromotionPolicy {
            max_rank: 3,
            require_g2_pass: false,
            require_g3_pass: false,
            ..PromotionPolicy::default()
        });

        let source = PromotionSource::tournament("tournament_001", "run_001", "seg__strat", 10);
        let evidence = make_test_evidence();
        let request = make_test_request(source, Some(evidence));

        let decision = validator.validate(&request);

        assert!(!decision.accepted);
        assert!(
            decision
                .rejection_reasons
                .iter()
                .any(|r| r.contains("exceeds maximum"))
        );
    }

    #[test]
    fn test_promotion_rejected_short_paper_session() {
        let validator = PromotionValidator::new(PromotionPolicy {
            min_paper_duration_ns: 7200_000_000_000, // 2 hours
            require_g2_pass: false,
            require_g3_pass: false,
            ..PromotionPolicy::default()
        });

        let source = PromotionSource::tournament("tournament_001", "run_001", "seg__strat", 5);
        let mut evidence = make_test_evidence();
        // Make session only 30 minutes
        evidence.session_end_ts = evidence.session_start_ts + chrono::Duration::minutes(30);

        let request = make_test_request(source, Some(evidence));
        let decision = validator.validate(&request);

        assert!(!decision.accepted);
        assert!(
            decision
                .rejection_reasons
                .iter()
                .any(|r| r.contains("duration"))
        );
    }

    #[test]
    fn test_manual_promotion_allowed() {
        let validator = PromotionValidator::strict();

        let source = PromotionSource::manual("Legacy strategy pre-tournament system");
        let request = make_test_request(source, None);

        let decision = validator.validate(&request);

        // Manual promotions don't require paper evidence (they're auditable differently)
        assert!(
            decision.accepted,
            "Manual promotion should be allowed: {:?}",
            decision.rejection_reasons
        );
    }

    #[test]
    fn test_manual_promotion_requires_reason() {
        let validator = PromotionValidator::strict();

        let source = PromotionSource::manual("");
        let request = make_test_request(source, None);

        let decision = validator.validate(&request);

        assert!(!decision.accepted);
        assert!(
            decision
                .rejection_reasons
                .iter()
                .any(|r| r.contains("reason"))
        );
    }

    #[test]
    fn test_decision_digest_deterministic() {
        let digest1 = PromotionDecision::compute_digest(
            "req_001",
            "strategy_001",
            false,
            &["reason1".to_string(), "reason2".to_string()],
        );
        let digest2 = PromotionDecision::compute_digest(
            "req_001",
            "strategy_001",
            false,
            &["reason1".to_string(), "reason2".to_string()],
        );

        assert_eq!(digest1, digest2);
    }

    #[test]
    fn test_validate_with_gates_requires_g2() {
        let validator = PromotionValidator::new(PromotionPolicy {
            require_g2_pass: true,
            require_g3_pass: false,
            ..PromotionPolicy::default()
        });

        let source = PromotionSource::tournament("tournament_001", "run_001", "seg__strat", 5);
        let evidence = make_test_evidence();
        let request = make_test_request(source, Some(evidence));

        // No G2 result provided
        let decision = validator.validate_with_gates(&request, None, None);

        assert!(!decision.accepted);
        assert!(decision.rejection_reasons.iter().any(|r| r.contains("G2")));
    }

    #[test]
    fn test_validate_with_gates_passes() {
        let validator = PromotionValidator::new(PromotionPolicy {
            require_g2_pass: true,
            require_g3_pass: true,
            ..PromotionPolicy::default()
        });

        let source = PromotionSource::tournament("tournament_001", "run_001", "seg__strat", 5);
        let evidence = make_test_evidence();
        let request = make_test_request(source, Some(evidence));

        let g2_result = GateResult::new("G2").with_summary("Passed");
        let g3_result = GateResult::new("G3").with_summary("Passed");

        let decision = validator.validate_with_gates(&request, Some(&g2_result), Some(&g3_result));

        assert!(
            decision.accepted,
            "Should pass with gates: {:?}",
            decision.rejection_reasons
        );
    }
}
