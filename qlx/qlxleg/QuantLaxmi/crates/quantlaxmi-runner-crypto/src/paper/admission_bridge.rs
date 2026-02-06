//! Bridge from your real Phase 18/19 admission decision types into UiSnapshot.
//!
//! IMPORTANT:
//! This file is the intended *only* place you should need to edit when you
//! plug in your actual QuantLaxmi admission decision structs.

use crate::paper::state::{DecisionMetrics, LastDecision, R3Eligibility, RefusalReason};

/// Minimal "normalized" admission decision that the runner/TUI understand.
/// Map your internal types into this.
#[derive(Debug, Clone, Default)]
pub struct NormalizedAdmission {
    pub eligibility: R3Eligibility,
    pub refusal_reasons: Vec<RefusalReason>,
    pub metrics: DecisionMetrics,
    /// Regime name (R0, R1, R2, R3) for sniper gating
    pub regime: String,
}

impl NormalizedAdmission {
    #[allow(clippy::too_many_arguments)]
    pub fn eligible(
        confidence: Option<f64>,
        d_perp: Option<f64>,
        fragility: Option<f64>,
        toxicity: Option<f64>,
        toxicity_persist: Option<f64>,
        fti_level: Option<f64>,
        fti_persist: Option<f64>,
        fti_thresh: Option<f64>,
        fti_elevated: Option<bool>,
        fti_calibrated: Option<bool>,
        regime: impl Into<String>,
    ) -> Self {
        Self {
            eligibility: R3Eligibility::Eligible,
            refusal_reasons: vec![],
            metrics: DecisionMetrics {
                confidence,
                d_perp,
                fragility,
                toxicity,
                toxicity_persist,
                fti_level,
                fti_persist,
                fti_thresh,
                fti_elevated,
                fti_calibrated,
            },
            regime: regime.into(),
        }
    }

    pub fn refused(
        reasons: Vec<RefusalReason>,
        metrics: DecisionMetrics,
        regime: impl Into<String>,
    ) -> Self {
        Self {
            eligibility: R3Eligibility::Refused,
            refusal_reasons: reasons,
            metrics,
            regime: regime.into(),
        }
    }
}

/// Convert a normalized admission into a LastDecision the UI can render.
pub fn to_last_decision(adm: NormalizedAdmission, decided_at_unix_ms: u64) -> LastDecision {
    LastDecision {
        eligibility: adm.eligibility,
        refusal_reasons: adm.refusal_reasons,
        metrics: adm.metrics,
        decided_at_unix_ms,
    }
}

/// Helper to build a common refusal reason quickly.
pub fn rr(code: &'static str, detail: impl Into<String>) -> RefusalReason {
    RefusalReason::new(code, detail)
}
