//! SANOS Calendar Carry TUI State
//!
//! Diagnostic state broadcast to the TUI via a separate `watch` channel.
//! Keeps SANOS-specific details out of PaperState.

use serde::{Deserialize, Serialize};

/// Top-level TUI diagnostic state for SANOS calendar carry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SanosTuiState {
    /// SANOS surface diagnostics per expiry.
    pub surfaces: Vec<SanosSurfaceView>,
    /// Gate statuses (12 strategy gates + margin).
    pub gates: Vec<SanosGateView>,
    /// IV term structure features.
    pub features: Option<SanosFeatureView>,
    /// Last calibration timestamp (ISO string).
    pub last_calibration_ts: Option<String>,
    /// Seconds since last calibration.
    pub secs_since_calibration: Option<f64>,
    /// Whether SANOS is in warmup (no calibration yet).
    pub warmup: bool,
    /// Last decision description.
    pub last_decision: Option<String>,
}

/// SANOS surface quality for a single expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanosSurfaceView {
    /// Expiry label (e.g., "2026-02-06").
    pub expiry: String,
    /// LP solver status ("Optimal", "Infeasible", etc.).
    pub lp_status: String,
    /// Maximum fit error (normalized call price units).
    pub max_fit_error: f64,
    /// Mean fit error.
    pub mean_fit_error: f64,
    /// Spread compliance (fraction of prices within bid-ask).
    pub spread_compliance: f64,
    /// Forward price.
    pub forward: f64,
    /// Time to expiry in years.
    pub time_to_expiry: f64,
    /// Background variance used in calibration.
    pub background_variance: f64,
}

/// Gate status for TUI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanosGateView {
    /// Gate name (e.g., "H1_SURFACE", "E3_FRICTION_FLOOR").
    pub name: String,
    /// Whether the gate passed.
    pub passed: bool,
    /// Current value (if applicable).
    pub value: Option<f64>,
    /// Threshold (if applicable).
    pub threshold: Option<f64>,
    /// Reason string on failure.
    pub reason: Option<String>,
}

/// IV term structure features for TUI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanosFeatureView {
    /// ATM IV for T1 (front).
    pub iv1: f64,
    /// ATM IV for T2 (back).
    pub iv2: Option<f64>,
    /// ATM IV for T3 (monthly).
    pub iv3: Option<f64>,
    /// Calendar gap T1→T2 (normalized call price).
    pub cal12: Option<f64>,
    /// Calendar gap T2→T3.
    pub cal23: Option<f64>,
    /// Term structure slope T1→T2.
    pub ts12: Option<f64>,
    /// Term structure slope T2→T3.
    pub ts23: Option<f64>,
    /// Term structure curvature.
    pub ts_curv: Option<f64>,
    /// Skew T1.
    pub sk1: Option<f64>,
    /// Skew T2.
    pub sk2: Option<f64>,
    /// Forward T1.
    pub f1: f64,
    /// Forward T2.
    pub f2: Option<f64>,
    /// Time to expiry T1.
    pub tty1: f64,
    /// Time to expiry T2.
    pub tty2: Option<f64>,
}
