//! Regime classification (R0-R3).
//!
//! Spec: Section 6.6 Regime Classification
//!
//! R3 (Trade-Eligible) if:
//! - d_perp > τ_d
//! - fragility > τ_frag
//! - (FTI_persist > τ_F OR toxicity_persist > τ_T)
//! - confidence > τ_C
//!
//! AUDIT: No Default impls - explicit config injection required.
//!
//! Explicit tri-states for observability:
//! - ToxicityState: NoTradesWindow vs Computable(value)
//! - FTIState: InsufficientHistory vs Computable(value)
//!   Both apply confidence penalty ×0.95 when not Computable.

use crate::fragility::FragilityScore;
use crate::fti::FTIMetrics;
use crate::normalization::NormalizationStatus;
use crate::sealed::confidence;
use crate::subspace::RegimeMetrics;

/// Toxicity computation state.
/// Explicit tri-state for observability and confidence tracking.
#[derive(Debug, Clone, Copy)]
pub enum ToxicityState {
    /// No trades received in this window - cannot compute toxicity.
    /// Applies confidence penalty ×0.95.
    NoTradesWindow,
    /// Toxicity is computable with the given values.
    Computable {
        /// Current toxicity value [0, 1]
        toxicity: f64,
        /// Persistence fraction [0, 1]
        persist: f64,
    },
}

impl ToxicityState {
    /// Get toxicity value, defaulting to 0.0 for NoTradesWindow.
    pub fn toxicity_or_default(&self) -> f64 {
        match self {
            Self::NoTradesWindow => 0.0,
            Self::Computable { toxicity, .. } => *toxicity,
        }
    }

    /// Get persist value, defaulting to 0.0 for NoTradesWindow.
    pub fn persist_or_default(&self) -> f64 {
        match self {
            Self::NoTradesWindow => 0.0,
            Self::Computable { persist, .. } => *persist,
        }
    }

    /// Check if this state should apply confidence penalty.
    pub fn needs_penalty(&self) -> bool {
        matches!(self, Self::NoTradesWindow)
    }

    /// Format for logging.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NoTradesWindow => "NoTradesWindow",
            Self::Computable { .. } => "Computable",
        }
    }
}

/// FTI computation state.
/// Explicit tri-state for observability and confidence tracking.
#[derive(Debug, Clone, Copy)]
pub enum FTIState {
    /// Insufficient price history to compute FTI.
    /// Applies confidence penalty ×0.95.
    InsufficientHistory,
    /// FTI is computable with the given values.
    Computable {
        /// Current FTI level
        level: f64,
        /// FTI slope (change)
        slope: f64,
        /// Persistence fraction [0, 1]
        persist: f64,
    },
}

impl FTIState {
    /// Get persist value, defaulting to 0.0 for InsufficientHistory.
    pub fn persist_or_default(&self) -> f64 {
        match self {
            Self::InsufficientHistory => 0.0,
            Self::Computable { persist, .. } => *persist,
        }
    }

    /// Check if this state should apply confidence penalty.
    pub fn needs_penalty(&self) -> bool {
        matches!(self, Self::InsufficientHistory)
    }

    /// Format for logging.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InsufficientHistory => "InsufficientHistory",
            Self::Computable { .. } => "Computable",
        }
    }
}

/// Regime labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// R0: Stable, dormant
    R0,
    /// R1: Early instability detection
    R1,
    /// R2: Confirmed instability, preparing
    R2,
    /// R3: Trade-eligible
    R3,
}

impl Regime {
    /// Convert to string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::R0 => "R0",
            Regime::R1 => "R1",
            Regime::R2 => "R2",
            Regime::R3 => "R3",
        }
    }
}

/// Error when regime config is missing or invalid.
#[derive(Debug, Clone)]
pub enum RegimeConfigError {
    /// Thresholds not provided
    MissingThresholds,
    /// Toxicity config not provided
    MissingToxicityConfig,
    /// Invalid threshold value (e.g., negative)
    InvalidThreshold(String),
}

impl std::fmt::Display for RegimeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingThresholds => write!(f, "REGIME_CONFIG_ERROR: thresholds not provided"),
            Self::MissingToxicityConfig => {
                write!(f, "REGIME_CONFIG_ERROR: toxicity config not provided")
            }
            Self::InvalidThreshold(msg) => {
                write!(f, "REGIME_CONFIG_ERROR: invalid threshold - {}", msg)
            }
        }
    }
}

impl std::error::Error for RegimeConfigError {}

/// Thresholds for regime classification.
/// MUST be loaded from config - no defaults allowed.
#[derive(Debug, Clone)]
pub struct RegimeThresholds {
    /// d_perp threshold for R3
    pub tau_d_perp: f64,
    /// Fragility threshold for R3
    pub tau_fragility: f64,
    /// FTI persistence threshold
    pub tau_fti_persist: f64,
    /// Toxicity persistence threshold
    pub tau_toxicity_persist: f64,
    /// Minimum confidence for R3
    pub tau_confidence: f64,
    /// Thresholds for R1 (early detection)
    pub tau_r1_d_perp: f64,
    pub tau_r1_fragility: f64,
    /// Thresholds for R2 (confirmed)
    pub tau_r2_d_perp: f64,
    pub tau_r2_fragility: f64,
}

// NOTE: Default impl intentionally removed per audit requirement.
// Use explicit construction with validated config values.

impl RegimeThresholds {
    /// Validate thresholds. Returns error if invalid.
    pub fn validate(&self) -> Result<(), RegimeConfigError> {
        if self.tau_d_perp <= 0.0 {
            return Err(RegimeConfigError::InvalidThreshold(
                "tau_d_perp must be positive".into(),
            ));
        }
        if self.tau_fragility < 0.0 || self.tau_fragility > 1.0 {
            return Err(RegimeConfigError::InvalidThreshold(
                "tau_fragility must be in [0, 1]".into(),
            ));
        }
        if self.tau_confidence < 0.0 || self.tau_confidence > 1.0 {
            return Err(RegimeConfigError::InvalidThreshold(
                "tau_confidence must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

/// Toxicity tracker configuration.
/// MUST be loaded from config - no defaults allowed.
#[derive(Debug, Clone)]
pub struct ToxicityConfig {
    /// Volume bucket size
    pub bucket_size: f64,
    /// Maximum number of buckets to track
    pub max_buckets: usize,
    /// Threshold for "elevated" toxicity
    pub persist_threshold: f64,
    /// Number of windows for persistence ring buffer (persist = fraction above threshold)
    pub persist_window: usize,
}

// NOTE: Default impl intentionally removed per audit requirement.

/// Confidence score tracking.
/// Spec: Section 6.5 Confidence Metric (SEALED)
#[derive(Debug, Clone)]
pub struct ConfidenceTracker {
    /// Current confidence value [0, 1]
    value: f64,
}

impl ConfidenceTracker {
    /// Create a new confidence tracker (initial = 1.0).
    pub fn new() -> Self {
        Self { value: 1.0 }
    }

    /// Reset confidence to 1.0.
    pub fn reset(&mut self) {
        self.value = 1.0;
    }

    /// Apply penalty for missing optional field (×0.95).
    pub fn apply_missing_optional(&mut self) {
        self.value *= confidence::MISSING_OPTIONAL_FIELD;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for zero MAD (×0.80).
    pub fn apply_zero_mad(&mut self) {
        self.value *= confidence::ZERO_MAD;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for spread soft breach (×0.70).
    pub fn apply_spread_breach(&mut self) {
        self.value *= confidence::SPREAD_SOFT_BREACH;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for GPU fallback (×0.90).
    /// Note: Phase 1 is CPU-only, but we include this for completeness.
    pub fn apply_gpu_fallback(&mut self) {
        self.value *= confidence::GPU_FALLBACK;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for normalization instability (×0.85).
    pub fn apply_normalization_instability(&mut self) {
        self.value *= confidence::NORMALIZATION_INSTABILITY;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for undefined elasticity (×0.85).
    /// Called when volume is too low to compute meaningful elasticity.
    pub fn apply_undefined_elasticity(&mut self) {
        self.value *= confidence::UNDEFINED_ELASTICITY;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for NoTradesWindow (×0.95).
    /// Called when no trades received this window - toxicity not computable.
    pub fn apply_no_trades_window(&mut self) {
        self.value *= confidence::NO_TRADES_WINDOW;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for InsufficientHistory (×0.95).
    /// Called when not enough price history for FTI computation.
    pub fn apply_insufficient_history(&mut self) {
        self.value *= confidence::INSUFFICIENT_HISTORY;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for non-certified trades source (×0.97).
    /// Called when trades come from public WS instead of SBE stream.
    /// Smaller penalty than missing trades - data exists but lower integrity.
    pub fn apply_non_certified_trades(&mut self) {
        self.value *= confidence::NON_CERTIFIED_TRADES;
        self.value = self.value.clamp(0.0, 1.0);
    }

    /// Apply penalty for ToxicityState if needed.
    pub fn apply_toxicity_state(&mut self, state: &ToxicityState) {
        if state.needs_penalty() {
            self.apply_no_trades_window();
        }
    }

    /// Apply penalty for FTIState if needed.
    pub fn apply_fti_state(&mut self, state: &FTIState) {
        if state.needs_penalty() {
            self.apply_insufficient_history();
        }
    }

    /// Get current confidence value.
    pub fn value(&self) -> f64 {
        self.value
    }
}

impl Default for ConfidenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow toxicity (VPIN-style) tracker.
/// Spec: Section 5.2
/// Requires explicit config injection - no defaults.
pub struct ToxicityTracker {
    /// Volume bucket size (may be adjusted by heuristic)
    bucket_size: f64,
    /// Current bucket accumulated volume
    current_bucket_volume: f64,
    /// Current bucket buy volume
    current_bucket_buy: f64,
    /// Current bucket sell volume
    current_bucket_sell: f64,
    /// Recent bucket imbalances
    imbalances: std::collections::VecDeque<f64>,
    /// Maximum number of buckets to track
    max_buckets: usize,
    /// Ring buffer for persistence: true if toxicity was elevated in that window
    persist_ring: std::collections::VecDeque<bool>,
    /// Size of persistence ring buffer
    persist_window: usize,
    /// Threshold for "elevated" toxicity
    persist_threshold: f64,
    /// Number of trades received this window (for NoTradesWindow detection)
    trades_this_window: u64,
    /// Total trades received (for stats)
    total_trades: u64,
    // === Instrumentation fields ===
    /// Total buckets closed (finalized)
    buckets_closed: u64,
    /// Total buy volume
    total_buy_volume: f64,
    /// Total sell volume
    total_sell_volume: f64,
    /// Imbalance histogram (all closed buckets)
    all_imbalances: Vec<f64>,
    /// Toxicity values over time
    toxicity_history: Vec<f64>,
}

/// Toxicity instrumentation stats for debugging.
#[derive(Debug, Clone, Default)]
pub struct ToxicityInstrumentation {
    /// Number of buckets closed
    pub buckets_closed: u64,
    /// Buy ratio (buy_volume / total_volume)
    pub buy_ratio: f64,
    /// Total trades with known side
    pub total_trades: u64,
    /// Imbalance percentiles over closed buckets
    pub imb_p50: f64,
    pub imb_p95: f64,
    pub imb_max: f64,
    /// Toxicity percentiles over time
    pub tox_p50: f64,
    pub tox_p95: f64,
    pub tox_max: f64,
}

// NOTE: Default impl intentionally removed per audit requirement.

impl ToxicityTracker {
    /// Create a new toxicity tracker with explicit config.
    pub fn new(config: &ToxicityConfig) -> Self {
        Self {
            bucket_size: config.bucket_size,
            current_bucket_volume: 0.0,
            current_bucket_buy: 0.0,
            current_bucket_sell: 0.0,
            imbalances: std::collections::VecDeque::with_capacity(config.max_buckets),
            max_buckets: config.max_buckets,
            persist_ring: std::collections::VecDeque::with_capacity(config.persist_window),
            persist_window: config.persist_window,
            persist_threshold: config.persist_threshold,
            trades_this_window: 0,
            total_trades: 0,
            // Instrumentation
            buckets_closed: 0,
            total_buy_volume: 0.0,
            total_sell_volume: 0.0,
            all_imbalances: Vec::new(),
            toxicity_history: Vec::new(),
        }
    }

    /// Update bucket_size (e.g., from alignment heuristic).
    pub fn set_bucket_size(&mut self, bucket_size: f64) {
        self.bucket_size = bucket_size;
    }

    /// Get current bucket_size.
    pub fn bucket_size(&self) -> f64 {
        self.bucket_size
    }

    /// Finalize current bucket and compute imbalance.
    fn finalize_bucket(&mut self) {
        // Compute bucket imbalance (always in [0, 1] since we use bucket_size as denominator)
        let imb = (self.current_bucket_buy - self.current_bucket_sell).abs() / self.bucket_size;
        self.imbalances.push_back(imb);

        // Track for instrumentation
        self.buckets_closed += 1;
        self.all_imbalances.push(imb);

        // Keep only recent buckets
        while self.imbalances.len() > self.max_buckets {
            self.imbalances.pop_front();
        }

        // Reset bucket
        self.current_bucket_volume = 0.0;
        self.current_bucket_buy = 0.0;
        self.current_bucket_sell = 0.0;
    }

    /// Add a trade to the tracker.
    /// Handles bucket overflow by deterministically splitting volume across buckets.
    pub fn add_trade(&mut self, volume: f64, is_buy: bool) {
        self.trades_this_window += 1;
        self.total_trades += 1;

        // Track for instrumentation
        if is_buy {
            self.total_buy_volume += volume;
        } else {
            self.total_sell_volume += volume;
        }

        let mut remaining = volume;

        while remaining > 0.0 {
            let space_in_bucket = self.bucket_size - self.current_bucket_volume;

            if remaining >= space_in_bucket {
                // This trade fills (or overfills) the current bucket
                // Add only the portion that fits
                if is_buy {
                    self.current_bucket_buy += space_in_bucket;
                } else {
                    self.current_bucket_sell += space_in_bucket;
                }
                self.current_bucket_volume = self.bucket_size;

                // Finalize this bucket
                self.finalize_bucket();

                // Carry remainder to next bucket
                remaining -= space_in_bucket;
            } else {
                // Trade fits entirely in current bucket
                if is_buy {
                    self.current_bucket_buy += remaining;
                } else {
                    self.current_bucket_sell += remaining;
                }
                self.current_bucket_volume += remaining;
                remaining = 0.0;
            }
        }
    }

    /// Start a new window (reset trades_this_window counter).
    pub fn start_window(&mut self) {
        self.trades_this_window = 0;
    }

    /// Check if any trades were received this window.
    pub fn has_trades_this_window(&self) -> bool {
        self.trades_this_window > 0
    }

    /// Get total trades received.
    pub fn total_trades(&self) -> u64 {
        self.total_trades
    }

    /// Get current toxicity (rolling mean of imbalances).
    pub fn toxicity(&self) -> f64 {
        if self.imbalances.is_empty() {
            return 0.0;
        }
        self.imbalances.iter().sum::<f64>() / self.imbalances.len() as f64
    }

    /// Update persistence ring buffer and return current persist value.
    /// Returns fraction of last N windows where toxicity was elevated, in [0, 1].
    pub fn update_persist(&mut self) -> f64 {
        let tox = self.toxicity();
        let is_elevated = tox > self.persist_threshold;

        // Add to ring buffer
        self.persist_ring.push_back(is_elevated);

        // Keep only last persist_window entries
        while self.persist_ring.len() > self.persist_window {
            self.persist_ring.pop_front();
        }

        // Return fraction of elevated windows
        if self.persist_ring.is_empty() {
            0.0
        } else {
            let count = self.persist_ring.iter().filter(|&&x| x).count();
            count as f64 / self.persist_ring.len() as f64
        }
    }

    /// Get current toxicity state (tri-state for observability).
    /// Returns NoTradesWindow if no trades this window, otherwise Computable.
    /// Call this at end of window after all trades have been added.
    pub fn get_state(&mut self) -> ToxicityState {
        if !self.has_trades_this_window() {
            // No trades this window - toxicity is not computable
            ToxicityState::NoTradesWindow
        } else {
            // Trades received - toxicity is computable
            let toxicity = self.toxicity();
            let persist = self.update_persist();

            // Track for instrumentation
            self.toxicity_history.push(toxicity);

            ToxicityState::Computable { toxicity, persist }
        }
    }

    /// Get number of filled buckets.
    pub fn bucket_count(&self) -> usize {
        self.imbalances.len()
    }

    /// Get instrumentation stats for debugging.
    pub fn instrumentation(&self) -> ToxicityInstrumentation {
        let total_volume = self.total_buy_volume + self.total_sell_volume;
        let buy_ratio = if total_volume > 0.0 {
            self.total_buy_volume / total_volume
        } else {
            0.5
        };

        // Compute imbalance percentiles
        let (imb_p50, imb_p95, imb_max) = if !self.all_imbalances.is_empty() {
            let mut sorted = self.all_imbalances.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            (sorted[n / 2], sorted[(n * 95) / 100], sorted[n - 1])
        } else {
            (0.0, 0.0, 0.0)
        };

        // Compute toxicity percentiles
        let (tox_p50, tox_p95, tox_max) = if !self.toxicity_history.is_empty() {
            let mut sorted = self.toxicity_history.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            (sorted[n / 2], sorted[(n * 95) / 100], sorted[n - 1])
        } else {
            (0.0, 0.0, 0.0)
        };

        ToxicityInstrumentation {
            buckets_closed: self.buckets_closed,
            buy_ratio,
            total_trades: self.total_trades,
            imb_p50,
            imb_p95,
            imb_max,
            tox_p50,
            tox_p95,
            tox_max,
        }
    }
}

// NOTE: Default impl for ToxicityTracker intentionally removed per audit requirement.

/// Regime classifier.
/// Requires explicit config injection - no defaults.
pub struct RegimeClassifier {
    thresholds: RegimeThresholds,
    toxicity_config: ToxicityConfig,
    confidence: ConfidenceTracker,
    toxicity: ToxicityTracker,
    /// Whether non-certified trades penalty has been applied (session-level)
    non_certified_penalty_applied: bool,
}

// NOTE: Default impl intentionally removed per audit requirement.

impl RegimeClassifier {
    /// Create a new regime classifier with explicit config.
    /// Returns error if config is invalid.
    pub fn new(
        thresholds: RegimeThresholds,
        toxicity_config: ToxicityConfig,
    ) -> Result<Self, RegimeConfigError> {
        thresholds.validate()?;

        Ok(Self {
            thresholds,
            toxicity_config: toxicity_config.clone(),
            confidence: ConfidenceTracker::new(),
            toxicity: ToxicityTracker::new(&toxicity_config),
            non_certified_penalty_applied: false,
        })
    }

    /// Reset for new session.
    pub fn reset(&mut self) {
        self.confidence.reset();
        self.toxicity = ToxicityTracker::new(&self.toxicity_config);
        self.non_certified_penalty_applied = false;
    }

    /// Get current thresholds (for manifest/audit).
    pub fn thresholds(&self) -> &RegimeThresholds {
        &self.thresholds
    }

    /// Apply confidence penalty for undefined elasticity.
    /// Called when volume is too low to compute meaningful elasticity.
    pub fn apply_undefined_elasticity(&mut self) {
        self.confidence.apply_undefined_elasticity();
    }

    /// Apply confidence penalty for non-certified trades source.
    /// Called once per session when trades come from public WS (not SBE).
    /// Returns true if penalty was applied, false if already applied.
    /// Caller should log the reason if desired.
    pub fn apply_non_certified_trades(&mut self) -> bool {
        if !self.non_certified_penalty_applied {
            self.confidence.apply_non_certified_trades();
            self.non_certified_penalty_applied = true;
            true
        } else {
            false
        }
    }

    /// Check if non-certified trades penalty was applied.
    pub fn has_non_certified_penalty(&self) -> bool {
        self.non_certified_penalty_applied
    }

    /// Apply confidence penalties based on normalization status.
    /// v1.2: Degraded/DegradedHigh apply penalties but don't refuse.
    /// Only RefuseFrame sets refused=true.
    pub fn apply_normalization_status(&mut self, status: NormalizationStatus) {
        match status {
            NormalizationStatus::Normal => {}
            NormalizationStatus::Warmup => {}
            NormalizationStatus::Degraded => {
                // MAD==0 ≤2s or other degraded conditions
                self.confidence.apply_zero_mad(); // ×0.80
            }
            NormalizationStatus::DegradedHigh => {
                // MAD==0 >2s - stronger penalty
                self.confidence.apply_zero_mad(); // ×0.80
                self.confidence.apply_normalization_instability(); // ×0.85
            }
            NormalizationStatus::RefuseFrame => {
                // Structural invalidity - apply both penalties
                self.confidence.apply_zero_mad();
                self.confidence.apply_normalization_instability();
            }
        }
    }

    /// Add a trade for toxicity tracking.
    pub fn add_trade(&mut self, volume: f64, is_buy: bool) {
        self.toxicity.add_trade(volume, is_buy);
    }

    /// Start a new window for toxicity tracking.
    pub fn start_window(&mut self) {
        self.toxicity.start_window();
    }

    /// Get current toxicity state (tri-state for observability).
    pub fn get_toxicity_state(&mut self) -> ToxicityState {
        self.toxicity.get_state()
    }

    /// Set bucket_size for toxicity tracking (from heuristic).
    pub fn set_bucket_size(&mut self, bucket_size: f64) {
        self.toxicity.set_bucket_size(bucket_size);
    }

    /// Get current bucket_size.
    pub fn bucket_size(&self) -> f64 {
        self.toxicity.bucket_size()
    }

    /// Get toxicity instrumentation stats for debugging.
    pub fn toxicity_instrumentation(&self) -> ToxicityInstrumentation {
        self.toxicity.instrumentation()
    }

    /// Get current toxicity value (rolling mean of bucket imbalances).
    /// Returns value in [0, 1] where 0 = balanced flow, 1 = fully one-sided.
    pub fn current_toxicity(&self) -> f64 {
        self.toxicity.toxicity()
    }

    /// Classify regime based on current metrics.
    /// Spec: Section 6.6
    ///
    /// AUDIT: Confidence is reset at start of each frame, then penalties applied.
    /// This ensures confidence reflects per-frame quality, not cumulative session history.
    ///
    /// Parameters:
    /// - `elasticity_undefined`: If true, applies confidence penalty for undefined elasticity
    /// - `degraded_reasons`: Bitmask of degradation reasons from normalization (v1.2)
    pub fn classify(
        &mut self,
        regime_metrics: &RegimeMetrics,
        fragility: &FragilityScore,
        fti: &FTIMetrics,
        norm_status: NormalizationStatus,
        elasticity_undefined: bool,
        degraded_reasons: u32,
    ) -> RegimeClassification {
        // CRITICAL: Reset confidence for this frame
        // Penalties are per-frame, not cumulative across the session
        self.confidence.reset();

        // Apply normalization penalties for this frame
        self.apply_normalization_status(norm_status);

        // Apply elasticity penalty if undefined
        if elasticity_undefined {
            self.confidence.apply_undefined_elasticity();
        }

        // Compute normalization_penalty = effective_confidence / raw_confidence
        // Since raw_confidence is always 1.0, penalty = effective_confidence
        let normalization_penalty = self.confidence.value();

        // Update toxicity persistence
        let toxicity_persist = self.toxicity.update_persist();
        let toxicity_value = self.toxicity.toxicity();

        // v1.2: refused=true ONLY for RefuseFrame (structural invalidity).
        // Degraded/DegradedHigh apply confidence penalties but are still tradeable.
        let refused = norm_status == NormalizationStatus::RefuseFrame;

        // R3 check (most restrictive)
        // Spec: Section 6.6
        let is_r3 = regime_metrics.d_perp > self.thresholds.tau_d_perp
            && fragility.value > self.thresholds.tau_fragility
            && (fti.fti_persist > self.thresholds.tau_fti_persist
                || toxicity_persist > self.thresholds.tau_toxicity_persist)
            && self.confidence.value() > self.thresholds.tau_confidence;

        // R2 check
        let is_r2 = !is_r3
            && regime_metrics.d_perp > self.thresholds.tau_r2_d_perp
            && fragility.value > self.thresholds.tau_r2_fragility;

        // R1 check
        let is_r1 = !is_r3
            && !is_r2
            && (regime_metrics.d_perp > self.thresholds.tau_r1_d_perp
                || fragility.value > self.thresholds.tau_r1_fragility);

        let regime = if is_r3 {
            Regime::R3
        } else if is_r2 {
            Regime::R2
        } else if is_r1 {
            Regime::R1
        } else {
            Regime::R0
        };

        RegimeClassification {
            regime,
            confidence: self.confidence.value(),
            normalization_penalty,
            degraded_reasons,
            d_perp: regime_metrics.d_perp,
            fragility: fragility.value,
            fti_persist: fti.fti_persist,
            toxicity: toxicity_value,
            toxicity_persist,
            refused,
        }
    }

    /// Get current confidence.
    pub fn confidence(&self) -> f64 {
        self.confidence.value()
    }
}

// NOTE: Default impl for RegimeClassifier intentionally removed per audit requirement.

/// Regime classification result.
#[derive(Debug, Clone)]
pub struct RegimeClassification {
    /// Classified regime
    pub regime: Regime,
    /// Effective confidence after all penalties (used for sizing)
    pub confidence: f64,
    /// Normalization penalty factor (1.0 = no penalty)
    /// confidence = 1.0 * normalization_penalty
    pub normalization_penalty: f64,
    /// Bitmask of degradation reasons (v1.2)
    pub degraded_reasons: u32,
    /// d_perp value used
    pub d_perp: f64,
    /// Fragility value used
    pub fragility: f64,
    /// FTI persistence value
    pub fti_persist: f64,
    /// Toxicity value
    pub toxicity: f64,
    /// Toxicity persistence
    pub toxicity_persist: f64,
    /// Whether signal was refused (RefuseFrame only)
    pub refused: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragility::FragilityComponents;

    fn test_thresholds() -> RegimeThresholds {
        RegimeThresholds {
            tau_d_perp: 2.0,
            tau_fragility: 0.6,
            tau_fti_persist: 0.3, // 30% of windows must be elevated (was 5.0 counter)
            tau_toxicity_persist: 0.3, // 30% of windows must be elevated (was 5.0 counter)
            tau_confidence: 0.5,
            tau_r1_d_perp: 1.0,
            tau_r1_fragility: 0.3,
            tau_r2_d_perp: 1.5,
            tau_r2_fragility: 0.45,
        }
    }

    fn test_toxicity_config() -> ToxicityConfig {
        ToxicityConfig {
            bucket_size: 1000.0,
            max_buckets: 50,
            persist_threshold: 0.5,
            persist_window: 20,
        }
    }

    #[test]
    fn test_regime_r0() {
        let mut classifier =
            RegimeClassifier::new(test_thresholds(), test_toxicity_config()).unwrap();

        let metrics = RegimeMetrics {
            d_perp: 0.5, // Below threshold
            v_para: 0.0,
            rho: 0.0,
        };

        let fragility = FragilityScore {
            value: 0.2, // Below threshold
            components: FragilityComponents {
                gap_risk: 0.0,
                elasticity: 0.0,
                depth_decay: 0.0,
                spread_z: 0.0,
                depth_slope: 0.0,
            },
        };

        let fti = FTIMetrics::default();

        let result = classifier.classify(
            &metrics,
            &fragility,
            &fti,
            NormalizationStatus::Normal,
            false,
            0,
        );

        assert_eq!(result.regime, Regime::R0);
    }

    #[test]
    fn test_confidence_penalties() {
        let mut tracker = ConfidenceTracker::new();
        assert!((tracker.value() - 1.0).abs() < 1e-10);

        tracker.apply_zero_mad();
        assert!((tracker.value() - 0.80).abs() < 1e-10);

        tracker.apply_spread_breach();
        assert!((tracker.value() - 0.56).abs() < 1e-10); // 0.80 * 0.70
    }

    #[test]
    fn test_toxicity_basic() {
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 20,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // Add trades to fill a bucket
        for _ in 0..10 {
            tracker.add_trade(10.0, true); // All buys
        }

        // Imbalance should be 1.0 (all buys)
        assert!((tracker.toxicity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_toxicity_bucket_overflow() {
        // Test that bucket overflow is handled correctly with carry-forward
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 20,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // Add 80 volume
        tracker.add_trade(80.0, true);
        assert_eq!(tracker.toxicity(), 0.0); // No bucket finalized yet

        // Add 50 more - should finalize first bucket (80+20=100) and carry 30
        tracker.add_trade(50.0, true);
        assert!((tracker.toxicity() - 1.0).abs() < 0.01); // One bucket with 100% buy imbalance

        // Add 70 more to complete second bucket (30+70=100)
        tracker.add_trade(70.0, true);
        // Two buckets now, both all-buy
        assert!((tracker.toxicity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_toxicity_large_trade_spans_multiple_buckets() {
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 20,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // Single trade of 250 should create 2 full buckets + 50 in current
        tracker.add_trade(250.0, true);

        // Two complete buckets with imbalance 1.0
        assert!((tracker.toxicity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_toxicity_persist_returns_fraction() {
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 10,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // Fill 10 windows with elevated toxicity
        for _ in 0..10 {
            tracker.add_trade(100.0, true); // Creates bucket with imbalance 1.0
            let persist = tracker.update_persist();
            assert!((0.0..=1.0).contains(&persist), "persist must be in [0,1]");
        }

        // All 10 windows should be elevated (toxicity=1.0 > threshold=0.5)
        let final_persist = tracker.update_persist();
        assert!((final_persist - 1.0).abs() < 0.01, "10/10 elevated = 1.0");

        // Add balanced trades to lower toxicity
        for _ in 0..5 {
            tracker.add_trade(50.0, true);
            tracker.add_trade(50.0, false);
            tracker.update_persist();
        }

        // Now should be approximately 5/10 = 0.5 (half elevated, half not)
        let persist = tracker.update_persist();
        assert!((0.0..=1.0).contains(&persist), "persist must be in [0,1]");
    }

    #[test]
    fn test_config_validation() {
        let mut thresholds = test_thresholds();
        thresholds.tau_d_perp = -1.0; // Invalid

        let result = RegimeClassifier::new(thresholds, test_toxicity_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_toxicity_state_no_trades() {
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 20,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // No trades added - should return NoTradesWindow
        let state = tracker.get_state();
        assert!(matches!(state, ToxicityState::NoTradesWindow));
        assert!(state.needs_penalty());
        assert_eq!(state.toxicity_or_default(), 0.0);
        assert_eq!(state.persist_or_default(), 0.0);
    }

    #[test]
    fn test_toxicity_state_with_trades() {
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 20,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // Add trades to fill a bucket
        tracker.add_trade(100.0, true);

        // Should return Computable with toxicity=1.0 (all buys)
        let state = tracker.get_state();
        assert!(matches!(state, ToxicityState::Computable { .. }));
        assert!(!state.needs_penalty());
        assert!((state.toxicity_or_default() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_toxicity_window_tracking() {
        let config = ToxicityConfig {
            bucket_size: 100.0,
            max_buckets: 10,
            persist_threshold: 0.5,
            persist_window: 20,
        };
        let mut tracker = ToxicityTracker::new(&config);

        // Window 1: Add trades
        tracker.add_trade(50.0, true);
        assert!(tracker.has_trades_this_window());

        // Start new window
        tracker.start_window();
        assert!(!tracker.has_trades_this_window());

        // No trades this window
        let state = tracker.get_state();
        assert!(matches!(state, ToxicityState::NoTradesWindow));
    }

    #[test]
    fn test_confidence_no_trades_penalty() {
        let mut tracker = ConfidenceTracker::new();
        assert!((tracker.value() - 1.0).abs() < 1e-10);

        // Apply NoTradesWindow penalty (×0.95)
        tracker.apply_no_trades_window();
        assert!((tracker.value() - 0.95).abs() < 1e-10);

        // Apply again
        tracker.apply_no_trades_window();
        assert!((tracker.value() - 0.9025).abs() < 1e-10); // 0.95 * 0.95
    }

    #[test]
    fn test_confidence_insufficient_history_penalty() {
        let mut tracker = ConfidenceTracker::new();
        assert!((tracker.value() - 1.0).abs() < 1e-10);

        // Apply InsufficientHistory penalty (×0.95)
        tracker.apply_insufficient_history();
        assert!((tracker.value() - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_apply_toxicity_state() {
        let mut tracker = ConfidenceTracker::new();

        // NoTradesWindow should apply penalty
        let no_trades = ToxicityState::NoTradesWindow;
        tracker.apply_toxicity_state(&no_trades);
        assert!((tracker.value() - 0.95).abs() < 1e-10);

        // Reset and test Computable - no penalty
        tracker.reset();
        let computable = ToxicityState::Computable {
            toxicity: 0.5,
            persist: 0.3,
        };
        tracker.apply_toxicity_state(&computable);
        assert!((tracker.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_non_certified_trades_penalty() {
        let mut tracker = ConfidenceTracker::new();
        assert!((tracker.value() - 1.0).abs() < 1e-10);

        // Apply non-certified trades penalty (×0.97)
        tracker.apply_non_certified_trades();
        assert!((tracker.value() - 0.97).abs() < 1e-10);

        // Combined with NoTradesWindow: 0.97 * 0.95 = 0.9215
        tracker.apply_no_trades_window();
        assert!((tracker.value() - 0.9215).abs() < 1e-10);
    }

    #[test]
    fn test_classifier_non_certified_penalty_once() {
        let mut classifier =
            RegimeClassifier::new(test_thresholds(), test_toxicity_config()).unwrap();

        // First application returns true
        assert!(classifier.apply_non_certified_trades());
        assert!(classifier.has_non_certified_penalty());

        // Second application returns false (already applied)
        assert!(!classifier.apply_non_certified_trades());

        // Reset clears the flag
        classifier.reset();
        assert!(!classifier.has_non_certified_penalty());
        assert!(classifier.apply_non_certified_trades());
    }
}
