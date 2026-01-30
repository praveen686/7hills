//! # QuantLaxmi Regime Detection
//!
//! Grassmann manifold-based regime detection for microstructure trading.
//!
//! ## Architecture
//!
//! ```text
//! Features → RegimeLift → Grassmann Subspace → RegimeHead → Labels/Shifts
//!    ↓           ↓              ↓                  ↓
//!  [x_t]    [Covariance]    [U_t ∈ Gr(k,n)]    [CPD + Prototypes]
//! ```
//!
//! ## Determinism Guarantees
//!
//! All operations are designed for WAL-auditability:
//! - Fixed-point accumulators (no float drift)
//! - Canonical eigenvector sign/ordering
//! - Quantized subspace representation
//! - SHA-256 digests for replay parity

pub mod canonical;
pub mod cpd;
pub mod events;
pub mod features;
pub mod grassmann;
pub mod lift;
pub mod prototypes;
pub mod ramanujan;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

// Re-exports
pub use canonical::{CanonicalSubspace, SubspaceDigest};
pub use cpd::{CusumDetector, RegimeShift};
pub use events::{RegimeLabelEvent, RegimeShiftEvent, RegimeSubspaceEvent};
pub use features::{FeatureVector, MicrostructureFeatures};
pub use grassmann::{grassmann_distance, principal_angles};
pub use lift::{RegimeLift, RegimeLiftConfig};
pub use prototypes::{PrototypeBank, RegimeLabel, RegimePrototype};
pub use ramanujan::{
    ramanujan_sum, MicrostructurePeriodicity, PeriodicityFeatures, RamanujanFilterBank,
    RamanujanPeriodicityDetector,
};

/// Regime detection engine combining lift + head.
///
/// This is the main entry point for regime detection.
pub struct RegimeEngine {
    /// Configuration
    config: RegimeEngineConfig,
    /// The geometric lifting stage
    lift: RegimeLift,
    /// Change-point detector
    cpd: CusumDetector,
    /// Prototype bank for labeling
    prototypes: PrototypeBank,
    /// Previous subspace for distance calculation
    prev_subspace: Option<CanonicalSubspace>,
    /// History of subspace distances for CPD
    distance_history: VecDeque<i64>,
    /// Online learning: feature statistics tracker
    feature_stats: OnlineFeatureStats,
    /// Online learning: enable automatic prototype capture
    online_learning_enabled: bool,
    /// Counter for prototype IDs
    prototype_counter: u32,
}

/// Online feature statistics for heuristic regime labeling.
#[derive(Debug, Clone, Default)]
pub struct OnlineFeatureStats {
    /// Rolling sum of mid_return
    return_sum: i64,
    /// Rolling sum of squared mid_return (for variance)
    return_sq_sum: i64,
    /// Rolling sum of absolute mid_return (for volatility)
    return_abs_sum: i64,
    /// Rolling sum of spread
    spread_sum: i64,
    /// Rolling sum of imbalance
    imbalance_sum: i64,
    /// Count of samples in current window
    count: usize,
    /// Window size for stats
    window_size: usize,
    /// Recent returns for autocorrelation
    recent_returns: VecDeque<i64>,
}

impl OnlineFeatureStats {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            recent_returns: VecDeque::with_capacity(window_size),
            ..Default::default()
        }
    }

    fn update(&mut self, features: &FeatureVector) {
        let ret = features.mid_return.mantissa;
        let spread = features.spread_bps.mantissa;
        let imbalance = features.imbalance.mantissa;

        self.return_sum += ret;
        self.return_sq_sum += ret * ret;
        self.return_abs_sum += ret.abs();
        self.spread_sum += spread;
        self.imbalance_sum += imbalance;
        self.count += 1;

        self.recent_returns.push_back(ret);
        if self.recent_returns.len() > self.window_size {
            // Remove old value from sums
            if let Some(old_ret) = self.recent_returns.pop_front() {
                self.return_sum -= old_ret;
                self.return_sq_sum -= old_ret * old_ret;
                self.return_abs_sum -= old_ret.abs();
                // Note: we don't track old spread/imbalance, so these are approximate
                self.count = self.count.saturating_sub(1);
            }
        }
    }

    /// Classify regime based on feature statistics using heuristics.
    ///
    /// Thresholds are calibrated for both tick data and minute-candle warmup data.
    /// Minute candles have ~10-50x smaller variance than tick data.
    fn classify_heuristic(&self) -> RegimeLabel {
        if self.count < 20 {
            return RegimeLabel::Unknown;
        }

        let n = self.count as i64;
        let mean_return = self.return_sum / n;
        let mean_abs_return = self.return_abs_sum / n;
        let variance = (self.return_sq_sum / n) - (mean_return * mean_return);
        let mean_spread = self.spread_sum / n;

        // Calculate return autocorrelation (simple lag-1)
        let autocorr = self.calculate_autocorr();

        // Heuristic classification based on return statistics:
        // - Quiet: low volatility, small moves
        // - TrendImpulse: directional drift with momentum (positive autocorr)
        // - MeanReversionChop: high volatility with mean reversion (negative autocorr)
        // - LiquidityDrought: wide spreads with volatility
        // - EventShock: extreme volatility

        // Thresholds scaled for minute-candle data (variance ~2-20 typical)
        // For tick data, these would be ~10-50x higher
        let low_vol_threshold = 3; // Very calm market
        let med_vol_threshold = 10; // Normal volatility
        let high_vol_threshold = 25; // Elevated volatility
        let extreme_vol_threshold = 50; // Extreme volatility
        let trend_threshold = 3; // Directional bias threshold
        let wide_spread_threshold = 80; // Wide spread (above neutral 50)

        // Classification logic with proper priority
        if variance > extreme_vol_threshold {
            // Extreme volatility = event/shock
            RegimeLabel::EventShock
        } else if mean_spread > wide_spread_threshold && variance > med_vol_threshold {
            // Wide spreads + volatility = liquidity issues
            RegimeLabel::LiquidityDrought
        } else if variance > high_vol_threshold && autocorr < -100 {
            // High vol + mean reversion = choppy market
            RegimeLabel::MeanReversionChop
        } else if mean_return.abs() > trend_threshold
            && autocorr > 50
            && variance > low_vol_threshold
        {
            // Directional move with momentum = trend
            RegimeLabel::TrendImpulse
        } else if variance <= low_vol_threshold && mean_abs_return <= 2 {
            // Very low volatility = quiet
            RegimeLabel::Quiet
        } else if variance <= med_vol_threshold {
            // Moderate volatility, no clear pattern = quiet-ish
            RegimeLabel::Quiet
        } else {
            // Default: if vol is elevated but no clear pattern, call it chop
            RegimeLabel::MeanReversionChop
        }
    }

    fn calculate_autocorr(&self) -> i64 {
        if self.recent_returns.len() < 10 {
            return 0;
        }

        let n = self.recent_returns.len();
        let mean = self.return_sum / n as i64;

        let mut cov = 0i64;
        let mut var = 0i64;

        for i in 1..n {
            let r0 = self.recent_returns[i - 1] - mean;
            let r1 = self.recent_returns[i] - mean;
            cov += r0 * r1;
            var += r0 * r0;
        }

        if var == 0 {
            return 0;
        }

        // Return autocorr scaled by 10000 (exp=-4)
        (cov * 10000) / var
    }

    fn reset(&mut self) {
        self.return_sum = 0;
        self.return_sq_sum = 0;
        self.return_abs_sum = 0;
        self.spread_sum = 0;
        self.imbalance_sum = 0;
        self.count = 0;
        self.recent_returns.clear();
    }
}

/// Configuration for the regime engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeEngineConfig {
    /// Number of features (n in Gr(k,n))
    pub n_features: usize,
    /// Subspace dimension (k in Gr(k,n))
    pub subspace_dim: usize,
    /// Rolling window size for covariance
    pub window_size: usize,
    /// CUSUM threshold for regime shift detection (mantissa, exp=-4)
    pub cpd_threshold_mantissa: i64,
    /// CUSUM drift parameter (mantissa, exp=-4)
    pub cpd_drift_mantissa: i64,
    /// Exponent for distance/threshold values
    pub distance_exponent: i8,
    /// Minimum confidence for label assignment (mantissa, exp=-4)
    pub min_confidence_mantissa: i64,
}

impl Default for RegimeEngineConfig {
    fn default() -> Self {
        Self {
            n_features: 6,                // mid_return, imbalance, spread, vol, pressure, vpin
            subspace_dim: 3,              // Top-3 principal components
            window_size: 64,              // 64-tick rolling window
            cpd_threshold_mantissa: 5000, // 0.5 with exp=-4
            cpd_drift_mantissa: 100,      // 0.01 drift
            distance_exponent: -4,
            min_confidence_mantissa: 2000, // 0.2 minimum margin
        }
    }
}

impl RegimeEngine {
    /// Create a new regime engine with given config.
    pub fn new(config: RegimeEngineConfig) -> Self {
        let lift = RegimeLift::new(RegimeLiftConfig {
            n_features: config.n_features,
            subspace_dim: config.subspace_dim,
            window_size: config.window_size,
        });

        let cpd = CusumDetector::new(
            config.cpd_threshold_mantissa,
            config.cpd_drift_mantissa,
            config.distance_exponent,
        );

        Self {
            config: config.clone(),
            lift,
            cpd,
            prototypes: PrototypeBank::new(),
            prev_subspace: None,
            distance_history: VecDeque::with_capacity(256),
            feature_stats: OnlineFeatureStats::new(config.window_size),
            online_learning_enabled: true, // Enable by default
            prototype_counter: 0,
        }
    }

    /// Create with online learning disabled (for replay/determinism).
    pub fn new_without_learning(config: RegimeEngineConfig) -> Self {
        let mut engine = Self::new(config);
        engine.online_learning_enabled = false;
        engine
    }

    /// Enable or disable online learning.
    pub fn set_online_learning(&mut self, enabled: bool) {
        self.online_learning_enabled = enabled;
    }

    /// Process a new feature vector and return regime events.
    ///
    /// Returns: (subspace_event, optional_shift_event, optional_label_event)
    pub fn process(
        &mut self,
        ts: DateTime<Utc>,
        symbol: &str,
        features: &FeatureVector,
    ) -> (
        Option<RegimeSubspaceEvent>,
        Option<RegimeShiftEvent>,
        Option<RegimeLabelEvent>,
    ) {
        // Update feature statistics for online learning
        self.feature_stats.update(features);

        // Step 1: Update lift with new features
        let subspace = match self.lift.update(features) {
            Some(s) => s,
            None => return (None, None, None), // Not enough data yet
        };

        // Step 2: Compute distance to previous subspace
        let distance_mantissa = if let Some(ref prev) = self.prev_subspace {
            grassmann_distance(prev, &subspace, self.config.distance_exponent)
        } else {
            0 // First subspace, no distance
        };

        // Step 3: Run CPD on distance
        let shift_event = self.cpd.update(ts, symbol, distance_mantissa);

        // Step 4: Online learning - capture prototype on regime shift
        if self.online_learning_enabled {
            if let Some(ref _shift) = shift_event {
                // Regime shift detected! Capture current subspace as new prototype
                let label = self.feature_stats.classify_heuristic();
                if label != RegimeLabel::Unknown {
                    self.prototype_counter += 1;
                    let prototype = RegimePrototype::new(
                        label,
                        format!("{}_{}", label.as_str(), self.prototype_counter),
                        subspace.clone(),
                        format!("Online learned at {}", ts),
                        "online_learning",
                    );
                    self.prototypes.add(prototype);
                    // Note: Don't reset feature stats - we need continuous data for classification
                    // The rolling window handles old data removal automatically
                }
            }
        }

        // Step 5: Label via prototypes or heuristics (with fallback)
        let label_event = if !self.prototypes.is_empty() {
            // Try prototype-based classification first
            match self.prototypes.classify(ts, symbol, &subspace) {
                Some(event) => Some(event),
                None => {
                    // Prototype classification failed (low confidence)
                    // Fall back to heuristic
                    let label = self.feature_stats.classify_heuristic();
                    if label != RegimeLabel::Unknown {
                        Some(RegimeLabelEvent {
                            ts,
                            symbol: symbol.to_string(),
                            regime_id: label.as_str().to_string(),
                            confidence_mantissa: 3000, // Lower confidence for fallback
                            distance_best_mantissa: 0,
                            distance_second_mantissa: 0,
                            distance_exponent: self.config.distance_exponent,
                            method: events::ClassificationMethod::Heuristic,
                            subspace_digest: subspace.digest(),
                        })
                    } else {
                        None
                    }
                }
            }
        } else {
            // No prototypes yet, use heuristic classification
            let label = self.feature_stats.classify_heuristic();
            if label != RegimeLabel::Unknown {
                Some(RegimeLabelEvent {
                    ts,
                    symbol: symbol.to_string(),
                    regime_id: label.as_str().to_string(),
                    confidence_mantissa: 5000, // 0.5 confidence for heuristic
                    distance_best_mantissa: 0,
                    distance_second_mantissa: 0,
                    distance_exponent: self.config.distance_exponent,
                    method: events::ClassificationMethod::Heuristic,
                    subspace_digest: subspace.digest(),
                })
            } else {
                None
            }
        };

        // Step 6: Create subspace event
        let subspace_event = RegimeSubspaceEvent {
            ts,
            symbol: symbol.to_string(),
            window_len: self.config.window_size,
            k: self.config.subspace_dim,
            n: self.config.n_features,
            subspace_digest: subspace.digest(),
            distance_to_prev_mantissa: distance_mantissa,
            distance_exponent: self.config.distance_exponent,
        };

        // Update state
        self.prev_subspace = Some(subspace);
        self.distance_history.push_back(distance_mantissa);
        if self.distance_history.len() > 256 {
            self.distance_history.pop_front();
        }

        (Some(subspace_event), shift_event, label_event)
    }

    /// Add a regime prototype for labeling.
    pub fn add_prototype(&mut self, prototype: RegimePrototype) {
        self.prototypes.add(prototype);
    }

    /// Reset the engine state (for new symbol/session).
    pub fn reset(&mut self) {
        self.lift.reset();
        self.cpd.reset();
        self.prev_subspace = None;
        self.distance_history.clear();
        self.feature_stats.reset();
        // Note: prototypes are preserved across resets (learned knowledge)
    }

    /// Get current heuristic regime classification (without requiring prototypes).
    pub fn current_heuristic_regime(&self) -> RegimeLabel {
        self.feature_stats.classify_heuristic()
    }

    /// Get number of learned prototypes.
    pub fn prototype_count(&self) -> usize {
        self.prototypes.len()
    }

    /// Get debug info about feature stats (for debugging).
    pub fn debug_feature_stats(&self) -> (usize, i64, i64, i64) {
        let n = self.feature_stats.count;
        if n == 0 {
            return (0, 0, 0, 0);
        }
        let mean_abs = self.feature_stats.return_abs_sum / n as i64;
        let variance = (self.feature_stats.return_sq_sum / n as i64)
            - (self.feature_stats.return_sum / n as i64).pow(2);
        let spread = self.feature_stats.spread_sum / n as i64;
        (n, mean_abs, variance, spread)
    }

    /// Get current config for serialization.
    pub fn config(&self) -> &RegimeEngineConfig {
        &self.config
    }
}

// =============================================================================
// Canonical Bytes for WAL
// =============================================================================

/// Trait for types that can be serialized to canonical bytes.
pub trait CanonicalBytes {
    fn canonical_bytes(&self) -> Vec<u8>;

    fn canonical_digest(&self) -> String {
        let bytes = self.canonical_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hex::encode(hasher.finalize())
    }
}

impl CanonicalBytes for RegimeEngineConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.n_features as u32).to_le_bytes());
        buf.extend_from_slice(&(self.subspace_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.window_size as u32).to_le_bytes());
        buf.extend_from_slice(&self.cpd_threshold_mantissa.to_le_bytes());
        buf.extend_from_slice(&self.cpd_drift_mantissa.to_le_bytes());
        buf.push(self.distance_exponent as u8);
        buf.extend_from_slice(&self.min_confidence_mantissa.to_le_bytes());
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_canonical_deterministic() {
        let c1 = RegimeEngineConfig::default();
        let c2 = RegimeEngineConfig::default();
        assert_eq!(c1.canonical_bytes(), c2.canonical_bytes());
        assert_eq!(c1.canonical_digest(), c2.canonical_digest());
    }

    #[test]
    fn test_engine_creation() {
        let engine = RegimeEngine::new(RegimeEngineConfig::default());
        assert_eq!(engine.config.n_features, 6);
        assert_eq!(engine.config.subspace_dim, 3);
    }
}
