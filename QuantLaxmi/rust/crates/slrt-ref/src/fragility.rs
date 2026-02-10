//! Liquidity Fragility Score computation.
//!
//! Spec: Section 5.3 Liquidity Fragility Score
//!
//! fragility =
//!   w1 * clip(gapRisk)
//! + w2 * clip(elasticity)
//! - w3 * clip(depth_decay)
//! + w4 * clip(spread_z)
//! + w5 * clip(min(depth_slope_bid, depth_slope_ask))
//!
//! AUDIT: No Default impls - explicit config injection required.

use crate::features::{SnapshotFeatures, TradeFlowFeatures};

/// Error when fragility config is missing or invalid.
#[derive(Debug, Clone)]
pub enum FragilityConfigError {
    /// Weights not provided - cannot compute fragility
    MissingWeights,
    /// Bounds not provided - cannot compute fragility
    MissingBounds,
    /// Invalid weight sum (should be positive)
    InvalidWeightSum,
}

impl std::fmt::Display for FragilityConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingWeights => write!(f, "FRAGILITY_CONFIG_ERROR: weights not provided"),
            Self::MissingBounds => write!(f, "FRAGILITY_CONFIG_ERROR: bounds not provided"),
            Self::InvalidWeightSum => {
                write!(f, "FRAGILITY_CONFIG_ERROR: weight sum must be positive")
            }
        }
    }
}

impl std::error::Error for FragilityConfigError {}

/// Weights for fragility computation.
/// MUST be loaded from config - no defaults allowed.
#[derive(Debug, Clone)]
pub struct FragilityWeights {
    pub w1_gap_risk: f64,
    pub w2_elasticity: f64,
    pub w3_depth_decay: f64,
    pub w4_spread_z: f64,
    pub w5_depth_slope: f64,
}

// NOTE: Default impl intentionally removed per audit requirement.
// Use FragilityWeights::from_config() or explicit construction.

/// Clipping bounds for fragility components.
/// MUST be loaded from config - no defaults allowed.
#[derive(Debug, Clone)]
pub struct FragilityClipBounds {
    pub gap_risk_min: f64,
    pub gap_risk_max: f64,
    pub elasticity_min: f64,
    pub elasticity_max: f64,
    pub depth_decay_min: f64,
    pub depth_decay_max: f64,
    pub spread_z_min: f64,
    pub spread_z_max: f64,
    pub depth_slope_min: f64,
    pub depth_slope_max: f64,
}

// NOTE: Default impl intentionally removed per audit requirement.

/// Clip a value to [min, max] and normalize to [0, 1].
fn clip_normalize(value: f64, min: f64, max: f64) -> f64 {
    let clipped = value.clamp(min, max);
    (clipped - min) / (max - min + 1e-12)
}

/// Liquidity fragility calculator.
/// Requires explicit config injection - no defaults.
pub struct FragilityCalculator {
    weights: FragilityWeights,
    bounds: FragilityClipBounds,
    /// Rolling stats for spread z-score
    spread_mean: f64,
    spread_var: f64,
    spread_count: u64,
    ew_alpha: f64,
}

// NOTE: Default impl intentionally removed per audit requirement.
// Use FragilityCalculator::new(weights, bounds) with explicit config.

impl FragilityCalculator {
    /// Create a new fragility calculator with explicit config.
    /// Returns error if config is invalid.
    pub fn new(
        weights: FragilityWeights,
        bounds: FragilityClipBounds,
    ) -> Result<Self, FragilityConfigError> {
        // Validate weights sum is positive
        let weight_sum = weights.w1_gap_risk.abs()
            + weights.w2_elasticity.abs()
            + weights.w3_depth_decay.abs()
            + weights.w4_spread_z.abs()
            + weights.w5_depth_slope.abs();

        if weight_sum <= 0.0 {
            return Err(FragilityConfigError::InvalidWeightSum);
        }

        Ok(Self {
            weights,
            bounds,
            spread_mean: 0.0,
            spread_var: 0.0,
            spread_count: 0,
            ew_alpha: 0.01, // EW decay for spread stats
        })
    }

    /// Get the elasticity max bound (used for state vector capping).
    pub fn elasticity_max(&self) -> f64 {
        self.bounds.elasticity_max
    }

    /// Update spread statistics and compute z-score.
    fn update_spread_z(&mut self, spread: f64) -> f64 {
        self.spread_count += 1;

        if self.spread_count == 1 {
            self.spread_mean = spread;
            self.spread_var = 0.0;
            return 0.0;
        }

        // Exponentially weighted update
        let delta = spread - self.spread_mean;
        self.spread_mean += self.ew_alpha * delta;
        self.spread_var = (1.0 - self.ew_alpha) * (self.spread_var + self.ew_alpha * delta * delta);

        let std_dev = self.spread_var.sqrt();
        if std_dev > 1e-12 {
            (spread - self.spread_mean) / std_dev
        } else {
            0.0
        }
    }

    /// Compute fragility score.
    /// Spec: Section 5.3
    ///
    /// fragility =
    ///   w1 * clip(gapRisk)
    /// + w2 * clip(elasticity)
    /// - w3 * clip(depth_decay)
    /// + w4 * clip(spread_z)
    /// + w5 * clip(min(depth_slope_bid, depth_slope_ask))
    pub fn compute(
        &mut self,
        snapshot: &SnapshotFeatures,
        trade_flow: &TradeFlowFeatures,
    ) -> FragilityScore {
        // Compute spread z-score
        let spread_z = self.update_spread_z(snapshot.spread_ticks);

        // Clip and normalize each component
        let gap_risk_norm = clip_normalize(
            snapshot.gap_risk,
            self.bounds.gap_risk_min,
            self.bounds.gap_risk_max,
        );

        let elasticity_norm = clip_normalize(
            trade_flow.elasticity,
            self.bounds.elasticity_min,
            self.bounds.elasticity_max,
        );

        let depth_decay_norm = clip_normalize(
            trade_flow.depth_collapse_rate,
            self.bounds.depth_decay_min,
            self.bounds.depth_decay_max,
        );

        let spread_z_norm =
            clip_normalize(spread_z, self.bounds.spread_z_min, self.bounds.spread_z_max);

        let min_depth_slope = snapshot.depth_slope_bid.min(snapshot.depth_slope_ask);
        let depth_slope_norm = clip_normalize(
            min_depth_slope,
            self.bounds.depth_slope_min,
            self.bounds.depth_slope_max,
        );

        // Compute weighted sum (note: depth_decay is subtracted)
        let fragility = self.weights.w1_gap_risk * gap_risk_norm
            + self.weights.w2_elasticity * elasticity_norm
            - self.weights.w3_depth_decay * depth_decay_norm
            + self.weights.w4_spread_z * spread_z_norm
            + self.weights.w5_depth_slope * depth_slope_norm;

        FragilityScore {
            value: fragility.clamp(0.0, 1.0),
            components: FragilityComponents {
                gap_risk: gap_risk_norm,
                elasticity: elasticity_norm,
                depth_decay: depth_decay_norm,
                spread_z: spread_z_norm,
                depth_slope: depth_slope_norm,
            },
        }
    }
}

/// Fragility score with component breakdown.
#[derive(Debug, Clone)]
pub struct FragilityScore {
    /// Final fragility score [0, 1]
    pub value: f64,
    /// Individual component values (normalized)
    pub components: FragilityComponents,
}

/// Individual fragility components (for debugging/auditing).
#[derive(Debug, Clone)]
pub struct FragilityComponents {
    pub gap_risk: f64,
    pub elasticity: f64,
    pub depth_decay: f64,
    pub spread_z: f64,
    pub depth_slope: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> FragilityWeights {
        FragilityWeights {
            w1_gap_risk: 0.25,
            w2_elasticity: 0.20,
            w3_depth_decay: 0.15,
            w4_spread_z: 0.20,
            w5_depth_slope: 0.20,
        }
    }

    fn test_bounds() -> FragilityClipBounds {
        FragilityClipBounds {
            gap_risk_min: 0.0,
            gap_risk_max: 10.0,
            elasticity_min: 0.0,
            elasticity_max: 10.0,
            depth_decay_min: -1.0,
            depth_decay_max: 1.0,
            spread_z_min: -3.0,
            spread_z_max: 3.0,
            depth_slope_min: 0.0,
            depth_slope_max: 1.0,
        }
    }

    #[test]
    fn test_clip_normalize() {
        assert!((clip_normalize(0.0, 0.0, 1.0) - 0.0).abs() < 1e-10);
        assert!((clip_normalize(0.5, 0.0, 1.0) - 0.5).abs() < 1e-10);
        assert!((clip_normalize(1.0, 0.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((clip_normalize(-1.0, 0.0, 1.0) - 0.0).abs() < 1e-10); // Clipped
        assert!((clip_normalize(2.0, 0.0, 1.0) - 1.0).abs() < 1e-10); // Clipped
    }

    #[test]
    fn test_fragility_requires_config() {
        let weights = test_weights();
        let bounds = test_bounds();
        let calc = FragilityCalculator::new(weights, bounds);
        assert!(calc.is_ok());
    }

    #[test]
    fn test_fragility_rejects_zero_weights() {
        let weights = FragilityWeights {
            w1_gap_risk: 0.0,
            w2_elasticity: 0.0,
            w3_depth_decay: 0.0,
            w4_spread_z: 0.0,
            w5_depth_slope: 0.0,
        };
        let bounds = test_bounds();
        let calc = FragilityCalculator::new(weights, bounds);
        assert!(matches!(calc, Err(FragilityConfigError::InvalidWeightSum)));
    }

    #[test]
    fn test_fragility_basic() {
        let mut calc = FragilityCalculator::new(test_weights(), test_bounds()).unwrap();

        let snapshot = SnapshotFeatures {
            gap_risk: 2.0,
            depth_slope_bid: 0.1,
            depth_slope_ask: 0.15,
            spread_ticks: 1.0,
            ..Default::default()
        };

        let trade_flow = TradeFlowFeatures {
            elasticity: 0.5,
            depth_collapse_rate: -0.1,
            ..Default::default()
        };

        let score = calc.compute(&snapshot, &trade_flow);
        assert!(score.value >= 0.0 && score.value <= 1.0);
    }
}
