//! Feature vector construction for regime detection.
//!
//! All features are fixed-point with explicit exponents to ensure determinism.

use serde::{Deserialize, Serialize};

/// Fixed-point feature value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FeatureValue {
    /// Mantissa (integer part)
    pub mantissa: i64,
    /// Exponent (e.g., -4 means divide by 10000)
    pub exponent: i8,
    /// Whether this value is present (vs missing)
    pub present: bool,
}

impl FeatureValue {
    /// Create a present feature value.
    pub fn present(mantissa: i64, exponent: i8) -> Self {
        Self {
            mantissa,
            exponent,
            present: true,
        }
    }

    /// Create a missing feature value.
    pub fn missing() -> Self {
        Self {
            mantissa: 0,
            exponent: 0,
            present: false,
        }
    }

    /// Convert to f64 for computation (internal use only).
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_f64(&self) -> Option<f64> {
        if self.present {
            Some(self.mantissa as f64 * 10f64.powi(self.exponent as i32))
        } else {
            None
        }
    }

    /// Normalize to a target exponent.
    pub fn normalize_to(&self, target_exp: i8) -> Self {
        if !self.present {
            return Self::missing();
        }

        let exp_diff = self.exponent as i32 - target_exp as i32;
        let new_mantissa = if exp_diff >= 0 {
            self.mantissa * 10i64.pow(exp_diff as u32)
        } else {
            self.mantissa / 10i64.pow((-exp_diff) as u32)
        };

        Self {
            mantissa: new_mantissa,
            exponent: target_exp,
            present: true,
        }
    }
}

/// Feature vector for regime detection.
///
/// Standard microstructure features used for Grassmann lifting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Mid-price return (bps, exp=-4)
    pub mid_return: FeatureValue,
    /// Order book imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty), scaled to [-10000, 10000]
    pub imbalance: FeatureValue,
    /// Spread in bps (exp=-4)
    pub spread_bps: FeatureValue,
    /// Short-horizon realized volatility proxy (exp=-4)
    pub vol_proxy: FeatureValue,
    /// Order flow pressure ratio: bid_qty / ask_qty, scaled (exp=-4)
    pub pressure: FeatureValue,
    /// VPIN-like metric (Volume-synchronized Probability of Informed Trading)
    pub vpin: FeatureValue,
}

impl FeatureVector {
    /// Create a new feature vector with all values present.
    pub fn new(
        mid_return: i64,
        imbalance: i64,
        spread_bps: i64,
        vol_proxy: i64,
        pressure: i64,
        vpin: i64,
    ) -> Self {
        Self {
            mid_return: FeatureValue::present(mid_return, -4),
            imbalance: FeatureValue::present(imbalance, -4),
            spread_bps: FeatureValue::present(spread_bps, -4),
            vol_proxy: FeatureValue::present(vol_proxy, -4),
            pressure: FeatureValue::present(pressure, -4),
            vpin: FeatureValue::present(vpin, -4),
        }
    }

    /// Convert to a dense f64 vector for matrix operations.
    ///
    /// Missing values are replaced with 0.0 (mean imputation assumption).
    pub(crate) fn to_dense(&self) -> [f64; 6] {
        [
            self.mid_return.to_f64().unwrap_or(0.0),
            self.imbalance.to_f64().unwrap_or(0.0),
            self.spread_bps.to_f64().unwrap_or(0.0),
            self.vol_proxy.to_f64().unwrap_or(0.0),
            self.pressure.to_f64().unwrap_or(0.0),
            self.vpin.to_f64().unwrap_or(0.0),
        ]
    }

    /// Check if all required features are present.
    pub fn is_complete(&self) -> bool {
        self.mid_return.present
            && self.imbalance.present
            && self.spread_bps.present
            && self.vol_proxy.present
            && self.pressure.present
            && self.vpin.present
    }

    /// Count of present features.
    pub fn present_count(&self) -> usize {
        [
            self.mid_return.present,
            self.imbalance.present,
            self.spread_bps.present,
            self.vol_proxy.present,
            self.pressure.present,
            self.vpin.present,
        ]
        .iter()
        .filter(|&&p| p)
        .count()
    }
}

/// Builder for microstructure features from raw tick data.
pub struct MicrostructureFeatures {
    /// Previous mid price for return calculation
    prev_mid: Option<i64>,
    /// Price exponent (stored for potential future IV-based scaling)
    #[allow(dead_code)]
    price_exponent: i8,
    /// Rolling window for volatility
    return_window: Vec<i64>,
    /// Window size for volatility calculation
    vol_window_size: usize,
}

impl MicrostructureFeatures {
    /// Create a new feature builder.
    pub fn new(price_exponent: i8, vol_window_size: usize) -> Self {
        Self {
            prev_mid: None,
            price_exponent,
            return_window: Vec::with_capacity(vol_window_size),
            vol_window_size,
        }
    }

    /// Compute features from a tick.
    ///
    /// # Arguments
    /// * `bid_price` - Bid price mantissa
    /// * `ask_price` - Ask price mantissa
    /// * `bid_qty` - Bid quantity
    /// * `ask_qty` - Ask quantity
    /// * `volume` - Trade volume (optional, for VPIN)
    pub fn compute(
        &mut self,
        bid_price: i64,
        ask_price: i64,
        bid_qty: i64,
        ask_qty: i64,
        _volume: Option<i64>,
    ) -> FeatureVector {
        let mid = (bid_price + ask_price) / 2;

        // Mid return (bps)
        let mid_return = if let Some(prev) = self.prev_mid {
            if prev != 0 {
                // Return in bps: (mid - prev) / prev * 10000
                ((mid - prev) * 10000) / prev
            } else {
                0
            }
        } else {
            0
        };

        // Update rolling window for volatility
        self.return_window.push(mid_return);
        if self.return_window.len() > self.vol_window_size {
            self.return_window.remove(0);
        }

        // Imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) * 10000
        let total_qty = bid_qty + ask_qty;
        let imbalance = if total_qty > 0 {
            ((bid_qty - ask_qty) * 10000) / total_qty
        } else {
            0
        };

        // Spread in bps: (ask - bid) / mid * 10000
        let spread_bps = if mid > 0 {
            ((ask_price - bid_price) * 10000) / mid
        } else {
            0
        };

        // Volatility proxy: sum of |returns| / n
        let vol_proxy = if !self.return_window.is_empty() {
            let sum_abs: i64 = self.return_window.iter().map(|r| r.abs()).sum();
            sum_abs / self.return_window.len() as i64
        } else {
            0
        };

        // Pressure ratio: bid_qty / ask_qty * 10000 (capped at 30000)
        let pressure = if ask_qty > 0 {
            let ratio = (bid_qty * 10000) / ask_qty;
            ratio.clamp(-30000, 30000)
        } else if bid_qty > 0 {
            30000 // Max ratio
        } else {
            10000 // Neutral (1.0)
        };

        // VPIN placeholder (would need volume bucketing)
        let vpin = 5000; // Neutral 0.5

        self.prev_mid = Some(mid);

        FeatureVector::new(mid_return, imbalance, spread_bps, vol_proxy, pressure, vpin)
    }

    /// Reset state for a new symbol/session.
    pub fn reset(&mut self) {
        self.prev_mid = None;
        self.return_window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_value_normalization() {
        let v = FeatureValue::present(100, -2); // 1.00
        let normalized = v.normalize_to(-4); // Should be 10000
        assert_eq!(normalized.mantissa, 10000);
        assert_eq!(normalized.exponent, -4);
    }

    #[test]
    fn test_feature_vector_complete() {
        let fv = FeatureVector::new(10, 500, 20, 15, 12000, 5000);
        assert!(fv.is_complete());
        assert_eq!(fv.present_count(), 6);
    }

    #[test]
    fn test_microstructure_features() {
        let mut builder = MicrostructureFeatures::new(-2, 10);

        // First tick
        let f1 = builder.compute(10000, 10010, 100, 100, None);
        assert_eq!(f1.mid_return.mantissa, 0); // No previous

        // Second tick with price change
        let f2 = builder.compute(10010, 10020, 100, 100, None);
        // Mid went from 10005 to 10015, return = 10/10005 * 10000 ≈ 9 bps
        assert!(f2.mid_return.mantissa > 0);
    }

    #[test]
    fn test_imbalance_calculation() {
        let mut builder = MicrostructureFeatures::new(-2, 10);

        // Strong bid imbalance
        let f = builder.compute(10000, 10010, 200, 100, None);
        // imbalance = (200 - 100) / 300 * 10000 = 3333
        assert_eq!(f.imbalance.mantissa, 3333);

        // Strong ask imbalance
        builder.reset();
        let f2 = builder.compute(10000, 10010, 100, 200, None);
        assert_eq!(f2.imbalance.mantissa, -3333);
    }
}
