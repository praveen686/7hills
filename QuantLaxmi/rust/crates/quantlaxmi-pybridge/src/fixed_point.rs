//! Fixed-point arithmetic exposed to Python.
//!
//! Matches Rust's mantissa+exponent encoding exactly, so Python can produce
//! identical bytes for hash-chain verification.

use pyo3::prelude::*;

/// Fixed-point number: actual_value = mantissa * 10^exponent.
///
/// Python usage:
///     fp = FixedPoint(mantissa=2345000, exponent=-2)
///     assert fp.to_float() == 23450.00
///     fp2 = FixedPoint.from_float(23450.00, -2)
///     assert fp2.mantissa == 2345000
#[pyclass]
#[derive(Clone, Debug)]
pub struct FixedPoint {
    #[pyo3(get, set)]
    pub mantissa: i64,
    #[pyo3(get, set)]
    pub exponent: i32,
}

#[pymethods]
impl FixedPoint {
    #[new]
    fn new(mantissa: i64, exponent: i32) -> Self {
        Self { mantissa, exponent }
    }

    /// Convert to Python float.
    fn to_float(&self) -> f64 {
        self.mantissa as f64 * 10f64.powi(self.exponent)
    }

    /// Create FixedPoint from a float with the given exponent.
    ///
    /// Example: FixedPoint.from_float(23450.00, -2) → mantissa=2345000
    #[staticmethod]
    fn from_float(value: f64, exponent: i32) -> Self {
        let scale = 10f64.powi(-exponent);
        let mantissa = (value * scale).round() as i64;
        Self { mantissa, exponent }
    }

    /// Confidence fixed-point (exponent=-4, 10000 = 1.0).
    #[staticmethod]
    fn from_confidence(value: f64) -> Self {
        Self::from_float(value, -4)
    }

    /// Spread basis-points (exponent=-2, 523 = 5.23 bps).
    #[staticmethod]
    fn from_spread_bps(value: f64) -> Self {
        Self::from_float(value, -2)
    }

    fn __repr__(&self) -> String {
        format!(
            "FixedPoint(mantissa={}, exponent={}, value={})",
            self.mantissa,
            self.exponent,
            self.to_float()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.mantissa == other.mantissa && self.exponent == other.exponent
    }
}
