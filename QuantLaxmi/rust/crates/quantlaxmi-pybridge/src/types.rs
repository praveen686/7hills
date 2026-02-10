//! PyO3 wrappers for core QuantLaxmi types.
//!
//! These provide Python-accessible versions of Signal, OrderIntent,
//! ExecutionFill, and PositionUpdate for the live trading hot-path.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Signal emitted by a Python strategy.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PySignal {
    #[pyo3(get, set)]
    pub strategy_id: String,
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub direction: String,
    #[pyo3(get, set)]
    pub conviction: f64,
    #[pyo3(get, set)]
    pub instrument_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: String,
    #[pyo3(get, set)]
    pub ttl_bars: i32,
    #[pyo3(get, set)]
    pub regime: String,
    #[pyo3(get, set)]
    pub reasoning: String,
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PySignal {
    #[new]
    #[pyo3(signature = (
        strategy_id,
        symbol,
        direction,
        conviction,
        instrument_type = "FUT".to_string(),
        strike = 0.0,
        expiry = String::new(),
        ttl_bars = 5,
        regime = String::new(),
        reasoning = String::new(),
        metadata = HashMap::new(),
    ))]
    fn new(
        strategy_id: String,
        symbol: String,
        direction: String,
        conviction: f64,
        instrument_type: String,
        strike: f64,
        expiry: String,
        ttl_bars: i32,
        regime: String,
        reasoning: String,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            strategy_id,
            symbol,
            direction,
            conviction,
            instrument_type,
            strike,
            expiry,
            ttl_bars,
            regime,
            reasoning,
            metadata,
        }
    }

    /// Validate the signal fields. Returns list of errors (empty = valid).
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.strategy_id.is_empty() {
            errors.push("strategy_id is empty".into());
        }
        if self.symbol.is_empty() {
            errors.push("symbol is empty".into());
        }
        match self.direction.as_str() {
            "long" | "short" | "flat" => {}
            _ => errors.push(format!("invalid direction: '{}'", self.direction)),
        }
        if !(0.0..=1.0).contains(&self.conviction) {
            errors.push(format!("conviction {} not in [0, 1]", self.conviction));
        }
        match self.instrument_type.as_str() {
            "FUT" | "CE" | "PE" | "SPREAD" => {}
            _ => errors.push(format!(
                "invalid instrument_type: '{}'",
                self.instrument_type
            )),
        }
        if self.ttl_bars < 0 {
            errors.push(format!("ttl_bars {} is negative", self.ttl_bars));
        }
        errors
    }

    /// Check if signal is valid (no validation errors).
    #[getter]
    fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "PySignal(strategy_id='{}', symbol='{}', direction='{}', conviction={:.4})",
            self.strategy_id, self.symbol, self.direction, self.conviction
        )
    }
}

/// Target position (post-allocation, pre-gate).
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTargetPosition {
    #[pyo3(get, set)]
    pub strategy_id: String,
    #[pyo3(get, set)]
    pub symbol: String,
    #[pyo3(get, set)]
    pub direction: String,
    #[pyo3(get, set)]
    pub weight: f64,
    #[pyo3(get, set)]
    pub instrument_type: String,
    #[pyo3(get, set)]
    pub strike: f64,
    #[pyo3(get, set)]
    pub expiry: String,
    #[pyo3(get, set)]
    pub conviction: f64,
}

#[pymethods]
impl PyTargetPosition {
    #[new]
    #[pyo3(signature = (
        strategy_id,
        symbol,
        direction,
        weight,
        instrument_type = "FUT".to_string(),
        strike = 0.0,
        expiry = String::new(),
        conviction = 0.0,
    ))]
    fn new(
        strategy_id: String,
        symbol: String,
        direction: String,
        weight: f64,
        instrument_type: String,
        strike: f64,
        expiry: String,
        conviction: f64,
    ) -> Self {
        Self {
            strategy_id,
            symbol,
            direction,
            weight,
            instrument_type,
            strike,
            expiry,
            conviction,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTargetPosition(strategy='{}', symbol='{}', dir='{}', weight={:.4})",
            self.strategy_id, self.symbol, self.direction, self.weight
        )
    }
}

/// Risk check result returned by check_risk_gate().
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyRiskCheckResult {
    #[pyo3(get)]
    pub approved: bool,
    #[pyo3(get)]
    pub gate: String,
    #[pyo3(get)]
    pub adjusted_weight: f64,
    #[pyo3(get)]
    pub refuse_reason: String,
}

#[pymethods]
impl PyRiskCheckResult {
    fn __repr__(&self) -> String {
        format!(
            "PyRiskCheckResult(approved={}, gate='{}', adjusted_weight={:.4})",
            self.approved, self.gate, self.adjusted_weight
        )
    }
}

/// Signal validation result.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyValidationResult {
    #[pyo3(get)]
    pub is_valid: bool,
    #[pyo3(get)]
    pub errors: Vec<String>,
}

#[pymethods]
impl PyValidationResult {
    fn __repr__(&self) -> String {
        if self.is_valid {
            "PyValidationResult(valid=True)".into()
        } else {
            format!("PyValidationResult(valid=False, errors={:?})", self.errors)
        }
    }
}
