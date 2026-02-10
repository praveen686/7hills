//! quantlaxmi_rs — PyO3 bridge for hot-path Python <-> Rust interop.
//!
//! Exposes:
//! - FixedPoint: deterministic fixed-point arithmetic
//! - PySignal, PyTargetPosition: core trading types
//! - check_risk_gate: sub-ms pre-trade risk validation
//! - canonical_digest, hash_chain_link: SHA-256 matching Rust WAL
//! - validate_signal: field-level signal validation

mod fixed_point;
mod hash;
mod risk;
mod types;

use pyo3::prelude::*;

use fixed_point::FixedPoint;
use hash::{canonical_digest, hash_chain_link, verify_hash_chain};
use risk::{check_risk_gate, PyPortfolioState, PyRiskLimits};
use types::{PyRiskCheckResult, PySignal, PyTargetPosition, PyValidationResult};

/// Validate a signal and return structured result.
#[pyfunction]
fn validate_signal(signal: &PySignal) -> PyValidationResult {
    let errors = signal.validate();
    PyValidationResult {
        is_valid: errors.is_empty(),
        errors,
    }
}

/// Python module: quantlaxmi_rs
#[pymodule]
fn quantlaxmi_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", "0.1.0")?;

    // Types
    m.add_class::<FixedPoint>()?;
    m.add_class::<PySignal>()?;
    m.add_class::<PyTargetPosition>()?;
    m.add_class::<PyRiskCheckResult>()?;
    m.add_class::<PyValidationResult>()?;
    m.add_class::<PyRiskLimits>()?;
    m.add_class::<PyPortfolioState>()?;

    // Functions
    m.add_function(wrap_pyfunction!(canonical_digest, m)?)?;
    m.add_function(wrap_pyfunction!(hash_chain_link, m)?)?;
    m.add_function(wrap_pyfunction!(verify_hash_chain, m)?)?;
    m.add_function(wrap_pyfunction!(check_risk_gate, m)?)?;
    m.add_function(wrap_pyfunction!(validate_signal, m)?)?;

    Ok(())
}
