//! Python-callable risk gate checks.
//!
//! Sub-millisecond pre-trade validation matching the Rust risk engine.

use pyo3::prelude::*;

use crate::types::{PyRiskCheckResult, PyTargetPosition};

/// Risk limits configuration (mirrors Python RiskLimits).
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyRiskLimits {
    #[pyo3(get, set)]
    pub max_portfolio_dd: f64,
    #[pyo3(get, set)]
    pub max_strategy_dd: f64,
    #[pyo3(get, set)]
    pub max_single_instrument: f64,
    #[pyo3(get, set)]
    pub max_single_stock_fno: f64,
    #[pyo3(get, set)]
    pub vpin_block_threshold: f64,
    #[pyo3(get, set)]
    pub max_total_exposure: f64,
    #[pyo3(get, set)]
    pub max_correlated_exposure: f64,
}

#[pymethods]
impl PyRiskLimits {
    #[new]
    #[pyo3(signature = (
        max_portfolio_dd = 0.05,
        max_strategy_dd = 0.03,
        max_single_instrument = 0.20,
        max_single_stock_fno = 0.05,
        vpin_block_threshold = 0.70,
        max_total_exposure = 1.50,
        max_correlated_exposure = 0.40,
    ))]
    fn new(
        max_portfolio_dd: f64,
        max_strategy_dd: f64,
        max_single_instrument: f64,
        max_single_stock_fno: f64,
        vpin_block_threshold: f64,
        max_total_exposure: f64,
        max_correlated_exposure: f64,
    ) -> Self {
        Self {
            max_portfolio_dd,
            max_strategy_dd,
            max_single_instrument,
            max_single_stock_fno,
            vpin_block_threshold,
            max_total_exposure,
            max_correlated_exposure,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRiskLimits(max_dd={:.2}, vpin_block={:.2}, max_exposure={:.2})",
            self.max_portfolio_dd, self.vpin_block_threshold, self.max_total_exposure
        )
    }
}

/// Portfolio state for risk evaluation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyPortfolioState {
    #[pyo3(get, set)]
    pub equity: f64,
    #[pyo3(get, set)]
    pub peak_equity: f64,
    #[pyo3(get, set)]
    pub total_exposure: f64,
    #[pyo3(get, set)]
    pub vpin: f64,
    #[pyo3(get, set)]
    pub strategy_dd: f64,
}

#[pymethods]
impl PyPortfolioState {
    #[new]
    #[pyo3(signature = (
        equity = 1.0,
        peak_equity = 1.0,
        total_exposure = 0.0,
        vpin = 0.0,
        strategy_dd = 0.0,
    ))]
    fn new(
        equity: f64,
        peak_equity: f64,
        total_exposure: f64,
        vpin: f64,
        strategy_dd: f64,
    ) -> Self {
        Self {
            equity,
            peak_equity,
            total_exposure,
            vpin,
            strategy_dd,
        }
    }
}

/// Index names exempt from stock FnO sub-limit.
const INDEX_NAMES: &[&str] = &[
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX",
];

/// Check risk gate for a target position against portfolio state and limits.
///
/// Matches Python `RiskManager._check_single()` exactly:
/// - Strict `>` comparisons (not `>=`) for all thresholds
/// - Concentration check includes existing position weight
/// - Flat direction auto-approved (exits reduce risk)
/// - Stock FnO sub-limit for non-index names
///
/// Args:
///     target: Target position to check
///     portfolio: Current portfolio state
///     limits: Risk limit configuration
///     current_instrument_weight: Existing weight in the same instrument (default 0.0)
///
/// Python usage:
///     result = check_risk_gate(target, portfolio, limits)
///     if result.approved:
///         execute_order(target)
///     else:
///         log_rejection(result.refuse_reason)
#[pyfunction]
#[pyo3(signature = (target, portfolio, limits, current_instrument_weight = 0.0))]
pub fn check_risk_gate(
    target: &PyTargetPosition,
    portfolio: &PyPortfolioState,
    limits: &PyRiskLimits,
    current_instrument_weight: f64,
) -> PyRiskCheckResult {
    // Flat signals always approved (exits reduce risk)
    if target.direction == "flat" {
        return PyRiskCheckResult {
            approved: true,
            gate: "pass".into(),
            adjusted_weight: 0.0,
            refuse_reason: String::new(),
        };
    }

    // Layer 1: VPIN toxicity check (strict >)
    if portfolio.vpin > limits.vpin_block_threshold {
        return PyRiskCheckResult {
            approved: false,
            gate: "block_vpin".into(),
            adjusted_weight: 0.0,
            refuse_reason: format!(
                "VPIN {:.4} > threshold {:.4}",
                portfolio.vpin, limits.vpin_block_threshold
            ),
        };
    }

    // Layer 2: Portfolio drawdown check (strict >)
    let portfolio_dd = if portfolio.peak_equity > 0.0 {
        1.0 - (portfolio.equity / portfolio.peak_equity)
    } else {
        0.0
    };
    if portfolio_dd > limits.max_portfolio_dd {
        return PyRiskCheckResult {
            approved: false,
            gate: "block_dd_portfolio".into(),
            adjusted_weight: 0.0,
            refuse_reason: format!(
                "Portfolio DD {:.4} > limit {:.4}",
                portfolio_dd, limits.max_portfolio_dd
            ),
        };
    }

    // Strategy drawdown check (strict >)
    if portfolio.strategy_dd > limits.max_strategy_dd {
        return PyRiskCheckResult {
            approved: false,
            gate: "block_dd_strategy".into(),
            adjusted_weight: 0.0,
            refuse_reason: format!(
                "Strategy DD {:.4} > limit {:.4}",
                portfolio.strategy_dd, limits.max_strategy_dd
            ),
        };
    }

    // Layer 3: Concentration — includes existing position weight
    let mut adjusted_weight = target.weight;

    // Single instrument limit
    if current_instrument_weight + adjusted_weight > limits.max_single_instrument {
        let max_add = (limits.max_single_instrument - current_instrument_weight).max(0.0);
        if max_add <= 0.001 {
            return PyRiskCheckResult {
                approved: false,
                gate: "block_concentration".into(),
                adjusted_weight: 0.0,
                refuse_reason: format!(
                    "{} weight={:.4} at limit",
                    target.symbol, current_instrument_weight
                ),
            };
        }
        adjusted_weight = adjusted_weight.min(max_add);
    }

    // Stock FnO sub-limit (non-index names only)
    let symbol_upper = target.symbol.to_uppercase();
    let is_index = INDEX_NAMES.iter().any(|&name| name == symbol_upper);
    if !is_index
        && current_instrument_weight + adjusted_weight > limits.max_single_stock_fno
    {
        let max_add = (limits.max_single_stock_fno - current_instrument_weight).max(0.0);
        if max_add <= 0.001 {
            return PyRiskCheckResult {
                approved: false,
                gate: "block_concentration".into(),
                adjusted_weight: 0.0,
                refuse_reason: format!(
                    "Stock FnO {} at {:.0}% limit",
                    target.symbol,
                    limits.max_single_stock_fno * 100.0
                ),
            };
        }
        adjusted_weight = adjusted_weight.min(max_add);
    }

    // Total exposure check (with size reduction)
    if portfolio.total_exposure + adjusted_weight > limits.max_total_exposure {
        let remaining = (limits.max_total_exposure - portfolio.total_exposure).max(0.0);
        if remaining <= 0.001 {
            return PyRiskCheckResult {
                approved: false,
                gate: "block_exposure".into(),
                adjusted_weight: 0.0,
                refuse_reason: format!(
                    "Total exposure at {:.0}% limit",
                    limits.max_total_exposure * 100.0
                ),
            };
        }
        adjusted_weight = adjusted_weight.min(remaining);
    }

    // Determine gate result
    let gate = if adjusted_weight < target.weight {
        "reduce_size"
    } else {
        "pass"
    };

    PyRiskCheckResult {
        approved: true,
        gate: gate.into(),
        adjusted_weight,
        refuse_reason: String::new(),
    }
}
