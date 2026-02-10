//! Options Greeks Calculator
//!
//! Implements Black-Scholes Greeks with extensions for:
//! - First-order Greeks: Delta, Gamma, Theta, Vega, Rho
//! - Second-order Greeks: Vanna, Volga (Vomma), Charm, Veta, Speed, Color
//! - Portfolio-level aggregation
//!
//! All calculations use f64 for precision with deterministic rounding.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::{E, PI};

/// Standard normal distribution for CDF/PDF calculations.
fn std_normal() -> Normal {
    Normal::new(0.0, 1.0).unwrap()
}

/// Standard normal PDF: φ(x) = (1/√(2π)) * e^(-x²/2)
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF: Φ(x)
fn norm_cdf(x: f64) -> f64 {
    std_normal().cdf(x)
}

/// Option type (Call or Put).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

/// Input parameters for Greeks calculation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OptionParams {
    /// Current spot price of underlying
    pub spot: f64,
    /// Strike price
    pub strike: f64,
    /// Time to expiration in years (e.g., 30 days = 30/365)
    pub time_to_expiry: f64,
    /// Risk-free interest rate (annualized, e.g., 0.05 for 5%)
    pub risk_free_rate: f64,
    /// Implied volatility (annualized, e.g., 0.20 for 20%)
    pub iv: f64,
    /// Dividend yield (annualized, e.g., 0.02 for 2%)
    pub dividend_yield: f64,
    /// Option type
    pub option_type: OptionType,
}

impl OptionParams {
    /// Create new option parameters.
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        iv: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            iv,
            dividend_yield: 0.0,
            option_type,
        }
    }

    /// Set dividend yield.
    pub fn with_dividend(mut self, dividend_yield: f64) -> Self {
        self.dividend_yield = dividend_yield;
        self
    }

    /// Calculate d1 from Black-Scholes formula.
    fn d1(&self) -> f64 {
        if self.time_to_expiry <= 0.0 || self.iv <= 0.0 {
            return 0.0;
        }
        let sqrt_t = self.time_to_expiry.sqrt();
        ((self.spot / self.strike).ln()
            + (self.risk_free_rate - self.dividend_yield + 0.5 * self.iv * self.iv)
                * self.time_to_expiry)
            / (self.iv * sqrt_t)
    }

    /// Calculate d2 from Black-Scholes formula.
    fn d2(&self) -> f64 {
        self.d1() - self.iv * self.time_to_expiry.sqrt()
    }

    /// Moneyness: ln(S/K)
    pub fn moneyness(&self) -> f64 {
        (self.spot / self.strike).ln()
    }

    /// Standardized moneyness: ln(S/K) / (σ√T)
    pub fn standardized_moneyness(&self) -> f64 {
        if self.time_to_expiry <= 0.0 || self.iv <= 0.0 {
            return 0.0;
        }
        self.moneyness() / (self.iv * self.time_to_expiry.sqrt())
    }
}

/// Complete set of Greeks for an option.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Greeks {
    // === First-Order Greeks ===
    /// Delta: ∂V/∂S - sensitivity to underlying price
    pub delta: f64,
    /// Gamma: ∂²V/∂S² - rate of change of delta
    pub gamma: f64,
    /// Theta: ∂V/∂t - time decay (per day)
    pub theta: f64,
    /// Vega: ∂V/∂σ - sensitivity to volatility (per 1% IV change)
    pub vega: f64,
    /// Rho: ∂V/∂r - sensitivity to interest rate (per 1% rate change)
    pub rho: f64,

    // === Second-Order Greeks ===
    /// Vanna: ∂²V/∂S∂σ = ∂Delta/∂σ = ∂Vega/∂S
    pub vanna: f64,
    /// Volga (Vomma): ∂²V/∂σ² = ∂Vega/∂σ
    pub volga: f64,
    /// Charm (Delta Decay): ∂²V/∂S∂t = ∂Delta/∂t
    pub charm: f64,
    /// Veta: ∂²V/∂σ∂t = ∂Vega/∂t
    pub veta: f64,
    /// Speed: ∂³V/∂S³ = ∂Gamma/∂S
    pub speed: f64,
    /// Color: ∂³V/∂S²∂t = ∂Gamma/∂t
    pub color: f64,

    // === Derived Metrics ===
    /// Theoretical option price
    pub price: f64,
    /// Intrinsic value
    pub intrinsic: f64,
    /// Extrinsic (time) value
    pub extrinsic: f64,
    /// Lambda (leverage): Delta * S / V
    pub lambda: f64,
}

impl Greeks {
    /// Calculate all Greeks for given option parameters.
    pub fn calculate(params: &OptionParams) -> Self {
        let s = params.spot;
        let k = params.strike;
        let t = params.time_to_expiry;
        let r = params.risk_free_rate;
        let q = params.dividend_yield;
        let sigma = params.iv;

        // Handle edge cases
        if t <= 0.0 {
            return Self::at_expiry(params);
        }
        if sigma <= 0.0 {
            return Self::zero_vol(params);
        }

        let sqrt_t = t.sqrt();
        let d1 = params.d1();
        let d2 = params.d2();

        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);
        let n_neg_d1 = norm_cdf(-d1);
        let n_neg_d2 = norm_cdf(-d2);
        let pdf_d1 = norm_pdf(d1);

        let exp_qt = E.powf(-q * t);
        let exp_rt = E.powf(-r * t);

        // === Option Price ===
        let price = match params.option_type {
            OptionType::Call => s * exp_qt * nd1 - k * exp_rt * nd2,
            OptionType::Put => k * exp_rt * n_neg_d2 - s * exp_qt * n_neg_d1,
        };

        // === Delta ===
        let delta = match params.option_type {
            OptionType::Call => exp_qt * nd1,
            OptionType::Put => -exp_qt * n_neg_d1,
        };

        // === Gamma (same for call and put) ===
        let gamma = exp_qt * pdf_d1 / (s * sigma * sqrt_t);

        // === Theta (per day) ===
        let theta_annual = match params.option_type {
            OptionType::Call => {
                -s * exp_qt * pdf_d1 * sigma / (2.0 * sqrt_t) - r * k * exp_rt * nd2
                    + q * s * exp_qt * nd1
            }
            OptionType::Put => {
                -s * exp_qt * pdf_d1 * sigma / (2.0 * sqrt_t) + r * k * exp_rt * n_neg_d2
                    - q * s * exp_qt * n_neg_d1
            }
        };
        let theta = theta_annual / 365.0; // Per day

        // === Vega (per 1% IV change) ===
        let vega = s * exp_qt * pdf_d1 * sqrt_t / 100.0;

        // === Rho (per 1% rate change) ===
        let rho = match params.option_type {
            OptionType::Call => k * t * exp_rt * nd2 / 100.0,
            OptionType::Put => -k * t * exp_rt * n_neg_d2 / 100.0,
        };

        // === Second-Order Greeks ===

        // Vanna: ∂Delta/∂σ
        let vanna = -exp_qt * pdf_d1 * d2 / sigma;

        // Volga (Vomma): ∂Vega/∂σ
        let volga = vega * 100.0 * d1 * d2 / sigma;

        // Charm: ∂Delta/∂t
        let charm = match params.option_type {
            OptionType::Call => {
                -exp_qt
                    * (pdf_d1 * (2.0 * (r - q) * t - d2 * sigma * sqrt_t)
                        / (2.0 * t * sigma * sqrt_t)
                        + q * nd1)
            }
            OptionType::Put => {
                exp_qt
                    * (pdf_d1 * (2.0 * (r - q) * t - d2 * sigma * sqrt_t)
                        / (2.0 * t * sigma * sqrt_t)
                        - q * n_neg_d1)
            }
        };

        // Veta: ∂Vega/∂t
        let veta = -s
            * exp_qt
            * pdf_d1
            * sqrt_t
            * (q + (r - q) * d1 / (sigma * sqrt_t) - (1.0 + d1 * d2) / (2.0 * t));

        // Speed: ∂Gamma/∂S
        let speed = -gamma * (1.0 + d1 / (sigma * sqrt_t)) / s;

        // Color: ∂Gamma/∂t
        let color = -exp_qt * pdf_d1 / (2.0 * s * t * sigma * sqrt_t)
            * (2.0 * q * t
                + 1.0
                + d1 * (2.0 * (r - q) * t - d2 * sigma * sqrt_t) / (sigma * sqrt_t));

        // === Derived Metrics ===
        let intrinsic = match params.option_type {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        };
        let extrinsic = (price - intrinsic).max(0.0);
        let lambda = if price > 0.0 { delta * s / price } else { 0.0 };

        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna,
            volga,
            charm,
            veta,
            speed,
            color,
            price,
            intrinsic,
            extrinsic,
            lambda,
        }
    }

    /// Greeks at expiry (T=0).
    fn at_expiry(params: &OptionParams) -> Self {
        let intrinsic = match params.option_type {
            OptionType::Call => (params.spot - params.strike).max(0.0),
            OptionType::Put => (params.strike - params.spot).max(0.0),
        };

        let delta = match params.option_type {
            OptionType::Call => {
                if params.spot > params.strike {
                    1.0
                } else {
                    0.0
                }
            }
            OptionType::Put => {
                if params.spot < params.strike {
                    -1.0
                } else {
                    0.0
                }
            }
        };

        Greeks {
            delta,
            price: intrinsic,
            intrinsic,
            ..Default::default()
        }
    }

    /// Greeks with zero volatility.
    fn zero_vol(params: &OptionParams) -> Self {
        let forward = params.spot
            * E.powf((params.risk_free_rate - params.dividend_yield) * params.time_to_expiry);
        let df = E.powf(-params.risk_free_rate * params.time_to_expiry);

        let (price, delta) = match params.option_type {
            OptionType::Call => {
                if forward > params.strike {
                    ((forward - params.strike) * df, df)
                } else {
                    (0.0, 0.0)
                }
            }
            OptionType::Put => {
                if forward < params.strike {
                    ((params.strike - forward) * df, -df)
                } else {
                    (0.0, 0.0)
                }
            }
        };

        let intrinsic = match params.option_type {
            OptionType::Call => (params.spot - params.strike).max(0.0),
            OptionType::Put => (params.strike - params.spot).max(0.0),
        };

        Greeks {
            delta,
            price,
            intrinsic,
            extrinsic: (price - intrinsic).max(0.0),
            ..Default::default()
        }
    }

    /// Scale Greeks by position quantity (positive = long, negative = short).
    pub fn scale(&self, qty: f64) -> Greeks {
        Greeks {
            delta: self.delta * qty,
            gamma: self.gamma * qty,
            theta: self.theta * qty,
            vega: self.vega * qty,
            rho: self.rho * qty,
            vanna: self.vanna * qty,
            volga: self.volga * qty,
            charm: self.charm * qty,
            veta: self.veta * qty,
            speed: self.speed * qty,
            color: self.color * qty,
            price: self.price * qty,
            intrinsic: self.intrinsic * qty,
            extrinsic: self.extrinsic * qty,
            lambda: self.lambda, // Lambda doesn't scale
        }
    }

    /// Add Greeks (for portfolio aggregation).
    pub fn add(&self, other: &Greeks) -> Greeks {
        Greeks {
            delta: self.delta + other.delta,
            gamma: self.gamma + other.gamma,
            theta: self.theta + other.theta,
            vega: self.vega + other.vega,
            rho: self.rho + other.rho,
            vanna: self.vanna + other.vanna,
            volga: self.volga + other.volga,
            charm: self.charm + other.charm,
            veta: self.veta + other.veta,
            speed: self.speed + other.speed,
            color: self.color + other.color,
            price: self.price + other.price,
            intrinsic: self.intrinsic + other.intrinsic,
            extrinsic: self.extrinsic + other.extrinsic,
            lambda: 0.0, // Doesn't aggregate meaningfully
        }
    }
}

/// Portfolio-level Greeks aggregator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortfolioGreeks {
    /// Aggregated Greeks
    pub total: Greeks,
    /// Number of positions
    pub position_count: usize,
    /// Notional exposure (sum of |delta| * spot * qty)
    pub notional_delta: f64,
    /// Dollar gamma (gamma * spot² * 0.01 / 2)
    pub dollar_gamma: f64,
    /// Theta per day in currency
    pub daily_theta: f64,
    /// Vega in currency per 1% IV move
    pub dollar_vega: f64,
}

impl PortfolioGreeks {
    /// Create empty portfolio.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a position's Greeks to the portfolio.
    pub fn add_position(&mut self, greeks: &Greeks, spot: f64, qty: f64, multiplier: f64) {
        let scaled = greeks.scale(qty * multiplier);
        self.total = self.total.add(&scaled);
        self.position_count += 1;

        self.notional_delta += scaled.delta.abs() * spot;
        self.dollar_gamma += scaled.gamma * spot * spot * 0.01 / 2.0;
        self.daily_theta += scaled.theta;
        self.dollar_vega += scaled.vega;
    }

    /// Get portfolio delta exposure as equivalent underlying shares.
    pub fn delta_equivalent(&self, spot: f64) -> f64 {
        if spot > 0.0 {
            self.total.delta * spot / spot // Returns delta directly
        } else {
            0.0
        }
    }

    /// Check if portfolio is delta neutral (within threshold).
    pub fn is_delta_neutral(&self, threshold: f64) -> bool {
        self.total.delta.abs() <= threshold
    }

    /// Check if portfolio is gamma positive (long gamma).
    pub fn is_long_gamma(&self) -> bool {
        self.total.gamma > 0.0
    }

    /// Check if portfolio is theta positive (collecting time decay).
    pub fn is_positive_theta(&self) -> bool {
        self.total.theta > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_atm_call_delta() {
        let params = OptionParams::new(
            100.0,
            100.0,
            0.25, // S=K=100, T=3mo
            0.05,
            0.20, // r=5%, σ=20%
            OptionType::Call,
        );
        let greeks = Greeks::calculate(&params);

        // ATM call delta should be ~0.5-0.6
        assert!(greeks.delta > 0.5 && greeks.delta < 0.65);
        // Gamma should be positive
        assert!(greeks.gamma > 0.0);
        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);
        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_atm_put_delta() {
        let params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Put);
        let greeks = Greeks::calculate(&params);

        // ATM put delta should be ~-0.4 to -0.5
        assert!(greeks.delta < -0.35 && greeks.delta > -0.55);
    }

    #[test]
    fn test_put_call_parity_delta() {
        let call_params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Call);
        let put_params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Put);

        let call_greeks = Greeks::calculate(&call_params);
        let put_greeks = Greeks::calculate(&put_params);

        // Put-call parity: Call_delta - Put_delta ≈ 1 (with dividend adjustment)
        let delta_diff = call_greeks.delta - put_greeks.delta;
        assert_relative_eq!(delta_diff, 1.0, epsilon = 0.02);
    }

    #[test]
    fn test_deep_itm_call() {
        let params = OptionParams::new(
            150.0,
            100.0,
            0.25, // Deep ITM: S=150, K=100
            0.05,
            0.20,
            OptionType::Call,
        );
        let greeks = Greeks::calculate(&params);

        // Deep ITM call delta approaches 1
        assert!(greeks.delta > 0.95);
        // Intrinsic value should be ~50
        assert_relative_eq!(greeks.intrinsic, 50.0, epsilon = 0.01);
    }

    #[test]
    fn test_deep_otm_call() {
        let params = OptionParams::new(
            50.0,
            100.0,
            0.25, // Deep OTM: S=50, K=100
            0.05,
            0.20,
            OptionType::Call,
        );
        let greeks = Greeks::calculate(&params);

        // Deep OTM call delta approaches 0
        assert!(greeks.delta < 0.05);
        // Intrinsic value should be 0
        assert_relative_eq!(greeks.intrinsic, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_gamma_same_for_call_put() {
        let call_params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Call);
        let put_params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Put);

        let call_greeks = Greeks::calculate(&call_params);
        let put_greeks = Greeks::calculate(&put_params);

        // Gamma should be identical for call and put at same strike
        assert_relative_eq!(call_greeks.gamma, put_greeks.gamma, epsilon = 1e-10);
    }

    #[test]
    fn test_portfolio_aggregation() {
        let call_params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Call);
        let put_params = OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Put);

        let call_greeks = Greeks::calculate(&call_params);
        let put_greeks = Greeks::calculate(&put_params);

        let mut portfolio = PortfolioGreeks::new();
        portfolio.add_position(&call_greeks, 100.0, 1.0, 100.0); // Long 1 call
        portfolio.add_position(&put_greeks, 100.0, 1.0, 100.0); // Long 1 put

        // Straddle should have near-zero delta
        assert!(portfolio.total.delta.abs() < 15.0); // 100 multiplier
                                                     // But positive gamma
        assert!(portfolio.total.gamma > 0.0);
    }
}
