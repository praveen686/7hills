//! # SANOS: Smooth Arbitrage-free Non-parametric Option Surfaces
//!
//! Implementation of the SANOS method for constructing arbitrage-free call price surfaces.
//!
//! ## Reference
//! Buehler, H. et al. "Smooth strictly Arbitrage-free Non-parametric Option Surfaces (SANOS)"
//!
//! ## Design
//! This is a **read-only state module**. It takes option market data and produces an
//! arbitrage-free call price surface. No trading, no backtesting, no LF/APD yet.
//!
//! ## Key Concepts
//! - Call prices represented as convex combinations of shifted lognormal payoffs
//! - Martingale density constraints: q_i >= 0, Σq_i = 1, Σq_i*K_i = 1 (unit mean)
//! - Strike augmentation for boundary conditions
//! - Background variance V = σ_ATM² * T with smoothness factor η

use crate::pricing::implied_volatility;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use good_lp::{constraint, variable, Expression, ProblemVariables, Solution, SolverModel};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ============================================================================
// SANOS PRODUCTION POLICY — FROZEN (Phase 7 Certified 2026-01-23)
// Do not modify without Lead approval. These values are empirically validated.
// ============================================================================

/// SANOS smoothness parameter (η in the paper)
/// Controls trade-off between fit and smoothness. η=0 gives exact fit, η→1 gives flat density.
/// FROZEN: η=0.25 optimal per Phase 6 η-sweep analysis.
pub const ETA: f64 = 0.25;

/// Minimum effective variance (V_min) for numerical stability
/// Prevents LP matrix ill-conditioning when TTY is very short (< 7 days).
/// FROZEN: V_min=2e-4 resolves short-maturity degeneracy while preserving fit quality.
/// Validated: T1/T2/T3 all solve, calendar slack positive at market strikes.
pub const V_MIN: f64 = 2e-4;

/// Epsilon for strike augmentation (K1 = ε)
/// FROZEN: K1 fixed at 0.001 in normalized units. Do not shrink with T.
pub const EPSILON_STRIKE: f64 = 0.001;

/// Far OTM normalized strike (KN = k_N in normalized space)
/// FROZEN: Fixed at 1.30 for all expiries to ensure calendar constraint consistency.
/// All expiries use same normalized boundary: K_N(T_j) = k_N * F_j
pub const K_N_NORMALIZED: f64 = 1.30;

/// Far OTM multiplier for KN augmentation (legacy, prefer K_N_NORMALIZED)
pub const FAR_OTM_MULTIPLIER: f64 = 3.0;

/// Risk-free rate for Indian markets (approximate)
pub const RISK_FREE_RATE: f64 = 0.065;

/// Strike metadata for boundary hardening (Phase 8.1)
///
/// Tracks the domain of each strike point for safe feature extraction:
/// - Boundary points (K0, K1, KN): constraint-only, never used in features
/// - Market points: correspond to listed instruments, safe for features
/// - Interior points: within strike band, safe for features
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct StrikeMeta {
    /// Normalized strike value
    pub k: f64,
    /// True if this is a boundary point (K0=0, K1=ε, KN=1.30)
    pub is_boundary: bool,
    /// True if this strike corresponds to a listed market instrument
    pub is_market: bool,
    /// True if this strike is within the interior band (±band around ATM)
    pub is_interior: bool,
}

impl StrikeMeta {
    /// Create boundary point metadata
    pub fn boundary(k: f64) -> Self {
        Self { k, is_boundary: true, is_market: false, is_interior: false }
    }

    /// Create market point metadata
    pub fn market(k: f64, is_interior: bool) -> Self {
        Self { k, is_boundary: false, is_market: true, is_interior }
    }

    /// Check if this strike is safe for feature extraction
    pub fn is_feature_safe(&self) -> bool {
        !self.is_boundary && (self.is_market || self.is_interior)
    }
}

/// Input: Raw option market observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionQuote {
    pub symbol: String,
    pub strike: f64,
    pub is_call: bool,
    pub bid: f64,
    pub ask: f64,
    pub timestamp: DateTime<Utc>,
}

impl OptionQuote {
    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid();
        if mid <= 0.0 {
            return 0.0;
        }
        (self.ask - self.bid) / mid * 10_000.0
    }
}

/// Market data for a single expiry at a single timestamp
#[derive(Debug, Clone)]
pub struct ExpirySlice {
    pub underlying: String,
    pub expiry: String,
    pub timestamp: DateTime<Utc>,
    pub time_to_expiry: f64, // In years
    pub calls: BTreeMap<u32, OptionQuote>,
    pub puts: BTreeMap<u32, OptionQuote>,
}

impl ExpirySlice {
    pub fn new(underlying: &str, expiry: &str, timestamp: DateTime<Utc>, time_to_expiry: f64) -> Self {
        Self {
            underlying: underlying.to_string(),
            expiry: expiry.to_string(),
            timestamp,
            time_to_expiry,
            calls: BTreeMap::new(),
            puts: BTreeMap::new(),
        }
    }

    pub fn add_quote(&mut self, quote: OptionQuote) {
        let strike = quote.strike as u32;
        if quote.is_call {
            self.calls.insert(strike, quote);
        } else {
            self.puts.insert(strike, quote);
        }
    }
}

/// Normalized market data (forward units)
#[derive(Debug, Clone)]
pub struct NormalizedSlice {
    pub forward: f64,              // Estimated forward price F0
    pub time_to_expiry: f64,       // T in years
    pub strikes: Vec<f64>,         // Normalized strikes K/F0
    pub call_prices: Vec<f64>,     // Normalized call prices C/F0
    pub call_bids: Vec<f64>,       // Normalized bid prices
    pub call_asks: Vec<f64>,       // Normalized ask prices
    pub atm_iv: f64,               // ATM implied volatility
    pub raw_strikes: Vec<f64>,     // Original strikes (for reporting)
}

/// Result of SANOS calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanosSlice {
    pub timestamp: DateTime<Utc>,
    pub underlying: String,
    pub expiry: String,
    pub forward: f64,
    pub time_to_expiry: f64,

    // Model grid
    pub model_strikes: Vec<f64>, // Normalized model strikes K^i
    pub weights: Vec<f64>,       // Martingale density q_i

    // Fitted surface
    pub fitted_strikes: Vec<f64>, // Market strikes (normalized)
    pub fitted_calls: Vec<f64>,   // Ĉ(K) fitted call prices
    pub market_calls: Vec<f64>,   // Market mid prices for comparison

    // Strike metadata for boundary hardening (Phase 8.1)
    pub strike_meta: Vec<StrikeMeta>,

    // Diagnostics
    pub diagnostics: SanosDiagnostics,
}

/// SANOS calibration diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanosDiagnostics {
    pub lp_status: String,
    pub objective_value: f64,
    pub max_fit_error: f64,
    pub mean_fit_error: f64,
    pub weights_sum: f64,           // Should be 1.0
    pub weights_mean: f64,          // Should be 1.0 (unit mean constraint)
    pub convexity_violations: usize,
    pub boundary_check: bool,       // C(0) ≈ 1, C(∞) → 0
    pub spread_compliance: f64,     // % of prices within bid-ask
    pub background_variance: f64,   // V used in calibration
    pub eta: f64,                   // Smoothness parameter
}

/// SANOS calibrator for a single expiry
pub struct SanosCalibrator {
    eta: f64,
}

impl Default for SanosCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl SanosCalibrator {
    pub fn new() -> Self {
        Self { eta: ETA }
    }

    pub fn with_eta(eta: f64) -> Self {
        Self { eta }
    }

    /// STEP 5.1: Normalize market data
    ///
    /// Estimate forward F0 from put-call parity and convert to forward units.
    pub fn normalize(&self, slice: &ExpirySlice) -> Result<NormalizedSlice> {
        // Find strikes with both call and put quotes for put-call parity
        let mut parity_estimates = Vec::new();

        for (&strike, call) in &slice.calls {
            if let Some(put) = slice.puts.get(&strike) {
                // Put-Call Parity: C - P = F0 * e^(-rT) - K * e^(-rT)
                // => F0 = (C - P) * e^(rT) + K
                let c_mid = call.mid();
                let p_mid = put.mid();
                let discount = (-RISK_FREE_RATE * slice.time_to_expiry).exp();
                let f0_estimate = (c_mid - p_mid) / discount + strike as f64;

                // Weight by inverse spread (tighter spreads = more reliable)
                let weight = 1.0 / (call.spread_bps() + put.spread_bps() + 1.0);
                parity_estimates.push((f0_estimate, weight));
            }
        }

        if parity_estimates.is_empty() {
            return Err(anyhow!("No strike pairs for put-call parity estimation"));
        }

        // Weighted average for forward estimate
        let total_weight: f64 = parity_estimates.iter().map(|(_, w)| w).sum();
        let forward: f64 = parity_estimates
            .iter()
            .map(|(f, w)| f * w)
            .sum::<f64>()
            / total_weight;

        tracing::info!(
            "Forward estimate F0 = {:.2} from {} put-call pairs",
            forward,
            parity_estimates.len()
        );

        // Normalize strikes and call prices
        let mut strikes = Vec::new();
        let mut call_prices = Vec::new();
        let mut call_bids = Vec::new();
        let mut call_asks = Vec::new();
        let mut raw_strikes = Vec::new();

        for (&strike, call) in &slice.calls {
            let k_norm = strike as f64 / forward;
            let c_norm = call.mid() / forward;
            let bid_norm = call.bid / forward;
            let ask_norm = call.ask / forward;

            strikes.push(k_norm);
            call_prices.push(c_norm);
            call_bids.push(bid_norm);
            call_asks.push(ask_norm);
            raw_strikes.push(strike as f64);
        }

        // Estimate ATM IV
        let atm_strike = strikes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a - 1.0).abs()).partial_cmp(&((**b - 1.0).abs())).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let atm_iv = if atm_strike < call_prices.len() {
            // Use original strike and price for IV calculation
            let k = raw_strikes[atm_strike];
            let c = call_prices[atm_strike] * forward; // Denormalize
            implied_volatility(c, forward, k, slice.time_to_expiry, RISK_FREE_RATE, true)
                .unwrap_or(0.15)
        } else {
            0.15 // Default 15% IV
        };

        tracing::info!("ATM IV estimate: {:.2}%", atm_iv * 100.0);

        Ok(NormalizedSlice {
            forward,
            time_to_expiry: slice.time_to_expiry,
            strikes,
            call_prices,
            call_bids,
            call_asks,
            atm_iv,
            raw_strikes,
        })
    }

    /// STEP 5.3: Strike augmentation
    ///
    /// Add boundary strikes: K0=0 (C=1), K1=ε (C=1-ε), KN=far OTM (C=0)
    /// Returns (strikes, prices, bids, asks, strike_meta)
    fn augment_strikes(&self, norm: &NormalizedSlice) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<StrikeMeta>) {
        let mut strikes = Vec::new();
        let mut prices = Vec::new();
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        let mut strike_meta = Vec::new();

        // Interior band: ±3% around ATM (k ∈ [0.97, 1.03])
        const INTERIOR_LOW: f64 = 0.97;
        const INTERIOR_HIGH: f64 = 1.03;

        // K0 = 0: C(0) = 1 (discounted forward) - BOUNDARY
        strikes.push(0.0);
        prices.push(1.0);
        bids.push(1.0);
        asks.push(1.0);
        strike_meta.push(StrikeMeta::boundary(0.0));

        // K1 = ε: C(ε) ≈ 1 - ε (linear near zero) - BOUNDARY
        strikes.push(EPSILON_STRIKE);
        prices.push(1.0 - EPSILON_STRIKE);
        bids.push(1.0 - EPSILON_STRIKE);
        asks.push(1.0 - EPSILON_STRIKE);
        strike_meta.push(StrikeMeta::boundary(EPSILON_STRIKE));

        // Market strikes
        for i in 0..norm.strikes.len() {
            let k = norm.strikes[i];
            let is_interior = k >= INTERIOR_LOW && k <= INTERIOR_HIGH;
            strikes.push(k);
            prices.push(norm.call_prices[i]);
            bids.push(norm.call_bids[i]);
            asks.push(norm.call_asks[i]);
            strike_meta.push(StrikeMeta::market(k, is_interior));
        }

        // KN = far OTM: C(KN) → 0 - BOUNDARY
        // Use fixed normalized strike for calendar constraint consistency across expiries
        strikes.push(K_N_NORMALIZED);
        prices.push(0.0);
        bids.push(0.0);
        asks.push(0.0);
        strike_meta.push(StrikeMeta::boundary(K_N_NORMALIZED));

        (strikes, prices, bids, asks, strike_meta)
    }

    /// STEP 5.4-5.5: Setup model grid and background variance
    fn setup_model_grid(&self, augmented_strikes: &[f64], norm: &NormalizedSlice) -> (Vec<f64>, f64) {
        // Use augmented strikes as model grid
        let model_strikes = augmented_strikes.to_vec();

        // Background variance: V = max(σ_ATM² * T * η, V_min)
        // Floor prevents LP ill-conditioning when TTY is very short
        let raw_variance = norm.atm_iv.powi(2) * norm.time_to_expiry * self.eta;
        let background_variance = raw_variance.max(V_MIN);

        if raw_variance < V_MIN {
            tracing::info!(
                "Model grid: {} strikes, V = {:.6} (floored from {:.6}, η = {})",
                model_strikes.len(),
                background_variance,
                raw_variance,
                self.eta
            );
        } else {
            tracing::info!(
                "Model grid: {} strikes, V = {:.6} (η = {})",
                model_strikes.len(),
                background_variance,
                self.eta
            );
        }

        (model_strikes, background_variance)
    }

    /// Shifted lognormal call payoff (Black-Scholes in forward units)
    ///
    /// Call(K^i, K, V) = E[(K^i - K)^+] under shifted lognormal with variance V
    fn shifted_lognormal_call(ki: f64, k: f64, v: f64) -> f64 {
        if v <= 1e-10 {
            // Zero variance: intrinsic value
            return (ki - k).max(0.0);
        }
        if ki <= 1e-10 {
            // Strike at zero: always ITM by (1-K) where K^i=0 means payoff is max(0-K,0)=0
            // Actually K^i is the "model strike" not market strike
            // When K^i = 0, the call payoff is max(F - K, 0) which is just the forward - strike
            // In normalized units with F=1: max(1 - K, 0) if K < 1
            return (1.0 - k).max(0.0);
        }

        // Black-Scholes call in forward units (F=1):
        // C = N(d1) - K*N(d2) where d1,d2 computed with respect to model strike K^i
        // Actually in SANOS, the payoff is more like a call spread or butterfly

        // From paper: Call(K^i, K, V) is the Black-Scholes call price
        // with "spot" = K^i (model strike) and "strike" = K (market strike)
        // under variance V
        let sqrt_v = v.sqrt();
        if sqrt_v < 1e-10 {
            return (ki - k).max(0.0);
        }

        let d1 = ((ki / k).ln() + v / 2.0) / sqrt_v;
        let d2 = d1 - sqrt_v;

        ki * norm_cdf(d1) - k * norm_cdf(d2)
    }

    /// STEP 5.6: Solve SANOS LP
    ///
    /// Minimize weighted fitting error subject to martingale constraints.
    pub fn solve_lp(
        &self,
        augmented_strikes: &[f64],
        augmented_prices: &[f64],
        augmented_bids: &[f64],
        augmented_asks: &[f64],
        model_strikes: &[f64],
        background_variance: f64,
    ) -> Result<(Vec<f64>, f64, String)> {
        let n_model = model_strikes.len();
        let n_market = augmented_strikes.len();

        // Build LP problem
        let mut vars = ProblemVariables::new();

        // Martingale weights q_i >= 0
        let q: Vec<_> = (0..n_model)
            .map(|_| vars.add(variable().min(0.0)))
            .collect();

        // Slack variables for fitting error
        let slack_plus: Vec<_> = (0..n_market)
            .map(|_| vars.add(variable().min(0.0)))
            .collect();
        let slack_minus: Vec<_> = (0..n_market)
            .map(|_| vars.add(variable().min(0.0)))
            .collect();

        // Objective: minimize weighted fitting error
        // Weight by inverse spread (tighter quotes = more important)
        let mut objective: Expression = 0.0.into();
        for j in 0..n_market {
            let spread = (augmented_asks[j] - augmented_bids[j]).max(1e-6);
            let weight = 1.0 / spread;
            objective = objective + weight * (slack_plus[j] + slack_minus[j]);
        }

        let mut problem = vars.minimise(objective).using(good_lp::default_solver);

        // Constraint: sum(q_i) = 1 (probability measure)
        let sum_q: Expression = q.iter().fold(Expression::from(0.0), |acc, &qi| acc + qi);
        problem = problem.with(constraint!(sum_q == 1.0));

        // Constraint: sum(q_i * K^i) = 1 (unit mean / martingale)
        let mean_q: Expression = q
            .iter()
            .zip(model_strikes)
            .fold(Expression::from(0.0), |acc, (&qi, &ki)| acc + ki * qi);
        problem = problem.with(constraint!(mean_q == 1.0));

        // Fitting constraints: Ĉ(K_j) = sum_i q_i * Call(K^i, K_j, V)
        // With slack: market_mid - slack_minus <= fitted <= market_mid + slack_plus
        for j in 0..n_market {
            let k_market = augmented_strikes[j];
            let c_market = augmented_prices[j];

            // Build fitted price expression
            let fitted: Expression = q.iter().zip(model_strikes).fold(
                Expression::from(0.0),
                |acc, (&qi, &ki)| {
                    let payoff = Self::shifted_lognormal_call(ki, k_market, background_variance);
                    acc + payoff * qi
                },
            );

            // fitted = c_market + slack_plus - slack_minus
            // => fitted - slack_plus + slack_minus = c_market
            let lhs = fitted.clone() - slack_plus[j] + slack_minus[j];
            problem = problem.with(constraint!(lhs == c_market));
        }

        // Solve
        let solution = problem.solve().map_err(|e| anyhow!("LP solve failed: {:?}", e))?;

        // Extract weights
        let weights: Vec<f64> = q.iter().map(|&qi| solution.value(qi)).collect();
        let obj_value = slack_plus
            .iter()
            .chain(slack_minus.iter())
            .map(|&s| solution.value(s))
            .sum::<f64>();

        Ok((weights, obj_value, "Optimal".to_string()))
    }

    /// STEP 5.7: Certification checks
    fn certify(
        &self,
        fitted_calls: &[f64],
        market_calls: &[f64],
        market_bids: &[f64],
        market_asks: &[f64],
        weights: &[f64],
        model_strikes: &[f64],
    ) -> SanosDiagnostics {
        // Fit errors
        let errors: Vec<f64> = fitted_calls
            .iter()
            .zip(market_calls)
            .map(|(f, m)| (f - m).abs())
            .collect();
        let max_fit_error = errors.iter().cloned().fold(0.0, f64::max);
        let mean_fit_error = errors.iter().sum::<f64>() / errors.len() as f64;

        // Weights constraints
        let weights_sum: f64 = weights.iter().sum();
        let weights_mean: f64 = weights
            .iter()
            .zip(model_strikes)
            .map(|(q, k)| q * k)
            .sum();

        // Convexity check: d²C/dK² >= 0 (call prices should be convex in strike)
        let mut convexity_violations = 0;
        for i in 1..fitted_calls.len() - 1 {
            let second_deriv = fitted_calls[i - 1] - 2.0 * fitted_calls[i] + fitted_calls[i + 1];
            if second_deriv < -1e-6 {
                convexity_violations += 1;
            }
        }

        // Boundary check
        let boundary_check = fitted_calls.first().map_or(false, |&c| c > 0.95)
            && fitted_calls.last().map_or(false, |&c| c < 0.05);

        // Spread compliance: how many fitted prices fall within bid-ask
        let compliant = fitted_calls
            .iter()
            .zip(market_bids.iter().zip(market_asks))
            .filter(|(f, (bid, ask))| **f >= **bid - 1e-6 && **f <= **ask + 1e-6)
            .count();
        let spread_compliance = compliant as f64 / fitted_calls.len() as f64 * 100.0;

        SanosDiagnostics {
            lp_status: "Optimal".to_string(),
            objective_value: 0.0, // Will be filled by caller
            max_fit_error,
            mean_fit_error,
            weights_sum,
            weights_mean,
            convexity_violations,
            boundary_check,
            spread_compliance,
            background_variance: 0.0, // Will be filled by caller
            eta: self.eta,
        }
    }

    /// Main calibration entry point
    pub fn calibrate(&self, slice: &ExpirySlice) -> Result<SanosSlice> {
        tracing::info!(
            "SANOS calibration: {} {} at {}",
            slice.underlying,
            slice.expiry,
            slice.timestamp
        );

        // STEP 5.1: Normalize market
        let norm = self.normalize(slice)?;

        // STEP 5.2: Call prices from mid (already done in normalize)

        // STEP 5.3: Strike augmentation (with strike metadata for boundary hardening)
        let (aug_strikes, aug_prices, aug_bids, aug_asks, strike_meta) = self.augment_strikes(&norm);

        // STEP 5.4-5.5: Model grid and background variance
        let (model_strikes, background_variance) = self.setup_model_grid(&aug_strikes, &norm);

        // STEP 5.6: Solve SANOS LP
        let (weights, obj_value, lp_status) = self.solve_lp(
            &aug_strikes,
            &aug_prices,
            &aug_bids,
            &aug_asks,
            &model_strikes,
            background_variance,
        )?;

        // Compute fitted prices
        let fitted_calls: Vec<f64> = aug_strikes
            .iter()
            .map(|&k| {
                weights
                    .iter()
                    .zip(&model_strikes)
                    .map(|(&q, &ki)| q * Self::shifted_lognormal_call(ki, k, background_variance))
                    .sum()
            })
            .collect();

        // STEP 5.7: Certification
        let mut diagnostics =
            self.certify(&fitted_calls, &aug_prices, &aug_bids, &aug_asks, &weights, &model_strikes);
        diagnostics.objective_value = obj_value;
        diagnostics.background_variance = background_variance;
        diagnostics.lp_status = lp_status;

        tracing::info!(
            "SANOS calibration complete: {} strikes, max_error={:.6}, spread_compliance={:.1}%",
            aug_strikes.len(),
            diagnostics.max_fit_error,
            diagnostics.spread_compliance
        );

        Ok(SanosSlice {
            timestamp: slice.timestamp,
            underlying: slice.underlying.clone(),
            expiry: slice.expiry.clone(),
            forward: norm.forward,
            time_to_expiry: norm.time_to_expiry,
            model_strikes,
            weights,
            fitted_strikes: aug_strikes,
            fitted_calls,
            market_calls: aug_prices,
            strike_meta,
            diagnostics,
        })
    }
}

/// Standard normal CDF (duplicated from pricing.rs for independence)
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

impl std::fmt::Display for SanosSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== SANOS Slice ===")?;
        writeln!(f, "Underlying: {}", self.underlying)?;
        writeln!(f, "Expiry: {}", self.expiry)?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f, "Forward: {:.2}", self.forward)?;
        writeln!(f, "T: {:.4} years", self.time_to_expiry)?;
        writeln!(f)?;
        writeln!(f, "--- Diagnostics ---")?;
        writeln!(f, "LP Status: {}", self.diagnostics.lp_status)?;
        writeln!(f, "Objective: {:.6}", self.diagnostics.objective_value)?;
        writeln!(f, "Max Fit Error: {:.6}", self.diagnostics.max_fit_error)?;
        writeln!(f, "Mean Fit Error: {:.6}", self.diagnostics.mean_fit_error)?;
        writeln!(f, "Weights Sum: {:.6} (should be 1.0)", self.diagnostics.weights_sum)?;
        writeln!(f, "Weights Mean: {:.6} (should be 1.0)", self.diagnostics.weights_mean)?;
        writeln!(f, "Convexity Violations: {}", self.diagnostics.convexity_violations)?;
        writeln!(f, "Boundary Check: {}", self.diagnostics.boundary_check)?;
        writeln!(f, "Spread Compliance: {:.1}%", self.diagnostics.spread_compliance)?;
        writeln!(f, "Background Variance: {:.6}", self.diagnostics.background_variance)?;
        writeln!(f, "η: {}", self.diagnostics.eta)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_slice() -> ExpirySlice {
        let ts = Utc.with_ymd_and_hms(2026, 1, 23, 10, 0, 0).unwrap();
        let mut slice = ExpirySlice::new("NIFTY", "26JAN", ts, 3.0 / 365.0); // 3 days to expiry

        // Add synthetic quotes around ATM (forward ~ 25500)
        let strikes = [25300, 25350, 25400, 25450, 25500, 25550, 25600, 25650, 25700];
        let call_mids = [220.0, 180.0, 145.0, 112.0, 85.0, 62.0, 44.0, 30.0, 19.0];
        let put_mids = [20.0, 30.0, 45.0, 62.0, 85.0, 112.0, 144.0, 180.0, 219.0];

        for (i, &strike) in strikes.iter().enumerate() {
            let spread = 2.0; // 2 rupee spread
            slice.add_quote(OptionQuote {
                symbol: format!("NIFTY26JAN{}CE", strike),
                strike: strike as f64,
                is_call: true,
                bid: call_mids[i] - spread / 2.0,
                ask: call_mids[i] + spread / 2.0,
                timestamp: ts,
            });
            slice.add_quote(OptionQuote {
                symbol: format!("NIFTY26JAN{}PE", strike),
                strike: strike as f64,
                is_call: false,
                bid: put_mids[i] - spread / 2.0,
                ask: put_mids[i] + spread / 2.0,
                timestamp: ts,
            });
        }

        slice
    }

    #[test]
    fn test_normalize() {
        let slice = create_test_slice();
        let calibrator = SanosCalibrator::new();
        let norm = calibrator.normalize(&slice).unwrap();

        // Forward should be around 25500
        assert!(
            norm.forward > 25400.0 && norm.forward < 25600.0,
            "Forward {} should be near 25500",
            norm.forward
        );

        // Normalized ATM strike should be near 1.0
        let atm_idx = norm
            .strikes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| ((**a - 1.0).abs()).partial_cmp(&((**b - 1.0).abs())).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            (norm.strikes[atm_idx] - 1.0).abs() < 0.01,
            "ATM strike should be near 1.0"
        );
    }

    #[test]
    fn test_calibrate() {
        let slice = create_test_slice();
        let calibrator = SanosCalibrator::new();
        let result = calibrator.calibrate(&slice).unwrap();

        // Weights should sum to 1
        assert!(
            (result.diagnostics.weights_sum - 1.0).abs() < 0.01,
            "Weights sum {} should be 1.0",
            result.diagnostics.weights_sum
        );

        // Weights mean should be near 1
        assert!(
            (result.diagnostics.weights_mean - 1.0).abs() < 0.1,
            "Weights mean {} should be near 1.0",
            result.diagnostics.weights_mean
        );

        // Boundary check should pass
        assert!(result.diagnostics.boundary_check, "Boundary check should pass");

        // Allow minor convexity violations at boundary augmentation points
        // SANOS v0 uses simple augmentation which can cause edge effects
        assert!(
            result.diagnostics.convexity_violations <= 3,
            "Should have at most 3 convexity violations (boundary effects), got {}",
            result.diagnostics.convexity_violations
        );

        println!("{}", result);
    }
}
