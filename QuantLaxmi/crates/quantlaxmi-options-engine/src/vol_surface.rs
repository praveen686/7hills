//! Volatility Surface Analysis
//!
//! Builds and analyzes the implied volatility surface for edge detection:
//! - IV calculation via Newton-Raphson
//! - Surface interpolation
//! - Skew and smile analysis
//! - Term structure analysis
//! - IV percentile for strategy selection

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::f64::consts::E;

use crate::greeks::{Greeks, OptionParams, OptionType};

/// IV calculation result.
#[derive(Debug, Clone, Copy)]
pub struct IVResult {
    pub iv: f64,
    pub iterations: u32,
    pub converged: bool,
}

/// Calculate implied volatility using Newton-Raphson.
///
/// # Arguments
/// * `market_price` - Observed option price
/// * `spot` - Underlying spot price
/// * `strike` - Option strike
/// * `time_to_expiry` - Time to expiry in years
/// * `risk_free_rate` - Risk-free rate
/// * `option_type` - Call or Put
///
/// # Returns
/// Implied volatility or None if doesn't converge.
pub fn calculate_iv(
    market_price: f64,
    spot: f64,
    strike: f64,
    time_to_expiry: f64,
    risk_free_rate: f64,
    option_type: OptionType,
) -> Option<IVResult> {
    const MAX_ITERATIONS: u32 = 100;
    const TOLERANCE: f64 = 1e-8;
    const MIN_IV: f64 = 0.001;
    const MAX_IV: f64 = 5.0;

    // Initial guess based on price/spot ratio
    let mut iv = (market_price / spot * 2.5).max(0.15).min(1.0);

    for i in 0..MAX_ITERATIONS {
        let params = OptionParams::new(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            iv,
            option_type,
        );
        let greeks = Greeks::calculate(&params);

        let price_diff = greeks.price - market_price;

        // Vega in terms of σ (not %)
        let vega_sigma = greeks.vega * 100.0;

        if vega_sigma.abs() < 1e-10 {
            break; // Vega too small, can't iterate
        }

        let iv_adjustment = price_diff / vega_sigma;
        iv -= iv_adjustment;

        // Clamp to valid range
        iv = iv.max(MIN_IV).min(MAX_IV);

        if price_diff.abs() < TOLERANCE {
            return Some(IVResult {
                iv,
                iterations: i + 1,
                converged: true,
            });
        }
    }

    // Return last guess even if not fully converged
    Some(IVResult {
        iv,
        iterations: MAX_ITERATIONS,
        converged: false,
    })
}

/// A point on the volatility surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VolPoint {
    /// Strike price
    pub strike: f64,
    /// Time to expiry in years
    pub expiry: f64,
    /// Implied volatility
    pub iv: f64,
    /// Option type used for calculation
    pub option_type: OptionType,
    /// Moneyness: ln(K/S)
    pub moneyness: f64,
    /// Delta of the option
    pub delta: f64,
}

/// Volatility smile at a single expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolSmile {
    /// Expiry in years
    pub expiry: f64,
    /// Days to expiry
    pub dte: u32,
    /// ATM volatility
    pub atm_iv: f64,
    /// Points sorted by strike
    pub points: Vec<VolPoint>,
    /// Skew: 25Δ put IV - 25Δ call IV
    pub skew_25d: f64,
    /// Butterfly: (25Δ put IV + 25Δ call IV) / 2 - ATM IV
    pub butterfly_25d: f64,
    /// Risk reversal: 25Δ call IV - 25Δ put IV
    pub risk_reversal_25d: f64,
}

impl VolSmile {
    /// Build a smile from option quotes.
    pub fn from_quotes(
        spot: f64,
        expiry: f64,
        risk_free_rate: f64,
        quotes: &[(f64, f64, OptionType)], // (strike, price, type)
    ) -> Self {
        let dte = (expiry * 365.0).round() as u32;
        let mut points = Vec::new();

        for &(strike, price, opt_type) in quotes {
            if let Some(iv_result) =
                calculate_iv(price, spot, strike, expiry, risk_free_rate, opt_type)
            {
                if iv_result.converged || iv_result.iv > 0.01 {
                    let params = OptionParams::new(
                        spot,
                        strike,
                        expiry,
                        risk_free_rate,
                        iv_result.iv,
                        opt_type,
                    );
                    let greeks = Greeks::calculate(&params);

                    points.push(VolPoint {
                        strike,
                        expiry,
                        iv: iv_result.iv,
                        option_type: opt_type,
                        moneyness: (strike / spot).ln(),
                        delta: greeks.delta,
                    });
                }
            }
        }

        // Sort by strike
        points.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap());

        // Calculate ATM IV (interpolate to spot)
        let atm_iv = Self::interpolate_iv_at_strike(&points, spot);

        // Calculate skew metrics
        let (skew_25d, butterfly_25d, risk_reversal_25d) = Self::calculate_skew_metrics(&points);

        VolSmile {
            expiry,
            dte,
            atm_iv,
            points,
            skew_25d,
            butterfly_25d,
            risk_reversal_25d,
        }
    }

    fn interpolate_iv_at_strike(points: &[VolPoint], target_strike: f64) -> f64 {
        if points.is_empty() {
            return 0.0;
        }
        if points.len() == 1 {
            return points[0].iv;
        }

        // Find bracketing points
        for i in 0..points.len() - 1 {
            if points[i].strike <= target_strike && points[i + 1].strike >= target_strike {
                let t =
                    (target_strike - points[i].strike) / (points[i + 1].strike - points[i].strike);
                return points[i].iv + t * (points[i + 1].iv - points[i].iv);
            }
        }

        // Extrapolate from nearest
        if target_strike < points[0].strike {
            points[0].iv
        } else {
            points.last().unwrap().iv
        }
    }

    fn calculate_skew_metrics(points: &[VolPoint]) -> (f64, f64, f64) {
        // Find 25Δ put and 25Δ call
        let mut put_25d_iv = None;
        let mut call_25d_iv = None;
        let mut atm_iv = None;

        for p in points {
            match p.option_type {
                OptionType::Put => {
                    if p.delta.abs() > 0.20 && p.delta.abs() < 0.30 {
                        put_25d_iv = Some(p.iv);
                    }
                    if p.delta.abs() > 0.45 && p.delta.abs() < 0.55 {
                        atm_iv = Some(p.iv);
                    }
                }
                OptionType::Call => {
                    if p.delta > 0.20 && p.delta < 0.30 {
                        call_25d_iv = Some(p.iv);
                    }
                    if p.delta > 0.45 && p.delta < 0.55 {
                        atm_iv = atm_iv.or(Some(p.iv));
                    }
                }
            }
        }

        let put_iv = put_25d_iv.unwrap_or(0.0);
        let call_iv = call_25d_iv.unwrap_or(0.0);
        let atm = atm_iv.unwrap_or(0.0);

        let skew = put_iv - call_iv;
        let butterfly = if put_iv > 0.0 && call_iv > 0.0 && atm > 0.0 {
            (put_iv + call_iv) / 2.0 - atm
        } else {
            0.0
        };
        let risk_reversal = call_iv - put_iv;

        (skew, butterfly, risk_reversal)
    }

    /// Get IV at specific delta.
    pub fn iv_at_delta(&self, target_delta: f64) -> Option<f64> {
        // Find closest delta
        let mut best = None;
        let mut best_diff = f64::MAX;

        for p in &self.points {
            let diff = (p.delta - target_delta).abs();
            if diff < best_diff {
                best_diff = diff;
                best = Some(p.iv);
            }
        }

        best
    }
}

/// Complete volatility surface across strikes and expiries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolSurface {
    /// Underlying spot price
    pub spot: f64,
    /// Smiles keyed by DTE
    pub smiles: BTreeMap<u32, VolSmile>,
    /// Historical ATM IV for percentile calculation
    pub iv_history: Vec<f64>,
    /// Current ATM IV (shortest expiry)
    pub current_atm_iv: f64,
    /// IV percentile (0-100)
    pub iv_percentile: f64,
}

impl VolSurface {
    /// Create empty surface.
    pub fn new(spot: f64) -> Self {
        Self {
            spot,
            smiles: BTreeMap::new(),
            iv_history: Vec::new(),
            current_atm_iv: 0.0,
            iv_percentile: 50.0,
        }
    }

    /// Add a smile to the surface.
    pub fn add_smile(&mut self, smile: VolSmile) {
        if smile.dte > 0 && smile.dte <= 7 {
            self.current_atm_iv = smile.atm_iv;
        }
        self.smiles.insert(smile.dte, smile);
    }

    /// Update IV history for percentile calculation.
    pub fn update_iv_history(&mut self, atm_iv: f64, max_history: usize) {
        self.iv_history.push(atm_iv);
        if self.iv_history.len() > max_history {
            self.iv_history.remove(0);
        }
        self.calculate_percentile();
    }

    fn calculate_percentile(&mut self) {
        if self.iv_history.len() < 10 {
            self.iv_percentile = 50.0;
            return;
        }

        let count_below = self
            .iv_history
            .iter()
            .filter(|&&iv| iv < self.current_atm_iv)
            .count();

        self.iv_percentile = (count_below as f64 / self.iv_history.len() as f64) * 100.0;
    }

    /// Get interpolated IV at any strike and DTE.
    pub fn iv_at(&self, strike: f64, dte: u32) -> Option<f64> {
        // Find bracketing expiries
        let dtes: Vec<u32> = self.smiles.keys().copied().collect();
        if dtes.is_empty() {
            return None;
        }

        // Find bracketing DTEs
        let mut lower_dte = None;
        let mut upper_dte = None;

        for &d in &dtes {
            if d <= dte {
                lower_dte = Some(d);
            }
            if d >= dte && upper_dte.is_none() {
                upper_dte = Some(d);
            }
        }

        match (lower_dte, upper_dte) {
            (Some(l), Some(u)) if l == u => {
                // Exact DTE match
                self.smiles
                    .get(&l)
                    .map(|s| VolSmile::interpolate_iv_at_strike(&s.points, strike))
            }
            (Some(l), Some(u)) => {
                // Interpolate between DTEs
                let lower_smile = self.smiles.get(&l)?;
                let upper_smile = self.smiles.get(&u)?;

                let lower_iv = VolSmile::interpolate_iv_at_strike(&lower_smile.points, strike);
                let upper_iv = VolSmile::interpolate_iv_at_strike(&upper_smile.points, strike);

                let t = (dte - l) as f64 / (u - l) as f64;
                Some(lower_iv + t * (upper_iv - lower_iv))
            }
            (Some(l), None) => {
                // Extrapolate from lower
                self.smiles
                    .get(&l)
                    .map(|s| VolSmile::interpolate_iv_at_strike(&s.points, strike))
            }
            (None, Some(u)) => {
                // Extrapolate from upper
                self.smiles
                    .get(&u)
                    .map(|s| VolSmile::interpolate_iv_at_strike(&s.points, strike))
            }
            _ => None,
        }
    }

    /// Get term structure (ATM IV across expiries).
    pub fn term_structure(&self) -> Vec<(u32, f64)> {
        self.smiles
            .iter()
            .map(|(&dte, smile)| (dte, smile.atm_iv))
            .collect()
    }

    /// Check if term structure is in contango (upward sloping).
    pub fn is_contango(&self) -> bool {
        let ts = self.term_structure();
        if ts.len() < 2 {
            return false;
        }
        ts.windows(2).all(|w| w[1].1 >= w[0].1)
    }

    /// Check if term structure is in backwardation (downward sloping).
    pub fn is_backwardation(&self) -> bool {
        let ts = self.term_structure();
        if ts.len() < 2 {
            return false;
        }
        ts.windows(2).all(|w| w[1].1 <= w[0].1)
    }

    /// Get average skew across expiries.
    pub fn average_skew(&self) -> f64 {
        let skews: Vec<f64> = self
            .smiles
            .values()
            .filter(|s| s.skew_25d.abs() > 0.001)
            .map(|s| s.skew_25d)
            .collect();

        if skews.is_empty() {
            0.0
        } else {
            skews.iter().sum::<f64>() / skews.len() as f64
        }
    }
}

/// Volatility regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolRegime {
    /// IV percentile < 25: Low volatility, buy premium
    LowVol,
    /// IV percentile 25-75: Normal volatility
    NormalVol,
    /// IV percentile > 75: High volatility, sell premium
    HighVol,
    /// IV percentile > 90: Extreme volatility
    ExtremeVol,
}

impl VolRegime {
    /// Classify regime from IV percentile.
    pub fn from_percentile(percentile: f64) -> Self {
        if percentile > 90.0 {
            VolRegime::ExtremeVol
        } else if percentile > 75.0 {
            VolRegime::HighVol
        } else if percentile < 25.0 {
            VolRegime::LowVol
        } else {
            VolRegime::NormalVol
        }
    }

    /// Get strategy bias for this regime.
    pub fn strategy_bias(&self) -> StrategyBias {
        match self {
            VolRegime::LowVol => StrategyBias::BuyPremium,
            VolRegime::NormalVol => StrategyBias::Neutral,
            VolRegime::HighVol => StrategyBias::SellPremium,
            VolRegime::ExtremeVol => StrategyBias::SellPremium,
        }
    }
}

/// Strategy bias based on volatility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyBias {
    /// Buy options (long vega, long gamma)
    BuyPremium,
    /// Sell options (short vega, short gamma)
    SellPremium,
    /// No strong bias
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_calculation() {
        // First calculate the actual BS price for these params
        let params =
            crate::greeks::OptionParams::new(100.0, 100.0, 0.25, 0.05, 0.20, OptionType::Call);
        let greeks = crate::greeks::Greeks::calculate(&params);
        let bs_price = greeks.price;

        // Now recover IV from that price
        let result = calculate_iv(bs_price, 100.0, 100.0, 0.25, 0.05, OptionType::Call);

        assert!(result.is_some());
        let iv = result.unwrap();
        assert!(iv.converged);
        assert!((iv.iv - 0.20).abs() < 0.02, "IV={} expected ~0.20", iv.iv);
    }

    #[test]
    fn test_vol_regime() {
        assert_eq!(VolRegime::from_percentile(10.0), VolRegime::LowVol);
        assert_eq!(VolRegime::from_percentile(50.0), VolRegime::NormalVol);
        assert_eq!(VolRegime::from_percentile(80.0), VolRegime::HighVol);
        assert_eq!(VolRegime::from_percentile(95.0), VolRegime::ExtremeVol);
    }
}
