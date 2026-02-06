//! SANOS Calendar Carry Adapter
//!
//! Core conversion: `OptionsSnapshot` → SANOS pipeline → `StrategyDecision`
//!
//! Pipeline per `evaluate()` call:
//! 1. Group `OptQuote`s by expiry
//! 2. Convert to `ExpirySlice` per expiry
//! 3. `SanosCalibrator.calibrate()` on each expiry
//! 4. Extract `Phase8Features` (IV, calendar gap, skew, forwards)
//! 5. Build `StraddleQuotes` at ATM for front/back
//! 6. Build `StrategyContext`, call `CalendarCarryStrategy.evaluate()`
//!
//! Calibration is throttled to every 60s (FROZEN_PARAMS interval).
//! Between calibrations, cached features are reused and only quotes update.

use chrono::{DateTime, Duration, Timelike, Utc};
use tracing::{debug, info, warn};

use quantlaxmi_options::sanos::{ExpirySlice, OptionQuote, SanosCalibrator, SanosSlice};
use quantlaxmi_options::strategies::{
    CalendarCarryStrategy, FROZEN_PARAMS, FrozenParams, GateCheckResult, Phase8Features,
    QuoteSnapshot, SessionMeta, StraddleQuotes, StrategyContext,
    StrategyDecision as SanosStrategyDecision,
};

use crate::paper::snapshot::{OptionsSnapshot, Right};

use super::tui_state::{SanosFeatureView, SanosSurfaceView};

/// NSE market hours (IST = UTC + 5:30).
const MARKET_CLOSE_HOUR: u32 = 15;
const MARKET_CLOSE_MINUTE: u32 = 30;

/// Result of a single adapter evaluation.
pub struct AdapterResult {
    /// Strategy decision from CalendarCarryStrategy.
    pub decision: SanosStrategyDecision,
    /// Gate check results (for TUI).
    pub gates: Option<GateCheckResult>,
    /// Surface views per expiry (for TUI).
    pub surfaces: Vec<SanosSurfaceView>,
    /// Feature view (for TUI).
    pub feature_view: Option<SanosFeatureView>,
    /// Last calibration timestamp.
    pub last_calibration_ts: Option<DateTime<Utc>>,
    /// Whether this tick used cached (vs fresh) calibration.
    pub used_cache: bool,
}

/// SANOS Calendar Carry Adapter.
///
/// Converts live `OptionsSnapshot` into SANOS-calibrated strategy decisions
/// with a configurable calibration throttle.
pub struct SanosCalendarCarryAdapter {
    /// SANOS calibrator.
    calibrator: SanosCalibrator,
    /// Calendar carry strategy (frozen params).
    strategy: CalendarCarryStrategy,
    /// Underlying symbol.
    underlying: String,
    /// Lot size for the underlying.
    lot_size: u32,
    /// Calibration interval in seconds.
    calibration_interval_secs: u64,
    /// Last calibration timestamp.
    last_calibration_ts: Option<DateTime<Utc>>,
    /// Cached SANOS slices from last calibration.
    cached_slices: Vec<SanosSlice>,
    /// Cached features from last calibration.
    cached_features: Option<Phase8Features>,
    /// Cached surface views from last calibration.
    cached_surfaces: Vec<SanosSurfaceView>,
    /// Sorted expiry strings from snapshot (T1, T2, T3).
    expiry_order: Vec<String>,
}

impl SanosCalendarCarryAdapter {
    /// Create a new adapter.
    ///
    /// When `relax_e_gates` is true the E1/E2/E3 economic hardener thresholds
    /// are set to extreme negative values so that any real-market gap passes,
    /// allowing the full trade pipeline to be exercised end-to-end.
    pub fn new(
        underlying: &str,
        lot_size: u32,
        calibration_interval_secs: u64,
        relax_e_gates: bool,
    ) -> Self {
        let strategy = if relax_e_gates {
            CalendarCarryStrategy::with_params(make_relaxed_params())
        } else {
            CalendarCarryStrategy::new()
        };
        Self {
            calibrator: SanosCalibrator::new(),
            strategy,
            underlying: underlying.to_string(),
            lot_size,
            calibration_interval_secs,
            last_calibration_ts: None,
            cached_slices: Vec::new(),
            cached_features: None,
            cached_surfaces: Vec::new(),
            expiry_order: Vec::new(),
        }
    }

    /// Evaluate a snapshot, returning a strategy decision and diagnostics.
    pub fn evaluate(&mut self, ts: DateTime<Utc>, snapshot: &OptionsSnapshot) -> AdapterResult {
        // 1. Group quotes by expiry
        let expiries = self.discover_expiries(snapshot);
        if expiries.len() < 2 {
            return AdapterResult {
                decision: SanosStrategyDecision::NoTrade {
                    reason: format!("Need at least 2 expiries, found {}", expiries.len()),
                    gates: empty_gates(),
                },
                gates: None,
                surfaces: Vec::new(),
                feature_view: None,
                last_calibration_ts: self.last_calibration_ts,
                used_cache: false,
            };
        }
        self.expiry_order = expiries;

        // 2. Check if we need to re-calibrate
        let needs_calibration = self.needs_calibration(ts);

        let (features, surfaces, used_cache) = if needs_calibration {
            match self.run_calibration(ts, snapshot) {
                Ok((feat, surf)) => (feat, surf, false),
                Err(e) => {
                    warn!(error = %e, "[SANOS-ADAPTER] Calibration failed, using cache");
                    match &self.cached_features {
                        Some(f) => (f.clone(), self.cached_surfaces.clone(), true),
                        None => {
                            return AdapterResult {
                                decision: SanosStrategyDecision::NoTrade {
                                    reason: format!("Calibration failed and no cache: {}", e),
                                    gates: empty_gates(),
                                },
                                gates: None,
                                surfaces: Vec::new(),
                                feature_view: None,
                                last_calibration_ts: self.last_calibration_ts,
                                used_cache: false,
                            };
                        }
                    }
                }
            }
        } else {
            match &self.cached_features {
                Some(f) => (f.clone(), self.cached_surfaces.clone(), true),
                None => {
                    return AdapterResult {
                        decision: SanosStrategyDecision::NoTrade {
                            reason: "No cached features available".to_string(),
                            gates: empty_gates(),
                        },
                        gates: None,
                        surfaces: Vec::new(),
                        feature_view: None,
                        last_calibration_ts: self.last_calibration_ts,
                        used_cache: false,
                    };
                }
            }
        };

        // 3. Build straddle quotes from live snapshot (always fresh)
        let atm_strike = self.find_atm_strike(&features);
        let front_expiry = &self.expiry_order[0].clone();
        let back_expiry = &self.expiry_order[1].clone();

        let front_straddle =
            match self.build_straddle_from_snapshot(snapshot, front_expiry, atm_strike, ts) {
                Some(s) => s,
                None => {
                    return AdapterResult {
                        decision: SanosStrategyDecision::NoTrade {
                            reason: format!(
                                "Cannot build front straddle at strike {:.0} expiry {}",
                                atm_strike, front_expiry
                            ),
                            gates: empty_gates(),
                        },
                        gates: None,
                        surfaces,
                        feature_view: Some(features_to_view(&features)),
                        last_calibration_ts: self.last_calibration_ts,
                        used_cache,
                    };
                }
            };

        let back_straddle =
            match self.build_straddle_from_snapshot(snapshot, back_expiry, atm_strike, ts) {
                Some(s) => s,
                None => {
                    return AdapterResult {
                        decision: SanosStrategyDecision::NoTrade {
                            reason: format!(
                                "Cannot build back straddle at strike {:.0} expiry {}",
                                atm_strike, back_expiry
                            ),
                            gates: empty_gates(),
                        },
                        gates: None,
                        surfaces,
                        feature_view: Some(features_to_view(&features)),
                        last_calibration_ts: self.last_calibration_ts,
                        used_cache,
                    };
                }
            };

        // 4. Build strategy context
        let meta = SessionMeta {
            underlying: self.underlying.clone(),
            t1_expiry: self.expiry_order[0].clone(),
            t2_expiry: self.expiry_order.get(1).cloned(),
            t3_expiry: self.expiry_order.get(2).cloned(),
            lot_size: self.lot_size,
            multiplier: 1.0,
            lp_status_t1: surfaces
                .first()
                .map(|s| s.lp_status.clone())
                .unwrap_or_else(|| "Unknown".to_string()),
            lp_status_t2: surfaces.get(1).map(|s| s.lp_status.clone()),
            lp_status_t3: surfaces.get(2).map(|s| s.lp_status.clone()),
        };

        let minutes_to_close = self.minutes_to_close(ts);
        let is_expiry_day_front = self.is_expiry_day(&self.expiry_order[0], ts);

        let ctx = StrategyContext {
            ts,
            features: features.clone(),
            front_straddle,
            back_straddle,
            meta,
            minutes_to_close,
            is_expiry_day_front,
        };

        // 5. Evaluate strategy
        let (decision, _audit) = self.strategy.evaluate(&ctx);

        // Extract gates from decision for TUI
        let gates = match &decision {
            SanosStrategyDecision::NoTrade { gates, .. } => Some(gates.clone()),
            SanosStrategyDecision::Enter { gates, .. } => Some(gates.clone()),
            _ => None,
        };

        let feature_view = Some(features_to_view(&features));

        AdapterResult {
            decision,
            gates,
            surfaces,
            feature_view,
            last_calibration_ts: self.last_calibration_ts,
            used_cache,
        }
    }

    /// Discover sorted expiries from snapshot quotes.
    fn discover_expiries(&self, snapshot: &OptionsSnapshot) -> Vec<String> {
        let mut expiry_set = std::collections::BTreeSet::new();
        for q in &snapshot.quotes {
            expiry_set.insert(q.expiry.clone());
        }
        expiry_set.into_iter().collect()
    }

    /// Check if calibration is needed.
    fn needs_calibration(&self, ts: DateTime<Utc>) -> bool {
        match self.last_calibration_ts {
            None => true,
            Some(last) => {
                let elapsed = (ts - last).num_seconds();
                elapsed >= self.calibration_interval_secs as i64
            }
        }
    }

    /// Run SANOS calibration on all available expiries.
    fn run_calibration(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &OptionsSnapshot,
    ) -> anyhow::Result<(Phase8Features, Vec<SanosSurfaceView>)> {
        let mut slices = Vec::new();
        let mut surfaces = Vec::new();

        for expiry in &self.expiry_order {
            // Build ExpirySlice from snapshot
            let tty = self.compute_time_to_expiry(ts, expiry);
            let expiry_slice = self.build_expiry_slice(snapshot, expiry, ts, tty);

            if expiry_slice.calls.is_empty() {
                debug!(expiry = %expiry, "[SANOS-ADAPTER] Skipping expiry with no calls");
                continue;
            }

            // Calibrate
            match self.calibrator.calibrate(&expiry_slice) {
                Ok(sanos_slice) => {
                    surfaces.push(SanosSurfaceView {
                        expiry: expiry.clone(),
                        lp_status: sanos_slice.diagnostics.lp_status.clone(),
                        max_fit_error: sanos_slice.diagnostics.max_fit_error,
                        mean_fit_error: sanos_slice.diagnostics.mean_fit_error,
                        spread_compliance: sanos_slice.diagnostics.spread_compliance,
                        forward: sanos_slice.forward,
                        time_to_expiry: sanos_slice.time_to_expiry,
                        background_variance: sanos_slice.diagnostics.background_variance,
                    });
                    slices.push(sanos_slice);
                }
                Err(e) => {
                    warn!(expiry = %expiry, error = %e, "[SANOS-ADAPTER] Calibration failed for expiry");
                    surfaces.push(SanosSurfaceView {
                        expiry: expiry.clone(),
                        lp_status: format!("FAILED: {}", e),
                        max_fit_error: f64::NAN,
                        mean_fit_error: f64::NAN,
                        spread_compliance: 0.0,
                        forward: 0.0,
                        time_to_expiry: 0.0,
                        background_variance: 0.0,
                    });
                }
            }
        }

        if slices.is_empty() {
            return Err(anyhow::anyhow!("No expiries calibrated successfully"));
        }

        // Build features from slices
        let features = build_features(&slices)?;

        // Cache results
        self.last_calibration_ts = Some(ts);
        self.cached_slices = slices;
        self.cached_features = Some(features.clone());
        self.cached_surfaces = surfaces.clone();

        info!(
            expiry_count = surfaces.len(),
            iv1 = features.iv1,
            iv2 = ?features.iv2,
            cal12 = ?features.cal12,
            "[SANOS-ADAPTER] Calibration complete"
        );

        Ok((features, surfaces))
    }

    /// Build an ExpirySlice from snapshot quotes for a given expiry.
    fn build_expiry_slice(
        &self,
        snapshot: &OptionsSnapshot,
        expiry: &str,
        ts: DateTime<Utc>,
        tty: f64,
    ) -> ExpirySlice {
        let mut slice = ExpirySlice::new(&self.underlying, expiry, ts, tty);

        for q in &snapshot.quotes {
            if q.expiry != expiry {
                continue;
            }

            let bid = q.bid.map(|b| b.price).unwrap_or(0.0);
            let ask = q.ask.map(|a| a.price).unwrap_or(0.0);

            if bid <= 0.0 || ask <= 0.0 || ask < bid {
                continue;
            }

            let quote = OptionQuote {
                symbol: q.tradingsymbol.clone(),
                strike: q.strike as f64,
                is_call: q.right == Right::Call,
                bid,
                ask,
                timestamp: ts,
            };

            slice.add_quote(quote);
        }

        slice
    }

    /// Build StraddleQuotes from live snapshot at ATM strike.
    fn build_straddle_from_snapshot(
        &self,
        snapshot: &OptionsSnapshot,
        expiry: &str,
        atm_strike: f64,
        ts: DateTime<Utc>,
    ) -> Option<StraddleQuotes> {
        let strike_i32 = atm_strike as i32;

        let ce = snapshot
            .quotes
            .iter()
            .find(|q| q.expiry == expiry && q.strike == strike_i32 && q.right == Right::Call)?;
        let pe = snapshot
            .quotes
            .iter()
            .find(|q| q.expiry == expiry && q.strike == strike_i32 && q.right == Right::Put)?;

        let ce_bid = ce.bid.map(|b| b.price).unwrap_or(0.0);
        let ce_ask = ce.ask.map(|a| a.price).unwrap_or(0.0);
        let pe_bid = pe.bid.map(|b| b.price).unwrap_or(0.0);
        let pe_ask = pe.ask.map(|a| a.price).unwrap_or(0.0);

        if ce_bid <= 0.0 || ce_ask <= 0.0 || pe_bid <= 0.0 || pe_ask <= 0.0 {
            return None;
        }

        Some(StraddleQuotes {
            expiry: expiry.to_string(),
            strike: atm_strike,
            ce: QuoteSnapshot {
                bid: ce_bid,
                ask: ce_ask,
                last_ts: ts,
            },
            pe: QuoteSnapshot {
                bid: pe_bid,
                ask: pe_ask,
                last_ts: ts,
            },
        })
    }

    /// Find ATM strike from cached SANOS forward or snapshot spot.
    fn find_atm_strike(&self, features: &Phase8Features) -> f64 {
        let forward = features.f1;
        let tick_size = if self.underlying == "BANKNIFTY" {
            100.0
        } else {
            50.0
        };
        (forward / tick_size).round() * tick_size
    }

    /// Compute time to expiry in years from expiry string (YYYY-MM-DD format).
    fn compute_time_to_expiry(&self, now: DateTime<Utc>, expiry: &str) -> f64 {
        if let Ok(exp_date) = chrono::NaiveDate::parse_from_str(expiry, "%Y-%m-%d") {
            let exp_dt = exp_date
                .and_hms_opt(10, 0, 0)
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
            if let Some(exp_dt) = exp_dt {
                let days = (exp_dt - now).num_seconds() as f64 / 86400.0;
                return (days / 365.0).max(1.0 / 365.0);
            }
        }
        7.0 / 365.0 // Default: 1 week
    }

    /// Minutes until market close (IST).
    fn minutes_to_close(&self, ts: DateTime<Utc>) -> u64 {
        let ist_ts = ts + Duration::hours(5) + Duration::minutes(30);
        let close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE;
        let current_minutes = ist_ts.hour() * 60 + ist_ts.minute();

        if current_minutes >= close_minutes {
            0
        } else {
            (close_minutes - current_minutes) as u64
        }
    }

    /// Check if today is the expiry day for a given expiry.
    fn is_expiry_day(&self, expiry: &str, ts: DateTime<Utc>) -> bool {
        if let Ok(exp_date) = chrono::NaiveDate::parse_from_str(expiry, "%Y-%m-%d") {
            let ist_ts = ts + Duration::hours(5) + Duration::minutes(30);
            let today = ist_ts.date_naive();
            return today == exp_date;
        }
        false
    }
}

/// Build `FrozenParams` with E-gate thresholds relaxed to extreme negatives.
///
/// This keeps all 9 non-economic gates (H1–H4, CARRY, R1, R2) at production
/// values while making E1, E2, E3 pass unconditionally:
/// - `gap_abs_*` → −10 000  (E1: any gap ≥ −10 000 → always true)
/// - `mu_friction` → −10 000 (E2/E3: required_gap deeply negative → always true)
/// - `floor_friction_round_*` → 0 (E3 floor disabled — unnecessary when μ is
///   deeply negative since required_gap = μ × max(obs, floor) is always << 0)
fn make_relaxed_params() -> FrozenParams {
    FrozenParams {
        gap_abs_nifty: -10_000.0,
        gap_abs_banknifty: -10_000.0,
        mu_friction: -10_000.0,
        floor_friction_round_nifty: 0.0,
        floor_friction_round_banknifty: 0.0,
        ..FROZEN_PARAMS
    }
}

// =============================================================================
// FEATURE EXTRACTION (ported from run_calendar_carry.rs)
// =============================================================================

/// Find index of strike nearest to target.
fn find_nearest_strike_idx(strikes: &[f64], target: f64) -> Option<usize> {
    strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let diff_a = (**a - target).abs();
            let diff_b = (**b - target).abs();
            diff_a.total_cmp(&diff_b)
        })
        .map(|(idx, _)| idx)
}

/// Extract IV from SANOS slice at ATM.
fn extract_atm_iv(slice: &SanosSlice) -> Option<f64> {
    let atm_idx = find_nearest_strike_idx(&slice.fitted_strikes, 1.0)?;
    let call_price = slice.fitted_calls[atm_idx];
    let k = slice.fitted_strikes[atm_idx];
    let tty = slice.time_to_expiry;
    extract_iv(call_price, k, tty)
}

/// Extract implied volatility from normalized call price using bisection.
fn extract_iv(call_price: f64, strike_norm: f64, tty: f64) -> Option<f64> {
    if call_price <= 0.0 || call_price >= 1.0 || tty <= 0.0 {
        return None;
    }
    let intrinsic = (1.0 - strike_norm).max(0.0);
    if call_price < intrinsic {
        return None;
    }

    let mut vol_low = 0.001;
    let mut vol_high = 5.0;
    let tolerance = 1e-6;

    for _ in 0..100 {
        let vol_mid = (vol_low + vol_high) / 2.0;
        let price_mid = bs_call_normalized(strike_norm, vol_mid, tty);
        if (price_mid - call_price).abs() < tolerance {
            return Some(vol_mid);
        }
        if price_mid > call_price {
            vol_high = vol_mid;
        } else {
            vol_low = vol_mid;
        }
    }
    Some((vol_low + vol_high) / 2.0)
}

/// Black-Scholes normalized call price (forward = 1).
fn bs_call_normalized(k: f64, vol: f64, t: f64) -> f64 {
    if vol <= 0.0 || t <= 0.0 {
        return (1.0 - k).max(0.0);
    }
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;
    if vol_sqrt_t < 1e-10 {
        return (1.0 - k).max(0.0);
    }
    let d1 = (-(k.ln()) + 0.5 * vol * vol * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    norm_cdf(d1) - k * norm_cdf(d2)
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn norm_cdf(x: f64) -> f64 {
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
    0.5 * (1.0 + sign * y)
}

/// Extract calendar gap from SANOS slices.
fn extract_calendar_gap(s1: &SanosSlice, s2: &SanosSlice) -> Option<f64> {
    let atm1_idx = find_nearest_strike_idx(&s1.fitted_strikes, 1.0)?;
    let atm2_idx = find_nearest_strike_idx(&s2.fitted_strikes, 1.0)?;
    Some(s2.fitted_calls[atm2_idx] - s1.fitted_calls[atm1_idx])
}

/// Extract skew from SANOS slice.
fn extract_skew(slice: &SanosSlice) -> Option<f64> {
    const K_LOW_TARGET: f64 = 0.97;
    const K_HIGH_TARGET: f64 = 1.03;

    let low_idx = find_nearest_strike_idx(&slice.fitted_strikes, K_LOW_TARGET)?;
    let high_idx = find_nearest_strike_idx(&slice.fitted_strikes, K_HIGH_TARGET)?;

    let k_low = slice.fitted_strikes[low_idx];
    let k_high = slice.fitted_strikes[high_idx];
    let tty = slice.time_to_expiry;

    let iv_low = extract_iv(slice.fitted_calls[low_idx], k_low, tty)?;
    let iv_high = extract_iv(slice.fitted_calls[high_idx], k_high, tty)?;

    if (k_high - k_low).abs() < 1e-6 {
        return None;
    }
    Some((iv_high - iv_low) / (k_high - k_low))
}

/// Build Phase8Features from SANOS slices.
fn build_features(slices: &[SanosSlice]) -> anyhow::Result<Phase8Features> {
    if slices.is_empty() {
        return Err(anyhow::anyhow!("No slices provided"));
    }

    let s1 = &slices[0];
    let iv1 = extract_atm_iv(s1).unwrap_or(0.0);
    let sk1 = extract_skew(s1);

    let (iv2, cal12, ts12, sk2, f2, tty2, k_atm2) = if slices.len() >= 2 {
        let s2 = &slices[1];
        let iv2_val = extract_atm_iv(s2).unwrap_or(0.0);
        let cal12_val = extract_calendar_gap(s1, s2);
        let ts12_val = {
            let sqrt_t1 = s1.time_to_expiry.sqrt();
            let sqrt_t2 = s2.time_to_expiry.sqrt();
            if sqrt_t2 != sqrt_t1 {
                Some((iv2_val * sqrt_t2 - iv1 * sqrt_t1) / (sqrt_t2 - sqrt_t1))
            } else {
                None
            }
        };
        let sk2_val = extract_skew(s2);
        (
            Some(iv2_val),
            cal12_val,
            ts12_val,
            sk2_val,
            Some(s2.forward),
            Some(s2.time_to_expiry),
            Some(1.0),
        )
    } else {
        (None, None, None, None, None, None, None)
    };

    let (iv3, cal23, ts23, ts_curv, sk3, f3, tty3, k_atm3) = if slices.len() >= 3 {
        let s3 = &slices[2];
        let iv3_val = extract_atm_iv(s3).unwrap_or(0.0);
        let cal23_val = extract_calendar_gap(&slices[1], s3);
        let ts23_val = if let Some(iv2_val) = iv2 {
            let sqrt_t2 = slices[1].time_to_expiry.sqrt();
            let sqrt_t3 = s3.time_to_expiry.sqrt();
            if sqrt_t3 != sqrt_t2 {
                Some((iv3_val * sqrt_t3 - iv2_val * sqrt_t2) / (sqrt_t3 - sqrt_t2))
            } else {
                None
            }
        } else {
            None
        };
        let ts_curv = match (ts12, ts23_val) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };
        let sk3_val = extract_skew(s3);
        (
            Some(iv3_val),
            cal23_val,
            ts23_val,
            ts_curv,
            sk3_val,
            Some(s3.forward),
            Some(s3.time_to_expiry),
            Some(1.0),
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    Ok(Phase8Features {
        iv1,
        iv2,
        iv3,
        cal12,
        cal23,
        ts12,
        ts23,
        ts_curv,
        sk1,
        sk2,
        sk3,
        f1: s1.forward,
        f2,
        f3,
        tty1: s1.time_to_expiry,
        tty2,
        tty3,
        k_atm1: 1.0,
        k_atm2,
        k_atm3,
    })
}

/// Convert Phase8Features to TUI view.
fn features_to_view(f: &Phase8Features) -> SanosFeatureView {
    SanosFeatureView {
        iv1: f.iv1,
        iv2: f.iv2,
        iv3: f.iv3,
        cal12: f.cal12,
        cal23: f.cal23,
        ts12: f.ts12,
        ts23: f.ts23,
        ts_curv: f.ts_curv,
        sk1: f.sk1,
        sk2: f.sk2,
        f1: f.f1,
        f2: f.f2,
        tty1: f.tty1,
        tty2: f.tty2,
    }
}

/// Create empty gate check result (all fail, used for early returns).
fn empty_gates() -> GateCheckResult {
    use quantlaxmi_options::strategies::GateResult;
    GateCheckResult {
        h1_surface: GateResult::fail("H1_SURFACE", "Not evaluated"),
        h2_calendar: GateResult::fail("H2_CALENDAR", "Not evaluated"),
        h3_quote_front: GateResult::fail("H3_QUOTE_FRONT", "Not evaluated"),
        h3_quote_back: GateResult::fail("H3_QUOTE_BACK", "Not evaluated"),
        h4_liquidity_front: GateResult::fail("H4_LIQ_FRONT", "Not evaluated"),
        h4_liquidity_back: GateResult::fail("H4_LIQ_BACK", "Not evaluated"),
        carry: GateResult::fail("CARRY", "Not evaluated"),
        r1_inversion: GateResult::fail("R1_INVERSION", "Not evaluated"),
        r2_skew: GateResult::fail("R2_SKEW", "Not evaluated"),
        e1_premium_gap: GateResult::fail("E1_PREMIUM_GAP", "Not evaluated"),
        e2_friction_dominance: GateResult::fail("E2_FRICTION_DOM", "Not evaluated"),
        e3_friction_floor: GateResult::fail("E3_FRICTION_FLOOR", "Not evaluated"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::snapshot::{OptQuote, PriceQty, SnapshotProvenance};

    /// Build a synthetic OptionsSnapshot with realistic NIFTY option quotes
    /// at two expiries to exercise the full SANOS calibration pipeline.
    fn make_synthetic_snapshot() -> OptionsSnapshot {
        let underlying = "NIFTY".to_string();
        let spot = 25500.0;
        let expiry_front = "2026-02-10".to_string();
        let expiry_back = "2026-02-17".to_string();

        let strikes: Vec<i32> = (24500..=26500).step_by(100).collect();

        let mut quotes = Vec::new();
        let mut token = 1000u32;

        for expiry in [&expiry_front, &expiry_back] {
            let is_front = expiry == &expiry_front;
            let base_iv = if is_front { 0.15 } else { 0.16 };

            for &strike in &strikes {
                let moneyness = (strike as f64 - spot) / spot;
                let iv = base_iv + 0.02 * moneyness.abs(); // crude smile

                // Black-Scholes-like mid price (very rough)
                let tty: f64 = if is_front { 5.0 / 365.0 } else { 12.0 / 365.0 };
                let d = moneyness / (iv * tty.sqrt());
                let call_mid = spot
                    * (0.4 * (-0.5 * d * d).exp() * iv * tty.sqrt())
                        .max(((spot - strike as f64).max(0.0)) + 2.0);
                let put_mid = call_mid - (spot - strike as f64);

                let spread = (call_mid * 0.02).max(0.5); // 2% spread or 0.50 min

                // Call quote
                let mut call_q = OptQuote::with_expiry(
                    token,
                    format!("NIFTY26FEB{}CE", strike),
                    strike,
                    Right::Call,
                    expiry.clone(),
                );
                call_q.bid = Some(PriceQty::new((call_mid - spread / 2.0).max(0.5), 1000));
                call_q.ask = Some(PriceQty::new(call_mid + spread / 2.0, 1000));
                call_q.last_update_ns = 1_000_000_000;
                call_q.age_ms = 100;
                quotes.push(call_q);
                token += 1;

                // Put quote
                let put_price = put_mid.max(1.0);
                let mut put_q = OptQuote::with_expiry(
                    token,
                    format!("NIFTY26FEB{}PE", strike),
                    strike,
                    Right::Put,
                    expiry.clone(),
                );
                put_q.bid = Some(PriceQty::new((put_price - spread / 2.0).max(0.5), 1000));
                put_q.ask = Some(PriceQty::new(put_price + spread / 2.0, 1000));
                put_q.last_update_ns = 1_000_000_000;
                put_q.age_ms = 100;
                quotes.push(put_q);
                token += 1;
            }
        }

        OptionsSnapshot {
            ts_ns: 1_000_000_000,
            underlying,
            expiry: expiry_front,
            spot: Some(spot),
            quotes,
            provenance: SnapshotProvenance::default(),
        }
    }

    #[test]
    fn test_adapter_evaluate_does_not_panic() {
        let mut adapter = SanosCalendarCarryAdapter::new("NIFTY", 25, 60, false);
        let ts = Utc::now();
        let snapshot = make_synthetic_snapshot();

        // This exercises: expiry discovery, calibration, feature extraction,
        // straddle building, strategy evaluation — the full pipeline.
        let result = adapter.evaluate(ts, &snapshot);

        // Should have discovered 2 expiries
        assert!(
            adapter.expiry_order.len() >= 2,
            "Expected at least 2 expiries, got {}",
            adapter.expiry_order.len()
        );

        // Should have produced a decision (NoTrade or Enter)
        match &result.decision {
            SanosStrategyDecision::NoTrade { reason, .. } => {
                println!("NoTrade: {}", reason);
            }
            SanosStrategyDecision::Enter { intent, .. } => {
                println!(
                    "Enter: front={} back={} strike={:.0}",
                    intent.front_expiry, intent.back_expiry, intent.front_strike
                );
            }
            SanosStrategyDecision::Hold => {
                println!("Hold");
            }
            SanosStrategyDecision::Exit { intent } => {
                println!("Exit: {}", intent.reason);
            }
        }

        // Should have surface views (calibration ran)
        assert!(
            !result.surfaces.is_empty(),
            "Expected surface views from calibration"
        );

        // Should have features
        assert!(result.feature_view.is_some(), "Expected feature view");
        let fv = result.feature_view.unwrap();
        assert!(fv.iv1 > 0.0, "Expected positive IV1, got {}", fv.iv1);
        assert!(fv.iv2.is_some(), "Expected IV2 for 2-expiry snapshot");

        // Should have calibration timestamp
        assert!(
            result.last_calibration_ts.is_some(),
            "Expected calibration timestamp"
        );

        // Should have gates
        assert!(result.gates.is_some(), "Expected gate check results");
    }

    #[test]
    fn test_adapter_calibration_cache() {
        let mut adapter = SanosCalendarCarryAdapter::new("NIFTY", 25, 60, false);
        let ts1 = Utc::now();
        let snapshot = make_synthetic_snapshot();

        // First evaluation: fresh calibration
        let r1 = adapter.evaluate(ts1, &snapshot);
        assert!(!r1.used_cache, "First evaluation should not use cache");

        // Second evaluation within interval: should use cache
        let ts2 = ts1 + Duration::seconds(30);
        let r2 = adapter.evaluate(ts2, &snapshot);
        assert!(
            r2.used_cache,
            "Second evaluation within 60s should use cache"
        );

        // Third evaluation after interval: fresh calibration
        let ts3 = ts1 + Duration::seconds(61);
        let r3 = adapter.evaluate(ts3, &snapshot);
        assert!(!r3.used_cache, "Evaluation after 60s should re-calibrate");
    }

    #[test]
    fn test_adapter_insufficient_expiries() {
        let mut adapter = SanosCalendarCarryAdapter::new("NIFTY", 25, 60, false);
        let ts = Utc::now();

        // Snapshot with only 1 expiry
        let mut snapshot = make_synthetic_snapshot();
        snapshot.quotes.retain(|q| q.expiry == "2026-02-10");

        let result = adapter.evaluate(ts, &snapshot);
        match result.decision {
            SanosStrategyDecision::NoTrade { reason, .. } => {
                assert!(
                    reason.contains("at least 2 expiries"),
                    "Expected expiry count error, got: {}",
                    reason
                );
            }
            _ => panic!("Expected NoTrade for single-expiry snapshot"),
        }
    }

    #[test]
    fn test_feature_extraction_helpers() {
        // Test norm_cdf
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(norm_cdf(3.0) > 0.99);
        assert!(norm_cdf(-3.0) < 0.01);

        // Test bs_call_normalized
        let atm = bs_call_normalized(1.0, 0.2, 1.0);
        assert!(atm > 0.0 && atm < 1.0, "ATM call should be between 0 and 1");

        // Deep ITM (k << 1)
        let itm = bs_call_normalized(0.5, 0.2, 1.0);
        assert!(itm > 0.4, "Deep ITM call should be > 0.4");

        // Deep OTM (k >> 1)
        let otm = bs_call_normalized(2.0, 0.2, 1.0);
        assert!(otm < 0.05, "Deep OTM call should be < 0.05");
    }
}
