//! SANOS Phase 8: Feature Extractor (Read-only)
//!
//! Extracts regime-gating and volatility carry features from certified SANOS surfaces.
//!
//! Features computed:
//! - ATM IV term structure (σ1, σ2, σ3)
//! - Term structure slope (TS12, TS23, TS_curv)
//! - Calendar price gaps (CAL12, CAL23)
//! - Skew per expiry (SK1, SK2, SK3)
//! - Roll-down estimates (ROLL12, ROLLC12)
//!
//! ## Modes (Commit D)
//! - **Manifest-driven**: When `session_manifest.json` exists, uses deterministic inventory.
//! - **Legacy**: Falls back to directory scanning + symbol parsing when no manifest.
//!
//! Usage:
//!   cargo run --bin sanos_features -- --session-dir <path> --underlying NIFTY

use anyhow::{Result, anyhow};
use chrono::{DateTime, NaiveDate, Utc};
use clap::Parser;
use csv::Writer as CsvWriter;
use quantlaxmi_options::sanos::{
    EPSILON_STRIKE, ETA, ExpirySlice, K_N_NORMALIZED, OptionQuote, SanosCalibrator, SanosSlice,
    StrikeMeta, V_MIN,
};
use quantlaxmi_runner_india::sanos_io::{
    self, SanosManifestInventory, SanosUnderlyingInventory,
};

/// Expected number of CSV columns (Phase 8.1 integrity)
const CSV_COLUMN_COUNT: usize = 25;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "sanos_features")]
#[command(about = "SANOS Phase 8 Feature Extractor")]
struct Args {
    /// Session directory containing captured tick data
    #[arg(long)]
    session_dir: PathBuf,

    /// Underlying to extract features for (NIFTY or BANKNIFTY)
    #[arg(long, default_value = "NIFTY")]
    underlying: String,

    /// Output CSV file
    #[arg(long, default_value = "sanos_features.csv")]
    output: PathBuf,

    /// Output manifest JSON file
    #[arg(long, default_value = "feature_manifest.json")]
    manifest: PathBuf,

    /// SANOS smoothness parameter η
    #[arg(long, default_value = "0.25")]
    eta: f64,
}

/// Tick event from captured session
#[derive(Debug, Deserialize)]
struct TickEvent {
    ts: DateTime<Utc>,
    #[allow(dead_code)]
    tradingsymbol: String,
    #[allow(dead_code)]
    instrument_token: u32,
    bid_price: i64,
    ask_price: i64,
    #[allow(dead_code)]
    bid_qty: u32,
    #[allow(dead_code)]
    ask_qty: u32,
    #[allow(dead_code)]
    ltp: i64,
    #[allow(dead_code)]
    ltq: u32,
    #[allow(dead_code)]
    volume: u64,
    price_exponent: i32,
    #[allow(dead_code)]
    integrity_tier: String,
}

/// Feature vector for one snapshot
#[derive(Debug, Clone, Serialize)]
struct FeatureRow {
    ts: String,
    underlying: String,
    session_id: String,

    // Expiry dates
    t1: String,
    t2: Option<String>,
    t3: Option<String>,

    // Forwards
    f1: f64,
    f2: Option<f64>,
    f3: Option<f64>,

    // TTY in years
    tty1: f64,
    tty2: Option<f64>,
    tty3: Option<f64>,

    // ATM IVs
    iv1: f64,
    iv2: Option<f64>,
    iv3: Option<f64>,

    // Term structure slopes (calendarized)
    ts12: Option<f64>,
    ts23: Option<f64>,
    ts_curv: Option<f64>,

    // Calendar price gaps at ATM
    cal12: Option<f64>,
    cal23: Option<f64>,

    // Skew per expiry
    sk1: Option<f64>,
    sk2: Option<f64>,
    sk3: Option<f64>,

    // Roll-down estimates
    roll12: Option<f64>,
    rollc12: Option<f64>,
}

/// Feature manifest (policies + grid points)
#[derive(Debug, Serialize)]
struct FeatureManifest {
    version: String,
    policies: PolicyConfig,
    grid_points: GridConfig,
    iv_solver: IVSolverConfig,
    session_info: SessionInfo,
}

#[derive(Debug, Serialize)]
struct PolicyConfig {
    eta: f64,
    v_min: f64,
    epsilon_strike: f64,
    k_n_normalized: f64, // Phase 8.1: fixed far OTM boundary
    strike_band: u32,
    expiry_policy: String,
}

#[derive(Debug, Serialize)]
struct GridConfig {
    k_low: f64,
    k_atm: f64,
    k_high: f64,
}

#[derive(Debug, Serialize)]
struct IVSolverConfig {
    method: String,
    vol_min: f64,
    vol_max: f64,
    tolerance: f64,
}

#[derive(Debug, Serialize)]
struct SessionInfo {
    session_dir: String,
    underlying: String,
    expiries: Vec<String>,
    snapshot_count: usize,
}

/// Parse option symbol: NIFTY26JAN25300CE -> (underlying, expiry, strike, is_call)
fn parse_symbol(symbol: &str) -> Option<(String, String, u32, bool)> {
    let symbol = symbol.to_uppercase();

    let is_call = symbol.ends_with("CE");
    let is_put = symbol.ends_with("PE");
    if !is_call && !is_put {
        return None;
    }

    let without_type = &symbol[..symbol.len() - 2];

    let (underlying, rest) = if let Some(rest) = without_type.strip_prefix("BANKNIFTY") {
        ("BANKNIFTY".to_string(), rest)
    } else if let Some(rest) = without_type.strip_prefix("FINNIFTY") {
        ("FINNIFTY".to_string(), rest)
    } else if let Some(rest) = without_type.strip_prefix("NIFTY") {
        ("NIFTY".to_string(), rest)
    } else {
        return None;
    };

    if rest.len() < 6 {
        return None;
    }

    let expiry = rest[..5].to_string();
    let strike: u32 = rest[5..].parse().ok()?;

    Some((underlying, expiry, strike, is_call))
}

/// Convert expiry code to date
fn expiry_to_date(expiry: &str) -> Option<NaiveDate> {
    if expiry.len() == 5 {
        let month_str = &expiry[2..];
        let month = match month_str {
            "JAN" => Some(1),
            "FEB" => Some(2),
            "MAR" => Some(3),
            "APR" => Some(4),
            "MAY" => Some(5),
            "JUN" => Some(6),
            "JUL" => Some(7),
            "AUG" => Some(8),
            "SEP" => Some(9),
            "OCT" => Some(10),
            "NOV" => Some(11),
            "DEC" => Some(12),
            _ => None,
        };

        if let Some(month) = month {
            let day: u32 = expiry[..2].parse().ok()?;
            return NaiveDate::from_ymd_opt(2026, month, day);
        }

        // YYMMDD compact format (e.g., "26203" = 2026-02-03)
        let year: i32 = 2000 + expiry[..2].parse::<i32>().ok()?;
        let month_char = expiry.chars().nth(2)?;
        let month: u32 = match month_char {
            '1'..='9' => month_char.to_digit(10)?,
            'O' => 10,
            'N' => 11,
            'D' => 12,
            _ => return None,
        };
        let day: u32 = expiry[3..].parse().ok()?;
        return NaiveDate::from_ymd_opt(year, month, day);
    }

    None
}

/// Calculate time to expiry in years
fn time_to_expiry(now: DateTime<Utc>, expiry: &str) -> f64 {
    if let Some(exp_date) = expiry_to_date(expiry) {
        let exp_datetime = exp_date
            .and_hms_opt(10, 0, 0)
            .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));

        if let Some(exp_dt) = exp_datetime {
            let days = (exp_dt - now).num_seconds() as f64 / 86400.0;
            return (days / 365.0).max(1.0 / 365.0);
        }
    }
    7.0 / 365.0
}

/// Discover all expiries in session for given underlying
fn discover_expiries(session_dir: &PathBuf, underlying: &str) -> Result<Vec<String>> {
    let mut expiries = std::collections::HashSet::new();

    for entry in std::fs::read_dir(session_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let symbol = path.file_name().unwrap().to_string_lossy().to_string();

        if let Some((und, exp, _, _)) = parse_symbol(&symbol)
            && und == underlying
        {
            expiries.insert(exp);
        }
    }

    let mut expiries: Vec<_> = expiries.into_iter().collect();
    expiries.sort_by_key(|e| expiry_to_date(e));

    Ok(expiries)
}

/// Load ticks for a specific underlying and expiry
fn load_ticks(
    session_dir: &PathBuf,
    underlying: &str,
    expiry: &str,
) -> Result<HashMap<String, Vec<TickEvent>>> {
    let mut symbol_ticks: HashMap<String, Vec<TickEvent>> = HashMap::new();

    for entry in std::fs::read_dir(session_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let symbol = path.file_name().unwrap().to_string_lossy().to_string();

        if let Some((und, exp, _strike, _is_call)) = parse_symbol(&symbol) {
            if und != underlying || exp != expiry {
                continue;
            }

            let ticks_file = path.join("ticks.jsonl");
            if !ticks_file.exists() {
                continue;
            }

            let file = File::open(&ticks_file)?;
            let reader = BufReader::new(file);
            let mut ticks = Vec::new();

            for line in reader.lines() {
                let line = line?;
                if let Ok(tick) = serde_json::from_str::<TickEvent>(&line) {
                    ticks.push(tick);
                }
            }

            symbol_ticks.insert(symbol, ticks);
        }
    }

    Ok(symbol_ticks)
}

/// Build ExpirySlice from tick data at a specific timestamp
fn build_slice(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying: &str,
    expiry: &str,
    target_ts: DateTime<Utc>,
    time_to_exp: f64,
) -> Result<ExpirySlice> {
    let mut slice = ExpirySlice::new(underlying, expiry, target_ts, time_to_exp);

    for (symbol, ticks) in symbol_ticks {
        let closest_tick = ticks
            .iter()
            .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());

        if let Some(tick) = closest_tick {
            if (tick.ts - target_ts).num_seconds().abs() > 5 {
                continue;
            }

            let (_und, _exp, strike, is_call) = parse_symbol(symbol).unwrap();

            let price_mult = 10f64.powi(tick.price_exponent);
            let bid = tick.bid_price as f64 * price_mult;
            let ask = tick.ask_price as f64 * price_mult;

            if bid <= 0.0 || ask <= 0.0 || ask < bid {
                continue;
            }

            let quote = OptionQuote {
                symbol: symbol.clone(),
                strike: strike as f64,
                is_call,
                bid,
                ask,
                timestamp: tick.ts,
            };

            slice.add_quote(quote);
        }
    }

    Ok(slice)
}

// =============================================================================
// MANIFEST-DRIVEN LOADERS (Commit D)
// =============================================================================

/// Load ticks for a specific expiry using manifest inventory (no directory scan).
fn load_ticks_manifest(
    session_dir: &PathBuf,
    underlying_inv: &SanosUnderlyingInventory,
    expiry: NaiveDate,
) -> Result<HashMap<String, Vec<TickEvent>>> {
    let mut symbol_ticks: HashMap<String, Vec<TickEvent>> = HashMap::new();

    // Get symbols for this expiry from the manifest
    let symbols = underlying_inv.get_symbols_for_expiry(expiry);

    for symbol in symbols {
        // Get tick file path from manifest
        if let Some(rel_path) = underlying_inv.get_tick_path(&symbol) {
            let ticks_file = session_dir.join(rel_path);
            if !ticks_file.exists() {
                continue;
            }

            let file = File::open(&ticks_file)?;
            let reader = BufReader::new(file);
            let mut ticks = Vec::new();

            for line in reader.lines() {
                let line = line?;
                if let Ok(tick) = serde_json::from_str::<TickEvent>(&line) {
                    ticks.push(tick);
                }
            }

            symbol_ticks.insert(symbol, ticks);
        }
    }

    Ok(symbol_ticks)
}

/// Build ExpirySlice from tick data using manifest instrument info (no symbol parsing).
fn build_slice_manifest(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying_inv: &SanosUnderlyingInventory,
    expiry: NaiveDate,
    target_ts: DateTime<Utc>,
    time_to_exp: f64,
) -> Result<ExpirySlice> {
    let expiry_str = expiry.format("%Y-%m-%d").to_string();
    let mut slice = ExpirySlice::new(&underlying_inv.underlying, &expiry_str, target_ts, time_to_exp);

    // Get instruments for this expiry from manifest
    let instruments = underlying_inv.get_instruments_for_expiry(expiry);

    for instr in instruments {
        if let Some(ticks) = symbol_ticks.get(&instr.tradingsymbol) {
            let closest_tick = ticks
                .iter()
                .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());

            if let Some(tick) = closest_tick {
                if (tick.ts - target_ts).num_seconds().abs() > 5 {
                    continue;
                }

                let price_mult = 10f64.powi(tick.price_exponent);
                let bid = tick.bid_price as f64 * price_mult;
                let ask = tick.ask_price as f64 * price_mult;

                if bid <= 0.0 || ask <= 0.0 || ask < bid {
                    continue;
                }

                let quote = OptionQuote {
                    symbol: instr.tradingsymbol.clone(),
                    strike: instr.strike,
                    is_call: instr.instrument_type == "CE",
                    bid,
                    ask,
                    timestamp: tick.ts,
                };

                slice.add_quote(quote);
            }
        }
    }

    Ok(slice)
}

/// Calculate time to expiry in years from NaiveDate
fn time_to_expiry_from_date(now: DateTime<Utc>, exp_date: NaiveDate) -> f64 {
    // Assume expiry at 15:30 IST (10:00 UTC)
    let exp_datetime = exp_date
        .and_hms_opt(10, 0, 0)
        .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));

    if let Some(exp_dt) = exp_datetime {
        let days = (exp_dt - now).num_seconds() as f64 / 86400.0;
        return (days / 365.0).max(1.0 / 365.0); // Minimum 1 day
    }

    7.0 / 365.0 // Fallback: 7 days
}

/// Extract implied volatility from fitted call price
/// Uses bisection with safe bounds
fn extract_iv(call_price: f64, strike_norm: f64, tty: f64) -> Option<f64> {
    // In forward-normalized units: F=1, so call price C and strike K are normalized
    // Use Black-Scholes inversion
    // C = N(d1) - K*N(d2) where d1 = (ln(1/K) + 0.5*σ²T) / (σ√T)

    if call_price <= 0.0 || call_price >= 1.0 || tty <= 0.0 {
        return None;
    }

    // Intrinsic value check
    let intrinsic = (1.0 - strike_norm).max(0.0);
    if call_price < intrinsic {
        return None;
    }

    // Bisection search for IV
    let mut vol_low = 0.001; // 0.1%
    let mut vol_high = 5.0; // 500%
    let tolerance = 1e-6;
    let max_iter = 100;

    for _ in 0..max_iter {
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

/// Black-Scholes call price in forward-normalized units (F=1)
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

/// Standard normal CDF using Abramowitz & Stegun approximation
fn norm_cdf(x: f64) -> f64 {
    // A&S formula 7.1.26, accurate to 1.5e-7
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

/// Strike selection result with metadata for logging
#[derive(Debug, Clone)]
struct SelectedStrike {
    index: usize,
    k: f64,
    meta: StrikeMeta,
}

/// Find nearest feature-safe grid point to target normalized strike
/// Uses StrikeMeta to exclude boundary points (K0, K1, KN)
fn find_nearest_strike_safe(
    model_strikes: &[f64],
    strike_meta: &[StrikeMeta],
    target: f64,
) -> Option<SelectedStrike> {
    model_strikes
        .iter()
        .zip(strike_meta.iter())
        .enumerate()
        .filter(|(_, (_, meta))| meta.is_feature_safe())
        .min_by(|(_, (a, _)), (_, (b, _))| {
            (**a - target)
                .abs()
                .partial_cmp(&(**b - target).abs())
                .unwrap()
        })
        .map(|(i, (k, meta))| SelectedStrike {
            index: i,
            k: *k,
            meta: *meta,
        })
}

/// Extract features from calibrated SANOS slices
/// Uses boundary-hardened strike selection (Phase 8.1)
fn extract_features(
    slices: &[SanosSlice],
    session_id: &str,
    underlying: &str,
) -> Result<FeatureRow> {
    if slices.is_empty() {
        return Err(anyhow!("No slices provided"));
    }

    let ts = slices[0].timestamp;

    // Get T1 (always present)
    let s1 = &slices[0];
    let f1 = s1.forward;
    let tty1 = s1.time_to_expiry;

    // Find ATM and wing points for T1 (boundary-safe selection)
    let atm1 = find_nearest_strike_safe(&s1.fitted_strikes, &s1.strike_meta, 1.0)
        .ok_or_else(|| anyhow!("No feature-safe ATM strike for T1"))?;
    let low1 = find_nearest_strike_safe(&s1.fitted_strikes, &s1.strike_meta, 0.97)
        .ok_or_else(|| anyhow!("No feature-safe low strike for T1"))?;
    let high1 = find_nearest_strike_safe(&s1.fitted_strikes, &s1.strike_meta, 1.03)
        .ok_or_else(|| anyhow!("No feature-safe high strike for T1"))?;

    // Log selected strike points (Phase 8.1 audit requirement)
    info!(
        "T1 selected strikes: k_low={:.4} (market={}), k_atm={:.4} (market={}), k_high={:.4} (market={})",
        low1.k, low1.meta.is_market, atm1.k, atm1.meta.is_market, high1.k, high1.meta.is_market
    );

    // Extract IVs for T1
    let c_atm1 = s1.fitted_calls[atm1.index];
    let iv1 = extract_iv(c_atm1, atm1.k, tty1).unwrap_or(0.0);

    let c_low1 = s1.fitted_calls[low1.index];
    let c_high1 = s1.fitted_calls[high1.index];
    let iv_low1 = extract_iv(c_low1, low1.k, tty1);
    let iv_high1 = extract_iv(c_high1, high1.k, tty1);

    // Skew for T1
    let sk1 = match (iv_low1, iv_high1) {
        (Some(ivl), Some(ivh)) if high1.k != low1.k => Some((ivh - ivl) / (high1.k - low1.k)),
        _ => None,
    };

    // T2 features (if available)
    let (t2, f2, tty2, iv2, sk2, cal12, ts12, roll12, rollc12) = if slices.len() >= 2 {
        let s2 = &slices[1];
        let f2 = s2.forward;
        let tty2 = s2.time_to_expiry;

        let atm2 = find_nearest_strike_safe(&s2.fitted_strikes, &s2.strike_meta, 1.0)
            .ok_or_else(|| anyhow!("No feature-safe ATM strike for T2"))?;
        let low2 = find_nearest_strike_safe(&s2.fitted_strikes, &s2.strike_meta, 0.97)
            .ok_or_else(|| anyhow!("No feature-safe low strike for T2"))?;
        let high2 = find_nearest_strike_safe(&s2.fitted_strikes, &s2.strike_meta, 1.03)
            .ok_or_else(|| anyhow!("No feature-safe high strike for T2"))?;

        info!(
            "T2 selected strikes: k_low={:.4} (market={}), k_atm={:.4} (market={}), k_high={:.4} (market={})",
            low2.k, low2.meta.is_market, atm2.k, atm2.meta.is_market, high2.k, high2.meta.is_market
        );

        let c_atm2 = s2.fitted_calls[atm2.index];
        let iv2 = extract_iv(c_atm2, atm2.k, tty2).unwrap_or(0.0);

        let c_low2 = s2.fitted_calls[low2.index];
        let c_high2 = s2.fitted_calls[high2.index];
        let iv_low2 = extract_iv(c_low2, low2.k, tty2);
        let iv_high2 = extract_iv(c_high2, high2.k, tty2);

        let sk2 = match (iv_low2, iv_high2) {
            (Some(ivl), Some(ivh)) if high2.k != low2.k => Some((ivh - ivl) / (high2.k - low2.k)),
            _ => None,
        };

        // Calendar price gap at ATM
        let cal12 = c_atm2 - c_atm1;

        // Term structure slope (calendarized)
        let sqrt_t1 = tty1.sqrt();
        let sqrt_t2 = tty2.sqrt();
        let ts12 = if sqrt_t2 != sqrt_t1 {
            (iv2 * sqrt_t2 - iv1 * sqrt_t1) / (sqrt_t2 - sqrt_t1)
        } else {
            0.0
        };

        // Roll-down estimates
        let roll12 = iv2 - iv1;
        let rollc12 = cal12;

        (
            Some(s2.expiry.clone()),
            Some(f2),
            Some(tty2),
            Some(iv2),
            sk2,
            Some(cal12),
            Some(ts12),
            Some(roll12),
            Some(rollc12),
        )
    } else {
        (None, None, None, None, None, None, None, None, None)
    };

    // T3 features (if available)
    let (t3, f3, tty3, iv3, sk3, cal23, ts23, ts_curv) = if slices.len() >= 3 {
        let s3 = &slices[2];
        let f3 = s3.forward;
        let tty3 = s3.time_to_expiry;

        let atm3 = find_nearest_strike_safe(&s3.fitted_strikes, &s3.strike_meta, 1.0)
            .ok_or_else(|| anyhow!("No feature-safe ATM strike for T3"))?;
        let low3 = find_nearest_strike_safe(&s3.fitted_strikes, &s3.strike_meta, 0.97)
            .ok_or_else(|| anyhow!("No feature-safe low strike for T3"))?;
        let high3 = find_nearest_strike_safe(&s3.fitted_strikes, &s3.strike_meta, 1.03)
            .ok_or_else(|| anyhow!("No feature-safe high strike for T3"))?;

        info!(
            "T3 selected strikes: k_low={:.4} (market={}), k_atm={:.4} (market={}), k_high={:.4} (market={})",
            low3.k, low3.meta.is_market, atm3.k, atm3.meta.is_market, high3.k, high3.meta.is_market
        );

        let c_atm3 = s3.fitted_calls[atm3.index];
        let iv3 = extract_iv(c_atm3, atm3.k, tty3).unwrap_or(0.0);

        let c_low3 = s3.fitted_calls[low3.index];
        let c_high3 = s3.fitted_calls[high3.index];
        let iv_low3 = extract_iv(c_low3, low3.k, tty3);
        let iv_high3 = extract_iv(c_high3, high3.k, tty3);

        let sk3 = match (iv_low3, iv_high3) {
            (Some(ivl), Some(ivh)) if high3.k != low3.k => Some((ivh - ivl) / (high3.k - low3.k)),
            _ => None,
        };

        // Calendar price gap T2->T3
        let cal23 = if tty2.is_some() {
            let s2 = &slices[1];
            let atm2 = find_nearest_strike_safe(&s2.fitted_strikes, &s2.strike_meta, 1.0).unwrap();
            let c_atm2 = s2.fitted_calls[atm2.index];
            Some(c_atm3 - c_atm2)
        } else {
            None
        };

        // Term structure slope T2->T3
        let ts23 = if let (Some(iv2_val), Some(tty2_val)) = (iv2, tty2) {
            let sqrt_t2 = tty2_val.sqrt();
            let sqrt_t3 = tty3.sqrt();
            if sqrt_t3 != sqrt_t2 {
                Some((iv3 * sqrt_t3 - iv2_val * sqrt_t2) / (sqrt_t3 - sqrt_t2))
            } else {
                None
            }
        } else {
            None
        };

        // Curvature
        let ts_curv = match (ts12, ts23) {
            (Some(ts12_val), Some(ts23_val)) => Some(ts23_val - ts12_val),
            _ => None,
        };

        (
            Some(s3.expiry.clone()),
            Some(f3),
            Some(tty3),
            Some(iv3),
            sk3,
            cal23,
            ts23,
            ts_curv,
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };

    Ok(FeatureRow {
        ts: ts.to_rfc3339(),
        underlying: underlying.to_string(),
        session_id: session_id.to_string(),
        t1: s1.expiry.clone(),
        t2,
        t3,
        f1,
        f2,
        f3,
        tty1,
        tty2,
        tty3,
        iv1,
        iv2,
        iv3,
        ts12,
        ts23,
        ts_curv,
        cal12,
        cal23,
        sk1,
        sk2,
        sk3,
        roll12,
        rollc12,
    })
}

// =============================================================================
// MANIFEST-DRIVEN MAIN (Commit D)
// =============================================================================

fn run_manifest_mode(args: &Args, inventory: &SanosManifestInventory) -> Result<()> {
    // Find the underlying entry for the requested underlying
    let underlying_inv = inventory
        .underlyings
        .iter()
        .find(|u| u.underlying.eq_ignore_ascii_case(&args.underlying))
        .ok_or_else(|| {
            anyhow!(
                "Underlying {} not found in session manifest. Available: {:?}",
                args.underlying,
                inventory.underlyings.iter().map(|u| &u.underlying).collect::<Vec<_>>()
            )
        })?;

    // Get expiries from manifest (sorted)
    let expiries = underlying_inv.get_sorted_expiries();
    info!(
        "Manifest-driven: {} expiries from universe_sha256={}",
        expiries.len(),
        underlying_inv.universe_sha256
    );

    if expiries.is_empty() {
        return Err(anyhow!("No expiries found for {} in manifest", args.underlying));
    }

    // Find timestamp range from first expiry ticks
    let first_ticks = load_ticks_manifest(&inventory.session_dir, underlying_inv, expiries[0])?;
    let mut min_ts: Option<DateTime<Utc>> = None;
    let mut max_ts: Option<DateTime<Utc>> = None;

    for ticks in first_ticks.values() {
        for tick in ticks {
            min_ts = Some(min_ts.map_or(tick.ts, |m| m.min(tick.ts)));
            max_ts = Some(max_ts.map_or(tick.ts, |m| m.max(tick.ts)));
        }
    }

    let min_ts = min_ts.ok_or_else(|| anyhow!("No ticks found"))?;
    let max_ts = max_ts.ok_or_else(|| anyhow!("No ticks found"))?;
    let target_ts = min_ts + (max_ts - min_ts) / 2;

    info!("Calibration timestamp: {}", target_ts);

    // Calibrate each expiry using manifest-driven loaders
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let mut slices: Vec<SanosSlice> = Vec::new();

    for expiry in &expiries {
        let expiry_str = expiry.format("%Y-%m-%d").to_string();
        info!("Calibrating {} {} (manifest-driven)", args.underlying, expiry_str);

        let ticks = load_ticks_manifest(&inventory.session_dir, underlying_inv, *expiry)?;
        let tte = time_to_expiry_from_date(target_ts, *expiry);

        let slice = build_slice_manifest(&ticks, underlying_inv, *expiry, target_ts, tte)?;

        if slice.calls.is_empty() || slice.puts.is_empty() {
            info!("  Skipping: insufficient data");
            continue;
        }

        match calibrator.calibrate(&slice) {
            Ok(sanos_slice) => {
                info!(
                    "  F0={:.2}, Σq={:.6}, LP={}",
                    sanos_slice.forward,
                    sanos_slice.diagnostics.weights_sum,
                    sanos_slice.diagnostics.lp_status
                );
                slices.push(sanos_slice);
            }
            Err(e) => {
                info!("  FAILED: {}", e);
            }
        }
    }

    if slices.is_empty() {
        return Err(anyhow!("No successful calibrations"));
    }

    // Extract features (reuse existing function)
    let features = extract_features(&slices, &inventory.session_id, &args.underlying)?;

    // Write CSV (same format as legacy mode)
    let csv_file = File::create(&args.output)?;
    let mut wtr = CsvWriter::from_writer(csv_file);

    // Header (strictly defined column order)
    let header = [
        "ts",
        "underlying",
        "session_id",
        "t1",
        "t2",
        "t3",
        "f1",
        "f2",
        "f3",
        "tty1",
        "tty2",
        "tty3",
        "iv1",
        "iv2",
        "iv3",
        "ts12",
        "ts23",
        "ts_curv",
        "cal12",
        "cal23",
        "sk1",
        "sk2",
        "sk3",
        "roll12",
        "rollc12",
    ];
    debug_assert_eq!(
        header.len(),
        CSV_COLUMN_COUNT,
        "Header column count mismatch"
    );
    wtr.write_record(header)?;

    // Data row
    let row = [
        features.ts.clone(),
        features.underlying.clone(),
        features.session_id.clone(),
        features.t1.clone(),
        features.t2.clone().unwrap_or_default(),
        features.t3.clone().unwrap_or_default(),
        format!("{:.2}", features.f1),
        features.f2.map(|v| format!("{:.2}", v)).unwrap_or_default(),
        features.f3.map(|v| format!("{:.2}", v)).unwrap_or_default(),
        format!("{:.6}", features.tty1),
        features
            .tty2
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .tty3
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        format!("{:.4}", features.iv1),
        features
            .iv2
            .map(|v| format!("{:.4}", v))
            .unwrap_or_default(),
        features
            .iv3
            .map(|v| format!("{:.4}", v))
            .unwrap_or_default(),
        features
            .ts12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .ts23
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .ts_curv
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .cal12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .cal23
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .sk1
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .sk2
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .sk3
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .roll12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .rollc12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
    ];

    debug_assert_eq!(row.len(), CSV_COLUMN_COUNT, "Row column count mismatch");
    wtr.write_record(&row)?;
    wtr.flush()?;

    info!(
        "Wrote features to {:?} ({} columns)",
        args.output, CSV_COLUMN_COUNT
    );

    // Write manifest with audit info
    let manifest = FeatureManifest {
        version: "0.8.1".to_string(),
        policies: PolicyConfig {
            eta: args.eta,
            v_min: V_MIN,
            epsilon_strike: EPSILON_STRIKE,
            k_n_normalized: K_N_NORMALIZED,
            strike_band: 20,
            expiry_policy: "T1T2T3".to_string(),
        },
        grid_points: GridConfig {
            k_low: 0.97,
            k_atm: 1.0,
            k_high: 1.03,
        },
        iv_solver: IVSolverConfig {
            method: "bisection".to_string(),
            vol_min: 0.001,
            vol_max: 5.0,
            tolerance: 1e-6,
        },
        session_info: SessionInfo {
            session_dir: inventory.session_dir.to_string_lossy().to_string(),
            underlying: args.underlying.clone(),
            expiries: slices.iter().map(|s| s.expiry.clone()).collect(),
            snapshot_count: 1,
        },
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&args.manifest, &manifest_json)?;
    info!(
        "Wrote manifest to {:?} (universe_sha256={})",
        args.manifest, underlying_inv.universe_sha256
    );

    // Print summary (same as legacy)
    print_feature_summary(&features);

    Ok(())
}

fn print_feature_summary(features: &FeatureRow) {
    println!("\n=== SANOS PHASE 8: FEATURE EXTRACTION ===");
    println!("Underlying: {}", features.underlying);
    println!("Timestamp: {}", features.ts);
    println!();

    println!("--- Term Structure ---");
    println!("Expiry     Forward      TTY(d)    ATM IV");
    println!(
        "{:<10} {:>10.2} {:>8.1}    {:.2}%",
        features.t1,
        features.f1,
        features.tty1 * 365.0,
        features.iv1 * 100.0
    );
    if let (Some(t2), Some(f2), Some(tty2), Some(iv2)) =
        (&features.t2, features.f2, features.tty2, features.iv2)
    {
        println!(
            "{:<10} {:>10.2} {:>8.1}    {:.2}%",
            t2, f2, tty2 * 365.0, iv2 * 100.0
        );
    }
    if let (Some(t3), Some(f3), Some(tty3), Some(iv3)) =
        (&features.t3, features.f3, features.tty3, features.iv3)
    {
        println!(
            "{:<10} {:>10.2} {:>8.1}    {:.2}%",
            t3, f3, tty3 * 365.0, iv3 * 100.0
        );
    }
    println!();

    println!("--- Features ---");
    if let Some(ts12) = features.ts12 {
        println!("TS12 (term slope T1->T2):  {:.4}", ts12);
    }
    if let Some(ts23) = features.ts23 {
        println!("TS23 (term slope T2->T3):  {:.4}", ts23);
    }
    if let Some(ts_curv) = features.ts_curv {
        println!("TS_curv (curvature):       {:.4}", ts_curv);
    }
    println!();

    if let Some(cal12) = features.cal12 {
        println!("CAL12 (price gap T1->T2):  {:.6}", cal12);
    }
    if let Some(cal23) = features.cal23 {
        println!("CAL23 (price gap T2->T3):  {:.6}", cal23);
    }
    println!();

    if let Some(sk1) = features.sk1 {
        println!("SK1 (skew T1):             {:.4}", sk1);
    }
    if let Some(sk2) = features.sk2 {
        println!("SK2 (skew T2):             {:.4}", sk2);
    }
    if let Some(sk3) = features.sk3 {
        println!("SK3 (skew T3):             {:.4}", sk3);
    }
    println!();

    if let Some(roll12) = features.roll12 {
        println!("ROLL12 (IV carry T1->T2):  {:.4}", roll12);
    }
    if let Some(rollc12) = features.rollc12 {
        println!("ROLLC12 (price carry):     {:.6}", rollc12);
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::level_filters::LevelFilter::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!("SANOS Phase 8: Feature Extractor");
    info!("Session: {:?}", args.session_dir);
    info!("Underlying: {}", args.underlying);

    // Try manifest-driven mode (Commit D)
    if let Some(inventory) = sanos_io::try_load_sanos_inventory(&args.session_dir)? {
        sanos_io::log_manifest_mode(&inventory);
        return run_manifest_mode(&args, &inventory);
    }

    // Legacy mode: discover expiries from directory scan
    sanos_io::log_legacy_mode(&args.session_dir);
    let expiries = discover_expiries(&args.session_dir, &args.underlying)?;
    info!("Found {} expiries: {:?}", expiries.len(), expiries);

    if expiries.is_empty() {
        return Err(anyhow!("No expiries found for {}", args.underlying));
    }

    // Find timestamp range
    let first_ticks = load_ticks(&args.session_dir, &args.underlying, &expiries[0])?;
    let mut min_ts: Option<DateTime<Utc>> = None;
    let mut max_ts: Option<DateTime<Utc>> = None;

    for ticks in first_ticks.values() {
        for tick in ticks {
            min_ts = Some(min_ts.map_or(tick.ts, |m| m.min(tick.ts)));
            max_ts = Some(max_ts.map_or(tick.ts, |m| m.max(tick.ts)));
        }
    }

    let min_ts = min_ts.ok_or_else(|| anyhow!("No ticks found"))?;
    let max_ts = max_ts.ok_or_else(|| anyhow!("No ticks found"))?;

    // Use mid-session for single snapshot (v1)
    let target_ts = min_ts + (max_ts - min_ts) / 2;
    info!("Calibration timestamp: {}", target_ts);

    // Calibrate each expiry
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let mut slices: Vec<SanosSlice> = Vec::new();

    for expiry in &expiries {
        info!("Calibrating {} {}", args.underlying, expiry);

        let ticks = load_ticks(&args.session_dir, &args.underlying, expiry)?;
        let tte = time_to_expiry(target_ts, expiry);

        let slice = build_slice(&ticks, &args.underlying, expiry, target_ts, tte)?;

        if slice.calls.is_empty() || slice.puts.is_empty() {
            info!("  Skipping: insufficient data");
            continue;
        }

        match calibrator.calibrate(&slice) {
            Ok(sanos_slice) => {
                info!(
                    "  F0={:.2}, Σq={:.6}, LP={}",
                    sanos_slice.forward,
                    sanos_slice.diagnostics.weights_sum,
                    sanos_slice.diagnostics.lp_status
                );
                slices.push(sanos_slice);
            }
            Err(e) => {
                info!("  FAILED: {}", e);
            }
        }
    }

    if slices.is_empty() {
        return Err(anyhow!("No successful calibrations"));
    }

    // Extract session ID from path
    let session_id = args
        .session_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Extract features
    info!("Extracting features...");
    let features = extract_features(&slices, &session_id, &args.underlying)?;

    // Write CSV using csv crate (Phase 8.1: proper CSV integrity)
    let csv_file = File::create(&args.output)?;
    let mut wtr = CsvWriter::from_writer(csv_file);

    // Header (strictly defined column order)
    let header = [
        "ts",
        "underlying",
        "session_id",
        "t1",
        "t2",
        "t3",
        "f1",
        "f2",
        "f3",
        "tty1",
        "tty2",
        "tty3",
        "iv1",
        "iv2",
        "iv3",
        "ts12",
        "ts23",
        "ts_curv",
        "cal12",
        "cal23",
        "sk1",
        "sk2",
        "sk3",
        "roll12",
        "rollc12",
    ];
    debug_assert_eq!(
        header.len(),
        CSV_COLUMN_COUNT,
        "Header column count mismatch"
    );
    wtr.write_record(header)?;

    // Data row
    let row = [
        features.ts.clone(),
        features.underlying.clone(),
        features.session_id.clone(),
        features.t1.clone(),
        features.t2.clone().unwrap_or_default(),
        features.t3.clone().unwrap_or_default(),
        format!("{:.2}", features.f1),
        features.f2.map(|v| format!("{:.2}", v)).unwrap_or_default(),
        features.f3.map(|v| format!("{:.2}", v)).unwrap_or_default(),
        format!("{:.6}", features.tty1),
        features
            .tty2
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .tty3
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        format!("{:.4}", features.iv1),
        features
            .iv2
            .map(|v| format!("{:.4}", v))
            .unwrap_or_default(),
        features
            .iv3
            .map(|v| format!("{:.4}", v))
            .unwrap_or_default(),
        features
            .ts12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .ts23
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .ts_curv
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .cal12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .cal23
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .sk1
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .sk2
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .sk3
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .roll12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
        features
            .rollc12
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default(),
    ];

    // Phase 8.1 integrity assertion: row column count must match header
    debug_assert_eq!(row.len(), CSV_COLUMN_COUNT, "Row column count mismatch");
    wtr.write_record(&row)?;
    wtr.flush()?;

    info!(
        "Wrote features to {:?} ({} columns)",
        args.output, CSV_COLUMN_COUNT
    );

    // Write manifest
    let manifest = FeatureManifest {
        version: "1.0".to_string(),
        policies: PolicyConfig {
            eta: ETA,
            v_min: V_MIN,
            epsilon_strike: EPSILON_STRIKE,
            k_n_normalized: K_N_NORMALIZED,
            strike_band: 20,
            expiry_policy: "T1T2T3".to_string(),
        },
        grid_points: GridConfig {
            k_low: 0.97,
            k_atm: 1.0,
            k_high: 1.03,
        },
        iv_solver: IVSolverConfig {
            method: "bisection".to_string(),
            vol_min: 0.001,
            vol_max: 5.0,
            tolerance: 1e-6,
        },
        session_info: SessionInfo {
            session_dir: args.session_dir.to_string_lossy().to_string(),
            underlying: args.underlying.clone(),
            expiries: slices.iter().map(|s| s.expiry.clone()).collect(),
            snapshot_count: 1,
        },
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&args.manifest, &manifest_json)?;
    info!("Wrote manifest to {:?}", args.manifest);

    // Print summary
    println!("\n=== SANOS PHASE 8: FEATURE EXTRACTION ===");
    println!("Underlying: {}", args.underlying);
    println!("Timestamp: {}", features.ts);
    println!();

    println!("--- Term Structure ---");
    println!("Expiry     Forward      TTY(d)    ATM IV");
    println!(
        "{:<10} {:>10.2} {:>8.1}    {:.2}%",
        features.t1,
        features.f1,
        features.tty1 * 365.0,
        features.iv1 * 100.0
    );
    if let (Some(t2), Some(f2), Some(tty2), Some(iv2)) =
        (&features.t2, features.f2, features.tty2, features.iv2)
    {
        println!(
            "{:<10} {:>10.2} {:>8.1}    {:.2}%",
            t2,
            f2,
            tty2 * 365.0,
            iv2 * 100.0
        );
    }
    if let (Some(t3), Some(f3), Some(tty3), Some(iv3)) =
        (&features.t3, features.f3, features.tty3, features.iv3)
    {
        println!(
            "{:<10} {:>10.2} {:>8.1}    {:.2}%",
            t3,
            f3,
            tty3 * 365.0,
            iv3 * 100.0
        );
    }
    println!();

    println!("--- Features ---");
    if let Some(ts12) = features.ts12 {
        println!("TS12 (term slope T1->T2):  {:.4}", ts12);
    }
    if let Some(ts23) = features.ts23 {
        println!("TS23 (term slope T2->T3):  {:.4}", ts23);
    }
    if let Some(ts_curv) = features.ts_curv {
        println!("TS_curv (curvature):       {:.4}", ts_curv);
    }
    println!();

    if let Some(cal12) = features.cal12 {
        println!("CAL12 (price gap T1->T2):  {:.6}", cal12);
    }
    if let Some(cal23) = features.cal23 {
        println!("CAL23 (price gap T2->T3):  {:.6}", cal23);
    }
    println!();

    if let Some(sk1) = features.sk1 {
        println!("SK1 (skew T1):             {:.4}", sk1);
    }
    if let Some(sk2) = features.sk2 {
        println!("SK2 (skew T2):             {:.4}", sk2);
    }
    if let Some(sk3) = features.sk3 {
        println!("SK3 (skew T3):             {:.4}", sk3);
    }
    println!();

    if let Some(roll12) = features.roll12 {
        println!("ROLL12 (IV carry T1->T2):  {:.4}", roll12);
    }
    if let Some(rollc12) = features.rollc12 {
        println!("ROLLC12 (price carry):     {:.6}", rollc12);
    }

    Ok(())
}
