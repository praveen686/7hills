//! SANOS Multi-Expiry Term Structure
//!
//! Phase 7: Calibrates SANOS across multiple expiries and checks calendar arbitrage constraints.
//!
//! Calendar arbitrage constraint: C(K, T1) ≤ C(K, T2) for T1 < T2 (at same strike K)
//! This ensures forward-start call prices are non-negative.
//!
//! Usage:
//!   cargo run --bin sanos_multi_expiry -- --session-dir <path> --underlying NIFTY

use anyhow::{anyhow, Result};
use chrono::{DateTime, NaiveDate, Utc};
use clap::Parser;
use kubera_options::sanos::{ExpirySlice, OptionQuote, SanosCalibrator, SanosSlice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "sanos_multi_expiry")]
#[command(about = "SANOS multi-expiry term structure calibration")]
struct Args {
    /// Session directory containing captured tick data
    #[arg(long)]
    session_dir: PathBuf,

    /// Underlying to calibrate (NIFTY or BANKNIFTY)
    #[arg(long, default_value = "NIFTY")]
    underlying: String,

    /// Output JSON file for multi-expiry result
    #[arg(long, default_value = "sanos_multi_expiry.json")]
    output: PathBuf,

    /// SANOS smoothness parameter η (default: 0.25)
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

/// Calendar arbitrage violation
#[derive(Debug, Clone, Serialize)]
struct CalendarViolation {
    strike: f64,
    near_expiry: String,
    far_expiry: String,
    near_call_price: f64,
    far_call_price: f64,
    violation_amount: f64,  // near - far (should be ≤ 0)
}

/// Multi-expiry calibration result
#[derive(Debug, Serialize)]
struct MultiExpiryResult {
    underlying: String,
    calibration_ts: DateTime<Utc>,
    num_expiries: usize,
    expiries: Vec<String>,

    // Per-expiry results
    slices: Vec<SanosSlice>,

    // Calendar arbitrage checks
    calendar_violations: Vec<CalendarViolation>,
    num_calendar_violations: usize,
    max_calendar_violation: f64,
    calendar_arbitrage_free: bool,

    // Term structure summary
    forwards: Vec<(String, f64)>,       // expiry -> forward
    time_to_expiries: Vec<(String, f64)>, // expiry -> TTY in years

    // Certification
    state_certified: bool,
    certification_notes: Vec<String>,
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

/// Convert expiry code to date for ordering
/// Supports two formats:
/// - "26JAN" = 26th January 2026 (monthly/near-weekly)
/// - "26203" = YYMMDD compact format (year 26, month 02, day 03) - weekly format
fn expiry_to_date(expiry: &str) -> Option<NaiveDate> {
    // Try DDMMM format first (e.g., "26JAN")
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

        // Try YYMMDD compact format (e.g., "26203" = 2026-02-03)
        // Format: YY M DD where M is single digit 1-9 for Jan-Sep, or "O"/"N"/"D" for Oct/Nov/Dec
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
        // Assume expiry at 15:30 IST (10:00 UTC)
        let exp_datetime = exp_date.and_hms_opt(10, 0, 0)
            .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));

        if let Some(exp_dt) = exp_datetime {
            let days = (exp_dt - now).num_seconds() as f64 / 86400.0;
            return (days / 365.0).max(1.0 / 365.0);  // Minimum 1 day
        }
    }

    // Fallback: assume 7 days
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

    // Sort by expiry date
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

/// Check calendar arbitrage between two slices
/// Returns violations where near_call > far_call at same strike
fn check_calendar_arbitrage(
    near: &SanosSlice,
    far: &SanosSlice,
    tolerance: f64,
) -> Vec<CalendarViolation> {
    let mut violations = Vec::new();

    // Build strike -> fitted_call map for far expiry
    let far_map: HashMap<u64, f64> = far.model_strikes
        .iter()
        .zip(far.fitted_calls.iter())
        .map(|(&k, &c)| ((k * 1000.0) as u64, c))
        .collect();

    // Check each near strike against far
    for (i, &strike) in near.model_strikes.iter().enumerate() {
        let strike_key = (strike * 1000.0) as u64;

        if let Some(&far_call) = far_map.get(&strike_key) {
            let near_call = near.fitted_calls[i];

            // Calendar arbitrage: near_call should be ≤ far_call
            // (further expiry should be worth at least as much as near expiry)
            if near_call > far_call + tolerance {
                violations.push(CalendarViolation {
                    strike,
                    near_expiry: near.expiry.clone(),
                    far_expiry: far.expiry.clone(),
                    near_call_price: near_call,
                    far_call_price: far_call,
                    violation_amount: near_call - far_call,
                });
            }
        }
    }

    violations
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::level_filters::LevelFilter::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!("SANOS Multi-Expiry Term Structure");
    info!("Session: {:?}", args.session_dir);
    info!("Underlying: {}", args.underlying);

    // Discover all expiries
    let expiries = discover_expiries(&args.session_dir, &args.underlying)?;
    info!("Found {} expiries: {:?}", expiries.len(), expiries);

    if expiries.is_empty() {
        return Err(anyhow!("No expiries found for {}", args.underlying));
    }

    // Find common timestamp (mid-session of first expiry data)
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
    let target_ts = min_ts + (max_ts - min_ts) / 2;

    info!("Calibration timestamp: {}", target_ts);

    // Calibrate each expiry
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let mut slices: Vec<SanosSlice> = Vec::new();
    let mut forwards: Vec<(String, f64)> = Vec::new();
    let mut time_to_expiries: Vec<(String, f64)> = Vec::new();

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

                forwards.push((expiry.clone(), sanos_slice.forward));
                time_to_expiries.push((expiry.clone(), tte));
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

    // Check calendar arbitrage between consecutive expiries
    let mut all_violations: Vec<CalendarViolation> = Vec::new();
    let tolerance = 0.0001;  // Tolerance for numerical noise

    for i in 0..slices.len() - 1 {
        let near = &slices[i];
        let far = &slices[i + 1];

        let violations = check_calendar_arbitrage(near, far, tolerance);
        if !violations.is_empty() {
            info!(
                "Calendar violations between {} and {}: {}",
                near.expiry, far.expiry, violations.len()
            );
        }
        all_violations.extend(violations);
    }

    let max_violation = all_violations
        .iter()
        .map(|v| v.violation_amount)
        .fold(0.0, f64::max);

    let calendar_free = all_violations.is_empty();

    // Certification
    let mut notes = Vec::new();
    let mut state_certified = true;

    // Check each slice's martingale constraints
    for slice in &slices {
        if (slice.diagnostics.weights_sum - 1.0).abs() > 0.01 {
            state_certified = false;
            notes.push(format!(
                "{}: martingale sum = {:.6}",
                slice.expiry, slice.diagnostics.weights_sum
            ));
        }
    }

    if !calendar_free {
        state_certified = false;
        notes.push(format!(
            "{} calendar arbitrage violations (max: {:.6})",
            all_violations.len(), max_violation
        ));
    }

    if state_certified {
        notes.push("All martingale and calendar arbitrage checks passed".to_string());
    }

    let result = MultiExpiryResult {
        underlying: args.underlying.clone(),
        calibration_ts: target_ts,
        num_expiries: slices.len(),
        expiries: slices.iter().map(|s| s.expiry.clone()).collect(),
        slices,
        calendar_violations: all_violations.clone(),
        num_calendar_violations: all_violations.len(),
        max_calendar_violation: max_violation,
        calendar_arbitrage_free: calendar_free,
        forwards,
        time_to_expiries,
        state_certified,
        certification_notes: notes.clone(),
    };

    // Save JSON output
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.output, &json)?;
    info!("Saved result to {:?}", args.output);

    // Print summary
    println!("\n=== SANOS MULTI-EXPIRY TERM STRUCTURE ===");
    println!("Underlying: {}", result.underlying);
    println!("Calibration: {}", result.calibration_ts);
    println!();

    println!("--- Term Structure ---");
    println!("{:<10} {:>12} {:>10}", "Expiry", "Forward", "TTY (days)");
    for ((exp, fwd), (_, tte)) in result.forwards.iter().zip(result.time_to_expiries.iter()) {
        println!("{:<10} {:>12.2} {:>10.1}", exp, fwd, tte * 365.0);
    }
    println!();

    println!("--- Calendar Arbitrage ---");
    if result.calendar_arbitrage_free {
        println!("Status: ✓ ARBITRAGE-FREE");
        println!("No violations detected between expiries");
    } else {
        println!("Status: ✗ VIOLATIONS DETECTED");
        println!("Violations: {}", result.num_calendar_violations);
        println!("Max violation: {:.6}", result.max_calendar_violation);
        for v in result.calendar_violations.iter().take(5) {
            println!(
                "  K={:.0}: {}={:.4} > {}={:.4}",
                v.strike, v.near_expiry, v.near_call_price, v.far_expiry, v.far_call_price
            );
        }
        if result.num_calendar_violations > 5 {
            println!("  ... and {} more", result.num_calendar_violations - 5);
        }
    }
    println!();

    println!("=== CERTIFICATION ===");
    println!(
        "STATE_CERTIFIED: {}",
        if result.state_certified { "✓ CERTIFIED" } else { "✗ NOT CERTIFIED" }
    );
    for note in &result.certification_notes {
        println!("  - {}", note);
    }

    Ok(())
}
