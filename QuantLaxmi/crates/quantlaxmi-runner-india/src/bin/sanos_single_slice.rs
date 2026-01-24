//! SANOS Single Slice Calibration
//!
//! Reads captured India NFO session data and calibrates SANOS for a single expiry
//! at a single timestamp.
//!
//! ## Modes (Commit D)
//! - **Manifest-driven**: When `session_manifest.json` exists, uses deterministic inventory.
//! - **Legacy**: Falls back to directory scanning + symbol parsing when no manifest.
//!
//! Usage:
//!   cargo run --bin sanos_single_slice -- --session-dir <path> --underlying NIFTY --expiry 26JAN

use anyhow::{Result, anyhow};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use clap::Parser;
use quantlaxmi_options::sanos::{ExpirySlice, OptionQuote, SanosCalibrator};
use quantlaxmi_runner_india::sanos_io::{
    SanosManifestInventory, SanosUnderlyingInventory, log_legacy_mode, log_manifest_mode,
    try_load_sanos_inventory,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "sanos_single_slice")]
#[command(about = "SANOS calibration for a single expiry at a single timestamp")]
struct Args {
    /// Session directory containing captured tick data
    #[arg(long)]
    session_dir: PathBuf,

    /// Underlying to calibrate (NIFTY or BANKNIFTY)
    #[arg(long, default_value = "NIFTY")]
    underlying: String,

    /// Expiry code (e.g., 26JAN)
    #[arg(long, default_value = "26JAN")]
    expiry: String,

    /// Time offset in seconds from session start (default: mid-session)
    #[arg(long)]
    time_offset_secs: Option<i64>,

    /// Output JSON file for SanosSlice
    #[arg(long, default_value = "sanos_slice.json")]
    output: PathBuf,

    /// SANOS smoothness parameter η (default: 0.25)
    #[arg(long, default_value = "0.25")]
    eta: f64,
}

/// Tick event from captured session
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields used for deserialization
struct TickEvent {
    ts: DateTime<Utc>,
    tradingsymbol: String,
    instrument_token: u32,
    bid_price: i64,
    ask_price: i64,
    bid_qty: u32,
    ask_qty: u32,
    ltp: i64,
    ltq: u32,
    volume: u64,
    price_exponent: i32,
    integrity_tier: String,
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

    let expiry = rest[..5].to_string(); // 26JAN
    let strike: u32 = rest[5..].parse().ok()?;

    Some((underlying, expiry, strike, is_call))
}

/// Load all ticks for symbols matching underlying and expiry
fn load_ticks(
    session_dir: &Path,
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

        // P3: Handle missing file_name gracefully
        let Some(fname) = path.file_name() else { continue; };
        let symbol = fname.to_string_lossy().to_string();

        // Parse and filter
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

            info!("Loaded {} ticks for {}", ticks.len(), symbol);
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
    time_to_expiry: f64,
) -> Result<ExpirySlice> {
    let mut slice = ExpirySlice::new(underlying, expiry, target_ts, time_to_expiry);

    for (symbol, ticks) in symbol_ticks {
        // Find tick closest to target timestamp
        let closest_tick = ticks
            .iter()
            .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());

        if let Some(tick) = closest_tick {
            // P3: Skip symbols that don't parse (shouldn't happen, but be defensive)
            let Some((_und, _exp, strike, is_call)) = parse_symbol(symbol) else {
                continue;
            };

            // Convert price with exponent (price_exponent=-2 means divide by 100)
            let price_mult = 10f64.powi(tick.price_exponent);
            let bid = tick.bid_price as f64 * price_mult;
            let ask = tick.ask_price as f64 * price_mult;

            // Skip invalid quotes
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
    session_dir: &Path,
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

            info!("Loaded {} ticks for {}", ticks.len(), symbol);
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
    let mut slice = ExpirySlice::new(
        &underlying_inv.underlying,
        &expiry_str,
        target_ts,
        time_to_exp,
    );

    // Get instruments for this expiry from manifest
    let instruments = underlying_inv.get_instruments_for_expiry(expiry);

    for instr in instruments {
        if let Some(ticks) = symbol_ticks.get(&instr.tradingsymbol) {
            let closest_tick = ticks
                .iter()
                .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());

            if let Some(tick) = closest_tick {
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

/// Match a short expiry code (e.g., "26JAN") to a NaiveDate in the manifest.
fn match_expiry_code(expiry_code: &str, available_expiries: &[NaiveDate]) -> Option<NaiveDate> {
    let code_upper = expiry_code.to_uppercase();

    for expiry in available_expiries {
        // Format expiry as "DDMMM" (e.g., "26JAN")
        let formatted = expiry.format("%d%b").to_string().to_uppercase();
        if formatted == code_upper {
            return Some(*expiry);
        }

        // Also try matching without leading zero (e.g., "9JAN" vs "09JAN")
        let formatted_no_zero = format!(
            "{}{}",
            expiry.format("%e").to_string().trim(), // Day without leading zero
            expiry.format("%b").to_string().to_uppercase()
        );
        if formatted_no_zero == code_upper {
            return Some(*expiry);
        }
    }

    None
}

/// Run manifest-driven calibration for a single slice.
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
                inventory
                    .underlyings
                    .iter()
                    .map(|u| &u.underlying)
                    .collect::<Vec<_>>()
            )
        })?;

    // Get expiries from manifest (sorted)
    let available_expiries = underlying_inv.get_sorted_expiries();
    info!(
        "Manifest-driven: {} expiries available, universe_sha256={}",
        available_expiries.len(),
        underlying_inv.universe_sha256
    );

    // Match the requested expiry code to a manifest expiry
    let target_expiry = match_expiry_code(&args.expiry, &available_expiries).ok_or_else(|| {
        anyhow!(
            "Expiry code {} not found in manifest. Available: {:?}",
            args.expiry,
            available_expiries
                .iter()
                .map(|e| e.format("%d%b").to_string())
                .collect::<Vec<_>>()
        )
    })?;

    info!(
        "Matched expiry code {} -> {}",
        args.expiry,
        target_expiry.format("%Y-%m-%d")
    );

    // Load ticks using manifest
    let symbol_ticks = load_ticks_manifest(&inventory.session_dir, underlying_inv, target_expiry)?;
    info!("Loaded {} symbols (manifest-driven)", symbol_ticks.len());

    if symbol_ticks.is_empty() {
        return Err(anyhow!(
            "No symbols found for {} {} in manifest",
            args.underlying,
            args.expiry
        ));
    }

    // Find session time range
    let mut min_ts: Option<DateTime<Utc>> = None;
    let mut max_ts: Option<DateTime<Utc>> = None;

    for ticks in symbol_ticks.values() {
        for tick in ticks {
            min_ts = Some(min_ts.map_or(tick.ts, |m| m.min(tick.ts)));
            max_ts = Some(max_ts.map_or(tick.ts, |m| m.max(tick.ts)));
        }
    }

    let min_ts = min_ts.ok_or_else(|| anyhow!("No ticks found"))?;
    let max_ts = max_ts.ok_or_else(|| anyhow!("No ticks found"))?;

    info!("Session range: {} to {}", min_ts, max_ts);

    // Select target timestamp
    let target_ts = if let Some(offset) = args.time_offset_secs {
        min_ts + Duration::seconds(offset)
    } else {
        min_ts + (max_ts - min_ts) / 2
    };

    info!("Target timestamp: {}", target_ts);

    // Calculate time to expiry from manifest date
    let time_to_expiry = time_to_expiry_from_date(target_ts, target_expiry);

    // Build slice using manifest
    let slice = build_slice_manifest(
        &symbol_ticks,
        underlying_inv,
        target_expiry,
        target_ts,
        time_to_expiry,
    )?;

    info!(
        "Built slice: {} calls, {} puts",
        slice.calls.len(),
        slice.puts.len()
    );

    if slice.calls.is_empty() || slice.puts.is_empty() {
        return Err(anyhow!("Need both calls and puts for SANOS calibration"));
    }

    // Run SANOS calibration
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let sanos_slice = calibrator.calibrate(&slice)?;

    // Print results
    println!("\n{}", sanos_slice);

    // Certification summary
    print_certification_summary(&sanos_slice);

    // Save to JSON
    let json = serde_json::to_string_pretty(&sanos_slice)?;
    std::fs::write(&args.output, &json)?;
    info!(
        "Saved SanosSlice to {:?} (universe_sha256={})",
        args.output, underlying_inv.universe_sha256
    );

    Ok(())
}

/// Print SANOS certification summary.
fn print_certification_summary(sanos_slice: &quantlaxmi_options::sanos::SanosSlice) {
    println!("=== SANOS v0 CERTIFICATION ===");
    let cert_status = if sanos_slice.diagnostics.weights_sum > 0.99
        && sanos_slice.diagnostics.weights_sum < 1.01
        && sanos_slice.diagnostics.weights_mean > 0.95
        && sanos_slice.diagnostics.weights_mean < 1.05
        && sanos_slice.diagnostics.boundary_check
        && sanos_slice.diagnostics.spread_compliance > 80.0
    {
        "CERTIFIED"
    } else {
        "FAILED"
    };

    println!("Status: {}", cert_status);
    println!(
        "Martingale Sum:  {} (should be 1.0)",
        if (sanos_slice.diagnostics.weights_sum - 1.0).abs() < 0.01 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!(
        "Martingale Mean: {} (should be 1.0)",
        if (sanos_slice.diagnostics.weights_mean - 1.0).abs() < 0.05 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!(
        "Boundary Check:  {}",
        if sanos_slice.diagnostics.boundary_check {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!(
        "Spread Compliance: {:.1}% {}",
        sanos_slice.diagnostics.spread_compliance,
        if sanos_slice.diagnostics.spread_compliance > 80.0 {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "Convexity:       {} violations {}",
        sanos_slice.diagnostics.convexity_violations,
        if sanos_slice.diagnostics.convexity_violations <= 3 {
            "✓"
        } else {
            "⚠"
        }
    );
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::level_filters::LevelFilter::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!("SANOS Single Slice Calibration");
    info!("Session: {:?}", args.session_dir);
    info!("Underlying: {}, Expiry: {}", args.underlying, args.expiry);

    // Try manifest-driven mode (Commit D)
    if let Some(inventory) = try_load_sanos_inventory(&args.session_dir)? {
        log_manifest_mode(&inventory);
        return run_manifest_mode(&args, &inventory);
    }

    // Legacy mode: directory scanning + symbol parsing
    log_legacy_mode(&args.session_dir);

    // Load ticks
    let symbol_ticks = load_ticks(&args.session_dir, &args.underlying, &args.expiry)?;
    info!("Loaded {} symbols", symbol_ticks.len());

    if symbol_ticks.is_empty() {
        return Err(anyhow!(
            "No symbols found for {} {}",
            args.underlying,
            args.expiry
        ));
    }

    // Find session time range
    let mut min_ts: Option<DateTime<Utc>> = None;
    let mut max_ts: Option<DateTime<Utc>> = None;

    for ticks in symbol_ticks.values() {
        for tick in ticks {
            min_ts = Some(min_ts.map_or(tick.ts, |m| m.min(tick.ts)));
            max_ts = Some(max_ts.map_or(tick.ts, |m| m.max(tick.ts)));
        }
    }

    let min_ts = min_ts.ok_or_else(|| anyhow!("No ticks found"))?;
    let max_ts = max_ts.ok_or_else(|| anyhow!("No ticks found"))?;

    info!("Session range: {} to {}", min_ts, max_ts);

    // Select target timestamp
    let target_ts = if let Some(offset) = args.time_offset_secs {
        min_ts + Duration::seconds(offset)
    } else {
        // Default: mid-session
        min_ts + (max_ts - min_ts) / 2
    };

    info!("Target timestamp: {}", target_ts);

    // Calculate time to expiry (assuming expiry is at 15:30 IST on expiry day)
    // For 26JAN, expiry is 2026-01-29 (next Thursday) at 15:30 IST
    // Simplified: assume 3 days for weekly expiry
    let time_to_expiry = 3.0 / 365.0; // ~3 days

    // Build slice
    let slice = build_slice(
        &symbol_ticks,
        &args.underlying,
        &args.expiry,
        target_ts,
        time_to_expiry,
    )?;

    info!(
        "Built slice: {} calls, {} puts",
        slice.calls.len(),
        slice.puts.len()
    );

    if slice.calls.is_empty() || slice.puts.is_empty() {
        return Err(anyhow!("Need both calls and puts for SANOS calibration"));
    }

    // Run SANOS calibration
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let sanos_slice = calibrator.calibrate(&slice)?;

    // Print results
    println!("\n{}", sanos_slice);

    // Certification summary (reuse shared function)
    print_certification_summary(&sanos_slice);

    // Save to JSON
    let json = serde_json::to_string_pretty(&sanos_slice)?;
    std::fs::write(&args.output, &json)?;
    info!("Saved SanosSlice to {:?}", args.output);

    Ok(())
}
