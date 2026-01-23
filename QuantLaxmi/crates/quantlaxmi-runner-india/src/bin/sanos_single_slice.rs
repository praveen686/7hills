//! SANOS Single Slice Calibration
//!
//! Reads captured India NFO session data and calibrates SANOS for a single expiry
//! at a single timestamp.
//!
//! Usage:
//!   cargo run --bin sanos_single_slice -- --session-dir <path> --underlying NIFTY --expiry 26JAN

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use clap::Parser;
use kubera_options::sanos::{ExpirySlice, OptionQuote, SanosCalibrator};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
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
struct TickEvent {
    ts: DateTime<Utc>,
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

/// Parse option symbol: NIFTY26JAN25300CE -> (underlying, expiry, strike, is_call)
fn parse_symbol(symbol: &str) -> Option<(String, String, u32, bool)> {
    let symbol = symbol.to_uppercase();

    let is_call = symbol.ends_with("CE");
    let is_put = symbol.ends_with("PE");
    if !is_call && !is_put {
        return None;
    }

    let without_type = &symbol[..symbol.len() - 2];

    let (underlying, rest) = if without_type.starts_with("BANKNIFTY") {
        ("BANKNIFTY".to_string(), &without_type[9..])
    } else if without_type.starts_with("NIFTY") {
        ("NIFTY".to_string(), &without_type[5..])
    } else if without_type.starts_with("FINNIFTY") {
        ("FINNIFTY".to_string(), &without_type[8..])
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
            let (_und, _exp, strike, is_call) = parse_symbol(symbol).unwrap();

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

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(
            tracing::level_filters::LevelFilter::INFO.into(),
        ))
        .init();

    let args = Args::parse();

    info!("SANOS Single Slice Calibration");
    info!("Session: {:?}", args.session_dir);
    info!("Underlying: {}, Expiry: {}", args.underlying, args.expiry);

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

    // Certification summary
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

    // Save to JSON
    let json = serde_json::to_string_pretty(&sanos_slice)?;
    std::fs::write(&args.output, &json)?;
    info!("Saved SanosSlice to {:?}", args.output);

    Ok(())
}
