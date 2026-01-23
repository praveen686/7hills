//! SANOS Temporal Stability Analysis
//!
//! Runs SANOS calibration repeatedly over time for a single expiry to prove
//! temporal stability of the arbitrage-free surface.
//!
//! Phase 6 deliverable: Prove SANOS remains feasible, stable, and well-conditioned over time.

use anyhow::{Result, anyhow};
use chrono::{DateTime, Duration, Utc};
use clap::Parser;
use quantlaxmi_options::sanos::{ExpirySlice, OptionQuote, SanosCalibrator, SanosSlice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser, Debug)]
#[command(name = "sanos_temporal")]
#[command(about = "SANOS temporal stability analysis over time")]
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

    /// Snapshot interval in seconds
    #[arg(long, default_value = "30")]
    interval_secs: i64,

    /// Output directory for reports
    #[arg(long, default_value = "sanos_temporal_output")]
    output_dir: PathBuf,

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

/// Stability metrics between consecutive SANOS snapshots
#[derive(Debug, Clone, Serialize)]
pub struct StabilityMetrics {
    pub timestamp: DateTime<Utc>,
    pub prev_timestamp: Option<DateTime<Utc>>,

    // Drift metrics (L1 norms)
    pub density_drift_l1: f64,     // ||p_{t+Δt} - p_t||_1
    pub weight_drift_l1: f64,      // ||q_{t+Δt} - q_t||_1
    pub fitted_call_drift_l1: f64, // ||Ĉ_{t+Δt} - Ĉ_t||_1 - KEY: surface stability

    // Conditioning metrics
    pub active_weights: usize,    // Number of non-zero q_i
    pub min_positive_weight: f64, // Smallest positive q_i
    pub max_weight: f64,          // Largest q_i
    pub weight_entropy: f64,      // -Σ q_i log(q_i) - higher = more spread

    // Solver health
    pub lp_status: String,
    pub objective_value: f64,
    pub fit_error_max: f64,
    pub fit_error_mean: f64,

    // Forward stability
    pub forward: f64,
    pub forward_change_pct: f64,

    // Core constraints
    pub weights_sum: f64,
    pub weights_mean: f64,
}

/// Temporal analysis result
#[derive(Debug, Serialize)]
pub struct TemporalAnalysis {
    pub underlying: String,
    pub expiry: String,
    pub start_ts: DateTime<Utc>,
    pub end_ts: DateTime<Utc>,
    pub interval_secs: i64,
    pub num_snapshots: usize,
    pub num_feasible: usize,
    pub num_infeasible: usize,

    // Aggregate drift statistics
    pub max_density_drift: f64,
    pub mean_density_drift: f64,
    pub max_weight_drift: f64,
    pub mean_weight_drift: f64,
    pub max_fitted_call_drift: f64, // KEY: Ĉ drift (surface stability)
    pub mean_fitted_call_drift: f64,

    // Conditioning summary
    pub min_active_weights: usize,
    pub mean_active_weights: f64,
    pub min_weight_observed: f64,

    // Forward stability
    pub forward_range: (f64, f64),
    pub max_forward_change_pct: f64,

    // Two-tier certification (Lead directive)
    pub state_certified: bool, // Based on Ĉ drift (primary)
    pub param_certified: bool, // Based on q drift (secondary)
    pub certification_notes: Vec<String>,

    // Per-snapshot metrics
    pub snapshots: Vec<StabilityMetrics>,
}

/// Parse option symbol
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
    time_to_expiry: f64,
) -> Result<ExpirySlice> {
    let mut slice = ExpirySlice::new(underlying, expiry, target_ts, time_to_expiry);

    for (symbol, ticks) in symbol_ticks {
        // Find tick closest to target timestamp
        let closest_tick = ticks
            .iter()
            .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());

        if let Some(tick) = closest_tick {
            // Only use ticks within 5 seconds of target
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

/// Calculate L1 norm between two weight vectors
fn l1_norm(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Calculate entropy of weight distribution
fn entropy(weights: &[f64]) -> f64 {
    weights
        .iter()
        .filter(|&&w| w > 1e-10)
        .map(|&w| -w * w.ln())
        .sum()
}

/// Compute stability metrics between current and previous slice
fn compute_metrics(current: &SanosSlice, previous: Option<&SanosSlice>) -> StabilityMetrics {
    let weights = &current.weights;

    // Conditioning metrics
    let active_weights = weights.iter().filter(|&&w| w > 1e-10).count();
    let min_positive_weight = weights
        .iter()
        .filter(|&&w| w > 1e-10)
        .cloned()
        .fold(f64::MAX, f64::min);
    let max_weight = weights.iter().cloned().fold(0.0, f64::max);
    let weight_entropy = entropy(weights);

    // Drift metrics (only if we have previous)
    let (density_drift_l1, weight_drift_l1, fitted_call_drift_l1, forward_change_pct, prev_ts) =
        if let Some(prev) = previous {
            let w_drift = l1_norm(weights, &prev.weights);

            // Density is proportional to weights for same model grid
            // For proper density, we'd need to compute p(K) = dC/dK, but weights are proxy
            let d_drift = w_drift; // Simplified: use weight drift as density proxy

            // KEY: Fitted call price drift - this is what matters for state stability
            let c_drift = l1_norm(&current.fitted_calls, &prev.fitted_calls);

            let fwd_change = (current.forward - prev.forward) / prev.forward * 100.0;

            (d_drift, w_drift, c_drift, fwd_change, Some(prev.timestamp))
        } else {
            (0.0, 0.0, 0.0, 0.0, None)
        };

    StabilityMetrics {
        timestamp: current.timestamp,
        prev_timestamp: prev_ts,
        density_drift_l1,
        weight_drift_l1,
        fitted_call_drift_l1,
        active_weights,
        min_positive_weight: if min_positive_weight == f64::MAX {
            0.0
        } else {
            min_positive_weight
        },
        max_weight,
        weight_entropy,
        lp_status: current.diagnostics.lp_status.clone(),
        objective_value: current.diagnostics.objective_value,
        fit_error_max: current.diagnostics.max_fit_error,
        fit_error_mean: current.diagnostics.mean_fit_error,
        forward: current.forward,
        forward_change_pct,
        weights_sum: current.diagnostics.weights_sum,
        weights_mean: current.diagnostics.weights_mean,
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

    info!("SANOS Temporal Stability Analysis");
    info!("Session: {:?}", args.session_dir);
    info!("Underlying: {}, Expiry: {}", args.underlying, args.expiry);
    info!("Interval: {}s", args.interval_secs);

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Load ticks
    let symbol_ticks = load_ticks(&args.session_dir, &args.underlying, &args.expiry)?;
    info!("Loaded {} symbols", symbol_ticks.len());

    if symbol_ticks.is_empty() {
        return Err(anyhow!("No symbols found"));
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
    let session_duration = (max_ts - min_ts).num_seconds();

    info!(
        "Session: {} to {} ({} seconds)",
        min_ts, max_ts, session_duration
    );

    // Time to expiry (simplified: assume 3 days for weekly)
    let time_to_expiry = 3.0 / 365.0;

    // Run SANOS at each interval
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let mut snapshots: Vec<StabilityMetrics> = Vec::new();
    let mut slices: Vec<SanosSlice> = Vec::new();
    let mut num_feasible = 0;
    let mut num_infeasible = 0;

    let mut current_ts = min_ts;
    while current_ts <= max_ts {
        info!("Calibrating at {}", current_ts);

        let slice = build_slice(
            &symbol_ticks,
            &args.underlying,
            &args.expiry,
            current_ts,
            time_to_expiry,
        )?;

        if slice.calls.is_empty() || slice.puts.is_empty() {
            info!("  Skipping: insufficient data");
            current_ts += Duration::seconds(args.interval_secs);
            continue;
        }

        match calibrator.calibrate(&slice) {
            Ok(sanos_slice) => {
                let prev = slices.last();
                let metrics = compute_metrics(&sanos_slice, prev);

                info!(
                    "  F0={:.2}, Σq={:.6}, drift={:.6}",
                    metrics.forward, metrics.weights_sum, metrics.weight_drift_l1
                );

                snapshots.push(metrics);
                slices.push(sanos_slice);
                num_feasible += 1;
            }
            Err(e) => {
                info!("  INFEASIBLE: {}", e);
                num_infeasible += 1;
            }
        }

        current_ts += Duration::seconds(args.interval_secs);
    }

    info!(
        "Completed: {} feasible, {} infeasible",
        num_feasible, num_infeasible
    );

    // Compute aggregate statistics
    let drift_metrics: Vec<_> = snapshots.iter().skip(1).collect(); // Skip first (no previous)

    let max_density_drift = drift_metrics
        .iter()
        .map(|m| m.density_drift_l1)
        .fold(0.0, f64::max);
    let mean_density_drift = if !drift_metrics.is_empty() {
        drift_metrics
            .iter()
            .map(|m| m.density_drift_l1)
            .sum::<f64>()
            / drift_metrics.len() as f64
    } else {
        0.0
    };

    let max_weight_drift = drift_metrics
        .iter()
        .map(|m| m.weight_drift_l1)
        .fold(0.0, f64::max);
    let mean_weight_drift = if !drift_metrics.is_empty() {
        drift_metrics.iter().map(|m| m.weight_drift_l1).sum::<f64>() / drift_metrics.len() as f64
    } else {
        0.0
    };

    let min_active_weights = snapshots
        .iter()
        .map(|m| m.active_weights)
        .min()
        .unwrap_or(0);
    let mean_active_weights = if !snapshots.is_empty() {
        snapshots
            .iter()
            .map(|m| m.active_weights as f64)
            .sum::<f64>()
            / snapshots.len() as f64
    } else {
        0.0
    };

    let min_weight_observed = snapshots
        .iter()
        .map(|m| m.min_positive_weight)
        .filter(|&w| w > 0.0)
        .fold(f64::MAX, f64::min);

    let forward_min = snapshots.iter().map(|m| m.forward).fold(f64::MAX, f64::min);
    let forward_max = snapshots.iter().map(|m| m.forward).fold(0.0, f64::max);
    let max_forward_change = drift_metrics
        .iter()
        .map(|m| m.forward_change_pct.abs())
        .fold(0.0, f64::max);

    // Compute fitted call (Ĉ) drift statistics - KEY for state certification
    let max_fitted_call_drift = drift_metrics
        .iter()
        .map(|m| m.fitted_call_drift_l1)
        .fold(0.0, f64::max);
    let mean_fitted_call_drift = if !drift_metrics.is_empty() {
        drift_metrics
            .iter()
            .map(|m| m.fitted_call_drift_l1)
            .sum::<f64>()
            / drift_metrics.len() as f64
    } else {
        0.0
    };

    // Two-tier certification (Lead directive):
    // - STATE_CERTIFIED: Based on Ĉ (fitted call) drift - surface stability (PRIMARY)
    // - PARAM_CERTIFIED: Based on q (weight) drift - LP solution stability (SECONDARY)

    let mut state_certified = true;
    let mut param_certified = true;
    let mut notes = Vec::new();

    // Check feasibility (blocks both certifications)
    if num_infeasible > 0 {
        state_certified = false;
        param_certified = false;
        notes.push(format!("{} infeasible snapshots", num_infeasible));
    }

    // Check martingale constraints (blocks both certifications)
    for m in &snapshots {
        if (m.weights_sum - 1.0).abs() > 0.01 {
            state_certified = false;
            param_certified = false;
            notes.push(format!("Martingale sum violated at {}", m.timestamp));
        }
        if (m.weights_mean - 1.0).abs() > 0.05 {
            state_certified = false;
            param_certified = false;
            notes.push(format!("Martingale mean violated at {}", m.timestamp));
        }
    }

    // STATE certification: Ĉ drift threshold = 0.05 (5% of normalized surface)
    // This is the PRIMARY certification - surface is stable for downstream use
    if max_fitted_call_drift > 0.05 {
        state_certified = false;
        notes.push(format!(
            "Surface drift: max Ĉ drift {:.4} > 0.05 threshold",
            max_fitted_call_drift
        ));
    }

    // PARAM certification: q drift threshold = 0.5 (LP solution jumping)
    // This is SECONDARY - only affects numerical reproducibility
    if max_weight_drift > 0.5 {
        param_certified = false;
        notes.push(format!(
            "Parameter non-identifiability: max q drift {:.4} > 0.5 threshold",
            max_weight_drift
        ));
    }

    // Informational notes (don't block certification)
    if min_active_weights < 3 {
        notes.push(format!(
            "Corner solution detected: min {} active weights",
            min_active_weights
        ));
    }

    if max_forward_change > 1.0 {
        notes.push(format!("Large forward change: {:.2}%", max_forward_change));
    }

    // Summary notes
    if state_certified && param_certified {
        notes.push("All stability checks passed".to_string());
    } else if state_certified && !param_certified {
        notes.push(
            "Surface stable (Ĉ), but LP parameters jump (q) - downstream Ĉ use is safe".to_string(),
        );
    }

    let analysis = TemporalAnalysis {
        underlying: args.underlying.clone(),
        expiry: args.expiry.clone(),
        start_ts: min_ts,
        end_ts: max_ts,
        interval_secs: args.interval_secs,
        num_snapshots: snapshots.len(),
        num_feasible,
        num_infeasible,
        max_density_drift,
        mean_density_drift,
        max_weight_drift,
        mean_weight_drift,
        max_fitted_call_drift,
        mean_fitted_call_drift,
        min_active_weights,
        mean_active_weights,
        min_weight_observed: if min_weight_observed == f64::MAX {
            0.0
        } else {
            min_weight_observed
        },
        forward_range: (forward_min, forward_max),
        max_forward_change_pct: max_forward_change,
        state_certified,
        param_certified,
        certification_notes: notes,
        snapshots,
    };

    // Save JSON output
    let json_path = args.output_dir.join("temporal_analysis.json");
    let json = serde_json::to_string_pretty(&analysis)?;
    std::fs::write(&json_path, &json)?;
    info!("Saved analysis to {:?}", json_path);

    // Generate density evolution data for plotting
    let density_csv_path = args.output_dir.join("density_evolution.csv");
    let mut csv_file = File::create(&density_csv_path)?;
    writeln!(
        csv_file,
        "timestamp,weight_drift,forward,active_weights,entropy"
    )?;
    for m in &analysis.snapshots {
        writeln!(
            csv_file,
            "{},{:.6},{:.2},{},{:.4}",
            m.timestamp, m.weight_drift_l1, m.forward, m.active_weights, m.weight_entropy
        )?;
    }
    info!("Saved density evolution to {:?}", density_csv_path);

    // Print summary
    println!("\n=== SANOS TEMPORAL ANALYSIS ===");
    println!("Underlying: {}", analysis.underlying);
    println!("Expiry: {}", analysis.expiry);
    println!("Session: {} to {}", analysis.start_ts, analysis.end_ts);
    println!("Interval: {}s", analysis.interval_secs);
    println!();
    println!("--- Feasibility ---");
    println!("Total snapshots: {}", analysis.num_snapshots);
    println!("Feasible: {}", analysis.num_feasible);
    println!("Infeasible: {}", analysis.num_infeasible);
    println!();
    println!("--- Drift Statistics ---");
    println!(
        "Max q drift (L1):  {:.6} (weight stability)",
        analysis.max_weight_drift
    );
    println!("Mean q drift (L1): {:.6}", analysis.mean_weight_drift);
    println!(
        "Max Ĉ drift (L1):  {:.6} (surface stability)",
        analysis.max_fitted_call_drift
    );
    println!("Mean Ĉ drift (L1): {:.6}", analysis.mean_fitted_call_drift);
    println!();
    println!("--- Conditioning ---");
    println!("Min active weights: {}", analysis.min_active_weights);
    println!("Mean active weights: {:.1}", analysis.mean_active_weights);
    println!("Min positive weight: {:.6}", analysis.min_weight_observed);
    println!();
    println!("--- Forward Stability ---");
    println!(
        "Forward range: {:.2} - {:.2}",
        analysis.forward_range.0, analysis.forward_range.1
    );
    println!(
        "Max forward change: {:.2}%",
        analysis.max_forward_change_pct
    );
    println!();
    println!("=== TWO-TIER CERTIFICATION ===");
    println!();
    println!(
        "STATE_CERTIFIED (Ĉ drift < 0.05): {}",
        if analysis.state_certified {
            "✓ CERTIFIED"
        } else {
            "✗ NOT CERTIFIED"
        }
    );
    println!(
        "  → Surface is {} for downstream use",
        if analysis.state_certified {
            "STABLE"
        } else {
            "UNSTABLE"
        }
    );
    println!();
    println!(
        "PARAM_CERTIFIED (q drift < 0.5):  {}",
        if analysis.param_certified {
            "✓ CERTIFIED"
        } else {
            "✗ NOT CERTIFIED"
        }
    );
    println!(
        "  → LP parameters are {} between snapshots",
        if analysis.param_certified {
            "stable"
        } else {
            "jumping (non-identifiable)"
        }
    );
    println!();
    println!("Notes:");
    for note in &analysis.certification_notes {
        println!("  - {}", note);
    }

    Ok(())
}
