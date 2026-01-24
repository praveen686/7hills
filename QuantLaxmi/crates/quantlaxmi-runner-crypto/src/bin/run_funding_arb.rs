//! Funding Rate Arbitrage Signal Generator.
//!
//! Generates ENTER/EXIT signals for funding rate arbitrage based on:
//! - Funding rate threshold (min rate to trigger entry)
//! - Basis threshold (min premium/discount)
//! - Spread threshold (max bid-ask spread)
//! - Quote staleness (max quote age)
//!
//! ## Strategy Overview
//!
//! When funding rate is positive (shorts pay longs):
//! - ENTER: long_spot + short_perp (collect funding)
//! - Profitable when: funding_collected > basis_cost + fees
//!
//! When funding rate is negative (longs pay shorts):
//! - ENTER: short_spot + long_perp (collect funding)
//! - Profitable when: funding_collected > basis_cost + fees
//!
//! ## Exit Policies
//! - `funding_settle`: Exit after funding settlement (8h window)
//! - `time_stop`: Exit after fixed duration
//! - `basis_revert`: Exit when basis mean-reverts
//!
//! ## Usage
//! ```bash
//! cargo run --bin run_funding_arb -- \
//!     --session-dir data/perp_sessions/perp_20260124_100000 \
//!     --min-funding-rate-bps 1.0 \
//!     --min-basis-bps 5.0 \
//!     --max-spread-bps 10.0 \
//!     --max-quote-age-ms 500
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use quantlaxmi_runner_crypto::binance_funding_capture::FundingEvent;
use quantlaxmi_runner_crypto::binance_perp_session::load_perp_session_manifest;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "run_funding_arb")]
#[command(about = "Generate funding rate arbitrage signals from captured session")]
struct Args {
    /// Path to perp session directory
    #[arg(long)]
    session_dir: PathBuf,

    /// Output directory for signals (default: session_dir/signals)
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Minimum funding rate to trigger entry (basis points per 8h)
    #[arg(long, default_value = "1.0")]
    min_funding_rate_bps: f64,

    /// Minimum basis to trigger entry (basis points)
    #[arg(long, default_value = "5.0")]
    min_basis_bps: f64,

    /// Maximum bid-ask spread allowed (basis points)
    #[arg(long, default_value = "10.0")]
    max_spread_bps: f64,

    /// Maximum quote age in milliseconds
    #[arg(long, default_value = "500")]
    max_quote_age_ms: u64,

    /// Exit policy: funding_settle, time_stop, basis_revert
    #[arg(long, default_value = "funding_settle")]
    exit_policy: String,

    /// Time stop duration in seconds (for time_stop policy)
    #[arg(long, default_value = "28800")]
    time_stop_secs: u64,

    /// Basis revert threshold in bps (for basis_revert policy)
    #[arg(long, default_value = "2.0")]
    basis_revert_bps: f64,

    /// Cooldown between entries in seconds
    #[arg(long, default_value = "600")]
    cooldown_secs: u64,

    /// Symbols to process (comma-separated, default: all in session)
    #[arg(long)]
    symbols: Option<String>,
}

// =============================================================================
// Data Structures
// =============================================================================

/// Quote event from spot or perp bookTicker stream.
#[derive(Debug, Clone, Deserialize)]
struct QuoteEvent {
    ts: DateTime<Utc>,
    #[allow(dead_code)]
    symbol: String,
    bid_price_mantissa: i64,
    ask_price_mantissa: i64,
    #[allow(dead_code)]
    bid_qty_mantissa: i64,
    #[allow(dead_code)]
    ask_qty_mantissa: i64,
    price_exponent: i8,
    #[serde(default)]
    #[allow(dead_code)]
    qty_exponent: i8,
}

impl QuoteEvent {
    fn bid_f64(&self) -> f64 {
        self.bid_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    fn ask_f64(&self) -> f64 {
        self.ask_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    fn mid_f64(&self) -> f64 {
        (self.bid_f64() + self.ask_f64()) / 2.0
    }

    fn spread_bps(&self) -> f64 {
        let mid = self.mid_f64();
        if mid > 0.0 {
            (self.ask_f64() - self.bid_f64()) / mid * 10000.0
        } else {
            f64::MAX
        }
    }
}

/// Signal intent for funding arbitrage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalIntent {
    /// Long spot, short perp (collect positive funding)
    LongSpotShortPerp,
    /// Short spot, long perp (collect negative funding)
    ShortSpotLongPerp,
}

/// Exit policy for the position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitPolicy {
    /// Exit after next funding settlement
    FundingSettle,
    /// Exit after fixed time duration
    TimeStop,
    /// Exit when basis mean-reverts
    BasisRevert,
}

impl std::str::FromStr for ExitPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "funding_settle" => Ok(ExitPolicy::FundingSettle),
            "time_stop" => Ok(ExitPolicy::TimeStop),
            "basis_revert" => Ok(ExitPolicy::BasisRevert),
            _ => anyhow::bail!("Unknown exit policy: {}", s),
        }
    }
}

/// Funding arbitrage signal event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbSignal {
    pub schema_version: u32,
    pub ts: DateTime<Utc>,
    pub symbol: String,

    // Entry conditions
    pub intent: SignalIntent,
    pub exit_policy: ExitPolicy,

    // Prices at signal time (mantissa form for determinism)
    pub spot_bid_mantissa: i64,
    pub spot_ask_mantissa: i64,
    pub perp_bid_mantissa: i64,
    pub perp_ask_mantissa: i64,
    pub price_exponent: i8,

    // Funding info
    pub funding_rate_mantissa: i64,
    pub rate_exponent: i8,
    pub next_funding_time_ms: i64,

    // Computed values (for audit, stored as f64 for readability)
    pub basis_bps: f64,
    pub funding_rate_bps: f64,
    pub spot_spread_bps: f64,
    pub perp_spread_bps: f64,

    // Exit parameters
    pub exit_time_stop_secs: Option<u64>,
    pub exit_basis_revert_bps: Option<f64>,

    // Gate pass/fail codes
    pub reason_codes: Vec<String>,
}

/// Gate counters for signal generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateCounters {
    pub total_funding_events: usize,
    pub total_decisions: usize,

    // Quote gates (Q1-Q4)
    pub q1_spot_missing: usize,
    pub q2_perp_missing: usize,
    pub q3_spot_stale: usize,
    pub q4_perp_stale: usize,
    pub q5_spot_spread_wide: usize,
    pub q6_perp_spread_wide: usize,

    // Economic gates (E1-E3)
    pub e1_funding_too_low: usize,
    pub e2_basis_too_low: usize,
    pub e3_cooldown_blocked: usize,

    // Pass counters
    pub all_gates_pass: usize,
    pub signals_generated: usize,
}

/// Run manifest for signal generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRunManifest {
    pub schema_version: u32,
    pub created_at_utc: String,
    pub run_id: String,
    pub session_dir: String,
    pub config: SignalRunConfig,
    pub gate_counters: GateCounters,
    pub symbols_processed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRunConfig {
    pub min_funding_rate_bps: f64,
    pub min_basis_bps: f64,
    pub max_spread_bps: f64,
    pub max_quote_age_ms: u64,
    pub exit_policy: String,
    pub time_stop_secs: u64,
    pub basis_revert_bps: f64,
    pub cooldown_secs: u64,
}

// =============================================================================
// Main Logic
// =============================================================================

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let exit_policy: ExitPolicy = args.exit_policy.parse()?;

    // Load session manifest
    let manifest =
        load_perp_session_manifest(&args.session_dir).context("Failed to load session manifest")?;

    tracing::info!("=== Funding Arb Signal Generation ===");
    tracing::info!("Session: {:?}", args.session_dir);
    tracing::info!(
        "Symbols in session: {:?}",
        manifest
            .symbols
            .iter()
            .map(|s| &s.symbol)
            .collect::<Vec<_>>()
    );

    // Determine symbols to process
    let symbols: Vec<String> = if let Some(ref sym_list) = args.symbols {
        sym_list
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect()
    } else {
        manifest.symbols.iter().map(|s| s.symbol.clone()).collect()
    };

    // Create output directory
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| args.session_dir.join("signals"));
    std::fs::create_dir_all(&output_dir)?;

    // Generate run ID
    let run_id = uuid::Uuid::new_v4().to_string()[..16].to_string();
    let run_dir = output_dir.join(&run_id);
    std::fs::create_dir_all(&run_dir)?;

    tracing::info!("Run ID: {}", run_id);
    tracing::info!("Output: {:?}", run_dir);

    // Process each symbol
    let mut all_signals: Vec<FundingArbSignal> = Vec::new();
    let mut total_counters = GateCounters::default();

    for symbol in &symbols {
        tracing::info!("Processing symbol: {}", symbol);

        let (signals, counters) = process_symbol(&args.session_dir, symbol, &args, exit_policy)?;

        tracing::info!(
            "  {} signals generated ({} funding events, {} passed gates)",
            signals.len(),
            counters.total_funding_events,
            counters.all_gates_pass
        );

        // Merge counters
        total_counters.total_funding_events += counters.total_funding_events;
        total_counters.total_decisions += counters.total_decisions;
        total_counters.q1_spot_missing += counters.q1_spot_missing;
        total_counters.q2_perp_missing += counters.q2_perp_missing;
        total_counters.q3_spot_stale += counters.q3_spot_stale;
        total_counters.q4_perp_stale += counters.q4_perp_stale;
        total_counters.q5_spot_spread_wide += counters.q5_spot_spread_wide;
        total_counters.q6_perp_spread_wide += counters.q6_perp_spread_wide;
        total_counters.e1_funding_too_low += counters.e1_funding_too_low;
        total_counters.e2_basis_too_low += counters.e2_basis_too_low;
        total_counters.e3_cooldown_blocked += counters.e3_cooldown_blocked;
        total_counters.all_gates_pass += counters.all_gates_pass;
        total_counters.signals_generated += counters.signals_generated;

        all_signals.extend(signals);
    }

    // Sort signals by timestamp
    all_signals.sort_by(|a, b| a.ts.cmp(&b.ts).then(a.symbol.cmp(&b.symbol)));

    // Write signals.jsonl
    let signals_path = run_dir.join("signals.jsonl");
    let mut signals_file = std::fs::File::create(&signals_path)?;
    for signal in &all_signals {
        let line = serde_json::to_string(signal)?;
        std::io::Write::write_all(&mut signals_file, line.as_bytes())?;
        std::io::Write::write_all(&mut signals_file, b"\n")?;
    }
    tracing::info!("Wrote {} signals to {:?}", all_signals.len(), signals_path);

    // Write run manifest
    let run_manifest = SignalRunManifest {
        schema_version: 1,
        created_at_utc: Utc::now().to_rfc3339(),
        run_id: run_id.clone(),
        session_dir: args.session_dir.to_string_lossy().to_string(),
        config: SignalRunConfig {
            min_funding_rate_bps: args.min_funding_rate_bps,
            min_basis_bps: args.min_basis_bps,
            max_spread_bps: args.max_spread_bps,
            max_quote_age_ms: args.max_quote_age_ms,
            exit_policy: args.exit_policy.clone(),
            time_stop_secs: args.time_stop_secs,
            basis_revert_bps: args.basis_revert_bps,
            cooldown_secs: args.cooldown_secs,
        },
        gate_counters: total_counters.clone(),
        symbols_processed: symbols.clone(),
    };

    let manifest_path = run_dir.join("run_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&run_manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;

    // Print summary
    tracing::info!("=== Signal Generation Summary ===");
    tracing::info!(
        "Total funding events: {}",
        total_counters.total_funding_events
    );
    tracing::info!("Total decisions: {}", total_counters.total_decisions);
    tracing::info!("Gates passed: {}", total_counters.all_gates_pass);
    tracing::info!("Signals generated: {}", total_counters.signals_generated);
    tracing::info!("");
    tracing::info!("Gate failures:");
    tracing::info!("  Q1 spot_missing: {}", total_counters.q1_spot_missing);
    tracing::info!("  Q2 perp_missing: {}", total_counters.q2_perp_missing);
    tracing::info!("  Q3 spot_stale: {}", total_counters.q3_spot_stale);
    tracing::info!("  Q4 perp_stale: {}", total_counters.q4_perp_stale);
    tracing::info!(
        "  Q5 spot_spread_wide: {}",
        total_counters.q5_spot_spread_wide
    );
    tracing::info!(
        "  Q6 perp_spread_wide: {}",
        total_counters.q6_perp_spread_wide
    );
    tracing::info!(
        "  E1 funding_too_low: {}",
        total_counters.e1_funding_too_low
    );
    tracing::info!("  E2 basis_too_low: {}", total_counters.e2_basis_too_low);
    tracing::info!(
        "  E3 cooldown_blocked: {}",
        total_counters.e3_cooldown_blocked
    );

    Ok(())
}

/// Process a single symbol and generate signals.
fn process_symbol(
    session_dir: &Path,
    symbol: &str,
    args: &Args,
    exit_policy: ExitPolicy,
) -> Result<(Vec<FundingArbSignal>, GateCounters)> {
    let sym_dir = session_dir.join(symbol.to_uppercase());

    // Load funding events
    let funding_path = sym_dir.join("funding.jsonl");
    let funding_events = load_funding_events(&funding_path)?;

    // Load spot quotes
    let spot_path = sym_dir.join("spot_quotes.jsonl");
    let spot_quotes = load_quote_events(&spot_path)?;

    // Load perp quotes
    let perp_path = sym_dir.join("perp_quotes.jsonl");
    let perp_quotes = load_quote_events(&perp_path)?;

    // Build quote index for fast lookup
    let spot_index = build_quote_index(&spot_quotes);
    let perp_index = build_quote_index(&perp_quotes);

    let mut signals = Vec::new();
    let mut counters = GateCounters::default();
    let mut last_entry_ts: Option<DateTime<Utc>> = None;

    counters.total_funding_events = funding_events.len();

    for funding in &funding_events {
        counters.total_decisions += 1;

        let reason_codes = Vec::new();

        // Q1: Find spot quote at funding timestamp
        let spot_quote = find_quote_at_or_before(&spot_index, funding.ts, args.max_quote_age_ms);
        let Some(spot_quote) = spot_quote else {
            counters.q1_spot_missing += 1;
            continue;
        };

        // Check spot staleness
        let spot_age_ms = (funding.ts - spot_quote.ts)
            .num_milliseconds()
            .unsigned_abs();
        if spot_age_ms > args.max_quote_age_ms {
            counters.q3_spot_stale += 1;
            continue;
        }

        // Q2: Find perp quote at funding timestamp
        let perp_quote = find_quote_at_or_before(&perp_index, funding.ts, args.max_quote_age_ms);
        let Some(perp_quote) = perp_quote else {
            counters.q2_perp_missing += 1;
            continue;
        };

        // Check perp staleness
        let perp_age_ms = (funding.ts - perp_quote.ts)
            .num_milliseconds()
            .unsigned_abs();
        if perp_age_ms > args.max_quote_age_ms {
            counters.q4_perp_stale += 1;
            continue;
        }

        // Q5: Check spot spread
        let spot_spread_bps = spot_quote.spread_bps();
        if spot_spread_bps > args.max_spread_bps {
            counters.q5_spot_spread_wide += 1;
            continue;
        }

        // Q6: Check perp spread
        let perp_spread_bps = perp_quote.spread_bps();
        if perp_spread_bps > args.max_spread_bps {
            counters.q6_perp_spread_wide += 1;
            continue;
        }

        // E1: Check funding rate threshold
        let funding_rate_bps = funding.funding_rate_bps();
        if funding_rate_bps.abs() < args.min_funding_rate_bps {
            counters.e1_funding_too_low += 1;
            continue;
        }

        // E2: Check basis threshold
        let spot_mid = spot_quote.mid_f64();
        let perp_mid = perp_quote.mid_f64();
        let basis_bps = if spot_mid > 0.0 {
            (perp_mid - spot_mid) / spot_mid * 10000.0
        } else {
            0.0
        };

        // For positive funding, we want perp > spot (positive basis)
        // For negative funding, we want spot > perp (negative basis)
        let basis_favorable = if funding_rate_bps > 0.0 {
            basis_bps >= args.min_basis_bps
        } else {
            basis_bps <= -args.min_basis_bps
        };

        if !basis_favorable {
            counters.e2_basis_too_low += 1;
            continue;
        }

        // E3: Check cooldown
        if let Some(last_ts) = last_entry_ts {
            let since_last = (funding.ts - last_ts).num_seconds();
            if since_last < args.cooldown_secs as i64 {
                counters.e3_cooldown_blocked += 1;
                continue;
            }
        }

        // All gates passed
        counters.all_gates_pass += 1;

        // Determine intent based on funding rate
        let intent = if funding_rate_bps > 0.0 {
            SignalIntent::LongSpotShortPerp
        } else {
            SignalIntent::ShortSpotLongPerp
        };

        // Create signal
        let signal = FundingArbSignal {
            schema_version: 1,
            ts: funding.ts,
            symbol: symbol.to_string(),
            intent,
            exit_policy,
            spot_bid_mantissa: spot_quote.bid_price_mantissa,
            spot_ask_mantissa: spot_quote.ask_price_mantissa,
            perp_bid_mantissa: perp_quote.bid_price_mantissa,
            perp_ask_mantissa: perp_quote.ask_price_mantissa,
            price_exponent: spot_quote.price_exponent,
            funding_rate_mantissa: funding.funding_rate_mantissa,
            rate_exponent: funding.rate_exponent,
            next_funding_time_ms: funding.next_funding_time_ms,
            basis_bps,
            funding_rate_bps,
            spot_spread_bps,
            perp_spread_bps,
            exit_time_stop_secs: if exit_policy == ExitPolicy::TimeStop {
                Some(args.time_stop_secs)
            } else {
                None
            },
            exit_basis_revert_bps: if exit_policy == ExitPolicy::BasisRevert {
                Some(args.basis_revert_bps)
            } else {
                None
            },
            reason_codes,
        };

        signals.push(signal);
        counters.signals_generated += 1;
        last_entry_ts = Some(funding.ts);
    }

    Ok((signals, counters))
}

// =============================================================================
// Data Loading
// =============================================================================

fn load_funding_events(path: &PathBuf) -> Result<Vec<FundingEvent>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open funding file: {:?}", path))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: FundingEvent = serde_json::from_str(&line)
            .with_context(|| format!("parse funding event: {}", line))?;
        events.push(event);
    }

    events.sort_by(|a, b| a.ts.cmp(&b.ts));
    Ok(events)
}

fn load_quote_events(path: &PathBuf) -> Result<Vec<QuoteEvent>> {
    let file = std::fs::File::open(path).with_context(|| format!("open quote file: {:?}", path))?;
    let reader = BufReader::new(file);

    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: QuoteEvent =
            serde_json::from_str(&line).with_context(|| format!("parse quote event: {}", line))?;
        events.push(event);
    }

    events.sort_by(|a, b| a.ts.cmp(&b.ts));
    Ok(events)
}

/// Build a time-indexed lookup structure for quotes.
fn build_quote_index(quotes: &[QuoteEvent]) -> Vec<&QuoteEvent> {
    // For now, just return a sorted slice reference
    // Could optimize with binary search or interval tree
    quotes.iter().collect()
}

/// Find quote at or before timestamp, within max_age_ms window.
fn find_quote_at_or_before<'a>(
    quotes: &[&'a QuoteEvent],
    ts: DateTime<Utc>,
    max_age_ms: u64,
) -> Option<&'a QuoteEvent> {
    // Binary search for the latest quote at or before ts
    let mut best: Option<&QuoteEvent> = None;

    for quote in quotes.iter().rev() {
        if quote.ts <= ts {
            let age_ms = (ts - quote.ts).num_milliseconds().unsigned_abs();
            if age_ms <= max_age_ms {
                best = Some(*quote);
                break;
            }
        }
    }

    // If no quote before, check if there's one slightly after
    if best.is_none() {
        for quote in quotes.iter() {
            if quote.ts >= ts {
                let age_ms = (quote.ts - ts).num_milliseconds().unsigned_abs();
                if age_ms <= max_age_ms {
                    best = Some(*quote);
                    break;
                }
            }
        }
    }

    best
}
