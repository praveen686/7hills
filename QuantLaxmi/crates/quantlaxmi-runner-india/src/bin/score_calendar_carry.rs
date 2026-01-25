//! score_calendar_carry
//!
//! Deterministic evaluator for `run_calendar_carry` signals.
//!
//! Phase Alpha-0.4 (scoring):
//! - Inputs: `--session-dir` and either `--signal-run-id` or `--signal-run-dir`
//! - Reads `signals.jsonl` (calendar carry ENTER intents)
//! - Loads option quote ticks from captured session
//! - Produces:
//!   - trades.jsonl
//!   - equity_curve.jsonl
//!   - metrics.json
//! - Writes a `run_manifest.json` binding all inputs and outputs.
//!
//! Determinism:
//! - No directory scanning beyond enumerating symbol tick files in legacy mode, sorted lexicographically.
//! - Stable sorting of signals and trades.
//! - Byte-stable JSON (pretty JSON for manifest/metrics).

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use quantlaxmi_runner_common::manifest_io::write_atomic;
use quantlaxmi_runner_common::run_manifest::{
    InputBinding, OutputBinding, RunManifest, config_hash, git_commit_string, hash_file,
    persist_run_manifest_atomic,
};

#[derive(Parser, Debug, Clone)]
#[command(name = "score_calendar_carry")]
struct Args {
    /// Session directory containing captured symbol subdirs.
    #[arg(long)]
    session_dir: PathBuf,

    /// Signal run directory for `run_calendar_carry` (contains signals.jsonl).
    /// If not provided, use --signal-run-id.
    #[arg(long)]
    signal_run_dir: Option<PathBuf>,

    /// Run ID under session_dir/runs/run_calendar_carry/<run_id>.
    #[arg(long)]
    signal_run_id: Option<String>,

    /// Entry/exit latency applied when sampling quotes.
    #[arg(long, default_value_t = 50)]
    latency_ms: u64,

    /// Shift entry timestamp by this many milliseconds (shift test).
    /// Positive shifts entry later; negative shifts entry earlier.
    #[arg(long, default_value_t = 0)]
    entry_shift_ms: i64,

    /// Shift exit timestamp by this many milliseconds (shift test).
    /// Positive shifts exit later; negative shifts exit earlier.
    #[arg(long, default_value_t = 0)]
    exit_shift_ms: i64,

    /// Round-trip friction in basis points, applied on notional at entry and exit.
    #[arg(long, default_value_t = 1.0)]
    friction_bps: f64,

    /// Holding period in seconds.
    #[arg(long, default_value_t = 600)]
    holding_secs: u64,

    /// Maximum quote age in milliseconds. If no quote within this window, trade is dropped.
    /// Set to 0 to disable (use first available quote). Default: 500ms.
    #[arg(long, default_value_t = 500)]
    max_quote_age_ms: u64,

    /// Use mid-price for fills (debug mode). Default: false (use bid/ask).
    #[arg(long, default_value_t = false)]
    use_mid: bool,

    /// Allow L1-only (synthetic) quotes. Default: false (L1Only quotes are REJECTED).
    /// By default, quotes with integrity_tier=L1Only are rejected to prevent
    /// illusory fills from manufactured liquidity. Set this flag to allow them.
    #[arg(long, default_value_t = false)]
    allow_l1only: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ScoreConfigHash {
    signal_run_id: String,
    latency_ms: u64,
    entry_shift_ms: i64,
    exit_shift_ms: i64,
    friction_bps: f64,
    holding_secs: u64,
    max_quote_age_ms: u64,
    use_mid: bool,
    /// If true, L1Only quotes are allowed (less strict). If false (default), L1Only rejected.
    allow_l1only: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
struct SignalEvent {
    schema_version: u32,
    ts: DateTime<Utc>,
    underlying: String,
    front_expiry: String,
    back_expiry: String,
    front_strike: f64,
    back_strike: f64,
    front_lots: i32,
    back_lots: i32,
    hedge_ratio: f64,
    cal_value: f64,
    cal_min: f64,
    friction_estimate: f64,
    reason_codes: Vec<String>,
}

/// Default integrity tier for backwards compatibility with old captures.
/// Old captures without integrity_tier field are assumed L2Present.
fn default_integrity_tier() -> String {
    "L2Present".to_string()
}

#[derive(Debug, Clone, Deserialize)]
struct TickEvent {
    ts: DateTime<Utc>,
    bid_price: i64,
    ask_price: i64,
    bid_qty: u32,
    ask_qty: u32,
    price_exponent: i32,
    /// Integrity tier: "L2Present" (real depth) or "L1Only" (synthetic spread).
    /// Synthetic quotes are rejected when --require-l2 is enabled (default).
    #[serde(default = "default_integrity_tier")]
    integrity_tier: String,
}

#[derive(Debug, Clone, Serialize)]
struct TradeRecord {
    schema_version: u32,
    entry_ts: DateTime<Utc>,
    exit_ts: DateTime<Utc>,
    underlying: String,
    front_expiry: String,
    back_expiry: String,
    front_strike: f64,
    back_strike: f64,
    entry_front_straddle: f64,
    entry_back_straddle: f64,
    exit_front_straddle: f64,
    exit_back_straddle: f64,
    pnl_gross: f64,
    pnl_net: f64,
    friction_bps: f64,
    latency_ms: u64,
    holding_secs: u64,
}

#[derive(Debug, Clone, Serialize)]
struct EquityPoint {
    ts: DateTime<Utc>,
    trade_index: usize,
    equity: f64,
}

/// Detailed drop reason counters for audit (Phase Alpha-1.3)
#[derive(Debug, Clone, Default, Serialize)]
struct DropReasons {
    /// Symbol not found in tick index
    symbol_not_found: usize,
    /// Entry quote missing (no ticks after entry_ts)
    missing_entry_quote: usize,
    /// Exit quote missing (no ticks after exit_ts)
    missing_exit_quote: usize,
    /// Entry quote found but too stale (age > max_quote_age_ms)
    stale_entry_quote: usize,
    /// Exit quote found but too stale
    stale_exit_quote: usize,
    /// Bad quote (zero bid/ask/qty)
    bad_quote: usize,
    /// L1-only quote rejected (synthetic spread, no real depth)
    l1only_rejected: usize,
    /// Maximum stale quote age observed (ms) - diagnostic for tuning max_quote_age_ms
    max_stale_age_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
struct Metrics {
    schema_version: u32,
    num_signals: usize,
    num_trades: usize,
    num_dropped: usize,
    gross_pnl: f64,
    net_pnl: f64,
    hit_rate: f64,
    max_drawdown: f64,
    avg_pnl: f64,
    // Quote staleness diagnostics
    avg_quote_age_ms: f64,
    max_quote_age_ms: i64,
    // Detailed drop reasons
    drop_reasons: DropReasons,
}

/// Audit-grade fill record (Phase Alpha-1.3: fills.jsonl)
#[derive(Debug, Clone, Serialize)]
struct FillRecord {
    schema_version: u32,
    signal_index: usize,
    entry_ts: DateTime<Utc>,
    exit_ts: DateTime<Utc>,
    underlying: String,
    front_expiry: String,
    back_expiry: String,
    front_strike: f64,
    back_strike: f64,
    // Entry leg quotes (front CE, front PE, back CE, back PE)
    entry_fce: LegFill,
    entry_fpe: LegFill,
    entry_bce: LegFill,
    entry_bpe: LegFill,
    // Exit leg quotes
    exit_fce: LegFill,
    exit_fpe: LegFill,
    exit_bce: LegFill,
    exit_bpe: LegFill,
    // Computed values
    front_straddle_entry: f64,
    back_straddle_entry: f64,
    front_straddle_exit: f64,
    back_straddle_exit: f64,
    pnl_front: f64,
    pnl_back: f64,
    pnl_gross: f64,
    friction_cost: f64,
    pnl_net: f64,
}

/// Per-leg fill details for audit
#[derive(Debug, Clone, Serialize)]
struct LegFill {
    symbol: String,
    bid: f64,
    ask: f64,
    mid: f64,
    spread: f64,
    quote_age_ms: i64,
    fill_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OptType {
    Call,
    Put,
}

#[derive(Debug, Clone)]
struct ParsedSymbol {
    underlying: String,
    expiry: NaiveDate,
    strike: f64,
    opt_type: OptType,
}

fn pow10_i32(exp: i32) -> f64 {
    10f64.powi(exp)
}

/// Quote result with bid, ask, and staleness info
#[derive(Debug, Clone, Copy)]
struct QuoteResult {
    bid: f64,
    ask: f64,
    mid: f64,
    quote_age_ms: i64,
}

/// Detailed quote lookup result for audit
#[derive(Debug, Clone, Copy)]
enum QuoteLookup {
    /// Valid quote found within staleness window
    Found(QuoteResult),
    /// No ticks at or after the target timestamp
    Missing,
    /// Found tick(s) but all had zero bid/ask/qty
    BadQuote,
    /// Found valid tick but it was beyond max_quote_age_ms window
    Stale { first_valid_age_ms: i64 },
    /// Quote rejected because it was L1Only (synthetic spread, no real depth)
    L1Only,
}

/// Find a valid quote at or after timestamp, respecting max_quote_age_ms constraint.
///
/// Corrected semantics (Phase Alpha-1.3):
/// - Scans all ticks at or after ts
/// - Skips bad quotes (zero bid/ask/qty) and continues scanning
/// - Returns first valid quote if its age <= max_quote_age_ms
/// - If first valid quote is beyond the window, returns Stale with the age
/// - Only returns Missing/BadQuote if no valid quotes found at all
///
/// P0 Research Integrity (Phase Alpha-1.4):
/// - When reject_l1only=true (default), rejects L1Only quotes (synthetic spread, no real depth)
/// - This prevents illusory fills from manufactured liquidity
fn find_quote_at_or_after(
    ticks: &[TickEvent],
    ts: DateTime<Utc>,
    max_quote_age_ms: u64,
    reject_l1only: bool,
) -> QuoteLookup {
    let mut found_any_tick = false;
    let mut found_bad_quote = false;
    let mut found_l1only = false;

    for t in ticks {
        if t.ts < ts {
            continue;
        }
        found_any_tick = true;

        // Check quote validity first
        if t.bid_price <= 0 || t.ask_price <= 0 || t.bid_qty == 0 || t.ask_qty == 0 {
            found_bad_quote = true;
            continue; // Keep scanning for a valid quote
        }

        // P0: Check integrity tier - reject L1Only (synthetic) quotes when reject_l1only is enabled
        if reject_l1only && t.integrity_tier == "L1Only" {
            found_l1only = true;
            continue; // Keep scanning for an L2Present quote
        }

        // Valid quote found - check if within staleness window
        let quote_age_ms = (t.ts - ts).num_milliseconds();

        if max_quote_age_ms > 0 && quote_age_ms > max_quote_age_ms as i64 {
            // First valid quote is beyond staleness window
            return QuoteLookup::Stale {
                first_valid_age_ms: quote_age_ms,
            };
        }

        // Valid quote within window - return it
        let scale = pow10_i32(t.price_exponent);
        let bid = t.bid_price as f64 * scale;
        let ask = t.ask_price as f64 * scale;
        let mid = (bid + ask) / 2.0;

        return QuoteLookup::Found(QuoteResult {
            bid,
            ask,
            mid,
            quote_age_ms,
        });
    }

    // Exhausted all ticks without finding a valid quote within window
    // Priority: L1Only > BadQuote > Missing (most specific rejection reason first)
    if !found_any_tick {
        QuoteLookup::Missing
    } else if found_l1only {
        QuoteLookup::L1Only
    } else if found_bad_quote {
        QuoteLookup::BadQuote
    } else {
        QuoteLookup::Missing
    }
}

fn parse_expiry_iso(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").with_context(|| format!("Invalid expiry date: {s}"))
}

fn month_str_to_num(m: &str) -> Option<u32> {
    match m {
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
    }
}

fn parse_option_symbol(symbol: &str) -> Option<ParsedSymbol> {
    // NSE option symbols: <UNDERLYING><DD><MON><STRIKE><CE|PE>
    // Example: NIFTY26JAN25100CE (no year suffix in 2026 format)
    // Older format with YY suffix: NIFTY26JAN2625100CE
    let s = symbol.trim().to_uppercase();
    let opt_type = if s.ends_with("CE") {
        OptType::Call
    } else if s.ends_with("PE") {
        OptType::Put
    } else {
        return None;
    };
    let core = &s[..s.len() - 2];

    // Find last digit run as strike.
    let mut idx = core.len();
    while idx > 0 && core.as_bytes()[idx - 1].is_ascii_digit() {
        idx -= 1;
    }
    if idx == core.len() {
        return None;
    }
    let strike_str = &core[idx..];
    let strike: f64 = strike_str.parse().ok()?;
    let prefix = &core[..idx];

    // Try 5-char format first (DDMON, no year) - common for 2026 NSE symbols
    if prefix.len() >= 5 {
        let date_part = &prefix[prefix.len() - 5..];
        let under = &prefix[..prefix.len() - 5];

        if let (Ok(dd), Some(mm)) = (
            date_part[0..2].parse::<u32>(),
            month_str_to_num(&date_part[2..5]),
        ) {
            // Infer year as 2026 (current trading year)
            if let Some(expiry) = NaiveDate::from_ymd_opt(2026, mm, dd) {
                return Some(ParsedSymbol {
                    underlying: under.to_string(),
                    expiry,
                    strike,
                    opt_type,
                });
            }
        }
    }

    // Fallback: try 7-char format (DDMONYY)
    if prefix.len() >= 7 {
        let date_part = &prefix[prefix.len() - 7..];
        let under = &prefix[..prefix.len() - 7];

        let dd: u32 = date_part[0..2].parse().ok()?;
        let mon = &date_part[2..5];
        let yy: i32 = date_part[5..7].parse().ok()?;
        let mm: u32 = month_str_to_num(mon)?;
        let yyyy = 2000 + yy;
        let expiry = NaiveDate::from_ymd_opt(yyyy, mm, dd)?;

        return Some(ParsedSymbol {
            underlying: under.to_string(),
            expiry,
            strike,
            opt_type,
        });
    }

    None
}

fn discover_symbol_tick_paths(session_dir: &Path) -> Result<HashMap<String, PathBuf>> {
    let mut entries: Vec<String> = Vec::new();
    for e in fs::read_dir(session_dir)
        .with_context(|| format!("Failed to read session dir: {}", session_dir.display()))?
    {
        let e = e?;
        if e.file_type()?.is_dir() {
            let name = e.file_name().to_string_lossy().to_string();
            // Ignore the run directories themselves
            if name == "runs" {
                continue;
            }
            let tick_path = session_dir.join(&name).join("ticks.jsonl");
            if tick_path.exists() {
                entries.push(name);
            }
        }
    }
    entries.sort();
    let mut map = HashMap::new();
    for sym in entries {
        map.insert(
            sym.to_uppercase(),
            session_dir.join(&sym).join("ticks.jsonl"),
        );
    }
    Ok(map)
}

fn load_ticks(path: &Path) -> Result<Vec<TickEvent>> {
    let f =
        File::open(path).with_context(|| format!("Failed to open ticks: {}", path.display()))?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let t: TickEvent = serde_json::from_str(&line)
            .with_context(|| format!("Bad tick JSON in {}", path.display()))?;
        out.push(t);
    }
    Ok(out)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let args = Args::parse();

    let signal_run_dir = match (&args.signal_run_dir, &args.signal_run_id) {
        (Some(p), _) => p.clone(),
        (None, Some(id)) => args
            .session_dir
            .join("runs")
            .join("run_calendar_carry")
            .join(id),
        (None, None) => {
            bail!("Provide --signal-run-dir or --signal-run-id");
        }
    };

    let signal_run_id = signal_run_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let signals_path = signal_run_dir.join("signals.jsonl");
    if !signals_path.exists() {
        bail!("signals.jsonl not found: {}", signals_path.display());
    }

    // Load signals
    let f = File::open(&signals_path).context("Failed to open signals.jsonl")?;
    let r = BufReader::new(f);
    let mut signals: Vec<SignalEvent> = Vec::new();
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let s: SignalEvent = serde_json::from_str(&line).context("Bad signal JSON")?;
        signals.push(s);
    }

    // Deterministic sort
    signals.sort_by(|a, b| {
        a.ts.cmp(&b.ts)
            .then(a.front_expiry.cmp(&b.front_expiry))
            .then(a.back_expiry.cmp(&b.back_expiry))
            .then(
                a.front_strike
                    .partial_cmp(&b.front_strike)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                a.back_strike
                    .partial_cmp(&b.back_strike)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Discover tick files
    let sym_to_path = discover_symbol_tick_paths(&args.session_dir)?;

    // Build parsed symbol index
    let mut index: Vec<(String, ParsedSymbol)> = Vec::new();
    for sym in sym_to_path.keys() {
        if let Some(p) = parse_option_symbol(sym) {
            index.push((sym.clone(), p));
        }
    }

    let symbol_for = |under: &str, expiry: NaiveDate, strike: f64, ot: OptType| -> Option<String> {
        // Exact match on strike (integer in symbol); signals may be f64 but should be .0
        for (sym, p) in &index {
            if p.underlying == under.to_uppercase()
                && p.expiry == expiry
                && (p.strike - strike).abs() < 1e-9
                && p.opt_type == ot
            {
                return Some(sym.clone());
            }
        }
        None
    };

    // Deterministic scoring run id
    let cfg = ScoreConfigHash {
        signal_run_id: signal_run_id.clone(),
        latency_ms: args.latency_ms,
        entry_shift_ms: args.entry_shift_ms,
        exit_shift_ms: args.exit_shift_ms,
        friction_bps: args.friction_bps,
        holding_secs: args.holding_secs,
        max_quote_age_ms: args.max_quote_age_ms,
        use_mid: args.use_mid,
        allow_l1only: args.allow_l1only,
    };
    let cfg_sha = config_hash(&cfg)?;
    let run_id = cfg_sha.chars().take(16).collect::<String>();

    let run_dir = args
        .session_dir
        .join("runs")
        .join("score_calendar_carry")
        .join(&run_id);
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("Failed to create run dir: {}", run_dir.display()))?;

    let trades_path = run_dir.join("trades.jsonl");
    let equity_path = run_dir.join("equity_curve.jsonl");
    let fills_path = run_dir.join("fills.jsonl");
    let metrics_path = run_dir.join("metrics.json");

    let mut trades_w =
        BufWriter::new(File::create(&trades_path).context("Failed to create trades.jsonl")?);
    let mut equity_w =
        BufWriter::new(File::create(&equity_path).context("Failed to create equity_curve.jsonl")?);
    let mut fills_w =
        BufWriter::new(File::create(&fills_path).context("Failed to create fills.jsonl")?);

    let latency = chrono::Duration::milliseconds(args.latency_ms as i64);

    // Tick cache per symbol
    let mut tick_cache: HashMap<String, Vec<TickEvent>> = HashMap::new();

    let mut trades: Vec<TradeRecord> = Vec::new();
    let mut equity: Vec<EquityPoint> = Vec::new();
    let mut cum = 0.0;

    // Quote age tracking for diagnostics
    let mut quote_ages: Vec<i64> = Vec::new();
    let mut drop_reasons = DropReasons::default();

    // Helper to classify drop reason from QuoteLookup
    fn classify_drop(lookup: &QuoteLookup, is_entry: bool, reasons: &mut DropReasons) {
        match lookup {
            QuoteLookup::Found(_) => {} // Not a drop
            QuoteLookup::Missing => {
                if is_entry {
                    reasons.missing_entry_quote += 1;
                } else {
                    reasons.missing_exit_quote += 1;
                }
            }
            QuoteLookup::BadQuote => {
                reasons.bad_quote += 1;
            }
            QuoteLookup::Stale { first_valid_age_ms } => {
                if is_entry {
                    reasons.stale_entry_quote += 1;
                } else {
                    reasons.stale_exit_quote += 1;
                }
                // Track max stale age for diagnostic tuning
                if *first_valid_age_ms > reasons.max_stale_age_ms {
                    reasons.max_stale_age_ms = *first_valid_age_ms;
                }
            }
            QuoteLookup::L1Only => {
                // P0: Synthetic quote rejected (no real L2 depth)
                reasons.l1only_rejected += 1;
            }
        }
    }

    for (i, sig) in signals.iter().enumerate() {
        let under = sig.underlying.to_uppercase();
        let fexp = match parse_expiry_iso(&sig.front_expiry) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let bexp = match parse_expiry_iso(&sig.back_expiry) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let f_ce = match symbol_for(&under, fexp, sig.front_strike, OptType::Call) {
            Some(s) => s,
            None => {
                drop_reasons.symbol_not_found += 1;
                continue;
            }
        };
        let f_pe = match symbol_for(&under, fexp, sig.front_strike, OptType::Put) {
            Some(s) => s,
            None => {
                drop_reasons.symbol_not_found += 1;
                continue;
            }
        };
        let b_ce = match symbol_for(&under, bexp, sig.back_strike, OptType::Call) {
            Some(s) => s,
            None => {
                drop_reasons.symbol_not_found += 1;
                continue;
            }
        };
        let b_pe = match symbol_for(&under, bexp, sig.back_strike, OptType::Put) {
            Some(s) => s,
            None => {
                drop_reasons.symbol_not_found += 1;
                continue;
            }
        };

        let entry_ts = sig.ts + latency + chrono::Duration::milliseconds(args.entry_shift_ms);
        let exit_ts = sig.ts
            + chrono::Duration::seconds(args.holding_secs as i64)
            + latency
            + chrono::Duration::milliseconds(args.exit_shift_ms);

        // Load ticks into cache if not present
        for sym in [&f_ce, &f_pe, &b_ce, &b_pe] {
            if !tick_cache.contains_key(sym) {
                let p = sym_to_path
                    .get(sym)
                    .with_context(|| format!("Missing tick path for {sym}"))?;
                let ticks = load_ticks(p)?;
                tick_cache.insert(sym.to_string(), ticks);
            }
        }

        let fce = tick_cache.get(&f_ce).unwrap().as_slice();
        let fpe = tick_cache.get(&f_pe).unwrap().as_slice();
        let bce = tick_cache.get(&b_ce).unwrap().as_slice();
        let bpe = tick_cache.get(&b_pe).unwrap().as_slice();

        // Get quotes with staleness constraint (returns QuoteLookup)
        // P0: reject_l1only=!allow_l1only enforces rejection of L1Only (synthetic) quotes
        let reject_l1only = !args.allow_l1only;
        let ql_fce_entry =
            find_quote_at_or_after(fce, entry_ts, args.max_quote_age_ms, reject_l1only);
        let ql_fpe_entry =
            find_quote_at_or_after(fpe, entry_ts, args.max_quote_age_ms, reject_l1only);
        let ql_bce_entry =
            find_quote_at_or_after(bce, entry_ts, args.max_quote_age_ms, reject_l1only);
        let ql_bpe_entry =
            find_quote_at_or_after(bpe, entry_ts, args.max_quote_age_ms, reject_l1only);

        let ql_fce_exit =
            find_quote_at_or_after(fce, exit_ts, args.max_quote_age_ms, reject_l1only);
        let ql_fpe_exit =
            find_quote_at_or_after(fpe, exit_ts, args.max_quote_age_ms, reject_l1only);
        let ql_bce_exit =
            find_quote_at_or_after(bce, exit_ts, args.max_quote_age_ms, reject_l1only);
        let ql_bpe_exit =
            find_quote_at_or_after(bpe, exit_ts, args.max_quote_age_ms, reject_l1only);

        // Extract QuoteResult or classify drop reason
        let q_fce_e = match ql_fce_entry {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, true, &mut drop_reasons);
                continue;
            }
        };
        let q_fpe_e = match ql_fpe_entry {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, true, &mut drop_reasons);
                continue;
            }
        };
        let q_bce_e = match ql_bce_entry {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, true, &mut drop_reasons);
                continue;
            }
        };
        let q_bpe_e = match ql_bpe_entry {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, true, &mut drop_reasons);
                continue;
            }
        };

        let q_fce_x = match ql_fce_exit {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, false, &mut drop_reasons);
                continue;
            }
        };
        let q_fpe_x = match ql_fpe_exit {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, false, &mut drop_reasons);
                continue;
            }
        };
        let q_bce_x = match ql_bce_exit {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, false, &mut drop_reasons);
                continue;
            }
        };
        let q_bpe_x = match ql_bpe_exit {
            QuoteLookup::Found(q) => q,
            ref other => {
                classify_drop(other, false, &mut drop_reasons);
                continue;
            }
        };

        // Track quote ages for diagnostics (all 8 legs per trade)
        quote_ages.push(q_fce_e.quote_age_ms);
        quote_ages.push(q_fpe_e.quote_age_ms);
        quote_ages.push(q_bce_e.quote_age_ms);
        quote_ages.push(q_bpe_e.quote_age_ms);
        quote_ages.push(q_fce_x.quote_age_ms);
        quote_ages.push(q_fpe_x.quote_age_ms);
        quote_ages.push(q_bce_x.quote_age_ms);
        quote_ages.push(q_bpe_x.quote_age_ms);

        // Compute fill prices based on mode
        let make_leg_fill = |sym: &str, q: QuoteResult, is_buy: bool| -> LegFill {
            let fill_price = if args.use_mid {
                q.mid
            } else if is_buy {
                q.ask
            } else {
                q.bid
            };
            LegFill {
                symbol: sym.to_string(),
                bid: q.bid,
                ask: q.ask,
                mid: q.mid,
                spread: q.ask - q.bid,
                quote_age_ms: q.quote_age_ms,
                fill_price,
            }
        };

        // Entry: short front (sell), long back (buy)
        // Exit: cover front (buy), close back (sell)
        let entry_fce_fill = make_leg_fill(&f_ce, q_fce_e, false); // sell
        let entry_fpe_fill = make_leg_fill(&f_pe, q_fpe_e, false); // sell
        let entry_bce_fill = make_leg_fill(&b_ce, q_bce_e, true); // buy
        let entry_bpe_fill = make_leg_fill(&b_pe, q_bpe_e, true); // buy
        let exit_fce_fill = make_leg_fill(&f_ce, q_fce_x, true); // buy (cover)
        let exit_fpe_fill = make_leg_fill(&f_pe, q_fpe_x, true); // buy (cover)
        let exit_bce_fill = make_leg_fill(&b_ce, q_bce_x, false); // sell (close)
        let exit_bpe_fill = make_leg_fill(&b_pe, q_bpe_x, false); // sell (close)

        let entry_front = entry_fce_fill.fill_price + entry_fpe_fill.fill_price;
        let entry_back = entry_bce_fill.fill_price + entry_bpe_fill.fill_price;
        let exit_front = exit_fce_fill.fill_price + exit_fpe_fill.fill_price;
        let exit_back = exit_bce_fill.fill_price + exit_bpe_fill.fill_price;

        // PnL: short front, long back
        let fl = sig.front_lots as f64;
        let bl = sig.back_lots as f64;
        let pnl_front = -(exit_front - entry_front) * fl;
        let pnl_back = (exit_back - entry_back) * bl;
        let pnl_gross = pnl_front + pnl_back;

        // Friction costs (bps) on notional at entry + exit
        let bps = args.friction_bps / 10000.0;
        let notional_entry = entry_front.abs() * fl.abs() + entry_back.abs() * bl.abs();
        let notional_exit = exit_front.abs() * fl.abs() + exit_back.abs() * bl.abs();
        let friction_cost = (notional_entry + notional_exit) * bps;
        let pnl_net = pnl_gross - friction_cost;

        // Write audit-grade fill record
        let fill_record = FillRecord {
            schema_version: 1,
            signal_index: i,
            entry_ts,
            exit_ts,
            underlying: under.clone(),
            front_expiry: sig.front_expiry.clone(),
            back_expiry: sig.back_expiry.clone(),
            front_strike: sig.front_strike,
            back_strike: sig.back_strike,
            entry_fce: entry_fce_fill,
            entry_fpe: entry_fpe_fill,
            entry_bce: entry_bce_fill,
            entry_bpe: entry_bpe_fill,
            exit_fce: exit_fce_fill,
            exit_fpe: exit_fpe_fill,
            exit_bce: exit_bce_fill,
            exit_bpe: exit_bpe_fill,
            front_straddle_entry: entry_front,
            back_straddle_entry: entry_back,
            front_straddle_exit: exit_front,
            back_straddle_exit: exit_back,
            pnl_front,
            pnl_back,
            pnl_gross,
            friction_cost,
            pnl_net,
        };
        let fill_json =
            serde_json::to_string(&fill_record).context("Failed to serialize FillRecord")?;
        writeln!(fills_w, "{}", fill_json).context("Failed to write fill")?;

        let tr = TradeRecord {
            schema_version: 1,
            entry_ts: sig.ts,
            exit_ts: sig.ts + chrono::Duration::seconds(args.holding_secs as i64),
            underlying: under.clone(),
            front_expiry: sig.front_expiry.clone(),
            back_expiry: sig.back_expiry.clone(),
            front_strike: sig.front_strike,
            back_strike: sig.back_strike,
            entry_front_straddle: entry_front,
            entry_back_straddle: entry_back,
            exit_front_straddle: exit_front,
            exit_back_straddle: exit_back,
            pnl_gross,
            pnl_net,
            friction_bps: args.friction_bps,
            latency_ms: args.latency_ms,
            holding_secs: args.holding_secs,
        };

        let json = serde_json::to_string(&tr).context("Failed to serialize TradeRecord")?;
        writeln!(trades_w, "{}", json).context("Failed to write trade")?;

        cum += pnl_net;
        let ep = EquityPoint {
            ts: tr.exit_ts,
            trade_index: trades.len(),
            equity: cum,
        };
        let ej = serde_json::to_string(&ep).context("Failed to serialize equity")?;
        writeln!(equity_w, "{}", ej).context("Failed to write equity")?;

        trades.push(tr);
        equity.push(ep);

        // deterministic progress: no logging per trade
        let _ = i;
    }

    trades_w.flush().context("Failed to flush trades")?;
    equity_w.flush().context("Failed to flush equity")?;
    fills_w.flush().context("Failed to flush fills")?;

    // Metrics
    let num_signals = signals.len();
    let num_trades = trades.len();
    let num_dropped = drop_reasons.symbol_not_found
        + drop_reasons.missing_entry_quote
        + drop_reasons.missing_exit_quote
        + drop_reasons.stale_entry_quote
        + drop_reasons.stale_exit_quote
        + drop_reasons.bad_quote;

    let gross_pnl: f64 = trades.iter().map(|t| t.pnl_gross).sum();
    let net_pnl: f64 = trades.iter().map(|t| t.pnl_net).sum();
    let wins = trades.iter().filter(|t| t.pnl_net > 0.0).count();
    let hit_rate = if num_trades > 0 {
        wins as f64 / num_trades as f64
    } else {
        0.0
    };

    // Max drawdown
    let mut peak = 0.0;
    let mut max_dd = 0.0;
    for e in &equity {
        if e.equity > peak {
            peak = e.equity;
        }
        let dd = peak - e.equity;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    let avg_pnl = if num_trades > 0 {
        net_pnl / num_trades as f64
    } else {
        0.0
    };

    // Quote age diagnostics
    let avg_quote_age_ms = if !quote_ages.is_empty() {
        quote_ages.iter().sum::<i64>() as f64 / quote_ages.len() as f64
    } else {
        0.0
    };
    let max_quote_age_ms = quote_ages.iter().copied().max().unwrap_or(0);

    let metrics = Metrics {
        schema_version: 1,
        num_signals,
        num_trades,
        num_dropped,
        gross_pnl,
        net_pnl,
        hit_rate,
        max_drawdown: max_dd,
        avg_pnl,
        avg_quote_age_ms,
        max_quote_age_ms,
        drop_reasons,
    };
    let metrics_bytes =
        serde_json::to_vec_pretty(&metrics).context("Failed to serialize metrics")?;
    write_atomic(&metrics_path, &metrics_bytes)
        .with_context(|| format!("Failed to write metrics: {}", metrics_path.display()))?;

    // RunManifest
    let git_commit = git_commit_string();

    // session manifest binding (best-effort)
    let sess_manifest_path = args.session_dir.join("session_manifest.json");
    let session_sha256 = if sess_manifest_path.exists() {
        let (sha, _) = hash_file(&sess_manifest_path)?;
        sha
    } else {
        "missing".to_string()
    };

    let session_binding = InputBinding {
        label: "session_manifest".to_string(),
        rel_path: "session_manifest.json".to_string(),
        sha256: session_sha256,
    };

    // input run manifest binding
    let input_rm_path = signal_run_dir.join("run_manifest.json");
    let (input_rm_sha, _) = if input_rm_path.exists() {
        hash_file(&input_rm_path)?
    } else {
        ("missing".to_string(), 0)
    };
    let input_run_binding = InputBinding {
        label: "input_run_manifest".to_string(),
        rel_path: signal_run_dir
            .join("run_manifest.json")
            .to_string_lossy()
            .to_string(),
        sha256: input_rm_sha,
    };

    let (signals_sha, _) = hash_file(&signals_path)?;
    let signals_binding = InputBinding {
        label: "input_signals".to_string(),
        rel_path: signal_run_dir
            .join("signals.jsonl")
            .to_string_lossy()
            .to_string(),
        sha256: signals_sha,
    };

    let inputs = vec![input_run_binding, signals_binding];

    // Output hashes
    let (tr_sha, tr_len) = hash_file(&trades_path)?;
    let (eq_sha, eq_len) = hash_file(&equity_path)?;
    let (fi_sha, fi_len) = hash_file(&fills_path)?;
    let (m_sha, m_len) = hash_file(&metrics_path)?;

    let outputs = vec![
        OutputBinding {
            label: "trades_jsonl".to_string(),
            rel_path: "trades.jsonl".to_string(),
            sha256: tr_sha,
            bytes_len: tr_len,
        },
        OutputBinding {
            label: "equity_curve_jsonl".to_string(),
            rel_path: "equity_curve.jsonl".to_string(),
            sha256: eq_sha,
            bytes_len: eq_len,
        },
        OutputBinding {
            label: "fills_jsonl".to_string(),
            rel_path: "fills.jsonl".to_string(),
            sha256: fi_sha,
            bytes_len: fi_len,
        },
        OutputBinding {
            label: "metrics_json".to_string(),
            rel_path: "metrics.json".to_string(),
            sha256: m_sha,
            bytes_len: m_len,
        },
    ];

    let rm = RunManifest {
        schema_version: 1,
        binary_name: "score_calendar_carry".to_string(),
        git_commit,
        run_id: run_id.clone(),
        session_dir: args.session_dir.to_string_lossy().to_string(),
        session_manifest: session_binding,
        universe_manifests: inputs,
        config_sha256: cfg_sha,
        outputs,
        wal_files: Vec::new(),
    };

    let rm_sha = persist_run_manifest_atomic(&run_dir, &rm)?;

    // also include the manifest sha in metrics (optional) - keep metrics stable
    let _ = rm_sha;

    Ok(())
}

// =============================================================================
// P2 Audit: CI Tests for L1Only Rejection (Research Integrity)
// =============================================================================
// These tests ensure that synthetic quotes (L1Only) are rejected by default
// in the scoring pipeline, preventing illusory fills from manufactured liquidity.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn make_tick(ts: DateTime<Utc>, bid: i64, ask: i64, qty: u32, tier: &str) -> TickEvent {
        TickEvent {
            ts,
            bid_price: bid,
            ask_price: ask,
            bid_qty: qty,
            ask_qty: qty,
            price_exponent: -2,
            integrity_tier: tier.to_string(),
        }
    }

    /// P2 CI Test: L1Only quotes are REJECTED when reject_l1only=true (default behavior)
    #[test]
    fn test_l1only_rejected_by_default() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks = vec![make_tick(now, 10000, 10100, 100, "L1Only")];

        let result = find_quote_at_or_after(&ticks, now, 1000, true); // reject_l1only=true
        assert!(
            matches!(result, QuoteLookup::L1Only),
            "L1Only tick should be rejected when reject_l1only=true, got {:?}",
            result
        );
    }

    /// P2 CI Test: L1Only quotes are ACCEPTED when reject_l1only=false (debug mode)
    #[test]
    fn test_l1only_accepted_when_flag_disabled() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks = vec![make_tick(now, 10000, 10100, 100, "L1Only")];

        let result = find_quote_at_or_after(&ticks, now, 1000, false); // reject_l1only=false
        assert!(
            matches!(result, QuoteLookup::Found(_)),
            "L1Only tick should be accepted when reject_l1only=false, got {:?}",
            result
        );
    }

    /// P2 CI Test: L2Present quotes are found even when L1Only comes first
    #[test]
    fn test_l2present_found_after_l1only() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks = vec![
            make_tick(now, 10000, 10100, 100, "L1Only"),
            make_tick(
                now + Duration::milliseconds(50),
                10000,
                10100,
                100,
                "L2Present",
            ),
        ];

        let result = find_quote_at_or_after(&ticks, now, 1000, true); // reject_l1only=true
        assert!(
            matches!(result, QuoteLookup::Found(_)),
            "Should find L2Present quote after skipping L1Only, got {:?}",
            result
        );

        if let QuoteLookup::Found(qr) = result {
            assert_eq!(qr.quote_age_ms, 50, "Quote age should be 50ms");
        }
    }

    /// P2 CI Test: L1Only rejection takes priority over BadQuote
    #[test]
    fn test_l1only_priority_over_badquote() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks = vec![
            make_tick(now, 0, 0, 0, "L2Present"), // BadQuote (zero prices)
            make_tick(
                now + Duration::milliseconds(10),
                10000,
                10100,
                100,
                "L1Only",
            ),
        ];

        let result = find_quote_at_or_after(&ticks, now, 1000, true);
        assert!(
            matches!(result, QuoteLookup::L1Only),
            "L1Only should take priority over BadQuote, got {:?}",
            result
        );
    }

    /// P2 CI Test: Stale L2Present is reported as Stale, not L1Only
    #[test]
    fn test_stale_l2present_not_masked_by_l1only() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks = vec![
            make_tick(now, 10000, 10100, 100, "L1Only"),
            // L2Present beyond staleness window (1500ms > 1000ms)
            make_tick(
                now + Duration::milliseconds(1500),
                10000,
                10100,
                100,
                "L2Present",
            ),
        ];

        let result = find_quote_at_or_after(&ticks, now, 1000, true); // max_age=1000ms
        assert!(
            matches!(result, QuoteLookup::Stale { .. }),
            "Should report Stale for L2Present beyond window, got {:?}",
            result
        );
    }

    /// P2 CI Test: Default integrity tier is L2Present for backward compatibility
    #[test]
    fn test_default_integrity_tier_is_l2present() {
        let tier = default_integrity_tier();
        assert_eq!(
            tier, "L2Present",
            "Default integrity tier must be L2Present for backward compatibility"
        );
    }

    /// P2 CI Test: All-L1Only sequence returns L1Only rejection
    #[test]
    fn test_all_l1only_sequence_rejected() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks = vec![
            make_tick(now, 10000, 10100, 100, "L1Only"),
            make_tick(
                now + Duration::milliseconds(100),
                10000,
                10100,
                100,
                "L1Only",
            ),
            make_tick(
                now + Duration::milliseconds(200),
                10000,
                10100,
                100,
                "L1Only",
            ),
        ];

        let result = find_quote_at_or_after(&ticks, now, 1000, true);
        assert!(
            matches!(result, QuoteLookup::L1Only),
            "All-L1Only sequence should be rejected, got {:?}",
            result
        );
    }

    /// P2 CI Test: Empty ticks returns Missing, not L1Only
    #[test]
    fn test_empty_ticks_returns_missing() {
        let now = Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap();
        let ticks: Vec<TickEvent> = vec![];

        let result = find_quote_at_or_after(&ticks, now, 1000, true);
        assert!(
            matches!(result, QuoteLookup::Missing),
            "Empty ticks should return Missing, got {:?}",
            result
        );
    }
}
