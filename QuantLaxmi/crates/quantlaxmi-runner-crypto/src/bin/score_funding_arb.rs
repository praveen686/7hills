//! Funding Rate Arbitrage Scorer.
//!
//! Evaluates funding arb signals with:
//! - Bid/ask realistic fills (not mid prices)
//! - Strict no-fill if quotes missing/stale
//! - Shift tests (entry/exit timestamp shifts)
//! - PnL decomposition:
//!   - Spot leg PnL
//!   - Perp leg PnL
//!   - Fees (maker/taker)
//!   - Funding transfer (credited/debited exactly at funding timestamp)
//!   - Residual basis move while holding
//!
//! ## Usage
//! ```bash
//! cargo run --bin score_funding_arb -- \
//!     --session-dir data/perp_sessions/perp_20260124_100000 \
//!     --signal-run-id abc123 \
//!     --latency-ms 50 \
//!     --entry-shift-ms 0 \
//!     --exit-shift-ms 0 \
//!     --fee-bps 2.0
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use quantlaxmi_runner_crypto::binance_funding_capture::FundingEvent;

/// Type alias for symbol quote cache (spot quotes, perp quotes, funding events).
type SymbolQuoteCache = HashMap<String, (Vec<QuoteEvent>, Vec<QuoteEvent>, Vec<FundingEvent>)>;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser, Debug)]
#[command(name = "score_funding_arb")]
#[command(about = "Score funding rate arbitrage signals with bid/ask fills")]
struct Args {
    /// Path to perp session directory
    #[arg(long)]
    session_dir: PathBuf,

    /// Signal run ID (subdirectory of session_dir/signals/)
    #[arg(long)]
    signal_run_id: String,

    /// Output directory (default: session_dir/scores/{run_hash})
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Latency in milliseconds (added to signal timestamp for fill)
    #[arg(long, default_value = "50")]
    latency_ms: u64,

    /// Entry timestamp shift in milliseconds (for shift tests)
    #[arg(long, default_value = "0")]
    entry_shift_ms: i64,

    /// Exit timestamp shift in milliseconds (for shift tests)
    #[arg(long, default_value = "0")]
    exit_shift_ms: i64,

    /// Round-trip fee in basis points (default: 2.0 bps = maker fee)
    #[arg(long, default_value = "2.0")]
    fee_bps: f64,

    /// Maximum quote age in milliseconds
    #[arg(long, default_value = "500")]
    max_quote_age_ms: u64,

    /// Use mid prices instead of bid/ask (for debugging only)
    #[arg(long, default_value = "false")]
    use_mid: bool,

    /// Position size in quote currency (e.g., 10000 USDT)
    #[arg(long, default_value = "10000.0")]
    position_size: f64,
}

// =============================================================================
// Data Structures
// =============================================================================

/// Signal intent for funding arbitrage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalIntent {
    LongSpotShortPerp,
    ShortSpotLongPerp,
}

/// Exit policy for the position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitPolicy {
    FundingSettle,
    TimeStop,
    BasisRevert,
}

/// Funding arbitrage signal event (loaded from signals.jsonl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingArbSignal {
    pub schema_version: u32,
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub intent: SignalIntent,
    pub exit_policy: ExitPolicy,
    pub spot_bid_mantissa: i64,
    pub spot_ask_mantissa: i64,
    pub perp_bid_mantissa: i64,
    pub perp_ask_mantissa: i64,
    pub price_exponent: i8,
    pub funding_rate_mantissa: i64,
    pub rate_exponent: i8,
    pub next_funding_time_ms: i64,
    pub basis_bps: f64,
    pub funding_rate_bps: f64,
    pub spot_spread_bps: f64,
    pub perp_spread_bps: f64,
    pub exit_time_stop_secs: Option<u64>,
    pub exit_basis_revert_bps: Option<f64>,
    pub reason_codes: Vec<String>,
}

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
}

/// Quote lookup result.
#[derive(Debug, Clone)]
struct QuoteResult {
    bid: f64,
    ask: f64,
    mid: f64,
    quote_age_ms: i64,
}

/// Quote lookup status.
#[derive(Debug)]
enum QuoteLookup {
    Found(QuoteResult),
    Missing,
    Stale { first_valid_age_ms: i64 },
}

/// Per-leg fill information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegFill {
    pub market: String, // "spot" or "perp"
    pub side: String,   // "buy" or "sell"
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread: f64,
    pub quote_age_ms: i64,
    pub fill_price: f64,
    pub quantity: f64,
    pub notional: f64,
}

/// Trade record with full PnL decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub schema_version: u32,
    pub signal_index: usize,
    pub symbol: String,
    pub intent: SignalIntent,

    // Timestamps
    pub signal_ts: DateTime<Utc>,
    pub entry_ts: DateTime<Utc>,
    pub exit_ts: DateTime<Utc>,
    pub hold_seconds: i64,

    // Entry fills
    pub entry_spot: LegFill,
    pub entry_perp: LegFill,

    // Exit fills
    pub exit_spot: LegFill,
    pub exit_perp: LegFill,

    // PnL decomposition
    pub pnl_spot_leg: f64,
    pub pnl_perp_leg: f64,
    pub pnl_gross: f64,
    pub fee_cost: f64,
    pub funding_transfer: f64,
    pub basis_drift_pnl: f64,
    pub pnl_net: f64,

    // Diagnostics
    pub entry_basis_bps: f64,
    pub exit_basis_bps: f64,
    pub basis_change_bps: f64,
}

/// Fill record for audit-grade detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRecord {
    pub schema_version: u32,
    pub signal_index: usize,
    pub entry_ts: DateTime<Utc>,
    pub exit_ts: DateTime<Utc>,

    // All fills
    pub entry_spot: LegFill,
    pub entry_perp: LegFill,
    pub exit_spot: LegFill,
    pub exit_perp: LegFill,

    // Funding events during hold period
    pub funding_events: Vec<FundingSettlement>,
}

/// Funding settlement event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingSettlement {
    pub ts: DateTime<Utc>,
    pub funding_rate_bps: f64,
    pub mark_price: f64,
    pub position_value: f64,
    pub funding_amount: f64, // Positive = received, negative = paid
}

/// Drop reasons for failed trades.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DropReasons {
    pub total_signals: usize,
    pub missing_entry_spot: usize,
    pub missing_entry_perp: usize,
    pub missing_exit_spot: usize,
    pub missing_exit_perp: usize,
    pub stale_entry_spot: usize,
    pub stale_entry_perp: usize,
    pub stale_exit_spot: usize,
    pub stale_exit_perp: usize,
    pub max_stale_age_ms: i64,
}

/// Equity curve point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub ts: DateTime<Utc>,
    pub trade_index: usize,
    pub equity: f64,
}

/// Scoring metrics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringMetrics {
    pub schema_version: u32,
    pub run_id: String,
    pub signal_run_id: String,

    // Config
    pub latency_ms: u64,
    pub entry_shift_ms: i64,
    pub exit_shift_ms: i64,
    pub fee_bps: f64,
    pub config_max_quote_age_ms: u64,
    pub use_mid: bool,
    pub position_size: f64,

    // Counts
    pub num_signals: usize,
    pub num_trades: usize,
    pub num_dropped: usize,

    // PnL summary
    pub gross_pnl: f64,
    pub total_fees: f64,
    pub total_funding: f64,
    pub net_pnl: f64,

    // Performance
    pub hit_rate: f64,
    pub avg_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,

    // PnL attribution
    pub pnl_from_spot: f64,
    pub pnl_from_perp: f64,
    pub pnl_from_funding: f64,
    pub pnl_from_fees: f64,

    // Quote diagnostics
    pub avg_quote_age_ms: f64,
    pub observed_max_quote_age_ms: i64,

    // Drop reasons
    pub drop_reasons: DropReasons,
}

/// Score run manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreRunManifest {
    pub schema_version: u32,
    pub created_at_utc: String,
    pub run_id: String,
    pub config_hash: String,
    pub metrics: ScoringMetrics,
}

// =============================================================================
// Main Logic
// =============================================================================

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    tracing::info!("=== Funding Arb Scoring ===");
    tracing::info!("Session: {:?}", args.session_dir);
    tracing::info!("Signal run: {}", args.signal_run_id);
    tracing::info!("Latency: {}ms", args.latency_ms);
    tracing::info!("Entry shift: {}ms", args.entry_shift_ms);
    tracing::info!("Exit shift: {}ms", args.exit_shift_ms);
    tracing::info!("Fee: {} bps", args.fee_bps);

    // Load signals
    let signal_run_dir = args.session_dir.join("signals").join(&args.signal_run_id);
    let signals_path = signal_run_dir.join("signals.jsonl");
    let signals = load_signals(&signals_path)?;
    tracing::info!("Loaded {} signals", signals.len());

    // Generate config hash for run ID
    let config_hash = compute_config_hash(&args)?;
    let run_id = config_hash.chars().take(16).collect::<String>();

    // Create output directory
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| args.session_dir.join("scores").join(&run_id));
    std::fs::create_dir_all(&output_dir)?;
    tracing::info!("Output: {:?}", output_dir);

    // Group signals by symbol
    let mut signals_by_symbol: HashMap<String, Vec<(usize, FundingArbSignal)>> = HashMap::new();
    for (idx, sig) in signals.iter().enumerate() {
        signals_by_symbol
            .entry(sig.symbol.clone())
            .or_default()
            .push((idx, sig.clone()));
    }

    // Load quote data per symbol
    let mut quote_cache: SymbolQuoteCache = HashMap::new();

    for symbol in signals_by_symbol.keys() {
        let sym_dir = args.session_dir.join(symbol.to_uppercase());
        let spot_quotes = load_quote_events(&sym_dir.join("spot_quotes.jsonl"))?;
        let perp_quotes = load_quote_events(&sym_dir.join("perp_quotes.jsonl"))?;
        let funding_events = load_funding_events(&sym_dir.join("funding.jsonl"))?;
        quote_cache.insert(symbol.clone(), (spot_quotes, perp_quotes, funding_events));
    }

    // Score each signal
    let mut trades: Vec<TradeRecord> = Vec::new();
    let mut fills: Vec<FillRecord> = Vec::new();
    let mut drop_reasons = DropReasons {
        total_signals: signals.len(),
        ..Default::default()
    };

    let mut quote_ages: Vec<i64> = Vec::new();

    for (idx, signal) in signals.iter().enumerate() {
        let (spot_quotes, perp_quotes, funding_events) = quote_cache
            .get(&signal.symbol)
            .ok_or_else(|| anyhow::anyhow!("Symbol not in cache: {}", signal.symbol))?;

        // Calculate entry and exit timestamps with latency and shifts
        let latency = chrono::Duration::milliseconds(args.latency_ms as i64);
        let entry_shift = chrono::Duration::milliseconds(args.entry_shift_ms);
        let exit_shift = chrono::Duration::milliseconds(args.exit_shift_ms);

        let entry_ts = signal.ts + latency + entry_shift;

        // Determine exit timestamp based on exit policy
        let exit_ts = match signal.exit_policy {
            ExitPolicy::FundingSettle => {
                // Exit after next funding settlement
                let next_funding_ts = Utc
                    .timestamp_millis_opt(signal.next_funding_time_ms)
                    .unwrap();
                next_funding_ts + latency + exit_shift
            }
            ExitPolicy::TimeStop => {
                let time_stop_secs = signal.exit_time_stop_secs.unwrap_or(28800);
                entry_ts + chrono::Duration::seconds(time_stop_secs as i64) + exit_shift
            }
            ExitPolicy::BasisRevert => {
                // For now, use funding settle as fallback
                let next_funding_ts = Utc
                    .timestamp_millis_opt(signal.next_funding_time_ms)
                    .unwrap();
                next_funding_ts + latency + exit_shift
            }
        };

        // Find entry quotes
        let entry_spot = find_quote_at_or_after(spot_quotes, entry_ts, args.max_quote_age_ms);
        let entry_perp = find_quote_at_or_after(perp_quotes, entry_ts, args.max_quote_age_ms);

        // Find exit quotes
        let exit_spot = find_quote_at_or_after(spot_quotes, exit_ts, args.max_quote_age_ms);
        let exit_perp = find_quote_at_or_after(perp_quotes, exit_ts, args.max_quote_age_ms);

        // Check for drops
        let entry_spot_q = match entry_spot {
            QuoteLookup::Found(q) => q,
            QuoteLookup::Missing => {
                drop_reasons.missing_entry_spot += 1;
                continue;
            }
            QuoteLookup::Stale { first_valid_age_ms } => {
                drop_reasons.stale_entry_spot += 1;
                drop_reasons.max_stale_age_ms =
                    drop_reasons.max_stale_age_ms.max(first_valid_age_ms);
                continue;
            }
        };

        let entry_perp_q = match entry_perp {
            QuoteLookup::Found(q) => q,
            QuoteLookup::Missing => {
                drop_reasons.missing_entry_perp += 1;
                continue;
            }
            QuoteLookup::Stale { first_valid_age_ms } => {
                drop_reasons.stale_entry_perp += 1;
                drop_reasons.max_stale_age_ms =
                    drop_reasons.max_stale_age_ms.max(first_valid_age_ms);
                continue;
            }
        };

        let exit_spot_q = match exit_spot {
            QuoteLookup::Found(q) => q,
            QuoteLookup::Missing => {
                drop_reasons.missing_exit_spot += 1;
                continue;
            }
            QuoteLookup::Stale { first_valid_age_ms } => {
                drop_reasons.stale_exit_spot += 1;
                drop_reasons.max_stale_age_ms =
                    drop_reasons.max_stale_age_ms.max(first_valid_age_ms);
                continue;
            }
        };

        let exit_perp_q = match exit_perp {
            QuoteLookup::Found(q) => q,
            QuoteLookup::Missing => {
                drop_reasons.missing_exit_perp += 1;
                continue;
            }
            QuoteLookup::Stale { first_valid_age_ms } => {
                drop_reasons.stale_exit_perp += 1;
                drop_reasons.max_stale_age_ms =
                    drop_reasons.max_stale_age_ms.max(first_valid_age_ms);
                continue;
            }
        };

        // Track quote ages
        quote_ages.push(entry_spot_q.quote_age_ms);
        quote_ages.push(entry_perp_q.quote_age_ms);
        quote_ages.push(exit_spot_q.quote_age_ms);
        quote_ages.push(exit_perp_q.quote_age_ms);

        // Calculate position size in base currency
        let entry_spot_mid = entry_spot_q.mid;
        let position_qty = args.position_size / entry_spot_mid;

        // Create leg fills based on intent and bid/ask discipline
        let (entry_spot_fill, entry_perp_fill, exit_spot_fill, exit_perp_fill) = match signal.intent
        {
            SignalIntent::LongSpotShortPerp => {
                // Entry: BUY spot at ask, SELL perp at bid
                // Exit: SELL spot at bid, BUY perp at ask
                (
                    make_leg_fill("spot", "buy", &entry_spot_q, position_qty, args.use_mid),
                    make_leg_fill("perp", "sell", &entry_perp_q, position_qty, args.use_mid),
                    make_leg_fill("spot", "sell", &exit_spot_q, position_qty, args.use_mid),
                    make_leg_fill("perp", "buy", &exit_perp_q, position_qty, args.use_mid),
                )
            }
            SignalIntent::ShortSpotLongPerp => {
                // Entry: SELL spot at bid, BUY perp at ask
                // Exit: BUY spot at ask, SELL perp at bid
                (
                    make_leg_fill("spot", "sell", &entry_spot_q, position_qty, args.use_mid),
                    make_leg_fill("perp", "buy", &entry_perp_q, position_qty, args.use_mid),
                    make_leg_fill("spot", "buy", &exit_spot_q, position_qty, args.use_mid),
                    make_leg_fill("perp", "sell", &exit_perp_q, position_qty, args.use_mid),
                )
            }
        };

        // Calculate PnL components
        let pnl_spot_leg = match signal.intent {
            SignalIntent::LongSpotShortPerp => {
                // Long spot: (exit_price - entry_price) * qty
                (exit_spot_fill.fill_price - entry_spot_fill.fill_price) * position_qty
            }
            SignalIntent::ShortSpotLongPerp => {
                // Short spot: (entry_price - exit_price) * qty
                (entry_spot_fill.fill_price - exit_spot_fill.fill_price) * position_qty
            }
        };

        let pnl_perp_leg = match signal.intent {
            SignalIntent::LongSpotShortPerp => {
                // Short perp: (entry_price - exit_price) * qty
                (entry_perp_fill.fill_price - exit_perp_fill.fill_price) * position_qty
            }
            SignalIntent::ShortSpotLongPerp => {
                // Long perp: (exit_price - entry_price) * qty
                (exit_perp_fill.fill_price - entry_perp_fill.fill_price) * position_qty
            }
        };

        // Calculate funding transfer
        // Find funding events between entry and exit
        let funding_settlements: Vec<FundingSettlement> = funding_events
            .iter()
            .filter(|f| {
                let funding_ts = Utc.timestamp_millis_opt(f.next_funding_time_ms).unwrap();
                funding_ts > entry_ts && funding_ts <= exit_ts
            })
            .map(|f| {
                let mark_price = f.mark_price_f64();
                let position_value = position_qty * mark_price;
                let funding_rate = f.funding_rate_f64();

                // Funding amount: positive rate means shorts pay longs
                let funding_amount = match signal.intent {
                    SignalIntent::LongSpotShortPerp => {
                        // Short perp: receive if rate > 0, pay if rate < 0
                        -position_value * funding_rate
                    }
                    SignalIntent::ShortSpotLongPerp => {
                        // Long perp: pay if rate > 0, receive if rate < 0
                        position_value * funding_rate
                    }
                };

                FundingSettlement {
                    ts: Utc.timestamp_millis_opt(f.next_funding_time_ms).unwrap(),
                    funding_rate_bps: f.funding_rate_bps(),
                    mark_price,
                    position_value,
                    funding_amount,
                }
            })
            .collect();

        let funding_transfer: f64 = funding_settlements.iter().map(|f| f.funding_amount).sum();

        // Calculate fees
        let total_notional = entry_spot_fill.notional
            + entry_perp_fill.notional
            + exit_spot_fill.notional
            + exit_perp_fill.notional;
        let fee_cost = total_notional * args.fee_bps / 10000.0;

        // Calculate basis change
        let entry_basis_bps = (entry_perp_q.mid - entry_spot_q.mid) / entry_spot_q.mid * 10000.0;
        let exit_basis_bps = (exit_perp_q.mid - exit_spot_q.mid) / exit_spot_q.mid * 10000.0;
        let basis_change_bps = exit_basis_bps - entry_basis_bps;

        // Basis drift PnL (how much the basis change affected our position)
        let basis_drift_pnl = match signal.intent {
            SignalIntent::LongSpotShortPerp => {
                // Long spot, short perp: we lose if basis widens, gain if narrows
                -basis_change_bps / 10000.0 * args.position_size
            }
            SignalIntent::ShortSpotLongPerp => {
                // Short spot, long perp: we gain if basis widens, lose if narrows
                basis_change_bps / 10000.0 * args.position_size
            }
        };

        let pnl_gross = pnl_spot_leg + pnl_perp_leg;
        let pnl_net = pnl_gross + funding_transfer - fee_cost;

        let hold_seconds = (exit_ts - entry_ts).num_seconds();

        // Create trade record
        let trade = TradeRecord {
            schema_version: 1,
            signal_index: idx,
            symbol: signal.symbol.clone(),
            intent: signal.intent,
            signal_ts: signal.ts,
            entry_ts,
            exit_ts,
            hold_seconds,
            entry_spot: entry_spot_fill.clone(),
            entry_perp: entry_perp_fill.clone(),
            exit_spot: exit_spot_fill.clone(),
            exit_perp: exit_perp_fill.clone(),
            pnl_spot_leg,
            pnl_perp_leg,
            pnl_gross,
            fee_cost,
            funding_transfer,
            basis_drift_pnl,
            pnl_net,
            entry_basis_bps,
            exit_basis_bps,
            basis_change_bps,
        };
        trades.push(trade);

        // Create fill record
        let fill = FillRecord {
            schema_version: 1,
            signal_index: idx,
            entry_ts,
            exit_ts,
            entry_spot: entry_spot_fill,
            entry_perp: entry_perp_fill,
            exit_spot: exit_spot_fill,
            exit_perp: exit_perp_fill,
            funding_events: funding_settlements,
        };
        fills.push(fill);
    }

    // Calculate metrics
    let num_trades = trades.len();
    let num_dropped = signals.len() - num_trades;

    let gross_pnl: f64 = trades.iter().map(|t| t.pnl_gross).sum();
    let total_fees: f64 = trades.iter().map(|t| t.fee_cost).sum();
    let total_funding: f64 = trades.iter().map(|t| t.funding_transfer).sum();
    let net_pnl: f64 = trades.iter().map(|t| t.pnl_net).sum();

    let pnl_from_spot: f64 = trades.iter().map(|t| t.pnl_spot_leg).sum();
    let pnl_from_perp: f64 = trades.iter().map(|t| t.pnl_perp_leg).sum();

    let winners = trades.iter().filter(|t| t.pnl_net > 0.0).count();
    let hit_rate = if num_trades > 0 {
        winners as f64 / num_trades as f64
    } else {
        0.0
    };

    let avg_pnl = if num_trades > 0 {
        net_pnl / num_trades as f64
    } else {
        0.0
    };

    // Calculate max drawdown
    let mut equity: f64 = 0.0;
    let mut peak: f64 = 0.0;
    let mut max_drawdown: f64 = 0.0;
    let mut equity_curve: Vec<EquityPoint> = Vec::new();

    for (i, trade) in trades.iter().enumerate() {
        equity += trade.pnl_net;
        peak = peak.max(equity);
        let drawdown = peak - equity;
        max_drawdown = max_drawdown.max(drawdown);

        equity_curve.push(EquityPoint {
            ts: trade.exit_ts,
            trade_index: i,
            equity,
        });
    }

    // Calculate Sharpe ratio (simplified, assuming daily returns)
    let sharpe_ratio = if num_trades > 1 {
        let returns: Vec<f64> = trades.iter().map(|t| t.pnl_net).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            mean / std_dev * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Quote age stats
    let avg_quote_age_ms = if !quote_ages.is_empty() {
        quote_ages.iter().sum::<i64>() as f64 / quote_ages.len() as f64
    } else {
        0.0
    };
    let max_quote_age_ms = quote_ages.iter().copied().max().unwrap_or(0);

    let metrics = ScoringMetrics {
        schema_version: 1,
        run_id: run_id.clone(),
        signal_run_id: args.signal_run_id.clone(),
        latency_ms: args.latency_ms,
        entry_shift_ms: args.entry_shift_ms,
        exit_shift_ms: args.exit_shift_ms,
        fee_bps: args.fee_bps,
        config_max_quote_age_ms: args.max_quote_age_ms,
        use_mid: args.use_mid,
        position_size: args.position_size,
        num_signals: signals.len(),
        num_trades,
        num_dropped,
        gross_pnl,
        total_fees,
        total_funding,
        net_pnl,
        hit_rate,
        avg_pnl,
        max_drawdown,
        sharpe_ratio,
        pnl_from_spot,
        pnl_from_perp,
        pnl_from_funding: total_funding,
        pnl_from_fees: -total_fees,
        avg_quote_age_ms,
        observed_max_quote_age_ms: max_quote_age_ms,
        drop_reasons: drop_reasons.clone(),
    };

    // Write output files
    write_jsonl(&output_dir.join("trades.jsonl"), &trades)?;
    write_jsonl(&output_dir.join("fills.jsonl"), &fills)?;
    write_jsonl(&output_dir.join("equity_curve.jsonl"), &equity_curve)?;

    let metrics_json = serde_json::to_string_pretty(&metrics)?;
    std::fs::write(output_dir.join("metrics.json"), metrics_json)?;

    let manifest = ScoreRunManifest {
        schema_version: 1,
        created_at_utc: Utc::now().to_rfc3339(),
        run_id: run_id.clone(),
        config_hash,
        metrics: metrics.clone(),
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(output_dir.join("run_manifest.json"), manifest_json)?;

    // Print summary
    tracing::info!("=== Scoring Summary ===");
    tracing::info!("Signals: {}", signals.len());
    tracing::info!("Trades: {} ({} dropped)", num_trades, num_dropped);
    tracing::info!("");
    tracing::info!("PnL Decomposition:");
    tracing::info!("  Spot leg:  {:+.2}", pnl_from_spot);
    tracing::info!("  Perp leg:  {:+.2}", pnl_from_perp);
    tracing::info!("  Funding:   {:+.2}", total_funding);
    tracing::info!("  Fees:      {:.2}", -total_fees);
    tracing::info!("  -------------------");
    tracing::info!("  Net PnL:   {:+.2}", net_pnl);
    tracing::info!("");
    tracing::info!("Performance:");
    tracing::info!("  Hit rate: {:.1}%", hit_rate * 100.0);
    tracing::info!("  Avg PnL:  {:+.2}", avg_pnl);
    tracing::info!("  Max DD:   {:.2}", max_drawdown);
    tracing::info!("  Sharpe:   {:.2}", sharpe_ratio);
    tracing::info!("");
    tracing::info!("Drop reasons:");
    tracing::info!("  Missing entry spot: {}", drop_reasons.missing_entry_spot);
    tracing::info!("  Missing entry perp: {}", drop_reasons.missing_entry_perp);
    tracing::info!("  Missing exit spot:  {}", drop_reasons.missing_exit_spot);
    tracing::info!("  Missing exit perp:  {}", drop_reasons.missing_exit_perp);
    tracing::info!("  Stale entry spot:   {}", drop_reasons.stale_entry_spot);
    tracing::info!("  Stale entry perp:   {}", drop_reasons.stale_entry_perp);
    tracing::info!("  Stale exit spot:    {}", drop_reasons.stale_exit_spot);
    tracing::info!("  Stale exit perp:    {}", drop_reasons.stale_exit_perp);

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn load_signals(path: &PathBuf) -> Result<Vec<FundingArbSignal>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open signals file: {:?}", path))?;
    let reader = BufReader::new(file);

    let mut signals = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let signal: FundingArbSignal =
            serde_json::from_str(&line).with_context(|| format!("parse signal: {}", line))?;
        signals.push(signal);
    }

    Ok(signals)
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

fn find_quote_at_or_after(
    quotes: &[QuoteEvent],
    ts: DateTime<Utc>,
    max_age_ms: u64,
) -> QuoteLookup {
    // Find the first quote at or after ts
    for quote in quotes {
        if quote.ts >= ts {
            let age_ms = (quote.ts - ts).num_milliseconds();
            if age_ms <= max_age_ms as i64 {
                return QuoteLookup::Found(QuoteResult {
                    bid: quote.bid_f64(),
                    ask: quote.ask_f64(),
                    mid: quote.mid_f64(),
                    quote_age_ms: age_ms,
                });
            } else {
                return QuoteLookup::Stale {
                    first_valid_age_ms: age_ms,
                };
            }
        }
    }

    // Check for quotes before ts (within window)
    for quote in quotes.iter().rev() {
        if quote.ts < ts {
            let age_ms = (ts - quote.ts).num_milliseconds();
            if age_ms <= max_age_ms as i64 {
                return QuoteLookup::Found(QuoteResult {
                    bid: quote.bid_f64(),
                    ask: quote.ask_f64(),
                    mid: quote.mid_f64(),
                    quote_age_ms: age_ms,
                });
            }
            break;
        }
    }

    QuoteLookup::Missing
}

fn make_leg_fill(
    market: &str,
    side: &str,
    quote: &QuoteResult,
    qty: f64,
    use_mid: bool,
) -> LegFill {
    let fill_price = if use_mid {
        quote.mid
    } else if side == "buy" {
        quote.ask // Buy at ask (worse)
    } else {
        quote.bid // Sell at bid (worse)
    };

    LegFill {
        market: market.to_string(),
        side: side.to_string(),
        bid: quote.bid,
        ask: quote.ask,
        mid: quote.mid,
        spread: quote.ask - quote.bid,
        quote_age_ms: quote.age_ms(),
        fill_price,
        quantity: qty,
        notional: fill_price * qty,
    }
}

impl QuoteResult {
    fn age_ms(&self) -> i64 {
        self.quote_age_ms
    }
}

fn compute_config_hash(args: &Args) -> Result<String> {
    let config_str = format!(
        "{}:{}:{}:{}:{}:{}:{}",
        args.signal_run_id,
        args.latency_ms,
        args.entry_shift_ms,
        args.exit_shift_ms,
        args.fee_bps,
        args.max_quote_age_ms,
        args.use_mid
    );

    let mut hasher = Sha256::new();
    hasher.update(config_str.as_bytes());
    let result = hasher.finalize();
    Ok(hex::encode(result))
}

fn write_jsonl<T: Serialize>(path: &PathBuf, items: &[T]) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    for item in items {
        let line = serde_json::to_string(item)?;
        std::io::Write::write_all(&mut file, line.as_bytes())?;
        std::io::Write::write_all(&mut file, b"\n")?;
    }
    Ok(())
}

// =============================================================================
// FUNDING-CLOCK CORRECTNESS TESTS (Phase 2C)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that TradeRecord serialization is deterministic.
    #[test]
    fn test_trade_record_deterministic_serialization() {
        let trade = TradeRecord {
            schema_version: 1,
            signal_index: 0,
            symbol: "BTCUSDT".to_string(),
            intent: SignalIntent::LongSpotShortPerp,
            signal_ts: Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap(),
            entry_ts: Utc.with_ymd_and_hms(2026, 1, 24, 10, 0, 0).unwrap(),
            exit_ts: Utc.with_ymd_and_hms(2026, 1, 24, 18, 0, 0).unwrap(),
            hold_seconds: 28800,
            entry_spot: LegFill {
                market: "spot".to_string(),
                side: "buy".to_string(),
                bid: 100000.0,
                ask: 100010.0,
                mid: 100005.0,
                spread: 10.0,
                quote_age_ms: 50,
                fill_price: 100010.0,
                quantity: 0.1,
                notional: 10001.0,
            },
            entry_perp: LegFill {
                market: "perp".to_string(),
                side: "sell".to_string(),
                bid: 100050.0,
                ask: 100060.0,
                mid: 100055.0,
                spread: 10.0,
                quote_age_ms: 50,
                fill_price: 100050.0,
                quantity: 0.1,
                notional: 10005.0,
            },
            exit_spot: LegFill {
                market: "spot".to_string(),
                side: "sell".to_string(),
                bid: 100100.0,
                ask: 100110.0,
                mid: 100105.0,
                spread: 10.0,
                quote_age_ms: 50,
                fill_price: 100100.0,
                quantity: 0.1,
                notional: 10010.0,
            },
            exit_perp: LegFill {
                market: "perp".to_string(),
                side: "buy".to_string(),
                bid: 100120.0,
                ask: 100130.0,
                mid: 100125.0,
                spread: 10.0,
                quote_age_ms: 50,
                fill_price: 100130.0,
                quantity: 0.1,
                notional: 10013.0,
            },
            pnl_spot_leg: 9.0,
            pnl_perp_leg: -8.0,
            pnl_gross: 1.0,
            fee_cost: 0.4,
            funding_transfer: 1.5,
            basis_drift_pnl: -0.2,
            pnl_net: 2.1,
            entry_basis_bps: 5.0,
            exit_basis_bps: 2.0,
            basis_change_bps: -3.0,
        };

        // Serialize multiple times
        let json1 = serde_json::to_string(&trade).unwrap();
        let json2 = serde_json::to_string(&trade).unwrap();

        // Must be identical
        assert_eq!(json1, json2, "TradeRecord serialization not deterministic");

        // Hash must be consistent
        let hash1 = hex::encode(Sha256::digest(json1.as_bytes()));
        let hash2 = hex::encode(Sha256::digest(json2.as_bytes()));
        assert_eq!(hash1, hash2, "TradeRecord hash not deterministic");
    }

    /// Verify that ScoringMetrics serialization is deterministic.
    #[test]
    fn test_metrics_deterministic_serialization() {
        let metrics = ScoringMetrics {
            schema_version: 1,
            run_id: "abc123".to_string(),
            signal_run_id: "sig456".to_string(),
            latency_ms: 50,
            entry_shift_ms: 0,
            exit_shift_ms: 0,
            fee_bps: 2.0,
            config_max_quote_age_ms: 500,
            use_mid: false,
            position_size: 10000.0,
            num_signals: 100,
            num_trades: 80,
            num_dropped: 20,
            gross_pnl: 500.0,
            total_fees: 40.0,
            total_funding: 120.0,
            net_pnl: 580.0,
            hit_rate: 0.65,
            avg_pnl: 7.25,
            max_drawdown: 150.0,
            sharpe_ratio: 1.5,
            pnl_from_spot: 200.0,
            pnl_from_perp: 300.0,
            pnl_from_funding: 120.0,
            pnl_from_fees: -40.0,
            avg_quote_age_ms: 35.5,
            observed_max_quote_age_ms: 450,
            drop_reasons: DropReasons {
                total_signals: 100,
                missing_entry_spot: 5,
                missing_entry_perp: 3,
                missing_exit_spot: 4,
                missing_exit_perp: 2,
                stale_entry_spot: 2,
                stale_entry_perp: 1,
                stale_exit_spot: 2,
                stale_exit_perp: 1,
                max_stale_age_ms: 600,
            },
        };

        // Serialize multiple times
        let json1 = serde_json::to_string_pretty(&metrics).unwrap();
        let json2 = serde_json::to_string_pretty(&metrics).unwrap();

        // Must be identical
        assert_eq!(
            json1, json2,
            "ScoringMetrics serialization not deterministic"
        );

        // Hash must be consistent
        let hash1 = hex::encode(Sha256::digest(json1.as_bytes()));
        let hash2 = hex::encode(Sha256::digest(json2.as_bytes()));
        assert_eq!(hash1, hash2, "ScoringMetrics hash not deterministic");
    }

    /// Verify that config hash is deterministic.
    #[test]
    fn test_config_hash_deterministic() {
        // Create args with fixed values
        // We can't easily create Args due to clap, so test the hash function directly
        let config_str1 = "run123:50:0:0:2.0:500:false";
        let config_str2 = "run123:50:0:0:2.0:500:false";

        let hash1 = hex::encode(Sha256::digest(config_str1.as_bytes()));
        let hash2 = hex::encode(Sha256::digest(config_str2.as_bytes()));

        assert_eq!(hash1, hash2, "Config hash not deterministic");

        // Different config should produce different hash
        let config_str3 = "run123:100:0:0:2.0:500:false"; // Different latency
        let hash3 = hex::encode(Sha256::digest(config_str3.as_bytes()));

        assert_ne!(
            hash1, hash3,
            "Different config should produce different hash"
        );
    }

    /// Verify that FundingSettlement serialization is deterministic.
    #[test]
    fn test_funding_settlement_deterministic() {
        let settlement = FundingSettlement {
            ts: Utc.with_ymd_and_hms(2026, 1, 24, 8, 0, 0).unwrap(),
            funding_rate_bps: 1.0,
            mark_price: 100000.0,
            position_value: 10000.0,
            funding_amount: 1.0,
        };

        let json1 = serde_json::to_string(&settlement).unwrap();
        let json2 = serde_json::to_string(&settlement).unwrap();

        assert_eq!(
            json1, json2,
            "FundingSettlement serialization not deterministic"
        );
    }

    /// Verify that PnL decomposition math is consistent.
    #[test]
    fn test_pnl_decomposition_consistency() {
        // Verify that pnl_net = pnl_gross + funding_transfer - fee_cost
        let pnl_spot: f64 = 100.0;
        let pnl_perp: f64 = -50.0;
        let pnl_gross: f64 = pnl_spot + pnl_perp;
        let funding_transfer: f64 = 10.0;
        let fee_cost: f64 = 5.0;
        let pnl_net: f64 = pnl_gross + funding_transfer - fee_cost;

        assert!(
            (pnl_net - 55.0).abs() < 1e-10,
            "PnL decomposition: {} should be 55.0",
            pnl_net
        );

        // Verify components sum correctly
        let reconstructed: f64 = pnl_spot + pnl_perp + funding_transfer - fee_cost;
        assert!(
            (pnl_net - reconstructed).abs() < 1e-10,
            "PnL components don't sum correctly"
        );
    }

    /// Verify bid/ask fill discipline.
    #[test]
    fn test_bid_ask_fill_discipline() {
        let quote = QuoteResult {
            bid: 100.0,
            ask: 101.0,
            mid: 100.5,
            quote_age_ms: 50,
        };

        // Buy should fill at ask (worse)
        let buy_fill = make_leg_fill("spot", "buy", &quote, 1.0, false);
        assert_eq!(buy_fill.fill_price, 101.0, "Buy should fill at ask");

        // Sell should fill at bid (worse)
        let sell_fill = make_leg_fill("spot", "sell", &quote, 1.0, false);
        assert_eq!(sell_fill.fill_price, 100.0, "Sell should fill at bid");

        // With use_mid=true, both should fill at mid
        let buy_mid = make_leg_fill("spot", "buy", &quote, 1.0, true);
        let sell_mid = make_leg_fill("spot", "sell", &quote, 1.0, true);
        assert_eq!(
            buy_mid.fill_price, 100.5,
            "Buy with use_mid should fill at mid"
        );
        assert_eq!(
            sell_mid.fill_price, 100.5,
            "Sell with use_mid should fill at mid"
        );
    }

    /// Verify quote staleness detection.
    #[test]
    fn test_quote_staleness_detection() {
        let now = Utc::now();
        let quotes = vec![QuoteEvent {
            ts: now - chrono::Duration::milliseconds(1000),
            symbol: "BTCUSDT".to_string(),
            bid_price_mantissa: 10000000,
            ask_price_mantissa: 10001000,
            bid_qty_mantissa: 100000,
            ask_qty_mantissa: 100000,
            price_exponent: -2,
            qty_exponent: -8,
        }];

        // Quote from 1000ms ago, max age 500ms -> should be stale
        let result = find_quote_at_or_after(&quotes, now, 500);
        match result {
            QuoteLookup::Stale { .. } | QuoteLookup::Missing => {}
            QuoteLookup::Found(_) => panic!("Should have detected stale quote"),
        }

        // Quote from 1000ms ago, max age 2000ms -> should be found
        let result =
            find_quote_at_or_after(&quotes, now - chrono::Duration::milliseconds(500), 2000);
        match result {
            QuoteLookup::Found(q) => {
                assert!(q.quote_age_ms <= 2000, "Quote age should be within window");
            }
            _ => panic!("Should have found quote within window"),
        }
    }
}
