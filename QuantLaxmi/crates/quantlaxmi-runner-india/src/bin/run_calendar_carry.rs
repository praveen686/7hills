//! Strategy v0: SANOS-Gated Calendar Carry Runner (NoTrade-only mode)
//!
//! Runs the calendar carry strategy on replay data and reports gate hit counts.
//! Initial mode is NoTrade-only to validate gate behavior before enabling paper execution.
//!
//! ## Modes (Commit D)
//! - **Manifest-driven**: When `session_manifest.json` exists, uses deterministic inventory.
//! - **Legacy**: Falls back to directory scanning + symbol parsing when no manifest.
//!
//! Usage:
//!   cargo run --bin run_calendar_carry -- --session-dir <path> --underlying NIFTY

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration, NaiveDate, Timelike, Utc};
use clap::Parser;
use quantlaxmi_options::sanos::{ExpirySlice, OptionQuote, SanosCalibrator, SanosSlice};
use quantlaxmi_options::strategies::{
    AuditRecord, CalendarCarryStrategy, GateCheckResult, Phase8Features, QuoteSnapshot,
    SessionMeta, StraddleQuotes, StrategyContext, StrategyDecision,
};
use quantlaxmi_runner_common::manifest_io::{sha256_hex, write_atomic};
use quantlaxmi_runner_common::run_manifest::{
    InputBinding, OutputBinding, RunManifest, config_hash, git_commit_string, hash_file,
    persist_run_manifest_atomic,
};
use quantlaxmi_runner_india::sanos_io::{
    SanosUnderlyingInventory, log_legacy_mode, log_manifest_mode, try_load_sanos_inventory,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::{Path, PathBuf};
use tracing::info;

/// NSE market hours (IST = UTC + 5:30)
const MARKET_OPEN_HOUR: u32 = 9;
const MARKET_OPEN_MINUTE: u32 = 15;
const MARKET_CLOSE_HOUR: u32 = 15;
const MARKET_CLOSE_MINUTE: u32 = 30;
const IST_OFFSET_HOURS: i64 = 5;
const IST_OFFSET_MINUTES: i64 = 30;

/// Check if timestamp is within NSE market hours (9:15 - 15:30 IST)
fn is_market_hours(ts: DateTime<Utc>) -> bool {
    // Convert UTC to IST
    let ist = ts + Duration::hours(IST_OFFSET_HOURS) + Duration::minutes(IST_OFFSET_MINUTES);
    let hour = ist.hour();
    let minute = ist.minute();

    let time_minutes = hour * 60 + minute;
    let open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE;
    let close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE;

    time_minutes >= open_minutes && time_minutes <= close_minutes
}

#[derive(Parser, Debug)]
#[command(name = "run_calendar_carry")]
#[command(about = "Strategy v0: SANOS-Gated Calendar Carry Runner")]
struct Args {
    /// Session directory containing captured tick data
    #[arg(long)]
    session_dir: PathBuf,

    /// Underlying to run strategy for (NIFTY or BANKNIFTY)
    #[arg(long, default_value = "NIFTY")]
    underlying: String,

    /// Output JSON file for audit records
    #[arg(long, default_value = "calendar_carry_audit.jsonl")]
    output: PathBuf,

    /// Decision interval in seconds
    #[arg(long, default_value = "60")]
    interval_secs: u64,

    /// SANOS smoothness parameter η
    #[arg(long, default_value = "0.25")]
    eta: f64,
}

/// Minimal deterministic run config used for RunManifest hashing.
#[derive(Debug, Clone, Serialize)]
struct RunConfigHash {
    underlying: String,
    interval_secs: u64,
    eta: f64,
}

/// Canonical signal event emitted when strategy generates an Enter intent.
///
/// This is intentionally small and stable so downstream backtests can remain generic.
#[derive(Debug, Clone, Serialize)]
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

/// Tick event from captured session
#[derive(Debug, Clone, Deserialize)]
struct TickEvent {
    ts: DateTime<Utc>,
    #[allow(dead_code)]
    tradingsymbol: String,
    #[allow(dead_code)]
    instrument_token: u32,
    bid_price: i64,
    ask_price: i64,
    bid_qty: u32,
    ask_qty: u32,
    ltp: i64,
    ltq: u32,
    #[allow(dead_code)]
    volume: u64,
    price_exponent: i32,
    #[allow(dead_code)]
    integrity_tier: String,
}

/// Quote truth audit record (Phase 9.2 Q1)
#[derive(Debug, Clone, Serialize)]
struct QuoteTruthAudit {
    symbol: String,
    capture_ts: DateTime<Utc>,
    best_bid: f64,
    best_ask: f64,
    bid_size: u32,
    ask_size: u32,
    last_price: f64,
    last_qty: u32,
    spread: f64,
    spread_bps: f64,
}

/// Cooldown in seconds after entry (Phase 9: S2)
const COOLDOWN_SECS: i64 = 600;

/// Phase 9.3: Maximum quote age in seconds for Q1-lite validation
const MAX_QUOTE_AGE_SECS: i64 = 2;

/// Phase 9.3: Q1-lite validation result for a single leg
#[derive(Debug, Clone)]
struct Q1LegValidation {
    symbol: String,
    bid_qty_zero: bool,
    ask_qty_zero: bool,
    quote_stale: bool,
    quote_age_ms: i64,
    spread_invalid: bool, // spread <= 0 or ask < bid
    passed: bool,
    fail_reason: Option<String>,
}

/// Phase 9.3: Q1-lite validation result for straddle (CE + PE)
#[derive(Debug, Clone)]
struct Q1StraddleValidation {
    ce: Q1LegValidation,
    pe: Q1LegValidation,
    any_failed: bool,
}

/// Gate hit counters
#[derive(Debug, Default, Serialize)]
struct GateCounters {
    total_decisions: u32,
    h1_surface_fail: u32,
    h2_calendar_fail: u32,
    h3_quote_front_fail: u32,
    h3_quote_back_fail: u32,
    h4_liquidity_front_fail: u32,
    h4_liquidity_back_fail: u32,
    carry_fail: u32,
    r1_inversion_fail: u32,
    r2_skew_fail: u32,
    // Phase 9 Completion: Economic hardeners
    e1_premium_gap_fail: u32,
    e2_friction_dominance_fail: u32,
    // Phase 9.2: Friction floor
    e3_friction_floor_fail: u32,
    e3_floor_binding_count: u32, // How often floor > observed
    // Phase 9.3: Q1-lite audit
    q1_front_fail: u32,
    q1_back_fail: u32,
    q1_bid_qty_zero: u32,
    q1_ask_qty_zero: u32,
    q1_quote_stale: u32,
    q1_spread_invalid: u32,
    all_gates_pass: u32,
    enter_after_cooldown: u32,
    cooldown_blocked: u32,
    // Market hours tracking
    outside_market_hours: u32,
    inside_market_hours: u32,
}

/// Active trade for tracking 10-minute exit PnL
#[derive(Debug, Clone)]
struct ActiveTrade {
    entry_ts: DateTime<Utc>,
    front_straddle_entry: StraddleQuotes,
    back_straddle_entry: StraddleQuotes,
    hedge_ratio: f64,
    entry_fill: ConservativeFill,
    friction_round_eff: f64,
}

/// Exit result with conservative fills
#[derive(Debug, Clone, Serialize)]
struct ExitResult {
    hold_seconds: i64,
    /// Conservative exit PnL (close longs at bid, shorts at ask)
    pnl_conservative: f64,
    /// Mid-mark exit PnL
    pnl_mid: f64,
    /// PnL per rupee of effective friction
    pnl_per_friction: f64,
    /// Entry friction
    entry_friction: f64,
    /// Effective friction (with floor)
    friction_round_eff: f64,
}

/// Conservative fill result (ask/bid fills)
#[derive(Debug, Clone, Serialize)]
struct ConservativeFill {
    /// Entry cost for short front straddle (we receive bid prices)
    entry_front_credit: f64,
    /// Entry cost for long back straddle (we pay ask prices)
    entry_back_debit: f64,
    /// Total entry cost (negative = net credit, positive = net debit)
    net_entry_cost: f64,
    /// Friction from entry (half-spreads)
    entry_friction: f64,
    /// Mark-to-mid value at entry
    mark_to_mid: f64,
}

/// Quote validation record
#[derive(Debug, Clone, Serialize)]
struct QuoteValidation {
    ce_front_bid: f64,
    ce_front_ask: f64,
    pe_front_bid: f64,
    pe_front_ask: f64,
    ce_back_bid: f64,
    ce_back_ask: f64,
    pe_back_bid: f64,
    pe_back_ask: f64,
    straddle_spread_front: f64,
    straddle_spread_back: f64,
    straddle_mid_front: f64,
    straddle_mid_back: f64,
}

/// Distribution stats for a metric
#[derive(Debug, Default)]
struct DistributionStats {
    values: Vec<f64>,
}

impl DistributionStats {
    fn add(&mut self, v: f64) {
        if v.is_finite() {
            self.values.push(v);
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return f64::NAN;
        }
        let mut sorted = self.values.clone();
        // P3: Use total_cmp to avoid panic on NaN (add() already filters non-finite values)
        sorted.sort_by(|a, b| a.total_cmp(b));
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn p50(&self) -> f64 {
        self.percentile(50.0)
    }
    fn p90(&self) -> f64 {
        self.percentile(90.0)
    }
    fn p99(&self) -> f64 {
        self.percentile(99.0)
    }
    fn count(&self) -> usize {
        self.values.len()
    }
}

impl GateCounters {
    fn update(&mut self, gates: &GateCheckResult) {
        self.total_decisions += 1;
        if !gates.h1_surface.passed {
            self.h1_surface_fail += 1;
        }
        if !gates.h2_calendar.passed {
            self.h2_calendar_fail += 1;
        }
        if !gates.h3_quote_front.passed {
            self.h3_quote_front_fail += 1;
        }
        if !gates.h3_quote_back.passed {
            self.h3_quote_back_fail += 1;
        }
        if !gates.h4_liquidity_front.passed {
            self.h4_liquidity_front_fail += 1;
        }
        if !gates.h4_liquidity_back.passed {
            self.h4_liquidity_back_fail += 1;
        }
        if !gates.carry.passed {
            self.carry_fail += 1;
        }
        if !gates.r1_inversion.passed {
            self.r1_inversion_fail += 1;
        }
        if !gates.r2_skew.passed {
            self.r2_skew_fail += 1;
        }
        if !gates.e1_premium_gap.passed {
            self.e1_premium_gap_fail += 1;
        }
        if !gates.e2_friction_dominance.passed {
            self.e2_friction_dominance_fail += 1;
        }
        if !gates.e3_friction_floor.passed {
            self.e3_friction_floor_fail += 1;
        }
        // Track floor binding: check if E3 reason contains "FLOOR_BINDING"
        if let Some(ref reason) = gates.e3_friction_floor.reason
            && reason.contains("FLOOR_BINDING")
        {
            self.e3_floor_binding_count += 1;
        }
        if gates.all_passed() {
            self.all_gates_pass += 1;
        }
    }

    fn pct(&self, n: u32) -> f64 {
        if self.total_decisions == 0 {
            0.0
        } else {
            100.0 * n as f64 / self.total_decisions as f64
        }
    }

    fn print_report(&self) {
        info!("=== Gate Hit Counts ===");
        info!("Total decisions:      {}", self.total_decisions);
        info!(
            "H1 surface fail:      {} ({:.1}%)",
            self.h1_surface_fail,
            self.pct(self.h1_surface_fail)
        );
        info!(
            "H2 calendar fail:     {} ({:.1}%)",
            self.h2_calendar_fail,
            self.pct(self.h2_calendar_fail)
        );
        info!(
            "H3 quote_front fail:  {} ({:.1}%)",
            self.h3_quote_front_fail,
            self.pct(self.h3_quote_front_fail)
        );
        info!(
            "H3 quote_back fail:   {} ({:.1}%)",
            self.h3_quote_back_fail,
            self.pct(self.h3_quote_back_fail)
        );
        info!(
            "H4 liq_front fail:    {} ({:.1}%)",
            self.h4_liquidity_front_fail,
            self.pct(self.h4_liquidity_front_fail)
        );
        info!(
            "H4 liq_back fail:     {} ({:.1}%)",
            self.h4_liquidity_back_fail,
            self.pct(self.h4_liquidity_back_fail)
        );
        info!(
            "Carry fail:           {} ({:.1}%)",
            self.carry_fail,
            self.pct(self.carry_fail)
        );
        info!(
            "R1 inversion fail:    {} ({:.1}%)",
            self.r1_inversion_fail,
            self.pct(self.r1_inversion_fail)
        );
        info!(
            "R2 skew fail:         {} ({:.1}%)",
            self.r2_skew_fail,
            self.pct(self.r2_skew_fail)
        );
        info!(
            "E1 premium_gap fail:  {} ({:.1}%)",
            self.e1_premium_gap_fail,
            self.pct(self.e1_premium_gap_fail)
        );
        info!(
            "E2 friction_dom fail: {} ({:.1}%)",
            self.e2_friction_dominance_fail,
            self.pct(self.e2_friction_dominance_fail)
        );
        info!(
            "E3 fric_floor fail:   {} ({:.1}%)",
            self.e3_friction_floor_fail,
            self.pct(self.e3_friction_floor_fail)
        );
        info!(
            "E3 floor binding:     {} ({:.1}%)",
            self.e3_floor_binding_count,
            self.pct(self.e3_floor_binding_count)
        );
        info!("--- Q1-lite Audit (Phase 9.3) ---");
        info!(
            "Q1 front fail:        {} ({:.1}%)",
            self.q1_front_fail,
            self.pct(self.q1_front_fail)
        );
        info!(
            "Q1 back fail:         {} ({:.1}%)",
            self.q1_back_fail,
            self.pct(self.q1_back_fail)
        );
        info!(
            "Q1 bid_qty=0:         {} ({:.1}%)",
            self.q1_bid_qty_zero,
            self.pct(self.q1_bid_qty_zero)
        );
        info!(
            "Q1 ask_qty=0:         {} ({:.1}%)",
            self.q1_ask_qty_zero,
            self.pct(self.q1_ask_qty_zero)
        );
        info!(
            "Q1 quote stale:       {} ({:.1}%)",
            self.q1_quote_stale,
            self.pct(self.q1_quote_stale)
        );
        info!(
            "Q1 spread invalid:    {} ({:.1}%)",
            self.q1_spread_invalid,
            self.pct(self.q1_spread_invalid)
        );
        info!("--- Market Hours ---");
        info!(
            "Inside market hours:  {} ({:.1}%)",
            self.inside_market_hours,
            self.pct(self.inside_market_hours)
        );
        info!(
            "Outside market hours: {} ({:.1}%)",
            self.outside_market_hours,
            self.pct(self.outside_market_hours)
        );
        info!("--- Summary ---");
        info!(
            "ALL GATES PASS:       {} ({:.1}%)",
            self.all_gates_pass,
            self.pct(self.all_gates_pass)
        );
        info!(
            "Cooldown blocked:     {} ({:.1}%)",
            self.cooldown_blocked,
            self.pct(self.cooldown_blocked)
        );
        info!("ENTER after cooldown: {}", self.enter_after_cooldown);
    }
}

/// Compute conservative fill for a trade entry
fn compute_conservative_fill(
    front_straddle: &StraddleQuotes,
    back_straddle: &StraddleQuotes,
    h: f64,
) -> ConservativeFill {
    // SHORT front straddle: we SELL, so we receive bid prices
    let front_bid = front_straddle.ce.bid + front_straddle.pe.bid;
    let front_ask = front_straddle.ce.ask + front_straddle.pe.ask;
    let entry_front_credit = front_bid; // We receive bid

    // LONG back straddle: we BUY, so we pay ask prices
    let back_bid = back_straddle.ce.bid + back_straddle.pe.bid;
    let back_ask = back_straddle.ce.ask + back_straddle.pe.ask;
    let entry_back_debit = h * back_ask; // We pay ask, scaled by hedge ratio

    // Net entry cost (positive = net debit)
    let net_entry_cost = entry_back_debit - entry_front_credit;

    // Mark-to-mid (what we would get at mid prices)
    let front_mid = front_straddle.mid();
    let back_mid = back_straddle.mid();
    let mark_to_mid = (h * back_mid) - front_mid;

    // Entry friction = slippage from mid to actual fill
    // = (ask - mid) for buys + (mid - bid) for sells
    // = half_spread_front + h * half_spread_back
    let spread_front = front_ask - front_bid;
    let spread_back = back_ask - back_bid;
    let entry_friction = (spread_front / 2.0) + h * (spread_back / 2.0);

    ConservativeFill {
        entry_front_credit,
        entry_back_debit,
        net_entry_cost,
        entry_friction,
        mark_to_mid,
    }
}

/// Compute conservative exit PnL for 10-minute hold
/// Exit: close LONG back leg at BID, close SHORT front leg at ASK
fn compute_exit_pnl(
    entry: &ActiveTrade,
    exit_front: &StraddleQuotes,
    exit_back: &StraddleQuotes,
    exit_ts: DateTime<Utc>,
) -> ExitResult {
    let h = entry.hedge_ratio;

    // Exit fills (conservative):
    // - Close SHORT front leg: we BUY back, pay ASK
    // - Close LONG back leg: we SELL, receive BID
    let exit_front_debit = exit_front.ce.ask + exit_front.pe.ask; // Buy at ask
    let exit_back_credit = h * (exit_back.ce.bid + exit_back.pe.bid); // Sell at bid

    // Entry fills were:
    // - SHORT front: received bid (credit)
    // - LONG back: paid ask (debit)
    let entry_front_credit = entry.entry_fill.entry_front_credit;
    let entry_back_debit = entry.entry_fill.entry_back_debit;

    // Conservative PnL:
    // Front leg: entry credit - exit debit (we received, now we pay)
    // Back leg: exit credit - entry debit (we paid, now we receive)
    let pnl_front = entry_front_credit - exit_front_debit;
    let pnl_back = exit_back_credit - entry_back_debit;
    let pnl_conservative = pnl_front + pnl_back;

    // Mid-mark PnL for comparison
    let exit_front_mid = exit_front.mid();
    let exit_back_mid = h * exit_back.mid();
    let entry_front_mid = entry.front_straddle_entry.mid();
    let entry_back_mid = h * entry.back_straddle_entry.mid();
    let pnl_mid = (entry_front_mid - exit_front_mid) + (exit_back_mid - entry_back_mid);

    // Hold time
    let hold_seconds = (exit_ts - entry.entry_ts).num_seconds();

    // PnL per friction
    let pnl_per_friction = if entry.friction_round_eff > 1e-9 {
        pnl_conservative / entry.friction_round_eff
    } else {
        0.0
    };

    ExitResult {
        hold_seconds,
        pnl_conservative,
        pnl_mid,
        pnl_per_friction,
        entry_friction: entry.entry_fill.entry_friction,
        friction_round_eff: entry.friction_round_eff,
    }
}

/// Build quote truth audit record (Phase 9.2 Q1)
#[allow(dead_code)]
fn build_quote_truth_audit(tick: &TickEvent, symbol: &str) -> QuoteTruthAudit {
    let price_mult = 10f64.powi(tick.price_exponent);
    let bid = tick.bid_price as f64 * price_mult;
    let ask = tick.ask_price as f64 * price_mult;
    let ltp = tick.ltp as f64 * price_mult;
    let spread = ask - bid;
    let mid = (bid + ask) / 2.0;
    let spread_bps = if mid > 0.0 {
        10000.0 * spread / mid
    } else {
        0.0
    };

    QuoteTruthAudit {
        symbol: symbol.to_string(),
        capture_ts: tick.ts,
        best_bid: bid,
        best_ask: ask,
        bid_size: tick.bid_qty,
        ask_size: tick.ask_qty,
        last_price: ltp,
        last_qty: tick.ltq,
        spread,
        spread_bps,
    }
}

/// Phase 9.3: Q1-lite validation for a single option leg
fn validate_q1_leg(tick: &TickEvent, symbol: &str, decision_ts: DateTime<Utc>) -> Q1LegValidation {
    let price_mult = 10f64.powi(tick.price_exponent);
    let bid = tick.bid_price as f64 * price_mult;
    let ask = tick.ask_price as f64 * price_mult;

    let quote_age_ms = (decision_ts - tick.ts).num_milliseconds();
    let bid_qty_zero = tick.bid_qty == 0;
    let ask_qty_zero = tick.ask_qty == 0;
    let quote_stale = quote_age_ms > (MAX_QUOTE_AGE_SECS * 1000);
    let spread_invalid = ask <= bid || (ask - bid) <= 0.0;

    let mut fail_reasons = Vec::new();
    if bid_qty_zero {
        fail_reasons.push("bid_qty=0");
    }
    if ask_qty_zero {
        fail_reasons.push("ask_qty=0");
    }
    if quote_stale {
        fail_reasons.push(format!("stale({}ms)", quote_age_ms).leak());
    }
    if spread_invalid {
        fail_reasons.push("spread<=0");
    }

    let passed = !bid_qty_zero && !ask_qty_zero && !quote_stale && !spread_invalid;
    let fail_reason = if fail_reasons.is_empty() {
        None
    } else {
        Some(fail_reasons.join("|"))
    };

    Q1LegValidation {
        symbol: symbol.to_string(),
        bid_qty_zero,
        ask_qty_zero,
        quote_stale,
        quote_age_ms,
        spread_invalid,
        passed,
        fail_reason,
    }
}

/// Phase 9.3: Q1-lite validation for a straddle (CE + PE)
fn validate_q1_straddle(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying: &str,
    expiry: &str,
    strike: f64,
    decision_ts: DateTime<Utc>,
) -> Option<Q1StraddleValidation> {
    let strike_u32 = strike.round() as u32;
    let ce_symbol = format!("{}{}{:05}CE", underlying, expiry, strike_u32);
    let pe_symbol = format!("{}{}{:05}PE", underlying, expiry, strike_u32);

    let ce_tick = find_closest_tick(symbol_ticks, &ce_symbol, decision_ts)?;
    let pe_tick = find_closest_tick(symbol_ticks, &pe_symbol, decision_ts)?;

    let ce_val = validate_q1_leg(ce_tick, &ce_symbol, decision_ts);
    let pe_val = validate_q1_leg(pe_tick, &pe_symbol, decision_ts);

    let any_failed = !ce_val.passed || !pe_val.passed;

    Some(Q1StraddleValidation {
        ce: ce_val,
        pe: pe_val,
        any_failed,
    })
}

/// Update Q1 counters based on validation results
fn update_q1_counters(
    counters: &mut GateCounters,
    front: &Q1StraddleValidation,
    back: &Q1StraddleValidation,
) {
    if front.any_failed {
        counters.q1_front_fail += 1;
    }
    if back.any_failed {
        counters.q1_back_fail += 1;
    }

    // Count individual failure reasons and log each failure
    for leg in [&front.ce, &front.pe, &back.ce, &back.pe] {
        if !leg.passed {
            // Log Q1 failure with symbol and fail_reason
            if let Some(ref reason) = leg.fail_reason {
                tracing::debug!(
                    symbol = %leg.symbol,
                    fail_reason = %reason,
                    quote_age_ms = leg.quote_age_ms,
                    "Q1-lite validation failed"
                );
            }
        }
        if leg.bid_qty_zero {
            counters.q1_bid_qty_zero += 1;
        }
        if leg.ask_qty_zero {
            counters.q1_ask_qty_zero += 1;
        }
        if leg.quote_stale {
            counters.q1_quote_stale += 1;
        }
        if leg.spread_invalid {
            counters.q1_spread_invalid += 1;
        }
    }
}

/// Build quote validation record
fn build_quote_validation(
    front_straddle: &StraddleQuotes,
    back_straddle: &StraddleQuotes,
) -> QuoteValidation {
    QuoteValidation {
        ce_front_bid: front_straddle.ce.bid,
        ce_front_ask: front_straddle.ce.ask,
        pe_front_bid: front_straddle.pe.bid,
        pe_front_ask: front_straddle.pe.ask,
        ce_back_bid: back_straddle.ce.bid,
        ce_back_ask: back_straddle.ce.ask,
        pe_back_bid: back_straddle.pe.bid,
        pe_back_ask: back_straddle.pe.ask,
        straddle_spread_front: front_straddle.spread(),
        straddle_spread_back: back_straddle.spread(),
        straddle_mid_front: front_straddle.mid(),
        straddle_mid_back: back_straddle.mid(),
    }
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

/// Minutes until market close (IST)
fn minutes_to_close(ts: DateTime<Utc>) -> u64 {
    // Convert UTC to IST (UTC + 5:30)
    let ist_ts = ts + Duration::hours(5) + Duration::minutes(30);
    let close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE;
    let current_minutes = ist_ts.hour() * 60 + ist_ts.minute();

    if current_minutes >= close_minutes {
        0
    } else {
        (close_minutes - current_minutes) as u64
    }
}

/// Discover all expiries in session for given underlying
fn discover_expiries(session_dir: &Path, underlying: &str) -> Result<Vec<String>> {
    let mut expiries = std::collections::HashSet::new();

    for entry in std::fs::read_dir(session_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        // P3: Handle missing file_name gracefully (shouldn't happen with read_dir)
        let Some(fname) = path.file_name() else {
            continue;
        };
        let symbol = fname.to_string_lossy().to_string();

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

/// Load all ticks for an underlying (all expiries)
fn load_all_ticks(session_dir: &Path, underlying: &str) -> Result<HashMap<String, Vec<TickEvent>>> {
    let mut symbol_ticks: HashMap<String, Vec<TickEvent>> = HashMap::new();

    for entry in std::fs::read_dir(session_dir)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        // P3: Handle missing file_name gracefully (shouldn't happen with read_dir)
        let Some(fname) = path.file_name() else {
            continue;
        };
        let symbol = fname.to_string_lossy().to_string();

        if let Some((und, _exp, _strike, _is_call)) = parse_symbol(&symbol) {
            if und != underlying {
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

// =============================================================================
// MANIFEST-DRIVEN LOADERS (Commit D)
// =============================================================================

/// Load all ticks for an underlying using manifest inventory (no directory scan).
fn load_all_ticks_manifest(
    session_dir: &Path,
    underlying_inv: &SanosUnderlyingInventory,
) -> Result<HashMap<String, Vec<TickEvent>>> {
    let mut symbol_ticks: HashMap<String, Vec<TickEvent>> = HashMap::new();

    for tick_info in &underlying_inv.tick_outputs {
        let ticks_file = session_dir.join(&tick_info.path);
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

        symbol_ticks.insert(tick_info.symbol.clone(), ticks);
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
) -> ExpirySlice {
    let expiry_str = expiry.format("%Y-%m-%d").to_string();
    let mut slice = ExpirySlice::new(
        &underlying_inv.underlying,
        &expiry_str,
        target_ts,
        time_to_exp,
    );

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

    slice
}

/// Calculate time to expiry in years from NaiveDate
fn time_to_expiry_from_date(now: DateTime<Utc>, exp_date: NaiveDate) -> f64 {
    let exp_datetime = exp_date
        .and_hms_opt(10, 0, 0)
        .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));

    if let Some(exp_dt) = exp_datetime {
        let days = (exp_dt - now).num_seconds() as f64 / 86400.0;
        return (days / 365.0).max(1.0 / 365.0);
    }

    7.0 / 365.0
}

/// Build ExpirySlice from tick data at a specific timestamp
fn build_slice(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying: &str,
    expiry: &str,
    target_ts: DateTime<Utc>,
    time_to_exp: f64,
) -> ExpirySlice {
    let mut slice = ExpirySlice::new(underlying, expiry, target_ts, time_to_exp);

    for (symbol, ticks) in symbol_ticks {
        let parsed = parse_symbol(symbol);
        if parsed.is_none() {
            continue;
        }
        let (_, exp, strike, is_call) = parsed.unwrap();
        if exp != expiry {
            continue;
        }

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

    slice
}

/// Build straddle quotes from tick data at ATM strike (legacy mode - uses symbol construction)
fn build_straddle_quotes(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying: &str,
    expiry: &str,
    atm_strike: f64,
    target_ts: DateTime<Utc>,
) -> Option<StraddleQuotes> {
    let strike_u32 = atm_strike.round() as u32;

    let ce_symbol = format!("{}{}{:05}CE", underlying, expiry, strike_u32);
    let pe_symbol = format!("{}{}{:05}PE", underlying, expiry, strike_u32);

    let ce_tick = find_closest_tick(symbol_ticks, &ce_symbol, target_ts)?;
    let pe_tick = find_closest_tick(symbol_ticks, &pe_symbol, target_ts)?;

    let ce_mult = 10f64.powi(ce_tick.price_exponent);
    let pe_mult = 10f64.powi(pe_tick.price_exponent);

    Some(StraddleQuotes {
        expiry: expiry.to_string(),
        strike: atm_strike,
        ce: QuoteSnapshot {
            bid: ce_tick.bid_price as f64 * ce_mult,
            ask: ce_tick.ask_price as f64 * ce_mult,
            last_ts: ce_tick.ts,
        },
        pe: QuoteSnapshot {
            bid: pe_tick.bid_price as f64 * pe_mult,
            ask: pe_tick.ask_price as f64 * pe_mult,
            last_ts: pe_tick.ts,
        },
    })
}

/// Build straddle quotes from tick data using manifest instrument info (no symbol parsing).
/// This is the manifest-driven equivalent of `build_straddle_quotes()`.
fn build_straddle_quotes_manifest(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying_inv: &SanosUnderlyingInventory,
    expiry: NaiveDate,
    atm_strike: f64,
    target_ts: DateTime<Utc>,
) -> Option<StraddleQuotes> {
    let instruments = underlying_inv.get_instruments_for_expiry(expiry);

    // Find CE and PE at ATM strike from manifest instruments
    let ce_instr = instruments
        .iter()
        .find(|i| i.instrument_type == "CE" && (i.strike - atm_strike).abs() < 0.01)?;
    let pe_instr = instruments
        .iter()
        .find(|i| i.instrument_type == "PE" && (i.strike - atm_strike).abs() < 0.01)?;

    let ce_tick = find_closest_tick(symbol_ticks, &ce_instr.tradingsymbol, target_ts)?;
    let pe_tick = find_closest_tick(symbol_ticks, &pe_instr.tradingsymbol, target_ts)?;

    let ce_mult = 10f64.powi(ce_tick.price_exponent);
    let pe_mult = 10f64.powi(pe_tick.price_exponent);

    Some(StraddleQuotes {
        expiry: expiry.format("%Y-%m-%d").to_string(),
        strike: atm_strike,
        ce: QuoteSnapshot {
            bid: ce_tick.bid_price as f64 * ce_mult,
            ask: ce_tick.ask_price as f64 * ce_mult,
            last_ts: ce_tick.ts,
        },
        pe: QuoteSnapshot {
            bid: pe_tick.bid_price as f64 * pe_mult,
            ask: pe_tick.ask_price as f64 * pe_mult,
            last_ts: pe_tick.ts,
        },
    })
}

/// Validate Q1-lite for straddle using manifest instrument info (no symbol parsing).
fn validate_q1_straddle_manifest(
    symbol_ticks: &HashMap<String, Vec<TickEvent>>,
    underlying_inv: &SanosUnderlyingInventory,
    expiry: NaiveDate,
    atm_strike: f64,
    decision_ts: DateTime<Utc>,
) -> Option<Q1StraddleValidation> {
    let instruments = underlying_inv.get_instruments_for_expiry(expiry);

    // Find CE and PE at ATM strike from manifest instruments
    let ce_instr = instruments
        .iter()
        .find(|i| i.instrument_type == "CE" && (i.strike - atm_strike).abs() < 0.01)?;
    let pe_instr = instruments
        .iter()
        .find(|i| i.instrument_type == "PE" && (i.strike - atm_strike).abs() < 0.01)?;

    let ce_tick = find_closest_tick(symbol_ticks, &ce_instr.tradingsymbol, decision_ts)?;
    let pe_tick = find_closest_tick(symbol_ticks, &pe_instr.tradingsymbol, decision_ts)?;

    let ce_val = validate_q1_leg(ce_tick, &ce_instr.tradingsymbol, decision_ts);
    let pe_val = validate_q1_leg(pe_tick, &pe_instr.tradingsymbol, decision_ts);

    let any_failed = !ce_val.passed || !pe_val.passed;

    Some(Q1StraddleValidation {
        ce: ce_val,
        pe: pe_val,
        any_failed,
    })
}

fn find_closest_tick<'a>(
    symbol_ticks: &'a HashMap<String, Vec<TickEvent>>,
    symbol: &str,
    target_ts: DateTime<Utc>,
) -> Option<&'a TickEvent> {
    // Try exact symbol first
    if let Some(ticks) = symbol_ticks.get(symbol) {
        return ticks
            .iter()
            .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());
    }

    // Try case-insensitive match
    for (key, ticks) in symbol_ticks {
        if key.to_uppercase() == symbol.to_uppercase() {
            return ticks
                .iter()
                .min_by_key(|t| (t.ts - target_ts).num_milliseconds().abs());
        }
    }

    None
}

/// P3: Find index of strike nearest to target (using total_cmp to avoid panic on NaN)
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

/// Extract IV from SANOS slice at ATM
fn extract_atm_iv(slice: &SanosSlice) -> Option<f64> {
    // Find ATM strike (k ≈ 1.0)
    let atm_idx = find_nearest_strike_idx(&slice.fitted_strikes, 1.0)?;

    let call_price = slice.fitted_calls[atm_idx];
    let k = slice.fitted_strikes[atm_idx];
    let tty = slice.time_to_expiry;

    extract_iv(call_price, k, tty)
}

/// Extract implied volatility from normalized call price
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

/// Find ATM strike from SANOS slice forward
fn find_atm_strike(slice: &SanosSlice, underlying: &str) -> f64 {
    let forward = slice.forward;
    let tick_size = if underlying == "BANKNIFTY" {
        100.0
    } else {
        50.0
    };
    (forward / tick_size).round() * tick_size
}

/// Extract calendar gap from SANOS slices
fn extract_calendar_gap(s1: &SanosSlice, s2: &SanosSlice) -> Option<f64> {
    // Find ATM indices using P3-safe helper
    let atm1_idx = find_nearest_strike_idx(&s1.fitted_strikes, 1.0)?;
    let atm2_idx = find_nearest_strike_idx(&s2.fitted_strikes, 1.0)?;

    Some(s2.fitted_calls[atm2_idx] - s1.fitted_calls[atm1_idx])
}

/// Extract skew from SANOS slice
/// Skew = (iv_high - iv_low) / (k_high - k_low) where k_low ≈ 0.97, k_high ≈ 1.03
fn extract_skew(slice: &SanosSlice) -> Option<f64> {
    const K_LOW_TARGET: f64 = 0.97;
    const K_HIGH_TARGET: f64 = 1.03;

    // Find nearest strike to k_low
    let low_idx = slice
        .fitted_strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - K_LOW_TARGET)
                .abs()
                .partial_cmp(&(**b - K_LOW_TARGET).abs())
                .unwrap()
        })?
        .0;

    // Find nearest strike to k_high
    let high_idx = slice
        .fitted_strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - K_HIGH_TARGET)
                .abs()
                .partial_cmp(&(**b - K_HIGH_TARGET).abs())
                .unwrap()
        })?
        .0;

    let k_low = slice.fitted_strikes[low_idx];
    let k_high = slice.fitted_strikes[high_idx];
    let tty = slice.time_to_expiry;

    // Extract IVs at these strikes
    let iv_low = extract_iv(slice.fitted_calls[low_idx], k_low, tty)?;
    let iv_high = extract_iv(slice.fitted_calls[high_idx], k_high, tty)?;

    if (k_high - k_low).abs() < 1e-6 {
        return None;
    }

    Some((iv_high - iv_low) / (k_high - k_low))
}

/// Build Phase8Features from SANOS slices
fn build_features(slices: &[SanosSlice]) -> Result<Phase8Features> {
    if slices.is_empty() {
        return Err(anyhow!("No slices provided"));
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
        let cal23_val = if slices.len() >= 2 {
            extract_calendar_gap(&slices[1], s3)
        } else {
            None
        };
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::level_filters::LevelFilter::INFO.into()),
        )
        .init();

    let args = Args::parse();

    // ---------------------------------------------------------------------
    // Deterministic Run ID + Run Directory
    // ---------------------------------------------------------------------
    let cfg = RunConfigHash {
        underlying: args.underlying.clone(),
        interval_secs: args.interval_secs,
        eta: args.eta,
    };
    let cfg_sha256 = config_hash(&cfg)?;

    let session_manifest_path = args.session_dir.join("session_manifest.json");
    let session_sha256 = if session_manifest_path.exists() {
        let (sha, _) = hash_file(&session_manifest_path)?;
        sha
    } else {
        // Legacy fallback: bind to session_dir string
        sha256_hex(args.session_dir.to_string_lossy().as_bytes())
    };

    // Deterministic run ID derived from (session_sha256 + cfg_sha256)
    let run_id_full = sha256_hex(format!("{}{}", session_sha256, cfg_sha256).as_bytes());
    let run_id = run_id_full.chars().take(16).collect::<String>();
    let run_dir = args
        .session_dir
        .join("runs")
        .join("run_calendar_carry")
        .join(&run_id);
    std::fs::create_dir_all(&run_dir)
        .with_context(|| format!("Failed to create run_dir: {}", run_dir.display()))?;

    // Resolve audit output path (default: inside run_dir)
    let audit_path = if args.output.is_absolute() {
        args.output.clone()
    } else {
        run_dir.join(&args.output)
    };
    let signals_path = run_dir.join("signals.jsonl");
    let gates_path = run_dir.join("gate_counters.json");

    let mut audit_writer = BufWriter::new(
        File::create(&audit_path)
            .with_context(|| format!("Failed to create audit output: {}", audit_path.display()))?,
    );
    let mut signals_writer = BufWriter::new(File::create(&signals_path).with_context(|| {
        format!(
            "Failed to create signals output: {}",
            signals_path.display()
        )
    })?);

    info!("Strategy v0: SANOS-Gated Calendar Carry Runner");
    info!("Session: {:?}", args.session_dir);
    info!("Underlying: {}", args.underlying);
    info!("Decision interval: {}s", args.interval_secs);
    info!("Run ID: {}", run_id);
    info!("Run dir: {}", run_dir.display());

    // Try manifest-driven mode (Commit D)
    let manifest_inventory = try_load_sanos_inventory(&args.session_dir)?;
    let underlying_inv = manifest_inventory.as_ref().and_then(|inv| {
        inv.underlyings
            .iter()
            .find(|u| u.underlying.eq_ignore_ascii_case(&args.underlying))
    });

    // Mode context: manifest-driven vs legacy
    // In manifest mode, underlying_inv is Some and we use manifest-based functions
    // In legacy mode, underlying_inv is None and we use parse_symbol() based functions
    let (expiries_naive, all_ticks, manifest_underlying) =
        if let (Some(inv), Some(u_inv)) = (&manifest_inventory, underlying_inv) {
            log_manifest_mode(inv);

            let expiries = u_inv.get_sorted_expiries();
            info!(
                "Manifest-driven: {} expiries, universe_sha256={}",
                expiries.len(),
                u_inv.universe_sha256
            );

            let ticks = load_all_ticks_manifest(&inv.session_dir, u_inv)?;
            info!("Loaded ticks for {} symbols (manifest-driven)", ticks.len());

            (expiries, ticks, Some(u_inv))
        } else {
            log_legacy_mode(&args.session_dir);

            let expiries_str = discover_expiries(&args.session_dir, &args.underlying)?;
            info!("Found {} expiries: {:?}", expiries_str.len(), expiries_str);

            // Convert string expiries to NaiveDate
            let expiries: Vec<NaiveDate> = expiries_str
                .iter()
                .filter_map(|e| expiry_to_date(e))
                .collect();

            let ticks = load_all_ticks(&args.session_dir, &args.underlying)?;
            info!("Loaded ticks for {} symbols", ticks.len());

            (expiries, ticks, None)
        };

    if expiries_naive.len() < 2 {
        return Err(anyhow!(
            "Need at least 2 expiries for calendar strategy, found {}",
            expiries_naive.len()
        ));
    }

    // For legacy mode compatibility and display/logging, we keep string expiries
    // Named 'expiries' for compatibility with rest of the code
    let expiries: Vec<String> = expiries_naive
        .iter()
        .map(|d| {
            d.format("%d%b")
                .to_string()
                .to_uppercase()
                .replace(" ", "0")
        })
        .collect();

    // Find timestamp range
    let mut min_ts: Option<DateTime<Utc>> = None;
    let mut max_ts: Option<DateTime<Utc>> = None;

    for ticks in all_ticks.values() {
        for tick in ticks {
            min_ts = Some(min_ts.map_or(tick.ts, |m| m.min(tick.ts)));
            max_ts = Some(max_ts.map_or(tick.ts, |m| m.max(tick.ts)));
        }
    }

    let min_ts = min_ts.ok_or_else(|| anyhow!("No ticks found"))?;
    let max_ts = max_ts.ok_or_else(|| anyhow!("No ticks found"))?;

    info!("Time range: {} to {}", min_ts, max_ts);

    // Initialize strategy and counters
    let strategy = CalendarCarryStrategy::new();
    let calibrator = SanosCalibrator::with_eta(args.eta);
    let mut counters = GateCounters::default();
    let mut audit_records: Vec<AuditRecord> = Vec::new();

    // Distribution stats (Phase 9)
    let mut dist_cal = DistributionStats::default();
    let mut dist_cal_min = DistributionStats::default();
    let mut dist_edge = DistributionStats::default();
    let mut dist_spread_front = DistributionStats::default();
    let mut dist_spread_back = DistributionStats::default();
    let mut dist_iv_term = DistributionStats::default();

    // Phase 9 Completion: Economic gate stats
    let mut dist_gap_premium = DistributionStats::default();
    let mut dist_friction_round = DistributionStats::default();
    let mut dist_friction_ratio = DistributionStats::default();
    let mut dist_entry_friction = DistributionStats::default();
    let mut dist_pnl_at_entry = DistributionStats::default(); // PnL implied at entry (conservative)

    // Phase 9.2: Effective friction and 10-minute exit stats
    let mut dist_friction_round_obs = DistributionStats::default();
    let mut dist_friction_round_eff = DistributionStats::default();
    let mut dist_pnl_10m_conservative = DistributionStats::default();
    let mut dist_pnl_10m_mid = DistributionStats::default();
    let mut dist_pnl_per_friction = DistributionStats::default();

    // Phase 9.3: Q1-lite quote age distribution
    let mut dist_quote_age_ms = DistributionStats::default();

    // Cooldown tracking (Phase 9: S2)
    let mut last_entry_ts: Option<DateTime<Utc>> = None;

    // Conservative fill tracking
    let mut total_entries: u32 = 0;
    let mut total_entry_friction: f64 = 0.0;
    let mut total_pnl_at_entry: f64 = 0.0;

    // Phase 9.2: Active trade tracking for 10-minute exit
    let mut active_trades: Vec<ActiveTrade> = Vec::new();
    let mut completed_exits: u32 = 0;
    let mut total_pnl_10m: f64 = 0.0;
    let mut winning_exits: u32 = 0;

    // Iterate through decision timestamps
    let mut current_ts = min_ts;
    let interval = Duration::seconds(args.interval_secs as i64);

    while current_ts <= max_ts {
        // Check market hours - skip decisions outside trading window
        if !is_market_hours(current_ts) {
            counters.outside_market_hours += 1;
            current_ts += interval;
            continue;
        }
        counters.inside_market_hours += 1;

        // Calibrate slices for all expiries at this timestamp
        // Track successful (expiry_date, expiry_str, slice) tuples to maintain alignment
        let mut successful_slices: Vec<(NaiveDate, String, SanosSlice)> = Vec::new();

        for (idx, expiry_date) in expiries_naive.iter().enumerate() {
            // Use manifest-driven or legacy functions based on mode
            let (slice, expiry_str) = if let Some(u_inv) = manifest_underlying {
                // Manifest mode: no symbol parsing
                let tte = time_to_expiry_from_date(current_ts, *expiry_date);
                let s = build_slice_manifest(&all_ticks, u_inv, *expiry_date, current_ts, tte);
                (s, expiries[idx].clone())
            } else {
                // Legacy mode: uses parse_symbol()
                let expiry_str = &expiries[idx];
                let tte = time_to_expiry(current_ts, expiry_str);
                let s = build_slice(&all_ticks, &args.underlying, expiry_str, current_ts, tte);
                (s, expiry_str.clone())
            };

            if slice.calls.is_empty() || slice.puts.is_empty() {
                continue;
            }

            match calibrator.calibrate(&slice) {
                Ok(sanos_slice) => {
                    successful_slices.push((*expiry_date, expiry_str, sanos_slice));
                }
                Err(_e) => {
                    // Skip failed calibration
                }
            }
        }

        if successful_slices.len() < 2 {
            current_ts += interval;
            continue;
        }

        // Extract aligned slices for feature building
        let slices: Vec<SanosSlice> = successful_slices
            .iter()
            .map(|(_, _, s)| s.clone())
            .collect();

        // Build features
        let features = match build_features(&slices) {
            Ok(f) => f,
            Err(_) => {
                current_ts += interval;
                continue;
            }
        };

        // Extract aligned expiry info from successful_slices
        let (front_expiry_date, front_expiry_str, _) = &successful_slices[0];
        let (back_expiry_date, back_expiry_str, _) = &successful_slices[1];

        // Find ATM strikes for front and back
        let front_atm = find_atm_strike(&slices[0], &args.underlying);
        let back_atm = find_atm_strike(&slices[1], &args.underlying);

        // Build straddle quotes (manifest-driven or legacy)
        let (front_straddle, back_straddle, q1_front, q1_back) =
            if let Some(u_inv) = manifest_underlying {
                // Manifest mode: use instrument info (no symbol parsing)
                let front = match build_straddle_quotes_manifest(
                    &all_ticks,
                    u_inv,
                    *front_expiry_date,
                    front_atm,
                    current_ts,
                ) {
                    Some(s) => s,
                    None => {
                        current_ts += interval;
                        continue;
                    }
                };

                let back = match build_straddle_quotes_manifest(
                    &all_ticks,
                    u_inv,
                    *back_expiry_date,
                    back_atm,
                    current_ts,
                ) {
                    Some(s) => s,
                    None => {
                        current_ts += interval;
                        continue;
                    }
                };

                // Phase 9.3: Q1-lite quote validation (manifest mode)
                let q1_f = validate_q1_straddle_manifest(
                    &all_ticks,
                    u_inv,
                    *front_expiry_date,
                    front_atm,
                    current_ts,
                );
                let q1_b = validate_q1_straddle_manifest(
                    &all_ticks,
                    u_inv,
                    *back_expiry_date,
                    back_atm,
                    current_ts,
                );

                (front, back, q1_f, q1_b)
            } else {
                // Legacy mode: uses symbol construction
                let front = match build_straddle_quotes(
                    &all_ticks,
                    &args.underlying,
                    front_expiry_str,
                    front_atm,
                    current_ts,
                ) {
                    Some(s) => s,
                    None => {
                        current_ts += interval;
                        continue;
                    }
                };

                let back = match build_straddle_quotes(
                    &all_ticks,
                    &args.underlying,
                    back_expiry_str,
                    back_atm,
                    current_ts,
                ) {
                    Some(s) => s,
                    None => {
                        current_ts += interval;
                        continue;
                    }
                };

                // Phase 9.3: Q1-lite quote validation (legacy mode)
                let q1_f = validate_q1_straddle(
                    &all_ticks,
                    &args.underlying,
                    front_expiry_str,
                    front_atm,
                    current_ts,
                );
                let q1_b = validate_q1_straddle(
                    &all_ticks,
                    &args.underlying,
                    back_expiry_str,
                    back_atm,
                    current_ts,
                );

                (front, back, q1_f, q1_b)
            };

        // Track Q1 stats if validation succeeded
        if let (Some(front_q1), Some(back_q1)) = (&q1_front, &q1_back) {
            update_q1_counters(&mut counters, front_q1, back_q1);
            // Track quote age distribution (all 4 legs)
            dist_quote_age_ms.add(front_q1.ce.quote_age_ms as f64);
            dist_quote_age_ms.add(front_q1.pe.quote_age_ms as f64);
            dist_quote_age_ms.add(back_q1.ce.quote_age_ms as f64);
            dist_quote_age_ms.add(back_q1.pe.quote_age_ms as f64);
        }

        // Build session meta (use aligned expiry strings from successful_slices)
        let meta = SessionMeta {
            underlying: args.underlying.clone(),
            t1_expiry: front_expiry_str.clone(),
            t2_expiry: Some(back_expiry_str.clone()),
            t3_expiry: successful_slices.get(2).map(|(_, s, _)| s.clone()),
            lot_size: if args.underlying == "BANKNIFTY" {
                15
            } else {
                25
            },
            multiplier: 1.0,
            lp_status_t1: slices[0].diagnostics.lp_status.clone(),
            lp_status_t2: slices.get(1).map(|s| s.diagnostics.lp_status.clone()),
            lp_status_t3: slices.get(2).map(|s| s.diagnostics.lp_status.clone()),
        };

        // Check if it's expiry day for front (use aligned expiry date)
        let is_expiry_day_front = *front_expiry_date == current_ts.date_naive();

        // Build context
        let ctx = StrategyContext {
            ts: current_ts,
            features,
            front_straddle,
            back_straddle,
            meta,
            minutes_to_close: minutes_to_close(current_ts),
            is_expiry_day_front,
        };

        // U2: Log ATM strike coherence (Phase 9)
        let k_atm_t1 = front_atm;
        let k_atm_t2 = back_atm;
        let k_norm_t1 = front_atm / slices[0].forward;
        let k_norm_t2 = back_atm / slices[1].forward;

        // Phase 9.2: Check for 10-minute exits on active trades
        let exit_threshold_secs = 600; // 10 minutes
        let mut trades_to_remove = Vec::new();

        for (idx, trade) in active_trades.iter().enumerate() {
            let hold_secs = (current_ts - trade.entry_ts).num_seconds();
            if hold_secs >= exit_threshold_secs {
                // Compute exit PnL with current quotes (from ctx)
                let exit_result =
                    compute_exit_pnl(trade, &ctx.front_straddle, &ctx.back_straddle, current_ts);

                // Track stats
                completed_exits += 1;
                total_pnl_10m += exit_result.pnl_conservative;
                if exit_result.pnl_conservative > 0.0 {
                    winning_exits += 1;
                }
                dist_pnl_10m_conservative.add(exit_result.pnl_conservative);
                dist_pnl_10m_mid.add(exit_result.pnl_mid);
                dist_pnl_per_friction.add(exit_result.pnl_per_friction);

                info!(
                    "[EXIT] entry_ts={} exit_ts={} hold={}s pnl_cons={:.2}₹ pnl_mid={:.2}₹ pnl/fric={:.2}",
                    trade.entry_ts,
                    current_ts,
                    exit_result.hold_seconds,
                    exit_result.pnl_conservative,
                    exit_result.pnl_mid,
                    exit_result.pnl_per_friction
                );

                trades_to_remove.push(idx);
            }
        }

        // Remove completed trades (in reverse order to preserve indices)
        for idx in trades_to_remove.into_iter().rev() {
            active_trades.remove(idx);
        }

        // Evaluate strategy
        let (decision, mut audit) = strategy.evaluate(&ctx);

        // Collect distribution stats from carry gate
        if let Some(cal_price) = audit.gates.as_ref().and_then(|g| g.carry.value) {
            dist_cal.add(cal_price);
        }
        if let Some(cal_min_price) = audit.gates.as_ref().and_then(|g| g.carry.threshold) {
            dist_cal_min.add(cal_min_price);
        }
        if let (Some(cal), Some(cal_min)) = (
            audit.gates.as_ref().and_then(|g| g.carry.value),
            audit.gates.as_ref().and_then(|g| g.carry.threshold),
        ) {
            dist_edge.add(cal - cal_min);
        }

        // Collect E1/E2 economic gate stats
        if let Some(gap_premium) = audit.gates.as_ref().and_then(|g| g.e1_premium_gap.value) {
            dist_gap_premium.add(gap_premium);
        }
        if let Some(ratio) = audit
            .gates
            .as_ref()
            .and_then(|g| g.e2_friction_dominance.value)
        {
            dist_friction_ratio.add(ratio);
        }

        // Collect spread stats
        dist_spread_front.add(ctx.front_straddle.spread());
        dist_spread_back.add(ctx.back_straddle.spread());

        // Collect IV term structure diff
        if let Some(iv3) = ctx.features.iv3 {
            dist_iv_term.add(ctx.features.iv1 - iv3);
        } else if let Some(iv2) = ctx.features.iv2 {
            dist_iv_term.add(ctx.features.iv1 - iv2);
        }

        // Build quote validation for every decision (for debugging)
        let quote_val = build_quote_validation(&ctx.front_straddle, &ctx.back_straddle);

        // Update counters based on decision
        match &decision {
            StrategyDecision::NoTrade { gates, .. } => {
                counters.update(gates);
            }
            StrategyDecision::Enter { gates, intent } => {
                counters.update(gates);

                // Check cooldown (Phase 9: S2)
                let in_cooldown = last_entry_ts
                    .is_some_and(|last| (current_ts - last).num_seconds() < COOLDOWN_SECS);

                if in_cooldown {
                    counters.cooldown_blocked += 1;
                    // Override to NO_TRADE in audit
                    audit.decision = "COOLDOWN_BLOCKED".to_string();
                    audit.reason_code = Some(format!(
                        "COOLDOWN|last_entry={}s_ago",
                        last_entry_ts.map_or(0, |t| (current_ts - t).num_seconds())
                    ));
                } else {
                    counters.enter_after_cooldown += 1;
                    last_entry_ts = Some(current_ts);

                    // Emit alpha signal (Enter intent) - deterministic JSONL
                    // Use ISO dates for expiry (scoring expects "%Y-%m-%d" format)
                    let signal = SignalEvent {
                        schema_version: 1,
                        ts: current_ts,
                        underlying: intent.underlying.clone(),
                        front_expiry: front_expiry_date.to_string(), // ISO "2026-01-26"
                        back_expiry: back_expiry_date.to_string(),   // ISO "2026-02-26"
                        front_strike: intent.front_strike,
                        back_strike: intent.back_strike,
                        front_lots: intent.front_lots,
                        back_lots: intent.back_lots,
                        hedge_ratio: intent.hedge_ratio,
                        cal_value: intent.cal_value,
                        cal_min: intent.cal_min,
                        friction_estimate: intent.friction_estimate,
                        reason_codes: vec!["ENTER".to_string()],
                    };
                    let line = serde_json::to_string(&signal)
                        .context("Failed to serialize SignalEvent")?;
                    writeln!(signals_writer, "{}", line).context("Failed to write SignalEvent")?;

                    // Phase 9 Completion: Conservative fill calculation
                    let fill = compute_conservative_fill(
                        &ctx.front_straddle,
                        &ctx.back_straddle,
                        intent.hedge_ratio,
                    );

                    // Track fill statistics
                    total_entries += 1;
                    total_entry_friction += fill.entry_friction;
                    total_pnl_at_entry += -fill.net_entry_cost; // Negative cost = positive PnL

                    // Compute observed and effective round-trip friction (Phase 9.2)
                    let friction_round_obs = 2.0 * fill.entry_friction;
                    let is_nifty = ctx.meta.underlying == "NIFTY";
                    let friction_floor = if is_nifty { 10.0 } else { 25.0 }; // FROZEN_PARAMS.floor_friction_round_*
                    let friction_round_eff = friction_round_obs.max(friction_floor);

                    dist_friction_round.add(friction_round_obs);
                    dist_friction_round_obs.add(friction_round_obs);
                    dist_friction_round_eff.add(friction_round_eff);
                    dist_entry_friction.add(fill.entry_friction);
                    dist_pnl_at_entry.add(-fill.net_entry_cost);

                    // Phase 9.2: Create ActiveTrade for 10-minute exit tracking
                    let active_trade = ActiveTrade {
                        entry_ts: current_ts,
                        front_straddle_entry: ctx.front_straddle.clone(),
                        back_straddle_entry: ctx.back_straddle.clone(),
                        hedge_ratio: intent.hedge_ratio,
                        entry_fill: fill.clone(),
                        friction_round_eff,
                    };
                    active_trades.push(active_trade);

                    // Log ATM strike coherence + conservative fill for actual entries
                    info!(
                        "[ENTRY] ts={} K_atm_T1={:.0} K_atm_T2={:.0} k_norm_T1={:.4} k_norm_T2={:.4} h={:.3}",
                        current_ts, k_atm_t1, k_atm_t2, k_norm_t1, k_norm_t2, intent.hedge_ratio
                    );
                    info!(
                        "[FILL] front_credit={:.2} back_debit={:.2} net_cost={:.2} fric_entry={:.2} mark_mid={:.2} fric_eff={:.2}",
                        fill.entry_front_credit,
                        fill.entry_back_debit,
                        fill.net_entry_cost,
                        fill.entry_friction,
                        fill.mark_to_mid,
                        friction_round_eff
                    );
                    info!(
                        "[QUOTES] ce_f={:.2}/{:.2} pe_f={:.2}/{:.2} ce_b={:.2}/{:.2} pe_b={:.2}/{:.2} sprd_f={:.2} sprd_b={:.2}",
                        quote_val.ce_front_bid,
                        quote_val.ce_front_ask,
                        quote_val.pe_front_bid,
                        quote_val.pe_front_ask,
                        quote_val.ce_back_bid,
                        quote_val.ce_back_ask,
                        quote_val.pe_back_bid,
                        quote_val.pe_back_ask,
                        quote_val.straddle_spread_front,
                        quote_val.straddle_spread_back
                    );
                }
            }
            _ => {}
        }

        audit_records.push(audit);

        current_ts += interval;
    }

    // Print gate hit report
    counters.print_report();

    // Print distribution stats (Phase 9)
    info!("=== Distribution Stats (price units) ===");
    info!(
        "cal (price):     p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_cal.p50(),
        dist_cal.p90(),
        dist_cal.p99(),
        dist_cal.count()
    );
    info!(
        "cal_min (price): p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_cal_min.p50(),
        dist_cal_min.p90(),
        dist_cal_min.p99(),
        dist_cal_min.count()
    );
    info!(
        "edge (cal-cal_min): p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_edge.p50(),
        dist_edge.p90(),
        dist_edge.p99(),
        dist_edge.count()
    );
    info!(
        "spread_front:    p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_spread_front.p50(),
        dist_spread_front.p90(),
        dist_spread_front.p99(),
        dist_spread_front.count()
    );
    info!(
        "spread_back:     p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_spread_back.p50(),
        dist_spread_back.p90(),
        dist_spread_back.p99(),
        dist_spread_back.count()
    );
    info!(
        "iv1-iv(back):    p50={:.4} p90={:.4} p99={:.4} n={}",
        dist_iv_term.p50(),
        dist_iv_term.p90(),
        dist_iv_term.p99(),
        dist_iv_term.count()
    );

    // Phase 9 Completion: Economic gate stats
    info!("=== Economic Gate Stats (Phase 9) ===");
    info!(
        "gap_premium:     p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_gap_premium.p50(),
        dist_gap_premium.p90(),
        dist_gap_premium.p99(),
        dist_gap_premium.count()
    );
    info!(
        "friction_ratio:  p50={:.2} p90={:.2} p99={:.2} n={}",
        dist_friction_ratio.p50(),
        dist_friction_ratio.p90(),
        dist_friction_ratio.p99(),
        dist_friction_ratio.count()
    );

    // Conservative fill summary
    info!("=== Conservative Fill Summary (entries only) ===");
    info!("Total entries:   {}", total_entries);
    if total_entries > 0 {
        let avg_entry_friction = total_entry_friction / total_entries as f64;
        let avg_friction_round = 2.0 * avg_entry_friction;
        let avg_pnl_at_entry = total_pnl_at_entry / total_entries as f64;
        info!("avg entry_friction:  {:.2} ₹", avg_entry_friction);
        info!("avg friction_round:  {:.2} ₹", avg_friction_round);
        info!(
            "avg pnl_at_entry:    {:.2} ₹ (negative = net debit)",
            avg_pnl_at_entry
        );
        info!("total pnl_at_entry:  {:.2} ₹", total_pnl_at_entry);

        // Distribution of per-entry values
        info!(
            "entry_friction:  p50={:.2} p90={:.2} n={}",
            dist_entry_friction.p50(),
            dist_entry_friction.p90(),
            dist_entry_friction.count()
        );
        info!(
            "friction_round:  p50={:.2} p90={:.2} n={}",
            dist_friction_round.p50(),
            dist_friction_round.p90(),
            dist_friction_round.count()
        );
        info!(
            "pnl_at_entry:    p50={:.2} p90={:.2} n={}",
            dist_pnl_at_entry.p50(),
            dist_pnl_at_entry.p90(),
            dist_pnl_at_entry.count()
        );

        // Key ratio: median gap_premium / median friction_round
        let median_gap_premium = dist_gap_premium.p50();
        let median_friction_round = dist_friction_round.p50();
        if median_friction_round > 0.0 {
            info!(
                "median gap/friction_round: {:.2}",
                median_gap_premium / median_friction_round
            );
        }
    }

    // Phase 9.2: 10-minute exit PnL summary
    info!("=== Phase 9.2: 10-Minute Exit PnL Summary ===");
    info!("Completed exits:     {}", completed_exits);
    info!("Remaining active:    {}", active_trades.len());
    if completed_exits > 0 {
        let win_rate = 100.0 * winning_exits as f64 / completed_exits as f64;
        let avg_pnl_10m = total_pnl_10m / completed_exits as f64;
        info!("Total PnL (10m):     {:.2} ₹", total_pnl_10m);
        info!("Avg PnL per trade:   {:.2} ₹", avg_pnl_10m);
        info!("Winning trades:      {} ({:.1}%)", winning_exits, win_rate);

        // Friction distributions (observed vs effective)
        info!(
            "friction_round_obs:  p50={:.2} p90={:.2} n={}",
            dist_friction_round_obs.p50(),
            dist_friction_round_obs.p90(),
            dist_friction_round_obs.count()
        );
        info!(
            "friction_round_eff:  p50={:.2} p90={:.2} n={}",
            dist_friction_round_eff.p50(),
            dist_friction_round_eff.p90(),
            dist_friction_round_eff.count()
        );

        // PnL distributions
        info!(
            "pnl_10m_cons:        p50={:.2} p90={:.2} n={}",
            dist_pnl_10m_conservative.p50(),
            dist_pnl_10m_conservative.p90(),
            dist_pnl_10m_conservative.count()
        );
        info!(
            "pnl_10m_mid:         p50={:.2} p90={:.2} n={}",
            dist_pnl_10m_mid.p50(),
            dist_pnl_10m_mid.p90(),
            dist_pnl_10m_mid.count()
        );
        info!(
            "pnl_per_friction:    p50={:.2} p90={:.2} n={}",
            dist_pnl_per_friction.p50(),
            dist_pnl_per_friction.p90(),
            dist_pnl_per_friction.count()
        );

        // Key metric: PnL per effective friction
        let median_pnl_per_fric = dist_pnl_per_friction.p50();
        info!(
            "Median PnL/friction: {:.2} (>1 = profitable after friction)",
            median_pnl_per_fric
        );
    }

    // Phase 9.3: Q1-lite quote audit summary
    info!("=== Phase 9.3: Q1-lite Quote Audit ===");
    info!(
        "quote_age_ms:        p50={:.0} p90={:.0} p99={:.0} n={}",
        dist_quote_age_ms.p50(),
        dist_quote_age_ms.p90(),
        dist_quote_age_ms.p99(),
        dist_quote_age_ms.count()
    );
    let total_q1_failures = counters.q1_front_fail + counters.q1_back_fail;
    let q1_fail_pct = if counters.total_decisions > 0 {
        100.0 * (total_q1_failures as f64) / (counters.total_decisions as f64 * 2.0) // 2 straddles per decision
    } else {
        0.0
    };
    info!("Q1 failure rate:     {:.1}% (target <5-10%)", q1_fail_pct);

    // ---------------------------------------------------------------------
    // Persist outputs (audit + gates + signals) and write RunManifest
    // ---------------------------------------------------------------------
    for record in &audit_records {
        let json = serde_json::to_string(record).context("Failed to serialize AuditRecord")?;
        writeln!(audit_writer, "{}", json).context("Failed to write audit JSONL")?;
    }
    audit_writer
        .flush()
        .context("Failed to flush audit output")?;
    signals_writer
        .flush()
        .context("Failed to flush signals output")?;

    // Gate counters JSON (small, deterministic)
    let gates_bytes =
        serde_json::to_vec_pretty(&counters).context("Failed to serialize gate counters")?;
    write_atomic(&gates_path, &gates_bytes)
        .with_context(|| format!("Failed to write gate counters: {}", gates_path.display()))?;

    info!(
        "Wrote audit records: {} -> {}",
        audit_records.len(),
        audit_path.display()
    );
    info!("Wrote signals -> {}", signals_path.display());
    info!("Wrote gate counters -> {}", gates_path.display());

    // Build RunManifest bindings
    let git_commit = git_commit_string();

    // Session manifest binding (if present)
    let session_binding = InputBinding {
        label: "session_manifest".to_string(),
        rel_path: "session_manifest.json".to_string(),
        sha256: session_sha256.clone(),
    };

    // Universe manifest binding for the requested underlying (manifest mode) if available
    let mut universe_bindings: Vec<InputBinding> = Vec::new();
    if let Some(u_inv) = manifest_underlying {
        let sub = u_inv.underlying_subdir.trim_end_matches('/');
        let rel = format!("{}/universe_manifest.json", sub);
        let (sha, _) = hash_file(&args.session_dir.join(&rel))?;
        universe_bindings.push(InputBinding {
            label: format!("universe_manifest:{}", u_inv.underlying.to_uppercase()),
            rel_path: rel,
            sha256: sha,
        });
    }

    // Output bindings (relative to run_dir)
    let (audit_sha, audit_len) = hash_file(&audit_path)?;
    let (signals_sha, signals_len) = hash_file(&signals_path)?;
    let (gates_sha, gates_len) = hash_file(&gates_path)?;

    let outputs = vec![
        OutputBinding {
            label: "audit_jsonl".to_string(),
            rel_path: audit_path
                .strip_prefix(&run_dir)
                .unwrap_or(&audit_path)
                .to_string_lossy()
                .to_string(),
            sha256: audit_sha,
            bytes_len: audit_len,
        },
        OutputBinding {
            label: "signals_jsonl".to_string(),
            rel_path: "signals.jsonl".to_string(),
            sha256: signals_sha,
            bytes_len: signals_len,
        },
        OutputBinding {
            label: "gate_counters".to_string(),
            rel_path: "gate_counters.json".to_string(),
            sha256: gates_sha,
            bytes_len: gates_len,
        },
    ];

    let rm = RunManifest {
        schema_version: 1,
        binary_name: "run_calendar_carry".to_string(),
        git_commit,
        run_id: run_id.clone(),
        session_dir: args.session_dir.to_string_lossy().to_string(),
        session_manifest: session_binding,
        universe_manifests: universe_bindings,
        config_sha256: cfg_sha256,
        outputs,
    };

    let run_manifest_sha = persist_run_manifest_atomic(&run_dir, &rm)?;
    info!(
        "RunManifest written -> {}/run_manifest.json (sha256={})",
        run_dir.display(),
        run_manifest_sha
    );

    Ok(())
}
