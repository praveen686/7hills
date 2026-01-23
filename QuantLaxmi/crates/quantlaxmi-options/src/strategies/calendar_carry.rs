//! # Strategy v0: SANOS-Gated Calendar Carry with Skew Regime Filter
//!
//! Implementation contract per Lead specification (2026-01-23).
//!
//! ## Thesis
//! Trade short-vs-long expiry variance carry only when:
//! - SANOS surface says term structure is stable and monotone
//! - Calendar gap at ATM is sufficiently large relative to spreads
//! - Skew regime is not indicating tail stress
//!
//! This is a relative value trade. It does not require predicting direction.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// FROZEN PARAMETERS (Strategy v0 — 2026-01-23)
// Do not modify without Lead approval.
// ============================================================================

/// Frozen strategy parameters
pub struct FrozenParams {
    /// Decision interval in seconds
    pub decision_interval_secs: u64,
    /// Lambda for carry gate (CAL_min_rel = λ × avg_spread)
    pub lambda: f64,
    /// Absolute edge floor in rupees (Phase 9 invariant)
    /// Enter only if CAL >= max(CAL_min_rel, edge_abs_rupees)
    pub edge_abs_rupees: f64,
    /// Risk budget in bps of forward notional
    pub risk_bps: f64,
    /// Maximum lots per trade
    pub max_lots: i32,
    /// Minimum vega hedge ratio
    pub h_min: f64,
    /// Maximum vega hedge ratio
    pub h_max: f64,
    /// NIFTY front leg max spread (bps)
    pub nifty_spread_ceiling_front_bps: f64,
    /// NIFTY back leg max spread (bps)
    pub nifty_spread_ceiling_back_bps: f64,
    /// BANKNIFTY front leg max spread (bps)
    pub banknifty_spread_ceiling_front_bps: f64,
    /// BANKNIFTY back leg max spread (bps)
    pub banknifty_spread_ceiling_back_bps: f64,
    /// Max term structure inversion for NIFTY (iv1 - iv3)
    pub nifty_ts_inversion_max: f64,
    /// Max term structure inversion for BANKNIFTY
    pub banknifty_ts_inversion_max: f64,
    /// Minimum skew value (regime filter)
    pub skew_stress_min: f64,
    /// Calendar monotonicity tolerance
    pub tol_cal: f64,
    /// Take profit multiplier (× friction)
    pub take_profit_mult: f64,
    /// Stop loss multiplier (× friction)
    pub stop_loss_mult: f64,
    /// Minutes before close to exit
    pub exit_minutes_before_close: u64,
    /// Quote staleness limit in seconds
    pub quote_staleness_secs: u64,
    // Phase 9 Completion: Economic hardeners
    /// E1: Minimum premium-based gap for NIFTY (₹)
    pub gap_abs_nifty: f64,
    /// E1: Minimum premium-based gap for BANKNIFTY (₹)
    pub gap_abs_banknifty: f64,
    /// E2: Friction dominance multiplier (gap_premium >= μ × friction_round)
    pub mu_friction: f64,
    // Phase 9.2: Friction floor (conservative safety bound)
    /// E3: Minimum realistic round-trip friction for NIFTY (₹)
    pub floor_friction_round_nifty: f64,
    /// E3: Minimum realistic round-trip friction for BANKNIFTY (₹)
    pub floor_friction_round_banknifty: f64,
}

pub const FROZEN_PARAMS: FrozenParams = FrozenParams {
    decision_interval_secs: 60,
    lambda: 1.5,
    edge_abs_rupees: 8.0, // Phase 9: absolute edge floor (Lead approved)
    risk_bps: 7.5,
    max_lots: 2,
    h_min: 0.5,
    h_max: 2.0,
    nifty_spread_ceiling_front_bps: 35.0,
    nifty_spread_ceiling_back_bps: 35.0,
    banknifty_spread_ceiling_front_bps: 55.0,
    banknifty_spread_ceiling_back_bps: 80.0,
    nifty_ts_inversion_max: 0.04,
    banknifty_ts_inversion_max: 0.05,
    skew_stress_min: -0.80,
    tol_cal: 1e-8,
    take_profit_mult: 1.0,
    stop_loss_mult: 2.0,
    exit_minutes_before_close: 15,
    quote_staleness_secs: 120,
    // Phase 9 Completion: Economic hardeners (Lead approved 2026-01-23)
    gap_abs_nifty: 12.0,     // E1: minimum premium gap NIFTY (₹)
    gap_abs_banknifty: 25.0, // E1: minimum premium gap BANKNIFTY (₹)
    mu_friction: 6.0,        // E2: gap_premium >= μ × friction_round
    // Phase 9.2: Friction floor (conservative safety bound)
    floor_friction_round_nifty: 10.0, // E3: minimum realistic friction NIFTY (₹)
    floor_friction_round_banknifty: 25.0, // E3: minimum realistic friction BANKNIFTY (₹)
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Quote snapshot for a single option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteSnapshot {
    pub bid: f64,
    pub ask: f64,
    pub last_ts: DateTime<Utc>,
}

impl QuoteSnapshot {
    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }

    pub fn is_valid(&self) -> bool {
        self.bid > 0.0 && self.ask > self.bid
    }

    pub fn staleness_secs(&self, now: DateTime<Utc>) -> i64 {
        (now - self.last_ts).num_seconds()
    }
}

/// Straddle quotes (CE + PE at same strike)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StraddleQuotes {
    pub expiry: String,
    pub strike: f64,
    pub ce: QuoteSnapshot,
    pub pe: QuoteSnapshot,
}

impl StraddleQuotes {
    pub fn mid(&self) -> f64 {
        self.ce.mid() + self.pe.mid()
    }

    pub fn spread(&self) -> f64 {
        (self.ce.ask + self.pe.ask) - (self.ce.bid + self.pe.bid)
    }

    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid();
        if mid <= 1e-9 {
            return f64::MAX;
        }
        1e4 * self.spread() / mid
    }

    pub fn is_valid(&self) -> bool {
        self.ce.is_valid() && self.pe.is_valid()
    }

    pub fn max_staleness_secs(&self, now: DateTime<Utc>) -> i64 {
        self.ce.staleness_secs(now).max(self.pe.staleness_secs(now))
    }
}

/// Phase 8 features for strategy context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase8Features {
    pub iv1: f64,
    pub iv2: Option<f64>,
    pub iv3: Option<f64>,
    pub cal12: Option<f64>,
    pub cal23: Option<f64>,
    pub ts12: Option<f64>,
    pub ts23: Option<f64>,
    pub ts_curv: Option<f64>,
    pub sk1: Option<f64>,
    pub sk2: Option<f64>,
    pub sk3: Option<f64>,
    pub f1: f64,
    pub f2: Option<f64>,
    pub f3: Option<f64>,
    pub tty1: f64,
    pub tty2: Option<f64>,
    pub tty3: Option<f64>,
    pub k_atm1: f64,
    pub k_atm2: Option<f64>,
    pub k_atm3: Option<f64>,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub underlying: String,
    pub t1_expiry: String,
    pub t2_expiry: Option<String>,
    pub t3_expiry: Option<String>,
    pub lot_size: u32,
    pub multiplier: f64,
    pub lp_status_t1: String,
    pub lp_status_t2: Option<String>,
    pub lp_status_t3: Option<String>,
}

/// Full context for strategy decision
#[derive(Debug, Clone)]
pub struct StrategyContext {
    pub ts: DateTime<Utc>,
    pub features: Phase8Features,
    pub front_straddle: StraddleQuotes,
    pub back_straddle: StraddleQuotes,
    pub meta: SessionMeta,
    pub minutes_to_close: u64,
    pub is_expiry_day_front: bool,
}

// ============================================================================
// GATE RESULTS
// ============================================================================

/// Result of a single gate check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub name: String,
    pub passed: bool,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
    pub reason: Option<String>,
}

impl GateResult {
    pub fn pass(name: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            value: None,
            threshold: None,
            reason: None,
        }
    }

    pub fn pass_with_value(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            value: Some(value),
            threshold: None,
            reason: None,
        }
    }

    pub fn fail(name: &str, reason: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            value: None,
            threshold: None,
            reason: Some(reason.to_string()),
        }
    }

    pub fn fail_with_values(name: &str, reason: &str, value: f64, threshold: f64) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            value: Some(value),
            threshold: Some(threshold),
            reason: Some(reason.to_string()),
        }
    }
}

/// Aggregated gate check results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCheckResult {
    pub h1_surface: GateResult,
    pub h2_calendar: GateResult,
    pub h3_quote_front: GateResult,
    pub h3_quote_back: GateResult,
    pub h4_liquidity_front: GateResult,
    pub h4_liquidity_back: GateResult,
    pub carry: GateResult,
    pub r1_inversion: GateResult,
    pub r2_skew: GateResult,
    // Phase 9 Completion: Economic hardeners
    pub e1_premium_gap: GateResult,
    pub e2_friction_dominance: GateResult,
    // Phase 9.2: Friction floor
    pub e3_friction_floor: GateResult,
}

impl GateCheckResult {
    pub fn all_passed(&self) -> bool {
        self.h1_surface.passed
            && self.h2_calendar.passed
            && self.h3_quote_front.passed
            && self.h3_quote_back.passed
            && self.h4_liquidity_front.passed
            && self.h4_liquidity_back.passed
            && self.carry.passed
            && self.r1_inversion.passed
            && self.r2_skew.passed
            && self.e1_premium_gap.passed
            && self.e2_friction_dominance.passed
            && self.e3_friction_floor.passed
    }

    pub fn first_failure_reason(&self) -> Option<String> {
        if !self.h1_surface.passed {
            return self.h1_surface.reason.clone();
        }
        if !self.h2_calendar.passed {
            return self.h2_calendar.reason.clone();
        }
        if !self.h3_quote_front.passed {
            return self.h3_quote_front.reason.clone();
        }
        if !self.h3_quote_back.passed {
            return self.h3_quote_back.reason.clone();
        }
        if !self.h4_liquidity_front.passed {
            return self.h4_liquidity_front.reason.clone();
        }
        if !self.h4_liquidity_back.passed {
            return self.h4_liquidity_back.reason.clone();
        }
        if !self.carry.passed {
            return self.carry.reason.clone();
        }
        if !self.r1_inversion.passed {
            return self.r1_inversion.reason.clone();
        }
        if !self.r2_skew.passed {
            return self.r2_skew.reason.clone();
        }
        if !self.e1_premium_gap.passed {
            return self.e1_premium_gap.reason.clone();
        }
        if !self.e2_friction_dominance.passed {
            return self.e2_friction_dominance.reason.clone();
        }
        if !self.e3_friction_floor.passed {
            return self.e3_friction_floor.reason.clone();
        }
        None
    }
}

// ============================================================================
// STRATEGY DECISION
// ============================================================================

/// Trade intent produced by strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterIntent {
    pub underlying: String,
    pub front_expiry: String,
    pub back_expiry: String,
    pub front_strike: f64,
    pub back_strike: f64,
    pub front_lots: i32, // negative = short
    pub back_lots: i32,  // positive = long
    pub hedge_ratio: f64,
    pub h_clamped: bool,
    pub cal_value: f64,
    pub cal_min: f64,
    pub friction_estimate: f64,
}

/// Exit intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitIntent {
    pub reason: String,
    pub pnl_bps: Option<f64>,
}

/// Strategy decision output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyDecision {
    NoTrade {
        reason: String,
        gates: GateCheckResult,
    },
    Enter {
        intent: EnterIntent,
        gates: GateCheckResult,
    },
    Exit {
        intent: ExitIntent,
    },
    Hold,
}

// ============================================================================
// AUDIT RECORD
// ============================================================================

/// Complete audit record for each decision tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub ts: DateTime<Utc>,
    pub underlying: String,
    pub decision: String,
    pub reason_code: Option<String>,
    pub gates: Option<GateCheckResult>,
    pub front_expiry: Option<String>,
    pub back_expiry: Option<String>,
    pub front_spread_bps: Option<f64>,
    pub back_spread_bps: Option<f64>,
    pub iv1: f64,
    pub iv_back: Option<f64>,
    pub cal_value: Option<f64>,
    pub cal_min: Option<f64>,
    pub sk_min: Option<f64>,
    pub hedge_ratio: Option<f64>,
    pub lots: Option<i32>,
    /// Whether hedge ratio was clamped to bounds
    pub h_clamped: Option<bool>,
}

// ============================================================================
// STRATEGY IMPLEMENTATION
// ============================================================================

pub struct CalendarCarryStrategy {
    params: &'static FrozenParams,
}

impl Default for CalendarCarryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl CalendarCarryStrategy {
    pub fn new() -> Self {
        Self {
            params: &FROZEN_PARAMS,
        }
    }

    /// Main evaluation function — deterministic decision per tick
    pub fn evaluate(&self, ctx: &StrategyContext) -> (StrategyDecision, AuditRecord) {
        let gates = self.check_all_gates(ctx);
        let audit_base = self.build_audit_base(ctx, &gates);

        if !gates.all_passed() {
            let reason = gates
                .first_failure_reason()
                .unwrap_or_else(|| "UNKNOWN_GATE_FAILURE".to_string());
            let mut audit = audit_base;
            audit.decision = "NO_TRADE".to_string();
            audit.reason_code = Some(reason.clone());
            return (StrategyDecision::NoTrade { reason, gates }, audit);
        }

        // All gates passed — construct trade
        let (intent, h_clamped) = self.construct_trade(ctx);

        let mut audit = audit_base;
        audit.decision = "ENTER".to_string();
        audit.hedge_ratio = Some(intent.hedge_ratio);
        audit.lots = Some(intent.back_lots);
        audit.cal_value = Some(intent.cal_value);
        audit.cal_min = Some(intent.cal_min);
        audit.h_clamped = Some(h_clamped);

        (StrategyDecision::Enter { intent, gates }, audit)
    }

    /// Check all gates in sequence
    fn check_all_gates(&self, ctx: &StrategyContext) -> GateCheckResult {
        let is_nifty = ctx.meta.underlying == "NIFTY";

        // Determine which expiry pair we're using
        let using_t2 = ctx.meta.t2_expiry.is_some() && ctx.features.iv2.is_some();

        // Compute hedge ratio early (needed for E1/E2)
        let h = self.compute_hedge_ratio(ctx, using_t2);

        GateCheckResult {
            h1_surface: self.check_h1_surface(ctx, using_t2),
            h2_calendar: self.check_h2_calendar(ctx, using_t2),
            h3_quote_front: self.check_h3_quote(&ctx.front_straddle, ctx.ts, "FRONT"),
            h3_quote_back: self.check_h3_quote(&ctx.back_straddle, ctx.ts, "BACK"),
            h4_liquidity_front: self.check_h4_liquidity(&ctx.front_straddle, is_nifty, true),
            h4_liquidity_back: self.check_h4_liquidity(&ctx.back_straddle, is_nifty, false),
            carry: self.check_carry_gate(ctx, using_t2),
            r1_inversion: self.check_r1_inversion(ctx, is_nifty, using_t2),
            r2_skew: self.check_r2_skew(ctx),
            e1_premium_gap: self.check_e1_premium_gap(ctx, h, is_nifty),
            e2_friction_dominance: self.check_e2_friction_dominance(ctx, h),
            e3_friction_floor: self.check_e3_friction_floor(ctx, h, is_nifty),
        }
    }

    /// H1: Surface availability
    fn check_h1_surface(&self, ctx: &StrategyContext, using_t2: bool) -> GateResult {
        // Check T1 LP status
        if ctx.meta.lp_status_t1 != "Optimal" {
            return GateResult::fail("H1_SURFACE", "T1_LP_NOT_OPTIMAL");
        }

        if using_t2 {
            // NIFTY: require T1, T2, T3
            if ctx.features.iv2.is_none() {
                return GateResult::fail("H1_SURFACE", "T2_FEATURES_MISSING");
            }
            if let Some(ref status) = ctx.meta.lp_status_t2 {
                if status != "Optimal" {
                    return GateResult::fail("H1_SURFACE", "T2_LP_NOT_OPTIMAL");
                }
            } else {
                return GateResult::fail("H1_SURFACE", "T2_LP_STATUS_MISSING");
            }
        } else {
            // BANKNIFTY fallback: require T1, T3
            if ctx.features.iv3.is_none() {
                return GateResult::fail("H1_SURFACE", "T3_FEATURES_MISSING");
            }
            if let Some(ref status) = ctx.meta.lp_status_t3 {
                if status != "Optimal" {
                    return GateResult::fail("H1_SURFACE", "T3_LP_NOT_OPTIMAL");
                }
            } else {
                return GateResult::fail("H1_SURFACE", "T3_LP_STATUS_MISSING");
            }
        }

        GateResult::pass("H1_SURFACE")
    }

    /// H2: Calendar monotonicity
    fn check_h2_calendar(&self, ctx: &StrategyContext, using_t2: bool) -> GateResult {
        let cal = if using_t2 {
            ctx.features.cal12
        } else {
            // For T1/T3, compute cal13 = cal12 + cal23 if both present, else use cal23
            match (ctx.features.cal12, ctx.features.cal23) {
                (Some(c12), Some(c23)) => Some(c12 + c23),
                (None, Some(c23)) => Some(c23),
                _ => None,
            }
        };

        match cal {
            Some(c) if c >= -self.params.tol_cal => GateResult::pass_with_value("H2_CALENDAR", c),
            Some(c) => GateResult::fail_with_values(
                "H2_CALENDAR",
                "CALENDAR_VIOLATION",
                c,
                -self.params.tol_cal,
            ),
            None => GateResult::fail("H2_CALENDAR", "CALENDAR_DATA_MISSING"),
        }
    }

    /// H3: Quote sanity
    fn check_h3_quote(
        &self,
        straddle: &StraddleQuotes,
        now: DateTime<Utc>,
        leg: &'static str,
    ) -> GateResult {
        let name = if leg == "FRONT" {
            "H3_QUOTE_FRONT"
        } else {
            "H3_QUOTE_BACK"
        };

        if !straddle.is_valid() {
            return GateResult::fail(name, "QUOTE_INVALID");
        }

        let staleness = straddle.max_staleness_secs(now);
        if staleness > self.params.quote_staleness_secs as i64 {
            return GateResult::fail_with_values(
                name,
                "QUOTE_STALE",
                staleness as f64,
                self.params.quote_staleness_secs as f64,
            );
        }

        GateResult::pass(name)
    }

    /// H4: Liquidity ceiling
    fn check_h4_liquidity(
        &self,
        straddle: &StraddleQuotes,
        is_nifty: bool,
        is_front: bool,
    ) -> GateResult {
        let name = if is_front {
            "H4_LIQUIDITY_FRONT"
        } else {
            "H4_LIQUIDITY_BACK"
        };
        let spread_bps = straddle.spread_bps();

        let ceiling = if is_nifty {
            if is_front {
                self.params.nifty_spread_ceiling_front_bps
            } else {
                self.params.nifty_spread_ceiling_back_bps
            }
        } else {
            if is_front {
                self.params.banknifty_spread_ceiling_front_bps
            } else {
                self.params.banknifty_spread_ceiling_back_bps
            }
        };

        if spread_bps <= ceiling {
            GateResult::pass_with_value(name, spread_bps)
        } else {
            GateResult::fail_with_values(name, "SPREAD_TOO_WIDE", spread_bps, ceiling)
        }
    }

    /// Carry gate: CAL >= max(CAL_min_rel, CAL_min_abs)
    /// Phase 9: Dual threshold - relative (spread-based) AND absolute (rupee floor)
    /// Returns detailed unit sanity info in the GateResult
    fn check_carry_gate(&self, ctx: &StrategyContext, using_t2: bool) -> GateResult {
        let cal = if using_t2 {
            ctx.features.cal12
        } else {
            match (ctx.features.cal12, ctx.features.cal23) {
                (Some(c12), Some(c23)) => Some(c12 + c23),
                (None, Some(c23)) => Some(c23),
                _ => None,
            }
        };

        let cal = match cal {
            Some(c) => c,
            None => return GateResult::fail("CARRY", "CALENDAR_DATA_MISSING"),
        };

        // All values in PRICE UNITS (rupees) for sanity
        // cal is normalized (F=1), convert to price units
        let cal_price = cal * ctx.features.f1;

        // Spreads are already in price units
        let spread_front_price = ctx.front_straddle.spread();
        let spread_back_price = ctx.back_straddle.spread();
        let avg_spread_price = (spread_front_price + spread_back_price) / 2.0;

        // Phase 9: Dual threshold
        // CAL_min_rel = λ × avg_spread (relative to spreads)
        // CAL_min_abs = edge_abs_rupees (absolute floor)
        // CAL_required = max(CAL_min_rel, CAL_min_abs)
        let cal_min_rel = self.params.lambda * avg_spread_price;
        let cal_min_abs = self.params.edge_abs_rupees;
        let cal_required = cal_min_rel.max(cal_min_abs);

        // Edge in price units
        let edge_price = cal_price - cal_required;

        // Determine which threshold is binding
        let binding = if cal_min_abs > cal_min_rel {
            "ABS"
        } else {
            "REL"
        };

        // Log unit sanity (this will be captured in audit)
        // Format: cal|cal_min_rel|cal_min_abs|cal_req|sprd_f|sprd_b|edge|binding
        let sanity_str = format!(
            "cal={:.2}|rel={:.2}|abs={:.2}|req={:.2}|sprd_f={:.2}|sprd_b={:.2}|edge={:.2}|{}",
            cal_price,
            cal_min_rel,
            cal_min_abs,
            cal_required,
            spread_front_price,
            spread_back_price,
            edge_price,
            binding
        );

        if cal_price >= cal_required {
            GateResult {
                name: "CARRY".to_string(),
                passed: true,
                value: Some(cal_price),
                threshold: Some(cal_required),
                reason: Some(sanity_str),
            }
        } else {
            GateResult {
                name: "CARRY".to_string(),
                passed: false,
                value: Some(cal_price),
                threshold: Some(cal_required),
                reason: Some(format!("INSUFFICIENT_EDGE|{}", sanity_str)),
            }
        }
    }

    /// R1: Term structure inversion limit
    /// Uses iv1-iv3 if available, falls back to iv1-iv2 for T1/T2 trades
    fn check_r1_inversion(
        &self,
        ctx: &StrategyContext,
        is_nifty: bool,
        using_t2: bool,
    ) -> GateResult {
        // Determine which IV to use for inversion check
        // If trading T1/T2, use iv2 if iv3 not available
        let (inversion, used_pair) = if let Some(iv3) = ctx.features.iv3 {
            (ctx.features.iv1 - iv3, "iv1-iv3")
        } else if using_t2 {
            // Fall back to iv1-iv2 when trading T1/T2 and iv3 missing
            match ctx.features.iv2 {
                Some(iv2) => (ctx.features.iv1 - iv2, "iv1-iv2"),
                None => return GateResult::fail("R1_INVERSION", "IV_DATA_MISSING"),
            }
        } else {
            // Trading T1/T3 but iv3 missing - cannot proceed
            return GateResult::fail("R1_INVERSION", "IV3_MISSING_FOR_T3_TRADE");
        };

        let max_inversion = if is_nifty {
            self.params.nifty_ts_inversion_max
        } else {
            self.params.banknifty_ts_inversion_max
        };

        if inversion <= max_inversion {
            GateResult {
                name: "R1_INVERSION".to_string(),
                passed: true,
                value: Some(inversion),
                threshold: Some(max_inversion),
                reason: Some(used_pair.to_string()),
            }
        } else {
            GateResult {
                name: "R1_INVERSION".to_string(),
                passed: false,
                value: Some(inversion),
                threshold: Some(max_inversion),
                reason: Some(format!("EXTREME_INVERSION|{}", used_pair)),
            }
        }
    }

    /// R2: Skew stress filter
    fn check_r2_skew(&self, ctx: &StrategyContext) -> GateResult {
        let skews: Vec<f64> = [ctx.features.sk1, ctx.features.sk2, ctx.features.sk3]
            .iter()
            .filter_map(|&s| s)
            .collect();

        if skews.is_empty() {
            return GateResult::fail("R2_SKEW", "NO_SKEW_DATA");
        }

        let min_skew = skews.iter().cloned().fold(f64::INFINITY, f64::min);

        if min_skew >= self.params.skew_stress_min {
            GateResult::pass_with_value("R2_SKEW", min_skew)
        } else {
            GateResult::fail_with_values(
                "R2_SKEW",
                "SKEW_STRESS",
                min_skew,
                self.params.skew_stress_min,
            )
        }
    }

    /// Compute hedge ratio (used by E1/E2 and construct_trade)
    fn compute_hedge_ratio(&self, ctx: &StrategyContext, using_t2: bool) -> f64 {
        let (iv_back, tty_back, f_back) = if using_t2 {
            (
                ctx.features.iv2.unwrap_or(ctx.features.iv1),
                ctx.features.tty2.unwrap_or(ctx.features.tty1),
                ctx.features.f2.unwrap_or(ctx.features.f1),
            )
        } else {
            (
                ctx.features.iv3.unwrap_or(ctx.features.iv1),
                ctx.features.tty3.unwrap_or(ctx.features.tty1),
                ctx.features.f3.unwrap_or(ctx.features.f1),
            )
        };

        let vega_front = self.bs_vega(
            ctx.features.f1,
            ctx.front_straddle.strike,
            ctx.features.tty1,
            ctx.features.iv1,
        );
        let vega_back = self.bs_vega(f_back, ctx.back_straddle.strike, tty_back, iv_back);

        let h = if vega_back > 1e-9 {
            vega_front / vega_back
        } else {
            1.0
        };
        h.clamp(self.params.h_min, self.params.h_max)
    }

    /// E1: Premium-based calendar gap (real tradable gap)
    /// gap_premium = (h × P_back) - P_front
    /// where P = mid(CE) + mid(PE) is straddle premium
    /// Gate: gap_premium >= GAP_ABS (underlying-specific)
    fn check_e1_premium_gap(&self, ctx: &StrategyContext, h: f64, is_nifty: bool) -> GateResult {
        // Actual straddle premiums from quotes (in rupees)
        let p_front = ctx.front_straddle.mid(); // CE mid + PE mid
        let p_back = ctx.back_straddle.mid();

        // Premium gap accounting for hedge ratio
        let gap_premium = (h * p_back) - p_front;

        // Underlying-specific threshold
        let gap_abs = if is_nifty {
            self.params.gap_abs_nifty
        } else {
            self.params.gap_abs_banknifty
        };

        // Detailed logging
        let sanity_str = format!(
            "p_front={:.2}|p_back={:.2}|h={:.3}|h×p_back={:.2}|gap={:.2}|req={:.2}",
            p_front,
            p_back,
            h,
            h * p_back,
            gap_premium,
            gap_abs
        );

        if gap_premium >= gap_abs {
            GateResult {
                name: "E1_PREMIUM_GAP".to_string(),
                passed: true,
                value: Some(gap_premium),
                threshold: Some(gap_abs),
                reason: Some(sanity_str),
            }
        } else {
            GateResult {
                name: "E1_PREMIUM_GAP".to_string(),
                passed: false,
                value: Some(gap_premium),
                threshold: Some(gap_abs),
                reason: Some(format!("INSUFFICIENT_GAP|{}", sanity_str)),
            }
        }
    }

    /// E2: Round-trip friction dominance test
    /// friction_entry = spread_front_str/2 + h × spread_back_str/2
    /// friction_round = 2 × friction_entry
    /// Gate: gap_premium >= μ × friction_round
    fn check_e2_friction_dominance(&self, ctx: &StrategyContext, h: f64) -> GateResult {
        // Straddle spreads (in rupees)
        let spread_front = ctx.front_straddle.spread();
        let spread_back = ctx.back_straddle.spread();

        // Entry friction (half-spread on each leg, weighted by hedge ratio)
        let friction_entry = (spread_front / 2.0) + h * (spread_back / 2.0);

        // Round-trip friction (entry + exit)
        let friction_round = 2.0 * friction_entry;

        // Premium gap (same calculation as E1)
        let p_front = ctx.front_straddle.mid();
        let p_back = ctx.back_straddle.mid();
        let gap_premium = (h * p_back) - p_front;

        // Required gap = μ × friction_round
        let required_gap = self.params.mu_friction * friction_round;

        // Friction dominance ratio
        let dominance_ratio = if friction_round > 1e-9 {
            gap_premium / friction_round
        } else {
            f64::MAX
        };

        // Detailed logging
        let sanity_str = format!(
            "sprd_f={:.2}|sprd_b={:.2}|h={:.3}|fric_entry={:.2}|fric_round={:.2}|gap={:.2}|req={:.2}|ratio={:.2}",
            spread_front,
            spread_back,
            h,
            friction_entry,
            friction_round,
            gap_premium,
            required_gap,
            dominance_ratio
        );

        if gap_premium >= required_gap {
            GateResult {
                name: "E2_FRICTION_DOMINANCE".to_string(),
                passed: true,
                value: Some(dominance_ratio),
                threshold: Some(self.params.mu_friction),
                reason: Some(sanity_str),
            }
        } else {
            GateResult {
                name: "E2_FRICTION_DOMINANCE".to_string(),
                passed: false,
                value: Some(dominance_ratio),
                threshold: Some(self.params.mu_friction),
                reason: Some(format!("FRICTION_DOMINATES|{}", sanity_str)),
            }
        }
    }

    /// E3: Friction floor gate (Phase 9.2)
    /// Uses effective friction = max(observed, floor) to prevent
    /// unrealistically low friction from passing the gate trivially.
    /// Gate: gap_premium >= μ × friction_round_eff
    fn check_e3_friction_floor(&self, ctx: &StrategyContext, h: f64, is_nifty: bool) -> GateResult {
        // Straddle spreads (observed)
        let spread_front = ctx.front_straddle.spread();
        let spread_back = ctx.back_straddle.spread();

        // Observed friction
        let friction_entry_obs = (spread_front / 2.0) + h * (spread_back / 2.0);
        let friction_round_obs = 2.0 * friction_entry_obs;

        // Friction floor (conservative assumption)
        let friction_floor = if is_nifty {
            self.params.floor_friction_round_nifty
        } else {
            self.params.floor_friction_round_banknifty
        };

        // Effective friction = max(observed, floor)
        let friction_round_eff = friction_round_obs.max(friction_floor);

        // Is floor binding?
        let floor_binding = friction_floor > friction_round_obs;

        // Premium gap
        let p_front = ctx.front_straddle.mid();
        let p_back = ctx.back_straddle.mid();
        let gap_premium = (h * p_back) - p_front;

        // Required gap = μ × friction_round_eff
        let required_gap = self.params.mu_friction * friction_round_eff;

        // Effective dominance ratio
        let dominance_ratio_eff = if friction_round_eff > 1e-9 {
            gap_premium / friction_round_eff
        } else {
            f64::MAX
        };

        // Detailed logging
        let sanity_str = format!(
            "fric_obs={:.2}|fric_floor={:.2}|fric_eff={:.2}|gap={:.2}|req={:.2}|ratio={:.2}|{}",
            friction_round_obs,
            friction_floor,
            friction_round_eff,
            gap_premium,
            required_gap,
            dominance_ratio_eff,
            if floor_binding {
                "FLOOR_BINDING"
            } else {
                "OBSERVED"
            }
        );

        if gap_premium >= required_gap {
            GateResult {
                name: "E3_FRICTION_FLOOR".to_string(),
                passed: true,
                value: Some(dominance_ratio_eff),
                threshold: Some(self.params.mu_friction),
                reason: Some(sanity_str),
            }
        } else {
            GateResult {
                name: "E3_FRICTION_FLOOR".to_string(),
                passed: false,
                value: Some(dominance_ratio_eff),
                threshold: Some(self.params.mu_friction),
                reason: Some(format!("FRICTION_FLOOR_BLOCKS|{}", sanity_str)),
            }
        }
    }

    /// Construct trade intent
    fn construct_trade(&self, ctx: &StrategyContext) -> (EnterIntent, bool) {
        let using_t2 = ctx.meta.t2_expiry.is_some() && ctx.features.iv2.is_some();

        // Get back expiry info
        let (back_expiry, iv_back, tty_back, f_back) = if using_t2 {
            (
                ctx.meta.t2_expiry.clone().unwrap(),
                ctx.features.iv2.unwrap(),
                ctx.features.tty2.unwrap(),
                ctx.features.f2.unwrap(),
            )
        } else {
            (
                ctx.meta.t3_expiry.clone().unwrap(),
                ctx.features.iv3.unwrap(),
                ctx.features.tty3.unwrap(),
                ctx.features.f3.unwrap(),
            )
        };

        // Compute vega hedge ratio
        let vega_front = self.bs_vega(
            ctx.features.f1,
            ctx.front_straddle.strike,
            ctx.features.tty1,
            ctx.features.iv1,
        );
        let vega_back = self.bs_vega(f_back, ctx.back_straddle.strike, tty_back, iv_back);

        let mut h = if vega_back > 1e-9 {
            vega_front / vega_back
        } else {
            1.0
        };
        let h_clamped = h < self.params.h_min || h > self.params.h_max;
        h = h.clamp(self.params.h_min, self.params.h_max);

        // Compute friction and cal values
        let spread_front = ctx.front_straddle.spread();
        let spread_back = ctx.back_straddle.spread();
        let friction = (spread_front + spread_back) / 2.0;

        let cal_value = if using_t2 {
            ctx.features.cal12.unwrap_or(0.0)
        } else {
            ctx.features.cal12.unwrap_or(0.0) + ctx.features.cal23.unwrap_or(0.0)
        };

        let avg_spread_normalized = friction / ctx.features.f1;
        let cal_min = self.params.lambda * avg_spread_normalized;

        // Compute lots
        let notional = ctx.features.f1 * ctx.meta.lot_size as f64;
        let front_premium = ctx.front_straddle.mid();
        let back_premium = h * ctx.back_straddle.mid();
        let gross_premium = front_premium + back_premium;

        let mut lots = if gross_premium > 1e-9 {
            ((self.params.risk_bps / 1e4) * notional / gross_premium).floor() as i32
        } else {
            1
        };
        lots = lots.max(1).min(self.params.max_lots);

        let intent = EnterIntent {
            underlying: ctx.meta.underlying.clone(),
            front_expiry: ctx.meta.t1_expiry.clone(),
            back_expiry,
            front_strike: ctx.front_straddle.strike,
            back_strike: ctx.back_straddle.strike,
            front_lots: -lots,                           // short front
            back_lots: (h * lots as f64).round() as i32, // long back
            hedge_ratio: h,
            h_clamped,
            cal_value,
            cal_min,
            friction_estimate: friction,
        };

        (intent, h_clamped)
    }

    /// Black-Scholes vega for straddle (CE + PE)
    fn bs_vega(&self, f: f64, k: f64, t: f64, iv: f64) -> f64 {
        if t <= 0.0 || iv <= 0.0 {
            return 0.0;
        }

        let sqrt_t = t.sqrt();
        let d1 = ((f / k).ln() + 0.5 * iv * iv * t) / (iv * sqrt_t);

        // Vega = F * sqrt(T) * N'(d1) where N' is standard normal pdf
        let nprime_d1 = (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt();

        // Straddle vega = 2 * call vega (put vega equals call vega)
        2.0 * f * sqrt_t * nprime_d1
    }

    /// Build base audit record
    fn build_audit_base(&self, ctx: &StrategyContext, gates: &GateCheckResult) -> AuditRecord {
        let using_t2 = ctx.meta.t2_expiry.is_some() && ctx.features.iv2.is_some();

        let sk_min = [ctx.features.sk1, ctx.features.sk2, ctx.features.sk3]
            .iter()
            .filter_map(|&s| s)
            .fold(f64::INFINITY, f64::min);

        AuditRecord {
            ts: ctx.ts,
            underlying: ctx.meta.underlying.clone(),
            decision: String::new(),
            reason_code: None,
            gates: Some(gates.clone()),
            front_expiry: Some(ctx.meta.t1_expiry.clone()),
            back_expiry: if using_t2 {
                ctx.meta.t2_expiry.clone()
            } else {
                ctx.meta.t3_expiry.clone()
            },
            front_spread_bps: Some(ctx.front_straddle.spread_bps()),
            back_spread_bps: Some(ctx.back_straddle.spread_bps()),
            iv1: ctx.features.iv1,
            iv_back: if using_t2 {
                ctx.features.iv2
            } else {
                ctx.features.iv3
            },
            cal_value: if using_t2 {
                ctx.features.cal12
            } else {
                match (ctx.features.cal12, ctx.features.cal23) {
                    (Some(c12), Some(c23)) => Some(c12 + c23),
                    _ => ctx.features.cal23,
                }
            },
            cal_min: None,
            sk_min: if sk_min.is_finite() {
                Some(sk_min)
            } else {
                None
            },
            hedge_ratio: None,
            lots: None,
            h_clamped: None,
        }
    }

    /// Evaluate exit conditions for open position
    pub fn evaluate_exit(
        &self,
        ctx: &StrategyContext,
        entry_friction: f64,
        current_pnl: f64,
    ) -> Option<StrategyDecision> {
        // Time exit: minutes before close
        if ctx.minutes_to_close <= self.params.exit_minutes_before_close {
            return Some(StrategyDecision::Exit {
                intent: ExitIntent {
                    reason: "TIME_EXIT".to_string(),
                    pnl_bps: Some(current_pnl),
                },
            });
        }

        // Gamma risk exit: last hour on expiry day
        if ctx.is_expiry_day_front && ctx.minutes_to_close <= 60 {
            return Some(StrategyDecision::Exit {
                intent: ExitIntent {
                    reason: "GAMMA_RISK_EXIT".to_string(),
                    pnl_bps: Some(current_pnl),
                },
            });
        }

        // Take profit
        let take_profit_threshold = self.params.take_profit_mult * entry_friction;
        if current_pnl >= take_profit_threshold {
            return Some(StrategyDecision::Exit {
                intent: ExitIntent {
                    reason: "TAKE_PROFIT".to_string(),
                    pnl_bps: Some(current_pnl),
                },
            });
        }

        // Stop loss
        let stop_loss_threshold = -self.params.stop_loss_mult * entry_friction;
        if current_pnl <= stop_loss_threshold {
            return Some(StrategyDecision::Exit {
                intent: ExitIntent {
                    reason: "STOP_LOSS".to_string(),
                    pnl_bps: Some(current_pnl),
                },
            });
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_frozen_params() {
        assert_eq!(FROZEN_PARAMS.lambda, 1.5);
        assert_eq!(FROZEN_PARAMS.risk_bps, 7.5);
        assert_eq!(FROZEN_PARAMS.max_lots, 2);
        assert_eq!(FROZEN_PARAMS.nifty_spread_ceiling_front_bps, 35.0);
        assert_eq!(FROZEN_PARAMS.banknifty_spread_ceiling_back_bps, 80.0);
    }

    #[test]
    fn test_straddle_spread_bps() {
        let now = Utc.with_ymd_and_hms(2026, 1, 23, 10, 0, 0).unwrap();
        let straddle = StraddleQuotes {
            expiry: "26JAN".to_string(),
            strike: 25300.0,
            ce: QuoteSnapshot {
                bid: 100.0,
                ask: 102.0,
                last_ts: now,
            },
            pe: QuoteSnapshot {
                bid: 95.0,
                ask: 97.0,
                last_ts: now,
            },
        };

        // mid = (100+102)/2 + (95+97)/2 = 101 + 96 = 197
        // spread = (102+97) - (100+95) = 199 - 195 = 4
        // spread_bps = 10000 * 4 / 197 ≈ 203
        let spread_bps = straddle.spread_bps();
        assert!(spread_bps > 200.0 && spread_bps < 210.0);
    }
}
