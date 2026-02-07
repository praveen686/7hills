//! Sniper Doctrine: Multi-Gate Admission System
//!
//! High conviction, low frequency trading with explicit refusal logging.
//! Every refused tick logs exactly which gate failed.

use crate::paper::intent::{PaperIntent, Side};
use crate::paper::state::{DecisionMetrics, R3Eligibility, RefusalReason};
use std::collections::VecDeque;

/// Sniper configuration parameters.
#[derive(Debug, Clone)]
pub struct SniperConfig {
    // Gate 1: Regime gating
    pub allowed_regimes: Vec<&'static str>, // Only R3
    pub confidence_min: f64,                // 0.85

    // Gate 2: Quantitative thresholds
    pub toxicity_max: f64,      // 0.60
    pub d_perp_max: f64,        // 0.30
    pub spread_max_bps: f64,    // 0.5 bps
    pub imbalance_min_abs: f64, // 0.25

    // Gate 3: Edge calculation
    pub fee_rate: f64,          // 0.001 (0.1% taker)
    pub sniper_buffer_bps: f64, // 15 bps additional buffer

    // Gate 4: Rate limiting
    pub cooldown_seconds: u64,        // 60
    pub max_entries_per_hour: u32,    // 3
    pub max_entries_per_session: u32, // 10

    // Gate 6: FTI confirmation (require FTI signal, not just toxicity)
    pub fti_persist_min: f64,      // 0.35 - require FTI confirmation
    pub fti_level_min: f64,        // 0.0 - minimum FTI level (optional stricter check)
    pub require_fti_confirm: bool, // true = require FTI, not just toxicity

    // Trade sizing
    pub qty: f64,

    // Behavior
    pub no_flip: bool, // true = don't flip position on same tick
}

impl Default for SniperConfig {
    fn default() -> Self {
        Self::sniper()
    }
}

impl SniperConfig {
    /// Production sniper mode - very restrictive, R3 only.
    pub fn sniper() -> Self {
        Self {
            // Only R3 trades - very restrictive
            allowed_regimes: vec!["R3"],
            confidence_min: 0.85,

            // Quantitative thresholds
            toxicity_max: 0.60,
            d_perp_max: 0.30, // Not used (gate removed), kept for reference
            spread_max_bps: 0.5,
            imbalance_min_abs: 0.25,

            // Edge: need to clear fees + spread + buffer
            fee_rate: 0.001,         // 0.1% taker
            sniper_buffer_bps: 15.0, // 15 bps safety margin

            // Rate limiting
            cooldown_seconds: 60,
            max_entries_per_hour: 3,
            max_entries_per_session: 10,

            // FTI confirmation: require actual follow-through, not just toxicity
            fti_persist_min: 0.35,     // 35% of windows must show FTI elevation
            fti_level_min: 0.0,        // No minimum level (persistence is enough)
            require_fti_confirm: true, // Yes, require FTI confirmation

            // Trade sizing
            qty: 0.001, // 0.001 BTC

            // Don't flip position within same session without closing first
            no_flip: true,
        }
    }

    /// Canary mode - Profitability test with controlled exits.
    /// Purpose: Force enough trades to test whether signal has edge after fees.
    /// Gates relaxed but not completely disabled to maintain signal directionality.
    /// Run until >= 50 accepted trades, then evaluate expectancy.
    pub fn canary() -> Self {
        Self {
            // ALL regimes allowed - no regime veto
            allowed_regimes: vec!["R0", "R1", "R2", "R3"],
            confidence_min: 0.55, // Low - almost always passes

            // Quantitative thresholds - effectively disabled
            toxicity_max: 1.00,      // Accept any toxicity
            d_perp_max: 100.0,       // Not used (gate removed)
            spread_max_bps: 5.0,     // Permissive
            imbalance_min_abs: 0.05, // Low (0.05) - just ensures directionality exists

            // Edge: ZERO buffer for canary
            fee_rate: 0.001,        // 0.1% taker
            sniper_buffer_bps: 0.0, // No buffer - just clear fees+spread

            // Rate limiting - DISABLED for canary
            cooldown_seconds: 0,
            max_entries_per_hour: 9999,    // Effectively disabled
            max_entries_per_session: 9999, // Effectively disabled

            // FTI confirmation DISABLED
            fti_persist_min: 0.0,
            fti_level_min: 0.0,
            require_fti_confirm: false,

            // Trade sizing - tiny position
            qty: 0.001, // 0.001 BTC max

            // NO FLIP - prevents immediate reversal churn
            no_flip: true,
        }
    }
}

/// Sniper state for tracking rate limits and cooldowns.
#[derive(Debug, Clone)]
pub struct SniperState {
    /// Last entry timestamp (ms)
    pub last_entry_ms: Option<u64>,
    /// Entry timestamps in last hour (for rate limiting)
    pub entries_last_hour: VecDeque<u64>,
    /// Total entries this session
    pub session_entries: u32,
    /// Last entry side (for no-flip policy)
    pub last_entry_side: Option<Side>,
    /// Session start timestamp (ms)
    pub session_start_ms: u64,
}

impl SniperState {
    pub fn new(session_start_ms: u64) -> Self {
        Self {
            last_entry_ms: None,
            entries_last_hour: VecDeque::with_capacity(10),
            session_entries: 0,
            last_entry_side: None,
            session_start_ms,
        }
    }

    /// Prune old entries and return count in last hour.
    pub fn entries_in_last_hour(&mut self, now_ms: u64) -> u32 {
        let hour_ago = now_ms.saturating_sub(3600 * 1000);
        while let Some(&ts) = self.entries_last_hour.front() {
            if ts < hour_ago {
                self.entries_last_hour.pop_front();
            } else {
                break;
            }
        }
        self.entries_last_hour.len() as u32
    }

    /// Record an entry.
    pub fn record_entry(&mut self, now_ms: u64, side: Side) {
        self.last_entry_ms = Some(now_ms);
        self.entries_last_hour.push_back(now_ms);
        self.session_entries += 1;
        self.last_entry_side = Some(side);
    }

    /// Check if cooldown has passed.
    pub fn cooldown_ok(&self, now_ms: u64, cooldown_seconds: u64) -> bool {
        match self.last_entry_ms {
            Some(last) => now_ms >= last + (cooldown_seconds * 1000),
            None => true,
        }
    }
}

/// Gate identification for refusal logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SniperGate {
    Gate0Hygiene,   // Crossed book, stale quotes, etc.
    Gate1Regime,    // R2/R3 only, confidence, warmup
    Gate2Metrics,   // d_perp, fragility, toxicity, spread
    Gate3Edge,      // Edge must clear costs
    Gate4RateLimit, // Cooldown, hourly/session limits
    Gate5Setup,     // Imbalance, tape confirmation
    Gate6FTI,       // FTI confirmation (not just toxicity-driven R3)
    Passed,         // All gates passed
}

impl SniperGate {
    pub fn as_str(&self) -> &'static str {
        match self {
            SniperGate::Gate0Hygiene => "GATE0_HYGIENE",
            SniperGate::Gate1Regime => "GATE1_REGIME",
            SniperGate::Gate2Metrics => "GATE2_METRICS",
            SniperGate::Gate3Edge => "GATE3_EDGE",
            SniperGate::Gate4RateLimit => "GATE4_RATE",
            SniperGate::Gate5Setup => "GATE5_SETUP",
            SniperGate::Gate6FTI => "GATE6_FTI",
            SniperGate::Passed => "PASSED",
        }
    }
}

/// What caused R3 to be true (or not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R3Cause {
    /// FTI persistence triggered R3
    FTI,
    /// Toxicity persistence triggered R3
    TOX,
    /// Both FTI and toxicity persistence are elevated
    Both,
    /// Neither is elevated (not R3 via persistence)
    None,
}

impl R3Cause {
    pub fn as_str(&self) -> &'static str {
        match self {
            R3Cause::FTI => "FTI",
            R3Cause::TOX => "TOX",
            R3Cause::Both => "BOTH",
            R3Cause::None => "NONE",
        }
    }

    /// Determine R3 cause from persistence values.
    /// Uses same thresholds as regime classifier (0.3).
    pub fn from_persistence(fti_persist: f64, toxicity_persist: f64) -> Self {
        let fti_high = fti_persist >= 0.3;
        let tox_high = toxicity_persist >= 0.3;

        match (fti_high, tox_high) {
            (true, true) => R3Cause::Both,
            (true, false) => R3Cause::FTI,
            (false, true) => R3Cause::TOX,
            (false, false) => R3Cause::None,
        }
    }
}

/// Input data for sniper admission.
#[derive(Debug, Clone)]
pub struct SniperInput {
    pub symbol: String,
    pub tick: u64,
    pub now_ms: u64,

    // From SLRT pipeline
    pub eligibility: R3Eligibility,
    pub regime: String,
    pub metrics: DecisionMetrics,
    pub slrt_refusal_reasons: Vec<RefusalReason>,

    // Market data
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
    pub spread_bps: f64,
    pub imbalance: f64,

    // Position
    pub position_size: f64,
}

/// Output of sniper admission.
#[derive(Debug, Clone)]
pub struct SniperOutput {
    /// What the strategy proposes THIS tick (may exist even if refused)
    pub proposed: Option<PaperIntent>,
    /// What was accepted THIS tick (None if any gate failed)
    pub accepted: Option<PaperIntent>,
    /// All refusal reasons (empty if accepted)
    pub refusal_reasons: Vec<RefusalReason>,
    /// Final gate that stopped execution (Passed if all cleared)
    pub final_gate: SniperGate,
    /// Metrics for display
    pub metrics: DecisionMetrics,
    /// What caused R3 to be true (FTI, TOX, BOTH, or NONE)
    pub r3_cause: R3Cause,
}

/// Helper to create a refusal reason.
fn rr(code: &'static str, detail: impl Into<String>) -> RefusalReason {
    RefusalReason::new(code, detail)
}

/// Main sniper admission function.
/// Returns proposed_intent, accepted_this_tick, refusal_reasons[], metrics.
pub fn sniper_admission(
    config: &SniperConfig,
    state: &mut SniperState,
    input: &SniperInput,
) -> SniperOutput {
    let mut reasons: Vec<RefusalReason> = Vec::new();
    let mut final_gate = SniperGate::Passed;

    // Copy SLRT's refusal reasons
    reasons.extend(input.slrt_refusal_reasons.iter().cloned());

    // ============================================================
    // GATE 0: Hard Hygiene (already checked by SLRT, but we log it)
    // ============================================================
    if input.best_bid <= 0.0 || input.best_ask <= 0.0 {
        reasons.push(rr("GATE0_NO_QUOTES", "missing bid or ask"));
        final_gate = SniperGate::Gate0Hygiene;
    }
    if input.best_bid >= input.best_ask {
        reasons.push(rr(
            "GATE0_CROSSED",
            format!("bid {} >= ask {}", input.best_bid, input.best_ask),
        ));
        final_gate = SniperGate::Gate0Hygiene;
    }

    // ============================================================
    // GATE 1: Regime Gating
    // ============================================================
    let in_allowed_regime = config.allowed_regimes.contains(&input.regime.as_str());

    if input.eligibility != R3Eligibility::Eligible {
        // SLRT already refused - this is expected, just annotate
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate1Regime;
        }
    }

    if !in_allowed_regime {
        reasons.push(rr(
            "GATE1_REGIME",
            format!("{} not in {:?}", input.regime, config.allowed_regimes),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate1Regime;
        }
    }

    // CONFIDENCE: No silent poisoning - treat None as ABSENT, not as 0.0 or 1.0
    match input.metrics.confidence {
        None => {
            reasons.push(rr("GATE1_CONFIDENCE_ABSENT", "confidence metric missing"));
            if final_gate == SniperGate::Passed {
                final_gate = SniperGate::Gate1Regime;
            }
        }
        Some(conf) if conf < config.confidence_min => {
            reasons.push(rr(
                "GATE1_CONFIDENCE",
                format!("{:.2} < {:.2}", conf, config.confidence_min),
            ));
            if final_gate == SniperGate::Passed {
                final_gate = SniperGate::Gate1Regime;
            }
        }
        Some(_) => {} // passes
    }

    // ============================================================
    // GATE 2: Quantitative Metrics
    // ============================================================
    // NOTE: d_perp gate REMOVED - R3 requires d_perp > 2.0 by definition,
    // so a d_perp_max check would contradict regime gating.
    // The regime classifier already handles d_perp semantics.

    // TOXICITY: No silent poisoning - treat None as ABSENT, not as 0.0 or 1.0
    match input.metrics.toxicity {
        None => {
            reasons.push(rr("GATE2_TOXICITY_ABSENT", "toxicity metric missing"));
            if final_gate == SniperGate::Passed {
                final_gate = SniperGate::Gate2Metrics;
            }
        }
        Some(tox) if tox > config.toxicity_max => {
            reasons.push(rr(
                "GATE2_TOXICITY",
                format!("{:.2} > {:.2}", tox, config.toxicity_max),
            ));
            if final_gate == SniperGate::Passed {
                final_gate = SniperGate::Gate2Metrics;
            }
        }
        Some(_) => {} // passes
    }

    if input.spread_bps > config.spread_max_bps {
        reasons.push(rr(
            "GATE2_SPREAD",
            format!(
                "{:.2}bps > {:.2}bps",
                input.spread_bps, config.spread_max_bps
            ),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate2Metrics;
        }
    }

    // ============================================================
    // GATE 5: Setup Confirmation (check early for proposal direction)
    // ============================================================
    let abs_imbalance = input.imbalance.abs();
    if abs_imbalance < config.imbalance_min_abs {
        reasons.push(rr(
            "GATE5_IMBALANCE",
            format!("|{:.2}| < {:.2}", input.imbalance, config.imbalance_min_abs),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate5Setup;
        }
    }

    // ============================================================
    // PROPOSAL DIRECTION (inventory-aware)
    // ============================================================
    let signal_buy = input.imbalance > config.imbalance_min_abs;
    let signal_sell = input.imbalance < -config.imbalance_min_abs;

    let proposed_side: Option<Side> = if input.position_size.abs() < 1e-9 {
        // FLAT: propose based on signal
        if signal_buy {
            Some(Side::Buy)
        } else if signal_sell {
            Some(Side::Sell)
        } else {
            None
        }
    } else if input.position_size > 0.0 {
        // LONG: only propose SELL (to close)
        if signal_sell { Some(Side::Sell) } else { None }
    } else {
        // SHORT: only propose BUY (to close)
        if signal_buy { Some(Side::Buy) } else { None }
    };

    // Create proposed intent (even if we'll refuse it)
    let proposed = proposed_side.map(|side| {
        PaperIntent::market(
            input.symbol.clone(),
            side,
            config.qty,
            format!("sniper-{}", input.tick),
            input.now_ms,
        )
    });

    // If no proposal, we're done (nothing to accept)
    if proposed.is_none() {
        // Still compute R3 cause for display
        let fti_persist = input.metrics.fti_persist.unwrap_or(0.0);
        let toxicity_persist = input.metrics.toxicity_persist.unwrap_or(0.0);
        let r3_cause = R3Cause::from_persistence(fti_persist, toxicity_persist);

        return SniperOutput {
            proposed: None,
            accepted: None,
            refusal_reasons: reasons,
            final_gate,
            metrics: input.metrics.clone(),
            r3_cause,
        };
    }

    let side = proposed_side.unwrap();
    let fill_price = match side {
        Side::Buy => input.best_ask,
        Side::Sell => input.best_bid,
    };

    // ============================================================
    // GATE 3: Edge Must Clear Costs
    // ============================================================
    let notional = config.qty * fill_price;
    let fee_cost_roundtrip = 2.0 * notional * config.fee_rate;
    let spread_cost = (input.best_ask - input.best_bid) * config.qty;
    let sniper_buffer = (config.sniper_buffer_bps / 10000.0) * notional;
    let min_edge_required = fee_cost_roundtrip + spread_cost + sniper_buffer;

    // Edge estimate from imbalance magnitude (simplified model)
    // Better models could use d_perp, toxicity, etc.
    let edge_estimate = abs_imbalance * 0.5 * notional * 0.01; // ~0.5% of imbalance as edge

    if edge_estimate < min_edge_required {
        reasons.push(rr(
            "GATE3_EDGE",
            format!(
                "est ${:.4} < required ${:.4} (fees=${:.4} spread=${:.4} buffer=${:.4})",
                edge_estimate, min_edge_required, fee_cost_roundtrip, spread_cost, sniper_buffer
            ),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate3Edge;
        }
    }

    // ============================================================
    // GATE 4: Rate Limiting
    // ============================================================
    if !state.cooldown_ok(input.now_ms, config.cooldown_seconds) {
        let elapsed_secs = state
            .last_entry_ms
            .map(|last| (input.now_ms.saturating_sub(last)) / 1000)
            .unwrap_or(0);
        reasons.push(rr(
            "GATE4_COOLDOWN",
            format!(
                "{}s elapsed < {}s required",
                elapsed_secs, config.cooldown_seconds
            ),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate4RateLimit;
        }
    }

    let entries_last_hour = state.entries_in_last_hour(input.now_ms);
    if entries_last_hour >= config.max_entries_per_hour {
        reasons.push(rr(
            "GATE4_HOURLY",
            format!(
                "{} entries >= {} max/hour",
                entries_last_hour, config.max_entries_per_hour
            ),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate4RateLimit;
        }
    }

    if state.session_entries >= config.max_entries_per_session {
        reasons.push(rr(
            "GATE4_SESSION",
            format!(
                "{} entries >= {} max/session",
                state.session_entries, config.max_entries_per_session
            ),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate4RateLimit;
        }
    }

    // No-flip policy: if last entry was opposite side, refuse
    if config.no_flip
        && let Some(last_side) = state.last_entry_side
        && last_side != side
        && input.position_size.abs() > 1e-9
    {
        reasons.push(rr(
            "GATE4_NO_FLIP",
            format!(
                "last={:?} current={:?} pos={:.6}",
                last_side, side, input.position_size
            ),
        ));
        if final_gate == SniperGate::Passed {
            final_gate = SniperGate::Gate4RateLimit;
        }
    }

    // ============================================================
    // GATE 6: FTI Confirmation
    // Require actual follow-through signal, not just toxicity-driven R3
    // Only evaluated when regime == R3 (otherwise irrelevant)
    // ============================================================
    let fti_persist = input.metrics.fti_persist.unwrap_or(0.0);
    let fti_level = input.metrics.fti_level.unwrap_or(0.0);
    let toxicity_persist = input.metrics.toxicity_persist.unwrap_or(0.0);

    // Compute R3 cause for display
    let r3_cause = R3Cause::from_persistence(fti_persist, toxicity_persist);

    // Only evaluate Gate6 when regime is R3 (the gate is meaningless otherwise)
    let is_r3 = input.regime == "R3";

    if config.require_fti_confirm && is_r3 {
        let fti_persist_ok = fti_persist >= config.fti_persist_min;
        let fti_level_ok = fti_level >= config.fti_level_min;

        if !fti_persist_ok {
            reasons.push(rr(
                "GATE6_FTI_PERSIST",
                format!(
                    "fti_persist {:.2} < {:.2} (R3 via {} only)",
                    fti_persist,
                    config.fti_persist_min,
                    r3_cause.as_str()
                ),
            ));
            if final_gate == SniperGate::Passed {
                final_gate = SniperGate::Gate6FTI;
            }
        }

        if fti_persist_ok && !fti_level_ok && config.fti_level_min > 0.0 {
            reasons.push(rr(
                "GATE6_FTI_LEVEL",
                format!("fti_level {:.4} < {:.4}", fti_level, config.fti_level_min),
            ));
            if final_gate == SniperGate::Passed {
                final_gate = SniperGate::Gate6FTI;
            }
        }
    }

    // ============================================================
    // FINAL DECISION
    // ============================================================
    let accepted = if final_gate == SniperGate::Passed {
        // All gates passed - record entry and return accepted
        state.record_entry(input.now_ms, side);
        proposed.clone()
    } else {
        None
    };

    SniperOutput {
        proposed,
        accepted,
        refusal_reasons: reasons,
        final_gate,
        metrics: input.metrics.clone(),
        r3_cause,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sniper_config_defaults() {
        let config = SniperConfig::default();
        assert_eq!(config.allowed_regimes, vec!["R3"]);
        assert_eq!(config.confidence_min, 0.85);
        assert_eq!(config.toxicity_max, 0.60);
    }

    #[test]
    fn test_sniper_state_cooldown() {
        let mut state = SniperState::new(0);
        assert!(state.cooldown_ok(1000, 60));

        state.record_entry(1000, Side::Buy);
        assert!(!state.cooldown_ok(30_000, 60)); // 30s < 60s
        assert!(state.cooldown_ok(61_000, 60)); // 61s > 60s
    }
}
