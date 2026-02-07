//! SLR Strategy - SLRT-GPU v1.1 Execution Binding
//!
//! This is NOT a new alpha. It is the canonical execution policy for
//! SLRT regime classifications, implementing:
//!
//! - §6.6: Regime permission (R2/R3 + confidence > τ_C + eligible_to_trade)
//! - §7.2: Direction logic (S_dir = α1*sign(μ-m) + α2*sign(ΔV) + α3*sign(I10))
//! - §7.4: Risk doctrine (immediate exit on downgrade/refusal)
//! - §7.5: Urgency mapping (u = 0.4*d_perp + 0.4*fragility + 0.2*toxicity)
//!
//! The strategy consumes SLRT classification outputs and converts them
//! to trading intents. It does NOT recompute or modify regime logic.

use crate::canonical::{
    CONFIG_ENCODING_VERSION, CanonicalBytes, canonical_hash, encode_i8, encode_i64,
};
use crate::context::{FillNotification, StrategyContext};
use crate::output::{DecisionOutput, OrderIntent, Side};
use crate::{EventKind, ReplayEvent, Strategy};
use anyhow::Result;
use quantlaxmi_models::events::{CONFIDENCE_EXPONENT, CorrelationContext, DecisionEvent};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// Strategy name constant.
pub const SLR_NAME: &str = "slr";

/// Strategy version - tracks SLRT-GPU spec version.
pub const SLR_VERSION: &str = "1.2.0";

// =============================================================================
// Strategy Funnel Statistics (for tuning diagnostics)
// =============================================================================

/// Funnel statistics for SLR strategy gates.
/// Tracks how many events pass each gate for tuning diagnostics.
#[derive(Debug, Clone, Default)]
pub struct SlrFunnelStats {
    /// Total price events seen
    pub seen_events: u64,
    /// Events where regime = R3 (or R2 if allow_r2)
    pub seen_eligible: u64,
    /// Events passing confidence threshold
    pub seen_conf_ok: u64,
    /// Events passing d_perp minimum
    pub seen_d_perp_ok: u64,
    /// Events with direction inputs present
    pub seen_dir_inputs_ok: u64,
    /// Events with non-zero direction signal
    pub seen_dir_nonzero: u64,
    /// Events passing spread veto (Grid C v1)
    pub seen_spread_ok: u64,
    /// Intents emitted
    pub intents: u64,
}

impl SlrFunnelStats {
    /// Print funnel statistics.
    pub fn print(&self) {
        println!("\n=== SLR STRATEGY FUNNEL ===");
        println!("seen_events:      {}", self.seen_events);
        println!(
            "seen_eligible:    {} (regime R3 or R2 if allowed)",
            self.seen_eligible
        );
        println!(
            "seen_conf_ok:     {} (confidence >= tau_C)",
            self.seen_conf_ok
        );
        println!("seen_d_perp_ok:   {} (d_perp >= min)", self.seen_d_perp_ok);
        println!(
            "seen_dir_inputs:  {} (all direction inputs present)",
            self.seen_dir_inputs_ok
        );
        println!(
            "seen_dir_nonzero: {} (|S_dir| >= tau_dir_enter)",
            self.seen_dir_nonzero
        );
        println!(
            "seen_spread_ok:   {} (spread <= max_spread_bps)",
            self.seen_spread_ok
        );
        println!("intents:          {}", self.intents);
        println!("===========================\n");
    }
}

// =============================================================================
// Configuration (SEALED - matches SLRT-GPU v1.1 spec)
// =============================================================================

/// SLR Strategy Configuration.
///
/// Fixed-point policy: no f64 fields in config hashing.
/// All thresholds match SLRT-GPU v1.1 SEALED spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SlrConfig {
    // ===== Regime Thresholds (§6.6 SEALED) =====
    /// d_perp threshold for R3 (mantissa, exp -2). Default: 200 = 2.0
    pub tau_d_perp_mantissa: i64,
    /// Fragility threshold for R3 (mantissa, exp -2). Default: 60 = 0.6
    pub tau_fragility_mantissa: i64,
    /// FTI persist threshold (mantissa, exp -2). Default: 30 = 0.3
    pub tau_fti_persist_mantissa: i64,
    /// Toxicity persist threshold (mantissa, exp -2). Default: 30 = 0.3
    pub tau_toxicity_persist_mantissa: i64,
    /// Minimum confidence for trading (mantissa, exp -2). Default: 50 = 0.5
    pub tau_confidence_mantissa: i64,
    /// Threshold exponent (shared). Default: -2
    pub threshold_exponent: i8,

    // ===== Entry Gate Thresholds (Cost-Aware) =====
    /// Minimum fragility for entry (mantissa, exp -2). Default: 83 = 0.83 (p95)
    pub min_fragility_mantissa: i64,
    /// Minimum toxicity for entry (mantissa, exp -2). Default: 23 = 0.23 (p95)
    pub min_toxicity_mantissa: i64,

    // ===== Direction Weights (§7.2) =====
    /// Weight for microprice deviation: sign(μ - m). Default: 40 (0.4)
    pub alpha1_mantissa: i64,
    /// Weight for signed volume: sign(ΔV_signed). Default: 40 (0.4)
    pub alpha2_mantissa: i64,
    /// Weight for imbalance: sign(I10). Default: 20 (0.2)
    pub alpha3_mantissa: i64,
    /// Weight exponent. Default: -2
    pub alpha_exponent: i8,

    // ===== Direction Hysteresis (Anti-Churn) =====
    /// Minimum |S_dir| to enter (integer votes, 1-3). Default: 2 (requires 2/3 alignment)
    pub tau_dir_enter: i8,
    /// Minimum |S_dir| to maintain position (exit when below). Default: 1
    pub tau_dir_exit: i8,

    // ===== Urgency Weights (§7.5) =====
    /// Urgency weight for d_perp. Default: 40 (0.4)
    pub urgency_d_perp_mantissa: i64,
    /// Urgency weight for fragility. Default: 40 (0.4)
    pub urgency_fragility_mantissa: i64,
    /// Urgency weight for toxicity. Default: 20 (0.2)
    pub urgency_toxicity_mantissa: i64,
    /// Urgency exponent. Default: -2
    pub urgency_exponent: i8,

    // ===== Urgency → Execution Mapping (§7.5 SEALED) =====
    /// Slippage cap in ticks for aggressive orders. Default: 2
    pub slip_cap_ticks: i64,
    /// Tick size (mantissa, uses price_exponent). Default: 10 = $0.10 @ -2
    pub tick_size_mantissa: i64,
    /// Urgency threshold for aggressive crossing (mantissa, exp -2). Default: 80 = 0.80
    pub u_aggressive_mantissa: i64,

    // ===== Anti-Churn Controls =====
    /// Minimum hold time before exit allowed (ms). Default: 2000
    pub min_hold_ms: i64,
    /// Cooldown after exit before new entry allowed (ms). Default: 1000
    pub cooldown_after_exit_ms: i64,

    // ===== Position Sizing =====
    /// Base position size (mantissa). Default: 1_000_000 (0.01 BTC @ -8)
    pub base_position_mantissa: i64,
    /// Quantity exponent. Default: -8
    pub qty_exponent: i8,
    /// Price exponent. Default: -2
    pub price_exponent: i8,

    // ===== Trade Permission =====
    /// Allow trading in R2 (in addition to R3). Default: false
    pub allow_r2_trades: bool,
    /// Minimum d_perp for trading (prevents noise trades). Default: 680 (6.8 = p99)
    pub min_d_perp_mantissa: i64,

    // ===== Spread-Aware Entry Veto (Grid C v1) =====
    /// Maximum spread in bps for entry (veto if spread > this). Default: 30
    /// Computation: spread_bps = 10_000 * (ask - bid) / mid
    /// Only applies to entries; exits always allowed regardless of spread.
    pub max_spread_bps_entry: i64,
}

impl Default for SlrConfig {
    fn default() -> Self {
        Self {
            // §6.6 SEALED thresholds
            tau_d_perp_mantissa: 200,          // 2.0
            tau_fragility_mantissa: 60,        // 0.6
            tau_fti_persist_mantissa: 30,      // 0.3
            tau_toxicity_persist_mantissa: 30, // 0.3
            tau_confidence_mantissa: 50,       // 0.5
            threshold_exponent: -2,

            // Entry gate thresholds (cost-aware, based on p90 quantiles)
            // Using p90 instead of p95 to avoid over-filtering
            min_fragility_mantissa: 78, // 0.78 (p90)
            min_toxicity_mantissa: 11,  // 0.11 (p90)

            // §7.2 Direction weights
            alpha1_mantissa: 40, // 0.4
            alpha2_mantissa: 40, // 0.4
            alpha3_mantissa: 20, // 0.2
            alpha_exponent: -2,

            // Direction hysteresis (anti-churn)
            tau_dir_enter: 2, // Require 2/3 votes aligned to enter
            tau_dir_exit: 1,  // Exit when alignment drops below 1

            // §7.5 Urgency weights
            urgency_d_perp_mantissa: 40,    // 0.4
            urgency_fragility_mantissa: 40, // 0.4
            urgency_toxicity_mantissa: 20,  // 0.2
            urgency_exponent: -2,

            // §7.5 Urgency → execution mapping
            slip_cap_ticks: 2,         // 2 ticks slippage cap for aggressive
            tick_size_mantissa: 10,    // $0.10 tick @ -2 exponent
            u_aggressive_mantissa: 60, // 0.60 threshold for crossing (default)

            // Anti-churn controls
            min_hold_ms: 2000,            // 2 seconds minimum hold
            cooldown_after_exit_ms: 1000, // 1 second cooldown after exit

            // Position sizing
            base_position_mantissa: 1_000_000, // 0.01 BTC
            qty_exponent: -8,
            price_exponent: -2,

            // Trade permission
            allow_r2_trades: false,
            min_d_perp_mantissa: 430, // 4.3 (p95)

            // Spread-aware entry veto (Grid C v1)
            max_spread_bps_entry: 30, // 30 bps max spread for entry
        }
    }
}

impl SlrConfig {
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Convert mantissa to f64 using threshold_exponent.
    fn threshold_to_f64(&self, mantissa: i64) -> f64 {
        mantissa as f64 * 10f64.powi(self.threshold_exponent as i32)
    }

    /// Get τ_confidence as f64.
    pub fn tau_confidence(&self) -> f64 {
        self.threshold_to_f64(self.tau_confidence_mantissa)
    }

    /// Get min_d_perp as f64.
    pub fn min_d_perp(&self) -> f64 {
        self.threshold_to_f64(self.min_d_perp_mantissa)
    }

    /// Get min_fragility as f64.
    pub fn min_fragility(&self) -> f64 {
        self.threshold_to_f64(self.min_fragility_mantissa)
    }

    /// Get min_toxicity as f64.
    pub fn min_toxicity(&self) -> f64 {
        self.threshold_to_f64(self.min_toxicity_mantissa)
    }

    /// Get u_aggressive threshold as f64.
    pub fn u_aggressive(&self) -> f64 {
        self.threshold_to_f64(self.u_aggressive_mantissa)
    }
}

impl CanonicalBytes for SlrConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(CONFIG_ENCODING_VERSION);

        // Thresholds
        encode_i64(&mut buf, self.tau_d_perp_mantissa);
        encode_i64(&mut buf, self.tau_fragility_mantissa);
        encode_i64(&mut buf, self.tau_fti_persist_mantissa);
        encode_i64(&mut buf, self.tau_toxicity_persist_mantissa);
        encode_i64(&mut buf, self.tau_confidence_mantissa);
        encode_i8(&mut buf, self.threshold_exponent);

        // Entry gate thresholds (cost-aware)
        encode_i64(&mut buf, self.min_fragility_mantissa);
        encode_i64(&mut buf, self.min_toxicity_mantissa);

        // Direction weights
        encode_i64(&mut buf, self.alpha1_mantissa);
        encode_i64(&mut buf, self.alpha2_mantissa);
        encode_i64(&mut buf, self.alpha3_mantissa);
        encode_i8(&mut buf, self.alpha_exponent);

        // Direction hysteresis
        encode_i8(&mut buf, self.tau_dir_enter);
        encode_i8(&mut buf, self.tau_dir_exit);

        // Urgency weights
        encode_i64(&mut buf, self.urgency_d_perp_mantissa);
        encode_i64(&mut buf, self.urgency_fragility_mantissa);
        encode_i64(&mut buf, self.urgency_toxicity_mantissa);
        encode_i8(&mut buf, self.urgency_exponent);

        // Urgency → execution mapping
        encode_i64(&mut buf, self.slip_cap_ticks);
        encode_i64(&mut buf, self.tick_size_mantissa);
        encode_i64(&mut buf, self.u_aggressive_mantissa);

        // Anti-churn controls
        encode_i64(&mut buf, self.min_hold_ms);
        encode_i64(&mut buf, self.cooldown_after_exit_ms);

        // Position sizing
        encode_i64(&mut buf, self.base_position_mantissa);
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);

        // Trade permission
        buf.push(if self.allow_r2_trades { 1 } else { 0 });
        encode_i64(&mut buf, self.min_d_perp_mantissa);

        // Spread-aware entry veto (Grid C v1)
        encode_i64(&mut buf, self.max_spread_bps_entry);

        buf
    }
}

// =============================================================================
// SLRT Classification State (consumed from events or computed inline)
// =============================================================================

/// SLRT regime classification result (consumed from event payload).
/// This mirrors slrt-ref::RegimeClassification but is local to avoid dependency cycles.
#[derive(Debug, Clone, Default)]
struct SlrtClassification {
    /// Regime label: 0=R0, 1=R1, 2=R2, 3=R3
    regime: u8,
    /// Eligible to trade (computed by engine: regime == R3 && !refused)
    /// Strategy reads this directly to avoid drift with engine classification.
    eligible_to_trade_from_engine: bool,
    /// Effective confidence [0, 1]
    confidence: f64,
    /// Normalization penalty factor [0, 1]
    normalization_penalty: f64,
    /// Degraded reasons bitmask
    degraded_reasons: u32,
    /// d_perp (distance from subspace)
    d_perp: f64,
    /// Fragility score [0, 1]
    fragility: f64,
    /// FTI persistence [0, 1]
    fti_persist: f64,
    /// Toxicity [0, 1]
    toxicity: f64,
    /// Toxicity persistence [0, 1]
    toxicity_persist: f64,
    /// Whether frame was refused (structural invalidity)
    refused: bool,
    // Direction computation inputs (from features) - §7.2 SEALED
    // MUST be Option: missing inputs → veto entry (no silent 0.0 defaults)
    /// Microprice deviation: μ - m
    microprice_dev: Option<f64>,
    /// Signed volume
    signed_volume: Option<f64>,
    /// Book imbalance at level 10
    imbalance_10: Option<f64>,
}

impl SlrtClassification {
    /// Parse from event payload (expects SLRT WAL format).
    /// §7.2 direction inputs are Option - no silent defaults.
    fn from_payload(payload: &serde_json::Value) -> Option<Self> {
        let regime = payload.get("regime")?.as_str()?;
        let regime_num = match regime {
            "R0" => 0,
            "R1" => 1,
            "R2" => 2,
            "R3" => 3,
            _ => return None,
        };

        Some(Self {
            regime: regime_num,
            // Read eligible_to_trade directly from payload to avoid drift with engine
            eligible_to_trade_from_engine: payload
                .get("eligible_to_trade")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            confidence: payload.get("confidence")?.as_f64()?,
            normalization_penalty: payload
                .get("normalization_penalty")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0),
            degraded_reasons: payload
                .get("degraded_reasons")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            d_perp: payload.get("d_perp")?.as_f64()?,
            fragility: payload.get("fragility")?.as_f64()?,
            fti_persist: payload
                .get("fti_persist")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            toxicity: payload
                .get("toxicity")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            toxicity_persist: payload
                .get("toxicity_persist")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            refused: payload
                .get("refused")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            // §7.2 Direction inputs - MUST be Option, no silent 0.0 defaults
            microprice_dev: payload.get("microprice_dev").and_then(|v| v.as_f64()),
            signed_volume: payload.get("signed_volume").and_then(|v| v.as_f64()),
            imbalance_10: payload.get("imbalance_10").and_then(|v| v.as_f64()),
        })
    }

    /// Check if eligible to trade - reads directly from engine-computed value.
    /// This prevents drift between engine's regime classification and strategy's interpretation.
    fn eligible_to_trade(&self, _allow_r2: bool) -> bool {
        // Use the engine-computed value directly to avoid mismatch
        self.eligible_to_trade_from_engine
    }

    /// Check if all §7.2 direction inputs are present.
    /// If any are missing → veto entry (no silent poisoning).
    #[cfg(test)]
    fn has_direction_inputs(&self) -> bool {
        self.microprice_dev.is_some() && self.signed_volume.is_some() && self.imbalance_10.is_some()
    }
}

// =============================================================================
// Position State
// =============================================================================

#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Fields preserved for future SL/TP implementation
enum PositionState {
    #[default]
    Flat,
    Long {
        entry_ts_ms: i64,
        entry_price: i64,
        qty_mantissa: i64,
        entry_regime: u8,
    },
    Short {
        entry_ts_ms: i64,
        entry_price: i64,
        qty_mantissa: i64,
        entry_regime: u8,
    },
}

// =============================================================================
// Strategy Implementation
// =============================================================================

/// Parameters for creating a decision (refactored from 8 args).
struct DecisionParams<'a> {
    direction: i8,
    decision_type: &'a str,
    tag: &'a str,
    classification: &'a SlrtClassification,
    size: i64,
    urgency: f64,
}

pub struct SlrStrategy {
    config: SlrConfig,
    config_hash: String,
    position: PositionState,
    /// Last known classification (for exit decisions)
    last_classification: Option<SlrtClassification>,
    /// Timestamp of last exit (for cooldown enforcement)
    last_exit_ts_ms: i64,
    /// Funnel statistics for tuning diagnostics
    pub funnel: SlrFunnelStats,
}

impl SlrStrategy {
    pub fn new(config: SlrConfig) -> Self {
        let config_hash = canonical_hash(&config);
        Self {
            config,
            config_hash,
            position: PositionState::Flat,
            last_classification: None,
            last_exit_ts_ms: 0,
            funnel: SlrFunnelStats::default(),
        }
    }

    /// Get the funnel statistics (for printing at end of backtest).
    pub fn funnel_stats(&self) -> &SlrFunnelStats {
        &self.funnel
    }

    /// Compute direction signal per §7.2 with vote counting for hysteresis.
    /// Returns the vote count: sum of sign(μ-m) + sign(ΔV_signed) + sign(I10)
    /// Range: -3 to +3, where magnitude indicates alignment strength.
    /// Returns None if any direction input is missing (no silent poisoning).
    fn compute_direction_votes(&self, c: &SlrtClassification) -> Option<i8> {
        // §7.2 SEALED: All three inputs required - no silent defaults
        let microprice_dev = c.microprice_dev?;
        let signed_volume = c.signed_volume?;
        let imbalance_10 = c.imbalance_10?;

        // Count votes (each term contributes -1, 0, or +1)
        let vote1 = sign(microprice_dev) as i8;
        let vote2 = sign(signed_volume) as i8;
        let vote3 = sign(imbalance_10) as i8;

        Some(vote1 + vote2 + vote3)
    }

    /// Check if direction alignment is sufficient for entry (hysteresis).
    fn direction_ok_for_entry(&self, votes: i8) -> bool {
        votes.abs() >= self.config.tau_dir_enter
    }

    /// Check if direction alignment has collapsed enough to exit (hysteresis).
    fn direction_collapsed(&self, votes: i8) -> bool {
        votes.abs() < self.config.tau_dir_exit
    }

    /// Compute urgency per §7.5 SEALED.
    /// u = clip(0.4*d_perp_norm + 0.4*fragility + 0.2*toxicity, 0, 1)
    /// Returns value in [0, 1] for urgency mapping.
    fn compute_urgency(&self, c: &SlrtClassification) -> f64 {
        let scale = 10f64.powi(self.config.urgency_exponent as i32);
        let w_d = self.config.urgency_d_perp_mantissa as f64 * scale;
        let w_f = self.config.urgency_fragility_mantissa as f64 * scale;
        let w_t = self.config.urgency_toxicity_mantissa as f64 * scale;

        // d_perp is unbounded, fragility and toxicity are [0,1]
        // Normalize d_perp by tau_d_perp for comparable scale
        let tau_d = self
            .config
            .threshold_to_f64(self.config.tau_d_perp_mantissa);
        let d_perp_norm = (c.d_perp / tau_d).min(1.0); // Cap at threshold ratio

        // Clip to [0, 1] for urgency mapping thresholds
        (w_d * d_perp_norm + w_f * c.fragility + w_t * c.toxicity).clamp(0.0, 1.0)
    }

    /// Map urgency to limit price per §7.5 SEALED.
    /// - u < 0.30 → passive (non-crossing): bid for buy, ask for sell
    /// - 0.30 ≤ u < u_aggressive → join (top-of-book): same as passive
    /// - u ≥ u_aggressive → aggressive with slippage cap: crossing limit
    fn price_for_urgency(&self, side: Side, urgency: f64, bid: i64, ask: i64) -> i64 {
        let tick = self.config.tick_size_mantissa;
        let slip_cap = self.config.slip_cap_ticks.saturating_mul(tick);
        let u_aggressive = self.config.u_aggressive();

        if urgency < 0.30 {
            // Passive: non-crossing limit
            match side {
                Side::Buy => bid,
                Side::Sell => ask,
            }
        } else if urgency < u_aggressive {
            // Join: top-of-book (same price, strategy refresh handles joins)
            match side {
                Side::Buy => bid,
                Side::Sell => ask,
            }
        } else {
            // Aggressive: crossing limit with slippage cap
            match side {
                Side::Buy => ask.saturating_add(slip_cap),
                Side::Sell => bid.saturating_sub(slip_cap),
            }
        }
    }

    /// Compute position size using confidence.
    /// size = base_size × effective_confidence
    fn compute_size(&self, c: &SlrtClassification) -> i64 {
        let base = self.config.base_position_mantissa;
        let scaled = (base as f64 * c.confidence) as i64;
        scaled.max(1) // Never zero
    }

    /// Check if should exit position due to regime downgrade (§7.4).
    fn should_exit_on_downgrade(&self, c: &SlrtClassification) -> bool {
        // Exit if: refused, or regime dropped to R0/R1
        if c.refused {
            return true;
        }
        match &self.position {
            PositionState::Long { entry_regime, .. }
            | PositionState::Short { entry_regime, .. } => {
                // Exit if regime dropped below entry regime threshold
                c.regime < *entry_regime || c.regime <= 1
            }
            PositionState::Flat => false,
        }
    }

    fn create_decision(&self, ctx: &StrategyContext, params: DecisionParams<'_>) -> DecisionEvent {
        let decision_id = Uuid::new_v4();
        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;

        // Convert confidence to mantissa
        let confidence_scale = 10i64.pow((-CONFIDENCE_EXPONENT) as u32);
        let confidence_mantissa =
            (params.classification.confidence * confidence_scale as f64) as i64;

        DecisionEvent {
            ts: ctx.ts,
            decision_id,
            strategy_id: self.strategy_id(),
            symbol: ctx.symbol.to_string(),
            decision_type: params.decision_type.to_string(),
            direction: params.direction,
            target_qty_mantissa: params.size,
            qty_exponent: self.config.qty_exponent,
            reference_price_mantissa: mid_mantissa,
            price_exponent: self.config.price_exponent,
            market_snapshot: ctx.market.clone(),
            confidence_mantissa,
            metadata: serde_json::json!({
                "tag": params.tag,
                "policy": "slr_v1.2.0",
                // SLRT classification (auditable)
                "regime": format!("R{}", params.classification.regime),
                "confidence": params.classification.confidence,
                "raw_confidence": 1.0,
                "normalization_penalty": params.classification.normalization_penalty,
                "degraded_reasons": params.classification.degraded_reasons,
                "d_perp": params.classification.d_perp,
                "fragility": params.classification.fragility,
                "fti_persist": params.classification.fti_persist,
                "toxicity": params.classification.toxicity,
                "toxicity_persist": params.classification.toxicity_persist,
                "refused": params.classification.refused,
                "eligible_to_trade": params.classification.eligible_to_trade(self.config.allow_r2_trades),
                // Direction inputs (§7.2)
                "microprice_dev": params.classification.microprice_dev,
                "signed_volume": params.classification.signed_volume,
                "imbalance_10": params.classification.imbalance_10,
                // Urgency (§7.5)
                "urgency": params.urgency,
            }),
            ctx: CorrelationContext {
                run_id: Some(ctx.run_id.to_string()),
                venue: Some("paper".to_string()),
                strategy_id: Some(self.strategy_id()),
                ..Default::default()
            },
        }
    }

    /// Create an order intent with urgency-controlled limit price (§7.5 SEALED).
    /// All orders are limit orders - no market orders allowed.
    fn create_intent(
        &self,
        parent_decision_id: Uuid,
        ctx: &StrategyContext,
        side: Side,
        size: i64,
        urgency: f64,
        tag: &str,
    ) -> OrderIntent {
        let bid = ctx.market.bid_price_mantissa();
        let ask = ctx.market.ask_price_mantissa();
        let limit_price = self.price_for_urgency(side, urgency, bid, ask);

        OrderIntent {
            parent_decision_id,
            symbol: ctx.symbol.to_string(),
            side,
            qty_mantissa: size,
            qty_exponent: self.config.qty_exponent,
            limit_price_mantissa: Some(limit_price), // §7.5: Always limit, never market
            price_exponent: self.config.price_exponent,
            tag: Some(tag.to_string()),
        }
    }
}

impl Strategy for SlrStrategy {
    fn name(&self) -> &str {
        SLR_NAME
    }

    fn version(&self) -> &str {
        SLR_VERSION
    }

    fn config_hash(&self) -> String {
        self.config_hash.clone()
    }

    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        // Only process depth/quote events (need price snapshot)
        let is_price_event = matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth);
        if !is_price_event {
            return vec![];
        }

        // Track funnel: price events seen
        self.funnel.seen_events += 1;

        // Parse SLRT classification from event payload
        // Expected: event enriched with slrt-ref outputs
        let classification = match SlrtClassification::from_payload(&event.payload) {
            Some(c) => c,
            None => {
                // No classification in payload - try to use last known
                // In production, classification should always be present
                match &self.last_classification {
                    Some(c) => c.clone(),
                    None => return vec![],
                }
            }
        };

        // Store for reference
        self.last_classification = Some(classification.clone());

        let ts_ms = ctx.ts.timestamp_millis();
        let mid = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;
        let mut outputs = vec![];

        // =================================================================
        // EXIT CHECK (§7.4): Exit on downgrade, refusal, or direction collapse
        // =================================================================
        // Compute direction votes for exit check (direction collapse hysteresis)
        let direction_votes = self.compute_direction_votes(&classification);

        match &self.position {
            PositionState::Long {
                qty_mantissa,
                entry_ts_ms,
                ..
            } => {
                // Check min_hold before allowing any exit
                let held_ms = ts_ms - entry_ts_ms;
                let can_exit = held_ms >= self.config.min_hold_ms;

                let should_exit = self.should_exit_on_downgrade(&classification)
                    || (can_exit && direction_votes.is_some_and(|v| self.direction_collapsed(v)));

                if should_exit && can_exit {
                    let size = *qty_mantissa;
                    let urgency = self.compute_urgency(&classification);
                    let d = self.create_decision(
                        ctx,
                        DecisionParams {
                            direction: 0,
                            decision_type: "exit",
                            tag: "exit_long_downgrade",
                            classification: &classification,
                            size,
                            urgency,
                        },
                    );
                    let intent = self.create_intent(
                        d.decision_id,
                        ctx,
                        Side::Sell,
                        size,
                        urgency,
                        "exit_long_downgrade",
                    );
                    outputs.push(DecisionOutput::new(d, intent));
                    self.last_exit_ts_ms = ts_ms;
                    self.position = PositionState::Flat;
                    return outputs;
                }
            }
            PositionState::Short {
                qty_mantissa,
                entry_ts_ms,
                ..
            } => {
                // Check min_hold before allowing any exit
                let held_ms = ts_ms - entry_ts_ms;
                let can_exit = held_ms >= self.config.min_hold_ms;

                let should_exit = self.should_exit_on_downgrade(&classification)
                    || (can_exit && direction_votes.is_some_and(|v| self.direction_collapsed(v)));

                if should_exit && can_exit {
                    let size = *qty_mantissa;
                    let urgency = self.compute_urgency(&classification);
                    let d = self.create_decision(
                        ctx,
                        DecisionParams {
                            direction: 0,
                            decision_type: "exit",
                            tag: "exit_short_downgrade",
                            classification: &classification,
                            size,
                            urgency,
                        },
                    );
                    let intent = self.create_intent(
                        d.decision_id,
                        ctx,
                        Side::Buy,
                        size,
                        urgency,
                        "exit_short_downgrade",
                    );
                    outputs.push(DecisionOutput::new(d, intent));
                    self.last_exit_ts_ms = ts_ms;
                    self.position = PositionState::Flat;
                    return outputs;
                }
            }
            PositionState::Flat => {}
        }

        // =================================================================
        // ENTRY CHECK (§6.6 + Cost-Aware Gates): Only if eligible to trade
        // =================================================================
        if matches!(self.position, PositionState::Flat) {
            // Cooldown check: prevent re-entry too soon after exit
            let since_exit_ms = ts_ms - self.last_exit_ts_ms;
            if self.last_exit_ts_ms > 0 && since_exit_ms < self.config.cooldown_after_exit_ms {
                return vec![]; // Still in cooldown
            }

            // Check eligibility (regime R3, or R2 if allowed)
            if !classification.eligible_to_trade(self.config.allow_r2_trades) {
                return vec![];
            }
            self.funnel.seen_eligible += 1;

            // Check confidence threshold
            if classification.confidence < self.config.tau_confidence() {
                return vec![];
            }
            self.funnel.seen_conf_ok += 1;

            // Check minimum d_perp (p99 threshold for instability)
            if classification.d_perp < self.config.min_d_perp() {
                return vec![];
            }
            self.funnel.seen_d_perp_ok += 1;

            // NOTE: Fragility and toxicity gates removed for now.
            // d_perp is the primary instability indicator. Fragility/toxicity
            // are not well-correlated with d_perp, so AND-gating filters too much.
            // Churn is controlled via direction hysteresis + cooldown instead.

            // Compute direction with hysteresis (§7.2 SEALED)
            // Returns None if any direction input is missing → veto entry
            let votes = match direction_votes {
                Some(v) => {
                    self.funnel.seen_dir_inputs_ok += 1;
                    v
                }
                None => {
                    // §7.2 VETO: Missing direction inputs - no silent poisoning
                    return vec![];
                }
            };

            // Direction hysteresis: require >= tau_dir_enter votes aligned
            if !self.direction_ok_for_entry(votes) {
                return vec![]; // Insufficient direction alignment
            }
            self.funnel.seen_dir_nonzero += 1;

            // Spread veto (Grid C v1): reject entry if spread too wide
            // spread_bps = 10_000 * (ask - bid) / mid
            // Only applies to entries; exits always allowed.
            let bid = ctx.market.bid_price_mantissa();
            let ask = ctx.market.ask_price_mantissa();
            if mid > 0 {
                let spread_bps = 10_000 * (ask - bid) / mid;
                if spread_bps > self.config.max_spread_bps_entry {
                    return vec![]; // Spread too wide for entry
                }
            }
            self.funnel.seen_spread_ok += 1;

            // Convert votes to direction sign
            let direction = if votes > 0 { 1i8 } else { -1i8 };

            // Compute size and urgency
            let size = self.compute_size(&classification);
            let urgency = self.compute_urgency(&classification);

            if direction > 0 {
                // LONG entry
                let d = self.create_decision(
                    ctx,
                    DecisionParams {
                        direction: 1,
                        decision_type: "entry",
                        tag: "entry_long",
                        classification: &classification,
                        size,
                        urgency,
                    },
                );
                let intent =
                    self.create_intent(d.decision_id, ctx, Side::Buy, size, urgency, "entry_long");
                outputs.push(DecisionOutput::new(d, intent));
                self.funnel.intents += 1;

                self.position = PositionState::Long {
                    entry_ts_ms: ts_ms,
                    entry_price: mid,
                    qty_mantissa: size,
                    entry_regime: classification.regime,
                };
            } else {
                // SHORT entry
                let d = self.create_decision(
                    ctx,
                    DecisionParams {
                        direction: -1,
                        decision_type: "entry",
                        tag: "entry_short",
                        classification: &classification,
                        size,
                        urgency,
                    },
                );
                let intent = self.create_intent(
                    d.decision_id,
                    ctx,
                    Side::Sell,
                    size,
                    urgency,
                    "entry_short",
                );
                outputs.push(DecisionOutput::new(d, intent));
                self.funnel.intents += 1;

                self.position = PositionState::Short {
                    entry_ts_ms: ts_ms,
                    entry_price: mid,
                    qty_mantissa: size,
                    entry_regime: classification.regime,
                };
            }
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, ctx: &StrategyContext) {
        let fill_ts_ms = ctx.ts.timestamp_millis();
        let fill_price = fill.price_mantissa;

        // Update position based on fill
        match (&self.position, fill.side) {
            (PositionState::Flat, Side::Buy) => {
                // Entry fill for long
                self.position = PositionState::Long {
                    entry_ts_ms: fill_ts_ms,
                    entry_price: fill_price,
                    qty_mantissa: fill.qty_mantissa,
                    entry_regime: self
                        .last_classification
                        .as_ref()
                        .map(|c| c.regime)
                        .unwrap_or(0),
                };
            }
            (PositionState::Flat, Side::Sell) => {
                // Entry fill for short
                self.position = PositionState::Short {
                    entry_ts_ms: fill_ts_ms,
                    entry_price: fill_price,
                    qty_mantissa: fill.qty_mantissa,
                    entry_regime: self
                        .last_classification
                        .as_ref()
                        .map(|c| c.regime)
                        .unwrap_or(0),
                };
            }
            (PositionState::Long { .. }, Side::Sell) => {
                // Exit fill from long
                self.position = PositionState::Flat;
            }
            (PositionState::Short { .. }, Side::Buy) => {
                // Exit fill from short
                self.position = PositionState::Flat;
            }
            _ => {
                // Unexpected fill - log but don't panic
            }
        }
    }

    fn print_diagnostics(&self) {
        self.funnel.print();
    }
}

/// Factory function for registry.
pub fn slr_factory(config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
    let config = match config_path {
        Some(path) => SlrConfig::from_toml(path)?,
        None => SlrConfig::default(),
    };
    Ok(Box::new(SlrStrategy::new(config)))
}

/// Sign function for direction computation.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_hash_deterministic() {
        let c1 = SlrConfig::default();
        let c2 = SlrConfig::default();
        assert_eq!(c1.canonical_bytes(), c2.canonical_bytes());
        assert_eq!(canonical_hash(&c1), canonical_hash(&c2));
    }

    #[test]
    fn test_direction_votes_all_positive() {
        let strategy = SlrStrategy::new(SlrConfig::default());
        let c = SlrtClassification {
            microprice_dev: Some(10.0),
            signed_volume: Some(100.0),
            imbalance_10: Some(0.5),
            ..Default::default()
        };
        // All 3 votes positive → +3
        assert_eq!(strategy.compute_direction_votes(&c), Some(3));
    }

    #[test]
    fn test_direction_votes_all_negative() {
        let strategy = SlrStrategy::new(SlrConfig::default());
        let c = SlrtClassification {
            microprice_dev: Some(-10.0),
            signed_volume: Some(-100.0),
            imbalance_10: Some(-0.5),
            ..Default::default()
        };
        // All 3 votes negative → -3
        assert_eq!(strategy.compute_direction_votes(&c), Some(-3));
    }

    #[test]
    fn test_direction_votes_mixed() {
        let strategy = SlrStrategy::new(SlrConfig::default());
        // +10 -> +1, -100 -> -1, +0.5 -> +1 = 1 (net votes)
        let c = SlrtClassification {
            microprice_dev: Some(10.0),
            signed_volume: Some(-100.0),
            imbalance_10: Some(0.5),
            ..Default::default()
        };
        assert_eq!(strategy.compute_direction_votes(&c), Some(1));
    }

    #[test]
    fn test_direction_votes_missing_inputs_returns_none() {
        // §7.2 SEALED: Missing direction inputs → veto entry (no silent poisoning)
        let strategy = SlrStrategy::new(SlrConfig::default());

        // Missing microprice_dev
        let c1 = SlrtClassification {
            microprice_dev: None,
            signed_volume: Some(100.0),
            imbalance_10: Some(0.5),
            ..Default::default()
        };
        assert_eq!(strategy.compute_direction_votes(&c1), None);

        // Missing signed_volume
        let c2 = SlrtClassification {
            microprice_dev: Some(10.0),
            signed_volume: None,
            imbalance_10: Some(0.5),
            ..Default::default()
        };
        assert_eq!(strategy.compute_direction_votes(&c2), None);

        // Missing imbalance_10
        let c3 = SlrtClassification {
            microprice_dev: Some(10.0),
            signed_volume: Some(100.0),
            imbalance_10: None,
            ..Default::default()
        };
        assert_eq!(strategy.compute_direction_votes(&c3), None);

        // All missing
        let c4 = SlrtClassification {
            microprice_dev: None,
            signed_volume: None,
            imbalance_10: None,
            ..Default::default()
        };
        assert_eq!(strategy.compute_direction_votes(&c4), None);
    }

    #[test]
    fn test_direction_hysteresis() {
        let strategy = SlrStrategy::new(SlrConfig::default());
        // Default: tau_dir_enter = 2, tau_dir_exit = 1

        // 3 votes → ok for entry
        assert!(strategy.direction_ok_for_entry(3));
        assert!(strategy.direction_ok_for_entry(-3));

        // 2 votes → ok for entry
        assert!(strategy.direction_ok_for_entry(2));
        assert!(strategy.direction_ok_for_entry(-2));

        // 1 vote → NOT ok for entry (< tau_dir_enter=2)
        assert!(!strategy.direction_ok_for_entry(1));
        assert!(!strategy.direction_ok_for_entry(-1));

        // 0 votes → NOT ok for entry
        assert!(!strategy.direction_ok_for_entry(0));

        // Exit collapse: 0 votes collapses (< tau_dir_exit=1)
        assert!(strategy.direction_collapsed(0));

        // 1+ votes does not collapse
        assert!(!strategy.direction_collapsed(1));
        assert!(!strategy.direction_collapsed(-1));
        assert!(!strategy.direction_collapsed(2));
    }

    #[test]
    fn test_eligible_to_trade() {
        // Strategy reads eligible_to_trade directly from engine-computed field
        // to avoid drift with engine's regime classification
        let c = SlrtClassification {
            regime: 3,
            refused: false,
            eligible_to_trade_from_engine: true,
            ..Default::default()
        };
        assert!(c.eligible_to_trade(false)); // Engine says eligible
        assert!(c.eligible_to_trade(true)); // Engine says eligible

        let c2 = SlrtClassification {
            regime: 2,
            refused: false,
            eligible_to_trade_from_engine: false, // Engine says R2 not eligible
            ..Default::default()
        };
        assert!(!c2.eligible_to_trade(false)); // Engine says not eligible
        assert!(!c2.eligible_to_trade(true)); // Engine says not eligible

        let c3 = SlrtClassification {
            regime: 3,
            refused: true,
            eligible_to_trade_from_engine: false, // Engine refused it
            ..Default::default()
        };
        assert!(!c3.eligible_to_trade(false)); // Engine says not eligible
        assert!(!c3.eligible_to_trade(true)); // Engine says not eligible
    }

    #[test]
    fn test_urgency_computation() {
        let strategy = SlrStrategy::new(SlrConfig::default());
        let c = SlrtClassification {
            d_perp: 2.0, // At threshold
            fragility: 0.6,
            toxicity: 0.5,
            ..Default::default()
        };
        let u = strategy.compute_urgency(&c);
        // u = 0.4 * (2.0/2.0) + 0.4 * 0.6 + 0.2 * 0.5
        // u = 0.4 * 1.0 + 0.24 + 0.1 = 0.74
        assert!((u - 0.74).abs() < 0.01);
    }

    #[test]
    fn test_size_with_confidence() {
        let strategy = SlrStrategy::new(SlrConfig::default());
        let c = SlrtClassification {
            confidence: 0.8,
            ..Default::default()
        };
        let size = strategy.compute_size(&c);
        // base = 1_000_000, size = 1_000_000 * 0.8 = 800_000
        assert_eq!(size, 800_000);
    }

    #[test]
    fn test_urgency_maps_to_limit_prices() {
        // §7.5 SEALED: Urgency controls execution via limit prices
        let strategy = SlrStrategy::new(SlrConfig::default());
        // Default config: tick_size_mantissa = 10, slip_cap_ticks = 2
        // slip_cap = 2 * 10 = 20

        let bid = 8300000; // $83,000.00 @ -2
        let ask = 8300100; // $83,001.00 @ -2

        // Passive (u < 0.30): non-crossing
        // Buy → limit at bid, Sell → limit at ask
        assert_eq!(strategy.price_for_urgency(Side::Buy, 0.20, bid, ask), bid);
        assert_eq!(strategy.price_for_urgency(Side::Sell, 0.20, bid, ask), ask);

        // Join (0.30 ≤ u < 0.60): top-of-book (same as passive)
        assert_eq!(strategy.price_for_urgency(Side::Buy, 0.45, bid, ask), bid);
        assert_eq!(strategy.price_for_urgency(Side::Sell, 0.45, bid, ask), ask);

        // Aggressive (u ≥ 0.60): crossing with slippage cap
        // Buy → ask + slip_cap, Sell → bid - slip_cap
        assert_eq!(
            strategy.price_for_urgency(Side::Buy, 0.80, bid, ask),
            ask + 20
        );
        assert_eq!(
            strategy.price_for_urgency(Side::Sell, 0.80, bid, ask),
            bid - 20
        );
    }

    #[test]
    fn test_has_direction_inputs() {
        // All present
        let c1 = SlrtClassification {
            microprice_dev: Some(1.0),
            signed_volume: Some(1.0),
            imbalance_10: Some(1.0),
            ..Default::default()
        };
        assert!(c1.has_direction_inputs());

        // One missing
        let c2 = SlrtClassification {
            microprice_dev: None,
            signed_volume: Some(1.0),
            imbalance_10: Some(1.0),
            ..Default::default()
        };
        assert!(!c2.has_direction_inputs());

        // All missing (default)
        let c3 = SlrtClassification::default();
        assert!(!c3.has_direction_inputs());
    }
}
