//! FundingAlignedMomentum Strategy (FAM-1) v0.1.0
//!
//! Combines:
//! - Funding bias direction (carry / positioning context)
//! - Momentum timing (entry only when price agrees)
//! - Grassmann gate (geometric lifting / regime filter)
//!
//! Notes on Grassmann gate:
//! - This is a deterministic "Grassmann-lite" gate in v0.1.0.
//! - It uses a lifted feature window and computes regime via
//!   directional efficiency + activity (vol proxy).
//! - Config includes Grassmann parameters (k, feature_dim, distance threshold)
//!   to lock in canonical hashing and enable future upgrade to true Gr(k,n)
//!   prototype matching without changing config schema.
//!
//! Strategy logic summary:
//! 1) Funding bias:
//!    - If funding > +threshold => bias SHORT
//!    - If funding < -threshold => bias LONG
//! 2) Momentum timing (bps over window):
//!    - Enter SHORT only if ret_bps <= -momentum_threshold_bps
//!    - Enter LONG only if ret_bps >= +momentum_threshold_bps
//! 3) Grassmann gate:
//!    - If gate says CHOP (or Unknown treated as CHOP) => block entries
//!    - Optional: exit on CHOP (regime_exit_on_chop)
//! 4) Exit/flip hysteresis:
//!    - Exit LONG when funding > -exit_band
//!    - Exit SHORT when funding < +exit_band
//!    - Flip if opposite entry condition hit (stable order: exit then entry)

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
pub const FUNDING_ALIGNED_MOMENTUM_NAME: &str = "funding_aligned_momentum";

/// Strategy version.
pub const FUNDING_ALIGNED_MOMENTUM_VERSION: &str = "0.1.0";

fn default_true() -> bool {
    true
}

/// Regime labels emitted by the Grassmann gate.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RegimeLabel {
    Trend,
    Chop,
    Unknown,
}

/// Grassmann / geometric-lifting gate configuration.
///
/// v0.1.0 uses a deterministic "Grassmann-lite" classifier:
/// - maintains a lifted rolling feature window
/// - computes directional efficiency + activity proxy
/// - maps to Trend/Chop/Unknown
///
/// The schema intentionally includes Grassmann parameters (k,n,threshold)
/// to support future upgrade to true Gr(k,n) prototype matching without
/// changing the config schema or param hash semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrassmannGateConfig {
    /// Enable the gate. If false, FAM trades purely on funding+momentum.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Feature sampling interval in milliseconds (e.g., 1000).
    pub sample_interval_ms: u32,

    /// Rolling window length in samples (e.g., 60).
    pub window_len: u32,

    /// Declared feature dimension (for canonical hash + forward compatibility).
    pub feature_dim: u32,

    /// Declared subspace dimension k (for canonical hash + forward compatibility).
    pub subspace_k: u32,

    /// Distance threshold (mantissa/exponent).
    ///
    /// In v0.1.0 (Grassmann-lite), we interpret this as an efficiency threshold:
    /// efficiency_mantissa / 10^(-distance_threshold_exponent)
    /// where efficiency is in [0,1] scaled by exponent.
    pub distance_threshold_mantissa: i64,
    pub distance_threshold_exponent: i8,

    /// If true: Unknown => treat as Chop (block entries).
    #[serde(default = "default_true")]
    pub unknown_is_chop: bool,

    /// Optional future: prototypes file for true Grassmann nearest-prototype matching.
    #[serde(default)]
    pub prototypes_path: Option<String>,

    /// If true: when regime flips to Chop, exit any open position.
    #[serde(default)]
    pub regime_exit_on_chop: bool,
}

impl Default for GrassmannGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_interval_ms: 1000,
            window_len: 60,
            feature_dim: 4, // we keep 4 lifted features in v0.1.0
            subspace_k: 2,
            // efficiency threshold ~0.35 (mantissa 35, exp -2)
            distance_threshold_mantissa: 35,
            distance_threshold_exponent: -2,
            unknown_is_chop: true,
            prototypes_path: None,
            regime_exit_on_chop: true,
        }
    }
}

/// Configuration for FundingAlignedMomentum strategy.
///
/// Fixed-point policy: no f64 fields in config hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingAlignedMomentumConfig {
    // ===== Funding bias (direction) =====
    pub funding_threshold_mantissa: i64,
    pub funding_threshold_exponent: i8,

    /// Exit band for hysteresis (same exponent as funding_threshold_exponent).
    /// Default: funding_threshold_mantissa / 2
    #[serde(default)]
    pub exit_band_mantissa: Option<i64>,

    // ===== Momentum timing =====
    /// Momentum window in seconds (e.g., 30, 60, 120).
    pub momentum_window_secs: u32,

    /// Momentum threshold in basis points (bps) for entry.
    /// Example: 10 => 0.10% move over momentum_window_secs.
    pub momentum_threshold_bps: i64,

    /// Position size (mantissa, uses qty_exponent).
    pub position_size_mantissa: i64,
    pub qty_exponent: i8,

    /// Price exponent (typically -2 for USD prices).
    pub price_exponent: i8,

    /// Allow trading on SpotQuote events (in addition to PerpQuote/PerpDepth).
    #[serde(default)]
    pub trade_on_spot_quotes: bool,

    // ===== Anti-churn constraints (Fix Set A) =====
    /// Minimum hold time in seconds after entry before allowing funding-based exits.
    /// Does NOT block stop-loss or take-profit exits.
    /// Default: 300 (5 minutes)
    #[serde(default = "default_min_hold_secs")]
    pub min_hold_secs: u32,

    /// Cooldown in seconds after any exit before allowing new entries.
    /// Default: 120 (2 minutes)
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u32,

    // ===== Price-based risk controls (Fix Set B) =====
    /// Stop-loss in basis points from entry price.
    /// 0 = disabled. Default: 60 bps
    #[serde(default = "default_stop_loss_bps")]
    pub stop_loss_bps: i64,

    /// Take-profit in basis points from entry price.
    /// 0 = disabled. Default: 40 bps
    #[serde(default = "default_take_profit_bps")]
    pub take_profit_bps: i64,

    /// Arming delay in seconds after entry fill before SL/TP can trigger.
    /// Prevents immediate microstructure churn (spread/fees).
    /// Default: 20s
    #[serde(default = "default_sl_tp_arm_secs")]
    pub sl_tp_arm_secs: u32,

    /// Grassmann / geometric-lifting gate.
    #[serde(default)]
    pub grassmann: GrassmannGateConfig,
}

fn default_min_hold_secs() -> u32 {
    300
}

fn default_cooldown_secs() -> u32 {
    120
}

fn default_stop_loss_bps() -> i64 {
    60
}

fn default_take_profit_bps() -> i64 {
    40
}

fn default_sl_tp_arm_secs() -> u32 {
    20
}

impl Default for FundingAlignedMomentumConfig {
    fn default() -> Self {
        Self {
            // 100 @ -6 => 0.0001
            funding_threshold_mantissa: 100,
            funding_threshold_exponent: -6,
            exit_band_mantissa: None,

            momentum_window_secs: 60,
            momentum_threshold_bps: 10,

            // 0.01 BTC @ -8
            position_size_mantissa: 1_000_000,
            qty_exponent: -8,
            price_exponent: -2,

            trade_on_spot_quotes: false,

            // Anti-churn (Fix Set A)
            min_hold_secs: 300, // 5 minutes
            cooldown_secs: 120, // 2 minutes

            // Risk controls (Fix Set B)
            stop_loss_bps: 60,
            take_profit_bps: 40,
            sl_tp_arm_secs: 20,

            grassmann: GrassmannGateConfig::default(),
        }
    }
}

impl FundingAlignedMomentumConfig {
    pub fn effective_exit_band(&self) -> i64 {
        self.exit_band_mantissa
            .unwrap_or(self.funding_threshold_mantissa / 2)
    }

    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}

impl CanonicalBytes for FundingAlignedMomentumConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(CONFIG_ENCODING_VERSION);

        // Funding bias
        encode_i64(&mut buf, self.funding_threshold_mantissa);
        encode_i8(&mut buf, self.funding_threshold_exponent);
        encode_i64(&mut buf, self.effective_exit_band());

        // Momentum
        encode_i64(&mut buf, self.momentum_window_secs as i64);
        encode_i64(&mut buf, self.momentum_threshold_bps);

        // Sizing / price units
        encode_i64(&mut buf, self.position_size_mantissa);
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);

        // Event source
        buf.push(if self.trade_on_spot_quotes { 1 } else { 0 });

        // Anti-churn constraints (Fix Set A)
        encode_i64(&mut buf, self.min_hold_secs as i64);
        encode_i64(&mut buf, self.cooldown_secs as i64);

        // Risk controls (Fix Set B)
        encode_i64(&mut buf, self.stop_loss_bps);
        encode_i64(&mut buf, self.take_profit_bps);
        encode_i64(&mut buf, self.sl_tp_arm_secs as i64);

        // Grassmann gate (fixed order)
        let g = &self.grassmann;
        buf.push(if g.enabled { 1 } else { 0 });
        encode_i64(&mut buf, g.sample_interval_ms as i64);
        encode_i64(&mut buf, g.window_len as i64);
        encode_i64(&mut buf, g.feature_dim as i64);
        encode_i64(&mut buf, g.subspace_k as i64);
        encode_i64(&mut buf, g.distance_threshold_mantissa);
        encode_i8(&mut buf, g.distance_threshold_exponent);
        buf.push(if g.unknown_is_chop { 1 } else { 0 });
        buf.push(if g.regime_exit_on_chop { 1 } else { 0 });

        // prototypes_path: canonicalize as presence flag + bytes if present
        match &g.prototypes_path {
            None => buf.push(0),
            Some(s) => {
                buf.push(1);
                // length-prefixed string bytes (deterministic)
                encode_i64(&mut buf, s.len() as i64);
                buf.extend_from_slice(s.as_bytes());
            }
        }

        buf
    }
}

// =============================================================================
// Grassmann gate state (v0.1.0: deterministic lifting + efficiency classifier)
// =============================================================================

struct GrassmannGateState {
    last_sample_ts_ms: Option<i64>,

    // Rolling lifted features
    // We keep only what we need to compute efficiency/activity deterministically:
    // - ret_bps per sample (signed)
    // - abs_ret_bps per sample
    // Optionally extend later to additional dims and true subspace computation.
    ret_bps_ring: Vec<i64>,
    abs_ret_bps_ring: Vec<i64>,
    write_idx: usize,
    filled: usize,

    regime: RegimeLabel,
    last_efficiency_mantissa: i64, // scaled using distance_threshold_exponent
}

impl GrassmannGateState {
    fn new(window_len: usize) -> Self {
        Self {
            last_sample_ts_ms: None,
            ret_bps_ring: vec![0; window_len],
            abs_ret_bps_ring: vec![0; window_len],
            write_idx: 0,
            filled: 0,
            regime: RegimeLabel::Unknown,
            last_efficiency_mantissa: 0,
        }
    }

    fn window_len(&self) -> usize {
        self.ret_bps_ring.len()
    }
}

// =============================================================================
// Position State Machine
// =============================================================================

/// Explicit position state for correct anti-churn logic.
///
/// Transitions:
/// - Flat → (emit entry) → PendingEntry
/// - PendingEntry → (entry fill) → InPosition
/// - PendingEntry → (reject/timeout) → Flat
/// - InPosition → (emit exit) → PendingExit
/// - PendingExit → (exit fill) → Flat
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Fields preserved for future timeout handling and logging
enum PositionState {
    /// No position, no pending orders.
    #[default]
    Flat,

    /// Entry signal emitted, awaiting fill confirmation.
    PendingEntry {
        submit_ts_ms: i64,
        ref_price_mid: i64,
        direction: i8, // 1 = long, -1 = short
    },

    /// Position confirmed via fill. min_hold starts from fill_ts_ms.
    InPosition {
        fill_ts_ms: i64,
        fill_price: i64,
        direction: i8, // 1 = long, -1 = short
        qty_mantissa: i64,
    },

    /// Exit signal emitted, awaiting fill confirmation.
    PendingExit {
        submit_ts_ms: i64,
        entry_fill_ts_ms: i64, // preserved for min_hold check
        entry_fill_price: i64, // preserved for reference
        direction: i8,
        qty_mantissa: i64,
    },
}

// =============================================================================
// Strategy implementation
// =============================================================================

pub struct FundingAlignedMomentumStrategy {
    config: FundingAlignedMomentumConfig,
    config_hash: String,

    // Funding state (mantissa in config exponent space)
    current_funding_rate_mantissa: i64,

    // Position state machine (replaces loose tracking fields)
    position_state: PositionState,

    /// Timestamp (ms) of last confirmed exit fill. Used for cooldown.
    last_exit_fill_ts_ms: Option<i64>,

    // Price sampling for momentum
    // Store mid prices with timestamps (ms) to compute ret over N secs.
    mid_ts_ms: Vec<i64>,
    mid_mantissa: Vec<i64>,

    // Grassmann gate state
    gate: GrassmannGateState,

    // Last known PERP bid/ask (for SL/TP checks).
    // CRITICAL: ctx.market can contain SPOT prices when spot events dominate.
    // SL/TP must use PERP prices since we trade perps!
    last_perp_bid: i64,
    last_perp_ask: i64,
}

impl FundingAlignedMomentumStrategy {
    pub fn new(config: FundingAlignedMomentumConfig) -> Self {
        let config_hash = canonical_hash(&config);
        let wl = config.grassmann.window_len.max(1) as usize;

        Self {
            config,
            config_hash,
            current_funding_rate_mantissa: 0,
            position_state: PositionState::Flat,
            last_exit_fill_ts_ms: None,
            mid_ts_ms: Vec::with_capacity(512),
            mid_mantissa: Vec::with_capacity(512),
            gate: GrassmannGateState::new(wl),
            last_perp_bid: 0,
            last_perp_ask: 0,
        }
    }

    // =========================================================================
    // Position state helpers
    // =========================================================================

    /// Returns true if we're in a confirmed position (not pending).
    fn is_in_position(&self) -> bool {
        matches!(self.position_state, PositionState::InPosition { .. })
    }

    /// Returns true if we have a confirmed LONG position.
    fn is_confirmed_long(&self) -> bool {
        matches!(
            self.position_state,
            PositionState::InPosition { direction: 1, .. }
        )
    }

    /// Returns true if we have a confirmed SHORT position.
    fn is_confirmed_short(&self) -> bool {
        matches!(
            self.position_state,
            PositionState::InPosition { direction: -1, .. }
        )
    }

    /// Returns true if we're flat (no position, no pending orders).
    fn is_flat(&self) -> bool {
        matches!(self.position_state, PositionState::Flat)
    }

    /// Returns true if we have a pending entry (awaiting fill).
    fn has_pending_entry(&self) -> bool {
        matches!(self.position_state, PositionState::PendingEntry { .. })
    }

    /// Returns true if we have a pending exit (awaiting fill).
    fn has_pending_exit(&self) -> bool {
        matches!(self.position_state, PositionState::PendingExit { .. })
    }

    /// Get entry fill timestamp if in position or pending exit.
    fn get_entry_fill_ts(&self) -> Option<i64> {
        match &self.position_state {
            PositionState::InPosition { fill_ts_ms, .. } => Some(*fill_ts_ms),
            PositionState::PendingExit {
                entry_fill_ts_ms, ..
            } => Some(*entry_fill_ts_ms),
            _ => None,
        }
    }

    /// Get entry fill price if in position or pending exit.
    fn get_entry_fill_price(&self) -> Option<i64> {
        match &self.position_state {
            PositionState::InPosition { fill_price, .. } => Some(*fill_price),
            PositionState::PendingExit {
                entry_fill_price, ..
            } => Some(*entry_fill_price),
            _ => None,
        }
    }

    /// Check if we're within the minimum hold period.
    /// Only applies to confirmed positions (uses fill timestamp).
    fn within_min_hold(&self, now_ms: i64) -> bool {
        if let Some(fill_ts) = self.get_entry_fill_ts() {
            let hold_ms = self.config.min_hold_secs as i64 * 1000;
            now_ms - fill_ts < hold_ms
        } else {
            false
        }
    }

    /// Check if we're within cooldown period (cannot enter new positions).
    /// Uses the timestamp of the last confirmed exit fill.
    fn within_cooldown(&self, now_ms: i64) -> bool {
        if let Some(exit_ts) = self.last_exit_fill_ts_ms {
            let cooldown_ms = self.config.cooldown_secs as i64 * 1000;
            now_ms - exit_ts < cooldown_ms
        } else {
            false
        }
    }

    /// Check if SL/TP is armed (arming delay has passed since entry fill).
    fn sl_tp_armed(&self, now_ms: i64) -> bool {
        let Some(fill_ts) = self.get_entry_fill_ts() else {
            return false;
        };
        let arm_ms = self.config.sl_tp_arm_secs as i64 * 1000;
        now_ms - fill_ts >= arm_ms
    }

    /// Check if stop-loss is triggered. Uses executable prices (bid/ask).
    fn check_stop_loss(&self, bid: i64, ask: i64) -> bool {
        if self.config.stop_loss_bps == 0 {
            return false;
        }
        let Some(entry_price) = self.get_entry_fill_price() else {
            return false;
        };
        let sl_distance =
            (entry_price as i128 * self.config.stop_loss_bps as i128 / 10_000i128) as i64;

        if self.is_confirmed_long() {
            // Exit long by selling at BID
            bid <= entry_price - sl_distance
        } else if self.is_confirmed_short() {
            // Exit short by buying at ASK
            ask >= entry_price + sl_distance
        } else {
            false
        }
    }

    /// Check if take-profit is triggered. Uses executable prices (bid/ask).
    fn check_take_profit(&self, bid: i64, ask: i64) -> bool {
        if self.config.take_profit_bps == 0 {
            return false;
        }
        let Some(entry_price) = self.get_entry_fill_price() else {
            return false;
        };
        let tp_distance =
            (entry_price as i128 * self.config.take_profit_bps as i128 / 10_000i128) as i64;

        if self.is_confirmed_long() {
            // Exit long by selling at BID
            bid >= entry_price + tp_distance
        } else if self.is_confirmed_short() {
            // Exit short by buying at ASK
            ask <= entry_price - tp_distance
        } else {
            false
        }
    }

    // =========================================================================
    // State transition helpers
    // =========================================================================

    /// Transition from Flat to PendingEntry when emitting an entry signal.
    fn transition_to_pending_entry(
        &mut self,
        submit_ts_ms: i64,
        ref_price_mid: i64,
        direction: i8,
    ) {
        self.position_state = PositionState::PendingEntry {
            submit_ts_ms,
            ref_price_mid,
            direction,
        };
    }

    /// Transition from InPosition to PendingExit when emitting an exit signal.
    fn transition_to_pending_exit(&mut self, submit_ts_ms: i64) {
        if let PositionState::InPosition {
            fill_ts_ms,
            fill_price,
            direction,
            qty_mantissa,
        } = self.position_state
        {
            self.position_state = PositionState::PendingExit {
                submit_ts_ms,
                entry_fill_ts_ms: fill_ts_ms,
                entry_fill_price: fill_price,
                direction,
                qty_mantissa,
            };
        }
    }

    fn create_decision(
        &self,
        ctx: &StrategyContext,
        direction: i8,
        decision_type: &str,
        tag: &str,
        extra: serde_json::Value,
    ) -> DecisionEvent {
        let decision_id = Uuid::new_v4();
        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;
        let confidence_mantissa = 10i64.pow((-CONFIDENCE_EXPONENT) as u32); // 1.0

        DecisionEvent {
            ts: ctx.ts,
            decision_id,
            strategy_id: self.strategy_id(),
            symbol: ctx.symbol.to_string(),
            decision_type: decision_type.to_string(),
            direction,
            target_qty_mantissa: self.config.position_size_mantissa,
            qty_exponent: self.config.qty_exponent,
            reference_price_mantissa: mid_mantissa,
            price_exponent: self.config.price_exponent,
            market_snapshot: ctx.market.clone(),
            confidence_mantissa,
            metadata: serde_json::json!({
                "tag": tag,
                "policy": "fam1_grassmann_lite_v0.1.0",
                "funding_rate_mantissa": self.current_funding_rate_mantissa,
                "funding_rate_exponent": self.config.funding_threshold_exponent,
                "funding_threshold_mantissa": self.config.funding_threshold_mantissa,
                "exit_band_mantissa": self.config.effective_exit_band(),
                "momentum_window_secs": self.config.momentum_window_secs,
                "momentum_threshold_bps": self.config.momentum_threshold_bps,
                "gate_enabled": self.config.grassmann.enabled,
                "gate_regime": format!("{:?}", self.gate.regime),
                "gate_eff_mantissa": self.gate.last_efficiency_mantissa,
                "gate_eff_exponent": self.config.grassmann.distance_threshold_exponent,
                "extra": extra,
            }),
            ctx: CorrelationContext {
                run_id: Some(ctx.run_id.to_string()),
                venue: Some("paper".to_string()),
                strategy_id: Some(self.strategy_id()),
                ..Default::default()
            },
        }
    }

    fn create_intent(
        &self,
        parent_decision_id: Uuid,
        ctx: &StrategyContext,
        side: Side,
        tag: &str,
    ) -> OrderIntent {
        OrderIntent {
            parent_decision_id,
            symbol: ctx.symbol.to_string(),
            side,
            qty_mantissa: self.config.position_size_mantissa,
            qty_exponent: self.config.qty_exponent,
            limit_price_mantissa: None,
            price_exponent: self.config.price_exponent,
            tag: Some(tag.to_string()),
        }
    }

    fn funding_bias(&self) -> i8 {
        // +1 => long bias, -1 => short bias, 0 => no bias
        if self.current_funding_rate_mantissa > self.config.funding_threshold_mantissa {
            -1 // bias short when funding positive
        } else if self.current_funding_rate_mantissa < -self.config.funding_threshold_mantissa {
            1 // bias long when funding negative
        } else {
            0
        }
    }

    fn should_exit_long_by_funding(&self) -> bool {
        self.current_funding_rate_mantissa > -self.config.effective_exit_band()
    }

    fn should_exit_short_by_funding(&self) -> bool {
        self.current_funding_rate_mantissa < self.config.effective_exit_band()
    }

    fn gate_allows_entries(&self) -> bool {
        if !self.config.grassmann.enabled {
            return true;
        }
        match self.gate.regime {
            RegimeLabel::Trend => true,
            RegimeLabel::Chop => false,
            RegimeLabel::Unknown => !self.config.grassmann.unknown_is_chop,
        }
    }

    fn compute_mid_from_ctx(ctx: &StrategyContext) -> i64 {
        (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2
    }

    fn push_mid_sample(&mut self, ts_ms: i64, mid_mantissa: i64) {
        // enforce monotonic timestamps (segment timestamps should already be monotonic)
        if let Some(last_ts) = self.mid_ts_ms.last().copied() {
            if ts_ms < last_ts {
                // If it happens, ignore out-of-order to preserve determinism.
                return;
            }
            // If same timestamp, overwrite last sample.
            if ts_ms == last_ts {
                if let Some(last_mid) = self.mid_mantissa.last_mut() {
                    *last_mid = mid_mantissa;
                }
                return;
            }
        }
        self.mid_ts_ms.push(ts_ms);
        self.mid_mantissa.push(mid_mantissa);

        // bounded memory
        if self.mid_ts_ms.len() > 10_000 {
            // keep last 5k
            let drain = self.mid_ts_ms.len() - 5_000;
            self.mid_ts_ms.drain(0..drain);
            self.mid_mantissa.drain(0..drain);
        }
    }

    fn mid_at_or_before(&self, target_ts_ms: i64) -> Option<i64> {
        // binary search last index with ts <= target
        if self.mid_ts_ms.is_empty() {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = self.mid_ts_ms.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.mid_ts_ms[mid] <= target_ts_ms {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            None
        } else {
            Some(self.mid_mantissa[lo - 1])
        }
    }

    fn compute_return_bps(&self, now_ts_ms: i64) -> Option<i64> {
        let window_ms = (self.config.momentum_window_secs as i64) * 1000;
        let then_ts = now_ts_ms - window_ms;
        let mid_now = self.mid_at_or_before(now_ts_ms)?;
        let mid_then = self.mid_at_or_before(then_ts)?;
        if mid_then <= 0 {
            return None;
        }
        // ret_bps = 10_000 * (mid_now - mid_then) / mid_then
        let num = (mid_now as i128 - mid_then as i128) * 10_000i128;
        let den = mid_then as i128;
        Some((num / den) as i64)
    }

    fn update_grassmann_gate(&mut self, ts_ms: i64) {
        if !self.config.grassmann.enabled {
            self.gate.regime = RegimeLabel::Unknown;
            return;
        }

        let interval = self.config.grassmann.sample_interval_ms.max(1) as i64;
        let should_sample = match self.gate.last_sample_ts_ms {
            None => true,
            Some(last) => ts_ms - last >= interval,
        };
        if !should_sample {
            return;
        }

        self.gate.last_sample_ts_ms = Some(ts_ms);

        // Use a short-horizon return as lifted feature #1/#2
        // (ret_bps, abs_ret_bps). This is stable, integer-based.
        // We re-use momentum window? No: sample interval determines this.
        let prev_ts = ts_ms - interval;
        let mid_now = self.mid_at_or_before(ts_ms);
        let mid_prev = self.mid_at_or_before(prev_ts);
        let ret_bps = match (mid_now, mid_prev) {
            (Some(a), Some(b)) if b > 0 => {
                (((a as i128 - b as i128) * 10_000i128) / (b as i128)) as i64
            }
            _ => 0,
        };
        let abs_ret_bps = ret_bps.abs();

        // Ring write
        let idx = self.gate.write_idx;
        self.gate.ret_bps_ring[idx] = ret_bps;
        self.gate.abs_ret_bps_ring[idx] = abs_ret_bps;
        self.gate.write_idx = (idx + 1) % self.gate.window_len();
        self.gate.filled = self
            .gate
            .filled
            .saturating_add(1)
            .min(self.gate.window_len());

        if self.gate.filled < self.gate.window_len() {
            self.gate.regime = RegimeLabel::Unknown;
            return;
        }

        // Directional efficiency = |sum(ret)| / sum(|ret|)
        let mut sum_ret: i128 = 0;
        let mut sum_abs: i128 = 0;
        for i in 0..self.gate.window_len() {
            sum_ret += self.gate.ret_bps_ring[i] as i128;
            sum_abs += self.gate.abs_ret_bps_ring[i] as i128;
        }
        if sum_abs <= 0 {
            self.gate.regime = RegimeLabel::Unknown;
            self.gate.last_efficiency_mantissa = 0;
            return;
        }

        // efficiency in [0,1], but we store as mantissa with exponent distance_threshold_exponent
        // scale = 10^(-exp)
        let exp = self.config.grassmann.distance_threshold_exponent;
        let scale: i128 = 10i128.pow((-exp) as u32);
        let eff_mantissa = ((sum_ret.abs() * scale) / sum_abs) as i64;
        self.gate.last_efficiency_mantissa = eff_mantissa;

        // classify:
        // - Trend if efficiency >= threshold and activity >= small floor
        // Activity floor prevents "trend" during dead flat tape:
        // use average abs_ret_bps >= 1 bps over window as a default floor.
        let thr = self.config.grassmann.distance_threshold_mantissa;
        let avg_abs = (sum_abs / (self.gate.window_len() as i128)) as i64;

        if eff_mantissa >= thr && avg_abs >= 1 {
            self.gate.regime = RegimeLabel::Trend;
        } else {
            self.gate.regime = RegimeLabel::Chop;
        }
    }
}

impl Strategy for FundingAlignedMomentumStrategy {
    fn name(&self) -> &str {
        FUNDING_ALIGNED_MOMENTUM_NAME
    }

    fn version(&self) -> &str {
        FUNDING_ALIGNED_MOMENTUM_VERSION
    }

    fn config_hash(&self) -> String {
        self.config_hash.clone()
    }

    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        // 1) Funding update
        if event.kind == EventKind::Funding
            && let Some(rate_mantissa) = event
                .payload
                .get("funding_rate_mantissa")
                .and_then(|v| v.as_i64())
        {
            let rate_exponent = event
                .payload
                .get("rate_exponent")
                .and_then(|v| v.as_i64())
                .unwrap_or(-8) as i8;

            // Convert to our funding_threshold_exponent for comparison
            let exp_diff = rate_exponent as i32 - self.config.funding_threshold_exponent as i32;
            self.current_funding_rate_mantissa = if exp_diff >= 0 {
                rate_mantissa * 10i64.pow(exp_diff as u32)
            } else {
                rate_mantissa / 10i64.pow((-exp_diff) as u32)
            };
        }

        // 2) Only act on price events (need a price snapshot + mid samples)
        let is_perp_event = matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth);
        let is_spot_event = matches!(event.kind, EventKind::SpotQuote);
        if !(is_perp_event || (is_spot_event && self.config.trade_on_spot_quotes)) {
            return vec![];
        }

        // CRITICAL: Track perp prices separately for SL/TP.
        // ctx.market can contain SPOT prices when spot events arrive (9M spot vs 500K perp).
        // SL/TP must use PERP prices since we trade perps!
        if is_perp_event {
            let perp_bid = ctx.market.bid_price_mantissa();
            let perp_ask = ctx.market.ask_price_mantissa();
            if perp_bid > 0 && perp_ask > 0 {
                self.last_perp_bid = perp_bid;
                self.last_perp_ask = perp_ask;
            }
        }

        let ts_ms = ctx.ts.timestamp_millis();
        let mid = Self::compute_mid_from_ctx(ctx);
        self.push_mid_sample(ts_ms, mid);

        // 3) Update gate (uses lifted features from sampled returns)
        self.update_grassmann_gate(ts_ms);

        // 4) Compute momentum return bps (entry timing)
        let ret_bps = self.compute_return_bps(ts_ms);

        // =========================================================================
        // State machine: Only emit signals from appropriate states
        // - Flat: can emit entry signals
        // - InPosition: can emit exit signals
        // - PendingEntry/PendingExit: wait for fill, no new signals
        // =========================================================================

        // If we have pending orders, don't emit new signals
        if self.has_pending_entry() || self.has_pending_exit() {
            return vec![];
        }

        let mut outputs = vec![];

        // =====================================================================
        // When InPosition: check exit conditions
        // =====================================================================
        if self.is_in_position() {
            let entry_price = self.get_entry_fill_price().unwrap_or(mid);
            let in_min_hold = self.within_min_hold(ts_ms);

            // Use PERP-specific bid/ask for SL/TP (NOT ctx.market which can be spot!)
            let bid = self.last_perp_bid;
            let ask = self.last_perp_ask;
            let sl_tp_armed = self.sl_tp_armed(ts_ms) && bid > 0 && ask > 0;

            // PRIORITY 1: SL/TP (ignores min_hold, but respects arming delay)
            let (sl_hit, tp_hit) = if sl_tp_armed {
                let sl = self.check_stop_loss(bid, ask);
                let tp = self.check_take_profit(bid, ask);
                (sl, tp)
            } else {
                (false, false)
            };

            if self.is_confirmed_long() && sl_hit {
                let d = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_long_stop_loss",
                    serde_json::json!({"entry_price": entry_price, "exit_bid": bid, "exit_ask": ask, "sl_tp_armed": sl_tp_armed}),
                );
                let intent =
                    self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long_stop_loss");
                outputs.push(DecisionOutput::new(d, intent));
                self.transition_to_pending_exit(ts_ms);
                return outputs;
            }
            if self.is_confirmed_long() && tp_hit {
                let d = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_long_take_profit",
                    serde_json::json!({"entry_price": entry_price, "exit_bid": bid, "exit_ask": ask, "sl_tp_armed": sl_tp_armed}),
                );
                let intent =
                    self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long_take_profit");
                outputs.push(DecisionOutput::new(d, intent));
                self.transition_to_pending_exit(ts_ms);
                return outputs;
            }
            if self.is_confirmed_short() && sl_hit {
                let d = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_short_stop_loss",
                    serde_json::json!({"entry_price": entry_price, "exit_bid": bid, "exit_ask": ask, "sl_tp_armed": sl_tp_armed}),
                );
                let intent =
                    self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short_stop_loss");
                outputs.push(DecisionOutput::new(d, intent));
                self.transition_to_pending_exit(ts_ms);
                return outputs;
            }
            if self.is_confirmed_short() && tp_hit {
                let d = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_short_take_profit",
                    serde_json::json!({"entry_price": entry_price, "exit_bid": bid, "exit_ask": ask, "sl_tp_armed": sl_tp_armed}),
                );
                let intent =
                    self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short_take_profit");
                outputs.push(DecisionOutput::new(d, intent));
                self.transition_to_pending_exit(ts_ms);
                return outputs;
            }

            // PRIORITY 2: Regime-exit on chop (respects min_hold)
            if self.config.grassmann.enabled
                && self.config.grassmann.regime_exit_on_chop
                && self.gate.regime == RegimeLabel::Chop
                && !in_min_hold
            {
                if self.is_confirmed_long() {
                    let d = self.create_decision(
                        ctx,
                        0,
                        "exit",
                        "exit_long_regime_chop",
                        serde_json::json!({}),
                    );
                    let intent =
                        self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long_regime_chop");
                    outputs.push(DecisionOutput::new(d, intent));
                    self.transition_to_pending_exit(ts_ms);
                    return outputs;
                }
                if self.is_confirmed_short() {
                    let d = self.create_decision(
                        ctx,
                        0,
                        "exit",
                        "exit_short_regime_chop",
                        serde_json::json!({}),
                    );
                    let intent =
                        self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short_regime_chop");
                    outputs.push(DecisionOutput::new(d, intent));
                    self.transition_to_pending_exit(ts_ms);
                    return outputs;
                }
            }

            // PRIORITY 3: Funding-based exits (respects min_hold)
            // Note: No special flip handling - exits happen first, entry on next event
            if !in_min_hold {
                if self.is_confirmed_long() && self.should_exit_long_by_funding() {
                    let d = self.create_decision(
                        ctx,
                        0,
                        "exit",
                        "exit_long",
                        serde_json::json!({"reason":"funding_hysteresis"}),
                    );
                    let intent = self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long");
                    outputs.push(DecisionOutput::new(d, intent));
                    self.transition_to_pending_exit(ts_ms);
                    return outputs;
                }
                if self.is_confirmed_short() && self.should_exit_short_by_funding() {
                    let d = self.create_decision(
                        ctx,
                        0,
                        "exit",
                        "exit_short",
                        serde_json::json!({"reason":"funding_hysteresis"}),
                    );
                    let intent = self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short");
                    outputs.push(DecisionOutput::new(d, intent));
                    self.transition_to_pending_exit(ts_ms);
                    return outputs;
                }
            }
        }

        // =====================================================================
        // When Flat: check entry conditions
        // =====================================================================
        if self.is_flat() {
            let bias = self.funding_bias();
            let m = self.config.momentum_threshold_bps;
            let gate_ok = self.gate_allows_entries();
            let in_cooldown = self.within_cooldown(ts_ms);

            let can_enter_long =
                gate_ok && !in_cooldown && bias == 1 && ret_bps.is_some() && ret_bps.unwrap() >= m;
            let can_enter_short = gate_ok
                && !in_cooldown
                && bias == -1
                && ret_bps.is_some()
                && ret_bps.unwrap() <= -m;

            if can_enter_short {
                let d = self.create_decision(
                    ctx,
                    -1,
                    "entry",
                    "entry_short",
                    serde_json::json!({"reason":"funding_bias+momentum","ret_bps":ret_bps}),
                );
                let intent = self.create_intent(d.decision_id, ctx, Side::Sell, "entry_short");
                outputs.push(DecisionOutput::new(d, intent));
                self.transition_to_pending_entry(ts_ms, mid, -1);
            } else if can_enter_long {
                let d = self.create_decision(
                    ctx,
                    1,
                    "entry",
                    "entry_long",
                    serde_json::json!({"reason":"funding_bias+momentum","ret_bps":ret_bps}),
                );
                let intent = self.create_intent(d.decision_id, ctx, Side::Buy, "entry_long");
                outputs.push(DecisionOutput::new(d, intent));
                self.transition_to_pending_entry(ts_ms, mid, 1);
            }
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, ctx: &StrategyContext) {
        let fill_ts_ms = ctx.ts.timestamp_millis();
        let fill_price = fill.price_mantissa;

        match &self.position_state {
            PositionState::PendingEntry { direction, .. } => {
                // Entry fill confirmed: transition to InPosition
                // min_hold timer starts NOW (from fill timestamp)
                self.position_state = PositionState::InPosition {
                    fill_ts_ms,
                    fill_price,
                    direction: *direction,
                    qty_mantissa: fill.qty_mantissa,
                };
            }
            PositionState::PendingExit { .. } => {
                // Exit fill confirmed: transition to Flat
                // Cooldown timer starts NOW (from fill timestamp)
                self.last_exit_fill_ts_ms = Some(fill_ts_ms);
                self.position_state = PositionState::Flat;
            }
            PositionState::Flat => {
                // Unexpected fill while flat - could be from a prior session
                // For robustness, interpret based on fill side
                let direction = match fill.side {
                    Side::Buy => 1,
                    Side::Sell => -1,
                };
                self.position_state = PositionState::InPosition {
                    fill_ts_ms,
                    fill_price,
                    direction,
                    qty_mantissa: fill.qty_mantissa,
                };
            }
            PositionState::InPosition { direction, .. } => {
                // Fill while already in position - could be partial fill or unexpected
                // Update quantity if it's an addition to existing position
                let is_same_direction = (*direction == 1 && fill.side == Side::Buy)
                    || (*direction == -1 && fill.side == Side::Sell);
                if is_same_direction {
                    // Adding to position
                    if let PositionState::InPosition {
                        qty_mantissa: qty, ..
                    } = &mut self.position_state
                    {
                        *qty += fill.qty_mantissa;
                    }
                }
                // If opposite direction, this might be a partial exit - for now, ignore
            }
        }
    }
}

/// Factory function for registry.
pub fn funding_aligned_momentum_factory(config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
    let config = match config_path {
        Some(path) => FundingAlignedMomentumConfig::from_toml(path)?,
        None => FundingAlignedMomentumConfig::default(),
    };
    Ok(Box::new(FundingAlignedMomentumStrategy::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MarketSnapshot;
    use chrono::Utc;

    #[test]
    fn test_config_hash_deterministic() {
        let c1 = FundingAlignedMomentumConfig::default();
        let c2 = FundingAlignedMomentumConfig::default();
        assert_eq!(c1.canonical_bytes(), c2.canonical_bytes());
        assert_eq!(canonical_hash(&c1), canonical_hash(&c2));
    }

    #[test]
    fn test_effective_exit_band_default() {
        let c = FundingAlignedMomentumConfig {
            funding_threshold_mantissa: 100,
            exit_band_mantissa: None,
            ..FundingAlignedMomentumConfig::default()
        };
        assert_eq!(c.effective_exit_band(), 50);
    }

    #[test]
    fn test_gate_unknown_before_window_filled() {
        let s = FundingAlignedMomentumStrategy::new(FundingAlignedMomentumConfig {
            grassmann: GrassmannGateConfig {
                enabled: true,
                window_len: 5,
                sample_interval_ms: 1000,
                ..GrassmannGateConfig::default()
            },
            ..FundingAlignedMomentumConfig::default()
        });

        // No samples => Unknown
        assert_eq!(s.gate.regime, RegimeLabel::Unknown);
    }

    #[test]
    fn test_entry_requires_funding_bias_and_momentum() {
        let mut s = FundingAlignedMomentumStrategy::new(FundingAlignedMomentumConfig {
            grassmann: GrassmannGateConfig {
                enabled: false, // ignore gate for this unit test
                ..GrassmannGateConfig::default()
            },
            momentum_window_secs: 1,
            momentum_threshold_bps: 1,
            ..FundingAlignedMomentumConfig::default()
        });

        // Force funding bias LONG (funding negative beyond threshold)
        s.current_funding_rate_mantissa = -200;

        // Market with higher price (~+20 bps from base 10_000_000)
        // on_event uses market mid, so set bid/ask to reflect the "moved" price
        let market = MarketSnapshot::v2_all_present(
            10_020_000, // bid = $100,200 (moved up ~20 bps)
            10_020_100, // ask = $100,201
            1_000_000,
            1_000_000,
            -2,
            -8,
            10,
            1234567890000000000,
        );

        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        // Seed historical mid sample at t0 - 1000ms (base price before move)
        let t0 = ctx.ts.timestamp_millis();
        s.push_mid_sample(t0 - 1000, 10_000_000); // base price

        let event = ReplayEvent {
            ts: ctx.ts,
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({}),
        };

        // on_event will push current market mid (~10_020_050) at t0
        // compute_return_bps will see: (10_020_050 - 10_000_000) / 10_000_000 * 10000 ≈ +20 bps
        let outs = s.on_event(&event, &ctx);
        assert!(
            !outs.is_empty(),
            "Expected entry signal with funding bias + positive momentum"
        );
        assert_eq!(outs[0].decision.direction, 1); // LONG entry
    }
}
