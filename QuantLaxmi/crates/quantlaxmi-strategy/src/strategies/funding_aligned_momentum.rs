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
    /// 0 = disabled. Default: 30 bps
    #[serde(default = "default_stop_loss_bps")]
    pub stop_loss_bps: i64,

    /// Take-profit in basis points from entry price.
    /// 0 = disabled. Default: 15 bps
    #[serde(default = "default_take_profit_bps")]
    pub take_profit_bps: i64,

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
    30
}

fn default_take_profit_bps() -> i64 {
    15
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
            stop_loss_bps: 30,
            take_profit_bps: 15,

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
// Strategy implementation
// =============================================================================

pub struct FundingAlignedMomentumStrategy {
    config: FundingAlignedMomentumConfig,
    config_hash: String,

    // Funding state (mantissa in config exponent space)
    current_funding_rate_mantissa: i64,

    // Position state
    position_qty_mantissa: i64,

    // Anti-churn state tracking
    /// Timestamp (ms) when position was entered. None if flat.
    position_entry_ts_ms: Option<i64>,
    /// Entry price mantissa for SL/TP calculations. None if flat.
    position_entry_price_mantissa: Option<i64>,
    /// Timestamp (ms) of last exit. Used for cooldown enforcement.
    last_exit_ts_ms: Option<i64>,

    // Price sampling for momentum
    // Store mid prices with timestamps (ms) to compute ret over N secs.
    mid_ts_ms: Vec<i64>,
    mid_mantissa: Vec<i64>,

    // Grassmann gate state
    gate: GrassmannGateState,
}

impl FundingAlignedMomentumStrategy {
    pub fn new(config: FundingAlignedMomentumConfig) -> Self {
        let config_hash = canonical_hash(&config);
        let wl = config.grassmann.window_len.max(1) as usize;

        Self {
            config,
            config_hash,
            current_funding_rate_mantissa: 0,
            position_qty_mantissa: 0,
            position_entry_ts_ms: None,
            position_entry_price_mantissa: None,
            last_exit_ts_ms: None,
            mid_ts_ms: Vec::with_capacity(512),
            mid_mantissa: Vec::with_capacity(512),
            gate: GrassmannGateState::new(wl),
        }
    }

    /// Check if we're within the minimum hold period (cannot do funding-based exits).
    fn within_min_hold(&self, now_ms: i64) -> bool {
        if let Some(entry_ts) = self.position_entry_ts_ms {
            let hold_ms = self.config.min_hold_secs as i64 * 1000;
            now_ms - entry_ts < hold_ms
        } else {
            false
        }
    }

    /// Check if we're within cooldown period (cannot enter new positions).
    fn within_cooldown(&self, now_ms: i64) -> bool {
        if let Some(exit_ts) = self.last_exit_ts_ms {
            let cooldown_ms = self.config.cooldown_secs as i64 * 1000;
            now_ms - exit_ts < cooldown_ms
        } else {
            false
        }
    }

    /// Check if stop-loss is triggered. Returns true if SL hit.
    /// For LONG: SL hit if current price <= entry - SL_bps
    /// For SHORT: SL hit if current price >= entry + SL_bps
    fn check_stop_loss(&self, current_mid: i64) -> bool {
        if self.config.stop_loss_bps == 0 {
            return false;
        }
        let Some(entry_price) = self.position_entry_price_mantissa else {
            return false;
        };
        // SL distance in price units: entry_price * sl_bps / 10000
        let sl_distance = entry_price * self.config.stop_loss_bps / 10000;

        if self.is_long() {
            current_mid <= entry_price - sl_distance
        } else if self.is_short() {
            current_mid >= entry_price + sl_distance
        } else {
            false
        }
    }

    /// Check if take-profit is triggered. Returns true if TP hit.
    /// For LONG: TP hit if current price >= entry + TP_bps
    /// For SHORT: TP hit if current price <= entry - TP_bps
    fn check_take_profit(&self, current_mid: i64) -> bool {
        if self.config.take_profit_bps == 0 {
            return false;
        }
        let Some(entry_price) = self.position_entry_price_mantissa else {
            return false;
        };
        // TP distance in price units: entry_price * tp_bps / 10000
        let tp_distance = entry_price * self.config.take_profit_bps / 10000;

        if self.is_long() {
            current_mid >= entry_price + tp_distance
        } else if self.is_short() {
            current_mid <= entry_price - tp_distance
        } else {
            false
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

    fn is_long(&self) -> bool {
        self.position_qty_mantissa > 0
    }
    fn is_short(&self) -> bool {
        self.position_qty_mantissa < 0
    }
    #[allow(dead_code)]
    fn is_flat(&self) -> bool {
        self.position_qty_mantissa == 0
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

        let ts_ms = ctx.ts.timestamp_millis();
        let mid = Self::compute_mid_from_ctx(ctx);
        self.push_mid_sample(ts_ms, mid);

        // 3) Update gate (uses lifted features from sampled returns)
        self.update_grassmann_gate(ts_ms);

        // 4) Compute momentum return bps (entry timing)
        let ret_bps = self.compute_return_bps(ts_ms);

        let mut outputs = vec![];

        // =====================================================================
        // PRIORITY 1: Stop-Loss / Take-Profit (always checked, ignores min_hold)
        // =====================================================================
        let sl_hit = self.check_stop_loss(mid);
        let tp_hit = self.check_take_profit(mid);

        if self.is_long() && sl_hit {
            let d = self.create_decision(
                ctx,
                0,
                "exit",
                "exit_long_stop_loss",
                serde_json::json!({"entry_price": self.position_entry_price_mantissa, "exit_price": mid}),
            );
            let intent = self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long_stop_loss");
            outputs.push(DecisionOutput::new(d, intent));
            self.last_exit_ts_ms = Some(ts_ms);
            self.position_entry_ts_ms = None;
            self.position_entry_price_mantissa = None;
            return outputs;
        }
        if self.is_long() && tp_hit {
            let d = self.create_decision(
                ctx,
                0,
                "exit",
                "exit_long_take_profit",
                serde_json::json!({"entry_price": self.position_entry_price_mantissa, "exit_price": mid}),
            );
            let intent =
                self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long_take_profit");
            outputs.push(DecisionOutput::new(d, intent));
            self.last_exit_ts_ms = Some(ts_ms);
            self.position_entry_ts_ms = None;
            self.position_entry_price_mantissa = None;
            return outputs;
        }
        if self.is_short() && sl_hit {
            let d = self.create_decision(
                ctx,
                0,
                "exit",
                "exit_short_stop_loss",
                serde_json::json!({"entry_price": self.position_entry_price_mantissa, "exit_price": mid}),
            );
            let intent = self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short_stop_loss");
            outputs.push(DecisionOutput::new(d, intent));
            self.last_exit_ts_ms = Some(ts_ms);
            self.position_entry_ts_ms = None;
            self.position_entry_price_mantissa = None;
            return outputs;
        }
        if self.is_short() && tp_hit {
            let d = self.create_decision(
                ctx,
                0,
                "exit",
                "exit_short_take_profit",
                serde_json::json!({"entry_price": self.position_entry_price_mantissa, "exit_price": mid}),
            );
            let intent =
                self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short_take_profit");
            outputs.push(DecisionOutput::new(d, intent));
            self.last_exit_ts_ms = Some(ts_ms);
            self.position_entry_ts_ms = None;
            self.position_entry_price_mantissa = None;
            return outputs;
        }

        // =====================================================================
        // PRIORITY 2: Regime-exit on chop (respects min_hold unless disabled)
        // =====================================================================
        let in_min_hold = self.within_min_hold(ts_ms);

        if self.config.grassmann.enabled
            && self.config.grassmann.regime_exit_on_chop
            && self.gate.regime == RegimeLabel::Chop
            && !in_min_hold
        // Respect min_hold for regime exits
        {
            if self.is_long() {
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
                self.last_exit_ts_ms = Some(ts_ms);
                self.position_entry_ts_ms = None;
                self.position_entry_price_mantissa = None;
                return outputs;
            }
            if self.is_short() {
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
                self.last_exit_ts_ms = Some(ts_ms);
                self.position_entry_ts_ms = None;
                self.position_entry_price_mantissa = None;
                return outputs;
            }
        }

        // =====================================================================
        // PRIORITY 3: Entry/Exit/Flip rules (with min_hold + cooldown)
        // =====================================================================
        let bias = self.funding_bias();
        let m = self.config.momentum_threshold_bps;

        // Entry conditions require: gate allow + momentum agree + NOT in cooldown
        let gate_ok = self.gate_allows_entries();
        let in_cooldown = self.within_cooldown(ts_ms);

        let can_enter_long =
            gate_ok && !in_cooldown && bias == 1 && ret_bps.is_some() && ret_bps.unwrap() >= m;
        let can_enter_short =
            gate_ok && !in_cooldown && bias == -1 && ret_bps.is_some() && ret_bps.unwrap() <= -m;

        // Funding-based exits only allowed OUTSIDE min_hold period
        let exit_long = self.is_long() && !in_min_hold && self.should_exit_long_by_funding();
        let exit_short = self.is_short() && !in_min_hold && self.should_exit_short_by_funding();

        if self.is_long() {
            // Flip requires: can_enter_short AND not in min_hold (flip = exit + entry)
            if can_enter_short && !in_min_hold {
                // FLIP: exit long then enter short (stable order)
                let exit = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_long",
                    serde_json::json!({"reason":"flip_to_short"}),
                );
                let exit_intent =
                    self.create_intent(exit.decision_id, ctx, Side::Sell, "exit_long");
                outputs.push(DecisionOutput::new(exit, exit_intent));
                // Track exit for cooldown (though flip bypasses cooldown)
                self.last_exit_ts_ms = Some(ts_ms);

                let entry = self.create_decision(
                    ctx,
                    -1,
                    "entry",
                    "entry_short",
                    serde_json::json!({"reason":"funding_bias+momentum","ret_bps":ret_bps}),
                );
                let entry_intent =
                    self.create_intent(entry.decision_id, ctx, Side::Sell, "entry_short");
                outputs.push(DecisionOutput::new(entry, entry_intent));
                // Set entry timestamp NOW for the new position
                self.position_entry_ts_ms = Some(ts_ms);
                self.position_entry_price_mantissa = Some(mid);
            } else if exit_long {
                let d = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_long",
                    serde_json::json!({"reason":"funding_hysteresis"}),
                );
                let intent = self.create_intent(d.decision_id, ctx, Side::Sell, "exit_long");
                outputs.push(DecisionOutput::new(d, intent));
                // Track exit for cooldown
                self.last_exit_ts_ms = Some(ts_ms);
                self.position_entry_ts_ms = None;
                self.position_entry_price_mantissa = None;
            }
        } else if self.is_short() {
            // Flip requires: can_enter_long AND not in min_hold
            if can_enter_long && !in_min_hold {
                // FLIP: exit short then enter long (stable order)
                let exit = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_short",
                    serde_json::json!({"reason":"flip_to_long"}),
                );
                let exit_intent =
                    self.create_intent(exit.decision_id, ctx, Side::Buy, "exit_short");
                outputs.push(DecisionOutput::new(exit, exit_intent));
                // Track exit for cooldown (though flip bypasses cooldown)
                self.last_exit_ts_ms = Some(ts_ms);

                let entry = self.create_decision(
                    ctx,
                    1,
                    "entry",
                    "entry_long",
                    serde_json::json!({"reason":"funding_bias+momentum","ret_bps":ret_bps}),
                );
                let entry_intent =
                    self.create_intent(entry.decision_id, ctx, Side::Buy, "entry_long");
                outputs.push(DecisionOutput::new(entry, entry_intent));
                // Set entry timestamp NOW for the new position
                self.position_entry_ts_ms = Some(ts_ms);
                self.position_entry_price_mantissa = Some(mid);
            } else if exit_short {
                let d = self.create_decision(
                    ctx,
                    0,
                    "exit",
                    "exit_short",
                    serde_json::json!({"reason":"funding_hysteresis"}),
                );
                let intent = self.create_intent(d.decision_id, ctx, Side::Buy, "exit_short");
                outputs.push(DecisionOutput::new(d, intent));
                // Track exit for cooldown
                self.last_exit_ts_ms = Some(ts_ms);
                self.position_entry_ts_ms = None;
                self.position_entry_price_mantissa = None;
            }
        } else {
            // flat
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
                // Set entry timestamp NOW (on signal generation) so min_hold works immediately
                self.position_entry_ts_ms = Some(ts_ms);
                self.position_entry_price_mantissa = Some(mid);
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
                // Set entry timestamp NOW (on signal generation) so min_hold works immediately
                self.position_entry_ts_ms = Some(ts_ms);
                self.position_entry_price_mantissa = Some(mid);
            }
            // else: no output; diagnostics handled in tournament layer.
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, _ctx: &StrategyContext) {
        // Update position quantity only. Timestamps are tracked on signal generation.
        match fill.side {
            Side::Buy => self.position_qty_mantissa += fill.qty_mantissa,
            Side::Sell => self.position_qty_mantissa -= fill.qty_mantissa,
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
