//! CRT-Ω (Omega) Strategy v1.0
//!
//! ## Causal Reversal Transport with Evidence Gating
//!
//! A regime-conditioned microstructure mean-reversion strategy that trades only when
//! there is statistical evidence of edge. The core innovation is trading the *residual*
//! between where price *should* have gone (given order flow + liquidity state) and
//! where it actually went.
//!
//! ## Strategy Components
//!
//! 1. **Transport Map**: Predicts expected return from microstructure state
//!    - φ (phi): Signed aggressor flow from trades
//!    - ψ (psi): Depth imbalance from L2 book
//!    - κ (kappa): Stiffness/fragility = spread / weighted_depth
//!
//! 2. **Residual Signal**: ε = realized_return - predicted_return
//!    - Trade when |z(ε)| is extreme (mean reversion of overshoot)
//!
//! 3. **SPRT Evidence Gate**: Sequential probability ratio test per regime cell
//!    - Only trade when P(edge > 0 after costs) is statistically significant
//!    - Turns trading OFF when evidence decays
//!
//! 4. **Kelly Sizing**: Drawdown-constrained position sizing
//!    - Size proportional to edge / variance
//!    - Hard caps on drawdown and position
//!
//! ## Fixed-Point Policy
//! All numeric fields use mantissa + exponent. No f64 in configs.
//!
//! ## References
//! - Impact overshoot + liquidity refill dynamics (microstructure)
//! - Sequential Probability Ratio Test (Wald, 1947)
//! - Fractional Kelly criterion with drawdown constraints

use crate::canonical::{
    CONFIG_ENCODING_VERSION, CanonicalBytes, canonical_hash, encode_i8, encode_i16, encode_i64,
};
use crate::context::{FillNotification, StrategyContext};
use crate::output::{DecisionOutput, OrderIntent, Side};
use crate::{EventKind, ReplayEvent, Strategy};
use anyhow::Result;
use quantlaxmi_models::events::{CONFIDENCE_EXPONENT, CorrelationContext, DecisionEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

/// Strategy name constant.
pub const CRT_OMEGA_NAME: &str = "crt_omega";

/// Strategy version.
pub const CRT_OMEGA_VERSION: &str = "1.0.0";

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for CRT-Ω strategy.
///
/// All numeric fields use mantissa + exponent for deterministic hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrtOmegaConfig {
    // ---- Transport Map Weights ----
    /// Flow weight (a) for transport map, mantissa with exp -6
    /// r_hat = a*φ + b*ψ - c*κ*sign(r_{t-1})
    #[serde(default = "default_flow_weight")]
    pub flow_weight_mantissa: i64,

    /// Depth imbalance weight (b), mantissa with exp -4
    #[serde(default = "default_psi_weight")]
    pub psi_weight_mantissa: i64,

    /// Anti-momentum weight (c), mantissa with exp -4
    #[serde(default = "default_kappa_weight")]
    pub kappa_weight_mantissa: i64,

    // ---- Residual Z-Score ----
    /// Rolling window for robust z-score (in event ticks)
    #[serde(default = "default_z_window")]
    pub z_window: i32,

    /// Entry threshold for residual z-score, mantissa with exp -2
    /// Enter when |z| > z_in
    #[serde(default = "default_z_in")]
    pub z_in_mantissa: i64,

    /// Exit threshold for residual z-score, mantissa with exp -2
    /// Exit when |z| <= z_out
    #[serde(default = "default_z_out")]
    pub z_out_mantissa: i64,

    /// Maximum hold duration (in event ticks)
    #[serde(default = "default_max_hold")]
    pub max_hold_ticks: i32,

    // ---- SPRT Evidence Gate ----
    /// Minimum observations per regime cell before trading
    #[serde(default = "default_min_obs")]
    pub min_obs_per_cell: i32,

    /// SPRT delta (minimum edge to detect), mantissa with exp -8
    #[serde(default = "default_sprt_delta")]
    pub sprt_delta_mantissa: i64,

    /// SPRT alpha (type I error), mantissa with exp -4
    #[serde(default = "default_sprt_alpha")]
    pub sprt_alpha_mantissa: i64,

    /// SPRT beta (type II error), mantissa with exp -4
    #[serde(default = "default_sprt_beta")]
    pub sprt_beta_mantissa: i64,

    /// Bypass SPRT evidence gate (for research only - NOT for production)
    /// When true, trades on z-score signals without requiring statistical evidence
    #[serde(default)]
    pub bypass_sprt: bool,

    // ---- Position Sizing ----
    /// Base position size, mantissa with qty_exponent
    pub position_size_mantissa: i64,

    /// Kelly fraction, mantissa with exp -4
    #[serde(default = "default_kelly_frac")]
    pub kelly_frac_mantissa: i64,

    /// Maximum position multiplier, mantissa with exp -4
    #[serde(default = "default_max_pos")]
    pub max_pos_multiplier_mantissa: i64,

    /// Drawdown stop threshold, mantissa with exp -4 (e.g., -600 = -6%)
    #[serde(default = "default_dd_stop")]
    pub dd_stop_mantissa: i64,

    // ---- Costs ----
    /// Maker fee in basis points, mantissa with exp -2
    #[serde(default = "default_maker_fee")]
    pub maker_fee_bps_mantissa: i64,

    /// Taker fee in basis points, mantissa with exp -2
    #[serde(default = "default_taker_fee")]
    pub taker_fee_bps_mantissa: i64,

    // ---- Exponents ----
    pub qty_exponent: i8,
    pub price_exponent: i8,

    // ---- Regime Bucketing ----
    /// Number of kappa buckets
    #[serde(default = "default_num_buckets")]
    pub num_kappa_buckets: i16,

    /// Number of spread buckets
    #[serde(default = "default_num_buckets")]
    pub num_spread_buckets: i16,

    /// Number of premium buckets (mark - mid)
    #[serde(default = "default_num_buckets")]
    pub num_prem_buckets: i16,
}

fn default_flow_weight() -> i64 { 1 }         // 1e-6
fn default_psi_weight() -> i64 { 200 }        // 0.02
fn default_kappa_weight() -> i64 { 1000 }     // 0.10
fn default_z_window() -> i32 { 900 }
fn default_z_in() -> i64 { 300 }              // 3.0
fn default_z_out() -> i64 { 50 }              // 0.5
fn default_max_hold() -> i32 { 180 }
fn default_min_obs() -> i32 { 60 }
fn default_sprt_delta() -> i64 { 1 }          // 1e-8
fn default_sprt_alpha() -> i64 { 500 }        // 0.05
fn default_sprt_beta() -> i64 { 1000 }        // 0.10
fn default_kelly_frac() -> i64 { 500 }        // 0.05 (conservative)
fn default_max_pos() -> i64 { 1000 }          // 0.10 (10% max weight)
fn default_dd_stop() -> i64 { -600 }          // -0.06 = -6%
fn default_maker_fee() -> i64 { 2 }           // 0.02 bps
fn default_taker_fee() -> i64 { 10 }          // 0.10 bps
fn default_num_buckets() -> i16 { 3 }

impl Default for CrtOmegaConfig {
    fn default() -> Self {
        Self {
            flow_weight_mantissa: default_flow_weight(),
            psi_weight_mantissa: default_psi_weight(),
            kappa_weight_mantissa: default_kappa_weight(),
            z_window: default_z_window(),
            z_in_mantissa: default_z_in(),
            z_out_mantissa: default_z_out(),
            max_hold_ticks: default_max_hold(),
            min_obs_per_cell: default_min_obs(),
            sprt_delta_mantissa: default_sprt_delta(),
            sprt_alpha_mantissa: default_sprt_alpha(),
            sprt_beta_mantissa: default_sprt_beta(),
            bypass_sprt: false,
            position_size_mantissa: 1_000_000, // 0.01 BTC with exp -8
            kelly_frac_mantissa: default_kelly_frac(),
            max_pos_multiplier_mantissa: default_max_pos(),
            dd_stop_mantissa: default_dd_stop(),
            maker_fee_bps_mantissa: default_maker_fee(),
            taker_fee_bps_mantissa: default_taker_fee(),
            qty_exponent: -8,
            price_exponent: -2,
            num_kappa_buckets: default_num_buckets(),
            num_spread_buckets: default_num_buckets(),
            num_prem_buckets: default_num_buckets(),
        }
    }
}

impl CanonicalBytes for CrtOmegaConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(CONFIG_ENCODING_VERSION);
        // Transport map
        encode_i64(&mut buf, self.flow_weight_mantissa);
        encode_i64(&mut buf, self.psi_weight_mantissa);
        encode_i64(&mut buf, self.kappa_weight_mantissa);
        // Z-score
        encode_i64(&mut buf, self.z_window as i64);
        encode_i64(&mut buf, self.z_in_mantissa);
        encode_i64(&mut buf, self.z_out_mantissa);
        encode_i64(&mut buf, self.max_hold_ticks as i64);
        // SPRT
        encode_i64(&mut buf, self.min_obs_per_cell as i64);
        encode_i64(&mut buf, self.sprt_delta_mantissa);
        encode_i64(&mut buf, self.sprt_alpha_mantissa);
        encode_i64(&mut buf, self.sprt_beta_mantissa);
        buf.push(if self.bypass_sprt { 1 } else { 0 });
        // Sizing
        encode_i64(&mut buf, self.position_size_mantissa);
        encode_i64(&mut buf, self.kelly_frac_mantissa);
        encode_i64(&mut buf, self.max_pos_multiplier_mantissa);
        encode_i64(&mut buf, self.dd_stop_mantissa);
        // Costs
        encode_i64(&mut buf, self.maker_fee_bps_mantissa);
        encode_i64(&mut buf, self.taker_fee_bps_mantissa);
        // Exponents
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);
        // Buckets
        encode_i16(&mut buf, self.num_kappa_buckets);
        encode_i16(&mut buf, self.num_spread_buckets);
        encode_i16(&mut buf, self.num_prem_buckets);
        buf
    }
}

impl CrtOmegaConfig {
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}

// ============================================================================
// SPRT (Sequential Probability Ratio Test) for Edge Detection
// ============================================================================

/// SPRT state for one regime cell.
///
/// Sequential t-gate: enables trading when t-stat exceeds threshold
/// This is a simpler, more robust alternative to SPRT that won't get stuck.
/// Uses Welford's algorithm for numerically stable mean/variance.
#[derive(Debug, Clone)]
struct TGate {
    n: u64,
    mean: f64,
    m2: f64,      // Running sum of squared deviations (Welford)
    on: bool,
    // Counters for debugging
    updates: u64,
}

const T_ON: f64 = 3.0;   // Enable when t-stat > 3.0 (more trades, lower variance)
const T_OFF: f64 = 1.0;  // Disable when t-stat < 1.0 (hysteresis)
const MIN_EDGE: f64 = 0.0; // No minimum edge - let T-gate filter (was 3 bps)
const STOP_LOSS: f64 = 0.0002; // 2 bps stop-loss (very aggressive)

// State machine timing constants (milliseconds) - ALL TIME-BASED, NEVER TICK-BASED
const MIN_HOLD_MS: i64 = 3_000;    // Minimum 3s hold before exit
const MAX_HOLD_MS: i64 = 10_000;   // Maximum 10s hold (align with H10 horizon)
const COOLDOWN_MS: i64 = 2_000;    // 2s cooldown after exit before new entry
const TARGET_HORIZON_MS: i64 = 10_000; // Target exit at 10s (H10)

/// Position state for precise entry/exit tracking and trade logging
#[derive(Debug, Clone, Default)]
struct PositionState {
    side: i8,            // +1 long, -1 short, 0 flat
    entry_ts_ms: i64,    // Entry timestamp (ms)
    entry_px: f64,       // Entry fill price
    entry_mid: f64,      // Mid at entry (for mid-based return)
    entry_bid: f64,      // Bid at entry
    entry_ask: f64,      // Ask at entry
    entry_cell: u32,     // Cell ID at entry
    entry_z: f64,        // Z-score at entry
    entry_w: f64,        // Position weight at entry
    last_exit_ts_ms: i64, // Last exit timestamp for cooldown
}

impl PositionState {
    fn is_flat(&self) -> bool { self.side == 0 }
    fn is_long(&self) -> bool { self.side > 0 }

    fn can_enter(&self, now_ts_ms: i64) -> bool {
        self.is_flat() && (self.last_exit_ts_ms == 0 || now_ts_ms - self.last_exit_ts_ms >= COOLDOWN_MS)
    }

    fn can_exit(&self, now_ts_ms: i64) -> bool {
        !self.is_flat() && (now_ts_ms - self.entry_ts_ms >= MIN_HOLD_MS)
    }

    fn max_hold_hit(&self, now_ts_ms: i64) -> bool {
        !self.is_flat() && (now_ts_ms - self.entry_ts_ms >= MAX_HOLD_MS)
    }

    fn target_horizon_hit(&self, now_ts_ms: i64) -> bool {
        !self.is_flat() && (now_ts_ms - self.entry_ts_ms >= TARGET_HORIZON_MS)
    }
}

impl TGate {
    fn new() -> Self {
        Self { n: 0, mean: 0.0, m2: 0.0, on: false, updates: 0 }
    }

    /// Update with new observation (in return space, e.g., 0.00004 for 4 bps)
    fn update(&mut self, y: f64) {
        self.updates += 1;
        self.n += 1;
        let d = y - self.mean;
        self.mean += d / self.n as f64;
        let d2 = y - self.mean;
        self.m2 += d * d2;

        // Decide after enough samples
        if self.n >= 30 {
            self.decide();
        }
    }

    fn var(&self) -> f64 {
        if self.n >= 2 { self.m2 / (self.n as f64 - 1.0) } else { 0.0 }
    }

    fn std(&self) -> f64 {
        self.var().max(0.0).sqrt()
    }

    fn t_stat(&self) -> f64 {
        if self.n < 30 { return 0.0; }
        let v = self.var();
        if v <= 0.0 { return 0.0; }
        self.mean / (v / self.n as f64).sqrt()
    }

    fn decide(&mut self) {
        let t = self.t_stat();
        if !self.on && t >= T_ON {
            self.on = true;
        }
        if self.on && t <= T_OFF {
            self.on = false;
        }
    }

    fn is_on(&self) -> bool {
        self.on
    }
}

// ============================================================================
// Rolling Statistics (Robust Z-Score)
// ============================================================================

/// Ring buffer for robust rolling statistics
#[derive(Debug, Clone)]
struct RollingStats {
    values: Vec<i64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl RollingStats {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![0; capacity],
            capacity,
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, value: i64) {
        self.values[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Compute robust z-score using median and MAD
    fn robust_z(&self, value: i64) -> i64 {
        if self.count < 50 {
            return 0;
        }

        // Get sorted values for median
        let mut sorted: Vec<i64> = self.values[..self.count].to_vec();
        sorted.sort_unstable();

        let median = sorted[self.count / 2];

        // Compute MAD (median absolute deviation)
        let mut deviations: Vec<i64> = sorted.iter().map(|&v| (v - median).abs()).collect();
        deviations.sort_unstable();
        let mad = deviations[self.count / 2];

        // Z = (value - median) / (1.4826 * MAD)
        // Scale: multiply by 100 for 2 decimal places
        if mad == 0 {
            return 0;
        }

        // 1.4826 ≈ 14826/10000
        let scaled_mad = (mad * 14826) / 10000;
        if scaled_mad == 0 {
            return 0;
        }

        ((value - median) * 100) / scaled_mad
    }
}

// ============================================================================
// Microstructure State
// ============================================================================

#[derive(Debug, Clone, Default)]
struct MicroState {
    // Order flow (signed aggressor volume, mantissa exp -8)
    phi_mantissa: i64,
    last_flow_ts_ms: i64,

    // Book state
    bid_price_mantissa: i64,
    ask_price_mantissa: i64,
    bid_qty_mantissa: i64,
    ask_qty_mantissa: i64,

    // Depth imbalance (-1 to +1, mantissa exp -4)
    psi_mantissa: i64,

    // Stiffness/fragility (spread / depth, mantissa exp -8)
    kappa_mantissa: i64,

    // Mark price (for perps)
    mark_price_mantissa: i64,

    // Premium (mark - mid, mantissa exp -8)
    prem_mantissa: i64,

    // Previous return (mantissa exp -8)
    prev_ret_mantissa: i64,
    prev_mid_mantissa: i64,
}

impl MicroState {
    fn mid_mantissa(&self) -> i64 {
        (self.bid_price_mantissa + self.ask_price_mantissa) / 2
    }

    fn spread_mantissa(&self) -> i64 {
        (self.ask_price_mantissa - self.bid_price_mantissa).max(0)
    }

    fn update_from_quote(&mut self, payload: &serde_json::Value) {
        if let Some(bid) = payload.get("bid_price_mantissa").and_then(|v| v.as_i64()) {
            self.bid_price_mantissa = bid;
        }
        if let Some(ask) = payload.get("ask_price_mantissa").and_then(|v| v.as_i64()) {
            self.ask_price_mantissa = ask;
        }
        if let Some(bq) = payload.get("bid_qty_mantissa").and_then(|v| v.as_i64()) {
            self.bid_qty_mantissa = bq;
        }
        if let Some(aq) = payload.get("ask_qty_mantissa").and_then(|v| v.as_i64()) {
            self.ask_qty_mantissa = aq;
        }

        // Compute depth imbalance: psi = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        let total = self.bid_qty_mantissa + self.ask_qty_mantissa;
        if total > 0 {
            self.psi_mantissa = ((self.bid_qty_mantissa - self.ask_qty_mantissa) * 10000) / total;
        }

        // Compute kappa (stiffness): spread / depth
        let spread = self.spread_mantissa();
        if total > 0 {
            self.kappa_mantissa = (spread * 100_000_000) / total;
        }

        // Update return
        let new_mid = self.mid_mantissa();
        if self.prev_mid_mantissa > 0 && new_mid > 0 {
            // Log return approximation: (new - old) / old * 10^8
            self.prev_ret_mantissa = ((new_mid - self.prev_mid_mantissa) * 100_000_000)
                / self.prev_mid_mantissa;
        }
        self.prev_mid_mantissa = new_mid;

        // Update premium
        if self.mark_price_mantissa > 0 && new_mid > 0 {
            self.prem_mantissa = ((self.mark_price_mantissa - new_mid) * 100_000_000) / new_mid;
        }
    }

    fn update_from_trade(&mut self, payload: &serde_json::Value, ts_ms: i64, decay_tau_ms: i64) {
        let qty = payload.get("qty_mantissa").and_then(|v| v.as_i64()).unwrap_or(0);
        let is_buyer_maker = payload.get("is_buyer_maker").and_then(|v| v.as_bool()).unwrap_or(false);

        // Signed flow: buyer is maker → aggressive sell → negative
        let signed_qty = if is_buyer_maker { -qty } else { qty };

        // Exponential decay of previous flow
        if self.last_flow_ts_ms > 0 && ts_ms > self.last_flow_ts_ms {
            let dt = ts_ms - self.last_flow_ts_ms;
            // decay = exp(-dt/tau) ≈ (tau - dt) / tau for small dt
            let decay = ((decay_tau_ms - dt.min(decay_tau_ms)) * 10000) / decay_tau_ms;
            self.phi_mantissa = (self.phi_mantissa * decay) / 10000 + signed_qty;
        } else {
            self.phi_mantissa = signed_qty;
        }
        self.last_flow_ts_ms = ts_ms;
    }

    fn update_from_mark(&mut self, payload: &serde_json::Value) {
        if let Some(mark) = payload.get("mark_price_mantissa").and_then(|v| v.as_i64()) {
            self.mark_price_mantissa = mark;
        }
    }
}

// ============================================================================
// Per-Cell Payoff Statistics (Welford online algorithm)
// ============================================================================

/// Running statistics for payoff diagnostics (uses Welford's algorithm)
#[derive(Debug, Clone, Default)]
struct PayoffStats {
    n: u64,
    mean: f64,
    m2: f64,     // Running sum of squared deviations
    pos: u64,    // Count of positive payoffs (hit rate)
}

impl PayoffStats {
    fn update(&mut self, y: f64) {
        self.n += 1;
        let delta = y - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = y - self.mean;
        self.m2 += delta * delta2;
        if y > 0.0 {
            self.pos += 1;
        }
    }

    fn var(&self) -> f64 {
        if self.n >= 2 {
            self.m2 / (self.n as f64 - 1.0)
        } else {
            0.0
        }
    }

    fn std(&self) -> f64 {
        self.var().max(0.0).sqrt()
    }

    fn hit_rate(&self) -> f64 {
        if self.n > 0 {
            self.pos as f64 / self.n as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// CRT-Ω Strategy
// ============================================================================

pub struct CrtOmegaStrategy {
    config: CrtOmegaConfig,
    config_hash: String,

    // Microstructure state
    micro: MicroState,

    // Rolling stats for residual z-score
    residual_stats: RollingStats,

    // SPRT per regime cell
    sprt_cells: HashMap<u32, TGate>,

    // Position state (unified)
    position_qty_mantissa: i64,
    pos: PositionState,         // Full position state for time-based logic
    current_tick: i64,

    // Trade logging (first 20 trades for debugging)
    trade_log_count: usize,

    // Hold duration statistics
    hold_durations_ms: Vec<i64>,

    // Equity tracking for drawdown
    peak_equity_mantissa: i64,
    current_equity_mantissa: i64,

    // Regime bucket edges (computed from first N observations)
    // NOTE: Premium dimension disabled - using only kappa × spread = 9 cells
    kappa_edges: Vec<i64>,
    spread_edges: Vec<i64>,
    kappa_median: i64,  // For 1-tick spread veto
    warmup_kappa: Vec<i64>,
    warmup_spread: Vec<i64>,
    warmup_done: bool,

    // Multi-horizon payoff tracking
    // Ring buffer of mid prices for computing delayed payoffs
    price_history: Vec<i64>,      // Mid prices (mantissa)
    price_history_idx: usize,     // Current write position
    price_history_ts: Vec<i64>,   // Timestamps (ms) for each price

    // Pending signal evaluations (for multi-horizon SPRT updates)
    // Each entry: (signal_tick, signal_dir, cell_id, entry_price, entry_ts_ms)
    pending_signals: Vec<(i64, i8, u32, i64, i64)>,

    // Per-cell payoff statistics for diagnostics
    cell_stats_pre: [PayoffStats; 9],   // Pre-cost payoffs
    cell_stats_post: [PayoffStats; 9],  // Post-cost payoffs

    // Kappa validation counters
    kappa_zero_count: u64,
    kappa_valid_count: u64,

    // Diagnostics
    total_signals: u64,
    signals_gated: u64,
    trades_taken: u64,
}

impl CrtOmegaStrategy {
    pub fn new(config: CrtOmegaConfig) -> Self {
        let config_hash = canonical_hash(&config);
        let z_window = config.z_window.max(100) as usize;

        // Price history buffer size: 60 seconds at ~10 updates/sec = 600
        // We need at least 30s of history for multi-horizon payoffs
        let price_history_size = 1000;

        Self {
            config,
            config_hash,
            micro: MicroState::default(),
            residual_stats: RollingStats::new(z_window),
            sprt_cells: HashMap::new(),
            position_qty_mantissa: 0,
            pos: PositionState::default(),
            current_tick: 0,
            trade_log_count: 0,
            hold_durations_ms: Vec::with_capacity(10000),
            peak_equity_mantissa: 100_000_000, // Start at 1.0
            current_equity_mantissa: 100_000_000,
            kappa_edges: Vec::new(),
            spread_edges: Vec::new(),
            kappa_median: 0,
            warmup_kappa: Vec::new(),
            warmup_spread: Vec::new(),
            warmup_done: false,
            price_history: vec![0; price_history_size],
            price_history_idx: 0,
            price_history_ts: vec![0; price_history_size],
            pending_signals: Vec::with_capacity(1000),
            cell_stats_pre: Default::default(),
            cell_stats_post: Default::default(),
            kappa_zero_count: 0,
            kappa_valid_count: 0,
            total_signals: 0,
            signals_gated: 0,
            trades_taken: 0,
        }
    }

    /// Compute transport map prediction: r_hat = a*φ + b*ψ - c*κ*sign(r_{t-1})
    fn compute_r_hat(&self) -> i64 {
        let a = self.config.flow_weight_mantissa;   // exp -6
        let b = self.config.psi_weight_mantissa;    // exp -4
        let c = self.config.kappa_weight_mantissa;  // exp -4

        // a * phi (phi is exp -8, a is exp -6, result is exp -14 → scale to exp -8)
        let flow_term = (a * self.micro.phi_mantissa) / 1_000_000;

        // b * psi (psi is exp -4, b is exp -4, result is exp -8)
        let psi_term = (b * self.micro.psi_mantissa) / 10000;

        // c * kappa * sign(prev_ret)
        let mom_sign = if self.micro.prev_ret_mantissa > 0 { 1 }
                       else if self.micro.prev_ret_mantissa < 0 { -1 }
                       else { 0 };
        // kappa is exp -8, c is exp -4, result is exp -12 → scale to exp -8
        let anti_mom_term = (c * self.micro.kappa_mantissa * mom_sign) / 10000;

        flow_term + psi_term - anti_mom_term
    }

    /// Get regime cell ID from current microstructure state
    /// Uses only kappa × spread = 9 cells (premium dimension disabled)
    fn get_cell_id(&self) -> u32 {
        if !self.warmup_done || self.kappa_edges.is_empty() {
            return 0;
        }

        let ki = bucket_index(self.micro.kappa_mantissa, &self.kappa_edges);
        let si = bucket_index(self.micro.spread_mantissa(), &self.spread_edges);

        let nk = self.config.num_kappa_buckets as u32;
        let ns = self.config.num_spread_buckets as u32;

        // 9 cells: ki * 3 + si (0..8)
        ki.min(nk - 1) * ns + si.min(ns - 1)
    }

    /// Check if 1-tick spread + low-kappa veto applies
    /// Refuse to trade when spread <= 1 tick AND kappa < median
    fn is_vetoed_regime(&self) -> bool {
        let spread = self.micro.spread_mantissa();
        let kappa = self.micro.kappa_mantissa;

        // 1 tick = 1 in price mantissa units (with exp -2, this is $0.01)
        // For BTC at ~$80k, tick size is typically 0.1 = 10 in mantissa
        let tick_size = 10i64; // $0.10 tick

        spread <= tick_size && kappa < self.kappa_median
    }

    /// Record price for multi-horizon payoff calculation
    fn record_price(&mut self, mid: i64, ts_ms: i64) {
        let idx = self.price_history_idx % self.price_history.len();
        self.price_history[idx] = mid;
        self.price_history_ts[idx] = ts_ms;
        self.price_history_idx += 1;
    }

    /// Find price at a given timestamp (searching backwards)
    fn find_price_at_ts(&self, target_ts_ms: i64) -> Option<i64> {
        let len = self.price_history.len();
        let current = self.price_history_idx;

        // Search backwards through history
        for offset in 0..len.min(current) {
            let idx = (current - 1 - offset) % len;
            let ts = self.price_history_ts[idx];
            if ts > 0 && ts <= target_ts_ms {
                return Some(self.price_history[idx]);
            }
        }
        None
    }

    /// Compute multi-horizon payoff for a signal
    /// Returns (pre_cost, post_cost) payoffs across 3s, 10s, 30s horizons
    fn compute_multi_horizon_payoff(
        &self,
        signal_dir: i8,
        entry_price: i64,
        entry_ts_ms: i64,
        current_ts_ms: i64,
    ) -> Option<(i64, i64)> {
        // Horizons in milliseconds
        const H3S: i64 = 3_000;
        const H10S: i64 = 10_000;
        const H30S: i64 = 30_000;

        let elapsed = current_ts_ms - entry_ts_ms;
        if elapsed < H30S {
            return None; // Not enough time passed
        }

        // Cost in exp -8 units (round-trip: entry + exit as taker)
        // taker_fee_bps_mantissa has exponent -2, so actual bps = mantissa * 10^-2
        // 1 bps = 0.0001 in return space = 10,000 in exp -8
        // 0.10 bps (mantissa=10, exp=-2) = 10 * 10^-2 / 10000 * 1e8 = 10 * 100 = 1,000 in exp -8
        // Round-trip (2×) = 2,000 in exp -8
        let cost_one_side = self.config.taker_fee_bps_mantissa * 100; // mantissa * 10^(8-4-2) = mantissa * 100

        // Entry uses bid/ask (realistic). Exit uses mid which is slightly optimistic.
        // Net effect: ~half-spread advantage at exit partially offsets entry cost.
        // Keep cost model simple: just round-trip fees
        let cost = 2 * cost_one_side;

        // Helper to compute log return (approximated as (p1 - p0) / p0 * 1e8)
        let logret = |p0: i64, p1: i64| -> i64 {
            if p0 == 0 { return 0; }
            ((p1 - p0) * 100_000_000) / p0
        };

        // Find prices at each horizon
        let p3 = self.find_price_at_ts(entry_ts_ms + H3S);
        let p10 = self.find_price_at_ts(entry_ts_ms + H10S);
        let p30 = self.find_price_at_ts(entry_ts_ms + H30S);

        // Compute PRE-COST payoffs (direction * return)
        let dir = signal_dir as i64;
        let y3_pre = p3.map(|p| dir * logret(entry_price, p)).unwrap_or(0);
        let y10_pre = p10.map(|p| dir * logret(entry_price, p)).unwrap_or(0);
        let y30_pre = p30.map(|p| dir * logret(entry_price, p)).unwrap_or(0);

        // Weighted average: 0.2*y3 + 0.5*y10 + 0.3*y30
        let weighted_pre = (2 * y3_pre + 5 * y10_pre + 3 * y30_pre) / 10;

        // Compute POST-COST payoffs (direction * return - cost)
        let y3_post = y3_pre - cost;
        let y10_post = y10_pre - cost;
        let y30_post = y30_pre - cost;
        let weighted_post = (2 * y3_post + 5 * y10_post + 3 * y30_post) / 10;

        Some((weighted_pre, weighted_post))
    }

    /// Compute position size using Kelly fraction (f64 version for TGate)
    ///
    /// CRITICAL FIXES:
    /// 1. Variance floor at (5 bps)^2 = 0.0005^2 = 2.5e-7 to prevent Kelly explosion
    /// 2. w_max = 0.10 (10% of account) hard cap
    /// 3. Conservative kelly_frac = 0.05 (1/20th Kelly)
    fn compute_size_from_stats_f64(&self, mean: f64, var: f64) -> i64 {
        // Variance floor: (5 bps)^2 = (0.0005)^2 = 2.5e-7
        // This prevents Kelly from exploding when variance is tiny
        const VAR_FLOOR: f64 = 0.0005 * 0.0005; // = 2.5e-7
        let var = var.max(VAR_FLOOR);

        // Kelly: f = mean / var
        let kelly_frac = self.config.kelly_frac_mantissa as f64 / 10000.0; // Convert to fraction
        let raw_kelly = (mean / var) * kelly_frac;

        // Hard cap at w_max (10% of account weight)
        // max_pos_multiplier_mantissa = 1000 → 0.10
        let w_max = self.config.max_pos_multiplier_mantissa as f64 / 10000.0;
        let w = raw_kelly.abs().min(w_max).max(0.001); // At least 0.1% if trading

        // Final size = base_size * w
        // base_size is in qty mantissa (e.g., 1_000_000 = 0.01 BTC with exp -8)
        (self.config.position_size_mantissa as f64 * w) as i64
    }

    /// Check drawdown stop
    fn is_dd_stopped(&self) -> bool {
        if self.peak_equity_mantissa == 0 {
            return false;
        }
        let dd = ((self.current_equity_mantissa - self.peak_equity_mantissa) * 10000)
            / self.peak_equity_mantissa;
        dd < self.config.dd_stop_mantissa
    }

    fn create_decision(
        &self,
        ctx: &StrategyContext,
        direction: i8,
        decision_type: &str,
        tag: &str,
        qty_mantissa: i64,
    ) -> DecisionEvent {
        let decision_id = Uuid::new_v4();
        let mid_mantissa = self.micro.mid_mantissa();
        let confidence_mantissa = 10i64.pow((-CONFIDENCE_EXPONENT) as u32);

        DecisionEvent {
            ts: ctx.ts,
            decision_id,
            strategy_id: self.strategy_id(),
            symbol: ctx.symbol.to_string(),
            decision_type: decision_type.to_string(),
            direction,
            target_qty_mantissa: qty_mantissa,
            qty_exponent: self.config.qty_exponent,
            reference_price_mantissa: mid_mantissa,
            price_exponent: self.config.price_exponent,
            market_snapshot: ctx.market.clone(),
            confidence_mantissa,
            metadata: serde_json::json!({
                "tag": tag,
                "phi_mantissa": self.micro.phi_mantissa,
                "psi_mantissa": self.micro.psi_mantissa,
                "kappa_mantissa": self.micro.kappa_mantissa,
                "prem_mantissa": self.micro.prem_mantissa,
                "cell_id": self.get_cell_id(),
                "tick": self.current_tick,
                "policy": "crt_omega_v1.0",
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
        qty_mantissa: i64,
        tag: &str,
    ) -> OrderIntent {
        OrderIntent {
            parent_decision_id,
            symbol: ctx.symbol.to_string(),
            side,
            qty_mantissa,
            qty_exponent: self.config.qty_exponent,
            limit_price_mantissa: None, // Market order
            price_exponent: self.config.price_exponent,
            tag: Some(tag.to_string()),
        }
    }

    fn is_long(&self) -> bool { self.pos.is_long() }
    fn is_flat(&self) -> bool { self.pos.is_flat() }

    /// Get current bid/ask as f64
    fn bid_f64(&self) -> f64 {
        self.micro.bid_price_mantissa as f64 * 0.01 // exp -2
    }
    fn ask_f64(&self) -> f64 {
        self.micro.ask_price_mantissa as f64 * 0.01 // exp -2
    }
    fn mid_f64(&self) -> f64 {
        (self.bid_f64() + self.ask_f64()) / 2.0
    }

    /// Fill price model: conservative (long pays ask, short sells bid)
    fn entry_fill_px(&self, side: i8) -> f64 {
        if side > 0 { self.ask_f64() } else { self.bid_f64() }
    }
    fn exit_fill_px(&self, side: i8) -> f64 {
        if side > 0 { self.bid_f64() } else { self.ask_f64() }
    }
}

impl Strategy for CrtOmegaStrategy {
    fn name(&self) -> &str { CRT_OMEGA_NAME }
    fn version(&self) -> &str { CRT_OMEGA_VERSION }
    fn config_hash(&self) -> String { self.config_hash.clone() }

    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        let ts_ms = event.ts.timestamp_millis();
        self.current_tick += 1;

        // Update microstructure state based on event type
        match event.kind {
            EventKind::PerpQuote | EventKind::SpotQuote => {
                self.micro.update_from_quote(&event.payload);
            }
            EventKind::PerpDepth => {
                self.micro.update_from_quote(&event.payload);
            }
            EventKind::Trade => {
                self.micro.update_from_trade(&event.payload, ts_ms, 5000);
            }
            EventKind::Funding => {
                self.micro.update_from_mark(&event.payload);
            }
            _ => return vec![],
        }

        // Warmup: collect regime bucket data (only if book is valid)
        // NOTE: Premium dimension disabled - using only kappa × spread = 9 cells
        if !self.warmup_done {
            // Book sanity check - STRICT validation:
            // 1. bid and ask must exist and be positive
            // 2. bid < ask (crossed book = invalid)
            // 3. spread >= 1 tick (not 0)
            // 4. depth must be positive
            // 5. kappa must be finite and > 0
            let bid = self.micro.bid_price_mantissa;
            let ask = self.micro.ask_price_mantissa;
            let spread = self.micro.spread_mantissa();
            let kappa = self.micro.kappa_mantissa;
            let depth = self.micro.bid_qty_mantissa + self.micro.ask_qty_mantissa;

            let book_valid = bid > 0 && ask > 0 && bid < ask && spread > 0 && depth > 0;

            // CRITICAL: Only accept kappa > 0 (not just valid book)
            // kappa = 0 means spread/depth is degenerate
            if book_valid && kappa > 0 {
                self.warmup_kappa.push(kappa);
                self.warmup_spread.push(spread);
                self.kappa_valid_count += 1;
            } else if book_valid && kappa == 0 {
                self.kappa_zero_count += 1;
            }

            if self.warmup_kappa.len() >= 500 {
                // Print feature summaries for debugging
                println!("\n=== CRT-Ω WARMUP DIAGNOSTICS ===");
                println!("kappa: valid={} zero={} ({}% valid)",
                    self.kappa_valid_count,
                    self.kappa_zero_count,
                    if self.kappa_valid_count + self.kappa_zero_count > 0 {
                        100.0 * self.kappa_valid_count as f64 / (self.kappa_valid_count + self.kappa_zero_count) as f64
                    } else { 0.0 }
                );
                summarize_feature("kappa", &self.warmup_kappa);
                summarize_feature("spread", &self.warmup_spread);

                // Compute quantile edges with robust algorithm
                self.kappa_edges = compute_quantile_edges(&self.warmup_kappa, self.config.num_kappa_buckets as usize);
                self.spread_edges = compute_quantile_edges(&self.warmup_spread, self.config.num_spread_buckets as usize);

                // Compute kappa_median for 1-tick spread veto
                let mut sorted_kappa = self.warmup_kappa.clone();
                sorted_kappa.sort_unstable();
                self.kappa_median = sorted_kappa[sorted_kappa.len() / 2];

                // Log computed edges
                println!("kappa_edges={:?}", self.kappa_edges);
                println!("spread_edges={:?}", self.spread_edges);
                println!("kappa_median={}", self.kappa_median);

                // Count unique cells to verify bucketing works (kappa × spread = 9 cells max)
                let mut unique_cells = std::collections::HashSet::new();
                for i in 0..self.warmup_kappa.len() {
                    let ki = bucket_index(self.warmup_kappa[i], &self.kappa_edges);
                    let si = bucket_index(self.warmup_spread[i], &self.spread_edges);
                    let nk = self.config.num_kappa_buckets as u32;
                    let ns = self.config.num_spread_buckets as u32;
                    let cell_id = ki.min(nk - 1) * ns + si.min(ns - 1);
                    unique_cells.insert(cell_id);
                }
                println!("unique_cells={} (9 max, kappa×spread)", unique_cells.len());

                self.warmup_done = true;
            }
            return vec![];
        }

        // Only trade on quote events (need prices)
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth | EventKind::SpotQuote) {
            return vec![];
        }

        // Record price for multi-horizon payoff calculation
        let mid = self.micro.mid_mantissa();
        self.record_price(mid, ts_ms);

        // Process pending signals that are old enough (>30s) for multi-horizon SPRT updates

        // Collect indices to remove and payoffs to record
        let mut to_remove = Vec::new();
        let mut payoff_updates: Vec<(u32, i64, i64)> = Vec::new(); // (cell_id, pre, post)

        for (idx, &(_signal_tick, signal_dir, cell_id, entry_price, entry_ts)) in self.pending_signals.iter().enumerate() {
            if let Some((payoff_pre, payoff_post)) = self.compute_multi_horizon_payoff(signal_dir, entry_price, entry_ts, ts_ms) {
                // Store for later update (avoid borrow issues)
                payoff_updates.push((cell_id, payoff_pre, payoff_post));
                to_remove.push(idx);
            }
            // Also remove if too old (>60s) to prevent memory bloat
            if ts_ms - entry_ts > 60_000 {
                to_remove.push(idx);
            }
        }

        // Apply payoff updates to t-gate and stats (SINGLE SOURCE OF TRUTH)
        // Both gate and stats use the SAME y_post_f value
        for (cell_id, payoff_pre, payoff_post) in payoff_updates {
            // Convert to f64 (payoff is in exp -8)
            let y_pre_f = payoff_pre as f64 / 100_000_000.0;
            let y_post_f = payoff_post as f64 / 100_000_000.0;

            // Update t-gate with post-cost payoff (this is what we're gating on)
            let cell = self.sprt_cells.entry(cell_id).or_insert_with(TGate::new);
            cell.update(y_post_f);

            // Track stats for diagnostics (SAME y_post_f)
            if (cell_id as usize) < 9 {
                self.cell_stats_pre[cell_id as usize].update(y_pre_f);
                self.cell_stats_post[cell_id as usize].update(y_post_f);
            }
        }

        // Remove processed signals (in reverse order to preserve indices)
        for idx in to_remove.into_iter().rev() {
            if idx < self.pending_signals.len() {
                self.pending_signals.swap_remove(idx);
            }
        }

        // Compute CRT residual
        let r_hat = self.compute_r_hat();
        let r_actual = self.micro.prev_ret_mantissa;
        let epsilon = r_actual - r_hat;

        // Update rolling stats and compute z-score
        self.residual_stats.push(epsilon);
        let z = self.residual_stats.robust_z(epsilon);

        // Determine signal direction
        let z_in = self.config.z_in_mantissa;
        let z_out = self.config.z_out_mantissa;
        let signal_dir: i8 = if z > z_in { -1 }       // Overshoot up → short mean reversion
                             else if z < -z_in { 1 }   // Overshoot down → long mean reversion
                             else { 0 };

        // Get regime cell and extract needed values before borrowing self again
        let cell_id = self.get_cell_id();

        // Record signal for multi-horizon SPRT update (instead of immediate 1-step payoff)
        if signal_dir != 0 {
            self.total_signals += 1;
            // CRITICAL: Use actual fill price (bid/ask), not mid, to match real execution
            // Long entry at ask, short entry at bid
            let entry_px_for_payoff = if signal_dir > 0 {
                self.micro.ask_price_mantissa
            } else {
                self.micro.bid_price_mantissa
            };
            self.pending_signals.push((self.current_tick, signal_dir, cell_id, entry_px_for_payoff, ts_ms));
        }

        // Check if evidence gate is ON (extract values from cell)
        // If bypass_sprt is set, always allow trading (research mode)
        let (evidence_on, cell_mean_f, cell_var_f) = {
            if self.config.bypass_sprt {
                // Bypass mode: always on, use default sizing
                (true, 0.00001_f64, 0.0001_f64) // 1 bps mean, 10 bps std
            } else {
                let cell = self.sprt_cells.get(&cell_id);
                match cell {
                    Some(c) => {
                        let on = c.is_on() && c.n >= self.config.min_obs_per_cell as u64;
                        (on, c.mean, c.var())
                    }
                    None => (false, 0.0, 0.0001),
                }
            }
        };

        // Check drawdown stop
        if self.is_dd_stopped() {
            // Force flatten if we have position
            if !self.is_flat() {
                let decision = self.create_decision(ctx, 0, "exit", "dd_stop", self.position_qty_mantissa.abs());
                let side = if self.is_long() { Side::Sell } else { Side::Buy };
                let intent = self.create_intent(decision.decision_id, ctx, side, self.position_qty_mantissa.abs(), "dd_stop");
                return vec![DecisionOutput::new(decision, intent)];
            }
            return vec![];
        }

        let mut outputs = vec![];

        // Get current prices for logging
        let bid = self.bid_f64();
        let ask = self.ask_f64();
        let mid = self.mid_f64();
        let z_f = z as f64 / 100.0; // Convert from exp -2 to actual z-score

        if self.pos.is_flat() {
            // ========== ENTRY LOGIC ==========
            // Check 1-tick spread + low-kappa veto
            let vetoed = self.is_vetoed_regime();

            // State machine: time-based cooldown check
            let can_enter = self.pos.can_enter(ts_ms);

            // T-gate is the sole gatekeeper: only trade when t-stat > T_ON AND mean > MIN_EDGE
            // No static cell whitelist - let the gate adapt to which cells have edge
            // MIN_EDGE filters out marginal cells (e.g., 1.8 bps) that don't survive variance
            let has_positive_edge = evidence_on && cell_mean_f > MIN_EDGE;

            if signal_dir != 0 && has_positive_edge && !vetoed && can_enter {
                let w = self.compute_size_from_stats_f64(cell_mean_f, cell_var_f);
                let side = signal_dir; // +1 long, -1 short
                let entry_px = self.entry_fill_px(side);

                // Update position state
                self.pos.side = side;
                self.pos.entry_ts_ms = ts_ms;
                self.pos.entry_px = entry_px;
                self.pos.entry_mid = mid;
                self.pos.entry_bid = bid;
                self.pos.entry_ask = ask;
                self.pos.entry_cell = cell_id;
                self.pos.entry_z = z_f;
                self.pos.entry_w = w as f64 / self.config.position_size_mantissa as f64;

                // Trade logging: first 20 entries
                if self.trade_log_count < 20 {
                    println!(
                        "ENTER ts={} cell={} side={:+} z={:.3} w={:.4} bid={:.2} ask={:.2} entry_px={:.2}",
                        ts_ms, cell_id, side, z_f, self.pos.entry_w, bid, ask, entry_px
                    );
                }

                // Create order
                let (order_side, tag) = if side > 0 {
                    (Side::Buy, "crt_entry_long")
                } else {
                    (Side::Sell, "crt_entry_short")
                };

                let decision = self.create_decision(ctx, side, "entry", tag, w);
                let intent = self.create_intent(decision.decision_id, ctx, order_side, w, tag);
                outputs.push(DecisionOutput::new(decision, intent));
                self.trades_taken += 1;
            } else if signal_dir != 0 {
                self.signals_gated += 1;
            }
        } else {
            // ========== EXIT LOGIC (TIME-BASED + STOP-LOSS) ==========
            let hold_ms = ts_ms - self.pos.entry_ts_ms;
            let z_abs = z.abs();

            // Stop-loss: exit immediately if unrealized loss > STOP_LOSS
            let side = self.pos.side as f64;
            let unrealized_ret = if self.pos.entry_px > 0.0 {
                (mid / self.pos.entry_px - 1.0) * side
            } else { 0.0 };
            let stop_loss_exit = unrealized_ret < -STOP_LOSS;

            // Exit conditions (ALL TIME-BASED):
            // 1. Stop-loss hit - immediate exit to cut losers
            // 2. Target horizon hit (10s) - primary exit
            // 3. Max hold hit (10s) - safety
            // 4. Z reverted AND min_hold passed
            let target_exit = self.pos.target_horizon_hit(ts_ms);
            let max_exit = self.pos.max_hold_hit(ts_ms);
            let z_exit = z_abs <= z_out && self.pos.can_exit(ts_ms);

            let should_exit = stop_loss_exit || target_exit || max_exit || z_exit;

            if should_exit {
                let side = self.pos.side;
                let exit_px = self.exit_fill_px(side);

                // Compute returns for logging
                let mid_ret = if self.pos.entry_mid > 0.0 {
                    (mid / self.pos.entry_mid).ln() * (side as f64)
                } else { 0.0 };
                let exec_ret = if self.pos.entry_px > 0.0 {
                    (exit_px / self.pos.entry_px).ln() * (side as f64)
                } else { 0.0 };

                // Cost (round-trip in return space)
                let cost = 2.0 * self.config.taker_fee_bps_mantissa as f64 * 0.0001 / 100.0; // bps to fraction
                let pnl = exec_ret - cost;

                // Trade logging: first 20 exits
                if self.trade_log_count < 20 {
                    println!(
                        "EXIT  ts={} cell={} side={:+} hold_ms={} exit_bid={:.2} exit_ask={:.2} exit_px={:.2} mid_ret={:.6e} exec_ret={:.6e} cost={:.6e} pnl={:.6e}",
                        ts_ms, self.pos.entry_cell, side, hold_ms, bid, ask, exit_px, mid_ret, exec_ret, cost, pnl
                    );
                    self.trade_log_count += 1;
                }

                // Record hold duration for stats
                self.hold_durations_ms.push(hold_ms);

                // Create order
                let (order_side, tag) = if side > 0 {
                    (Side::Sell, "crt_exit_long")
                } else {
                    (Side::Buy, "crt_exit_short")
                };

                let decision = self.create_decision(ctx, 0, "exit", tag, self.position_qty_mantissa.abs());
                let intent = self.create_intent(decision.decision_id, ctx, order_side, self.position_qty_mantissa.abs(), tag);
                outputs.push(DecisionOutput::new(decision, intent));

                // Reset position state
                self.pos.last_exit_ts_ms = ts_ms;
                self.pos.side = 0;
            }
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, _ctx: &StrategyContext) {
        match fill.side {
            Side::Buy => self.position_qty_mantissa += fill.qty_mantissa,
            Side::Sell => self.position_qty_mantissa -= fill.qty_mantissa,
        }

        // Update equity (simplified: based on fill price * qty)
        // In real impl, track PnL properly
        let fill_value = (fill.price_mantissa * fill.qty_mantissa) / 100_000_000;
        // This is a placeholder; real equity tracking needs full PnL calculation
        self.current_equity_mantissa = self.current_equity_mantissa.saturating_add(
            if fill.side == Side::Sell && self.is_flat() { fill_value / 100 } else { 0 }
        );
        self.peak_equity_mantissa = self.peak_equity_mantissa.max(self.current_equity_mantissa);
    }

    fn print_diagnostics(&self) {
        // Cost debug: verify the bps→exp-8 conversion is correct
        let cost_one_side = self.config.taker_fee_bps_mantissa * 100;
        let cost_rt = 2 * cost_one_side;
        let cost_frac = cost_rt as f64 / 1e8;
        let cost_bps = cost_frac * 10_000.0;
        println!("\n=== COST DEBUG ===");
        println!("taker_fee_bps_mantissa: {}", self.config.taker_fee_bps_mantissa);
        println!("cost_one_side (exp-8):  {}", cost_one_side);
        println!("cost_round_trip (exp-8): {}", cost_rt);
        println!("cost_as_fraction:       {:.8}", cost_frac);
        println!("cost_as_bps:            {:.4} bps", cost_bps);

        println!("\n=== CRT-Ω Diagnostics ===");
        println!("Total signals:   {}", self.total_signals);
        println!("Signals gated:   {} ({:.1}%)",
            self.signals_gated,
            if self.total_signals > 0 {
                self.signals_gated as f64 / self.total_signals as f64 * 100.0
            } else { 0.0 }
        );
        println!("Trades taken:    {}", self.trades_taken);
        println!("SPRT cells:      {} (9 max: kappa×spread)", self.sprt_cells.len());
        println!("Cells with edge: {}",
            self.sprt_cells.values().filter(|c| c.is_on()).count()
        );
        println!("Pending signals: {}", self.pending_signals.len());
        println!("Kappa median:    {}", self.kappa_median);
        println!("Kappa samples:   valid={} zero={}", self.kappa_valid_count, self.kappa_zero_count);
        println!("Current tick:    {}", self.current_tick);

        // Per-cell payoff statistics (THE CRITICAL DIAGNOSTIC)
        println!("\n=== PER-CELL PAYOFF STATS (PRE-COST) ===");
        println!("{:>4} {:>8} {:>12} {:>12} {:>8}", "cell", "n", "mean", "std", "hit%");
        for cid in 0..9 {
            let s = &self.cell_stats_pre[cid];
            if s.n < 10 {
                continue;
            }
            println!("{:>4} {:>8} {:>12.6} {:>12.6} {:>7.1}%",
                cid, s.n, s.mean, s.std(), s.hit_rate() * 100.0);
        }

        println!("\n=== PER-CELL PAYOFF STATS (POST-COST) ===");
        println!("{:>4} {:>8} {:>12} {:>12} {:>8}", "cell", "n", "mean", "std", "hit%");
        for cid in 0..9 {
            let s = &self.cell_stats_post[cid];
            if s.n < 10 {
                continue;
            }
            println!("{:>4} {:>8} {:>12.6} {:>12.6} {:>7.1}%",
                cid, s.n, s.mean, s.std(), s.hit_rate() * 100.0);
        }

        // T-GATE STATUS PER CELL (THE CRITICAL DEBUG)
        println!("\n=== T-GATE STATUS ===");
        println!("{:>4} {:>8} {:>12} {:>12} {:>8} {:>8}", "cell", "n", "mean", "std", "t-stat", "ON?");
        for cid in 0..9u32 {
            if let Some(g) = self.sprt_cells.get(&cid) {
                if g.n >= 10 {
                    println!("{:>4} {:>8} {:>12.6} {:>12.6} {:>8.2} {:>8}",
                        cid, g.n, g.mean, g.std(), g.t_stat(),
                        if g.is_on() { "YES" } else { "no" });
                }
            }
        }

        // Verify wiring: gate.n must match stats.n
        println!("\n=== WIRING CHECK (gate.n vs stats.n) ===");
        for cid in 0..9u32 {
            let gate_n = self.sprt_cells.get(&cid).map(|g| g.n).unwrap_or(0);
            let stats_n = self.cell_stats_post[cid as usize].n;
            let match_str = if gate_n == stats_n { "OK" } else { "MISMATCH!" };
            if gate_n > 0 || stats_n > 0 {
                println!("  cell {}: gate.n={} stats.n={} {}", cid, gate_n, stats_n, match_str);
            }
        }

        // Summary: is there any cell with pre-cost edge > 0?
        let total_pre: u64 = self.cell_stats_pre.iter().map(|s| s.n).sum();
        let cells_with_pre_edge: usize = self.cell_stats_pre.iter()
            .filter(|s| s.n >= 50 && s.mean > 0.0)
            .count();
        let cells_with_post_edge: usize = self.cell_stats_post.iter()
            .filter(|s| s.n >= 50 && s.mean > 0.0)
            .count();
        let gates_on: usize = self.sprt_cells.values().filter(|c| c.is_on()).count();

        println!("\n=== EDGE SUMMARY ===");
        println!("Total payoffs computed: {}", total_pre);
        println!("Cells with pre-cost edge (mean>0, n>=50): {}/9", cells_with_pre_edge);
        println!("Cells with post-cost edge (mean>0, n>=50): {}/9", cells_with_post_edge);
        println!("T-gates currently ON: {}/9", gates_on);

        if gates_on > 0 {
            println!("DIAGNOSIS: {} gate(s) ON → trades should happen!", gates_on);
        } else if cells_with_post_edge > 0 {
            println!("DIAGNOSIS: Edge exists in {} cell(s) but gates NOT on → check t-stat thresholds", cells_with_post_edge);
        } else if cells_with_pre_edge > 0 && cells_with_post_edge == 0 {
            println!("DIAGNOSIS: Edge exists PRE-COST but costs kill it → execution problem");
        } else {
            println!("DIAGNOSIS: No edge at all → alpha definition problem");
        }

        // Hold duration statistics
        if !self.hold_durations_ms.is_empty() {
            let mut sorted = self.hold_durations_ms.clone();
            sorted.sort_unstable();
            let n = sorted.len();
            let min = sorted[0];
            let p50 = sorted[n / 2];
            let p95 = sorted[(n * 95) / 100];
            let max = sorted[n - 1];
            let mean: f64 = sorted.iter().map(|&x| x as f64).sum::<f64>() / n as f64;

            println!("\n=== HOLD DURATION STATS (ms) ===");
            println!("n={} min={} p50={} p95={} max={} mean={:.0}", n, min, p50, p95, max, mean);
            println!("Target: MIN_HOLD={}ms MAX_HOLD={}ms HORIZON={}ms", MIN_HOLD_MS, MAX_HOLD_MS, TARGET_HORIZON_MS);
        }
    }
}

/// Factory function for registry.
pub fn crt_omega_factory(config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
    let config = match config_path {
        Some(path) => CrtOmegaConfig::from_toml(path)?,
        None => CrtOmegaConfig::default(),
    };
    Ok(Box::new(CrtOmegaStrategy::new(config)))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Summarize a feature array for debugging bucket edge collapse.
/// Call this during warmup to diagnose which features are degenerate.
#[allow(dead_code)]
fn summarize_feature(name: &str, values: &[i64]) {
    let finite: Vec<i64> = values.iter().copied().filter(|&x| x != i64::MIN && x != i64::MAX).collect();
    println!("\n[{name}] n={} finite={}", values.len(), finite.len());
    if finite.is_empty() { return; }

    let mut sorted = finite.clone();
    sorted.sort_unstable();

    // Count approximately unique values
    let mut unique_count = 0usize;
    let mut last = i64::MIN;
    for &z in &sorted {
        if z != last {
            unique_count += 1;
            last = z;
        }
    }
    println!("[{name}] approx_unique={unique_count}");

    // Print quantiles
    let qs = [0.0, 0.01, 0.10, 0.33, 0.50, 0.66, 0.90, 0.99, 1.0];
    for q in qs {
        let idx = ((q * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
        println!("[{name}] q={:>4.2} -> {}", q, sorted[idx]);
    }
}

/// Compute robust quantile edges for 3 buckets that guarantees non-degenerate bins.
/// Returns 4 edges: [e0, e1, e2, e3] representing bins [e0,e1), [e1,e2), [e2,e3].
fn compute_quantile_edges_robust(values: &[i64], num_buckets: usize) -> Vec<i64> {
    // Filter out invalid values
    let mut sorted: Vec<i64> = values.iter()
        .copied()
        .filter(|&x| x != i64::MIN && x != i64::MAX && x != 0) // Also filter zeros for spread/kappa
        .collect();

    // Fallback if empty or too few values
    if sorted.is_empty() {
        // Also try with zeros included
        sorted = values.iter()
            .copied()
            .filter(|&x| x != i64::MIN && x != i64::MAX)
            .collect();
    }

    if sorted.is_empty() {
        // Hard fallback: return well-separated edges
        return (0..=num_buckets).map(|i| i as i64 * 1000).collect();
    }

    sorted.sort_unstable();
    let n = sorted.len();

    // Compute raw quantile edges
    let mut edges: Vec<i64> = Vec::with_capacity(num_buckets + 1);
    for i in 0..=num_buckets {
        let p = i as f64 / num_buckets as f64;
        let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
        edges.push(sorted[idx]);
    }

    // Compute scale-aware epsilon for forcing monotonicity
    let range = (edges[num_buckets] - edges[0]).abs();
    let scale = if range > 0 { range } else { 1_000_000 }; // Default scale if flat
    let eps = (scale / 1_000_000).max(1); // At least 1

    // Enforce strictly increasing edges
    for i in 1..=num_buckets {
        if edges[i] <= edges[i - 1] {
            edges[i] = edges[i - 1] + eps;
        }
    }

    // If still basically flat, spread artificially
    if edges[num_buckets] - edges[0] < eps * num_buckets as i64 {
        let base = edges[0];
        for i in 0..=num_buckets {
            edges[i] = base + (i as i64 * eps * 100);
        }
    }

    // Log edges for debugging
    tracing::debug!(
        "quantile_edges: num_buckets={} n={} edges={:?}",
        num_buckets, n, edges
    );

    edges
}

/// Bin a value into one of 3 buckets using edges.
/// edges must have 4 elements for 3 buckets: [e0, e1, e2, e3].
fn bucket_index_robust(value: i64, edges: &[i64]) -> u32 {
    if edges.len() < 2 {
        return 0;
    }

    let num_buckets = edges.len() - 1;

    // Binary search style: find which bucket value falls into
    for i in 1..edges.len() {
        if value < edges[i] {
            return (i - 1) as u32;
        }
    }

    // Value >= last edge, return last bucket
    (num_buckets - 1) as u32
}

// Legacy wrapper for compatibility
fn compute_quantile_edges(values: &[i64], num_buckets: usize) -> Vec<i64> {
    compute_quantile_edges_robust(values, num_buckets)
}

fn bucket_index(value: i64, edges: &[i64]) -> u32 {
    bucket_index_robust(value, edges)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_canonical_bytes_deterministic() {
        let config1 = CrtOmegaConfig::default();
        let config2 = CrtOmegaConfig::default();
        assert_eq!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_config_different_values_different_hash() {
        let config1 = CrtOmegaConfig::default();
        let config2 = CrtOmegaConfig {
            z_in_mantissa: 400, // Different
            ..CrtOmegaConfig::default()
        };
        assert_ne!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_strategy_id_format() {
        let strategy = CrtOmegaStrategy::new(CrtOmegaConfig::default());
        let id = strategy.strategy_id();
        let parts: Vec<&str> = id.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], CRT_OMEGA_NAME);
        assert_eq!(parts[1], CRT_OMEGA_VERSION);
        assert_eq!(parts[2].len(), 64); // SHA-256
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(100);
        for i in 0..100 {
            stats.push(i * 1000);
        }
        // Should be able to compute z-score
        let z = stats.robust_z(50000);
        assert!(z.abs() < 200); // Roughly zero for middle value
    }

    #[test]
    fn test_tgate_enables_on_positive_edge() {
        let mut gate = TGate::new();
        // Feed positive observations (4 bps = 0.0004)
        for _ in 0..100 {
            gate.update(0.0004); // Positive edge
        }
        // Should have enabled with high t-stat
        assert!(gate.n >= 30);
        assert!(gate.t_stat() > 3.0);
        assert!(gate.is_on());
    }

    #[test]
    fn test_bucket_index() {
        let edges = vec![0, 100, 200, 300];
        assert_eq!(bucket_index(50, &edges), 0);
        assert_eq!(bucket_index(150, &edges), 1);
        assert_eq!(bucket_index(250, &edges), 2);
        assert_eq!(bucket_index(350, &edges), 2); // Above max goes to last bucket
    }

    #[test]
    fn test_robust_quantile_edges_degenerate() {
        // Test with all same values - should still produce 4 distinct edges
        let values = vec![100i64; 500];
        let edges = compute_quantile_edges_robust(&values, 3);
        assert_eq!(edges.len(), 4);
        // Edges must be strictly increasing
        for i in 1..edges.len() {
            assert!(edges[i] > edges[i - 1], "edges must be strictly increasing: {:?}", edges);
        }
    }

    #[test]
    fn test_robust_quantile_edges_varied() {
        // Test with varied values
        let values: Vec<i64> = (0..500).map(|i| i * 100).collect();
        let edges = compute_quantile_edges_robust(&values, 3);
        assert_eq!(edges.len(), 4);
        // Should span the range
        assert!(edges[0] <= 100);
        assert!(edges[3] >= 49000);
        // Edges must be strictly increasing
        for i in 1..edges.len() {
            assert!(edges[i] > edges[i - 1]);
        }
    }

    #[test]
    fn test_robust_quantile_edges_with_zeros() {
        // Test with many zeros (like spread might have)
        let mut values = vec![0i64; 400];
        values.extend((1..101).map(|i| i * 10));
        let edges = compute_quantile_edges_robust(&values, 3);
        assert_eq!(edges.len(), 4);
        // Should use the non-zero values for edges
        for i in 1..edges.len() {
            assert!(edges[i] > edges[i - 1], "edges must be strictly increasing: {:?}", edges);
        }
    }
}
