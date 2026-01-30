//! Order Generation Module for India Strategies
//!
//! Deterministic order generation: same replay + config = same orders.
//! Output is compatible with `backtest-kitesim --orders`.
//!
//! ## Strategies
//! - `india_micro_mm`: Single-leg microstructure scalper on ATM options
//!
//! ## Design Principles
//! - Deterministic: no RNG unless seeded
//! - Pure function: replay in, orders out
//! - No lookahead: decisions use only past data
//!
//! ## Regime Detection (Phase 28)
//! - Grassmann manifold-based regime detection
//! - Optional regime gating: only trade in favorable regimes

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use quantlaxmi_options::execution::{LegOrder, LegOrderType, LegSide, MultiLegOrder};
use quantlaxmi_options::replay::QuoteEvent;
use quantlaxmi_regime::{
    cpd::CusumDetector,
    features::FeatureVector,
    lift::{RegimeLift, RegimeLiftConfig},
    prototypes::RegimeLabel,
    ramanujan::{MicrostructurePeriodicity, PeriodicityFeatures},
};

use crate::kitesim_backtest::get_nse_lot_size;

/// Minimum spread in price units to place maker orders.
/// If spread < MIN_SPREAD, skip (no edge / bad data).
const MIN_SPREAD: f64 = 0.05; // NSE tick size

/// Returns the maker limit price for a given side.
/// Buy → post at bid (passive), Sell → post at ask (passive).
fn maker_limit_price(side: LegSide, bid: f64, ask: f64) -> f64 {
    match side {
        LegSide::Buy => bid,
        LegSide::Sell => ask,
    }
}

/// Check if quote has valid book for maker quoting.
/// Returns None if book is invalid (crossed, locked, or insufficient spread).
fn is_valid_maker_book(bid: f64, ask: f64) -> bool {
    bid.is_finite() && ask.is_finite() && ask > bid && (ask - bid) >= MIN_SPREAD
}

/// Strategy configuration for india_micro_mm
#[derive(Debug, Clone, Deserialize)]
pub struct MicroMmConfig {
    /// Maximum spread in bps to enter
    #[serde(default = "default_max_spread_bps")]
    pub max_spread_bps: f64,

    /// Minimum hold time in milliseconds
    #[serde(default = "default_min_hold_ms")]
    pub min_hold_ms: i64,

    /// Maximum hold time in milliseconds (force exit)
    #[serde(default = "default_max_hold_ms")]
    pub max_hold_ms: i64,

    /// Pressure threshold (bid_qty / ask_qty) to enter long
    #[serde(default = "default_pressure_long")]
    pub pressure_long: f64,

    /// Pressure threshold (ask_qty / bid_qty) to enter short
    #[serde(default = "default_pressure_short")]
    pub pressure_short: f64,

    /// Symbols to trade (if empty, trade all in replay)
    #[serde(default)]
    pub symbols: Vec<String>,

    /// Quantity per trade (in lots)
    #[serde(default = "default_lots")]
    pub lots: u32,

    // === Gate B1 Routing Config ===
    /// Max spread to post LIMIT (wider = use MARKET or skip)
    #[serde(default = "default_spread_bps_limit_max")]
    pub spread_bps_limit_max: f64,

    /// Max spread to use MARKET (wider = skip entirely)
    #[serde(default = "default_spread_bps_market_max")]
    pub spread_bps_market_max: f64,

    /// Min mid velocity (bps/sec) to trigger MARKET
    #[serde(default = "default_vel_bps_sec_market_min")]
    pub vel_bps_sec_market_min: f64,

    /// Min signal strength to trigger MARKET
    #[serde(default = "default_signal_strength_market_min")]
    pub signal_strength_market_min: f64,

    /// Min dt (ms) for velocity calculation to avoid blow-ups
    #[serde(default = "default_vel_dt_min_ms")]
    pub vel_dt_min_ms: i64,

    /// Max dt (ms) for velocity calculation (stale = reset)
    #[serde(default = "default_vel_dt_max_ms")]
    pub vel_dt_max_ms: i64,

    // === Phase 28: Regime Detection Config ===
    /// Enable regime gating (skip trades during unfavorable regimes)
    #[serde(default)]
    pub regime_gating_enabled: bool,

    /// Rolling window size for regime detection (default: 64 quotes)
    #[serde(default = "default_regime_window_size")]
    pub regime_window_size: usize,

    /// Subspace dimension k for Gr(k,6) (default: 3)
    #[serde(default = "default_regime_subspace_dim")]
    pub regime_subspace_dim: usize,

    /// CUSUM threshold for regime shift detection (mantissa, exp=-4)
    #[serde(default = "default_regime_cusum_threshold")]
    pub regime_cusum_threshold: i64,

    /// Allowed regime labels for trading (empty = trade all)
    #[serde(default)]
    pub regime_allowed_labels: Vec<String>,

    // === Phase 29: Ramanujan Periodicity Detection ===
    /// Enable Ramanujan periodicity filtering (block on HFT detection)
    #[serde(default)]
    pub ramanujan_enabled: bool,

    /// Max period Q to detect (default: 16)
    #[serde(default = "default_ramanujan_max_period")]
    pub ramanujan_max_period: usize,

    /// Detection threshold (energy ratio, default: 0.3)
    #[serde(default = "default_ramanujan_threshold")]
    pub ramanujan_threshold: f64,

    /// Block trading when HFT activity detected
    #[serde(default = "default_ramanujan_block_on_hft")]
    pub ramanujan_block_on_hft: bool,
}

fn default_max_spread_bps() -> f64 {
    30.0
}
fn default_min_hold_ms() -> i64 {
    30_000
}
fn default_max_hold_ms() -> i64 {
    90_000
}
fn default_pressure_long() -> f64 {
    1.5
}
fn default_pressure_short() -> f64 {
    1.5
}
fn default_lots() -> u32 {
    1
}
// Gate B1 routing defaults
fn default_spread_bps_limit_max() -> f64 {
    80.0
}
fn default_spread_bps_market_max() -> f64 {
    120.0
}
fn default_vel_bps_sec_market_min() -> f64 {
    250.0 // 25 bps/100ms = 250 bps/sec
}
fn default_signal_strength_market_min() -> f64 {
    0.05
}
fn default_vel_dt_min_ms() -> i64 {
    10 // Minimum 10ms between quotes for velocity
}
fn default_vel_dt_max_ms() -> i64 {
    5000 // 5 seconds = stale, reset velocity
}
// Phase 28: Regime detection defaults
fn default_regime_window_size() -> usize {
    64 // 64 quotes for rolling covariance
}
fn default_regime_subspace_dim() -> usize {
    3 // k=3 for Gr(3,6)
}
fn default_regime_cusum_threshold() -> i64 {
    5000 // 0.5 in mantissa with exp=-4
}
// Phase 29: Ramanujan periodicity defaults
fn default_ramanujan_max_period() -> usize {
    16 // Detect periods up to 16 quotes
}
fn default_ramanujan_threshold() -> f64 {
    0.3 // 30% energy ratio threshold
}
fn default_ramanujan_block_on_hft() -> bool {
    true // Block trading during detected HFT activity
}

impl Default for MicroMmConfig {
    fn default() -> Self {
        Self {
            max_spread_bps: default_max_spread_bps(),
            min_hold_ms: default_min_hold_ms(),
            max_hold_ms: default_max_hold_ms(),
            pressure_long: default_pressure_long(),
            pressure_short: default_pressure_short(),
            symbols: Vec::new(),
            lots: default_lots(),
            // Gate B1
            spread_bps_limit_max: default_spread_bps_limit_max(),
            spread_bps_market_max: default_spread_bps_market_max(),
            vel_bps_sec_market_min: default_vel_bps_sec_market_min(),
            signal_strength_market_min: default_signal_strength_market_min(),
            vel_dt_min_ms: default_vel_dt_min_ms(),
            vel_dt_max_ms: default_vel_dt_max_ms(),
            // Phase 28: Regime detection
            regime_gating_enabled: false,
            regime_window_size: default_regime_window_size(),
            regime_subspace_dim: default_regime_subspace_dim(),
            regime_cusum_threshold: default_regime_cusum_threshold(),
            regime_allowed_labels: Vec::new(),
            // Phase 29: Ramanujan periodicity
            ramanujan_enabled: false,
            ramanujan_max_period: default_ramanujan_max_period(),
            ramanujan_threshold: default_ramanujan_threshold(),
            ramanujan_block_on_hft: default_ramanujan_block_on_hft(),
        }
    }
}

/// Order file output format (compatible with backtest-kitesim)
#[derive(Debug, Clone, Serialize)]
pub struct OrderFile {
    pub strategy_name: String,
    pub orders: Vec<MultiLegOrder>,
}

// =============================================================================
// Gate B1.3: routing_decisions.jsonl schema (quantlaxmi.routing_decisions.v1)
// =============================================================================

/// Routing config snapshot for run header
#[derive(Debug, Clone, Serialize)]
pub struct RoutingConfigSnapshot {
    pub spread_bps_limit_max: f64,
    pub spread_bps_market_max: f64,
    pub vel_bps_sec_market_min: f64,
    pub signal_strength_market_min: f64,
    pub vel_dt_min_ms: i64,
    pub vel_dt_max_ms: i64,
}

impl From<&MicroMmConfig> for RoutingConfigSnapshot {
    fn from(cfg: &MicroMmConfig) -> Self {
        Self {
            spread_bps_limit_max: cfg.spread_bps_limit_max,
            spread_bps_market_max: cfg.spread_bps_market_max,
            vel_bps_sec_market_min: cfg.vel_bps_sec_market_min,
            signal_strength_market_min: cfg.signal_strength_market_min,
            vel_dt_min_ms: cfg.vel_dt_min_ms,
            vel_dt_max_ms: cfg.vel_dt_max_ms,
        }
    }
}

impl RoutingConfigSnapshot {
    /// Compute SHA256 hash of canonical config representation.
    /// Note: serde_json serialization of flat structs (no maps) is deterministic.
    /// If maps are added in future, use serde_json::to_value() + sort keys.
    pub fn compute_hash(&self) -> String {
        // Use expect() to fail loudly on serialization errors rather than
        // silently collapsing to empty string (which would give identical hashes)
        let canonical =
            serde_json::to_string(self).expect("RoutingConfigSnapshot must serialize to JSON");
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let result = hasher.finalize();
        format!("sha256:{}", hex::encode(&result[..16]))
    }
}

/// Run header record (first line of routing_decisions.jsonl)
#[derive(Debug, Clone, Serialize)]
pub struct RoutingRunHeader {
    pub record_type: &'static str,
    pub schema: &'static str,
    /// Schema revision for forward-compat (v1 family, rev 2 = B1.3 final)
    pub schema_rev: u32,
    pub gate: &'static str,
    pub strategy: String,
    pub run_id: String,
    pub replay_path: String,
    pub generated_at_utc: String,
    pub config: RoutingConfigSnapshot,
    /// SHA256 of canonical config JSON for cross-run comparison
    pub config_hash: String,
}

impl RoutingRunHeader {
    pub fn new(strategy: &str, run_id: &str, replay_path: &str, config: &MicroMmConfig) -> Self {
        let config_snapshot = RoutingConfigSnapshot::from(config);
        let config_hash = config_snapshot.compute_hash();
        Self {
            record_type: "run_header",
            schema: "quantlaxmi.routing_decisions.v1",
            schema_rev: 2, // B1.3 final: added vel_used, dt_ms, config_hash, intent_id
            gate: "IND_KITESIM_GATE_B1",
            strategy: strategy.to_string(),
            run_id: run_id.to_string(),
            replay_path: replay_path.to_string(),
            generated_at_utc: chrono::Utc::now().to_rfc3339(),
            config: config_snapshot,
            config_hash,
        }
    }
}

/// Quote snapshot at decision time
#[derive(Debug, Clone, Serialize)]
pub struct QuoteSnapshot {
    pub bid: f64,
    pub ask: f64,
    pub bid_qty: u64,
    pub ask_qty: u64,
    pub mid: f64,
}

/// Feature vector at decision time
#[derive(Debug, Clone, Serialize)]
pub struct RoutingFeatures {
    pub spread_bps: f64,
    pub pressure: f64,
    /// Velocity in bps/sec (0.0 when dt out of bounds or first quote)
    pub vel_bps_sec: f64,
    /// Absolute velocity (useful for models)
    pub vel_abs_bps_sec: f64,
    /// Whether velocity was used in routing decision (dt in bounds)
    pub vel_used: bool,
    /// Time since previous actionable quote (ms), 0 on first quote
    pub dt_ms: i64,
    pub signal_strength: f64,
}

/// Thresholds snapshot as applied (important when config evolves)
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdsSnapshot {
    pub spread_bps_limit_max: f64,
    pub spread_bps_market_max: f64,
    pub vel_bps_sec_market_min: f64,
    pub signal_strength_market_min: f64,
    /// Config bounds for velocity dt
    pub vel_dt_min_ms: i64,
    pub vel_dt_max_ms: i64,
    /// Whether observed dt was within [vel_dt_min_ms, vel_dt_max_ms]
    pub vel_dt_in_bounds: bool,
}

/// Signal interpretation
#[derive(Debug, Clone, Serialize)]
pub struct SignalSnapshot {
    pub side: String,
    /// +1 for Buy, -1 for Sell (model-friendly)
    pub direction: i8,
    pub should_trade: bool,
    pub fast_move: bool,
    pub strong_signal: bool,
}

/// Decision output
#[derive(Debug, Clone, Serialize)]
pub struct DecisionOutput {
    pub order_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<f64>,
}

/// Reason tracking
#[derive(Debug, Clone, Serialize)]
pub struct DecisionReason {
    pub primary: String,
    pub flags: Vec<String>,
}

/// Join IDs for linking to orders/fills
#[derive(Debug, Clone, Serialize)]
pub struct DecisionIds {
    /// Content-addressed ID (sha256 of decision inputs + outputs)
    pub decision_id: String,
    /// Stable intent ID for cross-run joins (same as decision_id when order_type+price included)
    pub intent_id: String,
    /// Human-readable emission ID (strategy_side_symbol:ts)
    pub order_id: String,
    pub leg_index: u32,
}

/// Per-decision record (one per emitted entry leg)
#[derive(Debug, Clone, Serialize)]
pub struct RoutingDecisionRecord {
    pub record_type: &'static str,
    pub schema: &'static str,
    /// Schema revision for forward-compat (v1 family, rev 2 = B1.3 final)
    pub schema_rev: u32,
    pub ts_utc: String,
    pub symbol: String,
    pub exchange: String,
    pub quote: QuoteSnapshot,
    pub features: RoutingFeatures,
    pub thresholds: ThresholdsSnapshot,
    pub signal: SignalSnapshot,
    pub decision: DecisionOutput,
    pub reason: DecisionReason,
    pub ids: DecisionIds,
}

impl RoutingDecisionRecord {
    /// Compute deterministic decision_id from content (sha256).
    /// Includes order_type + price for intent stability across config evolution.
    ///
    /// Price formatting: {:.2} is safe because:
    /// - NSE tick size is ₹0.05 (2 decimal places)
    /// - maker_limit_price() returns tick-aligned prices (bid/ask from quotes)
    /// - MARKET orders have price=None ("None" in hash)
    #[allow(clippy::too_many_arguments)]
    pub fn compute_decision_id(
        ts_utc: &str,
        symbol: &str,
        side: &str,
        bid: f64,
        ask: f64,
        pressure: f64,
        vel_bps_sec: f64,
        spread_bps_limit_max: f64,
        spread_bps_market_max: f64,
        vel_bps_sec_market_min: f64,
        signal_strength_market_min: f64,
        order_type: &str,
        price: Option<f64>,
    ) -> String {
        // Price is tick-aligned (NSE tick = 0.05), so {:.2} is deterministic
        let price_str = price
            .map(|p| format!("{:.2}", p))
            .unwrap_or_else(|| "None".to_string());
        let canonical = format!(
            "{}|{}|{}|{:.6}|{:.6}|{:.6}|{:.6}|{:.1}|{:.1}|{:.1}|{:.3}|{}|{}",
            ts_utc,
            symbol,
            side,
            bid,
            ask,
            pressure,
            vel_bps_sec,
            spread_bps_limit_max,
            spread_bps_market_max,
            vel_bps_sec_market_min,
            signal_strength_market_min,
            order_type,
            price_str
        );
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let result = hasher.finalize();
        format!("sha256:{}", hex::encode(&result[..16])) // Truncate to 128-bit for brevity
    }
}

/// Run footer record (last line of routing_decisions.jsonl)
#[derive(Debug, Clone, Serialize)]
pub struct RoutingRunFooter {
    pub record_type: &'static str,
    pub schema: &'static str,
    /// Schema revision for forward-compat (v1 family, rev 2 = B1.3 final)
    pub schema_rev: u32,
    pub counts: RoutingCounts,
}

/// Summary counts for footer
#[derive(Debug, Clone, Default, Serialize)]
pub struct RoutingCounts {
    pub decisions: u64,
    pub limit: u64,
    pub market: u64,
    pub market_by_vel: u64,
    pub market_by_strength: u64,
    pub market_by_both: u64,
    pub skipped_spread_band: u64,
    pub skipped_book_invalid: u64,
}

impl RoutingRunFooter {
    pub fn new(counts: RoutingCounts) -> Self {
        Self {
            record_type: "run_footer",
            schema: "quantlaxmi.routing_decisions.v1",
            schema_rev: 2, // B1.3 final
            counts,
        }
    }
}

/// Expected schema_rev for B1.3 final
const EXPECTED_SCHEMA_REV: u32 = 2;

/// Writer for streaming routing decisions to JSONL
pub struct RoutingDecisionsWriter {
    writer: BufWriter<std::fs::File>,
    counts: RoutingCounts,
    /// Expected schema_rev (from header) - enforced on all records
    expected_schema_rev: u32,
}

impl RoutingDecisionsWriter {
    pub fn new(path: &Path, header: &RoutingRunHeader) -> Result<Self> {
        // Enforce schema_rev consistency at construction
        anyhow::ensure!(
            header.schema_rev == EXPECTED_SCHEMA_REV,
            "Header schema_rev {} != expected {}",
            header.schema_rev,
            EXPECTED_SCHEMA_REV
        );

        let file = std::fs::File::create(path)
            .with_context(|| format!("create routing_decisions file: {:?}", path))?;
        let mut writer = BufWriter::new(file);

        // Write header
        let header_json = serde_json::to_string(header)?;
        writeln!(writer, "{}", header_json)?;

        Ok(Self {
            writer,
            counts: RoutingCounts::default(),
            expected_schema_rev: header.schema_rev,
        })
    }

    pub fn write_decision(&mut self, record: &RoutingDecisionRecord) -> Result<()> {
        // Enforce schema_rev consistency
        anyhow::ensure!(
            record.schema_rev == self.expected_schema_rev,
            "Decision schema_rev {} != expected {}",
            record.schema_rev,
            self.expected_schema_rev
        );
        let json = serde_json::to_string(record)?;
        writeln!(self.writer, "{}", json)?;
        self.counts.decisions += 1;
        Ok(())
    }

    pub fn inc_limit(&mut self) {
        self.counts.limit += 1;
    }

    pub fn inc_market(&mut self, by_vel: bool, by_strength: bool) {
        self.counts.market += 1;
        if by_vel && by_strength {
            self.counts.market_by_both += 1;
        }
        if by_vel {
            self.counts.market_by_vel += 1;
        }
        if by_strength {
            self.counts.market_by_strength += 1;
        }
    }

    pub fn inc_skipped_spread(&mut self) {
        self.counts.skipped_spread_band += 1;
    }

    pub fn inc_skipped_book(&mut self) {
        self.counts.skipped_book_invalid += 1;
    }

    pub fn finish(mut self) -> Result<RoutingCounts> {
        let footer = RoutingRunFooter::new(self.counts.clone());
        // Enforce schema_rev consistency
        anyhow::ensure!(
            footer.schema_rev == self.expected_schema_rev,
            "Footer schema_rev {} != expected {}",
            footer.schema_rev,
            self.expected_schema_rev
        );
        let footer_json = serde_json::to_string(&footer)?;
        writeln!(self.writer, "{}", footer_json)?;
        self.writer.flush()?;
        Ok(self.counts)
    }
}

/// Position state for tracking signals (NOT fills - generator doesn't know fills)
/// Gate B0: We only track last signal time to avoid spam, not position.
/// Gate B1: Added mid/ts tracking for velocity estimation (only on actionable quotes).
/// Phase 28: Added regime detection state.
#[derive(Debug, Clone, Default)]
struct SignalState {
    /// Last signal timestamp (to throttle order generation)
    last_signal_ts: Option<chrono::DateTime<chrono::Utc>>,
    /// Gate B1: Last observed mid on actionable quote (for velocity)
    last_mid: Option<f64>,
    /// Gate B1: Last observed quote timestamp on actionable quote (for dt)
    last_quote_ts: Option<chrono::DateTime<chrono::Utc>>,
}

/// Phase 28: Per-symbol regime detection state.
struct RegimeState {
    /// Geometric lift engine (features → subspace)
    lift: RegimeLift,
    /// CUSUM change-point detector
    cusum: CusumDetector,
    /// Current inferred regime label (heuristic until prototypes trained)
    current_regime: RegimeLabel,
    /// Whether regime lift is ready (window full)
    is_ready: bool,
    /// Number of regime shifts detected
    shift_count: u64,
    /// Phase 29: Ramanujan periodicity detector
    periodicity: Option<MicrostructurePeriodicity>,
    /// Latest periodicity features
    periodicity_features: Option<PeriodicityFeatures>,
}

impl RegimeState {
    fn new(config: &MicroMmConfig) -> Self {
        let lift_config = RegimeLiftConfig {
            n_features: 6,
            subspace_dim: config.regime_subspace_dim,
            window_size: config.regime_window_size,
        };
        // CUSUM detector: threshold from config, small drift for sensitivity, exp=-4
        let cusum = CusumDetector::new(
            config.regime_cusum_threshold,
            100, // drift_mantissa: small drift for sensitivity
            -4,  // exponent: 10^-4 scale
        );
        // Phase 29: Ramanujan periodicity detector (if enabled)
        let periodicity = if config.ramanujan_enabled {
            Some(MicrostructurePeriodicity::with_params(
                config.ramanujan_max_period,   // max_period
                4,                             // num_reps (filter replications)
                config.regime_window_size * 8, // buffer_size (8x window)
                config.ramanujan_threshold,    // threshold
                10000,                         // price_scale (basis points)
            ))
        } else {
            None
        };
        Self {
            lift: RegimeLift::new(lift_config),
            cusum,
            current_regime: RegimeLabel::Unknown,
            is_ready: false,
            shift_count: 0,
            periodicity,
            periodicity_features: None,
        }
    }

    /// Update regime state from quote, return true if regime shift detected
    fn update(&mut self, quote: &QuoteEvent, prev_quote: Option<&QuoteEvent>) -> bool {
        // Compute features from quote
        let mid = (quote.bid_f64() + quote.ask_f64()) / 2.0;

        // Mid return: bps change from previous quote
        let mid_return = if let Some(prev) = prev_quote {
            let prev_mid = (prev.bid_f64() + prev.ask_f64()) / 2.0;
            if prev_mid > 0.0 {
                (((mid - prev_mid) / prev_mid) * 10000.0) as i64
            } else {
                0
            }
        } else {
            0
        };

        // Book imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) * 10000
        let total_qty = quote.bid_qty as i64 + quote.ask_qty as i64;
        let imbalance = if total_qty > 0 {
            ((quote.bid_qty as i64 - quote.ask_qty as i64) * 10000) / total_qty
        } else {
            0
        };

        // Spread in basis points
        let spread_bps = if mid > 0.0 {
            (((quote.ask_f64() - quote.bid_f64()) / mid) * 10000.0) as i64
        } else {
            0
        };

        // Volatility proxy (spread as proxy)
        let vol_proxy = spread_bps;

        // Pressure: bid_qty * 100 / (bid_qty + ask_qty)
        let pressure = if total_qty > 0 {
            (quote.bid_qty as i64 * 100) / total_qty
        } else {
            50 // neutral
        };

        // VPIN proxy (using imbalance magnitude)
        let vpin = imbalance.abs();

        let features =
            FeatureVector::new(mid_return, imbalance, spread_bps, vol_proxy, pressure, vpin);

        // Update lift (covariance → SVD → subspace)
        let subspace = self.lift.update(&features);

        if let Some(subspace) = subspace {
            self.is_ready = true;

            // Update regime label based on eigenvalue spectrum (heuristic)
            let eigenvalues = subspace.eigenvalues();
            if eigenvalues.len() >= 2 {
                let ratio = eigenvalues[0] / eigenvalues[1].max(1e-10);

                // Heuristic regime classification based on eigenvalue concentration
                self.current_regime = if ratio > 10.0 {
                    // Highly concentrated variance = trending
                    RegimeLabel::TrendImpulse
                } else if ratio < 2.0 {
                    // Diffuse variance = choppy/mean-reverting
                    RegimeLabel::MeanReversionChop
                } else if eigenvalues[0] < 0.01 {
                    // Very low total variance = quiet
                    RegimeLabel::Quiet
                } else {
                    // Default
                    RegimeLabel::Unknown
                };
            }
        }

        // Phase 29: Update Ramanujan periodicity detector
        if let Some(ref mut periodicity) = self.periodicity {
            let ready = periodicity.update(mid_return, imbalance, spread_bps);
            if ready {
                self.periodicity_features = Some(periodicity.detect());
            }
        }

        // Check for regime shift via CUSUM (would need distance to prev subspace)
        // For now, we just track whether ready
        false
    }

    /// Check if current regime allows trading
    fn allows_trading(&self, allowed_labels: &[String], block_on_hft: bool) -> bool {
        if !self.is_ready {
            return true; // Allow trading until regime is established
        }

        // Phase 29: Block on HFT activity detected by Ramanujan periodicity
        if block_on_hft {
            if let Some(ref features) = self.periodicity_features {
                if features.hft_likely() {
                    return false; // Block: periodic HFT activity detected
                }
            }
        }

        if allowed_labels.is_empty() {
            return true; // No restrictions on regime label
        }
        let current_label_str = self.current_regime.as_str();
        allowed_labels.iter().any(|l| l == current_label_str)
    }

    /// Get periodicity features for logging
    fn periodicity_summary(&self) -> String {
        if let Some(ref features) = self.periodicity_features {
            format!(
                "periods={:?}, hft={}, mm={}",
                features.dominant_periods(),
                features.hft_likely(),
                features.market_maker_likely()
            )
        } else {
            "periodicity=disabled".to_string()
        }
    }
}

/// Compute mid price from quote
fn mid_f64(quote: &QuoteEvent) -> f64 {
    0.5 * (quote.bid_f64() + quote.ask_f64())
}

/// Compute mid velocity in bps/sec from previous to current mid
/// Returns 0.0 if dt is outside [dt_min, dt_max] or inputs invalid
fn mid_vel_bps_sec(
    prev_mid: f64,
    prev_ts: chrono::DateTime<chrono::Utc>,
    mid: f64,
    ts: chrono::DateTime<chrono::Utc>,
    dt_min_ms: i64,
    dt_max_ms: i64,
) -> f64 {
    if !prev_mid.is_finite() || prev_mid <= 0.0 || !mid.is_finite() {
        return 0.0;
    }
    let dt_ms = (ts - prev_ts).num_milliseconds();
    if dt_ms < dt_min_ms || dt_ms > dt_max_ms {
        return 0.0; // Too fast (spurious) or too stale
    }
    let dt_sec = dt_ms as f64 / 1000.0;
    let bps_move = ((mid - prev_mid) / prev_mid) * 10_000.0;
    bps_move / dt_sec
}

/// Load quotes from JSONL file
fn load_quotes(path: &Path) -> Result<Vec<QuoteEvent>> {
    let f = std::fs::File::open(path).with_context(|| format!("open replay file: {:?}", path))?;
    let br = BufReader::new(f);
    let mut out = Vec::new();

    for (i, line) in br.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", i + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let q: QuoteEvent = serde_json::from_str(&line)
            .with_context(|| format!("parse QuoteEvent JSON on line {}", i + 1))?;
        out.push(q);
    }

    Ok(out)
}

/// Calculate spread in bps from quote
fn spread_bps(quote: &QuoteEvent) -> f64 {
    let mid = (quote.bid_f64() + quote.ask_f64()) / 2.0;
    if mid <= 0.0 {
        return f64::MAX;
    }
    let spread = quote.ask_f64() - quote.bid_f64();
    (spread / mid) * 10_000.0
}

/// Calculate pressure ratio (bid_qty / ask_qty)
fn pressure_ratio(quote: &QuoteEvent) -> f64 {
    if quote.ask_qty == 0 {
        return f64::MAX;
    }
    quote.bid_qty as f64 / quote.ask_qty as f64
}

/// Phase 28/29: Regime gating statistics
#[derive(Debug, Default)]
struct RegimeGatingStats {
    quotes_processed: u64,
    regime_blocked: u64,
    regime_allowed: u64,
    /// Phase 29: Blocked due to HFT periodicity detection
    hft_blocked: u64,
}

/// Generate orders using india_micro_mm strategy
///
/// Gate B0: Entry-only, stateless signal generator.
/// Gate B1: Execution-aware routing (LIMIT vs MARKET) with dt-normalized velocity.
/// Gate B1.3: Optional routing_decisions.jsonl sidecar via writer parameter.
/// Phase 28: Grassmann manifold regime gating.
///
/// - Emits entry orders at bid (long) or ask (short) when pressure threshold met
/// - Routes LIMIT vs MARKET based on spread, velocity (bps/sec), and signal strength
/// - State (last_mid, last_quote_ts) only updated on actionable quotes
/// - Does NOT track position (generator doesn't know fills)
/// - Does NOT generate exits
/// - Throttles signals per symbol to avoid spam
/// - Phase 28: Optionally gates trading based on regime detection
fn generate_micro_mm_orders(
    quotes: &[QuoteEvent],
    config: &MicroMmConfig,
    mut routing_writer: Option<&mut RoutingDecisionsWriter>,
) -> Vec<MultiLegOrder> {
    let mut orders = Vec::new();
    let mut signals: HashMap<String, SignalState> = HashMap::new();

    // Phase 28: Per-symbol regime state
    let mut regime_states: HashMap<String, RegimeState> = HashMap::new();
    let mut regime_stats = RegimeGatingStats::default();

    // Minimum gap between signals for the same symbol (throttle)
    let min_signal_gap = chrono::Duration::milliseconds(config.min_hold_ms);

    // Log effective B1 routing config
    tracing::info!(
        "Gate B1 config: spread_limit_max={:.1}, spread_market_max={:.1}, vel_bps_sec_min={:.1}, strength_min={:.3}, dt_range=[{},{}]ms",
        config.spread_bps_limit_max,
        config.spread_bps_market_max,
        config.vel_bps_sec_market_min,
        config.signal_strength_market_min,
        config.vel_dt_min_ms,
        config.vel_dt_max_ms
    );

    // Phase 28: Log regime gating config
    if config.regime_gating_enabled {
        tracing::info!(
            "Phase 28 regime gating ENABLED: window={}, k={}, cusum_threshold={}, allowed={:?}",
            config.regime_window_size,
            config.regime_subspace_dim,
            config.regime_cusum_threshold,
            config.regime_allowed_labels
        );
    } else {
        tracing::info!("Phase 28 regime gating DISABLED (pass-through)");
    }

    // Phase 29: Log Ramanujan periodicity config
    if config.ramanujan_enabled {
        tracing::info!(
            "Phase 29 Ramanujan periodicity ENABLED: max_period={}, threshold={:.2}, block_on_hft={}",
            config.ramanujan_max_period,
            config.ramanujan_threshold,
            config.ramanujan_block_on_hft
        );
    }

    // Track previous quote per symbol for mid_return calculation
    let mut prev_quotes: HashMap<String, QuoteEvent> = HashMap::new();

    // Filter symbols if specified
    let trade_symbols: std::collections::HashSet<&str> = if config.symbols.is_empty() {
        quotes.iter().map(|q| q.tradingsymbol.as_str()).collect()
    } else {
        config.symbols.iter().map(|s| s.as_str()).collect()
    };

    for quote in quotes {
        if !trade_symbols.contains(quote.tradingsymbol.as_str()) {
            continue;
        }

        let symbol = &quote.tradingsymbol;
        let state = signals.entry(symbol.clone()).or_default();
        let lot_size = get_nse_lot_size(symbol);
        let quantity = lot_size * config.lots;

        // Phase 28/29: Update regime state for this symbol
        if config.regime_gating_enabled || config.ramanujan_enabled {
            let regime_state = regime_states
                .entry(symbol.clone())
                .or_insert_with(|| RegimeState::new(config));
            let prev_quote = prev_quotes.get(symbol);
            regime_state.update(quote, prev_quote);
            regime_stats.quotes_processed += 1;
        }

        // Track previous quote for next iteration
        prev_quotes.insert(symbol.clone(), quote.clone());

        let mid = mid_f64(quote);

        // Throttle: skip if too soon after last signal for this symbol
        // NOTE: Do NOT update state here - only on actionable quotes
        if let Some(last_ts) = state.last_signal_ts
            && quote.ts - last_ts < min_signal_gap
        {
            continue;
        }

        let spread = spread_bps(quote);
        let pressure = pressure_ratio(quote);

        // Skip if spread too wide for any order type
        // NOTE: Do NOT update state - this is not an actionable quote
        if spread > config.spread_bps_market_max {
            if let Some(ref mut w) = routing_writer.as_deref_mut() {
                w.inc_skipped_spread();
            }
            continue;
        }

        let bid = quote.bid_f64();
        let ask = quote.ask_f64();

        // Book sanity: skip if invalid
        // NOTE: Do NOT update state - this is not an actionable quote
        if !is_valid_maker_book(bid, ask) {
            if let Some(ref mut w) = routing_writer.as_deref_mut() {
                w.inc_skipped_book();
            }
            continue;
        }

        // Gate B1: Compute velocity (bps/sec) from previous actionable quote
        let vel_bps_sec = match (state.last_mid, state.last_quote_ts) {
            (Some(prev_mid), Some(prev_ts)) => mid_vel_bps_sec(
                prev_mid,
                prev_ts,
                mid,
                quote.ts,
                config.vel_dt_min_ms,
                config.vel_dt_max_ms,
            ),
            _ => 0.0,
        };

        // Compute dt_ms for thresholds snapshot
        let dt_ms = state
            .last_quote_ts
            .map(|prev_ts| (quote.ts - prev_ts).num_milliseconds())
            .unwrap_or(0);

        // Update state ONLY on actionable quotes (passed spread/book checks)
        state.last_mid = Some(mid);
        state.last_quote_ts = Some(quote.ts);

        let should_long = pressure >= config.pressure_long;
        let should_short = 1.0 / pressure >= config.pressure_short;

        if !should_long && !should_short {
            continue; // No signal
        }

        // Phase 28/29: Regime + Ramanujan gating check
        if config.regime_gating_enabled || config.ramanujan_enabled {
            if let Some(regime_state) = regime_states.get(symbol) {
                let block_on_hft = config.ramanujan_enabled && config.ramanujan_block_on_hft;
                if !regime_state.allows_trading(&config.regime_allowed_labels, block_on_hft) {
                    // Determine block reason for stats
                    if let Some(ref features) = regime_state.periodicity_features {
                        if features.hft_likely() && block_on_hft {
                            regime_stats.hft_blocked += 1;
                            tracing::trace!(
                                symbol = %symbol,
                                periodicity = %regime_state.periodicity_summary(),
                                "Signal blocked by Ramanujan HFT detection"
                            );
                            continue;
                        }
                    }
                    regime_stats.regime_blocked += 1;
                    tracing::trace!(
                        symbol = %symbol,
                        regime = ?regime_state.current_regime,
                        periodicity = %regime_state.periodicity_summary(),
                        "Signal blocked by regime gating"
                    );
                    continue; // Skip this signal due to unfavorable regime
                }
                regime_stats.regime_allowed += 1;
            }
        }

        // Gate B1: Compute signal strength (distance beyond threshold; >= 0)
        let long_strength = (pressure - config.pressure_long).max(0.0);
        let short_strength = ((1.0 / pressure) - config.pressure_short).max(0.0);

        // Gate B1: Routing decision with reason tracking
        let fast_move = vel_bps_sec.abs() >= config.vel_bps_sec_market_min;
        let strong_signal_long = long_strength >= config.signal_strength_market_min;
        let strong_signal_short = short_strength >= config.signal_strength_market_min;

        let choose_market_long = should_long
            && spread <= config.spread_bps_market_max
            && (fast_move || strong_signal_long);

        let choose_market_short = should_short
            && spread <= config.spread_bps_market_max
            && (fast_move || strong_signal_short);

        // Gate B0 + B1: Entry signals with routing
        if should_long {
            let (order_type, px_opt, order_type_str) = if choose_market_long {
                if let Some(ref mut w) = routing_writer.as_deref_mut() {
                    w.inc_market(fast_move, strong_signal_long);
                }
                (LegOrderType::Market, None, "Market")
            } else if spread <= config.spread_bps_limit_max {
                if let Some(ref mut w) = routing_writer.as_deref_mut() {
                    w.inc_limit();
                }
                let px = maker_limit_price(LegSide::Buy, bid, ask);
                (LegOrderType::Limit, Some(px), "Limit")
            } else {
                // Spread between limit_max and market_max: skip
                if let Some(ref mut w) = routing_writer.as_deref_mut() {
                    w.inc_skipped_spread();
                }
                continue;
            };

            let order_id = format!("micro_mm_long_{}:{}", symbol, quote.ts.to_rfc3339());
            let ts_utc = quote.ts.to_rfc3339();

            // B1.2: Always compute intent_id for join (even without routing_writer)
            let intent_id = RoutingDecisionRecord::compute_decision_id(
                &ts_utc,
                symbol,
                "Buy",
                bid,
                ask,
                pressure,
                vel_bps_sec,
                config.spread_bps_limit_max,
                config.spread_bps_market_max,
                config.vel_bps_sec_market_min,
                config.signal_strength_market_min,
                order_type_str,
                px_opt,
            );

            // Write routing decision record
            if let Some(ref mut w) = routing_writer.as_deref_mut() {
                let mut flags = Vec::new();
                let primary = if choose_market_long {
                    if fast_move {
                        flags.push("FAST_MOVE".to_string());
                    }
                    if strong_signal_long {
                        flags.push("STRONG_SIGNAL".to_string());
                    }
                    if fast_move {
                        "FAST_MOVE"
                    } else {
                        "STRONG_SIGNAL"
                    }
                } else {
                    flags.push("SPREAD_OK".to_string());
                    "SPREAD_OK"
                };

                // Compute vel_dt_in_bounds: true when dt is within [min, max] and there was a prior quote
                let vel_dt_in_bounds =
                    dt_ms > 0 && dt_ms >= config.vel_dt_min_ms && dt_ms <= config.vel_dt_max_ms;

                let record = RoutingDecisionRecord {
                    record_type: "decision",
                    schema: "quantlaxmi.routing_decisions.v1",
                    schema_rev: 2, // B1.3 final
                    ts_utc,
                    symbol: symbol.clone(),
                    exchange: "NFO".to_string(),
                    quote: QuoteSnapshot {
                        bid,
                        ask,
                        bid_qty: quote.bid_qty as u64,
                        ask_qty: quote.ask_qty as u64,
                        mid,
                    },
                    features: RoutingFeatures {
                        spread_bps: spread,
                        pressure,
                        vel_bps_sec,
                        vel_abs_bps_sec: vel_bps_sec.abs(),
                        vel_used: vel_dt_in_bounds,
                        dt_ms,
                        signal_strength: long_strength,
                    },
                    thresholds: ThresholdsSnapshot {
                        spread_bps_limit_max: config.spread_bps_limit_max,
                        spread_bps_market_max: config.spread_bps_market_max,
                        vel_bps_sec_market_min: config.vel_bps_sec_market_min,
                        signal_strength_market_min: config.signal_strength_market_min,
                        vel_dt_min_ms: config.vel_dt_min_ms,
                        vel_dt_max_ms: config.vel_dt_max_ms,
                        vel_dt_in_bounds,
                    },
                    signal: SignalSnapshot {
                        side: "Buy".to_string(),
                        direction: 1,
                        should_trade: true,
                        fast_move,
                        strong_signal: strong_signal_long,
                    },
                    decision: DecisionOutput {
                        order_type: order_type_str.to_string(),
                        price: px_opt,
                    },
                    reason: DecisionReason {
                        primary: primary.to_string(),
                        flags,
                    },
                    ids: DecisionIds {
                        decision_id: intent_id.clone(),
                        intent_id: intent_id.clone(), // Same as decision_id (includes order_type+price)
                        order_id: order_id.clone(),
                        leg_index: 0,
                    },
                };

                if let Err(e) = w.write_decision(&record) {
                    tracing::warn!("Failed to write routing decision: {}", e);
                }
            }

            let entry_order = MultiLegOrder {
                strategy_name: format!("micro_mm_long_{}", symbol),
                legs: vec![LegOrder {
                    tradingsymbol: symbol.clone(),
                    exchange: "NFO".to_string(),
                    side: LegSide::Buy,
                    quantity,
                    order_type,
                    price: px_opt,
                    intent_id: Some(intent_id), // B1.2: Propagate for routing→fill join
                }],
                total_margin_required: 0.0,
            };

            orders.push(entry_order);
            state.last_signal_ts = Some(quote.ts);
        } else if should_short {
            let (order_type, px_opt, order_type_str) = if choose_market_short {
                if let Some(ref mut w) = routing_writer.as_deref_mut() {
                    w.inc_market(fast_move, strong_signal_short);
                }
                (LegOrderType::Market, None, "Market")
            } else if spread <= config.spread_bps_limit_max {
                if let Some(ref mut w) = routing_writer.as_deref_mut() {
                    w.inc_limit();
                }
                let px = maker_limit_price(LegSide::Sell, bid, ask);
                (LegOrderType::Limit, Some(px), "Limit")
            } else {
                // Spread between limit_max and market_max: skip
                if let Some(ref mut w) = routing_writer.as_deref_mut() {
                    w.inc_skipped_spread();
                }
                continue;
            };

            let order_id = format!("micro_mm_short_{}:{}", symbol, quote.ts.to_rfc3339());
            let ts_utc = quote.ts.to_rfc3339();

            // B1.2: Always compute intent_id for join (even without routing_writer)
            let intent_id = RoutingDecisionRecord::compute_decision_id(
                &ts_utc,
                symbol,
                "Sell",
                bid,
                ask,
                pressure,
                vel_bps_sec,
                config.spread_bps_limit_max,
                config.spread_bps_market_max,
                config.vel_bps_sec_market_min,
                config.signal_strength_market_min,
                order_type_str,
                px_opt,
            );

            // Write routing decision record
            if let Some(ref mut w) = routing_writer.as_deref_mut() {
                let mut flags = Vec::new();
                let primary = if choose_market_short {
                    if fast_move {
                        flags.push("FAST_MOVE".to_string());
                    }
                    if strong_signal_short {
                        flags.push("STRONG_SIGNAL".to_string());
                    }
                    if fast_move {
                        "FAST_MOVE"
                    } else {
                        "STRONG_SIGNAL"
                    }
                } else {
                    flags.push("SPREAD_OK".to_string());
                    "SPREAD_OK"
                };

                // Compute vel_dt_in_bounds: true when dt is within [min, max] and there was a prior quote
                let vel_dt_in_bounds =
                    dt_ms > 0 && dt_ms >= config.vel_dt_min_ms && dt_ms <= config.vel_dt_max_ms;

                let record = RoutingDecisionRecord {
                    record_type: "decision",
                    schema: "quantlaxmi.routing_decisions.v1",
                    schema_rev: 2, // B1.3 final
                    ts_utc,
                    symbol: symbol.clone(),
                    exchange: "NFO".to_string(),
                    quote: QuoteSnapshot {
                        bid,
                        ask,
                        bid_qty: quote.bid_qty as u64,
                        ask_qty: quote.ask_qty as u64,
                        mid,
                    },
                    features: RoutingFeatures {
                        spread_bps: spread,
                        pressure,
                        vel_bps_sec,
                        vel_abs_bps_sec: vel_bps_sec.abs(),
                        vel_used: vel_dt_in_bounds,
                        dt_ms,
                        signal_strength: short_strength,
                    },
                    thresholds: ThresholdsSnapshot {
                        spread_bps_limit_max: config.spread_bps_limit_max,
                        spread_bps_market_max: config.spread_bps_market_max,
                        vel_bps_sec_market_min: config.vel_bps_sec_market_min,
                        signal_strength_market_min: config.signal_strength_market_min,
                        vel_dt_min_ms: config.vel_dt_min_ms,
                        vel_dt_max_ms: config.vel_dt_max_ms,
                        vel_dt_in_bounds,
                    },
                    signal: SignalSnapshot {
                        side: "Sell".to_string(),
                        direction: -1,
                        should_trade: true,
                        fast_move,
                        strong_signal: strong_signal_short,
                    },
                    decision: DecisionOutput {
                        order_type: order_type_str.to_string(),
                        price: px_opt,
                    },
                    reason: DecisionReason {
                        primary: primary.to_string(),
                        flags,
                    },
                    ids: DecisionIds {
                        decision_id: intent_id.clone(),
                        intent_id: intent_id.clone(), // Same as decision_id (includes order_type+price)
                        order_id: order_id.clone(),
                        leg_index: 0,
                    },
                };

                if let Err(e) = w.write_decision(&record) {
                    tracing::warn!("Failed to write routing decision: {}", e);
                }
            }

            let entry_order = MultiLegOrder {
                strategy_name: format!("micro_mm_short_{}", symbol),
                legs: vec![LegOrder {
                    tradingsymbol: symbol.clone(),
                    exchange: "NFO".to_string(),
                    side: LegSide::Sell,
                    quantity,
                    order_type,
                    price: px_opt,
                    intent_id: Some(intent_id), // B1.2: Propagate for routing→fill join
                }],
                total_margin_required: 0.0,
            };

            orders.push(entry_order);
            state.last_signal_ts = Some(quote.ts);
        }
    }

    // Phase 28/29: Log regime gating statistics
    if config.regime_gating_enabled || config.ramanujan_enabled {
        tracing::info!(
            "Phase 28/29 gating stats: quotes={}, regime_blocked={}, hft_blocked={}, allowed={}",
            regime_stats.quotes_processed,
            regime_stats.regime_blocked,
            regime_stats.hft_blocked,
            regime_stats.regime_allowed
        );

        // Log per-symbol regime state summary
        for (symbol, regime_state) in &regime_states {
            tracing::debug!(
                symbol = %symbol,
                regime = ?regime_state.current_regime,
                is_ready = regime_state.is_ready,
                shifts = regime_state.shift_count,
                periodicity = %regime_state.periodicity_summary(),
                "Final regime state"
            );
        }
    }

    orders
}

/// Main entry point for order generation CLI
pub async fn run_generate_orders(
    strategy: &str,
    replay_path: &str,
    out_path: &str,
    config_path: Option<&str>,
    routing_log_path: Option<&str>,
    _seed: u64, // Reserved for future RNG strategies
) -> Result<()> {
    tracing::info!("Generating orders with strategy: {}", strategy);
    tracing::info!("Replay: {}", replay_path);
    tracing::info!("Output: {}", out_path);
    if let Some(rl) = routing_log_path {
        tracing::info!("Routing log: {}", rl);
    }

    // Load replay quotes
    let quotes = load_quotes(Path::new(replay_path))?;
    tracing::info!("Loaded {} quotes from replay", quotes.len());

    // Generate orders based on strategy
    let (strategy_name, orders) = match strategy {
        "india_micro_mm" => {
            let config = if let Some(cfg_path) = config_path {
                let cfg_str = std::fs::read_to_string(cfg_path)
                    .with_context(|| format!("read config: {}", cfg_path))?;
                toml::from_str(&cfg_str).with_context(|| "parse MicroMmConfig TOML")?
            } else {
                MicroMmConfig::default()
            };

            tracing::info!(
                "MicroMM config: max_spread_bps={}, hold_range={}..{}ms, pressure_long={}, pressure_short={}",
                config.max_spread_bps,
                config.min_hold_ms,
                config.max_hold_ms,
                config.pressure_long,
                config.pressure_short
            );

            // Gate B1.3: Create routing decisions writer if path specified
            let mut routing_writer = if let Some(rl_path) = routing_log_path {
                // Generate deterministic run_id from output path
                let run_id = Path::new(out_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");

                let header = RoutingRunHeader::new(strategy, run_id, replay_path, &config);
                Some(RoutingDecisionsWriter::new(Path::new(rl_path), &header)?)
            } else {
                None
            };

            let orders = generate_micro_mm_orders(&quotes, &config, routing_writer.as_mut());

            // Finish routing log with footer
            if let Some(writer) = routing_writer {
                let counts = writer.finish()?;
                tracing::info!(
                    "Gate B1.3 routing log: decisions={}, LIMIT={}, MARKET={} (vel={}, strength={}, both={}), skipped: spread={}, book={}",
                    counts.decisions,
                    counts.limit,
                    counts.market,
                    counts.market_by_vel,
                    counts.market_by_strength,
                    counts.market_by_both,
                    counts.skipped_spread_band,
                    counts.skipped_book_invalid
                );
            }

            ("india_micro_mm".to_string(), orders)
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unknown strategy: {}. Available: india_micro_mm",
                strategy
            ));
        }
    };

    tracing::info!("Generated {} orders", orders.len());

    // Write output
    let order_file = OrderFile {
        strategy_name,
        orders,
    };

    let json = serde_json::to_string_pretty(&order_file)?;
    std::fs::write(out_path, &json)?;

    tracing::info!("Orders written to: {}", out_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MicroMmConfig::default();
        assert_eq!(config.max_spread_bps, 30.0);
        assert_eq!(config.min_hold_ms, 30_000);
        assert_eq!(config.max_hold_ms, 90_000);
        assert_eq!(config.lots, 1);
    }

    #[test]
    fn test_spread_bps_calculation() {
        let quote = QuoteEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "TEST".to_string(),
            bid: 10000, // 100.00
            ask: 10010, // 100.10
            bid_qty: 100,
            ask_qty: 100,
            price_exponent: -2,
        };

        let spread = spread_bps(&quote);
        // Spread = 0.10, mid = 100.05, bps = (0.10/100.05)*10000 ≈ 9.995
        assert!((spread - 9.995).abs() < 0.1);
    }

    #[test]
    fn test_pressure_ratio() {
        let quote = QuoteEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "TEST".to_string(),
            bid: 10000,
            ask: 10010,
            bid_qty: 150,
            ask_qty: 100,
            price_exponent: -2,
        };

        let pressure = pressure_ratio(&quote);
        assert!((pressure - 1.5).abs() < 0.001);
    }
}
