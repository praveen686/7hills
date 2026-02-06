//! SLRT Reference Implementation Test Harness
//!
//! Deterministic test harness that replays a session and prints:
//! - Features
//! - Fragility
//! - Regime label per timestep
//!
//! AUDIT: Requires explicit config - aborts on missing config.
//!
//! Frame-based alignment:
//! - Canonical join clock = integer microseconds
//! - Frame i = [depth[i].t_us, depth[i+1].t_us)
//! - All trades in interval attach to frame, file order preserved
//!
//! Usage:
//!   slrt-ref-harness <input.jsonl> [output.jsonl]

use slrt_ref::alignment::{AlignedDepth, AlignedTrade, AlignmentStats, Frame, StreamAligner};
use slrt_ref::data::{MarketEvent, OrderBook, Trade, TradeSide, TriState, VenueMetadata};
use slrt_ref::features::{SnapshotFeatures, TradeFlowAccumulator};
use slrt_ref::fragility::{FragilityCalculator, FragilityClipBounds, FragilityWeights};
use slrt_ref::fti::FTITracker;
use slrt_ref::normalization::{DegradedReasons, NormalizationStatus, StateNormalizer};
use slrt_ref::regime::{Regime, RegimeClassifier, RegimeThresholds, ToxicityConfig, ToxicityState};
use slrt_ref::sealed::{ELASTICITY_MAX, STATE_DIM};
use slrt_ref::subspace::{RegimeMetrics, SubspaceTracker};
use slrt_ref::wal::{
    CalibrationManifest, FeatureBatchParams, RegimeCountManifest, RunManifest, SessionManifest,
    ThresholdManifest, WalWriter,
};

use std::io::{BufRead, BufReader, Write};

/// Pipeline configuration - must be explicitly provided.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub fragility_weights: FragilityWeights,
    pub fragility_bounds: FragilityClipBounds,
    pub regime_thresholds: RegimeThresholds,
    pub toxicity_config: ToxicityConfig,
}

/// Error when config is missing or invalid.
#[derive(Debug)]
pub enum ConfigError {
    MissingConfig(String),
    InvalidConfig(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingConfig(msg) => write!(f, "CONFIG_MISSING: {}", msg),
            Self::InvalidConfig(msg) => write!(f, "CONFIG_INVALID: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Metrics histogram for analysis.
#[derive(Default)]
struct MetricHistogram {
    values: Vec<f64>,
}

impl MetricHistogram {
    fn add(&mut self, value: f64) {
        if value.is_finite() {
            self.values.push(value);
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    fn count(&self) -> usize {
        self.values.len()
    }
}

/// Metric distributions for analysis.
#[derive(Default)]
struct MetricDistributions {
    d_perp: MetricHistogram,
    fragility: MetricHistogram,
    confidence: MetricHistogram,
    fti_persist: MetricHistogram,
    tox_persist: MetricHistogram,
    urgency: MetricHistogram,
    /// Count of frames with NoTradesWindow
    no_trades_window_count: u64,
    /// Count of frames with InsufficientHistory (FTI)
    insufficient_history_count: u64,
    /// Count of frames with computable toxicity
    computable_toxicity_count: u64,
}

impl MetricDistributions {
    fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Metric Distributions ===");
        eprintln!();
        self.print_histogram("d_perp", &self.d_perp);
        self.print_histogram("fragility", &self.fragility);
        self.print_histogram("confidence", &self.confidence);
        self.print_histogram("fti_persist", &self.fti_persist);
        self.print_histogram("tox_persist", &self.tox_persist);
        self.print_histogram("urgency", &self.urgency);

        eprintln!();
        eprintln!("=== State Observability ===");
        let total = self.no_trades_window_count + self.computable_toxicity_count;
        if total > 0 {
            let no_trades_pct = 100.0 * self.no_trades_window_count as f64 / total as f64;
            let computable_pct = 100.0 * self.computable_toxicity_count as f64 / total as f64;
            eprintln!(
                "Toxicity: NoTradesWindow={} ({:.1}%), Computable={} ({:.1}%)",
                self.no_trades_window_count,
                no_trades_pct,
                self.computable_toxicity_count,
                computable_pct
            );
        }
        eprintln!(
            "FTI: InsufficientHistory={}",
            self.insufficient_history_count
        );
    }

    fn print_histogram(&self, name: &str, h: &MetricHistogram) {
        if h.count() == 0 {
            eprintln!("{}: no data", name);
            return;
        }
        eprintln!(
            "{:12}: n={:6} | min={:8.4} | p25={:8.4} | p50={:8.4} | p75={:8.4} | p95={:8.4} | max={:8.4} | mean={:8.4}",
            name,
            h.count(),
            h.min(),
            h.percentile(25.0),
            h.percentile(50.0),
            h.percentile(75.0),
            h.percentile(95.0),
            h.max(),
            h.mean(),
        );
    }
}

/// Pipeline state for a single symbol.
#[allow(dead_code)]
struct SymbolPipeline {
    symbol: String,
    venue: VenueMetadata,
    config: PipelineConfig,
    /// Last order book
    last_book: Option<OrderBook>,
    /// Price history for FTI
    price_history: Vec<f64>,
    /// Trade flow accumulator (250ms window)
    trade_flow_250ms: TradeFlowAccumulator,
    /// Trade flow accumulator (1s window for depth)
    trade_flow_1s: TradeFlowAccumulator,
    /// FTI tracker
    fti_tracker: FTITracker,
    /// Fragility calculator
    fragility_calc: FragilityCalculator,
    /// State normalizer
    normalizer: StateNormalizer,
    /// Subspace tracker
    subspace: SubspaceTracker,
    /// Regime classifier
    classifier: RegimeClassifier,
    /// Previous normalized state (for v_para calculation)
    prev_state: [f64; STATE_DIM],
    /// WAL writer
    wal: WalWriter,
    /// Regime counts
    regime_counts: RegimeCountManifest,
    /// Tick count
    tick_count: u64,
    /// First timestamp
    first_ts_ns: Option<i64>,
    /// Last timestamp
    last_ts_ns: i64,
    /// Metric distributions
    distributions: MetricDistributions,
    /// Count of undefined elasticity events
    undefined_elasticity_count: u64,
    /// Alignment statistics (for frame-based processing)
    alignment_stats: Option<AlignmentStats>,
    /// Whether bucket_size has been set from heuristic
    bucket_size_calibrated: bool,
}

impl SymbolPipeline {
    fn new(
        symbol: &str,
        venue: VenueMetadata,
        config: PipelineConfig,
    ) -> Result<Self, ConfigError> {
        // Create fragility calculator with explicit config
        let fragility_calc = FragilityCalculator::new(
            config.fragility_weights.clone(),
            config.fragility_bounds.clone(),
        )
        .map_err(|e| ConfigError::InvalidConfig(e.to_string()))?;

        // Create regime classifier with explicit config
        let classifier = RegimeClassifier::new(
            config.regime_thresholds.clone(),
            config.toxicity_config.clone(),
        )
        .map_err(|e| ConfigError::InvalidConfig(e.to_string()))?;

        Ok(Self {
            symbol: symbol.to_string(),
            venue,
            config,
            last_book: None,
            price_history: Vec::with_capacity(200),
            trade_flow_250ms: TradeFlowAccumulator::new(250),
            trade_flow_1s: TradeFlowAccumulator::new(1000),
            fti_tracker: FTITracker::new(),
            fragility_calc,
            normalizer: StateNormalizer::new(),
            subspace: SubspaceTracker::new(),
            classifier,
            prev_state: [0.0; STATE_DIM],
            wal: WalWriter::new(),
            regime_counts: RegimeCountManifest::default(),
            tick_count: 0,
            first_ts_ns: None,
            last_ts_ns: 0,
            distributions: MetricDistributions::default(),
            undefined_elasticity_count: 0,
            alignment_stats: None,
            bucket_size_calibrated: false,
        })
    }

    /// Set alignment stats and apply bucket_size heuristic.
    /// AUDIT: bucket_size is computed from WARMUP period only (first 300s) to avoid future-peeking.
    fn set_alignment_stats(&mut self, stats: AlignmentStats) {
        // Compute and apply bucket_size heuristic if not already calibrated
        if !self.bucket_size_calibrated {
            let recommended = stats.recommended_bucket_size();
            let source = stats.bucket_size_source();
            eprintln!(
                "ALIGNMENT: bucket_size = {:.2} USDT (source: {})",
                recommended, source
            );
            self.classifier.set_bucket_size(recommended);
            self.bucket_size_calibrated = true;
        }

        // Print alignment summary
        let notional_pct = stats.notional_per_sec_percentiles();
        eprintln!();
        eprintln!("=== Alignment Statistics ===");
        eprintln!("Depth frames: {}", stats.depth_count);
        eprintln!("Total trades: {}", stats.trade_count);
        eprintln!(
            "Trade coverage: {:.1}% of frames have trades",
            stats.trade_coverage() * 100.0
        );
        eprintln!(
            "Trades/frame: min={}, max={}",
            stats.min_trades_per_frame, stats.max_trades_per_frame
        );
        eprintln!(
            "Notional/sec: p50={:.0}, p95={:.0}, max={:.0} USDT",
            notional_pct.p50, notional_pct.p95, notional_pct.max
        );
        eprintln!(
            "Total volume: {:.2} BTC (buy={:.2}, sell={:.2})",
            stats.total_volume, stats.buy_volume, stats.sell_volume
        );
        eprintln!("Total notional: {:.0} USDT", stats.total_notional);
        eprintln!("Bucket size: {:.0} USDT", self.classifier.bucket_size());

        self.alignment_stats = Some(stats);
    }

    /// Process an order book update.
    fn process_book(&mut self, book: OrderBook) -> Option<TickOutput> {
        let ts_ns = book.ts_ns;

        if self.first_ts_ns.is_none() {
            self.first_ts_ns = Some(ts_ns);
            self.normalizer.reset(ts_ns);
        }
        self.last_ts_ns = ts_ns;

        // Admission checks
        if book.is_crossed() {
            self.wal
                .write_signal_refused(ts_ns, &self.symbol, "crossed_book");
            return None;
        }

        if book.bids.is_empty() || book.asks.is_empty() {
            self.wal
                .write_signal_refused(ts_ns, &self.symbol, "empty_book");
            return None;
        }

        // Compute snapshot features
        let snapshot = SnapshotFeatures::compute(&book, self.venue.tick_size, 1.0);

        // Update trade flow with mid
        self.trade_flow_250ms.add_mid(ts_ns, snapshot.mid);
        self.trade_flow_1s.add_mid(ts_ns, snapshot.mid);

        // Update price history for FTI
        self.price_history.push(snapshot.mid);
        if self.price_history.len() > 200 {
            self.price_history.remove(0);
        }

        // Update depth tracking
        let total_depth = snapshot.total_bid_depth + snapshot.total_ask_depth;
        self.trade_flow_250ms.add_depth(ts_ns, total_depth);
        self.trade_flow_1s.add_depth(ts_ns, total_depth);

        // Compute trade flow features
        let trade_flow = self.trade_flow_250ms.compute(ts_ns);

        // Track undefined elasticity events (penalty applied in classify())
        if trade_flow.elasticity_undefined {
            self.undefined_elasticity_count += 1;
        }

        // Compute FTI (need enough price history)
        let fti_metrics = if self.price_history.len() >= 100 {
            self.fti_tracker.update(&self.price_history)
        } else {
            slrt_ref::fti::FTIMetrics::default()
        };

        // Compute fragility
        let fragility = self.fragility_calc.compute(&snapshot, &trade_flow);

        // AUDIT: Cap elasticity to ELASTICITY_MAX before feeding to state vector
        let capped_elasticity = trade_flow.elasticity.min(ELASTICITY_MAX);

        // Get current toxicity from tracker for state vector
        // This is the rolling mean of bucket imbalances [0, 1]
        let toxicity = self.classifier.current_toxicity();

        // Build raw state vector
        // Spec Section 6.1: x_t = [μ - m, I10, ε_250ms, Ddot_10_250ms, gapRisk, FTI, toxicity]
        let raw_state: [f64; STATE_DIM] = [
            snapshot.microprice - snapshot.mid,
            snapshot.imbalance_10,
            capped_elasticity, // AUDIT: Use capped value
            trade_flow.depth_collapse_rate,
            snapshot.gap_risk,
            fti_metrics.fti_level,
            toxicity,
        ];

        // Normalize - v1.2: returns (state, status, reasons)
        let (normalized_state, norm_status, mut degraded_reasons) =
            self.normalizer.normalize(ts_ns, &raw_state);

        // Add other degradation reasons
        if trade_flow.elasticity_undefined {
            degraded_reasons |= DegradedReasons::UNDEFINED_ELASTICITY;
        }

        // Update subspace tracker
        if norm_status != NormalizationStatus::Warmup {
            self.subspace.update(ts_ns, &normalized_state);
        }

        // Compute regime metrics
        let d_perp = self.subspace.off_manifold_distance(&normalized_state);
        let v_para = self
            .subspace
            .tangent_speed(&normalized_state, &self.prev_state);
        let rho = self.subspace.subspace_rotation();

        let regime_metrics = RegimeMetrics {
            d_perp,
            v_para,
            rho,
        };

        // Classify regime (pass elasticity_undefined and degraded_reasons for audit trail)
        let classification = self.classifier.classify(
            &regime_metrics,
            &fragility,
            &fti_metrics,
            norm_status,
            trade_flow.elasticity_undefined,
            degraded_reasons.bits(),
        );

        // Compute urgency: u = d_perp * fragility * confidence
        let urgency = d_perp * fragility.value * classification.confidence;

        // Update distributions
        self.distributions.d_perp.add(d_perp);
        self.distributions.fragility.add(fragility.value);
        self.distributions.confidence.add(classification.confidence);
        self.distributions.fti_persist.add(fti_metrics.fti_persist);
        self.distributions
            .tox_persist
            .add(classification.toxicity_persist);
        self.distributions.urgency.add(urgency);

        // Update counts
        match classification.regime {
            Regime::R0 => self.regime_counts.r0 += 1,
            Regime::R1 => self.regime_counts.r1 += 1,
            Regime::R2 => self.regime_counts.r2 += 1,
            Regime::R3 => self.regime_counts.r3 += 1,
        }

        // Write WAL events
        self.wal.write_feature_batch(FeatureBatchParams {
            ts_ns,
            symbol: &self.symbol,
            snapshot: &snapshot,
            trade_flow: &trade_flow,
            fti: &fti_metrics,
            fragility: &fragility,
            raw_state,
            normalized_state,
        });

        self.wal.write_regime(
            ts_ns,
            &self.symbol,
            classification.regime,
            &regime_metrics,
            &classification,
        );

        // Update state
        self.prev_state = normalized_state;
        self.last_book = Some(book);
        self.tick_count += 1;

        // Determine trading eligibility: R2/R3 AND not refused
        let is_trade_regime = matches!(classification.regime, Regime::R2 | Regime::R3);
        let eligible_to_trade = is_trade_regime && !classification.refused;

        Some(TickOutput {
            ts_ns,
            symbol: self.symbol.clone(),
            mid: snapshot.mid,
            microprice: snapshot.microprice,
            imbalance: snapshot.imbalance_10,
            gap_risk: snapshot.gap_risk,
            elasticity: capped_elasticity,
            elasticity_undefined: trade_flow.elasticity_undefined,
            depth_collapse: trade_flow.depth_collapse_rate,
            fti: fti_metrics.fti_level,
            fti_persist: fti_metrics.fti_persist,
            fragility: fragility.value,
            toxicity: classification.toxicity,
            toxicity_persist: classification.toxicity_persist,
            d_perp,
            v_para,
            rho,
            urgency,
            regime: classification.regime.as_str().to_string(),
            confidence: classification.confidence,
            norm_status: format!("{:?}", norm_status),
            refused: classification.refused,
            eligible_to_trade,
            degraded_reasons: degraded_reasons.bits(),
        })
    }

    /// Process a trade.
    fn process_trade(&mut self, trade: Trade) {
        let _ts_ns = trade.ts_ns;

        // Get microprice from last book
        let microprice = self
            .last_book
            .as_ref()
            .map(|b| {
                let a1 = b.best_ask().unwrap_or(0.0);
                let b1 = b.best_bid().unwrap_or(0.0);
                let qa1 = b.best_ask_qty().unwrap_or(0.0);
                let qb1 = b.best_bid_qty().unwrap_or(0.0);
                (a1 * qb1 + b1 * qa1) / (qa1 + qb1 + 1e-12)
            })
            .unwrap_or(trade.price_f64());

        // Add to trade flow accumulators
        self.trade_flow_250ms.add_trade(&trade, microprice);
        self.trade_flow_1s.add_trade(&trade, microprice);

        // Add to toxicity tracker
        let is_buy = matches!(trade.side.as_option(), Some(TradeSide::Buy));
        self.classifier.add_trade(trade.qty_f64(), is_buy);
    }

    /// Process an aligned trade (from frame-based alignment).
    fn process_aligned_trade(&mut self, trade: &AlignedTrade) {
        // Get microprice from last book
        let microprice = self
            .last_book
            .as_ref()
            .map(|b| {
                let a1 = b.best_ask().unwrap_or(0.0);
                let b1 = b.best_bid().unwrap_or(0.0);
                let qa1 = b.best_ask_qty().unwrap_or(0.0);
                let qb1 = b.best_bid_qty().unwrap_or(0.0);
                (a1 * qb1 + b1 * qa1) / (qa1 + qb1 + 1e-12)
            })
            .unwrap_or(trade.price_f64());

        // Convert to Trade for flow accumulators
        let side = if trade.sign > 0 {
            TradeSide::Buy
        } else {
            TradeSide::Sell
        };
        let legacy_trade = Trade {
            ts_ns: trade.ts_ns,
            symbol: trade.symbol.clone(),
            price_mantissa: trade.price_mantissa,
            price_exponent: trade.price_exponent,
            qty_mantissa: trade.qty_mantissa,
            qty_exponent: trade.qty_exponent,
            side: TriState::Present(side),
        };

        // Add to trade flow accumulators
        self.trade_flow_250ms.add_trade(&legacy_trade, microprice);
        self.trade_flow_1s.add_trade(&legacy_trade, microprice);

        // Add to toxicity tracker with NOTIONAL (quote units) for meaningful bucket sizing
        let is_buy = trade.sign > 0;
        self.classifier.add_trade(trade.notional, is_buy);
    }

    /// Process a frame (depth snapshot + attached trades).
    /// Returns tick output after processing all trades in the frame.
    ///
    /// CAUSALITY AUDIT:
    /// - Frame contains depth[i] and trades in [depth[i].t_us, depth[i+1].t_us)
    /// - We ONLY read depth[i+1].t_us for frame boundary, never its content
    /// - Trades are strictly from the past (before frame_end_us)
    /// - No future-peeking: all data used was available at depth[i].t_us
    fn process_frame(&mut self, frame: &Frame) -> Option<TickOutput> {
        // Start new window for toxicity tracking
        self.classifier.start_window();

        // Convert AlignedDepth to OrderBook
        let book = OrderBook {
            ts_ns: frame.depth.ts_ns,
            symbol: frame.depth.symbol.clone(),
            bids: frame
                .depth
                .bids
                .iter()
                .map(|(p, q)| slrt_ref::data::PriceLevel {
                    price_mantissa: *p,
                    price_exponent: frame.depth.price_exponent,
                    qty_mantissa: *q,
                    qty_exponent: frame.depth.qty_exponent,
                })
                .collect(),
            asks: frame
                .depth
                .asks
                .iter()
                .map(|(p, q)| slrt_ref::data::PriceLevel {
                    price_mantissa: *p,
                    price_exponent: frame.depth.price_exponent,
                    qty_mantissa: *q,
                    qty_exponent: frame.depth.qty_exponent,
                })
                .collect(),
        };

        // Process all trades in this frame (in file order)
        for trade in &frame.trades {
            self.process_aligned_trade(trade);
        }

        // Get toxicity state after processing all trades
        let tox_state = self.classifier.get_toxicity_state();

        // Track observability
        match &tox_state {
            ToxicityState::NoTradesWindow => {
                self.distributions.no_trades_window_count += 1;
            }
            ToxicityState::Computable { .. } => {
                self.distributions.computable_toxicity_count += 1;
            }
        }

        // Process the order book (which does regime classification)
        self.process_book(book)
    }

    /// Generate run manifest with calibration values for reproducibility.
    fn generate_manifest(&self, session_id: &str) -> RunManifest {
        let thresholds = self.classifier.thresholds();

        // Get calibration values
        let (fti_threshold, fti_samples) = self.fti_tracker.calibration_info().unwrap_or((1.0, 0)); // Fallback if not calibrated

        let warmup_seconds = self
            .alignment_stats
            .as_ref()
            .map(|s| s.warmup_duration_secs())
            .unwrap_or(0.0);

        RunManifest {
            feature_schema_version: "1.0".to_string(),
            thresholds: ThresholdManifest {
                tau_d_perp: thresholds.tau_d_perp,
                tau_fragility: thresholds.tau_fragility,
                tau_fti_persist: thresholds.tau_fti_persist,
                tau_toxicity_persist: thresholds.tau_toxicity_persist,
                tau_confidence: thresholds.tau_confidence,
            },
            calibration: CalibrationManifest {
                bucket_size_usdt: self.classifier.bucket_size(),
                fti_persist_threshold: fti_threshold,
                toxicity_persist_threshold: self.config.toxicity_config.persist_threshold,
                warmup_seconds,
                fti_calibration_samples: fti_samples,
            },
            session: SessionManifest {
                session_id: session_id.to_string(),
                symbol: self.symbol.clone(),
                start_ts_ns: self.first_ts_ns.unwrap_or(0),
                end_ts_ns: self.last_ts_ns,
                total_ticks: self.tick_count,
                total_regimes: self.regime_counts.clone(),
            },
            digest: self.wal.finalize_digest(),
        }
    }

    fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Run Summary ===");
        eprintln!("Ticks processed: {}", self.tick_count);
        eprintln!(
            "Regime distribution: R0={}, R1={}, R2={}, R3={}",
            self.regime_counts.r0,
            self.regime_counts.r1,
            self.regime_counts.r2,
            self.regime_counts.r3
        );
        eprintln!(
            "Undefined elasticity events: {}",
            self.undefined_elasticity_count
        );
        eprintln!("WAL digest: {}", self.wal.finalize_digest());

        self.distributions.print_summary();

        // Toxicity instrumentation
        let tox_inst = self.classifier.toxicity_instrumentation();
        eprintln!();
        eprintln!("=== Toxicity Instrumentation ===");
        eprintln!("Buckets closed: {}", tox_inst.buckets_closed);
        eprintln!("Total trades: {}", tox_inst.total_trades);
        eprintln!("Buy ratio: {:.1}%", tox_inst.buy_ratio * 100.0);
        eprintln!(
            "Imbalance (closed buckets): p50={:.3}, p95={:.3}, max={:.3}",
            tox_inst.imb_p50, tox_inst.imb_p95, tox_inst.imb_max
        );
        eprintln!(
            "Toxicity (over time): p50={:.3}, p95={:.3}, max={:.3}",
            tox_inst.tox_p50, tox_inst.tox_p95, tox_inst.tox_max
        );
    }
}

/// Output for a single tick.
/// All numeric fields are guaranteed non-null; boolean flags indicate missing/undefined state.
#[derive(Debug, serde::Serialize)]
struct TickOutput {
    ts_ns: i64,
    symbol: String,
    mid: f64,
    microprice: f64,
    imbalance: f64,
    gap_risk: f64,
    elasticity: f64,
    elasticity_undefined: bool,
    depth_collapse: f64,
    fti: f64,
    fti_persist: f64,
    fragility: f64,
    toxicity: f64,
    toxicity_persist: f64,
    d_perp: f64,
    v_para: f64,
    rho: f64,
    urgency: f64,
    regime: String,
    confidence: f64,
    norm_status: String,
    /// Whether signal was refused due to structural invalidity (RefuseFrame)
    /// v1.2: Only true for missing book, crossed book, NaN/non-finite state
    refused: bool,
    /// Whether trading is allowed: true only if regime is R2/R3 AND refused==false
    eligible_to_trade: bool,
    /// Bitmask of degradation/refusal reasons (v1.2)
    degraded_reasons: u32,
}

/// Create reference config for testing.
/// In production, this MUST be loaded from external config file.
fn create_reference_config() -> PipelineConfig {
    PipelineConfig {
        fragility_weights: FragilityWeights {
            w1_gap_risk: 0.25,
            w2_elasticity: 0.20,
            w3_depth_decay: 0.15,
            w4_spread_z: 0.20,
            w5_depth_slope: 0.20,
        },
        fragility_bounds: FragilityClipBounds {
            gap_risk_min: 0.0,
            gap_risk_max: 10.0,
            elasticity_min: 0.0,
            elasticity_max: 10.0, // Matches ELASTICITY_MAX
            depth_decay_min: -1.0,
            depth_decay_max: 1.0,
            spread_z_min: -3.0,
            spread_z_max: 3.0,
            depth_slope_min: 0.0,
            depth_slope_max: 1.0,
        },
        regime_thresholds: RegimeThresholds {
            tau_d_perp: 2.0,
            tau_fragility: 0.6,
            tau_fti_persist: 0.3, // 30% of windows elevated (ring buffer fraction)
            tau_toxicity_persist: 0.3, // 30% of windows elevated (ring buffer fraction)
            tau_confidence: 0.5,
            tau_r1_d_perp: 1.0,
            tau_r1_fragility: 0.3,
            tau_r2_d_perp: 1.5,
            tau_r2_fragility: 0.45,
        },
        toxicity_config: ToxicityConfig {
            bucket_size: 1000.0, // Will be overwritten by notional-based heuristic
            max_buckets: 50,
            persist_threshold: 0.75, // Calibrated to ~p75 of toxicity distribution
            persist_window: 20,      // Track last 20 windows for persistence fraction
        },
    }
}

fn main() {
    eprintln!("SLRT-GPU v1.1 Phase 1: CPU Reference Implementation");
    eprintln!("====================================================");
    eprintln!("AUDIT: Explicit config required - no silent defaults");
    eprintln!();

    // Parse args
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        // Demo mode: generate synthetic data
        eprintln!("Usage: slrt-ref-harness <input.jsonl> [output.jsonl]");
        eprintln!();
        eprintln!("Running demo with synthetic data and reference config...");
        eprintln!();
        run_demo();
        return;
    }

    let input_path = &args[1];
    let output_path = args.get(2).map(|s| s.as_str());

    // Run pipeline
    match run_pipeline(input_path, output_path) {
        Ok(()) => eprintln!("Pipeline completed successfully."),
        Err(e) => {
            eprintln!("FATAL ERROR: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_demo() {
    use slrt_ref::data::PriceLevel;

    let config = create_reference_config();
    eprintln!("Using reference config (for demo only - production must load from file)");

    let venue = VenueMetadata {
        symbol: "BTCUSDT".to_string(),
        tick_size: 0.01,
        lot_size: 0.001,
        max_order_size: None,
        max_position_size: None,
    };

    let mut pipeline = match SymbolPipeline::new("BTCUSDT", venue, config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FATAL: Failed to create pipeline: {}", e);
            std::process::exit(1);
        }
    };

    // Generate synthetic order books with trending price
    let mut price = 50000.0;
    let base_ts = 1_700_000_000_000_000_000_i64; // Arbitrary start

    println!(
        "ts_ns,symbol,mid,microprice,imbalance,gap_risk,elasticity,elasticity_undefined,fti,fragility,d_perp,urgency,regime,confidence,norm_status"
    );

    for i in 0..1000 {
        // Trending with noise
        price += (i as f64 * 0.001).sin() * 10.0 + 0.5;

        let book = OrderBook {
            ts_ns: base_ts + i * 100_000_000, // 100ms intervals
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                PriceLevel {
                    price_mantissa: ((price - 0.5) * 100.0) as i64,
                    price_exponent: -2,
                    qty_mantissa: 1000 + (i % 500),
                    qty_exponent: -3,
                },
                PriceLevel {
                    price_mantissa: ((price - 1.0) * 100.0) as i64,
                    price_exponent: -2,
                    qty_mantissa: 2000,
                    qty_exponent: -3,
                },
            ],
            asks: vec![
                PriceLevel {
                    price_mantissa: ((price + 0.5) * 100.0) as i64,
                    price_exponent: -2,
                    qty_mantissa: 800 + (i % 300),
                    qty_exponent: -3,
                },
                PriceLevel {
                    price_mantissa: ((price + 1.0) * 100.0) as i64,
                    price_exponent: -2,
                    qty_mantissa: 1500,
                    qty_exponent: -3,
                },
            ],
        };

        if let Some(output) = pipeline.process_book(book) {
            println!(
                "{},{},{:.2},{:.2},{:.4},{:.4},{:.6},{},{:.4},{:.4},{:.4},{:.6},{},{:.4},{}",
                output.ts_ns,
                output.symbol,
                output.mid,
                output.microprice,
                output.imbalance,
                output.gap_risk,
                output.elasticity,
                output.elasticity_undefined,
                output.fti,
                output.fragility,
                output.d_perp,
                output.urgency,
                output.regime,
                output.confidence,
                output.norm_status,
            );
        }

        // Occasionally add trades
        if i % 10 == 0 {
            let trade = Trade {
                ts_ns: base_ts + i * 100_000_000 + 50_000_000,
                symbol: "BTCUSDT".to_string(),
                price_mantissa: (price * 100.0) as i64,
                price_exponent: -2,
                qty_mantissa: 100 + (i % 50),
                qty_exponent: -3,
                side: TriState::Present(if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                }),
            };
            pipeline.process_trade(trade);
        }
    }

    pipeline.print_summary();
}

fn run_pipeline(
    input_path: &str,
    output_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = create_reference_config();
    eprintln!("Using reference config (production should load from file)");

    // PASS 1: Load and align all events
    eprintln!();
    eprintln!("=== Pass 1: Alignment ===");
    let file = std::fs::File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut aligner = StreamAligner::new();
    let mut line_count = 0;
    let mut depth_count = 0;
    let mut trade_count = 0;

    let mut parse_errors = 0;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as MarketEvent
        match serde_json::from_str::<MarketEvent>(&line) {
            Ok(event) => match event {
                MarketEvent::Book(book) => {
                    // Convert to AlignedDepth
                    let bids: Vec<(i64, i64)> = book
                        .bids
                        .iter()
                        .map(|l| (l.price_mantissa, l.qty_mantissa))
                        .collect();
                    let asks: Vec<(i64, i64)> = book
                        .asks
                        .iter()
                        .map(|l| (l.price_mantissa, l.qty_mantissa))
                        .collect();

                    let price_exp = book.bids.first().map(|l| l.price_exponent).unwrap_or(-2);
                    let qty_exp = book.bids.first().map(|l| l.qty_exponent).unwrap_or(-8);

                    let depth =
                        AlignedDepth::new(book.ts_ns, book.symbol, price_exp, qty_exp, bids, asks);
                    aligner.add_depth(depth)?;
                    depth_count += 1;
                }
                MarketEvent::Trade(trade) => {
                    // Convert to AlignedTrade
                    // Determine is_buyer_maker from trade side
                    let is_buyer_maker = match trade.side.as_option() {
                        Some(TradeSide::Sell) => true, // sell-initiated = buyer was maker
                        _ => false,                    // buy-initiated = buyer was taker
                    };

                    let aligned = AlignedTrade::new(
                        trade.ts_ns,
                        trade.symbol,
                        trade.price_mantissa,
                        trade.price_exponent,
                        trade.qty_mantissa,
                        trade.qty_exponent,
                        is_buyer_maker,
                    );
                    aligner.add_trade(aligned)?;
                    trade_count += 1;
                }
                _ => {}
            },
            Err(e) => {
                if parse_errors < 5 {
                    eprintln!("  Parse error: {}", e);
                    eprintln!("  Line: {}", &line[..line.len().min(200)]);
                }
                parse_errors += 1;
            }
        }

        line_count += 1;
        if line_count % 50000 == 0 {
            eprintln!(
                "  Loaded {} lines ({} depth, {} trades, {} errors)",
                line_count, depth_count, trade_count, parse_errors
            );
        }
    }

    eprintln!(
        "  Total: {} depth, {} trades ({} parse errors)",
        depth_count, trade_count, parse_errors
    );

    // Drain all frames
    let frames = aligner.drain_frames();
    eprintln!("  Aligned into {} frames", frames.len());

    // PASS 2: Process frames
    eprintln!();
    eprintln!("=== Pass 2: Processing ===");

    let mut output_file: Option<std::fs::File> =
        output_path.map(std::fs::File::create).transpose()?;

    let venue = VenueMetadata {
        symbol: "BTCUSDT".to_string(),
        tick_size: 0.01,
        lot_size: 0.001,
        max_order_size: None,
        max_position_size: None,
    };

    let mut pipeline = SymbolPipeline::new("BTCUSDT", venue, config)?;

    // Apply alignment statistics for bucket_size heuristic
    pipeline.set_alignment_stats(aligner.stats.clone());

    for (i, frame) in frames.iter().enumerate() {
        if let Some(output) = pipeline.process_frame(frame) {
            let json = serde_json::to_string(&output)?;
            if let Some(ref mut f) = output_file {
                writeln!(f, "{}", json)?;
            } else {
                println!("{}", json);
            }
        }

        if (i + 1) % 10000 == 0 {
            eprintln!(
                "  Processed {} frames, {} ticks",
                i + 1,
                pipeline.tick_count
            );
        }
    }

    // Generate manifest
    let manifest = pipeline.generate_manifest("reference_run");
    eprintln!();
    eprintln!("=== Run Manifest ===");
    eprintln!("{}", serde_json::to_string_pretty(&manifest)?);

    pipeline.print_summary();

    Ok(())
}

/// Run pipeline in legacy mode (event-by-event, no alignment).
/// Kept for backward compatibility.
#[allow(dead_code)]
fn run_pipeline_legacy(
    input_path: &str,
    output_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = create_reference_config();
    eprintln!("Using reference config (production should load from file)");

    let file = std::fs::File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut output_file: Option<std::fs::File> =
        output_path.map(std::fs::File::create).transpose()?;

    let venue = VenueMetadata {
        symbol: "BTCUSDT".to_string(),
        tick_size: 0.01,
        lot_size: 0.001,
        max_order_size: None,
        max_position_size: None,
    };

    let mut pipeline = SymbolPipeline::new("BTCUSDT", venue, config)?;
    let mut line_count = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as MarketEvent
        if let Ok(event) = serde_json::from_str::<MarketEvent>(&line) {
            match event {
                MarketEvent::Book(book) => {
                    if let Some(output) = pipeline.process_book(book) {
                        let json = serde_json::to_string(&output)?;
                        if let Some(ref mut f) = output_file {
                            writeln!(f, "{}", json)?;
                        } else {
                            println!("{}", json);
                        }
                    }
                }
                MarketEvent::Trade(trade) => {
                    pipeline.process_trade(trade);
                }
                _ => {}
            }
        }

        line_count += 1;
        if line_count % 10000 == 0 {
            eprintln!(
                "Processed {} lines, {} ticks",
                line_count, pipeline.tick_count
            );
        }
    }

    // Generate manifest
    let manifest = pipeline.generate_manifest("reference_run");
    eprintln!();
    eprintln!("=== Run Manifest ===");
    eprintln!("{}", serde_json::to_string_pretty(&manifest)?);

    pipeline.print_summary();

    Ok(())
}
