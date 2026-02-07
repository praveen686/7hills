//! SLRT Pipeline Integration for Paper Trading
//!
//! Wraps the slrt-ref crate to compute regime classification and metrics.

use slrt_ref::data::{OrderBook, PriceLevel, Trade, TradeSide, TriState};
use slrt_ref::features::{SnapshotFeatures, TradeFlowAccumulator};
use slrt_ref::fragility::{FragilityCalculator, FragilityClipBounds, FragilityWeights};
use slrt_ref::fti::FTITracker;
use slrt_ref::normalization::{DegradedReasons, NormalizationStatus, StateNormalizer};
use slrt_ref::regime::{Regime, RegimeClassifier, RegimeThresholds, ToxicityConfig};
use slrt_ref::sealed::{ELASTICITY_MAX, STATE_DIM};
use slrt_ref::subspace::{RegimeMetrics, SubspaceTracker};

use crate::paper::admission_bridge::{NormalizedAdmission, rr};
use crate::paper::state::DecisionMetrics;

/// SLRT Pipeline configuration.
#[derive(Debug, Clone)]
pub struct SlrtConfig {
    pub symbol: String,
    pub tick_size: f64,
    pub bucket_size: f64,
}

impl Default for SlrtConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            tick_size: 0.01,    // $0.01 tick for BTC
            bucket_size: 100.0, // 100 USDT bucket for toxicity (faster accumulation)
        }
    }
}

/// SLRT Pipeline state - computes regime classification from market data.
pub struct SlrtPipeline {
    config: SlrtConfig,
    /// Last order book
    last_book: Option<OrderBook>,
    /// Price history for FTI
    price_history: Vec<f64>,
    /// Trade flow accumulator (250ms window)
    trade_flow_250ms: TradeFlowAccumulator,
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
    /// Previous normalized state
    prev_state: [f64; STATE_DIM],
    /// Tick count
    tick_count: u64,
    /// First timestamp
    first_ts_ns: Option<i64>,
}

impl SlrtPipeline {
    /// Create a new SLRT pipeline with default config.
    pub fn new(config: SlrtConfig) -> Result<Self, String> {
        // Default fragility weights (from slrt-ref defaults)
        let fragility_weights = FragilityWeights {
            w1_gap_risk: 0.25,
            w2_elasticity: 0.25,
            w3_depth_decay: 0.20,
            w4_spread_z: 0.15,
            w5_depth_slope: 0.15,
        };

        let fragility_bounds = FragilityClipBounds {
            gap_risk_min: 0.0,
            gap_risk_max: 10.0,
            elasticity_min: 0.0,
            elasticity_max: 10.0,
            depth_decay_min: -1.0,
            depth_decay_max: 1.0,
            spread_z_min: -5.0,
            spread_z_max: 5.0,
            depth_slope_min: -1.0,
            depth_slope_max: 1.0,
        };

        let regime_thresholds = RegimeThresholds {
            tau_d_perp: 2.0,
            tau_fragility: 0.6,
            tau_fti_persist: 0.3,
            tau_toxicity_persist: 0.3,
            tau_confidence: 0.5,
            tau_r1_d_perp: 1.0,
            tau_r1_fragility: 0.3,
            tau_r2_d_perp: 1.5,
            tau_r2_fragility: 0.45,
        };

        let toxicity_config = ToxicityConfig {
            bucket_size: config.bucket_size,
            max_buckets: 50,
            persist_threshold: 0.5,
            persist_window: 20,
        };

        let fragility_calc = FragilityCalculator::new(fragility_weights, fragility_bounds)
            .map_err(|e| e.to_string())?;

        let classifier =
            RegimeClassifier::new(regime_thresholds, toxicity_config).map_err(|e| e.to_string())?;

        Ok(Self {
            config,
            last_book: None,
            price_history: Vec::with_capacity(200),
            trade_flow_250ms: TradeFlowAccumulator::new(250),
            fti_tracker: FTITracker::new(),
            fragility_calc,
            normalizer: StateNormalizer::new(),
            subspace: SubspaceTracker::new(),
            classifier,
            prev_state: [0.0; STATE_DIM],
            tick_count: 0,
            first_ts_ns: None,
        })
    }

    /// Process a depth update from Binance.
    /// Returns admission decision if book is valid.
    pub fn process_depth(&mut self, depth: &BinanceDepth) -> Option<NormalizedAdmission> {
        let ts_ns = depth.ts_ns;

        // Always increment tick count first (must match engine's tick)
        self.tick_count += 1;

        // Initialize on first tick
        if self.first_ts_ns.is_none() {
            self.first_ts_ns = Some(ts_ns);
            self.normalizer.reset(ts_ns);
        }

        // Start new toxicity window for this frame
        self.classifier.start_window();

        // Convert to slrt-ref OrderBook
        let book = self.convert_depth(depth);

        // Admission checks
        if book.is_crossed() {
            return Some(NormalizedAdmission::refused(
                vec![rr("CROSSED_BOOK", "bid >= ask")],
                DecisionMetrics::default(),
                "UNKNOWN",
            ));
        }

        if book.bids.is_empty() || book.asks.is_empty() {
            return Some(NormalizedAdmission::refused(
                vec![rr("EMPTY_BOOK", "no bids or asks")],
                DecisionMetrics::default(),
                "UNKNOWN",
            ));
        }

        // Compute snapshot features
        let snapshot = SnapshotFeatures::compute(&book, self.config.tick_size, 1.0);

        // Update trade flow with mid
        self.trade_flow_250ms.add_mid(ts_ns, snapshot.mid);

        // Update price history for FTI
        self.price_history.push(snapshot.mid);
        if self.price_history.len() > 200 {
            self.price_history.remove(0);
        }

        // Update depth tracking
        let total_depth = snapshot.total_bid_depth + snapshot.total_ask_depth;
        self.trade_flow_250ms.add_depth(ts_ns, total_depth);

        // Compute trade flow features
        let trade_flow = self.trade_flow_250ms.compute(ts_ns);

        // Compute FTI (need enough price history)
        let fti_metrics = if self.price_history.len() >= 100 {
            self.fti_tracker.update(&self.price_history)
        } else {
            slrt_ref::fti::FTIMetrics::default()
        };

        // Compute fragility
        let fragility = self.fragility_calc.compute(&snapshot, &trade_flow);

        // Cap elasticity
        let capped_elasticity = trade_flow.elasticity.min(ELASTICITY_MAX);

        // Get current toxicity
        let toxicity = self.classifier.current_toxicity();

        // Build raw state vector
        let raw_state: [f64; STATE_DIM] = [
            snapshot.microprice - snapshot.mid,
            snapshot.imbalance_10,
            capped_elasticity,
            trade_flow.depth_collapse_rate,
            snapshot.gap_risk,
            fti_metrics.fti_level,
            toxicity,
        ];

        // Normalize
        let (normalized_state, norm_status, mut degraded_reasons) =
            self.normalizer.normalize(ts_ns, &raw_state);

        // Add degradation reasons
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

        // Classify regime
        let classification = self.classifier.classify(
            &regime_metrics,
            &fragility,
            &fti_metrics,
            norm_status,
            trade_flow.elasticity_undefined,
            degraded_reasons.bits(),
        );

        // Update state
        self.prev_state = normalized_state;
        self.last_book = Some(book);

        // Build metrics for TUI
        let metrics = DecisionMetrics {
            confidence: Some(classification.confidence),
            d_perp: Some(d_perp),
            fragility: Some(fragility.value),
            toxicity: Some(classification.toxicity),
            toxicity_persist: Some(classification.toxicity_persist),
            fti_level: Some(fti_metrics.fti_level),
            fti_persist: Some(fti_metrics.fti_persist),
            fti_thresh: Some(fti_metrics.persist_threshold),
            fti_elevated: Some(fti_metrics.is_elevated),
            fti_calibrated: Some(fti_metrics.calibrated),
        };

        // Map regime to eligibility
        let is_trade_regime = matches!(classification.regime, Regime::R2 | Regime::R3);
        let eligible = is_trade_regime && !classification.refused;

        let regime_str = classification.regime.as_str();

        if classification.refused {
            let mut reasons = Vec::new();

            // Decode refusal reasons from degraded_reasons bits
            if degraded_reasons.contains(DegradedReasons::MAD_ZERO_SHORT) {
                reasons.push(rr("MAD_ZERO_SHORT", "MAD is zero (<2s)"));
            }
            if degraded_reasons.contains(DegradedReasons::MAD_ZERO_LONG) {
                reasons.push(rr("MAD_ZERO_LONG", "MAD is zero (>2s)"));
            }
            if degraded_reasons.contains(DegradedReasons::UNDEFINED_ELASTICITY) {
                reasons.push(rr("UNDEFINED_ELASTICITY", "volume too low"));
            }
            if degraded_reasons.contains(DegradedReasons::NO_TRADES_WINDOW) {
                reasons.push(rr("NO_TRADES_WINDOW", "no trades in window"));
            }
            if degraded_reasons.contains(DegradedReasons::INSUFFICIENT_HISTORY) {
                reasons.push(rr("INSUFFICIENT_HISTORY", "not enough price history"));
            }

            // Add regime info
            reasons.push(rr(
                "REGIME",
                format!("{} (conf={:.2})", regime_str, classification.confidence),
            ));

            Some(NormalizedAdmission::refused(reasons, metrics, regime_str))
        } else if norm_status == NormalizationStatus::Warmup {
            let elapsed_secs = self
                .first_ts_ns
                .map(|first| (ts_ns - first) / 1_000_000_000)
                .unwrap_or(0);
            let tick_pct = (self.tick_count as f64 / 1000.0 * 100.0).min(100.0);
            let time_pct = (elapsed_secs as f64 / 30.0 * 100.0).min(100.0);

            // CANARY_MODE: bypass warmup refusal after minimal warmup (20s or 200 ticks)
            let canary_mode = std::env::var("CANARY_MODE")
                .map(|v| v == "1")
                .unwrap_or(false);
            let canary_warmup_ok = canary_mode && (elapsed_secs >= 20 || self.tick_count >= 200);

            if canary_warmup_ok {
                // Canary: treat as eligible despite normalizer warmup
                Some(NormalizedAdmission::eligible(
                    Some(classification.confidence),
                    Some(d_perp),
                    Some(fragility.value),
                    Some(classification.toxicity),
                    Some(classification.toxicity_persist),
                    Some(fti_metrics.fti_level),
                    Some(fti_metrics.fti_persist),
                    Some(fti_metrics.persist_threshold),
                    Some(fti_metrics.is_elevated),
                    Some(fti_metrics.calibrated),
                    regime_str,
                ))
            } else {
                Some(NormalizedAdmission::refused(
                    vec![rr(
                        "WARMUP",
                        format!(
                            "ticks: {}/1000 ({:.0}%) | time: {}s/30s ({:.0}%)",
                            self.tick_count, tick_pct, elapsed_secs, time_pct
                        ),
                    )],
                    metrics,
                    regime_str,
                ))
            }
        } else if eligible {
            Some(NormalizedAdmission::eligible(
                Some(classification.confidence),
                Some(d_perp),
                Some(fragility.value),
                Some(classification.toxicity),
                Some(classification.toxicity_persist),
                Some(fti_metrics.fti_level),
                Some(fti_metrics.fti_persist),
                Some(fti_metrics.persist_threshold),
                Some(fti_metrics.is_elevated),
                Some(fti_metrics.calibrated),
                regime_str,
            ))
        } else {
            // Not refused but not eligible (R0/R1)
            Some(NormalizedAdmission::refused(
                vec![rr("REGIME", format!("{} (not R2/R3)", regime_str))],
                metrics,
                regime_str,
            ))
        }
    }

    /// Process a trade from Binance.
    pub fn process_trade(&mut self, trade: &BinanceTrade) {
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
            .unwrap_or(trade.price);

        // Convert to slrt-ref Trade
        let slrt_trade = Trade {
            ts_ns: trade.ts_ns,
            symbol: self.config.symbol.clone(),
            price_mantissa: (trade.price * 100.0) as i64,
            price_exponent: -2,
            qty_mantissa: (trade.qty * 100_000_000.0) as i64,
            qty_exponent: -8,
            side: if trade.is_buyer_maker {
                TriState::Present(TradeSide::Sell) // Taker is seller
            } else {
                TriState::Present(TradeSide::Buy) // Taker is buyer
            },
        };

        // Add to trade flow accumulator
        self.trade_flow_250ms.add_trade(&slrt_trade, microprice);

        // Add to toxicity tracker (needs NOTIONAL volume, not quantity)
        let is_buy = !trade.is_buyer_maker;
        let notional = trade.qty * trade.price;
        self.classifier.add_trade(notional, is_buy);
    }

    /// Convert Binance depth to slrt-ref OrderBook.
    fn convert_depth(&self, depth: &BinanceDepth) -> OrderBook {
        let bids: Vec<PriceLevel> = depth
            .bids
            .iter()
            .map(|(price, qty)| PriceLevel {
                price_mantissa: (*price * 100.0) as i64,
                price_exponent: -2,
                qty_mantissa: (*qty * 100_000_000.0) as i64,
                qty_exponent: -8,
            })
            .collect();

        let asks: Vec<PriceLevel> = depth
            .asks
            .iter()
            .map(|(price, qty)| PriceLevel {
                price_mantissa: (*price * 100.0) as i64,
                price_exponent: -2,
                qty_mantissa: (*qty * 100_000_000.0) as i64,
                qty_exponent: -8,
            })
            .collect();

        OrderBook {
            ts_ns: depth.ts_ns,
            symbol: self.config.symbol.clone(),
            bids,
            asks,
        }
    }

    /// Get current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }
}

/// Binance depth snapshot (parsed from WebSocket).
#[derive(Debug, Clone)]
pub struct BinanceDepth {
    pub ts_ns: i64,
    pub bids: Vec<(f64, f64)>, // (price, qty)
    pub asks: Vec<(f64, f64)>, // (price, qty)
}

/// Binance trade (parsed from WebSocket).
#[derive(Debug, Clone)]
pub struct BinanceTrade {
    pub ts_ns: i64,
    pub price: f64,
    pub qty: f64,
    pub is_buyer_maker: bool,
}
