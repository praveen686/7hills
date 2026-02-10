//! SpreadMeanRevert Strategy - Mean Reversion Baseline
//!
//! Phase 12.1: Simple mean reversion strategy for tournament baseline.
//!
//! ## Strategy Logic
//! - Maintains short-horizon EMA of mid price
//! - LONG when mid deviates below EMA by > X bps (and spread is tight)
//! - SHORT when mid deviates above EMA by > X bps (and spread is tight)
//! - Exit when mid reverts to EMA band
//!
//! ## Fixed-Point Config
//! All numeric fields use mantissa + exponent, no f64.

use crate::canonical::{
    CONFIG_ENCODING_VERSION, CanonicalBytes, canonical_hash, encode_i8, encode_i32, encode_i64,
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
pub const SPREAD_MEAN_REVERT_NAME: &str = "spread_mean_revert";

/// Strategy version.
pub const SPREAD_MEAN_REVERT_VERSION: &str = "1.0.0";

/// Configuration for SpreadMeanRevert strategy.
///
/// ## Fixed-Point Policy
/// ALL numeric fields use mantissa + exponent or integer units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadMeanRevertConfig {
    /// EMA alpha (mantissa, exp -4).
    /// e.g., 1000 with exp -4 = 0.1000 = 10% weight on new value.
    pub ema_alpha_mantissa: i64,
    pub ema_alpha_exponent: i8,

    /// Entry threshold: deviation from EMA in basis points (mantissa, exp -2).
    /// e.g., 20 with exp -2 = 20.00 bps = 0.20%
    pub entry_bps_mantissa: i64,
    pub entry_exponent: i8,

    /// Exit threshold: reversion band in basis points (mantissa, exp -2).
    /// Exit when deviation < this threshold.
    /// e.g., 5 with exp -2 = 5.00 bps = 0.05%
    pub exit_bps_mantissa: i64,
    pub exit_exponent: i8,

    /// Maximum spread to trade (basis points, mantissa, exp -2).
    pub max_spread_bps_mantissa: i64,
    pub spread_exponent: i8,

    /// Minimum ticks before EMA is considered valid.
    pub warmup_ticks: i32,

    /// Position size (mantissa, uses qty_exponent).
    pub position_size_mantissa: i64,
    pub qty_exponent: i8,

    /// Price exponent (typically -2 for USD).
    pub price_exponent: i8,
}

impl Default for SpreadMeanRevertConfig {
    fn default() -> Self {
        Self {
            // 0.05 alpha = 5% weight on new value (20-tick effective window)
            ema_alpha_mantissa: 500,
            ema_alpha_exponent: -4,
            // 25 bps entry threshold
            entry_bps_mantissa: 25,
            entry_exponent: -2,
            // 5 bps exit threshold (reversion band)
            exit_bps_mantissa: 5,
            exit_exponent: -2,
            // Max 30 bps spread
            max_spread_bps_mantissa: 30,
            spread_exponent: -2,
            // 10 tick warmup
            warmup_ticks: 10,
            // 0.01 BTC position
            position_size_mantissa: 1_000_000,
            qty_exponent: -8,
            price_exponent: -2,
        }
    }
}

impl CanonicalBytes for SpreadMeanRevertConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(CONFIG_ENCODING_VERSION);
        encode_i64(&mut buf, self.ema_alpha_mantissa);
        encode_i8(&mut buf, self.ema_alpha_exponent);
        encode_i64(&mut buf, self.entry_bps_mantissa);
        encode_i8(&mut buf, self.entry_exponent);
        encode_i64(&mut buf, self.exit_bps_mantissa);
        encode_i8(&mut buf, self.exit_exponent);
        encode_i64(&mut buf, self.max_spread_bps_mantissa);
        encode_i8(&mut buf, self.spread_exponent);
        encode_i32(&mut buf, self.warmup_ticks);
        encode_i64(&mut buf, self.position_size_mantissa);
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);
        buf
    }
}

impl SpreadMeanRevertConfig {
    /// Load config from TOML file.
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}

/// SpreadMeanRevert Strategy.
pub struct SpreadMeanRevertStrategy {
    config: SpreadMeanRevertConfig,
    config_hash: String,

    // EMA state (mantissa, same exponent as prices)
    ema_mantissa: i64,
    tick_count: i32,

    // Position state
    position_qty_mantissa: i64,
}

impl SpreadMeanRevertStrategy {
    pub fn new(config: SpreadMeanRevertConfig) -> Self {
        let config_hash = canonical_hash(&config);
        Self {
            config,
            config_hash,
            ema_mantissa: 0,
            tick_count: 0,
            position_qty_mantissa: 0,
        }
    }

    /// Update EMA with new mid price (fixed-point arithmetic).
    ///
    /// EMA = alpha * new_value + (1 - alpha) * old_ema
    /// Using fixed-point: ema = (alpha_m * new + (10^(-exp) - alpha_m) * old) / 10^(-exp)
    fn update_ema(&mut self, mid_mantissa: i64) {
        if self.tick_count == 0 {
            // First tick: initialize EMA to mid
            self.ema_mantissa = mid_mantissa;
        } else {
            // EMA update in fixed-point
            // alpha is in mantissa form with exponent ema_alpha_exponent
            // scale = 10^(-exponent) e.g., 10000 for exp -4
            let scale = 10i64.pow((-self.config.ema_alpha_exponent) as u32);
            let alpha = self.config.ema_alpha_mantissa;
            let one_minus_alpha = scale - alpha;

            // new_ema = (alpha * new + one_minus_alpha * old) / scale
            let new_ema = (alpha * mid_mantissa + one_minus_alpha * self.ema_mantissa) / scale;
            self.ema_mantissa = new_ema;
        }
        self.tick_count += 1;
    }

    /// Calculate deviation from EMA in basis points.
    /// Returns (deviation_bps, is_above_ema)
    fn deviation_bps(&self, mid_mantissa: i64) -> (i64, bool) {
        if self.ema_mantissa == 0 {
            return (0, false);
        }

        // deviation_bps = (mid - ema) / ema * 10000
        let diff = mid_mantissa - self.ema_mantissa;
        let deviation = (diff.abs() * 10000) / self.ema_mantissa;
        let is_above = diff > 0;

        (deviation, is_above)
    }

    /// Check if spread is acceptable.
    fn spread_ok(&self, ctx: &StrategyContext) -> bool {
        ctx.market.spread_bps_mantissa() <= self.config.max_spread_bps_mantissa
    }

    /// Check if EMA has warmed up.
    fn ema_ready(&self) -> bool {
        self.tick_count >= self.config.warmup_ticks
    }

    fn create_decision(
        &self,
        ctx: &StrategyContext,
        direction: i8,
        decision_type: &str,
        tag: &str,
    ) -> DecisionEvent {
        let decision_id = Uuid::new_v4();
        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;
        let confidence_mantissa = 10i64.pow((-CONFIDENCE_EXPONENT) as u32);

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
                "policy": "spread_mean_revert_v1",
                "ema_mantissa": self.ema_mantissa,
                "deviation_bps": self.deviation_bps(mid_mantissa).0,
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
        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;

        OrderIntent {
            parent_decision_id,
            symbol: ctx.symbol.to_string(),
            side,
            qty_mantissa: self.config.position_size_mantissa,
            qty_exponent: self.config.qty_exponent,
            limit_price_mantissa: Some(mid_mantissa),
            price_exponent: self.config.price_exponent,
            tag: Some(tag.to_string()),
        }
    }
}

impl Strategy for SpreadMeanRevertStrategy {
    fn name(&self) -> &str {
        SPREAD_MEAN_REVERT_NAME
    }

    fn version(&self) -> &str {
        SPREAD_MEAN_REVERT_VERSION
    }

    fn config_hash(&self) -> String {
        self.config_hash.clone()
    }

    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        // Only process price events
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth) {
            return vec![];
        }

        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;

        // Update EMA first
        self.update_ema(mid_mantissa);

        // Need warmup period
        if !self.ema_ready() {
            return vec![];
        }

        let (deviation_bps, is_above_ema) = self.deviation_bps(mid_mantissa);
        let mut outputs = vec![];

        // Check exit conditions first (if in position)
        if self.position_qty_mantissa != 0 {
            // Exit when deviation shrinks below exit threshold (mean reversion complete)
            if deviation_bps < self.config.exit_bps_mantissa {
                let (direction, side, tag) = if self.position_qty_mantissa > 0 {
                    (-1, Side::Sell, "exit_long_reverted")
                } else {
                    (1, Side::Buy, "exit_short_reverted")
                };

                let decision = self.create_decision(ctx, direction, "exit", tag);
                let intent = self.create_intent(decision.decision_id, ctx, side, tag);
                outputs.push(DecisionOutput::new(decision, intent));
                return outputs;
            }
        }

        // Check entry conditions (only if flat and spread OK)
        if self.position_qty_mantissa == 0
            && self.spread_ok(ctx)
            && deviation_bps >= self.config.entry_bps_mantissa
        {
            if is_above_ema {
                // Price above EMA - expect reversion down - go short
                let decision = self.create_decision(ctx, -1, "entry", "mr_short");
                let intent = self.create_intent(decision.decision_id, ctx, Side::Sell, "mr_short");
                outputs.push(DecisionOutput::new(decision, intent));
            } else {
                // Price below EMA - expect reversion up - go long
                let decision = self.create_decision(ctx, 1, "entry", "mr_long");
                let intent = self.create_intent(decision.decision_id, ctx, Side::Buy, "mr_long");
                outputs.push(DecisionOutput::new(decision, intent));
            }
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, _ctx: &StrategyContext) {
        match fill.side {
            Side::Buy => self.position_qty_mantissa += fill.qty_mantissa,
            Side::Sell => self.position_qty_mantissa -= fill.qty_mantissa,
        }
    }
}

/// Factory function for registry.
pub fn spread_mean_revert_factory(config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
    let config = match config_path {
        Some(path) => SpreadMeanRevertConfig::from_toml(path)?,
        None => SpreadMeanRevertConfig::default(),
    };
    Ok(Box::new(SpreadMeanRevertStrategy::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MarketSnapshot;
    use chrono::Utc;

    fn make_test_market(bid: i64, ask: i64, spread_bps: i64) -> MarketSnapshot {
        MarketSnapshot::v2_all_present(
            bid,
            ask,
            1_000_000,
            1_000_000,
            -2,
            -8,
            spread_bps,
            1234567890000000000,
        )
    }

    #[test]
    fn test_config_canonical_bytes_deterministic() {
        let config1 = SpreadMeanRevertConfig::default();
        let config2 = SpreadMeanRevertConfig::default();
        assert_eq!(config1.canonical_bytes(), config2.canonical_bytes());
        assert_eq!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_config_different_values_different_hash() {
        let config1 = SpreadMeanRevertConfig::default();
        let config2 = SpreadMeanRevertConfig {
            entry_bps_mantissa: 50,
            ..SpreadMeanRevertConfig::default()
        };
        assert_ne!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_ema_initialization() {
        let mut strategy = SpreadMeanRevertStrategy::new(SpreadMeanRevertConfig::default());

        // First tick initializes EMA
        strategy.update_ema(10_000_000);
        assert_eq!(strategy.ema_mantissa, 10_000_000);
        assert_eq!(strategy.tick_count, 1);
    }

    #[test]
    fn test_ema_update() {
        let config = SpreadMeanRevertConfig {
            ema_alpha_mantissa: 5000, // 0.5 alpha for easy math
            ema_alpha_exponent: -4,
            ..SpreadMeanRevertConfig::default()
        };
        let mut strategy = SpreadMeanRevertStrategy::new(config);

        // First tick: EMA = 10000
        strategy.update_ema(10_000);
        assert_eq!(strategy.ema_mantissa, 10_000);

        // Second tick with alpha=0.5: EMA = 0.5 * 12000 + 0.5 * 10000 = 11000
        strategy.update_ema(12_000);
        assert_eq!(strategy.ema_mantissa, 11_000);
    }

    #[test]
    fn test_deviation_calculation() {
        let mut strategy = SpreadMeanRevertStrategy::new(SpreadMeanRevertConfig::default());
        strategy.ema_mantissa = 10_000;

        // 1% above EMA = 100 bps
        let (dev, above) = strategy.deviation_bps(10_100);
        assert_eq!(dev, 100);
        assert!(above);

        // 0.5% below EMA = 50 bps
        let (dev, above) = strategy.deviation_bps(9_950);
        assert_eq!(dev, 50);
        assert!(!above);
    }

    #[test]
    fn test_warmup_required() {
        let mut strategy = SpreadMeanRevertStrategy::new(SpreadMeanRevertConfig {
            warmup_ticks: 5,
            ..SpreadMeanRevertConfig::default()
        });

        let market = make_test_market(10_000_000, 10_000_100, 10);
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test",
            symbol: "BTCUSDT",
            market: &market,
        };

        // First 4 ticks should return empty
        for _ in 0..4 {
            let event = ReplayEvent {
                ts: Utc::now(),
                symbol: "BTCUSDT".to_string(),
                kind: EventKind::PerpQuote,
                payload: serde_json::json!({}),
            };
            let outputs = strategy.on_event(&event, &ctx);
            assert!(outputs.is_empty(), "Should return empty during warmup");
        }
    }

    #[test]
    fn test_spread_filter() {
        let config = SpreadMeanRevertConfig {
            max_spread_bps_mantissa: 20,
            ..SpreadMeanRevertConfig::default()
        };
        let strategy = SpreadMeanRevertStrategy::new(config);

        // Tight spread - OK
        let market_tight = make_test_market(10_000_000, 10_000_100, 10);
        let ctx_tight = StrategyContext {
            ts: Utc::now(),
            run_id: "test",
            symbol: "BTCUSDT",
            market: &market_tight,
        };
        assert!(strategy.spread_ok(&ctx_tight));

        // Wide spread - NOT OK
        let market_wide = make_test_market(10_000_000, 10_000_100, 50);
        let ctx_wide = StrategyContext {
            ts: Utc::now(),
            run_id: "test",
            symbol: "BTCUSDT",
            market: &market_wide,
        };
        assert!(!strategy.spread_ok(&ctx_wide));
    }
}
