//! MicroBreakout Strategy - Momentum Baseline
//!
//! Phase 12.1: Simple breakout strategy for tournament baseline.
//!
//! ## Strategy Logic
//! - Maintains rolling window of mid prices
//! - LONG when price breaks above rolling high by X bps (and spread is tight)
//! - SHORT when price breaks below rolling low by X bps (and spread is tight)
//! - Exit on opposite signal, time stop, or stop-loss
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
use std::collections::VecDeque;
use std::path::Path;
use uuid::Uuid;

/// Strategy name constant.
pub const MICRO_BREAKOUT_NAME: &str = "micro_breakout";

/// Strategy version.
pub const MICRO_BREAKOUT_VERSION: &str = "1.0.0";

/// Configuration for MicroBreakout strategy.
///
/// ## Fixed-Point Policy
/// ALL numeric fields use mantissa + exponent or integer units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroBreakoutConfig {
    /// Rolling window size (number of ticks).
    pub window_size: i32,

    /// Breakout threshold in basis points (mantissa, exp -2).
    /// e.g., 10 with exp -2 = 10.00 bps = 0.10%
    pub breakout_bps_mantissa: i64,
    pub breakout_exponent: i8,

    /// Maximum spread to trade (basis points, mantissa, exp -2).
    /// Skip signals when spread > this ceiling.
    pub max_spread_bps_mantissa: i64,
    pub spread_exponent: i8,

    /// Time stop: max holding time in seconds.
    /// Exit if position held longer than this.
    pub time_stop_secs: i32,

    /// Stop-loss in basis points (mantissa, exp -2).
    /// Exit if loss exceeds this threshold.
    pub stop_loss_bps_mantissa: i64,
    pub stop_loss_exponent: i8,

    /// Position size (mantissa, uses qty_exponent).
    pub position_size_mantissa: i64,
    pub qty_exponent: i8,

    /// Price exponent (typically -2 for USD).
    pub price_exponent: i8,
}

impl Default for MicroBreakoutConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            // 15 bps breakout threshold
            breakout_bps_mantissa: 15,
            breakout_exponent: -2,
            // Max 50 bps spread
            max_spread_bps_mantissa: 50,
            spread_exponent: -2,
            // 60 second time stop
            time_stop_secs: 60,
            // 30 bps stop loss
            stop_loss_bps_mantissa: 30,
            stop_loss_exponent: -2,
            // 0.01 BTC position
            position_size_mantissa: 1_000_000,
            qty_exponent: -8,
            price_exponent: -2,
        }
    }
}

impl CanonicalBytes for MicroBreakoutConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(CONFIG_ENCODING_VERSION);
        encode_i32(&mut buf, self.window_size);
        encode_i64(&mut buf, self.breakout_bps_mantissa);
        encode_i8(&mut buf, self.breakout_exponent);
        encode_i64(&mut buf, self.max_spread_bps_mantissa);
        encode_i8(&mut buf, self.spread_exponent);
        encode_i32(&mut buf, self.time_stop_secs);
        encode_i64(&mut buf, self.stop_loss_bps_mantissa);
        encode_i8(&mut buf, self.stop_loss_exponent);
        encode_i64(&mut buf, self.position_size_mantissa);
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);
        buf
    }
}

impl MicroBreakoutConfig {
    /// Load config from TOML file.
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
}

/// Position state for tracking entry.
#[derive(Debug, Clone, Default)]
struct PositionState {
    /// Current position (positive = long, negative = short)
    qty_mantissa: i64,
    /// Entry mid price (mantissa)
    entry_mid_mantissa: i64,
    /// Entry timestamp (nanos)
    entry_ts_ns: i64,
}

/// MicroBreakout Strategy.
pub struct MicroBreakoutStrategy {
    config: MicroBreakoutConfig,
    config_hash: String,

    // Rolling window of mid prices
    mid_window: VecDeque<i64>,

    // Position state
    position: PositionState,
}

impl MicroBreakoutStrategy {
    pub fn new(config: MicroBreakoutConfig) -> Self {
        let config_hash = canonical_hash(&config);
        Self {
            config,
            config_hash,
            mid_window: VecDeque::new(),
            position: PositionState::default(),
        }
    }

    /// Get rolling high from window.
    fn rolling_high(&self) -> Option<i64> {
        self.mid_window.iter().max().copied()
    }

    /// Get rolling low from window.
    fn rolling_low(&self) -> Option<i64> {
        self.mid_window.iter().min().copied()
    }

    /// Check if spread is acceptable.
    fn spread_ok(&self, ctx: &StrategyContext) -> bool {
        // spread_bps_mantissa from market uses same exponent convention
        ctx.market.spread_bps_mantissa() <= self.config.max_spread_bps_mantissa
    }

    /// Check time stop condition.
    fn time_stop_hit(&self, current_ts_ns: i64) -> bool {
        if self.position.qty_mantissa == 0 {
            return false;
        }
        let elapsed_ns = current_ts_ns - self.position.entry_ts_ns;
        let elapsed_secs = elapsed_ns / 1_000_000_000;
        elapsed_secs >= self.config.time_stop_secs as i64
    }

    /// Check stop-loss condition.
    fn stop_loss_hit(&self, current_mid_mantissa: i64) -> bool {
        if self.position.qty_mantissa == 0 {
            return false;
        }

        // Calculate PnL in bps
        let entry = self.position.entry_mid_mantissa;
        if entry == 0 {
            return false;
        }

        // PnL bps = (current - entry) / entry * 10000
        // For fixed-point: pnl_bps_mantissa = (current - entry) * 10000 / entry
        let diff = current_mid_mantissa - entry;
        let pnl_bps = (diff * 10000) / entry;

        // For long: loss when pnl_bps < 0
        // For short: loss when pnl_bps > 0
        let loss_bps = if self.position.qty_mantissa > 0 {
            -pnl_bps // Long position: invert
        } else {
            pnl_bps // Short position
        };

        // Compare with stop loss threshold (both in bps with same exponent)
        loss_bps >= self.config.stop_loss_bps_mantissa
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
                "policy": "micro_breakout_v1",
                "rolling_high": self.rolling_high(),
                "rolling_low": self.rolling_low(),
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

impl Strategy for MicroBreakoutStrategy {
    fn name(&self) -> &str {
        MICRO_BREAKOUT_NAME
    }

    fn version(&self) -> &str {
        MICRO_BREAKOUT_VERSION
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
        let current_ts_ns = ctx.ts.timestamp_nanos_opt().unwrap_or(0);

        // Update rolling window
        self.mid_window.push_back(mid_mantissa);
        while self.mid_window.len() > self.config.window_size as usize {
            self.mid_window.pop_front();
        }

        // Need full window for signals
        if self.mid_window.len() < self.config.window_size as usize {
            return vec![];
        }

        let mut outputs = vec![];

        // Check exit conditions first (if in position)
        if self.position.qty_mantissa != 0 {
            let should_exit = self.time_stop_hit(current_ts_ns) || self.stop_loss_hit(mid_mantissa);

            if should_exit {
                let (direction, side, tag) = if self.position.qty_mantissa > 0 {
                    (-1, Side::Sell, "exit_long")
                } else {
                    (1, Side::Buy, "exit_short")
                };

                let decision = self.create_decision(ctx, direction, "exit", tag);
                let intent = self.create_intent(decision.decision_id, ctx, side, tag);
                outputs.push(DecisionOutput::new(decision, intent));
                return outputs;
            }
        }

        // Check entry conditions (only if flat and spread OK)
        if self.position.qty_mantissa == 0 && self.spread_ok(ctx) {
            let rolling_high = self.rolling_high().unwrap_or(mid_mantissa);
            let rolling_low = self.rolling_low().unwrap_or(mid_mantissa);

            // Calculate breakout threshold
            // threshold = rolling_high * breakout_bps / 10000
            let breakout_up =
                rolling_high + (rolling_high * self.config.breakout_bps_mantissa) / 10000;
            let breakout_down =
                rolling_low - (rolling_low * self.config.breakout_bps_mantissa) / 10000;

            if mid_mantissa > breakout_up {
                // Breakout above - go long
                let decision = self.create_decision(ctx, 1, "entry", "breakout_long");
                let intent =
                    self.create_intent(decision.decision_id, ctx, Side::Buy, "breakout_long");
                outputs.push(DecisionOutput::new(decision, intent));
            } else if mid_mantissa < breakout_down {
                // Breakout below - go short
                let decision = self.create_decision(ctx, -1, "entry", "breakout_short");
                let intent =
                    self.create_intent(decision.decision_id, ctx, Side::Sell, "breakout_short");
                outputs.push(DecisionOutput::new(decision, intent));
            }
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, ctx: &StrategyContext) {
        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;
        let ts_ns = fill.ts.timestamp_nanos_opt().unwrap_or(0);

        match fill.side {
            Side::Buy => {
                self.position.qty_mantissa += fill.qty_mantissa;
                if self.position.qty_mantissa > 0 && self.position.entry_mid_mantissa == 0 {
                    // New long position
                    self.position.entry_mid_mantissa = mid_mantissa;
                    self.position.entry_ts_ns = ts_ns;
                }
            }
            Side::Sell => {
                self.position.qty_mantissa -= fill.qty_mantissa;
                if self.position.qty_mantissa < 0 && self.position.entry_mid_mantissa == 0 {
                    // New short position
                    self.position.entry_mid_mantissa = mid_mantissa;
                    self.position.entry_ts_ns = ts_ns;
                }
            }
        }

        // Reset entry tracking if flat
        if self.position.qty_mantissa == 0 {
            self.position.entry_mid_mantissa = 0;
            self.position.entry_ts_ns = 0;
        }
    }
}

/// Factory function for registry.
pub fn micro_breakout_factory(config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
    let config = match config_path {
        Some(path) => MicroBreakoutConfig::from_toml(path)?,
        None => MicroBreakoutConfig::default(),
    };
    Ok(Box::new(MicroBreakoutStrategy::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MarketSnapshot;
    use chrono::Utc;

    fn make_test_market(bid: i64, ask: i64) -> MarketSnapshot {
        MarketSnapshot::v2_all_present(
            bid,
            ask,
            1_000_000,
            1_000_000,
            -2,
            -8,
            10,
            1234567890000000000,
        )
    }

    fn make_test_market_with_spread(bid: i64, ask: i64, spread_bps: i64) -> MarketSnapshot {
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
        let config1 = MicroBreakoutConfig::default();
        let config2 = MicroBreakoutConfig::default();
        assert_eq!(config1.canonical_bytes(), config2.canonical_bytes());
        assert_eq!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_config_different_values_different_hash() {
        let config1 = MicroBreakoutConfig::default();
        let config2 = MicroBreakoutConfig {
            window_size: 30,
            ..MicroBreakoutConfig::default()
        };
        assert_ne!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_strategy_needs_full_window() {
        let mut strategy = MicroBreakoutStrategy::new(MicroBreakoutConfig {
            window_size: 5,
            ..MicroBreakoutConfig::default()
        });

        let market = make_test_market(10_000_000, 10_000_100);
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test",
            symbol: "BTCUSDT",
            market: &market,
        };

        // First 4 events should return empty (window not full)
        for _ in 0..4 {
            let event = ReplayEvent {
                ts: Utc::now(),
                symbol: "BTCUSDT".to_string(),
                kind: EventKind::PerpQuote,
                payload: serde_json::json!({}),
            };
            let outputs = strategy.on_event(&event, &ctx);
            assert!(outputs.is_empty(), "Should return empty until window full");
        }
    }

    #[test]
    fn test_spread_filter() {
        let config = MicroBreakoutConfig {
            max_spread_bps_mantissa: 20,
            ..MicroBreakoutConfig::default()
        };
        let strategy = MicroBreakoutStrategy::new(config);

        // Tight spread - OK
        let market_tight = make_test_market_with_spread(10_000_000, 10_000_100, 10);
        let ctx_tight = StrategyContext {
            ts: Utc::now(),
            run_id: "test",
            symbol: "BTCUSDT",
            market: &market_tight,
        };
        assert!(strategy.spread_ok(&ctx_tight));

        // Wide spread - NOT OK
        let market_wide = make_test_market_with_spread(10_000_000, 10_000_100, 50);
        let ctx_wide = StrategyContext {
            ts: Utc::now(),
            run_id: "test",
            symbol: "BTCUSDT",
            market: &market_wide,
        };
        assert!(!strategy.spread_ok(&ctx_wide));
    }
}
