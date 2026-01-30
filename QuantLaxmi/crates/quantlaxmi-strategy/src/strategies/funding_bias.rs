//! FundingBias Strategy v2.1
//!
//! Phase 27.1: Added exit logic + hysteresis band for complete entry/exit cycles.
//!
//! ## Strategy Logic
//! Entry:
//! - Goes SHORT perp when funding rate > threshold (longs pay shorts)
//! - Goes LONG perp when funding rate < -threshold (shorts pay longs)
//!
//! Exit (hysteresis):
//! - Exit LONG when funding rate >= exit_band (reverted toward positive)
//! - Exit SHORT when funding rate <= -exit_band (reverted toward negative)
//!
//! Flip:
//! - If long and funding crosses +threshold → exit long, enter short
//! - If short and funding crosses -threshold → exit short, enter long
//!
//! ## Fixed-Point Config
//! All numeric fields use mantissa + exponent, no f64.

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
pub const FUNDING_BIAS_NAME: &str = "funding_bias";

/// Strategy version (2.1.0 = Phase 27.1 with exit logic).
pub const FUNDING_BIAS_VERSION: &str = "2.1.0";

/// Configuration for FundingBias strategy.
///
/// ## Fixed-Point Policy
/// ALL numeric fields use mantissa + exponent or integer units.
/// NO f64 fields - this ensures deterministic config hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingBiasConfig {
    /// Funding rate threshold for entry (mantissa).
    /// e.g., 100 with exp -6 = 0.0001 = 1 basis point funding rate.
    ///
    /// Entry rules:
    /// - SHORT when funding_rate > threshold (longs pay shorts)
    /// - LONG when funding_rate < -threshold (shorts pay longs)
    pub threshold_mantissa: i64,

    /// Threshold exponent (typically -6 for funding rates).
    pub threshold_exponent: i8,

    /// Exit band for hysteresis (mantissa, same exponent as threshold).
    /// Default: threshold_mantissa / 2
    ///
    /// Exit rules:
    /// - Exit LONG when funding_rate >= exit_band
    /// - Exit SHORT when funding_rate <= -exit_band
    #[serde(default)]
    pub exit_band_mantissa: Option<i64>,

    /// Position size (mantissa, uses qty_exponent).
    /// e.g., 1_000_000 with exp -8 = 0.01 BTC.
    pub position_size_mantissa: i64,

    /// Quantity exponent (typically -8 for crypto).
    pub qty_exponent: i8,

    /// Price exponent (typically -2 for USD prices).
    pub price_exponent: i8,

    /// Allow trading on SpotQuote events (in addition to PerpQuote/PerpDepth).
    /// Useful when perp depth data is incomplete but spot quotes are available.
    /// Default: false (only trade on perp events)
    #[serde(default)]
    pub trade_on_spot_quotes: bool,
}

impl Default for FundingBiasConfig {
    fn default() -> Self {
        Self {
            // 100 with exp -6 = 0.0001 = 1 basis point
            threshold_mantissa: 100,
            threshold_exponent: -6,
            // Default exit band = threshold / 2
            exit_band_mantissa: None,
            // 1_000_000 with exp -8 = 0.01 BTC
            position_size_mantissa: 1_000_000,
            qty_exponent: -8,
            price_exponent: -2,
            // Default: only trade on perp events
            trade_on_spot_quotes: false,
        }
    }
}

impl FundingBiasConfig {
    /// Get effective exit band (default = threshold / 2).
    pub fn effective_exit_band(&self) -> i64 {
        self.exit_band_mantissa
            .unwrap_or(self.threshold_mantissa / 2)
    }
}

impl CanonicalBytes for FundingBiasConfig {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // First byte: encoding version
        buf.push(CONFIG_ENCODING_VERSION);
        // Fields in fixed order
        encode_i64(&mut buf, self.threshold_mantissa);
        encode_i8(&mut buf, self.threshold_exponent);
        encode_i64(&mut buf, self.effective_exit_band()); // Always encode effective value
        encode_i64(&mut buf, self.position_size_mantissa);
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);
        buf.push(if self.trade_on_spot_quotes { 1 } else { 0 });
        buf
    }
}

impl FundingBiasConfig {
    /// Load config from TOML file.
    pub fn from_toml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Get threshold as f64 (for display only).
    pub fn threshold_f64(&self) -> f64 {
        self.threshold_mantissa as f64 * 10f64.powi(self.threshold_exponent as i32)
    }

    /// Get position size as f64 (for display only).
    pub fn position_size_f64(&self) -> f64 {
        self.position_size_mantissa as f64 * 10f64.powi(self.qty_exponent as i32)
    }
}

/// FundingBias Strategy v2.1.
///
/// Implements the Phase 2 Strategy trait with complete entry/exit lifecycle:
/// - Authors DecisionEvent directly
/// - Returns DecisionOutput { decision, intents }
/// - Uses fixed-point config (no f64)
/// - Hysteresis exit band prevents flip-flop noise
pub struct FundingBiasStrategy {
    config: FundingBiasConfig,
    config_hash: String,

    // Runtime state (mutable)
    /// Current funding rate (mantissa, same exponent as threshold)
    current_funding_rate_mantissa: i64,
    /// Current position (mantissa, positive = long, negative = short)
    position_qty_mantissa: i64,
}

impl FundingBiasStrategy {
    /// Create a new FundingBias strategy with the given config.
    pub fn new(config: FundingBiasConfig) -> Self {
        let config_hash = canonical_hash(&config);
        Self {
            config,
            config_hash,
            current_funding_rate_mantissa: 0,
            position_qty_mantissa: 0,
        }
    }

    /// Create a DecisionEvent for a trading decision.
    fn create_decision(
        &self,
        ctx: &StrategyContext,
        direction: i8,
        decision_type: &str,
        tag: &str,
    ) -> DecisionEvent {
        let decision_id = Uuid::new_v4();

        // Reference price: use mid price from market snapshot
        let mid_mantissa = (ctx.market.bid_price_mantissa() + ctx.market.ask_price_mantissa()) / 2;

        // Confidence: 1.0 for simple rule-based strategy
        let confidence_mantissa = 10i64.pow((-CONFIDENCE_EXPONENT) as u32); // 10000 = 1.0

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
                "funding_rate_mantissa": self.current_funding_rate_mantissa,
                "funding_rate_exponent": self.config.threshold_exponent,
                "threshold_mantissa": self.config.threshold_mantissa,
                "exit_band_mantissa": self.config.effective_exit_band(),
                "policy": "funding_bias_v2.1",
            }),
            ctx: CorrelationContext {
                run_id: Some(ctx.run_id.to_string()),
                venue: Some("paper".to_string()),
                strategy_id: Some(self.strategy_id()),
                ..Default::default()
            },
        }
    }

    /// Create an OrderIntent for execution (market order).
    fn create_intent(
        &self,
        parent_decision_id: Uuid,
        ctx: &StrategyContext,
        side: Side,
        tag: &str,
    ) -> OrderIntent {
        // Market order: no limit price (executes at best available)
        OrderIntent {
            parent_decision_id,
            symbol: ctx.symbol.to_string(),
            side,
            qty_mantissa: self.config.position_size_mantissa,
            qty_exponent: self.config.qty_exponent,
            limit_price_mantissa: None, // Market order
            price_exponent: self.config.price_exponent,
            tag: Some(tag.to_string()),
        }
    }

    /// Check if we should exit a long position.
    fn should_exit_long(&self) -> bool {
        // Exit long when funding rises above -exit_band (lost negative edge)
        // Long entry requires funding < -threshold; exit when funding > -exit_band
        self.current_funding_rate_mantissa > -self.config.effective_exit_band()
    }

    /// Check if we should exit a short position.
    fn should_exit_short(&self) -> bool {
        // Exit short when funding drops below +exit_band (lost positive edge)
        // Short entry requires funding > +threshold; exit when funding < +exit_band
        self.current_funding_rate_mantissa < self.config.effective_exit_band()
    }

    /// Check if we should enter short.
    fn should_enter_short(&self) -> bool {
        self.current_funding_rate_mantissa > self.config.threshold_mantissa
    }

    /// Check if we should enter long.
    fn should_enter_long(&self) -> bool {
        self.current_funding_rate_mantissa < -self.config.threshold_mantissa
    }

    /// Check current position state.
    fn is_long(&self) -> bool {
        self.position_qty_mantissa > 0
    }

    fn is_short(&self) -> bool {
        self.position_qty_mantissa < 0
    }

    fn is_flat(&self) -> bool {
        self.position_qty_mantissa == 0
    }
}

impl Strategy for FundingBiasStrategy {
    fn name(&self) -> &str {
        FUNDING_BIAS_NAME
    }

    fn version(&self) -> &str {
        FUNDING_BIAS_VERSION
    }

    fn config_hash(&self) -> String {
        self.config_hash.clone()
    }

    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext) -> Vec<DecisionOutput> {
        // Update funding rate from Funding events
        // Data format: funding_rate_mantissa + rate_exponent (canonical integer format)
        if event.kind == EventKind::Funding {
            if let Some(rate_mantissa) = event
                .payload
                .get("funding_rate_mantissa")
                .and_then(|v| v.as_i64())
            {
                let rate_exponent = event
                    .payload
                    .get("rate_exponent")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-8) as i8;

                // Convert to our threshold exponent for comparison
                let exp_diff = rate_exponent as i32 - self.config.threshold_exponent as i32;
                self.current_funding_rate_mantissa = if exp_diff >= 0 {
                    rate_mantissa * 10i64.pow(exp_diff as u32)
                } else {
                    rate_mantissa / 10i64.pow((-exp_diff) as u32)
                };
            }
        }

        // Only trade on price updates (need prices for execution)
        // Accept perp events always; accept spot events if configured
        let is_perp_event = matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth);
        let is_spot_event = matches!(event.kind, EventKind::SpotQuote);
        if !is_perp_event && !(is_spot_event && self.config.trade_on_spot_quotes) {
            return vec![];
        }

        let mut outputs = vec![];

        // =================================================================
        // PHASE 27.1: Complete entry/exit/flip logic
        // =================================================================

        if self.is_long() {
            // Currently LONG - check for exit or flip to short
            if self.should_enter_short() {
                // FLIP: exit long + enter short (two decisions, stable order)
                let exit_decision = self.create_decision(ctx, 0, "exit", "exit_long");
                let exit_intent =
                    self.create_intent(exit_decision.decision_id, ctx, Side::Sell, "exit_long");
                outputs.push(DecisionOutput::new(exit_decision, exit_intent));

                let entry_decision = self.create_decision(ctx, -1, "entry", "entry_short");
                let entry_intent =
                    self.create_intent(entry_decision.decision_id, ctx, Side::Sell, "entry_short");
                outputs.push(DecisionOutput::new(entry_decision, entry_intent));
            } else if self.should_exit_long() {
                // EXIT LONG only (funding reverted inside hysteresis band)
                let decision = self.create_decision(ctx, 0, "exit", "exit_long");
                let intent = self.create_intent(decision.decision_id, ctx, Side::Sell, "exit_long");
                outputs.push(DecisionOutput::new(decision, intent));
            }
        } else if self.is_short() {
            // Currently SHORT - check for exit or flip to long
            if self.should_enter_long() {
                // FLIP: exit short + enter long (two decisions, stable order)
                let exit_decision = self.create_decision(ctx, 0, "exit", "exit_short");
                let exit_intent =
                    self.create_intent(exit_decision.decision_id, ctx, Side::Buy, "exit_short");
                outputs.push(DecisionOutput::new(exit_decision, exit_intent));

                let entry_decision = self.create_decision(ctx, 1, "entry", "entry_long");
                let entry_intent =
                    self.create_intent(entry_decision.decision_id, ctx, Side::Buy, "entry_long");
                outputs.push(DecisionOutput::new(entry_decision, entry_intent));
            } else if self.should_exit_short() {
                // EXIT SHORT only (funding reverted inside hysteresis band)
                let decision = self.create_decision(ctx, 0, "exit", "exit_short");
                let intent = self.create_intent(decision.decision_id, ctx, Side::Buy, "exit_short");
                outputs.push(DecisionOutput::new(decision, intent));
            }
        } else {
            // Currently FLAT - check for new entry
            if self.should_enter_short() {
                // ENTER SHORT (funding positive, shorts receive funding)
                let decision = self.create_decision(ctx, -1, "entry", "entry_short");
                let intent =
                    self.create_intent(decision.decision_id, ctx, Side::Sell, "entry_short");
                outputs.push(DecisionOutput::new(decision, intent));
            } else if self.should_enter_long() {
                // ENTER LONG (funding negative, longs receive funding)
                let decision = self.create_decision(ctx, 1, "entry", "entry_long");
                let intent = self.create_intent(decision.decision_id, ctx, Side::Buy, "entry_long");
                outputs.push(DecisionOutput::new(decision, intent));
            }
        }

        outputs
    }

    fn on_fill(&mut self, fill: &FillNotification, _ctx: &StrategyContext) {
        // Update position tracking
        match fill.side {
            Side::Buy => self.position_qty_mantissa += fill.qty_mantissa,
            Side::Sell => self.position_qty_mantissa -= fill.qty_mantissa,
        }
    }
}

/// Factory function for registry.
pub fn funding_bias_factory(config_path: Option<&Path>) -> Result<Box<dyn Strategy>> {
    let config = match config_path {
        Some(path) => FundingBiasConfig::from_toml(path)?,
        None => FundingBiasConfig::default(),
    };
    Ok(Box::new(FundingBiasStrategy::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MarketSnapshot;
    use chrono::Utc;

    fn make_test_market() -> MarketSnapshot {
        MarketSnapshot::v2_all_present(
            10_000_000,          // bid_price_mantissa: $100,000
            10_000_100,          // ask_price_mantissa: $100,001
            1_000_000,           // bid_qty_mantissa
            1_000_000,           // ask_qty_mantissa
            -2,                  // price_exponent
            -8,                  // qty_exponent
            10,                  // spread_bps_mantissa: 0.10 bps
            1234567890000000000, // book_ts_ns
        )
    }

    #[test]
    fn test_config_canonical_bytes_deterministic() {
        let config1 = FundingBiasConfig::default();
        let config2 = FundingBiasConfig::default();

        assert_eq!(config1.canonical_bytes(), config2.canonical_bytes());
        assert_eq!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_config_canonical_bytes_includes_version() {
        let config = FundingBiasConfig::default();
        let bytes = config.canonical_bytes();
        assert_eq!(bytes[0], CONFIG_ENCODING_VERSION);
    }

    #[test]
    fn test_config_different_values_different_hash() {
        let config1 = FundingBiasConfig::default();
        let config2 = FundingBiasConfig {
            threshold_mantissa: 200, // Different
            ..FundingBiasConfig::default()
        };

        assert_ne!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_effective_exit_band() {
        let config = FundingBiasConfig {
            threshold_mantissa: 100,
            exit_band_mantissa: None,
            ..FundingBiasConfig::default()
        };
        assert_eq!(config.effective_exit_band(), 50); // threshold / 2

        let config2 = FundingBiasConfig {
            threshold_mantissa: 100,
            exit_band_mantissa: Some(30),
            ..FundingBiasConfig::default()
        };
        assert_eq!(config2.effective_exit_band(), 30); // explicit override
    }

    #[test]
    fn test_strategy_id_format() {
        let strategy = FundingBiasStrategy::new(FundingBiasConfig::default());
        let id = strategy.strategy_id();

        // Should be "{name}:{version}:{hash}"
        let parts: Vec<&str> = id.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], FUNDING_BIAS_NAME);
        assert_eq!(parts[1], FUNDING_BIAS_VERSION);
        assert_eq!(parts[2].len(), 64); // Full SHA-256 hash
    }

    #[test]
    fn test_strategy_short_id() {
        let strategy = FundingBiasStrategy::new(FundingBiasConfig::default());
        let short_id = strategy.short_id();

        // Should be "{name}:{version}:{hash[0:8]}"
        let parts: Vec<&str> = short_id.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2].len(), 8); // Truncated hash
    }

    #[test]
    fn test_entry_short_when_funding_positive() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Set high funding rate to trigger short entry
        strategy.current_funding_rate_mantissa = 200; // Above threshold of 100

        let market = make_test_market();
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({}),
        };

        let outputs = strategy.on_event(&event, &ctx);

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        assert_eq!(output.decision.direction, -1); // Short
        assert_eq!(output.decision.decision_type, "entry");
        assert_eq!(output.intents[0].side, Side::Sell);
        assert_eq!(output.intents[0].tag, Some("entry_short".to_string()));
    }

    #[test]
    fn test_exit_short_when_funding_drops() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Simulate existing short position
        strategy.position_qty_mantissa = -1_000_000;
        // Funding dropped below exit band (threshold=100, exit_band=50)
        // Short was entered when funding > 100, now funding < 50 (lost edge)
        strategy.current_funding_rate_mantissa = 40; // Below exit_band (50)

        let market = make_test_market();
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({}),
        };

        let outputs = strategy.on_event(&event, &ctx);

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];
        assert_eq!(output.decision.decision_type, "exit");
        assert_eq!(output.intents[0].side, Side::Buy); // Buy to close short
        assert_eq!(output.intents[0].tag, Some("exit_short".to_string()));
    }

    #[test]
    fn test_flip_long_to_short() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Simulate existing long position
        strategy.position_qty_mantissa = 1_000_000;
        // Funding flipped to strongly positive (crosses entry threshold for short)
        strategy.current_funding_rate_mantissa = 200; // Above +threshold

        let market = make_test_market();
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({}),
        };

        let outputs = strategy.on_event(&event, &ctx);

        // Should have 2 outputs: exit_long + entry_short
        assert_eq!(outputs.len(), 2);

        // First: exit long
        assert_eq!(outputs[0].decision.decision_type, "exit");
        assert_eq!(outputs[0].intents[0].tag, Some("exit_long".to_string()));

        // Second: enter short
        assert_eq!(outputs[1].decision.decision_type, "entry");
        assert_eq!(outputs[1].intents[0].tag, Some("entry_short".to_string()));
    }

    #[test]
    fn test_no_signal_in_neutral_zone_when_flat() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Funding rate in neutral zone (between exit_band and threshold)
        // When flat, no entry (threshold=100 not crossed), no exit (no position)
        strategy.current_funding_rate_mantissa = 60; // Above exit_band (50) but below threshold (100)

        let market = make_test_market();
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({}),
        };

        let outputs = strategy.on_event(&event, &ctx);

        // Should have no output (neutral zone, no position)
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_hold_short_in_hysteresis_zone() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Have a short position, funding in hysteresis zone (above exit_band, below threshold)
        strategy.position_qty_mantissa = -1_000_000;
        strategy.current_funding_rate_mantissa = 70; // Above exit_band (50), below threshold (100)

        let market = make_test_market();
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        let event = ReplayEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            kind: EventKind::PerpQuote,
            payload: serde_json::json!({}),
        };

        let outputs = strategy.on_event(&event, &ctx);

        // Should hold position (in hysteresis zone - above exit_band, so no exit)
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_on_fill_updates_position() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        let market = make_test_market();
        let ctx = StrategyContext {
            ts: Utc::now(),
            run_id: "test-run",
            symbol: "BTCUSDT",
            market: &market,
        };

        // Buy fill (entry long)
        let fill = FillNotification {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            qty_mantissa: 1_000_000,
            qty_exponent: -8,
            price_mantissa: 10_000_000,
            price_exponent: -2,
            fee_mantissa: 1000,
            fee_exponent: -8,
            tag: Some("entry_long".to_string()),
        };

        strategy.on_fill(&fill, &ctx);
        assert_eq!(strategy.position_qty_mantissa, 1_000_000);
        assert!(strategy.is_long());

        // Sell fill (exit long)
        let fill2 = FillNotification {
            side: Side::Sell,
            qty_mantissa: 1_000_000,
            tag: Some("exit_long".to_string()),
            ..fill
        };

        strategy.on_fill(&fill2, &ctx);
        assert_eq!(strategy.position_qty_mantissa, 0);
        assert!(strategy.is_flat());
    }
}
