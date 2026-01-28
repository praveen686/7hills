//! FundingBias Strategy v2
//!
//! Migrated from backtest.rs to use Phase 2 Strategy SDK.
//!
//! ## Strategy Logic
//! - Goes SHORT perp when funding rate > threshold (longs pay shorts)
//! - Goes LONG perp when funding rate < -threshold (shorts pay longs)
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

/// Strategy version.
pub const FUNDING_BIAS_VERSION: &str = "2.0.0";

/// Configuration for FundingBias strategy.
///
/// ## Fixed-Point Policy
/// ALL numeric fields use mantissa + exponent or integer units.
/// NO f64 fields - this ensures deterministic config hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingBiasConfig {
    /// Funding rate threshold (mantissa, exponent = -6).
    /// e.g., 100 = 0.0001 = 1 basis point funding rate.
    ///
    /// The strategy will:
    /// - SHORT when funding_rate > threshold (longs pay shorts)
    /// - LONG when funding_rate < -threshold (shorts pay longs)
    pub threshold_mantissa: i64,

    /// Threshold exponent (typically -6 for funding rates).
    pub threshold_exponent: i8,

    /// Position size (mantissa, uses qty_exponent).
    /// e.g., 1_000_000 with exp -8 = 0.01 BTC.
    pub position_size_mantissa: i64,

    /// Quantity exponent (typically -8 for crypto).
    pub qty_exponent: i8,

    /// Price exponent (typically -2 for USD prices).
    pub price_exponent: i8,
}

impl Default for FundingBiasConfig {
    fn default() -> Self {
        Self {
            // 100 with exp -6 = 0.0001 = 1 basis point
            threshold_mantissa: 100,
            threshold_exponent: -6,
            // 1_000_000 with exp -8 = 0.01 BTC
            position_size_mantissa: 1_000_000,
            qty_exponent: -8,
            price_exponent: -2,
        }
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
        encode_i64(&mut buf, self.position_size_mantissa);
        encode_i8(&mut buf, self.qty_exponent);
        encode_i8(&mut buf, self.price_exponent);
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

/// FundingBias Strategy v2.
///
/// Implements the Phase 2 Strategy trait:
/// - Authors DecisionEvent directly
/// - Returns DecisionOutput { decision, intents }
/// - Uses fixed-point config (no f64)
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
                "policy": "funding_bias_v2",
            }),
            ctx: CorrelationContext {
                run_id: Some(ctx.run_id.to_string()),
                venue: Some("paper".to_string()),
                strategy_id: Some(self.strategy_id()),
                ..Default::default()
            },
        }
    }

    /// Create an OrderIntent for execution.
    ///
    /// # Arguments
    /// * `parent_decision_id` - The decision that authored this intent
    /// * `ctx` - Strategy context with market state
    /// * `side` - Buy or Sell
    /// * `tag` - Tracking tag
    fn create_intent(
        &self,
        parent_decision_id: Uuid,
        ctx: &StrategyContext,
        side: Side,
        tag: &str,
    ) -> OrderIntent {
        // Use mid price as reference for market order
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
        if event.kind == EventKind::Funding
            && let Some(rate) = event.payload.get("rate").and_then(|v| v.as_f64())
        {
            // Convert float funding rate to mantissa (matching threshold exponent)
            // funding rate is typically small (e.g., 0.0001 = 1bp)
            self.current_funding_rate_mantissa =
                (rate / 10f64.powi(self.config.threshold_exponent as i32)).round() as i64;
        }

        // Only trade on perp price updates (need prices)
        if !matches!(event.kind, EventKind::PerpQuote | EventKind::PerpDepth) {
            return vec![];
        }

        let mut outputs = vec![];

        // Strategy logic: trade based on funding rate direction
        // - Funding positive & not short → go short (shorts receive funding)
        // - Funding negative & not long → go long (longs receive funding)

        if self.current_funding_rate_mantissa > self.config.threshold_mantissa
            && self.position_qty_mantissa >= 0
        {
            // Funding positive, go short
            let decision = self.create_decision(ctx, -1, "entry", "funding_short");
            let intent = self.create_intent(decision.decision_id, ctx, Side::Sell, "funding_short");
            outputs.push(DecisionOutput::new(decision, intent));
        } else if self.current_funding_rate_mantissa < -self.config.threshold_mantissa
            && self.position_qty_mantissa <= 0
        {
            // Funding negative, go long
            let decision = self.create_decision(ctx, 1, "entry", "funding_long");
            let intent = self.create_intent(decision.decision_id, ctx, Side::Buy, "funding_long");
            outputs.push(DecisionOutput::new(decision, intent));
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
    fn test_strategy_returns_decision_output() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Set high funding rate to trigger short
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

        // Should have one output (short signal)
        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];

        // Decision should be authored
        assert_eq!(output.decision.direction, -1); // Short
        assert_eq!(output.decision.decision_type, "entry");
        assert!(output.decision.strategy_id.starts_with(FUNDING_BIAS_NAME));

        // Intent should match decision
        assert_eq!(output.intents.len(), 1);
        assert_eq!(output.intents[0].side, Side::Sell);
        assert_eq!(output.intents[0].tag, Some("funding_short".to_string()));
    }

    #[test]
    fn test_strategy_no_signal_below_threshold() {
        let mut strategy = FundingBiasStrategy::new(FundingBiasConfig::default());

        // Funding rate below threshold
        strategy.current_funding_rate_mantissa = 50; // Below threshold of 100

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

        // Should have no output (no signal)
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

        // Buy fill
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
            tag: Some("funding_long".to_string()),
        };

        strategy.on_fill(&fill, &ctx);
        assert_eq!(strategy.position_qty_mantissa, 1_000_000);

        // Sell fill
        let fill2 = FillNotification {
            side: Side::Sell,
            qty_mantissa: 500_000,
            ..fill
        };

        strategy.on_fill(&fill2, &ctx);
        assert_eq!(strategy.position_qty_mantissa, 500_000);
    }
}
