//! # Canonical Events Module
//!
//! Platform-wide canonical event definitions with fixed-point representation.
//! All prices and quantities use mantissa/exponent encoding to ensure
//! deterministic cross-platform behavior without floating-point drift.
//!
//! ## Event Types
//! - `QuoteEvent` - Best bid/ask from any venue (spot, perp, options)
//! - `DecisionEvent` - Strategy decision with causality chain
//! - `OrderEvent` - Order lifecycle events (from parent module, re-exported)
//! - `FillEvent` - Execution confirmations (from parent module, re-exported)
//! - `RiskEvent` - Risk limit violations (from parent module, re-exported)
//!
//! ## Correlation IDs
//! All events carry correlation IDs for distributed tracing:
//! - `session_id` - Capture/trading session identifier
//! - `run_id` - Analysis/backtest run identifier
//! - `symbol` - Trading symbol
//! - `venue` - Exchange/venue identifier
//! - `strategy_id` - Strategy that generated the event
//! - `decision_id` - Links decisions to orders
//! - `order_id` - Links orders to fills

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Correlation context for distributed tracing.
/// All events carry this context to enable full causality reconstruction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrelationContext {
    /// Capture/trading session identifier
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Analysis/backtest run identifier
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,

    /// Trading symbol (e.g., "BTCUSDT", "NIFTY26JAN24000CE")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,

    /// Exchange/venue identifier (e.g., "binance", "zerodha")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub venue: Option<String>,

    /// Strategy that generated/processed the event
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy_id: Option<String>,

    /// Decision identifier (links to orders)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_id: Option<Uuid>,

    /// Order identifier (links to fills)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_id: Option<Uuid>,
}

impl CorrelationContext {
    /// Create a new context with session and run IDs.
    pub fn new(session_id: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            session_id: Some(session_id.into()),
            run_id: Some(run_id.into()),
            ..Default::default()
        }
    }

    /// Add symbol context.
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Add venue context.
    pub fn with_venue(mut self, venue: impl Into<String>) -> Self {
        self.venue = Some(venue.into());
        self
    }

    /// Add strategy context.
    pub fn with_strategy(mut self, strategy_id: impl Into<String>) -> Self {
        self.strategy_id = Some(strategy_id.into());
        self
    }

    /// Add decision context.
    pub fn with_decision(mut self, decision_id: Uuid) -> Self {
        self.decision_id = Some(decision_id);
        self
    }

    /// Add order context.
    pub fn with_order(mut self, order_id: Uuid) -> Self {
        self.order_id = Some(order_id);
        self
    }
}

/// Canonical quote event from spot or perp bookTicker stream.
///
/// Uses fixed-point representation (mantissa + exponent) to ensure
/// deterministic cross-platform behavior. All arithmetic should use
/// the mantissa values directly; conversion to f64 is only for display.
///
/// # Examples
/// ```ignore
/// let quote = QuoteEvent {
///     ts: Utc::now(),
///     symbol: "BTCUSDT".to_string(),
///     bid_price_mantissa: 9000012,  // 90000.12 with exponent -2
///     ask_price_mantissa: 9000015,
///     bid_qty_mantissa: 150000000,  // 1.5 with exponent -8
///     ask_qty_mantissa: 200000000,
///     price_exponent: -2,
///     qty_exponent: -8,
///     venue: "binance".to_string(),
///     ctx: CorrelationContext::default(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteEvent {
    /// Timestamp (exchange time preferred, local time if unavailable)
    pub ts: DateTime<Utc>,

    /// Symbol (e.g., "BTCUSDT", "NIFTY26JAN24000CE")
    pub symbol: String,

    /// Best bid price as integer mantissa
    pub bid_price_mantissa: i64,

    /// Best ask price as integer mantissa
    pub ask_price_mantissa: i64,

    /// Best bid quantity as integer mantissa
    pub bid_qty_mantissa: i64,

    /// Best ask quantity as integer mantissa
    pub ask_qty_mantissa: i64,

    /// Exponent for price mantissas (e.g., -2 means divide by 100)
    pub price_exponent: i8,

    /// Exponent for quantity mantissas (e.g., -8 for BTC precision)
    pub qty_exponent: i8,

    /// Venue/exchange identifier
    #[serde(default)]
    pub venue: String,

    /// Correlation context for tracing
    #[serde(default, flatten)]
    pub ctx: CorrelationContext,
}

impl QuoteEvent {
    /// Convert bid price mantissa to f64 (for display only).
    pub fn bid_f64(&self) -> f64 {
        self.bid_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert ask price mantissa to f64 (for display only).
    pub fn ask_f64(&self) -> f64 {
        self.ask_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Compute mid price as f64 (for display only).
    pub fn mid_f64(&self) -> f64 {
        (self.bid_f64() + self.ask_f64()) / 2.0
    }

    /// Compute spread in basis points.
    pub fn spread_bps(&self) -> f64 {
        let mid = self.mid_f64();
        if mid > 0.0 {
            (self.ask_f64() - self.bid_f64()) / mid * 10000.0
        } else {
            f64::MAX
        }
    }

    /// Convert bid quantity mantissa to f64 (for display only).
    pub fn bid_qty_f64(&self) -> f64 {
        self.bid_qty_mantissa as f64 * 10f64.powi(self.qty_exponent as i32)
    }

    /// Convert ask quantity mantissa to f64 (for display only).
    pub fn ask_qty_f64(&self) -> f64 {
        self.ask_qty_mantissa as f64 * 10f64.powi(self.qty_exponent as i32)
    }

    /// Check if quote is valid (positive prices, non-negative quantities).
    pub fn is_valid(&self) -> bool {
        self.bid_price_mantissa > 0
            && self.ask_price_mantissa > 0
            && self.bid_qty_mantissa >= 0
            && self.ask_qty_mantissa >= 0
            && self.bid_price_mantissa <= self.ask_price_mantissa
    }

    /// Compute quote age in milliseconds from a reference time.
    pub fn age_ms(&self, now: DateTime<Utc>) -> i64 {
        (now - self.ts).num_milliseconds()
    }
}

/// Decision event from strategy execution.
///
/// Captures the full decision context including market state at decision time.
/// Used for replay parity validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEvent {
    /// Decision timestamp
    pub ts: DateTime<Utc>,

    /// Unique decision identifier
    pub decision_id: Uuid,

    /// Strategy that made the decision
    pub strategy_id: String,

    /// Symbol being traded
    pub symbol: String,

    /// Decision type (e.g., "entry", "exit", "rebalance", "hold")
    pub decision_type: String,

    /// Decision direction (positive = long, negative = short, zero = neutral)
    pub direction: i8,

    /// Target quantity (mantissa)
    pub target_qty_mantissa: i64,

    /// Quantity exponent
    pub qty_exponent: i8,

    /// Reference price at decision time (mantissa)
    pub reference_price_mantissa: i64,

    /// Price exponent
    pub price_exponent: i8,

    /// Market state snapshot at decision time
    pub market_snapshot: MarketSnapshot,

    /// Confidence score (0.0 to 1.0)
    #[serde(default)]
    pub confidence: f64,

    /// Strategy-specific metadata
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,

    /// Correlation context
    #[serde(default, flatten)]
    pub ctx: CorrelationContext,
}

/// Market state snapshot at decision time for replay validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    /// Best bid price (mantissa)
    pub bid_price_mantissa: i64,
    /// Best ask price (mantissa)
    pub ask_price_mantissa: i64,
    /// Best bid quantity (mantissa)
    pub bid_qty_mantissa: i64,
    /// Best ask quantity (mantissa)
    pub ask_qty_mantissa: i64,
    /// Price exponent
    pub price_exponent: i8,
    /// Quantity exponent
    pub qty_exponent: i8,
    /// Spread in basis points
    pub spread_bps: f64,
    /// Book timestamp (nanoseconds since epoch for causality)
    pub book_ts_ns: i64,
}

// =============================================================================
// FIXED-POINT PARSING UTILITIES
// =============================================================================

/// Pure string-to-mantissa parser (NO float conversion).
///
/// Parses decimal strings like "90000.12" directly to mantissa without
/// intermediate f64 conversion, avoiding cross-platform float drift.
///
/// # Examples
/// - "90000.12" with exponent -2 -> 9000012
/// - "1.50000000" with exponent -8 -> 150000000
/// - "-123.45" with exponent -2 -> -12345
///
/// # Errors
/// Returns error if the string is not a valid decimal number.
pub fn parse_to_mantissa_pure(s: &str, exponent: i8) -> Result<i64, ParseMantissaError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ParseMantissaError::EmptyString);
    }

    // Handle negative numbers
    let (is_negative, s) = if let Some(stripped) = s.strip_prefix('-') {
        (true, stripped)
    } else {
        (false, s)
    };

    // Split on decimal point
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() > 2 {
        return Err(ParseMantissaError::InvalidFormat(s.to_string()));
    }

    let int_part = parts[0];
    let frac_part = if parts.len() == 2 { parts[1] } else { "" };

    // Target decimal places = -exponent (e.g., exponent=-2 means 2 decimals)
    let target_decimals = (-exponent) as usize;

    // Build mantissa string: integer part + fractional part padded/truncated
    let mut mantissa_str = String::with_capacity(int_part.len() + target_decimals);
    mantissa_str.push_str(int_part);

    if frac_part.len() >= target_decimals {
        mantissa_str.push_str(&frac_part[..target_decimals]);

        // Round using next digit if any (banker's rounding)
        if frac_part.len() > target_decimals {
            let next_digit = frac_part.chars().nth(target_decimals).unwrap_or('0');
            if next_digit >= '5' {
                let mut val: i64 = mantissa_str
                    .parse()
                    .map_err(|_| ParseMantissaError::ParseInt(mantissa_str.clone()))?;
                val += 1;
                return Ok(if is_negative { -val } else { val });
            }
        }
    } else {
        mantissa_str.push_str(frac_part);
        for _ in 0..(target_decimals - frac_part.len()) {
            mantissa_str.push('0');
        }
    }

    let val: i64 = mantissa_str
        .parse()
        .map_err(|_| ParseMantissaError::ParseInt(mantissa_str))?;
    Ok(if is_negative { -val } else { val })
}

/// Errors from mantissa parsing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseMantissaError {
    #[error("empty string")]
    EmptyString,
    #[error("invalid decimal format: {0}")]
    InvalidFormat(String),
    #[error("failed to parse integer: {0}")]
    ParseInt(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mantissa_basic() {
        assert_eq!(parse_to_mantissa_pure("90000.12", -2).unwrap(), 9000012);
        assert_eq!(parse_to_mantissa_pure("1.50000000", -8).unwrap(), 150000000);
        assert_eq!(parse_to_mantissa_pure("100", -2).unwrap(), 10000);
        assert_eq!(parse_to_mantissa_pure("0.5", -2).unwrap(), 50);
    }

    #[test]
    fn test_parse_mantissa_negative() {
        assert_eq!(parse_to_mantissa_pure("-123.45", -2).unwrap(), -12345);
        assert_eq!(parse_to_mantissa_pure("-0.01", -2).unwrap(), -1);
    }

    #[test]
    fn test_parse_mantissa_rounding() {
        // 90000.125 with -2 should round up to 9000013
        assert_eq!(parse_to_mantissa_pure("90000.125", -2).unwrap(), 9000013);
        // 90000.124 with -2 should truncate to 9000012
        assert_eq!(parse_to_mantissa_pure("90000.124", -2).unwrap(), 9000012);
    }

    #[test]
    fn test_quote_event_validity() {
        let quote = QuoteEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            bid_price_mantissa: 9000012,
            ask_price_mantissa: 9000015,
            bid_qty_mantissa: 150000000,
            ask_qty_mantissa: 200000000,
            price_exponent: -2,
            qty_exponent: -8,
            venue: "binance".to_string(),
            ctx: CorrelationContext::default(),
        };

        assert!(quote.is_valid());
        assert!((quote.bid_f64() - 90000.12).abs() < 0.001);
        assert!((quote.ask_f64() - 90000.15).abs() < 0.001);
    }

    #[test]
    fn test_correlation_context_builder() {
        let ctx = CorrelationContext::new("session-123", "run-456")
            .with_symbol("BTCUSDT")
            .with_venue("binance")
            .with_strategy("hydra-v1");

        assert_eq!(ctx.session_id, Some("session-123".to_string()));
        assert_eq!(ctx.run_id, Some("run-456".to_string()));
        assert_eq!(ctx.symbol, Some("BTCUSDT".to_string()));
        assert_eq!(ctx.venue, Some("binance".to_string()));
        assert_eq!(ctx.strategy_id, Some("hydra-v1".to_string()));
    }
}
