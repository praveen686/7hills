//! Strategy output types.
//!
//! Strategies author both decisions AND intents together.
//! The engine does NOT infer intents from decisions.

use quantlaxmi_models::events::DecisionEvent;

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order intent authored by strategy.
///
/// This is what the strategy wants to execute. The engine
/// will route this through risk checks and execution.
#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub symbol: String,
    pub side: Side,
    /// Quantity (mantissa form, uses qty_exponent from config)
    pub qty_mantissa: i64,
    pub qty_exponent: i8,
    /// Limit price (mantissa), None = market order
    pub limit_price_mantissa: Option<i64>,
    pub price_exponent: i8,
    /// Tag for tracking (e.g., "funding_short", "basis_entry")
    pub tag: Option<String>,
}

impl OrderIntent {
    /// Create a market order intent.
    pub fn market(symbol: &str, side: Side, qty_mantissa: i64, qty_exponent: i8) -> Self {
        Self {
            symbol: symbol.to_string(),
            side,
            qty_mantissa,
            qty_exponent,
            limit_price_mantissa: None,
            price_exponent: -2,
            tag: None,
        }
    }

    /// Create a limit order intent.
    pub fn limit(
        symbol: &str,
        side: Side,
        qty_mantissa: i64,
        qty_exponent: i8,
        price_mantissa: i64,
        price_exponent: i8,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            side,
            qty_mantissa,
            qty_exponent,
            limit_price_mantissa: Some(price_mantissa),
            price_exponent,
            tag: None,
        }
    }

    /// Add a tag to the order intent.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tag = Some(tag.to_string());
        self
    }
}

/// Combined output from strategy - decision AND intents together.
///
/// The strategy authors both:
/// - `decision`: The DecisionEvent (recorded to trace by engine)
/// - `intents`: Order intents to execute (not inferred from decision)
///
/// This preserves authored intent while keeping DecisionEvent first-class.
#[derive(Debug, Clone)]
pub struct DecisionOutput {
    /// The decision event (will be recorded to trace by engine)
    pub decision: DecisionEvent,
    /// Order intents to execute (authored by strategy, NOT derived from decision)
    pub intents: Vec<OrderIntent>,
}

impl DecisionOutput {
    /// Create a new decision output with a single order intent.
    pub fn new(decision: DecisionEvent, intent: OrderIntent) -> Self {
        Self {
            decision,
            intents: vec![intent],
        }
    }

    /// Create a decision output with multiple intents.
    pub fn with_intents(decision: DecisionEvent, intents: Vec<OrderIntent>) -> Self {
        Self { decision, intents }
    }

    /// Create a decision output with no execution intents (observation only).
    pub fn observe_only(decision: DecisionEvent) -> Self {
        Self {
            decision,
            intents: vec![],
        }
    }
}
