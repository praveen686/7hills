use std::fmt;

/// Paper intent = a plan to place an order (but in paper mode, we log only).
#[derive(Debug, Clone)]
pub struct PaperIntent {
    pub symbol: String,
    pub side: Side,
    pub qty: f64,
    pub price: Option<f64>, // None = market intent
    pub intent_id: String,
    pub created_at_unix_ms: u64,
}

impl PaperIntent {
    pub fn market(
        symbol: impl Into<String>,
        side: Side,
        qty: f64,
        intent_id: impl Into<String>,
        ts: u64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            qty,
            price: None,
            intent_id: intent_id.into(),
            created_at_unix_ms: ts,
        }
    }

    pub fn limit(
        symbol: impl Into<String>,
        side: Side,
        qty: f64,
        price: f64,
        intent_id: impl Into<String>,
        ts: u64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            side,
            qty,
            price: Some(price),
            intent_id: intent_id.into(),
            created_at_unix_ms: ts,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}
