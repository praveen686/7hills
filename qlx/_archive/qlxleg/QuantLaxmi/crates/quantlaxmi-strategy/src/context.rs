//! Strategy context types.
//!
//! StrategyContext provides read-only market state and run metadata.
//! The trace builder lives in the ENGINE, not the context.

use crate::output::Side;
use chrono::{DateTime, Utc};
use quantlaxmi_models::events::MarketSnapshot;

/// Read-only context passed to strategy.
///
/// Contains current market state and run metadata.
/// The trace builder is NOT here - engine records decisions.
pub struct StrategyContext<'a> {
    /// Current simulation/replay timestamp
    pub ts: DateTime<Utc>,
    /// Run ID for correlation context
    pub run_id: &'a str,
    /// Current symbol being processed
    pub symbol: &'a str,
    /// Current market snapshot (reuse existing MarketSnapshot from quantlaxmi-models)
    pub market: &'a MarketSnapshot,
}

/// Fill notification sent to strategy for position tracking.
///
/// All values are in mantissa form for determinism.
#[derive(Debug, Clone)]
pub struct FillNotification {
    /// Fill timestamp
    pub ts: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Fill side
    pub side: Side,
    /// Filled quantity (mantissa)
    pub qty_mantissa: i64,
    /// Quantity exponent
    pub qty_exponent: i8,
    /// Fill price (mantissa)
    pub price_mantissa: i64,
    /// Price exponent
    pub price_exponent: i8,
    /// Fee (mantissa)
    pub fee_mantissa: i64,
    /// Fee exponent
    pub fee_exponent: i8,
    /// Tag from the original order intent
    pub tag: Option<String>,
}

impl FillNotification {
    /// Convert qty_mantissa to f64 (for display only).
    pub fn qty_f64(&self) -> f64 {
        self.qty_mantissa as f64 * 10f64.powi(self.qty_exponent as i32)
    }

    /// Convert price_mantissa to f64 (for display only).
    pub fn price_f64(&self) -> f64 {
        self.price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert fee_mantissa to f64 (for display only).
    pub fn fee_f64(&self) -> f64 {
        self.fee_mantissa as f64 * 10f64.powi(self.fee_exponent as i32)
    }
}
