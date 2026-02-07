//! Paper Fill Model Trait
//!
//! Defines the interface for paper trading fill simulation.
//! Runners implement this for their specific markets (India F&O, Crypto, etc.)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Fill side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillSide {
    Buy,
    Sell,
}

/// Fees paid on a fill.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Fees {
    pub total: f64,
}

/// A successful fill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub side: FillSide,
    pub qty: f64,
    pub price: f64,
    pub fees: Fees,
}

/// Reason for fill rejection.
///
/// This is a venue-agnostic rejection type. Runners can map their
/// specific rejection reasons to this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillRejection {
    /// No executable quote (bid/ask missing or invalid)
    NoExecutableQuote { reason: String },
    /// Quote is stale
    StaleQuote { age_ms: u32, threshold_ms: u32 },
    /// Insufficient quantity at top of book
    InsufficientQuantity { requested: i64, available: i64 },
    /// Other rejection reason
    Other { reason: String },
}

/// Top-of-book price provider for conservative MTM.
///
/// Implemented by snapshot types to provide bid/ask for any token.
/// Used by engine to compute equity with conservative marking:
/// - Long positions marked at bid (what we'd get if we sold)
/// - Short positions marked at ask (what we'd pay to cover)
pub trait TopOfBookProvider {
    /// Get best bid and ask for an instrument.
    ///
    /// Returns `Some((bid, ask))` if prices available, `None` otherwise.
    fn best_bid_ask(&self, token: u32) -> Option<(f64, f64)>;
}

/// Bid/ask simulator + slippage/fees.
///
/// Runners implement exchange-specific and instrument-specific rules.
///
/// ## Contract
///
/// - `try_fill` takes the intent by reference so identity is preserved
/// - Returns `Ok(Fill)` on successful fill
/// - Returns `Err(FillRejection)` on rejection (logged by engine)
/// - Fill price must be from executable quotes only (bid for sell, ask for buy)
/// - Never fill at mid or LTP
pub trait PaperFillModel<TSnapshot, TIntent>: Send + Sync {
    /// Attempt to fill an intent against the current snapshot.
    ///
    /// # Arguments
    /// - `ts`: Timestamp of the fill attempt
    /// - `snapshot`: Current market state
    /// - `intent`: The trade intent (identity preserved via reference)
    ///
    /// # Returns
    /// - `Ok(Fill)` if fill succeeded
    /// - `Err(FillRejection)` if fill was rejected
    fn try_fill(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &TSnapshot,
        intent: &TIntent,
    ) -> Result<Fill, FillRejection>;
}
