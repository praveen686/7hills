//! India F&O Fill Model
//!
//! Implements `PaperFillModel<OptionsSnapshot, TradeIntent>` for India F&O paper trading.
//!
//! ## Fill Rules
//!
//! 1. **Executable pricing only**
//!    - Buy fills at ask
//!    - Sell fills at bid
//!    - Never fill at LTP
//!
//! 2. **Loss awareness**
//!    - Reject if bid/ask missing
//!    - Reject if quote stale (age_ms > threshold)
//!    - Reject if insufficient quantity at top of book
//!
//! 3. **Deterministic costs**
//!    - All fees computed from fill price/qty and config
//!    - No runtime lookups or random factors
//!
//! ## Phase 1 Quantity Model
//!
//! - Top-of-book constrained: fill_qty = min(intent.qty, top_level.qty)
//! - Reject if top level quantity is 0 or missing

use chrono::{DateTime, Utc};
use tracing::debug;

use quantlaxmi_paper::{Fees, Fill, FillRejection, FillSide, PaperFillModel};

use super::fees_india::{
    FeesBreakdown, FillRejected, IndiaFnoFeeCalculator, IndiaFnoFeeConfig, Side,
};
use super::snapshot::OptionsSnapshot;
use super::strategy_adapter::TradeIntent;

// =============================================================================
// INDIA FILL RESULT
// =============================================================================

/// Extended fill result with itemized fees.
#[derive(Debug, Clone)]
pub struct IndiaFill {
    /// Base fill information
    pub fill: Fill,
    /// Itemized fee breakdown
    pub fees_breakdown: FeesBreakdown,
    /// Quote age at fill time (for audit)
    pub quote_age_ms: u32,
    /// Spread at fill time (for audit)
    pub spread_paise: i64,
}

// =============================================================================
// INDIA FILL MODEL
// =============================================================================

/// India F&O paper fill model.
///
/// Implements executable pricing with Budget 2026 fee rates.
pub struct IndiaFnoFillModel {
    /// Fee calculator
    calculator: IndiaFnoFeeCalculator,
    /// Fill counter (for logging)
    fill_count: u64,
    /// Rejection counter (for logging)
    reject_count: u64,
}

impl IndiaFnoFillModel {
    /// Create a new fill model with default Budget 2026 config.
    pub fn new() -> Self {
        Self {
            calculator: IndiaFnoFeeCalculator::default_budget_2026(),
            fill_count: 0,
            reject_count: 0,
        }
    }

    /// Create with custom fee config.
    pub fn with_config(config: IndiaFnoFeeConfig) -> Self {
        Self {
            calculator: IndiaFnoFeeCalculator::new(config),
            fill_count: 0,
            reject_count: 0,
        }
    }

    /// Get fill count.
    pub fn fill_count(&self) -> u64 {
        self.fill_count
    }

    /// Get rejection count.
    pub fn reject_count(&self) -> u64 {
        self.reject_count
    }

    /// Get fee config reference.
    pub fn config(&self) -> &IndiaFnoFeeConfig {
        self.calculator.config()
    }

    /// Try to fill an intent, returning extended fill or rejection.
    pub fn try_fill_extended(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &OptionsSnapshot,
        intent: &TradeIntent,
    ) -> Result<IndiaFill, FillRejected> {
        // 1. Find the quote for this instrument
        let quote = snapshot.get_quote(intent.instrument_token).ok_or_else(|| {
            FillRejected::NoExecutableQuote {
                reason: format!("No quote for token {}", intent.instrument_token),
            }
        })?;

        // 2. Check staleness
        let threshold = self.calculator.config().staleness_threshold_ms;
        if quote.age_ms > threshold {
            self.reject_count += 1;
            return Err(FillRejected::StaleQuote {
                age_ms: quote.age_ms,
                threshold_ms: threshold,
            });
        }

        // 3. Get executable price (bid for sell, ask for buy)
        let (price_level, fill_side) = match intent.side {
            Side::Buy => {
                let ask = quote.ask.ok_or_else(|| {
                    self.reject_count += 1;
                    FillRejected::NoExecutableQuote {
                        reason: format!("No ask for {}", quote.tradingsymbol),
                    }
                })?;
                if !ask.is_valid() {
                    self.reject_count += 1;
                    return Err(FillRejected::NoExecutableQuote {
                        reason: format!("Invalid ask for {}", quote.tradingsymbol),
                    });
                }
                (ask, FillSide::Buy)
            }
            Side::Sell => {
                let bid = quote.bid.ok_or_else(|| {
                    self.reject_count += 1;
                    FillRejected::NoExecutableQuote {
                        reason: format!("No bid for {}", quote.tradingsymbol),
                    }
                })?;
                if !bid.is_valid() {
                    self.reject_count += 1;
                    return Err(FillRejected::NoExecutableQuote {
                        reason: format!("Invalid bid for {}", quote.tradingsymbol),
                    });
                }
                (bid, FillSide::Sell)
            }
        };

        // 4. Check quantity (top-of-book constrained)
        let requested_qty = intent.qty as i64;
        let available_qty = price_level.qty as i64;
        if available_qty == 0 {
            self.reject_count += 1;
            return Err(FillRejected::InsufficientQuantity {
                requested: requested_qty,
                available: 0,
            });
        }

        // Fill quantity = min(requested, available)
        let fill_qty = requested_qty.min(available_qty);

        // 5. Calculate spread for audit
        let spread_paise = match (quote.bid, quote.ask) {
            (Some(b), Some(a)) if b.is_valid() && a.is_valid() => {
                ((a.price - b.price) * 100.0) as i64
            }
            _ => 0,
        };

        // 6. Calculate fees
        let price_paise = (price_level.price * 100.0) as i64;
        let fees_breakdown = self.calculator.calculate(
            intent.kind,
            intent.side,
            price_paise,
            fill_qty,
            intent.lot_size as i64,
            spread_paise,
        );

        // 7. Build fill
        let fill = Fill {
            ts,
            symbol: quote.tradingsymbol.clone(),
            side: fill_side,
            qty: fill_qty as f64,
            price: price_level.price,
            fees: Fees {
                total: fees_breakdown.total_rupees(),
            },
        };

        self.fill_count += 1;

        debug!(
            symbol = %fill.symbol,
            side = ?fill.side,
            qty = fill.qty,
            price = fill.price,
            fees = fees_breakdown.total_rupees(),
            "[FILL] Executed"
        );

        Ok(IndiaFill {
            fill,
            fees_breakdown,
            quote_age_ms: quote.age_ms,
            spread_paise,
        })
    }
}

impl Default for IndiaFnoFillModel {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PAPER FILL MODEL TRAIT IMPLEMENTATION
// =============================================================================

impl PaperFillModel<OptionsSnapshot, TradeIntent> for IndiaFnoFillModel {
    fn try_fill(
        &mut self,
        ts: DateTime<Utc>,
        snapshot: &OptionsSnapshot,
        intent: &TradeIntent,
    ) -> Result<Fill, FillRejection> {
        match self.try_fill_extended(ts, snapshot, intent) {
            Ok(india_fill) => Ok(india_fill.fill),
            Err(e) => {
                // Map India-specific rejection to generic FillRejection
                let rejection = match e {
                    FillRejected::NoExecutableQuote { reason } => {
                        FillRejection::NoExecutableQuote { reason }
                    }
                    FillRejected::StaleQuote {
                        age_ms,
                        threshold_ms,
                    } => FillRejection::StaleQuote {
                        age_ms,
                        threshold_ms,
                    },
                    FillRejected::InsufficientQuantity {
                        requested,
                        available,
                    } => FillRejection::InsufficientQuantity {
                        requested,
                        available,
                    },
                };
                Err(rejection)
            }
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::fees_india::InstrumentKind;
    use crate::paper::snapshot::{OptQuote, PriceQty, Right};
    use crate::paper::strategy_adapter::IntentTag;

    fn make_snapshot_with_quote(bid: Option<PriceQty>, ask: Option<PriceQty>) -> OptionsSnapshot {
        let mut snapshot = OptionsSnapshot::new("NIFTY".into(), "2025-10-02".into());

        let mut quote = OptQuote::new(123456, "NIFTY2510223400CE".into(), 23400, Right::Call);
        quote.bid = bid;
        quote.ask = ask;
        quote.age_ms = 100; // Fresh

        snapshot.quotes.push(quote);
        snapshot
    }

    fn make_intent(side: Side) -> TradeIntent {
        TradeIntent::new(
            123456,
            side,
            1,
            IntentTag::Manual,
            InstrumentKind::IndexOption,
            25,
        )
    }

    #[test]
    fn test_buy_fills_at_ask() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),  // bid
            Some(PriceQty::new(100.0, 100)), // ask
        );

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_ok());

        let fill = result.unwrap();
        assert_eq!(fill.fill.price, 100.0, "Buy should fill at ask price");
        assert_eq!(fill.fill.side, FillSide::Buy);
    }

    #[test]
    fn test_sell_fills_at_bid() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),  // bid
            Some(PriceQty::new(100.0, 100)), // ask
        );

        let intent = make_intent(Side::Sell);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_ok());

        let fill = result.unwrap();
        assert_eq!(fill.fill.price, 99.0, "Sell should fill at bid price");
        assert_eq!(fill.fill.side, FillSide::Sell);
    }

    #[test]
    fn test_reject_missing_ask_on_buy() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)), // bid only
            None,                           // no ask
        );

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_err());

        match result.unwrap_err() {
            FillRejected::NoExecutableQuote { .. } => {}
            _ => panic!("Expected NoExecutableQuote"),
        }
    }

    #[test]
    fn test_reject_missing_bid_on_sell() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            None,                            // no bid
            Some(PriceQty::new(100.0, 100)), // ask only
        );

        let intent = make_intent(Side::Sell);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_err());

        match result.unwrap_err() {
            FillRejected::NoExecutableQuote { .. } => {}
            _ => panic!("Expected NoExecutableQuote"),
        }
    }

    #[test]
    fn test_reject_stale_quote() {
        let mut model = IndiaFnoFillModel::new();

        let mut snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),
            Some(PriceQty::new(100.0, 100)),
        );

        // Make quote stale
        snapshot.quotes[0].age_ms = 10000; // 10 seconds, above 5s threshold

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_err());

        match result.unwrap_err() {
            FillRejected::StaleQuote {
                age_ms,
                threshold_ms,
            } => {
                assert_eq!(age_ms, 10000);
                assert_eq!(threshold_ms, 5000);
            }
            _ => panic!("Expected StaleQuote"),
        }
    }

    #[test]
    fn test_reject_zero_quantity() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 0)),  // bid with 0 qty
            Some(PriceQty::new(100.0, 0)), // ask with 0 qty
        );

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_err());

        match result.unwrap_err() {
            FillRejected::NoExecutableQuote { .. } => {}
            _ => panic!("Expected NoExecutableQuote (invalid quote)"),
        }
    }

    #[test]
    fn test_top_of_book_constrained_quantity() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 10)),  // bid with 10 qty
            Some(PriceQty::new(100.0, 10)), // ask with 10 qty
        );

        // Request 100 lots, but only 10 available
        let mut intent = make_intent(Side::Buy);
        intent.qty = 100;

        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_ok());

        let fill = result.unwrap();
        assert_eq!(
            fill.fill.qty, 10.0,
            "Fill qty should be constrained to available"
        );
    }

    #[test]
    fn test_fees_breakdown_included() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),
            Some(PriceQty::new(100.0, 100)),
        );

        let intent = make_intent(Side::Sell);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_ok());

        let fill = result.unwrap();

        // STT should be non-zero on sell
        assert!(fill.fees_breakdown.stt > 0, "STT should be applied on sell");

        // Brokerage should be present
        assert!(
            fill.fees_breakdown.brokerage > 0,
            "Brokerage should be applied"
        );

        // Total should match sum
        assert_eq!(
            fill.fees_breakdown.total,
            fill.fees_breakdown.brokerage
                + fill.fees_breakdown.stt
                + fill.fees_breakdown.exchange_txn
                + fill.fees_breakdown.sebi
                + fill.fees_breakdown.gst
                + fill.fees_breakdown.stamp
                + fill.fees_breakdown.slippage
        );
    }

    #[test]
    fn test_spread_recorded() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),  // bid
            Some(PriceQty::new(101.0, 100)), // ask (spread = ₹2 = 200 paise)
        );

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        let result = model.try_fill_extended(ts, &snapshot, &intent);
        assert!(result.is_ok());

        let fill = result.unwrap();
        assert_eq!(fill.spread_paise, 200, "Spread should be recorded in paise");
    }

    #[test]
    fn test_fill_count_tracking() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),
            Some(PriceQty::new(100.0, 100)),
        );

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        assert_eq!(model.fill_count(), 0);
        assert_eq!(model.reject_count(), 0);

        let _ = model.try_fill_extended(ts, &snapshot, &intent);
        assert_eq!(model.fill_count(), 1);

        // Cause a rejection
        let bad_snapshot = make_snapshot_with_quote(None, None);
        let _ = model.try_fill_extended(ts, &bad_snapshot, &intent);
        assert_eq!(model.reject_count(), 1);
    }

    #[test]
    fn test_paper_fill_model_trait() {
        let mut model = IndiaFnoFillModel::new();

        let snapshot = make_snapshot_with_quote(
            Some(PriceQty::new(99.0, 100)),
            Some(PriceQty::new(100.0, 100)),
        );

        let intent = make_intent(Side::Buy);
        let ts = Utc::now();

        // Use the trait method
        let result = model.try_fill(ts, &snapshot, &intent);
        assert!(result.is_ok());

        let fill = result.unwrap();
        assert_eq!(fill.price, 100.0);
    }
}
