//! SignalFrame — Canonical normalized signal input for strategies.
//!
//! Phase 20A: The single canonical structure that every strategy consumes,
//! independent of vendor quirks, schema versions, or connector-specific formats.
//!
//! ## Design Principles
//! - Integer-first: All values use mantissa + exponent (NO floats)
//! - Presence-aware: l1_state_bits tracks FieldState per field
//! - Deterministic: Fixed-size types, reproducible across runs
//! - Typed refusals: RefuseReason uses L1Field enum, not strings
//!
//! ## Hard Laws Enforced
//! - L1: No Fabrication — Absent/Null/Malformed fields → Err
//! - L2: Deterministic — Same inputs → identical result
//! - L3: Explicit Refusal — Missing required fields enumerated in Err
//! - L5: Zero Is Valid — Value(0) is accepted and converted

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::events::{
    FieldState, L1_ALL_VALUE, MarketSnapshot, MarketSnapshotV1, build_l1_state_bits,
    get_field_state, l1_slots,
};

// =============================================================================
// L1Field — Typed field identifiers (not stringly)
// =============================================================================

/// L1 market data fields (typed, not stringly).
///
/// Maps directly to l1_slots indices in events.rs.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum L1Field {
    BidPrice = 0,
    AskPrice = 1,
    BidQty = 2,
    AskQty = 3,
}

impl L1Field {
    /// Map to l1_slots slot index (matches events.rs l1_slots constants).
    #[inline]
    pub const fn slot(self) -> u8 {
        self as u8
    }

    /// All L1 fields for iteration.
    pub const ALL: [L1Field; 4] = [
        L1Field::BidPrice,
        L1Field::AskPrice,
        L1Field::BidQty,
        L1Field::AskQty,
    ];
}

impl fmt::Display for L1Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            L1Field::BidPrice => write!(f, "bid_price"),
            L1Field::AskPrice => write!(f, "ask_price"),
            L1Field::BidQty => write!(f, "bid_qty"),
            L1Field::AskQty => write!(f, "ask_qty"),
        }
    }
}

// =============================================================================
// RefuseReason — Typed refusal reasons
// =============================================================================

/// Reason for refusing to create a SignalFrame.
///
/// Uses typed field identifiers (L1Field), not strings.
/// This ensures pattern matching is exhaustive and digests are stable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefuseReason {
    /// Required field is absent (vendor didn't send).
    FieldAbsent(L1Field),

    /// Required field is null (vendor sent explicit null).
    FieldNull(L1Field),

    /// Required field is malformed (vendor sent unparseable data).
    FieldMalformed(L1Field),

    /// Exponent mismatch with expected value.
    ExponentMismatch {
        kind: ExponentKind,
        expected: i8,
        actual: i8,
    },

    /// Invariant violation (e.g., bid > ask).
    InvariantViolation(Invariant),
}

impl fmt::Display for RefuseReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefuseReason::FieldAbsent(field) => write!(f, "field '{}' absent", field),
            RefuseReason::FieldNull(field) => write!(f, "field '{}' null", field),
            RefuseReason::FieldMalformed(field) => write!(f, "field '{}' malformed", field),
            RefuseReason::ExponentMismatch {
                kind,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{} exponent mismatch: expected {}, got {}",
                    kind, expected, actual
                )
            }
            RefuseReason::InvariantViolation(inv) => write!(f, "invariant violated: {}", inv),
        }
    }
}

/// Exponent kind for mismatch errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExponentKind {
    Price,
    Qty,
}

impl fmt::Display for ExponentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExponentKind::Price => write!(f, "price"),
            ExponentKind::Qty => write!(f, "qty"),
        }
    }
}

/// Invariant violation details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Invariant {
    /// bid_price > ask_price (crossed book).
    BidExceedsAsk { bid_m: i64, ask_m: i64 },
}

impl fmt::Display for Invariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Invariant::BidExceedsAsk { bid_m, ask_m } => {
                write!(f, "bid {} > ask {} (crossed book)", bid_m, ask_m)
            }
        }
    }
}

// =============================================================================
// CorrelationId — Fixed-size bytes, not String
// =============================================================================

/// Fixed-size correlation ID (16 bytes, same layout as UUID).
///
/// Use raw bytes internally for determinism and performance.
/// Serialize as hex in JSON for human readability.
pub type CorrelationId = [u8; 16];

// =============================================================================
// SignalFrame — The canonical structure
// =============================================================================

/// Canonical signal frame consumed by all strategies.
///
/// ## Design Principles
/// - Integer-first: All values use mantissa + exponent (NO floats)
/// - Presence-aware: l1_state_bits tracks FieldState per field
/// - Deterministic: Fixed-size types, no heap allocation in hot path (except symbol)
/// - Dual timestamps: event_ts_ns (WAL event) vs book_ts_ns (exchange)
///
/// ## Timestamp Semantics
/// - `event_ts_ns`: The event time under which this frame is processed (WAL event ts)
/// - `book_ts_ns`: The exchange-provided timestamp for the book update
///
/// These may be equal now, but keeping both supports future latency models + replay parity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalFrame {
    // === Identity ===
    /// Correlation ID for tracing (16 bytes, same as UUID layout).
    pub correlation_id: CorrelationId,

    /// Trading symbol (e.g., "BTCUSDT").
    pub symbol: String,

    // === L1 Price Data (integer-first) ===
    /// Best bid price mantissa.
    pub bid_px_m: i64,

    /// Best ask price mantissa.
    pub ask_px_m: i64,

    /// Price exponent (shared for bid/ask).
    pub px_exp: i8,

    // === L1 Quantity Data ===
    /// Best bid quantity mantissa.
    pub bid_qty_m: i64,

    /// Best ask quantity mantissa.
    pub ask_qty_m: i64,

    /// Quantity exponent (shared for bid/ask qty).
    pub qty_exp: i8,

    // === Presence Tracking ===
    /// Packed 2-bit field states for L1 fields.
    /// Same encoding as MarketSnapshotV2:
    /// bits 0-1: bid_price, 2-3: ask_price, 4-5: bid_qty, 6-7: ask_qty.
    pub l1_state_bits: u16,

    // === Derived (computed, not from vendor) ===
    /// Spread in basis points mantissa (exponent = -2).
    /// Only meaningful when both bid_price and ask_price are Value.
    /// Computed using integer arithmetic (truncation toward zero).
    pub spread_bps_m: Option<i64>,

    // === Timestamps ===
    /// Event timestamp in nanoseconds since epoch.
    /// This is the WAL event time under which this frame is processed.
    pub event_ts_ns: i64,

    /// Book timestamp in nanoseconds since epoch.
    /// This is the exchange-provided timestamp for the book update.
    pub book_ts_ns: i64,
}

impl SignalFrame {
    /// Get field state for a given L1 field.
    #[inline]
    pub fn field_state(&self, field: L1Field) -> FieldState {
        get_field_state(self.l1_state_bits, field.slot())
    }

    /// Check if all prices are present (Value state).
    #[inline]
    pub fn has_valid_prices(&self) -> bool {
        self.field_state(L1Field::BidPrice) == FieldState::Value
            && self.field_state(L1Field::AskPrice) == FieldState::Value
    }

    /// Check if all quantities are present (Value state).
    #[inline]
    pub fn has_valid_quantities(&self) -> bool {
        self.field_state(L1Field::BidQty) == FieldState::Value
            && self.field_state(L1Field::AskQty) == FieldState::Value
    }

    /// Check if all L1 fields are present.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.l1_state_bits == L1_ALL_VALUE
    }
}

// =============================================================================
// RequiredL1 — Internal requirements (not public API)
// =============================================================================

/// Internal: L1 field requirements derived from manifest.
///
/// NOT exposed as public API — strategies don't construct this ad-hoc.
/// In Phase 20B, this will be derived from SignalSpec in the manifest.
#[derive(Debug, Clone, Default)]
pub struct RequiredL1 {
    /// Which fields are required, indexed by L1Field as u8.
    pub fields: [bool; 4],
    /// Expected price exponent (None = accept any).
    pub expected_px_exp: Option<i8>,
    /// Expected quantity exponent (None = accept any).
    pub expected_qty_exp: Option<i8>,
    /// Enforce bid <= ask invariant.
    pub enforce_bid_le_ask: bool,
}

impl RequiredL1 {
    /// All L1 fields required (common case).
    pub fn all() -> Self {
        Self {
            fields: [true, true, true, true],
            expected_px_exp: None,
            expected_qty_exp: None,
            enforce_bid_le_ask: true,
        }
    }

    /// Prices only (for spread signal).
    pub fn prices_only() -> Self {
        Self {
            fields: [true, true, false, false],
            expected_px_exp: None,
            expected_qty_exp: None,
            enforce_bid_le_ask: true,
        }
    }

    /// Check if field is required.
    #[inline]
    pub fn requires(&self, field: L1Field) -> bool {
        self.fields[field as usize]
    }
}

// =============================================================================
// signal_frame_from_market — The conversion function
// =============================================================================

/// Convert MarketSnapshot to SignalFrame with requirement validation.
///
/// ## Hard Laws Enforced
/// - L1: No Fabrication — Absent/Null/Malformed fields → Err
/// - L2: Deterministic — Same inputs → identical result
/// - L3: Explicit Refusal — Missing required fields enumerated in Err
/// - L5: Zero Is Valid — Value(0) is accepted
///
/// ## Version Handling
/// - V1: Prices valid if > 0; qty always Value (V1 struct always has qty fields)
/// - V2: Uses explicit l1_state_bits
///
/// ## Arguments
/// - `snapshot`: The market snapshot to convert
/// - `correlation_id`: 16-byte correlation ID for tracing
/// - `symbol`: Trading symbol (e.g., "BTCUSDT")
/// - `event_ts_ns`: WAL event timestamp (caller provides explicitly)
/// - `required`: L1 field requirements
pub fn signal_frame_from_market(
    snapshot: &MarketSnapshot,
    correlation_id: CorrelationId,
    symbol: &str,
    event_ts_ns: i64,
    required: &RequiredL1,
) -> Result<SignalFrame, Vec<RefuseReason>> {
    let mut errors = Vec::new();

    // Extract field states based on version
    let l1_bits = match snapshot {
        MarketSnapshot::V1(v1) => infer_l1_bits_v1(v1),
        MarketSnapshot::V2(v2) => v2.l1_state_bits,
    };

    // Check required fields
    for field in L1Field::ALL {
        if required.requires(field) {
            let state = get_field_state(l1_bits, field.slot());
            match state {
                FieldState::Value => {} // OK
                FieldState::Absent => errors.push(RefuseReason::FieldAbsent(field)),
                FieldState::Null => errors.push(RefuseReason::FieldNull(field)),
                FieldState::Malformed => errors.push(RefuseReason::FieldMalformed(field)),
            }
        }
    }

    // Check exponents if specified
    if let Some(expected) = required.expected_px_exp {
        let actual = snapshot.price_exponent();
        if actual != expected {
            errors.push(RefuseReason::ExponentMismatch {
                kind: ExponentKind::Price,
                expected,
                actual,
            });
        }
    }

    if let Some(expected) = required.expected_qty_exp {
        let actual = snapshot.qty_exponent();
        if actual != expected {
            errors.push(RefuseReason::ExponentMismatch {
                kind: ExponentKind::Qty,
                expected,
                actual,
            });
        }
    }

    // Check bid <= ask invariant (only if both prices are Value)
    let bid_state = get_field_state(l1_bits, l1_slots::BID_PRICE);
    let ask_state = get_field_state(l1_bits, l1_slots::ASK_PRICE);

    if required.enforce_bid_le_ask
        && bid_state == FieldState::Value
        && ask_state == FieldState::Value
    {
        let bid_m = snapshot.bid_price_mantissa();
        let ask_m = snapshot.ask_price_mantissa();
        if bid_m > ask_m {
            errors.push(RefuseReason::InvariantViolation(Invariant::BidExceedsAsk {
                bid_m,
                ask_m,
            }));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Compute spread if both prices present (integer arithmetic with i128 for overflow safety)
    let spread_bps_m = if bid_state == FieldState::Value && ask_state == FieldState::Value {
        Some(compute_spread_bps_integer(
            snapshot.bid_price_mantissa(),
            snapshot.ask_price_mantissa(),
        ))
    } else {
        None
    };

    Ok(SignalFrame {
        correlation_id,
        symbol: symbol.to_string(),
        bid_px_m: snapshot.bid_price_mantissa(),
        ask_px_m: snapshot.ask_price_mantissa(),
        px_exp: snapshot.price_exponent(),
        bid_qty_m: snapshot.bid_qty_mantissa(),
        ask_qty_m: snapshot.ask_qty_mantissa(),
        qty_exp: snapshot.qty_exponent(),
        l1_state_bits: l1_bits,
        spread_bps_m,
        event_ts_ns,
        book_ts_ns: snapshot.book_ts_ns(),
    })
}

/// Infer l1_state_bits for V1 snapshots (legacy semantics).
///
/// V1 semantics:
/// - Prices: must be > 0 to be valid; 0 or negative → Malformed
/// - Quantities: V1 struct always has qty fields, so they're always Value (L5: zero is valid)
///
/// This is safe because MarketSnapshotV1 struct explicitly has bid_qty_mantissa
/// and ask_qty_mantissa fields — they're never omitted.
fn infer_l1_bits_v1(v1: &MarketSnapshotV1) -> u16 {
    let bid_px = if v1.bid_price_mantissa > 0 {
        FieldState::Value
    } else {
        FieldState::Malformed
    };
    let ask_px = if v1.ask_price_mantissa > 0 {
        FieldState::Value
    } else {
        FieldState::Malformed
    };
    // V1 struct always has qty fields (never omitted), and 0 is valid (L5).
    // The i64 type cannot represent NaN; negative qty is technically possible
    // but we treat it as Value and let higher-level invariants catch it if needed.
    build_l1_state_bits(bid_px, ask_px, FieldState::Value, FieldState::Value)
}

/// Compute spread in basis points using pure integer arithmetic.
///
/// Formula: spread_bps = (ask - bid) / mid * 10000
///
/// Implementation uses i128 internally to avoid overflow when prices are large.
/// Rounding: truncation toward zero (deterministic).
/// Returns mantissa with exponent -2 (so 1000 = 10.00 bps).
fn compute_spread_bps_integer(bid_m: i64, ask_m: i64) -> i64 {
    let mid_2x = (bid_m as i128) + (ask_m as i128); // 2 * mid to avoid division rounding
    if mid_2x == 0 {
        return 0;
    }
    // (ask - bid) / mid * 10000, but using 2*mid:
    // = (ask - bid) * 10000 * 2 / (2 * mid)
    // = (ask - bid) * 20000 / mid_2x
    // Then multiply by 100 for exponent -2 (total multiplier = 2_000_000)
    let spread_wide = ((ask_m as i128) - (bid_m as i128)) * 2_000_000;
    (spread_wide / mid_2x) as i64
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{FieldState, MarketSnapshot, MarketSnapshotV1, build_l1_state_bits};

    fn test_correlation_id() -> CorrelationId {
        [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ]
    }

    const TEST_EVENT_TS: i64 = 1_706_443_200_000_000_000;

    #[test]
    fn test_signal_frame_from_v2_all_present() {
        let snap =
            MarketSnapshot::v2_all_present(10000, 10010, 500, 600, -2, -8, 100, TEST_EVENT_TS);

        let frame = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        )
        .unwrap();

        assert_eq!(frame.bid_px_m, 10000);
        assert_eq!(frame.ask_px_m, 10010);
        assert_eq!(frame.bid_qty_m, 500);
        assert_eq!(frame.ask_qty_m, 600);
        assert_eq!(frame.px_exp, -2);
        assert_eq!(frame.qty_exp, -8);
        assert!(frame.is_complete());
        assert!(frame.spread_bps_m.is_some());
        assert_eq!(frame.event_ts_ns, TEST_EVENT_TS);
        assert_eq!(frame.book_ts_ns, TEST_EVENT_TS);
    }

    #[test]
    fn test_signal_frame_refuses_absent_required() {
        let bits = build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent, // bid_qty absent
            FieldState::Value,
        );
        let snap =
            MarketSnapshot::v2_with_states(10000, 10010, 0, 600, -2, -8, 100, TEST_EVENT_TS, bits);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::FieldAbsent(L1Field::BidQty)))
        );
    }

    #[test]
    fn test_signal_frame_refuses_null_required() {
        let bits = build_l1_state_bits(
            FieldState::Value,
            FieldState::Null, // ask_price null
            FieldState::Value,
            FieldState::Value,
        );
        let snap =
            MarketSnapshot::v2_with_states(10000, 0, 500, 600, -2, -8, 100, TEST_EVENT_TS, bits);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::FieldNull(L1Field::AskPrice)))
        );
    }

    #[test]
    fn test_signal_frame_admits_zero_value() {
        // L5: zero is valid if state is Value
        let snap = MarketSnapshot::v2_all_present(10000, 10010, 0, 0, -2, -8, 100, TEST_EVENT_TS);

        let frame = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        )
        .unwrap();

        assert_eq!(frame.bid_qty_m, 0);
        assert_eq!(frame.ask_qty_m, 0);
        assert_eq!(frame.field_state(L1Field::BidQty), FieldState::Value);
        assert_eq!(frame.field_state(L1Field::AskQty), FieldState::Value);
    }

    #[test]
    fn test_signal_frame_refuses_crossed_book() {
        // bid > ask should refuse
        let snap =
            MarketSnapshot::v2_all_present(10010, 10000, 500, 600, -2, -8, 100, TEST_EVENT_TS);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::InvariantViolation(_)))
        );
    }

    #[test]
    fn test_signal_frame_exponent_mismatch() {
        let snap =
            MarketSnapshot::v2_all_present(10000, 10010, 500, 600, -3, -8, 100, TEST_EVENT_TS);

        let mut required = RequiredL1::all();
        required.expected_px_exp = Some(-2);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &required,
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            RefuseReason::ExponentMismatch {
                kind: ExponentKind::Price,
                expected: -2,
                actual: -3
            }
        )));
    }

    #[test]
    fn test_spread_bps_integer_arithmetic() {
        // bid=100.00, ask=100.10 (exponent -2)
        // spread = 0.10 / 100.05 * 10000 = 9.995 bps
        // With exponent -2: ~1000 (meaning 10.00 bps)
        let spread = compute_spread_bps_integer(10000, 10010);
        // (10010 - 10000) * 2_000_000 / 20010 = 10 * 2_000_000 / 20010 ≈ 999
        assert!(
            (990..=1010).contains(&spread),
            "spread was {} (expected ~999)",
            spread
        );
    }

    #[test]
    fn test_spread_bps_large_prices_no_overflow() {
        // Test with large prices that would overflow i64 without i128
        // Use mantissa values large enough that (ask-bid)*2_000_000 would overflow i64
        // i64::MAX ≈ 9.2e18, so we need prices where spread*2_000_000 > 9.2e18
        // That means spread > 4.6e12. With 10 bps spread: bid > 4.6e15
        let bid_m: i64 = 5_000_000_000_000_000; // 5e15
        let ask_m: i64 = 5_005_000_000_000_000; // 5.005e15 (10 bps = 0.1% spread)

        // (ask - bid) = 5e12
        // (ask - bid) * 2_000_000 = 1e19 > i64::MAX (would overflow without i128)

        let spread = compute_spread_bps_integer(bid_m, ask_m);

        // Should be approximately 1000 (10.00 bps with exponent -2)
        // spread_bps = (5e12 / 5.0025e15) * 10000 = 9.995 bps ≈ 1000
        assert!(
            (990..=1010).contains(&spread),
            "spread was {} (expected ~1000)",
            spread
        );
    }

    #[test]
    fn test_spread_bps_zero_mid() {
        // Edge case: mid = 0 should return 0, not panic
        let spread = compute_spread_bps_integer(0, 0);
        assert_eq!(spread, 0);
    }

    #[test]
    fn test_prices_only_requirement() {
        let bits = build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent,
            FieldState::Absent,
        );
        let snap =
            MarketSnapshot::v2_with_states(10000, 10010, 0, 0, -2, -8, 100, TEST_EVENT_TS, bits);

        // prices_only should succeed even without qty
        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::prices_only(),
        );

        assert!(result.is_ok());
        let frame = result.unwrap();
        assert!(frame.has_valid_prices());
        assert!(!frame.has_valid_quantities());
    }

    #[test]
    fn test_v1_legacy_semantics() {
        let v1 = MarketSnapshotV1 {
            bid_price_mantissa: 10000,
            ask_price_mantissa: 10010,
            bid_qty_mantissa: 500,
            ask_qty_mantissa: 600,
            price_exponent: -2,
            qty_exponent: -8,
            spread_bps_mantissa: 100,
            book_ts_ns: TEST_EVENT_TS,
        };
        let snap = MarketSnapshot::V1(v1);

        let frame = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        )
        .unwrap();

        // V1 should work with valid prices
        assert_eq!(frame.bid_px_m, 10000);
        assert!(frame.has_valid_quantities()); // V1 qty always Value
    }

    #[test]
    fn test_v1_rejects_zero_price() {
        // V1: price=0 is Malformed (not Value)
        let v1 = MarketSnapshotV1 {
            bid_price_mantissa: 0, // Invalid
            ask_price_mantissa: 10010,
            bid_qty_mantissa: 500,
            ask_qty_mantissa: 600,
            price_exponent: -2,
            qty_exponent: -8,
            spread_bps_mantissa: 100,
            book_ts_ns: TEST_EVENT_TS,
        };
        let snap = MarketSnapshot::V1(v1);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::FieldMalformed(L1Field::BidPrice)))
        );
    }

    #[test]
    fn test_v1_negative_price_is_malformed() {
        // V1: negative price is Malformed
        let v1 = MarketSnapshotV1 {
            bid_price_mantissa: -100,
            ask_price_mantissa: 10010,
            bid_qty_mantissa: 500,
            ask_qty_mantissa: 600,
            price_exponent: -2,
            qty_exponent: -8,
            spread_bps_mantissa: 100,
            book_ts_ns: TEST_EVENT_TS,
        };
        let snap = MarketSnapshot::V1(v1);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::FieldMalformed(L1Field::BidPrice)))
        );
    }

    #[test]
    fn test_dual_timestamps() {
        // Verify both timestamps are correctly set
        let book_ts = 1_706_443_200_000_000_000;
        let event_ts = 1_706_443_200_100_000_000; // 100ms later

        let snap = MarketSnapshot::v2_all_present(10000, 10010, 500, 600, -2, -8, 100, book_ts);

        let frame = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            event_ts,
            &RequiredL1::all(),
        )
        .unwrap();

        assert_eq!(frame.event_ts_ns, event_ts);
        assert_eq!(frame.book_ts_ns, book_ts);
        assert_ne!(frame.event_ts_ns, frame.book_ts_ns);
    }

    #[test]
    fn test_determinism_100_runs() {
        // Prove deterministic behavior: 100 identical runs produce identical results
        let snap =
            MarketSnapshot::v2_all_present(10000, 10010, 500, 600, -2, -8, 100, TEST_EVENT_TS);

        let mut frames = Vec::with_capacity(100);
        for _ in 0..100 {
            let frame = signal_frame_from_market(
                &snap,
                test_correlation_id(),
                "BTCUSDT",
                TEST_EVENT_TS,
                &RequiredL1::all(),
            )
            .unwrap();
            frames.push(frame);
        }

        // All frames must be identical
        let first = &frames[0];
        for (i, frame) in frames.iter().enumerate().skip(1) {
            assert_eq!(frame.bid_px_m, first.bid_px_m, "Run {} bid_px_m differs", i);
            assert_eq!(frame.ask_px_m, first.ask_px_m, "Run {} ask_px_m differs", i);
            assert_eq!(
                frame.bid_qty_m, first.bid_qty_m,
                "Run {} bid_qty_m differs",
                i
            );
            assert_eq!(
                frame.ask_qty_m, first.ask_qty_m,
                "Run {} ask_qty_m differs",
                i
            );
            assert_eq!(
                frame.spread_bps_m, first.spread_bps_m,
                "Run {} spread_bps_m differs",
                i
            );
            assert_eq!(
                frame.l1_state_bits, first.l1_state_bits,
                "Run {} l1_state_bits differs",
                i
            );
            assert_eq!(
                frame.event_ts_ns, first.event_ts_ns,
                "Run {} event_ts_ns differs",
                i
            );
            assert_eq!(
                frame.book_ts_ns, first.book_ts_ns,
                "Run {} book_ts_ns differs",
                i
            );
        }
    }

    #[test]
    fn test_l1_field_slot_matches_l1_slots() {
        // Verify L1Field::slot() matches l1_slots constants
        assert_eq!(L1Field::BidPrice.slot(), l1_slots::BID_PRICE);
        assert_eq!(L1Field::AskPrice.slot(), l1_slots::ASK_PRICE);
        assert_eq!(L1Field::BidQty.slot(), l1_slots::BID_QTY);
        assert_eq!(L1Field::AskQty.slot(), l1_slots::ASK_QTY);
    }

    #[test]
    fn test_refuse_reason_display() {
        // Verify Display impl produces readable messages
        let r1 = RefuseReason::FieldAbsent(L1Field::BidQty);
        assert_eq!(r1.to_string(), "field 'bid_qty' absent");

        let r2 = RefuseReason::ExponentMismatch {
            kind: ExponentKind::Price,
            expected: -2,
            actual: -3,
        };
        assert!(r2.to_string().contains("price exponent mismatch"));

        let r3 = RefuseReason::InvariantViolation(Invariant::BidExceedsAsk {
            bid_m: 100,
            ask_m: 99,
        });
        assert!(r3.to_string().contains("crossed book"));
    }

    #[test]
    fn test_multiple_errors_collected() {
        // Multiple problems should all be reported
        let bits = build_l1_state_bits(
            FieldState::Absent,    // bid_price absent
            FieldState::Malformed, // ask_price malformed
            FieldState::Value,
            FieldState::Value,
        );
        let snap = MarketSnapshot::v2_with_states(0, 0, 500, 600, -2, -8, 100, TEST_EVENT_TS, bits);

        let result = signal_frame_from_market(
            &snap,
            test_correlation_id(),
            "BTCUSDT",
            TEST_EVENT_TS,
            &RequiredL1::all(),
        );

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::FieldAbsent(L1Field::BidPrice)))
        );
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, RefuseReason::FieldMalformed(L1Field::AskPrice)))
        );
    }
}
