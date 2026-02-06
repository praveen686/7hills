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

// =============================================================================
// FIELD STATE FOR PRESENCE TRACKING (Doctrine: No Silent Poisoning)
// =============================================================================

/// Field state for L1 market data fields.
///
/// Distinguishes between "vendor didn't send" (Absent), "vendor sent null" (Null),
/// "vendor sent a valid value" (Value), and "vendor sent malformed data" (Malformed).
///
/// This is critical for the "No Silent Poisoning" doctrine: a quantity of 0 with
/// state `Value` is a real zero from the vendor, while 0 with state `Absent` means
/// the field was never received.
///
/// ## Encoding
/// 2 bits per field: 00=Absent, 01=Null, 10=Value, 11=Malformed
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldState {
    /// Vendor did not send this field (missing from payload)
    #[default]
    Absent = 0,
    /// Vendor explicitly sent null/None
    Null = 1,
    /// Vendor sent a valid value (mantissa is meaningful)
    Value = 2,
    /// Vendor sent malformed data (could not parse)
    Malformed = 3,
}

/// Slot indices for L1 field state bits.
///
/// Each field uses 2 bits in the packed `l1_state_bits` field.
pub mod l1_slots {
    pub const BID_PRICE: u8 = 0;
    pub const ASK_PRICE: u8 = 1;
    pub const BID_QTY: u8 = 2;
    pub const ASK_QTY: u8 = 3;
    // Slots 4-7 reserved for future L1 fields (16 bits = 8 slots)
}

/// Set a field state in packed bits.
///
/// # Arguments
/// * `bits` - Mutable reference to the packed state bits
/// * `slot` - Field slot index (0-7 for u16)
/// * `state` - The field state to set
#[inline]
pub fn set_field_state(bits: &mut u16, slot: u8, state: FieldState) {
    debug_assert!(slot < 8, "slot must be < 8 for u16");
    let shift = (slot as u16) * 2;
    *bits &= !(0b11 << shift); // Clear existing bits
    *bits |= ((state as u16) & 0b11) << shift; // Set new bits
}

/// Get a field state from packed bits.
///
/// # Arguments
/// * `bits` - The packed state bits
/// * `slot` - Field slot index (0-7 for u16)
///
/// # Returns
/// The field state for the given slot
#[inline]
pub fn get_field_state(bits: u16, slot: u8) -> FieldState {
    debug_assert!(slot < 8, "slot must be < 8 for u16");
    let shift = (slot as u16) * 2;
    match ((bits >> shift) & 0b11) as u8 {
        0 => FieldState::Absent,
        1 => FieldState::Null,
        2 => FieldState::Value,
        _ => FieldState::Malformed,
    }
}

/// Build l1_state_bits from individual field states.
///
/// # Arguments
/// * `bid_price` - State of bid_price_mantissa field
/// * `ask_price` - State of ask_price_mantissa field
/// * `bid_qty` - State of bid_qty_mantissa field
/// * `ask_qty` - State of ask_qty_mantissa field
#[inline]
pub fn build_l1_state_bits(
    bid_price: FieldState,
    ask_price: FieldState,
    bid_qty: FieldState,
    ask_qty: FieldState,
) -> u16 {
    let mut bits: u16 = 0;
    set_field_state(&mut bits, l1_slots::BID_PRICE, bid_price);
    set_field_state(&mut bits, l1_slots::ASK_PRICE, ask_price);
    set_field_state(&mut bits, l1_slots::BID_QTY, bid_qty);
    set_field_state(&mut bits, l1_slots::ASK_QTY, ask_qty);
    bits
}

/// All L1 fields present (common case for valid snapshots).
pub const L1_ALL_VALUE: u16 = 0b10_10_10_10; // Value for all 4 fields

/// Correlation context for distributed tracing.
/// All events carry this context to enable full causality reconstruction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrelationContext {
    /// Capture/trading session identifier
    pub session_id: Option<String>,

    /// Analysis/backtest run identifier
    pub run_id: Option<String>,

    /// Trading symbol (e.g., "BTCUSDT", "NIFTY26JAN24000CE")
    pub symbol: Option<String>,

    /// Exchange/venue identifier (e.g., "binance", "zerodha")
    pub venue: Option<String>,

    /// Strategy that generated/processed the event
    pub strategy_id: Option<String>,

    /// Decision identifier (links to orders)
    pub decision_id: Option<Uuid>,

    /// Order identifier (links to fills)
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
    pub venue: String,

    /// Correlation context for tracing
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

/// Fixed exponent for confidence values (-4 = 4 decimal places, e.g., 10000 = 1.0000)
pub const CONFIDENCE_EXPONENT: i8 = -4;

/// Fixed exponent for spread_bps values (-2 = 2 decimal places, e.g., 523 = 5.23 bps)
pub const SPREAD_BPS_EXPONENT: i8 = -2;

/// Decision event from strategy execution.
///
/// Captures the full decision context including market state at decision time.
/// Used for replay parity validation.
///
/// ## Fixed-Point Policy
/// - `confidence_mantissa`: Fixed exponent -4 (10000 = 1.0, range 0-10000)
/// - `spread_bps_mantissa` in MarketSnapshot: Fixed exponent -2 (523 = 5.23 bps)
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

    /// Confidence score as fixed-point mantissa (exponent = CONFIDENCE_EXPONENT = -4)
    /// Value 10000 = 1.0, 8500 = 0.85, 0 = 0.0
    pub confidence_mantissa: i64,

    /// Strategy-specific metadata
    pub metadata: serde_json::Value,

    /// Correlation context
    pub ctx: CorrelationContext,
}

impl DecisionEvent {
    /// Convert confidence_mantissa to f64 (for display only).
    ///
    /// # Warning
    /// Only use for display/logging. Do NOT use for computations that feed
    /// back into decisions or hashing.
    pub fn confidence_f64(&self) -> f64 {
        self.confidence_mantissa as f64 * 10f64.powi(CONFIDENCE_EXPONENT as i32)
    }

    /// Create confidence_mantissa from f64 value (0.0 to 1.0).
    ///
    /// # Migration Bridge
    /// This helper exists only for ingesting external float inputs (e.g., from
    /// legacy systems or libraries that provide confidence as f64).
    ///
    /// **Internal strategy computations MUST produce confidence directly as
    /// mantissa values using integer arithmetic to ensure determinism.**
    ///
    /// # Example
    /// ```ignore
    /// // GOOD: Direct mantissa computation (deterministic)
    /// let confidence_mantissa = 8500; // 0.85
    ///
    /// // ACCEPTABLE: Converting external float input (migration bridge)
    /// let external_confidence = legacy_api.get_confidence(); // returns f64
    /// let mantissa = DecisionEvent::confidence_from_f64(external_confidence);
    ///
    /// // BAD: Converting internal float computation (nondeterministic)
    /// let bad_confidence = some_float_calculation(); // DON'T DO THIS
    /// let bad_mantissa = DecisionEvent::confidence_from_f64(bad_confidence);
    /// ```
    pub fn confidence_from_f64(value: f64) -> i64 {
        (value * 10f64.powi(-CONFIDENCE_EXPONENT as i32)).round() as i64
    }
}

// =============================================================================
// MARKET SNAPSHOT (Versioned Enum for Schema Evolution)
// =============================================================================

/// Market state snapshot at decision time for replay validation.
///
/// ## Schema Versioning
/// - **V1**: Original schema (no presence tracking, legacy WAL compatible)
/// - **V2**: Adds `l1_state_bits` for explicit field presence tracking
///
/// ## Canonical Bytes
/// Each variant has a version discriminant byte (0x01 for V1, 0x02 for V2)
/// prepended to its canonical encoding. This ensures:
/// - V1 digests are distinct from V2 digests
/// - Replay parity is version-scoped (cross-version comparison is invalid)
///
/// ## Fixed-Point Policy
/// - `spread_bps_mantissa`: Fixed exponent -2 (523 = 5.23 bps)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "schema", rename_all = "snake_case")]
pub enum MarketSnapshot {
    /// Original schema (no presence tracking).
    /// Used for reading legacy WAL entries.
    V1(MarketSnapshotV1),
    /// Schema with explicit L1 field presence tracking.
    /// All new captures MUST use V2.
    V2(MarketSnapshotV2),
}

/// MarketSnapshot V1: Original schema without presence tracking.
///
/// **Warning**: V1 cannot distinguish "vendor sent 0" from "vendor didn't send".
/// Use only for legacy WAL compatibility. New code should use V2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotV1 {
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
    /// Spread in basis points as fixed-point mantissa (exponent = SPREAD_BPS_EXPONENT = -2)
    pub spread_bps_mantissa: i64,
    /// Book timestamp (nanoseconds since epoch for causality)
    pub book_ts_ns: i64,
}

/// MarketSnapshot V2: Schema with explicit L1 field presence tracking.
///
/// ## Doctrine: No Silent Poisoning
/// The `l1_state_bits` field tracks whether each L1 field was:
/// - `Value`: Vendor sent a valid value (mantissa is meaningful)
/// - `Absent`: Vendor did not send this field
/// - `Null`: Vendor explicitly sent null
/// - `Malformed`: Vendor sent unparseable data
///
/// This allows `bid_qty_mantissa = 0` with state `Value` to be distinguished
/// from `bid_qty_mantissa = 0` with state `Absent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotV2 {
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
    /// Spread in basis points as fixed-point mantissa (exponent = SPREAD_BPS_EXPONENT = -2)
    pub spread_bps_mantissa: i64,
    /// Book timestamp (nanoseconds since epoch for causality)
    pub book_ts_ns: i64,
    /// Packed 2-bit field states for L1 fields (prices + qty).
    /// Bits 0-1: bid_price, 2-3: ask_price, 4-5: bid_qty, 6-7: ask_qty.
    /// Use `get_field_state` and `set_field_state` to access.
    pub l1_state_bits: u16,
}

impl MarketSnapshot {
    /// Create a V2 snapshot with all fields present (common case).
    #[allow(clippy::too_many_arguments)]
    pub fn v2_all_present(
        bid_price_mantissa: i64,
        ask_price_mantissa: i64,
        bid_qty_mantissa: i64,
        ask_qty_mantissa: i64,
        price_exponent: i8,
        qty_exponent: i8,
        spread_bps_mantissa: i64,
        book_ts_ns: i64,
    ) -> Self {
        Self::V2(MarketSnapshotV2 {
            bid_price_mantissa,
            ask_price_mantissa,
            bid_qty_mantissa,
            ask_qty_mantissa,
            price_exponent,
            qty_exponent,
            spread_bps_mantissa,
            book_ts_ns,
            l1_state_bits: L1_ALL_VALUE,
        })
    }

    /// Create a V2 snapshot with explicit field states.
    #[allow(clippy::too_many_arguments)]
    pub fn v2_with_states(
        bid_price_mantissa: i64,
        ask_price_mantissa: i64,
        bid_qty_mantissa: i64,
        ask_qty_mantissa: i64,
        price_exponent: i8,
        qty_exponent: i8,
        spread_bps_mantissa: i64,
        book_ts_ns: i64,
        l1_state_bits: u16,
    ) -> Self {
        Self::V2(MarketSnapshotV2 {
            bid_price_mantissa,
            ask_price_mantissa,
            bid_qty_mantissa,
            ask_qty_mantissa,
            price_exponent,
            qty_exponent,
            spread_bps_mantissa,
            book_ts_ns,
            l1_state_bits,
        })
    }

    /// Access bid_price_mantissa (common accessor for both versions).
    #[inline]
    pub fn bid_price_mantissa(&self) -> i64 {
        match self {
            Self::V1(v1) => v1.bid_price_mantissa,
            Self::V2(v2) => v2.bid_price_mantissa,
        }
    }

    /// Access ask_price_mantissa (common accessor for both versions).
    #[inline]
    pub fn ask_price_mantissa(&self) -> i64 {
        match self {
            Self::V1(v1) => v1.ask_price_mantissa,
            Self::V2(v2) => v2.ask_price_mantissa,
        }
    }

    /// Access bid_qty_mantissa (common accessor for both versions).
    #[inline]
    pub fn bid_qty_mantissa(&self) -> i64 {
        match self {
            Self::V1(v1) => v1.bid_qty_mantissa,
            Self::V2(v2) => v2.bid_qty_mantissa,
        }
    }

    /// Access ask_qty_mantissa (common accessor for both versions).
    #[inline]
    pub fn ask_qty_mantissa(&self) -> i64 {
        match self {
            Self::V1(v1) => v1.ask_qty_mantissa,
            Self::V2(v2) => v2.ask_qty_mantissa,
        }
    }

    /// Access price_exponent (common accessor for both versions).
    #[inline]
    pub fn price_exponent(&self) -> i8 {
        match self {
            Self::V1(v1) => v1.price_exponent,
            Self::V2(v2) => v2.price_exponent,
        }
    }

    /// Access qty_exponent (common accessor for both versions).
    #[inline]
    pub fn qty_exponent(&self) -> i8 {
        match self {
            Self::V1(v1) => v1.qty_exponent,
            Self::V2(v2) => v2.qty_exponent,
        }
    }

    /// Access spread_bps_mantissa (common accessor for both versions).
    #[inline]
    pub fn spread_bps_mantissa(&self) -> i64 {
        match self {
            Self::V1(v1) => v1.spread_bps_mantissa,
            Self::V2(v2) => v2.spread_bps_mantissa,
        }
    }

    /// Access book_ts_ns (common accessor for both versions).
    #[inline]
    pub fn book_ts_ns(&self) -> i64 {
        match self {
            Self::V1(v1) => v1.book_ts_ns,
            Self::V2(v2) => v2.book_ts_ns,
        }
    }

    /// Get l1_state_bits (V2 only, returns 0 for V1).
    ///
    /// **Warning**: V1 returns 0 which means "presence unknown" (all Absent).
    /// This is a compatibility fallback, not a valid presence encoding.
    #[inline]
    pub fn l1_state_bits(&self) -> u16 {
        match self {
            Self::V1(_) => 0, // Legacy: presence unknown
            Self::V2(v2) => v2.l1_state_bits,
        }
    }

    /// Check if this is a V2 snapshot (has presence tracking).
    #[inline]
    pub fn is_v2(&self) -> bool {
        matches!(self, Self::V2(_))
    }

    /// Get the schema version discriminant byte for canonical encoding.
    #[inline]
    pub fn schema_version_byte(&self) -> u8 {
        match self {
            Self::V1(_) => 0x01,
            Self::V2(_) => 0x02,
        }
    }

    /// Get field state for bid_price (V2 only, returns Absent for V1).
    #[inline]
    pub fn bid_price_state(&self) -> FieldState {
        get_field_state(self.l1_state_bits(), l1_slots::BID_PRICE)
    }

    /// Get field state for ask_price (V2 only, returns Absent for V1).
    #[inline]
    pub fn ask_price_state(&self) -> FieldState {
        get_field_state(self.l1_state_bits(), l1_slots::ASK_PRICE)
    }

    /// Get field state for bid_qty (V2 only, returns Absent for V1).
    #[inline]
    pub fn bid_qty_state(&self) -> FieldState {
        get_field_state(self.l1_state_bits(), l1_slots::BID_QTY)
    }

    /// Get field state for ask_qty (V2 only, returns Absent for V1).
    #[inline]
    pub fn ask_qty_state(&self) -> FieldState {
        get_field_state(self.l1_state_bits(), l1_slots::ASK_QTY)
    }

    /// Convert spread_bps_mantissa to f64 (for display only).
    ///
    /// # Warning
    /// Only use for display/logging. Do NOT use for computations that feed
    /// back into decisions or hashing.
    pub fn spread_bps_f64(&self) -> f64 {
        self.spread_bps_mantissa() as f64 * 10f64.powi(SPREAD_BPS_EXPONENT as i32)
    }

    /// Create spread_bps_mantissa from f64 value.
    ///
    /// # Migration Bridge
    /// This helper exists only for ingesting external float inputs.
    /// **Internal spread computations MUST use integer arithmetic.**
    pub fn spread_bps_from_f64(value: f64) -> i64 {
        (value * 10f64.powi(-SPREAD_BPS_EXPONENT as i32)).round() as i64
    }
}

impl MarketSnapshotV2 {
    /// Get the V1 fields as a reference (for canonical encoding shared logic).
    pub fn as_v1_fields(&self) -> MarketSnapshotV1 {
        MarketSnapshotV1 {
            bid_price_mantissa: self.bid_price_mantissa,
            ask_price_mantissa: self.ask_price_mantissa,
            bid_qty_mantissa: self.bid_qty_mantissa,
            ask_qty_mantissa: self.ask_qty_mantissa,
            price_exponent: self.price_exponent,
            qty_exponent: self.qty_exponent,
            spread_bps_mantissa: self.spread_bps_mantissa,
            book_ts_ns: self.book_ts_ns,
        }
    }
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

    // =========================================================================
    // FIELD STATE TESTS
    // =========================================================================

    #[test]
    fn test_field_state_packing() {
        let mut bits: u16 = 0;
        set_field_state(&mut bits, l1_slots::BID_PRICE, FieldState::Value);
        set_field_state(&mut bits, l1_slots::ASK_PRICE, FieldState::Value);
        set_field_state(&mut bits, l1_slots::BID_QTY, FieldState::Absent);
        set_field_state(&mut bits, l1_slots::ASK_QTY, FieldState::Null);

        assert_eq!(
            get_field_state(bits, l1_slots::BID_PRICE),
            FieldState::Value
        );
        assert_eq!(
            get_field_state(bits, l1_slots::ASK_PRICE),
            FieldState::Value
        );
        assert_eq!(get_field_state(bits, l1_slots::BID_QTY), FieldState::Absent);
        assert_eq!(get_field_state(bits, l1_slots::ASK_QTY), FieldState::Null);
    }

    #[test]
    fn test_build_l1_state_bits_helper() {
        let bits = build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent,
            FieldState::Malformed,
        );
        assert_eq!(
            get_field_state(bits, l1_slots::BID_PRICE),
            FieldState::Value
        );
        assert_eq!(
            get_field_state(bits, l1_slots::ASK_PRICE),
            FieldState::Value
        );
        assert_eq!(get_field_state(bits, l1_slots::BID_QTY), FieldState::Absent);
        assert_eq!(
            get_field_state(bits, l1_slots::ASK_QTY),
            FieldState::Malformed
        );
    }

    #[test]
    fn test_l1_all_value_constant() {
        // L1_ALL_VALUE should have all 4 fields set to Value (0b10)
        assert_eq!(
            get_field_state(L1_ALL_VALUE, l1_slots::BID_PRICE),
            FieldState::Value
        );
        assert_eq!(
            get_field_state(L1_ALL_VALUE, l1_slots::ASK_PRICE),
            FieldState::Value
        );
        assert_eq!(
            get_field_state(L1_ALL_VALUE, l1_slots::BID_QTY),
            FieldState::Value
        );
        assert_eq!(
            get_field_state(L1_ALL_VALUE, l1_slots::ASK_QTY),
            FieldState::Value
        );
    }

    // =========================================================================
    // MARKET SNAPSHOT V1/V2 TESTS
    // =========================================================================

    #[test]
    fn test_market_snapshot_v2_serde_roundtrip() {
        let snap = MarketSnapshot::v2_all_present(1000, 1001, 500, 600, -2, -8, 10, 1234567890);

        let json = serde_json::to_string(&snap).unwrap();
        let parsed: MarketSnapshot = serde_json::from_str(&json).unwrap();

        assert!(parsed.is_v2());
        assert_eq!(parsed.bid_price_mantissa(), 1000);
        assert_eq!(parsed.ask_price_mantissa(), 1001);
        assert_eq!(parsed.bid_qty_mantissa(), 500);
        assert_eq!(parsed.ask_qty_mantissa(), 600);
        assert_eq!(parsed.l1_state_bits(), L1_ALL_VALUE);
    }

    #[test]
    fn test_market_snapshot_v1_serde_roundtrip() {
        let v1 = MarketSnapshotV1 {
            bid_price_mantissa: 2000,
            ask_price_mantissa: 2001,
            bid_qty_mantissa: 700,
            ask_qty_mantissa: 800,
            price_exponent: -2,
            qty_exponent: -8,
            spread_bps_mantissa: 5,
            book_ts_ns: 9876543210,
        };
        let snap = MarketSnapshot::V1(v1);

        let json = serde_json::to_string(&snap).unwrap();
        let parsed: MarketSnapshot = serde_json::from_str(&json).unwrap();

        assert!(!parsed.is_v2());
        assert_eq!(parsed.bid_price_mantissa(), 2000);
        assert_eq!(parsed.ask_price_mantissa(), 2001);
        // V1 returns 0 for l1_state_bits (presence unknown)
        assert_eq!(parsed.l1_state_bits(), 0);
    }

    #[test]
    fn test_market_snapshot_v2_with_custom_states() {
        // Create V2 with prices present, quantities absent (backtest scenario)
        let bits = build_l1_state_bits(
            FieldState::Value,  // bid_price
            FieldState::Value,  // ask_price
            FieldState::Absent, // bid_qty
            FieldState::Absent, // ask_qty
        );
        let snap = MarketSnapshot::v2_with_states(1000, 1001, 0, 0, -2, -8, 10, 1234567890, bits);

        assert!(snap.is_v2());
        assert_eq!(snap.bid_price_state(), FieldState::Value);
        assert_eq!(snap.ask_price_state(), FieldState::Value);
        assert_eq!(snap.bid_qty_state(), FieldState::Absent);
        assert_eq!(snap.ask_qty_state(), FieldState::Absent);
    }

    #[test]
    fn test_presence_correctness_zero_value_vs_absent() {
        // Doctrine test: qty=0 with state Value is different from qty=0 with state Absent

        // Case 1: qty=0 and state=Value (vendor sent 0)
        let snap_value = MarketSnapshot::v2_with_states(
            1000,
            1001,
            0,
            0, // zero quantities
            -2,
            -8,
            10,
            1234567890,
            build_l1_state_bits(
                FieldState::Value,
                FieldState::Value,
                FieldState::Value, // qty=0 but vendor sent it
                FieldState::Value,
            ),
        );

        // Case 2: qty=0 and state=Absent (vendor didn't send)
        let snap_absent = MarketSnapshot::v2_with_states(
            1000,
            1001,
            0,
            0, // zero quantities
            -2,
            -8,
            10,
            1234567890,
            build_l1_state_bits(
                FieldState::Value,
                FieldState::Value,
                FieldState::Absent, // qty not sent
                FieldState::Absent,
            ),
        );

        // Both have bid_qty_mantissa=0 but different states
        assert_eq!(
            snap_value.bid_qty_mantissa(),
            snap_absent.bid_qty_mantissa()
        );
        assert_ne!(snap_value.bid_qty_state(), snap_absent.bid_qty_state());

        // This is the doctrine: we can distinguish "sent 0" from "not sent"
        assert_eq!(snap_value.bid_qty_state(), FieldState::Value);
        assert_eq!(snap_absent.bid_qty_state(), FieldState::Absent);
    }

    #[test]
    fn test_market_snapshot_schema_version_bytes() {
        let v1 = MarketSnapshot::V1(MarketSnapshotV1 {
            bid_price_mantissa: 1000,
            ask_price_mantissa: 1001,
            bid_qty_mantissa: 500,
            ask_qty_mantissa: 600,
            price_exponent: -2,
            qty_exponent: -8,
            spread_bps_mantissa: 10,
            book_ts_ns: 1234567890,
        });

        let v2 = MarketSnapshot::v2_all_present(1000, 1001, 500, 600, -2, -8, 10, 1234567890);

        // Schema version bytes are different
        assert_eq!(v1.schema_version_byte(), 0x01);
        assert_eq!(v2.schema_version_byte(), 0x02);
    }
}
