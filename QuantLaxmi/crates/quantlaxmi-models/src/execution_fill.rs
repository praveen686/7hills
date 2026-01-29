//! Phase 24A: Execution Fill Record — Deterministic fill WAL schema.
//!
//! Provides audit-grade observability for order execution outcomes.
//! Every fill is recorded BEFORE updating position/ledger state.
//!
//! ## Fixed-Point Encoding
//! Monetary values use mantissa + exponent (i64 + i8):
//! - qty_mantissa / qty_exponent (required)
//! - price_mantissa / price_exponent (required)
//! - fee_mantissa / fee_exponent (optional — None if unknown/not reported)
//!
//! ## Canonical Bytes (Frozen v1)
//! Field order for SHA-256 digest computation is frozen.
//! All integers use **little-endian** (matching 23B WAL convention).
//! **digest is derived from canonical bytes and NOT included in canonical bytes.**
//!
//! 1. schema_version (u32 LE len + utf8)
//! 2. ts_ns (i64 LE)
//! 3. session_id (u32 LE len + utf8)
//! 4. seq (u64 LE)
//! 5. parent_intent_seq (0x00 None | 0x01 + u64 LE)
//! 6. strategy_id (u32 LE len + utf8)
//! 7. symbol (u32 LE len + utf8)
//! 8. side (u8: 0x00=Buy, 0x01=Sell)
//! 9. qty_mantissa (i64 LE)
//! 10. qty_exponent (i8)
//! 11. price_mantissa (i64 LE)
//! 12. price_exponent (i8)
//! 13. fee (0x00 None | 0x01 + i64 LE mantissa + i8 exponent)
//! 14. venue (u32 LE len + utf8)
//! 15. correlation_id (u32 LE len + utf8)
//! 16. fill_type (u8: 0x00=Full, 0x01=Partial)
//! 17. parent_intent_digest (0x00 None | 0x01 + u32 LE len + utf8)
//!
//! ## Key Structure
//! Primary key: (session_id, seq)
//! seq is monotonic within a session, increments for every fill.
//! Duplicate key = hard error (broken seq monotonicity).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema version for execution fill records (frozen v1).
pub const EXECUTION_FILL_SCHEMA_VERSION: &str = "1.0.0";

// =============================================================================
// Enums
// =============================================================================

/// Fill side (Buy/Sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillSide {
    Buy,
    Sell,
}

impl FillSide {
    /// Canonical byte for hashing (0x00=Buy, 0x01=Sell).
    pub fn canonical_byte(&self) -> u8 {
        match self {
            FillSide::Buy => 0x00,
            FillSide::Sell => 0x01,
        }
    }
}

/// Fill type (fill shape, not execution mode).
///
/// Note: "simulated" is NOT a fill type — use `venue` field for execution mode
/// ("sim", "paper", "binance", etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FillType {
    /// Full fill — order completely filled
    Full,
    /// Partial fill — order partially filled
    Partial,
}

impl FillType {
    /// Canonical byte for hashing (0x00=Full, 0x01=Partial).
    pub fn canonical_byte(&self) -> u8 {
        match self {
            FillType::Full => 0x00,
            FillType::Partial => 0x01,
        }
    }
}

// =============================================================================
// ExecutionFillRecord
// =============================================================================

/// A single execution fill record for the WAL.
///
/// Written to `wal/execution_fills.jsonl` BEFORE updating position/ledger state.
/// Every fill produces a record regardless of fill type (full/partial).
///
/// ## Manual/External Fills
/// For fills not linked to a tracked intent (parent_intent_seq=None, venue!="sim"):
/// - `strategy_id` may be "unknown"
/// - `correlation_id` may be "corr_unknown"
/// These sentinels are allowed only in that constrained case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionFillRecord {
    /// Schema version for compatibility checking.
    pub schema_version: String,

    /// Timestamp in nanoseconds since Unix epoch (simulated time).
    pub ts_ns: i64,

    /// Session identifier (e.g., "backtest_20260128_abc123").
    pub session_id: String,

    /// Monotonically increasing sequence number within session.
    /// Increments for every fill. Duplicate = hard error.
    pub seq: u64,

    /// Parent order intent sequence (links to order_intent.jsonl).
    /// None if fill is not linked to a tracked intent (e.g., manual order).
    pub parent_intent_seq: Option<u64>,

    /// Parent intent digest for robust audit joins (hex SHA-256).
    /// None if parent_intent_seq is None.
    pub parent_intent_digest: Option<String>,

    /// Strategy that originated this fill.
    /// Use "unknown" only for manual fills (parent_intent_seq=None, venue!="sim").
    pub strategy_id: String,

    /// Symbol traded.
    pub symbol: String,

    /// Fill side.
    pub side: FillSide,

    /// Filled quantity mantissa (fixed-point).
    pub qty_mantissa: i64,

    /// Quantity exponent (typically -8 for crypto, -4 for equities).
    pub qty_exponent: i8,

    /// Fill price mantissa (fixed-point).
    pub price_mantissa: i64,

    /// Price exponent (typically -2 for most markets).
    pub price_exponent: i8,

    /// Fee/commission mantissa (fixed-point).
    /// None if fee is unknown or not reported per fill.
    pub fee_mantissa: Option<i64>,

    /// Fee exponent (only meaningful if fee_mantissa is Some).
    pub fee_exponent: i8,

    /// Execution venue (e.g., "binance", "sim", "paper").
    /// Use venue to indicate execution mode, not fill_type.
    pub venue: String,

    /// Correlation ID linking to upstream events.
    /// Use "corr_unknown" only for manual fills (parent_intent_seq=None, venue!="sim").
    pub correlation_id: String,

    /// Fill type (full/partial). NOT execution mode — use venue for that.
    pub fill_type: FillType,

    /// SHA-256 hex digest of canonical bytes (derived, not included in canonical).
    pub digest: String,
}

impl ExecutionFillRecord {
    /// Create a new builder for ExecutionFillRecord.
    pub fn builder(strategy_id: &str, symbol: &str) -> ExecutionFillRecordBuilder {
        ExecutionFillRecordBuilder::new(strategy_id, symbol)
    }

    /// Compute canonical bytes for deterministic hashing.
    ///
    /// Field order is frozen and must not change.
    /// All integers use little-endian (matching 23B WAL convention).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(256);

        // 1. schema_version (u32 LE len + utf8)
        write_string(&mut bytes, &self.schema_version);

        // 2. ts_ns (i64 LE)
        bytes.extend_from_slice(&self.ts_ns.to_le_bytes());

        // 3. session_id (u32 LE len + utf8)
        write_string(&mut bytes, &self.session_id);

        // 4. seq (u64 LE)
        bytes.extend_from_slice(&self.seq.to_le_bytes());

        // 5. parent_intent_seq (0x00 None | 0x01 + u64 LE)
        write_option_u64(&mut bytes, &self.parent_intent_seq);

        // 6. strategy_id (u32 LE len + utf8)
        write_string(&mut bytes, &self.strategy_id);

        // 7. symbol (u32 LE len + utf8)
        write_string(&mut bytes, &self.symbol);

        // 8. side (u8: 0x00=Buy, 0x01=Sell)
        bytes.push(self.side.canonical_byte());

        // 9. qty_mantissa (i64 LE)
        bytes.extend_from_slice(&self.qty_mantissa.to_le_bytes());

        // 10. qty_exponent (i8)
        bytes.push(self.qty_exponent as u8);

        // 11. price_mantissa (i64 LE)
        bytes.extend_from_slice(&self.price_mantissa.to_le_bytes());

        // 12. price_exponent (i8)
        bytes.push(self.price_exponent as u8);

        // 13. fee (0x00 None | 0x01 + i64 LE mantissa + i8 exponent)
        write_option_fee(&mut bytes, &self.fee_mantissa, self.fee_exponent);

        // 14. venue (u32 LE len + utf8)
        write_string(&mut bytes, &self.venue);

        // 15. correlation_id (u32 LE len + utf8)
        write_string(&mut bytes, &self.correlation_id);

        // 16. fill_type (u8: 0x00=Full, 0x01=Partial)
        bytes.push(self.fill_type.canonical_byte());

        // 17. parent_intent_digest (0x00 None | 0x01 + u32 LE len + utf8)
        write_option_string(&mut bytes, &self.parent_intent_digest);

        bytes
    }

    /// Compute SHA-256 digest of canonical bytes.
    pub fn compute_digest(&self) -> String {
        let bytes = self.canonical_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for ExecutionFillRecord.
///
/// # Example
/// ```ignore
/// let record = ExecutionFillRecord::builder("my_strategy", "BTCUSDT")
///     .ts_ns(1706400000000000000)
///     .session_id("backtest_abc")
///     .seq(1)
///     .parent_intent_seq(5)
///     .parent_intent_digest("abc123...")
///     .side(FillSide::Buy)
///     .qty(100000000, -8)  // 1.0 BTC
///     .price(4200000, -2)  // $42,000.00
///     .fee(420, -2)        // $4.20 (optional)
///     .venue("sim")
///     .correlation_id("event_seq:123")
///     .fill_type(FillType::Full)
///     .build();
/// ```
pub struct ExecutionFillRecordBuilder {
    strategy_id: String,
    symbol: String,
    ts_ns: i64,
    session_id: String,
    seq: u64,
    parent_intent_seq: Option<u64>,
    parent_intent_digest: Option<String>,
    side: FillSide,
    qty_mantissa: i64,
    qty_exponent: i8,
    price_mantissa: i64,
    price_exponent: i8,
    fee_mantissa: Option<i64>,
    fee_exponent: i8,
    venue: String,
    correlation_id: String,
    fill_type: FillType,
}

impl ExecutionFillRecordBuilder {
    /// Create a new builder with required fields.
    pub fn new(strategy_id: &str, symbol: &str) -> Self {
        Self {
            strategy_id: strategy_id.to_string(),
            symbol: symbol.to_string(),
            ts_ns: 0,
            session_id: String::new(),
            seq: 0,
            parent_intent_seq: None,
            parent_intent_digest: None,
            side: FillSide::Buy,
            qty_mantissa: 0,
            qty_exponent: -8,
            price_mantissa: 0,
            price_exponent: -2,
            fee_mantissa: None,
            fee_exponent: -2,
            venue: "sim".to_string(),
            correlation_id: String::new(),
            fill_type: FillType::Full,
        }
    }

    /// Set timestamp in nanoseconds.
    pub fn ts_ns(mut self, ts_ns: i64) -> Self {
        self.ts_ns = ts_ns;
        self
    }

    /// Set session ID.
    pub fn session_id(mut self, session_id: &str) -> Self {
        self.session_id = session_id.to_string();
        self
    }

    /// Set sequence number.
    pub fn seq(mut self, seq: u64) -> Self {
        self.seq = seq;
        self
    }

    /// Set parent intent sequence (links to order_intent.jsonl).
    pub fn parent_intent_seq(mut self, seq: u64) -> Self {
        self.parent_intent_seq = Some(seq);
        self
    }

    /// Set parent intent digest (hex SHA-256 for robust audit joins).
    pub fn parent_intent_digest(mut self, digest: &str) -> Self {
        self.parent_intent_digest = Some(digest.to_string());
        self
    }

    /// Set fill side.
    pub fn side(mut self, side: FillSide) -> Self {
        self.side = side;
        self
    }

    /// Set quantity (mantissa + exponent).
    pub fn qty(mut self, mantissa: i64, exponent: i8) -> Self {
        self.qty_mantissa = mantissa;
        self.qty_exponent = exponent;
        self
    }

    /// Set price (mantissa + exponent).
    pub fn price(mut self, mantissa: i64, exponent: i8) -> Self {
        self.price_mantissa = mantissa;
        self.price_exponent = exponent;
        self
    }

    /// Set fee (mantissa + exponent). Optional — call only if fee is known.
    pub fn fee(mut self, mantissa: i64, exponent: i8) -> Self {
        self.fee_mantissa = Some(mantissa);
        self.fee_exponent = exponent;
        self
    }

    /// Set venue (e.g., "sim", "paper", "binance").
    pub fn venue(mut self, venue: &str) -> Self {
        self.venue = venue.to_string();
        self
    }

    /// Set correlation ID.
    pub fn correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = id.to_string();
        self
    }

    /// Set fill type (full/partial).
    pub fn fill_type(mut self, fill_type: FillType) -> Self {
        self.fill_type = fill_type;
        self
    }

    /// Build the record, computing the digest.
    pub fn build(self) -> ExecutionFillRecord {
        let mut record = ExecutionFillRecord {
            schema_version: EXECUTION_FILL_SCHEMA_VERSION.to_string(),
            ts_ns: self.ts_ns,
            session_id: self.session_id,
            seq: self.seq,
            parent_intent_seq: self.parent_intent_seq,
            parent_intent_digest: self.parent_intent_digest,
            strategy_id: self.strategy_id,
            symbol: self.symbol,
            side: self.side,
            qty_mantissa: self.qty_mantissa,
            qty_exponent: self.qty_exponent,
            price_mantissa: self.price_mantissa,
            price_exponent: self.price_exponent,
            fee_mantissa: self.fee_mantissa,
            fee_exponent: self.fee_exponent,
            venue: self.venue,
            correlation_id: self.correlation_id,
            fill_type: self.fill_type,
            digest: String::new(),
        };
        record.digest = record.compute_digest();
        record
    }
}

// =============================================================================
// Helper functions for canonical bytes
// =============================================================================

/// Write a length-prefixed UTF-8 string (u32 LE len + utf8).
fn write_string(bytes: &mut Vec<u8>, s: &str) {
    let s_bytes = s.as_bytes();
    bytes.extend_from_slice(&(s_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(s_bytes);
}

/// Write an Option<u64> (0x00 for None, 0x01 + u64 LE for Some).
fn write_option_u64(bytes: &mut Vec<u8>, opt: &Option<u64>) {
    match opt {
        None => bytes.push(0x00),
        Some(v) => {
            bytes.push(0x01);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Write an optional fee (0x00 for None, 0x01 + i64 LE mantissa + i8 exponent for Some).
fn write_option_fee(bytes: &mut Vec<u8>, mantissa: &Option<i64>, exponent: i8) {
    match mantissa {
        None => bytes.push(0x00),
        Some(m) => {
            bytes.push(0x01);
            bytes.extend_from_slice(&m.to_le_bytes());
            bytes.push(exponent as u8);
        }
    }
}

/// Write an Option<String> (0x00 for None, 0x01 + u32 LE len + utf8 for Some).
fn write_option_string(bytes: &mut Vec<u8>, opt: &Option<String>) {
    match opt {
        None => bytes.push(0x00),
        Some(s) => {
            bytes.push(0x01);
            write_string(bytes, s);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_side_canonical_bytes() {
        assert_eq!(FillSide::Buy.canonical_byte(), 0x00);
        assert_eq!(FillSide::Sell.canonical_byte(), 0x01);
    }

    #[test]
    fn test_fill_type_canonical_bytes() {
        assert_eq!(FillType::Full.canonical_byte(), 0x00);
        assert_eq!(FillType::Partial.canonical_byte(), 0x01);
    }

    #[test]
    fn test_record_digest_deterministic() {
        // Build the same record 100 times, digest must be identical
        let mut digests = Vec::new();

        for _ in 0..100 {
            let record = ExecutionFillRecord::builder("test_strat", "BTCUSDT")
                .ts_ns(1706400000000000000)
                .session_id("test_session")
                .seq(1)
                .parent_intent_seq(5)
                .parent_intent_digest("abc123def456")
                .side(FillSide::Buy)
                .qty(100000000, -8)
                .price(4200000, -2)
                .fee(420, -2)
                .venue("sim")
                .correlation_id("test_corr")
                .fill_type(FillType::Full)
                .build();

            digests.push(record.digest.clone());
        }

        let first = &digests[0];
        for d in &digests {
            assert_eq!(d, first, "Digest must be deterministic");
        }
    }

    #[test]
    fn test_record_digest_changes_with_content() {
        let base = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        // Change ts_ns
        let diff_ts = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000) // Different
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_ne!(base.digest, diff_ts.digest);

        // Change side
        let diff_side = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Sell) // Different
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_ne!(base.digest, diff_side.digest);

        // Change qty
        let diff_qty = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(200, -8) // Different
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_ne!(base.digest, diff_qty.digest);

        // Change fill_type
        let diff_type = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Partial) // Different
            .build();
        assert_ne!(base.digest, diff_type.digest);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let record = ExecutionFillRecord::builder("test_strat", "ETHUSDT")
            .ts_ns(1706400000000000000)
            .session_id("backtest_123")
            .seq(42)
            .parent_intent_seq(10)
            .parent_intent_digest("deadbeef1234")
            .side(FillSide::Sell)
            .qty(500000000, -8)
            .price(250000, -2)
            .fee(250, -2)
            .venue("binance")
            .correlation_id("event_seq:99")
            .fill_type(FillType::Full)
            .build();

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: ExecutionFillRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record, deserialized);
        assert_eq!(record.digest, deserialized.digest);
    }

    #[test]
    fn test_schema_version() {
        let record = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        assert_eq!(record.schema_version, EXECUTION_FILL_SCHEMA_VERSION);
        assert_eq!(record.schema_version, "1.0.0");
    }

    #[test]
    fn test_key_uniqueness_session_seq() {
        let r1 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess_a")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let r2 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess_a")
            .seq(2) // Different seq
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let r3 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess_b") // Different session
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        // Keys are (session_id, seq)
        let key1 = (&r1.session_id, r1.seq);
        let key2 = (&r2.session_id, r2.seq);
        let key3 = (&r3.session_id, r3.seq);

        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key2, key3);
    }

    #[test]
    fn test_parent_intent_fields_optional() {
        // With parent_intent_seq and parent_intent_digest
        let with_parent = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .parent_intent_seq(5)
            .parent_intent_digest("abc123")
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_eq!(with_parent.parent_intent_seq, Some(5));
        assert_eq!(with_parent.parent_intent_digest, Some("abc123".to_string()));

        // Without parent
        let without_parent = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_eq!(without_parent.parent_intent_seq, None);
        assert_eq!(without_parent.parent_intent_digest, None);

        // Digests should differ
        assert_ne!(with_parent.digest, without_parent.digest);
    }

    #[test]
    fn test_fee_optional() {
        // With fee
        let with_fee = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .fee(10, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_eq!(with_fee.fee_mantissa, Some(10));
        assert_eq!(with_fee.fee_exponent, -2);

        // Without fee (unknown)
        let without_fee = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            // No .fee() call
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        assert_eq!(without_fee.fee_mantissa, None);

        // Digests should differ
        assert_ne!(with_fee.digest, without_fee.digest);
    }

    #[test]
    fn test_builder_defaults() {
        let record = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1)
            .build();

        // Check defaults
        assert_eq!(record.ts_ns, 0);
        assert_eq!(record.side, FillSide::Buy);
        assert_eq!(record.qty_exponent, -8);
        assert_eq!(record.price_exponent, -2);
        assert_eq!(record.fee_exponent, -2);
        assert_eq!(record.fee_mantissa, None); // Fee is None by default
        assert_eq!(record.venue, "sim");
        assert_eq!(record.fill_type, FillType::Full); // Default is Full, not Simulated
        assert_eq!(record.parent_intent_seq, None);
        assert_eq!(record.parent_intent_digest, None);
    }

    #[test]
    fn test_manual_fill_sentinels() {
        // Manual fill with "unknown" sentinels
        let manual = ExecutionFillRecord::builder("unknown", "BTCUSDT")
            .ts_ns(1000)
            .session_id("external_import")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("binance") // External venue, not "sim"
            .correlation_id("corr_unknown")
            .fill_type(FillType::Full)
            .build();

        assert_eq!(manual.strategy_id, "unknown");
        assert_eq!(manual.correlation_id, "corr_unknown");
        assert_eq!(manual.parent_intent_seq, None);
        assert_eq!(manual.parent_intent_digest, None);
    }

    #[test]
    fn test_partial_fill() {
        let partial = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .parent_intent_seq(5)
            .side(FillSide::Buy)
            .qty(50, -8) // Partial qty
            .price(5000, -2)
            .venue("binance")
            .correlation_id("c1")
            .fill_type(FillType::Partial)
            .build();

        assert_eq!(partial.fill_type, FillType::Partial);
    }

    #[test]
    fn test_serialization_with_optional_none() {
        // Record with fee=None and parent_intent=None
        let record = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: ExecutionFillRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record, deserialized);
        assert_eq!(deserialized.fee_mantissa, None);
        assert_eq!(deserialized.parent_intent_seq, None);
        assert_eq!(deserialized.parent_intent_digest, None);
    }
}
