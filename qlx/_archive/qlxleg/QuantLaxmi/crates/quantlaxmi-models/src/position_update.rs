//! Phase 24D: Position/Ledger Update WAL Record (v1).
//!
//! Extends the audit chain: Intent (23B) → Fill (24A) → Position Update (24D).
//!
//! Records deterministic state transitions per applied fill, enabling replay
//! to prove state evolution parity independent of strategy callbacks.
//!
//! ## WAL Path
//! `wal/position_updates.jsonl`
//!
//! ## Primary Key
//! `(session_id, seq)` — strictly monotonic within session
//!
//! ## Write Timing (Frozen)
//! - AFTER applying the fill to internal position/ledger state
//! - BEFORE calling any callbacks that observe the updated state
//!
//! ## Canonical Bytes Field Order (Frozen v1, LE encoding)
//! 1. schema_version (u32 len + UTF-8)
//! 2. ts_ns (i64 LE)
//! 3. session_id (u32 len + UTF-8)
//! 4. seq (u64 LE)
//! 5. correlation_id (u32 len + UTF-8)
//! 6. strategy_id (u32 len + UTF-8)
//! 7. symbol (u32 len + UTF-8)
//! 8. fill_seq (u64 LE)
//! 9. position_qty_mantissa (i64 LE)
//! 10. qty_exponent (i8)
//! 11. avg_price_mantissa (0x00 if None, else 0x01 + i64 LE)
//! 12. price_exponent (i8)
//! 13. cash_delta_mantissa (i64 LE)
//! 14. cash_exponent (i8)
//! 15. realized_pnl_delta_mantissa (i64 LE)
//! 16. pnl_exponent (i8)
//! 17. fee_mantissa (0x00 if None, else 0x01 + i64 LE)
//! 18. fee_exponent (i8)
//! 19. venue (u32 len + UTF-8)
//!
//! Note: `digest` is NOT included in canonical bytes (derived field).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema version for PositionUpdateRecord (frozen v1).
pub const POSITION_UPDATE_SCHEMA_VERSION: &str = "1.0.0";

/// Phase 24D: Position/Ledger update WAL record (v1).
///
/// Records the post-state snapshot and deltas after applying a fill to position/ledger state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PositionUpdateRecord {
    /// Schema version for this record type (frozen)
    pub schema_version: String,

    /// Timestamp nanoseconds (simulated time); NOT compared in determinism gates (v1)
    pub ts_ns: i64,

    /// Session identifier
    pub session_id: String,

    /// Monotonic sequence within session
    pub seq: u64,

    // === Linkage ===
    /// Links upstream (correlation)
    pub correlation_id: String,

    /// Originating strategy
    pub strategy_id: String,

    /// Instrument symbol
    pub symbol: String,

    /// References execution_fills.jsonl seq (same session)
    pub fill_seq: u64,

    // === Post-state snapshot (after applying this fill) ===
    /// Position quantity mantissa (post-fill)
    pub position_qty_mantissa: i64,

    /// Position quantity exponent
    pub qty_exponent: i8,

    /// Average entry price mantissa AFTER this fill.
    /// None iff position is flat.
    pub avg_price_mantissa: Option<i64>,

    /// Price exponent (always present for deterministic encoding)
    pub price_exponent: i8,

    // === Ledger deltas attributable to THIS fill application ===
    /// Cash delta mantissa (negative for buy, positive for sell)
    pub cash_delta_mantissa: i64,

    /// Cash exponent
    pub cash_exponent: i8,

    /// Realized PnL delta mantissa (can be 0)
    pub realized_pnl_delta_mantissa: i64,

    /// PnL exponent
    pub pnl_exponent: i8,

    /// Fee charged on THIS fill (if known)
    pub fee_mantissa: Option<i64>,

    /// Fee exponent (always present for deterministic encoding)
    pub fee_exponent: i8,

    // === Execution context ===
    /// Venue: "sim" | "binance" | "paper" etc.
    pub venue: String,

    /// Canonical digest of this record (SHA-256, hex); derived field
    pub digest: String,
}

impl PositionUpdateRecord {
    /// Compute canonical bytes for hashing (frozen v1 field order, LE encoding).
    ///
    /// The `digest` field is NOT included in canonical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);

        // 1. schema_version (u32 len + UTF-8)
        let schema_bytes = self.schema_version.as_bytes();
        buf.extend_from_slice(&(schema_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(schema_bytes);

        // 2. ts_ns (i64 LE)
        buf.extend_from_slice(&self.ts_ns.to_le_bytes());

        // 3. session_id (u32 len + UTF-8)
        let session_bytes = self.session_id.as_bytes();
        buf.extend_from_slice(&(session_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(session_bytes);

        // 4. seq (u64 LE)
        buf.extend_from_slice(&self.seq.to_le_bytes());

        // 5. correlation_id (u32 len + UTF-8)
        let corr_bytes = self.correlation_id.as_bytes();
        buf.extend_from_slice(&(corr_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(corr_bytes);

        // 6. strategy_id (u32 len + UTF-8)
        let strat_bytes = self.strategy_id.as_bytes();
        buf.extend_from_slice(&(strat_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(strat_bytes);

        // 7. symbol (u32 len + UTF-8)
        let sym_bytes = self.symbol.as_bytes();
        buf.extend_from_slice(&(sym_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(sym_bytes);

        // 8. fill_seq (u64 LE)
        buf.extend_from_slice(&self.fill_seq.to_le_bytes());

        // 9. position_qty_mantissa (i64 LE)
        buf.extend_from_slice(&self.position_qty_mantissa.to_le_bytes());

        // 10. qty_exponent (i8)
        buf.push(self.qty_exponent as u8);

        // 11. avg_price_mantissa (0x00 if None, else 0x01 + i64 LE)
        match self.avg_price_mantissa {
            None => buf.push(0x00),
            Some(v) => {
                buf.push(0x01);
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // 12. price_exponent (i8)
        buf.push(self.price_exponent as u8);

        // 13. cash_delta_mantissa (i64 LE)
        buf.extend_from_slice(&self.cash_delta_mantissa.to_le_bytes());

        // 14. cash_exponent (i8)
        buf.push(self.cash_exponent as u8);

        // 15. realized_pnl_delta_mantissa (i64 LE)
        buf.extend_from_slice(&self.realized_pnl_delta_mantissa.to_le_bytes());

        // 16. pnl_exponent (i8)
        buf.push(self.pnl_exponent as u8);

        // 17. fee_mantissa (0x00 if None, else 0x01 + i64 LE)
        match self.fee_mantissa {
            None => buf.push(0x00),
            Some(v) => {
                buf.push(0x01);
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // 18. fee_exponent (i8)
        buf.push(self.fee_exponent as u8);

        // 19. venue (u32 len + UTF-8)
        let venue_bytes = self.venue.as_bytes();
        buf.extend_from_slice(&(venue_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(venue_bytes);

        buf
    }

    /// Compute SHA-256 digest of canonical bytes (lowercase hex).
    pub fn compute_digest(&self) -> String {
        let bytes = self.canonical_bytes();
        let hash = Sha256::digest(&bytes);
        hex::encode(hash)
    }

    /// Create a new builder for constructing PositionUpdateRecord.
    pub fn builder(strategy_id: &str, symbol: &str) -> PositionUpdateRecordBuilder {
        PositionUpdateRecordBuilder::new(strategy_id, symbol)
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for PositionUpdateRecord with fluent API.
#[derive(Debug, Clone)]
pub struct PositionUpdateRecordBuilder {
    schema_version: String,
    ts_ns: i64,
    session_id: String,
    seq: u64,
    correlation_id: String,
    strategy_id: String,
    symbol: String,
    fill_seq: u64,
    position_qty_mantissa: i64,
    qty_exponent: i8,
    avg_price_mantissa: Option<i64>,
    price_exponent: i8,
    cash_delta_mantissa: i64,
    cash_exponent: i8,
    realized_pnl_delta_mantissa: i64,
    pnl_exponent: i8,
    fee_mantissa: Option<i64>,
    fee_exponent: i8,
    venue: String,
}

impl PositionUpdateRecordBuilder {
    /// Create a new builder with required fields.
    pub fn new(strategy_id: &str, symbol: &str) -> Self {
        Self {
            schema_version: POSITION_UPDATE_SCHEMA_VERSION.to_string(),
            ts_ns: 0,
            session_id: String::new(),
            seq: 0,
            correlation_id: "corr_unknown".to_string(),
            strategy_id: strategy_id.to_string(),
            symbol: symbol.to_string(),
            fill_seq: 0,
            position_qty_mantissa: 0,
            qty_exponent: 0,
            avg_price_mantissa: None,
            price_exponent: 0,
            cash_delta_mantissa: 0,
            cash_exponent: 0,
            realized_pnl_delta_mantissa: 0,
            pnl_exponent: 0,
            fee_mantissa: None,
            fee_exponent: 0,
            venue: "sim".to_string(),
        }
    }

    /// Set timestamp nanoseconds.
    pub fn ts_ns(mut self, ts_ns: i64) -> Self {
        self.ts_ns = ts_ns;
        self
    }

    /// Set session identifier.
    pub fn session_id(mut self, session_id: &str) -> Self {
        self.session_id = session_id.to_string();
        self
    }

    /// Set monotonic sequence.
    pub fn seq(mut self, seq: u64) -> Self {
        self.seq = seq;
        self
    }

    /// Set correlation ID.
    pub fn correlation_id(mut self, correlation_id: &str) -> Self {
        self.correlation_id = correlation_id.to_string();
        self
    }

    /// Set fill sequence reference (links to execution_fills.jsonl).
    pub fn fill_seq(mut self, fill_seq: u64) -> Self {
        self.fill_seq = fill_seq;
        self
    }

    /// Set post-fill position quantity (mantissa + exponent).
    pub fn position_qty(mut self, mantissa: i64, exponent: i8) -> Self {
        self.position_qty_mantissa = mantissa;
        self.qty_exponent = exponent;
        self
    }

    /// Set average entry price (mantissa + exponent). None if position is flat.
    pub fn avg_price(mut self, mantissa: i64, exponent: i8) -> Self {
        self.avg_price_mantissa = Some(mantissa);
        self.price_exponent = exponent;
        self
    }

    /// Set average price to None (position is flat) with explicit exponent.
    pub fn avg_price_flat(mut self, exponent: i8) -> Self {
        self.avg_price_mantissa = None;
        self.price_exponent = exponent;
        self
    }

    /// Set cash delta (mantissa + exponent).
    /// Sign convention: negative for buy (spend cash), positive for sell (receive cash).
    pub fn cash_delta(mut self, mantissa: i64, exponent: i8) -> Self {
        self.cash_delta_mantissa = mantissa;
        self.cash_exponent = exponent;
        self
    }

    /// Set realized PnL delta (mantissa + exponent).
    pub fn realized_pnl_delta(mut self, mantissa: i64, exponent: i8) -> Self {
        self.realized_pnl_delta_mantissa = mantissa;
        self.pnl_exponent = exponent;
        self
    }

    /// Set fee (mantissa + exponent). Fee should be non-negative.
    pub fn fee(mut self, mantissa: i64, exponent: i8) -> Self {
        self.fee_mantissa = Some(mantissa);
        self.fee_exponent = exponent;
        self
    }

    /// Set fee exponent without a fee value (fee unknown).
    pub fn fee_exponent(mut self, exponent: i8) -> Self {
        self.fee_exponent = exponent;
        self
    }

    /// Set venue.
    pub fn venue(mut self, venue: &str) -> Self {
        self.venue = venue.to_string();
        self
    }

    /// Build the record, computing digest automatically.
    pub fn build(self) -> PositionUpdateRecord {
        let mut record = PositionUpdateRecord {
            schema_version: self.schema_version,
            ts_ns: self.ts_ns,
            session_id: self.session_id,
            seq: self.seq,
            correlation_id: self.correlation_id,
            strategy_id: self.strategy_id,
            symbol: self.symbol,
            fill_seq: self.fill_seq,
            position_qty_mantissa: self.position_qty_mantissa,
            qty_exponent: self.qty_exponent,
            avg_price_mantissa: self.avg_price_mantissa,
            price_exponent: self.price_exponent,
            cash_delta_mantissa: self.cash_delta_mantissa,
            cash_exponent: self.cash_exponent,
            realized_pnl_delta_mantissa: self.realized_pnl_delta_mantissa,
            pnl_exponent: self.pnl_exponent,
            fee_mantissa: self.fee_mantissa,
            fee_exponent: self.fee_exponent,
            venue: self.venue,
            digest: String::new(),
        };
        record.digest = record.compute_digest();
        record
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test 1: Canonical bytes are deterministic
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_canonical_bytes_deterministic() {
        let record1 = PositionUpdateRecord::builder("strat_a", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("session_123")
            .seq(1)
            .correlation_id("corr_1")
            .fill_seq(5)
            .position_qty(100000000, -8)
            .avg_price(4200000, -2)
            .cash_delta(-42000000, -2)
            .realized_pnl_delta(0, -2)
            .fee(420, -2)
            .venue("sim")
            .build();

        let record2 = PositionUpdateRecord::builder("strat_a", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("session_123")
            .seq(1)
            .correlation_id("corr_1")
            .fill_seq(5)
            .position_qty(100000000, -8)
            .avg_price(4200000, -2)
            .cash_delta(-42000000, -2)
            .realized_pnl_delta(0, -2)
            .fee(420, -2)
            .venue("sim")
            .build();

        assert_eq!(
            record1.canonical_bytes(),
            record2.canonical_bytes(),
            "Identical records should have identical canonical bytes"
        );
    }

    // -------------------------------------------------------------------------
    // Test 2: Digest is stable
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_digest_stable() {
        let record1 = PositionUpdateRecord::builder("strat_a", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("session_123")
            .seq(1)
            .correlation_id("corr_1")
            .fill_seq(5)
            .position_qty(100000000, -8)
            .avg_price(4200000, -2)
            .cash_delta(-42000000, -2)
            .realized_pnl_delta(0, -2)
            .fee(420, -2)
            .venue("sim")
            .build();

        let record2 = PositionUpdateRecord::builder("strat_a", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("session_123")
            .seq(1)
            .correlation_id("corr_1")
            .fill_seq(5)
            .position_qty(100000000, -8)
            .avg_price(4200000, -2)
            .cash_delta(-42000000, -2)
            .realized_pnl_delta(0, -2)
            .fee(420, -2)
            .venue("sim")
            .build();

        assert_eq!(
            record1.digest, record2.digest,
            "Identical records should have identical digests"
        );

        // Digest should be 64 hex chars (SHA-256)
        assert_eq!(record1.digest.len(), 64);
        assert!(record1.digest.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // -------------------------------------------------------------------------
    // Test 3: Option encodings (avg_price None/Some, fee None/Some)
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_option_encodings() {
        // Record with avg_price=Some, fee=Some
        let with_both = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .fee(10, -2)
            .venue("sim")
            .build();

        // Record with avg_price=None (flat position), fee=Some
        let with_flat = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(2)
            .fill_seq(2)
            .position_qty(0, -8) // Flat
            .avg_price_flat(-2)
            .cash_delta(5000, -2)
            .realized_pnl_delta(100, -2)
            .fee(10, -2)
            .venue("sim")
            .build();

        // Record with avg_price=Some, fee=None
        let without_fee = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(3)
            .fill_seq(3)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .fee_exponent(-2)
            .venue("sim")
            .build();

        // Record with avg_price=None, fee=None
        let without_both = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(4)
            .fill_seq(4)
            .position_qty(0, -8)
            .avg_price_flat(-2)
            .cash_delta(5000, -2)
            .realized_pnl_delta(100, -2)
            .fee_exponent(-2)
            .venue("sim")
            .build();

        // All should have valid, unique digests
        assert_ne!(with_both.digest, with_flat.digest);
        assert_ne!(with_both.digest, without_fee.digest);
        assert_ne!(with_flat.digest, without_both.digest);

        // Verify option values
        assert_eq!(with_both.avg_price_mantissa, Some(5000));
        assert_eq!(with_both.fee_mantissa, Some(10));

        assert_eq!(with_flat.avg_price_mantissa, None);
        assert_eq!(with_flat.fee_mantissa, Some(10));

        assert_eq!(without_fee.avg_price_mantissa, Some(5000));
        assert_eq!(without_fee.fee_mantissa, None);

        assert_eq!(without_both.avg_price_mantissa, None);
        assert_eq!(without_both.fee_mantissa, None);
    }

    // -------------------------------------------------------------------------
    // Test 4: Schema version is stable
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_schema_version_stable() {
        assert_eq!(POSITION_UPDATE_SCHEMA_VERSION, "1.0.0");

        let record = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        assert_eq!(record.schema_version, "1.0.0");
    }

    // -------------------------------------------------------------------------
    // Test 5: Different field values produce different digests
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_different_fields_different_digest() {
        let base = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        // Different position_qty
        let diff_qty = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(200, -8) // Different
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        // Different cash_delta
        let diff_cash = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-6000, -2) // Different
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        // Different fill_seq
        let diff_fill = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(2) // Different
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        assert_ne!(base.digest, diff_qty.digest);
        assert_ne!(base.digest, diff_cash.digest);
        assert_ne!(base.digest, diff_fill.digest);
    }

    // -------------------------------------------------------------------------
    // Test 6: Serialization roundtrip
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_serde_roundtrip() {
        let record = PositionUpdateRecord::builder("strat_a", "ETHUSDT")
            .ts_ns(1706400000000000000)
            .session_id("session_456")
            .seq(42)
            .correlation_id("corr_42")
            .fill_seq(10)
            .position_qty(500000000, -8)
            .avg_price(250000, -2)
            .cash_delta(-1250000, -2)
            .realized_pnl_delta(5000, -2)
            .fee(125, -2)
            .venue("binance")
            .build();

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: PositionUpdateRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record, deserialized);
        assert_eq!(record.digest, deserialized.digest);
    }

    // -------------------------------------------------------------------------
    // Test 7: Sign conventions
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_sign_conventions() {
        // Buy fill: cash delta negative (spend cash), position increases
        let buy_record = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100000000, -8) // 1 BTC long
            .avg_price(4200000, -2) // $42,000
            .cash_delta(-4200000, -2) // Spent $42,000 (negative)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        assert!(buy_record.position_qty_mantissa > 0);
        assert!(buy_record.cash_delta_mantissa < 0);

        // Sell fill: cash delta positive (receive cash), position decreases to flat
        let sell_record = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .fill_seq(2)
            .position_qty(0, -8) // Flat
            .avg_price_flat(-2)
            .cash_delta(4300000, -2) // Received $43,000 (positive)
            .realized_pnl_delta(100000, -2) // $1,000 profit
            .venue("sim")
            .build();

        assert_eq!(sell_record.position_qty_mantissa, 0);
        assert!(sell_record.cash_delta_mantissa > 0);
        assert!(sell_record.realized_pnl_delta_mantissa > 0);
    }

    // -------------------------------------------------------------------------
    // Test 8: Unknown/manual sentinels
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_sentinels() {
        // Default sentinel values
        let with_sentinels = PositionUpdateRecord::builder("unknown", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .correlation_id("corr_unknown")
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();

        assert_eq!(with_sentinels.strategy_id, "unknown");
        assert_eq!(with_sentinels.correlation_id, "corr_unknown");

        // Verify it still produces a valid digest
        assert_eq!(with_sentinels.digest.len(), 64);
    }

    // -------------------------------------------------------------------------
    // Test 9: Little-endian encoding verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_little_endian_encoding() {
        let record = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(0x0102030405060708i64)
            .session_id("s")
            .seq(0x1112131415161718u64)
            .fill_seq(0x2122232425262728u64)
            .position_qty(0x3132333435363738i64, -8)
            .avg_price(0x4142434445464748i64, -2)
            .cash_delta(0x5152535455565758i64, -2)
            .realized_pnl_delta(0x6162636465666768i64, -2)
            .venue("sim")
            .build();

        let bytes = record.canonical_bytes();

        // Find ts_ns position (after schema_version)
        // schema_version = "1.0.0" = 5 bytes + 4 byte len = 9 bytes
        let ts_offset = 9;
        let ts_bytes = &bytes[ts_offset..ts_offset + 8];
        // LE: least significant byte first
        assert_eq!(ts_bytes[0], 0x08);
        assert_eq!(ts_bytes[7], 0x01);
    }

    // -------------------------------------------------------------------------
    // Test 10: Canonical bytes length consistency
    // -------------------------------------------------------------------------

    #[test]
    fn test_position_update_canonical_bytes_length_consistency() {
        // Two records with same string lengths should have same canonical bytes length
        let record1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .correlation_id("corr")
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .fee(10, -2)
            .venue("sim")
            .build();

        let record2 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .ts_ns(2000)
            .session_id("sess")
            .seq(2)
            .correlation_id("corr")
            .fill_seq(2)
            .position_qty(200, -8)
            .avg_price(6000, -2)
            .cash_delta(-6000, -2)
            .realized_pnl_delta(100, -2)
            .fee(20, -2)
            .venue("sim")
            .build();

        assert_eq!(
            record1.canonical_bytes().len(),
            record2.canonical_bytes().len(),
            "Records with same string lengths should have same canonical bytes length"
        );
    }
}
