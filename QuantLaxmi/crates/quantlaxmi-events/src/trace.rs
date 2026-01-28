//! # Decision Trace Module
//!
//! Provides deterministic trace hashing for replay parity verification.
//!
//! ## Canonical Binary Encoding
//!
//! All `DecisionEvent` fields are encoded in a strict, deterministic binary format
//! that is **independent of JSON/serde serialization**. This ensures that:
//!
//! 1. Hash results are platform-independent (no float formatting differences)
//! 2. Hash results are serde-version-independent (no field order changes)
//! 3. Hash results are reproducible across compilations
//!
//! ### Encoding Rules
//!
//! - **Version byte**: First byte is the encoding version (currently 0x01)
//! - **Integers**: Little-endian byte encoding (consistent with most systems)
//! - **Strings**: Length-prefixed with u32 (4 bytes) followed by UTF-8 bytes
//! - **Optionals**: 1-byte presence marker (0x00 = None, 0x01 = Some) followed by value if present
//! - **UUIDs**: 16 bytes in standard byte representation
//! - **DateTime**: i64 timestamp in nanoseconds since Unix epoch
//! - **f64**: IEEE 754 bits as u64 little-endian (for canonical representation)
//! - **Nested structs**: Concatenated field encodings in declaration order
//! - **Maps/Sets**: Keys sorted lexicographically before encoding
//!
//! ### Field Order (DecisionEvent)
//!
//! After the version byte, fields are encoded in the following fixed order:
//! 1. ts (i64 nanoseconds)
//! 2. decision_id (16 bytes UUID)
//! 3. strategy_id (length-prefixed string)
//! 4. symbol (length-prefixed string)
//! 5. decision_type (length-prefixed string)
//! 6. direction (i8)
//! 7. target_qty_mantissa (i64)
//! 8. qty_exponent (i8)
//! 9. reference_price_mantissa (i64)
//! 10. price_exponent (i8)
//! 11. market_snapshot (nested struct)
//! 12. confidence (f64 bits)
//! 13. metadata (presence + canonical JSON with sorted keys if present)
//! 14. ctx (CorrelationContext with presence markers)
//!
//! ## Encoding Version History
//!
//! - v1 (0x01): Initial encoding format

use chrono::{DateTime, Utc};
use quantlaxmi_models::{CorrelationContext, DecisionEvent, MarketSnapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Current encoding version for canonical bytes.
/// Increment this when encoding format changes.
///
/// ## Version History
/// - v1 (0x01): Initial encoding with f64 confidence and spread_bps
/// - v2 (0x02): Fixed-point confidence_mantissa and spread_bps_mantissa (no floats)
/// - v3 (0x03): MarketSnapshot V2 with l1_state_bits + versioned enum with explicit presence tracking.
///   MarketSnapshot now has internal schema discriminant (0x01 for V1, 0x02 for V2).
///
/// ## Replay Parity Note
/// Replay parity MUST be evaluated within the same encoding version.
/// Cross-version parity comparison is a category error, not a failure.
pub const ENCODING_VERSION: u8 = 0x03;

/// Decision trace containing a sequence of decisions and their hash.
///
/// The trace hash is computed incrementally as decisions are recorded,
/// providing O(1) verification of replay parity.
///
/// ## Serialization
///
/// The trace can be serialized to JSON for persistence. The trace_hash is
/// stored as a hex string for human readability and easy comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    /// Encoding version used to generate this trace.
    pub encoding_version: u8,
    /// The sequence of decisions in order of occurrence.
    pub decisions: Vec<DecisionEvent>,
    /// SHA-256 hash of the canonical binary encoding of all decisions (hex).
    #[serde(with = "hex_hash")]
    pub trace_hash: [u8; 32],
}

/// Serde helper for hex encoding/decoding of hash bytes.
mod hex_hash {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(hash: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(hash))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom(format!(
                "expected 32 bytes, got {}",
                bytes.len()
            )));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(arr)
    }
}

impl DecisionTrace {
    /// Verify parity with another trace by comparing hashes.
    pub fn verify_parity(&self, other: &DecisionTrace) -> bool {
        self.trace_hash == other.trace_hash
    }

    /// Get the trace hash as a hex string (for logging/display).
    pub fn hash_hex(&self) -> String {
        hex::encode(self.trace_hash)
    }

    /// Get the number of decisions in the trace.
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Check if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Load a trace from a JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, TraceError> {
        let file = std::fs::File::open(path).map_err(|e| TraceError::Io(e.to_string()))?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| TraceError::Parse(e.to_string()))
    }

    /// Save the trace to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), TraceError> {
        let file = std::fs::File::create(path).map_err(|e| TraceError::Io(e.to_string()))?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(|e| TraceError::Serialize(e.to_string()))
    }
}

/// Errors that can occur during trace operations.
#[derive(Debug, Clone)]
pub enum TraceError {
    /// I/O error (file not found, permission denied, etc.)
    Io(String),
    /// JSON parsing error
    Parse(String),
    /// Serialization error
    Serialize(String),
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {}", msg),
            Self::Parse(msg) => write!(f, "Parse error: {}", msg),
            Self::Serialize(msg) => write!(f, "Serialize error: {}", msg),
        }
    }
}

impl std::error::Error for TraceError {}

/// Builder for constructing a decision trace with incremental hashing.
///
/// # Example
///
/// ```ignore
/// let mut builder = DecisionTraceBuilder::new();
/// builder.record(&decision1);
/// builder.record(&decision2);
/// let trace = builder.finalize();
/// ```
#[derive(Debug)]
pub struct DecisionTraceBuilder {
    decisions: Vec<DecisionEvent>,
    hasher: Sha256,
}

impl Default for DecisionTraceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTraceBuilder {
    /// Create a new empty trace builder.
    pub fn new() -> Self {
        Self {
            decisions: Vec::new(),
            hasher: Sha256::new(),
        }
    }

    /// Record a decision event, updating the incremental hash.
    pub fn record(&mut self, decision: &DecisionEvent) {
        let bytes = canonical_bytes(decision);
        self.hasher.update(&bytes);
        self.decisions.push(decision.clone());
    }

    /// Finalize the trace, consuming the builder and returning the complete trace.
    pub fn finalize(self) -> DecisionTrace {
        let hash = self.hasher.finalize();
        DecisionTrace {
            encoding_version: ENCODING_VERSION,
            decisions: self.decisions,
            trace_hash: hash.into(),
        }
    }

    /// Get the current decision count.
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Check if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
}

/// Result of replay parity verification.
#[derive(Debug, Clone)]
pub enum ReplayParityResult {
    /// Traces match - identical hash values.
    Match,
    /// Traces diverge at a specific index.
    /// Note: DecisionEvent fields are boxed to reduce enum size.
    Divergence {
        /// Index where divergence was first detected (0-based).
        index: usize,
        /// The original decision at the divergence point.
        original: Box<DecisionEvent>,
        /// The replay decision at the divergence point.
        replay: Box<DecisionEvent>,
        /// Human-readable reason for divergence.
        reason: String,
    },
    /// Traces have different lengths.
    LengthMismatch {
        /// Length of the original trace.
        original_len: usize,
        /// Length of the replay trace.
        replay_len: usize,
    },
}

/// Verify replay parity between an original trace and a replay trace.
///
/// This function performs two levels of verification:
/// 1. Fast path: Compare trace hashes (O(1))
/// 2. Slow path: If hashes differ, find the first divergent decision
///
/// # Returns
///
/// - `ReplayParityResult::Match` if traces are identical
/// - `ReplayParityResult::Divergence` with details of first difference
/// - `ReplayParityResult::LengthMismatch` if traces have different lengths
pub fn verify_replay_parity(
    original: &DecisionTrace,
    replay: &DecisionTrace,
) -> ReplayParityResult {
    // Fast path: hash comparison
    if original.trace_hash == replay.trace_hash {
        return ReplayParityResult::Match;
    }

    // Slow path: find first divergence
    let min_len = original.decisions.len().min(replay.decisions.len());

    for i in 0..min_len {
        let orig_bytes = canonical_bytes(&original.decisions[i]);
        let replay_bytes = canonical_bytes(&replay.decisions[i]);

        if orig_bytes != replay_bytes {
            let reason = find_divergence_reason(&original.decisions[i], &replay.decisions[i]);
            return ReplayParityResult::Divergence {
                index: i,
                original: Box::new(original.decisions[i].clone()),
                replay: Box::new(replay.decisions[i].clone()),
                reason,
            };
        }
    }

    // Length mismatch
    ReplayParityResult::LengthMismatch {
        original_len: original.decisions.len(),
        replay_len: replay.decisions.len(),
    }
}

/// Find a human-readable reason for divergence between two decisions.
fn find_divergence_reason(original: &DecisionEvent, replay: &DecisionEvent) -> String {
    if original.ts != replay.ts {
        return format!(
            "timestamp differs: original={}, replay={}",
            original.ts, replay.ts
        );
    }
    if original.decision_id != replay.decision_id {
        return format!(
            "decision_id differs: original={}, replay={}",
            original.decision_id, replay.decision_id
        );
    }
    if original.strategy_id != replay.strategy_id {
        return format!(
            "strategy_id differs: original={}, replay={}",
            original.strategy_id, replay.strategy_id
        );
    }
    if original.symbol != replay.symbol {
        return format!(
            "symbol differs: original={}, replay={}",
            original.symbol, replay.symbol
        );
    }
    if original.decision_type != replay.decision_type {
        return format!(
            "decision_type differs: original={}, replay={}",
            original.decision_type, replay.decision_type
        );
    }
    if original.direction != replay.direction {
        return format!(
            "direction differs: original={}, replay={}",
            original.direction, replay.direction
        );
    }
    if original.target_qty_mantissa != replay.target_qty_mantissa {
        return format!(
            "target_qty_mantissa differs: original={}, replay={}",
            original.target_qty_mantissa, replay.target_qty_mantissa
        );
    }
    if original.qty_exponent != replay.qty_exponent {
        return format!(
            "qty_exponent differs: original={}, replay={}",
            original.qty_exponent, replay.qty_exponent
        );
    }
    if original.reference_price_mantissa != replay.reference_price_mantissa {
        return format!(
            "reference_price_mantissa differs: original={}, replay={}",
            original.reference_price_mantissa, replay.reference_price_mantissa
        );
    }
    if original.price_exponent != replay.price_exponent {
        return format!(
            "price_exponent differs: original={}, replay={}",
            original.price_exponent, replay.price_exponent
        );
    }
    if original.confidence_mantissa != replay.confidence_mantissa {
        return format!(
            "confidence_mantissa differs: original={}, replay={}",
            original.confidence_mantissa, replay.confidence_mantissa
        );
    }

    // Compare market snapshots
    let orig_snap = &original.market_snapshot;
    let replay_snap = &replay.market_snapshot;

    // Check schema version mismatch (V1 vs V2)
    if orig_snap.schema_version_byte() != replay_snap.schema_version_byte() {
        return format!(
            "market_snapshot schema differs: original=V{}, replay=V{}",
            orig_snap.schema_version_byte(),
            replay_snap.schema_version_byte()
        );
    }

    if orig_snap.bid_price_mantissa() != replay_snap.bid_price_mantissa()
        || orig_snap.ask_price_mantissa() != replay_snap.ask_price_mantissa()
        || orig_snap.bid_qty_mantissa() != replay_snap.bid_qty_mantissa()
        || orig_snap.ask_qty_mantissa() != replay_snap.ask_qty_mantissa()
        || orig_snap.book_ts_ns() != replay_snap.book_ts_ns()
    {
        return "market_snapshot differs".to_string();
    }

    // For V2: also check l1_state_bits
    if orig_snap.l1_state_bits() != replay_snap.l1_state_bits() {
        return format!(
            "market_snapshot l1_state_bits differs: original={:#06x}, replay={:#06x}",
            orig_snap.l1_state_bits(),
            replay_snap.l1_state_bits()
        );
    }

    // Compare metadata
    if original.metadata != replay.metadata {
        return "metadata differs".to_string();
    }

    // Compare correlation context
    if original.ctx.session_id != replay.ctx.session_id
        || original.ctx.run_id != replay.ctx.run_id
        || original.ctx.symbol != replay.ctx.symbol
        || original.ctx.venue != replay.ctx.venue
        || original.ctx.strategy_id != replay.ctx.strategy_id
        || original.ctx.decision_id != replay.ctx.decision_id
        || original.ctx.order_id != replay.ctx.order_id
    {
        return "correlation_context differs".to_string();
    }

    "unknown field differs".to_string()
}

// =============================================================================
// CANONICAL BINARY ENCODING
// =============================================================================

/// Encode a DecisionEvent to canonical bytes for hashing.
///
/// This encoding is deterministic and independent of serde/JSON serialization.
/// See module-level documentation for encoding rules.
///
/// The first byte is always the encoding version (ENCODING_VERSION).
pub fn canonical_bytes(decision: &DecisionEvent) -> Vec<u8> {
    let mut buf = Vec::with_capacity(512);

    // 0. Encoding version byte (for future compatibility)
    buf.push(ENCODING_VERSION);

    // 1. ts: DateTime as nanoseconds since Unix epoch
    encode_datetime(&mut buf, &decision.ts);

    // 2. decision_id: UUID as 16 bytes
    encode_uuid(&mut buf, &decision.decision_id);

    // 3. strategy_id: length-prefixed string
    encode_string(&mut buf, &decision.strategy_id);

    // 4. symbol: length-prefixed string
    encode_string(&mut buf, &decision.symbol);

    // 5. decision_type: length-prefixed string
    encode_string(&mut buf, &decision.decision_type);

    // 6. direction: i8
    encode_i8(&mut buf, decision.direction);

    // 7. target_qty_mantissa: i64
    encode_i64(&mut buf, decision.target_qty_mantissa);

    // 8. qty_exponent: i8
    encode_i8(&mut buf, decision.qty_exponent);

    // 9. reference_price_mantissa: i64
    encode_i64(&mut buf, decision.reference_price_mantissa);

    // 10. price_exponent: i8
    encode_i8(&mut buf, decision.price_exponent);

    // 11. market_snapshot: nested struct
    encode_market_snapshot(&mut buf, &decision.market_snapshot);

    // 12. confidence_mantissa: i64 (fixed exponent = -4, not encoded)
    encode_i64(&mut buf, decision.confidence_mantissa);

    // 13. metadata: presence marker + canonical JSON if present
    encode_json_value(&mut buf, &decision.metadata);

    // 14. ctx: CorrelationContext with presence markers
    encode_correlation_context(&mut buf, &decision.ctx);

    buf
}

/// Encode DateTime as i64 nanoseconds since Unix epoch.
fn encode_datetime(buf: &mut Vec<u8>, dt: &DateTime<Utc>) {
    let nanos = dt.timestamp_nanos_opt().unwrap_or(0);
    buf.extend_from_slice(&nanos.to_le_bytes());
}

/// Encode UUID as 16 bytes.
fn encode_uuid(buf: &mut Vec<u8>, uuid: &Uuid) {
    buf.extend_from_slice(uuid.as_bytes());
}

/// Encode string with u32 length prefix.
fn encode_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len() as u32;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Encode optional string with presence marker.
fn encode_optional_string(buf: &mut Vec<u8>, opt: &Option<String>) {
    match opt {
        Some(s) => {
            buf.push(0x01); // Present
            encode_string(buf, s);
        }
        None => {
            buf.push(0x00); // Absent
        }
    }
}

/// Encode optional UUID with presence marker.
fn encode_optional_uuid(buf: &mut Vec<u8>, opt: &Option<Uuid>) {
    match opt {
        Some(uuid) => {
            buf.push(0x01); // Present
            encode_uuid(buf, uuid);
        }
        None => {
            buf.push(0x00); // Absent
        }
    }
}

/// Encode i8.
fn encode_i8(buf: &mut Vec<u8>, val: i8) {
    buf.push(val as u8);
}

/// Encode i64 in little-endian.
fn encode_i64(buf: &mut Vec<u8>, val: i64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Encode u16 in little-endian.
fn encode_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Encode MarketSnapshot (versioned enum).
///
/// ## Encoding Format
/// - First byte: Schema version discriminant (0x01 for V1, 0x02 for V2)
/// - Followed by V1 common fields (same layout for both versions)
/// - V2 appends l1_state_bits (u16 LE) at the end
///
/// ## Byte Layout
/// | Field | V1 Offset | V1 Size | V2 Offset | V2 Size |
/// |-------|-----------|---------|-----------|---------|
/// | schema discriminant | 0 | 1 | 0 | 1 |
/// | bid_price_mantissa | 1 | 8 | 1 | 8 |
/// | ask_price_mantissa | 9 | 8 | 9 | 8 |
/// | bid_qty_mantissa | 17 | 8 | 17 | 8 |
/// | ask_qty_mantissa | 25 | 8 | 25 | 8 |
/// | price_exponent | 33 | 1 | 33 | 1 |
/// | qty_exponent | 34 | 1 | 34 | 1 |
/// | spread_bps_mantissa | 35 | 8 | 35 | 8 |
/// | book_ts_ns | 43 | 8 | 43 | 8 |
/// | l1_state_bits | - | - | 51 | 2 |
/// | **Total** | | **51** | | **53** |
fn encode_market_snapshot(buf: &mut Vec<u8>, snap: &MarketSnapshot) {
    // Schema version discriminant (uniform encoding)
    buf.push(snap.schema_version_byte());

    // Common V1 fields (same layout for both versions)
    encode_i64(buf, snap.bid_price_mantissa());
    encode_i64(buf, snap.ask_price_mantissa());
    encode_i64(buf, snap.bid_qty_mantissa());
    encode_i64(buf, snap.ask_qty_mantissa());
    encode_i8(buf, snap.price_exponent());
    encode_i8(buf, snap.qty_exponent());
    encode_i64(buf, snap.spread_bps_mantissa());
    encode_i64(buf, snap.book_ts_ns());

    // V2-specific: l1_state_bits
    if let MarketSnapshot::V2(v2) = snap {
        encode_u16(buf, v2.l1_state_bits);
    }
}

/// Encode serde_json::Value with presence marker.
///
/// For deterministic hashing, we convert to canonical JSON with sorted keys.
/// If the value is null/empty, we encode as absent.
///
/// **Important**: serde_json does NOT sort keys by default. We must explicitly
/// sort object keys for deterministic encoding.
fn encode_json_value(buf: &mut Vec<u8>, value: &serde_json::Value) {
    if value.is_null() {
        buf.push(0x00); // Absent
    } else {
        buf.push(0x01); // Present
        // Convert to canonical JSON with sorted keys
        let canonical = canonicalize_json(value);
        let json_str = serde_json::to_string(&canonical).unwrap_or_default();
        encode_string(buf, &json_str);
    }
}

/// Recursively sort all object keys in a JSON value for deterministic encoding.
fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            // Sort keys and recursively canonicalize values
            let mut sorted: Vec<_> = map.iter().collect();
            sorted.sort_by_key(|(k, _)| *k);
            let mut new_map = serde_json::Map::new();
            for (k, v) in sorted {
                new_map.insert(k.clone(), canonicalize_json(v));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            // Recursively canonicalize array elements (order is preserved)
            Value::Array(arr.iter().map(canonicalize_json).collect())
        }
        // Primitive values are returned as-is
        _ => value.clone(),
    }
}

/// Encode CorrelationContext with presence markers for each field.
fn encode_correlation_context(buf: &mut Vec<u8>, ctx: &CorrelationContext) {
    encode_optional_string(buf, &ctx.session_id);
    encode_optional_string(buf, &ctx.run_id);
    encode_optional_string(buf, &ctx.symbol);
    encode_optional_string(buf, &ctx.venue);
    encode_optional_string(buf, &ctx.strategy_id);
    encode_optional_uuid(buf, &ctx.decision_id);
    encode_optional_uuid(buf, &ctx.order_id);
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Create a deterministic test decision for testing (using V2 MarketSnapshot).
    fn make_test_decision(id: u8, direction: i8) -> DecisionEvent {
        // Use a fixed timestamp for determinism
        let ts = Utc.with_ymd_and_hms(2026, 1, 25, 12, 0, 0).unwrap();

        // Use a predictable UUID based on id
        let decision_id =
            Uuid::parse_str(&format!("00000000-0000-0000-0000-00000000000{:x}", id)).unwrap();

        DecisionEvent {
            ts,
            decision_id,
            strategy_id: "basis_capture".to_string(),
            symbol: "BTCUSDT".to_string(),
            decision_type: "entry".to_string(),
            direction,
            target_qty_mantissa: 1000000, // 0.01 BTC with exponent -8
            qty_exponent: -8,
            reference_price_mantissa: 8871660, // 88716.60 with exponent -2
            price_exponent: -2,
            // V2 snapshot with all fields present
            market_snapshot: MarketSnapshot::v2_all_present(
                8871650,             // bid_price_mantissa
                8871670,             // ask_price_mantissa
                10000000,            // bid_qty_mantissa
                10000000,            // ask_qty_mantissa
                -2,                  // price_exponent
                -8,                  // qty_exponent
                23,                  // spread_bps_mantissa (0.23 bps)
                1737799200000000000, // book_ts_ns
            ),
            // Fixed-point: 8500 with exponent -4 = 0.85
            confidence_mantissa: 8500,
            metadata: serde_json::Value::Null,
            ctx: CorrelationContext {
                session_id: Some("session-001".to_string()),
                run_id: Some("run-001".to_string()),
                symbol: Some("BTCUSDT".to_string()),
                venue: Some("binance".to_string()),
                strategy_id: Some("basis_capture".to_string()),
                decision_id: Some(decision_id),
                order_id: None,
            },
        }
    }

    #[test]
    fn test_identical_traces_hash_identically() {
        // Build two identical traces
        let decisions = vec![
            make_test_decision(1, 1),  // Long entry
            make_test_decision(2, 0),  // Neutral (exit)
            make_test_decision(3, -1), // Short entry
        ];

        let mut builder1 = DecisionTraceBuilder::new();
        let mut builder2 = DecisionTraceBuilder::new();

        for decision in &decisions {
            builder1.record(decision);
            builder2.record(decision);
        }

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        // Hashes must be identical
        assert_eq!(
            trace1.trace_hash, trace2.trace_hash,
            "Identical decision sequences must produce identical trace hashes"
        );

        // Verify using verify_replay_parity
        let result = verify_replay_parity(&trace1, &trace2);
        assert!(
            matches!(result, ReplayParityResult::Match),
            "verify_replay_parity should return Match for identical traces"
        );
    }

    #[test]
    fn test_different_direction_detected_at_correct_index() {
        // Build original trace
        let mut original_decisions = vec![
            make_test_decision(1, 1),  // Long entry
            make_test_decision(2, 0),  // Neutral (exit)
            make_test_decision(3, -1), // Short entry
        ];

        let mut builder_orig = DecisionTraceBuilder::new();
        for decision in &original_decisions {
            builder_orig.record(decision);
        }
        let original = builder_orig.finalize();

        // Build replay trace with different direction at index 1
        original_decisions[1].direction = 1; // Changed from 0 to 1

        let mut builder_replay = DecisionTraceBuilder::new();
        for decision in &original_decisions {
            builder_replay.record(decision);
        }
        let replay = builder_replay.finalize();

        // Hashes must differ
        assert_ne!(
            original.trace_hash, replay.trace_hash,
            "Different decisions must produce different trace hashes"
        );

        // Verify divergence is detected at correct index
        let result = verify_replay_parity(&original, &replay);
        match result {
            ReplayParityResult::Divergence { index, reason, .. } => {
                assert_eq!(index, 1, "Divergence should be detected at index 1");
                assert!(
                    reason.contains("direction"),
                    "Reason should mention direction field, got: {}",
                    reason
                );
            }
            _ => panic!("Expected Divergence result, got: {:?}", result),
        }
    }

    #[test]
    fn test_different_timestamp_detected() {
        let decision1 = make_test_decision(1, 1);
        let mut decision2 = make_test_decision(1, 1);

        // Modify timestamp in decision2
        decision2.ts = Utc.with_ymd_and_hms(2026, 1, 25, 12, 0, 1).unwrap(); // +1 second

        let mut builder1 = DecisionTraceBuilder::new();
        let mut builder2 = DecisionTraceBuilder::new();

        builder1.record(&decision1);
        builder2.record(&decision2);

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        // Verify divergence
        let result = verify_replay_parity(&trace1, &trace2);
        match result {
            ReplayParityResult::Divergence { index, reason, .. } => {
                assert_eq!(index, 0);
                assert!(reason.contains("timestamp"));
            }
            _ => panic!("Expected Divergence result"),
        }
    }

    #[test]
    fn test_different_qty_mantissa_detected() {
        let decision1 = make_test_decision(1, 1);
        let mut decision2 = make_test_decision(1, 1);

        // Modify quantity in decision2
        decision2.target_qty_mantissa = 2000000; // Changed from 1000000

        let mut builder1 = DecisionTraceBuilder::new();
        let mut builder2 = DecisionTraceBuilder::new();

        builder1.record(&decision1);
        builder2.record(&decision2);

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        let result = verify_replay_parity(&trace1, &trace2);
        match result {
            ReplayParityResult::Divergence { index, reason, .. } => {
                assert_eq!(index, 0);
                assert!(reason.contains("target_qty_mantissa"));
            }
            _ => panic!("Expected Divergence result"),
        }
    }

    #[test]
    fn test_length_mismatch_detected() {
        let decision = make_test_decision(1, 1);

        let mut builder1 = DecisionTraceBuilder::new();
        let mut builder2 = DecisionTraceBuilder::new();

        builder1.record(&decision);
        builder1.record(&make_test_decision(2, -1));

        builder2.record(&decision);
        // builder2 has only 1 decision

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        let result = verify_replay_parity(&trace1, &trace2);
        match result {
            ReplayParityResult::LengthMismatch {
                original_len,
                replay_len,
            } => {
                assert_eq!(original_len, 2);
                assert_eq!(replay_len, 1);
            }
            _ => panic!("Expected LengthMismatch result"),
        }
    }

    #[test]
    fn test_empty_traces_match() {
        let builder1 = DecisionTraceBuilder::new();
        let builder2 = DecisionTraceBuilder::new();

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        let result = verify_replay_parity(&trace1, &trace2);
        assert!(matches!(result, ReplayParityResult::Match));
    }

    #[test]
    fn test_canonical_bytes_deterministic() {
        let decision = make_test_decision(1, 1);

        // Encode twice
        let bytes1 = canonical_bytes(&decision);
        let bytes2 = canonical_bytes(&decision);

        assert_eq!(
            bytes1, bytes2,
            "canonical_bytes must produce identical output for the same input"
        );
    }

    #[test]
    fn test_hash_hex_format() {
        let mut builder = DecisionTraceBuilder::new();
        builder.record(&make_test_decision(1, 1));
        let trace = builder.finalize();

        let hex = trace.hash_hex();
        assert_eq!(hex.len(), 64, "SHA-256 hex should be 64 characters");
        assert!(
            hex.chars().all(|c| c.is_ascii_hexdigit()),
            "Hash hex should only contain hex digits"
        );
    }

    #[test]
    fn test_metadata_affects_hash() {
        let mut decision1 = make_test_decision(1, 1);
        let mut decision2 = make_test_decision(1, 1);

        decision1.metadata = serde_json::json!({"edge_bps": 5.0});
        decision2.metadata = serde_json::json!({"edge_bps": 6.0});

        let mut builder1 = DecisionTraceBuilder::new();
        let mut builder2 = DecisionTraceBuilder::new();

        builder1.record(&decision1);
        builder2.record(&decision2);

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        assert_ne!(
            trace1.trace_hash, trace2.trace_hash,
            "Different metadata should produce different hashes"
        );
    }

    #[test]
    fn test_correlation_context_affects_hash() {
        let mut decision1 = make_test_decision(1, 1);
        let mut decision2 = make_test_decision(1, 1);

        decision1.ctx.session_id = Some("session-001".to_string());
        decision2.ctx.session_id = Some("session-002".to_string());

        let mut builder1 = DecisionTraceBuilder::new();
        let mut builder2 = DecisionTraceBuilder::new();

        builder1.record(&decision1);
        builder2.record(&decision2);

        let trace1 = builder1.finalize();
        let trace2 = builder2.finalize();

        assert_ne!(
            trace1.trace_hash, trace2.trace_hash,
            "Different correlation context should produce different hashes"
        );
    }

    #[test]
    fn test_confidence_mantissa_deterministic() {
        // Test that confidence_mantissa is encoded deterministically
        let mut d1 = make_test_decision(1, 1);
        let mut d2 = make_test_decision(1, 1);

        // Same confidence as fixed-point
        d1.confidence_mantissa = 8500; // 0.85
        d2.confidence_mantissa = 8500; // 0.85

        let bytes1 = canonical_bytes(&d1);
        let bytes2 = canonical_bytes(&d2);

        assert_eq!(
            bytes1, bytes2,
            "Same confidence_mantissa should produce same bytes"
        );

        // Different confidence should produce different bytes
        d2.confidence_mantissa = 9000; // 0.90

        let bytes3 = canonical_bytes(&d2);
        assert_ne!(
            bytes1, bytes3,
            "Different confidence_mantissa should produce different bytes"
        );
    }

    #[test]
    fn test_spread_bps_mantissa_deterministic() {
        // Test that spread_bps_mantissa is encoded deterministically
        // Create decisions with specific spread values
        let d1 = make_test_decision_with_spread(1, 1, 523); // 5.23 bps
        let d2 = make_test_decision_with_spread(1, 1, 523); // 5.23 bps

        let bytes1 = canonical_bytes(&d1);
        let bytes2 = canonical_bytes(&d2);

        assert_eq!(
            bytes1, bytes2,
            "Same spread_bps_mantissa should produce same bytes"
        );

        // Different spread should produce different bytes
        let d3 = make_test_decision_with_spread(1, 1, 600); // 6.00 bps

        let bytes3 = canonical_bytes(&d3);
        assert_ne!(
            bytes1, bytes3,
            "Different spread_bps_mantissa should produce different bytes"
        );
    }

    /// Helper to create test decision with specific spread_bps_mantissa.
    fn make_test_decision_with_spread(id: u8, direction: i8, spread_bps: i64) -> DecisionEvent {
        let ts = Utc.with_ymd_and_hms(2026, 1, 25, 12, 0, 0).unwrap();
        let decision_id =
            Uuid::parse_str(&format!("00000000-0000-0000-0000-00000000000{:x}", id)).unwrap();

        DecisionEvent {
            ts,
            decision_id,
            strategy_id: "basis_capture".to_string(),
            symbol: "BTCUSDT".to_string(),
            decision_type: "entry".to_string(),
            direction,
            target_qty_mantissa: 1000000,
            qty_exponent: -8,
            reference_price_mantissa: 8871660,
            price_exponent: -2,
            market_snapshot: MarketSnapshot::v2_all_present(
                8871650,
                8871670,
                10000000,
                10000000,
                -2,
                -8,
                spread_bps,
                1737799200000000000,
            ),
            confidence_mantissa: 8500,
            metadata: serde_json::Value::Null,
            ctx: CorrelationContext {
                session_id: Some("session-001".to_string()),
                run_id: Some("run-001".to_string()),
                symbol: Some("BTCUSDT".to_string()),
                venue: Some("binance".to_string()),
                strategy_id: Some("basis_capture".to_string()),
                decision_id: Some(decision_id),
                order_id: None,
            },
        }
    }

    #[test]
    fn test_trace_encoding_version_is_current() {
        // Canonical literal version test - update this when ENCODING_VERSION changes
        assert_eq!(
            ENCODING_VERSION, 0x03,
            "Update this test when encoding version changes"
        );

        let decision = make_test_decision(1, 1);
        let bytes = canonical_bytes(&decision);

        // First byte should be version
        assert_eq!(
            bytes[0], ENCODING_VERSION,
            "First byte should be encoding version"
        );
    }

    #[test]
    fn test_version_mismatch_detection() {
        // Build a trace with current version
        let mut builder = DecisionTraceBuilder::new();
        builder.record(&make_test_decision(1, 1));
        let trace = builder.finalize();

        // Verify trace has correct encoding version
        assert_eq!(trace.encoding_version, ENCODING_VERSION);

        // Simulate loading a trace with different version
        let mut old_trace = trace.clone();
        old_trace.encoding_version = 0x02; // Simulate v2 trace

        // Version mismatch should be detectable
        assert_ne!(
            old_trace.encoding_version, ENCODING_VERSION,
            "Should detect version mismatch between v2 and v3"
        );

        // In a real system, we would reject traces with mismatched versions
        // or have migration logic. For now, we just detect the mismatch.
        // Replay parity MUST be evaluated within the same encoding version.
    }

    #[test]
    fn test_fixed_point_conversion_helpers() {
        // Test DecisionEvent::confidence_from_f64
        assert_eq!(DecisionEvent::confidence_from_f64(1.0), 10000);
        assert_eq!(DecisionEvent::confidence_from_f64(0.85), 8500);
        assert_eq!(DecisionEvent::confidence_from_f64(0.0), 0);

        // Test MarketSnapshot::spread_bps_from_f64
        assert_eq!(MarketSnapshot::spread_bps_from_f64(5.23), 523);
        assert_eq!(MarketSnapshot::spread_bps_from_f64(1.0), 100);
        assert_eq!(MarketSnapshot::spread_bps_from_f64(0.0), 0);

        // Test round-trip: f64 -> mantissa -> f64
        let confidence = 0.8765;
        let mantissa = DecisionEvent::confidence_from_f64(confidence);
        let decision = make_test_decision(1, 1);
        let mut d = decision;
        d.confidence_mantissa = mantissa;
        let recovered = d.confidence_f64();
        assert!(
            (recovered - confidence).abs() < 0.0001,
            "Confidence round-trip should preserve value"
        );
    }

    #[test]
    fn test_no_floats_in_canonical_encoding() {
        // Verify that canonical_bytes produces the same output
        // regardless of any floating-point computation order differences.
        // This is guaranteed by using only i64 mantissas.

        let decision = make_test_decision(1, 1);

        // Encode multiple times
        let bytes1 = canonical_bytes(&decision);
        let bytes2 = canonical_bytes(&decision);
        let bytes3 = canonical_bytes(&decision);

        assert_eq!(bytes1, bytes2);
        assert_eq!(bytes2, bytes3);

        // Verify version byte
        assert_eq!(bytes1[0], 0x03, "Version should be 0x03");

        // Build trace and verify hash stability
        let mut b1 = DecisionTraceBuilder::new();
        let mut b2 = DecisionTraceBuilder::new();
        b1.record(&decision);
        b2.record(&decision);

        let t1 = b1.finalize();
        let t2 = b2.finalize();

        assert_eq!(
            t1.trace_hash, t2.trace_hash,
            "Hashes must be identical without floats"
        );
    }

    // =========================================================================
    // CANONICAL BYTES LAYOUT TESTS (Doctrine: No Silent Poisoning)
    // =========================================================================

    /// Helper to encode a MarketSnapshot in isolation for layout testing.
    fn encode_snapshot_bytes(snap: &MarketSnapshot) -> Vec<u8> {
        let mut buf = Vec::new();
        // Schema version discriminant
        buf.push(snap.schema_version_byte());
        // Common fields
        buf.extend_from_slice(&snap.bid_price_mantissa().to_le_bytes());
        buf.extend_from_slice(&snap.ask_price_mantissa().to_le_bytes());
        buf.extend_from_slice(&snap.bid_qty_mantissa().to_le_bytes());
        buf.extend_from_slice(&snap.ask_qty_mantissa().to_le_bytes());
        buf.push(snap.price_exponent() as u8);
        buf.push(snap.qty_exponent() as u8);
        buf.extend_from_slice(&snap.spread_bps_mantissa().to_le_bytes());
        buf.extend_from_slice(&snap.book_ts_ns().to_le_bytes());
        // V2-specific
        if let MarketSnapshot::V2(v2) = snap {
            buf.extend_from_slice(&v2.l1_state_bits.to_le_bytes());
        }
        buf
    }

    #[test]
    fn test_snapshot_v1_uniform_prefix() {
        // V1 canonical bytes MUST start with 0x01 discriminant
        use quantlaxmi_models::MarketSnapshotV1;

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

        let bytes = encode_snapshot_bytes(&v1);

        // First byte is version discriminant
        assert_eq!(bytes[0], 0x01, "V1 must start with 0x01 discriminant");

        // Total size: 1 (discriminant) + 50 (fields) = 51 bytes
        assert_eq!(bytes.len(), 51, "V1 should be exactly 51 bytes");
    }

    #[test]
    fn test_snapshot_v2_uniform_prefix() {
        // V2 canonical bytes MUST start with 0x02 discriminant
        use quantlaxmi_models::L1_ALL_VALUE;

        let v2 = MarketSnapshot::v2_all_present(1000, 1001, 500, 600, -2, -8, 10, 1234567890);

        let bytes = encode_snapshot_bytes(&v2);

        // First byte is version discriminant
        assert_eq!(bytes[0], 0x02, "V2 must start with 0x02 discriminant");

        // Total size: 1 (discriminant) + 50 (fields) + 2 (l1_state_bits) = 53 bytes
        assert_eq!(bytes.len(), 53, "V2 should be exactly 53 bytes");

        // Last 2 bytes are l1_state_bits (little-endian)
        let state_bits = u16::from_le_bytes([bytes[51], bytes[52]]);
        assert_eq!(
            state_bits, L1_ALL_VALUE,
            "l1_state_bits should be L1_ALL_VALUE"
        );
    }

    #[test]
    fn test_snapshot_byte_layout_offsets() {
        // Verify byte layout matches the spec table:
        // | Field               | V1/V2 Offset | Size |
        // | schema discriminant | 0            | 1    |
        // | bid_price_mantissa  | 1            | 8    |
        // | ask_price_mantissa  | 9            | 8    |
        // | bid_qty_mantissa    | 17           | 8    |
        // | ask_qty_mantissa    | 25           | 8    |
        // | price_exponent      | 33           | 1    |
        // | qty_exponent        | 34           | 1    |
        // | spread_bps_mantissa | 35           | 8    |
        // | book_ts_ns          | 43           | 8    |
        // | l1_state_bits (V2)  | 51           | 2    |

        let v2 = MarketSnapshot::v2_with_states(
            0x0102030405060708_i64, // bid_price (recognizable pattern)
            0x1112131415161718_i64, // ask_price
            0x2122232425262728_i64, // bid_qty
            0x3132333435363738_i64, // ask_qty
            -2,                     // price_exp
            -8,                     // qty_exp
            0x4142434445464748_i64, // spread_bps
            0x5152535455565758_i64, // book_ts_ns
            0xABCD,                 // l1_state_bits
        );

        let bytes = encode_snapshot_bytes(&v2);

        // Offset 0: schema discriminant
        assert_eq!(bytes[0], 0x02);

        // Offset 1-8: bid_price_mantissa (little-endian)
        assert_eq!(&bytes[1..9], &0x0102030405060708_i64.to_le_bytes());

        // Offset 9-16: ask_price_mantissa
        assert_eq!(&bytes[9..17], &0x1112131415161718_i64.to_le_bytes());

        // Offset 17-24: bid_qty_mantissa
        assert_eq!(&bytes[17..25], &0x2122232425262728_i64.to_le_bytes());

        // Offset 25-32: ask_qty_mantissa
        assert_eq!(&bytes[25..33], &0x3132333435363738_i64.to_le_bytes());

        // Offset 33: price_exponent (i8 as u8 = 0xFE for -2)
        assert_eq!(bytes[33], (-2_i8) as u8);

        // Offset 34: qty_exponent (i8 as u8 = 0xF8 for -8)
        assert_eq!(bytes[34], (-8_i8) as u8);

        // Offset 35-42: spread_bps_mantissa
        assert_eq!(&bytes[35..43], &0x4142434445464748_i64.to_le_bytes());

        // Offset 43-50: book_ts_ns
        assert_eq!(&bytes[43..51], &0x5152535455565758_i64.to_le_bytes());

        // Offset 51-52: l1_state_bits (V2 only)
        assert_eq!(&bytes[51..53], &0xABCD_u16.to_le_bytes());
    }

    #[test]
    fn test_v1_v2_digests_differ() {
        // Same field values but V1 vs V2 MUST produce different digests
        // because the schema discriminant differs (0x01 vs 0x02)
        use quantlaxmi_models::MarketSnapshotV1;

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

        let v1_bytes = encode_snapshot_bytes(&v1);
        let v2_bytes = encode_snapshot_bytes(&v2);

        // Schemas must differ
        assert_eq!(v1_bytes[0], 0x01);
        assert_eq!(v2_bytes[0], 0x02);

        // Common fields (offset 1-50) should be identical
        assert_eq!(
            &v1_bytes[1..51],
            &v2_bytes[1..51],
            "Common V1 fields should be byte-identical"
        );

        // But V2 has extra bytes for l1_state_bits
        assert_eq!(v1_bytes.len(), 51);
        assert_eq!(v2_bytes.len(), 53);

        // Full byte sequences must differ
        assert_ne!(v1_bytes, v2_bytes, "V1 and V2 canonical bytes must differ");
    }

    #[test]
    fn test_presence_bits_affect_digest() {
        // Same mantissa values but different presence states MUST produce
        // different digests. This is the "No Silent Poisoning" doctrine test.
        use quantlaxmi_models::{FieldState, build_l1_state_bits};

        // Scenario: bid_qty=0 with state Value vs bid_qty=0 with state Absent
        let bits_value = build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Value, // qty=0 but vendor sent it
            FieldState::Value,
        );

        let bits_absent = build_l1_state_bits(
            FieldState::Value,
            FieldState::Value,
            FieldState::Absent, // qty not sent (default 0)
            FieldState::Value,
        );

        let snap_value =
            MarketSnapshot::v2_with_states(1000, 1001, 0, 600, -2, -8, 10, 1234567890, bits_value);

        let snap_absent =
            MarketSnapshot::v2_with_states(1000, 1001, 0, 600, -2, -8, 10, 1234567890, bits_absent);

        let bytes_value = encode_snapshot_bytes(&snap_value);
        let bytes_absent = encode_snapshot_bytes(&snap_absent);

        // l1_state_bits (last 2 bytes) must differ
        let state_value = u16::from_le_bytes([bytes_value[51], bytes_value[52]]);
        let state_absent = u16::from_le_bytes([bytes_absent[51], bytes_absent[52]]);

        assert_ne!(
            state_value, state_absent,
            "Different presence states must produce different l1_state_bits"
        );

        // Full bytes must differ
        assert_ne!(
            bytes_value, bytes_absent,
            "Different presence states must produce different canonical bytes"
        );
    }

    #[test]
    fn test_market_snapshot_v1_v2_divergence_detected() {
        // If original uses V1 and replay uses V2, parity should detect schema divergence
        use quantlaxmi_models::MarketSnapshotV1;

        let ts = Utc.with_ymd_and_hms(2026, 1, 25, 12, 0, 0).unwrap();
        let decision_id = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();

        // Original with V1 snapshot
        let original = DecisionEvent {
            ts,
            decision_id,
            strategy_id: "test".to_string(),
            symbol: "BTCUSDT".to_string(),
            decision_type: "entry".to_string(),
            direction: 1,
            target_qty_mantissa: 1000,
            qty_exponent: -8,
            reference_price_mantissa: 88716,
            price_exponent: -2,
            market_snapshot: MarketSnapshot::V1(MarketSnapshotV1 {
                bid_price_mantissa: 88715,
                ask_price_mantissa: 88717,
                bid_qty_mantissa: 10000,
                ask_qty_mantissa: 10000,
                price_exponent: -2,
                qty_exponent: -8,
                spread_bps_mantissa: 23,
                book_ts_ns: 1737799200000000000,
            }),
            confidence_mantissa: 8500,
            metadata: serde_json::Value::Null,
            ctx: CorrelationContext::default(),
        };

        // Replay with V2 snapshot (same field values)
        let replay = DecisionEvent {
            market_snapshot: MarketSnapshot::v2_all_present(
                88715,
                88717,
                10000,
                10000,
                -2,
                -8,
                23,
                1737799200000000000,
            ),
            ..original.clone()
        };

        let mut builder_orig = DecisionTraceBuilder::new();
        let mut builder_replay = DecisionTraceBuilder::new();

        builder_orig.record(&original);
        builder_replay.record(&replay);

        let trace_orig = builder_orig.finalize();
        let trace_replay = builder_replay.finalize();

        // Hashes must differ
        assert_ne!(
            trace_orig.trace_hash, trace_replay.trace_hash,
            "V1 vs V2 snapshots must produce different hashes"
        );

        // Divergence should be detected with schema mismatch
        let result = verify_replay_parity(&trace_orig, &trace_replay);
        match result {
            ReplayParityResult::Divergence { reason, .. } => {
                assert!(
                    reason.contains("schema"),
                    "Should detect schema version mismatch, got: {}",
                    reason
                );
            }
            _ => panic!("Expected Divergence, got {:?}", result),
        }
    }
}
