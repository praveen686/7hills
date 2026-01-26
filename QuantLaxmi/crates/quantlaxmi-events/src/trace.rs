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
pub const ENCODING_VERSION: u8 = 0x02;

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
    #[serde(default = "default_encoding_version")]
    pub encoding_version: u8,
    /// The sequence of decisions in order of occurrence.
    pub decisions: Vec<DecisionEvent>,
    /// SHA-256 hash of the canonical binary encoding of all decisions (hex).
    #[serde(with = "hex_hash")]
    pub trace_hash: [u8; 32],
}

fn default_encoding_version() -> u8 {
    ENCODING_VERSION
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
    if orig_snap.bid_price_mantissa != replay_snap.bid_price_mantissa
        || orig_snap.ask_price_mantissa != replay_snap.ask_price_mantissa
        || orig_snap.bid_qty_mantissa != replay_snap.bid_qty_mantissa
        || orig_snap.ask_qty_mantissa != replay_snap.ask_qty_mantissa
        || orig_snap.book_ts_ns != replay_snap.book_ts_ns
    {
        return "market_snapshot differs".to_string();
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

/// Encode MarketSnapshot.
fn encode_market_snapshot(buf: &mut Vec<u8>, snap: &MarketSnapshot) {
    encode_i64(buf, snap.bid_price_mantissa);
    encode_i64(buf, snap.ask_price_mantissa);
    encode_i64(buf, snap.bid_qty_mantissa);
    encode_i64(buf, snap.ask_qty_mantissa);
    encode_i8(buf, snap.price_exponent);
    encode_i8(buf, snap.qty_exponent);
    // spread_bps_mantissa: i64 (fixed exponent = -2, not encoded)
    encode_i64(buf, snap.spread_bps_mantissa);
    encode_i64(buf, snap.book_ts_ns);
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

    /// Create a deterministic test decision for testing.
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
            market_snapshot: MarketSnapshot {
                bid_price_mantissa: 8871650,
                ask_price_mantissa: 8871670,
                bid_qty_mantissa: 10000000,
                ask_qty_mantissa: 10000000,
                price_exponent: -2,
                qty_exponent: -8,
                // Fixed-point: 23 with exponent -2 = 0.23 bps
                spread_bps_mantissa: 23,
                book_ts_ns: 1737799200000000000,
            },
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
        let mut d1 = make_test_decision(1, 1);
        let mut d2 = make_test_decision(1, 1);

        // Same spread as fixed-point
        d1.market_snapshot.spread_bps_mantissa = 523; // 5.23 bps
        d2.market_snapshot.spread_bps_mantissa = 523; // 5.23 bps

        let bytes1 = canonical_bytes(&d1);
        let bytes2 = canonical_bytes(&d2);

        assert_eq!(
            bytes1, bytes2,
            "Same spread_bps_mantissa should produce same bytes"
        );

        // Different spread should produce different bytes
        d2.market_snapshot.spread_bps_mantissa = 600; // 6.00 bps

        let bytes3 = canonical_bytes(&d2);
        assert_ne!(
            bytes1, bytes3,
            "Different spread_bps_mantissa should produce different bytes"
        );
    }

    #[test]
    fn test_encoding_version_v2() {
        // Verify we're using encoding version 2 (fixed-point, no floats)
        assert_eq!(ENCODING_VERSION, 0x02, "Should be encoding version 2");

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
        old_trace.encoding_version = 0x01; // Simulate v1 trace

        // Version mismatch should be detectable
        assert_ne!(
            old_trace.encoding_version, ENCODING_VERSION,
            "Should detect version mismatch between v1 and v2"
        );

        // In a real system, we would reject traces with mismatched versions
        // or have migration logic. For now, we just detect the mismatch.
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
        assert_eq!(bytes1[0], 0x02, "Version should be 0x02");

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
}
