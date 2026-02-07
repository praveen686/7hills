//! Deterministic ID generation for WAL hash stability.
//!
//! All IDs used in hashed records (DecisionEvent, etc.) must be deterministic
//! to ensure replay parity. This module provides SHA-256 based ID generation
//! that is stable across runs with identical inputs.
//!
//! # ID Scheme
//!
//! ```text
//! run_id = sha256("qlx.run.v1" || strategy_id || sorted(segment_ids) || params_json)
//! decision_id = sha256("qlx.decision.v1" || run_id || symbol || event_index || decision_seq || decision_type || direction)
//! order_id = sha256("qlx.order.v1" || decision_id || intent_index)
//! ```

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Generate a deterministic run_id from strategy, segments, and parameters.
///
/// # Arguments
/// * `strategy_id` - Strategy identifier (e.g., "slr", "fam1")
/// * `segment_ids` - List of segment identifiers (will be sorted)
/// * `params_json` - Canonical JSON of grid parameters (must be stable key order)
///
/// # Returns
/// Hex-encoded SHA-256 hash (64 characters)
pub fn compute_run_id(strategy_id: &str, segment_ids: &[String], params_json: &str) -> String {
    let mut hasher = Sha256::new();

    // Domain separator
    hasher.update(b"qlx.run.v1");

    // Strategy ID (length-prefixed)
    hasher.update((strategy_id.len() as u32).to_le_bytes());
    hasher.update(strategy_id.as_bytes());

    // Sorted segment IDs
    let mut sorted_segments = segment_ids.to_vec();
    sorted_segments.sort();
    hasher.update((sorted_segments.len() as u32).to_le_bytes());
    for seg in &sorted_segments {
        hasher.update((seg.len() as u32).to_le_bytes());
        hasher.update(seg.as_bytes());
    }

    // Params JSON (length-prefixed)
    hasher.update((params_json.len() as u32).to_le_bytes());
    hasher.update(params_json.as_bytes());

    hex::encode(hasher.finalize())
}

/// Generate a deterministic decision_id from run context and decision sequence.
///
/// # Arguments
/// * `run_id` - The deterministic run ID (hex string)
/// * `symbol` - Trading symbol (e.g., "BTCUSDT")
/// * `event_index` - Index of the event that triggered this decision (0-based)
/// * `decision_seq` - Sequence number of this decision within the run (0-based)
/// * `decision_type` - Type of decision (e.g., "entry", "exit", "order")
/// * `direction` - Direction: 1 for long, -1 for short, 0 for neutral
///
/// # Returns
/// 32-byte hash as `[u8; 32]`
pub fn compute_decision_id(
    run_id: &str,
    symbol: &str,
    event_index: u64,
    decision_seq: u64,
    decision_type: &str,
    direction: i8,
) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Domain separator
    hasher.update(b"qlx.decision.v1");

    // Run ID (hex string, length-prefixed)
    hasher.update((run_id.len() as u32).to_le_bytes());
    hasher.update(run_id.as_bytes());

    // Symbol (length-prefixed)
    hasher.update((symbol.len() as u32).to_le_bytes());
    hasher.update(symbol.as_bytes());

    // Event index (little-endian u64)
    hasher.update(event_index.to_le_bytes());

    // Decision sequence (little-endian u64)
    hasher.update(decision_seq.to_le_bytes());

    // Decision type (length-prefixed)
    hasher.update((decision_type.len() as u32).to_le_bytes());
    hasher.update(decision_type.as_bytes());

    // Direction (i8)
    hasher.update([direction as u8]);

    hasher.finalize().into()
}

/// Generate a deterministic order_id from decision_id and intent index.
///
/// # Arguments
/// * `decision_id` - The parent decision ID (32 bytes)
/// * `intent_index` - Index of this intent within the decision (0-based)
///
/// # Returns
/// 32-byte hash as `[u8; 32]`
pub fn compute_order_id(decision_id: &[u8; 32], intent_index: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Domain separator
    hasher.update(b"qlx.order.v1");

    // Decision ID (32 bytes)
    hasher.update(decision_id);

    // Intent index (little-endian u32)
    hasher.update(intent_index.to_le_bytes());

    hasher.finalize().into()
}

/// Convert 32-byte hash to hex string.
pub fn hash_to_hex(hash: &[u8; 32]) -> String {
    hex::encode(hash)
}

/// Convert 32-byte hash to a deterministic UUID (uses first 16 bytes).
///
/// This creates a UUID that can be used in existing code expecting Uuid types,
/// while maintaining determinism. The UUID version byte is preserved from the
/// hash (not forced to v4) to ensure true determinism.
pub fn hash_to_uuid(hash: &[u8; 32]) -> Uuid {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hash[..16]);
    Uuid::from_bytes(bytes)
}

/// Convert hex string to 32-byte hash.
pub fn hex_to_hash(hex_str: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(hex_str).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Some(arr)
}

/// Serialize parameters to canonical JSON (sorted keys).
///
/// Uses BTreeMap to ensure deterministic key ordering.
pub fn params_to_canonical_json(params: &BTreeMap<String, serde_json::Value>) -> String {
    serde_json::to_string(params).unwrap_or_else(|_| "{}".to_string())
}

/// Counter state for deterministic ID generation during a backtest run.
#[derive(Debug, Clone, Default)]
pub struct DeterministicIdState {
    /// Current event index (incremented per processed event)
    pub event_index: u64,
    /// Current decision sequence (incremented per emitted decision)
    pub decision_seq: u64,
    /// The deterministic run_id for this run
    pub run_id: String,
}

impl DeterministicIdState {
    /// Create a new state with the given run_id.
    pub fn new(run_id: String) -> Self {
        Self {
            event_index: 0,
            decision_seq: 0,
            run_id,
        }
    }

    /// Increment event index (call once per processed replay event).
    pub fn next_event(&mut self) {
        self.event_index += 1;
    }

    /// Generate the next decision_id and increment the sequence.
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol
    /// * `decision_type` - Type of decision
    /// * `direction` - Direction: 1 for long, -1 for short
    ///
    /// # Returns
    /// Tuple of (decision_id bytes, decision_id hex string, decision_id as Uuid)
    pub fn next_decision_id(
        &mut self,
        symbol: &str,
        decision_type: &str,
        direction: i8,
    ) -> ([u8; 32], String, Uuid) {
        let id = compute_decision_id(
            &self.run_id,
            symbol,
            self.event_index,
            self.decision_seq,
            decision_type,
            direction,
        );
        self.decision_seq += 1;
        (id, hash_to_hex(&id), hash_to_uuid(&id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_id_determinism() {
        let run_id_1 = compute_run_id(
            "slr",
            &["seg_a".to_string(), "seg_b".to_string()],
            r#"{"alpha":0.5}"#,
        );
        let run_id_2 = compute_run_id(
            "slr",
            &["seg_b".to_string(), "seg_a".to_string()], // Different order
            r#"{"alpha":0.5}"#,
        );

        // Should be same because segments are sorted
        assert_eq!(run_id_1, run_id_2);
        assert_eq!(run_id_1.len(), 64); // Hex SHA-256
    }

    #[test]
    fn test_run_id_differs_with_different_params() {
        let run_id_1 = compute_run_id("slr", &["seg_a".to_string()], r#"{"alpha":0.5}"#);
        let run_id_2 = compute_run_id("slr", &["seg_a".to_string()], r#"{"alpha":0.6}"#);

        assert_ne!(run_id_1, run_id_2);
    }

    #[test]
    fn test_decision_id_determinism() {
        let id1 = compute_decision_id("run123", "BTCUSDT", 100, 5, "entry", 1);
        let id2 = compute_decision_id("run123", "BTCUSDT", 100, 5, "entry", 1);

        assert_eq!(id1, id2);
    }

    #[test]
    fn test_decision_id_differs_with_sequence() {
        let id1 = compute_decision_id("run123", "BTCUSDT", 100, 5, "entry", 1);
        let id2 = compute_decision_id("run123", "BTCUSDT", 100, 6, "entry", 1);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_order_id_determinism() {
        let decision_id = [0u8; 32];
        let order_id_1 = compute_order_id(&decision_id, 0);
        let order_id_2 = compute_order_id(&decision_id, 0);

        assert_eq!(order_id_1, order_id_2);
    }

    #[test]
    fn test_order_id_differs_with_intent_index() {
        let decision_id = [0u8; 32];
        let order_id_1 = compute_order_id(&decision_id, 0);
        let order_id_2 = compute_order_id(&decision_id, 1);

        assert_ne!(order_id_1, order_id_2);
    }

    #[test]
    fn test_id_state_increments() {
        let mut state = DeterministicIdState::new("test_run".to_string());

        assert_eq!(state.event_index, 0);
        assert_eq!(state.decision_seq, 0);

        state.next_event();
        assert_eq!(state.event_index, 1);

        let (id1, _, uuid1) = state.next_decision_id("BTCUSDT", "entry", 1);
        assert_eq!(state.decision_seq, 1);

        let (id2, _, uuid2) = state.next_decision_id("BTCUSDT", "entry", 1);
        assert_eq!(state.decision_seq, 2);

        // Different sequences should produce different IDs
        assert_ne!(id1, id2);
        assert_ne!(uuid1, uuid2);
    }

    #[test]
    fn test_hex_conversion_roundtrip() {
        let original = [42u8; 32];
        let hex_str = hash_to_hex(&original);
        let recovered = hex_to_hash(&hex_str).unwrap();

        assert_eq!(original, recovered);
    }
}
