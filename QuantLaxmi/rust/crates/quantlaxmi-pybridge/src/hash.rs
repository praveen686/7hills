//! Canonical hash computation exposed to Python.
//!
//! Produces the same SHA-256 digest as the Rust WAL, ensuring
//! Python-generated events can be verified against the Rust hash chain.

use pyo3::prelude::*;
use sha2::{Digest, Sha256};

/// Compute SHA-256 of canonical bytes (matching Rust WAL exactly).
///
/// Python usage:
///     digest = canonical_digest(b'{"event_type":"signal",...}')
///     # Returns hex string like "a1b2c3..."
#[pyfunction]
pub fn canonical_digest(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute hash-chain link: SHA-256(prev_digest || "||" || current_bytes).
///
/// Matches Python `hashing.py`: `prev_hash.encode("ascii") + b"||" + line.encode("utf-8")`
/// Used for WAL integrity verification. The chain resets to GENESIS
/// on daily rotation.
///
/// Python usage:
///     link = hash_chain_link("GENESIS", b'{"event_type":"signal",...}')
#[pyfunction]
pub fn hash_chain_link(prev_digest: &str, current_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_digest.as_bytes());
    hasher.update(b"||");
    hasher.update(current_bytes);
    hex::encode(hasher.finalize())
}

/// Verify a hash chain: given a list of (digest, bytes) pairs,
/// check that each digest = SHA-256(prev_digest || bytes).
///
/// Returns (is_valid, first_broken_index_or_none).
#[pyfunction]
pub fn verify_hash_chain(
    genesis: &str,
    chain: Vec<(String, Vec<u8>)>,
) -> (bool, Option<usize>) {
    let mut prev = genesis.to_string();
    for (i, (expected_digest, bytes)) in chain.iter().enumerate() {
        let computed = hash_chain_link(&prev, bytes);
        if &computed != expected_digest {
            return (false, Some(i));
        }
        prev = computed;
    }
    (true, None)
}
