//! Canonical binary encoding for deterministic config hashing.
//!
//! ## Why Not JSON?
//! JSON text hashing is risky for determinism:
//! - Floats serialize inconsistently (1 vs 1.0, scientific notation)
//! - Optional fields can be omitted vs defaulted
//! - Serde versions can change formatting
//!
//! ## Canonical Encoding Rules
//! - First byte: `CONFIG_ENCODING_VERSION` (for forward compatibility)
//! - Fields encoded in fixed order (struct definition order)
//! - Integers as little-endian fixed-width bytes
//! - Option<T>: 0x00 = None, 0x01 + value = Some(value)
//! - Strings: u32 LE length + UTF-8 bytes

use sha2::{Digest, Sha256};

/// Config encoding version. Bump when encoding rules change.
pub const CONFIG_ENCODING_VERSION: u8 = 0x01;

/// Trait for deterministic binary encoding of config structs.
///
/// Implementations must encode fields in fixed order with no padding
/// or platform-dependent representations.
pub trait CanonicalBytes {
    /// Encode to canonical bytes.
    ///
    /// The first byte MUST be `CONFIG_ENCODING_VERSION`.
    fn canonical_bytes(&self) -> Vec<u8>;
}

/// Compute SHA-256 hash of canonical bytes.
pub fn canonical_hash<T: CanonicalBytes>(value: &T) -> String {
    let bytes = value.canonical_bytes();
    let hash = Sha256::digest(&bytes);
    hex::encode(hash)
}

/// Helper: encode i8 as single byte (little-endian).
#[inline]
pub fn encode_i8(buf: &mut Vec<u8>, value: i8) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Helper: encode i64 as 8 bytes (little-endian).
#[inline]
pub fn encode_i64(buf: &mut Vec<u8>, value: i64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Helper: encode i32 as 4 bytes (little-endian).
#[inline]
pub fn encode_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Helper: encode i128 as 16 bytes (little-endian).
#[inline]
pub fn encode_i128(buf: &mut Vec<u8>, value: i128) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Helper: encode u32 as 4 bytes (little-endian).
#[inline]
pub fn encode_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Helper: encode string as length-prefixed UTF-8.
#[inline]
pub fn encode_string(buf: &mut Vec<u8>, value: &str) {
    encode_u32(buf, value.len() as u32);
    buf.extend_from_slice(value.as_bytes());
}

/// Helper: encode Option<T> with presence marker.
#[inline]
pub fn encode_option<T, F>(buf: &mut Vec<u8>, value: &Option<T>, encode_fn: F)
where
    F: FnOnce(&mut Vec<u8>, &T),
{
    match value {
        None => buf.push(0x00),
        Some(v) => {
            buf.push(0x01);
            encode_fn(buf, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestConfig {
        threshold: i64,
        qty_exponent: i8,
        name: String,
    }

    impl CanonicalBytes for TestConfig {
        fn canonical_bytes(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            buf.push(CONFIG_ENCODING_VERSION);
            encode_i64(&mut buf, self.threshold);
            encode_i8(&mut buf, self.qty_exponent);
            encode_string(&mut buf, &self.name);
            buf
        }
    }

    #[test]
    fn test_canonical_bytes_deterministic() {
        let config1 = TestConfig {
            threshold: 100,
            qty_exponent: -8,
            name: "test".to_string(),
        };
        let config2 = TestConfig {
            threshold: 100,
            qty_exponent: -8,
            name: "test".to_string(),
        };

        assert_eq!(config1.canonical_bytes(), config2.canonical_bytes());
    }

    #[test]
    fn test_canonical_hash_deterministic() {
        let config1 = TestConfig {
            threshold: 100,
            qty_exponent: -8,
            name: "test".to_string(),
        };
        let config2 = TestConfig {
            threshold: 100,
            qty_exponent: -8,
            name: "test".to_string(),
        };

        assert_eq!(canonical_hash(&config1), canonical_hash(&config2));
    }

    #[test]
    fn test_canonical_bytes_includes_version() {
        let config = TestConfig {
            threshold: 100,
            qty_exponent: -8,
            name: "test".to_string(),
        };
        let bytes = config.canonical_bytes();
        assert_eq!(bytes[0], CONFIG_ENCODING_VERSION);
    }

    #[test]
    fn test_i8_encoding() {
        let mut buf = Vec::new();
        encode_i8(&mut buf, -8);
        // -8 as i8 in two's complement = 0xF8
        assert_eq!(buf, vec![0xF8]);
    }

    #[test]
    fn test_different_configs_different_hash() {
        let config1 = TestConfig {
            threshold: 100,
            qty_exponent: -8,
            name: "test".to_string(),
        };
        let config2 = TestConfig {
            threshold: 200, // Different
            qty_exponent: -8,
            name: "test".to_string(),
        };

        assert_ne!(canonical_hash(&config1), canonical_hash(&config2));
    }
}
