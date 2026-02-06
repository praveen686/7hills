//! # Canonical Binary Encoding Primitives
//!
//! Shared encoding utilities for deterministic binary serialization.
//! Used by config hashing, trace hashing, and WAL encoding.
//!
//! ## Why Binary Encoding?
//! - JSON text hashing is non-deterministic (float formatting, field order)
//! - Binary encoding guarantees cross-platform reproducibility
//! - Little-endian is used throughout for consistency
//!
//! ## Encoding Rules
//! - Integers: little-endian fixed-width bytes
//! - Strings: u32 LE length prefix + UTF-8 bytes
//! - Option<T>: 0x00 = None, 0x01 + value = Some
//! - DateTime: i64 microseconds since epoch
//! - UUID: 16 raw bytes

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Encode i8 as single byte (little-endian).
#[inline]
pub fn encode_i8(buf: &mut Vec<u8>, value: i8) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode i16 as 2 bytes (little-endian).
#[inline]
pub fn encode_i16(buf: &mut Vec<u8>, value: i16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode i32 as 4 bytes (little-endian).
#[inline]
pub fn encode_i32(buf: &mut Vec<u8>, value: i32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode i64 as 8 bytes (little-endian).
#[inline]
pub fn encode_i64(buf: &mut Vec<u8>, value: i64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode i128 as 16 bytes (little-endian).
#[inline]
pub fn encode_i128(buf: &mut Vec<u8>, value: i128) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode u16 as 2 bytes (little-endian).
#[inline]
pub fn encode_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode u32 as 4 bytes (little-endian).
#[inline]
pub fn encode_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode u64 as 8 bytes (little-endian).
#[inline]
pub fn encode_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Encode string as length-prefixed UTF-8.
#[inline]
pub fn encode_string(buf: &mut Vec<u8>, value: &str) {
    encode_u32(buf, value.len() as u32);
    buf.extend_from_slice(value.as_bytes());
}

/// Encode Option<T> with presence marker.
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

/// Encode Option<String> with presence marker.
#[inline]
pub fn encode_optional_string(buf: &mut Vec<u8>, opt: &Option<String>) {
    match opt {
        None => buf.push(0x00),
        Some(s) => {
            buf.push(0x01);
            encode_string(buf, s);
        }
    }
}

/// Encode DateTime<Utc> as microseconds since epoch (i64).
#[inline]
pub fn encode_datetime(buf: &mut Vec<u8>, dt: &DateTime<Utc>) {
    let micros = dt.timestamp_micros();
    encode_i64(buf, micros);
}

/// Encode UUID as 16 raw bytes.
#[inline]
pub fn encode_uuid(buf: &mut Vec<u8>, uuid: &Uuid) {
    buf.extend_from_slice(uuid.as_bytes());
}

/// Encode Option<Uuid> with presence marker.
#[inline]
pub fn encode_optional_uuid(buf: &mut Vec<u8>, opt: &Option<Uuid>) {
    match opt {
        None => buf.push(0x00),
        Some(uuid) => {
            buf.push(0x01);
            encode_uuid(buf, uuid);
        }
    }
}

/// Encode bool as single byte (0x00 = false, 0x01 = true).
#[inline]
pub fn encode_bool(buf: &mut Vec<u8>, value: bool) {
    buf.push(if value { 0x01 } else { 0x00 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i8_encoding() {
        let mut buf = Vec::new();
        encode_i8(&mut buf, -8);
        assert_eq!(buf, vec![0xF8]); // -8 in two's complement
    }

    #[test]
    fn test_i64_encoding() {
        let mut buf = Vec::new();
        encode_i64(&mut buf, 0x0102030405060708);
        assert_eq!(buf, vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_string_encoding() {
        let mut buf = Vec::new();
        encode_string(&mut buf, "test");
        assert_eq!(buf.len(), 4 + 4); // 4 bytes length + 4 bytes content
        assert_eq!(buf[0..4], [4, 0, 0, 0]); // length = 4 in LE
        assert_eq!(&buf[4..], b"test");
    }

    #[test]
    fn test_option_none() {
        let mut buf = Vec::new();
        let opt: Option<i64> = None;
        encode_option(&mut buf, &opt, |b, v| encode_i64(b, *v));
        assert_eq!(buf, vec![0x00]);
    }

    #[test]
    fn test_option_some() {
        let mut buf = Vec::new();
        let opt: Option<i64> = Some(42);
        encode_option(&mut buf, &opt, |b, v| encode_i64(b, *v));
        assert_eq!(buf[0], 0x01);
        assert_eq!(buf.len(), 1 + 8);
    }

    #[test]
    fn test_uuid_encoding() {
        let mut buf = Vec::new();
        let uuid = Uuid::nil();
        encode_uuid(&mut buf, &uuid);
        assert_eq!(buf.len(), 16);
    }
}
