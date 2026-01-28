//! # Universe Manifest I/O
//!
//! Atomic persistence and hashing for UniverseManifest.
//!
//! ## Purpose
//! Ensures every India capture session is self-describing by persisting
//! the UniverseManifest with cryptographic integrity verification.
//!
//! ## Files Written
//! - `universe_manifest.json` - Canonical JSON (compact, deterministic)
//! - `universe_manifest.sha256` - SHA-256 hash in sha256sum format
//!
//! ## Determinism Requirements
//! - Uses compact JSON (no pretty printing) for stable byte output
//! - UniverseManifest must use BTreeMap (not HashMap) for deterministic ordering
//! - Same inputs produce same hash

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Result of persisting a universe manifest.
#[derive(Debug, Clone)]
pub struct ManifestPersistResult {
    /// Path to the written manifest JSON file
    pub manifest_path: PathBuf,
    /// Path to the written SHA-256 hash file
    pub sha_path: PathBuf,
    /// SHA-256 hash (lowercase hex, 64 characters)
    pub sha256: String,
    /// Size of the manifest JSON in bytes
    pub bytes_len: usize,
}

/// Compute SHA-256 hash of bytes, returning lowercase hex string.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hash = hasher.finalize();
    hex::encode(hash)
}

/// Write bytes to a file atomically.
///
/// Writes to a temporary file first, then renames to the final path.
/// This prevents partial writes on crash.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create directory: {:?}", parent))?;

    // Write to temp file in same directory
    let temp_path = path.with_extension("tmp");
    {
        let mut file = File::create(&temp_path)
            .with_context(|| format!("Failed to create temp file: {:?}", temp_path))?;
        file.write_all(bytes)
            .with_context(|| format!("Failed to write to temp file: {:?}", temp_path))?;
        file.sync_all()
            .with_context(|| format!("Failed to sync temp file: {:?}", temp_path))?;
    }

    // Atomic rename
    fs::rename(&temp_path, path)
        .with_context(|| format!("Failed to rename {:?} to {:?}", temp_path, path))?;

    Ok(())
}

/// Persist a UniverseManifest to the session directory.
///
/// Writes two files:
/// - `universe_manifest.json` - Compact JSON representation
/// - `universe_manifest.sha256` - Hash file in sha256sum format
///
/// # Arguments
/// * `session_dir` - Directory to write manifest files
/// * `manifest` - The UniverseManifest to persist (must implement Serialize)
///
/// # Returns
/// * `ManifestPersistResult` with paths and hash
///
/// # Determinism
/// Uses compact JSON (no pretty printing) for deterministic byte output.
/// The manifest type should use BTreeMap instead of HashMap for field ordering.
pub fn persist_universe_manifest<T: Serialize>(
    session_dir: &Path,
    manifest: &T,
) -> Result<ManifestPersistResult> {
    // Serialize to compact JSON (deterministic)
    let bytes =
        serde_json::to_vec(manifest).context("Failed to serialize UniverseManifest to JSON")?;

    // Compute hash
    let sha256 = sha256_hex(&bytes);

    // Paths
    let manifest_path = session_dir.join("universe_manifest.json");
    let sha_path = session_dir.join("universe_manifest.sha256");

    // Write manifest atomically
    write_atomic(&manifest_path, &bytes)
        .with_context(|| format!("Failed to write manifest: {:?}", manifest_path))?;

    // Write hash file in sha256sum format: "<hash>  <filename>\n"
    let hash_content = format!("{}  universe_manifest.json\n", sha256);
    write_atomic(&sha_path, hash_content.as_bytes())
        .with_context(|| format!("Failed to write hash file: {:?}", sha_path))?;

    Ok(ManifestPersistResult {
        manifest_path,
        sha_path,
        sha256,
        bytes_len: bytes.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestManifest {
        underlying: String,
        strike_band: u32,
        instruments: Vec<String>,
        metadata: BTreeMap<String, String>,
    }

    #[test]
    fn test_sha256_hex() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(hash.len(), 64);
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_persist_universe_manifest() {
        let dir = tempdir().unwrap();
        let manifest = TestManifest {
            underlying: "BANKNIFTY".to_string(),
            strike_band: 10,
            instruments: vec!["BANKNIFTY26JAN48000CE".to_string()],
            metadata: BTreeMap::new(),
        };

        let result = persist_universe_manifest(dir.path(), &manifest).unwrap();

        // Verify files exist
        assert!(result.manifest_path.exists());
        assert!(result.sha_path.exists());

        // Verify hash is 64 hex chars
        assert_eq!(result.sha256.len(), 64);

        // Verify manifest can be read back
        let content = fs::read_to_string(&result.manifest_path).unwrap();
        let parsed: TestManifest = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed, manifest);

        // Verify hash file format
        let hash_content = fs::read_to_string(&result.sha_path).unwrap();
        assert!(hash_content.starts_with(&result.sha256));
        assert!(hash_content.contains("universe_manifest.json"));
    }

    #[test]
    fn test_deterministic_hash() {
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();

        let manifest = TestManifest {
            underlying: "NIFTY".to_string(),
            strike_band: 5,
            instruments: vec!["A".to_string(), "B".to_string()],
            metadata: BTreeMap::new(),
        };

        let result1 = persist_universe_manifest(dir1.path(), &manifest).unwrap();
        let result2 = persist_universe_manifest(dir2.path(), &manifest).unwrap();

        // Same manifest should produce same hash
        assert_eq!(result1.sha256, result2.sha256);
        assert_eq!(result1.bytes_len, result2.bytes_len);
    }
}
