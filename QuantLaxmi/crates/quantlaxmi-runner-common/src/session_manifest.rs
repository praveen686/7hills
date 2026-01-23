//! # Session Manifest
//!
//! Aggregate manifest for multi-underlying capture sessions.
//!
//! ## Purpose
//! Provides a single entrypoint file (`session_manifest.json`) that describes
//! the entire capture session, including:
//! - Per-underlying universe manifests and their hashes
//! - Tick output inventory
//! - Integrity summary (out-of-universe ticks, subscribe mode)
//!
//! ## Usage
//! This file becomes the contract boundary for replay and SANOS certification.
//! SANOS should read `session_manifest.json` to locate universe manifests
//! rather than inferring expiry dates from symbol parsing.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Schema version for forward compatibility
pub const SESSION_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Aggregate session manifest for multi-underlying capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    /// Schema version for forward compatibility
    pub schema_version: u32,
    /// UTC timestamp when manifest was created (ISO8601)
    pub created_at_utc: String,
    /// Unique session identifier (UUID)
    pub session_id: String,
    /// Capture mode identifier (e.g., "india_capture", "crypto_capture")
    pub capture_mode: String,
    /// Root output directory
    pub out_dir: String,
    /// Capture duration in seconds
    pub duration_secs: u64,
    /// Price exponent for mantissa conversion
    pub price_exponent: i8,
    /// Per-underlying entries
    pub underlyings: Vec<UnderlyingEntry>,
    /// Tick output files inventory
    pub tick_outputs: Vec<TickOutputEntry>,
    /// Integrity summary
    pub integrity: IntegritySummary,
}

/// Per-underlying entry in the session manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderlyingEntry {
    /// Underlying symbol (e.g., "NIFTY", "BANKNIFTY")
    pub underlying: String,
    /// Subdirectory relative to session root (e.g., "nifty/")
    pub subdir: String,
    /// Path to universe manifest (relative to session root)
    pub universe_manifest_path: String,
    /// SHA-256 hash of universe manifest
    pub universe_manifest_sha256: String,
    /// Number of instruments in the universe
    pub instrument_count: usize,
    /// T1 expiry date (ISO8601 date)
    pub t1_expiry: String,
    /// T2 expiry date if available
    pub t2_expiry: Option<String>,
    /// T3 expiry date if available
    pub t3_expiry: Option<String>,
    /// Strike step for the underlying
    pub strike_step: f64,
}

/// Tick output file entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickOutputEntry {
    /// Trading symbol
    pub symbol: String,
    /// Path to ticks file (relative to session root)
    pub path: String,
    /// Number of ticks written
    pub ticks_written: usize,
    /// Whether tick had L2 depth data
    pub has_depth: bool,
}

/// Integrity summary for the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegritySummary {
    /// Number of out-of-universe ticks dropped
    pub out_of_universe_ticks_dropped: usize,
    /// Subscribe mode used ("manifest_tokens" or "api_lookup")
    pub subscribe_mode: String,
    /// Optional notes (e.g., warnings, anomalies)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl SessionManifest {
    /// Create a new session manifest with current timestamp.
    pub fn new(
        session_id: String,
        capture_mode: String,
        out_dir: String,
        duration_secs: u64,
        price_exponent: i8,
    ) -> Self {
        Self {
            schema_version: SESSION_MANIFEST_SCHEMA_VERSION,
            created_at_utc: Utc::now().to_rfc3339(),
            session_id,
            capture_mode,
            out_dir,
            duration_secs,
            price_exponent,
            underlyings: Vec::new(),
            tick_outputs: Vec::new(),
            integrity: IntegritySummary {
                out_of_universe_ticks_dropped: 0,
                subscribe_mode: "unknown".to_string(),
                notes: Vec::new(),
            },
        }
    }

    /// Add an underlying entry.
    pub fn add_underlying(&mut self, entry: UnderlyingEntry) {
        self.underlyings.push(entry);
    }

    /// Add a tick output entry.
    pub fn add_tick_output(&mut self, entry: TickOutputEntry) {
        self.tick_outputs.push(entry);
    }

    /// Set integrity summary.
    pub fn set_integrity(&mut self, integrity: IntegritySummary) {
        self.integrity = integrity;
    }
}

/// Result of persisting a session manifest.
#[derive(Debug, Clone)]
pub struct SessionManifestPersistResult {
    /// Path to the written manifest JSON file
    pub manifest_path: PathBuf,
    /// Size of the manifest JSON in bytes
    pub bytes_len: usize,
}

/// Persist session manifest atomically to the session directory.
///
/// Writes `session_manifest.json` to the specified directory using
/// atomic write (temp file + rename) to prevent partial writes.
///
/// **Determinism**: Underlying entries are sorted lexicographically by `underlying`
/// before serialization to ensure stable canonical JSON for hashing/diffing.
pub fn persist_session_manifest_atomic(
    out_dir: &Path,
    manifest: &SessionManifest,
) -> Result<SessionManifestPersistResult> {
    use crate::manifest_io::write_atomic;

    // Clone and sort for deterministic output (Commit C audit fix 1.1)
    let mut sorted_manifest = manifest.clone();
    sorted_manifest
        .underlyings
        .sort_by(|a, b| a.underlying.cmp(&b.underlying));
    sorted_manifest
        .tick_outputs
        .sort_by(|a, b| a.symbol.cmp(&b.symbol));

    // Serialize to pretty JSON for readability (session manifest is human-facing)
    let bytes = serde_json::to_vec_pretty(&sorted_manifest)
        .context("Failed to serialize SessionManifest to JSON")?;

    let manifest_path = out_dir.join("session_manifest.json");

    write_atomic(&manifest_path, &bytes)
        .with_context(|| format!("Failed to write session manifest: {:?}", manifest_path))?;

    Ok(SessionManifestPersistResult {
        manifest_path,
        bytes_len: bytes.len(),
    })
}

// =============================================================================
// MANIFEST LOADERS (Commit D: SANOS manifest-driven mode)
// =============================================================================

/// Load session manifest from a session directory.
///
/// Reads `session_manifest.json` from the specified directory.
/// Returns `None` if the file does not exist.
pub fn load_session_manifest(out_dir: &Path) -> Result<SessionManifest> {
    let path = out_dir.join("session_manifest.json");
    let bytes = std::fs::read(&path)
        .with_context(|| format!("Failed to read session manifest: {}", path.display()))?;
    let sm: SessionManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("Failed to parse session manifest: {}", path.display()))?;
    Ok(sm)
}

/// Check if session manifest exists in the given directory.
pub fn session_manifest_exists(out_dir: &Path) -> bool {
    out_dir.join("session_manifest.json").exists()
}

/// Load universe manifest from a session directory.
///
/// The `rel_path` is relative to `out_dir` (e.g., "nifty/universe_manifest.json").
///
/// Note: Returns the raw JSON bytes deserialized. The caller is responsible
/// for providing the correct type. This function is generic to avoid
/// coupling runner-common to quantlaxmi-connectors-zerodha.
pub fn load_universe_manifest_bytes(out_dir: &Path, rel_path: &str) -> Result<Vec<u8>> {
    let path = out_dir.join(rel_path);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("Failed to read universe manifest: {}", path.display()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_session_manifest_creation() {
        let mut manifest = SessionManifest::new(
            "test-session-123".to_string(),
            "india_capture".to_string(),
            "/data/sessions/test".to_string(),
            300,
            -2,
        );

        manifest.add_underlying(UnderlyingEntry {
            underlying: "NIFTY".to_string(),
            subdir: "nifty/".to_string(),
            universe_manifest_path: "nifty/universe_manifest.json".to_string(),
            universe_manifest_sha256: "abc123".to_string(),
            instrument_count: 100,
            t1_expiry: "2026-01-30".to_string(),
            t2_expiry: Some("2026-02-06".to_string()),
            t3_expiry: Some("2026-02-27".to_string()),
            strike_step: 50.0,
        });

        manifest.add_tick_output(TickOutputEntry {
            symbol: "NIFTY26JAN24000CE".to_string(),
            path: "NIFTY26JAN24000CE/ticks.jsonl".to_string(),
            ticks_written: 1000,
            has_depth: true,
        });

        manifest.set_integrity(IntegritySummary {
            out_of_universe_ticks_dropped: 5,
            subscribe_mode: "manifest_tokens".to_string(),
            notes: vec![],
        });

        assert_eq!(manifest.schema_version, 1);
        assert_eq!(manifest.underlyings.len(), 1);
        assert_eq!(manifest.tick_outputs.len(), 1);
        assert_eq!(manifest.integrity.out_of_universe_ticks_dropped, 5);
    }

    #[test]
    fn test_persist_session_manifest() {
        let dir = tempdir().unwrap();
        let manifest = SessionManifest::new(
            "test-session".to_string(),
            "india_capture".to_string(),
            dir.path().to_string_lossy().to_string(),
            60,
            -2,
        );

        let result = persist_session_manifest_atomic(dir.path(), &manifest).unwrap();

        assert!(result.manifest_path.exists());
        assert!(result.bytes_len > 0);

        // Verify it can be read back
        let content = std::fs::read_to_string(&result.manifest_path).unwrap();
        let parsed: SessionManifest = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.session_id, "test-session");
    }
}
