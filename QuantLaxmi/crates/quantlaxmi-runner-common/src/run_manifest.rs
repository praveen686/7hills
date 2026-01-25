//! Run Manifest
//!
//! Audit-grade binding between an analysis run (SANOS/strategy/backtest) and its
//! capture inputs / produced outputs.
//!
//! Goals:
//! - Deterministic JSON (stable ordering)
//! - Hash binding to session + universe manifests
//! - Hash binding to all declared output files

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::manifest_io::{sha256_hex, write_atomic};

/// Canonical manifest describing one analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,

    /// Binary name (e.g., run_calendar_carry)
    pub binary_name: String,

    /// Git commit string (best-effort; may be "unknown")
    pub git_commit: String,

    /// Deterministic run ID (recommended: hash(session_manifest + config))
    pub run_id: String,

    /// Session directory (relative or absolute path string)
    pub session_dir: String,

    /// Session manifest binding
    pub session_manifest: InputBinding,

    /// Universe manifest bindings (typically one per underlying)
    pub universe_manifests: Vec<InputBinding>,

    /// Deterministic config hash for this run
    pub config_sha256: String,

    /// Declared outputs for this run
    pub outputs: Vec<OutputBinding>,

    /// WAL file bindings (path + sha256 for each WAL file)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wal_files: Vec<WalBinding>,
}

/// WAL file binding with integrity hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalBinding {
    /// WAL file type (e.g., "market", "decisions", "orders", "fills", "risk")
    pub file_type: String,
    /// Relative path to WAL file
    pub rel_path: String,
    /// SHA-256 hash of file contents
    pub sha256: String,
    /// Number of records in file
    pub record_count: u64,
    /// File size in bytes
    pub bytes_len: usize,
}

/// Input binding (path + sha256).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputBinding {
    pub label: String,
    pub rel_path: String,
    pub sha256: String,
}

/// Output file binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputBinding {
    pub label: String,
    pub rel_path: String,
    pub sha256: String,
    pub bytes_len: usize,
}

/// Persist `run_manifest.json` + a sha256 file next to it.
///
/// Returns the sha256 of the manifest JSON bytes.
pub fn persist_run_manifest_atomic(run_dir: &Path, rm: &RunManifest) -> Result<String> {
    // Deterministic ordering
    let mut rm_sorted = rm.clone();
    rm_sorted
        .universe_manifests
        .sort_by(|a, b| a.label.cmp(&b.label).then(a.rel_path.cmp(&b.rel_path)));
    rm_sorted
        .outputs
        .sort_by(|a, b| a.label.cmp(&b.label).then(a.rel_path.cmp(&b.rel_path)));
    rm_sorted
        .wal_files
        .sort_by(|a, b| a.file_type.cmp(&b.file_type));

    // Pretty JSON for human audit
    let bytes = serde_json::to_vec_pretty(&rm_sorted).context("Failed to serialize RunManifest")?;
    let sha = sha256_hex(&bytes);

    let manifest_path = run_dir.join("run_manifest.json");
    let sha_path = run_dir.join("run_manifest.sha256");

    write_atomic(&manifest_path, &bytes)
        .with_context(|| format!("Failed to write run manifest: {}", manifest_path.display()))?;

    let hash_content = format!("{}  run_manifest.json\n", sha);
    write_atomic(&sha_path, hash_content.as_bytes())
        .with_context(|| format!("Failed to write run manifest hash: {}", sha_path.display()))?;

    Ok(sha)
}

/// Compute sha256 + bytes_len for a file.
pub fn hash_file(path: &Path) -> Result<(String, usize)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;
    Ok((sha256_hex(&bytes), bytes.len()))
}

/// Compute a deterministic config hash from any Serialize payload.
pub fn config_hash<T: Serialize>(cfg: &T) -> Result<String> {
    let bytes = serde_json::to_vec(cfg).context("Failed to serialize config for hashing")?;
    Ok(sha256_hex(&bytes))
}

/// Best-effort git commit string.
///
/// Recommended: set `QUANTLAXMI_GIT_COMMIT` at build time (e.g., via CI / build.rs).
pub fn git_commit_string() -> String {
    std::env::var("QUANTLAXMI_GIT_COMMIT").unwrap_or_else(|_| "unknown".to_string())
}

/// Utility to create a binding from an on-disk JSON file.
pub fn bind_json_file(label: &str, rel_path: &str, base_dir: &Path) -> Result<InputBinding> {
    let p = base_dir.join(rel_path);
    let (sha, _) = hash_file(&p)?;
    Ok(InputBinding {
        label: label.to_string(),
        rel_path: rel_path.to_string(),
        sha256: sha,
    })
}
