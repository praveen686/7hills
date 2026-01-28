//! Promotion Pipeline — Gate artifacts and promotion records.
//!
//! Phase 20D: Makes signal promotion impossible without passing gates.
//!
//! ## Canonical Paths (Frozen)
//! ```text
//! {segment_dir}/
//! ├── segment_admission_summary.json    # Phase 19D
//! ├── gates/
//! │   ├── g0_manifest.json              # G0 output
//! │   ├── g1_determinism.json           # G1 output (optional)
//! │   ├── g2_integrity.json             # G2 output
//! │   └── gates_summary.json            # Combined summary
//! └── promotion/
//!     └── promotion_record.json         # Written only if gates passed
//! ```
//!
//! ## Hard Laws
//! - PromotionRecord is written ONLY if all required gates passed
//! - All maps are BTreeMap (deterministic iteration)
//! - File digests are computed over exact written bytes
//! - Gate outputs use pretty JSON (frozen choice)

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::process::Command;

// =============================================================================
// Schema Versions (Frozen)
// =============================================================================

pub const GATES_SUMMARY_SCHEMA: &str = "1.0.0";
pub const PROMOTION_RECORD_SCHEMA: &str = "1.0.0";

/// Tool version from Cargo.toml
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

// =============================================================================
// Canonical Paths (Frozen)
// =============================================================================

/// Canonical path for gates directory under segment.
pub fn gates_dir(segment_dir: &Path) -> std::path::PathBuf {
    segment_dir.join("gates")
}

/// Canonical path for promotion directory under segment.
pub fn promotion_dir(segment_dir: &Path) -> std::path::PathBuf {
    segment_dir.join("promotion")
}

/// Canonical path for G0 output.
pub fn g0_output_path(segment_dir: &Path) -> std::path::PathBuf {
    gates_dir(segment_dir).join("g0_manifest.json")
}

/// Canonical path for G1 output.
pub fn g1_output_path(segment_dir: &Path) -> std::path::PathBuf {
    gates_dir(segment_dir).join("g1_determinism.json")
}

/// Canonical path for G2 output.
pub fn g2_output_path(segment_dir: &Path) -> std::path::PathBuf {
    gates_dir(segment_dir).join("g2_integrity.json")
}

/// Canonical path for gates summary.
pub fn gates_summary_path(segment_dir: &Path) -> std::path::PathBuf {
    gates_dir(segment_dir).join("gates_summary.json")
}

/// Canonical path for promotion record.
pub fn promotion_record_path(segment_dir: &Path) -> std::path::PathBuf {
    promotion_dir(segment_dir).join("promotion_record.json")
}

/// Canonical path for WAL in session directory.
pub fn session_wal_path(session_dir: &Path) -> std::path::PathBuf {
    session_dir.join("wal").join("signals_admission.jsonl")
}

/// Canonical path for segment admission summary.
pub fn segment_summary_path(segment_dir: &Path) -> std::path::PathBuf {
    segment_dir.join("segment_admission_summary.json")
}

// =============================================================================
// GatesSummary — Combined gate results
// =============================================================================

/// Combined gate results with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatesSummary {
    /// Schema version
    pub schema_version: String,

    /// Overall pass/fail
    pub passed: bool,

    /// Timestamp (ISO 8601 UTC)
    pub timestamp: String,

    /// Tool version (quantlaxmi-gates crate version)
    pub tool_version: String,

    /// Exit code (0=pass, 1=fail, 2=error)
    pub exit_code: u8,

    /// G0 result (always present)
    pub g0: Option<GateOutcome>,

    /// G1 result (None if skipped)
    pub g1: Option<GateOutcome>,

    /// G2 result (None if skipped)
    pub g2: Option<GateOutcome>,
}

impl GatesSummary {
    /// Create a new summary with current timestamp.
    pub fn new() -> Self {
        Self {
            schema_version: GATES_SUMMARY_SCHEMA.to_string(),
            passed: true,
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_version: TOOL_VERSION.to_string(),
            exit_code: 0,
            g0: None,
            g1: None,
            g2: None,
        }
    }

    /// Create with a specific timestamp (for testing).
    pub fn with_timestamp(timestamp: impl Into<String>) -> Self {
        Self {
            schema_version: GATES_SUMMARY_SCHEMA.to_string(),
            passed: true,
            timestamp: timestamp.into(),
            tool_version: TOOL_VERSION.to_string(),
            exit_code: 0,
            g0: None,
            g1: None,
            g2: None,
        }
    }

    /// Add G0 outcome.
    pub fn with_g0(mut self, outcome: GateOutcome) -> Self {
        if !outcome.passed {
            self.passed = false;
            self.exit_code = 1;
        }
        self.g0 = Some(outcome);
        self
    }

    /// Add G1 outcome.
    pub fn with_g1(mut self, outcome: GateOutcome) -> Self {
        if !outcome.passed {
            self.passed = false;
            self.exit_code = 1;
        }
        self.g1 = Some(outcome);
        self
    }

    /// Add G2 outcome.
    pub fn with_g2(mut self, outcome: GateOutcome) -> Self {
        if !outcome.passed {
            self.passed = false;
            self.exit_code = 1;
        }
        self.g2 = Some(outcome);
        self
    }

    /// Set exit code for errors.
    pub fn with_error(mut self) -> Self {
        self.passed = false;
        self.exit_code = 2;
        self
    }

    /// Serialize to pretty JSON bytes.
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("GatesSummary serialization cannot fail")
    }
}

impl Default for GatesSummary {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual gate outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateOutcome {
    /// Gate name ("g0", "g1", "g2")
    pub gate: String,

    /// Pass/fail
    pub passed: bool,

    /// Relative path to output file under segment_dir
    pub output_path: String,

    /// SHA-256 hex of file bytes
    pub output_digest: String,
}

impl GateOutcome {
    /// Create outcome for a gate.
    pub fn new(gate: &str, passed: bool, output_path: &str, output_digest: &str) -> Self {
        Self {
            gate: gate.to_string(),
            passed,
            output_path: output_path.to_string(),
            output_digest: output_digest.to_string(),
        }
    }
}

// =============================================================================
// PromotionRecord — Audit-grade promotion receipt
// =============================================================================

/// Promotion record written only when all gates pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionRecord {
    /// Schema version
    pub schema_version: String,

    /// Timestamp (ISO 8601 UTC)
    pub timestamp: String,

    /// Unique promotion ID (UUID v4)
    pub promotion_id: String,

    // === Git identity (best-effort) ===
    /// HEAD commit SHA
    pub git_commit: Option<String>,

    /// Current branch name
    pub git_branch: Option<String>,

    /// Whether working directory is clean
    pub git_clean: Option<bool>,

    // === Manifest identity ===
    /// Path to manifest (as passed)
    pub manifest_path: String,

    /// SHA-256 of manifest content (raw bytes)
    pub manifest_hash: [u8; 32],

    /// Manifest version string
    pub manifest_version: String,

    // === Inputs ===
    /// Segment directory path
    pub segment_dir: String,

    /// Session directory path (if provided)
    pub session_dir: Option<String>,

    /// Replay directory path (if provided)
    pub replay_dir: Option<String>,

    // === Thresholds used ===
    /// Minimum coverage threshold for G2
    pub min_coverage: f64,

    /// Minimum events for G2
    pub min_events: usize,

    // === Gate results ===
    /// Must be true for record to exist
    pub gates_passed: bool,

    /// SHA-256 hex of gates_summary.json bytes
    pub gates_summary_digest: String,

    /// Individual gate output digests (BTreeMap for determinism)
    pub gate_digests: BTreeMap<String, String>,

    // === Tool identity ===
    /// quantlaxmi-gates crate version
    pub tool_version: String,

    /// Hostname where promotion ran
    pub hostname: Option<String>,

    // === Promotion digest ===
    /// SHA-256 hex of canonical bytes (excluding this field)
    pub digest: String,
}

impl PromotionRecord {
    /// Create a new promotion record builder.
    pub fn builder() -> PromotionRecordBuilder {
        PromotionRecordBuilder::new()
    }

    /// Compute canonical bytes for digest (excludes digest field).
    ///
    /// Field order is frozen for determinism.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. schema_version
        append_string(&mut bytes, &self.schema_version);

        // 2. timestamp
        append_string(&mut bytes, &self.timestamp);

        // 3. promotion_id
        append_string(&mut bytes, &self.promotion_id);

        // 4. git_commit
        append_option_string(&mut bytes, &self.git_commit);

        // 5. git_branch
        append_option_string(&mut bytes, &self.git_branch);

        // 6. git_clean
        append_option_bool(&mut bytes, self.git_clean);

        // 7. manifest_path
        append_string(&mut bytes, &self.manifest_path);

        // 8. manifest_hash (32 bytes raw)
        bytes.extend_from_slice(&self.manifest_hash);

        // 9. manifest_version
        append_string(&mut bytes, &self.manifest_version);

        // 10. segment_dir
        append_string(&mut bytes, &self.segment_dir);

        // 11. session_dir
        append_option_string(&mut bytes, &self.session_dir);

        // 12. replay_dir
        append_option_string(&mut bytes, &self.replay_dir);

        // 13. min_coverage (f64 as bits)
        bytes.extend_from_slice(&self.min_coverage.to_le_bytes());

        // 14. min_events (usize as u64)
        bytes.extend_from_slice(&(self.min_events as u64).to_le_bytes());

        // 15. gates_passed
        bytes.push(if self.gates_passed { 1 } else { 0 });

        // 16. gates_summary_digest
        append_string(&mut bytes, &self.gates_summary_digest);

        // 17. gate_digests (BTreeMap - sorted iteration guaranteed)
        bytes.extend_from_slice(&(self.gate_digests.len() as u32).to_le_bytes());
        for (key, value) in &self.gate_digests {
            append_string(&mut bytes, key);
            append_string(&mut bytes, value);
        }

        // 18. tool_version
        append_string(&mut bytes, &self.tool_version);

        // 19. hostname
        append_option_string(&mut bytes, &self.hostname);

        // NOTE: digest field is NOT included (circular)

        bytes
    }

    /// Compute and set the digest field.
    pub fn compute_digest(&mut self) {
        let canonical = self.canonical_bytes();
        self.digest = sha256_hex(&canonical);
    }

    /// Serialize to pretty JSON bytes.
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("PromotionRecord serialization cannot fail")
    }
}

/// Builder for PromotionRecord.
pub struct PromotionRecordBuilder {
    timestamp: Option<String>,
    promotion_id: Option<String>,
    git_commit: Option<String>,
    git_branch: Option<String>,
    git_clean: Option<bool>,
    manifest_path: String,
    manifest_hash: [u8; 32],
    manifest_version: String,
    segment_dir: String,
    session_dir: Option<String>,
    replay_dir: Option<String>,
    min_coverage: f64,
    min_events: usize,
    gates_passed: bool,
    gates_summary_digest: String,
    gate_digests: BTreeMap<String, String>,
    hostname: Option<String>,
}

impl PromotionRecordBuilder {
    pub fn new() -> Self {
        Self {
            timestamp: None,
            promotion_id: None,
            git_commit: None,
            git_branch: None,
            git_clean: None,
            manifest_path: String::new(),
            manifest_hash: [0u8; 32],
            manifest_version: String::new(),
            segment_dir: String::new(),
            session_dir: None,
            replay_dir: None,
            min_coverage: 0.0,
            min_events: 1,
            gates_passed: false,
            gates_summary_digest: String::new(),
            gate_digests: BTreeMap::new(),
            hostname: None,
        }
    }

    /// Set timestamp (for testing). If not set, uses current time.
    pub fn timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }

    /// Set promotion ID (for testing). If not set, generates UUID v4.
    pub fn promotion_id(mut self, id: impl Into<String>) -> Self {
        self.promotion_id = Some(id.into());
        self
    }

    pub fn git_info(
        mut self,
        commit: Option<String>,
        branch: Option<String>,
        clean: Option<bool>,
    ) -> Self {
        self.git_commit = commit;
        self.git_branch = branch;
        self.git_clean = clean;
        self
    }

    pub fn manifest(mut self, path: &str, hash: [u8; 32], version: &str) -> Self {
        self.manifest_path = path.to_string();
        self.manifest_hash = hash;
        self.manifest_version = version.to_string();
        self
    }

    pub fn segment_dir(mut self, dir: &str) -> Self {
        self.segment_dir = dir.to_string();
        self
    }

    pub fn session_dir(mut self, dir: Option<String>) -> Self {
        self.session_dir = dir;
        self
    }

    pub fn replay_dir(mut self, dir: Option<String>) -> Self {
        self.replay_dir = dir;
        self
    }

    pub fn thresholds(mut self, min_coverage: f64, min_events: usize) -> Self {
        self.min_coverage = min_coverage;
        self.min_events = min_events;
        self
    }

    pub fn gates_passed(mut self, passed: bool) -> Self {
        self.gates_passed = passed;
        self
    }

    pub fn gates_summary_digest(mut self, digest: &str) -> Self {
        self.gates_summary_digest = digest.to_string();
        self
    }

    pub fn add_gate_digest(mut self, gate: &str, digest: &str) -> Self {
        self.gate_digests
            .insert(gate.to_string(), digest.to_string());
        self
    }

    pub fn hostname(mut self, hostname: Option<String>) -> Self {
        self.hostname = hostname;
        self
    }

    /// Build the promotion record with computed digest.
    pub fn build(self) -> PromotionRecord {
        let timestamp = self
            .timestamp
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
        let promotion_id = self
            .promotion_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let mut record = PromotionRecord {
            schema_version: PROMOTION_RECORD_SCHEMA.to_string(),
            timestamp,
            promotion_id,
            git_commit: self.git_commit,
            git_branch: self.git_branch,
            git_clean: self.git_clean,
            manifest_path: self.manifest_path,
            manifest_hash: self.manifest_hash,
            manifest_version: self.manifest_version,
            segment_dir: self.segment_dir,
            session_dir: self.session_dir,
            replay_dir: self.replay_dir,
            min_coverage: self.min_coverage,
            min_events: self.min_events,
            gates_passed: self.gates_passed,
            gates_summary_digest: self.gates_summary_digest,
            gate_digests: self.gate_digests,
            tool_version: TOOL_VERSION.to_string(),
            hostname: self.hostname,
            digest: String::new(),
        };

        record.compute_digest();
        record
    }
}

impl Default for PromotionRecordBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute SHA-256 hex of bytes.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute SHA-256 hex of file contents.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Get git commit SHA (best-effort).
pub fn get_git_commit() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// Get git branch name (best-effort).
pub fn get_git_branch() -> Option<String> {
    Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Check if git working directory is clean (best-effort).
pub fn get_git_clean() -> Option<bool> {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| o.stdout.is_empty())
}

/// Get hostname (best-effort).
pub fn get_hostname() -> Option<String> {
    // Use gethostname syscall via std
    std::env::var("HOSTNAME").ok().or_else(|| {
        Command::new("hostname")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    })
}

// =============================================================================
// Canonical Encoding Helpers
// =============================================================================

fn append_string(bytes: &mut Vec<u8>, s: &str) {
    let s_bytes = s.as_bytes();
    bytes.extend_from_slice(&(s_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(s_bytes);
}

fn append_option_string(bytes: &mut Vec<u8>, opt: &Option<String>) {
    match opt {
        None => bytes.push(0x00),
        Some(s) => {
            bytes.push(0x01);
            append_string(bytes, s);
        }
    }
}

fn append_option_bool(bytes: &mut Vec<u8>, opt: Option<bool>) {
    match opt {
        None => bytes.push(0x00),
        Some(b) => {
            bytes.push(0x01);
            bytes.push(if b { 1 } else { 0 });
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MANIFEST_HASH: [u8; 32] = [0x42; 32];

    #[test]
    fn test_promotion_record_digest_is_deterministic() {
        // Build two records with identical fixed inputs
        let record1 = PromotionRecord::builder()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id("test-id-123")
            .git_info(
                Some("abc123".to_string()),
                Some("main".to_string()),
                Some(true),
            )
            .manifest("config/signals_manifest.json", TEST_MANIFEST_HASH, "0.1.0")
            .segment_dir("/tmp/segment")
            .session_dir(Some("/tmp/session".to_string()))
            .replay_dir(Some("/tmp/replay".to_string()))
            .thresholds(0.9, 100)
            .gates_passed(true)
            .gates_summary_digest("summary_digest_123")
            .add_gate_digest("g0", "g0_digest_abc")
            .add_gate_digest("g2", "g2_digest_def")
            .hostname(Some("test-host".to_string()))
            .build();

        let record2 = PromotionRecord::builder()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id("test-id-123")
            .git_info(
                Some("abc123".to_string()),
                Some("main".to_string()),
                Some(true),
            )
            .manifest("config/signals_manifest.json", TEST_MANIFEST_HASH, "0.1.0")
            .segment_dir("/tmp/segment")
            .session_dir(Some("/tmp/session".to_string()))
            .replay_dir(Some("/tmp/replay".to_string()))
            .thresholds(0.9, 100)
            .gates_passed(true)
            .gates_summary_digest("summary_digest_123")
            .add_gate_digest("g0", "g0_digest_abc")
            .add_gate_digest("g2", "g2_digest_def")
            .hostname(Some("test-host".to_string()))
            .build();

        assert_eq!(
            record1.digest, record2.digest,
            "Same inputs must produce same digest"
        );

        // Run 100 times to prove determinism (with identical inputs)
        for _ in 0..100 {
            let r = PromotionRecord::builder()
                .timestamp("2026-01-28T12:00:00Z")
                .promotion_id("test-id-123")
                .git_info(
                    Some("abc123".to_string()),
                    Some("main".to_string()),
                    Some(true),
                )
                .manifest("config/signals_manifest.json", TEST_MANIFEST_HASH, "0.1.0")
                .segment_dir("/tmp/segment")
                .session_dir(Some("/tmp/session".to_string()))
                .replay_dir(Some("/tmp/replay".to_string()))
                .thresholds(0.9, 100)
                .gates_passed(true)
                .gates_summary_digest("summary_digest_123")
                .add_gate_digest("g0", "g0_digest_abc")
                .add_gate_digest("g2", "g2_digest_def")
                .hostname(Some("test-host".to_string()))
                .build();
            assert_eq!(r.digest, record1.digest);
        }
    }

    #[test]
    fn test_promotion_record_digest_changes_with_inputs() {
        let record1 = PromotionRecord::builder()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id("test-id-1")
            .manifest("config/signals_manifest.json", TEST_MANIFEST_HASH, "0.1.0")
            .segment_dir("/tmp/segment")
            .gates_passed(true)
            .gates_summary_digest("digest1")
            .build();

        let record2 = PromotionRecord::builder()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id("test-id-1")
            .manifest("config/signals_manifest.json", TEST_MANIFEST_HASH, "0.1.0")
            .segment_dir("/tmp/segment")
            .gates_passed(true)
            .gates_summary_digest("digest2") // Different!
            .build();

        assert_ne!(
            record1.digest, record2.digest,
            "Different inputs must produce different digests"
        );
    }

    #[test]
    fn test_gates_summary_builder() {
        let g0 = GateOutcome::new("g0", true, "gates/g0_manifest.json", "abc123");
        let g2 = GateOutcome::new("g2", false, "gates/g2_integrity.json", "def456");

        let summary = GatesSummary::with_timestamp("2026-01-28T12:00:00Z")
            .with_g0(g0)
            .with_g2(g2);

        assert!(!summary.passed, "Summary should fail if any gate fails");
        assert_eq!(summary.exit_code, 1);
        assert!(summary.g0.is_some());
        assert!(summary.g1.is_none());
        assert!(summary.g2.is_some());
    }

    #[test]
    fn test_gates_summary_all_pass() {
        let g0 = GateOutcome::new("g0", true, "gates/g0_manifest.json", "abc");
        let g1 = GateOutcome::new("g1", true, "gates/g1_determinism.json", "def");
        let g2 = GateOutcome::new("g2", true, "gates/g2_integrity.json", "ghi");

        let summary = GatesSummary::new().with_g0(g0).with_g1(g1).with_g2(g2);

        assert!(summary.passed);
        assert_eq!(summary.exit_code, 0);
    }

    #[test]
    fn test_canonical_paths() {
        let segment = Path::new("/data/segment_001");
        let session = Path::new("/data/session_001");

        assert_eq!(gates_dir(segment), Path::new("/data/segment_001/gates"));
        assert_eq!(
            g0_output_path(segment),
            Path::new("/data/segment_001/gates/g0_manifest.json")
        );
        assert_eq!(
            session_wal_path(session),
            Path::new("/data/session_001/wal/signals_admission.jsonl")
        );
        assert_eq!(
            promotion_record_path(segment),
            Path::new("/data/segment_001/promotion/promotion_record.json")
        );
    }

    #[test]
    fn test_sha256_hex() {
        let data = b"hello world";
        let hash = sha256_hex(data);
        // Known SHA-256 of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_gate_digests_sorted() {
        // BTreeMap guarantees sorted order
        let record = PromotionRecord::builder()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id("test")
            .manifest("m.json", [0u8; 32], "0.1.0")
            .segment_dir("/tmp")
            .gates_passed(true)
            .gates_summary_digest("summary")
            .add_gate_digest("g2", "z")
            .add_gate_digest("g0", "a")
            .add_gate_digest("g1", "m")
            .build();

        let keys: Vec<&String> = record.gate_digests.keys().collect();
        assert_eq!(keys, vec!["g0", "g1", "g2"], "Keys must be sorted");
    }

    #[test]
    fn test_promotion_record_json_roundtrip() {
        let record = PromotionRecord::builder()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id("test-id")
            .manifest("config/signals_manifest.json", TEST_MANIFEST_HASH, "0.1.0")
            .segment_dir("/tmp/segment")
            .gates_passed(true)
            .gates_summary_digest("summary_digest")
            .build();

        let json = record.to_json_bytes();
        let parsed: PromotionRecord = serde_json::from_slice(&json).unwrap();

        assert_eq!(parsed.digest, record.digest);
        assert_eq!(parsed.manifest_hash, TEST_MANIFEST_HASH);
    }

    #[test]
    fn test_gates_summary_json_roundtrip() {
        let g0 = GateOutcome::new("g0", true, "gates/g0_manifest.json", "abc123");
        let summary = GatesSummary::with_timestamp("2026-01-28T12:00:00Z").with_g0(g0);

        let json = summary.to_json_bytes();
        let parsed: GatesSummary = serde_json::from_slice(&json).unwrap();

        assert_eq!(parsed.passed, summary.passed);
        assert_eq!(parsed.g0.as_ref().unwrap().gate, "g0");
    }
}
