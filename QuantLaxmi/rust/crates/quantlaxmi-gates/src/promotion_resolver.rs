//! Promotion Resolver — Runtime helper for checking signal promotion status.
//!
//! Phase 22A: Provides runtime lookup to answer:
//! "Is signal_id promoted under a promotion root?"
//!
//! ## Behavior
//! - Strictly deterministic: same (promotion_root, signal_id) → same result
//! - Best-effort cache for performance
//! - Graceful degradation when disabled
//!
//! ## Promotion Lookup
//! A signal is "promoted" if:
//! 1. `{promotion_root}/{signal_id}/promotion_record.json` exists
//! 2. Record parses successfully as PromotionRecord
//! 3. `record.gates_passed == true`

use crate::promotion_pipeline::{PromotionRecord, promotion_record_path};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// =============================================================================
// PromotionStatus — Result of promotion lookup
// =============================================================================

/// Result of a promotion status lookup.
#[derive(Debug, Clone)]
pub struct PromotionStatus {
    /// Whether the signal is promoted
    pub promoted: bool,

    /// Signal ID queried
    pub signal_id: String,

    /// Hash of promotion record (if found and valid)
    pub record_hash: Option<[u8; 32]>,

    /// Path to promotion record (if found)
    pub record_path: Option<PathBuf>,

    /// Error message if lookup failed
    pub error: Option<String>,
}

impl PromotionStatus {
    /// Create a "promoted" status.
    pub fn promoted(signal_id: &str, record_hash: [u8; 32], record_path: PathBuf) -> Self {
        Self {
            promoted: true,
            signal_id: signal_id.to_string(),
            record_hash: Some(record_hash),
            record_path: Some(record_path),
            error: None,
        }
    }

    /// Create a "not promoted" status (no record found).
    pub fn not_found(signal_id: &str) -> Self {
        Self {
            promoted: false,
            signal_id: signal_id.to_string(),
            record_hash: None,
            record_path: None,
            error: None,
        }
    }

    /// Create a "not promoted" status with error.
    pub fn with_error(signal_id: &str, error: impl Into<String>) -> Self {
        Self {
            promoted: false,
            signal_id: signal_id.to_string(),
            record_hash: None,
            record_path: None,
            error: Some(error.into()),
        }
    }

    /// Create a "not promoted" status (gates failed).
    pub fn gates_failed(signal_id: &str, record_path: PathBuf) -> Self {
        Self {
            promoted: false,
            signal_id: signal_id.to_string(),
            record_hash: None,
            record_path: Some(record_path),
            error: Some("Promotion record exists but gates_passed=false".to_string()),
        }
    }
}

// =============================================================================
// PromotionCacheEntry — Cached promotion record
// =============================================================================

/// Cached promotion record entry.
#[derive(Debug, Clone)]
pub struct PromotionCacheEntry {
    /// The promotion record
    pub record: PromotionRecord,

    /// SHA-256 hash of the record's canonical bytes
    pub record_hash: [u8; 32],

    /// When this entry was loaded
    pub loaded_at: Instant,
}

// =============================================================================
// PromotionResolver — Runtime promotion lookup
// =============================================================================

/// Cached promotion lookup for runtime enforcement.
///
/// Strictly deterministic: same promotion_root + signal_id → same result.
///
/// # Example
/// ```ignore
/// // Disabled mode (dev)
/// let resolver = PromotionResolver::disabled();
/// assert!(!resolver.is_enabled());
///
/// // Enabled mode (production)
/// let resolver = PromotionResolver::new("./promotions")?;
/// let status = resolver.is_promoted("spread");
/// if status.promoted {
///     println!("Signal is promoted with hash: {:?}", status.record_hash);
/// }
/// ```
pub struct PromotionResolver {
    /// Root directory containing promotion artifacts
    promotion_root: Option<PathBuf>,

    /// Cache: signal_id → PromotionCacheEntry
    cache: HashMap<String, PromotionCacheEntry>,

    /// Whether resolver is enabled
    enabled: bool,
}

impl PromotionResolver {
    /// Create a disabled resolver (dev mode, no promotion checking).
    ///
    /// All `is_promoted` calls return `promoted=true` (permissive).
    pub fn disabled() -> Self {
        Self {
            promotion_root: None,
            cache: HashMap::new(),
            enabled: false,
        }
    }

    /// Create an enabled resolver with promotion root.
    ///
    /// The promotion_root directory should contain subdirectories per signal_id,
    /// each with a `promotion/promotion_record.json` file.
    pub fn new(promotion_root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = promotion_root.as_ref().to_path_buf();

        // Verify the directory exists
        if !root.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Promotion root does not exist: {}", root.display()),
            ));
        }

        if !root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Promotion root is not a directory: {}", root.display()),
            ));
        }

        Ok(Self {
            promotion_root: Some(root),
            cache: HashMap::new(),
            enabled: true,
        })
    }

    /// Check if resolver is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the promotion root path (if enabled).
    pub fn promotion_root(&self) -> Option<&Path> {
        self.promotion_root.as_deref()
    }

    /// Check if a signal is promoted.
    ///
    /// If resolver is disabled, returns a permissive status (promoted=true).
    /// If enabled, looks up the promotion record and validates gates_passed.
    pub fn is_promoted(&mut self, signal_id: &str) -> PromotionStatus {
        // Disabled mode: always promoted (permissive for dev)
        if !self.enabled {
            return PromotionStatus {
                promoted: true,
                signal_id: signal_id.to_string(),
                record_hash: None,
                record_path: None,
                error: None,
            };
        }

        // Check cache first
        if let Some(entry) = self.cache.get(signal_id) {
            if entry.record.gates_passed {
                return PromotionStatus::promoted(
                    signal_id,
                    entry.record_hash,
                    self.record_path_for(signal_id),
                );
            } else {
                return PromotionStatus::gates_failed(signal_id, self.record_path_for(signal_id));
            }
        }

        // Load from disk
        self.load_and_check(signal_id)
    }

    /// Get full promotion record if available (for audit).
    pub fn get_record(&mut self, signal_id: &str) -> Option<&PromotionCacheEntry> {
        // Ensure loaded
        if !self.cache.contains_key(signal_id) {
            let _ = self.is_promoted(signal_id);
        }
        self.cache.get(signal_id)
    }

    /// Clear cache (for testing or reload).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Compute the path to a signal's promotion record.
    fn record_path_for(&self, signal_id: &str) -> PathBuf {
        let root = self.promotion_root.as_ref().expect("enabled but no root");
        // Structure: {promotion_root}/{signal_id}/ is the "segment" dir
        // promotion_record_path adds /promotion/promotion_record.json
        let segment_dir = root.join(signal_id);
        promotion_record_path(&segment_dir)
    }

    /// Load promotion record from disk and check status.
    fn load_and_check(&mut self, signal_id: &str) -> PromotionStatus {
        let record_path = self.record_path_for(signal_id);

        // Check if file exists
        if !record_path.exists() {
            return PromotionStatus::not_found(signal_id);
        }

        // Read and parse
        let content = match std::fs::read_to_string(&record_path) {
            Ok(c) => c,
            Err(e) => {
                return PromotionStatus::with_error(
                    signal_id,
                    format!("Failed to read promotion record: {}", e),
                );
            }
        };

        let record: PromotionRecord = match serde_json::from_str(&content) {
            Ok(r) => r,
            Err(e) => {
                return PromotionStatus::with_error(
                    signal_id,
                    format!("Failed to parse promotion record: {}", e),
                );
            }
        };

        // Compute hash from canonical bytes
        use sha2::{Digest, Sha256};
        let canonical = record.canonical_bytes();
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        let result = hasher.finalize();
        let mut record_hash = [0u8; 32];
        record_hash.copy_from_slice(&result);

        // Cache the entry
        let entry = PromotionCacheEntry {
            record: record.clone(),
            record_hash,
            loaded_at: Instant::now(),
        };
        self.cache.insert(signal_id.to_string(), entry);

        // Check gates_passed
        if record.gates_passed {
            PromotionStatus::promoted(signal_id, record_hash, record_path)
        } else {
            PromotionStatus::gates_failed(signal_id, record_path)
        }
    }
}

impl Default for PromotionResolver {
    fn default() -> Self {
        Self::disabled()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::promotion_pipeline::PromotionRecordBuilder;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_promotion_record(
        temp_dir: &TempDir,
        signal_id: &str,
        gates_passed: bool,
    ) -> PathBuf {
        // Create segment dir: {temp_dir}/{signal_id}
        let segment_dir = temp_dir.path().join(signal_id);
        let promotion_dir = segment_dir.join("promotion");
        fs::create_dir_all(&promotion_dir).unwrap();

        // Create promotion record
        let record = PromotionRecordBuilder::new()
            .timestamp("2026-01-28T12:00:00Z")
            .promotion_id(format!("test-{}", signal_id))
            .manifest("configs/signals_manifest.json", [0x42u8; 32], "0.1.0")
            .segment_dir(segment_dir.to_str().unwrap())
            .thresholds(0.9, 100)
            .gates_passed(gates_passed)
            .gates_summary_digest("test_digest")
            .build();

        let record_path = promotion_dir.join("promotion_record.json");
        let json = serde_json::to_string_pretty(&record).unwrap();
        fs::write(&record_path, json).unwrap();

        record_path
    }

    #[test]
    fn test_disabled_resolver_always_permits() {
        let mut resolver = PromotionResolver::disabled();

        assert!(!resolver.is_enabled());
        assert!(resolver.promotion_root().is_none());

        let status = resolver.is_promoted("any_signal");
        assert!(status.promoted);
        assert_eq!(status.signal_id, "any_signal");
        assert!(status.record_hash.is_none()); // No actual record
        assert!(status.error.is_none());
    }

    #[test]
    fn test_enabled_resolver_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        assert!(resolver.is_enabled());

        let status = resolver.is_promoted("nonexistent_signal");
        assert!(!status.promoted);
        assert_eq!(status.signal_id, "nonexistent_signal");
        assert!(status.record_hash.is_none());
        assert!(status.error.is_none()); // Not found is not an error
    }

    #[test]
    fn test_enabled_resolver_finds_promoted() {
        let temp_dir = TempDir::new().unwrap();
        create_test_promotion_record(&temp_dir, "spread", true);

        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        let status = resolver.is_promoted("spread");
        assert!(status.promoted);
        assert_eq!(status.signal_id, "spread");
        assert!(status.record_hash.is_some());
        assert!(status.record_path.is_some());
        assert!(status.error.is_none());
    }

    #[test]
    fn test_enabled_resolver_gates_failed() {
        let temp_dir = TempDir::new().unwrap();
        create_test_promotion_record(&temp_dir, "failed_signal", false);

        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        let status = resolver.is_promoted("failed_signal");
        assert!(!status.promoted);
        assert_eq!(status.signal_id, "failed_signal");
        assert!(status.record_hash.is_none());
        assert!(status.record_path.is_some());
        assert!(status.error.is_some());
        assert!(status.error.unwrap().contains("gates_passed=false"));
    }

    #[test]
    fn test_resolver_caches_results() {
        let temp_dir = TempDir::new().unwrap();
        create_test_promotion_record(&temp_dir, "cached_signal", true);

        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        assert_eq!(resolver.cache_size(), 0);

        // First lookup loads from disk
        let status1 = resolver.is_promoted("cached_signal");
        assert!(status1.promoted);
        assert_eq!(resolver.cache_size(), 1);

        // Second lookup uses cache
        let status2 = resolver.is_promoted("cached_signal");
        assert!(status2.promoted);
        assert_eq!(resolver.cache_size(), 1); // Still 1

        // Hashes should match
        assert_eq!(status1.record_hash, status2.record_hash);
    }

    #[test]
    fn test_resolver_clear_cache() {
        let temp_dir = TempDir::new().unwrap();
        create_test_promotion_record(&temp_dir, "signal1", true);
        create_test_promotion_record(&temp_dir, "signal2", true);

        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        resolver.is_promoted("signal1");
        resolver.is_promoted("signal2");
        assert_eq!(resolver.cache_size(), 2);

        resolver.clear_cache();
        assert_eq!(resolver.cache_size(), 0);
    }

    #[test]
    fn test_get_record() {
        let temp_dir = TempDir::new().unwrap();
        create_test_promotion_record(&temp_dir, "spread", true);

        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        // get_record should load and return the entry
        let entry = resolver.get_record("spread");
        assert!(entry.is_some());

        let entry = entry.unwrap();
        assert!(entry.record.gates_passed);
        assert_eq!(entry.record.promotion_id, "test-spread");
    }

    #[test]
    fn test_resolver_invalid_promotion_root() {
        // Non-existent directory
        let result = PromotionResolver::new("/nonexistent/path/12345");
        assert!(result.is_err());

        // File instead of directory
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("not_a_dir");
        fs::write(&file_path, "content").unwrap();

        let result = PromotionResolver::new(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolver_handles_malformed_json() {
        let temp_dir = TempDir::new().unwrap();

        // Create signal dir with malformed JSON
        let segment_dir = temp_dir.path().join("bad_signal");
        let promotion_dir = segment_dir.join("promotion");
        fs::create_dir_all(&promotion_dir).unwrap();

        let record_path = promotion_dir.join("promotion_record.json");
        fs::write(&record_path, "{ invalid json }").unwrap();

        let mut resolver = PromotionResolver::new(temp_dir.path()).unwrap();

        let status = resolver.is_promoted("bad_signal");
        assert!(!status.promoted);
        assert!(status.error.is_some());
        assert!(status.error.unwrap().contains("Failed to parse"));
    }

    #[test]
    fn test_promotion_status_constructors() {
        let promoted = PromotionStatus::promoted("sig1", [0u8; 32], PathBuf::from("/test"));
        assert!(promoted.promoted);
        assert!(promoted.record_hash.is_some());

        let not_found = PromotionStatus::not_found("sig2");
        assert!(!not_found.promoted);
        assert!(not_found.error.is_none());

        let with_error = PromotionStatus::with_error("sig3", "test error");
        assert!(!with_error.promoted);
        assert_eq!(with_error.error.unwrap(), "test error");

        let gates_failed = PromotionStatus::gates_failed("sig4", PathBuf::from("/test"));
        assert!(!gates_failed.promoted);
        assert!(gates_failed.error.is_some());
    }

    #[test]
    fn test_default_is_disabled() {
        let resolver = PromotionResolver::default();
        assert!(!resolver.is_enabled());
    }
}
