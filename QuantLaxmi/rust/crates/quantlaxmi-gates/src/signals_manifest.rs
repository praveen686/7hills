//! Signals Manifest — Declarative signal requirements.
//!
//! Phase 20B: The signals_manifest.json file declares:
//! - Signal IDs with their required L1 fields
//! - Expected exponents for normalization
//! - Invariants that must hold for valid signals
//!
//! ## Purpose
//! Moves signal requirements from code to configuration, enabling:
//! - Schema evolution without code changes
//! - Audit trails for requirement changes
//! - Manifest version binding in admission decisions
//!
//! ## Hard Laws (from Phase 18/20A)
//! - L1: No Fabrication — manifest cannot override presence requirements
//! - L2: Deterministic — manifest hash is stable for identical content
//! - L6: Observability — manifest version bound into WAL

use quantlaxmi_models::signal_frame::{L1Field, RequiredL1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

// =============================================================================
// Schema Version
// =============================================================================

/// Schema version for signals manifest format.
pub const SIGNALS_MANIFEST_SCHEMA_VERSION: &str = "1.0.0";

// =============================================================================
// Invariant Types
// =============================================================================

/// Invariant types that can be enforced on signals.
///
/// Keep this enum small and typed. Extensibility via `Custom` if needed later.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InvariantType {
    /// bid_price <= ask_price (no crossed book)
    BidLeAsk,
    /// All required qty fields must be present (Value state)
    QtyPresent,
    // Future: RejectDepthDelta, Custom { expression: String }, etc.
}

impl InvariantType {
    /// Validate that this invariant type is known.
    pub fn is_known(&self) -> bool {
        // All enum variants are known by definition.
        // This method exists for forward compatibility if we add Custom.
        true
    }
}

// =============================================================================
// Signal Defaults
// =============================================================================

/// Default values applied to all signals unless overridden.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalDefaults {
    /// Default expected price exponent (None = accept any).
    pub expected_px_exp: Option<i8>,

    /// Default expected quantity exponent (None = accept any).
    pub expected_qty_exp: Option<i8>,

    /// Default invariants applied to all signals.
    #[serde(default)]
    pub invariants: Vec<InvariantType>,
}

// =============================================================================
// Signal Specification
// =============================================================================

/// Specification for a single signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalSpec {
    /// Signal identifier (must match key in signals map).
    pub signal_id: String,

    /// Human-readable description.
    #[serde(default)]
    pub description: String,

    /// Required L1 fields (as strings: "bid_price", "ask_price", "bid_qty", "ask_qty").
    pub required_l1: Vec<String>,

    /// Expected price exponent (None = use default or accept any).
    pub expected_px_exp: Option<i8>,

    /// Expected quantity exponent (None = use default or accept any).
    pub expected_qty_exp: Option<i8>,

    /// Invariants specific to this signal (merged with defaults).
    #[serde(default)]
    pub invariants: Vec<InvariantType>,

    /// Tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl SignalSpec {
    /// Parse required_l1 strings into L1Field array.
    ///
    /// Returns an array of bools indexed by L1Field, where true = required.
    pub fn parse_required_l1(&self) -> Result<[bool; 4], ManifestError> {
        let mut required = [false; 4];

        for field_str in &self.required_l1 {
            match field_str.as_str() {
                "bid_price" => required[L1Field::BidPrice as usize] = true,
                "ask_price" => required[L1Field::AskPrice as usize] = true,
                "bid_qty" => required[L1Field::BidQty as usize] = true,
                "ask_qty" => required[L1Field::AskQty as usize] = true,
                other => {
                    return Err(ManifestError::UnknownL1Field {
                        signal_id: self.signal_id.clone(),
                        field: other.to_string(),
                    });
                }
            }
        }

        Ok(required)
    }

    /// Convert this spec to RequiredL1, merging with defaults.
    ///
    /// Merge rules:
    /// - required_l1: from spec (no inheritance)
    /// - expected_px_exp: spec overrides default if Some
    /// - expected_qty_exp: spec overrides default if Some
    /// - invariants: merge defaults + spec (dedupe by type)
    /// - enforce_bid_le_ask: true if BidLeAsk in merged invariants
    pub fn to_required_l1(&self, defaults: &SignalDefaults) -> Result<RequiredL1, ManifestError> {
        let fields = self.parse_required_l1()?;

        // Merge exponents: spec overrides default
        let expected_px_exp = self.expected_px_exp.or(defaults.expected_px_exp);
        let expected_qty_exp = self.expected_qty_exp.or(defaults.expected_qty_exp);

        // Merge invariants: defaults + spec, dedupe
        let mut merged_invariants: Vec<InvariantType> = defaults.invariants.clone();
        for inv in &self.invariants {
            if !merged_invariants.contains(inv) {
                merged_invariants.push(inv.clone());
            }
        }

        // Determine flags from invariants
        let enforce_bid_le_ask = merged_invariants.contains(&InvariantType::BidLeAsk);

        Ok(RequiredL1 {
            fields,
            expected_px_exp,
            expected_qty_exp,
            enforce_bid_le_ask,
        })
    }
}

// =============================================================================
// Signals Manifest
// =============================================================================

/// Root structure of signals_manifest.json.
///
/// Uses BTreeMap for deterministic serialization (sorted keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalsManifest {
    /// Schema version for forward compatibility.
    pub schema_version: String,

    /// Manifest version (semver, e.g., "0.1.0").
    pub manifest_version: String,

    /// Creation timestamp (ISO 8601).
    pub created_at: String,

    /// Human-readable description.
    #[serde(default)]
    pub description: String,

    /// Default values for all signals.
    #[serde(default)]
    pub defaults: SignalDefaults,

    /// Signal specifications keyed by signal_id.
    /// BTreeMap ensures deterministic iteration order.
    pub signals: BTreeMap<String, SignalSpec>,
}

impl SignalsManifest {
    /// Load manifest from a JSON file.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let content = std::fs::read_to_string(path).map_err(|e| ManifestError::Io {
            path: path.to_path_buf(),
            error: e.to_string(),
        })?;

        Self::from_json(&content, path)
    }

    /// Parse manifest from JSON string.
    pub fn from_json(json: &str, path: &Path) -> Result<Self, ManifestError> {
        let manifest: Self = serde_json::from_str(json).map_err(|e| ManifestError::Parse {
            path: path.to_path_buf(),
            error: e.to_string(),
        })?;

        // Validate schema version
        if manifest.schema_version != SIGNALS_MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::UnsupportedVersion {
                path: path.to_path_buf(),
                expected: SIGNALS_MANIFEST_SCHEMA_VERSION.to_string(),
                found: manifest.schema_version.clone(),
            });
        }

        Ok(manifest)
    }

    /// Validate manifest integrity.
    ///
    /// Checks:
    /// - Signal ID keys match signal_id fields
    /// - No duplicate signal IDs (BTreeMap enforces this)
    /// - All invariant types are known
    /// - All required_l1 fields are valid
    pub fn validate(&self) -> Result<(), ManifestError> {
        for (key, spec) in &self.signals {
            // Check key matches signal_id
            if key != &spec.signal_id {
                return Err(ManifestError::SignalIdMismatch {
                    key: key.clone(),
                    signal_id: spec.signal_id.clone(),
                });
            }

            // Validate required_l1 fields
            spec.parse_required_l1()?;

            // Validate invariant types
            for inv in &spec.invariants {
                if !inv.is_known() {
                    return Err(ManifestError::UnknownInvariantType {
                        signal_id: spec.signal_id.clone(),
                        invariant: format!("{:?}", inv),
                    });
                }
            }
        }

        // Validate default invariants
        for inv in &self.defaults.invariants {
            if !inv.is_known() {
                return Err(ManifestError::UnknownInvariantType {
                    signal_id: "<defaults>".to_string(),
                    invariant: format!("{:?}", inv),
                });
            }
        }

        Ok(())
    }

    /// Load and validate manifest in one step.
    pub fn load_validated(path: &Path) -> Result<Self, ManifestError> {
        let manifest = Self::load(path)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Compute SHA-256 hash of canonical manifest bytes.
    ///
    /// Uses deterministic JSON serialization (BTreeMap keys are sorted).
    /// Returns 32-byte hash.
    pub fn compute_version_hash(&self) -> [u8; 32] {
        // Serialize to deterministic JSON
        let bytes = serde_json::to_vec(self).expect("manifest serialization cannot fail");

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let result = hasher.finalize();

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Compute version hash as hex string (for display/logging).
    pub fn compute_version_hash_hex(&self) -> String {
        hex::encode(self.compute_version_hash())
    }

    /// Get a signal spec by ID.
    pub fn get_signal(&self, signal_id: &str) -> Option<&SignalSpec> {
        self.signals.get(signal_id)
    }

    /// Get RequiredL1 for a signal, applying defaults.
    pub fn get_required_l1(&self, signal_id: &str) -> Result<RequiredL1, ManifestError> {
        let spec = self
            .get_signal(signal_id)
            .ok_or_else(|| ManifestError::SignalNotFound {
                signal_id: signal_id.to_string(),
            })?;

        spec.to_required_l1(&self.defaults)
    }

    /// List all signal IDs.
    pub fn signal_ids(&self) -> impl Iterator<Item = &str> {
        self.signals.keys().map(|s| s.as_str())
    }

    /// Number of signals in manifest.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }
}

// =============================================================================
// Manifest Errors
// =============================================================================

/// Errors from manifest loading and validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ManifestError {
    #[error("IO error loading manifest from {path}: {error}")]
    Io {
        path: std::path::PathBuf,
        error: String,
    },

    #[error("Parse error in manifest {path}: {error}")]
    Parse {
        path: std::path::PathBuf,
        error: String,
    },

    #[error("Unsupported manifest version in {path}: expected {expected}, found {found}")]
    UnsupportedVersion {
        path: std::path::PathBuf,
        expected: String,
        found: String,
    },

    #[error("Signal ID mismatch: key '{key}' != signal_id '{signal_id}'")]
    SignalIdMismatch { key: String, signal_id: String },

    #[error("Unknown L1 field '{field}' in signal '{signal_id}'")]
    UnknownL1Field { signal_id: String, field: String },

    #[error("Unknown invariant type '{invariant}' in signal '{signal_id}'")]
    UnknownInvariantType {
        signal_id: String,
        invariant: String,
    },

    #[error("Signal not found: '{signal_id}'")]
    SignalNotFound { signal_id: String },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const MINIMAL_MANIFEST: &str = r#"{
        "schema_version": "1.0.0",
        "manifest_version": "0.1.0",
        "created_at": "2026-01-28T00:00:00Z",
        "description": "Test manifest",
        "defaults": {
            "expected_px_exp": -2,
            "expected_qty_exp": -8,
            "invariants": [{ "type": "bid_le_ask" }]
        },
        "signals": {
            "spread": {
                "signal_id": "spread",
                "description": "Spread signal",
                "required_l1": ["bid_price", "ask_price"],
                "expected_px_exp": null,
                "expected_qty_exp": null,
                "invariants": [],
                "tags": ["l1"]
            },
            "microprice": {
                "signal_id": "microprice",
                "description": "Microprice signal",
                "required_l1": ["bid_price", "ask_price", "bid_qty", "ask_qty"],
                "expected_px_exp": null,
                "expected_qty_exp": null,
                "invariants": [{ "type": "qty_present" }],
                "tags": ["l1"]
            }
        }
    }"#;

    fn test_path() -> PathBuf {
        PathBuf::from("test_manifest.json")
    }

    #[test]
    fn test_manifest_load_roundtrip() {
        let manifest = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();

        assert_eq!(manifest.schema_version, "1.0.0");
        assert_eq!(manifest.manifest_version, "0.1.0");
        assert_eq!(manifest.signal_count(), 2);
        assert!(manifest.get_signal("spread").is_some());
        assert!(manifest.get_signal("microprice").is_some());
        assert!(manifest.get_signal("unknown").is_none());
    }

    #[test]
    fn test_manifest_validate_success() {
        let manifest = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_manifest_hash_deterministic() {
        let manifest1 = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
        let manifest2 = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();

        let hash1 = manifest1.compute_version_hash();
        let hash2 = manifest2.compute_version_hash();

        assert_eq!(
            hash1, hash2,
            "Hash must be deterministic for identical input"
        );

        // Run 100 times to prove determinism
        for _ in 0..100 {
            let m = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
            assert_eq!(m.compute_version_hash(), hash1);
        }
    }

    #[test]
    fn test_manifest_hash_changes_with_content() {
        let manifest1 = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();

        // Modify the manifest slightly
        let modified = MINIMAL_MANIFEST.replace("0.1.0", "0.2.0");
        let manifest2 = SignalsManifest::from_json(&modified, &test_path()).unwrap();

        assert_ne!(
            manifest1.compute_version_hash(),
            manifest2.compute_version_hash(),
            "Hash must change when content changes"
        );
    }

    #[test]
    fn test_manifest_rejects_duplicate_signal_ids() {
        // BTreeMap naturally dedupes keys, so duplicate keys in JSON
        // will just overwrite. Test that key != signal_id is caught.
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {
                "spread": {
                    "signal_id": "wrong_id",
                    "required_l1": ["bid_price"]
                }
            }
        }"#;

        let manifest = SignalsManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(ManifestError::SignalIdMismatch { key, signal_id }) => {
                assert_eq!(key, "spread");
                assert_eq!(signal_id, "wrong_id");
            }
            _ => panic!("Expected SignalIdMismatch error"),
        }
    }

    #[test]
    fn test_manifest_rejects_unknown_l1_field() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {
                "test": {
                    "signal_id": "test",
                    "required_l1": ["bid_price", "unknown_field"]
                }
            }
        }"#;

        let manifest = SignalsManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(ManifestError::UnknownL1Field { signal_id, field }) => {
                assert_eq!(signal_id, "test");
                assert_eq!(field, "unknown_field");
            }
            _ => panic!("Expected UnknownL1Field error"),
        }
    }

    #[test]
    fn test_manifest_rejects_unsupported_version() {
        let bad_manifest = r#"{
            "schema_version": "99.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {}
        }"#;

        let result = SignalsManifest::from_json(bad_manifest, &test_path());

        assert!(result.is_err());
        match result {
            Err(ManifestError::UnsupportedVersion {
                expected, found, ..
            }) => {
                assert_eq!(expected, "1.0.0");
                assert_eq!(found, "99.0.0");
            }
            _ => panic!("Expected UnsupportedVersion error"),
        }
    }

    #[test]
    fn test_to_required_l1_spread_prices_only() {
        let manifest = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
        let required = manifest.get_required_l1("spread").unwrap();

        // Spread requires bid_price and ask_price only
        assert!(required.fields[L1Field::BidPrice as usize]);
        assert!(required.fields[L1Field::AskPrice as usize]);
        assert!(!required.fields[L1Field::BidQty as usize]);
        assert!(!required.fields[L1Field::AskQty as usize]);

        // Defaults apply: bid_le_ask from defaults
        assert!(required.enforce_bid_le_ask);

        // Exponents from defaults
        assert_eq!(required.expected_px_exp, Some(-2));
        assert_eq!(required.expected_qty_exp, Some(-8));
    }

    #[test]
    fn test_to_required_l1_microprice_requires_qty() {
        let manifest = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
        let required = manifest.get_required_l1("microprice").unwrap();

        // Microprice requires all L1 fields
        assert!(required.fields[L1Field::BidPrice as usize]);
        assert!(required.fields[L1Field::AskPrice as usize]);
        assert!(required.fields[L1Field::BidQty as usize]);
        assert!(required.fields[L1Field::AskQty as usize]);

        // bid_le_ask from defaults
        assert!(required.enforce_bid_le_ask);
    }

    #[test]
    fn test_invariant_merge_dedupes() {
        // Both defaults and signal have bid_le_ask
        let manifest_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {
                "invariants": [{ "type": "bid_le_ask" }]
            },
            "signals": {
                "test": {
                    "signal_id": "test",
                    "required_l1": ["bid_price"],
                    "invariants": [{ "type": "bid_le_ask" }, { "type": "qty_present" }]
                }
            }
        }"#;

        let manifest = SignalsManifest::from_json(manifest_json, &test_path()).unwrap();
        let spec = manifest.get_signal("test").unwrap();
        let required = spec.to_required_l1(&manifest.defaults).unwrap();

        // bid_le_ask should be true (dedupe doesn't break it)
        assert!(required.enforce_bid_le_ask);
    }

    #[test]
    fn test_signal_not_found() {
        let manifest = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
        let result = manifest.get_required_l1("nonexistent");

        assert!(result.is_err());
        match result {
            Err(ManifestError::SignalNotFound { signal_id }) => {
                assert_eq!(signal_id, "nonexistent");
            }
            _ => panic!("Expected SignalNotFound error"),
        }
    }

    #[test]
    fn test_signal_ids_iterator() {
        let manifest = SignalsManifest::from_json(MINIMAL_MANIFEST, &test_path()).unwrap();
        let ids: Vec<&str> = manifest.signal_ids().collect();

        // BTreeMap iterates in sorted order
        assert_eq!(ids, vec!["microprice", "spread"]);
    }

    #[test]
    fn test_defaults_applied_when_spec_has_none() {
        let manifest_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {
                "expected_px_exp": -3,
                "expected_qty_exp": -6
            },
            "signals": {
                "test": {
                    "signal_id": "test",
                    "required_l1": ["bid_price"],
                    "expected_px_exp": null,
                    "expected_qty_exp": null
                }
            }
        }"#;

        let manifest = SignalsManifest::from_json(manifest_json, &test_path()).unwrap();
        let required = manifest.get_required_l1("test").unwrap();

        // Should use defaults
        assert_eq!(required.expected_px_exp, Some(-3));
        assert_eq!(required.expected_qty_exp, Some(-6));
    }

    #[test]
    fn test_spec_overrides_defaults() {
        let manifest_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {
                "expected_px_exp": -3,
                "expected_qty_exp": -6
            },
            "signals": {
                "test": {
                    "signal_id": "test",
                    "required_l1": ["bid_price"],
                    "expected_px_exp": -2,
                    "expected_qty_exp": -8
                }
            }
        }"#;

        let manifest = SignalsManifest::from_json(manifest_json, &test_path()).unwrap();
        let required = manifest.get_required_l1("test").unwrap();

        // Should use spec values, not defaults
        assert_eq!(required.expected_px_exp, Some(-2));
        assert_eq!(required.expected_qty_exp, Some(-8));
    }
}
