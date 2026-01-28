//! Strategies Manifest — Declarative execution contracts.
//!
//! Phase 21A: The strategies_manifest.json file declares:
//! - Strategy IDs with their required signals
//! - Execution class (advisory/passive/aggressive)
//! - Execution constraints (order limits, position limits)
//! - Allowed behaviors (short, long, market orders)
//!
//! ## Hard Law
//! A signal cannot influence execution unless:
//! 1. It is promoted (20D), AND
//! 2. It is bound to a strategy in strategies_manifest.json
//!
//! ## Frozen Surfaces (Phase 21A)
//! - STRATEGIES_MANIFEST_SCHEMA_VERSION: "1.0.0"
//! - ExecutionClass variants + canonical bytes
//! - StrategiesManifest canonical bytes field order
//! - StrategySpec canonical bytes field order
//! - StrategyDefaults canonical bytes field order
//! - Signals/tags sorted for hash
//! - Defaults are informational only (v1)
//! - Advisory constraint rules
//! - Passive constraint rules

use crate::signals_manifest::SignalsManifest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

// =============================================================================
// Schema Version
// =============================================================================

/// Schema version for strategies manifest format.
pub const STRATEGIES_MANIFEST_SCHEMA_VERSION: &str = "1.0.0";

// =============================================================================
// Execution Class
// =============================================================================

/// Execution class — what kind of orders can this strategy emit?
///
/// Frozen: Do not reorder variants or change canonical bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionClass {
    /// Emits opinions only — no orders allowed
    Advisory,
    /// Limit orders only — no market orders
    Passive,
    /// Market orders allowed (or not, per allow_market_orders)
    Aggressive,
}

impl ExecutionClass {
    /// Canonical byte representation (frozen).
    ///
    /// Used in hash computation. Do NOT change these values.
    pub fn canonical_byte(self) -> u8 {
        match self {
            ExecutionClass::Advisory => 0x01,
            ExecutionClass::Passive => 0x02,
            ExecutionClass::Aggressive => 0x03,
        }
    }

    /// Parse from string (for error messages).
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionClass::Advisory => "advisory",
            ExecutionClass::Passive => "passive",
            ExecutionClass::Aggressive => "aggressive",
        }
    }
}

// =============================================================================
// Strategy Defaults
// =============================================================================

/// Default strategy constraints.
///
/// **v1 semantics:** Informational only. StrategySpec must be fully explicit.
/// Defaults are included in hash but do NOT apply to StrategySpec fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyDefaults {
    pub max_orders_per_min: Option<u32>,
    pub allow_short: Option<bool>,
    pub allow_long: Option<bool>,
    pub allow_market_orders: Option<bool>,
}

impl StrategyDefaults {
    /// Compute canonical bytes for hashing (frozen field order).
    ///
    /// Field order:
    /// 1. max_orders_per_min: 0x00 if None, 0x01 + u32 LE if Some
    /// 2. allow_short: 0x00 if None, 0x01 + u8 (0/1) if Some
    /// 3. allow_long: 0x00 if None, 0x01 + u8 (0/1) if Some
    /// 4. allow_market_orders: 0x00 if None, 0x01 + u8 (0/1) if Some
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. max_orders_per_min
        match self.max_orders_per_min {
            None => bytes.push(0x00),
            Some(v) => {
                bytes.push(0x01);
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }

        // 2. allow_short
        match self.allow_short {
            None => bytes.push(0x00),
            Some(v) => {
                bytes.push(0x01);
                bytes.push(if v { 1 } else { 0 });
            }
        }

        // 3. allow_long
        match self.allow_long {
            None => bytes.push(0x00),
            Some(v) => {
                bytes.push(0x01);
                bytes.push(if v { 1 } else { 0 });
            }
        }

        // 4. allow_market_orders
        match self.allow_market_orders {
            None => bytes.push(0x00),
            Some(v) => {
                bytes.push(0x01);
                bytes.push(if v { 1 } else { 0 });
            }
        }

        bytes
    }
}

// =============================================================================
// Strategy Specification
// =============================================================================

/// Individual strategy specification.
///
/// All fields are REQUIRED and explicit (defaults do not apply in v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySpec {
    pub strategy_id: String,

    #[serde(default)]
    pub description: String,

    // === Signal binding (non-empty) ===
    pub signals: Vec<String>,

    // === Execution intent ===
    pub execution_class: ExecutionClass,
    pub max_orders_per_min: u32,
    pub max_position_abs: i64,

    // === Constraints (all explicit) ===
    pub allow_short: bool,
    pub allow_long: bool,
    pub allow_market_orders: bool,

    #[serde(default)]
    pub tags: Vec<String>,
}

impl StrategySpec {
    /// Compute canonical bytes for hashing (frozen field order).
    ///
    /// Field order:
    /// 1. strategy_id (u32 LE len + UTF-8)
    /// 2. description (u32 LE len + UTF-8)
    /// 3. signals (u32 LE count + each string **sorted lexicographically**, u32 len + UTF-8)
    /// 4. execution_class (1 byte)
    /// 5. max_orders_per_min (u32 LE)
    /// 6. max_position_abs (i64 LE)
    /// 7. allow_short (u8: 0 or 1)
    /// 8. allow_long (u8: 0 or 1)
    /// 9. allow_market_orders (u8: 0 or 1)
    /// 10. tags (u32 LE count + each string **sorted lexicographically**, u32 len + UTF-8)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. strategy_id
        write_string(&mut bytes, &self.strategy_id);

        // 2. description
        write_string(&mut bytes, &self.description);

        // 3. signals (sorted for hash stability)
        let mut sorted_signals = self.signals.clone();
        sorted_signals.sort();
        bytes.extend_from_slice(&(sorted_signals.len() as u32).to_le_bytes());
        for sig in &sorted_signals {
            write_string(&mut bytes, sig);
        }

        // 4. execution_class
        bytes.push(self.execution_class.canonical_byte());

        // 5. max_orders_per_min
        bytes.extend_from_slice(&self.max_orders_per_min.to_le_bytes());

        // 6. max_position_abs
        bytes.extend_from_slice(&self.max_position_abs.to_le_bytes());

        // 7. allow_short
        bytes.push(if self.allow_short { 1 } else { 0 });

        // 8. allow_long
        bytes.push(if self.allow_long { 1 } else { 0 });

        // 9. allow_market_orders
        bytes.push(if self.allow_market_orders { 1 } else { 0 });

        // 10. tags (sorted for hash stability)
        let mut sorted_tags = self.tags.clone();
        sorted_tags.sort();
        bytes.extend_from_slice(&(sorted_tags.len() as u32).to_le_bytes());
        for tag in &sorted_tags {
            write_string(&mut bytes, tag);
        }

        bytes
    }
}

// =============================================================================
// Strategies Manifest
// =============================================================================

/// Root manifest structure.
///
/// Uses BTreeMap for deterministic serialization (sorted keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategiesManifest {
    pub schema_version: String,
    pub manifest_version: String,
    pub created_at: String,

    #[serde(default)]
    pub description: String,

    /// Informational only in v1; StrategySpec must be fully explicit.
    pub defaults: Option<StrategyDefaults>,

    /// Strategy specifications keyed by strategy_id.
    /// BTreeMap ensures deterministic iteration order.
    pub strategies: BTreeMap<String, StrategySpec>,
}

impl StrategiesManifest {
    /// Load manifest from a JSON file.
    pub fn load(path: &Path) -> Result<Self, StrategiesManifestError> {
        let content = std::fs::read_to_string(path).map_err(|e| StrategiesManifestError::Io {
            path: path.to_path_buf(),
            error: e.to_string(),
        })?;

        Self::from_json(&content, path)
    }

    /// Parse manifest from JSON string.
    pub fn from_json(json: &str, path: &Path) -> Result<Self, StrategiesManifestError> {
        let manifest: Self =
            serde_json::from_str(json).map_err(|e| StrategiesManifestError::Parse {
                path: path.to_path_buf(),
                error: e.to_string(),
            })?;

        // Validate schema version
        if manifest.schema_version != STRATEGIES_MANIFEST_SCHEMA_VERSION {
            return Err(StrategiesManifestError::UnsupportedVersion {
                path: path.to_path_buf(),
                expected: STRATEGIES_MANIFEST_SCHEMA_VERSION.to_string(),
                found: manifest.schema_version.clone(),
            });
        }

        Ok(manifest)
    }

    /// Validate manifest integrity.
    ///
    /// Checks:
    /// 1. schema_version == "1.0.0"
    /// 2. Key == spec.strategy_id (no mismatch)
    /// 3. signals non-empty
    /// 4. Execution-class constraints:
    ///    - Advisory: all limits=0, all allows=false
    ///    - Passive: allow_market_orders == false
    /// 5. Sanity limits:
    ///    - max_orders_per_min <= 1000
    ///    - Non-advisory: max_position_abs > 0
    pub fn validate(&self) -> Result<(), StrategiesManifestError> {
        for (key, spec) in &self.strategies {
            // 2. Key == spec.strategy_id
            if key != &spec.strategy_id {
                return Err(StrategiesManifestError::StrategyIdMismatch {
                    key: key.clone(),
                    spec_id: spec.strategy_id.clone(),
                });
            }

            // 3. signals non-empty
            if spec.signals.is_empty() {
                return Err(StrategiesManifestError::EmptySignals {
                    strategy_id: spec.strategy_id.clone(),
                });
            }

            // 4. Execution-class constraints
            match spec.execution_class {
                ExecutionClass::Advisory => {
                    // Advisory: all limits=0, all allows=false
                    if spec.max_orders_per_min != 0 {
                        return Err(StrategiesManifestError::AdvisoryConstraintViolation {
                            strategy_id: spec.strategy_id.clone(),
                            field: "max_orders_per_min".to_string(),
                            expected: "0".to_string(),
                            actual: spec.max_orders_per_min.to_string(),
                        });
                    }
                    if spec.max_position_abs != 0 {
                        return Err(StrategiesManifestError::AdvisoryConstraintViolation {
                            strategy_id: spec.strategy_id.clone(),
                            field: "max_position_abs".to_string(),
                            expected: "0".to_string(),
                            actual: spec.max_position_abs.to_string(),
                        });
                    }
                    if spec.allow_short {
                        return Err(StrategiesManifestError::AdvisoryConstraintViolation {
                            strategy_id: spec.strategy_id.clone(),
                            field: "allow_short".to_string(),
                            expected: "false".to_string(),
                            actual: "true".to_string(),
                        });
                    }
                    if spec.allow_long {
                        return Err(StrategiesManifestError::AdvisoryConstraintViolation {
                            strategy_id: spec.strategy_id.clone(),
                            field: "allow_long".to_string(),
                            expected: "false".to_string(),
                            actual: "true".to_string(),
                        });
                    }
                    if spec.allow_market_orders {
                        return Err(StrategiesManifestError::AdvisoryConstraintViolation {
                            strategy_id: spec.strategy_id.clone(),
                            field: "allow_market_orders".to_string(),
                            expected: "false".to_string(),
                            actual: "true".to_string(),
                        });
                    }
                }
                ExecutionClass::Passive => {
                    // Passive: allow_market_orders == false
                    if spec.allow_market_orders {
                        return Err(StrategiesManifestError::PassiveAllowsMarketOrders {
                            strategy_id: spec.strategy_id.clone(),
                        });
                    }
                }
                ExecutionClass::Aggressive => {
                    // No additional constraints
                }
            }

            // 5. Sanity limits
            if spec.max_orders_per_min > 1000 {
                return Err(StrategiesManifestError::RateLimitTooHigh {
                    strategy_id: spec.strategy_id.clone(),
                    value: spec.max_orders_per_min,
                });
            }

            // Non-advisory must have positive position limit
            if spec.execution_class != ExecutionClass::Advisory && spec.max_position_abs <= 0 {
                return Err(StrategiesManifestError::ZeroPositionLimit {
                    strategy_id: spec.strategy_id.clone(),
                    execution_class: spec.execution_class.as_str().to_string(),
                });
            }
        }

        Ok(())
    }

    /// Validate signal bindings against a signals manifest.
    ///
    /// Checks that every signal referenced in strategies exists in the signals manifest.
    pub fn validate_signal_bindings(
        &self,
        signals_manifest: &SignalsManifest,
    ) -> Result<(), StrategiesManifestError> {
        for spec in self.strategies.values() {
            for signal_id in &spec.signals {
                if signals_manifest.get_signal(signal_id).is_none() {
                    return Err(StrategiesManifestError::UnknownSignal {
                        strategy_id: spec.strategy_id.clone(),
                        signal_id: signal_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Load and validate manifest in one step.
    pub fn load_validated(path: &Path) -> Result<Self, StrategiesManifestError> {
        let manifest = Self::load(path)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Compute canonical bytes for hashing (frozen field order).
    ///
    /// Field order:
    /// 1. schema_version (u32 LE len + UTF-8 bytes)
    /// 2. manifest_version (u32 LE len + UTF-8 bytes)
    /// 3. created_at (u32 LE len + UTF-8 bytes)
    /// 4. description (u32 LE len + UTF-8 bytes)
    /// 5. defaults presence tag: 0x00 if None, 0x01 + StrategyDefaults bytes if Some
    /// 6. strategies (u32 LE count + for each key in sorted order: key string + StrategySpec bytes)
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // 1. schema_version
        write_string(&mut bytes, &self.schema_version);

        // 2. manifest_version
        write_string(&mut bytes, &self.manifest_version);

        // 3. created_at
        write_string(&mut bytes, &self.created_at);

        // 4. description
        write_string(&mut bytes, &self.description);

        // 5. defaults
        match &self.defaults {
            None => bytes.push(0x00),
            Some(defaults) => {
                bytes.push(0x01);
                bytes.extend_from_slice(&defaults.canonical_bytes());
            }
        }

        // 6. strategies (BTreeMap ensures sorted order)
        bytes.extend_from_slice(&(self.strategies.len() as u32).to_le_bytes());
        for (key, spec) in &self.strategies {
            write_string(&mut bytes, key);
            bytes.extend_from_slice(&spec.canonical_bytes());
        }

        bytes
    }

    /// Compute SHA-256 hash of canonical bytes.
    pub fn compute_version_hash(&self) -> [u8; 32] {
        let bytes = self.canonical_bytes();
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

    /// Get a strategy spec by ID.
    pub fn get_strategy(&self, strategy_id: &str) -> Option<&StrategySpec> {
        self.strategies.get(strategy_id)
    }

    /// List all strategy IDs.
    pub fn strategy_ids(&self) -> impl Iterator<Item = &str> {
        self.strategies.keys().map(|s| s.as_str())
    }

    /// Number of strategies in manifest.
    pub fn strategy_count(&self) -> usize {
        self.strategies.len()
    }

    /// Total signal binding count (sum of signals across all strategies).
    pub fn signal_binding_count(&self) -> usize {
        self.strategies.values().map(|s| s.signals.len()).sum()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Write a string with length prefix (u32 LE len + UTF-8 bytes).
fn write_string(bytes: &mut Vec<u8>, s: &str) {
    bytes.extend_from_slice(&(s.len() as u32).to_le_bytes());
    bytes.extend_from_slice(s.as_bytes());
}

// =============================================================================
// Manifest Errors
// =============================================================================

/// Errors from strategies manifest loading and validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StrategiesManifestError {
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

    #[error("Strategy ID mismatch: key '{key}' != strategy_id '{spec_id}'")]
    StrategyIdMismatch { key: String, spec_id: String },

    #[error("Empty signals array in strategy '{strategy_id}'")]
    EmptySignals { strategy_id: String },

    #[error("Unknown signal '{signal_id}' in strategy '{strategy_id}'")]
    UnknownSignal {
        strategy_id: String,
        signal_id: String,
    },

    #[error(
        "Advisory constraint violation in '{strategy_id}': {field} expected {expected}, got {actual}"
    )]
    AdvisoryConstraintViolation {
        strategy_id: String,
        field: String,
        expected: String,
        actual: String,
    },

    #[error("Passive strategy '{strategy_id}' cannot allow market orders")]
    PassiveAllowsMarketOrders { strategy_id: String },

    #[error("Rate limit too high in '{strategy_id}': {value} > 1000")]
    RateLimitTooHigh { strategy_id: String, value: u32 },

    #[error(
        "Non-advisory strategy '{strategy_id}' (class: {execution_class}) must have max_position_abs > 0"
    )]
    ZeroPositionLimit {
        strategy_id: String,
        execution_class: String,
    },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const VALID_MANIFEST: &str = r#"{
        "schema_version": "1.0.0",
        "manifest_version": "0.1.0",
        "created_at": "2026-01-28T00:00:00Z",
        "description": "Test manifest",
        "defaults": {
            "max_orders_per_min": 60,
            "allow_short": true,
            "allow_long": true,
            "allow_market_orders": false
        },
        "strategies": {
            "spread_passive": {
                "strategy_id": "spread_passive",
                "description": "Passive spread strategy",
                "signals": ["spread"],
                "execution_class": "passive",
                "max_orders_per_min": 120,
                "max_position_abs": 10000,
                "allow_short": true,
                "allow_long": true,
                "allow_market_orders": false,
                "tags": ["mm", "passive"]
            },
            "imbalance_advisory": {
                "strategy_id": "imbalance_advisory",
                "description": "Advisory only",
                "signals": ["book_imbalance"],
                "execution_class": "advisory",
                "max_orders_per_min": 0,
                "max_position_abs": 0,
                "allow_short": false,
                "allow_long": false,
                "allow_market_orders": false,
                "tags": ["advisory"]
            }
        }
    }"#;

    fn test_path() -> PathBuf {
        PathBuf::from("test_strategies_manifest.json")
    }

    #[test]
    fn test_manifest_load_roundtrip() {
        let manifest = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();

        assert_eq!(manifest.schema_version, "1.0.0");
        assert_eq!(manifest.manifest_version, "0.1.0");
        assert_eq!(manifest.strategy_count(), 2);
        assert!(manifest.get_strategy("spread_passive").is_some());
        assert!(manifest.get_strategy("imbalance_advisory").is_some());
        assert!(manifest.get_strategy("unknown").is_none());
    }

    #[test]
    fn test_manifest_validate_success() {
        let manifest = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_manifest_hash_deterministic() {
        let manifest1 = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();
        let manifest2 = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();

        let hash1 = manifest1.compute_version_hash();
        let hash2 = manifest2.compute_version_hash();

        assert_eq!(
            hash1, hash2,
            "Hash must be deterministic for identical input"
        );

        // Run 100 times to prove determinism
        for _ in 0..100 {
            let m = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();
            assert_eq!(m.compute_version_hash(), hash1);
        }
    }

    #[test]
    fn test_manifest_hash_changes_with_content() {
        let manifest1 = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();

        // Modify the manifest slightly
        let modified = VALID_MANIFEST.replace("0.1.0", "0.2.0");
        let manifest2 = StrategiesManifest::from_json(&modified, &test_path()).unwrap();

        assert_ne!(
            manifest1.compute_version_hash(),
            manifest2.compute_version_hash(),
            "Hash must change when content changes"
        );
    }

    #[test]
    fn test_hash_stable_under_signals_reorder() {
        // signals array: ["spread", "microprice"] vs ["microprice", "spread"]
        let manifest1_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread", "microprice"],
                    "execution_class": "aggressive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 1000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true,
                    "tags": []
                }
            }
        }"#;

        let manifest2_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["microprice", "spread"],
                    "execution_class": "aggressive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 1000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true,
                    "tags": []
                }
            }
        }"#;

        let m1 = StrategiesManifest::from_json(manifest1_json, &test_path()).unwrap();
        let m2 = StrategiesManifest::from_json(manifest2_json, &test_path()).unwrap();

        assert_eq!(
            m1.compute_version_hash(),
            m2.compute_version_hash(),
            "Hash must be stable under signals reorder (sorted for hash)"
        );
    }

    #[test]
    fn test_hash_stable_under_tags_reorder() {
        let manifest1_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "passive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 1000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false,
                    "tags": ["alpha", "beta", "gamma"]
                }
            }
        }"#;

        let manifest2_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "passive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 1000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false,
                    "tags": ["gamma", "alpha", "beta"]
                }
            }
        }"#;

        let m1 = StrategiesManifest::from_json(manifest1_json, &test_path()).unwrap();
        let m2 = StrategiesManifest::from_json(manifest2_json, &test_path()).unwrap();

        assert_eq!(
            m1.compute_version_hash(),
            m2.compute_version_hash(),
            "Hash must be stable under tags reorder (sorted for hash)"
        );
    }

    #[test]
    fn test_manifest_rejects_strategy_id_mismatch() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "spread_passive": {
                    "strategy_id": "wrong_id",
                    "signals": ["spread"],
                    "execution_class": "passive",
                    "max_orders_per_min": 120,
                    "max_position_abs": 10000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::StrategyIdMismatch { key, spec_id }) => {
                assert_eq!(key, "spread_passive");
                assert_eq!(spec_id, "wrong_id");
            }
            _ => panic!("Expected StrategyIdMismatch error"),
        }
    }

    #[test]
    fn test_manifest_rejects_empty_signals() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": [],
                    "execution_class": "passive",
                    "max_orders_per_min": 120,
                    "max_position_abs": 10000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::EmptySignals { strategy_id }) => {
                assert_eq!(strategy_id, "test");
            }
            _ => panic!("Expected EmptySignals error"),
        }
    }

    #[test]
    fn test_manifest_rejects_advisory_with_nonzero_orders() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "advisory",
                    "max_orders_per_min": 10,
                    "max_position_abs": 0,
                    "allow_short": false,
                    "allow_long": false,
                    "allow_market_orders": false
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::AdvisoryConstraintViolation {
                strategy_id,
                field,
                expected,
                actual,
            }) => {
                assert_eq!(strategy_id, "test");
                assert_eq!(field, "max_orders_per_min");
                assert_eq!(expected, "0");
                assert_eq!(actual, "10");
            }
            _ => panic!("Expected AdvisoryConstraintViolation error"),
        }
    }

    #[test]
    fn test_manifest_rejects_advisory_with_allow_short() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "advisory",
                    "max_orders_per_min": 0,
                    "max_position_abs": 0,
                    "allow_short": true,
                    "allow_long": false,
                    "allow_market_orders": false
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::AdvisoryConstraintViolation {
                field,
                expected,
                actual,
                ..
            }) => {
                assert_eq!(field, "allow_short");
                assert_eq!(expected, "false");
                assert_eq!(actual, "true");
            }
            _ => panic!("Expected AdvisoryConstraintViolation error"),
        }
    }

    #[test]
    fn test_manifest_rejects_passive_with_market_orders() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "passive",
                    "max_orders_per_min": 120,
                    "max_position_abs": 10000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::PassiveAllowsMarketOrders { strategy_id }) => {
                assert_eq!(strategy_id, "test");
            }
            _ => panic!("Expected PassiveAllowsMarketOrders error"),
        }
    }

    #[test]
    fn test_manifest_rejects_rate_limit_too_high() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "aggressive",
                    "max_orders_per_min": 1001,
                    "max_position_abs": 10000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::RateLimitTooHigh { strategy_id, value }) => {
                assert_eq!(strategy_id, "test");
                assert_eq!(value, 1001);
            }
            _ => panic!("Expected RateLimitTooHigh error"),
        }
    }

    #[test]
    fn test_manifest_rejects_zero_position_limit_non_advisory() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "passive",
                    "max_orders_per_min": 120,
                    "max_position_abs": 0,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(bad_manifest, &test_path()).unwrap();
        let result = manifest.validate();

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::ZeroPositionLimit {
                strategy_id,
                execution_class,
            }) => {
                assert_eq!(strategy_id, "test");
                assert_eq!(execution_class, "passive");
            }
            _ => panic!("Expected ZeroPositionLimit error"),
        }
    }

    #[test]
    fn test_manifest_rejects_unsupported_version() {
        let bad_manifest = r#"{
            "schema_version": "99.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {}
        }"#;

        let result = StrategiesManifest::from_json(bad_manifest, &test_path());

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::UnsupportedVersion {
                expected, found, ..
            }) => {
                assert_eq!(expected, "1.0.0");
                assert_eq!(found, "99.0.0");
            }
            _ => panic!("Expected UnsupportedVersion error"),
        }
    }

    #[test]
    fn test_execution_class_canonical_bytes() {
        assert_eq!(ExecutionClass::Advisory.canonical_byte(), 0x01);
        assert_eq!(ExecutionClass::Passive.canonical_byte(), 0x02);
        assert_eq!(ExecutionClass::Aggressive.canonical_byte(), 0x03);
    }

    #[test]
    fn test_strategy_defaults_canonical_bytes() {
        let defaults = StrategyDefaults {
            max_orders_per_min: Some(60),
            allow_short: Some(true),
            allow_long: Some(false),
            allow_market_orders: None,
        };

        let bytes = defaults.canonical_bytes();

        // Check format: 0x01 + 60u32 LE + 0x01 + 1u8 + 0x01 + 0u8 + 0x00
        assert_eq!(bytes[0], 0x01); // Some for max_orders_per_min
        assert_eq!(&bytes[1..5], &60u32.to_le_bytes());
        assert_eq!(bytes[5], 0x01); // Some for allow_short
        assert_eq!(bytes[6], 1); // true
        assert_eq!(bytes[7], 0x01); // Some for allow_long
        assert_eq!(bytes[8], 0); // false
        assert_eq!(bytes[9], 0x00); // None for allow_market_orders
    }

    #[test]
    fn test_signal_binding_count() {
        let manifest = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();

        // spread_passive has 1 signal, imbalance_advisory has 1 signal
        assert_eq!(manifest.signal_binding_count(), 2);
    }

    #[test]
    fn test_strategy_ids_sorted() {
        let manifest = StrategiesManifest::from_json(VALID_MANIFEST, &test_path()).unwrap();
        let ids: Vec<&str> = manifest.strategy_ids().collect();

        // BTreeMap iterates in sorted order
        assert_eq!(ids, vec!["imbalance_advisory", "spread_passive"]);
    }

    #[test]
    fn test_aggressive_allows_market_orders() {
        let manifest_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "aggressive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 5000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true,
                    "tags": []
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(manifest_json, &test_path()).unwrap();
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_none_defaults_canonical_bytes() {
        let defaults = StrategyDefaults::default();
        let bytes = defaults.canonical_bytes();

        // All None: 0x00 + 0x00 + 0x00 + 0x00
        assert_eq!(bytes, vec![0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_manifest_without_defaults() {
        let manifest_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread"],
                    "execution_class": "advisory",
                    "max_orders_per_min": 0,
                    "max_position_abs": 0,
                    "allow_short": false,
                    "allow_long": false,
                    "allow_market_orders": false
                }
            }
        }"#;

        let manifest = StrategiesManifest::from_json(manifest_json, &test_path()).unwrap();
        assert!(manifest.defaults.is_none());
        assert!(manifest.validate().is_ok());

        // Hash should still be deterministic
        let hash1 = manifest.compute_version_hash();
        let hash2 = manifest.compute_version_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_signal_binding_validation_rejects_unknown() {
        use crate::signals_manifest::SignalsManifest;

        // Create a signals manifest with only "spread"
        let signals_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {
                "spread": {
                    "signal_id": "spread",
                    "required_l1": ["bid_price", "ask_price"]
                }
            }
        }"#;

        // Create a strategies manifest referencing unknown signal
        let strategies_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread", "unknown_signal"],
                    "execution_class": "aggressive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 1000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true
                }
            }
        }"#;

        let signals_manifest =
            SignalsManifest::from_json(signals_json, &PathBuf::from("signals.json")).unwrap();
        let strategies_manifest =
            StrategiesManifest::from_json(strategies_json, &test_path()).unwrap();

        let result = strategies_manifest.validate_signal_bindings(&signals_manifest);

        assert!(result.is_err());
        match result {
            Err(StrategiesManifestError::UnknownSignal {
                strategy_id,
                signal_id,
            }) => {
                assert_eq!(strategy_id, "test");
                assert_eq!(signal_id, "unknown_signal");
            }
            _ => panic!("Expected UnknownSignal error"),
        }
    }

    #[test]
    fn test_signal_binding_validation_passes() {
        use crate::signals_manifest::SignalsManifest;

        // Create a signals manifest with signals
        let signals_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {
                "spread": {
                    "signal_id": "spread",
                    "required_l1": ["bid_price", "ask_price"]
                },
                "microprice": {
                    "signal_id": "microprice",
                    "required_l1": ["bid_price", "ask_price", "bid_qty", "ask_qty"]
                }
            }
        }"#;

        // Create a strategies manifest referencing known signals
        let strategies_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["spread", "microprice"],
                    "execution_class": "aggressive",
                    "max_orders_per_min": 30,
                    "max_position_abs": 1000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": true
                }
            }
        }"#;

        let signals_manifest =
            SignalsManifest::from_json(signals_json, &PathBuf::from("signals.json")).unwrap();
        let strategies_manifest =
            StrategiesManifest::from_json(strategies_json, &test_path()).unwrap();

        assert!(
            strategies_manifest
                .validate_signal_bindings(&signals_manifest)
                .is_ok()
        );
    }

    #[test]
    fn test_load_real_strategies_manifest() {
        // Test loading the actual config file
        let manifest_path = PathBuf::from("config/strategies_manifest.json");
        if !manifest_path.exists() {
            // Skip if running from wrong directory
            return;
        }

        let manifest = StrategiesManifest::load_validated(&manifest_path).unwrap();

        // Verify expected strategies exist
        assert_eq!(manifest.strategy_count(), 3);
        assert!(manifest.get_strategy("spread_passive").is_some());
        assert!(manifest.get_strategy("microprice_aggressive").is_some());
        assert!(manifest.get_strategy("imbalance_advisory").is_some());

        // Verify hash is deterministic
        let hash1 = manifest.compute_version_hash();
        let manifest2 = StrategiesManifest::load_validated(&manifest_path).unwrap();
        let hash2 = manifest2.compute_version_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_real_manifests_signal_binding() {
        // Test signal binding validation with real config files
        use crate::signals_manifest::SignalsManifest;

        let signals_path = PathBuf::from("config/signals_manifest.json");
        let strategies_path = PathBuf::from("config/strategies_manifest.json");

        if !signals_path.exists() || !strategies_path.exists() {
            // Skip if running from wrong directory
            return;
        }

        let signals_manifest = SignalsManifest::load_validated(&signals_path).unwrap();
        let strategies_manifest = StrategiesManifest::load_validated(&strategies_path).unwrap();

        // All signals in strategies should exist in signals manifest
        let result = strategies_manifest.validate_signal_bindings(&signals_manifest);
        assert!(
            result.is_ok(),
            "Signal binding validation failed: {:?}",
            result
        );
    }
}
