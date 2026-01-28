//! Strategy Admission Engine — Runtime enforcement for strategy-signal binding.
//!
//! Phase 22B: For each candidate (strategy_id, signal_id, correlation_id):
//! 1. Validates signal exists in signals_manifest
//! 2. Validates strategy exists in strategies_manifest
//! 3. Validates signal is bound to strategy
//! 4. Optionally validates signal is promoted
//! 5. Writes WAL event with canonical digest (21B)
//! 6. Returns Admit/Refuse verdict
//!
//! ## Hard Law
//! Strategies do NOT get to "decide" if they can act. They receive the verdict.
//!
//! ## WAL Format
//! Output: `wal/strategy_admission.jsonl`
//! Each line is a JSON-serialized StrategyAdmissionDecision.

use crate::promotion_resolver::{PromotionResolver, PromotionStatus};
use crate::signals_manifest::SignalsManifest;
use crate::strategies_manifest::{StrategiesManifest, StrategySpec};
use quantlaxmi_models::{
    StrategyAdmissionDecision, StrategyAdmissionOutcome, StrategyRefuseReason,
};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for StrategyAdmissionEngine.
#[derive(Debug, Clone)]
pub struct StrategyAdmissionEngineConfig {
    /// Path to strategies_manifest.json
    pub strategies_manifest_path: PathBuf,

    /// Path to signals_manifest.json
    pub signals_manifest_path: PathBuf,

    /// Optional promotion root directory
    pub promotion_root: Option<PathBuf>,

    /// WAL output path (e.g., "wal/strategy_admission.jsonl")
    pub wal_path: PathBuf,

    /// Session ID for all decisions
    pub session_id: String,

    /// Whether to require signal promotion for admission
    pub require_promotion: bool,
}

impl StrategyAdmissionEngineConfig {
    /// Create config with defaults.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            strategies_manifest_path: PathBuf::from("config/strategies_manifest.json"),
            signals_manifest_path: PathBuf::from("config/signals_manifest.json"),
            promotion_root: None,
            wal_path: PathBuf::from("wal/strategy_admission.jsonl"),
            session_id: session_id.into(),
            require_promotion: false,
        }
    }

    /// Set strategies manifest path.
    pub fn strategies_manifest(mut self, path: impl Into<PathBuf>) -> Self {
        self.strategies_manifest_path = path.into();
        self
    }

    /// Set signals manifest path.
    pub fn signals_manifest(mut self, path: impl Into<PathBuf>) -> Self {
        self.signals_manifest_path = path.into();
        self
    }

    /// Set promotion root.
    pub fn promotion_root(mut self, path: Option<PathBuf>) -> Self {
        self.promotion_root = path;
        self
    }

    /// Set WAL path.
    pub fn wal_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.wal_path = path.into();
        self
    }

    /// Set require_promotion flag.
    pub fn require_promotion(mut self, require: bool) -> Self {
        self.require_promotion = require;
        self
    }
}

// =============================================================================
// AdmissionVerdict — Result of evaluation
// =============================================================================

/// Result of admission evaluation.
#[derive(Debug, Clone)]
pub struct AdmissionVerdict {
    /// The decision (includes digest for audit)
    pub decision: StrategyAdmissionDecision,

    /// Strategy spec if admitted (for downstream use)
    pub strategy_spec: Option<StrategySpec>,

    /// Promotion status (if checked)
    pub promotion_status: Option<PromotionStatus>,
}

impl AdmissionVerdict {
    /// Check if this verdict allows execution.
    pub fn is_admitted(&self) -> bool {
        self.decision.is_admitted()
    }

    /// Check if this verdict refuses execution.
    pub fn is_refused(&self) -> bool {
        self.decision.is_refused()
    }
}

// =============================================================================
// StrategyAdmissionEngineError
// =============================================================================

/// Errors from the strategy admission engine.
#[derive(Debug, thiserror::Error)]
pub enum StrategyAdmissionEngineError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Strategies manifest error: {0}")]
    StrategiesManifest(String),

    #[error("Signals manifest error: {0}")]
    SignalsManifest(String),

    #[error("WAL write failed: {0}")]
    WalWrite(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

// =============================================================================
// StrategyAdmissionEngine
// =============================================================================

/// Runtime engine for strategy admission control.
///
/// For each (strategy_id, signal_id, correlation_id) candidate:
/// 1. Validates signal exists in signals_manifest
/// 2. Validates strategy exists in strategies_manifest
/// 3. Validates signal is bound to strategy
/// 4. Optionally validates signal is promoted
/// 5. Writes WAL event with canonical digest
/// 6. Returns Admit/Refuse verdict
///
/// ## Example
/// ```ignore
/// let config = StrategyAdmissionEngineConfig::new("session_001")
///     .strategies_manifest("config/strategies_manifest.json")
///     .signals_manifest("config/signals_manifest.json")
///     .wal_path("wal/strategy_admission.jsonl");
///
/// let mut engine = StrategyAdmissionEngine::new(config)?;
///
/// let verdict = engine.evaluate("spread_passive", "spread", "corr_123", timestamp_ns)?;
/// if verdict.is_admitted() {
///     // Strategy can proceed with this signal
/// }
/// ```
pub struct StrategyAdmissionEngine {
    /// Cached strategies manifest
    strategies_manifest: StrategiesManifest,
    /// SHA-256 hash of strategies manifest
    strategies_manifest_hash: [u8; 32],

    /// Cached signals manifest
    signals_manifest: SignalsManifest,
    /// SHA-256 hash of signals manifest
    signals_manifest_hash: [u8; 32],

    /// Promotion resolver (may be disabled)
    promotion_resolver: PromotionResolver,

    /// WAL writer
    wal_writer: BufWriter<File>,

    /// Session ID
    session_id: String,

    /// Whether to require promotion
    require_promotion: bool,

    /// Config paths (for reload)
    strategies_manifest_path: PathBuf,
    signals_manifest_path: PathBuf,
}

impl StrategyAdmissionEngine {
    /// Create engine from config.
    ///
    /// Loads and validates manifests, opens WAL file.
    pub fn new(
        config: StrategyAdmissionEngineConfig,
    ) -> Result<Self, StrategyAdmissionEngineError> {
        // Load strategies manifest
        let strategies_manifest =
            StrategiesManifest::load_validated(&config.strategies_manifest_path)
                .map_err(|e| StrategyAdmissionEngineError::StrategiesManifest(e.to_string()))?;
        let strategies_manifest_hash = strategies_manifest.compute_version_hash();

        // Load signals manifest
        let signals_manifest = SignalsManifest::load_validated(&config.signals_manifest_path)
            .map_err(|e| StrategyAdmissionEngineError::SignalsManifest(e.to_string()))?;
        let signals_manifest_hash = signals_manifest.compute_version_hash();

        // Validate signal bindings
        strategies_manifest
            .validate_signal_bindings(&signals_manifest)
            .map_err(|e| StrategyAdmissionEngineError::StrategiesManifest(e.to_string()))?;

        // Create promotion resolver
        let promotion_resolver = match &config.promotion_root {
            Some(root) => PromotionResolver::new(root)?,
            None => PromotionResolver::disabled(),
        };

        // Warn if promotion checking is disabled
        if !promotion_resolver.is_enabled() && config.require_promotion {
            return Err(StrategyAdmissionEngineError::Config(
                "require_promotion=true but no promotion_root provided".to_string(),
            ));
        }

        if !promotion_resolver.is_enabled() {
            tracing::warn!(
                "⚠️  PROMOTION CHECKING DISABLED ⚠️\n\
                 Signals are NOT being verified against promotion artifacts.\n\
                 This is acceptable for development but NOT for production.\n\
                 Enable with promotion_root configuration."
            );
        }

        // Create WAL directory if needed
        if let Some(parent) = config.wal_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open WAL file (append mode)
        let wal_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.wal_path)?;
        let wal_writer = BufWriter::new(wal_file);

        Ok(Self {
            strategies_manifest,
            strategies_manifest_hash,
            signals_manifest,
            signals_manifest_hash,
            promotion_resolver,
            wal_writer,
            session_id: config.session_id,
            require_promotion: config.require_promotion,
            strategies_manifest_path: config.strategies_manifest_path,
            signals_manifest_path: config.signals_manifest_path,
        })
    }

    /// Evaluate admission for a (strategy_id, signal_id) pair.
    ///
    /// This is the main entry point. It:
    /// 1. Checks all admission criteria
    /// 2. Writes WAL event (before returning)
    /// 3. Returns verdict
    ///
    /// The WAL write is synchronous and happens BEFORE the verdict is returned.
    /// If WAL write fails, returns error (fail-closed).
    pub fn evaluate(
        &mut self,
        strategy_id: &str,
        signal_id: &str,
        correlation_id: &str,
        ts_ns: i64,
    ) -> Result<AdmissionVerdict, StrategyAdmissionEngineError> {
        // Evaluate admission criteria
        let (outcome, reasons, strategy_spec, promotion_status) =
            self.evaluate_admission(strategy_id, signal_id);

        // Build decision
        let decision = StrategyAdmissionDecision::builder(strategy_id, signal_id)
            .ts_ns(ts_ns)
            .session_id(&self.session_id)
            .correlation_id(correlation_id)
            .strategies_manifest_hash(self.strategies_manifest_hash)
            .signals_manifest_hash(self.signals_manifest_hash);

        let decision = if outcome == StrategyAdmissionOutcome::Admit {
            decision.build_admit()
        } else {
            decision.build_refuse(reasons)
        };

        // Write to WAL (fail-closed: error if write fails)
        self.write_wal_entry(&decision)?;

        Ok(AdmissionVerdict {
            decision,
            strategy_spec,
            promotion_status,
        })
    }

    /// Batch evaluate multiple candidates.
    ///
    /// Each candidate gets its own WAL entry and verdict.
    pub fn evaluate_batch(
        &mut self,
        candidates: &[(String, String, String, i64)], // (strategy_id, signal_id, correlation_id, ts_ns)
    ) -> Result<Vec<AdmissionVerdict>, StrategyAdmissionEngineError> {
        let mut verdicts = Vec::with_capacity(candidates.len());

        for (strategy_id, signal_id, correlation_id, ts_ns) in candidates {
            let verdict = self.evaluate(strategy_id, signal_id, correlation_id, *ts_ns)?;
            verdicts.push(verdict);
        }

        // Flush after batch
        self.flush_wal()?;

        Ok(verdicts)
    }

    /// Flush WAL to disk.
    pub fn flush_wal(&mut self) -> std::io::Result<()> {
        self.wal_writer.flush()
    }

    /// Get strategies manifest hash (for audit).
    pub fn strategies_manifest_hash(&self) -> [u8; 32] {
        self.strategies_manifest_hash
    }

    /// Get signals manifest hash (for audit).
    pub fn signals_manifest_hash(&self) -> [u8; 32] {
        self.signals_manifest_hash
    }

    /// Get the strategies manifest.
    pub fn strategies_manifest(&self) -> &StrategiesManifest {
        &self.strategies_manifest
    }

    /// Get the signals manifest.
    pub fn signals_manifest(&self) -> &SignalsManifest {
        &self.signals_manifest
    }

    /// Reload manifests from disk (e.g., on SIGHUP).
    pub fn reload_manifests(&mut self) -> Result<(), StrategyAdmissionEngineError> {
        // Load strategies manifest
        let strategies_manifest =
            StrategiesManifest::load_validated(&self.strategies_manifest_path)
                .map_err(|e| StrategyAdmissionEngineError::StrategiesManifest(e.to_string()))?;

        // Load signals manifest
        let signals_manifest = SignalsManifest::load_validated(&self.signals_manifest_path)
            .map_err(|e| StrategyAdmissionEngineError::SignalsManifest(e.to_string()))?;

        // Validate signal bindings
        strategies_manifest
            .validate_signal_bindings(&signals_manifest)
            .map_err(|e| StrategyAdmissionEngineError::StrategiesManifest(e.to_string()))?;

        // Update cached values
        self.strategies_manifest_hash = strategies_manifest.compute_version_hash();
        self.signals_manifest_hash = signals_manifest.compute_version_hash();
        self.strategies_manifest = strategies_manifest;
        self.signals_manifest = signals_manifest;

        // Clear promotion cache
        self.promotion_resolver.clear_cache();

        tracing::info!("Manifests reloaded successfully");

        Ok(())
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Evaluate admission criteria.
    ///
    /// Returns (outcome, reasons, strategy_spec, promotion_status).
    fn evaluate_admission(
        &mut self,
        strategy_id: &str,
        signal_id: &str,
    ) -> (
        StrategyAdmissionOutcome,
        Vec<StrategyRefuseReason>,
        Option<StrategySpec>,
        Option<PromotionStatus>,
    ) {
        let mut reasons = Vec::new();
        let mut promotion_status = None;

        // 1. Check signal exists in signals_manifest
        if self.signals_manifest.get_signal(signal_id).is_none() {
            reasons.push(StrategyRefuseReason::SignalNotAdmitted {
                signal_id: signal_id.to_string(),
            });
        }

        // 2. Check strategy exists
        let strategy_spec = match self.strategies_manifest.get_strategy(strategy_id) {
            Some(spec) => Some(spec.clone()),
            None => {
                reasons.push(StrategyRefuseReason::StrategyNotFound {
                    strategy_id: strategy_id.to_string(),
                });
                None
            }
        };

        // 3. Check signal is bound to strategy (if strategy exists)
        if let Some(ref spec) = strategy_spec
            && !spec.signals.contains(&signal_id.to_string())
        {
            reasons.push(StrategyRefuseReason::SignalNotBound {
                signal_id: signal_id.to_string(),
                strategy_id: strategy_id.to_string(),
            });
        }

        // 4. Check promotion (if enabled and required)
        if self.promotion_resolver.is_enabled() && self.require_promotion {
            let status = self.promotion_resolver.is_promoted(signal_id);
            if !status.promoted {
                reasons.push(StrategyRefuseReason::SignalNotAdmitted {
                    signal_id: signal_id.to_string(),
                });
            }
            promotion_status = Some(status);
        }

        // Determine outcome
        let outcome = if reasons.is_empty() {
            StrategyAdmissionOutcome::Admit
        } else {
            StrategyAdmissionOutcome::Refuse
        };

        (outcome, reasons, strategy_spec, promotion_status)
    }

    /// Write a decision to the WAL.
    fn write_wal_entry(
        &mut self,
        decision: &StrategyAdmissionDecision,
    ) -> Result<(), StrategyAdmissionEngineError> {
        let json = serde_json::to_string(decision)?;
        writeln!(self.wal_writer, "{}", json)
            .map_err(|e| StrategyAdmissionEngineError::WalWrite(e.to_string()))?;
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_manifests(temp_dir: &TempDir) -> (PathBuf, PathBuf) {
        let config_dir = temp_dir.path().join("config");
        fs::create_dir_all(&config_dir).unwrap();

        // Create signals manifest
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
        let signals_path = config_dir.join("signals_manifest.json");
        fs::write(&signals_path, signals_json).unwrap();

        // Create strategies manifest
        let strategies_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "spread_passive": {
                    "strategy_id": "spread_passive",
                    "signals": ["spread"],
                    "execution_class": "passive",
                    "max_orders_per_min": 120,
                    "max_position_abs": 10000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false,
                    "tags": []
                },
                "microprice_aggressive": {
                    "strategy_id": "microprice_aggressive",
                    "signals": ["microprice", "spread"],
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
        let strategies_path = config_dir.join("strategies_manifest.json");
        fs::write(&strategies_path, strategies_json).unwrap();

        (signals_path, strategies_path)
    }

    fn create_engine(temp_dir: &TempDir) -> StrategyAdmissionEngine {
        let (signals_path, strategies_path) = create_test_manifests(temp_dir);
        let wal_path = temp_dir.path().join("wal/strategy_admission.jsonl");

        let config = StrategyAdmissionEngineConfig::new("test_session")
            .signals_manifest(signals_path)
            .strategies_manifest(strategies_path)
            .wal_path(wal_path)
            .require_promotion(false);

        StrategyAdmissionEngine::new(config).unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let temp_dir = TempDir::new().unwrap();
        let engine = create_engine(&temp_dir);

        assert_eq!(engine.strategies_manifest().strategy_count(), 2);
        assert_eq!(engine.signals_manifest().signal_count(), 2);
    }

    #[test]
    fn test_admit_valid_binding() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        let verdict = engine
            .evaluate("spread_passive", "spread", "corr_001", 1000000000)
            .unwrap();

        assert!(verdict.is_admitted());
        assert!(verdict.strategy_spec.is_some());
        assert_eq!(verdict.strategy_spec.unwrap().strategy_id, "spread_passive");
    }

    #[test]
    fn test_refuse_unknown_strategy() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        let verdict = engine
            .evaluate("nonexistent_strategy", "spread", "corr_001", 1000000000)
            .unwrap();

        assert!(verdict.is_refused());
        assert!(verdict.strategy_spec.is_none());
        assert_eq!(verdict.decision.refuse_reasons.len(), 1);
        match &verdict.decision.refuse_reasons[0] {
            StrategyRefuseReason::StrategyNotFound { strategy_id } => {
                assert_eq!(strategy_id, "nonexistent_strategy");
            }
            _ => panic!("Expected StrategyNotFound"),
        }
    }

    #[test]
    fn test_refuse_unknown_signal() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        let verdict = engine
            .evaluate(
                "spread_passive",
                "nonexistent_signal",
                "corr_001",
                1000000000,
            )
            .unwrap();

        assert!(verdict.is_refused());
        // Should have multiple reasons: signal not in manifest AND not bound to strategy
        assert!(verdict.decision.refuse_reasons.len() >= 1);
    }

    #[test]
    fn test_refuse_signal_not_bound() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        // microprice signal exists but is not bound to spread_passive
        let verdict = engine
            .evaluate("spread_passive", "microprice", "corr_001", 1000000000)
            .unwrap();

        assert!(verdict.is_refused());
        let has_not_bound = verdict
            .decision
            .refuse_reasons
            .iter()
            .any(|r| matches!(r, StrategyRefuseReason::SignalNotBound { .. }));
        assert!(has_not_bound);
    }

    #[test]
    fn test_wal_written() {
        let temp_dir = TempDir::new().unwrap();
        let wal_path = temp_dir.path().join("wal/strategy_admission.jsonl");

        let mut engine = create_engine(&temp_dir);

        // Make some evaluations
        engine
            .evaluate("spread_passive", "spread", "corr_001", 1000000000)
            .unwrap();
        engine
            .evaluate("nonexistent", "spread", "corr_002", 1000000001)
            .unwrap();

        engine.flush_wal().unwrap();

        // Read WAL and verify
        let content = fs::read_to_string(&wal_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines.len(), 2);

        // Parse and verify first entry
        let entry1: StrategyAdmissionDecision = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(entry1.strategy_id, "spread_passive");
        assert_eq!(entry1.signal_id, "spread");
        assert_eq!(entry1.outcome, StrategyAdmissionOutcome::Admit);

        // Parse and verify second entry
        let entry2: StrategyAdmissionDecision = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(entry2.strategy_id, "nonexistent");
        assert_eq!(entry2.outcome, StrategyAdmissionOutcome::Refuse);
    }

    #[test]
    fn test_digest_determinism() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        // Evaluate the same thing 100 times, digests should be identical
        let mut digests = Vec::new();
        for i in 0..100 {
            let verdict = engine
                .evaluate("spread_passive", "spread", "corr_test", 1000000000)
                .unwrap();
            digests.push(verdict.decision.digest.clone());

            // All digests should match the first one
            if i > 0 {
                assert_eq!(digests[i], digests[0], "Digest mismatch at iteration {}", i);
            }
        }
    }

    #[test]
    fn test_batch_evaluate() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        let candidates = vec![
            (
                "spread_passive".to_string(),
                "spread".to_string(),
                "corr_001".to_string(),
                1000000000i64,
            ),
            (
                "microprice_aggressive".to_string(),
                "microprice".to_string(),
                "corr_002".to_string(),
                1000000001i64,
            ),
            (
                "nonexistent".to_string(),
                "spread".to_string(),
                "corr_003".to_string(),
                1000000002i64,
            ),
        ];

        let verdicts = engine.evaluate_batch(&candidates).unwrap();

        assert_eq!(verdicts.len(), 3);
        assert!(verdicts[0].is_admitted());
        assert!(verdicts[1].is_admitted());
        assert!(verdicts[2].is_refused());
    }

    #[test]
    fn test_manifest_hashes_stable() {
        let temp_dir = TempDir::new().unwrap();
        let engine = create_engine(&temp_dir);

        let hash1 = engine.strategies_manifest_hash();
        let hash2 = engine.strategies_manifest_hash();
        assert_eq!(hash1, hash2);

        let hash3 = engine.signals_manifest_hash();
        let hash4 = engine.signals_manifest_hash();
        assert_eq!(hash3, hash4);
    }

    #[test]
    fn test_config_builder() {
        let config = StrategyAdmissionEngineConfig::new("session_001")
            .strategies_manifest("/path/to/strategies.json")
            .signals_manifest("/path/to/signals.json")
            .promotion_root(Some(PathBuf::from("/path/to/promotions")))
            .wal_path("/path/to/wal.jsonl")
            .require_promotion(true);

        assert_eq!(
            config.strategies_manifest_path,
            PathBuf::from("/path/to/strategies.json")
        );
        assert_eq!(
            config.signals_manifest_path,
            PathBuf::from("/path/to/signals.json")
        );
        assert_eq!(
            config.promotion_root,
            Some(PathBuf::from("/path/to/promotions"))
        );
        assert_eq!(config.wal_path, PathBuf::from("/path/to/wal.jsonl"));
        assert_eq!(config.session_id, "session_001");
        assert!(config.require_promotion);
    }

    #[test]
    fn test_require_promotion_without_root_fails() {
        let temp_dir = TempDir::new().unwrap();
        let (signals_path, strategies_path) = create_test_manifests(&temp_dir);
        let wal_path = temp_dir.path().join("wal/strategy_admission.jsonl");

        let config = StrategyAdmissionEngineConfig::new("test_session")
            .signals_manifest(signals_path)
            .strategies_manifest(strategies_path)
            .wal_path(wal_path)
            .require_promotion(true); // require but no root

        let result = StrategyAdmissionEngine::new(config);
        assert!(result.is_err());
        match result {
            Err(StrategyAdmissionEngineError::Config(msg)) => {
                assert!(msg.contains("require_promotion=true"));
            }
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_reload_manifests() {
        let temp_dir = TempDir::new().unwrap();
        let mut engine = create_engine(&temp_dir);

        let hash_before = engine.strategies_manifest_hash();

        // Reload should work even if files unchanged
        engine.reload_manifests().unwrap();

        let hash_after = engine.strategies_manifest_hash();
        assert_eq!(hash_before, hash_after);
    }
}
