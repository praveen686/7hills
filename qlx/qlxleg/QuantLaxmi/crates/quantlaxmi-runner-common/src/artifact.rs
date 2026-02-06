//! # Artifact Manifest System
//!
//! Standardized artifact structure for audit-grade reproducibility.
//!
//! ## Directory Structure
//! ```text
//! artifacts/<family>/<run_id>/
//!   manifest.json           # Run metadata + hashes
//!   inputs/
//!     universe.json         # Symbol universe spec
//!     quotes.jsonl          # Market data (or depth events)
//!     orders.json           # Order intents
//!   outputs/
//!     fills.jsonl           # Execution fills
//!     report.json           # Performance report
//!     diagnostics/
//!       strategy_gates.json # Gate decisions log
//!   vectorbt/
//!     market.csv            # OHLCV or quote data
//!     fills.csv             # Trade fills for VectorBT
//!     summary.json          # Metrics summary
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Artifact family (venue type)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactFamily {
    India,
    Crypto,
}

impl ArtifactFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            ArtifactFamily::India => "india",
            ArtifactFamily::Crypto => "crypto",
        }
    }
}

/// Run profile for categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunProfile {
    Smoke,     // Quick validation
    Research,  // Full research run
    Alpha,     // Live/paper trading
    Certified, // Deterministic certified run (Crypto)
}

/// Hash of a file for reproducibility tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHash {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

impl FileHash {
    /// Compute SHA256 hash of a file
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open file for hashing: {:?}", path))?;

        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        let mut size = 0u64;

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
            size += bytes_read as u64;
        }

        let hash = hasher.finalize();
        Ok(Self {
            path: path.to_string_lossy().to_string(),
            sha256: hex::encode(hash),
            size_bytes: size,
        })
    }

    /// Compute hash from bytes
    pub fn from_bytes(path: &str, data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        Self {
            path: path.to_string(),
            sha256: hex::encode(hash),
            size_bytes: data.len() as u64,
        }
    }
}

/// Build information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub version: String,
    pub git_commit: Option<String>,
    pub build_timestamp: DateTime<Utc>,
    pub rust_version: String,
    pub cargo_lock_hash: Option<String>,
}

impl BuildInfo {
    pub fn current() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("GIT_COMMIT").map(|s| s.to_string()),
            build_timestamp: Utc::now(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            cargo_lock_hash: None, // Populated separately if needed
        }
    }
}

/// Execution environment fingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEnv {
    pub hostname: String,
    pub os: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub duration_secs: Option<f64>,
}

impl ExecutionEnv {
    pub fn capture() -> Self {
        Self {
            hostname: hostname::get()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            os: std::env::consts::OS.to_string(),
            started_at: Utc::now(),
            finished_at: None,
            duration_secs: None,
        }
    }

    pub fn finish(&mut self) {
        let now = Utc::now();
        self.finished_at = Some(now);
        self.duration_secs = Some((now - self.started_at).num_milliseconds() as f64 / 1000.0);
    }
}

/// Input data sources with hashes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputSources {
    pub universe: Option<FileHash>,
    pub quotes: Option<FileHash>,
    pub depth_events: Option<FileHash>,
    pub trade_events: Option<FileHash>,
    pub tick_events: Option<FileHash>,
    pub orders: Option<FileHash>,
    pub config: Option<FileHash>,
}

/// Output artifacts with hashes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputArtifacts {
    pub fills: Option<FileHash>,
    pub report: Option<FileHash>,
    pub strategy_gates: Option<FileHash>,
    pub vectorbt_market: Option<FileHash>,
    pub vectorbt_fills: Option<FileHash>,
    pub vectorbt_summary: Option<FileHash>,
}

/// Strategy gate decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecision {
    pub timestamp: DateTime<Utc>,
    pub strategy: String,
    pub gate_type: String,
    pub previous_state: String,
    pub new_state: String,
    pub reason: String,
    pub confidence: Option<f64>,
    pub metrics: Option<serde_json::Value>,
}

/// Strategy diagnostics for audit
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyDiagnostics {
    pub gate_decisions: Vec<GateDecision>,
    pub regime_transitions: Vec<RegimeTransition>,
    pub context_validation: ContextValidation,
}

/// Regime transition record (HYDRA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransition {
    pub timestamp: DateTime<Utc>,
    pub previous_regime: String,
    pub new_regime: String,
    pub confidence: f64,
    pub features: serde_json::Value,
}

/// Context validation for India (underlying reference required)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextValidation {
    pub underlying_present: bool,
    pub underlying_symbols: Vec<String>,
    pub options_symbols: Vec<String>,
    pub history_window_bars: u32,
    pub min_history_required: u32,
    pub validation_passed: bool,
    pub validation_errors: Vec<String>,
}

/// Complete run manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub family: ArtifactFamily,
    pub profile: RunProfile,
    pub watermark: String,
    pub build: BuildInfo,
    pub environment: ExecutionEnv,
    pub inputs: InputSources,
    pub outputs: OutputArtifacts,
    pub diagnostics: StrategyDiagnostics,
    pub determinism: DeterminismInfo,
}

/// Determinism tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeterminismInfo {
    /// Whether this run claims determinism
    pub certified: bool,
    /// Combined hash of all inputs for reproducibility
    pub input_hash: Option<String>,
    /// Combined hash of all outputs for verification
    pub output_hash: Option<String>,
    /// Previous run ID this was compared against (if any)
    pub compared_to: Option<String>,
    /// Whether outputs matched the comparison run
    pub matches_previous: Option<bool>,
}

impl RunManifest {
    /// Create a new manifest for a run
    pub fn new(family: ArtifactFamily, profile: RunProfile) -> Self {
        let run_id = Uuid::new_v4().to_string();
        let watermark = format!(
            "QuantLaxmi-{}-{}-{}",
            family.as_str(),
            chrono::Utc::now().format("%Y%m%d-%H%M%S"),
            &run_id[..8]
        );

        Self {
            run_id,
            family,
            profile,
            watermark,
            build: BuildInfo::current(),
            environment: ExecutionEnv::capture(),
            inputs: InputSources::default(),
            outputs: OutputArtifacts::default(),
            diagnostics: StrategyDiagnostics::default(),
            determinism: DeterminismInfo::default(),
        }
    }

    /// Compute combined input hash
    pub fn compute_input_hash(&mut self) {
        let mut hasher = Sha256::new();

        if let Some(ref h) = self.inputs.universe {
            hasher.update(h.sha256.as_bytes());
        }
        if let Some(ref h) = self.inputs.quotes {
            hasher.update(h.sha256.as_bytes());
        }
        if let Some(ref h) = self.inputs.depth_events {
            hasher.update(h.sha256.as_bytes());
        }
        if let Some(ref h) = self.inputs.orders {
            hasher.update(h.sha256.as_bytes());
        }
        if let Some(ref h) = self.inputs.config {
            hasher.update(h.sha256.as_bytes());
        }

        self.determinism.input_hash = Some(hex::encode(hasher.finalize()));
    }

    /// Compute combined output hash
    pub fn compute_output_hash(&mut self) {
        let mut hasher = Sha256::new();

        if let Some(ref h) = self.outputs.fills {
            hasher.update(h.sha256.as_bytes());
        }
        if let Some(ref h) = self.outputs.report {
            hasher.update(h.sha256.as_bytes());
        }
        if let Some(ref h) = self.outputs.strategy_gates {
            hasher.update(h.sha256.as_bytes());
        }

        self.determinism.output_hash = Some(hex::encode(hasher.finalize()));
    }

    /// Mark run as finished
    pub fn finish(&mut self) {
        self.environment.finish();
        self.compute_input_hash();
        self.compute_output_hash();
    }

    /// Write manifest to directory
    pub fn write(&self, dir: &Path) -> Result<PathBuf> {
        let manifest_path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&manifest_path, json)?;
        Ok(manifest_path)
    }

    /// Load manifest from file
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let manifest: Self = serde_json::from_str(&content)?;
        Ok(manifest)
    }
}

/// Artifact directory builder
pub struct ArtifactBuilder {
    base_dir: PathBuf,
    manifest: RunManifest,
}

impl ArtifactBuilder {
    /// Create artifact structure for a new run
    pub fn new(base_dir: &Path, family: ArtifactFamily, profile: RunProfile) -> Result<Self> {
        let manifest = RunManifest::new(family, profile);
        let run_dir = base_dir
            .join("artifacts")
            .join(family.as_str())
            .join(&manifest.run_id);

        // Create directory structure
        fs::create_dir_all(run_dir.join("inputs"))?;
        fs::create_dir_all(run_dir.join("outputs/diagnostics"))?;
        fs::create_dir_all(run_dir.join("vectorbt"))?;

        Ok(Self {
            base_dir: run_dir,
            manifest,
        })
    }

    /// Get the run directory
    pub fn run_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Get the run ID
    pub fn run_id(&self) -> &str {
        &self.manifest.run_id
    }

    /// Register input universe file
    pub fn register_universe(&mut self, path: &Path) -> Result<PathBuf> {
        let dest = self.base_dir.join("inputs/universe.json");
        fs::copy(path, &dest)?;
        self.manifest.inputs.universe = Some(FileHash::from_file(&dest)?);
        Ok(dest)
    }

    /// Register input quotes file
    pub fn register_quotes(&mut self, path: &Path) -> Result<PathBuf> {
        let dest = self.base_dir.join("inputs/quotes.jsonl");
        fs::copy(path, &dest)?;
        self.manifest.inputs.quotes = Some(FileHash::from_file(&dest)?);
        Ok(dest)
    }

    /// Register input depth events file
    pub fn register_depth_events(&mut self, path: &Path) -> Result<PathBuf> {
        let dest = self.base_dir.join("inputs/depth_events.jsonl");
        fs::copy(path, &dest)?;
        self.manifest.inputs.depth_events = Some(FileHash::from_file(&dest)?);
        Ok(dest)
    }

    /// Register input orders file
    pub fn register_orders(&mut self, path: &Path) -> Result<PathBuf> {
        let dest = self.base_dir.join("inputs/orders.json");
        fs::copy(path, &dest)?;
        self.manifest.inputs.orders = Some(FileHash::from_file(&dest)?);
        Ok(dest)
    }

    /// Register config file
    pub fn register_config(&mut self, path: &Path) -> Result<PathBuf> {
        let dest = self.base_dir.join("inputs/config.toml");
        fs::copy(path, &dest)?;
        self.manifest.inputs.config = Some(FileHash::from_file(&dest)?);
        Ok(dest)
    }

    /// Write fills to output
    pub fn write_fills<T: Serialize>(&mut self, fills: &[T]) -> Result<PathBuf> {
        let path = self.base_dir.join("outputs/fills.jsonl");
        let mut file = File::create(&path)?;
        for fill in fills {
            let line = serde_json::to_string(fill)?;
            writeln!(file, "{}", line)?;
        }
        self.manifest.outputs.fills = Some(FileHash::from_file(&path)?);
        Ok(path)
    }

    /// Write report to output
    pub fn write_report<T: Serialize>(&mut self, report: &T) -> Result<PathBuf> {
        let path = self.base_dir.join("outputs/report.json");
        let json = serde_json::to_string_pretty(report)?;
        fs::write(&path, &json)?;
        self.manifest.outputs.report = Some(FileHash::from_file(&path)?);
        Ok(path)
    }

    /// Write strategy gates diagnostics
    pub fn write_strategy_gates(&mut self) -> Result<PathBuf> {
        let path = self
            .base_dir
            .join("outputs/diagnostics/strategy_gates.json");
        let json = serde_json::to_string_pretty(&self.manifest.diagnostics)?;
        fs::write(&path, &json)?;
        self.manifest.outputs.strategy_gates = Some(FileHash::from_file(&path)?);
        Ok(path)
    }

    /// Add a gate decision to diagnostics
    pub fn log_gate_decision(&mut self, decision: GateDecision) {
        self.manifest.diagnostics.gate_decisions.push(decision);
    }

    /// Add a regime transition to diagnostics
    pub fn log_regime_transition(&mut self, transition: RegimeTransition) {
        self.manifest
            .diagnostics
            .regime_transitions
            .push(transition);
    }

    /// Set context validation results
    pub fn set_context_validation(&mut self, validation: ContextValidation) {
        self.manifest.diagnostics.context_validation = validation;
    }

    /// Mark as certified deterministic run
    pub fn mark_certified(&mut self) {
        self.manifest.determinism.certified = true;
    }

    /// Get mutable reference to manifest for direct updates
    pub fn manifest_mut(&mut self) -> &mut RunManifest {
        &mut self.manifest
    }

    /// Get reference to manifest
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }

    /// Finalize and write manifest
    pub fn finalize(mut self) -> Result<RunManifest> {
        self.manifest.finish();
        self.manifest.write(&self.base_dir)?;
        Ok(self.manifest)
    }
}

/// Context validator for India pipeline
pub struct IndiaContextValidator {
    underlying_symbols: Vec<String>,
    options_symbols: Vec<String>,
    min_history_bars: u32,
}

impl IndiaContextValidator {
    pub fn new(min_history_bars: u32) -> Self {
        Self {
            underlying_symbols: Vec::new(),
            options_symbols: Vec::new(),
            min_history_bars,
        }
    }

    /// Add symbol to the universe
    pub fn add_symbol(&mut self, symbol: &str) {
        let upper = symbol.to_uppercase();

        // Check if it's an underlying (FUT or spot index)
        let is_underlying = upper.ends_with("FUT")
            || upper == "NIFTY"
            || upper == "BANKNIFTY"
            || upper == "NIFTY50"
            || upper == "BANKNIFTY-I"  // Spot proxy
            || upper.contains("FUT");

        if is_underlying && !self.underlying_symbols.contains(&upper) {
            self.underlying_symbols.push(upper);
        } else if (upper.contains("CE") || upper.contains("PE"))
            && !self.options_symbols.contains(&upper)
        {
            // Options have CE/PE suffix
            self.options_symbols.push(upper);
        }
    }

    /// Validate context requirements
    pub fn validate(&self, actual_history_bars: u32) -> ContextValidation {
        let mut errors = Vec::new();

        // Check for underlying presence
        let underlying_present = !self.underlying_symbols.is_empty();
        if !underlying_present && !self.options_symbols.is_empty() {
            errors.push(
                "FATAL: Options universe has no underlying reference (FUT or spot). \
                 HYDRA/AEON require underlying context for regime detection."
                    .to_string(),
            );
        }

        // Check history window
        if actual_history_bars < self.min_history_bars {
            errors.push(format!(
                "Insufficient history: {} bars available, {} required for regime warmup",
                actual_history_bars, self.min_history_bars
            ));
        }

        ContextValidation {
            underlying_present,
            underlying_symbols: self.underlying_symbols.clone(),
            options_symbols: self.options_symbols.clone(),
            history_window_bars: actual_history_bars,
            min_history_required: self.min_history_bars,
            validation_passed: errors.is_empty(),
            validation_errors: errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_file_hash() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("test.txt");
        fs::write(&path, "hello world").unwrap();

        let hash = FileHash::from_file(&path).unwrap();
        assert_eq!(hash.size_bytes, 11);
        assert!(!hash.sha256.is_empty());
    }

    #[test]
    fn test_manifest_creation() {
        let manifest = RunManifest::new(ArtifactFamily::India, RunProfile::Smoke);
        assert!(!manifest.run_id.is_empty());
        assert!(manifest.watermark.contains("india"));
    }

    #[test]
    fn test_artifact_builder() {
        let temp = TempDir::new().unwrap();
        let builder =
            ArtifactBuilder::new(temp.path(), ArtifactFamily::Crypto, RunProfile::Research)
                .unwrap();

        assert!(builder.run_dir().join("inputs").exists());
        assert!(builder.run_dir().join("outputs/diagnostics").exists());
        assert!(builder.run_dir().join("vectorbt").exists());
    }

    #[test]
    fn test_india_context_validator() {
        let mut validator = IndiaContextValidator::new(100);

        // Add only options
        validator.add_symbol("NIFTY25JAN25500CE");
        validator.add_symbol("NIFTY25JAN25500PE");

        let result = validator.validate(50);
        assert!(!result.validation_passed);
        assert!(!result.underlying_present);

        // Add underlying
        validator.add_symbol("NIFTYFUT");
        let result = validator.validate(150);
        assert!(result.validation_passed);
        assert!(result.underlying_present);
    }
}
