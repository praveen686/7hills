//! # Experiment Registry - Run Manifest and Reproducibility
//!
//! Every research run produces a `run_manifest.json` that captures:
//! - Input: segment IDs + per-stream digests
//! - Code: binary hash + git tag
//! - Params: configuration used
//! - Output digests: features, labels, scores
//!
//! This enables:
//! - Full reproducibility (same inputs + code = identical outputs)
//! - Audit trail (which data + code produced which results)
//! - Experiment comparison (A/B on params with same data)

use crate::features::{FEATURE_SCHEMA_VERSION, FeatureConfig, FeatureExtractionResult};
use crate::segment_manifest::{SegmentDigests, SegmentManifest, compute_binary_hash};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Run manifest schema version
pub const RUN_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Input segment reference with integrity proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInput {
    /// Segment ID (directory name)
    pub segment_id: String,
    /// Path to segment directory
    pub segment_path: String,
    /// Segment manifest state at time of use
    pub state: String,
    /// Per-stream digests (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digests: Option<SegmentDigests>,
    /// Symbols in segment
    pub symbols: Vec<String>,
    /// Segment start timestamp
    pub start_ts: String,
    /// Segment end timestamp (if finalized)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ts: Option<String>,
}

impl SegmentInput {
    /// Create from a segment manifest
    pub fn from_manifest(manifest: &SegmentManifest, path: &Path) -> Self {
        Self {
            segment_id: manifest.segment_id.clone(),
            segment_path: path.display().to_string(),
            state: format!("{:?}", manifest.state),
            digests: manifest.digests.clone(),
            symbols: manifest.symbols.clone(),
            start_ts: manifest.start_ts.to_rfc3339(),
            end_ts: manifest.end_ts.map(|t| t.to_rfc3339()),
        }
    }
}

/// Code provenance for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeProvenance {
    /// SHA256 of the binary that produced outputs
    pub binary_hash: String,
    /// Git tag or commit (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    /// Feature schema version
    pub feature_schema_version: u32,
    /// Run manifest schema version
    pub run_manifest_schema_version: u32,
}

impl CodeProvenance {
    pub fn current() -> Self {
        let binary_hash = compute_binary_hash().unwrap_or_else(|_| "UNKNOWN".to_string());
        let git_ref = get_git_ref();

        Self {
            binary_hash,
            git_ref,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            run_manifest_schema_version: RUN_MANIFEST_SCHEMA_VERSION,
        }
    }
}

/// Get current git ref (tag or commit)
fn get_git_ref() -> Option<String> {
    // Try to get tag first, then commit
    std::process::Command::new("git")
        .args(["describe", "--tags", "--exact-match"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout)
                            .ok()
                            .map(|s| s.trim().to_string())
                    } else {
                        None
                    }
                })
        })
}

/// Output artifact with digest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputArtifact {
    /// Artifact type (features, labels, scores)
    pub artifact_type: String,
    /// Path to output file
    pub path: String,
    /// SHA256 digest of output
    pub digest: String,
    /// Number of records/events
    pub record_count: usize,
    /// First timestamp in output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_ts: Option<String>,
    /// Last timestamp in output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_ts: Option<String>,
}

/// Run manifest - complete record of a research run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    /// Schema version
    pub schema_version: u32,
    /// Run ID (unique identifier)
    pub run_id: String,
    /// Run start timestamp
    pub started_at: DateTime<Utc>,
    /// Run end timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    /// Run directory
    pub run_dir: String,
    /// Input segments
    pub inputs: Vec<SegmentInput>,
    /// Code provenance
    pub code: CodeProvenance,
    /// Feature extraction configuration
    pub feature_config: FeatureConfig,
    /// Output artifacts
    pub outputs: Vec<OutputArtifact>,
    /// Combined input digest (for quick comparison)
    pub input_digest: String,
    /// Run status
    pub status: RunStatus,
    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
}

impl RunManifest {
    /// Create a new run manifest
    pub fn new(run_dir: &Path, inputs: Vec<SegmentInput>, feature_config: FeatureConfig) -> Self {
        let now = Utc::now();
        let run_id = format!(
            "exp_{}_{}",
            now.format("%Y%m%d_%H%M%S"),
            &Self::short_digest(&inputs)
        );

        // Compute combined input digest
        let input_digest = Self::compute_input_digest(&inputs);

        Self {
            schema_version: RUN_MANIFEST_SCHEMA_VERSION,
            run_id,
            started_at: now,
            ended_at: None,
            run_dir: run_dir.display().to_string(),
            inputs,
            code: CodeProvenance::current(),
            feature_config,
            outputs: Vec::new(),
            input_digest,
            status: RunStatus::Running,
            error: None,
        }
    }

    /// Compute short digest from inputs (first 8 chars)
    fn short_digest(inputs: &[SegmentInput]) -> String {
        let digest = Self::compute_input_digest(inputs);
        digest[..8].to_string()
    }

    /// Compute combined input digest
    fn compute_input_digest(inputs: &[SegmentInput]) -> String {
        let mut hasher = Sha256::new();
        for input in inputs {
            hasher.update(input.segment_id.as_bytes());
            if let Some(ref digests) = input.digests {
                if let Some(ref perp) = digests.perp {
                    hasher.update(perp.sha256.as_bytes());
                }
                if let Some(ref spot) = digests.spot {
                    hasher.update(spot.sha256.as_bytes());
                }
                if let Some(ref funding) = digests.funding {
                    hasher.update(funding.sha256.as_bytes());
                }
            }
        }
        hex::encode(hasher.finalize())
    }

    /// Add an output artifact
    pub fn add_output(&mut self, artifact: OutputArtifact) {
        self.outputs.push(artifact);
    }

    /// Add feature extraction result as output
    pub fn add_feature_result(&mut self, result: &FeatureExtractionResult) {
        self.outputs.push(OutputArtifact {
            artifact_type: "features".to_string(),
            path: result.output_path.clone(),
            digest: result.output_digest.clone(),
            record_count: result.output_events,
            first_ts: result.first_ts.clone(),
            last_ts: result.last_ts.clone(),
        });
    }

    /// Mark run as completed
    pub fn complete(&mut self) {
        self.ended_at = Some(Utc::now());
        self.status = RunStatus::Completed;
    }

    /// Mark run as failed
    pub fn fail(&mut self, error: &str) {
        self.ended_at = Some(Utc::now());
        self.status = RunStatus::Failed;
        self.error = Some(error.to_string());
    }

    /// Write manifest to disk
    pub fn write(&self, run_dir: &Path) -> anyhow::Result<()> {
        let manifest_path = run_dir.join("run_manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&manifest_path, json)?;
        Ok(())
    }

    /// Load manifest from disk
    pub fn load(run_dir: &Path) -> anyhow::Result<Self> {
        let manifest_path = run_dir.join("run_manifest.json");
        let content = std::fs::read_to_string(&manifest_path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

/// Generate run directory name from timestamp and input digest
pub fn generate_run_dir(base_dir: &Path, inputs: &[SegmentInput]) -> std::path::PathBuf {
    let now = Utc::now();
    let short_digest = &RunManifest::compute_input_digest(inputs)[..8];
    let dir_name = format!("exp_{}_{}", now.format("%Y%m%d_%H%M%S"), short_digest);
    base_dir.join(dir_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_run_manifest_creation() {
        let dir = tempdir().unwrap();
        let inputs = vec![SegmentInput {
            segment_id: "perp_20260125_100000".to_string(),
            segment_path: "/data/segments/perp_20260125_100000".to_string(),
            state: "Finalized".to_string(),
            digests: None,
            symbols: vec!["BTCUSDT".to_string()],
            start_ts: "2026-01-25T10:00:00Z".to_string(),
            end_ts: Some("2026-01-25T11:00:00Z".to_string()),
        }];

        let manifest = RunManifest::new(dir.path(), inputs, FeatureConfig::default());

        assert!(manifest.run_id.starts_with("exp_"));
        assert_eq!(manifest.status, RunStatus::Running);
        assert!(!manifest.input_digest.is_empty());
    }

    #[test]
    fn test_input_digest_deterministic() {
        let inputs = vec![SegmentInput {
            segment_id: "perp_20260125_100000".to_string(),
            segment_path: "/data/segments/perp_20260125_100000".to_string(),
            state: "Finalized".to_string(),
            digests: None,
            symbols: vec!["BTCUSDT".to_string()],
            start_ts: "2026-01-25T10:00:00Z".to_string(),
            end_ts: None,
        }];

        let digest1 = RunManifest::compute_input_digest(&inputs);
        let digest2 = RunManifest::compute_input_digest(&inputs);

        assert_eq!(digest1, digest2);
    }

    #[test]
    fn test_manifest_write_load_roundtrip() {
        let dir = tempdir().unwrap();
        let inputs = vec![SegmentInput {
            segment_id: "test_segment".to_string(),
            segment_path: "/test".to_string(),
            state: "Finalized".to_string(),
            digests: None,
            symbols: vec!["BTCUSDT".to_string()],
            start_ts: "2026-01-25T10:00:00Z".to_string(),
            end_ts: None,
        }];

        let mut manifest = RunManifest::new(dir.path(), inputs, FeatureConfig::default());
        manifest.complete();
        manifest.write(dir.path()).unwrap();

        let loaded = RunManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.run_id, manifest.run_id);
        assert_eq!(loaded.status, RunStatus::Completed);
    }
}
