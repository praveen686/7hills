//! Tournament Sharding v2 - Exactly-Once Semantics
//!
//! Enables distributed tournament execution across multiple machines/processes
//! with cryptographic proofs of completion and coverage.
//!
//! ## Design
//! - Deterministic shard assignment: task → shard is hash-based and stable
//! - Shard salt prevents accidental mixing across different tournament runs
//! - Completion proofs detect crashed/partial shards
//! - Global coverage proof ensures exactly-once execution
//!
//! ## Usage
//! ```bash
//! # Run 4 shards (can be parallel on different machines)
//! tournament ... --shard-index 0 --shard-count 4 --out-dir shard_0
//! tournament ... --shard-index 1 --shard-count 4 --out-dir shard_1
//! tournament ... --shard-index 2 --shard-count 4 --out-dir shard_2
//! tournament ... --shard-index 3 --shard-count 4 --out-dir shard_3
//!
//! # Merge shards (validates exactly-once before producing output)
//! tournament ... --merge-shards shard_0,shard_1,shard_2,shard_3 --out-dir merged
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{BufRead, Read, Write};
use std::path::Path;
use tracing::{info, warn};

use crate::tournament::{GridRunResult, build_grid_leaderboard, select_promotion_candidates};
use crate::two_pass;

// =============================================================================
// Constants
// =============================================================================

pub const SHARD_MANIFEST_SCHEMA: &str = "shard_manifest_v2";
pub const MERGED_MANIFEST_SCHEMA: &str = "merged_manifest_v2";

// =============================================================================
// Canonical Key and Hashing
// =============================================================================

/// Create canonical task key from segment and param_hash.
/// Format: segment\tparam_hash (TAB separator for unambiguous parsing)
#[inline]
pub fn task_key(segment: &str, param_hash: &str) -> String {
    format!("{}\t{}", segment, param_hash)
}

/// Parse canonical task key back to (segment, param_hash).
pub fn parse_task_key(key: &str) -> Option<(&str, &str)> {
    key.split_once('\t')
}

/// Hash a sorted list of task keys with domain separation.
/// This is the core primitive for all completion/coverage proofs.
///
/// Format:
/// ```text
/// qlx-task-keys-v1\n
/// count=N\n
/// key1\n
/// key2\n
/// ...
/// ```
///
/// Including count in preamble prevents "same concatenation" ambiguity.
pub fn hash_sorted_keys(keys: &[String], domain: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update(b"\n");
    hasher.update(format!("count={}\n", keys.len()).as_bytes());

    // Keys must be pre-sorted by caller
    for key in keys {
        hasher.update(key.as_bytes());
        hasher.update(b"\n");
    }

    hex::encode(hasher.finalize())
}

/// Compute shard salt from task list hash and run identity.
/// This prevents accidental mixing of shards from different tournament runs.
pub fn compute_shard_salt(task_list_hash: &str, strategy: &str, grid_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"qlx-shard-salt-v1|");
    hasher.update(task_list_hash.as_bytes());
    hasher.update(b"|");
    hasher.update(strategy.as_bytes());
    hasher.update(b"|");
    hasher.update(grid_hash.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute SHA256 of file contents (streaming, for large files).
pub fn hash_file(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {} for hashing", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

// =============================================================================
// Sharding Configuration
// =============================================================================

/// Sharding configuration.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// This shard's index (0-indexed)
    pub shard_index: usize,
    /// Total number of shards
    pub shard_count: usize,
}

impl ShardConfig {
    pub fn new(shard_index: usize, shard_count: usize) -> Result<Self> {
        if shard_count == 0 {
            anyhow::bail!("shard_count must be >= 1");
        }
        if shard_index >= shard_count {
            anyhow::bail!(
                "shard_index ({}) must be < shard_count ({})",
                shard_index,
                shard_count
            );
        }
        Ok(Self {
            shard_index,
            shard_count,
        })
    }

    /// Check if sharding is disabled (single-shard mode).
    pub fn is_single_shard(&self) -> bool {
        self.shard_count == 1
    }
}

// =============================================================================
// Shard Manifest v2
// =============================================================================

/// Shard manifest v2 with exactly-once proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    pub schema_version: String,
    pub shard_index: usize,
    pub shard_count: usize,

    // === Run Identity ===
    /// SHA256 of global task list (all shards see same value)
    pub task_list_hash: String,
    /// Salt binding this shard to specific run identity
    pub shard_salt: String,

    // === Assigned Set Proof ===
    /// Number of tasks assigned to this shard
    pub total_tasks_assigned: usize,
    /// SHA256 of sorted assigned task keys
    pub assigned_tasks_hash: String,
    /// Assigned task keys (segment\tparam_hash format)
    pub task_keys: Vec<String>,

    // === Completion Proof ===
    /// Number of tasks that completed successfully
    pub completed_runs: usize,
    /// SHA256 of sorted completed task keys
    pub completed_tasks_hash: String,
    /// Completed task keys (for debugging; can verify subset of assigned)
    pub completed_task_keys: Vec<String>,

    // === Output Integrity ===
    /// Number of rows written to results file
    pub results_rows_written: usize,
    /// SHA256 of results file bytes
    pub results_file_sha256: String,
    /// Results file format version (prevents silent format drift)
    pub results_format: String,

    // === Failure Tracking ===
    /// Number of failed runs (assigned - completed)
    pub failed_runs: usize,
    /// Count of duplicate result keys detected (should be 0)
    pub duplicate_result_keys: usize,
}

/// Results file format version constant.
pub const RESULTS_FORMAT_V1: &str = "grid_results_jsonl_v1";

/// Builder for constructing ShardManifest during shard execution.
pub struct ShardManifestBuilder {
    shard_config: ShardConfig,
    task_list_hash: String,
    shard_salt: String,
    assigned_keys: Vec<String>,
    completed_keys: Vec<String>,
    seen_keys: HashSet<String>,
    duplicate_count: usize,
}

impl ShardManifestBuilder {
    /// Create a new builder with shard configuration and run identity.
    pub fn new(
        shard_config: ShardConfig,
        task_list_hash: String,
        strategy: &str,
        grid_hash: &str,
        assigned_keys: Vec<String>,
    ) -> Self {
        let shard_salt = compute_shard_salt(&task_list_hash, strategy, grid_hash);
        Self {
            shard_config,
            task_list_hash,
            shard_salt,
            assigned_keys,
            completed_keys: Vec::new(),
            seen_keys: HashSet::new(),
            duplicate_count: 0,
        }
    }

    /// Record a completed task. Call this for each successful run.
    pub fn record_completion(&mut self, segment: &str, param_hash: &str) {
        let key = task_key(segment, param_hash);
        if self.seen_keys.contains(&key) {
            self.duplicate_count += 1;
            warn!("Duplicate task key detected: {}", key);
        } else {
            self.seen_keys.insert(key.clone());
            self.completed_keys.push(key);
        }
    }

    /// Finalize and build the manifest after writing results file.
    pub fn build(mut self, results_file_path: &Path) -> Result<ShardManifest> {
        // Sort keys for deterministic hashing
        self.assigned_keys.sort();
        self.completed_keys.sort();

        let assigned_tasks_hash = hash_sorted_keys(&self.assigned_keys, "qlx-task-keys-v1");
        let completed_tasks_hash = hash_sorted_keys(&self.completed_keys, "qlx-task-keys-v1");
        let results_file_sha256 = hash_file(results_file_path)?;

        let total_assigned = self.assigned_keys.len();
        let completed = self.completed_keys.len();

        Ok(ShardManifest {
            schema_version: SHARD_MANIFEST_SCHEMA.to_string(),
            shard_index: self.shard_config.shard_index,
            shard_count: self.shard_config.shard_count,
            task_list_hash: self.task_list_hash,
            shard_salt: self.shard_salt,
            total_tasks_assigned: total_assigned,
            assigned_tasks_hash,
            task_keys: self.assigned_keys,
            completed_runs: completed,
            completed_tasks_hash,
            completed_task_keys: self.completed_keys,
            results_rows_written: completed, // 1:1 with completed keys
            results_file_sha256,
            results_format: RESULTS_FORMAT_V1.to_string(),
            failed_runs: total_assigned.saturating_sub(completed),
            duplicate_result_keys: self.duplicate_count,
        })
    }
}

/// Write shard manifest to output directory.
pub fn write_shard_manifest(out_dir: &Path, manifest: &ShardManifest) -> Result<()> {
    let manifest_path = out_dir.join("shard_manifest.json");
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(&manifest_path, json)?;
    info!("Wrote {}", manifest_path.display());
    Ok(())
}

// =============================================================================
// Shard Assignment
// =============================================================================

/// Compute deterministic shard assignment for a task.
///
/// Uses SHA256(segment\tparam_hash) mod shard_count for stable distribution.
pub fn compute_shard_for_task(segment: &str, param_hash: &str, shard_count: usize) -> usize {
    let key = task_key(segment, param_hash);
    let hash = Sha256::digest(key.as_bytes());

    // Use first 8 bytes as u64 for modulo
    let hash_u64 = u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ]);

    (hash_u64 % shard_count as u64) as usize
}

/// Filter run tasks to only those assigned to this shard.
///
/// Returns (filtered_tasks, task_list_hash, assigned_keys) where:
/// - task_list_hash: hash of global task list (for merge validation)
/// - assigned_keys: canonical keys for this shard's tasks
pub fn filter_tasks_for_shard<T: Clone>(
    tasks: &[(String, String, T)], // (segment_name, param_hash, data)
    shard_config: &ShardConfig,
) -> (Vec<(String, String, T)>, String, Vec<String>) {
    // Build canonical keys and compute global task list hash
    let mut all_keys: Vec<String> = tasks
        .iter()
        .map(|(segment, param_hash, _)| task_key(segment, param_hash))
        .collect();
    all_keys.sort();

    let task_list_hash = hash_sorted_keys(&all_keys, "qlx-task-keys-v1");

    // Filter to this shard's tasks
    let mut filtered: Vec<(String, String, T)> = Vec::new();
    let mut assigned_keys: Vec<String> = Vec::new();

    for (segment, param_hash, data) in tasks {
        if compute_shard_for_task(segment, param_hash, shard_config.shard_count)
            == shard_config.shard_index
        {
            assigned_keys.push(task_key(segment, param_hash));
            filtered.push((segment.clone(), param_hash.clone(), data.clone()));
        }
    }

    (filtered, task_list_hash, assigned_keys)
}

// =============================================================================
// Merge Validation
// =============================================================================

/// Validation result for shard merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeValidation {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// Whether all tasks completed (no missing)
    pub complete: bool,
    /// Missing task keys (if any)
    pub missing_keys: Vec<String>,
    /// Global coverage hash (hash of all completed keys)
    pub global_completed_hash: String,
    /// Expected global hash (hash of all assigned keys)
    pub expected_global_hash: String,
}

/// Validate shard manifests for merge.
///
/// Implements the exactly-once validation algorithm:
/// 1. Validate run identity (salt, task_list_hash match)
/// 2. Validate assigned sets are correct
/// 3. Validate completions are subsets of assigned
/// 4. Validate output integrity
/// 5. Check global coverage
fn validate_shards_for_merge(manifests: &[ShardManifest]) -> MergeValidation {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    if manifests.is_empty() {
        return MergeValidation {
            valid: false,
            errors: vec!["No shard manifests provided".to_string()],
            warnings: vec![],
            complete: false,
            missing_keys: vec![],
            global_completed_hash: String::new(),
            expected_global_hash: String::new(),
        };
    }

    // Step 0: Validate run identity
    let expected_task_list_hash = &manifests[0].task_list_hash;
    let expected_shard_salt = &manifests[0].shard_salt;
    let expected_shard_count = manifests[0].shard_count;
    let expected_results_format = &manifests[0].results_format;

    for (i, m) in manifests.iter().enumerate() {
        if &m.task_list_hash != expected_task_list_hash {
            errors.push(format!(
                "Shard {} has different task_list_hash: {} vs {}",
                m.shard_index,
                &m.task_list_hash[..16],
                &expected_task_list_hash[..16]
            ));
        }
        if &m.shard_salt != expected_shard_salt {
            errors.push(format!(
                "Shard {} has different shard_salt: {} vs {}",
                m.shard_index,
                &m.shard_salt[..16],
                &expected_shard_salt[..16]
            ));
        }
        if m.shard_count != expected_shard_count {
            errors.push(format!(
                "Shard {} has different shard_count: {} vs {}",
                m.shard_index, m.shard_count, expected_shard_count
            ));
        }
        if m.schema_version != SHARD_MANIFEST_SCHEMA {
            warnings.push(format!(
                "Shard {} has schema_version '{}', expected '{}'",
                i, m.schema_version, SHARD_MANIFEST_SCHEMA
            ));
        }
        if &m.results_format != expected_results_format {
            errors.push(format!(
                "Shard {} has different results_format: '{}' vs '{}'",
                m.shard_index, m.results_format, expected_results_format
            ));
        }
    }

    // Check shard indices cover 0..shard_count-1 exactly once
    let mut seen_indices: Vec<bool> = vec![false; expected_shard_count];
    for m in manifests {
        if m.shard_index >= expected_shard_count {
            errors.push(format!(
                "Shard index {} >= shard_count {}",
                m.shard_index, expected_shard_count
            ));
        } else if seen_indices[m.shard_index] {
            errors.push(format!("Duplicate shard index: {}", m.shard_index));
        } else {
            seen_indices[m.shard_index] = true;
        }
    }

    let missing_shards: Vec<usize> = seen_indices
        .iter()
        .enumerate()
        .filter(|(_, seen)| !*seen)
        .map(|(i, _)| i)
        .collect();

    if !missing_shards.is_empty() {
        errors.push(format!("Missing shard indices: {:?}", missing_shards));
    }

    // Step 1: Validate assigned set hashes
    for m in manifests {
        // Recompute assigned_tasks_hash from task_keys
        let mut sorted_keys = m.task_keys.clone();
        sorted_keys.sort();
        let computed_hash = hash_sorted_keys(&sorted_keys, "qlx-task-keys-v1");

        if computed_hash != m.assigned_tasks_hash {
            errors.push(format!(
                "Shard {} assigned_tasks_hash mismatch: computed {} vs manifest {}",
                m.shard_index,
                &computed_hash[..16],
                &m.assigned_tasks_hash[..16]
            ));
        }

        if m.total_tasks_assigned != m.task_keys.len() {
            warnings.push(format!(
                "Shard {} total_tasks_assigned ({}) != task_keys.len() ({})",
                m.shard_index,
                m.total_tasks_assigned,
                m.task_keys.len()
            ));
        }
    }

    // Step 2: Validate completion is subset of assigned
    for m in manifests {
        let assigned_set: HashSet<&String> = m.task_keys.iter().collect();
        for key in &m.completed_task_keys {
            if !assigned_set.contains(key) {
                errors.push(format!(
                    "Shard {} completed key '{}' not in assigned set",
                    m.shard_index, key
                ));
            }
        }

        // Verify completed_tasks_hash
        let mut sorted_completed = m.completed_task_keys.clone();
        sorted_completed.sort();
        let computed_hash = hash_sorted_keys(&sorted_completed, "qlx-task-keys-v1");

        if computed_hash != m.completed_tasks_hash {
            errors.push(format!(
                "Shard {} completed_tasks_hash mismatch: computed {} vs manifest {}",
                m.shard_index,
                &computed_hash[..16],
                &m.completed_tasks_hash[..16]
            ));
        }

        if m.completed_runs != m.completed_task_keys.len() {
            warnings.push(format!(
                "Shard {} completed_runs ({}) != completed_task_keys.len() ({})",
                m.shard_index,
                m.completed_runs,
                m.completed_task_keys.len()
            ));
        }

        // Check for duplicates
        if m.duplicate_result_keys > 0 {
            errors.push(format!(
                "Shard {} reported {} duplicate result keys",
                m.shard_index, m.duplicate_result_keys
            ));
        }
    }

    // Step 3: Validate output integrity (row count)
    for m in manifests {
        if m.results_rows_written != m.completed_runs {
            warnings.push(format!(
                "Shard {} results_rows_written ({}) != completed_runs ({})",
                m.shard_index, m.results_rows_written, m.completed_runs
            ));
        }
    }

    // Step 4: Global coverage proof
    // Collect all assigned and completed keys
    let mut all_assigned: Vec<String> = Vec::new();
    let mut all_completed: Vec<String> = Vec::new();

    for m in manifests {
        all_assigned.extend(m.task_keys.iter().cloned());
        all_completed.extend(m.completed_task_keys.iter().cloned());
    }

    // Check for duplicates across shards (disjointness)
    let assigned_set: HashSet<String> = all_assigned.iter().cloned().collect();
    let completed_set: HashSet<String> = all_completed.iter().cloned().collect();

    if assigned_set.len() != all_assigned.len() {
        let dup_count = all_assigned.len() - assigned_set.len();
        errors.push(format!(
            "{} duplicate assigned keys across shards (disjointness violation)",
            dup_count
        ));
    }

    if completed_set.len() != all_completed.len() {
        let dup_count = all_completed.len() - completed_set.len();
        errors.push(format!(
            "{} duplicate completed keys across shards (disjointness violation)",
            dup_count
        ));
    }

    // Compute global hashes
    let mut sorted_all_assigned: Vec<String> = assigned_set.iter().cloned().collect();
    sorted_all_assigned.sort();
    let assigned_union_hash = hash_sorted_keys(&sorted_all_assigned, "qlx-task-keys-v1");

    let mut sorted_all_completed: Vec<String> = completed_set.iter().cloned().collect();
    sorted_all_completed.sort();
    let global_completed_hash = hash_sorted_keys(&sorted_all_completed, "qlx-task-keys-v1");

    // CRITICAL: Verify that union(assigned) covers the original task universe exactly
    // This is the ground truth check - task_list_hash is the authoritative source
    if assigned_union_hash != *expected_task_list_hash {
        errors.push(format!(
            "Global coverage mismatch: union(assigned_keys) hash {} != task_list_hash {}. \
             Shards do not cover the complete task universe.",
            &assigned_union_hash[..16],
            &expected_task_list_hash[..16]
        ));
    }

    // Check coverage: which assigned tasks were not completed
    let missing_keys: Vec<String> = assigned_set.difference(&completed_set).cloned().collect();

    let complete = missing_keys.is_empty();

    if !missing_keys.is_empty() {
        warnings.push(format!(
            "{} tasks not completed (partial merge)",
            missing_keys.len()
        ));
    }

    MergeValidation {
        valid: errors.is_empty(),
        errors,
        warnings,
        complete,
        missing_keys,
        global_completed_hash,
        // task_list_hash is the authoritative ground truth for the global task universe
        expected_global_hash: expected_task_list_hash.clone(),
    }
}

// =============================================================================
// Merged Manifest
// =============================================================================

/// Merged tournament manifest with coverage proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedTournamentManifest {
    pub schema_version: String,
    pub strategy: String,
    pub total_shards: usize,
    pub total_runs: usize,

    // === Run Identity ===
    pub task_list_hash: String,
    pub shard_salt: String,

    // === Coverage Proof ===
    /// Whether merge is complete (all tasks completed)
    pub complete: bool,
    /// SHA256 of all completed task keys (global coverage)
    pub global_completed_hash: String,
    /// Expected hash if all tasks completed
    pub expected_global_hash: String,
    /// Number of missing tasks (if incomplete)
    pub missing_task_count: usize,

    // === Shard Provenance ===
    /// SHA256 hashes of all input shard manifests
    pub shard_manifest_hashes: Vec<String>,
    /// Per-shard completion counts
    pub shard_completion_counts: Vec<(usize, usize, usize)>, // (index, assigned, completed)

    // === Output Integrity ===
    pub merged_results_sha256: String,
    pub merged_results_rows: usize,
}

// =============================================================================
// Merge Logic
// =============================================================================

/// Read shard manifest from a shard directory.
fn read_shard_manifest(shard_dir: &Path) -> Result<ShardManifest> {
    let manifest_path = shard_dir.join("shard_manifest.json");
    let content = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("Failed to read {}", manifest_path.display()))?;
    let manifest: ShardManifest = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {}", manifest_path.display()))?;
    Ok(manifest)
}

/// Read and validate results.jsonl from a shard directory.
fn read_shard_results(shard_dir: &Path, manifest: &ShardManifest) -> Result<Vec<GridRunResult>> {
    let results_path = shard_dir.join("results.jsonl");

    // Verify file hash matches manifest
    let actual_hash = hash_file(&results_path)?;
    if actual_hash != manifest.results_file_sha256 {
        anyhow::bail!(
            "Results file hash mismatch for shard {}: expected {} got {}",
            manifest.shard_index,
            &manifest.results_file_sha256[..16],
            &actual_hash[..16]
        );
    }

    let file = std::fs::File::open(&results_path)
        .with_context(|| format!("Failed to open {}", results_path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut results = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let result: GridRunResult = serde_json::from_str(&line).with_context(|| {
            format!(
                "Failed to parse line {} in {}",
                line_num + 1,
                results_path.display()
            )
        })?;
        results.push(result);
    }

    // Verify row count
    if results.len() != manifest.results_rows_written {
        anyhow::bail!(
            "Results row count mismatch for shard {}: expected {} got {}",
            manifest.shard_index,
            manifest.results_rows_written,
            results.len()
        );
    }

    Ok(results)
}

/// Merge multiple shard outputs into a single tournament result.
///
/// Implements exactly-once merge with cryptographic validation:
/// 1. Load and validate all shard manifests
/// 2. Verify shard completion proofs
/// 3. Check global coverage (exactly-once)
/// 4. Concatenate results and rebuild artifacts
///
/// If `require_complete` is true, fails when any tasks are missing.
/// If false (partial mode), proceeds with warning and writes missing_tasks.txt.
pub fn merge_shards(
    shard_dirs: &[&Path],
    out_dir: &Path,
    strategy: &str,
    two_pass_config: Option<&two_pass::TwoPassConfig>,
    require_complete: bool,
) -> Result<()> {
    info!(
        "=== Merging {} shards (v2 exactly-once) ===",
        shard_dirs.len()
    );

    // Create output directory
    std::fs::create_dir_all(out_dir)?;

    // Load all shard manifests
    let mut manifests: Vec<ShardManifest> = Vec::new();
    let mut manifest_hashes: Vec<String> = Vec::new();

    for shard_dir in shard_dirs {
        let manifest = read_shard_manifest(shard_dir)?;
        info!(
            "Shard {}/{}: {}/{} tasks completed from {}",
            manifest.shard_index,
            manifest.shard_count,
            manifest.completed_runs,
            manifest.total_tasks_assigned,
            shard_dir.display()
        );

        // Compute manifest hash for audit trail
        let manifest_content = std::fs::read_to_string(shard_dir.join("shard_manifest.json"))?;
        let hash = hex::encode(Sha256::digest(manifest_content.as_bytes()));
        manifest_hashes.push(hash);

        manifests.push(manifest);
    }

    // === Exactly-Once Validation ===
    info!("Validating shard proofs...");
    let validation = validate_shards_for_merge(&manifests);

    // Log validation results
    for warning in &validation.warnings {
        warn!("{}", warning);
    }

    if !validation.valid {
        for error in &validation.errors {
            tracing::error!("{}", error);
        }
        anyhow::bail!(
            "Shard validation failed with {} errors. Cannot merge.",
            validation.errors.len()
        );
    }

    if !validation.complete {
        if require_complete {
            // Write missing_tasks.txt for debugging even in complete mode
            let missing_path = out_dir.join("missing_tasks.txt");
            let mut missing_file = std::fs::File::create(&missing_path)?;
            for key in &validation.missing_keys {
                writeln!(missing_file, "{}", key)?;
            }
            info!(
                "Wrote {} missing task keys to {}",
                validation.missing_keys.len(),
                missing_path.display()
            );

            anyhow::bail!(
                "Incomplete merge: {} tasks missing. Use --merge-mode partial to allow incomplete merges.",
                validation.missing_keys.len()
            );
        } else {
            // Partial mode: proceed but write missing_tasks.txt
            warn!(
                "Incomplete merge: {} tasks missing. Proceeding with partial results.",
                validation.missing_keys.len()
            );

            let missing_path = out_dir.join("missing_tasks.txt");
            let mut missing_file = std::fs::File::create(&missing_path)?;
            for key in &validation.missing_keys {
                writeln!(missing_file, "{}", key)?;
            }
            info!(
                "Wrote {} missing task keys to {}",
                validation.missing_keys.len(),
                missing_path.display()
            );
        }
    } else {
        info!("All shard proofs validated. Exactly-once coverage confirmed.");
    }

    // Write validation report
    let validation_path = out_dir.join("merge_validation.json");
    let validation_json = serde_json::to_string_pretty(&validation)?;
    std::fs::write(&validation_path, validation_json)?;
    info!("Wrote {}", validation_path.display());

    // === Load Results (with integrity checks) ===
    let mut all_results: Vec<GridRunResult> = Vec::new();
    for (shard_dir, manifest) in shard_dirs.iter().zip(manifests.iter()) {
        let results = read_shard_results(shard_dir, manifest)?;
        info!(
            "Loaded {} verified results from shard {}",
            results.len(),
            manifest.shard_index
        );
        all_results.extend(results);
    }

    info!("Total merged results: {}", all_results.len());

    // Sort deterministically by (segment, param_hash)
    all_results.sort_by(|a, b| match a.segment.cmp(&b.segment) {
        std::cmp::Ordering::Equal => a.param_hash.cmp(&b.param_hash),
        other => other,
    });

    // Write merged results.jsonl
    let results_path = out_dir.join("results.jsonl");
    {
        let file = std::fs::File::create(&results_path)?;
        let mut writer = std::io::BufWriter::new(file);
        for result in &all_results {
            let line = serde_json::to_string(result)?;
            writeln!(writer, "{}", line)?;
        }
    }
    info!("Wrote {}", results_path.display());

    // Hash the merged results file
    let merged_results_sha256 = hash_file(&results_path)?;

    // Build and write leaderboard
    let leaderboard = build_grid_leaderboard(&all_results);
    let leaderboard_path = out_dir.join("leaderboard.json");
    let leaderboard_json = serde_json::to_string_pretty(&leaderboard)?;
    std::fs::write(&leaderboard_path, leaderboard_json)?;
    info!("Wrote {}", leaderboard_path.display());

    // Build and write promotion candidates
    let candidates = select_promotion_candidates(&leaderboard, strategy, 10);
    let candidates_path = out_dir.join("promotion_candidates.json");
    let candidates_json = serde_json::to_string_pretty(&candidates)?;
    std::fs::write(&candidates_path, candidates_json)?;
    info!("Wrote {}", candidates_path.display());

    // === Two-Pass Mode: Recompute aggregations ===
    if let Some(tp_config) = two_pass_config {
        info!("Recomputing Pass-1 aggregations from merged results...");
        let mut aggregations = two_pass::aggregate_pass1_results(&all_results, tp_config);
        let selected_configs = two_pass::select_configs_for_pass2(&mut aggregations, tp_config);

        // Write aggregation artifacts
        let agg_path = out_dir.join("pass1_config_agg.csv");
        {
            let file = std::fs::File::create(&agg_path)?;
            let mut writer = std::io::BufWriter::new(file);
            writeln!(
                writer,
                "param_hash,segment_count,valid_count,refused_count,refuse_rate,median_score,p20_score,mean_score,promo_score,promoted"
            )?;
            for agg in &aggregations {
                writeln!(
                    writer,
                    "{},{},{},{},{:.4},{:.6},{:.6},{:.6},{:.6},{}",
                    agg.param_hash,
                    agg.segment_count,
                    agg.valid_count,
                    agg.refused_count,
                    agg.refuse_rate,
                    agg.median_score,
                    agg.p20_score,
                    agg.mean_score,
                    agg.promo_score,
                    if agg.promoted { 1 } else { 0 }
                )?;
            }
        }
        info!("Wrote {}", agg_path.display());

        // Write selected configs
        let selected_path = out_dir.join("selected_configs.jsonl");
        {
            let file = std::fs::File::create(&selected_path)?;
            let mut writer = std::io::BufWriter::new(file);
            for cfg in &selected_configs {
                let line = serde_json::to_string(cfg)?;
                writeln!(writer, "{}", line)?;
            }
        }
        info!("Wrote {}", selected_path.display());

        info!(
            "Pass 1 merged: {} configs evaluated, {} promoted",
            aggregations.len(),
            selected_configs.len()
        );
    }

    // Compute stability metrics
    info!("Computing stability metrics...");
    let segments: std::collections::HashSet<String> =
        all_results.iter().map(|r| r.segment.clone()).collect();
    let stability = two_pass::compute_pass2_stability(&all_results, 10);
    let stability_stats = two_pass::compute_pass2_stats(&stability, segments.len());
    two_pass::write_pass2_stability(out_dir, &stability, &stability_stats)?;

    // Build per-shard completion summary
    let shard_completion_counts: Vec<(usize, usize, usize)> = manifests
        .iter()
        .map(|m| (m.shard_index, m.total_tasks_assigned, m.completed_runs))
        .collect();

    // Write merged manifest
    let merged_manifest = MergedTournamentManifest {
        schema_version: MERGED_MANIFEST_SCHEMA.to_string(),
        strategy: strategy.to_string(),
        total_shards: manifests.len(),
        total_runs: all_results.len(),
        task_list_hash: manifests[0].task_list_hash.clone(),
        shard_salt: manifests[0].shard_salt.clone(),
        complete: validation.complete,
        global_completed_hash: validation.global_completed_hash,
        expected_global_hash: validation.expected_global_hash,
        missing_task_count: validation.missing_keys.len(),
        shard_manifest_hashes: manifest_hashes,
        shard_completion_counts,
        merged_results_sha256,
        merged_results_rows: all_results.len(),
    };

    let manifest_path = out_dir.join("merged_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&merged_manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;
    info!("Wrote {}", manifest_path.display());

    info!("=== Merge Complete ===");
    info!("  Total runs: {}", all_results.len());
    info!("  Complete: {}", validation.complete);
    info!("  Leaderboard entries: {}", leaderboard.len());
    info!("  Output: {}", out_dir.display());

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_key_format() {
        let key = task_key("segment_2026_01", "abc123");
        assert_eq!(key, "segment_2026_01\tabc123");

        let (seg, hash) = parse_task_key(&key).unwrap();
        assert_eq!(seg, "segment_2026_01");
        assert_eq!(hash, "abc123");
    }

    #[test]
    fn test_hash_sorted_keys_deterministic() {
        let keys = vec![
            "seg_a\thash1".to_string(),
            "seg_b\thash2".to_string(),
            "seg_c\thash3".to_string(),
        ];

        let hash1 = hash_sorted_keys(&keys, "qlx-task-keys-v1");
        let hash2 = hash_sorted_keys(&keys, "qlx-task-keys-v1");
        assert_eq!(hash1, hash2);

        // Different domain should produce different hash
        let hash3 = hash_sorted_keys(&keys, "qlx-different-domain");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_shard_salt_deterministic() {
        let salt1 = compute_shard_salt("taskhash123", "strategy_a", "gridhash456");
        let salt2 = compute_shard_salt("taskhash123", "strategy_a", "gridhash456");
        assert_eq!(salt1, salt2);

        // Different inputs should produce different salt
        let salt3 = compute_shard_salt("taskhash123", "strategy_b", "gridhash456");
        assert_ne!(salt1, salt3);
    }

    #[test]
    fn test_shard_assignment_deterministic() {
        let shard1 = compute_shard_for_task("segment_2026_01", "abc123", 4);
        let shard2 = compute_shard_for_task("segment_2026_01", "abc123", 4);
        assert_eq!(shard1, shard2);
    }

    #[test]
    fn test_shard_assignment_distribution() {
        let mut counts = [0usize; 4];
        for i in 0..100 {
            let segment = format!("segment_{}", i);
            let param_hash = format!("hash_{}", i % 10);
            let shard = compute_shard_for_task(&segment, &param_hash, 4);
            counts[shard] += 1;
        }

        for (i, count) in counts.iter().enumerate() {
            assert!(*count > 0, "Shard {} got no tasks", i);
            assert!(*count < 100, "Shard {} got all tasks", i);
        }
    }

    #[test]
    fn test_shard_config_validation() {
        assert!(ShardConfig::new(0, 4).is_ok());
        assert!(ShardConfig::new(3, 4).is_ok());
        assert!(ShardConfig::new(4, 4).is_err());
        assert!(ShardConfig::new(0, 0).is_err());
    }

    #[test]
    fn test_filter_tasks_for_shard() {
        let tasks: Vec<(String, String, i32)> = (0..20)
            .map(|i| (format!("seg_{}", i), format!("hash_{}", i), i))
            .collect();

        let config = ShardConfig::new(0, 4).unwrap();
        let (filtered, hash1, assigned_keys) = filter_tasks_for_shard(&tasks, &config);

        // Should get roughly 1/4 of tasks
        assert!(!filtered.is_empty() && filtered.len() < 20);
        assert_eq!(filtered.len(), assigned_keys.len());

        // Hash should be consistent
        let (_, hash2, _) = filter_tasks_for_shard(&tasks, &config);
        assert_eq!(hash1, hash2);

        // Different shard should get different tasks
        let config2 = ShardConfig::new(1, 4).unwrap();
        let (filtered2, _, _) = filter_tasks_for_shard(&tasks, &config2);

        // No overlap between shard 0 and shard 1
        let keys1: HashSet<_> = filtered
            .iter()
            .map(|(s, h, _)| format!("{}:{}", s, h))
            .collect();
        let keys2: HashSet<_> = filtered2
            .iter()
            .map(|(s, h, _)| format!("{}:{}", s, h))
            .collect();
        assert!(keys1.is_disjoint(&keys2));
    }

    #[test]
    fn test_merge_validation_empty() {
        let validation = validate_shards_for_merge(&[]);
        assert!(!validation.valid);
        assert!(!validation.errors.is_empty());
    }

    #[test]
    fn test_merge_validation_global_coverage() {
        // Create two valid shards that together cover the full task universe
        let all_keys = vec![
            "seg_a\thash1".to_string(),
            "seg_b\thash2".to_string(),
            "seg_c\thash3".to_string(),
            "seg_d\thash4".to_string(),
        ];
        let mut sorted_all = all_keys.clone();
        sorted_all.sort();
        let task_list_hash = hash_sorted_keys(&sorted_all, "qlx-task-keys-v1");
        let shard_salt = compute_shard_salt(&task_list_hash, "test_strategy", "grid_hash");

        // Shard 0 gets tasks a, b
        let shard0_keys = vec!["seg_a\thash1".to_string(), "seg_b\thash2".to_string()];
        let mut sorted_s0 = shard0_keys.clone();
        sorted_s0.sort();
        let s0_hash = hash_sorted_keys(&sorted_s0, "qlx-task-keys-v1");

        // Shard 1 gets tasks c, d
        let shard1_keys = vec!["seg_c\thash3".to_string(), "seg_d\thash4".to_string()];
        let mut sorted_s1 = shard1_keys.clone();
        sorted_s1.sort();
        let s1_hash = hash_sorted_keys(&sorted_s1, "qlx-task-keys-v1");

        let manifest0 = ShardManifest {
            schema_version: SHARD_MANIFEST_SCHEMA.to_string(),
            shard_index: 0,
            shard_count: 2,
            task_list_hash: task_list_hash.clone(),
            shard_salt: shard_salt.clone(),
            total_tasks_assigned: 2,
            assigned_tasks_hash: s0_hash.clone(),
            task_keys: shard0_keys.clone(),
            completed_runs: 2,
            completed_tasks_hash: s0_hash.clone(),
            completed_task_keys: shard0_keys.clone(),
            results_rows_written: 2,
            results_file_sha256: "fake_hash".to_string(),
            results_format: RESULTS_FORMAT_V1.to_string(),
            failed_runs: 0,
            duplicate_result_keys: 0,
        };

        let manifest1 = ShardManifest {
            schema_version: SHARD_MANIFEST_SCHEMA.to_string(),
            shard_index: 1,
            shard_count: 2,
            task_list_hash: task_list_hash.clone(),
            shard_salt: shard_salt.clone(),
            total_tasks_assigned: 2,
            assigned_tasks_hash: s1_hash.clone(),
            task_keys: shard1_keys.clone(),
            completed_runs: 2,
            completed_tasks_hash: s1_hash.clone(),
            completed_task_keys: shard1_keys.clone(),
            results_rows_written: 2,
            results_file_sha256: "fake_hash".to_string(),
            results_format: RESULTS_FORMAT_V1.to_string(),
            failed_runs: 0,
            duplicate_result_keys: 0,
        };

        // Valid: union(assigned) == task_list_hash
        let validation = validate_shards_for_merge(&[manifest0.clone(), manifest1.clone()]);
        assert!(
            validation.valid,
            "Expected valid merge, got errors: {:?}",
            validation.errors
        );
        assert!(validation.complete);
        assert_eq!(validation.expected_global_hash, task_list_hash);

        // Invalid: missing shard 1 means union(assigned) != task_list_hash
        let validation_partial = validate_shards_for_merge(std::slice::from_ref(&manifest0));
        assert!(
            !validation_partial.valid,
            "Expected invalid merge with missing shard"
        );
        assert!(
            validation_partial
                .errors
                .iter()
                .any(|e| e.contains("Missing shard")),
            "Expected 'Missing shard' error, got: {:?}",
            validation_partial.errors
        );
    }

    #[test]
    fn test_merge_validation_results_format_mismatch() {
        let task_keys = vec!["seg_a\thash1".to_string()];
        let mut sorted = task_keys.clone();
        sorted.sort();
        let task_list_hash = hash_sorted_keys(&sorted, "qlx-task-keys-v1");
        let shard_salt = compute_shard_salt(&task_list_hash, "test", "grid");
        let hash = hash_sorted_keys(&sorted, "qlx-task-keys-v1");

        let manifest0 = ShardManifest {
            schema_version: SHARD_MANIFEST_SCHEMA.to_string(),
            shard_index: 0,
            shard_count: 2,
            task_list_hash: task_list_hash.clone(),
            shard_salt: shard_salt.clone(),
            total_tasks_assigned: 1,
            assigned_tasks_hash: hash.clone(),
            task_keys: task_keys.clone(),
            completed_runs: 1,
            completed_tasks_hash: hash.clone(),
            completed_task_keys: task_keys.clone(),
            results_rows_written: 1,
            results_file_sha256: "fake".to_string(),
            results_format: RESULTS_FORMAT_V1.to_string(),
            failed_runs: 0,
            duplicate_result_keys: 0,
        };

        let manifest1 = ShardManifest {
            results_format: "grid_results_jsonl_v2".to_string(), // Different format!
            shard_index: 1,
            ..manifest0.clone()
        };

        let validation = validate_shards_for_merge(&[manifest0, manifest1]);
        assert!(!validation.valid);
        assert!(
            validation
                .errors
                .iter()
                .any(|e| e.contains("results_format")),
            "Expected results_format mismatch error, got: {:?}",
            validation.errors
        );
    }
}
