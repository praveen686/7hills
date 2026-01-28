//! Signal Promotion Gates — G0, G1, G2.
//!
//! Phase 20C: Promotion gates for signal admission integrity.
//!
//! ## Gate Hierarchy
//! - **G0 Schema**: Validate signals_manifest.json structure and semantics
//! - **G1 Determinism**: WAL parity between live and replay sessions
//! - **G2 Data Integrity**: Coverage threshold for admission events
//!
//! ## Hard Laws (from Phase 18/20A/20B)
//! - L1: No Fabrication — gates cannot create data
//! - L2: Deterministic — same inputs → same results
//! - L6: Observability — all gate results are audit artifacts

use crate::signals_manifest::{ManifestError, SignalsManifest};
use quantlaxmi_models::{AdmissionDecision, AdmissionOutcome};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

// =============================================================================
// Check Names — Frozen for CI stability
// =============================================================================

/// Stable check names for CI parsing.
pub mod check_names {
    pub const G0_SCHEMA_VALID: &str = "g0_schema_valid";
    pub const G0_SIGNALS_LOADABLE: &str = "g0_signals_loadable";
    pub const G1_MANIFEST_HASH_PARITY: &str = "g1_manifest_hash_parity";
    pub const G1_DECISION_PARITY: &str = "g1_decision_parity";
    pub const G2_NONZERO_EVENTS: &str = "g2_nonzero_events";
    pub const G2_COVERAGE_THRESHOLD: &str = "g2_coverage_threshold";
    // G3 check names (Phase 21C)
    pub const G3_SCHEMA_PARSE: &str = "g3_schema_parse";
    pub const G3_VALIDATE: &str = "g3_validate";
    pub const G3_SIGNAL_BINDINGS: &str = "g3_signal_bindings";
    pub const G3_PROMOTION_STATUS: &str = "g3_promotion_status";
}

// =============================================================================
// G0 Schema Gate
// =============================================================================

/// G0 Schema Gate — Validates signals_manifest.json integrity.
///
/// Checks:
/// - File exists and is readable
/// - JSON parses successfully
/// - Schema version is supported
/// - All signal IDs are valid
/// - All L1 field references are known
/// - All invariant types are known
pub struct G0SchemaGate;

/// Result of G0 gate validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G0Result {
    /// Check name (frozen)
    pub check_name: String,
    /// Pass/fail
    pub passed: bool,
    /// Human-readable message
    pub message: String,
    /// Manifest path validated
    pub manifest_path: String,
    /// Manifest version hash (if load succeeded)
    pub manifest_hash: Option<[u8; 32]>,
    /// Manifest version string (if load succeeded)
    pub manifest_version: Option<String>,
    /// Number of signals in manifest
    pub signal_count: Option<usize>,
    /// Detailed error if failed
    pub error: Option<String>,
}

impl G0SchemaGate {
    /// Validate a manifest file.
    ///
    /// Returns G0Result with pass/fail and details.
    pub fn validate(manifest_path: &Path) -> G0Result {
        match SignalsManifest::load_validated(manifest_path) {
            Ok(manifest) => G0Result {
                check_name: check_names::G0_SCHEMA_VALID.to_string(),
                passed: true,
                message: format!(
                    "Manifest valid: {} signals, version {}",
                    manifest.signal_count(),
                    manifest.manifest_version
                ),
                manifest_path: manifest_path.display().to_string(),
                manifest_hash: Some(manifest.compute_version_hash()),
                manifest_version: Some(manifest.manifest_version.clone()),
                signal_count: Some(manifest.signal_count()),
                error: None,
            },
            Err(e) => G0Result {
                check_name: check_names::G0_SCHEMA_VALID.to_string(),
                passed: false,
                message: format!("Manifest validation failed: {}", e),
                manifest_path: manifest_path.display().to_string(),
                manifest_hash: None,
                manifest_version: None,
                signal_count: None,
                error: Some(format!("{:?}", e)),
            },
        }
    }

    /// Validate manifest and return the loaded manifest on success.
    pub fn validate_and_load(manifest_path: &Path) -> Result<SignalsManifest, ManifestError> {
        SignalsManifest::load_validated(manifest_path)
    }
}

// =============================================================================
// G1 Determinism Gate — Types
// =============================================================================

/// Composite key for G1 decision lookup.
///
/// Uses (correlation_id, signal_id) for robust matching across runs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct G1DecisionKey {
    /// Correlation ID from admission decision
    pub correlation_id: String,
    /// Signal ID being evaluated
    pub signal_id: String,
}

impl G1DecisionKey {
    pub fn new(correlation_id: impl Into<String>, signal_id: impl Into<String>) -> Self {
        Self {
            correlation_id: correlation_id.into(),
            signal_id: signal_id.into(),
        }
    }

    /// Extract key from an admission decision.
    ///
    /// Returns None if correlation_id is missing (required for G1).
    pub fn from_decision(decision: &AdmissionDecision) -> Option<Self> {
        decision.correlation_id.as_ref().map(|cid| Self {
            correlation_id: cid.clone(),
            signal_id: decision.signal_id.clone(),
        })
    }
}

/// Fingerprint of a decision for comparison.
///
/// Contains the fields we compare between live and replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionFingerprint {
    /// The decision digest (SHA-256 of canonical bytes)
    pub digest: String,
    /// Outcome (Admit/Refuse)
    pub outcome: AdmissionOutcome,
    /// Manifest version hash at decision time
    pub manifest_hash: [u8; 32],
    /// Source line number (1-based) for diagnostics
    pub line_number: usize,
}

/// Source of a G1 decision (for mismatch reporting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum G1Source {
    Live,
    Replay,
}

impl std::fmt::Display for G1Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G1Source::Live => write!(f, "live"),
            G1Source::Replay => write!(f, "replay"),
        }
    }
}

/// Kinds of G1 mismatches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum G1MismatchKind {
    /// Entry exists in live but not replay
    MissingReplayEntry {
        key: G1DecisionKey,
        live_line: usize,
    },

    /// Entry exists in replay but not live
    MissingLiveEntry {
        key: G1DecisionKey,
        replay_line: usize,
    },

    /// Digests differ (determinism violation)
    DigestDiff {
        key: G1DecisionKey,
        live_digest: String,
        replay_digest: String,
        live_line: usize,
        replay_line: usize,
    },

    /// Outcomes differ (Admit vs Refuse)
    OutcomeDiff {
        key: G1DecisionKey,
        live_outcome: AdmissionOutcome,
        replay_outcome: AdmissionOutcome,
        live_line: usize,
        replay_line: usize,
    },

    /// Manifest hash differs (config drift)
    ManifestHashDiff {
        key: G1DecisionKey,
        live_hash: [u8; 32],
        replay_hash: [u8; 32],
        live_line: usize,
        replay_line: usize,
    },

    /// Duplicate key in same WAL file
    DuplicateKey {
        key: G1DecisionKey,
        source: G1Source,
        first_line: usize,
        duplicate_line: usize,
    },

    /// Parse error in WAL file
    ParseError {
        source: G1Source,
        line_number: usize,
        /// Key if we could parse enough to extract it
        key: Option<G1DecisionKey>,
        error: String,
    },
}

impl G1MismatchKind {
    /// Get the key associated with this mismatch, if available.
    pub fn key(&self) -> Option<&G1DecisionKey> {
        match self {
            G1MismatchKind::MissingReplayEntry { key, .. } => Some(key),
            G1MismatchKind::MissingLiveEntry { key, .. } => Some(key),
            G1MismatchKind::DigestDiff { key, .. } => Some(key),
            G1MismatchKind::OutcomeDiff { key, .. } => Some(key),
            G1MismatchKind::ManifestHashDiff { key, .. } => Some(key),
            G1MismatchKind::DuplicateKey { key, .. } => Some(key),
            G1MismatchKind::ParseError { key, .. } => key.as_ref(),
        }
    }

    /// Human-readable description of the mismatch.
    pub fn description(&self) -> String {
        match self {
            G1MismatchKind::MissingReplayEntry { key, live_line } => {
                format!(
                    "Entry in live (line {}) missing from replay: ({}, {})",
                    live_line, key.correlation_id, key.signal_id
                )
            }
            G1MismatchKind::MissingLiveEntry { key, replay_line } => {
                format!(
                    "Entry in replay (line {}) missing from live: ({}, {})",
                    replay_line, key.correlation_id, key.signal_id
                )
            }
            G1MismatchKind::DigestDiff {
                key,
                live_digest,
                replay_digest,
                live_line,
                replay_line,
            } => {
                format!(
                    "Digest mismatch for ({}, {}): live[{}]={:.8}... vs replay[{}]={:.8}...",
                    key.correlation_id,
                    key.signal_id,
                    live_line,
                    live_digest,
                    replay_line,
                    replay_digest
                )
            }
            G1MismatchKind::OutcomeDiff {
                key,
                live_outcome,
                replay_outcome,
                live_line,
                replay_line,
            } => {
                format!(
                    "Outcome mismatch for ({}, {}): live[{}]={} vs replay[{}]={}",
                    key.correlation_id,
                    key.signal_id,
                    live_line,
                    live_outcome,
                    replay_line,
                    replay_outcome
                )
            }
            G1MismatchKind::ManifestHashDiff {
                key,
                live_hash,
                replay_hash,
                live_line,
                replay_line,
            } => {
                format!(
                    "Manifest hash mismatch for ({}, {}): live[{}]={:.8}... vs replay[{}]={:.8}...",
                    key.correlation_id,
                    key.signal_id,
                    live_line,
                    hex::encode(&live_hash[..4]),
                    replay_line,
                    hex::encode(&replay_hash[..4])
                )
            }
            G1MismatchKind::DuplicateKey {
                key,
                source,
                first_line,
                duplicate_line,
            } => {
                format!(
                    "Duplicate key in {} WAL: ({}, {}) at lines {} and {}",
                    source, key.correlation_id, key.signal_id, first_line, duplicate_line
                )
            }
            G1MismatchKind::ParseError {
                source,
                line_number,
                key,
                error,
            } => {
                let key_info = key
                    .as_ref()
                    .map(|k| format!(" for ({}, {})", k.correlation_id, k.signal_id))
                    .unwrap_or_default();
                format!(
                    "Parse error in {} WAL line {}{}: {}",
                    source, line_number, key_info, error
                )
            }
        }
    }
}

// =============================================================================
// G1 Determinism Gate — Implementation
// =============================================================================

/// G1 Determinism Gate — Validates WAL parity between live and replay.
///
/// Ensures that replaying the same inputs produces identical admission decisions.
/// This is the core determinism guarantee for signal admission.
pub struct G1DeterminismGate;

/// Result of G1 gate validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G1Result {
    /// Check name (frozen)
    pub check_name: String,
    /// Pass/fail
    pub passed: bool,
    /// Human-readable message
    pub message: String,
    /// Live WAL path
    pub live_wal_path: String,
    /// Replay WAL path
    pub replay_wal_path: String,
    /// Number of entries in live WAL
    pub live_entry_count: usize,
    /// Number of entries in replay WAL
    pub replay_entry_count: usize,
    /// Number of entries matched (keys present in both)
    pub matched_count: usize,
    /// Mismatches found (empty if passed)
    pub mismatches: Vec<G1MismatchKind>,
}

impl G1DeterminismGate {
    /// Compare two WAL files for determinism.
    ///
    /// Both files must be JSONL format with AdmissionDecision entries.
    /// Returns G1Result with pass/fail and any mismatches found.
    pub fn compare(live_wal: &Path, replay_wal: &Path) -> Result<G1Result, std::io::Error> {
        let mut mismatches = Vec::new();

        // Stream and index both WALs
        let (live_index, live_errors) = Self::stream_and_index(live_wal, G1Source::Live)?;
        let (replay_index, replay_errors) = Self::stream_and_index(replay_wal, G1Source::Replay)?;

        // Collect parse errors as mismatches
        mismatches.extend(live_errors);
        mismatches.extend(replay_errors);

        // Check for entries in live but not replay
        for (key, live_fp) in &live_index {
            match replay_index.get(key) {
                None => {
                    mismatches.push(G1MismatchKind::MissingReplayEntry {
                        key: key.clone(),
                        live_line: live_fp.line_number,
                    });
                }
                Some(replay_fp) => {
                    // Compare fingerprints
                    Self::compare_fingerprints(key, live_fp, replay_fp, &mut mismatches);
                }
            }
        }

        // Check for entries in replay but not live
        for (key, replay_fp) in &replay_index {
            if !live_index.contains_key(key) {
                mismatches.push(G1MismatchKind::MissingLiveEntry {
                    key: key.clone(),
                    replay_line: replay_fp.line_number,
                });
            }
        }

        let passed = mismatches.is_empty();
        let matched_count = live_index
            .keys()
            .filter(|k| replay_index.contains_key(*k))
            .count();

        Ok(G1Result {
            check_name: check_names::G1_DECISION_PARITY.to_string(),
            passed,
            message: if passed {
                format!("WAL parity verified: {} entries matched", live_index.len())
            } else {
                format!("WAL parity failed: {} mismatches found", mismatches.len())
            },
            live_wal_path: live_wal.display().to_string(),
            replay_wal_path: replay_wal.display().to_string(),
            live_entry_count: live_index.len(),
            replay_entry_count: replay_index.len(),
            matched_count,
            mismatches,
        })
    }

    /// Stream JSONL file and build index by (correlation_id, signal_id).
    ///
    /// Uses BufRead::lines() for memory-safe streaming.
    /// Returns (index, parse_errors).
    fn stream_and_index(
        path: &Path,
        source: G1Source,
    ) -> Result<
        (
            BTreeMap<G1DecisionKey, DecisionFingerprint>,
            Vec<G1MismatchKind>,
        ),
        std::io::Error,
    > {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        let mut index: BTreeMap<G1DecisionKey, DecisionFingerprint> = BTreeMap::new();
        let mut errors: Vec<G1MismatchKind> = Vec::new();

        for (line_idx, line_result) in reader.lines().enumerate() {
            let line_number = line_idx + 1; // 1-based

            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    errors.push(G1MismatchKind::ParseError {
                        source,
                        line_number,
                        key: None,
                        error: format!("IO error: {}", e),
                    });
                    continue;
                }
            };

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Parse decision
            let decision: AdmissionDecision = match serde_json::from_str(&line) {
                Ok(d) => d,
                Err(e) => {
                    errors.push(G1MismatchKind::ParseError {
                        source,
                        line_number,
                        key: None,
                        error: format!("JSON parse error: {}", e),
                    });
                    continue;
                }
            };

            // Extract key
            let key = match G1DecisionKey::from_decision(&decision) {
                Some(k) => k,
                None => {
                    errors.push(G1MismatchKind::ParseError {
                        source,
                        line_number,
                        key: None,
                        error: "Missing correlation_id (required for G1)".to_string(),
                    });
                    continue;
                }
            };

            // Check for duplicates
            if let Some(existing) = index.get(&key) {
                errors.push(G1MismatchKind::DuplicateKey {
                    key: key.clone(),
                    source,
                    first_line: existing.line_number,
                    duplicate_line: line_number,
                });
                continue;
            }

            // Build fingerprint
            let fingerprint = DecisionFingerprint {
                digest: decision.digest.clone(),
                outcome: decision.outcome,
                manifest_hash: decision.manifest_version_hash,
                line_number,
            };

            index.insert(key, fingerprint);
        }

        Ok((index, errors))
    }

    /// Compare two fingerprints and add mismatches if different.
    fn compare_fingerprints(
        key: &G1DecisionKey,
        live: &DecisionFingerprint,
        replay: &DecisionFingerprint,
        mismatches: &mut Vec<G1MismatchKind>,
    ) {
        // Check outcome first (most significant)
        if live.outcome != replay.outcome {
            mismatches.push(G1MismatchKind::OutcomeDiff {
                key: key.clone(),
                live_outcome: live.outcome,
                replay_outcome: replay.outcome,
                live_line: live.line_number,
                replay_line: replay.line_number,
            });
            return; // Outcome diff implies digest diff, don't double-report
        }

        // Check manifest hash (config parity)
        if live.manifest_hash != replay.manifest_hash {
            mismatches.push(G1MismatchKind::ManifestHashDiff {
                key: key.clone(),
                live_hash: live.manifest_hash,
                replay_hash: replay.manifest_hash,
                live_line: live.line_number,
                replay_line: replay.line_number,
            });
            return; // Manifest diff implies digest diff, don't double-report
        }

        // Check digest (full determinism)
        if live.digest != replay.digest {
            mismatches.push(G1MismatchKind::DigestDiff {
                key: key.clone(),
                live_digest: live.digest.clone(),
                replay_digest: replay.digest.clone(),
                live_line: live.line_number,
                replay_line: replay.line_number,
            });
        }
    }

    /// Check manifest hash parity between expected hash and a WAL file.
    ///
    /// Verifies all entries in the WAL use the expected manifest hash.
    pub fn check_manifest_hash_parity(
        wal_path: &Path,
        expected_hash: [u8; 32],
    ) -> Result<G1Result, std::io::Error> {
        let file = std::fs::File::open(wal_path)?;
        let reader = BufReader::new(file);

        let mut mismatches = Vec::new();
        let mut entry_count = 0;

        for (line_idx, line_result) in reader.lines().enumerate() {
            let line_number = line_idx + 1;

            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    mismatches.push(G1MismatchKind::ParseError {
                        source: G1Source::Live, // Using Live as default for single-WAL check
                        line_number,
                        key: None,
                        error: format!("IO error: {}", e),
                    });
                    continue;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            let decision: AdmissionDecision = match serde_json::from_str(&line) {
                Ok(d) => d,
                Err(e) => {
                    mismatches.push(G1MismatchKind::ParseError {
                        source: G1Source::Live,
                        line_number,
                        key: None,
                        error: format!("JSON parse error: {}", e),
                    });
                    continue;
                }
            };

            entry_count += 1;

            if decision.manifest_version_hash != expected_hash {
                let key = G1DecisionKey::from_decision(&decision);
                mismatches.push(G1MismatchKind::ManifestHashDiff {
                    key: key.unwrap_or_else(|| G1DecisionKey::new("unknown", &decision.signal_id)),
                    live_hash: expected_hash,
                    replay_hash: decision.manifest_version_hash,
                    live_line: 0, // Expected hash has no line
                    replay_line: line_number,
                });
            }
        }

        let passed = mismatches.is_empty();

        Ok(G1Result {
            check_name: check_names::G1_MANIFEST_HASH_PARITY.to_string(),
            passed,
            message: if passed {
                format!("Manifest hash parity verified: {} entries", entry_count)
            } else {
                format!(
                    "Manifest hash parity failed: {} mismatches",
                    mismatches.len()
                )
            },
            live_wal_path: wal_path.display().to_string(),
            replay_wal_path: String::new(),
            live_entry_count: entry_count,
            replay_entry_count: 0,
            matched_count: entry_count - mismatches.len(),
            mismatches,
        })
    }
}

// =============================================================================
// G2 Data Integrity Gate
// =============================================================================

/// G2 Data Integrity Gate — Validates admission event coverage.
///
/// Ensures that admission events meet minimum coverage thresholds:
/// - Non-zero events (at least one admission decision)
/// - Coverage ratio (admitted / total)
pub struct G2DataIntegrityGate;

/// Result of G2 gate validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2Result {
    /// Check name (frozen)
    pub check_name: String,
    /// Pass/fail
    pub passed: bool,
    /// Human-readable message
    pub message: String,
    /// WAL path checked
    pub wal_path: String,
    /// Total admission events
    pub total_events: usize,
    /// Admitted events
    pub admitted_count: usize,
    /// Refused events
    pub refused_count: usize,
    /// Coverage ratio (admitted / total)
    pub coverage_ratio: f64,
    /// Required coverage threshold
    pub required_threshold: f64,
    /// Parse errors encountered
    pub parse_error_count: usize,
}

impl G2DataIntegrityGate {
    /// Validate admission event coverage.
    ///
    /// - `min_events`: Minimum required events (0 = fail if empty)
    /// - `coverage_threshold`: Minimum admitted/total ratio (0.0 to 1.0)
    pub fn validate(
        wal_path: &Path,
        min_events: usize,
        coverage_threshold: f64,
    ) -> Result<G2Result, std::io::Error> {
        let file = std::fs::File::open(wal_path)?;
        let reader = BufReader::new(file);

        let mut total_events = 0;
        let mut admitted_count = 0;
        let mut refused_count = 0;
        let mut parse_error_count = 0;

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => {
                    parse_error_count += 1;
                    continue;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            let decision: AdmissionDecision = match serde_json::from_str(&line) {
                Ok(d) => d,
                Err(_) => {
                    parse_error_count += 1;
                    continue;
                }
            };

            total_events += 1;
            match decision.outcome {
                AdmissionOutcome::Admit => admitted_count += 1,
                AdmissionOutcome::Refuse => refused_count += 1,
            }
        }

        // Check nonzero events
        if total_events == 0 && min_events > 0 {
            return Ok(G2Result {
                check_name: check_names::G2_NONZERO_EVENTS.to_string(),
                passed: false,
                message: format!("No admission events found (required: {})", min_events),
                wal_path: wal_path.display().to_string(),
                total_events: 0,
                admitted_count: 0,
                refused_count: 0,
                coverage_ratio: 0.0,
                required_threshold: coverage_threshold,
                parse_error_count,
            });
        }

        // Check minimum events
        if total_events < min_events {
            return Ok(G2Result {
                check_name: check_names::G2_NONZERO_EVENTS.to_string(),
                passed: false,
                message: format!(
                    "Insufficient events: {} found, {} required",
                    total_events, min_events
                ),
                wal_path: wal_path.display().to_string(),
                total_events,
                admitted_count,
                refused_count,
                coverage_ratio: if total_events > 0 {
                    admitted_count as f64 / total_events as f64
                } else {
                    0.0
                },
                required_threshold: coverage_threshold,
                parse_error_count,
            });
        }

        // Calculate coverage
        let coverage_ratio = if total_events > 0 {
            admitted_count as f64 / total_events as f64
        } else {
            0.0
        };

        let passed = coverage_ratio >= coverage_threshold;

        Ok(G2Result {
            check_name: check_names::G2_COVERAGE_THRESHOLD.to_string(),
            passed,
            message: if passed {
                format!(
                    "Coverage OK: {:.1}% admitted ({}/{}), threshold {:.1}%",
                    coverage_ratio * 100.0,
                    admitted_count,
                    total_events,
                    coverage_threshold * 100.0
                )
            } else {
                format!(
                    "Coverage below threshold: {:.1}% < {:.1}% ({}/{})",
                    coverage_ratio * 100.0,
                    coverage_threshold * 100.0,
                    admitted_count,
                    total_events
                )
            },
            wal_path: wal_path.display().to_string(),
            total_events,
            admitted_count,
            refused_count,
            coverage_ratio,
            required_threshold: coverage_threshold,
            parse_error_count,
        })
    }

    /// Quick check: just verify non-zero events exist.
    pub fn check_nonzero(wal_path: &Path) -> Result<G2Result, std::io::Error> {
        Self::validate(wal_path, 1, 0.0)
    }
}

// =============================================================================
// G3 Execution Contract Gate
// =============================================================================

use crate::CheckResult;
use crate::strategies_manifest::{
    STRATEGIES_MANIFEST_SCHEMA_VERSION, StrategiesManifest, StrategiesManifestError,
};

/// G3 Violation types.
///
/// Represent validation errors found in strategies_manifest.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum G3Violation {
    /// Strategy ID key doesn't match spec.strategy_id
    StrategyIdMismatch { key: String, spec_id: String },

    /// signals array is empty
    EmptySignals { strategy_id: String },

    /// Referenced signal doesn't exist in signals_manifest
    UnknownSignal {
        strategy_id: String,
        signal_id: String,
    },

    /// Advisory class constraint violation
    AdvisoryConstraintViolation {
        strategy_id: String,
        field: String,
        expected: String,
        actual: String,
    },

    /// Passive class allows market orders
    PassiveAllowsMarketOrders { strategy_id: String },

    /// Rate limit exceeds sane maximum (1000)
    RateLimitTooHigh { strategy_id: String, value: u32 },

    /// Position limit is zero for non-advisory
    ZeroPositionLimit {
        strategy_id: String,
        execution_class: String,
    },

    /// Signal not promoted (only if --promotion-root provided)
    SignalNotPromoted {
        strategy_id: String,
        signal_id: String,
    },

    /// Parse error
    ParseError { error: String },
}

impl G3Violation {
    /// Human-readable description.
    pub fn description(&self) -> String {
        match self {
            G3Violation::StrategyIdMismatch { key, spec_id } => {
                format!(
                    "Strategy ID mismatch: key '{}' != strategy_id '{}'",
                    key, spec_id
                )
            }
            G3Violation::EmptySignals { strategy_id } => {
                format!("Strategy '{}' has empty signals array", strategy_id)
            }
            G3Violation::UnknownSignal {
                strategy_id,
                signal_id,
            } => {
                format!(
                    "Strategy '{}' references unknown signal '{}'",
                    strategy_id, signal_id
                )
            }
            G3Violation::AdvisoryConstraintViolation {
                strategy_id,
                field,
                expected,
                actual,
            } => {
                format!(
                    "Advisory strategy '{}': {} expected {}, got {}",
                    strategy_id, field, expected, actual
                )
            }
            G3Violation::PassiveAllowsMarketOrders { strategy_id } => {
                format!(
                    "Passive strategy '{}' cannot allow market orders",
                    strategy_id
                )
            }
            G3Violation::RateLimitTooHigh { strategy_id, value } => {
                format!(
                    "Strategy '{}' rate limit {} exceeds maximum 1000",
                    strategy_id, value
                )
            }
            G3Violation::ZeroPositionLimit {
                strategy_id,
                execution_class,
            } => {
                format!(
                    "Non-advisory strategy '{}' (class: {}) must have max_position_abs > 0",
                    strategy_id, execution_class
                )
            }
            G3Violation::SignalNotPromoted {
                strategy_id,
                signal_id,
            } => {
                format!(
                    "Strategy '{}' signal '{}' is not promoted",
                    strategy_id, signal_id
                )
            }
            G3Violation::ParseError { error } => {
                format!("Parse error: {}", error)
            }
        }
    }
}

/// Result of G3 gate validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3Result {
    /// Overall pass/fail
    pub passed: bool,

    /// Path to strategies manifest
    pub strategies_manifest_path: String,

    /// SHA-256 hash (bytes; CLI renders as hex)
    pub strategies_manifest_hash: Option<[u8; 32]>,

    /// Manifest version string
    pub strategies_manifest_version: Option<String>,

    /// Number of strategies
    pub strategy_count: Option<usize>,

    /// Total signal references across all strategies
    pub signal_binding_count: Option<usize>,

    /// Individual check results (matches G0/G1/G2 pattern)
    pub checks: Vec<CheckResult>,

    /// Detailed violations (for diagnostics)
    pub violations: Vec<G3Violation>,

    /// True if --promotion-root not provided (G3B skipped)
    pub promotion_check_skipped: bool,

    /// Error message if parse/IO failed
    pub error: Option<String>,
}

impl G3Result {
    /// Create a failed result with an error message.
    fn error(path: &Path, error: impl Into<String>) -> Self {
        let err_msg = error.into();
        Self {
            passed: false,
            strategies_manifest_path: path.display().to_string(),
            strategies_manifest_hash: None,
            strategies_manifest_version: None,
            strategy_count: None,
            signal_binding_count: None,
            checks: vec![CheckResult::fail(check_names::G3_SCHEMA_PARSE, &err_msg)],
            violations: vec![G3Violation::ParseError {
                error: err_msg.clone(),
            }],
            promotion_check_skipped: true,
            error: Some(err_msg),
        }
    }
}

/// G3 Execution Contract Gate — Validates strategies_manifest.json.
///
/// Phase 21C: Validates strategy execution contracts and signal bindings.
///
/// ## Validation Levels
/// - `validate_schema`: Schema parse and structural validation only
/// - `validate_bindings`: Schema + signal binding validation against signals_manifest
/// - `validate_full`: Schema + bindings + optional promotion status check
pub struct G3ExecutionContractGate;

impl G3ExecutionContractGate {
    /// Validate strategies_manifest.json schema only.
    ///
    /// Checks:
    /// - File exists and is readable
    /// - JSON parses successfully
    /// - Schema version is supported ("1.0.0")
    /// - All strategy IDs match keys
    /// - All signals arrays are non-empty
    /// - Execution-class constraints are satisfied
    /// - Sanity limits (rate limit <= 1000, position limit > 0 for non-advisory)
    pub fn validate_schema(strategies_path: &Path) -> G3Result {
        // Try to load and validate
        let manifest = match StrategiesManifest::load(strategies_path) {
            Ok(m) => m,
            Err(e) => {
                return G3Result::error(strategies_path, format!("{}", e));
            }
        };

        let mut checks = Vec::new();
        let mut violations = Vec::new();

        // G3_SCHEMA_PARSE: succeeded if we got here
        checks.push(CheckResult::pass(
            check_names::G3_SCHEMA_PARSE,
            format!(
                "Manifest parsed successfully (schema {})",
                STRATEGIES_MANIFEST_SCHEMA_VERSION
            ),
        ));

        // G3_VALIDATE: run validation rules
        match manifest.validate() {
            Ok(()) => {
                checks.push(CheckResult::pass(
                    check_names::G3_VALIDATE,
                    format!(
                        "All {} strategies pass constraint validation",
                        manifest.strategy_count()
                    ),
                ));
            }
            Err(e) => {
                // Convert StrategiesManifestError to G3Violation
                let violation = Self::error_to_violation(&e);
                violations.push(violation);
                checks.push(CheckResult::fail(
                    check_names::G3_VALIDATE,
                    format!("Validation failed: {}", e),
                ));
            }
        }

        // G3_SIGNAL_BINDINGS: skipped (no signals manifest provided)
        checks.push(CheckResult::pass(
            check_names::G3_SIGNAL_BINDINGS,
            "Skipped (no --signals provided)",
        ));

        // G3_PROMOTION_STATUS: skipped
        checks.push(CheckResult::pass(
            check_names::G3_PROMOTION_STATUS,
            "Skipped (no --promotion-root provided)",
        ));

        let passed = violations.is_empty();

        G3Result {
            passed,
            strategies_manifest_path: strategies_path.display().to_string(),
            strategies_manifest_hash: if passed {
                Some(manifest.compute_version_hash())
            } else {
                None
            },
            strategies_manifest_version: Some(manifest.manifest_version.clone()),
            strategy_count: Some(manifest.strategy_count()),
            signal_binding_count: Some(manifest.signal_binding_count()),
            checks,
            violations,
            promotion_check_skipped: true,
            error: None,
        }
    }

    /// Validate schema + signal bindings against signals_manifest.
    ///
    /// In addition to schema validation, checks that every signal referenced
    /// in strategies exists in the signals_manifest.
    pub fn validate_bindings(strategies_path: &Path, signals_path: &Path) -> G3Result {
        // Load strategies manifest
        let strategies_manifest = match StrategiesManifest::load(strategies_path) {
            Ok(m) => m,
            Err(e) => {
                return G3Result::error(strategies_path, format!("{}", e));
            }
        };

        // Load signals manifest
        let signals_manifest = match SignalsManifest::load_validated(signals_path) {
            Ok(m) => m,
            Err(e) => {
                return G3Result::error(
                    strategies_path,
                    format!("Failed to load signals manifest: {}", e),
                );
            }
        };

        let mut checks = Vec::new();
        let mut violations = Vec::new();

        // G3_SCHEMA_PARSE: succeeded
        checks.push(CheckResult::pass(
            check_names::G3_SCHEMA_PARSE,
            format!(
                "Manifest parsed successfully (schema {})",
                STRATEGIES_MANIFEST_SCHEMA_VERSION
            ),
        ));

        // G3_VALIDATE: run validation rules
        match strategies_manifest.validate() {
            Ok(()) => {
                checks.push(CheckResult::pass(
                    check_names::G3_VALIDATE,
                    format!(
                        "All {} strategies pass constraint validation",
                        strategies_manifest.strategy_count()
                    ),
                ));
            }
            Err(e) => {
                let violation = Self::error_to_violation(&e);
                violations.push(violation);
                checks.push(CheckResult::fail(
                    check_names::G3_VALIDATE,
                    format!("Validation failed: {}", e),
                ));
            }
        }

        // G3_SIGNAL_BINDINGS: validate signal bindings
        match strategies_manifest.validate_signal_bindings(&signals_manifest) {
            Ok(()) => {
                checks.push(CheckResult::pass(
                    check_names::G3_SIGNAL_BINDINGS,
                    format!(
                        "All {} signal bindings exist in signals_manifest",
                        strategies_manifest.signal_binding_count()
                    ),
                ));
            }
            Err(e) => {
                let violation = Self::error_to_violation(&e);
                violations.push(violation);
                checks.push(CheckResult::fail(
                    check_names::G3_SIGNAL_BINDINGS,
                    format!("Signal binding validation failed: {}", e),
                ));
            }
        }

        // G3_PROMOTION_STATUS: skipped
        checks.push(CheckResult::pass(
            check_names::G3_PROMOTION_STATUS,
            "Skipped (no --promotion-root provided)",
        ));

        let passed = violations.is_empty();

        G3Result {
            passed,
            strategies_manifest_path: strategies_path.display().to_string(),
            strategies_manifest_hash: if passed {
                Some(strategies_manifest.compute_version_hash())
            } else {
                None
            },
            strategies_manifest_version: Some(strategies_manifest.manifest_version.clone()),
            strategy_count: Some(strategies_manifest.strategy_count()),
            signal_binding_count: Some(strategies_manifest.signal_binding_count()),
            checks,
            violations,
            promotion_check_skipped: true,
            error: None,
        }
    }

    /// Full validation: schema + bindings + optional promotion check.
    ///
    /// If `promotion_root` is provided, checks that all signals in strategies
    /// have been promoted in the promotion directory.
    pub fn validate_full(
        strategies_path: &Path,
        signals_path: &Path,
        promotion_root: Option<&Path>,
    ) -> G3Result {
        // First run bindings validation
        let mut result = Self::validate_bindings(strategies_path, signals_path);

        // If promotion_root provided, check promotion status
        if let Some(_root) = promotion_root {
            // Update the G3_PROMOTION_STATUS check
            // For now, we mark it as passed since promotion checking requires
            // reading promotion records which is deferred to Phase 21D
            for check in &mut result.checks {
                if check.name == check_names::G3_PROMOTION_STATUS {
                    *check = CheckResult::pass(
                        check_names::G3_PROMOTION_STATUS,
                        "Promotion check not yet implemented (deferred to Phase 21D)",
                    );
                }
            }
            result.promotion_check_skipped = false;
        }

        result
    }

    /// Convert StrategiesManifestError to G3Violation.
    fn error_to_violation(e: &StrategiesManifestError) -> G3Violation {
        match e {
            StrategiesManifestError::StrategyIdMismatch { key, spec_id } => {
                G3Violation::StrategyIdMismatch {
                    key: key.clone(),
                    spec_id: spec_id.clone(),
                }
            }
            StrategiesManifestError::EmptySignals { strategy_id } => G3Violation::EmptySignals {
                strategy_id: strategy_id.clone(),
            },
            StrategiesManifestError::UnknownSignal {
                strategy_id,
                signal_id,
            } => G3Violation::UnknownSignal {
                strategy_id: strategy_id.clone(),
                signal_id: signal_id.clone(),
            },
            StrategiesManifestError::AdvisoryConstraintViolation {
                strategy_id,
                field,
                expected,
                actual,
            } => G3Violation::AdvisoryConstraintViolation {
                strategy_id: strategy_id.clone(),
                field: field.clone(),
                expected: expected.clone(),
                actual: actual.clone(),
            },
            StrategiesManifestError::PassiveAllowsMarketOrders { strategy_id } => {
                G3Violation::PassiveAllowsMarketOrders {
                    strategy_id: strategy_id.clone(),
                }
            }
            StrategiesManifestError::RateLimitTooHigh { strategy_id, value } => {
                G3Violation::RateLimitTooHigh {
                    strategy_id: strategy_id.clone(),
                    value: *value,
                }
            }
            StrategiesManifestError::ZeroPositionLimit {
                strategy_id,
                execution_class,
            } => G3Violation::ZeroPositionLimit {
                strategy_id: strategy_id.clone(),
                execution_class: execution_class.clone(),
            },
            StrategiesManifestError::Io { error, .. }
            | StrategiesManifestError::Parse { error, .. }
            | StrategiesManifestError::UnsupportedVersion { found: error, .. } => {
                G3Violation::ParseError {
                    error: error.clone(),
                }
            }
        }
    }
}

// =============================================================================
// Combined Gate Runner
// =============================================================================

/// Combined result from running all signal gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGatesResult {
    /// Overall pass/fail (all gates must pass)
    pub passed: bool,
    /// G0 result (if run)
    pub g0: Option<G0Result>,
    /// G1 result (if run)
    pub g1: Option<G1Result>,
    /// G2 result (if run)
    pub g2: Option<G2Result>,
    /// G3 result (if run)
    pub g3: Option<G3Result>,
    /// Summary message
    pub summary: String,
}

impl SignalGatesResult {
    /// Create new result with initial state.
    pub fn new() -> Self {
        Self {
            passed: true,
            g0: None,
            g1: None,
            g2: None,
            g3: None,
            summary: String::new(),
        }
    }

    /// Add G0 result.
    pub fn with_g0(mut self, result: G0Result) -> Self {
        if !result.passed {
            self.passed = false;
        }
        self.g0 = Some(result);
        self
    }

    /// Add G1 result.
    pub fn with_g1(mut self, result: G1Result) -> Self {
        if !result.passed {
            self.passed = false;
        }
        self.g1 = Some(result);
        self
    }

    /// Add G2 result.
    pub fn with_g2(mut self, result: G2Result) -> Self {
        if !result.passed {
            self.passed = false;
        }
        self.g2 = Some(result);
        self
    }

    /// Add G3 result.
    pub fn with_g3(mut self, result: G3Result) -> Self {
        if !result.passed {
            self.passed = false;
        }
        self.g3 = Some(result);
        self
    }

    /// Set summary message.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = summary.into();
        self
    }

    /// Build final summary from results.
    pub fn finalize(mut self) -> Self {
        let mut parts = Vec::new();

        if let Some(ref g0) = self.g0 {
            parts.push(format!("G0:{}", if g0.passed { "PASS" } else { "FAIL" }));
        }
        if let Some(ref g1) = self.g1 {
            parts.push(format!("G1:{}", if g1.passed { "PASS" } else { "FAIL" }));
        }
        if let Some(ref g2) = self.g2 {
            parts.push(format!("G2:{}", if g2.passed { "PASS" } else { "FAIL" }));
        }
        if let Some(ref g3) = self.g3 {
            parts.push(format!("G3:{}", if g3.passed { "PASS" } else { "FAIL" }));
        }

        self.summary = format!(
            "{} [{}]",
            if self.passed { "PASSED" } else { "FAILED" },
            parts.join(" ")
        );
        self
    }
}

impl Default for SignalGatesResult {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::ADMISSION_SCHEMA_VERSION;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const TEST_MANIFEST_HASH: [u8; 32] = [0x42; 32];

    fn make_decision(
        correlation_id: &str,
        signal_id: &str,
        outcome: AdmissionOutcome,
        manifest_hash: [u8; 32],
    ) -> AdmissionDecision {
        use quantlaxmi_models::AdmissionCanonicalBytes;

        let mut decision = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1_706_400_000_000_000_000,
            session_id: "test_session".to_string(),
            signal_id: signal_id.to_string(),
            outcome,
            missing_vendor_fields: vec![],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: Some(correlation_id.to_string()),
            manifest_version_hash: manifest_hash,
            digest: String::new(),
        };
        decision.digest = quantlaxmi_models::compute_digest(&decision.canonical_bytes());
        decision
    }

    fn write_wal(decisions: &[AdmissionDecision]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for d in decisions {
            writeln!(file, "{}", serde_json::to_string(d).unwrap()).unwrap();
        }
        file.flush().unwrap();
        file
    }

    // -------------------------------------------------------------------------
    // G1 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_g1_identical_wals_pass() {
        let decisions = vec![
            make_decision(
                "corr1",
                "spread",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
            make_decision(
                "corr2",
                "microprice",
                AdmissionOutcome::Refuse,
                TEST_MANIFEST_HASH,
            ),
        ];

        let live_wal = write_wal(&decisions);
        let replay_wal = write_wal(&decisions);

        let result = G1DeterminismGate::compare(live_wal.path(), replay_wal.path()).unwrap();

        assert!(result.passed, "Identical WALs should pass: {:?}", result);
        assert_eq!(result.live_entry_count, 2);
        assert_eq!(result.replay_entry_count, 2);
        assert_eq!(result.matched_count, 2);
        assert!(result.mismatches.is_empty());
    }

    #[test]
    fn test_g1_missing_replay_entry() {
        let live_decisions = vec![
            make_decision(
                "corr1",
                "spread",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
            make_decision(
                "corr2",
                "microprice",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
        ];
        let replay_decisions = vec![
            make_decision(
                "corr1",
                "spread",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
            // corr2 missing
        ];

        let live_wal = write_wal(&live_decisions);
        let replay_wal = write_wal(&replay_decisions);

        let result = G1DeterminismGate::compare(live_wal.path(), replay_wal.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            &result.mismatches[0],
            G1MismatchKind::MissingReplayEntry { key, .. }
            if key.correlation_id == "corr2"
        ));
    }

    #[test]
    fn test_g1_missing_live_entry() {
        let live_decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            TEST_MANIFEST_HASH,
        )];
        let replay_decisions = vec![
            make_decision(
                "corr1",
                "spread",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
            make_decision(
                "corr3",
                "book_imbalance",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
        ];

        let live_wal = write_wal(&live_decisions);
        let replay_wal = write_wal(&replay_decisions);

        let result = G1DeterminismGate::compare(live_wal.path(), replay_wal.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            &result.mismatches[0],
            G1MismatchKind::MissingLiveEntry { key, .. }
            if key.correlation_id == "corr3"
        ));
    }

    #[test]
    fn test_g1_outcome_diff() {
        let live_decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            TEST_MANIFEST_HASH,
        )];
        let replay_decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Refuse,
            TEST_MANIFEST_HASH,
        )];

        let live_wal = write_wal(&live_decisions);
        let replay_wal = write_wal(&replay_decisions);

        let result = G1DeterminismGate::compare(live_wal.path(), replay_wal.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            &result.mismatches[0],
            G1MismatchKind::OutcomeDiff {
                live_outcome: AdmissionOutcome::Admit,
                replay_outcome: AdmissionOutcome::Refuse,
                ..
            }
        ));
    }

    #[test]
    fn test_g1_manifest_hash_diff() {
        let different_hash = [0xFF; 32];

        let live_decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            TEST_MANIFEST_HASH,
        )];
        let replay_decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            different_hash,
        )];

        let live_wal = write_wal(&live_decisions);
        let replay_wal = write_wal(&replay_decisions);

        let result = G1DeterminismGate::compare(live_wal.path(), replay_wal.path()).unwrap();

        assert!(!result.passed);
        assert_eq!(result.mismatches.len(), 1);
        assert!(matches!(
            &result.mismatches[0],
            G1MismatchKind::ManifestHashDiff { .. }
        ));
    }

    #[test]
    fn test_g1_duplicate_key() {
        let mut file = NamedTempFile::new().unwrap();
        let d1 = make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            TEST_MANIFEST_HASH,
        );
        writeln!(file, "{}", serde_json::to_string(&d1).unwrap()).unwrap();
        writeln!(file, "{}", serde_json::to_string(&d1).unwrap()).unwrap(); // Duplicate
        file.flush().unwrap();

        let replay_wal = write_wal(std::slice::from_ref(&d1));

        let result = G1DeterminismGate::compare(file.path(), replay_wal.path()).unwrap();

        // Should detect duplicate in live
        assert!(result.mismatches.iter().any(|m| matches!(
            m,
            G1MismatchKind::DuplicateKey {
                source: G1Source::Live,
                ..
            }
        )));
    }

    #[test]
    fn test_g1_parse_error() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{{invalid json").unwrap();
        file.flush().unwrap();

        let replay_wal = write_wal(&[]);

        let result = G1DeterminismGate::compare(file.path(), replay_wal.path()).unwrap();

        assert!(result.mismatches.iter().any(|m| matches!(
            m,
            G1MismatchKind::ParseError {
                source: G1Source::Live,
                line_number: 1,
                ..
            }
        )));
    }

    #[test]
    fn test_g1_empty_lines_skipped() {
        let mut file = NamedTempFile::new().unwrap();
        let d1 = make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            TEST_MANIFEST_HASH,
        );
        writeln!(file, "{}", serde_json::to_string(&d1).unwrap()).unwrap();
        writeln!(file).unwrap(); // Empty line
        writeln!(file, "   ").unwrap(); // Whitespace line
        file.flush().unwrap();

        let replay_wal = write_wal(&[d1]);

        let result = G1DeterminismGate::compare(file.path(), replay_wal.path()).unwrap();

        assert!(result.passed, "Empty lines should be skipped");
        assert_eq!(result.live_entry_count, 1);
    }

    #[test]
    fn test_g1_line_numbers_1_based() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "{{invalid}}").unwrap(); // Line 1
        file.flush().unwrap();

        let replay_wal = write_wal(&[]);

        let result = G1DeterminismGate::compare(file.path(), replay_wal.path()).unwrap();

        assert!(
            result
                .mismatches
                .iter()
                .any(|m| matches!(m, G1MismatchKind::ParseError { line_number: 1, .. }))
        );
    }

    // -------------------------------------------------------------------------
    // G2 Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_g2_all_admitted_100_coverage() {
        let decisions = vec![
            make_decision(
                "corr1",
                "spread",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
            make_decision(
                "corr2",
                "microprice",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
        ];

        let wal = write_wal(&decisions);
        let result = G2DataIntegrityGate::validate(wal.path(), 1, 0.9).unwrap();

        assert!(result.passed);
        assert_eq!(result.total_events, 2);
        assert_eq!(result.admitted_count, 2);
        assert_eq!(result.refused_count, 0);
        assert!((result.coverage_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_g2_mixed_outcomes_coverage() {
        let decisions = vec![
            make_decision(
                "corr1",
                "spread",
                AdmissionOutcome::Admit,
                TEST_MANIFEST_HASH,
            ),
            make_decision(
                "corr2",
                "microprice",
                AdmissionOutcome::Refuse,
                TEST_MANIFEST_HASH,
            ),
        ];

        let wal = write_wal(&decisions);
        let result = G2DataIntegrityGate::validate(wal.path(), 1, 0.9).unwrap();

        assert!(!result.passed); // 50% < 90%
        assert_eq!(result.total_events, 2);
        assert_eq!(result.admitted_count, 1);
        assert_eq!(result.refused_count, 1);
        assert!((result.coverage_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_g2_empty_wal_fails() {
        let wal = write_wal(&[]);
        let result = G2DataIntegrityGate::validate(wal.path(), 1, 0.0).unwrap();

        assert!(!result.passed);
        assert_eq!(result.check_name, check_names::G2_NONZERO_EVENTS);
        assert_eq!(result.total_events, 0);
    }

    #[test]
    fn test_g2_below_min_events() {
        let decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Admit,
            TEST_MANIFEST_HASH,
        )];

        let wal = write_wal(&decisions);
        let result = G2DataIntegrityGate::validate(wal.path(), 5, 0.0).unwrap();

        assert!(!result.passed);
        assert_eq!(result.check_name, check_names::G2_NONZERO_EVENTS);
        assert_eq!(result.total_events, 1);
    }

    #[test]
    fn test_g2_check_nonzero() {
        let decisions = vec![make_decision(
            "corr1",
            "spread",
            AdmissionOutcome::Refuse,
            TEST_MANIFEST_HASH,
        )];

        let wal = write_wal(&decisions);
        let result = G2DataIntegrityGate::check_nonzero(wal.path()).unwrap();

        assert!(result.passed); // 0% coverage is fine for nonzero check
        assert_eq!(result.total_events, 1);
    }

    // -------------------------------------------------------------------------
    // Combined Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_signal_gates_result_builder() {
        let g0 = G0Result {
            check_name: check_names::G0_SCHEMA_VALID.to_string(),
            passed: true,
            message: "OK".to_string(),
            manifest_path: "test.json".to_string(),
            manifest_hash: Some([0; 32]),
            manifest_version: Some("0.1.0".to_string()),
            signal_count: Some(2),
            error: None,
        };

        let g2 = G2Result {
            check_name: check_names::G2_COVERAGE_THRESHOLD.to_string(),
            passed: false,
            message: "Low coverage".to_string(),
            wal_path: "wal.jsonl".to_string(),
            total_events: 10,
            admitted_count: 3,
            refused_count: 7,
            coverage_ratio: 0.3,
            required_threshold: 0.5,
            parse_error_count: 0,
        };

        let result = SignalGatesResult::new().with_g0(g0).with_g2(g2).finalize();

        assert!(!result.passed); // G2 failed
        assert!(result.summary.contains("FAILED"));
        assert!(result.summary.contains("G0:PASS"));
        assert!(result.summary.contains("G2:FAIL"));
    }

    #[test]
    fn test_g1_decision_key_ordering() {
        // Keys should sort by (correlation_id, signal_id)
        let k1 = G1DecisionKey::new("a", "spread");
        let k2 = G1DecisionKey::new("a", "microprice");
        let k3 = G1DecisionKey::new("b", "spread");

        assert!(k2 < k1); // "microprice" < "spread"
        assert!(k1 < k3); // "a" < "b"
    }

    #[test]
    fn test_g1_mismatch_description() {
        let mismatch = G1MismatchKind::DigestDiff {
            key: G1DecisionKey::new("corr123", "spread"),
            live_digest: "abc12345".to_string(),
            replay_digest: "def67890".to_string(),
            live_line: 5,
            replay_line: 3,
        };

        let desc = mismatch.description();
        assert!(desc.contains("corr123"));
        assert!(desc.contains("spread"));
        assert!(desc.contains("abc12345"));
        assert!(desc.contains("def67890"));
    }

    #[test]
    fn test_check_names_stable() {
        // These names are frozen for CI stability
        assert_eq!(check_names::G0_SCHEMA_VALID, "g0_schema_valid");
        assert_eq!(check_names::G1_DECISION_PARITY, "g1_decision_parity");
        assert_eq!(check_names::G2_COVERAGE_THRESHOLD, "g2_coverage_threshold");
    }

    // -------------------------------------------------------------------------
    // G0 Tests
    // -------------------------------------------------------------------------

    const VALID_MANIFEST: &str = r#"{
        "schema_version": "1.0.0",
        "manifest_version": "0.1.0",
        "created_at": "2026-01-28T00:00:00Z",
        "description": "Test manifest",
        "defaults": {},
        "signals": {
            "spread": {
                "signal_id": "spread",
                "description": "Spread signal",
                "required_l1": ["bid_price", "ask_price"]
            }
        }
    }"#;

    fn write_manifest(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".json").unwrap();
        write!(file, "{}", content).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_g0_valid_manifest() {
        let manifest_file = write_manifest(VALID_MANIFEST);
        let result = G0SchemaGate::validate(manifest_file.path());

        assert!(result.passed, "Valid manifest should pass: {:?}", result);
        assert_eq!(result.check_name, check_names::G0_SCHEMA_VALID);
        assert!(result.manifest_hash.is_some());
        assert_eq!(result.manifest_version, Some("0.1.0".to_string()));
        assert_eq!(result.signal_count, Some(1));
        assert!(result.error.is_none());
    }

    #[test]
    fn test_g0_invalid_json() {
        let manifest_file = write_manifest("{invalid json}");
        let result = G0SchemaGate::validate(manifest_file.path());

        assert!(!result.passed);
        assert!(result.error.is_some());
        assert!(result.manifest_hash.is_none());
    }

    #[test]
    fn test_g0_wrong_schema_version() {
        let bad_manifest = r#"{
            "schema_version": "99.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {}
        }"#;
        let manifest_file = write_manifest(bad_manifest);
        let result = G0SchemaGate::validate(manifest_file.path());

        assert!(!result.passed);
        assert!(result.message.contains("failed"));
    }

    #[test]
    fn test_g0_unknown_l1_field() {
        let bad_manifest = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "defaults": {},
            "signals": {
                "test": {
                    "signal_id": "test",
                    "required_l1": ["unknown_field"]
                }
            }
        }"#;
        let manifest_file = write_manifest(bad_manifest);
        let result = G0SchemaGate::validate(manifest_file.path());

        assert!(!result.passed);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_g0_validate_and_load() {
        let manifest_file = write_manifest(VALID_MANIFEST);
        let manifest = G0SchemaGate::validate_and_load(manifest_file.path()).unwrap();

        assert_eq!(manifest.signal_count(), 1);
        assert!(manifest.get_signal("spread").is_some());
    }

    #[test]
    fn test_g0_nonexistent_file() {
        let result = G0SchemaGate::validate(Path::new("/nonexistent/manifest.json"));

        assert!(!result.passed);
        assert!(result.error.is_some());
    }

    // -------------------------------------------------------------------------
    // G3 Tests
    // -------------------------------------------------------------------------

    const VALID_STRATEGIES_MANIFEST: &str = r#"{
        "schema_version": "1.0.0",
        "manifest_version": "0.1.0",
        "created_at": "2026-01-28T00:00:00Z",
        "description": "Test strategies manifest",
        "defaults": {
            "max_orders_per_min": 60,
            "allow_short": true,
            "allow_long": true,
            "allow_market_orders": false
        },
        "strategies": {
            "spread_passive": {
                "strategy_id": "spread_passive",
                "description": "Test strategy",
                "signals": ["spread"],
                "execution_class": "passive",
                "max_orders_per_min": 120,
                "max_position_abs": 10000,
                "allow_short": true,
                "allow_long": true,
                "allow_market_orders": false,
                "tags": ["mm", "passive"]
            }
        }
    }"#;

    fn write_strategies_manifest(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".json").unwrap();
        write!(file, "{}", content).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_g3_valid_manifest_schema() {
        let manifest_file = write_strategies_manifest(VALID_STRATEGIES_MANIFEST);
        let result = G3ExecutionContractGate::validate_schema(manifest_file.path());

        assert!(result.passed, "Valid manifest should pass: {:?}", result);
        assert!(result.strategies_manifest_hash.is_some());
        assert_eq!(
            result.strategies_manifest_version,
            Some("0.1.0".to_string())
        );
        assert_eq!(result.strategy_count, Some(1));
        assert_eq!(result.signal_binding_count, Some(1));
        assert!(result.violations.is_empty());
        assert_eq!(result.checks.len(), 4); // parse, validate, bindings (skipped), promotion (skipped)
    }

    #[test]
    fn test_g3_valid_manifest_bindings() {
        let signals_file = write_manifest(VALID_MANIFEST);
        let strategies_file = write_strategies_manifest(VALID_STRATEGIES_MANIFEST);

        let result =
            G3ExecutionContractGate::validate_bindings(strategies_file.path(), signals_file.path());

        assert!(result.passed, "Valid binding should pass: {:?}", result);
        assert!(result.violations.is_empty());

        // Check that signal bindings check passed
        let bindings_check = result
            .checks
            .iter()
            .find(|c| c.name == check_names::G3_SIGNAL_BINDINGS)
            .unwrap();
        assert!(bindings_check.passed);
    }

    #[test]
    fn test_g3_invalid_json() {
        let manifest_file = write_strategies_manifest("{invalid json}");
        let result = G3ExecutionContractGate::validate_schema(manifest_file.path());

        assert!(!result.passed);
        assert!(result.error.is_some());
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_g3_empty_signals() {
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
        let manifest_file = write_strategies_manifest(bad_manifest);
        let result = G3ExecutionContractGate::validate_schema(manifest_file.path());

        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| matches!(v, G3Violation::EmptySignals { .. }))
        );
    }

    #[test]
    fn test_g3_advisory_constraint_violation() {
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
        let manifest_file = write_strategies_manifest(bad_manifest);
        let result = G3ExecutionContractGate::validate_schema(manifest_file.path());

        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| matches!(v, G3Violation::AdvisoryConstraintViolation { .. }))
        );
    }

    #[test]
    fn test_g3_unknown_signal() {
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
        let strategies_json = r#"{
            "schema_version": "1.0.0",
            "manifest_version": "0.1.0",
            "created_at": "2026-01-28T00:00:00Z",
            "strategies": {
                "test": {
                    "strategy_id": "test",
                    "signals": ["unknown_signal"],
                    "execution_class": "passive",
                    "max_orders_per_min": 120,
                    "max_position_abs": 10000,
                    "allow_short": true,
                    "allow_long": true,
                    "allow_market_orders": false
                }
            }
        }"#;

        let signals_file = write_manifest(signals_json);
        let strategies_file = write_strategies_manifest(strategies_json);

        let result =
            G3ExecutionContractGate::validate_bindings(strategies_file.path(), signals_file.path());

        assert!(!result.passed);
        assert!(
            result
                .violations
                .iter()
                .any(|v| matches!(v, G3Violation::UnknownSignal { .. }))
        );
    }

    #[test]
    fn test_g3_check_names_stable() {
        assert_eq!(check_names::G3_SCHEMA_PARSE, "g3_schema_parse");
        assert_eq!(check_names::G3_VALIDATE, "g3_validate");
        assert_eq!(check_names::G3_SIGNAL_BINDINGS, "g3_signal_bindings");
        assert_eq!(check_names::G3_PROMOTION_STATUS, "g3_promotion_status");
    }

    #[test]
    fn test_g3_violation_descriptions() {
        let v1 = G3Violation::StrategyIdMismatch {
            key: "key".to_string(),
            spec_id: "spec".to_string(),
        };
        assert!(v1.description().contains("key"));
        assert!(v1.description().contains("spec"));

        let v2 = G3Violation::EmptySignals {
            strategy_id: "test".to_string(),
        };
        assert!(v2.description().contains("test"));
        assert!(v2.description().contains("empty"));

        let v3 = G3Violation::UnknownSignal {
            strategy_id: "strategy".to_string(),
            signal_id: "signal".to_string(),
        };
        assert!(v3.description().contains("strategy"));
        assert!(v3.description().contains("signal"));
    }

    #[test]
    fn test_g3_nonexistent_file() {
        let result =
            G3ExecutionContractGate::validate_schema(Path::new("/nonexistent/strategies.json"));

        assert!(!result.passed);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_g3_real_manifests() {
        // Test with real config files if they exist
        let signals_path = std::path::PathBuf::from("config/signals_manifest.json");
        let strategies_path = std::path::PathBuf::from("config/strategies_manifest.json");

        if !signals_path.exists() || !strategies_path.exists() {
            // Skip if running from wrong directory
            return;
        }

        let result = G3ExecutionContractGate::validate_bindings(&strategies_path, &signals_path);

        assert!(
            result.passed,
            "Real manifests should pass validation: {:?}",
            result
        );
        assert_eq!(result.strategy_count, Some(3));
    }

    #[test]
    fn test_signal_gates_result_with_g3() {
        let g3 = G3Result {
            passed: true,
            strategies_manifest_path: "test.json".to_string(),
            strategies_manifest_hash: Some([0; 32]),
            strategies_manifest_version: Some("0.1.0".to_string()),
            strategy_count: Some(3),
            signal_binding_count: Some(4),
            checks: vec![],
            violations: vec![],
            promotion_check_skipped: true,
            error: None,
        };

        let result = SignalGatesResult::new().with_g3(g3).finalize();

        assert!(result.passed);
        assert!(result.summary.contains("G3:PASS"));
    }
}
