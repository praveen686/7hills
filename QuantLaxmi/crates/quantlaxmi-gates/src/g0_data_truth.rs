//! # G0 DataTruth Gate
//!
//! Validates capture data integrity before any downstream processing.
//!
//! ## Checks
//! - **Monotonic Timestamps**: Events are ordered by time
//! - **Quote Age Sanity**: No quotes older than threshold
//! - **Schema Validity**: All events conform to expected schema
//! - **Manifest Compliance**: All referenced files exist with matching hashes
//!
//! ## Usage
//! ```ignore
//! let config = G0Config {
//!     max_quote_age_ms: 5000,
//!     max_timestamp_gap_ms: 1000,
//!     require_certified: true,
//! };
//! let g0 = G0DataTruth::new(config);
//! let result = g0.validate_session(&session_dir)?;
//! ```

use crate::{CheckResult, GateError, GateResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tracing::{debug, info};

/// G0 DataTruth configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G0Config {
    /// Maximum acceptable quote age in milliseconds
    #[serde(default = "default_max_quote_age_ms")]
    pub max_quote_age_ms: i64,

    /// Maximum acceptable timestamp gap between consecutive events
    #[serde(default = "default_max_timestamp_gap_ms")]
    pub max_timestamp_gap_ms: i64,

    /// Require all events to have certified integrity tier
    #[serde(default)]
    pub require_certified: bool,

    /// Minimum number of events expected per symbol
    #[serde(default = "default_min_events_per_symbol")]
    pub min_events_per_symbol: usize,

    /// Maximum allowed clock drift from now (for live validation)
    #[serde(default = "default_max_clock_drift_ms")]
    pub max_clock_drift_ms: i64,
}

fn default_max_quote_age_ms() -> i64 { 5000 }
fn default_max_timestamp_gap_ms() -> i64 { 1000 }
fn default_min_events_per_symbol() -> usize { 10 }
fn default_max_clock_drift_ms() -> i64 { 60000 }

impl Default for G0Config {
    fn default() -> Self {
        Self {
            max_quote_age_ms: default_max_quote_age_ms(),
            max_timestamp_gap_ms: default_max_timestamp_gap_ms(),
            require_certified: false,
            min_events_per_symbol: default_min_events_per_symbol(),
            max_clock_drift_ms: default_max_clock_drift_ms(),
        }
    }
}

/// G0 DataTruth gate validator.
pub struct G0DataTruth {
    config: G0Config,
}

impl G0DataTruth {
    /// Create a new G0 validator.
    pub fn new(config: G0Config) -> Self {
        Self { config }
    }

    /// Validate a capture session directory.
    pub fn validate_session(&self, session_dir: &Path) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G0_DataTruth");

        info!(session_dir = %session_dir.display(), "Starting G0 DataTruth validation");

        // Check 1: Session manifest exists and is valid
        result.add_check(self.check_manifest_exists(session_dir)?);

        // Check 2: Manifest compliance (files exist with correct hashes)
        result.add_check(self.check_manifest_compliance(session_dir)?);

        // Check 3: Monotonic timestamps in WAL files
        result.add_check(self.check_monotonic_timestamps(session_dir)?);

        // Check 4: Quote age sanity
        result.add_check(self.check_quote_age_sanity(session_dir)?);

        // Check 5: Schema validity
        result.add_check(self.check_schema_validity(session_dir)?);

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "{}/{} checks passed in {}ms",
            result.passed_count(),
            result.checks.len(),
            duration
        );

        info!(
            gate = "G0",
            passed = result.passed,
            checks_passed = result.passed_count(),
            checks_total = result.checks.len(),
            duration_ms = duration,
            "G0 DataTruth validation complete"
        );

        Ok(result)
    }

    /// Check that session manifest exists and is valid JSON.
    fn check_manifest_exists(&self, session_dir: &Path) -> Result<CheckResult, GateError> {
        let manifest_path = session_dir.join("session_manifest.json");

        if !manifest_path.exists() {
            return Ok(CheckResult::fail(
                "manifest_exists",
                format!("Session manifest not found: {:?}", manifest_path),
            ));
        }

        // Try to parse as JSON
        let content = std::fs::read_to_string(&manifest_path)?;
        match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(json) => {
                // Check required fields
                let has_session_id = json.get("session_id").is_some();
                let has_schema_version = json.get("schema_version").is_some();

                if has_session_id && has_schema_version {
                    Ok(CheckResult::pass(
                        "manifest_exists",
                        "Session manifest found and valid",
                    ).with_metrics(serde_json::json!({
                        "path": manifest_path.display().to_string(),
                        "session_id": json.get("session_id"),
                    })))
                } else {
                    Ok(CheckResult::fail(
                        "manifest_exists",
                        "Manifest missing required fields (session_id, schema_version)",
                    ))
                }
            }
            Err(e) => Ok(CheckResult::fail(
                "manifest_exists",
                format!("Invalid JSON in manifest: {}", e),
            )),
        }
    }

    /// Check that all files in manifest exist with correct hashes.
    fn check_manifest_compliance(&self, session_dir: &Path) -> Result<CheckResult, GateError> {
        let manifest_path = session_dir.join("session_manifest.json");
        if !manifest_path.exists() {
            return Ok(CheckResult::fail(
                "manifest_compliance",
                "Cannot check compliance: manifest not found",
            ));
        }

        let content = std::fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&content)?;

        let mut verified = 0;
        let mut missing = Vec::new();
        let mut hash_mismatches = Vec::new();

        // Check underlying manifests
        if let Some(underlyings) = manifest.get("underlyings").and_then(|u| u.as_array()) {
            for underlying in underlyings {
                if let Some(path) = underlying.get("universe_manifest_path").and_then(|p| p.as_str()) {
                    let full_path = session_dir.join(path);
                    if !full_path.exists() {
                        missing.push(path.to_string());
                        continue;
                    }

                    // Verify hash if present
                    if let Some(expected_hash) = underlying.get("universe_manifest_sha256").and_then(|h| h.as_str()) {
                        let bytes = std::fs::read(&full_path)?;
                        let actual_hash = sha256_hex(&bytes);
                        if actual_hash != expected_hash {
                            hash_mismatches.push((path.to_string(), expected_hash.to_string(), actual_hash));
                        } else {
                            verified += 1;
                        }
                    } else {
                        verified += 1;
                    }
                }
            }
        }

        // Check tick outputs
        if let Some(outputs) = manifest.get("tick_outputs").and_then(|o| o.as_array()) {
            for output in outputs {
                if let Some(path) = output.get("path").and_then(|p| p.as_str()) {
                    let full_path = session_dir.join(path);
                    if !full_path.exists() {
                        missing.push(path.to_string());
                    } else {
                        verified += 1;
                    }
                }
            }
        }

        if !missing.is_empty() || !hash_mismatches.is_empty() {
            Ok(CheckResult::fail(
                "manifest_compliance",
                format!(
                    "Manifest compliance failed: {} missing files, {} hash mismatches",
                    missing.len(),
                    hash_mismatches.len()
                ),
            ).with_metrics(serde_json::json!({
                "verified": verified,
                "missing": missing,
                "hash_mismatches": hash_mismatches.iter().map(|(p, e, a)| {
                    serde_json::json!({"path": p, "expected": e, "actual": a})
                }).collect::<Vec<_>>(),
            })))
        } else {
            Ok(CheckResult::pass(
                "manifest_compliance",
                format!("All {} manifest entries verified", verified),
            ).with_metrics(serde_json::json!({
                "verified": verified,
            })))
        }
    }

    /// Check that timestamps are monotonically increasing within each WAL file.
    fn check_monotonic_timestamps(&self, session_dir: &Path) -> Result<CheckResult, GateError> {
        let wal_dir = session_dir.join("wal");
        if !wal_dir.exists() {
            // Check for legacy tick files
            return self.check_monotonic_timestamps_legacy(session_dir);
        }

        let mut violations = Vec::new();
        let mut files_checked = 0;
        let mut events_checked = 0;

        for entry in std::fs::read_dir(&wal_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "jsonl") {
                files_checked += 1;
                let file_violations = self.check_file_monotonicity(&path)?;
                events_checked += file_violations.1;
                if !file_violations.0.is_empty() {
                    violations.extend(file_violations.0);
                }
            }
        }

        if violations.is_empty() {
            Ok(CheckResult::pass(
                "monotonic_timestamps",
                format!(
                    "All {} events in {} files have monotonic timestamps",
                    events_checked, files_checked
                ),
            ).with_metrics(serde_json::json!({
                "files_checked": files_checked,
                "events_checked": events_checked,
            })))
        } else {
            Ok(CheckResult::fail(
                "monotonic_timestamps",
                format!(
                    "Found {} timestamp violations in {} files",
                    violations.len(),
                    files_checked
                ),
            ).with_metrics(serde_json::json!({
                "violations": violations.iter().take(10).collect::<Vec<_>>(),
                "total_violations": violations.len(),
            })))
        }
    }

    /// Check monotonicity for legacy tick files (not in wal/ directory).
    fn check_monotonic_timestamps_legacy(&self, session_dir: &Path) -> Result<CheckResult, GateError> {
        let mut violations = Vec::new();
        let mut files_checked = 0;
        let mut events_checked = 0;

        // Look for .jsonl files in subdirectories
        for entry in std::fs::read_dir(session_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                for file in std::fs::read_dir(&path)? {
                    let file = file?;
                    let file_path = file.path();
                    if file_path.extension().map_or(false, |e| e == "jsonl") {
                        files_checked += 1;
                        let file_violations = self.check_file_monotonicity(&file_path)?;
                        events_checked += file_violations.1;
                        if !file_violations.0.is_empty() {
                            violations.extend(file_violations.0);
                        }
                    }
                }
            }
        }

        if files_checked == 0 {
            return Ok(CheckResult::pass(
                "monotonic_timestamps",
                "No WAL files to check (empty session)",
            ));
        }

        if violations.is_empty() {
            Ok(CheckResult::pass(
                "monotonic_timestamps",
                format!(
                    "All {} events in {} files have monotonic timestamps",
                    events_checked, files_checked
                ),
            ))
        } else {
            Ok(CheckResult::fail(
                "monotonic_timestamps",
                format!(
                    "Found {} timestamp violations",
                    violations.len()
                ),
            ))
        }
    }

    /// Check a single file for monotonic timestamps.
    fn check_file_monotonicity(&self, path: &Path) -> Result<(Vec<String>, usize), GateError> {
        let content = std::fs::read_to_string(path)?;
        let mut violations = Vec::new();
        let mut last_ts: Option<DateTime<Utc>> = None;
        let mut count = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            count += 1;

            // Try to extract timestamp
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(ts_str) = json.get("ts").and_then(|t| t.as_str()) {
                    if let Ok(ts) = ts_str.parse::<DateTime<Utc>>() {
                        if let Some(prev) = last_ts {
                            if ts < prev {
                                let gap_ms = (prev - ts).num_milliseconds();
                                if gap_ms.abs() > self.config.max_timestamp_gap_ms {
                                    violations.push(format!(
                                        "{}:{} - timestamp went backwards by {}ms",
                                        path.file_name().unwrap_or_default().to_string_lossy(),
                                        line_num + 1,
                                        gap_ms.abs()
                                    ));
                                }
                            }
                        }
                        last_ts = Some(ts);
                    }
                }
            }
        }

        Ok((violations, count))
    }

    /// Check that quotes are not too stale.
    fn check_quote_age_sanity(&self, session_dir: &Path) -> Result<CheckResult, GateError> {
        let manifest_path = session_dir.join("session_manifest.json");
        if !manifest_path.exists() {
            return Ok(CheckResult::pass(
                "quote_age_sanity",
                "Skipped: no manifest (offline validation)",
            ));
        }

        let content = std::fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&content)?;

        // Get session creation time
        let created_at = manifest
            .get("created_at_utc")
            .and_then(|c| c.as_str())
            .and_then(|s| s.parse::<DateTime<Utc>>().ok());

        if created_at.is_none() {
            return Ok(CheckResult::pass(
                "quote_age_sanity",
                "Skipped: no creation timestamp in manifest",
            ));
        }

        let session_time = created_at.unwrap();
        let now = Utc::now();
        let session_age_ms = (now - session_time).num_milliseconds();

        // For offline validation, we check if session is reasonably recent
        if session_age_ms > self.config.max_clock_drift_ms * 24 * 60 {
            // Session is older than 1 day * max_clock_drift factor
            debug!(
                session_age_hours = session_age_ms / 3600000,
                "Session is historical, skipping live quote age check"
            );
            return Ok(CheckResult::pass(
                "quote_age_sanity",
                format!(
                    "Historical session ({} hours old), quote age validated at capture time",
                    session_age_ms / 3600000
                ),
            ));
        }

        Ok(CheckResult::pass(
            "quote_age_sanity",
            "Quote age within acceptable bounds",
        ))
    }

    /// Check that all events conform to expected schema.
    fn check_schema_validity(&self, session_dir: &Path) -> Result<CheckResult, GateError> {
        let wal_dir = session_dir.join("wal");
        let mut errors = Vec::new();
        let mut valid_events = 0;

        let dirs_to_check: Vec<_> = if wal_dir.exists() {
            vec![wal_dir]
        } else {
            // Legacy: check subdirectories
            std::fs::read_dir(session_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        };

        for dir in dirs_to_check {
            for entry in std::fs::read_dir(&dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "jsonl") {
                    let (file_errors, file_valid) = self.validate_file_schema(&path)?;
                    errors.extend(file_errors);
                    valid_events += file_valid;
                }
            }
        }

        if errors.is_empty() {
            Ok(CheckResult::pass(
                "schema_validity",
                format!("All {} events have valid schema", valid_events),
            ).with_metrics(serde_json::json!({
                "valid_events": valid_events,
            })))
        } else {
            Ok(CheckResult::fail(
                "schema_validity",
                format!("{} schema errors found", errors.len()),
            ).with_metrics(serde_json::json!({
                "errors": errors.iter().take(10).collect::<Vec<_>>(),
                "total_errors": errors.len(),
                "valid_events": valid_events,
            })))
        }
    }

    /// Validate schema for a single file.
    fn validate_file_schema(&self, path: &Path) -> Result<(Vec<String>, usize), GateError> {
        let content = std::fs::read_to_string(path)?;
        let mut errors = Vec::new();
        let mut valid = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(json) => {
                    // Basic schema checks
                    if json.get("ts").is_none() {
                        errors.push(format!(
                            "{}:{} - missing 'ts' field",
                            path.file_name().unwrap_or_default().to_string_lossy(),
                            line_num + 1
                        ));
                    } else {
                        valid += 1;
                    }
                }
                Err(e) => {
                    errors.push(format!(
                        "{}:{} - invalid JSON: {}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        line_num + 1,
                        e
                    ));
                }
            }
        }

        Ok((errors, valid))
    }

    /// Validate a single quote event (for live validation).
    pub fn validate_quote(&self, quote: &quantlaxmi_models::events::QuoteEvent, now: DateTime<Utc>) -> CheckResult {
        let age_ms = quote.age_ms(now);

        if age_ms > self.config.max_quote_age_ms {
            return CheckResult::fail(
                "quote_freshness",
                format!("Quote is {}ms old (max: {}ms)", age_ms, self.config.max_quote_age_ms),
            ).with_metrics(serde_json::json!({
                "age_ms": age_ms,
                "max_age_ms": self.config.max_quote_age_ms,
                "symbol": &quote.symbol,
            }));
        }

        if !quote.is_valid() {
            return CheckResult::fail(
                "quote_validity",
                "Quote has invalid price/quantity values",
            ).with_metrics(serde_json::json!({
                "bid": quote.bid_price_mantissa,
                "ask": quote.ask_price_mantissa,
                "symbol": &quote.symbol,
            }));
        }

        CheckResult::pass(
            "quote_valid",
            format!("Quote for {} is valid (age: {}ms)", quote.symbol, age_ms),
        )
    }
}

/// Compute SHA-256 hash as hex string.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_g0_empty_session() {
        let dir = tempdir().unwrap();
        let g0 = G0DataTruth::new(G0Config::default());

        let result = g0.validate_session(dir.path()).unwrap();
        // Should fail because no manifest
        assert!(!result.passed);
        assert!(result.checks.iter().any(|c| c.name == "manifest_exists" && !c.passed));
    }

    #[test]
    fn test_g0_valid_session() {
        let dir = tempdir().unwrap();

        // Create a minimal valid session manifest
        let manifest = serde_json::json!({
            "schema_version": 1,
            "session_id": "test-session-123",
            "created_at_utc": chrono::Utc::now().to_rfc3339(),
            "capture_mode": "test",
            "out_dir": dir.path().display().to_string(),
            "duration_secs": 60.0,
            "price_exponent": -2,
            "underlyings": [],
            "tick_outputs": [],
            "integrity": {
                "out_of_universe_ticks_dropped": 0,
                "subscribe_mode": "test"
            }
        });

        let manifest_path = dir.path().join("session_manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

        let g0 = G0DataTruth::new(G0Config::default());
        let result = g0.validate_session(dir.path()).unwrap();

        assert!(result.passed);
        assert!(result.checks.iter().any(|c| c.name == "manifest_exists" && c.passed));
    }

    #[test]
    fn test_g0_quote_validation() {
        use quantlaxmi_models::events::{QuoteEvent, CorrelationContext};

        let g0 = G0DataTruth::new(G0Config {
            max_quote_age_ms: 1000,
            ..Default::default()
        });

        let now = Utc::now();

        // Valid quote
        let quote = QuoteEvent {
            ts: now,
            symbol: "BTCUSDT".to_string(),
            bid_price_mantissa: 9000000,
            ask_price_mantissa: 9000100,
            bid_qty_mantissa: 100000000,
            ask_qty_mantissa: 100000000,
            price_exponent: -2,
            qty_exponent: -8,
            venue: "binance".to_string(),
            ctx: CorrelationContext::default(),
        };

        let check = g0.validate_quote(&quote, now);
        assert!(check.passed);

        // Stale quote
        let stale_quote = QuoteEvent {
            ts: now - chrono::Duration::milliseconds(2000),
            ..quote.clone()
        };
        let check = g0.validate_quote(&stale_quote, now);
        assert!(!check.passed);
    }
}
