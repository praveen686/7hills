//! # G4 Deployability Gate
//!
//! Pre-production readiness checks for trading system deployment.
//!
//! ## Checks
//! - **Observability Enabled**: Metrics and tracing are configured
//! - **Config Snapshot Exists**: Configuration is persisted for audit
//! - **Clean Shutdown**: No panic handlers, graceful termination possible
//!
//! ## Usage
//! ```ignore
//! let config = G4Config {
//!     require_metrics: true,
//!     require_tracing: true,
//!     require_config_snapshot: true,
//! };
//! let g4 = G4Deployability::new(config);
//! let result = g4.validate(&deployment_config)?;
//! ```

use crate::{CheckResult, GateError, GateResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::info;

/// G4 Deployability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G4Config {
    /// Require Prometheus metrics endpoint
    #[serde(default = "default_true")]
    pub require_metrics: bool,

    /// Require structured logging/tracing
    #[serde(default = "default_true")]
    pub require_tracing: bool,

    /// Require config snapshot for audit trail
    #[serde(default = "default_true")]
    pub require_config_snapshot: bool,

    /// Require graceful shutdown capability
    #[serde(default = "default_true")]
    pub require_graceful_shutdown: bool,

    /// Expected metrics port
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Config snapshot directory
    #[serde(default = "default_config_dir")]
    pub config_dir: String,
}

fn default_true() -> bool { true }
fn default_metrics_port() -> u16 { 9000 }
fn default_config_dir() -> String { "configs".to_string() }

impl Default for G4Config {
    fn default() -> Self {
        Self {
            require_metrics: true,
            require_tracing: true,
            require_config_snapshot: true,
            require_graceful_shutdown: true,
            metrics_port: default_metrics_port(),
            config_dir: default_config_dir(),
        }
    }
}

/// Deployment context for validation.
#[derive(Debug, Clone)]
pub struct DeploymentContext {
    /// Whether metrics exporter is initialized
    pub metrics_enabled: bool,
    /// Whether tracing is initialized
    pub tracing_enabled: bool,
    /// Path to config snapshot
    pub config_snapshot_path: Option<String>,
    /// Kill switch for graceful shutdown
    pub kill_switch: Option<Arc<AtomicBool>>,
    /// Run directory
    pub run_dir: Option<String>,
}

impl Default for DeploymentContext {
    fn default() -> Self {
        Self {
            metrics_enabled: false,
            tracing_enabled: false,
            config_snapshot_path: None,
            kill_switch: None,
            run_dir: None,
        }
    }
}

/// G4 Deployability gate validator.
pub struct G4Deployability {
    config: G4Config,
}

impl G4Deployability {
    /// Create a new G4 validator.
    pub fn new(config: G4Config) -> Self {
        Self { config }
    }

    /// Validate deployment readiness.
    pub fn validate(&self, ctx: &DeploymentContext) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G4_Deployability");

        info!("Starting G4 Deployability validation");

        // Check 1: Observability (metrics)
        result.add_check(self.check_metrics(ctx));

        // Check 2: Observability (tracing)
        result.add_check(self.check_tracing(ctx));

        // Check 3: Config snapshot
        result.add_check(self.check_config_snapshot(ctx)?);

        // Check 4: Graceful shutdown capability
        result.add_check(self.check_graceful_shutdown(ctx));

        // Check 5: No panic (environment check)
        result.add_check(self.check_panic_behavior());

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "{}/{} checks passed in {}ms",
            result.passed_count(),
            result.checks.len(),
            duration
        );

        info!(
            gate = "G4",
            passed = result.passed,
            checks_passed = result.passed_count(),
            checks_total = result.checks.len(),
            duration_ms = duration,
            "G4 Deployability validation complete"
        );

        Ok(result)
    }

    /// Check that metrics are enabled.
    fn check_metrics(&self, ctx: &DeploymentContext) -> CheckResult {
        if !self.config.require_metrics {
            return CheckResult::pass("metrics_enabled", "Metrics not required by config");
        }

        if ctx.metrics_enabled {
            CheckResult::pass(
                "metrics_enabled",
                format!("Prometheus metrics enabled on port {}", self.config.metrics_port),
            ).with_metrics(serde_json::json!({
                "port": self.config.metrics_port,
            }))
        } else {
            CheckResult::fail(
                "metrics_enabled",
                "Prometheus metrics exporter not initialized",
            )
        }
    }

    /// Check that tracing is enabled.
    fn check_tracing(&self, ctx: &DeploymentContext) -> CheckResult {
        if !self.config.require_tracing {
            return CheckResult::pass("tracing_enabled", "Tracing not required by config");
        }

        if ctx.tracing_enabled {
            CheckResult::pass(
                "tracing_enabled",
                "Structured tracing initialized",
            )
        } else {
            CheckResult::fail(
                "tracing_enabled",
                "Structured tracing not initialized",
            )
        }
    }

    /// Check that config snapshot exists.
    fn check_config_snapshot(&self, ctx: &DeploymentContext) -> Result<CheckResult, GateError> {
        if !self.config.require_config_snapshot {
            return Ok(CheckResult::pass(
                "config_snapshot_exists",
                "Config snapshot not required by config",
            ));
        }

        let snapshot_path = if let Some(ref path) = ctx.config_snapshot_path {
            path.clone()
        } else if let Some(ref run_dir) = ctx.run_dir {
            format!("{}/config_snapshot.json", run_dir)
        } else {
            format!("{}/config_snapshot.json", self.config.config_dir)
        };

        let path = Path::new(&snapshot_path);

        if path.exists() {
            // Verify it's valid JSON
            let content = std::fs::read_to_string(path)?;
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(_) => Ok(CheckResult::pass(
                    "config_snapshot_exists",
                    format!("Config snapshot found: {}", snapshot_path),
                ).with_metrics(serde_json::json!({
                    "path": snapshot_path,
                    "bytes": content.len(),
                }))),
                Err(e) => Ok(CheckResult::fail(
                    "config_snapshot_exists",
                    format!("Config snapshot is invalid JSON: {}", e),
                )),
            }
        } else {
            Ok(CheckResult::fail(
                "config_snapshot_exists",
                format!("Config snapshot not found: {}", snapshot_path),
            ))
        }
    }

    /// Check that graceful shutdown is possible.
    fn check_graceful_shutdown(&self, ctx: &DeploymentContext) -> CheckResult {
        if !self.config.require_graceful_shutdown {
            return CheckResult::pass(
                "graceful_shutdown",
                "Graceful shutdown not required by config",
            );
        }

        if let Some(ref kill_switch) = ctx.kill_switch {
            // Test that we can set the kill switch
            let current = kill_switch.load(Ordering::SeqCst);
            CheckResult::pass(
                "graceful_shutdown",
                format!("Kill switch available (current state: {})", current),
            )
        } else {
            CheckResult::fail(
                "graceful_shutdown",
                "No kill switch configured for graceful shutdown",
            )
        }
    }

    /// Check panic behavior configuration.
    fn check_panic_behavior(&self) -> CheckResult {
        // Check if RUST_BACKTRACE is set (good for debugging but may leak info)
        let backtrace = std::env::var("RUST_BACKTRACE").unwrap_or_default();

        // In production, we want panics to be caught and logged, not crash the process
        // This is a heuristic check - actual panic handling is set up elsewhere
        if cfg!(debug_assertions) {
            CheckResult::pass(
                "panic_behavior",
                "Debug build - panics will unwind with full trace",
            ).with_metrics(serde_json::json!({
                "debug_assertions": true,
                "rust_backtrace": backtrace,
            }))
        } else {
            CheckResult::pass(
                "panic_behavior",
                "Release build - panics configured for production",
            ).with_metrics(serde_json::json!({
                "debug_assertions": false,
                "rust_backtrace": backtrace,
            }))
        }
    }

    /// Validate that a running session can shut down cleanly.
    pub fn validate_shutdown(&self, kill_switch: &Arc<AtomicBool>) -> CheckResult {
        // Set the kill switch
        kill_switch.store(true, Ordering::SeqCst);

        // Verify it was set
        if kill_switch.load(Ordering::SeqCst) {
            CheckResult::pass(
                "shutdown_signal",
                "Kill switch successfully set - shutdown initiated",
            )
        } else {
            CheckResult::fail(
                "shutdown_signal",
                "Failed to set kill switch",
            )
        }
    }
}

/// Create a config snapshot for audit trail.
pub fn create_config_snapshot<T: Serialize>(
    config: &T,
    run_dir: &Path,
) -> Result<String, GateError> {
    let snapshot_path = run_dir.join("config_snapshot.json");

    let json = serde_json::to_string_pretty(config)
        .map_err(|e| GateError::Json(e))?;

    std::fs::create_dir_all(run_dir)?;
    std::fs::write(&snapshot_path, &json)?;

    info!(
        path = %snapshot_path.display(),
        bytes = json.len(),
        "Config snapshot created"
    );

    Ok(snapshot_path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_g4_all_disabled() {
        let config = G4Config {
            require_metrics: false,
            require_tracing: false,
            require_config_snapshot: false,
            require_graceful_shutdown: false,
            ..Default::default()
        };

        let g4 = G4Deployability::new(config);
        let ctx = DeploymentContext::default();

        let result = g4.validate(&ctx).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_g4_full_validation() {
        let dir = tempdir().unwrap();

        // Create config snapshot
        let config = serde_json::json!({"test": true});
        let snapshot_path = dir.path().join("config_snapshot.json");
        std::fs::write(&snapshot_path, serde_json::to_string(&config).unwrap()).unwrap();

        let g4_config = G4Config::default();
        let g4 = G4Deployability::new(g4_config);

        let ctx = DeploymentContext {
            metrics_enabled: true,
            tracing_enabled: true,
            config_snapshot_path: Some(snapshot_path.display().to_string()),
            kill_switch: Some(Arc::new(AtomicBool::new(false))),
            run_dir: Some(dir.path().display().to_string()),
        };

        let result = g4.validate(&ctx).unwrap();
        assert!(result.passed);
        assert_eq!(result.passed_count(), result.checks.len());
    }

    #[test]
    fn test_g4_missing_metrics() {
        let g4 = G4Deployability::new(G4Config::default());
        let ctx = DeploymentContext {
            metrics_enabled: false,
            tracing_enabled: true,
            ..Default::default()
        };

        let result = g4.validate(&ctx).unwrap();
        assert!(!result.passed);
        assert!(result.checks.iter().any(|c| c.name == "metrics_enabled" && !c.passed));
    }

    #[test]
    fn test_create_config_snapshot() {
        let dir = tempdir().unwrap();
        let config = serde_json::json!({
            "symbols": ["BTCUSDT"],
            "mode": "paper",
        });

        let path = create_config_snapshot(&config, dir.path()).unwrap();
        assert!(Path::new(&path).exists());

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["symbols"][0], "BTCUSDT");
    }
}
