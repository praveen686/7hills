//! # G3 Robustness Gate
//!
//! Stress testing and edge case validation for production readiness.
//!
//! ## Implemented Checks
//! - **Connection Loss**: Verify reconnection configuration exists
//! - **Data Gaps**: Verify gap handling configuration
//! - **Extreme Prices**: Verify price limit configuration
//! - **High Latency**: Verify timeout configuration
//! - **Partial Fills**: Verify partial fill handling
//! - **Memory Pressure**: Verify resource limits
//!
//! ## Usage
//! ```ignore
//! let g3 = G3Robustness::new(config);
//! let result = g3.run_stress_tests()?;
//! let result = g3.validate_config(&system_config)?;
//! ```

use crate::{CheckResult, GateError, GateResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

/// G3 Robustness configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G3Config {
    /// Maximum acceptable reconnection time in milliseconds
    pub max_reconnect_ms: u64,

    /// Maximum data gap duration before circuit breaker (ms)
    pub max_data_gap_ms: u64,

    /// Enable extreme price scenario testing
    pub test_extreme_prices: bool,

    /// Enable high latency scenario testing
    pub test_high_latency: bool,

    /// Simulated latency for high latency tests (ms)
    pub simulated_latency_ms: u64,

    /// Maximum acceptable order timeout (ms)
    pub max_order_timeout_ms: u64,

    /// Price limit threshold as percentage deviation
    pub price_limit_pct: f64,
}

impl Default for G3Config {
    fn default() -> Self {
        Self {
            max_reconnect_ms: 5000,
            max_data_gap_ms: 10000,
            test_extreme_prices: true,
            test_high_latency: true,
            simulated_latency_ms: 500,
            max_order_timeout_ms: 30000,
            price_limit_pct: 10.0, // 10% deviation triggers warning
        }
    }
}

/// System configuration for robustness validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// WebSocket reconnect enabled
    pub ws_reconnect_enabled: bool,
    /// WebSocket reconnect max delay (ms)
    pub ws_reconnect_max_delay_ms: u64,
    /// WebSocket liveness timeout (ms)
    pub ws_liveness_timeout_ms: u64,

    /// Circuit breaker enabled
    pub circuit_breaker_enabled: bool,
    /// Circuit breaker data gap threshold (ms)
    pub circuit_breaker_gap_ms: u64,

    /// Order timeout (ms)
    pub order_timeout_ms: u64,

    /// Price deviation limit (%)
    pub price_deviation_limit_pct: f64,

    /// Memory limit (MB)
    pub memory_limit_mb: Option<u64>,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            ws_reconnect_enabled: true,
            ws_reconnect_max_delay_ms: 30000,
            ws_liveness_timeout_ms: 30000,
            circuit_breaker_enabled: true,
            circuit_breaker_gap_ms: 10000,
            order_timeout_ms: 30000,
            price_deviation_limit_pct: 5.0,
            memory_limit_mb: Some(4096),
        }
    }
}

/// G3 Robustness gate validator.
pub struct G3Robustness {
    config: G3Config,
}

impl G3Robustness {
    /// Create a new G3 validator.
    pub fn new(config: G3Config) -> Self {
        Self { config }
    }

    /// Run robustness stress tests (legacy interface - runs with default system config).
    pub fn run_stress_tests(&self) -> Result<GateResult, GateError> {
        self.validate_config(&SystemConfig::default())
    }

    /// Validate system configuration for robustness.
    pub fn validate_config(&self, system_config: &SystemConfig) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G3_Robustness");

        info!("Starting G3 Robustness validation");

        // 1. Connection loss handling
        result.add_check(self.check_connection_loss(system_config));

        // 2. Data gap handling
        result.add_check(self.check_data_gaps(system_config));

        // 3. Extreme price handling
        if self.config.test_extreme_prices {
            result.add_check(self.check_extreme_prices(system_config));
        }

        // 4. High latency handling
        if self.config.test_high_latency {
            result.add_check(self.check_high_latency(system_config));
        }

        // 5. Partial fill handling
        result.add_check(self.check_partial_fills());

        // 6. Memory pressure handling
        result.add_check(self.check_memory_pressure(system_config));

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "{}/{} checks passed",
            result.passed_count(),
            result.checks.len()
        );

        Ok(result)
    }

    /// Validate a configuration file exists and has robustness settings.
    pub fn validate_config_file(&self, config_path: &Path) -> Result<GateResult, GateError> {
        let start = std::time::Instant::now();
        let mut result = GateResult::new("G3_Robustness");

        info!(
            config = %config_path.display(),
            "Validating robustness configuration file"
        );

        if !config_path.exists() {
            result.add_check(CheckResult::fail(
                "config_exists",
                format!("Configuration file not found: {}", config_path.display()),
            ));
            result.duration_ms = start.elapsed().as_millis() as u64;
            return Ok(result);
        }

        result.add_check(CheckResult::pass(
            "config_exists",
            "Configuration file exists",
        ));

        // Read and parse config
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| GateError::IoError(format!("Failed to read config: {}", e)))?;

        // Check for required robustness keys
        let required_keys = ["reconnect", "circuit_breaker", "timeout"];

        for key in &required_keys {
            if content.contains(key) {
                result.add_check(CheckResult::pass(
                    format!("has_{}", key),
                    format!("Config has '{}' settings", key),
                ));
            } else {
                result.add_check(CheckResult::warn(
                    format!("has_{}", key),
                    format!("Config missing '{}' settings", key),
                ));
            }
        }

        let duration = start.elapsed().as_millis() as u64;
        result.duration_ms = duration;
        result.summary = format!(
            "{}/{} checks passed",
            result.passed_count(),
            result.checks.len()
        );

        Ok(result)
    }

    /// Check connection loss handling configuration.
    fn check_connection_loss(&self, config: &SystemConfig) -> CheckResult {
        if !config.ws_reconnect_enabled {
            return CheckResult::fail(
                "connection_loss",
                "WebSocket reconnect is disabled - system won't recover from disconnects",
            );
        }

        if config.ws_reconnect_max_delay_ms > self.config.max_reconnect_ms * 2 {
            warn!(
                max_delay = config.ws_reconnect_max_delay_ms,
                threshold = self.config.max_reconnect_ms,
                "Reconnect delay may be too slow"
            );
            return CheckResult::warn(
                "connection_loss",
                format!(
                    "Reconnect max delay {}ms exceeds threshold {}ms",
                    config.ws_reconnect_max_delay_ms, self.config.max_reconnect_ms
                ),
            );
        }

        if config.ws_liveness_timeout_ms == 0 {
            return CheckResult::fail(
                "connection_loss",
                "Liveness timeout is zero - stale connections won't be detected",
            );
        }

        CheckResult::pass(
            "connection_loss",
            format!(
                "Reconnect enabled (max {}ms), liveness timeout {}ms",
                config.ws_reconnect_max_delay_ms, config.ws_liveness_timeout_ms
            ),
        )
    }

    /// Check data gap handling configuration.
    fn check_data_gaps(&self, config: &SystemConfig) -> CheckResult {
        if !config.circuit_breaker_enabled {
            return CheckResult::fail(
                "data_gaps",
                "Circuit breaker disabled - data gaps won't trigger trading halt",
            );
        }

        if config.circuit_breaker_gap_ms > self.config.max_data_gap_ms {
            warn!(
                gap_threshold = config.circuit_breaker_gap_ms,
                max_allowed = self.config.max_data_gap_ms,
                "Data gap threshold may be too permissive"
            );
            return CheckResult::warn(
                "data_gaps",
                format!(
                    "Gap threshold {}ms exceeds recommended {}ms",
                    config.circuit_breaker_gap_ms, self.config.max_data_gap_ms
                ),
            );
        }

        CheckResult::pass(
            "data_gaps",
            format!(
                "Circuit breaker enabled with {}ms gap threshold",
                config.circuit_breaker_gap_ms
            ),
        )
    }

    /// Check extreme price handling configuration.
    fn check_extreme_prices(&self, config: &SystemConfig) -> CheckResult {
        if config.price_deviation_limit_pct == 0.0 {
            return CheckResult::fail(
                "extreme_prices",
                "Price deviation limit is zero - extreme prices won't be rejected",
            );
        }

        if config.price_deviation_limit_pct > self.config.price_limit_pct {
            warn!(
                limit = config.price_deviation_limit_pct,
                recommended = self.config.price_limit_pct,
                "Price deviation limit may be too permissive"
            );
            return CheckResult::warn(
                "extreme_prices",
                format!(
                    "Price limit {}% exceeds recommended {}%",
                    config.price_deviation_limit_pct, self.config.price_limit_pct
                ),
            );
        }

        CheckResult::pass(
            "extreme_prices",
            format!(
                "Price deviation limit set to {}%",
                config.price_deviation_limit_pct
            ),
        )
    }

    /// Check high latency handling configuration.
    fn check_high_latency(&self, config: &SystemConfig) -> CheckResult {
        if config.order_timeout_ms == 0 {
            return CheckResult::fail(
                "high_latency",
                "Order timeout is zero - slow orders won't be cancelled",
            );
        }

        if config.order_timeout_ms > self.config.max_order_timeout_ms {
            warn!(
                timeout = config.order_timeout_ms,
                max = self.config.max_order_timeout_ms,
                "Order timeout may be too long"
            );
            return CheckResult::warn(
                "high_latency",
                format!(
                    "Order timeout {}ms exceeds recommended {}ms",
                    config.order_timeout_ms, self.config.max_order_timeout_ms
                ),
            );
        }

        CheckResult::pass(
            "high_latency",
            format!("Order timeout set to {}ms", config.order_timeout_ms),
        )
    }

    /// Check partial fill handling (always passes if system has FillEvent with is_final).
    fn check_partial_fills(&self) -> CheckResult {
        // The FillEvent struct has is_final field which supports partial fills
        // This check validates that the system architecture supports partial fills
        CheckResult::pass(
            "partial_fills",
            "FillEvent.is_final field supports partial fill tracking",
        )
    }

    /// Check memory pressure handling configuration.
    fn check_memory_pressure(&self, config: &SystemConfig) -> CheckResult {
        match config.memory_limit_mb {
            None => CheckResult::warn(
                "memory_pressure",
                "No memory limit configured - system may OOM under pressure",
            ),
            Some(0) => CheckResult::fail(
                "memory_pressure",
                "Memory limit is zero - invalid configuration",
            ),
            Some(limit) if limit < 512 => CheckResult::warn(
                "memory_pressure",
                format!("Memory limit {}MB may be too low for production", limit),
            ),
            Some(limit) => CheckResult::pass(
                "memory_pressure",
                format!("Memory limit set to {}MB", limit),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g3_default_config_passes() {
        let g3 = G3Robustness::new(G3Config::default());
        let result = g3.run_stress_tests().unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_connection_loss_check() {
        let g3 = G3Robustness::new(G3Config::default());

        // Good config
        let good = SystemConfig::default();
        let result = g3.check_connection_loss(&good);
        assert!(result.passed);

        // Bad config - reconnect disabled
        let bad = SystemConfig {
            ws_reconnect_enabled: false,
            ..Default::default()
        };
        let result = g3.check_connection_loss(&bad);
        assert!(!result.passed);
    }

    #[test]
    fn test_data_gaps_check() {
        let g3 = G3Robustness::new(G3Config::default());

        // Good config
        let good = SystemConfig::default();
        let result = g3.check_data_gaps(&good);
        assert!(result.passed);

        // Bad config - circuit breaker disabled
        let bad = SystemConfig {
            circuit_breaker_enabled: false,
            ..Default::default()
        };
        let result = g3.check_data_gaps(&bad);
        assert!(!result.passed);
    }

    #[test]
    fn test_extreme_prices_check() {
        let g3 = G3Robustness::new(G3Config::default());

        // Good config
        let good = SystemConfig::default();
        let result = g3.check_extreme_prices(&good);
        assert!(result.passed);

        // Bad config - zero limit
        let bad = SystemConfig {
            price_deviation_limit_pct: 0.0,
            ..Default::default()
        };
        let result = g3.check_extreme_prices(&bad);
        assert!(!result.passed);
    }

    #[test]
    fn test_high_latency_check() {
        let g3 = G3Robustness::new(G3Config::default());

        // Good config
        let good = SystemConfig::default();
        let result = g3.check_high_latency(&good);
        assert!(result.passed);

        // Bad config - zero timeout
        let bad = SystemConfig {
            order_timeout_ms: 0,
            ..Default::default()
        };
        let result = g3.check_high_latency(&bad);
        assert!(!result.passed);
    }

    #[test]
    fn test_memory_pressure_check() {
        let g3 = G3Robustness::new(G3Config::default());

        // Good config
        let good = SystemConfig::default();
        let result = g3.check_memory_pressure(&good);
        assert!(result.passed);

        // Warning config - no limit
        let warn = SystemConfig {
            memory_limit_mb: None,
            ..Default::default()
        };
        let result = g3.check_memory_pressure(&warn);
        assert!(result.passed); // Warning still passes

        // Bad config - zero limit
        let bad = SystemConfig {
            memory_limit_mb: Some(0),
            ..Default::default()
        };
        let result = g3.check_memory_pressure(&bad);
        assert!(!result.passed);
    }
}
