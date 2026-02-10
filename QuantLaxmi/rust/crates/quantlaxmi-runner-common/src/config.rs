//! # Configuration Loading
//!
//! Shared configuration structures for QuantLaxmi runners.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
pub struct StrategyConfig {
    pub hydra: Option<quantlaxmi_core::hydra::HydraConfig>,
    pub aeon: Option<quantlaxmi_core::aeon::AeonConfig>,
}

/// Root configuration schema for the trading runner.
#[derive(Debug, Deserialize, Clone)]
pub struct RunnerConfig {
    pub mode: ModeInfo,
    pub risk: RiskInfo,
    pub execution: ExecutionInfo,
    pub strategy: Option<StrategyConfig>,
}

/// Information regarding the execution target and symbols.
#[derive(Debug, Deserialize, Clone)]
pub struct ModeInfo {
    pub symbols: Vec<String>,
}

/// Static risk constraints defined in configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct RiskInfo {
    pub max_order_value_usd: f64,
    pub max_notional_per_symbol_usd: f64,
}

/// Operational settings for the execution layer.
#[derive(Debug, Deserialize, Clone)]
pub struct ExecutionInfo {
    pub slippage_bps: Option<f64>,
    pub commission_model: Option<String>,
    pub lot_sizes: Option<HashMap<String, u32>>,
}

impl RunnerConfig {
    /// Load configuration from file path
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)
            .or_else(|_| std::fs::read_to_string(format!("../../{}", path)))
            .map_err(|_| anyhow::anyhow!("Could not find config file: {}", path))?;

        toml::from_str(&config_str).map_err(|e| anyhow::anyhow!("Failed to parse config: {}", e))
    }

    /// Get the config root directory.
    ///
    /// Returns the value of `QUANTLAXMI_CONFIG_ROOT` if set, otherwise `"configs"`.
    /// This allows the Rust workspace to find its config files when nested
    /// inside the monorepo (e.g. `QUANTLAXMI_CONFIG_ROOT=rust/configs`).
    pub fn config_root() -> String {
        std::env::var("QUANTLAXMI_CONFIG_ROOT").unwrap_or_else(|_| "configs".to_string())
    }

    /// Get default config path for a given execution mode
    pub fn default_path(mode: &quantlaxmi_core::ExecutionMode) -> String {
        let root = Self::config_root();
        match mode {
            quantlaxmi_core::ExecutionMode::Backtest => format!("{}/backtest.toml", root),
            quantlaxmi_core::ExecutionMode::Live => format!("{}/live.toml", root),
            quantlaxmi_core::ExecutionMode::Paper => format!("{}/paper.toml", root),
        }
    }
}
