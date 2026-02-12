//! Terminal Configuration — workspace persistence and app settings.

use serde::{Deserialize, Serialize};

/// Terminal-wide configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalConfig {
    /// FastAPI backend URL.
    pub api_url: String,
    /// Zerodha API credentials.
    pub zerodha: Option<ZerodhaConfig>,
    /// Binance API credentials.
    pub binance: Option<BinanceConfig>,
    /// Default workspace layout.
    pub default_workspace: String,
    /// Risk limits.
    pub risk: RiskLimits,
    /// Theme preference.
    pub theme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZerodhaConfig {
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinanceConfig {
    pub api_key: String,
    pub api_secret: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_order_value: f64,
    pub max_notional_per_symbol: f64,
    pub max_drawdown_pct: f64,
    pub max_daily_loss: f64,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:8000".into(),
            zerodha: None,
            binance: None,
            default_workspace: "trading".into(),
            risk: RiskLimits {
                max_order_value: 500_000.0,
                max_notional_per_symbol: 2_000_000.0,
                max_drawdown_pct: 5.0,
                max_daily_loss: 100_000.0,
            },
            theme: "dark".into(),
        }
    }
}
