//! # Report Assembly
//!
//! Shared report generation and export utilities.

use serde::{Serialize, Deserialize};
use std::path::Path;
use chrono::{DateTime, Utc};
use tracing::info;

/// Backtest/trading session report
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionReport {
    pub session_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub venue: String,
    pub symbols: Vec<String>,
    pub initial_equity: f64,
    pub final_equity: f64,
    pub total_pnl: f64,
    pub total_return_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown_pct: f64,
    pub total_trades: u64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_trade_pnl: f64,
    pub metadata: ReportMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub strategy: String,
    pub version: String,
    pub certified: bool,
    pub config_hash: String,
}

impl SessionReport {
    /// Create a new session report
    pub fn new(
        session_id: String,
        venue: String,
        symbols: Vec<String>,
        initial_equity: f64,
        strategy: String,
    ) -> Self {
        Self {
            session_id,
            start_time: Utc::now(),
            end_time: Utc::now(),
            venue,
            symbols,
            initial_equity,
            final_equity: initial_equity,
            total_pnl: 0.0,
            total_return_pct: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown_pct: 0.0,
            total_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_trade_pnl: 0.0,
            metadata: ReportMetadata {
                strategy,
                version: env!("CARGO_PKG_VERSION").to_string(),
                certified: false,
                config_hash: String::new(),
            },
        }
    }

    /// Save report to JSON file
    pub fn save(&self, out_dir: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(out_dir)?;
        let report_path = out_dir.join("report.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&report_path, json)?;
        info!("Report saved to: {}", report_path.display());
        Ok(())
    }

    /// Update metrics from TradingMetrics snapshot
    pub fn update_from_metrics(&mut self, snapshot: &kubera_core::MetricsSnapshot) {
        self.end_time = Utc::now();
        self.final_equity = snapshot.equity;
        self.total_pnl = snapshot.total_pnl;
        self.total_return_pct = snapshot.total_return_pct;
        self.sharpe_ratio = snapshot.sharpe_ratio;
        self.sortino_ratio = snapshot.sortino_ratio;
        self.max_drawdown_pct = snapshot.max_drawdown_pct;
        self.total_trades = snapshot.total_trades;
        self.win_rate = snapshot.win_rate;
        self.profit_factor = snapshot.profit_factor;
        self.avg_trade_pnl = if snapshot.total_trades > 0 {
            snapshot.total_pnl / snapshot.total_trades as f64
        } else {
            0.0
        };
    }
}

/// Create output directory for a run
pub fn create_run_dir(base: &str, run_id: &str) -> anyhow::Result<std::path::PathBuf> {
    let path = Path::new(base).join(run_id);
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

/// Generate a unique run ID
pub fn generate_run_id() -> String {
    let now = Utc::now();
    format!("{}_{}", now.format("%Y%m%d_%H%M%S"), &uuid::Uuid::new_v4().to_string()[..8])
}
