//! # QuantLaxmi Runner Common
//!
//! Shared runner infrastructure for both India and Crypto binaries.
//!
//! ## Isolation Guarantee
//! This crate has NO dependency on any venue-specific connectors:
//! - NO Zerodha imports
//! - NO Binance imports
//! - NO SBE imports
//!
//! ## Contents
//! - Circuit breakers and rate limiters
//! - Web server for metrics broadcasting
//! - Configuration loading utilities
//! - TUI framework
//! - Report assembly helpers
//! - VectorBT export utilities

pub mod artifact;
pub mod circuit_breakers;
pub mod config;
pub mod report;
pub mod tui;
pub mod vectorbt;
pub mod web_server;

pub use circuit_breakers::{
    CircuitBreakerStatus, LatencyCircuitBreaker, RateLimiter, TradingCircuitBreakers,
};
pub use config::{ExecutionInfo, ModeInfo, RiskInfo, RunnerConfig, StrategyConfig};
pub use web_server::{ServerState, WebMessage, start_server};

use kubera_core::ExecutionMode;
use kubera_core::{MetricsConfig, TradingMetrics};
use kubera_data::Level2Book;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Volatile state for a single instrument tracking.
pub struct SymbolState {
    pub last_price: f64,
    pub position: f64,
    pub book: Level2Book,
    pub strategy_active: bool,
}

/// Global application state for the TUI and orchestration.
pub struct AppState {
    pub symbols: std::collections::HashMap<String, SymbolState>,
    pub equity: f64,
    pub realized_pnl: f64,
    pub order_log: Vec<String>,
    pub mode: ExecutionMode,
    pub metrics: TradingMetrics,
    pub session_start: std::time::Instant,
    pub tick_count: u64,
    pub circuit_breakers: Option<TradingCircuitBreakers>,
}

impl AppState {
    pub fn new(
        symbols: Vec<String>,
        initial_capital: f64,
        mode: ExecutionMode,
        headless: bool,
        kill_switch: Arc<AtomicBool>,
        is_indian_fno: bool,
    ) -> Self {
        let mut symbols_state = std::collections::HashMap::new();
        for s in &symbols {
            symbols_state.insert(
                s.clone(),
                SymbolState {
                    last_price: 0.0,
                    position: 0.0,
                    book: Level2Book::new(s.clone()),
                    strategy_active: headless,
                },
            );
        }

        let metrics_config = MetricsConfig {
            initial_capital,
            risk_free_rate: 0.05,
            rolling_window: 252,
            min_trades_for_metrics: 5,
            trading_days_per_year: 252.0,
            var_confidence: 0.95,
            sampling_interval_seconds: 60.0,
        };
        let trading_metrics = TradingMetrics::new(metrics_config);

        let circuit_breakers = if mode != ExecutionMode::Backtest {
            let cb = if is_indian_fno {
                TradingCircuitBreakers::for_indian_fno(initial_capital, kill_switch)
            } else {
                TradingCircuitBreakers::new(initial_capital, kill_switch)
            };
            info!(
                "[RUNNER] Circuit breakers enabled (Indian F&O: {})",
                is_indian_fno
            );
            Some(cb)
        } else {
            None
        };

        Self {
            symbols: symbols_state,
            equity: initial_capital,
            realized_pnl: 0.0,
            order_log: Vec::new(),
            mode,
            metrics: trading_metrics,
            session_start: std::time::Instant::now(),
            tick_count: 0,
            circuit_breakers,
        }
    }
}

/// Initialize observability (metrics + tracing)
pub fn init_observability(service_name: &str) {
    let metrics_port = std::env::var("METRICS_PORT").unwrap_or_else(|_| "9000".to_string());
    let metrics_addr = format!("0.0.0.0:{}", metrics_port)
        .parse()
        .expect("Invalid metrics address");
    kubera_core::observability::init_metrics(metrics_addr);
    kubera_core::observability::init_tracing(service_name);
}

/// Create the shared tokio runtime with appropriate stack size
pub fn create_runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024) // 16MB stack
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create runtime: {}", e))
}
