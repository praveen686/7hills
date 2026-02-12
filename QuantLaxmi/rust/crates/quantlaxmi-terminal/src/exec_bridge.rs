//! Execution Bridge — wraps quantlaxmi-executor + quantlaxmi-risk for terminal access.
//!
//! Performs pre-trade risk checks before routing to the appropriate exchange.

use quantlaxmi_models::{OrderEvent, Side};
use quantlaxmi_risk::{RiskConfig, RiskEngine, RiskViolation};
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Result of an order placement attempt through the execution bridge.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExecResult {
    pub order_id: String,
    pub status: ExecStatus,
    pub message: String,
    pub risk_check: RiskCheckResult,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ExecStatus {
    Accepted,
    Rejected,
    Error,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub passed: bool,
    pub violation: Option<String>,
    pub latency_ns: u64,
}

/// Bridges frontend order intents to the execution layer with risk pre-checks.
pub struct ExecutionBridge {
    risk_engine: RiskEngine,
    kill_switch: Arc<AtomicBool>,
}

impl ExecutionBridge {
    pub fn new(max_order_value: f64, max_notional_per_symbol: f64) -> Self {
        let kill_switch = Arc::new(AtomicBool::new(false));
        let config = RiskConfig {
            max_order_value_usd: max_order_value,
            max_notional_per_symbol_usd: max_notional_per_symbol,
        };

        Self {
            risk_engine: RiskEngine::new(config, kill_switch.clone()),
            kill_switch,
        }
    }

    /// Check if an order passes risk gates before sending to exchange.
    pub fn pre_trade_check(
        &self,
        order: &OrderEvent,
        current_price: f64,
    ) -> RiskCheckResult {
        let start = std::time::Instant::now();
        let result = self.risk_engine.check_order(order, current_price);
        let latency_ns = start.elapsed().as_nanos() as u64;

        match result {
            Ok(()) => RiskCheckResult {
                passed: true,
                violation: None,
                latency_ns,
            },
            Err(violation) => RiskCheckResult {
                passed: false,
                violation: Some(format!("{}", violation)),
                latency_ns,
            },
        }
    }

    /// Activate the global kill switch — blocks all future orders.
    pub fn activate_kill_switch(&self) {
        self.kill_switch
            .store(true, std::sync::atomic::Ordering::SeqCst);
        tracing::error!("[EXEC BRIDGE] KILL SWITCH ACTIVATED");
    }

    /// Deactivate the global kill switch.
    pub fn deactivate_kill_switch(&self) {
        self.kill_switch
            .store(false, std::sync::atomic::Ordering::SeqCst);
        tracing::warn!("[EXEC BRIDGE] Kill switch deactivated");
    }

    /// Check if kill switch is currently active.
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Update position tracking in the risk engine.
    pub fn update_position(&mut self, symbol: String, quantity: f64) {
        self.risk_engine.update_position(symbol, quantity);
    }
}
