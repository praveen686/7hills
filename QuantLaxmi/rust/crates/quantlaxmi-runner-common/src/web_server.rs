//! # QuantLaxmi WebSocket Server
//!
//! Real-time metrics and control interface for paper trading.
//!
//! ## Description
//! Provides a WebSocket server that broadcasts trading metrics, status updates,
//! and order logs to connected clients. Supports bidirectional communication
//! for remote control of the trading system.
//!
//! ## Phase 17A Extensions
//! - `ExecutionControlView` streaming for Phase 16 control plane observability
//! - `OperatorRequest`/`OperatorResponse` for typed operator commands
//! - `/status` endpoint for one-shot JSON status

use crate::circuit_breakers::CircuitBreakerStatus;
use crate::control_view::{ExecutionControlView, OperatorRequest, OperatorResponse};
use axum::{
    Json, Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use futures::{sink::SinkExt, stream::StreamExt};
use quantlaxmi_gates::{KillSwitchEvent, ManualOverrideEvent, SessionTransitionEvent};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info};

/// WebSocket message types for client communication
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum WebMessage {
    /// Basic status update (simple equity/pnl broadcast)
    Status { equity: f64, realized_pnl: f64 },

    /// Per-symbol price and position update
    SymbolUpdate {
        symbol: String,
        last_price: f64,
        position: f64,
    },

    /// Order execution log message
    OrderLog { message: String },

    /// Command from UI (actions)
    Command { action: String, target: String },

    /// Comprehensive trading metrics snapshot
    MetricsSnapshot {
        // Return metrics
        total_pnl: f64,
        total_return_pct: f64,
        annualized_return: f64,

        // Risk metrics
        sharpe_ratio: f64,
        sortino_ratio: f64,
        calmar_ratio: f64,
        max_drawdown_pct: f64,
        current_drawdown_pct: f64,
        volatility: f64,
        var_95: f64,

        // Trade metrics
        total_trades: u64,
        win_rate: f64,
        profit_factor: f64,
        expectancy: f64,
        avg_win: f64,
        avg_loss: f64,
        largest_win: f64,
        largest_loss: f64,
        payoff_ratio: f64,

        // Efficiency
        kelly_fraction: f64,
        recovery_factor: f64,

        // Execution
        avg_slippage_bps: f64,
        total_commission: f64,

        // State
        equity: f64,
        peak_equity: f64,
    },

    /// Trade execution notification
    TradeExecution {
        symbol: String,
        side: String,
        quantity: f64,
        price: f64,
        pnl: f64,
        commission: f64,
        strategy: String,
    },

    /// Regime change notification (from Hydra)
    RegimeChange {
        previous: String,
        current: String,
        confidence: f64,
    },

    /// Expert weight update (from Hydra meta-allocator)
    ExpertWeights { weights: Vec<(String, f64)> },

    /// Kill switch status
    KillSwitch {
        triggered: bool,
        reason: Option<String>,
    },

    /// System health heartbeat
    Heartbeat {
        uptime_secs: u64,
        tick_count: u64,
        events_per_sec: f64,
    },

    // === Phase 17A: Execution Control Plane Observability ===
    /// Full execution control view (sent on connect + state changes).
    ExecutionControlView { view: ExecutionControlView },

    /// Session state transition event (real-time).
    SessionTransition { event: SessionTransitionEvent },

    /// Kill-switch activation/deactivation event.
    KillSwitchEvent { event: KillSwitchEvent },

    /// Manual override event.
    ManualOverrideEvent { event: ManualOverrideEvent },

    /// Operator request (inbound from client).
    OperatorRequest {
        correlation_id: Option<String>,
        request: OperatorRequest,
    },

    /// Operator response (outbound to client).
    OperatorResponse { response: OperatorResponse },
}

impl WebMessage {
    /// Create a MetricsSnapshot message from a quantlaxmi_core::MetricsSnapshot
    pub fn from_metrics_snapshot(m: &quantlaxmi_core::MetricsSnapshot) -> Self {
        WebMessage::MetricsSnapshot {
            total_pnl: m.total_pnl,
            total_return_pct: m.total_return_pct,
            annualized_return: m.annualized_return,
            sharpe_ratio: m.sharpe_ratio,
            sortino_ratio: m.sortino_ratio,
            calmar_ratio: m.calmar_ratio,
            max_drawdown_pct: m.max_drawdown_pct,
            current_drawdown_pct: m.current_drawdown_pct,
            volatility: m.volatility,
            var_95: m.var_95,
            total_trades: m.total_trades,
            win_rate: m.win_rate,
            profit_factor: m.profit_factor,
            expectancy: m.expectancy,
            avg_win: m.avg_win,
            avg_loss: m.avg_loss,
            largest_win: m.largest_win,
            largest_loss: m.largest_loss,
            payoff_ratio: m.payoff_ratio,
            kelly_fraction: m.kelly_fraction,
            recovery_factor: m.recovery_factor,
            avg_slippage_bps: m.avg_slippage_bps,
            total_commission: m.total_commission,
            equity: m.equity,
            peak_equity: m.peak_equity,
        }
    }
}

// =============================================================================
// Status Response (Phase 17A)
// =============================================================================

/// One-shot status response for /status endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Timestamp in nanoseconds.
    pub ts_ns: i64,

    /// Trading metrics snapshot (from quantlaxmi_core).
    pub metrics: Option<quantlaxmi_core::MetricsSnapshot>,

    /// Circuit breaker status.
    pub circuit_breakers: Option<CircuitBreakerStatus>,

    /// Execution control view (Phase 16 control plane).
    pub control: Option<ExecutionControlView>,
}

impl StatusResponse {
    /// Create a minimal status response with just timestamp.
    pub fn minimal(ts_ns: i64) -> Self {
        Self {
            ts_ns,
            metrics: None,
            circuit_breakers: None,
            control: None,
        }
    }
}

// =============================================================================
// Server State
// =============================================================================

/// Shared server state for WebSocket broadcasting.
pub struct ServerState {
    /// Broadcast channel for sending messages to all connected clients.
    pub tx: broadcast::Sender<WebMessage>,

    /// Shared execution control view (read-only snapshot, updated by runner core).
    pub control_view: Arc<RwLock<Option<ExecutionControlView>>>,

    /// Shared metrics snapshot (updated by runner core).
    pub metrics_snapshot: Arc<RwLock<Option<quantlaxmi_core::MetricsSnapshot>>>,

    /// Shared circuit breaker status (updated by runner core).
    pub circuit_breaker_status: Arc<RwLock<Option<CircuitBreakerStatus>>>,
}

impl ServerState {
    /// Create a new server state with default empty values.
    pub fn new(tx: broadcast::Sender<WebMessage>) -> Self {
        Self {
            tx,
            control_view: Arc::new(RwLock::new(None)),
            metrics_snapshot: Arc::new(RwLock::new(None)),
            circuit_breaker_status: Arc::new(RwLock::new(None)),
        }
    }

    /// Update the control view (called by runner core on state changes).
    pub async fn update_control_view(&self, view: ExecutionControlView) {
        *self.control_view.write().await = Some(view.clone());
        // Broadcast to connected clients
        let _ = self.tx.send(WebMessage::ExecutionControlView { view });
    }

    /// Broadcast a session transition event.
    pub fn broadcast_transition(&self, event: SessionTransitionEvent) {
        let _ = self.tx.send(WebMessage::SessionTransition { event });
    }

    /// Broadcast a kill-switch event.
    pub fn broadcast_kill_switch(&self, event: KillSwitchEvent) {
        let _ = self.tx.send(WebMessage::KillSwitchEvent { event });
    }

    /// Broadcast a manual override event.
    pub fn broadcast_override(&self, event: ManualOverrideEvent) {
        let _ = self.tx.send(WebMessage::ManualOverrideEvent { event });
    }

    /// Broadcast an operator response.
    pub fn broadcast_response(&self, response: OperatorResponse) {
        let _ = self.tx.send(WebMessage::OperatorResponse { response });
    }

    /// Get current status for /status endpoint.
    pub async fn get_status(&self) -> StatusResponse {
        let ts_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        StatusResponse {
            ts_ns,
            metrics: self.metrics_snapshot.read().await.clone(),
            circuit_breakers: self.circuit_breaker_status.read().await.clone(),
            control: self.control_view.read().await.clone(),
        }
    }
}

/// Start the WebSocket server
pub async fn start_server(state: Arc<ServerState>, port: u16) {
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/status", get(status_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .unwrap();
    info!(
        "Web control server listening on {}",
        listener.local_addr().unwrap()
    );
    info!("  WebSocket: ws://localhost:{}/ws", port);
    info!("  Health:    http://localhost:{}/health", port);
    info!("  Metrics:   http://localhost:{}/metrics", port);
    info!("  Status:    http://localhost:{}/status", port);
    axum::serve(listener, app).await.unwrap();
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "status": "ok",
        "service": "quantlaxmi-runner",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Metrics endpoint (placeholder for Prometheus integration)
async fn metrics_handler() -> impl IntoResponse {
    // This would typically be handled by the Prometheus exporter
    // but we provide a JSON fallback here
    axum::Json(serde_json::json!({
        "info": "Use Prometheus exporter on METRICS_PORT for full metrics",
        "hint": "Default: http://localhost:9000/metrics"
    }))
}

/// Status endpoint (Phase 17A) - One-shot JSON status.
async fn status_handler(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let status = state.get_status().await;
    Json(status)
}

/// WebSocket upgrade handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle an individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<ServerState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    info!("WebSocket client connected");

    // Spawn a task to send messages from the broadcast channel to the websocket
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap_or_default();
            if sender.send(Message::Text(json)).await.is_err() {
                debug!("WebSocket send failed, client disconnected");
                break;
            }
        }
    });

    // Handle incoming messages from the websocket (Commands)
    let tx_for_commands = state.tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            match serde_json::from_str::<WebMessage>(&text) {
                Ok(WebMessage::Command { action, target }) => {
                    info!(
                        "Received command from UI: action={}, target={}",
                        action, target
                    );

                    // Echo acknowledgment
                    let _ = tx_for_commands.send(WebMessage::OrderLog {
                        message: format!("[CMD] Received: {} {}", action, target),
                    });

                    // Command processing would happen here
                    // The runner would need to listen for these commands
                }
                Ok(_) => {
                    debug!("Received non-command message from client");
                }
                Err(e) => {
                    debug!("Failed to parse WebSocket message: {}", e);
                }
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = (&mut send_task) => {
            recv_task.abort();
            info!("WebSocket client disconnected (send task ended)");
        }
        _ = (&mut recv_task) => {
            send_task.abort();
            info!("WebSocket client disconnected (recv task ended)");
        }
    }
}
