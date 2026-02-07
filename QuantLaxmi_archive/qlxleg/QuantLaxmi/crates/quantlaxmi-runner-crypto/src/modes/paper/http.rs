//! HTTP endpoints for paper trading order intake.

use axum::{
    Json, Router,
    extract::{Query, State},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use super::run::{OrderCommand, PositionInfo, SubmitOrderAck, TopOfBook};

/// HTTP server state.
#[derive(Clone)]
pub struct HttpState {
    tx: mpsc::Sender<OrderCommand>,
}

/// Request to submit an order.
#[derive(Debug, Deserialize)]
pub struct SubmitOrderReq {
    #[serde(default)]
    pub symbol: String,
    pub side: String,       // "buy" | "sell"
    pub order_type: String, // "market" | "limit"
    pub qty: f64,
    pub limit_price: Option<f64>,
    pub tag: Option<String>,
}

/// Response to order submission.
#[derive(Debug, Serialize)]
pub struct SubmitOrderResp {
    pub order_id: u64,
    pub accepted: bool,
    pub reason: Option<String>,
}

/// Query parameters for symbol-based endpoints.
#[derive(Debug, Deserialize)]
pub struct SymbolQuery {
    #[serde(default)]
    pub symbol: String,
}

/// Start the HTTP server for paper trading.
pub async fn serve_http(bind: String, tx: mpsc::Sender<OrderCommand>) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/order", post(post_order))
        .route("/top", get(get_top))
        .route("/position", get(get_position))
        .route("/health", get(health))
        .with_state(HttpState { tx });

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!("Paper trading HTTP server listening on {}", bind);

    axum::serve(listener, app).await?;
    Ok(())
}

/// Health check endpoint.
async fn health() -> &'static str {
    "ok"
}

/// GET /top?symbol=... - Get top of book.
async fn get_top(State(st): State<HttpState>, Query(q): Query<SymbolQuery>) -> Json<TopOfBook> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let cmd = OrderCommand::GetTop {
        symbol: q.symbol,
        resp: resp_tx,
    };

    if st.tx.send(cmd).await.is_err() {
        return Json(TopOfBook {
            symbol: String::new(),
            best_bid: None,
            best_ask: None,
            spread: None,
        });
    }

    match resp_rx.await {
        Ok(top) => Json(top),
        Err(_) => Json(TopOfBook {
            symbol: String::new(),
            best_bid: None,
            best_ask: None,
            spread: None,
        }),
    }
}

/// GET /position?symbol=... - Get position info.
async fn get_position(
    State(st): State<HttpState>,
    Query(q): Query<SymbolQuery>,
) -> Json<PositionInfo> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let cmd = OrderCommand::GetPosition {
        symbol: q.symbol,
        resp: resp_tx,
    };

    if st.tx.send(cmd).await.is_err() {
        return Json(PositionInfo {
            symbol: String::new(),
            position: 0.0,
            realized_pnl: 0.0,
        });
    }

    match resp_rx.await {
        Ok(pos) => Json(pos),
        Err(_) => Json(PositionInfo {
            symbol: String::new(),
            position: 0.0,
            realized_pnl: 0.0,
        }),
    }
}

/// POST /order - Submit a new order.
async fn post_order(
    State(st): State<HttpState>,
    Json(req): Json<SubmitOrderReq>,
) -> Json<SubmitOrderResp> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let cmd = OrderCommand::Submit { req, resp: resp_tx };

    if let Err(e) = st.tx.send(cmd).await {
        return Json(SubmitOrderResp {
            order_id: 0,
            accepted: false,
            reason: Some(format!("order channel send failed: {e}")),
        });
    }

    match resp_rx.await {
        Ok(Ok(SubmitOrderAck { order_id })) => Json(SubmitOrderResp {
            order_id,
            accepted: true,
            reason: None,
        }),
        Ok(Err(e)) => Json(SubmitOrderResp {
            order_id: 0,
            accepted: false,
            reason: Some(e.to_string()),
        }),
        Err(e) => Json(SubmitOrderResp {
            order_id: 0,
            accepted: false,
            reason: Some(format!("order ack dropped: {e}")),
        }),
    }
}

/// Parse side string to Side enum.
pub fn parse_side(s: &str) -> anyhow::Result<quantlaxmi_executor::SimSide> {
    match s.to_lowercase().as_str() {
        "buy" => Ok(quantlaxmi_executor::SimSide::Buy),
        "sell" => Ok(quantlaxmi_executor::SimSide::Sell),
        _ => Err(anyhow::anyhow!("invalid side: {s} (expected buy|sell)")),
    }
}

/// Parse order type string to OrderType enum.
pub fn parse_order_type(s: &str) -> anyhow::Result<quantlaxmi_executor::OrderType> {
    match s.to_lowercase().as_str() {
        "market" => Ok(quantlaxmi_executor::OrderType::Market),
        "limit" => Ok(quantlaxmi_executor::OrderType::Limit),
        _ => Err(anyhow::anyhow!(
            "invalid order_type: {s} (expected market|limit)"
        )),
    }
}
