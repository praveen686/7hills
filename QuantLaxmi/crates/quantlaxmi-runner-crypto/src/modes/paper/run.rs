//! Paper trading main loop.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use quantlaxmi_executor::{PaperEngine, RiskEnvelope, SimConfig};
use quantlaxmi_models::depth::DepthEvent;
use quantlaxmi_wal::{JsonlWriter, SessionDir};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use super::artifacts::JsonlFillSink;
use super::http::SubmitOrderReq;

/// Configuration for paper trading mode.
#[derive(Debug, Clone)]
pub struct PaperModeConfig {
    pub base_dir: PathBuf,
    pub symbol: String,
    pub initial_capital: f64,
    pub fee_bps_maker: f64,
    pub fee_bps_taker: f64,
    pub http_bind: String,
}

impl Default for PaperModeConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("."),
            symbol: "BTCUSDT".to_string(),
            initial_capital: 10_000.0,
            fee_bps_maker: 2.0,
            fee_bps_taker: 10.0,
            http_bind: "127.0.0.1:8080".to_string(),
        }
    }
}

/// Commands sent from HTTP handlers to the paper trading loop.
#[derive(Debug)]
pub enum OrderCommand {
    Submit {
        req: SubmitOrderReq,
        resp: oneshot::Sender<anyhow::Result<SubmitOrderAck>>,
    },
    GetTop {
        symbol: String,
        resp: oneshot::Sender<TopOfBook>,
    },
    GetPosition {
        symbol: String,
        resp: oneshot::Sender<PositionInfo>,
    },
}

/// Top-of-book snapshot.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopOfBook {
    pub symbol: String,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub spread: Option<f64>,
}

/// Position information.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PositionInfo {
    pub symbol: String,
    pub position: f64,
    pub realized_pnl: f64,
}

/// Acknowledgment for a submitted order.
#[derive(Debug, serde::Serialize)]
pub struct SubmitOrderAck {
    pub order_id: u64,
}

/// Session summary written on graceful shutdown.
#[derive(Debug, serde::Serialize)]
pub struct SessionSummary {
    pub symbol: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub initial_capital: f64,
    pub final_position: f64,
    pub realized_pnl: f64,
    pub total_fills: usize,
    pub depth_events_processed: u64,
    pub shutdown_reason: String,
}

/// Run paper trading mode.
///
/// This is the main entry point for paper trading. It:
/// 1. Creates a session directory
/// 2. Opens WAL writers for depth and fills
/// 3. Creates the paper engine
/// 4. Starts the HTTP server for order intake
/// 5. Runs the main event loop
pub async fn run_paper_mode(
    cfg: PaperModeConfig,
    mut depth_rx: mpsc::Receiver<DepthEvent>,
) -> anyhow::Result<()> {
    // 1) Session directory
    let session = SessionDir::new_paper(&cfg.base_dir, &cfg.symbol).await?;
    info!("Paper session: {:?}", session.path());

    // 2) WAL writers
    let mut depth_wal: JsonlWriter<DepthEvent> =
        JsonlWriter::open_append(session.depth_wal_path()).await?;

    // 3) Fill sink -> writes Fill JSONL
    let sink = JsonlFillSink::open(session.fills_path()).await?;

    // 4) Engine
    let sim_cfg = SimConfig {
        fee_bps_maker: cfg.fee_bps_maker,
        fee_bps_taker: cfg.fee_bps_taker,
        initial_cash: cfg.initial_capital,
        ..SimConfig::default()
    };

    let risk = RiskEnvelope::for_equity(cfg.initial_capital);
    let mut engine: PaperEngine<JsonlFillSink> = PaperEngine::new(sim_cfg, risk, sink);

    // 5) Command channel for HTTP -> loop communication
    let (tx, mut rx) = mpsc::channel::<OrderCommand>(1024);

    // 6) Start HTTP server
    let http_bind = cfg.http_bind.clone();
    let http_handle = tokio::spawn(super::http::serve_http(http_bind, tx.clone()));

    // 7) Session tracking
    let started_at = Utc::now();
    let mut depth_events_processed: u64 = 0;
    let mut shutdown_reason = "channel_closed"; // Default if channels close

    info!(
        "Paper trading started for {} with ${:.0} initial capital",
        cfg.symbol, cfg.initial_capital
    );

    // 8) Set up shutdown signal channel
    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<&'static str>(1);

    // Spawn signal handlers
    let shutdown_tx_ctrl_c = shutdown_tx.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = shutdown_tx_ctrl_c.send("SIGINT").await;
    });

    #[cfg(unix)]
    {
        let shutdown_tx_term = shutdown_tx.clone();
        tokio::spawn(async move {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler");
            sigterm.recv().await;
            let _ = shutdown_tx_term.send("SIGTERM").await;
        });
    }

    drop(shutdown_tx); // Only signal handlers hold senders now

    // 9) Main event loop
    // Use biased select to ensure HTTP commands aren't starved by depth events
    loop {
        tokio::select! {
            biased;

            // Shutdown signal (highest priority - always check first)
            Some(signal) = shutdown_rx.recv() => {
                info!("Received {}, initiating graceful shutdown...", signal);
                shutdown_reason = signal;
                break;
            }

            // HTTP commands (high priority - must remain responsive)
            Some(cmd) = rx.recv() => {
                handle_cmd(&mut engine, &cfg.symbol, cmd).await;
            }

            // Depth events (bulk processing)
            Some(event) = depth_rx.recv() => {
                // Durability first: write to WAL
                depth_wal.write(&event).await?;
                depth_events_processed += 1;

                // Drive engine
                let fills = engine.on_depth(&event)?;

                if !fills.is_empty() {
                    info!(
                        "Fills generated: {} (pos={:.4}, pnl={:.2})",
                        fills.len(),
                        engine.position(&cfg.symbol),
                        engine.realized_pnl()
                    );
                }
            }

            // All channels closed
            else => {
                info!("All channels closed, shutting down...");
                break;
            }
        }
    }

    // 10) Graceful shutdown: flush all writers
    info!("Flushing fill sink...");
    if let Err(e) = engine.flush() {
        warn!("Error flushing fill sink: {}", e);
    }

    info!("Flushing depth WAL...");
    if let Err(e) = depth_wal.flush().await {
        warn!("Error flushing depth WAL: {}", e);
    }

    // 11) Write session summary
    let summary = SessionSummary {
        symbol: cfg.symbol.clone(),
        started_at,
        ended_at: Utc::now(),
        initial_capital: cfg.initial_capital,
        final_position: engine.position(&cfg.symbol),
        realized_pnl: engine.realized_pnl(),
        total_fills: engine.fills().len(),
        depth_events_processed,
        shutdown_reason: shutdown_reason.to_string(),
    };

    let summary_path = session.path().join("session_summary.json");
    match tokio::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?).await {
        Ok(_) => info!("Session summary written to {:?}", summary_path),
        Err(e) => warn!("Failed to write session summary: {}", e),
    }

    info!(
        "Paper session ended. Final position: {:.4}, PnL: {:.2}, Fills: {}, Depth events: {}",
        summary.final_position,
        summary.realized_pnl,
        summary.total_fills,
        summary.depth_events_processed
    );

    // 12) Wait for HTTP server to finish
    drop(tx); // Signal HTTP server to stop accepting new requests
    let _ = http_handle.await;

    Ok(())
}

/// Handle a command from HTTP.
async fn handle_cmd<S: quantlaxmi_executor::FillSink>(
    engine: &mut PaperEngine<S>,
    default_symbol: &str,
    cmd: OrderCommand,
) {
    match cmd {
        OrderCommand::Submit { req, resp } => {
            let out = submit_order(engine, default_symbol, req);
            let _ = resp.send(out);
        }
        OrderCommand::GetTop { symbol, resp } => {
            let sym = if symbol.is_empty() {
                default_symbol
            } else {
                &symbol
            };
            let best_bid = engine.best_bid(sym);
            let best_ask = engine.best_ask(sym);
            let spread = match (best_bid, best_ask) {
                (Some(b), Some(a)) => Some(a - b),
                _ => None,
            };
            let _ = resp.send(TopOfBook {
                symbol: sym.to_string(),
                best_bid,
                best_ask,
                spread,
            });
        }
        OrderCommand::GetPosition { symbol, resp } => {
            let sym = if symbol.is_empty() {
                default_symbol
            } else {
                &symbol
            };
            let _ = resp.send(PositionInfo {
                symbol: sym.to_string(),
                position: engine.position(sym),
                realized_pnl: engine.realized_pnl(),
            });
        }
    }
}

/// Process an order submission request.
///
/// All validation happens here at the boundary:
/// - qty > 0
/// - valid side/order_type (via parse_*)
/// - limit orders require limit_price
/// - market orders require book to have best bid/ask
fn submit_order<S: quantlaxmi_executor::FillSink>(
    engine: &mut PaperEngine<S>,
    default_symbol: &str,
    req: SubmitOrderReq,
) -> anyhow::Result<SubmitOrderAck> {
    // Validation: qty > 0
    if req.qty <= 0.0 {
        return Err(anyhow::anyhow!("qty must be > 0, got {}", req.qty));
    }

    // Validation: valid side and order_type
    let side = super::http::parse_side(&req.side)?;
    let order_type = super::http::parse_order_type(&req.order_type)?;

    let symbol = if req.symbol.is_empty() {
        default_symbol.to_string()
    } else {
        req.symbol
    };

    // Validation: market orders require the opposite side of book
    // - Market buy executes against asks (needs best_ask)
    // - Market sell executes against bids (needs best_bid)
    if matches!(order_type, quantlaxmi_executor::OrderType::Market) {
        match side {
            quantlaxmi_executor::SimSide::Buy => {
                if engine.best_ask(&symbol).is_none() {
                    return Err(anyhow::anyhow!(
                        "book not ready: market buy requires best_ask for {}",
                        symbol
                    ));
                }
            }
            quantlaxmi_executor::SimSide::Sell => {
                if engine.best_bid(&symbol).is_none() {
                    return Err(anyhow::anyhow!(
                        "book not ready: market sell requires best_bid for {}",
                        symbol
                    ));
                }
            }
        }
    }

    // Validation: limit orders require limit_price
    if matches!(order_type, quantlaxmi_executor::OrderType::Limit) && req.limit_price.is_none() {
        return Err(anyhow::anyhow!("limit_price required for limit order"));
    }

    let ts_ns = now_ts_ns();
    let id = engine.sim.next_order_id();

    let mut order = match order_type {
        quantlaxmi_executor::OrderType::Market => {
            quantlaxmi_executor::Order::market(id, symbol, side, req.qty)
        }
        quantlaxmi_executor::OrderType::Limit => {
            let px = req.limit_price.unwrap(); // Safe: validated above
            quantlaxmi_executor::Order::limit(id, symbol, side, req.qty, px)
        }
    };

    if let Some(tag) = req.tag {
        order = order.with_tag(tag);
    }
    order = order.with_created_ts_ns(ts_ns);

    let fills = engine.submit(ts_ns, order)?;

    info!(
        "Order {} submitted: {:?} {} @ {:?} -> {} fills",
        id,
        side,
        req.qty,
        req.limit_price,
        fills.len()
    );

    Ok(SubmitOrderAck { order_id: id })
}

/// Get current timestamp in nanoseconds.
fn now_ts_ns() -> u64 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    dur.as_secs() * 1_000_000_000 + u64::from(dur.subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_executor::{NoopFillSink, RiskEnvelope, SimConfig};
    use quantlaxmi_models::depth::{DepthEvent, DepthLevel, IntegrityTier};

    fn make_engine() -> PaperEngine<NoopFillSink> {
        let cfg = SimConfig::default();
        let risk = RiskEnvelope::for_equity(10_000.0);
        PaperEngine::new(cfg, risk, NoopFillSink)
    }

    fn make_depth_event(symbol: &str, bid: i64, ask: i64) -> DepthEvent {
        DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: symbol.to_string(),
            first_update_id: 1,
            last_update_id: 1,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![DepthLevel {
                price: bid,
                qty: 100_000_000,
            }],
            asks: vec![DepthLevel {
                price: ask,
                qty: 100_000_000,
            }],
            is_snapshot: true,
            integrity_tier: IntegrityTier::Certified,
            source: Some("test".to_string()),
        }
    }

    #[test]
    fn market_buy_before_depth_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "buy".to_string(),
            order_type: "market".to_string(),
            qty: 0.001,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("market buy requires best_ask"), "got: {}", err);
    }

    #[test]
    fn market_sell_before_depth_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "sell".to_string(),
            order_type: "market".to_string(),
            qty: 0.001,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("market sell requires best_bid"),
            "got: {}",
            err
        );
    }

    #[test]
    fn market_order_after_depth_accepted() {
        let mut engine = make_engine();

        // Apply depth first
        let event = make_depth_event("BTCUSDT", 1_000_000, 1_000_100); // 10000.00, 10001.00
        let _ = engine.on_depth(&event);

        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "buy".to_string(),
            order_type: "market".to_string(),
            qty: 0.001,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_ok(), "expected ok, got: {:?}", result);
    }

    #[test]
    fn limit_order_without_price_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "buy".to_string(),
            order_type: "limit".to_string(),
            qty: 0.001,
            limit_price: None, // Missing!
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("limit_price required"), "got: {}", err);
    }

    #[test]
    fn qty_zero_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "buy".to_string(),
            order_type: "market".to_string(),
            qty: 0.0,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("qty must be > 0"), "got: {}", err);
    }

    #[test]
    fn qty_negative_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "buy".to_string(),
            order_type: "market".to_string(),
            qty: -1.0,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("qty must be > 0"), "got: {}", err);
    }

    #[test]
    fn invalid_side_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "invalid".to_string(),
            order_type: "market".to_string(),
            qty: 0.001,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid side"), "got: {}", err);
    }

    #[test]
    fn invalid_order_type_rejected() {
        let mut engine = make_engine();
        let req = SubmitOrderReq {
            symbol: "BTCUSDT".to_string(),
            side: "buy".to_string(),
            order_type: "stop".to_string(), // Not supported
            qty: 0.001,
            limit_price: None,
            tag: None,
        };

        let result = submit_order(&mut engine, "BTCUSDT", req);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid order_type"), "got: {}", err);
    }
}
