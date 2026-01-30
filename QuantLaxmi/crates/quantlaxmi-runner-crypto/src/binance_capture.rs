//! Binance Spot capture -> canonical QuoteEvent JSONL.
//!
//! Uses Binance Spot bookTicker stream: best bid/ask updates.
//! Output format uses canonical `quantlaxmi_models::events::QuoteEvent`.
//!
//! Notes:
//! - Public stream (no API keys required)
//! - Deterministic fixed-point parsing (no float intermediates)
//! - Uses ResilientWs for auto-reconnect on disconnect

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::time::Instant;
use tracing::{error, warn};

use crate::ws_resilient::{ConnectionGap, ResilientWs, ResilientWsConfig};
use quantlaxmi_models::events::{CorrelationContext, QuoteEvent, parse_to_mantissa_pure};

#[derive(Debug, serde::Deserialize)]
struct BookTickerEvent {
    /// Event time (ms) - optional, not present in individual symbol streams
    #[serde(rename = "E")]
    event_time_ms: Option<i64>,
    /// Symbol
    #[serde(rename = "s")]
    symbol: String,
    /// Best bid price
    #[serde(rename = "b")]
    bid_price: String,
    /// Best bid qty
    #[serde(rename = "B")]
    bid_qty: String,
    /// Best ask price
    #[serde(rename = "a")]
    ask_price: String,
    /// Best ask qty
    #[serde(rename = "A")]
    ask_qty: String,
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
}

/// Statistics from bookTicker capture.
#[derive(Debug, Default)]
pub struct CaptureStats {
    pub events_written: usize,
}

impl std::fmt::Display for CaptureStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "events={}", self.events_written)
    }
}

/// Result from capture including connection gap info.
#[derive(Debug, Default)]
pub struct CaptureResult {
    pub stats: CaptureStats,
    pub connection_gaps: Vec<ConnectionGap>,
    pub total_reconnects: u32,
}

impl std::fmt::Display for CaptureResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.stats)?;
        if self.total_reconnects > 0 {
            write!(
                f,
                ", reconnects={}, gaps={}",
                self.total_reconnects,
                self.connection_gaps.len()
            )?;
        }
        Ok(())
    }
}

/// Capture spot bookTicker stream to canonical QuoteEvent JSONL.
/// Uses ResilientWs for auto-reconnect on disconnect.
pub async fn capture_book_ticker_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<CaptureResult> {
    let sym = symbol.to_lowercase();
    let url = format!("wss://stream.binance.com:9443/ws/{}@bookTicker", sym);

    let config = ResilientWsConfig {
        liveness_timeout: Duration::from_secs(30),
        read_timeout: Duration::from_secs(5),
        initial_backoff: Duration::from_secs(1),
        max_backoff: Duration::from_secs(30),
        max_reconnect_attempts: 100,
        ..Default::default()
    };

    let mut ws = ResilientWs::connect(&url, config)
        .await
        .with_context(|| format!("connect websocket: {}", url))?;

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(out_path)
        .await
        .with_context(|| format!("open output: {:?}", out_path))?;

    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let mut stats = CaptureStats::default();

    const BINANCE_PRICE_EXP: i8 = -2; // cents
    const BINANCE_QTY_EXP: i8 = -8; // base size precision (sufficient for BTC/ETH; canonical)

    while Instant::now() < deadline {
        let msg = match ws.next_message().await? {
            Some(m) => m,
            None => {
                error!("WebSocket reconnection exhausted for spot {}", symbol);
                break;
            }
        };

        if !msg.is_text() {
            continue;
        }
        let txt = match msg.into_text() {
            Ok(t) => t,
            Err(e) => {
                warn!("Failed to get text from message: {}", e);
                continue;
            }
        };

        // Skip non-bookTicker payloads
        let ev: BookTickerEvent = match serde_json::from_str(&txt) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let ts = match ev.event_time_ms {
            Some(ms) => ms_to_dt(ms),
            None => Utc::now(),
        };

        let q = QuoteEvent {
            ts,
            symbol: ev.symbol,
            bid_price_mantissa: parse_to_mantissa_pure(&ev.bid_price, BINANCE_PRICE_EXP)?,
            ask_price_mantissa: parse_to_mantissa_pure(&ev.ask_price, BINANCE_PRICE_EXP)?,
            bid_qty_mantissa: parse_to_mantissa_pure(&ev.bid_qty, BINANCE_QTY_EXP)?,
            ask_qty_mantissa: parse_to_mantissa_pure(&ev.ask_qty, BINANCE_QTY_EXP)?,
            price_exponent: BINANCE_PRICE_EXP,
            qty_exponent: BINANCE_QTY_EXP,
            venue: "binance".to_string(),
            ctx: CorrelationContext::default(),
        };

        let line = serde_json::to_string(&q)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        stats.events_written += 1;
    }

    file.flush().await?;

    Ok(CaptureResult {
        stats,
        connection_gaps: ws.connection_gaps().to_vec(),
        total_reconnects: ws.total_reconnects(),
    })
}
