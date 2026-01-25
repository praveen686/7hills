//! Binance Spot capture -> canonical QuoteEvent JSONL.
//!
//! Uses Binance Spot bookTicker stream: best bid/ask updates.
//! Output format uses canonical `quantlaxmi_models::events::QuoteEvent`.
//!
//! Notes:
//! - Public stream (no API keys required)
//! - Deterministic fixed-point parsing (no float intermediates)

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use futures_util::StreamExt;
use std::path::Path;
use tokio::io::AsyncWriteExt;

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

pub async fn capture_book_ticker_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<CaptureStats> {
    let sym = symbol.to_lowercase();
    let url = format!("wss://stream.binance.com:9443/ws/{}@bookTicker", sym);

    let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
        .await
        .with_context(|| format!("connect websocket: {}", url))?;

    let (_write, mut read) = ws_stream.split();

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(out_path)
        .await
        .with_context(|| format!("open output: {:?}", out_path))?;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(duration_secs);
    let mut stats = CaptureStats::default();

    const BINANCE_PRICE_EXP: i8 = -2; // cents
    const BINANCE_QTY_EXP: i8 = -8; // base size precision (sufficient for BTC/ETH; canonical)

    while tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), read.next()).await;
        let item = match msg {
            Ok(Some(v)) => v,
            Ok(None) => break,
            Err(_) => continue,
        };

        let msg = item?;
        if !msg.is_text() {
            continue;
        }
        let txt = msg.into_text()?;

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
    Ok(stats)
}
