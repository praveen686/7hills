//! Binance Futures (Perp) depth and bookTicker capture.
//!
//! Captures USD-M Futures order book data from fstream.binance.com.
//! This provides the perp side of the funding arbitrage strategy.
//!
//! Streams:
//! - `@depth@100ms` - Order book diff updates (100ms intervals)
//! - `@bookTicker` - Best bid/ask updates (real-time)
//!
//! Endpoint: wss://fstream.binance.com/ws/<symbol>@<stream>
//!
//! Notes:
//! - No API key required for public streams
//! - Perp trades 24/7 (no market hours)
//! - Different from spot: uses fstream.binance.com instead of stream.binance.com

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::AsyncWriteExt;

use crate::fixed_point::parse_to_mantissa_pure;
use crate::quote::QuoteEvent;

/// Perp depth event (L2 order book update).
/// Uses the same structure as spot DepthEvent for compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpDepthEvent {
    /// Event timestamp
    pub ts: DateTime<Utc>,
    /// Symbol (e.g., BTCUSDT)
    pub tradingsymbol: String,
    /// Market type
    pub market: String,
    /// First update ID in event
    pub first_update_id: u64,
    /// Last update ID in event
    pub last_update_id: u64,
    /// Price exponent for mantissa conversion
    pub price_exponent: i8,
    /// Quantity exponent for mantissa conversion
    pub qty_exponent: i8,
    /// Bid levels (price, qty as mantissa)
    pub bids: Vec<DepthLevel>,
    /// Ask levels (price, qty as mantissa)
    pub asks: Vec<DepthLevel>,
    /// Is this a full snapshot?
    pub is_snapshot: bool,
    /// Source identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Depth level (price, qty as mantissa).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLevel {
    pub price: i64,
    pub qty: i64,
}

/// Raw depth update from Binance Futures WebSocket.
#[derive(Debug, Deserialize)]
struct FuturesDepthUpdate {
    /// Event type
    #[serde(rename = "e")]
    event_type: String,
    /// Event time (ms)
    #[serde(rename = "E")]
    event_time_ms: i64,
    /// Transaction time (ms) - required for parsing but not used directly
    #[allow(dead_code)]
    #[serde(rename = "T")]
    transaction_time_ms: i64,
    /// Symbol
    #[serde(rename = "s")]
    symbol: String,
    /// First update ID
    #[serde(rename = "U")]
    first_update_id: u64,
    /// Final update ID
    #[serde(rename = "u")]
    last_update_id: u64,
    /// Previous final update ID
    #[serde(rename = "pu")]
    prev_last_update_id: u64,
    /// Bids to update
    #[serde(rename = "b")]
    bids: Vec<(String, String)>,
    /// Asks to update
    #[serde(rename = "a")]
    asks: Vec<(String, String)>,
}

/// Raw bookTicker from Binance Futures WebSocket.
#[derive(Debug, Deserialize)]
struct FuturesBookTicker {
    /// Event type
    #[serde(rename = "e")]
    event_type: String,
    /// Update ID
    #[serde(rename = "u")]
    update_id: u64,
    /// Event time (ms)
    #[serde(rename = "E")]
    event_time_ms: i64,
    /// Transaction time (ms) - required for parsing but not used directly
    #[allow(dead_code)]
    #[serde(rename = "T")]
    transaction_time_ms: i64,
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

/// Parse string to mantissa using deterministic fixed-point (no float intermediates).
pub fn parse_to_mantissa(s: &str, exponent: i8) -> Result<i64> {
    parse_to_mantissa_pure(s, exponent)
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
}

/// Statistics from perp capture.
#[derive(Debug, Default)]
pub struct PerpCaptureStats {
    pub events_written: usize,
    pub depth_updates: usize,
    pub bookticker_updates: usize,
    pub sequence_gaps: usize,
    pub last_update_id: u64,
    pub last_bid: f64,
    pub last_ask: f64,
}

impl std::fmt::Display for PerpCaptureStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "events={}, depth={}, ticker={}, gaps={}, bid={:.2}, ask={:.2}",
            self.events_written,
            self.depth_updates,
            self.bookticker_updates,
            self.sequence_gaps,
            self.last_bid,
            self.last_ask
        )
    }
}

/// Capture perp bookTicker stream to canonical QuoteEvent JSONL.
/// This is the simplest capture mode - just best bid/ask.
pub async fn capture_perp_bookticker_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<PerpCaptureStats> {
    let sym_lower = symbol.to_lowercase();
    let url = format!("wss://fstream.binance.com/ws/{}@bookTicker", sym_lower);

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
    let mut stats = PerpCaptureStats::default();

    const BINANCE_PRICE_EXP: i8 = -2; // cents
    const BINANCE_QTY_EXP: i8 = -8; // base size precision

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

        let ev: FuturesBookTicker = match serde_json::from_str::<FuturesBookTicker>(&txt) {
            Ok(e) if e.event_type == "bookTicker" => e,
            _ => continue,
        };

        let bid_m = parse_to_mantissa(&ev.bid_price, BINANCE_PRICE_EXP)?;
        let ask_m = parse_to_mantissa(&ev.ask_price, BINANCE_PRICE_EXP)?;

        stats.last_bid = bid_m as f64 * 10f64.powi(BINANCE_PRICE_EXP as i32);
        stats.last_ask = ask_m as f64 * 10f64.powi(BINANCE_PRICE_EXP as i32);
        stats.last_update_id = ev.update_id;
        stats.bookticker_updates += 1;

        let quote = QuoteEvent {
            ts: ms_to_dt(ev.event_time_ms),
            symbol: ev.symbol,
            bid_price_mantissa: bid_m,
            ask_price_mantissa: ask_m,
            bid_qty_mantissa: parse_to_mantissa(&ev.bid_qty, BINANCE_QTY_EXP)?,
            ask_qty_mantissa: parse_to_mantissa(&ev.ask_qty, BINANCE_QTY_EXP)?,
            price_exponent: BINANCE_PRICE_EXP,
            qty_exponent: BINANCE_QTY_EXP,
        };

        let line = serde_json::to_string(&quote)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        stats.events_written += 1;
    }

    file.flush().await?;
    Ok(stats)
}

/// Capture perp depth stream to PerpDepthEvent JSONL.
/// This captures full L2 order book updates.
pub async fn capture_perp_depth_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
) -> Result<PerpCaptureStats> {
    let sym_lower = symbol.to_lowercase();
    // Use 100ms update speed for depth
    let url = format!("wss://fstream.binance.com/ws/{}@depth@100ms", sym_lower);

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
    let mut stats = PerpCaptureStats::default();

    // First, fetch REST snapshot for bootstrap
    let snapshot_url = format!(
        "https://fapi.binance.com/fapi/v1/depth?symbol={}&limit=1000",
        symbol.to_uppercase()
    );
    let snapshot_resp: serde_json::Value = reqwest::get(&snapshot_url)
        .await
        .context("fetch perp depth snapshot")?
        .json()
        .await
        .context("parse perp depth snapshot")?;

    let snapshot_last_id = snapshot_resp["lastUpdateId"].as_u64().unwrap_or(0);

    // Parse and write snapshot
    let snapshot_bids: Vec<DepthLevel> = snapshot_resp["bids"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|b| {
            let price = parse_to_mantissa(b[0].as_str()?, price_exponent).ok()?;
            let qty = parse_to_mantissa(b[1].as_str()?, qty_exponent).ok()?;
            Some(DepthLevel { price, qty })
        })
        .collect();

    let snapshot_asks: Vec<DepthLevel> = snapshot_resp["asks"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|a| {
            let price = parse_to_mantissa(a[0].as_str()?, price_exponent).ok()?;
            let qty = parse_to_mantissa(a[1].as_str()?, qty_exponent).ok()?;
            Some(DepthLevel { price, qty })
        })
        .collect();

    let snapshot_event = PerpDepthEvent {
        ts: Utc::now(),
        tradingsymbol: symbol.to_uppercase(),
        market: "perp".to_string(),
        first_update_id: snapshot_last_id,
        last_update_id: snapshot_last_id,
        price_exponent,
        qty_exponent,
        bids: snapshot_bids,
        asks: snapshot_asks,
        is_snapshot: true,
        source: Some("binance_perp_depth_capture".to_string()),
    };

    let line = serde_json::to_string(&snapshot_event)?;
    file.write_all(line.as_bytes()).await?;
    file.write_all(b"\n").await?;
    stats.events_written += 1;
    let mut prev_last_update_id = snapshot_last_id;

    // Buffer updates until we find sync point
    let mut buffer: Vec<FuturesDepthUpdate> = Vec::new();
    let mut synced = false;

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

        let ev: FuturesDepthUpdate = match serde_json::from_str::<FuturesDepthUpdate>(&txt) {
            Ok(e) if e.event_type == "depthUpdate" => e,
            _ => continue,
        };

        // Binance Futures sync logic:
        // Drop updates where u < lastUpdateId
        // First valid update: U <= lastUpdateId+1 AND u >= lastUpdateId
        if !synced {
            if ev.last_update_id < snapshot_last_id {
                continue; // Drop old update
            }
            if ev.first_update_id <= snapshot_last_id + 1 && ev.last_update_id >= snapshot_last_id {
                synced = true;
                tracing::info!(
                    "Perp depth synced at update_id={} (snapshot={})",
                    ev.first_update_id,
                    snapshot_last_id
                );
            } else {
                buffer.push(ev);
                continue;
            }
        }

        // Check for sequence gaps
        if prev_last_update_id != 0 && ev.prev_last_update_id != prev_last_update_id {
            tracing::warn!(
                "Perp depth sequence gap: expected {}, got prev={}",
                prev_last_update_id,
                ev.prev_last_update_id
            );
            stats.sequence_gaps += 1;
        }
        prev_last_update_id = ev.last_update_id;

        // Convert to PerpDepthEvent
        let bids: Vec<DepthLevel> = ev
            .bids
            .iter()
            .filter_map(|(p, q)| {
                let price = parse_to_mantissa(p, price_exponent).ok()?;
                let qty = parse_to_mantissa(q, qty_exponent).ok()?;
                Some(DepthLevel { price, qty })
            })
            .collect();

        let asks: Vec<DepthLevel> = ev
            .asks
            .iter()
            .filter_map(|(p, q)| {
                let price = parse_to_mantissa(p, price_exponent).ok()?;
                let qty = parse_to_mantissa(q, qty_exponent).ok()?;
                Some(DepthLevel { price, qty })
            })
            .collect();

        // Track best bid/ask
        if let Some(best_bid) = bids.iter().filter(|l| l.qty > 0).max_by_key(|l| l.price) {
            stats.last_bid = best_bid.price as f64 * 10f64.powi(price_exponent as i32);
        }
        if let Some(best_ask) = asks.iter().filter(|l| l.qty > 0).min_by_key(|l| l.price) {
            stats.last_ask = best_ask.price as f64 * 10f64.powi(price_exponent as i32);
        }

        let depth_event = PerpDepthEvent {
            ts: ms_to_dt(ev.event_time_ms),
            tradingsymbol: ev.symbol,
            market: "perp".to_string(),
            first_update_id: ev.first_update_id,
            last_update_id: ev.last_update_id,
            price_exponent,
            qty_exponent,
            bids,
            asks,
            is_snapshot: false,
            source: Some("binance_perp_depth_capture".to_string()),
        };

        let line = serde_json::to_string(&depth_event)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        stats.events_written += 1;
        stats.depth_updates += 1;
        stats.last_update_id = ev.last_update_id;
    }

    file.flush().await?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mantissa_parsing() {
        assert_eq!(parse_to_mantissa("90000.12", -2).unwrap(), 9000012);
        assert_eq!(parse_to_mantissa("0.00012345", -8).unwrap(), 12345);
        assert_eq!(parse_to_mantissa("100000", -2).unwrap(), 10000000);
    }

    #[test]
    fn test_perp_depth_event_serialization() {
        let event = PerpDepthEvent {
            ts: Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            market: "perp".to_string(),
            first_update_id: 1000,
            last_update_id: 1001,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![DepthLevel {
                price: 9000000,
                qty: 100000000,
            }],
            asks: vec![DepthLevel {
                price: 9000100,
                qty: 50000000,
            }],
            is_snapshot: false,
            source: Some("test".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("BTCUSDT"));
        assert!(json.contains("\"market\":\"perp\""));

        let parsed: PerpDepthEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.market, "perp");
        assert_eq!(parsed.bids.len(), 1);
    }
}
