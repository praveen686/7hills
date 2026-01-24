//! Binance Futures funding rate capture -> FundingEvent JSONL.
//!
//! Captures the markPrice stream from Binance USD-M Futures.
//! This stream provides:
//! - Mark price (fair price used for funding/liquidation)
//! - Index price (spot reference)
//! - Funding rate (predicted 8h rate)
//! - Next funding time
//!
//! Endpoint: wss://fstream.binance.com/ws/<symbol>@markPrice
//!
//! Notes:
//! - No API key required for public streams
//! - Funding settles every 8h: 00:00, 08:00, 16:00 UTC
//! - Rate updates every ~3 seconds

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Funding event from markPrice stream.
/// This is the core data structure for funding rate arbitrage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingEvent {
    /// Event timestamp
    pub ts: DateTime<Utc>,
    /// Symbol (e.g., BTCUSDT)
    pub symbol: String,
    /// Mark price (fair price for funding/liquidation)
    pub mark_price: f64,
    /// Index price (spot reference)
    pub index_price: f64,
    /// Estimated settle price (only meaningful near funding time)
    pub estimated_settle_price: f64,
    /// Current predicted funding rate (8h rate as decimal, e.g., 0.0001 = 0.01%)
    pub funding_rate: f64,
    /// Next funding timestamp
    pub next_funding_time: DateTime<Utc>,
    /// Source identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Raw markPrice WebSocket event from Binance Futures.
#[derive(Debug, Deserialize)]
struct MarkPriceEvent {
    /// Event type
    #[serde(rename = "e")]
    event_type: String,
    /// Event time (ms)
    #[serde(rename = "E")]
    event_time_ms: i64,
    /// Symbol
    #[serde(rename = "s")]
    symbol: String,
    /// Mark price
    #[serde(rename = "p")]
    mark_price: String,
    /// Index price
    #[serde(rename = "i")]
    index_price: String,
    /// Estimated settle price
    #[serde(rename = "P")]
    estimated_settle_price: String,
    /// Funding rate
    #[serde(rename = "r")]
    funding_rate: String,
    /// Next funding time (ms)
    #[serde(rename = "T")]
    next_funding_time_ms: i64,
}

fn parse_f64(s: &str) -> Result<f64> {
    s.parse::<f64>()
        .with_context(|| format!("parse f64: {}", s))
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
}

/// Statistics from funding capture.
#[derive(Debug, Default)]
pub struct FundingCaptureStats {
    pub events_written: usize,
    pub funding_events: usize,
    pub min_funding_rate: f64,
    pub max_funding_rate: f64,
    pub last_funding_rate: f64,
    pub last_mark_price: f64,
    pub funding_settlements: usize,
}

impl std::fmt::Display for FundingCaptureStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "events={}, funding_range=[{:.4}%..{:.4}%], last_rate={:.4}%, settlements={}",
            self.events_written,
            self.min_funding_rate * 100.0,
            self.max_funding_rate * 100.0,
            self.last_funding_rate * 100.0,
            self.funding_settlements
        )
    }
}

/// Capture funding rate stream to JSONL.
pub async fn capture_funding_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<FundingCaptureStats> {
    let sym_lower = symbol.to_lowercase();
    let url = format!("wss://fstream.binance.com/ws/{}@markPrice", sym_lower);

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
    let mut stats = FundingCaptureStats {
        min_funding_rate: f64::MAX,
        max_funding_rate: f64::MIN,
        ..Default::default()
    };
    let mut last_funding_time_ms: i64 = 0;

    while tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(10), read.next()).await;
        let item = match msg {
            Ok(Some(v)) => v,
            Ok(None) => break,
            Err(_) => continue, // Timeout, keep waiting
        };

        let msg = item?;
        if !msg.is_text() {
            continue;
        }
        let txt = msg.into_text()?;

        // Parse markPrice event
        let ev: MarkPriceEvent = match serde_json::from_str::<MarkPriceEvent>(&txt) {
            Ok(e) if e.event_type == "markPriceUpdate" => e,
            _ => continue,
        };

        let funding_rate = parse_f64(&ev.funding_rate)?;
        let mark_price = parse_f64(&ev.mark_price)?;
        let index_price = parse_f64(&ev.index_price)?;
        let estimated_settle = parse_f64(&ev.estimated_settle_price)?;

        // Track funding settlements (when next_funding_time changes)
        if last_funding_time_ms != 0 && ev.next_funding_time_ms != last_funding_time_ms {
            stats.funding_settlements += 1;
        }
        last_funding_time_ms = ev.next_funding_time_ms;

        // Update stats
        stats.min_funding_rate = stats.min_funding_rate.min(funding_rate);
        stats.max_funding_rate = stats.max_funding_rate.max(funding_rate);
        stats.last_funding_rate = funding_rate;
        stats.last_mark_price = mark_price;
        stats.funding_events += 1;

        let funding_event = FundingEvent {
            ts: ms_to_dt(ev.event_time_ms),
            symbol: ev.symbol,
            mark_price,
            index_price,
            estimated_settle_price: estimated_settle,
            funding_rate,
            next_funding_time: ms_to_dt(ev.next_funding_time_ms),
            source: Some("binance_funding_capture".to_string()),
        };

        let line = serde_json::to_string(&funding_event)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        stats.events_written += 1;
    }

    file.flush().await?;

    // Fix stats if no events
    if stats.min_funding_rate == f64::MAX {
        stats.min_funding_rate = 0.0;
    }
    if stats.max_funding_rate == f64::MIN {
        stats.max_funding_rate = 0.0;
    }

    Ok(stats)
}

/// Capture funding for multiple symbols in parallel.
pub async fn capture_multi_funding_jsonl(
    symbols: &[String],
    out_dir: &Path,
    duration_secs: u64,
) -> Result<Vec<(String, FundingCaptureStats)>> {
    use futures::future::join_all;

    let tasks: Vec<_> = symbols
        .iter()
        .map(|sym| {
            let sym = sym.clone();
            let out_path = out_dir.join(format!("{}_funding.jsonl", sym.to_uppercase()));
            async move {
                let stats = capture_funding_jsonl(&sym, &out_path, duration_secs).await?;
                Ok::<_, anyhow::Error>((sym, stats))
            }
        })
        .collect();

    let results = join_all(tasks).await;
    let mut all_stats = Vec::new();
    for result in results {
        match result {
            Ok((sym, stats)) => all_stats.push((sym, stats)),
            Err(e) => tracing::warn!("Funding capture failed: {}", e),
        }
    }

    Ok(all_stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_event_serialization() {
        let event = FundingEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            mark_price: 100000.50,
            index_price: 100000.25,
            estimated_settle_price: 100000.40,
            funding_rate: 0.0001,
            next_funding_time: Utc::now() + chrono::Duration::hours(8),
            source: Some("test".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("BTCUSDT"));
        assert!(json.contains("funding_rate"));

        let parsed: FundingEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.symbol, "BTCUSDT");
        assert!((parsed.funding_rate - 0.0001).abs() < 1e-10);
    }
}
