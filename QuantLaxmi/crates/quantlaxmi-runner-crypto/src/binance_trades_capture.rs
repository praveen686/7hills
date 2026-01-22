//! Binance SBE Trades Capture → TradeEvent JSONL for Certified Replay.
//!
//! Uses Binance SBE (Simple Binary Encoding) websocket stream for trade data.
//! Trades are essential for:
//! - Microstructure validation (trade-through checks)
//! - Impact / slippage modeling
//! - VPIN and volume bucket features
//!
//! ## SBE Trade Stream
//! - URL: `wss://stream-sbe.binance.com:9443/stream`
//! - Stream: `{symbol}@trade`
//! - Template ID: 10000
//!
//! ## Output Format
//! TradeEvent JSONL with scaled integer mantissas for deterministic replay.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::io::AsyncWriteExt;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message;
use url::Url;

use kubera_models::IntegrityTier;
use kubera_sbe::{BinanceSbeDecoder, SBE_HEADER_SIZE, SbeHeader};

/// Trade event for certified replay.
/// Uses scaled integer mantissas to avoid f64 drift across platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    /// Exchange timestamp
    pub ts: DateTime<Utc>,
    /// Trading symbol (e.g., "BTCUSDT")
    pub tradingsymbol: String,
    /// Exchange trade ID
    pub trade_id: i64,
    /// Price mantissa (actual = mantissa * 10^price_exponent)
    pub price: i64,
    /// Quantity mantissa (actual = mantissa * 10^qty_exponent)
    pub qty: i64,
    /// Price exponent (e.g., -2 for 2 decimal places)
    pub price_exponent: i8,
    /// Quantity exponent (e.g., -8 for 8 decimal places)
    pub qty_exponent: i8,
    /// True if buyer was the market maker (seller was aggressor)
    pub is_buyer_maker: bool,
    /// Number of trades aggregated in this message
    pub trade_count: usize,
    /// Data integrity tier
    pub integrity_tier: IntegrityTier,
    /// Source identifier
    pub source: Option<String>,
}

/// Statistics from trade capture.
#[derive(Debug, Default)]
pub struct TradesCaptureStats {
    pub trades_written: usize,
    pub total_volume_mantissa: i64,
    pub buy_count: usize,
    pub sell_count: usize,
}

impl std::fmt::Display for TradesCaptureStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "trades={}, buys={}, sells={}, vol_mantissa={}",
            self.trades_written, self.buy_count, self.sell_count, self.total_volume_mantissa
        )
    }
}

/// Capture SBE trade stream to JSONL file.
///
/// # Arguments
/// * `symbol` - Trading pair (e.g., "BTCUSDT")
/// * `out_path` - Output file path
/// * `duration_secs` - Duration to capture
/// * `price_exponent` - Exponent for price (e.g., -2 for 2 decimal places)
/// * `qty_exponent` - Exponent for quantity (e.g., -8 for 8 decimal places)
/// * `api_key` - Binance API key (required for SBE stream)
pub async fn capture_sbe_trades_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
    price_exponent: i8,
    qty_exponent: i8,
    api_key: &str,
) -> Result<TradesCaptureStats> {
    let sym_upper = symbol.to_uppercase();
    let sym_lower = symbol.to_lowercase();

    // Connect to SBE stream
    let url_str = "wss://stream-sbe.binance.com:9443/stream";
    println!("Connecting to Binance SBE stream: {}", url_str);

    let url = Url::parse(url_str)?;
    let mut request = url.into_client_request()?;
    request
        .headers_mut()
        .insert("X-MBX-APIKEY", api_key.parse()?);
    request
        .headers_mut()
        .insert("Sec-WebSocket-Protocol", "binance-sbe".parse()?);

    let (ws_stream, _) = tokio_tungstenite::connect_async(request)
        .await
        .with_context(|| format!("connect SBE websocket: {}", url_str))?;

    println!("Connected to SBE stream");

    let (mut write, mut read) = ws_stream.split();

    // Subscribe to trade stream
    let sub = serde_json::json!({
        "method": "SUBSCRIBE",
        "params": [format!("{}@trade", sym_lower)],
        "id": 1
    });

    println!("Subscribing to {}@trade...", sym_lower);
    write.send(Message::Text(sub.to_string())).await?;

    // Wait for subscription confirmation
    let mut subscription_confirmed = false;
    let confirm_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while tokio::time::Instant::now() < confirm_deadline && !subscription_confirmed {
        let msg = tokio::time::timeout(std::time::Duration::from_millis(500), read.next()).await;
        if let Ok(Some(Ok(Message::Text(text)))) = msg {
            if text.contains("\"result\":null") {
                println!("Subscription confirmed");
                subscription_confirmed = true;
            } else if text.contains("\"error\"") {
                anyhow::bail!("Subscription error: {}", text);
            }
        }
    }

    if !subscription_confirmed {
        anyhow::bail!("Subscription not confirmed within 5 seconds");
    }

    // Open output file
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(out_path)
        .await
        .with_context(|| format!("open output: {:?}", out_path))?;

    let mut stats = TradesCaptureStats::default();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(duration_secs);

    // Process messages
    while tokio::time::Instant::now() < deadline {
        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), read.next()).await;
        let item = match msg {
            Ok(Some(v)) => v,
            Ok(None) => break,
            Err(_) => continue, // Timeout, retry
        };

        let msg = item?;

        match msg {
            Message::Binary(bin) => {
                if bin.len() < SBE_HEADER_SIZE {
                    continue;
                }

                let header = match SbeHeader::decode(&bin[..SBE_HEADER_SIZE]) {
                    Ok(h) => h,
                    Err(_) => continue,
                };

                // Only process trade messages (template 10000)
                if header.template_id != 10000 {
                    continue;
                }

                let agg_trade =
                    match BinanceSbeDecoder::decode_trade(&header, &bin[SBE_HEADER_SIZE..]) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("SBE trade decode error: {}", e);
                            continue;
                        }
                    };

                // Convert f64 to mantissa (deterministic from binary)
                let price_mantissa =
                    (agg_trade.price / 10f64.powi(price_exponent as i32)).round() as i64;
                let qty_mantissa =
                    (agg_trade.quantity / 10f64.powi(qty_exponent as i32)).round() as i64;

                let event = TradeEvent {
                    ts: agg_trade.exchange_time(),
                    tradingsymbol: sym_upper.clone(),
                    trade_id: 0, // AggTrade doesn't expose individual trade IDs
                    price: price_mantissa,
                    qty: qty_mantissa,
                    price_exponent,
                    qty_exponent,
                    is_buyer_maker: agg_trade.is_buyer_maker,
                    trade_count: agg_trade.trade_count,
                    integrity_tier: IntegrityTier::Certified,
                    source: Some("binance_sbe_trades_capture".to_string()),
                };

                let line = serde_json::to_string(&event)?;
                file.write_all(line.as_bytes()).await?;
                file.write_all(b"\n").await?;

                stats.trades_written += 1;
                stats.total_volume_mantissa += qty_mantissa;
                if agg_trade.is_buyer_maker {
                    stats.sell_count += 1; // Buyer is maker means seller was aggressor
                } else {
                    stats.buy_count += 1; // Buyer was aggressor
                }
            }
            Message::Text(text) => {
                if text.contains("\"error\"") {
                    eprintln!("Server error: {}", text);
                }
            }
            Message::Ping(p) => {
                let _ = write.send(Message::Pong(p)).await;
            }
            _ => {}
        }
    }

    file.flush().await?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_event_serialization() {
        let event = TradeEvent {
            ts: Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            trade_id: 123456789,
            price: 9000012,
            qty: 150000000,
            price_exponent: -2,
            qty_exponent: -8,
            is_buyer_maker: false,
            trade_count: 1,
            integrity_tier: IntegrityTier::Certified,
            source: Some("test".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: TradeEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.tradingsymbol, "BTCUSDT");
        assert_eq!(parsed.price, 9000012);
        assert_eq!(parsed.qty, 150000000);
        assert_eq!(parsed.price_exponent, -2);
        assert!(!parsed.is_buyer_maker);
    }

    #[test]
    fn test_mantissa_conversion() {
        // 90000.12 with exponent -2 should be 9000012
        let price = 90000.12f64;
        let exponent = -2i8;
        let mantissa = (price / 10f64.powi(exponent as i32)).round() as i64;
        assert_eq!(mantissa, 9000012);

        // 1.5 with exponent -8 should be 150000000
        let qty = 1.5f64;
        let exponent = -8i8;
        let mantissa = (qty / 10f64.powi(exponent as i32)).round() as i64;
        assert_eq!(mantissa, 150000000);
    }
}
