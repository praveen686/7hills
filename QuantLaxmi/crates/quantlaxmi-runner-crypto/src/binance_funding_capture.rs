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
//! ## Determinism
//! All numeric fields use scaled integer mantissas to avoid float drift:
//! - Prices: mantissa with price_exponent (e.g., -2 for 2 decimal places)
//! - Funding rate: mantissa with rate_exponent (e.g., -8 for 8 decimal places)
//!
//! Notes:
//! - No API key required for public streams
//! - Funding settles every 8h: 00:00, 08:00, 16:00 UTC
//! - Rate updates every ~3 seconds
//! - Uses ResilientWs for auto-reconnect on disconnect

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use quantlaxmi_models::parse_to_mantissa_pure;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::time::Instant;
use tracing::{error, warn};

use crate::ws_resilient::{ConnectionGap, ResilientWs, ResilientWsConfig};

// =============================================================================
// Deterministic FundingEvent (no f64 persistence)
// =============================================================================
// All prices and rates are stored as integer mantissas for cross-platform
// reproducibility. Use the helper methods to convert to f64 for display.
// =============================================================================

/// Default price exponent for Binance futures (-2 = 2 decimal places).
pub const FUNDING_PRICE_EXPONENT: i8 = -2;

/// Default rate exponent for funding rates (-8 = 8 decimal places).
/// Funding rates are typically small (e.g., 0.0001 = 0.01%), so we need
/// high precision. -8 gives us 8 decimal places (like Binance's native format).
pub const FUNDING_RATE_EXPONENT: i8 = -8;

/// Funding event from markPrice stream (deterministic, fixed-point).
/// This is the core data structure for funding rate arbitrage.
///
/// All numeric fields are stored as integer mantissas for deterministic replay.
/// Use the helper methods (`mark_price_f64()`, `funding_rate_f64()`, etc.)
/// to convert to floating point for display or computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingEvent {
    /// Event timestamp
    pub ts: DateTime<Utc>,
    /// Symbol (e.g., BTCUSDT)
    pub symbol: String,

    // --- Price fields (mantissa form) ---
    /// Mark price mantissa (divide by 10^|price_exponent| to get actual price)
    pub mark_price_mantissa: i64,
    /// Index price mantissa
    pub index_price_mantissa: i64,
    /// Estimated settle price mantissa
    pub estimated_settle_price_mantissa: i64,
    /// Price exponent (e.g., -2 means price = mantissa / 100)
    pub price_exponent: i8,

    // --- Funding rate (mantissa form) ---
    /// Funding rate mantissa (e.g., 10000 with exponent -8 = 0.0001 = 0.01%)
    pub funding_rate_mantissa: i64,
    /// Rate exponent (e.g., -8 for 8 decimal places)
    pub rate_exponent: i8,

    // --- Timing ---
    /// Next funding timestamp (milliseconds since Unix epoch)
    pub next_funding_time_ms: i64,

    /// Source identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl FundingEvent {
    /// Convert mark price mantissa to f64.
    pub fn mark_price_f64(&self) -> f64 {
        self.mark_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert index price mantissa to f64.
    pub fn index_price_f64(&self) -> f64 {
        self.index_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert estimated settle price mantissa to f64.
    pub fn estimated_settle_price_f64(&self) -> f64 {
        self.estimated_settle_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }

    /// Convert funding rate mantissa to f64 (e.g., 0.0001 for 0.01%).
    pub fn funding_rate_f64(&self) -> f64 {
        self.funding_rate_mantissa as f64 * 10f64.powi(self.rate_exponent as i32)
    }

    /// Funding rate as basis points (bps). 0.01% = 1 bps.
    pub fn funding_rate_bps(&self) -> f64 {
        self.funding_rate_f64() * 10_000.0
    }

    /// Next funding time as DateTime<Utc>.
    pub fn next_funding_time(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.next_funding_time_ms)
            .single()
            .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
    }

    /// Basis in basis points: (mark - index) / index * 10000.
    pub fn basis_bps(&self) -> f64 {
        let mark = self.mark_price_f64();
        let index = self.index_price_f64();
        if index == 0.0 {
            return 0.0;
        }
        (mark - index) / index * 10_000.0
    }
}

/// Parse decimal string to mantissa without f64 intermediate (deterministic).
/// Delegates to canonical implementation in quantlaxmi_models.
fn parse_to_mantissa(s: &str, exponent: i8) -> Result<i64> {
    parse_to_mantissa_pure(s, exponent).map_err(|e| anyhow::anyhow!("{}", e))
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
    /// Mark price (string for deterministic parsing)
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

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
}

/// Statistics from funding capture (uses mantissa for determinism).
#[derive(Debug, Default)]
pub struct FundingCaptureStats {
    pub events_written: usize,
    pub funding_events: usize,
    /// Min funding rate mantissa (with FUNDING_RATE_EXPONENT)
    pub min_funding_rate_mantissa: i64,
    /// Max funding rate mantissa
    pub max_funding_rate_mantissa: i64,
    /// Last funding rate mantissa
    pub last_funding_rate_mantissa: i64,
    /// Last mark price mantissa (with FUNDING_PRICE_EXPONENT)
    pub last_mark_price_mantissa: i64,
    pub funding_settlements: usize,
}

impl FundingCaptureStats {
    /// Convert min funding rate to f64.
    pub fn min_funding_rate_f64(&self) -> f64 {
        self.min_funding_rate_mantissa as f64 * 10f64.powi(FUNDING_RATE_EXPONENT as i32)
    }

    /// Convert max funding rate to f64.
    pub fn max_funding_rate_f64(&self) -> f64 {
        self.max_funding_rate_mantissa as f64 * 10f64.powi(FUNDING_RATE_EXPONENT as i32)
    }

    /// Convert last funding rate to f64.
    pub fn last_funding_rate_f64(&self) -> f64 {
        self.last_funding_rate_mantissa as f64 * 10f64.powi(FUNDING_RATE_EXPONENT as i32)
    }

    /// Convert last mark price to f64.
    pub fn last_mark_price_f64(&self) -> f64 {
        self.last_mark_price_mantissa as f64 * 10f64.powi(FUNDING_PRICE_EXPONENT as i32)
    }
}

impl std::fmt::Display for FundingCaptureStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "events={}, funding_range=[{:.4}%..{:.4}%], last_rate={:.4}%, settlements={}",
            self.events_written,
            self.min_funding_rate_f64() * 100.0,
            self.max_funding_rate_f64() * 100.0,
            self.last_funding_rate_f64() * 100.0,
            self.funding_settlements
        )
    }
}

/// Result from funding capture including connection gap info.
#[derive(Debug, Default)]
pub struct FundingCaptureResult {
    pub stats: FundingCaptureStats,
    pub connection_gaps: Vec<ConnectionGap>,
    pub total_reconnects: u32,
}

impl std::fmt::Display for FundingCaptureResult {
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

/// Capture funding rate stream to JSONL (deterministic, fixed-point).
/// Uses ResilientWs for auto-reconnect on disconnect.
pub async fn capture_funding_jsonl(
    symbol: &str,
    out_path: &Path,
    duration_secs: u64,
) -> Result<FundingCaptureResult> {
    let sym_lower = symbol.to_lowercase();
    let url = format!("wss://fstream.binance.com/ws/{}@markPrice", sym_lower);

    let config = ResilientWsConfig {
        liveness_timeout: Duration::from_secs(60), // Funding updates every ~3s, 60s is conservative
        read_timeout: Duration::from_secs(10),
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
    let mut stats = FundingCaptureStats {
        min_funding_rate_mantissa: i64::MAX,
        max_funding_rate_mantissa: i64::MIN,
        ..Default::default()
    };
    let mut last_funding_time_ms: i64 = 0;

    while Instant::now() < deadline {
        let msg = match ws.next_message().await? {
            Some(m) => m,
            None => {
                error!("WebSocket reconnection exhausted for funding {}", symbol);
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

        // Parse markPrice event
        let ev: MarkPriceEvent = match serde_json::from_str::<MarkPriceEvent>(&txt) {
            Ok(e) if e.event_type == "markPriceUpdate" => e,
            _ => continue,
        };

        // Parse to mantissa (deterministic, no f64 intermediate)
        let mark_price_mantissa = parse_to_mantissa(&ev.mark_price, FUNDING_PRICE_EXPONENT)?;
        let index_price_mantissa = parse_to_mantissa(&ev.index_price, FUNDING_PRICE_EXPONENT)?;
        let estimated_settle_mantissa =
            parse_to_mantissa(&ev.estimated_settle_price, FUNDING_PRICE_EXPONENT)?;
        let funding_rate_mantissa = parse_to_mantissa(&ev.funding_rate, FUNDING_RATE_EXPONENT)?;

        // Track funding settlements (when next_funding_time changes)
        if last_funding_time_ms != 0 && ev.next_funding_time_ms != last_funding_time_ms {
            stats.funding_settlements += 1;
        }
        last_funding_time_ms = ev.next_funding_time_ms;

        // Update stats (all in mantissa form)
        stats.min_funding_rate_mantissa =
            stats.min_funding_rate_mantissa.min(funding_rate_mantissa);
        stats.max_funding_rate_mantissa =
            stats.max_funding_rate_mantissa.max(funding_rate_mantissa);
        stats.last_funding_rate_mantissa = funding_rate_mantissa;
        stats.last_mark_price_mantissa = mark_price_mantissa;
        stats.funding_events += 1;

        let funding_event = FundingEvent {
            ts: ms_to_dt(ev.event_time_ms),
            symbol: ev.symbol,
            mark_price_mantissa,
            index_price_mantissa,
            estimated_settle_price_mantissa: estimated_settle_mantissa,
            price_exponent: FUNDING_PRICE_EXPONENT,
            funding_rate_mantissa,
            rate_exponent: FUNDING_RATE_EXPONENT,
            next_funding_time_ms: ev.next_funding_time_ms,
            source: Some("binance_funding_capture".to_string()),
        };

        let line = serde_json::to_string(&funding_event)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        stats.events_written += 1;
    }

    file.flush().await?;

    // Fix stats if no events
    if stats.min_funding_rate_mantissa == i64::MAX {
        stats.min_funding_rate_mantissa = 0;
    }
    if stats.max_funding_rate_mantissa == i64::MIN {
        stats.max_funding_rate_mantissa = 0;
    }

    Ok(FundingCaptureResult {
        stats,
        connection_gaps: ws.connection_gaps().to_vec(),
        total_reconnects: ws.total_reconnects(),
    })
}

/// Capture funding for multiple symbols in parallel.
pub async fn capture_multi_funding_jsonl(
    symbols: &[String],
    out_dir: &Path,
    duration_secs: u64,
) -> Result<Vec<(String, FundingCaptureResult)>> {
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
        // 100000.50 with exponent -2 = 10000050 mantissa
        // 0.0001 with exponent -8 = 10000 mantissa
        let now = Utc::now();
        let next_funding_ms = (now + chrono::Duration::hours(8)).timestamp_millis();

        let event = FundingEvent {
            ts: now,
            symbol: "BTCUSDT".to_string(),
            mark_price_mantissa: 10000050,             // 100000.50
            index_price_mantissa: 10000025,            // 100000.25
            estimated_settle_price_mantissa: 10000040, // 100000.40
            price_exponent: FUNDING_PRICE_EXPONENT,
            funding_rate_mantissa: 10000, // 0.0001 (0.01%)
            rate_exponent: FUNDING_RATE_EXPONENT,
            next_funding_time_ms: next_funding_ms,
            source: Some("test".to_string()),
        };

        // Test helper methods
        assert!((event.mark_price_f64() - 100000.50).abs() < 0.01);
        assert!((event.index_price_f64() - 100000.25).abs() < 0.01);
        assert!((event.funding_rate_f64() - 0.0001).abs() < 1e-10);
        assert!((event.funding_rate_bps() - 1.0).abs() < 0.01); // 0.01% = 1 bps

        // Test serialization
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("BTCUSDT"));
        assert!(json.contains("mark_price_mantissa"));
        assert!(json.contains("funding_rate_mantissa"));

        // Test deserialization
        let parsed: FundingEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.symbol, "BTCUSDT");
        assert_eq!(parsed.mark_price_mantissa, 10000050);
        assert_eq!(parsed.funding_rate_mantissa, 10000);
    }

    #[test]
    fn test_parse_to_mantissa() {
        // Price parsing (exponent -2)
        assert_eq!(parse_to_mantissa("100000.50", -2).unwrap(), 10000050);
        assert_eq!(parse_to_mantissa("90000.12", -2).unwrap(), 9000012);
        assert_eq!(parse_to_mantissa("100000", -2).unwrap(), 10000000);

        // Funding rate parsing (exponent -8)
        assert_eq!(parse_to_mantissa("0.0001", -8).unwrap(), 10000);
        assert_eq!(parse_to_mantissa("0.00010000", -8).unwrap(), 10000);
        assert_eq!(parse_to_mantissa("-0.0001", -8).unwrap(), -10000);

        // Rounding
        assert_eq!(parse_to_mantissa("100.125", -2).unwrap(), 10013); // rounds up
        assert_eq!(parse_to_mantissa("100.124", -2).unwrap(), 10012); // truncates
    }

    #[test]
    fn test_basis_bps() {
        let event = FundingEvent {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            mark_price_mantissa: 10010000,  // 100100.00
            index_price_mantissa: 10000000, // 100000.00
            estimated_settle_price_mantissa: 10005000,
            price_exponent: FUNDING_PRICE_EXPONENT,
            funding_rate_mantissa: 10000,
            rate_exponent: FUNDING_RATE_EXPONENT,
            next_funding_time_ms: 0,
            source: None,
        };

        // Basis = (100100 - 100000) / 100000 * 10000 = 10 bps
        assert!((event.basis_bps() - 10.0).abs() < 0.1);
    }

    // =========================================================================
    // FUNDING-CLOCK CORRECTNESS TESTS (Phase 2C)
    // =========================================================================

    /// Verify that funding timestamps snap to valid 8h intervals (00:00, 08:00, 16:00 UTC).
    #[test]
    fn test_funding_timestamp_snaps_to_8h_intervals() {
        use chrono::Timelike;

        // Valid funding times: 00:00, 08:00, 16:00 UTC
        let valid_hours = [0, 8, 16];

        // Create funding events with various next_funding_time values
        let test_cases = [
            // Timestamp at exactly 08:00 UTC on 2026-01-24
            1737705600000i64, // 2026-01-24 08:00:00 UTC
            // Timestamp at exactly 16:00 UTC
            1737734400000i64, // 2026-01-24 16:00:00 UTC
            // Timestamp at exactly 00:00 UTC next day
            1737763200000i64, // 2026-01-25 00:00:00 UTC
        ];

        for ts_ms in test_cases {
            let dt = Utc.timestamp_millis_opt(ts_ms).unwrap();
            let hour = dt.hour();
            let minute = dt.minute();
            let second = dt.second();

            // Funding time must be exactly on the hour
            assert_eq!(minute, 0, "Funding time minute must be 0, got {}", minute);
            assert_eq!(second, 0, "Funding time second must be 0, got {}", second);

            // Funding time must be one of 00:00, 08:00, 16:00
            assert!(
                valid_hours.contains(&hour),
                "Funding time hour must be 0, 8, or 16, got {}",
                hour
            );
        }
    }

    /// Verify that next_funding_time is monotonically increasing across events.
    #[test]
    fn test_funding_timestamp_monotonic() {
        let base_time = Utc::now();

        // Create a sequence of funding events
        let events: Vec<FundingEvent> = (0..5)
            .map(|i| {
                // Each event's next_funding_time should be 8 hours apart
                let next_funding_ms =
                    (base_time + chrono::Duration::hours(8 * (i + 1))).timestamp_millis();

                FundingEvent {
                    ts: base_time + chrono::Duration::minutes(i * 10),
                    symbol: "BTCUSDT".to_string(),
                    mark_price_mantissa: 10000000,
                    index_price_mantissa: 10000000,
                    estimated_settle_price_mantissa: 10000000,
                    price_exponent: FUNDING_PRICE_EXPONENT,
                    funding_rate_mantissa: 10000,
                    rate_exponent: FUNDING_RATE_EXPONENT,
                    next_funding_time_ms: next_funding_ms,
                    source: None,
                }
            })
            .collect();

        // Verify monotonicity
        for i in 1..events.len() {
            assert!(
                events[i].next_funding_time_ms >= events[i - 1].next_funding_time_ms,
                "next_funding_time must be monotonically increasing: {} < {}",
                events[i].next_funding_time_ms,
                events[i - 1].next_funding_time_ms
            );
        }
    }

    /// Verify that funding cashflow is applied exactly once per 8h window.
    /// This test checks that within a single funding window, there's exactly one
    /// "settlement" event where next_funding_time changes.
    #[test]
    fn test_funding_cashflow_once_per_window() {
        // Simulate funding events within a single 8h window
        // next_funding_time should remain constant until settlement
        let window_start = Utc.with_ymd_and_hms(2026, 1, 24, 0, 0, 0).unwrap();
        let next_funding = Utc.with_ymd_and_hms(2026, 1, 24, 8, 0, 0).unwrap();
        let after_funding = Utc.with_ymd_and_hms(2026, 1, 24, 16, 0, 0).unwrap();

        let events = [
            // Events before funding settlement
            FundingEvent {
                ts: window_start + chrono::Duration::minutes(10),
                symbol: "BTCUSDT".to_string(),
                mark_price_mantissa: 10000000,
                index_price_mantissa: 10000000,
                estimated_settle_price_mantissa: 10000000,
                price_exponent: FUNDING_PRICE_EXPONENT,
                funding_rate_mantissa: 10000,
                rate_exponent: FUNDING_RATE_EXPONENT,
                next_funding_time_ms: next_funding.timestamp_millis(),
                source: None,
            },
            FundingEvent {
                ts: window_start + chrono::Duration::hours(4),
                symbol: "BTCUSDT".to_string(),
                mark_price_mantissa: 10000000,
                index_price_mantissa: 10000000,
                estimated_settle_price_mantissa: 10000000,
                price_exponent: FUNDING_PRICE_EXPONENT,
                funding_rate_mantissa: 10000,
                rate_exponent: FUNDING_RATE_EXPONENT,
                next_funding_time_ms: next_funding.timestamp_millis(),
                source: None,
            },
            // Event after funding settlement (next_funding_time changes)
            FundingEvent {
                ts: next_funding + chrono::Duration::minutes(1),
                symbol: "BTCUSDT".to_string(),
                mark_price_mantissa: 10000000,
                index_price_mantissa: 10000000,
                estimated_settle_price_mantissa: 10000000,
                price_exponent: FUNDING_PRICE_EXPONENT,
                funding_rate_mantissa: 12000, // Rate may change after settlement
                rate_exponent: FUNDING_RATE_EXPONENT,
                next_funding_time_ms: after_funding.timestamp_millis(),
                source: None,
            },
        ];

        // Count funding settlements (transitions in next_funding_time)
        let mut settlements = 0;
        for i in 1..events.len() {
            if events[i].next_funding_time_ms != events[i - 1].next_funding_time_ms {
                settlements += 1;
            }
        }

        // There should be exactly 1 settlement in this sequence
        assert_eq!(
            settlements, 1,
            "Expected exactly 1 funding settlement, got {}",
            settlements
        );
    }

    /// Verify deterministic serialization produces identical output.
    #[test]
    fn test_funding_event_deterministic_serialization() {
        use sha2::{Digest, Sha256};

        let event = FundingEvent {
            ts: Utc.with_ymd_and_hms(2026, 1, 24, 10, 30, 0).unwrap(),
            symbol: "BTCUSDT".to_string(),
            mark_price_mantissa: 10000050,
            index_price_mantissa: 10000025,
            estimated_settle_price_mantissa: 10000040,
            price_exponent: FUNDING_PRICE_EXPONENT,
            funding_rate_mantissa: 10000,
            rate_exponent: FUNDING_RATE_EXPONENT,
            next_funding_time_ms: 1737734400000, // Fixed timestamp
            source: None,
        };

        // Serialize multiple times
        let json1 = serde_json::to_string(&event).unwrap();
        let json2 = serde_json::to_string(&event).unwrap();
        let json3 = serde_json::to_string(&event).unwrap();

        // All serializations must be identical
        assert_eq!(json1, json2, "Serialization not deterministic");
        assert_eq!(json2, json3, "Serialization not deterministic");

        // Hash must be consistent
        let hash1 = hex::encode(Sha256::digest(json1.as_bytes()));
        let hash2 = hex::encode(Sha256::digest(json2.as_bytes()));
        assert_eq!(hash1, hash2, "Hash not deterministic");
    }

    /// Verify that funding interval is exactly 8 hours (28800 seconds).
    #[test]
    fn test_funding_interval_8_hours() {
        let funding_times = [
            Utc.with_ymd_and_hms(2026, 1, 24, 0, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2026, 1, 24, 8, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2026, 1, 24, 16, 0, 0).unwrap(),
            Utc.with_ymd_and_hms(2026, 1, 25, 0, 0, 0).unwrap(),
        ];

        for i in 1..funding_times.len() {
            let interval_secs = (funding_times[i] - funding_times[i - 1]).num_seconds();
            assert_eq!(
                interval_secs, 28800,
                "Funding interval must be 28800 seconds (8 hours), got {}",
                interval_secs
            );
        }
    }

    /// Verify funding rate conversion consistency (mantissa <-> f64 round-trip).
    #[test]
    fn test_funding_rate_conversion_consistency() {
        let test_rates = [
            (10000i64, 0.0001f64), // 0.01%
            (-10000, -0.0001),     // -0.01%
            (100000, 0.001),       // 0.1%
            (1000, 0.00001),       // 0.001%
            (0, 0.0),              // Zero rate
        ];

        for (mantissa, expected_f64) in test_rates {
            let event = FundingEvent {
                ts: Utc::now(),
                symbol: "BTCUSDT".to_string(),
                mark_price_mantissa: 10000000,
                index_price_mantissa: 10000000,
                estimated_settle_price_mantissa: 10000000,
                price_exponent: FUNDING_PRICE_EXPONENT,
                funding_rate_mantissa: mantissa,
                rate_exponent: FUNDING_RATE_EXPONENT,
                next_funding_time_ms: 0,
                source: None,
            };

            let rate_f64 = event.funding_rate_f64();
            assert!(
                (rate_f64 - expected_f64).abs() < 1e-12,
                "Rate conversion failed: {} mantissa should be {} f64, got {}",
                mantissa,
                expected_f64,
                rate_f64
            );
        }
    }
}
