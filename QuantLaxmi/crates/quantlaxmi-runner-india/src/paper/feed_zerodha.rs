//! MarketFeedZerodha - Zerodha WebSocket to OptionsSnapshot adapter
//!
//! Implements the `MarketFeed<OptionsSnapshot>` trait for the paper trading engine.
//! Converts raw Zerodha ticks into periodic snapshots with staleness tracking.
//!
//! ## Architecture
//! - Connects to Zerodha Kite WebSocket in full mode (184-byte L2 depth)
//! - Maintains internal micro-book state for each instrument (RwLock protected)
//! - Emits OptionsSnapshot at configurable intervals (default: 100ms)
//! - Tracks staleness (age_ms) for each quote via nanosecond timestamps
//!
//! ## Correctness Guarantees
//!
//! 1. **Snapshot Atomicity**: Quote state is protected by RwLock to prevent torn reads
//! 2. **Deterministic Staleness**: All staleness computed from nanosecond timestamps
//! 3. **Reconnect Cleanup**: Old task aborted, state reset on reconnect
//! 4. **Provenance**: Every snapshot carries audit trail (connection_id, seq, ticks, drops)
//!
//! ## Loss Awareness
//! - Heartbeat timeout detection (reconnect trigger)
//! - Quote staleness tracking per instrument
//! - Connection state exposed in provenance

use anyhow::Result;
use async_trait::async_trait;
use byteorder::{BigEndian, ByteOrder};
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{Instant, interval};
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info, warn};

use quantlaxmi_paper::{MarketEvent, MarketFeed};

use crate::paper::mapping::InstrumentMap;
use crate::paper::snapshot::{OptQuote, OptionsSnapshot, PriceQty, SnapshotProvenance};

/// Configuration for MarketFeedZerodha.
#[derive(Debug, Clone)]
pub struct FeedConfig {
    /// Snapshot emit interval in milliseconds.
    pub snapshot_interval_ms: u64,
    /// Heartbeat timeout in seconds (triggers reconnect).
    pub heartbeat_timeout_secs: u64,
    /// Maximum reconnection attempts (0 = infinite).
    pub max_retries: u32,
    /// Initial reconnection delay in milliseconds.
    pub initial_delay_ms: u64,
    /// Maximum reconnection delay in milliseconds.
    pub max_delay_ms: u64,
    /// Backoff multiplier for exponential delay.
    pub backoff_multiplier: f64,
    /// Staleness threshold in milliseconds.
    pub staleness_threshold_ms: u32,
}

impl Default for FeedConfig {
    fn default() -> Self {
        Self {
            snapshot_interval_ms: 100,
            heartbeat_timeout_secs: 30,
            max_retries: 0, // Infinite
            initial_delay_ms: 1000,
            max_delay_ms: 60000,
            backoff_multiplier: 2.0,
            staleness_threshold_ms: 5000,
        }
    }
}

/// Internal state for a single instrument (protected by RwLock in QuoteBook).
#[derive(Debug, Clone)]
struct QuoteState {
    token: u32,
    tradingsymbol: String,
    strike: i32,
    right: crate::paper::snapshot::Right,
    bid: Option<PriceQty>,
    ask: Option<PriceQty>,
    ltp: Option<f64>,
    volume: Option<u32>,
    last_update_ns: i64,
    expiry: String,
}

impl QuoteState {
    fn new(meta: &crate::paper::mapping::InstrumentMeta) -> Self {
        Self {
            token: meta.token,
            tradingsymbol: meta.tradingsymbol.clone(),
            strike: meta.strike,
            right: meta.right,
            bid: None,
            ask: None,
            ltp: None,
            volume: None,
            last_update_ns: 0,
            expiry: meta.expiry_canonical.clone(),
        }
    }

    fn update(
        &mut self,
        tick_ts_ns: i64,
        bid: Option<PriceQty>,
        ask: Option<PriceQty>,
        ltp: f64,
        volume: u32,
    ) {
        self.bid = bid;
        self.ask = ask;
        self.ltp = Some(ltp);
        self.volume = Some(volume);
        self.last_update_ns = tick_ts_ns;
    }

    fn to_quote(&self) -> OptQuote {
        let mut q = OptQuote::with_expiry(
            self.token,
            self.tradingsymbol.clone(),
            self.strike,
            self.right,
            self.expiry.clone(),
        );
        q.bid = self.bid;
        q.ask = self.ask;
        if let Some(p) = self.ltp {
            q.last = Some(PriceQty::new(p, self.volume.unwrap_or(0)));
        }
        q.last_update_ns = self.last_update_ns;
        q
    }
}

/// Thread-safe quote book (RwLock-protected).
///
/// Prevents torn reads: snapshot builder takes read lock, tick handler takes write lock.
#[derive(Debug)]
struct QuoteBook {
    states: RwLock<HashMap<u32, QuoteState>>,
    underlying: String,
    expiry: String,
}

impl QuoteBook {
    fn new(instrument_map: &InstrumentMap) -> Self {
        let states: HashMap<u32, QuoteState> = instrument_map
            .instruments()
            .map(|meta| (meta.token, QuoteState::new(meta)))
            .collect();

        Self {
            states: RwLock::new(states),
            underlying: instrument_map.underlying().unwrap_or("UNKNOWN").to_string(),
            expiry: instrument_map
                .expiry_str()
                .unwrap_or("0000-00-00")
                .to_string(),
        }
    }

    /// Update a quote from tick (write lock).
    fn update_tick(
        &self,
        token: u32,
        tick_ts_ns: i64,
        bid: Option<PriceQty>,
        ask: Option<PriceQty>,
        ltp: f64,
        volume: u32,
    ) {
        if let Ok(mut states) = self.states.write()
            && let Some(state) = states.get_mut(&token)
        {
            state.update(tick_ts_ns, bid, ask, ltp, volume);
        }
    }

    /// Build snapshot (read lock - atomic view of all quotes).
    fn build_snapshot(&self, ts_ns: i64, provenance: SnapshotProvenance) -> OptionsSnapshot {
        let quotes: Vec<OptQuote> = if let Ok(states) = self.states.read() {
            states.values().map(|s| s.to_quote()).collect()
        } else {
            Vec::new()
        };

        let mut snapshot = OptionsSnapshot::new(self.underlying.clone(), self.expiry.clone());
        snapshot.quotes = quotes;
        snapshot.finalize(ts_ns, provenance);
        snapshot
    }
}

/// Parsed tick from Zerodha WebSocket.
#[derive(Debug)]
struct ParsedTick {
    token: u32,
    ltp: f64,
    volume: u32,
    best_bid: Option<PriceQty>,
    best_ask: Option<PriceQty>,
}

/// Connection state for cleanup on reconnect.
struct ConnectionState {
    /// Current connection ID (increments on reconnect).
    connection_id: AtomicU64,
    /// Snapshot sequence within this connection.
    seq: AtomicU64,
    /// Ticks processed since last snapshot.
    ticks_since_last: AtomicU32,
    /// Ticks dropped (unknown token or parse error).
    dropped_ticks: AtomicU32,
    /// Total subscribed tokens.
    subscribed_tokens: u32,
}

impl ConnectionState {
    fn new(subscribed_tokens: u32) -> Self {
        Self {
            connection_id: AtomicU64::new(0),
            seq: AtomicU64::new(0),
            ticks_since_last: AtomicU32::new(0),
            dropped_ticks: AtomicU32::new(0),
            subscribed_tokens,
        }
    }

    /// Reset counters on new connection (NOT connection_id - that increments).
    fn on_reconnect(&self) {
        self.connection_id.fetch_add(1, Ordering::SeqCst);
        self.seq.store(0, Ordering::SeqCst);
        self.ticks_since_last.store(0, Ordering::SeqCst);
        self.dropped_ticks.store(0, Ordering::SeqCst);
    }

    fn record_ticks(&self, count: u32, dropped: u32) {
        self.ticks_since_last.fetch_add(count, Ordering::Relaxed);
        self.dropped_ticks.fetch_add(dropped, Ordering::Relaxed);
    }

    fn build_provenance(&self, stale_quotes: u32) -> SnapshotProvenance {
        let ticks = self.ticks_since_last.swap(0, Ordering::Relaxed);
        let dropped = self.dropped_ticks.swap(0, Ordering::Relaxed);

        SnapshotProvenance {
            source: "zerodha_ws".to_string(),
            connection_id: self.connection_id.load(Ordering::Relaxed),
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            ticks_since_last: ticks,
            subscribed_tokens: self.subscribed_tokens,
            dropped_ticks: dropped,
            stale_quotes,
        }
    }
}

/// MarketFeedZerodha - implements MarketFeed<OptionsSnapshot>.
pub struct MarketFeedZerodha {
    /// API key for Kite Connect.
    api_key: String,
    /// Access token for authentication.
    access_token: String,
    /// Instrument mapping.
    instrument_map: InstrumentMap,
    /// Feed configuration.
    config: FeedConfig,
    /// Shutdown flag.
    running: Arc<AtomicBool>,
    /// Event receiver (for async iteration).
    rx: Option<mpsc::Receiver<MarketEvent<OptionsSnapshot>>>,
    /// Handle to the feed task (for cleanup on drop/reconnect).
    task_handle: Option<JoinHandle<()>>,
}

impl MarketFeedZerodha {
    /// Create a new feed with API credentials and instrument map.
    pub fn new(
        api_key: String,
        access_token: String,
        instrument_map: InstrumentMap,
        config: FeedConfig,
    ) -> Self {
        info!(
            "[FEED] Initialized MarketFeedZerodha: {} instruments for {} expiry {}",
            instrument_map.len(),
            instrument_map.underlying().unwrap_or("?"),
            instrument_map.expiry_str().unwrap_or("?")
        );

        Self {
            api_key,
            access_token,
            instrument_map,
            config,
            running: Arc::new(AtomicBool::new(true)),
            rx: None,
            task_handle: None,
        }
    }

    /// Create from auto-discovery results.
    pub fn from_discovery(
        api_key: String,
        access_token: String,
        instruments: &[quantlaxmi_connectors_zerodha::UniverseInstrument],
        config: FeedConfig,
    ) -> Self {
        let map = InstrumentMap::from_universe(instruments);
        Self::new(api_key, access_token, map, config)
    }

    /// Stop the feed and cleanup.
    pub fn stop(&mut self) {
        info!("[FEED] Stopping MarketFeedZerodha");
        self.running.store(false, Ordering::SeqCst);

        // Abort the task if running
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Start the WebSocket connection and snapshot emission loop.
    /// Returns a receiver for market events.
    pub async fn start(&mut self) -> Result<mpsc::Receiver<MarketEvent<OptionsSnapshot>>> {
        // Cleanup any existing task
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }

        let (tx, rx) = mpsc::channel(1000);

        let api_key = self.api_key.clone();
        let access_token = self.access_token.clone();
        let config = self.config.clone();
        let running = self.running.clone();

        // Build quote book (RwLock-protected)
        let quote_book = Arc::new(QuoteBook::new(&self.instrument_map));
        let tokens = self.instrument_map.token_list();
        let token_set: std::collections::HashSet<u32> = tokens.iter().copied().collect();

        // Connection state for provenance
        let conn_state = Arc::new(ConnectionState::new(tokens.len() as u32));

        let handle = tokio::spawn(async move {
            let mut retry_count = 0u32;
            let mut current_delay = config.initial_delay_ms;

            while running.load(Ordering::SeqCst) {
                // Reset state on new connection
                conn_state.on_reconnect();

                info!(
                    "[FEED] Starting connection {} (retry {})",
                    conn_state.connection_id.load(Ordering::Relaxed),
                    retry_count
                );

                match Self::run_connection(
                    &api_key,
                    &access_token,
                    &tokens,
                    &token_set,
                    &config,
                    &running,
                    &quote_book,
                    &conn_state,
                    &tx,
                )
                .await
                {
                    Ok(()) => {
                        info!("[FEED] WebSocket connection closed cleanly");
                        break;
                    }
                    Err(e) => {
                        error!(error = %e, retry = retry_count, "[FEED] WebSocket connection failed");

                        if config.max_retries > 0 && retry_count >= config.max_retries {
                            error!(
                                "[FEED] Max reconnection attempts ({}) reached",
                                config.max_retries
                            );
                            break;
                        }

                        if !running.load(Ordering::SeqCst) {
                            break;
                        }

                        info!(delay_ms = current_delay, "[FEED] Reconnecting in...");
                        tokio::time::sleep(Duration::from_millis(current_delay)).await;

                        current_delay = ((current_delay as f64 * config.backoff_multiplier) as u64)
                            .min(config.max_delay_ms);
                        retry_count += 1;
                    }
                }
            }

            info!("[FEED] Feed task exiting");
        });

        self.task_handle = Some(handle);
        Ok(rx)
    }

    /// Internal: run a single WebSocket connection session.
    #[allow(clippy::too_many_arguments)]
    async fn run_connection(
        api_key: &str,
        access_token: &str,
        tokens: &[u32],
        token_set: &std::collections::HashSet<u32>,
        config: &FeedConfig,
        running: &Arc<AtomicBool>,
        quote_book: &Arc<QuoteBook>,
        conn_state: &Arc<ConnectionState>,
        tx: &mpsc::Sender<MarketEvent<OptionsSnapshot>>,
    ) -> Result<()> {
        // Connect to Kite WebSocket
        let ws_url = format!(
            "wss://ws.kite.trade/?api_key={}&access_token={}",
            api_key, access_token
        );

        eprintln!("[FEED] Connecting to wss://ws.kite.trade/ ...");
        let (ws_stream, _) = tokio_tungstenite::connect_async(&ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        eprintln!("[FEED] WebSocket connected");
        info!("[FEED] WebSocket connected");

        // Subscribe to instruments
        let subscribe_msg = serde_json::json!({
            "a": "subscribe",
            "v": tokens
        });
        write.send(Message::Text(subscribe_msg.to_string())).await?;

        // Set mode to "full" for L2 depth
        let mode_msg = serde_json::json!({
            "a": "mode",
            "v": ["full", tokens]
        });
        write.send(Message::Text(mode_msg.to_string())).await?;

        eprintln!(
            "[FEED] Subscribed {} instruments in full (L2) mode",
            tokens.len()
        );

        // Snapshot emission timer
        let mut snapshot_interval = interval(Duration::from_millis(config.snapshot_interval_ms));
        let mut last_message_time = Instant::now();
        let heartbeat_timeout = Duration::from_secs(config.heartbeat_timeout_secs);

        loop {
            if !running.load(Ordering::SeqCst) {
                info!("[FEED] Shutdown requested");
                let _ = write.send(Message::Close(None)).await;
                break;
            }

            tokio::select! {
                // Snapshot emission tick
                _ = snapshot_interval.tick() => {
                    let ts_ns = Utc::now().timestamp_nanos_opt().unwrap_or(0);

                    // Build snapshot with atomic read of quote state
                    let snapshot = quote_book.build_snapshot(ts_ns, {
                        let stale = quote_book.states.read()
                            .map(|s| s.values().filter(|q| {
                                if q.last_update_ns == 0 { return true; }
                                let age_ms = (ts_ns - q.last_update_ns) / 1_000_000;
                                age_ms > config.staleness_threshold_ms as i64
                            }).count() as u32)
                            .unwrap_or(0);
                        conn_state.build_provenance(stale)
                    });

                    let event = MarketEvent::Snapshot {
                        ts: Utc::now(),
                        snapshot,
                    };

                    if tx.send(event).await.is_err() {
                        warn!("[FEED] Event channel closed");
                        break;
                    }
                }

                // WebSocket message
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Binary(data))) => {
                            last_message_time = Instant::now();
                            let tick_ts_ns = Utc::now().timestamp_nanos_opt().unwrap_or(0);

                            // Parse ticks and update quote book
                            let (ticks, dropped) = Self::parse_binary_ticks(&data, token_set);
                            conn_state.record_ticks(ticks.len() as u32, dropped);

                            for tick in ticks {
                                quote_book.update_tick(
                                    tick.token,
                                    tick_ts_ns,
                                    tick.best_bid,
                                    tick.best_ask,
                                    tick.ltp,
                                    tick.volume,
                                );
                            }
                        }
                        Some(Ok(Message::Text(text))) => {
                            last_message_time = Instant::now();
                            if text.len() > 2 {
                                debug!(msg = %text, "[FEED] Text message");
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            last_message_time = Instant::now();
                            let _ = write.send(Message::Pong(data)).await;
                        }
                        Some(Ok(Message::Pong(_))) => {
                            last_message_time = Instant::now();
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("[FEED] Server closed connection");
                            return Err(anyhow::anyhow!("Server closed connection"));
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "[FEED] WebSocket read error");
                            return Err(e.into());
                        }
                        None => {
                            info!("[FEED] WebSocket stream ended");
                            return Err(anyhow::anyhow!("Stream ended"));
                        }
                        _ => {}
                    }

                    // Check heartbeat timeout
                    if last_message_time.elapsed() > heartbeat_timeout {
                        warn!(
                            elapsed_secs = last_message_time.elapsed().as_secs(),
                            "[FEED] Heartbeat timeout"
                        );
                        return Err(anyhow::anyhow!("Heartbeat timeout"));
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse binary tick data from Kite WebSocket.
    /// Returns (parsed_ticks, dropped_count).
    fn parse_binary_ticks(
        data: &[u8],
        token_set: &std::collections::HashSet<u32>,
    ) -> (Vec<ParsedTick>, u32) {
        let mut ticks = Vec::new();
        let mut dropped = 0u32;

        if data.len() < 4 {
            return (ticks, dropped);
        }

        let num_packets = BigEndian::read_i16(&data[0..2]) as usize;
        let mut offset = 2;

        for _ in 0..num_packets {
            if offset + 2 > data.len() {
                break;
            }

            let packet_len = BigEndian::read_i16(&data[offset..offset + 2]) as usize;
            offset += 2;

            if offset + packet_len > data.len() || packet_len < 8 {
                dropped += 1;
                break;
            }

            let packet = &data[offset..offset + packet_len];
            let token = BigEndian::read_u32(&packet[0..4]);

            // Only process tokens we're tracking
            if !token_set.contains(&token) {
                dropped += 1;
                offset += packet_len;
                continue;
            }

            let ltp_paise = BigEndian::read_i32(&packet[4..8]);
            let ltp = ltp_paise as f64 / 100.0;

            let mut volume = 0u32;
            let mut best_bid: Option<PriceQty> = None;
            let mut best_ask: Option<PriceQty> = None;

            // Quote mode (44+ bytes) - includes OHLC and volume
            if packet_len >= 44 {
                volume = BigEndian::read_u32(&packet[28..32]);
            }

            // Full mode (184 bytes) - includes 5-level market depth
            // Per Kite Connect docs: https://kite.trade/docs/connect/v3/websocket/
            // - Market depth starts at byte 64 (not 44!)
            // - Each level: qty (int32) + price (int32) + orders (int16) + padding (2) = 12 bytes
            // - 5 bid levels (bytes 64-123) + 5 ask levels (bytes 124-183) = 120 bytes
            // - Prices are in paise, divide by 100 for rupees
            if packet_len >= 184 {
                let depth_start = 64; // Market depth starts at byte 64
                let price_divisor = 100.0; // Prices in paise -> divide by 100 for rupees

                // Best bid (first level of buy side)
                let bid_qty = BigEndian::read_i32(&packet[depth_start..depth_start + 4]);
                let bid_price_raw =
                    BigEndian::read_i32(&packet[depth_start + 4..depth_start + 8]) as f64;
                let bid_price = bid_price_raw / price_divisor;

                if bid_qty > 0 && bid_price > 0.0 {
                    best_bid = Some(PriceQty::new(bid_price, bid_qty as u32));
                }

                // Best ask (first level of sell side)
                let ask_offset = depth_start + 60; // 5 bid levels * 12 bytes = 60 -> byte 124
                let ask_qty = BigEndian::read_i32(&packet[ask_offset..ask_offset + 4]);
                let ask_price_raw =
                    BigEndian::read_i32(&packet[ask_offset + 4..ask_offset + 8]) as f64;
                let ask_price = ask_price_raw / price_divisor;

                if ask_qty > 0 && ask_price > 0.0 {
                    best_ask = Some(PriceQty::new(ask_price, ask_qty as u32));
                }
            }

            ticks.push(ParsedTick {
                token,
                ltp,
                volume,
                best_bid,
                best_ask,
            });

            offset += packet_len;
        }

        (ticks, dropped)
    }
}

impl Drop for MarketFeedZerodha {
    fn drop(&mut self) {
        self.stop();
    }
}

#[async_trait]
impl MarketFeed<OptionsSnapshot> for MarketFeedZerodha {
    async fn next(&mut self) -> Result<MarketEvent<OptionsSnapshot>> {
        // Initialize connection if needed
        if self.rx.is_none() {
            let rx = self.start().await?;
            self.rx = Some(rx);
        }

        // Wait for next event
        if let Some(rx) = &mut self.rx {
            match rx.recv().await {
                Some(event) => Ok(event),
                None => Err(anyhow::anyhow!("Feed channel closed")),
            }
        } else {
            Err(anyhow::anyhow!("Feed not initialized"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_config_default() {
        let config = FeedConfig::default();
        assert_eq!(config.snapshot_interval_ms, 100);
        assert_eq!(config.heartbeat_timeout_secs, 30);
        assert_eq!(config.staleness_threshold_ms, 5000);
    }

    #[test]
    fn test_connection_state_reconnect() {
        let state = ConnectionState::new(10);
        assert_eq!(state.connection_id.load(Ordering::Relaxed), 0);

        state.on_reconnect();
        assert_eq!(state.connection_id.load(Ordering::Relaxed), 1);

        state.record_ticks(5, 2);
        let prov = state.build_provenance(3);

        assert_eq!(prov.connection_id, 1);
        assert_eq!(prov.seq, 0);
        assert_eq!(prov.ticks_since_last, 5);
        assert_eq!(prov.dropped_ticks, 2);
        assert_eq!(prov.stale_quotes, 3);
        assert_eq!(prov.subscribed_tokens, 10);
    }

    #[test]
    fn test_parse_binary_ticks_empty() {
        let token_set: std::collections::HashSet<u32> = [123].into_iter().collect();
        let (ticks, dropped) = MarketFeedZerodha::parse_binary_ticks(&[0, 0], &token_set);
        assert!(ticks.is_empty());
        assert_eq!(dropped, 0);
    }
}
