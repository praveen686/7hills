//! Market feed for funding arbitrage.
//!
//! Combines 3 WebSocket streams into a single `MarketFeed<FundingArbSnapshot>`:
//! 1. Spot @bookTicker   — best bid/ask for spot legs
//! 2. Perp @bookTicker   — best bid/ask for perp legs
//! 3. Perp @markPrice    — funding rate + next settlement time
//!
//! Uses combined streams (`?streams=sym1@bookTicker/sym2@bookTicker/...`)
//! so total connections = 3 regardless of symbol count.
//!
//! Supports **dynamic symbol rotation** via `with_symbol_updates()`:
//! when the background scanner finds better opportunities, the feed
//! drops old WebSocket connections and reconnects with new symbols.

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use quantlaxmi_paper::{MarketEvent, MarketFeed};
use quantlaxmi_runner_crypto::ws_resilient::{ResilientWs, ResilientWsConfig};
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::snapshot::FundingArbSnapshot;

// ---------------------------------------------------------------------------
// Raw WebSocket JSON types
// ---------------------------------------------------------------------------

/// Combined stream wrapper: `{"stream":"btcusdt@bookTicker","data":{...}}`
#[derive(Debug, Deserialize)]
struct CombinedStream<T> {
    #[allow(dead_code)]
    stream: String,
    data: T,
}

/// Binance bookTicker event (works for both spot and futures).
#[derive(Debug, Deserialize)]
struct BookTicker {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "b")]
    bid_price: String,
    #[serde(rename = "B")]
    #[allow(dead_code)]
    bid_qty: String,
    #[serde(rename = "a")]
    ask_price: String,
    #[serde(rename = "A")]
    #[allow(dead_code)]
    ask_qty: String,
}

/// Binance markPrice event from futures stream.
#[derive(Debug, Deserialize)]
struct MarkPrice {
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "p")]
    mark_price: String,
    #[serde(rename = "i")]
    index_price: String,
    #[serde(rename = "r")]
    funding_rate: String,
    #[serde(rename = "T")]
    next_funding_time_ms: i64,
    #[serde(rename = "E")]
    event_time_ms: i64,
}

// ---------------------------------------------------------------------------
// Internal message envelope
// ---------------------------------------------------------------------------

enum FeedMsg {
    SpotBook { symbol: String, bid: f64, ask: f64, ts: DateTime<Utc> },
    PerpBook { symbol: String, bid: f64, ask: f64, ts: DateTime<Utc> },
    MarkPrice {
        symbol: String,
        mark: f64,
        index: f64,
        rate: f64,
        next_funding_ms: i64,
        ts: DateTime<Utc>,
    },
}

fn ms_to_dt(ms: i64) -> DateTime<Utc> {
    Utc.timestamp_millis_opt(ms)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap())
}

// ---------------------------------------------------------------------------
// WebSocket reader tasks
// ---------------------------------------------------------------------------

/// Build a combined stream URL for Binance spot bookTicker.
fn spot_book_url(symbols: &[String]) -> String {
    let streams: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}@bookTicker", s.to_lowercase()))
        .collect();
    format!(
        "wss://stream.binance.com:9443/stream?streams={}",
        streams.join("/")
    )
}

/// Build a combined stream URL for Binance futures bookTicker.
fn perp_book_url(symbols: &[String]) -> String {
    let streams: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}@bookTicker", s.to_lowercase()))
        .collect();
    format!(
        "wss://fstream.binance.com/stream?streams={}",
        streams.join("/")
    )
}

/// Build a combined stream URL for Binance futures markPrice.
fn mark_price_url(symbols: &[String]) -> String {
    let streams: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}@markPrice", s.to_lowercase()))
        .collect();
    format!(
        "wss://fstream.binance.com/stream?streams={}",
        streams.join("/")
    )
}

fn ws_config() -> ResilientWsConfig {
    ResilientWsConfig {
        liveness_timeout: Duration::from_secs(60),
        read_timeout: Duration::from_secs(10),
        initial_backoff: Duration::from_secs(1),
        max_backoff: Duration::from_secs(30),
        max_reconnect_attempts: 100,
        ..Default::default()
    }
}

/// Spawn a task that reads spot bookTicker messages.
async fn spawn_spot_reader(
    symbols: Vec<String>,
    tx: mpsc::Sender<FeedMsg>,
) -> Result<JoinHandle<()>> {
    let url = spot_book_url(&symbols);
    let mut ws = ResilientWs::connect(&url, ws_config()).await?;

    let handle = tokio::spawn(async move {
        loop {
            let msg = match ws.next_message().await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(e) => {
                    warn!("[FEED] Spot WS error: {}", e);
                    break;
                }
            };
            if !msg.is_text() {
                continue;
            }
            let txt = match msg.into_text() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if let Ok(combined) = serde_json::from_str::<CombinedStream<BookTicker>>(&txt) {
                let bid = combined.data.bid_price.parse::<f64>().unwrap_or(0.0);
                let ask = combined.data.ask_price.parse::<f64>().unwrap_or(0.0);
                let _ = tx
                    .send(FeedMsg::SpotBook {
                        symbol: combined.data.symbol,
                        bid,
                        ask,
                        ts: Utc::now(),
                    })
                    .await;
            }
        }
    });
    Ok(handle)
}

/// Spawn a task that reads perp bookTicker messages.
async fn spawn_perp_reader(
    symbols: Vec<String>,
    tx: mpsc::Sender<FeedMsg>,
) -> Result<JoinHandle<()>> {
    let url = perp_book_url(&symbols);
    let mut ws = ResilientWs::connect(&url, ws_config()).await?;

    let handle = tokio::spawn(async move {
        loop {
            let msg = match ws.next_message().await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(e) => {
                    warn!("[FEED] Perp WS error: {}", e);
                    break;
                }
            };
            if !msg.is_text() {
                continue;
            }
            let txt = match msg.into_text() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if let Ok(combined) = serde_json::from_str::<CombinedStream<BookTicker>>(&txt) {
                let bid = combined.data.bid_price.parse::<f64>().unwrap_or(0.0);
                let ask = combined.data.ask_price.parse::<f64>().unwrap_or(0.0);
                let _ = tx
                    .send(FeedMsg::PerpBook {
                        symbol: combined.data.symbol,
                        bid,
                        ask,
                        ts: Utc::now(),
                    })
                    .await;
            }
        }
    });
    Ok(handle)
}

/// Spawn a task that reads markPrice messages.
async fn spawn_mark_reader(
    symbols: Vec<String>,
    tx: mpsc::Sender<FeedMsg>,
) -> Result<JoinHandle<()>> {
    let url = mark_price_url(&symbols);
    let mut ws = ResilientWs::connect(&url, ws_config()).await?;

    let handle = tokio::spawn(async move {
        loop {
            let msg = match ws.next_message().await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(e) => {
                    warn!("[FEED] MarkPrice WS error: {}", e);
                    break;
                }
            };
            if !msg.is_text() {
                continue;
            }
            let txt = match msg.into_text() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if let Ok(combined) = serde_json::from_str::<CombinedStream<MarkPrice>>(&txt) {
                let d = &combined.data;
                let mark = d.mark_price.parse::<f64>().unwrap_or(0.0);
                let index = d.index_price.parse::<f64>().unwrap_or(0.0);
                let rate = d.funding_rate.parse::<f64>().unwrap_or(0.0);
                let _ = tx
                    .send(FeedMsg::MarkPrice {
                        symbol: d.symbol.clone(),
                        mark,
                        index,
                        rate,
                        next_funding_ms: d.next_funding_time_ms,
                        ts: ms_to_dt(d.event_time_ms),
                    })
                    .await;
            }
        }
    });
    Ok(handle)
}

// ---------------------------------------------------------------------------
// MarketFeed implementation
// ---------------------------------------------------------------------------

/// Combined feed for funding arbitrage.
///
/// Spawns 3 WS connections and merges updates into `FundingArbSnapshot`.
/// Supports dynamic symbol rotation via `with_symbol_updates()`.
pub struct FundingArbFeed {
    tx: mpsc::Sender<FeedMsg>,
    rx: mpsc::Receiver<FeedMsg>,
    snapshot: FundingArbSnapshot,
    update_count: u64,
    /// Emit a full snapshot every N updates (coalesce high-frequency ticks).
    snapshot_interval: u64,
    /// Whether we've received at least one markPrice update (warmup gate).
    has_funding_data: bool,
    num_symbols: usize,
    /// Current symbol list.
    current_symbols: Vec<String>,
    /// Symbols pinned by open positions — never rotated out.
    pinned_symbols: HashSet<String>,
    /// Channel for receiving symbol rotation updates.
    symbol_rx: Option<tokio::sync::watch::Receiver<Vec<String>>>,
    /// Channel for receiving pinned symbol updates (from strategy).
    pinned_rx: Option<tokio::sync::watch::Receiver<Vec<String>>>,
    /// Handles for active reader tasks (aborted on rotation).
    reader_handles: Vec<JoinHandle<()>>,
}

impl FundingArbFeed {
    /// Connect to all 3 streams for the given symbols.
    pub async fn connect(symbols: Vec<String>) -> Result<Self> {
        let (tx, rx) = mpsc::channel(4096);
        let num_symbols = symbols.len();

        let h1 = spawn_spot_reader(symbols.clone(), tx.clone())
            .await
            .context("spot WS connect")?;
        let h2 = spawn_perp_reader(symbols.clone(), tx.clone())
            .await
            .context("perp WS connect")?;
        let h3 = spawn_mark_reader(symbols.clone(), tx.clone())
            .await
            .context("markPrice WS connect")?;

        Ok(Self {
            tx,
            rx,
            snapshot: FundingArbSnapshot::default(),
            update_count: 0,
            snapshot_interval: 30,
            has_funding_data: false,
            num_symbols,
            current_symbols: symbols,
            pinned_symbols: HashSet::new(),
            symbol_rx: None,
            pinned_rx: None,
            reader_handles: vec![h1, h2, h3],
        })
    }

    /// Enable dynamic symbol rotation from a watch channel.
    pub fn with_symbol_updates(
        mut self,
        rx: tokio::sync::watch::Receiver<Vec<String>>,
    ) -> Self {
        self.symbol_rx = Some(rx);
        self
    }

    /// Set a watch channel for pinned symbols (open positions that must stay in feed).
    pub fn with_pinned_symbols(
        mut self,
        rx: tokio::sync::watch::Receiver<Vec<String>>,
    ) -> Self {
        self.pinned_rx = Some(rx);
        self
    }

    /// Rotate to a new symbol list: abort old WS tasks, spawn new ones.
    /// Compares as sets to avoid unnecessary reconnects from ordering noise.
    /// Pinned symbols (open positions) are always kept in the feed.
    async fn rotate_symbols(&mut self, new_symbols: Vec<String>) -> Result<()> {
        // Merge new symbols with pinned symbols (open positions must stay)
        let mut merged: Vec<String> = new_symbols.clone();
        for pinned in &self.pinned_symbols {
            if !merged.contains(pinned) {
                merged.push(pinned.clone());
            }
        }

        let old_set: HashSet<&String> = self.current_symbols.iter().collect();
        let merged_set: HashSet<&String> = merged.iter().collect();
        if old_set == merged_set {
            return Ok(());
        }

        info!(
            old = ?self.current_symbols,
            new = ?merged,
            pinned = ?self.pinned_symbols,
            "[FEED] Rotating symbols"
        );

        // Abort old reader tasks
        for handle in self.reader_handles.drain(..) {
            handle.abort();
        }

        // Drain stale messages from the channel
        while self.rx.try_recv().is_ok() {}

        // Remove snapshot data for symbols no longer tracked
        self.snapshot.symbols.retain(|k, _| merged_set.contains(k));

        // Spawn new readers
        let h1 = spawn_spot_reader(merged.clone(), self.tx.clone())
            .await
            .context("spot WS reconnect")?;
        let h2 = spawn_perp_reader(merged.clone(), self.tx.clone())
            .await
            .context("perp WS reconnect")?;
        let h3 = spawn_mark_reader(merged.clone(), self.tx.clone())
            .await
            .context("markPrice WS reconnect")?;

        self.reader_handles = vec![h1, h2, h3];
        self.has_funding_data = false;
        self.num_symbols = merged.len();
        self.current_symbols = merged;

        Ok(())
    }
}

#[async_trait]
impl MarketFeed<FundingArbSnapshot> for FundingArbFeed {
    async fn next(&mut self) -> Result<MarketEvent<FundingArbSnapshot>> {
        loop {
            // Update pinned symbols from strategy (non-blocking)
            if let Some(ref mut pin_rx) = self.pinned_rx
                && pin_rx.has_changed().unwrap_or(false)
            {
                self.pinned_symbols = pin_rx.borrow_and_update().iter().cloned().collect();
            }

            // Check for symbol rotation (non-blocking)
            if let Some(ref mut sym_rx) = self.symbol_rx
                && sym_rx.has_changed().unwrap_or(false)
            {
                let new_symbols = sym_rx.borrow_and_update().clone();
                if !new_symbols.is_empty() {
                    self.rotate_symbols(new_symbols).await?;
                }
            }

            match self.rx.recv().await {
                Some(msg) => {
                    let now = Utc::now();
                    match msg {
                        FeedMsg::SpotBook { symbol, bid, ask, ts } => {
                            let state = self.snapshot.get_or_insert(&symbol);
                            state.spot_bid = bid;
                            state.spot_ask = ask;
                            state.spot_ts = Some(ts);
                        }
                        FeedMsg::PerpBook { symbol, bid, ask, ts } => {
                            let state = self.snapshot.get_or_insert(&symbol);
                            state.perp_bid = bid;
                            state.perp_ask = ask;
                            state.perp_ts = Some(ts);
                        }
                        FeedMsg::MarkPrice {
                            symbol,
                            mark,
                            index,
                            rate,
                            next_funding_ms,
                            ts,
                        } => {
                            let state = self.snapshot.get_or_insert(&symbol);
                            state.mark_price = mark;
                            state.index_price = index;
                            state.funding_rate = rate;
                            state.next_funding_time_ms = next_funding_ms;
                            state.funding_ts = Some(ts);
                        }
                    }

                    self.update_count += 1;
                    self.snapshot.ts = now;

                    // Track warmup: wait until at least one symbol has funding data
                    if !self.has_funding_data {
                        let funded = self
                            .snapshot
                            .symbols
                            .values()
                            .filter(|s| s.has_funding())
                            .count();
                        if funded > 0 {
                            self.has_funding_data = true;
                            info!(
                                funded = funded,
                                total = self.num_symbols,
                                updates = self.update_count,
                                "[FEED] Warmup complete — funding data received"
                            );
                        }
                    }

                    // Coalesce: emit snapshot every N updates, but only after warmup
                    if self.has_funding_data
                        && self.update_count.is_multiple_of(self.snapshot_interval)
                    {
                        debug!(
                            updates = self.update_count,
                            symbols = self.snapshot.symbols.len(),
                            "[FEED] Emitting snapshot"
                        );
                        return Ok(MarketEvent::Snapshot {
                            ts: now,
                            snapshot: self.snapshot.clone(),
                        });
                    }
                }
                None => {
                    anyhow::bail!("All feed channels closed");
                }
            }
        }
    }
}
