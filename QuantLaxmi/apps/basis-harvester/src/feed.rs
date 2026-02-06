//! Market feed for basis mean-reversion.
//!
//! Combines 2 WebSocket streams into a single `MarketFeed<BasisSnapshot>`:
//! 1. Spot @bookTicker   — best bid/ask for spot legs
//! 2. Perp @bookTicker   — best bid/ask for perp legs
//!
//! No markPrice stream needed (unlike funding-harvester).
//!
//! Uses combined streams (`?streams=sym1@bookTicker/sym2@bookTicker/...`)
//! so total connections = 2 regardless of symbol count.
//!
//! Supports **dynamic symbol rotation** via `with_symbol_updates()`.

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use quantlaxmi_paper::{MarketEvent, MarketFeed};
use quantlaxmi_runner_crypto::ws_resilient::{ResilientWs, ResilientWsConfig};
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::snapshot::BasisSnapshot;

// ---------------------------------------------------------------------------
// Raw WebSocket JSON types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CombinedStream<T> {
    #[allow(dead_code)]
    stream: String,
    data: T,
}

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

// ---------------------------------------------------------------------------
// Internal message envelope
// ---------------------------------------------------------------------------

enum FeedMsg {
    SpotBook { symbol: String, bid: f64, ask: f64 },
    PerpBook { symbol: String, bid: f64, ask: f64 },
}

// ---------------------------------------------------------------------------
// WebSocket reader tasks
// ---------------------------------------------------------------------------

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
                    })
                    .await;
            }
        }
    });
    Ok(handle)
}

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

/// Combined feed for basis mean-reversion.
///
/// Spawns 2 WS connections (spot + perp bookTicker) and merges into `BasisSnapshot`.
/// Faster coalescing than funding-harvester (interval=10 vs 30).
pub struct BasisFeed {
    tx: mpsc::Sender<FeedMsg>,
    rx: mpsc::Receiver<FeedMsg>,
    snapshot: BasisSnapshot,
    update_count: u64,
    /// Emit a full snapshot every N updates (faster than funding-harvester).
    snapshot_interval: u64,
    /// Whether we've received at least one update from each stream type.
    has_spot: bool,
    has_perp: bool,
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

impl BasisFeed {
    /// Connect to both streams for the given symbols.
    pub async fn connect(symbols: Vec<String>) -> Result<Self> {
        let (tx, rx) = mpsc::channel(4096);

        let h1 = spawn_spot_reader(symbols.clone(), tx.clone())
            .await
            .context("spot WS connect")?;
        let h2 = spawn_perp_reader(symbols.clone(), tx.clone())
            .await
            .context("perp WS connect")?;

        Ok(Self {
            tx,
            rx,
            snapshot: BasisSnapshot::default(),
            update_count: 0,
            snapshot_interval: 10, // faster coalescing for HFT
            has_spot: false,
            has_perp: false,
            current_symbols: symbols,
            pinned_symbols: HashSet::new(),
            symbol_rx: None,
            pinned_rx: None,
            reader_handles: vec![h1, h2],
        })
    }

    pub fn with_symbol_updates(
        mut self,
        rx: tokio::sync::watch::Receiver<Vec<String>>,
    ) -> Self {
        self.symbol_rx = Some(rx);
        self
    }

    pub fn with_pinned_symbols(
        mut self,
        rx: tokio::sync::watch::Receiver<Vec<String>>,
    ) -> Self {
        self.pinned_rx = Some(rx);
        self
    }

    async fn rotate_symbols(&mut self, new_symbols: Vec<String>) -> Result<()> {
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

        for handle in self.reader_handles.drain(..) {
            handle.abort();
        }
        while self.rx.try_recv().is_ok() {}

        self.snapshot.symbols.retain(|k, _| merged_set.contains(k));

        let h1 = spawn_spot_reader(merged.clone(), self.tx.clone())
            .await
            .context("spot WS reconnect")?;
        let h2 = spawn_perp_reader(merged.clone(), self.tx.clone())
            .await
            .context("perp WS reconnect")?;

        self.reader_handles = vec![h1, h2];
        self.has_spot = false;
        self.has_perp = false;
        self.current_symbols = merged;

        Ok(())
    }
}

#[async_trait]
impl MarketFeed<BasisSnapshot> for BasisFeed {
    async fn next(&mut self) -> Result<MarketEvent<BasisSnapshot>> {
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
                        FeedMsg::SpotBook { symbol, bid, ask } => {
                            let state = self.snapshot.get_or_insert(&symbol);
                            state.spot_bid = bid;
                            state.spot_ask = ask;
                            state.spot_ts = Some(now);
                            self.has_spot = true;
                        }
                        FeedMsg::PerpBook { symbol, bid, ask } => {
                            let state = self.snapshot.get_or_insert(&symbol);
                            state.perp_bid = bid;
                            state.perp_ask = ask;
                            state.perp_ts = Some(now);
                            self.has_perp = true;
                        }
                    }

                    self.update_count += 1;
                    self.snapshot.ts = now;

                    // Warmup: wait until we have both spot and perp data
                    let warmed = self.has_spot && self.has_perp;
                    if !warmed && self.has_spot && self.has_perp {
                        info!(
                            updates = self.update_count,
                            "[FEED] Warmup complete — both spot and perp data received"
                        );
                    }

                    // Coalesce: emit snapshot every N updates, only after warmup
                    if warmed && self.update_count.is_multiple_of(self.snapshot_interval) {
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
