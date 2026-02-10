//! # Paper Trading Module
//!
//! Paper trading = live capture + WAL + Simulator.
//!
//! ## Architecture
//! ```text
//! Live SBE Stream → WAL Writer → Simulator.on_depth() → Fill events
//!                      ↓
//!                 JSONL file
//!                 (for replay)
//! ```
//!
//! ## WAL (Write-Ahead Log)
//! All incoming depth events are persisted to disk BEFORE processing.
//! This ensures:
//! - Crash recovery: replay from WAL on restart
//! - Audit trail: complete record of all market data
//! - Backtest parity: same data, same fills
//!
//! ## Unified Execution
//! **ALL matching logic lives in `crate::sim::Simulator`.**
//! PaperVenue is now a thin wrapper that:
//! - Ingests depth events (calls Simulator.on_depth)
//! - Submits orders (calls Simulator.submit)
//! - Returns fills from Simulator
//!
//! This guarantees paper == backtest behavior.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tracing::info;

use crate::sim::{Fill, Order, Side, SimConfig, Simulator};
use quantlaxmi_models::depth::DepthEvent;

/// Write-Ahead Log for paper trading sessions.
pub struct WriteAheadLog {
    session_dir: PathBuf,
    depth_file: tokio::fs::File,
    events_written: u64,
}

impl WriteAheadLog {
    /// Create a new WAL for a paper trading session.
    pub async fn new(base_dir: &Path, symbol: &str) -> Result<Self> {
        let session_id = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let session_dir = base_dir
            .join("paper_sessions")
            .join(format!("{}_{}", symbol, session_id));

        tokio::fs::create_dir_all(&session_dir)
            .await
            .context("Create WAL directory")?;

        let depth_path = session_dir.join("depth_wal.jsonl");
        let depth_file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&depth_path)
            .await
            .context("Open depth WAL file")?;

        info!("WAL initialized: {:?}", session_dir);

        Ok(Self {
            session_dir,
            depth_file,
            events_written: 0,
        })
    }

    /// Write a depth event to the WAL.
    pub async fn write_depth_event(&mut self, event: &DepthEvent) -> Result<()> {
        let line = serde_json::to_string(event)?;
        self.depth_file.write_all(line.as_bytes()).await?;
        self.depth_file.write_all(b"\n").await?;
        self.events_written += 1;
        Ok(())
    }

    /// Flush WAL to disk.
    pub async fn flush(&mut self) -> Result<()> {
        self.depth_file.flush().await?;
        Ok(())
    }

    /// Get session directory path.
    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    /// Get number of events written.
    pub fn events_written(&self) -> u64 {
        self.events_written
    }
}

/// Paper fill from the simulator (re-exported for convenience).
pub type PaperFill = Fill;

/// Paper venue that wraps the unified Simulator.
///
/// **This is now a THIN WRAPPER around `crate::sim::Simulator`.**
/// All matching logic, order book state, pending orders, and position tracking
/// happen in the Simulator. This guarantees paper == backtest behavior.
pub struct PaperVenue {
    /// The unified simulator (SINGLE SOURCE OF TRUTH)
    sim: Simulator,
    /// Symbol being traded
    symbol: String,
}

impl PaperVenue {
    /// Create a new paper venue.
    pub fn new(price_exponent: i8, qty_exponent: i8) -> Self {
        // Create Simulator with default config
        let sim_cfg = SimConfig {
            fee_bps_maker: 2.0,
            fee_bps_taker: 10.0,
            latency_ticks: 0,
            allow_partial_fills: true,
            initial_cash: 100_000.0,
        };

        // Exponents will be set on first depth event
        let _ = price_exponent;
        let _ = qty_exponent;

        Self {
            sim: Simulator::new(sim_cfg),
            symbol: String::new(),
        }
    }

    /// Create with custom config.
    pub fn with_config(cfg: SimConfig) -> Self {
        Self {
            sim: Simulator::new(cfg),
            symbol: String::new(),
        }
    }

    /// Apply a depth event to update the order book.
    ///
    /// This delegates to `Simulator.on_depth()` which:
    /// - Updates the order book
    /// - Checks pending limit orders for fills
    /// - Returns any fills generated
    pub fn apply_depth_event(&mut self, event: &DepthEvent) -> Result<Vec<Fill>> {
        self.symbol = event.tradingsymbol.clone();
        let fills = self.sim.on_depth(&event.tradingsymbol, event);
        Ok(fills)
    }

    /// Submit a limit order.
    ///
    /// This delegates to `Simulator.submit()` which:
    /// - If order crosses spread → immediate fill (taker)
    /// - Else → queued as pending (maker when filled later)
    pub fn submit_order(&mut self, symbol: String, side: Side, price: f64, qty: f64) -> u64 {
        let order_id = self.sim.next_order_id();
        let order = Order::limit(order_id, symbol, side, qty, price);

        info!(
            "Paper order submitted: {:?} {} @ {} (id={})",
            side, qty, price, order_id
        );

        let ts_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64;
        let _fills = self.sim.submit(ts_ns, order);

        order_id
    }

    /// Submit a market order.
    pub fn submit_market_order(&mut self, symbol: String, side: Side, qty: f64) -> u64 {
        let order_id = self.sim.next_order_id();
        let order = Order::market(order_id, symbol, side, qty);

        info!(
            "Paper market order submitted: {:?} {} (id={})",
            side, qty, order_id
        );

        let ts_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64;
        let _fills = self.sim.submit(ts_ns, order);

        order_id
    }

    /// Get best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.sim.best_bid(&self.symbol)
    }

    /// Get best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.sim.best_ask(&self.symbol)
    }

    /// Get current position (delegates to Simulator).
    pub fn position(&self) -> f64 {
        self.sim.position(&self.symbol)
    }

    /// Get realized PnL (delegates to Simulator).
    pub fn realized_pnl(&self) -> f64 {
        self.sim.realized_pnl()
    }

    /// Get all fills (delegates to Simulator).
    pub fn fills(&self) -> &[Fill] {
        self.sim.fills()
    }

    /// Get pending orders count.
    pub fn pending_orders(&self) -> Vec<&Order> {
        self.sim.pending_orders(&self.symbol)
    }

    /// Get underlying simulator (for advanced use).
    pub fn simulator(&self) -> &Simulator {
        &self.sim
    }

    /// Get mutable underlying simulator (for advanced use).
    pub fn simulator_mut(&mut self) -> &mut Simulator {
        &mut self.sim
    }
}

/// Paper trading session that combines WAL + PaperVenue.
pub struct PaperSession {
    pub wal: WriteAheadLog,
    pub venue: Arc<Mutex<PaperVenue>>,
    pub symbol: String,
}

impl PaperSession {
    /// Create a new paper trading session.
    pub async fn new(
        base_dir: &Path,
        symbol: &str,
        price_exponent: i8,
        qty_exponent: i8,
    ) -> Result<Self> {
        let wal = WriteAheadLog::new(base_dir, symbol).await?;
        let venue = Arc::new(Mutex::new(PaperVenue::new(price_exponent, qty_exponent)));

        Ok(Self {
            wal,
            venue,
            symbol: symbol.to_string(),
        })
    }

    /// Process a depth event: write to WAL, then apply to venue.
    pub async fn process_depth_event(&mut self, event: DepthEvent) -> Result<Vec<Fill>> {
        // WAL first (durability)
        self.wal.write_depth_event(&event).await?;

        // Then apply to venue (may generate fills)
        let mut venue = self.venue.lock().await;
        venue.apply_depth_event(&event)
    }

    /// Flush WAL to disk.
    pub async fn flush(&mut self) -> Result<()> {
        self.wal.flush().await
    }

    /// Get session summary.
    pub async fn summary(&self) -> PaperSessionSummary {
        let venue = self.venue.lock().await;
        PaperSessionSummary {
            symbol: self.symbol.clone(),
            events_processed: self.wal.events_written(),
            fills_count: venue.fills().len(),
            position: venue.position(),
            realized_pnl: venue.realized_pnl(),
            best_bid: venue.best_bid(),
            best_ask: venue.best_ask(),
        }
    }
}

/// Summary of a paper trading session.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PaperSessionSummary {
    pub symbol: String,
    pub events_processed: u64,
    pub fills_count: usize,
    pub position: f64,
    pub realized_pnl: f64,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantlaxmi_models::depth::{DepthLevel, IntegrityTier};

    #[test]
    fn test_paper_venue_fill_buy() {
        let mut venue = PaperVenue::new(-2, -8);

        // Apply snapshot with asks at 90001 and 90002
        let snapshot = DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            first_update_id: 100,
            last_update_id: 100,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![DepthLevel {
                price: 9000000,
                qty: 100000000,
            }],
            asks: vec![
                DepthLevel {
                    price: 9000100,
                    qty: 50000000,
                },
                DepthLevel {
                    price: 9000200,
                    qty: 100000000,
                },
            ],
            is_snapshot: true,
            integrity_tier: IntegrityTier::Certified,
            source: None,
        };
        venue.apply_depth_event(&snapshot).unwrap();

        // Submit buy order at 90001.50 (above best ask of 90001)
        // This crosses the spread, so it fills immediately as taker
        venue.submit_order("BTCUSDT".to_string(), Side::Buy, 90001.50, 0.3);

        // Should have filled immediately (crossed spread)
        assert_eq!(venue.fills().len(), 1);
        assert!((venue.fills()[0].price - 90001.0).abs() < 0.01);
        assert!(venue.position() > 0.0);
    }

    #[test]
    fn test_paper_venue_pending_order_fills_on_depth() {
        let mut venue = PaperVenue::new(-2, -8);

        // Apply snapshot: bid=90000, ask=90010
        let snapshot = DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            first_update_id: 100,
            last_update_id: 100,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![DepthLevel {
                price: 9000000,
                qty: 100000000,
            }],
            asks: vec![DepthLevel {
                price: 9001000,
                qty: 50000000,
            }],
            is_snapshot: true,
            integrity_tier: IntegrityTier::Certified,
            source: None,
        };
        venue.apply_depth_event(&snapshot).unwrap();

        // Submit buy order at 90005 (below best ask of 90010)
        // This does NOT cross, so it's queued as pending
        venue.submit_order("BTCUSDT".to_string(), Side::Buy, 90005.0, 0.3);

        // Should be pending, no fills yet
        assert_eq!(venue.fills().len(), 0);
        assert_eq!(venue.pending_orders().len(), 1);

        // Price drops: ask moves to 90003 (below our limit of 90005)
        let diff = DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            first_update_id: 101,
            last_update_id: 101,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![],
            asks: vec![DepthLevel {
                price: 9000300, // 90003
                qty: 50000000,
            }],
            is_snapshot: false,
            integrity_tier: IntegrityTier::Certified,
            source: None,
        };
        let fills = venue.apply_depth_event(&diff).unwrap();

        // Should fill as maker
        assert_eq!(fills.len(), 1);
        assert!((fills[0].price - 90003.0).abs() < 0.01);
        assert_eq!(venue.pending_orders().len(), 0);
    }

    #[test]
    fn test_paper_venue_no_fill_below_ask() {
        let mut venue = PaperVenue::new(-2, -8);

        // Apply snapshot with best ask at 90001
        let snapshot = DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            first_update_id: 100,
            last_update_id: 100,
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
            is_snapshot: true,
            integrity_tier: IntegrityTier::Certified,
            source: None,
        };
        venue.apply_depth_event(&snapshot).unwrap();

        // Submit buy order at 90000 (below best ask of 90001)
        venue.submit_order("BTCUSDT".to_string(), Side::Buy, 90000.0, 0.3);

        // Should be pending (doesn't cross)
        assert_eq!(venue.fills().len(), 0);
        assert_eq!(venue.pending_orders().len(), 1);

        // Diff update with no price change should NOT trigger fill
        let diff = DepthEvent {
            ts: chrono::Utc::now(),
            tradingsymbol: "BTCUSDT".to_string(),
            first_update_id: 101,
            last_update_id: 101,
            price_exponent: -2,
            qty_exponent: -8,
            bids: vec![],
            asks: vec![],
            is_snapshot: false,
            integrity_tier: IntegrityTier::Certified,
            source: None,
        };
        let fills = venue.apply_depth_event(&diff).unwrap();

        assert!(fills.is_empty());
        assert_eq!(venue.pending_orders().len(), 1);
    }
}
