//! # Paper Trading Module
//!
//! Paper trading = live capture + WAL + PaperVenue.
//!
//! ## Architecture
//! ```text
//! Live SBE Stream → WAL Writer → Event Bus → PaperVenue → Fill Simulator
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
//! ## PaperVenue
//! Simulates order execution without touching real exchange:
//! - Maintains local order book from depth events
//! - Fills orders when price crosses
//! - Tracks positions and PnL

use anyhow::{Context, Result};
use chrono::Utc;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tracing::{info, warn};

use kubera_models::depth::DepthEvent;

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

/// Order side for paper trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Paper order waiting to be filled.
#[derive(Debug, Clone)]
pub struct PaperOrder {
    pub id: u64,
    pub symbol: String,
    pub side: Side,
    pub price: i64,        // Mantissa
    pub qty: i64,          // Mantissa
    pub created_at: chrono::DateTime<Utc>,
}

/// Simulated fill from paper trading.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PaperFill {
    pub order_id: u64,
    pub symbol: String,
    pub side: String,
    pub fill_price: f64,
    pub fill_qty: f64,
    pub timestamp: chrono::DateTime<Utc>,
}

/// Paper venue that simulates order execution.
pub struct PaperVenue {
    /// Local order book state (mantissa-based for determinism)
    bids: BTreeMap<i64, i64>,
    asks: BTreeMap<i64, i64>,
    last_update_id: u64,

    /// Exponents for price/qty conversion
    price_exponent: i8,
    qty_exponent: i8,

    /// Pending orders
    orders: Vec<PaperOrder>,
    next_order_id: u64,

    /// Fill history
    fills: Vec<PaperFill>,

    /// Position tracking
    position: f64,
    realized_pnl: f64,
    avg_entry_price: f64,
}

impl PaperVenue {
    /// Create a new paper venue.
    pub fn new(price_exponent: i8, qty_exponent: i8) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_update_id: 0,
            price_exponent,
            qty_exponent,
            orders: Vec::new(),
            next_order_id: 1,
            fills: Vec::new(),
            position: 0.0,
            realized_pnl: 0.0,
            avg_entry_price: 0.0,
        }
    }

    /// Apply a depth event to update the order book.
    pub fn apply_depth_event(&mut self, event: &DepthEvent) -> Result<Vec<PaperFill>> {
        // Sequence validation
        if !event.is_snapshot && event.first_update_id != self.last_update_id + 1 {
            warn!(
                "Sequence gap in paper venue: expected {}, got {}",
                self.last_update_id + 1,
                event.first_update_id
            );
        }

        if event.is_snapshot {
            self.bids.clear();
            self.asks.clear();
            for level in &event.bids {
                if level.qty > 0 {
                    self.bids.insert(level.price, level.qty);
                }
            }
            for level in &event.asks {
                if level.qty > 0 {
                    self.asks.insert(level.price, level.qty);
                }
            }
        } else {
            for level in &event.bids {
                if level.qty == 0 {
                    self.bids.remove(&level.price);
                } else {
                    self.bids.insert(level.price, level.qty);
                }
            }
            for level in &event.asks {
                if level.qty == 0 {
                    self.asks.remove(&level.price);
                } else {
                    self.asks.insert(level.price, level.qty);
                }
            }
        }

        self.last_update_id = event.last_update_id;

        // Check if any pending orders can be filled
        self.check_fills()
    }

    /// Submit a paper order.
    pub fn submit_order(&mut self, symbol: String, side: Side, price: f64, qty: f64) -> u64 {
        let order_id = self.next_order_id;
        self.next_order_id += 1;

        let price_mantissa = (price / 10f64.powi(self.price_exponent as i32)).round() as i64;
        let qty_mantissa = (qty / 10f64.powi(self.qty_exponent as i32)).round() as i64;

        let order = PaperOrder {
            id: order_id,
            symbol,
            side,
            price: price_mantissa,
            qty: qty_mantissa,
            created_at: Utc::now(),
        };

        info!(
            "Paper order submitted: {:?} {} @ {} (id={})",
            side,
            qty,
            price,
            order_id
        );

        self.orders.push(order);
        order_id
    }

    /// Check if any pending orders can be filled.
    fn check_fills(&mut self) -> Result<Vec<PaperFill>> {
        let mut new_fills = Vec::new();
        let mut filled_order_ids = Vec::new();

        for order in &self.orders {
            let filled = match order.side {
                Side::Buy => {
                    // Buy order fills if best ask <= order price
                    if let Some((&best_ask, &ask_qty)) = self.asks.iter().next() {
                        if best_ask <= order.price && ask_qty > 0 {
                            Some((best_ask, ask_qty.min(order.qty)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                Side::Sell => {
                    // Sell order fills if best bid >= order price
                    if let Some((&best_bid, &bid_qty)) = self.bids.iter().next_back() {
                        if best_bid >= order.price && bid_qty > 0 {
                            Some((best_bid, bid_qty.min(order.qty)))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            };

            if let Some((fill_price_mantissa, fill_qty_mantissa)) = filled {
                let fill_price =
                    fill_price_mantissa as f64 * 10f64.powi(self.price_exponent as i32);
                let fill_qty = fill_qty_mantissa as f64 * 10f64.powi(self.qty_exponent as i32);

                let fill = PaperFill {
                    order_id: order.id,
                    symbol: order.symbol.clone(),
                    side: format!("{:?}", order.side),
                    fill_price,
                    fill_qty,
                    timestamp: Utc::now(),
                };

                // Update position
                match order.side {
                    Side::Buy => {
                        if self.position >= 0.0 {
                            // Adding to long or opening long
                            let new_cost = self.position * self.avg_entry_price + fill_price * fill_qty;
                            self.position += fill_qty;
                            self.avg_entry_price = new_cost / self.position;
                        } else {
                            // Covering short
                            let pnl = (self.avg_entry_price - fill_price) * fill_qty.min(-self.position);
                            self.realized_pnl += pnl;
                            self.position += fill_qty;
                            if self.position > 0.0 {
                                self.avg_entry_price = fill_price;
                            }
                        }
                    }
                    Side::Sell => {
                        if self.position <= 0.0 {
                            // Adding to short or opening short
                            let new_cost =
                                (-self.position) * self.avg_entry_price + fill_price * fill_qty;
                            self.position -= fill_qty;
                            self.avg_entry_price = new_cost / (-self.position);
                        } else {
                            // Closing long
                            let pnl = (fill_price - self.avg_entry_price) * fill_qty.min(self.position);
                            self.realized_pnl += pnl;
                            self.position -= fill_qty;
                            if self.position < 0.0 {
                                self.avg_entry_price = fill_price;
                            }
                        }
                    }
                }

                info!(
                    "Paper fill: {:?} {} @ {} (pnl={:.2})",
                    order.side, fill_qty, fill_price, self.realized_pnl
                );

                self.fills.push(fill.clone());
                new_fills.push(fill);
                filled_order_ids.push(order.id);
            }
        }

        // Remove filled orders
        self.orders.retain(|o| !filled_order_ids.contains(&o.id));

        Ok(new_fills)
    }

    /// Get best bid price as f64.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids
            .iter()
            .next_back()
            .map(|(&price, _)| price as f64 * 10f64.powi(self.price_exponent as i32))
    }

    /// Get best ask price as f64.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks
            .iter()
            .next()
            .map(|(&price, _)| price as f64 * 10f64.powi(self.price_exponent as i32))
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Get realized PnL.
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get all fills.
    pub fn fills(&self) -> &[PaperFill] {
        &self.fills
    }

    /// Get pending orders.
    pub fn pending_orders(&self) -> &[PaperOrder] {
        &self.orders
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
    pub async fn process_depth_event(&mut self, event: DepthEvent) -> Result<Vec<PaperFill>> {
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
    use kubera_models::depth::{DepthLevel, IntegrityTier};

    #[test]
    fn test_paper_venue_fill_buy() {
        let mut venue = PaperVenue::new(-2, -8);

        // Apply snapshot with asks at 90001 and 90002
        let snapshot = DepthEvent {
            ts: Utc::now(),
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
        venue.submit_order("BTCUSDT".to_string(), Side::Buy, 90001.50, 0.3);

        // Next depth update should trigger fill
        let diff = DepthEvent {
            ts: Utc::now(),
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

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, "Buy");
        assert!((fills[0].fill_price - 90001.0).abs() < 0.01);
        assert!(venue.position() > 0.0);
    }

    #[test]
    fn test_paper_venue_no_fill_below_ask() {
        let mut venue = PaperVenue::new(-2, -8);

        // Apply snapshot with best ask at 90001
        let snapshot = DepthEvent {
            ts: Utc::now(),
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

        // Submit buy order at 90000 (below best ask)
        venue.submit_order("BTCUSDT".to_string(), Side::Buy, 90000.0, 0.3);

        // Diff update should NOT trigger fill
        let diff = DepthEvent {
            ts: Utc::now(),
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
