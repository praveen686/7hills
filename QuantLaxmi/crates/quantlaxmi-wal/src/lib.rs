//! # QuantLaxmi Write-Ahead Log (WAL)
//!
//! JSONL-based write-ahead log for capturing all trading events with
//! correlation IDs for full causality reconstruction.
//!
//! ## Event Types
//! - `MarketEvent` - Market data (quotes, depth, trades)
//! - `DecisionEvent` - Strategy decisions
//! - `OrderEvent` - Order lifecycle
//! - `FillEvent` - Execution confirmations
//! - `RiskEvent` - Risk limit violations
//!
//! ## WAL Structure
//! ```text
//! sessions/{session_id}/
//!   wal/
//!     market.jsonl      # MarketEvent records
//!     decisions.jsonl   # DecisionEvent records
//!     orders.jsonl      # OrderEvent records
//!     fills.jsonl       # FillEvent records
//!     risk.jsonl        # RiskEvent records
//!   manifest.json       # Session metadata + WAL checksums
//! ```
//!
//! ## Replay Protocol
//! 1. Read manifest.json for session metadata
//! 2. Verify WAL file checksums match manifest
//! 3. Stream events in timestamp order across all WAL files
//! 4. Validate causality (decisions -> orders -> fills)

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

// Re-export canonical events
pub use quantlaxmi_models::events::{
    CorrelationContext, DecisionEvent, MarketSnapshot, ParseMantissaError, QuoteEvent,
    parse_to_mantissa_pure,
};
pub use quantlaxmi_models::{FillEvent, OrderEvent, RiskEvent};

/// WAL record types for polymorphic storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WalRecord {
    /// Market data event (quotes, depth, trades)
    Market(WalMarketRecord),
    /// Strategy decision
    Decision(DecisionEvent),
    /// Order lifecycle event
    Order(OrderEvent),
    /// Execution fill
    Fill(FillEvent),
    /// Risk violation
    Risk(RiskEvent),
}

/// Market event wrapper for WAL storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalMarketRecord {
    /// Timestamp
    pub ts: DateTime<Utc>,
    /// Symbol
    pub symbol: String,
    /// Event payload (quote, depth, trade)
    pub payload: MarketPayload,
    /// Correlation context
    #[serde(default, flatten)]
    pub ctx: CorrelationContext,
}

/// Market event payload variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MarketPayload {
    /// Best bid/ask quote
    Quote {
        bid_price_mantissa: i64,
        ask_price_mantissa: i64,
        bid_qty_mantissa: i64,
        ask_qty_mantissa: i64,
        price_exponent: i8,
        qty_exponent: i8,
    },
    /// Depth update
    Depth {
        first_update_id: u64,
        last_update_id: u64,
        bids: Vec<DepthLevel>,
        asks: Vec<DepthLevel>,
        is_snapshot: bool,
    },
    /// Trade event
    Trade {
        trade_id: i64,
        price_mantissa: i64,
        qty_mantissa: i64,
        is_buyer_maker: bool,
    },
}

/// Depth level (price, quantity as mantissas).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLevel {
    pub price: i64,
    pub qty: i64,
}

// =============================================================================
// WAL WRITER
// =============================================================================

/// Async JSONL WAL writer with file rotation support.
pub struct WalWriter {
    session_dir: PathBuf,
    market_file: Option<tokio::fs::File>,
    decision_file: Option<tokio::fs::File>,
    order_file: Option<tokio::fs::File>,
    fill_file: Option<tokio::fs::File>,
    risk_file: Option<tokio::fs::File>,
    counts: WalCounts,
}

/// Event counts for manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalCounts {
    pub market_events: u64,
    pub decision_events: u64,
    pub order_events: u64,
    pub fill_events: u64,
    pub risk_events: u64,
}

impl WalWriter {
    /// Create a new WAL writer for a session.
    pub async fn new(session_dir: &Path) -> Result<Self> {
        let wal_dir = session_dir.join("wal");
        tokio::fs::create_dir_all(&wal_dir)
            .await
            .context("create WAL directory")?;

        Ok(Self {
            session_dir: session_dir.to_path_buf(),
            market_file: None,
            decision_file: None,
            order_file: None,
            fill_file: None,
            risk_file: None,
            counts: WalCounts::default(),
        })
    }

    /// Get the session directory.
    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    /// Write a market event.
    pub async fn write_market(&mut self, record: WalMarketRecord) -> Result<()> {
        let file = if let Some(ref mut f) = self.market_file {
            f
        } else {
            let path = self.session_dir.join("wal/market.jsonl");
            self.market_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open market WAL")?,
            );
            self.market_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&record).context("serialize market record")?;
        line.push(b'\n');
        file.write_all(&line).await.context("write market record")?;
        self.counts.market_events += 1;
        Ok(())
    }

    /// Write a decision event.
    pub async fn write_decision(&mut self, event: DecisionEvent) -> Result<()> {
        let file = if let Some(ref mut f) = self.decision_file {
            f
        } else {
            let path = self.session_dir.join("wal/decisions.jsonl");
            self.decision_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open decisions WAL")?,
            );
            self.decision_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&event).context("serialize decision")?;
        line.push(b'\n');
        file.write_all(&line).await.context("write decision")?;
        self.counts.decision_events += 1;
        Ok(())
    }

    /// Write an order event.
    pub async fn write_order(&mut self, event: OrderEvent) -> Result<()> {
        let file = if let Some(ref mut f) = self.order_file {
            f
        } else {
            let path = self.session_dir.join("wal/orders.jsonl");
            self.order_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open orders WAL")?,
            );
            self.order_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&event).context("serialize order")?;
        line.push(b'\n');
        file.write_all(&line).await.context("write order")?;
        self.counts.order_events += 1;
        Ok(())
    }

    /// Write a fill event.
    pub async fn write_fill(&mut self, event: FillEvent) -> Result<()> {
        let file = if let Some(ref mut f) = self.fill_file {
            f
        } else {
            let path = self.session_dir.join("wal/fills.jsonl");
            self.fill_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open fills WAL")?,
            );
            self.fill_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&event).context("serialize fill")?;
        line.push(b'\n');
        file.write_all(&line).await.context("write fill")?;
        self.counts.fill_events += 1;
        Ok(())
    }

    /// Write a risk event.
    pub async fn write_risk(&mut self, event: RiskEvent) -> Result<()> {
        let file = if let Some(ref mut f) = self.risk_file {
            f
        } else {
            let path = self.session_dir.join("wal/risk.jsonl");
            self.risk_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open risk WAL")?,
            );
            self.risk_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&event).context("serialize risk event")?;
        line.push(b'\n');
        file.write_all(&line).await.context("write risk event")?;
        self.counts.risk_events += 1;
        Ok(())
    }

    /// Flush all WAL files.
    pub async fn flush(&mut self) -> Result<()> {
        if let Some(ref mut f) = self.market_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.decision_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.order_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.fill_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.risk_file {
            f.flush().await?;
        }
        Ok(())
    }

    /// Get current event counts.
    pub fn counts(&self) -> &WalCounts {
        &self.counts
    }

    /// Finalize WAL and return file hashes for manifest.
    pub async fn finalize(&mut self) -> Result<WalManifest> {
        self.flush().await?;

        let mut files = HashMap::new();
        let wal_dir = self.session_dir.join("wal");

        // Compute hashes for each file
        for (name, count) in [
            ("market.jsonl", self.counts.market_events),
            ("decisions.jsonl", self.counts.decision_events),
            ("orders.jsonl", self.counts.order_events),
            ("fills.jsonl", self.counts.fill_events),
            ("risk.jsonl", self.counts.risk_events),
        ] {
            let path = wal_dir.join(name);
            if path.exists() {
                let bytes = std::fs::read(&path)?;
                let hash = sha256_hex(&bytes);
                files.insert(
                    name.to_string(),
                    WalFileInfo {
                        path: format!("wal/{}", name),
                        sha256: hash,
                        record_count: count,
                        bytes_len: bytes.len(),
                    },
                );
            }
        }

        Ok(WalManifest {
            created_at: Utc::now(),
            counts: self.counts.clone(),
            files,
        })
    }
}

/// WAL manifest for session metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalManifest {
    pub created_at: DateTime<Utc>,
    pub counts: WalCounts,
    pub files: HashMap<String, WalFileInfo>,
}

/// WAL file metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalFileInfo {
    pub path: String,
    pub sha256: String,
    pub record_count: u64,
    pub bytes_len: usize,
}

// =============================================================================
// WAL READER
// =============================================================================

/// Synchronous JSONL WAL reader.
pub struct WalReader {
    session_dir: PathBuf,
}

impl WalReader {
    /// Open a WAL for reading.
    pub fn open(session_dir: &Path) -> Result<Self> {
        if !session_dir.exists() {
            anyhow::bail!("Session directory does not exist: {:?}", session_dir);
        }
        Ok(Self {
            session_dir: session_dir.to_path_buf(),
        })
    }

    /// Read market events.
    pub fn read_market_events(&self) -> Result<Vec<WalMarketRecord>> {
        self.read_jsonl("wal/market.jsonl")
    }

    /// Read decision events.
    pub fn read_decisions(&self) -> Result<Vec<DecisionEvent>> {
        self.read_jsonl("wal/decisions.jsonl")
    }

    /// Read order events.
    pub fn read_orders(&self) -> Result<Vec<OrderEvent>> {
        self.read_jsonl("wal/orders.jsonl")
    }

    /// Read fill events.
    pub fn read_fills(&self) -> Result<Vec<FillEvent>> {
        self.read_jsonl("wal/fills.jsonl")
    }

    /// Read risk events.
    pub fn read_risk_events(&self) -> Result<Vec<RiskEvent>> {
        self.read_jsonl("wal/risk.jsonl")
    }

    /// Read JSONL file into typed records.
    fn read_jsonl<T: DeserializeOwned>(&self, rel_path: &str) -> Result<Vec<T>> {
        let path = self.session_dir.join(rel_path);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = std::fs::File::open(&path)
            .with_context(|| format!("open WAL file: {:?}", path))?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("read line {}", line_num + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: T = serde_json::from_str(&line)
                .with_context(|| format!("parse line {}: {}", line_num + 1, &line[..line.len().min(100)]))?;
            records.push(record);
        }

        Ok(records)
    }

    /// Verify WAL file checksums against manifest.
    pub fn verify_integrity(&self, manifest: &WalManifest) -> Result<IntegrityReport> {
        let mut report = IntegrityReport::default();

        for (name, info) in &manifest.files {
            let path = self.session_dir.join(&info.path);
            if !path.exists() {
                report.missing_files.push(name.clone());
                continue;
            }

            let bytes = std::fs::read(&path)?;
            let actual_hash = sha256_hex(&bytes);

            if actual_hash != info.sha256 {
                report.hash_mismatches.push(HashMismatch {
                    file: name.clone(),
                    expected: info.sha256.clone(),
                    actual: actual_hash,
                });
            } else {
                report.verified_files.push(name.clone());
            }
        }

        report.passed = report.missing_files.is_empty() && report.hash_mismatches.is_empty();
        Ok(report)
    }
}

/// Integrity verification report.
#[derive(Debug, Clone, Default)]
pub struct IntegrityReport {
    pub passed: bool,
    pub verified_files: Vec<String>,
    pub missing_files: Vec<String>,
    pub hash_mismatches: Vec<HashMismatch>,
}

/// Hash mismatch detail.
#[derive(Debug, Clone)]
pub struct HashMismatch {
    pub file: String,
    pub expected: String,
    pub actual: String,
}

// =============================================================================
// UTILITIES
// =============================================================================

/// Compute SHA-256 hash as lowercase hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wal_writer_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("test_session");

        // Write some events
        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        let market = WalMarketRecord {
            ts: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            payload: MarketPayload::Quote {
                bid_price_mantissa: 9000012,
                ask_price_mantissa: 9000015,
                bid_qty_mantissa: 150000000,
                ask_qty_mantissa: 200000000,
                price_exponent: -2,
                qty_exponent: -8,
            },
            ctx: CorrelationContext::new("session-1", "run-1"),
        };

        writer.write_market(market.clone()).await.unwrap();
        writer.write_market(market).await.unwrap();

        let manifest = writer.finalize().await.unwrap();
        assert_eq!(manifest.counts.market_events, 2);

        // Read events back
        let reader = WalReader::open(&session_dir).unwrap();
        let events = reader.read_market_events().unwrap();
        assert_eq!(events.len(), 2);

        // Verify integrity
        let report = reader.verify_integrity(&manifest).unwrap();
        assert!(report.passed);
    }
}
