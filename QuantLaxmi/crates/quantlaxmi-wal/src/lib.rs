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

// Re-export admission types (Phase 18)
pub use quantlaxmi_models::{AdmissionDecision, AdmissionOutcome};

// Re-export order intent types (Phase 23B)
pub use quantlaxmi_models::OrderIntentRecord;

// Re-export execution fill types (Phase 24A)
pub use quantlaxmi_models::ExecutionFillRecord;

// Re-export position update types (Phase 24D)
pub use quantlaxmi_models::PositionUpdateRecord;

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
    admission_file: Option<tokio::fs::File>,
    order_intent_file: Option<tokio::fs::File>,
    execution_fills_file: Option<tokio::fs::File>,
    position_updates_file: Option<tokio::fs::File>,
    counts: WalCounts,
    /// Track last seen seq per session for monotonicity enforcement (execution fills)
    execution_fills_last_seq: std::collections::HashMap<String, u64>,
    /// Track last seen seq per session for monotonicity enforcement (position updates)
    position_updates_last_seq: std::collections::HashMap<String, u64>,
}

/// Event counts for manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WalCounts {
    pub market_events: u64,
    pub decision_events: u64,
    pub order_events: u64,
    pub fill_events: u64,
    pub risk_events: u64,
    pub admission_events: u64,
    pub order_intent_events: u64,
    pub execution_fills_events: u64,
    pub position_updates_events: u64,
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
            admission_file: None,
            order_intent_file: None,
            execution_fills_file: None,
            position_updates_file: None,
            counts: WalCounts::default(),
            execution_fills_last_seq: std::collections::HashMap::new(),
            position_updates_last_seq: std::collections::HashMap::new(),
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

    /// Write a signal admission decision (Phase 18).
    ///
    /// Written for BOTH Admit and Refuse outcomes - every admission attempt
    /// produces an auditable artifact.
    pub async fn write_admission(&mut self, decision: AdmissionDecision) -> Result<()> {
        let file = if let Some(ref mut f) = self.admission_file {
            f
        } else {
            let path = self.session_dir.join("wal/signals_admission.jsonl");
            self.admission_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open signals_admission WAL")?,
            );
            self.admission_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&decision).context("serialize admission decision")?;
        line.push(b'\n');
        file.write_all(&line)
            .await
            .context("write admission decision")?;
        self.counts.admission_events += 1;
        Ok(())
    }

    /// Write an order intent record (Phase 23B).
    ///
    /// Written BEFORE acting on permission — every order intent produces
    /// an auditable artifact regardless of permit/refuse outcome.
    pub async fn write_order_intent(&mut self, record: OrderIntentRecord) -> Result<()> {
        let file = if let Some(ref mut f) = self.order_intent_file {
            f
        } else {
            let path = self.session_dir.join("wal/order_intent.jsonl");
            self.order_intent_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open order_intent WAL")?,
            );
            self.order_intent_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&record).context("serialize order intent record")?;
        line.push(b'\n');
        file.write_all(&line)
            .await
            .context("write order intent record")?;
        self.counts.order_intent_events += 1;
        Ok(())
    }

    /// Write an execution fill record (Phase 24A).
    ///
    /// Written BEFORE updating position/ledger state — every fill produces
    /// an auditable artifact regardless of fill type (full/partial).
    ///
    /// Enforces monotonic seq within each session_id (duplicate key = hard error).
    pub async fn write_execution_fill(&mut self, record: ExecutionFillRecord) -> Result<()> {
        // Enforce monotonic seq per session (duplicate key = hard error)
        let last_seq = self
            .execution_fills_last_seq
            .get(&record.session_id)
            .copied();
        if let Some(prev) = last_seq
            && record.seq <= prev
        {
            anyhow::bail!(
                "Execution fill seq monotonicity violation: session_id={}, prev_seq={}, new_seq={} (duplicate key = broken seq monotonicity)",
                record.session_id,
                prev,
                record.seq
            );
        }
        self.execution_fills_last_seq
            .insert(record.session_id.clone(), record.seq);

        let file = if let Some(ref mut f) = self.execution_fills_file {
            f
        } else {
            let path = self.session_dir.join("wal/execution_fills.jsonl");
            self.execution_fills_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open execution_fills WAL")?,
            );
            self.execution_fills_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&record).context("serialize execution fill record")?;
        line.push(b'\n');
        file.write_all(&line)
            .await
            .context("write execution fill record")?;
        self.counts.execution_fills_events += 1;
        Ok(())
    }

    /// Write a position update record (Phase 24D).
    ///
    /// Records the post-state snapshot and deltas after applying a fill.
    /// Written AFTER state update, BEFORE callbacks that observe the updated state.
    ///
    /// Enforces monotonic seq within each session_id (duplicate key = hard error).
    pub async fn write_position_update(&mut self, record: PositionUpdateRecord) -> Result<()> {
        // Enforce monotonic seq per session (duplicate key = hard error)
        let last_seq = self
            .position_updates_last_seq
            .get(&record.session_id)
            .copied();
        if let Some(prev) = last_seq
            && record.seq <= prev
        {
            anyhow::bail!(
                "Position update seq monotonicity violation: session_id={}, prev_seq={}, new_seq={} (duplicate key = broken seq monotonicity)",
                record.session_id,
                prev,
                record.seq
            );
        }
        self.position_updates_last_seq
            .insert(record.session_id.clone(), record.seq);

        let file = if let Some(ref mut f) = self.position_updates_file {
            f
        } else {
            let path = self.session_dir.join("wal/position_updates.jsonl");
            self.position_updates_file = Some(
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .await
                    .context("open position_updates WAL")?,
            );
            self.position_updates_file.as_mut().unwrap()
        };

        let mut line = serde_json::to_vec(&record).context("serialize position update record")?;
        line.push(b'\n');
        file.write_all(&line)
            .await
            .context("write position update record")?;
        self.counts.position_updates_events += 1;
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
        if let Some(ref mut f) = self.admission_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.order_intent_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.execution_fills_file {
            f.flush().await?;
        }
        if let Some(ref mut f) = self.position_updates_file {
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
            ("signals_admission.jsonl", self.counts.admission_events),
            ("order_intent.jsonl", self.counts.order_intent_events),
            ("execution_fills.jsonl", self.counts.execution_fills_events),
            (
                "position_updates.jsonl",
                self.counts.position_updates_events,
            ),
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

impl WalManifest {
    /// Load manifest from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read manifest {}: {}", path.display(), e))?;
        let manifest: WalManifest = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse manifest {}: {}", path.display(), e))?;
        Ok(manifest)
    }
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

    /// Read signal admission decisions (Phase 18/19C).
    pub fn read_admission_decisions(&self) -> Result<Vec<AdmissionDecision>> {
        self.read_jsonl("wal/signals_admission.jsonl")
    }

    /// Read order intent records (Phase 23B).
    pub fn read_order_intent_records(&self) -> Result<Vec<OrderIntentRecord>> {
        self.read_jsonl("wal/order_intent.jsonl")
    }

    /// Read execution fill records (Phase 24A).
    pub fn read_execution_fills(&self) -> Result<Vec<ExecutionFillRecord>> {
        self.read_jsonl("wal/execution_fills.jsonl")
    }

    /// Read position update records (Phase 24D).
    pub fn read_position_updates(&self) -> Result<Vec<PositionUpdateRecord>> {
        self.read_jsonl("wal/position_updates.jsonl")
    }

    /// Read JSONL file into typed records.
    fn read_jsonl<T: DeserializeOwned>(&self, rel_path: &str) -> Result<Vec<T>> {
        let path = self.session_dir.join(rel_path);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file =
            std::fs::File::open(&path).with_context(|| format!("open WAL file: {:?}", path))?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("read line {}", line_num + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: T = serde_json::from_str(&line).with_context(|| {
                format!(
                    "parse line {}: {}",
                    line_num + 1,
                    &line[..line.len().min(100)]
                )
            })?;
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

    /// Verify stream file digests against manifest using streaming hash.
    ///
    /// This is a hard-fail verification: if any digest mismatch is found, returns Err.
    /// Uses streaming SHA-256 to handle large files without loading into memory.
    ///
    /// # Returns
    /// - `Ok(())` if all files in manifest.files have matching digests
    /// - `Err` with details if any file is missing or has hash mismatch
    ///
    /// # Note
    /// If manifest.files is empty, returns Ok (no digests to verify).
    pub fn verify_stream_digests(&self, manifest: &WalManifest) -> Result<()> {
        if manifest.files.is_empty() {
            tracing::info!(
                session_dir = %self.session_dir.display(),
                "No stream digests in manifest, skipping digest verification"
            );
            return Ok(());
        }

        for (name, info) in &manifest.files {
            let path = self.session_dir.join(&info.path);

            if !path.exists() {
                anyhow::bail!("Stream file missing: {} (path: {})", name, path.display());
            }

            let actual_hash = sha256_file_hex(&path)?;

            if actual_hash != info.sha256 {
                anyhow::bail!(
                    "Stream digest mismatch for '{}': expected={}, actual={}, path={}",
                    name,
                    info.sha256,
                    actual_hash,
                    path.display()
                );
            }

            tracing::debug!(
                file = %name,
                hash = %actual_hash,
                "Stream digest verified"
            );
        }

        tracing::info!(
            session_dir = %self.session_dir.display(),
            files_verified = manifest.files.len(),
            "All stream digests verified"
        );

        Ok(())
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

/// Compute SHA-256 hash of a file using streaming (no full file load).
///
/// Reads the file in 64KB chunks to handle large files efficiently.
/// Returns the hash as a lowercase hex string.
pub fn sha256_file_hex(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open file {}: {}", path.display(), e))?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536]; // 64KB buffer

    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| anyhow::anyhow!("Failed to read file {}: {}", path.display(), e))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

// =============================================================================
// PHASE 19D: ADMISSION SUMMARY & REPLAY ENFORCEMENT
// =============================================================================

use std::collections::BTreeMap;

/// Schema version for admission summary serialization.
pub const ADMISSION_SUMMARY_SCHEMA_VERSION: &str = "1.0.0";

/// Per-signal admission statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalAdmissionStats {
    /// Total events where this signal was evaluated
    pub evaluated: u64,
    /// Events where this signal was admitted
    pub admitted: u64,
    /// Events where this signal was refused
    pub refused: u64,
    /// Refusal reason counts (sorted for determinism)
    pub refusal_reason_counts: BTreeMap<String, u64>,
}

/// Time range for admission summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionTimeRange {
    pub first_ts_ns: i64,
    pub last_ts_ns: i64,
}

/// Segment admission summary (Phase 19D).
///
/// Materialized from WAL admission records. Provides operator-readable
/// aggregates without requiring re-processing of raw events.
///
/// ## Invariants
/// - All maps are `BTreeMap` for deterministic serialization order
/// - Counts are integers (no floats)
/// - Summary is byte-reproducible from the same WAL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentAdmissionSummary {
    /// Schema version for forward compatibility
    pub schema_version: String,
    /// Session identifier
    pub session_id: String,
    /// Time range of evaluated events
    pub time_range: Option<AdmissionTimeRange>,

    // === Totals ===
    /// Events where admission gating was evaluated (required_signals non-empty)
    pub evaluated_events: u64,
    /// Events where ALL signals were admitted (strategy was called)
    pub admitted_events: u64,
    /// Events where ANY signal was refused (strategy NOT called)
    pub refused_events: u64,

    // === Per-signal breakdown ===
    /// Per-signal statistics (sorted by signal_id)
    pub per_signal: BTreeMap<String, SignalAdmissionStats>,

    // === Global refusal reasons ===
    /// Missing vendor fields histogram (merged across all signals)
    pub missing_vendor_field_counts: BTreeMap<String, u64>,
    /// Missing internal fields histogram (merged across all signals)
    pub missing_internal_field_counts: BTreeMap<String, u64>,
    /// Null vendor fields histogram (merged across all signals)
    pub null_vendor_field_counts: BTreeMap<String, u64>,
}

impl SegmentAdmissionSummary {
    /// Create empty summary for a session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            schema_version: ADMISSION_SUMMARY_SCHEMA_VERSION.to_string(),
            session_id: session_id.into(),
            time_range: None,
            evaluated_events: 0,
            admitted_events: 0,
            refused_events: 0,
            per_signal: BTreeMap::new(),
            missing_vendor_field_counts: BTreeMap::new(),
            missing_internal_field_counts: BTreeMap::new(),
            null_vendor_field_counts: BTreeMap::new(),
        }
    }

    /// Materialize summary from WAL admission decisions.
    ///
    /// Groups decisions by correlation_id to compute event-level outcomes.
    /// An event is "admitted" only if ALL per-signal decisions are Admit.
    pub fn from_decisions(session_id: impl Into<String>, decisions: &[AdmissionDecision]) -> Self {
        let mut summary = Self::new(session_id);

        if decisions.is_empty() {
            return summary;
        }

        // Track time range
        let mut first_ts_ns = i64::MAX;
        let mut last_ts_ns = i64::MIN;

        // Group by correlation_id to determine event-level outcome
        let mut events_by_correlation: BTreeMap<String, Vec<&AdmissionDecision>> = BTreeMap::new();

        for decision in decisions {
            // Update time range
            first_ts_ns = first_ts_ns.min(decision.ts_ns);
            last_ts_ns = last_ts_ns.max(decision.ts_ns);

            // Group by correlation_id (or use digest if no correlation)
            let key = decision
                .correlation_id
                .clone()
                .unwrap_or_else(|| decision.digest.clone());
            events_by_correlation.entry(key).or_default().push(decision);

            // Per-signal stats
            let signal_stats = summary
                .per_signal
                .entry(decision.signal_id.clone())
                .or_default();
            signal_stats.evaluated += 1;

            if decision.outcome == AdmissionOutcome::Admit {
                signal_stats.admitted += 1;
            } else {
                signal_stats.refused += 1;

                // Track refusal reasons
                for field in &decision.missing_vendor_fields {
                    let key = format!("missing:{}", field);
                    *signal_stats
                        .refusal_reason_counts
                        .entry(key.clone())
                        .or_default() += 1;
                    *summary
                        .missing_vendor_field_counts
                        .entry(field.to_string())
                        .or_default() += 1;
                }
                for field in &decision.null_vendor_fields {
                    let key = format!("null:{}", field);
                    *signal_stats
                        .refusal_reason_counts
                        .entry(key.clone())
                        .or_default() += 1;
                    *summary
                        .null_vendor_field_counts
                        .entry(field.to_string())
                        .or_default() += 1;
                }
                for field in &decision.missing_internal_fields {
                    let key = format!("missing_internal:{}", field);
                    *signal_stats
                        .refusal_reason_counts
                        .entry(key.clone())
                        .or_default() += 1;
                    *summary
                        .missing_internal_field_counts
                        .entry(field.to_string())
                        .or_default() += 1;
                }
            }
        }

        // Compute event-level outcomes
        for event_decisions in events_by_correlation.values() {
            summary.evaluated_events += 1;

            // Event is admitted only if ALL signals are admitted
            let all_admitted = event_decisions
                .iter()
                .all(|d| d.outcome == AdmissionOutcome::Admit);

            if all_admitted {
                summary.admitted_events += 1;
            } else {
                summary.refused_events += 1;
            }
        }

        // Set time range
        if first_ts_ns != i64::MAX {
            summary.time_range = Some(AdmissionTimeRange {
                first_ts_ns,
                last_ts_ns,
            });
        }

        summary
    }

    /// Write summary to a JSON file.
    pub fn write(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("serialize admission summary")?;
        std::fs::write(path, json).context("write admission summary")?;
        Ok(())
    }

    /// Load summary from a JSON file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("read admission summary")?;
        let summary: Self = serde_json::from_str(&content).context("parse admission summary")?;
        Ok(summary)
    }
}

// =============================================================================
// ADMISSION INDEX (for replay enforcement)
// =============================================================================

/// Index of admission decisions for replay enforcement.
///
/// Provides O(1) lookup by correlation_id during replay.
/// Used when `--enforce-admission-from-wal` is enabled.
#[derive(Debug, Clone)]
pub struct AdmissionIndex {
    /// Decisions indexed by correlation_id
    by_correlation: HashMap<String, Vec<AdmissionDecision>>,
    /// Total decisions in index
    pub total_decisions: usize,
}

/// Mismatch detected during admission enforcement.
#[derive(Debug, Clone)]
pub struct AdmissionMismatch {
    pub correlation_id: String,
    pub reason: AdmissionMismatchReason,
}

/// Reason for admission enforcement mismatch.
#[derive(Debug, Clone)]
pub enum AdmissionMismatchReason {
    /// No WAL entry found for this correlation_id
    MissingWalEntry,
    /// WAL says Admit but current state would Refuse
    AdmitButWouldRefuse { current_missing: Vec<String> },
    /// WAL says Refuse but current state would Admit
    RefuseButWouldAdmit,
    /// Signal set mismatch (strategy requirements changed)
    SignalSetMismatch {
        wal_signals: Vec<String>,
        current_signals: Vec<String>,
    },
    /// Missing fields don't match
    MissingFieldsMismatch {
        wal_missing: Vec<String>,
        current_missing: Vec<String>,
    },
}

impl std::fmt::Display for AdmissionMismatchReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingWalEntry => write!(f, "no WAL entry for correlation_id"),
            Self::AdmitButWouldRefuse { current_missing } => {
                write!(
                    f,
                    "WAL=Admit but current would Refuse (missing: {:?})",
                    current_missing
                )
            }
            Self::RefuseButWouldAdmit => write!(f, "WAL=Refuse but current would Admit"),
            Self::SignalSetMismatch {
                wal_signals,
                current_signals,
            } => {
                write!(
                    f,
                    "signal set changed: WAL={:?}, current={:?}",
                    wal_signals, current_signals
                )
            }
            Self::MissingFieldsMismatch {
                wal_missing,
                current_missing,
            } => {
                write!(
                    f,
                    "missing fields differ: WAL={:?}, current={:?}",
                    wal_missing, current_missing
                )
            }
        }
    }
}

impl AdmissionIndex {
    /// Build index from WAL decisions.
    pub fn from_decisions(decisions: Vec<AdmissionDecision>) -> Self {
        let total_decisions = decisions.len();
        let mut by_correlation: HashMap<String, Vec<AdmissionDecision>> = HashMap::new();

        for decision in decisions {
            let key = decision
                .correlation_id
                .clone()
                .unwrap_or_else(|| decision.digest.clone());
            by_correlation.entry(key).or_default().push(decision);
        }

        Self {
            by_correlation,
            total_decisions,
        }
    }

    /// Load index from WAL file.
    pub fn from_wal(session_dir: &Path) -> Result<Self> {
        let reader = WalReader::open(session_dir)?;
        let decisions = reader.read_admission_decisions()?;
        Ok(Self::from_decisions(decisions))
    }

    /// Look up decisions for a correlation_id.
    pub fn get(&self, correlation_id: &str) -> Option<&Vec<AdmissionDecision>> {
        self.by_correlation.get(correlation_id)
    }

    /// Check if correlation_id exists in index.
    pub fn contains(&self, correlation_id: &str) -> bool {
        self.by_correlation.contains_key(correlation_id)
    }

    /// Get the event-level outcome for a correlation_id.
    ///
    /// Returns Admit only if ALL per-signal decisions are Admit.
    pub fn event_outcome(&self, correlation_id: &str) -> Option<AdmissionOutcome> {
        self.by_correlation.get(correlation_id).map(|decisions| {
            if decisions
                .iter()
                .all(|d| d.outcome == AdmissionOutcome::Admit)
            {
                AdmissionOutcome::Admit
            } else {
                AdmissionOutcome::Refuse
            }
        })
    }

    /// Verify that current evaluation matches WAL record.
    ///
    /// Returns Ok(outcome) if matches, Err(mismatch) if differs.
    pub fn verify_decision(
        &self,
        correlation_id: &str,
        current_decisions: &[AdmissionDecision],
    ) -> std::result::Result<AdmissionOutcome, AdmissionMismatch> {
        let wal_decisions = match self.by_correlation.get(correlation_id) {
            Some(d) => d,
            None => {
                return Err(AdmissionMismatch {
                    correlation_id: correlation_id.to_string(),
                    reason: AdmissionMismatchReason::MissingWalEntry,
                });
            }
        };

        // Check signal set match
        let mut wal_signals: Vec<String> =
            wal_decisions.iter().map(|d| d.signal_id.clone()).collect();
        let mut current_signals: Vec<String> = current_decisions
            .iter()
            .map(|d| d.signal_id.clone())
            .collect();
        wal_signals.sort();
        current_signals.sort();

        if wal_signals != current_signals {
            return Err(AdmissionMismatch {
                correlation_id: correlation_id.to_string(),
                reason: AdmissionMismatchReason::SignalSetMismatch {
                    wal_signals,
                    current_signals,
                },
            });
        }

        // Check per-signal outcomes match
        let wal_outcome = if wal_decisions
            .iter()
            .all(|d| d.outcome == AdmissionOutcome::Admit)
        {
            AdmissionOutcome::Admit
        } else {
            AdmissionOutcome::Refuse
        };

        let current_outcome = if current_decisions
            .iter()
            .all(|d| d.outcome == AdmissionOutcome::Admit)
        {
            AdmissionOutcome::Admit
        } else {
            AdmissionOutcome::Refuse
        };

        if wal_outcome != current_outcome {
            match (wal_outcome, current_outcome) {
                (AdmissionOutcome::Admit, AdmissionOutcome::Refuse) => {
                    let current_missing: Vec<String> = current_decisions
                        .iter()
                        .flat_map(|d| d.missing_vendor_fields.iter().map(|f| f.to_string()))
                        .collect();
                    return Err(AdmissionMismatch {
                        correlation_id: correlation_id.to_string(),
                        reason: AdmissionMismatchReason::AdmitButWouldRefuse { current_missing },
                    });
                }
                (AdmissionOutcome::Refuse, AdmissionOutcome::Admit) => {
                    return Err(AdmissionMismatch {
                        correlation_id: correlation_id.to_string(),
                        reason: AdmissionMismatchReason::RefuseButWouldAdmit,
                    });
                }
                _ => {} // Same outcome
            }
        }

        // Optionally verify missing fields match (stricter check)
        // For now, we only check outcome match

        Ok(wal_outcome)
    }

    /// Number of unique correlation_ids (events) in index.
    pub fn event_count(&self) -> usize {
        self.by_correlation.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test manifest hash (all zeros for testing)
    const TEST_MANIFEST_HASH: [u8; 32] = [0u8; 32];

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

    #[test]
    fn test_verify_stream_digests_pass() {
        use std::collections::HashMap;

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path();

        // Create a fake stream file with known content
        let wal_dir = session_dir.join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let stream_content = b"line1\nline2\nline3\n";
        let stream_path = wal_dir.join("test_stream.jsonl");
        std::fs::write(&stream_path, stream_content).unwrap();

        // Compute expected hash
        let expected_hash = sha256_hex(stream_content);

        // Create manifest with correct digest
        let mut files = HashMap::new();
        files.insert(
            "test_stream".to_string(),
            WalFileInfo {
                path: "wal/test_stream.jsonl".to_string(),
                sha256: expected_hash.clone(),
                record_count: 3,
                bytes_len: stream_content.len(),
            },
        );

        let manifest = WalManifest {
            created_at: Utc::now(),
            counts: WalCounts::default(),
            files,
        };

        // Verify should pass
        let reader = WalReader::open(session_dir).unwrap();
        let result = reader.verify_stream_digests(&manifest);
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);
    }

    #[test]
    fn test_verify_stream_digests_fail_on_modified_file() {
        use std::collections::HashMap;

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path();

        // Create a fake stream file
        let wal_dir = session_dir.join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let original_content = b"original content";
        let stream_path = wal_dir.join("test_stream.jsonl");
        std::fs::write(&stream_path, original_content).unwrap();

        // Compute hash of original content
        let original_hash = sha256_hex(original_content);

        // Create manifest with original hash
        let mut files = HashMap::new();
        files.insert(
            "test_stream".to_string(),
            WalFileInfo {
                path: "wal/test_stream.jsonl".to_string(),
                sha256: original_hash,
                record_count: 1,
                bytes_len: original_content.len(),
            },
        );

        let manifest = WalManifest {
            created_at: Utc::now(),
            counts: WalCounts::default(),
            files,
        };

        // Modify the file
        std::fs::write(&stream_path, b"MODIFIED content").unwrap();

        // Verify should fail with expected/actual in error message
        let reader = WalReader::open(session_dir).unwrap();
        let result = reader.verify_stream_digests(&manifest);
        assert!(result.is_err(), "Expected Err, got Ok");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("expected=") && err_msg.contains("actual="),
            "Error should contain expected/actual hashes: {}",
            err_msg
        );
        assert!(
            err_msg.contains("test_stream"),
            "Error should contain file name: {}",
            err_msg
        );
    }

    #[test]
    fn test_verify_stream_digests_empty_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path();

        // Empty manifest (no files)
        let manifest = WalManifest {
            created_at: Utc::now(),
            counts: WalCounts::default(),
            files: std::collections::HashMap::new(),
        };

        let reader = WalReader::open(session_dir).unwrap();
        let result = reader.verify_stream_digests(&manifest);
        assert!(result.is_ok(), "Empty manifest should pass: {:?}", result);
    }

    #[test]
    fn test_sha256_file_hex_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test_file.bin");

        // Create a file with known content
        let content = b"hello world streaming test";
        std::fs::write(&file_path, content).unwrap();

        // Streaming hash should match in-memory hash
        let streaming_hash = sha256_file_hex(&file_path).unwrap();
        let memory_hash = sha256_hex(content);

        assert_eq!(streaming_hash, memory_hash);
    }

    // =========================================================================
    // PHASE 18: Signal Admission WAL Tests
    // =========================================================================

    #[tokio::test]
    async fn test_admission_wal_roundtrip_jsonl() {
        use quantlaxmi_models::{
            ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome, VendorField,
        };

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("admission_test");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write Admit decision
        let admit = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1706400000000000000,
            session_id: "sess_001".to_string(),
            signal_id: "book_imbalance".to_string(),
            outcome: AdmissionOutcome::Admit,
            missing_vendor_fields: vec![],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: Some("corr_001".to_string()),
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "abc123".to_string(),
        };
        writer.write_admission(admit.clone()).await.unwrap();

        // Write Refuse decision
        let refuse = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1706400000000001000,
            session_id: "sess_001".to_string(),
            signal_id: "book_imbalance".to_string(),
            outcome: AdmissionOutcome::Refuse,
            missing_vendor_fields: vec![VendorField::BuyQuantity],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: None,
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "def456".to_string(),
        };
        writer.write_admission(refuse.clone()).await.unwrap();

        writer.flush().await.unwrap();

        // Read back lines
        let wal_path = session_dir.join("wal/signals_admission.jsonl");
        assert!(wal_path.exists(), "signals_admission.jsonl should exist");

        let content = std::fs::read_to_string(&wal_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "Should have 2 lines");

        // Deserialize and verify
        let read_admit: AdmissionDecision = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(read_admit.outcome, AdmissionOutcome::Admit);
        assert_eq!(read_admit.signal_id, "book_imbalance");
        assert!(read_admit.missing_vendor_fields.is_empty());

        let read_refuse: AdmissionDecision = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(read_refuse.outcome, AdmissionOutcome::Refuse);
        assert_eq!(
            read_refuse.missing_vendor_fields,
            vec![VendorField::BuyQuantity]
        );
    }

    #[tokio::test]
    async fn test_admission_wal_path_is_correct() {
        use quantlaxmi_models::{ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome};

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("path_test_session");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        let decision = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1706400000000000000,
            session_id: "sess_001".to_string(),
            signal_id: "test_signal".to_string(),
            outcome: AdmissionOutcome::Admit,
            missing_vendor_fields: vec![],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: None,
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "test".to_string(),
        };
        writer.write_admission(decision).await.unwrap();
        writer.flush().await.unwrap();

        // Verify exact path
        let expected_path = session_dir.join("wal/signals_admission.jsonl");
        assert!(
            expected_path.exists(),
            "File should exist at exactly: {}",
            expected_path.display()
        );
    }

    #[tokio::test]
    async fn test_admission_wal_write_is_append_only() {
        use quantlaxmi_models::{ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome};

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("append_test");

        // First write
        {
            let mut writer = WalWriter::new(&session_dir).await.unwrap();
            let decision = AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 1,
                session_id: "sess".to_string(),
                signal_id: "sig".to_string(),
                outcome: AdmissionOutcome::Admit,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: None,
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d1".to_string(),
            };
            writer.write_admission(decision).await.unwrap();
            writer.flush().await.unwrap();
            // Writer dropped here
        }

        // Second write (reopen)
        {
            let mut writer = WalWriter::new(&session_dir).await.unwrap();
            let decision = AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 2,
                session_id: "sess".to_string(),
                signal_id: "sig".to_string(),
                outcome: AdmissionOutcome::Refuse,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: None,
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d2".to_string(),
            };
            writer.write_admission(decision).await.unwrap();
            writer.flush().await.unwrap();
        }

        // Verify both lines exist
        let wal_path = session_dir.join("wal/signals_admission.jsonl");
        let content = std::fs::read_to_string(&wal_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "File should have 2 lines after reopen + write"
        );
    }

    #[tokio::test]
    async fn test_refuse_still_writes_to_wal() {
        use quantlaxmi_models::{
            ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome, VendorField,
        };

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("refuse_test");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write ONLY a Refuse decision (simulating missing vendor field)
        let refuse = AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1706400000000000000,
            session_id: "sess_001".to_string(),
            signal_id: "book_imbalance".to_string(),
            outcome: AdmissionOutcome::Refuse,
            missing_vendor_fields: vec![VendorField::SellQuantity],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: None,
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "refuse_digest".to_string(),
        };
        writer.write_admission(refuse).await.unwrap();

        let manifest = writer.finalize().await.unwrap();

        // Verify count
        assert_eq!(
            manifest.counts.admission_events, 1,
            "Refuse must still be counted"
        );

        // Verify file exists and contains the refuse decision
        let wal_path = session_dir.join("wal/signals_admission.jsonl");
        let content = std::fs::read_to_string(&wal_path).unwrap();
        assert!(
            content.contains("Refuse"),
            "WAL should contain Refuse outcome"
        );
        // VendorField::SellQuantity serializes as "SellQuantity" by default
        assert!(
            content.contains("SellQuantity"),
            "WAL should contain missing field: {}",
            content
        );
    }

    // =========================================================================
    // PHASE 19D: Admission Summary & Index Tests
    // =========================================================================

    #[test]
    fn test_segment_admission_summary_counts_match_wal() {
        use quantlaxmi_models::{
            ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome, VendorField,
        };

        // Create test decisions
        let decisions = vec![
            // Event 1 (corr_001): book_imbalance Admit
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 1000,
                session_id: "sess".to_string(),
                signal_id: "book_imbalance".to_string(),
                outcome: AdmissionOutcome::Admit,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_001".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d1".to_string(),
            },
            // Event 2 (corr_002): book_imbalance Refuse
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 2000,
                session_id: "sess".to_string(),
                signal_id: "book_imbalance".to_string(),
                outcome: AdmissionOutcome::Refuse,
                missing_vendor_fields: vec![VendorField::BuyQuantity],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_002".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d2".to_string(),
            },
            // Event 3 (corr_003): spread Admit, book_imbalance Refuse (event refused)
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 3000,
                session_id: "sess".to_string(),
                signal_id: "spread".to_string(),
                outcome: AdmissionOutcome::Admit,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_003".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d3a".to_string(),
            },
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 3000,
                session_id: "sess".to_string(),
                signal_id: "book_imbalance".to_string(),
                outcome: AdmissionOutcome::Refuse,
                missing_vendor_fields: vec![VendorField::SellQuantity],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_003".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d3b".to_string(),
            },
        ];

        let summary = SegmentAdmissionSummary::from_decisions("test_session", &decisions);

        // Verify totals
        assert_eq!(summary.evaluated_events, 3, "3 unique correlation_ids");
        assert_eq!(summary.admitted_events, 1, "Only corr_001 fully admitted");
        assert_eq!(summary.refused_events, 2, "corr_002 and corr_003 refused");

        // Verify per-signal stats
        let book_stats = summary.per_signal.get("book_imbalance").unwrap();
        assert_eq!(book_stats.evaluated, 3);
        assert_eq!(book_stats.admitted, 1);
        assert_eq!(book_stats.refused, 2);

        let spread_stats = summary.per_signal.get("spread").unwrap();
        assert_eq!(spread_stats.evaluated, 1);
        assert_eq!(spread_stats.admitted, 1);
        assert_eq!(spread_stats.refused, 0);

        // Verify missing field counts (VendorField::Display uses snake_case)
        assert_eq!(
            *summary
                .missing_vendor_field_counts
                .get("buy_quantity")
                .unwrap_or(&0),
            1
        );
        assert_eq!(
            *summary
                .missing_vendor_field_counts
                .get("sell_quantity")
                .unwrap_or(&0),
            1
        );

        // Verify time range
        let range = summary.time_range.as_ref().unwrap();
        assert_eq!(range.first_ts_ns, 1000);
        assert_eq!(range.last_ts_ns, 3000);
    }

    #[test]
    fn test_segment_admission_summary_deterministic_ordering() {
        use quantlaxmi_models::{
            ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome, VendorField,
        };

        // Create decisions in random order
        let decisions1 = vec![
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 2000,
                session_id: "s".to_string(),
                signal_id: "zebra".to_string(),
                outcome: AdmissionOutcome::Refuse,
                missing_vendor_fields: vec![VendorField::AskPrice],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("c2".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d2".to_string(),
            },
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 1000,
                session_id: "s".to_string(),
                signal_id: "alpha".to_string(),
                outcome: AdmissionOutcome::Admit,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("c1".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d1".to_string(),
            },
        ];

        // Create same decisions in different order
        let decisions2 = vec![decisions1[1].clone(), decisions1[0].clone()];

        let summary1 = SegmentAdmissionSummary::from_decisions("s", &decisions1);
        let summary2 = SegmentAdmissionSummary::from_decisions("s", &decisions2);

        // Serialize both
        let json1 = serde_json::to_string_pretty(&summary1).unwrap();
        let json2 = serde_json::to_string_pretty(&summary2).unwrap();

        // Should be byte-identical (BTreeMap ensures sorted keys)
        assert_eq!(
            json1, json2,
            "Summary should be deterministic regardless of input order"
        );
    }

    #[test]
    fn test_admission_index_lookup() {
        use quantlaxmi_models::{ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome};

        let decisions = vec![
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 1000,
                session_id: "s".to_string(),
                signal_id: "sig1".to_string(),
                outcome: AdmissionOutcome::Admit,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_a".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d1".to_string(),
            },
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 2000,
                session_id: "s".to_string(),
                signal_id: "sig1".to_string(),
                outcome: AdmissionOutcome::Refuse,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_b".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d2".to_string(),
            },
        ];

        let index = AdmissionIndex::from_decisions(decisions);

        // Test lookup
        assert!(index.contains("corr_a"));
        assert!(index.contains("corr_b"));
        assert!(!index.contains("corr_c"));

        // Test event outcome
        assert_eq!(index.event_outcome("corr_a"), Some(AdmissionOutcome::Admit));
        assert_eq!(
            index.event_outcome("corr_b"),
            Some(AdmissionOutcome::Refuse)
        );
        assert_eq!(index.event_outcome("corr_c"), None);

        // Test counts
        assert_eq!(index.event_count(), 2);
        assert_eq!(index.total_decisions, 2);
    }

    #[test]
    fn test_admission_index_multi_signal_event_outcome() {
        use quantlaxmi_models::{ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome};

        // Event with multiple signals - ALL must admit for event to admit
        let decisions = vec![
            // Signal 1: Admit
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 1000,
                session_id: "s".to_string(),
                signal_id: "sig1".to_string(),
                outcome: AdmissionOutcome::Admit,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_multi".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d1".to_string(),
            },
            // Signal 2: Refuse → event should be refused
            AdmissionDecision {
                schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
                ts_ns: 1000,
                session_id: "s".to_string(),
                signal_id: "sig2".to_string(),
                outcome: AdmissionOutcome::Refuse,
                missing_vendor_fields: vec![],
                null_vendor_fields: vec![],
                missing_internal_fields: vec![],
                correlation_id: Some("corr_multi".to_string()),
                manifest_version_hash: TEST_MANIFEST_HASH,
                digest: "d2".to_string(),
            },
        ];

        let index = AdmissionIndex::from_decisions(decisions);

        // Event has one Refuse → overall outcome is Refuse
        assert_eq!(
            index.event_outcome("corr_multi"),
            Some(AdmissionOutcome::Refuse),
            "Event with ANY refused signal should be refused"
        );
    }

    #[test]
    fn test_admission_index_verify_decision_mismatch() {
        use quantlaxmi_models::{
            ADMISSION_SCHEMA_VERSION, AdmissionDecision, AdmissionOutcome, VendorField,
        };

        let wal_decisions = vec![AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1000,
            session_id: "s".to_string(),
            signal_id: "sig1".to_string(),
            outcome: AdmissionOutcome::Admit,
            missing_vendor_fields: vec![],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: Some("corr_x".to_string()),
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "d1".to_string(),
        }];

        let index = AdmissionIndex::from_decisions(wal_decisions);

        // Test: Missing WAL entry
        let result = index.verify_decision("corr_unknown", &[]);
        assert!(result.is_err());
        match result.unwrap_err().reason {
            AdmissionMismatchReason::MissingWalEntry => {}
            other => panic!("Expected MissingWalEntry, got {:?}", other),
        }

        // Test: WAL says Admit but current would Refuse
        let current_refuse = vec![AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1000,
            session_id: "s".to_string(),
            signal_id: "sig1".to_string(),
            outcome: AdmissionOutcome::Refuse,
            missing_vendor_fields: vec![VendorField::BidPrice],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: Some("corr_x".to_string()),
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "d1_new".to_string(),
        }];

        let result = index.verify_decision("corr_x", &current_refuse);
        assert!(result.is_err());
        match result.unwrap_err().reason {
            AdmissionMismatchReason::AdmitButWouldRefuse { .. } => {}
            other => panic!("Expected AdmitButWouldRefuse, got {:?}", other),
        }

        // Test: Matching decision
        let current_admit = vec![AdmissionDecision {
            schema_version: ADMISSION_SCHEMA_VERSION.to_string(),
            ts_ns: 1000,
            session_id: "s".to_string(),
            signal_id: "sig1".to_string(),
            outcome: AdmissionOutcome::Admit,
            missing_vendor_fields: vec![],
            null_vendor_fields: vec![],
            missing_internal_fields: vec![],
            correlation_id: Some("corr_x".to_string()),
            manifest_version_hash: TEST_MANIFEST_HASH,
            digest: "d1".to_string(),
        }];

        let result = index.verify_decision("corr_x", &current_admit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), AdmissionOutcome::Admit);
    }

    // =========================================================================
    // PHASE 23B: Order Intent WAL Tests
    // =========================================================================

    #[tokio::test]
    async fn test_order_intent_wal_roundtrip() {
        use quantlaxmi_models::{
            ORDER_INTENT_SCHEMA_VERSION, OrderIntentPermission, OrderIntentRecord, OrderIntentSide,
            OrderIntentType, OrderRefuseReason,
        };

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("order_intent_test");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write Permit record
        let permit = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("test_session")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .order_type(OrderIntentType::Limit)
            .qty(100000000, -8)
            .limit_price(5000000, -2)
            .correlation_id("corr_001")
            .parent_admission_digest("admit_digest_abc")
            .build_permit();

        writer.write_order_intent(permit.clone()).await.unwrap();

        // Write Refuse record
        let refuse = OrderIntentRecord::builder("spread_passive", "BTCUSDT")
            .ts_ns(1706400000001000000)
            .session_id("test_session")
            .seq(2)
            .side(OrderIntentSide::Sell)
            .order_type(OrderIntentType::Market)
            .qty(50000000, -8)
            .price_exponent(-2)
            .correlation_id("corr_002")
            .parent_admission_digest("admit_digest_def")
            .build_refuse(OrderRefuseReason::Custom {
                reason: "Passive strategy cannot emit market orders".to_string(),
            });

        writer.write_order_intent(refuse.clone()).await.unwrap();

        let manifest = writer.finalize().await.unwrap();

        // Verify counts
        assert_eq!(
            manifest.counts.order_intent_events, 2,
            "Should have 2 order intent events"
        );

        // Verify file exists in manifest
        assert!(
            manifest.files.contains_key("order_intent.jsonl"),
            "Manifest should contain order_intent.jsonl"
        );

        // Read back via WalReader
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_order_intent_records().unwrap();

        assert_eq!(records.len(), 2, "Should read 2 records");

        // Verify Permit record
        let read_permit = &records[0];
        assert_eq!(read_permit.schema_version, ORDER_INTENT_SCHEMA_VERSION);
        assert_eq!(read_permit.session_id, "test_session");
        assert_eq!(read_permit.seq, 1);
        assert_eq!(read_permit.strategy_id, "spread_passive");
        assert_eq!(read_permit.symbol, "BTCUSDT");
        assert_eq!(read_permit.side, OrderIntentSide::Buy);
        assert_eq!(read_permit.order_type, OrderIntentType::Limit);
        assert_eq!(read_permit.qty_mantissa, 100000000);
        assert_eq!(read_permit.qty_exponent, -8);
        assert_eq!(read_permit.limit_price_mantissa, Some(5000000));
        assert_eq!(read_permit.price_exponent, -2);
        assert_eq!(read_permit.permission, OrderIntentPermission::Permit);
        assert!(read_permit.refuse_reason.is_none());
        assert_eq!(read_permit.correlation_id, "corr_001");
        assert_eq!(read_permit.parent_admission_digest, "admit_digest_abc");
        assert!(!read_permit.digest.is_empty());
        assert_eq!(read_permit.digest, permit.digest);

        // Verify Refuse record
        let read_refuse = &records[1];
        assert_eq!(read_refuse.seq, 2);
        assert_eq!(read_refuse.side, OrderIntentSide::Sell);
        assert_eq!(read_refuse.order_type, OrderIntentType::Market);
        assert!(read_refuse.limit_price_mantissa.is_none());
        assert_eq!(read_refuse.permission, OrderIntentPermission::Refuse);
        assert!(read_refuse.refuse_reason.is_some());
        assert_eq!(read_refuse.digest, refuse.digest);

        // Verify refuse reason content
        if let Some(OrderRefuseReason::Custom { reason }) = &read_refuse.refuse_reason {
            assert!(reason.contains("market orders"));
        } else {
            panic!("Expected Custom refuse reason");
        }
    }

    #[tokio::test]
    async fn test_order_intent_wal_append_only() {
        use quantlaxmi_models::{OrderIntentRecord, OrderIntentSide};

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("append_test");

        // First write
        {
            let mut writer = WalWriter::new(&session_dir).await.unwrap();
            let record = OrderIntentRecord::builder("strategy_a", "BTCUSDT")
                .ts_ns(1)
                .session_id("sess")
                .seq(1)
                .side(OrderIntentSide::Buy)
                .qty(1000, -8)
                .correlation_id("c1")
                .parent_admission_digest("d1")
                .build_permit();
            writer.write_order_intent(record).await.unwrap();
            writer.flush().await.unwrap();
        }

        // Second write (reopen)
        {
            let mut writer = WalWriter::new(&session_dir).await.unwrap();
            let record = OrderIntentRecord::builder("strategy_a", "BTCUSDT")
                .ts_ns(2)
                .session_id("sess")
                .seq(2)
                .side(OrderIntentSide::Sell)
                .qty(2000, -8)
                .correlation_id("c2")
                .parent_admission_digest("d2")
                .build_permit();
            writer.write_order_intent(record).await.unwrap();
            writer.flush().await.unwrap();
        }

        // Verify both records exist
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_order_intent_records().unwrap();

        assert_eq!(
            records.len(),
            2,
            "File should have 2 records after reopen + write"
        );
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    #[tokio::test]
    async fn test_order_intent_wal_manifest_includes_hash() {
        use quantlaxmi_models::{OrderIntentRecord, OrderIntentSide};

        let dir = tempfile::tempdir().unwrap();
        let session_dir = dir.path().join("hash_test");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write a record
        let record = OrderIntentRecord::builder("test_strat", "ETHUSDT")
            .ts_ns(1000)
            .session_id("sess")
            .seq(1)
            .side(OrderIntentSide::Buy)
            .qty(5000, -8)
            .correlation_id("corr")
            .parent_admission_digest("parent")
            .build_permit();
        writer.write_order_intent(record).await.unwrap();

        let manifest = writer.finalize().await.unwrap();

        // Verify manifest has file info
        let file_info = manifest.files.get("order_intent.jsonl").unwrap();
        assert_eq!(file_info.path, "wal/order_intent.jsonl");
        assert_eq!(file_info.record_count, 1);
        assert!(
            !file_info.sha256.is_empty(),
            "SHA256 hash should be computed"
        );
        assert!(file_info.bytes_len > 0, "File should have content");

        // Verify integrity
        let reader = WalReader::open(&session_dir).unwrap();
        let report = reader.verify_integrity(&manifest).unwrap();
        assert!(report.passed, "Integrity check should pass");
        assert!(
            report
                .verified_files
                .contains(&"order_intent.jsonl".to_string()),
            "order_intent.jsonl should be verified"
        );
    }

    // =========================================================================
    // Phase 24A: Execution Fill WAL Tests
    // =========================================================================

    #[tokio::test]
    async fn test_execution_fill_wal_roundtrip() {
        use quantlaxmi_models::{ExecutionFillRecord, FillSide, FillType};

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write a full fill with fee
        let full_fill = ExecutionFillRecord::builder("test_strat", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("backtest_123")
            .seq(1)
            .parent_intent_seq(5)
            .parent_intent_digest("abc123def456")
            .side(FillSide::Buy)
            .qty(100000000, -8)
            .price(4200000, -2)
            .fee(420, -2)
            .venue("sim")
            .correlation_id("event_seq:99")
            .fill_type(FillType::Full)
            .build();
        writer
            .write_execution_fill(full_fill.clone())
            .await
            .unwrap();

        // Write a partial fill without fee
        let partial_fill = ExecutionFillRecord::builder("test_strat", "ETHUSDT")
            .ts_ns(1706400001000000000)
            .session_id("backtest_123")
            .seq(2)
            .parent_intent_seq(6)
            .side(FillSide::Sell)
            .qty(50000000, -8)
            .price(250000, -2)
            // No fee
            .venue("binance")
            .correlation_id("event_seq:100")
            .fill_type(FillType::Partial)
            .build();
        writer
            .write_execution_fill(partial_fill.clone())
            .await
            .unwrap();

        let manifest = writer.finalize().await.unwrap();

        // Verify manifest has file info
        assert!(
            manifest.files.contains_key("execution_fills.jsonl"),
            "Manifest should contain execution_fills.jsonl"
        );

        // Read back and verify
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_execution_fills().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], full_fill);
        assert_eq!(records[1], partial_fill);
    }

    #[tokio::test]
    async fn test_execution_fill_wal_monotonic_seq_enforcement() {
        use quantlaxmi_models::{ExecutionFillRecord, FillSide, FillType};

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write seq=1
        let fill1 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill1).await.unwrap();

        // Try to write duplicate seq=1 (should fail)
        let fill_dup = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1) // Duplicate!
            .side(FillSide::Sell)
            .qty(100, -8)
            .price(5001, -2)
            .venue("sim")
            .correlation_id("c2")
            .fill_type(FillType::Full)
            .build();
        let result = writer.write_execution_fill(fill_dup).await;
        assert!(result.is_err(), "Duplicate seq should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("monotonicity violation"),
            "Error should mention monotonicity: {}",
            err_msg
        );

        // Try to write seq=0 (backwards, should fail)
        let fill_backwards = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(0) // Backwards!
            .side(FillSide::Buy)
            .qty(50, -8)
            .price(4999, -2)
            .venue("sim")
            .correlation_id("c3")
            .fill_type(FillType::Partial)
            .build();
        let result = writer.write_execution_fill(fill_backwards).await;
        assert!(result.is_err(), "Backwards seq should fail");

        // seq=2 should succeed
        let fill2 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(2)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5002, -2)
            .venue("sim")
            .correlation_id("c4")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill2).await.unwrap();

        // Verify only 2 records written (seq=1 and seq=2)
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_execution_fills().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    #[tokio::test]
    async fn test_execution_fill_wal_different_sessions() {
        use quantlaxmi_models::{ExecutionFillRecord, FillSide, FillType};

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Different sessions can have same seq
        let fill_a1 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("session_a")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill_a1).await.unwrap();

        let fill_b1 = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("session_b")
            .seq(1) // Same seq, different session - OK
            .side(FillSide::Sell)
            .qty(200, -8)
            .price(5001, -2)
            .venue("sim")
            .correlation_id("c2")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill_b1).await.unwrap();

        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_execution_fills().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_execution_fill_wal_manifest_includes_hash() {
        use quantlaxmi_models::{ExecutionFillRecord, FillSide, FillType};

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        let fill = ExecutionFillRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1)
            .side(FillSide::Buy)
            .qty(100, -8)
            .price(5000, -2)
            .venue("sim")
            .correlation_id("c1")
            .fill_type(FillType::Full)
            .build();
        writer.write_execution_fill(fill).await.unwrap();

        let manifest = writer.finalize().await.unwrap();

        // Verify manifest has file info
        let file_info = manifest.files.get("execution_fills.jsonl").unwrap();
        assert_eq!(file_info.path, "wal/execution_fills.jsonl");
        assert_eq!(file_info.record_count, 1);
        assert!(
            !file_info.sha256.is_empty(),
            "SHA256 hash should be computed"
        );
        assert!(file_info.bytes_len > 0, "File should have content");

        // Verify integrity
        let reader = WalReader::open(&session_dir).unwrap();
        let report = reader.verify_integrity(&manifest).unwrap();
        assert!(report.passed, "Integrity check should pass");
        assert!(
            report
                .verified_files
                .contains(&"execution_fills.jsonl".to_string()),
            "execution_fills.jsonl should be verified"
        );
    }

    // =========================================================================
    // Phase 24D: Position Update WAL Tests
    // =========================================================================

    #[tokio::test]
    async fn test_position_update_wal_roundtrip() {
        use quantlaxmi_models::PositionUpdateRecord;

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write a buy position update
        let buy_update = PositionUpdateRecord::builder("strat_a", "BTCUSDT")
            .ts_ns(1706400000000000000)
            .session_id("test_session")
            .seq(1)
            .correlation_id("corr_1")
            .fill_seq(1)
            .position_qty(100000000, -8) // 1 BTC
            .avg_price(4200000, -2) // $42,000
            .cash_delta(-4200000, -2) // Spent $42,000
            .realized_pnl_delta(0, -2)
            .fee(420, -2) // $4.20 fee
            .venue("sim")
            .build();
        writer
            .write_position_update(buy_update.clone())
            .await
            .unwrap();

        // Write a sell position update (close to flat)
        let sell_update = PositionUpdateRecord::builder("strat_a", "BTCUSDT")
            .ts_ns(1706400001000000000)
            .session_id("test_session")
            .seq(2)
            .correlation_id("corr_2")
            .fill_seq(2)
            .position_qty(0, -8) // Flat
            .avg_price_flat(-2)
            .cash_delta(4300000, -2) // Received $43,000
            .realized_pnl_delta(100000, -2) // $1,000 profit
            .fee(430, -2) // $4.30 fee
            .venue("sim")
            .build();
        writer
            .write_position_update(sell_update.clone())
            .await
            .unwrap();

        writer.finalize().await.unwrap();

        // Read back
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_position_updates().unwrap();

        assert_eq!(records.len(), 2);

        // Verify buy record
        assert_eq!(records[0].strategy_id, "strat_a");
        assert_eq!(records[0].symbol, "BTCUSDT");
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[0].fill_seq, 1);
        assert_eq!(records[0].position_qty_mantissa, 100000000);
        assert_eq!(records[0].avg_price_mantissa, Some(4200000));
        assert!(records[0].cash_delta_mantissa < 0); // Negative for buy
        assert_eq!(records[0].digest, buy_update.digest);

        // Verify sell record
        assert_eq!(records[1].seq, 2);
        assert_eq!(records[1].position_qty_mantissa, 0);
        assert_eq!(records[1].avg_price_mantissa, None); // Flat
        assert!(records[1].cash_delta_mantissa > 0); // Positive for sell
        assert!(records[1].realized_pnl_delta_mantissa > 0);
        assert_eq!(records[1].digest, sell_update.digest);
    }

    #[tokio::test]
    async fn test_position_update_wal_append_only() {
        use quantlaxmi_models::PositionUpdateRecord;

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        // First writer
        {
            let mut writer = WalWriter::new(&session_dir).await.unwrap();
            let update1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
                .session_id("sess")
                .seq(1)
                .fill_seq(1)
                .position_qty(100, -8)
                .avg_price(5000, -2)
                .cash_delta(-5000, -2)
                .realized_pnl_delta(0, -2)
                .venue("sim")
                .build();
            writer.write_position_update(update1).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // Second writer (append)
        {
            let mut writer = WalWriter::new(&session_dir).await.unwrap();
            let update2 = PositionUpdateRecord::builder("strat", "BTCUSDT")
                .session_id("sess")
                .seq(2)
                .fill_seq(2)
                .position_qty(200, -8)
                .avg_price(5000, -2)
                .cash_delta(-5000, -2)
                .realized_pnl_delta(0, -2)
                .venue("sim")
                .build();
            writer.write_position_update(update2).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // Read back - should have 2 records
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_position_updates().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    #[tokio::test]
    async fn test_position_update_monotonic_seq_enforced() {
        use quantlaxmi_models::PositionUpdateRecord;

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Write seq=1
        let update1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update1).await.unwrap();

        // Try duplicate seq=1 (should fail)
        let update_dup = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1) // Duplicate!
            .fill_seq(2)
            .position_qty(200, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        let result = writer.write_position_update(update_dup).await;
        assert!(result.is_err(), "Duplicate seq should fail");
        assert!(
            result.unwrap_err().to_string().contains("monotonicity"),
            "Error should mention monotonicity"
        );

        // Try backwards seq (should fail)
        let update_backwards = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(0) // Backwards!
            .fill_seq(3)
            .position_qty(300, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        let result = writer.write_position_update(update_backwards).await;
        assert!(result.is_err(), "Backwards seq should fail");

        // seq=2 should succeed
        let update2 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(2)
            .fill_seq(4)
            .position_qty(400, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update2).await.unwrap();

        // Verify only 2 records written (seq=1 and seq=2)
        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_position_updates().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    #[tokio::test]
    async fn test_position_update_different_sessions() {
        use quantlaxmi_models::PositionUpdateRecord;

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        // Different sessions can have same seq
        let update_a1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("session_a")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update_a1).await.unwrap();

        let update_b1 = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("session_b")
            .seq(1) // Same seq, different session - OK
            .fill_seq(1)
            .position_qty(200, -8)
            .avg_price(5001, -2)
            .cash_delta(-5001, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update_b1).await.unwrap();

        let reader = WalReader::open(&session_dir).unwrap();
        let records = reader.read_position_updates().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_position_update_manifest_includes_hash() {
        use quantlaxmi_models::PositionUpdateRecord;

        let temp_dir = tempfile::tempdir().unwrap();
        let session_dir = temp_dir.path().join("sess1");

        let mut writer = WalWriter::new(&session_dir).await.unwrap();

        let update = PositionUpdateRecord::builder("strat", "BTCUSDT")
            .session_id("sess")
            .seq(1)
            .fill_seq(1)
            .position_qty(100, -8)
            .avg_price(5000, -2)
            .cash_delta(-5000, -2)
            .realized_pnl_delta(0, -2)
            .venue("sim")
            .build();
        writer.write_position_update(update).await.unwrap();

        let manifest = writer.finalize().await.unwrap();

        // Verify manifest has file info
        let file_info = manifest.files.get("position_updates.jsonl").unwrap();
        assert_eq!(file_info.path, "wal/position_updates.jsonl");
        assert_eq!(file_info.record_count, 1);
        assert!(
            !file_info.sha256.is_empty(),
            "SHA256 hash should be computed"
        );
        assert!(file_info.bytes_len > 0, "File should have content");

        // Verify integrity
        let reader = WalReader::open(&session_dir).unwrap();
        let report = reader.verify_integrity(&manifest).unwrap();
        assert!(report.passed, "Integrity check should pass");
        assert!(
            report
                .verified_files
                .contains(&"position_updates.jsonl".to_string()),
            "position_updates.jsonl should be verified"
        );
    }
}
