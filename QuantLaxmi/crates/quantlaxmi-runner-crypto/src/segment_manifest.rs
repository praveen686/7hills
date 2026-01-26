//! Segment Manifest Management
//!
//! Provides segment-aware capture with:
//! - Manifest written at segment start (with "running" status)
//! - Heartbeat updates every 60 seconds
//! - Finalization on graceful shutdown or signal
//! - Automatic inventory maintenance for session families
//!
//! ## Segment Lifecycle
//! ```text
//! 1. Start capture → write segment_manifest.json (status: "running")
//! 2. Every 60s → update heartbeat_ts in manifest
//! 3. On signal/completion → finalize manifest with end_ts and stop_reason
//! 4. Auto-append to family inventory
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

/// Reason the segment stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StopReason {
    /// Segment is still running
    Running,
    /// Normal completion (duration elapsed)
    NormalCompletion,
    /// User interrupt (Ctrl+C / SIGINT)
    UserInterrupt,
    /// External kill (SIGTERM)
    ExternalKillSigterm,
    /// Terminal disconnect (SIGHUP)
    ExternalKillSighup,
    /// Panic or crash
    Panic,
    /// Network error
    NetworkError,
    /// Unknown (for recovery from hard stops)
    Unknown,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::Running => write!(f, "RUNNING"),
            StopReason::NormalCompletion => write!(f, "NORMAL_COMPLETION"),
            StopReason::UserInterrupt => write!(f, "USER_INTERRUPT"),
            StopReason::ExternalKillSigterm => write!(f, "EXTERNAL_KILL_SIGTERM"),
            StopReason::ExternalKillSighup => write!(f, "EXTERNAL_KILL_SIGHUP"),
            StopReason::Panic => write!(f, "PANIC"),
            StopReason::NetworkError => write!(f, "NETWORK_ERROR"),
            StopReason::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Event counts per stream type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCounts {
    pub spot_quotes: usize,
    pub perp_quotes: usize,
    pub funding: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub depth: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl EventCounts {
    pub fn total(&self) -> usize {
        self.spot_quotes + self.perp_quotes + self.funding + self.depth
    }
}

/// Gap detected between segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapInfo {
    pub previous_segment_id: String,
    pub gap_seconds: f64,
    pub reason: String,
}

/// Segment lifecycle state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SegmentState {
    /// Manifest written at start, capture in progress
    #[default]
    Bootstrap,
    /// Graceful shutdown with digests computed
    Finalized,
    /// Retroactively finalized from crashed segment
    FinalizedRetro,
}

/// Per-stream digest for integrity verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDigest {
    pub file_path: String,
    pub sha256: String,
    pub event_count: usize,
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_event_ts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_ts: Option<String>,
}

/// All stream digests for a segment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SegmentDigests {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<StreamDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perp: Option<StreamDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub funding: Option<StreamDigest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<StreamDigest>,
}

/// Capture configuration snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub include_spot: bool,
    pub include_depth: bool,
    pub price_exponent: i32,
    pub qty_exponent: i32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            include_spot: true,
            include_depth: false,
            price_exponent: -2,
            qty_exponent: -8,
        }
    }
}

/// Current segment manifest schema version.
/// Bump this when manifest structure changes.
pub const SEGMENT_MANIFEST_SCHEMA_VERSION: u32 = 3;

/// Segment manifest - written to segment_manifest.json in each segment directory.
///
/// ## Lifecycle
/// 1. BOOTSTRAP: Written immediately at segment start with schema assertions
/// 2. FINALIZED: Updated on graceful shutdown with digests and counts
/// 3. FINALIZED_RETRO: Created by retro-finalize for crashed segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentManifest {
    /// Schema version for this manifest format (must match SEGMENT_MANIFEST_SCHEMA_VERSION)
    pub schema_version: u32,
    /// Quote schema identifier (must be "canonical_v1" for valid captures)
    pub quote_schema: String,
    /// Segment lifecycle state
    pub state: SegmentState,
    /// Family ID (e.g., "perp_BTCUSDT_20260125")
    pub session_family_id: String,
    /// Segment ID (typically the folder name, e.g., "perp_20260125_051437")
    pub segment_id: String,
    /// Symbol(s) being captured
    pub symbols: Vec<String>,
    /// Capture mode (e.g., "capture-perp-session")
    pub capture_mode: String,
    /// Start timestamp
    pub start_ts: DateTime<Utc>,
    /// End timestamp (None if still running)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ts: Option<DateTime<Utc>>,
    /// Last heartbeat timestamp (updated every 60s while running)
    pub heartbeat_ts: DateTime<Utc>,
    /// Stop reason
    pub stop_reason: StopReason,
    /// Event counts
    pub events: EventCounts,
    /// Binary hash (SHA256 of the capture binary)
    pub binary_hash: String,
    /// Capture configuration snapshot
    pub config: CaptureConfig,
    /// Per-stream digests (populated on finalization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digests: Option<SegmentDigests>,
    /// Gap info if this segment follows a prior segment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap_from_prior: Option<GapInfo>,
    /// Duration in seconds (None if still running)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
}

impl SegmentManifest {
    /// Create a new bootstrap manifest at segment start.
    ///
    /// This is written immediately when capture begins, ensuring every segment
    /// has a manifest with schema assertions even if killed ungracefully.
    pub fn new(
        session_family_id: String,
        segment_id: String,
        symbols: Vec<String>,
        capture_mode: String,
        binary_hash: String,
        config: CaptureConfig,
    ) -> Self {
        let now = Utc::now();
        Self {
            schema_version: SEGMENT_MANIFEST_SCHEMA_VERSION,
            quote_schema: "canonical_v1".to_string(),
            state: SegmentState::Bootstrap,
            session_family_id,
            segment_id,
            symbols,
            capture_mode,
            start_ts: now,
            end_ts: None,
            heartbeat_ts: now,
            stop_reason: StopReason::Running,
            events: EventCounts::default(),
            binary_hash,
            config,
            digests: None,
            gap_from_prior: None,
            duration_secs: None,
        }
    }

    /// Update heartbeat timestamp.
    pub fn heartbeat(&mut self) {
        self.heartbeat_ts = Utc::now();
    }

    /// Finalize the manifest with stop reason, counts, and digests.
    ///
    /// Called on graceful shutdown. Sets state to FINALIZED.
    pub fn finalize(
        &mut self,
        stop_reason: StopReason,
        events: EventCounts,
        digests: Option<SegmentDigests>,
    ) {
        let now = Utc::now();
        self.end_ts = Some(now);
        self.stop_reason = stop_reason;
        self.events = events;
        self.digests = digests;
        self.state = SegmentState::Finalized;
        self.duration_secs = Some((now - self.start_ts).num_milliseconds() as f64 / 1000.0);
    }

    /// Retroactively finalize a crashed segment.
    ///
    /// Called by the retro-finalize command. Sets state to FINALIZED_RETRO.
    /// Infers start/end timestamps and duration from actual data in digests.
    pub fn retro_finalize(&mut self, events: EventCounts, digests: SegmentDigests) {
        // Infer actual data time range from digests
        let mut first_ts: Option<DateTime<Utc>> = None;
        let mut last_ts: Option<DateTime<Utc>> = None;

        // Check all digest sources for earliest/latest timestamps
        for d in [
            &digests.spot,
            &digests.perp,
            &digests.funding,
            &digests.depth,
        ]
        .into_iter()
        .flatten()
        {
            if let (Some(first_str), Some(last_str)) = (&d.first_event_ts, &d.last_event_ts) {
                // Parse timestamp strings
                if let (Ok(first), Ok(last)) = (
                    DateTime::parse_from_rfc3339(first_str).map(|t| t.with_timezone(&Utc)),
                    DateTime::parse_from_rfc3339(last_str).map(|t| t.with_timezone(&Utc)),
                ) {
                    first_ts = Some(first_ts.map_or(first, |t| t.min(first)));
                    last_ts = Some(last_ts.map_or(last, |t| t.max(last)));
                }
            }
        }

        // Update timestamps from actual data range
        if let Some(first) = first_ts {
            self.start_ts = first;
        }
        if let Some(last) = last_ts {
            self.end_ts = Some(last);
        } else if self.end_ts.is_none() {
            self.end_ts = Some(self.heartbeat_ts);
        }

        if self.stop_reason == StopReason::Running {
            self.stop_reason = StopReason::Unknown;
        }
        self.events = events;
        self.digests = Some(digests);
        self.state = SegmentState::FinalizedRetro;

        // Compute duration from actual data time range
        if let Some(end) = self.end_ts {
            self.duration_secs = Some((end - self.start_ts).num_milliseconds() as f64 / 1000.0);
        }
    }

    /// Check if this manifest is finalized (either gracefully or retro).
    pub fn is_finalized(&self) -> bool {
        matches!(
            self.state,
            SegmentState::Finalized | SegmentState::FinalizedRetro
        )
    }

    /// Write manifest to disk atomically (write temp + rename).
    ///
    /// This ensures crash safety - the manifest is either fully written or not at all.
    pub fn write(&self, segment_dir: &Path) -> Result<()> {
        let manifest_path = segment_dir.join("segment_manifest.json");
        let temp_path = segment_dir.join(".segment_manifest.json.tmp");
        let json = serde_json::to_string_pretty(self)?;

        // Write to temp file
        std::fs::write(&temp_path, &json)
            .with_context(|| format!("write temp manifest: {:?}", temp_path))?;

        // Atomic rename
        std::fs::rename(&temp_path, &manifest_path)
            .with_context(|| format!("rename manifest: {:?} -> {:?}", temp_path, manifest_path))?;

        Ok(())
    }

    /// Load manifest from disk.
    pub fn load(segment_dir: &Path) -> Result<Self> {
        let manifest_path = segment_dir.join("segment_manifest.json");
        let content = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read segment manifest: {:?}", manifest_path))?;
        serde_json::from_str(&content)
            .with_context(|| format!("parse segment manifest: {:?}", manifest_path))
    }
}

/// Inventory entry for a segment within a session family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventorySegment {
    pub segment_id: String,
    pub path: String,
    pub start_ts: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_ts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
    pub stop_reason: StopReason,
    pub events: EventCounts,
    pub usable: bool,
}

/// Gap analysis between segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryGap {
    pub from_segment: String,
    pub to_segment: String,
    pub from_ts: String,
    pub to_ts: String,
    pub duration_secs: f64,
}

/// Session family inventory - auto-maintained across segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInventory {
    /// Family ID (e.g., "perp_BTCUSDT_20260125")
    pub session_family: String,
    /// Primary symbol
    pub symbol: String,
    /// Capture mode
    pub mode: String,
    /// Binary hash (shared across segments, or "MIXED" if different)
    pub binary_hash: String,
    /// Schema version
    pub schema_version: String,
    /// Ordered list of segments
    pub segments: Vec<InventorySegment>,
    /// Gaps between segments
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<InventoryGap>,
    /// Notes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl SessionInventory {
    /// Create a new inventory for a session family.
    pub fn new(session_family: String, symbol: String, mode: String, binary_hash: String) -> Self {
        Self {
            session_family,
            symbol,
            mode,
            binary_hash,
            schema_version: "fixed_point_v1".to_string(),
            segments: Vec::new(),
            gaps: Vec::new(),
            notes: None,
        }
    }

    /// Add a segment to the inventory, computing gap from prior if applicable.
    pub fn add_segment(&mut self, manifest: &SegmentManifest) {
        // Check for gap from prior segment
        if let Some(last) = self.segments.last()
            && let Some(ref last_end) = last.end_ts
            && let Ok(last_end_dt) = DateTime::parse_from_rfc3339(last_end)
        {
            let start_ts = manifest.start_ts.to_rfc3339();
            let gap_secs = (manifest.start_ts - last_end_dt.with_timezone(&Utc)).num_milliseconds()
                as f64
                / 1000.0;
            if gap_secs > 1.0 {
                // Only record gaps > 1 second
                self.gaps.push(InventoryGap {
                    from_segment: last.segment_id.clone(),
                    to_segment: manifest.segment_id.clone(),
                    from_ts: last_end.clone(),
                    to_ts: start_ts,
                    duration_secs: gap_secs,
                });
            }
        }

        // Add segment entry
        let entry = InventorySegment {
            segment_id: manifest.segment_id.clone(),
            path: manifest.segment_id.clone(), // Relative path
            start_ts: manifest.start_ts.to_rfc3339(),
            end_ts: manifest.end_ts.map(|t| t.to_rfc3339()),
            duration_secs: manifest.duration_secs,
            stop_reason: manifest.stop_reason,
            events: manifest.events.clone(),
            usable: true,
        };
        self.segments.push(entry);

        // Check for binary hash consistency
        if self.binary_hash != manifest.binary_hash && self.binary_hash != "MIXED" {
            self.binary_hash = "MIXED".to_string();
        }
    }

    /// Update the last segment with finalized manifest data.
    pub fn update_last_segment(&mut self, manifest: &SegmentManifest) {
        if let Some(last) = self.segments.last_mut()
            && last.segment_id == manifest.segment_id
        {
            last.end_ts = manifest.end_ts.map(|t| t.to_rfc3339());
            last.duration_secs = manifest.duration_secs;
            last.stop_reason = manifest.stop_reason;
            last.events = manifest.events.clone();
        }
    }

    /// Write inventory to disk.
    pub fn write(&self, out_dir: &Path) -> Result<()> {
        // Extract date from session_family (e.g., "perp_BTCUSDT_20260125" -> "20260125")
        let date_part = self
            .session_family
            .split('_')
            .next_back()
            .unwrap_or("unknown");
        let filename = format!("perp_{}_inventory.json", date_part);
        let inventory_path = out_dir.join(filename);

        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&inventory_path, json)
            .with_context(|| format!("write inventory: {:?}", inventory_path))?;
        tracing::info!("Inventory updated: {:?}", inventory_path);
        Ok(())
    }

    /// Load inventory from disk, or create new if not found.
    pub fn load_or_create(
        out_dir: &Path,
        session_family: &str,
        symbol: &str,
        mode: &str,
        binary_hash: &str,
    ) -> Self {
        // Extract date from session_family
        let date_part = session_family.split('_').next_back().unwrap_or("unknown");
        let filename = format!("perp_{}_inventory.json", date_part);
        let inventory_path = out_dir.join(filename);

        if inventory_path.exists() {
            match std::fs::read_to_string(&inventory_path) {
                Ok(content) => match serde_json::from_str(&content) {
                    Ok(inv) => return inv,
                    Err(e) => {
                        tracing::warn!("Failed to parse inventory {:?}: {}", inventory_path, e);
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read inventory {:?}: {}", inventory_path, e);
                }
            }
        }

        Self::new(
            session_family.to_string(),
            symbol.to_string(),
            mode.to_string(),
            binary_hash.to_string(),
        )
    }
}

/// Compute SHA256 hash of the current binary.
pub fn compute_binary_hash() -> Result<String> {
    let exe_path = std::env::current_exe().context("get current exe path")?;
    let bytes = std::fs::read(&exe_path).context("read binary")?;
    let hash = Sha256::digest(&bytes);
    Ok(hex::encode(hash))
}

/// Compute digest for a stream file (JSONL format).
///
/// Returns None if the file doesn't exist or is empty.
pub fn compute_stream_digest(file_path: &Path) -> Result<Option<StreamDigest>> {
    use std::io::{BufRead, BufReader};

    if !file_path.exists() {
        return Ok(None);
    }

    let metadata = std::fs::metadata(file_path)?;
    let size_bytes = metadata.len();
    if size_bytes == 0 {
        return Ok(None);
    }

    let file = std::fs::File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut hasher = Sha256::new();
    let mut event_count = 0usize;
    let mut first_ts: Option<String> = None;
    let mut last_ts: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
        event_count += 1;

        // Try to extract timestamp from JSONL (look for "ts" field)
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line)
            && let Some(ts) = json.get("ts").and_then(|v| v.as_str())
        {
            if first_ts.is_none() {
                first_ts = Some(ts.to_string());
            }
            last_ts = Some(ts.to_string());
        }
    }

    if event_count == 0 {
        return Ok(None);
    }

    Ok(Some(StreamDigest {
        file_path: file_path.display().to_string(),
        sha256: hex::encode(hasher.finalize()),
        event_count,
        size_bytes,
        first_event_ts: first_ts,
        last_event_ts: last_ts,
    }))
}

/// Compute all stream digests for a segment directory.
///
/// Handles the actual directory structure where files are in symbol subdirectories:
/// ```text
/// segment_dir/
/// └── BTCUSDT/
///     ├── spot_quotes.jsonl
///     ├── perp_quotes.jsonl (or perp_depth.jsonl)
///     ├── funding.jsonl
///     └── perp_depth.jsonl (optional)
/// ```
pub fn compute_segment_digests(segment_dir: &Path) -> Result<SegmentDigests> {
    let mut spot_digest: Option<StreamDigest> = None;
    let mut perp_digest: Option<StreamDigest> = None;
    let mut funding_digest: Option<StreamDigest> = None;
    let mut depth_digest: Option<StreamDigest> = None;

    // Iterate through symbol subdirectories
    for entry in std::fs::read_dir(segment_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        // Skip non-symbol directories (like .tmp files)
        let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !dir_name.chars().all(|c| c.is_alphanumeric()) {
            continue;
        }

        // Check for spot quotes
        let spot_path = path.join("spot_quotes.jsonl");
        if spot_path.exists()
            && let Some(digest) = compute_stream_digest(&spot_path)?
        {
            spot_digest = Some(digest);
        }

        // Check for perp quotes (prefer perp_depth if exists)
        let perp_depth_path = path.join("perp_depth.jsonl");
        let perp_quotes_path = path.join("perp_quotes.jsonl");
        if perp_depth_path.exists()
            && let Some(digest) = compute_stream_digest(&perp_depth_path)?
        {
            perp_digest = Some(digest.clone());
            depth_digest = Some(digest);
        } else if perp_quotes_path.exists()
            && let Some(digest) = compute_stream_digest(&perp_quotes_path)?
        {
            perp_digest = Some(digest);
        }

        // Check for funding
        let funding_path = path.join("funding.jsonl");
        if funding_path.exists()
            && let Some(digest) = compute_stream_digest(&funding_path)?
        {
            funding_digest = Some(digest);
        }
    }

    Ok(SegmentDigests {
        spot: spot_digest,
        perp: perp_digest,
        funding: funding_digest,
        depth: depth_digest,
    })
}

/// Managed segment capture with automatic manifest handling.
pub struct ManagedSegment {
    pub manifest: Arc<Mutex<SegmentManifest>>,
    pub segment_dir: PathBuf,
    pub out_dir: PathBuf,
    shutdown_flag: Arc<AtomicBool>,
    heartbeat_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ManagedSegment {
    /// Start a new managed segment.
    pub async fn start(
        out_dir: &Path,
        symbols: &[String],
        capture_mode: &str,
        config: CaptureConfig,
    ) -> Result<Self> {
        let binary_hash = compute_binary_hash().unwrap_or_else(|_| "UNKNOWN".to_string());
        let start_time = Utc::now();

        // Generate segment ID and family ID
        let segment_tag = format!("perp_{}", start_time.format("%Y%m%d_%H%M%S"));
        let date_str = start_time.format("%Y%m%d").to_string();
        let primary_symbol = symbols.first().map(|s| s.as_str()).unwrap_or("UNKNOWN");
        let session_family_id = format!("perp_{}_{}", primary_symbol, date_str);

        // Create segment directory
        let segment_dir = out_dir.join(&segment_tag);
        std::fs::create_dir_all(&segment_dir)
            .with_context(|| format!("create segment dir: {:?}", segment_dir))?;

        // Create manifest
        let manifest = SegmentManifest::new(
            session_family_id.clone(),
            segment_tag.clone(),
            symbols.to_vec(),
            capture_mode.to_string(),
            binary_hash.clone(),
            config,
        );

        // Check for prior segment and compute gap
        let mut inventory = SessionInventory::load_or_create(
            out_dir,
            &session_family_id,
            primary_symbol,
            capture_mode,
            &binary_hash,
        );

        let mut manifest = manifest;
        if let Some(last) = inventory.segments.last()
            && let Some(ref last_end) = last.end_ts
            && let Ok(last_end_dt) = DateTime::parse_from_rfc3339(last_end)
        {
            let gap_secs =
                (start_time - last_end_dt.with_timezone(&Utc)).num_milliseconds() as f64 / 1000.0;
            manifest.gap_from_prior = Some(GapInfo {
                previous_segment_id: last.segment_id.clone(),
                gap_seconds: gap_secs,
                reason: "restart".to_string(),
            });
            tracing::info!(
                "Gap detected from prior segment {}: {:.1}s",
                last.segment_id,
                gap_secs
            );
        }

        // Write initial manifest
        manifest.write(&segment_dir)?;
        tracing::info!(
            "Segment manifest created: {:?}",
            segment_dir.join("segment_manifest.json")
        );

        // Add to inventory (will be updated on finalize)
        inventory.add_segment(&manifest);
        inventory.write(out_dir)?;

        let manifest = Arc::new(Mutex::new(manifest));
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        // Start heartbeat task
        let heartbeat_manifest = Arc::clone(&manifest);
        let heartbeat_dir = segment_dir.clone();
        let heartbeat_shutdown = Arc::clone(&shutdown_flag);
        let heartbeat_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                if heartbeat_shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let mut m = heartbeat_manifest.lock().await;
                m.heartbeat();
                if let Err(e) = m.write(&heartbeat_dir) {
                    tracing::warn!("Failed to write heartbeat: {}", e);
                }
            }
        });

        Ok(Self {
            manifest,
            segment_dir,
            out_dir: out_dir.to_path_buf(),
            shutdown_flag,
            heartbeat_handle: Some(heartbeat_handle),
        })
    }

    /// Get the segment directory path.
    pub fn segment_dir(&self) -> &Path {
        &self.segment_dir
    }

    /// Get the segment ID.
    pub async fn segment_id(&self) -> String {
        self.manifest.lock().await.segment_id.clone()
    }

    /// Finalize the segment with stop reason and event counts.
    pub async fn finalize(&mut self, stop_reason: StopReason, events: EventCounts) -> Result<()> {
        // Stop heartbeat
        self.shutdown_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.heartbeat_handle.take() {
            handle.abort();
        }

        // Compute stream digests
        let digests = compute_segment_digests(&self.segment_dir).ok();

        // Update manifest
        {
            let mut m = self.manifest.lock().await;
            m.finalize(stop_reason, events, digests);
            m.write(&self.segment_dir)?;
        }

        // Update inventory
        let manifest = self.manifest.lock().await;
        let mut inventory = SessionInventory::load_or_create(
            &self.out_dir,
            &manifest.session_family_id,
            manifest
                .symbols
                .first()
                .map(|s| s.as_str())
                .unwrap_or("UNKNOWN"),
            &manifest.capture_mode,
            &manifest.binary_hash,
        );
        inventory.update_last_segment(&manifest);
        inventory.write(&self.out_dir)?;

        tracing::info!(
            "Segment finalized: {} ({})",
            manifest.segment_id,
            stop_reason
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_reason_serialization() {
        let reason = StopReason::ExternalKillSighup;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, "\"EXTERNAL_KILL_SIGHUP\"");

        let parsed: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, reason);
    }

    #[test]
    fn test_event_counts_total() {
        let counts = EventCounts {
            spot_quotes: 100,
            perp_quotes: 200,
            funding: 10,
            depth: 0,
        };
        assert_eq!(counts.total(), 310);
    }

    #[test]
    fn test_manifest_lifecycle() {
        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260125".to_string(),
            "perp_20260125_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "capture-perp-session".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        assert_eq!(manifest.stop_reason, StopReason::Running);
        assert!(manifest.end_ts.is_none());
        assert_eq!(manifest.state, SegmentState::Bootstrap);
        assert_eq!(manifest.schema_version, SEGMENT_MANIFEST_SCHEMA_VERSION);
        assert_eq!(manifest.quote_schema, "canonical_v1");

        manifest.heartbeat();
        assert!(manifest.heartbeat_ts >= manifest.start_ts);

        let events = EventCounts {
            spot_quotes: 1000,
            perp_quotes: 2000,
            funding: 50,
            depth: 0,
        };
        manifest.finalize(StopReason::NormalCompletion, events, None);

        assert_eq!(manifest.stop_reason, StopReason::NormalCompletion);
        assert!(manifest.end_ts.is_some());
        assert!(manifest.duration_secs.is_some());
        assert_eq!(manifest.events.total(), 3050);
        assert_eq!(manifest.state, SegmentState::Finalized);
    }

    #[test]
    fn test_retro_finalize() {
        let mut manifest = SegmentManifest::new(
            "perp_BTCUSDT_20260125".to_string(),
            "perp_20260125_120000".to_string(),
            vec!["BTCUSDT".to_string()],
            "capture-perp-session".to_string(),
            "abc123".to_string(),
            CaptureConfig::default(),
        );

        assert_eq!(manifest.state, SegmentState::Bootstrap);
        assert_eq!(manifest.stop_reason, StopReason::Running);

        let events = EventCounts {
            spot_quotes: 500,
            perp_quotes: 1000,
            funding: 25,
            depth: 0,
        };
        let digests = SegmentDigests::default();
        manifest.retro_finalize(events, digests);

        assert_eq!(manifest.state, SegmentState::FinalizedRetro);
        assert_eq!(manifest.stop_reason, StopReason::Unknown);
        assert!(manifest.end_ts.is_some());
        assert!(manifest.digests.is_some());
    }
}
