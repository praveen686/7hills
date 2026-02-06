//! Replay Adapter for Crypto Segments
//!
//! Reads captured JSONL streams and emits a unified, time-ordered event stream.
//! Supports both backtest (offline) and paper trading (can be swapped with live feed).
//!
//! ## Usage
//! ```ignore
//! let adapter = SegmentReplayAdapter::open(segment_dir)?;
//! for event in adapter {
//!     match event.kind {
//!         EventKind::SpotQuote => { /* handle spot */ }
//!         EventKind::PerpQuote => { /* handle perp */ }
//!         EventKind::Funding => { /* handle funding */ }
//!     }
//! }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::Path;

/// Event kind discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    SpotQuote,
    PerpQuote,
    PerpDepth,
    Funding,
    Trade,
    Unknown,
}

/// Unified event envelope for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEvent {
    /// Event timestamp (used for ordering)
    pub ts: DateTime<Utc>,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Event kind
    pub kind: EventKind,
    /// Raw JSON payload (parsed lazily by consumer)
    pub payload: serde_json::Value,
}

impl PartialEq for ReplayEvent {
    fn eq(&self, other: &Self) -> bool {
        self.ts == other.ts
    }
}

impl Eq for ReplayEvent {}

impl PartialOrd for ReplayEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReplayEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (earliest first)
        other.ts.cmp(&self.ts)
    }
}

/// A stream source with its current head event.
struct StreamSource {
    kind: EventKind,
    symbol: String,
    lines: Lines<BufReader<File>>,
    current: Option<ReplayEvent>,
}

impl StreamSource {
    fn new(path: &Path, kind: EventKind, symbol: &str) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let file = File::open(path).with_context(|| format!("open {:?}", path))?;
        let reader = BufReader::new(file);
        let mut source = Self {
            kind,
            symbol: symbol.to_string(),
            lines: reader.lines(),
            current: None,
        };
        source.advance()?;
        Ok(Some(source))
    }

    fn advance(&mut self) -> Result<()> {
        loop {
            match self.lines.next() {
                Some(Ok(line)) if line.trim().is_empty() => continue,
                Some(Ok(line)) => {
                    let payload: serde_json::Value = serde_json::from_str(&line)
                        .with_context(|| format!("parse JSON: {}", &line[..line.len().min(100)]))?;

                    // Extract timestamp
                    let ts = if let Some(ts_str) = payload.get("ts").and_then(|v| v.as_str()) {
                        DateTime::parse_from_rfc3339(ts_str)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now())
                    } else {
                        Utc::now()
                    };

                    self.current = Some(ReplayEvent {
                        ts,
                        symbol: self.symbol.clone(),
                        kind: self.kind,
                        payload,
                    });
                    return Ok(());
                }
                Some(Err(e)) => return Err(e.into()),
                None => {
                    self.current = None;
                    return Ok(());
                }
            }
        }
    }

    fn peek(&self) -> Option<&ReplayEvent> {
        self.current.as_ref()
    }

    fn take(&mut self) -> Result<Option<ReplayEvent>> {
        let event = self.current.take();
        if event.is_some() {
            self.advance()?;
        }
        Ok(event)
    }
}

/// Wrapper for BinaryHeap that holds stream index and timestamp.
#[derive(Debug)]
struct HeapEntry {
    ts: DateTime<Utc>,
    stream_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.ts == other.ts
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap
        other.ts.cmp(&self.ts)
    }
}

/// Segment replay adapter - k-way merge of JSONL streams.
pub struct SegmentReplayAdapter {
    streams: Vec<StreamSource>,
    heap: BinaryHeap<HeapEntry>,
    events_emitted: usize,
}

impl SegmentReplayAdapter {
    /// Open a segment directory for replay.
    ///
    /// Discovers all symbol subdirectories and their JSONL files.
    pub fn open(segment_dir: &Path) -> Result<Self> {
        let mut streams = Vec::new();

        // Find symbol directories
        for entry in std::fs::read_dir(segment_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let symbol = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("UNKNOWN")
                .to_string();

            // Skip non-symbol directories (like metadata)
            if !symbol.chars().all(|c| c.is_alphanumeric()) {
                continue;
            }

            // Open spot quotes
            if let Some(source) = StreamSource::new(
                &path.join("spot_quotes.jsonl"),
                EventKind::SpotQuote,
                &symbol,
            )? {
                streams.push(source);
            }

            // Open perp quotes (prefer perp_depth.jsonl if exists, else perp_quotes.jsonl)
            let perp_depth_path = path.join("perp_depth.jsonl");
            let perp_quotes_path = path.join("perp_quotes.jsonl");
            if perp_depth_path.exists() {
                if let Some(source) =
                    StreamSource::new(&perp_depth_path, EventKind::PerpDepth, &symbol)?
                {
                    streams.push(source);
                }
            } else if let Some(source) =
                StreamSource::new(&perp_quotes_path, EventKind::PerpQuote, &symbol)?
            {
                streams.push(source);
            }

            // Open funding
            if let Some(source) =
                StreamSource::new(&path.join("funding.jsonl"), EventKind::Funding, &symbol)?
            {
                streams.push(source);
            }

            // Open agg_trades (needed for trade flow features: microprice_dev, signed_volume, imbalance)
            if let Some(source) =
                StreamSource::new(&path.join("agg_trades.jsonl"), EventKind::Trade, &symbol)?
            {
                streams.push(source);
            }
        }

        if streams.is_empty() {
            anyhow::bail!(
                "No JSONL streams found in segment directory: {:?}",
                segment_dir
            );
        }

        // Initialize heap with first event from each stream
        let mut heap = BinaryHeap::new();
        for (idx, stream) in streams.iter().enumerate() {
            if let Some(event) = stream.peek() {
                heap.push(HeapEntry {
                    ts: event.ts,
                    stream_idx: idx,
                });
            }
        }

        tracing::info!("Opened {} streams from {:?}", streams.len(), segment_dir);

        Ok(Self {
            streams,
            heap,
            events_emitted: 0,
        })
    }

    /// Get the next event in timestamp order.
    pub fn next_event(&mut self) -> Result<Option<ReplayEvent>> {
        let entry = match self.heap.pop() {
            Some(e) => e,
            None => return Ok(None),
        };

        let stream = &mut self.streams[entry.stream_idx];
        let event = stream.take()?.expect("heap entry implies event exists");

        // Re-add stream to heap if it has more events
        if let Some(next_event) = stream.peek() {
            self.heap.push(HeapEntry {
                ts: next_event.ts,
                stream_idx: entry.stream_idx,
            });
        }

        self.events_emitted += 1;
        Ok(Some(event))
    }

    /// Count of events emitted so far.
    pub fn events_emitted(&self) -> usize {
        self.events_emitted
    }

    /// Check if there are more events.
    pub fn has_more(&self) -> bool {
        !self.heap.is_empty()
    }
}

impl Iterator for SegmentReplayAdapter {
    type Item = Result<ReplayEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_event() {
            Ok(Some(event)) => Some(Ok(event)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Summary statistics from a replay.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayStats {
    pub total_events: usize,
    pub spot_events: usize,
    pub perp_events: usize,
    pub funding_events: usize,
    pub first_ts: Option<DateTime<Utc>>,
    pub last_ts: Option<DateTime<Utc>>,
    pub duration_secs: f64,
    pub symbols: Vec<String>,
}

impl ReplayStats {
    pub fn from_adapter(adapter: &mut SegmentReplayAdapter) -> Result<Self> {
        let mut stats = ReplayStats::default();
        let mut symbols = std::collections::HashSet::new();

        while let Some(event) = adapter.next_event()? {
            stats.total_events += 1;

            match event.kind {
                EventKind::SpotQuote => stats.spot_events += 1,
                EventKind::PerpQuote | EventKind::PerpDepth => stats.perp_events += 1,
                EventKind::Funding => stats.funding_events += 1,
                EventKind::Trade | EventKind::Unknown => {} // Not tracked in stats
            }

            if stats.first_ts.is_none() {
                stats.first_ts = Some(event.ts);
            }
            stats.last_ts = Some(event.ts);

            symbols.insert(event.symbol);
        }

        if let (Some(first), Some(last)) = (stats.first_ts, stats.last_ts) {
            stats.duration_secs = (last - first).num_milliseconds() as f64 / 1000.0;
        }

        stats.symbols = symbols.into_iter().collect();
        stats.symbols.sort();

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_segment() -> TempDir {
        let dir = TempDir::new().unwrap();
        let sym_dir = dir.path().join("BTCUSDT");
        std::fs::create_dir_all(&sym_dir).unwrap();

        // Create spot quotes
        let mut spot = File::create(sym_dir.join("spot_quotes.jsonl")).unwrap();
        writeln!(
            spot,
            r#"{{"ts":"2026-01-25T10:00:00Z","bid":100000,"ask":100001}}"#
        )
        .unwrap();
        writeln!(
            spot,
            r#"{{"ts":"2026-01-25T10:00:02Z","bid":100002,"ask":100003}}"#
        )
        .unwrap();

        // Create perp quotes
        let mut perp = File::create(sym_dir.join("perp_quotes.jsonl")).unwrap();
        writeln!(
            perp,
            r#"{{"ts":"2026-01-25T10:00:01Z","bid":100050,"ask":100051}}"#
        )
        .unwrap();
        writeln!(
            perp,
            r#"{{"ts":"2026-01-25T10:00:03Z","bid":100052,"ask":100053}}"#
        )
        .unwrap();

        // Create funding
        let mut funding = File::create(sym_dir.join("funding.jsonl")).unwrap();
        writeln!(
            funding,
            r#"{{"ts":"2026-01-25T10:00:01.500Z","rate":0.0001}}"#
        )
        .unwrap();

        dir
    }

    #[test]
    fn test_replay_ordering() {
        let dir = create_test_segment();
        let mut adapter = SegmentReplayAdapter::open(dir.path()).unwrap();

        let events: Vec<_> = adapter.by_ref().map(|r| r.unwrap()).collect();

        assert_eq!(events.len(), 5);

        // Verify timestamp ordering
        for window in events.windows(2) {
            assert!(window[0].ts <= window[1].ts);
        }

        // Verify event kinds
        assert_eq!(events[0].kind, EventKind::SpotQuote); // 10:00:00
        assert_eq!(events[1].kind, EventKind::PerpQuote); // 10:00:01
        assert_eq!(events[2].kind, EventKind::Funding); // 10:00:01.500
        assert_eq!(events[3].kind, EventKind::SpotQuote); // 10:00:02
        assert_eq!(events[4].kind, EventKind::PerpQuote); // 10:00:03
    }

    #[test]
    fn test_replay_stats() {
        let dir = create_test_segment();
        let mut adapter = SegmentReplayAdapter::open(dir.path()).unwrap();
        let stats = ReplayStats::from_adapter(&mut adapter).unwrap();

        assert_eq!(stats.total_events, 5);
        assert_eq!(stats.spot_events, 2);
        assert_eq!(stats.perp_events, 2);
        assert_eq!(stats.funding_events, 1);
        assert_eq!(stats.symbols, vec!["BTCUSDT"]);
    }
}
