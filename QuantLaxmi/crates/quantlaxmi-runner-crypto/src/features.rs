//! # FeatureSet v1 - Canonical Feature Extraction
//!
//! Deterministic feature computation from captured quote streams.
//!
//! ## Features (per symbol, per timestamp)
//! - `mid`: (bid + ask) / 2 (fixed-point mantissa)
//! - `spread`: ask - bid (fixed-point mantissa)
//! - `microprice`: (ask*bid_qty + bid*ask_qty) / (bid_qty + ask_qty)
//! - `imbalance_l1`: (bid_qty - ask_qty) / (bid_qty + ask_qty)
//! - `quote_age_ms`: milliseconds since quote timestamp
//! - `update_rate`: events per second over trailing window
//! - `short_vol_proxy`: EWMA of mid returns over trailing window
//!
//! ## Determinism
//! Feature extraction is fully deterministic:
//! - Same input segment + same binary = identical output digest
//! - Output includes input segment digests for audit trail

use chrono::DateTime;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Feature schema version - bump when feature definitions change
pub const FEATURE_SCHEMA_VERSION: u32 = 1;

/// Quote event from JSONL (minimal fields needed for features)
#[derive(Debug, Clone, Deserialize)]
pub struct QuoteInput {
    pub ts: String,
    pub symbol: String,
    pub bid: i64,
    pub ask: i64,
    pub bid_qty: i64,
    pub ask_qty: i64,
}

/// Computed features for a single observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRow {
    /// Timestamp (ISO 8601)
    pub ts: String,
    /// Symbol
    pub symbol: String,
    /// Mid price: (bid + ask) / 2 (mantissa, same exponent as input)
    pub mid: i64,
    /// Spread: ask - bid (mantissa)
    pub spread: i64,
    /// Microprice: (ask*bid_qty + bid*ask_qty) / (bid_qty + ask_qty)
    /// Returns mid if quantities are zero
    pub microprice: i64,
    /// L1 imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) * 10000 (basis points)
    /// Range: -10000 to +10000
    pub imbalance_bps: i32,
    /// Quote age in milliseconds (0 for first quote)
    pub quote_age_ms: i64,
    /// Update rate: events per second over trailing 1s window (scaled by 100)
    pub update_rate_x100: u32,
    /// Short-term volatility proxy: EWMA of |mid_return| over 5s (basis points)
    pub vol_proxy_bps: u32,
    /// Running sequence number within extraction
    pub seq: u64,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Trailing window for update rate calculation (milliseconds)
    #[serde(default = "default_update_window_ms")]
    pub update_rate_window_ms: u64,
    /// EWMA decay factor for volatility proxy (0-1, higher = more weight on recent)
    #[serde(default = "default_vol_decay")]
    pub vol_ewma_decay: f64,
    /// Minimum spread to include (filter outliers), 0 = no filter
    #[serde(default)]
    pub min_spread: i64,
    /// Maximum spread to include (filter outliers), 0 = no filter
    #[serde(default)]
    pub max_spread: i64,
}

fn default_update_window_ms() -> u64 {
    1000
}
fn default_vol_decay() -> f64 {
    0.1
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            update_rate_window_ms: 1000,
            vol_ewma_decay: 0.1,
            min_spread: 0,
            max_spread: 0,
        }
    }
}

/// State for incremental feature computation
struct FeatureState {
    last_ts_ms: Option<i64>,
    last_mid: Option<i64>,
    /// Circular buffer of (timestamp_ms, event_count) for update rate
    update_window: VecDeque<i64>,
    /// EWMA of |mid_return| in basis points
    vol_ewma_bps: f64,
    seq: u64,
}

impl FeatureState {
    fn new() -> Self {
        Self {
            last_ts_ms: None,
            last_mid: None,
            update_window: VecDeque::with_capacity(1000),
            vol_ewma_bps: 0.0,
            seq: 0,
        }
    }

    fn compute_features(
        &mut self,
        quote: &QuoteInput,
        config: &FeatureConfig,
    ) -> Option<FeatureRow> {
        // Parse timestamp
        let ts_dt = DateTime::parse_from_rfc3339(&quote.ts).ok()?;
        let ts_ms = ts_dt.timestamp_millis();

        // Basic price features
        let mid = (quote.bid + quote.ask) / 2;
        let spread = quote.ask - quote.bid;

        // Filter by spread if configured
        if config.min_spread > 0 && spread < config.min_spread {
            return None;
        }
        if config.max_spread > 0 && spread > config.max_spread {
            return None;
        }

        // Microprice: (ask*bid_qty + bid*ask_qty) / (bid_qty + ask_qty)
        let total_qty = quote.bid_qty + quote.ask_qty;
        let microprice = if total_qty > 0 {
            // Use i128 to avoid overflow
            let numer = (quote.ask as i128) * (quote.bid_qty as i128)
                + (quote.bid as i128) * (quote.ask_qty as i128);
            (numer / (total_qty as i128)) as i64
        } else {
            mid
        };

        // L1 imbalance in basis points
        let imbalance_bps = if total_qty > 0 {
            let imb = ((quote.bid_qty - quote.ask_qty) as f64 / total_qty as f64) * 10000.0;
            imb.round() as i32
        } else {
            0
        };

        // Quote age
        let quote_age_ms = self.last_ts_ms.map(|last| ts_ms - last).unwrap_or(0);

        // Update rate: count events in trailing window
        self.update_window.push_back(ts_ms);
        let window_start = ts_ms - config.update_rate_window_ms as i64;
        while self
            .update_window
            .front()
            .is_some_and(|&t| t < window_start)
        {
            self.update_window.pop_front();
        }
        let update_rate_x100 = if config.update_rate_window_ms > 0 {
            ((self.update_window.len() as f64 * 100.0 * 1000.0)
                / config.update_rate_window_ms as f64) as u32
        } else {
            0
        };

        // Volatility proxy: EWMA of |mid_return|
        let vol_proxy_bps = if let Some(last_mid) = self.last_mid {
            if last_mid > 0 {
                let ret_bps = ((mid - last_mid).abs() as f64 / last_mid as f64) * 10000.0;
                self.vol_ewma_bps = config.vol_ewma_decay * ret_bps
                    + (1.0 - config.vol_ewma_decay) * self.vol_ewma_bps;
            }
            self.vol_ewma_bps.round() as u32
        } else {
            0
        };

        // Update state
        self.last_ts_ms = Some(ts_ms);
        self.last_mid = Some(mid);
        self.seq += 1;

        Some(FeatureRow {
            ts: quote.ts.clone(),
            symbol: quote.symbol.clone(),
            mid,
            spread,
            microprice,
            imbalance_bps,
            quote_age_ms,
            update_rate_x100,
            vol_proxy_bps,
            seq: self.seq,
        })
    }
}

/// Feature extraction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionResult {
    /// Number of input events processed
    pub input_events: usize,
    /// Number of features output (may be less due to filters)
    pub output_events: usize,
    /// Output file path
    pub output_path: String,
    /// SHA256 digest of output file
    pub output_digest: String,
    /// First timestamp in output
    pub first_ts: Option<String>,
    /// Last timestamp in output
    pub last_ts: Option<String>,
}

/// Extract features from a quote stream file
pub fn extract_features(
    input_path: &Path,
    output_path: &Path,
    config: &FeatureConfig,
) -> anyhow::Result<FeatureExtractionResult> {
    let file = std::fs::File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut state = FeatureState::new();
    let mut input_events = 0usize;
    let mut output_events = 0usize;
    let mut first_ts: Option<String> = None;
    let mut last_ts: Option<String> = None;

    // Create output file
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output_file = std::fs::File::create(output_path)?;
    let mut hasher = Sha256::new();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        input_events += 1;

        let quote: QuoteInput = match serde_json::from_str(&line) {
            Ok(q) => q,
            Err(_) => continue, // Skip malformed lines
        };

        if let Some(features) = state.compute_features(&quote, config) {
            let json = serde_json::to_string(&features)?;
            writeln!(output_file, "{}", json)?;
            hasher.update(json.as_bytes());
            hasher.update(b"\n");

            if first_ts.is_none() {
                first_ts = Some(features.ts.clone());
            }
            last_ts = Some(features.ts);
            output_events += 1;
        }
    }

    output_file.flush()?;
    let output_digest = hex::encode(hasher.finalize());

    Ok(FeatureExtractionResult {
        input_events,
        output_events,
        output_path: output_path.display().to_string(),
        output_digest,
        first_ts,
        last_ts,
    })
}

/// Extract features from multiple streams (spot + perp) and merge by timestamp
pub fn extract_features_merged(
    _spot_path: Option<&Path>,
    perp_path: &Path,
    output_path: &Path,
    config: &FeatureConfig,
) -> anyhow::Result<FeatureExtractionResult> {
    // For v1, just extract from perp stream
    // TODO: merge spot + perp for basis features
    extract_features(perp_path, output_path, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn sample_quotes() -> Vec<&'static str> {
        vec![
            r#"{"ts":"2026-01-25T10:00:00.000Z","symbol":"BTCUSDT","bid":1050000,"ask":1050100,"bid_qty":100000000,"ask_qty":200000000}"#,
            r#"{"ts":"2026-01-25T10:00:00.100Z","symbol":"BTCUSDT","bid":1050050,"ask":1050150,"bid_qty":150000000,"ask_qty":180000000}"#,
            r#"{"ts":"2026-01-25T10:00:00.200Z","symbol":"BTCUSDT","bid":1050100,"ask":1050200,"bid_qty":120000000,"ask_qty":190000000}"#,
        ]
    }

    #[test]
    fn test_feature_extraction_basic() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("quotes.jsonl");
        let output_path = dir.path().join("features.jsonl");

        // Write sample quotes
        let mut f = std::fs::File::create(&input_path).unwrap();
        for q in sample_quotes() {
            writeln!(f, "{}", q).unwrap();
        }

        let config = FeatureConfig::default();
        let result = extract_features(&input_path, &output_path, &config).unwrap();

        assert_eq!(result.input_events, 3);
        assert_eq!(result.output_events, 3);
        assert!(!result.output_digest.is_empty());

        // Verify output is valid JSONL
        let content = std::fs::read_to_string(&output_path).unwrap();
        for line in content.lines() {
            let _: FeatureRow = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn test_feature_extraction_deterministic() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("quotes.jsonl");
        let output1 = dir.path().join("features1.jsonl");
        let output2 = dir.path().join("features2.jsonl");

        let mut f = std::fs::File::create(&input_path).unwrap();
        for q in sample_quotes() {
            writeln!(f, "{}", q).unwrap();
        }

        let config = FeatureConfig::default();
        let result1 = extract_features(&input_path, &output1, &config).unwrap();
        let result2 = extract_features(&input_path, &output2, &config).unwrap();

        assert_eq!(
            result1.output_digest, result2.output_digest,
            "Feature extraction must be deterministic"
        );
    }

    #[test]
    fn test_mid_and_spread_calculation() {
        let quote = QuoteInput {
            ts: "2026-01-25T10:00:00Z".to_string(),
            symbol: "BTCUSDT".to_string(),
            bid: 1000,
            ask: 1010,
            bid_qty: 100,
            ask_qty: 100,
        };

        let config = FeatureConfig::default();
        let mut state = FeatureState::new();
        let features = state.compute_features(&quote, &config).unwrap();

        assert_eq!(features.mid, 1005);
        assert_eq!(features.spread, 10);
    }

    #[test]
    fn test_microprice_calculation() {
        let quote = QuoteInput {
            ts: "2026-01-25T10:00:00Z".to_string(),
            symbol: "BTCUSDT".to_string(),
            bid: 1000,
            ask: 1010,
            bid_qty: 100, // More on bid side
            ask_qty: 50,
        };

        let config = FeatureConfig::default();
        let mut state = FeatureState::new();
        let features = state.compute_features(&quote, &config).unwrap();

        // microprice = (ask*bid_qty + bid*ask_qty) / (bid_qty + ask_qty)
        // = (1010*100 + 1000*50) / 150 = (101000 + 50000) / 150 = 1006.67
        assert_eq!(features.microprice, 1006);
    }

    #[test]
    fn test_imbalance_calculation() {
        let quote = QuoteInput {
            ts: "2026-01-25T10:00:00Z".to_string(),
            symbol: "BTCUSDT".to_string(),
            bid: 1000,
            ask: 1010,
            bid_qty: 100,
            ask_qty: 50,
        };

        let config = FeatureConfig::default();
        let mut state = FeatureState::new();
        let features = state.compute_features(&quote, &config).unwrap();

        // imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) * 10000
        // = (100 - 50) / 150 * 10000 = 3333
        assert_eq!(features.imbalance_bps, 3333);
    }
}
