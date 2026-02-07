//! Regime Warmup Module
//!
//! Handles warmup of the Grassmann regime detector using historical data.
//! This ensures the regime is known before any trades are taken.

use chrono::{DateTime, Utc};
use quantlaxmi_connectors_zerodha::HistoricalCandle;
use quantlaxmi_regime::FeatureVector;
use std::collections::HashMap;

/// Warmup state for the regime detector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarmupState {
    /// Fetching historical data
    #[default]
    Fetching,
    /// Processing historical data through detector
    Processing,
    /// Warmup complete, ready to trade
    Ready,
    /// Warmup failed (will trade cautiously)
    Failed,
}

impl std::fmt::Display for WarmupState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fetching => write!(f, "Fetching"),
            Self::Processing => write!(f, "Processing"),
            Self::Ready => write!(f, "Ready"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Warmup progress tracker
#[derive(Debug, Clone, Default)]
pub struct WarmupProgress {
    pub state: WarmupState,
    pub symbols_total: usize,
    pub symbols_fetched: usize,
    pub candles_processed: usize,
    pub candles_total: usize,
    pub error_message: Option<String>,
}

impl WarmupProgress {
    pub fn progress_pct(&self) -> f64 {
        match self.state {
            WarmupState::Fetching => {
                if self.symbols_total == 0 {
                    0.0
                } else {
                    (self.symbols_fetched as f64 / self.symbols_total as f64) * 50.0
                }
            }
            WarmupState::Processing => {
                if self.candles_total == 0 {
                    50.0
                } else {
                    50.0 + (self.candles_processed as f64 / self.candles_total as f64) * 50.0
                }
            }
            WarmupState::Ready => 100.0,
            WarmupState::Failed => 0.0,
        }
    }
}

/// Convert historical candles to simulated ticks for regime warmup
pub fn candles_to_feature_vectors(
    candles: &[HistoricalCandle],
) -> Vec<(DateTime<Utc>, f64, FeatureVector)> {
    let mut results = Vec::new();
    let mut prev_close = 0.0;

    for candle in candles {
        // Parse timestamp
        let ts = chrono::DateTime::parse_from_str(&candle.timestamp, "%Y-%m-%d %H:%M:%S")
            .or_else(|_| chrono::DateTime::parse_from_str(&candle.timestamp, "%Y-%m-%dT%H:%M:%S%z"))
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let close = candle.close;

        // Calculate returns in basis points
        let mid_return = if prev_close > 0.0 {
            (((close - prev_close) / prev_close) * 10000.0) as i64
        } else {
            0
        };

        // Create feature vector with simulated microstructure
        // For candles we don't have tick-level data, so use neutral values
        let features = FeatureVector::new(
            mid_return, 0,  // trade_imbalance (neutral)
            50, // bid_depth_ratio (neutral)
            50, // ask_depth_ratio (neutral)
            50, // spread_bps (neutral)
            0,  // tick_intensity (neutral)
        );

        results.push((ts, close, features));
        prev_close = close;
    }

    results
}

/// Aggregate candles from multiple symbols into a single warmup sequence
/// Uses the average return across all symbols for regime detection
pub fn aggregate_warmup_data(
    data: &HashMap<String, Vec<HistoricalCandle>>,
) -> Vec<(DateTime<Utc>, f64, FeatureVector)> {
    if data.is_empty() {
        return Vec::new();
    }

    // Convert each symbol's candles
    let mut all_vectors: HashMap<String, Vec<(DateTime<Utc>, f64, FeatureVector)>> = HashMap::new();

    for (symbol, candles) in data {
        all_vectors.insert(symbol.clone(), candles_to_feature_vectors(candles));
    }

    // Find the symbol with most candles (typically the underlying or most liquid)
    let (primary_symbol, primary_data) = all_vectors
        .iter()
        .max_by_key(|(_, v)| v.len())
        .map(|(s, v)| (s.clone(), v.clone()))
        .unwrap_or_default();

    if primary_data.is_empty() {
        return Vec::new();
    }

    tracing::info!(
        "[WARMUP] Using {} as primary symbol ({} candles)",
        primary_symbol,
        primary_data.len()
    );

    primary_data
}

/// Configuration for regime warmup
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Number of minutes of historical data to fetch
    pub lookback_minutes: i64,
    /// Minimum candles required for valid warmup
    pub min_candles: usize,
    /// Whether to block trading until warmup complete
    pub block_until_ready: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            lookback_minutes: 60, // 1 hour of data
            min_candles: 30,      // At least 30 candles
            block_until_ready: true,
        }
    }
}
