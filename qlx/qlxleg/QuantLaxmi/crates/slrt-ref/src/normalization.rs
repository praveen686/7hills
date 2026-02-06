//! Normalization using rolling median and MAD (Median Absolute Deviation).
//!
//! Spec: Section 6.2 Normalization (SEALED)
//!
//! - Warmup: 5 minutes or 30,000 ticks (minimum)
//! - Post-warmup: rolling median / MAD (30 minutes)
//! - Per-symbol only
//!
//! Normalization Failure Handling:
//! - MAD == 0 → confidence × 0.8
//! - MAD == 0 for > 2 seconds → RefuseSignal
//! - Reset only on session boundary (never on reconnect)

use crate::sealed::{
    MAD_ZERO_REFUSE_SECS, NORM_WINDOW_SECS, STATE_DIM, WARMUP_DURATION_SECS, WARMUP_MIN_TICKS,
};
use std::collections::VecDeque;

/// Normalization status.
/// v1.2: Split into Normal / Degraded / DegradedHigh / RefuseFrame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationStatus {
    /// Still in warmup period
    Warmup,
    /// Normal operation - full confidence
    Normal,
    /// Degraded quality - confidence penalty, still tradeable
    /// Examples: MAD==0 ≤2s, undefined elasticity, NoTradesWindow, spread breach
    Degraded,
    /// Degraded high - stronger confidence penalty, optionally block trading
    /// Examples: MAD==0 >2s
    DegradedHigh,
    /// Refuse frame - structural invalidity, do not trade
    /// Examples: missing book, crossed book, NaN/non-finite state
    RefuseFrame,
}

bitflags::bitflags! {
    /// Reason bitmap for degraded/refused status.
    /// Multiple reasons can be active simultaneously.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
    pub struct DegradedReasons: u32 {
        /// MAD == 0 on one or more dimensions (≤2s)
        const MAD_ZERO_SHORT = 1 << 0;
        /// MAD == 0 on one or more dimensions (>2s)
        const MAD_ZERO_LONG = 1 << 1;
        /// Elasticity undefined (insufficient trade volume)
        const UNDEFINED_ELASTICITY = 1 << 2;
        /// No trades in window (toxicity not computable)
        const NO_TRADES_WINDOW = 1 << 3;
        /// Insufficient price history (FTI not computable)
        const INSUFFICIENT_HISTORY = 1 << 4;
        /// Spread soft breach (spread > threshold)
        const SPREAD_BREACH = 1 << 5;
        /// Missing book (no bids or asks)
        const MISSING_BOOK = 1 << 6;
        /// Crossed book (best bid > best ask)
        const CROSSED_BOOK = 1 << 7;
        /// NaN or non-finite value in state vector
        const NON_FINITE_STATE = 1 << 8;
    }
}

impl DegradedReasons {
    /// Check if any RefuseFrame-level reason is set.
    pub fn has_refuse_reason(&self) -> bool {
        self.intersects(
            DegradedReasons::MISSING_BOOK
                | DegradedReasons::CROSSED_BOOK
                | DegradedReasons::NON_FINITE_STATE,
        )
    }

    /// Check if MAD_ZERO_LONG is set (DegradedHigh condition).
    pub fn has_degraded_high_reason(&self) -> bool {
        self.contains(DegradedReasons::MAD_ZERO_LONG)
    }
}

/// Single dimension normalizer using rolling median/MAD.
#[derive(Debug, Clone)]
struct DimensionNormalizer {
    /// Rolling window of values
    values: VecDeque<(i64, f64)>, // (ts_ns, value)
    /// Window size in nanoseconds
    window_ns: i64,
    /// Cached median
    cached_median: f64,
    /// Cached MAD
    cached_mad: f64,
    /// Last update timestamp
    last_update_ns: i64,
    /// Whether cache is valid
    cache_valid: bool,
}

impl DimensionNormalizer {
    fn new(window_secs: u64) -> Self {
        Self {
            values: VecDeque::new(),
            window_ns: window_secs as i64 * 1_000_000_000,
            cached_median: 0.0,
            cached_mad: 1.0,
            last_update_ns: 0,
            cache_valid: false,
        }
    }

    /// Add a new value and evict old ones.
    fn add(&mut self, ts_ns: i64, value: f64) {
        // Evict old values
        let cutoff = ts_ns - self.window_ns;
        while let Some((old_ts, _)) = self.values.front() {
            if *old_ts < cutoff {
                self.values.pop_front();
            } else {
                break;
            }
        }

        self.values.push_back((ts_ns, value));
        self.last_update_ns = ts_ns;
        self.cache_valid = false;
    }

    /// Compute median of current window.
    fn compute_median(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f64> = self.values.iter().map(|(_, v)| *v).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Compute MAD (Median Absolute Deviation) of current window.
    fn compute_mad(&self, median: f64) -> f64 {
        if self.values.is_empty() {
            return 1.0; // Avoid division by zero
        }

        let mut deviations: Vec<f64> = self
            .values
            .iter()
            .map(|(_, v)| (*v - median).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = deviations.len() / 2;
        if deviations.len().is_multiple_of(2) {
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[mid]
        }
    }

    /// Update cache if needed and return (median, mad).
    fn get_stats(&mut self) -> (f64, f64) {
        if !self.cache_valid {
            self.cached_median = self.compute_median();
            self.cached_mad = self.compute_mad(self.cached_median);
            self.cache_valid = true;
        }
        (self.cached_median, self.cached_mad)
    }

    /// Normalize a value using current median/MAD.
    fn normalize(&mut self, value: f64) -> (f64, bool) {
        let (median, mad) = self.get_stats();

        if mad < 1e-12 {
            // MAD is effectively zero
            (0.0, true)
        } else {
            ((value - median) / mad, false)
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// State vector normalizer (all dimensions).
/// Spec: Section 6.2
pub struct StateNormalizer {
    /// Per-dimension normalizers
    normalizers: Vec<DimensionNormalizer>,
    /// Session start timestamp
    session_start_ns: i64,
    /// Tick count since session start
    tick_count: u64,
    /// Last timestamp when MAD was zero
    last_zero_mad_ns: Option<i64>,
    /// Warmup duration in nanoseconds
    warmup_ns: i64,
    /// Minimum ticks for warmup
    warmup_min_ticks: u64,
}

impl StateNormalizer {
    /// Create a new state normalizer.
    pub fn new() -> Self {
        Self {
            normalizers: (0..STATE_DIM)
                .map(|_| DimensionNormalizer::new(NORM_WINDOW_SECS))
                .collect(),
            session_start_ns: 0,
            tick_count: 0,
            last_zero_mad_ns: None,
            warmup_ns: WARMUP_DURATION_SECS as i64 * 1_000_000_000,
            warmup_min_ticks: WARMUP_MIN_TICKS as u64,
        }
    }

    /// Reset for a new session.
    pub fn reset(&mut self, session_start_ns: i64) {
        for norm in &mut self.normalizers {
            norm.values.clear();
            norm.cache_valid = false;
        }
        self.session_start_ns = session_start_ns;
        self.tick_count = 0;
        self.last_zero_mad_ns = None;
    }

    /// Check if still in warmup period.
    fn is_warmup(&self, ts_ns: i64) -> bool {
        let elapsed_ns = ts_ns - self.session_start_ns;
        elapsed_ns < self.warmup_ns || self.tick_count < self.warmup_min_ticks
    }

    /// Add raw state vector values.
    pub fn add_raw(&mut self, ts_ns: i64, raw: &[f64; STATE_DIM]) {
        for (i, &value) in raw.iter().enumerate() {
            self.normalizers[i].add(ts_ns, value);
        }
        self.tick_count += 1;
    }

    /// Normalize a state vector.
    /// Returns (normalized_state, status, reasons).
    /// v1.2: Returns DegradedReasons bitmap for all degradation conditions.
    pub fn normalize(
        &mut self,
        ts_ns: i64,
        raw: &[f64; STATE_DIM],
    ) -> ([f64; STATE_DIM], NormalizationStatus, DegradedReasons) {
        let mut reasons = DegradedReasons::empty();

        // Check for NaN/non-finite values first (RefuseFrame condition)
        for &val in raw.iter() {
            if !val.is_finite() {
                reasons |= DegradedReasons::NON_FINITE_STATE;
                return ([0.0; STATE_DIM], NormalizationStatus::RefuseFrame, reasons);
            }
        }

        // Add to history
        self.add_raw(ts_ns, raw);

        // Check warmup
        if self.is_warmup(ts_ns) {
            return ([0.0; STATE_DIM], NormalizationStatus::Warmup, reasons);
        }

        // Normalize each dimension
        let mut normalized = [0.0f64; STATE_DIM];
        let mut any_zero_mad = false;

        for i in 0..STATE_DIM {
            let (norm_value, zero_mad) = self.normalizers[i].normalize(raw[i]);
            normalized[i] = norm_value;
            if zero_mad {
                any_zero_mad = true;
            }
        }

        // Handle zero MAD - now Degraded/DegradedHigh instead of RefuseSignal
        let status = if any_zero_mad {
            match self.last_zero_mad_ns {
                Some(last_ts) => {
                    let duration_secs = (ts_ns - last_ts) as f64 / 1_000_000_000.0;
                    if duration_secs > MAD_ZERO_REFUSE_SECS {
                        reasons |= DegradedReasons::MAD_ZERO_LONG;
                        NormalizationStatus::DegradedHigh
                    } else {
                        reasons |= DegradedReasons::MAD_ZERO_SHORT;
                        NormalizationStatus::Degraded
                    }
                }
                None => {
                    self.last_zero_mad_ns = Some(ts_ns);
                    reasons |= DegradedReasons::MAD_ZERO_SHORT;
                    NormalizationStatus::Degraded
                }
            }
        } else {
            self.last_zero_mad_ns = None;
            NormalizationStatus::Normal
        };

        (normalized, status, reasons)
    }

    /// Get current tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Get minimum values in any dimension.
    pub fn min_samples(&self) -> usize {
        self.normalizers.iter().map(|n| n.len()).min().unwrap_or(0)
    }
}

impl Default for StateNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd() {
        let mut norm = DimensionNormalizer::new(60);
        norm.add(1, 1.0);
        norm.add(2, 3.0);
        norm.add(3, 2.0);

        let median = norm.compute_median();
        assert!((median - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        let mut norm = DimensionNormalizer::new(60);
        norm.add(1, 1.0);
        norm.add(2, 2.0);
        norm.add(3, 3.0);
        norm.add(4, 4.0);

        let median = norm.compute_median();
        assert!((median - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mad() {
        let mut norm = DimensionNormalizer::new(60);
        norm.add(1, 1.0);
        norm.add(2, 2.0);
        norm.add(3, 3.0);
        norm.add(4, 4.0);
        norm.add(5, 5.0);

        let median = norm.compute_median(); // 3.0
        let mad = norm.compute_mad(median);
        // Deviations: |1-3|=2, |2-3|=1, |3-3|=0, |4-3|=1, |5-3|=2
        // Sorted: 0, 1, 1, 2, 2 -> median = 1
        assert!((mad - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_warmup() {
        let mut norm = StateNormalizer::new();
        norm.reset(0);

        let raw = [0.0; STATE_DIM];
        let (_, status, _reasons) = norm.normalize(1_000_000_000, &raw); // 1 second in

        assert_eq!(status, NormalizationStatus::Warmup);
    }

    #[test]
    fn test_refuse_frame_on_nan() {
        let mut norm = StateNormalizer::new();
        norm.reset(0);

        let mut raw = [0.0; STATE_DIM];
        raw[0] = f64::NAN; // Put NaN in first dimension
        let (_, status, reasons) = norm.normalize(1_000_000_000, &raw);

        assert_eq!(status, NormalizationStatus::RefuseFrame);
        assert!(reasons.contains(DegradedReasons::NON_FINITE_STATE));
    }
}
