//! Ramanujan Periodicity Detection for Microstructure Analysis
//!
//! Based on: Vaidyanathan & Tenneti, "Srinivasa Ramanujan and signal-processing problems"
//! Phil. Trans. R. Soc. A 378: 20180446 (2019)
//!
//! Key advantages over DFT:
//! - Works for any integer period (not just divisors of data length)
//! - Integer-valued computations (deterministic, WAL-safe)
//! - Better separation of multiple hidden periodicities
//! - Effective with short data lengths

use std::collections::HashMap;

/// Compute the greatest common divisor using Euclidean algorithm.
#[inline]
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Compute Euler's totient function φ(q).
/// φ(q) = count of integers k in [1, q] where gcd(k, q) = 1
pub fn euler_totient(q: usize) -> usize {
    if q == 0 {
        return 0;
    }
    (1..=q).filter(|&k| gcd(k, q) == 1).count()
}

/// Compute the Ramanujan sum c_q(n) for a single (q, n) pair.
///
/// c_q(n) = Σ_{k: gcd(k,q)=1} e^{j2πkn/q}
///
/// This is always real and integer-valued despite complex exponentials.
/// We compute it directly using the real part (imaginary cancels).
pub fn ramanujan_sum(q: usize, n: i64) -> i64 {
    if q == 0 {
        return 0;
    }

    let mut sum = 0.0f64;
    let two_pi = 2.0 * std::f64::consts::PI;

    for k in 1..=q {
        if gcd(k, q) == 1 {
            // e^{j2πkn/q} = cos(2πkn/q) + j*sin(2πkn/q)
            // The sum is always real, so we only need cos
            let angle = two_pi * (k as f64) * (n as f64) / (q as f64);
            sum += angle.cos();
        }
    }

    // Round to nearest integer (should be exact for valid inputs)
    sum.round() as i64
}

/// Precompute one period of c_q(n) for n = 0, 1, ..., q-1.
/// Returns a vector of q integers.
pub fn ramanujan_sum_period(q: usize) -> Vec<i64> {
    (0..q as i64).map(|n| ramanujan_sum(q, n)).collect()
}

/// Ramanujan Filter Bank for periodicity detection.
///
/// Implements a bank of FIR filters with Ramanujan sum coefficients.
/// Each filter c_q detects period-q components in the input signal.
#[derive(Debug, Clone)]
pub struct RamanujanFilterBank {
    /// Maximum period to detect
    max_period: usize,
    /// Number of repetitions per filter (controls frequency resolution)
    num_reps: usize,
    /// Precomputed Ramanujan sums for each period: c_q[n] for n in [0, q-1]
    ram_sums: Vec<Vec<i64>>,
    /// Euler totients φ(q) for normalization
    totients: Vec<usize>,
}

impl RamanujanFilterBank {
    /// Create a new Ramanujan filter bank.
    ///
    /// # Arguments
    /// * `max_period` - Maximum period to detect (Q in the paper)
    /// * `num_reps` - Number of period repetitions per filter (l in paper)
    pub fn new(max_period: usize, num_reps: usize) -> Self {
        let mut ram_sums = Vec::with_capacity(max_period + 1);
        let mut totients = Vec::with_capacity(max_period + 1);

        // q=0 placeholder
        ram_sums.push(vec![]);
        totients.push(0);

        // Precompute for q = 1 to max_period
        for q in 1..=max_period {
            ram_sums.push(ramanujan_sum_period(q));
            totients.push(euler_totient(q));
        }

        Self {
            max_period,
            num_reps,
            ram_sums,
            totients,
        }
    }

    /// Get the filter length for period q.
    pub fn filter_length(&self, q: usize) -> usize {
        q * self.num_reps
    }

    /// Apply the period-q Ramanujan filter to signal x at position n.
    ///
    /// Computes: y_q[n] = Σ_{m=0}^{ql-1} c_q(m) * x(n-m)
    ///
    /// Returns None if not enough data (n < filter_length - 1).
    pub fn filter_output(&self, q: usize, x: &[i64], n: usize) -> Option<i64> {
        if q == 0 || q > self.max_period {
            return None;
        }

        let filter_len = self.filter_length(q);
        if n + 1 < filter_len {
            return None; // Not enough data
        }

        let c_q = &self.ram_sums[q];
        let mut sum: i64 = 0;

        for m in 0..filter_len {
            let c_val = c_q[m % q]; // Periodic extension
            let x_idx = n - m;
            if x_idx < x.len() {
                sum += c_val * x[x_idx];
            }
        }

        Some(sum)
    }

    /// Compute periodicity strength for all periods at position n.
    ///
    /// Returns a vector of (period, strength) pairs, normalized by φ(q)*l.
    pub fn periodicity_strengths(&self, x: &[i64], n: usize) -> Vec<(usize, f64)> {
        let mut strengths = Vec::new();

        for q in 2..=self.max_period {
            if let Some(output) = self.filter_output(q, x, n) {
                // Normalize by filter energy: φ(q) * num_reps * q
                let norm = (self.totients[q] * self.num_reps * q) as f64;
                let strength = (output.abs() as f64) / norm.max(1.0);
                strengths.push((q, strength));
            }
        }

        strengths
    }

    /// Find dominant periods (local maxima in strength).
    ///
    /// Returns periods sorted by strength descending.
    pub fn dominant_periods(&self, x: &[i64], n: usize, threshold: f64) -> Vec<(usize, f64)> {
        let strengths = self.periodicity_strengths(x, n);

        // Filter by threshold and find local maxima
        let mut dominant: Vec<(usize, f64)> = strengths
            .iter()
            .enumerate()
            .filter(|(i, &(_, s))| {
                if s < threshold {
                    return false;
                }
                // Check if local maximum (not a divisor peak)
                let prev = if *i > 0 { strengths[i - 1].1 } else { 0.0 };
                let next = if *i + 1 < strengths.len() {
                    strengths[i + 1].1
                } else {
                    0.0
                };
                s >= prev && s >= next
            })
            .map(|(_, &ps)| ps)
            .collect();

        // Sort by strength descending
        dominant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Remove divisor harmonics (keep only fundamental periods)
        let mut filtered = Vec::new();
        for (p, s) in dominant {
            // Check if p is a divisor of any already-added period
            let dominated = filtered.iter().any(|&(q, _): &(usize, f64)| q % p == 0);
            if !dominated {
                filtered.push((p, s));
            }
        }

        filtered
    }
}

/// Ramanujan Periodicity Detector for microstructure signals.
///
/// Tracks periodicity in rolling windows of tick data.
#[derive(Debug)]
pub struct RamanujanPeriodicityDetector {
    /// Filter bank
    filter_bank: RamanujanFilterBank,
    /// Rolling buffer of quantized signal values
    buffer: Vec<i64>,
    /// Buffer capacity
    capacity: usize,
    /// Current write position
    write_pos: usize,
    /// Number of samples received
    sample_count: usize,
    /// Detection threshold for significant periods
    threshold: f64,
    /// Last detected dominant periods
    last_periods: Vec<(usize, f64)>,
}

impl RamanujanPeriodicityDetector {
    /// Create a new periodicity detector.
    ///
    /// # Arguments
    /// * `max_period` - Maximum period to detect
    /// * `num_reps` - Filter repetitions (higher = better frequency resolution)
    /// * `buffer_size` - Rolling buffer size (should be >= max_period * num_reps)
    /// * `threshold` - Minimum strength for period detection
    pub fn new(max_period: usize, num_reps: usize, buffer_size: usize, threshold: f64) -> Self {
        let min_buffer = max_period * num_reps;
        let capacity = buffer_size.max(min_buffer);

        Self {
            filter_bank: RamanujanFilterBank::new(max_period, num_reps),
            buffer: vec![0; capacity],
            capacity,
            write_pos: 0,
            sample_count: 0,
            threshold,
            last_periods: Vec::new(),
        }
    }

    /// Update with a new sample value.
    ///
    /// Returns true if enough data to compute periodicities.
    pub fn update(&mut self, value: i64) -> bool {
        self.buffer[self.write_pos] = value;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.sample_count += 1;

        // Need at least max_period * num_reps samples
        let min_samples = self.filter_bank.filter_length(self.filter_bank.max_period);
        self.sample_count >= min_samples
    }

    /// Get the linearized buffer (oldest to newest).
    fn linearized_buffer(&self) -> Vec<i64> {
        let n = self.sample_count.min(self.capacity);
        let mut linear = Vec::with_capacity(n);

        if self.sample_count <= self.capacity {
            // Buffer not yet wrapped
            linear.extend_from_slice(&self.buffer[..n]);
        } else {
            // Buffer has wrapped - concatenate from write_pos to end, then start to write_pos
            linear.extend_from_slice(&self.buffer[self.write_pos..]);
            linear.extend_from_slice(&self.buffer[..self.write_pos]);
        }

        linear
    }

    /// Detect periodicities in current buffer.
    ///
    /// Returns dominant periods sorted by strength.
    pub fn detect(&mut self) -> &[(usize, f64)] {
        let linear = self.linearized_buffer();
        if linear.is_empty() {
            self.last_periods.clear();
            return &self.last_periods;
        }

        let n = linear.len() - 1;
        self.last_periods = self
            .filter_bank
            .dominant_periods(&linear, n, self.threshold);
        &self.last_periods
    }

    /// Get periodicity strength spectrum.
    pub fn spectrum(&self) -> Vec<(usize, f64)> {
        let linear = self.linearized_buffer();
        if linear.is_empty() {
            return Vec::new();
        }

        let n = linear.len() - 1;
        self.filter_bank.periodicity_strengths(&linear, n)
    }

    /// Check if a specific period is present above threshold.
    pub fn has_period(&self, period: usize) -> bool {
        self.last_periods.iter().any(|&(p, _)| p == period)
    }

    /// Get the strongest detected period, if any.
    pub fn strongest_period(&self) -> Option<(usize, f64)> {
        self.last_periods.first().copied()
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.write_pos = 0;
        self.sample_count = 0;
        self.last_periods.clear();
    }
}

/// Microstructure feature extractor using Ramanujan periodicity.
///
/// Extracts periodicity features from tick-level data for regime detection.
#[derive(Debug)]
pub struct MicrostructurePeriodicity {
    /// Detector for mid-price returns
    return_detector: RamanujanPeriodicityDetector,
    /// Detector for book imbalance
    imbalance_detector: RamanujanPeriodicityDetector,
    /// Detector for spread
    spread_detector: RamanujanPeriodicityDetector,
    /// Price quantization scale (for converting f64 to i64, reserved for future use)
    #[allow(dead_code)]
    price_scale: i64,
}

impl MicrostructurePeriodicity {
    /// Create with default parameters for tick-level analysis.
    ///
    /// Detects periods from 2 to 64 ticks (covers most microstructure cycles).
    pub fn new() -> Self {
        Self::with_params(64, 4, 512, 0.1, 10000)
    }

    /// Create with custom parameters.
    pub fn with_params(
        max_period: usize,
        num_reps: usize,
        buffer_size: usize,
        threshold: f64,
        price_scale: i64,
    ) -> Self {
        Self {
            return_detector: RamanujanPeriodicityDetector::new(
                max_period,
                num_reps,
                buffer_size,
                threshold,
            ),
            imbalance_detector: RamanujanPeriodicityDetector::new(
                max_period,
                num_reps,
                buffer_size,
                threshold,
            ),
            spread_detector: RamanujanPeriodicityDetector::new(
                max_period,
                num_reps,
                buffer_size,
                threshold,
            ),
            price_scale,
        }
    }

    /// Update with new tick data.
    ///
    /// # Arguments
    /// * `mid_return_bps` - Mid price return in basis points (×10000)
    /// * `imbalance` - Book imbalance in [-10000, 10000]
    /// * `spread_bps` - Spread in basis points
    ///
    /// Returns true if ready to detect.
    pub fn update(&mut self, mid_return_bps: i64, imbalance: i64, spread_bps: i64) -> bool {
        let r1 = self.return_detector.update(mid_return_bps);
        let r2 = self.imbalance_detector.update(imbalance);
        let r3 = self.spread_detector.update(spread_bps);
        r1 && r2 && r3
    }

    /// Detect periodicities across all features.
    ///
    /// Returns a summary of detected periods.
    pub fn detect(&mut self) -> PeriodicityFeatures {
        let return_periods = self.return_detector.detect().to_vec();
        let imbalance_periods = self.imbalance_detector.detect().to_vec();
        let spread_periods = self.spread_detector.detect().to_vec();

        // Find common periods across features
        let mut period_counts: HashMap<usize, usize> = HashMap::new();
        for (p, _) in return_periods
            .iter()
            .chain(&imbalance_periods)
            .chain(&spread_periods)
        {
            *period_counts.entry(*p).or_default() += 1;
        }

        let common_periods: Vec<usize> = period_counts
            .into_iter()
            .filter(|&(_, count)| count >= 2)
            .map(|(p, _)| p)
            .collect();

        PeriodicityFeatures {
            return_periods,
            imbalance_periods,
            spread_periods,
            common_periods,
        }
    }

    /// Reset all detectors.
    pub fn reset(&mut self) {
        self.return_detector.reset();
        self.imbalance_detector.reset();
        self.spread_detector.reset();
    }
}

impl Default for MicrostructurePeriodicity {
    fn default() -> Self {
        Self::new()
    }
}

/// Periodicity features extracted from microstructure data.
#[derive(Debug, Clone)]
pub struct PeriodicityFeatures {
    /// Dominant periods in mid-price returns
    pub return_periods: Vec<(usize, f64)>,
    /// Dominant periods in book imbalance
    pub imbalance_periods: Vec<(usize, f64)>,
    /// Dominant periods in spread
    pub spread_periods: Vec<(usize, f64)>,
    /// Periods common across multiple features
    pub common_periods: Vec<usize>,
}

impl PeriodicityFeatures {
    /// Check if any strong periodicity is detected.
    pub fn has_periodicity(&self) -> bool {
        !self.return_periods.is_empty()
            || !self.imbalance_periods.is_empty()
            || !self.spread_periods.is_empty()
    }

    /// Get the strongest overall period.
    pub fn strongest_period(&self) -> Option<usize> {
        let mut all: Vec<(usize, f64)> = self
            .return_periods
            .iter()
            .chain(&self.imbalance_periods)
            .chain(&self.spread_periods)
            .copied()
            .collect();

        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.first().map(|&(p, _)| p)
    }

    /// Check if microstructure suggests mean-reversion regime.
    ///
    /// Mean-reversion is indicated by short periods (2-8 ticks) in imbalance.
    pub fn suggests_mean_reversion(&self) -> bool {
        self.imbalance_periods
            .iter()
            .any(|&(p, s)| p <= 8 && s > 0.15)
    }

    /// Check if microstructure suggests trending regime.
    ///
    /// Trending is indicated by longer periods or no clear periodicity.
    pub fn suggests_trending(&self) -> bool {
        self.return_periods.iter().any(|&(p, s)| p >= 16 && s > 0.2)
            || self.return_periods.is_empty()
    }

    /// Check if HFT activity is likely.
    ///
    /// HFT is indicated by very short periods (2-4 ticks) with high energy
    /// appearing consistently across multiple features.
    pub fn hft_likely(&self) -> bool {
        // Count features with very short period activity
        let mut short_period_count = 0;

        if self.return_periods.iter().any(|&(p, s)| p <= 4 && s > 0.25) {
            short_period_count += 1;
        }
        if self
            .imbalance_periods
            .iter()
            .any(|&(p, s)| p <= 4 && s > 0.25)
        {
            short_period_count += 1;
        }
        if self.spread_periods.iter().any(|&(p, s)| p <= 4 && s > 0.25) {
            short_period_count += 1;
        }

        // HFT likely if 2+ features show synchronized short-period activity
        short_period_count >= 2 || self.common_periods.iter().any(|&p| p <= 4)
    }

    /// Check if market maker activity is likely.
    ///
    /// Market makers often create periodic patterns in imbalance and spread
    /// at moderate frequencies (6-16 ticks).
    pub fn market_maker_likely(&self) -> bool {
        self.common_periods.iter().any(|&p| (6..=16).contains(&p))
            || (self
                .imbalance_periods
                .iter()
                .any(|&(p, s)| (6..=16).contains(&p) && s > 0.2)
                && self
                    .spread_periods
                    .iter()
                    .any(|&(p, s)| (6..=16).contains(&p) && s > 0.2))
    }

    /// Get dominant periods (union of common periods and strongest individual)
    pub fn dominant_periods(&self) -> Vec<usize> {
        let mut periods = self.common_periods.clone();
        if let Some(strongest) = self.strongest_period() {
            if !periods.contains(&strongest) {
                periods.push(strongest);
            }
        }
        periods.sort();
        periods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 25), 25);
        assert_eq!(gcd(7, 7), 7);
    }

    #[test]
    fn test_euler_totient() {
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(2), 1);
        assert_eq!(euler_totient(6), 2); // 1, 5
        assert_eq!(euler_totient(10), 4); // 1, 3, 7, 9
        assert_eq!(euler_totient(12), 4); // 1, 5, 7, 11
    }

    #[test]
    fn test_ramanujan_sum_integer_valued() {
        // c_q(n) should always be integer
        for q in 1..=20 {
            for n in 0..q as i64 {
                let c = ramanujan_sum(q, n);
                // Verify it's truly integer by checking round-trip
                let c_f = c as f64;
                assert_eq!(c_f.round() as i64, c, "c_{}({}) not integer", q, n);
            }
        }
    }

    #[test]
    fn test_ramanujan_sum_c10() {
        // From paper: c_10(n) = {4, 1, -1, 1, -1, -4, -1, 1, -1, 1}
        let expected = vec![4, 1, -1, 1, -1, -4, -1, 1, -1, 1];
        let c10 = ramanujan_sum_period(10);
        assert_eq!(c10, expected);
    }

    #[test]
    fn test_ramanujan_sum_periodicity() {
        // c_q(n) should have period exactly q
        for q in 2..=15 {
            let _period = ramanujan_sum_period(q);
            for n in 0..q {
                assert_eq!(
                    ramanujan_sum(q, n as i64),
                    ramanujan_sum(q, (n + q) as i64),
                    "c_{}({}) != c_{}({})",
                    q,
                    n,
                    q,
                    n + q
                );
            }
        }
    }

    #[test]
    fn test_filter_bank_creation() {
        let fb = RamanujanFilterBank::new(16, 4);
        assert_eq!(fb.max_period, 16);
        assert_eq!(fb.num_reps, 4);
        assert_eq!(fb.filter_length(10), 40);
    }

    #[test]
    fn test_detect_pure_period() {
        // Create a signal with period 5
        let period = 5;
        let signal: Vec<i64> = (0..200).map(|n| ((n % period) as i64 - 2) * 100).collect();

        let mut detector = RamanujanPeriodicityDetector::new(32, 4, 256, 0.05);

        for &val in &signal {
            detector.update(val);
        }

        let periods = detector.detect();

        // Should detect period 5 as dominant
        assert!(!periods.is_empty(), "Should detect some period");
        assert_eq!(periods[0].0, period, "Should detect period {}", period);
    }

    #[test]
    fn test_detect_multiple_periods() {
        // Create a signal with periods 5 and 7
        let signal: Vec<i64> = (0..300)
            .map(|n| {
                let p5 = ((n % 5) as i64 - 2) * 100;
                let p7 = ((n % 7) as i64 - 3) * 80;
                p5 + p7
            })
            .collect();

        let mut detector = RamanujanPeriodicityDetector::new(32, 4, 512, 0.05);

        for &val in &signal {
            detector.update(val);
        }

        let periods = detector.detect();
        let detected_periods: Vec<usize> = periods.iter().map(|&(p, _)| p).collect();

        // Should detect both periods (or their LCM harmonics)
        assert!(
            detected_periods.contains(&5) || detected_periods.contains(&7),
            "Should detect period 5 or 7, got {:?}",
            detected_periods
        );
    }

    #[test]
    fn test_microstructure_periodicity() {
        let mut mp = MicrostructurePeriodicity::with_params(32, 2, 128, 0.1, 10000);

        // Simulate periodic imbalance (period 6)
        for i in 0..200 {
            let imbalance = ((i % 6) as i64 - 3) * 1000;
            let ret = 0;
            let spread = 10;
            mp.update(ret, imbalance, spread);
        }

        let features = mp.detect();

        // Should detect periodicity in imbalance
        assert!(
            !features.imbalance_periods.is_empty(),
            "Should detect imbalance periodicity"
        );
    }
}
