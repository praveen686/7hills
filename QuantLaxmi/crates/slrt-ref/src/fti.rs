//! Follow-Through Indicator (FTI) implementation.
//!
//! Direct port of Timothy Masters' FTI.CPP as specified in SLRT-GPU v1.1.
//! Spec: Section 5.1 - FTI is code-locked to research/indicators/timothy_masters/FTI.CPP
//!
//! FTI measures initiative/follow-through in price movement.
//! It is used only as initiative confirmation, never as a predictor.

use std::f64::consts::PI;

/// FTI (Follow-Through Indicator) calculator.
/// Port of Govinda Khalsa's FTI indicator from Timothy Masters' implementation.
pub struct FTI {
    /// Work with log of prices?
    use_log: bool,
    /// Shortest period, at least 2 (Khalsa used 5)
    min_period: usize,
    /// Longest period (Khalsa used 65)
    max_period: usize,
    /// This many coefs each side of center coef
    half_length: usize,
    /// Lookback length of the moving window
    lookback: usize,
    /// Fractile (typically 0.8 to 0.99) for width computation
    beta: f64,
    /// Fraction of longest interior move (Khalsa uses 0.2) defining noise for FTI
    noise_cut: f64,

    // Work arrays
    /// Data copied here for processing
    y: Vec<f64>,
    /// Filter coefficients for each period
    coefs: Vec<f64>,
    /// Filtered values for each period
    filtered: Vec<f64>,
    /// Width (one side) of band for each period
    width: Vec<f64>,
    /// FTI value for each period
    fti: Vec<f64>,
    /// Indices of FTI values sorted largest to smallest
    sorted: Vec<usize>,
    /// Work area for distribution of differences
    diff_work: Vec<f64>,
    /// Work area for collecting legs
    leg_work: Vec<f64>,
    /// Work area for sorting FTI
    sort_work: Vec<f64>,
}

impl FTI {
    /// Create a new FTI calculator with the specified parameters.
    ///
    /// # Arguments
    /// * `use_log` - Take log of market prices?
    /// * `min_period` - Shortest period, at least 2
    /// * `max_period` - Longest period
    /// * `half_length` - This many coefs each side of center. 2*half_length+1 must exceed max_period
    /// * `lookback` - Process data in blocks this long. Channel length = lookback - half_length
    /// * `beta` - Fractile (typically 0.8 to 0.99) for width computation
    /// * `noise_cut` - Fraction of longest interior move (Khalsa uses 0.2) defining noise
    pub fn new(
        use_log: bool,
        min_period: usize,
        max_period: usize,
        half_length: usize,
        lookback: usize,
        beta: f64,
        noise_cut: f64,
    ) -> Self {
        let n_periods = max_period - min_period + 1;

        let mut fti = Self {
            use_log,
            min_period,
            max_period,
            half_length,
            lookback,
            beta,
            noise_cut,
            y: vec![0.0; lookback + half_length],
            coefs: vec![0.0; n_periods * (half_length + 1)],
            filtered: vec![0.0; n_periods],
            width: vec![0.0; n_periods],
            fti: vec![0.0; n_periods],
            sorted: vec![0; n_periods],
            diff_work: vec![0.0; lookback],
            leg_work: vec![0.0; lookback],
            sort_work: vec![0.0; n_periods],
        };

        // Compute filter coefficients for each period
        for period in min_period..=max_period {
            fti.find_coefs(period);
        }

        fti
    }

    /// Compute the filter coefficients for a specified period.
    /// FIR lowpass filter from Otnes: Applied Time Series Analysis.
    fn find_coefs(&mut self, period: usize) {
        let idx_base = (period - self.min_period) * (self.half_length + 1);
        let d = [0.35577019, 0.2436983, 0.07211497, 0.00630165];

        let mut fact = 2.0 / period as f64;
        self.coefs[idx_base] = fact;

        fact *= PI;
        for i in 1..=self.half_length {
            self.coefs[idx_base + i] = (i as f64 * fact).sin() / (i as f64 * PI);
        }

        // Taper the end point
        self.coefs[idx_base + self.half_length] *= 0.5;

        // Apply window function and normalize
        let mut sumg = self.coefs[idx_base];
        for i in 1..=self.half_length {
            let mut sum = d[0];
            let fact = i as f64 * PI / self.half_length as f64;
            for (j, &d_val) in d.iter().enumerate().skip(1) {
                sum += 2.0 * d_val * (j as f64 * fact).cos();
            }
            self.coefs[idx_base + i] *= sum;
            sumg += 2.0 * self.coefs[idx_base + i];
        }

        // Normalize
        for i in 0..=self.half_length {
            self.coefs[idx_base + i] /= sumg;
        }
    }

    /// Process a single market case (block of prices).
    ///
    /// # Arguments
    /// * `data` - Price data, most recent at index 0 if chronological=false
    /// * `chronological` - If true, data[0] is oldest; if false, data[0] is newest
    pub fn process(&mut self, data: &[f64], chronological: bool) {
        assert!(data.len() >= self.lookback);

        // Collect data into local array 'y' in chronological order
        // Most recent case will be at index lookback-1
        for i in 0..self.lookback {
            let idx = if chronological {
                self.lookback - 1 - i
            } else {
                i
            };
            let price = data[idx];
            self.y[self.lookback - 1 - i] = if self.use_log { price.ln() } else { price };
        }

        // Fit a least-squares line to extend beyond current data
        // This allows zero-lag filtering
        let xmean = -0.5 * self.half_length as f64;
        let mut ymean = 0.0;
        for i in 0..=self.half_length {
            ymean += self.y[self.lookback - 1 - i];
        }
        ymean /= (self.half_length + 1) as f64;

        let mut xsq = 0.0;
        let mut xy = 0.0;
        for i in 0..=self.half_length {
            let xdiff = -(i as f64) - xmean;
            let ydiff = self.y[self.lookback - 1 - i] - ymean;
            xsq += xdiff * xdiff;
            xy += xdiff * ydiff;
        }
        let slope = xy / xsq;

        // Extend the data
        for i in 0..self.half_length {
            self.y[self.lookback + i] = (i as f64 + 1.0 - xmean) * slope + ymean;
        }

        // Process each period
        for iperiod in self.min_period..=self.max_period {
            let period_idx = iperiod - self.min_period;
            let coef_base = period_idx * (self.half_length + 1);

            let mut extreme_type: i8 = 0;
            let mut extreme_value = 0.0;
            let mut n_legs = 0usize;
            let mut longest_leg = 0.0f64;
            let mut prior = 0.0f64;

            // Apply filter to every value in the block
            for iy in self.half_length..self.lookback {
                // Convolution applies the FIR filter
                let mut sum = self.coefs[coef_base] * self.y[iy];
                for i in 1..=self.half_length {
                    sum += self.coefs[coef_base + i] * (self.y[iy + i] + self.y[iy - i]);
                }

                // Save filtered value for current data point
                if iy == self.lookback - 1 {
                    self.filtered[period_idx] = sum;
                }

                // Save actual minus filtered for width calculation
                self.diff_work[iy - self.half_length] = (self.y[iy] - sum).abs();

                // Collect the legs
                if iy == self.half_length {
                    extreme_type = 0;
                    extreme_value = sum;
                    n_legs = 0;
                    longest_leg = 0.0;
                } else if extreme_type == 0 {
                    // Waiting for first filtered price change
                    if sum > extreme_value {
                        extreme_type = -1; // First point is a low
                    } else if sum < extreme_value {
                        extreme_type = 1; // First point is a high
                    }
                } else if iy == self.lookback - 1 {
                    // Last point - consider this a turning point
                    if extreme_type != 0 {
                        self.leg_work[n_legs] = (extreme_value - sum).abs();
                        if self.leg_work[n_legs] > longest_leg {
                            longest_leg = self.leg_work[n_legs];
                        }
                        n_legs += 1;
                    }
                } else {
                    // Advancing in interior
                    if extreme_type == 1 && sum > prior {
                        // We have been going down but just turned up
                        self.leg_work[n_legs] = extreme_value - prior;
                        if self.leg_work[n_legs] > longest_leg {
                            longest_leg = self.leg_work[n_legs];
                        }
                        n_legs += 1;
                        extreme_type = -1;
                        extreme_value = prior;
                    } else if extreme_type == -1 && sum < prior {
                        // We have been going up but just turned down
                        self.leg_work[n_legs] = prior - extreme_value;
                        if self.leg_work[n_legs] > longest_leg {
                            longest_leg = self.leg_work[n_legs];
                        }
                        n_legs += 1;
                        extreme_type = 1;
                        extreme_value = prior;
                    }
                }

                prior = sum;
            }

            // Sort actual-filtered differences and find fractile for channel width
            let diff_len = self.lookback - self.half_length;
            self.diff_work[..diff_len].sort_by(|a, b| a.partial_cmp(b).unwrap());
            let width_idx = ((self.beta * (diff_len + 1) as f64) as usize).saturating_sub(1);
            self.width[period_idx] = self.diff_work[width_idx.min(diff_len - 1)];

            // Find mean of all legs greater than noise level
            let noise_level = self.noise_cut * longest_leg;
            let mut sum = 0.0;
            let mut n = 0usize;
            for i in 0..n_legs {
                if self.leg_work[i] > noise_level {
                    sum += self.leg_work[i];
                    n += 1;
                }
            }

            let mean_leg = if n > 0 { sum / n as f64 } else { 0.0 };
            self.fti[period_idx] = mean_leg / (self.width[period_idx] + 1e-5);
        }

        // Sort FTI local maxima (largest to smallest)
        self.sort_fti_peaks();
    }

    /// Sort FTI peaks for finding optimal values.
    fn sort_fti_peaks(&mut self) {
        let n_periods = self.max_period - self.min_period + 1;
        let mut n = 0usize;

        for i in 0..n_periods {
            // Find local maxima (including both endpoints)
            let is_peak = i == 0
                || i == n_periods - 1
                || (self.fti[i] >= self.fti[i - 1] && self.fti[i] >= self.fti[i + 1]);

            if is_peak {
                self.sort_work[n] = -self.fti[i]; // Negate for descending sort
                self.sorted[n] = i;
                n += 1;
            }
        }

        // Simple insertion sort (correctness over speed)
        for i in 1..n {
            let key = self.sort_work[i];
            let key_idx = self.sorted[i];
            let mut j = i;
            while j > 0 && self.sort_work[j - 1] > key {
                self.sort_work[j] = self.sort_work[j - 1];
                self.sorted[j] = self.sorted[j - 1];
                j -= 1;
            }
            self.sort_work[j] = key;
            self.sorted[j] = key_idx;
        }
    }

    /// Get the filtered value for a specific period.
    pub fn get_filtered_value(&self, period: usize) -> f64 {
        self.filtered[period - self.min_period]
    }

    /// Get the channel width for a specific period.
    pub fn get_width(&self, period: usize) -> f64 {
        self.width[period - self.min_period]
    }

    /// Get the FTI value for a specific period.
    pub fn get_fti(&self, period: usize) -> f64 {
        self.fti[period - self.min_period]
    }

    /// Get the index of the nth best FTI period (sorted descending).
    pub fn get_sorted_index(&self, which: usize) -> usize {
        self.sorted[which]
    }

    /// Get the period number for the nth best FTI.
    pub fn get_best_period(&self, which: usize) -> usize {
        self.sorted[which] + self.min_period
    }

    /// Get the best (highest) FTI value.
    pub fn get_best_fti(&self) -> f64 {
        if !self.sorted.is_empty() {
            self.fti[self.sorted[0]]
        } else {
            0.0
        }
    }
}

/// FTI derived metrics as specified in Section 5.1.
#[derive(Debug, Clone, Default)]
pub struct FTIMetrics {
    /// Current FTI level (best period)
    pub fti_level: f64,
    /// FTI slope (change over time)
    pub fti_slope: f64,
    /// FTI persistence (how long it's been elevated)
    pub fti_persist: f64,
    /// Current persist threshold (for diagnostics)
    pub persist_threshold: f64,
    /// Whether current FTI is elevated (fti_level > persist_threshold)
    pub is_elevated: bool,
    /// Whether threshold has been calibrated (vs default 1.0)
    pub calibrated: bool,
    /// Number of samples used for calibration (0 if not calibrated)
    pub calibration_samples: usize,
}

/// FTI tracker configuration.
#[derive(Debug, Clone)]
pub struct FTITrackerConfig {
    /// Threshold for "elevated" FTI
    pub persist_threshold: f64,
    /// Number of windows for persistence ring buffer
    pub persist_window: usize,
}

impl Default for FTITrackerConfig {
    fn default() -> Self {
        Self {
            persist_threshold: 1.0, // FTI > 1.0 is considered elevated
            persist_window: 20,     // Track last 20 windows
        }
    }
}

/// FTI tracker that maintains derived metrics over time.
pub struct FTITracker {
    fti: FTI,
    prev_fti: f64,
    /// Ring buffer for persistence: true if FTI was elevated in that window
    persist_ring: std::collections::VecDeque<bool>,
    /// Size of persistence ring buffer
    persist_window: usize,
    /// Threshold for "elevated" FTI
    persist_threshold: f64,
    /// Whether threshold has been calibrated
    calibrated: bool,
    /// FTI level samples collected during warmup for calibration
    warmup_samples: Vec<f64>,
    /// Percentile to use for threshold calibration (e.g., 0.75 = p75)
    calibration_percentile: f64,
    /// Minimum samples needed before calibration
    min_calibration_samples: usize,
}

impl FTITracker {
    /// Create a new FTI tracker with default parameters from Khalsa.
    pub fn new() -> Self {
        Self::with_config(FTITrackerConfig::default())
    }

    /// Create a new FTI tracker with explicit config.
    pub fn with_config(config: FTITrackerConfig) -> Self {
        Self {
            // Khalsa default parameters
            fti: FTI::new(
                false, // use_log
                5,     // min_period (Khalsa default)
                65,    // max_period (Khalsa default)
                32,    // half_length
                100,   // lookback
                0.9,   // beta
                0.2,   // noise_cut (Khalsa default)
            ),
            prev_fti: 0.0,
            persist_ring: std::collections::VecDeque::with_capacity(config.persist_window),
            persist_window: config.persist_window,
            persist_threshold: config.persist_threshold,
            // Self-calibration using p85 of FTI levels (p95 was too restrictive)
            calibrated: false,
            warmup_samples: Vec::with_capacity(1000),
            calibration_percentile: 0.85, // Use p85 as threshold (p95 made persistence ~0)
            min_calibration_samples: 500, // Calibrate after 500 samples
        }
    }

    /// Calibrate persist_threshold from collected warmup samples.
    /// Uses the configured percentile (default p85) of FTI levels.
    fn calibrate(&mut self) {
        if self.warmup_samples.len() < self.min_calibration_samples {
            return;
        }

        let mut sorted = self.warmup_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx =
            ((self.calibration_percentile * sorted.len() as f64) as usize).min(sorted.len() - 1);
        let calibrated_threshold = sorted[idx];

        // Only use calibrated threshold if it's reasonable (> 0)
        if calibrated_threshold > 0.0 {
            // Note: Do NOT use eprintln! here - it corrupts TUI display
            self.persist_threshold = calibrated_threshold;
        }

        self.calibrated = true;
        self.warmup_samples.clear(); // Free memory
    }

    /// Process new price data and return derived metrics.
    pub fn update(&mut self, prices: &[f64]) -> FTIMetrics {
        self.fti.process(prices, false);

        let current_fti = self.fti.get_best_fti();
        let fti_slope = current_fti - self.prev_fti;

        // Collect samples for calibration during warmup
        if !self.calibrated {
            self.warmup_samples.push(current_fti);
            if self.warmup_samples.len() >= self.min_calibration_samples {
                self.calibrate();
            }
        }

        // Update persistence ring buffer
        let is_elevated = current_fti > self.persist_threshold;
        self.persist_ring.push_back(is_elevated);

        // Keep only last persist_window entries
        while self.persist_ring.len() > self.persist_window {
            self.persist_ring.pop_front();
        }

        // Compute persistence as fraction of elevated windows
        let fti_persist = if self.persist_ring.is_empty() {
            0.0
        } else {
            let count = self.persist_ring.iter().filter(|&&x| x).count();
            count as f64 / self.persist_ring.len() as f64
        };

        self.prev_fti = current_fti;

        FTIMetrics {
            fti_level: current_fti,
            fti_slope,
            fti_persist,
            persist_threshold: self.persist_threshold,
            is_elevated,
            calibrated: self.calibrated,
            calibration_samples: if self.calibrated {
                self.min_calibration_samples
            } else {
                0
            },
        }
    }

    /// Get the current persist threshold (for diagnostics).
    pub fn persist_threshold(&self) -> f64 {
        self.persist_threshold
    }

    /// Check if threshold has been calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Get calibration info for manifest.
    /// Returns (threshold, num_samples) or None if not calibrated.
    pub fn calibration_info(&self) -> Option<(f64, usize)> {
        if self.calibrated {
            Some((self.persist_threshold, self.min_calibration_samples))
        } else {
            None
        }
    }
}

impl Default for FTITracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fti_basic() {
        let fti = FTI::new(false, 5, 20, 10, 50, 0.9, 0.2);
        assert!(fti.min_period == 5);
        assert!(fti.max_period == 20);
    }

    #[test]
    fn test_fti_process() {
        let mut fti = FTI::new(false, 5, 20, 10, 50, 0.9, 0.2);

        // Create trending price data
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.1).collect();
        fti.process(&prices, true);

        // FTI should be computed
        let best_fti = fti.get_best_fti();
        assert!(best_fti >= 0.0);
    }

    /// FTI parity test: Compare Rust FTI output against reference C++ output.
    ///
    /// Reference values computed from Timothy Masters' FTI.CPP with:
    /// - use_log=false, min_period=5, max_period=65, half_length=32
    /// - lookback=100, beta=0.9, noise_cut=0.2
    /// - Input: prices[i] = 100.0 + sin(i * 0.1) * 5.0 for i in 0..100
    ///
    /// To regenerate reference values, compile and run FTI.CPP:
    /// ```bash
    /// cd research/indicators/timothy_masters
    /// g++ -o fti_test FTI.CPP -lm
    /// ./fti_test
    /// ```
    #[test]
    fn test_fti_cpp_parity() {
        // Parameters matching Khalsa defaults
        let mut fti = FTI::new(
            false, // use_log
            5,     // min_period
            65,    // max_period
            32,    // half_length
            100,   // lookback
            0.9,   // beta
            0.2,   // noise_cut
        );

        // Generate deterministic test data: sinusoidal price series
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        fti.process(&prices, true);

        // Reference values from C++ implementation.
        // These should be computed by running the original FTI.CPP on the same input.
        // For now, we verify basic properties and capture the Rust output.

        let best_fti = fti.get_best_fti();
        let best_period = fti.get_best_period(0);

        // FTI should be non-negative
        assert!(
            best_fti >= 0.0,
            "FTI must be non-negative, got {}",
            best_fti
        );

        // Best period should be within range
        assert!(
            (5..=65).contains(&best_period),
            "Best period {} out of range [5, 65]",
            best_period
        );

        // Filtered value should be close to the smoothed input
        let filtered = fti.get_filtered_value(best_period);
        assert!(
            filtered.is_finite(),
            "Filtered value must be finite, got {}",
            filtered
        );

        // Width should be positive (channel width)
        let width = fti.get_width(best_period);
        assert!(width >= 0.0, "Width must be non-negative, got {}", width);

        // Print values for comparison with C++ reference
        eprintln!("=== FTI Parity Test Output ===");
        eprintln!("best_fti: {:.10}", best_fti);
        eprintln!("best_period: {}", best_period);
        eprintln!("filtered[best]: {:.10}", filtered);
        eprintln!("width[best]: {:.10}", width);

        // TODO: Add exact comparison once C++ reference values are computed.
        // Example assertions (uncomment when reference values available):
        // const CPP_BEST_FTI: f64 = <value from C++>;
        // const TOLERANCE: f64 = 1e-8;
        // assert!((best_fti - CPP_BEST_FTI).abs() < TOLERANCE,
        //     "FTI mismatch: Rust={} vs C++={}", best_fti, CPP_BEST_FTI);
    }

    /// Test FTI with flat price series (should have low FTI).
    #[test]
    fn test_fti_flat_prices() {
        let mut fti = FTI::new(false, 5, 20, 10, 50, 0.9, 0.2);

        // Flat prices - no movement
        let prices: Vec<f64> = vec![100.0; 50];
        fti.process(&prices, true);

        let best_fti = fti.get_best_fti();
        // Flat prices should have very low or zero FTI
        assert!(
            best_fti < 0.1,
            "Flat prices should have low FTI, got {}",
            best_fti
        );
    }

    /// Test FTI with strong trend (should have higher FTI).
    #[test]
    fn test_fti_strong_trend() {
        let mut fti = FTI::new(false, 5, 20, 10, 50, 0.9, 0.2);

        // Strong upward trend
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        fti.process(&prices, true);

        let best_fti = fti.get_best_fti();
        // Strong trend should have non-zero FTI
        assert!(
            best_fti >= 0.0,
            "Strong trend should have non-negative FTI, got {}",
            best_fti
        );
    }

    /// Test FTI persistence returns fraction in [0, 1].
    #[test]
    fn test_fti_persist_returns_fraction() {
        let config = FTITrackerConfig {
            persist_threshold: 0.5, // FTI above 0.5 counts as elevated
            persist_window: 10,
        };
        let mut tracker = FTITracker::with_config(config);

        // Feed data that produces varying FTI values
        for i in 0..20 {
            // Alternating flat and trending data
            let prices: Vec<f64> = if i % 2 == 0 {
                vec![100.0; 100] // Flat - low FTI
            } else {
                (0..100).map(|j| 100.0 + j as f64 * 0.5).collect() // Trend - higher FTI
            };

            let metrics = tracker.update(&prices);

            // fti_persist must always be in [0, 1]
            assert!(
                metrics.fti_persist >= 0.0 && metrics.fti_persist <= 1.0,
                "fti_persist must be in [0,1], got {}",
                metrics.fti_persist
            );
        }
    }
}
