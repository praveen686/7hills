//! O(1) rolling statistics for basis z-score computation.
//!
//! `RingBuffer` maintains running sum and sum-of-squares for constant-time
//! mean and standard deviation. `BasisStats` wraps per-symbol tracking.

use std::collections::HashMap;

/// Fixed-size ring buffer with O(1) rolling mean and standard deviation.
///
/// Uses Welford-style incremental sum/sum_sq tracking. When full, the evicted
/// value is subtracted from running totals.
pub struct RingBuffer {
    buf: Vec<f64>,
    head: usize,
    len: usize,
    cap: usize,
    sum: f64,
    sum_sq: f64,
}

impl RingBuffer {
    /// Create a new ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            buf: vec![0.0; capacity],
            head: 0,
            len: 0,
            cap: capacity,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push a new value, evicting the oldest if full.
    pub fn push(&mut self, val: f64) {
        if self.len == self.cap {
            // Evict oldest
            let old = self.buf[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.len += 1;
        }
        self.buf[self.head] = val;
        self.sum += val;
        self.sum_sq += val * val;
        self.head = (self.head + 1) % self.cap;
    }

    /// Number of values currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is full (at capacity).
    pub fn is_full(&self) -> bool {
        self.len == self.cap
    }

    /// Rolling mean. Returns 0 if empty.
    pub fn mean(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.sum / self.len as f64
    }

    /// Rolling standard deviation (population). Returns 0 if fewer than 2 values.
    pub fn std(&self) -> f64 {
        if self.len < 2 {
            return 0.0;
        }
        let n = self.len as f64;
        let variance = (self.sum_sq / n) - (self.sum / n).powi(2);
        // Clamp to avoid negative variance from floating-point drift
        variance.max(0.0).sqrt()
    }

    /// Compute z-score for a given value. Returns 0 if std is too small.
    pub fn z_score(&self, val: f64) -> f64 {
        let s = self.std();
        if s < 1e-12 {
            return 0.0;
        }
        (val - self.mean()) / s
    }
}

/// Per-symbol basis statistics tracker.
pub struct BasisStats {
    buffers: HashMap<String, RingBuffer>,
    window_size: usize,
}

impl BasisStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            buffers: HashMap::new(),
            window_size,
        }
    }

    /// Push a basis observation for a symbol.
    pub fn push(&mut self, symbol: &str, basis_bps: f64) {
        let buf = self
            .buffers
            .entry(symbol.to_string())
            .or_insert_with(|| RingBuffer::new(self.window_size));
        buf.push(basis_bps);
    }

    /// Get rolling mean for a symbol.
    pub fn mean(&self, symbol: &str) -> f64 {
        self.buffers.get(symbol).map(|b| b.mean()).unwrap_or(0.0)
    }

    /// Get rolling std for a symbol.
    pub fn std(&self, symbol: &str) -> f64 {
        self.buffers.get(symbol).map(|b| b.std()).unwrap_or(0.0)
    }

    /// Compute z-score for a symbol given current basis.
    pub fn z_score(&self, symbol: &str, basis_bps: f64) -> f64 {
        self.buffers
            .get(symbol)
            .map(|b| b.z_score(basis_bps))
            .unwrap_or(0.0)
    }

    /// Whether a symbol has enough data for meaningful statistics.
    pub fn is_ready(&self, symbol: &str) -> bool {
        self.buffers
            .get(symbol)
            .map(|b| b.is_full())
            .unwrap_or(false)
    }

    /// Number of observations for a symbol.
    pub fn count(&self, symbol: &str) -> usize {
        self.buffers.get(symbol).map(|b| b.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_basic() {
        let mut rb = RingBuffer::new(3);
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.mean(), 0.0);
        assert_eq!(rb.std(), 0.0);

        rb.push(10.0);
        assert_eq!(rb.len(), 1);
        assert_eq!(rb.mean(), 10.0);
        assert_eq!(rb.std(), 0.0); // need 2 for std

        rb.push(20.0);
        assert_eq!(rb.len(), 2);
        assert!((rb.mean() - 15.0).abs() < 1e-10);

        rb.push(30.0);
        assert_eq!(rb.len(), 3);
        assert!(rb.is_full());
        assert!((rb.mean() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn ring_buffer_eviction() {
        let mut rb = RingBuffer::new(3);
        rb.push(10.0);
        rb.push(20.0);
        rb.push(30.0);
        // Mean = 20, buffer full

        rb.push(40.0); // evicts 10
        assert_eq!(rb.len(), 3);
        assert!((rb.mean() - 30.0).abs() < 1e-10); // (20+30+40)/3 = 30

        rb.push(50.0); // evicts 20
        assert!((rb.mean() - 40.0).abs() < 1e-10); // (30+40+50)/3 = 40
    }

    #[test]
    fn ring_buffer_std() {
        let mut rb = RingBuffer::new(4);
        // Push known values: 2, 4, 4, 4 → mean=3.5, variance = ((2-3.5)^2 + 3*(4-3.5)^2) / 4 = (2.25+0.75)/4 = 0.75
        for v in [2.0, 4.0, 4.0, 4.0] {
            rb.push(v);
        }
        assert!((rb.mean() - 3.5).abs() < 1e-10);
        let expected_std = 0.75_f64.sqrt(); // ~0.866
        assert!((rb.std() - expected_std).abs() < 1e-6);
    }

    #[test]
    fn ring_buffer_z_score() {
        let mut rb = RingBuffer::new(100);
        // Push constant values → std ≈ 0, z-score should be 0
        for _ in 0..100 {
            rb.push(5.0);
        }
        assert_eq!(rb.z_score(5.0), 0.0);
        assert_eq!(rb.z_score(10.0), 0.0); // std too small

        // Push known distribution
        let mut rb2 = RingBuffer::new(4);
        for v in [0.0, 0.0, 10.0, 10.0] {
            rb2.push(v);
        }
        let mean = rb2.mean(); // 5.0
        let std = rb2.std(); // 5.0
        assert!((mean - 5.0).abs() < 1e-10);
        assert!((std - 5.0).abs() < 1e-10);
        assert!((rb2.z_score(10.0) - 1.0).abs() < 1e-10);
        assert!((rb2.z_score(0.0) - -1.0).abs() < 1e-10);
    }

    #[test]
    fn basis_stats_multi_symbol() {
        let mut stats = BasisStats::new(3);
        stats.push("BTCUSDT", 5.0);
        stats.push("BTCUSDT", 5.0);
        stats.push("BTCUSDT", 5.0);
        stats.push("ETHUSDT", 10.0);

        assert!(stats.is_ready("BTCUSDT"));
        assert!(!stats.is_ready("ETHUSDT"));
        assert_eq!(stats.count("BTCUSDT"), 3);
        assert_eq!(stats.count("ETHUSDT"), 1);
        assert!((stats.mean("BTCUSDT") - 5.0).abs() < 1e-10);
        assert!((stats.mean("ETHUSDT") - 10.0).abs() < 1e-10);
    }
}
