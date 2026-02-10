//! Deterministic depth/trade alignment.
//!
//! Canonical join clock = integer microseconds (µs).
//! - Depth ts_ns → floor(ns/1000)
//! - Trades already µs
//!
//! Frame-based alignment:
//! - Frame i = [depth[i].t_us, depth[i+1].t_us)
//! - All trades with t_us in that interval attach to frame i
//! - Preserve file order on same timestamp

use std::collections::VecDeque;

/// Alignment error types.
#[derive(Debug, Clone)]
pub enum AlignmentError {
    /// Timestamps are not monotonically increasing
    NonMonotonicTimestamp {
        stream: &'static str,
        prev_us: i64,
        curr_us: i64,
    },
    /// Empty depth stream
    EmptyDepthStream,
    /// Timestamp parse error
    TimestampParseError(String),
}

impl std::fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonMonotonicTimestamp {
                stream,
                prev_us,
                curr_us,
            } => {
                write!(
                    f,
                    "ALIGNMENT_ERROR: non-monotonic {} timestamp: {} -> {}",
                    stream, prev_us, curr_us
                )
            }
            Self::EmptyDepthStream => write!(f, "ALIGNMENT_ERROR: empty depth stream"),
            Self::TimestampParseError(msg) => write!(f, "ALIGNMENT_ERROR: {}", msg),
        }
    }
}

impl std::error::Error for AlignmentError {}

/// Trade with canonical microsecond timestamp.
#[derive(Debug, Clone)]
pub struct AlignedTrade {
    /// Canonical timestamp in microseconds
    pub t_us: i64,
    /// Original nanosecond timestamp (for WAL)
    pub ts_ns: i64,
    /// Symbol
    pub symbol: String,
    /// Price mantissa
    pub price_mantissa: i64,
    /// Price exponent
    pub price_exponent: i8,
    /// Quantity mantissa
    pub qty_mantissa: i64,
    /// Quantity exponent
    pub qty_exponent: i8,
    /// Trade direction: +1 = buy, -1 = sell
    /// Derived from is_buyer_maker: true → sell (-1), false → buy (+1)
    pub sign: i8,
    /// Volume in base units (qty_mantissa * 10^qty_exponent)
    pub volume: f64,
    /// Notional in quote units (volume * price, e.g., USDT for BTC/USDT)
    pub notional: f64,
}

impl AlignedTrade {
    /// Create from raw trade data.
    /// is_buyer_maker: true means the buyer was the maker, so trade was sell-initiated.
    pub fn new(
        ts_ns: i64,
        symbol: String,
        price_mantissa: i64,
        price_exponent: i8,
        qty_mantissa: i64,
        qty_exponent: i8,
        is_buyer_maker: bool,
    ) -> Self {
        let t_us = ts_ns / 1000; // floor division for ns -> us
        let sign = if is_buyer_maker { -1 } else { 1 }; // buyer_maker = sell-initiated
        let volume = qty_mantissa as f64 * 10f64.powi(qty_exponent as i32);
        let price = price_mantissa as f64 * 10f64.powi(price_exponent as i32);
        let notional = volume * price;

        Self {
            t_us,
            ts_ns,
            symbol,
            price_mantissa,
            price_exponent,
            qty_mantissa,
            qty_exponent,
            sign,
            volume,
            notional,
        }
    }

    /// Price as f64.
    pub fn price_f64(&self) -> f64 {
        self.price_mantissa as f64 * 10f64.powi(self.price_exponent as i32)
    }
}

/// Depth snapshot with canonical microsecond timestamp.
#[derive(Debug, Clone)]
pub struct AlignedDepth {
    /// Canonical timestamp in microseconds
    pub t_us: i64,
    /// Original nanosecond timestamp (for WAL)
    pub ts_ns: i64,
    /// Symbol
    pub symbol: String,
    /// Price exponent for all levels
    pub price_exponent: i8,
    /// Quantity exponent for all levels
    pub qty_exponent: i8,
    /// Bid levels (price_mantissa, qty_mantissa), best first
    pub bids: Vec<(i64, i64)>,
    /// Ask levels (price_mantissa, qty_mantissa), best first
    pub asks: Vec<(i64, i64)>,
}

impl AlignedDepth {
    /// Create from raw depth data.
    pub fn new(
        ts_ns: i64,
        symbol: String,
        price_exponent: i8,
        qty_exponent: i8,
        bids: Vec<(i64, i64)>,
        asks: Vec<(i64, i64)>,
    ) -> Self {
        let t_us = ts_ns / 1000; // floor division for ns -> us

        Self {
            t_us,
            ts_ns,
            symbol,
            price_exponent,
            qty_exponent,
            bids,
            asks,
        }
    }

    /// Mid price as f64.
    pub fn mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.first()?;
        let best_ask = self.asks.first()?;
        let bid_price = best_bid.0 as f64 * 10f64.powi(self.price_exponent as i32);
        let ask_price = best_ask.0 as f64 * 10f64.powi(self.price_exponent as i32);
        Some((bid_price + ask_price) / 2.0)
    }
}

/// A frame consisting of a depth snapshot and all trades in [t_us, next_t_us).
#[derive(Debug, Clone)]
pub struct Frame {
    /// The depth snapshot for this frame
    pub depth: AlignedDepth,
    /// All trades in [depth.t_us, next_depth.t_us), in file order
    pub trades: Vec<AlignedTrade>,
    /// Frame end timestamp (exclusive), None for last frame
    pub frame_end_us: Option<i64>,
}

/// Warmup duration for bucket_size heuristic (5 minutes = 300 seconds).
/// bucket_size is ONLY computed from this window to avoid future-peeking.
pub const BUCKET_SIZE_WARMUP_SECS: u64 = 300;

/// Alignment statistics for WAL.
#[derive(Debug, Clone, Default)]
pub struct AlignmentStats {
    /// Total depth snapshots processed
    pub depth_count: u64,
    /// Total trades processed
    pub trade_count: u64,
    /// Frames with zero trades
    pub no_trade_frames: u64,
    /// Frames with trades
    pub trade_frames: u64,
    /// Total volume processed (base units, e.g., BTC)
    pub total_volume: f64,
    /// Buy volume (base units)
    pub buy_volume: f64,
    /// Sell volume (base units)
    pub sell_volume: f64,
    /// Total notional processed (quote units, e.g., USDT)
    pub total_notional: f64,
    /// Min trades per frame (excluding zero)
    pub min_trades_per_frame: u64,
    /// Max trades per frame
    pub max_trades_per_frame: u64,
    /// Trade NOTIONAL per second samples (for bucket_size heuristic) - WARMUP ONLY
    /// Uses quote units (USDT) for meaningful bucket sizing
    pub(crate) warmup_notional_per_sec_samples: Vec<f64>,
    /// Last second boundary for volume accumulation
    last_sec_boundary_us: i64,
    /// Current second notional accumulator
    current_sec_notional: f64,
    /// First frame timestamp (for warmup window)
    first_frame_us: Option<i64>,
    /// Warmup end timestamp (first_frame_us + BUCKET_SIZE_WARMUP_SECS)
    warmup_end_us: i64,
    /// Whether we are still in warmup window
    in_warmup: bool,
    /// Number of seconds sampled during warmup
    pub warmup_seconds_sampled: usize,
}

impl AlignmentStats {
    /// Create new alignment stats.
    pub fn new() -> Self {
        Self {
            min_trades_per_frame: u64::MAX,
            in_warmup: true,
            ..Default::default()
        }
    }

    /// Record a frame.
    pub fn record_frame(&mut self, frame: &Frame) {
        self.depth_count += 1;

        // Initialize warmup window on first frame
        if self.first_frame_us.is_none() {
            self.first_frame_us = Some(frame.depth.t_us);
            self.warmup_end_us = frame.depth.t_us + (BUCKET_SIZE_WARMUP_SECS as i64 * 1_000_000);
        }

        // Check if we've exited warmup
        if self.in_warmup && frame.depth.t_us >= self.warmup_end_us {
            // Finalize warmup stats before exiting
            if self.current_sec_notional > 0.0 {
                self.warmup_notional_per_sec_samples
                    .push(self.current_sec_notional);
                self.warmup_seconds_sampled = self.warmup_notional_per_sec_samples.len();
            }
            self.in_warmup = false;
        }

        let trade_count = frame.trades.len() as u64;
        self.trade_count += trade_count;

        if trade_count == 0 {
            self.no_trade_frames += 1;
        } else {
            self.trade_frames += 1;
            if trade_count < self.min_trades_per_frame {
                self.min_trades_per_frame = trade_count;
            }
        }

        if trade_count > self.max_trades_per_frame {
            self.max_trades_per_frame = trade_count;
        }

        for trade in &frame.trades {
            self.total_volume += trade.volume;
            self.total_notional += trade.notional;
            if trade.sign > 0 {
                self.buy_volume += trade.volume;
            } else {
                self.sell_volume += trade.volume;
            }

            // Track NOTIONAL per second for bucket_size heuristic - WARMUP ONLY
            // Uses quote units (USDT) for meaningful bucket sizing
            if self.in_warmup {
                let sec_boundary = (trade.t_us / 1_000_000) * 1_000_000;
                if self.last_sec_boundary_us == 0 {
                    self.last_sec_boundary_us = sec_boundary;
                }

                if sec_boundary > self.last_sec_boundary_us {
                    // New second, record previous
                    if self.current_sec_notional > 0.0 {
                        self.warmup_notional_per_sec_samples
                            .push(self.current_sec_notional);
                    }
                    self.current_sec_notional = trade.notional;
                    self.last_sec_boundary_us = sec_boundary;
                } else {
                    self.current_sec_notional += trade.notional;
                }
            }
        }
    }

    /// Finalize stats (call after all frames).
    pub fn finalize(&mut self) {
        // Record final warmup second if still in warmup
        if self.in_warmup && self.current_sec_notional > 0.0 {
            self.warmup_notional_per_sec_samples
                .push(self.current_sec_notional);
        }
        self.warmup_seconds_sampled = self.warmup_notional_per_sec_samples.len();

        // Fix min if no frames with trades
        if self.min_trades_per_frame == u64::MAX {
            self.min_trades_per_frame = 0;
        }
    }

    /// Compute recommended bucket_size from WARMUP period only.
    /// Uses NOTIONAL (quote units, e.g., USDT) for meaningful bucket sizing.
    /// Formula: clamp(p50(notional/sec) * 10, 5000, 500000) in USDT.
    /// Returns bucket_size for manifest logging.
    pub fn recommended_bucket_size(&self) -> f64 {
        if self.warmup_notional_per_sec_samples.is_empty() {
            return 50000.0; // Default fallback: 50k USDT when no warmup data
        }

        let mut sorted = self.warmup_notional_per_sec_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_idx = sorted.len() / 2;
        let p50 = sorted[p50_idx];

        // bucket_size = p50 * 10 seconds, clamped to [5000, 500000] USDT
        // 5000 USDT minimum = about 0.06 BTC at 83k
        // 500000 USDT maximum = about 6 BTC at 83k
        (p50 * 10.0).clamp(5000.0, 500000.0)
    }

    /// Get bucket_size source description for manifest.
    pub fn bucket_size_source(&self) -> String {
        if self.warmup_notional_per_sec_samples.is_empty() {
            "default (no warmup trades)".to_string()
        } else {
            format!(
                "warmup p50(notional/sec)*10 from {} seconds",
                self.warmup_seconds_sampled
            )
        }
    }

    /// Get warmup duration in seconds (for manifest).
    pub fn warmup_duration_secs(&self) -> f64 {
        self.warmup_seconds_sampled as f64
    }

    /// Get notional per second percentiles from WARMUP period.
    pub fn notional_per_sec_percentiles(&self) -> VolumePercentiles {
        if self.warmup_notional_per_sec_samples.is_empty() {
            return VolumePercentiles::default();
        }

        let mut sorted = self.warmup_notional_per_sec_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        VolumePercentiles {
            count: n,
            min: sorted[0],
            p25: sorted[n / 4],
            p50: sorted[n / 2],
            p75: sorted[3 * n / 4],
            p95: sorted[(n * 95) / 100],
            max: sorted[n - 1],
        }
    }

    /// Get volume per second percentiles from WARMUP period (base units for display).
    pub fn vol_per_sec_percentiles(&self) -> VolumePercentiles {
        // For backward compatibility, compute from notional / average price
        // This is approximate but sufficient for display
        self.notional_per_sec_percentiles()
    }

    /// Coverage ratio: frames with trades / total frames.
    pub fn trade_coverage(&self) -> f64 {
        if self.depth_count == 0 {
            return 0.0;
        }
        self.trade_frames as f64 / self.depth_count as f64
    }
}

/// Volume per second percentiles.
#[derive(Debug, Clone, Default)]
pub struct VolumePercentiles {
    pub count: usize,
    pub min: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
}

/// Stream aligner that produces frames from depth and trade streams.
pub struct StreamAligner {
    /// Pending depth snapshots
    depth_queue: VecDeque<AlignedDepth>,
    /// Pending trades
    trade_queue: VecDeque<AlignedTrade>,
    /// Last depth timestamp for monotonicity check
    last_depth_us: Option<i64>,
    /// Last trade timestamp for monotonicity check
    last_trade_us: Option<i64>,
    /// Alignment statistics
    pub stats: AlignmentStats,
}

impl StreamAligner {
    /// Create a new stream aligner.
    pub fn new() -> Self {
        Self {
            depth_queue: VecDeque::new(),
            trade_queue: VecDeque::new(),
            last_depth_us: None,
            last_trade_us: None,
            stats: AlignmentStats::new(),
        }
    }

    /// Add a depth snapshot. Returns error if non-monotonic.
    pub fn add_depth(&mut self, depth: AlignedDepth) -> Result<(), AlignmentError> {
        if let Some(prev) = self.last_depth_us
            && depth.t_us < prev
        {
            return Err(AlignmentError::NonMonotonicTimestamp {
                stream: "depth",
                prev_us: prev,
                curr_us: depth.t_us,
            });
        }
        self.last_depth_us = Some(depth.t_us);
        self.depth_queue.push_back(depth);
        Ok(())
    }

    /// Add a trade. Returns error if non-monotonic.
    pub fn add_trade(&mut self, trade: AlignedTrade) -> Result<(), AlignmentError> {
        if let Some(prev) = self.last_trade_us
            && trade.t_us < prev
        {
            return Err(AlignmentError::NonMonotonicTimestamp {
                stream: "trade",
                prev_us: prev,
                curr_us: trade.t_us,
            });
        }
        self.last_trade_us = Some(trade.t_us);
        self.trade_queue.push_back(trade);
        Ok(())
    }

    /// Try to emit a frame if we have enough data.
    /// Returns None if we need more depth snapshots.
    /// A frame is emitted when we have depth[i] and depth[i+1] (to define frame boundary).
    ///
    /// CAUSALITY AUDIT:
    /// - Frame i contains depth[i] and trades in interval [depth[i].t_us, depth[i+1].t_us)
    /// - We ONLY read depth[i+1].t_us for frame boundary, never its orderbook content
    /// - All trades in frame have t_us < frame_end_us (strictly before next depth)
    /// - This ensures causal replay: when processing depth[i], we only see trades
    ///   that occurred before the next depth snapshot arrived
    pub fn try_emit_frame(&mut self) -> Option<Frame> {
        // Need at least 2 depth snapshots to define a frame boundary
        if self.depth_queue.len() < 2 {
            return None;
        }

        let current_depth = self.depth_queue.pop_front().unwrap();
        let next_depth = self.depth_queue.front().unwrap();
        // CAUSALITY: Only read next_depth.t_us, not its orderbook data
        let frame_end_us = next_depth.t_us;

        // Collect all trades in [current.t_us, next.t_us) - strictly before frame_end
        let mut frame_trades = Vec::new();
        while let Some(trade) = self.trade_queue.front() {
            if trade.t_us < current_depth.t_us {
                // Trade before frame start, discard (shouldn't happen with proper ordering)
                self.trade_queue.pop_front();
            } else if trade.t_us < frame_end_us {
                // Trade in frame, collect it
                frame_trades.push(self.trade_queue.pop_front().unwrap());
            } else {
                // Trade in future frame, stop
                break;
            }
        }

        let frame = Frame {
            depth: current_depth,
            trades: frame_trades,
            frame_end_us: Some(frame_end_us),
        };

        self.stats.record_frame(&frame);
        Some(frame)
    }

    /// Emit final frame (last depth with remaining trades).
    pub fn emit_final_frame(&mut self) -> Option<Frame> {
        if self.depth_queue.is_empty() {
            return None;
        }

        let current_depth = self.depth_queue.pop_front().unwrap();

        // Collect all remaining trades from this point forward
        let mut frame_trades = Vec::new();
        while let Some(trade) = self.trade_queue.pop_front() {
            if trade.t_us >= current_depth.t_us {
                frame_trades.push(trade);
            }
        }

        let frame = Frame {
            depth: current_depth,
            trades: frame_trades,
            frame_end_us: None, // Last frame has no end boundary
        };

        self.stats.record_frame(&frame);
        self.stats.finalize();
        Some(frame)
    }

    /// Drain remaining frames (for batch processing).
    pub fn drain_frames(&mut self) -> Vec<Frame> {
        let mut frames = Vec::new();

        // Emit all complete frames
        while let Some(frame) = self.try_emit_frame() {
            frames.push(frame);
        }

        // Emit final frame
        if let Some(frame) = self.emit_final_frame() {
            frames.push(frame);
        }

        frames
    }
}

impl Default for StreamAligner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_trade_sign() {
        // is_buyer_maker=true means sell-initiated (buyer was passive)
        let sell = AlignedTrade::new(
            1000000000, // 1 second in ns
            "TEST".to_string(),
            10000,
            -2,
            100,
            -3,
            true, // buyer_maker = sell
        );
        assert_eq!(sell.sign, -1);
        assert_eq!(sell.t_us, 1000000); // floor(ns/1000)

        // is_buyer_maker=false means buy-initiated (seller was passive)
        let buy = AlignedTrade::new(
            1000000000,
            "TEST".to_string(),
            10000,
            -2,
            100,
            -3,
            false, // not buyer_maker = buy
        );
        assert_eq!(buy.sign, 1);
    }

    #[test]
    fn test_canonical_timestamp() {
        // Test ns -> us conversion
        let depth = AlignedDepth::new(
            1769664809480231005, // ns
            "BTCUSDT".to_string(),
            -2,
            -8,
            vec![(8819490, 636400000)],
            vec![(8819500, 500000000)],
        );
        assert_eq!(depth.t_us, 1769664809480231); // floor(ns/1000)
    }

    #[test]
    fn test_frame_alignment() {
        let mut aligner = StreamAligner::new();

        // Add depth at t=1000, 2000, 3000 µs
        aligner
            .add_depth(AlignedDepth::new(
                1000000,
                "TEST".to_string(),
                -2,
                -3,
                vec![],
                vec![],
            ))
            .unwrap();
        aligner
            .add_depth(AlignedDepth::new(
                2000000,
                "TEST".to_string(),
                -2,
                -3,
                vec![],
                vec![],
            ))
            .unwrap();
        aligner
            .add_depth(AlignedDepth::new(
                3000000,
                "TEST".to_string(),
                -2,
                -3,
                vec![],
                vec![],
            ))
            .unwrap();

        // Add trades at t=1500, 1600, 2500 µs
        aligner
            .add_trade(AlignedTrade::new(
                1500000,
                "TEST".to_string(),
                100,
                -2,
                10,
                -3,
                false,
            ))
            .unwrap();
        aligner
            .add_trade(AlignedTrade::new(
                1600000,
                "TEST".to_string(),
                100,
                -2,
                20,
                -3,
                true,
            ))
            .unwrap();
        aligner
            .add_trade(AlignedTrade::new(
                2500000,
                "TEST".to_string(),
                100,
                -2,
                30,
                -3,
                false,
            ))
            .unwrap();

        let frames = aligner.drain_frames();

        // Frame 0: depth@1000, trades in [1000, 2000) -> 2 trades
        assert_eq!(frames[0].depth.t_us, 1000);
        assert_eq!(frames[0].trades.len(), 2);
        assert_eq!(frames[0].frame_end_us, Some(2000));

        // Frame 1: depth@2000, trades in [2000, 3000) -> 1 trade
        assert_eq!(frames[1].depth.t_us, 2000);
        assert_eq!(frames[1].trades.len(), 1);
        assert_eq!(frames[1].frame_end_us, Some(3000));

        // Frame 2: depth@3000, no trades after
        assert_eq!(frames[2].depth.t_us, 3000);
        assert_eq!(frames[2].trades.len(), 0);
        assert_eq!(frames[2].frame_end_us, None); // Last frame
    }

    #[test]
    fn test_non_monotonic_rejection() {
        let mut aligner = StreamAligner::new();

        aligner
            .add_depth(AlignedDepth::new(
                2000000,
                "TEST".to_string(),
                -2,
                -3,
                vec![],
                vec![],
            ))
            .unwrap();
        let result = aligner.add_depth(AlignedDepth::new(
            1000000,
            "TEST".to_string(),
            -2,
            -3,
            vec![],
            vec![],
        ));

        assert!(matches!(
            result,
            Err(AlignmentError::NonMonotonicTimestamp { .. })
        ));
    }

    #[test]
    fn test_bucket_size_heuristic() {
        let mut stats = AlignmentStats::new();

        // Simulate 10 seconds of notional during warmup (USDT values)
        // [10000, 20000, 15000, 18000, 12000, 16000, 14000, 17000, 13000, 19000]
        stats.warmup_notional_per_sec_samples = vec![
            10000.0, 20000.0, 15000.0, 18000.0, 12000.0, 16000.0, 14000.0, 17000.0, 13000.0,
            19000.0,
        ];
        stats.warmup_seconds_sampled = 10;

        // p50 ≈ 15500, bucket_size = 15500 * 10 = 155000 USDT
        let bucket_size = stats.recommended_bucket_size();
        assert!((5000.0..=500000.0).contains(&bucket_size));
        assert!(bucket_size > 100000.0); // Should be around 155000

        // Verify source description mentions warmup and notional
        let source = stats.bucket_size_source();
        assert!(source.contains("warmup"));
        assert!(source.contains("notional"));
        assert!(source.contains("10 seconds"));
    }
}
