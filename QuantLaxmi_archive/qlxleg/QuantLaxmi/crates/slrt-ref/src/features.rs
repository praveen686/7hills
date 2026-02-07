//! Feature computation exactly as specified in SLRT-GPU v1.1.
//!
//! Spec: Section 4.0 FEATURE ENGINEERING STACK

use crate::data::{OrderBook, Trade, TradeSide, TriState};
use crate::sealed::{EPSILON, VOLUME_EPS};

/// Snapshot features computed from a single order book state.
/// Spec: Section 4.1 Snapshot Features
#[derive(Debug, Clone, Default)]
pub struct SnapshotFeatures {
    /// Mid price: m = (a1 + b1) / 2
    pub mid: f64,
    /// Microprice: μ = (a1 * qb1 + b1 * qa1) / (qa1 + qb1 + ε)
    pub microprice: f64,
    /// Top-k Imbalance for k=10: Ik = (Σ q_bid - Σ q_ask) / (Σ q_bid + Σ q_ask + ε)
    pub imbalance_10: f64,
    /// Depth slope bid side
    pub depth_slope_bid: f64,
    /// Depth slope ask side
    pub depth_slope_ask: f64,
    /// Gap risk (sweep cost normalized by tick size)
    pub gap_risk: f64,
    /// Total bid depth (top 10 levels)
    pub total_bid_depth: f64,
    /// Total ask depth (top 10 levels)
    pub total_ask_depth: f64,
    /// Spread in ticks
    pub spread_ticks: f64,
}

impl SnapshotFeatures {
    /// Compute all snapshot features from an order book.
    /// Spec: Section 4.1
    pub fn compute(book: &OrderBook, tick_size: f64, sweep_size: f64) -> Self {
        let a1 = book.best_ask().unwrap_or(0.0);
        let b1 = book.best_bid().unwrap_or(0.0);
        let qa1 = book.best_ask_qty().unwrap_or(0.0);
        let qb1 = book.best_bid_qty().unwrap_or(0.0);

        // Mid: m = (a1 + b1) / 2
        let mid = (a1 + b1) / 2.0;

        // Microprice: μ = (a1 * qb1 + b1 * qa1) / (qa1 + qb1 + ε)
        let microprice = (a1 * qb1 + b1 * qa1) / (qa1 + qb1 + EPSILON);

        // Top-k Imbalance (k=10): Ik = (Σ q_bid - Σ q_ask) / (Σ q_bid + Σ q_ask + ε)
        let k = 10.min(book.bids.len()).min(book.asks.len());
        let sum_bid: f64 = book.bids.iter().take(k).map(|l| l.qty_f64()).sum();
        let sum_ask: f64 = book.asks.iter().take(k).map(|l| l.qty_f64()).sum();
        let imbalance_10 = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPSILON);

        // Depth Slope (per side): median_i (Δpi / (Qi + ε))
        let depth_slope_bid = Self::compute_depth_slope(&book.bids);
        let depth_slope_ask = Self::compute_depth_slope(&book.asks);

        // Gap Risk / Sweep Cost
        let gap_risk = Self::compute_gap_risk(book, microprice, tick_size, sweep_size);

        // Total depths
        let total_bid_depth: f64 = book.bids.iter().take(10).map(|l| l.qty_f64()).sum();
        let total_ask_depth: f64 = book.asks.iter().take(10).map(|l| l.qty_f64()).sum();

        // Spread in ticks
        let spread_ticks = if tick_size > EPSILON {
            (a1 - b1) / tick_size
        } else {
            0.0
        };

        Self {
            mid,
            microprice,
            imbalance_10,
            depth_slope_bid,
            depth_slope_ask,
            gap_risk,
            total_bid_depth,
            total_ask_depth,
            spread_ticks,
        }
    }

    /// Compute depth slope for one side of the book.
    /// Spec: Section 4.1 - Depth Slope (per side)
    /// For each level i: Qi = Σ q_j (j ≤ i), Δpi = |pi - p1|
    /// Slope = median_i (Δpi / (Qi + ε))
    fn compute_depth_slope(levels: &[crate::data::PriceLevel]) -> f64 {
        if levels.is_empty() {
            return 0.0;
        }

        let p1 = levels[0].price_f64();
        let mut ratios = Vec::with_capacity(levels.len());
        let mut cumulative_qty = 0.0;

        for level in levels.iter() {
            cumulative_qty += level.qty_f64();
            let delta_p = (level.price_f64() - p1).abs();
            let ratio = delta_p / (cumulative_qty + EPSILON);
            ratios.push(ratio);
        }

        // Compute median
        Self::median(&mut ratios)
    }

    /// Compute gap risk (sweep cost normalized).
    /// Spec: Section 4.1 - Sweep Cost / Gap Risk
    /// Walk book for target size Q*, compute VWAP psweep
    /// gapRisk = |psweep - μ| / tickSize
    fn compute_gap_risk(
        book: &OrderBook,
        microprice: f64,
        tick_size: f64,
        target_size: f64,
    ) -> f64 {
        // Walk the ask side for buy sweep
        let buy_vwap = Self::compute_sweep_vwap(&book.asks, target_size);
        // Walk the bid side for sell sweep
        let sell_vwap = Self::compute_sweep_vwap(&book.bids, target_size);

        // Use the worse of the two
        let buy_gap = (buy_vwap - microprice).abs();
        let sell_gap = (sell_vwap - microprice).abs();
        let max_gap = buy_gap.max(sell_gap);

        if tick_size > EPSILON {
            max_gap / tick_size
        } else {
            max_gap
        }
    }

    /// Compute VWAP for sweeping target_size through levels.
    fn compute_sweep_vwap(levels: &[crate::data::PriceLevel], target_size: f64) -> f64 {
        let mut remaining = target_size;
        let mut total_value = 0.0;
        let mut total_qty = 0.0;

        for level in levels.iter() {
            if remaining <= EPSILON {
                break;
            }
            let qty = level.qty_f64();
            let price = level.price_f64();
            let fill_qty = qty.min(remaining);

            total_value += fill_qty * price;
            total_qty += fill_qty;
            remaining -= fill_qty;
        }

        if total_qty > EPSILON {
            total_value / total_qty
        } else {
            levels.first().map(|l| l.price_f64()).unwrap_or(0.0)
        }
    }

    /// Compute median of a slice (modifies the slice by sorting).
    fn median(values: &mut [f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }
}

/// Trade-flow features computed over rolling windows.
/// Spec: Section 4.2 Trade-Flow Features (Rolling Windows)
#[derive(Debug, Clone, Default)]
pub struct TradeFlowFeatures {
    /// Signed volume over window: ΔV_signed = Σ(sign_t * volume_t)
    pub signed_volume: f64,
    /// Elasticity: ε_W = |Δm_W| / (|ΔV_signed_W| + ε)
    /// Set to 0.0 when volume is below VOLUME_EPS (undefined).
    pub elasticity: f64,
    /// True if elasticity is undefined (volume < VOLUME_EPS).
    /// Signals that confidence penalty should be applied.
    pub elasticity_undefined: bool,
    /// Depth collapse rate: Ddot_k,W = (Dk(t) - Dk(t-W)) / W
    pub depth_collapse_rate: f64,
}

/// Rolling window buffer for trade-flow feature computation.
pub struct TradeFlowAccumulator {
    /// Window size in nanoseconds
    window_ns: i64,
    /// Trades in the current window
    trades: Vec<(i64, f64, i8)>, // (ts_ns, qty, sign)
    /// Mid prices for elasticity
    mids: Vec<(i64, f64)>, // (ts_ns, mid)
    /// Depths for collapse rate
    depths: Vec<(i64, f64)>, // (ts_ns, depth_k)
}

impl TradeFlowAccumulator {
    /// Create a new accumulator with the specified window size in milliseconds.
    pub fn new(window_ms: u64) -> Self {
        Self {
            window_ns: window_ms as i64 * 1_000_000,
            trades: Vec::new(),
            mids: Vec::new(),
            depths: Vec::new(),
        }
    }

    /// Add a trade to the accumulator.
    /// Spec: Section 4.2 - Signed Volume Inference
    /// - Use aggressor side if available
    /// - Else tick rule with microprice refinement
    pub fn add_trade(&mut self, trade: &Trade, microprice: f64) {
        let sign = match &trade.side {
            TriState::Present(TradeSide::Buy) => 1i8,
            TriState::Present(TradeSide::Sell) => -1i8,
            _ => {
                // Tick rule with microprice refinement
                let price = trade.price_f64();
                if price > microprice {
                    1i8
                } else if price < microprice {
                    -1i8
                } else {
                    0i8
                }
            }
        };

        self.trades.push((trade.ts_ns, trade.qty_f64(), sign));
    }

    /// Add a mid price observation.
    pub fn add_mid(&mut self, ts_ns: i64, mid: f64) {
        self.mids.push((ts_ns, mid));
    }

    /// Add a depth observation (sum of top-k levels).
    pub fn add_depth(&mut self, ts_ns: i64, depth_k: f64) {
        self.depths.push((ts_ns, depth_k));
    }

    /// Evict old entries outside the window.
    fn evict(&mut self, current_ts_ns: i64) {
        let cutoff = current_ts_ns - self.window_ns;
        self.trades.retain(|(ts, _, _)| *ts >= cutoff);
        self.mids.retain(|(ts, _)| *ts >= cutoff);
        self.depths.retain(|(ts, _)| *ts >= cutoff);
    }

    /// Compute trade-flow features at the current timestamp.
    pub fn compute(&mut self, current_ts_ns: i64) -> TradeFlowFeatures {
        self.evict(current_ts_ns);

        // Signed Volume: ΔV_signed = Σ(sign_t * volume_t)
        let signed_volume: f64 = self
            .trades
            .iter()
            .map(|(_, qty, sign)| *qty * (*sign as f64))
            .sum();

        // Elasticity: ε_W = |Δm_W| / (|ΔV_signed_W| + ε)
        // When abs(signed_volume) < VOLUME_EPS, elasticity is undefined.
        // Set to 0.0 deterministically.
        //
        // AUDIT: Only flag elasticity_undefined when:
        // - There ARE trades in the window (trades is not empty)
        // - BUT the volume is too low (< VOLUME_EPS)
        // If there are no trades at all, elasticity is simply 0.0 without penalty.
        let has_trades = !self.trades.is_empty();
        let (elasticity, elasticity_undefined) = if self.mids.len() >= 2 {
            if signed_volume.abs() < VOLUME_EPS {
                // Volume too low - only undefined if we had trades but volume still low
                (0.0, has_trades)
            } else {
                let first_mid = self.mids.first().map(|(_, m)| *m).unwrap_or(0.0);
                let last_mid = self.mids.last().map(|(_, m)| *m).unwrap_or(0.0);
                let delta_mid = (last_mid - first_mid).abs();
                (delta_mid / signed_volume.abs(), false)
            }
        } else {
            (0.0, false)
        };

        // Depth Collapse Rate: Ddot_k,W = (Dk(t) - Dk(t-W)) / W
        let depth_collapse_rate = if self.depths.len() >= 2 {
            let first_depth = self.depths.first().map(|(_, d)| *d).unwrap_or(0.0);
            let last_depth = self.depths.last().map(|(_, d)| *d).unwrap_or(0.0);
            let window_secs = self.window_ns as f64 / 1_000_000_000.0;
            (last_depth - first_depth) / window_secs.max(EPSILON)
        } else {
            0.0
        };

        TradeFlowFeatures {
            signed_volume,
            elasticity,
            elasticity_undefined,
            depth_collapse_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::PriceLevel;

    fn make_level(price: f64, qty: f64) -> PriceLevel {
        PriceLevel {
            price_mantissa: (price * 100.0) as i64,
            price_exponent: -2,
            qty_mantissa: (qty * 1000.0) as i64,
            qty_exponent: -3,
        }
    }

    #[test]
    fn test_mid_price() {
        let book = OrderBook {
            ts_ns: 0,
            symbol: "TEST".to_string(),
            bids: vec![make_level(100.0, 1.0)],
            asks: vec![make_level(101.0, 1.0)],
        };

        let features = SnapshotFeatures::compute(&book, 0.01, 1.0);
        assert!((features.mid - 100.5).abs() < 0.001);
    }

    #[test]
    fn test_microprice() {
        let book = OrderBook {
            ts_ns: 0,
            symbol: "TEST".to_string(),
            bids: vec![make_level(100.0, 2.0)], // More bid qty
            asks: vec![make_level(101.0, 1.0)],
        };

        let features = SnapshotFeatures::compute(&book, 0.01, 1.0);
        // μ = (101 * 2 + 100 * 1) / (1 + 2) = 302 / 3 ≈ 100.67
        assert!((features.microprice - 100.666).abs() < 0.01);
    }

    #[test]
    fn test_imbalance() {
        let book = OrderBook {
            ts_ns: 0,
            symbol: "TEST".to_string(),
            bids: vec![make_level(100.0, 3.0)],
            asks: vec![make_level(101.0, 1.0)],
        };

        let features = SnapshotFeatures::compute(&book, 0.01, 1.0);
        // I = (3 - 1) / (3 + 1) = 0.5
        assert!((features.imbalance_10 - 0.5).abs() < 0.001);
    }
}
