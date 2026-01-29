//! Equity Curve Tracking for Backtest
//!
//! Provides bar-by-bar equity snapshots and online max drawdown tracking.
//!
//! ## Schema: quantlaxmi.equity_curve.v1
//!
//! ## Usage
//! - Emit EquityPoint every N seconds during quote ingestion
//! - Track max drawdown online with MaxDrawdownTracker
//! - Compute Sharpe from the resulting equity_curve.jsonl

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Configuration for equity curve emission
#[derive(Debug, Clone)]
pub struct EquityCurveConfig {
    /// Bar interval in seconds (e.g., 1 for 1s bars)
    pub interval_secs: i64,
}

impl Default for EquityCurveConfig {
    fn default() -> Self {
        Self { interval_secs: 1 }
    }
}

/// Single point on the equity curve (emitted to equity_curve.jsonl)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    /// Bar timestamp (floored to interval)
    pub ts_utc: String,

    /// Total equity = cash + sum(position_qty * mid_price)
    pub equity_inr: f64,

    /// PnL since start (equity - initial_equity)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pnl_inr: Option<f64>,

    /// Cash component (premium received/paid)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cash_inr: Option<f64>,

    /// MTM component (position value)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtm_inr: Option<f64>,
}

impl EquityPoint {
    pub fn new(ts: DateTime<Utc>, equity_inr: f64) -> Self {
        Self {
            ts_utc: ts.to_rfc3339(),
            equity_inr,
            pnl_inr: None,
            cash_inr: None,
            mtm_inr: None,
        }
    }

    pub fn with_components(ts: DateTime<Utc>, cash_inr: f64, mtm_inr: f64) -> Self {
        let equity_inr = cash_inr + mtm_inr;
        Self {
            ts_utc: ts.to_rfc3339(),
            equity_inr,
            pnl_inr: None,
            cash_inr: Some(cash_inr),
            mtm_inr: Some(mtm_inr),
        }
    }
}

/// Online max drawdown tracker
///
/// Tracks peak equity and computes drawdown metrics in a single pass.
#[derive(Debug, Clone)]
pub struct MaxDrawdownTracker {
    /// Highest equity seen so far
    pub peak_equity: f64,

    /// Timestamp of peak
    pub peak_ts: DateTime<Utc>,

    /// Maximum drawdown in INR (peak - trough)
    pub max_drawdown_inr: f64,

    /// Maximum drawdown as percentage of peak
    pub max_drawdown_pct: f64,

    /// Timestamp of trough (where max DD occurred)
    pub trough_ts: DateTime<Utc>,

    /// Current equity
    pub current_equity: f64,

    /// Number of updates
    pub updates: u64,
}

impl MaxDrawdownTracker {
    /// Create a new tracker with initial equity
    pub fn new(initial_equity: f64, ts: DateTime<Utc>) -> Self {
        Self {
            peak_equity: initial_equity,
            peak_ts: ts,
            max_drawdown_inr: 0.0,
            max_drawdown_pct: 0.0,
            trough_ts: ts,
            current_equity: initial_equity,
            updates: 1,
        }
    }

    /// Update with new equity value
    pub fn update(&mut self, equity: f64, ts: DateTime<Utc>) {
        self.current_equity = equity;
        self.updates += 1;

        // New peak?
        if equity > self.peak_equity {
            self.peak_equity = equity;
            self.peak_ts = ts;
        }

        // Compute current drawdown from peak
        let dd_inr = self.peak_equity - equity;
        let dd_pct = if self.peak_equity > 0.0 {
            dd_inr / self.peak_equity
        } else {
            0.0
        };

        // New max drawdown?
        if dd_inr > self.max_drawdown_inr {
            self.max_drawdown_inr = dd_inr;
            self.max_drawdown_pct = dd_pct;
            self.trough_ts = ts;
        }
    }

    /// Get peak timestamp as RFC3339 string
    pub fn peak_ts_utc(&self) -> String {
        self.peak_ts.to_rfc3339()
    }

    /// Get trough timestamp as RFC3339 string
    pub fn trough_ts_utc(&self) -> String {
        self.trough_ts.to_rfc3339()
    }
}

/// Floor timestamp to bar boundary
///
/// Example: with interval_secs=1, floors to the nearest second
pub fn floor_to_bar(ts: DateTime<Utc>, interval_secs: i64) -> DateTime<Utc> {
    let epoch_secs = ts.timestamp();
    let floored_secs = (epoch_secs / interval_secs) * interval_secs;
    DateTime::from_timestamp(floored_secs, 0).unwrap_or(ts)
}

/// Online returns tracker for Sharpe computation
///
/// Uses Welford's algorithm for numerically stable variance.
#[derive(Debug, Clone, Default)]
pub struct OnlineReturnsTracker {
    /// Number of returns observed
    pub count: u64,

    /// Running mean of returns
    pub mean: f64,

    /// M2 for Welford variance (sum of squared deviations)
    pub m2: f64,

    /// Previous equity for return calculation
    prev_equity: Option<f64>,

    /// Sum of positive returns (for Sortino)
    pub sum_positive: f64,

    /// Sum of squared negative returns (for Sortino downside deviation)
    pub sum_sq_negative: f64,

    /// Count of negative returns
    pub count_negative: u64,
}

impl OnlineReturnsTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with new equity value, computing return from previous
    pub fn update(&mut self, equity: f64) {
        if let Some(prev) = self.prev_equity {
            if prev > 0.0 {
                // Simple return
                let ret = (equity - prev) / prev;
                self.add_return(ret);
            }
        }
        self.prev_equity = Some(equity);
    }

    /// Add a return value directly
    fn add_return(&mut self, ret: f64) {
        self.count += 1;
        let delta = ret - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = ret - self.mean;
        self.m2 += delta * delta2;

        // Track for Sortino
        if ret > 0.0 {
            self.sum_positive += ret;
        } else {
            self.sum_sq_negative += ret * ret;
            self.count_negative += 1;
        }
    }

    /// Get variance (population)
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get Sharpe ratio (non-annualized)
    pub fn sharpe(&self) -> f64 {
        let std = self.std_dev();
        if std > 0.0 { self.mean / std } else { 0.0 }
    }

    /// Get downside deviation (for Sortino)
    pub fn downside_deviation(&self) -> f64 {
        if self.count_negative > 0 {
            (self.sum_sq_negative / self.count_negative as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Get Sortino ratio (non-annualized)
    pub fn sortino(&self) -> f64 {
        let dd = self.downside_deviation();
        if dd > 0.0 { self.mean / dd } else { 0.0 }
    }
}

/// Summary of equity curve metrics (computed after curve is written)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EquityCurveSummary {
    pub bar_interval: String,
    pub bars: u32,

    pub initial_equity_inr: f64,
    pub final_equity_inr: f64,
    pub gross_pnl_inr: f64,

    pub max_drawdown_inr: f64,
    pub max_drawdown_pct: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub dd_peak_ts_utc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dd_trough_ts_utc: Option<String>,

    pub sharpe: f64,
    pub sortino: f64,

    pub mean_return: f64,
    pub std_return: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_to_bar() {
        let ts = DateTime::parse_from_rfc3339("2026-01-29T12:34:56.789Z")
            .unwrap()
            .with_timezone(&Utc);

        let floored = floor_to_bar(ts, 1);
        assert_eq!(floored.timestamp(), ts.timestamp()); // Same second

        let floored_10s = floor_to_bar(ts, 10);
        assert_eq!(floored_10s.timestamp() % 10, 0);
    }

    #[test]
    fn test_max_drawdown_tracker() {
        let t0 = Utc::now();
        let mut tracker = MaxDrawdownTracker::new(100.0, t0);

        // Rise to 110
        tracker.update(110.0, t0 + Duration::seconds(1));
        assert_eq!(tracker.peak_equity, 110.0);
        assert_eq!(tracker.max_drawdown_inr, 0.0);

        // Drop to 90 (20 from peak of 110)
        tracker.update(90.0, t0 + Duration::seconds(2));
        assert!((tracker.max_drawdown_inr - 20.0).abs() < 0.01);
        assert!((tracker.max_drawdown_pct - 0.1818).abs() < 0.01);

        // Recover to 105 (still below peak)
        tracker.update(105.0, t0 + Duration::seconds(3));
        assert_eq!(tracker.peak_equity, 110.0); // Peak unchanged
        assert!((tracker.max_drawdown_inr - 20.0).abs() < 0.01); // Max DD unchanged

        // New peak at 120
        tracker.update(120.0, t0 + Duration::seconds(4));
        assert_eq!(tracker.peak_equity, 120.0);
    }

    #[test]
    fn test_online_returns_sharpe() {
        let mut tracker = OnlineReturnsTracker::new();

        // Simulate equity: 100 -> 101 -> 102 -> 101 -> 103
        tracker.update(100.0);
        tracker.update(101.0); // +1%
        tracker.update(102.0); // +0.99%
        tracker.update(101.0); // -0.98%
        tracker.update(103.0); // +1.98%

        assert_eq!(tracker.count, 4); // 4 returns
        assert!(tracker.mean > 0.0); // Positive mean
        assert!(tracker.sharpe() > 0.0); // Positive Sharpe
    }
}
