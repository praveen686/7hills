//! Put-Call Ratio (PCR) Analysis
//!
//! Implements comprehensive PCR analysis for market sentiment:
//! - Volume PCR
//! - Open Interest PCR
//! - Dollar-weighted PCR
//! - Multi-timeframe PCR
//! - Strike-based PCR (near ATM vs far OTM)
//!
//! Trading signals derived from PCR extremes and divergences.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Single option data point for PCR calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionData {
    pub strike: f64,
    pub expiry_dte: u32,
    pub option_type: OptionDataType,
    pub volume: u64,
    pub open_interest: u64,
    pub last_price: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionDataType {
    Call,
    Put,
}

/// PCR calculation result.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PCRMetrics {
    /// Volume-based PCR (put_volume / call_volume)
    pub volume_pcr: f64,
    /// Open interest PCR (put_oi / call_oi)
    pub oi_pcr: f64,
    /// Dollar-weighted PCR (put_value / call_value)
    pub dollar_pcr: f64,
    /// Total put volume
    pub total_put_volume: u64,
    /// Total call volume
    pub total_call_volume: u64,
    /// Total put OI
    pub total_put_oi: u64,
    /// Total call OI
    pub total_call_oi: u64,
}

impl PCRMetrics {
    /// Calculate PCR from option chain data.
    pub fn from_chain(options: &[OptionData]) -> Self {
        let mut put_volume = 0u64;
        let mut call_volume = 0u64;
        let mut put_oi = 0u64;
        let mut call_oi = 0u64;
        let mut put_value = 0.0f64;
        let mut call_value = 0.0f64;

        for opt in options {
            match opt.option_type {
                OptionDataType::Put => {
                    put_volume += opt.volume;
                    put_oi += opt.open_interest;
                    put_value += opt.volume as f64 * opt.last_price;
                }
                OptionDataType::Call => {
                    call_volume += opt.volume;
                    call_oi += opt.open_interest;
                    call_value += opt.volume as f64 * opt.last_price;
                }
            }
        }

        let volume_pcr = if call_volume > 0 {
            put_volume as f64 / call_volume as f64
        } else {
            1.0
        };

        let oi_pcr = if call_oi > 0 {
            put_oi as f64 / call_oi as f64
        } else {
            1.0
        };

        let dollar_pcr = if call_value > 0.0 {
            put_value / call_value
        } else {
            1.0
        };

        Self {
            volume_pcr,
            oi_pcr,
            dollar_pcr,
            total_put_volume: put_volume,
            total_call_volume: call_volume,
            total_put_oi: put_oi,
            total_call_oi: call_oi,
        }
    }

    /// Calculate PCR for specific delta range (near ATM).
    pub fn from_chain_delta_range(options: &[OptionData], min_delta: f64, max_delta: f64) -> Self {
        let filtered: Vec<_> = options
            .iter()
            .filter(|o| o.delta.abs() >= min_delta && o.delta.abs() <= max_delta)
            .cloned()
            .collect();
        Self::from_chain(&filtered)
    }
}

/// PCR signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PCRSignal {
    /// PCR > 1.2: Extreme put buying (contrarian bullish)
    ExtremeBearish,
    /// PCR 0.9-1.2: Moderate put bias
    ModerateBearish,
    /// PCR 0.7-0.9: Neutral
    Neutral,
    /// PCR 0.5-0.7: Moderate call bias
    ModerateBullish,
    /// PCR < 0.5: Extreme call buying (contrarian bearish)
    ExtremeBullish,
}

impl PCRSignal {
    /// Interpret PCR value.
    pub fn from_pcr(pcr: f64) -> Self {
        if pcr > 1.2 {
            PCRSignal::ExtremeBearish
        } else if pcr > 0.9 {
            PCRSignal::ModerateBearish
        } else if pcr < 0.5 {
            PCRSignal::ExtremeBullish
        } else if pcr < 0.7 {
            PCRSignal::ModerateBullish
        } else {
            PCRSignal::Neutral
        }
    }

    /// Get contrarian trading bias (opposite of sentiment).
    pub fn contrarian_bias(&self) -> TradingBias {
        match self {
            PCRSignal::ExtremeBearish => TradingBias::Bullish, // Buy calls
            PCRSignal::ModerateBearish => TradingBias::SlightlyBullish,
            PCRSignal::Neutral => TradingBias::Neutral,
            PCRSignal::ModerateBullish => TradingBias::SlightlyBearish,
            PCRSignal::ExtremeBullish => TradingBias::Bearish, // Buy puts
        }
    }

    /// Get momentum trading bias (follow sentiment).
    pub fn momentum_bias(&self) -> TradingBias {
        match self {
            PCRSignal::ExtremeBearish => TradingBias::Bearish,
            PCRSignal::ModerateBearish => TradingBias::SlightlyBearish,
            PCRSignal::Neutral => TradingBias::Neutral,
            PCRSignal::ModerateBullish => TradingBias::SlightlyBullish,
            PCRSignal::ExtremeBullish => TradingBias::Bullish,
        }
    }
}

/// Trading bias from PCR analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingBias {
    Bullish,
    SlightlyBullish,
    Neutral,
    SlightlyBearish,
    Bearish,
}

/// Rolling PCR tracker for trend analysis.
#[derive(Debug, Clone)]
pub struct PCRTracker {
    /// History of PCR values
    history: VecDeque<PCRSnapshot>,
    /// Maximum history length
    max_history: usize,
    /// Smoothing window for moving average
    smoothing_window: usize,
}

/// PCR snapshot at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCRSnapshot {
    pub timestamp: i64,
    pub metrics: PCRMetrics,
}

impl PCRTracker {
    /// Create new tracker.
    pub fn new(max_history: usize, smoothing_window: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            smoothing_window,
        }
    }

    /// Add new PCR reading.
    pub fn update(&mut self, timestamp: i64, metrics: PCRMetrics) {
        self.history.push_back(PCRSnapshot { timestamp, metrics });
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get current PCR.
    pub fn current(&self) -> Option<&PCRMetrics> {
        self.history.back().map(|s| &s.metrics)
    }

    /// Get smoothed (moving average) PCR.
    pub fn smoothed_volume_pcr(&self) -> f64 {
        let window: Vec<_> = self
            .history
            .iter()
            .rev()
            .take(self.smoothing_window)
            .map(|s| s.metrics.volume_pcr)
            .collect();

        if window.is_empty() {
            1.0
        } else {
            window.iter().sum::<f64>() / window.len() as f64
        }
    }

    /// Get PCR percentile over history.
    pub fn percentile(&self) -> f64 {
        if self.history.len() < 10 {
            return 50.0;
        }

        let current = self
            .history
            .back()
            .map(|s| s.metrics.volume_pcr)
            .unwrap_or(1.0);
        let count_below = self
            .history
            .iter()
            .filter(|s| s.metrics.volume_pcr < current)
            .count();

        (count_below as f64 / self.history.len() as f64) * 100.0
    }

    /// Detect PCR divergence from price.
    ///
    /// Bullish divergence: Price making lower lows, PCR making higher highs (extreme fear)
    /// Bearish divergence: Price making higher highs, PCR making lower lows (complacency)
    pub fn detect_divergence(&self, price_trend: PriceTrend) -> Option<Divergence> {
        if self.history.len() < 5 {
            return None;
        }

        let pcr_trend = self.pcr_trend();

        match (price_trend, pcr_trend) {
            (PriceTrend::Down, PCRTrend::Up) => Some(Divergence::Bullish),
            (PriceTrend::Up, PCRTrend::Down) => Some(Divergence::Bearish),
            _ => None,
        }
    }

    fn pcr_trend(&self) -> PCRTrend {
        if self.history.len() < 3 {
            return PCRTrend::Flat;
        }

        let recent: Vec<_> = self
            .history
            .iter()
            .rev()
            .take(3)
            .map(|s| s.metrics.volume_pcr)
            .collect();

        if recent[0] > recent[1] && recent[1] > recent[2] {
            PCRTrend::Up
        } else if recent[0] < recent[1] && recent[1] < recent[2] {
            PCRTrend::Down
        } else {
            PCRTrend::Flat
        }
    }

    /// Get composite PCR signal combining multiple factors.
    pub fn composite_signal(&self) -> CompositeSignal {
        let current = self.current().map(|m| m.volume_pcr).unwrap_or(1.0);
        let smoothed = self.smoothed_volume_pcr();
        let percentile = self.percentile();

        let signal = PCRSignal::from_pcr(current);
        let trend = self.pcr_trend();

        // Strength based on percentile extremity
        let strength = if !(10.0..=90.0).contains(&percentile) {
            SignalStrength::Strong
        } else if !(20.0..=80.0).contains(&percentile) {
            SignalStrength::Moderate
        } else {
            SignalStrength::Weak
        };

        CompositeSignal {
            signal,
            strength,
            current_pcr: current,
            smoothed_pcr: smoothed,
            percentile,
            trend,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceTrend {
    Up,
    Down,
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PCRTrend {
    Up,
    Down,
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Divergence {
    /// Price down, PCR up (bullish reversal)
    Bullish,
    /// Price up, PCR down (bearish reversal)
    Bearish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    Strong,
    Moderate,
    Weak,
}

/// Composite signal from PCR analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeSignal {
    pub signal: PCRSignal,
    pub strength: SignalStrength,
    pub current_pcr: f64,
    pub smoothed_pcr: f64,
    pub percentile: f64,
    pub trend: PCRTrend,
}

/// Max Pain calculator.
///
/// Max Pain is the strike at which option writers have minimum payout,
/// based on the theory that price gravitates to max pain at expiry.
pub fn calculate_max_pain(options: &[OptionData], spot: f64) -> Option<f64> {
    // Get unique strikes (NaN-safe using total_cmp)
    let mut strikes: Vec<f64> = options.iter().map(|o| o.strike).collect();
    strikes.sort_by(|a, b| a.total_cmp(b));
    strikes.dedup();

    if strikes.is_empty() {
        return None;
    }

    let mut min_pain = f64::MAX;
    let mut max_pain_strike = spot;

    for &test_strike in &strikes {
        let mut total_pain = 0.0;

        for opt in options {
            let oi = opt.open_interest as f64;
            match opt.option_type {
                OptionDataType::Call => {
                    // Call pain = max(0, test_strike - strike) * OI
                    let intrinsic = (test_strike - opt.strike).max(0.0);
                    total_pain += intrinsic * oi;
                }
                OptionDataType::Put => {
                    // Put pain = max(0, strike - test_strike) * OI
                    let intrinsic = (opt.strike - test_strike).max(0.0);
                    total_pain += intrinsic * oi;
                }
            }
        }

        if total_pain < min_pain {
            min_pain = total_pain;
            max_pain_strike = test_strike;
        }
    }

    Some(max_pain_strike)
}

/// Options flow analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowAnalysis {
    /// Net call buying (buy volume - sell volume)
    pub net_call_flow: i64,
    /// Net put buying
    pub net_put_flow: i64,
    /// Large trades (> threshold)
    pub large_call_buys: u32,
    pub large_call_sells: u32,
    pub large_put_buys: u32,
    pub large_put_sells: u32,
    /// Premium spent
    pub call_premium_bought: f64,
    pub call_premium_sold: f64,
    pub put_premium_bought: f64,
    pub put_premium_sold: f64,
}

impl FlowAnalysis {
    /// Get net flow signal.
    pub fn signal(&self) -> TradingBias {
        let call_net = self.call_premium_bought - self.call_premium_sold;
        let put_net = self.put_premium_bought - self.put_premium_sold;

        if call_net > put_net * 1.5 {
            TradingBias::Bullish
        } else if put_net > call_net * 1.5 {
            TradingBias::Bearish
        } else {
            TradingBias::Neutral
        }
    }

    /// Smart money indicator (large trades bias).
    pub fn smart_money_bias(&self) -> TradingBias {
        let call_smart = self.large_call_buys as i32 - self.large_call_sells as i32;
        let put_smart = self.large_put_buys as i32 - self.large_put_sells as i32;

        if call_smart > put_smart + 2 {
            TradingBias::Bullish
        } else if put_smart > call_smart + 2 {
            TradingBias::Bearish
        } else {
            TradingBias::Neutral
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcr_calculation() {
        let options = vec![
            OptionData {
                strike: 100.0,
                expiry_dte: 30,
                option_type: OptionDataType::Call,
                volume: 1000,
                open_interest: 5000,
                last_price: 5.0,
                delta: 0.5,
            },
            OptionData {
                strike: 100.0,
                expiry_dte: 30,
                option_type: OptionDataType::Put,
                volume: 800,
                open_interest: 4000,
                last_price: 4.5,
                delta: -0.5,
            },
        ];

        let metrics = PCRMetrics::from_chain(&options);
        assert!((metrics.volume_pcr - 0.8).abs() < 0.01);
        assert!((metrics.oi_pcr - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_pcr_signal() {
        assert_eq!(PCRSignal::from_pcr(1.5), PCRSignal::ExtremeBearish);
        assert_eq!(PCRSignal::from_pcr(0.8), PCRSignal::Neutral);
        assert_eq!(PCRSignal::from_pcr(0.3), PCRSignal::ExtremeBullish);
    }

    #[test]
    fn test_max_pain() {
        let options = vec![
            OptionData {
                strike: 95.0,
                expiry_dte: 7,
                option_type: OptionDataType::Put,
                volume: 100,
                open_interest: 1000,
                last_price: 1.0,
                delta: -0.2,
            },
            OptionData {
                strike: 100.0,
                expiry_dte: 7,
                option_type: OptionDataType::Call,
                volume: 100,
                open_interest: 2000,
                last_price: 2.0,
                delta: 0.5,
            },
            OptionData {
                strike: 100.0,
                expiry_dte: 7,
                option_type: OptionDataType::Put,
                volume: 100,
                open_interest: 2000,
                last_price: 2.0,
                delta: -0.5,
            },
            OptionData {
                strike: 105.0,
                expiry_dte: 7,
                option_type: OptionDataType::Call,
                volume: 100,
                open_interest: 1000,
                last_price: 0.5,
                delta: 0.2,
            },
        ];

        let max_pain = calculate_max_pain(&options, 100.0);
        assert!(max_pain.is_some());
        // Max pain should be near 100 where both sides have most OI
        assert!((max_pain.unwrap() - 100.0).abs() < 10.0);
    }
}
