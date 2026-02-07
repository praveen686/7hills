//! Risk limits for basis mean-reversion.

/// Risk limit configuration.
#[derive(Debug, Clone)]
pub struct RiskLimits {
    /// Maximum total exposure across all pairs (USDT)
    pub max_total_exposure_usd: f64,
    /// Maximum exposure per pair (USDT)
    pub max_per_pair_usd: f64,
    /// Maximum portfolio drawdown percentage before halting
    pub max_drawdown_pct: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_total_exposure_usd: 10_000.0,
            max_per_pair_usd: 2_000.0,
            max_drawdown_pct: 5.0,
        }
    }
}

impl RiskLimits {
    /// Check if adding a new position would exceed total exposure.
    pub fn can_add_exposure(&self, current_exposure: f64, new_position_usd: f64) -> bool {
        current_exposure + new_position_usd <= self.max_total_exposure_usd
    }

    /// Check if a position size is within per-pair limit.
    pub fn within_pair_limit(&self, position_usd: f64) -> bool {
        position_usd <= self.max_per_pair_usd
    }

    /// Check if portfolio is within drawdown limit.
    pub fn within_drawdown(&self, initial_capital: f64, current_equity: f64) -> bool {
        if initial_capital <= 0.0 {
            return false;
        }
        let drawdown_pct = (1.0 - current_equity / initial_capital) * 100.0;
        drawdown_pct < self.max_drawdown_pct
    }
}
