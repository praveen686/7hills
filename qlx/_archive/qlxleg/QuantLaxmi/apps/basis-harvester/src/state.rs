//! TUI state for basis harvester rendering.

use crate::intent::BasisDirection;

/// Per-symbol z-score monitor display.
#[derive(Debug, Clone, Default)]
pub struct ZScoreDisplay {
    pub symbol: String,
    pub basis_bps: f64,
    pub z_score: f64,
    pub rolling_std: f64,
    pub rolling_mean: f64,
    pub observations: usize,
}

/// Active trade display.
#[derive(Debug, Clone)]
pub struct TradeDisplay {
    pub symbol: String,
    pub direction: BasisDirection,
    pub entry_basis_bps: f64,
    pub entry_z: f64,
    pub current_basis_bps: f64,
    pub current_z: f64,
    pub pnl_bps: f64,
    pub hold_secs: u64,
}

/// Trade statistics.
#[derive(Debug, Clone, Default)]
pub struct TradeStatistics {
    pub total_trades: u64,
    pub wins: u64,
    pub losses: u64,
    pub avg_hold_secs: f64,
    pub avg_pnl_bps: f64,
    pub total_pnl_bps: f64,
}

impl TradeStatistics {
    pub fn win_rate_pct(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.wins as f64 / self.total_trades as f64 * 100.0
    }
}

/// Complete state for TUI rendering.
#[derive(Debug, Clone, Default)]
pub struct BasisHarvesterState {
    /// Z-score monitor for all tracked symbols
    pub z_scores: Vec<ZScoreDisplay>,

    /// Active trades
    pub active_trades: Vec<TradeDisplay>,

    /// Trade statistics
    pub stats: TradeStatistics,

    /// Engine state (from PaperState)
    pub equity: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub fees_paid: f64,
    pub fills: u64,
    pub rejections: u64,
    pub last_decision: String,
    pub initial_capital: f64,

    /// Liveness
    pub is_finished: bool,
    pub tick_count: u64,
}
