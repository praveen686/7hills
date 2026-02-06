use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Option leg view for TUI display.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptionLegView {
    /// Trading symbol (e.g., "NIFTY2510223400CE")
    pub symbol: String,
    /// Strike price
    pub strike: i32,
    /// Right ("CE" or "PE")
    pub right: String,
    /// Expiry date
    pub expiry: String,
    /// Best bid price
    pub bid: Option<f64>,
    /// Best ask price
    pub ask: Option<f64>,
    /// Mid price
    pub mid: Option<f64>,
    /// Quote age in ms
    pub age_ms: u32,
}

/// Active position view for TUI display.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PositionView {
    /// Instrument symbol
    pub symbol: String,
    /// Position quantity (negative = short)
    pub qty: i32,
    /// Average entry price
    pub avg_price: f64,
    /// Current MTM value
    pub mtm: f64,
    /// Unrealized PnL for this position
    pub unrealized_pnl: f64,
}

/// Strategy view for TUI display.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StrategyView {
    /// Strategy name
    pub name: String,
    /// Underlying symbol (e.g., "NIFTY")
    pub underlying: String,
    /// Spot price
    pub spot: Option<f64>,
    /// Futures price (if available)
    pub futures: Option<f64>,
    /// Edge estimate in rupees
    pub edge_rupees: f64,
    /// Friction estimate in rupees
    pub friction_rupees: f64,
    /// Edge minus friction in rupees
    pub net_edge_rupees: f64,
    /// Entry threshold in rupees
    pub entry_threshold_rupees: f64,
    /// Exit threshold in rupees
    pub exit_threshold_rupees: f64,
    /// Front leg option being evaluated
    pub front_leg: Option<OptionLegView>,
    /// Back leg option being evaluated
    pub back_leg: Option<OptionLegView>,
    /// Active positions
    pub positions: Vec<PositionView>,
    /// Decision type (Accept/Refuse/Hold)
    pub decision_type: String,
    /// Detailed reason for current decision
    pub decision_reason: String,
}

/// Minimal state surface for UI / web dashboards.
/// Keep this stable and extend conservatively.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PaperState {
    pub ts: Option<DateTime<Utc>>,
    /// Cash balance (not equity - see equity field)
    pub cash: f64,
    /// Equity = cash + unrealized_pnl (conservative MTM)
    pub equity: f64,
    /// Unrealized PnL with conservative MTM (longs at bid, shorts at ask)
    pub unrealized_pnl: f64,
    /// Realized PnL from closed positions
    pub realized_pnl: f64,
    /// Total PnL = realized + unrealized
    pub total_pnl: f64,
    /// Total fees paid
    pub fees_paid: f64,
    /// Number of open positions
    pub open_positions: usize,
    /// Last decision reason string
    pub last_decision: Option<String>,
    /// Fill count (for TUI display)
    pub fills: u64,
    /// Rejection count (for TUI display)
    pub rejections: u64,
    /// Engine finished flag (for TUI shutdown)
    pub is_finished: bool,
    /// Strategy view (optional, populated by strategy adapter)
    pub strategy_view: Option<StrategyView>,
}
