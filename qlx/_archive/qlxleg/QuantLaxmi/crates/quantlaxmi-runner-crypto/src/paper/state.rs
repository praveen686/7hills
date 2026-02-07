use std::fmt;

/// High-level eligibility status shown by the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum R3Eligibility {
    Eligible,
    Refused,
    #[default]
    Unknown,
}

impl fmt::Display for R3Eligibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            R3Eligibility::Eligible => write!(f, "ELIGIBLE"),
            R3Eligibility::Refused => write!(f, "REFUSED"),
            R3Eligibility::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// A refusal reason intended for operator clarity (not just "debug strings").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusalReason {
    pub code: &'static str,
    pub detail: String,
}

impl RefusalReason {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

/// Core model outputs you requested to show in the TUI.
/// Keep these as Option<f64> so we never fabricate defaults.
#[derive(Debug, Clone, Default)]
pub struct DecisionMetrics {
    pub confidence: Option<f64>,
    pub d_perp: Option<f64>,
    pub fragility: Option<f64>,
    pub toxicity: Option<f64>,
    /// Toxicity persistence (fraction of recent windows where toxicity was elevated)
    pub toxicity_persist: Option<f64>,
    /// FTI level (follow-through indicator)
    pub fti_level: Option<f64>,
    /// FTI persistence (fraction of recent windows where FTI was elevated)
    pub fti_persist: Option<f64>,
    /// FTI persist threshold (calibrated or default 1.0)
    pub fti_thresh: Option<f64>,
    /// Whether current FTI is elevated (fti_level > fti_thresh)
    pub fti_elevated: Option<bool>,
    /// Whether FTI threshold has been calibrated
    pub fti_calibrated: Option<bool>,
}

/// Last trade info for display.
#[derive(Debug, Clone, Default)]
pub struct LastTrade {
    pub price: f64,
    pub qty: f64,
    pub is_buy: bool,
    pub ts_ms: u64,
}

/// Recent trades tape for display.
#[derive(Debug, Clone, Default)]
pub struct TradeTape {
    pub trades: Vec<LastTrade>,
    pub max_trades: usize,
}

impl TradeTape {
    pub fn new(max_trades: usize) -> Self {
        Self {
            trades: Vec::with_capacity(max_trades),
            max_trades,
        }
    }

    pub fn add(&mut self, trade: LastTrade) {
        self.trades.push(trade);
        if self.trades.len() > self.max_trades {
            self.trades.remove(0);
        }
    }

    /// Get trades in reverse order (most recent first)
    pub fn recent(&self) -> impl Iterator<Item = &LastTrade> {
        self.trades.iter().rev()
    }
}

/// Market data snapshot for display.
#[derive(Debug, Clone, Default)]
pub struct MarketData {
    pub symbol: String,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub mid_price: Option<f64>,
    pub spread_bps: Option<f64>,
    pub bid_qty: Option<f64>,
    pub ask_qty: Option<f64>,
    pub imbalance: Option<f64>,
    pub tick_count: u64,
    pub last_trade: Option<LastTrade>,
    pub trade_tape: TradeTape,
    pub trades_per_sec: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
}

/// Paper trading position for PnL tracking.
#[derive(Debug, Clone)]
pub struct PaperPosition {
    /// Current position size (positive = long, negative = short)
    pub size: f64,
    /// Average entry price (weighted average)
    pub avg_entry_price: f64,
    /// Total realized PnL from closed trades
    pub realized_pnl: f64,
    /// Current unrealized PnL (mark-to-market)
    pub unrealized_pnl: f64,
    /// Total fees paid
    pub total_fees: f64,
    /// Fee rate (e.g., 0.001 = 0.1%)
    pub fee_rate: f64,
    /// Total number of fills
    pub fill_count: u32,
    /// Last fill price
    pub last_fill_price: Option<f64>,
    /// Last fill side (true = buy)
    pub last_fill_is_buy: Option<bool>,
}

impl Default for PaperPosition {
    fn default() -> Self {
        Self {
            size: 0.0,
            avg_entry_price: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            total_fees: 0.0,
            fee_rate: 0.001, // 0.1% taker fee (Binance spot default)
            fill_count: 0,
            last_fill_price: None,
            last_fill_is_buy: None,
        }
    }
}

impl PaperPosition {
    /// Fill a trade and update position/PnL.
    pub fn fill(&mut self, qty: f64, price: f64, is_buy: bool) {
        let signed_qty = if is_buy { qty } else { -qty };

        // Calculate and accumulate fee (notional * fee_rate)
        let notional = qty * price;
        let fee = notional * self.fee_rate;
        self.total_fees += fee;

        // Check if this reduces or flips position
        let same_direction = (self.size >= 0.0 && is_buy) || (self.size <= 0.0 && !is_buy);

        if same_direction || self.size == 0.0 {
            // Adding to position - update average entry
            let total_cost = self.avg_entry_price * self.size.abs() + price * qty;
            let new_size = self.size + signed_qty;
            if new_size.abs() > 1e-12 {
                self.avg_entry_price = total_cost / new_size.abs();
            }
            self.size = new_size;
        } else {
            // Reducing position - realize PnL
            let close_qty = qty.min(self.size.abs());
            let pnl_per_unit = if self.size > 0.0 {
                price - self.avg_entry_price // Long: sell higher = profit
            } else {
                self.avg_entry_price - price // Short: buy lower = profit
            };
            self.realized_pnl += pnl_per_unit * close_qty;

            // Update position
            self.size += signed_qty;

            // If position flipped, set new entry price
            if (self.size > 0.0 && !is_buy) || (self.size < 0.0 && is_buy) {
                // Position flipped - remaining qty at new price
                let remaining = qty - close_qty;
                if remaining > 1e-12 {
                    self.avg_entry_price = price;
                }
            }
        }

        self.fill_count += 1;
        self.last_fill_price = Some(price);
        self.last_fill_is_buy = Some(is_buy);
    }

    /// Update unrealized PnL based on current market price.
    pub fn mark_to_market(&mut self, current_price: f64) {
        if self.size.abs() < 1e-12 {
            self.unrealized_pnl = 0.0;
        } else if self.size > 0.0 {
            // Long position
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.size;
        } else {
            // Short position
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.size.abs();
        }
    }

    /// Total PnL (realized + unrealized - fees).
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl - self.total_fees
    }
}

/// Last decision summary.
#[derive(Debug, Clone)]
pub struct LastDecision {
    pub eligibility: R3Eligibility,
    pub refusal_reasons: Vec<RefusalReason>,
    pub metrics: DecisionMetrics,
    pub decided_at_unix_ms: u64,
}

impl Default for LastDecision {
    fn default() -> Self {
        Self {
            eligibility: R3Eligibility::Unknown,
            refusal_reasons: vec![],
            metrics: DecisionMetrics::default(),
            decided_at_unix_ms: 0,
        }
    }
}

/// Sniper admission statistics for TUI display.
#[derive(Debug, Clone, Default)]
pub struct SniperStats {
    /// Entries in last hour
    pub entries_last_hour: u32,
    /// Maximum entries per hour allowed
    pub max_per_hour: u32,
    /// Total session entries
    pub session_entries: u32,
    /// Maximum session entries allowed
    pub max_per_session: u32,
    /// Seconds since last entry (None if no entries yet)
    pub secs_since_last: Option<u64>,
    /// Cooldown seconds required
    pub cooldown_secs: u64,
    /// Is cooldown satisfied?
    pub cooldown_ok: bool,
    /// Current regime (R0, R1, R2, R3)
    pub regime: String,
    /// What caused R3 to be true (FTI, TOX, BOTH, or NONE) - only meaningful when regime=R3
    pub r3_cause: String,
}

/// Single snapshot the TUI reads; updated by the runner.
#[derive(Debug, Clone, Default)]
pub struct UiSnapshot {
    pub last_decision: LastDecision,
    /// What the strategy wanted to do THIS TICK (may exist even if refused)
    pub proposed_this_tick: Option<crate::paper::intent::PaperIntent>,
    /// What was accepted THIS TICK - MUST be None when eligibility != ELIGIBLE
    pub accepted_this_tick: Option<crate::paper::intent::PaperIntent>,
    /// Historical memory of the last accepted intent (persists across ticks)
    pub last_accepted_historical: Option<crate::paper::intent::PaperIntent>,
    pub market_data: MarketData,
    pub position: PaperPosition,
    /// Sniper admission statistics
    pub sniper_stats: SniperStats,
}
