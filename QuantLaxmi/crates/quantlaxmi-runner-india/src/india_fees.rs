//! India Fee Model: Zerodha/NSE Options (NIFTY/BANKNIFTY)
//!
//! ## Fee Structure (as of 2026)
//! - Brokerage: ₹20/order (flat)
//! - STT: 0.1% on sell-side turnover (options)
//! - Exchange Txn: 0.03503% on turnover
//! - SEBI: ₹10/crore (0.0001%)
//! - Stamp Duty: 0.003% on buy-side turnover
//! - GST: 18% on (brokerage + exchange_txn + sebi)
//!
//! ## Usage
//! ```rust
//! let fee = IndiaFeeModel::zerodha_nse();
//! let cost = fee.estimate_cost(&fill_record);
//! ```

use serde::{Deserialize, Serialize};

/// Fee constants for Zerodha/NSE options trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndiaFeeModel {
    /// Brokerage per order (₹)
    pub brokerage_per_order_inr: f64,

    /// STT rate on sell-side turnover (0.001 = 0.1%)
    pub stt_rate: f64,

    /// Exchange transaction charge rate (0.0003503 = 0.03503%)
    pub exchange_txn_rate: f64,

    /// SEBI charges per crore (₹10/crore = 0.000001)
    pub sebi_rate: f64,

    /// Stamp duty on buy-side turnover (0.00003 = 0.003%)
    pub stamp_rate: f64,

    /// GST rate on (brokerage + exchange_txn + sebi) (0.18 = 18%)
    pub gst_rate: f64,

    /// Lot size (for options contracts)
    pub lot_size: u32,
}

impl Default for IndiaFeeModel {
    fn default() -> Self {
        Self::zerodha_nse()
    }
}

impl IndiaFeeModel {
    /// Standard Zerodha/NSE options fee structure
    pub fn zerodha_nse() -> Self {
        Self {
            brokerage_per_order_inr: 20.0,
            stt_rate: 0.001,              // 0.1% on sell
            exchange_txn_rate: 0.0003503, // 0.03503%
            sebi_rate: 0.000001,          // ₹10/crore = 1e-6
            stamp_rate: 0.00003,          // 0.003% on buy
            gst_rate: 0.18,               // 18%
            lot_size: 15,                 // BANKNIFTY lot size
        }
    }

    /// Zerodha/NSE with NIFTY lot size
    pub fn zerodha_nse_nifty() -> Self {
        Self {
            lot_size: 25, // NIFTY lot size
            ..Self::zerodha_nse()
        }
    }

    /// Estimate total cost for a fill in INR
    ///
    /// # Arguments
    /// * `fill_price` - Fill price in INR (premium per share)
    /// * `filled_qty` - Number of contracts filled
    /// * `side` - "Buy" or "Sell"
    ///
    /// # Returns
    /// `FillFeeBreakdown` with all fee components
    pub fn estimate_cost(&self, fill_price: f64, filled_qty: u32, side: &str) -> FillFeeBreakdown {
        let is_buy = side.eq_ignore_ascii_case("buy");
        let is_sell = !is_buy;

        // Turnover = premium × qty × lot_size
        let turnover_inr = fill_price * (filled_qty as f64) * (self.lot_size as f64);

        // Brokerage: flat per order
        let brokerage_inr = self.brokerage_per_order_inr;

        // STT: 0.1% on sell-side turnover only (for options)
        let stt_inr = if is_sell {
            turnover_inr * self.stt_rate
        } else {
            0.0
        };

        // Exchange transaction: on both sides
        let exchange_txn_inr = turnover_inr * self.exchange_txn_rate;

        // SEBI: ₹10/crore on both sides
        let sebi_inr = turnover_inr * self.sebi_rate;

        // Stamp duty: 0.003% on buy-side only
        let stamp_inr = if is_buy {
            turnover_inr * self.stamp_rate
        } else {
            0.0
        };

        // GST: 18% on (brokerage + exchange_txn + sebi)
        let gst_base = brokerage_inr + exchange_txn_inr + sebi_inr;
        let gst_inr = gst_base * self.gst_rate;

        // Total fees
        let total_inr = brokerage_inr + stt_inr + exchange_txn_inr + sebi_inr + stamp_inr + gst_inr;

        FillFeeBreakdown {
            turnover_inr,
            brokerage_inr,
            stt_inr,
            exchange_txn_inr,
            sebi_inr,
            stamp_inr,
            gst_inr,
            total_inr,
            side: side.to_string(),
            fill_price,
            filled_qty,
            lot_size: self.lot_size,
        }
    }

    /// Estimate round-trip cost for a position in INR
    ///
    /// Round-trip = buy fees + sell fees for the same notional
    pub fn estimate_round_trip(
        &self,
        entry_price: f64,
        exit_price: f64,
        qty: u32,
    ) -> RoundTripFeeBreakdown {
        let buy_fees = self.estimate_cost(entry_price, qty, "Buy");
        let sell_fees = self.estimate_cost(exit_price, qty, "Sell");
        let total_inr = buy_fees.total_inr + sell_fees.total_inr;

        RoundTripFeeBreakdown {
            entry_fees: buy_fees,
            exit_fees: sell_fees,
            total_inr,
        }
    }
}

/// Fee breakdown for a single fill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillFeeBreakdown {
    /// Turnover (premium × qty × lot_size)
    pub turnover_inr: f64,

    /// Brokerage (flat ₹20)
    pub brokerage_inr: f64,

    /// STT (0.1% on sell)
    pub stt_inr: f64,

    /// Exchange transaction charges
    pub exchange_txn_inr: f64,

    /// SEBI charges
    pub sebi_inr: f64,

    /// Stamp duty (0.003% on buy)
    pub stamp_inr: f64,

    /// GST (18% on brokerage + exchange_txn + sebi)
    pub gst_inr: f64,

    /// Total fees
    pub total_inr: f64,

    /// Trade side
    pub side: String,

    /// Fill price
    pub fill_price: f64,

    /// Filled quantity
    pub filled_qty: u32,

    /// Lot size used
    pub lot_size: u32,
}

/// Fee breakdown for round-trip (entry + exit)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundTripFeeBreakdown {
    pub entry_fees: FillFeeBreakdown,
    pub exit_fees: FillFeeBreakdown,
    pub total_inr: f64,
}

/// Fee ledger record (output to fee_ledger.jsonl)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeLedgerRecord {
    /// Schema identifier
    #[serde(rename = "schema")]
    pub schema: &'static str,

    /// Schema revision
    pub schema_rev: u32,

    /// Intent ID for joining with fills/routing_decisions
    pub intent_id: Option<String>,

    /// Order ID from exchange
    pub order_id: Option<String>,

    /// UTC timestamp
    pub ts_utc: String,

    /// Symbol traded
    pub symbol: String,

    /// Trade side
    pub side: String,

    /// Fee breakdown
    #[serde(flatten)]
    pub fees: FillFeeBreakdown,
}

impl FeeLedgerRecord {
    pub const SCHEMA: &'static str = "quantlaxmi.ledger.fees.v1";
    pub const SCHEMA_REV: u32 = 1;

    pub fn new(
        intent_id: Option<String>,
        order_id: Option<String>,
        ts_utc: String,
        symbol: String,
        fees: FillFeeBreakdown,
    ) -> Self {
        Self {
            schema: Self::SCHEMA,
            schema_rev: Self::SCHEMA_REV,
            intent_id,
            order_id,
            ts_utc,
            symbol,
            side: fees.side.clone(),
            fees,
        }
    }
}

/// Daily PnL report (quantlaxmi.reports.daily_pnl.v2)
///
/// V2 adds dual-track performance metrics:
/// - intent_edge: execution quality (fill vs mid)
/// - equity_curve: portfolio performance (MTM over time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyPnlReport {
    /// Schema identifier
    pub schema: &'static str,

    /// Schema revision
    pub schema_rev: u32,

    /// Report date (YYYY-MM-DD)
    pub date: String,

    /// Creation timestamp
    pub created_at: String,

    /// Strategy identifier
    pub strategy: String,

    /// Venue (NSE-Zerodha)
    pub venue: String,

    /// Trade counts
    pub counts: DailyPnlCounts,

    /// PnL breakdown
    pub pnl: DailyPnlBreakdown,

    /// Execution quality metrics
    pub execution: DailyExecutionMetrics,

    /// Risk metrics
    pub risk: DailyRiskMetrics,

    /// V2: Dual-track performance metrics
    pub performance: DailyPerformance,

    /// Notes/metadata
    #[serde(default)]
    pub notes: Vec<String>,
}

impl DailyPnlReport {
    pub const SCHEMA_V1: &'static str = "quantlaxmi.reports.daily_pnl.v1";
    pub const SCHEMA_V2: &'static str = "quantlaxmi.reports.daily_pnl.v2";
    pub const SCHEMA_REV: u32 = 2;

    pub fn new(date: &str, strategy: &str, venue: &str) -> Self {
        Self {
            schema: Self::SCHEMA_V2,
            schema_rev: Self::SCHEMA_REV,
            date: date.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            strategy: strategy.to_string(),
            venue: venue.to_string(),
            counts: DailyPnlCounts::default(),
            pnl: DailyPnlBreakdown::default(),
            execution: DailyExecutionMetrics::default(),
            risk: DailyRiskMetrics::default(),
            performance: DailyPerformance {
                schema_rev: 1,
                intent_edge: IntentEdgePerformance::default(),
                equity_curve: EquityCurvePerformance::default(),
            },
            notes: Vec::new(),
        }
    }
}

/// Trade count metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyPnlCounts {
    /// Total orders generated
    pub orders_total: u32,

    /// Total legs submitted
    pub legs_total: u32,

    /// Legs fully filled
    pub legs_filled: u32,

    /// Legs partially filled
    pub legs_partial: u32,

    /// Legs rejected
    pub legs_rejected: u32,

    /// Legs cancelled
    pub legs_cancelled: u32,

    /// Round-trips completed (entry + exit)
    pub round_trips: u32,

    /// Winning round-trips
    pub winners: u32,

    /// Losing round-trips
    pub losers: u32,
}

impl DailyPnlCounts {
    /// Win rate as percentage
    pub fn win_rate(&self) -> f64 {
        if self.round_trips == 0 {
            0.0
        } else {
            (self.winners as f64 / self.round_trips as f64) * 100.0
        }
    }
}

/// PnL breakdown in INR
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyPnlBreakdown {
    /// Gross MTM PnL (before fees)
    pub gross_mtm_inr: f64,

    /// Total fees
    pub fees_total_inr: f64,

    /// Fee breakdown
    pub fees_brokerage_inr: f64,
    pub fees_stt_inr: f64,
    pub fees_exchange_txn_inr: f64,
    pub fees_sebi_inr: f64,
    pub fees_stamp_inr: f64,
    pub fees_gst_inr: f64,

    /// Net MTM PnL (gross - fees)
    pub net_mtm_inr: f64,

    /// Realized PnL (closed positions)
    pub realized_inr: f64,

    /// Unrealized PnL (open positions)
    pub unrealized_inr: f64,
}

impl DailyPnlBreakdown {
    /// Compute net from gross and fees
    pub fn compute_net(&mut self) {
        self.net_mtm_inr = self.gross_mtm_inr - self.fees_total_inr;
    }

    /// Aggregate fee components into total
    pub fn aggregate_fees(&mut self) {
        self.fees_total_inr = self.fees_brokerage_inr
            + self.fees_stt_inr
            + self.fees_exchange_txn_inr
            + self.fees_sebi_inr
            + self.fees_stamp_inr
            + self.fees_gst_inr;
    }
}

/// Execution quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyExecutionMetrics {
    /// Total turnover (notional traded)
    pub turnover_inr: f64,

    /// Slippage p50 (bps)
    pub slippage_bps_p50: f64,

    /// Slippage p90 (bps)
    pub slippage_bps_p90: f64,

    /// Slippage p99 (bps)
    pub slippage_bps_p99: f64,

    /// Fill rate (legs_filled / legs_total)
    pub fill_rate_pct: f64,

    /// Average execution tax (cost of execution vs decision mid)
    pub avg_execution_tax_bps: f64,
}

/// Risk metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyRiskMetrics {
    /// Peak margin used (INR)
    pub peak_margin_inr: f64,

    /// Peak drawdown from session high (INR)
    pub peak_drawdown_inr: f64,

    /// Max single-trade loss (INR)
    pub max_loss_single_trade_inr: f64,

    /// Largest position size (contracts)
    pub max_position_contracts: u32,

    /// Number of delta breaches
    pub delta_breaches: u32,

    /// Number of gamma breaches
    pub gamma_breaches: u32,
}

/// Performance metrics container (v2: two orthogonal tracks)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyPerformance {
    /// Schema revision for performance block
    pub schema_rev: u32,

    /// Intent-edge performance (execution quality)
    pub intent_edge: IntentEdgePerformance,

    /// Equity curve performance (portfolio/MTM)
    pub equity_curve: EquityCurvePerformance,
}

/// Intent-edge performance: "is execution improving?"
///
/// Measures fill quality relative to mid_at_decision, minus fees.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntentEdgePerformance {
    /// Number of filled intents included in metrics
    pub included_intents: u32,

    /// Number of unfilled intents excluded
    pub excluded_unfilled_intents: u32,

    /// Gross execution edge (sum of fill vs mid)
    pub gross_pnl_inr: f64,

    /// Total fees for included intents
    pub fees_inr: f64,

    /// Net execution edge (gross - fees)
    pub net_pnl_inr: f64,

    /// Win rate (0..1)
    pub win_rate: f64,

    /// Profit factor (sum_wins / abs(sum_losses))
    pub profit_factor: f64,

    /// Average winner PnL
    pub avg_winner_inr: f64,

    /// Average loser PnL
    pub avg_loser_inr: f64,

    /// Expectancy per trade
    pub expectancy_inr: f64,

    /// Non-annualized Sharpe over per-intent net PnL stream
    pub sharpe_intent_edge: f64,

    /// Mean net PnL per intent
    pub mean_inr: f64,

    /// Std dev of net PnL
    pub std_inr: f64,
}

/// Equity curve performance: "is the book making money?"
///
/// Measures actual portfolio value over time (MTM + cash).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EquityCurvePerformance {
    /// Bar interval for equity curve (e.g., "1s")
    pub bar_interval: String,

    /// Number of bars in equity curve
    pub bars: u32,

    /// Gross MTM PnL (before fees)
    pub gross_pnl_inr: f64,

    /// Total fees
    pub fees_inr: f64,

    /// Net MTM PnL (gross - fees)
    pub net_pnl_inr: f64,

    /// Maximum drawdown in INR
    pub max_drawdown_inr: f64,

    /// Maximum drawdown as percentage of peak
    pub max_drawdown_pct: f64,

    /// Timestamp of drawdown peak
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dd_peak_ts_utc: Option<String>,

    /// Timestamp of drawdown trough
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dd_trough_ts_utc: Option<String>,

    /// Non-annualized Sharpe computed over equity returns
    pub sharpe_equity_curve: f64,

    /// Sortino ratio
    pub sortino: f64,

    /// Mean return per bar
    pub mean_return: f64,

    /// Std dev of returns
    pub std_return: f64,
}

/// Legacy performance metrics (kept for v1 compatibility)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DailyPerformanceMetrics {
    /// Profit factor (gross_win / gross_loss)
    pub profit_factor: f64,

    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,

    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,

    /// Average winner (INR)
    pub avg_winner_inr: f64,

    /// Average loser (INR)
    pub avg_loser_inr: f64,

    /// Expectancy (INR per trade)
    pub expectancy_inr: f64,
}

/// Aggregate fee ledger into daily report
pub fn aggregate_fees_for_report(fees: &[FillFeeBreakdown]) -> DailyPnlBreakdown {
    let mut pnl = DailyPnlBreakdown::default();

    for f in fees {
        pnl.fees_brokerage_inr += f.brokerage_inr;
        pnl.fees_stt_inr += f.stt_inr;
        pnl.fees_exchange_txn_inr += f.exchange_txn_inr;
        pnl.fees_sebi_inr += f.sebi_inr;
        pnl.fees_stamp_inr += f.stamp_inr;
        pnl.fees_gst_inr += f.gst_inr;
    }

    pnl.aggregate_fees();
    pnl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_fees() {
        let model = IndiaFeeModel::zerodha_nse();

        // Buy 1 lot at ₹200 premium
        let fees = model.estimate_cost(200.0, 1, "Buy");

        // Turnover = 200 × 1 × 15 = 3000
        assert_eq!(fees.turnover_inr, 3000.0);

        // Brokerage = ₹20
        assert_eq!(fees.brokerage_inr, 20.0);

        // STT = 0 on buy
        assert_eq!(fees.stt_inr, 0.0);

        // Exchange txn = 3000 × 0.0003503 ≈ 1.05
        assert!((fees.exchange_txn_inr - 1.0509).abs() < 0.01);

        // SEBI = 3000 × 0.000001 = 0.003
        assert!((fees.sebi_inr - 0.003).abs() < 0.001);

        // Stamp = 3000 × 0.00003 = 0.09
        assert!((fees.stamp_inr - 0.09).abs() < 0.01);

        // GST = 18% × (20 + 1.05 + 0.003) ≈ 3.79
        assert!((fees.gst_inr - 3.79).abs() < 0.1);

        // Total ≈ 25
        assert!(fees.total_inr > 24.0 && fees.total_inr < 26.0);
    }

    #[test]
    fn test_sell_fees_with_stt() {
        let model = IndiaFeeModel::zerodha_nse();

        // Sell 1 lot at ₹200 premium
        let fees = model.estimate_cost(200.0, 1, "Sell");

        // STT = 3000 × 0.001 = 3.0
        assert!((fees.stt_inr - 3.0).abs() < 0.01);

        // No stamp on sell
        assert_eq!(fees.stamp_inr, 0.0);

        // Total should be higher than buy due to STT
        assert!(fees.total_inr > 27.0);
    }

    #[test]
    fn test_round_trip() {
        let model = IndiaFeeModel::zerodha_nse();

        let rt = model.estimate_round_trip(200.0, 220.0, 1);

        // Entry (buy) + exit (sell) fees
        assert!(rt.total_inr > 50.0);

        // Verify components add up
        let expected = rt.entry_fees.total_inr + rt.exit_fees.total_inr;
        assert!((rt.total_inr - expected).abs() < 0.01);
    }

    #[test]
    fn test_large_order() {
        let model = IndiaFeeModel::zerodha_nse();

        // 10 lots at ₹500 premium
        let fees = model.estimate_cost(500.0, 10, "Sell");

        // Turnover = 500 × 10 × 15 = 75,000
        assert_eq!(fees.turnover_inr, 75000.0);

        // STT = 75000 × 0.001 = 75
        assert!((fees.stt_inr - 75.0).abs() < 0.01);

        // Brokerage still flat ₹20
        assert_eq!(fees.brokerage_inr, 20.0);
    }
}
