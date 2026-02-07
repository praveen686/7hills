//! India F&O Fee Model
//!
//! Implements deterministic, auditable fee calculations for NSE/BSE F&O paper trading.
//!
//! ## Budget 2026 Defaults (Feb 1-2, 2026)
//!
//! Per Reuters and official announcements:
//! - **STT Futures**: 0.05% (was 0.02%)
//! - **STT Options**: 0.15% (was 0.10%)
//!
//! ## Fee Components
//!
//! 1. **Brokerage**: Per-order (₹20) or turnover-based (₹10-50/crore)
//! 2. **STT**: Securities Transaction Tax (sell-side for options, configurable)
//! 3. **Exchange Txn**: NSE transaction charges
//! 4. **SEBI**: Turnover fees (0.002% futures, 0.003% options)
//! 5. **GST**: 18% on brokerage + exchange charges
//! 6. **Stamp Duty**: State-specific (buy-side typically)
//!
//! ## Determinism Guarantee
//!
//! All calculations use integer arithmetic (paise).
//! Same inputs → identical outputs byte-for-byte.
//!
//! ## References
//!
//! - NSE Charges: <https://www.nseindia.com/regulations/investor-services-nse-investor-protection-fund>
//! - Zerodha Charges: <https://zerodha.com/charges>
//! - Budget 2026 STT: Reuters, Feb 1, 2026

use serde::{Deserialize, Serialize};

// =============================================================================
// CORE TYPES
// =============================================================================

/// Instrument kind for fee calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstrumentKind {
    /// Index futures (NIFTY, BANKNIFTY futures)
    IndexFuture,
    /// Index options (NIFTY, BANKNIFTY options)
    IndexOption,
}

/// Trade side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Basis for STT calculation on options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OptSttBase {
    /// Apply STT on premium turnover only (Phase 1, simplified)
    #[default]
    PremiumOnly,
    /// Apply STT on intrinsic value at expiry (Phase 2, complex)
    IntrinsicOnExpiry,
}

/// Brokerage model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrokerageModel {
    /// Flat fee per executed order (e.g., Zerodha ₹20/order)
    PerOrder {
        /// Fee in paise per order
        paise_per_order: i64,
        /// Maximum fee in paise per order (cap)
        max_paise_per_order: i64,
    },
    /// Turnover-based fee (e.g., ₹10/crore for futures, ₹50/crore for options)
    TurnoverBased {
        /// Fee in paise per crore of turnover (futures)
        paise_per_crore_fut: i64,
        /// Fee in paise per crore of turnover (options)
        paise_per_crore_opt: i64,
    },
}

impl Default for BrokerageModel {
    fn default() -> Self {
        // Zerodha default: ₹20/order
        Self::PerOrder {
            paise_per_order: 20_00, // ₹20 = 2000 paise
            max_paise_per_order: 20_00,
        }
    }
}

// =============================================================================
// FEE BREAKDOWN
// =============================================================================

/// Itemized fee breakdown for a single fill.
///
/// All values in **paise** (1 rupee = 100 paise).
/// This ensures deterministic integer arithmetic with no float drift.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeesBreakdown {
    /// Brokerage charge
    pub brokerage: i64,
    /// Securities Transaction Tax
    pub stt: i64,
    /// Exchange transaction charge
    pub exchange_txn: i64,
    /// SEBI turnover fees
    pub sebi: i64,
    /// GST (18% on brokerage + exchange charges)
    pub gst: i64,
    /// Stamp duty (state-specific)
    pub stamp: i64,
    /// Slippage cost (if modeled)
    pub slippage: i64,
    /// Total of all fees
    pub total: i64,
}

impl FeesBreakdown {
    /// Compute total from components.
    pub fn compute_total(&mut self) {
        self.total = self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi
            + self.gst
            + self.stamp
            + self.slippage;
    }

    /// Convert total to rupees (for display).
    pub fn total_rupees(&self) -> f64 {
        self.total as f64 / 100.0
    }
}

// =============================================================================
// FEE CONFIGURATION
// =============================================================================

/// India F&O fee configuration.
///
/// All rates in **basis points** (bps). 1 bps = 0.01% = 0.0001.
/// Example: 5 bps = 0.05% = 0.0005.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndiaFnoFeeConfig {
    // -------------------------------------------------------------------------
    // STT (Securities Transaction Tax) - Budget 2026 rates
    // -------------------------------------------------------------------------
    /// STT rate for futures in bps (default: 5 = 0.05%)
    pub stt_fut_rate_bps: u32,
    /// STT rate for options in bps (default: 15 = 0.15%)
    pub stt_opt_rate_bps: u32,
    /// STT calculation base for options
    pub stt_opt_apply_on: OptSttBase,
    /// Apply STT on sell-side only (default: true for intraday)
    pub stt_sell_side_only: bool,

    // -------------------------------------------------------------------------
    // Exchange Transaction Charges (NSE)
    // -------------------------------------------------------------------------
    /// Exchange txn charge for options in bps (default: 0.3 = 0.003%)
    pub exch_txn_opt_rate_bps: u32,
    /// Exchange txn charge for futures in bps (default: 0.2 = 0.002%)
    pub exch_txn_fut_rate_bps: u32,

    // -------------------------------------------------------------------------
    // SEBI Turnover Fees
    // -------------------------------------------------------------------------
    /// SEBI fee for futures in bps (default: 0.2 = 0.002%)
    pub sebi_fut_rate_bps: u32,
    /// SEBI fee for options in bps (default: 0.3 = 0.003%)
    pub sebi_opt_rate_bps: u32,

    // -------------------------------------------------------------------------
    // GST
    // -------------------------------------------------------------------------
    /// GST rate in bps (default: 1800 = 18%)
    pub gst_rate_bps: u32,

    // -------------------------------------------------------------------------
    // Stamp Duty (state-specific)
    // -------------------------------------------------------------------------
    /// Stamp duty for options in bps (default: 3 = 0.03%)
    pub stamp_opt_rate_bps: u32,
    /// Stamp duty for futures in bps (default: 2 = 0.02%)
    pub stamp_fut_rate_bps: u32,
    /// Apply stamp duty on buy-side only (default: true)
    pub stamp_buy_side_only: bool,

    // -------------------------------------------------------------------------
    // Brokerage
    // -------------------------------------------------------------------------
    /// Brokerage model
    pub brokerage_model: BrokerageModel,

    // -------------------------------------------------------------------------
    // Slippage (Phase 1: disabled)
    // -------------------------------------------------------------------------
    /// Slippage model enabled
    pub slippage_enabled: bool,
    /// Slippage factor (k in: slippage = k * spread * qty)
    pub slippage_factor_bps: u32,

    // -------------------------------------------------------------------------
    // Staleness
    // -------------------------------------------------------------------------
    /// Quote staleness threshold in ms (reject fill if exceeded)
    pub staleness_threshold_ms: u32,
}

impl Default for IndiaFnoFeeConfig {
    fn default() -> Self {
        Self {
            // STT - Budget 2026 rates
            stt_fut_rate_bps: 5,  // 0.05%
            stt_opt_rate_bps: 15, // 0.15%
            stt_opt_apply_on: OptSttBase::PremiumOnly,
            stt_sell_side_only: true,

            // Exchange txn (NSE)
            exch_txn_opt_rate_bps: 5, // 0.05% (approx, varies slightly)
            exch_txn_fut_rate_bps: 2, // 0.02%

            // SEBI
            sebi_fut_rate_bps: 1, // 0.001% (₹10/crore)
            sebi_opt_rate_bps: 1, // 0.001%

            // GST
            gst_rate_bps: 1800, // 18%

            // Stamp duty (varies by state, using Maharashtra default)
            stamp_opt_rate_bps: 3, // 0.03%
            stamp_fut_rate_bps: 2, // 0.02%
            stamp_buy_side_only: true,

            // Brokerage (Zerodha default)
            brokerage_model: BrokerageModel::default(),

            // Slippage (Phase 1: disabled)
            slippage_enabled: false,
            slippage_factor_bps: 0,

            // Staleness
            staleness_threshold_ms: 5000,
        }
    }
}

// =============================================================================
// FEE CALCULATOR
// =============================================================================

/// India F&O fee calculator.
///
/// Stateless, deterministic fee computation.
#[derive(Debug, Clone)]
pub struct IndiaFnoFeeCalculator {
    config: IndiaFnoFeeConfig,
}

impl IndiaFnoFeeCalculator {
    /// Create a new calculator with the given config.
    pub fn new(config: IndiaFnoFeeConfig) -> Self {
        Self { config }
    }

    /// Create with default Budget 2026 config.
    pub fn default_budget_2026() -> Self {
        Self::new(IndiaFnoFeeConfig::default())
    }

    /// Get the config reference.
    pub fn config(&self) -> &IndiaFnoFeeConfig {
        &self.config
    }

    /// Calculate fees for a fill.
    ///
    /// # Parameters
    /// - `kind`: Instrument kind (future or option)
    /// - `side`: Trade side (buy or sell)
    /// - `price_paise`: Fill price in paise
    /// - `qty`: Fill quantity (lots)
    /// - `lot_size`: Contract lot size
    /// - `spread_paise`: Bid-ask spread in paise (for slippage)
    ///
    /// # Returns
    /// Itemized fee breakdown in paise.
    pub fn calculate(
        &self,
        kind: InstrumentKind,
        side: Side,
        price_paise: i64,
        qty: i64,
        lot_size: i64,
        spread_paise: i64,
    ) -> FeesBreakdown {
        let mut fees = FeesBreakdown::default();

        // Turnover = price * qty * lot_size (all in paise)
        let turnover_paise = price_paise * qty * lot_size;

        // 1. Brokerage
        fees.brokerage = self.calculate_brokerage(kind, turnover_paise);

        // 2. STT (sell-side only by default)
        fees.stt = self.calculate_stt(kind, side, turnover_paise);

        // 3. Exchange transaction charge
        fees.exchange_txn = self.calculate_exchange_txn(kind, turnover_paise);

        // 4. SEBI fees
        fees.sebi = self.calculate_sebi(kind, turnover_paise);

        // 5. GST (on brokerage + exchange charges)
        let gst_base = fees.brokerage + fees.exchange_txn;
        fees.gst = self.apply_rate_bps(gst_base, self.config.gst_rate_bps);

        // 6. Stamp duty (buy-side only by default)
        fees.stamp = self.calculate_stamp(kind, side, turnover_paise);

        // 7. Slippage (Phase 1: disabled)
        fees.slippage = if self.config.slippage_enabled {
            self.calculate_slippage(spread_paise, qty, lot_size)
        } else {
            0
        };

        // Compute total
        fees.compute_total();

        fees
    }

    /// Calculate brokerage based on model.
    fn calculate_brokerage(&self, kind: InstrumentKind, turnover_paise: i64) -> i64 {
        match self.config.brokerage_model {
            BrokerageModel::PerOrder {
                paise_per_order,
                max_paise_per_order,
            } => paise_per_order.min(max_paise_per_order),

            BrokerageModel::TurnoverBased {
                paise_per_crore_fut,
                paise_per_crore_opt,
            } => {
                // 1 crore = 10^7 rupees = 10^9 paise
                let paise_per_crore = match kind {
                    InstrumentKind::IndexFuture => paise_per_crore_fut,
                    InstrumentKind::IndexOption => paise_per_crore_opt,
                };
                // brokerage = turnover * (paise_per_crore / 10^9)
                // To avoid overflow: (turnover * paise_per_crore) / 10^9
                (turnover_paise * paise_per_crore) / 1_000_000_000
            }
        }
    }

    /// Calculate STT.
    fn calculate_stt(&self, kind: InstrumentKind, side: Side, turnover_paise: i64) -> i64 {
        // If sell-side only and this is buy, return 0
        if self.config.stt_sell_side_only && side == Side::Buy {
            return 0;
        }

        let rate_bps = match kind {
            InstrumentKind::IndexFuture => self.config.stt_fut_rate_bps,
            InstrumentKind::IndexOption => self.config.stt_opt_rate_bps,
        };

        self.apply_rate_bps(turnover_paise, rate_bps)
    }

    /// Calculate exchange transaction charge.
    fn calculate_exchange_txn(&self, kind: InstrumentKind, turnover_paise: i64) -> i64 {
        let rate_bps = match kind {
            InstrumentKind::IndexFuture => self.config.exch_txn_fut_rate_bps,
            InstrumentKind::IndexOption => self.config.exch_txn_opt_rate_bps,
        };

        self.apply_rate_bps(turnover_paise, rate_bps)
    }

    /// Calculate SEBI fees.
    fn calculate_sebi(&self, kind: InstrumentKind, turnover_paise: i64) -> i64 {
        let rate_bps = match kind {
            InstrumentKind::IndexFuture => self.config.sebi_fut_rate_bps,
            InstrumentKind::IndexOption => self.config.sebi_opt_rate_bps,
        };

        self.apply_rate_bps(turnover_paise, rate_bps)
    }

    /// Calculate stamp duty.
    fn calculate_stamp(&self, kind: InstrumentKind, side: Side, turnover_paise: i64) -> i64 {
        // If buy-side only and this is sell, return 0
        if self.config.stamp_buy_side_only && side == Side::Sell {
            return 0;
        }

        let rate_bps = match kind {
            InstrumentKind::IndexFuture => self.config.stamp_fut_rate_bps,
            InstrumentKind::IndexOption => self.config.stamp_opt_rate_bps,
        };

        self.apply_rate_bps(turnover_paise, rate_bps)
    }

    /// Calculate slippage cost.
    fn calculate_slippage(&self, spread_paise: i64, qty: i64, lot_size: i64) -> i64 {
        // slippage = k * spread * qty * lot_size
        // where k = slippage_factor_bps / 10000
        let k = self.config.slippage_factor_bps as i64;
        (k * spread_paise * qty * lot_size) / 10000
    }

    /// Apply a rate in bps to a value.
    /// result = value * rate_bps / 10000
    fn apply_rate_bps(&self, value_paise: i64, rate_bps: u32) -> i64 {
        (value_paise * rate_bps as i64) / 10000
    }
}

// =============================================================================
// FILL REJECTION
// =============================================================================

/// Reason for fill rejection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillRejected {
    /// No executable quote (bid/ask missing or stale)
    NoExecutableQuote { reason: String },
    /// Insufficient quantity at top of book
    InsufficientQuantity { requested: i64, available: i64 },
    /// Quote is stale (age > threshold)
    StaleQuote { age_ms: u32, threshold_ms: u32 },
}

// =============================================================================
// TRADE INTENT
// =============================================================================

/// Trade intent for paper trading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeIntent {
    /// Instrument token
    pub instrument_token: u32,
    /// Trade side
    pub side: Side,
    /// Quantity in lots
    pub qty: i64,
    /// Instrument kind
    pub kind: InstrumentKind,
    /// Lot size (for turnover calculation)
    pub lot_size: i64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_calculator() -> IndiaFnoFeeCalculator {
        IndiaFnoFeeCalculator::default_budget_2026()
    }

    #[test]
    fn test_buy_option_at_ask_no_stt() {
        let calc = default_calculator();

        // Buy 1 lot of NIFTY option at ₹100 (10000 paise), lot size 25
        let fees = calc.calculate(
            InstrumentKind::IndexOption,
            Side::Buy,
            10_000, // ₹100 = 10000 paise
            1,      // 1 lot
            25,     // lot size
            50,     // spread = 50 paise
        );

        // Turnover = 10000 * 1 * 25 = 250000 paise = ₹2500

        // STT should be 0 (sell-side only)
        assert_eq!(fees.stt, 0, "STT should be 0 on buy side");

        // Brokerage = ₹20 = 2000 paise (per order default)
        assert_eq!(fees.brokerage, 2000, "Brokerage should be ₹20");

        // Stamp duty should be applied (buy side)
        // stamp = 250000 * 3 / 10000 = 75 paise
        assert_eq!(fees.stamp, 75, "Stamp duty should be applied on buy");

        // Total should be sum of all components
        assert_eq!(
            fees.total,
            fees.brokerage
                + fees.stt
                + fees.exchange_txn
                + fees.sebi
                + fees.gst
                + fees.stamp
                + fees.slippage
        );
    }

    #[test]
    fn test_sell_option_at_bid_with_stt() {
        let calc = default_calculator();

        // Sell 1 lot of NIFTY option at ₹100 (10000 paise), lot size 25
        let fees = calc.calculate(
            InstrumentKind::IndexOption,
            Side::Sell,
            10_000, // ₹100 = 10000 paise
            1,      // 1 lot
            25,     // lot size
            50,     // spread = 50 paise
        );

        // Turnover = 10000 * 1 * 25 = 250000 paise = ₹2500

        // STT = 250000 * 15 / 10000 = 375 paise = ₹3.75
        assert_eq!(fees.stt, 375, "STT should be 0.15% of turnover on sell");

        // Stamp duty should be 0 (buy-side only)
        assert_eq!(fees.stamp, 0, "Stamp duty should be 0 on sell");

        // Verify STT is non-zero
        assert!(fees.stt > 0, "STT must be non-zero on option sell");
    }

    #[test]
    fn test_sell_future_stt_rate() {
        let calc = default_calculator();

        // Sell 1 lot of NIFTY future at ₹20000 (2000000 paise), lot size 25
        let fees = calc.calculate(
            InstrumentKind::IndexFuture,
            Side::Sell,
            2_000_000, // ₹20000
            1,         // 1 lot
            25,        // lot size
            100,       // spread
        );

        // Turnover = 2000000 * 1 * 25 = 50000000 paise = ₹500000

        // STT = 50000000 * 5 / 10000 = 25000 paise = ₹250
        assert_eq!(
            fees.stt, 25000,
            "STT should be 0.05% of turnover for futures"
        );
    }

    #[test]
    fn test_gst_on_brokerage_and_exchange() {
        let calc = default_calculator();

        let fees = calc.calculate(
            InstrumentKind::IndexOption,
            Side::Buy,
            10_000, // ₹100
            1,
            25,
            50,
        );

        // GST base = brokerage + exchange_txn
        let gst_base = fees.brokerage + fees.exchange_txn;

        // GST = gst_base * 1800 / 10000 = gst_base * 0.18
        let expected_gst = (gst_base * 1800) / 10000;

        assert_eq!(
            fees.gst, expected_gst,
            "GST should be 18% of brokerage + exchange"
        );
    }

    #[test]
    fn test_determinism() {
        let calc = default_calculator();

        // Same input should produce identical output
        let fees1 = calc.calculate(InstrumentKind::IndexOption, Side::Sell, 15_000, 2, 25, 100);
        let fees2 = calc.calculate(InstrumentKind::IndexOption, Side::Sell, 15_000, 2, 25, 100);

        assert_eq!(fees1.brokerage, fees2.brokerage);
        assert_eq!(fees1.stt, fees2.stt);
        assert_eq!(fees1.exchange_txn, fees2.exchange_txn);
        assert_eq!(fees1.sebi, fees2.sebi);
        assert_eq!(fees1.gst, fees2.gst);
        assert_eq!(fees1.stamp, fees2.stamp);
        assert_eq!(fees1.slippage, fees2.slippage);
        assert_eq!(fees1.total, fees2.total);
    }

    #[test]
    fn test_turnover_based_brokerage() {
        let config = IndiaFnoFeeConfig {
            brokerage_model: BrokerageModel::TurnoverBased {
                paise_per_crore_fut: 1_000, // ₹10/crore
                paise_per_crore_opt: 5_000, // ₹50/crore
            },
            ..Default::default()
        };
        let calc = IndiaFnoFeeCalculator::new(config);

        // Option: 1 crore turnover (100000000 paise)
        // price = ₹40000, qty = 1, lot = 25 → turnover = 40000 * 100 * 1 * 25 = 1 crore
        let fees = calc.calculate(
            InstrumentKind::IndexOption,
            Side::Buy,
            4_000_000, // ₹40000
            1,
            25,
            100,
        );

        // Turnover = 4000000 * 1 * 25 = 100000000 paise = 1 crore
        // Brokerage = 100000000 * 5000 / 1000000000 = 500 paise = ₹5
        // Wait, let me recalculate. paise_per_crore_opt = 5000 (₹50)
        // brokerage = turnover * paise_per_crore / 10^9
        // = 100000000 * 5000 / 1000000000 = 500 paise = ₹5
        assert_eq!(fees.brokerage, 500);
    }

    #[test]
    fn test_slippage_when_enabled() {
        let config = IndiaFnoFeeConfig {
            slippage_enabled: true,
            slippage_factor_bps: 50, // 0.5% of spread
            ..Default::default()
        };
        let calc = IndiaFnoFeeCalculator::new(config);

        // spread = 100 paise, qty = 2, lot = 25
        let fees = calc.calculate(
            InstrumentKind::IndexOption,
            Side::Buy,
            10_000,
            2,
            25,
            100, // spread = ₹1
        );

        // slippage = 50 * 100 * 2 * 25 / 10000 = 25 paise = ₹0.25
        assert_eq!(fees.slippage, 25);
    }

    #[test]
    fn test_slippage_disabled_by_default() {
        let calc = default_calculator();

        let fees = calc.calculate(InstrumentKind::IndexOption, Side::Buy, 10_000, 1, 25, 100);

        assert_eq!(fees.slippage, 0, "Slippage should be 0 when disabled");
    }

    #[test]
    fn test_fees_breakdown_total() {
        let calc = default_calculator();

        let fees = calc.calculate(InstrumentKind::IndexOption, Side::Sell, 20_000, 3, 25, 150);

        // Verify total is sum of components
        let computed_total = fees.brokerage
            + fees.stt
            + fees.exchange_txn
            + fees.sebi
            + fees.gst
            + fees.stamp
            + fees.slippage;

        assert_eq!(fees.total, computed_total);
    }

    #[test]
    fn test_budget_2026_stt_rates() {
        let config = IndiaFnoFeeConfig::default();

        // Verify Budget 2026 defaults
        assert_eq!(config.stt_fut_rate_bps, 5, "Futures STT should be 0.05%");
        assert_eq!(config.stt_opt_rate_bps, 15, "Options STT should be 0.15%");
    }

    #[test]
    fn test_zerodha_per_order_default() {
        let config = IndiaFnoFeeConfig::default();

        match config.brokerage_model {
            BrokerageModel::PerOrder {
                paise_per_order, ..
            } => {
                assert_eq!(paise_per_order, 2000, "Zerodha default should be ₹20/order");
            }
            _ => panic!("Default should be PerOrder model"),
        }
    }

    #[test]
    fn test_total_rupees_conversion() {
        let fees = FeesBreakdown {
            brokerage: 2000,
            stt: 375,
            exchange_txn: 125,
            sebi: 25,
            gst: 382,
            stamp: 75,
            slippage: 0,
            total: 2982,
        };

        assert_eq!(fees.total_rupees(), 29.82);
    }
}
