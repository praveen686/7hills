//! Multi-Leg Options Strategies
//!
//! Comprehensive strategy definitions with:
//! - Straddles, Strangles
//! - Iron Condors, Iron Butterflies
//! - Butterflies, Calendars
//! - Ratio spreads, Back spreads
//! - Jade Lizards, Broken Wing Butterflies
//!
//! Each strategy includes optimal entry conditions and Greeks targets.

use crate::greeks::{Greeks, OptionParams, OptionType, PortfolioGreeks};
use crate::vol_surface::VolRegime;
use serde::{Deserialize, Serialize};

/// Strategy leg definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyLeg {
    /// Leg identifier
    pub id: String,
    /// Option type
    pub option_type: OptionType,
    /// Strike selection relative to spot
    pub strike_selection: StrikeSelection,
    /// Quantity (positive = long, negative = short)
    pub quantity: i32,
    /// Expiry selection
    pub expiry_selection: ExpirySelection,
}

/// How to select the strike.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrikeSelection {
    /// At-the-money (nearest to spot)
    ATM,
    /// Delta-based selection (e.g., 0.25 for 25Δ)
    Delta(f64),
    /// Fixed offset in currency from spot
    FixedOffset(f64),
    /// Percentage offset from spot
    PercentOffset(f64),
    /// Specific strike value
    Absolute(f64),
}

/// How to select the expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpirySelection {
    /// Nearest expiry
    Nearest,
    /// Specific DTE
    DTE(u32),
    /// Weekly (7 DTE)
    Weekly,
    /// Monthly (30 DTE)
    Monthly,
    /// Quarterly (90 DTE)
    Quarterly,
}

/// Strategy type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyType {
    // === Volatility Plays ===
    /// Long ATM call + put (long gamma, long vega)
    LongStraddle,
    /// Short ATM call + put (short gamma, positive theta)
    ShortStraddle,
    /// Long OTM call + put (cheaper than straddle)
    LongStrangle,
    /// Short OTM call + put
    ShortStrangle,

    // === Iron Strategies ===
    /// Short strangle + long wings (defined risk)
    IronCondor,
    /// Short straddle + long wings
    IronButterfly,

    // === Butterfly Spreads ===
    /// Long wing + 2x short body + long wing (call or put)
    LongButterfly,
    /// Opposite of long butterfly
    ShortButterfly,
    /// Asymmetric butterfly for directional bias
    BrokenWingButterfly,

    // === Calendar/Diagonal ===
    /// Same strike, different expiries
    CalendarSpread,
    /// Different strikes, different expiries
    DiagonalSpread,
    /// Short near-term, long far-term ATM
    DoubleCalendar,

    // === Ratio Spreads ===
    /// Long 1 ATM, short 2 OTM (credit, upside risk)
    RatioCallSpread,
    /// Long 1 ATM, short 2 OTM puts
    RatioPutSpread,
    /// Long 2 OTM, short 1 ATM (backspread)
    CallBackspread,
    /// Long 2 OTM puts, short 1 ATM
    PutBackspread,

    // === Exotic/Advanced ===
    /// Short put + short call spread (undefined risk one side)
    JadeLizard,
    /// Short call + long put + long call (collar variant)
    Zebra,
    /// Double diagonal with calendar component
    DoubleDiagonal,

    // === Directional ===
    /// Bull call spread
    BullCallSpread,
    /// Bear put spread
    BearPutSpread,
    /// Bull put spread (credit)
    BullPutSpread,
    /// Bear call spread (credit)
    BearCallSpread,
}

impl StrategyType {
    /// Get the legs for this strategy type.
    pub fn legs(&self, params: &StrategyParams) -> Vec<StrategyLeg> {
        match self {
            StrategyType::LongStraddle => vec![
                StrategyLeg {
                    id: "long_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "long_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::ATM,
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::ShortStraddle => vec![
                StrategyLeg {
                    id: "short_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::ATM,
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::LongStrangle => vec![
                StrategyLeg {
                    id: "long_otm_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.25),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "long_otm_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.25),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::ShortStrangle => vec![
                StrategyLeg {
                    id: "short_otm_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.25),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_otm_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.25),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::IronCondor => vec![
                // Short strangle
                StrategyLeg {
                    id: "short_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.20),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.20),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                // Long wings
                StrategyLeg {
                    id: "long_call_wing".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.10),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "long_put_wing".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.10),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::IronButterfly => vec![
                // Short straddle
                StrategyLeg {
                    id: "short_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::ATM,
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                // Long wings
                StrategyLeg {
                    id: "long_call_wing".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.15),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "long_put_wing".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.15),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::LongButterfly => vec![
                StrategyLeg {
                    id: "long_lower".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.35),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_middle".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: -2 * params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "long_upper".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.15),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::CalendarSpread => vec![
                StrategyLeg {
                    id: "short_near".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: -params.quantity,
                    expiry_selection: ExpirySelection::Weekly,
                },
                StrategyLeg {
                    id: "long_far".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: params.quantity,
                    expiry_selection: ExpirySelection::Monthly,
                },
            ],

            StrategyType::JadeLizard => vec![
                // Short put
                StrategyLeg {
                    id: "short_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.30),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                // Short call spread
                StrategyLeg {
                    id: "short_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.30),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "long_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.15),
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::BullCallSpread => vec![
                StrategyLeg {
                    id: "long_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::ATM,
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_call".into(),
                    option_type: OptionType::Call,
                    strike_selection: StrikeSelection::Delta(0.25),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            StrategyType::BearPutSpread => vec![
                StrategyLeg {
                    id: "long_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::ATM,
                    quantity: params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
                StrategyLeg {
                    id: "short_put".into(),
                    option_type: OptionType::Put,
                    strike_selection: StrikeSelection::Delta(-0.25),
                    quantity: -params.quantity,
                    expiry_selection: params.expiry.clone(),
                },
            ],

            _ => vec![], // Other strategies can be added
        }
    }

    /// Get the Greeks profile for this strategy.
    pub fn greeks_profile(&self) -> GreeksProfile {
        match self {
            StrategyType::LongStraddle => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Long,
                theta: ThetaProfile::Negative,
                vega: VegaProfile::Long,
            },
            StrategyType::ShortStraddle => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Short,
                theta: ThetaProfile::Positive,
                vega: VegaProfile::Short,
            },
            StrategyType::LongStrangle => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Long,
                theta: ThetaProfile::Negative,
                vega: VegaProfile::Long,
            },
            StrategyType::ShortStrangle => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Short,
                theta: ThetaProfile::Positive,
                vega: VegaProfile::Short,
            },
            StrategyType::IronCondor => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Short,
                theta: ThetaProfile::Positive,
                vega: VegaProfile::Short,
            },
            StrategyType::IronButterfly => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Short,
                theta: ThetaProfile::Positive,
                vega: VegaProfile::Short,
            },
            StrategyType::LongButterfly => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Short,
                theta: ThetaProfile::Positive,
                vega: VegaProfile::Short,
            },
            StrategyType::CalendarSpread => GreeksProfile {
                delta: DeltaProfile::Neutral,
                gamma: GammaProfile::Short,    // Near-term gamma sold
                theta: ThetaProfile::Positive, // Near-term theta collected
                vega: VegaProfile::Long,       // Long back-month vega
            },
            StrategyType::BullCallSpread => GreeksProfile {
                delta: DeltaProfile::Bullish,
                gamma: GammaProfile::Mixed,
                theta: ThetaProfile::Mixed,
                vega: VegaProfile::Long,
            },
            StrategyType::BearPutSpread => GreeksProfile {
                delta: DeltaProfile::Bearish,
                gamma: GammaProfile::Mixed,
                theta: ThetaProfile::Mixed,
                vega: VegaProfile::Long,
            },
            _ => GreeksProfile::default(),
        }
    }

    /// Get ideal market conditions for this strategy.
    pub fn ideal_conditions(&self) -> StrategyConditions {
        match self {
            StrategyType::LongStraddle | StrategyType::LongStrangle => StrategyConditions {
                vol_regime: Some(VolRegime::LowVol),
                expect_movement: true,
                expect_vol_expansion: true,
                ideal_dte: 21..=45,
                min_expected_move_pct: 3.0,
            },
            StrategyType::ShortStraddle | StrategyType::ShortStrangle => StrategyConditions {
                vol_regime: Some(VolRegime::HighVol),
                expect_movement: false,
                expect_vol_expansion: false,
                ideal_dte: 30..=60,
                min_expected_move_pct: 0.0,
            },
            StrategyType::IronCondor => StrategyConditions {
                vol_regime: Some(VolRegime::HighVol),
                expect_movement: false,
                expect_vol_expansion: false,
                ideal_dte: 30..=45,
                min_expected_move_pct: 0.0,
            },
            StrategyType::CalendarSpread => StrategyConditions {
                vol_regime: Some(VolRegime::NormalVol),
                expect_movement: false,
                expect_vol_expansion: true, // Want back-month vol to expand
                ideal_dte: 7..=21,          // Near-term leg
                min_expected_move_pct: 0.0,
            },
            _ => StrategyConditions::default(),
        }
    }
}

/// Strategy parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    /// Base quantity per leg
    pub quantity: i32,
    /// Expiry selection
    pub expiry: ExpirySelection,
    /// Maximum debit to pay (for debit strategies)
    pub max_debit: Option<f64>,
    /// Minimum credit to receive (for credit strategies)
    pub min_credit: Option<f64>,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            quantity: 1,
            expiry: ExpirySelection::Monthly,
            max_debit: None,
            min_credit: None,
        }
    }
}

/// Greeks profile expectations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GreeksProfile {
    pub delta: DeltaProfile,
    pub gamma: GammaProfile,
    pub theta: ThetaProfile,
    pub vega: VegaProfile,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaProfile {
    Bullish,
    #[default]
    Neutral,
    Bearish,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GammaProfile {
    Long,
    #[default]
    Short,
    Mixed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThetaProfile {
    Positive,
    #[default]
    Negative,
    Mixed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VegaProfile {
    Long,
    #[default]
    Short,
    Mixed,
}

/// Conditions for strategy selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConditions {
    /// Preferred volatility regime
    pub vol_regime: Option<VolRegime>,
    /// Expecting significant movement?
    pub expect_movement: bool,
    /// Expecting volatility expansion?
    pub expect_vol_expansion: bool,
    /// Ideal DTE range
    pub ideal_dte: std::ops::RangeInclusive<u32>,
    /// Minimum expected move to justify strategy (%)
    pub min_expected_move_pct: f64,
}

impl Default for StrategyConditions {
    fn default() -> Self {
        Self {
            vol_regime: None,
            expect_movement: false,
            expect_vol_expansion: false,
            ideal_dte: 21..=45,
            min_expected_move_pct: 0.0,
        }
    }
}

/// Fully constructed strategy with resolved strikes and prices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructedStrategy {
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Resolved legs with actual strikes
    pub legs: Vec<ResolvedLeg>,
    /// Total debit (positive) or credit (negative) to enter
    pub net_premium: f64,
    /// Maximum loss
    pub max_loss: f64,
    /// Maximum profit
    pub max_profit: f64,
    /// Breakeven points
    pub breakevens: Vec<f64>,
    /// Aggregated Greeks
    pub greeks: PortfolioGreeks,
    /// Probability of profit estimate
    pub pop: f64,
}

/// A leg with resolved values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedLeg {
    /// Leg ID
    pub id: String,
    /// Option type
    pub option_type: OptionType,
    /// Resolved strike price
    pub strike: f64,
    /// Expiry in years
    pub expiry: f64,
    /// DTE
    pub dte: u32,
    /// Quantity (signed)
    pub quantity: i32,
    /// Premium per contract
    pub premium: f64,
    /// Implied volatility
    pub iv: f64,
    /// Greeks
    pub greeks: Greeks,
}

impl ConstructedStrategy {
    /// Calculate probability of profit using normal distribution.
    pub fn calculate_pop(spot: f64, iv: f64, dte: u32, breakevens: &[f64]) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};

        if breakevens.is_empty() {
            return 0.5;
        }

        let t = dte as f64 / 365.0;
        let std_dev = iv * t.sqrt() * spot;
        let normal = Normal::new(spot, std_dev).unwrap_or(Normal::new(0.0, 1.0).unwrap());

        // For simple strategies with 1-2 breakevens
        match breakevens.len() {
            1 => {
                // Single breakeven (directional)
                1.0 - normal.cdf(breakevens[0])
            }
            2 => {
                // Two breakevens (straddle-like)
                normal.cdf(breakevens[1]) - normal.cdf(breakevens[0])
            }
            _ => 0.5, // Complex
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_straddle_legs() {
        let params = StrategyParams::default();
        let legs = StrategyType::LongStraddle.legs(&params);

        assert_eq!(legs.len(), 2);
        assert_eq!(legs[0].option_type, OptionType::Call);
        assert_eq!(legs[1].option_type, OptionType::Put);
        assert_eq!(legs[0].quantity, 1);
        assert_eq!(legs[1].quantity, 1);
    }

    #[test]
    fn test_iron_condor_legs() {
        let params = StrategyParams::default();
        let legs = StrategyType::IronCondor.legs(&params);

        assert_eq!(legs.len(), 4);
        // Should have 2 short and 2 long
        let shorts: i32 = legs
            .iter()
            .filter(|l| l.quantity < 0)
            .map(|l| l.quantity)
            .sum();
        let longs: i32 = legs
            .iter()
            .filter(|l| l.quantity > 0)
            .map(|l| l.quantity)
            .sum();
        assert_eq!(shorts, -2);
        assert_eq!(longs, 2);
    }

    #[test]
    fn test_greeks_profile() {
        let profile = StrategyType::ShortStraddle.greeks_profile();
        assert_eq!(profile.delta, DeltaProfile::Neutral);
        assert_eq!(profile.gamma, GammaProfile::Short);
        assert_eq!(profile.theta, ThetaProfile::Positive);
        assert_eq!(profile.vega, VegaProfile::Short);
    }
}
