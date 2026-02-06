//! Strategy Selector - The Brain of the Options Engine
//!
//! Combines all signals to select optimal strategies:
//! - Regime detection (Grassmann + Ramanujan)
//! - Volatility surface analysis
//! - PCR signals
//! - Greeks optimization
//! - Risk constraints
//!
//! This is where the magic happens.

use quantlaxmi_regime::RegimeLabel;
use serde::{Deserialize, Serialize};

use crate::greeks::PortfolioGreeks;
use crate::pcr::{PCRSignal, TradingBias};
use crate::strategies::{DeltaProfile, ExpirySelection, StrategyParams, StrategyType};
use crate::vol_surface::{StrategyBias, VolRegime};

/// Market state snapshot for strategy selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Underlying spot price
    pub spot: f64,
    /// Microstructure regime from Grassmann detection
    pub regime: RegimeLabel,
    /// Volatility regime (IV percentile)
    pub vol_regime: VolRegime,
    /// Current IV percentile (0-100)
    pub iv_percentile: f64,
    /// ATM IV
    pub atm_iv: f64,
    /// IV skew (25d put - 25d call)
    pub skew: f64,
    /// PCR signal
    pub pcr_signal: PCRSignal,
    /// PCR value
    pub pcr_value: f64,
    /// Days to nearest expiry
    pub nearest_dte: u32,
    /// Expected move based on IV
    pub expected_move_pct: f64,
    /// Historical volatility (realized)
    pub realized_vol: f64,
    /// IV - RV spread (positive = IV rich)
    pub vol_risk_premium: f64,
}

impl MarketState {
    /// Calculate expected move from IV.
    pub fn calculate_expected_move(iv: f64, dte: u32) -> f64 {
        // 1 std dev move = IV * sqrt(DTE/365) * 100%
        iv * ((dte as f64) / 365.0).sqrt() * 100.0
    }

    /// Is IV rich compared to realized vol?
    pub fn is_iv_rich(&self) -> bool {
        self.vol_risk_premium > 0.02 // 2% premium
    }

    /// Is IV cheap compared to realized vol?
    pub fn is_iv_cheap(&self) -> bool {
        self.vol_risk_premium < -0.02
    }
}

/// Risk constraints for strategy selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConstraints {
    /// Maximum loss per trade (currency)
    pub max_loss: f64,
    /// Maximum delta exposure
    pub max_delta: f64,
    /// Maximum gamma exposure
    pub max_gamma: f64,
    /// Maximum vega exposure
    pub max_vega: f64,
    /// Maximum theta decay per day
    pub max_theta: f64,
    /// Required probability of profit
    pub min_pop: f64,
    /// Maximum positions open
    pub max_positions: u32,
}

impl Default for RiskConstraints {
    fn default() -> Self {
        Self {
            max_loss: 10000.0,
            max_delta: 500.0,
            max_gamma: 50.0,
            max_vega: 2000.0,
            max_theta: 500.0,
            min_pop: 0.50,
            max_positions: 5,
        }
    }
}

/// Strategy recommendation with scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRecommendation {
    /// Recommended strategy
    pub strategy: StrategyType,
    /// Strategy parameters
    pub params: StrategyParams,
    /// Overall score (0-100)
    pub score: f64,
    /// Component scores
    pub component_scores: ComponentScores,
    /// Reasoning
    pub reasoning: Vec<String>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Component scores breakdown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComponentScores {
    /// Regime alignment score
    pub regime_score: f64,
    /// Volatility alignment score
    pub vol_score: f64,
    /// PCR alignment score
    pub pcr_score: f64,
    /// Risk-adjusted score
    pub risk_score: f64,
    /// Edge score (IV vs RV)
    pub edge_score: f64,
}

/// Risk assessment for a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Maximum loss
    pub max_loss: f64,
    /// Probability of max loss
    pub prob_max_loss: f64,
    /// Expected P&L
    pub expected_pnl: f64,
    /// Risk/reward ratio
    pub risk_reward: f64,
    /// Greeks exposure
    pub greeks: PortfolioGreeks,
}

/// The Strategy Selector engine.
pub struct StrategySelector {
    /// Current market state
    market_state: Option<MarketState>,
    /// Risk constraints
    constraints: RiskConstraints,
    /// Current portfolio Greeks
    portfolio_greeks: PortfolioGreeks,
}

impl StrategySelector {
    /// Create new selector with constraints.
    pub fn new(constraints: RiskConstraints) -> Self {
        Self {
            market_state: None,
            constraints,
            portfolio_greeks: PortfolioGreeks::new(),
        }
    }

    /// Update market state.
    pub fn update_market_state(&mut self, state: MarketState) {
        self.market_state = Some(state);
    }

    /// Update portfolio Greeks.
    pub fn update_portfolio(&mut self, greeks: PortfolioGreeks) {
        self.portfolio_greeks = greeks;
    }

    /// Get top strategy recommendations.
    pub fn recommend(&self, count: usize) -> Vec<StrategyRecommendation> {
        let state = match &self.market_state {
            Some(s) => s,
            None => return vec![],
        };

        let mut candidates = vec![
            self.score_strategy(StrategyType::LongStraddle, state),
            self.score_strategy(StrategyType::ShortStraddle, state),
            self.score_strategy(StrategyType::LongStrangle, state),
            self.score_strategy(StrategyType::ShortStrangle, state),
            self.score_strategy(StrategyType::IronCondor, state),
            self.score_strategy(StrategyType::IronButterfly, state),
            self.score_strategy(StrategyType::LongButterfly, state),
            self.score_strategy(StrategyType::CalendarSpread, state),
            self.score_strategy(StrategyType::JadeLizard, state),
            self.score_strategy(StrategyType::BullCallSpread, state),
            self.score_strategy(StrategyType::BearPutSpread, state),
        ];

        // Sort by score descending (NaN-safe: NaN scores treated as worst)
        candidates.sort_by(|a, b| {
            match (a.score.is_nan(), b.score.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // a (NaN) goes after b
                (false, true) => std::cmp::Ordering::Less,    // a goes before b (NaN)
                (false, false) => b.score.total_cmp(&a.score), // descending order
            }
        });

        // Filter by constraints and return top N
        candidates
            .into_iter()
            .filter(|r| self.passes_constraints(r))
            .take(count)
            .collect()
    }

    /// Score a specific strategy.
    fn score_strategy(
        &self,
        strategy: StrategyType,
        state: &MarketState,
    ) -> StrategyRecommendation {
        let mut scores = ComponentScores::default();
        let mut reasoning = Vec::new();

        // 1. Regime Score (0-25)
        scores.regime_score = self.score_regime(strategy, state, &mut reasoning);

        // 2. Volatility Score (0-25)
        scores.vol_score = self.score_volatility(strategy, state, &mut reasoning);

        // 3. PCR Score (0-25)
        scores.pcr_score = self.score_pcr(strategy, state, &mut reasoning);

        // 4. Edge Score (0-15)
        scores.edge_score = self.score_edge(strategy, state, &mut reasoning);

        // 5. Risk Score (0-10)
        scores.risk_score = self.score_risk(strategy, state, &mut reasoning);

        let total_score = scores.regime_score
            + scores.vol_score
            + scores.pcr_score
            + scores.edge_score
            + scores.risk_score;

        // Build parameters
        let params = self.build_params(strategy, state);

        // Risk assessment
        let risk_assessment = self.assess_risk(strategy, state, &params);

        StrategyRecommendation {
            strategy,
            params,
            score: total_score,
            component_scores: scores,
            reasoning,
            risk_assessment,
        }
    }

    /// Score regime alignment.
    fn score_regime(
        &self,
        strategy: StrategyType,
        state: &MarketState,
        reasoning: &mut Vec<String>,
    ) -> f64 {
        let profile = strategy.greeks_profile();

        match state.regime {
            RegimeLabel::Quiet => {
                // Quiet regime: Sell premium, positive theta
                if matches!(profile.theta, crate::strategies::ThetaProfile::Positive) {
                    reasoning.push("Quiet regime: Positive theta strategies favored".into());
                    20.0
                } else if matches!(profile.theta, crate::strategies::ThetaProfile::Negative) {
                    5.0
                } else {
                    12.0
                }
            }
            RegimeLabel::MeanReversionChop => {
                // Mean reversion: Short gamma, sell premium
                if matches!(profile.gamma, crate::strategies::GammaProfile::Short) {
                    reasoning.push("Mean-reverting regime: Short gamma strategies favored".into());
                    22.0
                } else if matches!(profile.gamma, crate::strategies::GammaProfile::Long) {
                    8.0
                } else {
                    15.0
                }
            }
            RegimeLabel::TrendImpulse => {
                // Trending: Long gamma, directional strategies
                if matches!(profile.gamma, crate::strategies::GammaProfile::Long) {
                    reasoning.push("Trending regime: Long gamma strategies favored".into());
                    22.0
                } else if matches!(profile.delta, DeltaProfile::Bullish)
                    || matches!(profile.delta, DeltaProfile::Bearish)
                {
                    reasoning.push("Trending regime: Directional bias useful".into());
                    18.0
                } else {
                    10.0
                }
            }
            RegimeLabel::LiquidityDrought => {
                // Liquidity drought: Avoid trading, or use defined risk
                reasoning.push("Liquidity drought: Minimal trading recommended".into());
                match strategy {
                    StrategyType::IronCondor | StrategyType::IronButterfly => 12.0,
                    _ => 5.0,
                }
            }
            RegimeLabel::EventShock => {
                // Event shock: Long gamma to capture moves
                if matches!(profile.gamma, crate::strategies::GammaProfile::Long) {
                    reasoning.push("Event shock: Long gamma for potential large moves".into());
                    20.0
                } else {
                    reasoning.push("Event shock: Avoid short gamma".into());
                    5.0
                }
            }
            RegimeLabel::Unknown => {
                // Unknown: Prefer defined risk
                match strategy {
                    StrategyType::IronCondor
                    | StrategyType::IronButterfly
                    | StrategyType::LongButterfly => {
                        reasoning.push("Unknown regime: Defined risk preferred".into());
                        15.0
                    }
                    _ => 10.0,
                }
            }
        }
    }

    /// Score volatility alignment.
    fn score_volatility(
        &self,
        strategy: StrategyType,
        state: &MarketState,
        reasoning: &mut Vec<String>,
    ) -> f64 {
        let profile = strategy.greeks_profile();
        let bias = state.vol_regime.strategy_bias();

        match (bias, &profile.vega) {
            (StrategyBias::SellPremium, crate::strategies::VegaProfile::Short) => {
                reasoning.push(format!(
                    "High IV ({}th percentile): Premium selling favored",
                    state.iv_percentile as u32
                ));
                25.0
            }
            (StrategyBias::BuyPremium, crate::strategies::VegaProfile::Long) => {
                reasoning.push(format!(
                    "Low IV ({}th percentile): Premium buying favored",
                    state.iv_percentile as u32
                ));
                25.0
            }
            (StrategyBias::Neutral, _) => 15.0,
            _ => {
                // Misaligned
                8.0
            }
        }
    }

    /// Score PCR alignment.
    fn score_pcr(
        &self,
        strategy: StrategyType,
        state: &MarketState,
        reasoning: &mut Vec<String>,
    ) -> f64 {
        let profile = strategy.greeks_profile();
        let contrarian_bias = state.pcr_signal.contrarian_bias();

        // For directional strategies, align with contrarian PCR signal
        match (&profile.delta, contrarian_bias) {
            (DeltaProfile::Bullish, TradingBias::Bullish) => {
                reasoning.push(format!(
                    "PCR={:.2} (extreme fear): Contrarian bullish alignment",
                    state.pcr_value
                ));
                22.0
            }
            (DeltaProfile::Bearish, TradingBias::Bearish) => {
                reasoning.push(format!(
                    "PCR={:.2} (complacency): Contrarian bearish alignment",
                    state.pcr_value
                ));
                22.0
            }
            (DeltaProfile::Neutral, _) => {
                // Neutral strategies don't care about PCR direction
                match state.pcr_signal {
                    PCRSignal::Neutral => 15.0,
                    PCRSignal::ExtremeBearish | PCRSignal::ExtremeBullish => {
                        // Extreme PCR = expect reversal, neutral is safer
                        reasoning.push("Extreme PCR: Neutral delta safer".into());
                        18.0
                    }
                    _ => 12.0,
                }
            }
            _ => 10.0,
        }
    }

    /// Score edge (IV vs RV).
    fn score_edge(
        &self,
        strategy: StrategyType,
        state: &MarketState,
        reasoning: &mut Vec<String>,
    ) -> f64 {
        let profile = strategy.greeks_profile();

        if state.is_iv_rich() {
            // IV > RV: Edge in selling
            if matches!(profile.vega, crate::strategies::VegaProfile::Short) {
                reasoning.push(format!(
                    "IV rich (VRP={:.1}%): Edge in selling premium",
                    state.vol_risk_premium * 100.0
                ));
                15.0
            } else {
                5.0
            }
        } else if state.is_iv_cheap() {
            // IV < RV: Edge in buying
            if matches!(profile.vega, crate::strategies::VegaProfile::Long) {
                reasoning.push(format!(
                    "IV cheap (VRP={:.1}%): Edge in buying premium",
                    state.vol_risk_premium * 100.0
                ));
                15.0
            } else {
                5.0
            }
        } else {
            10.0 // Neutral
        }
    }

    /// Score risk characteristics.
    fn score_risk(
        &self,
        strategy: StrategyType,
        _state: &MarketState,
        reasoning: &mut Vec<String>,
    ) -> f64 {
        // Favor defined-risk strategies
        match strategy {
            StrategyType::IronCondor
            | StrategyType::IronButterfly
            | StrategyType::LongButterfly
            | StrategyType::BullCallSpread
            | StrategyType::BearPutSpread => {
                reasoning.push("Defined risk strategy".into());
                10.0
            }
            StrategyType::ShortStraddle | StrategyType::ShortStrangle => {
                reasoning.push("Undefined risk - requires careful management".into());
                5.0
            }
            StrategyType::JadeLizard => {
                reasoning.push("Semi-defined risk (capped on one side)".into());
                7.0
            }
            _ => 8.0,
        }
    }

    /// Build strategy parameters.
    fn build_params(&self, _strategy: StrategyType, state: &MarketState) -> StrategyParams {
        let expiry = if state.nearest_dte <= 14 {
            ExpirySelection::Weekly
        } else if state.nearest_dte <= 45 {
            ExpirySelection::Monthly
        } else {
            ExpirySelection::DTE(state.nearest_dte)
        };

        StrategyParams {
            quantity: 1,
            expiry,
            max_debit: None,
            min_credit: None,
        }
    }

    /// Assess risk for a strategy.
    fn assess_risk(
        &self,
        _strategy: StrategyType,
        state: &MarketState,
        _params: &StrategyParams,
    ) -> RiskAssessment {
        // Simplified risk assessment
        let expected_move = state.expected_move_pct / 100.0 * state.spot;

        RiskAssessment {
            max_loss: expected_move * 2.0, // 2x expected move
            prob_max_loss: 0.05,
            expected_pnl: expected_move * 0.3, // 30% of expected move
            risk_reward: 0.5,
            greeks: PortfolioGreeks::new(),
        }
    }

    /// Check if recommendation passes constraints.
    fn passes_constraints(&self, rec: &StrategyRecommendation) -> bool {
        rec.risk_assessment.max_loss <= self.constraints.max_loss && rec.score >= 40.0
        // Minimum score threshold
    }
}

/// Master decision output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    /// Action to take
    pub action: TradingAction,
    /// Strategy details
    pub strategy: Option<StrategyRecommendation>,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Decision reasoning
    pub reasoning: Vec<String>,
    /// Timestamp
    pub timestamp: i64,
}

/// Trading action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingAction {
    /// Enter new position
    Enter,
    /// Exit existing position
    Exit,
    /// Adjust existing position
    Adjust,
    /// No action - wait
    Wait,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_creation() {
        let selector = StrategySelector::new(RiskConstraints::default());
        assert!(selector.market_state.is_none());
    }

    #[test]
    fn test_recommendation_scoring() {
        let mut selector = StrategySelector::new(RiskConstraints::default());

        let state = MarketState {
            spot: 100.0,
            regime: RegimeLabel::MeanReversionChop,
            vol_regime: VolRegime::HighVol,
            iv_percentile: 85.0,
            atm_iv: 0.30,
            skew: 0.02,
            pcr_signal: PCRSignal::ExtremeBearish,
            pcr_value: 1.4,
            nearest_dte: 30,
            expected_move_pct: 5.0,
            realized_vol: 0.25,
            vol_risk_premium: 0.05,
        };

        selector.update_market_state(state);

        let recommendations = selector.recommend(3);
        assert!(!recommendations.is_empty());

        // In high vol + mean reversion, should favor premium selling
        let top_strategy = &recommendations[0].strategy;
        assert!(matches!(
            top_strategy,
            StrategyType::ShortStraddle
                | StrategyType::ShortStrangle
                | StrategyType::IronCondor
                | StrategyType::IronButterfly
        ));
    }
}
