//! QuantLaxmi Options Engine - The Master Orchestrator
//!
//! This is the crown jewel: a fully integrated options trading engine that combines:
//!
//! 1. **Grassmann Manifold Regime Detection** - Geometric market state classification
//! 2. **Ramanujan Periodicity Detection** - HFT/MM activity identification
//! 3. **Complete Greeks Engine** - All first and second-order Greeks
//! 4. **Volatility Surface Analysis** - Skew, term structure, IV percentile
//! 5. **PCR Sentiment Analysis** - Put-call ratio signals with divergence detection
//! 6. **Multi-Leg Strategy Selection** - 15+ strategies with optimal timing
//! 7. **Risk Management** - Position limits, Greeks constraints
//!
//! ## Philosophy
//!
//! "The market is a voting machine in the short run, but a weighing machine in the long run."
//! - Benjamin Graham
//!
//! This engine combines:
//! - **Microstructure Edge**: Trade when conditions favor you (regime + periodicity)
//! - **Volatility Edge**: Exploit IV/RV premium and skew anomalies
//! - **Sentiment Edge**: Use PCR extremes as contrarian signals
//! - **Structural Edge**: Select optimal strategy structure for the regime
//!
//! ## The Secret Sauce
//!
//! 1. Only trade when ALL signals align (regime + vol + PCR + edge)
//! 2. Position size inversely proportional to IV (larger when IV high, premium rich)
//! 3. Prefer defined-risk structures in uncertain regimes
//! 4. Use Ramanujan periodicity to avoid HFT-dominated periods
//! 5. Greeks-based adjustment triggers (delta, gamma thresholds)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use quantlaxmi_regime::{
    MicrostructurePeriodicity, PeriodicityFeatures, RegimeEngine, RegimeEngineConfig, RegimeLabel,
};

use crate::greeks::{Greeks, OptionParams, OptionType, PortfolioGreeks};
use crate::pcr::{calculate_max_pain, OptionData, PCRMetrics, PCRSignal, PCRTracker};
use crate::strategies::{ConstructedStrategy, ResolvedLeg, StrategyParams, StrategyType};
use crate::strategy_selector::{
    MarketState, RiskConstraints, StrategyRecommendation, StrategySelector, TradingAction,
    TradingDecision,
};
use crate::vol_surface::{VolRegime, VolSmile, VolSurface};

/// Configuration for the Options Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Underlying symbol
    pub symbol: String,
    /// Lot size (contract multiplier)
    pub lot_size: u32,
    /// Risk-free rate
    pub risk_free_rate: f64,
    /// Dividend yield
    pub dividend_yield: f64,
    /// Maximum positions
    pub max_positions: u32,
    /// Maximum loss per position
    pub max_loss_per_position: f64,
    /// Maximum total portfolio delta
    pub max_portfolio_delta: f64,
    /// Minimum IV percentile to sell premium
    pub min_iv_percentile_sell: f64,
    /// Maximum IV percentile to buy premium
    pub max_iv_percentile_buy: f64,
    /// Minimum score to trade
    pub min_strategy_score: f64,
    /// Enable Ramanujan HFT detection
    pub ramanujan_enabled: bool,
    /// Block on detected HFT activity
    pub block_on_hft: bool,
    /// Enable PCR signals
    pub pcr_enabled: bool,
    /// PCR lookback for percentile
    pub pcr_lookback: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            symbol: "NIFTY".into(),
            lot_size: 50,
            risk_free_rate: 0.065, // India 10Y ~6.5%
            dividend_yield: 0.012, // Nifty ~1.2%
            max_positions: 5,
            max_loss_per_position: 25000.0, // ₹25k max loss
            max_portfolio_delta: 500.0,
            min_iv_percentile_sell: 60.0,
            max_iv_percentile_buy: 40.0,
            min_strategy_score: 50.0,
            ramanujan_enabled: true,
            block_on_hft: true,
            pcr_enabled: true,
            pcr_lookback: 100,
        }
    }
}

/// The QuantLaxmi Options Engine.
pub struct OptionsEngine {
    /// Configuration
    config: EngineConfig,
    /// Grassmann regime detector
    regime_engine: RegimeEngine,
    /// Ramanujan periodicity detector
    periodicity: Option<MicrostructurePeriodicity>,
    /// Volatility surface
    vol_surface: VolSurface,
    /// PCR tracker
    pcr_tracker: PCRTracker,
    /// Strategy selector
    strategy_selector: StrategySelector,
    /// Current positions
    positions: HashMap<String, Position>,
    /// Portfolio Greeks
    portfolio_greeks: PortfolioGreeks,
    /// Current spot price
    spot: f64,
    /// Last regime label
    last_regime: RegimeLabel,
    /// Last periodicity features
    last_periodicity: Option<PeriodicityFeatures>,
    /// Historical IV for percentile
    iv_history: Vec<f64>,
    /// Realized volatility tracker
    realized_vol: RealizedVolTracker,
    /// Decision log
    decision_log: Vec<EngineDecision>,
}

/// A position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Position ID
    pub id: String,
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Entry timestamp
    pub entry_ts: DateTime<Utc>,
    /// Entry spot
    pub entry_spot: f64,
    /// Legs
    pub legs: Vec<PositionLeg>,
    /// Net premium (debit positive, credit negative)
    pub net_premium: f64,
    /// Current P&L
    pub current_pnl: f64,
    /// Current Greeks
    pub greeks: PortfolioGreeks,
    /// Max profit
    pub max_profit: f64,
    /// Max loss
    pub max_loss: f64,
}

/// A leg in a position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLeg {
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry: f64,
    pub quantity: i32,
    pub entry_price: f64,
    pub current_price: f64,
}

/// Realized volatility tracker.
#[derive(Debug, Clone)]
pub struct RealizedVolTracker {
    returns: Vec<f64>,
    max_size: usize,
}

impl RealizedVolTracker {
    fn new(max_size: usize) -> Self {
        Self {
            returns: Vec::with_capacity(max_size),
            max_size,
        }
    }

    fn update(&mut self, ret: f64) {
        self.returns.push(ret);
        if self.returns.len() > self.max_size {
            self.returns.remove(0);
        }
    }

    fn realized_vol(&self, annualize_factor: f64) -> f64 {
        if self.returns.len() < 10 {
            return 0.0;
        }

        let mean = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let variance = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (self.returns.len() - 1) as f64;

        variance.sqrt() * annualize_factor
    }
}

/// Engine decision record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineDecision {
    pub timestamp: DateTime<Utc>,
    pub action: TradingAction,
    pub strategy: Option<StrategyType>,
    pub score: f64,
    pub regime: RegimeLabel,
    pub vol_regime: VolRegime,
    pub iv_percentile: f64,
    pub pcr: f64,
    pub reasoning: Vec<String>,
}

impl OptionsEngine {
    /// Create new options engine.
    pub fn new(config: EngineConfig) -> Self {
        let regime_config = RegimeEngineConfig {
            n_features: 6,
            subspace_dim: 3,
            window_size: 32, // Reduced from 64 to work with 60-min warmup
            cpd_threshold_mantissa: 5000,
            cpd_drift_mantissa: 100,
            distance_exponent: -4,
            min_confidence_mantissa: 2000,
        };

        let periodicity = if config.ramanujan_enabled {
            Some(MicrostructurePeriodicity::new())
        } else {
            None
        };

        let risk_constraints = RiskConstraints {
            max_loss: config.max_loss_per_position,
            max_delta: config.max_portfolio_delta,
            ..Default::default()
        };

        Self {
            config: config.clone(),
            regime_engine: RegimeEngine::new(regime_config),
            periodicity,
            vol_surface: VolSurface::new(0.0),
            pcr_tracker: PCRTracker::new(config.pcr_lookback, 5),
            strategy_selector: StrategySelector::new(risk_constraints),
            positions: HashMap::new(),
            portfolio_greeks: PortfolioGreeks::new(),
            spot: 0.0,
            last_regime: RegimeLabel::Unknown,
            last_periodicity: None,
            iv_history: Vec::new(),
            realized_vol: RealizedVolTracker::new(252), // ~1 year
            decision_log: Vec::new(),
        }
    }

    /// Process a market tick.
    pub fn on_tick(
        &mut self,
        ts: DateTime<Utc>,
        spot: f64,
        features: &quantlaxmi_regime::FeatureVector,
    ) {
        // Update spot
        let prev_spot = self.spot;
        self.spot = spot;
        self.vol_surface.spot = spot;

        // Update realized volatility
        if prev_spot > 0.0 {
            let ret = (spot / prev_spot).ln();
            self.realized_vol.update(ret);
        }

        // Update regime detector
        let (_subspace_event, shift_event, label_event) =
            self.regime_engine
                .process(ts, &self.config.symbol, features);

        if let Some(label_event) = label_event {
            // Parse regime_id string to RegimeLabel
            self.last_regime = match label_event.regime_id.as_str() {
                "quiet" => RegimeLabel::Quiet,
                "mean_reversion_chop" => RegimeLabel::MeanReversionChop,
                "trend_impulse" => RegimeLabel::TrendImpulse,
                "liquidity_drought" => RegimeLabel::LiquidityDrought,
                "event_shock" => RegimeLabel::EventShock,
                _ => RegimeLabel::Unknown,
            };
        }

        // Update periodicity detector
        if let Some(ref mut periodicity) = self.periodicity {
            let ready = periodicity.update(
                features.mid_return.mantissa,
                features.imbalance.mantissa,
                features.spread_bps.mantissa,
            );
            if ready {
                self.last_periodicity = Some(periodicity.detect());
            }
        }
    }

    /// Process option chain update.
    pub fn on_chain_update(&mut self, ts: DateTime<Utc>, options: &[OptionData], spot: f64) {
        self.spot = spot;
        self.vol_surface.spot = spot;

        // Update PCR
        let pcr_metrics = PCRMetrics::from_chain(options);
        self.pcr_tracker.update(ts.timestamp(), pcr_metrics);

        // Calculate max pain
        let _max_pain = calculate_max_pain(options, spot);

        // Update IV history
        if !options.is_empty() {
            // Find ATM IV
            let atm_options: Vec<_> = options
                .iter()
                .filter(|o| (o.strike - spot).abs() < spot * 0.02)
                .collect();

            if !atm_options.is_empty() {
                let iv_values: Vec<f64> = atm_options
                    .iter()
                    .filter_map(|o| {
                        let result = crate::vol_surface::calculate_iv(
                            o.last_price,
                            spot,
                            o.strike,
                            o.expiry_dte as f64 / 365.0,
                            self.config.risk_free_rate,
                            match o.option_type {
                                crate::pcr::OptionDataType::Call => OptionType::Call,
                                crate::pcr::OptionDataType::Put => OptionType::Put,
                            },
                        );
                        result.map(|r| r.iv)
                    })
                    .collect();

                if !iv_values.is_empty() {
                    let avg_iv = iv_values.iter().sum::<f64>() / iv_values.len() as f64;

                    // Sanity check - IV should be reasonable (1% to 200%)
                    if avg_iv > 0.01 && avg_iv < 2.0 {
                        // Update current ATM IV
                        self.vol_surface.current_atm_iv = avg_iv;

                        self.vol_surface.update_iv_history(avg_iv, 252);
                        self.iv_history.push(avg_iv);
                        if self.iv_history.len() > 252 {
                            self.iv_history.remove(0);
                        }
                    }
                }
            }
        }
    }

    /// Get trading decision.
    pub fn decide(&mut self, ts: DateTime<Utc>) -> TradingDecision {
        // Build market state
        let market_state = self.build_market_state();

        // Check if we should trade at all
        let (should_trade, block_reasons) = self.should_trade(&market_state);

        if !should_trade {
            return TradingDecision {
                action: TradingAction::Wait,
                strategy: None,
                confidence: 0.0,
                reasoning: block_reasons,
                timestamp: ts.timestamp(),
            };
        }

        // Update strategy selector
        self.strategy_selector
            .update_market_state(market_state.clone());
        self.strategy_selector
            .update_portfolio(self.portfolio_greeks.clone());

        // Get recommendations
        let recommendations = self.strategy_selector.recommend(3);

        if recommendations.is_empty() {
            return TradingDecision {
                action: TradingAction::Wait,
                strategy: None,
                confidence: 0.0,
                reasoning: vec!["No suitable strategies found".into()],
                timestamp: ts.timestamp(),
            };
        }

        let top_rec = &recommendations[0];

        // Check score threshold
        if top_rec.score < self.config.min_strategy_score {
            return TradingDecision {
                action: TradingAction::Wait,
                strategy: Some(top_rec.clone()),
                confidence: top_rec.score / 100.0,
                reasoning: vec![format!(
                    "Score {:.1} below threshold {:.1}",
                    top_rec.score, self.config.min_strategy_score
                )],
                timestamp: ts.timestamp(),
            };
        }

        // Log decision
        self.log_decision(ts, TradingAction::Enter, top_rec, &market_state);

        TradingDecision {
            action: TradingAction::Enter,
            strategy: Some(top_rec.clone()),
            confidence: top_rec.score / 100.0,
            reasoning: top_rec.reasoning.clone(),
            timestamp: ts.timestamp(),
        }
    }

    /// Check if we should trade.
    fn should_trade(&self, state: &MarketState) -> (bool, Vec<String>) {
        let mut reasons = Vec::new();

        // Check HFT activity
        if self.config.block_on_hft {
            if let Some(ref features) = self.last_periodicity {
                if features.hft_likely() {
                    reasons.push("Blocked: HFT activity detected (Ramanujan)".into());
                    return (false, reasons);
                }
            }
        }

        // Check position limits
        if self.positions.len() >= self.config.max_positions as usize {
            reasons.push(format!(
                "Blocked: Max positions ({}) reached",
                self.config.max_positions
            ));
            return (false, reasons);
        }

        // Check portfolio delta
        if self.portfolio_greeks.total.delta.abs() > self.config.max_portfolio_delta {
            reasons.push(format!(
                "Blocked: Portfolio delta ({:.1}) exceeds limit ({:.1})",
                self.portfolio_greeks.total.delta.abs(),
                self.config.max_portfolio_delta
            ));
            return (false, reasons);
        }

        // Check unknown regime
        if matches!(state.regime, RegimeLabel::Unknown) {
            // Allow but with caution
            reasons.push("Caution: Unknown regime".into());
        }

        (true, reasons)
    }

    /// Build current market state.
    fn build_market_state(&self) -> MarketState {
        let iv_percentile = self.vol_surface.iv_percentile;
        let atm_iv = self.vol_surface.current_atm_iv;
        let realized_vol = self.realized_vol.realized_vol((252.0_f64).sqrt());

        let pcr_composite = self.pcr_tracker.composite_signal();
        let nearest_dte = self.vol_surface.smiles.keys().next().copied().unwrap_or(30);

        MarketState {
            spot: self.spot,
            regime: self.last_regime.clone(),
            vol_regime: VolRegime::from_percentile(iv_percentile),
            iv_percentile,
            atm_iv,
            skew: self.vol_surface.average_skew(),
            pcr_signal: pcr_composite.signal,
            pcr_value: pcr_composite.current_pcr,
            nearest_dte,
            expected_move_pct: MarketState::calculate_expected_move(atm_iv, nearest_dte),
            realized_vol,
            vol_risk_premium: atm_iv - realized_vol,
        }
    }

    /// Log a decision.
    fn log_decision(
        &mut self,
        ts: DateTime<Utc>,
        action: TradingAction,
        rec: &StrategyRecommendation,
        state: &MarketState,
    ) {
        self.decision_log.push(EngineDecision {
            timestamp: ts,
            action,
            strategy: Some(rec.strategy),
            score: rec.score,
            regime: state.regime.clone(),
            vol_regime: state.vol_regime,
            iv_percentile: state.iv_percentile,
            pcr: state.pcr_value,
            reasoning: rec.reasoning.clone(),
        });
    }

    /// Get current portfolio Greeks.
    pub fn portfolio_greeks(&self) -> &PortfolioGreeks {
        &self.portfolio_greeks
    }

    /// Get position count.
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }

    /// Get recent decisions.
    pub fn recent_decisions(&self, count: usize) -> &[EngineDecision] {
        let start = self.decision_log.len().saturating_sub(count);
        &self.decision_log[start..]
    }

    /// Get engine status.
    pub fn status(&self) -> EngineStatus {
        let pcr = self.pcr_tracker.composite_signal();

        EngineStatus {
            spot: self.spot,
            regime: self.last_regime.clone(),
            vol_regime: VolRegime::from_percentile(self.vol_surface.iv_percentile),
            iv_percentile: self.vol_surface.iv_percentile,
            atm_iv: self.vol_surface.current_atm_iv,
            realized_vol: self.realized_vol.realized_vol((252.0_f64).sqrt()),
            pcr: pcr.current_pcr,
            pcr_signal: pcr.signal,
            hft_detected: self
                .last_periodicity
                .as_ref()
                .map(|p| p.hft_likely())
                .unwrap_or(false),
            mm_detected: self
                .last_periodicity
                .as_ref()
                .map(|p| p.market_maker_likely())
                .unwrap_or(false),
            position_count: self.positions.len(),
            portfolio_delta: self.portfolio_greeks.total.delta,
            portfolio_gamma: self.portfolio_greeks.total.gamma,
            portfolio_theta: self.portfolio_greeks.total.theta,
            portfolio_vega: self.portfolio_greeks.total.vega,
        }
    }

    /// Get debug info about regime detection (count, mean_abs, variance, spread).
    pub fn regime_debug_stats(&self) -> (usize, i64, i64, i64) {
        self.regime_engine.debug_feature_stats()
    }

    /// Get current heuristic regime classification directly.
    pub fn heuristic_regime(&self) -> RegimeLabel {
        self.regime_engine.current_heuristic_regime()
    }

    /// Get number of learned prototypes.
    pub fn prototype_count(&self) -> usize {
        self.regime_engine.prototype_count()
    }
}

/// Engine status snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub spot: f64,
    pub regime: RegimeLabel,
    pub vol_regime: VolRegime,
    pub iv_percentile: f64,
    pub atm_iv: f64,
    pub realized_vol: f64,
    pub pcr: f64,
    pub pcr_signal: PCRSignal,
    pub hft_detected: bool,
    pub mm_detected: bool,
    pub position_count: usize,
    pub portfolio_delta: f64,
    pub portfolio_gamma: f64,
    pub portfolio_theta: f64,
    pub portfolio_vega: f64,
}

impl std::fmt::Display for EngineStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Spot={:.2} | Regime={:?} | IV={:.1}% ({}th pctl) | RV={:.1}% | PCR={:.2} ({:?}) | HFT={} MM={} | Δ={:.1} Γ={:.2} Θ={:.1} V={:.1}",
            self.spot,
            self.regime,
            self.atm_iv * 100.0,
            self.iv_percentile as u32,
            self.realized_vol * 100.0,
            self.pcr,
            self.pcr_signal,
            if self.hft_detected { "Y" } else { "N" },
            if self.mm_detected { "Y" } else { "N" },
            self.portfolio_delta,
            self.portfolio_gamma,
            self.portfolio_theta,
            self.portfolio_vega,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = OptionsEngine::new(EngineConfig::default());
        assert_eq!(engine.position_count(), 0);
    }

    #[test]
    fn test_engine_status() {
        let engine = OptionsEngine::new(EngineConfig::default());
        let status = engine.status();
        assert_eq!(status.position_count, 0);
    }
}
