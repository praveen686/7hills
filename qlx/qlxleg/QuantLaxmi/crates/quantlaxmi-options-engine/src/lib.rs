//! # QuantLaxmi Options Engine
//!
//! A world-class options trading engine combining advanced signal processing,
//! geometric regime detection, and systematic strategy selection.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        QUANTLAXMI OPTIONS ENGINE                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
//! │  │  GRASSMANN   │    │  RAMANUJAN   │    │     VOL      │              │
//! │  │   REGIME     │    │ PERIODICITY  │    │   SURFACE    │              │
//! │  │  DETECTION   │    │  DETECTION   │    │  ANALYSIS    │              │
//! │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
//! │         │                   │                   │                       │
//! │         │    ┌──────────────┴──────────────┐    │                       │
//! │         │    │                              │    │                       │
//! │         ▼    ▼                              ▼    ▼                       │
//! │  ┌─────────────────────────────────────────────────────┐               │
//! │  │                  MARKET STATE                        │               │
//! │  │  • Regime: Quiet | MeanReversion | Trend | VolExp   │               │
//! │  │  • IV Percentile & Risk Premium                      │               │
//! │  │  • PCR Signal & Sentiment                            │               │
//! │  │  • HFT/MM Activity Detection                         │               │
//! │  └─────────────────────────┬───────────────────────────┘               │
//! │                            │                                            │
//! │                            ▼                                            │
//! │  ┌─────────────────────────────────────────────────────┐               │
//! │  │              STRATEGY SELECTOR                       │               │
//! │  │  • Score 15+ strategies on regime alignment          │               │
//! │  │  • Volatility edge (IV vs RV)                        │               │
//! │  │  • PCR contrarian signals                            │               │
//! │  │  • Risk constraints                                  │               │
//! │  └─────────────────────────┬───────────────────────────┘               │
//! │                            │                                            │
//! │                            ▼                                            │
//! │  ┌─────────────────────────────────────────────────────┐               │
//! │  │              TRADING DECISION                        │               │
//! │  │  • Strategy: IronCondor | Straddle | Butterfly | ...│               │
//! │  │  • Position sizing based on IV                       │               │
//! │  │  • Greeks targets: Δ, Γ, Θ, V                        │               │
//! │  │  • Entry/Exit triggers                               │               │
//! │  └─────────────────────────────────────────────────────┘               │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Concepts
//!
//! ### 1. Grassmann Manifold Regime Detection
//!
//! Uses geometric analysis of rolling covariance matrices to detect market regimes:
//! - **Quiet**: Low variance, stable relationships → Sell premium
//! - **Mean Reversion**: Choppy, diffuse eigenvalues → Short gamma
//! - **Trending**: Concentrated eigenvalues → Long gamma, directional
//! - **Vol Expansion**: Increasing subspace distance → Long vega
//!
//! ### 2. Ramanujan Periodicity Detection
//!
//! Detects periodic patterns in microstructure data using Ramanujan sums:
//! - Short periods (2-4 ticks) with high energy → HFT activity
//! - Medium periods (6-16 ticks) → Market maker activity
//! - Used to avoid trading during unfavorable microstructure
//!
//! ### 3. Volatility Surface Analysis
//!
//! - IV percentile ranking for regime classification
//! - Skew analysis (25Δ put - call spread)
//! - Term structure (contango vs backwardation)
//! - IV vs RV premium for edge identification
//!
//! ### 4. PCR (Put-Call Ratio) Analysis
//!
//! - Volume and OI-based PCR
//! - Contrarian signals at extremes (>1.2 or <0.5)
//! - Divergence detection (price vs PCR)
//! - Max pain calculation
//!
//! ### 5. Multi-Leg Strategies
//!
//! **Volatility Plays:**
//! - Long/Short Straddle
//! - Long/Short Strangle
//!
//! **Income/Defined Risk:**
//! - Iron Condor
//! - Iron Butterfly
//! - Long Butterfly
//! - Jade Lizard
//!
//! **Calendar/Diagonal:**
//! - Calendar Spread
//! - Diagonal Spread
//! - Double Calendar
//!
//! **Directional:**
//! - Bull/Bear Call/Put Spreads
//! - Ratio Spreads
//! - Backspreads
//!
//! ## Usage
//!
//! ```rust,ignore
//! use quantlaxmi_options_engine::{OptionsEngine, EngineConfig};
//!
//! // Create engine
//! let config = EngineConfig {
//!     symbol: "NIFTY".into(),
//!     lot_size: 50,
//!     ..Default::default()
//! };
//! let mut engine = OptionsEngine::new(config);
//!
//! // Process market data
//! engine.on_tick(ts, spot, &features);
//! engine.on_chain_update(ts, &option_chain, spot);
//!
//! // Get trading decision
//! let decision = engine.decide(ts);
//! match decision.action {
//!     TradingAction::Enter => {
//!         println!("Enter {:?} with score {:.1}",
//!             decision.strategy.unwrap().strategy,
//!             decision.confidence * 100.0
//!         );
//!     }
//!     TradingAction::Wait => {
//!         println!("Wait: {:?}", decision.reasoning);
//!     }
//!     _ => {}
//! }
//! ```
//!
//! ## The Edge
//!
//! This engine provides edge through:
//!
//! 1. **Regime-Aware Trading**: Only trade strategies aligned with current regime
//! 2. **Volatility Premium**: Exploit IV/RV spread systematically
//! 3. **Microstructure Timing**: Avoid HFT-dominated periods (Ramanujan)
//! 4. **Contrarian Sentiment**: Use PCR extremes as entry signals
//! 5. **Greeks Optimization**: Maintain target exposures
//! 6. **Risk Management**: Defined risk preferred, position limits enforced

pub mod engine;
pub mod greeks;
pub mod pcr;
pub mod strategies;
pub mod strategy_selector;
pub mod vol_surface;
pub mod warmup;

// Re-exports
pub use engine::{EngineConfig, EngineStatus, OptionsEngine, Position};
pub use greeks::{Greeks, OptionParams, OptionType, PortfolioGreeks};
pub use pcr::{
    calculate_max_pain, OptionData, OptionDataType, PCRMetrics, PCRSignal, PCRTracker, TradingBias,
};
pub use strategies::{
    ConstructedStrategy, ExpirySelection, GreeksProfile, StrategyLeg, StrategyParams, StrategyType,
    StrikeSelection,
};
pub use strategy_selector::{
    MarketState, RiskConstraints, StrategyRecommendation, StrategySelector, TradingAction,
    TradingDecision,
};
pub use vol_surface::{calculate_iv, IVResult, VolRegime, VolSmile, VolSurface};
pub use warmup::{
    aggregate_warmup_data, candles_to_feature_vectors, WarmupConfig, WarmupProgress, WarmupState,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_calculation() {
        let params = OptionParams::new(
            100.0,
            100.0,
            0.25, // ATM, 3 months
            0.05,
            0.20, // 5% rate, 20% vol
            OptionType::Call,
        );
        let greeks = Greeks::calculate(&params);

        // ATM call delta ~0.5-0.6
        assert!(greeks.delta > 0.5 && greeks.delta < 0.65);
        // Positive gamma
        assert!(greeks.gamma > 0.0);
        // Negative theta
        assert!(greeks.theta < 0.0);
        // Positive vega
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_engine_integration() {
        let engine = OptionsEngine::new(EngineConfig::default());
        let status = engine.status();

        // Initial state checks
        assert_eq!(status.position_count, 0);
        assert!(status.portfolio_delta.abs() < 0.001);
    }
}
