"""RL + X-Trend Integration — full pipeline with mega features.

Wires the X-Trend backbone (303 mega features → d_hidden via VSN+LSTM+CrossAttention)
with the RL decision layer (Actor-Critic + KellySizer + ThompsonAllocator)
through a multi-asset walk-forward environment wrapping IndiaFnOEnv.

Patterns implemented:
  P1: TFT + Actor-Critic (backbone.py, rl_trading_agent.py, pipeline.py)
  P2: Contextual Thompson Sizing (thompson_sizing.py)
  P3: TFT + Deep Hedging (deep_hedging_pipeline.py)
  P4: Cross-Market India+Crypto (cross_market.py)
  P5: Attention Reward Shaping (attention_reward.py)
  N2: Standalone Deep Hedger for S1/S10 (deep_hedging_pipeline.py)
  N3: RL Optimal Execution + Hawkes Stopping (execution_pipeline.py)
  N4: Avellaneda-Stoikov Market Making (market_making_pipeline.py)
"""
# P1: Core integration (existing)
from .backbone import XTrendBackbone, MegaFeatureAdapter
from .rl_trading_agent import RLTradingAgent, RLConfig
from .integrated_env import (
    IntegratedTradingEnv,
    AssetCostModel,
    IndiaCostModel,
    CryptoCostModel,
    build_cost_models,
    is_india_trading_day,
    compute_overnight_gap,
    INDIA_SYMBOLS as INTEGRATED_INDIA_SYMBOLS,
    CRYPTO_SYMBOLS as INTEGRATED_CRYPTO_SYMBOLS,
    ALL_SYMBOLS as INTEGRATED_ALL_SYMBOLS,
    CRYPTO_FEE_RATE as INTEGRATED_CRYPTO_FEE_RATE,
)
from .pipeline import IntegratedPipeline, run_integrated_backtest

# P2: Thompson sizing
from .thompson_sizing import (
    ThompsonSizingAgent,
    ThompsonSizingPipeline,
    GradientBanditSizer,
)

# P3 + N2: Deep hedging
from .deep_hedging_pipeline import (
    TFTDeepHedgingPipeline,
    StandaloneDeepHedger,
)

# P4: Cross-market
from .cross_market import (
    CryptoFeatureAdapter,
    CrossMarketBackbone,
    CrossMarketAllocator,
    CrossMarketPipeline,
)

# P5: Attention reward
from .attention_reward import (
    AttentionRewardShaper,
    AttentionShapedEnv,
)

# N3: Execution + Hawkes stopping
from .execution_pipeline import (
    ExecutionCalibrator,
    OptimalExecutionPipeline,
    HawkesOptimalStopping,
)

# N4: Market making
from .market_making_pipeline import (
    CryptoMMCalibrator,
    MarketMakingPipeline,
)

__all__ = [
    # P1
    "XTrendBackbone",
    "MegaFeatureAdapter",
    "RLTradingAgent",
    "RLConfig",
    "IntegratedTradingEnv",
    "AssetCostModel",
    "IndiaCostModel",
    "CryptoCostModel",
    "build_cost_models",
    "is_india_trading_day",
    "compute_overnight_gap",
    "IntegratedPipeline",
    "run_integrated_backtest",
    # P2
    "ThompsonSizingAgent",
    "ThompsonSizingPipeline",
    "GradientBanditSizer",
    # P3 + N2
    "TFTDeepHedgingPipeline",
    "StandaloneDeepHedger",
    # P4
    "CryptoFeatureAdapter",
    "CrossMarketBackbone",
    "CrossMarketAllocator",
    "CrossMarketPipeline",
    # P5
    "AttentionRewardShaper",
    "AttentionShapedEnv",
    # N3
    "ExecutionCalibrator",
    "OptimalExecutionPipeline",
    "HawkesOptimalStopping",
    # N4
    "CryptoMMCalibrator",
    "MarketMakingPipeline",
]
