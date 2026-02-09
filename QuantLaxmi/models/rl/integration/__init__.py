"""RL + X-Trend Integration — full pipeline with mega features.

Wires the X-Trend backbone (287 mega features → d_hidden via VSN+LSTM+CrossAttention)
with the RL decision layer (Actor-Critic + KellySizer + ThompsonAllocator)
through a multi-asset walk-forward environment wrapping IndiaFnOEnv.
"""
from .backbone import XTrendBackbone, MegaFeatureAdapter
from .rl_trading_agent import RLTradingAgent, RLConfig
from .integrated_env import IntegratedTradingEnv
from .pipeline import IntegratedPipeline, run_integrated_backtest

__all__ = [
    "XTrendBackbone",
    "MegaFeatureAdapter",
    "RLTradingAgent",
    "RLConfig",
    "IntegratedTradingEnv",
    "IntegratedPipeline",
    "run_integrated_backtest",
]
