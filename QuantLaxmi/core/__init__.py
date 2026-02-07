"""QLX — QuantLaxmi Python research engine.

Lookahead-safe, immutable-data, cost-aware backtesting for quantitative
trading research.
"""

from core.base.timeguard import TimeGuard, LookaheadError
from core.pipeline.engine import ResearchEngine

__all__ = ["TimeGuard", "LookaheadError", "ResearchEngine"]
