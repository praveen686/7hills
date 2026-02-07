"""QLX — QuantLaxmi Python research engine.

Lookahead-safe, immutable-data, cost-aware backtesting for quantitative
trading research.
"""

from qlx.core.timeguard import TimeGuard, LookaheadError
from qlx.pipeline.engine import ResearchEngine

__all__ = ["TimeGuard", "LookaheadError", "ResearchEngine"]
