"""QLX — QuantLaxmi Python research engine.

Lookahead-safe, immutable-data, cost-aware backtesting for quantitative
trading research.
"""

from quantlaxmi.core.base.timeguard import TimeGuard, LookaheadError


def __getattr__(name: str):
    if name == "ResearchEngine":
        from quantlaxmi.core.pipeline.engine import ResearchEngine
        return ResearchEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TimeGuard", "LookaheadError", "ResearchEngine"]
