"""Historical return and momentum features.

All backward-looking: pct_change(n) for various n, momentum crossovers,
regime indicators.  Never touches future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.features.base import Feature


@dataclass(frozen=True)
class HistoricalReturns(Feature):
    """Percentage returns over multiple lookback periods."""

    periods: tuple[int, ...] = (1, 5, 10, 20, 60, 100, 200)

    @property
    def name(self) -> str:
        return "hist_ret"

    @property
    def lookback(self) -> int:
        return max(self.periods)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        out = pd.DataFrame(index=df.index)

        for p in self.periods:
            out[f"pct_{p}"] = close.pct_change(p)

        return out


@dataclass(frozen=True)
class Momentum(Feature):
    """Momentum regime features — cross-period comparisons and run counts.

    These are useful for detecting whether short-term momentum is stronger
    or weaker than long-term, and whether the market is in a trending or
    mean-reverting regime.
    """

    fast: int = 20
    slow: int = 100
    run_window: int = 14

    @property
    def name(self) -> str:
        return f"momentum_{self.fast}_{self.slow}"

    @property
    def lookback(self) -> int:
        return max(self.slow, self.run_window) + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        fast_ret = close.pct_change(self.fast)
        slow_ret = close.pct_change(self.slow)

        out = pd.DataFrame(index=df.index)

        # Momentum crossover: is short-term outpacing long-term?
        out["fast_gt_slow"] = (fast_ret > slow_ret).astype(np.int8)

        # Momentum acceleration: is short-term accelerating?
        out["fast_accel"] = (fast_ret > fast_ret.shift(self.fast)).astype(np.int8)

        # Up/down run: how many consecutive bars of sign-agreement?
        sign = np.sign(close.diff(1))
        out["run_sum"] = sign.rolling(self.run_window).sum()

        # Large move flags (5% threshold relative to slow lookback)
        out["large_up"] = (close > close.shift(self.slow) * 1.05).astype(np.int8)
        out["large_down"] = (close < close.shift(self.slow) * 0.95).astype(np.int8)

        return out
