"""Alternative bar aggregation — dollar bars, volume bars, tick bars.

The key insight from Lopez de Prado: fixed-time bars (1m, 1h, 1d) sample
information unevenly because market activity is not uniform over time.
Activity-based bars sample evenly in *information space* by triggering a
new bar whenever cumulative activity crosses a threshold.

All aggregators are pure functions: they accept an OHLCV and return a new one.
The input is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.core.types import OHLCV


@dataclass(frozen=True)
class DollarBarAggregator:
    """Aggregate tick/minute data into dollar bars.

    A new bar is emitted every time cumulative (close * volume) crosses
    ``threshold``.  This produces bars that are roughly equal in terms of
    the dollar value transacted, which gives more bars during active periods
    and fewer during quiet ones.
    """

    threshold: float  # e.g., 90_000_000 for BTC

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

    def transform(self, ohlcv: OHLCV) -> OHLCV:
        """Return a new OHLCV of dollar bars.  Input is not mutated."""
        src = ohlcv.df  # already a copy
        dollar_value = src["Close"].values * src["Volume"].values

        # --- vectorised bar boundary detection --------------------------------
        cumulative = np.cumsum(dollar_value)
        bar_ids = (cumulative // self.threshold).astype(np.int64)

        # Group by bar_id and aggregate
        src["_bar_id"] = bar_ids
        grouped = src.groupby("_bar_id")

        bars = pd.DataFrame(
            {
                "Open": grouped["Open"].first(),
                "High": grouped["High"].max(),
                "Low": grouped["Low"].min(),
                "Close": grouped["Close"].last(),
                "Volume": grouped["Volume"].sum(),
            }
        )

        # Preserve additional numeric columns (Quote volume, Trade count, etc.)
        skip = {"Open", "High", "Low", "Close", "Volume", "_bar_id"}
        for col in src.columns:
            if col not in skip and np.issubdtype(src[col].dtype, np.number):
                bars[col] = grouped[col].sum()

        # Use the *first* timestamp of each group as the bar timestamp
        bars.index = grouped.apply(lambda g: g.index[0])
        bars.index.name = None

        return OHLCV(bars)


@dataclass(frozen=True)
class VolumeBarAggregator:
    """Aggregate into volume bars — new bar every ``threshold`` units traded."""

    threshold: float

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

    def transform(self, ohlcv: OHLCV) -> OHLCV:
        src = ohlcv.df
        cumulative = np.cumsum(src["Volume"].values)
        bar_ids = (cumulative // self.threshold).astype(np.int64)

        src["_bar_id"] = bar_ids
        grouped = src.groupby("_bar_id")

        bars = pd.DataFrame(
            {
                "Open": grouped["Open"].first(),
                "High": grouped["High"].max(),
                "Low": grouped["Low"].min(),
                "Close": grouped["Close"].last(),
                "Volume": grouped["Volume"].sum(),
            }
        )
        bars.index = grouped.apply(lambda g: g.index[0])
        bars.index.name = None

        return OHLCV(bars)
