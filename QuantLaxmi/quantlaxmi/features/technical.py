"""Technical analysis feature transforms.

Each class produces one or more columns from OHLCV data using only
historical (backward-looking) computations.  All are pure: they read
from the input DataFrame and return a new DataFrame of features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlaxmi.features.base import Feature


@dataclass(frozen=True)
class RSI(Feature):
    """Relative Strength Index."""

    window: int = 14

    @property
    def name(self) -> str:
        return f"rsi_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / self.window, min_periods=self.window).mean()
        avg_loss = loss.ewm(alpha=1 / self.window, min_periods=self.window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        out = pd.DataFrame(index=df.index)
        out["value"] = rsi
        out["overbought"] = (rsi > 70).astype(np.int8)
        out["oversold"] = (rsi < 30).astype(np.int8)
        return out


@dataclass(frozen=True)
class BollingerBands(Feature):
    """Bollinger Bands — bandwidth and %B."""

    window: int = 20
    num_std: float = 2.0

    @property
    def name(self) -> str:
        return f"bb_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()

        upper = ma + self.num_std * std
        lower = ma - self.num_std * std

        out = pd.DataFrame(index=df.index)
        out["bandwidth"] = (upper - lower) / ma
        out["pct_b"] = (close - lower) / (upper - lower).replace(0, np.nan)
        return out


@dataclass(frozen=True)
class ATR(Feature):
    """Average True Range."""

    window: int = 14

    @property
    def name(self) -> str:
        return f"atr_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h, l, c = df["High"], df["Low"], df["Close"]
        prev_c = c.shift(1)

        tr = pd.concat(
            [h - l, (h - prev_c).abs(), (l - prev_c).abs()],
            axis=1,
        ).max(axis=1)

        out = pd.DataFrame(index=df.index)
        out["value"] = tr.rolling(self.window).mean()
        out["normalised"] = out["value"] / c  # ATR as fraction of price
        return out


@dataclass(frozen=True)
class SuperTrend(Feature):
    """SuperTrend indicator — trend direction and support/resistance level."""

    period: int = 14
    multiplier: float = 3.0

    @property
    def name(self) -> str:
        return f"supertrend_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h, l, c = df["High"].values, df["Low"].values, df["Close"].values
        prev_c = np.roll(c, 1)
        prev_c[0] = np.nan

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr = pd.Series(tr, index=df.index).rolling(self.period).mean().values

        hl2 = (h + l) / 2.0
        upper_basic = hl2 + self.multiplier * atr
        lower_basic = hl2 - self.multiplier * atr

        n = len(df)
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        supertrend = np.full(n, np.nan)
        direction = np.zeros(n, dtype=np.int8)

        # Find first valid bar (where ATR is available)
        start = self.period  # first non-NaN ATR index
        if start >= n:
            out = pd.DataFrame(index=df.index)
            out["direction"] = np.nan
            out["value"] = np.nan
            out["distance"] = np.nan
            return out

        upper_band[start] = upper_basic[start]
        lower_band[start] = lower_basic[start]
        supertrend[start] = upper_basic[start]
        direction[start] = -1

        for i in range(start + 1, n):
            # Upper band
            if upper_basic[i] < upper_band[i - 1] or c[i - 1] > upper_band[i - 1]:
                upper_band[i] = upper_basic[i]
            else:
                upper_band[i] = upper_band[i - 1]

            # Lower band
            if lower_basic[i] > lower_band[i - 1] or c[i - 1] < lower_band[i - 1]:
                lower_band[i] = lower_basic[i]
            else:
                lower_band[i] = lower_band[i - 1]

            # Direction and value
            if supertrend[i - 1] == upper_band[i - 1]:
                if c[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
            else:
                if c[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1

        # Mark warmup period as NaN
        dir_float = direction.astype(float)
        dir_float[:start] = np.nan

        out = pd.DataFrame(index=df.index)
        out["direction"] = dir_float
        out["value"] = supertrend
        out["distance"] = (c - supertrend) / c  # normalised distance
        return out


@dataclass(frozen=True)
class Stochastic(Feature):
    """Stochastic oscillator — %K and %D."""

    k_window: int = 14
    d_window: int = 3

    @property
    def name(self) -> str:
        return f"stoch_{self.k_window}"

    @property
    def lookback(self) -> int:
        return self.k_window + self.d_window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        low_min = df["Low"].rolling(self.k_window).min()
        high_max = df["High"].rolling(self.k_window).max()

        denom = (high_max - low_min).replace(0, np.nan)
        fast_k = 100.0 * (df["Close"] - low_min) / denom
        slow_k = fast_k.rolling(self.d_window).mean()
        slow_d = slow_k.rolling(self.d_window).mean()

        out = pd.DataFrame(index=df.index)
        out["fast_k"] = fast_k
        out["slow_k"] = slow_k
        out["slow_d"] = slow_d
        out["k_above_d"] = (slow_k > slow_d).astype(np.int8)
        return out
