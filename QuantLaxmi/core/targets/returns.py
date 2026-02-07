"""Target transforms — what we're trying to predict.

Unlike features, targets are *explicitly allowed* to look forward.  They
declare ``horizon`` so the pipeline engine can enforce safe CV gaps.

IMPORTANT: sigma's target used ``.rolling(n).mean()`` on the shifted future
return.  This smooths the target and creates serial correlation that inflates
R^2.  ``FutureReturn`` here returns the RAW future return by default.  If
smoothing is desired, it must be opted into explicitly with ``smooth=True``,
and a warning is printed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.base.types import OHLCV


@dataclass(frozen=True)
class FutureReturn:
    """Target = simple future return over ``horizon`` bars.

    y[t] = close[t + horizon] / close[t] - 1

    The last ``horizon`` rows are dropped (their target extends beyond
    the data).
    """

    horizon: int
    smooth: bool = False  # opt-in smoothing with explicit warning

    @property
    def name(self) -> str:
        return f"future_return_{self.horizon}"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    def transform(self, ohlcv: OHLCV) -> pd.Series:
        close = ohlcv.close  # already a copy
        y = close.shift(-self.horizon) / close - 1.0

        if self.smooth:
            warnings.warn(
                f"FutureReturn(smooth=True): applying .rolling({self.horizon}).mean() "
                f"to the target.  This creates serial autocorrelation in y and will "
                f"inflate R^2 / reduce effective sample size.  Use with caution.",
                UserWarning,
                stacklevel=2,
            )
            y = y.rolling(self.horizon).mean()

        # Drop rows where target is undefined (last `horizon` rows, plus
        # any additional NaN from smoothing warmup)
        y = y.dropna()
        y.name = self.name

        return y


@dataclass(frozen=True)
class TripleBarrier:
    """Triple barrier labeling (Lopez de Prado).

    For each bar t, look forward up to ``horizon`` bars.  The label is:
      - +1 if the upper barrier (close + atr * multiplier) is hit first
      - -1 if the lower barrier (close - atr * multiplier) is hit first
      -  0 if neither barrier is hit within the horizon (time stop)

    ATR is computed over ``atr_period`` bars *before* time t (no lookahead
    in the barrier width computation).
    """

    horizon: int
    atr_period: int = 14
    multiplier: float = 2.0

    @property
    def name(self) -> str:
        return f"triple_barrier_{self.horizon}"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {self.multiplier}")

    def transform(self, ohlcv: OHLCV) -> pd.Series:
        df = ohlcv.df
        h, l, c = df["High"].values, df["Low"].values, df["Close"].values

        # Compute ATR (backward-looking only)
        prev_c = np.roll(c, 1)
        prev_c[0] = np.nan
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr = pd.Series(tr).rolling(self.atr_period).mean().values

        n = len(df)
        labels = np.full(n, np.nan)

        for i in range(n - self.horizon):
            if np.isnan(atr[i]):
                continue

            upper = c[i] + atr[i] * self.multiplier
            lower = c[i] - atr[i] * self.multiplier

            # Scan forward window
            label = 0  # default: time stop
            for j in range(i + 1, min(i + 1 + self.horizon, n)):
                if h[j] >= upper:
                    label = 1
                    break
                if l[j] <= lower:
                    label = -1
                    break

            labels[i] = label

        result = pd.Series(labels, index=df.index, name=self.name).dropna()
        return result
