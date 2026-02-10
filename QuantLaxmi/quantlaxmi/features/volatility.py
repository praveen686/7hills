"""Realized volatility features from OHLCV bars.

Contains five estimators with different statistical properties:
  - Close-to-close (standard, baseline)
  - Parkinson (1980) — extreme-value, ~5x more efficient
  - Garman-Klass (1980) — uses full OHLC candle
  - Yang-Zhang (2000) — drift-independent, best for opening gaps
  - Vol-of-vol — second-order volatility dynamics

For option-chain IV computation, use core.pricing.iv_engine (GPU)
or core.pricing.sanos (LP-calibrated arbitrage-free surface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlaxmi.features.base import Feature


@dataclass(frozen=True)
class RealizedVol(Feature):
    """Realized volatility estimators from OHLCV bars.

    Produces: close_close, parkinson, garman_klass, yang_zhang, vol_of_vol
    """

    window: int = 20

    @property
    def name(self) -> str:
        return f"rvol_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        out = pd.DataFrame(index=df.index)

        # 1. Close-to-close (standard)
        log_ret = np.log(c / c.shift(1))
        out["close_close"] = log_ret.rolling(self.window).std() * np.sqrt(252)

        # 2. Parkinson (uses H-L range, ~5x more efficient)
        hl_sq = (np.log(h / l)) ** 2
        out["parkinson"] = np.sqrt(
            hl_sq.rolling(self.window).mean() / (4 * np.log(2)) * 252
        )

        # 3. Garman-Klass (uses O-H-L-C)
        u = np.log(h / o)
        d = np.log(l / o)
        cc = np.log(c / o)
        gk = 0.5 * (u - d) ** 2 - (2 * np.log(2) - 1) * cc**2
        out["garman_klass"] = np.sqrt(
            gk.rolling(self.window).mean().clip(lower=0) * 252
        )

        # 4. Yang-Zhang (drift-independent, best for opening gaps)
        oc = np.log(o / c.shift(1))
        co = np.log(c / o)

        sigma_oc = oc.rolling(self.window).var()
        sigma_co = co.rolling(self.window).var()
        sigma_rs = hl_sq.rolling(self.window).mean() / (4 * np.log(2))

        k = 0.34 / (1.34 + (self.window + 1) / (self.window - 1))
        yz_var = sigma_oc + k * sigma_co + (1 - k) * sigma_rs
        out["yang_zhang"] = np.sqrt(yz_var.clip(lower=0) * 252)

        # 5. Vol-of-vol: rolling std of realized vol changes
        out["vol_of_vol"] = out["close_close"].rolling(self.window).std()

        return out
