"""Crypto-specific alpha features.

These capture dynamics unique to 24/7 crypto markets:
- Volume regime shifts (institutional vs retail hours)
- Volatility clustering and regime detection
- Mean-reversion z-scores at multiple horizons
- Volume-weighted momentum (VWAP deviation)
- Intraday patterns (hour-of-day effects)
- Realized volatility vs implied (via ATR ratio)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlx.features.base import Feature


@dataclass(frozen=True)
class VolatilityRegime(Feature):
    """Detect volatility regime via rolling std ratio.

    Compares short-window realized vol to long-window realized vol.
    High ratio = expanding vol (breakout), low ratio = compressing (mean-revert).
    """

    fast: int = 24       # 1 day in hours
    slow: int = 168      # 1 week in hours

    @property
    def name(self) -> str:
        return f"vol_regime_{self.fast}_{self.slow}"

    @property
    def lookback(self) -> int:
        return self.slow + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df["Close"].pct_change()

        vol_fast = ret.rolling(self.fast).std()
        vol_slow = ret.rolling(self.slow).std()

        out = pd.DataFrame(index=df.index)
        out["vol_ratio"] = vol_fast / vol_slow.replace(0, np.nan)
        out["vol_expanding"] = (out["vol_ratio"] > 1.5).astype(np.int8)
        out["vol_compressing"] = (out["vol_ratio"] < 0.5).astype(np.int8)
        out["vol_fast"] = vol_fast
        out["vol_slow"] = vol_slow
        return out


@dataclass(frozen=True)
class MeanReversionZ(Feature):
    """Z-score of price relative to rolling mean at multiple horizons.

    Strong mean-reversion signal in crypto: extreme z-scores at slower
    horizons tend to revert.
    """

    windows: tuple[int, ...] = (24, 72, 168)  # 1d, 3d, 1w

    @property
    def name(self) -> str:
        tag = "_".join(str(w) for w in self.windows)
        return f"zscore_{tag}"

    @property
    def lookback(self) -> int:
        return max(self.windows) + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        out = pd.DataFrame(index=df.index)

        for w in self.windows:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            z = (close - ma) / std.replace(0, np.nan)
            out[f"z_{w}"] = z
            out[f"extreme_{w}"] = (z.abs() > 2.0).astype(np.int8)

        return out


@dataclass(frozen=True)
class VWAPDeviation(Feature):
    """Deviation of price from volume-weighted average price.

    VWAP deviation is a key institutional signal — when price is far from
    VWAP, it tends to revert.
    """

    window: int = 24  # hours

    @property
    def name(self) -> str:
        return f"vwap_dev_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = df["Volume"]

        cumvol = vol.rolling(self.window).sum()
        cumtpv = (typical * vol).rolling(self.window).sum()
        vwap = cumtpv / cumvol.replace(0, np.nan)

        deviation = (df["Close"] - vwap) / vwap.replace(0, np.nan)

        out = pd.DataFrame(index=df.index)
        out["deviation"] = deviation
        out["above_vwap"] = (deviation > 0).astype(np.int8)
        out["deviation_abs"] = deviation.abs()
        return out


@dataclass(frozen=True)
class VolumeProfile(Feature):
    """Volume dynamics — relative volume, volume trend, buyer/seller pressure.

    Volume spikes often precede directional moves in crypto.
    """

    window: int = 48  # 2 days

    @property
    def name(self) -> str:
        return f"vol_profile_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df["Volume"]
        close = df["Close"]

        vol_ma = vol.rolling(self.window).mean()
        vol_std = vol.rolling(self.window).std()

        out = pd.DataFrame(index=df.index)

        # Relative volume (current vs average)
        out["rel_volume"] = vol / vol_ma.replace(0, np.nan)

        # Volume z-score
        out["vol_zscore"] = (vol - vol_ma) / vol_std.replace(0, np.nan)

        # Volume trend (is volume increasing?)
        vol_fast = vol.rolling(self.window // 4).mean()
        out["vol_trend"] = vol_fast / vol_ma.replace(0, np.nan)

        # Dollar volume (proxy for liquidity)
        dollar_vol = vol * close
        dollar_ma = dollar_vol.rolling(self.window).mean()
        out["rel_dollar_vol"] = dollar_vol / dollar_ma.replace(0, np.nan)

        # Volume-price divergence: price up + volume down = weak rally
        price_up = (close.pct_change() > 0).astype(float)
        vol_up = (vol.pct_change() > 0).astype(float)
        out["vol_price_agree"] = (price_up == vol_up).astype(np.int8)

        return out


@dataclass(frozen=True)
class ReturnDistribution(Feature):
    """Higher moments of return distribution — skew and kurtosis.

    Crypto returns are fat-tailed.  Rolling skew/kurtosis shifts
    predict regime changes.
    """

    window: int = 72  # 3 days

    @property
    def name(self) -> str:
        return f"ret_dist_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        ret = df["Close"].pct_change()

        out = pd.DataFrame(index=df.index)
        out["skew"] = ret.rolling(self.window).skew()
        out["kurtosis"] = ret.rolling(self.window).kurt()

        # Tail risk: fraction of returns beyond 2 std
        std = ret.rolling(self.window).std()
        threshold = 2 * std
        out["tail_ratio"] = (
            ret.abs().rolling(self.window).apply(
                lambda x: (x > threshold.iloc[-1]).mean() if len(x) == self.window else np.nan,
                raw=False,
            )
        )

        return out


@dataclass(frozen=True)
class MultiTimeframeMomentum(Feature):
    """Momentum alignment across timeframes.

    When short, medium, and long momentum agree, moves tend to continue.
    When they disagree, reversals are more likely.
    """

    fast: int = 6       # 6h
    medium: int = 24    # 1d
    slow: int = 168     # 1w

    @property
    def name(self) -> str:
        return f"mtf_mom_{self.fast}_{self.medium}_{self.slow}"

    @property
    def lookback(self) -> int:
        return self.slow + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        ret_fast = close.pct_change(self.fast)
        ret_medium = close.pct_change(self.medium)
        ret_slow = close.pct_change(self.slow)

        dir_fast = np.sign(ret_fast)
        dir_medium = np.sign(ret_medium)
        dir_slow = np.sign(ret_slow)

        out = pd.DataFrame(index=df.index)
        out["ret_fast"] = ret_fast
        out["ret_medium"] = ret_medium
        out["ret_slow"] = ret_slow

        # Alignment score: +3 = all bullish, -3 = all bearish, mixed = 0ish
        out["alignment"] = dir_fast + dir_medium + dir_slow

        # Momentum divergence: fast vs slow disagree
        out["divergence"] = (dir_fast != dir_slow).astype(np.int8)

        return out


@dataclass(frozen=True)
class RangePosition(Feature):
    """Where price sits within recent range — Williams %R style.

    Crypto tends to cluster at range extremes before breakouts.
    """

    window: int = 48  # 2 days

    @property
    def name(self) -> str:
        return f"range_pos_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high_max = df["High"].rolling(self.window).max()
        low_min = df["Low"].rolling(self.window).min()
        rng = (high_max - low_min).replace(0, np.nan)

        out = pd.DataFrame(index=df.index)
        out["pct_range"] = (df["Close"] - low_min) / rng
        out["range_width"] = rng / df["Close"]  # normalized range width
        out["near_high"] = (out["pct_range"] > 0.9).astype(np.int8)
        out["near_low"] = (out["pct_range"] < 0.1).astype(np.int8)

        return out
