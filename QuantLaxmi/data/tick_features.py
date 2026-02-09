"""Tick-based microstructure feature extractors.

These features extract information from tick-level data that is *not*
available from aggregated 1-minute bars.  All functions are **fully causal**
— they only use data at or before time *t* to compute the feature at *t*.

Required input
--------------
All functions expect a DataFrame with at least:
    timestamp   datetime64   tick timestamp (IST, sorted ascending)
    ltp         float        last traded price
    volume      int64        cumulative volume since market open

Some functions additionally use:
    oi          int64        open interest

Functions
---------
compute_vpin          Volume-Synchronized Probability of Informed Trading
compute_kyle_lambda   Kyle's lambda (price impact per unit volume)
compute_roll_spread   Roll (1984) implied bid-ask spread
compute_amihud        Amihud (2002) illiquidity ratio
compute_trade_intensity  Trades per second + burst detection
compute_ofi           Order Flow Imbalance from tick-direction inference

References
----------
- Easley, Lopez de Prado, O'Hara (2012) "Flow toxicity and liquidity"
- Kyle (1985) "Continuous Auctions and Insider Trading"
- Roll (1984) "A Simple Implicit Measure of the Effective Bid-Ask Spread"
- Amihud (2002) "Illiquidity and stock returns"
- Cont, Kukanov, Stoikov (2014) "The Price Impact of Order Book Events"
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_sorted(ticks: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by timestamp if not already."""
    if not ticks["timestamp"].is_monotonic_increasing:
        return ticks.sort_values("timestamp").reset_index(drop=True)
    return ticks


def _tick_volume(ticks: pd.DataFrame) -> pd.Series:
    """Compute per-tick traded volume from cumulative volume.

    Zerodha's volume column is cumulative since market open.
    Per-tick volume = diff, clipped to >= 0.  First tick gets 0.
    """
    if ticks.empty:
        return pd.Series(dtype=np.float64)
    dv = ticks["volume"].diff().clip(lower=0).fillna(0)
    return dv


def _tick_direction(ticks: pd.DataFrame) -> pd.Series:
    """Classify each tick as buy (+1) or sell (-1) using the tick rule.

    The tick rule:
        - Up-tick (price > prev price) → buy (+1)
        - Down-tick (price < prev price) → sell (-1)
        - Zero-tick (price == prev price) → inherit previous direction

    First tick defaults to +1 (buy).
    """
    if ticks.empty:
        return pd.Series(dtype=np.float64)
    dp = ticks["ltp"].diff()
    direction = pd.Series(np.zeros(len(ticks)), index=ticks.index, dtype=np.float64)
    direction[dp > 0] = 1.0
    direction[dp < 0] = -1.0
    # Zero-ticks: forward-fill the last non-zero direction
    direction.iloc[0] = 1.0  # default first tick to buy
    direction = direction.replace(0.0, np.nan).ffill().fillna(1.0)
    return direction


# ---------------------------------------------------------------------------
# VPIN — Volume-Synchronized Probability of Informed Trading
# ---------------------------------------------------------------------------


def compute_vpin(
    ticks: pd.DataFrame,
    bucket_size: int = 50000,
    n_buckets: int = 50,
) -> pd.DataFrame:
    """Compute VPIN (Easley, Lopez de Prado, O'Hara 2012).

    VPIN estimates the probability of informed trading by measuring
    order flow imbalance across equal-volume buckets.

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have: timestamp, ltp, volume.
    bucket_size : int
        Number of units of volume per bucket.  For index futures with
        lot_size=25-75, a bucket of 50000 shares gives ~10-20 bars/day.
    n_buckets : int
        Rolling window of buckets for VPIN calculation.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (bucket end time), vpin, buy_volume,
        sell_volume, bucket_volume.
        One row per completed volume bucket.
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return pd.DataFrame(
            columns=["timestamp", "vpin", "buy_volume", "sell_volume", "bucket_volume"]
        )

    tick_vol = _tick_volume(ticks).values
    direction = _tick_direction(ticks).values
    timestamps = ticks["timestamp"].values
    ltp = ticks["ltp"].values

    # Classify volume as buy or sell
    buy_vol = np.where(direction > 0, tick_vol, 0)
    sell_vol = np.where(direction < 0, tick_vol, 0)

    # Accumulate into equal-volume buckets
    buckets: list[dict] = []
    cum_vol = 0.0
    cum_buy = 0.0
    cum_sell = 0.0
    bucket_start_idx = 0

    for i in range(len(ticks)):
        cum_vol += tick_vol[i]
        cum_buy += buy_vol[i]
        cum_sell += sell_vol[i]

        if cum_vol >= bucket_size:
            buckets.append({
                "timestamp": timestamps[i],
                "buy_volume": cum_buy,
                "sell_volume": cum_sell,
                "bucket_volume": cum_vol,
            })
            cum_vol = 0.0
            cum_buy = 0.0
            cum_sell = 0.0
            bucket_start_idx = i + 1

    if not buckets:
        return pd.DataFrame(
            columns=["timestamp", "vpin", "buy_volume", "sell_volume", "bucket_volume"]
        )

    df = pd.DataFrame(buckets)

    # VPIN = rolling mean of |buy_vol - sell_vol| / bucket_volume
    abs_imbalance = (df["buy_volume"] - df["sell_volume"]).abs()
    # Use expanding window until we have enough buckets, then rolling
    if len(df) >= n_buckets:
        df["vpin"] = (
            abs_imbalance.rolling(n_buckets).sum()
            / df["bucket_volume"].rolling(n_buckets).sum()
        )
    else:
        df["vpin"] = (
            abs_imbalance.expanding().sum()
            / df["bucket_volume"].expanding().sum()
        )

    return df


# ---------------------------------------------------------------------------
# Kyle's Lambda — Price Impact Coefficient
# ---------------------------------------------------------------------------


def compute_kyle_lambda(
    ticks: pd.DataFrame,
    window: int = 300,
    min_trades: int = 30,
) -> pd.Series:
    """Compute Kyle's lambda (price impact per signed volume).

    Lambda = Cov(dp, signed_volume) / Var(signed_volume)

    over a rolling window of `window` ticks.

    High lambda → low liquidity (prices move a lot per unit of volume).

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have: timestamp, ltp, volume.
    window : int
        Rolling window in number of ticks.
    min_trades : int
        Minimum ticks in window to produce a value (else NaN).

    Returns
    -------
    pd.Series
        Kyle's lambda at each tick, indexed like input.
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return pd.Series(dtype=np.float64, name="kyle_lambda")

    tick_vol = _tick_volume(ticks)
    direction = _tick_direction(ticks)
    dp = ticks["ltp"].diff().fillna(0)

    signed_vol = direction * tick_vol

    # Rolling covariance and variance
    cov_dp_sv = dp.rolling(window, min_periods=min_trades).cov(signed_vol)
    var_sv = signed_vol.rolling(window, min_periods=min_trades).var(ddof=1)

    # Lambda = cov / var, guarded against zero variance
    kyle_lambda = cov_dp_sv / var_sv.replace(0, np.nan)
    kyle_lambda.name = "kyle_lambda"
    kyle_lambda.index = ticks.index

    return kyle_lambda


# ---------------------------------------------------------------------------
# Roll Implied Spread
# ---------------------------------------------------------------------------


def compute_roll_spread(
    ticks: pd.DataFrame,
    window: int = 300,
    min_trades: int = 30,
) -> pd.Series:
    """Compute Roll (1984) implied bid-ask spread from tick autocovariance.

    spread = 2 * sqrt(-Cov(dp_t, dp_{t-1}))  if covariance is negative
           = 0                                 otherwise

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have: ltp.
    window : int
        Rolling window in number of ticks.
    min_trades : int
        Minimum ticks for valid computation.

    Returns
    -------
    pd.Series
        Implied spread at each tick.
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return pd.Series(dtype=np.float64, name="roll_spread")

    dp = ticks["ltp"].diff().fillna(0)
    dp_lag = dp.shift(1)

    # Rolling covariance of consecutive price changes
    autocov = dp.rolling(window, min_periods=min_trades).cov(dp_lag)

    # Spread = 2 * sqrt(-autocov) when autocov < 0, else 0
    neg_autocov = (-autocov).clip(lower=0)
    spread = 2.0 * np.sqrt(neg_autocov)
    spread.name = "roll_spread"
    spread.index = ticks.index

    return spread


# ---------------------------------------------------------------------------
# Amihud Illiquidity Ratio
# ---------------------------------------------------------------------------


def compute_amihud(
    ticks: pd.DataFrame,
    window: int = 300,
    min_trades: int = 30,
) -> pd.Series:
    """Compute Amihud (2002) illiquidity ratio from tick data.

    Amihud = mean(|return_t| / tick_volume_t)  over rolling window

    Higher Amihud → less liquid (prices move more per unit volume).

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have: ltp, volume.
    window : int
        Rolling window in ticks.
    min_trades : int
        Minimum ticks for valid computation.

    Returns
    -------
    pd.Series
        Amihud illiquidity at each tick.
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return pd.Series(dtype=np.float64, name="amihud_illiquidity")

    tick_vol = _tick_volume(ticks)
    ret = ticks["ltp"].pct_change().abs().fillna(0)

    # Avoid division by zero — only compute where volume > 0
    ratio = pd.Series(np.nan, index=ticks.index, dtype=np.float64)
    has_vol = tick_vol > 0
    ratio[has_vol] = ret[has_vol] / tick_vol[has_vol]

    amihud = ratio.rolling(window, min_periods=min_trades).mean()
    amihud.name = "amihud_illiquidity"

    return amihud


# ---------------------------------------------------------------------------
# Trade Intensity + Burst Detection
# ---------------------------------------------------------------------------


def compute_trade_intensity(
    ticks: pd.DataFrame,
    window: str = "60s",
    burst_threshold: float = 3.0,
) -> pd.DataFrame:
    """Compute trade arrival intensity and detect bursts.

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have: timestamp.
    window : str
        Time window for counting trades (pandas offset, e.g. "60s", "5min").
    burst_threshold : float
        A burst is detected when intensity exceeds
        ``burst_threshold * rolling_mean_intensity``.

    Returns
    -------
    pd.DataFrame
        Columns:
        - intensity: trades per second within the window
        - inter_arrival_ms: milliseconds since previous tick
        - is_burst: bool, True when intensity > threshold * rolling mean
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return pd.DataFrame(columns=["intensity", "inter_arrival_ms", "is_burst"])

    ts = ticks["timestamp"]

    # Set timestamp as index for time-based rolling
    df = pd.DataFrame({"timestamp": ts.values}, index=ts)

    # Count trades in rolling time window
    trade_count = df["timestamp"].rolling(window).count()

    # Convert window to seconds for rate calculation
    window_td = pd.Timedelta(window)
    window_secs = window_td.total_seconds()
    intensity = trade_count / window_secs

    # Inter-arrival time in milliseconds
    inter_arrival = ts.diff().dt.total_seconds() * 1000
    inter_arrival.iloc[0] = np.nan

    # Burst detection: intensity > threshold * 5-minute rolling mean
    # Use a 5-minute (300s) lookback for the baseline
    baseline_window = max(int(300 / window_secs), 10)
    rolling_mean = intensity.rolling(baseline_window, min_periods=5).mean()
    is_burst = intensity > (burst_threshold * rolling_mean)

    result = pd.DataFrame({
        "intensity": intensity.values,
        "inter_arrival_ms": inter_arrival.values,
        "is_burst": is_burst.values,
    }, index=ticks.index)

    return result


# ---------------------------------------------------------------------------
# Order Flow Imbalance (OFI)
# ---------------------------------------------------------------------------


def compute_ofi(
    ticks: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """Compute Order Flow Imbalance from tick price/volume changes.

    Since we don't have L2 order book data, we estimate OFI from the
    tick rule (classifying each trade as buy or sell) and volume deltas.

    OFI_t = sum of signed_volume over rolling window

    The normalized OFI divides by total volume in the window.

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have: timestamp, ltp, volume.
    window : int
        Rolling window in ticks for aggregation.

    Returns
    -------
    pd.DataFrame
        Columns:
        - ofi_raw: raw signed volume sum over window
        - ofi_normalized: ofi_raw / total_volume_in_window (range -1 to +1)
        - buy_pressure: fraction of volume classified as buys
        - tick_direction: per-tick buy (+1) or sell (-1)
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return pd.DataFrame(
            columns=["ofi_raw", "ofi_normalized", "buy_pressure", "tick_direction"]
        )

    tick_vol = _tick_volume(ticks)
    direction = _tick_direction(ticks)

    signed_vol = direction * tick_vol

    # Rolling sums
    ofi_raw = signed_vol.rolling(window, min_periods=1).sum()
    total_vol = tick_vol.rolling(window, min_periods=1).sum()

    ofi_normalized = ofi_raw / total_vol.replace(0, np.nan)

    # Buy pressure: fraction of volume that is buy-initiated
    buy_vol = (signed_vol.clip(lower=0)).rolling(window, min_periods=1).sum()
    buy_pressure = buy_vol / total_vol.replace(0, np.nan)

    result = pd.DataFrame({
        "ofi_raw": ofi_raw.values,
        "ofi_normalized": ofi_normalized.values,
        "buy_pressure": buy_pressure.values,
        "tick_direction": direction.values,
    }, index=ticks.index)

    return result


# ---------------------------------------------------------------------------
# Convenience: compute all features at once
# ---------------------------------------------------------------------------


def compute_all_features(
    ticks: pd.DataFrame,
    vpin_bucket_size: int = 50000,
    kyle_window: int = 300,
    roll_window: int = 300,
    amihud_window: int = 300,
    intensity_window: str = "60s",
    ofi_window: int = 100,
) -> pd.DataFrame:
    """Compute all microstructure features and return merged DataFrame.

    Parameters
    ----------
    ticks : pd.DataFrame
        Raw ticks with timestamp, ltp, volume, oi.
    **kwargs
        Parameters forwarded to individual feature functions.

    Returns
    -------
    pd.DataFrame
        Original tick columns plus: kyle_lambda, roll_spread,
        amihud_illiquidity, intensity, inter_arrival_ms, is_burst,
        ofi_raw, ofi_normalized, buy_pressure, tick_direction.
        VPIN is returned separately (volume-time, not tick-time).
    """
    ticks = _ensure_sorted(ticks)
    if ticks.empty:
        return ticks.copy()

    result = ticks.copy()

    # Kyle's lambda
    result["kyle_lambda"] = compute_kyle_lambda(ticks, window=kyle_window).values

    # Roll spread
    result["roll_spread"] = compute_roll_spread(ticks, window=roll_window).values

    # Amihud illiquidity
    result["amihud_illiquidity"] = compute_amihud(ticks, window=amihud_window).values

    # Trade intensity
    intensity_df = compute_trade_intensity(ticks, window=intensity_window)
    result["intensity"] = intensity_df["intensity"].values
    result["inter_arrival_ms"] = intensity_df["inter_arrival_ms"].values
    result["is_burst"] = intensity_df["is_burst"].values

    # OFI
    ofi_df = compute_ofi(ticks, window=ofi_window)
    result["ofi_raw"] = ofi_df["ofi_raw"].values
    result["ofi_normalized"] = ofi_df["ofi_normalized"].values
    result["buy_pressure"] = ofi_df["buy_pressure"].values
    result["tick_direction"] = ofi_df["tick_direction"].values

    return result
