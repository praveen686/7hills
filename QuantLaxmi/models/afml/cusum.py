"""
Symmetric CUSUM Filter & Daily Volatility Estimator.

Reference: Lopez de Prado, *Advances in Financial Machine Learning*,
Chapter 2 — Financial Data Structures, Section 2.5.2.1.

The CUSUM filter is a quality-control method that detects a shift in
the mean value of a measured quantity away from a target.  Applied to
log returns, it fires an event whenever the cumulative positive or
negative deviation from the running mean exceeds a threshold.

This produces an *event-driven* sampling of the price series:
events are sparse during calm markets and frequent during volatile
regimes — exactly the adaptive behaviour we want for triggering
feature computation and labeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_daily_vol(
    close: pd.Series,
    window: int = 50,
    min_periods: int | None = None,
) -> pd.Series:
    """Estimate daily volatility as an exponentially weighted std of log returns.

    Parameters
    ----------
    close : pd.Series
        Price series with a DatetimeIndex.
    window : int, default 50
        Span for the EWM (exponentially weighted moving) standard deviation.
    min_periods : int, optional
        Minimum number of observations required.  Defaults to
        ``max(1, window // 5)``.

    Returns
    -------
    pd.Series
        Daily vol estimate, same index as ``close``.  Leading NaNs
        where insufficient history exists.
    """
    if min_periods is None:
        min_periods = max(1, window // 5)

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.ewm(span=window, min_periods=min_periods).std()
    return vol


def cusum_filter(
    close: pd.Series,
    threshold: float | pd.Series,
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter — detect mean-shift events in log returns.

    The filter maintains two running sums:

    - ``s_pos`` accumulates positive deviations of log returns.
    - ``s_neg`` accumulates negative deviations (in absolute value).

    An event is triggered (and both accumulators reset to zero) whenever
    either sum exceeds ``threshold``.

    Parameters
    ----------
    close : pd.Series
        Price series with a DatetimeIndex.
    threshold : float or pd.Series
        Trigger threshold.  If a Series, it must be aligned with ``close``
        (e.g., a rolling volatility estimate), allowing the filter to be
        *adaptive* — more sensitive in low-vol regimes, less in high-vol.

    Returns
    -------
    pd.DatetimeIndex
        Timestamps at which events were triggered.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> prices = pd.Series(np.cumsum(np.random.randn(500)) + 100,
    ...                    index=pd.bdate_range("2020-01-01", periods=500))
    >>> events = cusum_filter(prices, threshold=0.02)
    >>> len(events) > 0
    True
    """
    if len(close) < 2:
        return pd.DatetimeIndex([])

    log_ret = np.log(close / close.shift(1)).dropna()

    if isinstance(threshold, pd.Series):
        # Align threshold with log_ret; forward-fill gaps
        threshold = threshold.reindex(log_ret.index, method="ffill")
    else:
        # Scalar — broadcast to all timestamps
        threshold = pd.Series(threshold, index=log_ret.index)

    events: list[pd.Timestamp] = []
    s_pos: float = 0.0
    s_neg: float = 0.0

    for t, ret in log_ret.items():
        h = threshold.loc[t]

        # Skip if threshold is NaN (insufficient vol history)
        if np.isnan(h) or h <= 0:
            continue

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos > h:
            events.append(t)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -h:
            events.append(t)
            s_pos = 0.0
            s_neg = 0.0

    return pd.DatetimeIndex(events)
