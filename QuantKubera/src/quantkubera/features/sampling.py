"""
Symmetric CUSUM Filter for event-driven sampling in QuantKubera.

Detects a shift in the mean value of log returns. Fires an event whenever 
the cumulative positive or negative deviation exceeds a threshold.
"""

import numpy as np
import pandas as pd
from typing import Union


def cusum_filter(
    close: pd.Series,
    threshold: Union[float, pd.Series],
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter — detect mean-shift events in log returns.

    Args:
        close: Price series with a DatetimeIndex.
        threshold: Trigger threshold. If a Series (e.g., volatility), 
                  the filter becomes adaptive.

    Returns:
        DatetimeIndex of timestamps where events were triggered.
    """
    if len(close) < 2:
        return pd.DatetimeIndex([])

    # Calculate log returns
    log_ret = np.log(close / close.shift(1)).dropna()

    # Align threshold with log_ret
    if isinstance(threshold, pd.Series):
        threshold = threshold.reindex(log_ret.index, method="ffill")
    else:
        threshold = pd.Series(threshold, index=log_ret.index)

    events = []
    s_pos = 0.0
    s_neg = 0.0

    for t, ret in log_ret.items():
        h = threshold.loc[t]

        # Skip if threshold is invalid
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
