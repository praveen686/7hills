"""
Triple Barrier Labeling Method for QuantKubera.

Labels each trade event based on three barriers:
  - Upper (Profit-Take): Hit if price reaches a designated profit target.
  - Lower (Stop-Loss): Hit if price drops to a designated loss limit.
  - Vertical (Time-Out): Hit if none of the above are reached within a max holding period.

Barriers are ideally set in units of volatility to adapt to market conditions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


def get_daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """Calculate daily volatility using exponentially weighted standard deviation of log returns."""
    log_ret = np.log(close).diff()
    vol = log_ret.ewm(span=span).std()
    return vol


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    target: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    num_days: int = 5,
) -> pd.DataFrame:
    """Apply the triple-barrier labeling method.

    Args:
        close: Price series with DatetimeIndex.
        events: Timestamps at which to evaluate barriers (e.g. all points or CUSUM-filtered).
        pt_sl: Multiplier for profit-take and stop-loss barriers (in units of vol).
        target: Pre-computed volatility series. If None, computes 50-day rolling vol.
        min_ret: Minimum required return to label as +1 or -1 at vertical barrier hit.
        num_days: Maximum holding period in trading days.

    Returns:
        DataFrame with columns:
            - ret: The return realized at the first barrier hit.
            - bin: Label (-1 for SL, 1 for PT, 0 or sign(ret) for Time-Out).
            - t1: Timestamp when the barrier was hit.
    """
    # Ensure DatetimeIndex
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")

    # Filter events to those present in close
    events = events[events.isin(close.index)]
    if len(events) == 0:
        return pd.DataFrame(columns=["ret", "bin", "t1"])

    # Compute daily volatility if not provided
    if target is None:
        target = get_daily_vol(close)

    # Align target to events
    target = target.reindex(events).dropna()
    events = target.index

    if len(events) == 0:
        return pd.DataFrame(columns=["ret", "bin", "t1"])

    pt_width, sl_width = pt_sl
    records = []

    for t0 in events:
        vol = target.loc[t0]
        entry_price = close.loc[t0]

        # Determine vertical barrier (Time-Out)
        loc_t0 = close.index.get_loc(t0)
        if isinstance(loc_t0, slice):
            loc_t0 = loc_t0.start
        loc_t1 = min(loc_t0 + num_days, len(close) - 1)
        t_vertical = close.index[loc_t1]

        # Slice price path
        path = close.loc[t0:t_vertical]
        if len(path) <= 1:
            continue

        # Log returns relative to entry
        log_returns = np.log(path / entry_price)

        # Check PT and SL barriers
        upper_threshold = pt_width * vol if pt_width > 0 else np.inf
        lower_threshold = -sl_width * vol if sl_width > 0 else -np.inf

        upper_hits = log_returns[log_returns >= upper_threshold]
        t_upper = upper_hits.index[0] if not upper_hits.empty else pd.NaT

        lower_hits = log_returns[log_returns <= lower_threshold]
        t_lower = lower_hits.index[0] if not lower_hits.empty else pd.NaT

        # Find first touch
        candidates = [t_vertical]
        if pd.notna(t_upper): candidates.append(t_upper)
        if pd.notna(t_lower): candidates.append(t_lower)
        first_touch = min(candidates)

        ret_at_touch = log_returns.loc[first_touch]

        # Assign label
        if first_touch == t_vertical and pd.isna(t_upper) and pd.isna(t_lower):
            # Time-out reached first
            if abs(ret_at_touch) < min_ret:
                label = 0
            else:
                label = int(np.sign(ret_at_touch))
        elif pd.notna(t_upper) and (pd.isna(t_lower) or t_upper <= t_lower):
            # Profit-take hit first
            label = 1
        else:
            # Stop-loss hit first
            label = -1

        records.append({
            "ret": ret_at_touch,
            "bin": label,
            "t1": first_touch
        })

    if not records:
        return pd.DataFrame(columns=["ret", "bin", "t1"])

    result = pd.DataFrame(records, index=events[: len(records)])
    return result


def meta_labeling(
    primary_preds: pd.Series,
    triple_barrier_labels: pd.Series,
) -> pd.Series:
    """Generate meta-labels from primary predictions and true labels.

    Meta-label is 1 if sign(preds) == sign(labels), and 0 otherwise.
    Essentially: 'Did the primary model get the direction right?'
    """
    # Align on common index
    preds, labels = primary_preds.align(triple_barrier_labels, join="inner")

    # Meta-label: 1 iff primary direction matches the realized direction
    meta = np.where(
        (preds.values != 0) & (labels.values != 0) & (np.sign(preds.values) == np.sign(labels.values)),
        1,
        0,
    )

    return pd.Series(meta, index=preds.index, name="meta_label", dtype=int)
