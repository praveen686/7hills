"""
Triple Barrier Labeling Method.

Reference: Lopez de Prado, *Advances in Financial Machine Learning*,
Chapter 3 — Meta-Labeling.

Labels each event with {-1, 0, +1} based on which of three barriers
is touched first:
  +1  — upper (profit-take) barrier hit first
  -1  — lower (stop-loss) barrier hit first
   0  — vertical (time) barrier hit first and return < min_ret

The barriers are set in *units of daily volatility*, making them
adaptive to the current market regime.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    target: Optional[pd.Series] = None,
    min_ret: float = 0.0,
    num_days: int = 5,
) -> pd.DataFrame:
    """Apply the triple-barrier labeling method.

    Parameters
    ----------
    close : pd.Series
        Price series with a DatetimeIndex.
    events : pd.DatetimeIndex
        Timestamps at which to evaluate barriers (e.g. from CUSUM filter).
        Must be a subset of ``close.index``.
    pt_sl : tuple[float, float]
        ``(profit_take_width, stop_loss_width)`` in units of daily vol.
        Set either to 0 to disable that barrier.
    target : pd.Series, optional
        Daily volatility estimate for each timestamp in ``events``.
        If *None*, a rolling 50-day exponential std of log returns is used.
    min_ret : float, default 0.0
        Minimum absolute return to be classified as +1 or -1 when the
        vertical barrier is touched.  Returns below ``min_ret`` get bin=0.
    num_days : int, default 5
        Maximum holding period in *trading days* (vertical barrier).

    Returns
    -------
    pd.DataFrame
        Columns:

        - **ret** : float — log return from entry to the first barrier hit.
        - **bin** : int — label in {-1, 0, +1}.
        - **t1** : Timestamp — the datetime at which the barrier was hit.

        Indexed by the event timestamps.

    Notes
    -----
    *Concurrent labels* are handled correctly: each event is evaluated
    independently against the forward price path from its own entry
    time, so overlapping events do not interfere with each other.
    """
    # ---- Validate inputs ------------------------------------------------
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")

    # Filter events to those present in close
    events = events[events.isin(close.index)]
    if len(events) == 0:
        return pd.DataFrame(columns=["ret", "bin", "t1"])

    # ---- Default daily vol if not provided ------------------------------
    if target is None:
        log_ret = np.log(close).diff()
        target = log_ret.ewm(span=50, min_periods=10).std()

    # Align target to events
    target = target.reindex(events)
    # Drop events where vol is NaN (can't set barriers)
    valid = target.dropna()
    events = valid.index

    if len(events) == 0:
        return pd.DataFrame(columns=["ret", "bin", "t1"])

    # ---- Compute barriers for each event --------------------------------
    pt_width, sl_width = pt_sl
    records: list[dict] = []

    for t0 in events:
        vol = target.loc[t0]
        entry_price = close.loc[t0]

        # Vertical barrier: num_days forward in trading-day index
        loc_t0 = close.index.get_loc(t0)
        if isinstance(loc_t0, slice):
            loc_t0 = loc_t0.start
        loc_t1 = min(loc_t0 + num_days, len(close) - 1)
        t1_vertical = close.index[loc_t1]

        # Forward price path from t0 to t1 (inclusive of both ends)
        path = close.loc[t0:t1_vertical]
        if len(path) <= 1:
            # Not enough forward data — skip
            continue

        # Log returns relative to entry
        log_returns = np.log(path / entry_price)

        # ---- Check upper barrier (profit take) --------------------------
        upper_thresh = pt_width * vol if pt_width > 0 else np.inf
        upper_hits = log_returns[log_returns >= upper_thresh]
        t_upper = upper_hits.index[0] if len(upper_hits) > 0 else pd.NaT

        # ---- Check lower barrier (stop loss) ----------------------------
        lower_thresh = -sl_width * vol if sl_width > 0 else -np.inf
        lower_hits = log_returns[log_returns <= lower_thresh]
        t_lower = lower_hits.index[0] if len(lower_hits) > 0 else pd.NaT

        # ---- Determine which barrier was touched first ------------------
        first_touch = _first_touch(t_upper, t_lower, t1_vertical)
        ret_at_touch = log_returns.loc[first_touch]

        # ---- Assign label -----------------------------------------------
        if first_touch == t1_vertical and pd.isna(t_upper) and pd.isna(t_lower):
            # Vertical barrier hit (neither PT nor SL touched)
            if abs(ret_at_touch) < min_ret:
                label = 0
            else:
                label = int(np.sign(ret_at_touch))
        elif pd.notna(t_upper) and (
            pd.isna(t_lower) or t_upper <= t_lower
        ):
            # Upper barrier first (or tied — profit-take wins)
            if t_upper <= t1_vertical:
                label = 1
            else:
                # Should not happen, but defensive
                label = int(np.sign(ret_at_touch)) if abs(ret_at_touch) >= min_ret else 0
        else:
            # Lower barrier first
            if t_lower <= t1_vertical:
                label = -1
            else:
                label = int(np.sign(ret_at_touch)) if abs(ret_at_touch) >= min_ret else 0

        records.append(
            {"ret": ret_at_touch, "bin": label, "t1": first_touch}
        )

    if not records:
        return pd.DataFrame(columns=["ret", "bin", "t1"])

    result = pd.DataFrame(records, index=events[: len(records)])
    result["bin"] = result["bin"].astype(int)
    return result


def _first_touch(
    t_upper: pd.Timestamp,
    t_lower: pd.Timestamp,
    t_vertical: pd.Timestamp,
) -> pd.Timestamp:
    """Return the earliest of the three barrier timestamps.

    NaT values (barrier not touched) are treated as infinitely late.
    """
    candidates = []
    if pd.notna(t_upper):
        candidates.append(t_upper)
    if pd.notna(t_lower):
        candidates.append(t_lower)
    candidates.append(t_vertical)  # always present
    return min(candidates)


def concurrent_label_count(events_t1: pd.Series, close_index: pd.DatetimeIndex) -> pd.Series:
    """Count how many labels are concurrent at each timestamp.

    Parameters
    ----------
    events_t1 : pd.Series
        Series indexed by event start time, values are barrier-touch times (t1).
    close_index : pd.DatetimeIndex
        Full price index.

    Returns
    -------
    pd.Series
        Number of concurrent active labels at each point in ``close_index``.
    """
    # For each event [t0, t1], increment a counter
    count = pd.Series(0, index=close_index)
    for t0, t1 in events_t1.items():
        if pd.isna(t1):
            continue
        count.loc[t0:t1] += 1
    return count
