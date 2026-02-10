"""Triple Barrier Target Builder for TFT pipeline.

Wraps AFML's triple_barrier_labels() to produce regime-adaptive
{-1, 0, +1} labels as an alternative to raw forward returns.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from quantlaxmi.models.afml import triple_barrier_labels, get_daily_vol
from quantlaxmi.models.afml.cusum import cusum_filter

logger = logging.getLogger(__name__)


class TripleBarrierTargetBuilder:
    """Build triple-barrier labels for a price series.

    Parameters
    ----------
    pt_sl : tuple[float, float]
        Profit-take and stop-loss widths in units of daily vol.
    num_days : int
        Vertical barrier (max holding period in trading days).
    min_ret : float
        Minimum return threshold for vertical barrier labels.
    vol_window : int
        Window for daily volatility estimation.
    use_cusum_events : bool
        If True, use CUSUM filter to select event timestamps.
        If False, use every bar as an event.
    cusum_threshold_mult : float
        Multiplier on daily vol for CUSUM threshold.
    """

    def __init__(
        self,
        pt_sl: tuple[float, float] = (1.0, 1.0),
        num_days: int = 5,
        min_ret: float = 0.0,
        vol_window: int = 50,
        use_cusum_events: bool = True,
        cusum_threshold_mult: float = 1.0,
    ):
        self.pt_sl = pt_sl
        self.num_days = num_days
        self.min_ret = min_ret
        self.vol_window = vol_window
        self.use_cusum_events = use_cusum_events
        self.cusum_threshold_mult = cusum_threshold_mult

    def build(self, close: pd.Series) -> pd.DataFrame:
        """Compute triple-barrier labels.

        Parameters
        ----------
        close : pd.Series
            Price series with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Columns: ret, bin, t1. Indexed by event timestamps.
        """
        daily_vol = get_daily_vol(close, window=self.vol_window)

        if self.use_cusum_events:
            threshold = daily_vol * self.cusum_threshold_mult
            events = cusum_filter(close, threshold=threshold)
            logger.info("CUSUM filter produced %d events from %d bars", len(events), len(close))
        else:
            events = close.index

        if len(events) == 0:
            return pd.DataFrame(columns=["ret", "bin", "t1"])

        result = triple_barrier_labels(
            close=close,
            events=events,
            pt_sl=self.pt_sl,
            target=daily_vol,
            min_ret=self.min_ret,
            num_days=self.num_days,
        )

        logger.info(
            "Triple barrier labels: %d events, distribution: %s",
            len(result),
            result["bin"].value_counts().to_dict() if len(result) > 0 else {},
        )
        return result
