"""Performance metrics — one function, one source of truth.

All metrics computed from the returns series.  No duplication across
different modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars
    skew: float
    kurtosis: float
    n_observations: int

    def summary(self) -> str:
        return (
            f"Total return:     {self.total_return:+.2%}\n"
            f"Ann. return:      {self.annualised_return:+.2%}\n"
            f"Ann. volatility:  {self.annualised_volatility:.2%}\n"
            f"Sharpe:           {self.sharpe_ratio:.3f}\n"
            f"Sortino:          {self.sortino_ratio:.3f}\n"
            f"Calmar:           {self.calmar_ratio:.3f}\n"
            f"Max drawdown:     {self.max_drawdown:.2%}\n"
            f"DD duration:      {self.max_drawdown_duration} bars\n"
            f"Skew:             {self.skew:.3f}\n"
            f"Kurtosis:         {self.kurtosis:.3f}\n"
            f"Observations:     {self.n_observations}"
        )


def compute_metrics(
    returns: pd.Series,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics from a return series."""
    n = len(returns)
    if n < 2:
        return PerformanceMetrics(
            total_return=0.0,
            annualised_return=0.0,
            annualised_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            skew=0.0,
            kurtosis=0.0,
            n_observations=n,
        )

    # Cumulative returns
    cum = (1 + returns).cumprod()
    total_return = float(cum.iloc[-1] - 1)

    # Annualised
    n_years = n / periods_per_year
    ann_return = float((1 + total_return) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    # Sharpe (arithmetic: mean/std * sqrt(N) — standard in quant finance)
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)) if ann_vol > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = ann_return / downside_vol if downside_vol > 0 else (float("inf") if ann_return > 0 else 0.0)

    # Drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # Drawdown duration (longest underwater streak)
    underwater = dd < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_durations = underwater.groupby(groups).sum()
        max_dd_dur = int(dd_durations.max())
    else:
        max_dd_dur = 0

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualised_return=ann_return,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        skew=float(returns.skew()),
        kurtosis=float(returns.kurtosis()),
        n_observations=n,
    )
