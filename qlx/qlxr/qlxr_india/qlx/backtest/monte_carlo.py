"""Monte Carlo Robustness Testing.

Block bootstrap (preserving autocorrelation) to estimate:
  - Confidence interval on Sharpe ratio
  - Probability of ruin (max DD > X%)
  - Expected worst-case monthly return
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    # Sharpe ratio distribution
    sharpe_mean: float = 0.0
    sharpe_std: float = 0.0
    sharpe_ci_5: float = 0.0
    sharpe_ci_95: float = 0.0
    sharpe_percentiles: dict = field(default_factory=dict)

    # Drawdown statistics
    max_dd_mean: float = 0.0
    max_dd_95: float = 0.0
    prob_ruin: float = 0.0      # P(max DD > ruin_threshold)
    ruin_threshold: float = 0.20

    # Return statistics
    annual_return_mean: float = 0.0
    annual_return_ci_5: float = 0.0
    annual_return_ci_95: float = 0.0
    worst_month_5: float = 0.0   # 5th percentile worst monthly return

    # Raw simulation data
    n_simulations: int = 0
    block_size: int = 0
    sharpe_distribution: list[float] = field(default_factory=list)
    max_dd_distribution: list[float] = field(default_factory=list)


def block_bootstrap(
    returns: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 21,
    ruin_threshold: float = 0.20,
    random_seed: int | None = 42,
) -> MonteCarloResult:
    """Run block bootstrap Monte Carlo simulation.

    Preserves autocorrelation structure by resampling contiguous
    blocks of returns rather than individual observations.

    Parameters
    ----------
    returns : np.ndarray
        Daily return series (e.g. from backtest).
    n_simulations : int
        Number of bootstrap paths.
    block_size : int
        Size of contiguous blocks (default 21 = ~1 month).
        Preserves intra-month autocorrelation.
    ruin_threshold : float
        Drawdown threshold for ruin probability (default 20%).
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    MonteCarloResult
    """
    rng = np.random.default_rng(random_seed)
    n = len(returns)

    if n < block_size * 2:
        return MonteCarloResult(n_simulations=0, block_size=block_size)

    # Number of blocks needed to match original length
    n_blocks = math.ceil(n / block_size)
    max_start = n - block_size

    sharpe_dist: list[float] = []
    max_dd_dist: list[float] = []
    annual_return_dist: list[float] = []
    worst_month_dist: list[float] = []
    ruin_count = 0

    for _ in range(n_simulations):
        # Sample block starts uniformly
        starts = rng.integers(0, max_start + 1, size=n_blocks)

        # Build synthetic return path
        sim_returns: list[float] = []
        for start in starts:
            block = returns[start: start + block_size]
            sim_returns.extend(block.tolist())

        # Trim to original length
        sim_arr = np.array(sim_returns[:n])

        # Compute Sharpe
        mu = np.mean(sim_arr)
        std = np.std(sim_arr, ddof=1)
        if std > 0:
            sharpe = mu / std * math.sqrt(252)
        else:
            sharpe = 0.0
        sharpe_dist.append(sharpe)

        # Compute max drawdown
        cum = np.cumsum(sim_arr)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = abs(float(np.min(dd))) if len(dd) > 0 else 0.0
        max_dd_dist.append(max_dd)

        if max_dd > ruin_threshold:
            ruin_count += 1

        # Annual return
        total_ret = float(np.sum(sim_arr))
        years = max(n / 252, 1 / 252)
        ann_ret = ((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0
        annual_return_dist.append(ann_ret)

        # Worst month (21-day rolling)
        if n >= 21:
            monthly_rets = [
                float(np.sum(sim_arr[i:i + 21]))
                for i in range(0, n - 20, 21)
            ]
            if monthly_rets:
                worst_month_dist.append(min(monthly_rets))

    # Compile results
    sharpe_arr = np.array(sharpe_dist)
    dd_arr = np.array(max_dd_dist)
    ann_arr = np.array(annual_return_dist)

    result = MonteCarloResult(
        sharpe_mean=float(np.mean(sharpe_arr)),
        sharpe_std=float(np.std(sharpe_arr, ddof=1)),
        sharpe_ci_5=float(np.percentile(sharpe_arr, 5)),
        sharpe_ci_95=float(np.percentile(sharpe_arr, 95)),
        sharpe_percentiles={
            p: float(np.percentile(sharpe_arr, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        max_dd_mean=float(np.mean(dd_arr)),
        max_dd_95=float(np.percentile(dd_arr, 95)),
        prob_ruin=ruin_count / n_simulations,
        ruin_threshold=ruin_threshold,
        annual_return_mean=float(np.mean(ann_arr)),
        annual_return_ci_5=float(np.percentile(ann_arr, 5)),
        annual_return_ci_95=float(np.percentile(ann_arr, 95)),
        worst_month_5=float(np.percentile(worst_month_dist, 5)) if worst_month_dist else 0.0,
        n_simulations=n_simulations,
        block_size=block_size,
        sharpe_distribution=sharpe_dist,
        max_dd_distribution=max_dd_dist,
    )

    return result


def format_monte_carlo(result: MonteCarloResult) -> str:
    """Format Monte Carlo results for display."""
    lines = [
        "Monte Carlo Robustness Test",
        "=" * 50,
        f"  Simulations:   {result.n_simulations}",
        f"  Block size:    {result.block_size} days",
        "",
        "Sharpe Ratio Distribution:",
        f"  Mean:          {result.sharpe_mean:+.2f}",
        f"  Std:           {result.sharpe_std:.2f}",
        f"  5th pctile:    {result.sharpe_ci_5:+.2f}",
        f"  95th pctile:   {result.sharpe_ci_95:+.2f}",
        "",
        "Drawdown Analysis:",
        f"  Mean max DD:   {result.max_dd_mean:.1%}",
        f"  95th max DD:   {result.max_dd_95:.1%}",
        f"  P(ruin>{result.ruin_threshold:.0%}): {result.prob_ruin:.1%}",
        "",
        "Return Distribution:",
        f"  Mean annual:   {result.annual_return_mean:.1%}",
        f"  5th pctile:    {result.annual_return_ci_5:.1%}",
        f"  95th pctile:   {result.annual_return_ci_95:.1%}",
        f"  Worst month:   {result.worst_month_5:.1%} (5th pctile)",
    ]
    return "\n".join(lines)
