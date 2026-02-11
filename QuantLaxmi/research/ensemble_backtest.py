"""Multi-Strategy Ensemble Research Backtest.

Combines daily return streams from three independent strategies on NIFTY 50:
  - S25: Divergence Flow Field (DFF composite signal)
  - S5:  Tick Microstructure (Hawkes intensity)
  - S4:  IV Mean-Reversion (SANOS ATM IV percentile)

Each strategy generates its own causal position time series independently.
Returns are combined at the portfolio level using three ensemble methods:
  1. Equal Weight
  2. Inverse Volatility (63-day rolling)
  3. Correlation-Adjusted (inverse vol with correlation penalty)

Parameters:
  - Date range: 2025-01-01 to 2026-02-06
  - All strategies use their best parameters
  - Costs applied per-strategy before combining
  - Sharpe: ddof=1, sqrt(252), all days including flat

Usage:
    python -m research.ensemble_backtest
    python research/ensemble_backtest.py
    python research/ensemble_backtest.py --start 2025-01-01 --end 2026-02-06
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quantlaxmi.data.store import MarketDataStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL = "NIFTY 50"
_INDEX_NAME = "Nifty 50"

# Per-leg transaction cost in index points for NIFTY
# 3 pts per side on ~22000 level
COST_PER_SIDE_NIFTY = 3.0 / 22000.0

# Results output directory
from quantlaxmi.data._paths import RESEARCH_RESULTS_DIR
RESULTS_DIR = RESEARCH_RESULTS_DIR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_nifty_close(
    store: MarketDataStore,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load NIFTY 50 daily close prices from nse_index_close.

    Returns DataFrame with columns: date, close. Sorted ascending.
    """
    df = store.sql(
        'SELECT date, "Closing Index Value" AS close '
        "FROM nse_index_close "
        'WHERE date BETWEEN ? AND ? AND LOWER("Index Name") = LOWER(?) '
        "ORDER BY date",
        [start, end, _INDEX_NAME],
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _compute_sharpe(daily_returns: np.ndarray) -> float:
    """Sharpe: mean/std(ddof=1) * sqrt(252), all daily returns."""
    n = len(daily_returns)
    if n < 2:
        return 0.0
    std = float(np.std(daily_returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(daily_returns) / std * np.sqrt(252))


def _compute_max_dd(daily_returns: np.ndarray) -> float:
    """Max drawdown from daily net returns (geometric compounding)."""
    if len(daily_returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def _count_trades(position: np.ndarray) -> int:
    """Count trade entries: sign flips or 0-to-nonzero transitions."""
    pos_sign = np.sign(position)
    sign_changes = np.zeros(len(position))
    sign_changes[1:] = np.abs(pos_sign[1:] - pos_sign[:-1])
    return int(np.sum((sign_changes > 0) & (pos_sign != 0)))


def _total_return(daily_returns: np.ndarray) -> float:
    """Total return from daily returns (geometric compounding)."""
    if len(daily_returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + daily_returns)
    return float(equity[-1] - 1.0)


# ---------------------------------------------------------------------------
# Strategy 1: S25 Divergence Flow Field
# ---------------------------------------------------------------------------


def _generate_s25_returns(
    store: MarketDataStore,
    start: date,
    end: date,
    entry_threshold: float = 0.3,
    signal_scale: float = 1.5,
    max_conviction: float = 0.8,
) -> pd.Series:
    """Generate S25 DFF daily net returns for NIFTY 50.

    Uses DivergenceFlowBuilder composite signal with T-1 lag.
    Best parameters from S25 sweep: threshold=0.3, scale=1.5.

    Returns
    -------
    pd.Series
        Daily net returns indexed by pd.Timestamp date.
    """
    from quantlaxmi.features.divergence_flow import DivergenceFlowBuilder

    print("  [S25 DFF] Building features...", flush=True)
    t0 = time.time()

    # Load NIFTY prices
    prices_df = _load_nifty_close(store, start.isoformat(), end.isoformat())
    if prices_df.empty or len(prices_df) < 30:
        print("  [S25 DFF] Insufficient price data")
        return pd.Series(dtype=float)

    # Build DFF features
    builder = DivergenceFlowBuilder()
    features = builder.build(start.isoformat(), end.isoformat(), store=store)

    if features.empty or "dff_composite" not in features.columns:
        print("  [S25 DFF] No DFF features available")
        return pd.Series(dtype=float)

    # Align features with prices on date
    features = features.reset_index()
    features.rename(columns={features.columns[0]: "date"}, inplace=True)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()

    merged = pd.merge(
        prices_df, features[["date", "dff_composite"]], on="date", how="inner"
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    if len(merged) < 10:
        print("  [S25 DFF] Insufficient merged data")
        return pd.Series(dtype=float)

    close = merged["close"].values
    composite = merged["dff_composite"].values

    # Causal position: position[t] = f(composite[t-1])
    lagged_composite = np.full(len(composite), np.nan)
    lagged_composite[1:] = composite[:-1]

    abs_signal = np.abs(lagged_composite)
    conviction = np.clip(abs_signal / signal_scale, 0.0, max_conviction)
    sign = np.sign(lagged_composite)
    position = np.where(abs_signal >= entry_threshold, sign * conviction, 0.0)
    position[0] = 0.0
    position = np.where(np.isnan(lagged_composite), 0.0, position)

    # Daily underlying returns
    daily_ret = np.zeros(len(close))
    daily_ret[1:] = close[1:] / close[:-1] - 1.0

    # Strategy gross returns
    gross_return = position * daily_ret

    # Transaction costs
    turnover = np.zeros(len(position))
    turnover[1:] = np.abs(position[1:] - position[:-1])
    turnover[0] = np.abs(position[0])
    cost = turnover * COST_PER_SIDE_NIFTY

    # Net returns
    net_return = gross_return - cost

    trades = _count_trades(position)
    elapsed = time.time() - t0
    print(
        f"  [S25 DFF] {len(merged)} days, {trades} trades, "
        f"Sharpe={_compute_sharpe(net_return):.2f} ({elapsed:.1f}s)",
        flush=True,
    )

    return pd.Series(
        net_return, index=merged["date"].values, name="s25_dff"
    )


# ---------------------------------------------------------------------------
# Strategy 2: S5 Hawkes / Tick Microstructure
# ---------------------------------------------------------------------------


def _is_trading_day(d: date) -> bool:
    """Simple trading day check: weekdays only (no holiday calendar)."""
    return d.weekday() < 5


def _hawkes_params_simple(
    timestamps: np.ndarray,
) -> tuple[float, float, float, float]:
    """Estimate Hawkes process parameters from trade arrival times.

    Simplified CPU-only version. Returns (mean_intensity, mu, alpha, beta).
    """
    if len(timestamps) < 20:
        return float("nan"), 1.0, 0.5, 1.5

    ts = timestamps.astype(np.float64)
    if ts[0] > 1e15:
        ts = ts / 1e9
    elif ts[0] > 1e12:
        ts = ts / 1e3
    ts = ts - ts[0]

    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt) < 10:
        return float("nan"), 1.0, 0.5, 1.5

    lambda_bar = len(timestamps) / max(ts[-1], 1e-6)

    bin_size = 10.0
    n_bins = max(int(ts[-1] / bin_size), 5)
    counts = np.histogram(ts, bins=n_bins)[0].astype(float)
    mean_c = np.mean(counts)
    var_c = np.var(counts, ddof=1)

    if mean_c > 0:
        ratio = var_c / mean_c
        n_est = max(0.01, min(0.95, 1 - 1 / math.sqrt(max(ratio, 1.01))))
    else:
        n_est = 0.3

    if len(counts) > 2:
        acf1 = abs(np.corrcoef(counts[:-1], counts[1:])[0, 1])
        acf1 = max(0.01, min(0.99, acf1))
        beta = -math.log(acf1) / bin_size
        beta = max(0.1, min(10.0, beta))
    else:
        beta = 1.5

    mu = lambda_bar * (1 - n_est)
    alpha = n_est * beta

    return float(lambda_bar), float(mu), float(alpha), float(beta)


def _generate_s5_returns(
    store: MarketDataStore,
    start: date,
    end: date,
    lookback: int = 60,
    entry_pctile: float = 0.80,
    cost_bps: float = 5,
) -> pd.Series:
    """Generate S5 Hawkes / tick microstructure daily net returns for NIFTY 50.

    Simplified pipeline:
      1. For each trading day with tick data, compute Hawkes intensity ratio
         (alpha/beta) as the key microstructure feature.
      2. Use causal trailing IC to determine direction (long/short).
      3. Enter when feature exceeds rolling percentile threshold.
      4. Position is binary (+1/-1) with T+1 execution (causal).

    Returns
    -------
    pd.Series
        Daily net returns indexed by pd.Timestamp date.
    """
    from scipy.stats import spearmanr

    print("  [S5 Hawkes] Building tick features...", flush=True)
    t0 = time.time()

    NIFTY_TOKEN = 256265

    # Check if ticks table exists with data in range
    tick_dates = store.available_dates("ticks")
    tick_dates = sorted(d for d in tick_dates if start <= d <= end)

    if len(tick_dates) < lookback + 10:
        print(f"  [S5 Hawkes] Only {len(tick_dates)} tick dates, need {lookback + 10}")
        return pd.Series(dtype=float)

    # Load NIFTY close prices for next-day return computation
    prices_df = _load_nifty_close(store, start.isoformat(), end.isoformat())
    if prices_df.empty:
        print("  [S5 Hawkes] No price data")
        return pd.Series(dtype=float)

    prices_df = prices_df.set_index("date")

    # Build daily Hawkes features from tick data
    daily_records: list[dict] = []

    for i, d in enumerate(tick_dates):
        df = store.sql(
            "SELECT timestamp, ltp FROM ticks "
            "WHERE date = ? AND instrument_token = ? "
            "AND timestamp >= '2000-01-01' AND ltp > 0.05 "
            "ORDER BY timestamp",
            [d.isoformat(), NIFTY_TOKEN],
        )
        if df is None or df.empty or len(df) < 200:
            continue

        prices = df["ltp"].values.astype(np.float64)
        timestamps = df["timestamp"].values

        # Hawkes parameters
        mean_intensity, mu, alpha, beta = _hawkes_params_simple(timestamps)
        hawkes_ratio = alpha / beta if beta > 0 else float("nan")

        daily_records.append({
            "date": pd.Timestamp(d),
            "hawkes_ratio": hawkes_ratio,
            "hawkes_mean": mean_intensity,
        })

        if (i + 1) % 50 == 0:
            print(
                f"    tick features: {i + 1}/{len(tick_dates)} "
                f"({len(daily_records)} valid, {time.time() - t0:.1f}s)",
                flush=True,
            )

    if len(daily_records) < lookback + 10:
        print(
            f"  [S5 Hawkes] Only {len(daily_records)} valid tick days, "
            f"need {lookback + 10}"
        )
        return pd.Series(dtype=float)

    feat_df = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)

    # Align with NIFTY close prices to get next-day return
    # next_day_return[t] = close[t+1]/close[t] - 1  (realized next day)
    price_dates = prices_df.index
    next_day_returns: list[float] = []
    for _, row in feat_df.iterrows():
        d = row["date"]
        # Find this date and the next date in price series
        mask_current = price_dates == d
        if not mask_current.any():
            next_day_returns.append(float("nan"))
            continue
        idx = price_dates.get_loc(d)
        if isinstance(idx, slice):
            idx = idx.start
        if idx + 1 < len(price_dates):
            curr_close = prices_df.iloc[idx]["close"]
            next_close = prices_df.iloc[idx + 1]["close"]
            next_day_returns.append(float((next_close - curr_close) / curr_close))
        else:
            next_day_returns.append(float("nan"))

    feat_df["next_day_return"] = next_day_returns

    # Backtest: causal trailing IC determines direction, hawkes_ratio > pctile -> entry
    feature = "hawkes_ratio"
    valid = feat_df.dropna(subset=[feature, "next_day_return"]).copy()
    valid = valid.sort_values("date").reset_index(drop=True)

    if len(valid) < lookback + 10:
        print(f"  [S5 Hawkes] Insufficient valid data after NaN drop")
        return pd.Series(dtype=float)

    cost_frac = cost_bps / 10_000
    all_dates: list[pd.Timestamp] = []
    all_daily_pnl: list[float] = []
    was_in_position = False  # track previous day's position for entry/exit cost

    for i in range(lookback, len(valid)):
        all_dates.append(valid["date"].iloc[i])

        # Causal IC from trailing window
        trail = slice(max(0, i - lookback), i)
        trail_feat = valid[feature].iloc[trail].values.astype(float)
        trail_ret = valid["next_day_return"].iloc[trail].values.astype(float)
        mask = ~(np.isnan(trail_feat) | np.isnan(trail_ret))
        if mask.sum() > 10:
            ic, _ = spearmanr(trail_feat[mask], trail_ret[mask])
            direction = 1 if ic > 0 else -1
        else:
            direction = 1

        # Rolling percentile threshold
        window = valid[feature].iloc[max(0, i - lookback + 1) : i + 1]
        threshold = window.quantile(entry_pctile)
        val = valid[feature].iloc[i]

        if val >= threshold:
            # Position at T, return realized at T+1
            ret = valid["next_day_return"].iloc[i] * direction
            is_entry = not was_in_position
            if is_entry:
                ret -= cost_frac  # entry cost
            all_daily_pnl.append(ret)
            was_in_position = True
        else:
            if was_in_position:
                # Exiting position — charge exit cost (no return on flat day)
                all_daily_pnl.append(-cost_frac)
            else:
                all_daily_pnl.append(0.0)
            was_in_position = False

    if not all_daily_pnl:
        return pd.Series(dtype=float)

    pnl_arr = np.array(all_daily_pnl)
    trades = int(np.sum(np.array(all_daily_pnl) != 0.0))
    elapsed = time.time() - t0
    print(
        f"  [S5 Hawkes] {len(all_daily_pnl)} days, {trades} trades, "
        f"Sharpe={_compute_sharpe(pnl_arr):.2f} ({elapsed:.1f}s)",
        flush=True,
    )

    return pd.Series(
        pnl_arr, index=pd.DatetimeIndex(all_dates), name="s5_hawkes"
    )


# ---------------------------------------------------------------------------
# Strategy 3: S4 IV Mean-Reversion
# ---------------------------------------------------------------------------


def _generate_s4_returns(
    store: MarketDataStore,
    start: date,
    end: date,
    iv_lookback: int = 30,
    entry_pctile: float = 0.80,
    exit_pctile: float = 0.50,
    hold_days: int = 7,
    cost_bps: float = 5,
) -> pd.Series:
    """Generate S4 IV Mean-Reversion daily net returns for NIFTY.

    Uses SANOS calibration to build ATM IV series, then mean-reversion
    entries on high IV percentile with causal T+1 execution.

    Returns
    -------
    pd.Series
        Daily net returns indexed by pd.Timestamp date. Includes all days
        (zero for flat days).
    """
    from quantlaxmi.strategies.s4_iv_mr.engine import build_iv_series, DayObs
    from quantlaxmi.strategies.s9_momentum.data import is_trading_day

    print("  [S4 IV MR] Building IV series (SANOS calibration)...", flush=True)
    t0 = time.time()

    # Build IV series (this is the expensive part — SANOS calibration per day)
    daily = build_iv_series(store, start, end, symbol="NIFTY")

    if len(daily) < iv_lookback + 10:
        print(f"  [S4 IV MR] Only {len(daily)} IV observations, need {iv_lookback + 10}")
        return pd.Series(dtype=float)

    print(
        f"  [S4 IV MR] IV series: {len(daily)} days, "
        f"{sum(1 for d in daily if d.sanos_ok)} SANOS OK ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Compute rolling IV percentile ranks (causal — uses lookback window ending at t)
    ivs = [d.atm_iv for d in daily]
    pctile_ranks: list[float] = []
    for i, v in enumerate(ivs):
        lookback_window = ivs[max(0, i - iv_lookback + 1) : i + 1]
        rank = sum(1 for x in lookback_window if x <= v) / len(lookback_window)
        pctile_ranks.append(rank)

    # Trading simulation with T+1 execution
    cost_frac = cost_bps / 10_000
    daily_pnl: list[float] = []
    daily_dates: list[pd.Timestamp] = []

    in_trade = False
    entry_idx = 0
    pending_entry = False

    for i in range(iv_lookback, len(daily)):
        obs = daily[i]
        daily_dates.append(pd.Timestamp(obs.date))

        # Execute pending entry from yesterday's signal
        if pending_entry and not in_trade:
            in_trade = True
            entry_idx = i
            pending_entry = False
            daily_pnl.append(-cost_frac)  # entry cost
            continue

        if not in_trade:
            # Check entry conditions (signal at close, execute T+1)
            if pctile_ranks[i] >= entry_pctile:
                pending_entry = True
                daily_pnl.append(0.0)
            else:
                daily_pnl.append(0.0)
        else:
            # In trade — compute daily P&L
            days_held = i - entry_idx
            spot_prev = daily[i - 1].spot
            spot_now = obs.spot
            day_return = (spot_now - spot_prev) / spot_prev

            # Check exit conditions
            should_exit = False
            if days_held >= hold_days:
                should_exit = True
            elif pctile_ranks[i] < exit_pctile:
                should_exit = True

            if should_exit or i == len(daily) - 1:
                # Charge exit cost on this day's return
                daily_pnl.append(day_return - cost_frac)
                in_trade = False
            else:
                daily_pnl.append(day_return)

    if not daily_pnl:
        return pd.Series(dtype=float)

    pnl_arr = np.array(daily_pnl)
    trades = sum(
        1 for i in range(1, len(pnl_arr))
        if pnl_arr[i] != 0.0 and pnl_arr[i - 1] == 0.0
    )
    elapsed = time.time() - t0
    print(
        f"  [S4 IV MR] {len(daily_pnl)} days, ~{trades} trades, "
        f"Sharpe={_compute_sharpe(pnl_arr):.2f} ({elapsed:.1f}s)",
        flush=True,
    )

    return pd.Series(
        pnl_arr, index=pd.DatetimeIndex(daily_dates), name="s4_iv_mr"
    )


# ---------------------------------------------------------------------------
# Ensemble methods
# ---------------------------------------------------------------------------


def _align_strategy_returns(
    strategy_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """Align strategy return series onto a common date index.

    Missing dates for a strategy are filled with 0.0 (flat = no position).

    Returns
    -------
    pd.DataFrame
        Columns = strategy names, index = dates, values = daily returns.
    """
    df = pd.DataFrame(strategy_returns)
    df = df.sort_index()
    df = df.fillna(0.0)
    return df


def ensemble_equal_weight(returns_df: pd.DataFrame) -> pd.Series:
    """Equal-weight ensemble: (1/N) * sum of strategy returns each day."""
    n = returns_df.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    return returns_df.mean(axis=1).rename("equal_weight")


def ensemble_inverse_vol(
    returns_df: pd.DataFrame,
    vol_window: int = 63,
) -> pd.Series:
    """Inverse-volatility ensemble: weight by 1/rolling_std.

    Uses 63-day trailing rolling std (causal). Days with insufficient
    history use equal weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Columns = strategies, rows = dates.
    vol_window : int
        Rolling window for volatility estimation.

    Returns
    -------
    pd.Series
        Ensemble daily returns.
    """
    n_strats = returns_df.shape[1]
    if n_strats == 0:
        return pd.Series(dtype=float)

    # Rolling std for each strategy (causal: uses trailing vol_window days)
    rolling_vol = returns_df.rolling(window=vol_window, min_periods=20).std(ddof=1)

    ensemble_rets = np.zeros(len(returns_df))
    strat_returns = returns_df.values  # (T, N)
    strat_vols = rolling_vol.values  # (T, N)

    for t in range(len(returns_df)):
        vols = strat_vols[t]
        rets = strat_returns[t]

        # Check for valid vols
        valid = ~np.isnan(vols) & (vols > 1e-12)
        if valid.sum() == 0:
            # Equal weight fallback
            ensemble_rets[t] = float(np.mean(rets))
            continue

        # Inverse vol weights (only for strategies with valid vol)
        inv_vols = np.zeros(n_strats)
        inv_vols[valid] = 1.0 / vols[valid]
        total_inv_vol = inv_vols.sum()
        if total_inv_vol < 1e-12:
            ensemble_rets[t] = float(np.mean(rets))
            continue

        weights = inv_vols / total_inv_vol
        ensemble_rets[t] = float(np.dot(weights, rets))

    return pd.Series(
        ensemble_rets, index=returns_df.index, name="inverse_vol"
    )


def ensemble_corr_adjusted(
    returns_df: pd.DataFrame,
    vol_window: int = 63,
    corr_threshold: float = 0.5,
    corr_penalty: float = 0.20,
) -> pd.Series:
    """Correlation-adjusted inverse-vol ensemble.

    Starts with inverse-vol weights, then penalizes strategies whose
    pairwise 63-day rolling correlation exceeds the threshold.

    For each pair (i, j) with corr > threshold, both weights are reduced
    by corr_penalty (20%). Weights are renormalized after penalization.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Columns = strategies, rows = dates.
    vol_window : int
        Rolling window for vol and corr estimation.
    corr_threshold : float
        Correlation level above which penalty kicks in.
    corr_penalty : float
        Multiplicative reduction factor (0.20 = reduce by 20%).

    Returns
    -------
    pd.Series
        Ensemble daily returns.
    """
    n_strats = returns_df.shape[1]
    if n_strats == 0:
        return pd.Series(dtype=float)

    # Rolling vol
    rolling_vol = returns_df.rolling(window=vol_window, min_periods=20).std(ddof=1)

    ensemble_rets = np.zeros(len(returns_df))
    strat_returns = returns_df.values
    strat_vols = rolling_vol.values
    col_names = returns_df.columns.tolist()

    for t in range(len(returns_df)):
        vols = strat_vols[t]
        rets = strat_returns[t]

        valid = ~np.isnan(vols) & (vols > 1e-12)
        if valid.sum() == 0:
            ensemble_rets[t] = float(np.mean(rets))
            continue

        # Inverse vol weights
        inv_vols = np.zeros(n_strats)
        inv_vols[valid] = 1.0 / vols[valid]
        total_inv_vol = inv_vols.sum()
        if total_inv_vol < 1e-12:
            ensemble_rets[t] = float(np.mean(rets))
            continue

        weights = inv_vols / total_inv_vol

        # Compute pairwise rolling correlation from trailing window
        if t >= vol_window:
            window_data = strat_returns[t - vol_window + 1 : t + 1]
            # Only if we have enough non-zero observations
            if len(window_data) >= 20:
                corr_matrix = np.corrcoef(window_data.T)
                # Penalize correlated pairs
                penalty = np.ones(n_strats)
                for i in range(n_strats):
                    for j in range(i + 1, n_strats):
                        if not np.isnan(corr_matrix[i, j]):
                            if abs(corr_matrix[i, j]) > corr_threshold:
                                penalty[i] *= (1.0 - corr_penalty)
                                penalty[j] *= (1.0 - corr_penalty)

                weights = weights * penalty
                w_sum = weights.sum()
                if w_sum > 1e-12:
                    weights = weights / w_sum

        ensemble_rets[t] = float(np.dot(weights, rets))

    return pd.Series(
        ensemble_rets, index=returns_df.index, name="corr_adjusted"
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _strategy_metrics(daily_returns: np.ndarray, name: str) -> dict:
    """Compute comprehensive metrics for a strategy or ensemble."""
    n = len(daily_returns)
    sharpe = _compute_sharpe(daily_returns)
    total_ret = _total_return(daily_returns)
    max_dd = _compute_max_dd(daily_returns)
    ann_ret = float((1 + total_ret) ** (252 / max(n, 1)) - 1) if n > 0 else 0.0

    # Count non-zero days as proxy for "in position" days
    in_position = np.sum(daily_returns != 0.0)

    # Win rate when in position
    if in_position > 0:
        wins = np.sum(daily_returns > 0)
        win_rate = float(wins / in_position)
    else:
        win_rate = 0.0

    # Daily vol (annualized)
    ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if n > 1 else 0.0

    return {
        "name": name,
        "n_days": n,
        "sharpe": sharpe,
        "total_return_pct": total_ret * 100,
        "ann_return_pct": ann_ret * 100,
        "max_dd_pct": max_dd * 100,
        "ann_vol_pct": ann_vol * 100,
        "win_rate": win_rate,
        "position_days": int(in_position),
    }


# ---------------------------------------------------------------------------
# Diversification ratio
# ---------------------------------------------------------------------------


def _diversification_ratio(returns_df: pd.DataFrame) -> float:
    """Diversification ratio: weighted avg vol / portfolio vol.

    Uses equal weights. Ratio > 1.0 means diversification is present.
    """
    n = returns_df.shape[1]
    if n < 2 or len(returns_df) < 2:
        return 1.0

    # Individual annualized vols
    individual_vols = returns_df.std(ddof=1) * np.sqrt(252)

    # Equally-weighted portfolio vol
    eq_portfolio = returns_df.mean(axis=1)
    portfolio_vol = float(eq_portfolio.std(ddof=1) * np.sqrt(252))

    if portfolio_vol < 1e-12:
        return 1.0

    avg_individual_vol = float(individual_vols.mean())
    return avg_individual_vol / portfolio_vol


# ---------------------------------------------------------------------------
# Main research runner
# ---------------------------------------------------------------------------


def run_ensemble_backtest(
    start: date = date(2025, 1, 1),
    end: date = date(2026, 2, 6),
) -> dict:
    """Run multi-strategy ensemble backtest on NIFTY 50.

    Returns
    -------
    dict
        Complete results including per-strategy metrics, correlations,
        ensemble metrics, and diversification ratio.
    """
    print("=" * 70)
    print("MULTI-STRATEGY ENSEMBLE BACKTEST — NIFTY 50")
    print("=" * 70)
    print(f"  Date range:   {start} to {end}")
    print(f"  Symbol:       {SYMBOL}")
    print(f"  Cost:         3 pts per side ({COST_PER_SIDE_NIFTY * 100:.4f}%)")
    print(f"  Sharpe:       ddof=1, sqrt(252), all days")
    print("=" * 70)
    print()

    store = MarketDataStore()
    t_total = time.time()

    # -------------------------------------------------------------------
    # 1. Generate per-strategy return streams
    # -------------------------------------------------------------------
    print("PHASE 1: Generate Strategy Returns")
    print("-" * 50)

    strategy_series: dict[str, pd.Series] = {}

    # S25 DFF
    try:
        s25_rets = _generate_s25_returns(store, start, end)
        if not s25_rets.empty:
            strategy_series["s25_dff"] = s25_rets
            print()
    except Exception as e:
        print(f"  [S25 DFF] FAILED: {e}\n")

    # S5 Hawkes
    try:
        s5_rets = _generate_s5_returns(store, start, end)
        if not s5_rets.empty:
            strategy_series["s5_hawkes"] = s5_rets
            print()
    except Exception as e:
        print(f"  [S5 Hawkes] FAILED: {e}\n")

    # S4 IV MR
    try:
        s4_rets = _generate_s4_returns(store, start, end)
        if not s4_rets.empty:
            strategy_series["s4_iv_mr"] = s4_rets
            print()
    except Exception as e:
        print(f"  [S4 IV MR] FAILED: {e}\n")

    if len(strategy_series) < 2:
        print(
            f"\nOnly {len(strategy_series)} strategies produced data. "
            f"Need at least 2 for ensemble. Aborting."
        )
        store.close()
        return {"error": "insufficient strategies"}

    # -------------------------------------------------------------------
    # 2. Align returns on common date index
    # -------------------------------------------------------------------
    print("PHASE 2: Align & Analyse")
    print("-" * 50)

    returns_df = _align_strategy_returns(strategy_series)
    print(f"  Aligned: {len(returns_df)} dates, {returns_df.shape[1]} strategies")
    print(f"  Date range: {returns_df.index.min()} to {returns_df.index.max()}")
    print()

    # -------------------------------------------------------------------
    # 3. Per-strategy standalone metrics
    # -------------------------------------------------------------------
    print("PHASE 3: Per-Strategy Metrics")
    print("-" * 50)

    strategy_metrics: dict[str, dict] = {}
    header = (
        f"  {'Strategy':<15} {'Days':>5} {'Sharpe':>7} {'Return':>8} "
        f"{'AnnRet':>8} {'MaxDD':>7} {'AnnVol':>8} {'WinR':>6} {'PosDays':>8}"
    )
    print(header)
    print("  " + "-" * 80)

    for col in returns_df.columns:
        rets = returns_df[col].values
        metrics = _strategy_metrics(rets, col)
        strategy_metrics[col] = metrics
        print(
            f"  {col:<15} {metrics['n_days']:5d} {metrics['sharpe']:7.2f} "
            f"{metrics['total_return_pct']:+7.2f}% {metrics['ann_return_pct']:+7.2f}% "
            f"{metrics['max_dd_pct']:6.2f}% {metrics['ann_vol_pct']:7.2f}% "
            f"{metrics['win_rate']*100:5.1f}% {metrics['position_days']:8d}"
        )
    print()

    # -------------------------------------------------------------------
    # 4. Pairwise correlation matrix
    # -------------------------------------------------------------------
    print("PHASE 4: Pairwise Correlations")
    print("-" * 50)

    corr_matrix = returns_df.corr()
    strat_names = returns_df.columns.tolist()

    # Print matrix header
    header_str = f"  {'':>15}"
    for name in strat_names:
        header_str += f"  {name:>12}"
    print(header_str)
    print("  " + "-" * (15 + 14 * len(strat_names)))

    for i, name_i in enumerate(strat_names):
        row_str = f"  {name_i:>15}"
        for j, name_j in enumerate(strat_names):
            row_str += f"  {corr_matrix.iloc[i, j]:12.4f}"
        print(row_str)
    print()

    # -------------------------------------------------------------------
    # 5. Ensemble methods
    # -------------------------------------------------------------------
    print("PHASE 5: Ensemble Methods")
    print("-" * 50)

    ensemble_results: dict[str, dict] = {}

    # Method 1: Equal weight
    eq_rets = ensemble_equal_weight(returns_df)
    eq_metrics = _strategy_metrics(eq_rets.values, "equal_weight")
    ensemble_results["equal_weight"] = eq_metrics

    # Method 2: Inverse volatility
    iv_rets = ensemble_inverse_vol(returns_df, vol_window=63)
    iv_metrics = _strategy_metrics(iv_rets.values, "inverse_vol")
    ensemble_results["inverse_vol"] = iv_metrics

    # Method 3: Correlation-adjusted
    ca_rets = ensemble_corr_adjusted(
        returns_df, vol_window=63, corr_threshold=0.5, corr_penalty=0.20
    )
    ca_metrics = _strategy_metrics(ca_rets.values, "corr_adjusted")
    ensemble_results["corr_adjusted"] = ca_metrics

    header = (
        f"  {'Method':<15} {'Days':>5} {'Sharpe':>7} {'Return':>8} "
        f"{'AnnRet':>8} {'MaxDD':>7} {'AnnVol':>8}"
    )
    print(header)
    print("  " + "-" * 65)

    for method_name, metrics in ensemble_results.items():
        print(
            f"  {method_name:<15} {metrics['n_days']:5d} {metrics['sharpe']:7.2f} "
            f"{metrics['total_return_pct']:+7.2f}% {metrics['ann_return_pct']:+7.2f}% "
            f"{metrics['max_dd_pct']:6.2f}% {metrics['ann_vol_pct']:7.2f}%"
        )
    print()

    # -------------------------------------------------------------------
    # 6. Diversification ratio
    # -------------------------------------------------------------------
    print("PHASE 6: Diversification Analysis")
    print("-" * 50)

    div_ratio = _diversification_ratio(returns_df)
    print(f"  Diversification Ratio (EW):  {div_ratio:.3f}")
    print(f"    (> 1.0 = portfolio vol < average individual vol)")
    print()

    # Per-strategy annualized vol
    individual_vols = returns_df.std(ddof=1) * np.sqrt(252)
    eq_portfolio = returns_df.mean(axis=1)
    portfolio_vol = float(eq_portfolio.std(ddof=1) * np.sqrt(252))

    print(f"  Individual Annualized Vols:")
    for col in returns_df.columns:
        print(f"    {col:<15}: {individual_vols[col]*100:7.2f}%")
    print(f"  EW Portfolio Vol:   {portfolio_vol*100:7.2f}%")
    print(f"  Avg Individual Vol: {individual_vols.mean()*100:7.2f}%")
    print()

    # -------------------------------------------------------------------
    # 7. Best strategy vs best ensemble
    # -------------------------------------------------------------------
    print("PHASE 7: Summary Comparison")
    print("-" * 50)

    all_metrics = {}
    all_metrics.update(strategy_metrics)
    all_metrics.update(ensemble_results)

    # Sort by Sharpe descending
    ranked = sorted(all_metrics.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    print(
        f"  {'Rank':>4} {'Name':<15} {'Sharpe':>7} {'Return':>8} "
        f"{'MaxDD':>7} {'AnnVol':>8}"
    )
    print("  " + "-" * 55)
    for rank, (name, metrics) in enumerate(ranked, 1):
        tag = ""
        if name in strategy_metrics:
            tag = " [strat]"
        else:
            tag = " [ensbl]"
        print(
            f"  {rank:4d} {name:<15} {metrics['sharpe']:7.2f} "
            f"{metrics['total_return_pct']:+7.2f}% {metrics['max_dd_pct']:6.2f}% "
            f"{metrics['ann_vol_pct']:7.2f}%{tag}"
        )
    print()

    # -------------------------------------------------------------------
    # 8. Save results
    # -------------------------------------------------------------------
    elapsed = time.time() - t_total
    print(f"Total elapsed: {elapsed:.1f}s")
    print("=" * 70)

    # Build JSON-serializable results
    results = {
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "symbol": SYMBOL,
        "n_strategies": len(strategy_series),
        "strategy_metrics": strategy_metrics,
        "correlation_matrix": corr_matrix.to_dict(),
        "ensemble_metrics": ensemble_results,
        "diversification_ratio": div_ratio,
        "individual_vols": {
            col: float(individual_vols[col]) for col in returns_df.columns
        },
        "portfolio_vol_ew": portfolio_vol,
        "ranking": [
            {"rank": r + 1, "name": n, "sharpe": m["sharpe"]}
            for r, (n, m) in enumerate(ranked)
        ],
    }

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as dt

    ts = dt.now().strftime("%Y-%m-%d_%H%M%S")
    json_path = RESULTS_DIR / f"ensemble_backtest_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON results saved to {json_path}")

    # Save daily returns CSV
    csv_path = RESULTS_DIR / f"ensemble_daily_returns_{ts}.csv"
    all_returns_df = returns_df.copy()
    all_returns_df["equal_weight"] = eq_rets.values
    all_returns_df["inverse_vol"] = iv_rets.values
    all_returns_df["corr_adjusted"] = ca_rets.values
    all_returns_df.index.name = "date"
    all_returns_df.to_csv(csv_path)
    print(f"Daily returns CSV saved to {csv_path}")

    store.close()
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from quantlaxmi.strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(
        description="Multi-Strategy Ensemble Backtest — NIFTY 50"
    )
    parser.add_argument(
        "--start", default="2025-01-01",
        help="Start date (YYYY-MM-DD, default: 2025-01-01)",
    )
    parser.add_argument(
        "--end", default="2026-02-06",
        help="End date (YYYY-MM-DD, default: 2026-02-06)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    with tee_to_results("ensemble_backtest"):
        run_ensemble_backtest(start=start, end=end)


if __name__ == "__main__":
    main()
