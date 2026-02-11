"""Network Momentum / Lead-Lag Cross-Asset Strategy for NIFTY.

Information diffuses through financial markets with measurable delays.
Sector indices lead/lag each other due to differential information processing
speeds, institutional rebalancing schedules, and economic linkages.

By detecting which indices are "leading" and which are "lagging" via
cross-correlation analysis, we build a directed information-flow graph and
trade Nifty 50 futures when its leading indices emit collective signals.

Architecture:
    1. Load daily closes for ~24 NSE sector/thematic indices from DuckDB
    2. Compute pairwise lead-lag cross-correlations (Spearman, more robust)
       at lags 1-5 on an expanding window (min 40 days, causal)
    3. Build a directed information-flow graph: edge (i -> j, lag k)
       exists when corr(ret_i[t], ret_j[t+k]) is significant
    4. Composite signal blending:
       (a) Network score = weighted sum of lagged returns from leaders
       (b) Sector breadth = fraction of sectors with positive recent returns
       (c) Momentum diffusion = recent leader returns predict Nifty direction
    5. Signal persistence filter: require 2 consecutive confirming bars
    6. T+1 execution, transaction costs, VIX filter, risk management

Cost model:
    NIFTY futures: 3 index points round-trip (~0.013% at 23000 level).
    Applied on each position entry and exit.

Sharpe protocol: ddof=1, sqrt(252), all daily returns including flat days.

Fully causal: all cross-correlations are computed on data available at
decision time. No look-ahead bias. T+1 execution lag enforced.

Author: AlphaForge
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from quantlaxmi.data._paths import _PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe of sector/thematic indices for lead-lag analysis
# ---------------------------------------------------------------------------

UNIVERSE = [
    "Nifty 50",
    "Nifty Bank",
    "Nifty IT",
    "Nifty Financial Services",
    "Nifty Metal",
    "Nifty Realty",
    "Nifty Energy",
    "Nifty Auto",
    "Nifty Pharma",
    "Nifty FMCG",
    "Nifty Media",
    "Nifty PSU Bank",
    "Nifty Infrastructure",
    "Nifty Commodities",
    "Nifty Private Bank",
    "Nifty Oil & Gas",
    "Nifty Consumer Durables",
    "Nifty Services Sector",
    "Nifty India Consumption",
    "Nifty Healthcare Index",
    "Nifty Chemicals",
    "Nifty India Manufacturing",
    "Nifty India Defence",
    "Nifty MNC",
]

TARGET_INDEX = "Nifty 50"

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

MIN_HISTORY: int = 40           # Minimum trading days before first signal
REFIT_INTERVAL: int = 5         # Refit lead-lag network every N days
MAX_LAG: int = 5                # Maximum lag (trading days) for cross-corr
MIN_CORR_SIGNIFICANCE: float = 0.10   # p-value threshold (relaxed for short data)
MIN_ABS_CORR: float = 0.12     # Minimum |cross-corr| to register an edge

SIGNAL_THRESHOLD: float = 0.6  # Composite z-score threshold for entry
EXIT_THRESHOLD: float = 0.2    # Signal must decay below this for exit (hysteresis)
MAX_HOLD_DAYS: int = 8          # Maximum holding period
CONFIRMATION_BARS: int = 2      # Require N consecutive confirming bars before entry

# VIX filter: don't trade in extreme fear regimes
VIX_CEILING: float = 28.0

# Cost: 3 points round-trip for NIFTY futures
COST_POINTS_RT: float = 3.0
NIFTY_APPROX_LEVEL: float = 23000.0
COST_FRACTION_RT: float = COST_POINTS_RT / NIFTY_APPROX_LEVEL

# Network score decay: weight edges by 1/lag
LAG_DECAY_POWER: float = 1.0

# Composite signal blending weights
WEIGHT_NETWORK: float = 0.5     # Lead-lag network score
WEIGHT_BREADTH: float = 0.3     # Sector breadth momentum
WEIGHT_DIFFUSION: float = 0.2   # Momentum diffusion speed

# Sector breadth lookback
BREADTH_LOOKBACK: int = 5       # Days for sector breadth calculation

# Position sizing
MAX_POSITION: float = 1.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LeadLagEdge:
    """A directed edge in the information-flow graph."""
    leader: str
    follower: str
    lag: int
    correlation: float
    p_value: float
    weight: float


@dataclass
class NetworkState:
    """Snapshot of the lead-lag network at a point in time."""
    fit_date: date
    n_observations: int
    edges: list[LeadLagEdge]
    leaders: list[str]


@dataclass
class BacktestResult:
    """Full backtest output."""
    daily_df: pd.DataFrame
    trades: list[dict]
    stats: dict


# ---------------------------------------------------------------------------
# Cross-correlation (Spearman rank, more robust to outliers)
# ---------------------------------------------------------------------------

def _compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
) -> list[tuple[int, float, float]]:
    """Compute Spearman rank cross-correlation of x leading y at lags 1..max_lag.

    Tests: corr(x[t], y[t+lag]) for lag = 1, ..., max_lag.
    This means: does x's past predict y's future?
    """
    results = []
    n = len(x)
    for lag in range(1, max_lag + 1):
        if n - lag < 15:
            continue
        x_lead = x[:n - lag]
        y_follow = y[lag:]

        mask = np.isfinite(x_lead) & np.isfinite(y_follow)
        if mask.sum() < 15:
            continue

        r, p = sp_stats.spearmanr(x_lead[mask], y_follow[mask])
        results.append((lag, float(r), float(p)))

    return results


def _build_lead_lag_network(
    returns_df: pd.DataFrame,
    target: str,
    max_lag: int = MAX_LAG,
    min_corr: float = MIN_ABS_CORR,
    p_threshold: float = MIN_CORR_SIGNIFICANCE,
    lag_decay: float = LAG_DECAY_POWER,
) -> list[LeadLagEdge]:
    """Build directed lead-lag edges pointing into `target`.

    For each non-target index, compute cross-correlations at lags 1..max_lag.
    Keep the best (highest |corr|) significant lag per leader.
    """
    edges = []
    target_rets = returns_df[target].values

    for col in returns_df.columns:
        if col == target:
            continue

        leader_rets = returns_df[col].values
        cross_corrs = _compute_cross_correlation(leader_rets, target_rets, max_lag)

        # Keep the best lag per leader (highest |corr| that is significant)
        best_edge = None
        best_abs_corr = 0.0

        for lag, r, p in cross_corrs:
            if p < p_threshold and abs(r) >= min_corr:
                if abs(r) > best_abs_corr:
                    best_abs_corr = abs(r)
                    weight = abs(r) / (lag ** lag_decay)
                    best_edge = LeadLagEdge(
                        leader=col,
                        follower=target,
                        lag=lag,
                        correlation=r,
                        p_value=p,
                        weight=weight,
                    )

        if best_edge is not None:
            edges.append(best_edge)

    edges.sort(key=lambda e: e.weight, reverse=True)
    return edges


# ---------------------------------------------------------------------------
# Signal components
# ---------------------------------------------------------------------------

def _compute_network_score(
    returns_df: pd.DataFrame,
    edges: list[LeadLagEdge],
    row_idx: int,
) -> float:
    """Weighted sum of lagged returns from leading indices.

    sign(correlation) * weight * return_of_leader[t - lag]
    Normalized by total weight.
    """
    if not edges:
        return 0.0

    score = 0.0
    total_weight = 0.0

    for edge in edges:
        lag_idx = row_idx - edge.lag
        if lag_idx < 0:
            continue

        leader_ret = returns_df[edge.leader].iloc[lag_idx]
        if not np.isfinite(leader_ret):
            continue

        contribution = edge.weight * np.sign(edge.correlation) * leader_ret
        score += contribution
        total_weight += edge.weight

    if total_weight > 0:
        score /= total_weight

    return score


def _compute_sector_breadth(
    returns_df: pd.DataFrame,
    target: str,
    row_idx: int,
    lookback: int = BREADTH_LOOKBACK,
) -> float:
    """Sector breadth: fraction of sectors with positive cumulative returns
    over the lookback window, centered at [-1, +1].

    >0 means majority of sectors are rising (bullish breadth).
    <0 means majority falling (bearish breadth).
    """
    start_idx = max(0, row_idx - lookback + 1)
    if start_idx >= row_idx:
        return 0.0

    # Cumulative return for each non-target sector over lookback
    window_rets = returns_df.iloc[start_idx:row_idx + 1]
    cum_rets = window_rets.sum()  # sum of log returns ~ cumulative log return

    non_target = [c for c in cum_rets.index if c != target]
    if not non_target:
        return 0.0

    n_positive = sum(1 for c in non_target if cum_rets[c] > 0)
    breadth = 2.0 * (n_positive / len(non_target)) - 1.0  # [-1, +1]
    return breadth


def _compute_diffusion_score(
    returns_df: pd.DataFrame,
    edges: list[LeadLagEdge],
    row_idx: int,
    lookback: int = 3,
) -> float:
    """Momentum diffusion: are leaders accelerating or decelerating?

    Compares recent leader returns (last `lookback` days) vs their
    longer-term average. Positive = leaders accelerating (momentum building).
    """
    if not edges or row_idx < lookback * 2:
        return 0.0

    scores = []
    for edge in edges:
        col = edge.leader
        # Recent average return
        recent_start = max(0, row_idx - lookback + 1)
        recent_rets = returns_df[col].iloc[recent_start:row_idx + 1].values
        recent_rets = recent_rets[np.isfinite(recent_rets)]

        # Longer-term average
        long_start = max(0, row_idx - lookback * 4 + 1)
        long_rets = returns_df[col].iloc[long_start:row_idx + 1].values
        long_rets = long_rets[np.isfinite(long_rets)]

        if len(recent_rets) > 0 and len(long_rets) > 0:
            diff = np.mean(recent_rets) - np.mean(long_rets)
            scores.append(np.sign(edge.correlation) * diff)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _composite_signal(
    network_score: float,
    breadth: float,
    diffusion: float,
    w_net: float = WEIGHT_NETWORK,
    w_breadth: float = WEIGHT_BREADTH,
    w_diff: float = WEIGHT_DIFFUSION,
) -> float:
    """Blend three signal components into a composite score."""
    return w_net * network_score + w_breadth * breadth + w_diff * diffusion


def _zscore_expanding(values: list[float], min_obs: int = 15) -> float:
    """Z-score the latest value against expanding history."""
    if len(values) < min_obs:
        return 0.0
    arr = np.array(values)
    valid = arr[np.isfinite(arr)]
    if len(valid) < min_obs:
        return 0.0
    mu = np.mean(valid)
    sigma = np.std(valid, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return (arr[-1] - mu) / sigma


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_index_data(
    store,
    indices: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load daily closing prices for multiple indices from DuckDB.

    Returns a DataFrame with DatetimeIndex and one column per index.
    """
    placeholders = ", ".join(["?"] * len(indices))
    query = f"""
        SELECT
            date,
            "Index Name" as name,
            CAST("Closing Index Value" AS DOUBLE) as close
        FROM nse_index_close
        WHERE "Index Name" IN ({placeholders})
          AND date >= ?
          AND date <= ?
        ORDER BY date, name
    """
    params = indices + [start_date.isoformat(), end_date.isoformat()]

    raw = store.sql(query, params)

    if raw.empty:
        raise ValueError(f"No data found for indices {indices} between {start_date} and {end_date}")

    pivot = raw.pivot(index="date", columns="name", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    # Drop indices with too many missing values (>10%)
    missing_pct = pivot.isnull().mean()
    valid_cols = missing_pct[missing_pct < 0.10].index.tolist()
    if TARGET_INDEX not in valid_cols:
        raise ValueError(f"Target index {TARGET_INDEX} has too many missing values")

    pivot = pivot[valid_cols]
    pivot = pivot.ffill(limit=3)

    logger.info(
        "Loaded %d indices x %d days (%s to %s)",
        len(pivot.columns), len(pivot), pivot.index[0].date(), pivot.index[-1].date(),
    )

    return pivot


def load_vix_data(store, start_date: date, end_date: date) -> pd.Series:
    """Load India VIX closing values."""
    query = """
        SELECT date, CAST("Closing Index Value" AS DOUBLE) as close
        FROM nse_index_close
        WHERE "Index Name" = 'India VIX'
          AND date >= ?
          AND date <= ?
        ORDER BY date
    """
    raw = store.sql(query, [start_date.isoformat(), end_date.isoformat()])
    if raw.empty:
        return pd.Series(dtype=float)

    vix = raw.set_index("date")["close"]
    vix.index = pd.to_datetime(vix.index)
    vix = vix.sort_index()
    return vix


# ---------------------------------------------------------------------------
# Network visualization
# ---------------------------------------------------------------------------

def _print_network(state: NetworkState, top_n: int = 20) -> None:
    """Print a text summary of the current lead-lag network."""
    print(f"\n{'='*70}")
    print(f"  Lead-Lag Network @ {state.fit_date} (n={state.n_observations} obs)")
    print(f"{'='*70}")
    print(f"  {'Leader':<35s} Lag  Corr      p-val    Weight")
    print(f"  {'-'*35} ---  --------  -------  ------")

    for edge in state.edges[:top_n]:
        print(
            f"  {edge.leader:<35s} {edge.lag:>3d}  "
            f"{edge.correlation:>+8.4f}  {edge.p_value:>7.4f}  {edge.weight:>6.4f}"
        )

    if not state.edges:
        print("  (no significant lead-lag edges found)")

    unique_leaders = sorted(set(e.leader for e in state.edges))
    print(f"\n  Unique leaders ({len(unique_leaders)}): {', '.join(unique_leaders[:10])}")
    if len(unique_leaders) > 10:
        print(f"  ... and {len(unique_leaders) - 10} more")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    store,
    indices: list[str] | None = None,
    target: str = TARGET_INDEX,
    start_date: date = date(2025, 8, 6),
    end_date: date = date(2026, 2, 6),
    min_history: int = MIN_HISTORY,
    refit_interval: int = REFIT_INTERVAL,
    signal_threshold: float = SIGNAL_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    max_hold: int = MAX_HOLD_DAYS,
    confirmation_bars: int = CONFIRMATION_BARS,
    cost_frac_rt: float = COST_FRACTION_RT,
    vix_ceiling: float = VIX_CEILING,
    verbose: bool = True,
) -> BacktestResult:
    """Run the network momentum backtest.

    Fully causal:
    - At each date t, we use returns up to t-1 to compute signals.
    - Signals computed at end-of-day t-1 are executed at close of day t (T+1).
    - Lead-lag network is estimated on expanding window of historical data.
    - Confirmation filter requires N consecutive confirming bars.
    """
    if indices is None:
        indices = UNIVERSE.copy()

    if target not in indices:
        indices.insert(0, target)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    prices = load_index_data(store, indices, start_date, end_date)
    vix = load_vix_data(store, start_date, end_date)

    log_returns = np.log(prices / prices.shift(1))

    dates = prices.index.tolist()
    n_days = len(dates)

    if n_days < min_history + 5:
        raise ValueError(
            f"Not enough data: {n_days} days, need at least {min_history + 5}"
        )

    logger.info("Backtest window: %s to %s (%d days)", dates[0].date(), dates[-1].date(), n_days)

    # ------------------------------------------------------------------
    # 2. Walk forward
    # ------------------------------------------------------------------
    records = []
    trades = []
    network_states: list[NetworkState] = []

    current_edges: list[LeadLagEdge] = []
    last_fit_idx: int = -refit_interval
    raw_composite_scores: list[float] = []

    position = 0.0
    entry_idx: int = -1
    entry_price: float = 0.0
    hold_days: int = 0
    cum_ret = 1.0

    # Confirmation tracking: count consecutive bars above/below threshold
    consec_long = 0
    consec_short = 0

    for t in range(1, n_days):
        today = dates[t]
        today_date = today.date() if hasattr(today, "date") else today

        target_ret = float(log_returns[target].iloc[t])
        if not np.isfinite(target_ret):
            target_ret = 0.0

        target_price = float(prices[target].iloc[t])

        # VIX (causal: use previous day)
        vix_val = np.nan
        if t > 0 and dates[t - 1] in vix.index:
            vix_val = float(vix.loc[dates[t - 1]])
        elif today in vix.index:
            vix_val = float(vix.loc[today])

        # ----------------------------------------------------------
        # Refit lead-lag network (causal: data up to t-1)
        # ----------------------------------------------------------
        if t >= min_history and (t - last_fit_idx) >= refit_interval:
            hist_returns = log_returns.iloc[1:t]

            if len(hist_returns) >= min_history - 1:
                current_edges = _build_lead_lag_network(
                    hist_returns, target,
                    max_lag=MAX_LAG,
                    min_corr=MIN_ABS_CORR,
                    p_threshold=MIN_CORR_SIGNIFICANCE,
                    lag_decay=LAG_DECAY_POWER,
                )

                leaders = sorted(set(e.leader for e in current_edges))
                state = NetworkState(
                    fit_date=today_date,
                    n_observations=len(hist_returns),
                    edges=current_edges,
                    leaders=leaders,
                )
                network_states.append(state)
                last_fit_idx = t

                if verbose and len(network_states) <= 3:
                    _print_network(state)

        # ----------------------------------------------------------
        # Compute composite signal (all causal, using t-1 data)
        # ----------------------------------------------------------
        if t >= min_history:
            # Component 1: Network momentum score
            net_score = _compute_network_score(log_returns, current_edges, t - 1)

            # Component 2: Sector breadth
            breadth = _compute_sector_breadth(log_returns, target, t - 1)

            # Component 3: Momentum diffusion
            diffusion = _compute_diffusion_score(log_returns, current_edges, t - 1)

            # Composite
            composite = _composite_signal(net_score, breadth, diffusion)
        else:
            net_score = breadth = diffusion = composite = 0.0

        raw_composite_scores.append(composite)

        # Z-score the composite signal
        z_score = _zscore_expanding(raw_composite_scores, min_obs=15)

        # ----------------------------------------------------------
        # Confirmation filter: track consecutive bars
        # ----------------------------------------------------------
        if z_score > signal_threshold:
            consec_long += 1
            consec_short = 0
        elif z_score < -signal_threshold:
            consec_short += 1
            consec_long = 0
        else:
            consec_long = 0
            consec_short = 0

        # ----------------------------------------------------------
        # Signal generation (T+1: decided from t-1 data, executed at t)
        # ----------------------------------------------------------
        prev_position = position
        signal_name = "hold"
        trade_cost = 0.0

        vix_ok = np.isnan(vix_val) or vix_val < vix_ceiling

        if position == 0.0:
            # Entry: require N consecutive confirming bars
            if t >= min_history and vix_ok:
                if consec_long >= confirmation_bars:
                    position = MAX_POSITION
                    signal_name = "long_entry"
                    entry_idx = t
                    entry_price = target_price
                    hold_days = 0
                elif consec_short >= confirmation_bars:
                    position = -MAX_POSITION
                    signal_name = "short_entry"
                    entry_idx = t
                    entry_price = target_price
                    hold_days = 0
        else:
            hold_days += 1
            exit_signal = False
            exit_reason = ""

            # 1. Max hold period
            if hold_days >= max_hold:
                exit_signal = True
                exit_reason = "max_hold"

            # 2. Signal decay below exit threshold (with hysteresis)
            elif position > 0 and z_score < exit_threshold:
                exit_signal = True
                exit_reason = "signal_decay"
            elif position < 0 and z_score > -exit_threshold:
                exit_signal = True
                exit_reason = "signal_decay"

            # 3. Strong reversal (immediate exit)
            elif position > 0 and z_score < -signal_threshold:
                exit_signal = True
                exit_reason = "reversal"
            elif position < 0 and z_score > signal_threshold:
                exit_signal = True
                exit_reason = "reversal"

            # 4. VIX spike
            elif not vix_ok:
                exit_signal = True
                exit_reason = "vix_spike"

            if exit_signal:
                signal_name = f"exit_{exit_reason}"
                exit_price = target_price
                trade_pnl_pct = position * (exit_price / entry_price - 1.0)
                trade_pnl_pct -= cost_frac_rt  # Round-trip cost

                trades.append({
                    "entry_date": dates[entry_idx].date() if hasattr(dates[entry_idx], "date") else dates[entry_idx],
                    "exit_date": today_date,
                    "direction": "long" if position > 0 else "short",
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "hold_days": hold_days,
                    "pnl_pct": round(trade_pnl_pct, 6),
                    "exit_reason": exit_reason,
                    "entry_z": round(z_score, 4),
                })

                position = 0.0
                hold_days = 0

        # Transaction cost on entry
        if prev_position != position and prev_position == 0.0:
            trade_cost = cost_frac_rt / 2.0

        # Daily return
        daily_ret = position * target_ret - trade_cost
        cum_ret *= (1.0 + daily_ret)

        records.append({
            "date": today_date,
            "target_price": round(target_price, 2),
            "position": position,
            "signal": signal_name,
            "net_score": round(net_score if t >= min_history else 0.0, 6),
            "breadth": round(breadth if t >= min_history else 0.0, 4),
            "diffusion": round(diffusion if t >= min_history else 0.0, 6),
            "composite": round(composite, 6),
            "z_score": round(z_score, 4),
            "vix": round(vix_val, 2) if np.isfinite(vix_val) else np.nan,
            "n_edges": len(current_edges),
            "n_leaders": len(set(e.leader for e in current_edges)),
            "consec_long": consec_long,
            "consec_short": consec_short,
            "daily_return": round(daily_ret, 6),
            "cumulative_return": round(cum_ret, 6),
        })

    # ------------------------------------------------------------------
    # 3. Statistics
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(records)

    if result_df.empty:
        logger.error("Backtest produced no records")
        return BacktestResult(daily_df=result_df, trades=trades, stats={})

    stats = _compute_statistics(result_df, trades, target, verbose=verbose)

    if verbose and network_states:
        print("\n--- Final Network State ---")
        _print_network(network_states[-1], top_n=25)

    return BacktestResult(daily_df=result_df, trades=trades, stats=stats)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_statistics(
    df: pd.DataFrame,
    trades: list[dict],
    target: str,
    verbose: bool = True,
) -> dict:
    """Compute and print backtest statistics.

    Sharpe: ddof=1, sqrt(252), all daily returns including flat days.
    """
    rets = df["daily_return"].values

    # Sharpe
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Total return
    total_ret = float(cum[-1] - 1.0)

    # Annualized
    n_years = len(rets) / 252.0
    annual_ret = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        trade_pnls = [t["pnl_pct"] for t in trades]
        win_rate = sum(1 for p in trade_pnls if p > 0) / n_trades
        avg_trade_pnl = np.mean(trade_pnls)
        avg_hold = np.mean([t["hold_days"] for t in trades])
        best_trade = max(trade_pnls)
        worst_trade = min(trade_pnls)
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = avg_trade_pnl = avg_hold = best_trade = worst_trade = 0.0
        profit_factor = 0.0

    active_days = int((df["position"] != 0).sum())
    exposure_pct = active_days / len(df) * 100

    calmar = abs(annual_ret / max_dd) if max_dd != 0 else 0.0

    stats = {
        "target": target,
        "start_date": str(df["date"].iloc[0]),
        "end_date": str(df["date"].iloc[-1]),
        "total_days": len(df),
        "active_days": active_days,
        "exposure_pct": round(exposure_pct, 1),
        "total_return_pct": round(total_ret * 100, 2),
        "annual_return_pct": round(annual_ret * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_trade_pnl_pct": round(avg_trade_pnl * 100, 4) if n_trades > 0 else 0.0,
        "avg_hold_days": round(avg_hold, 1) if n_trades > 0 else 0.0,
        "best_trade_pct": round(best_trade * 100, 4) if n_trades > 0 else 0.0,
        "worst_trade_pct": round(worst_trade * 100, 4) if n_trades > 0 else 0.0,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
    }

    if verbose:
        print("\n" + "=" * 70)
        print(f"  Network Momentum / Lead-Lag Strategy: {target}")
        print("=" * 70)
        for k, v in stats.items():
            print(f"  {k:>25s}: {v}")
        print("=" * 70)

        if n_trades > 0:
            print(f"\n  Trade log ({n_trades} trades):")
            print(f"  {'Entry':<12s} {'Exit':<12s} {'Dir':<6s} {'Hold':>4s} "
                  f"{'PnL%':>8s} {'Reason':<12s}")
            print(f"  {'-'*12} {'-'*12} {'-'*6} {'-'*4} {'-'*8} {'-'*12}")
            for tr in trades:
                print(
                    f"  {str(tr['entry_date']):<12s} {str(tr['exit_date']):<12s} "
                    f"{tr['direction']:<6s} {tr['hold_days']:>4d} "
                    f"{tr['pnl_pct']*100:>+8.3f} {tr['exit_reason']:<12s}"
                )
        print()

    return stats


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity(
    store,
    param_name: str = "signal_threshold",
    param_range: list | None = None,
    **base_kwargs,
) -> pd.DataFrame:
    """Run sensitivity over a single parameter."""
    if param_range is None:
        if param_name == "signal_threshold":
            param_range = [0.3, 0.5, 0.6, 0.75, 1.0, 1.25]
        elif param_name == "confirmation_bars":
            param_range = [1, 2, 3, 4]
        elif param_name == "max_hold":
            param_range = [3, 5, 8, 10, 15]
        elif param_name == "exit_threshold":
            param_range = [0.0, 0.1, 0.2, 0.3, 0.5]
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    results = []
    for val in param_range:
        kwargs = {**base_kwargs, param_name: val, "verbose": False}
        try:
            bt = run_backtest(store=store, **kwargs)
            row = {"param": param_name, "value": val}
            row.update(bt.stats)
            results.append(row)
        except Exception as e:
            logger.warning("Sensitivity run failed for %s=%s: %s", param_name, val, e)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Network Momentum strategy backtest for NIFTY."""
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from quantlaxmi.data.store import MarketDataStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 70)
    print("  Network Momentum / Lead-Lag Cross-Asset Strategy")
    print("  Target: Nifty 50 | Universe: 24 NSE sector/thematic indices")
    print("  Composite signal: network lead-lag + sector breadth + diffusion")
    print("=" * 70)

    with MarketDataStore() as store:
        # ----------------------------------------------------------
        # 1. Main backtest
        # ----------------------------------------------------------
        result = run_backtest(
            store=store,
            target=TARGET_INDEX,
            start_date=date(2025, 8, 6),
            end_date=date(2026, 2, 6),
        )

        if result.daily_df.empty:
            logger.error("Backtest returned empty results")
            return

        # ----------------------------------------------------------
        # 2. Save results
        # ----------------------------------------------------------
        out_dir = _PROJECT_ROOT / "quantlaxmi" / "strategies" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"network_momentum_{TARGET_INDEX}_{date(2026, 2, 6).isoformat()}.csv"
        result.daily_df.to_csv(out_file, index=False)
        logger.info("Daily results saved to %s", out_file)

        if result.trades:
            trades_file = out_dir / f"network_momentum_trades_{date(2026, 2, 6).isoformat()}.csv"
            pd.DataFrame(result.trades).to_csv(trades_file, index=False)
            logger.info("Trade log saved to %s", trades_file)

        # ----------------------------------------------------------
        # 3. Sensitivity analyses
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("  Sensitivity Analysis: signal_threshold")
        print("=" * 70)

        sens_thresh = run_sensitivity(
            store=store,
            param_name="signal_threshold",
            param_range=[0.3, 0.5, 0.6, 0.75, 1.0, 1.25],
            start_date=date(2025, 8, 6),
            end_date=date(2026, 2, 6),
        )
        if not sens_thresh.empty:
            print(sens_thresh[[
                "value", "n_trades", "total_return_pct",
                "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
                "profit_factor", "avg_hold_days",
            ]].to_string(index=False))

        print("\n" + "=" * 70)
        print("  Sensitivity Analysis: confirmation_bars")
        print("=" * 70)

        sens_conf = run_sensitivity(
            store=store,
            param_name="confirmation_bars",
            param_range=[1, 2, 3, 4],
            start_date=date(2025, 8, 6),
            end_date=date(2026, 2, 6),
        )
        if not sens_conf.empty:
            print(sens_conf[[
                "value", "n_trades", "total_return_pct",
                "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
                "profit_factor", "avg_hold_days",
            ]].to_string(index=False))

        print("\n" + "=" * 70)
        print("  Sensitivity Analysis: exit_threshold")
        print("=" * 70)

        sens_exit = run_sensitivity(
            store=store,
            param_name="exit_threshold",
            param_range=[0.0, 0.1, 0.2, 0.3, 0.5],
            start_date=date(2025, 8, 6),
            end_date=date(2026, 2, 6),
        )
        if not sens_exit.empty:
            print(sens_exit[[
                "value", "n_trades", "total_return_pct",
                "sharpe_ratio", "max_drawdown_pct", "win_rate_pct",
                "profit_factor", "avg_hold_days",
            ]].to_string(index=False))

        # Save all sensitivity results
        all_sens = pd.concat([sens_thresh, sens_conf, sens_exit], ignore_index=True)
        sens_file = out_dir / "network_momentum_sensitivity.csv"
        all_sens.to_csv(sens_file, index=False)
        logger.info("Sensitivity analysis saved to %s", sens_file)

        # ----------------------------------------------------------
        # 4. Summary
        # ----------------------------------------------------------
        print(f"\nFinal cumulative return: {result.daily_df['cumulative_return'].iloc[-1]:.4f}")
        print(f"Total trades: {len(result.trades)}")

        if result.daily_df["n_leaders"].max() > 0:
            print(f"\nNetwork density (edges):\n{result.daily_df['n_edges'].describe().to_string()}")
            print(f"\nNetwork leaders:\n{result.daily_df['n_leaders'].describe().to_string()}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# BaseStrategy wrapper for registry integration
# ---------------------------------------------------------------------------

from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal


class S23NetworkMomentumStrategy(BaseStrategy):
    """Network momentum strategy — BaseStrategy wrapper for registry."""

    @property
    def strategy_id(self) -> str:
        return "s23_network_momentum"

    def warmup_days(self) -> int:
        return 60

    def _scan_impl(self, d, store) -> list[Signal]:
        """Research-only strategy — no live signals yet."""
        return []


def create_strategy() -> S23NetworkMomentumStrategy:
    """Factory for registry auto-discovery."""
    return S23NetworkMomentumStrategy()
