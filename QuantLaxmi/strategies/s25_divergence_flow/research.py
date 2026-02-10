"""S25: Divergence Flow Field Research — Vectorized Backtest.

Uses DivergenceFlowBuilder to construct Helmholtz-decomposition features
from NSE 4-party participant OI, then runs a fully vectorized backtest
on NIFTY 50 and NIFTY BANK with realistic per-leg index-point costs.

Signal: composite[t-1] → position[t] (strictly causal).
Costs: per-leg index points (3 pts NIFTY / ~22000, 5 pts BANKNIFTY / ~50000).
Sharpe: mean(daily_rets) / std(daily_rets, ddof=1) * sqrt(252), all days.

Sweeps:
  - entry_threshold: [0.3, 0.5, 0.7]
  - signal_scale:    [1.5, 2.0, 3.0]
  - symbols:         NIFTY 50, NIFTY BANK

Walk-Forward:
  - Training window: 126 trading days (~6 months, z-score warmup only)
  - Test window: 63 trading days (~3 months, non-overlapping OOS evaluation)
  - Step: 63 trading days (4-5 test folds over 2024-06-01 to 2026-02-06)

Usage:
    python -m strategies.s25_divergence_flow.research --start 2025-01-01 --end 2026-02-06
    python -m strategies.s25_divergence_flow.research --sweep
    python -m strategies.s25_divergence_flow.research --walkforward
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.store import MarketDataStore
from features.divergence_flow import DivergenceFlowBuilder, DFFConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = ["NIFTY 50", "NIFTY BANK"]

# Exact "Index Name" values used in nse_index_close
_INDEX_NAME: dict[str, str] = {
    "NIFTY 50": "Nifty 50",
    "NIFTY BANK": "Nifty Bank",
}

# Per-leg transaction cost in index points (bid-ask + fees)
# NIFTY: ~3 pts per side on a ~22000 index level ≈ 0.000136 per side
# BANKNIFTY: ~5 pts per side on a ~50000 index level ≈ 0.0001 per side
COST_PER_SIDE: dict[str, float] = {
    "NIFTY 50": 3.0 / 22000.0,      # ~0.000136
    "NIFTY BANK": 5.0 / 50000.0,    # ~0.0001
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_index_close(
    store: MarketDataStore,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Load daily close prices from nse_index_close.

    Uses exact match with LOWER() to avoid ILIKE traps.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``close``. Sorted by date ascending.
    """
    index_name = _INDEX_NAME.get(symbol, symbol)
    df = store.sql(
        'SELECT date, "Closing Index Value" AS close '
        "FROM nse_index_close "
        'WHERE date BETWEEN ? AND ? AND LOWER("Index Name") = LOWER(?) '
        "ORDER BY date",
        [start, end, index_name],
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Vectorized backtest
# ---------------------------------------------------------------------------


def backtest_dff(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY 50",
    entry_threshold: float = 0.5,
    signal_scale: float = 2.0,
    max_conviction: float = 0.8,
    dff_config: DFFConfig | None = None,
) -> dict:
    """Run a fully vectorized backtest for the DFF strategy on one symbol.

    Signal generation (strictly causal):
        raw_signal[t] = composite[t-1]   (shift(1) — uses only data up to t-1)
        conviction[t] = clip(|raw_signal[t]| / signal_scale, 0, max_conviction)
        position[t]   = sign(raw_signal[t]) * conviction[t]  IF |raw_signal[t]| >= entry_threshold ELSE 0

    Returns:
        strategy_return[t] = position[t] * underlying_return[t]
        cost[t]            = |position[t] - position[t-1]| * cost_per_side

    Parameters
    ----------
    store : MarketDataStore
    start, end : date
        Backtest date range.
    symbol : str
        Index name (must be in SYMBOLS).
    entry_threshold : float
        Minimum |composite| to open a position.
    signal_scale : float
        Divisor to map composite → conviction.
    max_conviction : float
        Cap on absolute position size.
    dff_config : DFFConfig, optional
        Custom DFF configuration.

    Returns
    -------
    dict
        Backtest results including Sharpe, return, drawdown, etc.
    """
    cost_per_side = COST_PER_SIDE.get(symbol, 3.0 / 22000.0)

    # ---------------------------------------------------------------
    # 1. Load price data
    # ---------------------------------------------------------------
    prices_df = _load_index_close(
        store, symbol, start.isoformat(), end.isoformat()
    )
    if prices_df.empty or len(prices_df) < 30:
        return _empty_result(symbol, entry_threshold, signal_scale)

    # ---------------------------------------------------------------
    # 2. Build DFF features (with lookback buffer for warm-up)
    # ---------------------------------------------------------------
    builder = DivergenceFlowBuilder(config=dff_config)
    features = builder.build(start.isoformat(), end.isoformat(), store=store)

    if features.empty or "dff_composite" not in features.columns:
        return _empty_result(symbol, entry_threshold, signal_scale)

    # ---------------------------------------------------------------
    # 3. Align features with prices on date
    # ---------------------------------------------------------------
    features = features.reset_index()
    features.rename(columns={features.columns[0]: "date"}, inplace=True)
    features["date"] = pd.to_datetime(features["date"])

    merged = pd.merge(prices_df, features[["date", "dff_composite"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if len(merged) < 10:
        return _empty_result(symbol, entry_threshold, signal_scale)

    close = merged["close"].values
    composite = merged["dff_composite"].values

    # ---------------------------------------------------------------
    # 4. Causal position sizing: position[t] = f(composite[t-1])
    # ---------------------------------------------------------------
    # shift composite by 1 — position today depends on yesterday's signal
    lagged_composite = np.full(len(composite), np.nan)
    lagged_composite[1:] = composite[:-1]

    # Raw position: sign * conviction, zeroed when below threshold
    abs_signal = np.abs(lagged_composite)
    conviction = np.clip(abs_signal / signal_scale, 0.0, max_conviction)
    sign = np.sign(lagged_composite)
    position = np.where(abs_signal >= entry_threshold, sign * conviction, 0.0)
    # First day has no signal: position = 0
    position[0] = 0.0
    # NaN signals → flat
    position = np.where(np.isnan(lagged_composite), 0.0, position)

    # ---------------------------------------------------------------
    # 5. Daily returns: ret[t] = close[t]/close[t-1] - 1
    # ---------------------------------------------------------------
    daily_ret = np.zeros(len(close))
    daily_ret[1:] = close[1:] / close[:-1] - 1.0

    # ---------------------------------------------------------------
    # 6. Strategy returns (before costs)
    # ---------------------------------------------------------------
    gross_return = position * daily_ret

    # ---------------------------------------------------------------
    # 7. Transaction costs: turnover * cost_per_side
    # ---------------------------------------------------------------
    turnover = np.zeros(len(position))
    turnover[1:] = np.abs(position[1:] - position[:-1])
    # First day: opening a position is also a turnover event
    turnover[0] = np.abs(position[0])
    cost = turnover * cost_per_side

    # ---------------------------------------------------------------
    # 8. Net strategy returns
    # ---------------------------------------------------------------
    net_return = gross_return - cost

    # ---------------------------------------------------------------
    # 9. Statistics (all daily returns, including flat days = 0)
    # ---------------------------------------------------------------
    return _compute_stats(
        net_return=net_return,
        position=position,
        close=close,
        dates=merged["date"].values,
        symbol=symbol,
        entry_threshold=entry_threshold,
        signal_scale=signal_scale,
    )


def _compute_stats(
    net_return: np.ndarray,
    position: np.ndarray,
    close: np.ndarray,
    dates: np.ndarray,
    symbol: str,
    entry_threshold: float,
    signal_scale: float,
) -> dict:
    """Compute backtest statistics from daily net returns."""
    n = len(net_return)

    # Sharpe: mean / std(ddof=1) * sqrt(252) — all daily returns
    if n > 1:
        std = np.std(net_return, ddof=1)
        sharpe = float(np.mean(net_return) / std * np.sqrt(252)) if std > 1e-12 else 0.0
    else:
        sharpe = 0.0

    # Total return (geometric compounding)
    equity = np.cumprod(1.0 + net_return)
    total_return = float(equity[-1] - 1.0) if n > 0 else 0.0

    # Max drawdown
    if n > 0:
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(dd))
    else:
        max_dd = 0.0

    # Trade counting: a "trade" starts when we go from 0 to non-zero
    # or change direction (sign flip)
    pos_sign = np.sign(position)
    sign_changes = np.zeros(n)
    sign_changes[1:] = np.abs(pos_sign[1:] - pos_sign[:-1])
    # Count entries: going from 0→nonzero, or direction flips (sign_changes > 0 and new pos != 0)
    trades = int(np.sum((sign_changes > 0) & (pos_sign != 0)))

    # Win rate: days with positive net return when in position
    in_position = position != 0
    if in_position.any():
        win_days = np.sum((net_return > 0) & in_position)
        total_position_days = np.sum(in_position)
        win_rate = float(win_days / total_position_days) if total_position_days > 0 else 0.0
    else:
        win_rate = 0.0

    # Average holding period: total position-days / trades
    total_position_days = int(np.sum(in_position))
    avg_hold = float(total_position_days / trades) if trades > 0 else 0.0

    # Annualized return
    ann_return = float((1 + total_return) ** (252 / max(n, 1)) - 1) if n > 0 else 0.0

    return {
        "symbol": symbol,
        "entry_threshold": entry_threshold,
        "signal_scale": signal_scale,
        "trades": trades,
        "total_return_pct": total_return * 100,
        "ann_return_pct": ann_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": win_rate,
        "total_position_days": total_position_days,
        "avg_hold_days": avg_hold,
        "n_days": n,
        "daily_returns": net_return.tolist(),
    }


def _empty_result(symbol: str, entry_threshold: float, signal_scale: float) -> dict:
    """Return an empty result dict when insufficient data."""
    return {
        "symbol": symbol,
        "entry_threshold": entry_threshold,
        "signal_scale": signal_scale,
        "trades": 0,
        "total_return_pct": 0.0,
        "ann_return_pct": 0.0,
        "sharpe": 0.0,
        "max_dd_pct": 0.0,
        "win_rate": 0.0,
        "total_position_days": 0,
        "avg_hold_days": 0.0,
        "n_days": 0,
        "daily_returns": [],
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_single(
    store: MarketDataStore,
    start: date,
    end: date,
    entry_threshold: float = 0.5,
    signal_scale: float = 2.0,
) -> dict:
    """Run backtest for all symbols with a single parameter set."""
    print(f"\n{'='*70}")
    print(f"S25 Divergence Flow Field: {start} to {end}")
    print(f"  entry_threshold={entry_threshold}, signal_scale={signal_scale}")
    print("=" * 70)

    all_results = {}
    for symbol in SYMBOLS:
        t0 = time.time()
        result = backtest_dff(
            store, start, end, symbol=symbol,
            entry_threshold=entry_threshold,
            signal_scale=signal_scale,
        )
        elapsed = time.time() - t0
        all_results[symbol] = result
        print(f"\n  {symbol} ({elapsed:.1f}s):")
        print(f"    Days:       {result['n_days']}")
        print(f"    Trades:     {result['trades']}")
        print(f"    Win Rate:   {result['win_rate']*100:.1f}%")
        print(f"    Total Ret:  {result['total_return_pct']:+.2f}%")
        print(f"    Ann. Ret:   {result['ann_return_pct']:+.2f}%")
        print(f"    Sharpe:     {result['sharpe']:.2f}")
        print(f"    Max DD:     {result['max_dd_pct']:.2f}%")
        print(f"    Pos Days:   {result['total_position_days']}")
        print(f"    Avg Hold:   {result['avg_hold_days']:.1f} days")

    return all_results


def run_sweep(
    store: MarketDataStore,
    start: date,
    end: date,
) -> None:
    """Run a parameter sweep over entry_threshold and signal_scale."""
    entry_thresholds = [0.3, 0.5, 0.7]
    signal_scales = [1.5, 2.0, 3.0]

    print(f"\n{'='*70}")
    print(f"S25 Divergence Flow Field — Parameter Sweep: {start} to {end}")
    print(f"  entry_thresholds: {entry_thresholds}")
    print(f"  signal_scales:    {signal_scales}")
    print("=" * 70)

    header = (
        f"  {'Thresh':>6} {'Scale':>6} {'Symbol':>12} {'Trades':>6} "
        f"{'WinRate':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} "
        f"{'AvgHold':>8}"
    )
    print(f"\n{header}")
    print("  " + "-" * 75)

    best_sharpe = -999.0
    best_config: str | None = None

    for thresh, scale in itertools.product(entry_thresholds, signal_scales):
        for symbol in SYMBOLS:
            t0 = time.time()
            result = backtest_dff(
                store, start, end, symbol=symbol,
                entry_threshold=thresh,
                signal_scale=scale,
            )
            elapsed = time.time() - t0
            short_sym = symbol.replace("NIFTY ", "N")
            print(
                f"  {thresh:6.2f} {scale:6.2f} {symbol:>12} "
                f"{result['trades']:6d} "
                f"{result['win_rate']*100:7.1f}% "
                f"{result['total_return_pct']:+7.2f}% "
                f"{result['sharpe']:7.2f} "
                f"{result['max_dd_pct']:6.2f}% "
                f"{result['avg_hold_days']:7.1f}d"
                f"  ({elapsed:.1f}s)"
            )

            if result["sharpe"] > best_sharpe and result["trades"] >= 3:
                best_sharpe = result["sharpe"]
                best_config = (
                    f"{symbol} thresh={thresh} scale={scale} → "
                    f"Sharpe={result['sharpe']:.2f} "
                    f"Ret={result['total_return_pct']:+.2f}% "
                    f"WR={result['win_rate']*100:.0f}%"
                )

    if best_config:
        print(f"\n  BEST: {best_config}")
    print()


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------


def _backtest_dff_fold(
    store: MarketDataStore,
    fold_start: date,
    fold_end: date,
    symbol: str,
    entry_threshold: float,
    signal_scale: float,
    max_conviction: float = 0.8,
    warmup_calendar_days: int = 120,
    dff_config: DFFConfig | None = None,
) -> dict:
    """Run a vectorized backtest on a single walk-forward test fold.

    Builds DFF features starting from ``fold_start - warmup_calendar_days``
    to ensure z-score rolling windows are fully warmed up, then evaluates
    only the ``[fold_start, fold_end]`` test period.

    Parameters
    ----------
    store : MarketDataStore
    fold_start, fold_end : date
        Test period boundaries.
    symbol : str
        Index name (must be in SYMBOLS).
    entry_threshold : float
        Minimum |composite| to open a position.
    signal_scale : float
        Divisor to map composite to conviction.
    max_conviction : float
        Cap on absolute position size.
    warmup_calendar_days : int
        Calendar days before fold_start to load for feature warmup.
    dff_config : DFFConfig, optional
        Custom DFF configuration.

    Returns
    -------
    dict
        Backtest results for the test fold including daily_returns array.
    """
    cost_per_side = COST_PER_SIDE.get(symbol, 3.0 / 22000.0)

    # ---------------------------------------------------------------
    # 1. Load price data for the full range (warmup + test)
    # ---------------------------------------------------------------
    warmup_start = fold_start - timedelta(days=warmup_calendar_days)
    prices_df = _load_index_close(
        store, symbol, warmup_start.isoformat(), fold_end.isoformat()
    )
    if prices_df.empty or len(prices_df) < 30:
        return _empty_result(symbol, entry_threshold, signal_scale)

    # ---------------------------------------------------------------
    # 2. Build DFF features over warmup + test period
    #    The builder internally adds its own buffer for z-score warmup,
    #    but we pass the warmup start explicitly to maximise coverage.
    # ---------------------------------------------------------------
    builder = DivergenceFlowBuilder(config=dff_config)
    features = builder.build(
        warmup_start.isoformat(), fold_end.isoformat(), store=store
    )

    if features.empty or "dff_composite" not in features.columns:
        return _empty_result(symbol, entry_threshold, signal_scale)

    # ---------------------------------------------------------------
    # 3. Align features with prices on date
    # ---------------------------------------------------------------
    features = features.reset_index()
    features.rename(columns={features.columns[0]: "date"}, inplace=True)
    features["date"] = pd.to_datetime(features["date"])

    merged = pd.merge(
        prices_df, features[["date", "dff_composite"]], on="date", how="inner"
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    if len(merged) < 10:
        return _empty_result(symbol, entry_threshold, signal_scale)

    # ---------------------------------------------------------------
    # 4. Causal position sizing on FULL data: position[t] = f(composite[t-1])
    # ---------------------------------------------------------------
    close = merged["close"].values
    composite = merged["dff_composite"].values

    lagged_composite = np.full(len(composite), np.nan)
    lagged_composite[1:] = composite[:-1]

    abs_signal = np.abs(lagged_composite)
    conviction = np.clip(abs_signal / signal_scale, 0.0, max_conviction)
    sign = np.sign(lagged_composite)
    position = np.where(abs_signal >= entry_threshold, sign * conviction, 0.0)
    position[0] = 0.0
    position = np.where(np.isnan(lagged_composite), 0.0, position)

    # ---------------------------------------------------------------
    # 5. Daily returns
    # ---------------------------------------------------------------
    daily_ret = np.zeros(len(close))
    daily_ret[1:] = close[1:] / close[:-1] - 1.0

    # ---------------------------------------------------------------
    # 6. Strategy returns (before costs)
    # ---------------------------------------------------------------
    gross_return = position * daily_ret

    # ---------------------------------------------------------------
    # 7. Transaction costs
    # ---------------------------------------------------------------
    turnover = np.zeros(len(position))
    turnover[1:] = np.abs(position[1:] - position[:-1])
    turnover[0] = np.abs(position[0])
    cost = turnover * cost_per_side

    # ---------------------------------------------------------------
    # 8. Net strategy returns
    # ---------------------------------------------------------------
    net_return = gross_return - cost

    # ---------------------------------------------------------------
    # 9. Trim to test period [fold_start, fold_end] only
    # ---------------------------------------------------------------
    merged_dates = pd.to_datetime(merged["date"])
    fold_start_ts = pd.Timestamp(fold_start)
    fold_end_ts = pd.Timestamp(fold_end)
    test_mask = (merged_dates >= fold_start_ts) & (merged_dates <= fold_end_ts)

    test_net_return = net_return[test_mask.values]
    test_position = position[test_mask.values]
    test_close = close[test_mask.values]
    test_dates = merged_dates[test_mask].values

    if len(test_net_return) < 5:
        return _empty_result(symbol, entry_threshold, signal_scale)

    return _compute_stats(
        net_return=test_net_return,
        position=test_position,
        close=test_close,
        dates=test_dates,
        symbol=symbol,
        entry_threshold=entry_threshold,
        signal_scale=signal_scale,
    )


def _generate_walkforward_folds(
    store: MarketDataStore,
    wf_start: date,
    wf_end: date,
    train_window: int,
    test_window: int,
    step: int,
    symbol: str,
) -> list[tuple[date, date]]:
    """Generate walk-forward test fold date boundaries using trading-day calendar.

    Loads the actual trading-day calendar from nse_index_close for the given
    symbol, then generates non-overlapping test folds by stepping through
    the trading-day array.

    Parameters
    ----------
    store : MarketDataStore
    wf_start, wf_end : date
        Full walk-forward date range.
    train_window : int
        Number of trading days reserved for z-score warmup before first fold.
    test_window : int
        Number of trading days per test fold.
    step : int
        Number of trading days to advance between fold starts.
    symbol : str
        Index name used to derive the trading-day calendar.

    Returns
    -------
    list[tuple[date, date]]
        Each tuple is (fold_start_date, fold_end_date) in real calendar dates.
    """
    index_name = _INDEX_NAME.get(symbol, symbol)
    calendar_df = store.sql(
        'SELECT DISTINCT date FROM nse_index_close '
        'WHERE date BETWEEN ? AND ? AND LOWER("Index Name") = LOWER(?) '
        'ORDER BY date',
        [wf_start.isoformat(), wf_end.isoformat(), index_name],
    )
    if calendar_df is None or calendar_df.empty:
        return []

    trading_days = sorted(pd.to_datetime(calendar_df["date"]).dt.date.tolist())

    if len(trading_days) < train_window + test_window:
        return []

    folds: list[tuple[date, date]] = []
    # First test fold starts after the training window
    fold_start_idx = train_window
    while fold_start_idx + test_window <= len(trading_days):
        fold_start_dt = trading_days[fold_start_idx]
        fold_end_dt = trading_days[min(fold_start_idx + test_window - 1, len(trading_days) - 1)]
        folds.append((fold_start_dt, fold_end_dt))
        fold_start_idx += step

    return folds


def run_walkforward(
    store: MarketDataStore,
    wf_start: date = date(2024, 6, 1),
    wf_end: date = date(2026, 2, 6),
    train_window: int = 126,
    test_window: int = 63,
    step: int = 63,
    entry_threshold: float = 0.3,
    signal_scale: float = 1.5,
) -> dict:
    """Run walk-forward validation for the DFF strategy.

    Generates non-overlapping test folds, runs the vectorized backtest on
    each fold OOS, and reports per-fold + aggregate statistics.

    The "training" window is not ML training. S25 is rule-based; the
    training window only provides the rolling z-score warmup period so
    that the first day of each test fold has a fully formed signal.

    Parameters
    ----------
    store : MarketDataStore
    wf_start : date
        Start of the full walk-forward date range (default: 2024-06-01).
    wf_end : date
        End of the full walk-forward date range (default: 2026-02-06).
    train_window : int
        Trading days reserved for z-score warmup (default: 126).
    test_window : int
        Trading days per OOS test fold (default: 63).
    step : int
        Trading days to advance between fold starts (default: 63).
    entry_threshold : float
        Minimum |composite| to open a position (default: 0.3).
    signal_scale : float
        Divisor to map composite to conviction (default: 1.5).

    Returns
    -------
    dict
        Per-symbol results with fold details and aggregate OOS statistics.
    """
    print(f"\n{'='*70}")
    print("S25 Divergence Flow Field — Walk-Forward Validation")
    print(f"{'='*70}")
    print(f"  Date range:       {wf_start} to {wf_end}")
    print(f"  Train window:     {train_window} trading days (~{train_window // 21} months)")
    print(f"  Test window:      {test_window} trading days (~{test_window // 21} months)")
    print(f"  Step:             {step} trading days (non-overlapping)")
    print(f"  entry_threshold:  {entry_threshold}")
    print(f"  signal_scale:     {signal_scale}")
    print("=" * 70)

    all_results: dict[str, dict] = {}

    for symbol in SYMBOLS:
        print(f"\n{'─'*60}")
        print(f"  {symbol}")
        print(f"{'─'*60}")

        # ----------------------------------------------------------
        # Generate folds from trading-day calendar
        # ----------------------------------------------------------
        folds = _generate_walkforward_folds(
            store, wf_start, wf_end,
            train_window=train_window,
            test_window=test_window,
            step=step,
            symbol=symbol,
        )
        if not folds:
            print(f"    WARNING: Not enough data for walk-forward folds.")
            all_results[symbol] = {"folds": [], "oos_sharpe": 0.0}
            continue

        print(f"    Generated {len(folds)} test folds:")
        for i, (fs, fe) in enumerate(folds):
            print(f"      Fold {i+1}: {fs} to {fe}")

        # ----------------------------------------------------------
        # Run each fold
        # ----------------------------------------------------------
        fold_results: list[dict] = []
        all_oos_returns: list[float] = []

        header = (
            f"    {'Fold':>5} {'Start':>12} {'End':>12} {'Days':>5} "
            f"{'Trades':>6} {'WinRate':>8} {'Return':>8} "
            f"{'Sharpe':>7} {'MaxDD':>7}"
        )
        print(f"\n{header}")
        print("    " + "-" * 72)

        for fold_idx, (fold_start_dt, fold_end_dt) in enumerate(folds):
            t0 = time.time()
            result = _backtest_dff_fold(
                store,
                fold_start=fold_start_dt,
                fold_end=fold_end_dt,
                symbol=symbol,
                entry_threshold=entry_threshold,
                signal_scale=signal_scale,
                warmup_calendar_days=120,
            )
            elapsed = time.time() - t0

            result["fold_idx"] = fold_idx + 1
            result["fold_start"] = fold_start_dt.isoformat()
            result["fold_end"] = fold_end_dt.isoformat()
            fold_results.append(result)

            # Collect OOS daily returns for aggregate Sharpe
            all_oos_returns.extend(result.get("daily_returns", []))

            print(
                f"    {fold_idx + 1:5d} {fold_start_dt!s:>12} {fold_end_dt!s:>12} "
                f"{result['n_days']:5d} "
                f"{result['trades']:6d} "
                f"{result['win_rate']*100:7.1f}% "
                f"{result['total_return_pct']:+7.2f}% "
                f"{result['sharpe']:7.2f} "
                f"{result['max_dd_pct']:6.2f}%"
                f"  ({elapsed:.1f}s)"
            )

        # ----------------------------------------------------------
        # Aggregate OOS statistics
        # ----------------------------------------------------------
        oos_returns = np.array(all_oos_returns, dtype=np.float64)
        n_oos = len(oos_returns)

        if n_oos > 1:
            oos_std = float(np.std(oos_returns, ddof=1))
            oos_sharpe = (
                float(np.mean(oos_returns) / oos_std * np.sqrt(252))
                if oos_std > 1e-12
                else 0.0
            )
        else:
            oos_sharpe = 0.0

        # OOS total return (geometric compounding)
        oos_equity = np.cumprod(1.0 + oos_returns)
        oos_total_return = float(oos_equity[-1] - 1.0) if n_oos > 0 else 0.0

        # OOS max drawdown
        if n_oos > 0:
            oos_peak = np.maximum.accumulate(oos_equity)
            oos_dd = (oos_peak - oos_equity) / np.where(oos_peak > 0, oos_peak, 1.0)
            oos_max_dd = float(np.max(oos_dd))
        else:
            oos_max_dd = 0.0

        # Per-fold Sharpe statistics
        fold_sharpes = [r["sharpe"] for r in fold_results if r["n_days"] > 0]
        if len(fold_sharpes) > 1:
            sharpe_std = float(np.std(fold_sharpes, ddof=1))
            sharpe_min = float(min(fold_sharpes))
            sharpe_max = float(max(fold_sharpes))
            sharpe_mean = float(np.mean(fold_sharpes))
        elif len(fold_sharpes) == 1:
            sharpe_std = 0.0
            sharpe_min = fold_sharpes[0]
            sharpe_max = fold_sharpes[0]
            sharpe_mean = fold_sharpes[0]
        else:
            sharpe_std = 0.0
            sharpe_min = 0.0
            sharpe_max = 0.0
            sharpe_mean = 0.0

        total_trades = sum(r["trades"] for r in fold_results)

        print(f"\n    AGGREGATE OOS RESULTS ({symbol})")
        print(f"    {'─'*50}")
        print(f"      OOS Days:         {n_oos}")
        print(f"      OOS Total Return: {oos_total_return * 100:+.2f}%")
        print(f"      OOS Max DD:       {oos_max_dd * 100:.2f}%")
        print(f"      OOS Sharpe:       {oos_sharpe:.2f}")
        print(f"      Total Trades:     {total_trades}")
        print(f"    Sharpe Stability:")
        print(f"      Per-fold Mean:    {sharpe_mean:.2f}")
        print(f"      Per-fold Std:     {sharpe_std:.2f}")
        print(f"      Per-fold Min:     {sharpe_min:.2f}")
        print(f"      Per-fold Max:     {sharpe_max:.2f}")

        all_results[symbol] = {
            "folds": fold_results,
            "oos_sharpe": oos_sharpe,
            "oos_total_return_pct": oos_total_return * 100,
            "oos_max_dd_pct": oos_max_dd * 100,
            "oos_n_days": n_oos,
            "oos_total_trades": total_trades,
            "sharpe_mean": sharpe_mean,
            "sharpe_std": sharpe_std,
            "sharpe_min": sharpe_min,
            "sharpe_max": sharpe_max,
            "entry_threshold": entry_threshold,
            "signal_scale": signal_scale,
            "train_window": train_window,
            "test_window": test_window,
            "step": step,
        }

    # ----------------------------------------------------------
    # Cross-symbol summary
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    header_row = (
        f"  {'Symbol':>12} {'OOS Days':>9} {'OOS Ret':>9} "
        f"{'OOS Sharpe':>11} {'OOS MaxDD':>10} {'Trades':>7} "
        f"{'Sh.Std':>7} {'Sh.Min':>7} {'Sh.Max':>7}"
    )
    print(header_row)
    print("  " + "-" * 90)
    for symbol in SYMBOLS:
        r = all_results.get(symbol, {})
        if "oos_sharpe" not in r:
            continue
        print(
            f"  {symbol:>12} {r.get('oos_n_days', 0):9d} "
            f"{r.get('oos_total_return_pct', 0):+8.2f}% "
            f"{r['oos_sharpe']:11.2f} "
            f"{r.get('oos_max_dd_pct', 0):9.2f}% "
            f"{r.get('oos_total_trades', 0):7d} "
            f"{r.get('sharpe_std', 0):7.2f} "
            f"{r.get('sharpe_min', 0):7.2f} "
            f"{r.get('sharpe_max', 0):7.2f}"
        )
    print()

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S25 Divergence Flow Field Research")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward validation")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Entry threshold for single run (default: 0.5)",
    )
    parser.add_argument(
        "--scale", type=float, default=2.0,
        help="Signal scale for single run (default: 2.0)",
    )
    args = parser.parse_args()

    with tee_to_results("s25_divergence_flow"):
        store = MarketDataStore()

        if args.walkforward:
            run_walkforward(store)
        elif args.start and args.end:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
            if args.sweep:
                run_sweep(store, start, end)
            else:
                run_single(
                    store, start, end,
                    entry_threshold=args.threshold,
                    signal_scale=args.scale,
                )
        else:
            # Auto-detect from available data
            dates = store.available_dates("nse_index_close")
            if dates:
                start = min(dates)
                end = max(dates)
                print(f"Auto-detected date range: {start} to {end}")
            else:
                start = date(2025, 1, 1)
                end = date(2026, 2, 6)

            if args.sweep:
                run_sweep(store, start, end)
            else:
                run_single(
                    store, start, end,
                    entry_threshold=args.threshold,
                    signal_scale=args.scale,
                )

        store.close()


if __name__ == "__main__":
    main()
