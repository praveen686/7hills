"""S11: Statistical Arbitrage / Pairs Trading Research — Parameter sweep.

Uses cointegration testing from strategies/s11_pairs/strategy.py to find
mean-reverting stock pairs, then backtests spread trading.

Sweeps:
  - lookback: [40, 60, 90]
  - z_entry: [1.5, 2.0, 2.5]

Data: nse_fo_bhavcopy (STF records for stock futures).

Usage:
    python -m research.s11_pairs
    python -m research.s11_pairs --sweep
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from strategies.s9_momentum.data import is_trading_day
from data.store import MarketDataStore


COST_BPS = 10.0
Z_EXIT = 0.5
ADF_P_THRESHOLD = 0.05
HALF_LIFE_MIN = 3
HALF_LIFE_MAX = 30
HURST_MAX = 0.45
MAX_PAIRS = 3


@dataclass
class PairTrade:
    symbol_a: str
    symbol_b: str
    entry_date: date
    exit_date: date
    entry_z: float
    exit_z: float
    hedge_ratio: float
    pnl_pct: float
    exit_reason: str


def _hurst_exponent(ts: np.ndarray) -> float:
    """Estimate Hurst exponent via R/S analysis."""
    n = len(ts)
    if n < 20:
        return 0.5
    max_k = min(n // 2, 50)
    lags = range(10, max_k)
    rs_values = []
    for lag in lags:
        chunks = [ts[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))
    if len(rs_values) < 3:
        return 0.5
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    x_mean = np.mean(x)
    slope = np.sum((x - x_mean) * (y - np.mean(y))) / np.sum((x - x_mean) ** 2)
    return max(0.0, min(1.0, slope))


def _test_coint(pa: np.ndarray, pb: np.ndarray):
    """Test cointegration between two price series. Returns (beta, adf_p, half_life, hurst) or None."""
    n = len(pa)
    if n < 20:
        return None

    # OLS regression (exclude current bar)
    pa_h, pb_h = pa[:-1], pb[:-1]
    pb_mean = np.mean(pb_h)
    pa_mean = np.mean(pa_h)
    cov = np.sum((pa_h - pa_mean) * (pb_h - pb_mean))
    var = np.sum((pb_h - pb_mean) ** 2)
    if var < 1e-12:
        return None
    beta = cov / var

    spread = pa - beta * pb
    spread_hist = spread[:-1]
    spread_std = np.std(spread_hist, ddof=1)
    if spread_std < 1e-8:
        return None

    # ADF test (simplified)
    ds = np.diff(spread)
    s_lag = spread[:-1]
    if len(ds) < 10:
        return None
    phi = np.sum(ds * s_lag) / np.sum(s_lag ** 2)
    resid = ds - phi * s_lag
    se = np.sqrt(np.sum(resid ** 2) / (len(resid) - 1)) / np.sqrt(np.sum(s_lag ** 2))
    if se < 1e-12:
        return None
    t_stat = phi / se

    if t_stat < -3.51:
        p_value = 0.01
    elif t_stat < -2.89:
        p_value = 0.05
    elif t_stat < -2.58:
        p_value = 0.10
    else:
        p_value = 0.50

    if p_value > ADF_P_THRESHOLD:
        return None

    if phi >= 0:
        return None
    half_life = -math.log(2) / math.log(1 + phi)
    if half_life < HALF_LIFE_MIN or half_life > HALF_LIFE_MAX:
        return None

    hurst = _hurst_exponent(spread)
    if hurst > HURST_MAX:
        return None

    spread_mean = np.mean(spread_hist)
    current_z = (spread[-1] - spread_mean) / spread_std

    return {
        "beta": beta, "adf_p": p_value, "half_life": half_life,
        "hurst": hurst, "current_z": current_z, "spread_std": spread_std,
        "spread_mean": spread_mean,
    }


def backtest_pairs(
    store: MarketDataStore,
    start: date,
    end: date,
    lookback: int = 60,
    z_entry: float = 2.0,
    z_exit: float = Z_EXIT,
    max_pairs: int = MAX_PAIRS,
    cost_bps: float = COST_BPS,
) -> dict:
    """Backtest pairs trading strategy."""
    cost_frac = cost_bps / 10_000

    # Load all stock futures data
    d_start = (start - timedelta(days=lookback * 3)).isoformat()
    d_end = end.isoformat()

    try:
        prices_df = store.sql(
            "SELECT date, \"TckrSymb\" as symbol, \"ClsPric\" as close "
            "FROM nse_fo_bhavcopy "
            "WHERE date BETWEEN ? AND ? AND \"FinInstrmTp\" = 'STF' "
            "ORDER BY date, symbol",
            [d_start, d_end],
        )
    except Exception as e:
        print(f"  Failed to load STF data: {e}")
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0}

    if prices_df.empty:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0}

    # Pivot to get price matrix
    pivot = prices_df.pivot_table(index="date", columns="symbol", values="close")
    pivot = pivot.dropna(axis=1, thresh=lookback)
    pivot = pivot.ffill()

    if pivot.shape[1] < 2:
        return {"trades": 0, "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0}

    all_dates = sorted([date.fromisoformat(str(d)[:10]) if isinstance(d, str) else d
                        for d in pivot.index])
    symbols = list(pivot.columns)
    max_test = min(30, len(symbols))
    test_symbols = symbols[:max_test]

    trades: list[PairTrade] = []
    positions: dict[str, dict] = {}  # pair_key → position info
    daily_returns: list[float] = []

    for idx, d_val in enumerate(all_dates):
        if d_val < start:
            continue

        # Compute daily return from open positions
        day_pnl = 0.0
        for pair_key, pos in list(positions.items()):
            sym_a, sym_b = pos["symbol_a"], pos["symbol_b"]
            try:
                pa_now = float(pivot.loc[str(d_val) if isinstance(d_val, date) else d_val, sym_a])
                pb_now = float(pivot.loc[str(d_val) if isinstance(d_val, date) else d_val, sym_b])
                pa_prev = pos.get("last_price_a", pos["entry_price_a"])
                pb_prev = pos.get("last_price_b", pos["entry_price_b"])

                # P&L from spread change
                spread_now = pa_now - pos["hedge_ratio"] * pb_now
                spread_prev = pa_prev - pos["hedge_ratio"] * pb_prev

                if pos["direction"] == "short_spread":
                    day_pnl -= (spread_now - spread_prev) / pos["entry_price_a"]
                else:
                    day_pnl += (spread_now - spread_prev) / pos["entry_price_a"]

                pos["last_price_a"] = pa_now
                pos["last_price_b"] = pb_now

                # Check exit: z-score reverted
                z = (spread_now - pos["spread_mean"]) / pos["spread_std"]
                if abs(z) < z_exit:
                    pnl_a = (pa_now / pos["entry_price_a"] - 1)
                    pnl_b = (pb_now / pos["entry_price_b"] - 1)
                    if pos["direction"] == "short_spread":
                        total_pnl = (-pnl_a + pnl_b) / 2
                    else:
                        total_pnl = (pnl_a - pnl_b) / 2
                    total_pnl -= 2 * cost_frac  # roundtrip both legs
                    trades.append(PairTrade(
                        symbol_a=sym_a, symbol_b=sym_b,
                        entry_date=pos["entry_date"], exit_date=d_val,
                        entry_z=pos["entry_z"], exit_z=z,
                        hedge_ratio=pos["hedge_ratio"],
                        pnl_pct=total_pnl * 100, exit_reason="mean_reverted",
                    ))
                    del positions[pair_key]
            except (KeyError, IndexError):
                continue

        daily_returns.append(day_pnl)

        # Only scan for new pairs weekly (Monday)
        if d_val.weekday() != 0:
            continue
        if len(positions) >= max_pairs:
            continue

        # Find cointegrated pairs
        window_start = max(0, idx - lookback)
        for i in range(len(test_symbols)):
            if len(positions) >= max_pairs:
                break
            for j in range(i + 1, len(test_symbols)):
                if len(positions) >= max_pairs:
                    break
                sym_a, sym_b = test_symbols[i], test_symbols[j]
                pair_key = f"{sym_a}:{sym_b}"
                if pair_key in positions:
                    continue

                try:
                    pa = pivot[sym_a].iloc[window_start:idx + 1].dropna().values
                    pb = pivot[sym_b].iloc[window_start:idx + 1].dropna().values
                except (KeyError, IndexError):
                    continue

                n = min(len(pa), len(pb))
                if n < lookback:
                    continue
                pa, pb = pa[-n:], pb[-n:]

                result = _test_coint(pa, pb)
                if result is None:
                    continue

                if abs(result["current_z"]) < z_entry:
                    continue

                # Enter pair trade
                try:
                    price_a = float(pivot.loc[str(d_val) if isinstance(d_val, date) else d_val, sym_a])
                    price_b = float(pivot.loc[str(d_val) if isinstance(d_val, date) else d_val, sym_b])
                except (KeyError, IndexError):
                    continue

                direction = "short_spread" if result["current_z"] > z_entry else "long_spread"
                positions[pair_key] = {
                    "symbol_a": sym_a, "symbol_b": sym_b,
                    "hedge_ratio": result["beta"],
                    "entry_date": d_val, "entry_z": result["current_z"],
                    "entry_price_a": price_a, "entry_price_b": price_b,
                    "last_price_a": price_a, "last_price_b": price_b,
                    "spread_mean": result["spread_mean"],
                    "spread_std": result["spread_std"],
                    "direction": direction,
                }
                daily_returns[-1] -= 2 * cost_frac  # entry cost both legs

    # Close remaining positions at end
    for pair_key, pos in positions.items():
        try:
            last_row = pivot.iloc[-1]
            pa_now = float(last_row[pos["symbol_a"]])
            pb_now = float(last_row[pos["symbol_b"]])
            pnl_a = (pa_now / pos["entry_price_a"] - 1)
            pnl_b = (pb_now / pos["entry_price_b"] - 1)
            if pos["direction"] == "short_spread":
                total_pnl = (-pnl_a + pnl_b) / 2
            else:
                total_pnl = (pnl_a - pnl_b) / 2
            total_pnl -= 2 * cost_frac
            trades.append(PairTrade(
                symbol_a=pos["symbol_a"], symbol_b=pos["symbol_b"],
                entry_date=pos["entry_date"], exit_date=end,
                entry_z=pos["entry_z"], exit_z=0.0,
                hedge_ratio=pos["hedge_ratio"],
                pnl_pct=total_pnl * 100, exit_reason="end_of_backtest",
            ))
        except (KeyError, IndexError):
            continue

    # Compute metrics
    if not daily_returns or all(r == 0 for r in daily_returns):
        return {
            "lookback": lookback, "z_entry": z_entry,
            "trades": len(trades), "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0,
            "wins": 0, "win_rate": 0, "pairs_found": len(set(f"{t.symbol_a}:{t.symbol_b}" for t in trades)),
            "daily_returns": [],
        }

    rets = np.array(daily_returns)
    equity = np.cumprod(1 + rets)
    total_ret = (equity[-1] - 1) * 100
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)) if np.std(rets, ddof=1) > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd)) * 100

    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / len(trades) if trades else 0
    pairs_found = len(set(f"{t.symbol_a}:{t.symbol_b}" for t in trades))

    return {
        "lookback": lookback, "z_entry": z_entry,
        "trades": len(trades), "wins": wins, "win_rate": win_rate,
        "total_return_pct": total_ret, "sharpe": sharpe, "max_dd_pct": max_dd,
        "pairs_found": pairs_found, "trade_details": trades,
        "daily_returns": daily_returns,
    }


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    dates = store.available_dates("nse_fo_bhavcopy")
    return min(dates), max(dates)


def run_single(store: MarketDataStore, start: date, end: date,
               lookback: int = 60, z_entry: float = 2.0) -> dict:
    print(f"\n{'='*70}")
    print(f"S11 Pairs Trading: {start} to {end}")
    print(f"  lookback={lookback}, z_entry={z_entry}")
    print("=" * 70)

    t0 = time.time()
    result = backtest_pairs(store, start, end, lookback=lookback, z_entry=z_entry)
    elapsed = time.time() - t0
    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Trades:      {result['trades']}")
    print(f"    Pairs found: {result.get('pairs_found', 0)}")
    print(f"    Win Rate:    {result['win_rate']*100:.1f}%")
    print(f"    Total Ret:   {result['total_return_pct']:+.2f}%")
    print(f"    Sharpe:      {result['sharpe']:.2f}")
    print(f"    Max DD:      {result['max_dd_pct']:.2f}%")
    return result


def run_sweep(store: MarketDataStore, start: date, end: date) -> None:
    lookbacks = [40, 60, 90]
    z_entries = [1.5, 2.0, 2.5]

    print(f"\n{'='*70}")
    print(f"S11 Pairs Trading — Parameter Sweep: {start} to {end}")
    print(f"  lookbacks: {lookbacks}")
    print(f"  z_entries: {z_entries}")
    print("=" * 70)

    header = (f"  {'LB':>4} {'Z_ent':>5} {'Trades':>6} {'Pairs':>5} "
              f"{'WinRate':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"\n{header}")
    print("  " + "-" * 60)

    best_sharpe = -999
    best_config = None

    for lb, ze in itertools.product(lookbacks, z_entries):
        result = backtest_pairs(store, start, end, lookback=lb, z_entry=ze)
        print(f"  {lb:4d} {ze:5.1f} {result['trades']:6d} {result.get('pairs_found', 0):5d} "
              f"{result['win_rate']*100:7.1f}% {result['total_return_pct']:+7.2f}% "
              f"{result['sharpe']:7.2f} {result['max_dd_pct']:6.2f}%")

        if result['sharpe'] > best_sharpe and result['trades'] > 2:
            best_sharpe = result['sharpe']
            best_config = (lb, ze, result)

    if best_config:
        lb, ze, r = best_config
        print(f"\n  BEST: lb={lb} z_entry={ze} → "
              f"Sharpe={r['sharpe']:.2f} Ret={r['total_return_pct']:+.2f}% "
              f"WR={r['win_rate']*100:.0f}%")


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S11 Pairs Trading Research")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--z-entry", type=float, default=2.0)
    args = parser.parse_args()

    with tee_to_results("s11_pairs"):
        store = MarketDataStore()

        if args.start and args.end:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
        else:
            start, end = _get_date_range(store)
            print(f"Auto-detected date range: {start} to {end}")

        if args.sweep:
            run_sweep(store, start, end)
        else:
            run_single(store, start, end, lookback=args.lookback, z_entry=args.z_entry)

        store.close()


if __name__ == "__main__":
    main()
