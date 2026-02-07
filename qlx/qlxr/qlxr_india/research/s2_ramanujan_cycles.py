"""S2: Ramanujan Cycle Research — Causal periodogram + phase-based backtest.

All operations are strictly causal (no look-ahead):
  - Trailing-only detrending (left-aligned rolling mean)
  - Expanding-window periodogram (bars [0..j] only)
  - Causal linear convolution (np.convolve 'full' truncated to causal part)
  - Causal phase estimation via quadrature Ramanujan filters (sin/cos pair)
  - Sharpe computed from ALL daily returns (including zero-trade days)

Usage:
    python -m apps.india_fno.research.s2_ramanujan_cycles
    python -m apps.india_fno.research.s2_ramanujan_cycles --start 2025-09-01
    python -m apps.india_fno.research.s2_ramanujan_cycles --symbol BANKNIFTY
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import date

import numpy as np
import pandas as pd

from qlx.data.store import MarketDataStore
from qlx.features.ramanujan import (
    dominant_periods,
    ramanujan_periodogram,
    ramanujan_sum,
)


def _causal_detrend(close: np.ndarray, window: int) -> np.ndarray:
    """Trailing-only detrend using left-aligned rolling mean."""
    trend = pd.Series(close).rolling(window, min_periods=1).mean().values
    return close - trend


def _causal_filter(signal: np.ndarray, q: int) -> np.ndarray:
    """Causal Ramanujan filter via linear convolution.

    Uses np.convolve in 'full' mode, then takes only the causal part
    (output[j] depends only on signal[0..j]).
    """
    filt = np.array([ramanujan_sum(q, n) for n in range(q)], dtype=np.float64) / q
    # 'full' convolution, take first len(signal) elements = causal output
    conv = np.convolve(signal, filt, mode='full')[:len(signal)]
    return conv


def _causal_phase(signal: np.ndarray, period: int) -> np.ndarray:
    """Causal phase estimation using quadrature Ramanujan filters.

    Builds a cosine-like and sine-like filter pair at the target period,
    applies both causally, then computes phase = arctan2(sine_out, cosine_out).
    No FFT, no Hilbert — strictly causal.
    """
    n = len(signal)
    length = period * 2  # filter length = 2 full cycles

    # Build quadrature filter pair
    cos_filt = np.array([math.cos(2 * math.pi * k / period) for k in range(length)])
    sin_filt = np.array([math.sin(2 * math.pi * k / period) for k in range(length)])

    # Apply Hann window to reduce spectral leakage
    hann = np.hanning(length)
    cos_filt *= hann
    sin_filt *= hann

    # Causal convolution (output[j] uses only past data)
    cos_out = np.convolve(signal, cos_filt[::-1], mode='full')[:n]
    sin_out = np.convolve(signal, sin_filt[::-1], mode='full')[:n]

    phase = np.arctan2(sin_out, cos_out)
    return phase


def _expanding_periodogram(
    detrended: np.ndarray,
    j: int,
    max_period: int,
    min_window: int = 64,
) -> np.ndarray:
    """Compute periodogram using only bars [max(0, j-min_window)..j] (causal).

    Uses a trailing window of at least min_window bars ending at bar j.
    """
    start = max(0, j - min_window + 1)
    segment = detrended[start:j + 1]
    if len(segment) < max_period:
        return np.zeros(max_period)
    return ramanujan_periodogram(segment, max_period)


def _load_daily_bars(store: MarketDataStore, d: date, symbol: str) -> pd.DataFrame:
    """Load 1-min bars for a single day and underlying name."""
    try:
        df = store.sql(
            "SELECT * FROM nfo_1min WHERE date = ? AND name = ? "
            "AND instrument_type = 'FUT'",
            [d.isoformat(), symbol],
        )
        if df is not None and not df.empty:
            if "expiry" in df.columns:
                df["_exp"] = pd.to_datetime(df["expiry"], format="mixed", errors="coerce")
                min_exp = df["_exp"].min()
                df = df[df["_exp"] == min_exp].drop(columns=["_exp"])
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            return df

        df = store.sql(
            "SELECT * FROM nfo_1min WHERE date = ? AND name = ? LIMIT 500",
            [d.isoformat(), symbol],
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df


def run_research(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    max_period: int = 64,
    top_k: int = 3,
) -> None:
    """Main research loop — all operations strictly causal."""
    dates = store.available_dates("nfo_1min")
    dates = [d for d in dates if start <= d <= end]
    dates.sort()

    print(f"\nS2 Ramanujan Cycles Research (CAUSAL): {start} to {end}")
    print(f"  Symbol: {symbol}, max_period={max_period}, top_k={top_k}")
    print(f"  {len(dates)} available dates\n")

    daily_results: list[dict] = []
    t0 = time.time()

    for i, d in enumerate(dates):
        bars = _load_daily_bars(store, d, symbol)
        if bars.empty or len(bars) < max_period * 2:
            continue

        close = bars["close"].values.astype(np.float64)

        # 1. CAUSAL detrend: trailing-only rolling mean
        window = min(20, len(close) // 5)
        window = max(window, 2)
        detrended = _causal_detrend(close, window)

        # 2. CAUSAL periodogram: use trailing window for period detection
        #    at each bar j, only bars [0..j] contribute
        #    For efficiency, compute at end of first half and at end of day
        mid = len(detrended) // 2
        if mid < max_period:
            mid = max_period

        # Detect dominant period from first half of day (used for second half trades)
        energies_mid = _expanding_periodogram(detrended, mid, max_period, min_window=max_period)
        energies_mid[0] = 0  # skip DC
        dom_indices = np.argsort(energies_mid)[::-1][:top_k]
        dom_periods_causal = [int(idx + 1) for idx in dom_indices if energies_mid[idx] > 0]

        # Also compute end-of-day periodogram for reporting
        energies_end = _expanding_periodogram(detrended, len(detrended) - 1, max_period, min_window=max_period)
        energies_end[0] = 0

        if not dom_periods_causal:
            daily_results.append({
                "date": d, "n_bars": len(close),
                "dom_period_1": 0, "dom_period_2": 0, "dom_period_3": 0,
                "energy_1": 0.0, "day_return": float((close[-1] - close[0]) / close[0]),
                "n_trades": 0, "wins": 0, "day_pnl": 0.0,
            })
            continue

        primary = dom_periods_causal[0]

        # 3. CAUSAL filter: linear convolution (no circular FFT)
        comp = _causal_filter(detrended, primary)

        # 4. CAUSAL phase: quadrature filter pair (no Hilbert)
        phase = _causal_phase(detrended, primary)

        # Backtest: only trade in second half of day (period detected from first half)
        # This ensures the period detection does not use future data for the traded bars
        trade_start = mid + primary * 2  # allow filter warmup
        position = 0
        entry_price = 0.0
        day_pnl = 0.0
        n_trades = 0
        wins = 0

        for j in range(max(trade_start, 1), len(close)):
            if position == 0 and phase[j - 1] < -1.5 and phase[j] >= -1.5:
                position = 1
                entry_price = close[j]
            elif position == 1 and phase[j - 1] < 1.0 and phase[j] >= 1.0:
                pnl = (close[j] - entry_price) / entry_price
                day_pnl += pnl
                n_trades += 1
                if pnl > 0:
                    wins += 1
                position = 0

        # Close open position at end of day
        if position == 1:
            pnl = (close[-1] - entry_price) / entry_price
            day_pnl += pnl
            n_trades += 1
            if pnl > 0:
                wins += 1

        daily_results.append({
            "date": d,
            "n_bars": len(close),
            "dom_period_1": dom_periods_causal[0] if len(dom_periods_causal) > 0 else 0,
            "dom_period_2": dom_periods_causal[1] if len(dom_periods_causal) > 1 else 0,
            "dom_period_3": dom_periods_causal[2] if len(dom_periods_causal) > 2 else 0,
            "energy_1": float(energies_end[dom_periods_causal[0] - 1]) if dom_periods_causal else 0,
            "day_return": float((close[-1] - close[0]) / close[0]),
            "n_trades": n_trades,
            "wins": wins,
            "day_pnl": day_pnl,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(dates)} dates ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Processed {len(daily_results)} days in {elapsed:.1f}s\n")

    if not daily_results:
        print("  No valid data found.")
        return

    df = pd.DataFrame(daily_results)
    _print_results(df, symbol)


def _print_results(df: pd.DataFrame, symbol: str) -> None:
    print("=" * 70)
    print(f"S2 RAMANUJAN CYCLES — {symbol} (CAUSAL)")
    print("=" * 70)

    # Dominant period distribution
    print("\n  Dominant Period Distribution:")
    p1_counts = df["dom_period_1"].value_counts().head(10)
    for period, cnt in p1_counts.items():
        pct = cnt / len(df) * 100
        print(f"    Period {period:3d}: {cnt:4d} days ({pct:.1f}%)")

    # Period stability
    p1_series = df["dom_period_1"].values
    if len(p1_series) > 1:
        period_changes = np.sum(p1_series[1:] != p1_series[:-1])
        stability = 1 - period_changes / (len(p1_series) - 1)
        print(f"\n  Period stability: {stability:.1%} (1=always same, 0=changes every day)")

    # Phase backtest results
    total_trades = df["n_trades"].sum()
    total_wins = df["wins"].sum()
    total_pnl = df["day_pnl"].sum()
    n_days = len(df)

    print(f"\n  Phase Backtest (causal — trades only in 2nd half of day):")
    print(f"    Total trades:     {total_trades}")
    if total_trades > 0:
        print(f"    Win rate:         {total_wins/total_trades:.1%}")
    print(f"    Total P&L:        {total_pnl*100:+.3f}%")
    if n_days > 0:
        print(f"    Avg daily P&L:    {total_pnl/n_days*100:+.4f}%")

    # Sharpe from ALL daily returns (including zero-trade days)
    if n_days > 1:
        all_daily_pnl = df["day_pnl"].values
        std = np.std(all_daily_pnl, ddof=1)
        if std > 0:
            sharpe = float(np.mean(all_daily_pnl) / std * np.sqrt(252))
        else:
            sharpe = 0.0
        print(f"    Sharpe (ann.):    {sharpe:.2f}")

        if total_trades > 0:
            avg_per_trade = total_pnl / total_trades * 100
            print(f"    Avg per cycle:    {avg_per_trade:+.4f}%")

    # Correlation: dominant period energy vs day return magnitude
    if len(df) > 10:
        from scipy.stats import spearmanr
        energy = df["energy_1"].values
        abs_ret = np.abs(df["day_return"].values)
        corr, pval = spearmanr(energy, abs_ret)
        print(f"\n  Energy vs |return| correlation: {corr:.3f} (p={pval:.4f})")

    print()


def main() -> None:
    from research.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S2 Ramanujan Cycles Research")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol (default: NIFTY)")
    parser.add_argument("--max-period", type=int, default=64,
                        help="Max period to search (default: 64)")
    args = parser.parse_args()

    with tee_to_results("s2_ramanujan_cycles"):
        store = MarketDataStore()

        if args.start:
            start = date.fromisoformat(args.start)
        else:
            dates = store.available_dates("nfo_1min")
            start = min(dates) if dates else date(2025, 8, 1)

        end = date.fromisoformat(args.end) if args.end else date.today()
        print(f"Date range: {start} to {end}")

        run_research(store, start, end, symbol=args.symbol, max_period=args.max_period)
        store.close()


if __name__ == "__main__":
    main()
