"""S8: Expiry-Day Theta Harvest Research — Parameter sweep on iron condor 0DTE.

Sweeps:
  - short_otm_pct: [0.010, 0.015, 0.020]
  - max_vix: [15, 18, 21]
  - symbols: NIFTY, BANKNIFTY

Uses the existing backtest_expiry_theta() from strategies/s8_expiry_theta/strategy.py.

Usage:
    python -m research.s8_expiry_theta
    python -m research.s8_expiry_theta --sweep
    python -m research.s8_expiry_theta --start 2025-09-01 --end 2026-01-31
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import date

from strategies.s8_expiry_theta.strategy import backtest_expiry_theta
from core.market.store import MarketDataStore


SYMBOLS = ["NIFTY", "BANKNIFTY"]


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    dates = store.available_dates("nfo_1min")
    return min(dates), max(dates)


def run_single(
    store: MarketDataStore,
    start: date,
    end: date,
    short_otm_pct: float = 0.015,
    max_vix: float = 18.0,
) -> dict:
    """Run S8 backtest for both symbols with given parameters."""
    print(f"\n{'='*70}")
    print(f"S8 Expiry-Day Theta Harvest: {start} to {end}")
    print(f"  short_otm_pct={short_otm_pct}, max_vix={max_vix}")
    print("=" * 70)

    all_results = {}
    for symbol in SYMBOLS:
        t0 = time.time()
        result = backtest_expiry_theta(
            store, start, end,
            symbol=symbol,
            short_otm_pct=short_otm_pct,
            max_vix=max_vix,
        )
        elapsed = time.time() - t0
        all_results[symbol] = result
        print(f"\n  {symbol} ({elapsed:.1f}s):")
        print(f"    Trades:      {result['trades']}")
        print(f"    Wins:        {result['wins']}  ({result['win_rate']*100:.1f}%)")
        print(f"    Total Ret:   {result['total_return_pct']:+.2f}%")
        print(f"    Sharpe:      {result['sharpe']:.2f}")
        print(f"    Max DD:      {result['max_dd_pct']:.2f}%")
        print(f"    Avg Credit:  {result['avg_credit_pct']:.2f}%")

    return all_results


def run_sweep(store: MarketDataStore, start: date, end: date) -> None:
    """Parameter sweep across short_otm_pct and max_vix."""
    otm_pcts = [0.010, 0.015, 0.020]
    max_vix_list = [15, 18, 21]

    print(f"\n{'='*70}")
    print(f"S8 Expiry-Day Theta — Parameter Sweep: {start} to {end}")
    print(f"  short_otm_pct: {otm_pcts}")
    print(f"  max_vix: {max_vix_list}")
    print("=" * 70)

    header = (f"  {'OTM%':>6} {'MaxVIX':>6} {'Symbol':>10} {'Trades':>6} "
              f"{'WinRate':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'AvgCred':>8}")
    print(f"\n{header}")
    print("  " + "-" * 80)

    best_sharpe = -999
    best_config = None

    for otm, vix in itertools.product(otm_pcts, max_vix_list):
        for symbol in SYMBOLS:
            result = backtest_expiry_theta(
                store, start, end,
                symbol=symbol,
                short_otm_pct=otm,
                max_vix=vix,
            )
            print(f"  {otm:6.3f} {vix:6.0f} {symbol:>10} {result['trades']:6d} "
                  f"{result['win_rate']*100:7.1f}% {result['total_return_pct']:+7.2f}% "
                  f"{result['sharpe']:7.2f} {result['max_dd_pct']:6.2f}% "
                  f"{result['avg_credit_pct']:7.2f}%")

            if result['sharpe'] > best_sharpe and result['trades'] > 3:
                best_sharpe = result['sharpe']
                best_config = (otm, vix, symbol, result)

    if best_config:
        otm, vix, sym, r = best_config
        print(f"\n  BEST: {sym} otm={otm:.3f} vix={vix} → "
              f"Sharpe={r['sharpe']:.2f} Ret={r['total_return_pct']:+.2f}% "
              f"WR={r['win_rate']*100:.0f}%")


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S8 Expiry Theta Research")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--otm", type=float, default=0.015)
    parser.add_argument("--vix", type=float, default=18.0)
    args = parser.parse_args()

    with tee_to_results("s8_expiry_theta"):
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
            run_single(store, start, end, short_otm_pct=args.otm, max_vix=args.vix)

        store.close()


if __name__ == "__main__":
    main()
