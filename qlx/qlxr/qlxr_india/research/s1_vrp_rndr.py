"""S1: VRP-RNDR Research — Density-based backtests on real DuckDB data.

Runs both variants:
  1. Futures: run_multi_index_density_backtest (directional futures from density signals)
  2. Options: run_multi_index_options_backtest (bull put spreads from density signals)

Parameter sweep over entry_pctile and hold_days.

Usage:
    python -m apps.india_fno.research.s1_vrp_rndr
    python -m apps.india_fno.research.s1_vrp_rndr --start 2025-09-01 --end 2026-01-31
    python -m apps.india_fno.research.s1_vrp_rndr --sweep
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import date

from strategies.s1_vrp.options import run_multi_index_options_backtest
from strategies.s1_vrp.density import run_multi_index_density_backtest
from qlx.data.store import MarketDataStore


def _format_density_results(results: dict) -> str:
    """Format DensityBacktestResult dict."""
    lines = []
    header = (f"  {'Index':<14} {'Sharpe':>7} {'Return':>8} {'Ann.Ret':>8} "
              f"{'MaxDD':>7} {'WinRate':>8} {'Signals':>7}")
    lines.append(header)
    lines.append("  " + "-" * 70)
    for sym, r in sorted(results.items()):
        lines.append(
            f"  {sym:<14} {r.sharpe:7.2f} {r.total_return_pct:+7.2f}% "
            f"{r.annual_return_pct:+7.2f}% {r.max_dd_pct:6.2f}% "
            f"{r.win_rate*100:7.1f}% {r.n_signals:7d}"
        )
    return "\n".join(lines)


def _format_options_results(results: dict) -> str:
    """Format SpreadBacktestResult dict."""
    lines = []
    header = (f"  {'Index':<14} {'Sharpe':>7} {'RetOnRisk':>10} "
              f"{'MaxDD':>7} {'WinRate':>8} {'Signals':>7} {'AvgCred':>8}")
    lines.append(header)
    lines.append("  " + "-" * 70)
    for sym, r in sorted(results.items()):
        lines.append(
            f"  {sym:<14} {r.sharpe:7.2f} {r.total_return_on_risk_pct:+9.2f}% "
            f"{r.max_dd_pct:6.2f}% {r.win_rate*100:7.1f}% "
            f"{r.n_signals:7d} {r.avg_credit_pts:7.2f}pts"
        )
    return "\n".join(lines)


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    """Get full date range from nse_fo_bhavcopy."""
    dates = store.available_dates("nse_fo_bhavcopy")
    return min(dates), max(dates)


def run_single(
    store: MarketDataStore,
    start: date,
    end: date,
    entry_pctile: float = 0.80,
    hold_days: int = 5,
) -> None:
    """Run both variants with given parameters."""
    print(f"\n{'='*70}")
    print(f"S1 VRP-RNDR: {start} to {end}")
    print(f"  entry_pctile={entry_pctile}, hold_days={hold_days}")
    print("=" * 70)

    # Futures variant
    print("\n--- Futures Variant (directional) ---")
    t0 = time.time()
    futures_results = run_multi_index_density_backtest(
        store, start, end,
        entry_pctile=entry_pctile,
        hold_days=hold_days,
    )
    print(f"  ({time.time()-t0:.1f}s)")
    print(_format_density_results(futures_results))

    # Options variant (bull put spreads)
    print("\n--- Options Variant (bull put spreads) ---")
    t0 = time.time()
    options_results = run_multi_index_options_backtest(
        store, start, end,
        entry_pctile=entry_pctile,
        hold_days=hold_days,
    )
    print(f"  ({time.time()-t0:.1f}s)")
    print(_format_options_results(options_results))


def run_sweep(store: MarketDataStore, start: date, end: date) -> None:
    """Parameter sweep across entry_pctile and hold_days."""
    entry_pctiles = [0.70, 0.75, 0.80]
    hold_days_list = [3, 5, 7, 10]

    print(f"\n{'='*70}")
    print(f"S1 VRP-RNDR Parameter Sweep: {start} to {end}")
    print(f"  entry_pctiles: {entry_pctiles}")
    print(f"  hold_days: {hold_days_list}")
    print("=" * 70)

    # Futures sweep
    print("\n\n=== FUTURES VARIANT ===\n")
    print(f"  {'Entry%':>7} {'Hold':>5} | {'Index':<12} {'Sharpe':>7} {'Return':>8} "
          f"{'MaxDD':>7} {'WinRate':>8} {'Signals':>7}")
    print("  " + "-" * 75)

    for entry_p, hold_d in itertools.product(entry_pctiles, hold_days_list):
        results = run_multi_index_density_backtest(
            store, start, end,
            entry_pctile=entry_p,
            hold_days=hold_d,
        )
        for sym, r in sorted(results.items()):
            print(f"  {entry_p:7.2f} {hold_d:5d} | {sym:<12} {r.sharpe:7.2f} "
                  f"{r.total_return_pct:+7.2f}% {r.max_dd_pct:6.2f}% "
                  f"{r.win_rate*100:7.1f}% {r.n_signals:7d}")

    # Options sweep
    print("\n\n=== OPTIONS VARIANT (Bull Put Spreads) ===\n")
    print(f"  {'Entry%':>7} {'Hold':>5} | {'Index':<12} {'Sharpe':>7} {'RetOnRisk':>10} "
          f"{'MaxDD':>7} {'WinRate':>8} {'Signals':>7}")
    print("  " + "-" * 75)

    for entry_p, hold_d in itertools.product(entry_pctiles, hold_days_list):
        results = run_multi_index_options_backtest(
            store, start, end,
            entry_pctile=entry_p,
            hold_days=hold_d,
        )
        for sym, r in sorted(results.items()):
            print(f"  {entry_p:7.2f} {hold_d:5d} | {sym:<12} {r.sharpe:7.2f} "
                  f"{r.total_return_on_risk_pct:+9.2f}% {r.max_dd_pct:6.2f}% "
                  f"{r.win_rate*100:7.1f}% {r.n_signals:7d}")


def main() -> None:
    from research.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S1 VRP-RNDR Research")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep")
    parser.add_argument("--entry", type=float, default=0.80,
                        help="Entry percentile (default: 0.80)")
    parser.add_argument("--hold", type=int, default=5,
                        help="Hold days (default: 5)")
    args = parser.parse_args()

    with tee_to_results("s1_vrp_rndr"):
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
            run_single(store, start, end, entry_pctile=args.entry, hold_days=args.hold)

        store.close()


if __name__ == "__main__":
    main()
