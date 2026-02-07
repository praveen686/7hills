"""S4: IV Mean-Reversion Research — Multi-index backtest on real DuckDB data.

Optimizations over naive approach:
  - IV series cached per index (SANOS LP calibration is expensive, only run once)
  - Parallel IV series building across indices via ProcessPoolExecutor
  - Parameter sweep runs instantly on cached series (no re-calibration)

Usage:
    python -m apps.india_fno.research.s4_iv_mean_revert
    python -m apps.india_fno.research.s4_iv_mean_revert --start 2025-09-01 --end 2026-01-31
    python -m apps.india_fno.research.s4_iv_mean_revert --sweep
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

from strategies.s4_iv_mr.engine import (
    BacktestResult,
    DayObs,
    SUPPORTED_INDICES,
    build_iv_series,
    format_multi_index_results,
    run_from_series,
)
from core.market.store import MarketDataStore


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    dates = store.available_dates("nse_fo_bhavcopy")
    return min(dates), max(dates)


# ---------------------------------------------------------------------------
# Worker: build IV series for one index (runs in subprocess)
# ---------------------------------------------------------------------------

def _build_iv_series_worker(
    symbol: str,
    start_iso: str,
    end_iso: str,
) -> tuple[str, list[dict]]:
    """Build IV series for one index in a worker process.

    Returns (symbol, serialized_day_obs_list) — DayObs serialized as dicts
    because dataclasses may not pickle cleanly across process boundaries.
    """
    store = MarketDataStore()
    try:
        start = date.fromisoformat(start_iso)
        end = date.fromisoformat(end_iso)
        daily = build_iv_series(store, start, end, symbol=symbol)
        # Serialize DayObs to dicts for cross-process transfer
        return symbol, [
            {
                "date": d.date.isoformat(),
                "spot": d.spot,
                "atm_iv": d.atm_iv,
                "atm_var": d.atm_var,
                "forward": d.forward,
                "sanos_ok": d.sanos_ok,
                "symbol": d.symbol,
            }
            for d in daily
        ]
    finally:
        store.close()


def _deserialize_daily(records: list[dict]) -> list[DayObs]:
    """Convert serialized dicts back to DayObs."""
    return [
        DayObs(
            date=date.fromisoformat(r["date"]),
            spot=r["spot"],
            atm_iv=r["atm_iv"],
            atm_var=r["atm_var"],
            forward=r["forward"],
            sanos_ok=r["sanos_ok"],
            symbol=r["symbol"],
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Parallel IV series builder
# ---------------------------------------------------------------------------

def build_all_iv_series(
    start: date,
    end: date,
    symbols: list[str] | None = None,
    max_workers: int = 5,
) -> dict[str, list[DayObs]]:
    """Build IV series for all indices in parallel.

    Each index runs in a separate process with its own DuckDB connection.
    """
    if symbols is None:
        symbols = SUPPORTED_INDICES

    print(f"\n  Building IV series for {len(symbols)} indices "
          f"({max_workers} workers)...", flush=True)

    iv_cache: dict[str, list[DayObs]] = {}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _build_iv_series_worker,
                sym,
                start.isoformat(),
                end.isoformat(),
            ): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                symbol, records = future.result()
                daily = _deserialize_daily(records)
                iv_cache[symbol] = daily
                elapsed = time.time() - t0
                print(f"    {symbol:<14} {len(daily):4d} days, "
                      f"{sum(1 for d in daily if d.sanos_ok)} SANOS OK "
                      f"({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"    {sym:<14} FAILED: {e}", flush=True)

    total = time.time() - t0
    print(f"  All IV series built in {total:.1f}s\n", flush=True)
    return iv_cache


# ---------------------------------------------------------------------------
# Single run (uses cache)
# ---------------------------------------------------------------------------

def run_single(
    start: date,
    end: date,
    iv_lookback: int = 30,
    entry_pctile: float = 0.80,
) -> None:
    """Run backtest with given parameters."""
    print(f"\n{'='*70}")
    print(f"S4 IV Mean-Reversion: {start} to {end}")
    print(f"  iv_lookback={iv_lookback}, entry_pctile={entry_pctile}")
    print("=" * 70, flush=True)

    t0 = time.time()

    # Build IV series in parallel
    iv_cache = build_all_iv_series(start, end)

    # Run backtests on cached series
    results: dict[str, BacktestResult] = {}
    for sym, daily in sorted(iv_cache.items()):
        result = run_from_series(
            daily,
            iv_lookback=iv_lookback,
            entry_pctile=entry_pctile,
            symbol=sym,
        )
        results[sym] = result

    print(f"  ({time.time()-t0:.1f}s)")
    print(format_multi_index_results(results), flush=True)


# ---------------------------------------------------------------------------
# Parameter sweep (builds IV series ONCE, sweeps params instantly)
# ---------------------------------------------------------------------------

def run_sweep(start: date, end: date) -> None:
    """Parameter sweep — build IV series once, then sweep params instantly."""
    lookbacks = [30, 60, 90]
    entry_pctiles = [0.75, 0.80, 0.85]

    print(f"\n{'='*70}")
    print(f"S4 IV Mean-Reversion Parameter Sweep: {start} to {end}")
    print(f"  lookbacks: {lookbacks}")
    print(f"  entry_pctiles: {entry_pctiles}")
    print("=" * 70, flush=True)

    # Build IV series ONCE for all indices (the expensive part)
    t0 = time.time()
    iv_cache = build_all_iv_series(start, end)
    build_time = time.time() - t0

    # Sweep params on cached series (instant — no SANOS recalibration)
    print(f"  {'LB':>4} {'Entry%':>7} | {'Index':<14} {'Sharpe':>7} {'Return':>8} "
          f"{'MaxDD':>7} {'WinRate':>8} {'Signals':>7}")
    print("  " + "-" * 75, flush=True)

    t1 = time.time()
    best_sharpe = -999.0
    best_config = ""

    for lb, entry_p in itertools.product(lookbacks, entry_pctiles):
        for sym, daily in sorted(iv_cache.items()):
            r = run_from_series(
                daily,
                iv_lookback=lb,
                entry_pctile=entry_p,
                symbol=sym,
            )
            print(f"  {lb:4d} {entry_p:7.2f} | {sym:<14} {r.sharpe:7.2f} "
                  f"{r.total_return_pct:+7.2f}% {r.max_dd_pct:6.2f}% "
                  f"{r.win_rate*100:7.1f}% {r.n_signals:7d}", flush=True)

            if r.sharpe > best_sharpe and r.n_signals >= 3:
                best_sharpe = r.sharpe
                best_config = f"{sym} lb={lb} entry={entry_p}"

    sweep_time = time.time() - t1
    total_time = time.time() - t0

    print(f"\n  Timing: IV build={build_time:.1f}s, sweep={sweep_time:.1f}s, "
          f"total={total_time:.1f}s")
    print(f"  Best: {best_config} (Sharpe={best_sharpe:.2f})\n", flush=True)


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S4 IV Mean-Reversion Research")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep")
    parser.add_argument("--lookback", type=int, default=30,
                        help="IV lookback (default: 30)")
    parser.add_argument("--entry", type=float, default=0.80,
                        help="Entry percentile (default: 0.80)")
    args = parser.parse_args()

    with tee_to_results("s4_iv_mean_revert"):
        store = MarketDataStore()

        if args.start and args.end:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
        else:
            start, end = _get_date_range(store)
            print(f"Auto-detected date range: {start} to {end}", flush=True)

        store.close()

        if args.sweep:
            run_sweep(start, end)
        else:
            run_single(start, end, iv_lookback=args.lookback,
                       entry_pctile=args.entry)


if __name__ == "__main__":
    main()
