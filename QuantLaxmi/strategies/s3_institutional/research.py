"""S3: Institutional Flow Research — Signal quality evaluation.

For each available date:
  1. Run run_daily_scan() to get composite signals
  2. Record top-N signals with scores
  3. Compute 1/3/5-day forward returns from delivery data
  4. Measure hit rate and Information Coefficient (IC)

Usage:
    python -m apps.india_scanner.research.s3_institutional_flow
    python -m apps.india_scanner.research.s3_institutional_flow --start 2025-09-01 --end 2026-01-31
    python -m apps.india_scanner.research.s3_institutional_flow --top-n 10
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from strategies.s9_momentum.data import available_dates, is_trading_day
from strategies.s9_momentum.scanner import run_daily_scan
from data.store import MarketDataStore


def _next_trading_days(d: date, n: int) -> list[date]:
    """Return the next n trading days after d."""
    result = []
    current = d + timedelta(days=1)
    while len(result) < n:
        if is_trading_day(current):
            result.append(current)
        current += timedelta(days=1)
        if (current - d).days > n * 3:
            break
    return result


def _get_close_price(store: MarketDataStore, symbol: str, d: date) -> float | None:
    """Get delivery close price for symbol on date d."""
    df = store.sql(
        "SELECT * FROM nse_delivery WHERE date = ?", [d.isoformat()]
    )
    if df is None or df.empty:
        return None

    # Try matching symbol across various column names
    for col in ["SYMBOL", "symbol", "Symbol"]:
        if col in df.columns:
            row = df[df[col] == symbol]
            if not row.empty:
                for pcol in ["CLOSE_PRICE", "close_price", "ClsPric"]:
                    if pcol in row.columns:
                        return float(row[pcol].iloc[0])
    return None


def _get_cm_close(store: MarketDataStore, symbol: str, d: date) -> float | None:
    """Get CM bhavcopy close price as fallback."""
    df = store.sql(
        "SELECT * FROM nse_cm_bhavcopy WHERE date = ?", [d.isoformat()]
    )
    if df is None or df.empty:
        return None
    for col in ["TckrSymb", "SYMBOL", "symbol"]:
        if col in df.columns:
            row = df[df[col] == symbol]
            if not row.empty:
                for pcol in ["ClsPric", "CLOSE", "close"]:
                    if pcol in row.columns:
                        val = row[pcol].iloc[0]
                        if pd.notna(val) and float(val) > 0:
                            return float(val)
    return None


def run_research(
    store: MarketDataStore,
    start: date,
    end: date,
    top_n: int = 10,
    forward_horizons: tuple[int, ...] = (1, 3, 5),
) -> None:
    """Main research loop."""
    dates = available_dates(store, "nse_delivery")
    dates = [d for d in dates if start <= d <= end]
    dates.sort()

    print(f"\nS3 Institutional Flow Research: {start} to {end}")
    print(f"  {len(dates)} available dates, top-{top_n} signals")
    print(f"  Forward horizons: {forward_horizons} days")
    print()

    # Collect all signals with forward returns
    records: list[dict] = []
    t0 = time.time()

    for i, d in enumerate(dates):
        # Skip last few days (no forward data)
        if d > end - timedelta(days=max(forward_horizons) * 2):
            break

        try:
            signals = run_daily_scan(d, store=store, top_n=top_n)
        except Exception as e:
            print(f"  {d}: scan failed — {e}")
            continue

        if not signals:
            continue

        # Compute forward returns for each signal
        future_dates = {h: _next_trading_days(d, h) for h in forward_horizons}

        for sig in signals:
            entry_price = _get_cm_close(store, sig.symbol, d)
            if entry_price is None or entry_price <= 0:
                continue

            record = {
                "date": d,
                "symbol": sig.symbol,
                "score": sig.composite_score,
                "entry_price": entry_price,
            }

            for h in forward_horizons:
                fwd_dates = future_dates[h]
                if not fwd_dates:
                    record[f"fwd_{h}d"] = np.nan
                    continue
                exit_price = _get_cm_close(store, sig.symbol, fwd_dates[-1])
                if exit_price is not None and exit_price > 0:
                    record[f"fwd_{h}d"] = (exit_price - entry_price) / entry_price
                else:
                    record[f"fwd_{h}d"] = np.nan

            records.append(record)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(dates)} dates ({elapsed:.1f}s, "
                  f"{len(records)} signals collected)")

    elapsed = time.time() - t0
    print(f"\n  Collected {len(records)} signal-observations in {elapsed:.1f}s\n")

    if not records:
        print("  No valid signal-return pairs found.")
        return

    df = pd.DataFrame(records)

    # Results
    print("=" * 70)
    print("S3 RESULTS")
    print("=" * 70)

    for h in forward_horizons:
        col = f"fwd_{h}d"
        valid = df.dropna(subset=[col])
        if valid.empty:
            print(f"\n  {h}-day forward: no data")
            continue

        scores = valid["score"].values
        returns = valid[col].values

        # Direction hit rate: does sign(score) predict sign(return)?
        correct = np.sign(scores) == np.sign(returns)
        hit_rate = correct.mean()

        # Information Coefficient: Spearman rank correlation
        from scipy.stats import spearmanr
        ic, ic_pval = spearmanr(scores, returns)

        # Mean return for long signals (score > 0) vs short (score < 0)
        long_mask = scores > 0
        short_mask = scores < 0
        long_ret = returns[long_mask].mean() * 100 if long_mask.any() else 0
        short_ret = returns[short_mask].mean() * 100 if short_mask.any() else 0

        # Quintile analysis
        n_q = min(5, len(valid) // 10)
        if n_q >= 2:
            valid_sorted = valid.sort_values("score")
            quintile_size = len(valid_sorted) // n_q
            q_returns = []
            for q in range(n_q):
                s = q * quintile_size
                e = s + quintile_size if q < n_q - 1 else len(valid_sorted)
                q_ret = valid_sorted.iloc[s:e][col].mean() * 100
                q_returns.append(q_ret)
            q_spread = q_returns[-1] - q_returns[0]
        else:
            q_returns = []
            q_spread = 0

        print(f"\n  {h}-day forward ({len(valid)} obs):")
        print(f"    Hit rate:        {hit_rate:.1%}")
        print(f"    IC (Spearman):   {ic:.4f} (p={ic_pval:.4f})")
        print(f"    Mean long ret:   {long_ret:+.3f}%")
        print(f"    Mean short ret:  {short_ret:+.3f}%")
        if q_returns:
            print(f"    Quintile spread: {q_spread:+.3f}%")
            q_str = " | ".join(f"Q{i+1}={r:+.3f}%" for i, r in enumerate(q_returns))
            print(f"    Quintiles:       {q_str}")

    # Top symbols by signal frequency
    print(f"\n\n  Top 15 symbols by signal frequency:")
    sym_counts = df["symbol"].value_counts().head(15)
    for sym, cnt in sym_counts.items():
        sym_df = df[df["symbol"] == sym]
        avg_score = sym_df["score"].mean()
        avg_1d = sym_df["fwd_1d"].mean() * 100 if "fwd_1d" in sym_df else 0
        print(f"    {sym:<12} {cnt:4d} signals, avg_score={avg_score:+.2f}, "
              f"avg_1d_ret={avg_1d:+.3f}%")

    print()


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S3 Institutional Flow Research")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N signals per day")
    args = parser.parse_args()

    with tee_to_results("s3_institutional_flow"):
        store = MarketDataStore()

        if args.start and args.end:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
        else:
            dates = available_dates(store, "nse_delivery")
            if not dates:
                print("No delivery data available.")
                return
            start, end = min(dates), max(dates)
            print(f"Auto-detected date range: {start} to {end}")

        run_research(store, start, end, top_n=args.top_n)
        store.close()


if __name__ == "__main__":
    main()
