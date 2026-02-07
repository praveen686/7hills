"""S9: Cross-Sectional Stock FnO Momentum Research — Parameter sweep.

Uses run_daily_scan() from strategies/s9_momentum/scanner.py to get ranked
stock signals, then simulates weekly rebalance portfolio.

Sweeps:
  - top_n: [3, 5, 7]
  - lookback: [15, 20, 30]

Data: nse_delivery, nse_fo_bhavcopy, nse_fii_stats.

Usage:
    python -m research.s9_momentum
    python -m research.s9_momentum --sweep
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from strategies.s9_momentum.scanner import run_daily_scan
from strategies.s9_momentum.data import is_trading_day, get_fno
from core.market.store import MarketDataStore


COST_BPS = 10.0
logger = logging.getLogger(__name__)


@dataclass
class MomentumTrade:
    symbol: str
    direction: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    pnl_pct: float
    composite_score: float


def _get_stock_future_price(store: MarketDataStore, symbol: str, d: date) -> float | None:
    """Get closing price of nearest-expiry stock future."""
    d_str = d.isoformat()
    try:
        df = store.sql(
            "SELECT \"ClsPric\" as close, \"XpryDt\" as expiry "
            "FROM nse_fo_bhavcopy "
            "WHERE date = ? AND \"TckrSymb\" = ? AND \"FinInstrmTp\" = 'STF' "
            "ORDER BY \"XpryDt\" LIMIT 1",
            [d_str, symbol],
        )
        if df.empty:
            return None
        return float(df["close"].iloc[0])
    except Exception:
        return None


def backtest_momentum(
    store: MarketDataStore,
    start: date,
    end: date,
    top_n: int = 5,
    rebalance_day: int = 0,  # Monday
    cost_bps: float = COST_BPS,
) -> dict:
    """Backtest cross-sectional momentum with weekly rebalance."""
    cost_frac = cost_bps / 10_000

    trades: list[MomentumTrade] = []
    positions: dict[str, dict] = {}  # symbol → {direction, entry_date, entry_price, score}
    daily_returns: list[float] = []

    d = start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        # Track daily P&L from open positions
        day_pnl = 0.0
        for sym, pos in list(positions.items()):
            price = _get_stock_future_price(store, sym, d)
            if price is not None:
                prev_price = pos.get("last_price", pos["entry_price"])
                if pos["direction"] == "long":
                    day_pnl += (price / prev_price - 1) / max(len(positions), 1)
                else:
                    day_pnl -= (price / prev_price - 1) / max(len(positions), 1)
                pos["last_price"] = price

        daily_returns.append(day_pnl)

        # Rebalance on specified weekday
        if d.weekday() == rebalance_day:
            try:
                signals = run_daily_scan(d, store, top_n=top_n * 2)
            except Exception as e:
                logger.debug("Scan failed for %s: %s", d, e)
                d += timedelta(days=1)
                continue

            if not signals:
                d += timedelta(days=1)
                continue

            # New target portfolio: top_n longs + top_n shorts
            longs = [s for s in signals if s.composite_score > 0][:top_n]
            shorts = [s for s in signals if s.composite_score < 0][:top_n]
            target = {s.symbol: ("long", s.composite_score) for s in longs}
            target.update({s.symbol: ("short", s.composite_score) for s in shorts})

            # Close positions not in new target
            for sym in list(positions.keys()):
                if sym not in target:
                    price = _get_stock_future_price(store, sym, d)
                    if price is not None:
                        pnl = (price / positions[sym]["entry_price"] - 1)
                        if positions[sym]["direction"] == "short":
                            pnl = -pnl
                        pnl -= cost_frac
                        trades.append(MomentumTrade(
                            symbol=sym, direction=positions[sym]["direction"],
                            entry_date=positions[sym]["entry_date"], exit_date=d,
                            entry_price=positions[sym]["entry_price"], exit_price=price,
                            pnl_pct=pnl * 100,
                            composite_score=positions[sym]["score"],
                        ))
                    del positions[sym]

            # Open new positions
            for sym, (direction, score) in target.items():
                if sym in positions:
                    continue  # already held
                price = _get_stock_future_price(store, sym, d)
                if price is not None and price > 0:
                    positions[sym] = {
                        "direction": direction,
                        "entry_date": d,
                        "entry_price": price,
                        "last_price": price,
                        "score": score,
                    }
                    daily_returns[-1] -= cost_frac / max(len(target), 1)

        d += timedelta(days=1)

    # Close remaining positions at end
    for sym, pos in positions.items():
        price = _get_stock_future_price(store, sym, end)
        if price is not None:
            pnl = (price / pos["entry_price"] - 1)
            if pos["direction"] == "short":
                pnl = -pnl
            pnl -= cost_frac
            trades.append(MomentumTrade(
                symbol=sym, direction=pos["direction"],
                entry_date=pos["entry_date"], exit_date=end,
                entry_price=pos["entry_price"], exit_price=price,
                pnl_pct=pnl * 100, composite_score=pos["score"],
            ))

    # Compute metrics
    if not daily_returns or all(r == 0 for r in daily_returns):
        return {
            "top_n": top_n, "trades": len(trades),
            "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0,
            "wins": 0, "win_rate": 0, "daily_returns": [],
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

    return {
        "top_n": top_n,
        "trades": len(trades), "wins": wins, "win_rate": win_rate,
        "total_return_pct": total_ret, "sharpe": sharpe, "max_dd_pct": max_dd,
        "unique_symbols": len(set(t.symbol for t in trades)),
        "trade_details": trades,
        "daily_returns": daily_returns,
    }


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    dates = store.available_dates("nse_fo_bhavcopy")
    return min(dates), max(dates)


def run_single(store: MarketDataStore, start: date, end: date,
               top_n: int = 5) -> dict:
    print(f"\n{'='*70}")
    print(f"S9 Cross-Sectional Momentum: {start} to {end}")
    print(f"  top_n={top_n}")
    print("=" * 70)

    t0 = time.time()
    result = backtest_momentum(store, start, end, top_n=top_n)
    elapsed = time.time() - t0
    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Trades:       {result['trades']}")
    print(f"    Unique Syms:  {result.get('unique_symbols', 0)}")
    print(f"    Win Rate:     {result['win_rate']*100:.1f}%")
    print(f"    Total Ret:    {result['total_return_pct']:+.2f}%")
    print(f"    Sharpe:       {result['sharpe']:.2f}")
    print(f"    Max DD:       {result['max_dd_pct']:.2f}%")
    return result


def run_sweep(store: MarketDataStore, start: date, end: date) -> None:
    top_ns = [3, 5, 7]

    print(f"\n{'='*70}")
    print(f"S9 Momentum — Parameter Sweep: {start} to {end}")
    print(f"  top_n: {top_ns}")
    print("=" * 70)

    header = (f"  {'TopN':>4} {'Trades':>6} {'Syms':>4} "
              f"{'WinRate':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"\n{header}")
    print("  " + "-" * 55)

    best_sharpe = -999
    best_config = None

    for tn in top_ns:
        result = backtest_momentum(store, start, end, top_n=tn)
        print(f"  {tn:4d} {result['trades']:6d} {result.get('unique_symbols', 0):4d} "
              f"{result['win_rate']*100:7.1f}% {result['total_return_pct']:+7.2f}% "
              f"{result['sharpe']:7.2f} {result['max_dd_pct']:6.2f}%")

        if result['sharpe'] > best_sharpe and result['trades'] > 3:
            best_sharpe = result['sharpe']
            best_config = (tn, result)

    if best_config:
        tn, r = best_config
        print(f"\n  BEST: top_n={tn} → "
              f"Sharpe={r['sharpe']:.2f} Ret={r['total_return_pct']:+.2f}% "
              f"WR={r['win_rate']*100:.0f}%")


def main() -> None:
    from strategies.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S9 Momentum Research")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    with tee_to_results("s9_momentum"):
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
            run_single(store, start, end, top_n=args.top_n)

        store.close()


if __name__ == "__main__":
    main()
