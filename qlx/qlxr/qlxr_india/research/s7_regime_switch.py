"""S7: Information-Theoretic Regime Switching Research — Parameter sweep.

Uses classify_regime() from strategies/s7_regime/detector.py to classify
daily market regimes, then applies sub-strategy logic:
  - TRENDING → trend-following (SuperTrend direction)
  - MEAN_REVERTING → Bollinger Band fade
  - RANDOM → flat (no trade)

Sweeps:
  - lookback: [60, 100, 140]
  - supertrend_mult: [2, 3, 4]
  - symbols: NIFTY, BANKNIFTY

Usage:
    python -m research.s7_regime_switch
    python -m research.s7_regime_switch --sweep
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from strategies.s7_regime.detector import (
    MarketRegime,
    classify_regime,
    VPIN_TOXIC,
)
from strategies.s9_momentum.data import is_trading_day
from qlx.data.store import MarketDataStore


SYMBOLS = ["NIFTY", "BANKNIFTY"]
# Map from our symbol IDs to exact nse_index_close "Index Name" values
_INDEX_NAME = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
}
COST_BPS = 10.0


@dataclass
class RegimeTrade:
    symbol: str
    entry_date: date
    exit_date: date
    direction: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    regime: str
    sub_strategy: str


def _compute_vpin(prices: np.ndarray, window: int = 50) -> float:
    """VPIN via BVC (copied from strategy for standalone use)."""
    from scipy.stats import norm

    n = len(prices)
    if n < window + 1:
        return 0.0
    log_ret = np.diff(np.log(np.maximum(prices, 1e-8)))
    sigma = np.std(log_ret[-window:], ddof=1)
    if sigma < 1e-8:
        return 0.0
    recent_ret = log_ret[-window:]
    buy_frac = norm.cdf(recent_ret / sigma)
    imbalance = np.abs(2 * buy_frac - 1)
    return float(np.mean(imbalance))


def backtest_regime_switch(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    lookback: int = 100,
    supertrend_mult: float = 3.0,
    cost_bps: float = COST_BPS,
) -> dict:
    """Backtest the regime-switching strategy day by day."""
    cost_frac = cost_bps / 10_000

    # Load all daily closes
    d_str_start = (start - timedelta(days=lookback * 2)).isoformat()
    d_str_end = end.isoformat()

    index_name = _INDEX_NAME.get(symbol.upper(), f"Nifty {symbol}")
    df = store.sql(
        'SELECT date, "Closing Index Value" as close '
        "FROM nse_index_close "
        'WHERE date BETWEEN ? AND ? AND "Index Name" = ? '
        "ORDER BY date",
        [d_str_start, d_str_end, index_name],
    )

    if df.empty or len(df) < lookback + 10:
        return {"symbol": symbol, "lookback": lookback, "supertrend_mult": supertrend_mult,
                "trades": 0, "wins": 0, "win_rate": 0, "total_return_pct": 0, "sharpe": 0,
                "max_dd_pct": 0, "trade_details": []}

    dates_arr = [date.fromisoformat(str(x)[:10]) if isinstance(x, str) else x for x in df["date"].tolist()]
    prices_arr = df["close"].astype(float).values

    trades: list[RegimeTrade] = []
    position = None  # {"direction", "entry_date", "entry_price", "sub_strategy"}
    daily_returns: list[float] = []

    for i in range(lookback, len(prices_arr)):
        d_val = dates_arr[i]

        if d_val < start:
            continue

        window = prices_arr[i - lookback: i + 1]
        vpin = _compute_vpin(window)
        regime_obs = classify_regime(window, vpin=vpin, entropy_window=lookback)

        spot = float(prices_arr[i])
        prev_spot = float(prices_arr[i - 1])

        # Daily return from held position
        if position is not None:
            if position["direction"] == "long":
                day_ret = (spot / prev_spot - 1)
            else:
                day_ret = -(spot / prev_spot - 1)
            daily_returns.append(day_ret)
        else:
            daily_returns.append(0.0)

        # VPIN kill switch — exit any position
        if vpin > VPIN_TOXIC and position is not None:
            pnl = (spot / position["entry_price"] - 1)
            if position["direction"] == "short":
                pnl = -pnl
            pnl -= cost_frac  # exit cost
            trades.append(RegimeTrade(
                symbol=symbol, entry_date=position["entry_date"],
                exit_date=d_val, direction=position["direction"],
                entry_price=position["entry_price"], exit_price=spot,
                pnl_pct=pnl * 100, regime="vpin_kill",
                sub_strategy=position["sub_strategy"],
            ))
            position = None
            continue

        # Regime-contingent logic
        if regime_obs.regime == MarketRegime.TRENDING:
            # SuperTrend direction
            period = 14
            if i >= period + 2:
                returns = np.abs(np.diff(prices_arr[i - period: i + 1]))
                atr = np.mean(returns[-period:])
                mid = (prices_arr[i] + prices_arr[i - 1]) / 2
                upper = mid + supertrend_mult * atr
                lower = mid - supertrend_mult * atr

                deltas = np.diff(prices_arr[i - 15: i + 1])
                gains = np.mean(np.maximum(deltas, 0))
                losses = np.mean(np.maximum(-deltas, 0))
                rs = gains / losses if losses > 0 else 100
                rsi = 100 - (100 / (1 + rs))

                if spot > lower and 40 < rsi < 70:
                    desired = "long"
                elif spot < upper and 30 < rsi < 60:
                    desired = "short"
                else:
                    desired = None

                if desired and position is None:
                    position = {
                        "direction": desired, "entry_date": d_val,
                        "entry_price": spot, "sub_strategy": "trend",
                    }
                    daily_returns[-1] -= cost_frac  # entry cost
                elif desired and position is not None and position["direction"] != desired:
                    # Close old, open new
                    pnl = (spot / position["entry_price"] - 1)
                    if position["direction"] == "short":
                        pnl = -pnl
                    pnl -= cost_frac
                    trades.append(RegimeTrade(
                        symbol=symbol, entry_date=position["entry_date"],
                        exit_date=d_val, direction=position["direction"],
                        entry_price=position["entry_price"], exit_price=spot,
                        pnl_pct=pnl * 100, regime="trending",
                        sub_strategy="trend",
                    ))
                    position = {
                        "direction": desired, "entry_date": d_val,
                        "entry_price": spot, "sub_strategy": "trend",
                    }
                    daily_returns[-1] -= cost_frac

        elif regime_obs.regime == MarketRegime.MEAN_REVERTING:
            bb_window = 20
            if i >= bb_window:
                recent = prices_arr[i - bb_window + 1: i + 1]
                mu = np.mean(recent)
                std = np.std(recent, ddof=1)
                if std > 1e-8:
                    z = (spot - mu) / std

                    if position is not None and abs(z) < 0.5:
                        # Mean reverted — exit
                        pnl = (spot / position["entry_price"] - 1)
                        if position["direction"] == "short":
                            pnl = -pnl
                        pnl -= cost_frac
                        trades.append(RegimeTrade(
                            symbol=symbol, entry_date=position["entry_date"],
                            exit_date=d_val, direction=position["direction"],
                            entry_price=position["entry_price"], exit_price=spot,
                            pnl_pct=pnl * 100, regime="mean_reverting",
                            sub_strategy="mean_reversion",
                        ))
                        position = None

                    elif position is None:
                        if z < -2.0:
                            position = {
                                "direction": "long", "entry_date": d_val,
                                "entry_price": spot, "sub_strategy": "mean_reversion",
                            }
                            daily_returns[-1] -= cost_frac
                        elif z > 2.0:
                            position = {
                                "direction": "short", "entry_date": d_val,
                                "entry_price": spot, "sub_strategy": "mean_reversion",
                            }
                            daily_returns[-1] -= cost_frac

        elif regime_obs.regime == MarketRegime.RANDOM:
            # No directional edge — exit if in position
            if position is not None:
                pnl = (spot / position["entry_price"] - 1)
                if position["direction"] == "short":
                    pnl = -pnl
                pnl -= cost_frac
                trades.append(RegimeTrade(
                    symbol=symbol, entry_date=position["entry_date"],
                    exit_date=d_val, direction=position["direction"],
                    entry_price=position["entry_price"], exit_price=spot,
                    pnl_pct=pnl * 100, regime="random",
                    sub_strategy=position["sub_strategy"],
                ))
                position = None

    # Close any remaining position at end
    if position is not None:
        spot = float(prices_arr[-1])
        pnl = (spot / position["entry_price"] - 1)
        if position["direction"] == "short":
            pnl = -pnl
        pnl -= cost_frac
        trades.append(RegimeTrade(
            symbol=symbol, entry_date=position["entry_date"],
            exit_date=end, direction=position["direction"],
            entry_price=position["entry_price"], exit_price=spot,
            pnl_pct=pnl * 100, regime="end",
            sub_strategy=position["sub_strategy"],
        ))

    # Compute metrics
    if not daily_returns:
        return {"symbol": symbol, "lookback": lookback, "supertrend_mult": supertrend_mult,
                "trades": 0, "wins": 0, "win_rate": 0, "total_return_pct": 0, "sharpe": 0,
                "max_dd_pct": 0, "trade_details": [], "daily_returns": []}

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
        "symbol": symbol,
        "lookback": lookback,
        "supertrend_mult": supertrend_mult,
        "trades": len(trades),
        "wins": wins,
        "win_rate": win_rate,
        "total_return_pct": total_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "trade_details": trades,
        "daily_returns": daily_returns,
    }


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    dates = store.available_dates("nse_index_close")
    return min(dates), max(dates)


def run_single(store: MarketDataStore, start: date, end: date,
               lookback: int = 100, supertrend_mult: float = 3.0) -> dict:
    print(f"\n{'='*70}")
    print(f"S7 Regime Switch: {start} to {end}")
    print(f"  lookback={lookback}, supertrend_mult={supertrend_mult}")
    print("=" * 70)

    all_results = {}
    for symbol in SYMBOLS:
        t0 = time.time()
        result = backtest_regime_switch(
            store, start, end, symbol=symbol,
            lookback=lookback, supertrend_mult=supertrend_mult,
        )
        elapsed = time.time() - t0
        all_results[symbol] = result
        print(f"\n  {symbol} ({elapsed:.1f}s):")
        print(f"    Trades:     {result['trades']}")
        print(f"    Win Rate:   {result['win_rate']*100:.1f}%")
        print(f"    Total Ret:  {result['total_return_pct']:+.2f}%")
        print(f"    Sharpe:     {result['sharpe']:.2f}")
        print(f"    Max DD:     {result['max_dd_pct']:.2f}%")

    return all_results


def run_sweep(store: MarketDataStore, start: date, end: date) -> None:
    lookbacks = [60, 100, 140]
    st_mults = [2.0, 3.0, 4.0]

    print(f"\n{'='*70}")
    print(f"S7 Regime Switch — Parameter Sweep: {start} to {end}")
    print(f"  lookbacks: {lookbacks}")
    print(f"  supertrend_mult: {st_mults}")
    print("=" * 70)

    header = (f"  {'LB':>4} {'STM':>4} {'Symbol':>10} {'Trades':>6} "
              f"{'WinRate':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"\n{header}")
    print("  " + "-" * 65)

    best_sharpe = -999
    best_config = None

    for lb, stm in itertools.product(lookbacks, st_mults):
        for symbol in SYMBOLS:
            result = backtest_regime_switch(
                store, start, end, symbol=symbol,
                lookback=lb, supertrend_mult=stm,
            )
            print(f"  {lb:4d} {stm:4.1f} {symbol:>10} {result['trades']:6d} "
                  f"{result['win_rate']*100:7.1f}% {result['total_return_pct']:+7.2f}% "
                  f"{result['sharpe']:7.2f} {result['max_dd_pct']:6.2f}%")

            if result['sharpe'] > best_sharpe and result['trades'] > 3:
                best_sharpe = result['sharpe']
                best_config = (lb, stm, symbol, result)

    if best_config:
        lb, stm, sym, r = best_config
        print(f"\n  BEST: {sym} lb={lb} stm={stm} → "
              f"Sharpe={r['sharpe']:.2f} Ret={r['total_return_pct']:+.2f}% "
              f"WR={r['win_rate']*100:.0f}%")


def main() -> None:
    from research.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S7 Regime Switch Research")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--stm", type=float, default=3.0)
    args = parser.parse_args()

    with tee_to_results("s7_regime_switch"):
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
            run_single(store, start, end, lookback=args.lookback, supertrend_mult=args.stm)

        store.close()


if __name__ == "__main__":
    main()
