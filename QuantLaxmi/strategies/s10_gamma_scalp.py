"""S10: Gamma Scalping Research — IV percentile-based entry/exit.

Backtest: daily IV percentile check, enter when IV is cheap
(buy ATM straddle using actual option prices from nse_fo_bhavcopy),
exit when DTE < 3 or IV normalizes.

P&L model: actual straddle mark-to-market from nse_fo_bhavcopy daily close prices.
No toy Greek approximations.

Sweeps:
  - iv_pctile_threshold: [0.15, 0.20, 0.30]
  - min_dte: [10, 14, 21]
  - symbols: NIFTY, BANKNIFTY

Usage:
    python -m research.s10_gamma_scalp
    python -m research.s10_gamma_scalp --sweep
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from strategies.s9_momentum.data import is_trading_day
from core.data.store import MarketDataStore


SYMBOLS = ["NIFTY", "BANKNIFTY"]
# Cost per leg in index points (bid-ask + fees for monthly options)
COST_PTS_PER_LEG = {"NIFTY": 3.0, "BANKNIFTY": 5.0}


@dataclass
class GammaTrade:
    symbol: str
    entry_date: date
    exit_date: date
    entry_iv: float
    exit_iv: float
    entry_rv: float
    iv_pctile_entry: float
    entry_straddle_pts: float  # actual straddle cost in points
    exit_straddle_pts: float   # actual straddle value at exit
    pnl_pts: float             # P&L in index points
    pnl_pct: float             # P&L as % of spot
    cost_pts: float            # total transaction cost in points
    exit_reason: str
    dte_at_entry: int


def _get_straddle_price(
    store: MarketDataStore, d: date, symbol: str, strike: float, expiry_str: str,
) -> float | None:
    """Get ATM straddle close price (CE+PE) from nse_fo_bhavcopy."""
    d_str = d.isoformat()
    try:
        df = store.sql(
            'SELECT "OptnTp" as opt, "ClsPric" as close, "SttlmPric" as settle '
            "FROM nse_fo_bhavcopy "
            'WHERE date = ? AND "TckrSymb" = ? AND "StrkPric" = ? '
            'AND "OptnTp" IN (\'CE\', \'PE\') AND "XpryDt" = ?',
            [d_str, symbol, strike, expiry_str],
        )
        if len(df) < 2:
            return None
        total = 0.0
        for _, row in df.iterrows():
            settle = float(row["settle"])
            close = float(row["close"])
            price = settle if settle > 0 else close
            if price <= 0:
                return None
            total += price
        return total
    except Exception:
        return None


def _get_atm_iv_and_spot(
    store: MarketDataStore, symbol: str, d: date, min_dte: int = 7,
) -> tuple[float, float, int, float, str] | None:
    """Get ATM IV, spot price, DTE, ATM strike, and expiry string.

    Returns: (atm_iv, spot, dte, atm_strike, expiry_str) or None.
    Uses nse_contract_delta for NSE-computed IV when available,
    falls back to nse_fo_bhavcopy price-based estimation.
    """
    d_str = d.isoformat()

    # Get spot
    try:
        _idx_name = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}.get(
            symbol.upper(), f"Nifty {symbol}")
        spot_df = store.sql(
            'SELECT "Closing Index Value" as close FROM nse_index_close '
            'WHERE date = ? AND "Index Name" = ? LIMIT 1',
            [d_str, _idx_name],
        )
        if spot_df.empty:
            return None
        spot = float(spot_df["close"].iloc[0])
    except Exception:
        return None

    # Find ATM strike and nearest monthly expiry with DTE >= min_dte
    interval = 50 if symbol == "NIFTY" else 100
    atm_strike = round(spot / interval) * interval

    # Get available expiries from nse_fo_bhavcopy
    try:
        exp_df = store.sql(
            'SELECT DISTINCT "XpryDt" as expiry FROM nse_fo_bhavcopy '
            'WHERE date = ? AND "TckrSymb" = ? AND "OptnTp" = \'CE\' '
            'AND "StrkPric" = ? ORDER BY "XpryDt"',
            [d_str, symbol, atm_strike],
        )
        if exp_df.empty:
            return None

        for _, row in exp_df.iterrows():
            exp_str = str(row["expiry"])[:10]
            try:
                exp = date.fromisoformat(exp_str)
                dte = (exp - d).days
                if dte >= min_dte:
                    # Get IV from nse_contract_delta if available
                    atm_iv = _get_iv_for_strike(store, d, symbol, atm_strike, exp_str)
                    if atm_iv is None:
                        # Fallback: estimate from straddle price
                        straddle = _get_straddle_price(store, d, symbol, atm_strike, exp_str)
                        if straddle is not None and straddle > 0:
                            t_years = dte / 365
                            if t_years > 0:
                                atm_iv = (straddle / 2) / (spot * math.sqrt(t_years)) * math.sqrt(2 * math.pi)
                                atm_iv = max(0.05, min(atm_iv, 2.0))
                            else:
                                atm_iv = 0.15
                        else:
                            continue
                    return atm_iv, spot, dte, atm_strike, exp_str
            except Exception:
                continue
    except Exception:
        return None

    return None


def _get_iv_for_strike(
    store: MarketDataStore, d: date, symbol: str, strike: float, expiry_str: str,
) -> float | None:
    """Get IV from nse_contract_delta for a specific strike."""
    d_str = d.isoformat()
    try:
        iv_df = store.sql(
            'SELECT "IV" as iv FROM nse_contract_delta '
            'WHERE date = ? AND UPPER("SYMBOL") LIKE ? '
            'AND "INSTRUMENT" = \'CALL\' AND "IV" > 0 '
            'AND "STRIKE" = ? AND "EXPIRY_DT" = ?',
            [d_str, f"%{symbol.upper()}%", strike, expiry_str],
        )
        if not iv_df.empty:
            return float(iv_df["iv"].iloc[0]) / 100
    except Exception:
        pass
    return None


def backtest_gamma_scalp(
    store: MarketDataStore,
    start: date,
    end: date,
    symbol: str = "NIFTY",
    iv_pctile_threshold: float = 0.20,
    min_dte: int = 14,
    cost_pts_per_leg: float | None = None,
) -> dict:
    """Backtest gamma scalping using ACTUAL straddle prices from nse_fo_bhavcopy.

    Entry: buy ATM straddle at close prices from nse_fo_bhavcopy.
    Daily MTM: actual straddle close prices from nse_fo_bhavcopy.
    Exit: sell straddle at close prices.
    Cost: fixed pts per leg (3 pts NIFTY / 5 pts BANKNIFTY), 2 legs.
    """
    cpl = cost_pts_per_leg if cost_pts_per_leg is not None else COST_PTS_PER_LEG.get(symbol, 3.0)

    trades: list[GammaTrade] = []
    iv_history: list[float] = []
    spot_history: list[float] = []
    daily_returns: list[float] = []
    # position: {entry_date, entry_iv, entry_rv, entry_spot, entry_straddle,
    #            atm_strike, expiry_str, dte, iv_pctile, last_straddle}
    position = None

    d = start - timedelta(days=90)  # warmup for IV percentile
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        result = _get_atm_iv_and_spot(store, symbol, d, min_dte=7)
        if result is None:
            if d >= start:
                daily_returns.append(0.0)
            d += timedelta(days=1)
            continue

        atm_iv, spot, dte, atm_strike, expiry_str = result
        iv_history.append(atm_iv)
        spot_history.append(spot)

        if d < start:
            d += timedelta(days=1)
            continue

        # Compute IV percentile
        if len(iv_history) < 60:
            daily_returns.append(0.0)
            d += timedelta(days=1)
            continue

        recent_ivs = iv_history[-60:]
        iv_pctile = sum(1 for x in recent_ivs if x <= atm_iv) / len(recent_ivs)

        # Compute realized vol (20-day)
        if len(spot_history) >= 21:
            log_rets = [math.log(spot_history[-i] / spot_history[-i - 1])
                        for i in range(1, 21)]
            realized_vol = np.std(log_rets, ddof=1) * math.sqrt(252)
        else:
            realized_vol = atm_iv

        vrp = realized_vol - atm_iv

        if position is not None:
            # In position — compute actual daily MTM
            remaining_dte = position["dte"] - (d - position["entry_date"]).days

            # Get current straddle value from actual prices
            current_straddle = _get_straddle_price(
                store, d, symbol, position["atm_strike"], position["expiry_str"]
            )

            if current_straddle is not None:
                prev_straddle = position["last_straddle"]
                day_pnl_pts = current_straddle - prev_straddle
                day_ret = day_pnl_pts / position["entry_spot"]
                daily_returns.append(day_ret)
                position["last_straddle"] = current_straddle
                position["last_iv"] = atm_iv
            else:
                daily_returns.append(0.0)

            # Exit conditions
            if remaining_dte < 3:
                exit_reason = "dte_low"
            elif iv_pctile > 0.50:
                exit_reason = "iv_normalized"
            else:
                d += timedelta(days=1)
                continue

            # Close position — use last known straddle price
            exit_straddle = position["last_straddle"]
            entry_straddle = position["entry_straddle"]
            # Cost: entry 2 legs + exit 2 legs
            total_cost = cpl * 2 * 2
            gross_pnl = exit_straddle - entry_straddle
            net_pnl = gross_pnl - total_cost
            pnl_pct = net_pnl / position["entry_spot"] * 100

            trades.append(GammaTrade(
                symbol=symbol, entry_date=position["entry_date"], exit_date=d,
                entry_iv=position["entry_iv"], exit_iv=atm_iv,
                entry_rv=position["entry_rv"],
                iv_pctile_entry=position["iv_pctile"],
                entry_straddle_pts=round(entry_straddle, 2),
                exit_straddle_pts=round(exit_straddle, 2),
                pnl_pts=round(net_pnl, 2),
                pnl_pct=round(pnl_pct, 4),
                cost_pts=round(total_cost, 2),
                exit_reason=exit_reason,
                dte_at_entry=position["dte"],
            ))
            position = None
        else:
            daily_returns.append(0.0)

            # Entry conditions
            if iv_pctile > iv_pctile_threshold:
                d += timedelta(days=1)
                continue
            if vrp > -0.02:
                d += timedelta(days=1)
                continue
            if dte < min_dte:
                d += timedelta(days=1)
                continue

            # Get actual straddle price at entry
            # Use same expiry that _get_atm_iv_and_spot found
            entry_straddle = _get_straddle_price(store, d, symbol, atm_strike, expiry_str)
            if entry_straddle is None or entry_straddle <= 0:
                d += timedelta(days=1)
                continue

            # Enter: buy ATM straddle at actual prices
            entry_cost = cpl * 2  # 2 legs entry
            position = {
                "entry_date": d, "entry_iv": atm_iv,
                "entry_rv": realized_vol, "entry_spot": spot,
                "entry_straddle": entry_straddle,
                "last_straddle": entry_straddle,
                "last_iv": atm_iv,
                "atm_strike": atm_strike, "expiry_str": expiry_str,
                "dte": dte, "iv_pctile": iv_pctile,
            }
            # Deduct entry cost as fraction of spot
            daily_returns[-1] -= entry_cost / spot

        d += timedelta(days=1)

    # Close remaining position at last known price
    if position is not None:
        exit_straddle = position["last_straddle"]
        entry_straddle = position["entry_straddle"]
        total_cost = cpl * 2 * 2
        gross_pnl = exit_straddle - entry_straddle
        net_pnl = gross_pnl - total_cost
        pnl_pct = net_pnl / position["entry_spot"] * 100
        trades.append(GammaTrade(
            symbol=symbol, entry_date=position["entry_date"], exit_date=end,
            entry_iv=position["entry_iv"],
            exit_iv=iv_history[-1] if iv_history else 0,
            entry_rv=position["entry_rv"],
            iv_pctile_entry=position["iv_pctile"],
            entry_straddle_pts=round(entry_straddle, 2),
            exit_straddle_pts=round(exit_straddle, 2),
            pnl_pts=round(net_pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            cost_pts=round(total_cost, 2),
            exit_reason="end_of_backtest",
            dte_at_entry=position["dte"],
        ))

    # Metrics
    backtest_rets = daily_returns
    if not backtest_rets or all(r == 0 for r in backtest_rets):
        return {
            "symbol": symbol, "iv_pctile_threshold": iv_pctile_threshold,
            "min_dte": min_dte, "trades": len(trades),
            "total_return_pct": 0, "sharpe": 0, "max_dd_pct": 0,
            "wins": 0, "win_rate": 0, "daily_returns": [],
        }

    rets = np.array(backtest_rets)
    equity = np.cumprod(1 + rets)
    total_ret = (equity[-1] - 1) * 100
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)) if np.std(rets, ddof=1) > 0 else 0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd)) * 100

    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / len(trades) if trades else 0

    return {
        "symbol": symbol, "iv_pctile_threshold": iv_pctile_threshold,
        "min_dte": min_dte,
        "trades": len(trades), "wins": wins, "win_rate": win_rate,
        "total_return_pct": total_ret, "sharpe": sharpe, "max_dd_pct": max_dd,
        "avg_entry_iv": np.mean([t.entry_iv for t in trades]) if trades else 0,
        "avg_entry_rv": np.mean([t.entry_rv for t in trades]) if trades else 0,
        "avg_straddle_pts": np.mean([t.entry_straddle_pts for t in trades]) if trades else 0,
        "avg_cost_pts": np.mean([t.cost_pts for t in trades]) if trades else 0,
        "trade_details": trades,
        "daily_returns": backtest_rets,
    }


def _get_date_range(store: MarketDataStore) -> tuple[date, date]:
    dates = store.available_dates("nse_index_close")
    return min(dates), max(dates)


def run_single(store: MarketDataStore, start: date, end: date,
               iv_pctile: float = 0.20, min_dte: int = 14) -> dict:
    print(f"\n{'='*70}")
    print(f"S10 Gamma Scalping: {start} to {end}")
    print(f"  iv_pctile_threshold={iv_pctile}, min_dte={min_dte}")
    print("=" * 70)

    all_results = {}
    for symbol in SYMBOLS:
        t0 = time.time()
        result = backtest_gamma_scalp(
            store, start, end, symbol=symbol,
            iv_pctile_threshold=iv_pctile, min_dte=min_dte,
        )
        elapsed = time.time() - t0
        all_results[symbol] = result
        print(f"\n  {symbol} ({elapsed:.1f}s):")
        print(f"    Trades:     {result['trades']}")
        print(f"    Win Rate:   {result['win_rate']*100:.1f}%")
        print(f"    Total Ret:  {result['total_return_pct']:+.2f}%")
        print(f"    Sharpe:     {result['sharpe']:.2f}")
        print(f"    Max DD:     {result['max_dd_pct']:.2f}%")
        if result.get('avg_entry_iv'):
            print(f"    Avg IV:     {result['avg_entry_iv']:.4f}")
            print(f"    Avg RV:     {result['avg_entry_rv']:.4f}")
        if result.get('avg_straddle_pts'):
            print(f"    Avg Straddle: {result['avg_straddle_pts']:.1f} pts")
            print(f"    Avg Cost:   {result['avg_cost_pts']:.1f} pts")

    return all_results


def run_sweep(store: MarketDataStore, start: date, end: date) -> None:
    iv_pctiles = [0.15, 0.20, 0.30]
    min_dtes = [10, 14, 21]

    print(f"\n{'='*70}")
    print(f"S10 Gamma Scalp — Parameter Sweep: {start} to {end}")
    print(f"  iv_pctile_thresholds: {iv_pctiles}")
    print(f"  min_dtes: {min_dtes}")
    print("=" * 70)

    header = (f"  {'IVP':>5} {'DTE':>4} {'Symbol':>10} {'Trades':>6} "
              f"{'WinRate':>8} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"\n{header}")
    print("  " + "-" * 65)

    best_sharpe = -999
    best_config = None

    for ivp, dte in itertools.product(iv_pctiles, min_dtes):
        for symbol in SYMBOLS:
            result = backtest_gamma_scalp(
                store, start, end, symbol=symbol,
                iv_pctile_threshold=ivp, min_dte=dte,
            )
            print(f"  {ivp:5.2f} {dte:4d} {symbol:>10} {result['trades']:6d} "
                  f"{result['win_rate']*100:7.1f}% {result['total_return_pct']:+7.2f}% "
                  f"{result['sharpe']:7.2f} {result['max_dd_pct']:6.2f}%")

            if result['sharpe'] > best_sharpe and result['trades'] > 2:
                best_sharpe = result['sharpe']
                best_config = (ivp, dte, symbol, result)

    if best_config:
        ivp, dte, sym, r = best_config
        print(f"\n  BEST: {sym} ivp={ivp} dte={dte} → "
              f"Sharpe={r['sharpe']:.2f} Ret={r['total_return_pct']:+.2f}% "
              f"WR={r['win_rate']*100:.0f}%")


def main() -> None:
    from research.utils import tee_to_results

    parser = argparse.ArgumentParser(description="S10 Gamma Scalp Research")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--ivp", type=float, default=0.20)
    parser.add_argument("--dte", type=int, default=14)
    args = parser.parse_args()

    with tee_to_results("s10_gamma_scalp"):
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
            run_single(store, start, end, iv_pctile=args.ivp, min_dte=args.dte)

        store.close()


if __name__ == "__main__":
    main()
