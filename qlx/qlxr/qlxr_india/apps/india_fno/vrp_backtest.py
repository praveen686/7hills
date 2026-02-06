"""Volatility Risk Premium (VRP) backtest for NIFTY options.

Strategy:
  - Each day, compute ATM IV from nearest weekly expiry option chain
  - Compute 20-day realized vol (HAR-RV) from NIFTY close prices
  - VRP = IV - RV (the premium sellers collect)
  - When VRP > threshold: sell ATM straddle, delta-hedge with futures
  - Exit at expiry (or before if stop-loss hit)

Simplified backtest:
  - Daily P&L from straddle selling = theta collected - gamma losses
  - Approximated as: (IV² - RV²) / 2 × T per day (variance swap approximation)
  - More precisely: we track straddle mark-to-market each day
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from apps.india_fno.iv_engine import (
    compute_chain_iv,
    har_rv,
    realized_vol,
    OptionChainIV,
)
from apps.india_scanner.data import is_trading_day, get_fno, get_delivery
from qlx.data.store import MarketDataStore

logger = logging.getLogger(__name__)


@dataclass
class VRPDayResult:
    """Single day observation in the VRP backtest."""
    date: date
    spot: float
    atm_iv: float       # annualized
    rv_20d: float       # annualized (HAR-RV)
    vrp: float           # IV - RV
    iv_skew_25d: float
    straddle_price: float  # ATM straddle in points
    nearest_expiry: str


@dataclass
class VRPBacktestResult:
    """Full VRP backtest output."""
    daily: list[VRPDayResult]
    trades: list[dict]
    total_return_pct: float
    annual_return_pct: float
    sharpe: float
    max_dd_pct: float
    win_rate: float
    avg_vrp: float
    avg_iv: float
    avg_rv: float


def _get_nifty_close_series(store, start: date, end: date) -> pd.Series:
    """Build NIFTY close price series from F&O data.

    Starts from `start - 45 calendar days` to provide lookback for HAR-RV.
    """
    closes = {}
    d = start - timedelta(days=45)  # 45 cal days ≈ 30 trading days for HAR-RV

    while d <= end:
        if is_trading_day(d):
            try:
                fno = get_fno(store, d)
                if not fno.empty:
                    nifty_opts = fno[
                        (fno["TckrSymb"] == "NIFTY")
                        & (fno["FinInstrmTp"].isin(["IDO", "IDF"]))
                    ]
                    if not nifty_opts.empty and "UndrlygPric" in nifty_opts.columns:
                        spot = float(nifty_opts["UndrlygPric"].iloc[0])
                        if spot > 0:
                            closes[d] = spot
            except Exception:
                pass
        d += timedelta(days=1)

    if not closes:
        return pd.Series(dtype=float)

    series = pd.Series(closes).sort_index()
    return series


def compute_vrp_series(
    store,
    start: date,
    end: date,
) -> list[VRPDayResult]:
    """Compute daily IV, RV, and VRP for the date range."""

    # Build NIFTY close series going back enough for RV computation
    close_series = _get_nifty_close_series(store, start, end)
    if close_series.empty:
        logger.warning("No NIFTY close data available")
        return []

    # Compute realized vol
    rv_series = har_rv(close_series)

    results = []
    d = start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        if d not in close_series.index:
            d += timedelta(days=1)
            continue

        rv_val = rv_series.get(d, float("nan"))

        try:
            fno = get_fno(store, d)
            if fno.empty:
                d += timedelta(days=1)
                continue
            nifty_opts = fno[
                (fno["TckrSymb"] == "NIFTY") & (fno["FinInstrmTp"] == "IDO")
            ].copy()

            if nifty_opts.empty:
                d += timedelta(days=1)
                continue

            chain_iv = compute_chain_iv(nifty_opts)

            if np.isnan(chain_iv.atm_iv):
                d += timedelta(days=1)
                continue

            # ATM straddle price
            spot = chain_iv.spot
            chain = chain_iv.df.dropna(subset=["IV"])
            # Find nearest expiry
            nearest_exp = chain_iv.nearest_expiry
            nearest = chain[chain["XpryDt"] == nearest_exp] if nearest_exp else chain
            if nearest.empty:
                # Fallback: use all
                nearest = chain

            # ATM straddle = CE + PE at strike nearest to spot
            nearest = nearest.copy()
            nearest["_dist"] = abs(nearest["StrkPric"] - spot)
            atm_strike = nearest.loc[nearest["_dist"].idxmin(), "StrkPric"]
            atm_ce = nearest[
                (nearest["StrkPric"] == atm_strike)
                & (nearest["OptnTp"].str.strip() == "CE")
            ]
            atm_pe = nearest[
                (nearest["StrkPric"] == atm_strike)
                & (nearest["OptnTp"].str.strip() == "PE")
            ]
            straddle = 0.0
            if not atm_ce.empty and not atm_pe.empty:
                straddle = float(atm_ce["ClsPric"].iloc[0] + atm_pe["ClsPric"].iloc[0])

            vrp = chain_iv.atm_iv - rv_val if not np.isnan(rv_val) else float("nan")

            results.append(VRPDayResult(
                date=d,
                spot=spot,
                atm_iv=chain_iv.atm_iv,
                rv_20d=rv_val,
                vrp=vrp,
                iv_skew_25d=chain_iv.iv_skew_25d,
                straddle_price=straddle,
                nearest_expiry=nearest_exp,
            ))

            if len(results) % 10 == 0:
                logger.info("VRP computed for %d days (latest: %s, IV=%.1f%%, RV=%.1f%%)",
                            len(results), d, chain_iv.atm_iv * 100,
                            rv_val * 100 if not np.isnan(rv_val) else 0)

        except Exception as e:
            logger.warning("VRP computation failed for %s: %s", d, e)

        d += timedelta(days=1)

    return results


def run_vrp_backtest(
    store,
    start: date,
    end: date,
    entry_vrp: float = 0.03,     # Enter when VRP > 3% (IV 3pts above RV)
    exit_vrp: float = 0.0,       # Exit when VRP < 0 (RV catches up to IV)
    stop_loss_pct: float = 0.02, # Stop loss at 2% of capital per trade
) -> VRPBacktestResult:
    """Run VRP harvesting backtest.

    P&L model: weekly straddle selling.
    - When VRP > threshold: sell ATM straddle at day's close prices.
    - Hold for `hold_days` trading days, then mark-to-market the straddle.
    - P&L = (straddle premium collected - straddle value at exit) / spot.
    - Straddle value at exit = |spot_exit - strike| (intrinsic only at expiry)
      or next day's straddle close price (if exiting before expiry).
    - We approximate exit value using spot move: payoff ≈ |spot_exit - spot_entry|.
    - Net P&L = (straddle_premium - |spot_move|) / spot, then subtract costs.
    """
    logger.info("Computing VRP series %s to %s...", start, end)
    daily = compute_vrp_series(store, start, end)

    if not daily:
        return VRPBacktestResult(
            daily=[], trades=[], total_return_pct=0, annual_return_pct=0,
            sharpe=0, max_dd_pct=0, win_rate=0, avg_vrp=0, avg_iv=0, avg_rv=0,
        )

    # Build lookup by date for easy access
    by_date = {d.date: d for d in daily}

    # --- Trading simulation: weekly straddle selling ---
    trades: list[dict] = []
    daily_pnl: list[float] = []
    hold_days = 5  # hold for ~1 week (5 trading days)
    cost_bps = 30  # round-trip cost in bps (STT + brokerage + charges)
    cost_frac = cost_bps / 10000

    i = 0
    while i < len(daily):
        obs = daily[i]

        if np.isnan(obs.vrp):
            daily_pnl.append(0.0)
            i += 1
            continue

        # Entry condition
        if obs.vrp > entry_vrp and obs.straddle_price > 0:
            entry = obs
            # Find exit: hold_days trading days later
            exit_idx = min(i + hold_days, len(daily) - 1)
            exit_obs = daily[exit_idx]

            # P&L: premium collected - absolute spot move
            spot_move = abs(exit_obs.spot - entry.spot)
            premium_pct = entry.straddle_price / entry.spot
            payoff_pct = spot_move / entry.spot
            trade_pnl = premium_pct - payoff_pct - cost_frac

            trades.append({
                "entry_date": entry.date.isoformat(),
                "exit_date": exit_obs.date.isoformat(),
                "entry_iv": entry.atm_iv,
                "entry_rv": entry.rv_20d,
                "entry_vrp": entry.vrp,
                "exit_iv": exit_obs.atm_iv,
                "exit_rv": exit_obs.rv_20d,
                "exit_vrp": exit_obs.vrp,
                "spot_entry": entry.spot,
                "spot_exit": exit_obs.spot,
                "straddle_premium": entry.straddle_price,
                "spot_move": spot_move,
                "pnl_pct": trade_pnl,
                "exit_reason": "hold_period" if exit_idx < len(daily) - 1 else "end_of_backtest",
                "days_held": (exit_obs.date - entry.date).days,
            })

            # Assign daily P&L evenly across hold period
            days_in_trade = exit_idx - i
            if days_in_trade > 0:
                daily_share = trade_pnl / days_in_trade
                for j in range(i, exit_idx):
                    daily_pnl.append(daily_share)
            else:
                daily_pnl.append(trade_pnl)

            i = exit_idx  # jump to exit, then look for next entry
        else:
            daily_pnl.append(0.0)

        i += 1

    # --- Performance metrics ---
    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    trading_days = len([p for p in daily_pnl if p != 0.0])
    years = max(len(daily_pnl) / 252, 1 / 252)
    annual_return = total_return / years

    daily_nonzero = pnl_arr[pnl_arr != 0.0]
    if len(daily_nonzero) > 1:
        sharpe = float(np.mean(daily_nonzero) / np.std(daily_nonzero) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Win rate
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    win_rate = wins / len(trades) if trades else 0.0

    # Averages
    valid_days = [d for d in daily if not np.isnan(d.vrp)]
    avg_vrp = float(np.mean([d.vrp for d in valid_days])) if valid_days else 0.0
    avg_iv = float(np.mean([d.atm_iv for d in valid_days])) if valid_days else 0.0
    avg_rv = float(np.mean([d.rv_20d for d in valid_days])) if valid_days else 0.0

    return VRPBacktestResult(
        daily=daily,
        trades=trades,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return * 100,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        avg_vrp=avg_vrp,
        avg_iv=avg_iv,
        avg_rv=avg_rv,
    )


def format_vrp_results(result: VRPBacktestResult) -> str:
    """Format VRP backtest results for display."""
    lines = [
        "NIFTY Volatility Risk Premium — Backtest Results",
        "=" * 60,
        f"  Avg IV:           {result.avg_iv * 100:6.1f}%",
        f"  Avg RV:           {result.avg_rv * 100:6.1f}%",
        f"  Avg VRP:          {result.avg_vrp * 100:6.1f}% (IV - RV)",
        "",
        f"  Total return:     {result.total_return_pct:+.2f}%",
        f"  Annualized:       {result.annual_return_pct:+.2f}%",
        f"  Sharpe ratio:     {result.sharpe:.2f}",
        f"  Max drawdown:     {result.max_dd_pct:.2f}%",
        f"  Win rate:         {result.win_rate:.1%} ({len(result.trades)} trades)",
        "",
    ]

    if result.trades:
        lines.append("  Trades:")
        lines.append(f"  {'Entry':>12} {'Exit':>12} {'Days':>5} "
                      f"{'IV':>6} {'RV':>6} {'VRP':>6} "
                      f"{'Prem%':>7} {'Move%':>7} {'Net%':>8} {'Reason'}")
        lines.append("  " + "-" * 95)
        for t in result.trades:
            prem_pct = t.get("straddle_premium", 0) / t.get("spot_entry", 1) * 100
            move_pct = t.get("spot_move", 0) / t.get("spot_entry", 1) * 100
            lines.append(
                f"  {t['entry_date']:>12} {t['exit_date']:>12} {t['days_held']:5d} "
                f"{t['entry_iv'] * 100:5.1f}% {t['entry_rv'] * 100:5.1f}% "
                f"{t['entry_vrp'] * 100:5.1f}% "
                f"{prem_pct:6.2f}% {move_pct:6.2f}% "
                f"{t['pnl_pct'] * 100:+7.2f}% {t['exit_reason']}"
            )

    return "\n".join(lines)
