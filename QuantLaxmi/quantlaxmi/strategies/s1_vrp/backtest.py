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

from quantlaxmi.core.pricing.iv_engine import (
    compute_chain_iv,
    har_rv,
    realized_vol,
    OptionChainIV,
)
from quantlaxmi.strategies.s9_momentum.data import is_trading_day, get_fno, get_delivery
from quantlaxmi.data.store import MarketDataStore

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
            except Exception as e:
                logger.debug("Spot price lookup failed on %s: %s", d, e)
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
    cost_model=None,
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

    # --- Trading simulation: event-driven straddle selling ---
    # Causal: entry at bar i, exit resolved only when clock reaches exit bar.
    trades: list[dict] = []
    daily_pnl: list[float] = []
    hold_days = 5  # hold for ~1 week (5 trading days)
    # Use CostModel if provided, otherwise default 30 bps round-trip
    # (STT + brokerage + charges for India options)
    if cost_model is not None:
        cost_frac = cost_model.roundtrip_frac
    else:
        from quantlaxmi.core.backtest.costs import CostModel
        cost_model = CostModel(commission_bps=10, slippage_bps=5)
        cost_frac = cost_model.roundtrip_frac

    # Track open trade state
    open_trade: dict | None = None
    open_entry_idx: int = -1

    for i in range(len(daily)):
        obs = daily[i]

        # Check if we have an open trade that should exit
        if open_trade is not None:
            bars_held = i - open_entry_idx
            if bars_held >= hold_days or i == len(daily) - 1:
                # Exit: resolve P&L using current bar (causal)
                entry = open_trade
                spot_move = abs(obs.spot - entry["spot_entry"])
                premium_pct = entry["straddle_premium"] / entry["spot_entry"]
                payoff_pct = spot_move / entry["spot_entry"]
                trade_pnl = premium_pct - payoff_pct - cost_frac

                trades.append({
                    "entry_date": entry["entry_date"],
                    "exit_date": obs.date.isoformat(),
                    "entry_iv": entry["entry_iv"],
                    "entry_rv": entry["entry_rv"],
                    "entry_vrp": entry["entry_vrp"],
                    "exit_iv": obs.atm_iv,
                    "exit_rv": obs.rv_20d,
                    "exit_vrp": obs.vrp if not np.isnan(obs.vrp) else 0.0,
                    "spot_entry": entry["spot_entry"],
                    "spot_exit": obs.spot,
                    "straddle_premium": entry["straddle_premium"],
                    "spot_move": spot_move,
                    "pnl_pct": trade_pnl,
                    "exit_reason": "hold_period" if i < len(daily) - 1 else "end_of_backtest",
                    "days_held": (obs.date - date.fromisoformat(entry["entry_date"])).days,
                })

                # Daily P&L on exit day = full trade P&L minus what was already accrued
                daily_pnl.append(trade_pnl - entry.get("accrued_pnl", 0.0))
                open_trade = None
                continue
            else:
                # Still in trade: accrue daily P&L as mark-to-market
                # P&L so far = premium - |spot move so far| (no cost until exit)
                spot_move_so_far = abs(obs.spot - open_trade["spot_entry"])
                premium_pct = open_trade["straddle_premium"] / open_trade["spot_entry"]
                payoff_so_far = spot_move_so_far / open_trade["spot_entry"]
                mtm_pnl = premium_pct - payoff_so_far
                prev_mtm = open_trade.get("prev_mtm", premium_pct)
                daily_pnl.append(mtm_pnl - prev_mtm)
                open_trade["prev_mtm"] = mtm_pnl
                open_trade["accrued_pnl"] = mtm_pnl - premium_pct + (premium_pct - cost_frac)
                continue

        if np.isnan(obs.vrp):
            daily_pnl.append(0.0)
            continue

        # Entry condition (only when no open trade)
        if obs.vrp > entry_vrp and obs.straddle_price > 0:
            open_trade = {
                "entry_date": obs.date.isoformat(),
                "entry_iv": obs.atm_iv,
                "entry_rv": obs.rv_20d,
                "entry_vrp": obs.vrp,
                "spot_entry": obs.spot,
                "straddle_premium": obs.straddle_price,
                "prev_mtm": obs.straddle_price / obs.spot,
                "accrued_pnl": 0.0,
            }
            open_entry_idx = i
            daily_pnl.append(0.0)  # no P&L on entry day
        else:
            daily_pnl.append(0.0)

    # --- Performance metrics ---
    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    years = max(len(daily_pnl) / 252, 1 / 252)
    annual_return = total_return / years

    # Sharpe: use ALL daily returns (including zero/flat days) with ddof=1
    if len(pnl_arr) > 1 and np.std(pnl_arr, ddof=1) > 0:
        sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr, ddof=1) * np.sqrt(252))
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
