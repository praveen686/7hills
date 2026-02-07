"""IV Mean-Reversion Strategy for Indian Index Futures.

Signal:  SANOS ATM IV spikes above a rolling percentile threshold.
         High IV = fear = prices tend to bounce → go long futures.
Action:  Long index futures when ATM IV > rolling 80th percentile.
Exit:    After hold_days trading days, or if IV drops below rolling median.
Cost:    Index futures round-trip ~5bps (lower than options).

Supports: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, NIFTYNXT50

Can optionally gate entries with information-theoretic regime filters
(high entropy = random walk regime where mean-reversion works).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from qlx.pricing.sanos import fit_sanos, prepare_nifty_chain, SANOSResult
from strategies.s9_momentum.data import is_trading_day, get_fno
from qlx.data.store import MarketDataStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Supported index symbols
SUPPORTED_INDICES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTYNXT50"]


@dataclass
class DayObs:
    """Single day observation."""
    date: date
    spot: float
    atm_iv: float           # annualized ATM implied vol from SANOS
    atm_var: float           # ATM total variance (σ²T nearest expiry)
    forward: float
    sanos_ok: bool
    symbol: str = "NIFTY"    # which index this observation is for


@dataclass
class Trade:
    entry_date: date
    exit_date: date
    entry_spot: float
    exit_spot: float
    entry_iv: float
    exit_iv: float
    iv_pctile: float         # IV percentile rank at entry
    pnl_pct: float
    hold_days: int
    exit_reason: str
    symbol: str = "NIFTY"    # which index this trade is for


@dataclass
class BacktestResult:
    """Full backtest output."""
    daily: list[DayObs]
    trades: list[Trade]
    total_return_pct: float
    annual_return_pct: float
    sharpe: float
    max_dd_pct: float
    win_rate: float
    avg_iv: float
    n_signals: int


# ---------------------------------------------------------------------------
# Build daily ATM IV series from SANOS calibration
# ---------------------------------------------------------------------------

def build_iv_series(
    store,
    start: date,
    end: date,
    symbol: str = "NIFTY",
) -> list[DayObs]:
    """Calibrate SANOS each day and extract ATM IV for a given index.

    Args:
        store: MarketDataStore instance
        start: Start date
        end: End date
        symbol: Index symbol (NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, NIFTYNXT50)
    """
    results = []
    d = start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        try:
            fno = get_fno(store, d)
            if fno.empty:
                d += timedelta(days=1)
                continue
        except Exception:
            d += timedelta(days=1)
            continue

        chain_data = prepare_nifty_chain(fno, symbol=symbol, max_expiries=2)
        if chain_data is None:
            d += timedelta(days=1)
            continue

        spot = chain_data["spot"]
        forward = chain_data["forward"]
        atm_vars = chain_data["atm_variances"]

        # Try SANOS calibration
        sanos_ok = False
        atm_iv = math.sqrt(atm_vars[0])  # fallback: Brenner estimate
        try:
            result = fit_sanos(
                market_strikes=chain_data["market_strikes"],
                market_calls=chain_data["market_calls"],
                market_spreads=chain_data["market_spreads"],
                atm_variances=atm_vars,
                expiry_labels=chain_data["expiry_labels"],
                eta=0.50,
                n_model_strikes=100,
                K_min=0.7,
                K_max=1.5,
            )
            if result.lp_success:
                sanos_ok = True
                # Extract ATM IV from the surface (nearest expiry, K/F = 1.0)
                atm_strike = np.array([1.0])
                # Estimate T from expiry label
                exp_str = result.expiry_labels[0]
                try:
                    from datetime import datetime
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    T = max((exp_dt - d).days / 365.0, 1 / 365.0)
                except Exception:
                    T = atm_vars[0] / max(atm_iv ** 2, 1e-6) if atm_iv > 0 else 7 / 365.0
                iv_arr = result.iv(0, atm_strike, T)
                atm_iv = float(iv_arr[0])
        except Exception as e:
            logger.debug("SANOS failed for %s %s: %s", symbol, d, e)

        results.append(DayObs(
            date=d,
            spot=spot,
            atm_iv=atm_iv,
            atm_var=float(atm_vars[0]),
            forward=forward,
            sanos_ok=sanos_ok,
            symbol=symbol,
        ))

        if len(results) % 10 == 0:
            logger.info("%s IV series: %d days (latest %s, IV=%.1f%%)",
                        symbol, len(results), d, atm_iv * 100)

        d += timedelta(days=1)

    return results


# ---------------------------------------------------------------------------
# Compute rolling percentile of ATM IV
# ---------------------------------------------------------------------------

def rolling_percentile(values: list[float], window: int) -> list[float]:
    """Compute rolling percentile rank of current value within window."""
    ranks = []
    for i, v in enumerate(values):
        lookback = values[max(0, i - window + 1):i + 1]
        rank = sum(1 for x in lookback if x <= v) / len(lookback)
        ranks.append(rank)
    return ranks


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_from_series(
    daily: list[DayObs],
    iv_lookback: int = 60,
    entry_pctile: float = 0.80,
    exit_pctile: float = 0.50,
    hold_days: int = 7,
    cost_bps: float = 5,
    entropy_filter: np.ndarray | None = None,
    entropy_threshold: float = 0.0,
    symbol: str = "NIFTY",
) -> BacktestResult:
    """Run strategy on a pre-built IV series (avoids re-calibrating SANOS)."""
    if len(daily) < iv_lookback + 1:
        return BacktestResult(
            daily=daily, trades=[], total_return_pct=0, annual_return_pct=0,
            sharpe=0, max_dd_pct=0, win_rate=0, avg_iv=0, n_signals=0,
        )

    # Compute rolling IV percentile ranks
    ivs = [d.atm_iv for d in daily]
    pctile_ranks = rolling_percentile(ivs, iv_lookback)

    # --- Trading simulation ---
    trades: list[Trade] = []
    daily_pnl: list[float] = []
    cost_frac = cost_bps / 10_000
    n_signals = 0

    in_trade = False
    entry_idx = 0
    pending_entry = False  # T+1 execution: signal at close, enter next day

    for i in range(iv_lookback, len(daily)):
        obs = daily[i]

        # Execute pending entry from yesterday's signal
        if pending_entry and not in_trade:
            in_trade = True
            entry_idx = i
            pending_entry = False
            daily_pnl.append(0.0)  # no P&L on entry day
            continue

        if not in_trade:
            # Check entry conditions (signal fires, execution deferred to T+1)
            if pctile_ranks[i] >= entry_pctile:
                # Optional entropy gate
                if entropy_filter is not None:
                    if i < len(entropy_filter) and entropy_filter[i] < entropy_threshold:
                        daily_pnl.append(0.0)
                        continue

                n_signals += 1
                pending_entry = True  # enter at next bar (T+1)
                daily_pnl.append(0.0)
            else:
                daily_pnl.append(0.0)
        else:
            # In a trade — compute daily P&L
            days_held = i - entry_idx
            spot_prev = daily[i - 1].spot
            spot_now = obs.spot
            day_return = (spot_now - spot_prev) / daily[entry_idx].spot

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            if days_held >= hold_days:
                should_exit = True
                exit_reason = "max_hold"
            elif pctile_ranks[i] < exit_pctile:
                should_exit = True
                exit_reason = "iv_normalised"

            if should_exit or i == len(daily) - 1:
                if i == len(daily) - 1 and not should_exit:
                    exit_reason = "end_of_backtest"
                # Compute total trade P&L
                total_pnl = (obs.spot - daily[entry_idx].spot) / daily[entry_idx].spot - cost_frac
                trades.append(Trade(
                    entry_date=daily[entry_idx].date,
                    exit_date=obs.date,
                    entry_spot=daily[entry_idx].spot,
                    exit_spot=obs.spot,
                    entry_iv=daily[entry_idx].atm_iv,
                    exit_iv=obs.atm_iv,
                    iv_pctile=pctile_ranks[entry_idx],
                    pnl_pct=total_pnl,
                    hold_days=days_held,
                    exit_reason=exit_reason,
                    symbol=symbol,
                ))
                in_trade = False
                daily_pnl.append(day_return)
            else:
                daily_pnl.append(day_return)

    # --- Performance metrics ---
    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    years = max(len(daily_pnl) / 252, 1 / 252)
    annual_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    # Sharpe from ALL daily returns (including zero-trade days) with ddof=1
    if len(pnl_arr) > 1:
        std = np.std(pnl_arr, ddof=1)
        if std > 0:
            sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    if len(cumulative) > 0:
        peak = np.maximum.accumulate(cumulative)
        dd = cumulative - peak
        max_dd = float(abs(dd.min()))
    else:
        max_dd = 0.0

    # Win rate
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / len(trades) if trades else 0.0

    # Avg IV
    avg_iv = float(np.mean(ivs)) if ivs else 0.0

    return BacktestResult(
        daily=daily,
        trades=trades,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return * 100,
        sharpe=sharpe,
        max_dd_pct=max_dd * 100,
        win_rate=win_rate,
        avg_iv=avg_iv,
        n_signals=n_signals,
    )


def run_iv_mean_revert(
    store,
    start: date,
    end: date,
    iv_lookback: int = 60,
    entry_pctile: float = 0.80,
    exit_pctile: float = 0.50,
    hold_days: int = 7,
    cost_bps: float = 5,
    entropy_filter: np.ndarray | None = None,
    entropy_threshold: float = 0.0,
    symbol: str = "NIFTY",
) -> BacktestResult:
    """Build IV series from SANOS, then run mean-reversion backtest.

    Args:
        symbol: Index symbol (NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, NIFTYNXT50)
    """
    logger.info("Building %s IV series %s to %s...", symbol, start, end)
    daily = build_iv_series(store, start, end, symbol=symbol)
    return run_from_series(
        daily, iv_lookback, entry_pctile, exit_pctile,
        hold_days, cost_bps, entropy_filter, entropy_threshold, symbol=symbol,
    )


def run_multi_index_backtest(
    store,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    iv_lookback: int = 30,
    entry_pctile: float = 0.80,
    exit_pctile: float = 0.50,
    hold_days: int = 5,
    cost_bps: float = 5,
) -> dict[str, BacktestResult]:
    """Run IV mean-reversion backtest across multiple indices.

    Args:
        symbols: List of indices to backtest. Default: all supported.

    Returns:
        Dict mapping symbol -> BacktestResult
    """
    if symbols is None:
        symbols = SUPPORTED_INDICES

    results = {}
    for sym in symbols:
        logger.info("=" * 60)
        logger.info("Backtesting %s...", sym)
        try:
            result = run_iv_mean_revert(
                store, start, end,
                iv_lookback=iv_lookback,
                entry_pctile=entry_pctile,
                exit_pctile=exit_pctile,
                hold_days=hold_days,
                cost_bps=cost_bps,
                symbol=sym,
            )
            results[sym] = result
            logger.info("%s: %d trades, %.2f Sharpe, %.1f%% return",
                        sym, len(result.trades), result.sharpe, result.total_return_pct)
        except Exception as e:
            logger.error("Failed to backtest %s: %s", sym, e)

    return results


def format_multi_index_results(results: dict[str, BacktestResult]) -> str:
    """Format multi-index backtest results as a comparison table."""
    lines = [
        "Multi-Index IV Mean-Reversion — Backtest Comparison",
        "=" * 80,
        "",
        f"  {'Symbol':<12} {'Days':>5} {'Signals':>8} {'Trades':>7} "
        f"{'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'WinRate':>8}",
        "  " + "-" * 75,
    ]

    total_trades = 0
    combined_returns = []

    for sym, r in sorted(results.items()):
        lines.append(
            f"  {sym:<12} {len(r.daily):5d} {r.n_signals:8d} {len(r.trades):7d} "
            f"{r.total_return_pct:+7.2f}% {r.sharpe:7.2f} {r.max_dd_pct:6.2f}% "
            f"{r.win_rate:7.1%}"
        )
        total_trades += len(r.trades)
        combined_returns.append(r.total_return_pct)

    lines.append("  " + "-" * 75)
    avg_return = np.mean(combined_returns) if combined_returns else 0
    lines.append(f"  {'TOTAL':<12} {'-':>5} {'-':>8} {total_trades:7d} "
                 f"{avg_return:+7.2f}% (avg)")
    lines.append("")

    return "\n".join(lines)


def format_iv_results(result: BacktestResult, symbol: str = "NIFTY") -> str:
    """Format IV mean-reversion backtest results for display."""
    lines = [
        f"{symbol} IV Mean-Reversion — Backtest Results",
        "=" * 65,
        f"  Days analysed:    {len(result.daily)}",
        f"  Avg ATM IV:       {result.avg_iv * 100:6.1f}%",
        f"  Signals fired:    {result.n_signals}",
        f"  Trades taken:     {len(result.trades)}",
        "",
        f"  Total return:     {result.total_return_pct:+.2f}%",
        f"  Annualized:       {result.annual_return_pct:+.2f}%",
        f"  Sharpe ratio:     {result.sharpe:.2f}",
        f"  Max drawdown:     {result.max_dd_pct:.2f}%",
        f"  Win rate:         {result.win_rate:.1%}",
        "",
    ]

    if result.trades:
        lines.append(f"  {'Entry':>12} {'Exit':>12} {'Days':>5} "
                      f"{'EntIV':>6} {'ExIV':>6} {'Pctl':>5} "
                      f"{'P&L':>8} {'Reason'}")
        lines.append("  " + "-" * 80)
        for t in result.trades:
            lines.append(
                f"  {t.entry_date.isoformat():>12} {t.exit_date.isoformat():>12} "
                f"{t.hold_days:5d} "
                f"{t.entry_iv * 100:5.1f}% {t.exit_iv * 100:5.1f}% "
                f"{t.iv_pctile:.2f} "
                f"{t.pnl_pct * 100:+7.2f}% {t.exit_reason}"
            )

    return "\n".join(lines)
