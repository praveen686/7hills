"""Historical backtester for the India institutional footprint strategy.

Execution model:
  - Signal generated at EOD on date T (after NSE daily data available ~6 PM IST)
  - Entry at T+1 open
  - Exit at T+hold_days close, or earlier on signal flip
  - Equal weight across top-N signals
  - Costs applied per the India cost model

Outputs: equity curve, Sharpe, max drawdown, win rate, NIFTY comparison.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from apps.india_scanner.data import get_equity, is_trading_day
from apps.india_scanner.costs import DEFAULT_COSTS, IndiaCostModel
from apps.india_scanner.scanner import run_daily_scan
from apps.india_scanner.signals import CompositeSignal
from qlx.data.store import MarketDataStore

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single backtest trade."""

    symbol: str
    direction: str          # "long" or "short"
    signal_date: str        # date signal was generated
    entry_date: str         # T+1
    exit_date: str          # T+1+hold
    entry_price: float
    exit_price: float
    gross_pnl_pct: float    # before costs
    cost_pct: float
    net_pnl_pct: float


@dataclass
class BacktestResult:
    """Full backtest output."""

    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame  # date → equity
    total_return_pct: float
    ann_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_hold_days: float
    avg_pnl_pct: float
    # Benchmark
    benchmark_return_pct: float
    benchmark_sharpe: float


def _get_next_trading_day(d: date) -> date:
    """Get the next trading day after d."""
    current = d + timedelta(days=1)
    attempts = 0
    while not is_trading_day(current) and attempts < 10:
        current += timedelta(days=1)
        attempts += 1
    return current


def _get_trading_day_offset(d: date, offset: int) -> date:
    """Get the trading day 'offset' days after d."""
    current = d
    counted = 0
    while counted < offset:
        current += timedelta(days=1)
        if is_trading_day(current):
            counted += 1
        if (current - d).days > offset * 3:
            break
    return current


def _get_price(equity_df: pd.DataFrame, symbol: str, field: str = "CLOSE") -> float | None:
    """Extract a price for a symbol from equity data."""
    if equity_df.empty or "SYMBOL" not in equity_df.columns:
        return None
    rows = equity_df[equity_df["SYMBOL"].str.strip() == symbol]
    if rows.empty:
        return None
    val = rows.iloc[0].get(field, None)
    if pd.isna(val):
        return None
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None


def run_backtest(
    start: date,
    end: date,
    hold_days: int = 3,
    top_n: int = 5,
    initial_equity: float = 100_000.0,
    cost_model: IndiaCostModel | None = None,
    store=None,
    symbols: list[str] | None = None,
) -> BacktestResult:
    """Run a historical backtest of the institutional footprint strategy.

    For each trading day in [start, end]:
      1. Run the daily scan to get signals
      2. Enter top-N at next day's open
      3. Exit after hold_days at close
      4. Track equity curve with costs
    """
    if store is None:
        store = MarketDataStore()
    if cost_model is None:
        cost_model = DEFAULT_COSTS

    equity = initial_equity
    trades: list[BacktestTrade] = []
    equity_points: list[tuple[str, float]] = [(start.isoformat(), equity)]

    # Active positions: {symbol: {entry_date, entry_price, direction, weight, exit_target_date}}
    active: dict[str, dict] = {}

    # Iterate through trading days
    current = start
    while current <= end:
        if not is_trading_day(current):
            current += timedelta(days=1)
            continue

        logger.debug("Backtest: processing %s", current)

        # Check exits first
        to_exit = []
        for sym, pos in active.items():
            if current >= date.fromisoformat(pos["exit_target_date"]):
                to_exit.append(sym)

        # Process exits
        try:
            eq_df = get_equity(store, current)
        except Exception:
            eq_df = pd.DataFrame()

        for sym in to_exit:
            pos = active.pop(sym)
            exit_price = _get_price(eq_df, sym, "CLOSE")
            if exit_price is None:
                # Can't get exit price — use entry price (flat)
                exit_price = pos["entry_price"]

            entry_price = pos["entry_price"]
            direction = pos["direction"]
            weight = pos["weight"]

            if direction == "long":
                gross = (exit_price - entry_price) / entry_price
            else:
                gross = (entry_price - exit_price) / entry_price

            cost_frac = cost_model.roundtrip_cost_frac(entry_price * weight * equity)
            net = gross - cost_frac

            # Apply to equity
            equity += equity * weight * net

            entry_d = pos["entry_date"]
            hold = (current - date.fromisoformat(entry_d)).days

            trades.append(BacktestTrade(
                symbol=sym,
                direction=direction,
                signal_date=pos["signal_date"],
                entry_date=entry_d,
                exit_date=current.isoformat(),
                entry_price=entry_price,
                exit_price=exit_price,
                gross_pnl_pct=gross * 100,
                cost_pct=cost_frac * 100,
                net_pnl_pct=net * 100,
            ))

        # Run daily scan for new signals
        try:
            signals = run_daily_scan(
                target_date=current, store=store, top_n=top_n, symbols=symbols,
            )
        except Exception as e:
            logger.debug("Scan failed for %s: %s", current, e)
            signals = []

        # Enter new positions (only if we have capacity)
        slots = top_n - len(active)
        if slots > 0 and signals:
            # Get next trading day for entry
            entry_date = _get_next_trading_day(current)
            exit_target = _get_trading_day_offset(entry_date, hold_days)

            try:
                entry_eq_df = get_equity(store, entry_date)
            except Exception:
                entry_eq_df = pd.DataFrame()

            weight = 1.0 / top_n  # equal weight

            for sig in signals[:slots]:
                if sig.symbol in active:
                    continue

                entry_price = _get_price(entry_eq_df, sig.symbol, "OPEN")
                if entry_price is None:
                    continue

                direction = "long" if sig.composite_score > 0 else "short"
                active[sig.symbol] = {
                    "entry_date": entry_date.isoformat(),
                    "entry_price": entry_price,
                    "direction": direction,
                    "weight": weight,
                    "signal_date": current.isoformat(),
                    "exit_target_date": exit_target.isoformat(),
                }

        equity_points.append((current.isoformat(), equity))
        current += timedelta(days=1)

    # Force-close remaining positions at last available price
    for sym, pos in list(active.items()):
        trades.append(BacktestTrade(
            symbol=sym,
            direction=pos["direction"],
            signal_date=pos["signal_date"],
            entry_date=pos["entry_date"],
            exit_date=end.isoformat(),
            entry_price=pos["entry_price"],
            exit_price=pos["entry_price"],  # flat if no exit price
            gross_pnl_pct=0.0,
            cost_pct=0.0,
            net_pnl_pct=0.0,
        ))

    # Build equity curve DataFrame
    eq_df = pd.DataFrame(equity_points, columns=["date", "equity"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")

    # Compute summary stats
    total_ret = (equity / initial_equity - 1) * 100
    days = (end - start).days
    ann_ret = ((equity / initial_equity) ** (365 / max(days, 1)) - 1) * 100

    # Sharpe from equity curve
    returns = eq_df["equity"].pct_change().dropna()
    if len(returns) >= 2 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = eq_df["equity"].cummax()
    dd = (running_max - eq_df["equity"]) / running_max
    max_dd = dd.max() * 100

    # Win rate
    n_trades = len(trades)
    winners = sum(1 for t in trades if t.net_pnl_pct > 0)
    win_rate = winners / n_trades if n_trades > 0 else 0.0
    avg_pnl = sum(t.net_pnl_pct for t in trades) / n_trades if n_trades > 0 else 0.0
    avg_hold = (
        sum(
            (date.fromisoformat(t.exit_date) - date.fromisoformat(t.entry_date)).days
            for t in trades
        ) / n_trades
        if n_trades > 0 else 0.0
    )

    # Benchmark: NIFTY buy-and-hold (use first/last equity bhav if available)
    bench_ret = 0.0
    bench_sharpe = 0.0
    try:
        start_eq = get_equity(store, start)
        end_eq = get_equity(store, end)
        nifty_start = _get_price(start_eq, "NIFTY 50", "CLOSE")
        nifty_end = _get_price(end_eq, "NIFTY 50", "CLOSE")
        if nifty_start and nifty_end and nifty_start > 0:
            bench_ret = (nifty_end / nifty_start - 1) * 100
    except Exception:
        pass

    return BacktestResult(
        trades=trades,
        equity_curve=eq_df,
        total_return_pct=total_ret,
        ann_return_pct=ann_ret,
        sharpe=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        total_trades=n_trades,
        avg_hold_days=avg_hold,
        avg_pnl_pct=avg_pnl,
        benchmark_return_pct=bench_ret,
        benchmark_sharpe=bench_sharpe,
    )


def format_backtest_results(result: BacktestResult) -> str:
    """Format backtest results as a human-readable report."""
    lines = [
        "India Institutional Footprint — Backtest Results",
        "=" * 60,
        f"  Total return:      {result.total_return_pct:+.2f}%",
        f"  Annualized return: {result.ann_return_pct:+.2f}%",
        f"  Sharpe ratio:      {result.sharpe:.2f}",
        f"  Max drawdown:      {result.max_drawdown_pct:.2f}%",
        f"  Win rate:          {result.win_rate:.1%} ({result.total_trades} trades)",
        f"  Avg P&L per trade: {result.avg_pnl_pct:+.2f}%",
        f"  Avg holding period:{result.avg_hold_days:.1f} days",
        f"  Benchmark return:  {result.benchmark_return_pct:+.2f}% (NIFTY 50)",
        "",
        "Recent Trades:",
        f"  {'Symbol':15s} {'Dir':>5s} {'Entry':>10s} {'Exit':>10s} "
        f"{'Gross%':>7s} {'Cost%':>6s} {'Net%':>7s}",
        f"  {'-'*65}",
    ]

    for t in result.trades[-20:]:
        lines.append(
            f"  {t.symbol:15s} {t.direction:>5s} {t.entry_date:>10s} "
            f"{t.exit_date:>10s} {t.gross_pnl_pct:+7.2f} "
            f"{t.cost_pct:6.2f} {t.net_pnl_pct:+7.2f}"
        )

    return "\n".join(lines)
