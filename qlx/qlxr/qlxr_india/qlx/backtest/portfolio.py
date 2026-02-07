"""Portfolio backtesting with mandatory cost model.

This module provides a clean, from-scratch vectorised backtester.  It does
NOT depend on vectorbtpro for the core logic — it uses numpy for the
simulation loop and only optionally wraps vbt for richer analytics.

The key invariant: ``run_backtest`` will not execute without a ``CostModel``.
There is no way to accidentally run a zero-cost backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qlx.backtest.costs import CostModel


@dataclass(frozen=True)
class BacktestResult:
    """Immutable backtest output."""

    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame  # columns: entry_time, exit_time, side, entry_px, exit_px, pnl, cost
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    total_costs: float
    cost_model: CostModel

    def summary(self) -> str:
        return (
            f"--- Backtest (costs: {self.cost_model.roundtrip_bps:.1f} bps RT) ---\n"
            f"Total return:   {self.total_return:+.2%}\n"
            f"Sharpe ratio:   {self.sharpe_ratio:.3f}\n"
            f"Sortino ratio:  {self.sortino_ratio:.3f}\n"
            f"Max drawdown:   {self.max_drawdown:.2%}\n"
            f"Trades:         {self.n_trades}\n"
            f"Win rate:       {self.win_rate:.1%}\n"
            f"Profit factor:  {self.profit_factor:.2f}\n"
            f"Avg trade PnL:  {self.avg_trade_pnl:+.4%}\n"
            f"Total costs:    {self.total_costs:+.4%}"
        )


def run_backtest(
    prices: pd.Series,
    predictions: pd.Series,
    cost_model: CostModel,
    long_entry_th: float = 0.01,
    long_exit_th: float = 0.0,
    short_entry_th: float = -0.01,
    short_exit_th: float = 0.0,
    initial_capital: float = 1.0,
    execution_delay: int = 1,
) -> BacktestResult:
    """Run a vectorised long/short backtest with transaction costs.

    Parameters
    ----------
    prices : pd.Series
        Close prices aligned to predictions.
    predictions : pd.Series
        Model predictions (same index as prices).
    cost_model : CostModel
        Transaction cost model.  Required.
    long_entry_th : float
        Enter long when prediction > this.
    long_exit_th : float
        Exit long when prediction < this.
    short_entry_th : float
        Enter short when prediction < this.
    short_exit_th : float
        Exit short when prediction > this.
    execution_delay : int
        Number of bars to delay execution (default 1 = T+1).
        Signal at bar i becomes position at bar i + execution_delay.
    """
    assert prices.index.equals(predictions.index), "prices and predictions must share index"

    px = prices.values.astype(np.float64)
    pred = predictions.values.astype(np.float64)
    n = len(px)

    # First compute desired positions from predictions (no delay)
    desired_position = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        prev_des = desired_position[i - 1]

        if prev_des == 0:
            if pred[i] > long_entry_th:
                desired_position[i] = 1
            elif pred[i] < short_entry_th:
                desired_position[i] = -1
            else:
                desired_position[i] = 0
        elif prev_des == 1:
            if pred[i] < long_exit_th:
                desired_position[i] = 0
            else:
                desired_position[i] = 1
        elif prev_des == -1:
            if pred[i] > short_exit_th:
                desired_position[i] = 0
            else:
                desired_position[i] = -1

    # Apply execution delay: shift desired positions forward
    position = np.zeros(n, dtype=np.int8)
    if execution_delay > 0:
        position[execution_delay:] = desired_position[:-execution_delay]
    else:
        position = desired_position

    # Compute costs from position changes
    costs_arr = np.zeros(n, dtype=np.float64)
    one_way = cost_model.one_way_frac
    for i in range(1, n):
        if position[i] != position[i - 1]:
            costs_arr[i] = one_way

    # --- compute returns ---
    price_returns = np.diff(px) / px[:-1]
    # Position at bar i earns the return from bar i to bar i+1
    pos_returns = position[:-1].astype(np.float64) * price_returns - costs_arr[1:]

    equity = initial_capital * np.cumprod(1 + pos_returns)
    equity = np.insert(equity, 0, initial_capital)

    # --- extract trades ---
    trades = _extract_trades(position, px, prices.index, costs_arr)

    # --- compute metrics ---
    ret_series = pd.Series(pos_returns, index=prices.index[1:])
    eq_series = pd.Series(equity, index=prices.index)

    total_return = equity[-1] / equity[0] - 1.0
    total_costs = float(np.sum(costs_arr))

    sharpe = _sharpe(ret_series)
    sortino = _sortino(ret_series)
    max_dd = _max_drawdown(eq_series)

    if len(trades) > 0:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / len(trades)
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_pnl = trades["pnl"].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_pnl = 0.0

    return BacktestResult(
        equity_curve=eq_series,
        returns=ret_series,
        trades=trades,
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        n_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_pnl,
        total_costs=total_costs,
        cost_model=cost_model,
    )


def _extract_trades(
    position: np.ndarray,
    prices: np.ndarray,
    index: pd.DatetimeIndex,
    costs: np.ndarray,
) -> pd.DataFrame:
    """Parse position array into a trade ledger."""
    trades = []
    entry_idx = None
    entry_px = 0.0
    side = 0
    trade_cost = 0.0

    for i in range(len(position)):
        if entry_idx is None and position[i] != 0:
            # Opening a new position
            entry_idx = i
            entry_px = prices[i]
            side = position[i]
            trade_cost = costs[i]

        elif entry_idx is not None and position[i] != side:
            # Closing the position (or flipping)
            exit_px = prices[i]
            trade_cost += costs[i]
            pnl = side * (exit_px / entry_px - 1.0) - trade_cost

            trades.append(
                {
                    "entry_time": index[entry_idx],
                    "exit_time": index[i],
                    "side": "long" if side == 1 else "short",
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "pnl": pnl,
                    "cost": trade_cost,
                    "bars_held": i - entry_idx,
                }
            )

            # If flipping to opposite side, immediately open new trade
            if position[i] != 0 and position[i] != side:
                entry_idx = i
                entry_px = prices[i]
                side = position[i]
                trade_cost = costs[i]
            else:
                entry_idx = None
                trade_cost = 0.0

    # Close any open trade at the end
    if entry_idx is not None:
        exit_px = prices[-1]
        pnl = side * (exit_px / entry_px - 1.0) - trade_cost
        trades.append(
            {
                "entry_time": index[entry_idx],
                "exit_time": index[-1],
                "side": "long" if side == 1 else "short",
                "entry_px": entry_px,
                "exit_px": exit_px,
                "pnl": pnl,
                "cost": trade_cost,
                "bars_held": len(position) - 1 - entry_idx,
            }
        )

    if not trades:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "side", "entry_px", "exit_px", "pnl", "cost", "bars_held"]
        )

    return pd.DataFrame(trades)


def _sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())
