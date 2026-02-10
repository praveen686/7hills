"""Multi-Strategy Portfolio Backtest.

Runs all strategies simultaneously on shared capital. For each trading day:
  1. Each strategy emits signals
  2. Meta-allocator sizes and filters
  3. Risk manager checks gates
  4. Positions updated with costs
  5. Mark-to-market equity

Returns: combined equity curve, per-strategy attribution, drawdown series.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantlaxmi.core.allocator.meta import MetaAllocator
from quantlaxmi.core.allocator.regime import detect_regime
from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.core.risk.limits import RiskLimits
from quantlaxmi.core.risk.manager import PortfolioState as RiskPortfolioState, RiskManager
from quantlaxmi.strategies.protocol import Signal, StrategyProtocol

logger = logging.getLogger(__name__)


@dataclass
class StrategyAttribution:
    """Per-strategy performance attribution."""

    strategy_id: str
    signals_count: int = 0
    trades_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    daily_returns: list[float] = field(default_factory=list)


@dataclass
class PortfolioBacktestResult:
    """Multi-strategy backtest output."""

    # Equity curves
    dates: list[date] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    drawdown_series: list[float] = field(default_factory=list)

    # Per-strategy attribution
    strategy_attr: dict[str, StrategyAttribution] = field(default_factory=dict)

    # Aggregate metrics
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_signals: int = 0

    # Risk metrics
    avg_exposure: float = 0.0
    max_exposure: float = 0.0
    days_in_market: int = 0
    circuit_breaker_triggers: int = 0


def run_portfolio_backtest(
    store: MarketDataStore,
    strategies: list[StrategyProtocol],
    start: date,
    end: date,
    allocator: MetaAllocator | None = None,
    risk_manager: RiskManager | None = None,
    cost_bps: float = 5.0,
    initial_capital: float = 1.0,
) -> PortfolioBacktestResult:
    """Run multi-strategy portfolio backtest.

    Parameters
    ----------
    store : MarketDataStore
        Market data access.
    strategies : list[StrategyProtocol]
        All strategies to run.
    start, end : date
        Backtest date range.
    allocator : MetaAllocator
        Portfolio allocator (defaults to standard VIX-regime).
    risk_manager : RiskManager
        Risk gate system.
    cost_bps : float
        Round-trip cost in basis points.
    initial_capital : float
        Starting equity (1.0 = normalized).
    """
    if allocator is None:
        allocator = MetaAllocator()
    if risk_manager is None:
        risk_manager = RiskManager()

    cost_frac = cost_bps / 10_000
    result = PortfolioBacktestResult()

    # State tracking
    equity = initial_capital
    peak_equity = initial_capital
    positions: dict[str, dict] = {}  # "strategy:symbol" → position info
    strategy_equity: dict[str, float] = {s.strategy_id: 0.0 for s in strategies}
    strategy_peaks: dict[str, float] = {s.strategy_id: 0.0 for s in strategies}
    daily_returns: list[float] = []

    # Strategy attribution
    for s in strategies:
        result.strategy_attr[s.strategy_id] = StrategyAttribution(strategy_id=s.strategy_id)

    try:
        from quantlaxmi.strategies.s9_momentum.data import is_trading_day
    except ImportError:
        def is_trading_day(d):
            return d.weekday() < 5

    d = start
    while d <= end:
        if not is_trading_day(d):
            d += timedelta(days=1)
            continue

        # 1. Detect regime
        regime = detect_regime(store, d)

        # 2. Collect signals
        all_signals: list[Signal] = []
        for strategy in strategies:
            try:
                signals = strategy.scan(d, store)
                all_signals.extend(signals)
                attr = result.strategy_attr[strategy.strategy_id]
                attr.signals_count += len(signals)
            except Exception:
                continue

        result.total_signals += len(all_signals)

        # 3. Allocate
        targets = allocator.allocate(all_signals, regime)

        # 4. Risk check
        risk_state = RiskPortfolioState(
            equity=equity,
            peak_equity=peak_equity,
            positions={
                sym: {"direction": p["direction"], "weight": p["weight"], "strategy_id": p["strategy_id"]}
                for sym, p in positions.items()
            },
            strategy_equity=strategy_equity.copy(),
            strategy_peaks=strategy_peaks.copy(),
        )
        risk_results = risk_manager.check(targets, risk_state)
        approved = [r for r in risk_results if r.approved]

        # 5. Execute
        day_pnl = 0.0
        exposure = sum(abs(p["weight"]) for p in positions.values())

        for risk_result in approved:
            target = risk_result.target

            key = f"{target.strategy_id}:{target.symbol}"

            if target.direction == "flat":
                # Close position
                pos = positions.pop(key, None)
                if pos is not None:
                    # Simulate exit P&L (using daily close as proxy)
                    spot = _get_spot(store, target.symbol, d)
                    if spot > 0 and pos.get("entry_price", 0) > 0:
                        raw_pnl = (spot - pos["entry_price"]) / pos["entry_price"]
                        if pos["direction"] == "short":
                            raw_pnl = -raw_pnl
                        net_pnl = (raw_pnl - cost_frac) * pos["weight"]
                        day_pnl += net_pnl
                        strategy_equity[target.strategy_id] = strategy_equity.get(target.strategy_id, 0) + net_pnl

                        attr = result.strategy_attr[target.strategy_id]
                        attr.trades_count += 1
                        attr.gross_pnl += raw_pnl * pos["weight"]
                        attr.net_pnl += net_pnl
                continue

            if key in positions:
                continue  # already in position

            weight = risk_result.adjusted_weight
            if weight <= 0:
                continue

            spot = _get_spot(store, target.symbol, d)
            if spot <= 0:
                continue

            positions[key] = {
                "strategy_id": target.strategy_id,
                "symbol": target.symbol,
                "direction": target.direction,
                "weight": weight,
                "entry_price": spot,
                "entry_date": d.isoformat(),
            }

        # Mark-to-market existing positions
        mtm_pnl = 0.0
        for key, pos in list(positions.items()):
            spot = _get_spot(store, pos["symbol"], d)
            if spot > 0 and pos.get("entry_price", 0) > 0:
                raw_ret = (spot - pos["entry_price"]) / pos["entry_price"]
                if pos["direction"] == "short":
                    raw_ret = -raw_ret
                mtm_pnl += raw_ret * pos["weight"]

        # Update equity
        equity = initial_capital + mtm_pnl + sum(
            attr.net_pnl for attr in result.strategy_attr.values()
        )
        if equity > peak_equity:
            peak_equity = equity

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        day_ret = day_pnl  # simplified
        daily_returns.append(day_ret)

        result.dates.append(d)
        result.equity_curve.append(equity)
        result.drawdown_series.append(dd)

        if exposure > 0:
            result.days_in_market += 1
        result.avg_exposure += exposure
        result.max_exposure = max(result.max_exposure, exposure)

        d += timedelta(days=1)

    # Compute final metrics
    n_days = len(daily_returns)
    if n_days > 0:
        result.avg_exposure /= n_days
        result.total_return_pct = (equity / initial_capital - 1) * 100

        years = max(n_days / 252, 1 / 252)
        result.annual_return_pct = ((equity / initial_capital) ** (1 / years) - 1) * 100

        arr = np.array(daily_returns)
        if len(arr) > 1 and np.std(arr, ddof=1) > 0:
            result.sharpe = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))
            downside = arr[arr < 0]
            if len(downside) > 0:
                result.sortino = float(np.mean(arr) / np.std(downside, ddof=1) * np.sqrt(252))

        result.max_drawdown_pct = max(result.drawdown_series) * 100 if result.drawdown_series else 0

        # Per-strategy Sharpe
        for attr in result.strategy_attr.values():
            if attr.daily_returns and len(attr.daily_returns) > 1:
                arr_s = np.array(attr.daily_returns)
                std = np.std(arr_s, ddof=1)
                if std > 0:
                    attr.sharpe = float(np.mean(arr_s) / std * np.sqrt(252))

        result.total_trades = sum(a.trades_count for a in result.strategy_attr.values())

    return result


def _get_spot(store: MarketDataStore, symbol: str, d: date) -> float:
    """Get closing price for a symbol on a date."""
    d_str = d.isoformat()
    try:
        _idx_name = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}.get(
            symbol.upper(), f"Nifty {symbol}")
        df = store.sql(
            'SELECT "Closing Index Value" as close FROM nse_index_close '
            'WHERE date = ? AND "Index Name" = ? LIMIT 1',
            [d_str, _idx_name],
        )
        if not df.empty:
            return float(df["close"].iloc[0])
    except Exception:
        pass

    try:
        df = store.sql(
            "SELECT \"ClsPric\" as close FROM nse_fo_bhavcopy "
            "WHERE date = ? AND \"TckrSymb\" = ? AND \"FinInstrmTp\" IN ('IDF', 'STF') LIMIT 1",
            [d_str, symbol],
        )
        if not df.empty:
            return float(df["close"].iloc[0])
    except Exception:
        pass

    return 0.0
