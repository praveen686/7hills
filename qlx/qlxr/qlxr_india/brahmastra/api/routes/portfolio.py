"""Portfolio route — GET /api/portfolio.

Returns the current Brahmastra portfolio state: equity, drawdown,
active positions, recent trades, per-strategy equity, equity curve,
day P&L, drawdown history, and total P&L.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class PositionOut(BaseModel):
    strategy_id: str
    symbol: str
    direction: str
    weight: float
    instrument_type: str
    entry_date: str
    entry_price: float
    strike: float = 0.0
    expiry: str = ""
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class ClosedTradeOut(BaseModel):
    strategy_id: str
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    weight: float
    pnl_pct: float
    instrument_type: str = "FUT"
    exit_reason: str = ""


class StrategyEquityOut(BaseModel):
    strategy_id: str
    equity: float
    peak: float
    drawdown_pct: float


class EquityPointOut(BaseModel):
    date: str
    equity: float


class DrawdownPointOut(BaseModel):
    date: str
    drawdown_pct: float


class PortfolioOut(BaseModel):
    equity: float
    peak_equity: float
    cash: float
    drawdown_pct: float
    total_exposure: float
    total_return_pct: float
    total_pnl: float
    day_pnl: float
    win_rate: float
    n_positions: int
    n_closed_trades: int
    last_scan_date: str
    last_scan_time: str
    scan_count: int
    circuit_breaker_active: bool
    last_vix: float
    last_regime: str
    positions: list[PositionOut]
    recent_trades: list[ClosedTradeOut]
    strategy_equity: list[StrategyEquityOut]
    equity_curve: list[EquityPointOut]
    drawdown_history: list[DrawdownPointOut]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_drawdown_history(
    equity_history: list[dict],
) -> list[DrawdownPointOut]:
    """Derive drawdown series from equity history.

    For each point, drawdown_pct = (running_max - equity) / running_max * 100.
    """
    if not equity_history:
        return []

    result: list[DrawdownPointOut] = []
    running_max = 0.0

    for point in equity_history:
        eq = float(point.get("equity", 0.0))
        dt = str(point.get("date", ""))

        if eq > running_max:
            running_max = eq

        if running_max > 0:
            dd_pct = round((running_max - eq) / running_max * 100, 4)
        else:
            dd_pct = 0.0

        result.append(DrawdownPointOut(date=dt, drawdown_pct=dd_pct))

    return result


# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------

@router.get("", response_model=PortfolioOut)
async def get_portfolio(request: Request) -> PortfolioOut:
    """Return the full Brahmastra portfolio snapshot."""
    state = request.app.state.brahmastra

    positions = [
        PositionOut(**p.to_dict()) for p in state.active_positions()
    ]

    recent_trades = [
        ClosedTradeOut(**t.to_dict()) for t in state.closed_trades[-50:]
    ]

    strategy_eq: list[StrategyEquityOut] = []
    for sid, eq in sorted(state.strategy_equity.items()):
        peak = state.strategy_peaks.get(sid, 1.0)
        dd = state.strategy_dd(sid)
        strategy_eq.append(
            StrategyEquityOut(
                strategy_id=sid,
                equity=round(eq, 6),
                peak=round(peak, 6),
                drawdown_pct=round(dd * 100, 4),
            )
        )

    # Equity curve from equity_history
    equity_history = getattr(state, "equity_history", [])
    equity_curve = [
        EquityPointOut(
            date=str(point.get("date", "")),
            equity=round(float(point.get("equity", 0.0)), 6),
        )
        for point in equity_history
    ]

    # Drawdown history derived from equity curve
    drawdown_history = _compute_drawdown_history(equity_history)

    # Day P&L: difference between current equity and equity at session open
    equity_at_open = getattr(state, "equity_at_open", state.equity)
    day_pnl = round(state.equity - equity_at_open, 6)

    # Total P&L: absolute change from initial equity (1.0)
    total_pnl = round(state.equity - 1.0, 6)

    return PortfolioOut(
        equity=round(state.equity, 6),
        peak_equity=round(state.peak_equity, 6),
        cash=round(state.cash, 6),
        drawdown_pct=round(state.portfolio_dd * 100, 4),
        total_exposure=round(state.total_exposure, 6),
        total_return_pct=round(state.total_return_pct(), 4),
        total_pnl=total_pnl,
        day_pnl=day_pnl,
        win_rate=round(state.win_rate() * 100, 2),
        n_positions=len(state.positions),
        n_closed_trades=len(state.closed_trades),
        last_scan_date=state.last_scan_date,
        last_scan_time=state.last_scan_time,
        scan_count=state.scan_count,
        circuit_breaker_active=state.circuit_breaker_active,
        last_vix=round(state.last_vix, 2),
        last_regime=state.last_regime,
        positions=positions,
        recent_trades=recent_trades,
        strategy_equity=strategy_eq,
        equity_curve=equity_curve,
        drawdown_history=drawdown_history,
    )
