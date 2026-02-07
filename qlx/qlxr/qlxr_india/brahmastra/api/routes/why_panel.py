"""Why Panel routes — operator explainability endpoints.

Reads directly from the persisted WAL (JSONL event logs), NOT from
derived state.  Every field maps 1:1 to the Strategy Contract V1
and journal records.  No inference, no recomputation.

Endpoints:
    GET /api/why/signals/{signal_seq}/context
        Full signal context: direction, conviction, components, regime, reasoning.

    GET /api/why/gates/{signal_seq}
        Gate decision(s) for a signal: approved/blocked, reason, risk metrics.

    GET /api/why/trades/{strategy_id}/{symbol}/{date}
        Full decision chain: signal → gate → order → fill → snapshot.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/why", tags=["why-panel"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class SignalContextOut(BaseModel):
    """Full signal context for the Why Panel."""
    signal_seq: int
    ts: str
    strategy_id: str
    symbol: str
    direction: str
    conviction: float
    instrument_type: str
    strike: float = 0.0
    expiry: str = ""
    ttl_bars: int = 0
    regime: str = ""
    components: dict[str, Any] = {}
    reasoning: str = ""


class GateDecisionOut(BaseModel):
    """Single gate decision for a signal."""
    seq: int
    ts: str
    gate: str
    approved: bool
    adjusted_weight: float
    reason: str = ""
    vpin: float = 0.0
    portfolio_dd: float = 0.0
    strategy_dd: float = 0.0
    total_exposure: float = 0.0


class EventOut(BaseModel):
    """Generic event in a decision chain."""
    seq: int
    ts: str
    event_type: str
    strategy_id: str
    symbol: str
    payload: dict[str, Any] = {}


class TradeDecisionChainOut(BaseModel):
    """Full decision chain for a trade."""
    strategy_id: str
    symbol: str
    date: str
    signals: list[EventOut] = []
    gates: list[EventOut] = []
    orders: list[EventOut] = []
    fills: list[EventOut] = []
    risk_alerts: list[EventOut] = []
    snapshot: EventOut | None = None


class DaySummaryOut(BaseModel):
    """Quick summary of events for a day."""
    date: str
    total_events: int
    event_counts: dict[str, int] = {}
    strategies: list[str] = []
    symbols: list[str] = []


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get(
    "/signals/{signal_seq}/context",
    response_model=SignalContextOut,
    name="get_signal_context",
)
async def get_signal_context(
    signal_seq: int,
    date: str,
    request: Request,
) -> SignalContextOut:
    """Get full signal context for the Why Panel.

    Query params:
        date: YYYY-MM-DD — which day's WAL to search.

    Returns the signal event's payload including strategy-specific
    components (why fields), regime, and reasoning.
    """
    wal_svc = request.app.state.wal_query
    ctx = wal_svc.get_signal_context(signal_seq, date)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=f"No signal found with seq={signal_seq} on {date}",
        )
    return SignalContextOut(**ctx)


@router.get(
    "/gates/{signal_seq}",
    response_model=list[GateDecisionOut],
)
async def get_gate_decisions(
    signal_seq: int,
    date: str,
    request: Request,
) -> list[GateDecisionOut]:
    """Get gate decision(s) for a signal.

    Query params:
        date: YYYY-MM-DD — which day's WAL to search.

    Returns all gate decisions that followed this signal in the event
    stream (same strategy_id + symbol, seq order).
    """
    wal_svc = request.app.state.wal_query
    decisions = wal_svc.get_gate_decisions(signal_seq, date)
    return [GateDecisionOut(**d) for d in decisions]


@router.get(
    "/trades/{strategy_id}/{symbol}/{date}",
    response_model=TradeDecisionChainOut,
)
async def get_trade_decisions(
    strategy_id: str,
    symbol: str,
    date: str,
    request: Request,
) -> TradeDecisionChainOut:
    """Get the full decision chain for a trade.

    Path params:
        strategy_id: e.g. "s5_hawkes"
        symbol: e.g. "NIFTY"
        date: YYYY-MM-DD

    Returns signal → gate → order → fill → snapshot chain, all from WAL.
    """
    wal_svc = request.app.state.wal_query
    chain = wal_svc.get_trade_decision_chain(strategy_id, symbol, date)
    if chain is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trade events found for {strategy_id}/{symbol} on {date}",
        )
    return TradeDecisionChainOut(**chain)


@router.get(
    "/dates",
    response_model=list[str],
)
async def get_available_dates(request: Request) -> list[str]:
    """List dates with available event data."""
    wal_svc = request.app.state.wal_query
    return wal_svc.available_dates()


@router.get(
    "/summary/{date}",
    response_model=DaySummaryOut,
)
async def get_day_summary(date: str, request: Request) -> DaySummaryOut:
    """Quick summary of events on a day."""
    wal_svc = request.app.state.wal_query
    summary = wal_svc.day_summary(date)
    return DaySummaryOut(**summary)
