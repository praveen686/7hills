"""Diagnostics API routes — trade analytics, missed opportunities, ARS surface.

Endpoints:
    GET /api/diagnostics/trades/analytics    -> list[TradeAnalyticsOut]
    GET /api/diagnostics/trades/{trade_id}/analytics -> TradeAnalyticsOut
    GET /api/diagnostics/trades/summary      -> dict
    GET /api/diagnostics/missed              -> list[MissedOpportunityOut]
    GET /api/diagnostics/missed/summary      -> dict
    GET /api/diagnostics/ars/{strategy_id}   -> list[ARSPointOut]
    GET /api/diagnostics/ars                 -> ARSHeatmapOut
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class TradeAnalyticsOut(BaseModel):
    trade_id: str
    strategy_id: str
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    mfm: float
    mda: float
    efficiency: float
    exit_quality: float
    duration_days: int
    optimal_exit_price: float
    worst_price: float
    mfm_source: str
    price_path_available: bool


class MissedOpportunityOut(BaseModel):
    signal_seq: int
    ts: str
    strategy_id: str
    symbol: str
    direction: str
    conviction: float
    instrument_type: str
    ttl_bars: int
    regime: str
    block_reason: str
    gate: str
    risk_metrics: dict[str, Any]
    entry_price: float
    hypothetical_exit_price: float
    hypothetical_pnl_pct: float
    hypothetical_mfm: float
    price_data_available: bool


class ARSPointOut(BaseModel):
    date: str
    strategy_id: str
    max_conviction: float
    signal_count: int
    executed_count: int
    blocked_count: int


class ARSHeatmapOut(BaseModel):
    dates: list[str]
    strategies: list[str]
    matrix: list[list[float]]
    status_matrix: list[list[str]]


# ------------------------------------------------------------------
# Trade Analytics routes
# ------------------------------------------------------------------

@router.get(
    "/trades/summary",
    name="diagnostics_trade_summary",
)
async def get_trade_summary(request: Request) -> dict:
    """Get aggregated trade analytics summary by strategy."""
    svc = request.app.state.trade_analytics
    state = request.app.state.engine
    trades = state.closed_trades
    analytics = svc.analyze_all(trades)
    return svc.summary_by_strategy(analytics)


@router.get(
    "/trades/analytics",
    response_model=list[TradeAnalyticsOut],
    name="diagnostics_trade_analytics",
)
async def get_trade_analytics(
    request: Request,
    strategy_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[TradeAnalyticsOut]:
    """Get trade analytics for all or filtered closed trades."""
    svc = request.app.state.trade_analytics
    state = request.app.state.engine
    trades = state.closed_trades

    if strategy_id:
        trades = [t for t in trades if t.strategy_id == strategy_id]

    # Apply pagination
    trades = trades[offset:offset + limit]
    analytics = svc.analyze_all(trades)
    return [TradeAnalyticsOut(**a.to_dict()) for a in analytics]


@router.get(
    "/trades/{trade_id:path}/analytics",
    response_model=TradeAnalyticsOut,
    name="diagnostics_single_trade",
)
async def get_single_trade_analytics(
    trade_id: str,
    request: Request,
) -> TradeAnalyticsOut:
    """Get analytics for a single trade by trade_id."""
    svc = request.app.state.trade_analytics
    state = request.app.state.engine

    trade = None
    for t in state.closed_trades:
        if getattr(t, "trade_id", "") == trade_id:
            trade = t
            break

    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade not found: {trade_id}")

    analytics = svc.analyze_trade(trade)
    return TradeAnalyticsOut(**analytics.to_dict())


# ------------------------------------------------------------------
# Missed Opportunity routes
# ------------------------------------------------------------------

@router.get(
    "/missed/summary",
    name="diagnostics_missed_summary",
)
async def get_missed_summary(
    request: Request,
    start_date: str = Query(...),
    end_date: str = Query(...),
) -> dict:
    """Get aggregated missed opportunity summary by strategy."""
    svc = request.app.state.missed_opportunity
    opportunities = svc.analyze_range(start_date, end_date)
    return svc.summary_by_strategy(opportunities)


@router.get(
    "/missed",
    response_model=list[MissedOpportunityOut],
    name="diagnostics_missed",
)
async def get_missed_opportunities(
    request: Request,
    date: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    strategy_id: str | None = Query(None),
) -> list[MissedOpportunityOut]:
    """Get missed opportunities (blocked signals) for a date or range."""
    svc = request.app.state.missed_opportunity

    if date:
        opportunities = svc.analyze_missed(date)
    elif start_date and end_date:
        opportunities = svc.analyze_range(start_date, end_date)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'date' or both 'start_date' and 'end_date'",
        )

    if strategy_id:
        opportunities = [o for o in opportunities if o.strategy_id == strategy_id]

    return [MissedOpportunityOut(**o.to_dict()) for o in opportunities]


# ------------------------------------------------------------------
# ARS Surface routes
# ------------------------------------------------------------------

@router.get(
    "/ars",
    response_model=ARSHeatmapOut,
    name="diagnostics_ars_heatmap",
)
async def get_ars_heatmap(
    request: Request,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
) -> ARSHeatmapOut:
    """Get ARS heatmap across all strategies."""
    svc = request.app.state.ars_surface
    surface = svc.surface_all_strategies(start_date, end_date)
    hm = svc.heatmap_matrix(surface)
    return ARSHeatmapOut(**hm)


@router.get(
    "/ars/{strategy_id}",
    response_model=list[ARSPointOut],
    name="diagnostics_ars_strategy",
)
async def get_ars_strategy(
    strategy_id: str,
    request: Request,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
) -> list[ARSPointOut]:
    """Get ARS surface for a single strategy."""
    svc = request.app.state.ars_surface
    points = svc.surface_for_strategy(strategy_id, start_date, end_date)
    return [ARSPointOut(**p.to_dict()) for p in points]
