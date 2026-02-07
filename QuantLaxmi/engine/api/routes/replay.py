"""Replay API routes — time-travel playback endpoints.

Endpoints:
    GET /api/replay/dates         -> list[str]
    GET /api/replay/snapshot/{ts} -> SnapshotOut        (query param: date)
    GET /api/replay/timeline/{date} -> list[TimelineMarkerOut]
    GET /api/replay/step          -> StepOut            (query params)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/replay", tags=["replay"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class TimelineMarkerOut(BaseModel):
    """Single marker on the replay timeline."""
    ts: str
    seq: int
    event_type: str
    strategy_id: str = ""
    symbol: str = ""
    summary: str = ""


class SnapshotOut(BaseModel):
    """Portfolio snapshot at a point in time."""
    ts: str
    seq: int
    event_type: str
    strategy_id: str = ""
    symbol: str = ""
    payload: dict[str, Any] = {}


class StepEventOut(BaseModel):
    """Single event returned in a step response."""
    ts: str
    seq: int
    event_type: str
    strategy_id: str = ""
    symbol: str = ""
    payload: dict[str, Any] = {}


class StepOut(BaseModel):
    """Response from the step endpoint."""
    events: list[StepEventOut] = []
    snapshot: StepEventOut | None = None
    next_ts: str = ""
    has_more: bool = False


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get(
    "/dates",
    response_model=list[str],
    name="replay_dates",
)
async def get_replay_dates(request: Request) -> list[str]:
    """List dates with available event data for replay."""
    svc = request.app.state.replay_service
    return svc.available_dates()


@router.get(
    "/snapshot/{timestamp:path}",
    response_model=SnapshotOut,
    name="replay_snapshot",
)
async def get_replay_snapshot(
    timestamp: str,
    date: str,
    request: Request,
) -> SnapshotOut:
    """Get the nearest portfolio snapshot at or before the given timestamp.

    Path params:
        timestamp: ISO-8601 timestamp (URL-encoded if needed)

    Query params:
        date: YYYY-MM-DD — which day's WAL to search.
    """
    svc = request.app.state.replay_service
    snap = svc.snapshot_at(timestamp, date)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshot found at or before {timestamp} on {date}",
        )
    return SnapshotOut(**snap)


@router.get(
    "/timeline/{date}",
    response_model=list[TimelineMarkerOut],
    name="replay_timeline",
)
async def get_replay_timeline(
    date: str,
    request: Request,
) -> list[TimelineMarkerOut]:
    """Get timeline markers for a day's events.

    Returns markers for decision-relevant event types (SIGNAL, GATE_DECISION,
    ORDER, FILL, RISK_ALERT, SNAPSHOT).
    """
    svc = request.app.state.replay_service
    markers = svc.timeline(date)
    return [TimelineMarkerOut(**m) for m in markers]


@router.get(
    "/step",
    response_model=StepOut,
    name="replay_step",
)
async def get_replay_step(
    from_ts: str,
    delta_ms: int,
    date: str,
    request: Request,
) -> StepOut:
    """Advance replay playback by a time window.

    Query params:
        from_ts: exclusive lower bound timestamp (ISO-8601)
        delta_ms: window size in milliseconds
        date: YYYY-MM-DD
    """
    svc = request.app.state.replay_service
    result = svc.step(from_ts, delta_ms, date)
    return StepOut(
        events=[StepEventOut(**e) for e in result["events"]],
        snapshot=StepEventOut(**result["snapshot"]) if result["snapshot"] else None,
        next_ts=result["next_ts"],
        has_more=result["has_more"],
    )
