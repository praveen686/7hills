"""Phase 6 — Diagnostics API route tests.

Tests all diagnostics endpoints using httpx.AsyncClient with a
lightweight test FastAPI app (no real DuckDB or WAL files).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from engine.api.routes.diagnostics import router
from engine.services.ars_surface import ARSSurfaceService
from engine.services.missed_opportunity import MissedOpportunityService
from engine.services.trade_analytics import TradeAnalyticsService
from engine.state import PortfolioState, ClosedTrade
from core.events.envelope import EventEnvelope
from core.events.serde import serialize_envelope
from core.events.types import EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wal(tmp_dir: Path, day: str, envelopes: list[EventEnvelope]) -> None:
    """Write a list of envelopes to a JSONL WAL file."""
    path = tmp_dir / f"{day}.jsonl"
    with open(path, "w") as f:
        for e in envelopes:
            f.write(serialize_envelope(e) + "\n")


def _make_test_app(tmp_dir: Path | None = None) -> FastAPI:
    """Create a minimal FastAPI app with diagnostics routes and mocked services."""
    app = FastAPI()
    app.include_router(router)

    # Mock state with one closed trade
    state = PortfolioState()
    state.closed_trades = [
        ClosedTrade(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            entry_date="2025-09-01",
            exit_date="2025-09-05",
            entry_price=100.0,
            exit_price=105.0,
            weight=0.1,
            pnl_pct=0.05,
            trade_id="s1:NIFTY:2025-09-01:0",
        ),
        ClosedTrade(
            strategy_id="s2",
            symbol="BANKNIFTY",
            direction="short",
            entry_date="2025-09-02",
            exit_date="2025-09-06",
            entry_price=200.0,
            exit_price=195.0,
            weight=0.1,
            pnl_pct=0.025,
            trade_id="s2:BANKNIFTY:2025-09-02:0",
        ),
    ]
    app.state.engine = state

    # Mock store — returns empty DataFrames
    fake_store = MagicMock()
    fake_store.sql.return_value = pd.DataFrame()
    app.state.trade_analytics = TradeAnalyticsService(store=fake_store)

    # Tmp dir for WAL-based services
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    app.state.missed_opportunity = MissedOpportunityService(base_dir=tmp_dir)
    app.state.ars_surface = ARSSurfaceService(base_dir=tmp_dir)

    return app


def _make_app_with_wal(tmp_dir: Path) -> FastAPI:
    """Create a test app with WAL files for missed opportunity testing."""
    # Write a WAL with one blocked signal
    _write_wal(tmp_dir, "2025-09-01", [
        EventEnvelope(
            ts="2025-09-01T09:15:00.000000Z",
            seq=1,
            run_id="test-run",
            event_type=EventType.SIGNAL.value,
            source="signal_generator",
            strategy_id="s1",
            symbol="NIFTY",
            payload={
                "direction": "long",
                "conviction": 0.8,
                "instrument_type": "FUT",
                "ttl_bars": 5,
                "regime": "normal",
            },
        ),
        EventEnvelope(
            ts="2025-09-01T09:15:01.000000Z",
            seq=2,
            run_id="test-run",
            event_type=EventType.GATE_DECISION.value,
            source="risk_monitor",
            strategy_id="s1",
            symbol="NIFTY",
            payload={
                "approved": False,
                "reason": "portfolio_dd_limit",
                "gate": "portfolio_risk",
                "vpin": 0.35,
                "portfolio_dd": 0.08,
                "strategy_dd": 0.04,
                "total_exposure": 0.6,
            },
        ),
    ])
    return _make_test_app(tmp_dir=tmp_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trades_analytics_200():
    """GET /api/diagnostics/trades/analytics returns 200."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/trades/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2


@pytest.mark.asyncio
async def test_trades_summary_200():
    """GET /api/diagnostics/trades/summary returns 200."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/trades/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "s1" in data
        assert "s2" in data


@pytest.mark.asyncio
async def test_single_trade_200():
    """GET /api/diagnostics/trades/{trade_id}/analytics returns 200."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/diagnostics/trades/s1:NIFTY:2025-09-01:0/analytics"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["trade_id"] == "s1:NIFTY:2025-09-01:0"
        assert data["strategy_id"] == "s1"


@pytest.mark.asyncio
async def test_single_trade_404():
    """GET /api/diagnostics/trades/nonexistent/analytics returns 404."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/trades/nonexistent/analytics")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_missed_200_with_date(tmp_path):
    """GET /api/diagnostics/missed?date=2025-09-01 returns 200."""
    app = _make_app_with_wal(tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/missed?date=2025-09-01")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["strategy_id"] == "s1"


@pytest.mark.asyncio
async def test_missed_400_no_params():
    """GET /api/diagnostics/missed (no params) returns 400."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/missed")
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_missed_summary_200(tmp_path):
    """GET /api/diagnostics/missed/summary?start_date=...&end_date=... returns 200."""
    app = _make_app_with_wal(tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/diagnostics/missed/summary?start_date=2025-09-01&end_date=2025-09-01"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "s1" in data
        assert data["s1"]["n_blocked"] == 1


@pytest.mark.asyncio
async def test_ars_heatmap_200():
    """GET /api/diagnostics/ars returns 200."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/ars")
        assert resp.status_code == 200
        data = resp.json()
        assert "dates" in data
        assert "strategies" in data
        assert "matrix" in data
        assert "status_matrix" in data


@pytest.mark.asyncio
async def test_ars_strategy_200(tmp_path):
    """GET /api/diagnostics/ars/s1 returns 200."""
    # Write a WAL with a signal for s1
    _write_wal(tmp_path, "2025-09-01", [
        EventEnvelope(
            ts="2025-09-01T09:15:00.000000Z",
            seq=1,
            run_id="test-run",
            event_type=EventType.SIGNAL.value,
            source="signal_generator",
            strategy_id="s1",
            symbol="NIFTY",
            payload={"direction": "long", "conviction": 0.8},
        ),
        EventEnvelope(
            ts="2025-09-01T09:15:01.000000Z",
            seq=2,
            run_id="test-run",
            event_type=EventType.GATE_DECISION.value,
            source="risk_monitor",
            strategy_id="s1",
            symbol="NIFTY",
            payload={"approved": True},
        ),
    ])

    app = _make_test_app(tmp_dir=tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/ars/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["strategy_id"] == "s1"
        assert data[0]["max_conviction"] == pytest.approx(0.8, abs=1e-9)


@pytest.mark.asyncio
async def test_strategy_filter():
    """GET /api/diagnostics/trades/analytics?strategy_id=s1 filters correctly."""
    app = _make_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/diagnostics/trades/analytics?strategy_id=s1")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["strategy_id"] == "s1"
