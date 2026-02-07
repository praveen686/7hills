"""Phase 4 — Why Panel API route tests.

Tests the FastAPI endpoints for the Why Panel, using TestClient
with a real WalQueryService backed by a temp directory of JSONL events.

Core invariants:
  - GET /api/why/signals/{seq}/context returns 200 + correct payload
  - GET /api/why/gates/{seq} returns 200 + list of gate decisions
  - GET /api/why/trades/{strategy_id}/{symbol}/{date} returns full chain
  - 404 returned for non-existent signals/trades
  - GET /api/why/dates returns available dates
  - GET /api/why/summary/{date} returns event counts
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine.api.routes.why_panel import router
from engine.services.wal_query import WalQueryService
from engine.live.event_log import EventLogWriter
from core.events.types import EventType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TS = "2025-10-01T09:30:00.000000Z"
_RUN_ID = "api-test-001"


@pytest.fixture
def events_dir(tmp_path: Path):
    """Create a temp directory with a full event chain."""
    d = tmp_path / "events"

    writer = EventLogWriter(
        base_dir=d,
        run_id=_RUN_ID,
        fsync_policy="none",
    )

    # Signal
    sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="test",
        payload={
            "direction": "long",
            "conviction": 0.85,
            "instrument_type": "FUT",
            "strike": 0.0,
            "expiry": "",
            "ttl_bars": 5,
            "regime": "low_vol",
            "components": {
                "gex_regime": "positive",
                "raw_score": 0.72,
            },
            "reasoning": "Hawkes intensity spike",
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS,
    )

    # Gate
    writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="test",
        payload={
            "signal_seq": sig.seq,
            "gate": "risk_check",
            "approved": True,
            "adjusted_weight": 0.12,
            "reason": "",
            "vpin": 0.35,
            "portfolio_dd": 0.02,
            "strategy_dd": 0.01,
            "total_exposure": 0.45,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS,
    )

    # Order
    writer.emit(
        event_type=EventType.ORDER.value,
        source="test",
        payload={
            "order_id": "ord-001",
            "action": "submit",
            "side": "buy",
            "order_type": "market",
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS,
    )

    # Snapshot
    writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="test",
        payload={
            "equity": 1.05,
            "peak_equity": 1.08,
            "portfolio_dd": 0.028,
            "total_exposure": 0.45,
            "vpin": 0.35,
            "position_count": 3,
            "regime": "low_vol",
        },
        ts=_TS,
    )

    writer.close()

    return d, sig.seq


@pytest.fixture
def client(events_dir):
    """Create a FastAPI TestClient with WalQueryService wired up."""
    d, sig_seq = events_dir

    app = FastAPI()
    app.include_router(router)

    # Wire WalQueryService into app.state
    wal_svc = WalQueryService(base_dir=d)
    app.state.wal_query = wal_svc

    return TestClient(app), sig_seq


# ===================================================================
# Test Class: Signal Context Endpoint
# ===================================================================


class TestSignalContextEndpoint:
    """GET /api/why/signals/{signal_seq}/context."""

    def test_200_returns_signal(self, client):
        """Returns 200 with correct signal context."""
        tc, sig_seq = client
        resp = tc.get(f"/api/why/signals/{sig_seq}/context?date=2025-10-01")

        assert resp.status_code == 200
        data = resp.json()
        assert data["signal_seq"] == sig_seq
        assert data["strategy_id"] == "s5_hawkes"
        assert data["symbol"] == "NIFTY"
        assert data["direction"] == "long"
        assert data["conviction"] == 0.85
        assert data["regime"] == "low_vol"
        assert data["components"]["gex_regime"] == "positive"
        assert "Hawkes" in data["reasoning"]

    def test_404_missing_signal(self, client):
        """Returns 404 for non-existent signal."""
        tc, _ = client
        resp = tc.get("/api/why/signals/9999/context?date=2025-10-01")
        assert resp.status_code == 404

    def test_404_wrong_date(self, client):
        """Returns 404 for wrong date."""
        tc, sig_seq = client
        resp = tc.get(f"/api/why/signals/{sig_seq}/context?date=2025-10-05")
        assert resp.status_code == 404


# ===================================================================
# Test Class: Gate Decisions Endpoint
# ===================================================================


class TestGateDecisionsEndpoint:
    """GET /api/why/gates/{signal_seq}."""

    def test_200_returns_decisions(self, client):
        """Returns 200 with list of gate decisions."""
        tc, sig_seq = client
        resp = tc.get(f"/api/why/gates/{sig_seq}?date=2025-10-01")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["approved"] is True
        assert data[0]["gate"] == "risk_check"
        assert data[0]["vpin"] == 0.35

    def test_empty_for_missing_signal(self, client):
        """Returns empty list for non-existent signal."""
        tc, _ = client
        resp = tc.get("/api/why/gates/9999?date=2025-10-01")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []


# ===================================================================
# Test Class: Trade Decisions Endpoint
# ===================================================================


class TestTradeDecisionsEndpoint:
    """GET /api/why/trades/{strategy_id}/{symbol}/{date}."""

    def test_200_returns_chain(self, client):
        """Returns 200 with full decision chain."""
        tc, _ = client
        resp = tc.get("/api/why/trades/s5_hawkes/NIFTY/2025-10-01")

        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy_id"] == "s5_hawkes"
        assert data["symbol"] == "NIFTY"
        assert data["date"] == "2025-10-01"
        assert len(data["signals"]) == 1
        assert len(data["gates"]) == 1
        assert len(data["orders"]) == 1
        assert data["snapshot"] is not None

    def test_404_wrong_strategy(self, client):
        """Returns 404 for non-existent strategy."""
        tc, _ = client
        resp = tc.get("/api/why/trades/s99_fake/NIFTY/2025-10-01")
        assert resp.status_code == 404

    def test_404_wrong_date(self, client):
        """Returns 404 for date with no events."""
        tc, _ = client
        resp = tc.get("/api/why/trades/s5_hawkes/NIFTY/2025-12-25")
        assert resp.status_code == 404

    def test_chain_signal_has_payload(self, client):
        """Signal event in chain includes strategy-specific components."""
        tc, _ = client
        resp = tc.get("/api/why/trades/s5_hawkes/NIFTY/2025-10-01")

        data = resp.json()
        sig_payload = data["signals"][0]["payload"]
        assert sig_payload["components"]["gex_regime"] == "positive"
        assert sig_payload["conviction"] == 0.85


# ===================================================================
# Test Class: Utility Endpoints
# ===================================================================


class TestUtilityEndpoints:
    """GET /api/why/dates and /api/why/summary/{date}."""

    def test_dates_returns_list(self, client):
        """Returns list of available dates."""
        tc, _ = client
        resp = tc.get("/api/why/dates")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert "2025-10-01" in data

    def test_summary_returns_counts(self, client):
        """Returns event counts for a date."""
        tc, _ = client
        resp = tc.get("/api/why/summary/2025-10-01")

        assert resp.status_code == 200
        data = resp.json()
        assert data["date"] == "2025-10-01"
        assert data["total_events"] == 4  # signal + gate + order + snapshot
        assert "s5_hawkes" in data["strategies"]
        assert "NIFTY" in data["symbols"]

    def test_summary_empty_date(self, client):
        """Returns zero counts for empty date."""
        tc, _ = client
        resp = tc.get("/api/why/summary/2025-12-25")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] == 0
