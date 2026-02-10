"""Phase 5 — Replay API route tests.

Tests the FastAPI endpoints for the Replay Controls, using TestClient
with a real ReplayService backed by a temp directory of JSONL events.

Core invariants:
  - GET /api/replay/dates returns 200 + sorted list
  - GET /api/replay/snapshot/{ts} returns 200 nearest, 404 missing
  - GET /api/replay/timeline/{date} returns 200 markers, correct types
  - GET /api/replay/step returns 200 with events in window
  - Jump-to-trade consistency: marker → WhyPanel chain
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quantlaxmi.engine.api.routes.replay import router as replay_router
from quantlaxmi.engine.api.routes.why_panel import router as why_router
from quantlaxmi.engine.services.replay_service import ReplayService
from quantlaxmi.engine.services.wal_query import WalQueryService
from quantlaxmi.engine.live.event_log import EventLogWriter
from quantlaxmi.core.events.types import EventType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DAY = "2025-10-01"
_TS = "2025-10-01T09:30:00.000000Z"
_TS_1000 = "2025-10-01T10:00:00.000000Z"
_TS_1030 = "2025-10-01T10:30:00.000000Z"
_RUN_ID = "replay-api-test-001"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def events_dir(tmp_path: Path):
    """Create temp dir with a full event chain."""
    d = tmp_path / "events"
    writer = EventLogWriter(base_dir=d, run_id=_RUN_ID, fsync_policy="none")

    # Signal
    sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="test",
        payload={"direction": "long", "conviction": 0.85, "instrument_type": "FUT",
                 "regime": "low_vol", "components": {"raw_score": 0.72}},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS,
    )

    # Gate
    writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="test",
        payload={"gate": "risk_check", "approved": True, "adjusted_weight": 0.12,
                 "reason": "", "vpin": 0.35, "portfolio_dd": 0.02,
                 "strategy_dd": 0.01, "total_exposure": 0.45},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS,
    )

    # Order
    writer.emit(
        event_type=EventType.ORDER.value,
        source="test",
        payload={"order_id": "ord-001", "action": "submit", "side": "buy",
                 "order_type": "market"},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_1000,
    )

    # Fill
    writer.emit(
        event_type=EventType.FILL.value,
        source="test",
        payload={"fill_id": "fill-001", "side": "buy", "price": 19500.5,
                 "quantity": 50, "fees": 25.0},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_1000,
    )

    # Snapshot
    writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="test",
        payload={"equity": 1.05, "peak_equity": 1.08, "portfolio_dd": 0.028,
                 "total_exposure": 0.45, "vpin": 0.35, "position_count": 3,
                 "regime": "low_vol"},
        ts=_TS_1030,
    )

    writer.close()
    return d, sig.seq


@pytest.fixture
def client(events_dir):
    """Create a FastAPI TestClient with ReplayService + WalQueryService."""
    d, sig_seq = events_dir

    app = FastAPI()
    app.include_router(replay_router)
    app.include_router(why_router)

    app.state.replay_service = ReplayService(base_dir=d)
    app.state.wal_query = WalQueryService(base_dir=d)

    return TestClient(app), sig_seq


# ===================================================================
# TestReplayDates
# ===================================================================


class TestReplayDates:
    """GET /api/replay/dates."""

    def test_200_sorted_list(self, client):
        """Returns 200 with sorted date list."""
        tc, _ = client
        resp = tc.get("/api/replay/dates")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert _DAY in data

    def test_dates_are_sorted(self, client):
        """Dates are in ascending order."""
        tc, _ = client
        resp = tc.get("/api/replay/dates")
        data = resp.json()
        assert data == sorted(data)


# ===================================================================
# TestReplaySnapshot
# ===================================================================


class TestReplaySnapshot:
    """GET /api/replay/snapshot/{timestamp}."""

    def test_200_returns_nearest(self, client):
        """Returns 200 with nearest snapshot before timestamp."""
        tc, _ = client
        resp = tc.get(f"/api/replay/snapshot/{_TS_1030}?date={_DAY}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_type"] == "snapshot"
        assert data["payload"]["equity"] == 1.05

    def test_404_before_any_snapshot(self, client):
        """Returns 404 when no snapshot exists before timestamp."""
        tc, _ = client
        resp = tc.get(
            "/api/replay/snapshot/2025-10-01T08:00:00.000000Z"
            f"?date={_DAY}"
        )
        assert resp.status_code == 404

    def test_404_wrong_date(self, client):
        """Returns 404 for date with no events."""
        tc, _ = client
        resp = tc.get(f"/api/replay/snapshot/{_TS}?date=2025-12-25")
        assert resp.status_code == 404


# ===================================================================
# TestReplayTimeline
# ===================================================================


class TestReplayTimeline:
    """GET /api/replay/timeline/{date}."""

    def test_200_returns_markers(self, client):
        """Returns 200 with list of timeline markers."""
        tc, _ = client
        resp = tc.get(f"/api/replay/timeline/{_DAY}")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # 1 signal + 1 gate + 1 order + 1 fill + 1 snapshot = 5
        assert len(data) == 5

    def test_marker_types_correct(self, client):
        """All markers have valid event types."""
        tc, _ = client
        resp = tc.get(f"/api/replay/timeline/{_DAY}")
        data = resp.json()
        allowed = {"signal", "gate_decision", "order", "fill", "risk_alert", "snapshot"}
        for m in data:
            assert m["event_type"] in allowed

    def test_empty_date(self, client):
        """Returns empty list for date with no events."""
        tc, _ = client
        resp = tc.get("/api/replay/timeline/2025-12-25")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_marker_has_summary(self, client):
        """Each marker includes a summary string."""
        tc, _ = client
        resp = tc.get(f"/api/replay/timeline/{_DAY}")
        data = resp.json()
        for m in data:
            assert isinstance(m["summary"], str)
            assert len(m["summary"]) > 0


# ===================================================================
# TestReplayStep
# ===================================================================


class TestReplayStep:
    """GET /api/replay/step."""

    def test_200_returns_events_in_window(self, client):
        """Returns events within the time window."""
        tc, _ = client
        # Window from 09:00 to 09:30 + 1min => captures signal + gate at 09:30
        resp = tc.get(
            "/api/replay/step"
            "?from_ts=2025-10-01T09:00:00.000000Z"
            "&delta_ms=1860000"  # 31 minutes
            f"&date={_DAY}"
        )
        assert resp.status_code == 200
        data = resp.json()
        types = [e["event_type"] for e in data["events"]]
        assert "signal" in types
        assert "gate_decision" in types

    def test_has_more_true(self, client):
        """has_more is True when events exist after window."""
        tc, _ = client
        resp = tc.get(
            "/api/replay/step"
            "?from_ts=2025-10-01T09:00:00.000000Z"
            "&delta_ms=1860000"
            f"&date={_DAY}"
        )
        data = resp.json()
        assert data["has_more"] is True

    def test_respects_delta_ms(self, client):
        """Very small window captures no events."""
        tc, _ = client
        resp = tc.get(
            "/api/replay/step"
            "?from_ts=2025-10-01T09:00:00.000000Z"
            "&delta_ms=1000"  # 1 second
            f"&date={_DAY}"
        )
        data = resp.json()
        assert len(data["events"]) == 0

    def test_snapshot_returned(self, client):
        """Latest snapshot at or before window end is returned."""
        tc, _ = client
        # Window covering full day
        resp = tc.get(
            "/api/replay/step"
            "?from_ts=2025-10-01T00:00:00.000000Z"
            "&delta_ms=86400000"
            f"&date={_DAY}"
        )
        data = resp.json()
        assert data["snapshot"] is not None
        assert data["snapshot"]["payload"]["equity"] == 1.05

    def test_next_ts_populated(self, client):
        """next_ts field is a non-empty string."""
        tc, _ = client
        resp = tc.get(
            "/api/replay/step"
            "?from_ts=2025-10-01T09:00:00.000000Z"
            "&delta_ms=60000"
            f"&date={_DAY}"
        )
        data = resp.json()
        assert isinstance(data["next_ts"], str)
        assert len(data["next_ts"]) > 0


# ===================================================================
# TestJumpToTradeConsistency
# ===================================================================


class TestJumpToTradeConsistency:
    """Verify jump from replay marker to WhyPanel chain."""

    def test_signal_marker_matches_why_chain(self, client):
        """A signal marker's strategy_id+symbol+date leads to valid WhyPanel chain."""
        tc, _ = client

        # Get timeline markers
        resp = tc.get(f"/api/replay/timeline/{_DAY}")
        markers = resp.json()
        signal_markers = [m for m in markers if m["event_type"] == "signal"]
        assert len(signal_markers) >= 1

        m = signal_markers[0]
        strategy_id = m["strategy_id"]
        symbol = m["symbol"]

        # Jump to WhyPanel trade chain
        resp = tc.get(f"/api/why/trades/{strategy_id}/{symbol}/{_DAY}")
        assert resp.status_code == 200
        chain = resp.json()
        assert chain["strategy_id"] == strategy_id
        assert chain["symbol"] == symbol
        assert len(chain["signals"]) >= 1
