"""Phase 5 — ReplayService tests.

Tests the replay service layer that provides time-travel queries
over the persisted WAL event log.

Core invariants:
  - snapshot_at returns nearest-before SNAPSHOT (not after)
  - timeline filters to decision-relevant types only (no ticks/bars)
  - step returns events in (from_ts, next_ts] window
  - step.has_more is True when events exist after window
  - chained steps cover all events without duplication or loss
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.events.types import EventType
from engine.live.event_log import EventLogWriter
from engine.services.replay_service import ReplayService


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DAY = "2025-10-01"
_RUN_ID = "replay-svc-test-001"

# Timestamps spread across a trading session
_TS_0915 = "2025-10-01T09:15:00.000000Z"
_TS_0930 = "2025-10-01T09:30:00.000000Z"
_TS_1000 = "2025-10-01T10:00:00.000000Z"
_TS_1030 = "2025-10-01T10:30:00.000000Z"
_TS_1100 = "2025-10-01T11:00:00.000000Z"
_TS_1200 = "2025-10-01T12:00:00.000000Z"
_TS_1300 = "2025-10-01T13:00:00.000000Z"
_TS_1500 = "2025-10-01T15:00:00.000000Z"
_TS_1530 = "2025-10-01T15:30:00.000000Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_standard_day(base_dir: Path) -> dict:
    """Write a standard day of events for testing.

    Returns dict with seq numbers for key events.
    """
    writer = EventLogWriter(base_dir=base_dir, run_id=_RUN_ID, fsync_policy="none")

    # 09:15 — Session start snapshot
    snap1 = writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="test",
        payload={"equity": 1.00, "peak_equity": 1.00, "portfolio_dd": 0.0,
                 "total_exposure": 0.0, "vpin": 0.20, "position_count": 0, "regime": "low_vol"},
        ts=_TS_0915,
    )

    # 09:30 — Signal
    sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="test",
        payload={"direction": "long", "conviction": 0.85, "instrument_type": "FUT",
                 "regime": "low_vol", "components": {"raw_score": 0.72}},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_0930,
    )

    # 09:30 — Gate decision
    gate = writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="test",
        payload={"gate": "risk_check", "approved": True, "adjusted_weight": 0.12,
                 "reason": "", "vpin": 0.35, "portfolio_dd": 0.02,
                 "strategy_dd": 0.01, "total_exposure": 0.45},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_0930,
    )

    # 10:00 — Order
    order = writer.emit(
        event_type=EventType.ORDER.value,
        source="test",
        payload={"order_id": "ord-001", "action": "submit", "side": "buy",
                 "order_type": "market", "price": 19500.0},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_1000,
    )

    # 10:00 — Fill
    fill = writer.emit(
        event_type=EventType.FILL.value,
        source="test",
        payload={"fill_id": "fill-001", "side": "buy", "price": 19500.5,
                 "quantity": 50, "fees": 25.0},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_1000,
    )

    # 10:30 — Mid-day snapshot
    snap2 = writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="test",
        payload={"equity": 1.02, "peak_equity": 1.02, "portfolio_dd": 0.0,
                 "total_exposure": 0.45, "vpin": 0.30, "position_count": 1, "regime": "low_vol"},
        ts=_TS_1030,
    )

    # 12:00 — Risk alert
    alert = writer.emit(
        event_type=EventType.RISK_ALERT.value,
        source="test",
        payload={"alert_type": "vpin_elevated", "new_state": "caution",
                 "threshold": 0.5, "current_value": 0.55},
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_1200,
    )

    # 13:00 — Second signal (different strategy)
    sig2 = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="test",
        payload={"direction": "short", "conviction": 0.70, "instrument_type": "OPT",
                 "regime": "high_vol", "components": {"iv_pctile": 0.90}},
        strategy_id="s4_iv_mr",
        symbol="BANKNIFTY",
        ts=_TS_1300,
    )

    # 15:00 — End-of-day snapshot
    snap3 = writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="test",
        payload={"equity": 1.01, "peak_equity": 1.02, "portfolio_dd": 0.0098,
                 "total_exposure": 0.45, "vpin": 0.25, "position_count": 1, "regime": "low_vol"},
        ts=_TS_1500,
    )

    writer.close()

    return {
        "snap1": snap1.seq,
        "sig": sig.seq,
        "gate": gate.seq,
        "order": order.seq,
        "fill": fill.seq,
        "snap2": snap2.seq,
        "alert": alert.seq,
        "sig2": sig2.seq,
        "snap3": snap3.seq,
    }


@pytest.fixture
def replay_dir(tmp_path: Path):
    """Create temp directory with standard day events, return (dir, seqs)."""
    d = tmp_path / "events"
    seqs = _write_standard_day(d)
    return d, seqs


@pytest.fixture
def svc(replay_dir):
    """ReplayService backed by temp directory."""
    d, seqs = replay_dir
    return ReplayService(base_dir=d), seqs


# ===================================================================
# TestAvailableDates
# ===================================================================


class TestAvailableDates:
    """available_dates() returns sorted list of dates with data."""

    def test_returns_single_day(self, svc):
        service, _ = svc
        dates = service.available_dates()
        assert dates == [_DAY]

    def test_returns_empty_for_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        service = ReplayService(base_dir=d)
        assert service.available_dates() == []


# ===================================================================
# TestSnapshotAt
# ===================================================================


class TestSnapshotAt:
    """snapshot_at() nearest-before semantics."""

    def test_exact_match(self, svc):
        """Returns snapshot at exact timestamp."""
        service, seqs = svc
        snap = service.snapshot_at(_TS_0915, _DAY)
        assert snap is not None
        assert snap["seq"] == seqs["snap1"]
        assert snap["payload"]["equity"] == 1.00

    def test_nearest_before(self, svc):
        """Returns last snapshot before target (not after)."""
        service, seqs = svc
        # At 11:00, the last snapshot is snap2 at 10:30
        snap = service.snapshot_at(_TS_1100, _DAY)
        assert snap is not None
        assert snap["seq"] == seqs["snap2"]
        assert snap["payload"]["equity"] == 1.02

    def test_returns_latest_when_after_all(self, svc):
        """Returns last snapshot when target is after all events."""
        service, seqs = svc
        snap = service.snapshot_at(_TS_1530, _DAY)
        assert snap is not None
        assert snap["seq"] == seqs["snap3"]

    def test_returns_none_before_first(self, svc):
        """Returns None when target is before first snapshot."""
        service, _ = svc
        snap = service.snapshot_at("2025-10-01T09:00:00.000000Z", _DAY)
        assert snap is None

    def test_returns_none_empty_date(self, svc):
        """Returns None for a date with no events."""
        service, _ = svc
        snap = service.snapshot_at(_TS_0930, "2025-12-25")
        assert snap is None


# ===================================================================
# TestTimeline
# ===================================================================


class TestTimeline:
    """timeline() filters to decision-relevant types."""

    def test_correct_count(self, svc):
        """Returns markers for all timeline-relevant events."""
        service, _ = svc
        markers = service.timeline(_DAY)
        # 3 snapshots + 2 signals + 1 gate + 1 order + 1 fill + 1 alert = 9
        assert len(markers) == 9

    def test_excludes_ticks_and_bars(self, tmp_path):
        """Ticks and bars are not included in timeline."""
        d = tmp_path / "events"
        writer = EventLogWriter(base_dir=d, run_id=_RUN_ID, fsync_policy="none")

        writer.emit(
            event_type=EventType.TICK.value,
            source="test",
            payload={"ltp": 19500},
            ts=_TS_0930,
        )
        writer.emit(
            event_type=EventType.BAR_1M.value,
            source="test",
            payload={"open": 19500, "close": 19510},
            ts=_TS_0930,
        )
        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "long", "conviction": 0.5},
            strategy_id="s1",
            symbol="NIFTY",
            ts=_TS_0930,
        )
        writer.close()

        service = ReplayService(base_dir=d)
        markers = service.timeline(_DAY)
        assert len(markers) == 1
        assert markers[0]["event_type"] == EventType.SIGNAL.value

    def test_marker_has_summary(self, svc):
        """Each marker has a non-empty summary."""
        service, _ = svc
        markers = service.timeline(_DAY)
        for m in markers:
            assert isinstance(m["summary"], str)
            assert len(m["summary"]) > 0

    def test_marker_types_correct(self, svc):
        """All marker types are in the allowed set."""
        service, _ = svc
        markers = service.timeline(_DAY)
        allowed = {"signal", "gate_decision", "order", "fill", "risk_alert", "snapshot"}
        for m in markers:
            assert m["event_type"] in allowed

    def test_empty_date(self, svc):
        """Returns empty list for non-existent date."""
        service, _ = svc
        markers = service.timeline("2025-12-25")
        assert markers == []

    def test_signal_summary_format(self, svc):
        """Signal summary includes direction and conviction."""
        service, _ = svc
        markers = service.timeline(_DAY)
        signal_markers = [m for m in markers if m["event_type"] == "signal"]
        assert len(signal_markers) == 2
        assert "long" in signal_markers[0]["summary"]
        assert "0.85" in signal_markers[0]["summary"]


# ===================================================================
# TestStep
# ===================================================================


class TestStep:
    """step() returns events in time window."""

    def test_window_captures_events(self, svc):
        """Events within (from_ts, next_ts] are returned."""
        service, _ = svc
        # Window: (09:15, 09:30+60s] => should capture 09:30 events (signal + gate)
        result = service.step(
            from_ts=_TS_0915,
            delta_ms=15 * 60 * 1000 + 1000,  # 15 min + 1s to include 09:30
            day=_DAY,
        )
        types = [e["event_type"] for e in result["events"]]
        assert "signal" in types
        assert "gate_decision" in types

    def test_from_ts_exclusive(self, svc):
        """Event at exactly from_ts is NOT included (exclusive lower bound)."""
        service, seqs = svc
        # Start at 09:15 (snapshot time), window covering just 09:15 only
        result = service.step(from_ts=_TS_0915, delta_ms=0, day=_DAY)
        # Zero-width window should have no events
        assert len(result["events"]) == 0

    def test_has_more_true(self, svc):
        """has_more is True when events exist after window."""
        service, _ = svc
        # Small window at start — events exist after
        result = service.step(from_ts=_TS_0915, delta_ms=60000, day=_DAY)
        assert result["has_more"] is True

    def test_has_more_false_at_end(self, svc):
        """has_more is False when window covers all remaining events."""
        service, _ = svc
        # Large window from start to well past end of day
        result = service.step(
            from_ts="2025-10-01T00:00:00.000000Z",
            delta_ms=24 * 60 * 60 * 1000,  # 24 hours
            day=_DAY,
        )
        assert result["has_more"] is False

    def test_snapshot_included(self, svc):
        """Latest snapshot at or before window end is returned."""
        service, seqs = svc
        # Window up to 10:30 — snap2 should be the latest snapshot
        result = service.step(
            from_ts="2025-10-01T00:00:00.000000Z",
            delta_ms=10 * 60 * 60 * 1000 + 30 * 60 * 1000,  # 10:30
            day=_DAY,
        )
        assert result["snapshot"] is not None
        assert result["snapshot"]["seq"] == seqs["snap2"]

    def test_next_ts_correct(self, svc):
        """next_ts equals from_ts + delta_ms."""
        service, _ = svc
        result = service.step(from_ts=_TS_0915, delta_ms=60000, day=_DAY)
        # 09:15:00 + 60s = 09:16:00
        assert "09:16:00" in result["next_ts"]

    def test_chained_steps_cover_all(self, svc):
        """Chaining steps covers all events without duplication."""
        service, _ = svc
        all_seqs = set()
        from_ts = "2025-10-01T00:00:00.000000Z"
        delta = 60 * 60 * 1000  # 1-hour steps

        for _ in range(24):
            result = service.step(from_ts=from_ts, delta_ms=delta, day=_DAY)
            for e in result["events"]:
                assert e["seq"] not in all_seqs, f"Duplicate seq {e['seq']}"
                all_seqs.add(e["seq"])
            from_ts = result["next_ts"]
            if not result["has_more"]:
                break

        # Should have found all 9 timeline events
        assert len(all_seqs) == 9

    def test_empty_date(self, svc):
        """Step on non-existent date returns empty."""
        service, _ = svc
        result = service.step(from_ts=_TS_0915, delta_ms=60000, day="2025-12-25")
        assert result["events"] == []
        assert result["snapshot"] is None
        assert result["has_more"] is False

    def test_step_returns_all_timeline_types(self, svc):
        """Step over full day returns all timeline event types."""
        service, _ = svc
        result = service.step(
            from_ts="2025-10-01T00:00:00.000000Z",
            delta_ms=24 * 60 * 60 * 1000,
            day=_DAY,
        )
        types = {e["event_type"] for e in result["events"]}
        assert "signal" in types
        assert "gate_decision" in types
        assert "order" in types
        assert "fill" in types
        assert "risk_alert" in types
        assert "snapshot" in types


# ===================================================================
# TestMultiDay
# ===================================================================


class TestMultiDay:
    """Replay across multiple days."""

    def test_dates_sorted(self, tmp_path):
        """available_dates returns sorted list across multiple days."""
        d = tmp_path / "events"
        writer = EventLogWriter(base_dir=d, run_id=_RUN_ID, fsync_policy="none")

        # Day 2 first, then day 1
        writer.emit(
            event_type=EventType.SNAPSHOT.value,
            source="test",
            payload={"equity": 1.0},
            ts="2025-10-02T09:15:00.000000Z",
        )
        writer.emit(
            event_type=EventType.SNAPSHOT.value,
            source="test",
            payload={"equity": 1.0},
            ts="2025-10-01T09:15:00.000000Z",
        )
        writer.close()

        service = ReplayService(base_dir=d)
        dates = service.available_dates()
        assert dates == ["2025-10-01", "2025-10-02"]

    def test_timeline_isolated_per_day(self, tmp_path):
        """Timeline for one day doesn't include other day's events."""
        d = tmp_path / "events"
        writer = EventLogWriter(base_dir=d, run_id=_RUN_ID, fsync_policy="none")

        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "long", "conviction": 0.5},
            strategy_id="s1",
            symbol="NIFTY",
            ts="2025-10-01T09:30:00.000000Z",
        )
        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "short", "conviction": 0.6},
            strategy_id="s1",
            symbol="NIFTY",
            ts="2025-10-02T09:30:00.000000Z",
        )
        writer.close()

        service = ReplayService(base_dir=d)

        day1_markers = service.timeline("2025-10-01")
        day2_markers = service.timeline("2025-10-02")
        assert len(day1_markers) == 1
        assert len(day2_markers) == 1
        assert "long" in day1_markers[0]["summary"]
        assert "short" in day2_markers[0]["summary"]
