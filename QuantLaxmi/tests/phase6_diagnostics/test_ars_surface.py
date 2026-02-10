"""Phase 6 — ARS Surface Service tests.

Tests activation readiness surface construction, conviction extraction,
filtering, and heatmap matrix generation.
All tests are self-contained with temp WAL files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantlaxmi.engine.services.ars_surface import ARSSurfaceService, ARSPoint
from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.serde import serialize_envelope
from quantlaxmi.core.events.types import EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wal(tmp_dir: Path, day: str, envelopes: list[EventEnvelope]) -> None:
    """Write a list of envelopes to a JSONL WAL file."""
    path = tmp_dir / f"{day}.jsonl"
    with open(path, "w") as f:
        for e in envelopes:
            f.write(serialize_envelope(e) + "\n")


def _signal_env(
    seq: int,
    strategy_id: str = "s1",
    symbol: str = "NIFTY",
    conviction: float = 0.8,
    ts: str = "2025-09-01T09:15:00.000000Z",
) -> EventEnvelope:
    """Create a SIGNAL EventEnvelope."""
    return EventEnvelope(
        ts=ts,
        seq=seq,
        run_id="test-run",
        event_type=EventType.SIGNAL.value,
        source="signal_generator",
        strategy_id=strategy_id,
        symbol=symbol,
        payload={
            "direction": "long",
            "conviction": conviction,
            "instrument_type": "FUT",
            "ttl_bars": 5,
            "regime": "normal",
        },
    )


def _gate_env(
    seq: int,
    approved: bool,
    strategy_id: str = "s1",
    symbol: str = "NIFTY",
    ts: str = "2025-09-01T09:15:01.000000Z",
) -> EventEnvelope:
    """Create a GATE_DECISION EventEnvelope."""
    return EventEnvelope(
        ts=ts,
        seq=seq,
        run_id="test-run",
        event_type=EventType.GATE_DECISION.value,
        source="risk_monitor",
        strategy_id=strategy_id,
        symbol=symbol,
        payload={"approved": approved, "reason": "test", "gate": "test_gate"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicSurface:
    """test_basic_surface - signals on 2 days -> 2 ARSPoints with correct conviction."""

    def test_basic_surface(self, tmp_path):
        _write_wal(tmp_path, "2025-09-01", [
            _signal_env(seq=1, conviction=0.8, ts="2025-09-01T09:15:00.000000Z"),
            _gate_env(seq=2, approved=True, ts="2025-09-01T09:15:01.000000Z"),
        ])
        _write_wal(tmp_path, "2025-09-02", [
            _signal_env(seq=10, conviction=0.6, ts="2025-09-02T09:15:00.000000Z"),
            _gate_env(seq=11, approved=False, ts="2025-09-02T09:15:01.000000Z"),
        ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        points = svc.surface_for_strategy("s1")

        assert len(points) == 2
        # Sort by date for deterministic assertion
        points_sorted = sorted(points, key=lambda p: p.date)
        assert points_sorted[0].date == "2025-09-01"
        assert points_sorted[0].max_conviction == pytest.approx(0.8, abs=1e-9)
        assert points_sorted[0].signal_count == 1
        assert points_sorted[0].executed_count == 1
        assert points_sorted[0].blocked_count == 0

        assert points_sorted[1].date == "2025-09-02"
        assert points_sorted[1].max_conviction == pytest.approx(0.6, abs=1e-9)
        assert points_sorted[1].signal_count == 1
        assert points_sorted[1].executed_count == 0
        assert points_sorted[1].blocked_count == 1


class TestNoSignalDayZero:
    """test_no_signal_day_zero - day with no signals -> signal_count=0, conviction=0."""

    def test_no_signal_day_zero(self, tmp_path):
        # Write a day with events only for a different strategy
        _write_wal(tmp_path, "2025-09-01", [
            _signal_env(seq=1, strategy_id="s2", ts="2025-09-01T09:15:00.000000Z"),
            _gate_env(seq=2, approved=True, strategy_id="s2",
                      ts="2025-09-01T09:15:01.000000Z"),
        ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        # Query for s1 which has no signals
        points = svc.surface_for_strategy("s1")

        assert len(points) == 1
        assert points[0].signal_count == 0
        assert points[0].max_conviction == 0.0


class TestConvictionFromWal:
    """test_conviction_from_wal - max conviction extracted correctly from payload."""

    def test_conviction_from_wal(self, tmp_path):
        _write_wal(tmp_path, "2025-09-01", [
            _signal_env(seq=1, conviction=0.72, ts="2025-09-01T09:15:00.000000Z"),
            _gate_env(seq=2, approved=True, ts="2025-09-01T09:15:01.000000Z"),
        ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        points = svc.surface_for_strategy("s1")

        assert len(points) == 1
        assert points[0].max_conviction == pytest.approx(0.72, abs=1e-9)


class TestMultipleSignalsMax:
    """test_multiple_signals_max - multiple signals -> max_conviction is the highest."""

    def test_multiple_signals_max(self, tmp_path):
        _write_wal(tmp_path, "2025-09-01", [
            _signal_env(seq=1, conviction=0.5, symbol="NIFTY",
                        ts="2025-09-01T09:15:00.000000Z"),
            _gate_env(seq=2, approved=True, symbol="NIFTY",
                      ts="2025-09-01T09:15:01.000000Z"),
            _signal_env(seq=3, conviction=0.9, symbol="BANKNIFTY",
                        ts="2025-09-01T09:30:00.000000Z"),
            _gate_env(seq=4, approved=True, symbol="BANKNIFTY",
                      ts="2025-09-01T09:30:01.000000Z"),
            _signal_env(seq=5, conviction=0.7, symbol="FINNIFTY",
                        ts="2025-09-01T10:00:00.000000Z"),
            _gate_env(seq=6, approved=False, symbol="FINNIFTY",
                      ts="2025-09-01T10:00:01.000000Z"),
        ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        points = svc.surface_for_strategy("s1")

        assert len(points) == 1
        assert points[0].max_conviction == pytest.approx(0.9, abs=1e-9)
        assert points[0].signal_count == 3
        assert points[0].executed_count == 2
        assert points[0].blocked_count == 1


class TestEmptyWal:
    """test_empty_wal - no WAL files -> empty surface."""

    def test_empty_wal(self, tmp_path):
        # tmp_path is empty — no JSONL files
        svc = ARSSurfaceService(base_dir=tmp_path)
        points = svc.surface_for_strategy("s1")

        assert points == []


class TestSingleStrategyFilter:
    """test_single_strategy_filter - surface_for_strategy filters correctly."""

    def test_single_strategy_filter(self, tmp_path):
        _write_wal(tmp_path, "2025-09-01", [
            _signal_env(seq=1, strategy_id="s1", conviction=0.8,
                        ts="2025-09-01T09:15:00.000000Z"),
            _gate_env(seq=2, approved=True, strategy_id="s1",
                      ts="2025-09-01T09:15:01.000000Z"),
            _signal_env(seq=3, strategy_id="s2", conviction=0.6,
                        ts="2025-09-01T09:30:00.000000Z"),
            _gate_env(seq=4, approved=False, strategy_id="s2",
                      ts="2025-09-01T09:30:01.000000Z"),
        ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        s1_points = svc.surface_for_strategy("s1")
        s2_points = svc.surface_for_strategy("s2")

        assert len(s1_points) == 1
        assert s1_points[0].strategy_id == "s1"
        assert s1_points[0].max_conviction == pytest.approx(0.8, abs=1e-9)
        assert s1_points[0].executed_count == 1
        assert s1_points[0].blocked_count == 0

        assert len(s2_points) == 1
        assert s2_points[0].strategy_id == "s2"
        assert s2_points[0].max_conviction == pytest.approx(0.6, abs=1e-9)
        assert s2_points[0].executed_count == 0
        assert s2_points[0].blocked_count == 1


class TestDateRangeFilter:
    """test_date_range_filter - start_date/end_date filtering works."""

    def test_date_range_filter(self, tmp_path):
        for day_idx in range(1, 6):
            day = f"2025-09-0{day_idx}"
            _write_wal(tmp_path, day, [
                _signal_env(seq=day_idx * 10, conviction=0.1 * day_idx,
                            ts=f"{day}T09:15:00.000000Z"),
                _gate_env(seq=day_idx * 10 + 1, approved=True,
                          ts=f"{day}T09:15:01.000000Z"),
            ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        # Filter: only days 2-4
        points = svc.surface_for_strategy("s1", start_date="2025-09-02", end_date="2025-09-04")

        assert len(points) == 3
        dates = sorted([p.date for p in points])
        assert dates == ["2025-09-02", "2025-09-03", "2025-09-04"]


class TestHeatmapMatrixShape:
    """test_heatmap_matrix_shape - matrix dimensions match strategies x dates."""

    def test_heatmap_matrix_shape(self, tmp_path):
        # 3 strategies across 2 days
        _write_wal(tmp_path, "2025-09-01", [
            _signal_env(seq=1, strategy_id="s1", conviction=0.5,
                        ts="2025-09-01T09:15:00.000000Z"),
            _gate_env(seq=2, approved=True, strategy_id="s1",
                      ts="2025-09-01T09:15:01.000000Z"),
            _signal_env(seq=3, strategy_id="s2", conviction=0.7,
                        ts="2025-09-01T09:30:00.000000Z"),
            _gate_env(seq=4, approved=False, strategy_id="s2",
                      ts="2025-09-01T09:30:01.000000Z"),
            _signal_env(seq=5, strategy_id="s3", conviction=0.9,
                        ts="2025-09-01T10:00:00.000000Z"),
            _gate_env(seq=6, approved=True, strategy_id="s3",
                      ts="2025-09-01T10:00:01.000000Z"),
        ])
        _write_wal(tmp_path, "2025-09-02", [
            _signal_env(seq=10, strategy_id="s1", conviction=0.6,
                        ts="2025-09-02T09:15:00.000000Z"),
            _gate_env(seq=11, approved=True, strategy_id="s1",
                      ts="2025-09-02T09:15:01.000000Z"),
            _signal_env(seq=12, strategy_id="s3", conviction=0.4,
                        ts="2025-09-02T09:30:00.000000Z"),
            _gate_env(seq=13, approved=False, strategy_id="s3",
                      ts="2025-09-02T09:30:01.000000Z"),
        ])

        svc = ARSSurfaceService(base_dir=tmp_path)
        surface = svc.surface_all_strategies()
        hm = svc.heatmap_matrix(surface)

        # 3 strategies, 2 dates
        assert len(hm["strategies"]) == 3
        assert len(hm["dates"]) == 2
        assert hm["dates"] == ["2025-09-01", "2025-09-02"]
        assert sorted(hm["strategies"]) == ["s1", "s2", "s3"]

        # matrix: [n_strategies][n_dates]
        assert len(hm["matrix"]) == 3
        for row in hm["matrix"]:
            assert len(row) == 2

        # status_matrix: same shape
        assert len(hm["status_matrix"]) == 3
        for row in hm["status_matrix"]:
            assert len(row) == 2

        # s2 has no signal on day 2 -> conviction=0.0, status="none"
        s2_idx = hm["strategies"].index("s2")
        d2_idx = hm["dates"].index("2025-09-02")
        assert hm["matrix"][s2_idx][d2_idx] == 0.0
        assert hm["status_matrix"][s2_idx][d2_idx] == "none"

        # s1 on day 1: executed
        s1_idx = hm["strategies"].index("s1")
        d1_idx = hm["dates"].index("2025-09-01")
        assert hm["status_matrix"][s1_idx][d1_idx] == "executed"
        assert hm["matrix"][s1_idx][d1_idx] == pytest.approx(0.5, abs=1e-9)
