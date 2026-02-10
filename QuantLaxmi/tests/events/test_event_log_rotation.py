"""Phase 2 — EventLogWriter daily rotation tests.

Invariants:
  1. Daily rotation: events with different dates → different .jsonl files
  2. Correct file naming: YYYY-MM-DD.jsonl
  3. Files are created on demand (no pre-creation)
  4. Closing flushes all buffered data
  5. Event count tracking is accurate
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from quantlaxmi.engine.live.event_log import EventLogWriter, read_event_log
from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.serde import serialize_envelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "events"


@pytest.fixture
def writer(log_dir):
    w = EventLogWriter(base_dir=log_dir, run_id="test-rotation", fsync_policy="none")
    yield w
    w.close()


# ---------------------------------------------------------------------------
# TestDailyRotation
# ---------------------------------------------------------------------------

class TestDailyRotation:
    """Events with different dates land in different files."""

    def test_single_day_single_file(self, writer, log_dir):
        writer.emit(
            event_type="signal", source="test", payload={"v": 1},
            ts="2025-08-06T09:15:00.000000Z",
        )
        writer.emit(
            event_type="signal", source="test", payload={"v": 2},
            ts="2025-08-06T15:30:00.000000Z",
        )
        writer.flush()
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 1
        assert files[0].name == "2025-08-06.jsonl"

    def test_two_days_two_files(self, writer, log_dir):
        writer.emit(
            event_type="signal", source="test", payload={"v": 1},
            ts="2025-08-06T09:15:00.000000Z",
        )
        writer.emit(
            event_type="signal", source="test", payload={"v": 2},
            ts="2025-08-07T09:15:00.000000Z",
        )
        writer.flush()
        files = sorted(log_dir.glob("*.jsonl"))
        assert len(files) == 2
        assert files[0].name == "2025-08-06.jsonl"
        assert files[1].name == "2025-08-07.jsonl"

    def test_three_days_three_files(self, writer, log_dir):
        for day in ["2025-08-06", "2025-08-07", "2025-08-08"]:
            writer.emit(
                event_type="tick", source="test", payload={},
                ts=f"{day}T09:15:00.000000Z",
            )
        writer.flush()
        files = list(log_dir.glob("*.jsonl"))
        assert len(files) == 3

    def test_back_to_earlier_day_creates_file(self, writer, log_dir):
        """Even if we go back to an earlier date, we get the correct file."""
        writer.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-07T09:00:00.000000Z",
        )
        writer.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:00:00.000000Z",
        )
        writer.flush()
        files = sorted(log_dir.glob("*.jsonl"))
        assert len(files) == 2


# ---------------------------------------------------------------------------
# TestFileContent
# ---------------------------------------------------------------------------

class TestFileContent:
    """Written files contain valid JSONL."""

    def test_each_line_is_valid_json(self, writer, log_dir):
        for i in range(5):
            writer.emit(
                event_type="signal", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        writer.flush()

        path = log_dir / "2025-08-06.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            d = json.loads(line)
            assert "seq" in d
            assert "ts" in d
            assert "event_type" in d

    def test_read_event_log_returns_envelopes(self, writer, log_dir):
        writer.emit(
            event_type="signal", source="test",
            payload={"direction": "long"},
            strategy_id="S1", symbol="NIFTY",
            ts="2025-08-06T09:15:00.000000Z",
        )
        writer.flush()

        events = read_event_log(log_dir / "2025-08-06.jsonl")
        assert len(events) == 1
        assert events[0].event_type == "signal"
        assert events[0].strategy_id == "S1"
        assert events[0].payload["direction"] == "long"

    def test_events_across_days_readable(self, writer, log_dir):
        writer.emit(
            event_type="tick", source="test", payload={"d": 1},
            ts="2025-08-06T09:00:00.000000Z",
        )
        writer.emit(
            event_type="tick", source="test", payload={"d": 2},
            ts="2025-08-07T09:00:00.000000Z",
        )
        writer.flush()

        e1 = read_event_log(log_dir / "2025-08-06.jsonl")
        e2 = read_event_log(log_dir / "2025-08-07.jsonl")
        assert len(e1) == 1
        assert len(e2) == 1
        assert e1[0].payload["d"] == 1
        assert e2[0].payload["d"] == 2


# ---------------------------------------------------------------------------
# TestEventCount
# ---------------------------------------------------------------------------

class TestEventCount:
    """Event count and bytes tracking are accurate."""

    def test_event_count(self, writer):
        for i in range(10):
            writer.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        assert writer.event_count == 10

    def test_stats_dict(self, writer):
        writer.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        stats = writer.stats()
        assert stats["event_count"] == 1
        assert stats["bytes_written"] > 0
        assert stats["run_id"] == "test-rotation"

    def test_initial_count_zero(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="fresh")
        assert w.event_count == 0
        assert w.seq == 0
        w.close()


# ---------------------------------------------------------------------------
# TestDirectoryCreation
# ---------------------------------------------------------------------------

class TestDirectoryCreation:
    """Base directory is created automatically."""

    def test_nested_dir_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "events"
        w = EventLogWriter(base_dir=deep, run_id="nest")
        w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        w.close()
        assert deep.exists()
        assert (deep / "2025-08-06.jsonl").exists()


# ---------------------------------------------------------------------------
# TestCloseBehavior
# ---------------------------------------------------------------------------

class TestCloseBehavior:
    """Close flushes and double-close is safe."""

    def test_close_flushes(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="close-test", fsync_policy="none")
        w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        w.close()
        # File should be readable after close
        events = read_event_log(log_dir / "2025-08-06.jsonl")
        assert len(events) == 1

    def test_double_close_safe(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="double-close", fsync_policy="none")
        w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        w.close()
        w.close()  # should not raise

    def test_context_manager(self, log_dir):
        with EventLogWriter(base_dir=log_dir, run_id="ctx", fsync_policy="none") as w:
            w.emit(
                event_type="tick", source="test", payload={},
                ts="2025-08-06T09:15:00.000000Z",
            )
        events = read_event_log(log_dir / "2025-08-06.jsonl")
        assert len(events) == 1
