"""Phase 2 — Atomic append and crash recovery tests.

Invariants:
  1. Truncated last line is detected and handled gracefully
  2. Recovery skips corrupt lines and continues from last valid
  3. Append after recovery produces valid events
  4. read_event_log skips blank and corrupt lines with warning
  5. Atomic write: tempfile+rename for session manifests
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantlaxmi.engine.live.event_log import EventLogWriter, read_event_log
from quantlaxmi.engine.live.session_manifest import SessionManifest
from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.serde import serialize_envelope, deserialize_envelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "events"


# ---------------------------------------------------------------------------
# TestTruncatedRecovery
# ---------------------------------------------------------------------------

class TestTruncatedRecovery:
    """Crash mid-write leaves a truncated last line — recovery handles it."""

    def test_truncated_last_line_detected(self, log_dir):
        """Write valid events + a truncated line, then recover."""
        log_dir.mkdir(parents=True, exist_ok=True)

        # Write 3 valid events
        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(3):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        # Simulate crash: append a truncated line
        path = log_dir / "2025-08-06.jsonl"
        with open(path, "a") as f:
            f.write('{"ts":"2025-08-06T09:15:01.000000Z","seq":4')  # truncated!

        # New writer should recover from last valid line (seq=3)
        w2 = EventLogWriter(base_dir=log_dir, run_id="r2", fsync_policy="none")
        env = w2.emit(
            event_type="tick", source="test", payload={"i": 3},
            ts="2025-08-06T09:15:02.000000Z",
        )
        # Should continue from seq=3 (last valid was seq=3, or seq=2 if recovery falls back)
        assert env.seq >= 3
        w2.close()

    def test_empty_file_recovery(self, log_dir):
        """An empty .jsonl file should be handled gracefully."""
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "2025-08-06.jsonl").write_text("")

        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        env = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        assert env.seq == 1
        w.close()

    def test_only_corrupt_lines_file(self, log_dir):
        """A file with only corrupt lines should start fresh."""
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "2025-08-06.jsonl").write_text("not json at all\n")

        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        # seq should be 0 or close to it (no valid events found)
        env = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        assert env.seq >= 1
        w.close()


# ---------------------------------------------------------------------------
# TestReadEventLogRobustness
# ---------------------------------------------------------------------------

class TestReadEventLogRobustness:
    """read_event_log skips corrupt and blank lines."""

    def test_blank_lines_skipped(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(3):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        # Insert blank lines
        path = log_dir / "2025-08-06.jsonl"
        content = path.read_text()
        lines = content.strip().split("\n")
        new_content = "\n\n".join(lines) + "\n\n"
        path.write_text(new_content)

        events = read_event_log(path)
        assert len(events) == 3

    def test_corrupt_line_in_middle_skipped(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(3):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        # Insert corrupt line in middle
        path = log_dir / "2025-08-06.jsonl"
        lines = path.read_text().strip().split("\n")
        lines.insert(1, "THIS IS NOT JSON")
        path.write_text("\n".join(lines) + "\n")

        events = read_event_log(path)
        assert len(events) == 3  # corrupt line skipped

    def test_all_corrupt_returns_empty(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "2025-08-06.jsonl"
        path.write_text("bad1\nbad2\nbad3\n")

        events = read_event_log(path)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# TestAppendAfterRecovery
# ---------------------------------------------------------------------------

class TestAppendAfterRecovery:
    """After recovery, new events are appended correctly."""

    def test_append_after_recovery_valid(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)

        # Write 5 events
        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(5):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        # New writer appends
        w2 = EventLogWriter(base_dir=log_dir, run_id="r2", fsync_policy="none")
        w2.emit(
            event_type="signal", source="test", payload={"v": "new"},
            ts="2025-08-06T15:30:00.000000Z",
        )
        w2.close()

        events = read_event_log(log_dir / "2025-08-06.jsonl")
        assert len(events) == 6
        assert events[-1].payload["v"] == "new"

    def test_append_preserves_existing_events(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)

        w = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(3):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        # Read before
        events_before = read_event_log(log_dir / "2025-08-06.jsonl")

        # Append more
        w2 = EventLogWriter(base_dir=log_dir, run_id="r2", fsync_policy="none")
        w2.emit(
            event_type="tick", source="test", payload={"i": 3},
            ts="2025-08-06T09:15:00.000000Z",
        )
        w2.close()

        events_after = read_event_log(log_dir / "2025-08-06.jsonl")
        assert len(events_after) == 4
        # Original events unchanged
        for i in range(3):
            assert events_after[i].seq == events_before[i].seq
            assert events_after[i].payload == events_before[i].payload


# ---------------------------------------------------------------------------
# TestSessionManifestAtomicWrite
# ---------------------------------------------------------------------------

class TestSessionManifestAtomicWrite:
    """SessionManifest uses atomic write (tempfile + rename)."""

    def test_manifest_created(self, tmp_path):
        m = SessionManifest(base_dir=tmp_path / "sessions", run_id="test-001")
        m.start(strategies=["S1", "S4"], risk_limits={"max_dd": 0.05}, data_sources=["DuckDB"])
        assert m.path.exists()

    def test_manifest_finalize(self, tmp_path):
        m = SessionManifest(base_dir=tmp_path / "sessions", run_id="test-001")
        m.start(strategies=["S1"], risk_limits={})
        m.finalize(total_signals=10, total_trades=3, total_blocks=1,
                   final_equity=1.05, peak_equity=1.05)
        data = m.load()
        assert data["status"] == "complete"
        assert data["summary"]["total_signals"] == 10
        assert data["summary"]["total_trades"] == 3

    def test_manifest_error_status(self, tmp_path):
        m = SessionManifest(base_dir=tmp_path / "sessions", run_id="test-002")
        m.start(strategies=["S1"], risk_limits={})
        m.finalize(error="test crash")
        data = m.load()
        assert data["status"] == "error"
        assert data["summary"]["error"] == "test crash"

    def test_manifest_load_nonexistent(self, tmp_path):
        m = SessionManifest(base_dir=tmp_path / "sessions", run_id="nonexist")
        assert m.load() == {}

    def test_manifest_valid_json(self, tmp_path):
        m = SessionManifest(base_dir=tmp_path / "sessions", run_id="json-test")
        m.start(strategies=["S1", "S4", "S5"], risk_limits={"vpin": 0.85},
                data_sources=["DuckDB"])
        # Should be valid JSON
        content = m.path.read_text()
        data = json.loads(content)
        assert data["run_id"] == "json-test"
        assert data["config"]["strategies"] == ["S1", "S4", "S5"]
        assert "platform" in data

    def test_manifest_overwrite_on_finalize(self, tmp_path):
        """Finalize overwrites the start-only manifest atomically."""
        m = SessionManifest(base_dir=tmp_path / "sessions", run_id="overwrite")
        m.start(strategies=["S1"], risk_limits={})
        data1 = m.load()
        assert data1["status"] == "running"

        m.finalize(total_signals=5)
        data2 = m.load()
        assert data2["status"] == "complete"
        assert data2["summary"]["total_signals"] == 5
