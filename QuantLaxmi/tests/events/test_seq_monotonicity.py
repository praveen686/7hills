"""Phase 2 — Sequence monotonicity tests.

Invariants:
  1. Seq is monotonically increasing (no gaps, no duplicates, no reorder)
  2. Seq starts from 0 and increments by 1 on each emit
  3. Recovery from existing files continues seq numbering
  4. Multiple emit() calls produce strictly ordered seq values
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.live.event_log import EventLogWriter, read_event_log
from core.events.envelope import EventEnvelope
from core.events.serde import serialize_envelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "events"


# ---------------------------------------------------------------------------
# TestMonotonicSeq
# ---------------------------------------------------------------------------

class TestMonotonicSeq:
    """Sequence numbers are strictly monotonically increasing."""

    def test_first_event_seq_1(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="seq-test", fsync_policy="none")
        env = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        assert env.seq == 1
        w.close()

    def test_sequential_increment(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="seq-test", fsync_policy="none")
        seqs = []
        for i in range(20):
            env = w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
            seqs.append(env.seq)
        w.close()
        assert seqs == list(range(1, 21))

    def test_no_gaps(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="seq-test", fsync_policy="none")
        seqs = []
        for i in range(50):
            env = w.emit(
                event_type="signal" if i % 2 == 0 else "tick",
                source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
            seqs.append(env.seq)
        w.close()
        for i in range(1, len(seqs)):
            assert seqs[i] == seqs[i - 1] + 1, f"Gap at index {i}"

    def test_no_duplicates(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="seq-test", fsync_policy="none")
        seqs = set()
        for i in range(100):
            env = w.emit(
                event_type="tick", source="test", payload={},
                ts="2025-08-06T09:15:00.000000Z",
            )
            assert env.seq not in seqs, f"Duplicate seq {env.seq}"
            seqs.add(env.seq)
        w.close()
        assert len(seqs) == 100

    def test_seq_across_day_rotation(self, log_dir):
        """Seq continues incrementing across daily file rotation."""
        w = EventLogWriter(base_dir=log_dir, run_id="seq-test", fsync_policy="none")
        env1 = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        env2 = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-07T09:15:00.000000Z",
        )
        env3 = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-08T09:15:00.000000Z",
        )
        w.close()
        assert env1.seq == 1
        assert env2.seq == 2
        assert env3.seq == 3


# ---------------------------------------------------------------------------
# TestSeqRecovery
# ---------------------------------------------------------------------------

class TestSeqRecovery:
    """Seq recovery from existing JSONL files."""

    def test_recovery_continues_from_last(self, log_dir):
        """A new writer on the same dir resumes seq from last event."""
        w1 = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(5):
            w1.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w1.close()

        w2 = EventLogWriter(base_dir=log_dir, run_id="r2", fsync_policy="none")
        env = w2.emit(
            event_type="tick", source="test", payload={"i": 5},
            ts="2025-08-06T09:15:00.000000Z",
        )
        assert env.seq == 6
        w2.close()

    def test_recovery_from_multi_day_files(self, log_dir):
        """Recovery scans the latest file, even across days."""
        w1 = EventLogWriter(base_dir=log_dir, run_id="r1", fsync_policy="none")
        for i in range(3):
            w1.emit(
                event_type="tick", source="test", payload={},
                ts="2025-08-06T09:15:00.000000Z",
            )
        for i in range(7):
            w1.emit(
                event_type="tick", source="test", payload={},
                ts="2025-08-07T09:15:00.000000Z",
            )
        w1.close()

        w2 = EventLogWriter(base_dir=log_dir, run_id="r2", fsync_policy="none")
        env = w2.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-07T09:15:00.000000Z",
        )
        assert env.seq == 11
        w2.close()

    def test_recovery_from_empty_dir(self, log_dir):
        """Fresh dir starts at seq=0 (first emit gives seq=1)."""
        w = EventLogWriter(base_dir=log_dir, run_id="fresh", fsync_policy="none")
        assert w.seq == 0
        env = w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        assert env.seq == 1
        w.close()


# ---------------------------------------------------------------------------
# TestSeqInPersistedFile
# ---------------------------------------------------------------------------

class TestSeqInPersistedFile:
    """Persisted JSONL lines contain correct seq values."""

    def test_written_seq_matches_returned(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="persist", fsync_policy="none")
        returned_seqs = []
        for i in range(10):
            env = w.emit(
                event_type="signal", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
            returned_seqs.append(env.seq)
        w.close()

        events = read_event_log(log_dir / "2025-08-06.jsonl")
        file_seqs = [e.seq for e in events]
        assert returned_seqs == file_seqs

    def test_persisted_seq_monotonic(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="persist", fsync_policy="none")
        for i in range(25):
            w.emit(
                event_type="tick", source="test", payload={},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        events = read_event_log(log_dir / "2025-08-06.jsonl")
        for i in range(1, len(events)):
            assert events[i].seq > events[i - 1].seq


# ---------------------------------------------------------------------------
# TestNextSeqAllocator
# ---------------------------------------------------------------------------

class TestNextSeqAllocator:
    """next_seq() allocates monotonically."""

    def test_next_seq_starts_at_1(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="alloc", fsync_policy="none")
        assert w.next_seq() == 1
        w.close()

    def test_next_seq_increments(self, log_dir):
        w = EventLogWriter(base_dir=log_dir, run_id="alloc", fsync_policy="none")
        s1 = w.next_seq()
        s2 = w.next_seq()
        s3 = w.next_seq()
        assert s1 == 1
        assert s2 == 2
        assert s3 == 3
        w.close()
