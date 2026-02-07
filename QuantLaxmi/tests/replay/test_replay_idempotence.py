"""Phase 3 — Replay idempotence tests.

Tests that replaying a replay produces identical results:
  write(events) → read → write → read → compare ≡ identical

Also tests that the ReplayEngine and Comparator components are
idempotent when applied multiple times to the same event stream.

Core invariants:
  - serialize(deserialize(serialize(e))) == serialize(e)
  - compare(stream, stream) is always identical
  - write→read→write→read produces same events
  - ComparisonResult is stable across multiple comparisons
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.events.envelope import EventEnvelope
from core.events.serde import serialize_envelope, deserialize_envelope
from core.events.hashing import compute_chain, verify_chain

from engine.replay.reader import WalReader
from engine.replay.comparator import (
    ComparisonResult,
    compare_streams,
)
from engine.live.event_log import EventLogWriter, read_event_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = "2025-09-15T10:30:00.000000Z"
_RUN_ID = "idempotence-test-001"


def _make_events(n: int = 10) -> list[EventEnvelope]:
    """Generate a deterministic stream of n events."""
    events = []
    for i in range(1, n + 1):
        event_type = ["signal", "gate_decision", "order", "snapshot"][i % 4]
        payload = {
            "direction": "long" if i % 2 == 0 else "short",
            "conviction": round(0.5 + (i * 0.03), 4),
            "instrument_type": "FUT",
            "regime": "normal",
            "value": float(i) * 1.1,
        }
        events.append(EventEnvelope(
            ts=_TS,
            seq=i,
            run_id=_RUN_ID,
            event_type=event_type,
            source="test",
            strategy_id=f"s{i % 3}",
            symbol="NIFTY" if i % 2 == 0 else "BANKNIFTY",
            payload=payload,
        ))
    return events


def _write_and_read(events: list[EventEnvelope], base_dir: Path) -> list[EventEnvelope]:
    """Write events to JSONL, read them back."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "2025-09-15.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(serialize_envelope(e) + "\n")

    reader = WalReader(base_dir=base_dir)
    return reader.read_date("2025-09-15")


# ===================================================================
# Test Class: Serde Idempotence
# ===================================================================


class TestSerdeIdempotence:
    """serialize(deserialize(serialize(e))) == serialize(e)."""

    def test_single_event_triple_roundtrip(self):
        """3x roundtrip produces identical JSON."""
        e = _make_events(1)[0]
        s1 = serialize_envelope(e)
        e2 = deserialize_envelope(s1)
        s2 = serialize_envelope(e2)
        e3 = deserialize_envelope(s2)
        s3 = serialize_envelope(e3)
        assert s1 == s2 == s3

    def test_all_events_triple_roundtrip(self):
        """All events survive 3x roundtrip."""
        events = _make_events(20)
        for e in events:
            s1 = serialize_envelope(e)
            s2 = serialize_envelope(deserialize_envelope(s1))
            s3 = serialize_envelope(deserialize_envelope(s2))
            assert s1 == s2 == s3

    def test_nan_roundtrip(self):
        """NaN → null survives roundtrip."""
        e = EventEnvelope(
            ts=_TS, seq=1, run_id=_RUN_ID,
            event_type="signal", source="test",
            strategy_id="s1", symbol="NIFTY",
            payload={"conviction": float("nan"), "direction": "long"},
        )
        s1 = serialize_envelope(e)
        e2 = deserialize_envelope(s1)
        s2 = serialize_envelope(e2)
        assert s1 == s2
        # NaN became None/null
        assert e2.payload["conviction"] is None

    def test_inf_roundtrip(self):
        """Inf → null survives roundtrip."""
        e = EventEnvelope(
            ts=_TS, seq=1, run_id=_RUN_ID,
            event_type="signal", source="test",
            strategy_id="s1", symbol="NIFTY",
            payload={"value": float("inf"), "direction": "long"},
        )
        s1 = serialize_envelope(e)
        e2 = deserialize_envelope(s1)
        s2 = serialize_envelope(e2)
        assert s1 == s2
        assert e2.payload["value"] is None

    def test_nested_dict_roundtrip(self):
        """Nested dicts survive roundtrip."""
        e = EventEnvelope(
            ts=_TS, seq=1, run_id=_RUN_ID,
            event_type="signal", source="test",
            strategy_id="s1", symbol="NIFTY",
            payload={
                "components": {
                    "alpha": 0.5,
                    "beta": {"gamma": 1.2, "delta": [1, 2, 3]},
                },
                "direction": "long",
            },
        )
        s1 = serialize_envelope(e)
        s2 = serialize_envelope(deserialize_envelope(s1))
        assert s1 == s2

    def test_empty_payload_roundtrip(self):
        """Empty payload survives roundtrip."""
        e = EventEnvelope(
            ts=_TS, seq=1, run_id=_RUN_ID,
            event_type="snapshot", source="test",
            payload={},
        )
        s1 = serialize_envelope(e)
        s2 = serialize_envelope(deserialize_envelope(s1))
        assert s1 == s2


# ===================================================================
# Test Class: Write-Read Idempotence
# ===================================================================


class TestWriteReadIdempotence:
    """write(events) → read → write → read produces same events."""

    def test_single_cycle(self, tmp_path: Path):
        """Single write→read cycle preserves all events."""
        events = _make_events(10)
        loaded = _write_and_read(events, tmp_path / "cycle1")
        assert len(loaded) == len(events)
        for orig, read in zip(events, loaded):
            assert serialize_envelope(orig) == serialize_envelope(read)

    def test_double_cycle(self, tmp_path: Path):
        """Two write→read cycles produce identical events."""
        events = _make_events(10)

        # Cycle 1: write → read
        loaded1 = _write_and_read(events, tmp_path / "cycle1")

        # Cycle 2: write loaded1 → read
        loaded2 = _write_and_read(loaded1, tmp_path / "cycle2")

        # Compare: loaded1 == loaded2
        for e1, e2 in zip(loaded1, loaded2):
            assert serialize_envelope(e1) == serialize_envelope(e2)

    def test_triple_cycle(self, tmp_path: Path):
        """Three write→read cycles still identical."""
        events = _make_events(15)
        current = events
        for i in range(3):
            current = _write_and_read(current, tmp_path / f"cycle{i}")

        # Final read must match original
        for orig, final in zip(events, current):
            assert serialize_envelope(orig) == serialize_envelope(final)

    def test_event_log_writer_cycle(self, tmp_path: Path):
        """Using EventLogWriter instead of raw file write."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id="idem-test",
            fsync_policy="none",
        )

        # Write events
        for e in _make_events(8):
            writer.append(e)
        writer.close()

        # Read back
        reader = WalReader(base_dir=events_dir)
        loaded = reader.read_date("2025-09-15")
        assert len(loaded) == 8

        # Write again from loaded
        events_dir2 = tmp_path / "events2"
        writer2 = EventLogWriter(
            base_dir=events_dir2,
            run_id="idem-test",
            fsync_policy="none",
        )
        for e in loaded:
            writer2.append(e)
        writer2.close()

        # Read back again
        reader2 = WalReader(base_dir=events_dir2)
        loaded2 = reader2.read_date("2025-09-15")

        # Must be identical
        for e1, e2 in zip(loaded, loaded2):
            assert serialize_envelope(e1) == serialize_envelope(e2)

    def test_emit_cycle(self, tmp_path: Path):
        """EventLogWriter.emit() → read → compare."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id="emit-cycle",
            fsync_policy="none",
        )

        emitted = []
        for i in range(5):
            env = writer.emit(
                event_type="signal",
                source="test",
                payload={"conviction": 0.5 + i * 0.1, "direction": "long"},
                strategy_id=f"s{i}",
                symbol="NIFTY",
                ts=_TS,
            )
            emitted.append(env)
        writer.close()

        # Read back
        reader = WalReader(base_dir=events_dir)
        loaded = reader.read_date("2025-09-15")
        assert len(loaded) == 5

        # Compare payload content
        for orig, read in zip(emitted, loaded):
            assert orig.event_type == read.event_type
            assert orig.strategy_id == read.strategy_id
            assert orig.payload["conviction"] == read.payload["conviction"]
            assert orig.payload["direction"] == read.payload["direction"]


# ===================================================================
# Test Class: Comparator Idempotence
# ===================================================================


class TestComparatorIdempotence:
    """compare(stream, stream) is always identical, repeatedly."""

    def test_self_comparison_always_identical(self):
        """compare(X, X) == identical for any stream X."""
        streams = [
            _make_events(1),
            _make_events(10),
            _make_events(50),
            [],  # empty
        ]
        for stream in streams:
            result = compare_streams(stream, stream)
            assert result.identical, (
                f"Self-comparison failed for {len(stream)} events"
            )

    def test_comparison_deterministic(self):
        """Same inputs produce same ComparisonResult."""
        ref = _make_events(10)
        # Make a modified copy
        replay = _make_events(10)
        mod = EventEnvelope(
            ts=replay[5].ts, seq=replay[5].seq, run_id=replay[5].run_id,
            event_type=replay[5].event_type, source=replay[5].source,
            strategy_id=replay[5].strategy_id, symbol=replay[5].symbol,
            payload={**replay[5].payload, "conviction": 0.99},
        )
        replay_mod = replay[:5] + [mod] + replay[6:]

        result1 = compare_streams(ref, replay_mod)
        result2 = compare_streams(ref, replay_mod)

        assert result1.identical == result2.identical
        assert len(result1.diffs) == len(result2.diffs)
        assert result1.total_compared == result2.total_compared

    def test_comparison_commutative_detection(self):
        """compare(A, B) detects the same diff as compare(B, A)."""
        a = [EventEnvelope(
            ts=_TS, seq=1, run_id=_RUN_ID,
            event_type="signal", source="test",
            strategy_id="s1", symbol="NIFTY",
            payload={"direction": "long", "conviction": 0.80,
                     "instrument_type": "FUT", "regime": "normal"},
        )]
        b = [EventEnvelope(
            ts=_TS, seq=1, run_id=_RUN_ID,
            event_type="signal", source="test",
            strategy_id="s1", symbol="NIFTY",
            payload={"direction": "long", "conviction": 0.75,
                     "instrument_type": "FUT", "regime": "normal"},
        )]
        result_ab = compare_streams(a, b)
        result_ba = compare_streams(b, a)

        assert result_ab.identical == result_ba.identical
        assert len(result_ab.diffs) == len(result_ba.diffs)

    def test_to_dict_stable(self):
        """to_dict() output is stable across calls."""
        events = _make_events(10)
        result = compare_streams(events, events)

        d1 = json.dumps(result.to_dict(), sort_keys=True)
        d2 = json.dumps(result.to_dict(), sort_keys=True)
        assert d1 == d2

    def test_repeated_comparison_no_mutation(self):
        """Comparing multiple times doesn't mutate any events."""
        events = _make_events(10)
        original_serialized = [serialize_envelope(e) for e in events]

        for _ in range(5):
            compare_streams(events, events)

        after_serialized = [serialize_envelope(e) for e in events]
        assert original_serialized == after_serialized


# ===================================================================
# Test Class: Hash Chain Idempotence
# ===================================================================


class TestHashChainIdempotence:
    """Hash chain is stable across write-read cycles."""

    def test_hash_chain_write_read_stable(self, tmp_path: Path):
        """Hash chain from written events matches recomputed chain."""
        events = _make_events(10)
        lines = [serialize_envelope(e) for e in events]

        # Compute chain
        hashes = compute_chain(lines)

        # Verify chain
        valid, idx = verify_chain(lines, hashes)
        assert valid

        # Write to file, read back, recompute chain
        base_dir = tmp_path / "events"
        base_dir.mkdir(parents=True)
        path = base_dir / "2025-09-15.jsonl"
        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")

        # Read lines back
        with open(path, "r") as f:
            read_lines = [line.rstrip("\n") for line in f if line.rstrip("\n")]

        # Recompute chain from read lines
        hashes2 = compute_chain(read_lines)
        assert hashes == hashes2

    def test_hash_chain_idempotent_3x(self, tmp_path: Path):
        """3 compute_chain() calls on same input produce same result."""
        events = _make_events(20)
        lines = [serialize_envelope(e) for e in events]

        h1 = compute_chain(lines)
        h2 = compute_chain(lines)
        h3 = compute_chain(lines)
        assert h1 == h2 == h3

    def test_event_log_writer_hash_chain_stable(self, tmp_path: Path):
        """EventLogWriter with hash chain produces verifiable output."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id="hash-idem",
            fsync_policy="none",
            enable_hash_chain=True,
        )

        for e in _make_events(8):
            writer.append(e)
        writer.close()

        # Read JSONL
        jsonl_path = events_dir / "2025-09-15.jsonl"
        hash_path = events_dir / "2025-09-15.sha256"
        assert jsonl_path.exists()
        assert hash_path.exists()

        with open(jsonl_path) as f:
            lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        with open(hash_path) as f:
            hashes = [h.rstrip("\n") for h in f if h.rstrip("\n")]

        assert len(lines) == 8
        assert len(hashes) == 8

        # Verify chain
        valid, idx = verify_chain(lines, hashes)
        assert valid

        # Recompute and compare
        recomputed = compute_chain(lines)
        assert recomputed == hashes

    def test_wal_reader_hash_verification(self, tmp_path: Path):
        """WalReader with verify_hashes=True reads correctly."""
        events = _make_events(5)
        events_dir = tmp_path / "events"

        writer = EventLogWriter(
            base_dir=events_dir,
            run_id="hash-verify",
            fsync_policy="none",
            enable_hash_chain=True,
        )
        for e in events:
            writer.append(e)
        writer.close()

        reader = WalReader(
            base_dir=events_dir,
            verify_hashes=True,
        )
        loaded = reader.read_date("2025-09-15")
        assert len(loaded) == 5
        assert reader.stats()["hash_failures"] == 0
