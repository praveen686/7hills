"""Phase 3 — Replay hash chain verification tests.

Tests that:
  1. Hash chains produced by EventLogWriter are valid
  2. WalReader correctly verifies hash chains
  3. Tampered events break the hash chain
  4. Missing hash sidecar is handled gracefully
  5. Hash chain survives daily rotation
  6. Partial corruption detected at correct position

Core invariants:
  - EventLogWriter with enable_hash_chain=True produces verifiable .sha256
  - Any byte modification to a JSONL line breaks the chain from that point
  - verify_chain(lines, hashes) is the ground truth for integrity
  - Hash chain is independent of event content (works on raw bytes)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from qlx.events.envelope import EventEnvelope
from qlx.events.serde import serialize_envelope, deserialize_envelope
from qlx.events.hashing import (
    GENESIS,
    chain_hash,
    compute_chain,
    verify_chain,
)

from brahmastra.engine.event_log import EventLogWriter
from brahmastra.replay.reader import WalReader, WalValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = "hashchain-test"


def _make_event(seq: int, day: str = "2025-10-01") -> EventEnvelope:
    """Create a test event for a given day."""
    ts = f"{day}T09:15:00.000000Z"
    return EventEnvelope(
        ts=ts,
        seq=seq,
        run_id=_RUN_ID,
        event_type="signal",
        source="test",
        strategy_id=f"s{seq % 5}",
        symbol="NIFTY" if seq % 2 == 0 else "BANKNIFTY",
        payload={
            "direction": "long" if seq % 2 == 0 else "short",
            "conviction": round(0.5 + seq * 0.05, 4),
            "instrument_type": "FUT",
            "regime": "normal",
        },
    )


def _write_events_with_hashes(
    events: list[EventEnvelope],
    base_dir: Path,
    day: str = "2025-10-01",
) -> tuple[list[str], list[str]]:
    """Write events as JSONL with hash chain sidecar. Returns (lines, hashes)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    lines = [serialize_envelope(e) for e in events]
    hashes = compute_chain(lines)

    jsonl_path = base_dir / f"{day}.jsonl"
    hash_path = base_dir / f"{day}.sha256"

    with open(jsonl_path, "w") as f:
        for line in lines:
            f.write(line + "\n")
    with open(hash_path, "w") as f:
        for h in hashes:
            f.write(h + "\n")

    return lines, hashes


# ===================================================================
# Test Class: Hash Chain Basics
# ===================================================================


class TestHashChainBasics:
    """Fundamental hash chain properties."""

    def test_genesis_is_sha256(self):
        """GENESIS is SHA-256 of b'BRAHMASTRA_GENESIS'."""
        expected = hashlib.sha256(b"BRAHMASTRA_GENESIS").hexdigest()
        assert GENESIS == expected
        assert len(GENESIS) == 64

    def test_chain_hash_deterministic(self):
        """Same inputs produce same hash."""
        h1 = chain_hash(GENESIS, "hello")
        h2 = chain_hash(GENESIS, "hello")
        assert h1 == h2

    def test_chain_hash_depends_on_prev(self):
        """Different prev_hash produces different result."""
        h1 = chain_hash(GENESIS, "hello")
        h2 = chain_hash("a" * 64, "hello")
        assert h1 != h2

    def test_chain_hash_depends_on_line(self):
        """Different line produces different result."""
        h1 = chain_hash(GENESIS, "hello")
        h2 = chain_hash(GENESIS, "world")
        assert h1 != h2

    def test_compute_chain_length(self):
        """compute_chain returns one hash per line."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        assert len(hashes) == 3

    def test_compute_chain_first_uses_genesis(self):
        """First hash uses GENESIS as prev_hash."""
        lines = ["line1"]
        hashes = compute_chain(lines)
        expected = chain_hash(GENESIS, "line1")
        assert hashes[0] == expected

    def test_compute_chain_chaining(self):
        """Each hash chains from the previous."""
        lines = ["a", "b", "c"]
        hashes = compute_chain(lines)

        h0 = chain_hash(GENESIS, "a")
        h1 = chain_hash(h0, "b")
        h2 = chain_hash(h1, "c")

        assert hashes == [h0, h1, h2]

    def test_verify_valid_chain(self):
        """verify_chain returns True for valid chain."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        valid, idx = verify_chain(lines, hashes)
        assert valid
        assert idx == 3

    def test_verify_empty_chain(self):
        """Empty chain is valid."""
        valid, idx = verify_chain([], [])
        assert valid
        assert idx == 0

    def test_verify_single_element(self):
        """Single-element chain is valid."""
        lines = ["single line"]
        hashes = compute_chain(lines)
        valid, idx = verify_chain(lines, hashes)
        assert valid


# ===================================================================
# Test Class: Hash Chain Tamper Detection
# ===================================================================


class TestHashChainTamperDetection:
    """Tests that any modification breaks the chain."""

    def test_tamper_first_line(self):
        """Modifying first line breaks chain at index 0."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        tampered = ["LINE1", "line2", "line3"]
        valid, idx = verify_chain(tampered, hashes)
        assert not valid
        assert idx == 0

    def test_tamper_middle_line(self):
        """Modifying middle line breaks chain at that index."""
        lines = ["line1", "line2", "line3", "line4", "line5"]
        hashes = compute_chain(lines)
        tampered = lines.copy()
        tampered[2] = "TAMPERED"
        valid, idx = verify_chain(tampered, hashes)
        assert not valid
        assert idx == 2

    def test_tamper_last_line(self):
        """Modifying last line breaks chain at last index."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        tampered = lines.copy()
        tampered[-1] = "TAMPERED"
        valid, idx = verify_chain(tampered, hashes)
        assert not valid
        assert idx == 2

    def test_tamper_hash(self):
        """Modifying a hash breaks the chain."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        bad_hashes = hashes.copy()
        bad_hashes[1] = "0" * 64
        valid, idx = verify_chain(lines, bad_hashes)
        assert not valid
        assert idx == 1

    def test_deleted_line(self):
        """Removing a line causes length mismatch."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        short_lines = lines[:2]
        valid, idx = verify_chain(short_lines, hashes)
        assert not valid

    def test_inserted_line(self):
        """Inserting a line causes chain to break."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        extended = ["line1", "INSERTED", "line2", "line3"]
        valid, idx = verify_chain(extended, hashes)
        assert not valid

    def test_reordered_lines(self):
        """Reordering lines breaks the chain."""
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        reordered = ["line1", "line3", "line2"]
        valid, idx = verify_chain(reordered, hashes)
        assert not valid

    def test_tamper_serialized_event(self):
        """Tampering with a serialized event line breaks the chain."""
        events = [_make_event(i) for i in range(1, 6)]
        lines = [serialize_envelope(e) for e in events]
        hashes = compute_chain(lines)

        # Tamper: change conviction in line 3
        tampered = lines.copy()
        tampered[2] = tampered[2].replace('"conviction":', '"conviction":999.0,TAMPER"conviction":')
        valid, idx = verify_chain(tampered, hashes)
        assert not valid
        assert idx == 2

    def test_subtle_single_byte_tamper(self):
        """Even a single-byte change is detected."""
        events = [_make_event(i) for i in range(1, 4)]
        lines = [serialize_envelope(e) for e in events]
        hashes = compute_chain(lines)

        # Change one character
        original = lines[1]
        tampered_line = original[:-2] + "X" + original[-1:]
        tampered = [lines[0], tampered_line, lines[2]]
        valid, idx = verify_chain(tampered, hashes)
        assert not valid
        assert idx == 1


# ===================================================================
# Test Class: EventLogWriter Hash Chain
# ===================================================================


class TestEventLogWriterHashChain:
    """Tests that EventLogWriter produces valid hash chains."""

    def test_writer_creates_sidecar(self, tmp_path: Path):
        """EventLogWriter with hash chain creates .sha256 sidecar."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
            enable_hash_chain=True,
        )
        for e in [_make_event(i) for i in range(1, 6)]:
            writer.append(e)
        writer.close()

        assert (events_dir / "2025-10-01.jsonl").exists()
        assert (events_dir / "2025-10-01.sha256").exists()

    def test_writer_hash_chain_verifiable(self, tmp_path: Path):
        """Written hash chain passes verify_chain()."""
        events_dir = tmp_path / "events"
        events = [_make_event(i) for i in range(1, 11)]

        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
            enable_hash_chain=True,
        )
        for e in events:
            writer.append(e)
        writer.close()

        # Read lines and hashes
        with open(events_dir / "2025-10-01.jsonl") as f:
            lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        with open(events_dir / "2025-10-01.sha256") as f:
            hashes = [h.rstrip("\n") for h in f if h.rstrip("\n")]

        assert len(lines) == 10
        assert len(hashes) == 10

        valid, idx = verify_chain(lines, hashes)
        assert valid

    def test_writer_no_hash_chain_no_sidecar(self, tmp_path: Path):
        """EventLogWriter without hash chain does NOT create .sha256."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
            enable_hash_chain=False,
        )
        for e in [_make_event(i) for i in range(1, 4)]:
            writer.append(e)
        writer.close()

        assert (events_dir / "2025-10-01.jsonl").exists()
        assert not (events_dir / "2025-10-01.sha256").exists()

    def test_writer_hash_matches_recomputed(self, tmp_path: Path):
        """Writer-produced hashes match independently computed hashes."""
        events_dir = tmp_path / "events"
        events = [_make_event(i) for i in range(1, 8)]

        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
            enable_hash_chain=True,
        )
        for e in events:
            writer.append(e)
        writer.close()

        # Read lines
        with open(events_dir / "2025-10-01.jsonl") as f:
            lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        with open(events_dir / "2025-10-01.sha256") as f:
            writer_hashes = [h.rstrip("\n") for h in f if h.rstrip("\n")]

        # Recompute independently
        recomputed = compute_chain(lines)
        assert writer_hashes == recomputed

    def test_tamper_detection_after_write(self, tmp_path: Path):
        """Tampering with JSONL after write is detected by hash chain."""
        events_dir = tmp_path / "events"
        events = [_make_event(i) for i in range(1, 6)]

        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
            enable_hash_chain=True,
        )
        for e in events:
            writer.append(e)
        writer.close()

        # Tamper with line 3 in the JSONL file
        jsonl_path = events_dir / "2025-10-01.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()

        # Use "normal" (regime field) which appears in every event
        lines[2] = lines[2].replace("normal", "TAMPERED_REGIME")
        with open(jsonl_path, "w") as f:
            f.writelines(lines)

        # Verify hash chain detects tampering
        with open(jsonl_path) as f:
            tampered_lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
        with open(events_dir / "2025-10-01.sha256") as f:
            hashes = [h.rstrip("\n") for h in f if h.rstrip("\n")]

        valid, idx = verify_chain(tampered_lines, hashes)
        assert not valid
        assert idx == 2


# ===================================================================
# Test Class: WalReader Hash Verification
# ===================================================================


class TestWalReaderHashVerification:
    """Tests WalReader's hash chain verification mode."""

    def test_valid_hashes_no_error(self, tmp_path: Path):
        """WalReader with verify_hashes reads valid chain without error."""
        events = [_make_event(i) for i in range(1, 6)]
        _write_events_with_hashes(events, tmp_path, "2025-10-01")

        reader = WalReader(base_dir=tmp_path, verify_hashes=True)
        loaded = reader.read_date("2025-10-01")
        assert len(loaded) == 5
        assert reader.stats()["hash_failures"] == 0

    def test_tampered_hashes_logged(self, tmp_path: Path):
        """WalReader detects tampered hash chain (non-strict)."""
        events = [_make_event(i) for i in range(1, 6)]
        lines, hashes = _write_events_with_hashes(events, tmp_path, "2025-10-01")

        # Tamper with JSONL — use "normal" which is in every event's regime
        jsonl_path = tmp_path / "2025-10-01.jsonl"
        with open(jsonl_path) as f:
            content = f.readlines()
        content[1] = content[1].replace("normal", "TAMPERED_REGIME")
        with open(jsonl_path, "w") as f:
            f.writelines(content)

        reader = WalReader(base_dir=tmp_path, verify_hashes=True)
        loaded = reader.read_date("2025-10-01")
        # Events still returned (non-strict mode)
        assert len(loaded) == 5  # all lines still valid JSON
        assert reader.stats()["hash_failures"] == 1

    def test_missing_sidecar_graceful(self, tmp_path: Path):
        """WalReader gracefully handles missing .sha256 sidecar."""
        events = [_make_event(i) for i in range(1, 4)]
        tmp_path.mkdir(parents=True, exist_ok=True)

        # Write JSONL only (no .sha256)
        lines = [serialize_envelope(e) for e in events]
        jsonl_path = tmp_path / "2025-10-01.jsonl"
        with open(jsonl_path, "w") as f:
            for line in lines:
                f.write(line + "\n")

        reader = WalReader(base_dir=tmp_path, verify_hashes=True)
        loaded = reader.read_date("2025-10-01")
        assert len(loaded) == 3
        assert reader.stats()["hash_failures"] == 0  # no sidecar = skip

    def test_strict_mode_tampered_raises(self, tmp_path: Path):
        """WalReader strict_seq + verify_hashes + tampering raises."""
        events = [_make_event(i) for i in range(1, 4)]
        _write_events_with_hashes(events, tmp_path, "2025-10-01")

        # Tamper
        jsonl_path = tmp_path / "2025-10-01.jsonl"
        with open(jsonl_path) as f:
            content = f.readlines()
        content[0] = content[0].replace("NIFTY", "TAMPERED_NIFTY")
        with open(jsonl_path, "w") as f:
            f.writelines(content)

        reader = WalReader(
            base_dir=tmp_path,
            strict_seq=True,
            verify_hashes=True,
        )
        with pytest.raises(WalValidationError):
            reader.read_date("2025-10-01")


# ===================================================================
# Test Class: Multi-Day Hash Chain
# ===================================================================


class TestMultiDayHashChain:
    """Hash chain with daily rotation."""

    def test_two_day_hash_chains_independent(self, tmp_path: Path):
        """Each day has its own hash chain starting from GENESIS."""
        events_day1 = [_make_event(i, "2025-10-01") for i in range(1, 4)]
        events_day2 = [_make_event(i, "2025-10-02") for i in range(4, 7)]

        _write_events_with_hashes(events_day1, tmp_path, "2025-10-01")
        _write_events_with_hashes(events_day2, tmp_path, "2025-10-02")

        # Read and verify each day independently
        reader = WalReader(base_dir=tmp_path, verify_hashes=True)
        day1 = reader.read_date("2025-10-01")
        assert len(day1) == 3

        reader2 = WalReader(base_dir=tmp_path, verify_hashes=True)
        day2 = reader2.read_date("2025-10-02")
        assert len(day2) == 3

        assert reader.stats()["hash_failures"] == 0
        assert reader2.stats()["hash_failures"] == 0

    def test_event_log_writer_rotation_hashes(self, tmp_path: Path):
        """EventLogWriter rotation creates separate hash chains per day."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
            enable_hash_chain=True,
        )

        # Day 1 events
        for i in range(1, 4):
            writer.append(_make_event(i, "2025-10-01"))

        # Day 2 events
        for i in range(4, 7):
            writer.append(_make_event(i, "2025-10-02"))

        writer.close()

        # Both JSONL and SHA256 files exist for each day
        assert (events_dir / "2025-10-01.jsonl").exists()
        assert (events_dir / "2025-10-01.sha256").exists()
        assert (events_dir / "2025-10-02.jsonl").exists()
        assert (events_dir / "2025-10-02.sha256").exists()

        # Verify each day's chain
        for day in ["2025-10-01", "2025-10-02"]:
            with open(events_dir / f"{day}.jsonl") as f:
                lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
            with open(events_dir / f"{day}.sha256") as f:
                hashes = [h.rstrip("\n") for h in f if h.rstrip("\n")]

            valid, idx = verify_chain(lines, hashes)
            assert valid, f"Hash chain failed for {day} at index {idx}"

    def test_range_read_with_hash_verification(self, tmp_path: Path):
        """Reading a date range with hash verification."""
        for day_num in range(1, 4):
            day = f"2025-10-0{day_num}"
            events = [_make_event(i, day) for i in range(1, 4)]
            _write_events_with_hashes(events, tmp_path, day)

        reader = WalReader(base_dir=tmp_path, verify_hashes=True)
        all_events = reader.read_range("2025-10-01", "2025-10-03")
        assert len(all_events) == 9  # 3 events × 3 days
        assert reader.stats()["hash_failures"] == 0
        assert reader.stats()["files_read"] == 3
