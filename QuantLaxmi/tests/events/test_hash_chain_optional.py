"""Phase 2 — Hash chain tests.

Invariants:
  1. Chain hash is deterministic (same input → same output)
  2. Genesis hash is stable across runs
  3. Tamper detection: modifying any line breaks the chain
  4. verify_chain returns (True, N) for valid chain
  5. compute_chain produces verifiable hashes
  6. EventLogWriter with hash chain enabled writes .sha256 sidecar
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.events.hashing import chain_hash, verify_chain, compute_chain, GENESIS
from engine.live.event_log import EventLogWriter, read_event_log
from core.events.serde import serialize_envelope


# ---------------------------------------------------------------------------
# TestGenesis
# ---------------------------------------------------------------------------

class TestGenesis:
    """Genesis hash is stable and well-known."""

    def test_genesis_value(self):
        expected = hashlib.sha256(b"BRAHMASTRA_GENESIS").hexdigest()
        assert GENESIS == expected

    def test_genesis_is_hex(self):
        assert len(GENESIS) == 64
        int(GENESIS, 16)  # should not raise

    def test_genesis_stable_across_calls(self):
        from core.events import hashing as h1
        from core.events import hashing as h2
        assert h1.GENESIS == h2.GENESIS


# ---------------------------------------------------------------------------
# TestChainHash
# ---------------------------------------------------------------------------

class TestChainHash:
    """chain_hash is deterministic and collision-resistant."""

    def test_deterministic(self):
        h1 = chain_hash(GENESIS, "line one")
        h2 = chain_hash(GENESIS, "line one")
        assert h1 == h2

    def test_different_lines_different_hashes(self):
        h1 = chain_hash(GENESIS, "line one")
        h2 = chain_hash(GENESIS, "line two")
        assert h1 != h2

    def test_different_prev_different_hashes(self):
        h1 = chain_hash(GENESIS, "same line")
        h2 = chain_hash("a" * 64, "same line")
        assert h1 != h2

    def test_output_is_hex64(self):
        h = chain_hash(GENESIS, "test")
        assert len(h) == 64
        int(h, 16)

    def test_chain_builds_correctly(self):
        """Manual chain: genesis → h1 → h2 → h3."""
        h1 = chain_hash(GENESIS, "line1")
        h2 = chain_hash(h1, "line2")
        h3 = chain_hash(h2, "line3")
        # All different
        assert len({GENESIS, h1, h2, h3}) == 4


# ---------------------------------------------------------------------------
# TestVerifyChain
# ---------------------------------------------------------------------------

class TestVerifyChain:
    """verify_chain detects tampering and validates correct chains."""

    def test_valid_chain(self):
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        valid, idx = verify_chain(lines, hashes)
        assert valid is True
        assert idx == 3

    def test_tampered_first_line(self):
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        # Tamper first line
        tampered = ["TAMPERED", "line2", "line3"]
        valid, idx = verify_chain(tampered, hashes)
        assert valid is False
        assert idx == 0

    def test_tampered_middle_line(self):
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        tampered = ["line1", "TAMPERED", "line3"]
        valid, idx = verify_chain(tampered, hashes)
        assert valid is False
        assert idx == 1

    def test_tampered_last_line(self):
        lines = ["line1", "line2", "line3"]
        hashes = compute_chain(lines)
        tampered = ["line1", "line2", "TAMPERED"]
        valid, idx = verify_chain(tampered, hashes)
        assert valid is False
        assert idx == 2

    def test_tampered_hash(self):
        lines = ["line1", "line2"]
        hashes = compute_chain(lines)
        hashes[1] = "0" * 64  # replace hash
        valid, idx = verify_chain(lines, hashes)
        assert valid is False
        assert idx == 1

    def test_empty_chain(self):
        valid, idx = verify_chain([], [])
        assert valid is True
        assert idx == 0

    def test_length_mismatch(self):
        lines = ["line1", "line2"]
        hashes = compute_chain(["line1"])
        valid, idx = verify_chain(lines, hashes)
        assert valid is False

    def test_single_line(self):
        lines = ["single"]
        hashes = compute_chain(lines)
        valid, idx = verify_chain(lines, hashes)
        assert valid is True
        assert idx == 1


# ---------------------------------------------------------------------------
# TestComputeChain
# ---------------------------------------------------------------------------

class TestComputeChain:
    """compute_chain produces correct hash sequences."""

    def test_length_matches_input(self):
        lines = [f"line{i}" for i in range(10)]
        hashes = compute_chain(lines)
        assert len(hashes) == 10

    def test_first_hash_uses_genesis(self):
        lines = ["first line"]
        hashes = compute_chain(lines)
        expected = chain_hash(GENESIS, "first line")
        assert hashes[0] == expected

    def test_chaining_correct(self):
        lines = ["a", "b", "c"]
        hashes = compute_chain(lines)
        assert hashes[0] == chain_hash(GENESIS, "a")
        assert hashes[1] == chain_hash(hashes[0], "b")
        assert hashes[2] == chain_hash(hashes[1], "c")

    def test_deterministic(self):
        lines = ["x", "y", "z"]
        h1 = compute_chain(lines)
        h2 = compute_chain(lines)
        assert h1 == h2

    def test_empty_input(self):
        assert compute_chain([]) == []


# ---------------------------------------------------------------------------
# TestEventLogHashChain
# ---------------------------------------------------------------------------

class TestEventLogHashChain:
    """EventLogWriter with hash chain enabled."""

    def test_hash_sidecar_created(self, tmp_path):
        log_dir = tmp_path / "events"
        w = EventLogWriter(
            base_dir=log_dir, run_id="hash-test",
            fsync_policy="none", enable_hash_chain=True,
        )
        w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        w.close()
        assert (log_dir / "2025-08-06.jsonl").exists()
        assert (log_dir / "2025-08-06.sha256").exists()

    def test_hash_sidecar_line_count_matches(self, tmp_path):
        log_dir = tmp_path / "events"
        w = EventLogWriter(
            base_dir=log_dir, run_id="hash-test",
            fsync_policy="none", enable_hash_chain=True,
        )
        for i in range(5):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        jsonl_lines = (log_dir / "2025-08-06.jsonl").read_text().strip().split("\n")
        hash_lines = (log_dir / "2025-08-06.sha256").read_text().strip().split("\n")
        assert len(jsonl_lines) == len(hash_lines) == 5

    def test_hash_chain_verifiable(self, tmp_path):
        log_dir = tmp_path / "events"
        w = EventLogWriter(
            base_dir=log_dir, run_id="hash-test",
            fsync_policy="none", enable_hash_chain=True,
        )
        for i in range(5):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        jsonl_lines = (log_dir / "2025-08-06.jsonl").read_text().strip().split("\n")
        hash_lines = (log_dir / "2025-08-06.sha256").read_text().strip().split("\n")

        valid, idx = verify_chain(jsonl_lines, hash_lines)
        assert valid is True
        assert idx == 5

    def test_no_hash_sidecar_when_disabled(self, tmp_path):
        log_dir = tmp_path / "events"
        w = EventLogWriter(
            base_dir=log_dir, run_id="no-hash",
            fsync_policy="none", enable_hash_chain=False,
        )
        w.emit(
            event_type="tick", source="test", payload={},
            ts="2025-08-06T09:15:00.000000Z",
        )
        w.close()
        assert (log_dir / "2025-08-06.jsonl").exists()
        assert not (log_dir / "2025-08-06.sha256").exists()

    def test_tamper_detection_via_sidecar(self, tmp_path):
        """Modify a JSONL line → hash chain verification fails."""
        log_dir = tmp_path / "events"
        w = EventLogWriter(
            base_dir=log_dir, run_id="tamper-test",
            fsync_policy="none", enable_hash_chain=True,
        )
        for i in range(3):
            w.emit(
                event_type="tick", source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
        w.close()

        # Tamper with middle line
        path = log_dir / "2025-08-06.jsonl"
        lines = path.read_text().strip().split("\n")
        lines[1] = lines[1].replace("1", "999")  # modify payload
        path.write_text("\n".join(lines) + "\n")

        hash_lines = (log_dir / "2025-08-06.sha256").read_text().strip().split("\n")
        tampered_lines = path.read_text().strip().split("\n")

        valid, break_idx = verify_chain(tampered_lines, hash_lines)
        assert valid is False
        assert break_idx == 1  # break at the tampered line
