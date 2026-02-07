"""EventLogWriter — append-only JSONL persistence for BRAHMASTRA.

Writes EventEnvelopes to daily-rotated JSONL files:
  data/events/YYYY-MM-DD.jsonl

Features:
  - Single writer, monotonic seq allocation
  - Append-only, newline-delimited JSON
  - Atomic append (configurable fsync policy)
  - Daily rotation (new file per calendar day)
  - Optional rolling hash chain (written to .sha256 sidecar)
  - Crash recovery: detects truncated last line on open

Thread safety: NOT thread-safe. Use from a single asyncio task
(the WalTap) or protect with a lock externally.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from qlx.events.envelope import EventEnvelope
from qlx.events.serde import serialize_envelope, deserialize_envelope
from qlx.events.hashing import chain_hash, GENESIS

logger = logging.getLogger(__name__)


class EventLogWriter:
    """Append-only JSONL event log with daily rotation.

    Parameters
    ----------
    base_dir : Path
        Directory for event log files (e.g. ``data/events/``).
    run_id : str
        Stable identifier for this session.
    fsync_policy : str
        "every" = fsync after every event (safest, slowest).
        "batch" = fsync after flush() is called (default).
        "none" = never fsync (fastest, risk of data loss on crash).
    enable_hash_chain : bool
        If True, maintain a rolling SHA-256 hash chain and write
        hashes to a .sha256 sidecar file.
    """

    def __init__(
        self,
        base_dir: Path | str = Path("data/events"),
        run_id: str = "",
        fsync_policy: str = "batch",
        enable_hash_chain: bool = False,
    ):
        self._base_dir = Path(base_dir)
        self._run_id = run_id
        self._fsync_policy = fsync_policy
        self._enable_hash_chain = enable_hash_chain

        self._seq = 0
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._file: open | None = None
        self._hash_file: open | None = None
        self._last_hash: str = GENESIS
        self._event_count: int = 0
        self._bytes_written: int = 0

        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Recover seq from existing files
        self._recover_seq()

    def _recover_seq(self) -> None:
        """Scan existing JSONL files to recover the highest seq number.

        Also detects and handles truncated last lines.
        """
        jsonl_files = sorted(self._base_dir.glob("*.jsonl"))
        if not jsonl_files:
            return

        latest = jsonl_files[-1]
        try:
            with open(latest, "r") as f:
                lines = f.readlines()
        except OSError:
            return

        if not lines:
            return

        # Check for truncated last line (no trailing newline or invalid JSON)
        last_line = lines[-1].rstrip("\n")
        if last_line:
            try:
                env = deserialize_envelope(last_line)
                self._seq = env.seq
            except ValueError:
                # Truncated last line — mark it
                logger.warning(
                    "Truncated last line in %s, marking as corrupt_tail", latest,
                )
                with open(latest, "a") as f:
                    f.write("\n")  # ensure file ends with newline
                # Try second-to-last line
                if len(lines) >= 2:
                    try:
                        env = deserialize_envelope(lines[-2].rstrip("\n"))
                        self._seq = env.seq
                    except ValueError:
                        pass

        # Recover hash chain state if enabled
        if self._enable_hash_chain:
            hash_file = latest.with_suffix(".sha256")
            if hash_file.exists():
                try:
                    hash_lines = hash_file.read_text().strip().split("\n")
                    if hash_lines:
                        self._last_hash = hash_lines[-1]
                except OSError:
                    pass

    def next_seq(self) -> int:
        """Allocate the next monotonic sequence number."""
        self._seq += 1
        return self._seq

    def _ensure_file(self, ts: str) -> None:
        """Open or rotate to the correct daily file."""
        day = ts[:10]  # "YYYY-MM-DD"
        if day == self._current_date and self._file is not None:
            return

        # Close previous file
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._hash_file is not None:
            self._hash_file.close()
            self._hash_file = None

        self._current_date = day
        path = self._base_dir / f"{day}.jsonl"
        self._file = open(path, "a", encoding="utf-8")

        if self._enable_hash_chain:
            hash_path = self._base_dir / f"{day}.sha256"
            self._hash_file = open(hash_path, "a", encoding="utf-8")
            # Reset hash chain to GENESIS for new day (each day is independent)
            self._last_hash = GENESIS

        logger.info("EventLog opened: %s", path)

    def append(self, envelope: EventEnvelope) -> None:
        """Append a single event to the log.

        The envelope must have seq and run_id already set.
        """
        line = serialize_envelope(envelope)
        self._ensure_file(envelope.ts)

        self._file.write(line + "\n")
        self._event_count += 1
        self._bytes_written += len(line) + 1

        if self._enable_hash_chain:
            h = chain_hash(self._last_hash, line)
            self._hash_file.write(h + "\n")
            self._last_hash = h

        if self._fsync_policy == "every":
            self._do_fsync()

    def emit(
        self,
        event_type: str,
        source: str,
        payload: dict,
        strategy_id: str = "",
        symbol: str = "",
        ts: str | None = None,
    ) -> EventEnvelope:
        """Convenience: allocate seq, create envelope, and append.

        Returns the created EventEnvelope.
        """
        seq = self.next_seq()
        envelope = EventEnvelope.create(
            seq=seq,
            run_id=self._run_id,
            event_type=event_type,
            source=source,
            payload=payload,
            strategy_id=strategy_id,
            symbol=symbol,
            ts=ts,
        )
        self.append(envelope)
        return envelope

    def flush(self) -> None:
        """Flush buffered writes and optionally fsync."""
        if self._file is not None:
            self._file.flush()
            if self._fsync_policy in ("batch", "every"):
                self._do_fsync()
        if self._hash_file is not None:
            self._hash_file.flush()

    def _do_fsync(self) -> None:
        """Force OS-level sync to disk."""
        if self._file is not None:
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass
        if self._hash_file is not None:
            try:
                os.fsync(self._hash_file.fileno())
            except OSError:
                pass

    def close(self) -> None:
        """Flush and close all files."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._hash_file is not None:
            self._hash_file.close()
            self._hash_file = None
        logger.info(
            "EventLog closed: %d events, %d bytes",
            self._event_count, self._bytes_written,
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def event_count(self) -> int:
        return self._event_count

    def stats(self) -> dict:
        return {
            "run_id": self._run_id,
            "seq": self._seq,
            "event_count": self._event_count,
            "bytes_written": self._bytes_written,
            "current_date": self._current_date,
            "hash_chain_enabled": self._enable_hash_chain,
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Reader (for replay / verification)
# ---------------------------------------------------------------------------

def read_event_log(path: Path | str) -> list[EventEnvelope]:
    """Read all events from a JSONL file.

    Skips blank lines and lines that fail to parse (with a warning).
    """
    path = Path(path)
    events: list[EventEnvelope] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                events.append(deserialize_envelope(line))
            except ValueError as e:
                logger.warning("Skipping corrupt line %d in %s: %s", lineno, path, e)
    return events
