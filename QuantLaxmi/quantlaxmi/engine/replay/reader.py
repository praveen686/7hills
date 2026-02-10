"""WAL Reader — stream EventEnvelopes from persisted JSONL logs.

Reads data/events/YYYY-MM-DD.jsonl files and yields EventEnvelopes
in order (by seq within a file, chronological across files).

Validation:
  - Monotonic seq (gaps are warnings, duplicates are errors)
  - Optional hash chain verification against .sha256 sidecar
  - Truncated-tail handling (skip corrupt last line, warn)

Usage::

    reader = WalReader(base_dir=Path("data/events"))
    for envelope in reader.read_range("2025-08-06", "2025-12-31"):
        process(envelope)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.hashing import verify_chain
from quantlaxmi.core.events.serde import deserialize_envelope, serialize_envelope

logger = logging.getLogger(__name__)


class WalValidationError(Exception):
    """Raised when WAL validation fails (seq gap, hash mismatch, etc.)."""


class WalReader:
    """Read EventEnvelopes from JSONL event logs.

    Parameters
    ----------
    base_dir : Path
        Directory containing YYYY-MM-DD.jsonl files.
    strict_seq : bool
        If True, raise on seq gaps or duplicates. If False, warn.
    verify_hashes : bool
        If True, verify hash chain against .sha256 sidecar (if present).
    """

    def __init__(
        self,
        base_dir: Path | str = Path("data/events"),
        strict_seq: bool = False,
        verify_hashes: bool = False,
    ):
        self._base_dir = Path(base_dir)
        self._strict_seq = strict_seq
        self._verify_hashes = verify_hashes

        # Stats
        self._files_read: int = 0
        self._events_read: int = 0
        self._corrupt_lines: int = 0
        self._seq_gaps: int = 0
        self._hash_failures: int = 0

    def available_dates(self) -> list[str]:
        """Return sorted list of dates with JSONL files."""
        files = sorted(self._base_dir.glob("*.jsonl"))
        return [f.stem for f in files]

    def read_date(self, day: str) -> list[EventEnvelope]:
        """Read all events from a single day's JSONL file.

        Parameters
        ----------
        day : str
            Date in YYYY-MM-DD format.

        Returns
        -------
        list[EventEnvelope]
            Events in order, skipping corrupt lines.
        """
        path = self._base_dir / f"{day}.jsonl"
        if not path.exists():
            logger.debug("No event log for %s", day)
            return []

        lines = self._read_lines(path)
        events = self._parse_lines(lines, path)

        # Optional hash chain verification
        if self._verify_hashes:
            self._check_hash_chain(day, lines)

        self._files_read += 1
        return events

    def read_range(
        self,
        start: str,
        end: str,
    ) -> list[EventEnvelope]:
        """Read events across a date range (inclusive).

        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD).
        end : str
            End date (YYYY-MM-DD).

        Returns
        -------
        list[EventEnvelope]
            All events across the range, in order.
        """
        all_events: list[EventEnvelope] = []
        d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)

        while d <= end_d:
            day_str = d.isoformat()
            events = self.read_date(day_str)
            all_events.extend(events)
            d += timedelta(days=1)

        # Validate monotonic seq across all events
        self._validate_seq(all_events)

        return all_events

    def read_all(self) -> list[EventEnvelope]:
        """Read all available event logs in chronological order."""
        dates = self.available_dates()
        if not dates:
            return []
        return self.read_range(dates[0], dates[-1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_lines(self, path: Path) -> list[str]:
        """Read non-empty lines from a JSONL file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Cannot read %s: %s", path, e)
            return []

        lines = []
        for line in text.split("\n"):
            line = line.rstrip()
            if line:
                lines.append(line)
        return lines

    def _parse_lines(self, lines: list[str], path: Path) -> list[EventEnvelope]:
        """Parse JSONL lines into EventEnvelopes, skipping corrupt."""
        events: list[EventEnvelope] = []
        for lineno, line in enumerate(lines, 1):
            try:
                env = deserialize_envelope(line)
                events.append(env)
                self._events_read += 1
            except (ValueError, KeyError) as e:
                self._corrupt_lines += 1
                logger.warning(
                    "Skipping corrupt line %d in %s: %s", lineno, path, e,
                )
        return events

    def _validate_seq(self, events: list[EventEnvelope]) -> None:
        """Validate monotonic seq across events."""
        if len(events) < 2:
            return

        for i in range(1, len(events)):
            prev_seq = events[i - 1].seq
            curr_seq = events[i].seq

            if curr_seq <= prev_seq:
                self._seq_gaps += 1
                msg = (
                    f"Seq not monotonic: seq={curr_seq} after seq={prev_seq} "
                    f"(event_type={events[i].event_type})"
                )
                if self._strict_seq:
                    raise WalValidationError(msg)
                logger.warning(msg)

            elif curr_seq > prev_seq + 1:
                self._seq_gaps += 1
                msg = f"Seq gap: {prev_seq} -> {curr_seq} (missing {curr_seq - prev_seq - 1})"
                if self._strict_seq:
                    raise WalValidationError(msg)
                logger.warning(msg)

    def _check_hash_chain(self, day: str, lines: list[str]) -> None:
        """Verify hash chain against .sha256 sidecar."""
        hash_path = self._base_dir / f"{day}.sha256"
        if not hash_path.exists():
            logger.debug("No hash sidecar for %s, skipping verification", day)
            return

        try:
            hash_text = hash_path.read_text(encoding="utf-8")
            hashes = [h.rstrip() for h in hash_text.split("\n") if h.rstrip()]
        except OSError as e:
            logger.error("Cannot read hash sidecar %s: %s", hash_path, e)
            return

        valid, break_idx = verify_chain(lines, hashes)
        if not valid:
            self._hash_failures += 1
            msg = f"Hash chain broken for {day} at line {break_idx}"
            if self._strict_seq:
                raise WalValidationError(msg)
            logger.error(msg)
        else:
            logger.debug("Hash chain verified for %s (%d events)", day, len(lines))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_type(
        events: list[EventEnvelope],
        event_type: str,
    ) -> list[EventEnvelope]:
        """Filter events by event_type."""
        return [e for e in events if e.event_type == event_type]

    @staticmethod
    def filter_by_strategy(
        events: list[EventEnvelope],
        strategy_id: str,
    ) -> list[EventEnvelope]:
        """Filter events by strategy_id."""
        return [e for e in events if e.strategy_id == strategy_id]

    @staticmethod
    def filter_by_symbol(
        events: list[EventEnvelope],
        symbol: str,
    ) -> list[EventEnvelope]:
        """Filter events by symbol."""
        return [e for e in events if e.symbol == symbol]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "files_read": self._files_read,
            "events_read": self._events_read,
            "corrupt_lines": self._corrupt_lines,
            "seq_gaps": self._seq_gaps,
            "hash_failures": self._hash_failures,
        }
