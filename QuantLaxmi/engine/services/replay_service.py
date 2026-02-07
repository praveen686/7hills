"""Replay Service — time-travel queries over the persisted WAL.

Thin service wrapping WalReader with replay-specific queries:
  - available_dates: list dates with event data
  - snapshot_at: nearest SNAPSHOT at or before a given timestamp
  - timeline: filterable event markers for timeline visualization
  - step: advance playback window, returning events + snapshot

Thread-safe: each call creates a fresh WalReader (same pattern as WalQueryService).
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.events.envelope import EventEnvelope
from core.events.types import EventType
from engine.replay.reader import WalReader

logger = logging.getLogger(__name__)

# Event types shown on the replay timeline (excludes raw ticks/bars)
_TIMELINE_TYPES = frozenset({
    EventType.SIGNAL.value,
    EventType.GATE_DECISION.value,
    EventType.ORDER.value,
    EventType.FILL.value,
    EventType.RISK_ALERT.value,
    EventType.SNAPSHOT.value,
})

DEFAULT_WAL_DIR = Path("data/events")


class ReplayService:
    """Read-only replay queries over the persisted event WAL.

    Thread-safe: each call creates a fresh WalReader (no shared mutable state).
    """

    def __init__(self, base_dir: Path | str = DEFAULT_WAL_DIR):
        self._base_dir = Path(base_dir)

    def _reader(self) -> WalReader:
        return WalReader(base_dir=self._base_dir)

    # ------------------------------------------------------------------
    # available_dates
    # ------------------------------------------------------------------

    def available_dates(self) -> list[str]:
        """Return sorted list of dates with JSONL event files."""
        return self._reader().available_dates()

    # ------------------------------------------------------------------
    # snapshot_at — nearest SNAPSHOT at or before a given timestamp
    # ------------------------------------------------------------------

    def snapshot_at(self, ts: str, day: str) -> dict | None:
        """Return the last SNAPSHOT event with ts <= target.

        Parameters
        ----------
        ts : str
            Target timestamp (ISO-8601).
        day : str
            Date in YYYY-MM-DD format.

        Returns
        -------
        dict or None
            Envelope fields of the nearest snapshot, or None if none exists
            at or before the given timestamp.
        """
        events = self._reader().read_date(day)
        best: EventEnvelope | None = None
        for e in events:
            if e.event_type == EventType.SNAPSHOT.value and e.ts <= ts:
                best = e
        if best is None:
            return None
        return _envelope_to_dict(best)

    # ------------------------------------------------------------------
    # timeline — event markers for the timeline bar
    # ------------------------------------------------------------------

    def timeline(self, day: str) -> list[dict]:
        """Return timeline markers for a day's events.

        Filters to decision-relevant types (SIGNAL, GATE_DECISION, ORDER,
        FILL, RISK_ALERT, SNAPSHOT) and returns compact summaries suitable
        for rendering as timeline dots.
        """
        events = self._reader().read_date(day)
        markers: list[dict] = []
        for e in events:
            if e.event_type not in _TIMELINE_TYPES:
                continue
            markers.append({
                "ts": e.ts,
                "seq": e.seq,
                "event_type": e.event_type,
                "strategy_id": e.strategy_id,
                "symbol": e.symbol,
                "summary": _summarize(e),
            })
        return markers

    # ------------------------------------------------------------------
    # step — advance playback by a time window
    # ------------------------------------------------------------------

    def step(
        self,
        from_ts: str,
        delta_ms: int,
        day: str,
    ) -> dict:
        """Return events in (from_ts, from_ts+delta_ms] plus latest snapshot.

        Parameters
        ----------
        from_ts : str
            Exclusive lower bound timestamp (ISO-8601).
        delta_ms : int
            Window size in milliseconds of market time.
        day : str
            Date in YYYY-MM-DD format.

        Returns
        -------
        dict with keys:
            events: list[dict] — events in the window
            snapshot: dict|None — latest SNAPSHOT at or before window end
            next_ts: str — upper bound of this window (for chaining)
            has_more: bool — whether events exist after this window
        """
        # Compute upper bound timestamp
        next_ts = _add_ms(from_ts, delta_ms)

        events = self._reader().read_date(day)

        window_events: list[dict] = []
        latest_snapshot: EventEnvelope | None = None
        has_more = False

        for e in events:
            # Track latest snapshot at or before window end
            if e.event_type == EventType.SNAPSHOT.value and e.ts <= next_ts:
                latest_snapshot = e

            # Events in (from_ts, next_ts]
            if from_ts < e.ts <= next_ts:
                if e.event_type in _TIMELINE_TYPES:
                    window_events.append(_envelope_to_dict(e))

            # Check if there are events after window
            if e.ts > next_ts:
                has_more = True

        return {
            "events": window_events,
            "snapshot": _envelope_to_dict(latest_snapshot) if latest_snapshot else None,
            "next_ts": next_ts,
            "has_more": has_more,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _envelope_to_dict(e: EventEnvelope) -> dict:
    """Convert an EventEnvelope to a plain dict."""
    return {
        "ts": e.ts,
        "seq": e.seq,
        "event_type": e.event_type,
        "strategy_id": e.strategy_id,
        "symbol": e.symbol,
        "payload": e.payload,
    }


def _summarize(e: EventEnvelope) -> str:
    """One-line summary for a timeline marker."""
    p = e.payload
    t = e.event_type

    if t == EventType.SIGNAL.value:
        direction = p.get("direction", "?")
        conviction = p.get("conviction", 0)
        return f"{direction} conviction={conviction:.2f}"

    if t == EventType.GATE_DECISION.value:
        gate = p.get("gate", "?")
        approved = "PASS" if p.get("approved") else "BLOCK"
        return f"{gate}: {approved}"

    if t == EventType.ORDER.value:
        action = p.get("action", "?")
        side = p.get("side", "?")
        return f"{action} {side}"

    if t == EventType.FILL.value:
        side = p.get("side", "?")
        price = p.get("price", 0)
        return f"fill {side} @{price}"

    if t == EventType.RISK_ALERT.value:
        alert_type = p.get("alert_type", "?")
        return f"ALERT: {alert_type}"

    if t == EventType.SNAPSHOT.value:
        equity = p.get("equity", 0)
        dd = p.get("portfolio_dd", 0)
        return f"equity={equity:.4f} dd={dd:.4f}"

    return e.event_type


def _add_ms(iso_ts: str, delta_ms: int) -> str:
    """Add milliseconds to an ISO-8601 timestamp string.

    Handles the ISO-8601 timestamp format: YYYY-MM-DDTHH:MM:SS.ffffffZ
    """
    from datetime import datetime, timezone, timedelta

    # Parse — handle both 'Z' suffix and '+00:00'
    ts_str = iso_ts.rstrip("Z")
    try:
        dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
    except ValueError:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

    dt += timedelta(milliseconds=delta_ms)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
