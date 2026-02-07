"""EventEnvelope — the canonical wrapper for every BRAHMASTRA event.

Every event persisted to the WAL or flowing through the bus is wrapped
in an EventEnvelope.  The envelope carries:
  - ts: UTC ISO timestamp (wall clock, for human audit)
  - seq: monotonic sequence number (for ordering, gap detection)
  - run_id: stable per session (for grouping, replay)
  - event_type: one of EventType values
  - source: engine module that emitted the event
  - strategy_id: optional (required for decision events)
  - symbol: optional (required for instrument-specific events)
  - payload: typed dict (schema depends on event_type)

The envelope is a frozen dataclass — immutable once created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class EventEnvelope:
    """Immutable event envelope wrapping every BRAHMASTRA event.

    Attributes
    ----------
    ts : str
        UTC ISO-8601 timestamp with microsecond precision.
    seq : int
        Monotonically increasing sequence number (per run).
    run_id : str
        Stable identifier for this session/run.
    event_type : str
        One of EventType values.
    source : str
        Engine module that emitted the event (e.g. "orchestrator",
        "signal_generator", "risk_monitor").
    strategy_id : str
        Strategy identifier. Required for decision events, empty for
        market-only events.
    symbol : str
        Instrument symbol. Empty where irrelevant.
    payload : dict
        Typed payload (schema depends on event_type).
    """

    ts: str
    seq: int
    run_id: str
    event_type: str
    source: str
    strategy_id: str = ""
    symbol: str = ""
    payload: dict = field(default_factory=dict)

    @staticmethod
    def create(
        seq: int,
        run_id: str,
        event_type: str,
        source: str,
        payload: dict,
        strategy_id: str = "",
        symbol: str = "",
        ts: str | None = None,
    ) -> EventEnvelope:
        """Factory that stamps UTC timestamp if not provided."""
        if ts is None:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return EventEnvelope(
            ts=ts,
            seq=seq,
            run_id=run_id,
            event_type=event_type,
            source=source,
            strategy_id=strategy_id,
            symbol=symbol,
            payload=payload,
        )
