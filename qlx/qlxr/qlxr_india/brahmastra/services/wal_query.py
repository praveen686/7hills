"""WAL Query Service — read-only lookups into the event log.

Provides query methods for the Why Panel API endpoints.
All data comes directly from persisted JSONL WAL files via WalReader;
no derived state, no recomputation, no inference.

Usage::

    svc = WalQueryService(base_dir=Path("data/events"))
    signal = svc.get_signal_by_seq(42)
    chain  = svc.get_decision_chain(strategy_id="s5_hawkes",
                                     symbol="NIFTY", date="2025-09-15")
"""

from __future__ import annotations

import logging
from pathlib import Path

from qlx.events.envelope import EventEnvelope
from qlx.events.types import EventType
from brahmastra.replay.reader import WalReader

logger = logging.getLogger(__name__)

# Default WAL directory
DEFAULT_WAL_DIR = Path("data/events")


class WalQueryService:
    """Read-only query layer over the persisted event WAL.

    Thread-safe: each call creates a fresh WalReader (no shared mutable state).
    """

    def __init__(self, base_dir: Path | str = DEFAULT_WAL_DIR):
        self._base_dir = Path(base_dir)

    def _reader(self) -> WalReader:
        return WalReader(base_dir=self._base_dir)

    # ------------------------------------------------------------------
    # Primitive lookups
    # ------------------------------------------------------------------

    def get_event_by_seq(self, seq: int, day: str | None = None) -> EventEnvelope | None:
        """Find a single event by its seq number.

        If *day* is given, search only that day's file.
        Otherwise search all available dates (slower).
        """
        reader = self._reader()
        if day:
            events = reader.read_date(day)
        else:
            events = reader.read_all()

        for e in events:
            if e.seq == seq:
                return e
        return None

    def get_events_for_date(self, day: str) -> list[EventEnvelope]:
        """Return all events for a single date."""
        return self._reader().read_date(day)

    def get_events_by_type(
        self, event_type: str, day: str,
    ) -> list[EventEnvelope]:
        """Return events of a specific type for a date."""
        events = self._reader().read_date(day)
        return [e for e in events if e.event_type == event_type]

    def get_events_by_strategy(
        self, strategy_id: str, day: str,
    ) -> list[EventEnvelope]:
        """Return all events for a strategy on a date."""
        events = self._reader().read_date(day)
        return [e for e in events if e.strategy_id == strategy_id]

    # ------------------------------------------------------------------
    # Signal context (Why Panel endpoint #1)
    # ------------------------------------------------------------------

    def get_signal_context(self, signal_seq: int, day: str) -> dict | None:
        """Get full context for a signal: the signal event + strategy-specific fields.

        Returns a dict with:
            - signal: the signal EventEnvelope fields
            - components: strategy-specific why fields (from payload.components)
            - regime: regime at signal time
            - reasoning: human-readable explanation (if present)

        Returns None if no signal found at that seq.
        """
        reader = self._reader()
        events = reader.read_date(day)

        signal_event = None
        for e in events:
            if e.seq == signal_seq and e.event_type == EventType.SIGNAL.value:
                signal_event = e
                break

        if signal_event is None:
            return None

        payload = signal_event.payload
        return {
            "signal_seq": signal_event.seq,
            "ts": signal_event.ts,
            "strategy_id": signal_event.strategy_id,
            "symbol": signal_event.symbol,
            "direction": payload.get("direction", ""),
            "conviction": payload.get("conviction", 0.0),
            "instrument_type": payload.get("instrument_type", ""),
            "strike": payload.get("strike", 0.0),
            "expiry": payload.get("expiry", ""),
            "ttl_bars": payload.get("ttl_bars", 0),
            "regime": payload.get("regime", ""),
            "components": payload.get("components", {}),
            "reasoning": payload.get("reasoning", ""),
        }

    # ------------------------------------------------------------------
    # Gate decisions (Why Panel endpoint #2)
    # ------------------------------------------------------------------

    def get_gate_decisions(self, signal_seq: int, day: str) -> list[dict]:
        """Get gate decisions linked to a signal.

        The orchestrator emits gate_decision events right after the signal.
        We match by strategy_id + symbol + seq ordering: gate decisions for
        a signal are the gate_decision events with the same strategy_id and
        symbol that appear *after* the signal in the seq ordering.

        Returns a list of gate decision dicts.
        """
        reader = self._reader()
        events = reader.read_date(day)

        # Find the signal first
        signal_event = None
        for e in events:
            if e.seq == signal_seq and e.event_type == EventType.SIGNAL.value:
                signal_event = e
                break

        if signal_event is None:
            return []

        # Find gate decisions that follow this signal (same strategy + symbol)
        decisions = []
        found_signal = False
        for e in events:
            if e.seq == signal_seq:
                found_signal = True
                continue

            if not found_signal:
                continue

            # Stop when we hit the next signal for this strategy (next scan cycle)
            if (e.event_type == EventType.SIGNAL.value
                    and e.strategy_id == signal_event.strategy_id):
                break

            if (e.event_type == EventType.GATE_DECISION.value
                    and e.strategy_id == signal_event.strategy_id
                    and e.symbol == signal_event.symbol):
                payload = e.payload
                decisions.append({
                    "seq": e.seq,
                    "ts": e.ts,
                    "gate": payload.get("gate", ""),
                    "approved": payload.get("approved", False),
                    "adjusted_weight": payload.get("adjusted_weight", 0.0),
                    "reason": payload.get("reason", ""),
                    "vpin": payload.get("vpin", 0.0),
                    "portfolio_dd": payload.get("portfolio_dd", 0.0),
                    "strategy_dd": payload.get("strategy_dd", 0.0),
                    "total_exposure": payload.get("total_exposure", 0.0),
                })

        return decisions

    # ------------------------------------------------------------------
    # Trade decision chain (Why Panel endpoint #3)
    # ------------------------------------------------------------------

    def get_trade_decision_chain(
        self,
        strategy_id: str,
        symbol: str,
        day: str,
    ) -> dict | None:
        """Get full decision chain for a trade: signal → gate → order → fill.

        Searches a specific day for the complete chain matching strategy+symbol.
        Returns a dict with: signal, gates, orders, fills, risk_alerts, snapshot.
        Returns None if no signal found.
        """
        reader = self._reader()
        events = reader.read_date(day)

        # Collect all events matching strategy + symbol
        signals = []
        gates = []
        orders = []
        fills = []
        risk_alerts = []
        snapshot = None

        for e in events:
            if e.strategy_id == strategy_id and e.symbol == symbol:
                if e.event_type == EventType.SIGNAL.value:
                    signals.append(e)
                elif e.event_type == EventType.GATE_DECISION.value:
                    gates.append(e)
                elif e.event_type == EventType.ORDER.value:
                    orders.append(e)
                elif e.event_type == EventType.FILL.value:
                    fills.append(e)
                elif e.event_type == EventType.RISK_ALERT.value:
                    risk_alerts.append(e)

            # Snapshot events have no strategy_id; grab the last one for context
            if e.event_type == EventType.SNAPSHOT.value:
                snapshot = e

        if not signals:
            return None

        def _envelope_to_dict(e: EventEnvelope) -> dict:
            return {
                "seq": e.seq,
                "ts": e.ts,
                "event_type": e.event_type,
                "strategy_id": e.strategy_id,
                "symbol": e.symbol,
                "payload": e.payload,
            }

        return {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "date": day,
            "signals": [_envelope_to_dict(e) for e in signals],
            "gates": [_envelope_to_dict(e) for e in gates],
            "orders": [_envelope_to_dict(e) for e in orders],
            "fills": [_envelope_to_dict(e) for e in fills],
            "risk_alerts": [_envelope_to_dict(e) for e in risk_alerts],
            "snapshot": _envelope_to_dict(snapshot) if snapshot else None,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def available_dates(self) -> list[str]:
        """Return sorted list of dates with event data."""
        return self._reader().available_dates()

    def day_summary(self, day: str) -> dict:
        """Quick summary of events on a day (counts by type)."""
        events = self._reader().read_date(day)
        counts: dict[str, int] = {}
        strategies: set[str] = set()
        symbols: set[str] = set()

        for e in events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
            if e.strategy_id:
                strategies.add(e.strategy_id)
            if e.symbol:
                symbols.add(e.symbol)

        return {
            "date": day,
            "total_events": len(events),
            "event_counts": counts,
            "strategies": sorted(strategies),
            "symbols": sorted(symbols),
        }
