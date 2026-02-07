"""Specialized journal writers for execution and signal audit trails.

Thin wrappers over EventLogWriter that enforce typed payloads
and provide query-friendly access patterns.

ExecutionJournal: orders + fills lifecycle
SignalJournal: every signal generated (pre-gate + post-gate)
"""

from __future__ import annotations

import logging
from pathlib import Path

from brahmastra.engine.event_log import EventLogWriter
from qlx.events.envelope import EventEnvelope
from qlx.events.types import EventType
from qlx.events.payloads import (
    SignalPayload,
    GateDecisionPayload,
    OrderPayload,
    FillPayload,
)

logger = logging.getLogger(__name__)


class SignalJournal:
    """Logs every generated signal with full contract-required context.

    Includes signals that are later blocked by gates (pre-gate).
    Links gate decisions to signals via signal_seq.
    """

    def __init__(self, log: EventLogWriter):
        self._log = log
        self._signal_count = 0
        self._gate_count = 0

    def log_signal(
        self,
        strategy_id: str,
        symbol: str,
        payload: SignalPayload,
    ) -> EventEnvelope:
        """Log a pre-gate signal event. Returns the envelope (for seq linking)."""
        env = self._log.emit(
            event_type=EventType.SIGNAL.value,
            source="signal_journal",
            payload=payload.to_dict(),
            strategy_id=strategy_id,
            symbol=symbol,
        )
        self._signal_count += 1
        return env

    def log_gate_decision(
        self,
        strategy_id: str,
        symbol: str,
        payload: GateDecisionPayload,
    ) -> EventEnvelope:
        """Log a gate decision event. signal_seq links to the upstream signal."""
        env = self._log.emit(
            event_type=EventType.GATE_DECISION.value,
            source="signal_journal",
            payload=payload.to_dict(),
            strategy_id=strategy_id,
            symbol=symbol,
        )
        self._gate_count += 1
        return env

    def stats(self) -> dict:
        return {
            "signal_count": self._signal_count,
            "gate_count": self._gate_count,
        }


class ExecutionJournal:
    """Durable audit record for the order/fill lifecycle.

    Keyed by order_id: placement → ack/reject → fills → cancel/replace.
    """

    def __init__(self, log: EventLogWriter):
        self._log = log
        self._order_count = 0
        self._fill_count = 0

    def log_order(
        self,
        strategy_id: str,
        symbol: str,
        payload: OrderPayload,
    ) -> EventEnvelope:
        """Log an order lifecycle event (submit/ack/reject/cancel/replace)."""
        env = self._log.emit(
            event_type=EventType.ORDER.value,
            source="execution_journal",
            payload=payload.to_dict(),
            strategy_id=strategy_id,
            symbol=symbol,
        )
        self._order_count += 1
        return env

    def log_fill(
        self,
        strategy_id: str,
        symbol: str,
        payload: FillPayload,
    ) -> EventEnvelope:
        """Log a fill event (partial or full)."""
        env = self._log.emit(
            event_type=EventType.FILL.value,
            source="execution_journal",
            payload=payload.to_dict(),
            strategy_id=strategy_id,
            symbol=symbol,
        )
        self._fill_count += 1
        return env

    def stats(self) -> dict:
        return {
            "order_count": self._order_count,
            "fill_count": self._fill_count,
        }
