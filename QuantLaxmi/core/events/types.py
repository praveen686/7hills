"""Event type enumeration for BRAHMASTRA.

Every event flowing through the system has exactly one EventType.
Types are string-valued for JSON serialization stability.
"""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    """Well-known event types in the BRAHMASTRA event system.

    String-valued so they serialize to stable JSON without custom encoders.
    """

    # Market data
    TICK = "tick"
    BAR_1M = "bar_1m"

    # Strategy decisions
    SIGNAL = "signal"
    GATE_DECISION = "gate_decision"

    # Execution lifecycle
    ORDER = "order"
    FILL = "fill"

    # Risk
    RISK_ALERT = "risk_alert"
    SNAPSHOT = "snapshot"

    # Data quality
    MISSINGNESS = "missingness"

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
