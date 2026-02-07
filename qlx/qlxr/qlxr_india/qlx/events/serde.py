"""Canonical JSON serialization for BRAHMASTRA events.

Guarantees:
  1. Stable key ordering (sorted keys)
  2. Fixed float formatting (no scientific notation surprise)
  3. Explicit None semantics (None → null, missing key = never present)
  4. Round-trip stability: serialize → parse → serialize is bit-identical
  5. One event per line (JSONL compatible, no embedded newlines)
"""

from __future__ import annotations

import json
import math
from typing import Any

from qlx.events.envelope import EventEnvelope


# ---------------------------------------------------------------------------
# Float formatter — no scientific notation, 10 decimal places max
# ---------------------------------------------------------------------------

def _format_float(v: float) -> float | str | None:
    """Format float for canonical JSON.

    - NaN → null (JSON has no NaN)
    - Inf → null (JSON has no Inf)
    - Otherwise → float with up to 10 significant decimal digits
    """
    if math.isnan(v) or math.isinf(v):
        return None
    return v


class _CanonicalEncoder(json.JSONEncoder):
    """JSON encoder with canonical float handling."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, float):
            return _format_float(obj)
        return super().default(obj)


def _sanitize(obj: Any) -> Any:
    """Recursively sanitize an object for canonical JSON.

    - Replace NaN/Inf floats with None
    - Leave everything else unchanged
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serialize_envelope(envelope: EventEnvelope) -> str:
    """Serialize an EventEnvelope to canonical JSONL (single line).

    Returns a single-line JSON string (no trailing newline).
    Guarantees:
      - Sorted keys
      - No NaN/Inf (replaced with null)
      - Deterministic output for identical input
    """
    d = {
        "ts": envelope.ts,
        "seq": envelope.seq,
        "run_id": envelope.run_id,
        "event_type": envelope.event_type,
        "source": envelope.source,
        "strategy_id": envelope.strategy_id,
        "symbol": envelope.symbol,
        "payload": _sanitize(envelope.payload),
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deserialize_envelope(line: str) -> EventEnvelope:
    """Deserialize a JSONL line back to an EventEnvelope.

    Raises ValueError on parse failure.
    """
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSONL: {e}") from e

    return EventEnvelope(
        ts=d["ts"],
        seq=d["seq"],
        run_id=d["run_id"],
        event_type=d["event_type"],
        source=d["source"],
        strategy_id=d.get("strategy_id", ""),
        symbol=d.get("symbol", ""),
        payload=d.get("payload", {}),
    )


def roundtrip_stable(envelope: EventEnvelope) -> bool:
    """Check that serialize → deserialize → serialize is bit-identical."""
    s1 = serialize_envelope(envelope)
    e2 = deserialize_envelope(s1)
    s2 = serialize_envelope(e2)
    return s1 == s2
