"""Event stream comparator for replay parity verification.

Compares a reference event stream (from the original run) against a
replay event stream (from the replay engine) and produces a diff report.

Comparison is by content, not by timestamp (wall-clock differs between
runs). Events are matched by position in the filtered streams.

Match fields per event type:
  - signal: strategy_id, symbol, direction, conviction, instrument_type
  - gate_decision: strategy_id, symbol, approved, gate, adjusted_weight
  - order: strategy_id, symbol, action, side
  - fill: strategy_id, symbol, side, quantity, price
  - snapshot: equity, position_count, portfolio_dd
  - risk_alert: strategy_id, symbol, alert_type
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from quantlaxmi.core.events.envelope import EventEnvelope

logger = logging.getLogger(__name__)

# Fields to compare per event type (payload keys)
_COMPARE_FIELDS: dict[str, list[str]] = {
    "signal": ["direction", "conviction", "instrument_type", "regime"],
    "gate_decision": ["gate", "approved", "adjusted_weight", "reason"],
    "order": ["action", "side", "order_type"],
    "fill": ["side", "quantity", "price"],
    "risk_alert": ["alert_type", "new_state"],
    "snapshot": ["equity", "peak_equity", "portfolio_dd", "position_count"],
    "missingness": ["check_type", "symbol", "severity"],
}

# Floating-point tolerance for numeric comparisons
_FP_RTOL = 1e-10
_FP_ATOL = 1e-12


@dataclass(frozen=True)
class FieldDiff:
    """A single field difference between reference and replay."""

    event_type: str
    index: int          # position in the filtered stream
    field: str
    ref_value: object
    replay_value: object
    strategy_id: str = ""
    symbol: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing reference and replay event streams."""

    identical: bool = True
    total_compared: int = 0
    diffs: list[FieldDiff] = field(default_factory=list)
    missing_in_replay: int = 0  # events in ref but not in replay
    extra_in_replay: int = 0    # events in replay but not in ref
    event_type_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Replay Parity: {'PASS' if self.identical else 'FAIL'}",
            f"  Total compared: {self.total_compared}",
            f"  Diffs: {len(self.diffs)}",
            f"  Missing in replay: {self.missing_in_replay}",
            f"  Extra in replay: {self.extra_in_replay}",
        ]
        if self.event_type_counts:
            lines.append("  Event counts:")
            for etype, counts in sorted(self.event_type_counts.items()):
                lines.append(
                    f"    {etype}: ref={counts.get('ref', 0)} replay={counts.get('replay', 0)}"
                )
        if self.diffs:
            lines.append(f"  First 10 diffs:")
            for d in self.diffs[:10]:
                lines.append(
                    f"    [{d.event_type}#{d.index}] {d.field}: "
                    f"{d.ref_value!r} vs {d.replay_value!r} "
                    f"({d.strategy_id}/{d.symbol})"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "identical": self.identical,
            "total_compared": self.total_compared,
            "diffs": [
                {
                    "event_type": d.event_type,
                    "index": d.index,
                    "field": d.field,
                    "ref_value": _safe_repr(d.ref_value),
                    "replay_value": _safe_repr(d.replay_value),
                    "strategy_id": d.strategy_id,
                    "symbol": d.symbol,
                }
                for d in self.diffs
            ],
            "missing_in_replay": self.missing_in_replay,
            "extra_in_replay": self.extra_in_replay,
            "event_type_counts": self.event_type_counts,
        }


def _safe_repr(v: object) -> object:
    """Convert value to JSON-safe representation."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return v


def _values_equal(a: object, b: object) -> bool:
    """Compare two values with floating-point tolerance."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return a == b  # same sign
        if abs(a) < _FP_ATOL and abs(b) < _FP_ATOL:
            return True
        return abs(a - b) <= _FP_RTOL * max(abs(a), abs(b))
    return a == b


def compare_streams(
    ref_events: list[EventEnvelope],
    replay_events: list[EventEnvelope],
    event_types: list[str] | None = None,
) -> ComparisonResult:
    """Compare reference and replay event streams.

    Parameters
    ----------
    ref_events : list[EventEnvelope]
        Events from the original run.
    replay_events : list[EventEnvelope]
        Events from the replay run.
    event_types : list[str], optional
        Event types to compare. If None, compare all types in _COMPARE_FIELDS.

    Returns
    -------
    ComparisonResult
        Detailed comparison report.
    """
    types_to_check = event_types or list(_COMPARE_FIELDS.keys())
    result = ComparisonResult()

    for etype in types_to_check:
        ref_filtered = [e for e in ref_events if e.event_type == etype]
        replay_filtered = [e for e in replay_events if e.event_type == etype]

        result.event_type_counts[etype] = {
            "ref": len(ref_filtered),
            "replay": len(replay_filtered),
        }

        # Count mismatches
        if len(ref_filtered) != len(replay_filtered):
            result.identical = False
            diff = len(ref_filtered) - len(replay_filtered)
            if diff > 0:
                result.missing_in_replay += diff
            else:
                result.extra_in_replay += abs(diff)

        # Compare matching pairs
        compare_fields = _COMPARE_FIELDS.get(etype, [])
        n = min(len(ref_filtered), len(replay_filtered))

        for i in range(n):
            ref_env = ref_filtered[i]
            rep_env = replay_filtered[i]
            result.total_compared += 1

            # Compare envelope-level fields
            for attr in ("strategy_id", "symbol", "event_type"):
                ref_val = getattr(ref_env, attr)
                rep_val = getattr(rep_env, attr)
                if ref_val != rep_val:
                    result.identical = False
                    result.diffs.append(FieldDiff(
                        event_type=etype,
                        index=i,
                        field=f"envelope.{attr}",
                        ref_value=ref_val,
                        replay_value=rep_val,
                        strategy_id=ref_env.strategy_id,
                        symbol=ref_env.symbol,
                    ))

            # Compare payload fields
            for fld in compare_fields:
                ref_val = ref_env.payload.get(fld)
                rep_val = rep_env.payload.get(fld)
                if not _values_equal(ref_val, rep_val):
                    result.identical = False
                    result.diffs.append(FieldDiff(
                        event_type=etype,
                        index=i,
                        field=f"payload.{fld}",
                        ref_value=ref_val,
                        replay_value=rep_val,
                        strategy_id=ref_env.strategy_id,
                        symbol=ref_env.symbol,
                    ))

    return result
