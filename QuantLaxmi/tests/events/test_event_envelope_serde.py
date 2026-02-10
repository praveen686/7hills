"""Phase 2 — EventEnvelope + Serde tests.

Invariants:
  1. Roundtrip stability: serialize → parse → serialize is bit-identical
  2. Canonical JSON: sorted keys, compact separators, no NaN/Inf
  3. Frozen immutability: envelopes cannot be mutated after creation
  4. Factory stamps UTC timestamp when ts=None
  5. All payload types serialize cleanly via to_dict()
"""

from __future__ import annotations

import json
import math
import pytest

from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.types import EventType
from quantlaxmi.core.events.serde import (
    serialize_envelope,
    deserialize_envelope,
    roundtrip_stable,
    _sanitize,
)
from quantlaxmi.core.events.payloads import (
    TickPayload,
    Bar1mPayload,
    SignalPayload,
    GateDecisionPayload,
    OrderPayload,
    FillPayload,
    RiskAlertPayload,
    SnapshotPayload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_envelope(**overrides) -> EventEnvelope:
    defaults = dict(
        ts="2025-08-06T09:15:00.000000Z",
        seq=1,
        run_id="test-run-001",
        event_type="signal",
        source="test",
        strategy_id="S1",
        symbol="NIFTY",
        payload={"direction": "long", "conviction": 0.85},
    )
    defaults.update(overrides)
    return EventEnvelope(**defaults)


# ---------------------------------------------------------------------------
# TestRoundtripStability
# ---------------------------------------------------------------------------

class TestRoundtripStability:
    """serialize → deserialize → serialize is bit-identical."""

    def test_basic_roundtrip(self):
        env = _make_envelope()
        assert roundtrip_stable(env)

    def test_roundtrip_empty_payload(self):
        env = _make_envelope(payload={})
        assert roundtrip_stable(env)

    def test_roundtrip_nested_payload(self):
        env = _make_envelope(payload={
            "components": {"entropy": 2.3, "mi": 0.05},
            "direction": "long",
            "conviction": 0.7,
        })
        assert roundtrip_stable(env)

    def test_roundtrip_all_event_types(self):
        for etype in EventType:
            env = _make_envelope(event_type=etype.value, seq=etype.value.__hash__() % 1000)
            assert roundtrip_stable(env), f"Roundtrip failed for {etype.value}"

    def test_roundtrip_with_nan_replaced(self):
        """NaN in payload is replaced with null — roundtrip still stable."""
        env = _make_envelope(payload={"value": float("nan")})
        s1 = serialize_envelope(env)
        e2 = deserialize_envelope(s1)
        s2 = serialize_envelope(e2)
        assert s1 == s2
        # NaN should be replaced with None/null
        d = json.loads(s1)
        assert d["payload"]["value"] is None

    def test_roundtrip_with_inf_replaced(self):
        env = _make_envelope(payload={"value": float("inf")})
        s1 = serialize_envelope(env)
        d = json.loads(s1)
        assert d["payload"]["value"] is None
        assert roundtrip_stable(deserialize_envelope(s1))

    def test_roundtrip_with_neg_inf_replaced(self):
        env = _make_envelope(payload={"value": float("-inf")})
        s1 = serialize_envelope(env)
        d = json.loads(s1)
        assert d["payload"]["value"] is None

    def test_roundtrip_deterministic_3x(self):
        """3x serialize produces identical output."""
        env = _make_envelope(payload={"a": 1, "b": 2.0, "c": "text"})
        results = [serialize_envelope(env) for _ in range(3)]
        assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# TestCanonicalJSON
# ---------------------------------------------------------------------------

class TestCanonicalJSON:
    """JSON output is canonical: sorted keys, compact separators."""

    def test_sorted_keys(self):
        env = _make_envelope()
        s = serialize_envelope(env)
        d = json.loads(s)
        keys = list(d.keys())
        assert keys == sorted(keys), "Top-level keys must be sorted"

    def test_compact_separators(self):
        """No spaces after colons or commas."""
        env = _make_envelope()
        s = serialize_envelope(env)
        assert ": " not in s, "No space after colon"
        assert ", " not in s, "No space after comma"

    def test_single_line_no_newlines(self):
        env = _make_envelope()
        s = serialize_envelope(env)
        assert "\n" not in s
        assert "\r" not in s

    def test_ensure_ascii(self):
        """Output is ASCII-safe (no raw unicode)."""
        env = _make_envelope(payload={"note": "rupee ₹"})
        s = serialize_envelope(env)
        # json.dumps with ensure_ascii=True escapes unicode
        assert "₹" not in s or s.isascii()

    def test_payload_keys_sorted(self):
        """Payload dict keys are sorted in output."""
        env = _make_envelope(payload={"z_val": 1, "a_val": 2, "m_val": 3})
        s = serialize_envelope(env)
        d = json.loads(s)
        payload_keys = list(d["payload"].keys())
        assert payload_keys == sorted(payload_keys)


# ---------------------------------------------------------------------------
# TestSanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    """_sanitize replaces NaN/Inf in nested structures."""

    def test_nan_in_flat_dict(self):
        result = _sanitize({"v": float("nan")})
        assert result["v"] is None

    def test_inf_in_flat_dict(self):
        result = _sanitize({"v": float("inf")})
        assert result["v"] is None

    def test_neg_inf_in_flat_dict(self):
        result = _sanitize({"v": float("-inf")})
        assert result["v"] is None

    def test_nested_dict(self):
        result = _sanitize({"a": {"b": float("nan")}})
        assert result["a"]["b"] is None

    def test_list_with_nan(self):
        result = _sanitize([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_normal_values_unchanged(self):
        d = {"a": 1, "b": 2.5, "c": "text", "d": True, "e": None}
        assert _sanitize(d) == d

    def test_deeply_nested(self):
        d = {"l1": {"l2": {"l3": [float("inf"), 1.0]}}}
        result = _sanitize(d)
        assert result["l1"]["l2"]["l3"] == [None, 1.0]


# ---------------------------------------------------------------------------
# TestEnvelopeImmutability
# ---------------------------------------------------------------------------

class TestEnvelopeImmutability:
    """EventEnvelope is frozen — no mutation after creation."""

    def test_frozen_ts(self):
        env = _make_envelope()
        with pytest.raises(AttributeError):
            env.ts = "2025-01-01T00:00:00.000000Z"

    def test_frozen_seq(self):
        env = _make_envelope()
        with pytest.raises(AttributeError):
            env.seq = 999

    def test_frozen_payload(self):
        env = _make_envelope()
        with pytest.raises(AttributeError):
            env.payload = {"mutated": True}


# ---------------------------------------------------------------------------
# TestEnvelopeFactory
# ---------------------------------------------------------------------------

class TestEnvelopeFactory:
    """EventEnvelope.create() factory method."""

    def test_auto_timestamp(self):
        env = EventEnvelope.create(
            seq=1, run_id="r1", event_type="tick", source="test", payload={},
        )
        assert env.ts.endswith("Z")
        assert "T" in env.ts

    def test_explicit_timestamp(self):
        ts = "2025-08-06T09:15:00.000000Z"
        env = EventEnvelope.create(
            seq=1, run_id="r1", event_type="tick", source="test", payload={}, ts=ts,
        )
        assert env.ts == ts

    def test_default_strategy_id_empty(self):
        env = EventEnvelope.create(
            seq=1, run_id="r1", event_type="tick", source="test", payload={},
        )
        assert env.strategy_id == ""

    def test_default_symbol_empty(self):
        env = EventEnvelope.create(
            seq=1, run_id="r1", event_type="tick", source="test", payload={},
        )
        assert env.symbol == ""


# ---------------------------------------------------------------------------
# TestPayloadSerialization
# ---------------------------------------------------------------------------

class TestPayloadSerialization:
    """Each payload type serializes cleanly to dict and through serde."""

    def test_tick_payload(self):
        p = TickPayload(instrument_token=256265, ltp=23500.0, volume=1000)
        d = p.to_dict()
        assert d["instrument_token"] == 256265
        assert d["ltp"] == 23500.0
        env = _make_envelope(event_type="tick", payload=d)
        assert roundtrip_stable(env)

    def test_bar1m_payload(self):
        p = Bar1mPayload(open=23400.0, high=23500.0, low=23350.0, close=23480.0, volume=50000)
        d = p.to_dict()
        assert d["close"] == 23480.0
        env = _make_envelope(event_type="bar_1m", payload=d)
        assert roundtrip_stable(env)

    def test_signal_payload(self):
        p = SignalPayload(
            direction="long", conviction=0.85, instrument_type="CE",
            strike=23500.0, expiry="2025-08-14", regime="low_vol",
        )
        d = p.to_dict()
        assert d["direction"] == "long"
        env = _make_envelope(event_type="signal", payload=d)
        assert roundtrip_stable(env)

    def test_gate_decision_payload(self):
        p = GateDecisionPayload(
            signal_seq=5, gate="pass", approved=True,
            adjusted_weight=0.1, vpin=0.3, portfolio_dd=0.02,
        )
        d = p.to_dict()
        assert d["approved"] is True
        env = _make_envelope(event_type="gate_decision", payload=d)
        assert roundtrip_stable(env)

    def test_order_payload(self):
        p = OrderPayload(order_id="abc123", action="submit", side="buy", order_type="market")
        d = p.to_dict()
        env = _make_envelope(event_type="order", payload=d)
        assert roundtrip_stable(env)

    def test_fill_payload(self):
        p = FillPayload(order_id="abc123", fill_id="f001", side="buy", quantity=25, price=23500.0)
        d = p.to_dict()
        env = _make_envelope(event_type="fill", payload=d)
        assert roundtrip_stable(env)

    def test_risk_alert_payload(self):
        p = RiskAlertPayload(alert_type="vpin_toxic", new_state="blocked", threshold=0.85, current_value=0.91)
        d = p.to_dict()
        env = _make_envelope(event_type="risk_alert", payload=d)
        assert roundtrip_stable(env)

    def test_snapshot_payload(self):
        p = SnapshotPayload(
            equity=1.05, peak_equity=1.05, portfolio_dd=0.0,
            total_exposure=0.3, vpin=0.4, position_count=2,
            strategy_equity={"S1": 1.02, "S4": 1.03},
        )
        d = p.to_dict()
        env = _make_envelope(event_type="snapshot", payload=d)
        assert roundtrip_stable(env)


# ---------------------------------------------------------------------------
# TestPayloadImmutability
# ---------------------------------------------------------------------------

class TestPayloadImmutability:
    """All payloads are frozen dataclasses."""

    def test_signal_frozen(self):
        p = SignalPayload(direction="long", conviction=0.85, instrument_type="CE")
        with pytest.raises(AttributeError):
            p.direction = "short"

    def test_tick_frozen(self):
        p = TickPayload(instrument_token=256265, ltp=23500.0)
        with pytest.raises(AttributeError):
            p.ltp = 24000.0

    def test_order_frozen(self):
        p = OrderPayload(order_id="x", action="submit", side="buy", order_type="market")
        with pytest.raises(AttributeError):
            p.action = "cancel"


# ---------------------------------------------------------------------------
# TestDeserializationErrors
# ---------------------------------------------------------------------------

class TestDeserializationErrors:
    """Deserialization handles bad input gracefully."""

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            deserialize_envelope("")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            deserialize_envelope("{not valid json}")

    def test_truncated_json_raises(self):
        with pytest.raises(ValueError):
            deserialize_envelope('{"ts":"2025-08-06T09:15:00.000000Z","seq":1')

    def test_missing_required_field_raises(self):
        # Missing 'seq' field
        line = json.dumps({"ts": "x", "run_id": "r", "event_type": "t", "source": "s"})
        with pytest.raises(KeyError):
            deserialize_envelope(line)
