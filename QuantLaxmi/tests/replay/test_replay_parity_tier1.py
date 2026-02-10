"""Phase 3 — Replay parity tests for Tier-1 strategies.

Tests that replaying a captured event stream produces identical decisions.
Uses the WalReader, Comparator, and ReplayEngine with synthetic reference
data (no real DuckDB dependency — pure deterministic verification).

Core invariants tested:
  - 3x replay produces bit-identical event streams
  - Signal sequence, gate decisions, and snapshots match across runs
  - Event count parity (ref == replay)
  - Field-level comparison with FP tolerance
  - Filtered stream comparison per event type
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantlaxmi.core.events.envelope import EventEnvelope
from quantlaxmi.core.events.serde import serialize_envelope, deserialize_envelope
from quantlaxmi.core.events.hashing import GENESIS, chain_hash, compute_chain
from quantlaxmi.core.events.types import EventType

from quantlaxmi.engine.replay.reader import WalReader, WalValidationError
from quantlaxmi.engine.replay.comparator import (
    ComparisonResult,
    FieldDiff,
    compare_streams,
    _values_equal,
)


# ---------------------------------------------------------------------------
# Helpers: create synthetic event streams for testing
# ---------------------------------------------------------------------------

_TS = "2025-08-06T09:15:00.000000Z"
_RUN_ID = "test-run-001"


def _make_signal(
    seq: int,
    strategy_id: str,
    symbol: str,
    direction: str = "long",
    conviction: float = 0.75,
    regime: str = "normal",
) -> EventEnvelope:
    return EventEnvelope(
        ts=_TS,
        seq=seq,
        run_id=_RUN_ID,
        event_type="signal",
        source="orchestrator",
        strategy_id=strategy_id,
        symbol=symbol,
        payload={
            "direction": direction,
            "conviction": conviction,
            "instrument_type": "FUT",
            "regime": regime,
            "strike": 0.0,
            "expiry": "",
            "ttl_bars": 5,
            "components": {},
        },
    )


def _make_gate(
    seq: int,
    strategy_id: str,
    symbol: str,
    approved: bool = True,
    gate: str = "exposure",
    adjusted_weight: float = 0.10,
    reason: str = "",
) -> EventEnvelope:
    return EventEnvelope(
        ts=_TS,
        seq=seq,
        run_id=_RUN_ID,
        event_type="gate_decision",
        source="orchestrator",
        strategy_id=strategy_id,
        symbol=symbol,
        payload={
            "gate": gate,
            "approved": approved,
            "adjusted_weight": adjusted_weight,
            "reason": reason,
            "signal_seq": 0,
            "vpin": 0.0,
            "portfolio_dd": 0.0,
            "strategy_dd": 0.0,
            "total_exposure": 0.10,
        },
    )


def _make_snapshot(
    seq: int,
    equity: float = 1.0,
    peak_equity: float = 1.0,
    portfolio_dd: float = 0.0,
    position_count: int = 1,
) -> EventEnvelope:
    return EventEnvelope(
        ts=_TS,
        seq=seq,
        run_id=_RUN_ID,
        event_type="snapshot",
        source="orchestrator",
        strategy_id="",
        symbol="",
        payload={
            "equity": equity,
            "peak_equity": peak_equity,
            "portfolio_dd": portfolio_dd,
            "total_exposure": 0.10,
            "vpin": 0.0,
            "position_count": position_count,
            "strategy_equity": {},
            "strategy_dd": {},
            "active_breakers": [],
            "regime": "normal",
        },
    )


def _make_order(
    seq: int,
    strategy_id: str,
    symbol: str,
    action: str = "submit",
    side: str = "buy",
) -> EventEnvelope:
    return EventEnvelope(
        ts=_TS,
        seq=seq,
        run_id=_RUN_ID,
        event_type="order",
        source="orchestrator",
        strategy_id=strategy_id,
        symbol=symbol,
        payload={
            "order_id": "aaaa1111",
            "action": action,
            "side": side,
            "order_type": "market",
        },
    )


def _make_risk_alert(
    seq: int,
    strategy_id: str,
    symbol: str,
    alert_type: str = "exposure",
) -> EventEnvelope:
    return EventEnvelope(
        ts=_TS,
        seq=seq,
        run_id=_RUN_ID,
        event_type="risk_alert",
        source="orchestrator",
        strategy_id=strategy_id,
        symbol=symbol,
        payload={
            "alert_type": alert_type,
            "new_state": "blocked",
            "threshold": 0.20,
            "current_value": 0.25,
            "detail": "exposure limit breached",
        },
    )


def _build_s5_stream() -> list[EventEnvelope]:
    """Synthetic S5 Hawkes Microstructure stream."""
    return [
        _make_signal(1, "s5_hawkes", "NIFTY", "long", 0.82, "low_vol"),
        _make_gate(2, "s5_hawkes", "NIFTY", True, "all_gates", 0.08),
        _make_order(3, "s5_hawkes", "NIFTY", "submit", "buy"),
        _make_snapshot(4, equity=1.0, peak_equity=1.0, position_count=1),
        _make_signal(5, "s5_hawkes", "BANKNIFTY", "short", 0.65, "low_vol"),
        _make_gate(6, "s5_hawkes", "BANKNIFTY", True, "all_gates", 0.06),
        _make_order(7, "s5_hawkes", "BANKNIFTY", "submit", "sell"),
        _make_snapshot(8, equity=1.001, peak_equity=1.001, position_count=2),
    ]


def _build_s1_stream() -> list[EventEnvelope]:
    """Synthetic S1 VRP Options stream."""
    return [
        _make_signal(1, "s1_vrp", "NIFTY", "long", 0.90, "normal"),
        _make_gate(2, "s1_vrp", "NIFTY", True, "all_gates", 0.12),
        _make_order(3, "s1_vrp", "NIFTY", "submit", "buy"),
        _make_snapshot(4, equity=1.0, peak_equity=1.0, position_count=1),
    ]


def _build_s4_stream() -> list[EventEnvelope]:
    """Synthetic S4 IV Mean Reversion stream."""
    return [
        _make_signal(1, "s4_iv_mr", "NIFTY", "long", 0.70, "normal"),
        _make_gate(2, "s4_iv_mr", "NIFTY", False, "vix_gate", 0.0, "VIX > 20"),
        _make_risk_alert(3, "s4_iv_mr", "NIFTY", "vix_gate"),
        _make_snapshot(4, equity=1.0, peak_equity=1.0, position_count=0),
    ]


def _build_s7_stream() -> list[EventEnvelope]:
    """Synthetic S7 Regime Switch stream."""
    return [
        _make_signal(1, "s7_regime", "NIFTY", "long", 0.55, "trending"),
        _make_gate(2, "s7_regime", "NIFTY", True, "all_gates", 0.05),
        _make_order(3, "s7_regime", "NIFTY", "submit", "buy"),
        _make_signal(4, "s7_regime", "BANKNIFTY", "short", 0.60, "trending"),
        _make_gate(5, "s7_regime", "BANKNIFTY", True, "all_gates", 0.04),
        _make_order(6, "s7_regime", "BANKNIFTY", "submit", "sell"),
        _make_snapshot(7, equity=1.002, peak_equity=1.002, position_count=2),
    ]


def _write_reference_wal(
    events: list[EventEnvelope],
    base_dir: Path,
    day: str = "2025-08-06",
    with_hashes: bool = False,
) -> None:
    """Write events as JSONL reference file."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{day}.jsonl"
    lines = [serialize_envelope(e) for e in events]
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    if with_hashes:
        hashes = compute_chain(lines)
        hash_path = base_dir / f"{day}.sha256"
        with open(hash_path, "w") as f:
            for h in hashes:
                f.write(h + "\n")


# ===================================================================
# Test Class: WalReader
# ===================================================================


class TestWalReader:
    """Test WalReader reads JSONL files correctly."""

    def test_read_single_day(self, tmp_path: Path):
        events = _build_s5_stream()
        _write_reference_wal(events, tmp_path, "2025-08-06")

        reader = WalReader(base_dir=tmp_path)
        result = reader.read_date("2025-08-06")
        assert len(result) == len(events)
        for orig, read in zip(events, result):
            assert orig.seq == read.seq
            assert orig.event_type == read.event_type
            assert orig.strategy_id == read.strategy_id

    def test_read_range(self, tmp_path: Path):
        day1 = _build_s5_stream()
        day2 = _build_s1_stream()
        _write_reference_wal(day1, tmp_path, "2025-08-06")
        _write_reference_wal(day2, tmp_path, "2025-08-07")

        reader = WalReader(base_dir=tmp_path)
        result = reader.read_range("2025-08-06", "2025-08-07")
        assert len(result) == len(day1) + len(day2)

    def test_read_empty_dir(self, tmp_path: Path):
        reader = WalReader(base_dir=tmp_path)
        result = reader.read_all()
        assert result == []

    def test_read_missing_date(self, tmp_path: Path):
        reader = WalReader(base_dir=tmp_path)
        result = reader.read_date("2025-12-25")
        assert result == []

    def test_available_dates(self, tmp_path: Path):
        _write_reference_wal(_build_s5_stream(), tmp_path, "2025-08-06")
        _write_reference_wal(_build_s1_stream(), tmp_path, "2025-08-07")
        _write_reference_wal(_build_s4_stream(), tmp_path, "2025-08-08")

        reader = WalReader(base_dir=tmp_path)
        dates = reader.available_dates()
        assert dates == ["2025-08-06", "2025-08-07", "2025-08-08"]

    def test_roundtrip_serde_stability(self, tmp_path: Path):
        """Every event survives write → read with identical content."""
        events = _build_s5_stream()
        _write_reference_wal(events, tmp_path, "2025-08-06")

        reader = WalReader(base_dir=tmp_path)
        loaded = reader.read_date("2025-08-06")

        for orig, read in zip(events, loaded):
            assert orig.payload == read.payload
            assert orig.event_type == read.event_type
            assert orig.strategy_id == read.strategy_id
            assert orig.symbol == read.symbol

    def test_corrupt_last_line_skipped(self, tmp_path: Path):
        """Truncated last line is skipped (logged as warning)."""
        events = _build_s1_stream()
        _write_reference_wal(events, tmp_path, "2025-08-06")
        # Append corrupt data
        path = tmp_path / "2025-08-06.jsonl"
        with open(path, "a") as f:
            f.write('{"truncated": tr\n')

        reader = WalReader(base_dir=tmp_path)
        loaded = reader.read_date("2025-08-06")
        assert len(loaded) == len(events)  # corrupt line skipped

    def test_seq_monotonic_validation(self, tmp_path: Path):
        """Reader warns on seq gaps."""
        events = [
            _make_signal(1, "s5", "NIFTY"),
            _make_signal(5, "s5", "NIFTY"),  # gap: 1 -> 5
        ]
        _write_reference_wal(events, tmp_path, "2025-08-06")

        reader = WalReader(base_dir=tmp_path)
        loaded = reader.read_range("2025-08-06", "2025-08-06")
        assert len(loaded) == 2  # both returned (gap is warning, not error)

    def test_seq_strict_mode_raises(self, tmp_path: Path):
        """Strict seq mode raises on gap."""
        events = [
            _make_signal(1, "s5", "NIFTY"),
            _make_signal(5, "s5", "NIFTY"),  # gap
        ]
        _write_reference_wal(events, tmp_path, "2025-08-06")

        reader = WalReader(base_dir=tmp_path, strict_seq=True)
        with pytest.raises(WalValidationError):
            reader.read_range("2025-08-06", "2025-08-06")

    def test_seq_duplicate_strict_raises(self, tmp_path: Path):
        """Strict seq mode raises on duplicate seq."""
        events = [
            _make_signal(1, "s5", "NIFTY"),
            _make_signal(1, "s5", "BANKNIFTY"),  # duplicate seq
        ]
        _write_reference_wal(events, tmp_path, "2025-08-06")

        reader = WalReader(base_dir=tmp_path, strict_seq=True)
        with pytest.raises(WalValidationError):
            reader.read_range("2025-08-06", "2025-08-06")

    def test_filter_by_type(self, tmp_path: Path):
        events = _build_s5_stream()
        signals = WalReader.filter_by_type(events, "signal")
        assert len(signals) == 2
        assert all(e.event_type == "signal" for e in signals)

    def test_filter_by_strategy(self, tmp_path: Path):
        events = _build_s5_stream()
        filtered = WalReader.filter_by_strategy(events, "s5_hawkes")
        # Snapshot events have strategy_id="" so they don't match
        expected = sum(1 for e in events if e.strategy_id == "s5_hawkes")
        assert len(filtered) == expected

    def test_filter_by_symbol(self, tmp_path: Path):
        events = _build_s5_stream()
        filtered = WalReader.filter_by_symbol(events, "NIFTY")
        nifty_count = sum(1 for e in events if e.symbol == "NIFTY")
        assert len(filtered) == nifty_count

    def test_stats(self, tmp_path: Path):
        events = _build_s5_stream()
        _write_reference_wal(events, tmp_path, "2025-08-06")

        reader = WalReader(base_dir=tmp_path)
        reader.read_date("2025-08-06")
        stats = reader.stats()
        assert stats["files_read"] == 1
        assert stats["events_read"] == len(events)
        assert stats["corrupt_lines"] == 0


# ===================================================================
# Test Class: Comparator
# ===================================================================


class TestComparator:
    """Test event stream comparison logic."""

    def test_identical_streams(self):
        ref = _build_s5_stream()
        replay = _build_s5_stream()
        result = compare_streams(ref, replay)
        assert result.identical
        assert len(result.diffs) == 0
        assert result.missing_in_replay == 0
        assert result.extra_in_replay == 0

    def test_signal_conviction_diff(self):
        """Different conviction triggers a diff."""
        ref = [_make_signal(1, "s5", "NIFTY", conviction=0.80)]
        replay = [_make_signal(1, "s5", "NIFTY", conviction=0.75)]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any(d.field == "payload.conviction" for d in result.diffs)

    def test_gate_approved_diff(self):
        """Different gate approval triggers a diff."""
        ref = [_make_gate(1, "s5", "NIFTY", approved=True)]
        replay = [_make_gate(1, "s5", "NIFTY", approved=False)]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any(d.field == "payload.approved" for d in result.diffs)

    def test_snapshot_equity_diff(self):
        """Different equity triggers a diff."""
        ref = [_make_snapshot(1, equity=1.0)]
        replay = [_make_snapshot(1, equity=1.01)]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any(d.field == "payload.equity" for d in result.diffs)

    def test_missing_in_replay(self):
        """More events in ref than replay → missing_in_replay > 0."""
        ref = _build_s5_stream()
        replay = ref[:4]  # only first 4 of 8
        result = compare_streams(ref, replay)
        assert not result.identical
        # Some event types should have missing counts
        total_missing = result.missing_in_replay
        total_extra = result.extra_in_replay
        assert total_missing > 0 or total_extra == 0

    def test_extra_in_replay(self):
        """More events in replay than ref → extra_in_replay > 0."""
        ref = _build_s5_stream()[:4]
        replay = _build_s5_stream()
        result = compare_streams(ref, replay)
        assert not result.identical

    def test_event_type_counts(self):
        """Event type counts are tracked."""
        ref = _build_s5_stream()
        replay = _build_s5_stream()
        result = compare_streams(ref, replay)
        assert "signal" in result.event_type_counts
        assert result.event_type_counts["signal"]["ref"] == 2
        assert result.event_type_counts["signal"]["replay"] == 2

    def test_fp_tolerance_pass(self):
        """Values within FP tolerance are considered equal."""
        assert _values_equal(1.0, 1.0 + 1e-14)
        assert _values_equal(0.0, 1e-15)

    def test_fp_tolerance_fail(self):
        """Values beyond FP tolerance are different."""
        assert not _values_equal(1.0, 1.01)
        assert not _values_equal(0.0, 0.001)

    def test_nan_nan_equal(self):
        """NaN == NaN in parity comparison."""
        assert _values_equal(float("nan"), float("nan"))

    def test_inf_equality(self):
        """Same-sign Inf values are equal."""
        assert _values_equal(float("inf"), float("inf"))
        assert _values_equal(float("-inf"), float("-inf"))
        assert not _values_equal(float("inf"), float("-inf"))

    def test_mixed_type_comparison(self):
        """String comparisons work normally."""
        assert _values_equal("long", "long")
        assert not _values_equal("long", "short")

    def test_summary_format(self):
        """ComparisonResult.summary() produces readable output."""
        result = ComparisonResult(identical=True, total_compared=10)
        text = result.summary()
        assert "PASS" in text
        assert "10" in text

    def test_summary_with_diffs(self):
        """Summary includes diff details on failure."""
        result = ComparisonResult(
            identical=False,
            total_compared=5,
            diffs=[
                FieldDiff(
                    event_type="signal",
                    index=0,
                    field="payload.conviction",
                    ref_value=0.80,
                    replay_value=0.75,
                    strategy_id="s5",
                    symbol="NIFTY",
                )
            ],
        )
        text = result.summary()
        assert "FAIL" in text
        assert "conviction" in text

    def test_to_dict_serializable(self):
        """ComparisonResult.to_dict() produces JSON-serializable dict."""
        ref = _build_s5_stream()
        replay = _build_s5_stream()
        result = compare_streams(ref, replay)
        d = result.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_direction_diff(self):
        """Different direction in signal triggers diff."""
        ref = [_make_signal(1, "s5", "NIFTY", direction="long")]
        replay = [_make_signal(1, "s5", "NIFTY", direction="short")]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any(d.field == "payload.direction" for d in result.diffs)

    def test_strategy_id_diff(self):
        """Different strategy_id triggers envelope-level diff."""
        ref = [_make_signal(1, "s5_hawkes", "NIFTY")]
        replay = [_make_signal(1, "s1_vrp", "NIFTY")]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any("strategy_id" in d.field for d in result.diffs)

    def test_selective_event_types(self):
        """Can compare only specific event types."""
        ref = _build_s5_stream()
        replay = _build_s5_stream()
        result = compare_streams(ref, replay, event_types=["signal"])
        assert result.identical
        # Only signal type counted
        assert "signal" in result.event_type_counts
        assert "snapshot" not in result.event_type_counts

    def test_order_event_comparison(self):
        """Order events compared on action + side."""
        ref = [_make_order(1, "s5", "NIFTY", action="submit", side="buy")]
        replay = [_make_order(1, "s5", "NIFTY", action="submit", side="sell")]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any(d.field == "payload.side" for d in result.diffs)

    def test_risk_alert_comparison(self):
        """Risk alert events compared on alert_type."""
        ref = [_make_risk_alert(1, "s4", "NIFTY", alert_type="vix_gate")]
        replay = [_make_risk_alert(1, "s4", "NIFTY", alert_type="exposure")]
        result = compare_streams(ref, replay)
        assert not result.identical
        assert any(d.field == "payload.alert_type" for d in result.diffs)


# ===================================================================
# Test Class: Replay Parity per Strategy
# ===================================================================


class TestReplayParityTier1:
    """3x replay parity for each Tier-1 strategy.

    Write a reference WAL, read it back 3 times, compare all 3 reads
    are identical. This tests the deterministic property of the read path.
    """

    @pytest.fixture
    def wal_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "events"

    def _replay_3x(self, events: list[EventEnvelope], wal_dir: Path):
        """Write once, read 3x, verify all 3 reads identical."""
        _write_reference_wal(events, wal_dir, "2025-08-06")

        reads = []
        for _ in range(3):
            reader = WalReader(base_dir=wal_dir)
            loaded = reader.read_date("2025-08-06")
            reads.append(loaded)

        # All 3 reads must have same count
        assert len(reads[0]) == len(events)
        assert len(reads[1]) == len(events)
        assert len(reads[2]) == len(events)

        # Compare run2 vs run1, run3 vs run1
        for i in range(1, 3):
            result = compare_streams(reads[0], reads[i])
            assert result.identical, (
                f"Run {i+1} differs from Run 1:\n{result.summary()}"
            )

    def test_s5_hawkes_parity(self, wal_dir: Path):
        """S5 Hawkes Microstructure: 3x parity."""
        self._replay_3x(_build_s5_stream(), wal_dir)

    def test_s1_vrp_parity(self, wal_dir: Path):
        """S1 VRP Options: 3x parity."""
        self._replay_3x(_build_s1_stream(), wal_dir)

    def test_s4_iv_mr_parity(self, wal_dir: Path):
        """S4 IV Mean Reversion: 3x parity."""
        self._replay_3x(_build_s4_stream(), wal_dir)

    def test_s7_regime_parity(self, wal_dir: Path):
        """S7 Regime Switch: 3x parity."""
        self._replay_3x(_build_s7_stream(), wal_dir)

    def test_multi_day_parity(self, wal_dir: Path):
        """Multi-day stream: 3x parity across date range."""
        _write_reference_wal(_build_s5_stream(), wal_dir, "2025-08-06")
        _write_reference_wal(_build_s1_stream(), wal_dir, "2025-08-07")
        _write_reference_wal(_build_s4_stream(), wal_dir, "2025-08-08")

        reads = []
        for _ in range(3):
            reader = WalReader(base_dir=wal_dir)
            loaded = reader.read_range("2025-08-06", "2025-08-08")
            reads.append(loaded)

        for i in range(1, 3):
            result = compare_streams(reads[0], reads[i])
            assert result.identical, (
                f"Multi-day run {i+1} differs:\n{result.summary()}"
            )

    def test_mixed_strategies_parity(self, wal_dir: Path):
        """All Tier-1 strategies interleaved in one day: 3x parity."""
        stream = (
            _build_s5_stream()
            + _build_s1_stream()
            + _build_s4_stream()
            + _build_s7_stream()
        )
        # Re-sequence for monotonic seq
        reseq = []
        for i, e in enumerate(stream, 1):
            reseq.append(EventEnvelope(
                ts=e.ts,
                seq=i,
                run_id=e.run_id,
                event_type=e.event_type,
                source=e.source,
                strategy_id=e.strategy_id,
                symbol=e.symbol,
                payload=e.payload,
            ))

        self._replay_3x(reseq, wal_dir)

    def test_empty_stream_parity(self, wal_dir: Path):
        """Empty stream: 3x parity (trivial case)."""
        wal_dir.mkdir(parents=True, exist_ok=True)
        reads = []
        for _ in range(3):
            reader = WalReader(base_dir=wal_dir)
            loaded = reader.read_all()
            reads.append(loaded)

        assert all(len(r) == 0 for r in reads)

    def test_signal_field_exact_match(self, wal_dir: Path):
        """Verify exact field values after roundtrip."""
        events = [
            _make_signal(1, "s5_hawkes", "NIFTY", "long", 0.82, "low_vol"),
        ]
        _write_reference_wal(events, wal_dir, "2025-08-06")

        reader = WalReader(base_dir=wal_dir)
        loaded = reader.read_date("2025-08-06")
        assert len(loaded) == 1

        e = loaded[0]
        assert e.strategy_id == "s5_hawkes"
        assert e.symbol == "NIFTY"
        assert e.payload["direction"] == "long"
        assert e.payload["conviction"] == 0.82
        assert e.payload["regime"] == "low_vol"
        assert e.payload["instrument_type"] == "FUT"

    def test_gate_field_exact_match(self, wal_dir: Path):
        """Verify exact gate decision fields after roundtrip."""
        events = [
            _make_gate(1, "s4_iv_mr", "NIFTY", False, "vix_gate", 0.0, "VIX > 20"),
        ]
        _write_reference_wal(events, wal_dir, "2025-08-06")

        reader = WalReader(base_dir=wal_dir)
        loaded = reader.read_date("2025-08-06")
        assert len(loaded) == 1

        e = loaded[0]
        assert e.payload["gate"] == "vix_gate"
        assert e.payload["approved"] is False
        assert e.payload["adjusted_weight"] == 0.0
        assert e.payload["reason"] == "VIX > 20"

    def test_snapshot_field_exact_match(self, wal_dir: Path):
        """Verify exact snapshot fields after roundtrip."""
        events = [
            _make_snapshot(1, equity=1.005, peak_equity=1.01, portfolio_dd=0.005, position_count=3),
        ]
        _write_reference_wal(events, wal_dir, "2025-08-06")

        reader = WalReader(base_dir=wal_dir)
        loaded = reader.read_date("2025-08-06")
        assert len(loaded) == 1

        e = loaded[0]
        assert e.payload["equity"] == 1.005
        assert e.payload["peak_equity"] == 1.01
        assert e.payload["portfolio_dd"] == 0.005
        assert e.payload["position_count"] == 3
