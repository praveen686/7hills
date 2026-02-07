"""Phase 4 — WalQueryService tests.

Tests the read-only WAL query service that backs the Why Panel API.
Every test creates its own event data via EventLogWriter, then queries
it via WalQueryService.  No external state or derived data.

Core invariants:
  - get_signal_context returns 1:1 match of persisted signal payload
  - get_gate_decisions returns decisions linked to the correct signal
  - get_trade_decision_chain returns the full signal→gate→order→fill chain
  - available_dates returns only dates with .jsonl files
  - day_summary counts match actual events
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.events.envelope import EventEnvelope
from core.events.types import EventType
from core.events.serde import serialize_envelope

from engine.live.event_log import EventLogWriter
from engine.services.wal_query import WalQueryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_DAY1 = "2025-10-01T09:30:00.000000Z"
_TS_DAY2 = "2025-10-02T09:30:00.000000Z"
_RUN_ID = "why-test-001"


def _write_full_chain(base_dir: Path) -> dict:
    """Write a complete signal→gate→order→fill chain to WAL.

    Returns a dict of seq numbers for each event.
    """
    writer = EventLogWriter(
        base_dir=base_dir,
        run_id=_RUN_ID,
        fsync_policy="none",
    )

    # Signal
    sig_env = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="signal_journal",
        payload={
            "direction": "long",
            "conviction": 0.85,
            "instrument_type": "FUT",
            "strike": 0.0,
            "expiry": "",
            "ttl_bars": 5,
            "regime": "low_vol",
            "components": {
                "gex_regime": "positive",
                "raw_score": 0.72,
                "smoothed_score": 0.68,
                "reasoning": "GEX positive + elevated tick intensity",
            },
            "reasoning": "Hawkes intensity spike with positive GEX",
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_DAY1,
    )

    # Gate Decision
    gate_env = writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="signal_journal",
        payload={
            "signal_seq": sig_env.seq,
            "gate": "risk_check",
            "approved": True,
            "adjusted_weight": 0.12,
            "reason": "All gates passed",
            "vpin": 0.35,
            "portfolio_dd": 0.02,
            "strategy_dd": 0.01,
            "total_exposure": 0.45,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_DAY1,
    )

    # Order
    order_env = writer.emit(
        event_type=EventType.ORDER.value,
        source="execution_journal",
        payload={
            "order_id": "ord-001",
            "action": "submit",
            "side": "buy",
            "order_type": "market",
            "quantity": 50,
            "price": 24500.0,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_DAY1,
    )

    # Fill
    fill_env = writer.emit(
        event_type=EventType.FILL.value,
        source="execution_journal",
        payload={
            "order_id": "ord-001",
            "fill_id": "fill-001",
            "side": "buy",
            "quantity": 50,
            "price": 24501.5,
            "fees": 3.0,
            "is_partial": False,
        },
        strategy_id="s5_hawkes",
        symbol="NIFTY",
        ts=_TS_DAY1,
    )

    # Snapshot
    snap_env = writer.emit(
        event_type=EventType.SNAPSHOT.value,
        source="orchestrator",
        payload={
            "equity": 1.05,
            "peak_equity": 1.08,
            "portfolio_dd": 0.028,
            "total_exposure": 0.45,
            "vpin": 0.35,
            "position_count": 3,
            "regime": "low_vol",
        },
        ts=_TS_DAY1,
    )

    writer.close()

    return {
        "signal_seq": sig_env.seq,
        "gate_seq": gate_env.seq,
        "order_seq": order_env.seq,
        "fill_seq": fill_env.seq,
        "snap_seq": snap_env.seq,
    }


def _write_blocked_signal(base_dir: Path) -> dict:
    """Write a signal that gets blocked by a gate."""
    writer = EventLogWriter(
        base_dir=base_dir,
        run_id=_RUN_ID,
        fsync_policy="none",
    )

    sig = writer.emit(
        event_type=EventType.SIGNAL.value,
        source="signal_journal",
        payload={
            "direction": "short",
            "conviction": 0.60,
            "instrument_type": "FUT",
            "regime": "high_vol",
            "components": {"composite": 0.55, "sig_pctile": 0.30},
        },
        strategy_id="s1_vrp",
        symbol="BANKNIFTY",
        ts=_TS_DAY1,
    )

    gate = writer.emit(
        event_type=EventType.GATE_DECISION.value,
        source="signal_journal",
        payload={
            "signal_seq": sig.seq,
            "gate": "dd_strategy",
            "approved": False,
            "adjusted_weight": 0.0,
            "reason": "Strategy drawdown exceeds 5%",
            "vpin": 0.65,
            "portfolio_dd": 0.04,
            "strategy_dd": 0.06,
            "total_exposure": 0.80,
        },
        strategy_id="s1_vrp",
        symbol="BANKNIFTY",
        ts=_TS_DAY1,
    )

    # Risk alert for the block
    alert = writer.emit(
        event_type=EventType.RISK_ALERT.value,
        source="orchestrator",
        payload={
            "alert_type": "dd_strategy",
            "new_state": "blocked",
            "threshold": 0.05,
            "current_value": 0.06,
            "detail": "Strategy drawdown exceeds 5%",
        },
        strategy_id="s1_vrp",
        symbol="BANKNIFTY",
        ts=_TS_DAY1,
    )

    writer.close()
    return {
        "signal_seq": sig.seq,
        "gate_seq": gate.seq,
        "alert_seq": alert.seq,
    }


def _write_multi_day(base_dir: Path) -> None:
    """Write events across two days."""
    writer = EventLogWriter(
        base_dir=base_dir,
        run_id=_RUN_ID,
        fsync_policy="none",
    )

    # Day 1
    writer.emit(
        event_type=EventType.SIGNAL.value,
        source="test",
        payload={"direction": "long", "conviction": 0.7, "instrument_type": "FUT",
                 "regime": "normal", "components": {}},
        strategy_id="s4_iv_mr",
        symbol="NIFTY",
        ts=_TS_DAY1,
    )

    # Day 2
    writer.emit(
        event_type=EventType.SIGNAL.value,
        source="test",
        payload={"direction": "short", "conviction": 0.6, "instrument_type": "FUT",
                 "regime": "high_vol", "components": {}},
        strategy_id="s4_iv_mr",
        symbol="NIFTY",
        ts=_TS_DAY2,
    )

    writer.close()


# ===================================================================
# Test Class: Signal Context
# ===================================================================


class TestSignalContext:
    """get_signal_context returns 1:1 match of persisted signal payload."""

    def test_signal_found(self, tmp_path: Path):
        """Signal context matches persisted payload fields."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(seqs["signal_seq"], "2025-10-01")

        assert ctx is not None
        assert ctx["signal_seq"] == seqs["signal_seq"]
        assert ctx["strategy_id"] == "s5_hawkes"
        assert ctx["symbol"] == "NIFTY"
        assert ctx["direction"] == "long"
        assert ctx["conviction"] == 0.85
        assert ctx["instrument_type"] == "FUT"
        assert ctx["regime"] == "low_vol"
        assert ctx["components"]["gex_regime"] == "positive"
        assert ctx["components"]["raw_score"] == 0.72
        assert "Hawkes" in ctx["reasoning"]

    def test_signal_not_found(self, tmp_path: Path):
        """Returns None for non-existent signal seq."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(9999, "2025-10-01")
        assert ctx is None

    def test_wrong_date_returns_none(self, tmp_path: Path):
        """Returns None if signal exists on different date."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(seqs["signal_seq"], "2025-10-02")
        assert ctx is None

    def test_blocked_signal_context(self, tmp_path: Path):
        """Blocked signal still returns full context."""
        events_dir = tmp_path / "events"
        seqs = _write_blocked_signal(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(seqs["signal_seq"], "2025-10-01")

        assert ctx is not None
        assert ctx["direction"] == "short"
        assert ctx["conviction"] == 0.60
        assert ctx["components"]["composite"] == 0.55

    def test_gate_event_not_returned_as_signal(self, tmp_path: Path):
        """A gate_decision event should not be found as a signal."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(seqs["gate_seq"], "2025-10-01")
        assert ctx is None


# ===================================================================
# Test Class: Gate Decisions
# ===================================================================


class TestGateDecisions:
    """get_gate_decisions returns decisions linked to the correct signal."""

    def test_approved_gate(self, tmp_path: Path):
        """Returns approved gate decision for a signal."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        decisions = svc.get_gate_decisions(seqs["signal_seq"], "2025-10-01")

        assert len(decisions) == 1
        d = decisions[0]
        assert d["approved"] is True
        assert d["gate"] == "risk_check"
        assert d["adjusted_weight"] == 0.12
        assert d["vpin"] == 0.35
        assert d["portfolio_dd"] == 0.02
        assert d["strategy_dd"] == 0.01

    def test_blocked_gate(self, tmp_path: Path):
        """Returns blocked gate decision with reason."""
        events_dir = tmp_path / "events"
        seqs = _write_blocked_signal(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        decisions = svc.get_gate_decisions(seqs["signal_seq"], "2025-10-01")

        assert len(decisions) == 1
        d = decisions[0]
        assert d["approved"] is False
        assert "drawdown" in d["reason"].lower()
        assert d["strategy_dd"] == 0.06

    def test_no_gate_for_missing_signal(self, tmp_path: Path):
        """Returns empty list for non-existent signal."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        decisions = svc.get_gate_decisions(9999, "2025-10-01")
        assert decisions == []

    def test_gate_has_risk_metrics(self, tmp_path: Path):
        """Gate decision includes all risk metric fields."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        decisions = svc.get_gate_decisions(seqs["signal_seq"], "2025-10-01")

        d = decisions[0]
        required_fields = {"seq", "ts", "gate", "approved", "adjusted_weight",
                          "reason", "vpin", "portfolio_dd", "strategy_dd",
                          "total_exposure"}
        assert required_fields.issubset(set(d.keys()))


# ===================================================================
# Test Class: Trade Decision Chain
# ===================================================================


class TestTradeDecisionChain:
    """get_trade_decision_chain returns the full signal→gate→order→fill chain."""

    def test_full_chain(self, tmp_path: Path):
        """Returns complete chain: signal + gate + order + fill + snapshot."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")

        assert chain is not None
        assert chain["strategy_id"] == "s5_hawkes"
        assert chain["symbol"] == "NIFTY"
        assert chain["date"] == "2025-10-01"
        assert len(chain["signals"]) == 1
        assert len(chain["gates"]) == 1
        assert len(chain["orders"]) == 1
        assert len(chain["fills"]) == 1
        assert chain["snapshot"] is not None

    def test_blocked_chain(self, tmp_path: Path):
        """Blocked signal has signal + gate + risk_alert but no order/fill."""
        events_dir = tmp_path / "events"
        _write_blocked_signal(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s1_vrp", "BANKNIFTY", "2025-10-01")

        assert chain is not None
        assert len(chain["signals"]) == 1
        assert len(chain["gates"]) == 1
        assert len(chain["orders"]) == 0
        assert len(chain["fills"]) == 0
        assert len(chain["risk_alerts"]) == 1

    def test_chain_not_found(self, tmp_path: Path):
        """Returns None for non-existent strategy/symbol."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s99_fake", "NIFTY", "2025-10-01")
        assert chain is None

    def test_chain_wrong_date(self, tmp_path: Path):
        """Returns None for wrong date."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-05")
        assert chain is None

    def test_signal_payload_in_chain(self, tmp_path: Path):
        """Signal event in chain includes full payload."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")

        sig = chain["signals"][0]
        assert sig["payload"]["direction"] == "long"
        assert sig["payload"]["conviction"] == 0.85
        assert sig["payload"]["components"]["gex_regime"] == "positive"

    def test_gate_payload_in_chain(self, tmp_path: Path):
        """Gate event in chain includes full payload."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")

        gate = chain["gates"][0]
        assert gate["payload"]["approved"] is True
        assert gate["payload"]["vpin"] == 0.35

    def test_order_payload_in_chain(self, tmp_path: Path):
        """Order event in chain includes order_id and side."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")

        order = chain["orders"][0]
        assert order["payload"]["order_id"] == "ord-001"
        assert order["payload"]["side"] == "buy"

    def test_fill_payload_in_chain(self, tmp_path: Path):
        """Fill event in chain includes fill_id and price."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")

        fill = chain["fills"][0]
        assert fill["payload"]["fill_id"] == "fill-001"
        assert fill["payload"]["price"] == 24501.5
        assert fill["payload"]["fees"] == 3.0


# ===================================================================
# Test Class: Utilities
# ===================================================================


class TestUtilities:
    """available_dates, day_summary, get_event_by_seq."""

    def test_available_dates(self, tmp_path: Path):
        """Returns sorted list of dates with JSONL files."""
        events_dir = tmp_path / "events"
        _write_multi_day(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        dates = svc.available_dates()

        assert "2025-10-01" in dates
        assert "2025-10-02" in dates
        assert dates == sorted(dates)

    def test_available_dates_empty(self, tmp_path: Path):
        """Returns empty list when no JSONL files exist."""
        events_dir = tmp_path / "events"
        events_dir.mkdir(parents=True)

        svc = WalQueryService(base_dir=events_dir)
        assert svc.available_dates() == []

    def test_day_summary_counts(self, tmp_path: Path):
        """day_summary counts match actual events."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        summary = svc.day_summary("2025-10-01")

        assert summary["date"] == "2025-10-01"
        assert summary["total_events"] == 5  # signal + gate + order + fill + snapshot
        assert summary["event_counts"][EventType.SIGNAL.value] == 1
        assert summary["event_counts"][EventType.GATE_DECISION.value] == 1
        assert summary["event_counts"][EventType.ORDER.value] == 1
        assert summary["event_counts"][EventType.FILL.value] == 1
        assert summary["event_counts"][EventType.SNAPSHOT.value] == 1

    def test_day_summary_strategies(self, tmp_path: Path):
        """day_summary lists unique strategy IDs."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        summary = svc.day_summary("2025-10-01")

        assert "s5_hawkes" in summary["strategies"]
        assert "NIFTY" in summary["symbols"]

    def test_day_summary_empty_date(self, tmp_path: Path):
        """day_summary for empty date returns zero counts."""
        events_dir = tmp_path / "events"
        events_dir.mkdir(parents=True)

        svc = WalQueryService(base_dir=events_dir)
        summary = svc.day_summary("2025-12-25")

        assert summary["total_events"] == 0
        assert summary["event_counts"] == {}

    def test_get_event_by_seq(self, tmp_path: Path):
        """get_event_by_seq returns correct event."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        event = svc.get_event_by_seq(seqs["signal_seq"], day="2025-10-01")

        assert event is not None
        assert event.event_type == EventType.SIGNAL.value
        assert event.strategy_id == "s5_hawkes"

    def test_get_event_by_seq_not_found(self, tmp_path: Path):
        """Returns None for non-existent seq."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        event = svc.get_event_by_seq(9999, day="2025-10-01")
        assert event is None

    def test_get_events_by_type(self, tmp_path: Path):
        """get_events_by_type filters correctly."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        signals = svc.get_events_by_type(EventType.SIGNAL.value, "2025-10-01")
        assert len(signals) == 1
        assert signals[0].event_type == EventType.SIGNAL.value

    def test_get_events_by_strategy(self, tmp_path: Path):
        """get_events_by_strategy filters correctly."""
        events_dir = tmp_path / "events"
        _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        events = svc.get_events_by_strategy("s5_hawkes", "2025-10-01")

        # signal + gate + order + fill = 4 (snapshot has no strategy_id)
        assert len(events) == 4
        for e in events:
            assert e.strategy_id == "s5_hawkes"


# ===================================================================
# Test Class: Strategy-Specific Why Fields
# ===================================================================


class TestStrategyWhyFields:
    """Components field contains strategy-specific metadata per contract."""

    def test_s5_hawkes_fields(self, tmp_path: Path):
        """S5 Hawkes signal includes gex_regime, components, reasoning."""
        events_dir = tmp_path / "events"
        seqs = _write_full_chain(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(seqs["signal_seq"], "2025-10-01")

        assert ctx["components"]["gex_regime"] == "positive"
        assert ctx["components"]["raw_score"] == 0.72
        assert ctx["components"]["smoothed_score"] == 0.68
        assert "GEX positive" in ctx["components"]["reasoning"]

    def test_s1_vrp_fields(self, tmp_path: Path):
        """S1 VRP signal includes composite, sig_pctile."""
        events_dir = tmp_path / "events"
        seqs = _write_blocked_signal(events_dir)

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(seqs["signal_seq"], "2025-10-01")

        assert ctx["components"]["composite"] == 0.55
        assert ctx["components"]["sig_pctile"] == 0.30

    def test_s4_iv_mr_fields(self, tmp_path: Path):
        """S4 IV MR signal includes atm_iv, iv_pctile, spot."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
        )
        sig = writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={
                "direction": "long",
                "conviction": 0.70,
                "instrument_type": "FUT",
                "regime": "low_vol",
                "components": {
                    "atm_iv": 0.1234,
                    "iv_pctile": 0.1500,
                    "spot": 24300.50,
                },
            },
            strategy_id="s4_iv_mr",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )
        writer.close()

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(sig.seq, "2025-10-01")

        assert ctx["components"]["atm_iv"] == 0.1234
        assert ctx["components"]["iv_pctile"] == 0.15
        assert ctx["components"]["spot"] == 24300.50

    def test_s7_regime_fields(self, tmp_path: Path):
        """S7 Regime signal includes sub_strategy, entropy, mi, z_score."""
        events_dir = tmp_path / "events"
        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
        )
        sig = writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={
                "direction": "long",
                "conviction": 0.75,
                "instrument_type": "FUT",
                "regime": "trending",
                "components": {
                    "sub_strategy": "trend_following",
                    "regime": "trending",
                    "entropy": 0.2341,
                    "mi": 0.4567,
                    "z_score": -1.85,
                    "pct_b": 0.15,
                    "confidence": 0.80,
                },
            },
            strategy_id="s7_regime",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )
        writer.close()

        svc = WalQueryService(base_dir=events_dir)
        ctx = svc.get_signal_context(sig.seq, "2025-10-01")

        assert ctx["components"]["sub_strategy"] == "trend_following"
        assert ctx["components"]["entropy"] == 0.2341
        assert ctx["components"]["mi"] == 0.4567
        assert ctx["components"]["z_score"] == -1.85
        assert ctx["components"]["pct_b"] == 0.15
        assert ctx["components"]["confidence"] == 0.80


# ===================================================================
# Test Class: Multi-Strategy Chain Isolation
# ===================================================================


class TestChainIsolation:
    """Decision chains for different strategies don't leak into each other."""

    def test_two_strategies_same_day(self, tmp_path: Path):
        """Two strategies on same day produce independent chains."""
        events_dir = tmp_path / "events"

        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
        )

        # S5 signal for NIFTY
        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "long", "conviction": 0.8,
                     "instrument_type": "FUT", "regime": "normal",
                     "components": {"gex_regime": "positive"}},
            strategy_id="s5_hawkes",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )

        # S1 signal for NIFTY
        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "short", "conviction": 0.6,
                     "instrument_type": "FUT", "regime": "high_vol",
                     "components": {"composite": 0.45}},
            strategy_id="s1_vrp",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )

        # S5 gate
        writer.emit(
            event_type=EventType.GATE_DECISION.value,
            source="test",
            payload={"gate": "pass", "approved": True, "adjusted_weight": 0.1,
                     "vpin": 0.3, "portfolio_dd": 0.01, "strategy_dd": 0.005,
                     "total_exposure": 0.4},
            strategy_id="s5_hawkes",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )

        # S1 gate (blocked)
        writer.emit(
            event_type=EventType.GATE_DECISION.value,
            source="test",
            payload={"gate": "dd_check", "approved": False, "adjusted_weight": 0.0,
                     "reason": "DD exceeded", "vpin": 0.6,
                     "portfolio_dd": 0.04, "strategy_dd": 0.07,
                     "total_exposure": 0.8},
            strategy_id="s1_vrp",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )

        writer.close()

        svc = WalQueryService(base_dir=events_dir)

        # S5 chain should NOT contain S1's events
        s5_chain = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")
        assert s5_chain is not None
        assert len(s5_chain["signals"]) == 1
        assert s5_chain["signals"][0]["payload"]["direction"] == "long"
        assert len(s5_chain["gates"]) == 1
        assert s5_chain["gates"][0]["payload"]["approved"] is True

        # S1 chain should NOT contain S5's events
        s1_chain = svc.get_trade_decision_chain("s1_vrp", "NIFTY", "2025-10-01")
        assert s1_chain is not None
        assert len(s1_chain["signals"]) == 1
        assert s1_chain["signals"][0]["payload"]["direction"] == "short"
        assert len(s1_chain["gates"]) == 1
        assert s1_chain["gates"][0]["payload"]["approved"] is False

    def test_same_strategy_different_symbols(self, tmp_path: Path):
        """Same strategy on different symbols produces isolated chains."""
        events_dir = tmp_path / "events"

        writer = EventLogWriter(
            base_dir=events_dir,
            run_id=_RUN_ID,
            fsync_policy="none",
        )

        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "long", "conviction": 0.8,
                     "instrument_type": "FUT", "regime": "normal",
                     "components": {}},
            strategy_id="s5_hawkes",
            symbol="NIFTY",
            ts=_TS_DAY1,
        )

        writer.emit(
            event_type=EventType.SIGNAL.value,
            source="test",
            payload={"direction": "short", "conviction": 0.65,
                     "instrument_type": "FUT", "regime": "normal",
                     "components": {}},
            strategy_id="s5_hawkes",
            symbol="BANKNIFTY",
            ts=_TS_DAY1,
        )

        writer.close()

        svc = WalQueryService(base_dir=events_dir)

        nifty = svc.get_trade_decision_chain("s5_hawkes", "NIFTY", "2025-10-01")
        bnf = svc.get_trade_decision_chain("s5_hawkes", "BANKNIFTY", "2025-10-01")

        assert nifty is not None
        assert bnf is not None
        assert len(nifty["signals"]) == 1
        assert nifty["signals"][0]["payload"]["direction"] == "long"
        assert len(bnf["signals"]) == 1
        assert bnf["signals"][0]["payload"]["direction"] == "short"
