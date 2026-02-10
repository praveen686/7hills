"""Tests for circuit breaker auto-flatten behavior.

Covers:
  - Bug #6: RiskManager.positions_to_flatten() generates flat targets when CB active
  - Bug #8: Orchestrator.run_day() skips signals and auto-flattens when CB active
  - Circuit breaker state is properly checked and propagated
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantlaxmi.core.allocator.meta import TargetPosition
from quantlaxmi.core.allocator.regime import VIXRegime, VIXRegimeType
from quantlaxmi.core.risk.limits import RiskLimits
from quantlaxmi.core.risk.manager import (
    GateResult,
    PortfolioState as RiskPortfolioState,
    RiskManager,
)
from quantlaxmi.engine.orchestrator import Orchestrator
from quantlaxmi.engine.state import PortfolioState, Position
from quantlaxmi.strategies.protocol import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target(
    symbol: str = "NIFTY",
    direction: str = "long",
    weight: float = 0.05,
    strategy_id: str = "s1_vrp",
    instrument_type: str = "FUT",
) -> TargetPosition:
    return TargetPosition(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        weight=weight,
        instrument_type=instrument_type,
    )


def _risk_state(**kwargs) -> RiskPortfolioState:
    return RiskPortfolioState(**kwargs)


def _rm(limits: RiskLimits | None = None) -> RiskManager:
    return RiskManager(limits=limits or RiskLimits())


def _make_position(
    strategy_id: str = "s1_vrp",
    symbol: str = "NIFTY",
    direction: str = "long",
    weight: float = 0.05,
    instrument_type: str = "FUT",
    entry_price: float = 23000.0,
) -> Position:
    return Position(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        weight=weight,
        instrument_type=instrument_type,
        entry_date="2026-01-15",
        entry_price=entry_price,
    )


class FakeStrategy:
    """Minimal fake strategy that returns configurable signals."""

    def __init__(self, strategy_id: str, signals: list[Signal] | None = None):
        self.strategy_id = strategy_id
        self._signals = signals or []
        self.scan_called = False

    def scan(self, d, store):
        self.scan_called = True
        return self._signals


def _make_orchestrator(
    positions: list[Position] | None = None,
    circuit_breaker_active: bool = False,
    strategies: list | None = None,
) -> Orchestrator:
    """Build a minimal Orchestrator with mocked dependencies for testing."""
    # Temporary state file
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    state_file = Path(tmp.name)
    tmp.close()

    # Mock store
    mock_store = MagicMock()

    # Create orchestrator with mocks
    with patch("quantlaxmi.engine.orchestrator.PortfolioState.load") as mock_load:
        state = PortfolioState()
        mock_load.return_value = state
        orch = Orchestrator(
            store=mock_store,
            state_file=state_file,
        )

    # Set up positions
    if positions:
        for pos in positions:
            orch.state.open_position(pos)

    # Set circuit breaker state
    if circuit_breaker_active:
        orch.risk_manager._circuit_breaker_active = True

    # Register fake strategies if provided
    if strategies:
        for s in strategies:
            orch.registry.register(s)

    # Mock event log to avoid file I/O
    orch._event_log = MagicMock()
    orch._event_log.event_count = 0
    orch._signal_journal = MagicMock()
    orch._exec_journal = MagicMock()

    return orch


# ===================================================================
# 1. RiskManager.positions_to_flatten() — Bug #6
# ===================================================================

class TestPositionsToFlatten:
    """Test that RiskManager generates flat targets when circuit breaker fires."""

    def test_returns_empty_when_cb_inactive(self):
        rm = _rm()
        assert rm.circuit_breaker_active is False
        result = rm.positions_to_flatten([
            {"strategy_id": "s1_vrp", "symbol": "NIFTY", "instrument_type": "FUT"},
        ])
        assert result == []

    def test_returns_flat_targets_when_cb_active(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        positions = [
            {"strategy_id": "s1_vrp", "symbol": "NIFTY", "instrument_type": "FUT"},
            {"strategy_id": "s4_iv_mr", "symbol": "BANKNIFTY", "instrument_type": "FUT"},
        ]
        result = rm.positions_to_flatten(positions)
        assert len(result) == 2
        for t in result:
            assert t.direction == "flat"
            assert t.weight == 0.0
            assert t.metadata.get("exit_reason") == "circuit_breaker"

    def test_flat_targets_match_position_symbols(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        positions = [
            {"strategy_id": "s1_vrp", "symbol": "NIFTY", "instrument_type": "FUT"},
            {"strategy_id": "s5_hawkes", "symbol": "RELIANCE", "instrument_type": "CE"},
        ]
        result = rm.positions_to_flatten(positions)
        symbols = {t.symbol for t in result}
        assert symbols == {"NIFTY", "RELIANCE"}
        strat_ids = {t.strategy_id for t in result}
        assert strat_ids == {"s1_vrp", "s5_hawkes"}

    def test_flat_targets_preserve_instrument_type(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        positions = [
            {"strategy_id": "s1_vrp", "symbol": "NIFTY", "instrument_type": "CE"},
        ]
        result = rm.positions_to_flatten(positions)
        assert result[0].instrument_type == "CE"

    def test_flat_targets_default_instrument_type(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        positions = [
            {"strategy_id": "s1_vrp", "symbol": "NIFTY"},  # no instrument_type key
        ]
        result = rm.positions_to_flatten(positions)
        assert result[0].instrument_type == "FUT"

    def test_empty_positions_returns_empty(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        result = rm.positions_to_flatten([])
        assert result == []

    def test_cb_activated_by_vpin_then_flatten(self):
        """End-to-end: VPIN triggers CB, then flatten generates targets."""
        rm = _rm()
        # Trigger circuit breaker via VPIN
        rm.check([_target()], _risk_state(vpin=0.80))
        assert rm.circuit_breaker_active is True

        # Now auto-flatten should work
        result = rm.positions_to_flatten([
            {"strategy_id": "s1_vrp", "symbol": "NIFTY", "instrument_type": "FUT"},
        ])
        assert len(result) == 1
        assert result[0].direction == "flat"

    def test_cb_activated_by_dd_then_flatten(self):
        """End-to-end: Portfolio DD triggers CB, then flatten generates targets."""
        rm = _rm()
        # Trigger circuit breaker via drawdown
        rm.check([_target()], _risk_state(equity=0.90, peak_equity=1.0))
        assert rm.circuit_breaker_active is True

        result = rm.positions_to_flatten([
            {"strategy_id": "s4_iv_mr", "symbol": "BANKNIFTY", "instrument_type": "FUT"},
        ])
        assert len(result) == 1
        assert result[0].direction == "flat"
        assert result[0].metadata.get("exit_reason") == "circuit_breaker"

    def test_after_reset_no_flatten(self):
        """After circuit breaker reset, positions_to_flatten returns empty."""
        rm = _rm()
        rm._circuit_breaker_active = True
        rm.reset_circuit_breaker()
        assert rm.circuit_breaker_active is False
        result = rm.positions_to_flatten([
            {"strategy_id": "s1_vrp", "symbol": "NIFTY", "instrument_type": "FUT"},
        ])
        assert result == []

    def test_multiple_positions_all_flattened(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        positions = [
            {"strategy_id": f"s{i}", "symbol": sym, "instrument_type": "FUT"}
            for i, sym in enumerate(["NIFTY", "BANKNIFTY", "FINNIFTY", "RELIANCE", "TCS"])
        ]
        result = rm.positions_to_flatten(positions)
        assert len(result) == 5
        assert all(t.direction == "flat" for t in result)
        assert all(t.weight == 0.0 for t in result)


# ===================================================================
# 2. Orchestrator.run_day() circuit breaker check — Bug #8
# ===================================================================

class TestRunDayCircuitBreaker:
    """Test that run_day() skips signals and auto-flattens when CB active."""

    def test_run_day_skips_signals_when_cb_active(self):
        """When circuit breaker is active, strategies should NOT be scanned."""
        fake_strat = FakeStrategy("s1_vrp", signals=[
            Signal(
                strategy_id="s1_vrp",
                symbol="NIFTY",
                direction="long",
                conviction=0.8,
                instrument_type="FUT",
            ),
        ])
        orch = _make_orchestrator(
            circuit_breaker_active=True,
            strategies=[fake_strat],
        )

        # Mock detect_regime to return a normal regime
        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            summary = orch.run_day(date(2026, 2, 10))

        # Strategy should NOT have been scanned
        assert fake_strat.scan_called is False
        # No signals should be in summary
        assert summary["signals"] == []

    def test_run_day_flattens_positions_when_cb_active(self):
        """Open positions should be closed when CB active."""
        pos = _make_position(strategy_id="s1_vrp", symbol="NIFTY", weight=0.05)
        orch = _make_orchestrator(
            positions=[pos],
            circuit_breaker_active=True,
        )

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            # Mock _get_spot to return a price
            with patch.object(orch, "_get_spot", return_value=23500.0):
                summary = orch.run_day(date(2026, 2, 10))

        # Position should have been closed
        assert len(summary["actions"]) == 1
        action = summary["actions"][0]
        assert action["action"] == "close"
        assert action["symbol"] == "NIFTY"
        assert action["reason"] == "circuit_breaker"

        # No positions should remain
        assert len(orch.state.active_positions()) == 0

    def test_run_day_flattens_multiple_positions(self):
        """All open positions should be flattened, not just one."""
        positions = [
            _make_position(strategy_id="s1_vrp", symbol="NIFTY", weight=0.05),
            _make_position(strategy_id="s4_iv_mr", symbol="BANKNIFTY", weight=0.10,
                           entry_price=50000.0),
        ]
        orch = _make_orchestrator(
            positions=positions,
            circuit_breaker_active=True,
        )

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            with patch.object(orch, "_get_spot", return_value=23500.0):
                summary = orch.run_day(date(2026, 2, 10))

        # Both positions should have been closed
        assert len(summary["actions"]) == 2
        assert all(a["action"] == "close" for a in summary["actions"])
        closed_symbols = {a["symbol"] for a in summary["actions"]}
        assert closed_symbols == {"NIFTY", "BANKNIFTY"}

        # No positions should remain
        assert len(orch.state.active_positions()) == 0

    def test_run_day_still_updates_state_when_cb_active(self):
        """State should still be updated (scan_count, equity_history, etc.)."""
        orch = _make_orchestrator(circuit_breaker_active=True)
        initial_scan_count = orch.state.scan_count

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            summary = orch.run_day(date(2026, 2, 10))

        assert orch.state.scan_count == initial_scan_count + 1
        assert orch.state.last_scan_date == "2026-02-10"
        assert len(orch.state.equity_history) == 1

    def test_run_day_emits_snapshot_when_cb_active(self):
        """Portfolio snapshot should still be emitted in CB mode."""
        orch = _make_orchestrator(circuit_breaker_active=True)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            orch.run_day(date(2026, 2, 10))

        # Check that event_log.emit was called (for risk alert and/or snapshot)
        assert orch._event_log.emit.called

    def test_run_day_emits_risk_alert_when_cb_active(self):
        """A risk alert event should be emitted when CB triggers flatten."""
        orch = _make_orchestrator(circuit_breaker_active=True)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            orch.run_day(date(2026, 2, 10))

        # Find the risk_alert emit call
        risk_alert_calls = [
            call for call in orch._event_log.emit.call_args_list
            if call.kwargs.get("event_type") == "risk_alert"
               or (call.args and call.args[0] == "risk_alert")
        ]
        # Should have at least one risk alert (circuit_breaker_flatten)
        assert len(risk_alert_calls) >= 1

    def test_run_day_no_flatten_when_cb_inactive(self):
        """Normal flow: CB inactive, strategies scanned normally."""
        fake_strat = FakeStrategy("s1_vrp", signals=[])
        orch = _make_orchestrator(
            circuit_breaker_active=False,
            strategies=[fake_strat],
        )

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            summary = orch.run_day(date(2026, 2, 10))

        # Strategy SHOULD have been scanned
        assert fake_strat.scan_called is True

    def test_run_day_returns_regime_in_summary(self):
        """Even in CB mode, regime should be in summary."""
        orch = _make_orchestrator(circuit_breaker_active=True)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=25.0,
                regime=VIXRegimeType.ELEVATED,
                date=date(2026, 2, 10),
            )
            summary = orch.run_day(date(2026, 2, 10))

        assert summary["regime"]["type"] == "elevated"
        assert summary["regime"]["vix"] == 25.0

    def test_run_day_no_positions_cb_active_no_crash(self):
        """CB active but no open positions should not crash."""
        orch = _make_orchestrator(circuit_breaker_active=True)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            summary = orch.run_day(date(2026, 2, 10))

        assert summary["actions"] == []
        assert summary["signals"] == []


# ===================================================================
# 3. Circuit breaker state check integration
# ===================================================================

class TestCircuitBreakerStateCheck:
    """Test circuit breaker state is properly checked and propagated."""

    def test_circuit_breaker_property_exists(self):
        rm = _rm()
        assert hasattr(rm, "circuit_breaker_active")
        assert isinstance(rm.circuit_breaker_active, bool)

    def test_positions_to_flatten_method_exists(self):
        rm = _rm()
        assert hasattr(rm, "positions_to_flatten")
        assert callable(rm.positions_to_flatten)

    def test_circuit_breaker_set_by_vpin(self):
        rm = _rm()
        rm.check([_target()], _risk_state(vpin=0.80))
        assert rm.circuit_breaker_active is True

    def test_circuit_breaker_set_by_portfolio_dd(self):
        rm = _rm()
        rm.check([_target()], _risk_state(equity=0.90, peak_equity=1.0))
        assert rm.circuit_breaker_active is True

    def test_circuit_breaker_not_set_by_strategy_dd(self):
        rm = _rm()
        state = _risk_state(
            strategy_equity={"s1_vrp": 0.95},
            strategy_peaks={"s1_vrp": 1.0},
        )
        rm.check([_target(strategy_id="s1_vrp")], state)
        assert rm.circuit_breaker_active is False

    def test_circuit_breaker_not_set_by_concentration(self):
        rm = _rm()
        state = _risk_state(positions={"NIFTY": {"weight": 0.20}})
        rm.check([_target(symbol="NIFTY", weight=0.05)], state)
        assert rm.circuit_breaker_active is False

    def test_circuit_breaker_persists_across_checks(self):
        """Once set, CB stays active until explicitly reset."""
        rm = _rm()
        # Trigger CB
        rm.check([_target()], _risk_state(vpin=0.80))
        assert rm.circuit_breaker_active is True

        # Check again with normal state — CB should still be active
        rm.check([_target()], _risk_state(vpin=0.10))
        # Note: the check itself won't reset CB — that's by design
        # The VPIN gate won't fire, but CB was already set
        assert rm.circuit_breaker_active is True

    def test_circuit_breaker_reset_clears_state(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        rm.reset_circuit_breaker()
        assert rm.circuit_breaker_active is False

    def test_orchestrator_uses_risk_manager_cb(self):
        """Orchestrator should check risk_manager.circuit_breaker_active."""
        orch = _make_orchestrator(circuit_breaker_active=True)
        assert orch.risk_manager.circuit_breaker_active is True

    def test_flatten_exit_reason_is_circuit_breaker(self):
        """Closed trades from CB flatten should have exit_reason='circuit_breaker'."""
        pos = _make_position(strategy_id="s1_vrp", symbol="NIFTY", weight=0.05)
        orch = _make_orchestrator(
            positions=[pos],
            circuit_breaker_active=True,
        )

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime.return_value = VIXRegime(
                vix=15.0,
                regime=VIXRegimeType.NORMAL,
                date=date(2026, 2, 10),
            )
            with patch.object(orch, "_get_spot", return_value=23500.0):
                orch.run_day(date(2026, 2, 10))

        # Check closed trade has circuit_breaker exit reason
        assert len(orch.state.closed_trades) == 1
        trade = orch.state.closed_trades[0]
        assert trade.exit_reason == "circuit_breaker"
        assert trade.strategy_id == "s1_vrp"
        assert trade.symbol == "NIFTY"
