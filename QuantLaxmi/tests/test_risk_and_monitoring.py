"""Tests for Bug #14, #15, #16, #24 — risk Greek limits, reconciliation wiring,
drift monitor wiring, and fill event emission.

Run: python -m pytest tests/test_risk_and_monitoring.py -v --timeout=60
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bug #14: Greek limits
# ---------------------------------------------------------------------------
from quantlaxmi.core.risk.limits import RiskLimits
from quantlaxmi.core.risk.manager import (
    GateResult,
    PortfolioState as RiskPortfolioState,
    RiskCheckResult,
    RiskManager,
)
from quantlaxmi.core.allocator.meta import TargetPosition


class TestGreekLimits:
    """Bug #14: vega and theta limits in RiskLimits and RiskManager."""

    def test_risk_limits_has_vega_theta_defaults(self):
        """RiskLimits should have max_portfolio_vega and max_portfolio_theta."""
        limits = RiskLimits()
        assert limits.max_portfolio_vega == 50000.0
        assert limits.max_portfolio_theta == -25000.0

    def test_risk_limits_custom_vega_theta(self):
        """Custom vega/theta limits should be accepted."""
        limits = RiskLimits(max_portfolio_vega=30000.0, max_portfolio_theta=-10000.0)
        assert limits.max_portfolio_vega == 30000.0
        assert limits.max_portfolio_theta == -10000.0

    def test_portfolio_state_has_greeks(self):
        """RiskPortfolioState should carry portfolio_vega and portfolio_theta."""
        state = RiskPortfolioState(portfolio_vega=25000.0, portfolio_theta=-12000.0)
        assert state.portfolio_vega == 25000.0
        assert state.portfolio_theta == -12000.0

    def test_vega_breach_blocks_entry(self):
        """Exceeding max_portfolio_vega should block new entries."""
        limits = RiskLimits(max_portfolio_vega=50000.0)
        rm = RiskManager(limits=limits)

        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="CE",
        )
        state = RiskPortfolioState(portfolio_vega=60000.0)  # exceeds 50000

        results = rm.check([target], state)
        assert len(results) == 1
        assert results[0].blocked
        assert results[0].gate == GateResult.BLOCK_GREEKS
        assert "vega" in results[0].reason.lower()

    def test_negative_vega_breach_blocks_entry(self):
        """Negative vega exceeding limit (by absolute value) should also block."""
        limits = RiskLimits(max_portfolio_vega=50000.0)
        rm = RiskManager(limits=limits)

        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="short",
            weight=0.1,
            instrument_type="PE",
        )
        state = RiskPortfolioState(portfolio_vega=-55000.0)  # abs exceeds 50000

        results = rm.check([target], state)
        assert len(results) == 1
        assert results[0].blocked
        assert results[0].gate == GateResult.BLOCK_GREEKS

    def test_theta_breach_blocks_entry(self):
        """Portfolio theta more negative than limit should block new entries."""
        limits = RiskLimits(max_portfolio_theta=-25000.0)
        rm = RiskManager(limits=limits)

        target = TargetPosition(
            strategy_id="s1",
            symbol="BANKNIFTY",
            direction="long",
            weight=0.1,
            instrument_type="CE",
        )
        # theta=-30000 is more negative than limit -25000
        state = RiskPortfolioState(portfolio_theta=-30000.0)

        results = rm.check([target], state)
        assert len(results) == 1
        assert results[0].blocked
        assert results[0].gate == GateResult.BLOCK_GREEKS
        assert "theta" in results[0].reason.lower()

    def test_greeks_within_limits_pass(self):
        """When Greeks are within limits, targets should pass."""
        limits = RiskLimits(max_portfolio_vega=50000.0, max_portfolio_theta=-25000.0)
        rm = RiskManager(limits=limits)

        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="FUT",
        )
        state = RiskPortfolioState(
            portfolio_vega=20000.0,   # within limit
            portfolio_theta=-10000.0,  # within limit (less negative)
        )

        results = rm.check([target], state)
        assert len(results) == 1
        assert results[0].approved

    def test_flat_signal_bypasses_greek_check(self):
        """Flat signals (exits) should bypass all checks including Greeks."""
        limits = RiskLimits(max_portfolio_vega=50000.0)
        rm = RiskManager(limits=limits)

        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="flat",
            weight=0.0,
            instrument_type="FUT",
        )
        state = RiskPortfolioState(portfolio_vega=100000.0)  # way over limit

        results = rm.check([target], state)
        assert len(results) == 1
        assert results[0].approved  # flat always passes

    def test_block_greeks_enum_exists(self):
        """GateResult.BLOCK_GREEKS should exist."""
        assert hasattr(GateResult, "BLOCK_GREEKS")
        assert GateResult.BLOCK_GREEKS.value == "block_greeks"


# ---------------------------------------------------------------------------
# Bug #15: Reconciliation wired into Orchestrator
# ---------------------------------------------------------------------------
from quantlaxmi.engine.live.reconciliation import PositionReconciler, ReconciliationResult


class TestReconciliationWiring:
    """Bug #15: PositionReconciler is called by Orchestrator at end of day."""

    def _make_orchestrator(self, reconciler=None, tmp_path=None, with_strategy=True):
        """Create an Orchestrator with mocks, optionally with a reconciler."""
        from quantlaxmi.engine.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.sql.return_value = MagicMock(empty=True)

        mock_event_log = MagicMock()
        mock_event_log.event_count = 0
        mock_event_log.emit.return_value = MagicMock(seq=1)

        state_file = (tmp_path or Path("/tmp")) / "test_state.json"

        orch = Orchestrator(
            store=mock_store,
            event_log=mock_event_log,
            state_file=state_file,
            reconciler=reconciler,
        )

        # Register a no-op strategy so run_day doesn't bail early
        if with_strategy:
            mock_strategy = MagicMock()
            mock_strategy.strategy_id = "test_s1"
            mock_strategy.scan.return_value = []
            orch.registry = MagicMock()
            orch.registry.all.return_value = [mock_strategy]
            orch.allocator = MagicMock()
            orch.allocator.allocate.return_value = []

        return orch

    def test_orchestrator_accepts_reconciler(self, tmp_path):
        """Orchestrator should accept a reconciler parameter."""
        reconciler = MagicMock(spec=PositionReconciler)
        orch = self._make_orchestrator(reconciler=reconciler, tmp_path=tmp_path)
        assert orch._reconciler is reconciler

    def test_orchestrator_no_reconciler_by_default(self, tmp_path):
        """Orchestrator without reconciler should have _reconciler = None."""
        orch = self._make_orchestrator(tmp_path=tmp_path)
        assert orch._reconciler is None

    def test_reconciler_called_at_end_of_day(self, tmp_path):
        """When reconciler is wired, run_day() should call reconcile()."""
        reconciler = MagicMock(spec=PositionReconciler)
        reconciler.reconcile.return_value = ReconciliationResult(
            matched=[], is_clean=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        orch = self._make_orchestrator(reconciler=reconciler, tmp_path=tmp_path)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime_result = MagicMock()
            mock_regime_result.vix = 15.0
            mock_regime_result.regime.value = "normal"
            mock_regime.return_value = mock_regime_result

            orch.run_day(date(2026, 2, 10))

        reconciler.reconcile.assert_called_once()

    def test_reconciler_not_called_when_none(self, tmp_path):
        """When reconciler is None, run_day() should not crash."""
        orch = self._make_orchestrator(tmp_path=tmp_path)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime_result = MagicMock()
            mock_regime_result.vix = 15.0
            mock_regime_result.regime.value = "normal"
            mock_regime.return_value = mock_regime_result

            summary = orch.run_day(date(2026, 2, 10))

        assert summary["date"] == "2026-02-10"

    def test_reconciler_mismatch_logged(self, tmp_path, caplog):
        """When reconciliation finds mismatches, a warning should be logged."""
        reconciler = MagicMock(spec=PositionReconciler)
        reconciler.reconcile.return_value = ReconciliationResult(
            matched=[],
            mismatched=[{"symbol": "NIFTY", "internal_qty": 1, "broker_qty": 2}],
            is_clean=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        orch = self._make_orchestrator(reconciler=reconciler, tmp_path=tmp_path)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime_result = MagicMock()
            mock_regime_result.vix = 15.0
            mock_regime_result.regime.value = "normal"
            mock_regime.return_value = mock_regime_result

            with caplog.at_level(logging.WARNING):
                orch.run_day(date(2026, 2, 10))

        assert any("Reconciliation mismatch" in msg for msg in caplog.messages)

    def test_reconciler_exception_handled(self, tmp_path, caplog):
        """Reconciler exceptions should be caught and logged, not crash."""
        reconciler = MagicMock(spec=PositionReconciler)
        reconciler.reconcile.side_effect = RuntimeError("Broker API down")

        orch = self._make_orchestrator(reconciler=reconciler, tmp_path=tmp_path)

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime_result = MagicMock()
            mock_regime_result.vix = 15.0
            mock_regime_result.regime.value = "normal"
            mock_regime.return_value = mock_regime_result

            with caplog.at_level(logging.ERROR):
                summary = orch.run_day(date(2026, 2, 10))

        # Should not crash
        assert summary["date"] == "2026-02-10"
        assert any("Reconciliation failed" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Bug #24: DriftMonitor wired into TFT inference
# ---------------------------------------------------------------------------


class TestDriftMonitorWiring:
    """Bug #24: DriftMonitor is called during TFT inference predict()."""

    def _make_mock_metadata(self):
        """Create a minimal mock metadata for the inference pipeline."""
        meta = MagicMock()
        meta.feature_names = ["f1", "f2", "f3"]
        meta.asset_names = ["NIFTY", "BANKNIFTY"]
        meta.n_assets = 2
        meta.n_features = 3
        meta.version = 1
        meta.model_type = "x_trend"
        meta.config = {"seq_len": 5, "n_context": 2, "ctx_len": 5, "max_position": 0.25}
        meta.normalization = {"means": [0.0, 0.0, 0.0], "stds": [1.0, 1.0, 1.0]}
        return meta

    def test_inference_accepts_drift_monitor(self):
        """TFTInferencePipeline should accept a drift_monitor parameter."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline

        mock_model = MagicMock()
        meta = self._make_mock_metadata()
        drift_mon = MagicMock()

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
            drift_monitor=drift_mon,
        )
        assert pipe._drift_monitor is drift_mon

    def test_drift_monitor_none_by_default(self):
        """TFTInferencePipeline without drift_monitor should have None."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline

        mock_model = MagicMock()
        meta = self._make_mock_metadata()

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
        )
        assert pipe._drift_monitor is None

    def test_drift_monitor_called_during_predict(self):
        """When drift_monitor is wired, predict() should call check_drift()."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline
        from quantlaxmi.models.ml.tft.drift_monitor import DriftReport

        mock_model = MagicMock()
        meta = self._make_mock_metadata()

        drift_mon = MagicMock()
        drift_mon.check_drift.return_value = DriftReport(
            overall_status="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
            drift_monitor=drift_mon,
        )

        # Mock _build_features to return valid data
        features = np.random.randn(10, 2, 3)
        pipe._build_features = MagicMock(return_value=(features, None))

        # Mock _forward to return valid output
        pipe._forward = MagicMock(return_value=(
            {"NIFTY": 0.1, "BANKNIFTY": -0.05},
            {"NIFTY": 0.8, "BANKNIFTY": 0.6},
            {},
        ))
        pipe._get_feature_importance = MagicMock(return_value={})

        result = pipe.predict(date(2026, 2, 10), MagicMock())

        drift_mon.check_drift.assert_called_once()
        # Verify features were passed (flattened)
        call_args = drift_mon.check_drift.call_args
        flat_features = call_args[0][0]
        assert flat_features.shape == (20, 3)  # 10*2=20 rows, 3 features

    def test_drift_monitor_critical_logged(self, caplog):
        """Critical drift should be logged as a warning."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline
        from quantlaxmi.models.ml.tft.drift_monitor import DriftReport

        mock_model = MagicMock()
        meta = self._make_mock_metadata()

        drift_mon = MagicMock()
        drift_mon.check_drift.return_value = DriftReport(
            overall_status="critical",
            drifted_features=["f1", "f2", "f3", "f4"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
            drift_monitor=drift_mon,
        )

        features = np.random.randn(10, 2, 3)
        pipe._build_features = MagicMock(return_value=(features, None))
        pipe._forward = MagicMock(return_value=(
            {"NIFTY": 0.0, "BANKNIFTY": 0.0},
            {"NIFTY": 0.0, "BANKNIFTY": 0.0},
            {},
        ))
        pipe._get_feature_importance = MagicMock(return_value={})

        with caplog.at_level(logging.WARNING):
            pipe.predict(date(2026, 2, 10), MagicMock())

        assert any("drift detected" in msg.lower() for msg in caplog.messages)

    def test_drift_monitor_exception_handled(self, caplog):
        """Drift monitor exceptions should be caught, not crash inference."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline

        mock_model = MagicMock()
        meta = self._make_mock_metadata()

        drift_mon = MagicMock()
        drift_mon.check_drift.side_effect = RuntimeError("PSI calculation failed")

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
            drift_monitor=drift_mon,
        )

        features = np.random.randn(10, 2, 3)
        pipe._build_features = MagicMock(return_value=(features, None))
        pipe._forward = MagicMock(return_value=(
            {"NIFTY": 0.1, "BANKNIFTY": -0.05},
            {"NIFTY": 0.8, "BANKNIFTY": 0.6},
            {},
        ))
        pipe._get_feature_importance = MagicMock(return_value={})

        with caplog.at_level(logging.ERROR):
            result = pipe.predict(date(2026, 2, 10), MagicMock())

        # Should not crash
        assert result.positions["NIFTY"] == 0.1
        assert any("Drift monitor check failed" in msg for msg in caplog.messages)

    def test_drift_monitor_not_called_when_none(self):
        """When drift_monitor is None, predict should work without errors."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline

        mock_model = MagicMock()
        meta = self._make_mock_metadata()

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
        )

        features = np.random.randn(10, 2, 3)
        pipe._build_features = MagicMock(return_value=(features, None))
        pipe._forward = MagicMock(return_value=(
            {"NIFTY": 0.1, "BANKNIFTY": -0.05},
            {"NIFTY": 0.8, "BANKNIFTY": 0.6},
            {},
        ))
        pipe._get_feature_importance = MagicMock(return_value={})

        result = pipe.predict(date(2026, 2, 10), MagicMock())
        assert result.positions["NIFTY"] == 0.1

    def test_drift_monitor_not_called_when_no_features(self):
        """When features are None, drift monitor should be skipped."""
        from quantlaxmi.models.ml.tft.production.inference import TFTInferencePipeline

        mock_model = MagicMock()
        meta = self._make_mock_metadata()

        drift_mon = MagicMock()

        pipe = TFTInferencePipeline(
            model=mock_model,
            metadata=meta,
            norm_means=np.zeros(3),
            norm_stds=np.ones(3),
            drift_monitor=drift_mon,
        )

        pipe._build_features = MagicMock(return_value=(None, None))

        result = pipe.predict(date(2026, 2, 10), MagicMock())
        # Drift monitor should NOT be called when features are None
        drift_mon.check_drift.assert_not_called()
        assert result.metadata.get("error") == "no_features"


# ---------------------------------------------------------------------------
# Bug #16: Fill event emission after execution
# ---------------------------------------------------------------------------


class TestFillEventEmission:
    """Bug #16: FillPayload emitted after opening/closing positions."""

    def _make_orchestrator(self, tmp_path):
        """Create an Orchestrator with mocked store, event log, and exec journal."""
        from quantlaxmi.engine.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.sql.return_value = MagicMock(empty=True)

        mock_event_log = MagicMock()
        mock_event_log.event_count = 0
        mock_event_log.emit.return_value = MagicMock(seq=1)

        state_file = tmp_path / "test_state_fill.json"

        orch = Orchestrator(
            store=mock_store,
            event_log=mock_event_log,
            state_file=state_file,
        )
        # Replace exec journal with a mock to track log_fill calls
        orch._exec_journal = MagicMock()
        orch._exec_journal.log_fill.return_value = MagicMock(seq=1)
        orch._exec_journal.log_order.return_value = MagicMock(seq=1)
        return orch

    def test_fill_emitted_on_open(self, tmp_path):
        """Opening a position should emit a FillPayload."""
        from quantlaxmi.core.events.payloads import FillPayload

        orch = self._make_orchestrator(tmp_path)

        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="FUT",
        )

        action = orch._execute_target(target, 0.1, date(2026, 2, 10))
        assert action is not None
        assert action["action"] == "open"

        # Check that log_fill was called on the exec journal
        fill_calls = orch._exec_journal.log_fill.call_args_list
        assert len(fill_calls) >= 1

        last_fill_call = fill_calls[-1]
        payload = last_fill_call.kwargs.get("payload") or last_fill_call[1].get("payload")
        if payload is None:
            # positional args: log_fill(strategy_id, symbol, payload)
            payload = last_fill_call[0][2] if len(last_fill_call[0]) > 2 else last_fill_call[1]["payload"]
        assert isinstance(payload, FillPayload)
        assert payload.side == "buy"

    def test_fill_emitted_on_close(self, tmp_path):
        """Closing a position should emit a FillPayload."""
        from quantlaxmi.engine.state import Position
        from quantlaxmi.core.events.payloads import FillPayload

        orch = self._make_orchestrator(tmp_path)

        # First open a position
        pos = Position(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="FUT",
            entry_date="2026-02-09",
            entry_price=24000.0,
        )
        orch.state.open_position(pos)

        # Now close it
        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="flat",
            weight=0.0,
            instrument_type="FUT",
        )

        action = orch._execute_target(target, 0.0, date(2026, 2, 10))
        assert action is not None
        assert action["action"] == "close"

        # Check that log_fill was called
        fill_calls = orch._exec_journal.log_fill.call_args_list
        assert len(fill_calls) >= 1

        last_fill_call = fill_calls[-1]
        payload = last_fill_call.kwargs.get("payload") or last_fill_call[1].get("payload")
        if payload is None:
            payload = last_fill_call[0][2] if len(last_fill_call[0]) > 2 else last_fill_call[1]["payload"]
        assert isinstance(payload, FillPayload)
        assert payload.side == "sell"

    def test_no_fill_when_no_action(self, tmp_path):
        """When _execute_target returns None (already in position), no fill should be emitted."""
        from quantlaxmi.engine.state import Position

        orch = self._make_orchestrator(tmp_path)

        # Open a position first
        pos = Position(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="FUT",
            entry_date="2026-02-09",
            entry_price=24000.0,
        )
        orch.state.open_position(pos)

        # Reset mock call count
        orch._exec_journal.log_fill.reset_mock()

        # Try to open again (should skip — already in position)
        target = TargetPosition(
            strategy_id="s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="FUT",
        )

        action = orch._execute_target(target, 0.1, date(2026, 2, 10))
        assert action is None
        orch._exec_journal.log_fill.assert_not_called()

    def test_fill_on_open_has_correct_fields(self, tmp_path):
        """Fill payload for open should have buy side and non-empty IDs."""
        from quantlaxmi.core.events.payloads import FillPayload

        orch = self._make_orchestrator(tmp_path)

        target = TargetPosition(
            strategy_id="s1",
            symbol="BANKNIFTY",
            direction="short",
            weight=0.15,
            instrument_type="FUT",
        )

        orch._execute_target(target, 0.15, date(2026, 2, 10))

        fill_calls = orch._exec_journal.log_fill.call_args_list
        assert len(fill_calls) == 1

        call_kwargs = fill_calls[0].kwargs
        assert call_kwargs["strategy_id"] == "s1"
        assert call_kwargs["symbol"] == "BANKNIFTY"
        payload = call_kwargs["payload"]
        assert payload.order_id  # non-empty
        assert payload.fill_id   # non-empty
        assert payload.side == "buy"
        assert payload.is_partial is False

    def test_fill_events_in_run_day(self, tmp_path):
        """End-to-end: run_day should emit fill events for executed trades."""
        from quantlaxmi.engine.orchestrator import Orchestrator

        mock_store = MagicMock()
        mock_store.sql.return_value = MagicMock(empty=True)

        mock_event_log = MagicMock()
        mock_event_log.event_count = 0
        mock_event_log.emit.return_value = MagicMock(seq=1)

        state_file = tmp_path / "test_state_e2e.json"

        orch = Orchestrator(
            store=mock_store,
            event_log=mock_event_log,
            state_file=state_file,
        )

        # Mock exec_journal.log_fill so we can assert on call_count
        orch._exec_journal.log_fill = MagicMock()

        # Register a mock strategy that produces a signal
        mock_strategy = MagicMock()
        mock_strategy.strategy_id = "test_s1"
        from quantlaxmi.strategies.protocol import Signal
        mock_strategy.scan.return_value = [
            Signal(
                strategy_id="test_s1",
                symbol="NIFTY",
                direction="long",
                conviction=0.8,
                instrument_type="FUT",
            ),
        ]
        orch.registry = MagicMock()
        orch.registry.all.return_value = [mock_strategy]

        # Mock allocator to pass through
        mock_target = TargetPosition(
            strategy_id="test_s1",
            symbol="NIFTY",
            direction="long",
            weight=0.1,
            instrument_type="FUT",
        )
        orch.allocator = MagicMock()
        orch.allocator.allocate.return_value = [mock_target]

        with patch("quantlaxmi.engine.orchestrator.detect_regime") as mock_regime:
            mock_regime_result = MagicMock()
            mock_regime_result.vix = 15.0
            mock_regime_result.regime.value = "normal"
            mock_regime.return_value = mock_regime_result

            summary = orch.run_day(date(2026, 2, 10))

        # Should have at least one action
        assert len(summary["actions"]) >= 1
        assert summary["actions"][0]["action"] == "open"

        # Verify fill was emitted via the exec_journal
        assert orch._exec_journal.log_fill.call_count >= 1
