"""Tests for A/B Shadow Mode and Broker Position Reconciliation.

Item 9:  Shadow mode — run champion + challenger models in parallel.
Item 17: Position reconciliation — compare internal vs broker state.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlaxmi.engine.shadow_mode import (
    ShadowConfig,
    ShadowReport,
    ShadowRunner,
    _annualized_sharpe,
)
from quantlaxmi.engine.live.reconciliation import (
    PositionReconciler,
    ReconciliationResult,
)


# =====================================================================
# Helpers — lightweight mock models
# =====================================================================

class _ConstModel:
    """Model that always predicts a fixed value."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, features: np.ndarray) -> float:
        return self._value


class _ListModel:
    """Model that returns successive values from a list."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._idx = 0

    def predict(self, features: np.ndarray) -> float:
        val = self._values[self._idx % len(self._values)]
        self._idx += 1
        return val


class _NegModel:
    """Model that always returns the negative of another model's prediction."""

    def __init__(self, base: _ConstModel) -> None:
        self._base = base

    def predict(self, features: np.ndarray) -> float:
        return -self._base.predict(features)


# =====================================================================
# Shadow Mode Tests (Item 9)
# =====================================================================

class TestShadowStep:
    """Tests for ShadowRunner.step()."""

    def test_shadow_step_returns_both_signals(self):
        """Both champion and challenger produce signals in the result dict."""
        champion = _ConstModel(1.0)
        challenger = _ConstModel(0.5)
        runner = ShadowRunner(champion, challenger)

        features = np.array([1.0, 2.0, 3.0])
        result = runner.step(features, actual_return=0.01)

        assert "champion_signal" in result
        assert "challenger_signal" in result
        assert result["champion_signal"] == 1.0
        assert result["challenger_signal"] == 0.5

    def test_shadow_only_champion_used(self):
        """Verify challenger doesn't affect the champion signal value.

        The champion signal should be identical regardless of whether a
        challenger is present or what it predicts.
        """
        champion = _ConstModel(1.0)

        # Run without challenger
        runner_solo = ShadowRunner(champion, challenger_model=None)
        res_solo = runner_solo.step(np.array([1.0]), actual_return=0.01)

        # Run with challenger
        challenger = _ConstModel(-999.0)
        runner_shadow = ShadowRunner(champion, challenger)
        res_shadow = runner_shadow.step(np.array([1.0]), actual_return=0.01)

        # Champion signal is the same either way
        assert res_solo["champion_signal"] == res_shadow["champion_signal"]
        assert res_solo["champion_signal"] == 1.0

    def test_shadow_step_no_challenger(self):
        """When no challenger is set, challenger_signal is NaN."""
        champion = _ConstModel(1.0)
        runner = ShadowRunner(champion, challenger_model=None)

        result = runner.step(np.array([1.0]), actual_return=0.01)
        assert result["champion_signal"] == 1.0
        assert math.isnan(result["challenger_signal"])
        assert result["agreement"] is False

    def test_shadow_step_records_returns(self):
        """Returns are recorded correctly for both models."""
        champion = _ConstModel(1.0)
        challenger = _ConstModel(-1.0)
        runner = ShadowRunner(champion, challenger)

        runner.step(np.array([1.0]), actual_return=0.02)
        runner.step(np.array([1.0]), actual_return=-0.01)

        assert runner.champion_returns == [0.02, -0.01]
        assert runner.challenger_returns == [-0.02, 0.01]


class TestShadowEvaluate:
    """Tests for ShadowRunner.evaluate() and ShadowReport."""

    def test_shadow_evaluate_sharpe(self):
        """Correct Sharpe computation for both models."""
        # Champion always predicts +1, challenger always predicts +0.5
        champion = _ConstModel(1.0)
        challenger = _ConstModel(0.5)
        runner = ShadowRunner(champion, challenger)

        # 10 days of positive returns -> high Sharpe
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.005, size=10)
        for r in returns:
            runner.step(np.array([1.0]), actual_return=float(r))

        report = runner.evaluate()
        assert isinstance(report, ShadowReport)
        assert report.n_days == 10

        # Manually verify champion Sharpe
        champ_rets = np.array([1.0 * r for r in returns])
        expected_sharpe = (np.mean(champ_rets) / np.std(champ_rets, ddof=1)) * math.sqrt(252)
        assert abs(report.champion_sharpe - expected_sharpe) < 1e-8

        # Challenger Sharpe should be exactly half the champion returns
        # (same Sharpe since signal is just a scale factor)
        chall_rets = np.array([0.5 * r for r in returns])
        expected_chall = (np.mean(chall_rets) / np.std(chall_rets, ddof=1)) * math.sqrt(252)
        assert abs(report.challenger_sharpe - expected_chall) < 1e-8

    def test_shadow_evaluate_empty(self):
        """Evaluate with no data returns sensible defaults."""
        runner = ShadowRunner(_ConstModel(1.0), _ConstModel(0.5))
        report = runner.evaluate()
        assert report.n_days == 0
        assert report.recommendation == "insufficient_data"

    def test_shadow_agreement_pct(self):
        """Agreement percentage computed correctly."""
        # Champion: +1, +1, -1, -1, +1
        champion = _ListModel([1.0, 1.0, -1.0, -1.0, 1.0])
        # Challenger: +1, -1, -1, +1, +1
        challenger = _ListModel([1.0, -1.0, -1.0, 1.0, 1.0])
        runner = ShadowRunner(champion, challenger)

        for _ in range(5):
            runner.step(np.array([1.0]), actual_return=0.01)

        report = runner.evaluate()
        # Agreements: step0 (+/+), step2 (-/-), step4 (+/+) = 3/5 = 60%
        assert abs(report.agreement_pct - 0.6) < 1e-10

    def test_shadow_signal_correlation(self):
        """Signal correlation is computed correctly."""
        # Perfectly correlated signals (same direction, different scale)
        champion = _ListModel([1.0, 2.0, 3.0, 4.0, 5.0])
        challenger = _ListModel([2.0, 4.0, 6.0, 8.0, 10.0])
        runner = ShadowRunner(champion, challenger)

        for _ in range(5):
            runner.step(np.array([1.0]), actual_return=0.01)

        report = runner.evaluate()
        assert abs(report.signal_correlation - 1.0) < 1e-10


class TestShadowPromotion:
    """Tests for ShadowRunner.should_promote() and promote_challenger()."""

    def test_shadow_promote_when_better(self):
        """Challenger promoted when Sharpe improvement > threshold."""
        config = ShadowConfig(
            min_evaluation_days=5,
            min_sharpe_improvement=0.0,  # any improvement counts
        )

        # Challenger correctly predicts direction; champion gets it wrong often.
        # Use varying returns so std > 0 for both.
        #   returns: [+0.01, -0.01, +0.02, -0.005, +0.015, +0.01, +0.008]
        # Champion signals:  [-1, +1, -1, +1, -1, +1, -1]  → mostly wrong
        # Challenger signals: [+1, -1, +1, -1, +1, +1, +1] → mostly right
        returns = [0.01, -0.01, 0.02, -0.005, 0.015, 0.01, 0.008]
        champion = _ListModel([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        challenger = _ListModel([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])

        runner = ShadowRunner(champion, challenger, config)
        for r in returns:
            runner.step(np.array([1.0]), actual_return=r)

        report = runner.evaluate()
        # Challenger gains from correct directional bets; champion loses
        assert report.challenger_sharpe > report.champion_sharpe
        assert runner.should_promote() is True

    def test_shadow_no_promote_insufficient_data(self):
        """Rejected before min_evaluation_days."""
        champion = _ConstModel(0.1)
        challenger = _ConstModel(1.0)

        config = ShadowConfig(min_evaluation_days=10, min_sharpe_improvement=0.0)
        runner = ShadowRunner(champion, challenger, config)

        # Only 3 days — below threshold
        for _ in range(3):
            runner.step(np.array([1.0]), actual_return=0.01)

        assert runner.should_promote() is False
        report = runner.evaluate()
        assert report.recommendation == "insufficient_data"

    def test_shadow_no_promote_when_worse(self):
        """Challenger not promoted when it underperforms."""
        # Challenger predicts wrong direction
        champion = _ConstModel(1.0)
        challenger = _ConstModel(-1.0)

        config = ShadowConfig(min_evaluation_days=3, min_sharpe_improvement=0.1)
        runner = ShadowRunner(champion, challenger, config)

        # Positive returns: champion gains, challenger loses
        for _ in range(5):
            runner.step(np.array([1.0]), actual_return=0.01)

        assert runner.should_promote() is False
        report = runner.evaluate()
        assert report.recommendation == "keep_champion"

    def test_shadow_promote_clears_history(self):
        """After promotion, histories are cleared and challenger is None."""
        champion = _ConstModel(1.0)
        challenger = _ConstModel(2.0)
        runner = ShadowRunner(champion, challenger)

        for _ in range(5):
            runner.step(np.array([1.0]), actual_return=0.01)

        assert len(runner.champion_returns) == 5

        runner.promote_challenger()

        assert runner.champion is challenger
        assert runner.challenger is None
        assert len(runner.champion_returns) == 0
        assert len(runner.challenger_returns) == 0
        assert runner.start_date is None

    def test_shadow_no_promote_without_challenger(self):
        """should_promote() returns False when no challenger is set."""
        runner = ShadowRunner(_ConstModel(1.0), challenger_model=None)
        for _ in range(10):
            runner.step(np.array([1.0]), actual_return=0.01)
        assert runner.should_promote() is False


class TestShadowReport:
    """Tests for ShadowReport dataclass."""

    def test_report_to_dict(self):
        """ShadowReport.to_dict() produces a JSON-safe dictionary."""
        report = ShadowReport(
            champion_sharpe=1.5,
            challenger_sharpe=1.8,
            sharpe_improvement=0.3,
            signal_correlation=0.95,
            agreement_pct=0.85,
            n_days=20,
            champion_total_return=0.05,
            challenger_total_return=0.08,
            recommendation="promote_challenger",
        )
        d = report.to_dict()
        assert d["champion_sharpe"] == 1.5
        assert d["recommendation"] == "promote_challenger"
        assert isinstance(d["n_days"], int)


class TestAnnualizedSharpe:
    """Tests for the _annualized_sharpe helper."""

    def test_sharpe_constant_returns(self):
        """Constant positive returns → infinite-like Sharpe (limited by ddof=1)."""
        # With constant returns and ddof=1, std=0 → returns 0.0
        rets = np.array([0.01, 0.01, 0.01, 0.01])
        assert _annualized_sharpe(rets) == 0.0  # zero variance

    def test_sharpe_two_returns(self):
        """Sharpe with exactly 2 returns."""
        rets = np.array([0.01, 0.02])
        expected = (np.mean(rets) / np.std(rets, ddof=1)) * math.sqrt(252)
        assert abs(_annualized_sharpe(rets) - expected) < 1e-10

    def test_sharpe_single_return(self):
        """Fewer than 2 returns → 0.0."""
        assert _annualized_sharpe(np.array([0.01])) == 0.0

    def test_sharpe_empty(self):
        """Empty array → 0.0."""
        assert _annualized_sharpe(np.array([])) == 0.0


# =====================================================================
# Broker Position Reconciliation Tests (Item 17)
# =====================================================================

class TestReconcileMatching:
    """Tests for clean reconciliation scenarios."""

    def test_reconcile_matching_positions(self):
        """All positions match → is_clean=True."""
        internal = {
            "NIFTY26FEB20000CE": {"qty": 50, "avg_price": 150.0},
            "BANKNIFTY26FEB50000PE": {"qty": 25, "avg_price": 300.0},
        }
        broker = {
            "NIFTY26FEB20000CE": {"qty": 50, "avg_price": 150.0},
            "BANKNIFTY26FEB50000PE": {"qty": 25, "avg_price": 300.0},
        }
        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is True
        assert len(result.matched) == 2
        assert len(result.mismatched) == 0
        assert len(result.missing_internal) == 0
        assert len(result.missing_broker) == 0

    def test_reconcile_empty_both(self):
        """Both sides empty → is_clean=True."""
        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile({}, {})
        assert result.is_clean is True
        assert len(result.matched) == 0

    def test_reconcile_paper_mode(self):
        """No kite client and no broker positions → auto clean (paper mode)."""
        internal = {
            "NIFTY26FEB20000CE": {"qty": 50, "avg_price": 150.0},
        }
        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal)

        assert result.is_clean is True
        assert "NIFTY26FEB20000CE" in result.matched


class TestReconcileMismatches:
    """Tests for discrepancy detection."""

    def test_reconcile_qty_mismatch(self):
        """Different quantities detected as mismatch."""
        internal = {"NIFTY26FEB20000CE": {"qty": 50, "avg_price": 150.0}}
        broker = {"NIFTY26FEB20000CE": {"qty": 75, "avg_price": 150.0}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is False
        assert len(result.mismatched) == 1
        assert result.mismatched[0]["internal_qty"] == 50
        assert result.mismatched[0]["broker_qty"] == 75
        assert result.mismatched[0]["qty_diff"] == -25

    def test_reconcile_price_mismatch(self):
        """Large price difference detected as mismatch."""
        internal = {"SYM": {"qty": 10, "avg_price": 100.0}}
        broker = {"SYM": {"qty": 10, "avg_price": 105.0}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is False
        assert len(result.mismatched) == 1
        assert result.mismatched[0]["symbol"] == "SYM"

    def test_reconcile_missing_in_broker(self):
        """Internal has position, broker doesn't → missing_broker."""
        internal = {"PHANTOM": {"qty": 10, "avg_price": 100.0}}
        broker = {}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is False
        assert len(result.missing_broker) == 1
        assert result.missing_broker[0]["symbol"] == "PHANTOM"
        assert result.missing_broker[0]["qty"] == 10

    def test_reconcile_missing_internal(self):
        """Broker has position, internal doesn't → missing_internal."""
        internal = {}
        broker = {"SURPRISE": {"qty": 20, "avg_price": 200.0}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is False
        assert len(result.missing_internal) == 1
        assert result.missing_internal[0]["symbol"] == "SURPRISE"
        assert result.missing_internal[0]["qty"] == 20


class TestReconcileTolerance:
    """Tests for floating-point price tolerance."""

    def test_reconcile_price_tolerance(self):
        """Small FP differences within tolerance → matched."""
        # Price differs by 0.001% (well within 0.01% tolerance)
        internal = {"SYM": {"qty": 10, "avg_price": 100.0}}
        broker = {"SYM": {"qty": 10, "avg_price": 100.001}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is True
        assert "SYM" in result.matched

    def test_reconcile_price_beyond_tolerance(self):
        """Price difference > tolerance → mismatch."""
        internal = {"SYM": {"qty": 10, "avg_price": 100.0}}
        # 0.02% difference (beyond 0.01% tolerance)
        broker = {"SYM": {"qty": 10, "avg_price": 100.02}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is False
        assert len(result.mismatched) == 1

    def test_reconcile_custom_tolerance(self):
        """Custom tolerance can be configured."""
        internal = {"SYM": {"qty": 10, "avg_price": 100.0}}
        broker = {"SYM": {"qty": 10, "avg_price": 101.0}}

        # With 2% tolerance, 1% diff should match
        reconciler = PositionReconciler(kite_client=None, price_rel_tol=0.02)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is True
        assert "SYM" in result.matched

    def test_reconcile_near_zero_prices(self):
        """Near-zero prices use absolute tolerance."""
        internal = {"SYM": {"qty": 10, "avg_price": 0.0}}
        broker = {"SYM": {"qty": 10, "avg_price": 0.005}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        # abs_tol=0.01, so 0.005 diff is within tolerance
        assert result.is_clean is True


class TestReconcileEdgeCases:
    """Edge cases and integration tests."""

    def test_reconcile_case_insensitive_symbols(self):
        """Symbol comparison is case-insensitive."""
        internal = {"nifty26feb20000ce": {"qty": 50, "avg_price": 150.0}}
        broker = {"NIFTY26FEB20000CE": {"qty": 50, "avg_price": 150.0}}

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is True
        assert len(result.matched) == 1

    def test_reconcile_mixed_scenario(self):
        """Mix of matched, mismatched, and missing positions."""
        internal = {
            "MATCHED": {"qty": 10, "avg_price": 100.0},
            "MISMATCH": {"qty": 10, "avg_price": 100.0},
            "ONLY_INTERNAL": {"qty": 5, "avg_price": 50.0},
        }
        broker = {
            "MATCHED": {"qty": 10, "avg_price": 100.0},
            "MISMATCH": {"qty": 20, "avg_price": 100.0},
            "ONLY_BROKER": {"qty": 15, "avg_price": 75.0},
        }

        reconciler = PositionReconciler(kite_client=None)
        result = reconciler.reconcile(internal, broker)

        assert result.is_clean is False
        assert "MATCHED" in result.matched
        assert len(result.mismatched) == 1
        assert result.mismatched[0]["symbol"] == "MISMATCH"
        assert len(result.missing_broker) == 1
        assert result.missing_broker[0]["symbol"] == "ONLY_INTERNAL"
        assert len(result.missing_internal) == 1
        assert result.missing_internal[0]["symbol"] == "ONLY_BROKER"

    def test_reconcile_result_summary(self):
        """ReconciliationResult.summary() returns readable string."""
        result = ReconciliationResult(
            matched=["A", "B"],
            mismatched=[{"symbol": "C"}],
            missing_internal=[],
            missing_broker=[],
            is_clean=False,
            timestamp="2026-02-10T00:00:00",
        )
        s = result.summary()
        assert "matched=2" in s
        assert "mismatched=1" in s
        assert "is_clean=False" in s

    def test_reconcile_result_to_dict(self):
        """ReconciliationResult.to_dict() is JSON-safe."""
        result = ReconciliationResult(
            matched=["A"],
            mismatched=[],
            missing_internal=[],
            missing_broker=[],
            is_clean=True,
            timestamp="2026-02-10T00:00:00",
        )
        d = result.to_dict()
        assert d["is_clean"] is True
        assert d["matched"] == ["A"]

    def test_reconcile_with_mock_kite(self):
        """Integration test with a mock Kite client."""

        class MockKite:
            def positions(self):
                return {
                    "net": [
                        {"tradingsymbol": "NIFTY26FEB20000CE", "quantity": 50, "average_price": 150.0},
                        {"tradingsymbol": "BANKNIFTY26FEB50000PE", "quantity": 25, "average_price": 300.0},
                        # Zero-qty positions should be skipped
                        {"tradingsymbol": "CLOSED_SYM", "quantity": 0, "average_price": 0.0},
                    ]
                }

        internal = {
            "NIFTY26FEB20000CE": {"qty": 50, "avg_price": 150.0},
            "BANKNIFTY26FEB50000PE": {"qty": 25, "avg_price": 300.0},
        }

        reconciler = PositionReconciler(kite_client=MockKite())
        result = reconciler.reconcile(internal)

        assert result.is_clean is True
        assert len(result.matched) == 2
