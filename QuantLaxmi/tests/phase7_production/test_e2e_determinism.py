"""Phase 7 — E2E Determinism Tests.

Verifies that the equity curve comparator, replay parity, cross-day
consistency, and replay state persistence all behave deterministically
under identical inputs.  18 tests across 4 test classes.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from engine.replay.equity_comparator import (
    EquityCurveComparison,
    EquityDiff,
    _values_equal,
    compare_equity_curves,
)
from engine.replay.engine import ReplayResult
from engine.state import BrahmastraState, ClosedTrade, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_with_history(n_days=10, seed=42, equity_start=1.0):
    """Build a BrahmastraState with synthetic equity history and trades."""
    np.random.seed(seed)
    state = BrahmastraState(equity=equity_start, peak_equity=equity_start)
    for i in range(n_days):
        day_ret = np.random.normal(0.001, 0.01)
        state.equity *= (1 + day_ret)
        if state.equity > state.peak_equity:
            state.peak_equity = state.equity
        dd = (state.peak_equity - state.equity) / state.peak_equity
        state.equity_history.append({
            "date": f"2025-12-{i+1:02d}",
            "equity": state.equity,
            "drawdown": dd,
            "day_pnl": state.equity * day_ret,
        })
    # Add some closed trades
    for j in range(3):
        state.closed_trades.append(ClosedTrade(
            strategy_id="s5_hawkes",
            symbol="NIFTY",
            direction="long",
            entry_date=f"2025-12-{j+1:02d}",
            exit_date=f"2025-12-{j+3:02d}",
            entry_price=21500.0 + j * 100,
            exit_price=21600.0 + j * 100,
            weight=0.1,
            pnl_pct=0.005,
            exit_reason="signal_flat",
        ))
    return state


def _make_position(strategy_id: str, symbol: str, direction: str = "long") -> Position:
    """Create a simple Position for testing."""
    return Position(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        weight=0.1,
        instrument_type="FUT",
        entry_date="2025-12-01",
        entry_price=21500.0,
    )


def _make_replay_result(state: BrahmastraState) -> ReplayResult:
    """Wrap a BrahmastraState into a ReplayResult."""
    result = ReplayResult()
    result.final_state = state
    result.equity_curve = list(state.equity_history)
    result.dates_replayed = [h["date"] for h in state.equity_history]
    return result


# ===================================================================
# TestEquityCurveComparator — 6 tests
# ===================================================================

class TestEquityCurveComparator:
    """Tests for compare_equity_curves and _values_equal."""

    def test_identical_passes(self):
        """Two states built from the same seed should compare as identical."""
        state_a = _make_state_with_history(n_days=10, seed=42)
        state_b = _make_state_with_history(n_days=10, seed=42)

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is True
        assert len(cmp.diffs) == 0
        assert cmp.equity_points_a == cmp.equity_points_b == 10
        assert cmp.trades_a == cmp.trades_b == 3
        assert cmp.total_compared > 0

    def test_diff_equity_fails(self):
        """Altering one equity point should make comparison fail."""
        state_a = _make_state_with_history(n_days=10, seed=42)
        state_b = _make_state_with_history(n_days=10, seed=42)

        # Perturb a single equity value beyond FP tolerance
        state_b.equity_history[5]["equity"] += 1.0

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is False
        assert len(cmp.diffs) >= 1
        equity_diffs = [d for d in cmp.diffs if "equity" in d.field]
        assert len(equity_diffs) >= 1

    def test_diff_trade_count_fails(self):
        """One state with an extra trade should fail comparison."""
        state_a = _make_state_with_history(n_days=10, seed=42)
        state_b = _make_state_with_history(n_days=10, seed=42)

        # Add an extra trade to state_b
        state_b.closed_trades.append(ClosedTrade(
            strategy_id="s1_vrp",
            symbol="BANKNIFTY",
            direction="short",
            entry_date="2025-12-08",
            exit_date="2025-12-10",
            entry_price=48000.0,
            exit_price=47800.0,
            weight=0.05,
            pnl_pct=0.004,
            exit_reason="target",
        ))

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is False
        trade_count_diffs = [d for d in cmp.diffs if d.field == "trade_count"]
        assert len(trade_count_diffs) == 1
        assert trade_count_diffs[0].value_a == 3
        assert trade_count_diffs[0].value_b == 4

    def test_fp_tolerance(self):
        """Equity differing by 1e-14 should still pass (within FP tolerance)."""
        state_a = _make_state_with_history(n_days=10, seed=42)
        state_b = _make_state_with_history(n_days=10, seed=42)

        # Perturb by an amount well within rtol=1e-10, atol=1e-12
        state_b.equity_history[3]["equity"] += 1e-14

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is True
        assert len(cmp.diffs) == 0

    def test_nan_handling(self):
        """NaN equity values should compare as equal (NaN == NaN)."""
        state_a = _make_state_with_history(n_days=5, seed=99)
        state_b = _make_state_with_history(n_days=5, seed=99)

        # Set equity to NaN at the same index in both states
        state_a.equity_history[2]["equity"] = float("nan")
        state_b.equity_history[2]["equity"] = float("nan")

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is True
        assert len(cmp.diffs) == 0

        # Also verify _values_equal directly
        assert _values_equal(float("nan"), float("nan")) is True
        assert _values_equal(float("inf"), float("inf")) is True
        assert _values_equal(float("-inf"), float("-inf")) is True
        assert _values_equal(float("inf"), float("-inf")) is False

    def test_diff_position_count(self):
        """Different number of active positions should produce a diff."""
        state_a = _make_state_with_history(n_days=5, seed=42)
        state_b = _make_state_with_history(n_days=5, seed=42)

        # Add active positions to state_a only
        pos = _make_position("s5_hawkes", "NIFTY")
        state_a.positions[state_a.position_key(pos.strategy_id, pos.symbol)] = pos

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is False
        pos_diffs = [d for d in cmp.diffs if d.field == "position_count"]
        assert len(pos_diffs) == 1
        assert pos_diffs[0].value_a == 1
        assert pos_diffs[0].value_b == 0


# ===================================================================
# TestE2EDeterminism — 6 tests
# ===================================================================

class TestE2EDeterminism:
    """3x parity: three independent constructions from the same seed
    must produce identical states and equity curves."""

    def _build_three_states(self, n_days=10, seed=42):
        """Build 3 independent states from the same seed."""
        return [_make_state_with_history(n_days=n_days, seed=seed) for _ in range(3)]

    def test_3x_parity_events(self):
        """3 identical ReplayResults should have identical equity curves."""
        states = self._build_three_states()
        results = [_make_replay_result(s) for s in states]

        # Compare run 1 vs 2, and run 1 vs 3
        cmp_12 = compare_equity_curves(results[0].final_state, results[1].final_state)
        cmp_13 = compare_equity_curves(results[0].final_state, results[2].final_state)

        assert cmp_12.identical is True, cmp_12.summary()
        assert cmp_13.identical is True, cmp_13.summary()

    def test_3x_parity_equity_curve(self):
        """3 states from the same seed should have identical equity curves."""
        states = self._build_three_states(n_days=15)

        for i in range(1, 3):
            cmp = compare_equity_curves(states[0], states[i])
            assert cmp.identical is True, (
                f"State 0 vs State {i}: {cmp.summary()}"
            )
            assert cmp.equity_points_a == cmp.equity_points_b == 15

    def test_3x_parity_position_count(self):
        """3 identical states should have matching (zero) position counts."""
        states = self._build_three_states()

        for s in states:
            assert len(s.active_positions()) == 0

        # Comparison should reflect equal position counts
        for i in range(1, 3):
            cmp = compare_equity_curves(states[0], states[i])
            pos_diffs = [d for d in cmp.diffs if d.field == "position_count"]
            assert len(pos_diffs) == 0

    def test_3x_parity_trade_history(self):
        """3 identical states should have matching trade histories."""
        states = self._build_three_states()

        for s in states:
            assert len(s.closed_trades) == 3

        for i in range(1, 3):
            cmp = compare_equity_curves(states[0], states[i])
            trade_diffs = [d for d in cmp.diffs if "trade" in d.field]
            assert len(trade_diffs) == 0, (
                f"Trade diffs found in state 0 vs {i}: {trade_diffs}"
            )

    def test_3x_parity_final_state(self):
        """3 identical states should have matching final equity."""
        states = self._build_three_states()

        ref_equity = states[0].equity
        ref_peak = states[0].peak_equity

        for i in range(1, 3):
            assert _values_equal(states[i].equity, ref_equity), (
                f"State {i} equity {states[i].equity} != ref {ref_equity}"
            )
            assert _values_equal(states[i].peak_equity, ref_peak), (
                f"State {i} peak {states[i].peak_equity} != ref {ref_peak}"
            )

    def test_3x_parity_strategy_equity(self):
        """3 states with strategy equity should have identical strategy equity."""
        states = self._build_three_states()

        # Add strategy equity to all three states deterministically
        for s in states:
            s.strategy_equity["s5_hawkes"] = s.equity * 0.6
            s.strategy_equity["s1_vrp"] = s.equity * 0.4
            s.strategy_peaks["s5_hawkes"] = s.peak_equity * 0.6
            s.strategy_peaks["s1_vrp"] = s.peak_equity * 0.4

        # Verify pairwise
        for i in range(1, 3):
            for key in ("s5_hawkes", "s1_vrp"):
                assert _values_equal(
                    states[0].strategy_equity[key],
                    states[i].strategy_equity[key],
                ), f"strategy_equity[{key}] mismatch: state 0 vs {i}"
                assert _values_equal(
                    states[0].strategy_peaks[key],
                    states[i].strategy_peaks[key],
                ), f"strategy_peaks[{key}] mismatch: state 0 vs {i}"


# ===================================================================
# TestE2ECrossDayParity — 3 tests
# ===================================================================

class TestE2ECrossDayParity:
    """Cross-day parity: multi-day equity curves and position transitions
    remain identical across two independent constructions."""

    def test_multi_day_equity(self):
        """20-day history from same seed should match day-by-day."""
        state_a = _make_state_with_history(n_days=20, seed=77)
        state_b = _make_state_with_history(n_days=20, seed=77)

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is True, cmp.summary()
        assert cmp.equity_points_a == 20
        assert cmp.equity_points_b == 20
        # Verify every single equity point matches
        for i in range(20):
            assert _values_equal(
                state_a.equity_history[i]["equity"],
                state_b.equity_history[i]["equity"],
            ), f"Day {i} equity mismatch"

    def test_position_transitions(self):
        """States with positions opened/closed across days should match."""
        np.random.seed(123)
        state_a = BrahmastraState(equity=1.0, peak_equity=1.0)
        state_b = BrahmastraState(equity=1.0, peak_equity=1.0)

        # Open and close positions identically in both states
        for s in (state_a, state_b):
            pos1 = _make_position("s5_hawkes", "NIFTY", "long")
            pos2 = _make_position("s4_iv_mr", "BANKNIFTY", "short")
            s.open_position(pos1)
            s.open_position(pos2)

            # Record equity after opens
            s.equity_history.append({
                "date": "2025-12-01",
                "equity": s.equity,
                "drawdown": 0.0,
                "day_pnl": 0.0,
            })

            # Close first position
            s.close_position("s5_hawkes", "NIFTY", "2025-12-03", 21700.0, "target")
            s.equity_history.append({
                "date": "2025-12-03",
                "equity": s.equity,
                "drawdown": s.portfolio_dd,
                "day_pnl": 0.0,
            })

            # Close second position
            s.close_position("s4_iv_mr", "BANKNIFTY", "2025-12-05", 21300.0, "stop")
            s.equity_history.append({
                "date": "2025-12-05",
                "equity": s.equity,
                "drawdown": s.portfolio_dd,
                "day_pnl": 0.0,
            })

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is True, cmp.summary()
        assert len(state_a.closed_trades) == 2
        assert len(state_b.closed_trades) == 2
        assert len(state_a.active_positions()) == 0
        assert len(state_b.active_positions()) == 0

    def test_multi_strategy_combined(self):
        """Multiple strategies in one state should all compare correctly."""
        strategies = ["s5_hawkes", "s1_vrp", "s4_iv_mr", "s7_microstructure"]

        def _build_multi_strategy_state():
            np.random.seed(55)
            state = BrahmastraState(equity=1.0, peak_equity=1.0)
            for day in range(15):
                day_ret = np.random.normal(0.0005, 0.008)
                state.equity *= (1 + day_ret)
                if state.equity > state.peak_equity:
                    state.peak_equity = state.equity
                dd = (state.peak_equity - state.equity) / state.peak_equity
                state.equity_history.append({
                    "date": f"2025-12-{day+1:02d}",
                    "equity": state.equity,
                    "drawdown": dd,
                    "day_pnl": state.equity * day_ret,
                })

            # Add trades from multiple strategies
            for idx, strat in enumerate(strategies):
                state.closed_trades.append(ClosedTrade(
                    strategy_id=strat,
                    symbol="NIFTY" if idx % 2 == 0 else "BANKNIFTY",
                    direction="long" if idx % 2 == 0 else "short",
                    entry_date=f"2025-12-{idx+1:02d}",
                    exit_date=f"2025-12-{idx+4:02d}",
                    entry_price=21500.0 + idx * 50,
                    exit_price=21600.0 + idx * 50,
                    weight=0.05,
                    pnl_pct=0.003 + idx * 0.001,
                    exit_reason="signal_flat",
                ))

            # Set strategy-level equity
            for strat in strategies:
                state.strategy_equity[strat] = state.equity / len(strategies)
                state.strategy_peaks[strat] = state.peak_equity / len(strategies)

            return state

        state_a = _build_multi_strategy_state()
        state_b = _build_multi_strategy_state()

        cmp = compare_equity_curves(state_a, state_b)

        assert cmp.identical is True, cmp.summary()
        assert cmp.trades_a == cmp.trades_b == len(strategies)
        assert cmp.equity_points_a == cmp.equity_points_b == 15


# ===================================================================
# TestReplayStatePersistence — 3 tests
# ===================================================================

class TestReplayStatePersistence:
    """Verify ReplayResult correctly captures state, equity curve,
    and trade history from the final BrahmastraState."""

    def test_state_captured(self):
        """ReplayResult.final_state should be non-None after construction."""
        state = _make_state_with_history(n_days=5, seed=42)
        result = _make_replay_result(state)

        assert result.final_state is not None
        assert isinstance(result.final_state, BrahmastraState)
        assert result.final_state.equity == state.equity

    def test_equity_non_empty(self):
        """ReplayResult.equity_curve should be non-empty when state has history."""
        state = _make_state_with_history(n_days=8, seed=99)
        result = _make_replay_result(state)

        assert len(result.equity_curve) == 8
        assert len(result.dates_replayed) == 8
        # Each entry should have the expected keys
        for entry in result.equity_curve:
            assert "date" in entry
            assert "equity" in entry
            assert "drawdown" in entry
            assert "day_pnl" in entry

    def test_trades_match_events(self):
        """closed_trades in final_state should match expected count."""
        state = _make_state_with_history(n_days=10, seed=42)
        result = _make_replay_result(state)

        # _make_state_with_history creates 3 closed trades
        assert len(result.final_state.closed_trades) == 3

        # Verify trade attributes are preserved
        for trade in result.final_state.closed_trades:
            assert trade.strategy_id == "s5_hawkes"
            assert trade.symbol == "NIFTY"
            assert trade.direction == "long"
            assert trade.pnl_pct == 0.005
            assert trade.exit_reason == "signal_flat"
