"""Exhaustive risk gate enforcement tests.

Tests every gate in: pass, fail, boundary, interaction scenarios.
BLOCKER #1 — no live trading without 100% gate coverage.

Covers:
  - VPIN gate (hard block > 0.70)
  - Portfolio DD circuit breaker (> 5%)
  - Strategy DD circuit breaker (> 3%)
  - Concentration limits (single instrument, stock FnO, total exposure)
  - Gate ordering: VPIN > DD > concentration
  - Flat signals always pass
  - Circuit breaker activation/reset
  - Size reduction (REDUCE_SIZE)
  - Boundary values (at threshold, just above, just below)
"""

from __future__ import annotations

import pytest

from core.allocator.meta import TargetPosition
from core.risk.limits import RiskLimits
from core.risk.manager import (
    GateResult,
    PortfolioState,
    RiskCheckResult,
    RiskManager,
)


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


def _state(**kwargs) -> PortfolioState:
    return PortfolioState(**kwargs)


def _rm(limits: RiskLimits | None = None) -> RiskManager:
    return RiskManager(limits=limits or RiskLimits())


# ===================================================================
# 1. FLAT SIGNALS — must always pass (they reduce risk)
# ===================================================================

class TestFlatSignals:
    def test_flat_always_approved(self):
        rm = _rm()
        state = _state(vpin=0.99, equity=0.5, peak_equity=1.0)  # everything bad
        result = rm.check([_target(direction="flat", weight=0.0)], state)
        assert result[0].approved is True
        assert result[0].gate == GateResult.PASS

    def test_flat_weight_is_zero(self):
        rm = _rm()
        result = rm.check([_target(direction="flat", weight=0.0)], _state())
        assert result[0].adjusted_weight == 0.0

    def test_flat_passes_even_when_circuit_breaker_active(self):
        rm = _rm()
        rm._circuit_breaker_active = True
        result = rm.check([_target(direction="flat")], _state(vpin=0.95))
        assert result[0].approved is True

    def test_multiple_flats_all_pass(self):
        rm = _rm()
        targets = [_target(direction="flat", symbol=s) for s in ["NIFTY", "BANKNIFTY", "RELIANCE"]]
        results = rm.check(targets, _state(vpin=0.99))
        assert all(r.approved for r in results)


# ===================================================================
# 2. VPIN GATE — Layer 1 (highest priority)
# ===================================================================

class TestVPINGate:
    def test_vpin_below_threshold_passes(self):
        rm = _rm()
        result = rm.check([_target()], _state(vpin=0.50))
        assert result[0].approved is True

    def test_vpin_at_threshold_passes(self):
        """VPIN == 0.70 should pass (> is the condition, not >=)."""
        rm = _rm()
        result = rm.check([_target()], _state(vpin=0.70))
        assert result[0].approved is True

    def test_vpin_above_threshold_blocks(self):
        rm = _rm()
        result = rm.check([_target()], _state(vpin=0.71))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_VPIN

    def test_vpin_just_above_blocks(self):
        rm = _rm()
        result = rm.check([_target()], _state(vpin=0.70001))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_VPIN

    def test_vpin_extreme_blocks(self):
        rm = _rm()
        result = rm.check([_target()], _state(vpin=1.0))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_VPIN

    def test_vpin_zero_passes(self):
        rm = _rm()
        result = rm.check([_target()], _state(vpin=0.0))
        assert result[0].approved is True

    def test_vpin_block_sets_circuit_breaker(self):
        rm = _rm()
        assert rm.circuit_breaker_active is False
        rm.check([_target()], _state(vpin=0.80))
        assert rm.circuit_breaker_active is True

    def test_vpin_block_adjusted_weight_is_zero(self):
        rm = _rm()
        result = rm.check([_target(weight=0.15)], _state(vpin=0.80))
        assert result[0].adjusted_weight == 0.0

    def test_vpin_block_has_reason(self):
        rm = _rm()
        result = rm.check([_target()], _state(vpin=0.85))
        assert "VPIN" in result[0].reason

    def test_vpin_custom_threshold(self):
        rm = _rm(RiskLimits(vpin_block_threshold=0.50))
        result = rm.check([_target()], _state(vpin=0.55))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_VPIN

    def test_vpin_blocks_multiple_targets(self):
        rm = _rm()
        targets = [_target(symbol="NIFTY"), _target(symbol="BANKNIFTY")]
        results = rm.check(targets, _state(vpin=0.80))
        assert all(r.gate == GateResult.BLOCK_VPIN for r in results)

    def test_vpin_blocks_strong_conviction_signal(self):
        """Even conviction=1.0 signals are blocked by VPIN."""
        rm = _rm()
        t = _target(weight=0.20)
        result = rm.check([t], _state(vpin=0.80))
        assert result[0].approved is False


# ===================================================================
# 3. PORTFOLIO DD CIRCUIT BREAKER — Layer 2
# ===================================================================

class TestPortfolioDDBreaker:
    def test_no_dd_passes(self):
        rm = _rm()
        result = rm.check([_target()], _state(equity=1.0, peak_equity=1.0))
        assert result[0].approved is True

    def test_small_dd_passes(self):
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.97, peak_equity=1.0))
        assert result[0].approved is True

    def test_dd_at_threshold_passes(self):
        """DD just under 5% should pass (> is the condition, not >=).
        Note: equity=0.95 yields DD=0.0500...04 in IEEE 754, so use 0.9501."""
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.9501, peak_equity=1.0))
        assert result[0].approved is True

    def test_dd_exactly_at_fp_boundary_blocks(self):
        """equity=0.95 → DD=0.0500...04 (FP), which is > 0.05 → blocks."""
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.95, peak_equity=1.0))
        assert result[0].approved is False

    def test_dd_above_threshold_blocks(self):
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.949, peak_equity=1.0))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_DD_PORTFOLIO

    def test_dd_way_above_blocks(self):
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.80, peak_equity=1.0))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_DD_PORTFOLIO

    def test_dd_activates_circuit_breaker(self):
        rm = _rm()
        rm.check([_target()], _state(equity=0.90, peak_equity=1.0))
        assert rm.circuit_breaker_active is True

    def test_dd_block_reason_contains_percentage(self):
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.90, peak_equity=1.0))
        assert "Portfolio DD" in result[0].reason

    def test_dd_custom_threshold(self):
        rm = _rm(RiskLimits(max_portfolio_dd=0.02))
        result = rm.check([_target()], _state(equity=0.975, peak_equity=1.0))
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_DD_PORTFOLIO

    def test_dd_zero_peak_equity_no_crash(self):
        """Edge: peak_equity=0 should not crash."""
        rm = _rm()
        result = rm.check([_target()], _state(equity=0.0, peak_equity=0.0))
        assert result[0].approved is True  # 0/0 → dd=0

    def test_dd_negative_equity_blocks(self):
        rm = _rm()
        result = rm.check([_target()], _state(equity=-0.1, peak_equity=1.0))
        assert result[0].approved is False


# ===================================================================
# 4. STRATEGY DD CIRCUIT BREAKER — Layer 2
# ===================================================================

class TestStrategyDDBreaker:
    def test_strategy_no_dd_passes(self):
        rm = _rm()
        state = _state(strategy_equity={"s1_vrp": 1.0}, strategy_peaks={"s1_vrp": 1.0})
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].approved is True

    def test_strategy_small_dd_passes(self):
        rm = _rm()
        state = _state(strategy_equity={"s1_vrp": 0.98}, strategy_peaks={"s1_vrp": 1.0})
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].approved is True

    def test_strategy_dd_at_threshold_passes(self):
        """DD just under 3% passes. Use 0.9701 to avoid FP overshoot."""
        rm = _rm()
        state = _state(strategy_equity={"s1_vrp": 0.9701}, strategy_peaks={"s1_vrp": 1.0})
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].approved is True

    def test_strategy_dd_exactly_at_fp_boundary_blocks(self):
        """equity=0.97 → DD=0.0300...04 (FP), which is > 0.03 → blocks."""
        rm = _rm()
        state = _state(strategy_equity={"s1_vrp": 0.97}, strategy_peaks={"s1_vrp": 1.0})
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].approved is False

    def test_strategy_dd_above_threshold_blocks(self):
        rm = _rm()
        state = _state(strategy_equity={"s1_vrp": 0.969}, strategy_peaks={"s1_vrp": 1.0})
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_DD_STRATEGY

    def test_strategy_dd_blocks_only_affected_strategy(self):
        rm = _rm()
        state = _state(
            strategy_equity={"s1_vrp": 0.95, "s4_iv_mr": 1.0},
            strategy_peaks={"s1_vrp": 1.0, "s4_iv_mr": 1.0},
        )
        r1 = rm.check([_target(strategy_id="s1_vrp")], state)
        r2 = rm.check([_target(strategy_id="s4_iv_mr")], state)
        assert r1[0].approved is False
        assert r2[0].approved is True

    def test_unknown_strategy_defaults_to_no_dd(self):
        rm = _rm()
        result = rm.check([_target(strategy_id="unknown_strat")], _state())
        assert result[0].approved is True

    def test_strategy_dd_reason_contains_strategy_id(self):
        rm = _rm()
        state = _state(strategy_equity={"s5_hawkes": 0.90}, strategy_peaks={"s5_hawkes": 1.0})
        result = rm.check([_target(strategy_id="s5_hawkes")], state)
        assert "s5_hawkes" in result[0].reason

    def test_strategy_dd_custom_threshold(self):
        rm = _rm(RiskLimits(max_strategy_dd=0.01))
        state = _state(strategy_equity={"s1_vrp": 0.985}, strategy_peaks={"s1_vrp": 1.0})
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].approved is False


# ===================================================================
# 5. CONCENTRATION LIMITS — Layer 3
# ===================================================================

class TestSingleInstrumentLimit:
    def test_small_position_passes(self):
        rm = _rm()
        result = rm.check([_target(weight=0.05)], _state())
        assert result[0].approved is True
        assert result[0].gate == GateResult.PASS

    def test_at_limit_passes(self):
        rm = _rm()
        result = rm.check([_target(weight=0.20)], _state())
        assert result[0].approved is True

    def test_above_limit_with_no_existing_reduces(self):
        rm = _rm()
        result = rm.check([_target(weight=0.25)], _state())
        assert result[0].approved is True
        assert result[0].gate == GateResult.REDUCE_SIZE
        assert result[0].adjusted_weight <= 0.20

    def test_existing_position_plus_new_exceeds_limit(self):
        rm = _rm()
        state = _state(positions={"NIFTY": {"weight": 0.15, "direction": "long"}})
        result = rm.check([_target(symbol="NIFTY", weight=0.10)], state)
        assert result[0].approved is True
        assert result[0].adjusted_weight <= 0.05 + 1e-6  # max 0.20 - 0.15 = 0.05

    def test_already_at_limit_blocks(self):
        rm = _rm()
        state = _state(positions={"NIFTY": {"weight": 0.20, "direction": "long"}})
        result = rm.check([_target(symbol="NIFTY", weight=0.05)], state)
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_CONCENTRATION

    def test_different_symbol_not_affected(self):
        rm = _rm()
        state = _state(positions={"NIFTY": {"weight": 0.20, "direction": "long"}})
        result = rm.check([_target(symbol="BANKNIFTY", weight=0.10)], state)
        assert result[0].approved is True

    def test_custom_single_instrument_limit(self):
        rm = _rm(RiskLimits(max_single_instrument=0.10))
        result = rm.check([_target(weight=0.15)], _state())
        assert result[0].approved is True
        assert result[0].gate == GateResult.REDUCE_SIZE
        assert result[0].adjusted_weight <= 0.10


class TestStockFnOLimit:
    def test_index_name_not_subject_to_stock_fno_limit(self):
        rm = _rm()
        result = rm.check([_target(symbol="NIFTY", weight=0.10)], _state())
        assert result[0].approved is True

    def test_banknifty_is_index(self):
        rm = _rm()
        result = rm.check([_target(symbol="BANKNIFTY", weight=0.10)], _state())
        assert result[0].approved is True

    def test_stock_fno_within_limit_passes(self):
        rm = _rm()
        result = rm.check([_target(symbol="RELIANCE", weight=0.04)], _state())
        assert result[0].approved is True

    def test_stock_fno_above_limit_reduces(self):
        rm = _rm()
        result = rm.check([_target(symbol="RELIANCE", weight=0.08)], _state())
        assert result[0].approved is True
        assert result[0].adjusted_weight <= 0.05 + 1e-6

    def test_stock_fno_at_limit_blocks(self):
        rm = _rm()
        state = _state(positions={"RELIANCE": {"weight": 0.05, "direction": "long"}})
        result = rm.check([_target(symbol="RELIANCE", weight=0.03)], state)
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_CONCENTRATION

    def test_all_index_names_pass_stock_fno_check(self):
        rm = _rm()
        for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"]:
            result = rm.check([_target(symbol=sym, weight=0.10)], _state())
            assert result[0].approved is True, f"{sym} should be treated as index"

    def test_stock_name_case_insensitive(self):
        """Symbol comparison should be case-insensitive."""
        rm = _rm()
        # Lowercase stock name should still be subject to stock FnO limits
        result = rm.check([_target(symbol="reliance", weight=0.08)], _state())
        assert result[0].adjusted_weight <= 0.05 + 1e-6


class TestTotalExposureLimit:
    def test_below_total_exposure_passes(self):
        rm = _rm()
        result = rm.check([_target(weight=0.10)], _state())
        assert result[0].approved is True

    def test_high_existing_plus_new_exceeds_limit(self):
        rm = _rm()
        state = _state(positions={
            "NIFTY": {"weight": 0.50},
            "BANKNIFTY": {"weight": 0.50},
            "FINNIFTY": {"weight": 0.49},
        })
        result = rm.check([_target(symbol="MIDCPNIFTY", weight=0.05)], state)
        assert result[0].approved is True
        assert result[0].adjusted_weight <= 0.01 + 1e-6  # 1.50 - 1.49 = 0.01

    def test_at_exposure_limit_blocks(self):
        rm = _rm()
        state = _state(positions={
            "NIFTY": {"weight": 0.50},
            "BANKNIFTY": {"weight": 0.50},
            "FINNIFTY": {"weight": 0.50},
        })
        result = rm.check([_target(symbol="MIDCPNIFTY", weight=0.05)], state)
        assert result[0].approved is False
        assert result[0].gate == GateResult.BLOCK_EXPOSURE

    def test_custom_exposure_limit(self):
        rm = _rm(RiskLimits(max_total_exposure=0.50))
        state = _state(positions={"NIFTY": {"weight": 0.45}})
        result = rm.check([_target(symbol="BANKNIFTY", weight=0.10)], state)
        assert result[0].approved is True
        assert result[0].adjusted_weight <= 0.05 + 1e-6


# ===================================================================
# 6. GATE ORDERING: VPIN > DD > Concentration
# ===================================================================

class TestGateOrdering:
    def test_vpin_fires_before_dd(self):
        """When both VPIN and DD are breached, VPIN should block first."""
        rm = _rm()
        state = _state(vpin=0.80, equity=0.90, peak_equity=1.0)
        result = rm.check([_target()], state)
        assert result[0].gate == GateResult.BLOCK_VPIN

    def test_portfolio_dd_fires_before_strategy_dd(self):
        """When both portfolio and strategy DD breached, portfolio fires first."""
        rm = _rm()
        state = _state(
            equity=0.90, peak_equity=1.0,
            strategy_equity={"s1_vrp": 0.90}, strategy_peaks={"s1_vrp": 1.0},
        )
        result = rm.check([_target(strategy_id="s1_vrp")], state)
        assert result[0].gate == GateResult.BLOCK_DD_PORTFOLIO

    def test_strategy_dd_fires_before_concentration(self):
        rm = _rm()
        state = _state(
            strategy_equity={"s1_vrp": 0.95}, strategy_peaks={"s1_vrp": 1.0},
            positions={"NIFTY": {"weight": 0.20}},
        )
        result = rm.check([_target(strategy_id="s1_vrp", symbol="NIFTY", weight=0.05)], state)
        assert result[0].gate == GateResult.BLOCK_DD_STRATEGY

    def test_vpin_fires_before_concentration(self):
        rm = _rm()
        state = _state(
            vpin=0.80,
            positions={"NIFTY": {"weight": 0.20}},
        )
        result = rm.check([_target(symbol="NIFTY", weight=0.05)], state)
        assert result[0].gate == GateResult.BLOCK_VPIN


# ===================================================================
# 7. CIRCUIT BREAKER STATE
# ===================================================================

class TestCircuitBreaker:
    def test_initially_inactive(self):
        rm = _rm()
        assert rm.circuit_breaker_active is False

    def test_activated_by_vpin(self):
        rm = _rm()
        rm.check([_target()], _state(vpin=0.80))
        assert rm.circuit_breaker_active is True

    def test_activated_by_portfolio_dd(self):
        rm = _rm()
        rm.check([_target()], _state(equity=0.90, peak_equity=1.0))
        assert rm.circuit_breaker_active is True

    def test_reset(self):
        rm = _rm()
        rm.check([_target()], _state(vpin=0.80))
        assert rm.circuit_breaker_active is True
        rm.reset_circuit_breaker()
        assert rm.circuit_breaker_active is False

    def test_not_activated_by_strategy_dd(self):
        """Strategy DD should not activate the portfolio-level breaker."""
        rm = _rm()
        state = _state(strategy_equity={"s1_vrp": 0.95}, strategy_peaks={"s1_vrp": 1.0})
        rm.check([_target(strategy_id="s1_vrp")], state)
        assert rm.circuit_breaker_active is False

    def test_not_activated_by_concentration(self):
        rm = _rm()
        state = _state(positions={"NIFTY": {"weight": 0.20}})
        rm.check([_target(symbol="NIFTY", weight=0.05)], state)
        assert rm.circuit_breaker_active is False


# ===================================================================
# 8. PORTFOLIO STATE CALCULATIONS
# ===================================================================

class TestPortfolioState:
    def test_portfolio_dd_normal(self):
        s = _state(equity=0.95, peak_equity=1.0)
        assert abs(s.portfolio_dd - 0.05) < 1e-10

    def test_portfolio_dd_no_drawdown(self):
        s = _state(equity=1.0, peak_equity=1.0)
        assert s.portfolio_dd == 0.0

    def test_portfolio_dd_zero_peak(self):
        s = _state(equity=0.0, peak_equity=0.0)
        assert s.portfolio_dd == 0.0

    def test_strategy_dd_normal(self):
        s = _state(strategy_equity={"s1": 0.97}, strategy_peaks={"s1": 1.0})
        assert abs(s.strategy_dd("s1") - 0.03) < 1e-10

    def test_strategy_dd_unknown(self):
        s = _state()
        assert s.strategy_dd("unknown") == 0.0

    def test_total_exposure(self):
        s = _state(positions={
            "A": {"weight": 0.10},
            "B": {"weight": 0.20},
            "C": {"weight": 0.15},
        })
        assert abs(s.total_exposure() - 0.45) < 1e-10

    def test_total_exposure_empty(self):
        s = _state()
        assert s.total_exposure() == 0.0

    def test_instrument_weight_exists(self):
        s = _state(positions={"NIFTY": {"weight": 0.12}})
        assert abs(s.instrument_weight("NIFTY") - 0.12) < 1e-10

    def test_instrument_weight_missing(self):
        s = _state()
        assert s.instrument_weight("NIFTY") == 0.0


# ===================================================================
# 9. RISK LIMITS IMMUTABILITY
# ===================================================================

class TestRiskLimitsImmutable:
    def test_frozen(self):
        limits = RiskLimits()
        with pytest.raises(AttributeError):
            limits.max_portfolio_dd = 0.10  # type: ignore

    def test_defaults(self):
        limits = RiskLimits()
        assert limits.max_portfolio_dd == 0.05
        assert limits.max_strategy_dd == 0.03
        assert limits.max_single_instrument == 0.20
        assert limits.max_single_stock_fno == 0.05
        assert limits.vpin_block_threshold == 0.70
        assert limits.max_total_exposure == 1.50
        assert limits.max_correlated_exposure == 0.40

    def test_custom_values(self):
        limits = RiskLimits(max_portfolio_dd=0.10, vpin_block_threshold=0.50)
        assert limits.max_portfolio_dd == 0.10
        assert limits.vpin_block_threshold == 0.50


# ===================================================================
# 10. MULTI-TARGET SCENARIOS
# ===================================================================

class TestMultiTarget:
    def test_mixed_targets_flat_always_passes(self):
        rm = _rm()
        targets = [
            _target(symbol="NIFTY", direction="long", weight=0.10),
            _target(symbol="BANKNIFTY", direction="flat"),
        ]
        results = rm.check(targets, _state(vpin=0.80))
        assert results[0].approved is False  # VPIN blocks long
        assert results[1].approved is True   # flat always passes

    def test_multiple_strategies_independent_dd(self):
        rm = _rm()
        state = _state(
            strategy_equity={"s1_vrp": 0.96, "s4_iv_mr": 1.0},
            strategy_peaks={"s1_vrp": 1.0, "s4_iv_mr": 1.0},
        )
        targets = [
            _target(strategy_id="s1_vrp", symbol="NIFTY"),
            _target(strategy_id="s4_iv_mr", symbol="BANKNIFTY"),
        ]
        results = rm.check(targets, state)
        assert results[0].approved is False   # s1_vrp DD > 3%
        assert results[1].approved is True    # s4_iv_mr fine

    def test_empty_target_list(self):
        rm = _rm()
        results = rm.check([], _state())
        assert results == []


# ===================================================================
# 11. REDUCE SIZE GATE
# ===================================================================

class TestReduceSize:
    def test_reduce_size_is_approved(self):
        rm = _rm()
        result = rm.check([_target(weight=0.25)], _state())
        assert result[0].approved is True
        assert result[0].gate == GateResult.REDUCE_SIZE
        assert result[0].adjusted_weight < 0.25

    def test_no_reduce_when_within_limit(self):
        rm = _rm()
        result = rm.check([_target(weight=0.10)], _state())
        assert result[0].gate == GateResult.PASS
        assert result[0].adjusted_weight == 0.10

    def test_adjusted_weight_is_rounded(self):
        rm = _rm()
        result = rm.check([_target(weight=0.05)], _state())
        # Weight should be rounded to 6 decimal places
        assert result[0].adjusted_weight == round(result[0].adjusted_weight, 6)


# ===================================================================
# 12. RISKCHECRESULT PROPERTIES
# ===================================================================

class TestRiskCheckResult:
    def test_blocked_property(self):
        r = RiskCheckResult(
            target=_target(), gate=GateResult.BLOCK_VPIN,
            approved=False, adjusted_weight=0.0,
        )
        assert r.blocked is True

    def test_not_blocked_property(self):
        r = RiskCheckResult(
            target=_target(), gate=GateResult.PASS,
            approved=True, adjusted_weight=0.05,
        )
        assert r.blocked is False
