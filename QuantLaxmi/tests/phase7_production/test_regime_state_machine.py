"""Phase 7 — Regime State Machine tests.

Tests regime hysteresis, coordinator gating, position TTL archiving,
and position state machine transitions (direction flip, cooldown re-entry).

20 tests across 4 test classes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from strategies.s7_regime.detector import (
    classify_regime,
    classify_regime_with_hysteresis,
    reset_regime_state,
    MarketRegime,
    RegimeObservation,
)
from engine.services.regime_coordinator import (
    RegimeCoordinator,
    RegimeLabel,
    RegimeDecision,
    RegimeState,
)
from engine.state import BrahmastraState, Position, ClosedTrade


# ======================================================================
# TestRegimeHysteresis — 6 tests
# ======================================================================


class TestRegimeHysteresis:
    """Tests for classify_regime_with_hysteresis and reset_regime_state."""

    def test_holds_min_bars(self, seeded_prices_500: np.ndarray):
        """Hysteresis suppresses flip when bars_in_regime < min_hold."""
        # Get the raw classification so we know what regime the data produces
        raw = classify_regime(seeded_prices_500, vpin=0.0)

        # Pick a previous regime that differs from raw
        if raw.regime == MarketRegime.TRENDING:
            previous = MarketRegime.MEAN_REVERTING
        else:
            previous = MarketRegime.TRENDING

        obs = classify_regime_with_hysteresis(
            seeded_prices_500,
            vpin=0.0,
            previous_regime=previous,
            bars_in_regime=1,   # < min_hold=3
            min_hold=3,
        )
        # Suppressed: should keep previous regime
        assert obs.regime == previous, (
            f"Expected suppressed regime {previous}, got {obs.regime}"
        )

    def test_flips_after_hold(self, seeded_prices_500: np.ndarray):
        """Flip is allowed when bars_in_regime >= min_hold."""
        raw = classify_regime(seeded_prices_500, vpin=0.0)

        if raw.regime == MarketRegime.TRENDING:
            previous = MarketRegime.MEAN_REVERTING
        else:
            previous = MarketRegime.TRENDING

        obs = classify_regime_with_hysteresis(
            seeded_prices_500,
            vpin=0.0,
            previous_regime=previous,
            bars_in_regime=4,   # >= min_hold=3
            min_hold=3,
        )
        # Flip allowed: should return raw regime
        assert obs.regime == raw.regime, (
            f"Expected flipped regime {raw.regime}, got {obs.regime}"
        )

    def test_resets(self):
        """reset_regime_state returns correct default dict."""
        state = reset_regime_state()
        assert isinstance(state, dict)
        assert state["regime"] == MarketRegime.RANDOM
        assert state["bars_in_regime"] == 0
        assert state["confidence"] == 0.0

    def test_vpin_forces_random_after_hold(self, seeded_prices_500: np.ndarray):
        """High VPIN causes raw=RANDOM; after min_hold bars, RANDOM is returned."""
        obs = classify_regime_with_hysteresis(
            seeded_prices_500,
            vpin=0.75,
            previous_regime=MarketRegime.TRENDING,
            bars_in_regime=5,   # >= min_hold=3 => allow flip
            min_hold=3,
        )
        # VPIN>0.70 makes raw=RANDOM; bars_in_regime>=min_hold => flip allowed
        assert obs.regime == MarketRegime.RANDOM

    def test_confidence_reduced_during_suppression(self, seeded_prices_500: np.ndarray):
        """When hysteresis suppresses a flip, confidence is halved."""
        raw = classify_regime(seeded_prices_500, vpin=0.0)

        if raw.regime == MarketRegime.TRENDING:
            previous = MarketRegime.MEAN_REVERTING
        else:
            previous = MarketRegime.TRENDING

        obs = classify_regime_with_hysteresis(
            seeded_prices_500,
            vpin=0.0,
            previous_regime=previous,
            bars_in_regime=1,   # < min_hold => suppressed
            min_hold=3,
        )
        expected_conf = raw.confidence * 0.5
        assert abs(obs.confidence - expected_conf) < 1e-12, (
            f"Expected confidence {expected_conf}, got {obs.confidence}"
        )

    def test_first_obs_no_hysteresis(self, seeded_prices_500: np.ndarray):
        """With previous_regime=None, raw classification is returned unchanged."""
        raw = classify_regime(seeded_prices_500, vpin=0.0)
        obs = classify_regime_with_hysteresis(
            seeded_prices_500,
            vpin=0.0,
            previous_regime=None,
        )
        assert obs.regime == raw.regime
        assert abs(obs.confidence - raw.confidence) < 1e-12
        assert abs(obs.entropy - raw.entropy) < 1e-12
        assert abs(obs.mutual_info - raw.mutual_info) < 1e-12


# ======================================================================
# TestRegimeCoordinator — 6 tests
# ======================================================================


class TestRegimeCoordinator:
    """Tests for RegimeCoordinator: hysteresis, cooldown, cross-strategy blocking."""

    def test_random_blocks_all(self):
        """RANDOM regime blocks entry for all strategies."""
        coord = RegimeCoordinator(min_hold=3, cooldown_bars=2)
        decision = coord.update("random", 0.8)

        assert decision.blocking is True
        for sid in ("s5_hawkes", "s4_iv_mr", "s7_regime", "s1_vrp_options",
                     "s6_pair_coint", "s3_vol_carry"):
            assert coord.allows_entry(sid, decision) is False, (
                f"RANDOM should block {sid}"
            )

    def test_trending_allows_momentum(self):
        """TRENDING regime allows momentum strategies after cooldown expires."""
        coord = RegimeCoordinator(min_hold=1, cooldown_bars=0)
        decision = coord.update("trending", 0.9)
        # First obs: cooldown_active=False, blocking=False
        assert decision.regime == RegimeLabel.TRENDING
        assert decision.blocking is False
        assert decision.cooldown_active is False
        assert coord.allows_entry("s5_hawkes", decision) is True

    def test_mean_reverting_allows_s4(self):
        """MEAN_REVERTING regime allows s4_iv_mr after cooldown expires."""
        coord = RegimeCoordinator(min_hold=1, cooldown_bars=0)
        decision = coord.update("mean_reverting", 0.85)
        assert decision.regime == RegimeLabel.MEAN_REVERTING
        assert coord.allows_entry("s4_iv_mr", decision) is True

    def test_cooldown(self):
        """After a regime change, cooldown_active=True for cooldown_bars, then False."""
        coord = RegimeCoordinator(min_hold=1, cooldown_bars=2)

        # First obs: TRENDING (accepted unconditionally, no cooldown)
        d1 = coord.update("trending", 0.9)
        assert d1.cooldown_active is False

        # Second obs: same regime — bars_since_change increments to 1 (<= 2)
        d2 = coord.update("trending", 0.9)
        assert d2.cooldown_active is True  # bars_since_change=1 <= cooldown_bars=2

        # Third obs: same — bars_since_change=2 <= 2
        d3 = coord.update("trending", 0.9)
        assert d3.cooldown_active is True

        # Fourth obs: same — bars_since_change=3 > 2
        d4 = coord.update("trending", 0.9)
        assert d4.cooldown_active is False

    def test_reset(self):
        """After reset(), the first observation is accepted unconditionally again."""
        coord = RegimeCoordinator(min_hold=3, cooldown_bars=2)
        coord.update("trending", 0.9)
        coord.update("trending", 0.9)

        coord.reset()

        # Post-reset: first obs accepted unconditionally
        decision = coord.update("mean_reverting", 0.8)
        assert decision.regime == RegimeLabel.MEAN_REVERTING
        assert decision.suppressed is False
        assert decision.bars_in_regime == 1

    def test_3x_determinism(self):
        """Three independent coordinators fed the same sequence produce identical results."""
        sequence = [
            ("trending", 0.9),
            ("trending", 0.85),
            ("mean_reverting", 0.7),
            ("mean_reverting", 0.75),
            ("random", 0.6),
            ("trending", 0.8),
            ("trending", 0.9),
        ]
        all_results: list[list[RegimeDecision]] = []

        for _ in range(3):
            coord = RegimeCoordinator(min_hold=3, cooldown_bars=2)
            results = [coord.update(r, c) for r, c in sequence]
            all_results.append(results)

        for i in range(len(sequence)):
            regimes = {all_results[j][i].regime for j in range(3)}
            confs = {all_results[j][i].confidence for j in range(3)}
            supps = {all_results[j][i].suppressed for j in range(3)}
            blocks = {all_results[j][i].blocking for j in range(3)}
            cools = {all_results[j][i].cooldown_active for j in range(3)}
            bars = {all_results[j][i].bars_in_regime for j in range(3)}

            assert len(regimes) == 1, f"Bar {i}: regime mismatch {regimes}"
            assert len(confs) == 1, f"Bar {i}: confidence mismatch {confs}"
            assert len(supps) == 1, f"Bar {i}: suppressed mismatch {supps}"
            assert len(blocks) == 1, f"Bar {i}: blocking mismatch {blocks}"
            assert len(cools) == 1, f"Bar {i}: cooldown mismatch {cools}"
            assert len(bars) == 1, f"Bar {i}: bars_in_regime mismatch {bars}"


# ======================================================================
# TestPositionTTL — 5 tests
# ======================================================================


class TestPositionTTL:
    """Tests for BrahmastraState.archive_stale_positions (TTL eviction)."""

    def test_stale_archived(self, state_with_positions: BrahmastraState):
        """Only BANKNIFTY (65 days old) is archived at max_age=30."""
        archived = state_with_positions.archive_stale_positions(
            current_date="2025-12-05", max_age_days=30,
        )
        assert len(archived) == 1
        assert archived[0].symbol == "BANKNIFTY"
        assert archived[0].strategy_id == "s4_iv_mr"
        # BANKNIFTY should no longer be in active positions
        assert state_with_positions.get_position("s4_iv_mr", "BANKNIFTY") is None

    def test_fresh_kept(self, state_with_positions: BrahmastraState):
        """s5_hawkes:NIFTY (4 days old) is not archived."""
        state_with_positions.archive_stale_positions(
            current_date="2025-12-05", max_age_days=30,
        )
        pos = state_with_positions.get_position("s5_hawkes", "NIFTY")
        assert pos is not None
        assert pos.entry_date == "2025-12-01"

    def test_boundary_exact(self, state_with_positions: BrahmastraState):
        """Position at exactly max_age days is NOT archived (> not >=)."""
        # s1_vrp_options: entry 2025-11-05, current 2025-12-05 = 30 days
        archived = state_with_positions.archive_stale_positions(
            current_date="2025-12-05", max_age_days=30,
        )
        # Only BANKNIFTY (65 days) should be archived; s1_vrp at exactly 30 is kept
        archived_symbols = {t.symbol for t in archived}
        assert "NIFTY" not in archived_symbols or all(
            t.strategy_id != "s1_vrp_options" for t in archived
        )
        pos = state_with_positions.get_position("s1_vrp_options", "NIFTY")
        assert pos is not None, "s1_vrp_options at exactly 30 days should NOT be archived"

    def test_generates_closed_trade(self, state_with_positions: BrahmastraState):
        """Archived positions appear in state.closed_trades."""
        assert len(state_with_positions.closed_trades) == 0
        archived = state_with_positions.archive_stale_positions(
            current_date="2025-12-05", max_age_days=30,
        )
        assert len(archived) > 0
        # Each archived trade should also be in closed_trades
        for trade in archived:
            assert trade in state_with_positions.closed_trades

    def test_exit_reason_ttl(self, state_with_positions: BrahmastraState):
        """Archived trades have exit_reason='ttl_expired'."""
        archived = state_with_positions.archive_stale_positions(
            current_date="2025-12-05", max_age_days=30,
        )
        for trade in archived:
            assert trade.exit_reason == "ttl_expired", (
                f"Expected exit_reason='ttl_expired', got '{trade.exit_reason}'"
            )


# ======================================================================
# TestPositionStateMachine — 3 tests
# ======================================================================


class TestPositionStateMachine:
    """Tests for position lifecycle transitions integrated with regime gating."""

    def test_direction_flip_flat_then_entry(self, fresh_state: BrahmastraState):
        """Close a long position, then open a short in the same instrument."""
        # Open long
        fresh_state.open_position(Position(
            strategy_id="s5_hawkes", symbol="NIFTY", direction="long",
            weight=0.10, instrument_type="FUT",
            entry_date="2025-12-01", entry_price=21500.0,
        ))
        assert fresh_state.get_position("s5_hawkes", "NIFTY") is not None

        # Close long
        trade = fresh_state.close_position(
            strategy_id="s5_hawkes", symbol="NIFTY",
            exit_date="2025-12-03", exit_price=21600.0,
            exit_reason="signal_flip",
        )
        assert trade is not None
        assert trade.direction == "long"
        assert fresh_state.get_position("s5_hawkes", "NIFTY") is None

        # Open short in the same instrument
        fresh_state.open_position(Position(
            strategy_id="s5_hawkes", symbol="NIFTY", direction="short",
            weight=0.10, instrument_type="FUT",
            entry_date="2025-12-03", entry_price=21600.0,
        ))
        pos = fresh_state.get_position("s5_hawkes", "NIFTY")
        assert pos is not None
        assert pos.direction == "short"

    def test_re_entry_after_cooldown(self):
        """RegimeCoordinator allows entry once cooldown expires."""
        coord = RegimeCoordinator(min_hold=1, cooldown_bars=2)

        # First obs: TRENDING, accepted unconditionally
        d1 = coord.update("trending", 0.9)
        assert d1.cooldown_active is False
        assert coord.allows_entry("s5_hawkes", d1) is True

        # Bars 2,3: same regime, within cooldown window (bars_since_change <= 2)
        d2 = coord.update("trending", 0.9)
        assert d2.cooldown_active is True
        assert coord.allows_entry("s5_hawkes", d2) is False

        d3 = coord.update("trending", 0.9)
        assert d3.cooldown_active is True
        assert coord.allows_entry("s5_hawkes", d3) is False

        # Bar 4: bars_since_change=3 > cooldown_bars=2 => cooldown lifted
        d4 = coord.update("trending", 0.9)
        assert d4.cooldown_active is False
        assert coord.allows_entry("s5_hawkes", d4) is True

    def test_no_re_entry_during_cooldown(self):
        """RegimeCoordinator blocks entry during cooldown after a regime change."""
        coord = RegimeCoordinator(min_hold=2, cooldown_bars=2)

        # Build up bars in TRENDING so a flip is allowed
        coord.update("trending", 0.9)   # first obs (bar 1)
        coord.update("trending", 0.9)   # bar 2 (bars_in_regime=2)
        coord.update("trending", 0.9)   # bar 3 (bars_in_regime=3 > min_hold=2)

        # Now push MEAN_REVERTING => flip allowed (bars_in_regime=4 > min_hold=2)
        d_flip = coord.update("mean_reverting", 0.8)
        assert d_flip.regime == RegimeLabel.MEAN_REVERTING
        assert d_flip.cooldown_active is True   # just changed

        # During cooldown: entry blocked
        assert coord.allows_entry("s4_iv_mr", d_flip) is False

        # Next bar: bars_since_change=1 <= cooldown_bars=2 => still in cooldown
        d_next = coord.update("mean_reverting", 0.8)
        assert d_next.cooldown_active is True
        assert coord.allows_entry("s4_iv_mr", d_next) is False

        # Two more bars to clear cooldown
        coord.update("mean_reverting", 0.8)  # bars_since_change=2 <= 2
        d_clear = coord.update("mean_reverting", 0.8)  # bars_since_change=3 > 2
        assert d_clear.cooldown_active is False
        assert coord.allows_entry("s4_iv_mr", d_clear) is True
