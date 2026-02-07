"""RegimeCoordinator — hysteresis, cooldown, and cross-strategy blocking.

Wraps the raw regime detector (S7) to add production-grade filtering:
  1. Hysteresis: suppress regime flips within min_hold bars
  2. Re-entry cooldown: block entries for cooldown_bars after a regime change
  3. Cross-strategy blocking: block all strategy entries when RANDOM regime active

Thread-safe: uses no mutable module-level state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RegimeLabel(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM = "random"


@dataclass
class RegimeState:
    """Internal mutable state for the coordinator."""

    current_regime: RegimeLabel = RegimeLabel.RANDOM
    bars_in_regime: int = 0
    bars_since_change: int = 0
    last_raw_regime: RegimeLabel = RegimeLabel.RANDOM
    confidence: float = 0.0
    suppressed: bool = False


@dataclass(frozen=True)
class RegimeDecision:
    """Output of the coordinator for a single bar."""

    regime: RegimeLabel
    confidence: float
    suppressed: bool         # True if hysteresis suppressed a flip
    blocking: bool           # True if entries should be blocked
    cooldown_active: bool    # True if within re-entry cooldown
    bars_in_regime: int


class RegimeCoordinator:
    """Production regime gate with hysteresis and cooldown.

    Parameters
    ----------
    min_hold : int
        Minimum bars to hold a regime before allowing a flip.
    cooldown_bars : int
        Number of bars after a regime change during which re-entry is blocked.
    confidence_suppression : float
        Multiplier applied to confidence during suppression (default 0.5).
    """

    def __init__(
        self,
        min_hold: int = 3,
        cooldown_bars: int = 2,
        confidence_suppression: float = 0.5,
    ):
        self._min_hold = min_hold
        self._cooldown_bars = cooldown_bars
        self._confidence_suppression = confidence_suppression
        self._state = RegimeState()
        self._first_obs = True

    def update(
        self,
        raw_regime: str,
        raw_confidence: float,
    ) -> RegimeDecision:
        """Process a new bar's raw regime classification.

        Parameters
        ----------
        raw_regime : str
            Raw regime from detector ("trending", "mean_reverting", "random").
        raw_confidence : float
            Raw confidence from detector [0, 1].

        Returns
        -------
        RegimeDecision with filtered regime and blocking flags.
        """
        raw = RegimeLabel(raw_regime)
        s = self._state

        # First observation — accept unconditionally
        if self._first_obs:
            self._first_obs = False
            s.current_regime = raw
            s.last_raw_regime = raw
            s.confidence = raw_confidence
            s.bars_in_regime = 1
            s.bars_since_change = 0
            s.suppressed = False
            return RegimeDecision(
                regime=raw,
                confidence=raw_confidence,
                suppressed=False,
                blocking=(raw == RegimeLabel.RANDOM),
                cooldown_active=False,
                bars_in_regime=1,
            )

        s.bars_in_regime += 1
        s.bars_since_change += 1
        s.last_raw_regime = raw

        # Check if raw regime differs from current
        if raw != s.current_regime:
            # Hysteresis: suppress flip if we haven't held long enough
            if s.bars_in_regime <= self._min_hold:
                s.suppressed = True
                s.confidence = raw_confidence * self._confidence_suppression
                return RegimeDecision(
                    regime=s.current_regime,
                    confidence=s.confidence,
                    suppressed=True,
                    blocking=(s.current_regime == RegimeLabel.RANDOM),
                    cooldown_active=(s.bars_since_change <= self._cooldown_bars),
                    bars_in_regime=s.bars_in_regime,
                )
            else:
                # Allow the flip
                s.current_regime = raw
                s.bars_in_regime = 1
                s.bars_since_change = 0
                s.confidence = raw_confidence
                s.suppressed = False
                return RegimeDecision(
                    regime=raw,
                    confidence=raw_confidence,
                    suppressed=False,
                    blocking=(raw == RegimeLabel.RANDOM),
                    cooldown_active=True,  # just changed
                    bars_in_regime=1,
                )
        else:
            # Same regime — no suppression
            s.suppressed = False
            s.confidence = raw_confidence
            cooldown = s.bars_since_change <= self._cooldown_bars
            return RegimeDecision(
                regime=s.current_regime,
                confidence=raw_confidence,
                suppressed=False,
                blocking=(s.current_regime == RegimeLabel.RANDOM),
                cooldown_active=cooldown,
                bars_in_regime=s.bars_in_regime,
            )

    def reset(self) -> None:
        """Reset all internal state."""
        self._state = RegimeState()
        self._first_obs = True

    def allows_entry(self, strategy_id: str, decision: RegimeDecision) -> bool:
        """Check if a strategy is allowed to enter given the regime decision.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier (e.g. "s4_iv_mr", "s5_hawkes").
        decision : RegimeDecision
            Current regime decision.

        Returns
        -------
        True if entry is allowed.
        """
        # RANDOM blocks all entries
        if decision.blocking:
            return False

        # Cooldown blocks all entries
        if decision.cooldown_active:
            return False

        # TRENDING allows momentum strategies
        if decision.regime == RegimeLabel.TRENDING:
            return strategy_id in (
                "s5_hawkes", "s7_regime", "s1_vrp_options",
                "s6_pair_coint", "s3_vol_carry",
            )

        # MEAN_REVERTING allows mean-reversion strategies
        if decision.regime == RegimeLabel.MEAN_REVERTING:
            return strategy_id in (
                "s4_iv_mr", "s7_regime", "s1_vrp_options",
                "s5_hawkes", "s3_vol_carry",
            )

        return True

    @property
    def state(self) -> RegimeState:
        return self._state
