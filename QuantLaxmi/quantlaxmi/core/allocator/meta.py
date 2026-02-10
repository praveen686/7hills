"""Meta-Allocator: VIX-regime-based portfolio allocation across strategies.

Regime weights:
| VIX Regime | S1 Options | S4 IV MR | S5 Micro | New Strats |
|------------|-----------|----------|----------|------------|
| Low (<13)  | 30%       | 30%      | 20%      | 20%        |
| Normal     | 25%       | 25%      | 25%      | 25%        |
| Elevated   | 15%       | 35%      | 25%      | 25%        |
| Extreme    | 0%        | 0%       | 0%       | 0%         |

Extreme regime = kill switch — all cash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from quantlaxmi.core.allocator.regime import VIXRegime, VIXRegimeType, detect_regime
from quantlaxmi.core.allocator.sizing import conviction_to_size, kelly_fraction
from quantlaxmi.data.store import MarketDataStore
from quantlaxmi.strategies.protocol import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetPosition:
    """Target position output from the allocator.

    Represents the desired state after allocation, before risk checks.
    """

    strategy_id: str
    symbol: str
    direction: str        # "long", "short", "flat"
    weight: float         # portfolio weight [0, 1]
    instrument_type: str  # "FUT", "CE", "PE", "SPREAD"
    strike: float = 0.0
    expiry: str = ""
    conviction: float = 0.0
    metadata: dict = field(default_factory=dict)


# Regime → strategy weight allocation
# Key is strategy_id, value is weight for that regime
REGIME_WEIGHTS: dict[VIXRegimeType, dict[str, float]] = {
    VIXRegimeType.LOW: {
        "s1_vrp": 0.30,
        "s4_iv_mr": 0.30,
        "s5_hawkes": 0.20,
        "_other": 0.20,
    },
    VIXRegimeType.NORMAL: {
        "s1_vrp": 0.25,
        "s4_iv_mr": 0.25,
        "s5_hawkes": 0.25,
        "_other": 0.25,
    },
    VIXRegimeType.ELEVATED: {
        "s1_vrp": 0.15,
        "s4_iv_mr": 0.35,
        "s5_hawkes": 0.25,
        "_other": 0.25,
    },
    VIXRegimeType.EXTREME: {
        "s1_vrp": 0.0,
        "s4_iv_mr": 0.0,
        "s5_hawkes": 0.0,
        "_other": 0.0,
    },
}

# Default Kelly parameters per strategy (from backtest results)
STRATEGY_KELLY_PARAMS: dict[str, dict] = {
    "s1_vrp": {"win_rate": 0.65, "avg_win": 0.025, "avg_loss": 0.012},
    "s4_iv_mr": {"win_rate": 0.60, "avg_win": 0.020, "avg_loss": 0.010},
    "s5_hawkes": {"win_rate": 0.58, "avg_win": 0.018, "avg_loss": 0.010},
}


class MetaAllocator:
    """VIX-regime-aware portfolio allocator.

    Takes raw signals from multiple strategies and produces sized
    target positions with regime-adjusted weights.
    """

    def __init__(
        self,
        regime_weights: dict[VIXRegimeType, dict[str, float]] | None = None,
        kelly_params: dict[str, dict] | None = None,
        kelly_mult: float = 0.25,
        max_single_position: float = 0.20,
    ):
        self._regime_weights = regime_weights or REGIME_WEIGHTS
        self._kelly_params = kelly_params or STRATEGY_KELLY_PARAMS
        self._kelly_mult = kelly_mult
        self._max_single = max_single_position

    def allocate(
        self,
        signals: list[Signal],
        regime: VIXRegime,
    ) -> list[TargetPosition]:
        """Convert raw signals into sized target positions.

        Parameters
        ----------
        signals : list[Signal]
            Raw signals from all strategies.
        regime : VIXRegime
            Current VIX regime.

        Returns
        -------
        list[TargetPosition]
            Sized positions ready for risk checks.
        """
        if regime.regime == VIXRegimeType.EXTREME:
            logger.warning(
                "EXTREME regime (VIX=%.1f) — all strategies disabled, going flat",
                regime.vix,
            )
            # Emit flat signals for all active positions
            return [
                TargetPosition(
                    strategy_id=s.strategy_id,
                    symbol=s.symbol,
                    direction="flat",
                    weight=0.0,
                    instrument_type=s.instrument_type,
                    conviction=0.0,
                    metadata={"reason": "extreme_vix_killswitch"},
                )
                for s in signals
                if s.direction != "flat"
            ]

        weights = self._regime_weights.get(regime.regime, self._regime_weights[VIXRegimeType.NORMAL])
        targets: list[TargetPosition] = []

        for signal in signals:
            if signal.direction == "flat":
                targets.append(TargetPosition(
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    direction="flat",
                    weight=0.0,
                    instrument_type=signal.instrument_type,
                    strike=signal.strike,
                    expiry=signal.expiry,
                    conviction=0.0,
                    metadata=signal.metadata,
                ))
                continue

            # Strategy allocation weight
            strat_weight = weights.get(signal.strategy_id, weights.get("_other", 0.10))

            # Kelly sizing
            kp = self._kelly_params.get(signal.strategy_id, {
                "win_rate": 0.50, "avg_win": 0.015, "avg_loss": 0.015,
            })
            base_kelly = kelly_fraction(
                win_rate=kp["win_rate"],
                avg_win=kp["avg_win"],
                avg_loss=kp["avg_loss"],
                kelly_mult=self._kelly_mult,
                max_fraction=self._max_single,
            )

            # Final weight = strategy_allocation × Kelly × conviction
            position_weight = strat_weight * conviction_to_size(
                conviction=signal.conviction,
                base_fraction=base_kelly,
                max_fraction=self._max_single,
            )

            # Cap at max single position
            position_weight = min(position_weight, self._max_single)

            targets.append(TargetPosition(
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                direction=signal.direction,
                weight=round(position_weight, 6),
                instrument_type=signal.instrument_type,
                strike=signal.strike,
                expiry=signal.expiry,
                conviction=signal.conviction,
                metadata={
                    **signal.metadata,
                    "regime": regime.regime.value,
                    "regime_vix": regime.vix,
                    "strat_weight": strat_weight,
                    "kelly_base": round(base_kelly, 6),
                },
            ))

        return targets
