"""Risk Manager — Three-layer gate system for BRAHMASTRA.

Layer 1: VPIN Gate — blocks all entries when VPIN > threshold (toxic flow).
Layer 2: Drawdown Circuit Breaker — auto-flattens when DD exceeds limits.
Layer 3: Concentration Limits — caps exposure to single instruments/names.

Each gate is evaluated independently.  If ANY gate blocks, the position
is rejected or flattened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from quantlaxmi.core.allocator.meta import TargetPosition
from quantlaxmi.core.risk.limits import RiskLimits

logger = logging.getLogger(__name__)


class GateResult(Enum):
    PASS = "pass"
    BLOCK_VPIN = "block_vpin"
    BLOCK_DD_PORTFOLIO = "block_dd_portfolio"
    BLOCK_DD_STRATEGY = "block_dd_strategy"
    BLOCK_CONCENTRATION = "block_concentration"
    BLOCK_EXPOSURE = "block_exposure"
    REDUCE_SIZE = "reduce_size"


@dataclass
class RiskCheckResult:
    """Result of a risk check on a single target position."""

    target: TargetPosition
    gate: GateResult
    approved: bool
    adjusted_weight: float
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.approved


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""

    equity: float = 1.0
    peak_equity: float = 1.0
    positions: dict = field(default_factory=dict)       # symbol → {direction, weight, strategy_id}
    strategy_equity: dict = field(default_factory=dict)  # strategy_id → equity
    strategy_peaks: dict = field(default_factory=dict)   # strategy_id → peak equity
    vpin: float = 0.0                                    # current VPIN reading

    @property
    def portfolio_dd(self) -> float:
        """Current portfolio drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def strategy_dd(self, strategy_id: str) -> float:
        """Current drawdown for a specific strategy."""
        eq = self.strategy_equity.get(strategy_id, 1.0)
        peak = self.strategy_peaks.get(strategy_id, 1.0)
        if peak <= 0:
            return 0.0
        return (peak - eq) / peak

    def total_exposure(self) -> float:
        """Gross exposure as fraction of equity."""
        return sum(abs(p.get("weight", 0)) for p in self.positions.values())

    def instrument_weight(self, symbol: str) -> float:
        """Current weight in a specific instrument."""
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return abs(pos.get("weight", 0))


class RiskManager:
    """Three-layer risk gate system.

    Usage:
        rm = RiskManager(limits=RiskLimits())
        approved = rm.check(targets, portfolio_state)
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self._circuit_breaker_active = False

    def check(
        self,
        targets: list[TargetPosition],
        state: PortfolioState,
    ) -> list[RiskCheckResult]:
        """Run all risk gates on target positions.

        Returns a list of RiskCheckResult, one per target.
        Flat signals always pass (they reduce risk).
        """
        results: list[RiskCheckResult] = []

        for target in targets:
            # Flat signals always approved (exits reduce risk)
            if target.direction == "flat":
                results.append(RiskCheckResult(
                    target=target,
                    gate=GateResult.PASS,
                    approved=True,
                    adjusted_weight=0.0,
                ))
                continue

            result = self._check_single(target, state)
            results.append(result)

        # Log summary
        blocked = [r for r in results if r.blocked]
        if blocked:
            logger.warning(
                "Risk blocked %d/%d positions: %s",
                len(blocked),
                len(results),
                "; ".join(f"{r.target.symbol}:{r.gate.value}" for r in blocked),
            )

        return results

    def _check_single(
        self,
        target: TargetPosition,
        state: PortfolioState,
    ) -> RiskCheckResult:
        """Check a single target position against all gates."""

        # Layer 1: VPIN Gate
        if state.vpin > self.limits.vpin_block_threshold:
            self._circuit_breaker_active = True
            return RiskCheckResult(
                target=target,
                gate=GateResult.BLOCK_VPIN,
                approved=False,
                adjusted_weight=0.0,
                reason=f"VPIN={state.vpin:.3f} > {self.limits.vpin_block_threshold}",
            )

        # Layer 2: Drawdown Circuit Breaker
        if state.portfolio_dd > self.limits.max_portfolio_dd:
            self._circuit_breaker_active = True
            return RiskCheckResult(
                target=target,
                gate=GateResult.BLOCK_DD_PORTFOLIO,
                approved=False,
                adjusted_weight=0.0,
                reason=f"Portfolio DD={state.portfolio_dd:.1%} > {self.limits.max_portfolio_dd:.1%}",
            )

        if state.strategy_dd(target.strategy_id) > self.limits.max_strategy_dd:
            return RiskCheckResult(
                target=target,
                gate=GateResult.BLOCK_DD_STRATEGY,
                approved=False,
                adjusted_weight=0.0,
                reason=(
                    f"Strategy {target.strategy_id} DD="
                    f"{state.strategy_dd(target.strategy_id):.1%} > "
                    f"{self.limits.max_strategy_dd:.1%}"
                ),
            )

        # Layer 3: Concentration Limits
        adjusted_weight = target.weight

        # Single instrument limit
        current_weight = state.instrument_weight(target.symbol)
        if current_weight + adjusted_weight > self.limits.max_single_instrument:
            max_add = max(0, self.limits.max_single_instrument - current_weight)
            if max_add <= 0.001:
                return RiskCheckResult(
                    target=target,
                    gate=GateResult.BLOCK_CONCENTRATION,
                    approved=False,
                    adjusted_weight=0.0,
                    reason=f"{target.symbol} weight={current_weight:.1%} at limit",
                )
            adjusted_weight = min(adjusted_weight, max_add)

        # Stock FnO limit (for non-index names)
        index_names = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX", "BANKEX"}
        if target.symbol.upper() not in index_names:
            if current_weight + adjusted_weight > self.limits.max_single_stock_fno:
                max_add = max(0, self.limits.max_single_stock_fno - current_weight)
                if max_add <= 0.001:
                    return RiskCheckResult(
                        target=target,
                        gate=GateResult.BLOCK_CONCENTRATION,
                        approved=False,
                        adjusted_weight=0.0,
                        reason=f"Stock FnO {target.symbol} at {self.limits.max_single_stock_fno:.0%} limit",
                    )
                adjusted_weight = min(adjusted_weight, max_add)

        # Total exposure limit
        if state.total_exposure() + adjusted_weight > self.limits.max_total_exposure:
            remaining = max(0, self.limits.max_total_exposure - state.total_exposure())
            if remaining <= 0.001:
                return RiskCheckResult(
                    target=target,
                    gate=GateResult.BLOCK_EXPOSURE,
                    approved=False,
                    adjusted_weight=0.0,
                    reason=f"Total exposure at {self.limits.max_total_exposure:.0%} limit",
                )
            adjusted_weight = min(adjusted_weight, remaining)

        # Size was reduced
        gate = GateResult.PASS
        if adjusted_weight < target.weight:
            gate = GateResult.REDUCE_SIZE

        return RiskCheckResult(
            target=target,
            gate=gate,
            approved=True,
            adjusted_weight=round(adjusted_weight, 6),
        )

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def reset_circuit_breaker(self) -> None:
        self._circuit_breaker_active = False
        logger.info("Circuit breaker reset")
