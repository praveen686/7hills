"""Realistic Execution Simulation.

Extends the existing CostModel at qlx/backtest/costs.py with:
  - Market impact (Almgren-Chriss): impact = coefficient × (order_size / ADV)^0.5
  - Partial fills: max_fill_pct = 0.10 of ADV
  - Execution latency: configurable bar delay
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionModel:
    """Realistic execution cost and fill simulation.

    Extends the basic CostModel with market impact, partial fills,
    and execution latency.

    Attributes
    ----------
    commission_bps : float
        Commission per side in basis points.
    slippage_bps : float
        Base slippage per side in basis points.
    impact_coefficient : float
        Almgren-Chriss market impact coefficient (default 0.1).
        Total impact = coefficient × sqrt(order_size / ADV).
    max_fill_pct : float
        Maximum order size as fraction of ADV (default 10%).
        Orders larger than this are partially filled.
    latency_bars : int
        Execution latency in bars (default 1 = next bar).
        Signal at bar T → executed at bar T + latency_bars.
    funding_annual_pct : float
        Annual funding cost for carry positions (default 0).
    """

    commission_bps: float = 2.5
    slippage_bps: float = 2.5
    impact_coefficient: float = 0.10
    max_fill_pct: float = 0.10
    latency_bars: int = 1
    funding_annual_pct: float = 0.0

    @property
    def base_roundtrip_bps(self) -> float:
        """Base round-trip cost (commission + slippage) in bps."""
        return 2 * (self.commission_bps + self.slippage_bps)

    def market_impact_bps(self, order_size: float, adv: float) -> float:
        """Compute market impact in basis points.

        Uses Almgren-Chriss square-root model:
            impact = coefficient × sqrt(order_size / ADV) × 10000

        Parameters
        ----------
        order_size : float
            Order value in currency units.
        adv : float
            Average daily volume in currency units.

        Returns
        -------
        float
            One-way market impact in basis points.
        """
        if adv <= 0 or order_size <= 0:
            return 0.0
        participation = order_size / adv
        return self.impact_coefficient * math.sqrt(participation) * 10_000

    def total_cost_bps(self, order_size: float, adv: float) -> float:
        """Total round-trip cost including market impact."""
        impact = self.market_impact_bps(order_size, adv)
        return self.base_roundtrip_bps + 2 * impact

    def fill_quantity(self, order_size: float, adv: float) -> float:
        """Compute maximum fill quantity.

        Parameters
        ----------
        order_size : float
            Desired order value.
        adv : float
            Average daily volume.

        Returns
        -------
        float
            Actual fill value (may be less than order_size).
        """
        if adv <= 0:
            return 0.0
        max_fill = adv * self.max_fill_pct
        return min(order_size, max_fill)

    def apply_latency(
        self,
        signal_prices: np.ndarray,
        all_prices: np.ndarray,
    ) -> np.ndarray:
        """Delay execution by latency_bars.

        Parameters
        ----------
        signal_prices : np.ndarray
            Prices at signal generation time.
        all_prices : np.ndarray
            Full price series.

        Returns
        -------
        np.ndarray
            Prices at actual execution time (shifted).
        """
        if self.latency_bars <= 0:
            return signal_prices.copy()

        n = len(all_prices)
        execution_prices = np.full_like(signal_prices, np.nan)

        for i in range(len(signal_prices)):
            exec_idx = i + self.latency_bars
            if exec_idx < n:
                execution_prices[i] = all_prices[exec_idx]

        return execution_prices

    def simulate_fill(
        self,
        side: str,
        price: float,
        quantity: float,
        adv: float,
    ) -> tuple[float, float, float]:
        """Simulate a single order fill with costs.

        Parameters
        ----------
        side : str
            "buy" or "sell".
        price : float
            Market price at execution.
        quantity : float
            Order value desired.
        adv : float
            Average daily volume.

        Returns
        -------
        tuple[float, float, float]
            (fill_price, fill_quantity, total_cost_bps)
        """
        fill_qty = self.fill_quantity(quantity, adv)
        impact = self.market_impact_bps(fill_qty, adv) / 10_000

        if side == "buy":
            fill_price = price * (1 + impact + self.slippage_bps / 10_000)
        else:
            fill_price = price * (1 - impact - self.slippage_bps / 10_000)

        total_cost = self.commission_bps + self.slippage_bps + impact * 10_000

        return fill_price, fill_qty, total_cost


# Pre-configured models for different asset classes
INDEX_FUTURES_MODEL = ExecutionModel(
    commission_bps=2.0,
    slippage_bps=1.0,
    impact_coefficient=0.05,
    max_fill_pct=0.20,
    latency_bars=1,
)

STOCK_FUTURES_MODEL = ExecutionModel(
    commission_bps=3.0,
    slippage_bps=3.0,
    impact_coefficient=0.15,
    max_fill_pct=0.10,
    latency_bars=1,
)

OPTIONS_MODEL = ExecutionModel(
    commission_bps=5.0,
    slippage_bps=5.0,
    impact_coefficient=0.20,
    max_fill_pct=0.05,
    latency_bars=1,
)
