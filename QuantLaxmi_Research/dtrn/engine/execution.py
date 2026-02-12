"""Execution model for India FnO — costs, slippage, fill simulation."""
from __future__ import annotations

import numpy as np
from ..config import DTRNConfig


class ExecutionModel:
    """Simulate order execution with realistic India FnO costs.

    Cost components:
    1. Brokerage: flat per order (e.g., Rs.20)
    2. Exchange transaction charge: % of turnover
    3. STT: % on sell side (futures delivery)
    4. GST: 18% on brokerage
    5. Stamp duty: % of turnover (buy side)
    6. Slippage: estimated from spread + volatility
    """

    def __init__(self, config: DTRNConfig = None):
        if config is None:
            config = DTRNConfig()
        self.config = config

    def compute_cost(
        self,
        quantity: int,          # contracts (absolute)
        price: float,           # execution price
        is_sell: bool = False,
        instrument: str = "NIFTY",
    ) -> float:
        """Compute total transaction cost in INR for one leg.

        Returns cost as a positive number (always deducted from PnL).
        """
        if quantity == 0:
            return 0.0

        lot_size = self.config.lot_sizes.get(instrument, 75)
        turnover = abs(quantity) * price

        # 1. Brokerage (flat per order)
        brokerage = self.config.brokerage_per_order

        # 2. Exchange transaction charge
        exchange_charge = turnover * self.config.exchange_txn_pct

        # 3. STT (only on sell side for futures)
        stt = turnover * self.config.stt_pct if is_sell else 0.0

        # 4. GST on brokerage + exchange charge
        gst = (brokerage + exchange_charge) * self.config.gst_pct

        # 5. Stamp duty (buy side)
        stamp = turnover * self.config.stamp_duty_pct if not is_sell else 0.0

        total = brokerage + exchange_charge + stt + gst + stamp
        return total

    def compute_slippage(
        self,
        quantity: int,
        price: float,
        volatility: float = 0.0,  # recent realized vol (optional)
    ) -> float:
        """Estimate slippage in price points."""
        bps = self.config.slippage_bps
        # Increase slippage with size and volatility
        size_impact = 1.0 + 0.1 * (abs(quantity) / 75)  # scale with lots
        vol_impact = 1.0 + 10.0 * volatility  # vol increases slippage
        return price * bps * 1e-4 * size_impact * vol_impact

    def simulate_fill(
        self,
        current_position: int,
        target_position: int,
        price: float,
        instrument: str = "NIFTY",
        volatility: float = 0.0,
    ) -> dict:
        """Simulate execution from current to target position.

        Returns dict with:
        - fill_price: execution price (mid + slippage)
        - cost: total transaction cost in INR
        - delta: contracts changed
        - new_position: resulting position
        """
        delta = target_position - current_position

        if delta == 0:
            return {
                "fill_price": price,
                "cost": 0.0,
                "delta": 0,
                "new_position": current_position,
            }

        is_sell = delta < 0
        abs_delta = abs(delta)

        # Slippage
        slip = self.compute_slippage(abs_delta, price, volatility)
        fill_price = price + slip if delta > 0 else price - slip  # adverse

        # Cost
        cost = self.compute_cost(abs_delta, fill_price, is_sell, instrument)

        return {
            "fill_price": fill_price,
            "cost": cost,
            "delta": delta,
            "new_position": target_position,
        }

    def roundtrip_cost_bps(self, price: float, quantity: int = 75, instrument: str = "NIFTY") -> float:
        """Estimate roundtrip cost in basis points (for quick reference)."""
        buy_cost = self.compute_cost(quantity, price, is_sell=False, instrument=instrument)
        sell_cost = self.compute_cost(quantity, price, is_sell=True, instrument=instrument)
        total = buy_cost + sell_cost
        turnover = 2 * quantity * price
        return (total / turnover) * 10000 if turnover > 0 else 0.0
