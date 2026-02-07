"""Order Management System — computes diffs and generates orders.

The OMS compares target positions (from the orchestrator) against
current broker positions and generates the minimum set of orders
needed to reach the target state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qlx.execution.adaptor import BrokerAdaptor
from qlx.execution.types import (
    Fill,
    Order,
    OrderSide,
    OrderType,
    ProductType,
    PositionRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class TargetPos:
    """Desired position state."""

    symbol: str
    exchange: str
    quantity: int         # positive = long, negative = short, 0 = flat
    strategy_id: str
    order_type: OrderType = OrderType.MARKET
    price: float = 0.0   # for limit orders


class OrderManager:
    """Computes position diffs and manages order lifecycle."""

    def __init__(self, adaptor: BrokerAdaptor):
        self.adaptor = adaptor
        self._pending_orders: list[str] = []

    def reconcile_and_order(
        self,
        targets: list[TargetPos],
    ) -> list[Order]:
        """Compare targets vs current positions and generate orders.

        Parameters
        ----------
        targets : list[TargetPos]
            Desired position state for each instrument.

        Returns
        -------
        list[Order]
            Orders placed with the broker.
        """
        current = self.adaptor.positions()
        current_map: dict[str, int] = {}
        for p in current:
            key = f"{p.exchange}:{p.symbol}"
            current_map[key] = p.quantity

        orders_placed: list[Order] = []

        for target in targets:
            key = f"{target.exchange}:{target.symbol}"
            current_qty = current_map.get(key, 0)
            diff = target.quantity - current_qty

            if diff == 0:
                continue  # already at target

            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)

            order = Order(
                strategy_id=target.strategy_id,
                symbol=target.symbol,
                exchange=target.exchange,
                side=side,
                quantity=qty,
                order_type=target.order_type,
                product=ProductType.NRML,
                price=target.price,
            )

            try:
                order_id = self.adaptor.place_order(order)
                order.order_id = order_id
                self._pending_orders.append(order_id)
                orders_placed.append(order)
                logger.info(
                    "Placed: %s %d × %s (%s → %d) [%s]",
                    side.value, qty, target.symbol,
                    current_qty, target.quantity, order_id,
                )
            except Exception as e:
                logger.error("Failed to place order for %s: %s", target.symbol, e)

        return orders_placed

    def get_fills(self) -> list[Fill]:
        """Get all fills from pending orders."""
        fills: list[Fill] = []
        for oid in self._pending_orders:
            fills.extend(self.adaptor.fills(oid))
        return fills
