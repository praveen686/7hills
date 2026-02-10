"""Lightweight Order Management System for tracking order lifecycle.

Routes all order flow through a single abstraction:
  submit_order → pending
  fill_order   → filled  (with fill_price, slippage, commission)
  reject_order → rejected (with reason)

Used by the Orchestrator to ensure every position change goes through
an auditable order → fill pipeline instead of directly creating Positions.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class OrderManager:
    """Lightweight Order Management System for tracking order lifecycle."""

    def __init__(self) -> None:
        self._pending_orders: dict[str, dict] = {}
        self._filled_orders: list[dict] = []
        self._rejected_orders: list[dict] = []

    def submit_order(
        self,
        strategy_id: str,
        symbol: str,
        direction: str,
        weight: float,
        instrument_type: str,
        price: float,
        metadata: dict | None = None,
    ) -> str:
        """Submit an order and return order_id."""
        order_id = str(uuid.uuid4())[:8]
        order = {
            "order_id": order_id,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "direction": direction,
            "weight": weight,
            "instrument_type": instrument_type,
            "price": price,
            "status": "pending",
            "metadata": metadata or {},
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._pending_orders[order_id] = order
        logger.debug(
            "Order submitted: %s %s %s weight=%.4f price=%.2f [%s]",
            order_id, direction, symbol, weight, price, strategy_id,
        )
        return order_id

    def fill_order(
        self,
        order_id: str,
        fill_price: float,
        slippage: float = 0.0,
        commission: float = 0.0,
    ) -> dict:
        """Mark order as filled."""
        if order_id not in self._pending_orders:
            raise KeyError(f"Order {order_id} not found in pending orders")
        order = self._pending_orders.pop(order_id)
        order["status"] = "filled"
        order["fill_price"] = fill_price
        order["slippage"] = slippage
        order["commission"] = commission
        order["filled_at"] = datetime.now(timezone.utc).isoformat()
        self._filled_orders.append(order)
        logger.debug(
            "Order filled: %s %s %s at %.2f (slip=%.4f, comm=%.4f)",
            order_id, order["direction"], order["symbol"],
            fill_price, slippage, commission,
        )
        return order

    def reject_order(self, order_id: str, reason: str) -> dict:
        """Mark order as rejected."""
        if order_id not in self._pending_orders:
            raise KeyError(f"Order {order_id} not found in pending orders")
        order = self._pending_orders.pop(order_id)
        order["status"] = "rejected"
        order["reason"] = reason
        order["rejected_at"] = datetime.now(timezone.utc).isoformat()
        self._rejected_orders.append(order)
        logger.debug(
            "Order rejected: %s %s %s — %s",
            order_id, order["direction"], order["symbol"], reason,
        )
        return order

    @property
    def pending(self) -> dict[str, dict]:
        return dict(self._pending_orders)

    @property
    def filled(self) -> list[dict]:
        return list(self._filled_orders)

    @property
    def rejected(self) -> list[dict]:
        return list(self._rejected_orders)

    def reset(self) -> None:
        """Clear all order state (useful for testing)."""
        self._pending_orders.clear()
        self._filled_orders.clear()
        self._rejected_orders.clear()
