"""Paper trading broker adaptor — simulates order execution.

All orders are "filled" immediately at the order price (or last known
price).  No slippage, no partial fills.  Useful for testing the full
execution pipeline without real money.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from qlx.execution.types import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    PositionRecord,
)

logger = logging.getLogger(__name__)


class PaperAdaptor:
    """Simulated broker for paper trading.

    Fills all orders instantly at the stated price.
    Tracks positions and fills in memory.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._positions: dict[str, PositionRecord] = {}  # symbol → position

    def place_order(self, order: Order) -> str:
        """Simulate order placement — instant fill."""
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc).isoformat()

        order.order_id = order_id
        order.status = OrderStatus.COMPLETE
        order.filled_quantity = order.quantity
        order.average_price = order.price if order.price > 0 else 0.0
        order.timestamp = now
        self._orders[order_id] = order

        # Create fill
        fill = Fill(
            order_id=order_id,
            strategy_id=order.strategy_id,
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.quantity,
            price=order.average_price,
            timestamp=now,
            trade_id=f"PTRADE-{uuid.uuid4().hex[:8].upper()}",
        )
        self._fills.append(fill)

        # Update position
        self._update_position(order)

        logger.info(
            "PAPER FILL: %s %s %d × %s @ %.2f [%s]",
            order.side.value, order.symbol, order.quantity,
            order.exchange, order.average_price, order_id,
        )
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.OPEN:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def order_status(self, order_id: str) -> Order:
        return self._orders.get(order_id, Order(
            strategy_id="", symbol="", exchange="",
            side=OrderSide.BUY, quantity=0,
            status=OrderStatus.REJECTED,
        ))

    def positions(self) -> list[PositionRecord]:
        return [p for p in self._positions.values() if p.quantity != 0]

    def fills(self, order_id: str | None = None) -> list[Fill]:
        if order_id:
            return [f for f in self._fills if f.order_id == order_id]
        return list(self._fills)

    def _update_position(self, order: Order) -> None:
        """Update local position tracking after a fill."""
        key = f"{order.exchange}:{order.symbol}"
        pos = self._positions.get(key)

        qty_signed = order.quantity if order.side == OrderSide.BUY else -order.quantity

        if pos is None:
            self._positions[key] = PositionRecord(
                symbol=order.symbol,
                exchange=order.exchange,
                product=order.product.value,
                quantity=qty_signed,
                average_price=order.average_price,
                last_price=order.average_price,
                pnl=0.0,
                value=abs(qty_signed) * order.average_price,
                tradingsymbol=order.symbol,
            )
        else:
            new_qty = pos.quantity + qty_signed
            if new_qty == 0:
                # Position closed
                pnl = (order.average_price - pos.average_price) * pos.quantity
                pos.quantity = 0
                pos.pnl = pnl
            else:
                # Position modified
                if (pos.quantity > 0 and qty_signed > 0) or (pos.quantity < 0 and qty_signed < 0):
                    # Adding to position
                    total_cost = pos.average_price * abs(pos.quantity) + order.average_price * abs(qty_signed)
                    pos.average_price = total_cost / abs(new_qty)
                pos.quantity = new_qty
                pos.last_price = order.average_price
                pos.value = abs(new_qty) * order.average_price
