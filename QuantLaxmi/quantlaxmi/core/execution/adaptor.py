"""Broker adaptor protocol for order execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from quantlaxmi.core.execution.types import Fill, Order, PositionRecord


@runtime_checkable
class BrokerAdaptor(Protocol):
    """Contract for broker integration.

    Implementations: PaperAdaptor (simulated), KiteAdaptor (Zerodha live).
    """

    def place_order(self, order: Order) -> str:
        """Place an order and return broker order_id."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled."""
        ...

    def order_status(self, order_id: str) -> Order:
        """Get current status of an order."""
        ...

    def positions(self) -> list[PositionRecord]:
        """Get all current positions from broker."""
        ...

    def fills(self, order_id: str | None = None) -> list[Fill]:
        """Get fills, optionally filtered by order_id."""
        ...
