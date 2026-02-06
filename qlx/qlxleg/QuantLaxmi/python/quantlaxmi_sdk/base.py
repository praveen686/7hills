"""QuantLaxmi Python SDK - Base Strategy Classes

This module provides the base abstractions for building trading strategies
that integrate with the QuantLaxmi platform.

Historical note: This SDK was originally named kubera_sdk and was renamed
as part of the Phase 4.8 Kubera decommission.
"""

import warnings
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class MarketPayload(BaseModel):
    """Simplified market data payload for SDK prototype."""
    price: float
    size: float
    side: str


class MarketEvent(BaseModel):
    """Market tick event with timestamp, symbol, and payload."""
    exchange_time: datetime
    symbol: str
    payload: MarketPayload


class BarPayload(BaseModel):
    """OHLCV bar data."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class OrderStatus(BaseModel):
    """Order execution status update."""
    order_id: UUID
    status: str
    filled_quantity: float
    avg_price: float


class QuantLaxmiStrategy(ABC):
    """Base class for QuantLaxmi trading strategies.

    Subclass this to implement custom strategies with tick and bar handlers.
    """

    def __init__(self, name: str):
        self.name = name
        self.positions = {}

    @abstractmethod
    def on_tick(self, event: MarketEvent):
        """Handle incoming market tick."""
        pass

    @abstractmethod
    def on_bar(self, symbol: str, bar: BarPayload):
        """Handle completed bar."""
        pass

    def on_fill(self, status: OrderStatus):
        """Handle order fill notification."""
        print(f"[{self.name}] Order {status.order_id} filled: {status.status} @ {status.avg_price}")


# Backwards compatibility alias (deprecated)
class KuberaStrategy(QuantLaxmiStrategy):
    """Deprecated alias for QuantLaxmiStrategy.

    This class exists only for backwards compatibility during migration.
    Use QuantLaxmiStrategy instead.
    """

    def __init__(self, name: str):
        warnings.warn(
            "KuberaStrategy is deprecated; use QuantLaxmiStrategy instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(name)
