"""Execution types: Order, Fill, Position."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class ProductType(Enum):
    NRML = "NRML"      # Carry forward (F&O)
    MIS = "MIS"         # Intraday
    CNC = "CNC"         # Cash delivery


@dataclass
class Order:
    """Represents a single order to be sent to broker."""

    strategy_id: str
    symbol: str               # tradingsymbol (e.g. "NIFTY2620622000CE")
    exchange: str             # "NFO", "BFO", "NSE", "BSE"
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    product: ProductType = ProductType.NRML
    price: float = 0.0       # for LIMIT orders
    trigger_price: float = 0.0  # for SL orders

    # Broker fills these
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Fill:
    """Execution fill from broker."""

    order_id: str
    strategy_id: str
    symbol: str
    exchange: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: str
    trade_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PositionRecord:
    """Broker-reported position for reconciliation."""

    symbol: str
    exchange: str
    product: str
    quantity: int           # net quantity (positive = long, negative = short)
    average_price: float
    last_price: float
    pnl: float
    value: float
    tradingsymbol: str = ""
