"""Trading routes — order placement, cancellation, margin estimation.

POST   /api/trading/order           — place an order
DELETE /api/trading/order/{order_id} — cancel an order
POST   /api/trading/margin          — estimate margin requirement
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading", tags=["trading"])


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class OrderRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    orderType: str  # MARKET, LIMIT, SL, SL-M
    price: float | None = None
    triggerPrice: float | None = None
    product: str = "NRML"  # NRML or MIS


class OrderResponse(BaseModel):
    orderId: str
    status: str


class MarginRequest(BaseModel):
    symbol: str
    side: str
    quantity: int
    orderType: str
    price: float = 0
    product: str = "NRML"


class MarginResponse(BaseModel):
    required: float
    available: float


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.post("/order", response_model=OrderResponse)
async def place_order(req: OrderRequest, request: Request) -> OrderResponse:
    """Place a new order.

    In paper/backtest mode this creates a simulated fill.
    In live mode this would delegate to Kite Connect.
    """
    logger.info(
        "Order: %s %s %d %s @ %s (%s)",
        req.side, req.symbol, req.quantity, req.orderType,
        req.price or "MKT", req.product,
    )

    # Paper trading: immediate simulated fill
    order_id = str(uuid.uuid4())[:8]

    # If we have portfolio state, record the order
    try:
        state = request.app.state.engine
        state.record_order({
            "order_id": order_id,
            "symbol": req.symbol,
            "side": req.side,
            "quantity": req.quantity,
            "order_type": req.orderType,
            "price": req.price,
            "trigger_price": req.triggerPrice,
            "product": req.product,
            "status": "FILLED" if req.orderType == "MARKET" else "PENDING",
        })
    except Exception as e:
        logger.debug("record_order not available: %s", e)

    return OrderResponse(
        orderId=order_id,
        status="FILLED" if req.orderType == "MARKET" else "PENDING",
    )


@router.delete("/order/{order_id}")
async def cancel_order(order_id: str, request: Request) -> dict[str, str]:
    """Cancel a pending order."""
    logger.info("Cancel order: %s", order_id)

    try:
        state = request.app.state.engine
        state.cancel_order(order_id)
    except Exception as e:
        logger.debug("cancel_order not available: %s", e)

    return {"orderId": order_id, "status": "CANCELLED"}


@router.post("/margin", response_model=MarginResponse)
async def estimate_margin(req: MarginRequest, request: Request) -> MarginResponse:
    """Estimate margin requirement for an order.

    Simple approximation: qty * price * margin_pct.
    """
    price = req.price if req.price and req.price > 0 else 100.0

    # Try to get LTP from market data
    try:
        store = request.app.state.market_data_service.store
        from quantlaxmi.engine.api.routes.market import _INDEX_NAME_MAP
        index_name = _INDEX_NAME_MAP.get(req.symbol.upper())
        if index_name:
            df = store.sql(
                'SELECT "Closing Index Value" FROM nse_index_close '
                'WHERE LOWER("Index Name") = LOWER(?) '
                'ORDER BY "Date" DESC LIMIT 1',
                [index_name],
            )
            if not df.empty:
                price = float(df.iloc[0, 0])
    except Exception:
        pass

    # Margin rates
    margin_pct = 0.12 if req.product == "NRML" else 0.04  # MIS has lower margin
    required = req.quantity * price * margin_pct

    # Available from portfolio state
    available = 1_00_00_000.0  # Default 1 Crore
    try:
        state = request.app.state.engine
        available = state.cash * 1_00_00_000.0
    except Exception:
        pass

    return MarginResponse(required=round(required, 2), available=round(available, 2))
