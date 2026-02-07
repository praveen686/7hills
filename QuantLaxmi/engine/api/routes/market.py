"""Market data routes — option chains, volatility surfaces, VIX.

GET /api/market/chain/{symbol}     — option chain for a symbol
GET /api/market/option-chain       — option chain (query params, for frontend)
GET /api/market/surface/{symbol}   — volatility surface grid
GET /api/market/vix                — India VIX current value
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market", tags=["market"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class OptionChainOut(BaseModel):
    symbol: str
    date: str | None
    expiry: str | None
    n_strikes: int
    chain: list[dict[str, Any]]


class VolSurfaceOut(BaseModel):
    symbol: str
    date: str | None
    expiries: list[str]
    strikes: list[float]
    n_points: int
    surface: list[dict[str, Any]]


class VIXOut(BaseModel):
    value: float
    change: float
    change_pct: float
    timestamp: str


class OptionChainEntryOut(BaseModel):
    strike: float
    ce_ltp: float
    ce_oi: float
    ce_iv: float
    ce_delta: float
    pe_ltp: float
    pe_oi: float
    pe_iv: float
    pe_delta: float


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/chain/{symbol}", response_model=OptionChainOut)
async def get_option_chain(
    symbol: str,
    request: Request,
    trade_date: str | None = Query(
        None,
        description="Trading date as YYYY-MM-DD. Defaults to latest available.",
    ),
    expiry: str | None = Query(
        None,
        description="Expiry date as YYYY-MM-DD. Defaults to nearest expiry.",
    ),
) -> OptionChainOut:
    """Return the full option chain (all strikes, CE + PE) for an underlying."""
    svc = request.app.state.market_data_service

    d: date | None = None
    if trade_date is not None:
        try:
            d = date.fromisoformat(trade_date)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trade_date format: {exc}. Use YYYY-MM-DD.",
            ) from exc

    try:
        result = svc.get_option_chain(symbol.upper(), d=d, expiry=expiry)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch option chain for {symbol}: {exc}",
        ) from exc

    return OptionChainOut(**result)


@router.get("/surface/{symbol}", response_model=VolSurfaceOut)
async def get_vol_surface(
    symbol: str,
    request: Request,
    trade_date: str | None = Query(
        None,
        description="Trading date as YYYY-MM-DD. Defaults to latest available.",
    ),
) -> VolSurfaceOut:
    """Return the volatility surface (strikes x expiries) for an underlying.

    Returns last close prices and OI for each (strike, expiry, type) tuple.
    The frontend can compute implied vol via Black-Scholes if needed.
    """
    svc = request.app.state.market_data_service

    d: date | None = None
    if trade_date is not None:
        try:
            d = date.fromisoformat(trade_date)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid trade_date format: {exc}. Use YYYY-MM-DD.",
            ) from exc

    try:
        result = svc.get_vol_surface(symbol.upper(), d=d)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch vol surface for {symbol}: {exc}",
        ) from exc

    return VolSurfaceOut(**result)


@router.get("/vix", response_model=VIXOut)
async def get_vix(request: Request) -> VIXOut:
    """Return the latest India VIX value."""
    svc = request.app.state.market_data_service
    store = svc.store

    # Try nse_volatility table first
    try:
        dates = store.available_dates("nse_volatility")
        if dates:
            latest = dates[-1]
            df = store.sql(
                "SELECT * FROM nse_volatility WHERE date = ? LIMIT 5",
                [latest.isoformat()],
            )
            if not df.empty:
                # Look for VIX row
                for col in df.columns:
                    if "vix" in col.lower() or "india" in col.lower():
                        val = float(df[col].iloc[0])
                        if val > 0:
                            return VIXOut(
                                value=round(val, 2),
                                change=0.0,
                                change_pct=0.0,
                                timestamp=latest.isoformat(),
                            )
    except Exception as e:
        logger.debug("nse_volatility query failed: %s", e)

    # Fallback: check BrahmastraState
    state = request.app.state.engine
    if state.last_vix > 0:
        return VIXOut(
            value=round(state.last_vix, 2),
            change=0.0,
            change_pct=0.0,
            timestamp=state.last_scan_date or "",
        )

    # Default
    return VIXOut(value=15.0, change=0.0, change_pct=0.0, timestamp="")


@router.get("/option-chain", response_model=list[OptionChainEntryOut])
async def get_option_chain_flat(
    request: Request,
    symbol: str = Query("NIFTY", description="Underlying symbol"),
    expiry: str | None = Query(None, description="Expiry date as YYYY-MM-DD"),
) -> list[OptionChainEntryOut]:
    """Return option chain in the flat format the frontend expects.

    Pivots the raw chain data into one row per strike with CE and PE columns.
    """
    svc = request.app.state.market_data_service
    try:
        result = svc.get_option_chain(symbol.upper(), expiry=expiry)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch option chain for {symbol}: {exc}",
        ) from exc

    chain = result.get("chain", [])
    if not chain:
        return []

    # Group by strike, pivot CE/PE
    strikes: dict[float, dict] = {}
    for row in chain:
        strike = float(row.get("strike", 0))
        itype = str(row.get("instrument_type", "")).upper()
        close = float(row.get("close", 0) or 0)
        oi = float(row.get("oi", 0) or 0)
        volume = float(row.get("volume", 0) or 0)

        if strike not in strikes:
            strikes[strike] = {
                "strike": strike,
                "ce_ltp": 0, "ce_oi": 0, "ce_iv": 0, "ce_delta": 0,
                "pe_ltp": 0, "pe_oi": 0, "pe_iv": 0, "pe_delta": 0,
            }

        if itype == "CE":
            strikes[strike]["ce_ltp"] = close
            strikes[strike]["ce_oi"] = oi
        elif itype == "PE":
            strikes[strike]["pe_ltp"] = close
            strikes[strike]["pe_oi"] = oi

    entries = sorted(strikes.values(), key=lambda x: x["strike"])
    return [OptionChainEntryOut(**e) for e in entries]
