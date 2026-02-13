"""Market data routes — option chains, volatility surfaces, VIX, bars, symbols, orderbook.

GET /api/market/chain/{symbol}        — option chain for a symbol
GET /api/market/option-chain          — option chain (query params, for frontend)
GET /api/market/surface/{symbol}      — volatility surface grid
GET /api/market/vix                   — India VIX current value
GET /api/market/bars/{symbol}         — historical OHLCV bars
GET /api/market/symbols               — symbol search
GET /api/market/orderbook/{symbol}    — orderbook snapshot
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market", tags=["market"])
status_router = APIRouter(prefix="/api/status", tags=["status"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Connection status endpoint
# ------------------------------------------------------------------

class ConnectionStatus(BaseModel):
    zerodha: str
    binance: str
    mode: str
    engine_running: bool
    ticks_received: int
    bars_completed: int
    signals_emitted: int
    uptime_seconds: int
    strategies_registered: int
    tokens_subscribed: int


@status_router.get("/connections", response_model=ConnectionStatus)
async def get_connection_status(request: Request) -> ConnectionStatus:
    """Return live engine connection status and diagnostics."""
    live_engine = getattr(request.app.state, "live_engine", None)

    if live_engine and live_engine.running:
        stats = live_engine.stats()
        return ConnectionStatus(
            zerodha="connected" if stats.get("feed_connected") else "disconnected",
            binance="disconnected",
            mode=stats.get("mode", "unknown"),
            engine_running=True,
            ticks_received=stats.get("ticks_received", 0),
            bars_completed=stats.get("bars_completed", 0),
            signals_emitted=stats.get("signals_emitted", 0),
            uptime_seconds=stats.get("uptime_seconds", 0),
            strategies_registered=stats.get("strategies_registered", 0),
            tokens_subscribed=stats.get("tokens_subscribed", 0),
        )

    return ConnectionStatus(
        zerodha="disconnected",
        binance="disconnected",
        mode="offline",
        engine_running=False,
        ticks_received=0,
        bars_completed=0,
        signals_emitted=0,
        uptime_seconds=0,
        strategies_registered=0,
        tokens_subscribed=0,
    )


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

    # Fallback: check PortfolioState
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


# ------------------------------------------------------------------
# Bar / Symbol / Orderbook response models
# ------------------------------------------------------------------

class BarOut(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class SymbolOut(BaseModel):
    symbol: str
    name: str
    exchange: str
    type: str
    lot_size: int
    tick_size: float


class OrderbookOut(BaseModel):
    model_config = {"populate_by_name": True}

    symbol: str
    bids: list[list[float]]
    asks: list[list[float]]
    spread: float
    mid_price: float = Field(serialization_alias="midPrice")


# ------------------------------------------------------------------
# Index name mapping
# ------------------------------------------------------------------

_INDEX_NAME_MAP: dict[str, str] = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services",
    "MIDCPNIFTY": "NIFTY MidSmall Financial Services",
}

_HARDCODED_SYMBOLS: list[dict[str, Any]] = [
    {"symbol": "NIFTY", "name": "Nifty 50", "exchange": "NSE", "type": "INDEX", "lot_size": 25, "tick_size": 0.05},
    {"symbol": "BANKNIFTY", "name": "Nifty Bank", "exchange": "NSE", "type": "INDEX", "lot_size": 15, "tick_size": 0.05},
    {"symbol": "FINNIFTY", "name": "Nifty Financial Services", "exchange": "NSE", "type": "INDEX", "lot_size": 25, "tick_size": 0.05},
    {"symbol": "MIDCPNIFTY", "name": "NIFTY MidSmall Financial Services", "exchange": "NSE", "type": "INDEX", "lot_size": 50, "tick_size": 0.05},
    {"symbol": "BTCUSDT", "name": "Bitcoin / USDT", "exchange": "BINANCE", "type": "CRYPTO", "lot_size": 1, "tick_size": 0.01},
    {"symbol": "ETHUSDT", "name": "Ethereum / USDT", "exchange": "BINANCE", "type": "CRYPTO", "lot_size": 1, "tick_size": 0.01},
]


# ------------------------------------------------------------------
# GET /api/market/bars/{symbol}
# ------------------------------------------------------------------

@router.get("/bars/{symbol}", response_model=list[BarOut])
async def get_bars(
    symbol: str,
    request: Request,
    interval: str = Query("daily", description="Bar interval: daily or 1min"),
    start_date: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str | None = Query(None, description="End date YYYY-MM-DD"),
) -> list[BarOut]:
    """Return historical OHLCV bars for a symbol."""
    store = request.app.state.market_data_service.store
    sym = symbol.upper()

    # Parse dates
    d_end = date.today()
    d_start = d_end - timedelta(days=365)
    if start_date:
        try:
            d_start = date.fromisoformat(start_date)
        except ValueError as exc:
            raise HTTPException(400, f"Invalid start_date: {exc}") from exc
    if end_date:
        try:
            d_end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise HTTPException(400, f"Invalid end_date: {exc}") from exc

    bars: list[dict[str, Any]] = []

    # --- Crypto (BTCUSDT, ETHUSDT) ---
    if sym in ("BTCUSDT", "ETHUSDT"):
        try:
            from pathlib import Path
            from quantlaxmi.data._paths import MARKET_DIR

            kline_dir = Path(MARKET_DIR).parent / "binance" / sym / "1h"
            if kline_dir.exists():
                import pyarrow.parquet as pq
                df = pq.read_table(str(kline_dir)).to_pandas()
                if "open_time" in df.columns:
                    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
                elif "timestamp" in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

                df = df[(df["date"] >= d_start) & (df["date"] <= d_end)]

                if interval == "daily":
                    # Aggregate 1h to daily
                    df["date_ts"] = pd.to_datetime(df["date"])
                    agg = df.groupby("date_ts").agg(
                        open=("open", "first"),
                        high=("high", "max"),
                        low=("low", "min"),
                        close=("close", "last"),
                        volume=("volume", "sum"),
                    ).reset_index()
                    for _, row in agg.iterrows():
                        bars.append({
                            "time": int(row["date_ts"].timestamp()),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        })
                else:
                    for _, row in df.iterrows():
                        ts = int(row.get("open_time", 0)) // 1000 if "open_time" in df.columns else int(pd.Timestamp(row["date"]).timestamp())
                        bars.append({
                            "time": ts,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row.get("volume", 0)),
                        })
        except Exception as e:
            logger.warning("Crypto bars failed for %s: %s", sym, e)

        bars.sort(key=lambda b: b["time"])
        return [BarOut(**b) for b in bars]

    # --- Index daily (NIFTY, BANKNIFTY, etc.) ---
    if interval == "daily":
        index_name = _INDEX_NAME_MAP.get(sym)
        if index_name:
            try:
                df = store.sql(
                    'SELECT "Index Date", "Open Index Value", "High Index Value", '
                    '"Low Index Value", "Closing Index Value", "Volume" '
                    'FROM nse_index_close '
                    'WHERE LOWER("Index Name") = LOWER(?) '
                    'AND "date" >= ? AND "date" <= ? '
                    'ORDER BY "date"',
                    [index_name, d_start.isoformat(), d_end.isoformat()],
                )
                for _, row in df.iterrows():
                    dt = pd.to_datetime(row["Index Date"])
                    bars.append({
                        "time": int(dt.timestamp()),
                        "open": float(row.get("Open Index Value", 0) or 0),
                        "high": float(row.get("High Index Value", 0) or 0),
                        "low": float(row.get("Low Index Value", 0) or 0),
                        "close": float(row.get("Closing Index Value", 0) or 0),
                        "volume": float(row.get("Volume", 0) or 0),
                    })
            except Exception as e:
                logger.warning("Index bars failed for %s: %s", sym, e)

            bars.sort(key=lambda b: b["time"])
            return [BarOut(**b) for b in bars]

    # --- NFO 1-min bars (futures) ---
    if interval == "1min":
        try:
            avail = store.available_dates("nfo_1min")
            dates_in_range = [d for d in avail if d_start <= d <= d_end]
            for d in dates_in_range[-30:]:  # Limit to recent 30 days
                try:
                    df = store.get_symbol_bars(sym, d, table="nfo_1min")
                    if df.empty:
                        continue
                    for _, row in df.iterrows():
                        ts = row.get("timestamp") or row.get("date")
                        if ts is not None:
                            ts_val = int(pd.to_datetime(ts).timestamp())
                        else:
                            ts_val = int(pd.Timestamp(d).timestamp())
                        bars.append({
                            "time": ts_val,
                            "open": float(row.get("open", 0) or 0),
                            "high": float(row.get("high", 0) or 0),
                            "low": float(row.get("low", 0) or 0),
                            "close": float(row.get("close", 0) or 0),
                            "volume": float(row.get("volume", 0) or 0),
                        })
                except Exception:
                    continue
        except Exception as e:
            logger.warning("1min bars failed for %s: %s", sym, e)

        bars.sort(key=lambda b: b["time"])
        return [BarOut(**b) for b in bars]

    # --- Stock daily: try nse_equity_close ---
    try:
        df = store.sql(
            'SELECT "date", "Open Price" AS open, "High Price" AS high, '
            '"Low Price" AS low, "Close Price" AS close, '
            '"Total Traded Quantity" AS volume '
            'FROM nse_equity_close '
            'WHERE UPPER("Symbol") = ? '
            'AND "date" >= ? AND "date" <= ? '
            'ORDER BY "date"',
            [sym, d_start.isoformat(), d_end.isoformat()],
        )
        for _, row in df.iterrows():
            dt = pd.to_datetime(row["date"])
            bars.append({
                "time": int(dt.timestamp()),
                "open": float(row.get("open", 0) or 0),
                "high": float(row.get("high", 0) or 0),
                "low": float(row.get("low", 0) or 0),
                "close": float(row.get("close", 0) or 0),
                "volume": float(row.get("volume", 0) or 0),
            })
    except Exception as e:
        logger.debug("Equity bars failed for %s: %s", sym, e)

    bars.sort(key=lambda b: b["time"])
    return [BarOut(**b) for b in bars]


# ------------------------------------------------------------------
# GET /api/market/symbols
# ------------------------------------------------------------------

@router.get("/symbols", response_model=list[SymbolOut])
async def get_symbols(
    request: Request,
    q: str = Query("", description="Search filter"),
) -> list[SymbolOut]:
    """Return available symbols, optionally filtered by query string."""
    results: list[dict[str, Any]] = []

    # Hardcoded indices + crypto
    for s in _HARDCODED_SYMBOLS:
        results.append(s)

    # Try instruments from DuckDB
    store = request.app.state.market_data_service.store
    try:
        avail = store.available_dates("instruments")
        if avail:
            latest = avail[-1]
            df = store.get_instruments(latest)
            if not df.empty:
                # Get FnO symbols
                for _, row in df.head(200).iterrows():
                    sym = str(row.get("tradingsymbol", "") or "")
                    name = str(row.get("name", "") or sym)
                    exch = str(row.get("exchange", "NSE") or "NSE")
                    lot = int(row.get("lot_size", 1) or 1)
                    tick = float(row.get("tick_size", 0.05) or 0.05)
                    if sym and sym not in {s["symbol"] for s in results}:
                        results.append({
                            "symbol": sym,
                            "name": name,
                            "exchange": exch,
                            "type": "FNO",
                            "lot_size": lot,
                            "tick_size": tick,
                        })
    except Exception as e:
        logger.debug("Instruments query failed: %s", e)

    # Filter
    query = q.upper().strip()
    if query:
        results = [
            s for s in results
            if query in s["symbol"].upper() or query in s["name"].upper()
        ]

    return [SymbolOut(**s) for s in results]


# ------------------------------------------------------------------
# GET /api/market/orderbook/{symbol}
# ------------------------------------------------------------------

@router.get("/orderbook/{symbol}", response_model=OrderbookOut)
async def get_orderbook(
    symbol: str,
    request: Request,
) -> OrderbookOut:
    """Return orderbook snapshot for a symbol.

    In backtest mode: synthetic book from latest close +/- spread.
    In live mode: from Kite depth API (if connected).
    """
    store = request.app.state.market_data_service.store
    sym = symbol.upper()

    # Try to get the latest close price
    close = 0.0
    index_name = _INDEX_NAME_MAP.get(sym)
    if index_name:
        try:
            df = store.sql(
                'SELECT "Closing Index Value" FROM nse_index_close '
                'WHERE LOWER("Index Name") = LOWER(?) '
                'ORDER BY "date" DESC LIMIT 1',
                [index_name],
            )
            if not df.empty:
                close = float(df.iloc[0, 0])
        except Exception:
            pass

    if close <= 0:
        # Try equity close
        try:
            df = store.sql(
                'SELECT "Close Price" FROM nse_equity_close '
                'WHERE UPPER("Symbol") = ? '
                'ORDER BY "date" DESC LIMIT 1',
                [sym],
            )
            if not df.empty:
                close = float(df.iloc[0, 0])
        except Exception:
            pass

    if close <= 0:
        close = 100.0  # Fallback

    # Synthetic orderbook: 5 levels each side
    spread_pct = 0.001  # 0.1% spread
    half_spread = close * spread_pct / 2
    mid = close
    tick = 0.05

    bids: list[list[float]] = []
    asks: list[list[float]] = []
    for i in range(5):
        bid_price = round(mid - half_spread - i * tick, 2)
        ask_price = round(mid + half_spread + i * tick, 2)
        # Synthetic sizes: larger near mid
        size = float((5 - i) * 100)
        bids.append([bid_price, size])
        asks.append([ask_price, size])

    spread = round(asks[0][0] - bids[0][0], 2)

    return OrderbookOut(
        symbol=sym,
        bids=bids,
        asks=asks,
        spread=spread,
        mid_price=mid,
    )
