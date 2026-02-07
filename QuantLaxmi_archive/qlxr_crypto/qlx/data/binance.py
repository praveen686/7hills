"""Binance data connector — REST klines, JSON WebSocket streams, and SBE streams.

Ported from ``QuantLaxmi`` Rust codebase:
- ``apps/basis-harvester/src/scanner.rs`` → REST endpoints
- ``apps/basis-harvester/src/feed.rs``    → combined WS bookTicker streams
- ``crates/quantlaxmi-sbe/src/lib.rs``    → SBE binary decoding

Provides three layers:

1. **REST** — :func:`fetch_klines` for historical OHLCV download
2. **JSON WS** — :class:`BookTickerFeed` for live spot/perp best bid/ask
3. **SBE WS** — :class:`SbeTradeStream` for ultra-low-latency trade data

All functions load credentials from the project ``.env`` file via
:func:`load_binance_env`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
import requests
import websocket
from dotenv import load_dotenv

from qlx.core.types import OHLCV
from qlx.data.sbe import AggTrade, DepthUpdate, decode_message
from qlx.data.ws import ResilientWs, ResilientWsConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_binance_env(env_path: str | Path | None = None) -> dict[str, str]:
    """Load Binance credentials from ``.env`` file.

    Search order:
    1. Explicit *env_path*
    2. ``qlx_python/.env`` (project root)
    3. Already-set environment variables
    """
    if env_path:
        load_dotenv(env_path)
    else:
        # Walk upward to find .env
        here = Path(__file__).resolve().parent
        for parent in (here, here.parent, here.parent.parent):
            candidate = parent / ".env"
            if candidate.exists():
                load_dotenv(candidate)
                break

    return {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "api_key_ed25519": os.getenv("BINANCE_API_KEY_ED25519", ""),
        "use_testnet": os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true",
        "environment": os.getenv("BINANCE_ENVIRONMENT", "MAINNET"),
    }


# ---------------------------------------------------------------------------
# URL builders (mirror feed.rs)
# ---------------------------------------------------------------------------

# Spot
SPOT_REST_BASE = "https://api.binance.com"
SPOT_WS_BASE = "wss://stream.binance.com:9443"

# Futures (USD-M)
FAPI_REST_BASE = "https://fapi.binance.com"
PERP_WS_BASE = "wss://fstream.binance.com"

# SBE
SBE_WS_BASE = "wss://stream-sbe.binance.com:9443"


def _spot_book_url(symbols: list[str]) -> str:
    """Build combined spot @bookTicker stream URL (mirror of ``spot_book_url`` in feed.rs)."""
    streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols)
    return f"{SPOT_WS_BASE}/stream?streams={streams}"


def _perp_book_url(symbols: list[str]) -> str:
    """Build combined perp @bookTicker stream URL (mirror of ``perp_book_url`` in feed.rs)."""
    streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols)
    return f"{PERP_WS_BASE}/stream?streams={streams}"


def _sbe_stream_url() -> str:
    """SBE WebSocket endpoint."""
    return f"{SBE_WS_BASE}/stream"


# ---------------------------------------------------------------------------
# REST: Historical Klines
# ---------------------------------------------------------------------------

_KLINE_COLUMNS = [
    "Open_time", "Open", "High", "Low", "Close", "Volume",
    "Close_time", "Quote_volume", "Trade_count",
    "Taker_buy_base", "Taker_buy_quote", "Ignore",
]


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    limit: int = 1000,
    market: str = "spot",
) -> OHLCV:
    """Download historical klines from Binance REST API.

    Parameters
    ----------
    symbol : str
        e.g. ``"BTCUSDT"``
    interval : str
        Binance interval string: ``1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M``
    start, end : str or datetime, optional
        Time range.  Strings parsed as ``pd.Timestamp``.
    limit : int
        Max candles per request (Binance cap: 1000 for spot, 1500 for futures).
    market : str
        ``"spot"`` or ``"futures"``.

    Returns
    -------
    OHLCV
        Canonical OHLCV with UTC DatetimeIndex.
    """
    if market == "futures":
        base = FAPI_REST_BASE
        endpoint = "/fapi/v1/klines"
    else:
        base = SPOT_REST_BASE
        endpoint = "/api/v3/klines"

    params: dict = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    if start is not None:
        ts = pd.Timestamp(start)
        params["startTime"] = int(ts.timestamp() * 1000)
    if end is not None:
        ts = pd.Timestamp(end)
        params["endTime"] = int(ts.timestamp() * 1000)

    all_rows: list = []
    url = f"{base}{endpoint}"

    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)

        # Paginate if we got exactly `limit` rows and have no explicit end
        if len(rows) < limit or end is not None:
            break

        # Move start to 1ms after last candle close
        last_close_time = rows[-1][6]
        params["startTime"] = last_close_time + 1

    if not all_rows:
        raise ValueError(f"No kline data returned for {symbol} {interval}")

    df = pd.DataFrame(all_rows, columns=_KLINE_COLUMNS)

    # Convert types
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    for col in ("Open", "High", "Low", "Close", "Volume", "Quote_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("Open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return OHLCV(df[["Open", "High", "Low", "Close", "Volume"]])


def fetch_premium_index() -> pd.DataFrame:
    """Fetch all premium index entries from Binance Futures (public).

    Mirrors ``fetch_premium_index()`` in scanner.rs.
    """
    url = f"{FAPI_REST_BASE}/fapi/v1/premiumIndex"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def fetch_24h_volumes() -> dict[str, float]:
    """Fetch 24h quote volume for all futures symbols.

    Mirrors ``fetch_24h_volumes()`` in scanner.rs.
    """
    url = f"{FAPI_REST_BASE}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return {
        t["symbol"]: float(t["quoteVolume"])
        for t in resp.json()
        if "quoteVolume" in t
    }


def fetch_spot_symbols() -> set[str]:
    """Fetch actively trading USDT spot pairs.

    Mirrors ``fetch_spot_symbols()`` in scanner.rs.
    """
    url = f"{SPOT_REST_BASE}/api/v3/exchangeInfo?permissions=SPOT"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return {
        s["symbol"]
        for s in resp.json()["symbols"]
        if s["status"] == "TRADING" and s["symbol"].endswith("USDT")
    }


def fetch_funding_rates(
    symbol: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch historical funding rates from Binance Futures (public).

    Parameters
    ----------
    symbol : str
        e.g. ``"BTCUSDT"``
    start, end : str or datetime, optional
        Time range.
    limit : int
        Max records per request (Binance cap: 1000).

    Returns
    -------
    DataFrame
        Columns: fundingRate, markPrice.  Index: UTC DatetimeIndex.
    """
    url = f"{FAPI_REST_BASE}/fapi/v1/fundingRate"
    params: dict = {"symbol": symbol.upper(), "limit": limit}

    if start is not None:
        ts = pd.Timestamp(start)
        params["startTime"] = int(ts.timestamp() * 1000)
    if end is not None:
        ts = pd.Timestamp(end)
        params["endTime"] = int(ts.timestamp() * 1000)

    all_rows: list = []
    while True:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        # Paginate: next page starts after last funding time
        params["startTime"] = rows[-1]["fundingTime"] + 1

    if not all_rows:
        raise ValueError(f"No funding rate data for {symbol}")

    df = pd.DataFrame(all_rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    if "markPrice" in df.columns:
        df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.set_index("fundingTime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# JSON WS: BookTicker Feed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BookTicker:
    """Best bid/ask snapshot — mirrors ``BookTicker`` struct in feed.rs."""

    symbol: str
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    source: str  # "spot" or "perp"


class BookTickerFeed:
    """Combined spot + perp bookTicker WebSocket feed.

    Mirrors ``BasisFeed`` in ``apps/basis-harvester/src/feed.rs``.
    Uses combined streams so total connections = 2 regardless of symbol count.

    Usage::

        feed = await BookTickerFeed.connect(["BTCUSDT", "ETHUSDT"])
        async for tick in feed:
            print(tick)  # BookTicker(...)
    """

    def __init__(
        self,
        symbols: list[str],
        spot_ws: ResilientWs | None = None,
        perp_ws: ResilientWs | None = None,
    ):
        self.symbols = symbols
        self._spot_ws = spot_ws
        self._perp_ws = perp_ws
        self._queue: asyncio.Queue[BookTicker] = asyncio.Queue(maxsize=8192)
        self._tasks: list[asyncio.Task] = []

    @classmethod
    async def connect(
        cls,
        symbols: list[str],
        include_spot: bool = True,
        include_perp: bool = True,
    ) -> BookTickerFeed:
        """Connect to Binance combined bookTicker streams."""
        config = ResilientWsConfig(
            liveness_timeout=60.0,
            read_timeout=10.0,
            initial_backoff=1.0,
            max_backoff=30.0,
            max_reconnect_attempts=100,
        )

        spot_ws = None
        perp_ws = None

        if include_spot:
            url = _spot_book_url(symbols)
            spot_ws = await ResilientWs.connect(url, config=config)

        if include_perp:
            url = _perp_book_url(symbols)
            perp_ws = await ResilientWs.connect(url, config=config)

        feed = cls(symbols=symbols, spot_ws=spot_ws, perp_ws=perp_ws)
        feed._start_readers()
        return feed

    def _start_readers(self) -> None:
        if self._spot_ws:
            self._tasks.append(asyncio.create_task(self._read_loop(self._spot_ws, "spot")))
        if self._perp_ws:
            self._tasks.append(asyncio.create_task(self._read_loop(self._perp_ws, "perp")))

    async def _read_loop(self, ws: ResilientWs, source: str) -> None:
        """Read JSON bookTicker messages and enqueue parsed BookTicker."""
        async for raw in ws:
            if isinstance(raw, bytes):
                continue  # skip binary frames in JSON mode
            try:
                msg = json.loads(raw)
                data = msg.get("data", msg)  # combined stream wraps in {"stream":..,"data":..}
                tick = BookTicker(
                    symbol=data["s"],
                    bid_price=float(data["b"]),
                    ask_price=float(data["a"]),
                    bid_qty=float(data["B"]),
                    ask_qty=float(data["A"]),
                    source=source,
                )
                try:
                    self._queue.put_nowait(tick)
                except asyncio.QueueFull:
                    # Drop oldest to prevent backpressure
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._queue.put_nowait(tick)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.debug("Skipping malformed bookTicker: %s", e)

    async def next(self) -> BookTicker:
        """Get the next book ticker update."""
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[BookTicker]:
        return self

    async def __anext__(self) -> BookTicker:
        return await self._queue.get()

    async def close(self) -> None:
        """Shut down all connections and background tasks."""
        for task in self._tasks:
            task.cancel()
        if self._spot_ws:
            await self._spot_ws.close()
        if self._perp_ws:
            await self._perp_ws.close()


# ---------------------------------------------------------------------------
# SBE WS: Trade Stream
# ---------------------------------------------------------------------------

class SbeStream:
    """Ultra-low-latency SBE stream from Binance.

    Connects to ``wss://stream-sbe.binance.com:9443/stream`` with raw
    ``X-MBX-APIKEY`` and ``Sec-WebSocket-Protocol: binance-sbe`` headers,
    matching the exact handshake that ``tokio-tungstenite`` performs in the
    Rust codebase.

    Uses ``websocket-client`` (sync, in a thread executor) because the
    ``websockets`` library's subprotocol negotiation adds validation that
    Binance's SBE gateway rejects.  The ED25519 API key is required.

    Usage::

        env = load_binance_env()
        stream = await SbeStream.connect(
            symbols=["BTCUSDT"],
            api_key=env["api_key_ed25519"],
        )
        async for event in stream:
            if isinstance(event, AggTrade):
                print(f"{event.price} x {event.quantity}")
    """

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self._ws: websocket.WebSocket | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    async def connect(
        cls,
        symbols: list[str],
        api_key: str | None = None,
        stream_type: str = "trade",
    ) -> SbeStream:
        """Connect and subscribe to SBE streams.

        Parameters
        ----------
        symbols : list[str]
            e.g. ``["BTCUSDT", "ETHUSDT"]``
        api_key : str, optional
            Binance ED25519 API key.  If *None*, loads from ``.env``.
        stream_type : str
            Stream suffix: ``"trade"``, ``"depth"``, etc.
        """
        if api_key is None:
            env = load_binance_env()
            api_key = env["api_key_ed25519"]
            if not api_key:
                raise ValueError(
                    "BINANCE_API_KEY_ED25519 not found in .env — "
                    "SBE requires the ED25519 key"
                )

        instance = cls(symbols=symbols)
        instance._loop = asyncio.get_running_loop()

        # Connect in thread (websocket-client is synchronous)
        ws = await instance._loop.run_in_executor(
            None, cls._sync_connect, api_key,
        )
        instance._ws = ws

        # Subscribe
        params = [f"{s.lower()}@{stream_type}" for s in symbols]
        sub_msg = json.dumps({"method": "SUBSCRIBE", "params": params, "id": 1})
        await instance._loop.run_in_executor(None, ws.send, sub_msg)

        logger.info("SBE subscribed to %s", params)
        return instance

    @staticmethod
    def _sync_connect(api_key: str) -> websocket.WebSocket:
        """Synchronous WebSocket connect with raw SBE headers."""
        ws = websocket.WebSocket()
        ws.connect(
            _sbe_stream_url(),
            header=[
                f"X-MBX-APIKEY: {api_key}",
                "Sec-WebSocket-Protocol: binance-sbe",
            ],
        )
        return ws

    async def next(self) -> AggTrade | DepthUpdate | None:
        """Get the next decoded SBE event."""
        if self._ws is None or self._loop is None:
            return None

        while True:
            try:
                opcode, data = await self._loop.run_in_executor(
                    None, self._ws.recv_data,
                )
            except Exception as e:
                logger.error("SBE recv error: %s", e)
                return None

            if opcode == 1:  # text — JSON control message
                text = data.decode("utf-8", errors="replace")
                logger.debug("SBE control: %s", text[:200])
                continue
            elif opcode == 2:  # binary — SBE frame
                try:
                    return decode_message(data)
                except ValueError as e:
                    logger.warning("SBE decode error: %s", e)
                    continue
            else:
                continue

    def __aiter__(self) -> AsyncIterator[AggTrade | DepthUpdate]:
        return self

    async def __anext__(self) -> AggTrade | DepthUpdate:
        event = await self.next()
        if event is None:
            raise StopAsyncIteration
        return event

    async def close(self) -> None:
        if self._ws and self._loop:
            await self._loop.run_in_executor(None, self._ws.close)
