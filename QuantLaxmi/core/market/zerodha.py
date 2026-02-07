"""Zerodha (Kite Connect) data connector — REST historical data and live WebSocket ticks.

Provides:

1. **Auth** — :func:`create_kite_session` handles login with TOTP
2. **REST** — :func:`fetch_historical` downloads OHLCV candles
3. **Live WS** — :class:`KiteTickFeed` streams live tick data

All functions load credentials from the project ``.env`` via :func:`load_zerodha_env`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

import pandas as pd
import pyotp
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

from core.base.types import OHLCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_zerodha_env(env_path: str | Path | None = None) -> dict[str, str]:
    """Load Zerodha credentials from ``.env`` file."""
    if env_path:
        load_dotenv(env_path)
    else:
        here = Path(__file__).resolve().parent
        for parent in (here, here.parent, here.parent.parent):
            candidate = parent / ".env"
            if candidate.exists():
                load_dotenv(candidate)
                break

    return {
        "user_id": os.getenv("ZERODHA_USER_ID", ""),
        "password": os.getenv("ZERODHA_PASSWORD", ""),
        "totp_secret": os.getenv("ZERODHA_TOTP_SECRET", ""),
        "api_key": os.getenv("ZERODHA_API_KEY", ""),
        "api_secret": os.getenv("ZERODHA_API_SECRET", ""),
        "redirect_url": os.getenv("ZERODHA_REDIRECT_URL", ""),
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def generate_totp(secret: str) -> str:
    """Generate a TOTP code from the secret."""
    return pyotp.TOTP(secret).now()


def create_kite_session(
    api_key: str | None = None,
    api_secret: str | None = None,
    request_token: str | None = None,
    access_token: str | None = None,
    env_path: str | Path | None = None,
) -> KiteConnect:
    """Create an authenticated KiteConnect session.

    Parameters
    ----------
    api_key, api_secret : str, optional
        Override credentials (otherwise loaded from .env).
    request_token : str, optional
        If provided, generates a new session from the request token.
    access_token : str, optional
        If provided, reuses an existing session.
    env_path : str or Path, optional
        Explicit path to .env file.

    Returns
    -------
    KiteConnect
        Authenticated client ready for API calls.

    Notes
    -----
    The Kite login flow requires a browser redirect to obtain ``request_token``.
    For headless use, provide ``access_token`` directly from a cached session.
    The TOTP secret is available for scripts that automate the web login.
    """
    env = load_zerodha_env(env_path)
    api_key = api_key or env["api_key"]
    api_secret = api_secret or env["api_secret"]

    kite = KiteConnect(api_key=api_key)

    if access_token:
        kite.set_access_token(access_token)
    elif request_token:
        data = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        logger.info("Kite session created (access_token set)")
    else:
        logger.warning(
            "No access_token or request_token provided. "
            "Login URL: %s", kite.login_url()
        )

    return kite


# ---------------------------------------------------------------------------
# REST: Historical Data
# ---------------------------------------------------------------------------

# Kite interval strings
INTERVALS = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "60minute",
    "1d": "day",
}


def fetch_historical(
    kite: KiteConnect,
    instrument_token: int,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    continuous: bool = False,
    oi: bool = False,
) -> OHLCV:
    """Download historical OHLCV from Zerodha Kite.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated Kite client.
    instrument_token : int
        Kite instrument token (e.g. 256265 for NIFTY 50).
    interval : str
        One of: ``1m, 3m, 5m, 10m, 15m, 30m, 1h, 1d``.
    start, end : str or datetime, optional
        Time range.  Defaults to last 60 days.
    continuous : bool
        Continuous data for futures.
    oi : bool
        Include open interest.

    Returns
    -------
    OHLCV
        Canonical OHLCV with UTC DatetimeIndex.
    """
    kite_interval = INTERVALS.get(interval)
    if kite_interval is None:
        raise ValueError(f"Unsupported interval '{interval}'. Use one of: {list(INTERVALS)}")

    now = datetime.now(timezone.utc)
    if end is None:
        end_dt = now
    else:
        end_dt = pd.Timestamp(end).to_pydatetime()

    if start is None:
        start_dt = end_dt - timedelta(days=60)
    else:
        start_dt = pd.Timestamp(start).to_pydatetime()

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start_dt.strftime("%Y-%m-%d"),
        to_date=end_dt.strftime("%Y-%m-%d"),
        interval=kite_interval,
        continuous=continuous,
        oi=oi,
    )

    if not data:
        raise ValueError(
            f"No historical data for token {instrument_token} "
            f"({start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d})"
        )

    df = pd.DataFrame(data)

    # Normalize column names
    col_map = {"date": "Date", "open": "Open", "high": "High", "low": "Low",
               "close": "Close", "volume": "Volume"}
    df = df.rename(columns=col_map)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return OHLCV(df[["Open", "High", "Low", "Close", "Volume"]])


def fetch_instruments(kite: KiteConnect, exchange: str = "NSE") -> pd.DataFrame:
    """Fetch instrument list for an exchange.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated client.
    exchange : str
        ``"NSE"``, ``"BSE"``, ``"NFO"``, ``"CDS"``, ``"MCX"``, ``"BFO"``.

    Returns
    -------
    pd.DataFrame
        Instrument list with columns: instrument_token, tradingsymbol, name, etc.
    """
    instruments = kite.instruments(exchange)
    return pd.DataFrame(instruments)


# ---------------------------------------------------------------------------
# Live WebSocket Ticks
# ---------------------------------------------------------------------------

@dataclass
class KiteTick:
    """Single tick from Kite WebSocket."""

    instrument_token: int
    last_price: float
    timestamp: datetime | None = None
    volume: int = 0
    oi: int = 0
    buy_qty: int = 0
    sell_qty: int = 0
    ohlc: dict = field(default_factory=dict)
    depth: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict) -> KiteTick:
        return cls(
            instrument_token=raw.get("instrument_token", 0),
            last_price=raw.get("last_price", 0.0),
            timestamp=raw.get("exchange_timestamp"),
            volume=raw.get("volume_traded", 0),
            oi=raw.get("oi", 0),
            buy_qty=raw.get("total_buy_quantity", 0),
            sell_qty=raw.get("total_sell_quantity", 0),
            ohlc=raw.get("ohlc", {}),
            depth=raw.get("depth", {}),
        )


class KiteTickFeed:
    """Live tick feed using Kite WebSocket.

    The Kite ticker uses a callback-based API.  This class bridges it into
    an async-iterable interface using a queue.

    Usage::

        env = load_zerodha_env()
        kite = create_kite_session(access_token="...")
        feed = KiteTickFeed(
            api_key=env["api_key"],
            access_token=kite.access_token,
            tokens=[256265, 260105],  # NIFTY, BANKNIFTY
        )
        feed.start()
        async for tick in feed:
            print(tick)
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        tokens: list[int],
        mode: str = "full",
    ):
        self.api_key = api_key
        self.access_token = access_token
        self.tokens = tokens
        self.mode = mode  # "full", "quote", or "ltp"
        self._queue: asyncio.Queue[KiteTick] = asyncio.Queue(maxsize=8192)
        self._ticker: KiteTicker | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the Kite WebSocket connection in a background thread."""
        self._loop = asyncio.get_event_loop()
        self._ticker = KiteTicker(self.api_key, self.access_token)

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect

        # KiteTicker runs in its own thread via twisted
        self._ticker.connect(threaded=True)

    def _on_connect(self, ws, response) -> None:
        logger.info("Kite WS connected, subscribing to %d tokens", len(self.tokens))
        ws.subscribe(self.tokens)
        mode_map = {"full": ws.MODE_FULL, "quote": ws.MODE_QUOTE, "ltp": ws.MODE_LTP}
        ws.set_mode(mode_map.get(self.mode, ws.MODE_FULL), self.tokens)

    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        for raw in ticks:
            tick = KiteTick.from_raw(raw)
            if self._loop:
                self._loop.call_soon_threadsafe(self._enqueue, tick)

    def _enqueue(self, tick: KiteTick) -> None:
        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(tick)

    def _on_close(self, ws, code, reason) -> None:
        logger.warning("Kite WS closed: code=%s reason=%s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        logger.error("Kite WS error: code=%s reason=%s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.info("Kite WS reconnecting (attempt %d)", attempts_count)

    async def next(self) -> KiteTick:
        """Get the next tick."""
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[KiteTick]:
        return self

    async def __anext__(self) -> KiteTick:
        return await self._queue.get()

    def subscribe(self, new_tokens: list[int]) -> None:
        """Subscribe to additional instrument tokens (live, after start)."""
        self.tokens = list(set(self.tokens) | set(new_tokens))
        if self._ticker:
            self._ticker.subscribe(new_tokens)
            mode_map = {
                "full": self._ticker.MODE_FULL,
                "quote": self._ticker.MODE_QUOTE,
                "ltp": self._ticker.MODE_LTP,
            }
            self._ticker.set_mode(
                mode_map.get(self.mode, self._ticker.MODE_FULL), new_tokens,
            )

    def unsubscribe(self, old_tokens: list[int]) -> None:
        """Unsubscribe from instrument tokens (live, after start)."""
        remove = set(old_tokens)
        self.tokens = [t for t in self.tokens if t not in remove]
        if self._ticker:
            self._ticker.unsubscribe(old_tokens)

    def stop(self) -> None:
        """Stop the WebSocket connection."""
        if self._ticker:
            self._ticker.close()
