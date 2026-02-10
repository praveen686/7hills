"""Resilient WebSocket client with auto-reconnect and liveness monitoring.

Python port of ``quantlaxmi-runner-crypto::ws_resilient::ResilientWs``.

Features
--------
- Automatic reconnect with exponential backoff (1s -> 30s cap)
- Liveness timeout: forces reconnect if no data for N seconds
- Connection gap recording for audit
- Transparent ping/pong handling
- Optional custom headers (needed for SBE: ``X-MBX-APIKEY``)

Usage
-----
::

    ws = await ResilientWs.connect("wss://...", config=ResilientWsConfig())
    async for msg in ws:
        process(msg)
    print(ws.connection_gaps)

Provenance
----------
Extracted from ``QuantLaxmi_archive/qlxr_crypto/qlx/data/ws.py``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResilientWsConfig:
    """Mirrors ``ResilientWsConfig`` from the Rust crate."""

    initial_backoff: float = 1.0          # seconds
    max_backoff: float = 30.0             # seconds
    backoff_multiplier: float = 2.0
    liveness_timeout: float = 30.0        # seconds
    read_timeout: float = 5.0             # seconds
    max_reconnect_attempts: int = 100
    enable_ping: bool = True
    ping_interval: float = 30.0           # seconds


# ---------------------------------------------------------------------------
# Connection gap record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConnectionGap:
    """Recorded gap between disconnect and reconnect."""

    disconnect_ts: datetime
    reconnect_ts: datetime
    gap_ms: int
    reason: str
    attempts: int


# ---------------------------------------------------------------------------
# Resilient WebSocket
# ---------------------------------------------------------------------------

@dataclass
class ResilientWs:
    """Self-healing WebSocket connection.

    Use :meth:`connect` to create, then iterate with ``async for``.
    """

    url: str
    config: ResilientWsConfig = field(default_factory=ResilientWsConfig)
    extra_headers: dict[str, str] | None = None
    subprotocols: list[str] | None = None

    # Internal state (not part of public API)
    _ws: object | None = field(default=None, repr=False)  # websockets ClientConnection
    _last_message_time: float = field(default=0.0, repr=False)
    _last_ping_time: float = field(default=0.0, repr=False)
    _gaps: list[ConnectionGap] = field(default_factory=list, repr=False)
    _total_reconnects: int = field(default=0, repr=False)

    @classmethod
    async def connect(
        cls,
        url: str,
        config: ResilientWsConfig | None = None,
        extra_headers: dict[str, str] | None = None,
        subprotocols: list[str] | None = None,
    ) -> ResilientWs:
        """Create and connect a resilient WebSocket."""
        cfg = config or ResilientWsConfig()
        instance = cls(
            url=url,
            config=cfg,
            extra_headers=extra_headers,
            subprotocols=subprotocols,
        )
        await instance._connect()
        return instance

    async def _connect(self) -> None:
        """Establish the initial WebSocket connection."""
        logger.info("Connecting to %s", self.url)
        kwargs: dict = {}
        if self.extra_headers:
            kwargs["additional_headers"] = self.extra_headers
        if self.subprotocols:
            kwargs["subprotocols"] = self.subprotocols

        self._ws = await websockets.connect(
            self.url,
            ping_interval=None,   # we handle pings ourselves
            ping_timeout=None,
            close_timeout=5,
            max_size=2**22,       # 4 MB max frame
            **kwargs,
        )
        now = time.monotonic()
        self._last_message_time = now
        self._last_ping_time = now

    async def _reconnect(self, reason: str) -> bool:
        """Attempt reconnection with exponential backoff.

        Returns True if reconnected, False if max attempts exhausted.
        """
        disconnect_ts = datetime.now(timezone.utc)
        backoff = self.config.initial_backoff
        attempts = 0

        while attempts < self.config.max_reconnect_attempts:
            attempts += 1
            logger.info(
                "Reconnect attempt %d/%d (backoff=%.1fs, reason=%s)",
                attempts, self.config.max_reconnect_attempts, backoff, reason,
            )
            await asyncio.sleep(backoff)

            try:
                await self._connect()
                reconnect_ts = datetime.now(timezone.utc)
                gap_ms = int((reconnect_ts - disconnect_ts).total_seconds() * 1000)
                gap = ConnectionGap(
                    disconnect_ts=disconnect_ts,
                    reconnect_ts=reconnect_ts,
                    gap_ms=gap_ms,
                    reason=reason,
                    attempts=attempts,
                )
                self._gaps.append(gap)
                self._total_reconnects += 1
                logger.info(
                    "Reconnected (gap=%dms, attempts=%d, total=%d)",
                    gap_ms, attempts, self._total_reconnects,
                )
                return True
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %s", attempts, e)
                backoff = min(backoff * self.config.backoff_multiplier, self.config.max_backoff)

        logger.error("Max reconnect attempts (%d) exhausted", self.config.max_reconnect_attempts)
        return False

    async def next_message(self) -> bytes | str | None:
        """Get the next message, handling reconnection transparently.

        Returns None only when max reconnect attempts are exhausted.
        """
        while True:
            if self._ws is None:
                if not await self._reconnect("no connection"):
                    return None

            # Check liveness
            now = time.monotonic()
            if now - self._last_message_time > self.config.liveness_timeout:
                logger.warning(
                    "Liveness timeout (%.0fs), forcing reconnect",
                    now - self._last_message_time,
                )
                await self._close_ws()
                if not await self._reconnect("liveness timeout"):
                    return None
                continue

            # Send ping if due
            if (
                self.config.enable_ping
                and now - self._last_ping_time > self.config.ping_interval
            ):
                try:
                    assert self._ws is not None
                    await self._ws.ping()
                    self._last_ping_time = time.monotonic()
                except Exception as e:
                    logger.warning("Ping failed: %s", e)
                    await self._close_ws()
                    if not await self._reconnect(f"ping failed: {e}"):
                        return None
                    continue

            # Read next message with timeout
            try:
                assert self._ws is not None
                msg = await asyncio.wait_for(
                    self._ws.recv(), timeout=self.config.read_timeout,
                )
                self._last_message_time = time.monotonic()
                return msg
            except asyncio.TimeoutError:
                # Normal -- just loop back to check liveness
                continue
            except ConnectionClosed as e:
                reason = f"connection closed: {e.code} {e.reason}"
                logger.warning(reason)
                await self._close_ws()
                if not await self._reconnect(reason):
                    return None
                continue
            except Exception as e:
                reason = f"ws error: {e}"
                logger.warning(reason)
                await self._close_ws()
                if not await self._reconnect(reason):
                    return None
                continue

    async def send(self, data: str | bytes) -> None:
        """Send a message (e.g. subscription request)."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(data)

    async def close(self) -> None:
        """Gracefully close the connection."""
        await self._close_ws()

    async def _close_ws(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("WebSocket close failed: %s", e)
            self._ws = None

    def __aiter__(self) -> AsyncIterator[bytes | str]:
        return self

    async def __anext__(self) -> bytes | str:
        msg = await self.next_message()
        if msg is None:
            raise StopAsyncIteration
        return msg

    @property
    def connection_gaps(self) -> list[ConnectionGap]:
        return list(self._gaps)

    @property
    def total_reconnects(self) -> int:
        return self._total_reconnects

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            return self._ws.open  # type: ignore[union-attr]
        except AttributeError:
            return self._ws is not None
