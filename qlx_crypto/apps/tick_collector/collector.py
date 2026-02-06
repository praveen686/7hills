"""Tick collector — WebSocket manager for aggTrade + bookTicker streams.

Connects to fstream.binance.com via ResilientWs, subscribes to aggTrade +
bookTicker for high-funding symbols. Feeds ticks into TickStore and
LiveRegimeTracker.

Symbol universe is refreshed every scan_interval by querying premiumIndex
for symbols with annualized funding > threshold.

Dynamic subscribe/unsubscribe:
  - Connect to wss://fstream.binance.com/ws (raw endpoint)
  - Send SUBSCRIBE/UNSUBSCRIBE JSON messages via ws.send()
  - Up to 1024 streams per connection (we need 2 * N_symbols ≈ 30)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from apps.tick_collector.storage import BookTick, TickStore, TradeTick
from qlx.data.ws import ResilientWs, ResilientWsConfig

logger = logging.getLogger(__name__)

PERP_WS_RAW = "wss://fstream.binance.com/ws"
FAPI_REST = "https://fapi.binance.com"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CollectorConfig:
    """Configuration for the tick collector."""

    # Symbol selection
    min_ann_funding_pct: float = 15.0    # scan symbols with funding > this
    min_volume_usd: float = 5_000_000    # min 24h volume
    max_symbols: int = 20                # max symbols to track
    scan_interval_sec: float = 3600.0    # re-scan every hour

    # WebSocket
    ws_liveness_timeout: float = 30.0
    ws_read_timeout: float = 5.0
    ws_max_reconnect: int = 200

    # Flush
    flush_interval_sec: float = 300.0
    flush_threshold: int = 50_000

    # Also collect top N by volume (for cascade detection)
    top_volume_count: int = 5


# ---------------------------------------------------------------------------
# Symbol scanner
# ---------------------------------------------------------------------------

def scan_symbols(config: CollectorConfig) -> list[dict]:
    """Scan for high-funding symbols via REST.

    Returns list of dicts: {symbol, ann_funding_pct, volume_24h_usd}.
    Sorted by annualized funding rate (descending).
    """
    # Fetch premium index for funding rates
    resp = requests.get(f"{FAPI_REST}/fapi/v1/premiumIndex", timeout=15)
    resp.raise_for_status()
    premium = resp.json()

    # Fetch 24h volumes
    resp2 = requests.get(f"{FAPI_REST}/fapi/v1/ticker/24hr", timeout=15)
    resp2.raise_for_status()
    volumes = {t["symbol"]: float(t.get("quoteVolume", 0)) for t in resp2.json()}

    candidates = []
    for p in premium:
        sym = p["symbol"]
        if not sym.endswith("USDT"):
            continue
        rate = float(p.get("lastFundingRate", 0))
        ann = rate * 3 * 365 * 100  # 3 settlements/day * 365 days * 100%
        vol = volumes.get(sym, 0)
        if vol < config.min_volume_usd:
            continue
        candidates.append({
            "symbol": sym,
            "ann_funding_pct": ann,
            "volume_24h_usd": vol,
            "funding_rate": rate,
        })

    # Sort by positive funding first (we earn from positive)
    by_funding = sorted(
        [c for c in candidates if c["ann_funding_pct"] > config.min_ann_funding_pct],
        key=lambda x: x["ann_funding_pct"],
        reverse=True,
    )

    # Also add top by volume for cascade/microstructure signals
    by_volume = sorted(candidates, key=lambda x: x["volume_24h_usd"], reverse=True)
    selected_syms = {c["symbol"] for c in by_funding[:config.max_symbols]}

    for c in by_volume:
        if len(selected_syms) >= config.max_symbols:
            break
        if c["symbol"] not in selected_syms:
            by_funding.append(c)
            selected_syms.add(c["symbol"])

    return by_funding[:config.max_symbols]


# ---------------------------------------------------------------------------
# TickCollector — core async collector
# ---------------------------------------------------------------------------

class TickCollector:
    """Manages WebSocket connection and tick ingestion.

    Lifecycle:
      1. scan_symbols() → get initial symbol set
      2. connect() → open WS to fstream.binance.com/ws
      3. subscribe(symbols) → send SUBSCRIBE for aggTrade + bookTicker
      4. run() → read loop: parse JSON, feed TickStore + LiveRegimeTracker
      5. Periodically re-scan and adjust subscriptions
    """

    def __init__(
        self,
        store: TickStore,
        config: CollectorConfig | None = None,
        on_trade: object | None = None,  # callback(symbol, TradeTick)
        on_book: object | None = None,   # callback(symbol, BookTick)
    ):
        self.config = config or CollectorConfig()
        self.store = store
        self._on_trade = on_trade
        self._on_book = on_book

        self._ws: ResilientWs | None = None
        self._subscribed: set[str] = set()  # currently subscribed symbols
        self._sub_id: int = 1

        # Stats
        self._trade_count: int = 0
        self._book_count: int = 0
        self._start_time: float = 0.0
        self._last_scan_time: float = 0.0
        self._running: bool = False

    async def connect(self) -> None:
        """Connect to Binance Futures WS (raw endpoint, no initial streams)."""
        ws_config = ResilientWsConfig(
            liveness_timeout=self.config.ws_liveness_timeout,
            read_timeout=self.config.ws_read_timeout,
            max_reconnect_attempts=self.config.ws_max_reconnect,
            initial_backoff=1.0,
            max_backoff=30.0,
            enable_ping=True,
            ping_interval=30.0,
        )
        self._ws = await ResilientWs.connect(PERP_WS_RAW, config=ws_config)
        logger.info("Connected to %s", PERP_WS_RAW)

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to aggTrade + bookTicker for given symbols."""
        if self._ws is None:
            raise RuntimeError("Not connected")

        new_syms = set(symbols) - self._subscribed
        if not new_syms:
            return

        params = []
        for sym in new_syms:
            params.append(f"{sym.lower()}@aggTrade")
            params.append(f"{sym.lower()}@bookTicker")

        msg = json.dumps({
            "method": "SUBSCRIBE",
            "params": params,
            "id": self._sub_id,
        })
        self._sub_id += 1
        await self._ws.send(msg)
        self._subscribed.update(new_syms)
        logger.info("Subscribed to %d new symbols: %s", len(new_syms), sorted(new_syms))

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from aggTrade + bookTicker for given symbols."""
        if self._ws is None:
            return

        remove_syms = set(symbols) & self._subscribed
        if not remove_syms:
            return

        params = []
        for sym in remove_syms:
            params.append(f"{sym.lower()}@aggTrade")
            params.append(f"{sym.lower()}@bookTicker")

        msg = json.dumps({
            "method": "UNSUBSCRIBE",
            "params": params,
            "id": self._sub_id,
        })
        self._sub_id += 1
        await self._ws.send(msg)
        self._subscribed -= remove_syms

        # Close storage writers for removed symbols
        for sym in remove_syms:
            self.store.remove_symbol(sym)

        logger.info("Unsubscribed from %d symbols: %s", len(remove_syms), sorted(remove_syms))

    async def _rescan_and_adjust(self) -> list[dict]:
        """Re-scan symbols and adjust subscriptions. Returns new symbol info."""
        loop = asyncio.get_running_loop()
        symbol_info = await loop.run_in_executor(None, scan_symbols, self.config)
        new_syms = {s["symbol"] for s in symbol_info}

        to_add = new_syms - self._subscribed
        to_remove = self._subscribed - new_syms

        if to_remove:
            await self.unsubscribe(list(to_remove))
        if to_add:
            await self.subscribe(list(to_add))

        self._last_scan_time = time.monotonic()
        return symbol_info

    def _parse_message(self, raw: str) -> None:
        """Parse a JSON WS message and route to trade/book handler."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Control message (subscribe response)
        if "result" in msg or "id" in msg:
            logger.debug("WS control: %s", str(msg)[:200])
            return

        event_type = msg.get("e")

        if event_type == "aggTrade":
            self._handle_trade(msg)
        elif event_type == "bookTicker":
            self._handle_book(msg)
        else:
            logger.debug("Unknown event type: %s", event_type)

    def _handle_trade(self, data: dict) -> None:
        """Process an aggTrade message."""
        symbol = data["s"]
        tick = TradeTick(
            agg_trade_id=data["a"],
            timestamp_ms=data["T"],
            price=float(data["p"]),
            quantity=float(data["q"]),
            is_buyer_maker=data["m"],
        )
        self.store.add_trade(symbol, tick)
        self._trade_count += 1

        if self._on_trade is not None:
            self._on_trade(symbol, tick)

    def _handle_book(self, data: dict) -> None:
        """Process a bookTicker message."""
        symbol = data["s"]
        tick = BookTick(
            timestamp_ms=data.get("E", data.get("T", 0)),
            bid_price=float(data["b"]),
            ask_price=float(data["a"]),
            bid_qty=float(data["B"]),
            ask_qty=float(data["A"]),
        )
        self.store.add_book(symbol, tick)
        self._book_count += 1

        if self._on_book is not None:
            self._on_book(symbol, tick)

    async def run(self, duration_sec: float | None = None) -> None:
        """Main collection loop.

        Reads WS messages, feeds to store, periodically re-scans and flushes.
        Runs forever unless duration_sec is set or stop() is called.
        """
        if self._ws is None:
            await self.connect()

        self._running = True
        self._start_time = time.monotonic()
        self._last_scan_time = self._start_time

        # Initial scan + subscribe
        symbol_info = await self._rescan_and_adjust()
        logger.info(
            "Initial scan: %d symbols — %s",
            len(symbol_info),
            ", ".join(s["symbol"] for s in symbol_info[:5]),
        )

        last_flush = time.monotonic()
        last_status = time.monotonic()

        while self._running:
            if self._ws is None:
                break

            # Check duration limit
            if duration_sec is not None:
                elapsed = time.monotonic() - self._start_time
                if elapsed >= duration_sec:
                    logger.info("Duration limit reached (%.0fs)", elapsed)
                    break

            # Read next message
            msg = await self._ws.next_message()
            if msg is None:
                logger.error("WebSocket connection lost permanently")
                break

            if isinstance(msg, str):
                self._parse_message(msg)

            now = time.monotonic()

            # Periodic flush
            if now - last_flush >= self.config.flush_interval_sec:
                self.store.maybe_flush()
                last_flush = now

            # Periodic re-scan
            if now - self._last_scan_time >= self.config.scan_interval_sec:
                try:
                    await self._rescan_and_adjust()
                except Exception as e:
                    logger.warning("Re-scan failed: %s", e)

            # Periodic status log (every 60s)
            if now - last_status >= 60.0:
                rate = self._trade_count / max(now - self._start_time, 1)
                logger.info(
                    "Stats: %d trades, %d books, %.1f trades/s, %d symbols",
                    self._trade_count, self._book_count, rate,
                    len(self._subscribed),
                )
                last_status = now

        # Final flush
        self.store.flush_all()
        logger.info(
            "Collector stopped. Total: %d trades, %d books",
            self._trade_count, self._book_count,
        )

    def stop(self) -> None:
        """Signal the collector to stop."""
        self._running = False

    async def close(self) -> None:
        """Stop and clean up."""
        self.stop()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self.store.close()

    def stats(self) -> dict:
        """Current collector statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "elapsed_sec": elapsed,
            "subscribed_symbols": sorted(self._subscribed),
            "n_symbols": len(self._subscribed),
            "total_trades": self._trade_count,
            "total_books": self._book_count,
            "trades_per_sec": self._trade_count / max(elapsed, 1),
            "books_per_sec": self._book_count / max(elapsed, 1),
            "ws_reconnects": self._ws.total_reconnects if self._ws else 0,
            "ws_connected": self._ws.is_connected if self._ws else False,
            "store": self.store.stats(),
        }
