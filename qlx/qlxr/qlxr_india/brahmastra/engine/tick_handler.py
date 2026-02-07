"""Tick handler — receives KiteTicker ticks, builds bars, computes microstructure.

Responsibilities:
  1. Receive raw KiteTick objects from the WebSocket feed.
  2. Aggregate ticks into 1-minute OHLCV bars (per instrument_token).
  3. Compute rolling VPIN and tick entropy from the tick stream.
  4. Publish "tick" events (every tick) and "bar_1m" events (on bar close).
  5. Batch-insert completed bars into DuckDB for persistence.

The handler maintains per-instrument state: current bar accumulator,
tick price/volume buffers for microstructure features.  All operations
are async and non-blocking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from qlx.data.zerodha import KiteTick
from qlx.features.microstructure import tick_entropy, vpin_from_ticks

from brahmastra.engine.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Size of rolling tick buffers for microstructure features
_TICK_BUFFER_SIZE = 2000

# VPIN parameters tuned for Indian index futures (~tick every 100ms)
_VPIN_BUCKET_SIZE = 500_000.0    # 5 lakh INR per bucket
_VPIN_N_BUCKETS = 50
_VPIN_SIGMA_WINDOW = 100

# Tick entropy parameters
_ENTROPY_WINDOW = 100
_ENTROPY_BINS = 10


@dataclass
class _BarAccumulator:
    """Accumulates ticks into a single 1-minute bar for one instrument."""

    instrument_token: int
    minute_key: str = ""       # "YYYY-MM-DD HH:MM" in IST
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    oi: int = 0
    tick_count: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    def reset(self, minute_key: str, tick: KiteTick) -> None:
        """Start a new bar from the given tick."""
        self.minute_key = minute_key
        self.open = tick.last_price
        self.high = tick.last_price
        self.low = tick.last_price
        self.close = tick.last_price
        self.volume = tick.volume
        self.oi = tick.oi
        self.tick_count = 1
        self.first_timestamp = tick.timestamp
        self.last_timestamp = tick.timestamp

    def update(self, tick: KiteTick) -> None:
        """Update the current bar with a new tick."""
        price = tick.last_price
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume = tick.volume   # Kite volume is cumulative for the day
        self.oi = tick.oi
        self.tick_count += 1
        self.last_timestamp = tick.timestamp

    def to_dict(self) -> dict:
        """Export completed bar as a dict suitable for DuckDB insert."""
        return {
            "instrument_token": self.instrument_token,
            "minute": self.minute_key,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "oi": self.oi,
            "tick_count": self.tick_count,
            "first_ts": self.first_timestamp.isoformat() if self.first_timestamp else "",
            "last_ts": self.last_timestamp.isoformat() if self.last_timestamp else "",
        }


@dataclass
class _TickBuffer:
    """Rolling circular buffer of prices and volumes for microstructure."""

    prices: np.ndarray = field(default_factory=lambda: np.zeros(_TICK_BUFFER_SIZE, dtype=np.float64))
    volumes: np.ndarray = field(default_factory=lambda: np.zeros(_TICK_BUFFER_SIZE, dtype=np.float64))
    head: int = 0
    count: int = 0

    def append(self, price: float, volume: float) -> None:
        self.prices[self.head] = price
        self.volumes[self.head] = volume
        self.head = (self.head + 1) % _TICK_BUFFER_SIZE
        if self.count < _TICK_BUFFER_SIZE:
            self.count += 1

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return contiguous arrays ordered oldest-to-newest."""
        if self.count < _TICK_BUFFER_SIZE:
            return self.prices[:self.count].copy(), self.volumes[:self.count].copy()
        # Circular: oldest is at self.head, wrap around
        p = np.concatenate([self.prices[self.head:], self.prices[:self.head]])
        v = np.concatenate([self.volumes[self.head:], self.volumes[:self.head]])
        return p, v


class TickHandler:
    """Processes raw KiteTicker ticks into bars and microstructure features.

    Parameters
    ----------
    event_bus : EventBus
        The engine event bus for publishing tick/bar events.
    db_connection : Any, optional
        DuckDB connection for batch-inserting completed bars.
        If None, bars are only published as events.
    vpin_interval : int
        Compute VPIN every N ticks (default 50) to avoid overhead.
    """

    def __init__(
        self,
        event_bus: EventBus,
        db_connection: Any | None = None,
        vpin_interval: int = 50,
    ) -> None:
        self.event_bus = event_bus
        self._db = db_connection
        self._vpin_interval = vpin_interval

        # Per-instrument state
        self._bars: dict[int, _BarAccumulator] = {}
        self._buffers: dict[int, _TickBuffer] = {}
        self._tick_counters: dict[int, int] = {}

        # Latest microstructure values (per instrument)
        self._latest_vpin: dict[int, float] = {}
        self._latest_entropy: dict[int, float] = {}

        # Batch of completed bars waiting for DuckDB insert
        self._pending_bars: list[dict] = []
        self._bar_flush_threshold = 100

        # Stats
        self._total_ticks = 0
        self._total_bars = 0

    async def on_tick(self, tick: KiteTick) -> None:
        """Process a single raw tick from KiteTicker.

        This is the main entry point called from the collection loop.

        Steps:
          1. Publish "tick" event with enriched data.
          2. Feed tick into the 1-minute bar accumulator.
          3. If the minute boundary is crossed, close the old bar,
             publish a "bar_1m" event, and batch-insert into DuckDB.
          4. Periodically compute VPIN and tick entropy.
        """
        token = tick.instrument_token
        self._total_ticks += 1

        # --- Tick buffer for microstructure ---
        buf = self._buffers.get(token)
        if buf is None:
            buf = _TickBuffer()
            self._buffers[token] = buf
        buf.append(tick.last_price, float(tick.volume))

        # --- Periodic microstructure computation ---
        counter = self._tick_counters.get(token, 0) + 1
        self._tick_counters[token] = counter

        vpin_val = self._latest_vpin.get(token, float("nan"))
        entropy_val = self._latest_entropy.get(token, float("nan"))

        if counter % self._vpin_interval == 0 and buf.count >= _VPIN_SIGMA_WINDOW + 10:
            vpin_val, entropy_val = await self._compute_microstructure(token)

        # --- Publish tick event ---
        tick_data = {
            "instrument_token": token,
            "ltp": tick.last_price,
            "volume": tick.volume,
            "oi": tick.oi,
            "buy_qty": tick.buy_qty,
            "sell_qty": tick.sell_qty,
            "timestamp": tick.timestamp,
            "vpin": vpin_val,
            "entropy": entropy_val,
        }
        await self.event_bus.publish(EventType.TICK, tick_data, source="tick_handler")

        # --- Bar aggregation ---
        await self._aggregate_bar(token, tick)

    async def _aggregate_bar(self, token: int, tick: KiteTick) -> None:
        """Aggregate tick into the 1-minute bar for this instrument."""
        # Determine minute key in IST
        ts = tick.timestamp
        if ts is None:
            ts = datetime.now(IST)
        elif ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)

        minute_key = ts.strftime("%Y-%m-%d %H:%M")

        bar = self._bars.get(token)

        if bar is None:
            # First tick for this instrument
            bar = _BarAccumulator(instrument_token=token)
            bar.reset(minute_key, tick)
            self._bars[token] = bar
            return

        if minute_key == bar.minute_key:
            # Same minute -- update bar
            bar.update(tick)
            return

        # --- Minute boundary crossed: close old bar, start new one ---
        completed_bar = bar.to_dict()
        self._total_bars += 1

        # Enrich with microstructure
        completed_bar["vpin"] = self._latest_vpin.get(token, float("nan"))
        completed_bar["entropy"] = self._latest_entropy.get(token, float("nan"))

        # Publish bar_1m event
        await self.event_bus.publish(
            EventType.BAR_1M, completed_bar, source="tick_handler",
        )

        # Queue for DuckDB batch insert
        self._pending_bars.append(completed_bar)
        if len(self._pending_bars) >= self._bar_flush_threshold:
            await self._flush_bars()

        # Start new bar
        bar.reset(minute_key, tick)

    async def _compute_microstructure(self, token: int) -> tuple[float, float]:
        """Compute VPIN and entropy from the tick buffer.

        Runs in a thread executor to avoid blocking the event loop on
        the numpy computation.
        """
        buf = self._buffers.get(token)
        if buf is None or buf.count < _VPIN_SIGMA_WINDOW + 10:
            return float("nan"), float("nan")

        prices, volumes = buf.get_arrays()

        loop = asyncio.get_running_loop()

        def _compute() -> tuple[float, float]:
            vpin_arr = vpin_from_ticks(
                prices,
                volumes,
                bucket_size=_VPIN_BUCKET_SIZE,
                n_buckets=_VPIN_N_BUCKETS,
                sigma_window=_VPIN_SIGMA_WINDOW,
            )
            entropy_arr = tick_entropy(
                prices,
                window=_ENTROPY_WINDOW,
                n_bins=_ENTROPY_BINS,
            )
            # Take latest non-NaN values
            v = float(vpin_arr[~np.isnan(vpin_arr)][-1]) if np.any(~np.isnan(vpin_arr)) else float("nan")
            e = float(entropy_arr[~np.isnan(entropy_arr)][-1]) if np.any(~np.isnan(entropy_arr)) else float("nan")
            return v, e

        vpin_val, entropy_val = await loop.run_in_executor(None, _compute)

        self._latest_vpin[token] = vpin_val
        self._latest_entropy[token] = entropy_val

        return vpin_val, entropy_val

    async def _flush_bars(self) -> None:
        """Batch-insert pending completed bars into DuckDB."""
        if not self._pending_bars or self._db is None:
            self._pending_bars.clear()
            return

        bars = self._pending_bars.copy()
        self._pending_bars.clear()

        loop = asyncio.get_running_loop()

        def _insert() -> int:
            try:
                # Build INSERT from dicts
                if not bars:
                    return 0

                cols = list(bars[0].keys())
                placeholders = ", ".join(["?"] * len(cols))
                col_str = ", ".join(cols)

                self._db.execute(
                    f"CREATE TABLE IF NOT EXISTS live_bars_1m ("
                    f"  instrument_token INTEGER,"
                    f"  minute VARCHAR,"
                    f"  open DOUBLE,"
                    f"  high DOUBLE,"
                    f"  low DOUBLE,"
                    f"  close DOUBLE,"
                    f"  volume BIGINT,"
                    f"  oi BIGINT,"
                    f"  tick_count INTEGER,"
                    f"  first_ts VARCHAR,"
                    f"  last_ts VARCHAR,"
                    f"  vpin DOUBLE,"
                    f"  entropy DOUBLE"
                    f")"
                )

                for bar in bars:
                    values = [bar.get(c) for c in cols]
                    self._db.execute(
                        f"INSERT INTO live_bars_1m ({col_str}) VALUES ({placeholders})",
                        values,
                    )
                return len(bars)
            except Exception as e:
                logger.error("DuckDB bar insert failed: %s", e)
                return 0

        count = await loop.run_in_executor(None, _insert)
        if count > 0:
            logger.debug("Flushed %d bars to DuckDB", count)

    async def flush(self) -> None:
        """Force-flush any remaining bars (call on shutdown)."""
        # Close any open bars
        for token, bar in self._bars.items():
            if bar.tick_count > 0:
                completed = bar.to_dict()
                completed["vpin"] = self._latest_vpin.get(token, float("nan"))
                completed["entropy"] = self._latest_entropy.get(token, float("nan"))
                self._pending_bars.append(completed)

        await self._flush_bars()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_vpin(self, instrument_token: int) -> float:
        """Return the latest VPIN for an instrument."""
        return self._latest_vpin.get(instrument_token, float("nan"))

    def get_entropy(self, instrument_token: int) -> float:
        """Return the latest tick entropy for an instrument."""
        return self._latest_entropy.get(instrument_token, float("nan"))

    def stats(self) -> dict:
        """Return handler-level diagnostics."""
        return {
            "total_ticks": self._total_ticks,
            "total_bars": self._total_bars,
            "instruments": len(self._bars),
            "pending_bars": len(self._pending_bars),
            "vpin": {k: round(v, 4) for k, v in self._latest_vpin.items()},
            "entropy": {k: round(v, 4) for k, v in self._latest_entropy.items()},
        }
