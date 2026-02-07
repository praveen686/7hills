"""WAL Tap — non-blocking bus subscriber that persists EventEnvelopes.

The WalTap subscribes to all event types on the EventBus and feeds
them to the EventLogWriter.  It runs as a background asyncio task.

Design:
  - Non-blocking: uses an asyncio.Queue between bus and writer
  - Backpressure-aware: bounded queue
  - NO SILENT DROP: if the queue overflows, the tap raises
    WalOverflowError which should hard-stop the engine
  - Flush on shutdown: drains the queue before closing

The tap does NOT modify events — it's a pure persistence pass-through.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from brahmastra.engine.event_bus import EventBus, Event, EventType
from brahmastra.engine.event_log import EventLogWriter
from qlx.events.envelope import EventEnvelope
from qlx.events.serde import serialize_envelope

logger = logging.getLogger(__name__)


class WalOverflowError(Exception):
    """Raised when the WAL tap queue overflows.

    This is a fatal error — the engine must stop because we cannot
    guarantee no events were silently dropped.
    """


class WalTap:
    """Subscribes to EventBus and persists all events to the EventLog.

    Parameters
    ----------
    bus : EventBus
        The engine's event bus.
    log : EventLogWriter
        The JSONL event log writer.
    max_queue_size : int
        Maximum pending events before overflow error.
    """

    def __init__(
        self,
        bus: EventBus,
        log: EventLogWriter,
        max_queue_size: int = 16384,
    ):
        self._bus = bus
        self._log = log
        self._max_queue_size = max_queue_size
        self._queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(
            maxsize=max_queue_size,
        )
        self._task: asyncio.Task | None = None
        self._running = False
        self._persisted_count: int = 0
        self._overflow = False

    def subscribe_all(self) -> None:
        """Subscribe to all known event types on the bus.

        Call this before starting the background task.
        """
        for etype in EventType:
            queue = self._bus.subscribe(etype.value)
            # We'll drain these in the background task
            asyncio.ensure_future(self._drain_bus_queue(etype.value, queue))

    async def _drain_bus_queue(
        self,
        event_type: str,
        bus_queue: asyncio.Queue[Event],
    ) -> None:
        """Drain a single bus subscription queue into the WAL queue.

        Converts bus Events to EventEnvelopes and enqueues for persistence.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(bus_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Convert bus Event to EventEnvelope
            if isinstance(event.data, EventEnvelope):
                envelope = event.data
            else:
                # Legacy event — wrap in envelope
                envelope = self._log.emit(
                    event_type=event.event_type,
                    source=event.source,
                    payload=event.data if isinstance(event.data, dict) else {"raw": str(event.data)},
                )
                continue  # emit already appended

            try:
                self._queue.put_nowait(envelope)
            except asyncio.QueueFull:
                self._overflow = True
                logger.critical(
                    "WAL tap queue overflow at seq=%d! Events may be lost.",
                    envelope.seq,
                )
                raise WalOverflowError(
                    f"WAL tap queue full ({self._max_queue_size} events). "
                    f"Last seq={envelope.seq}. Engine must stop."
                )

    async def start(self) -> None:
        """Start the background persistence task."""
        self._running = True
        self.subscribe_all()
        self._task = asyncio.create_task(self._persist_loop())
        logger.info("WAL tap started (queue_size=%d)", self._max_queue_size)

    async def _persist_loop(self) -> None:
        """Background loop: drain queue and write to log."""
        while self._running:
            try:
                envelope = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._log.append(envelope)
            self._persisted_count += 1

        # Drain remaining on shutdown
        while not self._queue.empty():
            try:
                envelope = self._queue.get_nowait()
                self._log.append(envelope)
                self._persisted_count += 1
            except asyncio.QueueEmpty:
                break

    async def stop(self) -> None:
        """Stop the background task and flush."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._log.flush()
        logger.info("WAL tap stopped: %d events persisted", self._persisted_count)

    @property
    def persisted_count(self) -> int:
        return self._persisted_count

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def overflow_detected(self) -> bool:
        return self._overflow

    def stats(self) -> dict:
        return {
            "persisted_count": self._persisted_count,
            "pending": self.pending,
            "overflow": self._overflow,
            "running": self._running,
        }
