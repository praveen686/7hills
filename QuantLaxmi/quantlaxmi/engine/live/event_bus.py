"""AsyncIO pub/sub event bus for the QuantLaxmi real-time engine.

All engine components communicate through typed events on this bus.
Each subscriber gets its own asyncio.Queue, so slow consumers do not
block publishers or other subscribers.

Event types:
  - tick         : raw KiteTicker tick received
  - bar_1m       : 1-minute OHLCV bar completed
  - signal       : strategy signal emitted
  - fill         : order fill / execution report
  - risk_alert   : risk limit breached
  - state_update : state file changed

Usage::

    bus = EventBus()
    queue = bus.subscribe("tick")

    # Publisher side
    await bus.publish("tick", {"instrument_token": 256265, "ltp": 23500.0})

    # Subscriber side
    data = await queue.get()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Well-known event types flowing through the engine."""

    TICK = "tick"
    BAR_1M = "bar_1m"
    SIGNAL = "signal"
    FILL = "fill"
    RISK_ALERT = "risk_alert"
    STATE_UPDATE = "state_update"


@dataclass(frozen=True)
class Event:
    """Envelope wrapping every event on the bus.

    Attributes
    ----------
    event_type : str
        One of the EventType values (or any custom string).
    data : Any
        Payload -- dict, dataclass, numpy array, etc.
    timestamp : float
        monotonic clock timestamp at publish time.
    source : str
        Name of the component that published the event.
    """

    event_type: str
    data: Any
    timestamp: float
    source: str = ""


class EventBus:
    """Lock-free, asyncio-native pub/sub event bus.

    * Non-blocking: ``publish`` never awaits individual subscribers.
    * Per-subscriber backpressure: if a subscriber's queue is full, the
      oldest unread event is dropped (bounded queue, no OOM risk).
    * Thread-safe ``publish_sync`` for use from KiteTicker callbacks
      that fire on a Twisted thread.
    """

    def __init__(self, max_queue_size: int = 4096) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Event]]] = {}
        self._max_queue_size = max_queue_size
        self._loop: asyncio.AbstractEventLoop | None = None
        self._publish_count: int = 0
        self._drop_count: int = 0

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        max_size: int | None = None,
    ) -> asyncio.Queue[Event]:
        """Create a new subscription queue for *event_type*.

        Returns an ``asyncio.Queue`` that receives :class:`Event` objects
        whenever ``publish`` is called for the matching type.

        Parameters
        ----------
        event_type : str
            Event type to subscribe to (e.g. ``"tick"``).
        max_size : int, optional
            Override per-subscriber queue depth.  Defaults to the bus-level
            ``max_queue_size``.

        Returns
        -------
        asyncio.Queue[Event]
        """
        size = max_size if max_size is not None else self._max_queue_size
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=size)
        self._subscribers.setdefault(event_type, []).append(queue)
        logger.debug(
            "New subscriber for '%s' (total=%d)",
            event_type,
            len(self._subscribers[event_type]),
        )
        return queue

    def unsubscribe(self, event_type: str, queue: asyncio.Queue[Event]) -> None:
        """Remove a subscriber queue."""
        subs = self._subscribers.get(event_type, [])
        try:
            subs.remove(queue)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(
        self,
        event_type: str,
        data: Any,
        source: str = "",
    ) -> int:
        """Publish an event to all subscribers of *event_type*.

        Non-blocking: if a subscriber queue is full, the oldest event is
        discarded to make room.

        Returns the number of subscribers that received the event.
        """
        event = Event(
            event_type=event_type,
            data=data,
            timestamp=time.monotonic(),
            source=source,
        )
        subs = self._subscribers.get(event_type, [])
        delivered = 0
        for queue in subs:
            try:
                queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                # Drop oldest, enqueue new
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                    delivered += 1
                    self._drop_count += 1
                except asyncio.QueueFull:
                    self._drop_count += 1

        self._publish_count += 1
        return delivered

    def publish_sync(
        self,
        event_type: str,
        data: Any,
        source: str = "",
    ) -> None:
        """Thread-safe publish -- schedules ``publish`` on the event loop.

        Use this from non-asyncio threads (e.g. KiteTicker callback threads).
        """
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        self._loop.call_soon_threadsafe(
            self._sync_enqueue, event_type, data, source,
        )

    def _sync_enqueue(
        self,
        event_type: str,
        data: Any,
        source: str,
    ) -> None:
        """Synchronous enqueue called from ``call_soon_threadsafe``."""
        event = Event(
            event_type=event_type,
            data=data,
            timestamp=time.monotonic(),
            source=source,
        )
        subs = self._subscribers.get(event_type, [])
        for queue in subs:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass
                self._drop_count += 1
        self._publish_count += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return bus-level diagnostics."""
        return {
            "publish_count": self._publish_count,
            "drop_count": self._drop_count,
            "subscribers": {
                etype: len(subs) for etype, subs in self._subscribers.items()
            },
            "queue_depths": {
                etype: [q.qsize() for q in subs]
                for etype, subs in self._subscribers.items()
            },
        }

    def subscriber_count(self, event_type: str) -> int:
        return len(self._subscribers.get(event_type, []))

    def reset(self) -> None:
        """Remove all subscribers (for testing / shutdown)."""
        self._subscribers.clear()
        self._publish_count = 0
        self._drop_count = 0
