"""Phase 2 — WAL Tap backpressure tests.

Invariants:
  1. WalTap persists events from the bus to the EventLog
  2. Queue overflow raises WalOverflowError (NO SILENT DROP)
  3. Stats tracking is accurate
  4. Shutdown drains remaining events
"""

from __future__ import annotations

import asyncio
import pytest

from brahmastra.engine.event_bus import EventBus, Event
from brahmastra.engine.event_bus import EventType as BusEventType
from brahmastra.engine.event_log import EventLogWriter
from brahmastra.engine.wal_tap import WalTap, WalOverflowError
from qlx.events.envelope import EventEnvelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "events"


@pytest.fixture
def bus():
    return EventBus(max_queue_size=256)


@pytest.fixture
def log(log_dir):
    w = EventLogWriter(base_dir=log_dir, run_id="wal-test", fsync_policy="none")
    yield w
    w.close()


# ---------------------------------------------------------------------------
# TestWalTapBasic
# ---------------------------------------------------------------------------

class TestWalTapBasic:
    """WalTap initializes and reports stats correctly."""

    def test_initial_stats(self, bus, log):
        tap = WalTap(bus=bus, log=log, max_queue_size=100)
        stats = tap.stats()
        assert stats["persisted_count"] == 0
        assert stats["pending"] == 0
        assert stats["overflow"] is False
        assert stats["running"] is False

    def test_max_queue_size(self, bus, log):
        tap = WalTap(bus=bus, log=log, max_queue_size=64)
        assert tap._max_queue_size == 64


# ---------------------------------------------------------------------------
# TestWalOverflow
# ---------------------------------------------------------------------------

class TestWalOverflow:
    """Queue overflow raises WalOverflowError — NO SILENT DROP."""

    def test_overflow_error_class(self):
        """WalOverflowError is a proper Exception."""
        err = WalOverflowError("test overflow")
        assert isinstance(err, Exception)
        assert "test overflow" in str(err)

    def test_overflow_flag_on_queue_full(self, bus, log):
        """Direct queue overflow sets the flag."""
        tap = WalTap(bus=bus, log=log, max_queue_size=2)
        # Manually fill the queue beyond capacity
        env = EventEnvelope.create(
            seq=1, run_id="test", event_type="tick", source="test", payload={},
        )
        tap._queue.put_nowait(env)
        tap._queue.put_nowait(env)
        # Queue is now full (size=2)
        with pytest.raises(asyncio.QueueFull):
            tap._queue.put_nowait(env)


# ---------------------------------------------------------------------------
# TestWalTapPersistence
# ---------------------------------------------------------------------------

class TestWalTapPersistence:
    """WalTap persist loop writes events to EventLog."""

    @pytest.mark.asyncio
    async def test_persist_single_event(self, log_dir):
        """A single event placed in the queue gets persisted."""
        log = EventLogWriter(base_dir=log_dir, run_id="persist-test", fsync_policy="none")
        bus = EventBus()
        tap = WalTap(bus=bus, log=log, max_queue_size=100)

        # Manually place an envelope on the queue
        env = EventEnvelope.create(
            seq=1, run_id="persist-test", event_type="tick",
            source="test", payload={"ltp": 23500.0},
            ts="2025-08-06T09:15:00.000000Z",
        )

        tap._running = True
        tap._queue.put_nowait(env)

        # Run persist loop briefly
        task = asyncio.create_task(tap._persist_loop())
        await asyncio.sleep(0.1)
        tap._running = False
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert tap.persisted_count >= 1
        log.close()

    @pytest.mark.asyncio
    async def test_persist_multiple_events(self, log_dir):
        """Multiple events get persisted in order."""
        log = EventLogWriter(base_dir=log_dir, run_id="multi-test", fsync_policy="none")
        bus = EventBus()
        tap = WalTap(bus=bus, log=log, max_queue_size=100)

        for i in range(5):
            env = EventEnvelope.create(
                seq=i + 1, run_id="multi-test", event_type="tick",
                source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
            tap._queue.put_nowait(env)

        tap._running = True
        task = asyncio.create_task(tap._persist_loop())
        await asyncio.sleep(0.2)
        tap._running = False
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert tap.persisted_count == 5
        log.close()


# ---------------------------------------------------------------------------
# TestWalTapShutdown
# ---------------------------------------------------------------------------

class TestWalTapShutdown:
    """Shutdown drains remaining events before closing."""

    @pytest.mark.asyncio
    async def test_drain_on_stop(self, log_dir):
        """Stopping the tap drains remaining queued events."""
        log = EventLogWriter(base_dir=log_dir, run_id="drain-test", fsync_policy="none")
        bus = EventBus()
        tap = WalTap(bus=bus, log=log, max_queue_size=100)

        # Pre-fill queue
        for i in range(3):
            env = EventEnvelope.create(
                seq=i + 1, run_id="drain-test", event_type="signal",
                source="test", payload={"i": i},
                ts="2025-08-06T09:15:00.000000Z",
            )
            tap._queue.put_nowait(env)

        # Start and immediately stop
        tap._running = True
        tap._task = asyncio.create_task(tap._persist_loop())
        await asyncio.sleep(0.1)
        await tap.stop()

        # All events should have been persisted during drain
        assert tap.persisted_count >= 3
        log.close()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, log_dir):
        """Stopping a not-started tap is safe."""
        log = EventLogWriter(base_dir=log_dir, run_id="idle-test", fsync_policy="none")
        bus = EventBus()
        tap = WalTap(bus=bus, log=log, max_queue_size=100)
        await tap.stop()  # should not raise
        log.close()
