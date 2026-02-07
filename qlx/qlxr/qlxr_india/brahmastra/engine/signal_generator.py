"""Signal generator — runs strategy scans on live 1-minute bar data.

Listens for "bar_1m" events on the event bus.  When a new bar completes,
runs all registered strategies and publishes "signal" events for any
non-flat signals emitted.

In live mode strategies receive the latest bar data (not a full historical
date scan).  Each strategy's ``scan`` method is invoked with today's date
and the MarketDataStore (which includes the DuckDB live_bars_1m table for
intraday data).

The generator deduplicates: the same (strategy_id, symbol, direction) triple
is not re-emitted within a configurable cooldown window to avoid flooding
the downstream risk monitor and execution layer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from qlx.strategy.protocol import Signal, StrategyProtocol
from qlx.strategy.registry import StrategyRegistry
from qlx.data.store import MarketDataStore

from brahmastra.engine.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Minimum seconds between re-emitting the same signal
_SIGNAL_COOLDOWN_SEC = 120.0


@dataclass(frozen=True)
class LiveSignal:
    """Enriched signal emitted by the generator with live-specific context."""

    signal: Signal
    bar_minute: str           # the bar that triggered this signal
    generated_at: float       # monotonic timestamp
    instrument_token: int     # Kite token of the triggering bar
    bar_close: float          # close price of the triggering bar
    bar_vpin: float           # VPIN at bar close
    bar_entropy: float        # tick entropy at bar close


class SignalGenerator:
    """Runs registered strategies on each completed 1-minute bar.

    Parameters
    ----------
    event_bus : EventBus
        For subscribing to bar events and publishing signal events.
    registry : StrategyRegistry
        Contains all strategies to run on each bar.
    store : MarketDataStore
        Data access layer passed to strategy.scan().
    cooldown_sec : float
        Minimum seconds between re-emitting identical signals.
    """

    def __init__(
        self,
        event_bus: EventBus,
        registry: StrategyRegistry,
        store: MarketDataStore,
        cooldown_sec: float = _SIGNAL_COOLDOWN_SEC,
    ) -> None:
        self.event_bus = event_bus
        self.registry = registry
        self.store = store
        self._cooldown_sec = cooldown_sec

        # Subscribe to bar events
        self._bar_queue = event_bus.subscribe(EventType.BAR_1M)

        # Dedup: (strategy_id, symbol, direction) -> last emit monotonic time
        self._last_emit: dict[tuple[str, str, str], float] = {}

        # Stats
        self._bars_processed = 0
        self._signals_emitted = 0
        self._signals_deduped = 0
        self._strategy_errors = 0

        # Control
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the signal generation loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="signal_generator")
        logger.info(
            "SignalGenerator started: %d strategies registered",
            len(self.registry),
        )

    async def stop(self) -> None:
        """Stop the signal generation loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            "SignalGenerator stopped: %d bars processed, %d signals emitted",
            self._bars_processed,
            self._signals_emitted,
        )

    async def _run_loop(self) -> None:
        """Main loop: read bar events, run strategies, publish signals."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._bar_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            bar_data = event.data
            self._bars_processed += 1

            # Run all strategies on this bar
            await self._run_strategies(bar_data)

    async def _run_strategies(self, bar_data: dict) -> None:
        """Execute all registered strategies and publish resulting signals.

        Strategies are run in a thread executor so that CPU-bound scans
        do not block the event loop.
        """
        today = datetime.now(IST).date()
        strategies = self.registry.all()

        if not strategies:
            return

        loop = asyncio.get_running_loop()

        for strategy in strategies:
            try:
                # Run strategy scan in executor to avoid blocking
                signals = await loop.run_in_executor(
                    None,
                    strategy.scan,
                    today,
                    self.store,
                )

                for sig in signals:
                    if sig.direction == "flat":
                        continue  # flat signals are not emitted as events

                    # Dedup check
                    key = (sig.strategy_id, sig.symbol, sig.direction)
                    now = time.monotonic()
                    last = self._last_emit.get(key, 0.0)

                    if now - last < self._cooldown_sec:
                        self._signals_deduped += 1
                        continue

                    # Build enriched live signal
                    live_sig = LiveSignal(
                        signal=sig,
                        bar_minute=bar_data.get("minute", ""),
                        generated_at=now,
                        instrument_token=bar_data.get("instrument_token", 0),
                        bar_close=bar_data.get("close", 0.0),
                        bar_vpin=bar_data.get("vpin", float("nan")),
                        bar_entropy=bar_data.get("entropy", float("nan")),
                    )

                    # Publish signal event
                    await self.event_bus.publish(
                        EventType.SIGNAL,
                        {
                            "strategy_id": sig.strategy_id,
                            "symbol": sig.symbol,
                            "direction": sig.direction,
                            "conviction": sig.conviction,
                            "instrument_type": sig.instrument_type,
                            "strike": sig.strike,
                            "expiry": sig.expiry,
                            "ttl_bars": sig.ttl_bars,
                            "metadata": sig.metadata,
                            "bar_minute": live_sig.bar_minute,
                            "bar_close": live_sig.bar_close,
                            "bar_vpin": live_sig.bar_vpin,
                            "bar_entropy": live_sig.bar_entropy,
                        },
                        source="signal_generator",
                    )

                    self._last_emit[key] = now
                    self._signals_emitted += 1

                    logger.info(
                        "SIGNAL %s %s %s conv=%.2f bar=%s",
                        sig.strategy_id,
                        sig.symbol,
                        sig.direction,
                        sig.conviction,
                        bar_data.get("minute", "?"),
                    )

            except Exception as e:
                self._strategy_errors += 1
                logger.error(
                    "Strategy %s scan failed: %s",
                    strategy.strategy_id,
                    e,
                    exc_info=True,
                )

    def clear_cooldowns(self) -> None:
        """Reset all signal cooldown timers (e.g. on session restart)."""
        self._last_emit.clear()

    def stats(self) -> dict:
        """Return generator-level diagnostics."""
        return {
            "running": self._running,
            "bars_processed": self._bars_processed,
            "signals_emitted": self._signals_emitted,
            "signals_deduped": self._signals_deduped,
            "strategy_errors": self._strategy_errors,
            "strategies": self.registry.list_ids(),
            "active_cooldowns": len(self._last_emit),
        }
