"""Central live-mode orchestrator.

Wires all live components (EventBus, TickHandler, SignalGenerator,
RiskMonitor, StateManager, GDriveSync) into a single coordinated
engine that can be started/stopped from the CLI or FastAPI lifespan.

Usage::

    engine = LiveEngine(store, registry, state_file, mode="paper")
    await engine.start()
    # ... runs until stopped
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quantlaxmi.data._paths import DATA_ROOT, PORTFOLIO_STATE
from quantlaxmi.data.zerodha import (
    create_kite_session,
    load_zerodha_env,
    KiteTickFeed,
)
from quantlaxmi.engine.live.event_bus import EventBus, EventType
from quantlaxmi.engine.live.tick_handler import TickHandler
from quantlaxmi.engine.live.signal_generator import SignalGenerator
from quantlaxmi.engine.live.instruments import (
    resolve_all_tokens,
    resolve_index_tokens,
    token_to_symbol,
    TOKEN_SYMBOLS,
)
from quantlaxmi.engine.state import PortfolioState

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Directory for live bar Parquet exports
LIVE_BARS_DIR = DATA_ROOT / "live_bars"


class LiveEngine:
    """Coordinates all live-mode components.

    Parameters
    ----------
    store : MarketDataStore
        Shared data store for strategies.
    registry : StrategyRegistry
        Discovered strategy registry.
    state_file : str | Path
        Path to portfolio state JSON.
    mode : str
        ``"paper"`` for paper trading (default), ``"live"`` for real orders.
    """

    def __init__(
        self,
        store: Any,
        registry: Any,
        state_file: str | Path | None = None,
        mode: str = "paper",
    ):
        self.mode = mode
        self.store = store
        self.registry = registry
        self.state_file = Path(state_file) if state_file else PORTFOLIO_STATE

        # Core components
        self.event_bus = EventBus()
        self.tick_handler: TickHandler | None = None
        self.signal_generator: SignalGenerator | None = None
        self.risk_monitor: Any = None
        self.state_manager: Any = None
        self.gdrive_sync: Any = None

        # Kite feed
        self.kite: Any = None
        self.feed: KiteTickFeed | None = None
        self.token_map: dict[str, int] = {}

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self.running = False

        # Stats
        self._start_time: float = 0
        self._ticks_received: int = 0
        self._bars_completed: int = 0
        self._signals_emitted: int = 0

    async def start(self) -> None:
        """Start the live engine.

        1. Authenticate with Kite (via env credentials)
        2. Resolve instrument tokens
        3. Start KiteTickFeed
        4. Start TickHandler + SignalGenerator
        5. Start RiskMonitor + StateManager
        6. Start GDrive sync (if configured)
        7. Launch tick consumption + persist loops
        """
        self._start_time = time.monotonic()
        logger.info("LiveEngine starting (mode=%s)...", self.mode)

        # 1. Authenticate with Kite
        env = load_zerodha_env()
        access_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "")

        if access_token:
            self.kite = create_kite_session(access_token=access_token)
            logger.info("Kite session created from access token")
        else:
            # Try session cache
            from quantlaxmi.data._paths import ZERODHA_SESSION_CACHE
            if ZERODHA_SESSION_CACHE.exists():
                import json
                try:
                    cache = json.loads(ZERODHA_SESSION_CACHE.read_text())
                    access_token = cache.get("access_token", "")
                    if access_token:
                        self.kite = create_kite_session(access_token=access_token)
                        logger.info("Kite session restored from cache")
                except Exception as exc:
                    logger.warning("Session cache invalid: %s", exc)

            if not self.kite:
                logger.warning(
                    "No Kite access token. Set ZERODHA_ACCESS_TOKEN or "
                    "login via /api/auth/zerodha. Running without live feed."
                )

        # 2. Resolve instrument tokens
        if self.kite:
            try:
                self.token_map = resolve_all_tokens(self.kite)
            except Exception as exc:
                logger.warning("Token resolution failed, using index-only: %s", exc)
                self.token_map = resolve_index_tokens()
        else:
            self.token_map = resolve_index_tokens()

        tokens = list(self.token_map.values())
        logger.info("Subscribed tokens: %s", self.token_map)

        # 3. Initialize TickHandler
        db_conn = self.store._conn if hasattr(self.store, "_conn") else None
        self.tick_handler = TickHandler(self.event_bus, db_conn)

        # Register token → symbol mappings for risk monitor
        for sym, tok in self.token_map.items():
            TOKEN_SYMBOLS[tok] = sym

        # 4. Initialize SignalGenerator
        self.signal_generator = SignalGenerator(
            self.event_bus, self.registry, self.store,
        )

        # 5. Start RiskMonitor
        try:
            from quantlaxmi.engine.live.risk_monitor import RiskMonitor
            from quantlaxmi.core.risk.limits import RiskLimits
            state = PortfolioState.load(self.state_file)
            self.risk_monitor = RiskMonitor(
                event_bus=self.event_bus,
                state=state,
                limits=RiskLimits(),
            )
            for sym, tok in self.token_map.items():
                self.risk_monitor.register_token_mapping(tok, sym)
            await self.risk_monitor.start()
            logger.info("RiskMonitor started")
        except Exception as exc:
            logger.warning("RiskMonitor start failed (non-fatal): %s", exc)
            self.risk_monitor = None

        # 6. Start StateManager
        try:
            from quantlaxmi.engine.live.state_manager import StateManager
            self.state_manager = StateManager()
            await self.state_manager.start()
            self.state_manager.set_engine_running(True)
            logger.info("StateManager started")
        except Exception as exc:
            logger.warning("StateManager start failed (non-fatal): %s", exc)
            self.state_manager = None

        # 7. Start KiteTickFeed (only if authenticated)
        if self.kite and tokens:
            try:
                self.feed = KiteTickFeed(
                    api_key=env["api_key"],
                    access_token=self.kite.access_token,
                    tokens=tokens,
                    mode="full",
                )
                self.feed.start()
                logger.info("KiteTickFeed started (%d tokens)", len(tokens))
            except Exception as exc:
                logger.error("KiteTickFeed start failed: %s", exc)
                self.feed = None

        # 8. Start SignalGenerator
        await self.signal_generator.start()
        logger.info("SignalGenerator started")

        # 9. Launch background tasks
        if self.feed:
            self._tasks.append(asyncio.create_task(self._tick_loop()))

        self._tasks.append(asyncio.create_task(self._persist_loop()))
        self._tasks.append(asyncio.create_task(self._stats_loop()))

        # 10. Start GDrive sync
        try:
            from quantlaxmi.engine.live.gdrive_sync import GDriveSync
            self.gdrive_sync = GDriveSync()
            await self.gdrive_sync.start()
            logger.info("GDriveSync started")
        except Exception as exc:
            logger.warning("GDriveSync start failed (non-fatal): %s", exc)
            self.gdrive_sync = None

        # 11. Schedule post-market tasks
        self._tasks.append(asyncio.create_task(self._post_market_loop()))

        self.running = True
        logger.info(
            "LiveEngine started (mode=%s, tokens=%d, strategies=%d)",
            self.mode, len(self.token_map), len(self.registry),
        )

    async def _tick_loop(self) -> None:
        """Consume ticks from KiteTickFeed and dispatch to TickHandler."""
        logger.info("Tick loop started")
        try:
            async for tick in self.feed:
                self._ticks_received += 1
                await self.tick_handler.on_tick(tick)
        except asyncio.CancelledError:
            logger.info("Tick loop cancelled")
        except Exception as exc:
            logger.error("Tick loop error: %s", exc, exc_info=True)

    async def _persist_loop(self) -> None:
        """Every 5 minutes: flush DuckDB bars and export to Parquet."""
        logger.info("Persist loop started (interval=300s)")
        try:
            while True:
                await asyncio.sleep(300)
                if self.tick_handler:
                    await self.tick_handler.flush()
                    self._bars_completed = self.tick_handler.stats().get(
                        "bars_inserted", 0,
                    )
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._export_bars_to_parquet)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Persist loop error: %s", exc, exc_info=True)

    async def _stats_loop(self) -> None:
        """Log engine stats every 60 seconds."""
        try:
            while True:
                await asyncio.sleep(60)
                stats = self.stats()
                logger.info(
                    "Engine stats: ticks=%d bars=%d signals=%d uptime=%ds",
                    stats["ticks_received"],
                    stats["bars_completed"],
                    stats["signals_emitted"],
                    stats["uptime_seconds"],
                )
                if self.state_manager:
                    self.state_manager.update_engine_stats(stats)
        except asyncio.CancelledError:
            pass

    async def _post_market_loop(self) -> None:
        """Run post-market tasks after 15:35 IST daily.

        - Telegram data download
        - Data format conversion
        - Google Drive sync
        """
        try:
            while True:
                now = datetime.now(IST)
                # Target: 15:35 IST
                target = now.replace(hour=15, minute=35, second=0, microsecond=0)
                if now >= target:
                    # Already past today's target, schedule for tomorrow
                    target += timedelta(days=1)

                wait_seconds = (target - now).total_seconds()
                logger.info(
                    "Post-market task scheduled in %.0f seconds (%s IST)",
                    wait_seconds, target.strftime("%Y-%m-%d %H:%M"),
                )
                await asyncio.sleep(wait_seconds)

                # Execute post-market tasks
                await self._run_post_market_tasks()
        except asyncio.CancelledError:
            pass

    async def _run_post_market_tasks(self) -> None:
        """Execute post-market data collection."""
        logger.info("Running post-market tasks...")

        # 1. Telegram NFO data download
        try:
            import sys
            telegram_dir = Path(__file__).resolve().parents[3] / "telegram"
            sys.path.insert(0, str(telegram_dir.parent))
            from telegram.telegram_downloader import download_nfo_data
            await download_nfo_data(message_limit=50)
            logger.info("Telegram NFO data downloaded")
        except Exception as exc:
            logger.warning("Telegram download failed: %s", exc)

        # 2. Convert feather → Parquet (handled by download_nfo_data via _ingest_new_dates)
        try:
            from quantlaxmi.data.convert import convert_all, discover_sources, discover_converted
            sources = discover_sources()
            converted = discover_converted()
            all_source_dates = sorted(
                sources["nfo_feather"] | sources["bfo_feather"]
                | sources["tick_zip"] | sources["tick_pkl"]
                | sources["instrument_pkl"]
            )
            all_converted = (
                converted.get("nfo_1min", set()) & converted.get("bfo_1min", set())
            )
            new_dates = [d for d in all_source_dates if d not in all_converted]
            if new_dates:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, convert_all, new_dates)
                logger.info("Telegram data converted (%d dates)", len(new_dates))
        except Exception as exc:
            logger.warning("Telegram convert failed: %s", exc)

        # 3. Trigger GDrive sync
        if self.gdrive_sync:
            await self.gdrive_sync.sync_once()
            logger.info("Post-market GDrive sync completed")

    def _export_bars_to_parquet(self) -> None:
        """Export today's live bars from DuckDB to Parquet for archival."""
        try:
            today = datetime.now(IST).strftime("%Y-%m-%d")
            export_dir = LIVE_BARS_DIR / f"date={today}"
            export_dir.mkdir(parents=True, exist_ok=True)

            db_conn = self.store._conn if hasattr(self.store, "_conn") else None
            if db_conn is None:
                return

            df = db_conn.execute(
                "SELECT * FROM live_bars_1m WHERE minute LIKE ?",
                [f"{today}%"],
            ).fetchdf()

            if df.empty:
                return

            out_path = export_dir / "bars_1m.parquet"
            df.to_parquet(str(out_path), index=False)
            logger.info(
                "Exported %d bars to %s", len(df), out_path,
            )
        except Exception as exc:
            logger.debug("Bar export failed: %s", exc)

    async def stop(self) -> None:
        """Stop all live engine components gracefully."""
        logger.info("LiveEngine stopping...")
        self.running = False

        # Flush remaining bars
        if self.tick_handler:
            try:
                await self.tick_handler.flush()
            except Exception as exc:
                logger.warning("Flush on stop failed: %s", exc)

        # Export final bars
        self._export_bars_to_parquet()

        # Stop components
        if self.signal_generator:
            await self.signal_generator.stop()

        if self.risk_monitor:
            try:
                await self.risk_monitor.stop()
            except Exception as exc:
                logger.warning("RiskMonitor stop failed: %s", exc)

        if self.state_manager:
            try:
                self.state_manager.set_engine_running(False)
                await self.state_manager.stop()
            except Exception as exc:
                logger.warning("StateManager stop failed: %s", exc)

        if self.gdrive_sync:
            try:
                await self.gdrive_sync.stop()
            except Exception as exc:
                logger.warning("GDriveSync stop failed: %s", exc)

        if self.feed:
            try:
                self.feed.stop()
            except Exception as exc:
                logger.warning("KiteTickFeed stop failed: %s", exc)

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("LiveEngine stopped.")

    def stats(self) -> dict[str, Any]:
        """Return engine diagnostics."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0

        # Get signal count from generator
        sig_stats = (
            self.signal_generator.stats()
            if self.signal_generator
            else {}
        )
        self._signals_emitted = sig_stats.get("signals_emitted", 0)

        # Get tick handler stats
        th_stats = self.tick_handler.stats() if self.tick_handler else {}

        result = {
            "mode": self.mode,
            "running": self.running,
            "uptime_seconds": int(uptime),
            "ticks_received": self._ticks_received,
            "bars_completed": th_stats.get("bars_inserted", 0),
            "signals_emitted": self._signals_emitted,
            "tokens_subscribed": len(self.token_map),
            "strategies_registered": len(self.registry) if self.registry else 0,
            "feed_connected": self.feed is not None,
            "kite_authenticated": self.kite is not None,
            "event_bus": self.event_bus.stats(),
            "tick_handler": th_stats,
            "signal_generator": sig_stats,
        }

        if self.risk_monitor:
            try:
                result["risk_monitor"] = self.risk_monitor.stats()
            except Exception:
                pass

        return result
