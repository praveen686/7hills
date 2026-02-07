"""Depth collector — streams 5-level depth for futures + options to parquet.

Connects to Zerodha KiteTicker in MODE_FULL, receives depth ticks for
NIFTY/BANKNIFTY futures and near-ATM options, converts them to flat
DepthTick records, and writes to date-partitioned parquet via DepthStore.

Options handling:
- Subscribes to ±N strikes around ATM for nearest K weekly expiries
- Dynamically recenters strike window when spot moves ≥ 3 strike widths
- Automatically re-resolves tokens on weekly expiry rollover

Market hours guard: 9:15 AM - 3:30 PM IST, auto-exit after 3:35 PM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from collectors.auth import headless_login
from qlx.data.zerodha import KiteTickFeed, load_zerodha_env

from .storage import DepthStore, DepthStoreConfig, DepthTick
from .tokens import (
    FuturesToken,
    OptionToken,
    check_recenter_needed,
    fetch_spot_prices,
    resolve_futures_tokens,
    resolve_option_tokens,
    should_reroll,
)

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Market hours
MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30
EXIT_H, EXIT_M = 15, 35

# Recenter check interval
RECENTER_INTERVAL_SEC = 60.0


class DepthCollector:
    """Streams 5-level depth from KiteTicker into DepthStore.

    Handles both futures (2 tokens) and near-ATM options (~240 tokens)
    with dynamic ATM recentering.
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        flush_interval_sec: float = 60.0,
        flush_threshold: int = 10_000,
        indices: list[str] | None = None,
        n_strikes: int = 15,
        n_expiries: int = 2,
        duration: int | None = None,
        futures_only: bool = False,
    ):
        if base_dir is None:
            base_dir = Path("data/zerodha/5level")

        self.config = DepthStoreConfig(
            base_dir=base_dir,
            flush_interval_sec=flush_interval_sec,
            flush_threshold=flush_threshold,
        )
        self.indices = indices or ["NIFTY", "BANKNIFTY"]
        self.n_strikes = n_strikes
        self.n_expiries = n_expiries
        self.duration = duration
        self.futures_only = futures_only

        self._store: DepthStore | None = None
        self._feed: KiteTickFeed | None = None
        self._futures_tokens: dict[int, FuturesToken] = {}
        self._option_tokens: dict[int, OptionToken] = {}
        self._instruments_df: pd.DataFrame | None = None
        self._spot_prices: dict[str, float] = {}
        self._recenter_centers: dict[str, float] = {}
        self._running = True
        self._tick_count = 0
        self._last_tick_time: float = 0.0
        self._last_status_time: float = 0.0
        self._last_recenter_time: float = 0.0

    @property
    def _all_token_ids(self) -> list[int]:
        return list(self._futures_tokens.keys()) + list(self._option_tokens.keys())

    def _token_lookup(self, token_id: int) -> FuturesToken | OptionToken | None:
        return self._futures_tokens.get(token_id) or self._option_tokens.get(token_id)

    async def run(self) -> None:
        """Main collection loop."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        # Auth
        logger.info("Authenticating with Zerodha...")
        kite = headless_login()
        env = load_zerodha_env()

        # Cache instrument dump (used for options resolution + recentering)
        logger.info("Fetching NFO instrument dump...")
        self._instruments_df = pd.DataFrame(kite.instruments("NFO"))
        logger.info("Loaded %d NFO instruments", len(self._instruments_df))

        # Resolve futures tokens
        self._futures_tokens = resolve_futures_tokens(kite, self.indices)
        if not self._futures_tokens:
            logger.error("No futures tokens resolved, exiting")
            return

        if should_reroll(self._futures_tokens):
            logger.info("Futures expiry crossed, re-resolving...")
            self._futures_tokens = resolve_futures_tokens(kite, self.indices)

        # Resolve option tokens (unless futures-only mode)
        if not self.futures_only:
            self._spot_prices = fetch_spot_prices(kite, self.indices)
            logger.info("Spot prices: %s", self._spot_prices)

            for name in self.indices:
                spot = self._spot_prices.get(name)
                if spot is None:
                    logger.warning("No spot price for %s, skipping options", name)
                    continue
                opts = resolve_option_tokens(
                    self._instruments_df, name, spot,
                    n_strikes=self.n_strikes, n_expiries=self.n_expiries,
                )
                self._option_tokens.update(opts)
                self._recenter_centers[name] = spot

        # Init store
        self._store = DepthStore(self.config)

        # Start feed with all tokens
        all_tokens = self._all_token_ids
        self._feed = KiteTickFeed(
            api_key=env["api_key"],
            access_token=kite.access_token,
            tokens=all_tokens,
            mode="full",
        )

        n_fut = len(self._futures_tokens)
        n_opt = len(self._option_tokens)
        logger.info(
            "Starting depth feed: %d futures + %d options = %d total tokens",
            n_fut, n_opt, len(all_tokens),
        )
        self._feed.start()

        start_time = time.monotonic()
        self._last_status_time = start_time
        self._last_tick_time = start_time
        self._last_recenter_time = start_time

        try:
            while self._running:
                # Use timeout so exit checks run even when no ticks arrive
                try:
                    tick = await asyncio.wait_for(self._feed.next(), timeout=5.0)
                except asyncio.TimeoutError:
                    # No tick in 5s — check exit conditions
                    now_ist = datetime.now(IST)
                    if self._past_exit_time(now_ist):
                        logger.info("Past exit time (%02d:%02d IST), stopping",
                                    EXIT_H, EXIT_M)
                        break
                    if self.duration and (time.monotonic() - start_time) >= self.duration:
                        logger.info("Duration limit (%ds) reached", self.duration)
                        break
                    continue

                # Duration limit
                if self.duration and (time.monotonic() - start_time) >= self.duration:
                    logger.info("Duration limit (%ds) reached", self.duration)
                    break

                # Market hours check
                now_ist = datetime.now(IST)
                if self._past_exit_time(now_ist):
                    logger.info("Past exit time (%02d:%02d IST), stopping",
                                EXIT_H, EXIT_M)
                    break

                # Lookup token
                token_info = self._token_lookup(tick.instrument_token)
                if token_info is None:
                    continue

                # Build DepthTick with instrument metadata
                if isinstance(token_info, FuturesToken):
                    depth_tick = DepthTick.from_kite_tick(
                        tick, token_info.storage_key,
                        strike=0.0,
                        expiry=str(token_info.expiry),
                        option_type="FUT",
                    )
                    # Track spot price from futures LTP
                    self._spot_prices[token_info.index_name] = tick.last_price
                else:
                    depth_tick = DepthTick.from_kite_tick(
                        tick, token_info.storage_key,
                        strike=token_info.strike,
                        expiry=str(token_info.expiry),
                        option_type=token_info.option_type,
                    )

                self._store.add_tick(token_info.storage_key, depth_tick)
                self._tick_count += 1

                now = time.monotonic()

                # Gap detection (only for futures — options can have natural gaps)
                if isinstance(token_info, FuturesToken):
                    if self._last_tick_time and (now - self._last_tick_time) > 5.0:
                        gap = now - self._last_tick_time
                        logger.warning("Futures tick gap: %.1fs", gap)
                    self._last_tick_time = now

                # Periodic flush
                self._store.maybe_flush()

                # Recenter check
                if (not self.futures_only
                        and now - self._last_recenter_time >= RECENTER_INTERVAL_SEC):
                    self._maybe_recenter()
                    self._last_recenter_time = now

                # Status log every 60s
                if now - self._last_status_time >= 60.0:
                    self._log_status()
                    self._last_status_time = now

        except asyncio.CancelledError:
            logger.info("Collection cancelled")
        finally:
            self._cleanup()

    def _maybe_recenter(self) -> None:
        """Check if ATM has shifted enough to warrant re-subscribing options."""
        if self._instruments_df is None or self._feed is None:
            return

        indices_to_recenter = check_recenter_needed(
            self._option_tokens, self._spot_prices, self._recenter_centers,
        )

        for name in indices_to_recenter:
            spot = self._spot_prices[name]
            logger.info(
                "Recentering %s: spot=%.1f, was=%.1f",
                name, spot, self._recenter_centers[name],
            )

            # Resolve new option tokens
            new_opts = resolve_option_tokens(
                self._instruments_df, name, spot,
                n_strikes=self.n_strikes, n_expiries=self.n_expiries,
            )

            # Compute diff
            old_ids = {
                t for t, o in self._option_tokens.items()
                if o.index_name == name
            }
            new_ids = set(new_opts.keys())

            to_unsub = old_ids - new_ids
            to_sub = new_ids - old_ids

            if to_unsub:
                self._feed.unsubscribe(list(to_unsub))
            if to_sub:
                self._feed.subscribe(list(to_sub))

            # Update maps
            for t in to_unsub:
                self._option_tokens.pop(t, None)
            self._option_tokens.update(new_opts)
            self._recenter_centers[name] = spot

            logger.info(
                "Recentered %s: -%d unsub, +%d sub, %d total opt tokens",
                name, len(to_unsub), len(to_sub), len(self._option_tokens),
            )

    def _past_exit_time(self, now: datetime) -> bool:
        exit_time = now.replace(
            hour=EXIT_H, minute=EXIT_M, second=0, microsecond=0
        )
        return now >= exit_time

    def _shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False

    def _log_status(self) -> None:
        stats = self._store.stats() if self._store else {}
        logger.info(
            "Status: %d ticks received, %d stored | fut=%d opt=%d tokens | spots=%s | files=%s",
            self._tick_count,
            stats.get("total_stored", 0),
            len(self._futures_tokens),
            len(self._option_tokens),
            {k: f"{v:.1f}" for k, v in self._spot_prices.items()},
            {k: v.get("stored", 0) for k, v in stats.get("per_symbol", {}).items()},
        )

    def _cleanup(self) -> None:
        logger.info("Cleaning up...")
        if self._store:
            flushed = self._store.flush_all()
            if flushed:
                logger.info("Final flush: %s", flushed)
            self._store.close()
        if self._feed:
            self._feed.stop()
        logger.info(
            "Collection complete: %d ticks total (%d fut tokens, %d opt tokens)",
            self._tick_count, len(self._futures_tokens), len(self._option_tokens),
        )
