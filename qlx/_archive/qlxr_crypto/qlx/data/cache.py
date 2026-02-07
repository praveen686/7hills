"""Local parquet cache for historical kline data.

Downloads from Binance REST, stores as parquet, supports incremental
updates.  One file per symbol-interval pair under ``data/``.

Usage::

    cache = KlineCache("data/klines")
    ohlcv = cache.get("BTCUSDT", "1h", start="2024-01-01")
    # First call downloads; subsequent calls read from parquet + fetch only new bars.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from qlx.core.types import OHLCV
from qlx.data.binance import fetch_klines, fetch_24h_volumes

logger = logging.getLogger(__name__)


class KlineCache:
    """Parquet-backed kline cache with incremental updates."""

    def __init__(self, cache_dir: str | Path = "data/klines"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str, market: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_{interval}_{market}.parquet"

    def get(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | None = "2024-01-01",
        end: str | None = None,
        market: str = "spot",
    ) -> OHLCV:
        """Get OHLCV data, using cache when available.

        If cached data exists, only fetches bars newer than the last
        cached bar.  If no cache exists, downloads from *start*.
        """
        path = self._path(symbol, interval, market)

        cached_df = None
        if path.exists():
            cached_df = pd.read_parquet(path)
            cached_df.index = pd.to_datetime(cached_df.index, utc=True)

            # Incremental: fetch from last cached bar onward
            last_ts = cached_df.index[-1]
            logger.info(
                "%s %s: cache has %d bars up to %s, fetching new...",
                symbol, interval, len(cached_df), last_ts,
            )
            try:
                new = fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start=last_ts.isoformat(),
                    end=end,
                    limit=1000,
                    market=market,
                )
                new_df = new.df
                # Merge: keep cached (complete) bars over new (possibly incomplete) ones
                combined = pd.concat([cached_df, new_df])
                combined = combined[~combined.index.duplicated(keep="first")]
                combined = combined.sort_index()
            except Exception as e:
                logger.warning("Incremental fetch failed (%s), using cache only", e)
                combined = cached_df
        else:
            logger.info("%s %s: no cache, downloading from %s...", symbol, interval, start)
            ohlcv = fetch_klines(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
                limit=1000,
                market=market,
            )
            combined = ohlcv.df

        # Apply date filter
        if start:
            start_ts = pd.Timestamp(start, tz="UTC")
            combined = combined[combined.index >= start_ts]
        if end:
            end_ts = pd.Timestamp(end, tz="UTC")
            combined = combined[combined.index <= end_ts]

        # Save to parquet
        combined.to_parquet(path)
        logger.info(
            "%s %s: cached %d bars (%s → %s)",
            symbol, interval, len(combined),
            combined.index[0], combined.index[-1],
        )

        return OHLCV(combined[["Open", "High", "Low", "Close", "Volume"]])

    def get_universe(
        self,
        symbols: list[str],
        interval: str = "1h",
        start: str | None = "2024-01-01",
        market: str = "spot",
    ) -> dict[str, OHLCV]:
        """Download/cache multiple symbols. Returns dict of symbol → OHLCV."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get(sym, interval, start=start, market=market)
                time.sleep(0.2)  # Rate limit courtesy
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", sym, e)
        return result

    def list_cached(self) -> list[str]:
        """List all cached symbol-interval pairs."""
        return [p.stem for p in self.cache_dir.glob("*.parquet")]


def top_liquid_symbols(n: int = 20, min_volume_usd: float = 50_000_000) -> list[str]:
    """Get top N liquid USDT perpetual symbols by 24h volume.

    Filters to >$50M daily volume by default.
    """
    volumes = fetch_24h_volumes()
    usdt_pairs = {
        sym: vol for sym, vol in volumes.items()
        if sym.endswith("USDT") and vol >= min_volume_usd
    }
    sorted_pairs = sorted(usdt_pairs.items(), key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in sorted_pairs[:n]]
