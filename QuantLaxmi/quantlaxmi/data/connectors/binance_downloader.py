"""Binance historical klines downloader — Hive-partitioned Parquet storage.

Downloads klines for configured symbols and intervals, stores them
as Hive-partitioned Parquet files for efficient date-range queries.

Storage layout::

    common/data/binance/{symbol}/{interval}/date=YYYY-MM-DD/*.parquet

Usage::

    python -m data.connectors.binance_downloader \\
        --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
        --intervals 1m,5m,1h,1d \\
        --days 365

The downloader is idempotent: existing date partitions are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantlaxmi.data.connectors.binance_connector import (
    BinanceConnector,
    DEFAULT_SYMBOLS,
    INTERVALS,
    INTERVAL_MINUTES,
    MAX_KLINES_PER_REQUEST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default storage root (relative to this file)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "binance"

# Default intervals to download
DEFAULT_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Rate limit: sleep between API requests (seconds)
DEFAULT_RATE_LIMIT = 0.15


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class BinanceDownloader:
    """Downloads and stores Binance klines as Hive-partitioned Parquet.

    Parameters
    ----------
    symbols : list[str]
        Trading pairs to download (e.g. ``["BTCUSDT", "ETHUSDT"]``).
    intervals : list[str]
        Candle intervals (e.g. ``["1m", "1h", "1d"]``).
    data_root : Path or str
        Root directory for Parquet storage.
    connector : BinanceConnector, optional
        Reuse an existing connector instance.

    Usage::

        dl = BinanceDownloader(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            intervals=["1m", "5m", "1h", "1d"],
        )
        dl.download(days=365)
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        intervals: list[str] | None = None,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        connector: BinanceConnector | None = None,
    ):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.intervals = intervals or DEFAULT_INTERVALS
        self.data_root = Path(data_root)
        self.connector = connector or BinanceConnector()

        # Validate intervals
        for iv in self.intervals:
            if iv not in INTERVALS:
                raise ValueError(
                    f"Unsupported interval '{iv}'. "
                    f"Use one of: {list(INTERVALS)}"
                )

    def _partition_dir(
        self, symbol: str, interval: str, date_str: str,
    ) -> Path:
        """Build the Hive partition path for a given date."""
        return self.data_root / symbol / interval / f"date={date_str}"

    def _date_exists(
        self, symbol: str, interval: str, date_str: str,
    ) -> bool:
        """Check whether a date partition already has data."""
        part_dir = self._partition_dir(symbol, interval, date_str)
        if not part_dir.exists():
            return False
        parquet_files = list(part_dir.glob("*.parquet"))
        return len(parquet_files) > 0

    def _save_day(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        date_str: str,
    ) -> Path:
        """Save a single day's data as a Parquet file in the partition."""
        part_dir = self._partition_dir(symbol, interval, date_str)
        part_dir.mkdir(parents=True, exist_ok=True)

        out_file = part_dir / f"{symbol}_{interval}_{date_str}.parquet"

        # Reset index so timestamp becomes a column
        df_out = df.reset_index()

        # Convert timezone-aware timestamps to UTC then remove tz
        # (Parquet handles tz-naive more cleanly across readers)
        for col in df_out.select_dtypes(include=["datetimetz"]).columns:
            df_out[col] = df_out[col].dt.tz_convert("UTC").dt.tz_localize(None)

        table = pa.Table.from_pandas(df_out, preserve_index=False)
        pq.write_table(table, out_file, compression="snappy")

        return out_file

    def download_symbol_interval(
        self,
        symbol: str,
        interval: str,
        days: int = 30,
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ) -> dict:
        """Download klines for one symbol/interval pair.

        Parameters
        ----------
        symbol : str
            Trading pair.
        interval : str
            Candle interval.
        days : int
            Number of days of history to download.
        rate_limit : float
            Seconds to sleep between API requests.

        Returns
        -------
        dict
            Summary: ``{"symbol", "interval", "days_downloaded",
            "days_skipped", "total_rows", "errors"}``.
        """
        now = datetime.now(timezone.utc)
        stats = {
            "symbol": symbol,
            "interval": interval,
            "days_downloaded": 0,
            "days_skipped": 0,
            "total_rows": 0,
            "errors": [],
        }

        interval_mins = INTERVAL_MINUTES.get(interval, 60)
        # For intervals >= 1d, we fetch in larger chunks
        is_daily_or_above = interval_mins >= 1440

        logger.info(
            "Downloading %s %s — last %d days", symbol, interval, days,
        )

        if is_daily_or_above:
            # For daily+ intervals, fetch everything in one chunked call
            # and partition by date afterward
            try:
                df = self.connector.fetch_klines_chunked(
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    rate_limit_sleep=rate_limit,
                )
                if df.empty:
                    return stats

                # Group by date and save each partition
                df_with_date = df.copy()
                df_with_date["_date"] = df_with_date.index.strftime("%Y-%m-%d")

                for date_str, group in df_with_date.groupby("_date"):
                    if self._date_exists(symbol, interval, date_str):
                        stats["days_skipped"] += 1
                        continue
                    group = group.drop(columns=["_date"])
                    self._save_day(group, symbol, interval, date_str)
                    stats["days_downloaded"] += 1
                    stats["total_rows"] += len(group)

            except Exception as exc:
                msg = f"{symbol} {interval}: {exc}"
                logger.error(msg)
                stats["errors"].append(msg)

        else:
            # For intraday intervals, fetch day by day
            for day_offset in range(days, 0, -1):
                day_start = now - timedelta(days=day_offset)
                day_end = day_start + timedelta(days=1)
                date_str = day_start.strftime("%Y-%m-%d")

                # Idempotent: skip existing
                if self._date_exists(symbol, interval, date_str):
                    stats["days_skipped"] += 1
                    continue

                try:
                    df = self.connector.fetch_klines(
                        symbol=symbol,
                        interval=interval,
                        start_date=day_start,
                        end_date=day_end,
                        limit=MAX_KLINES_PER_REQUEST,
                    )
                    if df.empty:
                        continue

                    self._save_day(df, symbol, interval, date_str)
                    stats["days_downloaded"] += 1
                    stats["total_rows"] += len(df)

                except ValueError:
                    # No data for this day — not an error
                    pass
                except Exception as exc:
                    msg = f"{symbol} {interval} {date_str}: {exc}"
                    logger.warning(msg)
                    stats["errors"].append(msg)

                if rate_limit > 0:
                    time.sleep(rate_limit)

                # Progress log every 30 days
                if (days - day_offset) % 30 == 0 and day_offset != days:
                    logger.info(
                        "  %s %s: %d/%d days processed (%d rows)",
                        symbol, interval,
                        days - day_offset, days,
                        stats["total_rows"],
                    )

        logger.info(
            "  %s %s complete: %d downloaded, %d skipped, %d rows, %d errors",
            symbol, interval,
            stats["days_downloaded"],
            stats["days_skipped"],
            stats["total_rows"],
            len(stats["errors"]),
        )

        return stats

    def download(
        self,
        days: int = 30,
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ) -> list[dict]:
        """Download klines for all configured symbols and intervals.

        Parameters
        ----------
        days : int
            Number of days of history to download.
        rate_limit : float
            Seconds to sleep between API requests.

        Returns
        -------
        list[dict]
            Per-symbol/interval download summaries.
        """
        all_stats = []
        total_combos = len(self.symbols) * len(self.intervals)

        logger.info(
            "Starting download: %d symbols x %d intervals = %d combinations, "
            "%d days each",
            len(self.symbols), len(self.intervals), total_combos, days,
        )

        for i, symbol in enumerate(self.symbols):
            for j, interval in enumerate(self.intervals):
                combo_idx = i * len(self.intervals) + j + 1
                logger.info(
                    "[%d/%d] %s %s ...",
                    combo_idx, total_combos, symbol, interval,
                )

                stats = self.download_symbol_interval(
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    rate_limit=rate_limit,
                )
                all_stats.append(stats)

        # Print summary
        total_rows = sum(s["total_rows"] for s in all_stats)
        total_downloaded = sum(s["days_downloaded"] for s in all_stats)
        total_skipped = sum(s["days_skipped"] for s in all_stats)
        total_errors = sum(len(s["errors"]) for s in all_stats)

        logger.info(
            "Download complete: %d day-partitions downloaded, "
            "%d skipped, %d total rows, %d errors",
            total_downloaded, total_skipped, total_rows, total_errors,
        )

        return all_stats

    # ----- Backfill -----

    def backfill(
        self,
        days: int = 365,
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ) -> list[dict]:
        """Backfill mode: download up to 1 year of history.

        Convenience wrapper around :meth:`download` with a higher
        default day count.  Binance allows deep history for most
        spot pairs.
        """
        logger.info(
            "Starting backfill: %d days for %d symbols x %d intervals",
            days, len(self.symbols), len(self.intervals),
        )
        return self.download(days=days, rate_limit=rate_limit)

    # ----- Read back -----

    def load_parquet(
        self,
        symbol: str,
        interval: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Load previously downloaded data from Parquet partitions.

        Parameters
        ----------
        symbol : str
            Trading pair.
        interval : str
            Candle interval.
        start_date, end_date : str, optional
            Date range filter (``YYYY-MM-DD``).  If omitted,
            loads all available data.

        Returns
        -------
        pd.DataFrame
            Concatenated OHLCV data with UTC ``timestamp`` index.
        """
        base_dir = self.data_root / symbol / interval

        if not base_dir.exists():
            raise FileNotFoundError(
                f"No data directory: {base_dir}"
            )

        # Find all date partitions
        date_dirs = sorted(base_dir.glob("date=*"))
        if not date_dirs:
            raise FileNotFoundError(
                f"No date partitions in {base_dir}"
            )

        # Filter by date range
        if start_date or end_date:
            filtered = []
            for d in date_dirs:
                date_str = d.name.split("=")[1]
                if start_date and date_str < start_date:
                    continue
                if end_date and date_str > end_date:
                    continue
                filtered.append(d)
            date_dirs = filtered

        if not date_dirs:
            raise FileNotFoundError(
                f"No data for {symbol} {interval} in requested range"
            )

        all_dfs = []
        for d in date_dirs:
            for pf in d.glob("*.parquet"):
                df = pd.read_parquet(pf)
                all_dfs.append(df)

        if not all_dfs:
            raise FileNotFoundError(
                f"No parquet files found for {symbol} {interval}"
            )

        df = pd.concat(all_dfs, ignore_index=True)

        # Restore UTC datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="first")]

        return df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the Binance downloader."""
    parser = argparse.ArgumentParser(
        description="Download Binance klines to Hive-partitioned Parquet",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated trading pairs (default: BTCUSDT,ETHUSDT,SOLUSDT)",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default=",".join(DEFAULT_INTERVALS),
        help="Comma-separated intervals (default: 1m,5m,15m,1h,4h,1d)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to download (default: 30)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill mode: download --days of history (default 365 if not specified)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Data storage root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help="Seconds between API requests (default: 0.15)",
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    intervals = [i.strip() for i in args.intervals.split(",")]

    dl = BinanceDownloader(
        symbols=symbols,
        intervals=intervals,
        data_root=Path(args.data_root),
    )

    if args.backfill:
        days = args.days if args.days != 30 else 365
        stats = dl.backfill(days=days, rate_limit=args.rate_limit)
    else:
        stats = dl.download(days=args.days, rate_limit=args.rate_limit)

    # Print summary table
    print("\n" + "=" * 72)
    print("  Binance Klines Download Summary")
    print("=" * 72)
    print(f"  {'Symbol':<12} {'Interval':<10} {'Downloaded':<12} "
          f"{'Skipped':<10} {'Rows':<10} {'Errors':<8}")
    print("-" * 72)
    for s in stats:
        print(
            f"  {s['symbol']:<12} {s['interval']:<10} "
            f"{s['days_downloaded']:<12} {s['days_skipped']:<10} "
            f"{s['total_rows']:<10} {len(s['errors']):<8}"
        )
    print("=" * 72)

    total_rows = sum(s["total_rows"] for s in stats)
    total_errors = sum(len(s["errors"]) for s in stats)
    print(f"  Total rows: {total_rows:,}  |  Errors: {total_errors}")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
