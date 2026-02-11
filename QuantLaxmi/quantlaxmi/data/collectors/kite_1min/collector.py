"""Kite Historical Data Collector — Futures + Spot, Minute + Daily.

Downloads OHLCV+OI candles from Zerodha Kite Connect API for NSE indices.

KITE API LIMITATIONS (verified 2026-02-08, docs: kite.trade/docs/connect/v3/historical):
  - continuous=True ONLY works with interval="day" (docs: "day candle records")
  - Minute data for expired contracts returns 0 bars (Kite purges after expiry)
  - Minute data limited to ~60 days per API request

DOWNLOAD STRATEGY:
  1. Continuous DAILY futures: continuous=True + day interval → 2 years of history
  2. Active contract MINUTE data: all currently-listed FUT contracts (Feb/Mar/Apr)
  3. Spot index MINUTE data: index tokens → 2 years of 1-min history

Usage:
    python -m data.collectors.kite_1min.collector                    # Everything
    python -m data.collectors.kite_1min.collector --symbols NIFTY    # Just NIFTY
    python -m data.collectors.kite_1min.collector --no-spot          # Futures only
    python -m data.collectors.kite_1min.collector --no-futures       # Spot only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

from quantlaxmi.data._paths import KITE_1MIN_DIR


# ---------------------------------------------------------------------------
# Instrument Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentSpec:
    """Specification for a downloadable instrument."""
    symbol: str          # e.g. "NIFTY"
    token: int           # Kite instrument_token
    label: str           # Human-readable label
    interval: str        # Kite interval: "minute" or "day"
    continuous: bool     # continuous flag for API call
    storage_key: str     # Directory name for Hive partitioning


# Index (spot) tokens — return underlying index level, no OI
_INDEX_TOKENS: dict[str, int] = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "FINNIFTY": 257801,
    "MIDCPNIFTY": 288009,
}

# Kite API constraints
CHUNK_DAYS_MINUTE = 58   # Max days per request for minute data (~60 day Kite limit)
CHUNK_DAYS_DAILY = 365   # Max days per request for daily data
MAX_HISTORY_DAYS = 730   # Kite allows ~2 years of historical data
API_PAUSE_SEC = 0.35     # Rate limit: ~3 requests/sec on Kite Connect
IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Parquet Schemas
# ---------------------------------------------------------------------------

_SCHEMA_MINUTE = pa.schema([
    ("timestamp", pa.timestamp("us", tz="Asia/Kolkata")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("oi", pa.int64()),
])

_SCHEMA_DAILY = pa.schema([
    ("timestamp", pa.timestamp("us", tz="Asia/Kolkata")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("oi", pa.int64()),
])


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

@dataclass
class CollectStats:
    """Aggregated statistics for a collection run."""
    total_bars: int = 0
    dates_downloaded: int = 0
    dates_skipped: int = 0
    chunks_fetched: int = 0
    errors: list[str] = field(default_factory=list)


class Kite1MinCollector:
    """Downloads OHLCV+OI candles and stores as Hive-partitioned Parquet.

    Parameters
    ----------
    base_dir : Path, optional
        Root output directory. Default: ``common/data/kite_1min/``
    symbols : list[str], optional
        Which indices to download. Default: all four.
    days : int, optional
        Number of calendar days to look back. Default: 730 (max).
    include_futures : bool
        Whether to download futures data. Default: True.
    include_spot : bool
        Whether to download spot index data. Default: True.
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        symbols: list[str] | None = None,
        days: int = MAX_HISTORY_DAYS,
        include_futures: bool = True,
        include_spot: bool = True,
    ):
        if base_dir is None:
            base_dir = KITE_1MIN_DIR
        self.base_dir = Path(base_dir)
        self.symbols = [s.upper() for s in (symbols or list(_INDEX_TOKENS.keys()))]
        self.days = min(days, MAX_HISTORY_DAYS)
        self.include_futures = include_futures
        self.include_spot = include_spot
        self._kite = None
        self._nfo_instruments: pd.DataFrame | None = None

    def _authenticate(self):
        """Authenticate with Kite API via headless login."""
        from quantlaxmi.data.collectors.auth import headless_login
        print("[AUTH] Authenticating with Zerodha Kite...")
        self._kite = headless_login()
        print("[AUTH] Authentication successful")
        return self._kite

    def _fetch_nfo_instruments(self) -> pd.DataFrame:
        """Fetch and cache the NFO instruments list."""
        if self._nfo_instruments is not None:
            return self._nfo_instruments
        print("[INFO] Fetching NFO instruments list...")
        self._nfo_instruments = pd.DataFrame(self._kite.instruments("NFO"))
        n_futs = len(self._nfo_instruments[self._nfo_instruments["instrument_type"] == "FUT"])
        print(f"[INFO] Got {len(self._nfo_instruments)} instruments ({n_futs} futures)")
        return self._nfo_instruments

    def _build_instrument_list(self) -> list[InstrumentSpec]:
        """Build the full list of instruments to download.

        For each symbol:
          - Spot index: 1-min bars for 2 years (token from _INDEX_TOKENS)
          - Continuous daily futures: day bars with continuous=True for 2 years
          - Active contract minute: 1-min bars for each currently-listed FUT contract
        """
        specs: list[InstrumentSpec] = []

        if self.include_futures:
            instruments = self._fetch_nfo_instruments()

        for sym in self.symbols:
            # --- Spot index: minute bars, 2 years ---
            if self.include_spot and sym in _INDEX_TOKENS:
                specs.append(InstrumentSpec(
                    symbol=sym,
                    token=_INDEX_TOKENS[sym],
                    label=f"{sym} spot index (1-min)",
                    interval="minute",
                    continuous=False,
                    storage_key=f"{sym}_SPOT",
                ))

            if not self.include_futures:
                continue

            # --- Continuous daily futures: day bars, 2 years ---
            # Use front-month token with continuous=True
            fut_mask = (
                (instruments["name"] == sym)
                & (instruments["instrument_type"] == "FUT")
                & (instruments["segment"] == "NFO-FUT")
            )
            fut_contracts = instruments[fut_mask].sort_values("expiry")

            if fut_contracts.empty:
                print(f"[WARN] No FUT contracts found for {sym}")
                continue

            front_token = int(fut_contracts.iloc[0]["instrument_token"])
            front_symbol = fut_contracts.iloc[0]["tradingsymbol"]

            specs.append(InstrumentSpec(
                symbol=sym,
                token=front_token,
                label=f"{sym} continuous daily ({front_symbol}, continuous=True)",
                interval="day",
                continuous=True,
                storage_key=f"{sym}_FUT_DAILY",
            ))

            # --- Active contract minute: 1-min bars for each listed contract ---
            for _, row in fut_contracts.iterrows():
                contract_token = int(row["instrument_token"])
                contract_symbol = row["tradingsymbol"]
                expiry = row["expiry"]
                specs.append(InstrumentSpec(
                    symbol=sym,
                    token=contract_token,
                    label=f"{contract_symbol} (1-min, expiry={expiry})",
                    interval="minute",
                    continuous=False,
                    storage_key=f"{sym}_FUT_{contract_symbol}",
                ))

        return specs

    def _existing_dates(self, storage_key: str) -> set[str]:
        """Return set of date strings (YYYY-MM-DD) already on disk for a symbol."""
        sym_dir = self.base_dir / storage_key
        if not sym_dir.exists():
            return set()
        dates: set[str] = set()
        for d in sym_dir.iterdir():
            if d.is_dir() and d.name.startswith("date="):
                date_str = d.name.split("=", 1)[1]
                parquet_files = list(d.glob("*.parquet"))
                if parquet_files and any(f.stat().st_size > 0 for f in parquet_files):
                    dates.add(date_str)
        return dates

    def _fetch_chunk_raw(
        self,
        instrument: InstrumentSpec,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        """Fetch a single chunk of data from Kite, returning raw DataFrame."""
        if self._kite is None:
            raise RuntimeError("Not authenticated. Call _authenticate() first.")

        try:
            data = self._kite.historical_data(
                instrument_token=instrument.token,
                from_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                interval=instrument.interval,
                continuous=instrument.continuous,
                oi=True,
            )
        except Exception as e:
            logger.debug("Fetch failed for %s (%s -> %s): %s",
                         instrument.label, start, end, e)
            return None

        if not data:
            return None

        return pd.DataFrame(data)

    def _write_day_partition(
        self,
        storage_key: str,
        date_str: str,
        day_df: pd.DataFrame,
    ) -> int:
        """Write a single day's data as a Hive-partitioned Parquet file."""
        partition_dir = self.base_dir / storage_key / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        out_path = partition_dir / f"{storage_key}.parquet"

        write_df = day_df[
            ["timestamp", "open", "high", "low", "close", "volume", "oi"]
        ].copy()
        write_df = write_df.sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(
            write_df, schema=_SCHEMA_MINUTE, preserve_index=False
        )
        pq.write_table(table, out_path, compression="zstd")
        return len(write_df)

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names, types, and timezone from Kite response."""
        df = df.rename(columns={"date": "timestamp"})

        if "oi" not in df.columns:
            df["oi"] = 0

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

        df["volume"] = df["volume"].fillna(0).astype("int64")
        df["oi"] = df["oi"].fillna(0).astype("int64")
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype("float64")

        df["_date"] = df["timestamp"].dt.date
        return df

    def _fetch_instrument(
        self, instrument: InstrumentSpec, stats: CollectStats
    ) -> None:
        """Fetch all available data for a single instrument, chunk by chunk."""
        storage_key = instrument.storage_key
        existing = self._existing_dates(storage_key)

        chunk_days = CHUNK_DAYS_DAILY if instrument.interval == "day" else CHUNK_DAYS_MINUTE
        end_dt = datetime.now(IST)
        start_dt = end_dt - timedelta(days=self.days)

        print(f"\n{'='*60}")
        print(f"  {instrument.label}")
        print(f"  token={instrument.token}  interval={instrument.interval}  "
              f"continuous={instrument.continuous}")
        print(f"  Range: {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d} ({self.days} days)")
        print(f"  Existing dates on disk: {len(existing)}")
        print(f"  Output: {self.base_dir / storage_key}/")
        print(f"{'='*60}")

        cursor = start_dt
        chunk_num = 0

        while cursor < end_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
            chunk_num += 1

            print(
                f"  [{instrument.symbol}] Chunk {chunk_num}: "
                f"{cursor:%Y-%m-%d} -> {chunk_end:%Y-%m-%d} ... ",
                end="", flush=True,
            )

            df = self._fetch_chunk_raw(instrument, cursor, chunk_end)
            stats.chunks_fetched += 1

            if df is None or df.empty:
                print("empty (no data)")
                cursor = chunk_end + timedelta(days=1)
                time.sleep(API_PAUSE_SEC)
                continue

            df = self._normalize_df(df)

            bars_in_chunk = 0
            dates_new = 0
            dates_skip = 0

            for dt, day_df in df.groupby("_date"):
                date_str = dt.isoformat()
                if date_str in existing:
                    dates_skip += 1
                    stats.dates_skipped += 1
                    continue

                n_bars = self._write_day_partition(storage_key, date_str, day_df)
                existing.add(date_str)
                bars_in_chunk += n_bars
                dates_new += 1
                stats.dates_downloaded += 1
                stats.total_bars += n_bars

            print(f"{bars_in_chunk} bars, {dates_new} new dates, {dates_skip} skipped")
            cursor = chunk_end + timedelta(days=1)
            time.sleep(API_PAUSE_SEC)

    def run(self) -> CollectStats:
        """Execute the full collection pipeline."""
        stats = CollectStats()

        # Step 1: Auth
        self._authenticate()

        # Step 2: Build instrument list (fetches NFO instruments internally)
        instruments = self._build_instrument_list()

        if not instruments:
            print("[WARN] No instruments to download.")
            return stats

        print(f"\n[INFO] Will download {len(instruments)} instrument(s):")
        for inst in instruments:
            print(f"  - {inst.label} → {inst.storage_key}/")
        print(f"[INFO] Lookback: {self.days} calendar days")
        print(f"[INFO] Output: {self.base_dir}")

        # Step 3: Fetch each instrument
        for inst in instruments:
            try:
                self._fetch_instrument(inst, stats)
            except Exception as e:
                msg = f"Failed to fetch {inst.label}: {e}"
                print(f"[ERROR] {msg}")
                stats.errors.append(msg)
                logger.exception(msg)

        # Summary
        print(f"\n{'='*60}")
        print("  COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total bars downloaded:  {stats.total_bars:,}")
        print(f"  New dates written:      {stats.dates_downloaded}")
        print(f"  Dates skipped (exist):  {stats.dates_skipped}")
        print(f"  API chunks fetched:     {stats.chunks_fetched}")
        if stats.errors:
            print(f"  Errors:                 {len(stats.errors)}")
            for err in stats.errors:
                print(f"    - {err}")
        print(f"{'='*60}\n")

        return stats


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OHLCV+OI from Zerodha Kite for NSE index futures and spot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m data.collectors.kite_1min.collector                          # Everything
  python -m data.collectors.kite_1min.collector --symbols NIFTY,BANKNIFTY --days 60
  python -m data.collectors.kite_1min.collector --no-spot                # Futures only
  python -m data.collectors.kite_1min.collector --no-futures             # Spot only
        """,
    )
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY)")
    parser.add_argument("--days", type=int, default=MAX_HISTORY_DAYS,
                        help=f"Calendar days lookback (default/max: {MAX_HISTORY_DAYS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: common/data/kite_1min/)")
    parser.add_argument("--no-futures", action="store_true", help="Skip futures data")
    parser.add_argument("--no-spot", action="store_true", help="Skip spot index data")

    args = parser.parse_args()
    symbols = args.symbols.split(",") if args.symbols else None
    base_dir = Path(args.output) if args.output else None

    collector = Kite1MinCollector(
        base_dir=base_dir,
        symbols=symbols,
        days=args.days,
        include_futures=not args.no_futures,
        include_spot=not args.no_spot,
    )
    collector.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
