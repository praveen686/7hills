"""MarketDataStore — DuckDB + hive-partitioned Parquet query layer.

Provides SQL-queryable access to converted market data:
  - nfo_1min: NSE F&O 1-minute OHLCV bars
  - bfo_1min: BSE F&O 1-minute OHLCV bars
  - ticks: tick-level LTP / volume / OI
  - instruments: daily instrument master (Zerodha tokens → symbols)

The store wraps a DuckDB in-memory connection with views pointing at
hive-partitioned parquet directories. No data is duplicated — DuckDB
reads parquet files directly with predicate pushdown.

Usage:
    from qlx.data.store import MarketDataStore

    store = MarketDataStore()
    chain = store.get_option_chain("NIFTY", date(2026, 2, 5), "2026-02-06")
    bars  = store.get_symbol_bars("NIFTY50", date(2026, 2, 5))
    ticks = store.get_ticks(256265, date(2026, 2, 5))
    df    = store.sql("SELECT * FROM nfo_1min WHERE name='NIFTY' AND date > '2026-02-05'")
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_ROOT = Path(
    os.environ.get(
        "QLX_DATA_ROOT",
        "/home/ubuntu/Desktop/7hills/qlx/qlxr/qlxr_common/qlxr_data",
    )
)

MARKET_DIR = _DATA_ROOT / "market"


class MarketDataStore:
    """DuckDB-backed query layer over hive-partitioned parquet files."""

    def __init__(self, market_dir: str | Path | None = None):
        self.market_dir = Path(market_dir) if market_dir else MARKET_DIR
        self._con = duckdb.connect()
        self._lock = threading.Lock()
        self._register_views()

    def _register_views(self) -> None:
        """Create DuckDB views pointing to parquet directories."""
        views = {
            "nfo_1min": self.market_dir / "nfo_1min" / "*" / "*.parquet",
            "bfo_1min": self.market_dir / "bfo_1min" / "*" / "*.parquet",
            "ticks": self.market_dir / "ticks" / "*" / "*.parquet",
            "instruments": self.market_dir / "instruments" / "*" / "*.parquet",
        }
        for name, glob_path in views.items():
            # Only create view if directory exists and has data
            cat_dir = self.market_dir / name
            if cat_dir.exists() and any(cat_dir.rglob("*.parquet")):
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet('{glob_path}', "
                    f"hive_partitioning=true, hive_types_autocast=false)"
                )
                logger.debug("Registered view: %s", name)
            else:
                logger.debug("Skipped view %s (no data)", name)

        # Auto-register any nse_* directories with parquet data
        if self.market_dir.exists():
            for cat_dir in sorted(self.market_dir.iterdir()):
                if (
                    cat_dir.is_dir()
                    and cat_dir.name.startswith("nse_")
                    and any(cat_dir.rglob("*.parquet"))
                ):
                    name = cat_dir.name
                    glob_path = cat_dir / "*" / "*.parquet"
                    self._con.execute(
                        f"CREATE OR REPLACE VIEW {name} AS "
                        f"SELECT * FROM read_parquet('{glob_path}', "
                        f"hive_partitioning=true, hive_types_autocast=false, "
                        f"union_by_name=true)"
                    )
                    logger.debug("Registered view: %s", name)

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Raw SQL access
    # ------------------------------------------------------------------

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""
        with self._lock:
            if params:
                return self._con.execute(query, params).fetchdf()
            return self._con.execute(query).fetchdf()

    def explain(self, query: str) -> str:
        """Return the query plan (useful for checking pushdown)."""
        with self._lock:
            return self._con.execute(f"EXPLAIN {query}").fetchone()[0]

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def available_dates(self, category: str = "nfo_1min") -> list[date]:
        """List dates that have data for a category."""
        cat_dir = self.market_dir / category
        if not cat_dir.exists():
            return []
        dates = []
        for d_dir in cat_dir.iterdir():
            if d_dir.is_dir() and d_dir.name.startswith("date="):
                try:
                    dates.append(date.fromisoformat(d_dir.name[5:]))
                except ValueError:
                    continue
        return sorted(dates)

    def get_option_chain(
        self,
        name: str,
        d: date,
        expiry: str | None = None,
    ) -> pd.DataFrame:
        """Get option chain (all strikes, CE+PE) for an underlying on a date.

        Parameters
        ----------
        name : str
            Underlying name (e.g. "NIFTY", "BANKNIFTY", "SENSEX").
        d : date
            Trading date.
        expiry : str, optional
            Expiry date as "YYYY-MM-DD". If None, uses nearest expiry.

        Returns
        -------
        pd.DataFrame with columns: date, strike, instrument_type, open, high, low, close, volume, oi, symbol
        """
        d_str = d.isoformat()

        # Determine which table (NFO vs BFO) based on underlying
        bfo_names = {"SENSEX", "BANKEX", "SENSEX50"}
        table = "bfo_1min" if name.upper() in bfo_names else "nfo_1min"

        with self._lock:
            if expiry is None:
                # Find nearest expiry
                expiry_q = self._con.execute(
                    f"SELECT DISTINCT expiry FROM {table} "
                    f"WHERE date={d_str!r} AND name={name!r} "
                    f"AND instrument_type IN ('CE', 'PE') "
                    f"ORDER BY expiry LIMIT 1"
                ).fetchone()
                if not expiry_q:
                    return pd.DataFrame()
                expiry = str(expiry_q[0])

            return self._con.execute(
                f"SELECT date, strike, instrument_type, open, high, low, close, volume, oi, symbol "
                f"FROM {table} "
                f"WHERE date={d_str!r} AND name={name!r} AND expiry={expiry!r} "
                f"AND instrument_type IN ('CE', 'PE') "
                f"ORDER BY date, strike, instrument_type"
            ).fetchdf()

    def get_symbol_bars(
        self,
        symbol: str,
        d: date,
        table: str = "nfo_1min",
    ) -> pd.DataFrame:
        """Get 1-minute OHLCV bars for a specific symbol on a date.

        Parameters
        ----------
        symbol : str
            Full trading symbol (e.g. "NIFTY26FEB23000CE").
        d : date
            Trading date.
        table : str
            Source table ("nfo_1min" or "bfo_1min").
        """
        d_str = d.isoformat()
        with self._lock:
            return self._con.execute(
                f"SELECT date, open, high, low, close, volume, oi "
                f"FROM {table} "
                f"WHERE date={d_str!r} AND symbol={symbol!r} "
                f"ORDER BY date"
            ).fetchdf()

    def get_ticks(
        self,
        instrument_token: int,
        d: date,
    ) -> pd.DataFrame:
        """Get tick-level data for an instrument on a date.

        Parameters
        ----------
        instrument_token : int
            Zerodha instrument token.
        d : date
            Trading date.
        """
        d_str = d.isoformat()
        with self._lock:
            return self._con.execute(
                f"SELECT timestamp, ltp, volume, oi "
                f"FROM ticks "
                f"WHERE date={d_str!r} AND instrument_token={instrument_token} "
                f"ORDER BY timestamp"
            ).fetchdf()

    def get_instruments(self, d: date) -> pd.DataFrame:
        """Get instrument master for a date."""
        d_str = d.isoformat()
        with self._lock:
            return self._con.execute(
                f"SELECT * FROM instruments WHERE date={d_str!r}"
            ).fetchdf()

    def resolve_token(
        self,
        tradingsymbol: str,
        d: date,
    ) -> int | None:
        """Look up instrument_token for a tradingsymbol on a date."""
        d_str = d.isoformat()
        with self._lock:
            row = self._con.execute(
                f"SELECT instrument_token FROM instruments "
                f"WHERE date={d_str!r} AND tradingsymbol={tradingsymbol!r} "
                f"LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    def get_underlyings(self, d: date, table: str = "nfo_1min") -> list[str]:
        """List unique underlying names available on a date."""
        d_str = d.isoformat()
        with self._lock:
            rows = self._con.execute(
                f"SELECT DISTINCT name FROM {table} WHERE date={d_str!r} ORDER BY name"
            ).fetchall()
            return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Ingestion hook
    # ------------------------------------------------------------------

    def ingest_day(self, d: date, force: bool = False) -> dict[str, int]:
        """Convert and ingest a single day's data into the store.

        This is the daily ingestion hook — call it after downloading
        new data from Telegram.
        """
        from qlx.data.convert import convert_day

        results = convert_day(d, force=force)
        if results:
            # Re-register views to pick up new partitions
            self._register_views()
            logger.info("Ingested %s: %s", d, results)
        return results

    def ingest_nse_day(self, d: date, force: bool = False) -> dict[str, int]:
        """Convert and ingest NSE daily files for a single date."""
        from qlx.data.nse_convert import convert_nse_day

        results = convert_nse_day(d, force=force)
        if results:
            self._register_views()
            logger.info("Ingested NSE %s: %s", d, results)
        return results

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary of available data."""
        info = {}
        # Core categories
        categories = ["nfo_1min", "bfo_1min", "ticks", "instruments"]
        # Auto-discover nse_* categories
        if self.market_dir.exists():
            for cat_dir in sorted(self.market_dir.iterdir()):
                if cat_dir.is_dir() and cat_dir.name.startswith("nse_"):
                    categories.append(cat_dir.name)

        for cat in categories:
            dates = self.available_dates(cat)
            if dates:
                cat_dir = self.market_dir / cat
                total_bytes = sum(f.stat().st_size for f in cat_dir.rglob("*.parquet"))
                info[cat] = {
                    "dates": len(dates),
                    "first": dates[0].isoformat(),
                    "last": dates[-1].isoformat(),
                    "size_mb": round(total_bytes / 1e6, 1),
                }
        return info
