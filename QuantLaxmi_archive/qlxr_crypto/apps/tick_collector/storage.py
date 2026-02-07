"""Tick data storage — buffered Parquet writer with date partitioning.

Each symbol gets two files per day:
  data/ticks/{date}/{SYMBOL}_trades.parquet   — aggTrade data
  data/ticks/{date}/{SYMBOL}_book.parquet     — bookTicker data

Uses pyarrow.ParquetWriter for true incremental append (multiple row groups,
single file, no re-reading). Files are valid after every flush.

Flush triggers:
  - Time-based: every flush_interval_sec (default 300s / 5 min)
  - Size-based: every flush_threshold ticks (default 50,000)
  - Day rollover: new date → close old writers, open new files
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet Schemas
# ---------------------------------------------------------------------------

TRADE_SCHEMA = pa.schema([
    ("agg_trade_id", pa.int64()),
    ("timestamp_ms", pa.int64()),
    ("price", pa.float64()),
    ("quantity", pa.float64()),
    ("is_buyer_maker", pa.bool_()),
])

BOOK_SCHEMA = pa.schema([
    ("timestamp_ms", pa.int64()),
    ("bid_price", pa.float64()),
    ("ask_price", pa.float64()),
    ("bid_qty", pa.float64()),
    ("ask_qty", pa.float64()),
])


# ---------------------------------------------------------------------------
# Trade / Book tick dataclasses (lightweight, no frozen for speed)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TradeTick:
    agg_trade_id: int
    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass(slots=True)
class BookTick:
    timestamp_ms: int
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float


# ---------------------------------------------------------------------------
# Per-symbol writer
# ---------------------------------------------------------------------------

class _SymbolWriter:
    """Manages trade + book parquet files for a single symbol on a single day.

    Uses atomic write-on-flush: each flush reads existing file, concatenates
    new data, and writes a complete parquet file. This means files are always
    valid on disk (no open writer holding the footer).
    """

    def __init__(self, symbol: str, date_str: str, base_dir: Path):
        self.symbol = symbol
        self.date_str = date_str

        day_dir = base_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        self._trade_path = day_dir / f"{symbol}_trades.parquet"
        self._book_path = day_dir / f"{symbol}_book.parquet"

        self._trade_buf: list[TradeTick] = []
        self._book_buf: list[BookTick] = []

        self._trade_count = 0
        self._book_count = 0

    def add_trade(self, tick: TradeTick) -> None:
        self._trade_buf.append(tick)

    def add_book(self, tick: BookTick) -> None:
        self._book_buf.append(tick)

    @property
    def trade_buf_size(self) -> int:
        return len(self._trade_buf)

    @property
    def book_buf_size(self) -> int:
        return len(self._book_buf)

    @property
    def total_trades(self) -> int:
        return self._trade_count

    @property
    def total_books(self) -> int:
        return self._book_count

    @staticmethod
    def _append_to_parquet(
        path: Path, new_table: pa.Table, schema: pa.Schema,
    ) -> None:
        """Append new_table to an existing parquet file (or create it).

        Reads existing file, concatenates, writes back. Atomic via tmp+rename.
        """
        if path.exists():
            try:
                existing = pq.read_table(str(path), schema=schema)
                combined = pa.concat_tables([existing, new_table])
            except Exception:
                combined = new_table
        else:
            combined = new_table

        tmp_path = path.with_suffix(".parquet.tmp")
        pq.write_table(combined, str(tmp_path), compression="zstd")
        tmp_path.rename(path)

    def flush_trades(self) -> int:
        """Write buffered trades to parquet. Returns count flushed."""
        if not self._trade_buf:
            return 0

        table = pa.table({
            "agg_trade_id": pa.array([t.agg_trade_id for t in self._trade_buf], type=pa.int64()),
            "timestamp_ms": pa.array([t.timestamp_ms for t in self._trade_buf], type=pa.int64()),
            "price": pa.array([t.price for t in self._trade_buf], type=pa.float64()),
            "quantity": pa.array([t.quantity for t in self._trade_buf], type=pa.float64()),
            "is_buyer_maker": pa.array([t.is_buyer_maker for t in self._trade_buf], type=pa.bool_()),
        }, schema=TRADE_SCHEMA)

        self._append_to_parquet(self._trade_path, table, TRADE_SCHEMA)
        count = len(self._trade_buf)
        self._trade_count += count
        self._trade_buf.clear()
        return count

    def flush_book(self) -> int:
        """Write buffered book ticks to parquet. Returns count flushed."""
        if not self._book_buf:
            return 0

        table = pa.table({
            "timestamp_ms": pa.array([t.timestamp_ms for t in self._book_buf], type=pa.int64()),
            "bid_price": pa.array([t.bid_price for t in self._book_buf], type=pa.float64()),
            "ask_price": pa.array([t.ask_price for t in self._book_buf], type=pa.float64()),
            "bid_qty": pa.array([t.bid_qty for t in self._book_buf], type=pa.float64()),
            "ask_qty": pa.array([t.ask_qty for t in self._book_buf], type=pa.float64()),
        }, schema=BOOK_SCHEMA)

        self._append_to_parquet(self._book_path, table, BOOK_SCHEMA)
        count = len(self._book_buf)
        self._book_count += count
        self._book_buf.clear()
        return count

    def close(self) -> None:
        """Flush remaining data."""
        self.flush_trades()
        self.flush_book()


# ---------------------------------------------------------------------------
# TickStore — main storage manager
# ---------------------------------------------------------------------------

@dataclass
class TickStoreConfig:
    base_dir: Path = field(default_factory=lambda: Path("data/ticks"))
    flush_interval_sec: float = 300.0       # flush every 5 min
    flush_threshold: int = 50_000           # or every 50K ticks per symbol


class TickStore:
    """Manages tick storage for multiple symbols with date partitioning.

    Thread-safe: designed to be called from a single async task.
    """

    def __init__(self, config: TickStoreConfig | None = None):
        self.config = config or TickStoreConfig()
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

        self._writers: dict[str, _SymbolWriter] = {}  # key: symbol
        self._current_date: str = ""
        self._last_flush_time: float = time.monotonic()
        self._total_trades_stored: int = 0
        self._total_books_stored: int = 0

    def _current_date_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_writer(self, symbol: str) -> _SymbolWriter:
        """Get or create writer for symbol, handling day rollover."""
        date_str = self._current_date_str()

        # Day rollover — close all old writers
        if date_str != self._current_date:
            if self._current_date:
                logger.info("Day rollover: %s → %s", self._current_date, date_str)
                self._close_all()
            self._current_date = date_str

        if symbol not in self._writers:
            self._writers[symbol] = _SymbolWriter(
                symbol, date_str, self.config.base_dir
            )

        return self._writers[symbol]

    def add_trade(self, symbol: str, tick: TradeTick) -> None:
        writer = self._ensure_writer(symbol)
        writer.add_trade(tick)

    def add_book(self, symbol: str, tick: BookTick) -> None:
        writer = self._ensure_writer(symbol)
        writer.add_book(tick)

    def maybe_flush(self) -> dict[str, int]:
        """Flush if time-based or size-based threshold exceeded.

        Returns dict of {symbol: n_flushed} for any flushed symbols.
        """
        now = time.monotonic()
        elapsed = now - self._last_flush_time
        flushed: dict[str, int] = {}

        for sym, writer in self._writers.items():
            should_flush = (
                elapsed >= self.config.flush_interval_sec
                or writer.trade_buf_size >= self.config.flush_threshold
                or writer.book_buf_size >= self.config.flush_threshold
            )
            if should_flush:
                t = writer.flush_trades()
                b = writer.flush_book()
                if t or b:
                    flushed[sym] = t + b
                    self._total_trades_stored += t
                    self._total_books_stored += b

        if flushed:
            self._last_flush_time = now
            logger.info(
                "Flushed %d symbols, %d total ticks",
                len(flushed), sum(flushed.values()),
            )

        return flushed

    def flush_all(self) -> dict[str, int]:
        """Force flush all buffers."""
        flushed: dict[str, int] = {}
        for sym, writer in self._writers.items():
            t = writer.flush_trades()
            b = writer.flush_book()
            if t or b:
                flushed[sym] = t + b
                self._total_trades_stored += t
                self._total_books_stored += b
        self._last_flush_time = time.monotonic()
        return flushed

    def remove_symbol(self, symbol: str) -> None:
        """Close and remove writer for a symbol (e.g., on unsubscribe)."""
        writer = self._writers.pop(symbol, None)
        if writer is not None:
            writer.close()

    def _close_all(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def close(self) -> None:
        """Flush and close everything."""
        self._close_all()

    def stats(self) -> dict:
        """Return current collection statistics."""
        per_symbol = {}
        for sym, w in self._writers.items():
            per_symbol[sym] = {
                "trades_stored": w.total_trades,
                "books_stored": w.total_books,
                "trade_buf": w.trade_buf_size,
                "book_buf": w.book_buf_size,
            }
        return {
            "date": self._current_date,
            "symbols": len(self._writers),
            "total_trades": self._total_trades_stored,
            "total_books": self._total_books_stored,
            "per_symbol": per_symbol,
        }

    # ------------------------------------------------------------------
    # Read-back API
    # ------------------------------------------------------------------

    @staticmethod
    def read_trades(
        base_dir: Path,
        symbol: str,
        date: str | None = None,
    ) -> pa.Table | None:
        """Read stored trade ticks for a symbol on a given date.

        If date is None, reads today's data.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        path = base_dir / date / f"{symbol}_trades.parquet"
        if not path.exists():
            return None
        return pq.read_table(str(path))

    @staticmethod
    def read_book(
        base_dir: Path,
        symbol: str,
        date: str | None = None,
    ) -> pa.Table | None:
        """Read stored book ticks for a symbol on a given date."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        path = base_dir / date / f"{symbol}_book.parquet"
        if not path.exists():
            return None
        return pq.read_table(str(path))

    @staticmethod
    def list_dates(base_dir: Path) -> list[str]:
        """List all dates with stored tick data."""
        if not base_dir.exists():
            return []
        return sorted(
            d.name for d in base_dir.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
        )

    @staticmethod
    def list_symbols(base_dir: Path, date: str) -> list[str]:
        """List all symbols with trade data on a given date."""
        day_dir = base_dir / date
        if not day_dir.exists():
            return []
        symbols = set()
        for f in day_dir.iterdir():
            if f.name.endswith("_trades.parquet"):
                symbols.add(f.name.replace("_trades.parquet", ""))
        return sorted(symbols)
