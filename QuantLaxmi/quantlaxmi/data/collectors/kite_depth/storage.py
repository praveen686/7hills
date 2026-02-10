"""Depth tick storage — buffered Parquet writer with IST date partitioning.

Each symbol gets one file per day:
  zerodha/5level/{date}/{SYMBOL}.parquet

Files: NIFTY_FUT, BANKNIFTY_FUT, NIFTY_OPT, BANKNIFTY_OPT

Uses atomic write-on-flush: each flush reads existing file, concatenates
new data, and writes a complete parquet file via tmp+rename.

Flush triggers:
  - Time-based: every flush_interval_sec (default 60s)
  - Size-based: every flush_threshold ticks (default 10,000)
  - Day rollover: new IST date → close old writers, open new files
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quantlaxmi.data._paths import KITE_DEPTH_DIR

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Parquet Schema — 41 flat columns
# ---------------------------------------------------------------------------

DEPTH_SCHEMA = pa.schema([
    ("timestamp_ms", pa.int64()),
    ("instrument_token", pa.int32()),
    ("symbol", pa.string()),
    ("last_price", pa.float64()),
    ("volume", pa.int64()),
    ("oi", pa.int64()),
    ("total_buy_qty", pa.int64()),
    ("total_sell_qty", pa.int64()),
    # Instrument identification (options)
    ("strike", pa.float64()),          # 0.0 for futures
    ("expiry", pa.string()),           # "YYYY-MM-DD"
    ("option_type", pa.string()),      # "CE", "PE", or "FUT"
    # Bid levels 1-5
    ("bid_price_1", pa.float64()),
    ("bid_qty_1", pa.int64()),
    ("bid_orders_1", pa.int32()),
    ("bid_price_2", pa.float64()),
    ("bid_qty_2", pa.int64()),
    ("bid_orders_2", pa.int32()),
    ("bid_price_3", pa.float64()),
    ("bid_qty_3", pa.int64()),
    ("bid_orders_3", pa.int32()),
    ("bid_price_4", pa.float64()),
    ("bid_qty_4", pa.int64()),
    ("bid_orders_4", pa.int32()),
    ("bid_price_5", pa.float64()),
    ("bid_qty_5", pa.int64()),
    ("bid_orders_5", pa.int32()),
    # Ask levels 1-5
    ("ask_price_1", pa.float64()),
    ("ask_qty_1", pa.int64()),
    ("ask_orders_1", pa.int32()),
    ("ask_price_2", pa.float64()),
    ("ask_qty_2", pa.int64()),
    ("ask_orders_2", pa.int32()),
    ("ask_price_3", pa.float64()),
    ("ask_qty_3", pa.int64()),
    ("ask_orders_3", pa.int32()),
    ("ask_price_4", pa.float64()),
    ("ask_qty_4", pa.int64()),
    ("ask_orders_4", pa.int32()),
    ("ask_price_5", pa.float64()),
    ("ask_qty_5", pa.int64()),
    ("ask_orders_5", pa.int32()),
])


# ---------------------------------------------------------------------------
# DepthTick dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DepthTick:
    """Single depth tick flattened for parquet storage."""

    timestamp_ms: int
    instrument_token: int
    symbol: str
    last_price: float
    volume: int
    oi: int
    total_buy_qty: int
    total_sell_qty: int
    # Instrument identification
    strike: float
    expiry: str
    option_type: str
    # Bid levels
    bid_price_1: float
    bid_qty_1: int
    bid_orders_1: int
    bid_price_2: float
    bid_qty_2: int
    bid_orders_2: int
    bid_price_3: float
    bid_qty_3: int
    bid_orders_3: int
    bid_price_4: float
    bid_qty_4: int
    bid_orders_4: int
    bid_price_5: float
    bid_qty_5: int
    bid_orders_5: int
    # Ask levels
    ask_price_1: float
    ask_qty_1: int
    ask_orders_1: int
    ask_price_2: float
    ask_qty_2: int
    ask_orders_2: int
    ask_price_3: float
    ask_qty_3: int
    ask_orders_3: int
    ask_price_4: float
    ask_qty_4: int
    ask_orders_4: int
    ask_price_5: float
    ask_qty_5: int
    ask_orders_5: int

    @classmethod
    def from_kite_tick(
        cls,
        tick,
        symbol: str,
        strike: float = 0.0,
        expiry: str = "",
        option_type: str = "FUT",
    ) -> DepthTick:
        """Build DepthTick from a KiteTick.

        Pads with zeros if fewer than 5 levels are present.
        Falls back to local IST time if tick.timestamp is None.
        """
        if tick.timestamp is not None:
            ts_ms = int(tick.timestamp.timestamp() * 1000)
        else:
            ts_ms = int(datetime.now(IST).timestamp() * 1000)

        depth = tick.depth or {}
        buys = depth.get("buy", [])
        sells = depth.get("sell", [])

        # Pad to 5 levels
        def _level(levels, i):
            if i < len(levels):
                lv = levels[i]
                return (
                    float(lv.get("price", 0.0)),
                    int(lv.get("quantity", 0)),
                    int(lv.get("orders", 0)),
                )
            return (0.0, 0, 0)

        b = [_level(buys, i) for i in range(5)]
        a = [_level(sells, i) for i in range(5)]

        return cls(
            timestamp_ms=ts_ms,
            instrument_token=tick.instrument_token,
            symbol=symbol,
            last_price=tick.last_price,
            volume=tick.volume,
            oi=tick.oi,
            total_buy_qty=tick.buy_qty,
            total_sell_qty=tick.sell_qty,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            bid_price_1=b[0][0], bid_qty_1=b[0][1], bid_orders_1=b[0][2],
            bid_price_2=b[1][0], bid_qty_2=b[1][1], bid_orders_2=b[1][2],
            bid_price_3=b[2][0], bid_qty_3=b[2][1], bid_orders_3=b[2][2],
            bid_price_4=b[3][0], bid_qty_4=b[3][1], bid_orders_4=b[3][2],
            bid_price_5=b[4][0], bid_qty_5=b[4][1], bid_orders_5=b[4][2],
            ask_price_1=a[0][0], ask_qty_1=a[0][1], ask_orders_1=a[0][2],
            ask_price_2=a[1][0], ask_qty_2=a[1][1], ask_orders_2=a[1][2],
            ask_price_3=a[2][0], ask_qty_3=a[2][1], ask_orders_3=a[2][2],
            ask_price_4=a[3][0], ask_qty_4=a[3][1], ask_orders_4=a[3][2],
            ask_price_5=a[4][0], ask_qty_5=a[4][1], ask_orders_5=a[4][2],
        )


# ---------------------------------------------------------------------------
# Per-symbol writer
# ---------------------------------------------------------------------------

class _DepthSymbolWriter:
    """Manages a single parquet file for one symbol on one day.

    Uses atomic write-on-flush: read existing → concat → write tmp → rename.
    """

    def __init__(self, symbol: str, date_str: str, base_dir: Path):
        self.symbol = symbol
        self.date_str = date_str

        day_dir = base_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        self._path = day_dir / f"{symbol}.parquet"
        self._buf: list[DepthTick] = []
        self._total_count = 0
        self._last_flush_time: float = time.monotonic()

    def add(self, tick: DepthTick) -> None:
        self._buf.append(tick)

    @property
    def buf_size(self) -> int:
        return len(self._buf)

    @property
    def total_stored(self) -> int:
        return self._total_count

    @staticmethod
    def _append_to_parquet(path: Path, new_table: pa.Table) -> None:
        """Append new_table to an existing parquet file (or create it).

        Reads existing file, concatenates, writes back. Atomic via tmp+rename.
        """
        if path.exists():
            try:
                existing = pq.read_table(str(path), schema=DEPTH_SCHEMA)
                combined = pa.concat_tables([existing, new_table])
            except Exception:
                combined = new_table
        else:
            combined = new_table

        tmp_path = path.with_suffix(".parquet.tmp")
        pq.write_table(combined, str(tmp_path), compression="zstd")
        tmp_path.rename(path)

    def flush(self) -> int:
        """Write buffered ticks to parquet. Returns count flushed."""
        if not self._buf:
            return 0

        arrays = {
            "timestamp_ms": pa.array([t.timestamp_ms for t in self._buf], type=pa.int64()),
            "instrument_token": pa.array([t.instrument_token for t in self._buf], type=pa.int32()),
            "symbol": pa.array([t.symbol for t in self._buf], type=pa.string()),
            "last_price": pa.array([t.last_price for t in self._buf], type=pa.float64()),
            "volume": pa.array([t.volume for t in self._buf], type=pa.int64()),
            "oi": pa.array([t.oi for t in self._buf], type=pa.int64()),
            "total_buy_qty": pa.array([t.total_buy_qty for t in self._buf], type=pa.int64()),
            "total_sell_qty": pa.array([t.total_sell_qty for t in self._buf], type=pa.int64()),
            "strike": pa.array([t.strike for t in self._buf], type=pa.float64()),
            "expiry": pa.array([t.expiry for t in self._buf], type=pa.string()),
            "option_type": pa.array([t.option_type for t in self._buf], type=pa.string()),
        }

        for prefix in ("bid", "ask"):
            for i in range(1, 6):
                arrays[f"{prefix}_price_{i}"] = pa.array(
                    [getattr(t, f"{prefix}_price_{i}") for t in self._buf], type=pa.float64()
                )
                arrays[f"{prefix}_qty_{i}"] = pa.array(
                    [getattr(t, f"{prefix}_qty_{i}") for t in self._buf], type=pa.int64()
                )
                arrays[f"{prefix}_orders_{i}"] = pa.array(
                    [getattr(t, f"{prefix}_orders_{i}") for t in self._buf], type=pa.int32()
                )

        table = pa.table(arrays, schema=DEPTH_SCHEMA)
        self._append_to_parquet(self._path, table)

        count = len(self._buf)
        self._total_count += count
        self._buf.clear()
        return count

    def close(self) -> None:
        """Flush remaining data."""
        self.flush()


# ---------------------------------------------------------------------------
# DepthStore — main storage manager
# ---------------------------------------------------------------------------

@dataclass
class DepthStoreConfig:
    base_dir: Path = field(
        default_factory=lambda: KITE_DEPTH_DIR
    )
    flush_interval_sec: float = 60.0
    flush_threshold: int = 10_000


class DepthStore:
    """Manages depth tick storage for multiple symbols with IST date partitioning."""

    def __init__(self, config: DepthStoreConfig | None = None):
        self.config = config or DepthStoreConfig()
        self.config.base_dir.mkdir(parents=True, exist_ok=True)

        self._writers: dict[tuple[str, str], _DepthSymbolWriter] = {}
        self._current_date: str = ""
        self._total_stored: int = 0

    @staticmethod
    def _current_date_str() -> str:
        return datetime.now(IST).strftime("%Y-%m-%d")

    def _ensure_writer(self, symbol: str) -> _DepthSymbolWriter:
        """Get or create writer for symbol, handling day rollover."""
        date_str = self._current_date_str()

        # Day rollover — close all old writers
        if date_str != self._current_date:
            if self._current_date:
                logger.info("Day rollover: %s → %s", self._current_date, date_str)
                self._close_all()
            self._current_date = date_str

        key = (symbol, date_str)
        if key not in self._writers:
            self._writers[key] = _DepthSymbolWriter(
                symbol, date_str, self.config.base_dir
            )

        return self._writers[key]

    def add_tick(self, symbol: str, tick: DepthTick) -> None:
        writer = self._ensure_writer(symbol)
        writer.add(tick)

    def maybe_flush(self, force: bool = False) -> dict[str, int]:
        """Flush if time-based or size-based threshold exceeded.

        Time check is per-writer so high-volume writers (options) don't
        starve low-volume writers (futures) by resetting a global timer.

        Returns dict of {symbol: n_flushed} for any flushed symbols.
        """
        now = time.monotonic()
        flushed: dict[str, int] = {}

        for (sym, _date), writer in self._writers.items():
            elapsed = now - writer._last_flush_time
            should_flush = (
                force
                or elapsed >= self.config.flush_interval_sec
                or writer.buf_size >= self.config.flush_threshold
            )
            if should_flush and writer.buf_size > 0:
                n = writer.flush()
                if n:
                    flushed[sym] = n
                    self._total_stored += n
                writer._last_flush_time = now

        if flushed:
            logger.info(
                "Flushed %d symbols, %d ticks",
                len(flushed), sum(flushed.values()),
            )

        return flushed

    def flush_all(self) -> dict[str, int]:
        """Force flush all buffers."""
        return self.maybe_flush(force=True)

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
        for (sym, _date), w in self._writers.items():
            per_symbol[sym] = {
                "stored": w.total_stored,
                "buffered": w.buf_size,
            }
        return {
            "date": self._current_date,
            "symbols": len(self._writers),
            "total_stored": self._total_stored,
            "per_symbol": per_symbol,
        }

    # ------------------------------------------------------------------
    # Read-back API
    # ------------------------------------------------------------------

    @staticmethod
    def read_depth(
        base_dir: Path,
        symbol: str,
        date: str | None = None,
    ) -> pa.Table | None:
        """Read stored depth ticks for a symbol on a given date."""
        if date is None:
            date = datetime.now(IST).strftime("%Y-%m-%d")
        path = base_dir / date / f"{symbol}.parquet"
        if not path.exists():
            return None
        return pq.read_table(str(path))

    @staticmethod
    def list_dates(base_dir: Path) -> list[str]:
        """List all dates with stored depth data."""
        if not base_dir.exists():
            return []
        return sorted(
            d.name for d in base_dir.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
        )

    @staticmethod
    def list_symbols(base_dir: Path, date: str) -> list[str]:
        """List all symbols with depth data on a given date."""
        day_dir = base_dir / date
        if not day_dir.exists():
            return []
        return sorted(
            f.stem for f in day_dir.iterdir()
            if f.suffix == ".parquet" and not f.name.endswith(".tmp")
        )
