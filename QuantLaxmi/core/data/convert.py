"""Convert raw Telegram-sourced tick data to hive-partitioned Parquet.

Source layout (in DOWNLOAD_DIR):
  YYYY-MM-DD-index-nfo-data.feather   → 1-min OHLCV for NFO index options/futures
  YYYY-MM-DD-bfo-data.feather         → 1-min OHLCV for BFO options
  YYYY-MM-DD_tick_data.zip            → zstd-compressed pickle: {token: [[ts, ltp, ...], ...]}
  instrument_df_YYYY-MM-DD.zip        → zstd-compressed pickle: instrument master DataFrame

Output layout (in MARKET_DIR):
  nfo_1min/date=YYYY-MM-DD/data.parquet    # NFO 1-min OHLCV
  bfo_1min/date=YYYY-MM-DD/data.parquet    # BFO 1-min OHLCV
  ticks/date=YYYY-MM-DD/data.parquet       # Tick-level LTP+vol+OI
  instruments/date=YYYY-MM-DD/data.parquet # Instrument master

Usage:
  python -m qlx.data.convert                    # convert all available dates
  python -m qlx.data.convert --dates 2026-02-05 # convert specific date(s)
  python -m qlx.data.convert --dry-run          # show what would be converted
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(
    os.environ.get(
        "QLX_DATA_ROOT",
        "/home/ubuntu/Desktop/7hills/QuantLaxmi/data",
    )
)

DOWNLOAD_DIR = _DATA_ROOT / "telegram_source_files" / "india_tick_data"
UNPACKED_DIR = _DATA_ROOT / "telegram_source_files" / "india_tick_unpacked"
MARKET_DIR = _DATA_ROOT / "market"

# ---------------------------------------------------------------------------
# Date extraction from filenames
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date(filename: str) -> date | None:
    """Pull YYYY-MM-DD from any filename."""
    m = _DATE_RE.search(filename)
    if m:
        return date.fromisoformat(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Discovery — find all available dates per source type
# ---------------------------------------------------------------------------


def discover_sources() -> dict[str, set[date]]:
    """Scan DOWNLOAD_DIR and return {source_type: {dates}}."""
    result: dict[str, set[date]] = {
        "nfo_feather": set(),
        "bfo_feather": set(),
        "tick_zip": set(),
        "tick_pkl": set(),
        "instrument_pkl": set(),
    }

    if not DOWNLOAD_DIR.exists():
        return result

    for f in DOWNLOAD_DIR.iterdir():
        d = _extract_date(f.name)
        if d is None:
            continue
        if f.name.endswith("-index-nfo-data.feather"):
            result["nfo_feather"].add(d)
        elif f.name.endswith("-bfo-data.feather"):
            result["bfo_feather"].add(d)
        elif f.name.endswith("_tick_data.zip"):
            result["tick_zip"].add(d)
        elif f.name.endswith("_tick_data.pkl") or (
            f.name.startswith("tick_data_") and f.name.endswith(".pkl")
        ):
            result["tick_pkl"].add(d)
        elif f.name.startswith("instrument_df_") and f.name.endswith(".pkl"):
            result["instrument_pkl"].add(d)

    # Also check unpacked dir
    if UNPACKED_DIR.exists():
        for d_dir in UNPACKED_DIR.iterdir():
            if not d_dir.is_dir():
                continue
            d = _extract_date(d_dir.name)
            if d is None:
                continue
            for f in d_dir.iterdir():
                if "tick_data" in f.name and f.name.endswith(".pkl") and f.stat().st_size > 0:
                    result["tick_pkl"].add(d)
                elif "instrument_df" in f.name and f.name.endswith(".pkl") and f.stat().st_size > 0:
                    result["instrument_pkl"].add(d)

    return result


def discover_converted() -> dict[str, set[date]]:
    """Scan MARKET_DIR and return {category: {dates already converted}}."""
    result: dict[str, set[date]] = {}
    for category in ("nfo_1min", "bfo_1min", "ticks", "instruments"):
        result[category] = set()
        cat_dir = MARKET_DIR / category
        if not cat_dir.exists():
            continue
        for part_dir in cat_dir.iterdir():
            if part_dir.is_dir() and part_dir.name.startswith("date="):
                d_str = part_dir.name[5:]  # strip "date="
                try:
                    result[category].add(date.fromisoformat(d_str))
                except ValueError:
                    continue
    return result


# ---------------------------------------------------------------------------
# Feather → Parquet converters
# ---------------------------------------------------------------------------


def _feather_path(d: date, kind: str) -> Path:
    """Build path to a feather source file."""
    if kind == "nfo":
        return DOWNLOAD_DIR / f"{d.isoformat()}-index-nfo-data.feather"
    elif kind == "bfo":
        return DOWNLOAD_DIR / f"{d.isoformat()}-bfo-data.feather"
    raise ValueError(f"Unknown feather kind: {kind}")


def _parquet_out(category: str, d: date) -> Path:
    """Build hive-partitioned output path."""
    return MARKET_DIR / category / f"date={d.isoformat()}" / "data.parquet"


def convert_feather(d: date, kind: str) -> int:
    """Convert a single feather file to parquet. Returns row count."""
    src = _feather_path(d, kind)
    category = "nfo_1min" if kind == "nfo" else "bfo_1min"
    dst = _parquet_out(category, d)

    if not src.exists():
        logger.warning("Source not found: %s", src)
        return 0

    df = pd.read_feather(src)

    if df.empty:
        return 0

    # Normalize the timezone-aware datetime to timezone-naive UTC+5:30 for storage
    if "date" in df.columns and hasattr(df["date"].dtype, "tz"):
        df["date"] = df["date"].dt.tz_localize(None)

    # Downcast float64 → float32 for price columns (saves ~40% space)
    float_cols = df.select_dtypes("float64").columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)

    return len(df)


# ---------------------------------------------------------------------------
# Tick pickle → Parquet
# ---------------------------------------------------------------------------


def _find_tick_pkl(d: date) -> Path | None:
    """Locate the tick_data pkl for a date (could be in main dir or unpacked)."""
    # Check unpacked dir first (has date subfolders)
    unpacked = UNPACKED_DIR / d.isoformat() / f"tick_data_{d.isoformat()}.pkl"
    if unpacked.exists() and unpacked.stat().st_size > 0:
        return unpacked

    # Check main download dir — two naming conventions
    for pattern in [
        f"tick_data_{d.isoformat()}.pkl",
        f"{d.isoformat()}_tick_data.pkl",
    ]:
        p = DOWNLOAD_DIR / pattern
        if p.exists() and p.stat().st_size > 0:
            return p

    return None


def _find_instrument_pkl(d: date) -> Path | None:
    """Locate the instrument_df pkl for a date."""
    unpacked = UNPACKED_DIR / d.isoformat() / f"instrument_df_{d.isoformat()}.pkl"
    if unpacked.exists() and unpacked.stat().st_size > 0:
        return unpacked

    p = DOWNLOAD_DIR / f"instrument_df_{d.isoformat()}.pkl"
    if p.exists() and p.stat().st_size > 0:
        return p

    return None


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress zstd-compressed bytes."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data, max_output_size=500 * 1024 * 1024)  # 500 MB max


def _decompress_zstd_file(src: Path, dst: Path) -> None:
    """Decompress a zstd-compressed file to disk."""
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as ifh, open(dst, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)


def _load_pkl_from_bytes(data: bytes) -> object:
    """Load a pickle from bytes, with zstd decompression if needed."""
    # Check for zstd magic
    if len(data) >= 4 and data[:4] == b"\x28\xb5\x2f\xfd":
        data = _decompress_zstd(data)
    return pickle.loads(data)


def _load_tick_dicts(d: date) -> dict | None:
    """Load tick data dict(s) for a date, handling all source formats.

    Returns merged {token: [[ts, ltp, ...], ...]} dict, or None.
    Avoids writing intermediate pkl files to disk.
    """
    import io
    import zipfile

    # 1. Check for already-extracted pkl
    existing = _find_tick_pkl(d)
    if existing:
        try:
            with open(existing, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data:
                return data
        except Exception as e:
            logger.warning("Corrupt pkl %s, will re-extract: %s", existing, e)
            existing.unlink()

    # 2. Extract from zip (handles nested zstd and direct formats)
    zip_path = DOWNLOAD_DIR / f"{d.isoformat()}_tick_data.zip"
    if not zip_path.exists():
        return None

    if not zipfile.is_zipfile(zip_path):
        # Try direct zstd decompression
        try:
            raw = _decompress_zstd(zip_path.read_bytes())
            return pickle.loads(raw)
        except Exception as e:
            logger.warning("Failed to extract tick data for %s: %s", d, e)
            return None

    combined: dict = {}
    instrument_extracted = False

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "tick_data" not in name and "instrument" not in name:
                continue
            inner_bytes = zf.read(name)

            try:
                obj = _load_pkl_from_bytes(inner_bytes)
            except Exception as e:
                logger.warning("Failed to load %s from zip: %s", name, e)
                continue

            if "tick_data" in name and isinstance(obj, dict):
                combined.update(obj)
            elif "instrument" in name:
                # Save instrument pkl to disk for later use
                out_dir = UNPACKED_DIR / d.isoformat()
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"instrument_df_{d.isoformat()}.pkl"
                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                instrument_extracted = True

    return combined if combined else None


def _ensure_instrument_pkl(d: date) -> Path | None:
    """Find or extract the instrument pkl for a date."""
    existing = _find_instrument_pkl(d)
    if existing:
        return existing

    import zipfile

    # Check if instrument is inside the tick_data zip (older format)
    zip_path = DOWNLOAD_DIR / f"{d.isoformat()}_tick_data.zip"
    out_dir = UNPACKED_DIR / d.isoformat()
    out_path = out_dir / f"instrument_df_{d.isoformat()}.pkl"

    # Try tick_data zip (instrument is often bundled inside)
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if "instrument" not in name:
                    continue
                inner_bytes = zf.read(name)
                try:
                    obj = _load_pkl_from_bytes(inner_bytes)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as f:
                        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                    return out_path
                except Exception as e:
                    logger.warning("Failed to extract instrument for %s: %s", d, e)

    return None


def _flatten_tick_dict(data: dict) -> tuple[list, list, list, list, list]:
    """Flatten tick dict to columnar lists. Returns (tokens, timestamps, ltps, volumes, ois)."""
    tokens = []
    timestamps = []
    ltps = []
    volumes = []
    ois = []

    for token, ticks in data.items():
        if not ticks:
            continue
        n = len(ticks)
        tick_len = len(ticks[0])

        tokens.extend([token] * n)

        if tick_len == 2:
            for t in ticks:
                timestamps.append(t[0])
                ltps.append(t[1])
                volumes.append(0)
                ois.append(0)
        elif tick_len == 4:
            for t in ticks:
                timestamps.append(t[0])
                ltps.append(t[1])
                volumes.append(t[2])
                ois.append(t[3])
        else:
            tokens = tokens[:-n]
            logger.warning("Unknown tick format for token %s: len=%d", token, tick_len)

    return tokens, timestamps, ltps, volumes, ois


def convert_ticks(d: date) -> int:
    """Convert tick data for a date to parquet. Returns row count.

    Loads directly from zip (no intermediate pkl files on disk).
    """
    dst = _parquet_out("ticks", d)

    data = _load_tick_dicts(d)
    if data is None:
        logger.warning("No tick data found for %s", d)
        return 0

    tokens, timestamps, ltps, volumes, ois = _flatten_tick_dict(data)

    if not tokens:
        return 0

    table = pa.table({
        "instrument_token": pa.array(tokens, type=pa.int32()),
        "timestamp": pa.array(timestamps, type=pa.timestamp("us")),
        "ltp": pa.array(ltps, type=pa.float32()),
        "volume": pa.array(volumes, type=pa.int64()),
        "oi": pa.array(ois, type=pa.int64()),
    })

    # Sort by timestamp then token for better compression
    table = table.sort_by([("timestamp", "ascending"), ("instrument_token", "ascending")])

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="zstd")

    return len(table)


def convert_instruments(d: date) -> int:
    """Convert instrument master pkl for a date to parquet. Returns row count."""
    dst = _parquet_out("instruments", d)

    pkl_path = _ensure_instrument_pkl(d)
    if pkl_path is None:
        logger.warning("No instrument pkl found for %s", d)
        return 0

    with open(pkl_path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Instrument pkl for %s is not a valid DataFrame", d)
        return 0

    # Downcast types for space efficiency
    if "instrument_token" in df.columns:
        df["instrument_token"] = df["instrument_token"].astype(np.int32)
    if "lot_size" in df.columns:
        df["lot_size"] = df["lot_size"].astype(np.int32)
    for col in ("last_price", "strike", "tick_size"):
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # Ensure string columns are proper strings (not mixed object types)
    for col in ("expiry", "exchange_token", "tradingsymbol", "name",
                "instrument_type", "segment", "exchange"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)

    return len(df)


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def convert_day(d: date, force: bool = False) -> dict[str, int]:
    """Convert all available source data for a single date.

    Returns {category: row_count} for each successfully converted source.
    """
    converted = discover_converted() if not force else {k: set() for k in ("nfo_1min", "bfo_1min", "ticks", "instruments")}
    results: dict[str, int] = {}

    # NFO feather
    if d not in converted.get("nfo_1min", set()):
        src = _feather_path(d, "nfo")
        if src.exists():
            n = convert_feather(d, "nfo")
            if n:
                results["nfo_1min"] = n
    else:
        logger.debug("nfo_1min already converted for %s", d)

    # BFO feather
    if d not in converted.get("bfo_1min", set()):
        src = _feather_path(d, "bfo")
        if src.exists():
            n = convert_feather(d, "bfo")
            if n:
                results["bfo_1min"] = n
    else:
        logger.debug("bfo_1min already converted for %s", d)

    # Ticks
    if d not in converted.get("ticks", set()):
        n = convert_ticks(d)
        if n:
            results["ticks"] = n
    else:
        logger.debug("ticks already converted for %s", d)

    # Instruments
    if d not in converted.get("instruments", set()):
        n = convert_instruments(d)
        if n:
            results["instruments"] = n
    else:
        logger.debug("instruments already converted for %s", d)

    return results


def convert_all(
    dates: list[date] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, int]]:
    """Convert all available dates (or a specific list).

    Returns {date_str: {category: row_count}}.
    """
    sources = discover_sources()
    converted = discover_converted() if not force else {k: set() for k in ("nfo_1min", "bfo_1min", "ticks", "instruments")}

    # All dates across all source types
    all_dates = sorted(
        sources["nfo_feather"]
        | sources["bfo_feather"]
        | sources["tick_zip"]
        | sources["tick_pkl"]
        | sources["instrument_pkl"]
    )

    if dates:
        all_dates = sorted(set(dates) & set(all_dates))

    # Filter to unconverted
    to_convert = []
    for d in all_dates:
        needs = False
        if d in sources["nfo_feather"] and d not in converted.get("nfo_1min", set()):
            needs = True
        if d in sources["bfo_feather"] and d not in converted.get("bfo_1min", set()):
            needs = True
        if (d in sources["tick_zip"] or d in sources["tick_pkl"]) and d not in converted.get("ticks", set()):
            needs = True
        if d in sources["instrument_pkl"] and d not in converted.get("instruments", set()):
            needs = True
        if force or needs:
            to_convert.append(d)

    logger.info(
        "Found %d dates with source data, %d need conversion",
        len(all_dates),
        len(to_convert),
    )

    if dry_run:
        for d in to_convert:
            types = []
            if d in sources["nfo_feather"]:
                types.append("nfo")
            if d in sources["bfo_feather"]:
                types.append("bfo")
            if d in sources["tick_zip"] or d in sources["tick_pkl"]:
                types.append("ticks")
            if d in sources["instrument_pkl"]:
                types.append("instruments")
            print(f"  {d.isoformat()}: {', '.join(types)}")
        return {}

    results: dict[str, dict[str, int]] = {}
    t0 = time.time()

    for i, d in enumerate(to_convert, 1):
        d_str = d.isoformat()
        t1 = time.time()
        try:
            day_results = convert_day(d, force=force)
            results[d_str] = day_results
            elapsed = time.time() - t1
            summary = ", ".join(f"{k}={v:,}" for k, v in day_results.items())
            logger.info(
                "[%d/%d] %s: %s (%.1fs)",
                i, len(to_convert), d_str, summary or "nothing", elapsed,
            )
        except Exception as e:
            logger.error("[%d/%d] %s: FAILED: %s", i, len(to_convert), d_str, e)
            results[d_str] = {"error": str(e)}

    total_time = time.time() - t0
    total_rows = sum(
        sum(v for v in day.values() if isinstance(v, int))
        for day in results.values()
    )
    logger.info(
        "Conversion complete: %d dates, %d total rows, %.1fs",
        len(results), total_rows, total_time,
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw Telegram tick data to hive-partitioned Parquet"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        type=date.fromisoformat,
        help="Specific dates to convert (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if parquet already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without doing it",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = convert_all(
        dates=args.dates,
        force=args.force,
        dry_run=args.dry_run,
    )

    if not args.dry_run and results:
        # Print summary
        categories = ("nfo_1min", "bfo_1min", "ticks", "instruments")
        for cat in categories:
            count = sum(1 for d in results.values() if cat in d)
            rows = sum(d.get(cat, 0) for d in results.values() if isinstance(d.get(cat), int))
            print(f"  {cat}: {count} days, {rows:,} rows")

        # Print output size
        if MARKET_DIR.exists():
            total_bytes = sum(
                f.stat().st_size
                for f in MARKET_DIR.rglob("*.parquet")
            )
            print(f"\n  Total parquet size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
