"""Data loader for DTRN — loads 1-min bars from feather/parquet files.

Scans three data sources in priority order:
  1. DATA_ROOT — recent feather files from Telegram india_tick_data
  2. TELEGRAM_ROOT — full Telegram archive (790+ dates, 2022-12 to present)
  3. KITE_1MIN_ROOT — Kite broker 1-min parquet (Hive-partitioned)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DATA_ROOT, TELEGRAM_ROOT, KITE_1MIN_ROOT, INSTRUMENTS

logger = logging.getLogger(__name__)


def _scan_feather_dates(root: Path) -> set[date]:
    """Scan a directory for *-index-nfo-data.feather files and extract dates."""
    dates: set[date] = set()
    if not root.exists():
        return dates
    for f in root.glob("*-index-nfo-data.feather"):
        try:
            d = date.fromisoformat(f.stem.split("-index-nfo-data")[0])
            dates.add(d)
        except (ValueError, IndexError):
            continue
    return dates


def _scan_kite_dates(kite_root: Path) -> set[date]:
    """Scan Kite 1-min Hive-partitioned dirs for available dates.

    Structure: kite_root/{INST}_FUT_{CONTRACT}/date=YYYY-MM-DD/*.parquet
    Skips _DAILY and _SPOT directories.
    """
    dates: set[date] = set()
    if not kite_root.exists():
        return dates
    for contract_dir in kite_root.iterdir():
        if not contract_dir.is_dir():
            continue
        name = contract_dir.name
        if name.endswith("_DAILY") or name.endswith("_SPOT") or "_FUT_" not in name:
            continue
        for date_dir in contract_dir.iterdir():
            if date_dir.is_dir() and date_dir.name.startswith("date="):
                try:
                    d = date.fromisoformat(date_dir.name[5:])  # strip "date="
                    dates.add(d)
                except ValueError:
                    continue
    return dates


def list_available_dates(
    data_root: Path = DATA_ROOT,
    telegram_root: Path = TELEGRAM_ROOT,
    kite_root: Path = KITE_1MIN_ROOT,
) -> list[date]:
    """List all available trading dates from all data sources.

    Scans:
      1. DATA_ROOT feather files (recent Telegram downloads)
      2. TELEGRAM_ROOT feather files (full archive)
      3. KITE_1MIN_ROOT Hive-partitioned parquet dirs

    Returns sorted union of all dates.
    """
    all_dates: set[date] = set()

    # Source 1: DATA_ROOT feather
    dr_dates = _scan_feather_dates(data_root)
    all_dates |= dr_dates

    # Source 2: Telegram archive feather
    tg_dates = _scan_feather_dates(telegram_root)
    all_dates |= tg_dates

    # Source 3: Kite 1-min parquet
    kite_dates = _scan_kite_dates(kite_root)
    all_dates |= kite_dates

    result = sorted(all_dates)
    if result:
        logger.info(
            "Data availability: %d dates (%s to %s) — "
            "DATA_ROOT=%d, Telegram=%d, Kite=%d",
            len(result), result[0], result[-1],
            len(dr_dates), len(tg_dates), len(kite_dates),
        )
    else:
        logger.warning("No data found in any source")
    return result


def _try_read_feather(fpath: Path, instrument: str) -> Optional[pd.DataFrame]:
    """Read a single feather file and extract instrument data.

    Returns processed DataFrame or None if file is corrupt/empty/missing instrument.
    """
    try:
        df = pd.read_feather(fpath)
    except Exception as e:
        logger.debug("Failed to read feather %s: %s", fpath, e)
        return None

    # Check required columns exist (empty/corrupt files may have no columns)
    required_cols = {"name", "instrument_type", "date", "open", "high", "low",
                     "close", "volume", "oi", "expiry"}
    if not required_cols.issubset(df.columns):
        logger.debug("Feather %s missing columns: %s", fpath.name,
                      required_cols - set(df.columns))
        return None

    # Filter for instrument futures
    mask = (df["name"] == instrument) & (df["instrument_type"] == "FUT")
    df = df[mask].copy()

    if df.empty:
        return None

    # Pick nearest expiry (most liquid)
    df["expiry_date"] = pd.to_datetime(df["expiry"])
    nearest_expiry = df["expiry_date"].min()
    df = df[df["expiry_date"] == nearest_expiry].copy()

    # Clean datetime index
    df["datetime"] = pd.to_datetime(df["date"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    df = df.set_index("datetime").sort_index()

    # Keep only OHLCV + OI
    df = df[["open", "high", "low", "close", "volume", "oi"]].copy()

    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep="last")]

    # Filter market hours (9:15 to 15:30 IST)
    df = df.between_time("09:15", "15:30")

    return df if len(df) > 0 else None


def load_day_feather(
    trading_date: date,
    instrument: str = "NIFTY",
    data_root: Path = DATA_ROOT,
) -> Optional[pd.DataFrame]:
    """Load 1-min bars for a single instrument on a single day from feather.

    Tries DATA_ROOT first, then TELEGRAM_ROOT as fallback.
    Handles corrupt/empty files gracefully (holidays, partial downloads).
    Filters for FUT, picks nearest expiry (most liquid).
    Returns DataFrame with IST datetime index (tz-naive), columns: open, high, low, close, volume, oi.
    Returns None if file not found or no data for instrument.
    """
    fname = f"{trading_date.isoformat()}-index-nfo-data.feather"

    # Try DATA_ROOT first, then Telegram archive as fallback
    for root in [data_root, TELEGRAM_ROOT]:
        fpath = root / fname
        if not fpath.exists():
            continue
        result = _try_read_feather(fpath, instrument)
        if result is not None:
            return result

    return None


def load_day_kite(
    trading_date: date,
    instrument: str = "NIFTY",
    kite_root: Path = KITE_1MIN_ROOT,
) -> Optional[pd.DataFrame]:
    """Fallback: load from Kite 1-min parquet files.

    Scans specific contract dirs (e.g. NIFTY_FUT_NIFTY26FEBFUT/) for actual
    1-min bars. The DAILY dir only has daily bars (1 row/date).

    Uses pyarrow directly to avoid pandas tz compatibility issues with
    Asia/Kolkata timestamps in Hive-partitioned parquets.
    """
    if not kite_root.exists():
        return None

    date_str = f"date={trading_date.isoformat()}"

    # Scan all specific contract dirs for this instrument
    # Pattern: {INSTRUMENT}_FUT_{CONTRACT}/ (exclude _DAILY and _SPOT)
    candidates = []
    for d in kite_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(f"{instrument}_FUT_") and not name.endswith("_DAILY"):
            date_dir = d / date_str
            if date_dir.exists():
                pqs = list(date_dir.glob("*.parquet"))
                if pqs:
                    candidates.append(pqs[0])

    if not candidates:
        return None

    # Pick the file with the most rows (nearest expiry = most data)
    best_df = None
    best_rows = 0
    for pq_path in candidates:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(pq_path)
            # Use timestamp_as_object=True to avoid pandas tz compat issues
            df = table.to_pandas(timestamp_as_object=True)
        except Exception:
            try:
                df = pd.read_parquet(pq_path)
            except Exception as e:
                logger.debug("Failed to read %s: %s", pq_path, e)
                continue
        if len(df) > best_rows:
            best_df = df
            best_rows = len(df)

    if best_df is None or best_rows < 10:
        return None

    df = best_df

    # Normalize columns
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume", "oi"):
            col_map[col] = lower
    df = df.rename(columns=col_map)

    # Ensure datetime index
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
        df["datetime"] = df["datetime"].dt.tz_localize(None)
        df = df.set_index("datetime")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
        if hasattr(df["datetime"].dt, "tz") and df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        return None

    df = df.sort_index()

    needed = ["open", "high", "low", "close", "volume"]
    present = [c for c in needed if c in df.columns]
    if len(present) < 4:
        return None

    if "oi" not in df.columns:
        df["oi"] = 0

    df = df[["open", "high", "low", "close", "volume", "oi"]].copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.between_time("09:15", "15:30")

    return df


def load_day(
    trading_date: date,
    instrument: str = "NIFTY",
) -> Optional[pd.DataFrame]:
    """Load 1-min bars for instrument on given date. Tries feather first, then Kite."""
    df = load_day_feather(trading_date, instrument)
    if df is not None and len(df) > 10:
        return df
    return load_day_kite(trading_date, instrument)


def load_date_range(
    start: date,
    end: date,
    instrument: str = "NIFTY",
) -> pd.DataFrame:
    """Load 1-min bars for date range. Returns concatenated DataFrame."""
    frames = []
    current = start
    while current <= end:
        df = load_day(current, instrument)
        if df is not None and len(df) > 0:
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oi"])

    result = pd.concat(frames)
    result = result.sort_index()
    return result


def load_multi_instrument(
    trading_date: date,
    instruments: list[str] = None,
) -> dict[str, pd.DataFrame]:
    """Load 1-min bars for multiple instruments on one day."""
    if instruments is None:
        instruments = INSTRUMENTS
    result = {}
    for inst in instruments:
        df = load_day(trading_date, inst)
        if df is not None:
            result[inst] = df
    return result
