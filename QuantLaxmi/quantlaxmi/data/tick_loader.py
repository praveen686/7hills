"""TickLoader — read, filter, and resample Zerodha tick data from Hive-partitioned Parquet.

Data layout
-----------
    {data_root}/market/ticks/date=YYYY-MM-DD/*.parquet

Columns (per Parquet file)
--------------------------
    instrument_token  int32         Zerodha instrument token
    timestamp         datetime64    IST (naive) — see CORRUPTION NOTE
    ltp               float32       Last traded price
    volume            int64         Cumulative volume (since market open)
    oi                int64         Open interest

CORRUPTION NOTE
~~~~~~~~~~~~~~~
A small number of rows per day (~0.003%) have timestamp = 1970-01-01 05:30:00.
These are *initial instrument state snapshots* pushed by the Kite WebSocket at
epoch 0 (interpreted as IST → +5h30m).  They carry ltp = 0.05 (minimum tick)
and zero volume.  The loader **drops them automatically** via a
``timestamp > 2000-01-01`` filter.  No data rewriting is needed.

Instrument mapping
------------------
    {data_root}/market/instruments/date=YYYY-MM-DD/data.parquet
    Columns: instrument_token (int32), tradingsymbol, name, exchange,
             segment, instrument_type, expiry, strike, lot_size, ...

Usage
-----
    from quantlaxmi.data.tick_loader import TickLoader

    tl = TickLoader()
    ticks = tl.load(instrument_token=256265, date="2026-02-05")
    bars  = tl.resample(ticks, "1min")
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

from quantlaxmi.data._paths import TICK_DIR, MARKET_DIR

INSTRUMENT_DIR = MARKET_DIR / "instruments"

# Epoch-corruption cutoff (see module docstring)
_EPOCH_CUTOFF = pd.Timestamp("2000-01-01")

# NSE market hours (IST, naive)
_MARKET_OPEN = pd.Timestamp("09:15:00").time()
_MARKET_CLOSE = pd.Timestamp("15:30:00").time()

# Standard resample frequencies (pandas offset aliases)
_VALID_FREQS = {
    "1s": "1s",
    "5s": "5s",
    "10s": "10s",
    "15s": "15s",
    "30s": "30s",
    "1min": "1min",
    "2min": "2min",
    "3min": "3min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
    # Aliases people might type
    "1T": "1min",
    "5T": "5min",
    "15T": "15min",
    "30T": "30min",
    "1H": "1h",
}


# ---------------------------------------------------------------------------
# TickLoader
# ---------------------------------------------------------------------------


class TickLoader:
    """Read, filter, clean, and resample Zerodha tick data.

    Parameters
    ----------
    tick_dir : Path | str | None
        Root of the Hive-partitioned tick Parquet store.
        Defaults to ``{QUANTLAXMI_DATA_ROOT}/market/ticks``.
    instrument_dir : Path | str | None
        Root of the Hive-partitioned instrument Parquet store.
        Defaults to ``{QUANTLAXMI_DATA_ROOT}/market/instruments``.
    market_hours_only : bool
        If True (default), drop ticks outside 09:15–15:30 IST.
    """

    def __init__(
        self,
        tick_dir: str | Path | None = None,
        instrument_dir: str | Path | None = None,
        market_hours_only: bool = True,
    ):
        self.tick_dir = Path(tick_dir) if tick_dir else TICK_DIR
        self.instrument_dir = Path(instrument_dir) if instrument_dir else INSTRUMENT_DIR
        self.market_hours_only = market_hours_only

        if not self.tick_dir.exists():
            raise FileNotFoundError(f"Tick directory not found: {self.tick_dir}")

        # Cache for instrument lookups (date_str → DataFrame)
        self._instrument_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_dates(self) -> list[str]:
        """Return sorted list of date strings (YYYY-MM-DD) with tick data."""
        dates = []
        for d in self.tick_dir.iterdir():
            if d.is_dir() and d.name.startswith("date="):
                dates.append(d.name[5:])
        return sorted(dates)

    def load(
        self,
        instrument_token: int | list[int] | None = None,
        date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load tick data with optional filters.

        Parameters
        ----------
        instrument_token : int or list[int], optional
            Filter to specific instrument(s).  If None, returns all.
        date : str, optional
            Single date "YYYY-MM-DD".  Shorthand for start_date=end_date=date.
        start_date : str, optional
            Inclusive start date "YYYY-MM-DD".
        end_date : str, optional
            Inclusive end date "YYYY-MM-DD".
        columns : list[str], optional
            Columns to read.  Defaults to all.

        Returns
        -------
        pd.DataFrame
            Cleaned ticks sorted by timestamp with columns:
            instrument_token, timestamp, ltp, volume, oi
            (plus 'date' partition column if multi-date).
        """
        # Resolve date range
        if date is not None:
            start_date = end_date = date

        # Find matching partition directories
        target_dirs = self._resolve_partitions(start_date, end_date)
        if not target_dirs:
            logger.warning("No tick data found for date range %s to %s", start_date, end_date)
            return pd.DataFrame(
                columns=["instrument_token", "timestamp", "ltp", "volume", "oi"]
            )

        frames: list[pd.DataFrame] = []
        for d_dir in target_dirs:
            parquet_files = list(d_dir.glob("*.parquet"))
            if not parquet_files:
                continue

            df = pq.read_table(
                parquet_files[0],
                columns=columns,
            ).to_pandas()

            # -- Clean epoch-corrupted timestamps --
            df = df[df["timestamp"] > _EPOCH_CUTOFF]

            # -- Filter by instrument_token --
            if instrument_token is not None:
                if isinstance(instrument_token, (list, tuple)):
                    df = df[df["instrument_token"].isin(instrument_token)]
                else:
                    df = df[df["instrument_token"] == instrument_token]

            # -- Drop ticks outside market hours --
            if self.market_hours_only and len(df) > 0:
                ts_time = df["timestamp"].dt.time
                df = df[
                    (ts_time >= _MARKET_OPEN) & (ts_time <= _MARKET_CLOSE)
                ]

            if len(df) > 0:
                # Add partition date if not already present
                if "date" not in df.columns:
                    df["date"] = d_dir.name[5:]  # "date=YYYY-MM-DD" → "YYYY-MM-DD"
                frames.append(df)

        if not frames:
            return pd.DataFrame(
                columns=["instrument_token", "timestamp", "ltp", "volume", "oi"]
            )

        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Loaded %d ticks (%d instruments, %d dates)",
            len(result),
            result["instrument_token"].nunique(),
            result["date"].nunique() if "date" in result.columns else 1,
        )
        return result

    def resample(
        self,
        ticks: pd.DataFrame,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """Resample tick data into OHLCV+VWAP bars.

        Parameters
        ----------
        ticks : pd.DataFrame
            Output of :meth:`load` (must have timestamp, ltp, volume columns).
        freq : str
            Bar frequency: ``1s, 5s, 30s, 1min, 5min, 15min, 30min, 1h``.

        Returns
        -------
        pd.DataFrame
            Columns: datetime, open, high, low, close, volume, vwap,
            trade_count, oi.  Indexed by datetime.
        """
        freq = _VALID_FREQS.get(freq, freq)

        if ticks.empty:
            return pd.DataFrame(
                columns=[
                    "datetime", "open", "high", "low", "close",
                    "volume", "vwap", "trade_count", "oi",
                ]
            )

        df = ticks.copy()
        df = df.set_index("timestamp").sort_index()

        # Compute per-tick trade volume (volume is cumulative in Zerodha data)
        # Delta volume: difference from previous tick.  First tick of day gets 0.
        df["tick_volume"] = df["volume"].diff().clip(lower=0).fillna(0)

        # Resample
        bars = pd.DataFrame()
        bars["open"] = df["ltp"].resample(freq).first()
        bars["high"] = df["ltp"].resample(freq).max()
        bars["low"] = df["ltp"].resample(freq).min()
        bars["close"] = df["ltp"].resample(freq).last()
        bars["trade_count"] = df["ltp"].resample(freq).count()

        # Volume: sum of per-tick deltas within bar
        bars["volume"] = df["tick_volume"].resample(freq).sum()

        # VWAP: tick-volume-weighted average price within bar
        df["_pv"] = df["ltp"] * df["tick_volume"]
        pv_sum = df["_pv"].resample(freq).sum()
        vol_sum = df["tick_volume"].resample(freq).sum()
        bars["vwap"] = np.where(vol_sum > 0, pv_sum / vol_sum, bars["close"])

        # OI: last OI value in bar
        bars["oi"] = df["oi"].resample(freq).last()

        # Drop bars with no ticks
        bars = bars.dropna(subset=["open"])
        bars.index.name = "datetime"

        return bars

    # ------------------------------------------------------------------
    # Instrument mapping
    # ------------------------------------------------------------------

    def resolve_symbol(
        self,
        instrument_token: int,
        date_str: str | None = None,
    ) -> str | None:
        """Map instrument_token → tradingsymbol using the instruments table.

        Parameters
        ----------
        instrument_token : int
            Zerodha instrument token.
        date_str : str, optional
            Date "YYYY-MM-DD" for the lookup.  Uses latest available if None.

        Returns
        -------
        str or None
            The tradingsymbol, or None if not found.
        """
        instruments = self._load_instruments(date_str)
        if instruments is None:
            return None

        match = instruments[instruments["instrument_token"] == instrument_token]
        if match.empty:
            return None
        return match.iloc[0]["tradingsymbol"]

    def resolve_token(
        self,
        tradingsymbol: str,
        date_str: str | None = None,
    ) -> int | None:
        """Map tradingsymbol → instrument_token.

        Parameters
        ----------
        tradingsymbol : str
            Full or partial symbol to search for (exact match).
        date_str : str, optional
            Date for lookup. Uses latest available if None.

        Returns
        -------
        int or None
        """
        instruments = self._load_instruments(date_str)
        if instruments is None:
            return None

        match = instruments[instruments["tradingsymbol"] == tradingsymbol]
        if match.empty:
            return None
        return int(match.iloc[0]["instrument_token"])

    def search_instruments(
        self,
        pattern: str,
        date_str: str | None = None,
        exchange: str | None = None,
        segment: str | None = None,
        instrument_type: str | None = None,
    ) -> pd.DataFrame:
        """Search instruments by partial tradingsymbol match.

        Parameters
        ----------
        pattern : str
            Substring to search in tradingsymbol (case-insensitive).
        date_str : str, optional
            Date for lookup.
        exchange : str, optional
            Filter by exchange (e.g. "NSE", "NFO").
        segment : str, optional
            Filter by segment (e.g. "INDICES", "NFO-FUT").
        instrument_type : str, optional
            Filter by type (e.g. "FUT", "CE", "PE", "EQ").

        Returns
        -------
        pd.DataFrame
            Matching rows from the instruments table.
        """
        instruments = self._load_instruments(date_str)
        if instruments is None:
            return pd.DataFrame()

        mask = instruments["tradingsymbol"].str.contains(
            pattern, case=False, na=False
        )
        if exchange is not None:
            mask &= instruments["exchange"] == exchange
        if segment is not None:
            mask &= instruments["segment"] == segment
        if instrument_type is not None:
            mask &= instruments["instrument_type"] == instrument_type

        return instruments[mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(self, date_str: str | None = None, sample_limit: int = 10) -> dict:
        """Run quick diagnostics on the tick data.

        Returns a dict with: total_dates, sample_dates (list of per-date stats),
        total_tokens, epoch_corruption_info.
        """
        dates = self.available_dates()
        result: dict = {
            "total_dates": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "sample_stats": [],
        }

        # Sample evenly across dates
        step = max(1, len(dates) // sample_limit)
        sample_dates = dates[::step][:sample_limit]

        for d in sample_dates:
            d_dir = self.tick_dir / f"date={d}"
            parquet_files = list(d_dir.glob("*.parquet"))
            if not parquet_files:
                continue

            df = pq.read_table(parquet_files[0]).to_pandas()
            is_bad = df["timestamp"] <= _EPOCH_CUTOFF
            good = df[~is_bad]

            stat = {
                "date": d,
                "total_rows": len(df),
                "epoch_rows": int(is_bad.sum()),
                "epoch_pct": round(is_bad.mean() * 100, 4),
                "unique_tokens": int(df["instrument_token"].nunique()),
                "tokens_with_volume": int(
                    (good.groupby("instrument_token")["volume"].max() > 0).sum()
                ) if len(good) > 0 else 0,
                "ts_min": str(good["timestamp"].min()) if len(good) > 0 else None,
                "ts_max": str(good["timestamp"].max()) if len(good) > 0 else None,
            }
            result["sample_stats"].append(stat)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_partitions(
        self,
        start_date: str | None,
        end_date: str | None,
    ) -> list[Path]:
        """Return list of partition directories matching the date range."""
        all_dirs = sorted(
            d for d in self.tick_dir.iterdir()
            if d.is_dir() and d.name.startswith("date=")
        )

        if start_date is None and end_date is None:
            return all_dirs

        result = []
        for d_dir in all_dirs:
            d_str = d_dir.name[5:]  # "date=YYYY-MM-DD" → "YYYY-MM-DD"
            if start_date and d_str < start_date:
                continue
            if end_date and d_str > end_date:
                continue
            result.append(d_dir)

        return result

    def _load_instruments(self, date_str: str | None = None) -> pd.DataFrame | None:
        """Load instrument master for a date (cached)."""
        if date_str is None:
            # Use latest available date
            inst_dates = sorted(
                d.name[5:]
                for d in self.instrument_dir.iterdir()
                if d.is_dir() and d.name.startswith("date=")
            )
            if not inst_dates:
                logger.warning("No instrument data found in %s", self.instrument_dir)
                return None
            date_str = inst_dates[-1]

        if date_str in self._instrument_cache:
            return self._instrument_cache[date_str]

        inst_dir = self.instrument_dir / f"date={date_str}"
        if not inst_dir.exists():
            # Fall back to nearest available date
            all_dates = sorted(
                d.name[5:]
                for d in self.instrument_dir.iterdir()
                if d.is_dir() and d.name.startswith("date=")
            )
            if not all_dates:
                return None
            # Find nearest date <= date_str, or earliest available
            candidates = [d for d in all_dates if d <= date_str]
            if candidates:
                date_str = candidates[-1]
            else:
                date_str = all_dates[0]
            inst_dir = self.instrument_dir / f"date={date_str}"

        parquet_files = list(inst_dir.glob("*.parquet"))
        if not parquet_files:
            return None

        df = pq.read_table(parquet_files[0]).to_pandas()
        self._instrument_cache[date_str] = df
        logger.debug("Loaded %d instruments for %s", len(df), date_str)
        return df
