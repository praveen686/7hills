"""Data loaders — normalise heterogeneous sources into canonical OHLCV.

Every loader returns an ``OHLCV`` object with:
  - DatetimeIndex (UTC, monotonically increasing)
  - Columns: Open, High, Low, Close, Volume (at minimum)

No global state is mutated.  No warnings are suppressed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.base.types import OHLCV

# ---------------------------------------------------------------------------
# Column name mappings for common data sources
# ---------------------------------------------------------------------------
_BINANCE_KLINE_COLS = {
    "Open time": "Open_time",
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume",
    "Close time": "Close_time",
    "Quote asset volume": "Quote volume",
    "Number of trades": "Trade count",
    "Taker buy base asset volume": "Taker base volume",
    "Taker buy quote asset volume": "Taker quote volume",
}


def from_dataframe(
    df: pd.DataFrame,
    time_col: str | None = None,
    col_map: dict[str, str] | None = None,
) -> OHLCV:
    """Wrap an existing DataFrame as OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.  Not mutated.
    time_col : str, optional
        Column to use as DatetimeIndex.  If *None*, the existing index is used.
    col_map : dict, optional
        Rename columns to canonical OHLCV names.
    """
    out = df.copy()

    if col_map:
        out = out.rename(columns=col_map)

    if time_col and time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col])
        out = out.set_index(time_col)

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    out = out.sort_index()

    return OHLCV(out)


def from_csv(
    path: str | Path,
    time_col: str | None = None,
    col_map: dict[str, str] | None = None,
    **csv_kwargs,
) -> OHLCV:
    """Load OHLCV from a CSV file.

    Handles Binance kline format automatically if the first column is
    ``Open time`` (millisecond epoch).
    """
    path = Path(path)
    df = pd.read_csv(path, **csv_kwargs)

    # Auto-detect Binance kline format
    if "Open time" in df.columns and df["Open time"].dtype in ("int64", "float64"):
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        if time_col is None:
            time_col = "Open time"

    return from_dataframe(df, time_col=time_col, col_map=col_map)
