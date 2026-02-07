"""Core type definitions and protocols for QLX.

Every transform in the system implements one of these protocols.  The protocols
encode the *contract* that prevents lookahead bias: features must have
lookforward == 0, targets declare their horizon explicitly, and the pipeline
engine uses these declarations to enforce safe cross-validation gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names — every loader must normalise to these.
# ---------------------------------------------------------------------------
OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class OHLCV:
    """Immutable wrapper around an OHLCV DataFrame.

    The wrapped DataFrame is never mutated.  All downstream transforms
    receive a *copy* via ``.df`` property, and the original is stored in
    ``_data``.
    """

    _data: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in OHLCV_COLUMNS if c not in self._data.columns]
        if missing:
            raise ValueError(f"OHLCV data missing columns: {missing}")
        if not isinstance(self._data.index, pd.DatetimeIndex):
            raise ValueError("OHLCV index must be a DatetimeIndex")
        if not self._data.index.is_monotonic_increasing:
            raise ValueError("OHLCV index must be monotonically increasing")

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy — callers may mutate freely without affecting source."""
        return self._data.copy()

    @property
    def close(self) -> pd.Series:
        return self._data["Close"].copy()

    @property
    def high(self) -> pd.Series:
        return self._data["High"].copy()

    @property
    def low(self) -> pd.Series:
        return self._data["Low"].copy()

    @property
    def open(self) -> pd.Series:
        return self._data["Open"].copy()

    @property
    def volume(self) -> pd.Series:
        return self._data["Volume"].copy()

    def __len__(self) -> int:
        return len(self._data)

    def slice(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> OHLCV:
        return OHLCV(self._data.loc[start:end].copy())


# ---------------------------------------------------------------------------
# Transform protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class FeatureTransform(Protocol):
    """Contract for feature generators.

    ``lookforward`` **must** return 0.  If a feature ever needs future data it
    is not a feature — it is a target or a label, and must implement
    ``TargetTransform`` instead.  ``TimeGuard`` rejects any FeatureTransform
    with lookforward != 0.
    """

    @property
    def name(self) -> str: ...

    @property
    def lookback(self) -> int: ...

    @property
    def lookforward(self) -> int: ...

    def transform(self, ohlcv: OHLCV) -> pd.DataFrame: ...


@runtime_checkable
class TargetTransform(Protocol):
    """Contract for target / label generators.

    ``horizon`` is the number of future bars the target peeks into.  The
    pipeline engine uses this to set the minimum CV gap.
    """

    @property
    def name(self) -> str: ...

    @property
    def horizon(self) -> int: ...

    def transform(self, ohlcv: OHLCV) -> pd.Series: ...
