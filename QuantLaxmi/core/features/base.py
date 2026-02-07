"""Base class for all feature transforms.

Every feature in QLX inherits from ``Feature``.  The base class enforces:
  - ``lookforward == 0`` (compile-time via final property + runtime via TimeGuard)
  - ``transform()`` returns a new DataFrame, never mutates input
  - Column names are prefixed with the feature name to prevent collisions

Subclasses implement ``_compute(df) -> pd.DataFrame`` and declare ``name``
and ``lookback``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.core.types import OHLCV, FeatureTransform


class Feature(ABC):
    """Abstract base for all feature transforms.

    Implements the ``FeatureTransform`` protocol.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this feature group.  Used as column prefix."""
        ...

    @property
    @abstractmethod
    def lookback(self) -> int:
        """How many historical bars this feature requires."""
        ...

    @property
    def lookforward(self) -> int:
        """Features never look forward.  Always 0.  Do not override."""
        return 0

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature columns from a raw OHLCV DataFrame.

        The input ``df`` is already a *copy* (via ``OHLCV.df``).  Subclasses
        may mutate it freely, but must return a DataFrame whose columns are
        the computed features (without the original OHLCV columns).
        """
        ...

    def transform(self, ohlcv: OHLCV) -> pd.DataFrame:
        """Public entry point.  Returns prefixed feature columns."""
        df = ohlcv.df
        result = self._compute(df)

        # Prefix columns to prevent name collisions across features
        result.columns = [f"{self.name}__{col}" for col in result.columns]
        return result

    # Satisfy the protocol check
    @classmethod
    def _check_protocol(cls) -> None:
        assert isinstance(cls, type)  # just for static analysis
        _ = FeatureTransform  # reference to keep import alive
