"""FeatureMatrix — composable builder for feature DataFrames.

Usage::

    matrix = (
        FeatureMatrix(ohlcv)
        .add(RSI(window=14))
        .add(BollingerBands(window=20))
        .add(SuperTrend(period=14, multiplier=3))
        .add(HistoricalReturns(periods=(1, 5, 10, 20, 60)))
        .add(CyclicalTime())
        .build()
    )

Each ``.add()`` validates the feature via TimeGuard and records it.
``.build()`` materialises all features into a single DataFrame, drops
rows with NaN/inf (warmup period), and returns clean aligned data.

The builder is *additive*: calling ``.add()`` returns a **new**
FeatureMatrix with the additional feature appended (the original is
unchanged).  This makes the builder safe to pass around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qlx.core.timeguard import TimeGuard
from qlx.core.types import OHLCV, FeatureTransform
from qlx.features.base import Feature


@dataclass(frozen=True)
class FeatureMatrix:
    """Immutable builder for a feature matrix."""

    ohlcv: OHLCV
    _transforms: tuple[Feature, ...] = ()

    def add(self, feature: Feature) -> FeatureMatrix:
        """Return a new builder with ``feature`` appended."""
        TimeGuard.validate_feature(feature)

        # Check for duplicate feature names
        existing = {t.name for t in self._transforms}
        if feature.name in existing:
            raise ValueError(
                f"Duplicate feature name '{feature.name}'. "
                f"Use different parameters to create distinct features."
            )

        return FeatureMatrix(
            ohlcv=self.ohlcv,
            _transforms=(*self._transforms, feature),
        )

    @property
    def max_lookback(self) -> int:
        """Maximum lookback across all registered features."""
        if not self._transforms:
            return 0
        return max(t.lookback for t in self._transforms)

    @property
    def feature_names(self) -> list[str]:
        """List of all registered feature group names."""
        return [t.name for t in self._transforms]

    def build(self, dropna: bool = True) -> pd.DataFrame:
        """Materialise features into a single DataFrame.

        Parameters
        ----------
        dropna : bool
            If True (default), drop rows that contain any NaN or inf.
            These come from the warmup period of lookback-based features.

        Returns
        -------
        pd.DataFrame
            Clean feature matrix with DatetimeIndex.
        """
        if not self._transforms:
            raise ValueError("No features registered.  Call .add() first.")

        frames = []
        for t in self._transforms:
            result = t.transform(self.ohlcv)
            frames.append(result)

        combined = pd.concat(frames, axis=1)

        if dropna:
            combined = combined.replace([np.inf, -np.inf], np.nan)
            combined = combined.dropna()

        # Final dtype check: everything should be numeric
        non_numeric = combined.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(
                f"Non-numeric feature columns detected: {list(non_numeric)}. "
                f"All features must be numeric."
            )

        return combined
