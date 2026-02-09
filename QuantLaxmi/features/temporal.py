"""Temporal features with cyclical encoding.

Sigma used raw integer features for hour/day/month.  This is incorrect for
tree models (they can handle it) but catastrophic for linear models: the
model sees hour 23 and hour 0 as maximally distant when they are 1 hour
apart.  Cyclical sin/cos encoding preserves the circular topology.

We also deliberately exclude ``year`` as a feature.  Year is a monotonically
increasing integer with no periodicity — a model trained on 2021-2023 has
never seen 2024 and the feature can only hurt out-of-sample performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.base import Feature


@dataclass(frozen=True)
class CyclicalTime(Feature):
    """Cyclical sin/cos encoding of hour-of-day, day-of-week, day-of-month."""

    @property
    def name(self) -> str:
        return "time"

    @property
    def lookback(self) -> int:
        return 0  # purely point-in-time

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        out = pd.DataFrame(index=idx)

        # Hour of day (period = 24)
        hour = idx.hour + idx.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day of week (period = 7)
        dow = idx.dayofweek.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # Day of month (period = ~30.44)
        dom = idx.day.astype(float)
        out["dom_sin"] = np.sin(2 * np.pi * dom / 30.44)
        out["dom_cos"] = np.cos(2 * np.pi * dom / 30.44)

        # Month of year (period = 12)
        month = idx.month.astype(float)
        out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

        return out
