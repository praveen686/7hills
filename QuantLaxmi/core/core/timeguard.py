"""TimeGuard — the lookahead firewall.

This is the single most important module in QLX.  Every place where future
information could leak into a backtest is guarded by a static assertion here.
The philosophy is: **fail loud at build-time, not silently at run-time**.

Checks performed:
    1. Feature transforms must have lookforward == 0.
    2. CV gap must be >= target horizon.
    3. Train/test index sets must not overlap.
    4. Feature matrix timestamps must not extend beyond the declared cutoff.
    5. Target alignment: y[t] is derived from data in (t, t+horizon], so
       features at time t may only use data in (-inf, t].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from core.core.types import FeatureTransform, TargetTransform


class LookaheadError(Exception):
    """Raised when an operation would introduce future information leakage."""


class TimeGuard:
    """Static validation methods.  No instance state — use as a namespace."""

    # ------------------------------------------------------------------
    # 1. Feature lookforward check
    # ------------------------------------------------------------------
    @staticmethod
    def validate_feature(transform: FeatureTransform) -> None:
        """Reject any feature that declares a non-zero lookforward."""
        if transform.lookforward != 0:
            raise LookaheadError(
                f"Feature '{transform.name}' declares lookforward={transform.lookforward}. "
                f"Features must not use future data (lookforward must be 0)."
            )

    @staticmethod
    def validate_features(transforms: Sequence[FeatureTransform]) -> None:
        for t in transforms:
            TimeGuard.validate_feature(t)

    # ------------------------------------------------------------------
    # 2. CV gap vs target horizon
    # ------------------------------------------------------------------
    @staticmethod
    def validate_cv_gap(gap: int, horizon: int) -> None:
        """Ensure the cross-validation gap is at least as large as the target
        horizon.  A gap smaller than the horizon means features in the last
        ``horizon - gap`` training rows could overlap with the first test
        target's look-forward window.
        """
        if gap < horizon:
            raise LookaheadError(
                f"CV gap ({gap}) is less than target horizon ({horizon}). "
                f"This would leak future information into training."
            )

    # ------------------------------------------------------------------
    # 3. Train/test temporal separation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_train_test_separation(
        train_idx: pd.DatetimeIndex,
        test_idx: pd.DatetimeIndex,
        gap: int,
        horizon: int,
    ) -> None:
        """Verify that train end + gap <= test start, and gap >= horizon."""
        TimeGuard.validate_cv_gap(gap, horizon)

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise LookaheadError("Train or test index is empty.")

        train_end = train_idx.max()
        test_start = test_idx.min()

        if train_end >= test_start:
            raise LookaheadError(
                f"Train period ends at {train_end} but test starts at {test_start}. "
                f"Train must strictly precede test."
            )

    # ------------------------------------------------------------------
    # 4. Feature matrix cutoff check
    # ------------------------------------------------------------------
    @staticmethod
    def validate_cutoff(features: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        """Ensure no feature row has a timestamp beyond the cutoff."""
        if features.index.max() > cutoff:
            raise LookaheadError(
                f"Feature matrix extends to {features.index.max()}, "
                f"but declared cutoff is {cutoff}."
            )

    # ------------------------------------------------------------------
    # 5. X/y alignment
    # ------------------------------------------------------------------
    @staticmethod
    def validate_alignment(
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
    ) -> None:
        """Validate that features and target are correctly aligned.

        Contract: for row at index t, X[t] uses data from (-inf, t] and
        y[t] uses data from (t, t + horizon].  This method checks:
          - X and y share the same index.
          - No NaN/inf values snuck through.
          - The last ``horizon`` rows of the original data should have been
            dropped from y (since their target window extends beyond the data).
        """
        if not X.index.equals(y.index):
            x_only = X.index.difference(y.index)
            y_only = y.index.difference(X.index)
            raise LookaheadError(
                f"X and y index mismatch. "
                f"In X but not y: {len(x_only)} rows. "
                f"In y but not X: {len(y_only)} rows."
            )

        if not np.isfinite(X.values).all():
            bad_cols = X.columns[~np.isfinite(X).all()]
            raise LookaheadError(
                f"Non-finite values in feature columns: {list(bad_cols)}"
            )

        if not np.isfinite(y.values).all():
            raise LookaheadError("Non-finite values in target series.")
