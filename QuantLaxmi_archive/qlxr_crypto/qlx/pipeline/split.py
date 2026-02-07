"""Cross-validation splitters with mandatory lookahead gaps.

The gap is not optional.  You cannot create a splitter with gap < horizon.
This is enforced at construction time, not at run time — fail early.

Both splitters yield ``SplitResult`` named tuples that carry the fold
indices plus metadata (fold number, train/test sizes).  The pipeline
engine iterates these directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd

from qlx.core.timeguard import LookaheadError, TimeGuard


class SplitResult(NamedTuple):
    """One fold of a cross-validation split."""

    fold: int
    train_idx: np.ndarray  # integer positional indices
    test_idx: np.ndarray
    gap_idx: np.ndarray  # the embargo zone — neither train nor test


@dataclass(frozen=True)
class WalkForwardSplit:
    """Rolling (fixed-width) walk-forward cross-validation.

    Parameters
    ----------
    window : int
        Total width of train + gap + test.
    train_frac : float
        Fraction of window allocated to training (before the gap).
    gap : int
        Number of bars between end of train and start of test.
        Must be >= ``horizon``.
    horizon : int
        Target look-forward declared by the TargetTransform.
    step : int or None
        How far to advance the window between folds.  Defaults to
        test_size (non-overlapping test sets).
    """

    window: int
    train_frac: float
    gap: int
    horizon: int
    step: int | None = None

    def __post_init__(self) -> None:
        TimeGuard.validate_cv_gap(self.gap, self.horizon)

        train_size = int(self.window * self.train_frac)
        remaining = self.window - train_size - self.gap
        if remaining <= 0:
            raise ValueError(
                f"Window ({self.window}) is too small for train_frac={self.train_frac} "
                f"and gap={self.gap}.  No room for test data."
            )

    @property
    def train_size(self) -> int:
        return int(self.window * self.train_frac)

    @property
    def test_size(self) -> int:
        return self.window - self.train_size - self.gap

    def split(self, n_samples: int) -> Iterator[SplitResult]:
        """Generate fold indices for a dataset of length ``n_samples``."""
        step = self.step if self.step is not None else self.test_size
        fold = 0

        start = 0
        while start + self.window <= n_samples:
            train_end = start + self.train_size
            gap_end = train_end + self.gap
            test_end = start + self.window

            yield SplitResult(
                fold=fold,
                train_idx=np.arange(start, train_end),
                test_idx=np.arange(gap_end, test_end),
                gap_idx=np.arange(train_end, gap_end),
            )
            fold += 1
            start += step

    def n_splits(self, n_samples: int) -> int:
        step = self.step if self.step is not None else self.test_size
        return max(0, (n_samples - self.window) // step + 1)


@dataclass(frozen=True)
class ExpandingSplit:
    """Expanding (anchored) walk-forward cross-validation.

    Training set grows over time while test set size remains constant.

    Parameters
    ----------
    min_train : int
        Minimum training set size for the first fold.
    test_size : int
        Fixed test set size.
    gap : int
        Embargo bars between train end and test start.
    horizon : int
        Target horizon (gap must >= this).
    step : int or None
        How far to advance between folds.  Defaults to test_size.
    """

    min_train: int
    test_size: int
    gap: int
    horizon: int
    step: int | None = None

    def __post_init__(self) -> None:
        TimeGuard.validate_cv_gap(self.gap, self.horizon)

    def split(self, n_samples: int) -> Iterator[SplitResult]:
        step = self.step if self.step is not None else self.test_size
        fold = 0

        train_end = self.min_train
        while train_end + self.gap + self.test_size <= n_samples:
            gap_end = train_end + self.gap
            test_end = gap_end + self.test_size

            yield SplitResult(
                fold=fold,
                train_idx=np.arange(0, train_end),
                test_idx=np.arange(gap_end, test_end),
                gap_idx=np.arange(train_end, gap_end),
            )
            fold += 1
            train_end += step

    def n_splits(self, n_samples: int) -> int:
        step = self.step if self.step is not None else self.test_size
        first_possible = self.min_train + self.gap + self.test_size
        if first_possible > n_samples:
            return 0
        return (n_samples - first_possible) // step + 1
