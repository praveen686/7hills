"""
Combinatorial Purged K-Fold Cross-Validation (CPCV).

Reference: Lopez de Prado, *Advances in Financial Machine Learning*,
Chapter 12 — Back-testing through Cross-Validation.

Standard K-Fold CV leaks information through two channels:
  1. Training samples whose labels overlap with test-set event windows.
  2. Serial correlation in financial time series.

CPCV addresses both by:
  - Purging: removing training observations whose evaluation window
    overlaps with any test observation (within ``purge_window``).
  - Embargo: additionally removing a percentage of training observations
    immediately after each test set boundary (``embargo_pct``).
  - Combinatorial paths: instead of a single train/test split per fold,
    CPCV enumerates all C(n_splits, n_test_groups) ways to select
    ``n_test_groups`` contiguous folds as the test set, producing far
    more backtest paths and enabling distribution-of-returns analysis.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator


class CombPurgedKFoldCV(BaseCrossValidator):
    """Combinatorial Purged K-Fold cross-validator.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds (groups) to split the data into.
    n_test_groups : int, default 2
        Number of folds to use as the test set in each split.
        Total number of splits = C(n_splits, n_test_groups).
    purge_window : int, default 0
        Number of rows to purge from the training set on each side
        of the test boundaries.  Rows within ``purge_window`` of any
        test-set row are removed from training.
    embargo_pct : float, default 0.0
        Fraction of total dataset length to embargo *after* each
        test-set boundary.  This is an additional buffer beyond
        ``purge_window`` to account for serial correlation.

    Notes
    -----
    When ``n_test_groups == 1`` this reduces to standard purged K-Fold
    (with embargo).  Increasing ``n_test_groups`` multiplies the number
    of paths but shrinks each training set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_window: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2, got %d" % n_splits)
        if n_test_groups < 1 or n_test_groups >= n_splits:
            raise ValueError(
                "n_test_groups must be in [1, n_splits), got %d" % n_test_groups
            )
        if purge_window < 0:
            raise ValueError("purge_window must be >= 0, got %d" % purge_window)
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1), got %.4f" % embargo_pct)

        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def get_n_splits(
        self,
        X: Optional[object] = None,
        y: Optional[object] = None,
        groups: Optional[object] = None,
    ) -> int:
        """Return the number of splitting iterations (combinatorial paths).

        Equal to C(n_splits, n_test_groups).
        """
        from math import comb

        return comb(self.n_splits, self.n_test_groups)

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray | pd.Series] = None,
        groups: Optional[np.ndarray | pd.Series] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index arrays for each combinatorial split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.  Only ``len(X)`` is used.
        y : ignored
        groups : ignored

        Yields
        ------
        train_indices : np.ndarray[int]
            Indices for training (with purging + embargo applied).
        test_indices : np.ndarray[int]
            Indices for testing.
        """
        n_samples = len(X)
        if n_samples < self.n_splits:
            raise ValueError(
                "Cannot have n_splits=%d > n_samples=%d"
                % (self.n_splits, n_samples)
            )

        # --- 1. Partition indices into n_splits roughly-equal folds ----
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        folds: list[np.ndarray] = []
        current = 0
        for size in fold_sizes:
            folds.append(indices[current : current + size])
            current += size

        embargo_size = int(n_samples * self.embargo_pct)

        # --- 2. Enumerate all C(k, t) combinations ----------------------
        for test_combo in combinations(range(self.n_splits), self.n_test_groups):
            test_indices = np.concatenate([folds[i] for i in test_combo])
            test_indices.sort()

            # Start with all non-test indices
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # --- 3. Purge: remove training rows within purge_window -----
            if self.purge_window > 0:
                test_min = int(test_indices.min())
                test_max = int(test_indices.max())
                purge_lo = max(0, test_min - self.purge_window)
                purge_hi = min(n_samples - 1, test_max + self.purge_window)
                train_mask[purge_lo:test_min] = False
                train_mask[test_max + 1 : purge_hi + 1] = False

            # --- 4. Embargo: remove rows right after each test block ----
            if embargo_size > 0:
                # Identify contiguous blocks of test indices
                blocks = self._contiguous_blocks(test_indices)
                for block in blocks:
                    block_end = int(block[-1])
                    emb_start = block_end + 1
                    emb_end = min(n_samples, emb_start + embargo_size)
                    train_mask[emb_start:emb_end] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contiguous_blocks(sorted_arr: np.ndarray) -> list[np.ndarray]:
        """Split a sorted 1-D array into contiguous blocks."""
        if len(sorted_arr) == 0:
            return []
        breaks = np.where(np.diff(sorted_arr) != 1)[0] + 1
        return np.split(sorted_arr, breaks)
