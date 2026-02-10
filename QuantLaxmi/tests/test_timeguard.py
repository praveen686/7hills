"""Tests for the TimeGuard lookahead firewall.

These tests prove that the safety system catches the exact classes of bugs
that plagued sigma: lookahead in features, insufficient CV gaps, and
misaligned X/y indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.core.base.timeguard import LookaheadError, TimeGuard
from quantlaxmi.core.base.types import OHLCV


# ---------------------------------------------------------------------------
# Fake transforms for testing
# ---------------------------------------------------------------------------

class GoodFeature:
    name = "good"
    lookback = 10
    lookforward = 0

    def transform(self, ohlcv):
        return pd.DataFrame({"x": np.ones(len(ohlcv))}, index=ohlcv.df.index)


class BadFeature:
    """A feature that illegally looks forward."""

    name = "bad"
    lookback = 10
    lookforward = 5  # THIS IS THE BUG

    def transform(self, ohlcv):
        return pd.DataFrame({"x": np.ones(len(ohlcv))}, index=ohlcv.df.index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateFeature:
    def test_good_feature_passes(self):
        TimeGuard.validate_feature(GoodFeature())

    def test_bad_feature_raises(self):
        with pytest.raises(LookaheadError, match="lookforward=5"):
            TimeGuard.validate_feature(BadFeature())

    def test_validate_features_batch(self):
        with pytest.raises(LookaheadError):
            TimeGuard.validate_features([GoodFeature(), BadFeature()])


class TestValidateCVGap:
    def test_gap_equals_horizon_passes(self):
        TimeGuard.validate_cv_gap(gap=150, horizon=150)

    def test_gap_exceeds_horizon_passes(self):
        TimeGuard.validate_cv_gap(gap=200, horizon=150)

    def test_gap_less_than_horizon_raises(self):
        with pytest.raises(LookaheadError, match="gap.*less than.*horizon"):
            TimeGuard.validate_cv_gap(gap=100, horizon=150)

    def test_zero_gap_zero_horizon_passes(self):
        # Edge case: point-in-time targets
        TimeGuard.validate_cv_gap(gap=0, horizon=0)


class TestValidateTrainTestSeparation:
    def test_proper_separation_passes(self):
        train = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=100, freq="h"))
        test = pd.DatetimeIndex(pd.date_range("2023-01-10", periods=50, freq="h"))
        TimeGuard.validate_train_test_separation(train, test, gap=10, horizon=10)

    def test_overlapping_raises(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="h")
        train = pd.DatetimeIndex(dates[:60])
        test = pd.DatetimeIndex(dates[50:80])  # overlaps!
        with pytest.raises(LookaheadError, match="strictly precede"):
            TimeGuard.validate_train_test_separation(train, test, gap=0, horizon=0)

    def test_empty_train_raises(self):
        test = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=10, freq="h"))
        with pytest.raises(LookaheadError, match="empty"):
            TimeGuard.validate_train_test_separation(
                pd.DatetimeIndex([]), test, gap=0, horizon=0
            )


class TestValidateAlignment:
    def test_matching_indices_pass(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="h")
        X = pd.DataFrame({"a": np.ones(50), "b": np.zeros(50)}, index=idx)
        y = pd.Series(np.ones(50), index=idx)
        TimeGuard.validate_alignment(X, y, horizon=10)

    def test_mismatched_indices_raise(self):
        idx_x = pd.date_range("2023-01-01", periods=50, freq="h")
        idx_y = pd.date_range("2023-01-02", periods=50, freq="h")
        X = pd.DataFrame({"a": np.ones(50)}, index=idx_x)
        y = pd.Series(np.ones(50), index=idx_y)
        with pytest.raises(LookaheadError, match="mismatch"):
            TimeGuard.validate_alignment(X, y, horizon=10)

    def test_inf_in_features_raises(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="h")
        X = pd.DataFrame({"a": [1, 2, np.inf, 4, 5, 6, 7, 8, 9, 10]}, index=idx)
        y = pd.Series(np.ones(10), index=idx)
        with pytest.raises(LookaheadError, match="Non-finite"):
            TimeGuard.validate_alignment(X, y, horizon=1)

    def test_nan_in_target_raises(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="h")
        X = pd.DataFrame({"a": np.ones(10)}, index=idx)
        y = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10], index=idx)
        with pytest.raises(LookaheadError, match="Non-finite.*target"):
            TimeGuard.validate_alignment(X, y, horizon=1)


class TestValidateCutoff:
    def test_within_cutoff_passes(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="h")
        X = pd.DataFrame({"a": np.ones(50)}, index=idx)
        TimeGuard.validate_cutoff(X, cutoff=pd.Timestamp("2023-01-10"))

    def test_beyond_cutoff_raises(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="h")
        X = pd.DataFrame({"a": np.ones(50)}, index=idx)
        with pytest.raises(LookaheadError, match="cutoff"):
            TimeGuard.validate_cutoff(X, cutoff=pd.Timestamp("2023-01-01 12:00"))
