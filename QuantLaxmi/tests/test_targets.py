"""Tests for target transforms — FutureReturn and TripleBarrier."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quantlaxmi.core.targets import FutureReturn, TripleBarrier


class TestFutureReturn:
    def test_correct_length(self, sample_ohlcv):
        target = FutureReturn(horizon=10)
        y = target.transform(sample_ohlcv)
        # Should drop last 10 rows (future is unknown)
        assert len(y) == len(sample_ohlcv) - 10

    def test_no_nan(self, sample_ohlcv):
        target = FutureReturn(horizon=10)
        y = target.transform(sample_ohlcv)
        assert not y.isnull().any()

    def test_horizon_1(self, sample_ohlcv):
        """With horizon=1, target should be next-bar return."""
        target = FutureReturn(horizon=1)
        y = target.transform(sample_ohlcv)
        close = sample_ohlcv.close
        expected = (close.shift(-1) / close - 1).dropna()
        # Compare values
        np.testing.assert_allclose(y.values, expected.values, rtol=1e-10)

    def test_smooth_emits_warning(self, sample_ohlcv):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            target = FutureReturn(horizon=10, smooth=True)
            _ = target.transform(sample_ohlcv)
            assert len(w) == 1
            assert "autocorrelation" in str(w[0].message).lower()

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            FutureReturn(horizon=0)

    def test_does_not_mutate_input(self, sample_ohlcv):
        original_len = len(sample_ohlcv)
        target = FutureReturn(horizon=10)
        _ = target.transform(sample_ohlcv)
        assert len(sample_ohlcv) == original_len


class TestTripleBarrier:
    def test_output_length(self, sample_ohlcv):
        tb = TripleBarrier(horizon=20, atr_period=14)
        y = tb.transform(sample_ohlcv)
        # Should be shorter due to ATR warmup + horizon at end
        assert len(y) < len(sample_ohlcv)
        assert len(y) > 0

    def test_label_values(self, sample_ohlcv):
        tb = TripleBarrier(horizon=20, atr_period=14)
        y = tb.transform(sample_ohlcv)
        assert set(y.unique()).issubset({-1, 0, 1})

    def test_no_nan(self, sample_ohlcv):
        tb = TripleBarrier(horizon=20, atr_period=14)
        y = tb.transform(sample_ohlcv)
        assert not y.isnull().any()

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError):
            TripleBarrier(horizon=0)
