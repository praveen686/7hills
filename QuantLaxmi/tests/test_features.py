"""Tests for the feature system — transforms, matrix builder, and composition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.core.base.timeguard import LookaheadError
from quantlaxmi.features import (
    RSI,
    ATR,
    BollingerBands,
    CyclicalTime,
    FeatureMatrix,
    HistoricalReturns,
    Momentum,
    Stochastic,
    SuperTrend,
)


class TestRSI:
    def test_output_shape(self, sample_ohlcv):
        rsi = RSI(window=14)
        result = rsi.transform(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert all(col.startswith("rsi_14__") for col in result.columns)

    def test_rsi_bounds(self, sample_ohlcv):
        rsi = RSI(window=14)
        result = rsi.transform(sample_ohlcv)
        values = result.iloc[:, 0].dropna()  # rsi value column
        assert (values >= 0).all()
        assert (values <= 100).all()

    def test_lookforward_is_zero(self):
        assert RSI(window=14).lookforward == 0


class TestBollingerBands:
    def test_output_columns(self, sample_ohlcv):
        bb = BollingerBands(window=20)
        result = bb.transform(sample_ohlcv)
        assert "bb_20__bandwidth" in result.columns
        assert "bb_20__pct_b" in result.columns


class TestSuperTrend:
    def test_direction_values(self, sample_ohlcv):
        st = SuperTrend(period=14, multiplier=3.0)
        result = st.transform(sample_ohlcv)
        direction = result["supertrend_14__direction"].dropna()
        assert len(direction) > 0
        assert set(direction.unique()).issubset({-1.0, 1.0})


class TestStochastic:
    def test_fast_k_bounds(self, sample_ohlcv):
        stoch = Stochastic(k_window=14, d_window=3)
        result = stoch.transform(sample_ohlcv)
        fast_k = result.iloc[:, 0].dropna()
        assert (fast_k >= 0).all()
        assert (fast_k <= 100).all()


class TestATR:
    def test_positive_values(self, sample_ohlcv):
        atr = ATR(window=14)
        result = atr.transform(sample_ohlcv)
        values = result.iloc[:, 0].dropna()
        assert (values >= 0).all()


class TestHistoricalReturns:
    def test_all_periods_present(self, sample_ohlcv):
        hr = HistoricalReturns(periods=(1, 5, 10))
        result = hr.transform(sample_ohlcv)
        assert "hist_ret__pct_1" in result.columns
        assert "hist_ret__pct_5" in result.columns
        assert "hist_ret__pct_10" in result.columns


class TestCyclicalTime:
    def test_sin_cos_bounds(self, sample_ohlcv):
        ct = CyclicalTime()
        result = ct.transform(sample_ohlcv)
        for col in result.columns:
            values = result[col].dropna()
            assert (values >= -1.0).all()
            assert (values <= 1.0).all()

    def test_lookback_is_zero(self):
        assert CyclicalTime().lookback == 0


class TestFeatureMatrix:
    def test_builds_successfully(self, sample_ohlcv):
        matrix = (
            FeatureMatrix(sample_ohlcv)
            .add(RSI(window=14))
            .add(BollingerBands(window=20))
            .add(HistoricalReturns(periods=(1, 5, 10)))
            .add(CyclicalTime())
        )
        result = matrix.build()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert np.isfinite(result.values).all()

    def test_no_nan_after_build(self, sample_ohlcv):
        result = (
            FeatureMatrix(sample_ohlcv)
            .add(RSI(window=14))
            .build()
        )
        assert not result.isnull().any().any()

    def test_immutable_builder(self, sample_ohlcv):
        base = FeatureMatrix(sample_ohlcv)
        with_rsi = base.add(RSI(window=14))
        with_both = with_rsi.add(BollingerBands(window=20))

        assert len(base._transforms) == 0
        assert len(with_rsi._transforms) == 1
        assert len(with_both._transforms) == 2

    def test_duplicate_name_raises(self, sample_ohlcv):
        with pytest.raises(ValueError, match="Duplicate"):
            FeatureMatrix(sample_ohlcv).add(RSI(window=14)).add(RSI(window=14))

    def test_different_params_ok(self, sample_ohlcv):
        matrix = (
            FeatureMatrix(sample_ohlcv)
            .add(RSI(window=14))
            .add(RSI(window=28))  # different window -> different name
        )
        result = matrix.build()
        assert "rsi_14__value" in result.columns
        assert "rsi_28__value" in result.columns

    def test_empty_build_raises(self, sample_ohlcv):
        with pytest.raises(ValueError, match="No features"):
            FeatureMatrix(sample_ohlcv).build()

    def test_all_features_together(self, sample_ohlcv):
        """Smoke test: all feature types compose correctly."""
        result = (
            FeatureMatrix(sample_ohlcv)
            .add(RSI(window=14))
            .add(BollingerBands(window=20))
            .add(SuperTrend(period=14, multiplier=3.0))
            .add(Stochastic(k_window=14, d_window=3))
            .add(ATR(window=14))
            .add(HistoricalReturns(periods=(1, 5, 10, 20)))
            .add(Momentum(fast=10, slow=50, run_window=10))
            .add(CyclicalTime())
            .build()
        )
        assert result.shape[1] > 20  # many columns
        assert np.isfinite(result.values).all()
