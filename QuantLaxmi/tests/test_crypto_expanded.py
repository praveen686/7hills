"""Tests for expanded crypto feature sources.

Tests cover:
  - FundingRateRegime: mock 8h data, 7 features, z-scores, momentum
  - OpenInterestDynamics: mock OI data, z-scores, momentum, concentration
  - LongShortPositioning: mock L/S data, divergence computation
  - AltcoinBreadth: mock BTC/ETH/SOL klines, ratio, correlation, breadth
  - LiquidationProxy: mock OHLCV, cascade detection, volume spike
  - MegaFeatureBuilder integration: group 25 present
  - No look-ahead: all features use only past data
  - BinanceConnector new methods: verify existence and correct URLs
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.features.crypto_expanded import (
    AltcoinBreadth,
    FundingRateRegime,
    LiquidationProxy,
    LongShortPositioning,
    OpenInterestDynamics,
    build_crypto_expanded_features,
)


# ---------------------------------------------------------------------------
# Helpers -- generate mock data
# ---------------------------------------------------------------------------


def _make_funding_df(n_days: int = 30) -> pd.DataFrame:
    """Create mock 8-hourly funding rate data for BTC + ETH."""
    dates = pd.date_range("2025-06-01", periods=n_days * 3, freq="8h", tz="UTC")
    np.random.seed(42)
    rates = np.random.normal(0.0001, 0.0005, len(dates))
    symbols = ["BTCUSDT", "ETHUSDT"] * (len(dates) // 2)
    if len(symbols) < len(dates):
        symbols.append("BTCUSDT")
    symbols = symbols[: len(dates)]
    return pd.DataFrame(
        {"fundingRate": rates, "symbol": symbols, "markPrice": 50000.0},
        index=dates,
    )


def _make_oi_df(n_days: int = 60) -> pd.DataFrame:
    """Create mock daily OI data for BTC + ETH."""
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D", tz="UTC")
    np.random.seed(42)
    btc_oi = np.random.uniform(5e9, 15e9, n_days)
    eth_oi = np.random.uniform(2e9, 8e9, n_days)
    return pd.DataFrame(
        {"oi_btc": btc_oi, "oi_eth": eth_oi},
        index=dates,
    )


def _make_ls_df(n_days: int = 30, cols: str = "global") -> pd.DataFrame:
    """Create mock L/S ratio data."""
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D", tz="UTC")
    np.random.seed(42)
    if cols == "taker":
        return pd.DataFrame(
            {
                "buySellRatio": np.random.uniform(0.4, 0.6, n_days),
                "buyVol": np.random.uniform(1e8, 5e8, n_days),
                "sellVol": np.random.uniform(1e8, 5e8, n_days),
            },
            index=dates,
        )
    return pd.DataFrame(
        {
            "longShortRatio": np.random.uniform(0.8, 1.2, n_days),
            "longAccount": np.random.uniform(0.45, 0.55, n_days),
            "shortAccount": np.random.uniform(0.45, 0.55, n_days),
        },
        index=dates,
    )


def _make_ohlcv_df(n_days: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    """Create mock daily OHLCV data with guaranteed positive prices."""
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D", tz="UTC")
    np.random.seed(42)
    # Use geometric random walk to keep prices positive
    log_returns = np.random.normal(0, 0.01, n_days)
    close = base_price * np.exp(np.cumsum(log_returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    open_ = close * (1 + np.random.normal(0, 0.003, n_days))
    volume = np.random.uniform(1e4, 5e4, n_days)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


# ===========================================================================
# FundingRateRegime Tests
# ===========================================================================


class TestFundingRateRegime:
    """Tests for funding rate features."""

    def test_7_features_produced(self):
        """Verify all 7 expected funding features are produced."""
        df = _make_funding_df(30)
        result = FundingRateRegime().compute(df)

        expected_cols = [
            "fr_mean_8h", "fr_std_8h", "fr_skew", "fr_max_abs",
            "fr_z_score", "fr_momentum_3d", "fr_extreme_count",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_no_nan_for_sufficient_data(self):
        """With 30 days of data, features past warmup should not be all NaN."""
        df = _make_funding_df(60)
        result = FundingRateRegime().compute(df)

        # After 21 days warmup, at least some values should be non-NaN
        for col in ["fr_mean_8h", "fr_std_8h", "fr_extreme_count"]:
            non_nan = result[col].dropna()
            assert len(non_nan) > 0, f"{col} is all NaN"

    def test_z_score_centered(self):
        """Z-score should be approximately centered around 0."""
        df = _make_funding_df(90)
        result = FundingRateRegime().compute(df)
        z = result["fr_z_score"].dropna()
        assert abs(z.mean()) < 2.0, f"Z-score mean is {z.mean()}, expected near 0"

    def test_empty_input(self):
        """Empty input should return empty DataFrame."""
        result = FundingRateRegime().compute(pd.DataFrame())
        assert result.empty

    def test_none_input(self):
        """None input should return empty DataFrame."""
        result = FundingRateRegime().compute(None)
        assert result.empty

    def test_missing_column(self):
        """Missing fundingRate column should return empty."""
        df = pd.DataFrame({"other": [1, 2, 3]}, index=pd.date_range("2025-01-01", periods=3))
        result = FundingRateRegime().compute(df)
        assert result.empty

    def test_momentum_3d_lag(self):
        """Momentum should be NaN for first 3 days (causal)."""
        df = _make_funding_df(10)
        result = FundingRateRegime().compute(df)
        # First 3 rows of momentum should be NaN
        assert result["fr_momentum_3d"].iloc[:3].isna().all()


# ===========================================================================
# OpenInterestDynamics Tests
# ===========================================================================


class TestOpenInterestDynamics:
    """Tests for OI dynamics features."""

    def test_6_features_produced(self):
        """Verify all 6 expected OI features are produced."""
        df = _make_oi_df(60)
        result = OpenInterestDynamics().compute(df)

        expected_cols = [
            "oi_btc_usd_m", "oi_eth_usd_m", "oi_total_z21",
            "oi_momentum_5d", "oi_expanding", "oi_concentration",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_z_score_21d_warmup(self):
        """Z-score should be NaN for first 20 days."""
        df = _make_oi_df(60)
        result = OpenInterestDynamics().compute(df)
        assert result["oi_total_z21"].iloc[:20].isna().all()

    def test_concentration_bounded(self):
        """BTC concentration should be between 0 and 1."""
        df = _make_oi_df(60)
        result = OpenInterestDynamics().compute(df)
        conc = result["oi_concentration"].dropna()
        assert (conc >= 0).all() and (conc <= 1).all()

    def test_expanding_binary(self):
        """OI expanding should be 0 or 1."""
        df = _make_oi_df(60)
        result = OpenInterestDynamics().compute(df)
        expanding = result["oi_expanding"].dropna()
        assert set(expanding.unique()).issubset({0, 1})

    def test_momentum_causal(self):
        """5-day momentum should be NaN for first 5 days."""
        df = _make_oi_df(60)
        result = OpenInterestDynamics().compute(df)
        assert result["oi_momentum_5d"].iloc[:5].isna().all()

    def test_empty_input(self):
        result = OpenInterestDynamics().compute(pd.DataFrame())
        assert result.empty


# ===========================================================================
# LongShortPositioning Tests
# ===========================================================================


class TestLongShortPositioning:
    """Tests for L/S positioning features."""

    def test_global_features(self):
        """Global L/S ratio produces ratio + z-score."""
        global_ls = _make_ls_df(30, "global")
        result = LongShortPositioning().compute(global_ls=global_ls)

        assert "ls_ratio_global" in result.columns
        assert "ls_ratio_z_5d" in result.columns

    def test_top_ratio(self):
        """Top-trader ratio is present when provided."""
        top_ls = _make_ls_df(30, "global")
        result = LongShortPositioning().compute(top_ls=top_ls)

        assert "ls_top_ratio" in result.columns

    def test_divergence_computation(self):
        """Divergence = global - top when both provided."""
        global_ls = _make_ls_df(30, "global")
        top_ls = _make_ls_df(30, "global")
        # Make top ratios systematically lower
        top_ls["longShortRatio"] = top_ls["longShortRatio"] - 0.1

        result = LongShortPositioning().compute(
            global_ls=global_ls, top_ls=top_ls,
        )

        assert "ls_divergence" in result.columns
        # Divergence should be mostly positive (global > top)
        div = result["ls_divergence"].dropna()
        assert div.mean() > 0

    def test_taker_features(self):
        """Taker features present when taker data provided."""
        taker_ls = _make_ls_df(30, "taker")
        result = LongShortPositioning().compute(taker_ls=taker_ls)

        assert "ls_taker_buy_pct" in result.columns
        assert "ls_taker_z_5d" in result.columns
        assert "ls_taker_flip" in result.columns

    def test_taker_buy_pct_bounded(self):
        """Taker buy pct should be between 0 and 1."""
        taker_ls = _make_ls_df(30, "taker")
        result = LongShortPositioning().compute(taker_ls=taker_ls)
        pct = result["ls_taker_buy_pct"].dropna()
        assert (pct >= 0).all() and (pct <= 1).all()

    def test_all_none_returns_empty(self):
        """All None inputs should return empty DataFrame."""
        result = LongShortPositioning().compute()
        assert result.empty

    def test_taker_flip_values(self):
        """Taker flip should only be -1, 0, or 1."""
        taker_ls = _make_ls_df(30, "taker")
        result = LongShortPositioning().compute(taker_ls=taker_ls)
        flip = result["ls_taker_flip"].dropna()
        assert set(flip.unique()).issubset({-1, 0, 1})


# ===========================================================================
# AltcoinBreadth Tests
# ===========================================================================


class TestAltcoinBreadth:
    """Tests for altcoin breadth features."""

    def test_6_features_produced(self):
        """Verify all 6 altcoin breadth features."""
        btc = _make_ohlcv_df(60, 50000)
        eth = _make_ohlcv_df(60, 3000)
        sol = _make_ohlcv_df(60, 100)

        result = AltcoinBreadth().compute(btc, eth, sol)

        expected_cols = [
            "ab_eth_btc_ratio", "ab_eth_btc_z20", "ab_sol_momentum",
            "ab_altcoin_spread", "ab_correlation_btc_eth", "ab_breadth_2of3",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_eth_btc_ratio_positive(self):
        """ETH/BTC ratio should be positive."""
        btc = _make_ohlcv_df(60, 50000)
        eth = _make_ohlcv_df(60, 3000)

        result = AltcoinBreadth().compute(btc, eth)
        ratio = result["ab_eth_btc_ratio"].dropna()
        assert (ratio > 0).all()

    def test_correlation_bounded(self):
        """BTC-ETH correlation should be between -1 and 1."""
        btc = _make_ohlcv_df(60, 50000)
        eth = _make_ohlcv_df(60, 3000)

        result = AltcoinBreadth().compute(btc, eth)
        corr = result["ab_correlation_btc_eth"].dropna()
        assert (corr >= -1.0001).all() and (corr <= 1.0001).all()

    def test_breadth_range(self):
        """Breadth 2-of-3 should be 0, 1, 2, or 3."""
        btc = _make_ohlcv_df(60, 50000)
        eth = _make_ohlcv_df(60, 3000)
        sol = _make_ohlcv_df(60, 100)

        result = AltcoinBreadth().compute(btc, eth, sol)
        breadth = result["ab_breadth_2of3"].dropna()
        assert set(breadth.unique()).issubset({0, 1, 2, 3})

    def test_sol_none_degrades(self):
        """With SOL=None, SOL features are NaN but others work."""
        btc = _make_ohlcv_df(60, 50000)
        eth = _make_ohlcv_df(60, 3000)

        result = AltcoinBreadth().compute(btc, eth, sol_df=None)
        assert "ab_eth_btc_ratio" in result.columns
        assert result["ab_sol_momentum"].isna().all()

    def test_empty_btc_returns_empty(self):
        result = AltcoinBreadth().compute(pd.DataFrame(), _make_ohlcv_df(30, 3000))
        assert result.empty

    def test_empty_eth_returns_empty(self):
        result = AltcoinBreadth().compute(_make_ohlcv_df(30, 50000), pd.DataFrame())
        assert result.empty


# ===========================================================================
# LiquidationProxy Tests
# ===========================================================================


class TestLiquidationProxy:
    """Tests for liquidation proxy features."""

    def test_4_features_produced(self):
        """Verify all 4 liquidation proxy features."""
        df = _make_ohlcv_df(60)
        result = LiquidationProxy().compute(df)

        expected_cols = [
            "liq_extreme_move_z", "liq_volume_spike",
            "liq_cascade_ratio", "liq_bounce_strength",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_cascade_ratio_bounded(self):
        """Cascade ratio should be in [0, 1]."""
        df = _make_ohlcv_df(60)
        result = LiquidationProxy().compute(df)
        cascade = result["liq_cascade_ratio"].dropna()
        assert (cascade >= -0.001).all() and (cascade <= 1.001).all()

    def test_bounce_strength_bounded(self):
        """Bounce strength should be in [0, 1]."""
        df = _make_ohlcv_df(60)
        result = LiquidationProxy().compute(df)
        bounce = result["liq_bounce_strength"].dropna()
        assert (bounce >= -0.001).all() and (bounce <= 1.001).all()

    def test_volume_spike_positive(self):
        """Volume spike ratio should be positive."""
        df = _make_ohlcv_df(60)
        result = LiquidationProxy().compute(df)
        spike = result["liq_volume_spike"].dropna()
        assert (spike > 0).all()

    def test_extreme_move_z_warmup(self):
        """Z-score should be NaN during warmup period."""
        df = _make_ohlcv_df(60)
        result = LiquidationProxy().compute(df)
        assert result["liq_extreme_move_z"].iloc[:20].isna().all()

    def test_empty_input(self):
        result = LiquidationProxy().compute(pd.DataFrame())
        assert result.empty

    def test_missing_columns(self):
        """Missing required columns should return empty."""
        df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2025-01-01", periods=3))
        result = LiquidationProxy().compute(df)
        assert result.empty


# ===========================================================================
# MegaFeatureBuilder Integration Test
# ===========================================================================


class TestMegaIntegration:
    """Verify group 25 is wired into MegaFeatureBuilder."""

    def test_crypto_expanded_in_builders(self):
        """Group 25 crypto_expanded should be in the builder list."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder()
        # The build() method creates a builders list; check that our builder method exists
        assert hasattr(builder, "_build_crypto_expanded_features")

    def test_builder_method_signature(self):
        """Builder method should accept (start_date, end_date)."""
        from quantlaxmi.features.mega import MegaFeatureBuilder
        import inspect

        builder = MegaFeatureBuilder()
        sig = inspect.signature(builder._build_crypto_expanded_features)
        params = list(sig.parameters.keys())
        assert "start_date" in params
        assert "end_date" in params

    def test_builder_returns_empty_gracefully(self):
        """With no Binance data dir, should return empty DataFrame."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder(binance_dir="/nonexistent/path")
        result = builder._build_crypto_expanded_features("2025-01-01", "2025-12-31")
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# No Look-Ahead Tests
# ===========================================================================


class TestNoLookAhead:
    """Verify all features are causal -- only use past data."""

    def test_funding_rate_causal(self):
        """Changing future funding should not affect past features."""
        df_full = _make_funding_df(30)
        df_truncated = df_full.iloc[:60]  # First 20 days (3 per day)

        result_full = FundingRateRegime().compute(df_full)
        result_trunc = FundingRateRegime().compute(df_truncated)

        # Features for the first 20 days should be identical
        common_dates = result_trunc.index
        for col in result_trunc.columns:
            full_vals = result_full.loc[common_dates, col]
            trunc_vals = result_trunc[col]
            # Replace NaN with sentinel for comparison
            full_filled = full_vals.fillna(-999)
            trunc_filled = trunc_vals.fillna(-999)
            np.testing.assert_array_almost_equal(
                full_filled.values, trunc_filled.values,
                decimal=10,
                err_msg=f"Look-ahead detected in {col}",
            )

    def test_oi_dynamics_causal(self):
        """Changing future OI should not affect past features."""
        df_full = _make_oi_df(60)
        df_truncated = df_full.iloc[:30]

        result_full = OpenInterestDynamics().compute(df_full)
        result_trunc = OpenInterestDynamics().compute(df_truncated)

        common_dates = result_trunc.index
        for col in result_trunc.columns:
            full_vals = result_full.loc[common_dates, col]
            trunc_vals = result_trunc[col]
            full_filled = full_vals.fillna(-999)
            trunc_filled = trunc_vals.fillna(-999)
            np.testing.assert_array_almost_equal(
                full_filled.values, trunc_filled.values,
                decimal=10,
                err_msg=f"Look-ahead detected in {col}",
            )

    def test_altcoin_breadth_causal(self):
        """Altcoin breadth should not use future data."""
        btc = _make_ohlcv_df(60, 50000)
        eth = _make_ohlcv_df(60, 3000)

        btc_trunc = btc.iloc[:30]
        eth_trunc = eth.iloc[:30]

        result_full = AltcoinBreadth().compute(btc, eth)
        result_trunc = AltcoinBreadth().compute(btc_trunc, eth_trunc)

        common_dates = result_trunc.index
        for col in result_trunc.columns:
            full_vals = result_full.loc[common_dates, col]
            trunc_vals = result_trunc[col]
            full_filled = full_vals.fillna(-999)
            trunc_filled = trunc_vals.fillna(-999)
            np.testing.assert_array_almost_equal(
                full_filled.values, trunc_filled.values,
                decimal=10,
                err_msg=f"Look-ahead detected in {col}",
            )

    def test_liquidation_proxy_causal(self):
        """Liquidation proxy should not use future data."""
        df_full = _make_ohlcv_df(60)
        df_trunc = df_full.iloc[:30]

        result_full = LiquidationProxy().compute(df_full)
        result_trunc = LiquidationProxy().compute(df_trunc)

        common_dates = result_trunc.index
        for col in result_trunc.columns:
            full_vals = result_full.loc[common_dates, col]
            trunc_vals = result_trunc[col]
            full_filled = full_vals.fillna(-999)
            trunc_filled = trunc_vals.fillna(-999)
            np.testing.assert_array_almost_equal(
                full_filled.values, trunc_filled.values,
                decimal=10,
                err_msg=f"Look-ahead detected in {col}",
            )


# ===========================================================================
# BinanceConnector New Methods Test
# ===========================================================================


class TestBinanceConnectorMethods:
    """Verify new FAPI methods exist on BinanceConnector."""

    def test_methods_exist(self):
        """All 5 new FAPI methods should exist."""
        from quantlaxmi.data.connectors.binance_connector import BinanceConnector

        methods = [
            "fetch_funding_rate_history",
            "fetch_open_interest",
            "fetch_long_short_ratio",
            "fetch_top_long_short_ratio",
            "fetch_taker_long_short_ratio",
        ]
        for method in methods:
            assert hasattr(BinanceConnector, method), f"Missing method: {method}"

    def test_fapi_base_url(self):
        """FAPI_BASE should point to Binance Futures API."""
        from quantlaxmi.data.connectors.binance_connector import BinanceConnector

        assert BinanceConnector.FAPI_BASE == "https://fapi.binance.com"

    def test_funding_rate_method_signature(self):
        """fetch_funding_rate_history should accept symbol, start_time, end_time, limit."""
        import inspect
        from quantlaxmi.data.connectors.binance_connector import BinanceConnector

        sig = inspect.signature(BinanceConnector.fetch_funding_rate_history)
        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "start_time" in params
        assert "end_time" in params
        assert "limit" in params

    def test_ls_ratio_method_signature(self):
        """fetch_long_short_ratio should accept symbol, period, limit."""
        import inspect
        from quantlaxmi.data.connectors.binance_connector import BinanceConnector

        sig = inspect.signature(BinanceConnector.fetch_long_short_ratio)
        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "period" in params
        assert "limit" in params

    def test_taker_method_signature(self):
        """fetch_taker_long_short_ratio should accept symbol, period, limit."""
        import inspect
        from quantlaxmi.data.connectors.binance_connector import BinanceConnector

        sig = inspect.signature(BinanceConnector.fetch_taker_long_short_ratio)
        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "period" in params
        assert "limit" in params


# ===========================================================================
# build_crypto_expanded_features integration test
# ===========================================================================


class TestBuildCryptoExpanded:
    """Tests for the aggregate builder function."""

    def test_returns_empty_without_data(self):
        """With no binance_dir and no connector, returns empty."""
        result = build_crypto_expanded_features("2025-01-01", "2025-12-31")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_with_nonexistent_dir(self):
        """With nonexistent dir and no connector, returns empty."""
        result = build_crypto_expanded_features(
            "2025-01-01", "2025-12-31",
            binance_dir="/nonexistent/dir/that/does/not/exist",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty
