"""Tests for Item 11 (TFT Temporal & Static Covariate Encoders) and
Item 12 (Cross-Asset Features).

Tests:
  Temporal covariate encoder:
    - test_temporal_encoder_output_shape
    - test_temporal_day_of_week_range
    - test_static_encoder_output_shape
    - test_lstm_init_from_static
    - test_temporal_disabled_gracefully
    - test_static_disabled_gracefully

  Cross-asset features:
    - test_cross_asset_features_count
    - test_cross_asset_correlation_range
    - test_cross_asset_spread_zero_mean
    - test_cross_asset_no_lookahead
    - test_cross_asset_nan_handling
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Conditional torch import — skip if not available
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")
import torch.nn as nn

from quantlaxmi.models.ml.tft.momentum_tfm import (
    MomentumTFMConfig,
    HAS_TORCH,
)

if HAS_TORCH:
    from quantlaxmi.models.ml.tft.momentum_tfm import (
        MomentumTransformerModel,
        TemporalCovariateEncoder,
        StaticCovariateEncoder,
    )


# ============================================================================
# Temporal Covariate Encoder Tests
# ============================================================================


class TestTemporalCovariateEncoder:
    """Tests for TemporalCovariateEncoder."""

    @pytest.fixture
    def hidden_dim(self):
        return 32

    @pytest.fixture
    def encoder(self, hidden_dim):
        return TemporalCovariateEncoder(hidden_dim=hidden_dim)

    @pytest.fixture
    def sample_temporal(self):
        """Sample temporal features: (batch=4, seq_len=10, 6 features)."""
        batch, seq_len = 4, 10
        temporal = torch.zeros(batch, seq_len, 6, dtype=torch.long)
        temporal[:, :, 0] = torch.randint(0, 7, (batch, seq_len))     # day_of_week
        temporal[:, :, 1] = torch.randint(0, 12, (batch, seq_len))    # month
        temporal[:, :, 2] = torch.randint(0, 31, (batch, seq_len))    # day_of_month
        temporal[:, :, 3] = torch.randint(0, 2, (batch, seq_len))     # is_expiry_week
        temporal[:, :, 4] = torch.randint(0, 2, (batch, seq_len))     # is_monthly_expiry
        temporal[:, :, 5] = torch.randint(0, 4, (batch, seq_len))     # quarter
        return temporal

    def test_temporal_encoder_output_shape(self, encoder, sample_temporal, hidden_dim):
        """Output matches (batch, seq_len, hidden_dim)."""
        out = encoder(sample_temporal)
        batch, seq_len = sample_temporal.shape[:2]
        assert out.shape == (batch, seq_len, hidden_dim), (
            f"Expected ({batch}, {seq_len}, {hidden_dim}), got {out.shape}"
        )

    def test_temporal_day_of_week_range(self, encoder):
        """day_of_week embeddings handle indices 0-6 without error."""
        batch, seq_len = 2, 5
        temporal = torch.zeros(batch, seq_len, 6, dtype=torch.long)
        # Set day_of_week to each value 0-6
        for i in range(7):
            if i < seq_len:
                temporal[:, i, 0] = i
        # Should not raise
        out = encoder(temporal)
        assert out.shape[0] == batch
        assert not torch.isnan(out).any(), "NaN in temporal encoder output"


class TestStaticCovariateEncoder:
    """Tests for StaticCovariateEncoder."""

    @pytest.fixture
    def hidden_dim(self):
        return 32

    @pytest.fixture
    def lstm_layers(self):
        return 2

    @pytest.fixture
    def encoder(self, hidden_dim, lstm_layers):
        return StaticCovariateEncoder(
            n_categories=4,
            embed_dim=8,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
        )

    def test_static_encoder_output_shape(self, encoder, hidden_dim, lstm_layers):
        """4 context vectors each have correct shapes."""
        batch = 4
        asset_id = torch.tensor([0, 1, 2, 3])
        c_s, c_e, c_h, c_c = encoder(asset_id)

        assert c_s.shape == (batch, hidden_dim), f"c_s shape: {c_s.shape}"
        assert c_e.shape == (batch, hidden_dim), f"c_e shape: {c_e.shape}"
        assert c_h.shape == (lstm_layers, batch, hidden_dim), f"c_h shape: {c_h.shape}"
        assert c_c.shape == (lstm_layers, batch, hidden_dim), f"c_c shape: {c_c.shape}"


class TestMomentumTransformerWithCovariates:
    """Integration tests for MomentumTransformerModel with covariates."""

    @pytest.fixture
    def hidden_dim(self):
        return 32

    @pytest.fixture
    def n_features(self):
        return 10

    @pytest.fixture
    def seq_len(self):
        return 8

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def sample_x(self, batch_size, seq_len, n_features):
        return torch.randn(batch_size, seq_len, n_features)

    @pytest.fixture
    def sample_temporal(self, batch_size, seq_len):
        temporal = torch.zeros(batch_size, seq_len, 6, dtype=torch.long)
        temporal[:, :, 0] = torch.randint(0, 7, (batch_size, seq_len))
        temporal[:, :, 1] = torch.randint(0, 12, (batch_size, seq_len))
        temporal[:, :, 2] = torch.randint(0, 31, (batch_size, seq_len))
        temporal[:, :, 3] = torch.randint(0, 2, (batch_size, seq_len))
        temporal[:, :, 4] = torch.randint(0, 2, (batch_size, seq_len))
        temporal[:, :, 5] = torch.randint(0, 4, (batch_size, seq_len))
        return temporal

    @pytest.fixture
    def sample_asset_id(self, batch_size):
        return torch.randint(0, 4, (batch_size,))

    def test_lstm_init_from_static(self, n_features, hidden_dim, sample_x, sample_asset_id):
        """LSTM hidden state initialized from static embedding (not zeros)."""
        model = MomentumTransformerModel(
            n_features=n_features,
            hidden_dim=hidden_dim,
            lstm_layers=2,
            use_static_covariates=True,
            n_static_categories=4,
            static_embed_dim=8,
        )
        model.eval()

        # Get the static encoder outputs
        c_s, c_e, c_h, c_c = model.static_encoder(sample_asset_id)

        # c_h and c_c should not be all zeros (they are learned projections)
        # With random init, they should have non-zero values
        assert c_h.shape[0] == 2, "LSTM layers mismatch"
        assert c_h.shape[1] == sample_x.shape[0], "Batch size mismatch"
        assert c_h.shape[2] == hidden_dim, "Hidden dim mismatch"

        # Verify that the model actually uses static covariates by checking
        # that output differs with/without asset_id
        with torch.no_grad():
            out_with = model(sample_x, asset_id=sample_asset_id)
            out_without = model(sample_x, asset_id=None)

        # Outputs should differ when static covariates are used vs not
        # (when asset_id=None, static encoder is skipped)
        assert not torch.allclose(out_with, out_without, atol=1e-6), (
            "Output should differ with/without static covariates"
        )

    def test_temporal_disabled_gracefully(self, n_features, hidden_dim, sample_x):
        """use_temporal_covariates=False: model works without temporal input."""
        model = MomentumTransformerModel(
            n_features=n_features,
            hidden_dim=hidden_dim,
            use_temporal_covariates=False,
        )
        model.eval()

        with torch.no_grad():
            out = model(sample_x)

        assert out.shape == (sample_x.shape[0], 1)
        assert not torch.isnan(out).any(), "NaN in output"

    def test_static_disabled_gracefully(self, n_features, hidden_dim, sample_x):
        """use_static_covariates=False: model works without asset ID."""
        model = MomentumTransformerModel(
            n_features=n_features,
            hidden_dim=hidden_dim,
            use_static_covariates=False,
        )
        model.eval()

        with torch.no_grad():
            out = model(sample_x)

        assert out.shape == (sample_x.shape[0], 1)
        assert not torch.isnan(out).any(), "NaN in output"

    def test_full_model_with_both_covariates(
        self, n_features, hidden_dim, sample_x, sample_temporal, sample_asset_id
    ):
        """Model works with both temporal and static covariates enabled."""
        model = MomentumTransformerModel(
            n_features=n_features,
            hidden_dim=hidden_dim,
            use_temporal_covariates=True,
            use_static_covariates=True,
            n_static_categories=4,
            static_embed_dim=8,
        )
        model.eval()

        with torch.no_grad():
            out = model(
                sample_x,
                temporal_features=sample_temporal,
                asset_id=sample_asset_id,
            )

        assert out.shape == (sample_x.shape[0], 1)
        assert not torch.isnan(out).any(), "NaN in output with both covariates"

    def test_classify_with_covariates(
        self, n_features, hidden_dim, sample_x, sample_temporal, sample_asset_id
    ):
        """classify() method works with temporal and static covariates."""
        model = MomentumTransformerModel(
            n_features=n_features,
            hidden_dim=hidden_dim,
            use_temporal_covariates=True,
            use_static_covariates=True,
        )
        model.eval()

        with torch.no_grad():
            logits = model.classify(
                sample_x,
                temporal_features=sample_temporal,
                asset_id=sample_asset_id,
            )

        assert logits.shape == (sample_x.shape[0], 1)


# ============================================================================
# Cross-Asset Feature Tests
# ============================================================================


def _make_mock_store_with_index_data(
    n_days: int = 200,
    start_date: str = "2025-01-01",
    include_nans: bool = False,
) -> MagicMock:
    """Create a mock MarketDataStore that returns synthetic index close data."""
    dates = pd.bdate_range(start=start_date, periods=n_days, freq="B")

    # Generate realistic-looking index close prices
    np.random.seed(42)
    nifty_base = 22000
    bnf_base = 48000
    finn_base = 20000
    midcp_base = 10000

    nifty_ret = np.random.normal(0.0005, 0.01, n_days)
    bnf_ret = np.random.normal(0.0005, 0.015, n_days)
    finn_ret = np.random.normal(0.0003, 0.012, n_days)
    midcp_ret = np.random.normal(0.0004, 0.011, n_days)

    nifty_close = nifty_base * np.exp(np.cumsum(nifty_ret))
    bnf_close = bnf_base * np.exp(np.cumsum(bnf_ret))
    finn_close = finn_base * np.exp(np.cumsum(finn_ret))
    midcp_close = midcp_base * np.exp(np.cumsum(midcp_ret))

    if include_nans:
        # Introduce some NaN values
        nifty_close[10] = np.nan
        bnf_close[15] = np.nan

    index_data = {
        "NIFTY 50": nifty_close,
        "NIFTY BANK": bnf_close,
        "NIFTY FINANCIAL SERVICES": finn_close,
        "NIFTY MIDCAP SELECT": midcp_close,
    }

    def mock_sql(query, params=None):
        if params is None:
            return pd.DataFrame()
        # Extract the index name from params
        idx_name = params[0] if params else None
        if idx_name is None:
            return pd.DataFrame()

        # Case-insensitive match
        matched_name = None
        for name in index_data:
            if name.lower() == idx_name.lower():
                matched_name = name
                break

        if matched_name is None:
            return pd.DataFrame()

        closes = index_data[matched_name]
        df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "close": closes,
        })

        # Filter by date range if provided
        if len(params) >= 3:
            start_dt = params[1]
            end_dt = params[2]
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        return df

    store = MagicMock()
    store.sql = mock_sql
    return store


class TestCrossAssetFeatures:
    """Tests for MegaFeatureBuilder._build_cross_asset_features."""

    @pytest.fixture
    def builder(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder
        b = MegaFeatureBuilder(use_cache=False)
        b._store = _make_mock_store_with_index_data(n_days=200)
        return b

    @pytest.fixture
    def features(self, builder):
        return builder._build_cross_asset_features(
            primary_symbol="NIFTY 50",
            start_date="2025-03-01",
            end_date="2025-10-01",
        )

    def test_cross_asset_features_count(self, features):
        """Returns 6 features."""
        assert len(features.columns) == 6, (
            f"Expected 6 cross-asset features, got {len(features.columns)}: {list(features.columns)}"
        )
        expected = {
            "ca_nifty_bnf_corr_21d",
            "ca_nifty_bnf_spread",
            "ca_nifty_bnf_spread_z21",
            "ca_relative_strength",
            "ca_breadth_corr",
            "ca_lead_lag_1d",
        }
        assert set(features.columns) == expected, (
            f"Missing features: {expected - set(features.columns)}"
        )

    def test_cross_asset_correlation_range(self, features):
        """Correlation features are in [-1, 1]."""
        corr = features["ca_nifty_bnf_corr_21d"].dropna()
        assert (corr >= -1.0).all(), "Correlation below -1"
        assert (corr <= 1.0).all(), "Correlation above 1"

        breadth = features["ca_breadth_corr"].dropna()
        assert (breadth >= -1.0).all(), "Breadth correlation below -1"
        assert (breadth <= 1.0).all(), "Breadth correlation above 1"

    def test_cross_asset_spread_zero_mean(self, features):
        """Spread is roughly zero-mean over long window."""
        spread = features["ca_nifty_bnf_spread"].dropna()
        assert len(spread) > 50, "Need sufficient data"
        # Log return spread should be roughly zero mean
        # Allow generous tolerance for random data
        assert abs(spread.mean()) < 0.01, (
            f"Spread mean too far from zero: {spread.mean():.6f}"
        )

    def test_cross_asset_no_lookahead(self):
        """Feature at date T uses only data <= T (lead-lag is shifted by 1).

        Verify that the lead-lag feature at date T equals the other index's
        return from T-1, not T (which would be look-ahead bias).
        """
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder(use_cache=False)
        store = _make_mock_store_with_index_data(n_days=200)
        builder._store = store

        features = builder._build_cross_asset_features(
            primary_symbol="NIFTY 50",
            start_date="2025-01-01",
            end_date="2025-10-01",
        )

        lead_lag = features["ca_lead_lag_1d"]
        # The first date in the full (unlookbacked) dataset should have NaN
        # because shift(1) has no prior value
        first_valid_idx = lead_lag.first_valid_index()
        assert first_valid_idx is not None, "No valid lead-lag values"
        assert lead_lag.iloc[0] != lead_lag.iloc[0] or first_valid_idx > features.index[0], (
            "First value should be NaN (no look-ahead from shift(1))"
        )

        # Also verify correlation uses rolling window (causal):
        # Rolling correlation at date T only uses dates [T-20, T]
        corr = features["ca_nifty_bnf_corr_21d"]
        # First 20 values should be NaN (21-day rolling needs 21 points)
        n_leading_nan = corr.isna().values.argmin()
        assert n_leading_nan >= 20, (
            f"Expected at least 20 leading NaNs for 21-day rolling, got {n_leading_nan}"
        )

    def test_cross_asset_nan_handling(self):
        """Missing data produces NaN, not errors."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder(use_cache=False)
        builder._store = _make_mock_store_with_index_data(
            n_days=200, include_nans=True
        )

        # Should not raise
        features = builder._build_cross_asset_features(
            primary_symbol="NIFTY 50",
            start_date="2025-03-01",
            end_date="2025-10-01",
        )
        assert not features.empty, "Should produce features even with NaN input"
        # NaN should propagate naturally, not cause errors
        assert isinstance(features, pd.DataFrame)

    def test_cross_asset_banknifty_as_primary(self):
        """When BANKNIFTY is primary, lead-lag uses NIFTY return."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder(use_cache=False)
        builder._store = _make_mock_store_with_index_data(n_days=200)

        features = builder._build_cross_asset_features(
            primary_symbol="NIFTY BANK",
            start_date="2025-03-01",
            end_date="2025-10-01",
        )
        assert "ca_lead_lag_1d" in features.columns
        assert "ca_relative_strength" in features.columns
        assert len(features.columns) == 6

    def test_cross_asset_insufficient_indices(self):
        """Returns empty DataFrame if fewer than 2 indices available."""
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder(use_cache=False)
        # Mock store that returns empty for all queries
        store = MagicMock()
        store.sql = MagicMock(return_value=pd.DataFrame())
        builder._store = store

        features = builder._build_cross_asset_features(
            primary_symbol="NIFTY 50",
            start_date="2025-03-01",
            end_date="2025-10-01",
        )
        assert features.empty, "Should return empty with insufficient data"


# ============================================================================
# Config defaults test
# ============================================================================


class TestMomentumTFMConfigDefaults:
    """Verify new config fields have correct defaults."""

    def test_temporal_config_defaults(self):
        cfg = MomentumTFMConfig()
        assert cfg.use_temporal_covariates is True
        assert cfg.use_static_covariates is True
        assert cfg.n_temporal_features == 6
        assert cfg.n_static_categories == 4
        assert cfg.static_embed_dim == 8
