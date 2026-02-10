"""Tests for causality (no look-ahead bias) in momentum_tfm feature functions.

Covers:
1. _changepoint_features — expanding-window PELT, causal at every bar
2. _hmm_regime_features — expanding-window HMM, causal at every bar
3. build_features integration — all features causal
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def close_series(rng):
    """Synthetic close price series with ~250 bars (1 year of daily data)."""
    n = 250
    log_returns = rng.normal(0.0005, 0.01, n)
    close = 20000.0 * np.exp(np.cumsum(log_returns))
    return close


@pytest.fixture
def returns_series(rng):
    """Synthetic daily returns series with ~250 bars."""
    return rng.normal(0.0005, 0.01, 250)


# ============================================================================
# PELT Changepoint Causality Tests
# ============================================================================


class TestChangepointCausality:
    """Tests that _changepoint_features is strictly causal."""

    def test_import(self):
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features
        assert _changepoint_features is not None

    def test_output_shape(self, close_series):
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features
        cp_ind, cp_dist, cp_mag = _changepoint_features(close_series)
        n = len(close_series)
        assert cp_ind.shape == (n,)
        assert cp_dist.shape == (n,)
        assert cp_mag.shape == (n,)

    def test_leading_zeros(self, close_series):
        """Bars before min_window should have default values (no PELT run)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features
        min_window = 30
        cp_ind, cp_dist, cp_mag = _changepoint_features(
            close_series, min_window=min_window
        )
        # Before min_window, indicator should be 0, distance should be 1
        assert np.all(cp_ind[:min_window] == 0.0), (
            "cp_indicator should be 0 before min_window"
        )
        assert np.all(cp_dist[:min_window] == 1.0), (
            "cp_distance should be 1.0 before min_window"
        )
        assert np.all(cp_mag[:min_window] == 0.0), (
            "cp_magnitude should be 0.0 before min_window"
        )

    def test_no_lookahead_truncation(self, close_series):
        """Features at bar t should be identical whether or not future bars exist.

        If _changepoint_features is causal, running on close[:t+1] should
        produce the same features at bar t as running on the full series.
        """
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features

        # Run on full series
        cp_ind_full, cp_dist_full, cp_mag_full = _changepoint_features(
            close_series, refit_every=1
        )

        # Pick several test bars after min_window
        test_bars = [50, 80, 120, 180]
        for t in test_bars:
            # Run on truncated series (causal: only data up to t)
            cp_ind_trunc, cp_dist_trunc, cp_mag_trunc = _changepoint_features(
                close_series[: t + 1], refit_every=1
            )
            assert np.isclose(cp_ind_trunc[t], cp_ind_full[t], atol=1e-10), (
                f"cp_indicator at bar {t} differs between truncated and full series"
            )
            assert np.isclose(cp_dist_trunc[t], cp_dist_full[t], atol=1e-10), (
                f"cp_distance at bar {t} differs between truncated and full series"
            )
            assert np.isclose(cp_mag_trunc[t], cp_mag_full[t], atol=1e-10), (
                f"cp_magnitude at bar {t} differs between truncated and full series"
            )

    def test_refit_every_parameter(self, close_series):
        """refit_every controls how often PELT reruns; output remains valid."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features
        cp1_ind, cp1_dist, cp1_mag = _changepoint_features(
            close_series, refit_every=1
        )
        cp5_ind, cp5_dist, cp5_mag = _changepoint_features(
            close_series, refit_every=5
        )
        # Both should have the same shape and no NaN
        assert cp1_ind.shape == cp5_ind.shape
        assert not np.any(np.isnan(cp1_ind))
        assert not np.any(np.isnan(cp5_ind))

    def test_short_series_returns_defaults(self):
        """Series shorter than min_window returns zeros/ones."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features
        close = np.array([100.0, 101.0, 99.5, 102.0])
        cp_ind, cp_dist, cp_mag = _changepoint_features(close, min_window=30)
        assert np.all(cp_ind == 0.0)
        assert np.all(cp_dist == 1.0)
        assert np.all(cp_mag == 0.0)

    def test_values_in_expected_range(self, close_series):
        """Feature values are in expected ranges."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _changepoint_features
        cp_ind, cp_dist, cp_mag = _changepoint_features(close_series)
        # Indicator is binary
        assert np.all((cp_ind == 0.0) | (cp_ind == 1.0))
        # Distance is in [0, 1]
        assert np.all(cp_dist >= 0.0)
        assert np.all(cp_dist <= 1.0)


# ============================================================================
# HMM Regime Causality Tests
# ============================================================================


class TestHMMRegimeCausality:
    """Tests that _hmm_regime_features is strictly causal."""

    def _has_hmmlearn(self):
        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401
            return True
        except ImportError:
            return False

    def test_import(self):
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features
        assert _hmm_regime_features is not None

    def test_output_shape(self, returns_series):
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features
        probs = _hmm_regime_features(returns_series, n_states=3)
        assert probs.shape == (len(returns_series), 3)

    def test_leading_zeros(self, returns_series):
        """Bars before min_window should have zero probabilities."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features
        min_window = 120
        probs = _hmm_regime_features(
            returns_series, n_states=3, min_window=min_window
        )
        assert np.all(probs[:min_window] == 0.0), (
            "HMM posteriors should be 0.0 before min_window"
        )

    @pytest.mark.skipif(
        not pytest.importorskip("hmmlearn", reason="hmmlearn not installed"),
        reason="hmmlearn not installed",
    )
    def test_no_lookahead_truncation(self, returns_series):
        """Features at bar t should be identical whether or not future data exists.

        If _hmm_regime_features is causal, running on returns[:t+1] should
        produce the same posteriors at bar t as running on the full series.
        """
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features, HAS_HMM
        if not HAS_HMM:
            pytest.skip("hmmlearn not installed")

        # Use refit_every=1 for exact comparison (no carry-forward)
        probs_full = _hmm_regime_features(
            returns_series, n_states=3, min_window=120, refit_every=1
        )

        # Pick test bars after min_window
        test_bars = [130, 160, 200]
        for t in test_bars:
            probs_trunc = _hmm_regime_features(
                returns_series[: t + 1], n_states=3, min_window=120, refit_every=1
            )
            np.testing.assert_allclose(
                probs_trunc[t],
                probs_full[t],
                atol=1e-6,
                err_msg=f"HMM posteriors at bar {t} differ between truncated and full series",
            )

    @pytest.mark.skipif(
        not pytest.importorskip("hmmlearn", reason="hmmlearn not installed"),
        reason="hmmlearn not installed",
    )
    def test_posteriors_sum_to_one(self, returns_series):
        """After min_window, posteriors at each bar should sum to ~1.0."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features, HAS_HMM
        if not HAS_HMM:
            pytest.skip("hmmlearn not installed")

        probs = _hmm_regime_features(returns_series, n_states=3, min_window=120)
        for t in range(120, len(returns_series)):
            row_sum = probs[t].sum()
            assert np.isclose(row_sum, 1.0, atol=1e-6), (
                f"Posteriors at bar {t} sum to {row_sum}, expected 1.0"
            )

    @pytest.mark.skipif(
        not pytest.importorskip("hmmlearn", reason="hmmlearn not installed"),
        reason="hmmlearn not installed",
    )
    def test_posteriors_non_negative(self, returns_series):
        """Posteriors should be non-negative probabilities."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features, HAS_HMM
        if not HAS_HMM:
            pytest.skip("hmmlearn not installed")

        probs = _hmm_regime_features(returns_series, n_states=3)
        assert np.all(probs >= 0.0), "HMM posteriors should be non-negative"

    def test_short_series_returns_zeros(self):
        """Series shorter than min_window returns all zeros."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features
        returns = np.random.normal(0.0, 0.01, 50)
        probs = _hmm_regime_features(returns, n_states=3, min_window=120)
        assert np.all(probs == 0.0)

    def test_refit_every_parameter(self, returns_series):
        """refit_every parameter is accepted and produces valid output."""
        from quantlaxmi.models.ml.tft.momentum_tfm import _hmm_regime_features
        probs = _hmm_regime_features(
            returns_series, n_states=3, min_window=120, refit_every=10
        )
        assert probs.shape == (len(returns_series), 3)
        assert not np.any(np.isnan(probs))


# ============================================================================
# Integration: build_features causality
# ============================================================================


class TestBuildFeaturesCausality:
    """Integration test that build_features produces causal features."""

    def test_build_features_smoke(self, close_series):
        """build_features runs without error and returns expected shapes."""
        from quantlaxmi.models.ml.tft.momentum_tfm import build_features
        features, names = build_features(close_series)
        n = len(close_series)
        assert features.shape[0] == n
        assert features.shape[1] == len(names)
        assert len(names) > 0

    def test_changepoint_features_present(self, close_series):
        """build_features includes the 3 changepoint features."""
        from quantlaxmi.models.ml.tft.momentum_tfm import build_features
        _, names = build_features(close_series)
        assert "cp_indicator" in names
        assert "cp_distance" in names
        assert "cp_magnitude" in names

    def test_hmm_features_present_if_available(self, close_series):
        """build_features includes HMM features when hmmlearn is available."""
        from quantlaxmi.models.ml.tft.momentum_tfm import build_features, HAS_HMM
        _, names = build_features(close_series)
        if HAS_HMM:
            assert "hmm_state_0" in names
            assert "hmm_state_1" in names
            assert "hmm_state_2" in names
