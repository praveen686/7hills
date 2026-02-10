"""Unit tests for S12 Vedic Fractional Alpha features.

Tests mathematical correctness of:
  - Mittag-Leffler function identities
  - Mock theta convergence
  - Aryabhata phase
  - Madhava kernel properties
  - Fractional differencing
"""

import math

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mittag-Leffler
# ---------------------------------------------------------------------------

from quantlaxmi.features.fractional import (
    mittag_leffler,
    estimate_alpha_msd,
    estimate_alpha_waiting,
    estimate_alpha,
    fractional_differentiation,
    solve_ffpe_1d,
    FractionalFeatures,
)


class TestMittagLeffler:
    """Mittag-Leffler special-case identities."""

    def test_exp_identity(self):
        """E_{1,1}(z) = exp(z) for α=1, β=1."""
        for z in [0.0, 0.5, 1.0, -0.5, 2.0, -2.0]:
            ml = mittag_leffler(z, alpha=1.0, beta=1.0)
            assert abs(ml - math.exp(z)) < 1e-8, \
                f"E_{{1,1}}({z}) = {ml}, expected {math.exp(z)}"

    def test_cos_identity(self):
        """E_{2,1}(-z²) = cos(z) for α=2, β=1."""
        for z in [0.0, 0.3, 0.5, 1.0, math.pi / 4]:
            ml = mittag_leffler(-z**2, alpha=2.0, beta=1.0)
            assert abs(ml - math.cos(z)) < 1e-6, \
                f"E_{{2,1}}(-{z}²) = {ml}, expected {math.cos(z)}"

    def test_zero(self):
        """E_{α,β}(0) = 1/Γ(β)."""
        for alpha in [0.5, 1.0, 1.5]:
            for beta in [1.0, 2.0]:
                ml = mittag_leffler(0.0, alpha=alpha, beta=beta)
                expected = 1.0 / math.gamma(beta)
                assert abs(ml - expected) < 1e-10

    def test_vectorized(self):
        """Mittag-Leffler works on arrays."""
        z = np.array([0.0, 0.5, 1.0])
        result = mittag_leffler(z, alpha=1.0, beta=1.0)
        expected = np.exp(z)
        np.testing.assert_allclose(result, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# Alpha estimators
# ---------------------------------------------------------------------------

class TestAlphaEstimators:
    """Fractional α estimation."""

    def test_brownian_alpha_near_one(self):
        """Pure Brownian motion should give α ≈ 1."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        alpha = estimate_alpha_msd(returns, max_lag=50)
        # Brownian: H ≈ 0.5, α = 2H ≈ 1.0
        assert 0.5 < alpha < 1.8, f"Brownian α={alpha}, expected ≈1.0"

    def test_insufficient_data_returns_one(self):
        """With very few data points, should return 1.0."""
        returns = np.array([0.01, -0.01, 0.005])
        alpha = estimate_alpha_msd(returns, max_lag=50)
        assert alpha == 1.0

    def test_consensus_combines(self):
        """estimate_alpha should be a weighted average of MSD and waiting."""
        rng = np.random.default_rng(123)
        returns = rng.normal(0, 0.01, 200)
        a = estimate_alpha(returns)
        a_msd = estimate_alpha_msd(returns)
        a_wt = estimate_alpha_waiting(returns)
        expected = 0.6 * a_msd + 0.4 * a_wt
        assert abs(a - expected) < 1e-10


# ---------------------------------------------------------------------------
# Fractional differencing
# ---------------------------------------------------------------------------

class TestFractionalDifferencing:
    """Fractional differencing (Hosking 1981)."""

    def test_d_zero_identity(self):
        """d=0 should return the original series (w_0 = 1, all others ~0)."""
        series = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = fractional_differentiation(series, d=0.0, threshold=1e-10)
        # d=0: only w_0=1 survives, so result = series
        np.testing.assert_allclose(result, series, rtol=1e-10)

    def test_d_one_first_diff(self):
        """d=1 should approximate first differencing."""
        series = np.array([100.0, 101.0, 103.0, 106.0, 110.0, 115.0,
                           121.0, 128.0, 136.0, 145.0])
        result = fractional_differentiation(series, d=1.0, threshold=1e-10)
        first_diff = np.diff(series)
        # After the first element (which needs full weight window)
        # the fractional diff with d=1 should match first diff
        # Check where both are valid
        valid = ~np.isnan(result[1:])
        if valid.any():
            np.testing.assert_allclose(
                result[1:][valid], first_diff[valid], rtol=0.1,
            )

    def test_optimal_d(self):
        """d=0.226 should produce valid (non-NaN) output for long series."""
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        result = fractional_differentiation(prices, d=0.226, max_window=100)
        valid = ~np.isnan(result)
        assert valid.sum() >= 100, f"Only {valid.sum()} valid values"

    def test_output_length(self):
        """Output should be same length as input."""
        series = np.arange(50, dtype=float)
        result = fractional_differentiation(series, d=0.5)
        assert len(result) == len(series)


# ---------------------------------------------------------------------------
# FFPE solver
# ---------------------------------------------------------------------------

class TestFFPESolver:
    """Fractional Fokker-Planck solver."""

    def test_normalised_output(self):
        """FFPE output should be a valid probability density (sums to ~1)."""
        x_grid = np.linspace(-1, 1, 32)
        dx = x_grid[1] - x_grid[0]
        p = solve_ffpe_1d(alpha=1.0, drift=0.0, diffusion=0.01,
                          x_grid=x_grid, dt=0.1, n_steps=3)
        total = np.sum(p) * dx
        assert abs(total - 1.0) < 0.1, f"Density sums to {total}"

    def test_non_negative(self):
        """FFPE output should be non-negative."""
        x_grid = np.linspace(-2, 2, 64)
        p = solve_ffpe_1d(alpha=0.8, drift=0.01, diffusion=0.05,
                          x_grid=x_grid, dt=0.5, n_steps=5)
        assert np.all(p >= 0), "Negative probabilities in FFPE output"


# ---------------------------------------------------------------------------
# Mock Theta
# ---------------------------------------------------------------------------

from quantlaxmi.features.mock_theta import (
    mock_theta_f,
    mock_theta_phi,
    mock_theta_chi,
    return_to_q,
    ramanujan_volatility_distortion,
    MockThetaFeatures,
)


class TestMockTheta:
    """Ramanujan mock theta functions."""

    def test_f_at_zero(self):
        """f(0) = 1 (only n=0 term survives)."""
        assert mock_theta_f(0.0) == 1.0

    def test_phi_at_zero(self):
        """φ(0) = 1."""
        assert mock_theta_phi(0.0) == 1.0

    def test_chi_at_zero(self):
        """χ(0) = 1."""
        assert mock_theta_chi(0.0) == 1.0

    def test_convergence_small_q(self):
        """For small q, all mock thetas should converge to finite values."""
        q = 0.01
        f = mock_theta_f(q)
        phi = mock_theta_phi(q)
        chi = mock_theta_chi(q)
        assert np.isfinite(f) and f > 0
        assert np.isfinite(phi) and phi > 0
        assert np.isfinite(chi) and chi > 0

    def test_unit_circle_guard(self):
        """For |q| >= 1, functions should return 0."""
        assert mock_theta_f(1.0) == 0.0
        assert mock_theta_phi(1.5) == 0.0
        assert mock_theta_chi(-1.0) == 0.0

    def test_phi_geq_f(self):
        """φ(q) >= f(q) for 0 < q < 1 (f has squared denominator products)."""
        q = 0.05
        f = mock_theta_f(q)
        phi = mock_theta_phi(q)
        # φ has (1+q^k), f has (1+q^k)² → larger denominator → f ≤ φ
        assert phi >= f, f"φ({q})={phi} < f({q})={f}"


class TestReturnToQ:
    """Return → q mapping."""

    def test_zero_return(self):
        """q(0) = exp(-π) ≈ 0.0432."""
        q = return_to_q(np.array([0.0]), sigma=0.01)
        expected = math.exp(-math.pi)
        assert abs(q[0] - expected) < 1e-10

    def test_positive_range(self):
        """q should be in (0, 1) for all finite returns."""
        returns = np.array([-0.1, -0.01, 0.0, 0.01, 0.1])
        q = return_to_q(returns, sigma=0.01)
        assert np.all(q > 0)
        assert np.all(q < 1)

    def test_monotonic_in_abs_return(self):
        """q should increase with |return|."""
        returns = np.array([0.0, 0.01, 0.05, 0.10])
        q = return_to_q(returns, sigma=0.01)
        assert np.all(np.diff(q) > 0)


class TestVolatilityDistortion:
    """Ramanujan continued-fraction volatility transform."""

    def test_zero_vol(self):
        """R(0) = exp(0) / 1 = 1."""
        assert abs(ramanujan_volatility_distortion(0.0) - 1.0) < 1e-10

    def test_high_vol_compression(self):
        """High vol should produce small output (compressed)."""
        low = ramanujan_volatility_distortion(0.1)
        high = ramanujan_volatility_distortion(2.0)
        assert high < low, "High vol should compress more"

    def test_positive_output(self):
        """Output should always be positive."""
        for v in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
            assert ramanujan_volatility_distortion(v) > 0


# ---------------------------------------------------------------------------
# Vedic Angular
# ---------------------------------------------------------------------------

from quantlaxmi.features.vedic_angular import (
    madhava_kernel,
    angular_coherence,
    update_regime_centroids,
    aryabhata_phase,
    VedicAngularFeatures,
)


class TestMadhavaKernel:
    """Madhava angular kernel properties."""

    def test_self_kernel_is_one(self):
        """K(x, x) = 1 for any nonzero x."""
        x = np.array([1.0, 2.0, 3.0])
        k = madhava_kernel(x, x, order=4)
        assert abs(k - 1.0) < 1e-8, f"K(x,x) = {k}, expected 1.0"

    def test_symmetry(self):
        """K(x, y) = K(y, x)."""
        x = np.array([1.0, 0.5, 0.3])
        y = np.array([0.2, 0.8, 0.4])
        k_xy = madhava_kernel(x, y, order=4)
        k_yx = madhava_kernel(y, x, order=4)
        assert abs(k_xy - k_yx) < 1e-10

    def test_converges_to_cos(self):
        """High-order Madhava kernel ≈ cos(θ)."""
        x = np.array([1.0, 0.0])
        y = np.array([0.5, 0.866])  # ~60 degrees
        k_high = madhava_kernel(x, y, order=10)
        # cos(θ) where θ = arccos(x̂·ŷ)
        cos_theta = np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))
        assert abs(k_high - cos_theta) < 1e-4

    def test_zero_vector_returns_zero(self):
        """K(0, y) = 0."""
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 2.0])
        assert madhava_kernel(x, y) == 0.0

    def test_order_one_is_cos(self):
        """Order-1 kernel should be cos(θ) (first two terms: 1 - θ²/2)."""
        x = np.array([1.0, 0.0])
        y = np.array([1.0, 1.0])
        k = madhava_kernel(x, y, order=1)
        cos_theta = np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))
        theta = math.acos(np.clip(cos_theta, -1, 1))
        expected = 1.0 - theta**2 / 2.0
        assert abs(k - expected) < 1e-10


class TestAngularCoherence:
    """Angular coherence with regime centroids."""

    def test_empty_centroids(self):
        """With no centroids, returns unknown."""
        feat = np.array([1.0, 2.0, 3.0])
        regime, coh, res = angular_coherence(feat, {})
        assert regime == "unknown"
        assert coh == 0.0

    def test_exact_match(self):
        """Feature exactly at centroid should have coherence ≈ 1."""
        centroid = np.array([0.1, 0.2, 0.3])
        centroids = {"test": centroid.copy()}
        regime, coh, res = angular_coherence(centroid, centroids)
        assert regime == "test"
        assert abs(coh - 1.0) < 1e-6


class TestAryabhataPhase:
    """Aryabhata sine-difference phase estimation."""

    def test_sinusoidal_signal(self):
        """Phase should track a known sinusoidal signal."""
        period = 20
        t = np.arange(100)
        prices = 100 + 5 * np.sin(2 * math.pi * t / period)
        phase, phase_vel = aryabhata_phase(prices, period)

        # Phase should have valid values
        valid = ~np.isnan(phase)
        assert valid.sum() > 50, f"Only {valid.sum()} valid phase values"

        # Phase should be in [0, 1]
        valid_ph = phase[valid]
        assert np.all(valid_ph >= 0) and np.all(valid_ph <= 1)

    def test_constant_price(self):
        """Constant price → NaN phase (zero amplitude)."""
        prices = np.full(50, 100.0)
        phase, _ = aryabhata_phase(prices, 10)
        # With constant price, detrended amp ≈ 0 → should return NaN
        assert np.all(np.isnan(phase))

    def test_short_period_guard(self):
        """Period < 2 should return all NaN."""
        prices = np.arange(50, dtype=float) + 100
        phase, _ = aryabhata_phase(prices, 1)
        assert np.all(np.isnan(phase))


class TestRegimeCentroids:
    """Centroid update mechanics."""

    def test_initial_centroid(self):
        """First update should set centroid = features."""
        centroids: dict = {}
        feat = np.array([1.0, 2.0, 3.0])
        centroids = update_regime_centroids(centroids, feat, "test")
        np.testing.assert_array_equal(centroids["test"], feat)

    def test_decay_update(self):
        """Second update should blend via EMA."""
        centroids = {"test": np.array([1.0, 0.0, 0.0])}
        feat = np.array([0.0, 1.0, 0.0])
        centroids = update_regime_centroids(centroids, feat, "test", decay=0.5)
        # 0.5 * [1,0,0] + 0.5 * [0,1,0] = [0.5, 0.5, 0], normalised
        expected = np.array([0.5, 0.5, 0.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(centroids["test"], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Feature classes (integration)
# ---------------------------------------------------------------------------

class TestFeatureClasses:
    """Integration tests for Feature subclasses."""

    @pytest.fixture
    def ohlcv_df(self):
        """Synthetic OHLCV DataFrame."""
        rng = np.random.default_rng(42)
        n = 200
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        return pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + abs(rng.normal(0, 0.003, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.003, n))),
            "Close": close,
            "Volume": rng.integers(1000, 10000, n),
        })

    def test_fractional_features_shape(self, ohlcv_df):
        """FractionalFeatures should produce correct columns."""
        feat = FractionalFeatures(window=30)
        result = feat._compute(ohlcv_df)
        expected_cols = {
            "frac_alpha", "frac_alpha_msd", "frac_alpha_waiting",
            "frac_hurst", "frac_d_series",
            "ffpe_entropy", "ffpe_skew", "ffpe_tail_ratio",
        }
        assert set(result.columns) == expected_cols
        assert len(result) == len(ohlcv_df)

    def test_mock_theta_features_shape(self, ohlcv_df):
        """MockThetaFeatures should produce correct columns."""
        feat = MockThetaFeatures(window=20)
        result = feat._compute(ohlcv_df)
        expected_cols = {
            "mock_theta_f", "mock_theta_phi", "mock_theta_chi",
            "mock_theta_divergence", "mock_theta_ratio", "vol_distortion",
        }
        assert set(result.columns) == expected_cols

    def test_vedic_angular_features_shape(self, ohlcv_df):
        """VedicAngularFeatures should produce correct columns."""
        feat = VedicAngularFeatures(window=30)
        result = feat._compute(ohlcv_df)
        expected_cols = {
            "angular_coherence", "angular_regime",
            "madhava_higher_order",
            "aryabhata_phase", "aryabhata_phase_velocity",
        }
        assert set(result.columns) == expected_cols

    def test_feature_name_prefix(self, ohlcv_df):
        """Feature.transform() should prefix columns correctly."""
        from quantlaxmi.core.base.types import OHLCV

        # Create OHLCV wrapper (needs DatetimeIndex)
        ohlcv_df_ts = ohlcv_df.copy()
        ohlcv_df_ts.index = pd.date_range("2025-01-01", periods=len(ohlcv_df), freq="min")
        ohlcv = OHLCV(ohlcv_df_ts)
        feat = MockThetaFeatures(window=20)
        result = feat.transform(ohlcv)
        for col in result.columns:
            assert col.startswith("mock_theta__"), f"Column {col} not prefixed"
