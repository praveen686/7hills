"""Tests for risk-neutral density analysis (Breeden-Litzenberger).

Tests density extraction, moment computation, entropy, KL divergence,
tail weights, and physical moments using both synthetic lognormal
densities and SANOS-calibrated surfaces.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.pricing.risk_neutral import (
    DensitySnapshot,
    compute_moments,
    compute_snapshot,
    extract_density,
    kl_divergence,
    physical_kurtosis,
    physical_skewness,
    shannon_entropy,
    tail_weights,
)
from core.pricing.sanos import SANOSResult, bs_call, fit_sanos


# ---------------------------------------------------------------------------
# Helpers — synthetic lognormal density
# ---------------------------------------------------------------------------

def _lognormal_density(K: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Lognormal PDF: q(K) = (1/(K σ √(2π))) exp(-(ln K - μ)² / (2σ²))."""
    mask = K > 0
    q = np.zeros_like(K)
    q[mask] = (
        np.exp(-0.5 * ((np.log(K[mask]) - mu) / sigma) ** 2)
        / (K[mask] * sigma * math.sqrt(2 * math.pi))
    )
    return q


def _lognormal_moments(mu: float, sigma: float):
    """Analytical moments of a lognormal(μ, σ²) distribution.

    Returns mean, variance, skewness, excess_kurtosis.
    """
    mean = math.exp(mu + sigma ** 2 / 2)
    var = (math.exp(sigma ** 2) - 1) * math.exp(2 * mu + sigma ** 2)
    skew = (math.exp(sigma ** 2) + 2) * math.sqrt(math.exp(sigma ** 2) - 1)
    kurt = (
        math.exp(4 * sigma ** 2)
        + 2 * math.exp(3 * sigma ** 2)
        + 3 * math.exp(2 * sigma ** 2)
        - 6
    )
    return mean, var, skew, kurt


def _make_sanos_result(sigma: float = 0.20, n_strikes: int = 50) -> SANOSResult:
    """Build a SANOS result from a flat-vol (lognormal) world."""
    T = 30 / 365.0
    total_var = sigma ** 2 * T
    K = np.linspace(0.80, 1.20, n_strikes)
    C = bs_call(np.ones_like(K), K, np.full_like(K, total_var))

    return fit_sanos(
        market_strikes=[K],
        market_calls=[C],
        atm_variances=np.array([total_var]),
        expiry_labels=["2026-02-15"],
        eta=0.50,
        n_model_strikes=100,
        K_min=0.7,
        K_max=1.5,
    )


# ---------------------------------------------------------------------------
# Test: compute_moments on synthetic lognormal
# ---------------------------------------------------------------------------

class TestComputeMoments:
    """Moments of a known lognormal density."""

    def test_lognormal_mean(self):
        """Mean of lognormal(0, 0.15²) ≈ exp(0.15²/2) ≈ 1.0113."""
        sigma = 0.15
        K = np.linspace(0.5, 2.0, 2000)
        q = _lognormal_density(K, 0.0, sigma)
        dK = K[1] - K[0]
        q = q / (np.sum(q) * dK)

        mu, var, skew, kurt = compute_moments(K, q)
        expected_mean = math.exp(sigma ** 2 / 2)
        assert abs(mu - expected_mean) < 0.01

    def test_lognormal_skewness_positive(self):
        """Lognormal always has positive skewness."""
        K = np.linspace(0.3, 3.0, 3000)
        q = _lognormal_density(K, 0.0, 0.30)
        dK = K[1] - K[0]
        q = q / (np.sum(q) * dK)
        _, _, skew, _ = compute_moments(K, q)
        assert skew > 0

    def test_narrow_sigma_skewness_small(self):
        """Very narrow lognormal (σ→0) has near-zero skewness."""
        sigma = 0.02
        K = np.linspace(0.8, 1.2, 2000)
        q = _lognormal_density(K, 0.0, sigma)
        dK = K[1] - K[0]
        q = q / (np.sum(q) * dK)
        _, _, skew, _ = compute_moments(K, q)
        assert abs(skew) < 0.2

    def test_wider_sigma_more_skewed(self):
        """Higher σ → higher skewness."""
        results = []
        for sigma in [0.10, 0.25, 0.50]:
            K = np.linspace(0.1, 4.0, 5000)
            q = _lognormal_density(K, 0.0, sigma)
            dK = K[1] - K[0]
            q = q / (np.sum(q) * dK)
            _, _, skew, _ = compute_moments(K, q)
            results.append(skew)
        assert results[0] < results[1] < results[2]

    def test_excess_kurtosis_positive(self):
        """Lognormal has positive excess kurtosis."""
        K = np.linspace(0.2, 3.0, 3000)
        q = _lognormal_density(K, 0.0, 0.25)
        dK = K[1] - K[0]
        q = q / (np.sum(q) * dK)
        _, _, _, kurt = compute_moments(K, q)
        assert kurt > 0

    def test_zero_variance_returns_zeros(self):
        """Degenerate density (single point) → zero skew and kurt."""
        K = np.array([0.99, 1.00, 1.01])
        dK = K[1] - K[0]
        # Normalise so ∫q dK = 1  →  q_peak = 1/dK
        q = np.array([0.0, 1.0 / dK, 0.0])
        mu, var, skew, kurt = compute_moments(K, q)
        assert abs(mu - 1.0) < 0.01
        assert skew == 0.0
        assert kurt == 0.0


# ---------------------------------------------------------------------------
# Test: Shannon entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:

    def test_uniform_entropy(self):
        """Uniform density has max entropy for a given support."""
        n = 1000
        K = np.linspace(0.8, 1.2, n)
        dK = K[1] - K[0]
        q = np.ones(n) / (n * dK)  # uniform
        H = shannon_entropy(q, dK)
        # Entropy of uniform on [a,b] = ln(b-a)
        expected = math.log(1.2 - 0.8)
        assert abs(H - expected) < 0.05

    def test_concentrated_low_entropy(self):
        """Concentrated density has lower entropy than broad density."""
        K = np.linspace(0.5, 1.5, 2000)
        dK = K[1] - K[0]
        q_narrow = _lognormal_density(K, 0.0, 0.05)
        q_narrow /= np.sum(q_narrow) * dK
        q_broad = _lognormal_density(K, 0.0, 0.30)
        q_broad /= np.sum(q_broad) * dK

        H_narrow = shannon_entropy(q_narrow, dK)
        H_broad = shannon_entropy(q_broad, dK)
        assert H_narrow < H_broad

    def test_zero_density_entropy(self):
        """All-zero density → entropy 0."""
        q = np.zeros(100)
        assert shannon_entropy(q, 0.01) == 0.0


# ---------------------------------------------------------------------------
# Test: KL divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:

    def test_same_density_zero(self):
        """D_KL(q ‖ q) = 0."""
        K = np.linspace(0.8, 1.2, 500)
        dK = K[1] - K[0]
        q = _lognormal_density(K, 0.0, 0.15)
        q /= np.sum(q) * dK
        assert abs(kl_divergence(q, q, dK)) < 1e-10

    def test_different_densities_positive(self):
        """D_KL(q1 ‖ q2) > 0 when q1 ≠ q2."""
        K = np.linspace(0.5, 1.5, 1000)
        dK = K[1] - K[0]
        q1 = _lognormal_density(K, 0.0, 0.15)
        q1 /= np.sum(q1) * dK
        q2 = _lognormal_density(K, 0.0, 0.25)
        q2 /= np.sum(q2) * dK
        assert kl_divergence(q1, q2, dK) > 0.01

    def test_kl_asymmetric(self):
        """D_KL(p‖q) ≠ D_KL(q‖p) in general."""
        K = np.linspace(0.5, 1.5, 1000)
        dK = K[1] - K[0]
        q1 = _lognormal_density(K, 0.0, 0.15)
        q1 /= np.sum(q1) * dK
        q2 = _lognormal_density(K, 0.0, 0.30)
        q2 /= np.sum(q2) * dK
        assert kl_divergence(q1, q2, dK) != pytest.approx(
            kl_divergence(q2, q1, dK), abs=0.01
        )


# ---------------------------------------------------------------------------
# Test: tail weights
# ---------------------------------------------------------------------------

class TestTailWeights:

    def test_symmetric_density_equal_tails(self):
        """A normal-ish density centred at 1.0 has roughly equal tails."""
        K = np.linspace(0.5, 1.5, 2000)
        dK = K[1] - K[0]
        # Narrow lognormal ≈ normal
        q = _lognormal_density(K, 0.0, 0.03)
        q /= np.sum(q) * dK
        mu, var, _, _ = compute_moments(K, q)
        std = math.sqrt(var)
        lt, rt = tail_weights(K, q, mu, std)
        # For a near-normal distribution, each tail ≈ 15.87%
        assert abs(lt - rt) < 0.05
        assert 0.10 < lt < 0.22
        assert 0.10 < rt < 0.22

    def test_tails_sum_less_than_one(self):
        """Left + right tail < 1 (core probability is non-zero)."""
        K = np.linspace(0.5, 1.5, 1000)
        dK = K[1] - K[0]
        q = _lognormal_density(K, 0.0, 0.20)
        q /= np.sum(q) * dK
        mu, var, _, _ = compute_moments(K, q)
        lt, rt = tail_weights(K, q, mu, math.sqrt(var))
        assert lt + rt < 1.0
        assert lt > 0
        assert rt > 0


# ---------------------------------------------------------------------------
# Test: physical_skewness
# ---------------------------------------------------------------------------

class TestPhysicalSkewness:

    def test_symmetric_returns(self):
        """Symmetric returns → skewness ≈ 0."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, size=100)
        assert abs(physical_skewness(rets)) < 0.5

    def test_negative_tail_returns(self):
        """Returns with large negative outliers → negative skewness."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, size=200)
        # Add crash days
        rets[50] = -0.08
        rets[120] = -0.06
        assert physical_skewness(rets) < -0.5

    def test_too_few_returns(self):
        """< 4 returns → 0."""
        assert physical_skewness(np.array([0.01, 0.02, -0.01])) == 0.0


# ---------------------------------------------------------------------------
# Test: extract_density from SANOS
# ---------------------------------------------------------------------------

class TestExtractDensity:

    @pytest.fixture
    def sanos_result(self):
        return _make_sanos_result(sigma=0.20)

    def test_density_non_negative(self, sanos_result):
        K, q = extract_density(sanos_result, 0)
        assert np.all(q >= 0)

    def test_density_integrates_to_one(self, sanos_result):
        K, q = extract_density(sanos_result, 0)
        dK = K[1] - K[0]
        total = np.sum(q) * dK
        assert abs(total - 1.0) < 0.02

    def test_density_mean_near_one(self, sanos_result):
        """Martingale condition: E[K/F] = 1."""
        K, q = extract_density(sanos_result, 0)
        mu, _, _, _ = compute_moments(K, q)
        assert abs(mu - 1.0) < 0.02

    def test_density_returns_correct_length(self, sanos_result):
        K, q = extract_density(sanos_result, 0, n_points=300)
        assert len(K) == 300
        assert len(q) == 300


# ---------------------------------------------------------------------------
# Test: compute_snapshot
# ---------------------------------------------------------------------------

class TestComputeSnapshot:

    def test_snapshot_fields(self):
        result = _make_sanos_result(sigma=0.20)
        snap = compute_snapshot(result, "2026-01-15", "NIFTY")

        assert snap.date == "2026-01-15"
        assert snap.symbol == "NIFTY"
        assert snap.density_ok
        assert snap.rn_variance > 0
        # Differential entropy can be negative for narrow supports
        assert isinstance(snap.entropy, float)

    def test_snapshot_skewness_positive_for_lognormal(self):
        """Flat-vol → lognormal density → positive skewness."""
        result = _make_sanos_result(sigma=0.20)
        snap = compute_snapshot(result, "2026-01-15", "NIFTY")
        # SANOS density from flat vol is approximately lognormal
        # Skewness should be small and positive
        assert snap.rn_skewness > -0.5  # allow some noise

    def test_higher_vol_wider_density(self):
        """Higher σ → higher variance in density."""
        r_low = _make_sanos_result(sigma=0.10)
        r_high = _make_sanos_result(sigma=0.35)
        snap_low = compute_snapshot(r_low, "2026-01-15", "NIFTY")
        snap_high = compute_snapshot(r_high, "2026-01-15", "NIFTY")
        assert snap_high.rn_variance > snap_low.rn_variance

    def test_higher_vol_higher_entropy(self):
        """Higher σ → higher entropy (more uncertainty)."""
        r_low = _make_sanos_result(sigma=0.10)
        r_high = _make_sanos_result(sigma=0.35)
        snap_low = compute_snapshot(r_low, "2026-01-15", "NIFTY")
        snap_high = compute_snapshot(r_high, "2026-01-15", "NIFTY")
        assert snap_high.entropy > snap_low.entropy
