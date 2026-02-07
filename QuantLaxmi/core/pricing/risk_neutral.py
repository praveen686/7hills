"""Risk-neutral density analysis via Breeden-Litzenberger (1978).

Extracts the risk-neutral probability distribution from SANOS-calibrated
volatility surfaces and computes distributional statistics for trading
signal generation.

Mathematical foundation:
    q(K) = e^{rT} × ∂²C/∂K²

    The second derivative of call prices w.r.t. strike yields the
    risk-neutral density.  SANOS computes this analytically via its
    mixture-of-normals representation.

From the density we extract:
    - Standardised moments (skewness, kurtosis)
    - Shannon entropy (market uncertainty)
    - KL divergence between consecutive densities (information flow)
    - Tail-weight decomposition (left/right beyond ±1σ)
    - Physical (realised) skewness for comparison
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.pricing.sanos import SANOSResult


# ---------------------------------------------------------------------------
# Density extraction
# ---------------------------------------------------------------------------

def extract_density(
    result: SANOSResult,
    expiry_idx: int = 0,
    n_points: int = 500,
    K_lo: float = 0.80,
    K_hi: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract risk-neutral density from a calibrated SANOS surface.

    Strikes are in normalised coordinates (K/F), so 1.0 = ATM.

    Returns
    -------
    K : ndarray, shape (n_points,)
        Normalised strike grid.
    q : ndarray, shape (n_points,)
        Probability density values (non-negative, integrates to ~1).
    """
    K = np.linspace(K_lo, K_hi, n_points)
    q = result.density(expiry_idx, K)

    # Clamp to non-negative (numerical noise)
    q = np.maximum(q, 0.0)

    # Normalise so ∫q dK = 1
    dK = K[1] - K[0]
    total = np.sum(q) * dK
    if total > 1e-12:
        q = q / total

    return K, q


# ---------------------------------------------------------------------------
# Moment computation
# ---------------------------------------------------------------------------

def compute_moments(
    K: np.ndarray,
    q: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute mean, variance, skewness, excess-kurtosis of a density.

    Parameters
    ----------
    K : strike grid
    q : density values (must be non-negative and integrate to ~1)

    Returns
    -------
    mean, variance, skewness, excess_kurtosis
    """
    dK = K[1] - K[0]

    mu = float(np.sum(K * q) * dK)
    var = float(np.sum((K - mu) ** 2 * q) * dK)

    if var < 1e-14:
        return mu, var, 0.0, 0.0

    std = math.sqrt(var)
    z = (K - mu) / std

    skew = float(np.sum(z ** 3 * q) * dK)
    kurt = float(np.sum(z ** 4 * q) * dK) - 3.0  # excess

    return mu, var, skew, kurt


# ---------------------------------------------------------------------------
# Information-theoretic measures
# ---------------------------------------------------------------------------

def shannon_entropy(q: np.ndarray, dK: float) -> float:
    """Shannon entropy  H = -∫ q(K) ln q(K) dK.

    Higher entropy = market is more uncertain about the outcome.
    """
    mask = q > 1e-15
    if not np.any(mask):
        return 0.0
    return float(-np.sum(q[mask] * np.log(q[mask])) * dK)


def kl_divergence(q_new: np.ndarray, q_old: np.ndarray, dK: float) -> float:
    """KL divergence  D_KL(q_new ‖ q_old) = ∫ q_new ln(q_new / q_old) dK.

    Measures how much the density changed overnight.
    Symmetric variant (Jensen–Shannon) is also possible but KL is simpler
    and always ≥ 0.
    """
    eps = 1e-15
    qn = np.maximum(q_new, eps)
    qo = np.maximum(q_old, eps)
    return float(np.sum(qn * np.log(qn / qo)) * dK)


# ---------------------------------------------------------------------------
# Tail-weight decomposition
# ---------------------------------------------------------------------------

def tail_weights(
    K: np.ndarray,
    q: np.ndarray,
    mu: float,
    std: float,
) -> tuple[float, float]:
    """Probability mass in the left and right tails (beyond ±1σ).

    Left tail  = P(K < μ - σ)   → crash-risk pricing
    Right tail = P(K > μ + σ)   → rally anticipation
    """
    dK = K[1] - K[0]
    left = float(np.sum(q[K < mu - std]) * dK)
    right = float(np.sum(q[K > mu + std]) * dK)
    return left, right


# ---------------------------------------------------------------------------
# Physical (realised) moments
# ---------------------------------------------------------------------------

def physical_skewness(log_returns: np.ndarray) -> float:
    """Bias-corrected Fisher skewness of log returns."""
    n = len(log_returns)
    if n < 4:
        return 0.0
    mu = np.mean(log_returns)
    std = np.std(log_returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(
        (n / ((n - 1) * (n - 2))) * np.sum(((log_returns - mu) / std) ** 3)
    )


def physical_kurtosis(log_returns: np.ndarray) -> float:
    """Excess kurtosis of log returns (bias-corrected)."""
    n = len(log_returns)
    if n < 5:
        return 0.0
    mu = np.mean(log_returns)
    std = np.std(log_returns, ddof=1)
    if std < 1e-12:
        return 0.0
    m4 = float(np.mean(((log_returns - mu) / std) ** 4))
    # Bias-corrected excess kurtosis
    adj = (n - 1) / ((n - 2) * (n - 3))
    return adj * ((n + 1) * m4 - 3 * (n - 1)) + 3.0 - 3.0  # excess


# ---------------------------------------------------------------------------
# Combined snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensitySnapshot:
    """All density-derived features for one day / symbol / expiry."""

    date: str
    symbol: str

    # Risk-neutral moments (from SANOS density, normalised coords)
    rn_mean: float
    rn_variance: float
    rn_skewness: float
    rn_kurtosis: float

    # Information-theoretic
    entropy: float
    left_tail: float
    right_tail: float

    # Quality
    density_ok: bool


def compute_snapshot(
    result: SANOSResult,
    date_str: str,
    symbol: str,
    expiry_idx: int = 0,
) -> DensitySnapshot:
    """Full density analysis for a single day/symbol."""
    K, q = extract_density(result, expiry_idx)
    dK = K[1] - K[0]

    mu, var, skew, kurt = compute_moments(K, q)
    H = shannon_entropy(q, dK)
    std = math.sqrt(max(var, 1e-14))
    lt, rt = tail_weights(K, q, mu, std)

    # Quality: density should integrate close to 1 and have positive variance
    ok = var > 1e-8 and (np.sum(q) * dK) > 0.90

    return DensitySnapshot(
        date=date_str,
        symbol=symbol,
        rn_mean=mu,
        rn_variance=var,
        rn_skewness=skew,
        rn_kurtosis=kurt,
        entropy=H,
        left_tail=lt,
        right_tail=rt,
        density_ok=ok,
    )
