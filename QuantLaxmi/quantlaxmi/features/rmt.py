"""Random Matrix Theory features for regime detection.

Compares the eigenvalue spectrum of a rolling correlation matrix of sector
index returns against the Marchenko-Pastur (MP) distribution to separate
genuine market structure from noise.

Features produced:
  - absorption_ratio: fraction of variance explained by top-k eigenvalues
  - mp_excess: ratio of largest eigenvalue to MP upper bound
  - effective_dimension: number of eigenvalues exceeding MP noise bound
  - eigen_entropy: Shannon entropy of the eigenvalue distribution

References:
  - Laloux, Cizeau, Bouchaud, Potters (1999) — RMT for financial correlations
  - Kritzman, Li, Page, Rigobon (2010) — Absorption Ratio
"""

from __future__ import annotations

import numpy as np


def marchenko_pastur_bounds(
    n_obs: int,
    n_assets: int,
    sigma2: float = 1.0,
) -> tuple[float, float]:
    """Marchenko-Pastur upper and lower bounds for eigenvalues.

    For a random matrix of n_obs observations and n_assets variables,
    eigenvalues of the correlation matrix fall within [λ-, λ+]:

        λ± = σ² (1 ± √(n_assets / n_obs))²

    Parameters
    ----------
    n_obs : int
        Number of observations (rows) in the returns matrix.
    n_assets : int
        Number of assets (columns).
    sigma2 : float
        Variance of the underlying distribution (1.0 for correlation matrix).

    Returns
    -------
    (lambda_minus, lambda_plus)
    """
    q = n_assets / n_obs
    sqrt_q = np.sqrt(q)
    lambda_plus = sigma2 * (1.0 + sqrt_q) ** 2
    lambda_minus = sigma2 * max(0.0, (1.0 - sqrt_q) ** 2)
    return float(lambda_minus), float(lambda_plus)


def rmt_features(
    returns_matrix: np.ndarray,
    top_k: int = 5,
) -> dict[str, float]:
    """Compute RMT features from a returns matrix.

    Parameters
    ----------
    returns_matrix : ndarray of shape (n_obs, n_assets)
        Daily log returns for each asset (rows = days, columns = assets).
        Must have n_obs > n_assets for meaningful MP bounds.
    top_k : int
        Number of top eigenvalues for absorption ratio.

    Returns
    -------
    dict with keys:
        absorption_ratio : float in [0, 1]
            Fraction of total variance in top-k eigenvalues.
            High (>0.8) = systemic/herding; Low (<0.5) = dispersed.
        mp_excess : float >= 0
            Largest eigenvalue / MP upper bound.
            >1 means genuine market factor; >>1 means dominant factor.
        effective_dimension : int
            Number of eigenvalues exceeding the MP upper bound.
            Low = concentrated risk; High = many independent factors.
        eigen_entropy : float >= 0
            Shannon entropy of the normalised eigenvalue distribution.
            Low = one factor dominates; High = many independent factors.
        lambda_max : float
            Largest eigenvalue (for diagnostics).
        mp_upper : float
            Marchenko-Pastur upper bound (for diagnostics).
    """
    n_obs, n_assets = returns_matrix.shape

    if n_obs < 3 or n_assets < 2:
        return _nan_result()

    # Correlation matrix (handles constant columns gracefully)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(returns_matrix.T)

    if not np.isfinite(corr).all():
        # Replace NaN correlations (from constant columns) with 0
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

    # Eigendecomposition (symmetric → use eigvalsh for speed)
    eigenvalues = np.linalg.eigvalsh(corr)  # sorted ascending
    eigenvalues = eigenvalues[::-1]  # largest first
    eigenvalues = np.maximum(eigenvalues, 0.0)  # numerical floor

    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        return _nan_result()

    # MP bounds
    _, mp_upper = marchenko_pastur_bounds(n_obs, n_assets)

    # Absorption ratio: Σ(top-k) / Σ(all)
    k = min(top_k, len(eigenvalues))
    absorption_ratio = float(eigenvalues[:k].sum() / total_var)

    # MP excess: λ_max / λ+
    lambda_max = float(eigenvalues[0])
    mp_excess = lambda_max / mp_upper if mp_upper > 0 else 0.0

    # Effective dimension: count eigenvalues > MP upper bound
    effective_dimension = int(np.sum(eigenvalues > mp_upper))

    # Eigen entropy: -Σ p_i log(p_i)
    p = eigenvalues / total_var
    p = p[p > 1e-15]  # avoid log(0)
    eigen_entropy = float(-np.sum(p * np.log(p)))

    return {
        "absorption_ratio": absorption_ratio,
        "mp_excess": mp_excess,
        "effective_dimension": effective_dimension,
        "eigen_entropy": eigen_entropy,
        "lambda_max": lambda_max,
        "mp_upper": mp_upper,
    }


def rolling_rmt_features(
    returns_matrix: np.ndarray,
    window: int = 60,
    top_k: int = 5,
) -> dict[str, np.ndarray]:
    """Compute rolling RMT features over a returns matrix.

    Parameters
    ----------
    returns_matrix : ndarray of shape (n_days, n_assets)
        Full daily log returns for each asset.
    window : int
        Rolling window size (days).
    top_k : int
        Number of top eigenvalues for absorption ratio.

    Returns
    -------
    dict of arrays, each of length n_days, NaN for warmup period.
    """
    n_days, n_assets = returns_matrix.shape

    out = {
        "rmt_absorption": np.full(n_days, np.nan),
        "rmt_mp_excess": np.full(n_days, np.nan),
        "rmt_eff_dim": np.full(n_days, np.nan),
        "rmt_eigen_entropy": np.full(n_days, np.nan),
    }

    for i in range(window, n_days):
        chunk = returns_matrix[i - window: i]

        # Skip if too many NaN columns
        valid_cols = ~np.isnan(chunk).any(axis=0)
        if valid_cols.sum() < 5:
            continue

        chunk_clean = chunk[:, valid_cols]
        feats = rmt_features(chunk_clean, top_k=top_k)

        if not np.isnan(feats["absorption_ratio"]):
            out["rmt_absorption"][i] = feats["absorption_ratio"]
            out["rmt_mp_excess"][i] = feats["mp_excess"]
            out["rmt_eff_dim"][i] = feats["effective_dimension"]
            out["rmt_eigen_entropy"][i] = feats["eigen_entropy"]

    return out


def _nan_result() -> dict[str, float]:
    return {
        "absorption_ratio": np.nan,
        "mp_excess": np.nan,
        "effective_dimension": np.nan,
        "eigen_entropy": np.nan,
        "lambda_max": np.nan,
        "mp_upper": np.nan,
    }
