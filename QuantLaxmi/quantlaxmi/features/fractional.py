"""Fractional calculus features for anomalous diffusion regime detection.

Implements:
  - Mittag-Leffler function E_{α,β}(z) — the fractional generalisation of exp
  - MSD-based and waiting-time Hurst/α estimators
  - Fractional differencing (Hosking 1981) — preserves long memory while
    achieving stationarity
  - 1-D Fractional Fokker-Planck equation (FFPE) solver via Mittag-Leffler
    matrix exponential for probability density evolution

The fractional order α classifies market microstructure regimes:
  α < 1  →  subdiffusive (trapping, mean-reversion)
  α ≈ 1  →  normal diffusion (Brownian, no edge)
  α > 1  →  superdiffusive (momentum cascades, Lévy flights)

References:
  - Mainardi (2010), "Fractional Calculus and Waves in Linear Viscoelasticity"
  - Hosking (1981), "Fractional differencing"
  - Metzler & Klafter (2000), "The random walk's guide to anomalous diffusion"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn  # type: ignore[import-untyped]

from quantlaxmi.features.base import Feature


# ---------------------------------------------------------------------------
# Mittag-Leffler function
# ---------------------------------------------------------------------------

def mittag_leffler(z: float | np.ndarray, alpha: float,
                   beta: float = 1.0, n_terms: int = 100) -> float | np.ndarray:
    """Generalised Mittag-Leffler function E_{α,β}(z).

    E_{α,β}(z) = Σ_{k=0}^{N} z^k / Γ(α·k + β)

    Special cases:
      E_{1,1}(z) = exp(z)
      E_{2,1}(-z²) = cos(z)
    """
    scalar = np.isscalar(z)
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    result = np.zeros_like(z, dtype=np.float64)

    for k in range(n_terms):
        denom = gamma_fn(alpha * k + beta)
        if denom == 0 or np.isinf(denom):
            break
        term = z**k / denom
        result += term
        # Early termination for converged entries
        if np.all(np.abs(term) < 1e-15):
            break

    return float(result[0]) if scalar else result


# ---------------------------------------------------------------------------
# Alpha estimators
# ---------------------------------------------------------------------------

def estimate_alpha_msd(returns: np.ndarray, max_lag: int = 50) -> float:
    """Estimate anomalous diffusion exponent via MSD scaling.

    MSD(τ) = E[X²(τ)] ~ τ^{2H}  →  H = slope/2  →  α = 2H

    Uses log-log OLS regression of MSD vs lag.
    """
    cumret = np.cumsum(returns)
    n = len(cumret)
    max_lag = min(max_lag, n // 3)
    if max_lag < 4:
        return 1.0  # insufficient data — assume normal

    lags = np.arange(2, max_lag + 1)
    msds = np.empty(len(lags))

    for i, tau in enumerate(lags):
        displacements = cumret[tau:] - cumret[:-tau]
        msds[i] = np.mean(displacements**2)

    # Guard against zero/negative MSD
    valid = msds > 0
    if valid.sum() < 3:
        return 1.0

    log_lags = np.log(lags[valid])
    log_msds = np.log(msds[valid])

    # OLS slope
    slope, _ = np.polyfit(log_lags, log_msds, 1)
    hurst = np.clip(slope / 2.0, 0.01, 1.5)
    alpha = 2.0 * hurst
    return float(alpha)


def estimate_alpha_waiting(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Estimate α from waiting-time distribution of exceedances.

    Inter-event times between |return| > threshold follow a power-law
    P(τ > t) ~ t^{-α} in anomalous diffusion regimes.

    Uses Hill estimator on waiting times.
    """
    if threshold <= 0:
        threshold = np.std(returns, ddof=1) * 0.5 if len(returns) > 2 else 1e-8

    exceedance_idx = np.where(np.abs(returns) > threshold)[0]
    if len(exceedance_idx) < 10:
        return 1.0  # insufficient events

    waiting_times = np.diff(exceedance_idx).astype(float)
    waiting_times = waiting_times[waiting_times > 0]

    if len(waiting_times) < 5:
        return 1.0

    # Hill estimator for tail index
    sorted_wt = np.sort(waiting_times)[::-1]
    k = max(int(len(sorted_wt) * 0.1), 5)  # top 10%
    k = min(k, len(sorted_wt) - 1)

    log_ratios = np.log(sorted_wt[:k]) - np.log(sorted_wt[k])
    if np.sum(log_ratios) == 0:
        return 1.0
    alpha_hill = k / np.sum(log_ratios)
    return float(np.clip(alpha_hill, 0.1, 3.0))


def estimate_alpha(returns: np.ndarray,
                   timestamps: np.ndarray | None = None,
                   max_lag: int = 50) -> float:
    """Consensus α from MSD + waiting-time estimators (weighted average).

    MSD is more robust for trending series; waiting-time captures
    fat-tail structure.  We weight MSD 0.6, waiting-time 0.4.
    """
    alpha_msd = estimate_alpha_msd(returns, max_lag=max_lag)

    alpha_wt = estimate_alpha_waiting(returns)

    return 0.6 * alpha_msd + 0.4 * alpha_wt


# ---------------------------------------------------------------------------
# Fractional differencing (Hosking 1981)
# ---------------------------------------------------------------------------

def fractional_differentiation(series: np.ndarray, d: float,
                               threshold: float = 1e-5,
                               max_window: int | None = None) -> np.ndarray:
    """Fractionally difference a time series of order d.

    The binomial expansion weights are:
      w_0 = 1
      w_k = -w_{k-1} * (d - k + 1) / k

    Weights are truncated when |w_k| < threshold OR window reaches
    max_window.  d=0.226 was optimal in the LCFT-FFP backtest.

    For d < 0.5, weights decay very slowly (power-law), so max_window
    prevents the window from exceeding the series length.  Default
    max_window = len(series).

    Returns array of same length; leading values where the window
    hasn't filled are set to NaN.
    """
    n = len(series)
    if max_window is None:
        max_window = n
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k > max_window:
            break

    weights = np.array(weights[::-1])  # reverse for convolution
    width = len(weights)

    result = np.full(n, np.nan)
    for i in range(width - 1, n):
        result[i] = np.dot(weights, series[i - width + 1: i + 1])

    return result


# ---------------------------------------------------------------------------
# 1-D Fractional Fokker-Planck solver
# ---------------------------------------------------------------------------

def solve_ffpe_1d(alpha: float, drift: float, diffusion: float,
                  x_grid: np.ndarray, dt: float,
                  n_steps: int) -> np.ndarray:
    """Solve the 1-D fractional Fokker-Planck equation.

    ∂^α p / ∂t^α = -drift · ∂p/∂x + diffusion · ∂²p/∂x²

    Uses Mittag-Leffler time evolution:
      p(x, t_{n+1}) = E_α(-A · dt^α) · p(x, t_n)

    where A is the spatial operator discretised on x_grid.

    Returns p(x, t_final) — the probability density at the last step.
    """
    nx = len(x_grid)
    dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0

    # Spatial operator (tridiagonal: -drift·∂/∂x + D·∂²/∂x²)
    A = np.zeros((nx, nx))
    for i in range(1, nx - 1):
        # Central difference for ∂²/∂x²
        A[i, i - 1] += diffusion / dx**2
        A[i, i]     -= 2 * diffusion / dx**2
        A[i, i + 1] += diffusion / dx**2
        # Upwind difference for -drift·∂/∂x
        if drift >= 0:
            A[i, i]     -= drift / dx
            A[i, i - 1] += drift / dx
        else:
            A[i, i]     += drift / dx
            A[i, i + 1] -= drift / dx

    # Initial condition: Gaussian centered at mean of grid
    mean_x = 0.5 * (x_grid[0] + x_grid[-1])
    sigma0 = 0.1 * (x_grid[-1] - x_grid[0])
    p = np.exp(-0.5 * ((x_grid - mean_x) / max(sigma0, 1e-8))**2)
    p /= (np.sum(p) * dx + 1e-30)  # normalise

    # Time-stepping via Mittag-Leffler matrix exponential
    # For each step: p_{n+1} = Σ_k (-A·dt^α)^k / Γ(α·k+1) · p_n
    # Truncate at order 10 for efficiency
    dt_alpha = dt**alpha
    for _ in range(n_steps):
        p_new = p.copy()
        Ak_p = p.copy()  # A^0 · p
        for k in range(1, 11):
            Ak_p = A @ Ak_p * (-dt_alpha)
            denom = gamma_fn(alpha * k + 1)
            if denom == 0 or np.isinf(denom):
                break
            p_new += Ak_p / denom
        # Enforce non-negativity and re-normalise
        p_new = np.maximum(p_new, 0)
        total = np.sum(p_new) * dx
        if total > 1e-30:
            p_new /= total
        p = p_new

    return p


# ---------------------------------------------------------------------------
# Feature class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FractionalFeatures(Feature):
    """Fractional calculus features for anomalous diffusion detection.

    Produces per bar:
      - frac_alpha       : consensus fractional order α
      - frac_alpha_msd   : MSD-based α estimate
      - frac_alpha_waiting : waiting-time α estimate
      - frac_hurst       : Hurst exponent H = α/2
      - frac_d_series    : fractionally differenced price (d=0.226)
      - ffpe_entropy     : entropy of FFPE probability density
      - ffpe_skew        : skewness of FFPE density
      - ffpe_tail_ratio  : ratio of tail mass to center mass
    """

    window: int = 60
    frac_d: float = 0.226
    ffpe_grid_size: int = 32
    ffpe_steps: int = 5

    @property
    def name(self) -> str:
        return "fractional"

    @property
    def lookback(self) -> int:
        return self.window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].values.astype(np.float64)
        n = len(df)
        out = pd.DataFrame(index=df.index)

        # Pre-compute log returns
        log_ret = np.diff(np.log(np.maximum(close, 1e-8)))
        log_ret = np.concatenate([[0.0], log_ret])

        # Fractional differencing of full series (once)
        d_series = fractional_differentiation(close, self.frac_d)

        # Output arrays
        alpha_arr = np.full(n, np.nan)
        alpha_msd_arr = np.full(n, np.nan)
        alpha_wt_arr = np.full(n, np.nan)
        hurst_arr = np.full(n, np.nan)
        entropy_arr = np.full(n, np.nan)
        skew_arr = np.full(n, np.nan)
        tail_arr = np.full(n, np.nan)

        for i in range(self.window, n):
            rets = log_ret[i - self.window + 1: i + 1]

            a_msd = estimate_alpha_msd(rets, max_lag=min(30, self.window // 2))
            a_wt = estimate_alpha_waiting(rets)
            a = 0.6 * a_msd + 0.4 * a_wt

            alpha_arr[i] = a
            alpha_msd_arr[i] = a_msd
            alpha_wt_arr[i] = a_wt
            hurst_arr[i] = a / 2.0

            # FFPE density stats (lightweight: small grid, few steps)
            vol = np.std(rets, ddof=1) if len(rets) > 1 else 0.01
            drift_est = np.mean(rets)
            x_grid = np.linspace(-4 * vol, 4 * vol, self.ffpe_grid_size)
            dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0

            p = solve_ffpe_1d(
                alpha=np.clip(a, 0.1, 2.0),
                drift=drift_est,
                diffusion=vol**2,
                x_grid=x_grid,
                dt=1.0,
                n_steps=self.ffpe_steps,
            )

            # Shannon entropy
            p_safe = p[p > 1e-30]
            entropy_arr[i] = -np.sum(p_safe * np.log(p_safe) * dx)

            # Skewness of density
            mean_p = np.sum(x_grid * p * dx)
            var_p = np.sum((x_grid - mean_p)**2 * p * dx)
            if var_p > 1e-30:
                skew_arr[i] = np.sum((x_grid - mean_p)**3 * p * dx) / var_p**1.5

            # Tail ratio: mass in |x| > 2σ vs |x| < σ
            sigma_p = np.sqrt(max(var_p, 1e-30))
            tail_mask = np.abs(x_grid - mean_p) > 2 * sigma_p
            center_mask = np.abs(x_grid - mean_p) < sigma_p
            tail_mass = np.sum(p[tail_mask] * dx)
            center_mass = np.sum(p[center_mask] * dx)
            tail_arr[i] = tail_mass / max(center_mass, 1e-10)

        out["frac_alpha"] = alpha_arr
        out["frac_alpha_msd"] = alpha_msd_arr
        out["frac_alpha_waiting"] = alpha_wt_arr
        out["frac_hurst"] = hurst_arr
        out["frac_d_series"] = d_series
        out["ffpe_entropy"] = entropy_arr
        out["ffpe_skew"] = skew_arr
        out["ffpe_tail_ratio"] = tail_arr

        return out
