"""Vedic angular features: Madhava kernel + Aryabhata sine-difference phase.

Two ancient Indian mathematical frameworks applied to trading signals:

1. **Madhava Angular Kernel** (Kerala School, ~1400 CE)
   The infinite series for arctan/cos discovered by Madhava of Sangamagrama
   250 years before Gregory/Leibniz.  We use higher-order angular kernels
   on the feature hypersphere to capture nonlinear regime curvature beyond
   simple cosine similarity:
     K_M(x,y) = Σ_{n=0}^{P} (-1)^n θ^{2n} / (2n)!
   where θ = arccos(x̂ · ŷ).

2. **Aryabhata Sine-Difference Phase** (499 CE)
   Aryabhata's recursion Δ²sin(nh) ≈ -sin(nh)·h² is the oldest known
   discrete second-order recurrence for trigonometric functions.  We use
   it to track oscillation phase in detrended prices, providing entry
   timing within detected cycles.

References:
  - Madhava (~1400), "Tantrasangraha" (reconstructed)
  - Aryabhata (499), "Aryabhatiya" — Ganitapada verse 12
  - Plofker (2009), "Mathematics in India"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.base import Feature
from features.ramanujan import dominant_periods


# ---------------------------------------------------------------------------
# Madhava Angular Kernel
# ---------------------------------------------------------------------------

def madhava_kernel(x: np.ndarray, y: np.ndarray, order: int = 4) -> float:
    """Madhava angular kernel on the unit hypersphere.

    K_M(x, y) = Σ_{n=0}^{P} (-1)^n θ^{2n} / (2n)!

    where θ = arccos(x̂ · ŷ) and x̂, ŷ are unit-normalised.

    For P→∞ this converges to cos(θ), but finite-order truncation
    captures higher-order curvature that cosine similarity misses.

    Returns scalar kernel value in (-1, 1].
    """
    # Normalise to unit sphere
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm < 1e-30 or y_norm < 1e-30:
        return 0.0

    x_hat = x / x_norm
    y_hat = y / y_norm

    cos_theta = np.clip(np.dot(x_hat, y_hat), -1.0, 1.0)
    theta = math.acos(cos_theta)

    # Madhava series: cos(θ) = Σ (-1)^n θ^{2n} / (2n)!
    result = 0.0
    for n in range(order + 1):
        result += ((-1) ** n) * (theta ** (2 * n)) / math.factorial(2 * n)

    return float(result)


def angular_coherence(features: np.ndarray,
                      regime_centroids: dict[str, np.ndarray],
                      order: int = 4) -> tuple[str, float, float]:
    """Project features to hypersphere and compute Madhava kernel vs centroids.

    Parameters
    ----------
    features : 1-D array of current feature values
    regime_centroids : dict mapping regime name → centroid vector
    order : Madhava kernel expansion order

    Returns
    -------
    (best_regime, coherence_score, higher_order_residual)
      - best_regime: name of closest regime centroid
      - coherence_score: kernel value to closest centroid
      - higher_order_residual: difference between full kernel and cos(θ)
    """
    if not regime_centroids:
        return ("unknown", 0.0, 0.0)

    best_regime = "unknown"
    best_coherence = -2.0
    best_residual = 0.0

    for regime, centroid in regime_centroids.items():
        if len(centroid) != len(features):
            continue
        k_full = madhava_kernel(features, centroid, order=order)
        k_cos = madhava_kernel(features, centroid, order=1)  # cos(θ) only
        residual = abs(k_full - k_cos)

        if k_full > best_coherence:
            best_coherence = k_full
            best_regime = regime
            best_residual = residual

    return (best_regime, float(best_coherence), float(best_residual))


def update_regime_centroids(centroids: dict[str, np.ndarray],
                            features: np.ndarray,
                            regime: str,
                            decay: float = 0.99) -> dict[str, np.ndarray]:
    """Exponential moving average update of regime centroid vectors.

    centroid_new = decay * centroid_old + (1 - decay) * features

    Normalised to unit sphere after update.
    """
    if regime not in centroids:
        centroids[regime] = features.copy()
    else:
        centroids[regime] = decay * centroids[regime] + (1 - decay) * features
        norm = np.linalg.norm(centroids[regime])
        if norm > 1e-30:
            centroids[regime] /= norm

    return centroids


# ---------------------------------------------------------------------------
# Aryabhata Sine-Difference Phase
# ---------------------------------------------------------------------------

def aryabhata_phase(prices: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate oscillation phase using Aryabhata's sine-difference recursion.

    Aryabhata (499 CE) discovered:  Δ²sin(nh) ≈ -sin(nh) · h²

    Algorithm:
      1. Detrend price (remove linear trend)
      2. Normalise to [-1, 1]
      3. Apply Aryabhata recursion to estimate instantaneous phase φ(t)
      4. Phase velocity = Δφ(t)

    Parameters
    ----------
    prices : 1-D price array
    period : estimated dominant period (from Ramanujan periodogram)

    Returns
    -------
    (phase_array, phase_velocity_array) both in [0, 1]
    """
    n = len(prices)
    phase = np.full(n, np.nan)
    phase_vel = np.full(n, np.nan)

    if period < 2 or n < period + 2:
        return phase, phase_vel

    # Detrend: remove linear regression
    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, prices, 1)
    detrended = prices - np.polyval(coeffs, x)

    # Normalise to [-1, 1]
    amp = np.max(np.abs(detrended))
    if amp < 1e-10:
        return phase, phase_vel
    normalised = detrended / amp

    # h = 2π / period (angular step size)
    h = 2.0 * math.pi / period

    # Aryabhata recursion: estimate phase at each point
    # sin(φ) ≈ normalised[t], so φ = arcsin(normalised[t])
    # Clip to valid arcsin range
    clipped = np.clip(normalised, -0.9999, 0.9999)
    raw_phase = np.arcsin(clipped)  # [-π/2, π/2]

    # Use second differences to disambiguate quadrant
    # Aryabhata's relation: Δ²sin(nh) ≈ -sin(nh)·h²
    for i in range(1, n - 1):
        second_diff = normalised[i + 1] - 2 * normalised[i] + normalised[i - 1]
        predicted_sd = -normalised[i] * h**2

        # If second difference matches prediction, we're in rising phase
        # If signs disagree, we're in the descending arc → shift to [π/2, 3π/2]
        if second_diff * predicted_sd > 0:
            # Agreement: phase in [-π/2, π/2]
            pass
        else:
            # Disagreement: phase in [π/2, 3π/2]
            raw_phase[i] = math.pi - raw_phase[i]

    # Normalise to [0, 1] (one full cycle = 0 to 1)
    phase_norm = (raw_phase + math.pi / 2) / (2 * math.pi)
    phase_norm = np.mod(phase_norm, 1.0)
    phase[:] = phase_norm

    # Phase velocity
    phase_vel[1:] = np.diff(phase_norm)
    # Handle wraparound
    phase_vel[phase_vel > 0.5] -= 1.0
    phase_vel[phase_vel < -0.5] += 1.0

    return phase, phase_vel


# ---------------------------------------------------------------------------
# Feature class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VedicAngularFeatures(Feature):
    """Vedic angular features: Madhava coherence + Aryabhata phase.

    Produces per bar:
      - angular_coherence    : Madhava kernel value to best regime centroid
      - angular_regime       : index of best-matching regime (0/1/2)
      - madhava_higher_order : residual from higher-order kernel terms
      - aryabhata_phase      : oscillation phase ∈ [0, 1]
      - aryabhata_phase_velocity : rate of phase change
    """

    window: int = 60
    madhava_order: int = 4
    centroid_decay: float = 0.99

    @property
    def name(self) -> str:
        return "vedic_angular"

    @property
    def lookback(self) -> int:
        return self.window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].values.astype(np.float64)
        n = len(df)
        out = pd.DataFrame(index=df.index)

        # Compute log returns
        log_ret = np.diff(np.log(np.maximum(close, 1e-8)))
        log_ret = np.concatenate([[0.0], log_ret])

        # Build simple rolling features for angular analysis
        # (volatility, momentum, mean return — enough for regime separation)
        vol_arr = np.full(n, np.nan)
        mom_arr = np.full(n, np.nan)
        mean_arr = np.full(n, np.nan)

        for i in range(self.window, n):
            rets = log_ret[i - self.window + 1: i + 1]
            vol_arr[i] = np.std(rets, ddof=1)
            mom_arr[i] = np.sum(rets)
            mean_arr[i] = np.mean(rets)

        # Regime centroids (adaptive: subdiffusive, normal, superdiffusive)
        regime_names = ["subdiffusive", "normal", "superdiffusive"]
        centroids: dict[str, np.ndarray] = {}

        coherence_arr = np.full(n, np.nan)
        regime_arr = np.full(n, np.nan)
        residual_arr = np.full(n, np.nan)

        for i in range(self.window, n):
            if np.isnan(vol_arr[i]):
                continue

            feat_vec = np.array([vol_arr[i], mom_arr[i], mean_arr[i]])

            if centroids:
                best_regime, coh, resid = angular_coherence(
                    feat_vec, centroids, order=self.madhava_order,
                )
                coherence_arr[i] = coh
                regime_arr[i] = regime_names.index(best_regime) if best_regime in regime_names else 1
                residual_arr[i] = resid
            else:
                coherence_arr[i] = 0.0
                regime_arr[i] = 1  # default: normal
                residual_arr[i] = 0.0

            # Classify into regime based on simple momentum heuristic
            # (actual α-based classification is done in signals.py)
            rets = log_ret[i - self.window + 1: i + 1]
            autocorr = np.corrcoef(rets[:-1], rets[1:])[0, 1] if len(rets) > 2 else 0
            if autocorr < -0.1:
                regime_label = "subdiffusive"
            elif autocorr > 0.1:
                regime_label = "superdiffusive"
            else:
                regime_label = "normal"

            centroids = update_regime_centroids(
                centroids, feat_vec, regime_label, decay=self.centroid_decay,
            )

        # Aryabhata phase: detect dominant period, then track phase
        phase_arr = np.full(n, np.nan)
        phase_vel_arr = np.full(n, np.nan)

        for i in range(self.window * 2, n):
            window_prices = close[i - self.window: i]
            window_rets = log_ret[i - self.window + 1: i + 1]

            # Detect dominant period using Ramanujan periodogram
            periods = dominant_periods(
                window_rets - window_rets.mean(),
                max_period=min(32, self.window // 2),
                top_k=1,
            )
            period = periods[0] if periods else self.window // 4

            # Track phase in recent window
            ph, pv = aryabhata_phase(window_prices, period)

            # Use the last value (current bar)
            if not np.isnan(ph[-1]):
                phase_arr[i] = ph[-1]
            if not np.isnan(pv[-1]):
                phase_vel_arr[i] = pv[-1]

        out["angular_coherence"] = coherence_arr
        out["angular_regime"] = regime_arr
        out["madhava_higher_order"] = residual_arr
        out["aryabhata_phase"] = phase_arr
        out["aryabhata_phase_velocity"] = phase_vel_arr

        return out
