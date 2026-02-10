"""Ramanujan mock theta function features for hidden periodicity detection.

Mock theta functions are q-series that mimic theta functions near the unit
circle but fail to be true modular forms.  They detect quasi-periodic
structures invisible to standard Fourier / FFT analysis.

Three third-order mock thetas are implemented:
  f(q) = Σ q^{n²} / Π_{k=1}^{n} (1+q^k)²
  φ(q) = Σ q^{n²} / Π_{k=1}^{n} (1+q^k)
  χ(q) = Σ q^{n²} / Π_{k=1}^{n} (1-q^k+q^{2k})

The return-to-q mapping transforms market returns into the nome domain
where these functions operate:
  q(r) = exp(-π / (1 + |r|/σ))

Additionally implements Ramanujan's continued-fraction volatility
distortion operator (from user's LCFT research):
  R(V) = exp(-α·V) / (1 + V/(1 + 2V/(1 + 3V/...)))

References:
  - Ramanujan (1920), "Mock theta functions" (lost notebook)
  - Andrews & Garvan (2012), "Ramanujan's lost notebook: Part IV"
  - Zwegers (2002), "Mock theta functions" (PhD thesis)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlaxmi.features.base import Feature


# ---------------------------------------------------------------------------
# Core mock theta functions
# ---------------------------------------------------------------------------

def mock_theta_f(q: float, n_terms: int = 50) -> float:
    """Third-order mock theta f(q).

    f(q) = Σ_{n=0}^{N} q^{n²} / Π_{k=1}^{n} (1 + q^k)²

    Converges rapidly for |q| < 1.  For |q| >= 1, returns 0.
    """
    if abs(q) >= 1.0:
        return 0.0

    total = 1.0  # n=0 term: q^0 / empty_product = 1
    prod = 1.0   # running product Π(1+q^k)²

    for n in range(1, n_terms + 1):
        prod *= (1.0 + q**n) ** 2
        if prod == 0 or abs(prod) < 1e-300:
            break
        qn2 = q ** (n * n)
        if abs(qn2) < 1e-300:
            break
        total += qn2 / prod

    return total


def mock_theta_phi(q: float, n_terms: int = 50) -> float:
    """Third-order mock theta φ(q).

    φ(q) = Σ_{n=0}^{N} q^{n²} / Π_{k=1}^{n} (1 + q^k)
    """
    if abs(q) >= 1.0:
        return 0.0

    total = 1.0
    prod = 1.0

    for n in range(1, n_terms + 1):
        prod *= (1.0 + q**n)
        if prod == 0 or abs(prod) < 1e-300:
            break
        qn2 = q ** (n * n)
        if abs(qn2) < 1e-300:
            break
        total += qn2 / prod

    return total


def mock_theta_chi(q: float, n_terms: int = 50) -> float:
    """Third-order mock theta χ(q).

    χ(q) = Σ_{n=0}^{N} q^{n²} / Π_{k=1}^{n} (1 - q^k + q^{2k})
    """
    if abs(q) >= 1.0:
        return 0.0

    total = 1.0
    prod = 1.0

    for n in range(1, n_terms + 1):
        qk = q**n
        prod *= (1.0 - qk + qk**2)
        if prod == 0 or abs(prod) < 1e-300:
            break
        qn2 = q ** (n * n)
        if abs(qn2) < 1e-300:
            break
        total += qn2 / prod

    return total


# ---------------------------------------------------------------------------
# Return → q-domain mapping
# ---------------------------------------------------------------------------

def return_to_q(returns: np.ndarray, sigma: float) -> np.ndarray:
    """Map financial returns to the nome domain for mock theta evaluation.

    q(r) = exp(-π / (1 + |r|/σ))

    Properties:
      - q ∈ (0, exp(-π)) ≈ (0, 0.0432) for r=0
      - q → 1 as |r|/σ → ∞ (large moves push q toward unit circle)
      - Smooth, monotonically increasing in |r|
    """
    sigma = max(sigma, 1e-10)
    abs_r = np.abs(returns)
    q = np.exp(-math.pi / (1.0 + abs_r / sigma))
    return q


# ---------------------------------------------------------------------------
# Ramanujan continued-fraction volatility distortion
# ---------------------------------------------------------------------------

def ramanujan_volatility_distortion(vol: float, alpha: float = 0.2,
                                    depth: int = 20) -> float:
    """Continued-fraction volatility transform.

    R(V) = exp(-α·V) / CF(V)
    where CF(V) = 1 + V/(1 + 2V/(1 + 3V/...))

    The continued fraction compresses high volatility nonlinearly,
    producing a signal that's sensitive to vol regime transitions.
    """
    # Evaluate continued fraction bottom-up
    cf = 1.0
    for k in range(depth, 0, -1):
        cf = 1.0 + k * vol / max(cf, 1e-30)

    return math.exp(-alpha * vol) / max(cf, 1e-30)


# ---------------------------------------------------------------------------
# Feature class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MockThetaFeatures(Feature):
    """Mock theta function features for hidden periodicity detection.

    Produces per bar:
      - mock_theta_f      : f(q) value from rolling return-to-q
      - mock_theta_phi    : φ(q) value
      - mock_theta_chi    : χ(q) value
      - mock_theta_divergence : |Δ(f/φ)| — regime transition speed
      - mock_theta_ratio  : f/φ near-modular ratio
      - vol_distortion    : Ramanujan continued-fraction vol transform
    """

    window: int = 20
    n_terms: int = 50

    @property
    def name(self) -> str:
        return "mock_theta"

    @property
    def lookback(self) -> int:
        return self.window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].values.astype(np.float64)
        n = len(df)
        out = pd.DataFrame(index=df.index)

        # Log returns
        log_ret = np.diff(np.log(np.maximum(close, 1e-8)))
        log_ret = np.concatenate([[0.0], log_ret])

        # Output arrays
        f_arr = np.full(n, np.nan)
        phi_arr = np.full(n, np.nan)
        chi_arr = np.full(n, np.nan)
        div_arr = np.full(n, np.nan)
        ratio_arr = np.full(n, np.nan)
        vd_arr = np.full(n, np.nan)

        prev_ratio = None

        for i in range(self.window, n):
            rets = log_ret[i - self.window + 1: i + 1]
            sigma = np.std(rets, ddof=1) if len(rets) > 1 else 1e-8
            sigma = max(sigma, 1e-8)

            # Aggregate q from mean absolute return in window
            mean_abs_ret = np.mean(np.abs(rets))
            q_val = math.exp(-math.pi / (1.0 + mean_abs_ret / sigma))

            f_val = mock_theta_f(q_val, self.n_terms)
            phi_val = mock_theta_phi(q_val, self.n_terms)
            chi_val = mock_theta_chi(q_val, self.n_terms)

            f_arr[i] = f_val
            phi_arr[i] = phi_val
            chi_arr[i] = chi_val

            # Ratio f/φ (near-modular)
            current_ratio = f_val / max(phi_val, 1e-30)
            ratio_arr[i] = current_ratio

            # Divergence: rate of change of ratio
            if prev_ratio is not None:
                div_arr[i] = abs(current_ratio - prev_ratio)
            prev_ratio = current_ratio

            # Volatility distortion
            vol = sigma * math.sqrt(252)  # annualise
            vd_arr[i] = ramanujan_volatility_distortion(vol)

        out["mock_theta_f"] = f_arr
        out["mock_theta_phi"] = phi_arr
        out["mock_theta_chi"] = chi_arr
        out["mock_theta_divergence"] = div_arr
        out["mock_theta_ratio"] = ratio_arr
        out["vol_distortion"] = vd_arr

        return out
