"""Ramanujan sum filter bank for periodicity detection.

The Ramanujan sum c_q(n) is defined as:
    c_q(n) = sum_{k=1..q, gcd(k,q)=1} exp(2*pi*i*k*n/q)

For integer n, c_q(n) is real-valued (via the Mobius function):
    c_q(n) = mu(q/gcd(n,q)) * phi(q) / phi(q/gcd(n,q))

Key property: c_q(n) detects period-q components in integer sequences,
acting as a narrowband filter centered on period q.  A bank of these
filters decomposes any signal into its periodic components without
requiring Fourier analysis — resolution is exact at integer periods.

This gives us an edge over FFT for detecting trading-day cycles
(weekly=5, fortnightly=10, monthly=21, expiry=varies) because
Ramanujan sums have perfect integer-period selectivity.

References:
  - Ramanujan (1918), "On certain trigonometrical sums"
  - Planat (2002), "Ramanujan sums for signal processing"
  - Mainardi et al. (2008), "Ramanujan sums via number-theoretic methods"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from qlx.features.base import Feature


# ---------------------------------------------------------------------------
# Number-theoretic helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _euler_phi(n: int) -> int:
    """Euler's totient function phi(n)."""
    result = n
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


@lru_cache(maxsize=4096)
def _mobius(n: int) -> int:
    """Mobius function mu(n)."""
    if n == 1:
        return 1
    count = 0
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            temp //= p
            count += 1
            if temp % p == 0:
                return 0  # p^2 divides n
        p += 1
    if temp > 1:
        count += 1
    return (-1) ** count


def ramanujan_sum(q: int, n: int) -> float:
    """Compute c_q(n) using the Mobius function formula.

    c_q(n) = mu(q/d) * phi(q) / phi(q/d) where d = gcd(n, q)
    """
    if q <= 0:
        return 0.0
    d = math.gcd(abs(n), q)
    q_over_d = q // d
    mu_val = _mobius(q_over_d)
    if mu_val == 0:
        return 0.0
    return mu_val * _euler_phi(q) / max(_euler_phi(q_over_d), 1)


# ---------------------------------------------------------------------------
# Filter bank operations
# ---------------------------------------------------------------------------

def build_filter(q: int, length: int) -> np.ndarray:
    """Build a FIR filter for period q using Ramanujan sums.

    The filter h_q[n] = c_q(n) / q extracts the period-q component.
    """
    filt = np.array([ramanujan_sum(q, n) for n in range(length)], dtype=np.float64)
    return filt / q


def ramanujan_periodogram(signal: np.ndarray, max_period: int = 64) -> np.ndarray:
    """Compute the Ramanujan periodogram.

    For each candidate period q in [1, max_period], compute the energy of
    the period-q component:
        E_q = |<signal, c_q>|^2 / (N * phi(q)^2)

    Returns array of shape (max_period,) with energy at each period.
    """
    N = len(signal)
    energies = np.zeros(max_period)

    for q in range(1, max_period + 1):
        cq = np.array([ramanujan_sum(q, n) for n in range(N)], dtype=np.float64)
        projection = np.dot(signal, cq)
        phi_q = _euler_phi(q)
        if phi_q > 0:
            energies[q - 1] = (projection**2) / (N * phi_q**2)

    return energies


def dominant_periods(
    signal: np.ndarray, max_period: int = 64, top_k: int = 3,
) -> list[int]:
    """Find the top-k dominant periods in a signal."""
    energies = ramanujan_periodogram(signal, max_period)
    energies[0] = 0  # skip DC component
    indices = np.argsort(energies)[::-1][:top_k]
    return [int(i + 1) for i in indices if energies[i] > 0]


def filter_bank_decompose(
    signal: np.ndarray,
    periods: list[int],
) -> dict[int, np.ndarray]:
    """Decompose a signal using a bank of Ramanujan filters.

    Returns dict mapping period -> filtered component (same length as signal).
    Uses circular convolution via FFT for efficiency.
    """
    N = len(signal)
    components = {}

    for q in periods:
        filt = build_filter(q, q)
        sig_fft = np.fft.fft(signal, N)
        filt_fft = np.fft.fft(filt, N)
        components[q] = np.real(np.fft.ifft(sig_fft * filt_fft))

    return components


# ---------------------------------------------------------------------------
# Feature: Ramanujan Periodicity Detector (OHLCV-compatible)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RamanujanPeriodicity(Feature):
    """Detect cyclical patterns in price and volume using Ramanujan sums.

    Produces for each bar:
      - dominant_period: strongest detected period in returns
      - period_energy: energy of dominant period (signal strength)
      - cycle_phase: current position within dominant cycle (0-1)
      - vol_period: dominant period in volume
      - period_stability: consistency of detected period over time
    """

    window: int = 64
    max_period: int = 32

    @property
    def name(self) -> str:
        return f"ramanujan_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].values
        volume = df["Volume"].values
        n = len(df)
        out = pd.DataFrame(index=df.index)

        dom_period = np.full(n, np.nan)
        period_energy = np.full(n, np.nan)
        cycle_phase = np.full(n, np.nan)
        vol_period = np.full(n, np.nan)
        period_stability = np.full(n, np.nan)

        recent_periods: list[int] = []

        for i in range(self.window, n):
            # Log returns in window
            window_prices = close[i - self.window : i]
            log_ret = np.diff(np.log(np.maximum(window_prices, 1e-8)))
            log_ret = log_ret - log_ret.mean()

            # Ramanujan periodogram
            energies = ramanujan_periodogram(log_ret, self.max_period)
            energies[0] = 0  # skip DC

            best_q = int(np.argmax(energies)) + 1
            dom_period[i] = best_q
            period_energy[i] = energies[best_q - 1]

            # Current phase within the dominant cycle
            cycle_phase[i] = ((i - self.window) % best_q) / best_q

            # Volume periodicity
            vol_window = volume[i - self.window : i].astype(float)
            if vol_window.std() > 0:
                vol_norm = (vol_window - vol_window.mean()) / vol_window.std()
                vol_energies = ramanujan_periodogram(vol_norm, self.max_period)
                vol_energies[0] = 0
                vol_period[i] = int(np.argmax(vol_energies)) + 1

            # Period stability: how consistently the same period is detected
            recent_periods.append(best_q)
            if len(recent_periods) > self.window:
                recent_periods = recent_periods[-self.window :]
            if len(recent_periods) >= 10:
                mode_count = max(
                    recent_periods.count(p) for p in set(recent_periods)
                )
                period_stability[i] = mode_count / len(recent_periods)

        out["dominant_period"] = dom_period
        out["period_energy"] = period_energy
        out["cycle_phase"] = cycle_phase
        out["vol_period"] = vol_period
        out["period_stability"] = period_stability

        return out
