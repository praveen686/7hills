"""FTI (Follow Through Index) — Numba JIT + vectorbtpro streaming indicator.

Govinda Khalsa's trend quality metric, ported from the C++ reference
implementation in research_artefacts/indicators/timothy_masters/FTI.CPP.

Sections:
  A. Numba kernels (_quickselect_nb, fti_coefs_nb,
     fti_process_window_scratch_nb, fti_process_window_nb, fti_1d_nb, fti_nb)
  B. vectorbtpro IndicatorFactory (FTI_VBT)
  C. QuantLaxmi Feature subclass (FollowThroughIndex)
  D. Streaming (scratch-buffer API)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.indicators.factory import IndicatorFactory
import vectorbtpro.utils.chunking as ch
from vectorbtpro.base import chunking as base_ch
from numba import prange

from quantlaxmi.features.base import Feature


# ── Section A: Numba Kernels ────────────────────────────────────────────────


@register_jitted(cache=True)
def _quickselect_nb(arr, n, k):
    """Find k-th smallest element in arr[:n].  Modifies arr[:n] in-place.

    Hoare-partition quickselect, O(n) expected.
    Used for beta-fractile channel width instead of O(n log n) full sort.
    """
    lo = 0
    hi = n - 1
    while lo < hi:
        pivot = arr[k]
        i = lo
        j = hi
        while True:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
            else:
                break
        if j < k:
            lo = i
        if k < i:
            hi = j
    return arr[k]


@register_jitted(cache=True)
def fti_coefs_nb(min_period, max_period, half_length):
    """Pre-compute FIR lowpass filter coefficients for all periods.

    Returns 2D array shape (n_periods, half_length+1).
    Sinc kernel -> Blackman-Harris window -> normalize to unit DC gain.

    Matches C++ FTI::find_coefs() in FTI.CPP lines 158-207.
    """
    n_periods = max_period - min_period + 1
    H = half_length
    coefs = np.empty((n_periods, H + 1))

    d0 = 0.35577019
    d1 = 0.2436983
    d2 = 0.07211497
    d3 = 0.00630165
    pi = math.pi

    for ip in range(n_periods):
        period = min_period + ip

        # Step 1: ideal sinc kernel
        fact = 2.0 / period
        coefs[ip, 0] = fact
        fact_pi = fact * pi
        for i in range(1, H + 1):
            coefs[ip, i] = math.sin(i * fact_pi) / (i * pi)

        # Step 2: taper endpoint
        coefs[ip, H] *= 0.5

        # Step 3: Blackman-Harris window + accumulate normaliser
        sumg = coefs[ip, 0]
        for i in range(1, H + 1):
            w = d0
            f = i * pi / H
            w += 2.0 * d1 * math.cos(f)
            w += 2.0 * d2 * math.cos(2.0 * f)
            w += 2.0 * d3 * math.cos(3.0 * f)
            coefs[ip, i] *= w
            sumg += 2.0 * coefs[ip, i]

        # Step 4: normalise to unity DC gain
        for i in range(H + 1):
            coefs[ip, i] /= sumg

    return coefs


@register_jitted(cache=True)
def fti_process_window_scratch_nb(
    prices, coefs, half_length, min_period, max_period,
    beta, noise_cut, use_log,
    y, filtered_vals, width_vals, fti_vals, diff_work, leg_work,
):
    """Core FTI algorithm using preallocated scratch buffers.

    Faithful port of C++ FTI::process() (FTI.CPP lines 218-402).
    Fixes the LS extrapolation bug present in masters.py.
    Uses quickselect O(n) for beta-fractile instead of full sort O(n log n).

    Parameters
    ----------
    prices : 1D array, chronological (oldest first, newest last).
    coefs  : 2D array from fti_coefs_nb.
    y, filtered_vals, width_vals, fti_vals, diff_work, leg_work :
        Preallocated scratch arrays (use fti_alloc_scratch_nb to create).

    Returns
    -------
    (filtered, best_period, best_width, best_fti) — all scalars.
    """
    lookback = len(prices)
    H = half_length
    L = lookback
    n_periods = max_period - min_period + 1

    if L < H + 2:
        return np.nan, np.nan, np.nan, np.nan

    # ── Copy into work array with room for extrapolation ──
    if use_log:
        for i in range(L):
            y[i] = math.log(max(prices[i], 1e-12))
    else:
        for i in range(L):
            y[i] = prices[i]

    # ── Least-squares extrapolation for zero-lag ──
    # Matches C++ FTI.CPP lines 256-271 EXACTLY.
    # NOTE: masters.py has a bug here — it pairs x=0 with the OLDEST
    # point instead of the NEWEST.  We follow the C++ (correct) version.
    xmean = -0.5 * H
    ymean = 0.0
    for i in range(H + 1):
        ymean += y[L - 1 - i]
    ymean /= (H + 1.0)

    xsq = 0.0
    xy = 0.0
    for i in range(H + 1):
        xdiff = -i - xmean          # x=0 for newest (i=0)
        ydiff = y[L - 1 - i] - ymean  # y[L-1] is newest
        xsq += xdiff * xdiff
        xy += xdiff * ydiff
    slope = xy / (xsq + 1e-30)

    for i in range(H):
        y[L + i] = (i + 1.0 - xmean) * slope + ymean

    # ── Per-period processing ──
    channel_len = L - H

    # Zero fti_vals (may contain stale data from previous call)
    for ip in range(n_periods):
        fti_vals[ip] = 0.0

    for ip in range(n_periods):
        extreme_type = 0
        extreme_value = 0.0
        n_legs = 0
        longest_leg = 0.0
        prior = 0.0

        for iy in range(H, L):
            # Symmetric FIR convolution (FTI.CPP lines 294-296)
            s = coefs[ip, 0] * y[iy]
            for i in range(1, H + 1):
                s += coefs[ip, i] * (y[iy + i] + y[iy - i])

            if iy == L - 1:
                filtered_vals[ip] = s

            diff_work[iy - H] = abs(y[iy] - s)

            # ── Swing leg detection (FTI.CPP lines 307-344) ──
            if iy == H:
                extreme_type = 0
                extreme_value = s
                n_legs = 0
                longest_leg = 0.0
            elif extreme_type == 0:
                if s > extreme_value:
                    extreme_type = -1
                elif s < extreme_value:
                    extreme_type = 1
            elif iy == L - 1:
                if extreme_type != 0:
                    leg = abs(extreme_value - s)
                    leg_work[n_legs] = leg
                    n_legs += 1
                    if leg > longest_leg:
                        longest_leg = leg
            else:
                if extreme_type == 1 and s > prior:
                    leg = extreme_value - prior
                    leg_work[n_legs] = leg
                    n_legs += 1
                    if leg > longest_leg:
                        longest_leg = leg
                    extreme_type = -1
                    extreme_value = prior
                elif extreme_type == -1 and s < prior:
                    leg = prior - extreme_value
                    leg_work[n_legs] = leg
                    n_legs += 1
                    if leg > longest_leg:
                        longest_leg = leg
                    extreme_type = 1
                    extreme_value = prior

            prior = s

        # Channel width: beta-fractile via quickselect O(n)
        idx = int(beta * (channel_len + 1)) - 1
        if idx < 0:
            idx = 0
        if idx >= channel_len:
            idx = channel_len - 1
        width_vals[ip] = _quickselect_nb(diff_work, channel_len, idx)

        # FTI = mean(non-noise legs) / width
        if n_legs > 0 and longest_leg > 0.0:
            noise_level = noise_cut * longest_leg
            leg_sum = 0.0
            n_big = 0
            for i in range(n_legs):
                if leg_work[i] > noise_level:
                    leg_sum += leg_work[i]
                    n_big += 1
            if n_big > 0:
                fti_vals[ip] = (leg_sum / n_big) / (width_vals[ip] + 1e-5)

    # Find best period (max FTI) — matches C++ get_sorted_index(0)
    best_ip = 0
    best_val = fti_vals[0]
    for ip in range(1, n_periods):
        if fti_vals[ip] > best_val:
            best_val = fti_vals[ip]
            best_ip = ip

    best_period = float(min_period + best_ip)
    best_fti = fti_vals[best_ip]

    # Convert filtered value and width from log-space to price-space
    if use_log:
        best_filtered = math.exp(filtered_vals[best_ip])
        v = filtered_vals[best_ip]
        w = width_vals[best_ip]
        best_width = 0.5 * (math.exp(v + w) - math.exp(v - w))
    else:
        best_filtered = filtered_vals[best_ip]
        best_width = width_vals[best_ip]

    return best_filtered, best_period, best_width, best_fti


@register_jitted(cache=True)
def fti_process_window_nb(
    prices, coefs, half_length, min_period, max_period,
    beta, noise_cut, use_log,
):
    """Core FTI algorithm for a single lookback window.

    Allocates scratch internally — for hot-path streaming, prefer
    fti_process_window_scratch_nb with preallocated buffers.

    Faithful port of C++ FTI::process() (FTI.CPP lines 218-402).
    Fixes the LS extrapolation bug present in masters.py.
    """
    lookback = len(prices)
    H = half_length
    L = lookback
    n_periods = max_period - min_period + 1

    if L < H + 2:
        return np.nan, np.nan, np.nan, np.nan

    channel_len = L - H
    y = np.empty(L + H)
    filtered_vals = np.empty(n_periods)
    width_vals = np.empty(n_periods)
    fti_vals = np.empty(n_periods)
    diff_work = np.empty(channel_len)
    leg_work = np.empty(channel_len)

    return fti_process_window_scratch_nb(
        prices, coefs, half_length, min_period, max_period,
        beta, noise_cut, use_log,
        y, filtered_vals, width_vals, fti_vals, diff_work, leg_work,
    )


@register_jitted(cache=True)
def fti_1d_nb(
    close, coefs, lookback, half_length, min_period, max_period,
    beta, noise_cut, use_log,
):
    """Rolling FTI over 1D close array.

    Preallocates scratch buffers once, reused across all bars.
    Returns (filtered, best_period, best_width, best_fti) each of length N.
    First lookback-1 values are NaN.
    """
    N = len(close)
    filtered = np.full(N, np.nan)
    best_period = np.full(N, np.nan)
    best_width = np.full(N, np.nan)
    best_fti = np.full(N, np.nan)

    H = half_length
    L = lookback
    n_periods = max_period - min_period + 1
    channel_len = L - H

    # Preallocate scratch — reused every bar (zero-alloc hot path)
    y = np.empty(L + H)
    fv = np.empty(n_periods)
    wv = np.empty(n_periods)
    ftv = np.empty(n_periods)
    dw = np.empty(channel_len)
    lw = np.empty(channel_len)

    for i in range(lookback - 1, N):
        window = close[i - lookback + 1: i + 1]
        f, p, w, fval = fti_process_window_scratch_nb(
            window, coefs, half_length, min_period, max_period,
            beta, noise_cut, use_log,
            y, fv, wv, ftv, dw, lw,
        )
        filtered[i] = f
        best_period[i] = p
        best_width[i] = w
        best_fti[i] = fval

    return filtered, best_period, best_width, best_fti


@register_jitted(cache=True, tags={"can_parallel"})
def fti_nb(
    close_2d, lookback, half_length, min_period, max_period,
    beta, noise_cut, use_log,
):
    """2D FTI with parallel column processing.  All params are scalars."""
    max_period_c = min(max_period, 2 * half_length)
    coefs = fti_coefs_nb(min_period, max_period_c, half_length)

    n_rows = close_2d.shape[0]
    n_cols = close_2d.shape[1]
    filtered = np.empty((n_rows, n_cols))
    best_period_out = np.empty((n_rows, n_cols))
    best_width = np.empty((n_rows, n_cols))
    best_fti = np.empty((n_rows, n_cols))

    for col in prange(n_cols):
        f, p, w, fv = fti_1d_nb(
            close_2d[:, col], coefs, lookback, half_length,
            min_period, max_period_c, beta, noise_cut, use_log,
        )
        filtered[:, col] = f
        best_period_out[:, col] = p
        best_width[:, col] = w
        best_fti[:, col] = fv

    return filtered, best_period_out, best_width, best_fti


# ── Section B: vectorbtpro IndicatorFactory ──────────────────────────────────


def _fti_apply(
    close, lookback, min_period, max_period,
    half_length=33, beta=0.95, noise_cut=0.20, use_log=True,
):
    """Python wrapper for IndicatorFactory — handles array param broadcasting."""
    n_rows, n_cols = close.shape

    lb_arr = np.atleast_1d(np.asarray(lookback, dtype=np.int64))
    minp_arr = np.atleast_1d(np.asarray(min_period, dtype=np.int64))
    maxp_arr = np.atleast_1d(np.asarray(max_period, dtype=np.int64))
    hl = int(half_length)

    filtered = np.empty((n_rows, n_cols))
    best_period = np.empty((n_rows, n_cols))
    best_width = np.empty((n_rows, n_cols))
    best_fti = np.empty((n_rows, n_cols))

    for col in range(n_cols):
        lb = int(lb_arr[col % len(lb_arr)])
        minp = int(minp_arr[col % len(minp_arr)])
        maxp = min(int(maxp_arr[col % len(maxp_arr)]), 2 * hl)

        coefs = fti_coefs_nb(minp, maxp, hl)
        f, p, w, fv = fti_1d_nb(
            close[:, col], coefs, lb, hl, minp, maxp,
            float(beta), float(noise_cut), bool(use_log),
        )
        filtered[:, col] = f
        best_period[:, col] = p
        best_width[:, col] = w
        best_fti[:, col] = fv

    return filtered, best_period, best_width, best_fti


FTI_VBT = IndicatorFactory(
    class_name="FTI",
    short_name="fti",
    input_names=["close"],
    param_names=["lookback", "min_period", "max_period"],
    output_names=["filtered", "best_period", "best_width", "best_fti"],
).with_apply_func(
    _fti_apply,
    kwargs_as_args=["half_length", "beta", "noise_cut", "use_log"],
    lookback=200,
    half_length=33,
    min_period=5,
    max_period=65,
    beta=0.95,
    noise_cut=0.20,
    use_log=True,
)


# ── Section C: QuantLaxmi Feature Subclass ───────────────────────────────────


@dataclass(frozen=True)
class FollowThroughIndex(Feature):
    """FTI trend quality metric (Numba-accelerated).

    Produces for each bar:
      - filtered   : lowpass-filtered price at the best period
      - best_period: period with the highest FTI
      - best_width : channel width at the best period (price-space)
      - best_fti   : Follow Through Index value
    """

    lookback_window: int = 200
    min_period: int = 5
    max_period: int = 65
    half_length: int = 33
    beta: float = 0.95
    noise_cut: float = 0.20

    @property
    def name(self) -> str:
        return f"fti_{self.lookback_window}"

    @property
    def lookback(self) -> int:
        return self.lookback_window

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].values.astype(np.float64)
        maxp = min(self.max_period, 2 * self.half_length)
        coefs = fti_coefs_nb(self.min_period, maxp, self.half_length)
        f, p, w, fv = fti_1d_nb(
            close, coefs, self.lookback_window, self.half_length,
            self.min_period, maxp, self.beta, self.noise_cut, True,
        )
        out = pd.DataFrame(index=df.index)
        out["filtered"] = f
        out["best_period"] = p
        out["best_width"] = w
        out["best_fti"] = fv
        return out


# ── Section D: Streaming (scratch-buffer API) ───────────────────────────────


@register_jitted(cache=True)
def fti_alloc_scratch_nb(lookback, half_length, min_period, max_period):
    """Allocate reusable scratch buffers for streaming FTI.

    Call once per stream/instrument.  Pass returned tuple to
    fti_process_window_scratch_nb on each completed bar.

    Returns
    -------
    (y, filtered_vals, width_vals, fti_vals, diff_work, leg_work)
    """
    max_period = min(max_period, 2 * half_length)
    n_periods = max_period - min_period + 1
    channel_len = lookback - half_length
    return (
        np.empty(lookback + half_length),   # y
        np.empty(n_periods),                # filtered_vals
        np.empty(n_periods),                # width_vals
        np.empty(n_periods),                # fti_vals
        np.empty(channel_len),              # diff_work
        np.empty(channel_len),              # leg_work
    )


def fti_stream_process(price_buffer, coefs, half_length=33,
                       min_period=5, max_period=65,
                       beta=0.95, noise_cut=0.20, use_log=True,
                       scratch=None):
    """Process one bar of streaming FTI.

    Parameters
    ----------
    price_buffer : 1D array of recent prices (chronological, length = lookback).
    coefs        : Pre-computed coefficients from fti_coefs_nb().
    scratch      : Optional tuple from fti_alloc_scratch_nb() for zero-alloc
                   hot path.  If None, scratch is allocated internally.

    Returns
    -------
    dict with keys: filtered, best_period, best_width, best_fti.
    """
    max_period = min(max_period, 2 * half_length)
    if scratch is not None:
        y, fv, wv, ftv, dw, lw = scratch
        f, p, w, fval = fti_process_window_scratch_nb(
            price_buffer, coefs, half_length, min_period, max_period,
            beta, noise_cut, use_log,
            y, fv, wv, ftv, dw, lw,
        )
    else:
        f, p, w, fval = fti_process_window_nb(
            price_buffer, coefs, half_length, min_period, max_period,
            beta, noise_cut, use_log,
        )
    return {
        "filtered": f,
        "best_period": p,
        "best_width": w,
        "best_fti": fval,
    }
