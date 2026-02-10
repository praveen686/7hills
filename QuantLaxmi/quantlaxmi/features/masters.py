"""Timothy Masters indicator library — Python port.

Ported from the C++ reference implementation in
research_artefacts/indicators/timothy_masters/.

Indicators:
  - FTI (Follow Through Index) — Govinda Khalsa's trend quality metric
  - Mutual Information — nonlinear serial dependency
  - ADX (Average Directional Index) — Wilder's trend strength
  - Detrended RSI — RSI residual after removing trend
  - Price Variance Ratio — multi-scale volatility clustering
  - Price Intensity — directional conviction (close-open)/TR
"""

from __future__ import annotations

import math
import numpy as np


# ── FTI (Follow Through Index) ──────────────────────────────────────────────

_BH_D = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])


def _fti_coefs(period: int, half_length: int) -> np.ndarray:
    """Compute FIR lowpass filter coefficients (Otnes/Blackman-Harris windowed sinc)."""
    H = half_length
    c = np.zeros(H + 1)

    # Step 1: ideal sinc kernel
    fact = 2.0 / period
    c[0] = fact
    fact_pi = fact * math.pi
    for i in range(1, H + 1):
        c[i] = math.sin(i * fact_pi) / (i * math.pi)

    # Step 2: taper endpoint
    c[H] *= 0.5

    # Step 3: Blackman-Harris window + accumulate normaliser
    sumg = c[0]
    for i in range(1, H + 1):
        w = _BH_D[0]
        f = i * math.pi / H
        for j in range(1, 4):
            w += 2.0 * _BH_D[j] * math.cos(j * f)
        c[i] *= w
        sumg += 2.0 * c[i]

    # Step 4: normalise to unity DC gain
    c /= sumg
    return c


def fti_single(
    prices: np.ndarray,
    *,
    use_log: bool = True,
    min_period: int = 5,
    max_period: int = 65,
    half_length: int = 32,
    beta: float = 0.95,
    noise_cut: float = 0.20,
) -> dict:
    """Compute FTI for a single lookback window ending at the last price.

    Parameters
    ----------
    prices : 1-D array, chronological (oldest first, newest last).
             Length must be >= half_length + 2.
    use_log : work in log-price space (recommended).
    min_period, max_period : filter period range.
    half_length : filter half-length (2*half_length+1 > max_period).
    beta : fractile for channel width (0.95 = 95th percentile).
    noise_cut : fraction of longest leg below which a leg is noise (0.20).

    Returns
    -------
    dict with keys:
      best_fti      — FTI of the best period
      best_period   — period with highest FTI
      best_filtered — filtered price at best period
      best_width    — channel width at best period
      fti_array     — FTI for each period in [min_period, max_period]
    """
    lookback = len(prices)
    if lookback < half_length + 2:
        return {"best_fti": np.nan, "best_period": min_period,
                "best_filtered": np.nan, "best_width": np.nan,
                "fti_array": np.array([])}

    max_period = min(max_period, 2 * half_length)

    # Copy into work array (chronological: oldest=0, newest=lookback-1)
    y = np.empty(lookback + half_length)
    if use_log:
        y[:lookback] = np.log(np.maximum(prices, 1e-12))
    else:
        y[:lookback] = prices

    # Least-squares extrapolation for zero-lag
    H = half_length
    L = lookback
    seg = y[L - 1 - H: L]  # most recent H+1 points
    xs = -np.arange(H + 1, dtype=np.float64)
    xmean = xs.mean()
    ymean = seg.mean()
    xd = xs - xmean
    yd = seg - ymean
    slope = np.dot(xd, yd) / (np.dot(xd, xd) + 1e-30)
    for k in range(H):
        y[L + k] = (k + 1.0 - xmean) * slope + ymean

    # Pre-compute filter coefficients for all periods
    n_periods = max_period - min_period + 1
    all_coefs = [_fti_coefs(p, H) for p in range(min_period, max_period + 1)]

    filtered_vals = np.zeros(n_periods)
    width_vals = np.zeros(n_periods)
    fti_vals = np.zeros(n_periods)

    channel_len = L - H

    for ip, period in enumerate(range(min_period, max_period + 1)):
        c = all_coefs[ip]
        diff_work = np.empty(channel_len)
        legs = []
        extreme_type = 0
        extreme_value = 0.0
        longest_leg = 0.0
        prior = 0.0

        for iy in range(H, L):
            # Symmetric convolution
            s = c[0] * y[iy]
            for i in range(1, H + 1):
                s += c[i] * (y[iy + i] + y[iy - i])

            if iy == L - 1:
                filtered_vals[ip] = s

            diff_work[iy - H] = abs(y[iy] - s)

            # Swing leg detection
            if iy == H:
                extreme_type = 0
                extreme_value = s
                legs.clear()
                longest_leg = 0.0
            elif extreme_type == 0:
                if s > extreme_value:
                    extreme_type = -1
                elif s < extreme_value:
                    extreme_type = 1
            elif iy == L - 1:
                if extreme_type != 0:
                    leg = abs(extreme_value - s)
                    legs.append(leg)
                    if leg > longest_leg:
                        longest_leg = leg
            else:
                if extreme_type == 1 and s > prior:
                    leg = extreme_value - prior
                    legs.append(leg)
                    if leg > longest_leg:
                        longest_leg = leg
                    extreme_type = -1
                    extreme_value = prior
                elif extreme_type == -1 and s < prior:
                    leg = prior - extreme_value
                    legs.append(leg)
                    if leg > longest_leg:
                        longest_leg = leg
                    extreme_type = 1
                    extreme_value = prior

            prior = s

        # Channel width (beta-fractile)
        diff_sorted = np.sort(diff_work)
        idx = int(beta * (channel_len + 1)) - 1
        idx = max(idx, 0)
        idx = min(idx, channel_len - 1)
        width_vals[ip] = diff_sorted[idx]

        # FTI = mean(non-noise legs) / width
        if legs and longest_leg > 0:
            noise_level = noise_cut * longest_leg
            big_legs = [lg for lg in legs if lg > noise_level]
            if big_legs:
                mean_leg = sum(big_legs) / len(big_legs)
                fti_vals[ip] = mean_leg / (width_vals[ip] + 1e-5)

    # Find best period (local maxima sorted by FTI)
    best_ip = int(np.argmax(fti_vals))
    best_period = min_period + best_ip
    best_filtered = filtered_vals[best_ip]
    if use_log:
        best_filtered = math.exp(best_filtered)
        width_price = 0.5 * (
            math.exp(filtered_vals[best_ip] + width_vals[best_ip])
            - math.exp(filtered_vals[best_ip] - width_vals[best_ip])
        )
    else:
        width_price = width_vals[best_ip]

    return {
        "best_fti": fti_vals[best_ip],
        "best_period": best_period,
        "best_filtered": best_filtered,
        "best_width": width_price,
        "fti_array": fti_vals,
    }


def rolling_fti(
    closes: np.ndarray,
    *,
    lookback: int = 128,
    min_period: int = 5,
    max_period: int = 65,
    half_length: int = 32,
    beta: float = 0.95,
    noise_cut: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling FTI over the price series.

    Returns (fti_best, fti_period, fti_width) arrays of length len(closes).
    """
    n = len(closes)
    fti_best = np.full(n, np.nan)
    fti_period = np.full(n, np.nan)
    fti_width = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        window = closes[i - lookback + 1: i + 1]
        result = fti_single(
            window,
            min_period=min_period,
            max_period=max_period,
            half_length=half_length,
            beta=beta,
            noise_cut=noise_cut,
        )
        fti_best[i] = result["best_fti"]
        fti_period[i] = result["best_period"]
        fti_width[i] = result["best_width"]

    return fti_best, fti_period, fti_width


# ── Mutual Information ──────────────────────────────────────────────────────

def mutual_information(prices: np.ndarray, word_length: int = 2) -> float:
    """Compute mutual information between current direction and history word.

    Measures nonlinear serial dependence in price movements.
    Prices should be in reverse chronological order (newest first).
    """
    nx = len(prices)
    n = nx - word_length - 1
    if n < 4:
        return 0.0

    m = 2 ** word_length
    nb = 2 * m
    bins = np.zeros(nb, dtype=np.int64)
    dep_marg = np.zeros(2, dtype=np.float64)

    for i in range(n):
        k = 1 if prices[i] > prices[i + 1] else 0
        dep_marg[k] += 1
        for j in range(1, word_length + 1):
            k *= 2
            if prices[i + j] > prices[i + j + 1]:
                k += 1
        bins[k] += 1

    dep_marg /= n
    MI = 0.0
    for i in range(m):
        hist_marg = (bins[i] + bins[i + m]) / n
        if hist_marg < 1e-15:
            continue
        p0 = bins[i] / n
        if p0 > 0:
            MI += p0 * math.log(p0 / (hist_marg * max(dep_marg[0], 1e-15)))
        p1 = bins[i + m] / n
        if p1 > 0:
            MI += p1 * math.log(p1 / (hist_marg * max(dep_marg[1], 1e-15)))

    return MI


def rolling_mutual_information(
    closes: np.ndarray,
    window: int = 100,
    word_length: int = 2,
) -> np.ndarray:
    """Rolling mutual information over a price series."""
    n = len(closes)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        # Pass in reverse chronological order (newest first)
        seg = closes[i - window + 1: i + 1][::-1]
        out[i] = mutual_information(seg, word_length=word_length)
    return out


# ── ADX (Average Directional Index) ────────────────────────────────────────

def rolling_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 14,
) -> np.ndarray:
    """Compute ADX using Wilder's smoothing method.

    Returns array of length n with values in [0, 100].
    First 2*lookback bars are NaN.
    """
    n = len(closes)
    out = np.full(n, np.nan)
    if n < 2 * lookback + 1:
        return out

    lb = lookback
    alpha = (lb - 1.0) / lb

    # Phase 1: accumulate initial sums (bars 1..lb)
    DMSplus = 0.0
    DMSminus = 0.0
    ATR = 0.0
    for i in range(1, lb + 1):
        dm_up = highs[i] - highs[i - 1]
        dm_dn = lows[i - 1] - lows[i]
        if dm_up >= dm_dn:
            dm_dn = 0.0
        else:
            dm_up = 0.0
        if dm_up < 0:
            dm_up = 0.0
        if dm_dn < 0:
            dm_dn = 0.0
        DMSplus += dm_up
        DMSminus += dm_dn
        tr = highs[i] - lows[i]
        tr = max(tr, abs(highs[i] - closes[i - 1]))
        tr = max(tr, abs(closes[i - 1] - lows[i]))
        ATR += tr

    # Phase 2: secondary init (bars lb+1..2*lb-1) — accumulate ADX
    ADX_sum = 0.0
    adx_count = 0
    for i in range(lb + 1, 2 * lb):
        dm_up = highs[i] - highs[i - 1]
        dm_dn = lows[i - 1] - lows[i]
        if dm_up >= dm_dn:
            dm_dn = 0.0
        else:
            dm_up = 0.0
        if dm_up < 0:
            dm_up = 0.0
        if dm_dn < 0:
            dm_dn = 0.0
        DMSplus = alpha * DMSplus + dm_up
        DMSminus = alpha * DMSminus + dm_dn
        tr = highs[i] - lows[i]
        tr = max(tr, abs(highs[i] - closes[i - 1]))
        tr = max(tr, abs(closes[i - 1] - lows[i]))
        ATR = alpha * ATR + tr
        DIplus = DMSplus / (ATR + 1e-10)
        DIminus = DMSminus / (ATR + 1e-10)
        DX = abs(DIplus - DIminus) / (DIplus + DIminus + 1e-10)
        ADX_sum += DX
        adx_count += 1

    ADX = ADX_sum / max(adx_count, 1)

    # Phase 3: steady-state
    for i in range(2 * lb, n):
        dm_up = highs[i] - highs[i - 1]
        dm_dn = lows[i - 1] - lows[i]
        if dm_up >= dm_dn:
            dm_dn = 0.0
        else:
            dm_up = 0.0
        if dm_up < 0:
            dm_up = 0.0
        if dm_dn < 0:
            dm_dn = 0.0
        DMSplus = alpha * DMSplus + dm_up
        DMSminus = alpha * DMSminus + dm_dn
        tr = highs[i] - lows[i]
        tr = max(tr, abs(highs[i] - closes[i - 1]))
        tr = max(tr, abs(closes[i - 1] - lows[i]))
        ATR = alpha * ATR + tr
        DIplus = DMSplus / (ATR + 1e-10)
        DIminus = DMSminus / (ATR + 1e-10)
        DX = abs(DIplus - DIminus) / (DIplus + DIminus + 1e-10)
        ADX = alpha * ADX + DX / lb
        out[i] = 100.0 * ADX

    return out


# ── Detrended RSI ───────────────────────────────────────────────────────────

def _wilder_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute Wilder-smoothed RSI."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    diff = np.diff(closes)
    upsum = 0.0
    dnsum = 0.0
    for i in range(period):
        if diff[i] > 0:
            upsum += diff[i]
        else:
            dnsum -= diff[i]
    upsum /= period
    dnsum /= period

    denom = upsum + dnsum
    out[period] = 100.0 * upsum / denom if denom > 1e-15 else 50.0

    for i in range(period, n - 1):
        d = diff[i]
        if d > 0:
            upsum = ((period - 1) * upsum + d) / period
            dnsum = (period - 1) * dnsum / period
        else:
            dnsum = ((period - 1) * dnsum - d) / period
            upsum = (period - 1) * upsum / period
        denom = upsum + dnsum
        out[i + 1] = 100.0 * upsum / denom if denom > 1e-15 else 50.0

    return out


def rolling_detrended_rsi(
    closes: np.ndarray,
    short_period: int = 5,
    long_period: int = 14,
    reg_length: int = 60,
) -> np.ndarray:
    """Detrended RSI: residual of short RSI regressed on long RSI."""
    n = len(closes)
    out = np.full(n, np.nan)

    rsi_short = _wilder_rsi(closes, short_period)
    rsi_long = _wilder_rsi(closes, long_period)

    warmup = long_period + reg_length
    if n < warmup + 1:
        return out

    for i in range(warmup, n):
        x_seg = rsi_long[i - reg_length + 1: i + 1]
        y_seg = rsi_short[i - reg_length + 1: i + 1]

        if np.any(np.isnan(x_seg)) or np.any(np.isnan(y_seg)):
            continue

        xm = x_seg.mean()
        ym = y_seg.mean()
        xd = x_seg - xm
        yd = y_seg - ym
        xss = np.dot(xd, xd)
        coef = np.dot(xd, yd) / (xss + 1e-30)

        out[i] = (rsi_short[i] - ym) - coef * (rsi_long[i] - xm)

    return out


# ── Price Variance Ratio ────────────────────────────────────────────────────

def rolling_price_variance_ratio(
    closes: np.ndarray,
    short_length: int = 10,
    mult: int = 4,
) -> np.ndarray:
    """Price variance ratio: Var_short(log_returns) / Var_long(log_returns).

    Output centred around 0 via F-CDF transform.
    Positive = increasing volatility / trending.
    """
    from scipy.stats import f as f_dist

    n = len(closes)
    long_length = short_length * mult
    out = np.full(n, np.nan)
    if n < long_length + 1:
        return out

    log_ret = np.diff(np.log(np.maximum(closes, 1e-12)))

    for i in range(long_length, n - 1):
        short_seg = log_ret[i - short_length + 1: i + 1]
        long_seg = log_ret[i - long_length + 1: i + 1]
        var_s = np.var(short_seg)
        var_l = np.var(long_seg)
        if var_l < 1e-20:
            continue
        vr = var_s / var_l
        # F-CDF: dfn=4, dfd=4*mult (change variance ratio)
        out[i + 1] = 100.0 * f_dist.cdf(vr, 4, 4 * mult) - 50.0

    return out


# ── Price Intensity ─────────────────────────────────────────────────────────

def rolling_price_intensity(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    smooth: int = 10,
) -> np.ndarray:
    """Price intensity: (close-open)/true_range, EMA smoothed.

    Captures directional conviction relative to the bar's range.
    Output in approximately [-1, 1] before smoothing.
    """
    n = len(closes)
    raw = np.full(n, np.nan)

    # Bar 0
    denom = max(highs[0] - lows[0], 1e-10)
    raw[0] = (closes[0] - opens[0]) / denom

    for i in range(1, n):
        denom = highs[i] - lows[i]
        denom = max(denom, abs(highs[i] - closes[i - 1]))
        denom = max(denom, abs(closes[i - 1] - lows[i]))
        denom = max(denom, 1e-10)
        raw[i] = (closes[i] - opens[i]) / denom

    # EMA smoothing
    if smooth > 1:
        alpha = 2.0 / (smooth + 1.0)
        smoothed = raw[0]
        for i in range(1, n):
            if np.isnan(raw[i]):
                continue
            smoothed = alpha * raw[i] + (1 - alpha) * smoothed
            raw[i] = smoothed

    return raw
