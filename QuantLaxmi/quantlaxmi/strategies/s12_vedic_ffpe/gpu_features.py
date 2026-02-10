"""GPU-accelerated batch feature computation for S12 Vedic FFPE.

Uses PyTorch on T4 GPU to vectorise the hot-path computations:
  - MSD-based α estimation (all rolling windows simultaneously)
  - Mock theta q-series evaluation (all bars at once)
  - Ramanujan volatility distortion (all bars at once)
  - Fractional differencing (conv1d)

CPU fallbacks for operations that don't benefit from GPU:
  - Angular coherence (3-element vectors)
  - Aryabhata phase (sequential quadrant disambiguation)
  - Waiting-time α estimator (variable-length exceedances)

Architecture mirrors S5's GPU pattern: torch on main process only.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import logging

from quantlaxmi.features.fractional import estimate_alpha_waiting
from quantlaxmi.features.mock_theta import ramanujan_volatility_distortion
from quantlaxmi.features.vedic_angular import (
    angular_coherence,
    update_regime_centroids,
    aryabhata_phase,
)
from quantlaxmi.features.ramanujan import dominant_periods, ramanujan_periodogram
from quantlaxmi.features.information import rolling_entropy
from quantlaxmi.features.rmt import rolling_rmt_features
from quantlaxmi.features.masters import (
    rolling_fti, rolling_mutual_information, rolling_adx,
    rolling_detrended_rsi, rolling_price_variance_ratio,
    rolling_price_intensity,
)

logger = logging.getLogger(__name__)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE = torch.float64  # need double precision for mock theta convergence


# ---------------------------------------------------------------------------
# GPU batch kernels
# ---------------------------------------------------------------------------

def gpu_batch_alpha_msd(
    log_ret: np.ndarray,
    window: int = 60,
    max_lag: int = 30,
) -> np.ndarray:
    """Batch MSD-based α estimation for all rolling windows on GPU.

    Instead of looping bar-by-bar, unfolds the cumulative-return series
    into overlapping windows and computes all MSDs simultaneously.

    Returns array of length len(log_ret) with α values (NaN for warmup).
    """
    n = len(log_ret)
    max_lag = min(max_lag, window // 3)
    if max_lag < 4 or n < window + 1:
        result = np.full(n, np.nan)
        if n >= window + 1:
            result[window:] = 1.0
        return result

    cumret = torch.tensor(
        np.cumsum(log_ret), dtype=_DTYPE, device=_DEVICE,
    )

    # Sliding windows: (n - window + 1, window)
    windows = cumret.unfold(0, window, 1)
    n_windows = windows.shape[0]

    lags = torch.arange(2, max_lag + 1, dtype=_DTYPE, device=_DEVICE)
    log_lags = torch.log(lags)
    n_lags = len(lags)

    # MSD for all lags × all windows in one pass
    log_msds = torch.empty(n_windows, n_lags, dtype=_DTYPE, device=_DEVICE)

    for j, tau in enumerate(range(2, max_lag + 1)):
        displ = windows[:, tau:] - windows[:, :-tau]  # (n_windows, window-tau)
        msd = (displ ** 2).mean(dim=1)                # (n_windows,)
        log_msds[:, j] = torch.log(msd.clamp(min=1e-30))

    # Vectorised OLS: slope = cov(x,y) / var(x)
    x = log_lags.unsqueeze(0).expand(n_windows, -1)
    y = log_msds

    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    cov_xy = ((x - x_mean) * (y - y_mean)).sum(dim=1)
    var_x = ((x - x_mean) ** 2).sum(dim=1).clamp(min=1e-30)

    slope = cov_xy / var_x
    hurst = torch.clamp(slope / 2.0, 0.01, 1.5)
    alpha = (2.0 * hurst).cpu().numpy()

    # windows[j] covers cumret[j:j+window] → bar index j+window-1
    # Original loop starts at bar `window`, so we skip windows[0]
    result = np.full(n, np.nan)
    result[window: window + n_windows - 1] = alpha[1:]

    return result


def gpu_batch_mock_theta(
    mean_abs_rets: np.ndarray,
    sigmas: np.ndarray,
    n_terms: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch mock theta f, φ, and ratio for all bars on GPU.

    Vectorises the n_terms series loop across all bars simultaneously.
    Returns (f_arr, phi_arr, ratio_arr).
    """
    # q = exp(-π / (1 + |r|/σ))
    q_np = np.exp(-math.pi / (1.0 + mean_abs_rets / np.maximum(sigmas, 1e-10)))
    q = torch.tensor(q_np, dtype=_DTYPE, device=_DEVICE)

    total_f = torch.ones_like(q)
    prod_f = torch.ones_like(q)
    total_phi = torch.ones_like(q)
    prod_phi = torch.ones_like(q)

    for nn in range(1, n_terms + 1):
        qk = q ** nn
        prod_f *= (1.0 + qk) ** 2
        prod_phi *= (1.0 + qk)

        qn2 = q ** (nn * nn)
        total_f += qn2 / prod_f.clamp(min=1e-300)
        total_phi += qn2 / prod_phi.clamp(min=1e-300)

        if qn2.abs().max().item() < 1e-300:
            break

    f_arr = total_f.cpu().numpy()
    phi_arr = total_phi.cpu().numpy()
    ratio_arr = f_arr / np.maximum(phi_arr, 1e-30)

    return f_arr, phi_arr, ratio_arr


def gpu_batch_vol_distortion(
    ann_vols: np.ndarray,
    alpha: float = 0.2,
    depth: int = 20,
) -> np.ndarray:
    """Batch Ramanujan continued-fraction volatility distortion on GPU."""
    vol = torch.tensor(ann_vols, dtype=_DTYPE, device=_DEVICE)
    cf = torch.ones_like(vol)

    for k in range(depth, 0, -1):
        cf = 1.0 + k * vol / cf.clamp(min=1e-30)

    result = torch.exp(-alpha * vol) / cf.clamp(min=1e-30)
    return result.cpu().numpy()


def gpu_fractional_diff(
    series: np.ndarray,
    d: float,
    max_window: int | None = None,
    threshold: float = 1e-5,
) -> np.ndarray:
    """Fractional differencing via GPU conv1d.

    Replaces the Python dot-product loop with a single conv1d call.
    """
    n = len(series)
    if max_window is None:
        max_window = n

    # Compute weights (CPU — small, one-time)
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

    width = len(weights)
    # Reversed weights for the convolution
    weights_rev = np.array(weights[::-1], dtype=np.float64)

    # conv1d: output[i] = Σ_k kernel[k] * input[i+k]
    # This gives result[i + width - 1] = Σ weights_rev[k] * series[i+k]
    series_t = torch.tensor(
        series, dtype=_DTYPE, device=_DEVICE,
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, n)

    kernel = torch.tensor(
        weights_rev, dtype=_DTYPE, device=_DEVICE,
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, width)

    output = F.conv1d(series_t, kernel)  # (1, 1, n - width + 1)
    output_np = output.squeeze().cpu().numpy()

    result = np.full(n, np.nan)
    result[width - 1:] = output_np
    return result


# ---------------------------------------------------------------------------
# Orchestrator: full feature array with GPU + CPU
# ---------------------------------------------------------------------------

def compute_features_array_gpu(
    closes: np.ndarray,
    *,
    alpha_window: int = 60,
    frac_d: float = 0.226,
    mock_theta_n_terms: int = 50,
) -> dict[str, np.ndarray]:
    """GPU-accelerated version of compute_features_array.

    Heavy ops (MSD α, mock theta, vol distortion, frac diff) run on GPU.
    Light ops (coherence, phase, waiting-time α) run on CPU.

    Returns same dict[str, ndarray] as the CPU version.
    """
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  GPU features ({gpu_name}):")
    t0 = time.time()

    n = len(closes)
    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])

    out: dict[str, np.ndarray] = {}

    # --- Phase 1: GPU batch operations ---

    # 1a. Batch MSD α
    t1 = time.time()
    alpha_msd = gpu_batch_alpha_msd(log_ret, window=alpha_window,
                                     max_lag=min(30, alpha_window // 2))
    print(f"    α MSD:     {time.time()-t1:.2f}s")

    # 1b. Rolling stats for mock theta (CPU vectorised — cheap)
    t1 = time.time()
    sigmas = np.full(n, np.nan)
    mean_abs = np.full(n, np.nan)
    rolling_mom = np.full(n, np.nan)
    rolling_mean = np.full(n, np.nan)
    rolling_vol = np.full(n, np.nan)

    for i in range(alpha_window, n):
        rets = log_ret[i - alpha_window + 1: i + 1]
        s = np.std(rets, ddof=1)
        sigmas[i] = max(s, 1e-8)
        mean_abs[i] = np.mean(np.abs(rets))
        rolling_mom[i] = np.sum(rets)
        rolling_mean[i] = np.mean(rets)
        rolling_vol[i] = s
    print(f"    Stats:     {time.time()-t1:.2f}s")

    # 1c. Batch mock theta on GPU
    t1 = time.time()
    valid_mask = ~np.isnan(sigmas)
    valid_idx = np.where(valid_mask)[0]

    f_arr = np.full(n, np.nan)
    phi_arr = np.full(n, np.nan)
    ratio_arr = np.full(n, np.nan)
    div_arr = np.full(n, np.nan)
    vd_arr = np.full(n, np.nan)

    if len(valid_idx) > 0:
        f_batch, phi_batch, ratio_batch = gpu_batch_mock_theta(
            mean_abs[valid_idx], sigmas[valid_idx], mock_theta_n_terms,
        )
        f_arr[valid_idx] = f_batch
        phi_arr[valid_idx] = phi_batch
        ratio_arr[valid_idx] = ratio_batch

        # Divergence: |Δratio|
        div_arr[valid_idx[1:]] = np.abs(np.diff(ratio_batch))
        div_arr[valid_idx[0]] = 0.0

        # Batch vol distortion on GPU
        ann_vols = sigmas[valid_idx] * math.sqrt(252)
        vd_arr[valid_idx] = gpu_batch_vol_distortion(ann_vols)

    print(f"    MockΘ+Vol: {time.time()-t1:.2f}s")

    # 1d. Fractional differencing on GPU
    t1 = time.time()
    d_series = gpu_fractional_diff(closes, frac_d, max_window=alpha_window)
    print(f"    FracDiff:  {time.time()-t1:.2f}s")

    # --- Phase 2: CPU operations (cheap) ---

    # 2a. Waiting-time α (CPU per-bar — variable-length exceedances)
    t1 = time.time()
    alpha_wt = np.full(n, np.nan)
    for i in range(alpha_window, n):
        rets = log_ret[i - alpha_window + 1: i + 1]
        alpha_wt[i] = estimate_alpha_waiting(rets)
    print(f"    α Wait:    {time.time()-t1:.2f}s")

    # 2b. Consensus α and regime
    alpha_consensus = np.full(n, np.nan)
    valid = ~(np.isnan(alpha_msd) | np.isnan(alpha_wt))
    alpha_consensus[valid] = 0.6 * alpha_msd[valid] + 0.4 * alpha_wt[valid]

    regime = np.full(n, np.nan)
    regime[alpha_consensus < 0.85] = 0   # subdiffusive
    regime[alpha_consensus > 1.15] = 2   # superdiffusive
    regime[(alpha_consensus >= 0.85) & (alpha_consensus <= 1.15)] = 1  # normal

    # 2c. Angular coherence (CPU — tiny vectors)
    t1 = time.time()
    coherence_arr = np.full(n, np.nan)
    regime_centroids: dict[str, np.ndarray] = {}

    for i in range(alpha_window, n):
        if np.isnan(rolling_vol[i]):
            continue
        feat_vec = np.array([rolling_vol[i], rolling_mom[i], rolling_mean[i]])

        if regime_centroids:
            _, coh, _ = angular_coherence(feat_vec, regime_centroids, order=4)
            coherence_arr[i] = coh
        else:
            coherence_arr[i] = 0.5

        # Classify regime label for centroid update
        autocorr = 0.0
        rets = log_ret[i - alpha_window + 1: i + 1]
        if len(rets) > 2:
            c = np.corrcoef(rets[:-1], rets[1:])
            if c.shape == (2, 2) and np.isfinite(c[0, 1]):
                autocorr = c[0, 1]
        if autocorr < -0.1:
            label = "subdiffusive"
        elif autocorr > 0.1:
            label = "superdiffusive"
        else:
            label = "normal"
        regime_centroids = update_regime_centroids(
            regime_centroids, feat_vec, label, decay=0.99,
        )
    print(f"    Coherence: {time.time()-t1:.2f}s")

    # 2d. Aryabhata phase (CPU — Ramanujan periodogram + phase)
    t1 = time.time()
    phase_arr = np.full(n, np.nan)

    for i in range(alpha_window * 2, n):
        wp = closes[i - alpha_window: i]
        wr = log_ret[i - alpha_window + 1: i + 1]
        periods = dominant_periods(
            wr - wr.mean(),
            max_period=min(32, alpha_window // 2),
            top_k=1,
        )
        period = periods[0] if periods else alpha_window // 4
        ph, _ = aryabhata_phase(wp, period)
        if not np.isnan(ph[-1]):
            phase_arr[i] = ph[-1]
    print(f"    Phase:     {time.time()-t1:.2f}s")

    total = time.time() - t0
    print(f"    TOTAL:     {total:.2f}s")

    # --- Assemble output ---
    out["alpha"] = alpha_consensus
    out["alpha_msd"] = alpha_msd
    out["alpha_waiting"] = alpha_wt
    out["hurst"] = alpha_consensus / 2.0
    out["mock_theta_f"] = f_arr
    out["mock_theta_phi"] = phi_arr
    out["mock_theta_ratio"] = ratio_arr
    out["mock_theta_div"] = div_arr
    out["vol_distortion"] = vd_arr
    out["coherence"] = coherence_arr
    out["phase"] = phase_arr
    out["frac_d_series"] = d_series
    out["regime"] = regime

    return out


# ---------------------------------------------------------------------------
# Fast backtest using pre-computed features
# ---------------------------------------------------------------------------

def backtest_from_features(
    features: dict[str, np.ndarray],
    closes: np.ndarray,
    dates: list,
    *,
    alpha_window: int = 60,
    alpha_lo: float = 0.85,
    alpha_hi: float = 1.15,
    min_conviction: float = 0.15,
    phase_lo: float = 0.0,
    phase_hi: float = 0.5,
    cost_bps: float = 5.0,
    hold_days: int = 5,
    mt_div_threshold: float = 0.5,
    coherence_weight: float = 1.0,
    phase_reject_factor: float = 0.3,
    frac_d_direction: bool = False,
    frac_d_conviction: float = 0.0,
) -> dict:
    """Run EOD backtest using pre-computed feature arrays.

    Avoids re-computing features bar-by-bar — just applies signal logic
    and portfolio simulation over the pre-computed arrays.
    """
    n = len(closes)
    warmup = alpha_window + 10
    if n < warmup + 20:
        return {"trades": 0}

    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])

    cost_frac = cost_bps / 10_000
    next_day_ret = np.full(n, np.nan)
    next_day_ret[:-1] = np.diff(closes) / closes[:-1]

    # --- Generate signals from pre-computed features ---
    prev_regime = 1  # normal
    bars_in_regime = 0

    position = 0   # 0=flat, 1=long, -1=short
    entry_day = 0
    daily_pnl: list[float] = []
    trades: list[dict] = []
    regime_counts = {"subdiffusive": 0, "normal": 0, "superdiffusive": 0}

    for i in range(warmup, n - 1):
        alpha = features["alpha"][i] if i < n else np.nan
        if np.isnan(alpha):
            daily_pnl.append(0.0)
            continue

        # Regime with hysteresis
        raw_regime = 0 if alpha < alpha_lo else (2 if alpha > alpha_hi else 1)
        if raw_regime != prev_regime and bars_in_regime < 3:
            regime = prev_regime
            bars_in_regime += 1
        elif raw_regime != prev_regime:
            regime = raw_regime
            prev_regime = raw_regime
            bars_in_regime = 1
        else:
            regime = raw_regime
            bars_in_regime += 1

        regime_name = ["subdiffusive", "normal", "superdiffusive"][regime]
        regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

        # Feature values with NaN guards
        coherence = features["coherence"][i] if not np.isnan(features["coherence"][i]) else 0.5
        mt_div = features["mock_theta_div"][i] if not np.isnan(features["mock_theta_div"][i]) else 0.0
        phase = features["phase"][i] if not np.isnan(features["phase"][i]) else 0.5
        frac_d_val = features["frac_d_series"][i] if not np.isnan(features["frac_d_series"][i]) else 0.0
        mom = np.sum(log_ret[max(0, i - alpha_window + 1): i + 1])

        # Phase gate
        phase_factor = 1.0 if (phase_lo <= phase <= phase_hi) else phase_reject_factor

        # Coherence with configurable weight
        coh = 1.0 - coherence_weight + coherence_weight * coherence

        # Direction source: raw momentum or fractionally differenced series
        if frac_d_direction:
            dir_signal = frac_d_val
        else:
            dir_signal = mom

        # Signal logic (mirrors compute_daily_signal)
        direction = 0
        conviction = 0.0

        if regime == 0:  # subdiffusive → contrarian
            base_conv = coh * (1 - mt_div) * phase_factor
            if frac_d_conviction > 0:
                base_conv *= (1.0 + frac_d_conviction * abs(frac_d_val))
            conviction = float(np.clip(base_conv, 0, 1))
            direction = -1 if dir_signal > 0 else (1 if dir_signal < 0 else 0)
        elif regime == 2:  # superdiffusive → momentum
            base_conv = coh * (alpha - 1.0) * phase_factor
            if frac_d_conviction > 0:
                base_conv *= (1.0 + frac_d_conviction * abs(frac_d_val))
            conviction = float(np.clip(base_conv, 0, 1))
            direction = 1 if dir_signal > 0 else (-1 if dir_signal < 0 else 0)
        else:  # normal → flat unless spike
            if mt_div > mt_div_threshold:
                conviction = float(np.clip(0.5 * mt_div * coh * phase_factor, 0, 1))
                direction = 1 if frac_d_val > 0 else (-1 if frac_d_val < 0 else 0)

        if conviction < min_conviction or direction == 0:
            direction = 0
            conviction = 0.0

        # --- Portfolio simulation ---
        if position != 0:
            days_held = i - entry_day
            should_exit = days_held >= hold_days
            should_exit |= (direction != 0 and direction != position)

            if should_exit:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    trade_ret = ret * position - cost_frac
                    daily_pnl.append(trade_ret)
                    trades.append({
                        "date": dates[i] if i < len(dates) else i,
                        "direction": "long" if position == 1 else "short",
                        "ret": trade_ret,
                        "regime": regime_name,
                        "alpha": alpha,
                    })
                else:
                    daily_pnl.append(0.0)
                position = 0
            else:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    daily_pnl.append(ret * position)
                else:
                    daily_pnl.append(0.0)
                continue

        if position == 0 and direction != 0:
            position = direction
            entry_day = i
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    if not daily_pnl:
        return {"trades": 0}

    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])

    if len(pnl_arr) > 1:
        std = np.std(pnl_arr, ddof=1)
        sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    wins = sum(1 for t in trades if t["ret"] > 0)

    return {
        "trades": len(trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / max(len(trades), 1),
        "regime_counts": regime_counts,
        "daily_returns": daily_pnl,
        "trade_list": trades,
    }


# ---------------------------------------------------------------------------
# Gate-free linear signal backtest (for Optuna)
# ---------------------------------------------------------------------------

def _zscore(arr: np.ndarray) -> np.ndarray:
    """Rolling-free z-score of non-NaN values. NaNs stay NaN."""
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return out
    m = np.nanmean(arr)
    s = np.nanstd(arr, ddof=1)
    if s > 1e-12:
        out[valid] = (arr[valid] - m) / s
    return out


def backtest_linear_signal(
    features: dict[str, np.ndarray],
    closes: np.ndarray,
    dates: list,
    *,
    w_alpha: float = 0.0,
    w_coherence: float = 0.0,
    w_phase: float = 0.0,
    w_mock_theta: float = 0.0,
    w_frac_d: float = 0.0,
    w_momentum: float = 0.0,
    entry_threshold: float = 0.3,
    hold_days: int = 5,
    cost_bps: float = 5.0,
    alpha_window: int = 60,
) -> dict:
    """Gate-free backtest: weighted linear combination of all features.

    signal = Σ wᵢ · zᵢ   (z-scored features for fair comparison)
    direction = sign(signal), conviction = |signal|
    Enter when conviction > entry_threshold, exit on hold_days or flip.

    No regime gates, no if/else, no mock theta threshold.
    Lets Optuna discover which features have causal power.
    """
    n = len(closes)
    warmup = alpha_window + 10
    if n < warmup + 20:
        return {"trades": 0, "sharpe": 0.0}

    # Z-score all features once
    z_alpha = _zscore(features["alpha"] - 1.0)  # center around 1.0
    z_coh = _zscore(features["coherence"])
    z_phase = _zscore(features["phase"])
    z_mt = _zscore(features["mock_theta_div"])
    z_fd = _zscore(features["frac_d_series"])

    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])
    z_mom = np.full(n, np.nan)
    for i in range(alpha_window, n):
        z_mom[i] = np.sum(log_ret[i - alpha_window + 1: i + 1])
    z_mom = _zscore(z_mom)

    cost_frac = cost_bps / 10_000
    next_day_ret = np.full(n, np.nan)
    next_day_ret[:-1] = np.diff(closes) / closes[:-1]

    position = 0
    entry_day = 0
    daily_pnl: list[float] = []
    trades: list[dict] = []

    for i in range(warmup, n - 1):
        # Composite signal — pure linear combination
        vals = [
            (w_alpha, z_alpha[i]),
            (w_coherence, z_coh[i]),
            (w_phase, z_phase[i]),
            (w_mock_theta, z_mt[i]),
            (w_frac_d, z_fd[i]),
            (w_momentum, z_mom[i]),
        ]

        signal = 0.0
        any_nan = False
        for w, v in vals:
            if np.isnan(v):
                any_nan = True
                break
            signal += w * v

        if any_nan:
            daily_pnl.append(0.0)
            continue

        direction = 1 if signal > 0 else (-1 if signal < 0 else 0)
        conviction = abs(signal)

        # Position management
        if position != 0:
            days_held = i - entry_day
            should_exit = days_held >= hold_days
            should_exit |= (direction != 0 and direction != position
                            and conviction >= entry_threshold)

            if should_exit:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    trade_ret = ret * position - cost_frac
                    daily_pnl.append(trade_ret)
                    trades.append({"ret": trade_ret})
                else:
                    daily_pnl.append(0.0)
                position = 0
            else:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    daily_pnl.append(ret * position)
                else:
                    daily_pnl.append(0.0)
                continue

        if position == 0 and conviction >= entry_threshold and direction != 0:
            position = direction
            entry_day = i
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    if not daily_pnl:
        return {"trades": 0, "sharpe": 0.0}

    pnl_arr = np.array(daily_pnl)
    if len(pnl_arr) > 1:
        std = np.std(pnl_arr, ddof=1)
        sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    wins = sum(1 for t in trades if t["ret"] > 0)

    return {
        "trades": len(trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / max(len(trades), 1),
    }


# ---------------------------------------------------------------------------
# Recalibrated gated backtest (informed by Optuna feature importance)
# ---------------------------------------------------------------------------
#
# Optuna findings (NIFTY, 500 trials, Sharpe 2.77):
#   coherence     52.8%  ← #1 signal: high coherence → go long
#   phase         19.3%  ← #2 signal: higher phase → go long
#   momentum      10.5%  ← #3 signal: CONTRARIAN (w = -1.9)
#   frac_d         5.4%  ← mild contrarian
#   mock_theta     3.1%  ← mild positive
#   alpha          2.7%  ← barely matters as standalone
#
# Architecture changes vs original:
#   1. All regimes can trade (soft modulation, not hard gate)
#   2. Direction is contrarian (anti-momentum) — dominant finding
#   3. Coherence is the PRIMARY conviction driver
#   4. Phase is continuous factor, not binary gate
#   5. α regime softly adjusts conviction (sub boosts, super dampens)

def backtest_recalibrated(
    features: dict[str, np.ndarray],
    closes: np.ndarray,
    dates: list,
    *,
    alpha_window: int = 60,
    alpha_lo: float = 0.85,
    alpha_hi: float = 1.15,
    # Optuna-informed weights (normalised from fANOVA importance)
    w_coherence: float = 1.4,
    w_phase: float = 1.3,
    w_mock_theta: float = 0.7,
    w_frac_d: float = 0.25,
    # Regime soft factors (how much α modulates conviction)
    regime_sub_boost: float = 1.3,     # subdiffusive: boost contrarian
    regime_norm_factor: float = 1.0,   # normal: baseline
    regime_super_factor: float = 0.7,  # superdiffusive: dampen (contrarian risky in trends)
    # Entry/exit
    min_conviction: float = 0.20,
    hold_days: int = 16,
    cost_bps: float = 5.0,
) -> dict:
    """Recalibrated gated backtest using Optuna-discovered feature weights.

    Key differences from original:
      - ALL regimes can trade (α is a soft modulator)
      - Direction is CONTRARIAN (anti-momentum)
      - Conviction = coherence (primary) + phase + mock_theta + frac_d
      - Regime adjusts conviction softly (sub boosts, super dampens)
    """
    n = len(closes)
    warmup = alpha_window + 10
    if n < warmup + 20:
        return {"trades": 0}

    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])

    cost_frac = cost_bps / 10_000
    next_day_ret = np.full(n, np.nan)
    next_day_ret[:-1] = np.diff(closes) / closes[:-1]

    # Z-score features for fair weighting
    z_coh = _zscore(features["coherence"])
    z_phase = _zscore(features["phase"])
    z_mt = _zscore(features["mock_theta_div"])
    z_fd = _zscore(features["frac_d_series"])

    prev_regime = 1
    bars_in_regime = 0
    position = 0
    entry_day = 0
    daily_pnl: list[float] = []
    trades: list[dict] = []
    regime_counts = {"subdiffusive": 0, "normal": 0, "superdiffusive": 0}

    for i in range(warmup, n - 1):
        alpha = features["alpha"][i]
        if np.isnan(alpha):
            daily_pnl.append(0.0)
            continue

        # Regime with hysteresis (kept for interpretability)
        raw_regime = 0 if alpha < alpha_lo else (2 if alpha > alpha_hi else 1)
        if raw_regime != prev_regime and bars_in_regime < 3:
            regime = prev_regime
            bars_in_regime += 1
        elif raw_regime != prev_regime:
            regime = raw_regime
            prev_regime = raw_regime
            bars_in_regime = 1
        else:
            regime = raw_regime
            bars_in_regime += 1

        regime_name = ["subdiffusive", "normal", "superdiffusive"][regime]
        regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

        # Feature values (NaN → skip)
        coh_z = z_coh[i] if not np.isnan(z_coh[i]) else 0.0
        ph_z = z_phase[i] if not np.isnan(z_phase[i]) else 0.0
        mt_z = z_mt[i] if not np.isnan(z_mt[i]) else 0.0
        fd_z = z_fd[i] if not np.isnan(z_fd[i]) else 0.0

        # Momentum (contrarian)
        mom = np.sum(log_ret[max(0, i - alpha_window + 1): i + 1])

        # --- Direction: CONTRARIAN (Optuna's #1 finding) ---
        direction = -1 if mom > 0 else (1 if mom < 0 else 0)

        # --- Conviction: weighted combination of ALL features ---
        raw_conviction = (
            w_coherence * coh_z
            + w_phase * ph_z
            + w_mock_theta * mt_z
            + w_frac_d * (-fd_z)  # frac_d is contrarian (negative weight)
        )
        # Take abs: conviction is always positive, direction is separate
        conviction = abs(raw_conviction)

        # --- Soft regime modulation ---
        if regime == 0:  # subdiffusive → contrarian is right → boost
            conviction *= regime_sub_boost
        elif regime == 2:  # superdiffusive → contrarian is risky → dampen
            conviction *= regime_super_factor
        else:  # normal
            conviction *= regime_norm_factor

        # Normalise to [0, 1] range
        conviction = float(np.clip(conviction / 5.0, 0, 1))

        if conviction < min_conviction or direction == 0:
            direction = 0
            conviction = 0.0

        # --- Portfolio simulation ---
        if position != 0:
            days_held = i - entry_day
            should_exit = days_held >= hold_days
            should_exit |= (direction != 0 and direction != position)

            if should_exit:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    trade_ret = ret * position - cost_frac
                    daily_pnl.append(trade_ret)
                    trades.append({
                        "date": dates[i] if i < len(dates) else i,
                        "direction": "long" if position == 1 else "short",
                        "ret": trade_ret,
                        "regime": regime_name,
                        "alpha": alpha,
                    })
                else:
                    daily_pnl.append(0.0)
                position = 0
            else:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    daily_pnl.append(ret * position)
                else:
                    daily_pnl.append(0.0)
                continue

        if position == 0 and direction != 0 and conviction >= min_conviction:
            position = direction
            entry_day = i
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    if not daily_pnl:
        return {"trades": 0}

    pnl_arr = np.array(daily_pnl)
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])

    if len(pnl_arr) > 1:
        std = np.std(pnl_arr, ddof=1)
        sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    wins = sum(1 for t in trades if t["ret"] > 0)

    return {
        "trades": len(trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / max(len(trades), 1),
        "regime_counts": regime_counts,
        "daily_returns": daily_pnl,
        "trade_list": trades,
    }


# ---------------------------------------------------------------------------
# Expanded feature computation (original 6 + Tier 1 + RMT + IV + VPIN)
# ---------------------------------------------------------------------------

def _rolling_yang_zhang(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Yang-Zhang realized volatility estimator (annualized)."""
    n = len(closes)
    out = np.full(n, np.nan)

    log_hl_sq = np.log(highs / np.maximum(lows, 1e-8)) ** 2

    for i in range(window + 1, n):
        w = slice(i - window, i)
        oc = np.log(opens[w] / np.maximum(closes[i - window - 1:i - 1], 1e-8))
        co = np.log(closes[w] / np.maximum(opens[w], 1e-8))

        sigma_oc = np.var(oc, ddof=1)
        sigma_co = np.var(co, ddof=1)
        sigma_rs = np.mean(log_hl_sq[w]) / (4.0 * np.log(2.0))

        k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
        yz_var = sigma_oc + k * sigma_co + (1.0 - k) * sigma_rs
        out[i] = np.sqrt(max(yz_var, 0.0) * 252.0)

    return out


def _rolling_atr_normalised(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """Normalised ATR (ATR / close)."""
    n = len(closes)
    out = np.full(n, np.nan)

    # True range
    tr = np.full(n, np.nan)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Rolling mean of TR, normalised by close
    for i in range(window, n):
        atr = np.mean(tr[i - window + 1:i + 1])
        out[i] = atr / max(closes[i], 1e-8)

    return out


def _rolling_vpin(
    closes: np.ndarray,
    volumes: np.ndarray,
    window: int = 50,
) -> np.ndarray:
    """Simplified VPIN from daily bars using Bulk Volume Classification."""
    n = len(closes)
    out = np.full(n, np.nan)

    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])

    for i in range(window, n):
        w = slice(i - window + 1, i + 1)
        rets = log_ret[w]
        vols = volumes[w]

        if np.sum(vols) < 1e-8:
            continue

        # BVC: fraction of volume classified as buy
        sigma = np.std(rets, ddof=1)
        if sigma < 1e-12:
            continue

        from scipy.stats import norm
        buy_frac = norm.cdf(rets / sigma)
        buy_vol = buy_frac * vols
        sell_vol = (1.0 - buy_frac) * vols

        total = np.sum(vols)
        imbalance = np.abs(buy_vol - sell_vol)
        out[i] = np.sum(imbalance) / total

    return out


def _rolling_period_energy(
    closes: np.ndarray,
    window: int = 64,
    max_period: int = 32,
) -> np.ndarray:
    """Ramanujan periodogram energy of dominant period."""
    n = len(closes)
    out = np.full(n, np.nan)

    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])

    for i in range(window, n):
        rets = log_ret[i - window + 1:i + 1]
        centered = rets - rets.mean()
        periodogram = ramanujan_periodogram(centered, max_period=max_period)
        total_e = periodogram.sum()
        if total_e > 1e-12:
            out[i] = periodogram.max() / total_e  # normalised peak energy
        else:
            out[i] = 0.0

    return out


def _date_to_str(d) -> str:
    """Normalize any date type to 'YYYY-MM-DD' string."""
    if isinstance(d, str):
        return d[:10]
    if hasattr(d, "isoformat"):
        return d.isoformat()[:10]
    return str(d)[:10]


def _compute_sanos_features(
    store,
    symbol: str,
    dates: list,
    closes: np.ndarray,
    *,
    sanos_cache: dict | None = None,
) -> dict[str, np.ndarray]:
    """Compute SANOS risk-neutral density features from option chain data.

    For each date with available option data, calibrates SANOS surface
    and extracts:
      sanos_skew      — risk-neutral skewness (negative = crash fear)
      sanos_left_tail — left tail probability mass P(K < μ - σ)
      sanos_entropy   — Shannon entropy of density (market uncertainty)
      sanos_kl        — KL divergence from previous day (regime shift speed)

    Data sources (priority):
      1. nse_fo_bhavcopy (daily settlement — has TckrSymb, StrkPric, etc.)
      2. nfo_1min (construct EOD snapshot from last 1-min bar per option)

    Returns dict of 4 arrays, each length len(dates).
    """
    import pandas as pd

    n = len(dates)
    skew_arr = np.full(n, np.nan)
    left_tail_arr = np.full(n, np.nan)
    entropy_arr = np.full(n, np.nan)
    kl_arr = np.full(n, np.nan)

    if store is None:
        return {
            "sanos_skew": skew_arr,
            "sanos_left_tail": left_tail_arr,
            "sanos_entropy": entropy_arr,
            "sanos_kl": kl_arr,
        }

    # Use pre-computed cache if provided (from research.py)
    if sanos_cache is not None:
        date_strs = [_date_to_str(d) for d in dates]
        for i, ds in enumerate(date_strs):
            if ds in sanos_cache:
                snap = sanos_cache[ds]
                skew_arr[i] = snap["skew"]
                left_tail_arr[i] = snap["left_tail"]
                entropy_arr[i] = snap["entropy"]
                kl_arr[i] = snap["kl"]
        return {
            "sanos_skew": skew_arr,
            "sanos_left_tail": left_tail_arr,
            "sanos_entropy": entropy_arr,
            "sanos_kl": kl_arr,
        }

    # Inline computation (slower — used when no cache provided)
    try:
        from quantlaxmi.core.pricing.sanos import fit_sanos, prepare_nifty_chain
        from quantlaxmi.core.pricing.risk_neutral import (
            extract_density, compute_moments, shannon_entropy,
            kl_divergence, tail_weights, physical_skewness,
        )
    except ImportError:
        logger.debug("SANOS imports failed — returning NaN arrays")
        return {
            "sanos_skew": skew_arr,
            "sanos_left_tail": left_tail_arr,
            "sanos_entropy": entropy_arr,
            "sanos_kl": kl_arr,
        }

    instrument = "IDO" if symbol in ("NIFTY", "BANKNIFTY",
                                      "FINNIFTY", "MIDCPNIFTY") else "STO"
    prev_density = None
    prev_K = None
    date_strs = [_date_to_str(d) for d in dates]

    for i, ds in enumerate(date_strs):
        try:
            fno_df = store.sql(
                "SELECT * FROM nse_fo_bhavcopy WHERE date = ?", [ds],
            )
            if fno_df is None or fno_df.empty:
                continue

            chain = prepare_nifty_chain(
                fno_df, symbol=symbol, instrument=instrument, max_expiries=2,
            )
            if chain is None:
                continue

            result = fit_sanos(
                market_strikes=chain["market_strikes"],
                market_calls=chain["market_calls"],
                atm_variances=chain["atm_variances"],
                expiry_labels=chain.get("expiry_labels"),
                eta=0.50,
                n_model_strikes=80,
            )
            if not result.lp_success:
                continue

            K, q = extract_density(result, expiry_idx=0, n_points=300)
            mu, var, skew, kurt = compute_moments(K, q)
            dK = K[1] - K[0]
            std = math.sqrt(max(var, 1e-14))

            skew_arr[i] = skew
            H = shannon_entropy(q, dK)
            entropy_arr[i] = H
            lt, rt = tail_weights(K, q, mu, std)
            left_tail_arr[i] = lt

            if prev_density is not None and prev_K is not None:
                if len(prev_density) == len(q):
                    kl_arr[i] = kl_divergence(q, prev_density, dK)

            prev_density = q
            prev_K = K

        except Exception as exc:
            logger.debug("SANOS failed for %s on %s: %s", symbol, ds, exc)
            continue

    return {
        "sanos_skew": skew_arr,
        "sanos_left_tail": left_tail_arr,
        "sanos_entropy": entropy_arr,
        "sanos_kl": kl_arr,
    }


def compute_features_expanded_gpu(
    closes: np.ndarray,
    dates: list,
    *,
    alpha_window: int = 60,
    frac_d: float = 0.226,
    mock_theta_n_terms: int = 50,
    # OHLCV for new features (optional — omit for close-only mode)
    opens: np.ndarray | None = None,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    # Multi-index returns for RMT (optional)
    sector_returns: np.ndarray | None = None,
    # Hawkes ratio (pre-computed from tick data, optional)
    hawkes_ratio: np.ndarray | None = None,
    # SANOS cache (pre-computed from option chain, optional)
    sanos_cache: dict | None = None,
    # Store for option chain data (optional)
    store=None,
    symbol: str = "NIFTY",
) -> dict[str, np.ndarray]:
    """Expanded GPU feature pipeline: original 6 + 12 new features.

    Original features (from compute_features_array_gpu):
      alpha, coherence, phase, mock_theta_div, frac_d_series, momentum

    New Tier 1 features:
      period_energy     — Ramanujan periodogram normalised peak energy
      entropy           — rolling Shannon entropy (predictability)
      yang_zhang_vol    — Yang-Zhang realized vol (annualized)
      atr_norm          — Normalised ATR (ATR/close)
      vpin              — Volume-synced PIN (order flow toxicity)

    New RMT features:
      rmt_absorption    — absorption ratio (systemic risk)
      rmt_mp_excess     — largest eigenvalue / MP upper bound

    SANOS risk-neutral density features:
      sanos_skew        — risk-neutral skewness (crash fear)
      sanos_left_tail   — left tail probability mass
      sanos_entropy     — Shannon entropy of density
      sanos_kl          — KL divergence from previous day
    """
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Expanded GPU features ({gpu_name}):")
    t_total = time.time()

    n = len(closes)

    # === Phase 1: Original features (reuse compute_features_array_gpu) ===
    t1 = time.time()
    base = compute_features_array_gpu(
        closes,
        alpha_window=alpha_window,
        frac_d=frac_d,
        mock_theta_n_terms=mock_theta_n_terms,
    )
    print(f"    Base features: {time.time()-t1:.2f}s")

    # Momentum (same as base backtest computes)
    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])
    momentum = np.full(n, np.nan)
    for i in range(alpha_window, n):
        momentum[i] = np.sum(log_ret[i - alpha_window + 1:i + 1])

    out = dict(base)  # copy base features
    out["momentum"] = momentum

    # === Phase 2: Tier 1 features ===

    # 2a. Ramanujan period energy
    t1 = time.time()
    out["period_energy"] = _rolling_period_energy(
        closes, window=min(64, alpha_window), max_period=32,
    )
    print(f"    Period energy: {time.time()-t1:.2f}s")

    # 2b. Rolling entropy
    t1 = time.time()
    out["entropy"] = rolling_entropy(closes, word_length=2, window=100)
    print(f"    Entropy:       {time.time()-t1:.2f}s")

    # 2c. Yang-Zhang vol (needs OHLC)
    t1 = time.time()
    if opens is not None and highs is not None and lows is not None:
        out["yang_zhang_vol"] = _rolling_yang_zhang(
            opens, highs, lows, closes, window=20,
        )
    else:
        # Fallback: close-to-close vol
        cc_vol = np.full(n, np.nan)
        for i in range(21, n):
            rets = log_ret[i - 19:i + 1]
            cc_vol[i] = np.std(rets, ddof=1) * np.sqrt(252)
        out["yang_zhang_vol"] = cc_vol
    print(f"    YZ vol:        {time.time()-t1:.2f}s")

    # 2d. Normalised ATR (needs HLC)
    t1 = time.time()
    if highs is not None and lows is not None:
        out["atr_norm"] = _rolling_atr_normalised(highs, lows, closes, window=14)
    else:
        # Proxy: rolling range / close
        range_proxy = np.full(n, np.nan)
        for i in range(14, n):
            w = closes[i - 13:i + 1]
            range_proxy[i] = (w.max() - w.min()) / max(closes[i], 1e-8)
        out["atr_norm"] = range_proxy
    print(f"    ATR norm:      {time.time()-t1:.2f}s")

    # 2e. VPIN (needs volume)
    t1 = time.time()
    if volumes is not None and np.nansum(volumes) > 0:
        out["vpin"] = _rolling_vpin(closes, volumes, window=50)
    else:
        out["vpin"] = np.full(n, np.nan)
    print(f"    VPIN:          {time.time()-t1:.2f}s")

    # === Phase 3: RMT features ===
    t1 = time.time()
    if sector_returns is not None and sector_returns.shape[1] >= 5:
        rmt = rolling_rmt_features(sector_returns, window=60, top_k=5)
        out["rmt_absorption"] = rmt["rmt_absorption"]
        out["rmt_mp_excess"] = rmt["rmt_mp_excess"]
    else:
        out["rmt_absorption"] = np.full(n, np.nan)
        out["rmt_mp_excess"] = np.full(n, np.nan)
    print(f"    RMT:           {time.time()-t1:.2f}s")

    # === Phase 4: Hawkes ratio (from tick data, pre-computed) ===
    if hawkes_ratio is not None:
        out["hawkes_ratio"] = hawkes_ratio
    else:
        out["hawkes_ratio"] = np.full(n, np.nan)

    # === Phase 5: SANOS risk-neutral density features ===
    t1 = time.time()
    sanos = _compute_sanos_features(
        store, symbol, dates, closes,
        sanos_cache=sanos_cache,
    )
    out["sanos_skew"] = sanos["sanos_skew"]
    out["sanos_left_tail"] = sanos["sanos_left_tail"]
    out["sanos_entropy"] = sanos["sanos_entropy"]
    out["sanos_kl"] = sanos["sanos_kl"]
    print(f"    SANOS:         {time.time()-t1:.2f}s")

    # === Phase 6: Timothy Masters indicators ===
    t1 = time.time()
    fti_best, _fti_period, _fti_width = rolling_fti(closes, lookback=128)
    out["fti"] = fti_best
    print(f"    FTI:           {time.time()-t1:.2f}s")

    t1 = time.time()
    out["mutual_info"] = rolling_mutual_information(closes, window=100)
    print(f"    Mutual info:   {time.time()-t1:.2f}s")

    t1 = time.time()
    if highs is not None and lows is not None:
        out["adx"] = rolling_adx(highs, lows, closes, lookback=14)
    else:
        out["adx"] = np.full(n, np.nan)
    print(f"    ADX:           {time.time()-t1:.2f}s")

    t1 = time.time()
    out["detrended_rsi"] = rolling_detrended_rsi(
        closes, short_period=5, long_period=14, reg_length=60,
    )
    print(f"    Detrend RSI:   {time.time()-t1:.2f}s")

    t1 = time.time()
    out["price_var_ratio"] = rolling_price_variance_ratio(
        closes, short_length=10, mult=4,
    )
    print(f"    PVR:           {time.time()-t1:.2f}s")

    t1 = time.time()
    if opens is not None and highs is not None and lows is not None:
        out["price_intensity"] = rolling_price_intensity(
            opens, highs, lows, closes, smooth=10,
        )
    else:
        out["price_intensity"] = np.full(n, np.nan)
    print(f"    Price int:     {time.time()-t1:.2f}s")

    total = time.time() - t_total
    print(f"    EXPANDED TOT:  {total:.2f}s")

    return out


# ---------------------------------------------------------------------------
# Expanded gate-free linear backtest (14 features for Optuna)
# ---------------------------------------------------------------------------

def backtest_expanded_linear(
    features: dict[str, np.ndarray],
    closes: np.ndarray,
    dates: list,
    *,
    # Original 6 feature weights
    w_alpha: float = 0.0,
    w_coherence: float = 0.0,
    w_phase: float = 0.0,
    w_mock_theta: float = 0.0,
    w_frac_d: float = 0.0,
    w_momentum: float = 0.0,
    # New Tier 1 feature weights
    w_period_energy: float = 0.0,
    w_entropy: float = 0.0,
    w_yang_zhang: float = 0.0,
    w_atr_norm: float = 0.0,
    w_vpin: float = 0.0,
    # RMT feature weights
    w_rmt_absorption: float = 0.0,
    w_rmt_mp_excess: float = 0.0,
    # SANOS feature weights
    w_sanos_skew: float = 0.0,
    w_sanos_left_tail: float = 0.0,
    w_sanos_entropy: float = 0.0,
    w_sanos_kl: float = 0.0,
    # Hawkes feature weight
    w_hawkes_ratio: float = 0.0,
    # Timothy Masters feature weights
    w_fti: float = 0.0,
    w_mutual_info: float = 0.0,
    w_adx: float = 0.0,
    w_detrended_rsi: float = 0.0,
    w_price_var_ratio: float = 0.0,
    w_price_intensity: float = 0.0,
    # Entry/exit
    entry_threshold: float = 0.3,
    hold_days: int = 5,
    cost_bps: float = 5.0,
    alpha_window: int = 60,
) -> dict:
    """Gate-free backtest: weighted linear combination of ALL features.

    signal = Σ wᵢ · zᵢ   (z-scored features for fair comparison)
    direction = sign(signal), conviction = |signal|
    """
    n = len(closes)
    warmup = max(alpha_window + 10, 128)  # FTI needs 128
    if n < warmup + 20:
        return {"trades": 0, "sharpe": 0.0}

    # Z-score all features
    feature_map = [
        (w_alpha, _zscore(features["alpha"] - 1.0)),
        (w_coherence, _zscore(features["coherence"])),
        (w_phase, _zscore(features["phase"])),
        (w_mock_theta, _zscore(features["mock_theta_div"])),
        (w_frac_d, _zscore(features["frac_d_series"])),
        (w_momentum, _zscore(features.get("momentum", np.full(n, np.nan)))),
        (w_period_energy, _zscore(features.get("period_energy", np.full(n, np.nan)))),
        (w_entropy, _zscore(features.get("entropy", np.full(n, np.nan)))),
        (w_yang_zhang, _zscore(features.get("yang_zhang_vol", np.full(n, np.nan)))),
        (w_atr_norm, _zscore(features.get("atr_norm", np.full(n, np.nan)))),
        (w_vpin, _zscore(features.get("vpin", np.full(n, np.nan)))),
        (w_rmt_absorption, _zscore(features.get("rmt_absorption", np.full(n, np.nan)))),
        (w_rmt_mp_excess, _zscore(features.get("rmt_mp_excess", np.full(n, np.nan)))),
        (w_sanos_skew, _zscore(features.get("sanos_skew", np.full(n, np.nan)))),
        (w_sanos_left_tail, _zscore(features.get("sanos_left_tail", np.full(n, np.nan)))),
        (w_sanos_entropy, _zscore(features.get("sanos_entropy", np.full(n, np.nan)))),
        (w_sanos_kl, _zscore(features.get("sanos_kl", np.full(n, np.nan)))),
        (w_hawkes_ratio, _zscore(features.get("hawkes_ratio", np.full(n, np.nan)))),
        (w_fti, _zscore(features.get("fti", np.full(n, np.nan)))),
        (w_mutual_info, _zscore(features.get("mutual_info", np.full(n, np.nan)))),
        (w_adx, _zscore(features.get("adx", np.full(n, np.nan)))),
        (w_detrended_rsi, _zscore(features.get("detrended_rsi", np.full(n, np.nan)))),
        (w_price_var_ratio, _zscore(features.get("price_var_ratio", np.full(n, np.nan)))),
        (w_price_intensity, _zscore(features.get("price_intensity", np.full(n, np.nan)))),
    ]

    cost_frac = cost_bps / 10_000
    next_day_ret = np.full(n, np.nan)
    next_day_ret[:-1] = np.diff(closes) / closes[:-1]

    position = 0
    entry_day = 0
    daily_pnl: list[float] = []
    trades: list[dict] = []

    for i in range(warmup, n - 1):
        signal = 0.0
        any_nan = False
        any_active = False
        for w, z_arr in feature_map:
            if abs(w) < 1e-10:
                continue
            any_active = True
            v = z_arr[i]
            if np.isnan(v):
                any_nan = True
                break
            signal += w * v

        if any_nan or not any_active:
            daily_pnl.append(0.0)
            continue

        direction = 1 if signal > 0 else (-1 if signal < 0 else 0)
        conviction = abs(signal)

        if position != 0:
            days_held = i - entry_day
            should_exit = days_held >= hold_days
            should_exit |= (direction != 0 and direction != position
                            and conviction >= entry_threshold)

            if should_exit:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    trade_ret = ret * position - cost_frac
                    daily_pnl.append(trade_ret)
                    trades.append({"ret": trade_ret})
                else:
                    daily_pnl.append(0.0)
                position = 0
            else:
                ret = next_day_ret[i]
                if not np.isnan(ret):
                    daily_pnl.append(ret * position)
                else:
                    daily_pnl.append(0.0)
                continue

        if position == 0 and conviction >= entry_threshold and direction != 0:
            position = direction
            entry_day = i
            daily_pnl.append(0.0)
        else:
            daily_pnl.append(0.0)

    if not daily_pnl:
        return {"trades": 0, "sharpe": 0.0}

    pnl_arr = np.array(daily_pnl)
    std = np.std(pnl_arr, ddof=1) if len(pnl_arr) > 1 else 0.0
    sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    wins = sum(1 for t in trades if t["ret"] > 0)

    return {
        "trades": len(trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / max(len(trades), 1),
    }


# ---------------------------------------------------------------------------
# Intraday backtest: daily signal direction + 1-min bar execution
# ---------------------------------------------------------------------------

def backtest_intraday_expanded(
    features: dict[str, np.ndarray],
    closes: np.ndarray,
    dates: list,
    intraday_bars: dict,  # date → np.ndarray of 1-min close prices
    *,
    w_alpha: float = 0.0, w_coherence: float = 0.0, w_phase: float = 0.0,
    w_mock_theta: float = 0.0, w_frac_d: float = 0.0, w_momentum: float = 0.0,
    w_period_energy: float = 0.0, w_entropy: float = 0.0,
    w_yang_zhang: float = 0.0, w_atr_norm: float = 0.0, w_vpin: float = 0.0,
    w_rmt_absorption: float = 0.0, w_rmt_mp_excess: float = 0.0,
    w_sanos_skew: float = 0.0, w_sanos_left_tail: float = 0.0,
    w_sanos_entropy: float = 0.0, w_sanos_kl: float = 0.0,
    w_hawkes_ratio: float = 0.0,
    w_fti: float = 0.0, w_mutual_info: float = 0.0, w_adx: float = 0.0,
    w_detrended_rsi: float = 0.0, w_price_var_ratio: float = 0.0,
    w_price_intensity: float = 0.0,
    entry_threshold: float = 0.3,
    target_mult: float = 2.0,
    stop_mult: float = 1.5,
    max_trades_per_day: int = 3,
    cost_bps: float = 5.0,
    alpha_window: int = 60,
) -> dict:
    """Intraday backtest: daily features give direction, 1-min bars for execution."""
    n = len(closes)
    warmup = max(alpha_window + 10, 128)
    if n < warmup + 20:
        return {"trades": 0, "sharpe": 0.0}

    feature_map = [
        (w_alpha, _zscore(features["alpha"] - 1.0)),
        (w_coherence, _zscore(features["coherence"])),
        (w_phase, _zscore(features["phase"])),
        (w_mock_theta, _zscore(features["mock_theta_div"])),
        (w_frac_d, _zscore(features["frac_d_series"])),
        (w_momentum, _zscore(features.get("momentum", np.full(n, np.nan)))),
        (w_period_energy, _zscore(features.get("period_energy", np.full(n, np.nan)))),
        (w_entropy, _zscore(features.get("entropy", np.full(n, np.nan)))),
        (w_yang_zhang, _zscore(features.get("yang_zhang_vol", np.full(n, np.nan)))),
        (w_atr_norm, _zscore(features.get("atr_norm", np.full(n, np.nan)))),
        (w_vpin, _zscore(features.get("vpin", np.full(n, np.nan)))),
        (w_rmt_absorption, _zscore(features.get("rmt_absorption", np.full(n, np.nan)))),
        (w_rmt_mp_excess, _zscore(features.get("rmt_mp_excess", np.full(n, np.nan)))),
        (w_sanos_skew, _zscore(features.get("sanos_skew", np.full(n, np.nan)))),
        (w_sanos_left_tail, _zscore(features.get("sanos_left_tail", np.full(n, np.nan)))),
        (w_sanos_entropy, _zscore(features.get("sanos_entropy", np.full(n, np.nan)))),
        (w_sanos_kl, _zscore(features.get("sanos_kl", np.full(n, np.nan)))),
        (w_hawkes_ratio, _zscore(features.get("hawkes_ratio", np.full(n, np.nan)))),
        (w_fti, _zscore(features.get("fti", np.full(n, np.nan)))),
        (w_mutual_info, _zscore(features.get("mutual_info", np.full(n, np.nan)))),
        (w_adx, _zscore(features.get("adx", np.full(n, np.nan)))),
        (w_detrended_rsi, _zscore(features.get("detrended_rsi", np.full(n, np.nan)))),
        (w_price_var_ratio, _zscore(features.get("price_var_ratio", np.full(n, np.nan)))),
        (w_price_intensity, _zscore(features.get("price_intensity", np.full(n, np.nan)))),
    ]

    cost_frac = cost_bps / 10_000
    daily_pnl: list[float] = []
    all_trades: list[dict] = []
    bar_warmup = 30

    for i in range(warmup, n):
        d = dates[i]
        signal = 0.0
        skip = False
        any_active = False
        for w, z_arr in feature_map:
            if abs(w) < 1e-10:
                continue
            any_active = True
            v = z_arr[i]
            if np.isnan(v):
                skip = True
                break
            signal += w * v

        if skip or not any_active or abs(signal) < entry_threshold:
            daily_pnl.append(0.0)
            continue

        direction = 1 if signal > 0 else -1
        bars = intraday_bars.get(d)
        if bars is None or len(bars) < bar_warmup + 30:
            daily_pnl.append(0.0)
            continue

        bar_rets = np.diff(bars[:bar_warmup]) / np.maximum(bars[:bar_warmup - 1], 1e-8)
        sigma = np.std(bar_rets, ddof=1)
        if sigma < 1e-10:
            daily_pnl.append(0.0)
            continue

        target_ret = target_mult * sigma
        stop_ret = stop_mult * sigma
        time_stop_bar = len(bars) - 15

        day_pnl = 0.0
        day_trades = 0
        pos = 0
        entry_px = 0.0

        for j in range(bar_warmup, len(bars)):
            if pos == 0 and day_trades < max_trades_per_day:
                pos = direction
                entry_px = bars[j]
                day_trades += 1
                continue
            if pos != 0:
                ret = (bars[j] - entry_px) / entry_px * pos
                if ret >= target_ret:
                    day_pnl += ret - cost_frac
                    all_trades.append({"ret": ret - cost_frac, "exit": "target"})
                    pos = 0
                elif ret <= -stop_ret:
                    day_pnl += ret - cost_frac
                    all_trades.append({"ret": ret - cost_frac, "exit": "stop"})
                    pos = 0
                elif j >= time_stop_bar:
                    day_pnl += ret - cost_frac
                    all_trades.append({"ret": ret - cost_frac, "exit": "time"})
                    pos = 0

        if pos != 0:
            ret = (bars[-1] - entry_px) / entry_px * pos
            day_pnl += ret - cost_frac
            all_trades.append({"ret": ret - cost_frac, "exit": "eod"})

        daily_pnl.append(day_pnl)

    if not daily_pnl:
        return {"trades": 0, "sharpe": 0.0}

    pnl_arr = np.array(daily_pnl)
    std = np.std(pnl_arr, ddof=1) if len(pnl_arr) > 1 else 0.0
    sharpe = float(np.mean(pnl_arr) / std * np.sqrt(252)) if std > 0 else 0.0
    cumulative = np.cumsum(pnl_arr)
    total_return = float(cumulative[-1])
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0
    wins = sum(1 for t in all_trades if t["ret"] > 0)

    exit_counts: dict[str, int] = {}
    for t in all_trades:
        exit_counts[t["exit"]] = exit_counts.get(t["exit"], 0) + 1

    return {
        "trades": len(all_trades),
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate": wins / max(len(all_trades), 1),
        "exit_counts": exit_counts,
    }
