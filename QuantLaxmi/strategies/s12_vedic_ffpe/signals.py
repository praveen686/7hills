"""S12 Vedic FFPE — Signal generation pipeline.

Fully causal, step-by-step signal construction:
  1. Base features: realized vol, returns
  2. Fractional α estimation (rolling 60-day)
  3. Regime classification with hysteresis
  4. Mock theta q-series + volatility distortion
  5. Angular coherence via Madhava kernel
  6. Aryabhata phase tracking
  7. Fractional differencing (d=0.226)
  8. Composite signal:
     - α < 0.85: contrarian (mean-reversion)
     - α > 1.15: momentum (trend-following)
     - 0.85 ≤ α ≤ 1.15: flat (unless mock_theta spikes)

All features use only past data.  No look-ahead bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from features.fractional import (
    estimate_alpha,
    estimate_alpha_msd,
    estimate_alpha_waiting,
    fractional_differentiation,
    solve_ffpe_1d,
)
from features.mock_theta import (
    mock_theta_f,
    mock_theta_phi,
    mock_theta_chi,
    ramanujan_volatility_distortion,
)
from features.vedic_angular import (
    madhava_kernel,
    angular_coherence,
    update_regime_centroids,
    aryabhata_phase,
)
from features.ramanujan import dominant_periods


# ---------------------------------------------------------------------------
# Signal output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VedicSignal:
    """Raw signal from the Vedic FFPE pipeline."""
    direction: str          # "long" | "short" | "flat"
    conviction: float       # [0, 1]
    regime: str             # "subdiffusive" | "normal" | "superdiffusive"
    alpha: float            # fractional diffusion order
    phase: float            # Aryabhata phase [0, 1]
    coherence: float        # angular coherence to regime centroid
    mock_theta_div: float   # mock theta divergence (regime transition speed)
    frac_d_value: float     # latest fractionally differenced price
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regime classification with hysteresis
# ---------------------------------------------------------------------------

def classify_regime(alpha: float,
                    alpha_lo: float = 0.85,
                    alpha_hi: float = 1.15) -> str:
    """Classify α into regime."""
    if alpha < alpha_lo:
        return "subdiffusive"
    elif alpha > alpha_hi:
        return "superdiffusive"
    else:
        return "normal"


def classify_regime_hysteresis(alpha: float,
                               prev_regime: str,
                               bars_in_regime: int,
                               alpha_lo: float = 0.85,
                               alpha_hi: float = 1.15,
                               min_hold: int = 3) -> tuple[str, int]:
    """Classify with hysteresis: suppress flips if regime held < min_hold bars."""
    new_regime = classify_regime(alpha, alpha_lo, alpha_hi)
    if new_regime != prev_regime and bars_in_regime < min_hold:
        return prev_regime, bars_in_regime + 1
    elif new_regime != prev_regime:
        return new_regime, 1
    else:
        return prev_regime, bars_in_regime + 1


# ---------------------------------------------------------------------------
# Core signal computation
# ---------------------------------------------------------------------------

def compute_daily_signal(
    close: np.ndarray,
    *,
    alpha_window: int = 60,
    alpha_lo: float = 0.85,
    alpha_hi: float = 1.15,
    frac_d: float = 0.226,
    madhava_order: int = 4,
    centroid_decay: float = 0.99,
    min_conviction: float = 0.15,
    mock_theta_n_terms: int = 50,
    phase_entry_lo: float = 0.0,
    phase_entry_hi: float = 0.5,
    prev_regime: str = "normal",
    bars_in_regime: int = 0,
    regime_centroids: dict | None = None,
) -> tuple[VedicSignal, dict]:
    """Compute composite signal from price array (closing prices, 1 per bar).

    Parameters
    ----------
    close : 1-D array of closing prices (most recent = last element)
    alpha_window : lookback for α estimation
    alpha_lo, alpha_hi : regime thresholds
    frac_d : fractional differencing order
    madhava_order : Madhava kernel expansion order
    centroid_decay : EMA decay for regime centroids
    min_conviction : minimum conviction to trade
    mock_theta_n_terms : mock theta series truncation
    phase_entry_lo, phase_entry_hi : favorable phase window
    prev_regime : previous regime label (for hysteresis)
    bars_in_regime : bars spent in current regime
    regime_centroids : dict of regime centroid vectors (mutable state)

    Returns
    -------
    (signal, updated_state) where updated_state has keys:
      prev_regime, bars_in_regime, regime_centroids
    """
    if regime_centroids is None:
        regime_centroids = {}

    n = len(close)
    if n < alpha_window + 5:
        return (
            VedicSignal("flat", 0.0, "normal", 1.0, 0.5, 0.0, 0.0, 0.0),
            {"prev_regime": prev_regime, "bars_in_regime": bars_in_regime,
             "regime_centroids": regime_centroids},
        )

    # --- Step 1: Log returns ---
    log_ret = np.diff(np.log(np.maximum(close, 1e-8)))

    # Use most recent alpha_window returns
    recent_rets = log_ret[-alpha_window:]

    # --- Step 2: Fractional α estimation ---
    a_msd = estimate_alpha_msd(recent_rets, max_lag=min(30, alpha_window // 2))
    a_wt = estimate_alpha_waiting(recent_rets)
    alpha = 0.6 * a_msd + 0.4 * a_wt

    # --- Step 3: Regime classification with hysteresis ---
    regime, bars = classify_regime_hysteresis(
        alpha, prev_regime, bars_in_regime, alpha_lo, alpha_hi,
    )

    # --- Step 4: Mock theta features ---
    sigma = np.std(recent_rets, ddof=1) if len(recent_rets) > 1 else 1e-8
    sigma = max(sigma, 1e-8)
    mean_abs_ret = np.mean(np.abs(recent_rets))
    q_val = math.exp(-math.pi / (1.0 + mean_abs_ret / sigma))

    mt_f = mock_theta_f(q_val, mock_theta_n_terms)
    mt_phi = mock_theta_phi(q_val, mock_theta_n_terms)
    mt_chi = mock_theta_chi(q_val, mock_theta_n_terms)
    mt_ratio = mt_f / max(mt_phi, 1e-30)

    # Volatility distortion
    ann_vol = sigma * math.sqrt(252)
    vol_dist = ramanujan_volatility_distortion(ann_vol)

    # Mock theta divergence (change in ratio from prev bar)
    # We approximate by comparing current ratio to a baseline
    mt_div = abs(mt_ratio - 1.0)  # divergence from unity

    # --- Step 5: Angular coherence ---
    # Feature vector: (vol, momentum, mean_ret)
    vol = np.std(recent_rets, ddof=1)
    mom = np.sum(recent_rets)
    mean_r = np.mean(recent_rets)
    feat_vec = np.array([vol, mom, mean_r])

    if regime_centroids:
        best_regime_ang, coherence, higher_order = angular_coherence(
            feat_vec, regime_centroids, order=madhava_order,
        )
    else:
        coherence = 0.5
        higher_order = 0.0

    # Update centroids
    regime_centroids = update_regime_centroids(
        regime_centroids, feat_vec, regime, decay=centroid_decay,
    )

    # --- Step 6: Aryabhata phase ---
    phase_window = min(alpha_window, n)
    window_prices = close[-phase_window:]
    window_rets = log_ret[-(phase_window - 1):]
    periods = dominant_periods(
        window_rets - window_rets.mean(),
        max_period=min(32, phase_window // 2),
        top_k=1,
    )
    period = periods[0] if periods else phase_window // 4
    phase_arr, phase_vel_arr = aryabhata_phase(window_prices, period)
    phase = float(phase_arr[-1]) if not np.isnan(phase_arr[-1]) else 0.5

    # --- Step 7: Fractional differencing ---
    d_series = fractional_differentiation(close[-alpha_window:], frac_d)
    frac_d_val = float(d_series[-1]) if not np.isnan(d_series[-1]) else 0.0

    # --- Step 8: Phase gate ---
    phase_favorable = phase_entry_lo <= phase <= phase_entry_hi

    # --- Step 9: Composite signal ---
    direction = "flat"
    conviction = 0.0

    if regime == "subdiffusive":
        # Contrarian: mean-reversion at phase extrema
        conviction = float(np.clip(
            coherence * (1 - mt_div) * (1.0 if phase_favorable else 0.3), 0, 1,
        ))
        # Direction: opposite of recent momentum
        if mom > 0:
            direction = "short"
        elif mom < 0:
            direction = "long"
        else:
            direction = "flat"
            conviction = 0.0

    elif regime == "superdiffusive":
        # Momentum: trend-following with α-based conviction
        conviction = float(np.clip(
            coherence * (alpha - 1.0) * (1.0 if phase_favorable else 0.3), 0, 1,
        ))
        if mom > 0:
            direction = "long"
        elif mom < 0:
            direction = "short"
        else:
            direction = "flat"
            conviction = 0.0

    elif regime == "normal":
        # Default flat unless mock theta divergence spikes (early regime entry)
        if mt_div > 0.5:
            # Early regime transition detected
            conviction = float(np.clip(
                0.5 * mt_div * coherence * (1.0 if phase_favorable else 0.3), 0, 1,
            ))
            # Use frac_d_series momentum for direction
            if frac_d_val > 0:
                direction = "long"
            elif frac_d_val < 0:
                direction = "short"
        else:
            direction = "flat"
            conviction = 0.0

    # Apply minimum conviction threshold
    if conviction < min_conviction:
        direction = "flat"
        conviction = 0.0

    signal = VedicSignal(
        direction=direction,
        conviction=conviction,
        regime=regime,
        alpha=alpha,
        phase=phase,
        coherence=coherence,
        mock_theta_div=mt_div,
        frac_d_value=frac_d_val,
        metadata={
            "alpha_msd": a_msd,
            "alpha_waiting": a_wt,
            "mock_theta_f": mt_f,
            "mock_theta_phi": mt_phi,
            "mock_theta_chi": mt_chi,
            "mock_theta_ratio": mt_ratio,
            "vol_distortion": vol_dist,
            "higher_order_residual": higher_order,
            "dominant_period": period,
            "frac_d": frac_d,
        },
    )

    state = {
        "prev_regime": regime,
        "bars_in_regime": bars,
        "regime_centroids": regime_centroids,
    }

    return signal, state


def compute_features_array(
    closes: np.ndarray,
    *,
    alpha_window: int = 60,
    frac_d: float = 0.226,
    mock_theta_n_terms: int = 50,
) -> dict[str, np.ndarray]:
    """Compute all S12 features for a full price array.

    Returns dict of feature name → array (same length as closes).
    Used by research.py for feature analysis.
    """
    n = len(closes)
    log_ret = np.diff(np.log(np.maximum(closes, 1e-8)))
    log_ret = np.concatenate([[0.0], log_ret])

    # Output arrays
    out = {
        "alpha": np.full(n, np.nan),
        "alpha_msd": np.full(n, np.nan),
        "alpha_waiting": np.full(n, np.nan),
        "hurst": np.full(n, np.nan),
        "mock_theta_f": np.full(n, np.nan),
        "mock_theta_phi": np.full(n, np.nan),
        "mock_theta_ratio": np.full(n, np.nan),
        "mock_theta_div": np.full(n, np.nan),
        "vol_distortion": np.full(n, np.nan),
        "coherence": np.full(n, np.nan),
        "phase": np.full(n, np.nan),
        "frac_d_series": np.full(n, np.nan),
        "regime": np.full(n, np.nan),  # 0=sub, 1=normal, 2=super
    }

    # Fractional differencing (full series, once)
    d_series = fractional_differentiation(closes, frac_d)
    out["frac_d_series"] = d_series

    # State
    regime_centroids: dict[str, np.ndarray] = {}
    prev_ratio = np.nan

    for i in range(alpha_window, n):
        rets = log_ret[i - alpha_window + 1: i + 1]

        # Alpha
        a_msd = estimate_alpha_msd(rets, max_lag=min(30, alpha_window // 2))
        a_wt = estimate_alpha_waiting(rets)
        a = 0.6 * a_msd + 0.4 * a_wt
        out["alpha"][i] = a
        out["alpha_msd"][i] = a_msd
        out["alpha_waiting"][i] = a_wt
        out["hurst"][i] = a / 2.0

        # Regime
        if a < 0.85:
            out["regime"][i] = 0
            regime_label = "subdiffusive"
        elif a > 1.15:
            out["regime"][i] = 2
            regime_label = "superdiffusive"
        else:
            out["regime"][i] = 1
            regime_label = "normal"

        # Mock theta
        sigma = np.std(rets, ddof=1) if len(rets) > 1 else 1e-8
        sigma = max(sigma, 1e-8)
        mean_abs = np.mean(np.abs(rets))
        q_val = math.exp(-math.pi / (1.0 + mean_abs / sigma))

        mt_f = mock_theta_f(q_val, mock_theta_n_terms)
        mt_phi = mock_theta_phi(q_val, mock_theta_n_terms)
        ratio = mt_f / max(mt_phi, 1e-30)
        out["mock_theta_f"][i] = mt_f
        out["mock_theta_phi"][i] = mt_phi
        out["mock_theta_ratio"][i] = ratio
        out["mock_theta_div"][i] = abs(ratio - prev_ratio) if not np.isnan(prev_ratio) else 0
        prev_ratio = ratio

        # Vol distortion
        ann_vol = sigma * math.sqrt(252)
        out["vol_distortion"][i] = ramanujan_volatility_distortion(ann_vol)

        # Angular coherence
        vol = np.std(rets, ddof=1)
        mom = np.sum(rets)
        mean_r = np.mean(rets)
        feat_vec = np.array([vol, mom, mean_r])
        if regime_centroids:
            _, coh, _ = angular_coherence(feat_vec, regime_centroids, order=4)
            out["coherence"][i] = coh
        else:
            out["coherence"][i] = 0.5
        regime_centroids = update_regime_centroids(
            regime_centroids, feat_vec, regime_label, decay=0.99,
        )

        # Phase
        if i >= alpha_window * 2:
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
                out["phase"][i] = ph[-1]

    return out
