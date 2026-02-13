"""Momentum Transformer — TFT-inspired model with Changepoint Detection.

A simplified Temporal Fusion Transformer architecture adapted for momentum
trading, combining Variable Selection Networks, LSTM encoding, multi-head
attention, and optional changepoint features via the ``ruptures`` library.

Architecture
------------
1. Variable Selection Network (VSN) with Gated Residual Networks (GRN)
2. 2-layer LSTM encoder (hidden_dim=64)
3. Multi-head attention (4 heads) over encoded sequence
4. Tanh output head -> position signal in [-1, 1]

Training
--------
- Walk-forward: train 120d, test 20d, step 20d
- Pre-train with BCE on next-day return sign
- Fine-tune with Sharpe ratio loss (negative Sharpe of strategy returns)
- Adam optimizer, lr=1e-3, weight_decay=1e-5
- Early stopping: patience=10 on validation Sharpe

Data
----
Reads from ``MarketDataStore.sql()`` using the ``nse_index_close`` view.
Falls back to sklearn GradientBoosting if PyTorch is unavailable.

References
----------
- Lim et al. (2021), "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting"
- Wood et al. (2022), "Trading with the Momentum Transformer: An Intelligent
  and Interpretable Architecture"
- Killick et al. (2012), "Optimal Detection of Changepoints with a Linear
  Computational Cost" (PELT algorithm)
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("PyTorch available, device=%s", _DEVICE)
except ImportError:
    HAS_TORCH = False
    _DEVICE = None
    logger.warning("PyTorch not available; will use sklearn fallback")

try:
    import ruptures  # type: ignore[import-untyped]

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    logger.info("ruptures not installed; changepoint features disabled")

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]

    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    logger.info("hmmlearn not installed; HMM features disabled")

# Core project imports
from quantlaxmi.features.fractional import fractional_differentiation
from quantlaxmi.data.store import MarketDataStore


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MomentumTFMConfig:
    """Hyperparameters for the Momentum Transformer strategy."""

    # Architecture
    hidden_dim: int = 64
    lstm_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    sequence_length: int = 60

    # Training
    train_window: int = 120
    test_window: int = 20
    step_size: int = 20
    pretrain_epochs: int = 30
    finetune_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    patience: int = 10

    # Feature engineering
    frac_diff_d: float = 0.3
    changepoint_penalty: float = 3.0
    changepoint_lookback: int = 5

    # Backtest
    cost_bps: float = 5.0  # round-trip cost in basis points

    # Position sizing / risk management
    max_position: float = 0.25       # max absolute position size (25% allocation)
    position_smooth: float = 0.3     # exponential smoothing factor for new signals
    max_daily_turnover: float = 0.5  # max position change per day

    # Multi-horizon forecasting (default off for backward compat)
    multi_horizon: bool = False
    # Auxiliary horizon offsets and loss weights
    # Main T+1 always has weight 1.0; these are the auxiliary ones
    aux_horizons: tuple[int, ...] = (2, 3, 5)
    aux_weights: tuple[float, ...] = (0.3, 0.2, 0.1)

    # ALiBi positional encoding (default on)
    use_alibi: bool = True

    # Use scaled dot-product attention (Flash Attention backend when available)
    use_sdpa: bool = True

    # Temporal and static covariate encoders (Lim et al., 2021)
    use_temporal_covariates: bool = True
    use_static_covariates: bool = True
    n_temporal_features: int = 6   # day_of_week, month, day_of_month, is_expiry_week, is_monthly_expiry, quarter
    n_static_categories: int = 4   # NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
    static_embed_dim: int = 8

    # Derived
    n_features: int = field(init=False)

    def __post_init__(self):
        # Base features count (calculated in build_features, set here as max)
        self.n_features = 40  # upper bound; actual count set at runtime


# ============================================================================
# Feature Engineering
# ============================================================================

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index (Wilder's smoothed)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)

    if len(close) < period + 1:
        return np.full_like(close, 50.0)

    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average via pandas (leverages Cython internals)."""
    return pd.Series(x).ewm(span=span, min_periods=span).mean().values


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average."""
    return pd.Series(x).rolling(window, min_periods=window).mean().values


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (ddof=1)."""
    return pd.Series(x).rolling(window, min_periods=window).std(ddof=1).values


def _macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (12, 26, 9): returns (macd_line, signal_line, histogram)."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_pctb(close: np.ndarray, window: int = 20,
                    n_std: float = 2.0) -> np.ndarray:
    """Bollinger Band %B: (close - lower) / (upper - lower)."""
    sma = _sma(close, window)
    std = _rolling_std(close, window)
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower
    pctb = np.where(band_width > 0, (close - lower) / band_width, 0.5)
    return pctb


def _changepoint_features(close: np.ndarray,
                          penalty: float = 3.0,
                          lookback: int = 5,
                          min_window: int = 30,
                          refit_every: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal changepoint features via PELT algorithm (ruptures library).

    Uses an expanding window so that features at time *t* depend only on
    ``close[:t+1]`` — no future data is ever seen.

    PELT is refitted every ``refit_every`` bars for efficiency; intermediate
    bars carry forward the most recent result.

    Returns
    -------
    cp_indicator : array
        Binary — 1 if a changepoint was detected within ``lookback`` bars.
    cp_distance : array
        Normalized distance (0..1) to the nearest changepoint.
    cp_magnitude : array
        Return magnitude at the nearest changepoint.
    """
    n = len(close)
    cp_indicator = np.zeros(n)
    cp_distance = np.ones(n)  # default: far from any changepoint
    cp_magnitude = np.zeros(n)

    if not HAS_RUPTURES or n < min_window:
        return cp_indicator, cp_distance, cp_magnitude

    # Full log-returns series (we'll slice causally below)
    log_returns = np.diff(np.log(close))
    if len(log_returns) < 20:
        return cp_indicator, cp_distance, cp_magnitude

    # Track last-known changepoints for carry-forward between refits
    last_changepoints: list[int] = []
    last_fit_t = -1

    for t in range(min_window, n):
        # Refit periodically for efficiency
        if last_fit_t < 0 or (t - last_fit_t) >= refit_every:
            try:
                # CAUSAL: only use log_returns up to time t
                # log_returns[i] = log(close[i+1]) - log(close[i]),
                # so log_returns[:t] uses close[:t+1] which is causal at time t.
                lr_causal = log_returns[:t]
                algo = ruptures.Pelt(model="rbf", min_size=5).fit(
                    lr_causal.reshape(-1, 1)
                )
                changepoints = algo.predict(pen=penalty)
                # ruptures returns indices (1-based, last element is len(signal))
                changepoints = [cp for cp in changepoints if cp < len(lr_causal)]
                last_changepoints = changepoints
                last_fit_t = t
            except Exception:
                # Keep previous changepoints on failure
                pass

        if not last_changepoints:
            continue

        # Find nearest changepoint to current bar
        nearest_dist = n  # large default
        nearest_cp = -1
        for cp in last_changepoints:
            cp_idx = cp + 1  # map from log_returns index back to close index
            dist = abs(t - cp_idx)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_cp = cp_idx

        # Binary indicator: changepoint within lookback
        if nearest_dist <= lookback:
            cp_indicator[t] = 1.0

        # Normalized distance
        cp_distance[t] = min(nearest_dist / max(lookback, 1), 1.0)

        # Magnitude at nearest changepoint
        if 0 < nearest_cp < n:
            cp_magnitude[t] = (
                (close[nearest_cp] - close[max(0, nearest_cp - 1)])
                / close[max(0, nearest_cp - 1)]
            )

    return cp_indicator, cp_distance, cp_magnitude


def _hmm_regime_features(returns: np.ndarray, n_states: int = 3,
                         min_window: int = 120,
                         refit_every: int = 20) -> np.ndarray:
    """Causal HMM regime probabilities using expanding-window forward filter.

    At each time *t* (>= ``min_window``), ``predict_proba`` is called on
    ``returns[:t+1]`` so that only data up to *t* is used — no future data
    leaks into the posterior.  The HMM is refitted every ``refit_every``
    bars for efficiency; between refits the same model parameters are reused
    but the causal window still expands each bar.

    Returns an (n, n_states) array of posterior state probabilities.
    If hmmlearn is unavailable or fitting fails, returns zeros.
    """
    n = len(returns)
    if not HAS_HMM or n < min_window:
        return np.zeros((n, n_states))

    posteriors = np.zeros((n, n_states))

    try:
        last_fit = -1
        model = None
        for t in range(min_window, n):
            # Refit periodically for efficiency
            if model is None or (t - last_fit) >= refit_every:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",
                    n_iter=50,
                    random_state=42,
                    verbose=False,
                )
                X_train = returns[:t].reshape(-1, 1)
                model.fit(X_train)
                last_fit = t

            # Forward-only: predict on data up to t (causal).
            # predict_proba on [0..t] runs forward-backward on that window
            # only, so data from t+1..T is never seen.
            X_causal = returns[: t + 1].reshape(-1, 1)
            probs = model.predict_proba(X_causal)
            posteriors[t] = probs[-1]  # only take the last row (time t)
    except Exception:
        logger.debug("HMM fitting failed; returning zeros")
        return np.zeros((n, n_states))

    return posteriors


def build_features(close: np.ndarray,
                   volume: Optional[np.ndarray] = None,
                   cfg: Optional[MomentumTFMConfig] = None,
                   ) -> tuple[np.ndarray, list[str]]:
    """Construct the full feature matrix from a close price array.

    Parameters
    ----------
    close : np.ndarray
        Daily close prices (ascending date order).
    volume : np.ndarray, optional
        Daily volume. If None, volume-based features are filled with 0.
    cfg : MomentumTFMConfig, optional
        Configuration for feature engineering parameters.

    Returns
    -------
    features : np.ndarray of shape (n, n_features)
        The feature matrix. Leading rows with insufficient history are NaN.
    names : list[str]
        Feature names, in column order.
    """
    if cfg is None:
        cfg = MomentumTFMConfig()

    n = len(close)
    features: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 1. Momentum returns: 1d, 5d, 10d, 21d, 63d
    # ------------------------------------------------------------------
    log_close = np.log(close)
    for lag in [1, 5, 10, 21, 63]:
        ret = np.full(n, np.nan)
        if n > lag:
            ret[lag:] = log_close[lag:] - log_close[:-lag]
        features[f"ret_{lag}d"] = ret

    # ------------------------------------------------------------------
    # 2. Realized volatility: 5d, 21d
    # ------------------------------------------------------------------
    daily_ret = np.full(n, np.nan)
    daily_ret[1:] = np.diff(log_close)
    for window in [5, 21]:
        features[f"rvol_{window}d"] = _rolling_std(daily_ret, window)

    # ------------------------------------------------------------------
    # 3. RSI: 7, 14, 21
    # ------------------------------------------------------------------
    for period in [7, 14, 21]:
        rsi_vals = _rsi(close, period)
        # Normalize to [-1, 1] range for neural net
        features[f"rsi_{period}"] = (rsi_vals - 50.0) / 50.0

    # ------------------------------------------------------------------
    # 4. MACD (12, 26, 9)
    # ------------------------------------------------------------------
    macd_line, signal_line, histogram = _macd(close)
    # Normalize by ATR-like scaling (21d vol of close)
    close_std = _rolling_std(close, 21)
    safe_std = np.where(close_std > 0, close_std, 1.0)
    features["macd_line"] = macd_line / safe_std
    features["macd_signal"] = signal_line / safe_std
    features["macd_hist"] = histogram / safe_std

    # ------------------------------------------------------------------
    # 5. Bollinger Band %B (20, 2)
    # ------------------------------------------------------------------
    features["bb_pctb"] = _bollinger_pctb(close, 20, 2.0)

    # ------------------------------------------------------------------
    # 6. Volume ratio (20-day normalized)
    # ------------------------------------------------------------------
    if volume is not None and len(volume) == n:
        vol_sma = _sma(volume.astype(np.float64), 20)
        safe_vol_sma = np.where(vol_sma > 0, vol_sma, 1.0)
        features["volume_ratio"] = volume / safe_vol_sma
    else:
        features["volume_ratio"] = np.zeros(n)

    # ------------------------------------------------------------------
    # 7. Fractionally differentiated close (d=0.3)
    # ------------------------------------------------------------------
    # max_window caps the convolution length so we get valid values even
    # for short series (d<0.5 weights decay as a power law and may need
    # thousands of terms to reach threshold).
    frac_close = fractional_differentiation(
        close, d=cfg.frac_diff_d, max_window=min(63, n - 1)
    )
    # Normalize by rolling std
    frac_std = _rolling_std(frac_close, 21)
    safe_frac_std = np.where((frac_std > 0) & ~np.isnan(frac_std), frac_std, 1.0)
    features["frac_diff_close"] = frac_close / safe_frac_std

    # ------------------------------------------------------------------
    # 8. Moving average crossovers: 5/20, 10/50
    # ------------------------------------------------------------------
    for fast, slow in [(5, 20), (10, 50)]:
        sma_fast = _sma(close, fast)
        sma_slow = _sma(close, slow)
        # Normalized crossover: (fast - slow) / slow
        safe_slow = np.where(sma_slow > 0, sma_slow, 1.0)
        features[f"ma_cross_{fast}_{slow}"] = (sma_fast - sma_slow) / safe_slow

    # ------------------------------------------------------------------
    # 9. Changepoint features (ruptures PELT)
    # ------------------------------------------------------------------
    cp_ind, cp_dist, cp_mag = _changepoint_features(
        close, penalty=cfg.changepoint_penalty, lookback=cfg.changepoint_lookback
    )
    features["cp_indicator"] = cp_ind
    features["cp_distance"] = cp_dist
    features["cp_magnitude"] = cp_mag

    # ------------------------------------------------------------------
    # 10. HMM regime probabilities
    # ------------------------------------------------------------------
    if HAS_HMM and n > 60:
        daily_ret_clean = np.nan_to_num(daily_ret, nan=0.0)
        hmm_probs = _hmm_regime_features(daily_ret_clean, n_states=3)
        for s in range(hmm_probs.shape[1]):
            features[f"hmm_state_{s}"] = hmm_probs[:, s]

    # ------------------------------------------------------------------
    # 11. Additional momentum features
    # ------------------------------------------------------------------
    # Rate of change (normalized)
    for window in [5, 10, 21]:
        roc = np.full(n, np.nan)
        if n > window:
            roc[window:] = (close[window:] - close[:-window]) / close[:-window]
        features[f"roc_{window}d"] = roc

    # Momentum oscillator: (close - SMA) / std
    for window in [10, 21]:
        sma_w = _sma(close, window)
        std_w = _rolling_std(close, window)
        safe_std_w = np.where(std_w > 0, std_w, 1.0)
        features[f"mom_osc_{window}d"] = (close - sma_w) / safe_std_w

    # Volatility ratio (short/long)
    rvol_5 = features["rvol_5d"]
    rvol_21 = features["rvol_21d"]
    safe_rvol_21 = np.where((rvol_21 > 0) & ~np.isnan(rvol_21), rvol_21, 1.0)
    features["vol_ratio_5_21"] = np.where(
        ~np.isnan(rvol_5), rvol_5 / safe_rvol_21, np.nan
    )

    # Price distance from 52-week high / low (approximation using 252-day)
    if n >= 252:
        rolling_high = pd.Series(close).rolling(252, min_periods=252).max().values
        rolling_low = pd.Series(close).rolling(252, min_periods=252).min().values
        hl_range = rolling_high - rolling_low
        safe_range = np.where(hl_range > 0, hl_range, 1.0)
        features["dist_52w_high"] = (close - rolling_high) / safe_range
        features["dist_52w_low"] = (close - rolling_low) / safe_range
    else:
        features["dist_52w_high"] = np.zeros(n)
        features["dist_52w_low"] = np.zeros(n)

    # ------------------------------------------------------------------
    # Assemble into matrix
    # ------------------------------------------------------------------
    names = list(features.keys())
    mat = np.column_stack([features[k] for k in names])

    return mat, names


# ============================================================================
# PyTorch Model Components
# ============================================================================

if HAS_TORCH:

    class GatedResidualNetwork(nn.Module):
        """Gated Residual Network (GRN) — core building block of TFT.

        GRN(a, c) = LayerNorm(a + GLU(W1 * ELU(W2 * a + W3 * c + b)))

        When no context vector is provided, the W3*c term is omitted.
        """

        def __init__(self, input_dim: int, hidden_dim: int,
                     output_dim: int, dropout: float = 0.1,
                     context_dim: Optional[int] = None):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # for GLU
            self.context_proj = (
                nn.Linear(context_dim, hidden_dim, bias=False)
                if context_dim is not None
                else None
            )
            self.skip = (
                nn.Linear(input_dim, output_dim)
                if input_dim != output_dim
                else nn.Identity()
            )
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor,
                    context: Optional[torch.Tensor] = None) -> torch.Tensor:
            residual = self.skip(x)

            h = self.fc1(x)
            if self.context_proj is not None and context is not None:
                h = h + self.context_proj(context)
            h = F.elu(h)
            h = self.dropout(h)

            gate_input = self.fc2(h)
            # GLU: split into value and gate, apply sigmoid to gate
            value, gate = gate_input.chunk(2, dim=-1)
            h = value * torch.sigmoid(gate)

            return self.layer_norm(residual + h)

    class VariableSelectionNetwork(nn.Module):
        """Variable Selection Network (VSN) — learns which features matter.

        For each of n_features inputs, a GRN produces a weight (via softmax).
        The output is the weighted sum of per-feature GRN-transformed inputs.
        """

        def __init__(self, n_features: int, hidden_dim: int,
                     dropout: float = 0.1):
            super().__init__()
            self.n_features = n_features
            self.hidden_dim = hidden_dim

            # Per-feature GRN: transforms each scalar feature to hidden_dim
            self.feature_grns = nn.ModuleList([
                GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
                for _ in range(n_features)
            ])

            # Flattened input GRN for computing selection weights
            self.weight_grn = GatedResidualNetwork(
                n_features, hidden_dim, n_features, dropout
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, n_features)

            Returns
            -------
            Tensor of shape (batch, seq_len, hidden_dim)
            """
            batch, seq_len, _ = x.shape

            # Variable selection weights: (batch, seq_len, n_features)
            weights = self.softmax(self.weight_grn(x))

            # Transform each feature independently
            transformed = []
            for i in range(self.n_features):
                # (batch, seq_len, 1) -> (batch, seq_len, hidden_dim)
                feat_i = x[:, :, i : i + 1]
                transformed.append(self.feature_grns[i](feat_i))

            # Stack: (batch, seq_len, n_features, hidden_dim)
            transformed = torch.stack(transformed, dim=2)

            # Weight and sum: (batch, seq_len, hidden_dim)
            weights_expanded = weights.unsqueeze(-1)  # (batch, seq_len, n_features, 1)
            output = (transformed * weights_expanded).sum(dim=2)

            return output

    class TemporalCovariateEncoder(nn.Module):
        """Temporal Covariate Encoder (Lim et al., 2021).

        Takes integer-coded temporal features and embeds each via learned
        embeddings, then projects the concatenated embeddings down to
        hidden_dim.

        Temporal features:
        - day_of_week: Embedding(7, 4)
        - month: Embedding(12, 4)
        - day_of_month: Embedding(31, 4)
        - is_expiry_week: Embedding(2, 2)
        - is_monthly_expiry: Embedding(2, 2)
        - quarter: Embedding(4, 2)
        Total temporal embedding dim = 18
        """

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.day_of_week_embed = nn.Embedding(7, 4)
            self.month_embed = nn.Embedding(12, 4)
            self.day_of_month_embed = nn.Embedding(31, 4)
            self.is_expiry_week_embed = nn.Embedding(2, 2)
            self.is_monthly_expiry_embed = nn.Embedding(2, 2)
            self.quarter_embed = nn.Embedding(4, 2)

            self.total_embed_dim = 4 + 4 + 4 + 2 + 2 + 2  # = 18
            self.projection = nn.Linear(self.total_embed_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
            """Embed temporal features.

            Parameters
            ----------
            temporal_features : Tensor of shape (batch, seq_len, 6)
                Integer-coded temporal features:
                [day_of_week(0-6), month(0-11), day_of_month(0-30),
                 is_expiry_week(0-1), is_monthly_expiry(0-1), quarter(0-3)]

            Returns
            -------
            Tensor of shape (batch, seq_len, hidden_dim)
            """
            dow = self.day_of_week_embed(temporal_features[:, :, 0].long())      # (B, S, 4)
            month = self.month_embed(temporal_features[:, :, 1].long())           # (B, S, 4)
            dom = self.day_of_month_embed(temporal_features[:, :, 2].long())      # (B, S, 4)
            expiry_wk = self.is_expiry_week_embed(temporal_features[:, :, 3].long())  # (B, S, 2)
            monthly_exp = self.is_monthly_expiry_embed(temporal_features[:, :, 4].long())  # (B, S, 2)
            quarter = self.quarter_embed(temporal_features[:, :, 5].long())       # (B, S, 2)

            concat = torch.cat([dow, month, dom, expiry_wk, monthly_exp, quarter], dim=-1)  # (B, S, 18)
            projected = self.projection(concat)  # (B, S, hidden_dim)
            return self.layer_norm(projected)

    class StaticCovariateEncoder(nn.Module):
        """Static Covariate Encoder (Lim et al., 2021).

        Takes an asset index (0-3 for NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
        and produces four context vectors:
        - c_s: for variable selection gating
        - c_e: for post-LSTM enrichment
        - c_h: for LSTM hidden state initialization
        - c_c: for LSTM cell state initialization
        """

        def __init__(self, n_categories: int, embed_dim: int, hidden_dim: int,
                     lstm_layers: int = 2):
            super().__init__()
            self.embedding = nn.Embedding(n_categories, embed_dim)
            self.lstm_layers = lstm_layers

            # Four context projections
            self.fc_s = nn.Linear(embed_dim, hidden_dim)  # variable selection
            self.fc_e = nn.Linear(embed_dim, hidden_dim)  # enrichment
            self.fc_h = nn.Linear(embed_dim, hidden_dim)  # LSTM hidden init
            self.fc_c = nn.Linear(embed_dim, hidden_dim)  # LSTM cell init

        def forward(self, asset_id: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute context vectors from asset identity.

            Parameters
            ----------
            asset_id : Tensor of shape (batch,) — integer asset indices (0-3)

            Returns
            -------
            c_s : (batch, hidden_dim) — variable selection context
            c_e : (batch, hidden_dim) — enrichment context
            c_h : (lstm_layers, batch, hidden_dim) — LSTM hidden init
            c_c : (lstm_layers, batch, hidden_dim) — LSTM cell init
            """
            embed = self.embedding(asset_id.long())  # (batch, embed_dim)
            c_s = self.fc_s(embed)   # (batch, hidden_dim)
            c_e = self.fc_e(embed)   # (batch, hidden_dim)

            # Expand for multi-layer LSTM: (lstm_layers, batch, hidden_dim)
            c_h_single = self.fc_h(embed)  # (batch, hidden_dim)
            c_c_single = self.fc_c(embed)  # (batch, hidden_dim)
            c_h = c_h_single.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
            c_c = c_c_single.unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()

            return c_s, c_e, c_h, c_c

    class ALiBi(nn.Module):
        """Attention with Linear Biases (Press et al., 2022).

        Adds position-dependent linear bias to attention scores:
            bias[h, i, j] = -m_h * |i - j|
        where m_h = 2^(-8*h/n_heads) is a geometric sequence of slopes.

        No learnable parameters -- purely geometric positional encoding.
        """

        def __init__(self, n_heads: int):
            super().__init__()
            self.n_heads = n_heads
            # Compute slopes: m_h = 2^(-8*h/n_heads) for h in 1..n_heads
            slopes = torch.tensor([
                2.0 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)
            ])
            self.register_buffer("slopes", slopes)  # (n_heads,)

        def forward(self, seq_len: int) -> torch.Tensor:
            """Compute ALiBi bias matrix.

            Parameters
            ----------
            seq_len : int
                Sequence length.

            Returns
            -------
            bias : Tensor of shape (n_heads, seq_len, seq_len)
                Attention bias to add to QK^T / sqrt(d_k).
            """
            # |i - j| distance matrix
            positions = torch.arange(seq_len, device=self.slopes.device)
            distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # (S, S)
            # bias[h, i, j] = -m_h * |i - j|
            bias = -self.slopes.view(-1, 1, 1) * distance.unsqueeze(0).float()
            return bias  # (n_heads, S, S)

    class SDPAttention(nn.Module):
        """Scaled Dot-Product Attention using torch.nn.functional.scaled_dot_product_attention.

        Automatically uses Flash Attention 2 or memory-efficient attention when
        available on the hardware. Falls back to standard math attention otherwise.

        Replaces nn.MultiheadAttention for the self-attention layer.
        """

        def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                     use_alibi: bool = True):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.dropout_p = dropout

            self.alibi = ALiBi(num_heads) if use_alibi else None

            # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
            self._has_sdpa = hasattr(F, "scaled_dot_product_attention")

        def forward(self, x: torch.Tensor, is_causal: bool = True,
                    need_weights: bool = False
                    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Self-attention with optional ALiBi bias and causal masking.

            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, embed_dim)
            is_causal : bool
                If True, apply causal masking.
            need_weights : bool
                If True, return attention weights (disables Flash path).

            Returns
            -------
            output : Tensor of shape (batch, seq_len, embed_dim)
            attn_weights : Optional Tensor of shape (batch, n_heads, seq_len, seq_len)
            """
            B, S, E = x.shape
            H = self.num_heads
            D = self.head_dim

            Q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)
            K = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
            V = self.v_proj(x).view(B, S, H, D).transpose(1, 2)

            # Build attention bias from ALiBi
            attn_bias = None
            if self.alibi is not None:
                attn_bias = self.alibi(S)  # (H, S, S)

            # Add causal mask to bias
            if is_causal and attn_bias is not None:
                causal = torch.triu(
                    torch.full((S, S), float("-inf"), device=x.device),
                    diagonal=1,
                )  # (S, S)
                attn_bias = attn_bias + causal.unsqueeze(0)  # (H, S, S)

            dropout_p = self.dropout_p if self.training else 0.0

            if self._has_sdpa and not need_weights:
                # Use SDPA (Flash Attention 2 / memory-efficient backend)
                # When we have attn_bias, we must NOT use is_causal=True
                # since causality is already in the bias
                if attn_bias is not None:
                    # Expand bias to (B, H, S, S)
                    attn_mask = attn_bias.unsqueeze(0).expand(B, -1, -1, -1)
                    out = F.scaled_dot_product_attention(
                        Q, K, V,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=False,  # causal already in mask
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        Q, K, V,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )
                attn_weights = None
            else:
                # Fallback: manual attention computation
                scale = math.sqrt(D)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, S, S)
                if attn_bias is not None:
                    scores = scores + attn_bias.unsqueeze(0)
                elif is_causal:
                    causal = torch.triu(
                        torch.full((S, S), float("-inf"), device=x.device),
                        diagonal=1,
                    )
                    scores = scores + causal
                attn_weights = torch.softmax(scores, dim=-1)
                if dropout_p > 0:
                    attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
                out = torch.matmul(attn_weights, V)

            # Reshape back
            out = out.transpose(1, 2).contiguous().view(B, S, E)
            out = self.out_proj(out)
            return out, attn_weights

    class MomentumTransformerModel(nn.Module):
        """Full Momentum Transformer architecture.

        Components:
        1. Variable Selection Network (feature importance)
        2. Optional Temporal Covariate Encoder (day_of_week, month, etc.)
        3. Optional Static Covariate Encoder (asset identity)
        4. 2-layer LSTM encoder (with optional static-initialized hidden state)
        5. Multi-head self-attention over encoded sequence (SDPA + ALiBi)
        6. Output head: Linear -> Tanh -> position in [-1, 1]
        7. Optional multi-horizon auxiliary heads (T+2, T+3, T+5)
        """

        def __init__(self, n_features: int, hidden_dim: int = 64,
                     lstm_layers: int = 2, n_heads: int = 4,
                     dropout: float = 0.1, use_sdpa: bool = True,
                     use_alibi: bool = True, multi_horizon: bool = False,
                     aux_horizons: tuple[int, ...] = (2, 3, 5),
                     use_temporal_covariates: bool = False,
                     use_static_covariates: bool = False,
                     n_static_categories: int = 4,
                     static_embed_dim: int = 8):
            super().__init__()
            self.n_features = n_features
            self.hidden_dim = hidden_dim
            self.multi_horizon = multi_horizon
            self.aux_horizons = aux_horizons
            self.use_sdpa = use_sdpa
            self.use_temporal_covariates = use_temporal_covariates
            self.use_static_covariates = use_static_covariates
            self.lstm_layers = lstm_layers

            # 1. Variable Selection Network
            self.vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout)

            # 2. Temporal Covariate Encoder (optional)
            self.temporal_encoder = None
            if use_temporal_covariates:
                self.temporal_encoder = TemporalCovariateEncoder(hidden_dim)

            # 3. Static Covariate Encoder (optional)
            self.static_encoder = None
            if use_static_covariates:
                self.static_encoder = StaticCovariateEncoder(
                    n_categories=n_static_categories,
                    embed_dim=static_embed_dim,
                    hidden_dim=hidden_dim,
                    lstm_layers=lstm_layers,
                )
                # Gating for static context on VSN output
                self.static_vsn_gate = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                # Enrichment layer for post-LSTM features
                self.static_enrichment_grn = GatedResidualNetwork(
                    hidden_dim, hidden_dim, hidden_dim, dropout,
                    context_dim=hidden_dim,
                )

            # Determine LSTM input size: hidden_dim from VSN + hidden_dim from temporal (if enabled)
            lstm_input_dim = hidden_dim
            if use_temporal_covariates:
                lstm_input_dim = hidden_dim * 2
            # Projection to hidden_dim if temporal covariates change the input size
            self.lstm_input_proj = None
            if lstm_input_dim != hidden_dim:
                self.lstm_input_proj = nn.Linear(lstm_input_dim, hidden_dim)

            # 4. LSTM encoder
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.lstm_norm = nn.LayerNorm(hidden_dim)

            # 5. Multi-head attention (SDPA with ALiBi or legacy nn.MHA)
            if use_sdpa:
                self.self_attn = SDPAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    use_alibi=use_alibi,
                )
                self._legacy_attn = False
            else:
                self.self_attn = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self._legacy_attn = True
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_grn = GatedResidualNetwork(
                hidden_dim, hidden_dim, hidden_dim, dropout
            )

            # 6. Output head — predict position signal for the last timestep
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Tanh(),
            )

            # Classification head for pre-training (next-day return sign)
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

            # 7. Multi-horizon auxiliary heads (share LSTM encoder)
            self.aux_heads = nn.ModuleDict()
            if multi_horizon:
                for h in aux_horizons:
                    self.aux_heads[f"T+{h}"] = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                        nn.Tanh(),
                    )

        def _encode(self, x: torch.Tensor,
                    temporal_features: Optional[torch.Tensor] = None,
                    asset_id: Optional[torch.Tensor] = None,
                    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            """Shared encoder: VSN -> LSTM -> Attention.

            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, n_features)
            temporal_features : Tensor of shape (batch, seq_len, 6), optional
                Integer-coded temporal covariates.
            asset_id : Tensor of shape (batch,), optional
                Integer asset index for static covariates.

            Returns (last_hidden, attn_out, attn_weights).
            """
            # 1. Variable selection
            vsn_out = self.vsn(x)  # (batch, seq_len, hidden_dim)

            # 1b. Static covariate: gate VSN output with c_s
            c_s, c_e, c_h, c_c = None, None, None, None
            if self.static_encoder is not None and asset_id is not None:
                c_s, c_e, c_h, c_c = self.static_encoder(asset_id)
                # Gate VSN output: vsn_out * sigmoid(c_s)
                gate = self.static_vsn_gate(c_s)  # (batch, hidden_dim)
                vsn_out = vsn_out * gate.unsqueeze(1)  # broadcast over seq_len

            # 2. Temporal covariate embedding (optional)
            lstm_input = vsn_out
            if self.temporal_encoder is not None and temporal_features is not None:
                temporal_embed = self.temporal_encoder(temporal_features)  # (batch, seq_len, hidden_dim)
                lstm_input = torch.cat([vsn_out, temporal_embed], dim=-1)  # (batch, seq_len, 2*hidden_dim)

            # Project back to hidden_dim if needed
            if self.lstm_input_proj is not None:
                lstm_input = self.lstm_input_proj(lstm_input)

            # 3. LSTM encoding (with optional static-initialized hidden state)
            if c_h is not None and c_c is not None:
                lstm_out, _ = self.lstm(lstm_input, (c_h, c_c))
            else:
                lstm_out, _ = self.lstm(lstm_input)
            lstm_out = self.lstm_norm(lstm_out)

            # 3b. Static enrichment: enrich post-LSTM features with c_e
            if self.static_encoder is not None and c_e is not None:
                lstm_out = self.static_enrichment_grn(
                    lstm_out, context=c_e.unsqueeze(1).expand_as(lstm_out)
                )

            # 4. Multi-head self-attention (causal mask to prevent look-ahead)
            if self._legacy_attn:
                seq_len = lstm_out.size(1)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=lstm_out.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_out, attn_weights = self.self_attn(
                    lstm_out, lstm_out, lstm_out,
                    attn_mask=causal_mask,
                    need_weights=True,
                )
            else:
                attn_out, attn_weights = self.self_attn(
                    lstm_out, is_causal=True, need_weights=False,
                )

            attn_out = self.attn_norm(lstm_out + attn_out)
            attn_out = self.attn_grn(attn_out)

            last_hidden = attn_out[:, -1, :]  # (batch, hidden_dim)
            return last_hidden, attn_out, attn_weights

        def forward(self, x: torch.Tensor,
                    temporal_features: Optional[torch.Tensor] = None,
                    asset_id: Optional[torch.Tensor] = None,
                    return_attention: bool = False
                    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, n_features)
            temporal_features : Tensor of shape (batch, seq_len, 6), optional
                Integer-coded temporal covariates. Only used if
                use_temporal_covariates=True.
            asset_id : Tensor of shape (batch,), optional
                Integer asset index (0-3). Only used if use_static_covariates=True.
            return_attention : bool
                If True, return attention weights as the last element.

            Returns
            -------
            If multi_horizon=False:
                position : Tensor of shape (batch, 1) -- values in [-1, 1]
            If multi_horizon=True:
                (position_T1, position_T2, position_T3, position_T5) each (batch, 1)
            If return_attention=True, attn_weights is appended as the last element.
            """
            last_hidden, attn_out, attn_weights = self._encode(
                x, temporal_features=temporal_features, asset_id=asset_id,
            )

            # Main T+1 position signal
            position = self.output_head(last_hidden)  # (batch, 1)

            if self.multi_horizon:
                aux_preds = {}
                for key, head in self.aux_heads.items():
                    aux_preds[key] = head(last_hidden)  # (batch, 1)
                # Return (T+1, T+2, T+3, T+5, ...) in order
                results = [position]
                for h in self.aux_horizons:
                    results.append(aux_preds[f"T+{h}"])
                if return_attention:
                    results.append(attn_weights)
                return tuple(results)

            if return_attention:
                return position, attn_weights
            return position

        def classify(self, x: torch.Tensor,
                     temporal_features: Optional[torch.Tensor] = None,
                     asset_id: Optional[torch.Tensor] = None,
                     ) -> torch.Tensor:
            """Pre-training classification: predict next-day return sign.

            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, n_features)
            temporal_features : optional temporal covariates
            asset_id : optional asset identity

            Returns logits (batch, 1) -- apply sigmoid for probability.
            """
            last_hidden, _, _ = self._encode(
                x, temporal_features=temporal_features, asset_id=asset_id,
            )
            return self.cls_head(last_hidden)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def sharpe_loss(positions: torch.Tensor,
                    returns: torch.Tensor,
                    annualize: float = math.sqrt(252)) -> torch.Tensor:
        """Differentiable Sharpe ratio loss (negative Sharpe).

        Parameters
        ----------
        positions : Tensor (batch,) — position signals in [-1, 1]
        returns : Tensor (batch,) — next-day returns
        annualize : float — annualization factor

        Returns
        -------
        Scalar tensor: -Sharpe (minimizing this maximizes Sharpe).
        """
        strategy_returns = positions * returns
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std(correction=1)

        # Avoid division by zero — return a small differentiable penalty
        # that encourages the model to take positions
        if std_ret < 1e-8:
            return -torch.abs(positions).mean() * 0.01

        sharpe = mean_ret / std_ret * annualize
        return -sharpe

    def multi_horizon_loss(
        predictions: tuple[torch.Tensor, ...],
        returns_dict: dict[int, torch.Tensor],
        aux_weights: tuple[float, ...] = (0.3, 0.2, 0.1),
        aux_horizons: tuple[int, ...] = (2, 3, 5),
    ) -> torch.Tensor:
        """Multi-horizon Sharpe loss.

        L = L_T+1 + w2*L_T+2 + w3*L_T+3 + w5*L_T+5

        Parameters
        ----------
        predictions : tuple of Tensors
            (pos_T1, pos_T2, pos_T3, pos_T5) each (batch, 1)
        returns_dict : dict[int, Tensor]
            Mapping from horizon (1, 2, 3, 5) to return tensors (batch,).
        aux_weights : tuple[float]
            Weights for auxiliary horizons.
        aux_horizons : tuple[int]
            Horizon offsets.

        Returns
        -------
        Scalar loss tensor.
        """
        # Main T+1 loss
        pos_t1 = predictions[0].squeeze(-1)
        ret_t1 = returns_dict[1]
        total_loss = sharpe_loss(pos_t1, ret_t1)

        # Auxiliary losses
        for i, (h, w) in enumerate(zip(aux_horizons, aux_weights)):
            if h not in returns_dict:
                continue
            pos_h = predictions[i + 1].squeeze(-1)
            ret_h = returns_dict[h]
            # Only compute on valid (non-NaN) samples
            valid = ~torch.isnan(ret_h)
            if valid.sum() < 2:
                continue
            aux_loss = sharpe_loss(pos_h[valid], ret_h[valid])
            total_loss = total_loss + w * aux_loss

        return total_loss


# ============================================================================
# sklearn Fallback Model
# ============================================================================

class SklearnFallbackModel:
    """GradientBoosting fallback when PyTorch is unavailable.

    Uses the same feature set but with a flat (non-sequential) model.
    Walk-forward training is still applied.
    """

    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on feature matrix X and target y (next-day returns)."""
        mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
        if mask.sum() < 30:
            logger.warning("Insufficient clean samples (%d) for sklearn fit", mask.sum())
            return
        self.model.fit(X[mask], y[mask])
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict position signal, clipped to [-1, 1]."""
        if not self._fitted:
            return np.zeros(len(X))
        mask = ~np.any(np.isnan(X), axis=1)
        preds = np.zeros(len(X))
        if mask.sum() > 0:
            raw = self.model.predict(X[mask])
            preds[mask] = np.clip(raw, -1.0, 1.0)
        return preds


# ============================================================================
# Data Preparation Utilities
# ============================================================================

def _prepare_sequences(features: np.ndarray, targets: np.ndarray,
                       seq_len: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sequences for the model.

    Parameters
    ----------
    features : (n, n_features)
    targets : (n,) — next-day returns aligned with features
    seq_len : int — lookback window

    Returns
    -------
    X : (n_samples, seq_len, n_features)
    y : (n_samples,)
    """
    n = len(features)
    if n < seq_len + 1:
        return np.empty((0, seq_len, features.shape[1])), np.empty(0)

    X_list = []
    y_list = []
    for i in range(seq_len, n):
        window = features[i - seq_len : i]
        # Skip windows with NaN
        if np.any(np.isnan(window)):
            continue
        if np.isnan(targets[i]):
            continue
        X_list.append(window)
        y_list.append(targets[i])

    if not X_list:
        return np.empty((0, seq_len, features.shape[1])), np.empty(0)

    return np.array(X_list), np.array(y_list)


def _normalize_features(features: np.ndarray,
                        train_mask: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features using training-set statistics only.

    Returns (normalized_features, means, stds).
    NaN values are preserved.
    """
    train_feats = features[train_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(train_feats, axis=0)
        stds = np.nanstd(train_feats, axis=0, ddof=1)

    # Prevent division by zero
    stds = np.where(stds > 1e-10, stds, 1.0)

    normalized = (features - means) / stds
    return normalized, means, stds


# ============================================================================
# Training Loop (PyTorch)
# ============================================================================

if HAS_TORCH:

    def _train_pretrain(model: MomentumTransformerModel,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        cfg: MomentumTFMConfig) -> None:
        """Pre-train with binary cross-entropy on next-day return sign."""
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # Labels: 1 if next-day return > 0, else 0
        y_train_cls = (y_train > 0).astype(np.float32)
        y_val_cls = (y_val > 0).astype(np.float32)

        X_t = torch.tensor(X_train, dtype=torch.float32, device=_DEVICE)
        y_t = torch.tensor(y_train_cls, dtype=torch.float32, device=_DEVICE).unsqueeze(1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(cfg.pretrain_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                logits = model.classify(batch_X)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch_X)

            # Validation accuracy
            if len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32, device=_DEVICE)
                    y_v = torch.tensor(y_val_cls, dtype=torch.float32, device=_DEVICE)
                    logits_v = model.classify(X_v).squeeze()
                    preds_v = (torch.sigmoid(logits_v) > 0.5).float()
                    val_acc = (preds_v == y_v).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        logger.debug("Pre-training early stop at epoch %d (val_acc=%.4f)",
                                     epoch, best_val_acc)
                        break

    def _train_finetune(model: MomentumTransformerModel,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        cfg: MomentumTFMConfig) -> None:
        """Fine-tune with Sharpe ratio loss."""
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr * 0.1,  # lower LR for fine-tuning
            weight_decay=cfg.weight_decay,
        )

        X_t = torch.tensor(X_train, dtype=torch.float32, device=_DEVICE)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=_DEVICE)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(cfg.batch_size * 4, len(X_train)),
                            shuffle=True)

        best_val_sharpe = -np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(cfg.finetune_epochs):
            model.train()
            for batch_X, batch_y in loader:
                positions = model(batch_X).squeeze()
                loss = sharpe_loss(positions, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation Sharpe
            if len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    X_v = torch.tensor(X_val, dtype=torch.float32, device=_DEVICE)
                    y_v = torch.tensor(y_val, dtype=torch.float32, device=_DEVICE)
                    pos_v = model(X_v).squeeze()
                    strat_ret = pos_v * y_v
                    mean_r = strat_ret.mean().item()
                    std_r = strat_ret.std(correction=1).item()
                    val_sharpe = (mean_r / max(std_r, 1e-8)) * math.sqrt(252)

                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        logger.debug("Fine-tune early stop at epoch %d (val_sharpe=%.4f)",
                                     epoch, best_val_sharpe)
                        break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(_DEVICE)


# ============================================================================
# Walk-Forward Engine
# ============================================================================

def _walk_forward_torch(features: np.ndarray, returns: np.ndarray,
                        cfg: MomentumTFMConfig,
                        start_idx: int = 0,
                        nan_mask: Optional[np.ndarray] = None,
                        ) -> np.ndarray:
    """Walk-forward train/predict with the Momentum Transformer.

    Parameters
    ----------
    features : (n, n_features) — imputed features (forward-filled + zero-filled)
    returns : (n,) — next-day returns
    cfg : configuration
    start_idx : int — first index for walk-forward start
    nan_mask : (n, n_features) bool — True where original data was NaN
        (pre-imputation).  Used for per-fold adaptive feature selection:
        each fold only trains on features with >30% real data in the
        training window.

    Returns
    -------
    positions : (n,) — out-of-sample position signals
    """
    n = len(features)
    positions = np.full(n, np.nan)

    train_size = cfg.train_window
    test_size = cfg.test_window
    step = cfg.step_size
    seq_len = cfg.sequence_length

    fold_idx = 0
    start = start_idx

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        # --- Per-fold adaptive feature selection ---
        # Only use features with >30% real data in the training window
        if nan_mask is not None:
            train_nan = nan_mask[start:train_end]
            feat_coverage = 1.0 - train_nan.mean(axis=0)  # fraction non-NaN
            fold_feat_mask = feat_coverage > 0.3
            n_fold_feats = fold_feat_mask.sum()
            if n_fold_feats < 5:
                logger.warning("Fold %d: only %d features with >30%% coverage, skipping",
                               fold_idx, n_fold_feats)
                start += step
                fold_idx += 1
                continue
            fold_features = features[:, fold_feat_mask]
        else:
            fold_features = features
            n_fold_feats = fold_features.shape[1]

        logger.debug("Walk-forward fold %d: train=[%d:%d], test=[%d:%d], feats=%d",
                     fold_idx, start, train_end, train_end, test_end, n_fold_feats)

        # Normalize using training set statistics
        train_mask_arr = np.zeros(n, dtype=bool)
        train_mask_arr[start:train_end] = True
        norm_feats, means, stds = _normalize_features(fold_features, train_mask_arr)

        # Prepare sequences
        X_train, y_train = _prepare_sequences(
            norm_feats[start:train_end],
            returns[start:train_end],
            seq_len,
        )
        X_test, y_test = _prepare_sequences(
            norm_feats[max(0, train_end - seq_len) : test_end],
            returns[max(0, train_end - seq_len) : test_end],
            seq_len,
        )

        if len(X_train) < 10:
            logger.warning("Fold %d: insufficient training samples (%d), skipping",
                           fold_idx, len(X_train))
            start += step
            fold_idx += 1
            continue

        # Split training into train/val (last 20% for validation)
        val_split = max(1, int(len(X_train) * 0.2))
        X_tr, X_vl = X_train[:-val_split], X_train[-val_split:]
        y_tr, y_vl = y_train[:-val_split], y_train[-val_split:]

        # Initialize model — n_features adapts per fold
        model = MomentumTransformerModel(
            n_features=int(n_fold_feats),
            hidden_dim=cfg.hidden_dim,
            lstm_layers=cfg.lstm_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            use_sdpa=cfg.use_sdpa,
            use_alibi=cfg.use_alibi,
            multi_horizon=cfg.multi_horizon,
            aux_horizons=cfg.aux_horizons,
        ).to(_DEVICE)

        # Pre-train with classification loss
        _train_pretrain(model, X_tr, y_tr, X_vl, y_vl, cfg)

        # Fine-tune with Sharpe loss
        _train_finetune(model, X_tr, y_tr, X_vl, y_vl, cfg)

        # Predict on test set
        if len(X_test) > 0:
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_test, dtype=torch.float32, device=_DEVICE)
                preds = model(X_t).squeeze().cpu().numpy()
                if preds.ndim == 0:
                    preds = np.array([preds.item()])

            # Map predictions back to original indices
            pred_start = train_end
            pred_end = min(pred_start + len(preds), n)
            n_preds = pred_end - pred_start
            positions[pred_start:pred_end] = preds[:n_preds]

        start += step
        fold_idx += 1

        # Clean up GPU memory between folds
        del model
        if HAS_TORCH:
            torch.cuda.empty_cache()

    return positions


def _walk_forward_sklearn(features: np.ndarray, returns: np.ndarray,
                          cfg: MomentumTFMConfig,
                          start_idx: int = 0,
                          ) -> np.ndarray:
    """Walk-forward with sklearn GradientBoosting fallback."""
    n = len(features)
    positions = np.full(n, np.nan)

    train_size = cfg.train_window
    test_size = cfg.test_window
    step = cfg.step_size

    start = start_idx
    fold_idx = 0

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        # Normalize
        train_mask = np.zeros(n, dtype=bool)
        train_mask[start:train_end] = True
        norm_feats, _, _ = _normalize_features(features, train_mask)

        X_train = norm_feats[start:train_end]
        y_train = returns[start:train_end]
        X_test = norm_feats[train_end:test_end]

        model = SklearnFallbackModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        positions[train_end:test_end] = preds

        start += step
        fold_idx += 1

    return positions


# ============================================================================
# Data Loading — Kite spot 1-min → daily close (488+ days)
# ============================================================================

from quantlaxmi.data._paths import KITE_1MIN_DIR
_KITE_DIR = str(KITE_1MIN_DIR)

_SYMBOL_TO_KITE = {
    "NIFTY 50": "NIFTY_SPOT",
    "NIFTY50": "NIFTY_SPOT",
    "NIFTY BANK": "BANKNIFTY_SPOT",
    "NIFTY FINANCIAL SERVICES": "FINNIFTY_SPOT",
    "NIFTY MIDCAP SELECT": "MIDCPNIFTY_SPOT",
}


def _load_daily_close(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    store: MarketDataStore,
) -> pd.DataFrame:
    """Load daily close prices — tries Kite spot first (2 yr), falls back to DuckDB.

    Returns DataFrame with columns [date, close].
    """
    import pyarrow.parquet as pq
    from pathlib import Path

    kite_name = _SYMBOL_TO_KITE.get(symbol.upper().strip())
    if kite_name is None:
        # Try case-insensitive partial match
        for k, v in _SYMBOL_TO_KITE.items():
            if k.lower() in symbol.lower() or symbol.lower() in k.lower():
                kite_name = v
                break

    kite_dir = Path(_KITE_DIR) / kite_name if kite_name else None

    if kite_dir is not None and kite_dir.exists():
        # Load daily close from Kite 1-min spot (last bar of each day)
        records = []
        for d_dir in sorted(kite_dir.iterdir()):
            if not d_dir.is_dir() or not d_dir.name.startswith("date="):
                continue
            d_str = d_dir.name[5:]
            if d_str < start.strftime("%Y-%m-%d") or d_str > end.strftime("%Y-%m-%d"):
                continue
            pfiles = list(d_dir.glob("*.parquet"))
            if not pfiles:
                continue
            try:
                day_df = pq.read_table(pfiles[0]).to_pandas()
                if day_df.empty:
                    continue
                day_close = float(pd.to_numeric(day_df["close"], errors="coerce").iloc[-1])
                if np.isnan(day_close) or day_close <= 0:
                    continue
                records.append({"date": d_str, "close": day_close})
            except Exception as e:
                logger.debug("Kite spot close read failed for %s: %s", d_str, e)
                continue

        if records:
            df = pd.DataFrame(records)
            logger.info(
                "Loaded %d daily closes from Kite spot (%s, %s to %s)",
                len(df), kite_name, df["date"].iloc[0], df["date"].iloc[-1],
            )
            return df

    # Fallback: DuckDB nse_index_close
    logger.info("Falling back to nse_index_close (DuckDB)")
    df = store.sql(
        'SELECT date, "Closing Index Value" as close '
        'FROM nse_index_close '
        'WHERE LOWER("Index Name") = LOWER(?) AND date BETWEEN ? AND ? '
        'ORDER BY date',
        [symbol, start.isoformat(), end.isoformat()],
    )
    return df


# ============================================================================
# Backtest Entry Point
# ============================================================================

def run_backtest(
    symbol: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    store: MarketDataStore,
    cfg: Optional[MomentumTFMConfig] = None,
    use_torch: Optional[bool] = None,
    external_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the Momentum Transformer backtest on a given symbol.

    Parameters
    ----------
    symbol : str
        Index name (e.g. "NIFTY 50", "NIFTY BANK").
    start_date, end_date : str or Timestamp
        Backtest date range (inclusive).
    store : MarketDataStore
        Initialized data store with ``nse_index_close`` view.
    cfg : MomentumTFMConfig, optional
        Hyperparameters. If None, uses defaults.
    use_torch : bool, optional
        Force PyTorch (True) or sklearn (False). If None, auto-detects.
    external_features : pd.DataFrame, optional
        Pre-built feature matrix (e.g. from MegaFeatureBuilder).  Indexed
        by date (datetime).  When provided, these features are used INSTEAD
        of the built-in ``build_features()`` output, giving the model access
        to 150+ cross-asset signals.

    Returns
    -------
    pd.DataFrame
        Columns: [date, position, predicted_return, actual_return,
                  strategy_return, cumulative_pnl]
        Indexed by integer; ``date`` column is pd.Timestamp.

    Raises
    ------
    ValueError
        If no data is available for the requested symbol/date range.
    """
    if cfg is None:
        cfg = MomentumTFMConfig()
    if use_torch is None:
        use_torch = HAS_TORCH

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # ------------------------------------------------------------------
    # 1. Load data — prefer Kite spot 1-min (488+ days) over DuckDB (125 days)
    # ------------------------------------------------------------------
    logger.info("Loading data for %s [%s, %s]", symbol, start.date(), end.date())

    df = _load_daily_close(symbol, start, end, store)

    if df.empty:
        raise ValueError(
            f"No data found for symbol={symbol!r} between {start_date} and {end_date}. "
            f"Check Kite spot data or nse_index_close view."
        )

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    if len(df) < cfg.sequence_length + cfg.train_window + cfg.test_window:
        raise ValueError(
            f"Insufficient data: got {len(df)} rows, need at least "
            f"{cfg.sequence_length + cfg.train_window + cfg.test_window} "
            f"(seq_len + train_window + test_window)."
        )

    close = df["close"].values.astype(np.float64)
    dates = df["date"].values

    logger.info("Loaded %d daily observations", len(close))

    # ------------------------------------------------------------------
    # 2. Build features
    # ------------------------------------------------------------------
    if external_features is not None:
        # Align external features to our date index (normalize to date-only)
        ext = external_features.copy()
        ext.index = pd.to_datetime(ext.index).normalize()
        our_dates = pd.to_datetime(dates).normalize()
        aligned = ext.reindex(our_dates)
        feat_matrix = aligned.values.astype(np.float64)
        feat_names = list(aligned.columns)
        n_matched = aligned.notna().any(axis=1).sum()
        logger.info(
            "Using %d external features (%d/%d dates have data)",
            feat_matrix.shape[1], n_matched, len(our_dates),
        )
    else:
        feat_matrix, feat_names = build_features(close, volume=None, cfg=cfg)

    # Drop features that are >80% NaN (uninformative)
    nan_frac = np.isnan(feat_matrix).mean(axis=0)
    keep_mask = nan_frac < 0.8
    if not np.all(keep_mask):
        n_dropped = (~keep_mask).sum()
        feat_matrix = feat_matrix[:, keep_mask]
        feat_names = [n for n, k in zip(feat_names, keep_mask) if k]
        logger.info("Dropped %d features with >80%% NaN", n_dropped)

    # Save pre-imputation NaN mask for per-fold adaptive feature selection.
    # Each walk-forward fold will only use features that have >30% real data
    # in its training window — early folds get fewer but cleaner features.
    pre_impute_nan = np.isnan(feat_matrix).copy()

    # Forward-fill then zero-fill remaining NaN (post z-scoring NaN=0 means
    # "at the training-set mean", which is the least-informative imputation).
    for col_idx in range(feat_matrix.shape[1]):
        col = feat_matrix[:, col_idx]
        # Forward-fill
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[mask] = col[idx[mask]]
        # Zero-fill anything still NaN (leading NaNs before first valid)
        feat_matrix[:, col_idx] = np.nan_to_num(col, nan=0.0)

    n_features = feat_matrix.shape[1]
    cfg.n_features = n_features

    logger.info("Final feature matrix: %d features x %d rows (%.1f%% non-NaN before fill)",
                n_features, feat_matrix.shape[0], (1.0 - nan_frac[keep_mask].mean()) * 100)

    # ------------------------------------------------------------------
    # 3. Compute forward returns (target)
    # ------------------------------------------------------------------
    n = len(close)
    log_close = np.log(close)
    next_day_return = np.full(n, np.nan)
    next_day_return[:-1] = log_close[1:] - log_close[:-1]

    # ------------------------------------------------------------------
    # 4. Walk-forward evaluation
    # ------------------------------------------------------------------
    if use_torch:
        logger.info("Running walk-forward with PyTorch Momentum Transformer on %s", _DEVICE)
        positions = _walk_forward_torch(feat_matrix, next_day_return, cfg,
                                        nan_mask=pre_impute_nan)
    else:
        logger.info("Running walk-forward with sklearn GradientBoosting fallback")
        positions = _walk_forward_sklearn(feat_matrix, next_day_return, cfg)

    # ------------------------------------------------------------------
    # 4b. Position sizing: clip, vol-scale, smooth, and cap turnover
    # ------------------------------------------------------------------
    # Clip raw positions to max allocation
    positions = np.clip(positions, -cfg.max_position, cfg.max_position)

    # Vol-scale positions: reduce size for high-vol instruments (BANKNIFTY ~1.5x NIFTY)
    daily_ret_for_vol = np.full(n, np.nan)
    daily_ret_for_vol[1:] = np.diff(log_close)
    trailing_vol = pd.Series(daily_ret_for_vol).rolling(20, min_periods=10).std(ddof=1).values
    baseline_vol = 0.01  # ~1% daily vol (NIFTY baseline)
    vol_scale = np.where(
        (~np.isnan(trailing_vol)) & (trailing_vol > 0),
        np.clip(baseline_vol / trailing_vol, 0.3, 2.0),
        1.0,
    )
    positions = positions * vol_scale
    positions = np.clip(positions, -cfg.max_position, cfg.max_position)

    # Smooth positions to reduce turnover
    smoothed = np.full_like(positions, np.nan)
    prev_pos = 0.0
    for i in range(n):
        if np.isnan(positions[i]):
            smoothed[i] = np.nan
            continue
        raw = positions[i]
        new_pos = (1.0 - cfg.position_smooth) * prev_pos + cfg.position_smooth * raw
        # Cap daily turnover
        delta = new_pos - prev_pos
        if abs(delta) > cfg.max_daily_turnover:
            new_pos = prev_pos + np.sign(delta) * cfg.max_daily_turnover
        # Re-clip after smoothing/capping
        new_pos = np.clip(new_pos, -cfg.max_position, cfg.max_position)
        smoothed[i] = new_pos
        prev_pos = new_pos

    positions = smoothed

    # ------------------------------------------------------------------
    # 5. Compute backtest results
    # ------------------------------------------------------------------
    # Actual returns on each day (close-to-close)
    actual_return = np.full(n, np.nan)
    actual_return[1:] = log_close[1:] - log_close[:-1]

    # Strategy return: position[t] * actual_return[t+1]
    # Position on day t is based on data up to day t; earns return on day t+1.
    # This ensures T+1 lag (no look-ahead).
    strategy_return = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isnan(positions[i]) and not np.isnan(actual_return[i + 1]):
            # Transaction costs on position changes
            cost = 0.0
            if i > 0 and not np.isnan(positions[i - 1]):
                turnover = abs(positions[i] - positions[i - 1])
            else:
                turnover = abs(positions[i])
            cost = turnover * cfg.cost_bps / 10_000.0
            strategy_return[i + 1] = positions[i] * actual_return[i + 1] - cost

    # Cumulative PnL — geometric compounding (convert log returns to simple)
    simple_strat = np.expm1(np.nan_to_num(strategy_return, nan=0.0))
    cumulative_pnl = np.cumprod(1.0 + simple_strat) - 1.0

    # ------------------------------------------------------------------
    # 6. Assemble output DataFrame
    # ------------------------------------------------------------------
    result = pd.DataFrame({
        "date": dates,
        "position": positions,
        "predicted_return": next_day_return,
        "actual_return": actual_return,
        "strategy_return": strategy_return,
        "cumulative_pnl": cumulative_pnl,
    })

    # ------------------------------------------------------------------
    # 7. Summary statistics
    # ------------------------------------------------------------------
    valid_strat = result["strategy_return"].dropna()
    if len(valid_strat) > 1:
        # Sharpe on simple returns for interpretability
        simple_valid = np.expm1(valid_strat.values)
        ann_sharpe = (
            (np.mean(simple_valid) / np.std(simple_valid, ddof=1)) * math.sqrt(252)
        )
        total_ret = cumulative_pnl[-1]
        max_dd = _max_drawdown_equity(simple_strat)
        n_trades = (np.diff(np.nan_to_num(positions, nan=0.0)) != 0).sum()

        logger.info(
            "Backtest complete: Sharpe=%.3f, TotalReturn=%.2f%%, MaxDD=%.2f%%, "
            "Trades=%d, OOS_days=%d",
            ann_sharpe, total_ret * 100, max_dd * 100, n_trades, len(valid_strat),
        )

    return result


def _max_drawdown_equity(simple_returns: np.ndarray) -> float:
    """Compute maximum drawdown from an array of simple returns.

    Builds the equity curve via geometric compounding and computes
    drawdown as a fraction of peak equity, yielding a value in [0, 1].
    """
    equity = np.cumprod(1.0 + simple_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = np.where(running_max > 0, (running_max - equity) / running_max, 0.0)
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


# ============================================================================
# Convenience: standalone run
# ============================================================================

_ALL_INDICES = ["NIFTY 50", "NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY MIDCAP SELECT"]


def main():
    """CLI entry point — runs backtest across all indices (or a single one with --symbol)."""
    import argparse

    parser = argparse.ArgumentParser(description="Momentum Transformer Backtest")
    parser.add_argument("--model", choices=["tft", "xtrend", "xtrend-rl"], default="tft",
                        help="Model architecture: tft (default), xtrend (X-Trend), or xtrend-rl (X-Trend + RL)")
    parser.add_argument("--symbol", default=None,
                        help="Single index name (default: run all 4 indices)")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-06", help="End date (YYYY-MM-DD)")
    parser.add_argument("--loss-mode", choices=["sharpe", "joint_mle"], default="sharpe",
                        help="X-Trend loss mode: sharpe or joint_mle (MLE + Sharpe)")
    parser.add_argument("--no-torch", action="store_true", help="Force sklearn fallback")
    parser.add_argument("--mega-features", action="store_true",
                        help="Use MegaFeatureBuilder (all data sources)")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--test-window", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = MomentumTFMConfig(
        hidden_dim=args.hidden_dim,
        sequence_length=args.seq_len,
        train_window=args.train_window,
        test_window=args.test_window,
    )

    symbols = [args.symbol] if args.symbol else _ALL_INDICES
    store = MarketDataStore()
    all_results: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # X-Trend + RL path: full integrated pipeline
    # ------------------------------------------------------------------
    if args.model == "xtrend-rl":
        raise NotImplementedError("xtrend-rl removed. Use 'xtrend'.")

    # ------------------------------------------------------------------
    # X-Trend path: multi-asset joint training
    # ------------------------------------------------------------------
    if args.model == "xtrend":
        from quantlaxmi.models.ml.tft.x_trend import run_xtrend_backtest, XTrendConfig

        print(f"\n{'='*60}")
        print(f"X-Trend: Multi-asset joint backtest")
        print(f"{'='*60}")

        # Build a multi-asset price DataFrame from Kite spot data
        price_records: dict[str, list] = {"date": []}
        _all_dates: set[str] = set()
        sym_frames: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            df = _load_daily_close(symbol, pd.Timestamp(args.start), pd.Timestamp(args.end), store)
            if df.empty:
                print(f"  SKIPPED {symbol}: no data")
                continue
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"]).set_index("date")
            sym_frames[symbol] = df

        if len(sym_frames) < 2:
            print("  ERROR: Need at least 2 assets for X-Trend")
            return all_results

        # Outer-join all assets on date
        combined = pd.DataFrame()
        for sym, df in sym_frames.items():
            combined[sym] = df["close"]
        combined = combined.sort_index().ffill().dropna()
        combined = combined.reset_index().rename(columns={"index": "date"})

        x_cfg = XTrendConfig(
            d_hidden=args.hidden_dim,
            n_assets=len(sym_frames),
            train_window=args.train_window if args.train_window != 120 else 252,
            test_window=args.test_window if args.test_window != 20 else 63,
            loss_mode=args.loss_mode,
        )

        xtrend_results = run_xtrend_backtest(
            combined, list(sym_frames.keys()), x_cfg
        )
        all_results = xtrend_results

        for sym, res in all_results.items():
            print(f"\n  {sym}:")
            valid = res["strategy_return"].dropna()
            if len(valid) > 1:
                sv = np.expm1(valid.values)
                sharpe = (np.mean(sv) / np.std(sv, ddof=1)) * math.sqrt(252)
                tr = res["cumulative_pnl"].iloc[-1]
                print(f"    Sharpe: {sharpe:.4f}, Return: {tr:.2%}, OOS: {len(valid)} days")
            else:
                print(f"    Insufficient OOS data")

        return all_results

    # ------------------------------------------------------------------
    # TFT path (original)
    # ------------------------------------------------------------------
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Running: {symbol}")
        print(f"{'='*60}")

        ext_feats = None
        if args.mega_features:
            from quantlaxmi.features.mega import MegaFeatureBuilder
            logger.info("Building mega features for %s...", symbol)
            builder = MegaFeatureBuilder()
            ext_feats, ext_names = builder.build(symbol, args.start, args.end)
            logger.info("Mega features: %d features, %d rows", len(ext_names), len(ext_feats))

        try:
            result = run_backtest(
                symbol=symbol,
                start_date=args.start,
                end_date=args.end,
                store=store,
                cfg=cfg,
                use_torch=not args.no_torch,
                external_features=ext_feats,
            )
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            continue

        all_results[symbol] = result

        if args.mega_features and ext_feats is not None:
            print(f"  [MEGA FEATURES: {ext_feats.shape[1]} features from all data sources]")
        print(f"  Period: {args.start} to {args.end}")
        print(f"  Observations: {len(result)}")

        valid = result["strategy_return"].dropna()
        if len(valid) > 1:
            simple_valid = np.expm1(valid.values)
            sharpe = (np.mean(simple_valid) / np.std(simple_valid, ddof=1)) * math.sqrt(252)
            total_ret = result["cumulative_pnl"].iloc[-1]
            max_dd = _max_drawdown_equity(np.expm1(np.nan_to_num(
                result["strategy_return"].values, nan=0.0
            )))
            print(f"  Annualized Sharpe: {sharpe:.4f}")
            print(f"  Total Return: {total_ret:.2%}")
            print(f"  Max Drawdown: {max_dd:.2%}")
            print(f"  Hit Rate: {(valid > 0).mean():.2%}")
            print(f"  OOS Days: {len(valid)}")
        else:
            print("  Insufficient out-of-sample data for statistics.")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"{'SUMMARY':^60}")
        print(f"{'='*60}")
        print(f"{'Index':<30} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'OOS':>5}")
        print(f"{'-'*60}")
        for sym, res in all_results.items():
            valid = res["strategy_return"].dropna()
            if len(valid) > 1:
                sv = np.expm1(valid.values)
                sh = (np.mean(sv) / np.std(sv, ddof=1)) * math.sqrt(252)
                tr = res["cumulative_pnl"].iloc[-1]
                md = _max_drawdown_equity(np.expm1(np.nan_to_num(
                    res["strategy_return"].values, nan=0.0
                )))
                print(f"{sym:<30} {sh:>8.4f} {tr:>9.2%} {md:>8.2%} {len(valid):>5}")
            else:
                print(f"{sym:<30} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>5}")
        print(f"{'='*60}\n")

    return all_results


if __name__ == "__main__":
    main()
