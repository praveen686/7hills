"""Pattern 5: Attention Reward Shaping — cross-attention weight spikes as auxiliary RL reward.

When the X-Trend backbone's cross-attention entropy drops sharply (the model
suddenly focuses on a few context sequences), it signals a regime change.
We convert these attention spikes into small auxiliary reward bonuses so the
RL agent learns faster during regime transitions.

Theory:
    - Cross-attention weights w_i ∈ simplex over n_context items.
    - Entropy H = -sum(w_i * log(w_i)) measures attention spread.
    - High H → diffuse attention (no clear regime analogy).
    - Low H → focused attention (model has found a relevant past regime).
    - A *drop* in H (z-score < -threshold below rolling mean) → spike event.
    - Spike events → bonus added to base RL reward for faster convergence.

Architecture:
    AttentionRewardShaper
    ├── precompute_bonuses()   — run backbone with attention for entire fold
    │   ├── extract attention weights per (day, asset) via extract_hidden_with_attention
    │   ├── compute per-step entropy (averaged across assets and heads)
    │   ├── detect spikes via causal rolling z-score (window=21)
    │   └── return bonus array: bonus_scale * spike_indicator
    └── get_bonus(step_idx)    — O(1) lookup into precomputed array

    AttentionShapedEnv
    ├── wraps IntegratedTradingEnv
    ├── reset() → precomputes bonuses via shaper, injects into base env
    └── step() → delegates to base env (which already reads reward_bonuses)

References:
    - Wood et al. (2023) "X-Trend" — cross-attention for regime recognition
    - Ng et al. (1999) "Policy invariance under reward transformations"
    - MEMORY.md Pattern 5: TFT Attention as Reward Shaping
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from models.rl.integration.backbone import XTrendBackbone
from models.rl.integration.integrated_env import IntegratedTradingEnv


# ---------------------------------------------------------------------------
# Attention entropy utilities
# ---------------------------------------------------------------------------


def _attention_entropy(
    attn_weights: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute mean entropy of attention weight distribution.

    Parameters
    ----------
    attn_weights : ndarray of shape (batch, n_heads, 1, n_context)
        Raw attention weights from ``extract_hidden_with_attention``.
        Each (head, query) slice sums to 1.0 over the n_context dim.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        Mean entropy across all batch items and heads.
        Units: nats.  Range: [0, log(n_context)].
    """
    # Flatten to (batch * n_heads, n_context)
    w = attn_weights.reshape(-1, attn_weights.shape[-1])  # (B*H, n_ctx)

    # Clamp to avoid numerical issues (weights should already sum to 1,
    # but floating point can produce tiny negatives after softmax)
    w = np.clip(w, eps, 1.0)

    # Renormalize to guarantee exact sum-to-one after clipping
    w = w / w.sum(axis=-1, keepdims=True)

    # Shannon entropy: H = -sum(w * log(w))
    log_w = np.log(w + eps)
    h_per_row = -np.sum(w * log_w, axis=-1)  # (B*H,)

    return float(np.mean(h_per_row))


def _causal_rolling_zscore(
    series: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """Compute causal (backward-looking only) rolling z-score.

    For each index t, the z-score is computed using the mean and std
    of series[max(0, t-window+1) : t+1].  This ensures no look-ahead.

    Parameters
    ----------
    series : 1-D array of length T
    window : int
        Rolling window size.

    Returns
    -------
    z_scores : 1-D array of length T (NaN for first element if std==0)
    """
    n = len(series)
    z = np.zeros(n, dtype=np.float64)

    # Use cumulative sums for O(n) computation
    cum_sum = np.zeros(n + 1, dtype=np.float64)
    cum_sq = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        cum_sum[i + 1] = cum_sum[i] + series[i]
        cum_sq[i + 1] = cum_sq[i] + series[i] ** 2

    for t in range(n):
        start = max(0, t - window + 1)
        count = t - start + 1
        if count < 2:
            z[t] = 0.0
            continue
        s = cum_sum[t + 1] - cum_sum[start]
        sq = cum_sq[t + 1] - cum_sq[start]
        mean = s / count
        # Unbiased variance (ddof=1)
        var = (sq - count * mean * mean) / (count - 1)
        if var < 1e-20:
            z[t] = 0.0
        else:
            std = math.sqrt(var)
            z[t] = (series[t] - mean) / std

    return z


# ============================================================================
# AttentionRewardShaper
# ============================================================================


class AttentionRewardShaper:
    """Extract cross-attention spikes from the backbone and convert to reward bonuses.

    Workflow:
        1. For each day in the fold, run ``extract_hidden_with_attention`` on
           each asset to get attention weights.
        2. Compute per-day entropy (averaged across assets and heads).
        3. Compute causal rolling z-score of the entropy time series.
        4. Detect spikes: days where z < -spike_threshold (entropy significantly
           below average, meaning the model is focusing).
        5. Assign reward bonus = bonus_scale on spike days, 0 otherwise.

    Parameters
    ----------
    bonus_scale : float
        Magnitude of the reward bonus on spike days.  Should be small
        relative to base reward (~0.01) to shape without dominating.
    spike_threshold : float
        Z-score threshold for spike detection.  A spike is declared when
        z < -spike_threshold (i.e. entropy is spike_threshold standard
        deviations below the rolling mean).
    rolling_window : int
        Window size for the causal rolling z-score computation.
    eps : float
        Small constant for entropy computation.
    """

    def __init__(
        self,
        bonus_scale: float = 0.01,
        spike_threshold: float = 2.0,
        rolling_window: int = 21,
        eps: float = 1e-12,
    ) -> None:
        self.bonus_scale = bonus_scale
        self.spike_threshold = spike_threshold
        self.rolling_window = rolling_window
        self.eps = eps

        # Precomputed bonus array (populated by precompute_bonuses)
        self._bonuses: Optional[np.ndarray] = None
        # Raw entropy series (for diagnostics)
        self._entropy_series: Optional[np.ndarray] = None
        # Z-score series (for diagnostics)
        self._zscore_series: Optional[np.ndarray] = None
        # Spike mask (for diagnostics)
        self._spike_mask: Optional[np.ndarray] = None

    def precompute_bonuses(
        self,
        backbone: XTrendBackbone,
        features: np.ndarray,
        fold_start: int,
        fold_end: int,
        dates: Optional[np.ndarray] = None,
        batch_size: int = 64,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Run the backbone with attention extraction and compute bonus array.

        Parameters
        ----------
        backbone : XTrendBackbone
            Pre-trained, frozen backbone with ``model.extract_hidden_with_attention``.
        features : ndarray of shape (n_days, n_assets, n_features)
            Normalized feature tensor.
        fold_start : int
            Start index of the current fold in the features array.
        fold_end : int
            End index (exclusive) of the current fold.
        dates : optional
            DatetimeIndex or array of dates (for logging only).
        batch_size : int
            GPU batch size for forward passes.
        rng : optional numpy RNG
            For context set sampling.  If None, creates one with seed 42.

        Returns
        -------
        bonuses : ndarray of shape (fold_len,)
            Reward bonus for each step in the fold.
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for AttentionRewardShaper")

        if rng is None:
            rng = np.random.default_rng(42)

        fold_len = fold_end - fold_start
        n_assets = features.shape[1]
        seq_len = backbone.cfg.seq_len

        # Collect per-day entropy values
        entropy_per_day = np.zeros(fold_len, dtype=np.float64)

        # We process day by day, batching across assets for each day
        from models.ml.tft.x_trend import build_context_set

        backbone.eval()
        dev = next(backbone.parameters()).device

        for t_offset in range(fold_len):
            day_idx = fold_start + t_offset
            if day_idx < seq_len:
                # Not enough history — entropy is max (uniform attention)
                n_context = backbone.cfg.n_context
                if n_context > 0:
                    entropy_per_day[t_offset] = math.log(n_context)
                continue

            # Collect data for all assets at this day
            asset_tgt_list = []
            asset_ctx_list = []
            asset_tid_list = []
            asset_cid_list = []
            valid_assets = []

            for a in range(n_assets):
                target_window = features[day_idx - seq_len: day_idx, a, :]
                if np.any(np.isnan(target_window)):
                    continue

                ctx_seqs, ctx_ids = build_context_set(
                    features,
                    target_start=day_idx - seq_len,
                    n_context=backbone.cfg.n_context,
                    ctx_len=backbone.cfg.ctx_len,
                    rng=rng,
                )
                asset_tgt_list.append(target_window)
                asset_ctx_list.append(ctx_seqs)
                asset_tid_list.append(a)
                asset_cid_list.append(ctx_ids)
                valid_assets.append(a)

            if not valid_assets:
                # No valid assets — assign max entropy
                n_context = backbone.cfg.n_context
                if n_context > 0:
                    entropy_per_day[t_offset] = math.log(n_context)
                continue

            # Batch forward pass with attention extraction
            tgt_arr = np.array(asset_tgt_list, dtype=np.float32)
            ctx_arr = np.array(asset_ctx_list, dtype=np.float32)
            tid_arr = np.array(asset_tid_list, dtype=np.int64)
            cid_arr = np.array(asset_cid_list, dtype=np.int64)

            n_total = len(valid_assets)
            day_entropies = []

            with torch.no_grad():
                for bs in range(0, n_total, batch_size):
                    be = min(bs + batch_size, n_total)
                    tgt_t = torch.tensor(tgt_arr[bs:be], device=dev)
                    ctx_t = torch.tensor(ctx_arr[bs:be], device=dev)
                    tid_t = torch.tensor(tid_arr[bs:be], device=dev)
                    cid_t = torch.tensor(cid_arr[bs:be], device=dev)

                    _, attn_w = backbone.model.extract_hidden_with_attention(
                        tgt_t, ctx_t, tid_t, cid_t,
                    )
                    # attn_w: (batch_chunk, n_heads, 1, n_context)
                    attn_np = attn_w.cpu().numpy()

                    # Compute entropy per sample in this chunk
                    for j in range(be - bs):
                        h = _attention_entropy(
                            attn_np[j: j + 1],  # keep batch dim
                            eps=self.eps,
                        )
                        day_entropies.append(h)

            # Average entropy across all valid assets for this day
            entropy_per_day[t_offset] = float(np.mean(day_entropies))

        # Compute causal rolling z-score of the entropy series
        z_scores = _causal_rolling_zscore(entropy_per_day, window=self.rolling_window)

        # Detect spikes: z < -threshold means entropy is significantly below average
        # (model is focusing attention → regime change signal)
        spike_mask = z_scores < -self.spike_threshold

        # Build bonus array
        bonuses = np.where(spike_mask, self.bonus_scale, 0.0)

        # Store for diagnostics
        self._bonuses = bonuses
        self._entropy_series = entropy_per_day
        self._zscore_series = z_scores
        self._spike_mask = spike_mask

        n_spikes = int(np.sum(spike_mask))
        logger.info(
            "AttentionRewardShaper: fold [%d:%d], %d spikes / %d days (%.1f%%), "
            "entropy mean=%.4f, std=%.4f",
            fold_start, fold_end, n_spikes, fold_len,
            100.0 * n_spikes / max(fold_len, 1),
            float(np.mean(entropy_per_day)),
            float(np.std(entropy_per_day, ddof=1)) if fold_len > 1 else 0.0,
        )

        return bonuses

    def get_bonus(self, step_idx: int) -> float:
        """Return the precomputed reward bonus for a given step.

        Parameters
        ----------
        step_idx : int
            Step index within the current fold (0-based).

        Returns
        -------
        float
            Reward bonus (>= 0).  Returns 0.0 if bonuses have not been
            precomputed or step_idx is out of range.
        """
        if self._bonuses is None:
            return 0.0
        if step_idx < 0 or step_idx >= len(self._bonuses):
            return 0.0
        return float(self._bonuses[step_idx])

    @property
    def entropy_series(self) -> Optional[np.ndarray]:
        """Raw entropy time series from last precompute (for diagnostics)."""
        return self._entropy_series

    @property
    def zscore_series(self) -> Optional[np.ndarray]:
        """Z-score time series from last precompute (for diagnostics)."""
        return self._zscore_series

    @property
    def spike_mask(self) -> Optional[np.ndarray]:
        """Boolean spike mask from last precompute (for diagnostics)."""
        return self._spike_mask

    @property
    def n_spikes(self) -> int:
        """Number of spikes detected in the last precompute."""
        if self._spike_mask is None:
            return 0
        return int(np.sum(self._spike_mask))

    @property
    def spike_rate(self) -> float:
        """Fraction of days that are spikes in the last precompute."""
        if self._spike_mask is None:
            return 0.0
        return float(np.mean(self._spike_mask))


# ============================================================================
# AttentionShapedEnv
# ============================================================================


class AttentionShapedEnv:
    """Gymnasium-like wrapper that adds attention reward shaping to IntegratedTradingEnv.

    On ``reset()``, this wrapper:
    1. Resets the base environment (which pre-computes hidden states).
    2. Runs ``AttentionRewardShaper.precompute_bonuses()`` to extract attention
       entropy spikes for the fold.
    3. Injects the bonus array into the base environment via ``set_reward_bonuses()``.

    On ``step()``, the base environment automatically adds the precomputed bonus
    to the reward at each step (see ``IntegratedTradingEnv.step``).

    Parameters
    ----------
    base_env : IntegratedTradingEnv
        The wrapped multi-asset trading environment.
    shaper : AttentionRewardShaper
        The reward shaper that extracts attention spikes.
    bonus_scale : float
        Override for the shaper's bonus scale (if provided, updates shaper).
    """

    def __init__(
        self,
        base_env: IntegratedTradingEnv,
        shaper: AttentionRewardShaper,
        bonus_scale: float = 0.01,
    ) -> None:
        self.base_env = base_env
        self.shaper = shaper

        # Update shaper's bonus_scale if caller overrides
        if bonus_scale != self.shaper.bonus_scale:
            self.shaper.bonus_scale = bonus_scale

        # Track fold boundaries to avoid re-computing when unnecessary
        self._last_fold_start: Optional[int] = None
        self._last_fold_end: Optional[int] = None

        # RNG for reproducible context set sampling during attention extraction
        self._rng = np.random.default_rng(123)

    # -- Delegated properties from base env --

    @property
    def state_dim(self) -> int:
        """State vector dimension (same as base env)."""
        return self.base_env.state_dim

    @property
    def action_dim(self) -> int:
        """Action dimension (same as base env)."""
        return self.base_env.action_dim

    @property
    def n_assets(self) -> int:
        """Number of tradeable assets."""
        return self.base_env.n_assets

    @property
    def backbone(self):
        """Access the backbone through the base env."""
        return self.base_env.backbone

    @property
    def features(self) -> np.ndarray:
        """Access the feature tensor through the base env."""
        return self.base_env.features

    @property
    def targets(self) -> np.ndarray:
        """Access the targets through the base env."""
        return self.base_env.targets

    @property
    def symbols(self) -> list:
        """Asset symbols from the base env."""
        return self.base_env.symbols

    def reset(
        self,
        fold_start_idx: int,
        fold_end_idx: int,
    ) -> np.ndarray:
        """Reset the environment and precompute attention reward bonuses.

        Parameters
        ----------
        fold_start_idx : int
            Start index of the fold in the feature tensor.
        fold_end_idx : int
            End index (exclusive) of the fold.

        Returns
        -------
        state : ndarray of shape (state_dim,)
            Initial state vector.
        """
        # 1. Reset base environment (pre-computes hidden states)
        state = self.base_env.reset(fold_start_idx, fold_end_idx)

        # 2. Precompute attention bonuses (only if fold changed)
        fold_changed = (
            self._last_fold_start != fold_start_idx
            or self._last_fold_end != fold_end_idx
        )

        if fold_changed:
            logger.info(
                "AttentionShapedEnv: precomputing bonuses for fold [%d:%d]...",
                fold_start_idx, fold_end_idx,
            )
            bonuses = self.shaper.precompute_bonuses(
                backbone=self.base_env.backbone,
                features=self.base_env.features,
                fold_start=fold_start_idx,
                fold_end=fold_end_idx,
                rng=self._rng,
            )
            self._last_fold_start = fold_start_idx
            self._last_fold_end = fold_end_idx
        else:
            # Reuse existing bonuses (same fold, just new episode)
            bonuses = self.shaper._bonuses
            if bonuses is None:
                # Safety fallback: recompute
                bonuses = self.shaper.precompute_bonuses(
                    backbone=self.base_env.backbone,
                    features=self.base_env.features,
                    fold_start=fold_start_idx,
                    fold_end=fold_end_idx,
                    rng=self._rng,
                )

        # 3. Inject bonuses into base environment
        self.base_env.set_reward_bonuses(bonuses)

        return state

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Step the base environment (bonuses already injected via set_reward_bonuses).

        Parameters
        ----------
        actions : ndarray of shape (n_assets,)
            Continuous position targets in [-1, 1].

        Returns
        -------
        state : ndarray of shape (state_dim,)
        reward : float — base reward + attention bonus
        done : bool
        info : dict
        """
        state, reward, done, info = self.base_env.step(actions)

        # Annotate info with attention shaping metadata
        step_idx = self.base_env._step_idx - 1  # step already incremented
        if self.shaper._bonuses is not None and 0 <= step_idx < len(self.shaper._bonuses):
            info["attention_bonus"] = float(self.shaper._bonuses[step_idx])
            if self.shaper._entropy_series is not None:
                info["attention_entropy"] = float(self.shaper._entropy_series[step_idx])
            if self.shaper._zscore_series is not None:
                info["attention_entropy_zscore"] = float(self.shaper._zscore_series[step_idx])
        else:
            info["attention_bonus"] = 0.0

        return state, reward, done, info

    def set_reward_bonuses(self, bonuses: np.ndarray) -> None:
        """Override: directly set bonuses on the base environment.

        Useful if you want to inject custom bonuses instead of using
        the shaper's precomputed ones.
        """
        self.base_env.set_reward_bonuses(bonuses)

    def get_cost_per_leg(self, symbol: str) -> float:
        """Delegate to base environment."""
        return self.base_env.get_cost_per_leg(symbol)
