"""Causal data augmentation for TFT training.

Three safe augmentation strategies for financial time-series:

1. **Block Bootstrap** — resample contiguous blocks within each episode.
   Preserves intra-block autocorrelation. Same block selection is applied
   to X_tgt AND all X_ctx slices to maintain cross-asset temporal alignment.

2. **Feature Group Dropout** — zero-out entire feature groups per episode.
   Forces robustness. Groups defined by column prefix (e.g. all "opt_*").

3. **Calibrated Noise** — small Gaussian noise (σ=0.01–0.03) on z-scored
   continuous features. Drawn per-channel for stability.

SAFETY:
    - NO Mixup between assets (creates unrealistic trajectories)
    - NO temporal jittering (breaks causality)
    - Block bootstrap uses a shared time-index map across X_tgt and all
      X_ctx slices to preserve cross-asset temporal alignment
    - Y (scalar target) is never modified
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.utils.data as _torch_data
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _torch_data = None  # type: ignore


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for TFT data augmentation."""

    enabled: bool = True

    # How many augmented copies per real episode (total = real + real*factor)
    augmentation_factor: int = 3

    # Block bootstrap
    block_bootstrap: bool = True
    block_lengths: list[int] = field(default_factory=lambda: [7, 10, 14])

    # Feature group dropout
    feature_group_dropout: bool = True
    p_group_drop: float = 0.10  # probability of dropping each group
    # Groups defined by prefix → column indices (set at runtime)
    feature_groups: dict[str, list[int]] = field(default_factory=dict)

    # Calibrated noise
    calibrated_noise: bool = True
    noise_std: float = 0.02  # σ for Gaussian noise on z-scored features

    # Random seed (per-epoch reseeding for variety)
    seed: int = 42


# ============================================================================
# Feature group classification
# ============================================================================

# Prefix → group name mapping (matches tft_30trial.py and mega.py)
_PREFIX_TO_GROUP: dict[str, str] = {
    "px_": "price",
    "opt_": "options",
    "optx_": "options",
    "inst_": "institutional",
    "brd_": "breadth",
    "vix_": "vix",
    "intra_": "intraday",
    "fut_": "futures",
    "fii_": "fii",
    "nfo_": "nfo_1min",
    "pvol_": "participant_vol",
    "dff_": "divergence_flow",
    "ca_": "cross_asset",
    "ns_": "news_sentiment",
    "micro_": "microstructure",
    "mkt_": "market_activity",
    "mwpl_": "mwpl",
    "settle_": "settlement",
    "nsevol_": "nse_vol",
    "cdelta_": "contract_delta",
    "deoi_": "deltaeq_oi",
    "preopen_": "preopen",
    "oisp_": "oi_spurts",
    "cryp_": "crypto",
    "macro_": "macro",
}


def classify_features_into_groups(
    feature_names: list[str],
) -> dict[str, list[int]]:
    """Map feature names to groups by prefix. Returns {group: [col_indices]}."""
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(feature_names):
        matched = False
        for prefix, group in _PREFIX_TO_GROUP.items():
            if name.startswith(prefix):
                groups.setdefault(group, []).append(idx)
                matched = True
                break
        if not matched:
            groups.setdefault("_other", []).append(idx)
    return groups


# ============================================================================
# Augmentation transforms (operate on numpy arrays)
# ============================================================================


def block_bootstrap(
    x_tgt: np.ndarray,
    x_ctx: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample contiguous blocks along the time axis.

    The SAME block index map is applied to x_tgt and ALL context slices
    in x_ctx to preserve cross-asset temporal alignment.

    Parameters
    ----------
    x_tgt : (seq_len, n_features)
    x_ctx : (n_context, seq_len, n_features)
    block_length : int, block size (e.g. 7, 10, 14)
    rng : numpy random generator

    Returns
    -------
    (x_tgt_aug, x_ctx_aug) with same shapes as input
    """
    seq_len = x_tgt.shape[0]
    if block_length >= seq_len:
        return x_tgt.copy(), x_ctx.copy()

    # Build the shared time-index map
    n_blocks = int(np.ceil(seq_len / block_length))
    # Sample block start positions (with replacement)
    max_start = seq_len - block_length
    block_starts = rng.integers(0, max_start + 1, size=n_blocks)

    # Build concatenated index array, trim to seq_len
    indices = np.concatenate(
        [np.arange(s, s + block_length) for s in block_starts]
    )[:seq_len]

    # Apply same indices to target and all context slices
    x_tgt_aug = x_tgt[indices]
    x_ctx_aug = x_ctx[:, indices, :]  # (n_context, seq_len, n_features)

    return x_tgt_aug, x_ctx_aug


def feature_group_dropout(
    x: np.ndarray,
    groups: dict[str, list[int]],
    p_drop: float,
    rng: np.random.Generator,
    protected_groups: set[str] | None = None,
) -> np.ndarray:
    """Zero-out entire feature groups with probability p_drop.

    Parameters
    ----------
    x : (..., n_features) — last dim is features
    groups : {group_name: [col_indices]}
    p_drop : probability per group
    rng : numpy random generator
    protected_groups : groups that are never dropped (e.g. {"price"})

    Returns
    -------
    x_aug : same shape, with dropped groups zeroed
    """
    if protected_groups is None:
        protected_groups = {"price", "_other"}

    x_aug = x.copy()
    for group_name, col_indices in groups.items():
        if group_name in protected_groups:
            continue
        if rng.random() < p_drop:
            x_aug[..., col_indices] = 0.0
    return x_aug


def calibrated_noise(
    x: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add small Gaussian noise to z-scored features.

    Independent N(0, σ²) noise is drawn per element (timestep × feature).

    Parameters
    ----------
    x : (..., n_features)
    noise_std : σ for N(0, σ²) noise
    rng : numpy random generator

    Returns
    -------
    x_aug : same shape
    """
    noise = rng.normal(0.0, noise_std, size=x.shape).astype(x.dtype)
    return x + noise


# ============================================================================
# Augmented Dataset
# ============================================================================


if _HAS_TORCH:

    class AugmentedTFTDataset(_torch_data.Dataset):
        """Wraps a base TFT dataset with on-the-fly augmentation.

        For indices [0, N) returns original samples.
        For indices [N, N*(1+factor)) returns augmented copies.

        Each augmented copy applies (per the config):
          1. Block bootstrap (shared time map across tgt + ctx)
          2. Feature group dropout
          3. Calibrated noise

        Parameters
        ----------
        base_X_tgt : np.ndarray (N, seq_len, n_features)
        base_X_ctx : np.ndarray (N, n_context, seq_len, n_features)
        base_X_tid : np.ndarray (N,)
        base_X_cid : np.ndarray (N, n_context)
        base_Y     : np.ndarray (N,)
        config     : AugmentationConfig
        feature_names : list[str] — for group classification
        """

        def __init__(
            self,
            base_X_tgt: np.ndarray,
            base_X_ctx: np.ndarray,
            base_X_tid: np.ndarray,
            base_X_cid: np.ndarray,
            base_Y: np.ndarray,
            config: AugmentationConfig,
            feature_names: list[str] | None = None,
        ) -> None:
            self.N = len(base_Y)
            self.factor = config.augmentation_factor if config.enabled else 0
            self.config = config

            # Store numpy originals for augmentation
            self._np_tgt = base_X_tgt
            self._np_ctx = base_X_ctx
            self._np_tid = base_X_tid
            self._np_cid = base_X_cid
            self._np_Y = base_Y

            # Pre-convert originals to tensors
            self.T_tgt = torch.as_tensor(base_X_tgt, dtype=torch.float32)
            self.T_ctx = torch.as_tensor(base_X_ctx, dtype=torch.float32)
            self.T_tid = torch.as_tensor(base_X_tid, dtype=torch.int64)
            self.T_cid = torch.as_tensor(base_X_cid, dtype=torch.int64)
            self.T_Y = torch.as_tensor(base_Y, dtype=torch.float32)

            # Feature groups for dropout
            if feature_names is not None and config.feature_group_dropout:
                self._groups = classify_features_into_groups(feature_names)
            else:
                self._groups = {}

            # Block length choices
            self._block_lengths = config.block_lengths

            # Epoch counter for per-sample deterministic seeding
            self._epoch: int = 0

        def __len__(self) -> int:
            return self.N * (1 + self.factor)

        def __getitem__(self, idx: int) -> tuple:
            if idx < self.N:
                # Original sample — return pre-converted tensors
                return (
                    self.T_tgt[idx],
                    self.T_ctx[idx],
                    self.T_tid[idx],
                    self.T_cid[idx],
                    self.T_Y[idx],
                )

            # Augmented sample — deterministic RNG seeded from (seed, epoch, idx)
            # so each epoch produces different augmentations even with persistent workers
            real_idx = idx % self.N
            rng = np.random.default_rng(
                self.config.seed + self._epoch * 100_000 + idx
            )

            x_tgt = self._np_tgt[real_idx].copy()  # (seq_len, n_features)
            x_ctx = self._np_ctx[real_idx].copy()  # (n_context, seq_len, n_features)

            # 1. Block bootstrap (shared time map)
            if self.config.block_bootstrap:
                bl = rng.choice(self._block_lengths)
                x_tgt, x_ctx = block_bootstrap(x_tgt, x_ctx, bl, rng)

            # 2. Feature group dropout — draw mask ONCE, apply to BOTH tgt and ctx
            if self.config.feature_group_dropout and self._groups:
                drop_mask = np.ones(x_tgt.shape[-1], dtype=np.float32)
                for gname, cols in self._groups.items():
                    if gname in {"price", "_other"}:
                        continue
                    if rng.random() < self.config.p_group_drop:
                        drop_mask[cols] = 0.0
                x_tgt = x_tgt * drop_mask[np.newaxis, :]
                x_ctx = x_ctx * drop_mask[np.newaxis, np.newaxis, :]

            # 3. Calibrated noise
            if self.config.calibrated_noise:
                x_tgt = calibrated_noise(x_tgt, self.config.noise_std, rng)
                x_ctx = calibrated_noise(x_ctx, self.config.noise_std, rng)

            return (
                torch.as_tensor(x_tgt, dtype=torch.float32),
                torch.as_tensor(x_ctx, dtype=torch.float32),
                self.T_tid[real_idx],
                self.T_cid[real_idx],
                self.T_Y[real_idx],  # Y is NEVER modified
            )

        def reseed(self, epoch: int) -> None:
            """Set epoch for per-sample deterministic seeding.

            Each augmented sample's RNG is seeded from (config.seed, epoch, idx),
            so different epochs produce different augmentations even with
            persistent DataLoader workers (no worker communication needed).
            """
            self._epoch = epoch
