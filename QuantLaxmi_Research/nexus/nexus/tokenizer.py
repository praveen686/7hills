"""Multi-Scale VQ-VAE Tokenizer for Market Data.

Converts raw OHLCV market data into discrete token sequences that capture
multi-scale market structure. This is the data preprocessing bridge between
raw market data and the Mamba-2 backbone.

Three scales:
    1. Micro  (1-min bars)        -> local price dynamics
    2. Meso   (hourly aggregates) -> intraday patterns
    3. Macro  (daily bars)        -> trend structure

Each scale uses:
    - Delta encoding (predict deltas, not levels) for stationarity
    - Learnable projection to d_model with LayerNorm
    - Optional VQ discretization for compression / codebook learning

Design principles:
    - ALL features are strictly causal (no look-ahead bias)
    - .shift(1) applied wherever current-bar data is used as input
    - Volume features use raw deltas (not log) to handle zero volume
    - Log returns for prices (numerically stable, approximately normal)

The VQ codebook learns a finite vocabulary of "market states" at each scale,
analogous to how language models tokenize text into subwords. This provides:
    - Compression: raw floats -> discrete tokens (log2(512) = 9 bits)
    - Regularization: forces the model to learn archetypal market patterns
    - Interpretability: each codebook entry = a recognizable market micro-state

References:
    VQ-VAE: van den Oord et al., NeurIPS 2017
    VQ-VAE-2: Razavi et al., NeurIPS 2019
    SoundStream: Zeghidour et al., 2021 (multi-scale VQ for audio)
    Multi-scale finance: Muller et al., "Heterogeneous Agents Model", 1997
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TokenizerConfig:
    """Configuration for the multi-scale tokenizer."""

    d_model: int = 256              # Output embedding dimension
    n_features_micro: int = 8       # Features per micro (1-min) bar
    n_features_meso: int = 8        # Features per meso (hourly) bar
    n_features_macro: int = 8       # Features per macro (daily) bar

    # VQ parameters
    use_vq: bool = True             # Whether to use VQ discretization
    num_embeddings: int = 512       # Codebook size
    d_embedding: int = 64           # VQ embedding dimension
    commitment_cost: float = 0.25   # Beta for commitment loss
    ema_decay: float = 0.99         # EMA decay for codebook updates
    dead_code_threshold: int = 2    # Reset codes unused for this many batches

    # Multi-scale fusion
    fusion_method: str = "concat"   # "concat" or "add"


# ---------------------------------------------------------------------------
# Delta Encoder -- convert levels to returns/deltas
# ---------------------------------------------------------------------------


class DeltaEncoder(nn.Module):
    """Converts price levels to returns/deltas for stationarity.

    For price columns (OHLC): computes log returns = log(p_t / p_{t-1}).
    For volume/OI columns: computes raw deltas = v_t - v_{t-1}.

    This is critical for financial data:
        - Raw prices are non-stationary (random walk)
        - Returns are approximately stationary
        - The model should learn from CHANGES, not LEVELS

    All computations use shift(1) to ensure strict causality:
    the delta at time t uses only data from times t and t-1.

    Parameters
    ----------
    n_features : int
        Total number of input features.
    price_columns : list of int, optional
        Indices of price-like columns (use log returns).
        Defaults to first 4 columns (OHLC).
    volume_columns : list of int, optional
        Indices of volume-like columns (use raw deltas).
        Defaults to column 4 (volume).
    eps : float
        Epsilon for log return computation (avoid log(0)).
    """

    def __init__(
        self,
        n_features: int,
        price_columns: Optional[List[int]] = None,
        volume_columns: Optional[List[int]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_features = n_features
        self.price_columns = (
            price_columns if price_columns is not None
            else list(range(min(4, n_features)))
        )
        self.volume_columns = (
            volume_columns if volume_columns is not None
            else ([4] if n_features > 4 else [])
        )
        self.eps = eps

        # Passthrough columns: neither price nor volume
        all_special = set(self.price_columns) | set(self.volume_columns)
        self.passthrough_columns = [
            i for i in range(n_features) if i not in all_special
        ]

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Compute deltas from raw levels.

        Parameters
        ----------
        x : (B, L, n_features) -- raw OHLCV data

        Returns
        -------
        (B, L, n_features) -- delta-encoded data.
            First timestep is zero-filled (no previous bar to diff against).
        """
        B, L, F = x.shape
        out = torch.zeros_like(x)

        # Log returns for price columns: log(x_t / x_{t-1})
        # Prices are clamped to eps to handle non-positive inputs safely
        # (real market prices are always positive, but this guards against
        # synthetic data or pre-processed features that may go negative)
        if self.price_columns:
            prices = x[:, :, self.price_columns].clamp(min=self.eps)  # (B, L, n_price)
            prices_prev = torch.roll(prices, shifts=1, dims=1)
            prices_prev[:, 0, :] = prices[:, 0, :]  # First row: no delta
            log_ret = torch.log(prices / prices_prev)
            log_ret[:, 0, :] = 0.0  # Zero-fill first timestep
            out[:, :, self.price_columns] = log_ret

        # Raw deltas for volume columns: v_t - v_{t-1}
        if self.volume_columns:
            vols = x[:, :, self.volume_columns]
            vols_prev = torch.roll(vols, shifts=1, dims=1)
            vols_prev[:, 0, :] = vols[:, 0, :]
            vol_delta = vols - vols_prev
            vol_delta[:, 0, :] = 0.0
            out[:, :, self.volume_columns] = vol_delta

        # Passthrough columns: keep as-is (e.g., pre-computed features)
        if self.passthrough_columns:
            out[:, :, self.passthrough_columns] = x[:, :, self.passthrough_columns]

        return out


# ---------------------------------------------------------------------------
# Scale Projection -- per-scale learnable projection
# ---------------------------------------------------------------------------


class ScaleProjection(nn.Module):
    """Per-scale learnable linear projection to d_model with LayerNorm.

    Each scale (micro/meso/macro) has its own projection head that maps
    raw delta-encoded features to the shared d_model embedding space.

    Architecture:
        input -> Linear(d_in, d_model) -> LayerNorm -> GELU -> Dropout
              -> Linear(d_model, d_model) -> LayerNorm

    Parameters
    ----------
    n_features : int
        Number of input features for this scale.
    d_model : int
        Output embedding dimension.
    dropout : float
        Dropout rate after projection.
    """

    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Project features to d_model.

        Parameters
        ----------
        x : (B, L, n_features)

        Returns
        -------
        (B, L, d_model)
        """
        return self.proj(x)


# ---------------------------------------------------------------------------
# Vector Quantizer -- VQ-VAE codebook with EMA updates
# ---------------------------------------------------------------------------


class VectorQuantizer(nn.Module):
    """VQ-VAE codebook with EMA updates and dead-code restart.

    Maintains a learnable codebook of ``num_embeddings`` vectors.  During
    the forward pass each input vector is snapped to its nearest codebook
    entry.  The codebook is updated via exponential moving averages (EMA)
    of the assigned vectors -- no codebook gradients needed.

    Dead codes (entries that receive no assignments for several batches)
    are re-initialized from a randomly sampled encoder output to keep the
    full codebook utilized.

    Parameters
    ----------
    num_embeddings : int
        Codebook size (number of discrete tokens).
    d_embedding : int
        Dimension of each codebook vector.
    commitment_cost : float
        Weight for the commitment loss (encoder pulled toward codebook).
    ema_decay : float
        Decay rate for EMA codebook updates.
    dead_code_threshold : int
        Reset a code if unused for this many consecutive forward passes.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        d_embedding: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: int = 2,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.d_embedding = d_embedding
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook embeddings
        self.register_buffer(
            "embeddings", torch.randn(num_embeddings, d_embedding)
        )
        nn.init.uniform_(
            self.embeddings, -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        # EMA cluster sizes and sums
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_cluster_sum", self.embeddings.clone())

        # Dead code tracking: how many batches since last assignment
        self.register_buffer(
            "usage_count", torch.zeros(num_embeddings, dtype=torch.long)
        )

    def forward(
        self, z: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Quantize input vectors to nearest codebook entries.

        Parameters
        ----------
        z : (B, L, d_embedding) -- continuous encoder outputs

        Returns
        -------
        quantized : (B, L, d_embedding)
            Quantized vectors (straight-through gradient estimator).
        vq_loss : scalar
            VQ loss = codebook_loss + commitment_cost * commitment_loss.
        encoding_indices : (B, L)
            Integer indices into codebook.
        perplexity : scalar
            exp(entropy) of codebook usage (higher = better utilization).
        """
        B, L, D = z.shape
        assert D == self.d_embedding, (
            f"Input dim {D} != codebook dim {self.d_embedding}"
        )

        # Flatten to (B*L, D) for distance computation
        z_flat = z.reshape(-1, D)  # (N, D)

        # Compute distances: ||z - e||^2 = ||z||^2 - 2*z*e^T + ||e||^2
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_flat @ self.embeddings.t()
            + self.embeddings.pow(2).sum(dim=1, keepdim=True).t()
        )  # (N, num_embeddings)

        # Find nearest codebook entry
        encoding_indices = distances.argmin(dim=1)  # (N,)

        # Quantize: look up codebook vectors
        quantized_flat = self.embeddings[encoding_indices]  # (N, D)
        quantized = quantized_flat.reshape(B, L, D)

        # Compute losses
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: gradients pass through quantization
        quantized = z + (quantized - z).detach()

        # EMA codebook update (only during training)
        if self.training:
            self._ema_update(z_flat, encoding_indices)
            self._restart_dead_codes(z_flat, encoding_indices)

        # Perplexity: measure of codebook utilization
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        encoding_indices = encoding_indices.reshape(B, L)

        return quantized, vq_loss, encoding_indices, perplexity

    @torch.no_grad()
    def _ema_update(
        self, z_flat: "torch.Tensor", indices: "torch.Tensor"
    ) -> None:
        """Update codebook via exponential moving average.

        Parameters
        ----------
        z_flat : (N, D) -- flattened encoder outputs
        indices : (N,) -- assigned codebook indices
        """
        encodings = F.one_hot(indices, self.num_embeddings).float()  # (N, K)

        # Update cluster sizes
        cluster_size = encodings.sum(dim=0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1.0 - self.ema_decay
        )

        # Update cluster sums
        cluster_sum = encodings.t() @ z_flat  # (K, D)
        self.ema_cluster_sum.mul_(self.ema_decay).add_(
            cluster_sum, alpha=1.0 - self.ema_decay
        )

        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size.sum()
        smoothed_size = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.num_embeddings * 1e-5)
            * n
        )

        # Update embeddings
        self.embeddings.copy_(self.ema_cluster_sum / smoothed_size.unsqueeze(1))

    @torch.no_grad()
    def _restart_dead_codes(
        self, z_flat: "torch.Tensor", indices: "torch.Tensor"
    ) -> None:
        """Re-initialize codebook entries that have not been used recently.

        Dead codes waste codebook capacity. When a code goes unused for
        ``dead_code_threshold`` consecutive batches, it is re-initialized
        to a randomly chosen encoder output (with small noise).

        Parameters
        ----------
        z_flat : (N, D) -- flattened encoder outputs (source for restarts)
        indices : (N,) -- assigned codebook indices this batch
        """
        # Track which codes were used this batch
        used = torch.zeros(
            self.num_embeddings, dtype=torch.bool, device=indices.device
        )
        used.scatter_(0, indices, True)

        # Increment unused counter; reset used codes to 0
        self.usage_count += 1
        self.usage_count[used] = 0

        # Find dead codes
        dead_mask = self.usage_count >= self.dead_code_threshold
        n_dead = dead_mask.sum().item()

        if n_dead > 0 and z_flat.size(0) > 0:
            # Sample random encoder outputs as replacements
            replace_idx = torch.randint(
                0, z_flat.size(0), (n_dead,), device=z_flat.device
            )
            new_codes = z_flat[replace_idx]
            # Add small noise for diversity
            new_codes = new_codes + torch.randn_like(new_codes) * 0.01

            self.embeddings[dead_mask] = new_codes
            self.ema_cluster_size[dead_mask] = 1.0
            self.ema_cluster_sum[dead_mask] = new_codes
            self.usage_count[dead_mask] = 0


# ---------------------------------------------------------------------------
# Multi-Scale Tokenizer -- main module
# ---------------------------------------------------------------------------


class MultiScaleTokenizer(nn.Module):
    """Multi-scale VQ tokenizer for market data.

    Converts raw OHLCV + features into discrete token sequences that
    capture multi-scale market structure.

    Pipeline per scale:
        raw data -> DeltaEncoder (returns/deltas) -> ScaleProjection (to d_model)
        -> (optional) VQ codebook -> fused output

    Cross-scale fusion aligns temporal resolutions: micro (1-min) is
    average-pooled to match macro (daily) length, meso (hourly) is pooled
    similarly. The aligned representations are concatenated (or summed)
    and projected to d_model via a fusion MLP.

    Parameters
    ----------
    cfg : TokenizerConfig
        Tokenizer configuration.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg

        # Per-scale delta encoders
        self.delta_micro = DeltaEncoder(cfg.n_features_micro)
        self.delta_meso = DeltaEncoder(cfg.n_features_meso)
        self.delta_macro = DeltaEncoder(cfg.n_features_macro)

        # Per-scale projections to d_model
        self.proj_micro = ScaleProjection(cfg.n_features_micro, cfg.d_model)
        self.proj_meso = ScaleProjection(cfg.n_features_meso, cfg.d_model)
        self.proj_macro = ScaleProjection(cfg.n_features_macro, cfg.d_model)

        # Optional VQ layer (shared across scales, applied after fusion)
        self.vq: Optional[VectorQuantizer] = None
        self.pre_vq_proj: Optional[nn.Linear] = None
        self.post_vq_proj: Optional[nn.Linear] = None
        if cfg.use_vq:
            self.pre_vq_proj = nn.Linear(cfg.d_model, cfg.d_embedding)
            self.vq = VectorQuantizer(
                num_embeddings=cfg.num_embeddings,
                d_embedding=cfg.d_embedding,
                commitment_cost=cfg.commitment_cost,
                ema_decay=cfg.ema_decay,
                dead_code_threshold=cfg.dead_code_threshold,
            )
            self.post_vq_proj = nn.Linear(cfg.d_embedding, cfg.d_model)

        # Cross-scale fusion
        if cfg.fusion_method == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(cfg.d_model * 3, cfg.d_model * 2),
                nn.LayerNorm(cfg.d_model * 2),
                nn.GELU(),
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
            )
        else:
            # Additive fusion: just layer-norm the sum
            self.fusion = nn.LayerNorm(cfg.d_model)

    def encode_scale(
        self,
        data: "torch.Tensor",
        scale_name: str,
    ) -> "torch.Tensor":
        """Process a single scale through delta encoding + projection.

        Parameters
        ----------
        data : (B, L, n_features) -- raw data at this scale
        scale_name : str -- one of "micro", "meso", "macro"

        Returns
        -------
        (B, L, d_model) -- projected embeddings for this scale
        """
        if scale_name == "micro":
            delta = self.delta_micro(data)
            return self.proj_micro(delta)
        elif scale_name == "meso":
            delta = self.delta_meso(data)
            return self.proj_meso(delta)
        elif scale_name == "macro":
            delta = self.delta_macro(data)
            return self.proj_macro(delta)
        else:
            raise ValueError(
                f"Unknown scale: {scale_name}. Use 'micro', 'meso', 'macro'."
            )

    def fuse_scales(
        self,
        micro: "torch.Tensor",
        meso: "torch.Tensor",
        macro: "torch.Tensor",
    ) -> "torch.Tensor":
        """Temporal alignment and cross-scale fusion.

        Aligns the three scales to the MACRO (shortest) temporal resolution
        by average-pooling the finer scales, then fuses via concat+MLP or
        addition+LayerNorm.

        Parameters
        ----------
        micro : (B, L_micro, d_model)
        meso : (B, L_meso, d_model)
        macro : (B, L_macro, d_model)

        Returns
        -------
        (B, L_macro, d_model) -- fused multi-scale representation
        """
        L_out = macro.size(1)

        # Average-pool micro and meso to macro resolution
        micro_aligned = self._adaptive_pool(micro, L_out)
        meso_aligned = self._adaptive_pool(meso, L_out)

        if self.cfg.fusion_method == "concat":
            fused = torch.cat([micro_aligned, meso_aligned, macro], dim=-1)
            return self.fusion(fused)
        else:
            fused = micro_aligned + meso_aligned + macro
            return self.fusion(fused)

    @staticmethod
    def _adaptive_pool(x: "torch.Tensor", target_len: int) -> "torch.Tensor":
        """Average-pool sequence to target length.

        Parameters
        ----------
        x : (B, L, D)
        target_len : int

        Returns
        -------
        (B, target_len, D)
        """
        if x.size(1) == target_len:
            return x
        if x.size(1) < target_len:
            # Upsample: repeat-interleave then trim
            repeat_factor = math.ceil(target_len / x.size(1))
            x = x.repeat_interleave(repeat_factor, dim=1)
            return x[:, :target_len, :]

        # Downsample: adaptive average pooling along time dimension
        x_perm = x.permute(0, 2, 1)  # (B, D, L)
        pooled = F.adaptive_avg_pool1d(x_perm, target_len)  # (B, D, target_len)
        return pooled.permute(0, 2, 1)  # (B, target_len, D)

    def forward(
        self,
        market_data: Dict[str, "torch.Tensor"],
    ) -> Dict[str, "torch.Tensor"]:
        """Full tokenization pipeline.

        Parameters
        ----------
        market_data : dict mapping scale name to raw tensor
            Expected keys: "micro", "meso", "macro"
            Each value: (B, L_scale, n_features_scale)

        Returns
        -------
        dict with:
            tokens : (B, L_out, d_model) -- fused multi-scale tokens
            vq_loss : scalar (0 if VQ disabled)
            encoding_indices : (B, L_out) or None
            perplexity : scalar or None
            micro_emb : (B, L_micro, d_model) -- per-scale embeddings
            meso_emb : (B, L_meso, d_model)
            macro_emb : (B, L_macro, d_model)
        """
        # Encode each scale
        micro_emb = self.encode_scale(market_data["micro"], "micro")
        meso_emb = self.encode_scale(market_data["meso"], "meso")
        macro_emb = self.encode_scale(market_data["macro"], "macro")

        # Cross-scale fusion
        fused = self.fuse_scales(micro_emb, meso_emb, macro_emb)

        # Optional VQ quantization
        vq_loss = torch.tensor(0.0, device=fused.device)
        encoding_indices = None
        perplexity = None

        if self.vq is not None:
            z_pre = self.pre_vq_proj(fused)
            quantized, vq_loss, encoding_indices, perplexity = self.vq(z_pre)
            fused = self.post_vq_proj(quantized)

        return {
            "tokens": fused,
            "vq_loss": vq_loss,
            "encoding_indices": encoding_indices,
            "perplexity": perplexity,
            "micro_emb": micro_emb,
            "meso_emb": meso_emb,
            "macro_emb": macro_emb,
        }


# ---------------------------------------------------------------------------
# Market Feature Extractor -- compute standard features from raw OHLCV
# ---------------------------------------------------------------------------


class MarketFeatureExtractor:
    """Computes standard causal features from raw OHLCV data.

    All features are strictly causal: they use only data available at or
    before time t.  Where a feature is derived from the current bar
    (e.g., realized volatility), the computation uses ``.shift(1)`` so
    that the value at time t reflects only information up to t-1.

    Features computed (per bar):
        1. log_return:    log(close_t / close_{t-1})
        2. rvol_5:        realized vol (5-day rolling std of log returns), shifted
        3. rvol_21:       realized vol (21-day), shifted
        4. volume_delta:  volume_t - volume_{t-1}
        5. hl_range:      (high - low) / close, shifted
        6. vwap_dev:      (close - vwap) / close, shifted (vwap = typical price)
        7. open_gap:      log(open_t / close_{t-1})

    The caller decides which features to use; this class computes all of
    them and returns a dict of numpy arrays or a torch tensor.

    Notes
    -----
    This class uses numpy/pandas for feature computation because it
    operates on raw DataFrames before the data enters the model. The
    torch model receives the output via NexusDataset.

    Parameters
    ----------
    rvol_windows : tuple of int
        Windows for realized volatility computation.
    """

    def __init__(self, rvol_windows: Tuple[int, ...] = (5, 21)):
        self.rvol_windows = rvol_windows

    def extract(self, df) -> dict:
        """Extract features from an OHLCV DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns: open, high, low, close, volume
            (case-insensitive).

        Returns
        -------
        dict of str -> np.ndarray
            Feature name -> feature values (shape (N,)).
            NaN-filled where insufficient history exists.
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for MarketFeatureExtractor")

        try:
            import pandas as pd  # noqa: F841
        except ImportError:
            raise ImportError("pandas is required for MarketFeatureExtractor")

        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        close = df["close"].values.astype(np.float64)
        open_ = (
            df["open"].values.astype(np.float64) if "open" in df.columns
            else close.copy()
        )
        high = (
            df["high"].values.astype(np.float64) if "high" in df.columns
            else close.copy()
        )
        low = (
            df["low"].values.astype(np.float64) if "low" in df.columns
            else close.copy()
        )
        volume = (
            df["volume"].values.astype(np.float64) if "volume" in df.columns
            else np.ones_like(close)
        )

        features = {}
        N = len(close)

        # 1. Log return: log(close_t / close_{t-1})
        log_ret = np.full(N, np.nan)
        log_ret[1:] = np.log(close[1:] / (close[:-1] + 1e-10) + 1e-10)
        features["log_return"] = log_ret

        # 2-3. Realized volatility (shifted by 1 for causality)
        for w in self.rvol_windows:
            rvol = np.full(N, np.nan)
            for i in range(w, N):
                window_rets = log_ret[i - w + 1 : i + 1]
                valid = window_rets[~np.isnan(window_rets)]
                if len(valid) >= 2:
                    rvol[i] = np.std(valid, ddof=1)
            # Shift by 1: rvol at time t uses returns up to t-1
            rvol_shifted = np.full(N, np.nan)
            rvol_shifted[1:] = rvol[:-1]
            features[f"rvol_{w}"] = rvol_shifted

        # 4. Volume delta: volume_t - volume_{t-1}
        vol_delta = np.full(N, np.nan)
        vol_delta[1:] = volume[1:] - volume[:-1]
        features["volume_delta"] = vol_delta

        # 5. High-low range: (high - low) / close, shifted
        hl_range = (high - low) / (close + 1e-10)
        hl_shifted = np.full(N, np.nan)
        hl_shifted[1:] = hl_range[:-1]
        features["hl_range"] = hl_shifted

        # 6. VWAP deviation: (close - vwap) / close, shifted
        # Approximate VWAP as (high + low + close) / 3 (typical price)
        typical = (high + low + close) / 3.0
        vwap_dev = (close - typical) / (close + 1e-10)
        vwap_shifted = np.full(N, np.nan)
        vwap_shifted[1:] = vwap_dev[:-1]
        features["vwap_dev"] = vwap_shifted

        # 7. Open gap: log(open_t / close_{t-1})
        open_gap = np.full(N, np.nan)
        open_gap[1:] = np.log(open_[1:] / (close[:-1] + 1e-10) + 1e-10)
        features["open_gap"] = open_gap

        return features

    def extract_tensor(self, df) -> "torch.Tensor":
        """Extract features and return as a torch tensor.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain OHLCV columns.

        Returns
        -------
        torch.Tensor of shape (N, n_features)
            Features in order: log_return, rvol_5, rvol_21, volume_delta,
            hl_range, vwap_dev, open_gap.
            NaN values are replaced with 0.0.
        """
        if not HAS_TORCH:
            raise ImportError("torch is required for extract_tensor")

        features = self.extract(df)
        # Stack in consistent order
        feature_names = ["log_return"]
        for w in self.rvol_windows:
            feature_names.append(f"rvol_{w}")
        feature_names.extend(["volume_delta", "hl_range", "vwap_dev", "open_gap"])

        arrays = []
        for name in feature_names:
            arr = features[name]
            arr = np.nan_to_num(arr, nan=0.0)
            arrays.append(arr)

        stacked = np.stack(arrays, axis=-1)  # (N, n_features)
        return torch.tensor(stacked, dtype=torch.float32)
