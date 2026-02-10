"""X-Trend: Cross-Attentive Few-Shot Learning for Trend-Following.

Implementation of Wood et al. (2023), arXiv:2310.10500.

Key innovations over the Momentum Transformer (momentum_tfm.py):
1. Uses 8 lean features (5 normalized returns + 3 MACDs) instead of 287.
2. Cross-attention over context sets of past market regimes.
3. Entity embeddings for multi-asset joint training.
4. Joint MLE + Sharpe loss for better gradient signal.
5. Changepoint detection for regime segmentation.

Architecture
------------
XTrendModel
├── EntityEmbedding           — learnable per-asset embedding
├── VariableSelectionNetwork   — softmax-weighted per-feature GRNs (8 features)
├── LSTMEncoder               — 2-layer LSTM with entity-initialized h0/c0
├── ContextEncoder             — encodes support set sequences
├── CrossAttention             — Q from target, K/V from context set
├── Decoder                    — fuses encoder + cross-attention outputs
└── OutputHead                 — Tanh position or Gaussian (μ, σ) for MLE
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    _DEVICE = None

try:
    import ruptures

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class XTrendConfig:
    """Hyperparameters for the X-Trend model."""

    # Architecture
    d_hidden: int = 64
    n_heads: int = 4
    lstm_layers: int = 2
    dropout: float = 0.1
    n_features: int = 24  # 8 paper + 16 intraday

    # Sequence lengths
    seq_len: int = 42  # ~2 months target sequence
    ctx_len: int = 42  # context sequence length
    n_context: int = 16  # context set size |C|

    # Multi-asset
    n_assets: int = 4  # NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY

    # Training
    train_window: int = 150  # ~7 months rolling train
    test_window: int = 42  # ~2 months test
    step_size: int = 21  # monthly step
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    batch_size: int = 32

    # Loss
    loss_mode: str = "sharpe"  # "sharpe", "joint_mle", "joint_quantile"
    mle_weight: float = 0.1  # α in joint loss

    # Purge gap
    purge_gap: int = 5  # days between train_end and test_start to avoid look-ahead

    # Position sizing
    max_position: float = 0.25
    vol_target: float = 0.15  # 15% annual vol target
    cost_bps: float = 5.0
    position_smooth: float = 0.3
    max_daily_turnover: float = 0.5


# ============================================================================
# PyTorch modules
# ============================================================================

if HAS_TORCH:

    class GatedResidualNetwork(nn.Module):
        """Gated Residual Network — GRN(a, c) = LN(a + GLU(W1·ELU(W2·a + W3·c)))."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            context_dim: Optional[int] = None,
        ):
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

        def forward(
            self, x: torch.Tensor, context: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            residual = self.skip(x)
            h = self.fc1(x)
            if self.context_proj is not None and context is not None:
                # Broadcast context to match h shape
                if context.dim() < h.dim():
                    context = context.unsqueeze(-2).expand_as(h)
                h = h + self.context_proj(context)
            h = F.elu(h)
            h = self.dropout(h)
            gate_input = self.fc2(h)
            value, gate = gate_input.chunk(2, dim=-1)
            h = value * torch.sigmoid(gate)
            return self.layer_norm(residual + h)

    class VariableSelectionNetwork(nn.Module):
        """VSN with optional entity context for per-feature weighting."""

        def __init__(
            self,
            n_features: int,
            hidden_dim: int,
            dropout: float = 0.1,
            context_dim: Optional[int] = None,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_dim = hidden_dim

            self.feature_grns = nn.ModuleList(
                [
                    GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
                    for _ in range(n_features)
                ]
            )
            self.weight_grn = GatedResidualNetwork(
                n_features,
                hidden_dim,
                n_features,
                dropout,
                context_dim=context_dim,
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(
            self, x: torch.Tensor, context: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (batch, seq_len, n_features)
            context : (batch, d_h) optional entity embedding

            Returns
            -------
            (batch, seq_len, hidden_dim)
            """
            # Variable selection weights
            weights = self.softmax(self.weight_grn(x, context))

            # Transform each feature independently
            transformed = []
            for i in range(self.n_features):
                feat_i = x[:, :, i : i + 1]
                transformed.append(self.feature_grns[i](feat_i))

            # Stack: (batch, seq_len, n_features, hidden_dim)
            transformed = torch.stack(transformed, dim=2)

            # Weight and sum
            weights_expanded = weights.unsqueeze(-1)
            output = (transformed * weights_expanded).sum(dim=2)
            return output

    class EntityEmbedding(nn.Module):
        """Learnable per-asset embedding for multi-asset joint training."""

        def __init__(self, n_assets: int, d_hidden: int):
            super().__init__()
            self.embedding = nn.Embedding(n_assets, d_hidden)

        def forward(self, asset_ids: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            asset_ids : (batch,) or (batch, n_context) — integer asset indices

            Returns
            -------
            (batch, d_hidden) or (batch, n_context, d_hidden)
            """
            return self.embedding(asset_ids)

    class CrossAttentionBlock(nn.Module):
        """Cross-attention: Q from target, K/V from context set.

        Includes self-attention on V (context) to identify similar regimes,
        then cross-attention Q×K→V' to transfer patterns to target.
        """

        def __init__(self, d_hidden: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_hidden,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_hidden,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(d_hidden)
            self.norm2 = nn.LayerNorm(d_hidden)

        def forward(
            self,
            query: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            query : (batch, d_hidden) — target representation
            keys : (batch, n_context, d_hidden) — context summaries
            values : (batch, n_context, d_hidden) — context values

            Returns
            -------
            (batch, d_hidden) — cross-attended representation
            """
            # Self-attention on context V to find similar regimes
            v_prime, _ = self.self_attn(values, values, values)
            v_prime = self.norm1(values + v_prime)

            # Cross-attention: target queries context
            q = query.unsqueeze(1)  # (batch, 1, d_hidden)
            out, _ = self.cross_attn(q, keys, v_prime)
            out = self.norm2(q + out)
            return out.squeeze(1)  # (batch, d_hidden)

        def forward_with_weights(
            self,
            query: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass that also returns cross-attention weights.

            Parameters
            ----------
            query : (batch, d_hidden) — target representation
            keys : (batch, n_context, d_hidden) — context summaries
            values : (batch, n_context, d_hidden) — context values

            Returns
            -------
            output : (batch, d_hidden) — cross-attended representation
            attn_weights : (batch, n_heads, 1, n_context) — attention weights
            """
            # Self-attention on context V
            v_prime, _ = self.self_attn(values, values, values)
            v_prime = self.norm1(values + v_prime)

            # Cross-attention with weights
            q = query.unsqueeze(1)  # (batch, 1, d_hidden)
            out, attn_weights = self.cross_attn(
                q, keys, v_prime, need_weights=True, average_attn_weights=False,
            )
            out = self.norm2(q + out)
            return out.squeeze(1), attn_weights  # (batch, d), (batch, n_heads, 1, n_ctx)

    class XTrendModel(nn.Module):
        """Full X-Trend architecture.

        Forward pass:
        1. Entity embed target + context assets
        2. VSN with entity context
        3. LSTM encode target + each context sequence
        4. Cross-attention: target queries context summaries
        5. Decode fused representation
        6. Output position ∈ [-1, 1]
        """

        def __init__(self, cfg: XTrendConfig):
            super().__init__()
            self.cfg = cfg
            d = cfg.d_hidden

            # Entity embedding
            self.entity_embed = EntityEmbedding(cfg.n_assets, d)

            # h0, c0 projections from entity embedding
            self.h0_proj = nn.Linear(d, d * cfg.lstm_layers)
            self.c0_proj = nn.Linear(d, d * cfg.lstm_layers)

            # Variable Selection Network (8 features → d_hidden)
            self.vsn = VariableSelectionNetwork(
                cfg.n_features, d, cfg.dropout, context_dim=d
            )

            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size=d,
                hidden_size=d,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            )
            self.lstm_norm = nn.LayerNorm(d)

            # Cross-attention block
            self.cross_attn = CrossAttentionBlock(d, cfg.n_heads, cfg.dropout)

            # Decoder: fuses target LSTM output + cross-attention output
            self.decoder = nn.Sequential(
                nn.Linear(2 * d, d),
                nn.ELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(d),
            )

            # Output heads
            self.position_head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.ELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d // 2, 1),
                nn.Tanh(),
            )

            # Gaussian head for MLE loss mode
            self.gaussian_mu = nn.Linear(d, 1)
            self.gaussian_log_sigma = nn.Linear(d, 1)

        def _init_hidden(
            self, entity_emb: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Initialize LSTM h0, c0 from entity embedding.

            Parameters
            ----------
            entity_emb : (batch, d_hidden)

            Returns
            -------
            h0, c0 : each (n_layers, batch, d_hidden)
            """
            batch = entity_emb.size(0)
            d = self.cfg.d_hidden
            n_layers = self.cfg.lstm_layers

            h0 = self.h0_proj(entity_emb).view(batch, n_layers, d).permute(1, 0, 2).contiguous()
            c0 = self.c0_proj(entity_emb).view(batch, n_layers, d).permute(1, 0, 2).contiguous()
            return h0, c0

        def _encode_sequence(
            self,
            seq: torch.Tensor,
            entity_emb: torch.Tensor,
        ) -> torch.Tensor:
            """Encode a sequence through VSN + LSTM.

            Parameters
            ----------
            seq : (batch, seq_len, n_features)
            entity_emb : (batch, d_hidden)

            Returns
            -------
            (batch, seq_len, d_hidden)
            """
            vsn_out = self.vsn(seq, entity_emb)
            h0, c0 = self._init_hidden(entity_emb)
            lstm_out, _ = self.lstm(vsn_out, (h0, c0))
            return self.lstm_norm(lstm_out)

        def forward(
            self,
            target_seq: torch.Tensor,
            context_set: torch.Tensor,
            target_id: torch.Tensor,
            context_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            target_seq : (batch, seq_len, 8)
            context_set : (batch, n_context, ctx_len, 8)
            target_id : (batch,) — integer asset indices
            context_ids : (batch, n_context) — integer asset indices

            Returns
            -------
            position : (batch, 1) ∈ [-1, 1]  (sharpe mode)
            OR (mu, log_sigma) : (batch, 1), (batch, 1)  (mle mode)
            """
            batch = target_seq.size(0)
            n_ctx = context_set.size(1)
            d = self.cfg.d_hidden

            # 1. Entity embeddings
            s_target = self.entity_embed(target_id)  # (batch, d)

            # 2. Encode target sequence
            h_target = self._encode_sequence(target_seq, s_target)  # (batch, seq_len, d)
            h_target_last = h_target[:, -1, :]  # (batch, d)

            # 3. Encode each context sequence
            ctx_keys = []
            ctx_vals = []
            for c in range(n_ctx):
                ctx_seq_c = context_set[:, c, :, :]  # (batch, ctx_len, 8)
                ctx_id_c = context_ids[:, c]  # (batch,)
                s_ctx_c = self.entity_embed(ctx_id_c)  # (batch, d)
                h_c = self._encode_sequence(ctx_seq_c, s_ctx_c)  # (batch, ctx_len, d)
                ctx_keys.append(h_c[:, -1, :])  # (batch, d)
                ctx_vals.append(h_c[:, -1, :])  # (batch, d)

            K = torch.stack(ctx_keys, dim=1)  # (batch, n_ctx, d)
            V = torch.stack(ctx_vals, dim=1)  # (batch, n_ctx, d)

            # 4. Cross-attention
            y = self.cross_attn(h_target_last, K, V)  # (batch, d)

            # 5. Decode
            h_fused = torch.cat([h_target_last, y], dim=-1)  # (batch, 2d)
            decoded = self.decoder(h_fused)  # (batch, d)

            # 6. Output
            if self.cfg.loss_mode in ("joint_mle", "joint_quantile"):
                mu = self.gaussian_mu(decoded)  # (batch, 1)
                log_sigma = self.gaussian_log_sigma(decoded)  # (batch, 1)
                return mu, log_sigma
            else:
                position = self.position_head(decoded)  # (batch, 1)
                return position

        def extract_hidden(
            self,
            target_seq: torch.Tensor,
            context_set: torch.Tensor,
            target_id: torch.Tensor,
            context_ids: torch.Tensor,
        ) -> torch.Tensor:
            """Return decoder hidden state (batch, d_hidden) — for RL backbone mode.

            Same forward path as forward() but returns the decoder output
            *before* the position/Gaussian output heads.  This provides a
            rich, compressed representation of the market state suitable for
            feeding into an RL decision layer.

            Parameters
            ----------
            target_seq : (batch, seq_len, n_features)
            context_set : (batch, n_context, ctx_len, n_features)
            target_id : (batch,) — integer asset indices
            context_ids : (batch, n_context) — integer asset indices

            Returns
            -------
            decoded : (batch, d_hidden)
            """
            batch = target_seq.size(0)
            n_ctx = context_set.size(1)
            d = self.cfg.d_hidden

            # 1. Entity embeddings
            s_target = self.entity_embed(target_id)  # (batch, d)

            # 2. Encode target sequence
            h_target = self._encode_sequence(target_seq, s_target)  # (batch, seq_len, d)
            h_target_last = h_target[:, -1, :]  # (batch, d)

            # 3. Encode each context sequence
            ctx_keys = []
            ctx_vals = []
            for c in range(n_ctx):
                ctx_seq_c = context_set[:, c, :, :]  # (batch, ctx_len, n_features)
                ctx_id_c = context_ids[:, c]  # (batch,)
                s_ctx_c = self.entity_embed(ctx_id_c)  # (batch, d)
                h_c = self._encode_sequence(ctx_seq_c, s_ctx_c)  # (batch, ctx_len, d)
                ctx_keys.append(h_c[:, -1, :])  # (batch, d)
                ctx_vals.append(h_c[:, -1, :])  # (batch, d)

            K = torch.stack(ctx_keys, dim=1)  # (batch, n_ctx, d)
            V = torch.stack(ctx_vals, dim=1)  # (batch, n_ctx, d)

            # 4. Cross-attention
            y = self.cross_attn(h_target_last, K, V)  # (batch, d)

            # 5. Decode — return before output heads
            h_fused = torch.cat([h_target_last, y], dim=-1)  # (batch, 2d)
            decoded = self.decoder(h_fused)  # (batch, d)

            return decoded

        def extract_hidden_with_attention(
            self,
            target_seq: torch.Tensor,
            context_set: torch.Tensor,
            target_id: torch.Tensor,
            context_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return decoder hidden AND cross-attention weights.

            Same as extract_hidden() but uses forward_with_weights() on the
            CrossAttentionBlock to also return attention weights for Pattern 5
            (attention reward shaping).

            Returns
            -------
            decoded : (batch, d_hidden)
            attn_weights : (batch, n_heads, 1, n_context)
            """
            batch = target_seq.size(0)
            n_ctx = context_set.size(1)
            d = self.cfg.d_hidden

            # 1. Entity embeddings
            s_target = self.entity_embed(target_id)

            # 2. Encode target sequence
            h_target = self._encode_sequence(target_seq, s_target)
            h_target_last = h_target[:, -1, :]

            # 3. Encode each context sequence
            ctx_keys = []
            ctx_vals = []
            for c in range(n_ctx):
                ctx_seq_c = context_set[:, c, :, :]
                ctx_id_c = context_ids[:, c]
                s_ctx_c = self.entity_embed(ctx_id_c)
                h_c = self._encode_sequence(ctx_seq_c, s_ctx_c)
                ctx_keys.append(h_c[:, -1, :])
                ctx_vals.append(h_c[:, -1, :])

            K = torch.stack(ctx_keys, dim=1)
            V = torch.stack(ctx_vals, dim=1)

            # 4. Cross-attention with weights
            y, attn_weights = self.cross_attn.forward_with_weights(h_target_last, K, V)

            # 5. Decode
            h_fused = torch.cat([h_target_last, y], dim=-1)
            decoded = self.decoder(h_fused)

            return decoded, attn_weights

        def predict_position(
            self,
            target_seq: torch.Tensor,
            context_set: torch.Tensor,
            target_id: torch.Tensor,
            context_ids: torch.Tensor,
        ) -> torch.Tensor:
            """Always return position ∈ [-1, 1], regardless of loss mode.

            For Gaussian mode, uses PTP mapping: pos = 2*Φ(μ/σ) - 1.
            """
            out = self.forward(target_seq, context_set, target_id, context_ids)
            if isinstance(out, tuple):
                mu, log_sigma = out
                sigma = torch.exp(log_sigma).clamp(min=1e-6)
                # PTP: position = 2 * Φ(μ/σ) - 1
                z = mu / sigma
                prob_up = torch.distributions.Normal(0, 1).cdf(z)
                position = 2.0 * prob_up - 1.0
                return position
            return out

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def sharpe_loss(
        positions: torch.Tensor,
        returns: torch.Tensor,
        annualize: float = math.sqrt(252),
    ) -> torch.Tensor:
        """Differentiable Sharpe ratio loss (negative Sharpe).

        Parameters
        ----------
        positions : (batch,) — position signals ∈ [-1, 1]
        returns : (batch,) — vol-scaled next-day returns
        annualize : float

        Returns
        -------
        Scalar: -Sharpe
        """
        strategy_returns = positions * returns
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std(correction=1)
        if std_ret < 1e-8:
            # Return a small differentiable penalty that encourages
            # the model to take positions
            return -torch.abs(positions).mean() * 0.01
        return -(mean_ret / std_ret) * annualize

    def gaussian_nll_loss(
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood.

        L = -Σ log N(r | μ, σ²) = Σ [log σ + 0.5 * ((r - μ)/σ)²]
        """
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        nll = log_sigma + 0.5 * ((targets - mu) / sigma) ** 2
        return nll.mean()

    def joint_loss(
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """Joint MLE + Sharpe loss (Eq. 21 in Wood et al. 2023).

        L = α * L_MLE + L_Sharpe

        Uses PTP mapping to convert Gaussian output to positions for Sharpe.
        """
        # MLE component
        l_mle = gaussian_nll_loss(mu, log_sigma, targets)

        # PTP → position → Sharpe
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        z = mu / sigma
        prob_up = torch.distributions.Normal(0, 1).cdf(z)
        positions = (2.0 * prob_up - 1.0).squeeze(-1)
        l_sharpe = sharpe_loss(positions, targets.squeeze(-1))

        return alpha * l_mle + l_sharpe

    # ------------------------------------------------------------------
    # Context set construction
    # ------------------------------------------------------------------

    def build_context_set(
        features: np.ndarray,
        target_start: int,
        n_context: int,
        ctx_len: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct causal context set from historical data.

        Parameters
        ----------
        features : (n_days, n_assets, 8)
        target_start : int — first day of target sequence (exclusive upper bound for context)
        n_context : int — number of context sequences to sample
        ctx_len : int — length of each context sequence
        rng : numpy RNG

        Returns
        -------
        ctx_seqs : (n_context, ctx_len, 8) — context sequences
        ctx_ids : (n_context,) — asset IDs for each context sequence
        """
        n_days, n_assets, n_feat = features.shape

        # Available context pool: all (asset, start_idx) pairs where
        # start_idx + ctx_len <= target_start
        max_start = target_start - ctx_len
        if max_start <= 0:
            # Not enough history — return zeros
            ctx_seqs = np.zeros((n_context, ctx_len, n_feat))
            ctx_ids = np.zeros(n_context, dtype=np.int64)
            return ctx_seqs, ctx_ids

        # Build pool of candidate (asset, start) pairs
        candidates = []
        for asset_idx in range(n_assets):
            for start in range(0, max_start + 1):
                window = features[start : start + ctx_len, asset_idx, :]
                # Only include if no NaN in the window
                if not np.any(np.isnan(window)):
                    candidates.append((asset_idx, start))

        if len(candidates) == 0:
            ctx_seqs = np.zeros((n_context, ctx_len, n_feat))
            ctx_ids = np.zeros(n_context, dtype=np.int64)
            return ctx_seqs, ctx_ids

        # Sample with replacement if needed
        n_sample = min(n_context, len(candidates))
        replace = n_context > len(candidates)
        chosen_indices = rng.choice(len(candidates), size=n_context, replace=replace)

        ctx_seqs = np.zeros((n_context, ctx_len, n_feat))
        ctx_ids = np.zeros(n_context, dtype=np.int64)

        for i, idx in enumerate(chosen_indices):
            asset_idx, start = candidates[idx]
            ctx_seqs[i] = features[start : start + ctx_len, asset_idx, :]
            ctx_ids[i] = asset_idx

        return ctx_seqs, ctx_ids

    def segment_regimes_pelt(
        close: np.ndarray, penalty: float = 3.0, min_size: int = 21
    ) -> list[tuple[int, int]]:
        """Segment price series into regimes via PELT changepoint detection.

        Returns list of (start, end) index pairs for each regime.
        Falls back to fixed-size windows if ruptures unavailable.
        """
        n = len(close)

        if not HAS_RUPTURES or n < 2 * min_size:
            # Fixed windows fallback
            segments = []
            start = 0
            while start + min_size <= n:
                end = min(start + min_size * 3, n)
                segments.append((start, end))
                start = end
            return segments

        log_returns = np.diff(np.log(close))
        try:
            algo = ruptures.Pelt(model="rbf", min_size=min_size).fit(
                log_returns.reshape(-1, 1)
            )
            changepoints = algo.predict(pen=penalty)
        except Exception:
            logger.debug("PELT failed; using fixed windows")
            segments = []
            start = 0
            while start + min_size <= n:
                end = min(start + min_size * 3, n)
                segments.append((start, end))
                start = end
            return segments

        # Convert changepoints to segments (changepoints are 1-based end indices)
        segments = []
        prev = 0
        for cp in changepoints:
            cp = min(cp, n)  # clamp to array length
            if cp > prev:
                segments.append((prev, cp))
            prev = cp
        if prev < n:
            segments.append((prev, n))

        return segments

    # ------------------------------------------------------------------
    # Walk-forward backtest
    # ------------------------------------------------------------------

    def run_xtrend_backtest(
        prices_df: pd.DataFrame,
        symbols: list[str],
        cfg: Optional[XTrendConfig] = None,
    ) -> dict[str, pd.DataFrame]:
        """Run X-Trend walk-forward backtest on multiple assets.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Must have a 'date' column and one column per symbol.
        symbols : list of str
            Asset column names.
        cfg : XTrendConfig, optional

        Returns
        -------
        dict mapping symbol → DataFrame with columns:
            [date, position, actual_return, strategy_return, cumulative_pnl]
        """
        if cfg is None:
            cfg = XTrendConfig()

        from quantlaxmi.features.trend import TrendFeatureBuilder
        from quantlaxmi.data._paths import KITE_1MIN_DIR

        builder = TrendFeatureBuilder()
        kite_dir = KITE_1MIN_DIR if KITE_1MIN_DIR.exists() else None
        features, targets, vol = builder.build(prices_df, symbols, kite_1min_dir=kite_dir)
        # features: (n_days, n_assets, n_features)  — 24 if intraday available, else 8
        # targets: (n_days, n_assets)

        n_days, n_assets, n_feat = features.shape
        cfg.n_assets = n_assets
        cfg.n_features = n_feat  # 24 if intraday data found, else 8

        logger.info(
            "X-Trend backtest: %d days, %d assets, features shape %s",
            n_days, n_assets, features.shape,
        )

        # All-asset positions (initially NaN)
        all_positions = np.full((n_days, n_assets), np.nan)

        rng = np.random.default_rng(42)

        # Walk-forward
        train_size = cfg.train_window
        test_size = cfg.test_window
        step = cfg.step_size
        seq_len = cfg.seq_len

        fold_idx = 0
        start = 0

        # Find first valid start: need enough history for features to be non-NaN
        # Feature warmup: longest MACD EMA (96) + MACD_NORM (63) + MACD_STD (126) = 285
        # Plus normalized return 252d. Plus seq_len for full window.
        from quantlaxmi.features.trend import MACD_STD_WINDOW, MACD_NORM_WINDOW, RETURN_WINDOWS
        feature_warmup = max(
            max(RETURN_WINDOWS),
            max(pair[1] for pair in [(8, 24), (16, 48), (32, 96)]) + MACD_NORM_WINDOW + MACD_STD_WINDOW,
        )
        min_history = feature_warmup + seq_len + 10
        purge_gap = cfg.purge_gap
        while start + train_size + purge_gap + test_size <= n_days:
            train_end = start + train_size
            test_start = train_end + purge_gap
            test_end = min(test_start + test_size, n_days)

            if train_end < min_history:
                start += step
                continue

            logger.info(
                "Fold %d: train=[%d:%d], purge=%d, test=[%d:%d]",
                fold_idx, start, train_end, purge_gap, test_start, test_end,
            )

            # --- Normalize features using train stats ---
            train_feats = features[start:train_end]  # (train_size, n_assets, 8)
            # Compute mean/std per feature across all assets and time
            flat_train = train_feats.reshape(-1, cfg.n_features)
            # Remove NaN rows
            valid_mask = ~np.any(np.isnan(flat_train), axis=1)
            if valid_mask.sum() < 30:
                logger.warning("Fold %d: insufficient valid training samples", fold_idx)
                start += step
                fold_idx += 1
                continue

            feat_mean = np.nanmean(flat_train[valid_mask], axis=0)
            feat_std = np.nanstd(flat_train[valid_mask], axis=0, ddof=1)
            feat_std = np.where(feat_std > 1e-10, feat_std, 1.0)

            # Normalize full range
            norm_features = (features - feat_mean) / feat_std

            # --- Build training episodes ---
            train_targets = targets[start:train_end]  # (train_size, n_assets)

            # Collect valid (target_seq, target, context) episodes
            X_target = []
            X_context = []
            X_target_id = []
            X_context_id = []
            Y = []

            for asset_idx in range(n_assets):
                for t in range(start + seq_len, train_end):
                    target_window = norm_features[t - seq_len : t, asset_idx, :]
                    if np.any(np.isnan(target_window)):
                        continue
                    tgt = targets[t, asset_idx]
                    if np.isnan(tgt):
                        continue

                    # Build context set (causal: before target start)
                    ctx_seqs, ctx_ids = build_context_set(
                        norm_features,
                        target_start=t - seq_len,
                        n_context=cfg.n_context,
                        ctx_len=cfg.ctx_len,
                        rng=rng,
                    )

                    X_target.append(target_window)
                    X_context.append(ctx_seqs)
                    X_target_id.append(asset_idx)
                    X_context_id.append(ctx_ids)
                    Y.append(tgt)

            if len(X_target) < 10:
                logger.warning(
                    "Fold %d: only %d episodes, skipping", fold_idx, len(X_target)
                )
                start += step
                fold_idx += 1
                continue

            X_target_arr = np.array(X_target, dtype=np.float32)
            X_context_arr = np.array(X_context, dtype=np.float32)
            X_target_id_arr = np.array(X_target_id, dtype=np.int64)
            X_context_id_arr = np.array(X_context_id, dtype=np.int64)
            Y_arr = np.array(Y, dtype=np.float32)

            # --- Train model ---
            model = XTrendModel(cfg).to(_DEVICE)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )

            # AMP setup
            use_amp = _DEVICE is not None and _DEVICE.type == "cuda"
            scaler = torch.amp.GradScaler("cuda") if use_amp else None

            # Split into train/val (last 20%)
            n_total = len(X_target_arr)
            n_val = max(1, int(n_total * 0.2))
            n_train = n_total - n_val

            best_val_metric = -np.inf
            patience_counter = 0
            best_state = None

            for epoch in range(cfg.epochs):
                model.train()

                # Shuffle training indices
                perm = rng.permutation(n_train)

                epoch_loss = 0.0
                n_batches = 0

                for batch_start in range(0, n_train, cfg.batch_size):
                    batch_idx = perm[batch_start : batch_start + cfg.batch_size]

                    tgt_seq = torch.tensor(
                        X_target_arr[batch_idx], device=_DEVICE
                    )
                    ctx_set = torch.tensor(
                        X_context_arr[batch_idx], device=_DEVICE
                    )
                    tgt_id = torch.tensor(
                        X_target_id_arr[batch_idx], device=_DEVICE
                    )
                    ctx_id = torch.tensor(
                        X_context_id_arr[batch_idx], device=_DEVICE
                    )
                    y_batch = torch.tensor(Y_arr[batch_idx], device=_DEVICE)

                    optimizer.zero_grad()

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            if cfg.loss_mode == "joint_mle":
                                mu, log_sigma = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                                loss = joint_loss(
                                    mu, log_sigma, y_batch.unsqueeze(-1), cfg.mle_weight
                                )
                            else:
                                positions = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                                loss = sharpe_loss(positions.squeeze(-1), y_batch)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if cfg.loss_mode == "joint_mle":
                            mu, log_sigma = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                            loss = joint_loss(
                                mu, log_sigma, y_batch.unsqueeze(-1), cfg.mle_weight
                            )
                        else:
                            positions = model(tgt_seq, ctx_set, tgt_id, ctx_id)
                            loss = sharpe_loss(positions.squeeze(-1), y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                # Validation
                if n_val > 0:
                    model.eval()
                    with torch.no_grad():
                        val_idx = np.arange(n_train, n_total)
                        v_tgt = torch.tensor(
                            X_target_arr[val_idx], device=_DEVICE
                        )
                        v_ctx = torch.tensor(
                            X_context_arr[val_idx], device=_DEVICE
                        )
                        v_tid = torch.tensor(
                            X_target_id_arr[val_idx], device=_DEVICE
                        )
                        v_cid = torch.tensor(
                            X_context_id_arr[val_idx], device=_DEVICE
                        )
                        v_y = torch.tensor(Y_arr[val_idx], device=_DEVICE)

                        v_pos = model.predict_position(
                            v_tgt, v_ctx, v_tid, v_cid
                        ).squeeze(-1)
                        strat_ret = v_pos * v_y
                        mean_r = strat_ret.mean().item()
                        std_r = strat_ret.std(correction=1).item()
                        val_sharpe = (
                            mean_r / max(std_r, 1e-8)
                        ) * math.sqrt(252)

                    if val_sharpe > best_val_metric:
                        best_val_metric = val_sharpe
                        patience_counter = 0
                        best_state = {
                            k: v.cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.patience:
                            logger.debug(
                                "Fold %d: early stop at epoch %d (val_sharpe=%.3f)",
                                fold_idx, epoch, best_val_metric,
                            )
                            break

            # Restore best model
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(_DEVICE)

            # --- Predict on test set ---
            model.eval()
            with torch.no_grad():
                for asset_idx in range(n_assets):
                    for t in range(test_start, test_end):
                        if t < seq_len:
                            continue
                        target_window = norm_features[t - seq_len : t, asset_idx, :]
                        if np.any(np.isnan(target_window)):
                            continue

                        ctx_seqs, ctx_ids = build_context_set(
                            norm_features,
                            target_start=t - seq_len,
                            n_context=cfg.n_context,
                            ctx_len=cfg.ctx_len,
                            rng=rng,
                        )

                        tgt_t = torch.tensor(
                            target_window[np.newaxis], dtype=torch.float32, device=_DEVICE
                        )
                        ctx_t = torch.tensor(
                            ctx_seqs[np.newaxis], dtype=torch.float32, device=_DEVICE
                        )
                        tid_t = torch.tensor(
                            [asset_idx], dtype=torch.long, device=_DEVICE
                        )
                        cid_t = torch.tensor(
                            ctx_ids[np.newaxis], dtype=torch.long, device=_DEVICE
                        )

                        pos = model.predict_position(tgt_t, ctx_t, tid_t, cid_t)
                        all_positions[t, asset_idx] = pos.item()

            start += step
            fold_idx += 1

            # Cleanup
            del model
            torch.cuda.empty_cache()

        # --- Assemble results ---
        dates = prices_df["date"].values
        results: dict[str, pd.DataFrame] = {}

        for j, sym in enumerate(symbols):
            close_j = prices_df[sym].values.astype(np.float64)
            log_close_j = np.log(close_j)

            pos_j = all_positions[:, j]

            # Position sizing
            pos_j = np.clip(pos_j, -cfg.max_position, cfg.max_position)

            # Smooth
            smoothed = np.full_like(pos_j, np.nan)
            prev_pos = 0.0
            for i in range(n_days):
                if np.isnan(pos_j[i]):
                    smoothed[i] = np.nan
                    continue
                raw = pos_j[i]
                new_pos = (1.0 - cfg.position_smooth) * prev_pos + cfg.position_smooth * raw
                delta = new_pos - prev_pos
                if abs(delta) > cfg.max_daily_turnover:
                    new_pos = prev_pos + np.sign(delta) * cfg.max_daily_turnover
                new_pos = np.clip(new_pos, -cfg.max_position, cfg.max_position)
                smoothed[i] = new_pos
                prev_pos = new_pos
            pos_j = smoothed

            # Actual returns
            actual_ret = np.full(n_days, np.nan)
            actual_ret[1:] = log_close_j[1:] - log_close_j[:-1]

            # Strategy return with T+1 lag
            strat_ret = np.full(n_days, np.nan)
            for i in range(n_days - 1):
                if not np.isnan(pos_j[i]) and not np.isnan(actual_ret[i + 1]):
                    turnover = (
                        abs(pos_j[i] - pos_j[i - 1])
                        if i > 0 and not np.isnan(pos_j[i - 1])
                        else abs(pos_j[i])
                    )
                    cost = turnover * cfg.cost_bps / 10_000.0
                    strat_ret[i + 1] = pos_j[i] * actual_ret[i + 1] - cost

            simple_strat = np.expm1(np.nan_to_num(strat_ret, nan=0.0))
            cum_pnl = np.cumprod(1.0 + simple_strat) - 1.0

            results[sym] = pd.DataFrame(
                {
                    "date": dates,
                    "position": pos_j,
                    "actual_return": actual_ret,
                    "strategy_return": strat_ret,
                    "cumulative_pnl": cum_pnl,
                }
            )

            # Log stats
            valid = pd.Series(strat_ret).dropna()
            if len(valid) > 1:
                sv = np.expm1(valid.values)
                sh = (np.mean(sv) / np.std(sv, ddof=1)) * math.sqrt(252)
                tr = cum_pnl[-1]
                logger.info(
                    "%s: Sharpe=%.3f, Return=%.2f%%, OOS=%d days",
                    sym, sh, tr * 100, len(valid),
                )

        return results

else:
    # No torch — provide a stub
    class XTrendConfig:  # type: ignore[no-redef]
        pass

    def run_xtrend_backtest(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("PyTorch is required for X-Trend backtest")
