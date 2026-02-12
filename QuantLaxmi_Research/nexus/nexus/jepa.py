"""JEPA World Model for Financial Markets.

Joint Embedding Predictive Architecture adapted for time series.
Instead of predicting future PRICES (noisy, unpredictable), NEXUS predicts
future LATENT REPRESENTATIONS of market states (semantic, learnable).

This is fundamentally different from:
- Autoregressive models (GPT): predict next token in observation space
- VAE/diffusion: reconstruct observations from latent (generative)
- Contrastive learning: push apart negatives (needs careful sampling)

JEPA predicts in REPRESENTATION SPACE:
    Context (visible past) → Encoder → z_context
    Target (masked future) → EMA Encoder → z_target (no gradient!)
    Predictor: z_context → ẑ_target
    Loss: ||ẑ_target - z_target||²

Why this is better for markets:
1. Latent space abstracts away noise (individual tick fluctuations don't matter)
2. EMA target encoder provides a stable training signal
3. No mode collapse (unlike contrastive methods, no negatives needed)
4. The predictor learns MARKET DYNAMICS in latent space (a world model!)

Architecture:
    ContextEncoder (Mamba-2)  ──→  z_context  ──→  Predictor  ──→  ẑ_target
    TargetEncoder  (EMA)      ──→  z_target  ──→  ||ẑ - z||²  (JEPA loss)

References:
    I-JEPA: Assran et al., CVPR 2023
    V-JEPA 2: Meta AI, June 2025
    TS-JEPA: "Joint Embeddings Go Temporal", NeurIPS 2024
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2 import Mamba2Backbone
from .hyperbolic import EuclideanToLorentz, LorentzToEuclidean, lorentz_distance


# ---------------------------------------------------------------------------
# Predictor Network
# ---------------------------------------------------------------------------

class JEPAPredictor(nn.Module):
    """Predicts target latent representations from context.

    Takes the context encoding and positional information of masked
    target positions, and predicts what the target encoder would produce.

    This is a lightweight MLP with residual connections — the "world model"
    that learns market dynamics in latent space.
    """

    def __init__(
        self,
        d_latent: int,
        depth: int = 4,
        d_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent

        # Positional encoding for target positions
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, d_latent) * 0.02)

        # Context aggregation (pool over sequence)
        self.context_proj = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Prediction MLP with residual connections
        layers = []
        for i in range(depth):
            layers.append(PredictorBlock(d_hidden, dropout))
        self.blocks = nn.Sequential(*layers)

        # Output projection back to latent space
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_latent),
        )

    def forward(
        self,
        context_encoding: torch.Tensor,  # (B, L_ctx, d_latent) — from context encoder
        target_positions: torch.Tensor,   # (B, L_tgt) — integer positions of targets
    ) -> torch.Tensor:
        """Predict target latent representations.

        Parameters
        ----------
        context_encoding : (B, L_ctx, d_latent) — encoded visible context
        target_positions : (B, L_tgt) — positions of masked targets

        Returns
        -------
        (B, L_tgt, d_latent) — predicted target representations
        """
        B, L_tgt = target_positions.shape

        # Aggregate context (mean pool + broadcast to target length)
        ctx = context_encoding.mean(dim=1, keepdim=True)  # (B, 1, d_latent)
        ctx = self.context_proj(ctx)  # (B, 1, d_hidden)
        ctx = ctx.expand(-1, L_tgt, -1)  # (B, L_tgt, d_hidden)

        # Add positional information for actual target positions
        # (Fix J2: use target_positions to index pos_embed, not sequential 0..L_tgt-1)
        if target_positions.max() < self.pos_embed.size(1):
            # Gather positional embeddings at the actual target positions
            # target_positions: (B, L_tgt) -> index into pos_embed (1, 1024, d_latent)
            pos_idx = target_positions.unsqueeze(-1).expand(-1, -1, self.d_latent)  # (B, L_tgt, d_latent)
            pos = self.pos_embed.expand(B, -1, -1).gather(1, pos_idx)  # (B, L_tgt, d_latent)
        else:
            # Fallback: positions exceed pos_embed size, use sequential
            pos = self.pos_embed[:, :L_tgt, :].expand(B, -1, -1)  # (B, L_tgt, d_latent)
        pos_proj = F.linear(pos,
                           self.context_proj[0].weight[:, :self.d_latent],
                           self.context_proj[0].bias)

        h = ctx + pos_proj  # (B, L_tgt, d_hidden)

        # Predict through residual blocks
        h = self.blocks(h)

        # Project to latent space
        return self.out_proj(h)  # (B, L_tgt, d_latent)


class PredictorBlock(nn.Module):
    """Residual block for the predictor network."""

    def __init__(self, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 4, d_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


# ---------------------------------------------------------------------------
# JEPA World Model
# ---------------------------------------------------------------------------

class JEPAWorldModel(nn.Module):
    """JEPA-based world model for financial markets.

    Learns to predict future market states in latent space.
    The context encoder processes visible past data through Mamba-2.
    The target encoder (EMA of context encoder) encodes masked future data.
    The predictor learns to bridge context → target in latent space.

    This IS a world model: given current state (context), it predicts
    future states (targets) — the fundamental building block for planning.

    Optionally embeds latent representations in hyperbolic space for
    natural hierarchical structure.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        d_latent: int = 128,
        d_state: int = 64,
        n_layers: int = 6,
        predictor_depth: int = 4,
        ema_decay: float = 0.996,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_hyperbolic: bool = True,
        d_hyperbolic: int = 64,
        curvature: float = -1.0,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.ema_decay = ema_decay
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

        # Context encoder (Mamba-2 backbone → latent projection)
        self.context_encoder = Mamba2Backbone(
            d_input=d_input,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_latent),
        )

        # Target encoder (EMA copy — NO GRADIENT)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_proj = copy.deepcopy(self.context_proj)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

        # Predictor (lightweight, learns dynamics)
        self.predictor = JEPAPredictor(
            d_latent=d_latent,
            depth=predictor_depth,
            d_hidden=d_model * 2,
            dropout=dropout,
        )

        # Optional: hyperbolic projection
        if use_hyperbolic:
            self.to_hyperbolic = EuclideanToLorentz(d_latent, d_hyperbolic, curvature)
            self.from_hyperbolic = LorentzToEuclidean(d_hyperbolic, d_latent, curvature)

        # Reward predictor (for planning): latent → scalar reward
        self.reward_head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
        )

        # Value function (for planning): latent → scalar value
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
        )

    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        """EMA update of target encoder from context encoder."""
        tau = self.ema_decay
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1.0 - tau)
        for p_ctx, p_tgt in zip(self.context_proj.parameters(),
                                self.target_proj.parameters()):
            p_tgt.data.mul_(tau).add_(p_ctx.data, alpha=1.0 - tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through context encoder.

        Parameters
        ----------
        x : (B, L, d_input)

        Returns
        -------
        (B, L, d_latent) — latent representations
        """
        h = self.context_encoder(x)
        return self.context_proj(h)

    def encode_to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project to hyperbolic space.

        Parameters
        ----------
        x : (B, L, d_input)

        Returns
        -------
        (B, L, d_hyperbolic+1) — points on Lorentz hyperboloid
        """
        z = self.encode(x)
        if self.use_hyperbolic:
            return self.to_hyperbolic(z)
        return z

    def forward(
        self,
        context_data: torch.Tensor,    # (B, L_ctx, d_input)
        target_data: torch.Tensor,     # (B, L_tgt, d_input)
        target_positions: torch.Tensor, # (B, L_tgt) int positions
    ) -> dict:
        """JEPA forward pass.

        1. Encode context through context encoder
        2. Encode target through EMA target encoder (no gradient)
        3. Predict target latent from context
        4. Compute JEPA loss: ||predicted - target||²

        Returns dict with:
            jepa_loss: L2 prediction loss in latent space
            context_latent: (B, L_ctx, d_latent)
            target_latent: (B, L_tgt, d_latent) — detached
            predicted_latent: (B, L_tgt, d_latent)
            reward_pred: (B, L_tgt, 1) — predicted rewards
            value_pred: (B, L_ctx, 1) — value estimates
        """
        # 1. Context encoding (with gradients)
        context_latent = self.encode(context_data)  # (B, L_ctx, d_latent)

        # 2. Target encoding (NO gradients — EMA encoder)
        with torch.no_grad():
            h_tgt = self.target_encoder(target_data)
            target_latent = self.target_proj(h_tgt)  # (B, L_tgt, d_latent)

        # 3. Predict target from context
        predicted_latent = self.predictor(context_latent, target_positions)

        # 4. JEPA loss: smooth L1 in latent space (robust to outliers)
        jepa_loss = F.smooth_l1_loss(predicted_latent, target_latent.detach())

        # 5. Reward and value predictions
        reward_pred = self.reward_head(predicted_latent)
        value_pred = self.value_head(context_latent.mean(dim=1, keepdim=True))

        # 6. Optional: hyperbolic distance regularization
        hyp_loss = torch.tensor(0.0, device=context_data.device)
        if self.use_hyperbolic:
            # Encode both to hyperbolic space
            ctx_hyp = self.to_hyperbolic(context_latent.mean(dim=1))
            tgt_hyp = self.to_hyperbolic(target_latent.mean(dim=1))
            pred_hyp = self.to_hyperbolic(predicted_latent.mean(dim=1))

            # Distance between predicted and target should be small
            hyp_loss = lorentz_distance(pred_hyp, tgt_hyp, self.curvature).mean()

        return {
            "jepa_loss": jepa_loss,
            "hyperbolic_loss": hyp_loss,
            "context_latent": context_latent,
            "target_latent": target_latent,
            "predicted_latent": predicted_latent,
            "reward_pred": reward_pred,
            "value_pred": value_pred,
        }

    def collapse_diagnostics(self, z: torch.Tensor) -> dict:
        """Compute collapse diagnostic metrics for latent representations.

        Collapse occurs when the model degenerates to predicting the mean
        for all inputs, losing all discriminative information. This method
        computes three complementary diagnostics to detect collapse early.

        Parameters
        ----------
        z : (B, d_latent) -- batch of latent representations

        Returns
        -------
        dict with:
        - per_dim_variance: (d_latent,) variance of each latent dimension.
            Should be > 0.01; if all dimensions collapse to ~0, the model
            is predicting a constant regardless of input.
        - effective_rank: scalar, nuclear norm / operator norm.
            Should be > d_latent/4 for healthy representations.
            Values near 1.0 indicate rank collapse (all info in 1 direction).
        - mean_cosine_sim: scalar, average pairwise cosine similarity.
            Should be < 0.5; values approaching 1.0 mean all representations
            are nearly identical (complete collapse).
        """
        B, D = z.shape

        # 1. Per-dimension variance: how spread out each latent dimension is
        per_dim_variance = z.var(dim=0)  # (d_latent,)

        # 2. Effective rank via nuclear norm / operator norm
        # High effective rank = information spread across many dimensions
        # Low effective rank = model using only 1-2 dimensions (partial collapse)
        if B >= 2:
            z_centered = z - z.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(z_centered)
                nuclear_norm = s.sum()
                operator_norm = s[0]
                effective_rank = (nuclear_norm / (operator_norm + 1e-8)).item()
            except Exception:
                effective_rank = float(D)
        else:
            effective_rank = float(D)

        # 3. Mean pairwise cosine similarity
        # If all representations are identical, cosine sim = 1.0 (full collapse)
        if B >= 2:
            z_norm = F.normalize(z, dim=-1)
            cos_sim_matrix = z_norm @ z_norm.t()  # (B, B)
            # Exclude diagonal (self-similarity = 1)
            mask = ~torch.eye(B, device=z.device, dtype=torch.bool)
            mean_cosine_sim = cos_sim_matrix[mask].mean().item()
        else:
            mean_cosine_sim = 0.0

        return {
            "per_dim_variance": per_dim_variance,
            "effective_rank": effective_rank,
            "mean_cosine_sim": mean_cosine_sim,
        }

    def imagine(
        self,
        current_latent: torch.Tensor,  # (B, d_latent) -- current state
        horizon: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Imagine future states using the learned world model.

        Autoregressively apply the predictor to generate imagined
        future latent states. This is the core of model-based planning.

        Parameters
        ----------
        current_latent : (B, d_latent) — current market state encoding
        horizon : int — number of future steps to imagine

        Returns
        -------
        imagined_states : (B, horizon, d_latent) — imagined future states
        imagined_rewards : (B, horizon, 1) — predicted rewards per step
        """
        B = current_latent.size(0)
        device = current_latent.device

        states = []
        rewards = []
        state = current_latent.unsqueeze(1)  # (B, 1, d_latent)

        for t in range(horizon):
            # Predict next state from current
            pos = torch.full((B, 1), t, dtype=torch.long, device=device)
            next_state = self.predictor(state, pos)  # (B, 1, d_latent)

            # Predict reward for this state
            reward = self.reward_head(next_state)  # (B, 1, 1)

            states.append(next_state.squeeze(1))
            rewards.append(reward.squeeze(1))

            # Update state for next step
            state = next_state

        imagined_states = torch.stack(states, dim=1)    # (B, horizon, d_latent)
        imagined_rewards = torch.stack(rewards, dim=1)  # (B, horizon, 1)

        return imagined_states, imagined_rewards
