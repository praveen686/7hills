"""NEXUS: The complete model — tying all components together.

                    ┌─────────────────────────────────────────────┐
                    │              NEXUS ARCHITECTURE              │
                    │                                             │
                    │  Market Data (tick/1min/daily)               │
                    │       │                                     │
                    │       ▼                                     │
                    │  ┌─────────────┐                            │
                    │  │ Multi-Scale │  VQ tokenization            │
                    │  │ Tokenizer   │  + delta encoding           │
                    │  └──────┬──────┘                            │
                    │         │                                   │
                    │    ┌────┴────┐                              │
                    │    │ Mamba-2 │  O(n) selective state space   │
                    │    │Backbone │  (NOT transformer O(n²))     │
                    │    └────┬────┘                              │
                    │         │                                   │
                    │  ┌──────┴──────────────────────────┐        │
                    │  │     JEPA World Model             │        │
                    │  │                                  │        │
                    │  │  Context ──→ Predictor ──→ ẑ_tgt │        │
                    │  │  Target  ──→ EMA Enc   ──→ z_tgt │        │
                    │  │  Loss: ||ẑ_tgt - z_tgt||²       │        │
                    │  └──────┬──────────────────────────┘        │
                    │         │                                   │
                    │  ┌──────┴──────┐                            │
                    │  │  Hyperbolic │  Lorentz H^d manifold       │
                    │  │ Latent Space│  (hierarchical geometry)    │
                    │  └──────┬──────┘                            │
                    │         │                                   │
                    │    ┌────┴────┐   ┌──────────────┐           │
                    │    │Topology │   │  MPC Planner │           │
                    │    │ Sensor  │   │ (TD-MPC2)    │           │
                    │    │ (TDA)   │   │              │           │
                    │    └────┬────┘   └──────┬───────┘           │
                    │         │               │                   │
                    │         └───────┬───────┘                   │
                    │                 ▼                            │
                    │          ┌────────────┐                     │
                    │          │  Position  │                     │
                    │          │  Vector    │  per-asset positions │
                    │          └────────────┘                     │
                    └─────────────────────────────────────────────┘

This is the FIRST system that combines:
1. JEPA (predict in latent space, not observation space)
2. Mamba-2 (O(n) temporal encoding, not O(n²) attention)
3. Hyperbolic geometry (natural for hierarchical markets)
4. Persistent homology (topological regime sensing)
5. Model-based planning (TD-MPC2 in latent space)

...for financial markets. Each piece is SOTA in its domain.
Together they form something genuinely new.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NexusConfig
from .mamba2 import Mamba2Backbone
from .jepa import JEPAWorldModel
from .hyperbolic import (
    EuclideanToLorentz,
    LorentzToEuclidean,
    lorentz_distance,
)
from .topology import TopologicalSensor
from .planner import NexusPlanner


class NEXUS(nn.Module):
    """Neural Exchange Unified Simulator.

    End-to-end model: raw market data → optimal position vector.

    Training has 3 phases:
    1. JEPA pre-training: learn market representations (self-supervised)
    2. World model training: learn dynamics + reward + value (TD learning)
    3. Planning: CEM trajectory optimization in latent space (no gradient)

    Inference:
    1. Encode current market data → latent state
    2. (Optional) Compute topological features for regime awareness
    3. Plan optimal action via CEM in latent space
    4. Output: position vector per asset
    """

    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.cfg = cfg

        # ── JEPA World Model (Mamba-2 encoder + predictor) ────────────
        self.world_model = JEPAWorldModel(
            d_input=cfg.total_features,
            d_model=cfg.d_model,
            d_latent=cfg.d_latent,
            d_state=cfg.d_state,
            n_layers=cfg.n_layers,
            predictor_depth=cfg.predictor_depth,
            ema_decay=cfg.ema_decay,
            n_heads=cfg.n_heads,
            dropout=0.1,
            use_hyperbolic=True,
            d_hyperbolic=cfg.d_hyperbolic,
            curvature=cfg.curvature,
        )

        # ── Topological Sensor ────────────────────────────────────────
        self.topo_sensor = TopologicalSensor(
            window=cfg.tda_window,
            max_homology_dim=cfg.tda_max_dim,
            n_landscape_bins=cfg.tda_n_bins,
        )

        # ── Latent Planner (TD-MPC2) ─────────────────────────────────
        self.planner = NexusPlanner(
            d_latent=cfg.d_latent,
            d_action=cfg.d_action,
            horizon=cfg.horizon,
            n_samples=cfg.n_samples,
            n_elites=cfg.n_elites,
            n_iterations=cfg.n_iterations,
            discount=cfg.discount,
            max_position=cfg.max_position,
        )

        # ── Regime classifier (from topological features) ────────────
        self.regime_head = nn.Sequential(
            nn.Linear(cfg.tda_features + cfg.d_latent, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # 4 regimes: bull, bear, range, crisis
        )

        # ── Direct policy head (for fast inference without planning) ──
        self.policy_head = nn.Sequential(
            nn.LayerNorm(cfg.d_latent),
            nn.Linear(cfg.d_latent, cfg.d_latent),
            nn.GELU(),
            nn.Linear(cfg.d_latent, cfg.d_action),
            nn.Tanh(),  # positions in [-1, 1]
        )

    def encode(self, market_data: torch.Tensor) -> torch.Tensor:
        """Encode market data to latent representation.

        Parameters
        ----------
        market_data : (B, L, n_features) — raw market data

        Returns
        -------
        (B, d_latent) — latent market state (last timestep)
        """
        z_seq = self.world_model.encode(market_data)  # (B, L, d_latent)
        return z_seq[:, -1, :]  # Use last timestep as current state

    def encode_sequence(self, market_data: torch.Tensor) -> torch.Tensor:
        """Encode market data to latent sequence.

        Parameters
        ----------
        market_data : (B, L, n_features)

        Returns
        -------
        (B, L, d_latent) — full latent sequence
        """
        return self.world_model.encode(market_data)

    def forward(
        self,
        context_data: torch.Tensor,     # (B, L_ctx, n_features)
        target_data: torch.Tensor,      # (B, L_tgt, n_features)
        target_positions: torch.Tensor,  # (B, L_tgt) int
        rewards: Optional[torch.Tensor] = None,  # (B, L_tgt)
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for training.

        Computes:
        1. JEPA loss (representation learning)
        2. Hyperbolic loss (hierarchical structure)
        3. Policy output (direct position prediction)
        4. Regime classification

        Parameters
        ----------
        context_data : visible past market data
        target_data : masked future market data
        target_positions : integer positions of targets
        rewards : optional ground-truth rewards for supervised training

        Returns
        -------
        dict with losses, predictions, and diagnostics
        """
        # 1. JEPA forward pass
        jepa_out = self.world_model(context_data, target_data, target_positions)

        # 2. Extract current state (last context timestep)
        context_latent = jepa_out["context_latent"]  # (B, L_ctx, d_latent)
        current_state = context_latent[:, -1, :]      # (B, d_latent)

        # 3. Direct policy (fast inference path)
        raw_positions = self.policy_head(current_state)  # (B, d_action)
        positions = raw_positions * self.cfg.max_position  # Scale to max position

        # 4. Topological features (regime sensing)
        topo_features = self.topo_sensor(context_latent)  # (B, n_windows, 8)
        topo_summary = topo_features.mean(dim=1)  # (B, 8)

        # 5. Regime classification
        regime_input = torch.cat([current_state, topo_summary], dim=-1)
        regime_logits = self.regime_head(regime_input)  # (B, 4)

        # 6. Compute losses
        total_loss = (
            self.cfg.jepa_loss_weight * jepa_out["jepa_loss"]
            + self.cfg.hyperbolic_loss_weight * jepa_out["hyperbolic_loss"]
        )

        # Optional: supervised reward prediction loss
        reward_loss = torch.tensor(0.0, device=context_data.device)
        if rewards is not None:
            reward_pred = jepa_out["reward_pred"].squeeze(-1)  # (B, L_tgt)
            reward_loss = F.mse_loss(reward_pred, rewards)
            total_loss = total_loss + reward_loss

        return {
            "total_loss": total_loss,
            "jepa_loss": jepa_out["jepa_loss"],
            "hyperbolic_loss": jepa_out["hyperbolic_loss"],
            "reward_loss": reward_loss,
            "positions": positions,
            "regime_logits": regime_logits,
            "context_latent": context_latent,
            "target_latent": jepa_out["target_latent"],
            "predicted_latent": jepa_out["predicted_latent"],
            "topo_features": topo_features,
        }

    @torch.no_grad()
    def act(
        self,
        market_data: torch.Tensor,
        use_planning: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Select action (inference mode).

        Parameters
        ----------
        market_data : (B, L, n_features) — current market window
        use_planning : bool — if True, use CEM planning (slower, better)
                               if False, use direct policy (fast)

        Returns
        -------
        positions : (B, d_action) — per-asset position vector
        info : dict — diagnostics (planned value, regime, topo features)
        """
        self.eval()

        # Encode current state
        z_seq = self.world_model.encode(market_data)
        current_state = z_seq[:, -1, :]

        # Topological features
        topo_features = self.topo_sensor(z_seq)
        topo_summary = topo_features.mean(dim=1)

        # Regime
        regime_input = torch.cat([current_state, topo_summary], dim=-1)
        regime_logits = self.regime_head(regime_input)
        regime = torch.argmax(regime_logits, dim=-1)

        if use_planning:
            # CEM planning in latent space
            positions, plan_info = self.planner.plan(current_state)
            plan_info["regime"] = regime
            plan_info["topo_features"] = topo_summary
            return positions, plan_info
        else:
            # Direct policy (fast path)
            positions = self.policy_head(current_state) * self.cfg.max_position
            return positions, {
                "regime": regime,
                "topo_features": topo_summary,
                "planned_value": torch.zeros(market_data.size(0), device=market_data.device),
            }

    def imagine_futures(
        self,
        market_data: torch.Tensor,
        horizon: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Imagine future market states (for visualization / analysis).

        Parameters
        ----------
        market_data : (B, L, n_features)
        horizon : number of future steps

        Returns
        -------
        imagined_states : (B, horizon, d_latent)
        imagined_rewards : (B, horizon, 1)
        """
        z_seq = self.world_model.encode(market_data)
        current_state = z_seq[:, -1, :]
        return self.world_model.imagine(current_state, horizon)

    def get_hyperbolic_embeddings(
        self,
        market_data: torch.Tensor,
    ) -> torch.Tensor:
        """Get hyperbolic embeddings for visualization.

        Maps market states to the Poincaré disk (2D projection
        of the Lorentz hyperboloid) for beautiful visualizations.

        Parameters
        ----------
        market_data : (B, L, n_features)

        Returns
        -------
        (B, L, d_hyperbolic+1) — points on Lorentz hyperboloid
        """
        return self.world_model.encode_to_hyperbolic(market_data)

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "world_model": _count(self.world_model),
            "topo_sensor": _count(self.topo_sensor),
            "planner": _count(self.planner),
            "regime_head": _count(self.regime_head),
            "policy_head": _count(self.policy_head),
            "total": _count(self),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_nexus(
    n_features: int = 192,
    n_assets: int = 6,
    size: str = "base",
) -> NEXUS:
    """Create NEXUS model with preset configurations.

    Parameters
    ----------
    n_features : total input features per timestep
    n_assets : number of tradeable assets
    size : "small" (dev/test), "base" (production), "large" (research)

    Returns
    -------
    NEXUS model instance
    """
    presets = {
        "small": dict(
            d_model=128, d_latent=64, d_state=32, n_layers=3,
            predictor_depth=2, d_hyperbolic=32, n_heads=4,
            n_samples=128, n_elites=16, n_iterations=3,
        ),
        "base": dict(
            d_model=256, d_latent=128, d_state=64, n_layers=6,
            predictor_depth=4, d_hyperbolic=64, n_heads=8,
            n_samples=512, n_elites=64, n_iterations=6,
        ),
        "large": dict(
            d_model=512, d_latent=256, d_state=128, n_layers=12,
            predictor_depth=6, d_hyperbolic=128, n_heads=16,
            n_samples=1024, n_elites=128, n_iterations=8,
        ),
    }

    if size not in presets:
        raise ValueError(f"Unknown size: {size}. Use 'small', 'base', or 'large'.")

    cfg = NexusConfig(
        n_assets=n_assets,
        n_features_daily=n_features // n_assets,
        d_action=n_assets,
        **presets[size],
    )

    return NEXUS(cfg)
