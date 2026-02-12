"""NEXUS configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NexusConfig:
    """Full NEXUS configuration.

    Sections
    --------
    Mamba-2 backbone, JEPA world model, hyperbolic latent space,
    topological sensor, MPC planner, and training hyperparameters.
    """

    # ── Mamba-2 Backbone ──────────────────────────────────────────────
    d_model: int = 256              # Model dimension
    d_state: int = 64               # SSM state dimension (N)
    d_conv: int = 4                 # Local convolution width
    expand: int = 2                 # Block expansion factor (E)
    n_layers: int = 6               # Number of Mamba-2 layers
    dt_rank: str = "auto"           # Rank of Δ projection ("auto" = d_model // 16)
    dt_min: float = 0.001           # Minimum Δ
    dt_max: float = 0.1             # Maximum Δ
    n_heads: int = 8                # Number of heads for SSD (Mamba-2)

    # ── JEPA World Model ──────────────────────────────────────────────
    d_latent: int = 128             # Latent representation dimension
    predictor_depth: int = 4        # Predictor MLP depth
    ema_decay: float = 0.996        # EMA decay for target encoder
    mask_ratio: float = 0.5         # Fraction of future timesteps to mask
    jepa_loss_weight: float = 1.0   # Weight for JEPA prediction loss

    # ── Hyperbolic Latent Space (Lorentz model H^d) ───────────────────
    curvature: float = -1.0         # Sectional curvature K (negative for hyperbolic)
    d_hyperbolic: int = 64          # Hyperbolic embedding dimension (+1 for time coord)
    hyperbolic_loss_weight: float = 0.1  # Weight for hyperbolic regularization

    # ── Topological Sensor ────────────────────────────────────────────
    tda_window: int = 50            # Sliding window for persistence computation
    tda_max_dim: int = 1            # Max homology dimension (0=components, 1=loops)
    tda_n_bins: int = 20            # Persistence landscape discretization
    tda_features: int = 8           # Number of topological features

    # ── MPC Planner (TD-MPC2 style) ──────────────────────────────────
    horizon: int = 5                # Planning horizon (trading days)
    n_samples: int = 512            # Number of trajectory samples for CEM
    n_elites: int = 64              # Number of elite trajectories
    n_iterations: int = 6           # CEM iterations
    temperature: float = 0.5        # Softmax temperature for elite selection
    discount: float = 0.99          # Reward discount factor
    d_action: int = 6               # Action dimension (position per asset)

    # ── Multi-Scale Input ─────────────────────────────────────────────
    n_assets: int = 6               # NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, BTC, ETH
    n_features_daily: int = 32      # Daily OHLCV + technical features per asset
    n_features_intraday: int = 16   # Intraday features per asset
    seq_len: int = 252              # Lookback window (1 year)
    context_len: int = 126          # Context (visible) length
    target_len: int = 126           # Target (masked) length

    # ── Training ──────────────────────────────────────────────────────
    lr: float = 3e-4                # Learning rate
    weight_decay: float = 1e-5      # Weight decay
    batch_size: int = 32            # Batch size
    epochs: int = 200               # Maximum epochs
    warmup_steps: int = 1000        # LR warmup steps
    grad_clip: float = 1.0          # Gradient clipping norm
    mixed_precision: bool = True    # AMP on CUDA

    # ── Walk-Forward ──────────────────────────────────────────────────
    train_window: int = 504         # 2 years rolling train
    test_window: int = 63           # 3 months test
    purge_gap: int = 5              # Days between train/test (no look-ahead)
    step_size: int = 21             # Monthly step

    # ── Position Sizing ───────────────────────────────────────────────
    max_position: float = 0.25      # Maximum position per asset
    vol_target: float = 0.15        # Annual vol target
    cost_bps: float = 5.0           # Transaction cost (bps)

    @property
    def d_inner(self) -> int:
        """Inner dimension after expansion."""
        return self.d_model * self.expand

    @property
    def dt_rank_value(self) -> int:
        """Computed Δ rank."""
        if self.dt_rank == "auto":
            return max(self.d_model // 16, 1)
        return int(self.dt_rank)

    @property
    def total_features(self) -> int:
        """Total input features per timestep."""
        return self.n_assets * self.n_features_daily
