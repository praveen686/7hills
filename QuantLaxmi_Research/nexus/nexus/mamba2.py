"""Mamba-2: Selective State Space Model with Structured State Space Duality.

Implements the Mamba-2 architecture (Dao & Gu, 2024) from scratch in PyTorch.
Key innovation: Selective Scan — the model learns DATA-DEPENDENT state transitions,
deciding what to remember and what to forget at each timestep. This gives
transformer-quality modeling at O(n) complexity instead of O(n²).

For financial time series:
- O(n) handles 100K+ tick-level tokens (transformers would be O(n²) = 10B ops)
- Selective scan naturally models regime shifts (forget old regime, remember new)
- State space formulation has natural connection to Kalman filtering (quant intuition)

Architecture per layer:
    x → Linear(d→d_inner) → Conv1d(d_conv) → SiLU → SSM(selective) → Linear(d_inner→d)
        ↓                                                     ↑
        └──── Linear(d→d_inner) → SiLU ──── (gate) ──────────┘

References:
    Mamba-2: Dao & Gu, "Transformers are SSMs", arXiv:2405.21060
    Mamba-1: Gu & Dao, arXiv:2312.00752
    S4: Gu et al., ICLR 2022
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Selective Scan (core Mamba operation)
# ---------------------------------------------------------------------------

def selective_scan(
    x: torch.Tensor,       # (B, L, D)
    delta: torch.Tensor,    # (B, L, D) — timestep-dependent discretization
    A: torch.Tensor,        # (D, N) — state matrix (learned, data-independent)
    B: torch.Tensor,        # (B, L, N) — input matrix (data-dependent!)
    C: torch.Tensor,        # (B, L, N) — output matrix (data-dependent!)
    D: torch.Tensor,        # (D,) — skip connection
) -> torch.Tensor:
    """Selective scan — the heart of Mamba.

    Unlike classical SSMs where A, B, C are fixed, Mamba makes B and C
    DATA-DEPENDENT. This is the key innovation: the model learns WHAT
    to remember (B selects which inputs enter the state) and WHAT to
    output (C selects which state components produce output).

    Discrete-time SSM:
        h_t = Ā·h_{t-1} + B̄·x_t
        y_t = C_t·h_t + D·x_t

    where Ā = exp(Δ·A), B̄ = Δ·B (zero-order hold discretization)

    Parameters
    ----------
    x : (B, L, D) — input sequence
    delta : (B, L, D) — per-step discretization (learned, data-dependent)
    A : (D, N) — diagonal state matrix (negative for stability)
    B : (B, L, N) — input projection (data-dependent)
    C : (B, L, N) — output projection (data-dependent)
    D : (D,) — skip connection weight

    Returns
    -------
    y : (B, L, D) — output sequence
    """
    batch, seq_len, d_model = x.shape
    _, _, n_state = B.shape

    # Discretize: Ā = exp(Δ·A), B̄ = Δ·B
    # A is (D, N), delta is (B, L, D) → deltaA is (B, L, D, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

    # Sequential scan (could be parallelized with associative scan)
    h = torch.zeros(batch, d_model, n_state, device=x.device, dtype=x.dtype)
    ys = []

    for t in range(seq_len):
        # h_t = Ā_t · h_{t-1} + B̄_t · x_t
        h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
        # y_t = C_t · h_t
        y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, L, D)

    # Skip connection
    y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y


# ---------------------------------------------------------------------------
# Mamba-2 Block
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Single Mamba-2 block with selective scan + gating.

    Architecture:
        x → Linear(d→d_inner) → Conv1d(d_conv) → SiLU → SSM → ×gate → Linear(d_inner→d)
            ↓                                                    ↑
            └─── Linear(d→d_inner) → SiLU ──────────────────────┘
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank
        self.n_heads = n_heads

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise convolution (local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters
        # Δ (delta) projection: input-dependent discretization step
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize dt bias for stability
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A matrix: diagonal, initialized as negative (stable dynamics)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # B, C projections (data-dependent — the SELECTIVE part)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D) — input sequence

        Returns
        -------
        (B, L, D) — output sequence
        """
        residual = x
        x = self.norm(x)

        # Project to inner dimension: split into main path + gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Depthwise conv (local context mixing)
        x_main = rearrange(x_main, "b l d -> b d l")
        x_main = self.conv1d(x_main)[:, :, :x.size(1)]  # causal: trim padding
        x_main = rearrange(x_main, "b d l -> b l d")
        x_main = F.silu(x_main)

        # Compute data-dependent SSM parameters
        A = -torch.exp(self.A_log)  # (d_inner, N) — negative for stability
        delta = F.softplus(self.dt_proj(x_main))  # (B, L, d_inner)
        B = self.B_proj(x_main)  # (B, L, N)
        C = self.C_proj(x_main)  # (B, L, N)

        # Selective scan
        y = selective_scan(x_main, delta, A, B, C, self.D)

        # Gate
        y = y * F.silu(z)

        # Project back
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual


# ---------------------------------------------------------------------------
# Mamba-2 Backbone (stack of blocks)
# ---------------------------------------------------------------------------

class Mamba2Backbone(nn.Module):
    """Stack of Mamba-2 blocks for temporal encoding.

    Replaces transformer encoder with O(n) complexity.
    Processes market data sequences of any length efficiently.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 6,
        dt_rank: int = 16,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Mamba-2 layers
        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_input) — raw input features

        Returns
        -------
        (B, L, d_model) — encoded representations
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
