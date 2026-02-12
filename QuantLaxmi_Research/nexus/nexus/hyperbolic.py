"""Hyperbolic geometry operations on the Lorentz model H^d.

Markets have natural HIERARCHICAL structure:
  Macro regime → Sector flows → Stock movements → Option greeks → Tick microstructure

Euclidean space cannot embed trees without O(n) dimensions.
Hyperbolic space (negative curvature) embeds ANY tree in O(log n) dimensions
with arbitrarily small distortion. This is mathematically proven (Sarkar 2011).

We use the Lorentz model (hyperboloid model) because:
1. Numerically more stable than Poincaré ball (no boundary issues)
2. Distance computation uses inner product (GPU-friendly)
3. Geodesics have closed-form expressions
4. Natural connection to special relativity (market "spacetime")

The Lorentz model H^d_K:
  H^d_K = {x ∈ R^{d+1} : ⟨x, x⟩_L = 1/K, x_0 > 0}

where ⟨x, y⟩_L = -x_0·y_0 + x_1·y_1 + ... + x_d·y_d  (Minkowski inner product)
and K < 0 is the curvature.

References:
    Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model", NeurIPS 2018
    Schwethelm et al., "Fully Hyperbolic CNNs", ICLR 2024
    LResNet, arXiv:2412.14695
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lorentz model operations
# ---------------------------------------------------------------------------

def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product ⟨x, y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ.

    Parameters
    ----------
    x, y : (..., d+1) — points in ambient Minkowski space

    Returns
    -------
    (...,) — Minkowski inner products
    """
    # Split time and space components
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """Geodesic distance on Lorentz hyperboloid H^d_K.

    d_K(x, y) = (1/√|K|) · arcosh(-K · ⟨x, y⟩_L)

    Parameters
    ----------
    x, y : (..., d+1) — points on H^d_K
    K : float — negative curvature

    Returns
    -------
    (...,) — geodesic distances
    """
    sqrt_neg_K = math.sqrt(-K)
    inner = minkowski_dot(x, y)
    # Clamp for numerical stability (arcosh domain is [1, ∞))
    inner = torch.clamp(-K * inner, min=1.0 + 1e-7)
    return torch.acosh(inner) / sqrt_neg_K


def project_to_hyperboloid(x: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """Project from Euclidean R^d to Lorentz hyperboloid H^d_K.

    Given spatial coordinates x ∈ R^d, compute time coordinate x₀ such that
    ⟨x, x⟩_L = 1/K, i.e., x₀ = √(1/|K| + ||x_spatial||²)

    Parameters
    ----------
    x : (..., d) — spatial coordinates

    Returns
    -------
    (..., d+1) — points on H^d_K (prepended time coordinate)
    """
    sq_norm = (x * x).sum(dim=-1, keepdim=True)
    time_coord = torch.sqrt(1.0 / (-K) + sq_norm)
    return torch.cat([time_coord, x], dim=-1)


def exp_map_0(v: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """Exponential map at the origin of H^d_K.

    Maps tangent vector v ∈ T_o(H^d) to point on hyperboloid.
    The origin o = (1/√|K|, 0, 0, ..., 0).

    exp_o(v) = cosh(√|K|·||v||) · o + sinh(√|K|·||v||) / (√|K|·||v||) · v̄

    where v̄ = (0, v₁, ..., v_d) is the tangent vector lifted to ambient space.

    Parameters
    ----------
    v : (..., d) — tangent vectors at origin (spatial components only)

    Returns
    -------
    (..., d+1) — points on H^d_K
    """
    sqrt_neg_K = math.sqrt(-K)
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-7)
    scaled_norm = sqrt_neg_K * v_norm

    # Origin on H^d_K
    time_coord = torch.cosh(scaled_norm) / sqrt_neg_K
    space_coord = torch.sinh(scaled_norm) / (sqrt_neg_K * v_norm) * v

    return torch.cat([time_coord, space_coord], dim=-1)


def log_map_0(x: torch.Tensor, K: float = -1.0) -> torch.Tensor:
    """Logarithmic map at the origin — inverse of exp_map_0.

    Maps point on H^d_K back to tangent vector at origin.

    Parameters
    ----------
    x : (..., d+1) — points on H^d_K

    Returns
    -------
    (..., d) — tangent vectors at origin
    """
    sqrt_neg_K = math.sqrt(-K)
    x_spatial = x[..., 1:]
    x_time = x[..., :1]

    # d(o, x) = arcosh(√|K| · x₀) / √|K|
    dist = torch.acosh(torch.clamp(sqrt_neg_K * x_time, min=1.0 + 1e-7)) / sqrt_neg_K
    spatial_norm = torch.clamp(torch.norm(x_spatial, dim=-1, keepdim=True), min=1e-7)

    return dist / spatial_norm * x_spatial


def lorentz_centroid(x: torch.Tensor, weights: Optional[torch.Tensor] = None,
                     K: float = -1.0) -> torch.Tensor:
    """Weighted Lorentz centroid (Einstein midpoint).

    The Fréchet mean on H^d_K, computed via the closed-form Einstein midpoint:
        c = Σ wᵢxᵢ / √|⟨Σ wᵢxᵢ, Σ wᵢxᵢ⟩_L · |K||

    Parameters
    ----------
    x : (N, d+1) — points on H^d_K
    weights : (N,) optional — non-negative weights (default: uniform)

    Returns
    -------
    (d+1,) — centroid on H^d_K
    """
    if weights is None:
        weights = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()

    # Weighted sum in ambient space
    weighted_sum = (weights.unsqueeze(-1) * x).sum(dim=0)

    # Project back to hyperboloid
    inner = minkowski_dot(weighted_sum, weighted_sum)
    scale = 1.0 / torch.sqrt(torch.clamp(torch.abs(inner * (-K)), min=1e-10))
    return weighted_sum * scale


# ---------------------------------------------------------------------------
# Lorentz Linear Layer (hyperbolic neural network building block)
# ---------------------------------------------------------------------------

class LorentzLinear(nn.Module):
    """Linear transformation in Lorentz model.

    Maps between tangent spaces: log → Euclidean linear → exp.
    This is the standard approach from Ganea et al. (2018).

    More efficient: operate on spatial coordinates only (LResNet approach).
    x₀ is recomputed from the hyperboloid constraint.
    """

    def __init__(self, in_features: int, out_features: int, K: float = -1.0,
                 bias: bool = True):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Initialize with small values for stability
        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., in_features+1) — points on H^d_K

        Returns
        -------
        (..., out_features+1) — transformed points on H^d_K
        """
        # Extract spatial coordinates (drop time component)
        x_spatial = x[..., 1:]

        # Linear transform in spatial coordinates (LResNet approach)
        y_spatial = self.linear(x_spatial)

        # Recompute time coordinate from hyperboloid constraint
        y = project_to_hyperboloid(y_spatial, self.K)

        return y


# ---------------------------------------------------------------------------
# Euclidean ↔ Hyperbolic projection layers
# ---------------------------------------------------------------------------

class EuclideanToLorentz(nn.Module):
    """Project Euclidean embeddings to Lorentz hyperboloid.

    Uses exponential map at origin: R^d → H^d_K.
    """

    def __init__(self, d_euclidean: int, d_hyperbolic: int, K: float = -1.0):
        super().__init__()
        self.K = K
        self.proj = nn.Linear(d_euclidean, d_hyperbolic)
        nn.init.uniform_(self.proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., d_euclidean) — Euclidean vectors

        Returns
        -------
        (..., d_hyperbolic+1) — points on H^d_K
        """
        v = self.proj(x)
        return exp_map_0(v, self.K)


class LorentzToEuclidean(nn.Module):
    """Project Lorentz hyperboloid points to Euclidean space.

    Uses logarithmic map at origin: H^d_K → R^d.
    """

    def __init__(self, d_hyperbolic: int, d_euclidean: int, K: float = -1.0):
        super().__init__()
        self.K = K
        self.proj = nn.Linear(d_hyperbolic, d_euclidean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., d_hyperbolic+1) — points on H^d_K

        Returns
        -------
        (..., d_euclidean) — Euclidean vectors
        """
        v = log_map_0(x, self.K)
        return self.proj(v)


# ---------------------------------------------------------------------------
# Hyperbolic Attention (for regime comparison)
# ---------------------------------------------------------------------------

class LorentzAttention(nn.Module):
    """Attention mechanism using hyperbolic distances.

    Instead of dot-product attention: score = -d_H(q, k) (negative distance).
    Closer states in hyperbolic space get higher attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, K: float = -1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,    # (B, L_q, d_model+1) on H^d
        key: torch.Tensor,      # (B, L_k, d_model+1) on H^d
        value: torch.Tensor,    # (B, L_k, d_model) Euclidean
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hyperbolic attention.

        Scores are negative geodesic distances (closer = higher score).
        Values remain Euclidean for computational efficiency.
        """
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        # Log map to tangent space for linear projections
        q_tang = log_map_0(query, self.K)  # (B, L_q, d_model)
        k_tang = log_map_0(key, self.K)    # (B, L_k, d_model)

        # Project and reshape for multi-head
        q = self.q_proj(q_tang).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k_tang).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores (scaled dot product in tangent space)
        # This approximates -d_H(q, k) for nearby points
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, n_heads, L_q, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.out_proj(out)
