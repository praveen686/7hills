"""Topological Regime Sensor using Persistent Homology.

Markets undergo structural phase transitions that are invisible to
traditional indicators (moving averages, momentum, volatility).
Persistent homology detects these transitions by analyzing the
TOPOLOGICAL STRUCTURE of market data.

Key insight: When a market regime changes, the SHAPE of the data changes:
- Bull market: returns cluster in upper half-plane → high β₀ (many components)
- Transition: returns form cycles → β₁ increases (loops appear)
- Crisis: returns collapse to tight cluster → β₀ drops, entropy spikes

Persistent homology computes:
- β₀: number of connected components (market fragmentation)
- β₁: number of loops/cycles (cyclical instability)
- Persistence entropy: H = -Σ pᵢ log(pᵢ) (structural complexity)
- Persistence landscape: functional summary for statistical analysis

Implementation uses Takens embedding to convert 1D time series
to point clouds in R^d, then computes Vietoris-Rips persistence.

References:
    Investigation of Indian stock markets using TDA (ScienceDirect 2024)
    Change point detection using TDA (MDPI 2024)
    Topological features for crisis prediction (Neural Computing 2024)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Takens Embedding
# ---------------------------------------------------------------------------

def takens_embedding(
    x: np.ndarray,
    dim: int = 3,
    tau: int = 1,
) -> np.ndarray:
    """Takens time-delay embedding: R → R^d.

    Converts a 1D time series into a point cloud in R^d by creating
    delay vectors: [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

    By Takens' theorem (1981), if d ≥ 2·dim(attractor) + 1,
    this embedding preserves the topology of the underlying attractor.

    Parameters
    ----------
    x : (N,) — 1D time series
    dim : int — embedding dimension
    tau : int — time delay

    Returns
    -------
    (N - (dim-1)*tau, dim) — point cloud
    """
    N = len(x)
    M = N - (dim - 1) * tau
    if M <= 0:
        raise ValueError(f"Series too short ({N}) for dim={dim}, tau={tau}")

    points = np.zeros((M, dim))
    for i in range(dim):
        points[:, i] = x[i * tau: i * tau + M]
    return points


# ---------------------------------------------------------------------------
# Persistence computation (pure NumPy implementation)
# ---------------------------------------------------------------------------

def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    sq = np.sum(X ** 2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    D = np.maximum(D, 0.0)
    return np.sqrt(D)


def vietoris_rips_H0(distances: np.ndarray) -> List[Tuple[float, float]]:
    """Compute H0 persistence diagram via single-linkage clustering.

    H0 tracks connected components as the filtration radius increases.
    Birth = 0 (each point starts as its own component).
    Death = distance at which two components merge.

    This is equivalent to Kruskal's MST algorithm.

    Parameters
    ----------
    distances : (N, N) — pairwise distance matrix

    Returns
    -------
    List of (birth, death) pairs for H0
    """
    N = distances.shape[0]

    # Get upper triangle as edge list, sorted by weight
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((distances[i, j], i, j))
    edges.sort()

    # Union-Find for single linkage
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    persistence = []
    for dist, i, j in edges:
        if union(i, j):
            persistence.append((0.0, dist))

    # One component never dies (infinite persistence) — exclude it
    return persistence


def compute_persistence(
    point_cloud: np.ndarray,
    max_dim: int = 1,
    max_edge: float = float("inf"),
) -> dict:
    """Compute persistence diagrams up to dimension max_dim.

    H0: exact via single-linkage clustering (equivalent to Kruskal MST).
    H1: spectral approximation using graph Laplacian eigenvalues
        (for exact H1 persistent homology, use ripser or giotto-tda).

    Parameters
    ----------
    point_cloud : (N, d) — point cloud
    max_dim : int — maximum homology dimension
    max_edge : float — maximum filtration value

    Returns
    -------
    dict with keys "H0", "H1", ... containing lists of (birth, death) pairs
    """
    distances = _pairwise_distances(point_cloud)

    result = {}

    # H0: connected components (exact via single-linkage)
    result["H0"] = vietoris_rips_H0(distances)

    # H1: spectral proxy for cycle-like structure (NOT exact persistent homology).
    # Uses Euler characteristic at multiple filtration radii to estimate beta_1.
    # For exact H1, use ripser or giotto-tda.
    if max_dim >= 1:
        result["H1"] = _spectral_cycle_proxy(distances, max_edge)

    return result


def _spectral_cycle_proxy(distances: np.ndarray, max_edge: float) -> List[Tuple[float, float]]:
    """Spectral proxy for cycle-like structure (NOT exact persistent homology).

    Estimates beta_1 at multiple filtration radii using the Euler
    characteristic: beta_1 = edges - vertices + components. The number
    of connected components is derived from near-zero eigenvalues of
    the graph Laplacian.

    This is a FAST APPROXIMATION. It captures the essential topological
    information for regime detection but does NOT compute true persistent
    homology. Specifically:
    - Birth/death times are approximate (tied to sampled percentile radii)
    - Persistence pairs are heuristically tracked, not via boundary matrices
    - May miss or double-count short-lived cycles

    For exact H1 persistent homology, use:
        import ripser; ripser.ripser(X, maxdim=1)
    or:
        from gtda.homology import VietorisRipsPersistence
    """
    N = distances.shape[0]
    if N < 4:
        return []

    # Sample several filtration radii
    radii = np.percentile(distances[distances > 0], np.linspace(10, 90, 20))

    persistence_pairs = []
    prev_beta1 = 0

    for r in radii:
        # Build adjacency at radius r
        adj = (distances <= r).astype(float)
        np.fill_diagonal(adj, 0.0)

        # Graph Laplacian
        degree = adj.sum(axis=1)
        laplacian = np.diag(degree) - adj

        # Eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(laplacian)
            eigenvalues = np.sort(eigenvalues)
        except np.linalg.LinAlgError:
            continue

        # Count near-zero eigenvalues (components)
        threshold = 1e-6
        n_components = np.sum(eigenvalues < threshold)

        # β₁ ≈ edges - vertices + components (Euler characteristic)
        n_edges = int(adj.sum() / 2)
        beta1 = max(0, n_edges - N + n_components)

        # Track births/deaths of cycles
        if beta1 > prev_beta1:
            for _ in range(beta1 - prev_beta1):
                persistence_pairs.append((r, float("inf")))
        elif beta1 < prev_beta1 and persistence_pairs:
            # Close some open cycles
            for _ in range(prev_beta1 - beta1):
                for k in range(len(persistence_pairs) - 1, -1, -1):
                    if persistence_pairs[k][1] == float("inf"):
                        persistence_pairs[k] = (persistence_pairs[k][0], r)
                        break

        prev_beta1 = beta1

    # Filter out infinite persistence pairs (open at end)
    return [(b, d) for b, d in persistence_pairs if d < float("inf")]


# ---------------------------------------------------------------------------
# Topological Features
# ---------------------------------------------------------------------------

def persistence_entropy(diagram: List[Tuple[float, float]]) -> float:
    """Persistence entropy: H = -Σ pᵢ log(pᵢ).

    Measures the complexity/disorder of the topological structure.
    High entropy → many features with similar persistence (complex structure).
    Low entropy → dominated by few long-lived features (simple structure).

    Parameters
    ----------
    diagram : list of (birth, death) pairs

    Returns
    -------
    float — persistence entropy (≥ 0)
    """
    if not diagram:
        return 0.0

    lifetimes = np.array([d - b for b, d in diagram if d > b and np.isfinite(d)])
    if len(lifetimes) == 0 or lifetimes.sum() == 0:
        return 0.0

    probs = lifetimes / lifetimes.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def betti_numbers(diagram: dict, radius: float) -> dict:
    """Compute Betti numbers at a given filtration radius.

    Parameters
    ----------
    diagram : dict with "H0", "H1" persistence diagrams
    radius : float — filtration radius

    Returns
    -------
    dict with "beta_0", "beta_1"
    """
    result = {}
    for key in ["H0", "H1"]:
        if key in diagram:
            pairs = diagram[key]
            count = sum(1 for b, d in pairs if b <= radius < d)
            result[f"beta_{key[1]}"] = count
        else:
            result[f"beta_{key[1]}"] = 0
    return result


def persistence_landscape(
    diagram: List[Tuple[float, float]],
    n_bins: int = 20,
    k: int = 1,
) -> np.ndarray:
    """Compute k-th persistence landscape.

    The persistence landscape is a functional summary of persistence
    diagrams, enabling statistical analysis (means, distances, etc.).

    λ_k(t) = k-th largest value of min(t - b, d - t) over all (b,d) pairs.

    Parameters
    ----------
    diagram : list of (birth, death) pairs
    n_bins : int — discretization resolution
    k : int — landscape order (1 = largest, 2 = second largest, ...)

    Returns
    -------
    (n_bins,) — discretized landscape values
    """
    if not diagram:
        return np.zeros(n_bins)

    finite_pairs = [(b, d) for b, d in diagram if np.isfinite(d)]
    if not finite_pairs:
        return np.zeros(n_bins)

    births = np.array([b for b, _ in finite_pairs])
    deaths = np.array([d for _, d in finite_pairs])

    t_min = births.min()
    t_max = deaths.max()
    t_grid = np.linspace(t_min, t_max, n_bins)

    landscape = np.zeros(n_bins)
    for i, t in enumerate(t_grid):
        tent_values = []
        for b, d in finite_pairs:
            val = min(t - b, d - t)
            if val > 0:
                tent_values.append(val)

        tent_values.sort(reverse=True)
        if len(tent_values) >= k:
            landscape[i] = tent_values[k - 1]

    return landscape


# ---------------------------------------------------------------------------
# Topological Sensor Module (PyTorch)
# ---------------------------------------------------------------------------

class TopologicalSensor(nn.Module):
    """Computes topological features from latent representations.

    Takes a sequence of latent states, applies Takens embedding,
    computes persistent homology, and extracts topological features
    that serve as regime indicators.

    Implementation notes:
    - H0 (connected components): exact via single-linkage clustering
    - H1 (cycles/loops): spectral approximation via graph Laplacian
      eigenvalues, NOT exact persistent homology. For exact H1, replace
      compute_persistence() internals with ripser or giotto-tda.

    Features (per window):
    1. beta_0 at median radius — number of connected components
    2. beta_1 at median radius — number of loops/cycles (approximate)
    3. H0 persistence entropy — component complexity
    4. H1 persistence entropy — cycle complexity (approximate)
    5. Max H0 persistence — most persistent component
    6. Max H1 persistence — most persistent cycle (approximate)
    7. H0 landscape integral — total H0 significance
    8. H1 landscape integral — total H1 significance (approximate)
    """

    def __init__(
        self,
        window: int = 50,
        takens_dim: int = 3,
        takens_tau: int = 1,
        max_homology_dim: int = 1,
        n_landscape_bins: int = 20,
    ):
        super().__init__()
        self.window = window
        self.takens_dim = takens_dim
        self.takens_tau = takens_tau
        self.max_dim = max_homology_dim
        self.n_bins = n_landscape_bins

        # Number of topological features
        self.n_features = 8

        # Learnable projection from topo features to model dimension
        self.feature_proj = nn.Linear(self.n_features, self.n_features)

    @torch.no_grad()
    def compute_features(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """Compute topological features from a latent sequence.

        Parameters
        ----------
        latent_seq : (B, L, d_latent) — sequence of latent representations

        Returns
        -------
        (B, n_windows, n_features) — topological features per window
        """
        B, L, D = latent_seq.shape
        device = latent_seq.device

        # Use first principal component of latent space for Takens embedding
        # (could also use multiple PCs for richer topology)
        latent_np = latent_seq.float().cpu().numpy()

        all_features = []
        for b in range(B):
            batch_features = []
            # Sliding windows
            n_windows = max(1, L - self.window + 1)
            for w in range(0, n_windows, max(1, n_windows // 20)):  # subsample windows
                window_data = latent_np[b, w:w + self.window]

                # Use first PC for Takens embedding
                if window_data.shape[1] > 1:
                    # Simple PCA: use SVD
                    centered = window_data - window_data.mean(axis=0)
                    try:
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        pc1 = centered @ Vt[0]
                    except np.linalg.LinAlgError:
                        pc1 = window_data[:, 0]
                else:
                    pc1 = window_data[:, 0]

                # Takens embedding
                try:
                    cloud = takens_embedding(pc1, self.takens_dim, self.takens_tau)
                except ValueError:
                    batch_features.append(np.zeros(self.n_features))
                    continue

                # Persistence
                persistence = compute_persistence(cloud, self.max_dim)

                # Extract features
                feats = np.zeros(self.n_features)

                # Betti numbers at median radius
                all_dists = _pairwise_distances(cloud)
                median_r = np.median(all_dists[all_dists > 0]) if np.any(all_dists > 0) else 1.0
                betti = betti_numbers(persistence, median_r)
                feats[0] = betti.get("beta_0", 0)
                feats[1] = betti.get("beta_1", 0)

                # Persistence entropy
                feats[2] = persistence_entropy(persistence.get("H0", []))
                feats[3] = persistence_entropy(persistence.get("H1", []))

                # Max persistence
                h0_lives = [d - b for b, d in persistence.get("H0", []) if np.isfinite(d)]
                h1_lives = [d - b for b, d in persistence.get("H1", []) if np.isfinite(d)]
                feats[4] = max(h0_lives) if h0_lives else 0.0
                feats[5] = max(h1_lives) if h1_lives else 0.0

                # Landscape integrals
                l0 = persistence_landscape(persistence.get("H0", []), self.n_bins)
                l1 = persistence_landscape(persistence.get("H1", []), self.n_bins)
                feats[6] = l0.sum()
                feats[7] = l1.sum()

                batch_features.append(feats)

            all_features.append(np.array(batch_features) if batch_features
                               else np.zeros((1, self.n_features)))

        # Pad to same length and convert to tensor
        max_windows = max(f.shape[0] for f in all_features)
        padded = np.zeros((B, max_windows, self.n_features))
        for b in range(B):
            n = all_features[b].shape[0]
            padded[b, :n] = all_features[b]

        return torch.tensor(padded, dtype=torch.float32, device=device)

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """Compute and project topological features.

        Parameters
        ----------
        latent_seq : (B, L, d_latent)

        Returns
        -------
        (B, n_windows, n_features) — projected topological features
        """
        raw_features = self.compute_features(latent_seq)
        return self.feature_proj(raw_features)
