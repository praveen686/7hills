"""Dynamic Topology Learner — online feature dependency graph.

The topology tracks conditional dependencies between features using an online
EWMA covariance matrix -> regularized precision matrix -> top-K adjacency.

Key design choices:
1. EWMA covariance for non-stationarity (markets change)
2. Precision matrix (inverse covariance) captures CONDITIONAL dependence
   (not just correlation — important because many features are correlated
   but conditionally independent given others)
3. Top-K sparsification with hysteresis prevents topology thrashing
4. Max edge flip rate caps structural changes per step
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class DynamicTopology:
    """Online dynamic topology learner over d features.

    Maintains:
    - EWMA covariance matrix Sigma_t in R^(d x d)
    - Edge score matrix S_t in R^(d x d) (from precision matrix)
    - Binary adjacency A_t in {0,1}^(d x d)
    - Weighted adjacency W_t in R^(d x d)

    Update per step:
    1. Update EWMA covariance with new feature vector
    2. Compute regularized precision P = (Sigma + lambda*I)^(-1)
    3. Edge scores = |P_ij| for i != j (off-diagonal entries)
    4. Apply top-K sparsification with hysteresis
    5. Enforce max edge flip rate
    """

    def __init__(
        self,
        d: int,
        ewma_span: int = 120,
        top_k: int = 6,
        tau_on: float = 0.15,
        tau_off: float = 0.08,
        max_flip_rate: float = 0.02,
        precision_reg: float = 1e-4,
    ):
        self.d = d
        self.alpha = 2.0 / (ewma_span + 1)
        self.top_k = min(top_k, d - 1)
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.max_flip_rate = max_flip_rate
        self.precision_reg = precision_reg

        # Store init params for reset
        self._ewma_span = ewma_span

        # State
        self.mean = np.zeros(d)
        self.cov = np.eye(d)  # EWMA covariance
        self.score = np.zeros((d, d))  # edge score matrix
        self.adjacency = np.zeros((d, d), dtype=np.int8)  # binary
        self.weights = np.zeros((d, d))  # weighted adjacency
        self.n_updates = 0

        # Statistics for monitoring
        self.edge_count_history: list[int] = []
        self.flip_count_history: list[int] = []

    def reset(self):
        """Reset all state."""
        self.__init__(
            self.d,
            self._ewma_span,
            self.top_k,
            self.tau_on,
            self.tau_off,
            self.max_flip_rate,
            self.precision_reg,
        )

    def update(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Update topology with new feature observation.

        x: (d,) feature vector
        mask: (d,) binary mask (1=valid, 0=missing). Missing features are
              excluded from covariance update.

        Returns: (d, d) adjacency matrix
        """
        assert x.shape == (self.d,), f"Expected ({self.d},), got {x.shape}"

        if mask is not None:
            valid_idx = np.where(mask > 0.5)[0]
            if len(valid_idx) < 2:
                return self.adjacency.copy()
        else:
            valid_idx = np.arange(self.d)

        # 1. Update EWMA mean and covariance
        # Use proper EWMA: maintain E[x] and E[xx^T], then Cov = E[xx^T] - E[x]E[x]^T
        if self.n_updates == 0:
            self.mean = x.copy()
            self.second_moment = np.outer(x, x)  # E[xx^T]
            # cov stays as identity until we have enough data
        else:
            # Update second moment: E[xx^T]
            outer = np.outer(x, x)
            if mask is not None:
                valid_mask = np.outer(mask, mask)
                # Only update entries where both features are valid.
                # Invalid entries retain their old second_moment values unchanged.
                old_sm = self.second_moment.copy()
                new_sm = (1 - self.alpha) * old_sm + self.alpha * outer
                self.second_moment = np.where(valid_mask > 0.5, new_sm, old_sm)

                # EWMA mean: only update valid features, keep old for invalid
                old_mean = self.mean.copy()
                new_mean = (1 - self.alpha) * old_mean + self.alpha * x
                self.mean = np.where(mask > 0.5, new_mean, old_mean)
            else:
                # No mask — update everything
                self.mean = (1 - self.alpha) * self.mean + self.alpha * x
                self.second_moment = (1 - self.alpha) * self.second_moment + self.alpha * outer

            # Cov = E[xx^T] - E[x]E[x]^T
            self.cov = self.second_moment - np.outer(self.mean, self.mean)
            # Ensure positive semi-definite (numerical safety)
            np.fill_diagonal(self.cov, np.maximum(np.diag(self.cov), 1e-8))

        self.n_updates += 1

        # 2. Compute precision matrix (regularized inverse)
        # Only do this periodically (every 5 steps after warmup) for efficiency
        if self.n_updates >= 20 and self.n_updates % 5 == 0:
            self._update_precision_and_adjacency()

        return self.adjacency.copy()

    def _update_precision_and_adjacency(self):
        """Compute precision matrix and update adjacency."""
        # Regularized precision
        reg_cov = self.cov + self.precision_reg * np.eye(self.d)

        try:
            precision = np.linalg.inv(reg_cov)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            precision = np.linalg.pinv(reg_cov)

        # Edge scores: |P_ij| for i != j, normalized per COLUMN (incoming edges)
        # We select incoming edges to each node j, so normalize by column max
        # so that scores within each column are comparable.
        raw_scores = np.abs(precision)
        np.fill_diagonal(raw_scores, 0.0)

        col_max = raw_scores.max(axis=0, keepdims=True)  # (1, d) — max incoming per node
        col_max = np.maximum(col_max, 1e-10)
        self.score = raw_scores / col_max

        # 3. Apply top-K with hysteresis
        old_adjacency = self.adjacency.copy()
        new_adjacency = np.zeros_like(self.adjacency)

        for j in range(self.d):
            scores_j = self.score[:, j].copy()  # incoming edges to node j
            scores_j[j] = 0.0  # no self-loops

            # Sort by score (descending)
            ranked = np.argsort(scores_j)[::-1]

            count = 0
            for i in ranked:
                if count >= self.top_k:
                    break

                score_ij = scores_j[i]
                was_active = old_adjacency[i, j] > 0

                # Hysteresis: different thresholds for activation vs deactivation
                if was_active:
                    # Keep active unless score drops below tau_off
                    if score_ij >= self.tau_off:
                        new_adjacency[i, j] = 1
                        count += 1
                else:
                    # Activate only if score exceeds tau_on
                    if score_ij >= self.tau_on:
                        new_adjacency[i, j] = 1
                        count += 1

        # 4. Enforce max edge flip rate
        total_possible = self.d * (self.d - 1)
        max_flips = max(1, int(total_possible * self.max_flip_rate))

        flips = new_adjacency != old_adjacency
        n_flips = flips.sum()

        if n_flips > max_flips:
            # Keep only the most "confident" flips
            flip_indices = np.argwhere(flips)
            flip_scores = np.array([self.score[i, j] for i, j in flip_indices])

            # Sort by score descending — keep top max_flips
            keep_idx = np.argsort(flip_scores)[::-1][:max_flips]

            new_adjacency = old_adjacency.copy()
            for k in keep_idx:
                i, j = flip_indices[k]
                new_adjacency[i, j] = 1 - old_adjacency[i, j]

        self.adjacency = new_adjacency

        # Update weighted adjacency (score * adjacency)
        self.weights = self.score * self.adjacency

        # Track statistics
        self.edge_count_history.append(int(self.adjacency.sum()))
        self.flip_count_history.append(int(n_flips))

    def get_adjacency(self) -> np.ndarray:
        """Get current binary adjacency matrix."""
        return self.adjacency.copy()

    def get_weights(self) -> np.ndarray:
        """Get current weighted adjacency matrix."""
        return self.weights.copy()

    def get_scores(self) -> np.ndarray:
        """Get raw edge score matrix."""
        return self.score.copy()

    def get_stats(self) -> dict:
        """Get topology statistics."""
        return {
            "n_edges": int(self.adjacency.sum()),
            "n_updates": self.n_updates,
            "avg_edges_per_node": float(self.adjacency.sum()) / max(self.d, 1),
            "edge_density": float(self.adjacency.sum()) / max(self.d * (self.d - 1), 1),
            "mean_weight": (
                float(self.weights[self.adjacency > 0].mean())
                if self.adjacency.sum() > 0
                else 0.0
            ),
        }
