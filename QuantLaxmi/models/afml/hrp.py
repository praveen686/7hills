"""
Hierarchical Risk Parity (HRP) Portfolio Optimizer.

Reference: Lopez de Prado, "Building Diversified Portfolios that
Outperform Out-of-Sample", *Journal of Portfolio Management*, 2016.

HRP replaces the fragile covariance-matrix inversion of Markowitz with
a three-step procedure that is:
  1. Numerically stable (no matrix inversion).
  2. Robust to estimation error (tree structure regularizes).
  3. Guaranteed to produce positive weights that sum to 1.

Steps:
  (1) **Tree clustering** — compute a distance matrix from the
      correlation matrix and apply Ward's agglomerative clustering.
  (2) **Quasi-diagonalization** — reorder assets so that correlated
      assets are adjacent (seriation via dendrogram leaf order).
  (3) **Recursive bisection** — split the sorted list in half,
      allocate inversely proportional to cluster variance, and recurse.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    """HRP portfolio optimizer.

    Parameters
    ----------
    linkage_method : str, default "ward"
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
        Common choices: ``"single"``, ``"complete"``, ``"average"``,
        ``"ward"``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> ret = pd.DataFrame(rng.standard_normal((252, 5)),
    ...                    columns=["A", "B", "C", "D", "E"])
    >>> hrp = HierarchicalRiskParity()
    >>> w = hrp.allocate(ret.cov(), returns=ret)
    >>> abs(sum(w.values()) - 1.0) < 1e-10
    True
    """

    def __init__(self, linkage_method: str = "ward") -> None:
        self.linkage_method = linkage_method
        self._link: Optional[np.ndarray] = None  # stored linkage matrix
        self._sorted_idx: Optional[list[int]] = None

    def allocate(
        self,
        cov_matrix: pd.DataFrame | np.ndarray,
        returns: Optional[pd.DataFrame | np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute HRP weights.

        Parameters
        ----------
        cov_matrix : pd.DataFrame or np.ndarray, shape (n, n)
            Covariance matrix of asset returns.  If a DataFrame, column
            names are used as asset labels.
        returns : pd.DataFrame or np.ndarray, optional
            Historical return matrix (T x n).  Used only to compute the
            correlation matrix for clustering.  If *None*, the correlation
            is derived from ``cov_matrix``.

        Returns
        -------
        dict[str, float]
            Mapping of asset name (or integer index) to portfolio weight.
            Weights are non-negative and sum to 1.0.
        """
        cov, labels = self._prepare_cov(cov_matrix)
        n = cov.shape[0]

        if n == 1:
            return {labels[0]: 1.0}

        # --- Correlation matrix ------------------------------------------
        if returns is not None:
            if isinstance(returns, pd.DataFrame):
                corr = returns.corr().values.copy()
            else:
                corr = np.corrcoef(returns, rowvar=False).copy()
        else:
            corr = self._cov_to_corr(cov)

        # Clip to valid range (numerical noise can push beyond [-1, 1])
        np.clip(corr, -1.0, 1.0, out=corr)

        # --- Step 1: Tree Clustering ------------------------------------
        dist = self._corr_to_distance(corr)
        condensed = squareform(dist, checks=False)

        # Handle degenerate cases (zero-distance pairs from identical assets)
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=1e10, neginf=0.0)

        self._link = linkage(condensed, method=self.linkage_method)

        # --- Step 2: Quasi-Diagonalization (seriation) -------------------
        self._sorted_idx = list(leaves_list(self._link).astype(int))

        # --- Step 3: Recursive Bisection ---------------------------------
        weights = np.ones(n)
        self._recursive_bisect(cov, self._sorted_idx, weights)

        # Normalize (should already sum to ~1, but defensive)
        total = weights.sum()
        if total > 0:
            weights /= total

        return {labels[i]: float(weights[i]) for i in range(n)}

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_cov(
        cov_matrix: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract numpy array and labels from covariance input."""
        if isinstance(cov_matrix, pd.DataFrame):
            labels = list(cov_matrix.columns.astype(str))
            cov = cov_matrix.values.copy()
        else:
            cov = np.array(cov_matrix, dtype=float).copy()
            labels = [str(i) for i in range(cov.shape[0])]

        # Handle singular / near-singular matrices by adding small ridge
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-12:
            ridge = max(1e-10, abs(eigvals.min()) * 2)
            cov += np.eye(cov.shape[0]) * ridge

        return cov, labels

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-12  # avoid division by zero
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)
        return corr

    @staticmethod
    def _corr_to_distance(corr: np.ndarray) -> np.ndarray:
        """Convert correlation matrix to distance matrix.

        Uses the standard AFML distance: d(i,j) = sqrt(0.5 * (1 - rho(i,j))).
        """
        # Clamp to avoid negative values under sqrt
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def _cluster_variance(cov: np.ndarray, cluster_idx: list[int]) -> float:
        """Compute the variance of the inverse-variance portfolio within a cluster.

        This is the variance of the minimum-variance (1/var) portfolio
        for the assets in ``cluster_idx``.
        """
        sub_cov = cov[np.ix_(cluster_idx, cluster_idx)]
        diag_var = np.diag(sub_cov)
        # Inverse-variance weights (within cluster)
        inv_var = 1.0 / np.maximum(diag_var, 1e-16)
        w = inv_var / inv_var.sum()
        return float(w @ sub_cov @ w)

    def _recursive_bisect(
        self,
        cov: np.ndarray,
        sorted_idx: list[int],
        weights: np.ndarray,
    ) -> None:
        """Recursively bisect the quasi-diagonalized asset list and allocate weights.

        Allocation rule at each split:
            alpha = 1 - V_left / (V_left + V_right)
        Left cluster gets weight ``alpha``, right gets ``1 - alpha``.
        """
        if len(sorted_idx) <= 1:
            return

        mid = len(sorted_idx) // 2
        left = sorted_idx[:mid]
        right = sorted_idx[mid:]

        v_left = self._cluster_variance(cov, left)
        v_right = self._cluster_variance(cov, right)

        total_var = v_left + v_right
        if total_var < 1e-30:
            alpha = 0.5
        else:
            alpha = 1.0 - v_left / total_var

        # Scale weights
        for idx in left:
            weights[idx] *= alpha
        for idx in right:
            weights[idx] *= (1.0 - alpha)

        # Recurse
        self._recursive_bisect(cov, left, weights)
        self._recursive_bisect(cov, right, weights)
