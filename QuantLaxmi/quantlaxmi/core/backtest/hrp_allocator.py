"""HRP-based portfolio allocator for strategy ensembles.

Wraps AFML's HierarchicalRiskParity to allocate across strategies
based on their return covariance structure.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from quantlaxmi.models.afml import HierarchicalRiskParity

logger = logging.getLogger(__name__)


class HRPAllocator:
    """Allocate weights across strategies using Hierarchical Risk Parity.

    Parameters
    ----------
    lookback : int
        Number of trading days of return history to use for covariance estimation.
    linkage_method : str
        Linkage method for hierarchical clustering.
    min_history : int
        Minimum number of observations required before producing weights.
    """

    def __init__(
        self,
        lookback: int = 63,
        linkage_method: str = "ward",
        min_history: int = 21,
    ):
        self.lookback = lookback
        self.min_history = min_history
        self._hrp = HierarchicalRiskParity(linkage_method=linkage_method)

    def allocate(self, strategy_returns: pd.DataFrame) -> dict[str, float]:
        """Compute HRP weights from a matrix of strategy daily returns.

        Parameters
        ----------
        strategy_returns : pd.DataFrame
            Columns are strategy names, rows are daily returns.
            Most recent `lookback` rows are used.

        Returns
        -------
        dict[str, float]
            Strategy name -> portfolio weight (non-negative, sum to 1.0).
            Falls back to equal-weight if insufficient history.
        """
        n_strategies = strategy_returns.shape[1]
        if n_strategies == 0:
            return {}

        if n_strategies == 1:
            return {strategy_returns.columns[0]: 1.0}

        # Use most recent lookback window
        recent = strategy_returns.tail(self.lookback).dropna()

        if len(recent) < self.min_history:
            logger.warning(
                "HRP: insufficient history (%d < %d), using equal weight",
                len(recent), self.min_history,
            )
            w = 1.0 / n_strategies
            return {col: w for col in strategy_returns.columns}

        cov = recent.cov()
        weights = self._hrp.allocate(cov, returns=recent)

        logger.info("HRP weights: %s", {k: round(v, 4) for k, v in weights.items()})
        return weights
