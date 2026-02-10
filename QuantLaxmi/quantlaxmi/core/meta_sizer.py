"""Meta-labeling position sizer.

Wraps AFML's meta_labeling() + bet_size() into a single callable
that converts strategy signals + historical labels into position sizes.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from quantlaxmi.models.afml import meta_labeling, bet_size

logger = logging.getLogger(__name__)


class MetaLabelSizer:
    """Convert primary strategy signals into sized positions via meta-labeling.

    The meta-labeling approach:
    1. Primary model predicts direction (+1/-1).
    2. Meta model predicts whether the primary will be correct (probability).
    3. bet_size() converts probability -> position size.

    Parameters
    ----------
    max_leverage : float
        Maximum position size (absolute value).
    discretize : bool
        Whether to discretize bet sizes into steps.
    step_size : float
        Step size for discretization.
    """

    def __init__(
        self,
        max_leverage: float = 1.0,
        discretize: bool = False,
        step_size: float = 0.0,
    ):
        self.max_leverage = max_leverage
        self.discretize = discretize
        self.step_size = step_size

    def compute_meta_labels(
        self,
        primary_preds: pd.Series,
        true_labels: pd.Series,
    ) -> pd.Series:
        """Generate meta-labels from primary predictions and true labels.

        Returns
        -------
        pd.Series
            Meta-labels in {0, 1}.
        """
        return meta_labeling(primary_preds, true_labels)

    def size_positions(
        self,
        meta_probs: pd.Series,
        primary_direction: pd.Series,
    ) -> pd.Series:
        """Convert meta-model probabilities + primary direction into sized positions.

        Parameters
        ----------
        meta_probs : pd.Series
            P(primary is correct) in [0, 1].
        primary_direction : pd.Series
            Primary model's side predictions in {-1, +1}.

        Returns
        -------
        pd.Series
            Signed position sizes: direction * |bet_size|.
        """
        sizes = bet_size(
            meta_probs,
            max_leverage=self.max_leverage,
            discretize=self.discretize,
            step_size=self.step_size,
        )
        # Apply direction: positive size means agree with primary
        signed = sizes * primary_direction
        logger.debug("MetaLabelSizer: mean abs size=%.3f", sizes.abs().mean())
        return signed
