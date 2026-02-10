"""
AlphaForge Infrastructure — AFML building blocks.

Implements core techniques from Marcos Lopez de Prado's
*Advances in Financial Machine Learning* (AFML):

- Combinatorial Purged K-Fold Cross-Validation (Chapter 12)
- Triple Barrier Labeling (Chapter 3)
- Symmetric CUSUM Filter (Chapter 2)
- Hierarchical Risk Parity (JFE 2016)
- Meta-Labeling & Bet Sizing (Chapter 3)
"""

from .cpcv import CombPurgedKFoldCV
from .triple_barrier import triple_barrier_labels
from .cusum import cusum_filter, get_daily_vol
from .hrp import HierarchicalRiskParity
from .meta_label import meta_labeling, bet_size

__all__ = [
    "CombPurgedKFoldCV",
    "triple_barrier_labels",
    "cusum_filter",
    "get_daily_vol",
    "HierarchicalRiskParity",
    "meta_labeling",
    "bet_size",
]
