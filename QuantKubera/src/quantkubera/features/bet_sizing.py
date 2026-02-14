"""
Bet Sizing logic for QuantKubera.
Processes meta-model probabilities into position sizes.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def bet_size(
    meta_probs: pd.Series,
    max_leverage: float = 1.0,
    discretize: bool = False,
    step_size: float = 0.0,
) -> pd.Series:
    """Convert meta-model probabilities into position sizes.
    
    Uses a probit transform (inverse CDF of normal distribution) to map 
    probabilities to a symmetric bet size in [-max_leverage, +max_leverage].
    
    p = 0.5 => size = 0
    p = 1.0 => size = +max_leverage
    p = 0.0 => size = -max_leverage
    """
    p = meta_probs.values.astype(float)
    
    # Clamp to avoid numerical issues (inf)
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)
    
    # Probit transform: probability -> z-score -> 2p-1 mapping
    # This is the symmetric bet sizing formula from Lopez de Prado
    z = norm.ppf(p)
    size = 2.0 * norm.cdf(z) - 1.0
    
    # Apply leverage cap
    size = size * max_leverage
    
    # Discretize if requested (e.g., 0.25 steps)
    if discretize and step_size > 0:
        size = np.round(size / step_size) * step_size
        size = np.clip(size, -max_leverage, max_leverage)
        
    return pd.Series(size, index=meta_probs.index, name="bet_size")
