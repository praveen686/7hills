"""
Meta-Labeling and Bet Sizing.

Reference: Lopez de Prado, *Advances in Financial Machine Learning*,
Chapter 3 — Meta-Labeling (Section 3.6) and Bet Sizing (Section 10.4).

**Meta-labeling** decouples two decisions:
  1. *Side* — a primary model predicts direction (+1 / -1).
  2. *Size* — a secondary (meta) model predicts whether the primary
     model's prediction will be profitable (1) or not (0).

This decomposition is powerful because:
  - The primary model can focus on recall (catching opportunities).
  - The meta model boosts precision (filtering false positives).
  - The output probability of the meta model drives **bet sizing**.

**Bet sizing** converts the meta-model's confidence (probability) into
a position size using a sigmoid-like mapping, optionally capped by
``max_leverage``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


def meta_labeling(
    primary_preds: pd.Series | np.ndarray,
    triple_barrier_labels: pd.Series | np.ndarray,
) -> pd.Series:
    """Generate meta-labels from a primary model's predictions and true labels.

    A meta-label is 1 if the primary model's directional call was correct
    (i.e., the sign of the true triple-barrier label matches the primary
    prediction), and 0 otherwise.

    Parameters
    ----------
    primary_preds : pd.Series or np.ndarray
        Primary model's side predictions.  Values should be in {-1, +1}.
        Zero predictions are treated as "no trade" and receive meta-label 0.
    triple_barrier_labels : pd.Series or np.ndarray
        True labels from the triple-barrier method.  Values in {-1, 0, +1}.

    Returns
    -------
    pd.Series
        Meta-labels in {0, 1}, same index as ``primary_preds`` (if Series)
        or integer-indexed (if ndarray).

    Examples
    --------
    >>> import pandas as pd
    >>> preds = pd.Series([1, 1, -1, -1, 1])
    >>> truth = pd.Series([1, -1, -1, 0, 1])
    >>> meta_labeling(preds, truth).tolist()
    [1, 0, 1, 0, 1]
    """
    preds = _to_series(primary_preds, name="primary_preds")
    labels = _to_series(triple_barrier_labels, name="triple_barrier_labels")

    # Align on common index if both are Series with meaningful indices
    if isinstance(primary_preds, pd.Series) and isinstance(triple_barrier_labels, pd.Series):
        preds, labels = preds.align(labels, join="inner")

    # Meta-label: 1 iff primary direction matches the realised direction
    # Zero primary prediction => no trade => meta-label 0
    # Zero true label => neutral outcome => primary was neither right nor wrong => 0
    meta = np.where(
        (preds.values != 0) & (labels.values != 0) & (np.sign(preds.values) == np.sign(labels.values)),
        1,
        0,
    )

    return pd.Series(meta, index=preds.index, name="meta_label", dtype=int)


def bet_size(
    meta_probs: pd.Series | np.ndarray,
    max_leverage: float = 1.0,
    discretize: bool = False,
    step_size: float = 0.0,
) -> pd.Series:
    """Convert meta-model probabilities into position sizes.

    Uses a sigmoid-like mapping based on the inverse CDF (probit):

        z = Phi^{-1}(p)           # map probability to z-score
        size = 2 * Phi(z) - 1     # map back via symmetric sigmoid

    When ``p = 0.5`` (no edge), size = 0.
    When ``p -> 1``, size -> +1.
    When ``p -> 0``, size -> -1 (meta model says primary is wrong).

    The result is then clamped to ``[-max_leverage, +max_leverage]``.

    Parameters
    ----------
    meta_probs : pd.Series or np.ndarray
        Probability that the primary model's prediction is correct.
        Values should be in [0, 1].
    max_leverage : float, default 1.0
        Maximum absolute position size.
    discretize : bool, default False
        If True, discretize the output into steps of ``step_size``.
    step_size : float, default 0.0
        Step size for discretization (ignored if ``discretize=False``).
        E.g., ``step_size=0.25`` produces sizes in {0, 0.25, 0.5, 0.75, 1.0}.

    Returns
    -------
    pd.Series
        Position sizes in ``[-max_leverage, +max_leverage]``.
        Positive = agree with primary model, negative = disagree.

    Notes
    -----
    In typical usage the meta-model outputs P(correct) in [0.5, 1.0].
    Values below 0.5 indicate the meta-model thinks the primary is
    *wrong* — which can be used to take the opposite side.
    """
    probs = _to_series(meta_probs, name="meta_probs")
    p = probs.values.astype(float)

    # Clamp to (epsilon, 1-epsilon) to avoid inf from norm.ppf
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)

    # Probit transform: probability -> z-score -> symmetric bet size
    z = norm.ppf(p)
    size = 2.0 * norm.cdf(z) - 1.0  # equivalent to 2p-1 for normal CDF

    # This simplification (2*Phi(Phi^{-1}(p)) - 1 = 2p - 1) is exact,
    # but we keep the probit form for extensibility (e.g., averaging
    # multiple signals in z-space before converting back).

    # Apply leverage cap
    size = np.clip(size, -max_leverage, max_leverage)

    # Discretize if requested
    if discretize and step_size > 0:
        size = np.round(size / step_size) * step_size
        size = np.clip(size, -max_leverage, max_leverage)

    return pd.Series(size, index=probs.index, name="bet_size")


def avg_active_signals(
    signals: pd.DataFrame,
    molecule: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """Average active signals at each timestamp.

    When multiple concurrent events are active, their individual bet
    sizes should be averaged to avoid over-leveraging.

    Parameters
    ----------
    signals : pd.DataFrame
        Must have columns ``["t1", "size"]`` where ``t1`` is the barrier
        touch time and ``size`` is the bet size for that event.
        Indexed by event start time.
    molecule : pd.DatetimeIndex, optional
        Subset of timestamps to compute.  If *None*, uses all unique
        timestamps between min(index) and max(t1).

    Returns
    -------
    pd.Series
        Average bet size at each timestamp.
    """
    if molecule is None:
        t_min = signals.index.min()
        t_max = signals["t1"].max()
        molecule = pd.date_range(t_min, t_max, freq="B")

    out = pd.Series(dtype=float, index=molecule, name="avg_signal")
    for t in molecule:
        # Active events: started before or at t, barrier not yet touched
        active = signals[(signals.index <= t) & (signals["t1"] >= t)]
        if len(active) > 0:
            out.loc[t] = active["size"].mean()
        else:
            out.loc[t] = 0.0

    return out


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_series(x: pd.Series | np.ndarray, name: str = "") -> pd.Series:
    """Ensure input is a pd.Series."""
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x), name=name)
