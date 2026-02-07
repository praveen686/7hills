"""Position sizing via Fractional Kelly Criterion.

Uses quarter-Kelly (f_kelly × 0.25) as the default fraction, which
provides roughly 75% of the Kelly growth rate with dramatically lower
variance and drawdown.

Kelly formula:  f* = (p × W − (1 − p) × L) / W

Where:
  p = win probability
  W = average win size
  L = average loss size
  f* = fraction of capital to risk

Quarter-Kelly: f = 0.25 × f*
"""

from __future__ import annotations

import math


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_mult: float = 0.25,
    max_fraction: float = 0.20,
) -> float:
    """Compute fractional Kelly position size.

    Parameters
    ----------
    win_rate : float
        Historical win probability in [0, 1].
    avg_win : float
        Average winning trade return (positive, e.g. 0.02 for 2%).
    avg_loss : float
        Average losing trade magnitude (positive, e.g. 0.01 for 1%).
    kelly_mult : float
        Kelly fraction multiplier (default 0.25 = quarter-Kelly).
    max_fraction : float
        Hard cap on position size as fraction of portfolio.

    Returns
    -------
    float
        Recommended position fraction in [0, max_fraction].
    """
    if avg_win <= 0 or avg_loss <= 0 or win_rate <= 0:
        return 0.0

    # Edge = expected value per dollar risked
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss
    if edge <= 0:
        return 0.0  # negative edge → no bet

    # Full Kelly fraction
    f_star = edge / avg_win

    # Apply fractional Kelly and cap
    f = kelly_mult * f_star
    return max(0.0, min(f, max_fraction))


def conviction_to_size(
    conviction: float,
    base_fraction: float,
    max_fraction: float = 0.20,
) -> float:
    """Scale position size by signal conviction.

    Parameters
    ----------
    conviction : float
        Signal conviction in [0, 1].
    base_fraction : float
        Base Kelly fraction from kelly_fraction().
    max_fraction : float
        Maximum position size.

    Returns
    -------
    float
        Adjusted position fraction.
    """
    return min(conviction * base_fraction, max_fraction)
