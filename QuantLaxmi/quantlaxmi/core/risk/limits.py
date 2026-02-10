"""Risk limit configuration for BRAHMASTRA.

All thresholds in one place.  Immutable via frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    """Risk parameter configuration.

    Attributes
    ----------
    max_portfolio_dd : float
        Maximum portfolio drawdown before circuit breaker (default 5%).
    max_strategy_dd : float
        Maximum single-strategy drawdown before that strategy is disabled (3%).
    max_single_instrument : float
        Maximum portfolio weight in any single instrument (20%).
    max_single_stock_fno : float
        Maximum weight in a single stock FnO name (5%).
    vpin_block_threshold : float
        VPIN level above which all new entries are blocked (0.7).
    max_total_exposure : float
        Maximum gross exposure as fraction of equity (1.5 = 150%).
    max_correlated_exposure : float
        Maximum weight in correlated positions (same direction, same sector) (40%).
    """

    max_portfolio_dd: float = 0.05
    max_strategy_dd: float = 0.03
    max_single_instrument: float = 0.20
    max_single_stock_fno: float = 0.05
    vpin_block_threshold: float = 0.70
    max_total_exposure: float = 1.50
    max_correlated_exposure: float = 0.40
    max_portfolio_vega: float = 50000.0
    max_portfolio_theta: float = -25000.0
