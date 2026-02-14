"""Performance metrics for QuantKubera backtesting."""

import numpy as np
import pandas as pd
from typing import Dict, Union


def calculate_sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - (rf_rate / periods)
    return np.sqrt(periods) * (excess_returns.mean() / excess_returns.std())


def calculate_sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> float:
    """Calculate annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (rf_rate / periods)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2 or downside_returns.std() == 0:
        return 0.0
    return np.sqrt(periods) * (excess_returns.mean() / downside_returns.std())


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.exp(np.cumsum(returns))  # Assuming log returns
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(np.min(drawdown))


def calculate_calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
    """Calculate Calmar ratio (Annualized Return / Max Drawdown)."""
    mdd = abs(calculate_max_drawdown(returns))
    if mdd == 0:
        return 0.0
    annual_return = returns.mean() * periods
    return annual_return / mdd


def calculate_all_metrics(returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> Dict[str, float]:
    """Calculate a comprehensive suite of performance metrics."""
    if len(returns) == 0:
        return {}
        
    cum_returns = np.exp(np.sum(returns)) - 1
    annual_ret = returns.mean() * periods
    annual_vol = returns.std() * np.sqrt(periods)
    
    return {
        'total_return': float(cum_returns),
        'annual_return': float(annual_ret),
        'annual_volatility': float(annual_vol),
        'sharpe_ratio': calculate_sharpe_ratio(returns, rf_rate, periods),
        'sortino_ratio': calculate_sortino_ratio(returns, rf_rate, periods),
        'max_drawdown': calculate_max_drawdown(returns),
        'calmar_ratio': calculate_calmar_ratio(returns, periods),
        'win_rate': float(np.mean(returns > 0))
    }
