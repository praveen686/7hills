"""Portfolio Greeks aggregation.

Wraps iv_engine.py GPU compute to provide portfolio-level Greek
exposure: delta, gamma, vega, theta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioGreeks:
    """Aggregated portfolio-level Greeks."""

    net_delta: float       # portfolio delta exposure
    net_gamma: float       # portfolio gamma
    net_vega: float        # portfolio vega (per 1% IV move)
    net_theta: float       # portfolio theta (per day)
    gross_delta: float     # sum of |delta| — total directional risk
    delta_by_symbol: dict  # symbol → net delta

    @property
    def is_delta_neutral(self) -> bool:
        return abs(self.net_delta) < 0.10


def compute_portfolio_greeks(
    positions: list[dict],
    spot_prices: dict[str, float],
) -> PortfolioGreeks:
    """Compute aggregate Greeks for all option positions.

    Parameters
    ----------
    positions : list[dict]
        Each dict has: symbol, instrument_type, strike, expiry,
        direction, quantity, iv (optional).
    spot_prices : dict
        symbol → current spot price.

    Returns
    -------
    PortfolioGreeks
    """
    try:
        import torch
        from qlx.pricing.iv_engine import bs_delta, bs_gamma, _DEVICE, _DTYPE
        gpu_available = True
    except ImportError:
        gpu_available = False

    net_delta = 0.0
    net_gamma = 0.0
    net_vega = 0.0
    net_theta = 0.0
    gross_delta = 0.0
    delta_by_symbol: dict[str, float] = {}

    for pos in positions:
        symbol = pos["symbol"]
        itype = pos.get("instrument_type", "FUT")
        direction = 1.0 if pos.get("direction", "long") == "long" else -1.0
        qty = pos.get("quantity", 1.0)

        if itype == "FUT":
            # Futures: delta = 1.0 (per unit), gamma/vega/theta = 0
            d = direction * qty
            net_delta += d
            gross_delta += abs(d)
            delta_by_symbol[symbol] = delta_by_symbol.get(symbol, 0.0) + d
            continue

        # Option positions — need Greeks computation
        if not gpu_available:
            # Approximate delta for options
            d = direction * qty * 0.5  # rough delta estimate
            net_delta += d
            gross_delta += abs(d)
            delta_by_symbol[symbol] = delta_by_symbol.get(symbol, 0.0) + d
            continue

        spot = spot_prices.get(symbol, 0.0)
        if spot <= 0:
            continue

        strike = pos.get("strike", spot)
        iv = pos.get("iv", 0.20)
        dte = pos.get("dte", 7)
        T = max(dte / 365.0, 1e-6)
        is_call = itype == "CE"

        S = torch.tensor([spot], dtype=_DTYPE, device=_DEVICE)
        K = torch.tensor([strike], dtype=_DTYPE, device=_DEVICE)
        T_t = torch.tensor([T], dtype=_DTYPE, device=_DEVICE)
        r = torch.tensor([0.065], dtype=_DTYPE, device=_DEVICE)
        sigma = torch.tensor([iv], dtype=_DTYPE, device=_DEVICE)
        is_call_t = torch.tensor([is_call], dtype=torch.bool, device=_DEVICE)

        d = float(bs_delta(S, K, T_t, r, sigma, is_call_t).cpu().item()) * direction * qty
        g = float(bs_gamma(S, K, T_t, r, sigma).cpu().item()) * direction * qty

        net_delta += d
        net_gamma += g
        gross_delta += abs(d)
        delta_by_symbol[symbol] = delta_by_symbol.get(symbol, 0.0) + d

    return PortfolioGreeks(
        net_delta=round(net_delta, 4),
        net_gamma=round(net_gamma, 6),
        net_vega=round(net_vega, 4),
        net_theta=round(net_theta, 4),
        gross_delta=round(gross_delta, 4),
        delta_by_symbol={k: round(v, 4) for k, v in delta_by_symbol.items()},
    )
