"""Risk route — GET /api/risk.

Returns the current risk state: drawdown, exposure, VIX regime,
circuit breaker status, per-strategy risk metrics, portfolio Greeks,
VaR estimates, and drawdown history — all computed from live state.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/risk", tags=["risk"])

IST = timezone(timedelta(hours=5, minutes=30))


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class StrategyRiskOut(BaseModel):
    strategy_id: str
    equity: float
    drawdown_pct: float
    n_positions: int
    exposure: float


class GreeksOut(BaseModel):
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float
    gross_delta: float
    delta_by_symbol: dict[str, float]


class VaROut(BaseModel):
    var_95: float
    var_99: float
    method: str          # "historical" or "parametric_estimate"
    n_observations: int


class DrawdownPointOut(BaseModel):
    date: str
    drawdown_pct: float


class RiskOut(BaseModel):
    portfolio_drawdown_pct: float
    total_exposure: float
    cash: float
    circuit_breaker_active: bool
    last_vpin: float
    last_vix: float
    last_regime: str
    n_positions: int
    n_strategies: int
    strategies: list[StrategyRiskOut]
    concentration: dict[str, Any]
    greeks: GreeksOut
    var: VaROut
    drawdown_history: list[DrawdownPointOut]


# ------------------------------------------------------------------
# Greeks computation (inline, no external dependency needed at route
# level — reuses the same logic as core.risk.greeks but avoids the
# torch import path for the API hot path)
# ------------------------------------------------------------------

def _compute_greeks_from_positions(
    positions: list,
    spot_prices: dict[str, float] | None = None,
) -> GreeksOut:
    """Compute portfolio Greeks from active positions.

    For futures: delta = direction_sign * weight.
    For options: approximate via moneyness-adjusted delta.
      - CE: delta in [0, 1], approximated by 0.5 near ATM.
        Adjusted using simple N(d1) proxy when strike is known.
      - PE: delta in [-1, 0], approximated by -0.5 near ATM.
    Gamma, vega, theta use simplified approximations for options.
    """
    if spot_prices is None:
        spot_prices = {}

    net_delta = 0.0
    net_gamma = 0.0
    net_vega = 0.0
    net_theta = 0.0
    gross_delta = 0.0
    delta_by_symbol: dict[str, float] = {}

    for pos in positions:
        direction_sign = 1.0 if pos.direction == "long" else -1.0
        qty = abs(pos.weight)
        itype = pos.instrument_type

        if itype == "FUT" or itype == "SPREAD":
            # Futures: delta = 1.0 per unit, no gamma/vega/theta
            d = direction_sign * qty
            net_delta += d
            gross_delta += abs(d)
            delta_by_symbol[pos.symbol] = (
                delta_by_symbol.get(pos.symbol, 0.0) + d
            )
            continue

        # Options: CE or PE
        spot = spot_prices.get(pos.symbol, 0.0)
        if spot <= 0:
            spot = pos.current_price if pos.current_price > 0 else pos.entry_price
        if spot <= 0:
            # No price data available; use rough ATM approximation
            raw_delta = 0.5 if itype == "CE" else -0.5
        else:
            strike = pos.strike if pos.strike > 0 else spot
            moneyness = math.log(spot / strike) if strike > 0 else 0.0

            # Compute DTE
            dte = _compute_dte(pos.expiry) if pos.expiry else 7
            T = max(dte / 365.0, 1e-6)

            # Approximate IV (use metadata if available)
            iv = 0.20
            if hasattr(pos, "metadata") and isinstance(pos.metadata, dict):
                iv = pos.metadata.get("iv", 0.20)

            # Simplified N(d1) proxy: d1 ~ moneyness / (iv * sqrt(T))
            vol_sqrt_t = iv * math.sqrt(T)
            if vol_sqrt_t > 0:
                d1 = moneyness / vol_sqrt_t + 0.5 * vol_sqrt_t
            else:
                d1 = 0.0

            # Standard normal CDF approximation (Abramowitz & Stegun)
            nd1 = _norm_cdf(d1)

            if itype == "CE":
                raw_delta = nd1
            else:  # PE
                raw_delta = nd1 - 1.0

            # Gamma: N'(d1) / (S * iv * sqrt(T))
            nprime_d1 = _norm_pdf(d1)
            if spot > 0 and vol_sqrt_t > 0:
                raw_gamma = nprime_d1 / (spot * vol_sqrt_t)
            else:
                raw_gamma = 0.0

            # Vega: S * N'(d1) * sqrt(T) / 100  (per 1% IV move)
            raw_vega = spot * nprime_d1 * math.sqrt(T) / 100.0

            # Theta: simplified approximation
            # theta ~ -S * N'(d1) * iv / (2 * sqrt(T)) / 365
            if T > 1e-6:
                raw_theta = -spot * nprime_d1 * iv / (2.0 * math.sqrt(T)) / 365.0
            else:
                raw_theta = 0.0

            net_gamma += direction_sign * qty * raw_gamma
            net_vega += direction_sign * qty * raw_vega
            net_theta += direction_sign * qty * raw_theta

        d = direction_sign * qty * raw_delta
        net_delta += d
        gross_delta += abs(d)
        delta_by_symbol[pos.symbol] = (
            delta_by_symbol.get(pos.symbol, 0.0) + d
        )

    return GreeksOut(
        net_delta=round(net_delta, 4),
        net_gamma=round(net_gamma, 6),
        net_vega=round(net_vega, 4),
        net_theta=round(net_theta, 4),
        gross_delta=round(gross_delta, 4),
        delta_by_symbol={k: round(v, 4) for k, v in delta_by_symbol.items()},
    )


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (exact to machine precision)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _compute_dte(expiry: str) -> int:
    """Compute days to expiry from an ISO date string."""
    if not expiry:
        return 7
    try:
        from datetime import date as date_cls
        exp_date = date_cls.fromisoformat(expiry)
        today = datetime.now(IST).date()
        dte = (exp_date - today).days
        return max(dte, 1)
    except (ValueError, TypeError):
        return 7


# ------------------------------------------------------------------
# VaR computation
# ------------------------------------------------------------------

def _compute_var(
    equity_history: list[dict],
    portfolio_value: float,
) -> VaROut:
    """Compute Value-at-Risk from historical equity returns.

    VaR 95% = 1.6449 * std(daily_returns) * portfolio_value
    VaR 99% = 2.3263 * std(daily_returns) * portfolio_value

    If insufficient history (< 5 observations), use a parametric
    estimate based on typical India FnO volatility (~20% annualized).
    """
    if len(equity_history) >= 5:
        # Compute daily returns from equity series
        equities = [float(p.get("equity", 0.0)) for p in equity_history]
        returns: list[float] = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                ret = (equities[i] - equities[i - 1]) / equities[i - 1]
                returns.append(ret)

        if len(returns) >= 4:
            # Use ddof=1 for unbiased std (as per MEMORY.md Sharpe protocol)
            mean_ret = sum(returns) / len(returns)
            var_sum = sum((r - mean_ret) ** 2 for r in returns)
            std_ret = math.sqrt(var_sum / (len(returns) - 1))

            if std_ret > 0:
                var_95 = round(1.6449 * std_ret * portfolio_value, 6)
                var_99 = round(2.3263 * std_ret * portfolio_value, 6)
                return VaROut(
                    var_95=var_95,
                    var_99=var_99,
                    method="historical",
                    n_observations=len(returns),
                )

    # Parametric fallback: assume ~20% annualized vol for India FnO
    # Daily vol = 20% / sqrt(252) ~= 1.26%
    daily_vol_estimate = 0.20 / math.sqrt(252)
    var_95 = round(1.6449 * daily_vol_estimate * portfolio_value, 6)
    var_99 = round(2.3263 * daily_vol_estimate * portfolio_value, 6)
    return VaROut(
        var_95=var_95,
        var_99=var_99,
        method="parametric_estimate",
        n_observations=0,
    )


# ------------------------------------------------------------------
# Drawdown history
# ------------------------------------------------------------------

def _compute_drawdown_history(
    equity_history: list[dict],
) -> list[DrawdownPointOut]:
    """Derive drawdown series from equity history.

    For each point, drawdown_pct = (running_max - equity) / running_max * 100.
    """
    if not equity_history:
        return []

    result: list[DrawdownPointOut] = []
    running_max = 0.0

    for point in equity_history:
        eq = float(point.get("equity", 0.0))
        dt = str(point.get("date", ""))

        if eq > running_max:
            running_max = eq

        if running_max > 0:
            dd_pct = round((running_max - eq) / running_max * 100, 4)
        else:
            dd_pct = 0.0

        result.append(DrawdownPointOut(date=dt, drawdown_pct=dd_pct))

    return result


# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------

@router.get("", response_model=RiskOut)
async def get_risk(request: Request) -> RiskOut:
    """Return the current portfolio risk snapshot."""
    state = request.app.state.engine

    # Per-strategy risk breakdown
    strategy_risks: list[StrategyRiskOut] = []
    strategy_ids = set(state.strategy_equity.keys())

    # Also include strategies that have open positions but may not have equity tracked yet
    for pos in state.active_positions():
        strategy_ids.add(pos.strategy_id)

    for sid in sorted(strategy_ids):
        positions = state.positions_by_strategy(sid)
        exposure = sum(abs(p.weight) for p in positions)
        eq = state.strategy_equity.get(sid, 1.0)
        dd = state.strategy_dd(sid)

        strategy_risks.append(
            StrategyRiskOut(
                strategy_id=sid,
                equity=round(eq, 6),
                drawdown_pct=round(dd * 100, 4),
                n_positions=len(positions),
                exposure=round(exposure, 6),
            )
        )

    # Concentration analysis: how much exposure per symbol
    symbol_exposure: dict[str, float] = {}
    for pos in state.active_positions():
        sym = pos.symbol
        symbol_exposure[sym] = symbol_exposure.get(sym, 0.0) + abs(pos.weight)

    # Sort by exposure descending
    top_symbols = dict(
        sorted(symbol_exposure.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    # Direction breakdown
    long_exposure = sum(
        abs(p.weight) for p in state.active_positions() if p.direction == "long"
    )
    short_exposure = sum(
        abs(p.weight) for p in state.active_positions() if p.direction == "short"
    )

    concentration = {
        "top_symbols": {k: round(v, 6) for k, v in top_symbols.items()},
        "long_exposure": round(long_exposure, 6),
        "short_exposure": round(short_exposure, 6),
        "net_exposure": round(long_exposure - short_exposure, 6),
        "gross_exposure": round(long_exposure + short_exposure, 6),
    }

    # Portfolio Greeks — computed from active positions
    active_positions = state.active_positions()

    # Build spot prices dict from positions' current prices
    spot_prices: dict[str, float] = {}
    for pos in active_positions:
        if pos.current_price > 0:
            spot_prices[pos.symbol] = pos.current_price
        elif pos.entry_price > 0:
            spot_prices[pos.symbol] = pos.entry_price

    greeks = _compute_greeks_from_positions(active_positions, spot_prices)

    # VaR from equity history
    equity_history = getattr(state, "equity_history", [])
    var = _compute_var(equity_history, state.equity)

    # Drawdown history from equity curve
    drawdown_history = _compute_drawdown_history(equity_history)

    return RiskOut(
        portfolio_drawdown_pct=round(state.portfolio_dd * 100, 4),
        total_exposure=round(state.total_exposure, 6),
        cash=round(state.cash, 6),
        circuit_breaker_active=state.circuit_breaker_active,
        last_vpin=round(state.last_vpin, 6),
        last_vix=round(state.last_vix, 2),
        last_regime=state.last_regime,
        n_positions=len(state.positions),
        n_strategies=len(strategy_ids),
        strategies=strategy_risks,
        concentration=concentration,
        greeks=greeks,
        var=var,
        drawdown_history=drawdown_history,
    )
