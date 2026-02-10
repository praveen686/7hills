"""S12 Vedic FFPE — Intraday entry/exit engine.

Processes 1-minute bars in streaming fashion:
  - Entry: composite signal > threshold AND phase is favorable
  - Exit: target hit (2σ), stop-loss (1.5σ), time-stop (last 30 min), signal flip
  - Maximum 3 trades per day per instrument
  - Cost: 5 bps per roundtrip
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from quantlaxmi.strategies.s12_vedic_ffpe.signals import compute_daily_signal


@dataclass
class IntradayTrade:
    """Record of a single intraday trade."""
    entry_bar: int
    entry_price: float
    direction: str
    conviction: float
    regime: str
    exit_bar: int | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0


@dataclass
class IntradayState:
    """Mutable state for the intraday engine."""
    position: Literal["long", "short", "flat"] = "flat"
    entry_price: float = 0.0
    entry_bar: int = 0
    conviction: float = 0.0
    regime: str = "normal"
    trades_today: int = 0
    trades: list[IntradayTrade] = field(default_factory=list)
    # Signal state carried forward
    prev_regime: str = "normal"
    bars_in_regime: int = 0
    regime_centroids: dict | None = None


def run_intraday(
    bars_close: np.ndarray,
    bars_high: np.ndarray,
    bars_low: np.ndarray,
    *,
    alpha_window: int = 60,
    signal_lookback: int = 60,
    entry_threshold: float = 0.20,
    target_sigma: float = 2.0,
    stop_sigma: float = 1.5,
    max_trades: int = 3,
    cost_bps: float = 5.0,
    last_n_bars_flat: int = 30,
    total_bars: int | None = None,
    alpha_lo: float = 0.85,
    alpha_hi: float = 1.15,
    frac_d: float = 0.226,
) -> IntradayState:
    """Run the intraday engine on 1-minute bars for a single day.

    Parameters
    ----------
    bars_close, bars_high, bars_low : 1-D arrays of 1-min bar data
    alpha_window : lookback for α estimation (in daily bars, approximated)
    signal_lookback : how many bars of history to use for signal
    entry_threshold : minimum conviction to enter
    target_sigma, stop_sigma : exit thresholds in rolling σ units
    max_trades : maximum trades per day
    cost_bps : roundtrip transaction cost in basis points
    last_n_bars_flat : flatten position in last N bars of day
    total_bars : total bars in day (for time-stop); defaults to len(bars_close)
    alpha_lo, alpha_hi : regime classification thresholds
    frac_d : fractional differencing order

    Returns
    -------
    IntradayState with completed trades and their PnL
    """
    n = len(bars_close)
    if total_bars is None:
        total_bars = n

    cost_frac = cost_bps / 10_000

    state = IntradayState()

    # Need sufficient history for signal computation
    min_bars = max(signal_lookback + 5, alpha_window + 5)

    for i in range(min_bars, n):
        # Use all closing prices up to bar i (ensures enough data for alpha_window)
        history = bars_close[:i + 1]

        # Compute signal
        signal, sig_state = compute_daily_signal(
            history,
            alpha_window=alpha_window,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            frac_d=frac_d,
            min_conviction=0.0,  # we apply threshold ourselves
            prev_regime=state.prev_regime,
            bars_in_regime=state.bars_in_regime,
            regime_centroids=state.regime_centroids,
        )
        state.prev_regime = sig_state["prev_regime"]
        state.bars_in_regime = sig_state["bars_in_regime"]
        state.regime_centroids = sig_state.get("regime_centroids")

        current_price = bars_close[i]

        # --- Position management ---
        if state.position != "flat":
            # Check exits
            bars_held = i - state.entry_bar

            # Rolling volatility for target/stop
            recent = bars_close[max(0, i - 20):i + 1]
            if len(recent) > 2:
                log_ret = np.diff(np.log(np.maximum(recent, 1e-8)))
                sigma = np.std(log_ret, ddof=1) * state.entry_price
            else:
                sigma = state.entry_price * 0.001

            sigma = max(sigma, state.entry_price * 0.0001)

            pnl_raw = (current_price - state.entry_price) if state.position == "long" \
                else (state.entry_price - current_price)

            exit_reason = ""

            # Target: 2σ
            if pnl_raw >= target_sigma * sigma:
                exit_reason = "target"

            # Stop-loss: 1.5σ
            elif pnl_raw <= -stop_sigma * sigma:
                exit_reason = "stop"

            # Time-stop: last 30 min of day
            elif i >= total_bars - last_n_bars_flat:
                exit_reason = "time_stop"

            # Signal flip
            elif signal.direction != "flat" and signal.direction != state.position \
                    and signal.conviction >= entry_threshold:
                exit_reason = "signal_flip"

            if exit_reason:
                pnl_net = (pnl_raw / state.entry_price) - cost_frac
                trade = IntradayTrade(
                    entry_bar=state.entry_bar,
                    entry_price=state.entry_price,
                    direction=state.position,
                    conviction=state.conviction,
                    regime=state.regime,
                    exit_bar=i,
                    exit_price=current_price,
                    exit_reason=exit_reason,
                    pnl=pnl_net,
                )
                state.trades.append(trade)
                state.position = "flat"
                state.entry_price = 0.0

        # --- Entry logic ---
        if state.position == "flat" and state.trades_today < max_trades:
            # Don't enter in last N bars
            if i >= total_bars - last_n_bars_flat:
                continue

            if signal.direction in ("long", "short") and signal.conviction >= entry_threshold:
                state.position = signal.direction
                state.entry_price = current_price
                state.entry_bar = i
                state.conviction = signal.conviction
                state.regime = signal.regime
                state.trades_today += 1

    # Force-close any open position at end of day
    if state.position != "flat":
        current_price = bars_close[-1]
        pnl_raw = (current_price - state.entry_price) if state.position == "long" \
            else (state.entry_price - current_price)
        pnl_net = (pnl_raw / state.entry_price) - cost_frac
        trade = IntradayTrade(
            entry_bar=state.entry_bar,
            entry_price=state.entry_price,
            direction=state.position,
            conviction=state.conviction,
            regime=state.regime,
            exit_bar=n - 1,
            exit_price=current_price,
            exit_reason="eod_close",
            pnl=pnl_net,
        )
        state.trades.append(trade)
        state.position = "flat"

    return state
