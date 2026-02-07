"""Funding harvester strategy logic.

Implements the entry/exit rules from the research backtest:
  - Entry: smoothed annualized funding > entry_threshold
  - Exit: smoothed annualized funding < exit_threshold
  - Max concurrent positions capped
  - Equal weight per position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from apps.funding_paper.scanner import FundingSnapshot
from apps.funding_paper.state import PortfolioState, Position

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy parameters (from research backtest optimisation)."""

    entry_threshold_pct: float = 20.0   # enter when ann funding > this
    exit_threshold_pct: float = 3.0     # exit when ann funding < this
    max_positions: int = 10
    cost_per_leg_bps: float = 12.0      # entry or exit cost per leg
    min_volume_usd: float = 50_000_000  # minimum 24h volume


@dataclass(frozen=True)
class TradeSignal:
    """A trade decision: enter, exit, or hold."""

    symbol: str
    action: str                  # "enter", "exit", "hold"
    ann_funding_pct: float       # current annualized funding
    reason: str                  # human-readable explanation


def generate_signals(
    state: PortfolioState,
    snapshots: list[FundingSnapshot],
    config: StrategyConfig,
) -> list[TradeSignal]:
    """Generate trade signals from current funding snapshots.

    Uses smoothed funding rates (rolling mean of last N observations)
    for entry/exit decisions to avoid whipsaw from single-snapshot spikes.
    Falls back to raw rate if no history available.

    Returns list of TradeSignal for each actionable decision.
    """
    signals = []
    snapshot_map = {s.symbol: s for s in snapshots}

    # Update funding rate history with latest observations
    state.update_funding_rates({s.symbol: s.funding_rate for s in snapshots})

    def _smoothed_or_raw(sym: str, raw_ann: float) -> float:
        """Get smoothed annualized rate, falling back to raw."""
        smoothed = state.smoothed_ann_funding(sym)
        return smoothed if smoothed is not None else raw_ann

    # --- Check exits first ---
    for sym, pos in list(state.positions.items()):
        snap = snapshot_map.get(sym)
        if snap is None:
            signals.append(TradeSignal(
                symbol=sym,
                action="exit",
                ann_funding_pct=0.0,
                reason="Symbol delisted from premium index",
            ))
            continue

        smooth_ann = _smoothed_or_raw(sym, snap.ann_funding_pct)

        if smooth_ann < config.exit_threshold_pct:
            signals.append(TradeSignal(
                symbol=sym,
                action="exit",
                ann_funding_pct=snap.ann_funding_pct,
                reason=f"Smoothed funding {smooth_ann:.1f}% < {config.exit_threshold_pct}%",
            ))
        else:
            signals.append(TradeSignal(
                symbol=sym,
                action="hold",
                ann_funding_pct=snap.ann_funding_pct,
                reason=f"Smoothed {smooth_ann:.1f}% > exit threshold",
            ))

    # --- Check entries (only if room for new positions) ---
    pending_exits = sum(1 for s in signals if s.action == "exit")
    current_count = len(state.positions) - pending_exits

    if current_count < config.max_positions:
        candidates = []
        for snap in snapshots:
            if snap.symbol in state.positions:
                continue
            smooth_ann = _smoothed_or_raw(snap.symbol, snap.ann_funding_pct)
            if smooth_ann > config.entry_threshold_pct and \
               snap.volume_24h_usd >= config.min_volume_usd:
                candidates.append((snap, smooth_ann))

        # Sort by smoothed funding descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        slots = config.max_positions - current_count
        for snap, smooth_ann in candidates[:slots]:
            signals.append(TradeSignal(
                symbol=snap.symbol,
                action="enter",
                ann_funding_pct=snap.ann_funding_pct,
                reason=f"Smoothed {smooth_ann:.1f}% > {config.entry_threshold_pct}% entry",
            ))

    return signals


def execute_signals(
    state: PortfolioState,
    signals: list[TradeSignal],
    config: StrategyConfig,
) -> list[str]:
    """Apply trade signals to portfolio state.

    Modifies state in-place. Returns list of log messages.
    """
    messages = []
    cost_frac = config.cost_per_leg_bps / 10_000
    now = datetime.now(timezone.utc).isoformat()

    for signal in signals:
        if signal.action == "exit":
            pos = state.positions.pop(signal.symbol, None)
            if pos is not None:
                state.equity *= (1 - cost_frac)
                state.total_costs_paid += cost_frac
                state.total_exits += 1
                msg = (f"EXIT  {signal.symbol:15s}  "
                       f"funding={signal.ann_funding_pct:+5.1f}%  "
                       f"pnl={pos.net_pnl:+.4f}  "
                       f"settlements={pos.n_settlements}  "
                       f"{signal.reason}")
                messages.append(msg)
                logger.info(msg)

        elif signal.action == "enter":
            # Equal weight across target position count
            n_after = len(state.positions) + 1
            weight = 1.0 / max(n_after, 1)

            state.equity *= (1 - cost_frac)
            state.total_costs_paid += cost_frac
            state.total_entries += 1

            state.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                entry_time=now,
                entry_ann_funding=signal.ann_funding_pct,
                notional_weight=weight,
                accumulated_cost=cost_frac,
            )

            msg = (f"ENTER {signal.symbol:15s}  "
                   f"funding={signal.ann_funding_pct:+5.1f}%  "
                   f"weight={weight:.2f}  "
                   f"{signal.reason}")
            messages.append(msg)
            logger.info(msg)

    # Rebalance weights if positions changed
    if state.positions:
        w = 1.0 / len(state.positions)
        for pos in state.positions.values():
            pos.notional_weight = w

    # Record equity snapshot on any trade event
    had_trades = any(s.action in ("enter", "exit") for s in signals)
    if had_trades:
        state.record_equity("trade")

    state.last_scan_time = now
    return messages


def record_funding_settlement(
    state: PortfolioState,
    snapshots: list[FundingSnapshot],
) -> list[str]:
    """Record a funding settlement — credit funding to open positions.

    Called when we detect that a funding settlement has occurred.
    """
    messages = []
    snapshot_map = {s.symbol: s for s in snapshots}

    for sym, pos in state.positions.items():
        snap = snapshot_map.get(sym)
        if snap is None:
            continue

        funding_earned = snap.funding_rate * pos.notional_weight
        pos.accumulated_funding += funding_earned
        pos.n_settlements += 1
        state.total_funding_earned += funding_earned
        state.equity *= (1 + funding_earned)

        msg = (f"FUND  {sym:15s}  "
               f"rate={snap.funding_rate:+.6f}  "
               f"earned={funding_earned:+.6f}  "
               f"total={pos.accumulated_funding:+.4f}")
        messages.append(msg)
        logger.debug(msg)

    if state.positions:
        state.record_equity("settlement")

    return messages
