"""Daily composite scanner — orchestrates data loading and signal computation.

Runs the full pipeline:
  1. Load 20+ days of equity + delivery + F&O + FII/DII data
  2. Compute delivery signals, OI signals, FII flow
  3. Produce ranked composite scores
  4. Return top-N actionable signals
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from apps.india_scanner.data import (
    get_delivery,
    get_equity,
    get_fii_dii,
    get_futures_oi,
    is_trading_day,
)
from apps.india_scanner.signals import (
    CompositeSignal,
    FIIFlowSignal,
    LOOKBACK_DAYS,
    compute_composite_scores,
    compute_delivery_signals,
    compute_fii_flow_signal,
    compute_oi_signals,
)
from apps.india_scanner.universe import get_fno_symbols
from qlx.data.store import MarketDataStore

logger = logging.getLogger(__name__)


def _get_trading_days_before(d: date, n: int) -> list[date]:
    """Get up to n trading days before (and including) d."""
    days = []
    current = d
    while len(days) < n:
        if is_trading_day(current):
            days.append(current)
        current -= timedelta(days=1)
        # Safety: don't go back more than 2x the needed days
        if (d - current).days > n * 3:
            break
    return sorted(days)


def _get_prev_trading_day(d: date) -> date:
    """Get the trading day before d."""
    current = d - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
        if (d - current).days > 10:
            break
    return current


def run_daily_scan(
    target_date: date,
    store=None,
    top_n: int = 10,
    symbols: list[str] | None = None,
) -> list[CompositeSignal]:
    """Run the full daily scan for a given date.

    Loads historical data, computes all three signals, and returns
    top-N composite signals ranked by |score|.
    """
    if store is None:
        store = MarketDataStore()

    if symbols is None:
        symbols = get_fno_symbols()

    symbol_set = set(symbols)

    # Need LOOKBACK_DAYS + 1 trading days of history
    needed_days = LOOKBACK_DAYS + 5  # extra buffer for holidays
    trading_days = _get_trading_days_before(target_date, needed_days)

    if target_date not in trading_days:
        logger.warning("%s is not a trading day", target_date)
        return []

    logger.info("Scanning %s — loading %d days of history", target_date, len(trading_days))

    # Load data
    equity_history: dict[date, pd.DataFrame] = {}
    delivery_history: dict[date, pd.DataFrame] = {}
    fii_dii_history: dict[date, pd.DataFrame] = {}

    for d in trading_days:
        try:
            equity_history[d] = get_equity(store, d)
        except Exception as e:
            logger.debug("No equity data for %s: %s", d, e)

        try:
            delivery_history[d] = get_delivery(store, d)
        except Exception as e:
            logger.debug("No delivery data for %s: %s", d, e)

        try:
            fii_dii_history[d] = get_fii_dii(store, d)
        except Exception as e:
            logger.debug("No FII/DII data for %s: %s", d, e)

    # Signal 1: Delivery
    delivery_signals = compute_delivery_signals(
        equity_history, delivery_history, target_date, symbol_set,
    )
    logger.info("Delivery signals: %d stocks, %d with score != 0",
                len(delivery_signals),
                sum(1 for s in delivery_signals.values() if s.score != 0))

    # Signal 2: OI Buildup
    prev_day = _get_prev_trading_day(target_date)
    oi_signals = {}
    try:
        oi_today = get_futures_oi(store, target_date)
        oi_prev = get_futures_oi(store, prev_day)
        eq_today = equity_history.get(target_date, pd.DataFrame())
        eq_prev = equity_history.get(prev_day, pd.DataFrame())

        oi_signals = compute_oi_signals(
            oi_today, oi_prev, eq_today, eq_prev, symbol_set,
        )
        logger.info("OI signals: %d stocks", len(oi_signals))
    except Exception as e:
        logger.warning("OI signal computation failed: %s", e)

    # Signal 3: FII Flow
    fii_signal = compute_fii_flow_signal(fii_dii_history, target_date)
    if fii_signal:
        logger.info("FII flow: %.0f Cr net (%s)",
                     fii_signal.cumulative_net_inr_cr, fii_signal.regime)

    # Composite
    composites = compute_composite_scores(delivery_signals, oi_signals, fii_signal)

    # Filter to symbols in our universe and take top N
    composites = [c for c in composites if c.symbol in symbol_set]
    top = [c for c in composites if c.composite_score != 0][:top_n]

    logger.info("Top %d signals (of %d non-zero):", len(top),
                sum(1 for c in composites if c.composite_score != 0))
    for c in top:
        logger.info("  %s: composite=%.2f (del=%.1f, oi=%.1f, fii=%.1f)",
                     c.symbol, c.composite_score,
                     c.delivery_score, c.oi_score, c.fii_score)

    return top


def format_scan_results(signals: list[CompositeSignal], target_date: date) -> str:
    """Format scan results as a human-readable table."""
    lines = [
        f"India Institutional Footprint Scan — {target_date}",
        f"{'='*70}",
        f"{'Rank':>4s}  {'Symbol':15s}  {'Composite':>10s}  "
        f"{'Delivery':>9s}  {'OI':>6s}  {'FII':>5s}  {'Direction':>10s}",
        f"{'-'*70}",
    ]

    for i, sig in enumerate(signals, 1):
        direction = "LONG" if sig.composite_score > 0 else "SHORT"
        oi_class = ""
        if sig.oi_signal:
            oi_class = f" ({sig.oi_signal.classification})"

        lines.append(
            f"{i:4d}  {sig.symbol:15s}  {sig.composite_score:+10.2f}  "
            f"{sig.delivery_score:+9.1f}  {sig.oi_score:+6.1f}  "
            f"{sig.fii_score:+5.1f}  {direction:>10s}{oi_class}"
        )

    if not signals:
        lines.append("  No actionable signals found.")

    return "\n".join(lines)
