"""Quick status check for the India scanner.

Shows: current positions, recent signals, data freshness, performance.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from apps.india_scanner.data import available_dates
from apps.india_scanner.state import ScannerState, compute_performance


def format_status(
    state: ScannerState,
    store=None,
) -> str:
    """Format a quick status report."""
    lines = [
        "India Institutional Footprint Scanner — Status",
        "=" * 60,
    ]

    # Portfolio
    ret_pct = (state.equity / state.initial_equity - 1) * 100
    lines.append(f"  Equity: Rs {state.equity:,.0f} ({ret_pct:+.2f}%)")
    lines.append(f"  Positions: {len(state.positions)}")
    lines.append(f"  Total entries: {state.total_entries}")
    lines.append(f"  Total exits: {state.total_exits}")
    lines.append(f"  Last scan: {state.last_scan_date or 'never'}")
    lines.append(f"  Started: {state.started_at[:10] if state.started_at else 'N/A'}")

    # Open positions
    if state.positions:
        lines.append(f"\n  Open Positions:")
        lines.append(f"  {'Symbol':15s} {'Dir':>5s} {'Entry':>10s} "
                      f"{'Price':>10s} {'Days':>5s} {'Score':>6s}")
        lines.append(f"  {'-'*55}")
        for sym, pos in sorted(state.positions.items()):
            lines.append(
                f"  {pos.symbol:15s} {pos.direction:>5s} "
                f"{pos.entry_date:>10s} {pos.entry_price:10.1f} "
                f"{pos.days_held:5d} {pos.composite_score:+6.2f}"
            )

    # Performance
    perf = compute_performance(state.equity_history, state.closed_trades)
    if perf:
        lines.append(f"\n  Performance:")
        lines.append(f"    Sharpe:    {perf.sharpe:.2f}")
        lines.append(f"    Max DD:    {perf.max_drawdown_pct:.2f}%")
        lines.append(f"    Win rate:  {perf.win_rate:.1%}")
        lines.append(f"    Days:      {perf.days_running:.0f}")

    # Data freshness
    if store is not None:
        equity_dates = available_dates(store, "nse_delivery")
        if equity_dates:
            latest = equity_dates[-1]
            lines.append(f"\n  Data: {len(equity_dates)} days available, "
                          f"latest: {latest}")
        else:
            lines.append(f"\n  Data: no data available — run nse_daily collector first")

    return "\n".join(lines)
