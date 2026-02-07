"""Quick status check for the funding paper trader.

Usage:
    python3 -m apps.funding_paper.status
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps.funding_paper.state import compute_performance

STATE_PATH = Path("data/funding_paper_state.json")


def main():
    # Check if process is running
    result = subprocess.run(
        ["pgrep", "-f", "apps.funding_paper"],
        capture_output=True, text=True,
    )
    running = bool(result.stdout.strip())

    if not STATE_PATH.exists():
        print("No state file found. Paper trader has not run yet.")
        return

    data = json.loads(STATE_PATH.read_text())
    now = datetime.now(timezone.utc)

    # Parse timestamps
    started = data.get("started_at", "")
    last_scan = data.get("last_scan_time", "")
    equity = data.get("equity", 1.0)
    positions = data.get("positions", {})

    if last_scan:
        last_dt = datetime.fromisoformat(last_scan)
        age_min = (now - last_dt).total_seconds() / 60
    else:
        age_min = float("inf")

    # Status
    status = "RUNNING" if running else "STOPPED"
    stale = " (STALE — last scan >15min ago)" if age_min > 15 else ""

    print(f"Status: {status}{stale}")
    print(f"Started: {started[:19] if started else '?'}")
    print(f"Last scan: {last_scan[:19] if last_scan else '?'} ({age_min:.0f}min ago)")
    print(f"Equity: {equity:.6f} ({equity - 1:+.4%})")
    print(f"Positions: {len(positions)}")
    print(f"Entries: {data.get('total_entries', 0)}  "
          f"Exits: {data.get('total_exits', 0)}")
    print(f"Funding earned: {data.get('total_funding_earned', 0):+.6f}  "
          f"Costs paid: {data.get('total_costs_paid', 0):.6f}")

    # Performance metrics
    equity_history = data.get("equity_history", [])
    perf = compute_performance(equity_history)
    if perf:
        print(f"\nPerformance ({perf.days_running:.1f} days, {perf.n_snapshots} snapshots):")
        print(f"  Ann return: {perf.ann_return_pct:+.1f}%")
        print(f"  Max DD:     {perf.max_drawdown_pct:.2f}%")
        print(f"  Sharpe:     {perf.sharpe:.2f}")

    # Positions detail
    if positions:
        print(f"\nOpen Positions:")
        total_funding = 0.0
        total_cost = 0.0
        for sym, pos in sorted(positions.items(),
                                key=lambda x: x[1]["accumulated_funding"],
                                reverse=True):
            net = pos["accumulated_funding"] - pos["accumulated_cost"]
            total_funding += pos["accumulated_funding"]
            total_cost += pos["accumulated_cost"]
            print(f"  {sym:15s}  "
                  f"fund={pos['accumulated_funding']:+.6f}  "
                  f"cost={pos['accumulated_cost']:.4f}  "
                  f"net={net:+.6f}  "
                  f"settl={pos['n_settlements']}")
        print(f"  {'':15s}  "
              f"fund={total_funding:+.6f}  "
              f"cost={total_cost:.4f}  "
              f"net={total_funding - total_cost:+.6f}")


if __name__ == "__main__":
    main()
