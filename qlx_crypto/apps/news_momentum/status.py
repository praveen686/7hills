"""Quick status check for the news momentum paper trader.

Usage: python -m apps.news_momentum.status
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from apps.news_momentum.state import DEFAULT_STATE_FILE, TradingState


def main():
    state = TradingState.load(DEFAULT_STATE_FILE)
    now = datetime.now(timezone.utc)

    started = datetime.fromisoformat(state.started_at) if state.started_at else now
    uptime = now - started

    print(f"News Momentum Paper Trader — Status")
    print(f"{'=' * 50}")
    print(f"  Started:    {state.started_at}")
    print(f"  Uptime:     {uptime}")
    print(f"  Total:      {state.total_trades} trades")
    print(f"  Active:     {len(state.active_trades)}")
    print(f"  Closed:     {len(state.closed_trades)}")
    print(f"  Win rate:   {state.win_rate():.1f}%")
    print(f"  Avg P&L:    {state.avg_pnl():+.3f}%")
    print(f"  Total P&L:  {state.total_pnl():+.3f}%")

    if state.active_trades:
        print(f"\n  Active trades:")
        for sym, t in sorted(state.active_trades.items()):
            age = t.age_minutes()
            print(f"    {sym:<12} {t.direction:>5} @ {t.entry_price:.4f}  "
                  f"age={age:.1f}m  score={t.score:+.2f}")

    if state.closed_trades:
        recent = state.closed_trades[-5:]
        print(f"\n  Recent exits:")
        for ct in reversed(recent):
            print(f"    {ct.symbol:<12} {ct.direction:>5} "
                  f"P&L={ct.pnl_pct:+.3f}%  {ct.reason}")


if __name__ == "__main__":
    main()
