"""Funding Harvester — Live Paper Trader.

Runs the funding rate arbitrage strategy in paper-trading mode:
  1. Scans Binance for current funding rates every SCAN_INTERVAL
  2. Detects funding settlements (every 8h) and credits positions
  3. Enters new positions when annualized funding > 20%
  4. Exits positions when funding drops below 3%
  5. Displays live dashboard in the terminal

State persists to JSON — survives restarts without losing positions.

Usage:
    python3 -m apps.funding_paper
    python3 -m apps.funding_paper --scan-interval 120  # scan every 2min
    python3 -m apps.funding_paper --entry 15 --exit 5  # custom thresholds
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from apps.funding_paper.scanner import FundingSnapshot, scan_funding
from apps.funding_paper.state import PortfolioState, compute_performance
from apps.funding_paper.strategy import (
    StrategyConfig,
    execute_signals,
    generate_signals,
    record_funding_settlement,
)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def clear_screen():
    print("\033[2J\033[H", end="")


def render_dashboard(
    state: PortfolioState,
    snapshots: list[FundingSnapshot],
    config: StrategyConfig,
    messages: list[str],
    scan_count: int,
    state_path: Path,
):
    """Render the terminal dashboard."""
    clear_screen()
    now = datetime.now(timezone.utc)

    # Header
    print("=" * 78)
    print("  FUNDING HARVESTER — Paper Trading Engine")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  Scan #{scan_count}")
    print("=" * 78)

    # Portfolio summary
    pnl = state.equity - 1.0
    pnl_sign = "+" if pnl >= 0 else ""
    print(f"\n  Equity: {state.equity:.6f}  ({pnl_sign}{pnl:.4%})")
    print(f"  Positions: {len(state.positions)}/{config.max_positions}  "
          f"| Entries: {state.total_entries}  Exits: {state.total_exits}")
    print(f"  Funding earned: {state.total_funding_earned:+.6f}  "
          f"| Costs paid: {state.total_costs_paid:.6f}")

    # Performance stats (from equity history)
    perf = compute_performance(state.equity_history)
    if perf is not None:
        print(f"  {perf.days_running:.1f}d running | "
              f"Ann: {perf.ann_return_pct:+.1f}% | "
              f"MaxDD: {perf.max_drawdown_pct:.2f}% | "
              f"Sharpe: {perf.sharpe:.2f} | "
              f"Snaps: {perf.n_snapshots}")

    # Open positions
    if state.positions:
        print(f"\n  {'Symbol':15s} {'Weight':>7s} {'Funding':>9s} {'Cost':>8s} "
              f"{'Net PnL':>9s} {'Settl':>6s} {'Since':>12s}")
        print(f"  {'-'*70}")
        for sym, pos in sorted(state.positions.items(),
                                key=lambda x: x[1].accumulated_funding, reverse=True):
            current_f = 0.0
            for snap in snapshots:
                if snap.symbol == sym:
                    current_f = snap.ann_funding_pct
                    break
            entry_dt = pos.entry_time[:10] if pos.entry_time else "?"
            print(
                f"  {sym:15s} "
                f"{pos.notional_weight:6.1%} "
                f"{current_f:+8.1f}% "
                f"{pos.accumulated_cost:8.4f} "
                f"{pos.net_pnl:+8.4f} "
                f"{pos.n_settlements:6d} "
                f"{entry_dt:>12s}"
            )
    else:
        print("\n  No open positions — waiting for funding opportunities...")

    # Top opportunities (not in position)
    held_syms = set(state.positions.keys())
    opportunities = [s for s in snapshots
                     if s.symbol not in held_syms and s.ann_funding_pct > 10][:10]
    if opportunities:
        print(f"\n  Top Opportunities (not in position):")
        print(f"  {'Symbol':15s} {'AnnFund':>8s} {'Vol24h':>10s} {'MarkPx':>12s} {'ToSettle':>10s}")
        print(f"  {'-'*60}")
        for snap in opportunities:
            vol_m = snap.volume_24h_usd / 1e6
            print(
                f"  {snap.symbol:15s} "
                f"{snap.ann_funding_pct:+7.1f}% "
                f"{vol_m:8.1f}M "
                f"{snap.mark_price:12.4f} "
                f"{snap.time_to_funding_min:8.0f}min"
            )

    # Recent activity
    if messages:
        print(f"\n  Recent Activity:")
        for msg in messages[-8:]:
            print(f"    {msg}")

    # Footer
    print(f"\n  Config: entry >{config.entry_threshold_pct}% | "
          f"exit <{config.exit_threshold_pct}% | "
          f"cost {config.cost_per_leg_bps}bps/leg | "
          f"vol >${config.min_volume_usd/1e6:.0f}M")
    print(f"  State: {state_path}")
    print("  Press Ctrl+C to stop")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_running = True


def _handle_sigint(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="Funding Harvester Paper Trader")
    parser.add_argument("--scan-interval", type=int, default=300,
                        help="Seconds between scans (default: 300)")
    parser.add_argument("--entry", type=float, default=20.0,
                        help="Entry threshold: ann funding %% (default: 20)")
    parser.add_argument("--exit", type=float, default=3.0,
                        help="Exit threshold: ann funding %% (default: 3)")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="Max concurrent positions (default: 10)")
    parser.add_argument("--cost-bps", type=float, default=12.0,
                        help="Cost per leg in bps (default: 12)")
    parser.add_argument("--state-file", type=str,
                        default="data/funding_paper_state.json",
                        help="Path to state JSON file")
    parser.add_argument("--once", action="store_true",
                        help="Run one scan and exit (no loop)")
    parser.add_argument("--min-volume", type=float, default=50.0,
                        help="Min 24h volume in $M (default: 50)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing state and start fresh")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    state_path = Path(args.state_file)
    if args.reset and state_path.exists():
        state_path.unlink()
        print(f"Cleared state: {state_path}")

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    Path("data").mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("data/funding_paper.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    config = StrategyConfig(
        entry_threshold_pct=args.entry,
        exit_threshold_pct=args.exit,
        max_positions=args.max_positions,
        cost_per_leg_bps=args.cost_bps,
        min_volume_usd=args.min_volume * 1e6,
    )

    # Load persisted state
    state = PortfolioState.load(state_path)
    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    signal.signal(signal.SIGINT, _handle_sigint)

    all_messages: list[str] = []
    scan_count = 0

    while _running:
        try:
            scan_count += 1

            # 1. Scan current funding rates
            snapshots = scan_funding()

            # 2. Check for funding settlement
            if state.check_settlement():
                settlement_msgs = record_funding_settlement(state, snapshots)
                all_messages.extend(settlement_msgs)
                now_str = datetime.now(timezone.utc).strftime("%H:%M")
                all_messages.append(f"--- Funding settlement at {now_str} ---")

            # 3. Generate and execute signals
            signals = generate_signals(state, snapshots, config)
            trade_msgs = execute_signals(state, signals, config)
            all_messages.extend(trade_msgs)

            # 4. Save state
            state.save(state_path)

            # 5. Render dashboard
            render_dashboard(state, snapshots, config, all_messages,
                             scan_count, state_path)

            if args.once:
                break

            # 6. Wait for next scan
            time.sleep(args.scan_interval)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("Scan error: %s", e)
            all_messages.append(f"ERROR: {e}")
            time.sleep(30)

    # Final save
    state.save(state_path)
    print(f"\nFinal equity: {state.equity:.6f} ({state.equity - 1:+.4%})")
    print(f"Positions: {len(state.positions)}")
    print(f"State saved to {state_path}")


if __name__ == "__main__":
    main()
