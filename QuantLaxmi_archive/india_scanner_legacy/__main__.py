"""India Institutional Footprint Scanner — CLI.

Subcommands:
    scan      — Run daily scan for a date (default: today)
    backtest  — Historical backtest over a date range
    paper     — Paper trading cycle (scan + enter/exit positions)
    status    — Show current state and performance

Usage:
    python -m apps.india_scanner scan --date 2026-02-03
    python -m apps.india_scanner backtest --start 2025-06-01 --end 2026-01-31
    python -m apps.india_scanner paper --once
    python -m apps.india_scanner status
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from apps.india_scanner.backtest import format_backtest_results, run_backtest
from apps.india_scanner.costs import DEFAULT_COSTS
from apps.india_scanner.data import is_trading_day
from apps.india_scanner.scanner import format_scan_results, run_daily_scan
from apps.india_scanner.state import ScannerState, PaperPosition
from apps.india_scanner.status import format_status
from qlx.data.store import MarketDataStore

logger = logging.getLogger(__name__)

_running = True


def _handle_sigint(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> None:
    """Run a daily scan and print results."""
    target = date.fromisoformat(args.date)
    store = MarketDataStore()

    signals = run_daily_scan(target, store=store, top_n=args.top_n)
    print(format_scan_results(signals, target))


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a historical backtest."""
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    store = MarketDataStore()

    print(f"Running backtest: {start} to {end}, hold={args.hold}d, top-{args.top_n}")
    print("This may take a while for large date ranges...\n")

    result = run_backtest(
        start=start,
        end=end,
        hold_days=args.hold,
        top_n=args.top_n,
        store=store,
    )
    print(format_backtest_results(result))


def cmd_paper(args: argparse.Namespace) -> None:
    """Run a paper trading cycle."""
    state_path = Path(args.state_file)
    store = MarketDataStore()

    if args.reset and state_path.exists():
        state_path.unlink()
        print(f"Cleared state: {state_path}")

    state = ScannerState.load(state_path)
    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    signal.signal(signal.SIGINT, _handle_sigint)

    cost_model = DEFAULT_COSTS
    scan_count = 0

    while _running:
        try:
            scan_count += 1
            today = date.today()

            if not is_trading_day(today):
                print(f"{today} is not a trading day. Waiting...")
                if args.once:
                    break
                time.sleep(args.scan_interval)
                continue

            print(f"\n--- Scan #{scan_count} — {today} ---")

            # 1. Process exits for positions past hold period
            to_exit = [
                sym for sym, pos in state.positions.items()
                if pos.target_exit_date_reached
            ]

            for sym in to_exit:
                pos = state.positions.pop(sym)
                # Mark exit (we don't have real prices in paper mode)
                state.total_exits += 1
                state.closed_trades.append({
                    "symbol": sym,
                    "direction": pos.direction,
                    "entry_date": pos.entry_date,
                    "exit_date": today.isoformat(),
                    "entry_price": pos.entry_price,
                    "exit_price": pos.entry_price,  # simplified — no live price
                    "weight": pos.weight,
                    "net_pnl_pct": 0.0,
                    "hold_days": pos.days_held,
                })
                print(f"  EXIT: {sym} after {pos.days_held} days")

            # Increment days held for remaining positions
            for pos in state.positions.values():
                pos.days_held += 1

            # 2. Run daily scan
            signals = run_daily_scan(today, store=store, top_n=args.top_n)
            print(format_scan_results(signals, today))

            # 3. Enter new positions
            slots = args.top_n - len(state.positions)
            if slots > 0 and signals:
                weight = 1.0 / args.top_n
                for sig in signals[:slots]:
                    if sig.symbol in state.positions:
                        continue

                    state.positions[sig.symbol] = PaperPosition(
                        symbol=sig.symbol,
                        direction="long" if sig.composite_score > 0 else "short",
                        entry_date=today.isoformat(),
                        entry_price=0.0,  # simplified — set at next open
                        composite_score=sig.composite_score,
                        weight=weight,
                        hold_days=args.hold,
                    )
                    state.total_entries += 1
                    print(f"  ENTER: {sig.symbol} "
                          f"({'long' if sig.composite_score > 0 else 'short'}) "
                          f"score={sig.composite_score:+.2f}")

            # 4. Record state
            state.last_scan_date = today.isoformat()
            state.record_equity(today.isoformat())
            state.save(state_path)

            # 5. Show summary
            print(f"\n  Equity: Rs {state.equity:,.0f} | "
                  f"Positions: {len(state.positions)} | "
                  f"Entries: {state.total_entries} Exits: {state.total_exits}")

            if args.once:
                break

            print(f"\n  Sleeping {args.scan_interval}s until next scan...")
            time.sleep(args.scan_interval)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("Paper trading error: %s", e)
            if args.once:
                break
            time.sleep(60)

    state.save(state_path)
    print(f"\nState saved to {state_path}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current status."""
    state_path = Path(args.state_file)
    store = MarketDataStore()

    state = ScannerState.load(state_path)
    print(format_status(state, store))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="india_scanner",
        description="India Institutional Footprint Scanner",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--state-file", default="data/india_scanner_state.json",
                        help="Paper trading state file")

    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Run daily scan")
    p_scan.add_argument("--date", default=date.today().isoformat(),
                        help="Target date (YYYY-MM-DD)")
    p_scan.add_argument("--top-n", type=int, default=10, help="Top N signals")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run historical backtest")
    p_bt.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_bt.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_bt.add_argument("--hold", type=int, default=3, help="Hold period in days")
    p_bt.add_argument("--top-n", type=int, default=5, help="Top N signals per day")

    # paper
    p_paper = sub.add_parser("paper", help="Paper trading")
    p_paper.add_argument("--once", action="store_true", help="Single cycle, then exit")
    p_paper.add_argument("--scan-interval", type=int, default=3600,
                         help="Seconds between scans (default: 3600)")
    p_paper.add_argument("--hold", type=int, default=3, help="Hold period in days")
    p_paper.add_argument("--top-n", type=int, default=5, help="Max positions")
    p_paper.add_argument("--reset", action="store_true", help="Clear state and restart")

    # status
    sub.add_parser("status", help="Show current status")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    Path("data").mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("data/india_scanner.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    cmd_map = {
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "status": cmd_status,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
