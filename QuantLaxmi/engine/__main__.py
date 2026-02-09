"""QuantLaxmi Engine CLI entry point.

Usage:
    python -m engine paper --once              # single day scan
    python -m engine paper --date 2026-02-05
    python -m engine paper --range 2025-01-01 2026-01-31
    python -m engine replay --date 2025-08-06  # replay single date
    python -m engine replay --range 2025-08-06 2025-12-31
    python -m engine replay --range START END --times 3  # 3x parity
    python -m engine status                    # portfolio status
    python -m engine live                      # (future: live trading)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger("engine")


def cmd_paper(args: argparse.Namespace) -> None:
    """Run paper trading mode."""
    from data.store import MarketDataStore
    from strategies.registry import StrategyRegistry
    from core.allocator.meta import MetaAllocator
    from core.risk.manager import RiskManager
    from engine.orchestrator import Orchestrator

    store = MarketDataStore()
    registry = StrategyRegistry()
    registry.discover()

    if len(registry) == 0:
        print("No strategies discovered! Check strategy module paths.")
        return

    print(f"Discovered {len(registry)} strategies: {registry.list_ids()}")

    orchestrator = Orchestrator(
        store=store,
        registry=registry,
        state_file=Path(args.state_file),
    )

    if args.range:
        # Date range mode
        start = date.fromisoformat(args.range[0])
        end = date.fromisoformat(args.range[1])
        d = start
        while d <= end:
            try:
                from strategies.s9_momentum.data import is_trading_day
                if is_trading_day(d):
                    orchestrator.run_day(d)
            except ImportError:
                orchestrator.run_day(d)
            d += timedelta(days=1)
    elif args.date:
        # Single date mode
        d = date.fromisoformat(args.date)
        summary = orchestrator.run_day(d)
        _print_summary(summary)
    else:
        # Default: today (--once)
        d = date.today()
        summary = orchestrator.run_day(d)
        _print_summary(summary)

    store.close()


def cmd_replay(args: argparse.Namespace) -> None:
    """Run replay engine for parity verification."""
    from data.store import MarketDataStore
    from strategies.registry import StrategyRegistry
    from engine.replay.engine import ReplayEngine

    store = MarketDataStore()
    registry = StrategyRegistry()
    registry.discover()

    if len(registry) == 0:
        print("No strategies discovered! Check strategy module paths.")
        return

    print(f"Discovered {len(registry)} strategies: {registry.list_ids()}")

    engine = ReplayEngine(
        store=store,
        registry=registry,
        ref_events_dir=Path(args.events_dir),
        verify_hashes=args.verify_hashes,
    )

    if args.range:
        start, end = args.range
    elif args.date:
        start = end = args.date
    else:
        print("Error: --date or --range required for replay")
        return

    if args.times > 1:
        # N-way parity check
        print(f"\nRunning {args.times}x replay parity check: {start} to {end}")
        results = engine.replay_n_times(start, end, n=args.times)
        all_identical = all(r.comparison.identical for r in results[1:])
        print(f"\n{'=' * 60}")
        print(f"Replay Parity ({args.times}x): {'PASS' if all_identical else 'FAIL'}")
        for i, r in enumerate(results):
            status = "baseline" if i == 0 else ("MATCH" if r.comparison.identical else "DIFF")
            print(f"  Run {i+1}: {r.replay_event_count} events [{status}]")
        if not all_identical:
            for i, r in enumerate(results[1:], 2):
                if not r.comparison.identical:
                    print(f"\nRun {i} diffs:")
                    print(r.comparison.summary())
    else:
        # Single replay against reference
        print(f"\nReplaying: {start} to {end}")
        result = engine.replay_range(start, end)
        print(f"\n{'=' * 60}")
        print(result.comparison.summary())

        # Save artifacts
        output_dir = Path(args.output_dir)
        engine.save_artifacts(result, output_dir)
        print(f"\nArtifacts saved to: {output_dir}")

    store.close()


def cmd_status(args: argparse.Namespace) -> None:
    """Show current portfolio status."""
    from engine.state import PortfolioState

    state = PortfolioState.load(Path(args.state_file))

    print("QuantLaxmi Portfolio Status")
    print("=" * 60)
    print(f"  Equity:         {state.equity:.4f} ({state.total_return_pct():+.2f}%)")
    print(f"  Peak equity:    {state.peak_equity:.4f}")
    print(f"  Drawdown:       {state.portfolio_dd:.2%}")
    print(f"  Total exposure: {state.total_exposure:.2%}")
    print(f"  Win rate:       {state.win_rate():.1%}")
    print(f"  Trades:         {len(state.closed_trades)}")
    print(f"  Last scan:      {state.last_scan_date}")
    print(f"  Regime:         {state.last_regime} (VIX={state.last_vix:.1f})")
    print()

    positions = state.active_positions()
    if positions:
        print(f"Active Positions ({len(positions)}):")
        print(f"  {'Strategy':<12} {'Symbol':<12} {'Dir':<6} {'Weight':>8} {'Entry':>10} {'Date'}")
        print("  " + "-" * 65)
        for p in positions:
            print(
                f"  {p.strategy_id:<12} {p.symbol:<12} {p.direction:<6} "
                f"{p.weight:>7.4f} {p.entry_price:>10.2f} {p.entry_date}"
            )
    else:
        print("No active positions.")

    print()

    # Strategy-level equity
    if state.strategy_equity:
        print("Strategy Equity:")
        for sid, eq in sorted(state.strategy_equity.items()):
            dd = state.strategy_dd(sid)
            print(f"  {sid:<16} equity={eq:.4f}  dd={dd:.2%}")

    # Recent trades
    if state.closed_trades:
        recent = state.closed_trades[-10:]
        print(f"\nRecent Trades (last {len(recent)}):")
        print(f"  {'Strategy':<12} {'Symbol':<10} {'Dir':<6} {'PnL':>8} {'Reason'}")
        print("  " + "-" * 55)
        for t in recent:
            print(
                f"  {t.strategy_id:<12} {t.symbol:<10} {t.direction:<6} "
                f"{t.pnl_pct * 100:>+7.2f}% {t.exit_reason}"
            )


def _print_summary(summary: dict) -> None:
    """Pretty-print orchestrator summary."""
    regime = summary.get("regime", {})
    print(f"\nDate: {summary['date']}  Regime: {regime.get('type', '?')} (VIX={regime.get('vix', 0):.1f})")

    signals = summary.get("signals", [])
    if signals:
        print(f"\nSignals ({len(signals)}):")
        for s in signals:
            print(f"  {s['strategy']:<12} {s['symbol']:<12} {s['direction']:<6} conv={s['conviction']:.2f}")

    actions = summary.get("actions", [])
    if actions:
        print(f"\nActions ({len(actions)}):")
        for a in actions:
            if a["action"] == "open":
                print(
                    f"  OPEN  {a['strategy']:<12} {a['symbol']:<12} "
                    f"{a['direction']:<6} w={a['weight']:.4f} @ {a.get('entry_price', 0):.2f}"
                )
            else:
                print(
                    f"  CLOSE {a['strategy']:<12} {a['symbol']:<12} "
                    f"pnl={a.get('pnl_pct', 0) * 100:+.2f}% reason={a.get('reason', '')}"
                )
    else:
        print("\nNo actions taken.")

    blocked = [r for r in summary.get("risk_checks", []) if not r["approved"]]
    if blocked:
        print(f"\nBlocked ({len(blocked)}):")
        for r in blocked:
            print(f"  {r['strategy']:<12} {r['symbol']:<12} gate={r['gate']} reason={r['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="engine",
        description="QuantLaxmi Engine — India FnO Trading System",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--state-file", default="data/state/portfolio.json",
        help="Path to state JSON file",
    )

    sub = parser.add_subparsers(dest="command")

    # paper
    p_paper = sub.add_parser("paper", help="Paper trading mode")
    p_paper.add_argument("--once", action="store_true", help="Run once for today")
    p_paper.add_argument("--date", help="Scan a specific date (YYYY-MM-DD)")
    p_paper.add_argument("--range", nargs=2, help="Scan date range: START END")

    # status
    sub.add_parser("status", help="Show portfolio status")

    # replay
    p_replay = sub.add_parser("replay", help="Replay events for parity verification")
    p_replay.add_argument("--date", help="Replay a single date (YYYY-MM-DD)")
    p_replay.add_argument("--range", nargs=2, help="Replay date range: START END")
    p_replay.add_argument(
        "--times", type=int, default=1,
        help="Number of replay runs for parity check (default: 1)",
    )
    p_replay.add_argument(
        "--events-dir", default="data/events",
        help="Directory containing reference event logs",
    )
    p_replay.add_argument(
        "--output-dir", default="data/replay_artifacts",
        help="Directory for replay output artifacts",
    )
    p_replay.add_argument(
        "--verify-hashes", action="store_true",
        help="Verify hash chain on reference WAL",
    )

    # live (placeholder)
    sub.add_parser("live", help="Live trading mode (requires QUANTLAXMI_MODE=live)")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "paper":
        cmd_paper(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "live":
        raise NotImplementedError(
            "Live trading mode is not yet implemented.\n"
            "Requires: Kite WebSocket feed, OMS integration, and risk guard rails.\n"
            "Use 'engine paper' for paper trading."
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
