"""CLRS -- Crypto Liquidity Regime Strategy Paper Trader.

Usage:
    python3 -m quantlaxmi.strategies.s26_crypto_flow
    python3 -m quantlaxmi.strategies.s26_crypto_flow --with-ticks
    python3 -m quantlaxmi.strategies.s26_crypto_flow --scan-interval 300
    python3 -m quantlaxmi.strategies.s26_crypto_flow --once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from quantlaxmi.data._paths import CRYPTO_TICK_DATA

logger = logging.getLogger(__name__)

from quantlaxmi.strategies.s26_crypto_flow.signals import SignalConfig
from quantlaxmi.strategies.s26_crypto_flow.state import PortfolioState, compute_performance
from quantlaxmi.strategies.s26_crypto_flow.strategy import (
    check_ttl_exits,
    execute_signals,
    generate_all_signals,
    record_funding_settlement,
    scan_and_compute,
)

# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def clear_screen():
    print("\033[2J\033[H", end="")


def render_dashboard(
    state: PortfolioState,
    config: SignalConfig,
    messages: list[str],
    scan_count: int,
    state_path: Path,
    tick_info: dict | None = None,
):
    """Render live terminal dashboard."""
    clear_screen()
    now = datetime.now(timezone.utc)

    # Header
    print("=" * 80)
    tick_label = " + TICK" if tick_info else ""
    print(f"  CLRS -- Crypto Liquidity Regime Strategy (Paper{tick_label})")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  |  Scan #{scan_count}")
    print("=" * 80)

    # Portfolio summary
    pnl = state.equity - 1.0
    sign = "+" if pnl >= 0 else ""
    wr = (state.total_wins / state.total_exits * 100) if state.total_exits > 0 else 0
    print(f"\n  Equity: {state.equity:.6f}  ({sign}{pnl:.4%})")
    print(f"  Positions: {state.n_positions} "
          f"(carry={len(state.carry_positions)}, "
          f"residual={len(state.residual_positions)}, "
          f"cascade={len(state.cascade_positions)}, "
          f"revert={len(state.reversion_positions)})")
    print(f"  Entries: {state.total_entries}  Exits: {state.total_exits}  "
          f"Win rate: {wr:.0f}%")
    print(f"  Funding earned: {state.total_funding_earned:+.6f}  "
          f"Costs paid: {state.total_costs_paid:.6f}")

    # Performance
    perf = compute_performance(state.equity_history)
    if perf is not None:
        print(f"  {perf.days_running:.1f}d | Ann: {perf.ann_return_pct:+.1f}% | "
              f"MaxDD: {perf.max_drawdown_pct:.2f}% | Sharpe: {perf.sharpe:.2f}")

    # Tick collector stats
    if tick_info:
        print(f"\n  [Tick Data] {tick_info.get('trades', 0):,} trades, "
              f"{tick_info.get('books', 0):,} books, "
              f"{tick_info.get('rate', 0):.1f} t/s, "
              f"{tick_info.get('symbols', 0)} symbols")

    # Positions by signal type
    for label, pool in [
        ("A: Enhanced Carry", state.carry_positions),
        ("B: Residual Carry", state.residual_positions),
        ("C: Cascade", state.cascade_positions),
        ("D: Reversion", state.reversion_positions),
    ]:
        if pool:
            print(f"\n  [{label}]")
            print(f"  {'Symbol':12s} {'Dir':5s} {'Weight':>7s} {'PnL':>9s} "
                  f"{'Cost':>8s} {'Bars':>5s} {'Str':>5s}")
            print(f"  {'-' * 55}")
            for sym, pos in sorted(pool.items()):
                print(f"  {sym:12s} {pos.direction:5s} {pos.notional_weight:6.1%} "
                      f"{pos.net_pnl:+8.4f} {pos.accumulated_cost:8.4f} "
                      f"{pos.hold_bars:5d} {pos.strength:5.2f}")

    # Recent activity
    if messages:
        print(f"\n  Recent Activity:")
        for msg in messages[-10:]:
            print(f"    {msg}")

    # Footer
    print(f"\n  Config: VPIN<{config.carry_vpin_max} | "
          f"Fund>{config.carry_min_ann_funding}% | "
          f"Hawkes>{config.cascade_hawkes_entry} | "
          f"Cost {config.cost_per_leg_bps}bps")
    print(f"  State: {state_path}")
    print("  Press Ctrl+C to stop")


# ---------------------------------------------------------------------------
# Synchronous main loop (no ticks)
# ---------------------------------------------------------------------------

_running = True


def _handle_sigint(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")


def run_sync(args, config, state, state_path):
    """Run the paper trader without tick collection (sync loop)."""
    signal.signal(signal.SIGINT, _handle_sigint)

    all_messages: list[str] = []
    scan_count = 0

    while _running:
        try:
            scan_count += 1
            logger.info("--- Scan #%d ---", scan_count)

            snapshots, ctx = scan_and_compute(state, config)
            regimes = ctx["regimes"]
            snap_map = ctx["snap_map"]
            funding_matrix = ctx["funding_matrix"]

            if state.check_settlement():
                settle_msgs = record_funding_settlement(state, snap_map)
                all_messages.extend(settle_msgs)
                ts = datetime.now(timezone.utc).strftime("%H:%M")
                all_messages.append(f"--- Funding settlement at {ts} ---")

            signals = generate_all_signals(
                state, regimes, snap_map, funding_matrix, config
            )
            ttl_exits = check_ttl_exits(state, config)
            signals.extend(ttl_exits)

            trade_msgs = execute_signals(state, signals, snap_map, config)
            all_messages.extend(trade_msgs)

            state.save(state_path)
            render_dashboard(state, config, all_messages, scan_count, state_path)

            logger.info(
                "Scan #%d: %d signals, %d positions, equity=%.6f",
                scan_count, len(signals), state.n_positions, state.equity,
            )

            if args.once:
                break
            time.sleep(args.scan_interval)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("Scan error: %s", e)
            all_messages.append(f"ERROR: {e}")
            time.sleep(30)

    state.save(state_path)


# ---------------------------------------------------------------------------
# Async main loop (with tick collection)
# ---------------------------------------------------------------------------

async def run_with_ticks(args, config, state, state_path):
    """Run the paper trader WITH background tick collection."""
    from quantlaxmi.data.collectors.crypto_tick.collector import CollectorConfig, TickCollector
    from quantlaxmi.data.collectors.crypto_tick.features import LiveRegimeTracker
    from quantlaxmi.data.collectors.crypto_tick.storage import TickStore, TickStoreConfig

    # Set up tick infrastructure
    tick_store = TickStore(TickStoreConfig(base_dir=CRYPTO_TICK_DATA))
    tracker = LiveRegimeTracker(
        vpin_bucket_size=args.vpin_bucket_size,
        vpin_n_buckets=args.vpin_buckets,
    )
    tick_config = CollectorConfig(
        min_ann_funding_pct=15.0,
        min_volume_usd=config.min_volume_usd,
        max_symbols=20,
        scan_interval_sec=3600,
    )
    collector = TickCollector(
        store=tick_store,
        config=tick_config,
        on_trade=tracker.on_trade,
        on_book=tracker.on_book,
    )

    # Start collector in background
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _sigint():
        stop_event.set()
        collector.stop()
        print("\n\nShutting down...")

    loop.add_signal_handler(signal.SIGINT, _sigint)
    loop.add_signal_handler(signal.SIGTERM, _sigint)

    collect_task = asyncio.create_task(collector.run())

    all_messages: list[str] = []
    scan_count = 0

    try:
        while not stop_event.is_set():
            scan_count += 1
            logger.info("--- Scan #%d ---", scan_count)

            try:
                # Get tick-level regimes (if collector has warmed up)
                tick_regimes = tracker.all_regimes() or None

                # Phase 1: Scan + compute (with tick regimes)
                snapshots, ctx = await loop.run_in_executor(
                    None, lambda: scan_and_compute(state, config, tick_regimes)
                )
                regimes = ctx["regimes"]
                snap_map = ctx["snap_map"]
                funding_matrix = ctx["funding_matrix"]

                if state.check_settlement():
                    settle_msgs = record_funding_settlement(state, snap_map)
                    all_messages.extend(settle_msgs)
                    ts = datetime.now(timezone.utc).strftime("%H:%M")
                    all_messages.append(f"--- Funding settlement at {ts} ---")

                # Phase 2: Generate signals
                signals = generate_all_signals(
                    state, regimes, snap_map, funding_matrix, config
                )
                ttl_exits = check_ttl_exits(state, config)
                signals.extend(ttl_exits)

                # Phase 3: Execute
                trade_msgs = execute_signals(state, signals, snap_map, config)
                all_messages.extend(trade_msgs)

                state.save(state_path)

                # Render with tick info
                cstats = collector.stats()
                tick_info = {
                    "trades": cstats["total_trades"],
                    "books": cstats["total_books"],
                    "rate": cstats["trades_per_sec"],
                    "symbols": cstats["n_symbols"],
                }
                render_dashboard(
                    state, config, all_messages, scan_count,
                    state_path, tick_info=tick_info,
                )

                logger.info(
                    "Scan #%d: %d signals, %d positions, equity=%.6f, "
                    "tick trades=%d",
                    scan_count, len(signals), state.n_positions,
                    state.equity, cstats["total_trades"],
                )

                if args.once:
                    break

            except Exception as e:
                logging.exception("Scan error: %s", e)
                all_messages.append(f"ERROR: {e}")

            # Wait for next scan (or stop)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.scan_interval)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # normal: scan interval elapsed

    finally:
        collector.stop()
        try:
            await asyncio.wait_for(collect_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            collect_task.cancel()
        await collector.close()

    state.save(state_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLRS Paper Trader")
    parser.add_argument("--scan-interval", type=int, default=300,
                        help="Seconds between scans (default: 300)")
    parser.add_argument("--state-file", type=str,
                        default="data/crypto_flow_state.json",
                        help="Path to state JSON file")
    parser.add_argument("--once", action="store_true",
                        help="Run one scan and exit")
    parser.add_argument("--reset", action="store_true",
                        help="Clear state and start fresh")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug logging")

    # Tick collection
    parser.add_argument("--with-ticks", action="store_true",
                        help="Run with background tick collector for real-time "
                             "VPIN/OFI/Hawkes (stores to data/ticks/)")
    parser.add_argument("--vpin-bucket-size", type=float, default=50_000.0,
                        help="VPIN bucket size in USD (default: 50000)")
    parser.add_argument("--vpin-buckets", type=int, default=50,
                        help="VPIN rolling window buckets (default: 50)")

    # Carry params (the only proven signal)
    parser.add_argument("--carry-entry", type=float, default=20.0,
                        help="Min ann funding %% for carry entry (default: 20)")
    parser.add_argument("--carry-exit", type=float, default=3.0,
                        help="Exit when ann funding < this (default: 3)")

    # Portfolio params
    parser.add_argument("--max-carry", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=8.0)
    parser.add_argument("--min-volume", type=float, default=10.0,
                        help="Min 24h volume in $M (default: 10, NOT 50)")

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
            logging.FileHandler("data/crypto_flow.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    config = SignalConfig(
        carry_min_ann_funding=args.carry_entry,
        carry_exit_funding=args.carry_exit,
        max_carry_positions=args.max_carry,
        cost_per_leg_bps=args.cost_bps,
        min_volume_usd=args.min_volume * 1e6,
    )

    state = PortfolioState.load(state_path)
    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    if args.with_ticks:
        asyncio.run(run_with_ticks(args, config, state, state_path))
    else:
        run_sync(args, config, state, state_path)

    print(f"\nFinal equity: {state.equity:.6f} ({state.equity - 1:+.4%})")
    print(f"Positions: {state.n_positions}")
    print(f"State saved to {state_path}")


if __name__ == "__main__":
    main()
