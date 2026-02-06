"""India F&O microstructure paper trader.

Usage:
    python -m apps.india_microstructure collect              # collect chain snapshots only
    python -m apps.india_microstructure paper                # collect + paper trade
    python -m apps.india_microstructure paper --reset        # clear state + start fresh
    python -m apps.india_microstructure status               # show current state
    python -m apps.india_microstructure analyze              # run analytics on latest snapshot

Collects NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY option chain snapshots
every 3 minutes and trades NIFTY/BANKNIFTY futures based on:
  - GEX regime (dealer gamma exposure)
  - OI delta flow (institutional positioning)
  - IV term structure (panic/complacency)
  - Futures basis (leverage positioning)
  - Put-call ratio (sentiment extreme)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from apps.india_microstructure.analytics import (
    MicrostructureSnapshot,
    analyze_snapshot,
)
from apps.india_microstructure.auth import headless_login
from apps.india_microstructure.collector import (
    SNAPSHOT_DIR,
    load_instrument_map,
    run_collector,
    save_snapshot,
    snapshot_chain,
)
from apps.india_microstructure.paper_state import (
    AnalyticsLog,
    MicroPaperState,
    DEFAULT_STATE_FILE,
)
from apps.india_microstructure.signals import (
    EXIT_THRESHOLD,
    FLAT_DECAY_SCANS,
    TradeSignal,
    generate_signal,
    reset_signal_state,
    _flat_streak,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

ALL_SYMBOLS = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]
# Only trade the liquid indices
TRADEABLE = ["NIFTY", "BANKNIFTY"]

# Max hold time for a position (minutes)
MAX_HOLD_MINUTES = 120  # 2 hours
# Minimum hold time — don't exit before this regardless of signal
MIN_HOLD_MINUTES = 15


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(
    state: MicroPaperState,
    snapshots: dict[str, MicrostructureSnapshot],
    signals: dict[str, TradeSignal],
    scan_num: int,
) -> None:
    """Print terminal dashboard."""
    print("\033[2J\033[H", end="")  # clear screen
    now = datetime.now(IST)
    print("=" * 80)
    print("  INDIA F&O MICROSTRUCTURE PAPER TRADER")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S IST')}  |  Scan #{scan_num}")
    print("=" * 80)

    # Performance
    n_closed = len(state.closed_trades)
    print(
        f"\n  Entries: {state.total_entries}  |  "
        f"Active: {len(state.positions)}  |  "
        f"Closed: {n_closed}  |  "
        f"Win rate: {state.win_rate():.0f}%"
    )
    if n_closed > 0:
        print(
            f"  Avg P&L: {state.avg_pnl():+.3f}%  |  "
            f"Total P&L: {state.total_pnl():+.3f}%"
        )

    # Active positions
    if state.positions:
        print(f"\n{'─' * 80}")
        print(f"  {'Symbol':<14} {'Dir':>5} {'Entry':>10} {'Now':>10} "
              f"{'P&L%':>8} {'Age':>6} {'Conv':>5} {'Regime':<12}")
        print(f"{'─' * 80}")

        for sym, pos in state.positions.items():
            sig = signals.get(sym)
            current_price = sig.futures if sig else pos.entry_price
            pnl = pos.unrealized_pnl(current_price)
            age = pos.age_minutes()
            pnl_color = "\033[32m" if pnl >= 0 else "\033[31m"
            reset = "\033[0m"
            print(
                f"  {sym:<14} {pos.direction:>5} "
                f"{pos.entry_price:>10.1f} {current_price:>10.1f} "
                f"{pnl_color}{pnl:>+7.3f}%{reset} "
                f"{age:>5.0f}m {pos.conviction:>5.2f} {pos.gex_regime:<12}"
            )

    # Analytics dashboard per symbol
    print(f"\n{'─' * 80}")
    print(f"  {'Symbol':<12} {'Spot':>10} {'Fut':>10} {'Basis':>7} "
          f"{'GEX':>10} {'Regime':<10} {'PCR':>6} {'IVnear':>6} {'IVfar':>6} "
          f"{'Signal':>7}")
    print(f"{'─' * 80}")

    for sym in ALL_SYMBOLS:
        snap = snapshots.get(sym)
        sig = signals.get(sym)
        if not snap:
            continue

        sig_str = f"{sig.direction}" if sig else "n/a"
        if sig and sig.direction != "flat":
            sig_str += f"({sig.conviction:.1f})"

        print(
            f"  {sym:<12} {snap.spot:>10.1f} {snap.futures:>10.1f} "
            f"{snap.basis.basis_points:>+6.0f}p "
            f"{snap.gex.net_gex_cr:>+10.0f} {snap.gex.regime:<10} "
            f"{snap.pcr.pcr_oi:>5.2f} "
            f"{snap.iv_term.near_iv*100:>5.1f}% "
            f"{snap.iv_term.far_iv*100:>5.1f}% "
            f"{sig_str:>7}"
        )

    # Signal details for tradeable
    if signals:
        print(f"\n{'─' * 80}")
        print("  Signal Breakdown:")
        for sym in TRADEABLE:
            sig = signals.get(sym)
            if sig:
                flat_n = _flat_streak.get(sym, 0)
                print(f"    {sym}: {sig.reasoning} flat_streak={flat_n}")

    # Recent closed trades
    if state.closed_trades:
        recent = state.closed_trades[-5:]
        print(f"\n{'─' * 80}")
        print("  Recent Exits:")
        for ct in reversed(recent):
            pnl_color = "\033[32m" if ct.pnl_pct >= 0 else "\033[31m"
            reset = "\033[0m"
            print(
                f"    {ct.symbol:<12} {ct.direction:>5} "
                f"{pnl_color}{ct.pnl_pct:>+7.3f}%{reset}  "
                f"{ct.exit_reason}  [{ct.gex_regime}]"
            )

    print(f"\n{'─' * 80}")
    print(f"  Config: entry≥0.40  EMA_α=0.3  min_hold={MIN_HOLD_MINUTES}m  "
          f"max_hold={MAX_HOLD_MINUTES}m  decay={FLAT_DECAY_SCANS}scans  cost=5bps")
    print(f"  Trade: {','.join(TRADEABLE)}  |  Consecutive≥2 raw signals required")
    print("=" * 80, flush=True)


# ---------------------------------------------------------------------------
# Paper trading loop
# ---------------------------------------------------------------------------

def run_paper(
    interval_seconds: int = 180,
    state_file: Path = DEFAULT_STATE_FILE,
    reset: bool = False,
) -> None:
    """Run the microstructure paper trader."""
    if reset and state_file.exists():
        state_file.unlink()
        logger.info("State cleared")
    if reset:
        reset_signal_state()

    state = MicroPaperState.load(state_file)

    kite = headless_login()
    imap = load_instrument_map(kite, symbols=ALL_SYMBOLS)

    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False
        logger.info("Shutting down...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(
        "Paper trader started: collect=%s, trade=%s, interval=%ds",
        ALL_SYMBOLS, TRADEABLE, interval_seconds,
    )

    scan_num = 0
    prev_snapshots: dict[str, pd.DataFrame] = {}

    while running:
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now < market_open:
            wait = (market_open - now).total_seconds()
            logger.info("Waiting for market open (%.0f sec)", wait)
            time.sleep(min(wait, 60))
            continue

        if now > market_close:
            # Close all positions at market close
            signals: dict[str, TradeSignal] = {}
            for sym in list(state.positions.keys()):
                pos = state.positions[sym]
                exit_price = pos.entry_price
                exit_spot = pos.entry_spot
                closed = state.record_exit(sym, exit_price, exit_spot, "market_close")
                if closed:
                    logger.info(
                        "CLOSE %s %s: P&L=%+.3f%% (market close)",
                        closed.direction, sym, closed.pnl_pct,
                    )
            state.save(state_file)
            logger.info("Market closed, all positions exited")
            break

        # Reload instrument map if date changed
        today = str(now.date())
        if imap.loaded_date != today:
            imap = load_instrument_map(kite, symbols=ALL_SYMBOLS)

        scan_num += 1

        # 1. Snapshot all indices
        current_dfs: dict[str, pd.DataFrame] = {}
        for sym in ALL_SYMBOLS:
            try:
                df = snapshot_chain(kite, imap, sym)
                if df is not None:
                    save_snapshot(df, sym)
                    current_dfs[sym] = df
            except Exception as e:
                logger.error("Snapshot failed for %s: %s", sym, e)

        # 2. Run analytics
        snapshots: dict[str, MicrostructureSnapshot] = {}
        for sym, df in current_dfs.items():
            prev = prev_snapshots.get(sym)
            try:
                snap = analyze_snapshot(df, prev)
                snapshots[sym] = snap
            except Exception as e:
                logger.error("Analytics failed for %s: %s", sym, e)

        # 3. Generate signals
        signals: dict[str, TradeSignal] = {}
        for sym, snap in snapshots.items():
            sig = generate_signal(snap)
            signals[sym] = sig

            # Log analytics
            state.log_analytics(AnalyticsLog(
                timestamp=snap.timestamp,
                symbol=sym,
                spot=snap.spot,
                futures=snap.futures,
                net_gex=snap.gex.net_gex_cr,
                gex_regime=snap.gex.regime,
                gex_flip=snap.gex.gex_flip_strike,
                pcr_oi=snap.pcr.pcr_oi,
                basis_pts=snap.basis.basis_points,
                basis_zscore=snap.basis.basis_zscore,
                iv_near=snap.iv_term.near_iv,
                iv_far=snap.iv_term.far_iv,
                iv_slope=snap.iv_term.slope,
                oi_flow_score=snap.oi_flow.score if snap.oi_flow else 0,
                combined_score=sum(
                    sig.components[k] * w
                    for k, w in [("gex", 0.3), ("oi_flow", 0.25), ("iv_term", 0.2),
                                 ("basis", 0.15), ("pcr", 0.1)]
                ),
                signal_direction=sig.direction,
            ))

        # 4. Execute trades (only on tradeable symbols)
        for sym in TRADEABLE:
            sig = signals.get(sym)
            if sig is None:
                continue

            # Check exits first
            if sym in state.positions:
                pos = state.positions[sym]
                current_price = sig.futures
                age = pos.age_minutes()

                should_exit = False
                exit_reason = ""

                # a) Max hold always applies (no min hold check)
                if age > MAX_HOLD_MINUTES:
                    should_exit = True
                    exit_reason = "max_hold"
                # b) Signal flipped direction (smoothed EMA reversed)
                #    Only after minimum hold time
                elif age >= MIN_HOLD_MINUTES and sig.direction != "flat" and sig.direction != pos.direction:
                    should_exit = True
                    exit_reason = "signal_flip"
                # c) Signal decayed — flat for FLAT_DECAY_SCANS consecutive scans
                #    Only after minimum hold time
                elif age >= MIN_HOLD_MINUTES and _flat_streak.get(sym, 0) >= FLAT_DECAY_SCANS:
                    should_exit = True
                    exit_reason = "signal_decay"

                if should_exit:
                    closed = state.record_exit(
                        sym, current_price, sig.spot, exit_reason,
                    )
                    if closed:
                        logger.info(
                            "EXIT %s %s @ %.1f: P&L=%+.3f%% (%s)",
                            closed.direction.upper(), sym, current_price,
                            closed.pnl_pct, exit_reason,
                        )

            # Check entries (signal must be non-flat, which requires
            # smoothed score > threshold AND consecutive raw signals)
            if sym not in state.positions and sig.direction != "flat":
                pos = state.record_entry(
                    symbol=sym,
                    direction=sig.direction,
                    futures_price=sig.futures,
                    spot=sig.spot,
                    conviction=sig.conviction,
                    gex_regime=sig.gex_regime,
                    reasoning=sig.reasoning,
                )
                logger.info(
                    "ENTRY %s %s @ %.1f  conv=%.2f  [%s]",
                    sig.direction.upper(), sym, sig.futures,
                    sig.conviction, sig.reasoning,
                )

        # 5. Save state and render
        state.save(state_file)
        try:
            render_dashboard(state, snapshots, signals, scan_num)
        except Exception as e:
            logger.error("Dashboard render failed: %s", e)

        prev_snapshots = current_dfs

        # Wait for next cycle
        time.sleep(interval_seconds)

    state.save(state_file)
    logger.info("Paper trader stopped. Final state saved.")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status(state_file: Path = DEFAULT_STATE_FILE) -> None:
    state = MicroPaperState.load(state_file)
    print(f"\nPositions: {len(state.positions)}")
    for sym, pos in state.positions.items():
        print(f"  {sym}: {pos.direction} @ {pos.entry_price:.1f} "
              f"(age {pos.age_minutes():.0f}m, conv={pos.conviction:.2f})")
    print(f"\nClosed trades: {len(state.closed_trades)}")
    if state.closed_trades:
        print(f"  Win rate: {state.win_rate():.0f}%")
        print(f"  Total P&L: {state.total_pnl():+.3f}%")
        print(f"  Avg P&L: {state.avg_pnl():+.3f}%")
    print(f"\nAnalytics log: {len(state.analytics_log)} entries")
    if state.analytics_log:
        last = state.analytics_log[-1]
        print(f"  Last: {last.timestamp} {last.symbol} "
              f"spot={last.spot:.1f} gex={last.net_gex:.0f}({last.gex_regime}) "
              f"pcr={last.pcr_oi:.2f} signal={last.signal_direction}")


# ---------------------------------------------------------------------------
# Analyze latest snapshot
# ---------------------------------------------------------------------------

def analyze_latest() -> None:
    """Run analytics on the latest saved snapshots."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    day_dir = SNAPSHOT_DIR / today

    if not day_dir.exists():
        print(f"No snapshots for {today}")
        return

    for sym in ALL_SYMBOLS:
        files = sorted(day_dir.glob(f"{sym}_*.parquet"))
        if not files:
            print(f"  {sym}: no snapshots")
            continue

        df = pd.read_parquet(files[-1])
        prev = pd.read_parquet(files[-2]) if len(files) > 1 else None
        snap = analyze_snapshot(df, prev)
        sig = generate_signal(snap)

        print(f"\n{'='*60}")
        print(f"  {sym} @ {snap.spot:.1f} (fut {snap.futures:.1f})")
        print(f"  GEX: {snap.gex.net_gex_cr:+.0f} Cr | Regime: {snap.gex.regime} | Flip: {snap.gex.gex_flip_strike:.0f}")
        print(f"  Max Pain: {snap.max_pain.max_pain_strike:.0f} ({snap.max_pain.distance_pct:+.2f}%)")
        print(f"  Basis: {snap.basis.basis_points:+.0f} pts ({snap.basis.basis_pct:+.1f}% ann)")
        print(f"  PCR: OI={snap.pcr.pcr_oi:.3f} Vol={snap.pcr.pcr_volume:.3f} [{snap.pcr.signal}]")
        print(f"  IV: near={snap.iv_term.near_iv*100:.1f}% far={snap.iv_term.far_iv*100:.1f}% slope={snap.iv_term.slope:.2f} [{snap.iv_term.signal}]")
        if snap.oi_flow:
            print(f"  OI Flow: call_chg={snap.oi_flow.call_oi_change:+,} put_chg={snap.oi_flow.put_oi_change:+,} [{snap.oi_flow.direction}]")
        print(f"  Signal: {sig.direction} (conv={sig.conviction:.2f})")
        print(f"  Reasoning: {sig.reasoning}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="India F&O microstructure trader")
    sub = parser.add_subparsers(dest="cmd")

    # collect
    p_collect = sub.add_parser("collect", help="Collect chain snapshots only")
    p_collect.add_argument("--interval", type=int, default=180)

    # paper
    p_paper = sub.add_parser("paper", help="Collect + paper trade")
    p_paper.add_argument("--interval", type=int, default=180)
    p_paper.add_argument("--reset", action="store_true")
    p_paper.add_argument("--state-file", type=str, default=None)

    # status
    sub.add_parser("status", help="Show current state")

    # analyze
    sub.add_parser("analyze", help="Analyze latest snapshots")

    args = parser.parse_args()

    if args.cmd == "collect":
        kite = headless_login()
        run_collector(kite, interval_seconds=args.interval)

    elif args.cmd == "paper":
        state_file = Path(args.state_file) if args.state_file else DEFAULT_STATE_FILE
        run_paper(
            interval_seconds=args.interval,
            state_file=state_file,
            reset=args.reset,
        )

    elif args.cmd == "status":
        show_status()

    elif args.cmd == "analyze":
        analyze_latest()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
