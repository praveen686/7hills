"""RNDR (Risk-Neutral Density Regime) Paper Trader — CLI.

Trades: BANKNIFTY, MIDCPNIFTY, FINNIFTY (NIFTY excluded — too efficient)

Subcommands:
    scan      — Process a single day (calibrate SANOS, extract density, signal check)
    paper     — Paper trading loop (daily scan + position management)
    backfill  — Build density history from cached bhavcopy data
    status    — Show current state, positions, and performance

Usage:
    python -m apps.india_fno.rndr scan
    python -m apps.india_fno.rndr scan --date 2026-02-03
    python -m apps.india_fno.rndr paper --once
    python -m apps.india_fno.rndr backfill --start 2024-03-01 --end 2025-05-15 --simulate
    python -m apps.india_fno.rndr status
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from apps.india_fno.density_strategy import (
    DEFAULT_COST_BPS,
    DEFAULT_ENTRY_PCTILE,
    DEFAULT_EXIT_PCTILE,
    DEFAULT_HOLD_DAYS,
    DEFAULT_LOOKBACK,
    DEFAULT_PHYS_WINDOW,
    DensityDayObs,
    _calibrate_density,
    _rolling_percentile,
    build_density_series,
    compute_composite_signal,
)
from apps.india_fno.risk_neutral import (
    extract_density,
    kl_divergence,
    physical_skewness,
)
from apps.india_fno.rndr.state import (
    DEFAULT_STATE_FILE,
    RNDR_SYMBOLS,
    DensityObservation,
    DensityPaperState,
)
from apps.india_scanner.bhavcopy import BhavcopyCache, is_trading_day

logger = logging.getLogger(__name__)

_running = True


def _handle_sigint(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_single_day(
    cache: BhavcopyCache,
    d: date,
    symbol: str,
    state: DensityPaperState,
) -> DensityObservation | None:
    """Calibrate SANOS for one day/symbol, compute density features incrementally.

    Uses prev_densities from state for KL divergence, and trailing spots
    from state history for physical skewness.
    """
    snap, result, spot, atm_iv = _calibrate_density(cache, d, symbol)

    if snap is None or not snap.density_ok:
        return None

    # Physical skewness from trailing spots in state history
    history = state.density_histories.get(symbol, [])
    spots = [o.spot for o in history] + [spot]
    phys_skew = 0.0
    if len(spots) > state.phys_window:
        log_rets = np.diff(np.log(spots[-state.phys_window - 1:]))
        phys_skew = physical_skewness(log_rets)

    # KL divergence and entropy change vs yesterday
    K, q = extract_density(result, 0)
    dK = K[1] - K[0]
    kl = 0.0
    d_entropy = 0.0

    prev_q = state.prev_densities.get(symbol)
    if prev_q is not None and len(prev_q) == len(q):
        prev_q_arr = np.array(prev_q)
        kl = kl_divergence(q, prev_q_arr, dK)
        # Entropy change: need previous entropy
        if history:
            d_entropy = snap.entropy - history[-1].entropy

    skew_premium = phys_skew - snap.rn_skewness

    # Update prev_densities for next day
    state.prev_densities[symbol] = q.tolist()

    return DensityObservation(
        date=d.isoformat(),
        symbol=symbol,
        spot=float(spot),
        atm_iv=float(atm_iv),
        rn_skewness=float(snap.rn_skewness),
        rn_kurtosis=float(snap.rn_kurtosis),
        entropy=float(snap.entropy),
        left_tail=float(snap.left_tail),
        right_tail=float(snap.right_tail),
        phys_skewness=float(phys_skew),
        skew_premium=float(skew_premium),
        entropy_change=float(d_entropy),
        kl_div=float(kl),
        density_ok=bool(snap.density_ok),
    )


def _format_signal(signal_type: str) -> str:
    labels = {
        "enter": ">> ENTER LONG",
        "exit_hold": "<< EXIT (max hold)",
        "exit_signal": "<< EXIT (signal decay)",
        "hold": "-- HOLD",
        "wait": "   (no signal)",
        "warmup": "   (warming up)",
    }
    return labels.get(signal_type, signal_type)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _format_status(state: DensityPaperState) -> str:
    lines = [
        "",
        "RNDR (Risk-Neutral Density Regime) — Paper Trader",
        "=" * 70,
    ]

    # Config
    lines.append(
        f"  Config: lookback={state.lookback}d, "
        f"entry={state.entry_pctile:.0%}, "
        f"exit={state.exit_pctile:.0%}, "
        f"max_hold={state.max_hold_days}d, "
        f"cost={state.cost_bps:.0f}bps"
    )
    lines.append(f"  Indices: {', '.join(state.symbols)}")

    # Per-index summary
    lines.append("")
    lines.append(
        f"  {'Index':<12} {'Days':>5} {'Signal':>8} {'Pctile':>7} {'Position':<20}"
    )
    lines.append("  " + "-" * 60)

    for sym in state.symbols:
        history = state.density_histories.get(sym, [])
        n_obs = len(history)

        sig_str = "-"
        pctile_str = "-"
        if state.have_enough_history(sym):
            sig_type, composite, sig_pctile = state.check_signal(sym)
            sig_str = f"{composite:+.3f}"
            pctile_str = f"{sig_pctile:.2f}"

        pos = state.positions.get(sym)
        if pos is not None and n_obs > 0:
            current_spot = history[-1].spot
            unrealised = (current_spot - pos.entry_spot) / pos.entry_spot * 100
            pos_str = f"LONG {pos.hold_days}d ({unrealised:+.1f}%)"
        elif pos is not None:
            pos_str = f"LONG {pos.hold_days}d"
        else:
            pos_str = "FLAT"

        lines.append(
            f"  {sym:<12} {n_obs:5d} {sig_str:>8} {pctile_str:>7} {pos_str:<20}"
        )

    # Signal sparklines
    lines.append("")
    lines.append("  Signal Sparklines (last 20 days composite):")
    for sym in state.symbols:
        if not state.have_enough_history(sym):
            continue
        series = state.get_density_day_obs(sym)
        signals = compute_composite_signal(series, state.lookback)
        recent = signals[-20:] if len(signals) >= 20 else signals
        if len(recent) > 1:
            lo, hi = min(recent), max(recent)
            spark_chars = " ▁▂▃▄▅▆▇█"
            if hi > lo:
                spark = "".join(
                    spark_chars[min(int((v - lo) / (hi - lo) * 8), 8)]
                    for v in recent
                )
            else:
                spark = "▄" * len(recent)
            lines.append(f"    {sym:<12} {spark} ({lo:+.3f} -> {hi:+.3f})")

    # Performance
    lines.append("")
    n_trades = len(state.closed_trades)
    active_count = state.active_position_count()
    lines.append(f"  Trades:     {n_trades} closed, {active_count} active")

    if n_trades > 0:
        lines.append(
            f"  Equity:     {state.equity:.4f} "
            f"({state.total_return_pct():+.2f}%)"
        )
        lines.append(f"  Win rate:   {state.win_rate():.0%}")
        lines.append(f"  Avg P&L:    {state.avg_pnl_pct():+.2f}%")

        # Trades by symbol
        by_sym = state.trades_by_symbol()
        lines.append("")
        lines.append("  Trades by Index:")
        for sym in state.symbols:
            trades = by_sym.get(sym, [])
            if trades:
                wins = sum(1 for t in trades if t.pnl_pct > 0)
                total_pnl = sum(t.pnl_pct for t in trades) * 100
                lines.append(
                    f"    {sym:<12}: {len(trades):3d} trades, "
                    f"{wins}/{len(trades)} wins, {total_pnl:+.1f}% total"
                )

        # Recent trades
        lines.append("")
        lines.append(
            f"  {'Sym':<12} {'Entry':>10} {'Exit':>10} {'Days':>4} "
            f"{'Signal':>7} {'P&L':>7} {'Reason'}"
        )
        lines.append("  " + "-" * 68)
        for t in state.closed_trades[-15:]:
            lines.append(
                f"  {t.symbol:<12} {t.entry_date:>10} {t.exit_date:>10} "
                f"{t.hold_days:4d} "
                f"{t.entry_signal:+6.3f} "
                f"{t.pnl_pct * 100:+6.2f}% {t.exit_reason}"
            )
        if len(state.closed_trades) > 15:
            lines.append(
                f"  ... ({len(state.closed_trades) - 15} earlier trades)"
            )

    elif state.started_at:
        lines.append(f"  Started:    {state.started_at}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    """Process a single day: calibrate SANOS, extract density, signal check."""
    target = date.fromisoformat(args.date)
    cache = BhavcopyCache(args.data_dir)
    state_path = Path(args.state_file)
    state = DensityPaperState.load(state_path)

    if not is_trading_day(target):
        print(f"{target} is not a trading day.")
        return

    print(f"\n--- RNDR Scanning {target} ---\n")

    for sym in state.symbols:
        obs = _process_single_day(cache, target, sym, state)
        if obs is None:
            print(f"  {sym}: No data or SANOS failed")
            continue

        state.append_observation(obs)
        sig_type, composite, sig_pctile = state.check_signal(sym)

        print(f"  {sym}:")
        print(f"    Spot:       {obs.spot:,.1f}")
        print(f"    ATM IV:     {obs.atm_iv * 100:.1f}%")
        print(f"    RN Skew:    {obs.rn_skewness:.3f}")
        print(f"    Skew Prem:  {obs.skew_premium:+.3f}")
        print(f"    Left Tail:  {obs.left_tail:.3f}")
        print(f"    Entropy:    {obs.entropy:.3f} (dH={obs.entropy_change:+.3f})")
        print(f"    KL Div:     {obs.kl_div:.4f}")
        print(f"    Composite:  {composite:+.3f} (pctile={sig_pctile:.2f})")
        print(f"    Signal:     {_format_signal(sig_type)}")
        print()

    state.save(state_path)
    print(f"State saved to {state_path}")


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading loop: daily scan -> signal -> position management."""
    state_path = Path(args.state_file)
    cache = BhavcopyCache(args.data_dir)

    if args.reset and state_path.exists():
        state_path.unlink()
        print(f"Cleared state: {state_path}")

    state = DensityPaperState.load(state_path)

    # Apply config from CLI
    state.lookback = args.lookback
    state.entry_pctile = args.entry
    state.exit_pctile = args.exit_pctile
    state.max_hold_days = args.hold
    state.cost_bps = args.cost

    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    signal.signal(signal.SIGINT, _handle_sigint)

    scan_count = 0

    while _running:
        try:
            scan_count += 1
            today = date.fromisoformat(args.date) if args.date else date.today()

            if not is_trading_day(today):
                print(f"{today} is not a trading day. Waiting...")
                if args.once:
                    break
                time.sleep(args.scan_interval)
                continue

            # Skip if already scanned today
            if state.last_scan_date == today.isoformat():
                print(f"Already scanned {today}.")
                if args.once:
                    break
                time.sleep(args.scan_interval)
                continue

            print(f"\n{'=' * 60}")
            print(f"RNDR Scan #{scan_count} — {today}")
            print("=" * 60)

            for sym in state.symbols:
                obs = _process_single_day(cache, today, sym, state)
                if obs is None:
                    print(f"\n  {sym}: No bhavcopy data or SANOS failed")
                    continue

                state.append_observation(obs)

                # Increment hold days for active position
                if state.positions.get(sym) is not None:
                    state.increment_hold(sym)

                sig_type, composite, sig_pctile = state.check_signal(sym)

                print(f"\n  {sym}:")
                print(
                    f"    Spot={obs.spot:,.1f}, IV={obs.atm_iv * 100:.1f}%, "
                    f"signal={composite:+.3f}, pctile={sig_pctile:.2f}"
                )
                print(f"    Signal: {_format_signal(sig_type)}")

                # Execute signal
                if sig_type == "enter":
                    state.enter_position(sym, obs, composite, sig_pctile)
                    print(f"    >> ENTERED long {sym} @ {obs.spot:,.1f}")

                elif sig_type in ("exit_hold", "exit_signal"):
                    reason = "max_hold" if sig_type == "exit_hold" else "signal_decay"
                    trade = state.exit_position(sym, obs, reason)
                    print(
                        f"    << EXITED: {trade.pnl_pct * 100:+.2f}% "
                        f"({trade.hold_days}d, {reason})"
                    )

            # Update and save
            state.last_scan_date = today.isoformat()
            state.save(state_path)

            # Summary
            print(f"\n  --- Summary ---")
            print(
                f"  Equity: {state.equity:.4f} ({state.total_return_pct():+.2f}%) | "
                f"Trades: {len(state.closed_trades)} | "
                f"Active: {state.active_position_count()}/{len(state.symbols)}"
            )

            if args.once:
                break

            print(f"\n  Sleeping {args.scan_interval}s until next scan...")
            time.sleep(args.scan_interval)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("RNDR paper trading error: %s", e)
            if args.once:
                break
            time.sleep(60)

    state.save(state_path)
    print(f"\nState saved to {state_path}")


def cmd_backfill(args: argparse.Namespace) -> None:
    """Build density history from cached bhavcopy data."""
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    state_path = Path(args.state_file)
    cache = BhavcopyCache(args.data_dir)

    state = DensityPaperState.load(state_path)

    # Apply config from CLI if given
    if args.lookback is not None:
        state.lookback = args.lookback
    if args.entry is not None:
        state.entry_pctile = args.entry
    if args.exit_pctile is not None:
        state.exit_pctile = args.exit_pctile
    if args.hold is not None:
        state.max_hold_days = args.hold
    if args.cost is not None:
        state.cost_bps = args.cost

    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    print(f"RNDR Backfilling density history: {start} to {end}")
    print(f"Indices: {', '.join(state.symbols)}")
    if args.simulate:
        print("  (with paper trading simulation)")
    print()

    t0 = time.monotonic()

    for sym in state.symbols:
        print(f"  {sym}: building density series...", end="", flush=True)
        series = build_density_series(
            cache, start, end, sym, state.phys_window
        )
        print(f" {len(series)} days")

        if not series:
            print(f"    No data for {sym}")
            continue

        # Populate state from series
        for obs in series:
            state.append_observation(DensityObservation.from_day_obs(obs))

        # Store final prev_densities for live continuation
        # We need to redo the last SANOS calibration to get the raw density
        last_obs = series[-1]
        snap, result, _, _ = _calibrate_density(cache, last_obs.date, sym)
        if result is not None:
            K, q = extract_density(result, 0)
            state.prev_densities[sym] = q.tolist()

        # Simulate paper trading if requested
        if args.simulate and len(series) >= state.lookback:
            signals = compute_composite_signal(series, state.lookback)
            n = len(series)

            for i in range(state.lookback, n):
                d_obs = series[i]
                d_iso = d_obs.date.isoformat()
                # Find the matching stored observation
                obs = state.density_histories[sym][-1]
                for stored_obs in state.density_histories[sym]:
                    if stored_obs.date == d_iso:
                        obs = stored_obs
                        break

                sig_pctile = _rolling_percentile(signals, i, state.lookback)
                composite = signals[i]

                pos = state.positions.get(sym)

                if pos is not None:
                    state.increment_hold(sym)

                if pos is None:
                    if sig_pctile >= state.entry_pctile and composite > 0:
                        state.enter_position(sym, obs, composite, sig_pctile)
                else:
                    should_exit = (
                        pos.hold_days >= state.max_hold_days
                        or sig_pctile < state.exit_pctile
                    )
                    if should_exit:
                        reason = (
                            "max_hold"
                            if pos.hold_days >= state.max_hold_days
                            else "signal_decay"
                        )
                        state.exit_position(sym, obs, reason)

            # Close any open position at end of data
            pos = state.positions.get(sym)
            if pos is not None:
                last_stored = state.density_histories[sym][-1]
                state.exit_position(sym, last_stored, "end_of_data")

            sym_trades = [t for t in state.closed_trades if t.symbol == sym]
            if sym_trades:
                wins = sum(1 for t in sym_trades if t.pnl_pct > 0)
                total_pnl = sum(t.pnl_pct for t in sym_trades) * 100
                print(
                    f"    -> {len(sym_trades)} trades, "
                    f"{wins}/{len(sym_trades)} wins, "
                    f"{total_pnl:+.1f}% total"
                )

        state.last_scan_date = end.isoformat()

    state.save(state_path)
    elapsed = time.monotonic() - t0

    print(f"\n  Processed in {elapsed:.1f}s")
    for sym in state.symbols:
        n_obs = len(state.density_histories.get(sym, []))
        print(f"    {sym}: {n_obs} observations")

    if args.simulate:
        print(f"\n  Trades: {len(state.closed_trades)}")
        print(
            f"  Equity: {state.equity:.4f} ({state.total_return_pct():+.2f}%)"
        )
        if state.closed_trades:
            print(f"  Win rate: {state.win_rate():.0%}")

    print(f"\n  State saved to {state_path}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current state and performance."""
    state_path = Path(args.state_file)
    state = DensityPaperState.load(state_path)
    print(_format_status(state))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="india_fno.rndr",
        description="RNDR (Risk-Neutral Density Regime) Paper Trader "
                    "(BANKNIFTY, MIDCPNIFTY, FINNIFTY)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--data-dir", default="data/india",
        help="Bhavcopy cache directory",
    )
    parser.add_argument(
        "--state-file", default=str(DEFAULT_STATE_FILE),
        help="Paper trading state file",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Process a single day")
    p_scan.add_argument(
        "--date", default=date.today().isoformat(),
        help="Target date (YYYY-MM-DD)",
    )

    # --- paper ---
    p_paper = sub.add_parser("paper", help="Paper trading loop")
    p_paper.add_argument(
        "--once", action="store_true", help="Single cycle, then exit"
    )
    p_paper.add_argument("--date", default=None, help="Override date (testing)")
    p_paper.add_argument(
        "--scan-interval", type=int, default=3600,
        help="Seconds between scans (default: 3600)",
    )
    p_paper.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    p_paper.add_argument("--entry", type=float, default=DEFAULT_ENTRY_PCTILE)
    p_paper.add_argument("--exit-pctile", type=float, default=DEFAULT_EXIT_PCTILE)
    p_paper.add_argument("--hold", type=int, default=DEFAULT_HOLD_DAYS)
    p_paper.add_argument("--cost", type=float, default=DEFAULT_COST_BPS)
    p_paper.add_argument(
        "--reset", action="store_true", help="Clear state and restart"
    )

    # --- backfill ---
    p_bf = sub.add_parser(
        "backfill", help="Build density history from cached bhavcopy"
    )
    p_bf.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_bf.add_argument("--end", default=None, help="End date (default: today)")
    p_bf.add_argument(
        "--simulate", action="store_true",
        help="Also simulate paper trades during backfill",
    )
    p_bf.add_argument("--lookback", type=int, default=None)
    p_bf.add_argument("--entry", type=float, default=None)
    p_bf.add_argument("--exit-pctile", type=float, default=None)
    p_bf.add_argument("--hold", type=int, default=None)
    p_bf.add_argument("--cost", type=float, default=None)

    # --- status ---
    sub.add_parser("status", help="Show current state and performance")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    Path("data").mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("data/rndr_paper.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    cmd_map = {
        "scan": cmd_scan,
        "paper": cmd_paper,
        "backfill": cmd_backfill,
        "status": cmd_status,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
