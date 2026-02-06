"""Multi-Index IV Mean-Reversion Paper Trader — CLI.

Trades: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY

Subcommands:
    scan      — Process a single day (fetch F&O data, SANOS → ATM IV, signal check)
    backtest  — Full historical backtest over a date range
    paper     — Paper trading loop (daily scan + position management)
    status    — Show current state, position, and performance
    backfill  — Build IV history from all available F&O data

Usage:
    python -m apps.india_fno scan
    python -m apps.india_fno scan --date 2026-02-03
    python -m apps.india_fno backtest --start 2025-01-01 --end 2026-01-31
    python -m apps.india_fno paper --once
    python -m apps.india_fno status
    python -m apps.india_fno backfill --start 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from apps.india_fno.iv_mean_revert import (
    DayObs,
    build_iv_series,
    format_iv_results,
    rolling_percentile,
    run_from_series,
    run_multi_index_backtest,
    format_multi_index_results,
)
from apps.india_fno.paper_state import (
    DEFAULT_STATE_FILE,
    IVObservation,
    MultiIndexPaperState,
    TRADEABLE_INDICES,
)
from apps.india_fno.sanos import fit_sanos, prepare_nifty_chain
from apps.india_scanner.data import is_trading_day, get_fno
from qlx.data.store import MarketDataStore

logger = logging.getLogger(__name__)

_running = True


def _handle_sigint(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calibrate_day(store, d: date, symbol: str = "NIFTY") -> IVObservation | None:
    """Fetch F&O data for `d`, calibrate SANOS, return ATM IV observation for a symbol."""
    try:
        fno = get_fno(store, d)
        if fno.empty:
            return None
    except Exception:
        return None

    chain_data = prepare_nifty_chain(fno, symbol=symbol, max_expiries=2)
    if chain_data is None:
        return None

    spot = chain_data["spot"]
    forward = chain_data["forward"]
    atm_vars = chain_data["atm_variances"]

    sanos_ok = False
    atm_iv = math.sqrt(atm_vars[0])  # fallback: Brenner estimate

    try:
        result = fit_sanos(
            market_strikes=chain_data["market_strikes"],
            market_calls=chain_data["market_calls"],
            market_spreads=chain_data["market_spreads"],
            atm_variances=atm_vars,
            expiry_labels=chain_data["expiry_labels"],
            eta=0.50,
            n_model_strikes=100,
            K_min=0.7,
            K_max=1.5,
        )
        if result.lp_success:
            sanos_ok = True
            atm_strike = np.array([1.0])
            exp_str = result.expiry_labels[0]
            try:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
                T = max((exp_dt - d).days / 365.0, 1 / 365.0)
            except Exception:
                T = atm_vars[0] / max(atm_iv ** 2, 1e-6) if atm_iv > 0 else 7 / 365.0
            iv_arr = result.iv(0, atm_strike, T)
            atm_iv = float(iv_arr[0])
    except Exception as e:
        logger.debug("SANOS failed for %s %s: %s", symbol, d, e)

    return IVObservation(
        date=d.isoformat(),
        spot=spot,
        atm_iv=atm_iv,
        atm_var=float(atm_vars[0]),
        forward=forward,
        sanos_ok=sanos_ok,
        symbol=symbol,
    )


def _format_signal(signal_type: str) -> str:
    """Format signal type for display."""
    labels = {
        "enter": ">> ENTER LONG",
        "exit_hold": "<< EXIT (max hold)",
        "exit_iv": "<< EXIT (IV normalised)",
        "hold": "-- HOLD",
        "wait": "   (no signal)",
        "warmup": "   (warming up)",
    }
    return labels.get(signal_type, signal_type)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _format_status(state: MultiIndexPaperState) -> str:
    """Format current state as a dashboard string."""
    lines = [
        "",
        "Multi-Index IV Mean-Reversion — Paper Trader",
        "=" * 70,
    ]

    # Config
    lines.append(
        f"  Config: lookback={state.iv_lookback}d, "
        f"entry={state.entry_pctile:.0%}, "
        f"exit={state.exit_pctile:.0%}, "
        f"max_hold={state.max_hold_days}d, "
        f"cost={state.cost_bps:.0f}bps"
    )
    lines.append(f"  Indices: {', '.join(state.symbols)}")

    # Per-index summary
    lines.append("")
    lines.append(f"  {'Index':<12} {'Days':>5} {'IV':>7} {'Pctile':>7} {'Position':<20}")
    lines.append("  " + "-" * 60)

    for sym in state.symbols:
        history = state.iv_histories.get(sym, [])
        n_obs = len(history)

        if n_obs > 0:
            latest = history[-1]
            pctile = state.percentile_rank(sym)
            iv_str = f"{latest.atm_iv * 100:.1f}%"
            pctile_str = f"{pctile:.2f}"
        else:
            iv_str = "-"
            pctile_str = "-"

        pos = state.positions.get(sym)
        if pos is not None:
            # Calculate unrealized P&L
            if n_obs > 0:
                current_spot = history[-1].spot
                unrealised = (current_spot - pos.entry_spot) / pos.entry_spot * 100
                pos_str = f"LONG {pos.hold_days}d ({unrealised:+.1f}%)"
            else:
                pos_str = f"LONG {pos.hold_days}d"
        else:
            pos_str = "FLAT"

        lines.append(f"  {sym:<12} {n_obs:5d} {iv_str:>7} {pctile_str:>7} {pos_str:<20}")

    # IV Sparklines for each index
    lines.append("")
    lines.append("  IV Sparklines (last 20 days):")
    for sym in state.symbols:
        history = state.iv_histories.get(sym, [])
        recent = history[-20:] if len(history) >= 20 else history
        if len(recent) > 1:
            ivs = [o.atm_iv for o in recent]
            lo, hi = min(ivs), max(ivs)
            spark_chars = " ▁▂▃▄▅▆▇█"
            if hi > lo:
                spark = "".join(
                    spark_chars[min(int((v - lo) / (hi - lo) * 8), 8)]
                    for v in ivs
                )
            else:
                spark = "▄" * len(ivs)
            lines.append(f"    {sym:<12} {spark} ({lo * 100:.1f}% → {hi * 100:.1f}%)")

    # Performance
    lines.append("")
    n_trades = len(state.closed_trades)
    active_count = state.active_position_count()
    lines.append(f"  Trades:     {n_trades} closed, {active_count} active")

    if n_trades > 0:
        lines.append(f"  Equity:     {state.equity:.4f} "
                     f"({state.total_return_pct():+.2f}%)")
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
                lines.append(f"    {sym:<12}: {len(trades):3d} trades, "
                             f"{wins}/{len(trades)} wins, {total_pnl:+.1f}% total")

        # Recent trades
        lines.append("")
        lines.append(f"  {'Sym':<10} {'Entry':>10} {'Exit':>10} {'Days':>4} "
                     f"{'EntIV':>6} {'ExIV':>6} "
                     f"{'P&L':>7} {'Reason'}")
        lines.append("  " + "-" * 72)
        for t in state.closed_trades[-15:]:  # Last 15 trades
            lines.append(
                f"  {t.symbol:<10} {t.entry_date:>10} {t.exit_date:>10} "
                f"{t.hold_days:4d} "
                f"{t.entry_iv * 100:5.1f}% {t.exit_iv * 100:5.1f}% "
                f"{t.pnl_pct * 100:+6.2f}% {t.exit_reason}"
            )
        if len(state.closed_trades) > 15:
            lines.append(f"  ... ({len(state.closed_trades) - 15} earlier trades)")

    elif state.started_at:
        lines.append(f"  Started:    {state.started_at}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    """Process a single day: fetch F&O data, SANOS calibration, signal check."""
    target = date.fromisoformat(args.date)
    store = MarketDataStore()
    state_path = Path(args.state_file)
    state = MultiIndexPaperState.load(state_path)

    if not is_trading_day(target):
        print(f"{target} is not a trading day.")
        return

    print(f"\n--- Scanning {target} ---\n")

    for sym in state.symbols:
        obs = _calibrate_day(store, target, symbol=sym)
        if obs is None:
            print(f"  {sym}: No data available")
            continue

        # Add observation and compute percentile
        state.append_observation(obs)
        pctile = state.percentile_rank(sym)
        sig = state.check_signal(sym)

        print(f"  {sym}:")
        print(f"    Spot:   {obs.spot:,.1f}")
        print(f"    ATM IV: {obs.atm_iv * 100:.1f}%")
        print(f"    SANOS:  {'OK' if obs.sanos_ok else 'fallback'}")
        print(f"    Pctile: {pctile:.2f}")
        print(f"    Signal: {_format_signal(sig)}")
        print()

    state.save(state_path)
    print(f"State saved to {state_path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a full historical backtest across all indices."""
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    store = MarketDataStore()

    print(f"Running multi-index backtest: {start} to {end}...")
    print(f"Indices: {', '.join(TRADEABLE_INDICES)}\n")

    results = run_multi_index_backtest(
        store, start, end,
        symbols=TRADEABLE_INDICES,
        iv_lookback=args.lookback,
        entry_pctile=args.entry,
        exit_pctile=args.exit_pctile,
        hold_days=args.hold,
        cost_bps=args.cost,
    )

    print(format_multi_index_results(results))


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading loop: daily scan → signal → position management."""
    state_path = Path(args.state_file)
    store = MarketDataStore()

    if args.reset and state_path.exists():
        state_path.unlink()
        print(f"Cleared state: {state_path}")

    state = MultiIndexPaperState.load(state_path)

    # Apply config from CLI
    state.iv_lookback = args.lookback
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
            print(f"Scan #{scan_count} — {today}")
            print("=" * 60)

            # Process each index
            for sym in state.symbols:
                obs = _calibrate_day(store, today, symbol=sym)
                if obs is None:
                    print(f"\n  {sym}: No F&O data")
                    continue

                state.append_observation(obs)

                # Increment hold days for active position
                if state.positions.get(sym) is not None:
                    state.increment_hold(sym)

                pctile = state.percentile_rank(sym)
                sig = state.check_signal(sym)

                print(f"\n  {sym}:")
                print(f"    Spot={obs.spot:,.1f}, IV={obs.atm_iv * 100:.1f}%, "
                      f"pctile={pctile:.2f}")
                print(f"    Signal: {_format_signal(sig)}")

                # Execute signal
                if sig == "enter":
                    state.enter_position(sym, obs, pctile)
                    print(f"    >> ENTERED long {sym} @ {obs.spot:,.1f}")

                elif sig in ("exit_hold", "exit_iv"):
                    reason = "max_hold" if sig == "exit_hold" else "iv_normalised"
                    trade = state.exit_position(sym, obs, reason)
                    print(f"    << EXITED: {trade.pnl_pct * 100:+.2f}% "
                          f"({trade.hold_days}d, {reason})")

            # Update and save
            state.last_scan_date = today.isoformat()
            state.save(state_path)

            # Summary
            print(f"\n  --- Summary ---")
            print(f"  Equity: {state.equity:.4f} ({state.total_return_pct():+.2f}%) | "
                  f"Trades: {len(state.closed_trades)} | "
                  f"Active: {state.active_position_count()}/{len(state.symbols)}")

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


def _calibrate_day_worker(args_tuple: tuple) -> IVObservation | None:
    """Top-level function for ProcessPoolExecutor (must be picklable).

    Args:
        args_tuple: (d_iso, symbol) — all serializable.
    """
    d_iso, symbol = args_tuple
    d = date.fromisoformat(d_iso)
    store = MarketDataStore()
    return _calibrate_day(store, d, symbol=symbol)


def cmd_backfill(args: argparse.Namespace) -> None:
    """Build IV history from F&O data for all indices."""
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    state_path = Path(args.state_file)

    state = MultiIndexPaperState.load(state_path)

    # Apply config from CLI if given
    if args.lookback:
        state.iv_lookback = args.lookback
    if args.entry:
        state.entry_pctile = args.entry
    if args.exit_pctile:
        state.exit_pctile = args.exit_pctile
    if args.hold:
        state.max_hold_days = args.hold

    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    n_workers = len(state.symbols)
    print(f"Backfilling IV history: {start} to {end}")
    print(f"Indices: {', '.join(state.symbols)} ({n_workers} workers)")
    if args.simulate:
        print("  (with paper trading simulation)")
    print()

    # Collect trading days to process
    trading_days: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d):
            trading_days.append(d)
        d += timedelta(days=1)

    days_processed = 0
    signals_fired = 0
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for d in trading_days:
            d_iso = d.isoformat()

            # Figure out which symbols need calibration on this day
            symbols_to_cal = []
            for sym in state.symbols:
                history = state.iv_histories.get(sym, [])
                if history and d_iso <= history[-1].date:
                    continue
                symbols_to_cal.append(sym)

            if not symbols_to_cal:
                d += timedelta(days=1)
                continue

            # Submit all symbols for this day in parallel
            futures = {
                pool.submit(_calibrate_day_worker, (d_iso, sym)): sym
                for sym in symbols_to_cal
            }

            any_added = False
            for fut in as_completed(futures):
                sym = futures[fut]
                obs = fut.result()
                if obs is None:
                    continue

                state.append_observation(obs)
                any_added = True

                # Simulate paper trading if requested (sequential — depends on state)
                if args.simulate:
                    if state.positions.get(sym) is not None:
                        state.increment_hold(sym)

                    sig = state.check_signal(sym)

                    if sig == "enter":
                        pctile = state.percentile_rank(sym)
                        state.enter_position(sym, obs, pctile)
                        signals_fired += 1
                    elif sig in ("exit_hold", "exit_iv"):
                        reason = "max_hold" if sig == "exit_hold" else "iv_normalised"
                        state.exit_position(sym, obs, reason)

            if any_added:
                days_processed += 1

            if args.simulate:
                state.last_scan_date = d_iso

            if days_processed % 20 == 0 and days_processed > 0:
                elapsed = time.monotonic() - t0
                rate = days_processed / elapsed
                remaining = len(trading_days) - days_processed
                print(f"  {days_processed}/{len(trading_days)} days "
                      f"(latest: {d}, {rate:.1f} days/s, ~{remaining/max(rate,0.1):.0f}s left)")

    state.save(state_path)
    elapsed = time.monotonic() - t0

    print(f"\n  Processed {days_processed} trading days in {elapsed:.1f}s")
    for sym in state.symbols:
        n_obs = len(state.iv_histories.get(sym, []))
        print(f"    {sym}: {n_obs} observations")

    if args.simulate:
        print(f"\n  Signals: {signals_fired}")
        print(f"  Trades: {len(state.closed_trades)}")
        print(f"  Equity: {state.equity:.4f} ({state.total_return_pct():+.2f}%)")
        if state.closed_trades:
            print(f"  Win rate: {state.win_rate():.0%}")

    print(f"\n  State saved to {state_path}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current state and performance."""
    state_path = Path(args.state_file)
    state = MultiIndexPaperState.load(state_path)
    print(_format_status(state))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="india_fno",
        description="Multi-Index IV Mean-Reversion Paper Trader (NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE),
                        help="Paper trading state file")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Process a single day")
    p_scan.add_argument("--date", default=date.today().isoformat(),
                        help="Target date (YYYY-MM-DD)")

    # --- backtest ---
    p_bt = sub.add_parser("backtest", help="Full historical backtest")
    p_bt.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_bt.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_bt.add_argument("--lookback", type=int, default=30,
                      help="IV rolling window (default: 30)")
    p_bt.add_argument("--entry", type=float, default=0.80,
                      help="Entry percentile threshold (default: 0.80)")
    p_bt.add_argument("--exit-pctile", type=float, default=0.50,
                      help="Exit percentile threshold (default: 0.50)")
    p_bt.add_argument("--hold", type=int, default=5,
                      help="Max hold days (default: 5)")
    p_bt.add_argument("--cost", type=float, default=5.0,
                      help="Round-trip cost in bps (default: 5)")

    # --- paper ---
    p_paper = sub.add_parser("paper", help="Paper trading loop")
    p_paper.add_argument("--once", action="store_true",
                         help="Single cycle, then exit")
    p_paper.add_argument("--date", default=None,
                         help="Override date (for testing)")
    p_paper.add_argument("--scan-interval", type=int, default=3600,
                         help="Seconds between scans (default: 3600)")
    p_paper.add_argument("--lookback", type=int, default=30)
    p_paper.add_argument("--entry", type=float, default=0.80)
    p_paper.add_argument("--exit-pctile", type=float, default=0.50)
    p_paper.add_argument("--hold", type=int, default=5)
    p_paper.add_argument("--cost", type=float, default=5.0)
    p_paper.add_argument("--reset", action="store_true",
                         help="Clear state and restart")

    # --- backfill ---
    p_bf = sub.add_parser("backfill",
                          help="Build IV history from F&O data")
    p_bf.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_bf.add_argument("--end", default=None, help="End date (default: today)")
    p_bf.add_argument("--simulate", action="store_true",
                      help="Also simulate paper trades during backfill")
    p_bf.add_argument("--lookback", type=int, default=None)
    p_bf.add_argument("--entry", type=float, default=None)
    p_bf.add_argument("--exit-pctile", type=float, default=None)
    p_bf.add_argument("--hold", type=int, default=None)

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
            logging.FileHandler("data/india_fno.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    cmd_map = {
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "paper": cmd_paper,
        "backfill": cmd_backfill,
        "status": cmd_status,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
