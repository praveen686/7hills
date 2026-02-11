"""India News Sentiment Paper Trader — CLI.

Scrapes Indian business RSS feeds, extracts F&O stock mentions,
scores sentiment with FinBERT, and paper-trades stock futures.

Subcommands:
    scan    — Fetch RSS, extract stocks, score with FinBERT, print signals
    paper   — Single paper trading cycle (--once for cron, uses Kite prices)
    paper --live  — Continuous intraday mode using Kite live prices
    status  — Dashboard with active/closed trades, equity, win rate

Usage:
    python -m apps.india_news scan
    python -m apps.india_news scan --max-age 120
    python -m apps.india_news paper --once
    python -m apps.india_news paper --live --scan-interval 1800
    python -m apps.india_news status
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from quantlaxmi.data.collectors.news.headline_archive import archive_headlines
from quantlaxmi.data.collectors.news.scraper import scan_india_news
from quantlaxmi.data.collectors.news.state import DEFAULT_STATE_FILE, IndiaNewsTradingState
from quantlaxmi.data.collectors.news.strategy import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_COST_BPS,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_SCORE_THRESHOLD,
    generate_signals,
    score_headlines,
)
from quantlaxmi.strategies.s9_momentum.data import is_trading_day

logger = logging.getLogger(__name__)

_running = True


def _handle_sigint(signum, frame):
    global _running
    _running = False
    logger.info("Received signal %d, shutting down...", signum)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_live_price(kite, symbol: str) -> float | None:
    """Get live price from Kite API for an NSE equity."""
    try:
        key = f"NSE:{symbol}"
        data = kite.quote([key])
        return float(data[key]["last_price"])
    except Exception as e:
        logger.debug("Kite quote failed for %s: %s", symbol, e)
        return None


def _get_kite():
    """Lazy-load a Kite session (reuses microstructure auth)."""
    from quantlaxmi.data.collectors.auth import headless_login
    return headless_login()


def _load_classifier():
    """Lazy-load FinBERT classifier."""
    from quantlaxmi.core.nlp.sentiment import get_classifier
    return get_classifier()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def _format_status(state: IndiaNewsTradingState) -> str:
    """Format state as a dashboard string."""
    lines = [
        "",
        "India News Sentiment — Paper Trader",
        "=" * 60,
    ]

    # Config
    lines.append(
        f"  Config: max_pos={state.max_positions}, "
        f"TTL={state.ttl_days}d, cost={state.cost_bps:.0f}bps, "
        f"conf>={state.confidence_threshold:.2f}, "
        f"|score|>={state.score_threshold:.2f}"
    )

    if state.started_at:
        lines.append(f"  Started:    {state.started_at}")
    if state.last_scan_date:
        lines.append(f"  Last scan:  {state.last_scan_date}")

    # Active trades
    lines.append("")
    n_active = len(state.active_trades)
    lines.append(f"  Active Trades ({n_active}/{state.max_positions}):")
    if n_active > 0:
        lines.append(f"    {'Symbol':<12} {'Dir':<6} {'Entry':>8} "
                     f"{'Days':>4} {'Score':>6} {'Event':<12}")
        lines.append("    " + "-" * 54)
        for t in state.active_trades:
            lines.append(
                f"    {t.symbol:<12} {t.direction:<6} {t.entry_price:8.2f} "
                f"{t.hold_days:4d} {t.score:+5.2f} {t.event_type:<12}"
            )
    else:
        lines.append("    (none)")

    # Performance
    lines.append("")
    n_closed = len(state.closed_trades)
    lines.append(f"  Closed Trades: {n_closed}")

    if n_closed > 0:
        lines.append(f"  Equity:        {state.equity:.4f} "
                     f"({state.total_return_pct():+.2f}%)")
        lines.append(f"  Win rate:      {state.win_rate():.0%}")
        lines.append(f"  Avg P&L:       {state.avg_pnl_pct():+.2f}%")

        # Recent closed trades
        lines.append("")
        lines.append(f"    {'Symbol':<12} {'Dir':<6} {'Entry':>8} {'Exit':>8} "
                     f"{'Days':>4} {'P&L':>7} {'Reason'}")
        lines.append("    " + "-" * 60)
        for t in state.closed_trades[-10:]:
            lines.append(
                f"    {t.symbol:<12} {t.direction:<6} {t.entry_price:8.2f} "
                f"{t.exit_price:8.2f} {t.hold_days:4d} "
                f"{t.pnl_pct * 100:+6.2f}% {t.exit_reason}"
            )
        if n_closed > 10:
            lines.append(f"    ... ({n_closed - 10} earlier trades)")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> None:
    """Fetch RSS, extract stocks, score with FinBERT, print signals."""
    print("\nScanning India business RSS feeds...\n")

    items = scan_india_news(max_age_minutes=args.max_age)
    print(f"  Fetched {len(items)} headlines")

    # Show all headlines with stock mentions
    stock_items = [item for item in items if item.stocks]
    print(f"  {len(stock_items)} mention F&O stocks\n")

    if not stock_items:
        print("  No stock mentions found in recent headlines.")
        return

    # Score with FinBERT
    print("  Loading FinBERT...")
    classifier = _load_classifier()
    scored = score_headlines(items, classifier)
    print(f"  Scored {len(scored)} headlines\n")

    # Show scored headlines
    print(f"  {'Score':>6} {'Conf':>5} {'Event':<12} {'Stocks':<20} {'Headline'}")
    print("  " + "-" * 80)
    for item, sh in scored:
        stocks_str = ",".join(item.stocks)[:19]
        title_short = sh.title[:50]
        print(f"  {sh.weighted_score:+5.2f} {sh.confidence:5.2f} "
              f"{sh.event_type:<12} {stocks_str:<20} {title_short}")

    # Generate signals
    signals = generate_signals(
        scored,
        confidence_threshold=args.confidence,
        score_threshold=args.score,
        max_positions=args.max_positions,
    )

    print(f"\n  Signals ({len(signals)}):")
    if signals:
        for sig in signals:
            print(f"    {sig.direction.upper():<6} {sig.symbol:<12} "
                  f"score={sig.avg_score:+.2f} conf={sig.max_confidence:.2f} "
                  f"event={sig.event_type} ({sig.n_headlines} headlines)")
    else:
        print("    (no signals above threshold)")
    print()


def cmd_paper(args: argparse.Namespace) -> None:
    """Paper trading cycle: scan → score → enter/exit positions."""
    state_path = Path(args.state_file)
    kite = _get_kite()

    if args.reset and state_path.exists():
        state_path.unlink()
        print(f"Cleared state: {state_path}")

    state = IndiaNewsTradingState.load(state_path)

    # Apply config
    state.max_positions = args.max_positions
    state.ttl_days = args.ttl_days
    state.cost_bps = args.cost
    state.confidence_threshold = args.confidence
    state.score_threshold = args.score

    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    today = date.fromisoformat(args.date) if args.date else date.today()

    if not is_trading_day(today):
        print(f"{today} is not a trading day.")
        state.save(state_path)
        return

    if state.last_scan_date == today.isoformat():
        print(f"Already scanned {today}.")
        state.save(state_path)
        return

    print(f"\n{'=' * 60}")
    print(f"India News Paper Trading — {today}")
    print("=" * 60)

    # Step 1: Increment hold days for existing positions
    state.increment_hold_days()

    # Step 2: Exit expired trades (TTL)
    for expired in state.expired_trades():
        price = _get_live_price(kite, expired.symbol)
        if price is None:
            logger.warning("No price for %s on %s, keeping position", expired.symbol, today)
            continue
        state.exit_trade(expired.symbol, price, today.isoformat(), "ttl")
        print(f"  EXIT (TTL) {expired.symbol} @ {price:.2f}")

    # Step 3: Scan news and score
    print("\n  Scanning RSS feeds...")
    items = scan_india_news(max_age_minutes=args.max_age)
    print(f"  {len(items)} headlines fetched")

    # Archive all headlines for future backtesting
    n_archived = archive_headlines(items)
    if n_archived:
        print(f"  {n_archived} new headlines archived")

    stock_items = [item for item in items if item.stocks]
    if stock_items:
        print(f"  {len(stock_items)} mention F&O stocks")
        print("  Scoring with FinBERT...")
        classifier = _load_classifier()
        scored = score_headlines(items, classifier)

        signals = generate_signals(
            scored,
            confidence_threshold=state.confidence_threshold,
            score_threshold=state.score_threshold,
            max_positions=state.max_positions,
        )
    else:
        signals = []

    # Step 4: Exit on signal flip (opposite signal for held stock)
    active_syms = state.active_symbols()
    signal_map = {s.symbol: s for s in signals}
    for active in list(state.active_trades):
        if active.symbol in signal_map:
            sig = signal_map[active.symbol]
            if sig.direction != active.direction:
                price = _get_live_price(kite, active.symbol)
                if price is not None:
                    state.exit_trade(active.symbol, price, today.isoformat(), "signal_flip")
                    print(f"  EXIT (flip) {active.symbol} @ {price:.2f}")

    # Step 5: Enter new positions
    slots = state.available_slots()
    entered = 0
    for sig in signals:
        if entered >= slots:
            break
        if sig.symbol in state.active_symbols():
            continue
        price = _get_live_price(kite, sig.symbol)
        if price is None:
            logger.warning("No price for %s on %s, skipping entry", sig.symbol, today)
            continue
        state.enter_trade(
            symbol=sig.symbol,
            direction=sig.direction,
            price=price,
            date_str=today.isoformat(),
            score=sig.avg_score,
            confidence=sig.max_confidence,
            event_type=sig.event_type,
        )
        print(f"  ENTER {sig.direction.upper()} {sig.symbol} @ {price:.2f} "
              f"(score={sig.avg_score:+.2f})")
        entered += 1

    # Update current prices for all active trades
    for t in state.active_trades:
        cp = _get_live_price(kite, t.symbol)
        if cp is not None:
            t.current_price = cp

    # Save
    state.last_scan_date = today.isoformat()
    state.save(state_path)

    # Summary
    print(f"\n  --- Summary ---")
    print(f"  Active: {len(state.active_trades)}/{state.max_positions}")
    print(f"  Closed: {len(state.closed_trades)}")
    print(f"  Equity: {state.equity:.4f} ({state.total_return_pct():+.2f}%)")
    print(f"  State saved to {state_path}\n")


def cmd_paper_live(args: argparse.Namespace) -> None:
    """Continuous intraday paper trading with live Kite prices."""
    import signal as sig_mod
    sig_mod.signal(sig_mod.SIGINT, _handle_sigint)
    sig_mod.signal(sig_mod.SIGTERM, _handle_sigint)

    state_path = Path(args.state_file)
    if args.reset and state_path.exists():
        state_path.unlink()

    state = IndiaNewsTradingState.load(state_path)
    state.max_positions = args.max_positions
    state.ttl_days = args.ttl_days
    state.cost_bps = args.cost
    state.confidence_threshold = args.confidence
    state.score_threshold = args.score

    if not state.started_at:
        state.started_at = datetime.now(timezone.utc).isoformat()

    kite = _get_kite()
    classifier = _load_classifier()
    scan_num = 0

    logger.info("Live mode started: scan every %ds, max_pos=%d",
                args.scan_interval, state.max_positions)

    while _running:
        scan_num += 1
        today = date.today()

        print(f"\n{'=' * 60}")
        print(f"India News LIVE — Scan #{scan_num} @ {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)

        # 1. Scan RSS
        items = scan_india_news(max_age_minutes=args.max_age)
        print(f"  {len(items)} headlines fetched")

        # Archive
        n_archived = archive_headlines(items)
        if n_archived:
            print(f"  {n_archived} new headlines archived")

        # 2. Score with FinBERT
        stock_items = [item for item in items if item.stocks]
        if stock_items:
            scored = score_headlines(items, classifier)
            signals = generate_signals(
                scored,
                confidence_threshold=state.confidence_threshold,
                score_threshold=state.score_threshold,
                max_positions=state.max_positions,
            )
        else:
            signals = []

        print(f"  {len(signals)} signals")

        # 3. Exit on signal flip
        signal_map = {s.symbol: s for s in signals}
        for active in list(state.active_trades):
            if active.symbol in signal_map:
                sig = signal_map[active.symbol]
                if sig.direction != active.direction:
                    price = _get_live_price(kite, active.symbol)
                    if price is not None:
                        state.exit_trade(active.symbol, price, today.isoformat(), "signal_flip")
                        print(f"  EXIT (flip) {active.symbol} @ {price:.2f}")

        # 4. Exit TTL (check hold_days)
        for expired in state.expired_trades():
            price = _get_live_price(kite, expired.symbol)
            if price is not None:
                state.exit_trade(expired.symbol, price, today.isoformat(), "ttl")
                print(f"  EXIT (TTL) {expired.symbol} @ {price:.2f}")

        # 5. Enter new positions with LIVE prices
        slots = state.available_slots()
        entered = 0
        for sig in signals:
            if entered >= slots:
                break
            if sig.symbol in state.active_symbols():
                continue
            price = _get_live_price(kite, sig.symbol)
            if price is None:
                logger.warning("No live price for %s, skipping", sig.symbol)
                continue
            state.enter_trade(
                symbol=sig.symbol,
                direction=sig.direction,
                price=price,
                date_str=today.isoformat(),
                score=sig.avg_score,
                confidence=sig.max_confidence,
                event_type=sig.event_type,
            )
            print(f"  ENTER {sig.direction.upper()} {sig.symbol} @ {price:.2f} "
                  f"(score={sig.avg_score:+.2f}, live)")
            entered += 1

        # 6. Update current prices for all active trades
        for t in state.active_trades:
            cp = _get_live_price(kite, t.symbol)
            if cp is not None:
                t.current_price = cp

        # Save + summary
        state.last_scan_date = today.isoformat()
        state.save(state_path)

        print(f"\n  Active: {len(state.active_trades)}/{state.max_positions} | "
              f"Closed: {len(state.closed_trades)} | "
              f"Equity: {state.equity:.4f} ({state.total_return_pct():+.2f}%)",
              flush=True)

        # Wait for next cycle
        import time
        time.sleep(args.scan_interval)

    state.save(state_path)
    logger.info("Live mode stopped. State saved.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current state and performance."""
    state_path = Path(args.state_file)
    state = IndiaNewsTradingState.load(state_path)
    print(_format_status(state))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="india_news",
        description="India News Sentiment Paper Trader",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE),
                        help="Paper trading state file")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- scan ---
    p_scan = sub.add_parser("scan", help="Scan RSS, score, show signals")
    p_scan.add_argument("--max-age", type=int, default=60,
                        help="Max headline age in minutes (default: 60)")
    p_scan.add_argument("--confidence", type=float,
                        default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Min confidence threshold (default: 0.70)")
    p_scan.add_argument("--score", type=float,
                        default=DEFAULT_SCORE_THRESHOLD,
                        help="Min |score| threshold (default: 0.50)")
    p_scan.add_argument("--max-positions", type=int,
                        default=DEFAULT_MAX_POSITIONS,
                        help="Max signals to show (default: 5)")

    # --- paper ---
    p_paper = sub.add_parser("paper", help="Paper trading cycle")
    p_paper.add_argument("--once", action="store_true",
                         help="Single cycle then exit (for cron)")
    p_paper.add_argument("--live", action="store_true",
                         help="Continuous intraday mode with Kite live prices")
    p_paper.add_argument("--scan-interval", type=int, default=1800,
                         help="Seconds between scans in live mode (default: 1800)")
    p_paper.add_argument("--date", default=None,
                         help="Override date (YYYY-MM-DD)")
    p_paper.add_argument("--max-age", type=int, default=60,
                         help="Max headline age in minutes")
    p_paper.add_argument("--max-positions", type=int,
                         default=DEFAULT_MAX_POSITIONS)
    p_paper.add_argument("--ttl-days", type=int, default=3,
                         help="Trade TTL in days (default: 3)")
    p_paper.add_argument("--cost", type=float, default=DEFAULT_COST_BPS,
                         help="Round-trip cost in bps (default: 30)")
    p_paper.add_argument("--confidence", type=float,
                         default=DEFAULT_CONFIDENCE_THRESHOLD)
    p_paper.add_argument("--score", type=float,
                         default=DEFAULT_SCORE_THRESHOLD)
    p_paper.add_argument("--reset", action="store_true",
                         help="Clear state and restart")

    # --- status ---
    sub.add_parser("status", help="Show paper trading dashboard")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    from quantlaxmi.data._paths import LOGS_DIR
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(LOGS_DIR / "india_news.log")),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    if args.command == "paper" and getattr(args, "live", False):
        cmd_paper_live(args)
    else:
        cmd_map = {
            "scan": cmd_scan,
            "paper": cmd_paper,
            "status": cmd_status,
        }
        cmd_map[args.command](args)


if __name__ == "__main__":
    main()
