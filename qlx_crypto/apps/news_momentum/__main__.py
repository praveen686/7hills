"""News-driven momentum paper trader.

Usage:
    python -m apps.news_momentum                   # run continuously
    python -m apps.news_momentum --once             # single scan + trade cycle
    python -m apps.news_momentum --reset            # clear state and start fresh
    python -m apps.news_momentum --scan-interval 60 # scan every 60 seconds

Requires: PyTorch, transformers, feedparser (see requirements in sentiment.py)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from apps.news_momentum.scraper import scan_news
from apps.news_momentum.sentiment import get_classifier
from apps.news_momentum.state import DEFAULT_STATE_FILE, TradingState
from apps.news_momentum.strategy import (
    ActiveTrade as StrategyActiveTrade,
    StrategyConfig,
    check_exits,
    generate_signals,
    score_headlines,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

_price_cache: dict[str, float] = {}
_price_cache_time: float = 0
PRICE_CACHE_TTL = 5  # seconds


def fetch_mark_prices() -> dict[str, float]:
    """Fetch current mark prices from Binance Futures."""
    global _price_cache, _price_cache_time
    now = time.time()
    if now - _price_cache_time < PRICE_CACHE_TTL and _price_cache:
        return _price_cache
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10
        )
        resp.raise_for_status()
        prices = {}
        for item in resp.json():
            sym = item.get("symbol", "")
            mp = item.get("markPrice")
            if mp:
                prices[sym] = float(mp)
        _price_cache = prices
        _price_cache_time = now
        return prices
    except Exception as e:
        logger.warning("Failed to fetch mark prices: %s", e)
        return _price_cache


def fetch_volumes() -> dict[str, float]:
    """Fetch 24h quote volumes from Binance Futures."""
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10
        )
        resp.raise_for_status()
        return {
            item["symbol"]: float(item.get("quoteVolume", 0))
            for item in resp.json()
        }
    except Exception as e:
        logger.warning("Failed to fetch volumes: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def render_dashboard(
    state: TradingState,
    config: StrategyConfig,
    prices: dict[str, float],
    last_scan_headlines: int,
    last_scan_signals: int,
) -> None:
    """Print a compact terminal dashboard."""
    print("\033[2J\033[H", end="")  # clear screen
    now = datetime.now(timezone.utc)
    print("=" * 78)
    print("  NEWS MOMENTUM PAPER TRADER")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 78)

    # Performance summary
    n_closed = len(state.closed_trades)
    print(
        f"\n  Total trades: {state.total_trades}  |  "
        f"Active: {len(state.active_trades)}  |  "
        f"Closed: {n_closed}  |  "
        f"Win rate: {state.win_rate():.0f}%"
    )
    if n_closed > 0:
        print(
            f"  Avg P&L: {state.avg_pnl():+.3f}%  |  "
            f"Total P&L: {state.total_pnl():+.3f}%"
        )
    print(
        f"  Last scan: {last_scan_headlines} headlines, "
        f"{last_scan_signals} signals"
    )

    # Active trades
    print(f"\n{'─' * 78}")
    print(f"  {'Symbol':<12} {'Dir':>5} {'Entry':>10} {'Now':>10} "
          f"{'P&L%':>8} {'Age':>6} {'TTL':>5} {'Score':>6}")
    print(f"{'─' * 78}")

    for sym, trade in sorted(state.active_trades.items()):
        price = prices.get(sym, trade.entry_price)
        if trade.direction == "long":
            pnl = (price - trade.entry_price) / trade.entry_price * 100
        else:
            pnl = (trade.entry_price - price) / trade.entry_price * 100

        age = trade.age_minutes()
        pnl_color = "\033[32m" if pnl >= 0 else "\033[31m"
        reset = "\033[0m"

        print(
            f"  {sym:<12} {trade.direction:>5} "
            f"{trade.entry_price:>10.4f} {price:>10.4f} "
            f"{pnl_color}{pnl:>+7.3f}%{reset} "
            f"{age:>5.1f}m {trade.ttl_minutes:>4.0f}m "
            f"{trade.score:>+5.2f}"
        )

    if not state.active_trades:
        print("  (no active trades)")

    # Recent closed trades
    if state.closed_trades:
        recent = state.closed_trades[-5:]
        print(f"\n{'─' * 78}")
        print(f"  Recent exits:")
        for ct in reversed(recent):
            pnl_color = "\033[32m" if ct.pnl_pct >= 0 else "\033[31m"
            reset = "\033[0m"
            print(
                f"    {ct.symbol:<12} {ct.direction:>5} "
                f"{pnl_color}{ct.pnl_pct:>+7.3f}%{reset}  "
                f"{ct.reason}"
            )

    # Config
    print(f"\n{'─' * 78}")
    print(
        f"  Config: conf≥{config.min_confidence:.0%}  "
        f"|score|≥{config.min_abs_score}  "
        f"TTL={config.position_ttl_minutes:.0f}m  "
        f"max_pos={config.max_positions}  "
        f"cost={config.cost_per_trade_bps:.0f}bps"
    )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_cycle(
    state: TradingState,
    config: StrategyConfig,
    classifier,
    cryptopanic_token: str | None,
) -> tuple[int, int]:
    """Run one scan → classify → signal → trade cycle.

    Returns (n_headlines, n_signals).
    """
    prices = fetch_mark_prices()
    volumes = fetch_volumes()

    # 1. Scan news (use wider window for RSS, strategy filters by age)
    news_items = scan_news(
        cryptopanic_token=cryptopanic_token,
        max_age_minutes=30,
        include_coingecko=getattr(config, 'include_coingecko', False),
    )
    logger.info("Fetched %d headlines", len(news_items))

    # Track seen headlines
    new_headlines = []
    for item in news_items:
        normalized = item.title.strip().lower()
        if normalized not in state.headlines_seen:
            state.headlines_seen.append(normalized)
            new_headlines.append(item)

    # Trim seen list
    if len(state.headlines_seen) > 500:
        state.headlines_seen = state.headlines_seen[-500:]

    # 2. Score headlines with FinBERT
    scored = score_headlines(news_items, classifier)
    logger.info("Scored %d headlines", len(scored))

    # 3. Check exits (TTL expired)
    expired_syms = check_exits(
        {
            sym: StrategyActiveTrade(
                symbol=t.symbol,
                direction=t.direction,
                entry_time=t.entry_time,
                entry_price=t.entry_price,
                ttl_minutes=t.ttl_minutes,
                score=t.score,
                headlines=t.headlines,
            )
            for sym, t in state.active_trades.items()
        },
        prices,
    )

    for sym in expired_syms:
        exit_price = prices.get(sym, state.active_trades[sym].entry_price)
        closed = state.record_exit(sym, exit_price, reason="ttl_expired")
        if closed:
            logger.info(
                "EXIT %s %s: P&L=%+.3f%% (TTL expired)",
                closed.direction.upper(), sym, closed.pnl_pct,
            )

    # 4. Generate new signals
    strategy_active = {
        sym: StrategyActiveTrade(
            symbol=t.symbol,
            direction=t.direction,
            entry_time=t.entry_time,
            entry_price=t.entry_price,
            ttl_minutes=t.ttl_minutes,
            score=t.score,
            headlines=t.headlines,
        )
        for sym, t in state.active_trades.items()
    }

    signals = generate_signals(scored, strategy_active, config, volumes)
    logger.info("Generated %d trade signals", len(signals))

    # 5. Execute entries
    for sig in signals:
        entry_price = prices.get(sig.symbol)
        if entry_price is None:
            logger.warning("No price for %s, skipping", sig.symbol)
            continue

        trade = state.record_entry(
            symbol=sig.symbol,
            direction=sig.direction,
            price=entry_price,
            ttl_minutes=config.position_ttl_minutes,
            score=sig.score,
            confidence=sig.confidence,
            n_headlines=sig.n_headlines,
            headlines=sig.headlines,
        )
        logger.info(
            "ENTRY %s %s @ %.4f  score=%+.2f  conf=%.2f  [%s]",
            trade.direction.upper(), trade.symbol, trade.entry_price,
            trade.score, trade.confidence, sig.reason,
        )

    # 6. Save state
    state.save()

    return len(news_items), len(signals)


def main():
    parser = argparse.ArgumentParser(description="News momentum paper trader")
    parser.add_argument("--once", action="store_true", help="Run single cycle")
    parser.add_argument("--reset", action="store_true", help="Clear state")
    parser.add_argument(
        "--scan-interval", type=int, default=90,
        help="Seconds between scans (default: 90)",
    )
    parser.add_argument(
        "--ttl", type=float, default=15,
        help="Trade TTL in minutes (default: 15)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="Max concurrent positions (default: 5)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.80,
        help="Min FinBERT confidence (default: 0.80)",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.60,
        help="Min absolute sentiment score (default: 0.60)",
    )
    parser.add_argument(
        "--min-volume", type=float, default=50e6,
        help="Min 24h volume in USD (default: 50M)",
    )
    parser.add_argument(
        "--cryptopanic-token", type=str, default=None,
        help="CryptoPanic API token (optional)",
    )
    parser.add_argument(
        "--include-coingecko", action="store_true",
        help="Include CoinGecko trending (disabled by default - too noisy)",
    )
    parser.add_argument(
        "--state-file", type=str, default=None,
        help="State file path",
    )
    args = parser.parse_args()

    state_path = Path(args.state_file) if args.state_file else DEFAULT_STATE_FILE

    if args.reset and state_path.exists():
        state_path.unlink()
        logger.info("State cleared")

    config = StrategyConfig(
        min_confidence=args.min_confidence,
        min_abs_score=args.min_score,
        position_ttl_minutes=args.ttl,
        max_positions=args.max_positions,
        min_volume_usd=args.min_volume,
        include_coingecko=args.include_coingecko,
    )

    # Load state
    state = TradingState.load(state_path)

    # Load FinBERT (takes a few seconds)
    logger.info("Loading FinBERT sentiment model...")
    classifier = get_classifier()
    logger.info("Model ready")

    # Handle SIGTERM/SIGINT
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False
        logger.info("Received signal %d, shutting down...", signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.once:
        n_headlines, n_signals = run_cycle(
            state, config, classifier, args.cryptopanic_token
        )
        prices = fetch_mark_prices()
        render_dashboard(state, config, prices, n_headlines, n_signals)
        return

    # Continuous mode
    logger.info(
        "Starting continuous mode (scan every %ds, TTL=%dm, max_pos=%d)",
        args.scan_interval, int(config.position_ttl_minutes),
        config.max_positions,
    )

    last_headlines = 0
    last_signals = 0

    while running:
        try:
            last_headlines, last_signals = run_cycle(
                state, config, classifier, args.cryptopanic_token
            )
        except Exception as e:
            logger.error("Cycle failed: %s", e, exc_info=True)

        # Render dashboard
        try:
            prices = fetch_mark_prices()
            render_dashboard(
                state, config, prices, last_headlines, last_signals
            )
        except Exception as e:
            logger.error("Dashboard render failed: %s", e)

        # Wait for next cycle, checking for exits every 10s
        wait_until = time.time() + args.scan_interval
        while running and time.time() < wait_until:
            # Check for TTL exits mid-cycle
            prices = fetch_mark_prices()
            expired = []
            for sym, trade in list(state.active_trades.items()):
                if trade.is_expired():
                    expired.append(sym)

            for sym in expired:
                exit_price = prices.get(sym, state.active_trades[sym].entry_price)
                closed = state.record_exit(sym, exit_price, reason="ttl_expired")
                if closed:
                    logger.info(
                        "EXIT %s %s: P&L=%+.3f%% (TTL expired mid-cycle)",
                        closed.direction.upper(), sym, closed.pnl_pct,
                    )
                    state.save()

            if expired:
                render_dashboard(
                    state, config, prices, last_headlines, last_signals
                )

            time.sleep(10)

    logger.info("Shutdown complete. Final state saved.")
    state.save()


if __name__ == "__main__":
    main()
