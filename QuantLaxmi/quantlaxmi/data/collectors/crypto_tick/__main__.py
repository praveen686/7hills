"""Tick Collector CLI -- collect, store, and query Binance tick data.

Usage:
    python -m quantlaxmi.data.collectors.crypto_tick collect
    python -m quantlaxmi.data.collectors.crypto_tick collect --duration 60
    python -m quantlaxmi.data.collectors.crypto_tick status
    python -m quantlaxmi.data.collectors.crypto_tick read BTCUSDT
    python -m quantlaxmi.data.collectors.crypto_tick read BTCUSDT --date 2026-02-05
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from quantlaxmi.data.collectors.crypto_tick.collector import CollectorConfig, TickCollector, scan_symbols
from quantlaxmi.data.collectors.crypto_tick.features import LiveRegimeTracker
from quantlaxmi.data.collectors.crypto_tick.storage import TickStore, TickStoreConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def clear_screen():
    print("\033[2J\033[H", end="")


def render_status(collector: TickCollector, tracker: LiveRegimeTracker):
    """Render live terminal dashboard."""
    clear_screen()
    now = datetime.now(timezone.utc)
    stats = collector.stats()
    feat_stats = tracker.stats()

    print("=" * 80)
    print("  Tick Collector -- Binance Futures aggTrade + bookTicker")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    print(f"\n  Symbols: {stats['n_symbols']}  |  "
          f"Connected: {'YES' if stats['ws_connected'] else 'NO'}  |  "
          f"Reconnects: {stats['ws_reconnects']}")
    print(f"  Trades: {stats['total_trades']:,}  |  "
          f"Books: {stats['total_books']:,}  |  "
          f"Rate: {stats['trades_per_sec']:.1f} trades/s")
    print(f"  Running: {stats['elapsed_sec']:.0f}s")

    store_stats = stats.get("store", {})
    print(f"  Stored: {store_stats.get('total_trades', 0):,} trades, "
          f"{store_stats.get('total_books', 0):,} books to parquet")

    # Per-symbol regime
    regimes = feat_stats.get("per_symbol", {})
    if regimes:
        print(f"\n  {'Symbol':14s} {'VPIN':>6s} {'OFI':>7s} {'Hawkes':>7s} "
              f"{'Trades':>8s} {'Books':>8s} {'Cal':>4s}")
        print(f"  {'-' * 60}")
        for sym in sorted(regimes.keys()):
            r = regimes[sym]
            cal = "Y" if r["hawkes_calibrated"] else "N"
            print(f"  {sym:14s} {r['vpin']:6.3f} {r['ofi']:+7.3f} "
                  f"{r['hawkes_ratio']:7.2f} {r['n_trades']:8,} "
                  f"{r['n_books']:8,} {cal:>4s}")

    print(f"\n  Data dir: {store_stats.get('date', 'N/A')}")
    print("  Press Ctrl+C to stop")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

async def cmd_collect(args):
    """Run the live tick collector."""
    config = CollectorConfig(
        min_ann_funding_pct=args.min_funding,
        min_volume_usd=args.min_volume * 1e6,
        max_symbols=args.max_symbols,
        scan_interval_sec=args.scan_interval,
        flush_interval_sec=args.flush_interval,
    )
    store_config = TickStoreConfig(
        base_dir=Path(args.data_dir),
        flush_interval_sec=args.flush_interval,
    )

    store = TickStore(store_config)
    tracker = LiveRegimeTracker(
        vpin_bucket_size=args.vpin_bucket_size,
        vpin_n_buckets=args.vpin_buckets,
    )

    collector = TickCollector(
        store=store,
        config=config,
        on_trade=tracker.on_trade,
        on_book=tracker.on_book,
    )

    # Handle Ctrl+C
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _sigint():
        print("\n\nShutting down...")
        collector.stop()
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _sigint)
    loop.add_signal_handler(signal.SIGTERM, _sigint)

    duration = args.duration if args.duration > 0 else None

    # Run collector with periodic display refresh
    collect_task = asyncio.create_task(collector.run(duration_sec=duration))

    # Display loop (every 2 seconds)
    display_task = None
    if not args.quiet:
        async def display_loop():
            while not stop_event.is_set():
                try:
                    render_status(collector, tracker)
                except Exception:
                    pass
                await asyncio.sleep(2.0)

        display_task = asyncio.create_task(display_loop())

    try:
        await collect_task
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        if display_task:
            display_task.cancel()
            try:
                await display_task
            except asyncio.CancelledError:
                pass
        await collector.close()

    # Final stats
    stats = collector.stats()
    print(f"\nFinal: {stats['total_trades']:,} trades, "
          f"{stats['total_books']:,} books, "
          f"{stats['elapsed_sec']:.0f}s, "
          f"{stats['n_symbols']} symbols")


def cmd_status(args):
    """Show stored data statistics."""
    base_dir = Path(args.data_dir)
    dates = TickStore.list_dates(base_dir)

    if not dates:
        print(f"No tick data found in {base_dir}")
        return

    print(f"Tick data in {base_dir}:")
    print(f"  Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print()

    for date in dates[-5:]:  # last 5 days
        symbols = TickStore.list_symbols(base_dir, date)
        total_trades = 0
        total_books = 0
        total_size_mb = 0.0

        for sym in symbols:
            trades_path = base_dir / date / f"{sym}_trades.parquet"
            book_path = base_dir / date / f"{sym}_book.parquet"

            if trades_path.exists():
                import pyarrow.parquet as pq
                try:
                    meta = pq.read_metadata(str(trades_path))
                    total_trades += meta.num_rows
                    total_size_mb += trades_path.stat().st_size / 1e6
                except Exception:
                    pass

            if book_path.exists():
                try:
                    meta = pq.read_metadata(str(book_path))
                    total_books += meta.num_rows
                    total_size_mb += book_path.stat().st_size / 1e6
                except Exception:
                    pass

        print(f"  {date}: {len(symbols)} symbols, "
              f"{total_trades:,} trades, {total_books:,} books, "
              f"{total_size_mb:.1f} MB")


def cmd_read(args):
    """Read and display stored tick data."""
    import pyarrow.parquet as pq
    base_dir = Path(args.data_dir)

    table = TickStore.read_trades(base_dir, args.symbol, args.date)
    if table is None:
        date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(f"No trade data for {args.symbol} on {date}")
        return

    df = table.to_pandas()
    print(f"Trade data for {args.symbol} ({args.date or 'today'}):")
    print(f"  Rows: {len(df):,}")
    print(f"  Time range: {df['timestamp_ms'].min()} -- {df['timestamp_ms'].max()}")

    if len(df) > 0:
        t_start = datetime.fromtimestamp(df["timestamp_ms"].iloc[0] / 1000, tz=timezone.utc)
        t_end = datetime.fromtimestamp(df["timestamp_ms"].iloc[-1] / 1000, tz=timezone.utc)
        duration = (t_end - t_start).total_seconds()
        print(f"  Duration: {duration:.0f}s ({t_start.strftime('%H:%M:%S')} -- {t_end.strftime('%H:%M:%S')} UTC)")
        print(f"  Price: {df['price'].min():.4f} -- {df['price'].max():.4f}")
        print(f"  Avg trade size: ${df['price'].mean() * df['quantity'].mean():,.2f}")
        print(f"  Buy ratio: {(~df['is_buyer_maker']).mean():.1%}")

    if args.head:
        print(f"\n  First {args.head} rows:")
        print(df.head(args.head).to_string(index=False))


def cmd_scan(args):
    """Show current high-funding symbols that would be collected."""
    config = CollectorConfig(
        min_ann_funding_pct=args.min_funding,
        min_volume_usd=args.min_volume * 1e6,
        max_symbols=args.max_symbols,
    )
    symbols = scan_symbols(config)

    print(f"High-funding symbols (>{config.min_ann_funding_pct}% ann, >${args.min_volume}M vol):")
    print(f"  {'Symbol':14s} {'Ann Funding':>12s} {'24h Vol':>10s} {'Rate/8h':>10s}")
    print(f"  {'-' * 50}")
    for s in symbols:
        vol_m = s["volume_24h_usd"] / 1e6
        print(f"  {s['symbol']:14s} {s['ann_funding_pct']:+11.1f}% "
              f"${vol_m:9.1f}M {s['funding_rate']:+.6f}")
    print(f"\n  Total: {len(symbols)} symbols")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Binance Tick Collector -- aggTrade + bookTicker -> Parquet",
    )
    parser.add_argument("--data-dir", default="data/ticks",
                        help="Base directory for tick storage (default: data/ticks)")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Run live tick collector")
    p_collect.add_argument("--duration", type=float, default=0,
                           help="Collection duration in seconds (0 = infinite)")
    p_collect.add_argument("--min-funding", type=float, default=15.0,
                           help="Min annualized funding %% (default: 15)")
    p_collect.add_argument("--min-volume", type=float, default=5.0,
                           help="Min 24h volume in $M (default: 5)")
    p_collect.add_argument("--max-symbols", type=int, default=20,
                           help="Max symbols to collect (default: 20)")
    p_collect.add_argument("--scan-interval", type=float, default=3600.0,
                           help="Re-scan interval in seconds (default: 3600)")
    p_collect.add_argument("--flush-interval", type=float, default=300.0,
                           help="Parquet flush interval in seconds (default: 300)")
    p_collect.add_argument("--vpin-bucket-size", type=float, default=50_000.0,
                           help="VPIN bucket size in USD (default: 50000)")
    p_collect.add_argument("--vpin-buckets", type=int, default=50,
                           help="VPIN rolling window buckets (default: 50)")
    p_collect.add_argument("--quiet", action="store_true",
                           help="Suppress terminal display")

    # status
    sub.add_parser("status", help="Show stored data statistics")

    # read
    p_read = sub.add_parser("read", help="Read stored tick data")
    p_read.add_argument("symbol", help="Symbol to read (e.g., BTCUSDT)")
    p_read.add_argument("--date", default=None, help="Date (YYYY-MM-DD, default: today)")
    p_read.add_argument("--head", type=int, default=0, help="Show first N rows")

    # scan
    p_scan = sub.add_parser("scan", help="Show current high-funding symbols")
    p_scan.add_argument("--min-funding", type=float, default=15.0)
    p_scan.add_argument("--min-volume", type=float, default=5.0)
    p_scan.add_argument("--max-symbols", type=int, default=20)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    Path("data").mkdir(exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("data/tick_collector.log"),
            logging.StreamHandler() if args.verbose else logging.NullHandler(),
        ],
    )

    if args.command == "collect":
        asyncio.run(cmd_collect(args))
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "read":
        cmd_read(args)
    elif args.command == "scan":
        cmd_scan(args)


if __name__ == "__main__":
    main()
