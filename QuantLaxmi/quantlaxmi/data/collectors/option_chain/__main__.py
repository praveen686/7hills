"""Option Chain Collector CLI -- collect and query India FnO option chain snapshots.

Usage:
    python -m quantlaxmi.data.collectors.option_chain collect
    python -m quantlaxmi.data.collectors.option_chain collect --interval 120
    python -m quantlaxmi.data.collectors.option_chain status
    python -m quantlaxmi.data.collectors.option_chain status --date 2026-02-10
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quantlaxmi.data.collectors.option_chain.collector import (
    SNAPSHOT_DIR,
    list_snapshots,
    run_collector,
)

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def cmd_collect(args):
    """Run the live option chain collector."""
    from quantlaxmi.data.collectors.auth import headless_login

    kite = headless_login()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    run_collector(kite, interval_seconds=args.interval, symbols=symbols)


def cmd_status(args):
    """Show stored snapshot statistics."""
    base_dir = SNAPSHOT_DIR

    if not base_dir.exists():
        print(f"No snapshot data found in {base_dir}")
        return

    # List dates
    dates = sorted(
        d.name for d in base_dir.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
    )

    if not dates:
        print(f"No snapshot data found in {base_dir}")
        return

    target_dates = dates[-5:] if args.date is None else [args.date]

    print(f"Option chain snapshots in {base_dir}:")
    print(f"  Total dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print()

    for date in target_dates:
        day_dir = base_dir / date
        if not day_dir.exists():
            print(f"  {date}: no data")
            continue

        files = sorted(day_dir.glob("*.parquet"))
        symbols = set()
        total_rows = 0
        total_size_mb = 0.0

        for f in files:
            sym = f.stem.rsplit("_", 1)[0]
            symbols.add(sym)
            total_size_mb += f.stat().st_size / 1e6
            try:
                df = pd.read_parquet(f)
                total_rows += len(df)
            except Exception:
                pass

        times = [f.stem.rsplit("_", 1)[1] for f in files if "_" in f.stem]
        first_t = times[0] if times else "?"
        last_t = times[-1] if times else "?"

        print(
            f"  {date}: {len(files)} snapshots, {sorted(symbols)}, "
            f"{total_rows:,} rows, {total_size_mb:.1f} MB, "
            f"{first_t}–{last_t}"
        )


def cmd_read(args):
    """Read a specific snapshot file."""
    files = list_snapshots(symbol=args.symbol, date=args.date)

    if not files:
        print(f"No snapshots found for {args.symbol or 'all'} on {args.date or 'today'}")
        return

    if args.latest:
        files = [files[-1]]

    for f in files:
        df = pd.read_parquet(f)
        print(f"\n{f.name}: {len(df)} contracts")
        if len(df) > 0:
            spot = df["underlying_price"].iloc[0]
            fut = df["futures_price"].iloc[0]
            print(f"  Spot: {spot:.1f}, Futures: {fut:.1f}, Basis: {fut - spot:.2f}")
            print(f"  Expiries: {sorted(df['expiry'].unique())}")
            print(f"  Strikes: {df['strike'].min():.0f} – {df['strike'].max():.0f}")
            total_oi = df["oi"].sum()
            print(f"  Total OI: {total_oi:,.0f}")

        if args.head:
            print(df.head(args.head).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="India FnO Option Chain Snapshot Collector",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")

    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Run live option chain collector")
    p_collect.add_argument("--interval", type=int, default=180,
                           help="Snapshot interval in seconds (default: 180)")
    p_collect.add_argument("--symbols", type=str, default=None,
                           help="Comma-separated symbols (default: all 4 indices)")

    # status
    p_status = sub.add_parser("status", help="Show stored snapshot statistics")
    p_status.add_argument("--date", type=str, default=None,
                          help="Show specific date (default: last 5 days)")

    # read
    p_read = sub.add_parser("read", help="Read stored snapshots")
    p_read.add_argument("--symbol", type=str, default=None,
                        help="Filter by symbol")
    p_read.add_argument("--date", type=str, default=None,
                        help="Date (YYYY-MM-DD, default: today)")
    p_read.add_argument("--latest", action="store_true",
                        help="Show only latest snapshot")
    p_read.add_argument("--head", type=int, default=0,
                        help="Show first N rows")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "read":
        cmd_read(args)


if __name__ == "__main__":
    main()
