"""CLI for Kite 5-level depth collector (futures + options).

Usage:
    python -m apps.kite_depth collect                       # futures + options
    python -m apps.kite_depth collect --futures-only        # futures only
    python -m apps.kite_depth collect --duration 60         # collect for 60s
    python -m apps.kite_depth collect --n-strikes 20        # wider strike window
    python -m apps.kite_depth status                        # show stored data stats
    python -m apps.kite_depth read NIFTY_FUT                # read today's futures depth
    python -m apps.kite_depth read NIFTY_OPT                # read today's options depth
    python -m apps.kite_depth read NIFTY_OPT --date 2026-02-06
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from quantlaxmi.data._paths import KITE_DEPTH_DIR

DEFAULT_BASE_DIR = KITE_DEPTH_DIR


def cmd_collect(args: argparse.Namespace) -> None:
    """Run live depth collection."""
    from .collector import DepthCollector

    collector = DepthCollector(
        base_dir=args.base_dir,
        duration=args.duration,
        indices=args.indices.split(",") if args.indices else None,
        n_strikes=args.n_strikes,
        n_expiries=args.n_expiries,
        futures_only=args.futures_only,
    )
    asyncio.run(collector.run())


def cmd_status(args: argparse.Namespace) -> None:
    """Show stored data statistics."""
    from .storage import DepthStore

    base_dir = args.base_dir
    dates = DepthStore.list_dates(base_dir)

    if not dates:
        print(f"No data found in {base_dir}")
        return

    print(f"Data directory: {base_dir}")
    print(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print()

    total_rows = 0
    for d in dates:
        symbols = DepthStore.list_symbols(base_dir, d)
        day_rows = 0
        sym_info = []
        for s in symbols:
            table = DepthStore.read_depth(base_dir, s, d)
            n = len(table) if table is not None else 0
            day_rows += n
            sym_info.append(f"{s}={n:,}")
        total_rows += day_rows
        print(f"  {d}: {', '.join(sym_info)} ({day_rows:,} total)")

    print(f"\nTotal rows: {total_rows:,}")


def cmd_read(args: argparse.Namespace) -> None:
    """Read and display depth data for a symbol."""
    from .storage import DepthStore

    table = DepthStore.read_depth(args.base_dir, args.symbol, args.date)
    if table is None:
        date_str = args.date or "today"
        print(f"No data for {args.symbol} on {date_str}")
        return

    print(f"Schema: {len(table.schema)} columns")
    print(f"Rows: {len(table):,}")
    print(f"Columns: {table.column_names}")

    df = table.to_pandas()

    # Show unique strikes/expiries for options
    if "strike" in df.columns and df["strike"].max() > 0:
        strikes = sorted(df["strike"].unique())
        expiries = sorted(df["expiry"].unique())
        print(f"Strikes: {len(strikes)} ({strikes[0]:.0f}–{strikes[-1]:.0f})")
        print(f"Expiries: {expiries}")
        print(f"Types: {sorted(df['option_type'].unique())}")

    print()

    if args.tail:
        print(df.tail(args.tail).to_string())
    else:
        print(df.head(args.head).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kite_depth",
        description="Kite 5-level depth collector for NSE futures + options",
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help="Base directory for parquet storage",
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Live depth collection")
    p_collect.add_argument(
        "--duration", type=int, default=None,
        help="Collection duration in seconds (default: until market close)",
    )
    p_collect.add_argument(
        "--indices", type=str, default=None,
        help="Comma-separated index names (default: NIFTY,BANKNIFTY)",
    )
    p_collect.add_argument(
        "--n-strikes", type=int, default=15,
        help="Number of strikes each side of ATM (default: 15)",
    )
    p_collect.add_argument(
        "--n-expiries", type=int, default=2,
        help="Number of nearest expiries to track (default: 2)",
    )
    p_collect.add_argument(
        "--futures-only", action="store_true",
        help="Only collect futures depth (no options)",
    )

    # status
    sub.add_parser("status", help="Show stored data statistics")

    # read
    p_read = sub.add_parser("read", help="Read depth data for a symbol")
    p_read.add_argument("symbol", help="Symbol name (e.g. NIFTY_FUT, NIFTY_OPT)")
    p_read.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD)")
    p_read.add_argument("--head", type=int, default=20, help="Show first N rows")
    p_read.add_argument("--tail", type=int, default=None, help="Show last N rows")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "read":
        cmd_read(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
