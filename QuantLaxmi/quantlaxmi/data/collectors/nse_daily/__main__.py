"""CLI for NSE Daily Data Collector.

Usage:
    python -m apps.nse_daily collect                    # download today's files
    python -m apps.nse_daily collect --date 2026-02-05  # specific date
    python -m apps.nse_daily backfill --from 2026-01-01 --to 2026-02-05
    python -m apps.nse_daily status                     # show stored data stats
    python -m apps.nse_daily status --date 2026-02-05   # detailed status for a date
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from quantlaxmi.data._paths import NSE_DAILY_DIR

DEFAULT_BASE_DIR = NSE_DAILY_DIR


def _parse_date(s: str) -> date:
    """Parse YYYY-MM-DD date string."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_collect(args: argparse.Namespace) -> None:
    """Download files for a single date."""
    from .collector import NSEDailyCollector

    ingest = not args.no_ingest
    collector = NSEDailyCollector(base_dir=args.base_dir, tier=args.tier, ingest=ingest)
    try:
        d = _parse_date(args.date) if args.date else None
        result = collector.collect(d)
        print(
            f"{result.date}: {result.downloaded} downloaded, "
            f"{result.skipped} skipped, {result.missing} missing, "
            f"{result.failed} failed"
        )
    finally:
        collector.close()


def cmd_backfill(args: argparse.Namespace) -> None:
    """Download files for a date range."""
    from .collector import NSEDailyCollector

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    if start > end:
        print(f"Error: --from ({start}) must be before --to ({end})")
        sys.exit(1)

    ingest = not args.no_ingest
    collector = NSEDailyCollector(base_dir=args.base_dir, tier=args.tier, ingest=ingest)
    try:
        results = collector.backfill(start, end)
        # Summary
        total_dl = sum(r.downloaded for r in results)
        total_skip = sum(r.skipped for r in results)
        total_miss = sum(r.missing for r in results)
        total_fail = sum(r.failed for r in results)
        trading_days = sum(
            1 for r in results
            if r.downloaded + r.skipped + r.missing + r.failed > 0
        )
        print(
            f"\nBackfill complete: {trading_days} trading days, "
            f"{total_dl} downloaded, {total_skip} skipped, "
            f"{total_miss} missing, {total_fail} failed"
        )
    finally:
        collector.close()


def cmd_status(args: argparse.Namespace) -> None:
    """Show stored data statistics."""
    from .files import ALL_FILES

    base_dir: Path = args.base_dir

    if not base_dir.exists():
        print(f"No data found in {base_dir}")
        return

    if args.date:
        # Detailed status for a specific date
        d = _parse_date(args.date)
        day_dir = base_dir / d.isoformat()
        if not day_dir.exists():
            print(f"No data for {d.isoformat()}")
            return

        print(f"Date: {d.isoformat()}")
        print(f"Directory: {day_dir}")
        print()

        expected = {f.name for f in ALL_FILES}
        present = {p.name for p in day_dir.iterdir() if p.is_file()}

        for f in ALL_FILES:
            path = day_dir / f.name
            if path.exists():
                size = path.stat().st_size
                status = f"{size:>10,} bytes"
            else:
                status = "     MISSING"
            tier_label = "T1" if f.tier == 1 else "T2"
            print(f"  [{tier_label}] {f.name:<30s} {status}")

        extra = present - expected
        if extra:
            print(f"\n  Extra files: {', '.join(sorted(extra))}")

        print(f"\n  {len(present & expected)}/{len(expected)} files present")
        return

    # Overview: list all dates
    dates = sorted(
        d.name for d in base_dir.iterdir()
        if d.is_dir() and _is_date_dir(d.name)
    )

    if not dates:
        print(f"No data found in {base_dir}")
        return

    print(f"Data directory: {base_dir}")
    print(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print()

    total_files = 0
    total_bytes = 0
    for d_str in dates:
        day_dir = base_dir / d_str
        files = [p for p in day_dir.iterdir() if p.is_file()]
        day_bytes = sum(p.stat().st_size for p in files)
        total_files += len(files)
        total_bytes += day_bytes
        print(f"  {d_str}: {len(files):>2} files, {day_bytes:>12,} bytes")

    print(f"\nTotal: {total_files} files, {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.1f} MB)")


def _is_date_dir(name: str) -> bool:
    """Check if a directory name looks like YYYY-MM-DD."""
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nse_daily",
        description="NSE Daily Data Collector — download archive files from NSE",
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help="Base directory for downloads (default: data/nse/daily)",
    )
    parser.add_argument(
        "--tier", type=int, default=None, choices=[1, 2],
        help="Only download files up to this tier (1=critical only, 2=all)",
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Download files for a date")
    p_collect.add_argument(
        "--date", type=str, default=None,
        help="Date to collect (YYYY-MM-DD, default: today IST)",
    )
    p_collect.add_argument(
        "--no-ingest", action="store_true",
        help="Skip parquet conversion after download",
    )

    # backfill
    p_backfill = sub.add_parser("backfill", help="Download files for a date range")
    p_backfill.add_argument(
        "--from", dest="start", type=str, required=True,
        help="Start date (YYYY-MM-DD)",
    )
    p_backfill.add_argument(
        "--to", dest="end", type=str, required=True,
        help="End date (YYYY-MM-DD)",
    )
    p_backfill.add_argument(
        "--no-ingest", action="store_true",
        help="Skip parquet conversion after download",
    )

    # status
    p_status = sub.add_parser("status", help="Show stored data statistics")
    p_status.add_argument(
        "--date", type=str, default=None,
        help="Show detailed status for a specific date",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "backfill":
        cmd_backfill(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
