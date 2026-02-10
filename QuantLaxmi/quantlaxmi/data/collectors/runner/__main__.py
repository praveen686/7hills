"""CLI for BRAHMASTRA qlx_runner.

Usage:
    python -m apps.qlx_runner status                          # show system status
    python -m apps.qlx_runner run add_timestamps              # run a specific task
    python -m apps.qlx_runner run add_timestamps --dry-run    # preview only
    python -m apps.qlx_runner run-phase phase0                # run all phase 0 tasks
    python -m apps.qlx_runner run-phase phase0 --force        # force re-process
    python -m apps.qlx_runner tasks                           # list registered tasks
    python -m apps.qlx_runner log                             # show activity log
    python -m apps.qlx_runner log --tail 50                   # last 50 lines
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from .log import read_log


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_status(args: argparse.Namespace) -> None:
    """Show system status."""
    from .tasks import status

    info = status()
    print("\n  BRAHMASTRA — System Status")
    print("  " + "=" * 44)

    for task_name, st in info.get("tasks", {}).items():
        icon = "OK" if st == "DONE" else "--"
        print(f"    [{icon}] {task_name:<28s} {st}")

    registered = info.get("registered", {})
    if registered:
        print()
        for phase, task_names in registered.items():
            print(f"    {phase}: {', '.join(task_names)}")

    print()


def cmd_tasks(args: argparse.Namespace) -> None:
    """List all registered tasks."""
    from .tasks import list_tasks

    tasks = list_tasks(phase=args.phase)
    if not tasks:
        print(f"  No tasks registered" + (f" for phase '{args.phase}'" if args.phase else ""))
        return

    print(f"\n  Registered Tasks" + (f" (phase: {args.phase})" if args.phase else ""))
    print("  " + "=" * 50)
    for t in tasks:
        deps = f" [requires: {', '.join(t.dependencies)}]" if t.dependencies else ""
        print(f"    [{t.phase}] {t.name:<25s} {t.description}{deps}")
    print()


def cmd_run(args: argparse.Namespace) -> None:
    """Run a specific task."""
    from .tasks import run_task

    dates = [_parse_date(d) for d in args.dates] if args.dates else None
    result = run_task(
        args.task_name,
        dates=dates,
        force=args.force,
        dry_run=args.dry_run,
    )

    status = result["status"]
    elapsed = result.get("elapsed", 0)
    if status == "ok":
        print(f"\n  {args.task_name}: OK ({elapsed:.1f}s)")
    else:
        print(f"\n  {args.task_name}: FAILED ({elapsed:.1f}s)")
        print(f"  Error: {result.get('error', 'unknown')}")
        sys.exit(1)


def cmd_run_phase(args: argparse.Namespace) -> None:
    """Run all tasks in a phase."""
    from .tasks import run_phase

    dates = [_parse_date(d) for d in args.dates] if args.dates else None
    results = run_phase(
        args.phase_name,
        dates=dates,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"\n  Phase '{args.phase_name}' Summary:")
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    fail = sum(1 for r in results.values() if r["status"] == "fail")
    skip = sum(1 for r in results.values() if r["status"] == "skipped")
    print(f"    {ok} ok, {fail} failed, {skip} skipped")

    if fail:
        sys.exit(1)


def cmd_log(args: argparse.Namespace) -> None:
    """Show the activity log."""
    content = read_log()
    if args.tail:
        lines = content.split("\n")
        content = "\n".join(lines[-args.tail :])
    print(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qlx_runner",
        description="BRAHMASTRA Trading System Runner",
    )
    sub = parser.add_subparsers(dest="command")

    # status
    sub.add_parser("status", help="Show system status")

    # tasks
    p_tasks = sub.add_parser("tasks", help="List registered tasks")
    p_tasks.add_argument("--phase", default=None, help="Filter by phase")

    # run <task>
    p_run = sub.add_parser("run", help="Run a specific task")
    p_run.add_argument("task_name", help="Task name to run")
    p_run.add_argument("--dates", nargs="+", help="Specific dates (YYYY-MM-DD)")
    p_run.add_argument("--force", action="store_true", help="Force re-process")
    p_run.add_argument("--dry-run", action="store_true", help="Show what would run")

    # run-phase <phase>
    p_phase = sub.add_parser("run-phase", help="Run all tasks in a phase")
    p_phase.add_argument("phase_name", help="Phase name (e.g. phase0, phase1)")
    p_phase.add_argument("--dates", nargs="+", help="Specific dates (YYYY-MM-DD)")
    p_phase.add_argument("--force", action="store_true", help="Force re-process")
    p_phase.add_argument("--dry-run", action="store_true", help="Show what would run")

    # log
    p_log = sub.add_parser("log", help="Show activity log")
    p_log.add_argument("--tail", type=int, default=None, help="Show last N lines")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command == "status":
        cmd_status(args)
    elif args.command == "tasks":
        cmd_tasks(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "run-phase":
        cmd_run_phase(args)
    elif args.command == "log":
        cmd_log(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
