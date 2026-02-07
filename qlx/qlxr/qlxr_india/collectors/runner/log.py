"""Activity log for BRAHMASTRA qlx_runner.

Append-only markdown log that records every action, result, and decision.
Each entry has a timestamp, phase, task, status, and message.

Log file: qlxr_india/logs/qlx_runner.md
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "qlx_runner.md"


def _ensure_log() -> None:
    """Create log directory and header if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text(
            "# BRAHMASTRA qlx_runner Activity Log\n\n"
            "Auto-generated. Each entry is append-only.\n\n"
            "---\n\n"
        )


def log(
    phase: str,
    task: str,
    status: str,
    message: str,
    *,
    detail: str | None = None,
) -> None:
    """Append an entry to the activity log.

    Parameters
    ----------
    phase : str
        e.g. "phase0", "phase1"
    task : str
        Short task identifier, e.g. "add_timestamps", "build_iv"
    status : str
        One of: START, OK, FAIL, SKIP, INFO
    message : str
        One-line summary.
    detail : str or None
        Optional multi-line detail block (rendered as code block).
    """
    _ensure_log()

    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    entry = f"### [{now}] {phase} / {task} — **{status}**\n\n{message}\n"
    if detail:
        entry += f"\n```\n{detail}\n```\n"
    entry += "\n---\n\n"

    with open(LOG_FILE, "a") as f:
        f.write(entry)


def log_start(phase: str, task: str, message: str) -> None:
    log(phase, task, "START", message)


def log_ok(phase: str, task: str, message: str, detail: str | None = None) -> None:
    log(phase, task, "OK", message, detail=detail)


def log_fail(phase: str, task: str, message: str, detail: str | None = None) -> None:
    log(phase, task, "FAIL", message, detail=detail)


def log_info(phase: str, task: str, message: str, detail: str | None = None) -> None:
    log(phase, task, "INFO", message, detail=detail)


def read_log() -> str:
    """Return the full log contents."""
    _ensure_log()
    return LOG_FILE.read_text()
