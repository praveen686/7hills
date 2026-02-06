"""JSONL headline archive for backtesting news strategies.

Appends every fetched headline to a monthly JSONL file:
  data/india/headlines/{YYYY-MM}.jsonl

Each line is a JSON object:
  {"ts": "ISO8601", "title": "...", "source": "...", "url": "...",
   "stocks": [...], "indices": [...], "event_type": "..."}

Deduplicates by (normalized_title, source) within the same file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from apps.india_news.scraper import IndiaNewsItem

logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path("data/india/headlines")


def _normalize(title: str) -> str:
    """Lowercase, strip whitespace and punctuation for dedup."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _month_file(dt: datetime) -> Path:
    return ARCHIVE_DIR / f"{dt.strftime('%Y-%m')}.jsonl"


def archive_headlines(items: list[IndiaNewsItem]) -> int:
    """Append new headlines to the monthly JSONL archive.

    Returns the number of new (non-duplicate) headlines written.
    """
    if not items:
        return 0

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    path = _month_file(now)

    # Load existing keys for dedup
    existing: set[tuple[str, str]] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing.add((_normalize(obj["title"]), obj["source"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    written = 0
    with open(path, "a") as f:
        for item in items:
            key = (_normalize(item.title), item.source)
            if key in existing:
                continue
            existing.add(key)

            record = {
                "ts": item.published_at.isoformat() if item.published_at else now.isoformat(),
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "stocks": item.stocks,
                "indices": item.indices,
                "event_type": item.event_type,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    if written:
        logger.info("Archived %d new headlines to %s", written, path)
    return written


def read_archive(
    start: datetime | None = None,
    end: datetime | None = None,
    symbol: str | None = None,
) -> list[dict]:
    """Read archived headlines, optionally filtered by date range and symbol.

    Returns list of dicts (raw JSONL records).
    """
    if not ARCHIVE_DIR.exists():
        return []

    results: list[dict] = []
    for path in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Date filter
                if start or end:
                    try:
                        ts = datetime.fromisoformat(obj["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if start and ts < start:
                            continue
                        if end and ts > end:
                            continue
                    except (KeyError, ValueError):
                        continue

                # Symbol filter
                if symbol and symbol not in obj.get("stocks", []):
                    continue

                results.append(obj)

    return results
