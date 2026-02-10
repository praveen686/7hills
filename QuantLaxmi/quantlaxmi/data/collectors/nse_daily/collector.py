"""NSE Daily Collector — downloads archive files from nsearchives.nseindia.com.

nsearchives.nseindia.com is a static file server (Akamai CDN) that does NOT
require cookies from www.nseindia.com. Only a valid User-Agent header is needed.
Retries with exponential backoff on transient failures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from .files import ALL_FILES, BASE_URL, NSEFile

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Session headers that mimic a browser
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds: 2, 4, 8

# www.nseindia.com requires cookies from a homepage visit before API calls work
_WWW_NSE_URL = "https://www.nseindia.com"


@dataclass
class DownloadResult:
    """Summary of a single date's download run."""

    date: date
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    missing: int = 0
    details: dict[str, str] = field(default_factory=dict)


class NSEDailyCollector:
    """Downloads NSE daily archive files into date-partitioned storage."""

    def __init__(self, base_dir: Path | None = None, tier: int | None = None, ingest: bool = True):
        self.base_dir = base_dir or Path("data/nse/daily")
        self.tier = tier
        self.ingest = ingest
        self._session: requests.Session | None = None
        self._www_session: requests.Session | None = None  # for www.nseindia.com API calls

    @property
    def files(self) -> list[NSEFile]:
        """Files to download, filtered by tier if specified."""
        if self.tier is not None:
            return [f for f in ALL_FILES if f.tier <= self.tier]
        return list(ALL_FILES)

    def _init_session(self) -> None:
        """Create a new session with browser-like headers.

        nsearchives.nseindia.com serves static files and only requires
        a valid User-Agent header — no cookies from www.nseindia.com needed.
        """
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        logger.info("Session initialized")

    def _ensure_session(self) -> requests.Session:
        """Return an active session, creating one if needed."""
        if self._session is None:
            self._init_session()
        assert self._session is not None
        return self._session

    def _init_www_session(self) -> None:
        """Create a session for www.nseindia.com and pre-fetch cookies."""
        self._www_session = requests.Session()
        www_headers = {**_HEADERS}
        # Use gzip/deflate only — brotli requires an extra library that may not be installed,
        # and requests will save raw br bytes without decoding if the library is missing.
        www_headers["Accept-Encoding"] = "gzip, deflate"
        # NSE API returns JSON
        www_headers["Accept"] = "application/json, text/plain, */*"
        self._www_session.headers.update(www_headers)
        try:
            self._www_session.get(_WWW_NSE_URL, timeout=10)
            logger.info("www.nseindia.com session initialized (cookies acquired)")
        except requests.RequestException as e:
            logger.warning("Failed to acquire www.nseindia.com cookies: %s", e)

    def _ensure_www_session(self) -> requests.Session:
        if self._www_session is None:
            self._init_www_session()
        assert self._www_session is not None
        return self._www_session

    def _refresh_session(self) -> None:
        """Force a fresh session (on repeated failures)."""
        logger.info("Refreshing session...")
        if self._session:
            self._session.close()
            self._session = None
        self._init_session()

    def _download_file(self, nse_file: NSEFile, d: date, out_dir: Path) -> str:
        """Download a single file. Returns status string."""
        local_path = out_dir / nse_file.name

        # Skip if already downloaded
        if local_path.exists() and local_path.stat().st_size > 0:
            return "skipped"

        url = nse_file.url_for_date(d)
        uses_www = nse_file.base_url != BASE_URL

        for attempt in range(1, MAX_RETRIES + 1):
            session = self._ensure_www_session() if uses_www else self._ensure_session()
            try:
                resp = session.get(url, timeout=30)

                if resp.status_code == 200:
                    local_path.write_bytes(resp.content)
                    logger.info("  %s — %d bytes", nse_file.name, len(resp.content))
                    return "downloaded"

                if resp.status_code == 404:
                    logger.debug("  %s — 404 (not available)", nse_file.name)
                    return "missing"

                if resp.status_code == 403:
                    logger.warning(
                        "  %s — 403 (attempt %d/%d), refreshing session",
                        nse_file.name, attempt, MAX_RETRIES,
                    )
                    if uses_www:
                        if self._www_session:
                            self._www_session.close()
                        self._www_session = None
                    else:
                        self._refresh_session()
                    time.sleep(BACKOFF_BASE ** attempt)
                    continue

                # Other HTTP errors
                logger.warning(
                    "  %s — HTTP %d (attempt %d/%d)",
                    nse_file.name, resp.status_code, attempt, MAX_RETRIES,
                )
                time.sleep(BACKOFF_BASE ** attempt)

            except requests.Timeout:
                logger.warning(
                    "  %s — timeout (attempt %d/%d), refreshing session",
                    nse_file.name, attempt, MAX_RETRIES,
                )
                self._refresh_session()
                time.sleep(BACKOFF_BASE ** attempt)

            except requests.RequestException as e:
                logger.warning(
                    "  %s — error: %s (attempt %d/%d)",
                    nse_file.name, e, attempt, MAX_RETRIES,
                )
                time.sleep(BACKOFF_BASE ** attempt)

        return "failed"

    def collect(self, d: date | None = None) -> DownloadResult:
        """Download all files for a given date (default: today IST).

        Skips weekends (Sat/Sun). Returns a DownloadResult summary.
        """
        if d is None:
            d = datetime.now(IST).date()

        result = DownloadResult(date=d)

        # Skip weekends
        if d.weekday() >= 5:  # 5=Sat, 6=Sun
            logger.info("Skipping %s (weekend)", d.isoformat())
            return result

        out_dir = self.base_dir / d.isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Collecting %d files for %s -> %s", len(self.files), d.isoformat(), out_dir)

        for nse_file in self.files:
            status = self._download_file(nse_file, d, out_dir)
            result.details[nse_file.name] = status

            if status == "downloaded":
                result.downloaded += 1
            elif status == "skipped":
                result.skipped += 1
            elif status == "missing":
                result.missing += 1
            else:
                result.failed += 1

        logger.info(
            "Done %s: %d downloaded, %d skipped, %d missing, %d failed",
            d.isoformat(), result.downloaded, result.skipped, result.missing, result.failed,
        )

        # Auto-ingest into parquet after download
        if self.ingest and (result.downloaded > 0 or result.skipped > 0):
            self._ingest(d)

        return result

    def backfill(self, start: date, end: date) -> list[DownloadResult]:
        """Download files for a range of dates, skipping weekends.

        Pauses 1 second between dates to be polite to NSE servers.
        """
        results = []
        current = start
        while current <= end:
            result = self.collect(current)
            results.append(result)
            current += timedelta(days=1)
            # Rate limit between dates (skip pause for weekends)
            if current <= end and current.weekday() < 5:
                time.sleep(1)
        return results

    def _ingest(self, d: date) -> None:
        """Convert downloaded files to hive-partitioned parquet."""
        try:
            from quantlaxmi.data.nse_convert import convert_nse_day

            results = convert_nse_day(d)
            if results:
                total = sum(results.values())
                logger.info(
                    "Ingested %s: %d categories, %d rows",
                    d.isoformat(), len(results), total,
                )
            else:
                logger.debug("Ingest %s: nothing new to convert", d.isoformat())
        except Exception as e:
            logger.error("Ingest failed for %s: %s", d.isoformat(), e)

    def close(self) -> None:
        """Close HTTP sessions."""
        if self._session:
            self._session.close()
            self._session = None
        if self._www_session:
            self._www_session.close()
            self._www_session = None
