"""GDELT DOC 2.0 API collector for historical news headlines.

Fetches headlines from the GDELT global news database for:
  - India financial markets (NIFTY, BANKNIFTY, FnO stocks)
  - Crypto markets (Bitcoin, Ethereum)

GDELT coverage: 2017-present, 15-minute update cadence, 250 records/request.
Free, no API key required.

Headlines are stored in the same JSONL archive format as headline_archive.py,
enabling seamless integration with NewsSentimentBuilder.

API docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import requests

from quantlaxmi.data._paths import HEADLINE_ARCHIVE_DIR
from quantlaxmi.data.collectors.news.scraper import classify_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GDELT API
# ---------------------------------------------------------------------------

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS_PER_REQUEST = 250
REQUEST_DELAY_SECONDS = 2.0  # GDELT says 5s but 2s works with jitter + backoff on 429
_SESSION_ID = random.randint(10000, 99999)
_ARCHIVE_LOCK = Lock()  # protects file writes during concurrent category fetches
_REQUEST_LOCK = Lock()  # serializes HTTP requests across threads to respect IP-level rate limit
_last_request_time = 0.0  # monotonic timestamp of last request

# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------

# India financial markets — search Indian sources for market keywords
INDIA_MARKET_QUERIES = [
    'NIFTY sourceCountry:IN sourcelang:eng',
    'BANKNIFTY sourceCountry:IN sourcelang:eng',
    '"stock market" India sourceCountry:IN sourcelang:eng',
    'Sensex sourceCountry:IN sourcelang:eng',
    '"mutual fund" OR "FII" OR "DII" sourceCountry:IN sourcelang:eng',
]

# India FnO stocks — top names
INDIA_STOCK_QUERIES = [
    '"Reliance Industries" OR "TCS" OR "HDFC Bank" OR "Infosys" sourceCountry:IN sourcelang:eng',
    '"ICICI Bank" OR "SBI" OR "Bharti Airtel" OR "ITC" sourceCountry:IN sourcelang:eng',
    '"Kotak" OR "Axis Bank" OR "Bajaj Finance" OR "Wipro" sourceCountry:IN sourcelang:eng',
    '"Tata Motors" OR "Adani" OR "Maruti" OR "L&T" sourceCountry:IN sourcelang:eng',
]

# Crypto markets — global English sources
CRYPTO_QUERIES = [
    'bitcoin OR BTC sourcelang:eng',
    'ethereum OR ETH sourcelang:eng',
    'cryptocurrency OR "crypto market" sourcelang:eng',
]

# US financial markets
US_MARKET_QUERIES = [
    '"S&P 500" OR "SP500" OR "Dow Jones" sourcelang:eng sourceCountry:US',
    'NASDAQ OR "Wall Street" OR "Fed" OR "Federal Reserve" sourcelang:eng sourceCountry:US',
    '"Treasury" OR "bond market" OR "interest rate" sourcelang:eng sourceCountry:US',
    '"tech stocks" OR "earnings season" OR "IPO" sourcelang:eng sourceCountry:US',
]

US_STOCK_QUERIES = [
    '"Apple" OR "Microsoft" OR "NVIDIA" OR "Google" sourcelang:eng sourceCountry:US',
    '"Amazon" OR "Meta" OR "Tesla" OR "Netflix" sourcelang:eng sourceCountry:US',
    '"JPMorgan" OR "Goldman Sachs" OR "Bank of America" sourcelang:eng sourceCountry:US',
]

# European financial markets
EUROPE_MARKET_QUERIES = [
    '"FTSE" OR "DAX" OR "CAC 40" OR "Euro Stoxx" sourcelang:eng',
    '"ECB" OR "Bank of England" OR "European Central Bank" sourcelang:eng',
    '"Euro" OR "Pound" OR "European markets" sourcelang:eng',
    '"Brexit" OR "Eurozone" OR "EU economy" sourcelang:eng',
]

# International / global macro
INTL_QUERIES = [
    '"global markets" OR "world economy" OR "trade war" sourcelang:eng',
    '"oil price" OR "crude oil" OR "OPEC" sourcelang:eng',
    '"gold price" OR "commodities" OR "commodity markets" sourcelang:eng',
    '"China economy" OR "Shanghai" OR "Hang Seng" OR "Nikkei" sourcelang:eng',
    '"IMF" OR "World Bank" OR "emerging markets" sourcelang:eng',
    '"geopolitical" OR "sanctions" OR "tariffs" sourcelang:eng',
]


def _gdelt_datetime(dt: datetime) -> str:
    """Format datetime for GDELT API (YYYYMMDDHHmmss)."""
    return dt.strftime("%Y%m%d%H%M%S")


def _normalize_title(title: str) -> str:
    """Lowercase, strip whitespace and punctuation for dedup."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def fetch_gdelt_articles(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    max_records: int = MAX_RECORDS_PER_REQUEST,
) -> list[dict]:
    """Fetch articles from GDELT DOC 2.0 API.

    Parameters
    ----------
    query : str
        GDELT search query (supports boolean operators, source filters).
    start_dt, end_dt : datetime
        Date range (UTC).
    max_records : int
        Max articles per request (max 250).

    Returns
    -------
    list[dict]
        Raw article dicts with keys: title, url, seendate, domain, language, sourcecountry.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, MAX_RECORDS_PER_REQUEST),
        "startdatetime": _gdelt_datetime(start_dt),
        "enddatetime": _gdelt_datetime(end_dt),
        "sort": "DateDesc",
    }

    global _last_request_time

    for attempt in range(5):
        try:
            # Global rate limiter: ensure minimum spacing between requests across all threads
            with _REQUEST_LOCK:
                now = time.monotonic()
                min_gap = REQUEST_DELAY_SECONDS * random.uniform(0.9, 1.1)
                elapsed = now - _last_request_time
                if elapsed < min_gap:
                    time.sleep(min_gap - elapsed)
                _last_request_time = time.monotonic()

            resp = requests.get(
                GDELT_DOC_ENDPOINT,
                params=params,
                timeout=30,
                headers={"User-Agent": f"QuantLaxmi/1.0 (session={_SESSION_ID})"},
            )
            if resp.status_code == 429:
                # Check Retry-After header first, fallback to escalating backoff
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = int(retry_after)
                    except ValueError:
                        wait = 5 * (attempt + 1)
                else:
                    wait = 5 * (attempt + 1)
                logger.info("GDELT rate limited, waiting %ds (attempt %d/5)", wait, attempt + 1)
                time.sleep(wait)
                # Update last_request_time after backoff so next request respects gap
                with _REQUEST_LOCK:
                    _last_request_time = time.monotonic()
                continue
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return articles
        except requests.exceptions.JSONDecodeError:
            logger.debug("No JSON from GDELT for query=%s, likely no results", query[:60])
            return []
        except Exception as e:
            logger.warning("GDELT fetch failed (attempt %d/5): %s", attempt + 1, e)
            if attempt < 4:
                time.sleep(5)
    return []


def _parse_gdelt_date(seendate: str) -> datetime | None:
    """Parse GDELT seendate like '20260210T083000Z' to datetime."""
    try:
        # Format: YYYYMMDDTHHmmssZ
        clean = seendate.replace("T", "").replace("Z", "")
        return datetime.strptime(clean, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _extract_stocks_from_title(title: str) -> list[str]:
    """Quick stock extraction from GDELT titles (reuses entity module)."""
    try:
        from quantlaxmi.data.collectors.news.entity import extract_stocks, extract_indices
        stocks = extract_stocks(title)
        indices = extract_indices(title)
        return stocks, indices
    except Exception:
        return [], []


def _articles_to_archive_records(articles: list[dict], source_tag: str) -> list[dict]:
    """Convert GDELT articles to headline archive JSONL format."""
    records = []
    for art in articles:
        title = art.get("title", "").strip()
        if not title or len(title) < 10:
            continue

        dt = _parse_gdelt_date(art.get("seendate", ""))
        if dt is None:
            continue

        stocks, indices = _extract_stocks_from_title(title)
        event_type = classify_event(title)

        records.append({
            "ts": dt.isoformat(),
            "title": title,
            "source": f"gdelt_{source_tag}_{art.get('domain', 'unknown')}",
            "url": art.get("url", ""),
            "stocks": stocks,
            "indices": indices,
            "event_type": event_type,
        })
    return records


# ---------------------------------------------------------------------------
# Historical backfill
# ---------------------------------------------------------------------------

def backfill_headlines(
    start_date: str,
    end_date: str,
    categories: list[str] | None = None,
    chunk_days: int = 7,
) -> int:
    """Backfill historical headlines from GDELT into the headline archive.

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    categories : list[str] | None
        Which categories to fetch. Available:
        "india_market", "india_stocks", "crypto",
        "us_market", "us_stocks", "europe", "intl".
        Default: all categories.
    chunk_days : int
        Days per API chunk (GDELT returns max 250 per request).

    Returns
    -------
    int
        Total new headlines written to archive.
    """
    if categories is None:
        categories = [
            "india_market", "india_stocks", "crypto",
            "us_market", "us_stocks", "europe", "intl",
        ]

    query_map = {
        "india_market": INDIA_MARKET_QUERIES,
        "india_stocks": INDIA_STOCK_QUERIES,
        "crypto": CRYPTO_QUERIES,
        "us_market": US_MARKET_QUERIES,
        "us_stocks": US_STOCK_QUERIES,
        "europe": EUROPE_MARKET_QUERIES,
        "intl": INTL_QUERIES,
    }

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    HEADLINE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing keys for dedup
    existing_keys: set[tuple[str, str]] = set()
    for path in HEADLINE_ARCHIVE_DIR.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_keys.add((_normalize_title(obj["title"]), obj.get("source", "")))
                except (json.JSONDecodeError, KeyError):
                    continue

    total_written = 0
    total_fetched = 0

    def _process_category(cat: str) -> tuple[int, int]:
        """Process a single category sequentially, writing to shared archive."""
        cat_fetched = 0
        cat_written = 0
        queries = query_map.get(cat, [])
        for query in queries:
            # Chunk by date range
            chunk_start = start_dt
            while chunk_start < end_dt:
                chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)

                articles = fetch_gdelt_articles(query, chunk_start, chunk_end)
                cat_fetched += len(articles)

                if articles:
                    records = _articles_to_archive_records(articles, cat)

                    # Write to monthly JSONL files (thread-safe)
                    with _ARCHIVE_LOCK:
                        for rec in records:
                            key = (_normalize_title(rec["title"]), rec["source"])
                            if key in existing_keys:
                                continue
                            existing_keys.add(key)

                            try:
                                ts = datetime.fromisoformat(rec["ts"])
                            except ValueError:
                                continue

                            month_file = HEADLINE_ARCHIVE_DIR / f"{ts.strftime('%Y-%m')}.jsonl"
                            with open(month_file, "a") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            cat_written += 1

                logger.info(
                    "GDELT [%s] %s to %s: %d articles, %d new (cat total)",
                    cat,
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    len(articles),
                    cat_written,
                )

                chunk_start = chunk_end + timedelta(seconds=1)
                # Rate limiting handled by global _REQUEST_LOCK in fetch_gdelt_articles

        return cat_fetched, cat_written

    # Process categories concurrently — GDELT rate limits per IP (5s/req),
    # so 3 workers share the global _REQUEST_LOCK; concurrency helps with
    # I/O overlap (parsing, dedup, file writes) while requests are serialized.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_process_category, cat): cat for cat in categories}
        for future in as_completed(futures):
            cat = futures[future]
            try:
                cat_fetched, cat_written = future.result()
                total_fetched += cat_fetched
                total_written += cat_written
                logger.info("GDELT category '%s' done: %d fetched, %d new", cat, cat_fetched, cat_written)
            except Exception as e:
                logger.error("GDELT category '%s' failed: %s", cat, e)

    logger.info(
        "GDELT backfill complete: %d fetched, %d new headlines archived",
        total_fetched, total_written,
    )
    return total_written
