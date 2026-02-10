"""Historical crypto news collectors for free data sources.

Collects headlines from:
  1. CryptoPanic — aggregated crypto news from 50+ sources
  2. CoinGecko News — major coin coverage
  3. Alternative.me Fear & Greed Index — daily sentiment back to 2018
  4. CryptoCompare News — categorized articles by coin

All collectors write to the same JSONL archive format used by
headline_archive.py and read by NewsSentimentBuilder.

Each headline is stored as:
  {"ts": "ISO8601", "title": "...", "source": "...", "url": "...",
   "stocks": [], "indices": [], "event_type": "..."}
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import requests

from quantlaxmi.data._paths import HEADLINE_ARCHIVE_DIR

logger = logging.getLogger(__name__)

_ARCHIVE_LOCK = Lock()
_USER_AGENT = "QuantLaxmi/1.0 (CryptoNewsCollector)"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, strip whitespace and punctuation for dedup."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _load_existing_keys() -> set[tuple[str, str]]:
    """Load (normalized_title, source) pairs from all archive files."""
    keys: set[tuple[str, str]] = set()
    if not HEADLINE_ARCHIVE_DIR.exists():
        return keys
    for path in HEADLINE_ARCHIVE_DIR.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    keys.add((_normalize_title(obj["title"]), obj.get("source", "")))
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def _write_record(rec: dict, existing_keys: set[tuple[str, str]]) -> bool:
    """Write a single record to the monthly archive file. Returns True if written."""
    key = (_normalize_title(rec["title"]), rec["source"])
    if key in existing_keys:
        return False
    existing_keys.add(key)

    try:
        ts = datetime.fromisoformat(rec["ts"])
    except ValueError:
        return False

    month_file = HEADLINE_ARCHIVE_DIR / f"{ts.strftime('%Y-%m')}.jsonl"
    with open(month_file, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def _classify_crypto_event(title: str) -> str:
    """Classify crypto headline into event type."""
    hl = title.lower()
    if any(kw in hl for kw in ["hack", "exploit", "breach", "vulnerability", "rug pull"]):
        return "security"
    if any(kw in hl for kw in ["sec", "regulat", "ban", "legal", "lawsuit", "compliance"]):
        return "regulatory"
    if any(kw in hl for kw in ["etf", "institutional", "grayscale", "blackrock", "fund"]):
        return "institutional"
    if any(kw in hl for kw in ["halving", "upgrade", "fork", "merge", "layer"]):
        return "technical"
    if any(kw in hl for kw in ["fed", "inflation", "rate", "cpi", "macro", "treasury"]):
        return "macro"
    if any(kw in hl for kw in ["surge", "crash", "rally", "dump", "pump", "ath", "all-time"]):
        return "price_action"
    if any(kw in hl for kw in ["defi", "nft", "dao", "dex", "lending", "staking"]):
        return "defi"
    return "general"


def _extract_crypto_symbols(title: str) -> list[str]:
    """Extract crypto ticker symbols from a title."""
    hl = title.upper()
    symbols = []
    known = {
        "BTC": "BTC", "BITCOIN": "BTC",
        "ETH": "ETH", "ETHEREUM": "ETH",
        "SOL": "SOL", "SOLANA": "SOL",
        "BNB": "BNB", "XRP": "XRP", "RIPPLE": "XRP",
        "ADA": "ADA", "CARDANO": "ADA",
        "DOGE": "DOGE", "DOGECOIN": "DOGE",
        "AVAX": "AVAX", "DOT": "DOT", "MATIC": "MATIC",
        "LINK": "LINK", "UNI": "UNI",
    }
    for keyword, symbol in known.items():
        if keyword in hl and symbol not in symbols:
            symbols.append(symbol)
    return symbols


# ---------------------------------------------------------------------------
# 1. CryptoPanic
# ---------------------------------------------------------------------------

CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts/"
CRYPTOPANIC_DELAY = 1.0  # 1 req/sec rate limit


def _parse_cryptopanic_response(data: dict) -> list[dict]:
    """Parse CryptoPanic API response into archive records."""
    records = []
    results = data.get("results", [])
    for item in results:
        title = (item.get("title") or "").strip()
        if not title or len(title) < 10:
            continue

        published = item.get("published_at", "")
        if not published:
            continue

        # Parse ISO datetime from CryptoPanic
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        # Source domain
        source_info = item.get("source", {})
        domain = source_info.get("domain", "unknown") if isinstance(source_info, dict) else "unknown"

        # Currencies mentioned
        currencies = item.get("currencies", [])
        symbols = []
        if currencies:
            for c in currencies:
                code = c.get("code", "") if isinstance(c, dict) else ""
                if code and code not in symbols:
                    symbols.append(code)

        if not symbols:
            symbols = _extract_crypto_symbols(title)

        # Votes for sentiment hints (stored in title metadata)
        votes = item.get("votes", {})
        vote_str = ""
        if isinstance(votes, dict):
            pos = votes.get("positive", 0) or 0
            neg = votes.get("negative", 0) or 0
            if pos + neg > 0:
                vote_str = f" [votes: +{pos}/-{neg}]"

        records.append({
            "ts": dt.isoformat(),
            "title": title + vote_str if vote_str else title,
            "source": f"cryptopanic_{domain}",
            "url": item.get("url", ""),
            "stocks": symbols,
            "indices": [],
            "event_type": _classify_crypto_event(title),
        })
    return records


def backfill_cryptopanic(
    start_date: str,
    end_date: str,
    currencies: str = "BTC,ETH",
) -> int:
    """Backfill headlines from CryptoPanic API.

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    currencies : str
        Comma-separated currency codes to filter.

    Returns
    -------
    int
        Number of new headlines written.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    HEADLINE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_existing_keys()
    total_written = 0
    page = 1
    max_pages = 200  # Safety limit
    consecutive_empty = 0

    logger.info("CryptoPanic backfill: %s to %s, currencies=%s", start_date, end_date, currencies)

    while page <= max_pages:
        params = {
            "public": "true",
            "kind": "news",
            "currencies": currencies,
            "page": page,
        }

        try:
            resp = requests.get(
                CRYPTOPANIC_BASE,
                params=params,
                timeout=30,
                headers={"User-Agent": _USER_AGENT},
            )

            if resp.status_code == 429:
                logger.info("CryptoPanic rate limited, waiting 10s (page %d)", page)
                time.sleep(10)
                continue

            if resp.status_code == 403:
                # Free tier might be restricted; try with auth_token=free
                params["auth_token"] = "free"
                resp = requests.get(
                    CRYPTOPANIC_BASE,
                    params=params,
                    timeout=30,
                    headers={"User-Agent": _USER_AGENT},
                )

            if resp.status_code != 200:
                logger.warning("CryptoPanic HTTP %d on page %d, stopping", resp.status_code, page)
                break

            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.warning("CryptoPanic: no JSON on page %d, stopping", page)
            break
        except Exception as e:
            logger.warning("CryptoPanic fetch failed (page %d): %s", page, e)
            break

        records = _parse_cryptopanic_response(data)

        if not records:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                logger.info("CryptoPanic: 3 consecutive empty pages, stopping")
                break
        else:
            consecutive_empty = 0

        # Filter by date range and write
        page_written = 0
        all_before_start = True
        with _ARCHIVE_LOCK:
            for rec in records:
                try:
                    ts = datetime.fromisoformat(rec["ts"])
                except ValueError:
                    continue

                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                if ts < start_dt:
                    continue  # Before our range
                if ts > end_dt:
                    all_before_start = False
                    continue  # After our range

                all_before_start = False
                if _write_record(rec, existing_keys):
                    page_written += 1

        total_written += page_written
        logger.info(
            "CryptoPanic page %d: %d records, %d new",
            page, len(records), page_written,
        )

        # Check if we've gone past our date range (oldest articles before start_date)
        # CryptoPanic returns newest first, so if all are before start, stop
        if records and all_before_start:
            logger.info("CryptoPanic: all articles before start_date, stopping")
            break

        # Check for next page
        next_url = data.get("next")
        if not next_url:
            logger.info("CryptoPanic: no more pages")
            break

        page += 1
        time.sleep(CRYPTOPANIC_DELAY)

    logger.info("CryptoPanic backfill complete: %d new headlines", total_written)
    return total_written


# ---------------------------------------------------------------------------
# 2. CoinGecko News
# ---------------------------------------------------------------------------

COINGECKO_NEWS_URL = "https://api.coingecko.com/api/v3/news"
COINGECKO_DELAY = 6.0  # Conservative: ~10 req/min to avoid 429


def _parse_coingecko_response(data: list | dict) -> list[dict]:
    """Parse CoinGecko news API response into archive records."""
    records = []
    # CoinGecko returns {"data": [...]} or just a list
    if isinstance(data, dict):
        items = data.get("data", [])
    elif isinstance(data, list):
        items = data
    else:
        return records

    for item in items:
        title = (item.get("title") or "").strip()
        if not title or len(title) < 10:
            continue

        # CoinGecko uses updated_at as ISO timestamp
        ts_str = item.get("updated_at") or item.get("created_at", "")
        if not ts_str:
            continue

        try:
            if isinstance(ts_str, (int, float)):
                dt = datetime.fromtimestamp(ts_str, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError, OSError):
            continue

        news_site = item.get("news_site") or item.get("author", "unknown")
        if isinstance(news_site, dict):
            news_site = news_site.get("name", "unknown")

        symbols = _extract_crypto_symbols(title)

        records.append({
            "ts": dt.isoformat(),
            "title": title,
            "source": f"coingecko_{news_site}".replace(" ", "_").lower(),
            "url": item.get("url", ""),
            "stocks": symbols,
            "indices": [],
            "event_type": _classify_crypto_event(title),
        })
    return records


def backfill_coingecko_news(
    start_date: str,
    end_date: str,
) -> int:
    """Backfill headlines from CoinGecko News API.

    Note: CoinGecko free tier provides limited historical depth.
    The API may only return recent articles.

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".

    Returns
    -------
    int
        Number of new headlines written.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    HEADLINE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_existing_keys()
    total_written = 0
    page = 1
    max_pages = 200  # Allow deep pagination
    backoff = 60  # Initial backoff on 429

    logger.info("CoinGecko News backfill: %s to %s", start_date, end_date)

    while page <= max_pages:
        try:
            resp = requests.get(
                COINGECKO_NEWS_URL,
                params={"page": page},
                timeout=30,
                headers={"User-Agent": _USER_AGENT},
            )

            if resp.status_code == 429:
                logger.info("CoinGecko rate limited, backing off %ds (page %d)", backoff, page)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)  # Exponential backoff, cap 5 min
                continue

            if resp.status_code != 200:
                logger.warning("CoinGecko HTTP %d on page %d, stopping", resp.status_code, page)
                break

            # Reset backoff on success
            backoff = 60
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.warning("CoinGecko: no JSON on page %d, stopping", page)
            break
        except Exception as e:
            logger.warning("CoinGecko fetch failed (page %d): %s", page, e)
            break

        records = _parse_coingecko_response(data)

        if not records:
            logger.info("CoinGecko: empty page %d, stopping", page)
            break

        # Filter by date range and write
        page_written = 0
        with _ARCHIVE_LOCK:
            for rec in records:
                try:
                    ts = datetime.fromisoformat(rec["ts"])
                except ValueError:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < start_dt or ts > end_dt:
                    continue
                if _write_record(rec, existing_keys):
                    page_written += 1

        total_written += page_written
        logger.info("CoinGecko page %d: %d records, %d new", page, len(records), page_written)

        page += 1
        time.sleep(COINGECKO_DELAY)

    logger.info("CoinGecko News backfill complete: %d new headlines", total_written)
    return total_written


# ---------------------------------------------------------------------------
# 3. Alternative.me Fear & Greed Index
# ---------------------------------------------------------------------------

FEAR_GREED_URL = "https://api.alternative.me/fng/"


def _parse_fear_greed_response(data: dict) -> list[dict]:
    """Parse Fear & Greed Index response into archive records."""
    records = []
    items = data.get("data", [])
    for item in items:
        try:
            value = int(item.get("value", -1))
        except (ValueError, TypeError):
            continue

        if value < 0 or value > 100:
            continue

        classification = item.get("value_classification", "Unknown")

        # Timestamp is Unix seconds
        try:
            ts_unix = int(item.get("timestamp", 0))
            if ts_unix == 0:
                continue
            dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            continue

        # Create headline-style record
        title = f"Crypto Fear & Greed: {value} ({classification})"

        records.append({
            "ts": dt.isoformat(),
            "title": title,
            "source": "fear_greed_index",
            "url": "https://alternative.me/crypto/fear-and-greed-index/",
            "stocks": [],
            "indices": ["CRYPTO_FNG"],
            "event_type": "sentiment",
            # Extra fields for direct feature use (not breaking JSONL format —
            # NewsSentimentBuilder ignores unknown keys)
            "fng_value": value,
            "fng_classification": classification,
        })
    return records


def fetch_fear_greed_index(
    start_date: str,
    end_date: str,
) -> int:
    """Fetch Crypto Fear & Greed Index from Alternative.me.

    Returns ALL daily values in a single request (back to Feb 2018).
    Filters to the requested date range before writing.

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".

    Returns
    -------
    int
        Number of new records written.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    HEADLINE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_existing_keys()

    logger.info("Fear & Greed Index backfill: %s to %s", start_date, end_date)

    try:
        resp = requests.get(
            FEAR_GREED_URL,
            params={"limit": 0, "format": "json"},
            timeout=30,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Fear & Greed fetch failed: %s", e)
        return 0

    records = _parse_fear_greed_response(data)

    # Filter by date range and write
    total_written = 0
    with _ARCHIVE_LOCK:
        for rec in records:
            try:
                ts = datetime.fromisoformat(rec["ts"])
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < start_dt or ts > end_dt:
                continue
            if _write_record(rec, existing_keys):
                total_written += 1

    logger.info(
        "Fear & Greed backfill complete: %d total records, %d in range, %d new written",
        len(records), sum(1 for r in records if _in_range(r, start_dt, end_dt)), total_written,
    )
    return total_written


def _in_range(rec: dict, start_dt: datetime, end_dt: datetime) -> bool:
    """Check if a record's timestamp is within the given range."""
    try:
        ts = datetime.fromisoformat(rec["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return start_dt <= ts <= end_dt
    except (ValueError, KeyError):
        return False


# ---------------------------------------------------------------------------
# 4. CryptoCompare News
# ---------------------------------------------------------------------------

CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
CRYPTOCOMPARE_DELAY = 0.5  # 50 req/sec limit, be conservative


def _parse_cryptocompare_response(data: dict) -> list[dict]:
    """Parse CryptoCompare news API response into archive records."""
    records = []
    items = data.get("Data", [])
    for item in items:
        title = (item.get("title") or "").strip()
        if not title or len(title) < 10:
            continue

        # published_on is Unix timestamp
        try:
            ts_unix = int(item.get("published_on", 0))
            if ts_unix == 0:
                continue
            dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            continue

        source = item.get("source", "unknown")
        if isinstance(source, dict):
            source = source.get("name", "unknown")

        # Categories — e.g., "BTC|ETH|Trading"
        categories = item.get("categories", "")
        symbols = _extract_crypto_symbols(title)
        # Also extract from categories
        if isinstance(categories, str):
            for cat in categories.split("|"):
                cat_upper = cat.strip().upper()
                if cat_upper in ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT"):
                    if cat_upper not in symbols:
                        symbols.append(cat_upper)

        # Truncate body for reference (not stored in JSONL, just used for classification)
        body = (item.get("body") or "")[:200]

        records.append({
            "ts": dt.isoformat(),
            "title": title,
            "source": f"cryptocompare_{source}".replace(" ", "_").lower(),
            "url": item.get("url") or item.get("guid", ""),
            "stocks": symbols,
            "indices": [],
            "event_type": _classify_crypto_event(title + " " + body),
        })
    return records


def backfill_cryptocompare(
    start_date: str,
    end_date: str,
    lang: str = "EN",
) -> int:
    """Backfill headlines from CryptoCompare News API.

    Uses the `lTs` (last timestamp) parameter for backward pagination.

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".
    lang : str
        Language filter (default: EN).

    Returns
    -------
    int
        Number of new headlines written.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    HEADLINE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_existing_keys()
    total_written = 0

    # Start from end_date and paginate backwards using lTs
    lts = int(end_dt.timestamp())
    start_ts = int(start_dt.timestamp())
    max_iterations = 500  # Safety limit
    iteration = 0

    logger.info("CryptoCompare backfill: %s to %s", start_date, end_date)

    while iteration < max_iterations:
        iteration += 1

        params = {
            "lang": lang,
            "lTs": lts,
        }

        try:
            resp = requests.get(
                CRYPTOCOMPARE_NEWS_URL,
                params=params,
                timeout=30,
                headers={"User-Agent": _USER_AGENT},
            )

            if resp.status_code == 429:
                logger.info("CryptoCompare rate limited, waiting 5s")
                time.sleep(5)
                continue

            if resp.status_code != 200:
                logger.warning("CryptoCompare HTTP %d, stopping", resp.status_code)
                break

            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.warning("CryptoCompare: no JSON, stopping")
            break
        except Exception as e:
            logger.warning("CryptoCompare fetch failed: %s", e)
            break

        records = _parse_cryptocompare_response(data)

        if not records:
            logger.info("CryptoCompare: empty response, stopping")
            break

        # Find oldest timestamp for next pagination
        oldest_ts = lts
        for rec in records:
            try:
                ts = datetime.fromisoformat(rec["ts"])
                rec_ts = int(ts.timestamp())
                if rec_ts < oldest_ts:
                    oldest_ts = rec_ts
            except (ValueError, KeyError):
                continue

        # Stop if we haven't made progress (no older articles)
        if oldest_ts >= lts and iteration > 1:
            logger.info("CryptoCompare: no older articles, stopping")
            break

        # Stop if we've gone past start_date
        if oldest_ts < start_ts:
            logger.info("CryptoCompare: reached start_date")

        # Filter by date range and write
        page_written = 0
        with _ARCHIVE_LOCK:
            for rec in records:
                try:
                    ts = datetime.fromisoformat(rec["ts"])
                except ValueError:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < start_dt or ts > end_dt:
                    continue
                if _write_record(rec, existing_keys):
                    page_written += 1

        total_written += page_written
        logger.info(
            "CryptoCompare iter %d: %d records, %d new (lTs=%d)",
            iteration, len(records), page_written, oldest_ts,
        )

        # If oldest is before our start, we're done
        if oldest_ts <= start_ts:
            break

        # Move pagination cursor
        lts = oldest_ts - 1
        time.sleep(CRYPTOCOMPARE_DELAY)

    logger.info("CryptoCompare backfill complete: %d new headlines", total_written)
    return total_written


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def backfill_all_crypto_news(
    start_date: str,
    end_date: str,
) -> int:
    """Run all crypto news collectors sequentially.

    Parameters
    ----------
    start_date, end_date : str
        Date range "YYYY-MM-DD".

    Returns
    -------
    int
        Total new headlines across all sources.
    """
    total = 0

    collectors = [
        ("CryptoPanic", lambda: backfill_cryptopanic(start_date, end_date)),
        ("CoinGecko", lambda: backfill_coingecko_news(start_date, end_date)),
        ("Fear & Greed", lambda: fetch_fear_greed_index(start_date, end_date)),
        ("CryptoCompare", lambda: backfill_cryptocompare(start_date, end_date)),
    ]

    for name, collector_fn in collectors:
        logger.info("--- Running %s collector ---", name)
        try:
            n = collector_fn()
            total += n
            logger.info("%s: %d new headlines", name, n)
        except Exception as e:
            logger.error("%s collector failed: %s", name, e)
            # Continue with other collectors

    logger.info("All crypto news backfill complete: %d total new headlines", total)
    return total
