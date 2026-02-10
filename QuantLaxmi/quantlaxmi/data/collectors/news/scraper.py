"""India business news RSS scraper.

Fetches headlines from major Indian financial news sources,
extracts stock mentions, and classifies event types.

Sources:
  - Economic Times (markets + companies)
  - The Hindu BusinessLine
  - LiveMint (markets + companies)
  - Business Standard (markets + companies)
  - Moneycontrol
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import feedparser
import requests

from quantlaxmi.data.collectors.news.entity import extract_indices, extract_stocks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feed URLs
# ---------------------------------------------------------------------------

INDIA_RSS_FEEDS: dict[str, str] = {
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_companies": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
    "hindu_bl": "https://www.thehindubusinessline.com/feeder/default.rss",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "livemint_companies": "https://www.livemint.com/rss/companies",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "bs_companies": "https://www.business-standard.com/rss/companies-101.rss",
    "moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
}

CRYPTO_RSS_FEEDS: dict[str, str] = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "theblock": "https://www.theblock.co/rss.xml",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
}

US_RSS_FEEDS: dict[str, str] = {
    "cnbc_markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "cnbc_economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "wsj_world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
}

EUROPE_RSS_FEEDS: dict[str, str] = {
    "ft_markets": "https://www.ft.com/markets?format=rss",
    "ft_world": "https://www.ft.com/world?format=rss",
    "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "guardian_business": "https://www.theguardian.com/uk/business/rss",
}

INTL_RSS_FEEDS: dict[str, str] = {
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera_economy": "https://www.aljazeera.com/xml/rss/all.xml",
    "scmp_economy": "https://www.scmp.com/rss/5/feed",
    "cnbc_world": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
}

ALL_RSS_FEEDS: dict[str, str] = {
    **INDIA_RSS_FEEDS,
    **CRYPTO_RSS_FEEDS,
    **US_RSS_FEEDS,
    **EUROPE_RSS_FEEDS,
    **INTL_RSS_FEEDS,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndiaNewsItem:
    """A single India business news headline with metadata."""

    title: str
    source: str
    url: str
    published_at: datetime
    stocks: list[str]       # NSE F&O symbols mentioned
    indices: list[str]      # Index references (NIFTY, BANKNIFTY)
    event_type: str         # earnings, regulatory, macro, corporate, general


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

_EVENT_PATTERNS: dict[str, list[str]] = {
    "regulatory": [
        "SEBI", "RBI", "regulatory", "regulation", "compliance",
        "penalty", "fine", "ban", "approval", "licence", "license",
        "drug approval", "USFDA", "FDA", "RERA", "TRAI",
    ],
    "earnings": [
        "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
        "profit", "revenue", "earnings", "net income", "EBITDA",
        "profit margin", "operating margin", "EBITDA margin",
        "beats estimates", "misses estimates", "top line", "bottom line",
        "PAT", "operating profit", "dividend",
    ],
    "macro": [
        "GDP", "inflation", "CPI", "WPI", "rate cut", "rate hike",
        "repo rate", "monetary policy", "fiscal deficit", "Union Budget",
        "budget", "IIP", "PMI", "trade deficit", "current account",
        "rupee", "forex reserves", "FII", "FPI", "DII",
    ],
    "corporate": [
        "merger", "acquisition", "takeover", "buyback", "stake",
        "joint venture", "partnership", "expansion", "capex",
        "order win", "contract", "deal", "IPO", "listing",
        "delisting", "bonus", "split", "rights issue",
        "management change", "CEO", "MD", "chairman",
        "promoter", "pledge",
    ],
}


def classify_event(headline: str) -> str:
    """Classify a headline into an event type based on keywords.

    Returns one of: earnings, regulatory, macro, corporate, general.
    If multiple match, returns the first match in priority order.
    """
    hl_lower = headline.lower()
    for event_type, keywords in _EVENT_PATTERNS.items():
        for kw in keywords:
            if kw.lower() in hl_lower:
                return event_type
    return "general"


# ---------------------------------------------------------------------------
# RSS fetching
# ---------------------------------------------------------------------------

_REQUEST_HEADERS = {
    "User-Agent": "QLX/1.0 (India News Scanner)",
}


def scan_india_news(
    max_age_minutes: int = 60,
    feeds: dict[str, str] | None = None,
    include_crypto: bool = False,
    include_us: bool = False,
    include_europe: bool = False,
    include_intl: bool = False,
    include_all: bool = False,
) -> list[IndiaNewsItem]:
    """Fetch recent headlines from financial RSS feeds.

    Args:
        max_age_minutes: Discard headlines older than this (default 60).
        feeds: Override feed dict (for testing).
        include_crypto: If True, also fetch from crypto RSS feeds.
        include_us: If True, also fetch from US financial feeds.
        include_europe: If True, also fetch from European financial feeds.
        include_intl: If True, also fetch from international feeds.
        include_all: If True, fetch from ALL feeds (overrides individual flags).

    Returns:
        Deduplicated list of IndiaNewsItem sorted by recency.
    """
    if feeds is None:
        if include_all:
            feeds = ALL_RSS_FEEDS
        else:
            feeds = dict(INDIA_RSS_FEEDS)
            if include_crypto:
                feeds.update(CRYPTO_RSS_FEEDS)
            if include_us:
                feeds.update(US_RSS_FEEDS)
            if include_europe:
                feeds.update(EUROPE_RSS_FEEDS)
            if include_intl:
                feeds.update(INTL_RSS_FEEDS)

    now = datetime.now(timezone.utc)
    cutoff_ts = now.timestamp() - max_age_minutes * 60
    all_items: list[IndiaNewsItem] = []

    for source_name, feed_url in feeds.items():
        try:
            resp = requests.get(
                feed_url, timeout=10, headers=_REQUEST_HEADERS,
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            for entry in feed.entries[:30]:
                # Parse publication time
                pub_struct = (
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                )
                if pub_struct:
                    pub_dt = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                else:
                    pub_dt = now

                if pub_dt.timestamp() < cutoff_ts:
                    continue

                title = entry.get("title", "").strip()
                if not title:
                    continue

                stocks = extract_stocks(title)
                indices = extract_indices(title)
                event_type = classify_event(title)

                all_items.append(IndiaNewsItem(
                    title=title,
                    source=source_name,
                    url=entry.get("link", ""),
                    published_at=pub_dt,
                    stocks=stocks,
                    indices=indices,
                    event_type=event_type,
                ))
        except Exception as e:
            logger.warning("RSS feed %s failed: %s", source_name, e)

    # Deduplicate by normalized title
    seen: set[str] = set()
    unique: list[IndiaNewsItem] = []
    for item in all_items:
        key = item.title.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Sort most recent first
    unique.sort(key=lambda x: x.published_at, reverse=True)

    logger.info(
        "India news scan: %d items from %d feeds, %d unique after dedup",
        len(all_items), len(feeds), len(unique),
    )
    return unique
