"""Real-time crypto news aggregator.

Sources:
  - CryptoPanic API (free, aggregates 50+ sources)
  - CoinGecko trending (free)
  - RSS feeds from major crypto outlets

Each headline is normalized into a NewsItem dataclass for downstream
sentiment analysis and signal generation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Common coin ticker → Binance symbol mapping
COIN_ALIASES: dict[str, str] = {
    "BTC": "BTCUSDT", "BITCOIN": "BTCUSDT",
    "ETH": "ETHUSDT", "ETHEREUM": "ETHUSDT", "ETHER": "ETHUSDT",
    "SOL": "SOLUSDT", "SOLANA": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT", "RIPPLE": "XRPUSDT",
    "DOGE": "DOGEUSDT", "DOGECOIN": "DOGEUSDT",
    "ADA": "ADAUSDT", "CARDANO": "ADAUSDT",
    "AVAX": "AVAXUSDT", "AVALANCHE": "AVAXUSDT",
    "DOT": "DOTUSDT", "POLKADOT": "DOTUSDT",
    "LINK": "LINKUSDT", "CHAINLINK": "LINKUSDT",
    "MATIC": "MATICUSDT", "POLYGON": "MATICUSDT",
    "UNI": "UNIUSDT", "UNISWAP": "UNIUSDT",
    "ATOM": "ATOMUSDT", "COSMOS": "ATOMUSDT",
    "LTC": "LTCUSDT", "LITECOIN": "LTCUSDT",
    "NEAR": "NEARUSDT",
    "APT": "APTUSDT", "APTOS": "APTUSDT",
    "ARB": "ARBUSDT", "ARBITRUM": "ARBUSDT",
    "OP": "OPUSDT", "OPTIMISM": "OPUSDT",
    "SUI": "SUIUSDT",
    "SEI": "SEIUSDT",
    "TIA": "TIAUSDT", "CELESTIA": "TIAUSDT",
    "FET": "FETUSDT",
    "RENDER": "RENDERUSDT", "RNDR": "RENDERUSDT",
    "INJ": "INJUSDT", "INJECTIVE": "INJUSDT",
    "PEPE": "PEPEUSDT",
    "WIF": "WIFUSDT",
    "BONK": "BONKUSDT",
    "SHIB": "SHIBUSDT",
    "FIL": "FILUSDT", "FILECOIN": "FILUSDT",
    "AAVE": "AAVEUSDT",
    "MKR": "MKRUSDT", "MAKER": "MKRUSDT",
    "CRV": "CRVUSDT", "CURVE": "CRVUSDT",
}

# All known Binance USDT perp symbols (populated on first scan)
_binance_symbols: set[str] = set()


def _load_binance_symbols() -> set[str]:
    """Fetch all USDT perpetual symbols from Binance."""
    global _binance_symbols
    if _binance_symbols:
        return _binance_symbols
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10
        )
        resp.raise_for_status()
        for s in resp.json().get("symbols", []):
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
                _binance_symbols.add(s["symbol"])
    except Exception as e:
        logger.warning("Failed to load Binance symbols: %s", e)
    return _binance_symbols


@dataclass(frozen=True)
class NewsItem:
    """A single news headline with metadata."""

    title: str
    source: str                # e.g. "cryptopanic", "coindesk_rss"
    url: str
    published_at: datetime     # UTC
    coins: list[str]           # Binance symbols mentioned, e.g. ["BTCUSDT"]
    raw_currencies: list[str]  # Original currency codes from the source

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.published_at).total_seconds()


# ---------------------------------------------------------------------------
# Coin extraction from text
# ---------------------------------------------------------------------------

# Pattern to match $BTC, $ETH, etc. or standalone tickers
_TICKER_RE = re.compile(
    r"\$([A-Z]{2,10})\b|(?<![a-zA-Z])([A-Z]{2,10})(?:USDT|USD|/USD)\b"
)


def extract_coins(text: str, source_currencies: list[str] | None = None) -> list[str]:
    """Extract Binance symbols mentioned in text.

    Checks against known aliases and actual Binance symbols.
    """
    symbols = set()

    # From source metadata (CryptoPanic provides currency codes)
    for code in (source_currencies or []):
        code_upper = code.upper()
        if code_upper in COIN_ALIASES:
            symbols.add(COIN_ALIASES[code_upper])
        elif f"{code_upper}USDT" in _load_binance_symbols():
            symbols.add(f"{code_upper}USDT")

    # From text patterns
    for match in _TICKER_RE.finditer(text.upper()):
        ticker = match.group(1) or match.group(2)
        if ticker in COIN_ALIASES:
            symbols.add(COIN_ALIASES[ticker])
        elif f"{ticker}USDT" in _load_binance_symbols():
            symbols.add(f"{ticker}USDT")

    # Also check full coin names in text
    text_upper = text.upper()
    for name, sym in COIN_ALIASES.items():
        if len(name) > 3 and name in text_upper:  # only match longer names
            symbols.add(sym)

    return sorted(symbols)


# ---------------------------------------------------------------------------
# CryptoPanic API (free tier, no auth for public posts)
# ---------------------------------------------------------------------------

CRYPTOPANIC_API = "https://cryptopanic.com/api/v1/posts/"


def fetch_cryptopanic(
    auth_token: str | None = None,
    kind: str = "news",
    num_pages: int = 1,
) -> list[NewsItem]:
    """Fetch recent news from CryptoPanic.

    Requires a free API token from https://cryptopanic.com/developers/api/
    Without a token, this source is skipped.
    """
    if not auth_token:
        return []

    items = []
    params: dict = {"kind": kind, "auth_token": auth_token, "public": "true"}

    url = CRYPTOPANIC_API
    for _ in range(num_pages):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("CryptoPanic rate limited, backing off")
                time.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("CryptoPanic fetch failed: %s", e)
            break

        for post in data.get("results", []):
            try:
                pub_str = post.get("published_at", "")
                pub_dt = datetime.fromisoformat(
                    pub_str.replace("Z", "+00:00")
                )
                currencies = [
                    c.get("code", "")
                    for c in post.get("currencies", [])
                ]
                title = post.get("title", "")
                coins = extract_coins(title, currencies)

                items.append(NewsItem(
                    title=title,
                    source="cryptopanic",
                    url=post.get("url", ""),
                    published_at=pub_dt,
                    coins=coins,
                    raw_currencies=currencies,
                ))
            except Exception as e:
                logger.debug("Skip CryptoPanic post: %s", e)

        # Pagination
        url = data.get("next")
        if not url:
            break

    return items


# ---------------------------------------------------------------------------
# RSS feeds
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    # Crypto sources
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "theblock": "https://www.theblock.co/rss.xml",
    "bitcoinmag": "https://bitcoinmagazine.com/.rss/full/",
    "blockworks": "https://blockworks.co/feed",
    "dailyhodl": "https://dailyhodl.com/feed/",
    "unchained": "https://unchainedcrypto.com/feed/",
}


def fetch_rss(
    max_age_minutes: int = 30,
) -> list[NewsItem]:
    """Fetch recent headlines from crypto RSS feeds."""
    items = []
    now = datetime.now(timezone.utc)

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            resp = requests.get(
                feed_url, timeout=10,
                headers={"User-Agent": "QLX/1.0 (RSS Reader)"},
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:20]:
                # Parse publication time
                pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub_struct:
                    pub_dt = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                else:
                    pub_dt = now

                age_min = (now - pub_dt).total_seconds() / 60
                if age_min > max_age_minutes:
                    continue

                title = entry.get("title", "")
                coins = extract_coins(title)

                items.append(NewsItem(
                    title=title,
                    source=f"{source_name}_rss",
                    url=entry.get("link", ""),
                    published_at=pub_dt,
                    coins=coins,
                    raw_currencies=[],
                ))
        except Exception as e:
            logger.warning("RSS feed %s failed: %s", source_name, e)

    return items


# ---------------------------------------------------------------------------
# CoinGecko trending (free, no auth)
# ---------------------------------------------------------------------------


def fetch_coingecko_trending() -> list[NewsItem]:
    """Fetch trending coins from CoinGecko and create pseudo-headlines.

    Useful as a supplementary signal — trending coins often have news.
    """
    items = []
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
        )
        resp.raise_for_status()
        now = datetime.now(timezone.utc)

        for coin_entry in resp.json().get("coins", []):
            coin = coin_entry.get("item", {})
            symbol = coin.get("symbol", "").upper()
            name = coin.get("name", "")
            rank = coin.get("score", 0) + 1  # 0-indexed

            binance_sym = None
            if symbol in COIN_ALIASES:
                binance_sym = COIN_ALIASES[symbol]
            elif f"{symbol}USDT" in _load_binance_symbols():
                binance_sym = f"{symbol}USDT"

            if not binance_sym:
                continue

            price_change = coin.get("data", {}).get(
                "price_change_percentage_24h", {}
            ).get("usd", 0)

            title = (
                f"{name} ({symbol}) trending #{rank} on CoinGecko, "
                f"24h change {price_change:+.1f}%"
            )

            items.append(NewsItem(
                title=title,
                source="coingecko_trending",
                url=f"https://www.coingecko.com/en/coins/{coin.get('id', '')}",
                published_at=now,
                coins=[binance_sym],
                raw_currencies=[symbol],
            ))
    except Exception as e:
        logger.warning("CoinGecko trending failed: %s", e)

    return items


# ---------------------------------------------------------------------------
# Unified scanner
# ---------------------------------------------------------------------------


def scan_news(
    cryptopanic_token: str | None = None,
    max_age_minutes: int = 30,
    include_coingecko: bool = False,
) -> list[NewsItem]:
    """Scan all news sources, return deduplicated items sorted by recency.

    Args:
        cryptopanic_token: API token for CryptoPanic (optional)
        max_age_minutes: Filter out headlines older than this
        include_coingecko: Include CoinGecko trending (default False - too noisy)
    """
    all_items = []

    # CryptoPanic
    cp_items = fetch_cryptopanic(auth_token=cryptopanic_token)
    all_items.extend(cp_items)

    # RSS feeds
    rss_items = fetch_rss(max_age_minutes=max_age_minutes)
    all_items.extend(rss_items)

    # CoinGecko trending (disabled by default - generates noise signals)
    cg_items = []
    if include_coingecko:
        cg_items = fetch_coingecko_trending()
        all_items.extend(cg_items)

    # Deduplicate by title similarity (exact match for now)
    seen_titles: set[str] = set()
    unique = []
    for item in all_items:
        normalized = item.title.strip().lower()
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(item)

    # Sort by most recent first
    unique.sort(key=lambda x: x.published_at, reverse=True)

    # Filter by age
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_minutes * 60
    unique = [
        item for item in unique
        if item.published_at.timestamp() > cutoff
    ]

    logger.info(
        "Scanned %d items (%d CryptoPanic, %d RSS, %d CoinGecko), %d unique after dedup",
        len(all_items), len(cp_items), len(rss_items), len(cg_items), len(unique),
    )
    return unique
