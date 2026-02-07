"""News-driven momentum signal generator.

Strategy logic:
  1. Score each headline with FinBERT sentiment
  2. Filter to high-confidence, coin-specific signals
  3. Aggregate sentiment per coin (multiple headlines may mention same coin)
  4. Generate trade signals when aggregate sentiment exceeds threshold
  5. Trades have a time-to-live (TTL) — auto-exit after N minutes

Key insight: crypto markets are inefficient enough that news takes
minutes to fully price in. An LLM can classify text faster than most
retail traders can read the headline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from apps.news_momentum.scraper import NewsItem
from apps.news_momentum.sentiment import SentimentClassifier, SentimentResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the news momentum strategy."""

    # Sentiment thresholds
    min_confidence: float = 0.80       # minimum FinBERT confidence to act
    min_abs_score: float = 0.60        # minimum |score| to act
    # Trade parameters
    position_ttl_minutes: float = 15   # auto-exit after this
    max_positions: int = 5             # max concurrent trades
    cost_per_trade_bps: float = 12.0   # cost per leg in bps
    # Filters
    max_news_age_seconds: float = 600  # only act on news < 10 min old
    min_headlines_per_coin: int = 1    # need N headlines to confirm signal
    # Volume filter
    min_volume_usd: float = 50e6       # minimum 24h volume to trade
    # Source filters
    include_coingecko: bool = False    # CoinGecko trending (too noisy by default)


@dataclass(frozen=True)
class ScoredHeadline:
    """A headline with its sentiment analysis attached."""

    news: NewsItem
    sentiment: SentimentResult
    coins: list[str]           # Binance symbols


@dataclass(frozen=True)
class TradeSignal:
    """A trade decision from news analysis."""

    symbol: str
    direction: str             # "long" or "short"
    score: float               # aggregate sentiment score
    confidence: float          # aggregate confidence
    n_headlines: int           # number of supporting headlines
    headlines: list[str]       # the actual headlines
    reason: str


@dataclass
class ActiveTrade:
    """A currently active momentum trade."""

    symbol: str
    direction: str             # "long" or "short"
    entry_time: str            # ISO format
    entry_price: float         # mark price at entry
    ttl_minutes: float         # time to live
    score: float
    headlines: list[str]


def score_headlines(
    news_items: list[NewsItem],
    classifier: SentimentClassifier,
) -> list[ScoredHeadline]:
    """Run sentiment classification on all headlines."""
    if not news_items:
        return []

    texts = [item.title for item in news_items]
    results = classifier.classify(texts)

    scored = []
    for item, sent in zip(news_items, results):
        scored.append(ScoredHeadline(
            news=item,
            sentiment=sent,
            coins=item.coins,
        ))

    return scored


def aggregate_by_coin(
    scored: list[ScoredHeadline],
    config: StrategyConfig,
) -> dict[str, dict]:
    """Aggregate sentiment scores per coin.

    Returns {symbol: {score, confidence, n_headlines, headlines}}.
    """
    coin_data: dict[str, dict] = {}

    for sh in scored:
        # Filter by age
        if sh.news.age_seconds > config.max_news_age_seconds:
            continue

        # Filter by confidence
        if sh.sentiment.confidence < config.min_confidence:
            continue

        for sym in sh.coins:
            if sym not in coin_data:
                coin_data[sym] = {
                    "scores": [],
                    "confidences": [],
                    "headlines": [],
                }
            coin_data[sym]["scores"].append(sh.sentiment.score)
            coin_data[sym]["confidences"].append(sh.sentiment.confidence)
            coin_data[sym]["headlines"].append(sh.news.title)

    # Compute aggregates
    result = {}
    for sym, data in coin_data.items():
        n = len(data["scores"])
        avg_score = sum(data["scores"]) / n
        avg_conf = sum(data["confidences"]) / n
        result[sym] = {
            "score": avg_score,
            "confidence": avg_conf,
            "n_headlines": n,
            "headlines": data["headlines"],
        }

    return result


def generate_signals(
    scored: list[ScoredHeadline],
    active_trades: dict[str, ActiveTrade],
    config: StrategyConfig,
    volumes: dict[str, float] | None = None,
) -> list[TradeSignal]:
    """Generate trade signals from scored headlines.

    Only generates signals for coins not already in an active trade.
    """
    signals = []
    coin_agg = aggregate_by_coin(scored, config)

    slots = config.max_positions - len(active_trades)
    if slots <= 0:
        return signals

    # Rank by absolute score
    candidates = sorted(
        coin_agg.items(),
        key=lambda x: abs(x[1]["score"]),
        reverse=True,
    )

    for sym, data in candidates:
        if slots <= 0:
            break

        # Skip already-traded symbols
        if sym in active_trades:
            continue

        # Minimum headlines
        if data["n_headlines"] < config.min_headlines_per_coin:
            continue

        # Minimum absolute score
        if abs(data["score"]) < config.min_abs_score:
            continue

        # Volume filter
        if volumes and volumes.get(sym, 0) < config.min_volume_usd:
            continue

        direction = "long" if data["score"] > 0 else "short"
        reason = (
            f"{data['n_headlines']} headlines, "
            f"score={data['score']:+.2f}, "
            f"conf={data['confidence']:.2f}"
        )

        signals.append(TradeSignal(
            symbol=sym,
            direction=direction,
            score=data["score"],
            confidence=data["confidence"],
            n_headlines=data["n_headlines"],
            headlines=data["headlines"][:3],  # top 3
            reason=reason,
        ))
        slots -= 1

    return signals


def check_exits(
    active_trades: dict[str, ActiveTrade],
    current_prices: dict[str, float],
) -> list[str]:
    """Check which active trades should be exited (TTL expired).

    Returns list of symbols to exit.
    """
    now = datetime.now(timezone.utc)
    exits = []

    for sym, trade in active_trades.items():
        entry_dt = datetime.fromisoformat(trade.entry_time)
        age_min = (now - entry_dt).total_seconds() / 60

        if age_min >= trade.ttl_minutes:
            exits.append(sym)

    return exits
