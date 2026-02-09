"""Signal generation for India news sentiment trading.

Scores headlines with FinBERT (reused from news_momentum),
aggregates per-stock sentiment, applies event-type weights,
and generates trade signals filtered by confidence and score thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from data.collectors.news.scraper import IndiaNewsItem
from core.nlp.sentiment import SentimentClassifier, SentimentResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Event-type multiplier for sentiment scores
EVENT_WEIGHTS: dict[str, float] = {
    "earnings": 1.5,
    "regulatory": 1.2,
    "macro": 0.8,
    "corporate": 1.3,
    "general": 1.0,
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_MAX_POSITIONS = 5
DEFAULT_TTL_DAYS = 3
DEFAULT_COST_BPS = 30.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredHeadline:
    """A headline with its FinBERT sentiment score."""

    title: str
    source: str
    label: str         # positive / negative / neutral
    score: float       # -1 to +1
    confidence: float  # 0 to 1
    event_type: str
    weighted_score: float  # score * event_weight


@dataclass(frozen=True)
class IndiaTradeSignal:
    """Aggregated trade signal for a single stock."""

    symbol: str
    direction: str         # "long" or "short"
    avg_score: float       # average weighted sentiment score
    max_confidence: float  # highest confidence among headlines
    event_type: str        # dominant event type
    n_headlines: int
    headlines: list[str]   # the actual headline texts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_headlines(
    items: list[IndiaNewsItem],
    classifier: SentimentClassifier,
) -> list[tuple[IndiaNewsItem, ScoredHeadline]]:
    """Score a batch of news items with FinBERT.

    Returns list of (original_item, scored_headline) pairs.
    Only items that mention at least one F&O stock are scored.
    """
    # Filter to items with stock mentions
    stock_items = [item for item in items if item.stocks]
    if not stock_items:
        return []

    texts = [item.title for item in stock_items]
    results: list[SentimentResult] = classifier.classify(texts)

    scored: list[tuple[IndiaNewsItem, ScoredHeadline]] = []
    for item, sr in zip(stock_items, results):
        weight = EVENT_WEIGHTS.get(item.event_type, 1.0)
        scored.append((item, ScoredHeadline(
            title=sr.text,
            source=item.source,
            label=sr.label,
            score=sr.score,
            confidence=sr.confidence,
            event_type=item.event_type,
            weighted_score=sr.score * weight,
        )))

    return scored


def generate_signals(
    scored_items: list[tuple[IndiaNewsItem, ScoredHeadline]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> list[IndiaTradeSignal]:
    """Aggregate scored headlines per stock and generate trade signals.

    For each stock:
    1. Collect all scored headlines mentioning it
    2. Filter by confidence >= threshold
    3. Compute average weighted score
    4. If |avg_score| >= score_threshold, emit a signal

    Returns at most max_positions signals, sorted by |avg_score| descending.
    """
    # Group by stock symbol
    by_stock: dict[str, list[tuple[IndiaNewsItem, ScoredHeadline]]] = {}
    for item, sh in scored_items:
        for symbol in item.stocks:
            if symbol not in by_stock:
                by_stock[symbol] = []
            by_stock[symbol].append((item, sh))

    signals: list[IndiaTradeSignal] = []

    for symbol, pairs in by_stock.items():
        # Filter by confidence
        confident = [(item, sh) for item, sh in pairs if sh.confidence >= confidence_threshold]
        if not confident:
            continue

        # Aggregate
        scores = [sh.weighted_score for _, sh in confident]
        avg_score = sum(scores) / len(scores)
        max_conf = max(sh.confidence for _, sh in confident)

        # Dominant event type
        event_counts: dict[str, int] = {}
        for _, sh in confident:
            event_counts[sh.event_type] = event_counts.get(sh.event_type, 0) + 1
        dominant_event = max(event_counts, key=event_counts.get)  # type: ignore

        if abs(avg_score) < score_threshold:
            continue

        direction = "long" if avg_score > 0 else "short"

        signals.append(IndiaTradeSignal(
            symbol=symbol,
            direction=direction,
            avg_score=avg_score,
            max_confidence=max_conf,
            event_type=dominant_event,
            n_headlines=len(confident),
            headlines=[sh.title for _, sh in confident],
        ))

    # Sort by absolute score descending, take top N
    signals.sort(key=lambda s: abs(s.avg_score), reverse=True)
    return signals[:max_positions]
