"""NewsSentimentBuilder — daily sentiment features from news headlines.

Reads the headline archive (JSONL), runs FinBERT scoring (with on-disk
caching), and aggregates per-date features for the TFT model.

Features (prefix ``ns_``):
    ns_sent_mean       — mean FinBERT score across all headlines that day
    ns_sent_std        — std of scores (disagreement/uncertainty)
    ns_sent_max        — most bullish headline score
    ns_sent_min        — most bearish headline score
    ns_pos_ratio       — fraction of positive headlines
    ns_neg_ratio       — fraction of negative headlines
    ns_confidence_mean — mean FinBERT confidence
    ns_news_count      — number of headlines (attention proxy)
    ns_sent_5d_ma      — 5-day rolling mean of ns_sent_mean
    ns_news_count_5d   — 5-day rolling sum of headline count
    ns_sent_momentum   — ns_sent_mean − ns_sent_5d_ma

Causality
---------
Headlines published on day D contribute to day D+1's features (T+1 lag).
This ensures zero look-ahead bias even if some headlines arrive after
market close on day D.

Score caching
-------------
FinBERT scores are cached in ``data/india/headlines/scores_cache.jsonl``.
Each line: {"title_hash": "...", "score": float, "confidence": float, "label": str}
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantlaxmi.data._paths import HEADLINE_ARCHIVE_DIR

logger = logging.getLogger(__name__)

SCORE_CACHE_FILE = HEADLINE_ARCHIVE_DIR / "scores_cache.jsonl"


def _title_hash(title: str) -> str:
    """Deterministic hash of a headline title for cache lookup."""
    return hashlib.sha256(title.strip().lower().encode()).hexdigest()[:16]


class _ScoreCache:
    """On-disk cache for FinBERT sentiment scores to avoid re-scoring."""

    def __init__(self, cache_path: Path = SCORE_CACHE_FILE):
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._cache[obj["title_hash"]] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("Score cache loaded: %d entries", len(self._cache))

    def get(self, title: str) -> dict | None:
        h = _title_hash(title)
        return self._cache.get(h)

    def put(self, title: str, score: float, confidence: float, label: str) -> None:
        h = _title_hash(title)
        entry = {
            "title_hash": h,
            "score": score,
            "confidence": confidence,
            "label": label,
        }
        self._cache[h] = entry
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def __len__(self) -> int:
        return len(self._cache)

    def has(self, title: str) -> bool:
        return _title_hash(title) in self._cache


def score_headlines_bulk(
    headlines: list[dict],
    cache: _ScoreCache | None = None,
    batch_size: int = 64,
) -> list[dict]:
    """Score headlines with FinBERT, using cache to skip already-scored ones.

    Parameters
    ----------
    headlines : list[dict]
        Raw JSONL records with at least "title" key.
    cache : _ScoreCache | None
        Score cache. If None, a new one is created.
    batch_size : int
        FinBERT batch size.

    Returns
    -------
    list[dict]
        Headlines with added "sentiment_score", "sentiment_confidence",
        "sentiment_label" keys.
    """
    if cache is None:
        cache = _ScoreCache()

    # Separate cached vs uncached
    uncached_titles = []
    uncached_indices = []
    results = []

    for i, hl in enumerate(headlines):
        title = hl.get("title", "")
        cached = cache.get(title)
        if cached:
            results.append({
                **hl,
                "sentiment_score": cached["score"],
                "sentiment_confidence": cached["confidence"],
                "sentiment_label": cached["label"],
            })
        else:
            uncached_titles.append(title)
            uncached_indices.append(i)
            results.append(hl)  # placeholder

    # Score uncached with FinBERT
    if uncached_titles:
        logger.info("Scoring %d uncached headlines with FinBERT...", len(uncached_titles))
        from quantlaxmi.core.nlp.sentiment import get_classifier
        classifier = get_classifier()
        scored = classifier.classify(uncached_titles, batch_size=batch_size)

        for idx, sr in zip(uncached_indices, scored):
            hl = headlines[idx]
            cache.put(sr.text, sr.score, sr.confidence, sr.label)
            results[idx] = {
                **hl,
                "sentiment_score": sr.score,
                "sentiment_confidence": sr.confidence,
                "sentiment_label": sr.label,
            }
        logger.info("FinBERT scoring complete, cache now has %d entries", len(cache))

    return results


class NewsSentimentBuilder:
    """Build daily sentiment features from the headline archive.

    Usage
    -----
        builder = NewsSentimentBuilder()
        df = builder.build("2025-01-01", "2026-02-10")
        # df has DatetimeIndex, columns: ns_sent_mean, ns_sent_std, ...
    """

    def __init__(
        self,
        archive_dir: Path | None = None,
        use_finbert: bool = True,
        lag_days: int = 1,
    ):
        """
        Parameters
        ----------
        archive_dir : Path | None
            Override headline archive directory.
        use_finbert : bool
            If True, run FinBERT on uncached headlines.
            If False, use GDELT tone scores (if available) or skip scoring.
        lag_days : int
            Causality lag. Headlines from day D contribute to day D+lag_days.
            Default 1 (T+1) ensures zero look-ahead.
        """
        self.archive_dir = archive_dir or HEADLINE_ARCHIVE_DIR
        self.use_finbert = use_finbert
        self.lag_days = lag_days

    def _load_headlines(self, start_date: str, end_date: str) -> list[dict]:
        """Load headlines from archive, filtering by date range.

        Reads directly from self.archive_dir (supports custom paths for testing).
        """
        # We need headlines from earlier dates because of the lag
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

        # Fetch a buffer before start for lag + rolling windows
        fetch_start = start_dt - timedelta(days=self.lag_days + 30)
        fetch_end = end_dt

        if not self.archive_dir.exists():
            return []

        results: list[dict] = []
        for fpath in sorted(self.archive_dir.glob("*.jsonl")):
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    try:
                        ts = datetime.fromisoformat(obj["ts"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < fetch_start or ts > fetch_end:
                            continue
                    except (KeyError, ValueError):
                        continue

                    results.append(obj)
        return results

    def build(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Build daily sentiment features.

        Parameters
        ----------
        start_date, end_date : str
            Date range "YYYY-MM-DD" (inclusive).

        Returns
        -------
        pd.DataFrame
            Indexed by date, columns prefixed with ``ns_``.
            Returns empty DataFrame if no headlines available.
        """
        headlines = self._load_headlines(start_date, end_date)
        if not headlines:
            logger.warning("No headlines found in archive for %s to %s", start_date, end_date)
            return pd.DataFrame()

        # Score with FinBERT (cached)
        if self.use_finbert:
            cache = _ScoreCache()
            scored = score_headlines_bulk(headlines, cache=cache)
        else:
            # Use raw headlines without sentiment scores — only count-based features
            scored = [{**hl, "sentiment_score": 0.0, "sentiment_confidence": 0.0,
                       "sentiment_label": "neutral"} for hl in headlines]

        # Parse dates and build per-day aggregates
        daily: dict[str, list[dict]] = {}
        for hl in scored:
            try:
                ts = datetime.fromisoformat(hl["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                # Apply causality lag: headlines from day D → features for day D+lag
                feature_date = (ts + timedelta(days=self.lag_days)).strftime("%Y-%m-%d")
                if feature_date not in daily:
                    daily[feature_date] = []
                daily[feature_date].append(hl)
            except (KeyError, ValueError):
                continue

        if not daily:
            return pd.DataFrame()

        # Aggregate features per day
        rows = []
        for date_str in sorted(daily.keys()):
            hls = daily[date_str]
            scores = [h["sentiment_score"] for h in hls]
            confs = [h["sentiment_confidence"] for h in hls]
            labels = [h["sentiment_label"] for h in hls]

            n = len(scores)
            scores_arr = np.array(scores)
            confs_arr = np.array(confs)

            n_pos = sum(1 for l in labels if l == "positive")
            n_neg = sum(1 for l in labels if l == "negative")

            rows.append({
                "date": pd.Timestamp(date_str),
                "ns_sent_mean": float(np.mean(scores_arr)),
                "ns_sent_std": float(np.std(scores_arr, ddof=1)) if n > 1 else 0.0,
                "ns_sent_max": float(np.max(scores_arr)),
                "ns_sent_min": float(np.min(scores_arr)),
                "ns_pos_ratio": n_pos / n if n > 0 else 0.0,
                "ns_neg_ratio": n_neg / n if n > 0 else 0.0,
                "ns_confidence_mean": float(np.mean(confs_arr)),
                "ns_news_count": float(n),
            })

        df = pd.DataFrame(rows).set_index("date").sort_index()

        # Rolling features (causal — only past data)
        df["ns_sent_5d_ma"] = df["ns_sent_mean"].rolling(5, min_periods=1).mean()
        df["ns_news_count_5d"] = df["ns_news_count"].rolling(5, min_periods=1).sum()
        df["ns_sent_momentum"] = df["ns_sent_mean"] - df["ns_sent_5d_ma"]

        # Filter to requested date range
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        df = df.loc[mask]

        logger.info(
            "NewsSentimentBuilder: %d days, %d features, %d total headlines scored",
            len(df), len(df.columns), sum(len(v) for v in daily.values()),
        )
        return df
