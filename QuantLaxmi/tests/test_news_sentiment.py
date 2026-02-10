"""Tests for news sentiment pipeline: GDELT collector, feature builder, causality."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# GDELT collector tests
# ---------------------------------------------------------------------------


class TestGDELT:
    """Test GDELT API interaction and data parsing."""

    def test_gdelt_datetime_format(self):
        from quantlaxmi.data.collectors.news.gdelt import _gdelt_datetime

        dt = datetime(2026, 2, 10, 9, 15, 30, tzinfo=timezone.utc)
        assert _gdelt_datetime(dt) == "20260210091530"

    def test_parse_gdelt_date(self):
        from quantlaxmi.data.collectors.news.gdelt import _parse_gdelt_date

        dt = _parse_gdelt_date("20260210T083000Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 10
        assert dt.hour == 8
        assert dt.minute == 30

    def test_parse_gdelt_date_invalid(self):
        from quantlaxmi.data.collectors.news.gdelt import _parse_gdelt_date

        assert _parse_gdelt_date("invalid") is None
        assert _parse_gdelt_date("") is None
        assert _parse_gdelt_date(None) is None

    def test_articles_to_archive_records(self):
        from quantlaxmi.data.collectors.news.gdelt import _articles_to_archive_records

        articles = [
            {
                "title": "NIFTY hits all-time high on FII buying",
                "url": "https://example.com/1",
                "seendate": "20260210T083000Z",
                "domain": "economictimes.com",
                "language": "English",
                "sourcecountry": "India",
            },
            {
                "title": "",  # empty title, should be filtered
                "url": "https://example.com/2",
                "seendate": "20260210T090000Z",
                "domain": "test.com",
            },
        ]
        records = _articles_to_archive_records(articles, "india_market")
        assert len(records) == 1
        assert records[0]["title"] == "NIFTY hits all-time high on FII buying"
        assert "gdelt_india_market" in records[0]["source"]
        assert records[0]["event_type"] in ("macro", "general")  # "FII" triggers macro

    def test_normalize_title(self):
        from quantlaxmi.data.collectors.news.gdelt import _normalize_title

        assert _normalize_title("  Hello, World!  ") == "hello world"
        assert _normalize_title("NIFTY +2%") == "nifty 2"

    def test_query_categories_complete(self):
        """Verify all 7 categories have query lists."""
        from quantlaxmi.data.collectors.news.gdelt import (
            INDIA_MARKET_QUERIES, INDIA_STOCK_QUERIES, CRYPTO_QUERIES,
            US_MARKET_QUERIES, US_STOCK_QUERIES, EUROPE_MARKET_QUERIES,
            INTL_QUERIES,
        )
        assert len(INDIA_MARKET_QUERIES) >= 3
        assert len(INDIA_STOCK_QUERIES) >= 3
        assert len(CRYPTO_QUERIES) >= 3
        assert len(US_MARKET_QUERIES) >= 3
        assert len(US_STOCK_QUERIES) >= 2
        assert len(EUROPE_MARKET_QUERIES) >= 3
        assert len(INTL_QUERIES) >= 4


# ---------------------------------------------------------------------------
# RSS scraper feed config tests
# ---------------------------------------------------------------------------


class TestRSSFeeds:
    """Verify RSS feed configuration."""

    def test_india_feeds_defined(self):
        from quantlaxmi.data.collectors.news.scraper import INDIA_RSS_FEEDS
        assert len(INDIA_RSS_FEEDS) >= 8

    def test_crypto_feeds_defined(self):
        from quantlaxmi.data.collectors.news.scraper import CRYPTO_RSS_FEEDS
        assert len(CRYPTO_RSS_FEEDS) >= 4

    def test_us_feeds_defined(self):
        from quantlaxmi.data.collectors.news.scraper import US_RSS_FEEDS
        assert len(US_RSS_FEEDS) >= 5

    def test_europe_feeds_defined(self):
        from quantlaxmi.data.collectors.news.scraper import EUROPE_RSS_FEEDS
        assert len(EUROPE_RSS_FEEDS) >= 4

    def test_intl_feeds_defined(self):
        from quantlaxmi.data.collectors.news.scraper import INTL_RSS_FEEDS
        assert len(INTL_RSS_FEEDS) >= 3

    def test_all_feeds_union(self):
        from quantlaxmi.data.collectors.news.scraper import (
            ALL_RSS_FEEDS, INDIA_RSS_FEEDS, CRYPTO_RSS_FEEDS,
            US_RSS_FEEDS, EUROPE_RSS_FEEDS, INTL_RSS_FEEDS,
        )
        expected = len(INDIA_RSS_FEEDS) + len(CRYPTO_RSS_FEEDS) + len(US_RSS_FEEDS) \
            + len(EUROPE_RSS_FEEDS) + len(INTL_RSS_FEEDS)
        assert len(ALL_RSS_FEEDS) == expected

    def test_all_feed_urls_are_strings(self):
        from quantlaxmi.data.collectors.news.scraper import ALL_RSS_FEEDS
        for name, url in ALL_RSS_FEEDS.items():
            assert isinstance(name, str)
            assert isinstance(url, str)
            assert url.startswith("http")


# ---------------------------------------------------------------------------
# NewsSentimentBuilder tests
# ---------------------------------------------------------------------------


def _make_archive(tmpdir: Path, headlines: list[dict]) -> None:
    """Write test headlines to a JSONL archive."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, list[dict]] = {}
    for hl in headlines:
        ts = datetime.fromisoformat(hl["ts"])
        key = ts.strftime("%Y-%m")
        if key not in by_month:
            by_month[key] = []
        by_month[key].append(hl)

    for month, hls in by_month.items():
        path = tmpdir / f"{month}.jsonl"
        with open(path, "w") as f:
            for hl in hls:
                f.write(json.dumps(hl) + "\n")


def _sample_headlines(n_days: int = 30, per_day: int = 5) -> list[dict]:
    """Generate synthetic headlines for testing."""
    headlines = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    titles_pos = [
        "Markets rally as investors cheer strong earnings",
        "NIFTY surges to new highs on global optimism",
        "FII inflows boost Indian equities market",
        "Tech stocks lead gains in broad market rally",
        "GDP growth exceeds expectations, rupee strengthens",
    ]
    titles_neg = [
        "Markets tumble on recession fears and rate hike concerns",
        "NIFTY falls sharply as FII selling accelerates",
        "Oil prices surge, raising inflation worries",
        "Global trade tensions weigh on emerging markets",
        "Corporate earnings disappoint, dragging indices lower",
    ]
    all_titles = titles_pos + titles_neg

    for day in range(n_days):
        dt = base + timedelta(days=day)
        for j in range(per_day):
            title = all_titles[(day * per_day + j) % len(all_titles)]
            headlines.append({
                "ts": (dt + timedelta(hours=j * 2)).isoformat(),
                "title": f"{title} — day {day} item {j}",
                "source": "test_source",
                "url": f"https://test.com/{day}/{j}",
                "stocks": ["RELIANCE"] if j % 3 == 0 else [],
                "indices": ["NIFTY"] if j % 2 == 0 else [],
                "event_type": "macro" if j % 4 == 0 else "general",
            })
    return headlines


class TestNewsSentimentBuilder:
    """Test the feature builder."""

    def test_build_returns_dataframe(self, tmp_path):
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        headlines = _sample_headlines(n_days=30, per_day=5)
        archive_dir = tmp_path / "headlines"
        _make_archive(archive_dir, headlines)

        builder = NewsSentimentBuilder(archive_dir=archive_dir, use_finbert=False)
        df = builder.build("2026-01-02", "2026-01-25")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "ns_sent_mean" in df.columns
        assert "ns_news_count" in df.columns
        assert "ns_sent_5d_ma" in df.columns
        assert "ns_sent_momentum" in df.columns

    def test_feature_count(self, tmp_path):
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        headlines = _sample_headlines(n_days=30, per_day=5)
        archive_dir = tmp_path / "headlines"
        _make_archive(archive_dir, headlines)

        builder = NewsSentimentBuilder(archive_dir=archive_dir, use_finbert=False)
        df = builder.build("2026-01-02", "2026-01-25")

        # Should have exactly 11 features
        assert len(df.columns) == 11

    def test_causality_t_plus_1_lag(self, tmp_path):
        """Headlines from day D must contribute to day D+1 features (T+1 lag)."""
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        # Single headline on Jan 5
        headlines = [{
            "ts": "2026-01-05T10:00:00+00:00",
            "title": "Test headline for causality check",
            "source": "test",
            "url": "https://test.com/1",
            "stocks": [],
            "indices": [],
            "event_type": "general",
        }]
        archive_dir = tmp_path / "headlines"
        _make_archive(archive_dir, headlines)

        builder = NewsSentimentBuilder(archive_dir=archive_dir, use_finbert=False, lag_days=1)
        df = builder.build("2026-01-05", "2026-01-07")

        # Headline from Jan 5 should appear in Jan 6 features (T+1 lag)
        if not df.empty:
            assert pd.Timestamp("2026-01-06") in df.index
            # Jan 5 should NOT have any features (no headlines from Jan 4)
            assert pd.Timestamp("2026-01-05") not in df.index

    def test_empty_archive(self, tmp_path):
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        archive_dir = tmp_path / "empty_headlines"
        archive_dir.mkdir(parents=True)

        builder = NewsSentimentBuilder(archive_dir=archive_dir, use_finbert=False)
        df = builder.build("2026-01-01", "2026-01-31")
        assert df.empty

    def test_no_archive_dir(self, tmp_path):
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        builder = NewsSentimentBuilder(archive_dir=tmp_path / "nonexistent", use_finbert=False)
        df = builder.build("2026-01-01", "2026-01-31")
        assert df.empty

    def test_pos_neg_ratio_bounded(self, tmp_path):
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        headlines = _sample_headlines(n_days=20, per_day=10)
        archive_dir = tmp_path / "headlines"
        _make_archive(archive_dir, headlines)

        builder = NewsSentimentBuilder(archive_dir=archive_dir, use_finbert=False)
        df = builder.build("2026-01-02", "2026-01-18")

        if not df.empty:
            assert df["ns_pos_ratio"].between(0, 1).all()
            assert df["ns_neg_ratio"].between(0, 1).all()
            assert df["ns_news_count"].ge(0).all()

    def test_rolling_features_are_causal(self, tmp_path):
        """Rolling features (5d MA) must not look ahead."""
        from quantlaxmi.features.news_sentiment import NewsSentimentBuilder

        headlines = _sample_headlines(n_days=30, per_day=3)
        archive_dir = tmp_path / "headlines"
        _make_archive(archive_dir, headlines)

        builder = NewsSentimentBuilder(archive_dir=archive_dir, use_finbert=False)
        df = builder.build("2026-01-02", "2026-01-25")

        if len(df) >= 6:
            # The 5d MA on day 6 should equal the mean of days 2-6
            sent_vals = df["ns_sent_mean"].iloc[:5].values
            expected_ma = np.mean(sent_vals)
            actual_ma = df["ns_sent_5d_ma"].iloc[4]
            np.testing.assert_allclose(actual_ma, expected_ma, rtol=1e-10)


# ---------------------------------------------------------------------------
# Score cache tests
# ---------------------------------------------------------------------------


class TestScoreCache:

    def test_cache_put_and_get(self, tmp_path):
        from quantlaxmi.features.news_sentiment import _ScoreCache

        cache = _ScoreCache(cache_path=tmp_path / "cache.jsonl")
        cache.put("Hello world", 0.8, 0.95, "positive")

        result = cache.get("Hello world")
        assert result is not None
        assert result["score"] == 0.8
        assert result["label"] == "positive"

    def test_cache_persistence(self, tmp_path):
        from quantlaxmi.features.news_sentiment import _ScoreCache

        path = tmp_path / "cache.jsonl"
        cache1 = _ScoreCache(cache_path=path)
        cache1.put("Test headline", 0.5, 0.8, "neutral")

        # Reload from disk
        cache2 = _ScoreCache(cache_path=path)
        result = cache2.get("Test headline")
        assert result is not None
        assert result["score"] == 0.5

    def test_cache_miss(self, tmp_path):
        from quantlaxmi.features.news_sentiment import _ScoreCache

        cache = _ScoreCache(cache_path=tmp_path / "cache.jsonl")
        assert cache.get("nonexistent") is None

    def test_cache_has(self, tmp_path):
        from quantlaxmi.features.news_sentiment import _ScoreCache

        cache = _ScoreCache(cache_path=tmp_path / "cache.jsonl")
        assert not cache.has("test")
        cache.put("test", 0.1, 0.2, "negative")
        assert cache.has("test")


# ---------------------------------------------------------------------------
# MegaFeatureBuilder integration
# ---------------------------------------------------------------------------


class TestMegaIntegration:
    """Verify news_sentiment is wired into MegaFeatureBuilder."""

    def test_builder_list_includes_news_sentiment(self):
        """The builders list in MegaFeatureBuilder.build() must include news_sentiment."""
        import inspect
        from quantlaxmi.features.mega import MegaFeatureBuilder

        source = inspect.getsource(MegaFeatureBuilder.build)
        assert "news_sentiment" in source

    def test_build_method_exists(self):
        from quantlaxmi.features.mega import MegaFeatureBuilder

        builder = MegaFeatureBuilder.__new__(MegaFeatureBuilder)
        assert hasattr(builder, "_build_news_sentiment_features")
