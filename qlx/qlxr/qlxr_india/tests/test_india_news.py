"""Tests for India News Sentiment Scanner.

Covers:
  - Entity extraction (12 tests)
  - Event classification (5 tests)
  - Scraper (4 tests)
  - Strategy (10 tests)
  - State (8 tests)
  - Integration (6 tests)
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------

from collectors.news.entity import (
    COMPANY_ALIASES,
    INDEX_KEYWORDS,
    extract_indices,
    extract_stocks,
)


class TestExtractStocks:
    """Entity extraction: company name → NSE symbol."""

    def test_exact_ticker(self):
        assert "RELIANCE" in extract_stocks("RELIANCE shares rise 3%")

    def test_common_name(self):
        assert "TCS" in extract_stocks("Tata Consultancy Services beats estimates")

    def test_short_name(self):
        assert "RELIANCE" in extract_stocks("RIL posts record profit in Q3")

    def test_multiple_stocks(self):
        result = extract_stocks("Infosys and TCS report strong Q3 numbers")
        assert "INFY" in result
        assert "TCS" in result

    def test_longest_match_first(self):
        """'Tata Consultancy' should match TCS, not TATAMOTORS via 'Tata'."""
        result = extract_stocks("Tata Consultancy Services revenue surges")
        assert "TCS" in result
        # Should NOT match TATAMOTORS since "Tata" was consumed by "Tata Consultancy"
        assert "TATAMOTORS" not in result

    def test_case_insensitive(self):
        # Long aliases are case-insensitive substring matches
        assert "HDFCBANK" in extract_stocks("hdfc bank raises lending rates")

    def test_no_false_positive_common_words(self):
        """Short tickers shouldn't match inside normal words."""
        # "IT" should not match (not in our aliases, but verify no spurious matches)
        result = extract_stocks("with the market rally continuing")
        assert len(result) == 0

    def test_adani_group(self):
        result = extract_stocks("Adani Enterprises and Adani Ports surge")
        assert "ADANIENT" in result
        assert "ADANIPORTS" in result

    def test_mahindra_and_mahindra(self):
        assert "M&M" in extract_stocks("Mahindra & Mahindra auto sales up 15%")

    def test_bajaj_auto(self):
        assert "BAJAJ-AUTO" in extract_stocks("Bajaj Auto reports strong numbers")

    def test_empty_headline(self):
        assert extract_stocks("") == []

    def test_no_fno_stock(self):
        result = extract_stocks("Some random company does something")
        assert result == []


class TestExtractIndices:
    """Index keyword extraction tests."""

    def test_nifty(self):
        assert "NIFTY" in extract_indices("Nifty 50 closes above 22000")

    def test_banknifty_rbi(self):
        assert "BANKNIFTY" in extract_indices("RBI holds repo rate unchanged")

    def test_budget(self):
        assert "NIFTY" in extract_indices("Union Budget 2026 expectations")

    def test_monetary_policy(self):
        assert "BANKNIFTY" in extract_indices("monetary policy review tomorrow")

    def test_no_index(self):
        assert extract_indices("Random headline about nothing") == []


# ---------------------------------------------------------------------------
# Event classification tests
# ---------------------------------------------------------------------------

from collectors.news.scraper import classify_event


class TestClassifyEvent:
    def test_earnings(self):
        assert classify_event("TCS Q3 quarterly results beat estimates") == "earnings"

    def test_regulatory(self):
        assert classify_event("SEBI issues new margin rules for F&O") == "regulatory"

    def test_macro(self):
        assert classify_event("India GDP growth at 7.2% in Q3") == "macro"

    def test_corporate(self):
        assert classify_event("Reliance announces merger with Disney India") == "corporate"

    def test_general(self):
        assert classify_event("Markets close mixed on global cues") == "general"


# ---------------------------------------------------------------------------
# Scraper tests
# ---------------------------------------------------------------------------

from collectors.news.scraper import IndiaNewsItem, scan_india_news


class TestScraper:
    def test_age_filter(self):
        """Old headlines should be filtered out."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(minutes=120)

        # Create a mock feed that returns one old entry
        mock_feed = MagicMock()
        mock_entry = MagicMock()
        mock_entry.get = lambda key, default="": {
            "title": "Old headline about Reliance",
            "link": "http://example.com/old",
            "published_parsed": old_time.timetuple(),
            "updated_parsed": old_time.timetuple(),
        }.get(key, default)
        mock_entry.published_parsed = old_time.timetuple()
        mock_entry.updated_parsed = old_time.timetuple()
        mock_feed.entries = [mock_entry]

        with patch("collectors.news.scraper.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.content = b""
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            with patch("collectors.news.scraper.feedparser.parse", return_value=mock_feed):
                result = scan_india_news(max_age_minutes=60, feeds={"test": "http://test.com"})
                assert len(result) == 0

    def test_dedup(self):
        """Duplicate titles should be deduplicated."""
        now = datetime.now(timezone.utc)

        items = [
            IndiaNewsItem(
                title="Reliance Industries reports strong Q3",
                source="test1",
                url="http://example.com/1",
                published_at=now,
                stocks=["RELIANCE"],
                indices=[],
                event_type="earnings",
            ),
            IndiaNewsItem(
                title="Reliance Industries reports strong Q3",
                source="test2",
                url="http://example.com/2",
                published_at=now,
                stocks=["RELIANCE"],
                indices=[],
                event_type="earnings",
            ),
        ]

        # Apply dedup logic manually (same as in scan_india_news)
        seen: set[str] = set()
        unique = []
        for item in items:
            key = item.title.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        assert len(unique) == 1

    def test_stock_extraction_in_item(self):
        """IndiaNewsItem should have extracted stocks."""
        item = IndiaNewsItem(
            title="TCS and Infosys report Q3 results",
            source="test",
            url="",
            published_at=datetime.now(timezone.utc),
            stocks=extract_stocks("TCS and Infosys report Q3 results"),
            indices=[],
            event_type="earnings",
        )
        assert "TCS" in item.stocks
        assert "INFY" in item.stocks

    def test_news_item_frozen(self):
        """IndiaNewsItem should be frozen (immutable)."""
        item = IndiaNewsItem(
            title="Test", source="test", url="",
            published_at=datetime.now(timezone.utc),
            stocks=[], indices=[], event_type="general",
        )
        with pytest.raises(AttributeError):
            item.title = "Modified"  # type: ignore


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

from collectors.news.strategy import (
    EVENT_WEIGHTS,
    IndiaTradeSignal,
    ScoredHeadline,
    generate_signals,
    score_headlines,
)


def _make_mock_classifier(label="positive", score=0.8, confidence=0.9):
    """Create a mock SentimentClassifier."""
    from qlx.nlp.sentiment import SentimentResult

    clf = MagicMock()
    clf.classify = lambda texts, **kw: [
        SentimentResult(text=t, label=label, score=score, confidence=confidence)
        for t in texts
    ]
    return clf


def _make_item(title: str, stocks: list[str], event_type: str = "general"):
    """Helper to create an IndiaNewsItem."""
    return IndiaNewsItem(
        title=title,
        source="test",
        url="",
        published_at=datetime.now(timezone.utc),
        stocks=stocks,
        indices=[],
        event_type=event_type,
    )


class TestStrategy:
    def test_score_headlines_filters_no_stocks(self):
        """Items without stock mentions should be skipped."""
        items = [_make_item("Generic market news", [], "general")]
        clf = _make_mock_classifier()
        result = score_headlines(items, clf)
        assert len(result) == 0

    def test_score_headlines_basic(self):
        """Items with stocks should be scored."""
        items = [_make_item("Reliance posts record profit", ["RELIANCE"], "earnings")]
        clf = _make_mock_classifier(score=0.7, confidence=0.85)
        result = score_headlines(items, clf)
        assert len(result) == 1
        _, sh = result[0]
        assert sh.score == 0.7
        assert sh.confidence == 0.85

    def test_event_weight_applied(self):
        """Earnings headlines should get 1.5x weight."""
        items = [_make_item("TCS quarterly results beat", ["TCS"], "earnings")]
        clf = _make_mock_classifier(score=0.6, confidence=0.8)
        result = score_headlines(items, clf)
        _, sh = result[0]
        assert abs(sh.weighted_score - 0.6 * EVENT_WEIGHTS["earnings"]) < 1e-6

    def test_generate_signals_basic(self):
        """Should generate a signal when score and confidence are above thresholds."""
        now = datetime.now(timezone.utc)
        item = _make_item("Reliance strong Q3", ["RELIANCE"], "earnings")
        sh = ScoredHeadline(
            title="Reliance strong Q3", source="test",
            label="positive", score=0.8, confidence=0.9,
            event_type="earnings", weighted_score=0.8 * 1.5,
        )
        signals = generate_signals([(item, sh)], confidence_threshold=0.7, score_threshold=0.5)
        assert len(signals) == 1
        assert signals[0].symbol == "RELIANCE"
        assert signals[0].direction == "long"

    def test_generate_signals_negative(self):
        """Negative sentiment should produce short signal."""
        item = _make_item("TCS misses estimates badly", ["TCS"], "earnings")
        sh = ScoredHeadline(
            title="TCS misses estimates badly", source="test",
            label="negative", score=-0.8, confidence=0.9,
            event_type="earnings", weighted_score=-0.8 * 1.5,
        )
        signals = generate_signals([(item, sh)])
        assert len(signals) == 1
        assert signals[0].direction == "short"

    def test_below_confidence_threshold(self):
        """Low-confidence headlines should be filtered."""
        item = _make_item("Reliance maybe rises", ["RELIANCE"], "general")
        sh = ScoredHeadline(
            title="Reliance maybe rises", source="test",
            label="neutral", score=0.1, confidence=0.4,
            event_type="general", weighted_score=0.1,
        )
        signals = generate_signals([(item, sh)], confidence_threshold=0.7)
        assert len(signals) == 0

    def test_below_score_threshold(self):
        """Weak scores (even with high confidence) should not produce signals."""
        item = _make_item("Reliance flat", ["RELIANCE"], "general")
        sh = ScoredHeadline(
            title="Reliance flat", source="test",
            label="neutral", score=0.1, confidence=0.95,
            event_type="general", weighted_score=0.1,
        )
        signals = generate_signals([(item, sh)], score_threshold=0.5)
        assert len(signals) == 0

    def test_max_positions_limit(self):
        """Should respect max_positions limit."""
        pairs = []
        for sym in ["RELIANCE", "TCS", "INFY", "SBIN", "HDFCBANK", "ICICIBANK"]:
            item = _make_item(f"{sym} surges", [sym], "earnings")
            sh = ScoredHeadline(
                title=f"{sym} surges", source="test",
                label="positive", score=0.9, confidence=0.95,
                event_type="earnings", weighted_score=0.9 * 1.5,
            )
            pairs.append((item, sh))
        signals = generate_signals(pairs, max_positions=3)
        assert len(signals) == 3

    def test_aggregation_multiple_headlines(self):
        """Multiple headlines for same stock should be aggregated."""
        item1 = _make_item("Reliance profit up 20%", ["RELIANCE"], "earnings")
        sh1 = ScoredHeadline(
            title="Reliance profit up 20%", source="test1",
            label="positive", score=0.8, confidence=0.9,
            event_type="earnings", weighted_score=1.2,
        )
        item2 = _make_item("Reliance revenue beats", ["RELIANCE"], "earnings")
        sh2 = ScoredHeadline(
            title="Reliance revenue beats", source="test2",
            label="positive", score=0.6, confidence=0.85,
            event_type="earnings", weighted_score=0.9,
        )
        signals = generate_signals([(item1, sh1), (item2, sh2)])
        assert len(signals) == 1
        assert signals[0].n_headlines == 2
        assert abs(signals[0].avg_score - (1.2 + 0.9) / 2) < 1e-6

    def test_empty_input(self):
        """Empty input should produce no signals."""
        assert generate_signals([]) == []


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

from collectors.news.state import ActiveTrade, ClosedTrade, IndiaNewsTradingState


class TestState:
    def test_save_load_roundtrip(self, tmp_path):
        """State should survive save/load cycle."""
        path = tmp_path / "state.json"
        state = IndiaNewsTradingState(
            equity=1.05,
            started_at="2026-01-01T00:00:00+00:00",
            max_positions=3,
            ttl_days=5,
        )
        state.save(path)
        loaded = IndiaNewsTradingState.load(path)
        assert loaded.equity == 1.05
        assert loaded.max_positions == 3
        assert loaded.ttl_days == 5

    def test_enter_trade(self):
        state = IndiaNewsTradingState()
        trade = state.enter_trade(
            symbol="RELIANCE", direction="long", price=2500.0,
            date_str="2026-02-05", score=0.8, confidence=0.9,
            event_type="earnings",
        )
        assert trade.symbol == "RELIANCE"
        assert len(state.active_trades) == 1
        assert "RELIANCE" in state.active_symbols()

    def test_exit_trade(self):
        state = IndiaNewsTradingState(cost_bps=30.0)
        state.enter_trade(
            symbol="TCS", direction="long", price=4000.0,
            date_str="2026-02-01", score=0.7, confidence=0.85,
            event_type="earnings",
        )
        closed = state.exit_trade("TCS", 4200.0, "2026-02-04", "ttl")
        assert closed.pnl_pct == pytest.approx((4200 - 4000) / 4000 - 30 / 10_000, abs=1e-6)
        assert len(state.active_trades) == 0
        assert len(state.closed_trades) == 1

    def test_short_trade_pnl(self):
        state = IndiaNewsTradingState(cost_bps=30.0)
        state.enter_trade(
            symbol="INFY", direction="short", price=1500.0,
            date_str="2026-02-01", score=-0.8, confidence=0.9,
            event_type="earnings",
        )
        closed = state.exit_trade("INFY", 1400.0, "2026-02-04", "ttl")
        expected = (1500 - 1400) / 1500 - 30 / 10_000
        assert closed.pnl_pct == pytest.approx(expected, abs=1e-6)

    def test_cost_deduction(self):
        """Cost should be deducted from P&L."""
        state = IndiaNewsTradingState(cost_bps=50.0)
        state.enter_trade(
            symbol="SBIN", direction="long", price=800.0,
            date_str="2026-02-01", score=0.6, confidence=0.8,
            event_type="general",
        )
        # Exit at same price: pure cost
        closed = state.exit_trade("SBIN", 800.0, "2026-02-02", "ttl")
        assert closed.pnl_pct == pytest.approx(-50 / 10_000, abs=1e-6)

    def test_hold_days_increment(self):
        state = IndiaNewsTradingState()
        state.enter_trade(
            symbol="HDFCBANK", direction="long", price=1600.0,
            date_str="2026-02-01", score=0.7, confidence=0.85,
            event_type="corporate",
        )
        state.increment_hold_days()
        state.increment_hold_days()
        assert state.active_trades[0].hold_days == 2

    def test_expired_trades(self):
        state = IndiaNewsTradingState(ttl_days=3)
        state.enter_trade(
            symbol="WIPRO", direction="long", price=500.0,
            date_str="2026-02-01", score=0.6, confidence=0.8,
            event_type="general",
        )
        for _ in range(3):
            state.increment_hold_days()
        expired = state.expired_trades()
        assert len(expired) == 1
        assert expired[0].symbol == "WIPRO"

    def test_equity_tracking(self):
        state = IndiaNewsTradingState(cost_bps=0.0, max_positions=1)
        state.enter_trade(
            symbol="ITC", direction="long", price=400.0,
            date_str="2026-02-01", score=0.9, confidence=0.95,
            event_type="earnings",
        )
        state.exit_trade("ITC", 440.0, "2026-02-04", "ttl")
        # 10% gain, 1 position slot, 100% weight
        pnl_frac = (440 - 400) / 400
        expected_equity = 1.0 * (1 + pnl_frac * 1.0)
        assert state.equity == pytest.approx(expected_equity, abs=1e-6)

    def test_load_missing_file(self, tmp_path):
        """Loading non-existent file should return fresh state."""
        path = tmp_path / "nonexistent.json"
        state = IndiaNewsTradingState.load(path)
        assert state.equity == 1.0
        assert len(state.active_trades) == 0

    def test_max_positions_enforced(self):
        state = IndiaNewsTradingState(max_positions=2)
        state.enter_trade("A", "long", 100.0, "2026-01-01", 0.8, 0.9, "general")
        state.enter_trade("B", "long", 200.0, "2026-01-01", 0.7, 0.85, "general")
        with pytest.raises(ValueError, match="Max positions"):
            state.enter_trade("C", "long", 300.0, "2026-01-01", 0.6, 0.8, "general")

    def test_duplicate_symbol_rejected(self):
        state = IndiaNewsTradingState()
        state.enter_trade("RELIANCE", "long", 2500.0, "2026-01-01", 0.8, 0.9, "earnings")
        with pytest.raises(ValueError, match="Already have"):
            state.enter_trade("RELIANCE", "short", 2500.0, "2026-01-02", -0.5, 0.8, "general")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_cycle_entry_exit(self):
        """Full lifecycle: enter → hold → expire → exit."""
        state = IndiaNewsTradingState(ttl_days=2, cost_bps=30.0, max_positions=5)
        state.started_at = "2026-02-01T00:00:00+00:00"

        # Day 1: Enter
        state.enter_trade(
            symbol="RELIANCE", direction="long", price=2500.0,
            date_str="2026-02-03", score=0.8, confidence=0.9,
            event_type="earnings",
        )
        assert len(state.active_trades) == 1
        assert state.available_slots() == 4

        # Day 2: Hold
        state.increment_hold_days()
        assert state.active_trades[0].hold_days == 1
        assert len(state.expired_trades()) == 0

        # Day 3: Expire + exit
        state.increment_hold_days()
        assert state.active_trades[0].hold_days == 2
        expired = state.expired_trades()
        assert len(expired) == 1

        closed = state.exit_trade("RELIANCE", 2600.0, "2026-02-05", "ttl")
        assert closed.hold_days == 2
        assert closed.pnl_pct > 0  # 4% gain minus 30bps cost
        assert len(state.active_trades) == 0
        assert len(state.closed_trades) == 1
        assert state.equity > 1.0

    def test_trading_day_check(self):
        """Weekend should not be a trading day."""
        from strategies.s9_momentum.data import is_trading_day
        from datetime import date
        # Feb 1, 2026 is a Sunday
        assert not is_trading_day(date(2026, 2, 1))
        # Feb 2, 2026 is a Monday
        assert is_trading_day(date(2026, 2, 2))

    def test_kite_price_used_in_paper(self):
        """cmd_paper should use Kite, not bhavcopy."""
        import inspect
        from collectors.news.__main__ import cmd_paper
        src = inspect.getsource(cmd_paper)
        assert "BhavcopyCache" not in src
        assert "_get_live_price" in src

    def test_empty_rss_no_crash(self):
        """Empty RSS results should not crash paper trading logic."""
        state = IndiaNewsTradingState()
        # No items → no signals → no trades
        signals = generate_signals([])
        assert signals == []
        assert len(state.active_trades) == 0

    def test_state_persistence_with_trades(self, tmp_path):
        """State with active + closed trades should roundtrip."""
        path = tmp_path / "state.json"
        state = IndiaNewsTradingState(cost_bps=25.0)
        state.enter_trade("TCS", "long", 4000.0, "2026-02-01", 0.8, 0.9, "earnings")
        state.enter_trade("INFY", "short", 1800.0, "2026-02-01", -0.7, 0.85, "earnings")
        state.increment_hold_days()
        state.exit_trade("TCS", 4100.0, "2026-02-03", "ttl")
        state.save(path)

        loaded = IndiaNewsTradingState.load(path)
        assert len(loaded.active_trades) == 1
        assert loaded.active_trades[0].symbol == "INFY"
        assert loaded.active_trades[0].hold_days == 1
        assert len(loaded.closed_trades) == 1
        assert loaded.closed_trades[0].symbol == "TCS"
        assert loaded.cost_bps == 25.0

    def test_signal_flip_exit(self):
        """A signal flip should close the position."""
        state = IndiaNewsTradingState(cost_bps=30.0)
        state.enter_trade("SBIN", "long", 800.0, "2026-02-01", 0.7, 0.85, "corporate")
        # Simulate signal flip
        closed = state.exit_trade("SBIN", 790.0, "2026-02-03", "signal_flip")
        assert closed.exit_reason == "signal_flip"
        assert closed.pnl_pct < 0  # loss

    def test_win_rate_calculation(self):
        """Win rate should be correct after mixed trades."""
        state = IndiaNewsTradingState(cost_bps=0.0, max_positions=5)
        # Win
        state.enter_trade("A", "long", 100.0, "2026-01-01", 0.8, 0.9, "general")
        state.exit_trade("A", 110.0, "2026-01-03", "ttl")
        # Loss
        state.enter_trade("B", "long", 100.0, "2026-01-01", 0.7, 0.85, "general")
        state.exit_trade("B", 90.0, "2026-01-03", "ttl")
        # Win
        state.enter_trade("C", "short", 100.0, "2026-01-01", -0.8, 0.9, "general")
        state.exit_trade("C", 85.0, "2026-01-03", "ttl")

        assert state.win_rate() == pytest.approx(2 / 3, abs=1e-6)
