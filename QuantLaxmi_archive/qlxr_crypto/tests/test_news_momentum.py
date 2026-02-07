"""Tests for the news momentum strategy components."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from apps.news_momentum.scraper import COIN_ALIASES, NewsItem, extract_coins
from apps.news_momentum.state import ActiveTrade, ClosedTrade, TradingState
from apps.news_momentum.strategy import (
    ScoredHeadline,
    StrategyConfig,
    TradeSignal,
    aggregate_by_coin,
    check_exits,
    generate_signals,
)


# ---------------------------------------------------------------------------
# Coin extraction tests
# ---------------------------------------------------------------------------


class TestExtractCoins(unittest.TestCase):
    def test_dollar_ticker(self):
        coins = extract_coins("$BTC is pumping today")
        self.assertIn("BTCUSDT", coins)

    def test_full_name(self):
        coins = extract_coins("Ethereum hits new milestone")
        self.assertIn("ETHUSDT", coins)

    def test_ticker_with_usd(self):
        coins = extract_coins("SOL/USD pair is volatile")
        self.assertIn("SOLUSDT", coins)

    def test_no_false_positives(self):
        coins = extract_coins("The SEC announced new rules today")
        self.assertEqual(coins, [])

    def test_source_currencies(self):
        coins = extract_coins("Price moves", source_currencies=["BTC", "ETH"])
        self.assertIn("BTCUSDT", coins)
        self.assertIn("ETHUSDT", coins)

    def test_multiple_coins(self):
        coins = extract_coins("$BTC and Ethereum both rallying")
        self.assertIn("BTCUSDT", coins)
        self.assertIn("ETHUSDT", coins)


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------


class TestStatePersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_file = Path(self.tmpdir) / "test_state.json"

    def test_save_load_roundtrip(self):
        state = TradingState()
        state.record_entry(
            "BTCUSDT", "long", 50000.0, 15.0, 0.8, 0.9, 2,
            ["headline 1", "headline 2"],
        )
        state.save(self.state_file)

        loaded = TradingState.load(self.state_file)
        self.assertEqual(len(loaded.active_trades), 1)
        self.assertIn("BTCUSDT", loaded.active_trades)
        self.assertEqual(loaded.active_trades["BTCUSDT"].entry_price, 50000.0)
        self.assertEqual(loaded.total_trades, 1)

    def test_load_nonexistent(self):
        state = TradingState.load(Path("/tmp/nonexistent_xyz.json"))
        self.assertEqual(len(state.active_trades), 0)
        self.assertEqual(state.total_trades, 0)

    def test_record_entry_and_exit(self):
        state = TradingState()
        state.record_entry("ETHUSDT", "short", 3000.0, 10.0, -0.7, 0.85, 1, ["bad news"])
        self.assertEqual(len(state.active_trades), 1)
        self.assertEqual(state.total_trades, 1)

        closed = state.record_exit("ETHUSDT", 2900.0, "ttl_expired")
        self.assertIsNotNone(closed)
        self.assertEqual(len(state.active_trades), 0)
        self.assertEqual(len(state.closed_trades), 1)
        # Short: (3000 - 2900) / 3000 * 100 = 3.33%
        self.assertAlmostEqual(closed.pnl_pct, 3.333, places=2)

    def test_long_pnl(self):
        state = TradingState()
        state.record_entry("SOLUSDT", "long", 100.0, 15.0, 0.6, 0.8, 1, ["sol news"])
        closed = state.record_exit("SOLUSDT", 105.0, "ttl_expired")
        self.assertAlmostEqual(closed.pnl_pct, 5.0, places=2)

    def test_win_rate(self):
        state = TradingState()
        # 2 winners, 1 loser
        state.record_entry("A", "long", 100.0, 15.0, 0.5, 0.7, 1, [])
        state.record_exit("A", 110.0, "ttl")
        state.record_entry("B", "long", 100.0, 15.0, 0.5, 0.7, 1, [])
        state.record_exit("B", 90.0, "ttl")
        state.record_entry("C", "short", 100.0, 15.0, -0.5, 0.7, 1, [])
        state.record_exit("C", 95.0, "ttl")

        self.assertAlmostEqual(state.win_rate(), 66.67, places=1)

    def test_exit_nonexistent(self):
        state = TradingState()
        closed = state.record_exit("FAKE", 100.0, "ttl")
        self.assertIsNone(closed)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


def _make_scored(title, score, confidence, coins, age_seconds=60):
    """Helper to create a ScoredHeadline."""
    from apps.news_momentum.sentiment import SentimentResult

    label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    now = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)

    return ScoredHeadline(
        news=NewsItem(
            title=title,
            source="test",
            url="",
            published_at=now,
            coins=coins,
            raw_currencies=[],
        ),
        sentiment=SentimentResult(
            text=title,
            label=label,
            score=score,
            confidence=confidence,
        ),
        coins=coins,
    )


class TestAggregation(unittest.TestCase):
    def test_single_coin(self):
        config = StrategyConfig()
        scored = [
            _make_scored("BTC up", 0.8, 0.9, ["BTCUSDT"]),
        ]
        agg = aggregate_by_coin(scored, config)
        self.assertIn("BTCUSDT", agg)
        self.assertAlmostEqual(agg["BTCUSDT"]["score"], 0.8)

    def test_multiple_headlines_averaged(self):
        config = StrategyConfig()
        scored = [
            _make_scored("BTC up", 0.8, 0.9, ["BTCUSDT"]),
            _make_scored("BTC rally", 0.6, 0.8, ["BTCUSDT"]),
        ]
        agg = aggregate_by_coin(scored, config)
        self.assertAlmostEqual(agg["BTCUSDT"]["score"], 0.7)
        self.assertEqual(agg["BTCUSDT"]["n_headlines"], 2)

    def test_old_news_filtered(self):
        config = StrategyConfig(max_news_age_seconds=300)
        scored = [
            _make_scored("old news", 0.9, 0.9, ["BTCUSDT"], age_seconds=600),
        ]
        agg = aggregate_by_coin(scored, config)
        self.assertEqual(len(agg), 0)

    def test_low_confidence_filtered(self):
        config = StrategyConfig(min_confidence=0.80)
        scored = [
            _make_scored("low conf", 0.9, 0.5, ["BTCUSDT"]),
        ]
        agg = aggregate_by_coin(scored, config)
        self.assertEqual(len(agg), 0)


class TestSignalGeneration(unittest.TestCase):
    def test_generates_long_signal(self):
        config = StrategyConfig(min_abs_score=0.5, min_headlines_per_coin=1)
        scored = [
            _make_scored("BTC pumping", 0.8, 0.9, ["BTCUSDT"]),
        ]
        signals = generate_signals(scored, {}, config)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, "long")

    def test_generates_short_signal(self):
        config = StrategyConfig(min_abs_score=0.5, min_headlines_per_coin=1)
        scored = [
            _make_scored("BTC crash", -0.8, 0.9, ["BTCUSDT"]),
        ]
        signals = generate_signals(scored, {}, config)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, "short")

    def test_skips_already_traded(self):
        from apps.news_momentum.strategy import ActiveTrade as SAT
        config = StrategyConfig(min_abs_score=0.5, min_headlines_per_coin=1)
        scored = [
            _make_scored("BTC up", 0.8, 0.9, ["BTCUSDT"]),
        ]
        active = {
            "BTCUSDT": SAT(
                symbol="BTCUSDT", direction="long",
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_price=50000.0, ttl_minutes=15, score=0.8,
                headlines=["old"],
            ),
        }
        signals = generate_signals(scored, active, config)
        self.assertEqual(len(signals), 0)

    def test_respects_max_positions(self):
        config = StrategyConfig(
            min_abs_score=0.3, min_headlines_per_coin=1, max_positions=1,
        )
        scored = [
            _make_scored("BTC up", 0.8, 0.9, ["BTCUSDT"]),
            _make_scored("ETH up", 0.7, 0.85, ["ETHUSDT"]),
        ]
        signals = generate_signals(scored, {}, config)
        self.assertEqual(len(signals), 1)

    def test_score_below_threshold_skipped(self):
        config = StrategyConfig(min_abs_score=0.8, min_headlines_per_coin=1)
        scored = [
            _make_scored("BTC meh", 0.3, 0.9, ["BTCUSDT"]),
        ]
        signals = generate_signals(scored, {}, config)
        self.assertEqual(len(signals), 0)


class TestExitCheck(unittest.TestCase):
    def test_expired_trade_exits(self):
        from apps.news_momentum.strategy import ActiveTrade as SAT
        old_time = (
            datetime.now(timezone.utc) - timedelta(minutes=20)
        ).isoformat()
        active = {
            "BTCUSDT": SAT(
                symbol="BTCUSDT", direction="long",
                entry_time=old_time,
                entry_price=50000.0, ttl_minutes=15, score=0.8,
                headlines=[],
            ),
        }
        exits = check_exits(active, {"BTCUSDT": 51000.0})
        self.assertIn("BTCUSDT", exits)

    def test_fresh_trade_stays(self):
        from apps.news_momentum.strategy import ActiveTrade as SAT
        now = datetime.now(timezone.utc).isoformat()
        active = {
            "BTCUSDT": SAT(
                symbol="BTCUSDT", direction="long",
                entry_time=now,
                entry_price=50000.0, ttl_minutes=15, score=0.8,
                headlines=[],
            ),
        }
        exits = check_exits(active, {"BTCUSDT": 51000.0})
        self.assertEqual(exits, [])


if __name__ == "__main__":
    unittest.main()
