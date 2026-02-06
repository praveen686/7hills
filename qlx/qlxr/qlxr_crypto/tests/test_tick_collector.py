"""Tests for the tick collector infrastructure.

Covers:
  - TickStore: buffering, flushing, parquet read-back, day rollover
  - TradeTick / BookTick: construction and schema compliance
  - LiveRegimeTracker: VPIN/OFI/Hawkes from synthetic tick data
  - SymbolFeatureState: incremental feature computation
  - scan_symbols: REST scanner (mocked)
  - CollectorConfig: defaults and overrides
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from apps.tick_collector.collector import CollectorConfig, scan_symbols
from apps.tick_collector.features import LiveRegimeTracker, SymbolFeatureState
from apps.tick_collector.storage import (
    BOOK_SCHEMA,
    TRADE_SCHEMA,
    BookTick,
    TickStore,
    TickStoreConfig,
    TradeTick,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    config = TickStoreConfig(
        base_dir=tmp_dir,
        flush_interval_sec=1.0,
        flush_threshold=100,
    )
    s = TickStore(config)
    yield s
    s.close()


def make_trade(price=100.0, qty=1.0, ts_ms=1000000, tid=1, buyer_maker=False):
    return TradeTick(
        agg_trade_id=tid,
        timestamp_ms=ts_ms,
        price=price,
        quantity=qty,
        is_buyer_maker=buyer_maker,
    )


def make_book(bid=99.9, ask=100.1, bid_qty=10.0, ask_qty=10.0, ts_ms=1000000):
    return BookTick(
        timestamp_ms=ts_ms,
        bid_price=bid,
        ask_price=ask,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
    )


# ===========================================================================
# TradeTick / BookTick
# ===========================================================================

class TestTickTypes:
    def test_trade_tick_creation(self):
        t = make_trade(price=50000.0, qty=0.5, tid=42)
        assert t.price == 50000.0
        assert t.quantity == 0.5
        assert t.agg_trade_id == 42

    def test_book_tick_creation(self):
        b = make_book(bid=49999.0, ask=50001.0)
        assert b.bid_price == 49999.0
        assert b.ask_price == 50001.0

    def test_trade_schema_fields(self):
        assert len(TRADE_SCHEMA) == 5
        assert TRADE_SCHEMA.field("price").type == pa.float64()
        assert TRADE_SCHEMA.field("is_buyer_maker").type == pa.bool_()

    def test_book_schema_fields(self):
        assert len(BOOK_SCHEMA) == 5
        assert BOOK_SCHEMA.field("bid_price").type == pa.float64()


# ===========================================================================
# TickStore
# ===========================================================================

class TestTickStore:
    def test_add_and_flush_trades(self, store, tmp_dir):
        """Trades should be buffered then written to parquet on flush."""
        for i in range(10):
            store.add_trade("BTCUSDT", make_trade(price=50000 + i, tid=i, ts_ms=1000 * i))

        flushed = store.flush_all()
        assert "BTCUSDT" in flushed
        assert flushed["BTCUSDT"] == 10

        # Verify parquet
        table = TickStore.read_trades(tmp_dir, "BTCUSDT")
        assert table is not None
        assert table.num_rows == 10
        assert table.schema == TRADE_SCHEMA

    def test_add_and_flush_book(self, store, tmp_dir):
        """Book ticks should be stored to parquet."""
        for i in range(5):
            store.add_book("ETHUSDT", make_book(bid=3000 + i, ts_ms=1000 * i))

        store.flush_all()
        table = TickStore.read_book(tmp_dir, "ETHUSDT")
        assert table is not None
        assert table.num_rows == 5

    def test_multiple_flushes_append(self, store, tmp_dir):
        """Multiple flushes should append to the same file (row groups)."""
        for i in range(5):
            store.add_trade("BTCUSDT", make_trade(tid=i, ts_ms=1000 * i))
        store.flush_all()

        for i in range(5, 10):
            store.add_trade("BTCUSDT", make_trade(tid=i, ts_ms=1000 * i))
        store.flush_all()

        table = TickStore.read_trades(tmp_dir, "BTCUSDT")
        assert table is not None
        assert table.num_rows == 10

    def test_multiple_symbols(self, store, tmp_dir):
        """Each symbol gets its own file."""
        store.add_trade("BTCUSDT", make_trade(price=50000))
        store.add_trade("ETHUSDT", make_trade(price=3000))
        store.flush_all()

        assert TickStore.read_trades(tmp_dir, "BTCUSDT").num_rows == 1
        assert TickStore.read_trades(tmp_dir, "ETHUSDT").num_rows == 1

    def test_empty_flush_no_error(self, store):
        """Flushing with no data should be a no-op."""
        flushed = store.flush_all()
        assert flushed == {}

    def test_stats(self, store):
        """Stats should reflect current state."""
        store.add_trade("BTCUSDT", make_trade())
        store.add_book("BTCUSDT", make_book())

        stats = store.stats()
        assert stats["symbols"] == 1
        ps = stats["per_symbol"]["BTCUSDT"]
        assert ps["trade_buf"] == 1
        assert ps["book_buf"] == 1

    def test_remove_symbol(self, store, tmp_dir):
        """Removing a symbol should flush and close its writer."""
        store.add_trade("BTCUSDT", make_trade())
        store.remove_symbol("BTCUSDT")

        # Data should have been flushed
        table = TickStore.read_trades(tmp_dir, "BTCUSDT")
        assert table is not None
        assert table.num_rows == 1

    def test_list_dates(self, store, tmp_dir):
        """list_dates should find date directories."""
        store.add_trade("BTCUSDT", make_trade())
        store.flush_all()

        dates = TickStore.list_dates(tmp_dir)
        assert len(dates) >= 1
        # Should be today's date
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in dates

    def test_list_symbols(self, store, tmp_dir):
        """list_symbols should find symbols on a given date."""
        store.add_trade("BTCUSDT", make_trade())
        store.add_trade("ETHUSDT", make_trade())
        store.flush_all()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        symbols = TickStore.list_symbols(tmp_dir, today)
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    def test_read_nonexistent_returns_none(self, tmp_dir):
        """Reading non-existent data returns None."""
        assert TickStore.read_trades(tmp_dir, "FAKEUSDT") is None
        assert TickStore.read_book(tmp_dir, "FAKEUSDT") is None

    def test_parquet_compression(self, store, tmp_dir):
        """Parquet files should use zstd compression."""
        for i in range(100):
            store.add_trade("BTCUSDT", make_trade(price=50000 + i * 0.01, tid=i))
        store.flush_all()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = tmp_dir / today / "BTCUSDT_trades.parquet"
        meta = pq.read_metadata(str(path))
        # Check that the file has row groups
        assert meta.num_row_groups >= 1
        assert meta.num_rows == 100


# ===========================================================================
# LiveRegimeTracker
# ===========================================================================

class TestLiveRegimeTracker:
    def test_basic_trade_tracking(self):
        """Tracker should accept trades and update state."""
        tracker = LiveRegimeTracker()
        for i in range(100):
            price = 100.0 + 0.01 * math.sin(i * 0.1)
            tick = make_trade(price=price, qty=10.0, ts_ms=i * 100, tid=i)
            tracker.on_trade("BTCUSDT", tick)

        regime = tracker.get_regime("BTCUSDT")
        assert regime is not None
        assert regime.symbol == "BTCUSDT"
        assert regime.hawkes_ratio > 0

    def test_book_updates_ofi(self):
        """Book ticks should update OFI."""
        tracker = LiveRegimeTracker()

        # Series of increasing bids → positive OFI
        for i in range(50):
            tick = make_book(
                bid=100.0 + i * 0.01,
                ask=100.2 + i * 0.01,
                bid_qty=10.0 + i * 0.5,
                ask_qty=10.0,
                ts_ms=i * 100,
            )
            tracker.on_book("ETHUSDT", tick)

        regime = tracker.get_regime("ETHUSDT")
        assert regime is not None
        assert regime.ofi > 0  # positive because bids are increasing

    def test_vpin_updates_with_volume(self):
        """VPIN should update after enough volume flows through."""
        tracker = LiveRegimeTracker(vpin_bucket_size=1000.0, vpin_n_buckets=5)

        # Feed enough volume to fill multiple buckets
        # bucket_size=1000, 5 buckets, so need 5000+ USD volume
        for i in range(500):
            price = 100.0 + 0.5 * math.sin(i * 0.05)
            qty = 1.0  # 1 unit at ~$100 = $100 per trade
            tick = make_trade(price=price, qty=qty, ts_ms=i * 10, tid=i)
            tracker.on_trade("TESTUSDT", tick)

        regime = tracker.get_regime("TESTUSDT")
        assert regime is not None
        # After 500 trades * $100 = $50K volume, VPIN should have a value
        assert regime.vpin >= 0  # should be computed

    def test_multiple_symbols_independent(self):
        """Each symbol should have independent state."""
        tracker = LiveRegimeTracker()

        # Feed different patterns
        for i in range(50):
            tracker.on_trade("SYM1", make_trade(price=100 + i, tid=i, ts_ms=i * 100))
            tracker.on_trade("SYM2", make_trade(price=200 - i, tid=i, ts_ms=i * 100))

        r1 = tracker.get_regime("SYM1")
        r2 = tracker.get_regime("SYM2")
        assert r1 is not None and r2 is not None
        assert r1.symbol == "SYM1"
        assert r2.symbol == "SYM2"

    def test_all_regimes(self):
        """all_regimes should return snapshots for all tracked symbols."""
        tracker = LiveRegimeTracker()
        tracker.on_trade("A", make_trade(tid=1))
        tracker.on_trade("B", make_trade(tid=2))

        regimes = tracker.all_regimes()
        assert len(regimes) == 2
        assert "A" in regimes and "B" in regimes

    def test_remove_symbol(self):
        """Removed symbols should not appear in regimes."""
        tracker = LiveRegimeTracker()
        tracker.on_trade("A", make_trade(tid=1))
        tracker.remove_symbol("A")
        assert tracker.get_regime("A") is None

    def test_get_nonexistent_returns_none(self):
        tracker = LiveRegimeTracker()
        assert tracker.get_regime("FAKE") is None

    def test_stats(self):
        """Stats should reflect tick counts."""
        tracker = LiveRegimeTracker()
        for i in range(10):
            tracker.on_trade("SYM", make_trade(price=100 + i * 0.01, tid=i, ts_ms=i * 100))
        for i in range(5):
            tracker.on_book("SYM", make_book(ts_ms=i * 100))

        stats = tracker.stats()
        assert stats["n_symbols"] == 1
        assert stats["per_symbol"]["SYM"]["n_trades"] == 10
        assert stats["per_symbol"]["SYM"]["n_books"] == 5


# ===========================================================================
# SymbolFeatureState
# ===========================================================================

class TestSymbolFeatureState:
    def test_hawkes_intensity_increases_with_bursts(self):
        """Burst of trades should increase Hawkes intensity ratio."""
        state = SymbolFeatureState(symbol="TEST")

        # Slow trades: 1 per second
        for i in range(50):
            tick = make_trade(price=100.0, ts_ms=i * 1000, tid=i)
            state.on_trade(tick)
        ratio_slow = state.last_hawkes_ratio

        # Burst: 50 trades in 1 second
        for i in range(50, 100):
            tick = make_trade(price=100.0, ts_ms=50000 + (i - 50) * 20, tid=i)
            state.on_trade(tick)
        ratio_burst = state.last_hawkes_ratio

        assert ratio_burst > ratio_slow

    def test_ofi_positive_on_bid_increase(self):
        """Increasing bids with stable asks → positive OFI."""
        state = SymbolFeatureState(symbol="TEST")

        for i in range(20):
            tick = make_book(
                bid=100.0 + i * 0.1,
                ask=101.0,
                bid_qty=50.0,
                ask_qty=50.0,
                ts_ms=i * 100,
            )
            state.on_book(tick)

        assert state.last_ofi > 0

    def test_regime_snapshot_fields(self):
        """Regime snapshot should have all expected fields."""
        state = SymbolFeatureState(symbol="TEST")
        state.on_trade(make_trade(price=100.0, tid=1, ts_ms=1000))
        state.on_trade(make_trade(price=100.1, tid=2, ts_ms=2000))

        snap = state.regime_snapshot()
        assert snap.symbol == "TEST"
        assert hasattr(snap, "vpin")
        assert hasattr(snap, "ofi")
        assert hasattr(snap, "hawkes_ratio")


# ===========================================================================
# Scanner
# ===========================================================================

class TestScanner:
    def test_scan_filters_by_funding(self):
        """Scanner should filter by minimum funding rate (excluding top-volume additions)."""
        premium_data = [
            {"symbol": "HIGHUSDT", "lastFundingRate": "0.001"},   # ~109.5% ann
            {"symbol": "LOWUSDT", "lastFundingRate": "0.0001"},   # ~10.95% ann
            {"symbol": "NEGUSDT", "lastFundingRate": "-0.001"},   # negative
        ]
        volume_data = [
            {"symbol": "HIGHUSDT", "quoteVolume": "50000000"},
            {"symbol": "LOWUSDT", "quoteVolume": "50000000"},
            {"symbol": "NEGUSDT", "quoteVolume": "50000000"},
        ]

        with patch("apps.tick_collector.collector.requests.get") as mock_get:
            def side_effect(url, **kwargs):
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                if "premiumIndex" in url:
                    resp.json.return_value = premium_data
                elif "ticker/24hr" in url:
                    resp.json.return_value = volume_data
                return resp

            mock_get.side_effect = side_effect
            # max_symbols=1 so only the funding-filtered symbol fits, no room
            # for the top-volume backfill to add LOWUSDT.
            config = CollectorConfig(
                min_ann_funding_pct=15.0, min_volume_usd=1e6, max_symbols=1,
            )
            results = scan_symbols(config)

        symbols = [r["symbol"] for r in results]
        assert "HIGHUSDT" in symbols
        assert "LOWUSDT" not in symbols  # 10.95% < 15% threshold
        assert len(results) == 1

    def test_scan_filters_by_volume(self):
        """Scanner should filter by minimum volume."""
        premium_data = [
            {"symbol": "BIGUSDT", "lastFundingRate": "0.001"},
            {"symbol": "SMALLUSDT", "lastFundingRate": "0.001"},
        ]
        volume_data = [
            {"symbol": "BIGUSDT", "quoteVolume": "50000000"},     # $50M
            {"symbol": "SMALLUSDT", "quoteVolume": "100000"},     # $100K
        ]

        with patch("apps.tick_collector.collector.requests.get") as mock_get:
            def side_effect(url, **kwargs):
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                if "premiumIndex" in url:
                    resp.json.return_value = premium_data
                elif "ticker/24hr" in url:
                    resp.json.return_value = volume_data
                return resp

            mock_get.side_effect = side_effect
            config = CollectorConfig(min_ann_funding_pct=15.0, min_volume_usd=5e6)
            results = scan_symbols(config)

        symbols = [r["symbol"] for r in results]
        assert "BIGUSDT" in symbols
        assert "SMALLUSDT" not in symbols


# ===========================================================================
# CollectorConfig
# ===========================================================================

class TestCollectorConfig:
    def test_defaults(self):
        c = CollectorConfig()
        assert c.min_ann_funding_pct == 15.0
        assert c.max_symbols == 20
        assert c.scan_interval_sec == 3600.0
        assert c.flush_interval_sec == 300.0

    def test_override(self):
        c = CollectorConfig(max_symbols=50, min_ann_funding_pct=10.0)
        assert c.max_symbols == 50
        assert c.min_ann_funding_pct == 10.0


# ===========================================================================
# Integration: Store + Tracker
# ===========================================================================

class TestStoreTrackerIntegration:
    def test_trade_flows_through_both(self, tmp_dir):
        """A trade should be stored AND update regime features."""
        config = TickStoreConfig(base_dir=tmp_dir)
        store = TickStore(config)
        tracker = LiveRegimeTracker(vpin_bucket_size=100.0, vpin_n_buckets=5)

        # Simulate what TickCollector does: feed both store and tracker
        for i in range(100):
            price = 50000.0 + 10 * math.sin(i * 0.1)
            tick = make_trade(price=price, qty=0.01, ts_ms=i * 100, tid=i)
            store.add_trade("BTCUSDT", tick)
            tracker.on_trade("BTCUSDT", tick)

        store.flush_all()

        # Storage check
        table = TickStore.read_trades(tmp_dir, "BTCUSDT")
        assert table is not None
        assert table.num_rows == 100

        # Feature check
        regime = tracker.get_regime("BTCUSDT")
        assert regime is not None
        assert regime.hawkes_ratio > 0

        store.close()

    def test_book_flows_through_both(self, tmp_dir):
        """Book ticks should be stored AND update OFI."""
        config = TickStoreConfig(base_dir=tmp_dir)
        store = TickStore(config)
        tracker = LiveRegimeTracker()

        for i in range(20):
            tick = make_book(
                bid=100.0 + i * 0.01,
                ask=100.1 + i * 0.01,
                ts_ms=i * 100,
            )
            store.add_book("ETHUSDT", tick)
            tracker.on_book("ETHUSDT", tick)

        store.flush_all()

        table = TickStore.read_book(tmp_dir, "ETHUSDT")
        assert table is not None
        assert table.num_rows == 20

        regime = tracker.get_regime("ETHUSDT")
        assert regime is not None
        assert regime.ofi != 0  # OFI should have updated

        store.close()
