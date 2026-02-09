"""Tests for Kite 5-level depth collector (futures + options).

Covers:
- DepthTick.from_kite_tick() with full and partial depth
- DepthTick option metadata (strike, expiry, option_type)
- DepthStore buffer → flush → read-back cycle
- DepthStore date rollover
- resolve_futures_tokens() with mock instrument DataFrame
- resolve_option_tokens() with mock instruments
- should_reroll() before/after expiry
- check_recenter_needed() at various spot levels
- Schema column count and types (41 columns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.collectors.kite_depth.storage import (
    DEPTH_SCHEMA,
    DepthStore,
    DepthStoreConfig,
    DepthTick,
)
from data.collectors.kite_depth.tokens import (
    FuturesToken,
    OptionToken,
    check_recenter_needed,
    resolve_futures_tokens,
    resolve_option_tokens,
    should_reroll,
)

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockTick:
    """Minimal mock of KiteTick for testing."""
    instrument_token: int = 12345
    last_price: float = 23500.0
    timestamp: datetime | None = None
    volume: int = 1000000
    oi: int = 50000
    buy_qty: int = 200000
    sell_qty: int = 180000
    depth: dict = field(default_factory=dict)


def _full_depth() -> dict:
    """Generate a realistic 5-level depth dict."""
    return {
        "buy": [
            {"price": 23500.0, "quantity": 100, "orders": 5},
            {"price": 23499.5, "quantity": 200, "orders": 8},
            {"price": 23499.0, "quantity": 150, "orders": 3},
            {"price": 23498.5, "quantity": 300, "orders": 12},
            {"price": 23498.0, "quantity": 250, "orders": 7},
        ],
        "sell": [
            {"price": 23500.5, "quantity": 120, "orders": 6},
            {"price": 23501.0, "quantity": 180, "orders": 9},
            {"price": 23501.5, "quantity": 160, "orders": 4},
            {"price": 23502.0, "quantity": 280, "orders": 11},
            {"price": 23502.5, "quantity": 220, "orders": 8},
        ],
    }


def _partial_depth() -> dict:
    """Depth with only 2 buy levels and 3 sell levels."""
    return {
        "buy": [
            {"price": 23500.0, "quantity": 100, "orders": 5},
            {"price": 23499.5, "quantity": 200, "orders": 8},
        ],
        "sell": [
            {"price": 23500.5, "quantity": 120, "orders": 6},
            {"price": 23501.0, "quantity": 180, "orders": 9},
            {"price": 23501.5, "quantity": 160, "orders": 4},
        ],
    }


def _mock_nfo_instruments() -> pd.DataFrame:
    """Create a mock NFO instruments DataFrame with futures + options."""
    today = datetime.now(IST).date()
    rows = [
        # Futures
        {"instrument_token": 11111, "name": "NIFTY", "instrument_type": "FUT",
         "tradingsymbol": "NIFTY26FEBFUT", "expiry": today + timedelta(days=20),
         "strike": 0.0},
        {"instrument_token": 11112, "name": "NIFTY", "instrument_type": "FUT",
         "tradingsymbol": "NIFTY26MARFUT", "expiry": today + timedelta(days=50),
         "strike": 0.0},
        {"instrument_token": 22221, "name": "BANKNIFTY", "instrument_type": "FUT",
         "tradingsymbol": "BANKNIFTY26FEBFUT", "expiry": today + timedelta(days=20),
         "strike": 0.0},
    ]

    # NIFTY options: strikes from 23000 to 24000 in 50-pt steps, 2 expiries
    for exp_offset, exp_label in [(7, "W1"), (14, "W2")]:
        exp = today + timedelta(days=exp_offset)
        for strike in range(23000, 24050, 50):
            for otype in ["CE", "PE"]:
                rows.append({
                    "instrument_token": hash(f"NIFTY_{strike}_{otype}_{exp}") % 10_000_000,
                    "name": "NIFTY",
                    "instrument_type": otype,
                    "tradingsymbol": f"NIFTY{exp_label}{strike}{otype}",
                    "expiry": exp,
                    "strike": float(strike),
                })

    # BANKNIFTY options: strikes from 49000 to 51000 in 100-pt steps
    for exp_offset in [7, 14]:
        exp = today + timedelta(days=exp_offset)
        for strike in range(49000, 51100, 100):
            for otype in ["CE", "PE"]:
                rows.append({
                    "instrument_token": hash(f"BN_{strike}_{otype}_{exp}") % 10_000_000,
                    "name": "BANKNIFTY",
                    "instrument_type": otype,
                    "tradingsymbol": f"BANKNIFTY{strike}{otype}",
                    "expiry": exp,
                    "strike": float(strike),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_column_count(self):
        assert len(DEPTH_SCHEMA) == 41

    def test_new_columns_present(self):
        names = DEPTH_SCHEMA.names
        assert "strike" in names
        assert "expiry" in names
        assert "option_type" in names

    def test_column_types(self):
        import pyarrow as pa
        schema = DEPTH_SCHEMA
        assert schema.field("timestamp_ms").type == pa.int64()
        assert schema.field("instrument_token").type == pa.int32()
        assert schema.field("symbol").type == pa.string()
        assert schema.field("strike").type == pa.float64()
        assert schema.field("expiry").type == pa.string()
        assert schema.field("option_type").type == pa.string()
        assert schema.field("bid_price_1").type == pa.float64()
        assert schema.field("bid_qty_1").type == pa.int64()
        assert schema.field("bid_orders_1").type == pa.int32()

    def test_all_depth_columns(self):
        names = DEPTH_SCHEMA.names
        for i in range(1, 6):
            for prefix in ("bid", "ask"):
                assert f"{prefix}_price_{i}" in names
                assert f"{prefix}_qty_{i}" in names
                assert f"{prefix}_orders_{i}" in names


# ---------------------------------------------------------------------------
# DepthTick tests
# ---------------------------------------------------------------------------

class TestDepthTick:
    def test_from_kite_tick_full_depth(self):
        ts = datetime(2026, 2, 6, 10, 30, 0, tzinfo=IST)
        tick = _MockTick(
            instrument_token=12345,
            last_price=23500.0,
            timestamp=ts,
            volume=1000000,
            oi=50000,
            buy_qty=200000,
            sell_qty=180000,
            depth=_full_depth(),
        )

        dt = DepthTick.from_kite_tick(tick, "NIFTY_FUT")

        assert dt.symbol == "NIFTY_FUT"
        assert dt.instrument_token == 12345
        assert dt.last_price == 23500.0
        assert dt.volume == 1000000
        assert dt.oi == 50000
        assert dt.timestamp_ms == int(ts.timestamp() * 1000)
        # Default futures metadata
        assert dt.strike == 0.0
        assert dt.expiry == ""
        assert dt.option_type == "FUT"

        # Bid/ask levels
        assert dt.bid_price_1 == 23500.0
        assert dt.bid_qty_1 == 100
        assert dt.ask_price_5 == 23502.5

    def test_from_kite_tick_with_option_metadata(self):
        ts = datetime(2026, 2, 6, 10, 30, 0, tzinfo=IST)
        tick = _MockTick(timestamp=ts, depth=_full_depth())

        dt = DepthTick.from_kite_tick(
            tick, "NIFTY_OPT",
            strike=23500.0,
            expiry="2026-02-13",
            option_type="CE",
        )

        assert dt.symbol == "NIFTY_OPT"
        assert dt.strike == 23500.0
        assert dt.expiry == "2026-02-13"
        assert dt.option_type == "CE"

    def test_from_kite_tick_partial_depth(self):
        tick = _MockTick(
            timestamp=datetime(2026, 2, 6, 10, 30, tzinfo=IST),
            depth=_partial_depth(),
        )

        dt = DepthTick.from_kite_tick(tick, "BANKNIFTY_FUT")

        # Bid: 2 levels, rest zeros
        assert dt.bid_price_2 == 23499.5
        assert dt.bid_price_3 == 0.0
        assert dt.bid_qty_3 == 0

        # Ask: 3 levels, rest zeros
        assert dt.ask_price_3 == 23501.5
        assert dt.ask_price_4 == 0.0

    def test_from_kite_tick_no_timestamp(self):
        tick = _MockTick(timestamp=None, depth=_full_depth())
        dt = DepthTick.from_kite_tick(tick, "NIFTY_FUT")
        assert dt.timestamp_ms > 0

    def test_from_kite_tick_empty_depth(self):
        tick = _MockTick(timestamp=datetime(2026, 2, 6, 10, 30, tzinfo=IST), depth={})
        dt = DepthTick.from_kite_tick(tick, "NIFTY_FUT")
        for i in range(1, 6):
            assert getattr(dt, f"bid_price_{i}") == 0.0
            assert getattr(dt, f"ask_price_{i}") == 0.0


# ---------------------------------------------------------------------------
# DepthStore tests
# ---------------------------------------------------------------------------

class TestDepthStore:
    def test_buffer_flush_readback(self, tmp_path):
        config = DepthStoreConfig(base_dir=tmp_path, flush_interval_sec=60.0)
        store = DepthStore(config)

        ts = datetime(2026, 2, 6, 10, 30, tzinfo=IST)
        tick = _MockTick(timestamp=ts, depth=_full_depth())

        with patch.object(DepthStore, "_current_date_str", return_value="2026-02-06"):
            for _ in range(10):
                dt = DepthTick.from_kite_tick(tick, "NIFTY_FUT")
                store.add_tick("NIFTY_FUT", dt)

            flushed = store.flush_all()
        assert flushed["NIFTY_FUT"] == 10

        table = DepthStore.read_depth(tmp_path, "NIFTY_FUT", "2026-02-06")
        assert table is not None
        assert len(table) == 10
        assert len(table.schema) == 41

        store.close()

    def test_options_and_futures_separate_files(self, tmp_path):
        config = DepthStoreConfig(base_dir=tmp_path)
        store = DepthStore(config)

        ts = datetime(2026, 2, 6, 10, 30, tzinfo=IST)
        tick = _MockTick(timestamp=ts, depth=_full_depth())

        with patch.object(DepthStore, "_current_date_str", return_value="2026-02-06"):
            # Futures tick
            dt_fut = DepthTick.from_kite_tick(tick, "NIFTY_FUT")
            store.add_tick("NIFTY_FUT", dt_fut)

            # Options tick
            dt_opt = DepthTick.from_kite_tick(
                tick, "NIFTY_OPT",
                strike=23500.0, expiry="2026-02-13", option_type="CE",
            )
            store.add_tick("NIFTY_OPT", dt_opt)

            store.flush_all()

        symbols = DepthStore.list_symbols(tmp_path, "2026-02-06")
        assert sorted(symbols) == ["NIFTY_FUT", "NIFTY_OPT"]

        # Verify option metadata persists
        opt_table = DepthStore.read_depth(tmp_path, "NIFTY_OPT", "2026-02-06")
        df = opt_table.to_pandas()
        assert df["strike"].iloc[0] == 23500.0
        assert df["option_type"].iloc[0] == "CE"
        assert df["expiry"].iloc[0] == "2026-02-13"

        store.close()

    def test_multiple_flushes_append(self, tmp_path):
        config = DepthStoreConfig(base_dir=tmp_path)
        store = DepthStore(config)

        ts = datetime(2026, 2, 6, 10, 30, tzinfo=IST)
        tick = _MockTick(timestamp=ts, depth=_full_depth())

        with patch.object(DepthStore, "_current_date_str", return_value="2026-02-06"):
            for _ in range(5):
                store.add_tick("NIFTY_FUT", DepthTick.from_kite_tick(tick, "NIFTY_FUT"))
            store.flush_all()

            for _ in range(3):
                store.add_tick("NIFTY_FUT", DepthTick.from_kite_tick(tick, "NIFTY_FUT"))
            store.flush_all()

        table = DepthStore.read_depth(tmp_path, "NIFTY_FUT", "2026-02-06")
        assert len(table) == 8

        store.close()

    def test_date_rollover(self, tmp_path):
        config = DepthStoreConfig(base_dir=tmp_path)
        store = DepthStore(config)

        ts = datetime(2026, 2, 6, 10, 30, tzinfo=IST)
        tick = _MockTick(timestamp=ts, depth=_full_depth())

        with patch.object(DepthStore, "_current_date_str", return_value="2026-02-06"):
            store.add_tick("NIFTY_FUT", DepthTick.from_kite_tick(tick, "NIFTY_FUT"))
            store.flush_all()

        with patch.object(DepthStore, "_current_date_str", return_value="2026-02-07"):
            store.add_tick("NIFTY_FUT", DepthTick.from_kite_tick(tick, "NIFTY_FUT"))
            store.flush_all()

        dates = DepthStore.list_dates(tmp_path)
        assert "2026-02-06" in dates
        assert "2026-02-07" in dates

        store.close()

    def test_stats(self, tmp_path):
        config = DepthStoreConfig(base_dir=tmp_path)
        store = DepthStore(config)

        ts = datetime(2026, 2, 6, 10, 30, tzinfo=IST)
        tick = _MockTick(timestamp=ts, depth=_full_depth())

        store.add_tick("NIFTY_FUT", DepthTick.from_kite_tick(tick, "NIFTY_FUT"))
        store.add_tick("NIFTY_OPT", DepthTick.from_kite_tick(
            tick, "NIFTY_OPT", strike=23500.0, expiry="2026-02-13", option_type="CE",
        ))
        store.flush_all()

        stats = store.stats()
        assert stats["total_stored"] == 2
        assert "NIFTY_FUT" in stats["per_symbol"]
        assert "NIFTY_OPT" in stats["per_symbol"]

        store.close()

    def test_empty_helpers(self, tmp_path):
        assert DepthStore.list_dates(tmp_path) == []
        assert DepthStore.list_symbols(tmp_path, "2026-02-06") == []
        assert DepthStore.read_depth(tmp_path, "NIFTY_FUT", "2099-01-01") is None


# ---------------------------------------------------------------------------
# Token resolution tests
# ---------------------------------------------------------------------------

class TestFuturesTokens:
    def test_resolve_futures_tokens(self):
        kite = MagicMock()
        kite.instruments.return_value = _mock_nfo_instruments().to_dict("records")

        tokens = resolve_futures_tokens(kite, ["NIFTY", "BANKNIFTY"])

        assert len(tokens) == 2
        assert 11111 in tokens
        assert 22221 in tokens
        assert tokens[11111].storage_key == "NIFTY_FUT"
        assert tokens[22221].storage_key == "BANKNIFTY_FUT"

    def test_resolve_picks_nearest_expiry(self):
        kite = MagicMock()
        kite.instruments.return_value = _mock_nfo_instruments().to_dict("records")

        tokens = resolve_futures_tokens(kite, ["NIFTY"])
        assert tokens[11111].tradingsymbol == "NIFTY26FEBFUT"

    def test_should_reroll_before_expiry(self):
        today = date(2026, 2, 6)
        tokens = {11111: FuturesToken("NIFTY", 11111, "X", date(2026, 2, 27), "NIFTY_FUT")}
        assert should_reroll(tokens, today) is False

    def test_should_reroll_after_expiry(self):
        today = date(2026, 2, 28)
        tokens = {11111: FuturesToken("NIFTY", 11111, "X", date(2026, 2, 27), "NIFTY_FUT")}
        assert should_reroll(tokens, today) is True


class TestOptionTokens:
    def test_resolve_option_tokens(self):
        instruments = _mock_nfo_instruments()
        tokens = resolve_option_tokens(instruments, "NIFTY", spot_price=23500.0,
                                       n_strikes=5, n_expiries=2)

        # 5 each side + ATM = 11 strikes × 2 types × 2 expiries = 44
        assert len(tokens) == 44

        # All should be NIFTY options
        for ot in tokens.values():
            assert ot.index_name == "NIFTY"
            assert ot.option_type in ("CE", "PE")
            assert ot.storage_key == "NIFTY_OPT"
            assert 23250 <= ot.strike <= 23750  # ±5 strikes × 50pt

    def test_resolve_option_tokens_single_expiry(self):
        instruments = _mock_nfo_instruments()
        tokens = resolve_option_tokens(instruments, "NIFTY", spot_price=23500.0,
                                       n_strikes=3, n_expiries=1)

        # 3 each side + ATM = 7 strikes × 2 types × 1 expiry = 14
        assert len(tokens) == 14

    def test_resolve_option_tokens_unknown_index(self):
        instruments = _mock_nfo_instruments()
        tokens = resolve_option_tokens(instruments, "FINNIFTY", spot_price=20000.0)
        assert len(tokens) == 0

    def test_resolve_banknifty_options(self):
        instruments = _mock_nfo_instruments()
        tokens = resolve_option_tokens(instruments, "BANKNIFTY", spot_price=50000.0,
                                       n_strikes=5, n_expiries=1)

        for ot in tokens.values():
            assert ot.index_name == "BANKNIFTY"
            assert ot.storage_key == "BANKNIFTY_OPT"

    def test_check_recenter_not_needed(self):
        tokens = {
            1: OptionToken("NIFTY", 1, "X", date(2026, 2, 13), 23500.0, "CE", "NIFTY_OPT"),
        }
        # Spot moved 100 pts, threshold is 150 (3×50)
        result = check_recenter_needed(tokens, {"NIFTY": 23600.0}, {"NIFTY": 23500.0})
        assert result == []

    def test_check_recenter_needed(self):
        tokens = {
            1: OptionToken("NIFTY", 1, "X", date(2026, 2, 13), 23500.0, "CE", "NIFTY_OPT"),
        }
        # Spot moved 200 pts, threshold is 150 (3×50)
        result = check_recenter_needed(tokens, {"NIFTY": 23700.0}, {"NIFTY": 23500.0})
        assert result == ["NIFTY"]

    def test_check_recenter_banknifty_threshold(self):
        tokens = {
            1: OptionToken("BANKNIFTY", 1, "X", date(2026, 2, 13), 50000.0, "CE", "BANKNIFTY_OPT"),
        }
        # Spot moved 250, threshold is 300 (3×100)
        result = check_recenter_needed(tokens, {"BANKNIFTY": 50250.0}, {"BANKNIFTY": 50000.0})
        assert result == []

        # Spot moved 350, past threshold
        result = check_recenter_needed(tokens, {"BANKNIFTY": 50350.0}, {"BANKNIFTY": 50000.0})
        assert result == ["BANKNIFTY"]
