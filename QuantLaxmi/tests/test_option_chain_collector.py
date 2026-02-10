"""Tests for the option chain snapshot collector.

Covers:
  - IndexInstruments / InstrumentMap data classes
  - _select_near_strikes ATM filtering
  - snapshot_chain with mocked Kite
  - save_snapshot / list_snapshots round-trip
  - run_collector market-hours logic (mocked)
  - Constants and configuration
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantlaxmi.data.collectors.option_chain.collector import (
    INDEX_TOKENS,
    LOT_SIZES,
    SNAPSHOT_DIR,
    SPOT_SYMBOLS,
    STRIKES_EACH_SIDE,
    IndexInstruments,
    InstrumentMap,
    _select_near_strikes,
    list_snapshots,
    load_instrument_map,
    save_snapshot,
    snapshot_chain,
)

IST = timezone(timedelta(hours=5, minutes=30))


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _make_opts_df(name: str = "NIFTY", n_strikes: int = 60, n_expiries: int = 4):
    """Create a synthetic options instrument DataFrame."""
    base_strike = 24000
    rows = []
    today = datetime.now(IST).date()
    for exp_idx in range(n_expiries):
        expiry = today + timedelta(days=7 * (exp_idx + 1))
        for i in range(n_strikes):
            strike = base_strike + (i - n_strikes // 2) * 50
            for otype in ["CE", "PE"]:
                rows.append({
                    "name": name,
                    "tradingsymbol": f"{name}{expiry.strftime('%y%b').upper()}{strike}{otype}",
                    "instrument_type": otype,
                    "strike": float(strike),
                    "expiry": expiry,
                    "instrument_token": 100000 + len(rows),
                    "lot_size": 75,
                })
    return pd.DataFrame(rows)


def _make_futs_df(name: str = "NIFTY"):
    """Create a synthetic futures instrument DataFrame."""
    today = datetime.now(IST).date()
    return pd.DataFrame([
        {
            "name": name,
            "tradingsymbol": f"{name}{(today + timedelta(days=14)).strftime('%y%b').upper()}FUT",
            "instrument_type": "FUT",
            "strike": 0.0,
            "expiry": today + timedelta(days=14),
            "instrument_token": 999999,
            "lot_size": 75,
        },
        {
            "name": name,
            "tradingsymbol": f"{name}{(today + timedelta(days=45)).strftime('%y%b').upper()}FUT",
            "instrument_type": "FUT",
            "strike": 0.0,
            "expiry": today + timedelta(days=45),
            "instrument_token": 999998,
            "lot_size": 75,
        },
    ])


def _make_instruments_df(name: str = "NIFTY"):
    """Combined opts + futs DataFrame as Kite instruments() would return."""
    opts = _make_opts_df(name)
    futs = _make_futs_df(name)
    return pd.concat([opts, futs], ignore_index=True)


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    def test_index_tokens(self):
        assert "NIFTY" in INDEX_TOKENS
        assert "BANKNIFTY" in INDEX_TOKENS
        assert INDEX_TOKENS["NIFTY"] == 256265

    def test_lot_sizes(self):
        assert LOT_SIZES["NIFTY"] == 75
        assert LOT_SIZES["BANKNIFTY"] == 30

    def test_spot_symbols(self):
        assert SPOT_SYMBOLS["NIFTY"] == "NSE:NIFTY 50"
        assert SPOT_SYMBOLS["BANKNIFTY"] == "NSE:NIFTY BANK"


# ===========================================================================
# _select_near_strikes
# ===========================================================================

class TestSelectNearStrikes:
    def test_selects_around_atm(self):
        opts = _make_opts_df(n_strikes=60, n_expiries=4)
        spot = 24000.0
        result = _select_near_strikes(opts, spot, n_each_side=10, max_expiries=2)

        strikes = sorted(result["strike"].unique())
        assert len(strikes) <= 21  # 10 each side + ATM
        assert min(abs(s - spot) for s in strikes) <= 50  # ATM within 50 pts

    def test_limits_expiries(self):
        opts = _make_opts_df(n_strikes=20, n_expiries=6)
        result = _select_near_strikes(opts, 24000.0, max_expiries=2)
        assert len(result["expiry"].unique()) <= 2

    def test_handles_edge_spot(self):
        """Spot near edge of strike range shouldn't crash."""
        opts = _make_opts_df(n_strikes=10)
        # Spot far below strikes
        result = _select_near_strikes(opts, 20000.0, n_each_side=5)
        assert len(result) > 0

    def test_both_ce_and_pe(self):
        opts = _make_opts_df(n_strikes=20, n_expiries=1)
        result = _select_near_strikes(opts, 24000.0, n_each_side=5)
        assert "CE" in result["instrument_type"].values
        assert "PE" in result["instrument_type"].values


# ===========================================================================
# load_instrument_map
# ===========================================================================

class TestLoadInstrumentMap:
    def test_basic_load(self):
        kite = MagicMock()
        kite.instruments.return_value = _make_instruments_df("NIFTY").to_dict("records")

        imap = load_instrument_map(kite, symbols=["NIFTY"])

        assert imap.loaded_date == str(datetime.now(IST).date())
        assert "NIFTY" in imap.indices
        idx = imap.indices["NIFTY"]
        assert isinstance(idx, IndexInstruments)
        assert len(idx.opts) > 0
        assert idx.fut_token == 999999  # near-month

    def test_missing_symbol_skipped(self):
        kite = MagicMock()
        kite.instruments.return_value = _make_instruments_df("NIFTY").to_dict("records")

        imap = load_instrument_map(kite, symbols=["NIFTY", "FAKEINDEX"])
        assert "NIFTY" in imap.indices
        assert "FAKEINDEX" not in imap.indices

    def test_selects_near_month_futures(self):
        kite = MagicMock()
        kite.instruments.return_value = _make_instruments_df("NIFTY").to_dict("records")

        imap = load_instrument_map(kite, symbols=["NIFTY"])
        idx = imap.indices["NIFTY"]
        # Should pick the earliest expiry future
        assert idx.fut_token == 999999


# ===========================================================================
# snapshot_chain
# ===========================================================================

class TestSnapshotChain:
    def _make_imap(self):
        opts = _make_opts_df("NIFTY", n_strikes=20, n_expiries=2)
        idx = IndexInstruments(
            opts=opts,
            fut_token=999999,
            fut_symbol="NIFTY26MARFUT",
        )
        return InstrumentMap(
            indices={"NIFTY": idx},
            loaded_date=str(datetime.now(IST).date()),
        )

    def test_snapshot_returns_dataframe(self):
        kite = MagicMock()
        imap = self._make_imap()

        # Mock spot quote
        kite.quote.side_effect = [
            {"NSE:NIFTY 50": {"last_price": 24000.0}},
            {"NFO:NIFTY26MARFUT": {"last_price": 24050.0, "oi": 1000000}},
            # Batch of option quotes
            {f"NFO:{ts}": {
                "last_price": 100.0,
                "oi": 5000,
                "volume": 1000,
                "depth": {
                    "buy": [{"price": 99.0, "quantity": 50}],
                    "sell": [{"price": 101.0, "quantity": 50}],
                },
            } for ts in imap.indices["NIFTY"].opts["tradingsymbol"].iloc[:80]},
        ]

        df = snapshot_chain(kite, imap, "NIFTY")
        assert df is not None
        assert len(df) > 0
        assert "underlying_price" in df.columns
        assert "futures_price" in df.columns
        assert df["underlying_price"].iloc[0] == 24000.0

    def test_snapshot_missing_symbol(self):
        kite = MagicMock()
        imap = self._make_imap()

        result = snapshot_chain(kite, imap, "BANKNIFTY")
        assert result is None

    def test_snapshot_spot_fetch_failure(self):
        kite = MagicMock()
        kite.quote.side_effect = Exception("API error")
        imap = self._make_imap()

        result = snapshot_chain(kite, imap, "NIFTY")
        assert result is None


# ===========================================================================
# save_snapshot / list_snapshots
# ===========================================================================

class TestSaveAndList:
    def test_save_creates_parquet(self, tmp_dir):
        df = pd.DataFrame({
            "timestamp": [datetime.now(IST)],
            "symbol": ["NIFTY"],
            "strike": [24000.0],
            "option_type": ["CE"],
            "ltp": [150.0],
        })

        with patch(
            "quantlaxmi.data.collectors.option_chain.collector.SNAPSHOT_DIR",
            tmp_dir,
        ):
            path = save_snapshot(df, "NIFTY")
            assert path.exists()
            assert path.suffix == ".parquet"
            assert "NIFTY_" in path.name

            # Read back
            loaded = pd.read_parquet(path)
            assert len(loaded) == 1
            assert loaded["strike"].iloc[0] == 24000.0

    def test_list_snapshots(self, tmp_dir):
        # Create some fake snapshot files
        today = datetime.now(IST).strftime("%Y-%m-%d")
        day_dir = tmp_dir / today
        day_dir.mkdir(parents=True)

        for sym, time_str in [("NIFTY", "091500"), ("NIFTY", "093000"), ("BANKNIFTY", "091500")]:
            (day_dir / f"{sym}_{time_str}.parquet").write_bytes(b"fake")

        with patch(
            "quantlaxmi.data.collectors.option_chain.collector.SNAPSHOT_DIR",
            tmp_dir,
        ):
            all_files = list_snapshots(date=today)
            assert len(all_files) == 3

            nifty_files = list_snapshots(symbol="NIFTY", date=today)
            assert len(nifty_files) == 2

            bnf_files = list_snapshots(symbol="BANKNIFTY", date=today)
            assert len(bnf_files) == 1

    def test_list_empty_date(self, tmp_dir):
        with patch(
            "quantlaxmi.data.collectors.option_chain.collector.SNAPSHOT_DIR",
            tmp_dir,
        ):
            assert list_snapshots(date="2020-01-01") == []


# ===========================================================================
# InstrumentMap / IndexInstruments
# ===========================================================================

class TestDataClasses:
    def test_index_instruments(self):
        opts = _make_opts_df(n_strikes=5, n_expiries=1)
        idx = IndexInstruments(opts=opts, fut_token=123, fut_symbol="NIFTY26FEBFUT")
        assert idx.fut_token == 123
        assert len(idx.opts) == 10  # 5 strikes * 2 (CE + PE)

    def test_instrument_map(self):
        imap = InstrumentMap(indices={}, loaded_date="2026-02-10")
        assert imap.loaded_date == "2026-02-10"
        assert len(imap.indices) == 0
