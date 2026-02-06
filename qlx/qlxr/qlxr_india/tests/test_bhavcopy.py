"""Tests for bhavcopy data module — parsing, caching, helpers."""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from apps.india_scanner.bhavcopy import (
    BhavcopyCache,
    NSESession,
    _parse_csv_from_zip,
    _parse_delivery_from_file,
    _parse_fno_from_file,
    extract_nearest_futures_oi,
    is_trading_day,
)


# ---------------------------------------------------------------------------
# is_trading_day
# ---------------------------------------------------------------------------


class TestIsTradingDay:
    def test_weekday_is_trading(self):
        # 2026-02-02 is Monday
        assert is_trading_day(date(2026, 2, 2)) is True

    def test_saturday_not_trading(self):
        assert is_trading_day(date(2026, 1, 31)) is False  # Saturday

    def test_sunday_not_trading(self):
        assert is_trading_day(date(2026, 2, 1)) is False  # Sunday

    def test_known_holiday(self):
        assert is_trading_day(date(2026, 1, 26)) is False  # Republic Day

    def test_regular_wednesday(self):
        assert is_trading_day(date(2026, 2, 4)) is True  # Wednesday


# ---------------------------------------------------------------------------
# CSV from ZIP parsing
# ---------------------------------------------------------------------------


def _make_zip_bytes(csv_content: str, filename: str = "data.csv") -> bytes:
    """Create a zip file in memory with a single CSV."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, csv_content)
    return buf.getvalue()


class TestParseCSVFromZip:
    def test_basic_parse(self):
        csv = "SYMBOL,CLOSE\nRELIANCE,2800\nTCS,3500\n"
        df = _parse_csv_from_zip(_make_zip_bytes(csv))
        assert len(df) == 2
        assert df.iloc[0]["SYMBOL"] == "RELIANCE"

    def test_with_hint(self):
        csv = "A,B\n1,2\n"
        zb = _make_zip_bytes(csv, "bhav.csv")
        df = _parse_csv_from_zip(zb, filename_hint="bhav")
        assert len(df) == 1


# ---------------------------------------------------------------------------
# extract_nearest_futures_oi
# ---------------------------------------------------------------------------


class TestExtractFuturesOI:
    def test_basic_extraction(self):
        df = pd.DataFrame({
            "INSTRUMENT": ["FUTSTK", "FUTSTK", "FUTSTK", "OPTIDX"],
            "SYMBOL": ["RELIANCE", "RELIANCE", "TCS", "NIFTY"],
            "EXPIRY_DT": ["27-Feb-2026", "26-Mar-2026", "27-Feb-2026", "27-Feb-2026"],
            "CLOSE": [2800, 2810, 3500, 100],
            "OPEN_INT": [5000000, 3000000, 2000000, 10000000],
            "CHG_IN_OI": [100000, 50000, -50000, 200000],
        })
        result = extract_nearest_futures_oi(df, date(2026, 2, 4))
        assert len(result) == 2  # RELIANCE + TCS, no OPTIDX
        rel = result[result["SYMBOL"] == "RELIANCE"]
        assert rel.iloc[0]["OPEN_INT"] == 5000000  # nearest expiry

    def test_empty_input(self):
        df = pd.DataFrame()
        result = extract_nearest_futures_oi(df, date(2026, 2, 4))
        assert result.empty

    def test_no_futstk(self):
        df = pd.DataFrame({
            "INSTRUMENT": ["OPTIDX"],
            "SYMBOL": ["NIFTY"],
            "EXPIRY_DT": ["27-Feb-2026"],
            "CLOSE": [100],
            "OPEN_INT": [10000],
            "CHG_IN_OI": [1000],
        })
        result = extract_nearest_futures_oi(df, date(2026, 2, 4))
        assert result.empty


# ---------------------------------------------------------------------------
# BhavcopyCache
# ---------------------------------------------------------------------------


class TestBhavcopyCache:
    def test_save_and_load(self, tmp_path: Path):
        cache = BhavcopyCache(tmp_path)
        df = pd.DataFrame({"SYMBOL": ["RELIANCE", "TCS"], "CLOSE": [2800, 3500]})
        cache._save("equity", date(2026, 2, 3), df)

        assert cache._has("equity", date(2026, 2, 3))
        loaded = cache._load("equity", date(2026, 2, 3))
        assert len(loaded) == 2
        assert loaded.iloc[0]["SYMBOL"] == "RELIANCE"

    def test_available_dates(self, tmp_path: Path):
        cache = BhavcopyCache(tmp_path)
        df = pd.DataFrame({"A": [1]})
        cache._save("equity", date(2026, 2, 3), df)
        cache._save("equity", date(2026, 2, 4), df)

        dates = cache.available_dates("equity")
        assert dates == [date(2026, 2, 3), date(2026, 2, 4)]

    def test_no_data_dir(self, tmp_path: Path):
        cache = BhavcopyCache(tmp_path / "nonexistent")
        assert cache.available_dates("equity") == []

    def test_get_equity_cached(self, tmp_path: Path):
        """get_equity delegates to get_delivery — save to 'delivery' category."""
        cache = BhavcopyCache(tmp_path)
        df = pd.DataFrame({
            "SYMBOL": ["RELIANCE"],
            "SERIES": ["EQ"],
            "CLOSE": [2800],
        })
        cache._save("delivery", date(2026, 2, 3), df)
        result = cache.get_equity(date(2026, 2, 3))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# nse_daily fallback
# ---------------------------------------------------------------------------


class TestNseDailyFallback:
    """BhavcopyCache reads from nse_daily raw files before falling back to HTTP."""

    def test_get_fno_from_nse_daily(self, tmp_path: Path):
        """get_fno reads fo_bhavcopy.csv.zip from nse_daily dir."""
        d = date(2026, 2, 5)
        cache_dir = tmp_path / "cache"
        nse_daily_dir = tmp_path / "nse_daily"

        # Create a raw FO bhavcopy zip in nse_daily dir
        day_dir = nse_daily_dir / d.isoformat()
        day_dir.mkdir(parents=True)

        csv_content = (
            "TckrSymb,FinInstrmTp,XpryDt,OpnPric,HghPric,LwPric,ClsPric,"
            "OpnIntrst,ChngInOpnIntrst,TtlTradgVol\n"
            "RELIANCE,STF,27-Feb-2026,2800,2850,2780,2820,5000000,100000,50000\n"
        )
        zip_path = day_dir / "fo_bhavcopy.csv.zip"
        zip_path.write_bytes(_make_zip_bytes(csv_content, "bhav.csv"))

        cache = BhavcopyCache(base_dir=cache_dir, nse_daily_dir=nse_daily_dir)
        df = cache.get_fno(d)

        assert len(df) == 1
        assert df.iloc[0]["SYMBOL"] == "RELIANCE"
        assert df.iloc[0]["INSTRUMENT"] == "FUTSTK"  # STF mapped
        assert df.iloc[0]["CLOSE"] == 2820
        # Should also be cached as parquet now
        assert cache._has("fno", d)

    def test_get_fno_falls_back_to_download(self, tmp_path: Path):
        """Without nse_daily file, get_fno falls back to HTTP download."""
        d = date(2026, 2, 5)
        cache_dir = tmp_path / "cache"
        nse_daily_dir = tmp_path / "nse_daily"  # empty — no files

        cache = BhavcopyCache(base_dir=cache_dir, nse_daily_dir=nse_daily_dir)

        mock_df = pd.DataFrame({
            "SYMBOL": ["TCS"],
            "INSTRUMENT": ["FUTSTK"],
            "CLOSE": [3500],
        })

        with patch("apps.india_scanner.bhavcopy.download_fno_bhav", return_value=mock_df) as mock_dl:
            df = cache.get_fno(d)

        mock_dl.assert_called_once()
        assert len(df) == 1
        assert df.iloc[0]["SYMBOL"] == "TCS"

    def test_get_delivery_from_nse_daily(self, tmp_path: Path):
        """get_delivery reads delivery_bhavcopy.csv from nse_daily dir."""
        d = date(2026, 2, 5)
        cache_dir = tmp_path / "cache"
        nse_daily_dir = tmp_path / "nse_daily"

        day_dir = nse_daily_dir / d.isoformat()
        day_dir.mkdir(parents=True)

        csv_content = (
            " SYMBOL , SERIES , OPEN_PRICE , HIGH_PRICE , LOW_PRICE , "
            "CLOSE_PRICE , TRADED_QTY , TURNOVER_LACS , DELIV_PER \n"
            "RELIANCE,EQ,2800,2850,2780,2820,1000000,28000,45.5\n"
        )
        (day_dir / "delivery_bhavcopy.csv").write_text(csv_content)

        cache = BhavcopyCache(base_dir=cache_dir, nse_daily_dir=nse_daily_dir)
        df = cache.get_delivery(d)

        assert len(df) == 1
        assert "OPEN" in df.columns  # renamed from OPEN_PRICE
        assert "DELIVERY_PCT" in df.columns  # renamed from DELIV_PER
        assert df.iloc[0]["OPEN"] == 2800
        assert cache._has("delivery", d)

    def test_nse_daily_disabled(self, tmp_path: Path):
        """nse_daily_dir=None disables the fallback entirely."""
        d = date(2026, 2, 5)
        cache = BhavcopyCache(base_dir=tmp_path, nse_daily_dir=None)

        mock_df = pd.DataFrame({"SYMBOL": ["TCS"], "CLOSE": [3500]})
        with patch("apps.india_scanner.bhavcopy.download_fno_bhav", return_value=mock_df) as mock_dl:
            df = cache.get_fno(d)

        mock_dl.assert_called_once()
        assert len(df) == 1

    def test_empty_nse_daily_file_ignored(self, tmp_path: Path):
        """A zero-byte file in nse_daily dir is ignored (falls through to download)."""
        d = date(2026, 2, 5)
        cache_dir = tmp_path / "cache"
        nse_daily_dir = tmp_path / "nse_daily"

        day_dir = nse_daily_dir / d.isoformat()
        day_dir.mkdir(parents=True)
        (day_dir / "fo_bhavcopy.csv.zip").write_bytes(b"")  # empty file

        cache = BhavcopyCache(base_dir=cache_dir, nse_daily_dir=nse_daily_dir)

        mock_df = pd.DataFrame({"SYMBOL": ["TCS"], "CLOSE": [3500]})
        with patch("apps.india_scanner.bhavcopy.download_fno_bhav", return_value=mock_df) as mock_dl:
            df = cache.get_fno(d)

        mock_dl.assert_called_once()
