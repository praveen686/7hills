"""Tests for NSE Daily Data Collector.

Covers:
- URL generation for each date format (YYYYMMDD, DDMMYYYY, DD-Mon-YYYY)
- URL generation for each file definition
- Weekend detection (skip Sat/Sun)
- Collector idempotency (skip existing files)
- File count validation (23 files total)
- Backfill date range iteration
- Tier filtering

No real network calls — requests.Session is mocked.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apps.nse_daily.files import ALL_FILES, BASE_URL, NSEFile, format_date
from apps.nse_daily.collector import DownloadResult, NSEDailyCollector


# ---------------------------------------------------------------------------
# format_date tests
# ---------------------------------------------------------------------------

class TestFormatDate:
    def test_yyyymmdd(self):
        d = date(2026, 2, 5)
        assert format_date(d, "YYYYMMDD") == "20260205"

    def test_ddmmyyyy(self):
        d = date(2026, 2, 5)
        assert format_date(d, "DDMMYYYY") == "05022026"

    def test_dd_mon_yyyy(self):
        d = date(2026, 2, 5)
        assert format_date(d, "DD-Mon-YYYY") == "05-Feb-2026"

    def test_dd_mon_yyyy_december(self):
        d = date(2025, 12, 25)
        assert format_date(d, "DD-Mon-YYYY") == "25-Dec-2025"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown date format"):
            format_date(date(2026, 1, 1), "INVALID")


# ---------------------------------------------------------------------------
# NSEFile.url_for_date tests
# ---------------------------------------------------------------------------

class TestNSEFileUrl:
    def test_fo_bhavcopy_url(self):
        f = ALL_FILES[0]  # fo_bhavcopy.csv.zip
        assert f.name == "fo_bhavcopy.csv.zip"
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/content/fo/BhavCopy_NSE_FO_0_0_0_20260205_F_0000.csv.zip"

    def test_cm_bhavcopy_url(self):
        f = ALL_FILES[1]
        assert f.name == "cm_bhavcopy.csv.zip"
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/content/cm/BhavCopy_NSE_CM_0_0_0_20260205_F_0000.csv.zip"

    def test_participant_oi_url(self):
        f = ALL_FILES[2]
        assert f.name == "participant_oi.csv"
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/content/nsccl/fao_participant_oi_05022026.csv"

    def test_settlement_prices_url(self):
        f = ALL_FILES[4]
        assert f.name == "settlement_prices.csv"
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/archives/nsccl/sett/FOSett_prce_05022026.csv"

    def test_volatility_url(self):
        f = ALL_FILES[5]
        assert f.name == "volatility.csv"
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/archives/nsccl/volt/FOVOLT_05022026.csv"

    def test_fii_stats_url(self):
        # fii_stats uses DD-Mon-YYYY
        fii = next(f for f in ALL_FILES if f.name == "fii_stats.xls")
        url = fii.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/content/fo/fii_stats_05-Feb-2026.xls"

    def test_market_activity_url(self):
        ma = next(f for f in ALL_FILES if f.name == "market_activity.zip")
        url = ma.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/archives/fo/mkt/fo05022026.zip"

    def test_delivery_bhavcopy_url(self):
        f = next(f for f in ALL_FILES if f.name == "delivery_bhavcopy.csv")
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/products/content/sec_bhavdata_full_05022026.csv"

    def test_mto_url(self):
        f = next(f for f in ALL_FILES if f.name == "mto.dat")
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/archives/equities/mto/MTO_05022026.DAT"

    def test_margin_data_url(self):
        f = next(f for f in ALL_FILES if f.name == "margin_data.dat")
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/archives/nsccl/var/C_VAR1_05022026_1.DAT"

    def test_52wk_highlow_url(self):
        f = next(f for f in ALL_FILES if f.name == "52wk_highlow.csv")
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/content/CM_52_wk_High_low_05022026.csv"

    def test_bulk_deals_url_is_static(self):
        """bulk_deals has no date in URL — same URL regardless of date."""
        f = next(f for f in ALL_FILES if f.name == "bulk_deals.csv")
        url = f.url_for_date(date(2026, 2, 5))
        assert url == f"{BASE_URL}/content/equities/bulk.csv"

    def test_all_files_generate_valid_urls(self):
        """Every file definition produces a URL with its base URL."""
        d = date(2026, 1, 15)
        for f in ALL_FILES:
            url = f.url_for_date(d)
            assert url.startswith(f.base_url), f"{f.name} URL doesn't start with base"
            assert "{date}" not in url, f"{f.name} URL still has {{date}} placeholder"


# ---------------------------------------------------------------------------
# ALL_FILES validation
# ---------------------------------------------------------------------------

class TestAllFiles:
    def test_total_file_count(self):
        assert len(ALL_FILES) == 23

    def test_tier1_count(self):
        tier1 = [f for f in ALL_FILES if f.tier == 1]
        assert len(tier1) == 9

    def test_tier2_count(self):
        tier2 = [f for f in ALL_FILES if f.tier == 2]
        assert len(tier2) == 14

    def test_unique_names(self):
        names = [f.name for f in ALL_FILES]
        assert len(names) == len(set(names))

    def test_valid_date_formats(self):
        valid = {"YYYYMMDD", "DDMMYYYY", "DD-Mon-YYYY"}
        for f in ALL_FILES:
            assert f.date_format in valid, f"{f.name} has invalid format {f.date_format}"

    def test_optional_files(self):
        optional = {f.name for f in ALL_FILES if f.optional}
        assert optional == {"security_ban.csv", "bulk_deals.csv", "block_deals.csv", "top_gainers.json"}


# ---------------------------------------------------------------------------
# Collector — weekend detection
# ---------------------------------------------------------------------------

class TestWeekendSkip:
    def test_skip_saturday(self, tmp_path):
        collector = NSEDailyCollector(base_dir=tmp_path)
        saturday = date(2026, 2, 7)  # Saturday
        assert saturday.weekday() == 5
        result = collector.collect(saturday)
        assert result.downloaded == 0
        assert result.skipped == 0
        assert result.failed == 0

    def test_skip_sunday(self, tmp_path):
        collector = NSEDailyCollector(base_dir=tmp_path)
        sunday = date(2026, 2, 8)  # Sunday
        assert sunday.weekday() == 6
        result = collector.collect(sunday)
        assert result.downloaded == 0

    def test_weekday_attempts_download(self, tmp_path):
        """A weekday should attempt downloads (mocked to return 404)."""
        collector = NSEDailyCollector(base_dir=tmp_path)
        monday = date(2026, 2, 2)  # Monday
        assert monday.weekday() == 0

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = {}

        with patch.object(collector, "_init_session"), \
             patch.object(collector, "_init_www_session"):
            collector._session = mock_session
            collector._www_session = mock_session
            result = collector.collect(monday)

        # All files should be "missing" (404)
        assert result.missing == 23
        assert result.downloaded == 0


# ---------------------------------------------------------------------------
# Collector — idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_skip_existing_files(self, tmp_path):
        """Files that already exist with size > 0 are skipped."""
        d = date(2026, 2, 5)
        day_dir = tmp_path / d.isoformat()
        day_dir.mkdir(parents=True)

        # Pre-create all files
        for f in ALL_FILES:
            (day_dir / f.name).write_text("dummy content")

        collector = NSEDailyCollector(base_dir=tmp_path)
        # No session needed — all files should be skipped before any HTTP call
        result = collector.collect(d)
        assert result.skipped == 23
        assert result.downloaded == 0

    def test_redownload_empty_files(self, tmp_path):
        """Empty files (size=0) are re-downloaded."""
        d = date(2026, 2, 5)
        day_dir = tmp_path / d.isoformat()
        day_dir.mkdir(parents=True)

        # Pre-create one empty file
        (day_dir / ALL_FILES[0].name).write_text("")

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = {}

        collector = NSEDailyCollector(base_dir=tmp_path)
        with patch.object(collector, "_init_session"), \
             patch.object(collector, "_init_www_session"):
            collector._session = mock_session
            collector._www_session = mock_session
            result = collector.collect(d)

        # The empty file should be attempted (missing/404), rest are missing too
        assert result.missing == 23


# ---------------------------------------------------------------------------
# Collector — download success
# ---------------------------------------------------------------------------

class TestDownloadSuccess:
    def test_successful_download(self, tmp_path):
        """A 200 response writes content to disk."""
        d = date(2026, 2, 5)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"CSV,DATA,HERE\n1,2,3"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = {}

        collector = NSEDailyCollector(base_dir=tmp_path)
        with patch.object(collector, "_init_session"), \
             patch.object(collector, "_init_www_session"):
            collector._session = mock_session
            collector._www_session = mock_session
            result = collector.collect(d)

        assert result.downloaded == 23
        assert result.failed == 0

        # Verify files on disk
        day_dir = tmp_path / d.isoformat()
        for f in ALL_FILES:
            path = day_dir / f.name
            assert path.exists()
            assert path.read_bytes() == b"CSV,DATA,HERE\n1,2,3"


# ---------------------------------------------------------------------------
# Collector — tier filtering
# ---------------------------------------------------------------------------

class TestTierFiltering:
    def test_tier1_only(self, tmp_path):
        collector = NSEDailyCollector(base_dir=tmp_path, tier=1)
        assert len(collector.files) == 9
        assert all(f.tier == 1 for f in collector.files)

    def test_tier2_includes_all(self, tmp_path):
        collector = NSEDailyCollector(base_dir=tmp_path, tier=2)
        assert len(collector.files) == 23

    def test_no_tier_includes_all(self, tmp_path):
        collector = NSEDailyCollector(base_dir=tmp_path)
        assert len(collector.files) == 23


# ---------------------------------------------------------------------------
# Collector — backfill
# ---------------------------------------------------------------------------

class TestBackfill:
    def test_backfill_skips_weekends(self, tmp_path):
        """Backfill over Mon-Sun should process 5 weekdays."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = {}

        collector = NSEDailyCollector(base_dir=tmp_path)

        with patch.object(collector, "_init_session"), \
             patch("apps.nse_daily.collector.time.sleep"):
            collector._session = mock_session
            results = collector.backfill(
                date(2026, 2, 2),  # Monday
                date(2026, 2, 8),  # Sunday
            )

        assert len(results) == 7  # 7 days in range
        # Only weekdays should have downloads attempted
        weekday_results = [r for r in results if r.missing > 0]
        assert len(weekday_results) == 5

    def test_backfill_single_day(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = {}

        collector = NSEDailyCollector(base_dir=tmp_path)

        with patch.object(collector, "_init_session"), \
             patch("apps.nse_daily.collector.time.sleep"):
            collector._session = mock_session
            results = collector.backfill(date(2026, 2, 5), date(2026, 2, 5))

        assert len(results) == 1
        assert results[0].downloaded == 23

    def test_backfill_creates_date_dirs(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"data"

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.cookies = {}

        collector = NSEDailyCollector(base_dir=tmp_path)

        with patch.object(collector, "_init_session"), \
             patch("apps.nse_daily.collector.time.sleep"):
            collector._session = mock_session
            collector.backfill(date(2026, 2, 2), date(2026, 2, 4))

        assert (tmp_path / "2026-02-02").is_dir()
        assert (tmp_path / "2026-02-03").is_dir()
        assert (tmp_path / "2026-02-04").is_dir()


# ---------------------------------------------------------------------------
# Collector — retry and session refresh
# ---------------------------------------------------------------------------

class TestRetry:
    def test_403_triggers_session_refresh(self, tmp_path):
        """A 403 should refresh the session and retry."""
        d = date(2026, 2, 5)

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = b"data"

        mock_session = MagicMock()
        # First call 403, second call 200, then 200 for rest
        mock_session.get.side_effect = [resp_403, resp_200] + [resp_200] * 15
        mock_session.cookies = {}

        collector = NSEDailyCollector(base_dir=tmp_path, tier=1)

        with patch.object(collector, "_init_session"), \
             patch.object(collector, "_refresh_session"), \
             patch("apps.nse_daily.collector.time.sleep"):
            collector._session = mock_session
            result = collector.collect(d)

        assert result.downloaded == 9
        assert result.failed == 0


# ---------------------------------------------------------------------------
# DownloadResult
# ---------------------------------------------------------------------------

class TestDownloadResult:
    def test_default_values(self):
        r = DownloadResult(date=date(2026, 2, 5))
        assert r.downloaded == 0
        assert r.skipped == 0
        assert r.failed == 0
        assert r.missing == 0
        assert r.details == {}

    def test_date_preserved(self):
        d = date(2026, 2, 5)
        r = DownloadResult(date=d)
        assert r.date == d
