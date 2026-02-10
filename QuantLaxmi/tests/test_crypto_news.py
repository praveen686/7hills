"""Tests for crypto news collectors.

Tests parsing, dedup, date filtering, and archive format for all four
collectors: CryptoPanic, CoinGecko, Fear & Greed, CryptoCompare.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import pytest

# Patch HEADLINE_ARCHIVE_DIR before importing crypto_news
_tmpdir = tempfile.mkdtemp()
_test_archive_dir = Path(_tmpdir) / "test_headlines"
_test_archive_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _patch_archive_dir(tmp_path):
    """Redirect all archive writes to a temp directory."""
    archive_dir = tmp_path / "headlines"
    archive_dir.mkdir()
    with patch("quantlaxmi.data.collectors.news.crypto_news.HEADLINE_ARCHIVE_DIR", archive_dir):
        yield archive_dir


# ---------------------------------------------------------------------------
# CryptoPanic
# ---------------------------------------------------------------------------

MOCK_CRYPTOPANIC_RESPONSE = {
    "results": [
        {
            "title": "Bitcoin surges past $100,000 as institutional demand grows",
            "published_at": "2025-12-15T14:30:00Z",
            "source": {"domain": "coindesk.com"},
            "url": "https://coindesk.com/bitcoin-100k",
            "currencies": [{"code": "BTC"}, {"code": "ETH"}],
            "votes": {"positive": 42, "negative": 3},
        },
        {
            "title": "Ethereum Layer 2 adoption accelerates with new rollup technology",
            "published_at": "2025-12-14T10:00:00Z",
            "source": {"domain": "theblock.co"},
            "url": "https://theblock.co/eth-l2",
            "currencies": [{"code": "ETH"}],
            "votes": {"positive": 15, "negative": 1},
        },
        {
            "title": "Short",  # Too short — should be skipped
            "published_at": "2025-12-13T08:00:00Z",
            "source": {"domain": "x.com"},
            "url": "",
            "currencies": [],
            "votes": {},
        },
    ],
    "next": None,
}


class TestCryptoPanicParse:
    def test_parse_records(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_cryptopanic_response

        records = _parse_cryptopanic_response(MOCK_CRYPTOPANIC_RESPONSE)
        assert len(records) == 2  # "Short" title filtered out

        r0 = records[0]
        assert "Bitcoin surges" in r0["title"]
        assert r0["source"] == "cryptopanic_coindesk.com"
        assert "BTC" in r0["stocks"]
        assert "ETH" in r0["stocks"]
        assert r0["event_type"] == "institutional"  # "institutional" keyword matches before "surge"
        assert r0["ts"].startswith("2025-12-15")
        assert r0["url"] == "https://coindesk.com/bitcoin-100k"
        assert r0["indices"] == []
        # Votes appended
        assert "+42/-3" in r0["title"]

    def test_parse_empty_response(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_cryptopanic_response

        records = _parse_cryptopanic_response({"results": []})
        assert records == []

    def test_parse_missing_fields(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_cryptopanic_response

        data = {
            "results": [
                {"title": "Some crypto news article headline text", "published_at": "2025-06-01T12:00:00Z"},
            ]
        }
        records = _parse_cryptopanic_response(data)
        assert len(records) == 1
        assert records[0]["source"] == "cryptopanic_unknown"

    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_backfill_writes_to_archive(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import backfill_cryptopanic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CRYPTOPANIC_RESPONSE
        mock_get.return_value = mock_resp

        n = backfill_cryptopanic("2025-12-01", "2025-12-31")
        assert n == 2

        # Verify archive files exist
        files = list(_patch_archive_dir.glob("*.jsonl"))
        assert len(files) == 1
        assert files[0].name == "2025-12.jsonl"

        # Verify JSONL format
        with open(files[0]) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        assert all("ts" in l and "title" in l and "source" in l for l in lines)


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------

MOCK_COINGECKO_RESPONSE = {
    "data": [
        {
            "title": "Solana DEX volumes hit all-time high amid memecoin frenzy",
            "updated_at": "2025-11-20T16:45:00Z",
            "news_site": "CoinDesk",
            "url": "https://coindesk.com/sol-dex",
        },
        {
            "title": "Federal Reserve signals rate pause, crypto markets react",
            "updated_at": "2025-11-19T09:30:00Z",
            "news_site": "The Block",
            "url": "https://theblock.co/fed-crypto",
        },
    ]
}


class TestCoinGeckoParse:
    def test_parse_records(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_coingecko_response

        records = _parse_coingecko_response(MOCK_COINGECKO_RESPONSE)
        assert len(records) == 2

        r0 = records[0]
        assert "Solana" in r0["title"]
        assert r0["source"] == "coingecko_coindesk"
        assert "SOL" in r0["stocks"]
        assert r0["event_type"] == "price_action"  # "all-time high" matches before "dex"

        r1 = records[1]
        assert r1["source"] == "coingecko_the_block"
        assert r1["event_type"] == "macro"  # "fed" keyword

    def test_parse_list_format(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_coingecko_response

        # CoinGecko might return a bare list
        data = [
            {
                "title": "Bitcoin mining difficulty adjustment reaches new record",
                "updated_at": "2025-10-10T12:00:00Z",
                "news_site": "CoinTelegraph",
                "url": "https://example.com/btc-mining",
            }
        ]
        records = _parse_coingecko_response(data)
        assert len(records) == 1
        assert "BTC" in records[0]["stocks"]

    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_backfill_writes_to_archive(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import backfill_coingecko_news

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_COINGECKO_RESPONSE
        # First call returns data, second returns empty to stop
        mock_resp_empty = MagicMock()
        mock_resp_empty.status_code = 200
        mock_resp_empty.json.return_value = {"data": []}
        mock_get.side_effect = [mock_resp, mock_resp_empty]

        n = backfill_coingecko_news("2025-11-01", "2025-11-30")
        assert n == 2


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

MOCK_FEAR_GREED_RESPONSE = {
    "data": [
        {
            "value": "25",
            "value_classification": "Extreme Fear",
            "timestamp": "1735689600",  # 2025-01-01
        },
        {
            "value": "73",
            "value_classification": "Greed",
            "timestamp": "1735776000",  # 2025-01-02
        },
        {
            "value": "50",
            "value_classification": "Neutral",
            "timestamp": "1735862400",  # 2025-01-03
        },
        {
            "value": "10",
            "value_classification": "Extreme Fear",
            "timestamp": "1735948800",  # 2025-01-04
        },
    ]
}


class TestFearGreedParse:
    def test_parse_records(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_fear_greed_response

        records = _parse_fear_greed_response(MOCK_FEAR_GREED_RESPONSE)
        assert len(records) == 4

        r0 = records[0]
        assert r0["title"] == "Crypto Fear & Greed: 25 (Extreme Fear)"
        assert r0["source"] == "fear_greed_index"
        assert r0["fng_value"] == 25
        assert r0["fng_classification"] == "Extreme Fear"
        assert r0["event_type"] == "sentiment"
        assert "CRYPTO_FNG" in r0["indices"]
        assert r0["stocks"] == []

    def test_value_range(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_fear_greed_response

        records = _parse_fear_greed_response(MOCK_FEAR_GREED_RESPONSE)
        for r in records:
            assert 0 <= r["fng_value"] <= 100

    def test_invalid_value_filtered(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_fear_greed_response

        data = {
            "data": [
                {"value": "150", "value_classification": "Invalid", "timestamp": "1735689600"},
                {"value": "-5", "value_classification": "Invalid", "timestamp": "1735776000"},
                {"value": "50", "value_classification": "Neutral", "timestamp": "1735862400"},
            ]
        }
        records = _parse_fear_greed_response(data)
        assert len(records) == 1  # Only value=50 is valid
        assert records[0]["fng_value"] == 50

    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_fetch_writes_to_archive(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import fetch_fear_greed_index

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_FEAR_GREED_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        n = fetch_fear_greed_index("2025-01-01", "2025-01-04")
        assert n == 4

        files = list(_patch_archive_dir.glob("*.jsonl"))
        assert len(files) >= 1

    def test_classifications(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_fear_greed_response

        records = _parse_fear_greed_response(MOCK_FEAR_GREED_RESPONSE)
        classifications = [r["fng_classification"] for r in records]
        assert "Extreme Fear" in classifications
        assert "Greed" in classifications
        assert "Neutral" in classifications


# ---------------------------------------------------------------------------
# CryptoCompare
# ---------------------------------------------------------------------------

MOCK_CRYPTOCOMPARE_RESPONSE = {
    "Data": [
        {
            "title": "SEC approves spot Bitcoin ETF applications from major asset managers",
            "published_on": 1734307200,  # 2024-12-16T00:00:00Z
            "source": "CoinDesk",
            "url": "https://coindesk.com/sec-btc-etf",
            "categories": "BTC|ETH|Regulation",
            "body": "The Securities and Exchange Commission has approved several spot Bitcoin ETF applications...",
        },
        {
            "title": "DeFi protocol hack leads to $50M loss in smart contract exploit",
            "published_on": 1734220800,  # 2024-12-15T00:00:00Z
            "source": "The Block",
            "url": "https://theblock.co/defi-hack",
            "categories": "DeFi|Security",
            "body": "A major DeFi lending protocol suffered a smart contract exploit...",
        },
        {
            "title": "Too short",  # Should be filtered
            "published_on": 1734134400,
            "source": "Unknown",
            "url": "",
            "categories": "",
            "body": "",
        },
    ]
}


class TestCryptoCompareParse:
    def test_parse_records(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_cryptocompare_response

        records = _parse_cryptocompare_response(MOCK_CRYPTOCOMPARE_RESPONSE)
        assert len(records) == 2  # "Too short" filtered out

        r0 = records[0]
        assert "SEC approves" in r0["title"]
        assert r0["source"] == "cryptocompare_coindesk"
        assert "BTC" in r0["stocks"]
        assert "ETH" in r0["stocks"]
        # "sec" + "etf" → could be regulatory or institutional
        assert r0["event_type"] in ("regulatory", "institutional")

        r1 = records[1]
        assert "hack" in r1["title"].lower()
        assert r1["event_type"] == "security"

    def test_categories_extract_symbols(self):
        from quantlaxmi.data.collectors.news.crypto_news import _parse_cryptocompare_response

        data = {
            "Data": [
                {
                    "title": "Market update: Mixed signals across major cryptocurrencies",
                    "published_on": 1734307200,
                    "source": "CryptoDaily",
                    "url": "https://example.com",
                    "categories": "BTC|SOL|ADA",
                    "body": "",
                }
            ]
        }
        records = _parse_cryptocompare_response(data)
        assert len(records) == 1
        assert "BTC" in records[0]["stocks"]
        assert "SOL" in records[0]["stocks"]
        assert "ADA" in records[0]["stocks"]

    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_backfill_writes_to_archive(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import backfill_cryptocompare

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CRYPTOCOMPARE_RESPONSE
        # Second call returns empty to stop pagination
        mock_resp_empty = MagicMock()
        mock_resp_empty.status_code = 200
        mock_resp_empty.json.return_value = {"Data": []}
        mock_get.side_effect = [mock_resp, mock_resp_empty]

        n = backfill_cryptocompare("2024-12-01", "2024-12-31")
        assert n == 2


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedup:
    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_duplicate_titles_skipped(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import backfill_cryptopanic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CRYPTOPANIC_RESPONSE
        mock_get.return_value = mock_resp

        # First call writes 2
        n1 = backfill_cryptopanic("2025-12-01", "2025-12-31")
        assert n1 == 2

        # Second call — same data, should write 0
        n2 = backfill_cryptopanic("2025-12-01", "2025-12-31")
        assert n2 == 0

    def test_normalize_title(self):
        from quantlaxmi.data.collectors.news.crypto_news import _normalize_title

        assert _normalize_title("Bitcoin Surges!!! $100K") == "bitcoin surges 100k"
        assert _normalize_title("  Hello, World.  ") == "hello world"
        assert _normalize_title("BTC: +10%") == "btc 10"


# ---------------------------------------------------------------------------
# Archive format
# ---------------------------------------------------------------------------

class TestArchiveFormat:
    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_output_matches_headline_archive_format(self, mock_get, _patch_archive_dir):
        """Verify output is compatible with headline_archive.read_archive()."""
        from quantlaxmi.data.collectors.news.crypto_news import backfill_cryptopanic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CRYPTOPANIC_RESPONSE
        mock_get.return_value = mock_resp

        backfill_cryptopanic("2025-12-01", "2025-12-31")

        files = list(_patch_archive_dir.glob("*.jsonl"))
        assert len(files) >= 1

        with open(files[0]) as f:
            for line in f:
                rec = json.loads(line.strip())
                # Required fields
                assert "ts" in rec
                assert "title" in rec
                assert "source" in rec
                assert "url" in rec
                assert "stocks" in rec
                assert "indices" in rec
                assert "event_type" in rec

                # Type checks
                assert isinstance(rec["ts"], str)
                assert isinstance(rec["title"], str)
                assert isinstance(rec["source"], str)
                assert isinstance(rec["stocks"], list)
                assert isinstance(rec["indices"], list)

                # Timestamp parseable
                dt = datetime.fromisoformat(rec["ts"])
                assert dt.year > 2000

    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_fear_greed_extra_fields_dont_break_format(self, mock_get, _patch_archive_dir):
        """Fear & Greed adds fng_value/fng_classification; verify standard fields still present."""
        from quantlaxmi.data.collectors.news.crypto_news import fetch_fear_greed_index

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_FEAR_GREED_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        fetch_fear_greed_index("2025-01-01", "2025-01-04")

        files = list(_patch_archive_dir.glob("*.jsonl"))
        assert len(files) >= 1

        with open(files[0]) as f:
            for line in f:
                rec = json.loads(line.strip())
                # Standard fields present
                assert "ts" in rec
                assert "title" in rec
                assert "source" in rec
                assert rec["source"] == "fear_greed_index"
                # Extra fields present
                assert "fng_value" in rec
                assert isinstance(rec["fng_value"], int)


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

class TestCryptoEventClassification:
    def test_security_event(self):
        from quantlaxmi.data.collectors.news.crypto_news import _classify_crypto_event

        assert _classify_crypto_event("Major DeFi hack drains $100M from protocol") == "security"
        assert _classify_crypto_event("Exploit found in smart contract vulnerability") == "security"

    def test_regulatory_event(self):
        from quantlaxmi.data.collectors.news.crypto_news import _classify_crypto_event

        assert _classify_crypto_event("SEC files lawsuit against crypto exchange") == "regulatory"
        assert _classify_crypto_event("New regulation framework proposed for digital assets") == "regulatory"

    def test_institutional_event(self):
        from quantlaxmi.data.collectors.news.crypto_news import _classify_crypto_event

        assert _classify_crypto_event("BlackRock files for Bitcoin ETF approval") == "institutional"

    def test_macro_event(self):
        from quantlaxmi.data.collectors.news.crypto_news import _classify_crypto_event

        assert _classify_crypto_event("Fed raises interest rate, markets tumble") == "macro"

    def test_general_fallback(self):
        from quantlaxmi.data.collectors.news.crypto_news import _classify_crypto_event

        assert _classify_crypto_event("Blockchain technology continues to evolve") == "general"


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

class TestSymbolExtraction:
    def test_extract_btc_eth(self):
        from quantlaxmi.data.collectors.news.crypto_news import _extract_crypto_symbols

        syms = _extract_crypto_symbols("Bitcoin and Ethereum lead crypto rally")
        assert "BTC" in syms
        assert "ETH" in syms

    def test_extract_altcoins(self):
        from quantlaxmi.data.collectors.news.crypto_news import _extract_crypto_symbols

        syms = _extract_crypto_symbols("Solana and Cardano outperform this week")
        assert "SOL" in syms
        assert "ADA" in syms

    def test_no_duplicates(self):
        from quantlaxmi.data.collectors.news.crypto_news import _extract_crypto_symbols

        syms = _extract_crypto_symbols("BTC Bitcoin BTC price update")
        assert syms.count("BTC") == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_api_failure_returns_zero(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import backfill_cryptopanic

        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")
        n = backfill_cryptopanic("2025-01-01", "2025-01-31")
        assert n == 0

    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_backfill_all_continues_on_failure(self, mock_get, _patch_archive_dir):
        from quantlaxmi.data.collectors.news.crypto_news import backfill_all_crypto_news

        # All APIs fail
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")
        n = backfill_all_crypto_news("2025-01-01", "2025-01-31")
        assert n == 0  # No crash, returns 0


# ---------------------------------------------------------------------------
# Date filtering
# ---------------------------------------------------------------------------

class TestDateFiltering:
    @patch("quantlaxmi.data.collectors.news.crypto_news.requests.get")
    def test_out_of_range_articles_excluded(self, mock_get, _patch_archive_dir):
        """Articles outside date range should not be written."""
        from quantlaxmi.data.collectors.news.crypto_news import backfill_cryptopanic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_CRYPTOPANIC_RESPONSE
        mock_get.return_value = mock_resp

        # Request only Jan 2025 — the mock data is from Dec 2025
        n = backfill_cryptopanic("2025-01-01", "2025-01-31")
        assert n == 0  # All articles are Dec 2025, outside Jan range
