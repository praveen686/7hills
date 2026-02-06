"""Tests for the Binance data connector — REST parsing and URL builders."""

from __future__ import annotations

import pytest

from qlx.data.binance import (
    _perp_book_url,
    _spot_book_url,
    load_binance_env,
    BookTicker,
)


class TestUrlBuilders:
    def test_spot_book_url_single(self):
        url = _spot_book_url(["BTCUSDT"])
        assert "stream.binance.com:9443" in url
        assert "btcusdt@bookTicker" in url

    def test_spot_book_url_multiple(self):
        url = _spot_book_url(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        assert "btcusdt@bookTicker" in url
        assert "ethusdt@bookTicker" in url
        assert "solusdt@bookTicker" in url
        # Streams are slash-separated
        assert "/" in url.split("streams=")[1]

    def test_perp_book_url(self):
        url = _perp_book_url(["BTCUSDT"])
        assert "fstream.binance.com" in url
        assert "btcusdt@bookTicker" in url

    def test_symbols_lowercased(self):
        url = _spot_book_url(["BTCUSDT"])
        assert "BTCUSDT" not in url
        assert "btcusdt" in url


class TestBookTicker:
    def test_frozen(self):
        bt = BookTicker(
            symbol="BTCUSDT",
            bid_price=50000.0,
            ask_price=50001.0,
            bid_qty=1.0,
            ask_qty=2.0,
            source="spot",
        )
        assert bt.symbol == "BTCUSDT"
        assert bt.source == "spot"
        with pytest.raises(AttributeError):
            bt.bid_price = 99999.0  # type: ignore[misc]


class TestLoadEnv:
    def test_returns_dict_keys(self):
        env = load_binance_env()
        assert "api_key" in env
        assert "api_secret" in env
        assert "api_key_ed25519" in env
        assert "use_testnet" in env
        assert "environment" in env

    def test_env_values_loaded(self):
        """Verify our .env file is found and loaded."""
        env = load_binance_env()
        # Should have loaded real keys from .env
        assert len(env["api_key"]) > 0
        assert len(env["api_secret"]) > 0
