"""Tests for the Zerodha (Kite) data connector — env loading and helpers."""

from __future__ import annotations

import pytest

from data.zerodha import (
    INTERVALS,
    KiteTick,
    generate_totp,
    load_zerodha_env,
)


class TestLoadEnv:
    def test_returns_dict_keys(self):
        env = load_zerodha_env()
        assert "user_id" in env
        assert "password" in env
        assert "totp_secret" in env
        assert "api_key" in env
        assert "api_secret" in env

    def test_env_values_loaded(self):
        env = load_zerodha_env()
        assert len(env["api_key"]) > 0
        assert len(env["user_id"]) > 0


class TestTotp:
    def test_generates_6_digit_code(self):
        env = load_zerodha_env()
        if not env["totp_secret"]:
            pytest.skip("No TOTP secret in .env")
        code = generate_totp(env["totp_secret"])
        assert len(code) == 6
        assert code.isdigit()


class TestIntervals:
    def test_all_mapped(self):
        assert "1m" in INTERVALS
        assert "1h" in INTERVALS
        assert "1d" in INTERVALS
        assert INTERVALS["1h"] == "60minute"


class TestKiteTick:
    def test_from_raw(self):
        raw = {
            "instrument_token": 256265,
            "last_price": 19500.50,
            "volume_traded": 1234567,
            "total_buy_quantity": 100000,
            "total_sell_quantity": 90000,
            "ohlc": {"open": 19400, "high": 19600, "low": 19350, "close": 19500},
        }
        tick = KiteTick.from_raw(raw)
        assert tick.instrument_token == 256265
        assert tick.last_price == 19500.50
        assert tick.volume == 1234567
        assert tick.ohlc["high"] == 19600

    def test_from_raw_missing_fields(self):
        """Gracefully handles missing fields with defaults."""
        tick = KiteTick.from_raw({})
        assert tick.instrument_token == 0
        assert tick.last_price == 0.0
        assert tick.volume == 0
