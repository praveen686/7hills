"""Tests for the SBE binary decoder — ported from quantlaxmi-sbe Rust tests."""

from __future__ import annotations

import struct

import pytest

from qlx.data.sbe import (
    SBE_HEADER_SIZE,
    TEMPLATE_DEPTH,
    TEMPLATE_TRADES,
    AggTrade,
    DepthUpdate,
    L2Level,
    SbeHeader,
    TradeEntry,
    _us_to_datetime,
    decode_depth,
    decode_header,
    decode_message,
    decode_trade,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

class TestSbeHeader:
    def test_decode_depth_header(self):
        """Exact test vector from Rust: [0x1a, 0x00, 0x13, 0x27, 0x01, 0x00, 0x00, 0x00]"""
        data = bytes([0x1A, 0x00, 0x13, 0x27, 0x01, 0x00, 0x00, 0x00])
        header = SbeHeader.decode(data)
        assert header.block_length == 26
        assert header.template_id == 10003  # 0x2713
        assert header.schema_id == 1
        assert header.version == 0

    def test_decode_trade_header(self):
        data = struct.pack("<HHHH", 18, TEMPLATE_TRADES, 1, 0)
        header = SbeHeader.decode(data)
        assert header.template_id == TEMPLATE_TRADES
        assert header.block_length == 18

    def test_short_buffer_raises(self):
        with pytest.raises(ValueError, match="too short"):
            SbeHeader.decode(b"\x00\x01\x02")

    def test_header_is_frozen(self):
        header = SbeHeader(block_length=10, template_id=100, schema_id=1, version=0)
        with pytest.raises(AttributeError):
            header.block_length = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------

class TestTimestamp:
    def test_us_to_datetime(self):
        """Same test as Rust: 1766053116014248 us."""
        ts_us = 1766053116014248
        dt = _us_to_datetime(ts_us)
        assert dt.timestamp() > 0
        assert dt.year > 2020

    def test_round_trip(self):
        ts_us = 1700000000000000  # 2023-11-14 ~
        dt = _us_to_datetime(ts_us)
        recovered_us = int(dt.timestamp() * 1_000_000)
        assert abs(recovered_us - ts_us) < 2  # sub-microsecond rounding


# ---------------------------------------------------------------------------
# Trade decoder
# ---------------------------------------------------------------------------

def _build_trade_frame(
    transact_time_us: int = 1700000000000000,
    event_time_us: int = 1700000000100000,
    price_exp: int = -2,
    qty_exp: int = -8,
    trades: list[tuple[int, int, int, bool]] | None = None,
) -> bytes:
    """Build a complete SBE trade frame (header + body) for testing."""
    if trades is None:
        # Single trade: price=50000.00, qty=0.01
        trades = [(1, 5000000, 1000000, False)]

    # Header: block_length=18, template=10000, schema=1, version=0
    header = struct.pack("<HHHH", 18, TEMPLATE_TRADES, 1, 0)

    # Body fixed: transact_time(i64), event_time(i64), price_exp(i8), qty_exp(i8)
    body = struct.pack("<qqbb", transact_time_us, event_time_us, price_exp, qty_exp)

    # Group header: block_length=25(u16), count(u32)
    body += struct.pack("<HI", 25, len(trades))

    # Trade entries: trade_id(i64), price_mantissa(i64), qty_mantissa(i64), is_buyer_maker(u8)
    for tid, pm, qm, ibm in trades:
        body += struct.pack("<qqqB", tid, pm, qm, 1 if ibm else 0)

    return header + body


class TestDecodeTrade:
    def test_single_trade(self):
        frame = _build_trade_frame(
            price_exp=-2,
            qty_exp=-8,
            trades=[(42, 5000000, 1000000, False)],
        )
        result = decode_message(frame)
        assert isinstance(result, AggTrade)
        assert result.trade_count == 1
        assert abs(result.price - 50000.0) < 0.01
        assert abs(result.quantity - 0.01) < 1e-10
        assert result.is_buyer_maker is False

    def test_multiple_trades(self):
        frame = _build_trade_frame(
            price_exp=-2,
            qty_exp=-8,
            trades=[
                (1, 5000000, 1000000, False),
                (2, 5000100, 2000000, True),
            ],
        )
        result = decode_message(frame)
        assert isinstance(result, AggTrade)
        assert result.trade_count == 2
        # Last trade is the aggregate
        assert abs(result.price - 50001.0) < 0.01
        assert result.is_buyer_maker is True

    def test_trades_tuple_preserved(self):
        frame = _build_trade_frame(
            trades=[(10, 100, 200, False), (11, 101, 201, True)],
            price_exp=0,
            qty_exp=0,
        )
        result = decode_message(frame)
        assert isinstance(result, AggTrade)
        assert len(result.trades) == 2
        assert result.trades[0].trade_id == 10
        assert result.trades[1].trade_id == 11

    def test_wrong_template_raises(self):
        header = SbeHeader(block_length=18, template_id=9999, schema_id=1, version=0)
        with pytest.raises(ValueError, match="Invalid template ID"):
            decode_trade(header, b"\x00" * 100)

    def test_short_body_raises(self):
        header = SbeHeader(block_length=18, template_id=TEMPLATE_TRADES, schema_id=1, version=0)
        with pytest.raises(ValueError, match="too short"):
            decode_trade(header, b"\x00" * 10)

    def test_exchange_time_property(self):
        frame = _build_trade_frame(transact_time_us=1700000000000000)
        result = decode_message(frame)
        assert isinstance(result, AggTrade)
        dt = result.exchange_time
        assert dt.year == 2023


# ---------------------------------------------------------------------------
# Depth decoder
# ---------------------------------------------------------------------------

def _build_depth_frame(
    transact_time_us: int = 1700000000000000,
    first_id: int = 100,
    last_id: int = 105,
    price_exp: int = -2,
    qty_exp: int = -8,
    bids: list[tuple[int, int]] | None = None,
    asks: list[tuple[int, int]] | None = None,
) -> bytes:
    """Build a complete SBE depth frame (header + body) for testing."""
    if bids is None:
        bids = [(5000000, 100000000)]  # price=50000.00, qty=1.0
    if asks is None:
        asks = [(5000100, 200000000)]  # price=50001.00, qty=2.0

    # Header: block_length=26, template=10003, schema=1, version=0
    header = struct.pack("<HHHH", 26, TEMPLATE_DEPTH, 1, 0)

    # Body fixed: transact_time(i64), first_update_id(i64), last_update_id(i64),
    #             price_exp(i8), qty_exp(i8)
    body = struct.pack("<qqqbb", transact_time_us, first_id, last_id, price_exp, qty_exp)

    # Bids group: block_length=16(u16), count(u16)
    body += struct.pack("<HH", 16, len(bids))
    for pm, qm in bids:
        body += struct.pack("<qq", pm, qm)

    # Asks group: block_length=16(u16), count(u16)
    body += struct.pack("<HH", 16, len(asks))
    for pm, qm in asks:
        body += struct.pack("<qq", pm, qm)

    return header + body


class TestDecodeDepth:
    def test_single_level(self):
        frame = _build_depth_frame(
            price_exp=-2,
            qty_exp=-8,
            bids=[(5000000, 100000000)],
            asks=[(5000100, 200000000)],
        )
        result = decode_message(frame)
        assert isinstance(result, DepthUpdate)
        assert len(result.bids) == 1
        assert len(result.asks) == 1
        assert abs(result.bids[0].price - 50000.0) < 0.01
        assert abs(result.bids[0].size - 1.0) < 1e-8
        assert abs(result.asks[0].price - 50001.0) < 0.01
        assert abs(result.asks[0].size - 2.0) < 1e-8

    def test_multiple_levels(self):
        frame = _build_depth_frame(
            price_exp=-2,
            qty_exp=-2,
            bids=[
                (5000000, 100),  # 50000.00, 1.00
                (4999900, 200),  # 49999.00, 2.00
            ],
            asks=[
                (5000100, 150),  # 50001.00, 1.50
                (5000200, 300),  # 50002.00, 3.00
                (5000300, 500),  # 50003.00, 5.00
            ],
        )
        result = decode_message(frame)
        assert isinstance(result, DepthUpdate)
        assert len(result.bids) == 2
        assert len(result.asks) == 3

    def test_update_ids(self):
        frame = _build_depth_frame(first_id=1000, last_id=1005)
        result = decode_message(frame)
        assert isinstance(result, DepthUpdate)
        assert result.first_update_id == 1000
        assert result.last_update_id == 1005

    def test_wrong_template_raises(self):
        header = SbeHeader(block_length=26, template_id=9999, schema_id=1, version=0)
        with pytest.raises(ValueError, match="Invalid template ID"):
            decode_depth(header, b"\x00" * 100)

    def test_exchange_time_property(self):
        frame = _build_depth_frame(transact_time_us=1700000000000000)
        result = decode_message(frame)
        assert isinstance(result, DepthUpdate)
        assert result.exchange_time.year == 2023


# ---------------------------------------------------------------------------
# Unknown template
# ---------------------------------------------------------------------------

class TestUnknownTemplate:
    def test_unknown_template_raises(self):
        data = struct.pack("<HHHH", 10, 55555, 1, 0) + b"\x00" * 50
        with pytest.raises(ValueError, match="Unknown SBE template"):
            decode_message(data)


# ---------------------------------------------------------------------------
# L2Level and TradeEntry are frozen
# ---------------------------------------------------------------------------

class TestFrozenTypes:
    def test_l2level_frozen(self):
        lvl = L2Level(price=100.0, size=1.0)
        with pytest.raises(AttributeError):
            lvl.price = 200.0  # type: ignore[misc]

    def test_trade_entry_frozen(self):
        te = TradeEntry(trade_id=1, price=50000.0, quantity=0.1, is_buyer_maker=False)
        with pytest.raises(AttributeError):
            te.price = 60000.0  # type: ignore[misc]
