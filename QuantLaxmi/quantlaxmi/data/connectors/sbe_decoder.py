"""Simple Binary Encoding (SBE) decoder for Binance market data.

Direct Python port of ``quantlaxmi-sbe`` Rust crate.  Decodes binary
WebSocket frames from ``wss://stream-sbe.binance.com:9443/stream`` into
typed dataclasses.

Supported message types
-----------------------
=============  ===========================  ============================
Template ID    Rust type                    Python type
=============  ===========================  ============================
10000          TradesStreamEvent            :class:`AggTrade`
10003          DepthDiffStreamEvent         :class:`DepthUpdate`
=============  ===========================  ============================

Binary layout
-------------
All integers are **little-endian**.  Floating-point values are encoded as
``mantissa * 10^exponent`` where mantissa is i64 and exponent is i8.

References
----------
- Binance SBE docs: https://github.com/binance/binance-spot-api-docs
- SBE spec: https://github.com/FIXTradingCommunity/fix-simple-binary-encoding

Provenance
----------
Extracted from ``QuantLaxmi_archive/qlxr_crypto/qlx/data/sbe.py``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SBE_HEADER_SIZE: int = 8

TEMPLATE_TRADES: int = 10000
TEMPLATE_DEPTH: int = 10003


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SbeHeader:
    """8-byte SBE message header preceding every frame."""

    block_length: int   # u16
    template_id: int    # u16
    schema_id: int      # u16
    version: int        # u16

    @classmethod
    def decode(cls, data: bytes | bytearray | memoryview) -> SbeHeader:
        if len(data) < SBE_HEADER_SIZE:
            raise ValueError(f"Data too short for SBE header: {len(data)} bytes")
        bl, tid, sid, ver = struct.unpack_from("<HHHH", data, 0)
        return cls(block_length=bl, template_id=tid, schema_id=sid, version=ver)


@dataclass(frozen=True)
class L2Level:
    """Single price level in the order book."""

    price: float
    size: float


@dataclass(frozen=True)
class TradeEntry:
    """Raw trade within an SBE trades group."""

    trade_id: int
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass(frozen=True)
class AggTrade:
    """Decoded aggregated trade event (template 10000)."""

    price: float
    quantity: float
    transact_time_us: int
    event_time_us: int
    is_buyer_maker: bool
    trade_count: int
    trades: tuple[TradeEntry, ...] = field(repr=False)

    @property
    def exchange_time(self) -> datetime:
        return _us_to_datetime(self.transact_time_us)


@dataclass(frozen=True)
class DepthUpdate:
    """Decoded order-book depth diff (template 10003)."""

    transact_time_us: int
    first_update_id: int
    last_update_id: int
    bids: tuple[L2Level, ...]
    asks: tuple[L2Level, ...]

    @property
    def exchange_time(self) -> datetime:
        return _us_to_datetime(self.transact_time_us)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode_header(data: bytes | bytearray | memoryview) -> SbeHeader:
    """Decode the 8-byte SBE header from the start of *data*."""
    return SbeHeader.decode(data)


def decode_trade(header: SbeHeader, body: bytes | bytearray | memoryview) -> AggTrade:
    """Decode a trades stream event (template 10000).

    Parameters
    ----------
    header : SbeHeader
        Pre-decoded message header.
    body : bytes
        Payload *after* the 8-byte header.
    """
    if header.template_id != TEMPLATE_TRADES:
        raise ValueError(
            f"Invalid template ID for trades: {header.template_id} (expected {TEMPLATE_TRADES})"
        )
    if len(body) < 24:
        raise ValueError(f"Trade body too short: {len(body)} bytes")

    # Fixed fields: transact_time(i64), event_time(i64), price_exp(i8), qty_exp(i8)
    transact_time_us = struct.unpack_from("<q", body, 0)[0]
    event_time_us = struct.unpack_from("<q", body, 8)[0]
    price_exp = struct.unpack_from("<b", body, 16)[0]
    qty_exp = struct.unpack_from("<b", body, 17)[0]

    price_mult = 10.0 ** price_exp
    qty_mult = 10.0 ** qty_exp

    # Group header at offset 18: block_length(u16), count(u32)
    group_offset = 18
    if len(body) < group_offset + 6:
        raise ValueError("Trade body too short for group header")

    trade_block_len = struct.unpack_from("<H", body, group_offset)[0]
    group_count = struct.unpack_from("<I", body, group_offset + 2)[0]

    if trade_block_len == 0:
        raise ValueError("Invalid trade block length: 0")

    # Parse each 25-byte trade entry
    trades: list[TradeEntry] = []
    offset = group_offset + 6

    for _ in range(group_count):
        if len(body) < offset + 25:
            raise ValueError(f"Trade body too short for entry at offset {offset}")

        trade_id = struct.unpack_from("<q", body, offset)[0]
        price_mantissa = struct.unpack_from("<q", body, offset + 8)[0]
        qty_mantissa = struct.unpack_from("<q", body, offset + 16)[0]
        is_buyer_maker = body[offset + 24] != 0

        trades.append(TradeEntry(
            trade_id=trade_id,
            price=price_mantissa * price_mult,
            quantity=qty_mantissa * qty_mult,
            is_buyer_maker=is_buyer_maker,
        ))
        offset += trade_block_len

    if not trades:
        raise ValueError("No trades in SBE message")

    last = trades[-1]
    return AggTrade(
        price=last.price,
        quantity=last.quantity,
        transact_time_us=transact_time_us,
        event_time_us=event_time_us,
        is_buyer_maker=last.is_buyer_maker,
        trade_count=len(trades),
        trades=tuple(trades),
    )


def decode_depth(header: SbeHeader, body: bytes | bytearray | memoryview) -> DepthUpdate:
    """Decode an order-book depth diff (template 10003).

    Parameters
    ----------
    header : SbeHeader
        Pre-decoded message header.
    body : bytes
        Payload *after* the 8-byte header.
    """
    if header.template_id != TEMPLATE_DEPTH:
        raise ValueError(
            f"Invalid template ID for depth: {header.template_id} (expected {TEMPLATE_DEPTH})"
        )
    if len(body) < 26:
        raise ValueError(f"Depth body too short: {len(body)} bytes")

    transact_time_us = struct.unpack_from("<q", body, 0)[0]
    first_update_id = struct.unpack_from("<q", body, 8)[0]
    last_update_id = struct.unpack_from("<q", body, 16)[0]
    price_exp = struct.unpack_from("<b", body, 24)[0]
    qty_exp = struct.unpack_from("<b", body, 25)[0]

    price_mult = 10.0 ** price_exp
    qty_mult = 10.0 ** qty_exp

    offset = 26

    # Bids group: block_length(u16), count(u16)
    if len(body) < offset + 4:
        raise ValueError("Depth body too short for bids group header")

    bid_block_len = struct.unpack_from("<H", body, offset)[0]
    bid_count = struct.unpack_from("<H", body, offset + 2)[0]
    offset += 4

    bids: list[L2Level] = []
    for i in range(bid_count):
        if len(body) < offset + 16:
            raise ValueError(f"Depth body too short for bid entry {i} at offset {offset}")
        price_m = struct.unpack_from("<q", body, offset)[0]
        qty_m = struct.unpack_from("<q", body, offset + 8)[0]
        bids.append(L2Level(price=price_m * price_mult, size=qty_m * qty_mult))
        offset += bid_block_len

    # Asks group: block_length(u16), count(u16)
    if len(body) < offset + 4:
        raise ValueError(f"Depth body too short for asks group header at offset {offset}")

    ask_block_len = struct.unpack_from("<H", body, offset)[0]
    ask_count = struct.unpack_from("<H", body, offset + 2)[0]
    offset += 4

    asks: list[L2Level] = []
    for i in range(ask_count):
        if len(body) < offset + 16:
            raise ValueError(f"Depth body too short for ask entry {i} at offset {offset}")
        price_m = struct.unpack_from("<q", body, offset)[0]
        qty_m = struct.unpack_from("<q", body, offset + 8)[0]
        asks.append(L2Level(price=price_m * price_mult, size=qty_m * qty_mult))
        offset += ask_block_len

    return DepthUpdate(
        transact_time_us=transact_time_us,
        first_update_id=first_update_id,
        last_update_id=last_update_id,
        bids=tuple(bids),
        asks=tuple(asks),
    )


def decode_message(data: bytes | bytearray | memoryview) -> AggTrade | DepthUpdate:
    """Decode a full SBE frame (header + body) and dispatch by template ID."""
    header = decode_header(data)
    body = data[SBE_HEADER_SIZE:]

    if header.template_id == TEMPLATE_TRADES:
        return decode_trade(header, body)
    elif header.template_id == TEMPLATE_DEPTH:
        return decode_depth(header, body)
    else:
        raise ValueError(f"Unknown SBE template ID: {header.template_id}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _us_to_datetime(timestamp_us: int) -> datetime:
    """Convert microsecond epoch to timezone-aware UTC datetime."""
    secs = timestamp_us / 1_000_000
    return datetime.fromtimestamp(secs, tz=timezone.utc)
