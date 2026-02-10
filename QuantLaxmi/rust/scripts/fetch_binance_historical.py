#!/usr/bin/env python3
"""
Fetch historical Binance Futures data for CRT-Omega backtesting.

This script fetches:
1. Aggregated trades (aggTrades) - for order flow signals
2. Klines (1m) - for price/quote approximation
3. Funding rates - for perp-specific signals

Usage:
    python scripts/fetch_binance_historical.py --symbol BTCUSDT --days 7 --output data/historical

The output is formatted as QuantLaxmi canonical JSONL for direct backtest consumption.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

# Binance Futures API base URL
FAPI_BASE = "https://fapi.binance.com"

def fetch_json(url: str, max_retries: int = 3) -> list:
    """Fetch JSON from URL with retries."""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                wait = 2 ** attempt
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise
    return []


def fetch_agg_trades(symbol: str, start_time: int, end_time: int, limit: int = 1000) -> list:
    """Fetch aggregated trades for a time range."""
    url = f"{FAPI_BASE}/fapi/v1/aggTrades?symbol={symbol}&startTime={start_time}&endTime={end_time}&limit={limit}"
    return fetch_json(url)


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> list:
    """Fetch klines for a time range."""
    url = f"{FAPI_BASE}/fapi/v1/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
    return fetch_json(url)


def fetch_funding_rate(symbol: str, start_time: int, end_time: int, limit: int = 1000) -> list:
    """Fetch historical funding rates."""
    url = f"{FAPI_BASE}/fapi/v1/fundingRate?symbol={symbol}&startTime={start_time}&endTime={end_time}&limit={limit}"
    return fetch_json(url)


def fetch_mark_price_klines(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> list:
    """Fetch mark price klines."""
    url = f"{FAPI_BASE}/fapi/v1/markPriceKlines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"
    return fetch_json(url)


def convert_trade_to_canonical(trade: dict, symbol: str) -> dict:
    """Convert Binance aggTrade to QuantLaxmi canonical format."""
    # aggTrade fields: a=aggTradeId, p=price, q=quantity, f=firstTradeId, l=lastTradeId, T=timestamp, m=isBuyerMaker
    ts_ms = trade["T"]
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    price = float(trade["p"])
    qty = float(trade["q"])
    is_buyer_maker = trade["m"]  # True = buyer is maker = aggressive SELL

    return {
        "ts": ts,
        "symbol": symbol,
        "trade_id": trade["a"],
        "price_mantissa": int(price * 100),  # exp -2
        "qty_mantissa": int(qty * 100_000_000),  # exp -8
        "is_buyer_maker": is_buyer_maker,
        "price_exponent": -2,
        "qty_exponent": -8,
        "venue": "binance_futures",
        "ctx": {}
    }


def convert_kline_to_quote(kline: list, symbol: str) -> dict:
    """Convert Binance kline to QuantLaxmi spot quote format (approximation)."""
    # Kline: [openTime, open, high, low, close, volume, closeTime, quoteVolume, trades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
    ts_ms = kline[0]
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    # Use close price as mid, approximate bid/ask with small spread
    close = float(kline[4])
    spread = close * 0.0001  # 1 bps spread approximation

    bid = close - spread / 2
    ask = close + spread / 2

    # Use volume as rough qty estimate
    volume = float(kline[5])

    return {
        "ts": ts,
        "symbol": symbol,
        "bid_price_mantissa": int(bid * 100),
        "ask_price_mantissa": int(ask * 100),
        "bid_qty_mantissa": int(volume * 100_000_000 / 2),
        "ask_qty_mantissa": int(volume * 100_000_000 / 2),
        "price_exponent": -2,
        "qty_exponent": -8,
        "venue": "binance",
        "ctx": {}
    }


def convert_funding_to_canonical(fr: dict, symbol: str) -> dict:
    """Convert Binance funding rate to QuantLaxmi format."""
    ts_ms = fr["fundingTime"]
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    rate = float(fr["fundingRate"])

    return {
        "ts": ts,
        "symbol": symbol,
        "funding_rate_mantissa": int(rate * 100_000_000),  # exp -8
        "rate_exponent": -8,
        "mark_price_mantissa": int(float(fr.get("markPrice", 0)) * 100) if fr.get("markPrice") else 0,
        "price_exponent": -2,
        "venue": "binance_futures",
        "ctx": {}
    }


def convert_trade_to_depth(trade: dict, symbol: str) -> dict:
    """Convert an aggTrade to a depth snapshot for tick-level resolution.

    This creates synthetic L2 depth from trade data:
    - If is_buyer_maker=True (aggressive sell), trade price ≈ bid
    - If is_buyer_maker=False (aggressive buy), trade price ≈ ask
    """
    ts_ms = trade["T"]
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    price = float(trade["p"])
    qty = float(trade["q"])
    is_buyer_maker = trade["m"]  # True = aggressive sell (hit bid)

    # Approximate spread based on typical perp spreads (2-5 bps)
    spread = price * 0.0003  # 3 bps spread

    if is_buyer_maker:
        # Aggressive sell hit the bid, so trade price ≈ bid
        bid = price
        ask = price + spread
    else:
        # Aggressive buy lifted the ask, so trade price ≈ ask
        ask = price
        bid = price - spread

    bid_price = int(bid * 100)
    ask_price = int(ask * 100)
    qty_mantissa = int(qty * 100_000_000)

    return {
        "ts": ts,
        "tradingsymbol": symbol,
        "market": "perp",
        "first_update_id": trade["a"],  # aggTradeId for sequencing
        "last_update_id": trade["a"],
        "price_exponent": -2,
        "qty_exponent": -8,
        "bids": [
            {"price": bid_price, "qty": qty_mantissa},
            {"price": bid_price - 10, "qty": qty_mantissa // 2},
            {"price": bid_price - 20, "qty": qty_mantissa // 4},
        ],
        "asks": [
            {"price": ask_price, "qty": qty_mantissa},
            {"price": ask_price + 10, "qty": qty_mantissa // 2},
            {"price": ask_price + 20, "qty": qty_mantissa // 4},
        ],
        "is_snapshot": True,
        "venue": "binance_futures",
        "ctx": {"from_trade": True, "is_buyer_maker": is_buyer_maker}
    }


def convert_perp_kline_to_depth(kline: list, symbol: str) -> dict:
    """Convert kline to perp depth format matching QuantLaxmi L2 schema."""
    ts_ms = kline[0]
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

    close = float(kline[4])
    high = float(kline[2])
    low = float(kline[3])
    volume = float(kline[5])

    # Approximate spread from high-low range
    spread = (high - low) * 0.1  # Rough approximation
    spread = max(spread, close * 0.0001)  # Min 1 bps

    bid = close - spread / 2
    ask = close + spread / 2

    # Create L2 book format with bids/asks arrays (matching original format)
    bid_price = int(bid * 100)
    ask_price = int(ask * 100)
    qty = int(volume * 100_000_000 / 10)  # Distribute volume across levels

    return {
        "ts": ts,
        "tradingsymbol": symbol,
        "market": "perp",
        "first_update_id": int(ts_ms),
        "last_update_id": int(ts_ms),
        "price_exponent": -2,
        "qty_exponent": -8,
        "bids": [
            {"price": bid_price, "qty": qty},
            {"price": bid_price - 10, "qty": qty},
            {"price": bid_price - 20, "qty": qty},
        ],
        "asks": [
            {"price": ask_price, "qty": qty},
            {"price": ask_price + 10, "qty": qty},
            {"price": ask_price + 20, "qty": qty},
        ],
        "is_snapshot": True,  # Each kline is a full snapshot, not a delta
        "venue": "binance_futures",
        "ctx": {}
    }


def fetch_all_data(symbol: str, start_dt: datetime, end_dt: datetime, output_dir: Path, include_trades: bool = True, high_res: bool = False):
    """Fetch all historical data for a symbol.

    Args:
        high_res: If True, generate perp_depth from trades (tick-level) instead of klines (1-min)
    """

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Create output directory
    symbol_dir = output_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for {symbol} from {start_dt} to {end_dt}")

    # 1. Fetch klines (1m interval) for spot_quotes approximation
    print("Fetching 1m klines for spot quotes...")
    klines = []
    current_start = start_ms
    while current_start < end_ms:
        batch = fetch_klines(symbol, "1m", current_start, end_ms, limit=1500)
        if not batch:
            break
        klines.extend(batch)
        current_start = batch[-1][6] + 1  # closeTime + 1ms
        print(f"  Fetched {len(klines)} klines...")
        time.sleep(0.1)  # Rate limit respect

    # Write spot quotes
    spot_quotes_file = symbol_dir / "spot_quotes.jsonl"
    with open(spot_quotes_file, "w") as f:
        for kline in klines:
            quote = convert_kline_to_quote(kline, symbol)
            f.write(json.dumps(quote) + "\n")
    print(f"  Wrote {len(klines)} spot quotes to {spot_quotes_file}")

    # 2. Fetch perp depth (from klines or trades)
    trades = []  # Initialize for later use
    depth_count = 0

    if high_res:
        # High-resolution mode: fetch trades first and use them for perp_depth
        print("HIGH-RES MODE: Fetching aggregated trades for tick-level depth...")
        current_start = start_ms
        batch_count = 0
        while current_start < end_ms:
            batch = fetch_agg_trades(symbol, current_start, end_ms, limit=1000)
            if not batch:
                break
            trades.extend(batch)
            current_start = batch[-1]["T"] + 1
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Fetched {len(trades)} trades...")
            time.sleep(0.05)

            # Limit for high-res mode (can be adjusted)
            if len(trades) > 2_000_000:
                print("  Reached 2M trade limit for high-res mode, stopping...")
                break

        print(f"  Total trades fetched: {len(trades)}")
        print("  Generating tick-level perp_depth from trades...")

        perp_depth_file = symbol_dir / "perp_depth.jsonl"
        with open(perp_depth_file, "w") as f:
            for trade in trades:
                depth = convert_trade_to_depth(trade, symbol)
                f.write(json.dumps(depth) + "\n")
                depth_count += 1
        print(f"  Wrote {depth_count} tick-level perp depth events to {perp_depth_file}")
    else:
        # Standard mode: use 1m klines for perp_depth
        print("Fetching 1m perp klines for perp depth...")
        perp_depth_file = symbol_dir / "perp_depth.jsonl"
        with open(perp_depth_file, "w") as f:
            for kline in klines:  # Reuse same klines
                depth = convert_perp_kline_to_depth(kline, symbol)
                f.write(json.dumps(depth) + "\n")
                depth_count += 1
        print(f"  Wrote {depth_count} perp depth events to {perp_depth_file}")

    # 3. Fetch funding rates
    print("Fetching funding rates...")
    funding_rates = []
    current_start = start_ms
    while current_start < end_ms:
        batch = fetch_funding_rate(symbol, current_start, end_ms, limit=1000)
        if not batch:
            break
        funding_rates.extend(batch)
        if len(batch) < 1000:
            break
        current_start = batch[-1]["fundingTime"] + 1
        time.sleep(0.1)

    funding_file = symbol_dir / "funding.jsonl"
    with open(funding_file, "w") as f:
        for fr in funding_rates:
            event = convert_funding_to_canonical(fr, symbol)
            f.write(json.dumps(event) + "\n")
    print(f"  Wrote {len(funding_rates)} funding events to {funding_file}")

    # 4. Fetch aggregated trades (optional, can be large)
    # In high_res mode, trades were already fetched above
    if include_trades and not high_res:
        print("Fetching aggregated trades (this may take a while)...")
        current_start = start_ms
        batch_count = 0
        while current_start < end_ms:
            batch = fetch_agg_trades(symbol, current_start, end_ms, limit=1000)
            if not batch:
                break
            trades.extend(batch)
            current_start = batch[-1]["T"] + 1
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Fetched {len(trades)} trades...")
            time.sleep(0.05)  # Lighter rate limit for trades

            # Limit to avoid huge files
            if len(trades) > 5_000_000:
                print("  Reached 5M trade limit, stopping...")
                break

    # Write trades file if we have them
    if trades:
        trades_file = symbol_dir / "agg_trades.jsonl"
        with open(trades_file, "w") as f:
            for trade in trades:
                event = convert_trade_to_canonical(trade, symbol)
                f.write(json.dumps(event) + "\n")
        print(f"  Wrote {len(trades)} trades to {trades_file}")

    # 5. Create segment manifest
    manifest = {
        "schema_version": 9,
        "quote_schema": "canonical_v1",
        "state": "FINALIZED",
        "segment_id": f"historical_{symbol}_{start_dt.strftime('%Y%m%d')}",
        "symbols": [symbol],
        "capture_mode": "historical_fetch",
        "start_ts": start_dt.isoformat(),
        "end_ts": end_dt.isoformat(),
        "events": {
            "spot_quotes": len(klines),
            "perp_quotes": depth_count,
            "funding": len(funding_rates),
            "trades": len(trades)
        },
        "high_res": high_res,
        "config": {
            "price_exponent": -2,
            "qty_exponent": -8
        }
    }

    manifest_file = output_dir / "segment_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {manifest_file}")

    print(f"\nDone! Data saved to {output_dir}")
    print(f"  Klines: {len(klines)}")
    print(f"  Funding rates: {len(funding_rates)}")
    if include_trades:
        print(f"  Trades: {len(trades)}")


def main():
    parser = argparse.ArgumentParser(description="Fetch historical Binance Futures data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--days", type=int, default=7, help="Number of days to fetch")
    parser.add_argument("--output", default="data/historical", help="Output directory")
    parser.add_argument("--no-trades", action="store_true", help="Skip fetching trades (faster)")
    parser.add_argument("--high-res", action="store_true", help="Use tick-level data from trades for depth (slower, more accurate)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD), defaults to now")

    args = parser.parse_args()

    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)

    start_dt = end_dt - timedelta(days=args.days)

    output_dir = Path(args.output) / f"{args.symbol}_{args.days}d_{start_dt.strftime('%Y%m%d')}"

    # For high-res mode with many days, warn user
    if args.high_res and args.days > 2:
        print(f"WARNING: High-res mode with {args.days} days will fetch many trades. Consider using --days 1 or --days 2.")

    fetch_all_data(
        symbol=args.symbol,
        start_dt=start_dt,
        end_dt=end_dt,
        output_dir=output_dir,
        include_trades=not args.no_trades,
        high_res=args.high_res
    )


if __name__ == "__main__":
    main()
