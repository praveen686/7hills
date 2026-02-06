"""Example: live Binance data — REST klines + JSON bookTicker + SBE trades.

Run with:
    python3 examples/binance_live.py --mode klines
    python3 examples/binance_live.py --mode bookticker
    python3 examples/binance_live.py --mode sbe
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# ---------------------------------------------------------------------------
# 1. REST: Download historical klines
# ---------------------------------------------------------------------------

def demo_klines(symbol: str, interval: str) -> None:
    from qlx.data.binance import fetch_klines

    print(f"Fetching {symbol} {interval} klines from Binance REST...")
    ohlcv = fetch_klines(symbol=symbol, interval=interval, limit=100, market="spot")
    df = ohlcv.df
    print(f"\nReceived {len(df)} candles")
    print(f"Time range: {df.index[0]} → {df.index[-1]}")
    print(f"\nLast 5 candles:")
    print(df.tail().to_string())
    print(f"\nClose stats: mean={df['Close'].mean():.2f}  std={df['Close'].std():.2f}")


# ---------------------------------------------------------------------------
# 2. JSON WS: Live bookTicker
# ---------------------------------------------------------------------------

async def demo_bookticker(symbols: list[str], count: int) -> None:
    from qlx.data.binance import BookTickerFeed

    print(f"Connecting to bookTicker for {symbols}...")
    feed = await BookTickerFeed.connect(symbols, include_spot=True, include_perp=True)
    print("Connected. Streaming ticks:\n")

    received = 0
    async for tick in feed:
        spread_bps = (tick.ask_price - tick.bid_price) / tick.bid_price * 10_000
        print(
            f"[{tick.source:4s}] {tick.symbol:<12s}  "
            f"bid={tick.bid_price:<12.4f} ask={tick.ask_price:<12.4f}  "
            f"spread={spread_bps:.1f}bps"
        )
        received += 1
        if received >= count:
            break

    await feed.close()
    print(f"\nReceived {received} ticks.")


# ---------------------------------------------------------------------------
# 3. SBE: Ultra-low-latency trades
# ---------------------------------------------------------------------------

async def demo_sbe(symbols: list[str], count: int) -> None:
    from qlx.data.binance import SbeStream, load_binance_env

    env = load_binance_env()
    api_key = env["api_key_ed25519"]
    if not api_key:
        print("ERROR: BINANCE_API_KEY_ED25519 not found in .env")
        sys.exit(1)

    print(f"Connecting to SBE trade stream for {symbols}...")
    stream = await SbeStream.connect(symbols=symbols, api_key=api_key)
    print("Connected. Decoding SBE frames:\n")

    received = 0
    async for event in stream:
        from qlx.data.sbe import AggTrade, DepthUpdate
        if isinstance(event, AggTrade):
            side = "SELL" if event.is_buyer_maker else "BUY "
            print(
                f"[TRADE] {event.exchange_time.strftime('%H:%M:%S.%f')[:-3]}  "
                f"price={event.price:<12.4f} qty={event.quantity:<10.6f}  "
                f"{side}  trades={event.trade_count}"
            )
        elif isinstance(event, DepthUpdate):
            top_bid = event.bids[0].price if event.bids else 0
            top_ask = event.asks[0].price if event.asks else 0
            print(
                f"[DEPTH] {event.exchange_time.strftime('%H:%M:%S.%f')[:-3]}  "
                f"bid={top_bid:<12.4f} ask={top_ask:<12.4f}  "
                f"levels={len(event.bids)}x{len(event.asks)}"
            )

        received += 1
        if received >= count:
            break

    await stream.close()
    print(f"\nDecoded {received} SBE events.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QLX Binance live data demo")
    parser.add_argument(
        "--mode", choices=["klines", "bookticker", "sbe"], default="klines",
        help="Data mode: klines (REST), bookticker (JSON WS), sbe (binary WS)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to stream (default: BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--interval", default="1h", help="Kline interval (default: 1h)",
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of ticks/events to receive before stopping (default: 50)",
    )
    args = parser.parse_args()

    if args.mode == "klines":
        demo_klines(args.symbols[0], args.interval)
    elif args.mode == "bookticker":
        asyncio.run(demo_bookticker(args.symbols, args.count))
    elif args.mode == "sbe":
        asyncio.run(demo_sbe(args.symbols, args.count))


if __name__ == "__main__":
    main()
