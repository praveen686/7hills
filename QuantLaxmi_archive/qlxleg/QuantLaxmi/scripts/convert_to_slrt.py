#!/usr/bin/env python3
"""Convert captured session data to SLRT MarketEvent format."""

import json
import sys
from datetime import datetime
from pathlib import Path

def parse_ts_to_ns(ts_str: str) -> int:
    """Convert ISO timestamp string to nanoseconds."""
    # Handle various timestamp formats
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    dt = datetime.fromisoformat(ts_str)
    return int(dt.timestamp() * 1_000_000_000)

def convert_depth(line: str) -> dict:
    """Convert depth event to SLRT Book format."""
    d = json.loads(line)

    # Convert bids and asks to PriceLevel format
    # Format can be either:
    # - Object: {"price": ..., "qty": ...}
    # - Array: [price, qty]
    bids = []
    for bid in d.get('bids', []):
        if isinstance(bid, dict):
            bids.append({
                "price_mantissa": int(bid['price']),
                "price_exponent": d.get('price_exponent', -2),
                "qty_mantissa": int(bid['qty']),
                "qty_exponent": d.get('qty_exponent', -8),
            })
        elif isinstance(bid, list) and len(bid) >= 2:
            bids.append({
                "price_mantissa": int(bid[0]),
                "price_exponent": d.get('price_exponent', -2),
                "qty_mantissa": int(bid[1]),
                "qty_exponent": d.get('qty_exponent', -8),
            })

    asks = []
    for ask in d.get('asks', []):
        if isinstance(ask, dict):
            asks.append({
                "price_mantissa": int(ask['price']),
                "price_exponent": d.get('price_exponent', -2),
                "qty_mantissa": int(ask['qty']),
                "qty_exponent": d.get('qty_exponent', -8),
            })
        elif isinstance(ask, list) and len(ask) >= 2:
            asks.append({
                "price_mantissa": int(ask[0]),
                "price_exponent": d.get('price_exponent', -2),
                "qty_mantissa": int(ask[1]),
                "qty_exponent": d.get('qty_exponent', -8),
            })

    return {
        "type": "Book",
        "ts_ns": parse_ts_to_ns(d['ts']),
        "symbol": d.get('tradingsymbol', 'BTCUSDT'),
        "bids": bids,
        "asks": asks,
    }

def convert_trade(line: str) -> dict:
    """Convert trade event to SLRT Trade format."""
    t = json.loads(line)

    # Determine side from is_buyer_maker
    # is_buyer_maker=true means seller was aggressor
    # is_buyer_maker=false means buyer was aggressor
    if t.get('is_buyer_maker', False):
        side = {"Present": "Sell"}
    else:
        side = {"Present": "Buy"}

    return {
        "type": "Trade",
        "ts_ns": parse_ts_to_ns(t['ts']),
        "symbol": t.get('tradingsymbol', 'BTCUSDT'),
        "price_mantissa": int(t['price']),
        "price_exponent": t.get('price_exponent', -2),
        "qty_mantissa": int(t['qty']),
        "qty_exponent": t.get('qty_exponent', -8),
        "side": side,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_to_slrt.py <session_dir> [output.jsonl]", file=sys.stderr)
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Find the symbol directory (e.g., BTCUSDT)
    symbol_dirs = [d for d in session_dir.iterdir() if d.is_dir()]
    if not symbol_dirs:
        print(f"No symbol directories found in {session_dir}", file=sys.stderr)
        sys.exit(1)

    symbol_dir = symbol_dirs[0]
    print(f"Processing symbol: {symbol_dir.name}", file=sys.stderr)

    events = []

    # Load depth events (apply diffs to maintain book state)
    depth_file = symbol_dir / "perp_depth.jsonl"
    if depth_file.exists():
        print(f"Loading depth from {depth_file}...", file=sys.stderr)

        # Book state: {price: qty}
        bid_book = {}
        ask_book = {}
        price_exp = -2
        qty_exp = -8
        initialized = False

        with open(depth_file) as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        d = json.loads(line)
                        price_exp = d.get('price_exponent', -2)
                        qty_exp = d.get('qty_exponent', -8)

                        if d.get('is_snapshot', False):
                            # Full snapshot - reset book
                            bid_book = {}
                            ask_book = {}
                            for bid in d.get('bids', []):
                                if isinstance(bid, dict):
                                    bid_book[bid['price']] = bid['qty']
                            for ask in d.get('asks', []):
                                if isinstance(ask, dict):
                                    ask_book[ask['price']] = ask['qty']
                            initialized = True
                        elif initialized:
                            # Apply diff - update only changed levels
                            for bid in d.get('bids', []):
                                if isinstance(bid, dict):
                                    if bid['qty'] == 0:
                                        bid_book.pop(bid['price'], None)
                                    else:
                                        bid_book[bid['price']] = bid['qty']
                            for ask in d.get('asks', []):
                                if isinstance(ask, dict):
                                    if ask['qty'] == 0:
                                        ask_book.pop(ask['price'], None)
                                    else:
                                        ask_book[ask['price']] = ask['qty']

                        if initialized:
                            # Build book event with top 20 levels
                            sorted_bids = sorted(bid_book.items(), key=lambda x: -x[0])[:20]
                            sorted_asks = sorted(ask_book.items(), key=lambda x: x[0])[:20]

                            bids = [{"price_mantissa": p, "price_exponent": price_exp,
                                     "qty_mantissa": q, "qty_exponent": qty_exp}
                                    for p, q in sorted_bids]
                            asks = [{"price_mantissa": p, "price_exponent": price_exp,
                                     "qty_mantissa": q, "qty_exponent": qty_exp}
                                    for p, q in sorted_asks]

                            events.append({
                                "type": "Book",
                                "ts_ns": parse_ts_to_ns(d['ts']),
                                "symbol": d.get('tradingsymbol', 'BTCUSDT'),
                                "bids": bids,
                                "asks": asks,
                            })
                    except Exception as e:
                        if i < 5:
                            print(f"  Depth parse error line {i}: {e}", file=sys.stderr)
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i+1} depth events, {len(events)} books", file=sys.stderr)
        print(f"  Total book snapshots: {len(events)}", file=sys.stderr)

    depth_count = len(events)

    # Load trade events
    trades_file = symbol_dir / "agg_trades.jsonl"
    if trades_file.exists():
        print(f"Loading trades from {trades_file}...", file=sys.stderr)
        trade_count = 0
        with open(trades_file) as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        events.append(convert_trade(line))
                        trade_count += 1
                    except Exception as e:
                        if i < 5:
                            print(f"  Trade parse error line {i}: {e}", file=sys.stderr)
                if (i + 1) % 10000 == 0:
                    print(f"  Loaded {i+1} trade events", file=sys.stderr)
        print(f"  Total trade events: {trade_count}", file=sys.stderr)

    # Sort by timestamp
    print(f"Sorting {len(events)} events by timestamp...", file=sys.stderr)
    events.sort(key=lambda e: e['ts_ns'])

    # Output
    if output_path:
        print(f"Writing to {output_path}...", file=sys.stderr)
        with open(output_path, 'w') as f:
            for e in events:
                f.write(json.dumps(e) + '\n')
    else:
        for e in events:
            print(json.dumps(e))

    print(f"Done: {depth_count} depth + {len(events) - depth_count} trades = {len(events)} total", file=sys.stderr)

if __name__ == "__main__":
    main()
