# Kite 5-Level Depth Collector

Streams NIFTY + BANKNIFTY **futures and near-ATM options** 5-level order book depth from Zerodha KiteTicker (`MODE_FULL`, ~1 tick/sec per instrument) into date-partitioned parquet files.

No historical L2 order book data exists for NSE — this collector is the path to OFI (Order Flow Imbalance) computation for India equity derivatives.

## Quick Start

```bash
cd QuantLaxmi

# Futures + options depth (default: ±15 strikes, 2 nearest expiries)
python -m apps.kite_depth collect

# Futures only (no options)
python -m apps.kite_depth collect --futures-only

# Custom strike window
python -m apps.kite_depth collect --n-strikes 20 --n-expiries 3

# Short test run
python -m apps.kite_depth collect --duration 60

# Check stored data
python -m apps.kite_depth status

# Read futures depth
python -m apps.kite_depth read NIFTY_FUT

# Read options depth (shows strikes/expiries summary)
python -m apps.kite_depth read NIFTY_OPT --date 2026-02-06
```

## What Gets Collected

| File | Instruments | Tokens | ~Rows/day |
|------|-------------|--------|-----------|
| `NIFTY_FUT.parquet` | NIFTY near-month futures | 1 | ~22.5K |
| `BANKNIFTY_FUT.parquet` | BANKNIFTY near-month futures | 1 | ~22.5K |
| `NIFTY_OPT.parquet` | NIFTY ±15 strikes × CE/PE × 2 expiries | ~124 | ~2.8M |
| `BANKNIFTY_OPT.parquet` | BANKNIFTY ±15 strikes × CE/PE × 2 expiries | ~124 | ~2.8M |
| **Total** | | **~250** | **~5.6M** |

Estimated storage: ~100-150 MB/day with zstd compression.

## Output

```
data/zerodha/5level/
  2026-02-06/
    NIFTY_FUT.parquet        # futures depth
    BANKNIFTY_FUT.parquet
    NIFTY_OPT.parquet        # all NIFTY option strikes in one file
    BANKNIFTY_OPT.parquet
```

## Parquet Schema (41 columns)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ms` | int64 | Exchange timestamp (epoch ms) |
| `instrument_token` | int32 | Kite instrument token |
| `symbol` | string | `NIFTY_FUT` / `NIFTY_OPT` / etc. |
| `last_price` | float64 | Last traded price |
| `volume` | int64 | Cumulative traded volume |
| `oi` | int64 | Open interest |
| `total_buy_qty` | int64 | Aggregate buy quantity |
| `total_sell_qty` | int64 | Aggregate sell quantity |
| `strike` | float64 | Strike price (0.0 for futures) |
| `expiry` | string | Expiry date "YYYY-MM-DD" |
| `option_type` | string | "CE", "PE", or "FUT" |
| `bid_price_1..5` | float64 | Bid price levels 1 (best) to 5 |
| `bid_qty_1..5` | int64 | Bid quantity at each level |
| `bid_orders_1..5` | int32 | Order count at each level |
| `ask_price_1..5` | float64 | Ask price levels 1 (best) to 5 |
| `ask_qty_1..5` | int64 | Ask quantity at each level |
| `ask_orders_1..5` | int32 | Order count at each level |

## Dynamic ATM Recentering

The options strike window tracks spot price in real-time:

1. At startup: fetch spot via REST, select ±N strikes around ATM
2. Every 60s: check if spot has moved ≥ 3 strike widths from center
   - NIFTY: recenter threshold = 150 pts (3 × 50pt strikes)
   - BANKNIFTY: recenter threshold = 300 pts (3 × 100pt strikes)
3. If triggered: compute new strike window, unsubscribe old / subscribe new tokens
4. Spot price tracked from futures LTP (no extra REST calls during collection)

## Daemon Operation

```bash
# Start as background daemon
./scripts/kite_depth.sh

# Crontab entry for auto-restart (Mon-Fri, 9:00-16:00 IST)
*/5 3-10 * * 1-5 /home/ubuntu/Desktop/7hills/QuantLaxmi/scripts/watchdog_kite_depth.sh
```

## Architecture

```
KiteTicker MODE_FULL (Twisted thread, ~250 tokens)
    │
    ▼
KiteTickFeed (async queue bridge, subscribe/unsubscribe)
    │
    ▼
DepthCollector.run()
    ├── headless_login() → KiteConnect auth
    ├── kite.instruments("NFO") → cache instrument dump
    ├── resolve_futures_tokens() → 2 futures tokens
    ├── resolve_option_tokens() → ~248 option tokens (±15 strikes × 2 expiries × 2 indices)
    ├── KiteTickFeed.start() → subscribe all
    └── async for tick:
        ├── DepthTick.from_kite_tick() → flatten 5 levels + strike/expiry/type
        ├── DepthStore.add_tick() → route to NIFTY_FUT / NIFTY_OPT / etc.
        ├── DepthStore.maybe_flush() → atomic parquet write (60s / 10K ticks)
        └── _maybe_recenter() → dynamic subscribe/unsubscribe (every 60s)
```

## Tests

```bash
python -m pytest tests/test_kite_depth.py -v   # 27 tests
```

Covers: schema validation (41 cols), DepthTick conversion (futures + options metadata),
DepthStore flush/append/rollover, futures token resolution, options token resolution
(ATM selection, n_strikes, n_expiries), recenter threshold logic (NIFTY + BANKNIFTY).
