# Phase 27-IND Smoke Tests (KiteSim)

Fixture data for validating India KiteSim backtest pipeline.
Created: 2026-01-29 from BANKNIFTY capture session.

## Source Data

- **Session**: `banknifty_phase27ind_20260129_112835`
- **Underlying**: BANKNIFTY
- **ATM Strike**: 59700
- **Expiry**: 2026-02-24 (monthly)
- **Lot Size**: 30

## Files

| File | Description |
|------|-------------|
| `quotes_atm_ce.jsonl` | ATM CE (59700CE) quotes in QuoteEvent format |
| `quotes_straddle.jsonl` | ATM CE + PE merged, sorted by timestamp |
| `orders_sanity_atm_ce.json` | Single-leg buy+sell orders |
| `orders_straddle_test.json` | Multi-leg straddle open+close |
| `expected_sanity_report.json` | Expected report for sanity test |
| `expected_straddle_report.json` | Expected report for straddle test |

## QuoteEvent Schema

```json
{
  "ts": "2026-01-29T05:58:39.133639399Z",
  "tradingsymbol": "BANKNIFTY26FEB59700CE",
  "bid": 105100,
  "ask": 105330,
  "bid_qty": 120,
  "ask_qty": 30,
  "price_exponent": -2
}
```

Prices are integer mantissas: `actual_price = bid * 10^price_exponent`
Example: 105100 * 10^-2 = 1051.00

## Conversion from ticks.jsonl

Use `scripts/india_make_quotes.sh` to convert raw capture data:

```bash
./scripts/india_make_quotes.sh data/sessions/<session_dir>
# Creates quotes.jsonl in each instrument subdir

./scripts/india_make_quotes.sh data/sessions/<session_dir> --merge
# Also creates quotes_all.jsonl (all instruments merged, sorted)
```

## Test 1: ATM CE Sanity (Single-Leg)

```bash
./target/release/quantlaxmi-india backtest-kitesim \
  --replay tests/fixtures/india_kitesim/phase27ind_smoke/quotes_atm_ce.jsonl \
  --orders tests/fixtures/india_kitesim/phase27ind_smoke/orders_sanity_atm_ce.json \
  --out /tmp/sanity_results \
  --latency-ms 150
```

**Expected Output** (`/tmp/sanity_results/report.json`):
- `orders_total`: 2
- `legs_total`: 2
- `legs_filled`: 2
- `rollbacks`: 0
- `timeouts`: 0
- `total_pnl`: -148.50 (negative due to spread)

## Test 2: ATM Straddle (Multi-Leg)

```bash
./target/release/quantlaxmi-india backtest-kitesim \
  --replay tests/fixtures/india_kitesim/phase27ind_smoke/quotes_straddle.jsonl \
  --orders tests/fixtures/india_kitesim/phase27ind_smoke/orders_straddle_test.json \
  --out /tmp/straddle_results \
  --latency-ms 150
```

**Expected Output** (`/tmp/straddle_results/report.json`):
- `orders_total`: 2
- `legs_total`: 4
- `legs_filled`: 4
- `rollbacks`: 0
- `timeouts`: 0
- `total_pnl`: -174.00

## What These Tests Validate

1. **Replay ordering**: Quotes consumed in timestamp order
2. **Fill model**: Market orders cross bid/ask correctly
3. **Slippage**: ~12-15 bps for market orders (reasonable)
4. **Multi-leg execution**: All legs of composite order fill
5. **PnL computation**: Correct sign and magnitude

## Phase 27-IND Gates (Frozen)

### Data Gates
- Duration >= 3h
- Instruments >= 20
- ATM CE ticks >= 10,000 (or >= 7,500 minimum)
- ATM PE ticks >= 10,000 (or >= 7,500 minimum)
- Median ticks across all instruments >= 3,000
- ATM CE/PE L2Present ratio >= 70%

### Evaluation Gates
- Completed trades >= 5
- Deterministic rerun: same report for same replay+orders
- Report JSON written

## CLI Reference

```
quantlaxmi-india backtest-kitesim [OPTIONS] --replay <REPLAY> --orders <ORDERS>

Options:
  --qty-scale <N>       Fixed-point quantity scale [default: 1]
  --strategy <NAME>     Strategy label for report metadata
  --replay <PATH>       Path to replay quotes (JSONL of QuoteEvent)
  --orders <PATH>       Path to orders JSON (MultiLegOrder list)
  --intents <PATH>      Path to intents JSON (scheduled timestamps)
  --depth <PATH>        Path to depth replay (DepthEvent JSONL) for L2 mode
  --out <DIR>           Output directory [default: artifacts/kitesim]
  --timeout-ms <MS>     Atomic execution timeout [default: 5000]
  --latency-ms <MS>     Simulated placement latency [default: 150]
  --slippage-bps <BPS>  Taker slippage [default: 0]
  --adverse-bps <BPS>   Adverse selection penalty cap [default: 0]
  --stale-quote-ms <MS> Reject if last quote older than this [default: 10000]
  --hedge-on-failure    Hedge on failure (rollback neutralization)
```
