# Backtest Analysis Report - 2026-01-30

## Executive Summary

Backtest of the `funding_bias` strategy against crypto capture data revealed:
- **Strategy logic is correct** - entry/exit conditions work as designed
- **Previous capture (perp_20260129) was incomplete** - WebSocket dropped after 9 hours
- **Current capture (perp_20260130) is healthy** - all streams flowing with resilient reconnect

## Capture Sessions

### Previous Session: perp_20260129_053328 (INCOMPLETE)

| Stream | Events | Start | End | Duration | Status |
|--------|--------|-------|-----|----------|--------|
| spot_quotes | 2,397,185 | 05:33:29 | 14:52:45 | **9.3h** | STALLED |
| perp_depth | 43,805 | 05:33:29 | 06:48:19 | **1.25h** | STALLED |
| funding | 26,050 | 05:33:30 | 03:16:00+1d | 21.7h | OK |

**Root cause**: WebSocket "Connection reset without closing handshake" - no auto-reconnect.
**Impact**: Price data (spot_quotes, perp_depth) stopped early, limiting backtest window.

### Current Session: perp_20260130_034709 (ACTIVE)

| Stream | Events (1.5h) | Rate/hour | Projected 48h | Status |
|--------|---------------|-----------|---------------|--------|
| spot_quotes | 675,108 | 443,744 | ~21.3M | FLOWING |
| perp_depth | 52,093 | 34,240 | ~1.6M | FLOWING |
| funding | 1,822 | 1,197 | ~57K | FLOWING |

**Process**: PID 936603, running since 2026-01-30 03:47 UTC
**Expected completion**: 2026-02-01 03:47 UTC (48h total)
**Resilient features**: Auto-reconnect, liveness watchdog, depth re-sync

## Backtest Results

### Test Configuration

```toml
# /tmp/funding_bias_spot.toml
threshold_mantissa = 25          # 0.25 bps entry threshold
threshold_exponent = -6
position_size_mantissa = 10000000  # 0.1 BTC
qty_exponent = -8
price_exponent = -2
trade_on_spot_quotes = true      # NEW: enables spot quote trading
```

### Results: funding_bias with spot prices

| Metric | Value |
|--------|-------|
| Events processed | 2,467,040 |
| Decisions | 1 |
| Fills | 1 (short entry) |
| Entry price | $88,233.59 |
| Unrealized PnL | **+$60.52** |
| Return | +0.61% |

### Trade Analysis

**Entry (05:33:30 UTC)**:
- Funding rate: 3287 (exp -8) = 0.33 bps
- Threshold: 25 (exp -6) = 0.25 bps
- Condition: 0.33 > 0.25 → **SHORT ENTRY**

**No Exit**:
- Exit band: 12.5 (exp -6) = 0.125 bps
- Minimum funding during price data window: ~0.53 bps (at 14:52)
- Minimum funding overall: 0.086 bps (at 18:53, after spot_quotes stopped)
- Position remained open because exit condition never met during price data coverage

### Funding Rate Timeline

```
Time (UTC)    Funding (bps)   Event
05:33:30      0.33            SHORT ENTRY (funding > 0.25 threshold)
14:52:45      0.53            spot_quotes STOPPED (WebSocket drop)
18:53:00      0.086           Funding minimum (no price data to trade)
03:16:00+1d   0.23            funding stream ended
```

## Code Changes Made

### 1. Strategy Enhancement: `trade_on_spot_quotes` flag

**File**: `crates/quantlaxmi-strategy/src/strategies/funding_bias.rs`

Added configuration option to trade on SpotQuote events when perp depth data is incomplete:

```rust
/// Allow trading on SpotQuote events (in addition to PerpQuote/PerpDepth).
/// Useful when perp depth data is incomplete but spot quotes are available.
#[serde(default)]
pub trade_on_spot_quotes: bool,
```

### 2. CLI Enhancement: `--use-spot-prices` flag

**File**: `crates/quantlaxmi-runner-crypto/src/lib.rs`

Added CLI flag to use spot prices for exchange execution:

```rust
/// Use spot prices for execution instead of perp prices.
#[arg(long, default_value_t = false)]
use_spot_prices: bool,
```

## Recommendations

### Immediate Actions
1. **Monitor current capture** - Check `wc -l data/perp_sessions/perp_20260130_034709/BTCUSDT/*.jsonl` periodically
2. **Verify all streams stay in sync** - All three should have similar latest timestamps
3. **Check reconnect logs** - `tail -f /tmp/capture_resilient_v2.log` for any reconnect events

### Strategy Tuning
For future backtests with complete data:

| Threshold | Exit Band | Expected Behavior |
|-----------|-----------|-------------------|
| 25 (0.25 bps) | 12.5 | Conservative - fewer trades, hold through noise |
| 30 (0.30 bps) | 10 | Moderate - trigger when funding drops below 0.10 bps |
| 20 (0.20 bps) | 8 | Aggressive - more entries, tighter exits |

### Data Quality Checklist for 48h Capture

- [ ] All three streams have matching end timestamps (within 1 minute)
- [ ] No large gaps in any stream (check for timestamp jumps > 1 minute)
- [ ] Funding rate covers at least one 8-hour funding interval
- [ ] perp_depth maintains continuous L2 orderbook data
- [ ] File sizes grow proportionally over time

## Commands Reference

```bash
# Check capture status
ps aux | grep 936603

# Monitor event counts
watch -n 60 'wc -l data/perp_sessions/perp_20260130_034709/BTCUSDT/*.jsonl'

# Check latest timestamps
for f in data/perp_sessions/perp_20260130_034709/BTCUSDT/*.jsonl; do
  echo "$f: $(tail -1 $f | jq -r '.ts')"
done

# Run backtest with spot prices
./target/release/quantlaxmi-crypto backtest \
  --segment-dir data/perp_sessions/perp_20260130_034709 \
  --strategy funding_bias \
  --strategy-config /tmp/funding_bias_spot.toml \
  --use-sdk \
  --use-spot-prices \
  --output-json /tmp/backtest_results.json
```

## Appendix: Event Kind Mapping

| File | EventKind | Strategy Filter |
|------|-----------|-----------------|
| spot_quotes.jsonl | SpotQuote | `trade_on_spot_quotes = true` |
| perp_depth.jsonl | PerpDepth | Always accepted |
| perp_quotes.jsonl | PerpQuote | Always accepted |
| funding.jsonl | Funding | Updates funding rate only |

The strategy updates its internal funding rate from Funding events, but only makes trading decisions when it receives a price event (PerpQuote, PerpDepth, or SpotQuote if enabled).
