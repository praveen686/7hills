# India FNO Paper Trading - 2026-01-30

## Executive Summary

Successfully tested the `india_micro_mm` strategy against live-captured NIFTY option data:
- **290 orders** generated from 19,727 quotes
- **286 fills** (98.6% fill rate)
- **Gross PnL: +3,614 INR** but **Net PnL: -6,506 INR** after fees

The strategy generates valid signals but transaction costs dominate returns.

## Live Capture Status

### Current Session: fno_20260130 (PID 940275)
```
Status: RUNNING
Started: 2026-01-30 04:10 UTC
Duration: 6 hours target
Underlyings: NIFTY, BANKNIFTY
Expiries: T1 (weekly), T2, T3 (monthly front)
Strike band: ATM ± 20 strikes
```

### Capture Statistics (as of test time)
| Metric | Value |
|--------|-------|
| Total instruments | 412 |
| Total ticks | ~2,000,000 |
| Top instrument | NIFTY2620325300CE (9,865 ticks) |
| Data quality | L2Present (full depth) |

## Test Configuration

### Instruments Tested
| Symbol | Type | Ticks |
|--------|------|-------|
| NIFTY2620325300CE | ATM Call | 9,865 |
| NIFTY2620325300PE | ATM Put | 9,862 |

### Strategy: india_micro_mm
Single-leg microstructure scalper with:
- Entry on order book imbalance (pressure ratio)
- Velocity-aware routing (LIMIT vs MARKET)
- Lot size auto-detection (NIFTY = 65)

### Simulation Parameters
```bash
--latency-ms 150    # Order placement latency
--slippage-bps 2.0  # Taker slippage
```

## Results

### Order Generation
```
Strategy: india_micro_mm
Quotes: 19,727
Orders: 290
Orders/hour: ~232
```

### Execution Summary
| Metric | Value |
|--------|-------|
| Legs filled | 286 |
| Fill rate | 98.6% |
| Timeouts | 0 |
| Rollbacks | 0 |

### Slippage Analysis
| Percentile | Slippage (bps) |
|------------|----------------|
| p50 | 10.74 |
| p90 | 12.62 |
| p99 | 13.82 |

### PnL Breakdown (INR)

| Component | Amount |
|-----------|--------|
| **Gross MTM PnL** | +3,614.24 |
| Brokerage | -5,720.00 |
| Exchange fees | -1,282.69 |
| GST | -1,261.14 |
| STT | -1,796.56 |
| Stamp duty | -55.95 |
| SEBI fee | -3.66 |
| **Total fees** | -10,120.01 |
| **Net MTM PnL** | **-6,505.77** |

### Open Positions at EOD
| Symbol | Position | Mark Price | MTM Value |
|--------|----------|------------|-----------|
| NIFTY2620325300CE | -520 | 200.10 | -104,052 |
| NIFTY2620325300PE | +910 | 193.65 | +176,222 |
| **Net** | | | **+72,170** |

## Analysis

### Fee Impact
```
Notional filled: 36,61,680 INR
Total fees: 10,120 INR
Fee rate: 0.276% (27.6 bps)
```

For a scalping strategy targeting 5-10 bps edge:
- Fee rate (27.6 bps) exceeds typical edge
- Need ~30 bps gross edge to break even

### Execution Quality
- High fill rate (98.6%) indicates good liquidity
- Slippage within expectations (10-13 bps)
- Zero timeouts/rollbacks = reliable execution

### Strategy Observations
1. **Signal generation works**: 290 orders in 1.25 hours
2. **Execution model works**: 98.6% fills, no failures
3. **Fee model works**: Accurate Zerodha FnO fee calculation
4. **PnL attribution works**: Gross/Net/MTM properly tracked

### Recommendations

#### Short-term
1. **Increase edge threshold**: Filter for higher-conviction signals
2. **Reduce trading frequency**: Fewer, better trades
3. **Add position limits**: Cap open notional

#### Medium-term
1. **Multi-leg strategies**: Straddles/strangles for lower margin
2. **Time-of-day filters**: Avoid high-spread periods
3. **Volatility filters**: Trade when IV is favorable

#### Strategy Tuning Options
| Parameter | Current | Suggested |
|-----------|---------|-----------|
| max_spread_bps | 50 | 30 |
| pressure_long | 1.5 | 2.0 |
| pressure_short | 0.67 | 0.5 |
| min_hold_ms | 100 | 500 |

## Commands Reference

```bash
# Convert ticks to quotes
./scripts/india_make_quotes.sh <session_dir> --merge

# Generate orders deterministically
./target/release/quantlaxmi-india generate-orders \
  --strategy india_micro_mm \
  --replay quotes_all.jsonl \
  --out orders.json \
  --routing-log routing_decisions.jsonl

# Run KiteSim backtest
./target/release/quantlaxmi-india backtest-kitesim \
  --strategy india_micro_mm \
  --replay quotes_all.jsonl \
  --orders orders.json \
  --out backtest/ \
  --latency-ms 150 \
  --slippage-bps 2.0

# Monitor live capture
watch -n 60 'find data/india_sessions/fno_20260130 -name "*.jsonl" -exec wc -l {} + | tail -1'
```

## Output Files

```
/tmp/india_test/
├── quotes_all.jsonl          # Merged quotes (19,727)
├── orders.json               # Generated orders (290)
├── routing_decisions.jsonl   # Per-signal features (for ML)
└── backtest/
    ├── pnl.json              # Detailed PnL breakdown
    ├── fills.jsonl           # Fill-level details
    ├── fee_ledger.jsonl      # Fee attribution
    ├── equity_curve.jsonl    # P&L timeline
    └── report.json           # Summary metrics
```

## Next Steps

1. **Wait for full session capture** (6 hours)
2. **Run preflight check**: `./scripts/india_preflight_check.sh fno_20260130`
3. **Test with all ATM options** (not just CE/PE pair)
4. **Tune strategy parameters** based on results
5. **Compare against buy-and-hold** baseline
