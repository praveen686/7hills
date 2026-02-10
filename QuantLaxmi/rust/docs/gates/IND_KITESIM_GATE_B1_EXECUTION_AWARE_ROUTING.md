# IND KiteSim Gate B1: Execution-Aware Routing (LIMIT vs MARKET)

**Date**: 2026-01-29
**Status**: COMPLETE
**Objective**: Convert execution from uncontrolled tax into a controlled cost by routing LIMIT vs MARKET based on predicted adverse selection.

## Background (from Phase 27)

- `execution_tax_bps` is the regression target (-167.82 bps baseline)
- M2.1 established price-through dominance (adverse selection), not queue advantage
- Strategy is directional alpha with maker-style execution
- **Goal**: Reduce execution tax while preserving directional alpha

## Definitions

### Slippage / Edge (Simulator truth)
```rust
// Buy:  slip_bps = (fill_px - mid) / mid * 10000  // positive = worse than mid
// Sell: slip_bps = (mid - fill_px) / mid * 10000  // positive = worse than mid
// mean_edge_bps = -mean(slip_bps)  // flip: positive edge = good
```

### Execution Tax
```
execution_tax_bps = (mtm_pnl - edge_adjusted_pnl) / notional_filled * 10000
                  = -mean_edge_bps (by construction)
```

## B1 Routing Rule (Deterministic Baseline)

### Features computed at signal time:
- `spread_bps` = (ask - bid) / mid * 10000
- `mid_vel_bps` = (mid - prev_mid) / prev_mid * 10000
- `signal_strength` = distance beyond pressure threshold (clamped >= 0)

### Routing decision:
```
Route MARKET if:
  spread_bps <= spread_bps_market_max AND
  (abs(mid_vel_bps) >= vel_bps_market_min OR signal_strength >= signal_strength_market_min)

Route LIMIT if:
  spread_bps <= spread_bps_limit_max AND NOT(MARKET conditions)

SKIP if:
  spread_bps > spread_bps_market_max OR
  (spread_bps > spread_bps_limit_max AND NOT(MARKET conditions))
```

### Initial Thresholds (Baseline)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| spread_bps_limit_max | 80 | Post only on reasonable spread |
| spread_bps_market_max | 120 | Don't cross when spread is insane |
| vel_bps_market_min | 25 | Fast move = posting gets picked off |
| signal_strength_market_min | 0.05 | Strong signal = pay cost for certainty |

## Implementation

### Files Modified
- `crates/quantlaxmi-runner-india/src/order_generation.rs`
  - Added `RoutingCfg` struct with baseline thresholds
  - Added `mid_f64()` and `mid_vel_bps()` helpers
  - Added `last_mid`, `last_quote_ts` to `SignalState`
  - Updated `generate_micro_mm_orders()` with B1 routing logic

### Routing Logic (Pseudocode)
```rust
if should_long {
    if choose_market_long {
        order_type = Market, price = None
    } else if spread <= limit_max {
        order_type = Limit, price = bid
    } else {
        skip // spread between limit_max and market_max
    }
}
```

## Acceptance Criteria (Gate-grade)

### A. Time consistency
- [x] All routing features computed from quote at signal time only
- [x] `last_mid` updated per quote for velocity estimation
- [x] No lookahead fields used

### B. Economic outcome
- [x] Execution tax becomes less negative: -103 bps > -168 bps (+65 bps improvement)
- [x] MTM improved: Rs 148,664 > Rs 62,700 (+137%)
- [x] Fill rate acceptable: 818 vs 541 (+51%)

### C. Logging / Audit
- [x] Routing summary logged: `LIMIT={}, MARKET={}, total={}`
- [x] **B1.3 COMPLETE**: `routing_decisions.jsonl` sidecar with per-decision feature vectors

## Results

### Baseline (Phase 27, all LIMIT)
```json
{
  "mtm_pnl": 62700.0,
  "execution_tax_bps": -167.82,
  "legs_filled": 541,
  "routing": "100% LIMIT"
}
```

### B1 Results (2026-01-29)
```json
{
  "mtm_pnl": 148663.5,
  "execution_tax_bps": -103.08,
  "edge_p50_bps": -12.1,
  "legs_filled": 818,
  "queue_consumption_fills": 127,
  "routing": { "LIMIT": 251, "MARKET": 691 }
}
```

### Improvement Summary
| Metric | Baseline | B1 | Delta |
|--------|----------|-----|-------|
| execution_tax_bps | -167.82 | -103.08 | **+65 bps** |
| mtm_pnl | Rs 62,700 | Rs 148,664 | **+137%** |
| edge_p50_bps | -127.6 | -12.1 | **+115 bps** |
| legs_filled | 541 | 818 | +51% |

## Next Steps

- **B1.2**: Train a predictor for `execution_tax_bps` using `routing_decisions.jsonl` features
- **B2**: Inventory caps + exposure symmetry
- **B3**: Horizon-labeled PnL (30s/2m/5m forward)

## B1.3 routing_decisions.jsonl Schema

```jsonl
// Line 1: run_header
{"record_type":"run_header","schema":"quantlaxmi.routing_decisions.v1","gate":"IND_KITESIM_GATE_B1",...}

// Lines 2-N: decision records
{"record_type":"decision","ts_utc":"...","symbol":"...","quote":{...},"features":{...},"decision":{...},"reason":{...},"ids":{...}}

// Last line: run_footer
{"record_type":"run_footer","counts":{"decisions":942,"limit":251,"market":691,"market_by_vel":0,"market_by_strength":691,"market_by_both":0,...}}
```

Key fields per decision:
- `features`: spread_bps, pressure, vel_bps_sec, signal_strength
- `thresholds`: config snapshot at decision time
- `decision`: order_type (Market/Limit), price
- `reason`: primary (FAST_MOVE/STRONG_SIGNAL/SPREAD_OK), flags[]
- `ids`: decision_id (sha256), order_id, leg_index

## Command Reference

```bash
# Generate orders with B1 routing + routing_decisions.jsonl sidecar
cargo run --release -p quantlaxmi-runner-india --bin quantlaxmi-india -- generate-orders \
  --strategy india_micro_mm \
  --replay data/sessions/banknifty_phase27ind_20260129_112835/quotes_all.jsonl \
  --out data/sessions/banknifty_phase27ind_20260129_112835/orders_micro_mm_b1.json \
  --routing-log data/sessions/banknifty_phase27ind_20260129_112835/routing_decisions_b1.jsonl

# Run backtest
cargo run --release -p quantlaxmi-runner-india --bin quantlaxmi-india -- backtest-kitesim \
  --replay data/sessions/banknifty_phase27ind_20260129_112835/quotes_all.jsonl \
  --orders data/sessions/banknifty_phase27ind_20260129_112835/orders_micro_mm_b1.json \
  --out data/sessions/banknifty_phase27ind_20260129_112835/micro_mm_results_b1 \
  --timeout-ms 30000 \
  --latency-ms 150 \
  --slippage-bps 0 \
  --strategy india_micro_mm_b1
```
