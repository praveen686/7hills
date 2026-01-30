# IND KiteSim Phase 27: Maker Correctness Gates

**Date**: 2026-01-29
**Status**: COMPLETE
**Session**: `data/sessions/banknifty_phase27ind_20260129_112835/`

## Gate Summary

| Gate | Name | Guarantee |
|------|------|-----------|
| M0 | Market→Limit conversion | Orders use `LegOrderType::Limit` with `maker_limit_price()` |
| M1 | Proper LIMIT fill semantics | Limit orders fill at limit price, not at crossed bid/ask |
| B0.2 | Stateless generator | Generator tracks only `last_signal_ts`, not phantom positions |
| B0.3 | MTM valuation | Open positions valued at conservative marks (bid for longs, ask for shorts) |
| B0.4.1 | Invalid book tracking | `invalid_book_fills` counted separately, excluded from edge stats |
| B0.4.2 | Time-consistent edge | `slip_quote_mode: "current"` - edge measured at fill time |
| M2 | Queue consumption model | LIMIT fills only when bid/ask qty decreases at price level |
| M2.1 | Queue priority | `queue_ahead_qty` tracks position in queue; fills only after working through queue |

## Key Equations

### PnL Decomposition

```
mtm_pnl = cashflow + mtm_value
        = edge_adjusted_pnl + execution_alpha

where:
  execution_alpha = -edge_cost
  edge_cost = (mean_edge_bps / 10000) * notional_filled
  edge_adjusted_pnl = mtm_pnl - edge_cost
```

### Edge Computation (time-consistent)

```rust
// In record_slip():
let mid = (bid + ask) / 2.0;
let slip_bps = match side {
    Buy  => ((fill_px - mid) / mid) * 10000.0,  // positive = worse than mid
    Sell => ((mid - fill_px) / mid) * 10000.0,  // positive = worse than mid
};
// mean_edge_bps = -mean(slip_bps)  // flip sign: positive edge = good
```

### Execution Tax (normalized)

```
execution_tax_bps = (mtm_pnl - edge_adjusted_pnl) / notional_filled * 10000
                  = -mean_edge_bps  (by construction)
```

## What Is Now INVALID to Claim

After these gates, the following claims are **provably false** and must not be made:

1. **"Positive MTM implies maker edge"** - FALSE
   MTM can be positive from directional alpha while execution edge is negative.

2. **"High fill rate proves liquidity provision"** - FALSE
   High fills often mean price moved through your limit (adverse selection).

3. **"100% favorable fills means good execution"** - FALSE (pre-M2.1)
   M2 without queue priority showed 100% favorable due to temporal artifact.

4. **"+X bps edge means we're earning spread"** - REQUIRES VERIFICATION
   Must check `queue_priority_blocked` vs `queue_consumption_fills` ratio.

## Interpretation Guide

| Condition | Interpretation |
|-----------|----------------|
| `edge_adjusted_pnl > mtm_pnl` | Direction saved me (signal > execution cost) |
| `edge_adjusted_pnl < mtm_pnl` | Execution alpha contributed (true maker edge) |
| `mean_edge_bps < 0` | Paying to get filled (adverse selection) |
| `mean_edge_bps > 0` | Earning on fills (maker edge, rare) |
| `queue_priority_blocked >> queue_consumption_fills` | True queue-based fills |
| `queue_priority_blocked << queue_consumption_fills` | Price-through fills (direction) |

## Phase 27 Results

```json
{
  "mtm_pnl": 62700.0,
  "edge_adjusted_pnl": 303049.8,
  "execution_tax_bps": -167.8,
  "edge_interpretation": "direction_saved_me",
  "maker_scorecard": {
    "mean_edge_bps": -167.8,
    "favorable_fill_pct": 9.8,
    "queue_consumption_fills": 541,
    "queue_priority_blocked": 1
  }
}
```

**Conclusion**: This is a **directional strategy with maker-style execution**, NOT a market maker.

- Directional alpha: ~Rs 303k
- Execution cost: ~Rs 240k
- Net realized: ~Rs 62.7k

## Files Modified

- `crates/quantlaxmi-options/src/kitesim.rs` - M2/M2.1 queue model, B0.4.1/B0.4.2 stats
- `crates/quantlaxmi-runner-india/src/kitesim_backtest.rs` - MTM, edge decomposition
- `crates/quantlaxmi-runner-india/src/order_generation.rs` - M0 limit orders, B0.2 stateless

## Next Research Paths

### Path B (Recommended): Directional Alpha + Execution-Aware Routing

- **B1**: LIMIT vs MARKET decision rule based on predicted adverse selection
- **B2**: Inventory caps, exposure symmetry, EOD flatten
- **B3**: Horizon-labeled PnL (30s/2m/5m forward)

### Path A (Later): True Market Making

Requires microstructure predictors:
- Queue depth modeling
- Imbalance signals
- Spread regime detection

### Path C: Hybrid Predictive Maker

Quote only when `P(edge <= 0 | state) > threshold`
