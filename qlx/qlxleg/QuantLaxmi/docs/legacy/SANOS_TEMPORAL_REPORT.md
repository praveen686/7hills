# SANOS Temporal Stability Report

**Date:** 2026-01-23
**Phase:** 6 - Temporalization
**Session:** nifty_banknifty_20260123_1002
**Duration:** 10 minutes (599 seconds)
**Interval:** 30 seconds
**Total Snapshots:** 20 per underlying

---

## Executive Summary

| Underlying | Feasible | Max Weight Drift | Status |
|------------|----------|------------------|--------|
| **NIFTY** | 20/20 | 0.476 | **CERTIFIED** |
| **BANKNIFTY** | 20/20 | 0.960 | **NOT CERTIFIED** |

NIFTY demonstrates temporal stability within acceptable bounds.
BANKNIFTY exhibits excessive weight drift, indicating LP solution instability.

---

## NIFTY 26JAN Analysis

### Feasibility
- Total snapshots: 20
- Feasible: 20 (100%)
- Infeasible: 0

### Drift Statistics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max density drift (L1) | 0.476 | < 0.5 | ✓ PASS |
| Mean density drift (L1) | 0.205 | - | OK |
| Max weight drift (L1) | 0.476 | < 0.5 | ✓ PASS |
| Mean weight drift (L1) | 0.205 | - | OK |

### Conditioning
| Metric | Value | Status |
|--------|-------|--------|
| Min active weights | 6 | ✓ Good (no corner solutions) |
| Mean active weights | 6.2 | ✓ Stable |
| Min positive weight | 0.000166 | ✓ No degeneracy |

### Forward Stability
| Metric | Value |
|--------|-------|
| Forward range | 25,288.12 - 25,341.57 |
| Forward span | 53.45 points |
| Max change | 0.10% |

### Martingale Constraints
All 20 snapshots satisfy:
- Σq_i = 1.000000 ✓
- Σq_i×K_i = 1.000000 ✓

### Certification
**Status: CERTIFIED**
- All stability checks passed
- No solver instabilities
- Smooth drift profile

---

## BANKNIFTY 26JAN Analysis

### Feasibility
- Total snapshots: 20
- Feasible: 20 (100%)
- Infeasible: 0

### Drift Statistics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max density drift (L1) | 0.960 | < 0.5 | ✗ FAIL |
| Mean density drift (L1) | 0.475 | - | Elevated |
| Max weight drift (L1) | 0.960 | < 0.5 | ✗ FAIL |
| Mean weight drift (L1) | 0.475 | - | Elevated |

### Conditioning
| Metric | Value | Status |
|--------|-------|--------|
| Min active weights | 5 | ⚠ Low |
| Mean active weights | 5.8 | ⚠ Lower than NIFTY |
| Min positive weight | 0.000417 | ✓ OK |

### Forward Stability
| Metric | Value |
|--------|-------|
| Forward range | 59,115.29 - 59,226.60 |
| Forward span | 111.31 points |
| Max change | 0.08% |

### Martingale Constraints
All 20 snapshots satisfy:
- Σq_i = 1.000000 ✓
- Σq_i×K_i = 1.000000 ✓

### Certification
**Status: NOT CERTIFIED**
- Large weight drift: 0.9599 exceeds 0.5 threshold
- LP solution jumping between extreme configurations

---

## Root Cause Analysis: BANKNIFTY Instability

### Hypothesis 1: Wider Bid-Ask Spreads
BANKNIFTY options have approximately 50% wider spreads than NIFTY (observed in Phase 5). Wider spreads create more ambiguity in mid-price, leading to noisier input to the LP.

### Hypothesis 2: Fewer Liquid Strikes
BANKNIFTY's 100-point strike intervals vs NIFTY's 50-point intervals mean fewer data points for the LP to anchor on.

### Hypothesis 3: Sensitivity to η
The smoothness parameter η=0.25 may be too low for BANKNIFTY's noisier data. Increasing η would regularize the solution at the cost of fit quality.

### Recommended Mitigations (for Phase 7)
1. Increase η for BANKNIFTY (e.g., η=0.35)
2. Use bid-ask weighted constraints instead of mid
3. Add temporal regularization (penalize drift from previous solution)

---

## Temporal Evolution Visualization

### Weight Drift Over Time (30-second intervals)

```
NIFTY Weight Drift (L1 norm)
Time    Drift
04:32   0.107
04:32   0.111
04:33   0.151
04:33   0.266 ← spike
04:34   0.070
04:34   0.313 ← spike
04:35   0.404 ← spike
04:35   0.160
04:36   0.075
04:36   0.123
04:37   0.244
04:37   0.293
04:38   0.365
04:38   0.072
04:39   0.134
04:39   0.157
04:40   0.241
04:40   0.128
04:41   0.476 ← max
```

### Forward Price Evolution

```
NIFTY Forward (F0)
|
25342 |                        *
25335 |                    *
25328 |                *
25321 |            *
25314 | *      *        *   *  *
25307 |   *  *    *  * * *  *
25300 |    *
25293 |   *
25286 |
      +--------------------------------
       04:31  04:34  04:37  04:40  Time
```

The forward moves smoothly within a 53-point band (~0.2% range), consistent with normal index movement over 10 minutes.

---

## Certified Temporal Window

For NIFTY 26JAN expiry:

```
SANOS_TEMPORAL_CERTIFIED {
  underlying: "NIFTY",
  expiry: "26JAN",
  start_ts: "2026-01-23T04:31:51Z",
  end_ts: "2026-01-23T04:41:51Z",
  num_snapshots: 20,
  max_density_drift: 0.476,
  max_weight_drift: 0.476,
  forward_range: (25288.12, 25341.57),
  certified: true
}
```

---

## Files Generated

| File | Description |
|------|-------------|
| `sanos_temporal_nifty/temporal_analysis.json` | Full NIFTY analysis |
| `sanos_temporal_nifty/density_evolution.csv` | NIFTY drift time series |
| `sanos_temporal_banknifty/temporal_analysis.json` | Full BANKNIFTY analysis |
| `sanos_temporal_banknifty/density_evolution.csv` | BANKNIFTY drift time series |

---

## Recommendations for Phase 7

1. **Proceed with NIFTY for multi-expiry SANOS** - temporal stability proven
2. **Investigate BANKNIFTY η tuning** before including in production
3. **Consider temporal regularization** in LP formulation:
   ```
   minimize: fitting_error + λ × ||q_t - q_{t-1}||_1
   ```
4. **Collect longer session data** (30-60 minutes) to validate stability at scale

---

## Certification Signature

**SANOS Phase 6 - Temporalization**

| Underlying | Status |
|------------|--------|
| NIFTY | ✓ CERTIFIED |
| BANKNIFTY | ✗ NOT CERTIFIED (pending η tuning) |

**Certified by:** SANOS Temporal Analysis Pipeline
**Engine:** `sanos_temporal`
**Version:** Phase 6 (single-expiry temporal stability)
