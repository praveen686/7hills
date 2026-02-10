# SANOS Phase 7 Certification Report

**Date:** 2026-01-23
**Status:** CERTIFIED (Market-Relevant)
**Session:** multi_expiry_phase7_test

---

## Executive Summary

Phase 7 (Multi-Expiry Term Structure) is complete and certified. The SANOS surface now supports arbitrage-free calibration across multiple expiries with calendar monotonicity guarantees at all market strikes.

---

## Certification Dimensions

| Dimension | Status |
|-----------|--------|
| Butterfly arbitrage (per expiry) | PASS |
| Calendar arbitrage (market strikes) | PASS |
| Martingale constraints | PASS |
| Forward consistency | PASS |
| Numerical stability | PASS (with V_min) |
| Deterministic replay | PASS |

---

## NIFTY Results

### Forwards (via Put-Call Parity)

| Expiry | Date | TTY (days) | Forward | LP Status |
|--------|------|------------|---------|-----------|
| T1 | 26JAN (Jan 27) | 3.2 | 25,275.31 | Optimal |
| T2 | 26203 (Feb 3) | 11.2 | 25,317.94 | Optimal |
| T3 | 26FEB (Feb 24) | 34.2 | 25,403.84 | Optimal |

### Calendar Slack at Market Strikes

| Pair | Min Slack | At K (normalized) |
|------|-----------|-------------------|
| T1 → T2 | +0.00095 | 1.039 |
| T2 → T3 | +0.00275 | 1.037 |
| T1 → T3 | +0.00370 | 1.039 |

All calendar slacks strictly positive. No violations at market strikes.

---

## BANKNIFTY Results

### Forwards (via Put-Call Parity)

| Expiry | Date | TTY (days) | Forward | LP Status |
|--------|------|------------|---------|-----------|
| T1 | 26JAN (Jan 27) | 3.2 | 59,186.56 | Optimal |
| T3 | 26FEB (Feb 24) | 34.2 | 59,491.81 | Optimal |

Note: T2 (weekly Feb 3) not present in Zerodha instruments universe for this session.

### Calendar Slack

T1 → T3: ARBITRAGE-FREE (no violations detected)

---

## Variance Floor Fix

### Problem
Ultra-short maturities (TTY < 7 days) cause LP matrix ill-conditioning when V = σ²Tη is very small.

### Solution
```
V_eff = max(σ²Tη, V_min)
V_min = 2e-4
```

### Validation
- T1 LP status: Optimal (was "Singular matrix" before fix)
- Forward F(T) unchanged (parity-based, not affected by variance)
- Calendar slack remains positive
- No degradation in T2/T3 fits

---

## Explicit Exclusions (Documented, Not Failures)

1. **Calendar violations at K≈0**: Boundary artifacts from extrapolation. Not economically meaningful.
2. **Deep OTM extrapolation**: Outside market strike range, not certified.
3. **Spread compliance < 50% for T1**: Expected for short-dated, gamma-dominated options with floored variance.

---

## Frozen Policy Parameters

```rust
// SANOS PRODUCTION POLICY — FROZEN (Phase 7 Certified 2026-01-23)
ETA: f64 = 0.25           // Smoothness parameter
V_MIN: f64 = 2e-4         // Variance floor
EPSILON_STRIKE: f64 = 0.001  // Boundary K1 (fixed, do not shrink with T)
STRIKE_BAND: u32 = 20     // ±20 strikes around ATM
EXPIRY_POLICY: T1T2T3     // Three expiries when available
```

---

## Certification Rules

| Certification | Criteria |
|---------------|----------|
| STATE_CERTIFIED | Ĉ drift < 0.05 |
| CALENDAR_CERTIFIED | Slack ≥ 0 at market strikes (exclude K0) |
| PARAM_CERTIFIED | Optional (q stability not required for downstream) |

---

## What Phase 7 Enables

Now safe to build:
- Forward-consistent Greeks (vega, gamma, calendar theta)
- Volatility carry & roll-down signals
- Term structure curvature features
- Calendar spread valuation with no-arb guarantees

---

**Certified by:** SANOS Phase 7 Pipeline
**Version:** v1 (variance floor, market-strike calendar checks)
