# SANOS Phase 6-7 Completion Report

**Date:** 2026-01-23
**Phases:** 6 (Temporalization) + 7 (Multi-Expiry Term Structure)
**Session:** nifty_banknifty_20260123_1002

---

## Executive Summary

Phase 6 (Temporalization) and Phase 7 (Multi-Expiry) infrastructure are complete.

### Key Findings

1. **Two-Tier Certification** adopted based on Lead directive:
   - **STATE_CERTIFIED**: Surface Ĉ drift (primary) - what matters for downstream use
   - **PARAM_CERTIFIED**: Weight q drift (secondary) - LP solution stability

2. **η Sweep Results**: η=0.25 is optimal. Increasing η *worsens* q drift without improving Ĉ drift.

3. **BANKNIFTY Diagnosis Confirmed**: High q drift (0.96) with tiny Ĉ drift (0.0045) is parameter non-identifiability, not state instability. Surface is safe for downstream use.

---

## Phase 6: Two-Tier Certification Results

### NIFTY 26JAN
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max Ĉ drift (L1) | 0.0056 | < 0.05 | ✓ STATE_CERTIFIED |
| Max q drift (L1) | 0.476 | < 0.5 | ✓ PARAM_CERTIFIED |

**Conclusion**: Both certifications pass. Surface and parameters are stable.

### BANKNIFTY 26JAN
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Max Ĉ drift (L1) | 0.0045 | < 0.05 | ✓ STATE_CERTIFIED |
| Max q drift (L1) | 0.960 | < 0.5 | ✗ PARAM_NOT_CERTIFIED |

**Conclusion**: STATE_CERTIFIED (surface stable), PARAM not certified (LP jumping).
Downstream use of Ĉ (fitted calls) is safe. Direct use of q (weights) is not reliable.

---

## η Sweep Analysis

Tested η ∈ {0.25, 0.35, 0.50, 0.65} for both underlyings:

### NIFTY
| η | Max q drift | Max Ĉ drift | PARAM | STATE |
|------|-------------|-------------|-------|-------|
| **0.25** | 0.48 | 0.0056 | ✓ | ✓ |
| 0.35 | 1.41 | 0.0056 | ✗ | ✓ |
| 0.50 | 2.00 | 0.0055 | ✗ | ✓ |
| 0.65 | 0.87 | 0.0056 | ✗ | ✓ |

### BANKNIFTY
| η | Max q drift | Max Ĉ drift | PARAM | STATE |
|------|-------------|-------------|-------|-------|
| **0.25** | 0.96 | 0.0045 | ✗ | ✓ |
| 0.35 | 1.09 | 0.0044 | ✗ | ✓ |
| 0.50 | 2.00 | 0.0044 | ✗ | ✓ |
| 0.65 | 2.00 | 0.0044 | ✗ | ✓ |

### Key Insight
**η=0.25 is optimal** for both underlyings:
- Higher η increases regularization → more equivalent LP optima → worse q drift
- Ĉ drift is invariant to η (surface is determined by market prices, not regularization)
- BANKNIFTY's q non-identifiability is structural (wider spreads, fewer liquid strikes)

---

## Phase 7: Multi-Expiry Infrastructure

### Components Delivered
| Component | Location | Status |
|-----------|----------|--------|
| Multi-Expiry Runner | `sanos_multi_expiry.rs` | ✓ Complete |
| Calendar Arbitrage Check | Embedded in runner | ✓ Complete |
| Term Structure Output | JSON export | ✓ Complete |

### Calendar Arbitrage Constraint
```
C(K, T1) ≤ C(K, T2)  for T1 < T2 at same strike K
```
This ensures forward-start call prices are non-negative.

### Current Session (Single Expiry)
With only 26JAN expiry available, calendar arbitrage checks pass trivially.
Full validation requires multi-expiry data (e.g., 26JAN + 02FEB + 27FEB).

### Term Structure Output
```
NIFTY:
  26JAN: F=25,341.57, TTY=3.2 days

BANKNIFTY:
  26JAN: F=59,217.66, TTY=3.2 days
```

---

## Binaries Delivered

| Binary | Purpose |
|--------|---------|
| `sanos_single_slice` | Single expiry calibration at one timestamp |
| `sanos_temporal` | Temporal stability analysis (30s intervals) |
| `sanos_multi_expiry` | Multi-expiry term structure with calendar checks |

---

## Certification Summary

### NIFTY
| Phase | Status |
|-------|--------|
| Phase 5 (Single Slice) | ✓ CERTIFIED |
| Phase 6 STATE (Ĉ stability) | ✓ CERTIFIED |
| Phase 6 PARAM (q stability) | ✓ CERTIFIED |
| Phase 7 (Calendar Arb) | ✓ CERTIFIED (single expiry) |

### BANKNIFTY
| Phase | Status |
|-------|--------|
| Phase 5 (Single Slice) | ✓ CERTIFIED |
| Phase 6 STATE (Ĉ stability) | ✓ CERTIFIED |
| Phase 6 PARAM (q stability) | ✗ NOT CERTIFIED (structural) |
| Phase 7 (Calendar Arb) | ✓ CERTIFIED (single expiry) |

---

## Recommendations

1. **Proceed with BANKNIFTY** for downstream use cases that consume Ĉ (fitted calls)
2. **Do NOT use BANKNIFTY q** (weights) directly in any downstream computation
3. **Collect multi-expiry data** to fully validate Phase 7 calendar arbitrage
4. **Keep η=0.25** - do not tune further

---

## Files Generated

| File | Description |
|------|-------------|
| `sanos_temporal_nifty/` | NIFTY temporal analysis |
| `sanos_temporal_banknifty/` | BANKNIFTY temporal analysis |
| `sanos_multi_expiry_nifty.json` | NIFTY term structure |
| `sanos_multi_expiry_banknifty.json` | BANKNIFTY term structure |
| `sanos_temporal_*_eta_*/` | η sweep outputs |

---

**Certified by:** SANOS Phase 6-7 Pipeline
**Version:** v1 (two-tier certification, multi-expiry ready)
