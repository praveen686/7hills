# Strategy v0: SANOS-Gated Calendar Carry with Skew Regime Filter

**Version:** 0.1.0
**Date:** 2026-01-23
**Status:** SPEC APPROVED (Awaiting Implementation)
**Dependencies:** Phase 7 (SANOS Multi-Expiry), Phase 8.1 (Boundary Hardening)

---

## 0. Thesis

Trade short-vs-long expiry variance carry only when:
- The SANOS surface says the term structure is stable and monotone
- The calendar gap at ATM is sufficiently large relative to spreads
- Skew regime is not indicating tail stress

**This is a relative value trade. It does not require predicting direction.**

---

## 1. Instruments and Construction

### 1.1 Universe

Per underlying U ∈ {NIFTY, BANKNIFTY}:
- **Expiries:** T1, T2, T3 (policy frozen per Phase 7)
- **Strike:** ATM-nearest market strike per expiry (k_atm ≈ 1.0)

### 1.2 Position Type (Two Legs)

**Primary Trade: ATM Straddle Calendar (T1 vs T2)**

| Leg | Direction | Instrument |
|-----|-----------|------------|
| Front | Short | 1× ATM straddle on T1 |
| Back | Long | h× ATM straddle on T2 |

Where:
- Straddle = CE + PE at same strike
- h = vega hedge ratio (see Section 4)

**Why Straddle First:**
- Reduces directional delta sensitivity
- Calendar focuses on carry/theta vs vega differences

**Fallback (T2 Missing):**
If T2 is unavailable (e.g., BANKNIFTY early state):
- Short T1 straddle
- Long h× T3 straddle

---

## 2. Feature Inputs (from Phase 8)

All features computed per snapshot from `sanos_features` extractor:

| Feature | Description | Source |
|---------|-------------|--------|
| iv1, iv2, iv3 | ATM implied volatility per expiry | IV extraction at k_atm |
| CAL12 | Calendar gap T1→T2 | Ĉ(T2, k_atm) - Ĉ(T1, k_atm) |
| CAL23 | Calendar gap T2→T3 | Ĉ(T3, k_atm) - Ĉ(T2, k_atm) |
| TS12, TS23, TS_curv | Term structure slopes | Calendarized IV slopes |
| SK1, SK2, SK3 | Skew per expiry | (IV_high - IV_low) / (k_high - k_low) |

**All feature points are market strikes (Phase 8.1 certified).**

---

## 3. Entry Logic (Gates)

### 3.1 Certification Gates (HARD)

Enter **only if**:

```
□ Phase 7 surface for all required expiries is LP_OPTIMAL
□ Calendar monotonicity on market strikes: no violations
□ Feature points are market strikes (Phase 8.1 ensures this)
```

**If any fails → NO TRADE**

### 3.2 Liquidity Gates (HARD)

At entry time, require for each leg:

```
□ Mid exists: bid > 0 AND ask > bid
□ Relative spread constraint:
    - NIFTY:     straddle_spread < 35 bps
    - BANKNIFTY: straddle_spread < 55 bps
```

### 3.3 Carry Gates (HARD)

Define microstructure-aware minimum calendar gap:

```
CAL_min = λ × (spread_T1_straddle + spread_T2_straddle)
```

**Parameters:**
- λ = 1.5 (requires gap ≥ 1.5× total spreads)

**Gate:**
```
□ CAL12 ≥ CAL_min
```

**Rationale:** Theoretical advantage must exceed friction.

### 3.4 Regime Gates (SOFT)

Avoid tail-risk regimes with steepening skew and inverted term structure.

```
□ Term structure inversion not extreme:
    (iv1 - iv3) ≤ 0.04  (4 vol points max inversion)

□ Skew stress filter:
    min(SK1, SK2, SK3) ≥ SK_min
    SK_min = -0.80
```

**Note:** Current session SK2 ≈ -0.27, well within bounds.

---

## 4. Position Sizing

### 4.1 Hedge Ratio h (Vega-Neutral)

Compute per expiry:
```
Vega(Tj) = Vega(CE) + Vega(PE)  at ATM strike
```

Use Black-Scholes Greeks with F_j, iv_j, T_j, K.

Hedge ratio:
```
h = Vega(T1) / Vega(T2)
```

**Result:** Portfolio vega ≈ 0

### 4.2 Risk Cap

Define max risk per trade in premium terms:

```
RiskBudget = R bps of forward notional
R = 5-10 bps (initial)
```

Convert to lots using:
- Lot size (NIFTY: 25, BANKNIFTY: 15)
- Option multiplier
- Straddle premium

**Hard cap:** Max lots per underlying per session

---

## 5. Execution Logic (Microstructure-Safe)

### 5.1 Order Type

**Passive-to-mid logic:**
1. Place limit orders at mid or mid ± 0.1×spread
2. Allow 2-3 retries, then cancel
3. If not filled within time window (10-15 seconds), skip entry

### 5.2 Legging Risk Management

**Sequential execution (back-leg first):**
1. Enter long back leg first (T2 straddle)
2. Then enter short front leg (T1 straddle)

**If second leg fails:**
- Immediately unwind first leg (same logic)

---

## 6. Exit Logic

### 6.1 Time-Based Exit (Default)

Close by:
- t_exit = 15-30 minutes before market close, **OR**
- When T1 enters very last hour on expiry day (gamma risk)

### 6.2 Profit/Stop Exits (v0)

Compute mark-to-mid PnL.

| Exit Type | Threshold |
|-----------|-----------|
| Take Profit | +1.0× estimated friction |
| Stop Loss | -2.0× estimated friction |

**Symmetric design avoids "death by a thousand cuts."**

---

## 7. Cost Model

### 7.1 Per-Fill Estimation

For each fill, estimate:
- Half-spread cost per leg
- Exchange fees + brokerage + STT

### 7.2 Minimum Viable Model

```
friction_bps = sum(half_spreads) + fixed_bps
```

**Observed spreads (Phase 8 session):**
- NIFTY ATM: 18-31 bps
- BANKNIFTY ATM: 25-40 bps (estimated)

---

## 8. Backtest / Paper Integration

### 8.1 Inputs

- WAL ticks → orderbook snapshots
- SANOS surfaces per snapshot (Phase 7)
- Phase 8 features per snapshot

### 8.2 Engine Loop (Deterministic)

At each decision timestamp (every 60s):

```
1. Load latest features from sanos_features
2. Evaluate all gates (certification, liquidity, carry, regime)
3. If entry conditions met:
   a. Compute hedge ratio h
   b. Generate orders (back-leg first)
   c. Simulate fills using book model
4. Log trades + gate pass/fail reasons
5. Evaluate exit conditions for open positions
```

---

## 9. Implementation Deliverables

### 9.1 Core Files

| File | Description |
|------|-------------|
| `kubera-options/src/strategies/calendar_carry.rs` | Strategy logic: `evaluate(features, spreads, time) -> Decision` |
| `quantlaxmi-runner-india/src/bin/run_calendar_carry.rs` | Replay → features → strategy → paper fills |
| `docs/STRATEGY_V0_CALENDAR_CARRY.md` | This document |

### 9.2 Decision Enum

```rust
pub enum StrategyDecision {
    NoTrade { reason: GateFailure },
    Enter {
        underlying: String,
        front_expiry: String,
        back_expiry: String,
        front_lots: i32,  // negative = short
        back_lots: i32,   // positive = long
        hedge_ratio: f64,
    },
    Exit {
        reason: ExitReason,
    },
    Hold,
}

pub enum GateFailure {
    CertificationFailed(String),
    LiquidityInsufficient { leg: String, spread_bps: f64 },
    CarryInsufficient { cal12: f64, cal_min: f64 },
    RegimeStress { metric: String, value: f64 },
}

pub enum ExitReason {
    TakeProfit { pnl_bps: f64 },
    StopLoss { pnl_bps: f64 },
    TimeExit,
    GammaRisk,
}
```

---

## 10. Frozen Policy Parameters

```rust
// STRATEGY V0 POLICY — FROZEN (2026-01-23)
pub const LAMBDA: f64 = 1.5;                    // Calendar gap multiple
pub const NIFTY_SPREAD_CEILING_BPS: f64 = 35.0;
pub const BANKNIFTY_SPREAD_CEILING_BPS: f64 = 55.0;
pub const TS_INVERSION_MAX: f64 = 0.04;         // 4 vol points
pub const SKEW_STRESS_MIN: f64 = -0.80;
pub const TAKE_PROFIT_MULT: f64 = 1.0;          // × friction
pub const STOP_LOSS_MULT: f64 = 2.0;            // × friction
pub const DECISION_INTERVAL_SECS: u64 = 60;
pub const EXIT_MINUTES_BEFORE_CLOSE: u64 = 30;
```

---

## 11. Audit Fields (per trade log)

Every trade must log:

```json
{
  "ts": "2026-01-23T10:30:00Z",
  "underlying": "NIFTY",
  "decision": "Enter",
  "gates": {
    "certification": "PASS",
    "liquidity_t1": { "spread_bps": 22.5, "pass": true },
    "liquidity_t2": { "spread_bps": 28.1, "pass": true },
    "carry": { "cal12": 0.00578, "cal_min": 0.00450, "pass": true },
    "regime_ts": { "iv1_minus_iv3": 0.020, "pass": true },
    "regime_skew": { "min_sk": -0.27, "pass": true }
  },
  "position": {
    "front_expiry": "26JAN",
    "back_expiry": "26203",
    "front_lots": -2,
    "back_lots": 3,
    "hedge_ratio": 1.47
  },
  "features_snapshot": {
    "iv1": 0.1079,
    "iv2": 0.1076,
    "cal12": 0.00578,
    "sk1": -0.02,
    "sk2": -0.27
  }
}
```

---

## 12. Risk Disclosures

1. **Gamma risk on expiry day:** Front-month straddle has explosive gamma near expiry. Mandatory exit 60 minutes before close on expiry day.

2. **Legging risk:** Sequential execution means exposure during fill window. Mitigated by back-leg-first ordering.

3. **Model risk:** SANOS surface may not perfectly reflect true fair value. Variance floor (V_min) introduces smoothing.

4. **Liquidity withdrawal:** Spread ceilings may not trigger if liquidity collapses intraday. Consider volatility-adjusted spread gates in v1.

---

**Certified by:** SANOS Strategy Pipeline
**Version:** v0.1.0 (Calendar Carry with Skew Regime Filter)
