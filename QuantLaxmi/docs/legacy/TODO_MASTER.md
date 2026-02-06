# QuantLaxmi – Master TODO Plan

**As of:** Jan 24, 2026
**Scope:** India (FNO) + Crypto (24×7)
**Objective:** Parallelize progress without compromising statistical rigor

---

## PART A — INDIA TRACK (DATA-GATED, DISCIPLINED)

### A0. Current Status (Frozen Facts)

| Item | Value |
|------|-------|
| Strategy | Calendar Carry (SANOS-gated) |
| Alpha version | Alpha-1.3 (Market-Realistic) |
| Best config | holding=300s, max_quote_age_ms=1000, fills=bid/ask |
| Sessions with signals | 1 |
| Total signals | 5 |
| **Verdict** | Engineering complete, research blocked by data |

---

### A1. What We WILL Do Now (Jan 24–26)

These are non-controversial, non-overfitting tasks.

#### A1.1 Freeze Alpha-1.3 (MANDATORY)

- **Status:** ✅ DONE
- **Tag:** `alpha_1_3_frozen`
- **No logic changes allowed** until ≥50 trades
- **Only allowed edits:**
  - Logging
  - Diagnostics
  - Performance (non-semantic)

**Outcome:** Prevents self-sabotage.

#### A1.2 India Session Readiness Checklist

- **Status:** TODO
- Create `INDIA_SESSION_CHECKLIST.md`:
  - [ ] Capture start ≥09:25 IST
  - [ ] Capture end ≥14:30 IST
  - [ ] Disk free ≥20GB
  - [ ] Zerodha login stable
  - [ ] No process restarts
  - [ ] Clock sync verified

**Outcome:** Each session is either valid or discarded immediately.

#### A1.3 Automated Post-Session Audit (HIGH VALUE)

- **Status:** ✅ DONE (batch_score_calendar_carry.sh)
- Outputs:
  - Signals generated
  - Trades executed
  - Dropped trades (by reason)
  - Net PnL
  - Quote age distribution

**Outcome:** Instantly know whether the day produced usable research data.

#### A1.4 Define India Promotion Gates (Write Only)

- **Status:** TODO
- Document (do not tune):

| Gate | Criterion |
|------|-----------|
| Alpha-2 promotion | ≥50 trades, ≥7 sessions |
| Alpha rejection | Worst session > −25% of total PnL |
| Alpha revision | No single-day PnL dominance >40% |

**Outcome:** Removes hindsight bias later.

---

### A2. What We CANNOT Do (Until Data Exists)

🚫 **Explicitly forbidden until ≥50 trades:**

- Parameter tuning
- ML / regime logic
- Capital sizing
- Strategy comparison
- Sharpe analysis

**This is not negotiable.**

---

### A3. What Must Happen on Every India Trading Day (From Jan 27)

#### A3.1 Mandatory Daily Flow

**During market hours:**
- Full tick capture (no gaps)

**After market close:**
1. Run `run_calendar_carry`
2. Verify `signals.jsonl`
3. Run scoring grid:
   - holding: 300s / 600s
   - staleness: 500 / 1000 / ∞
4. Append results to rolling ledger

**Outcome:** Each day adds information, not noise.

---

## PART B — CRYPTO TRACK (UNBLOCKED, ACCELERATED)

Crypto is where real research velocity happens.

### B0. Strategic Intent

Crypto is **not** a copy of India logic. It is used to:

- Validate microstructure hypotheses
- Stress-test execution logic
- Generate statistically meaningful samples fast

---

### B1. Immediate Crypto TODO (START NOW)

#### B1.1 Crypto Capture Standardization

- **Status:** TODO
- **Symbols:**
  - BTCUSDT (mandatory)
  - ETHUSDT (mandatory)
- **Capture:**
  - Full depth / bookTicker
  - Trades if available
- **Session length:** Rolling 24h windows
- **Storage:** Same `data/sessions/` layout

**Outcome:** Structural parity with India.

#### B1.2 Crypto Calendar-Carry Analog (Core Research Task)

| India Concept | Crypto Equivalent |
|---------------|-------------------|
| Monthly expiry | Perp vs dated futures |
| Front / Back | Near vs far funding |
| Carry | Funding + basis |

**Tasks:**
1. Define "calendar" in crypto terms
2. Implement signal logic (clean, minimal)
3. No reuse of India heuristics blindly

**Outcome:** True research, not porting.

#### B1.3 Crypto Scoring Engine (Reuse 80%)

- **Status:** TODO
- **Reuse:**
  - Scoring infra
  - Shift tests
  - Quote age logic
  - No-fill rules
- **Adjust:**
  - Funding inclusion
  - Perp execution rules
  - Continuous time (no expiry gaps)

**Outcome:** Apples-to-apples robustness tests.

---

### B2. Crypto Experiments That ARE Allowed Early

Because crypto is abundant:

✅ Holding time sweeps
✅ Execution stress tests
✅ Slippage sensitivity
✅ Latency sensitivity
✅ Multi-position concurrency

🚫 **Still forbidden:**
- ML without labels
- Capital scaling without stability

---

### B3. Crypto Promotion Gates (Faster, Still Strict)

| Gate | Criterion |
|------|-----------|
| Minimum trades | ≥500 |
| Minimum duration | ≥7 rolling days |
| Max drawdown | < 30% of gross |
| Funding dominance | < 50% |

Crypto teaches behavior, not just PnL.

---

## PART C — CONVERGENCE PLAN (India ↔ Crypto)

### C1. What Crypto Can Validate for India

- Exit timing sensitivity
- Spread impact realism
- Quote staleness tolerance
- Execution fragility

Crypto results **inform**, but do not override India.

### C2. What India Validates for Crypto

- Regulatory realism
- Liquidity cliffs
- Expiry behavior
- True carry economics

They cross-check each other.

---

## FINAL EXECUTION PRIORITY

### This Week (Jan 24–26)

1. ✅ Finish India TODO A1.1, A1.3
2. ⏳ Start Crypto capture
3. ⏳ Define crypto calendar logic

### From Jan 27 Onwards

- **India:** discipline
- **Crypto:** velocity

---

## One-Sentence North Star

> **India demands patience. Crypto rewards rigor. Together, they prevent self-deception.**
