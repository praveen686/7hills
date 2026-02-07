# Arbitrage Capture Plan - Formal Audit

**Status: APPROVED — PROCEED TO IMPLEMENTATION**

---

## 0. Executive Verdict

Your plan is coherent, internally consistent, audit-safe, and correctly sequenced.
- No architectural blockers
- No infra churn required
- You are officially in the **execution + data accrual phase**

The attached arbitrage capture document is fit for production implementation as-is, with only minor clarifications, not redesign.

---

## 1. Repository & Governance Audit

| Finding | Status |
|---------|--------|
| Single authoritative system (QuantLaxmi) | PASS |
| Legacy isolation correctly enforced | PASS |
| Dual-binary model is clean and future-proof | PASS |
| Migration gates are explicit and testable | PASS |
| No hidden coupling between India/Crypto stacks | PASS |

**Audit Decision: PASS**

You have avoided the most common failure mode: parallel evolution of infra.
QuantKubera1 is properly quarantined and governed by deletion gates.

---

## 2. CI, Determinism & Compliance

| Finding | Status |
|---------|--------|
| Rust 1.93 compatibility confirmed | PASS |
| Strict clippy/fmt only on active system | PASS |
| DeterminismInfo + RunManifest enforced globally | PASS |
| SHA-256 hashing at artifact boundary | PASS |
| Paper trading is replay-safe and WAL-backed | PASS |

**Audit Decision: PASS — INSTITUTIONAL GRADE**

This places QuantLaxmi above "research toy" level. It is already audit-defensible.

---

## 3. Binance (Crypto) Readiness Review

**What You Got Exactly Right:**
- Integer mantissa discipline
- Certified replay with golden fixtures
- Bootstrap correctness test (this is rare and correct)
- Paper trading before strategy work
- VectorBT export only after determinism

This sequence is textbook correct for any serious arb system.

**Audit Decision: PASS — READY FOR DATA ACCUMULATION**

---

## 4. Binance Data Strategy (Critical Section)

**Your Key Insight (Correct):**
> BTC / ETH alone are NOT sufficient for arbitrage

This is 100% correct. Arbitrage requires:
- Graph connectivity
- Redundancy
- Alternative price discovery paths

Your Profile 0 / 1 / 2 tiering is exactly how professional desks operate.

| Profile | Verdict | Notes |
|---------|---------|-------|
| Profile 0 (BTC/ETH) | MANDATORY | Health + determinism |
| Profile 1 (Core Arb) | CORRECT MINIMUM | This is where alpha lives |
| Profile 2 (Liquidity basket) | CORRECTLY DEFERRED | Avoids premature bloat |

**Audit Decision: PASS — DO NOT REDUCE THIS UNIVERSE**

Do not start with fewer symbols. Breadth is a requirement, not a luxury.

---

## 5. Attached Implementation Plan Audit

### Phase 1 – Trades Capture
- Correct gap identified
- Correct event schema
- Correct integrity tiering
- JSONL is appropriate here

**No changes required**

### Phase 2 – Multi-Symbol Session Capture
- Per-symbol task isolation is correct
- Session-level manifest is correct
- Actor model reuse is good
- strict flag semantics are correct

**Strongly approved**

One minor clarification:
> Session manifest should aggregate:
> - gap stats
> - per-symbol determinism hashes
> - min/max timestamps

(This is additive, not blocking.)

### Phase 3 – Mark Price / Funding
- Correct expectation that SBE may not expose this
- REST polling fallback is acceptable
- FundingEvent schema is correct

**One rule to enforce:**
> REST-sourced funding events MUST be explicitly marked NON_CERTIFIED

Do not mix integrity tiers silently.

### Phase 4 – VectorBT Export
- Schema is sensible
- Separation of market vs fills is correct
- You resisted premature feature explosion (good)

**Approved**

### Phase 5 – Orchestration Script
- Operationally sound
- Validation step is mandatory (good)
- Export happens only after validation (correct)

**Approved**

---

## 6. Quality Gates Review

This table is excellent and should not be weakened:

| Gate | Verdict |
|------|---------|
| Bootstrap correctness | MANDATORY |
| Gap rate strict=0 | CORRECT |
| Determinism hash | CORRECT |
| Economics sanity | CORRECT |

This is not negotiable if you want credible arb research.

---

## 7. Zerodha (India) Status

**Correctly parked.**

You have done the right thing by:
- Not speculating
- Not adding hacks
- Not contaminating crypto infra with India uncertainty

Revisit only when markets are open.

---

## 8. Strategic Alignment Check

Your mental model is now correct and stable:
- QuantLaxmi is the product
- Data precedes strategy
- Arbitrage requires graph breadth
- Infra is frozen; execution has begun

This is the correct irreversible transition.

---

## 9. Final Instructions (Authoritative)

### Immediate Actions
1. Implement Phase 1 + Phase 2
2. Run 2–6 hour sessions with:
   - Profile 0 (daily)
   - Profile 1 (arb research)

### Do NOT:
- Tune strategies yet
- Add symbols beyond Profile 1
- Weaken strict mode

---

## Final Audit Verdict

**APPROVED — BEGIN IMPLEMENTATION**

This is a clean handoff.
You are no longer designing infrastructure — you are building a trading system.

**Proceed.**
