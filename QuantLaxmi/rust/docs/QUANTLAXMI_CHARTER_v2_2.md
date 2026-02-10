# QuantLaxmi Trading Platform
## Master Technical Specification & Project Charter v2.2

**Status:** Authoritative "Pole Star" specification for QuantLaxmi
**Last Updated:** 2026-01-27
**Applies to:** India FNO + Crypto (24×7)
**Core commitments:** Determinism, audit-grade replayability, governed innovation, friction-aware profitability, production-grade observability and safety controls.

---

## Executive State

QuantLaxmi is now **infrastructure-complete, selection-complete, and eligibility-complete**. The system enforces deterministic, audit-grade progression from data capture through capital eligibility, with no capital allocation logic yet introduced.

---

## Canonical Lifecycle (Authoritative)

```
Data → Signal → Tournament → Paper → Promotion → Capital Eligibility → Capital Buckets → (Portfolio Allocation)
```

The system is intentionally paused **before portfolio allocation**.

---

## Table of Contents

1. [Project Mission and Vision](#10-project-mission-and-vision)
2. [Current System State](#20-current-system-state)
3. [Completed Phases Summary](#30-completed-phases-summary)
4. [Hard Invariants](#40-hard-invariants)
5. [Architecture Overview](#50-architecture-overview)
6. [Gates Layer](#60-gates-layer)
7. [What Does Not Exist Yet](#70-what-does-not-exist-yet)
8. [Technology Stack](#80-technology-stack)
9. [Forward Roadmap](#90-forward-roadmap)

---

## 1.0 Project Mission and Vision

### 1.1 Mission

To build a **profitable, production-grade, auditably correct multi-venue trading platform** (India FNO + Crypto) whose edge is driven by:

1) **Microstructure-aware execution** (queue/imbalance/latency + robust fill realism)
2) **Regime-aware signal generation**
3) **Deterministic strategy lifecycle** with formal promotion gates

All decisions must remain **deterministic, replayable, observable, and governed by promotion gates**.

### 1.2 Vision

QuantLaxmi is a system where **every live decision can be reproduced**, every unit of PnL can be attributed (fees, slippage, latency, adverse selection), and innovation is introduced safely through a governed pipeline that prevents "ghost alpha."

### 1.3 Canonical One-Line State

> QuantLaxmi now deterministically ranks, proves, promotes, and authorizes strategies to touch capital under explicit constraints — with capital pools defined but not yet allocated.

---

## 2.0 Current System State

### 2.1 What Is Complete

| Capability | Status | Phase |
|------------|--------|-------|
| WAL-first capture | ✅ Complete | Phase 1 |
| Deterministic replay | ✅ Complete | Phase 1 |
| Promotion gates (G0-G4) | ✅ Complete | Phase 2 |
| Paper trading loop | ✅ Complete | Phase 11 |
| Strategy SDK + Registry | ✅ Complete | Phase 12.1 |
| Tournament runner | ✅ Complete | Phase 12.2 |
| Promotion tightening | ✅ Complete | Phase 12.3 |
| Capital eligibility | ✅ Complete | Phase 13.1 |
| Capital buckets | ✅ Complete | Phase 13.2a |

### 2.2 What Is Next

| Capability | Status | Phase |
|------------|--------|-------|
| Portfolio selector | Pending | Phase 13.2b |
| Capital allocation | Pending | Phase 13.3 |

---

## 3.0 Completed Phases Summary

### Phase 0–10: Core Foundations

* Deterministic WAL-first capture
* Replay parity (G1)
* Bundle schema v1.2 + validator
* Promotion pipeline (G1 → G2 → G3)
* Paper trading loop with RiskEnvelope v1
* Audit-grade artifacts and Gold Samples

### Phase 11: Paper Evidence Loop

* Deterministic paper evidence
* Attribution + AlphaScore computation
* Evidence embedded in bundles
* Closed-loop verification

### Phase 12.1: Strategy Pack v1

* Three baseline strategies: `funding_bias`, `micro_breakout`, `spread_mean_revert`
* Fixed-point arithmetic throughout
* StrategyRegistry integration

### Phase 12.2: Tournament Runner

* Deterministic tournament execution
* Leaderboard with reproducible ranking
* Audit-grade tournament manifests
* Validation parity with bundles

### Phase 12.3: Promotion Tightening

**Invariant enforced:** No G3 without paper evidence

* `PromotionPolicy` encoded in code
* Tournament → eligibility only
* Paper evidence mandatory for G2 → G3
* Deterministic `PromotionDecision` artifacts with SHA-256 digests

### Phase 13.1: Capital Eligibility Layer

* `CapitalEligibility` as first-class gate
* Hard vs conditional eligibility distinction
* Policy presets (default / strict / lenient)
* Deterministic eligibility decisions with digests
* 13 tests enforcing mandatory invariants

### Phase 13.2a: Capital Buckets

* Venue-isolated capital pools (`CapitalBucket`)
* `BucketRegistry` for governed bucket management
* `BucketEligibilityBinding` for explicit strategy-bucket bindings
* `BucketSnapshot` for deterministic audit snapshots
* `FixedPoint` for capital arithmetic (no floats)
* 13 tests enforcing bucket invariants

---

## 4.0 Hard Invariants (Do Not Violate)

These are platform laws. Any code that violates them is rejected.

1. **Determinism everywhere** — Same inputs must produce same outputs
2. **No silent promotion or eligibility** — Every decision is explicit and hashed
3. **Venue isolation** — India ≠ Crypto, Perps ≠ Spot ≠ Options
4. **No capital math before Phase 13.2+** — Buckets exist but are not allocated
5. **All decisions are hashed artifacts** — Promotion, eligibility, binding decisions
6. **No phase may reinterpret a prior decision** — Decisions are immutable
7. **Optimizers consume state; they do not mutate it**
8. **Strategies do not own capital** — They are granted access to buckets

---

## 5.0 Architecture Overview

### 5.1 Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Portfolio Allocation                      │  ← Phase 13.2b+
│                    (NOT YET IMPLEMENTED)                     │
├─────────────────────────────────────────────────────────────┤
│                     Capital Buckets                          │  ← Phase 13.2a ✅
│   BucketRegistry │ CapitalBucket │ BucketEligibilityBinding │
├─────────────────────────────────────────────────────────────┤
│                   Capital Eligibility                        │  ← Phase 13.1 ✅
│   EligibilityValidator │ EligibilityPolicy │ EligibilityDecision │
├─────────────────────────────────────────────────────────────┤
│                      Promotion                               │  ← Phase 12.3 ✅
│   PromotionValidator │ PromotionPolicy │ PromotionDecision  │
├─────────────────────────────────────────────────────────────┤
│                      Tournament                              │  ← Phase 12.2 ✅
│   TournamentRunner │ Leaderboard │ TournamentManifest       │
├─────────────────────────────────────────────────────────────┤
│                   Strategy SDK                               │  ← Phase 12.1 ✅
│   StrategyRegistry │ Strategy trait │ StrategyContext       │
├─────────────────────────────────────────────────────────────┤
│                   Gates (G0-G4)                              │  ← Phase 2 ✅
│   G0 DataTruth │ G1 Replay │ G2 Backtest │ G3 Robust │ G4 Deploy │
├─────────────────────────────────────────────────────────────┤
│                   Core Infrastructure                        │  ← Phase 1 ✅
│   WAL │ Replay │ Events │ Models │ Connectors               │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Crate Structure

**Core Crates:**
- `quantlaxmi-models` — Canonical schemas, events, tournament/promotion artifacts
- `quantlaxmi-events` — Decision trace hashing, fixed-point numerics
- `quantlaxmi-wal` — Write-ahead log
- `quantlaxmi-gates` — G0-G4 gates, promotion, eligibility, buckets
- `quantlaxmi-strategy` — Strategy SDK and registry
- `quantlaxmi-executor` — Execution engine, risk envelope
- `quantlaxmi-risk` — Risk policy engine

**Runner Crates:**
- `quantlaxmi-runner-common` — Shared runner infrastructure
- `quantlaxmi-runner-crypto` — Crypto backtest, paper, live
- `quantlaxmi-runner-india` — India F&O runner (in progress)

---

## 6.0 Gates Layer

The gates layer (`quantlaxmi-gates`) now contains:

### 6.1 Validation Gates (G0-G4)

| Gate | Purpose | Status |
|------|---------|--------|
| G0 DataTruth | Validate capture data integrity | ✅ |
| G1 ReplayParity | Ensure deterministic replay matches live | ✅ |
| G2 BacktestCorrectness | Validate backtest assumptions | ✅ |
| G3 Robustness | Stress testing and edge cases | ✅ |
| G4 Deployability | Pre-production readiness checks | ✅ |

### 6.2 Lifecycle Gates

| Gate | Purpose | Status |
|------|---------|--------|
| Promotion | Tournament → Paper → G3 advancement | ✅ Phase 12.3 |
| Capital Eligibility | G3 → Capital permission | ✅ Phase 13.1 |
| Capital Buckets | Venue-isolated capital pools | ✅ Phase 13.2a |

---

## 7.0 What Does Not Exist Yet

These are explicitly deferred to future phases:

- ❌ Capital allocation
- ❌ Portfolio optimization
- ❌ Strategy sizing
- ❌ Rebalancing logic
- ❌ Cross-strategy arbitration
- ❌ Runtime capital enforcement

---

## 8.0 Technology Stack

### 8.1 Rust (Production Spine)

All production code is Rust:
- Deterministic event processing
- WAL + replay
- Gates and validators
- Execution engine
- Risk policy

### 8.2 Fixed-Point Arithmetic

All monetary and ratio computations use fixed-point (mantissa + exponent):
- No floats in configs, traces, or execution paths
- `FixedPoint` struct for capital amounts
- Basis points (bps) for ratios

### 8.3 Deterministic Hashing

All decision artifacts are hashed with SHA-256:
- `PromotionDecision.decision_digest`
- `EligibilityDecision.decision_digest`
- `BucketBindingDecision.digest`
- `BucketSnapshot.digest`

---

## 9.0 Forward Roadmap

### Phase 13.2b: Portfolio Selector (Policy-Only)

**Status:** NEXT

Consumes:
- `CapitalEligibility`
- `CapitalBucket`
- `BucketEligibilityBinding`

Produces:
- Allocation intents (priority ordering)
- No capital quantities yet

### Phase 13.3: Capital Allocation

**Status:** After 13.2b

- Capital math appears
- Rebalancing exists
- Risk budgeting enforced
- Execution engines receive numbers

### Phase 14+: Adaptive Intelligence

- EARNHFT Router integration
- RL-based regime adaptation
- Hierarchical policy selection

---

## Appendix: Schema Versions

| Artifact | Schema Version |
|----------|----------------|
| Segment Manifest | v9 |
| Promotion Decision | `promotion_decision_v1.0` |
| Eligibility Decision | `eligibility_decision_v1.0` |
| Capital Bucket | `capital_bucket_v1.0` |
| Alpha Score | `alpha_score_v1.0` |
| G1 Promotion Gate | `g1_gate_v1.0` |

---

*End of Charter v2.2*
