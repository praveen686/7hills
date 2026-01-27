# QuantLaxmi Roadmap Phases
## Execution Plan Aligned to Charter v2.2

**Status:** Operational roadmap for implementation sequencing
**Last Updated:** 2026-01-27
**Current Phase:** 13.2a Complete → 13.2b Next

---

## Table of Contents

1. [Operating Model](#1-operating-model)
2. [Phase 0: Repo Hygiene](#2-phase-0-repo-hygiene-complete)
3. [Phase 1: Platform Spine](#3-phase-1-platform-spine-complete)
4. [Phase 2: Edge Protector](#4-phase-2-edge-protector-complete)
5. [Phase 11: Paper Evidence Loop](#5-phase-11-paper-evidence-loop-complete)
6. [Phase 12: Strategy Lifecycle](#6-phase-12-strategy-lifecycle-complete)
7. [Phase 13: Capital Lifecycle](#7-phase-13-capital-lifecycle-active)
8. [Phase 14+: Future Phases](#8-phase-14-future-phases)
9. [Definition of Done](#9-definition-of-done)

---

## 1) Operating Model

### 1.1 Engineering Modes
- **Backbone mode:** deterministic capture → WAL → replay → gates → risk → execution
- **Research mode:** offline experimentation producing artifacts + evidence

### 1.2 Promotion Discipline
- Research outputs require:
  - Canonical outputs
  - Replayability
  - Gate compliance

### 1.3 Release Policy
- Every merged change must preserve:
  - Event schema stability
  - WAL readability
  - Replay parity invariants

---

## 2) Phase 0: Repo Hygiene (COMPLETE)

### Deliverables
- ✅ `docs/QUANTLAXMI_CHARTER_v2_2.md`
- ✅ `docs/PRODUCTION_CONTRACT.md`
- ✅ `docs/ROADMAP_PHASES.md`
- ✅ `docs/PHASE_STATUS.md`

### Acceptance
- ✅ CI green on main
- ✅ Documentation present

---

## 3) Phase 1: Platform Spine (COMPLETE)

### Deliverables
- ✅ Canonical events (`quantlaxmi-events`, `quantlaxmi-models`)
- ✅ WAL v1 (`quantlaxmi-wal`)
- ✅ Manifests binding
- ✅ Deterministic replay
- ✅ Gates framework (`quantlaxmi-gates`)
- ✅ Correlation IDs everywhere

### Acceptance
- ✅ Replay yields identical decision trace hash
- ✅ G0 and G4 pass on fixture sessions
- ✅ No float intermediates in canonical parsing

---

## 4) Phase 2: Edge Protector (COMPLETE)

### Deliverables
- ✅ Risk policy engine v1
- ✅ Regime Router v1 (deterministic)
- ✅ Toxicity filters
- ✅ Full G0-G4 implementation
- ✅ Paper trading harness

### Acceptance
- ✅ Strategies operate in paper mode with risk ladder
- ✅ G0-G4 runnable
- ✅ Simple strategy survives friction

---

## 5) Phase 11: Paper Evidence Loop (COMPLETE)

### Deliverables
- ✅ Deterministic paper evidence
- ✅ Attribution + AlphaScore computation
- ✅ Evidence embedded in bundles
- ✅ Closed-loop verification

### Acceptance
- ✅ Paper sessions produce verifiable evidence
- ✅ AlphaScore computation deterministic

---

## 6) Phase 12: Strategy Lifecycle (COMPLETE)

### Phase 12.1: Strategy Pack v1 (COMPLETE)

**Deliverables:**
- ✅ `funding_bias` strategy
- ✅ `micro_breakout` strategy
- ✅ `spread_mean_revert` strategy
- ✅ Strategy SDK with registry
- ✅ Fixed-point arithmetic

### Phase 12.2: Tournament Runner (COMPLETE)

**Deliverables:**
- ✅ `TournamentRunner` with deterministic execution
- ✅ `LeaderboardV1` with reproducible ranking
- ✅ `TournamentManifest` for audit
- ✅ Attribution summary integration

### Phase 12.3: Promotion Tightening (COMPLETE)

**Invariant:** No G3 without paper evidence

**Deliverables:**
- ✅ `PromotionSource` (Tournament, Paper, Manual)
- ✅ `PaperEvidence` struct
- ✅ `PromotionPolicy` with rules
- ✅ `PromotionValidator` with gate integration
- ✅ `PromotionDecision` with digest

**Tests:** 8 tests enforcing invariants

---

## 7) Phase 13: Capital Lifecycle (ACTIVE)

### Phase 13.1: Capital Eligibility (COMPLETE)

**Question:** "Is this G3 strategy allowed to touch capital under what constraints?"

**Deliverables:**
- ✅ `Venue` enum
- ✅ `EligibilityStatus` (Eligible, Ineligible, Conditional)
- ✅ `EligibilityPolicy` with presets
- ✅ `EligibilityValidator`
- ✅ `EligibilityDecision` with digest

**Tests:** 13 tests covering all scenarios

### Phase 13.2a: Capital Buckets (COMPLETE)

**Question:** "What capital exists, where is it allowed to operate, and under what constraints?"

**Deliverables:**
- ✅ `CapitalBucket` — venue-isolated capital pool
- ✅ `BucketRegistry` — governed management
- ✅ `BucketEligibilityBinding` — explicit binding
- ✅ `BucketBindingDecision` with digest
- ✅ `BucketSnapshot` for audit
- ✅ `FixedPoint` for capital arithmetic
- ✅ `Currency`, `RiskClass` enums

**Tests:** 13 tests covering all operations

**What This Phase Does NOT Do:**
- ❌ No capital allocation
- ❌ No sizing
- ❌ No optimizer hooks
- ❌ No rebalancing

### Phase 13.2b: Portfolio Selector (NEXT)

**Question:** "Which eligible strategies may draw from which buckets, and in what priority order?"

**Will Consume:**
- `CapitalEligibility`
- `CapitalBucket`
- `BucketEligibilityBinding`

**Will Produce:**
- Allocation intents (priority ordering)
- Still no capital quantities

**Scope:**
- Multiple eligible strategies compete
- Bucket constraints respected
- Priority ordering determined
- No sizing math

### Phase 13.3: Capital Allocation (PENDING)

**Question:** "How much capital does each strategy receive?"

**Will Introduce:**
- Capital math
- Rebalancing logic
- Risk budgeting
- Execution engines receive numbers

---

## 8) Phase 14+: Future Phases

### Phase 14: Adaptive Intelligence

**Objective:** Introduce hierarchical RL as controlled upgrade

**Components:**
- EARNHFT Router (selects agent profile)
- RL Agent (execution policy)
- Q-Teacher (offline training)

**Safety Posture:**
- RL outputs bounded parameters first
- Full policy control only after gate compliance

### Phase 15+: Multi-Venue Expansion

- Additional crypto venues
- India options support
- Cross-venue arbitrage

---

## 9) Definition of Done

A phase is "done" only when:

1. ✅ Acceptance criteria are met
2. ✅ CI includes regression tests
3. ✅ Replay parity remains intact
4. ✅ Clippy clean
5. ✅ Documentation updated
6. ✅ All tests pass

---

## Quick Reference: Current State

```
Lifecycle Position:

Data ──────────────────────────────────────────────────────────────────────────►
      │                                                                        │
      │  [COMPLETE]                                                            │
      │  Capture → WAL → Replay → Gates → Tournament → Paper → Promotion      │
      │                                                                        │
      │  [COMPLETE]                                                            │
      │  → Capital Eligibility → Capital Buckets                               │
      │                                                                        │
      │  [NEXT]                                                                │
      │  → Portfolio Selector → Capital Allocation                             │
      │                                                                        │
      │  [FUTURE]                                                              │
      │  → Live Execution → Adaptive Intelligence                              │
      └────────────────────────────────────────────────────────────────────────┘
```

---

*End of Roadmap*
