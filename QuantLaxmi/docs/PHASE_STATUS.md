# QuantLaxmi Phase Status
## Current Implementation State

**Last Updated:** 2026-01-27
**Current Phase:** 13.2a Complete, 13.2b Pending

---

## Phase Completion Matrix

| Phase | Name | Status | Completion Date |
|-------|------|--------|-----------------|
| 0 | Repo Hygiene | ✅ Complete | - |
| 1 | Platform Spine | ✅ Complete | - |
| 2 | Edge Protector | ✅ Complete | - |
| 3-10 | Core Foundations | ✅ Complete | - |
| 11 | Paper Evidence Loop | ✅ Complete | - |
| 12.1 | Strategy Pack v1 | ✅ Complete | 2026-01-26 |
| 12.2 | Tournament Runner | ✅ Complete | 2026-01-26 |
| 12.3 | Promotion Tightening | ✅ Complete | 2026-01-27 |
| 13.1 | Capital Eligibility | ✅ Complete | 2026-01-27 |
| 13.2a | Capital Buckets | ✅ Complete | 2026-01-27 |
| 13.2b | Portfolio Selector | ⏳ Pending | - |
| 13.3 | Capital Allocation | ⏳ Pending | - |

---

## Recently Completed Phases (Detail)

### Phase 12.3: Promotion Tightening

**Invariant:** No G3 without paper evidence

**Deliverables:**
- `crates/quantlaxmi-gates/src/promotion.rs`
- `PromotionSource` enum (Tournament, Paper, Manual)
- `PaperEvidence` struct with validation
- `PromotionPolicy` with configurable rules
- `PromotionValidator` with gate integration
- `PromotionDecision` with deterministic digest

**Tests:** 8 tests covering all promotion paths

**Key Types:**
```rust
pub struct PromotionRequest {
    pub strategy_id: String,
    pub source: PromotionSource,
    pub paper_evidence: Option<PaperEvidence>,
    pub tournament_id: Option<String>,
    // ...
}

pub struct PromotionDecision {
    pub accepted: bool,
    pub rejection_reasons: Vec<String>,
    pub decision_digest: String,
    // ...
}
```

---

### Phase 13.1: Capital Eligibility Layer

**Question Answered:** "Is this G3 strategy allowed to touch capital under what constraints?"

**Deliverables:**
- `crates/quantlaxmi-gates/src/capital_eligibility.rs`
- `Venue` enum (BinancePerp, BinanceSpot, NseF, NseO, Paper)
- `EligibilityStatus` enum (Eligible, Ineligible, Conditional)
- `EligibilityPolicy` with presets (default, strict, lenient)
- `EligibilityValidator` with hard/soft check distinction
- `EligibilityDecision` with deterministic digest

**Tests:** 13 tests covering all eligibility scenarios

**Mandatory Invariants Enforced:**
1. No G3 → No eligibility
2. No paper evidence → No eligibility
3. Promotion rejected → No eligibility
4. Drawdown breach → Ineligible
5. Alpha below threshold → Ineligible
6. Insufficient paper trades → Ineligible
7. Win rate below threshold → Ineligible

---

### Phase 13.2a: Capital Buckets

**Question Answered:** "What capital exists, where is it allowed to operate, and under what constraints?"

**Deliverables:**
- `crates/quantlaxmi-gates/src/capital_buckets.rs`
- `BucketId`, `StrategyId`, `SnapshotId`, `Symbol` identifiers
- `FixedPoint` for capital arithmetic
- `Currency` enum (USD, USDT, INR)
- `RiskClass` enum (Conservative, Moderate, Aggressive, Experimental)
- `BucketConstraints` with builder pattern
- `CapitalBucket` — venue-isolated capital pool
- `BucketRegistry` — governed bucket management
- `BucketEligibilityBinding` — explicit strategy-bucket binding
- `BucketBindingDecision` with deterministic digest
- `BucketSnapshot` for audit snapshots

**Tests:** 13 tests covering all bucket operations

**Core Invariants Enforced:**
1. Buckets are venue-isolated (Crypto ≠ India)
2. Strategies do not own capital — granted access via binding
3. Ineligible strategies rejected at binding time
4. Venue mismatch rejected
5. Max concurrent strategies enforced
6. Duplicate bindings rejected
7. All decisions produce deterministic digests

---

## Test Coverage Summary

| Crate | Tests | Status |
|-------|-------|--------|
| quantlaxmi-gates | 54 | ✅ All passing |
| quantlaxmi-models | 57 | ✅ All passing |
| quantlaxmi-runner-crypto | 52 | ✅ All passing |
| quantlaxmi-strategy | 39 | ✅ All passing |
| **Workspace Total** | 280+ | ✅ All passing |

---

## Next Phase: 13.2b Portfolio Selector

**Status:** Specification pending

**Will Consume:**
- `CapitalEligibility`
- `CapitalBucket`
- `BucketEligibilityBinding`

**Will Produce:**
- Allocation intents (priority ordering)
- No capital quantities (that's Phase 13.3)

**Scope:**
- Multiple eligible strategies compete
- Bucket constraints respected
- Priority ordering determined
- Still no sizing math

---

## Architectural Freeze Points

The following are now contractual surfaces and cannot change without a Phase bump:

| Surface | Frozen In |
|---------|-----------|
| `EligibilityStatus` semantics | Phase 13.1 |
| Hard vs conditional check distinction | Phase 13.1 |
| `PromotionDecision` digest computation | Phase 12.3 |
| `EligibilityDecision` digest computation | Phase 13.1 |
| `BucketBindingDecision` digest computation | Phase 13.2a |
| `BucketSnapshot` digest computation | Phase 13.2a |
| Venue isolation enforcement | Phase 13.2a |
| Policy preset meanings | Phase 13.1 |

---

*End of Phase Status*
