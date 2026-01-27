# QuantLaxmi Phase Status
## Current Implementation State

**Last Updated:** 2026-01-27
**Current Phase:** 13.3 Complete, Phase 14 Pending

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
| 13.2b | Portfolio Selector | ✅ Complete | 2026-01-27 |
| 13.3 | Capital Allocation | ✅ Complete | 2026-01-27 |

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
| quantlaxmi-gates | 76 | ✅ All passing |
| quantlaxmi-models | 57 | ✅ All passing |
| quantlaxmi-runner-crypto | 52 | ✅ All passing |
| quantlaxmi-strategy | 39 | ✅ All passing |
| **Workspace Total** | 300+ | ✅ All passing |

---

### Phase 13.2b: Portfolio Selector

**Question Answered:** "Given governed capital buckets and eligible strategies, what is the admissible portfolio structure?"

**Deliverables:**
- `crates/quantlaxmi-gates/src/portfolio_selector.rs`
- `IntentId` — unique intent identifier
- `PortfolioIntent` — per-bucket ordering of eligible strategies
- `StrategyIntent` — priority-ordered strategy with digests
- `PortfolioPolicy` — configurable ordering rules and limits
- `OrderingRule` enum (EligibilityTier, AlphaScoreDescending, DrawdownAscending, etc.)
- `PortfolioRejection` — audit artifact for rejected strategies
- `PortfolioSelector` — policy-driven selection engine
- `StrategyOrderingMetrics` — metrics for ordering decisions

**Tests:** 9 tests covering all portfolio selection scenarios

**Core Invariants Enforced:**
1. Read-only from bucket snapshots (no mutations)
2. No capital quantities (policy-only priority ordering)
3. Bucket constraints absolute (max_concurrent_strategies)
4. All decisions produce deterministic SHA-256 digests
5. Same inputs → identical digest

### Phase 13.3: Capital Allocation

**Question Answered:** "How much capital does each strategy receive?"

**Deliverables:**
- `crates/quantlaxmi-gates/src/capital_allocation.rs`
- `PlanId` — unique allocation plan identifier
- `AllocationPolicy` — reserve ratio, min allocation, caps, skip rules
- `AllocationMode` enum (EqualSplit, PriorityFill, ScoreProportional)
- `StrategyAllocation` — assigned capital + reasons + skip tracking
- `AllocationPlan` — per-bucket capital assignments with SHA-256 digest
- `AllocationDecision` — audit artifact with validation checks
- `AllocationCheck` — individual validation result
- `RebalancePolicy` enum (FixedInterval, OnNewSnapshot, OnNewIntent)
- `Allocator` — pure function allocation engine

**Tests:** 13 tests covering all allocation scenarios

**Core Invariants Enforced:**
1. Allocation is pure function of inputs + policy (deterministic)
2. No re-evaluation of eligibility (trusts prior digests)
3. Bucket constraints absolute (never violated)
4. Ordering respected (no skip unless policy allows)
5. No hidden state (all params in AllocationPolicy)
6. Audit artifacts first-class (deterministic SHA-256 digests)

---

## Next Phase: Phase 14 (Future)

**Status:** Specification pending

**Phase 14: Adaptive Intelligence**
- EARNHFT Router (selects agent profile)
- RL Agent (execution policy)
- Q-Teacher (offline training)

**Phase 15+: Multi-Venue Expansion**
- Additional crypto venues
- India options support
- Cross-venue arbitrage

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
| `PortfolioIntent` digest computation | Phase 13.2b |
| `OrderingRule` semantics | Phase 13.2b |
| Priority ordering (no quantities) | Phase 13.2b |
| `AllocationPlan` digest computation | Phase 13.3 |
| `AllocationPolicy` fingerprint | Phase 13.3 |
| Reserve ratio semantics | Phase 13.3 |
| Skip/ordering enforcement | Phase 13.3 |

---

*End of Phase Status*
