# QuantLaxmi Phase Status
## Current Implementation State

**Last Updated:** 2026-01-28
**Current Phase:** 20B In Progress (Signals Manifest)

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
| 14.1 | Execution Budget | ✅ Complete | 2026-01-27 |
| 14.2 baseline_v1 | Live Execution Binding | ✅ Complete | 2026-01-27 |
| 15.2 | MTM + Drawdown Layer | ✅ Complete | 2026-01-27 |
| 15.3 | Deterministic Intent Shaping | ✅ Complete | 2026-01-27 |
| 16 | Execution Session & Control Plane | ✅ Complete | 2026-01-27 |
| 17A | Execution Observability & Operator UX | ✅ Complete | 2026-01-28 |
| 19 | MarketSnapshot V2 + Canonical Encoding v3 | ✅ Complete | 2026-01-28 |
| 19C | Signal Admission Gating | ✅ Complete | 2026-01-28 |
| 19D | WAL Enforcement & Replay | ✅ Complete | 2026-01-28 |
| 20A | SignalFrame Spine | ✅ Complete | 2026-01-28 |
| 20B | Signals Manifest | 🟡 In Progress | - |

---

## Recently Completed Phases (Detail)

### Phase 20A: SignalFrame Spine

**Goal:** Canonical normalized signal input for strategies, independent of vendor quirks.

**Deliverables:**
- `crates/quantlaxmi-models/src/signal_frame.rs`
- `SignalFrame` struct with dual timestamps (event_ts_ns, book_ts_ns)
- `CorrelationId` as `[u8; 16]` (fixed-size bytes, not String)
- `L1Field` typed enum (BidPrice, AskPrice, BidQty, AskQty)
- `RefuseReason` with typed variants (not stringly-typed)
- `signal_frame_from_market()` conversion function
- `RequiredL1` internal requirements structure

**Tests:** 18 unit tests covering:
- All L1-L5 hard laws
- Zero-is-valid (L5) admission
- Crossed book rejection
- Exponent mismatch detection
- V1 legacy semantics
- i128 overflow-safe spread calculation
- Determinism (100 runs identical)

**Key Types:**
```rust
pub type CorrelationId = [u8; 16];

pub struct SignalFrame {
    pub correlation_id: CorrelationId,
    pub symbol: String,
    pub bid_px_m: i64,
    pub ask_px_m: i64,
    pub px_exp: i8,
    pub bid_qty_m: i64,
    pub ask_qty_m: i64,
    pub qty_exp: i8,
    pub l1_state_bits: u16,
    pub spread_bps_m: Option<i64>,
    pub event_ts_ns: i64,
    pub book_ts_ns: i64,
}

pub enum RefuseReason {
    FieldAbsent(L1Field),
    FieldNull(L1Field),
    FieldMalformed(L1Field),
    ExponentMismatch { kind: ExponentKind, expected: i8, actual: i8 },
    InvariantViolation(Invariant),
}
```

**Hard Laws Enforced:**
- L1: No Fabrication — Absent/Null/Malformed fields → Err
- L2: Deterministic — Same inputs → identical result
- L3: Explicit Refusal — Missing required fields enumerated
- L5: Zero Is Valid — Value(0) is accepted

---

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
| quantlaxmi-events | 184 | ✅ All passing |
| quantlaxmi-models | 106 | ✅ All passing |
| quantlaxmi-gates | 198 | ✅ All passing |
| quantlaxmi-runner-crypto | 70 | ✅ All passing |
| quantlaxmi-strategy | 39 | ✅ All passing |
| quantlaxmi-wal | 26 | ✅ All passing |
| Other crates | 166 | ✅ All passing |
| **Workspace Total** | 789 | ✅ All passing |

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

### Phase 14.1: Execution Budget

**Question Answered:** "How do allocation plans become runtime-enforceable budgets?"

**Deliverables:**
- `crates/quantlaxmi-gates/src/execution_budget.rs`
- `BudgetId` — derived deterministically (SHA-256 from strategy+bucket+plan)
- `DeltaId` — derived deterministically (not UUID)
- `ExecutionBudget` — with reserved/committed capital split
- `BudgetManager` — apply_allocation_plan, check_order, reserve_for_order, release_order, process_fill, release_position
- `BudgetDelta` — WAL-bound change events with deterministic digest
- `BudgetPolicy` — max order fraction, rate limits, position limits
- `OrderConstraints` — per-order enforcement limits
- `RateLimitTracker` — deterministic floor-division windowing
- `BudgetSnapshot` — audit snapshot with digest
- `OrderCheckResult` — pre-trade validation result

**Tests:** 12 tests covering all budget scenarios

**Core Invariants Enforced:**
1. Budgets derived from AllocationPlan — never invented
2. Budget enforcement is deterministic and auditable
3. No wall-clock affects identity or digests (event timestamps only)
4. Reserved capital (open orders) vs committed capital (open positions) tracked separately
5. All budget state changes produce BudgetDelta WAL events
6. Rate limits use floor-division windowing (deterministic)
7. Budget violations are hard rejections (no soft limits)
8. All artifacts have deterministic SHA-256 digests

**What This Phase Does NOT Do:**
- ❌ No venue connections
- ❌ No order submission to exchanges
- ❌ No fill processing from venues
- ❌ No PnL accounting

### Phase 14.2 baseline_v1: Live Execution Binding (COMPLETE)

**Question Answered:** "How do budgets become executed trades with deterministic audit trail?"

**Deliverables:**
- `crates/quantlaxmi-models/src/execution_events.rs` — Canonical execution event types
- `crates/quantlaxmi-runner-crypto/src/binance_perp_execution.rs` — Live execution engine
- `crates/quantlaxmi-runner-crypto/src/bin/validate_execution_session.rs` — Validator CLI

**Canonical Events:**
- `OrderIntentEvent` — Strategy wants to trade (entry point)
- `OrderSubmitEvent` — Order sent to exchange (budget reserved)
- `OrderAckEvent` — Exchange acknowledged order
- `OrderRejectEvent` — Exchange rejected order (budget rolled back)
- `OrderFillEvent` — Partial/full fill (budget committed)
- `OrderCancelEvent` — Order cancelled (remaining released)
- `PositionCloseEvent` — Position closed (capital released)

**Key Types:**
- `IntentId` — Derived deterministically (SHA-256 from strategy+bucket+ts+seq)
- `ClientOrderId` — Derived deterministically (truncated to 32 chars)
- `FillId` — Derived deterministically from exchange fill ID
- `IdempotencyKey` — Prevents duplicate processing of exchange events
- `LiveOrderState` — State machine enum for order lifecycle
- `LiveExecutionEngine` — Order lifecycle management with budget integration

**Tests:** 16 tests covering all execution scenarios

**Core Invariants Enforced:**
1. No order leaves without budget check + reservation delta
2. Every exchange event reconciles budget ledger with WAL-bound artifacts
3. Rollback on failure is deterministic
4. Idempotent processing of exchange events (no double-reserve/commit)
5. All IDs derived deterministically (SHA-256, no UUIDs)
6. All events have deterministic digests
7. State machine transitions are explicit and auditable

**What This Phase Does NOT Do:**
- ❌ No actual venue connections (stubbed interfaces)
- ❌ No WebSocket streaming (stubbed)
- ❌ No real order submission (stubbed)
- ❌ No PnL accounting (Phase 14.3+)

---

### Phase 19D: WAL Enforcement & Replay (COMPLETE)

**Question Answered:** "How do we ensure WAL is the authoritative source of truth for replay?"

**Deliverables:**
- `crates/quantlaxmi-wal/src/lib.rs` — AdmissionIndex, SegmentAdmissionSummary
- `crates/quantlaxmi-runner-crypto/src/backtest.rs` — AdmissionMode enum, decide_admission()
- `crates/quantlaxmi-runner-crypto/tests/phase19_admission_integration.rs` — 21 tests

**Key Types:**
- `AdmissionMode` — EvaluateLive vs EnforceFromWal
- `AdmissionIndex` — O(1) lookup by correlation_id for replay
- `AdmissionMismatchPolicy` — Fail vs Warn on mismatch
- `AdmissionMismatchReason` — MissingWalEntry, AdmitButWouldRefuse, etc.
- `SegmentAdmissionSummary` — Materialized summary from WAL

**Tests:** 21 tests covering all admission scenarios including:
- `test_replay_enforced_blocks_strategy_on_refuse`
- `test_replay_enforced_calls_strategy_on_admit`
- `test_replay_enforced_missing_wal_entry_policy_fail`
- `test_replay_enforced_missing_wal_entry_policy_warn`

**Core Invariants Enforced:**
1. WAL is authoritative truth — live evaluation cannot override
2. Strategy invocation gated by WAL decisions in enforce mode
3. Missing WAL entry → refuse (doctrine: cannot prove admission)
4. No new admission WAL writes in enforce mode (replay integrity)
5. Mismatch policy controls fail vs warn behavior
6. All decisions deterministic with SHA-256 digests

---

## Next Phase: Future Work

**Phase 20+: Adaptive Intelligence**
- EARNHFT Router (selects agent profile)
- RL Agent (execution policy)
- Q-Teacher (offline training)

**Future: Multi-Venue Expansion**
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
| `BudgetId` derivation (deterministic) | Phase 14.1 |
| `DeltaId` derivation (deterministic) | Phase 14.1 |
| `ExecutionBudget` digest computation | Phase 14.1 |
| `BudgetDelta` digest computation | Phase 14.1 |
| Reserved vs committed capital split | Phase 14.1 |
| Rate limit floor-division windowing | Phase 14.1 |
| `BudgetSnapshot` digest computation | Phase 14.1 |
| `IntentId` derivation (deterministic) | Phase 14.2 |
| `ClientOrderId` derivation (deterministic) | Phase 14.2 |
| `FillId` derivation (deterministic) | Phase 14.2 |
| `IdempotencyKey` derivation | Phase 14.2 |
| `LiveOrderState` state machine | Phase 14.2 |
| `OrderIntentEvent` digest computation | Phase 14.2 |
| `OrderSubmitEvent` digest computation | Phase 14.2 |
| `OrderAckEvent` digest computation | Phase 14.2 |
| `OrderRejectEvent` digest computation | Phase 14.2 |
| `OrderFillEvent` digest computation | Phase 14.2 |
| `OrderCancelEvent` digest computation | Phase 14.2 |
| `PositionCloseEvent` digest computation | Phase 14.2 |
| `MarketSnapshot` V2 canonical encoding | Phase 19 |
| `AdmissionDecision` digest computation | Phase 19C |
| `AdmissionOutcome` semantics | Phase 19C |
| Signal admission gating invariants | Phase 19C |
| `AdmissionIndex` lookup semantics | Phase 19D |
| `AdmissionMode` EvaluateLive vs EnforceFromWal | Phase 19D |
| WAL-authoritative replay semantics | Phase 19D |
| Missing WAL entry → refuse behavior | Phase 19D |

---

*End of Phase Status*
