# QuantLaxmi Production Contract
## Canonical Events + WAL + Replay + Hashing + Gates (Implementable Spec)

**Status:** Enforceable specification.
Any component that violates this contract is considered non-production.

---

## 1) Canonical Event Contract

### 1.1 Canonical Event Types (Required)

#### Market
- QuoteEvent (L1)
- DepthEvent (L2 snapshot/delta) [optional to produce, must be supported]
- TradeEvent
- FundingEvent (crypto)
- InstrumentEvent
- SessionEvent

#### Trading Pipeline
- DecisionEvent
- RiskEvent
- OrderEvent
- FillEvent
- FeatureEvent (optional)

---

## 2) Canonical Numeric Contract (Fixed Point)

### 2.1 Format
All prices/qty in canonical events use:
- mantissa: i64
- exponent: i32

Example:
- price = mantissa * 10^exponent

### 2.2 Rules
- no float intermediates in parsing
- deterministic rounding rules
- overflow detection required (fail gate if overflow occurs)

---

## 3) Required IDs and Correlation Rules

Every event record must include:
- session_id
- run_id
- venue
- symbol
- strategy_id (if applicable)
- decision_id (if applicable)
- risk_decision_id (if applicable)
- order_id (if applicable)
- fill_id (if applicable)

All logs/spans must carry the same identifiers.

---

## 4) WAL Contract

### 4.1 WAL Contents
WAL must record:
- canonical market events
- all DecisionEvents
- all RiskEvents
- all OrderEvents
- all FillEvents
- health/session events

### 4.2 WAL Format
Phase 1: JSONL
- 1 record = 1 line
- stable field order required for hashing (see hashing contract)

---

## 5) Manifests and Binding

### 5.1 Session Manifest Must Include
- universe symbols
- instrument metadata snapshot hash
- source identity (venue + connector version)
- event schema versions
- checksums for capture artifacts

### 5.2 Run Manifest Must Include
- code hash (git commit + build metadata)
- config snapshot hash
- strategy hash (if distinct)
- WAL path(s)
- WAL checksum(s)
- decision trace hash
- gate report checksum

---

## 6) Replay Contract

### 6.1 Function
Replay(WAL, Config, CodeHash) -> DecisionTrace

### 6.2 Invariant
DecisionTraceHash(original) == DecisionTraceHash(replay)

### 6.3 DecisionTrace Hash
Minimum hashing set:
- all DecisionEvents in canonical serialization order

Recommended hashing extension:
- include RiskEvents and Order intents

---

## 7) Deterministic Ordering Rules

Events must be ordered using:
1) venue timestamp (if present)
2) capture timestamp
3) sequence number (if present)
4) symbol
5) stable tie-breaker (connector-defined)

Failure to preserve ordering is a gate failure (G1).

---

## 8) Nondeterminism Policy

### 8.1 Forbidden
- unseeded randomness
- nondeterministic inference operators without explicit control
- concurrency-dependent event order at strategy boundary

### 8.2 Allowed Only If Recorded and Verified
- seeded randomness with seed recorded in config/WAL
- bounded nondeterminism in inference with repeatability checks

---

## 9) Promotion Gates (Implementable)

### G0 DataTruth (Required)
Must validate:
- schema compliance
- monotonicity where required
- quote_age sanity (distribution bounds configured per venue)
- manifest presence and integrity
- no NaN/Inf values
- no negative qty/price

### G1 ReplayParity (Required)
Must validate:
- decision trace hash parity (original vs replay)
- stable ordering invariants

### G2 Anti-Overfit Suite (Required)
At minimum:
- time-shift test
- random-entry baseline
- permutation within day/regime
- label shuffle for supervised models
- feature ablation

### G3 Robustness (Required)
- walk-forward
- regime slicing
- cost sensitivity sweeps
- perturbation stress

### G4 Deployability (Required)
- panic-free critical paths
- safe shutdown
- observability enabled
- bounded resource usage (configured)
- configuration completeness

---

## 10) Acceptance Criteria: "Production-Ready"

A module/strategy is production-ready only if:
- replay parity proven (G1)
- anti-overfit suite passed (G2)
- robustness passed (G3)
- deployability passed (G4)
- full event + WAL compliance maintained
