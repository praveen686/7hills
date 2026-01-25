# QuantLaxmi Roadmap Phases
## Execution Plan Aligned to Charter v2.1 and Production Contract

**Status:** Operational roadmap for implementation sequencing.
**Rule:** No phase may be considered complete until its acceptance criteria are met.

---

## Table of Contents

1. [Operating Model](#1-operating-model)
2. [Phase 0: Repo Hygiene and Working Agreements](#2-phase-0-repo-hygiene-and-working-agreements)
3. [Phase 1: Platform Spine (Deterministic Baseline)](#3-phase-1-platform-spine-deterministic-baseline)
4. [Phase 2: Edge Protector (Real-World Stability)](#4-phase-2-edge-protector-real-world-stability)
5. [Phase 3: Edge Builders (Differentiated Alpha)](#5-phase-3-edge-builders-differentiated-alpha)
6. [Phase 4: Adaptive Intelligence (EARNHFT Upgrade)](#6-phase-4-adaptive-intelligence-earnhft-upgrade)
7. [Release Gates and Promotion Workflow](#7-release-gates-and-promotion-workflow)
8. [Definition of Done (DoD)](#8-definition-of-done-dod)

---

## 1) Operating Model

### 1.1 Engineering modes
- **Backbone mode:** deterministic capture → WAL → replay → gates → risk → execution → observability
- **Research mode:** offline experimentation producing artifacts (features/models) + evidence (gate reports)

### 1.2 Promotion discipline
- Research outputs are not allowed into production runners unless they:
  - produce canonical outputs
  - are replayable
  - pass required gates

### 1.3 Release policy
- Every merged change must preserve:
  - event schema stability
  - WAL readability
  - replay parity invariants
  - gate execution in CI

---

## 2) Phase 0: Repo Hygiene and Working Agreements

### 2.1 Deliverables
- `docs/QUANTLAXMI_CHARTER_v2_1.md` committed
- `docs/PRODUCTION_CONTRACT.md` committed
- `docs/ROADMAP_PHASES.md` committed
- `docs/ARCHITECTURE_DIAGRAMS.md` committed

### 2.2 CI baseline
- `cargo test` for workspace
- lint/format checks
- a single "fixture run" pipeline (tiny sample) reserved for gates

### 2.3 Acceptance Criteria
- CI green on main
- documentation present and referenced in `README.md` or `docs/index.md`

---

## 3) Phase 1: Platform Spine (Deterministic Baseline)

**Objective:** Prove capture → canonical events → WAL → manifests → replay parity.

### 3.1 Deliverables

#### A) Canonical events
- Introduce `quantlaxmi-events` (or `quantlaxmi-models::events`)
- Move fixed-point parsing and canonical QuoteEvent into canonical events crate
- Ensure both **Binance** and **Zerodha** emit canonical QuoteEvent

#### B) WAL v1 (JSONL)
- Introduce `quantlaxmi-wal`
- Record at least:
  - SessionEvent
  - QuoteEvent
  - DecisionEvent
  - RiskEvent (even if minimal)
  - OrderEvent (intent + ack/reject at minimum)
  - FillEvent (if available; paper mode acceptable initially)

#### C) Manifests binding
- `run_manifest` must include:
  - config hash
  - code hash
  - WAL path + checksum
  - decision trace hash
  - gate report checksum (even if stub)
- `session_manifest` must include:
  - universe
  - schema versions
  - capture metadata

#### D) Deterministic replay v1
- Implement `replay` command:
  - consumes WAL + config snapshot
  - produces decision trace hash
  - confirms parity with original

#### E) Gates framework v1
- Introduce `quantlaxmi-gates`
- Implement fully:
  - **G0 DataTruth**
  - **G4 Deployability**
- Scaffold:
  - G1 ReplayParity
  - G2 Anti-Overfit Suite
  - G3 Robustness

#### F) Correlation IDs everywhere
- Ensure tracing spans and logs include:
  - session_id, run_id, symbol, venue, strategy_id, decision_id, risk_decision_id, order_id

### 3.2 Acceptance Criteria
- Run a small live/paper session, produce WAL + manifests
- Replay yields identical decision trace hash
- G0 and G4 pass on fixture session in CI
- No float intermediates in canonical event parsing paths
- Observability exporter available and non-crashing

---

## 4) Phase 2: Edge Protector (Real-World Stability)

**Objective:** Make "profitability survivable" by preventing common live failure modes.

### 4.1 Deliverables

#### A) Risk policy engine v1
- Implement deterministic pre-trade checks:
  - max position/notional
  - max order rate
  - quote_age bounds
  - spread bounds
  - liquidity floor (if L2 available; else L1 proxy)
- Implement escalation ladder:
  - WARN → THROTTLE → HALT
- Ensure every risk decision is written to WAL as RiskEvent

#### B) Regime Router v1 (deterministic)
- Implement regime classification:
  - volatility bucket + liquidity bucket + trend bucket
- Router selects "profile":
  - sizing band
  - max aggressiveness
  - market-making vs taker toggle
- Router output recorded to WAL/DecisionEvent context

#### C) Toxicity filters (microstructure defense)
- Implement:
  - VPIN or VPIN-lite
  - imbalance/staleness filters
  - spread shock detection
- Use filters to throttle/avoid trading

#### D) Expand gates
- Implement **G1 ReplayParity** fully
- Implement **G2 Anti-Overfit Suite** minimally:
  - time-shift test
  - random-entry baseline
  - simple permutation test (within day)
  - (supervised) label shuffle
  - basic ablation hooks
- Implement **G3 Robustness** minimally:
  - walk-forward skeleton
  - fee 2x and slippage +X bps sweep

#### E) Paper trading / simulator harness v1
- Latency model
- Slippage model
- Partial fill model
- (Optional) FIFO queue approximation

### 4.2 Acceptance Criteria
- Strategies operate in paper mode with:
  - risk ladder functioning
  - no uncontrolled "close all" hard-coded global rules
  - WAL includes risk and router decisions
- G0–G4 implemented and runnable
- A simple strategy survives friction and does not profit on placebo suite

---

## 5) Phase 3: Edge Builders (Differentiated Alpha)

**Objective:** Add unique alpha modules under strict promotion governance.

### 5.1 Deliverables

#### A) Options surface feature engine (India)
- skew slope/curvature features
- term structure features
- expiry/roll window features
- surface state recorded as FeatureEvent summaries + hashes

#### B) Carry / calendar engine (India + crypto)
- Crypto:
  - funding schedule + basis approximation
  - feasibility scoring (margin, liquidation proxy)
- India:
  - expiry calendar + holiday effects + rollover features
- Output: carry score + risk flags + regime overlays

#### C) Indic mathematics feature library (research quarantine)
- Ramanujan periodic dictionary features
- Mock theta/q-series transforms
- Continued fraction approximants (streaming)
- Kuttaka-like constraint solver (calendar feasibility)
- Stability tests and perturbation robustness suite

#### D) ML ranker/filter (gated)
- Use ML primarily for:
  - ranking trade candidates
  - do-not-trade anomaly detection
- Not direct execution control (yet)

### 5.2 Acceptance Criteria
- At least one strategy shows stable profitability net of costs across regimes
- Indic features demonstrate:
  - stability under perturbations
  - improved regime separation metrics
  - no placebo profitability
- All modules are replayable and gate-compliant

---

## 6) Phase 4: Adaptive Intelligence (EARNHFT Upgrade)

**Objective:** Introduce hierarchical RL as a controlled upgrade after backbone maturity.

### 6.1 Deliverables

#### A) EARNHFT architecture
- Router: selects agent profile
- Agent: execution policy
- Teacher/Q-Teacher: offline training pipeline

#### B) Training and deployment discipline
- RL trained exclusively in simulation and paper trading first
- Inference deployed via deterministic runtime (ONNX preferred)
- Policy updates are versioned and gate-validated

#### C) Safety posture
- RL outputs bounded parameters first (sizing/spreads/throttles)
- Full policy control only if it passes:
  - replay parity
  - robustness
  - drawdown containment
  - regime-sliced stability

### 6.2 Acceptance Criteria
- RL policy improves execution quality without harming:
  - replay parity
  - risk containment
  - observability/auditability
- Rollback is instant (policy version selection)

---

## 7) Release Gates and Promotion Workflow

### 7.1 Promotion states
- Research → Candidate → Shadow (paper) → Limited Live → Production

### 7.2 Required evidence at each step
- Gate reports (G0–G4)
- Replay parity evidence
- Cost sensitivity results
- Regime-sliced metrics
- Run manifests + WAL checksums

---

## 8) Definition of Done (DoD)

A phase is "done" only when:
- acceptance criteria are met
- CI includes regression tests for the new capabilities
- replay parity remains intact
- documentation is updated
