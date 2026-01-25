# QuantLaxmi Trading Platform
## Master Technical Specification & Project Charter v2.1 (Exhaustive Pole Star)

**Status:** Authoritative "Pole Star" specification for QuantLaxmi
**Applies to:** India FNO + Crypto (24×7)
**Core commitments:** Determinism, audit-grade replayability, governed innovation, friction-aware profitability, production-grade observability and safety controls.

---

## Table of Contents

1. [Project Mission and Vision](#10-project-mission-and-vision)
2. [Project Goals and Success Criteria](#20-project-goals-and-success-criteria)
3. [Guiding Principles (Project Constitution)](#30-guiding-principles-project-constitution)
4. [Scope and Non-Scope](#40-scope-and-non-scope)
5. [High-Level Architecture: Backbone + Edge Factory](#50-high-level-architecture-backbone--edge-factory)
6. [Canonical Data Model (Single Truth Across Venues)](#60-canonical-data-model-single-truth-across-venues)
7. [WAL + Manifests + Replay Contract](#70-wal--manifests--replay-contract)
8. [Technology Mandate and Language Roles](#80-technology-mandate-and-language-roles)
9. [Signal Generation and Alpha Philosophy](#90-signal-generation-and-alpha-philosophy)
10. [Strategy Optimization and Adaptation: ML and RL](#100-strategy-optimization-and-adaptation-ml-and-rl)
11. [Reliability and Validation: Promotion Gates](#110-reliability-and-validation-promotion-gates-g0g4)
12. [Execution and Risk: Production Behavior Contract](#120-execution-and-risk-production-behavior-contract)
13. [Observability and Operations](#130-observability-and-operations)
14. [Repository Structure Mapping](#140-repository-structure-mapping-quantlaxmi-implementation-reality)
15. [Security, Safety, and Compliance](#150-security-safety-and-compliance-operational-discipline)
16. [Phased Development Roadmap](#160-phased-development-roadmap-dependency-correct)
17. [Formal Definitions and Glossary](#170-formal-definitions-and-glossary)
18. [Appendix: Mandatory Acceptance Checklists](#180-appendix-mandatory-acceptance-checklists)

---

## 1.0 Project Mission and Vision

This document defines the formal charter and technical master specification for the QuantLaxmi trading platform. It establishes the project's strategic objectives, scope boundaries, architecture, governance rules, implementation standards, and measures of success.

### 1.1 Mission

To build a **profitable, production-grade, auditably correct multi-venue trading platform** (India FNO + Crypto) whose edge is driven by:

1) **Microstructure-aware execution** (queue/imbalance/latency + robust fill realism)
2) **Regime-aware signal generation**, including **Indic mathematics** (e.g., Ramanujan/mock theta/q-series feature families) as governed, testable feature transforms
3) **Adaptive strategy refinement**, including **Hierarchical Reinforcement Learning (EARNHFT)** as a controlled evolution path—introduced only after the deterministic backbone and gates are mature

All of the above must remain **deterministic, replayable, observable, and governed by promotion gates**.

### 1.2 Vision (What "done" looks like)

QuantLaxmi is a system where **every live decision can be reproduced**, every unit of PnL can be attributed (fees, slippage, latency, adverse selection), and innovation (new math/models) is introduced safely through a governed pipeline that prevents "ghost alpha."

---

## 2.0 Project Goals and Success Criteria

QuantLaxmi is considered successful only when all criteria below are met.

### 2.1 Audit-Grade Replayability

- Every live run must be reproducible such that **decision traces match a deterministic hash** (SHA-256).
- **Replay is the primary debugging path:** if a failure cannot be replayed from WAL + config + code hash, it is not considered fixed.

### 2.2 Rigorous Validation Through Promotion Gates

Strategies/models must pass automated gates G0–G4, including a **formal anti-overfit suite** (placebo/time-shift/permutation/baselines/ablations). Any strategy that appears profitable on controls fails promotion.

### 2.3 End-to-End PnL Attribution

For every fill, the system must compute and record:

- fee attribution
- slippage attribution (model vs realized)
- latency/queue adverse-selection proxies (where feasible)
- realized vs expected execution quality
- regime context at decision-time

### 2.4 Robust Profitability Across Regimes

- Profitability must be stable across regimes (bull/bear/choppy, high/low liquidity, high/low vol).
- The system must detect regime shifts and switch execution/strategy profiles via a **Router** (Regime Router v1; EARNHFT Router v2).

### 2.5 Governed Innovation (Research Quarantine)

- Research components (Python/C++/GPU) are **quarantined** from production.
- Only features/models that pass gates (and are replayable) can be promoted into production runners.

---

## 3.0 Guiding Principles (Project Constitution)

1) **Determinism over Cleverness**
   If results cannot be reproduced from captured data + configuration + code hash, they are not real.

2) **Profitability Must Survive Friction**
   All claimed edge must be validated under realistic fees, slippage, latency, partial fills, and execution constraints.

3) **Research Is Quarantined Until Promoted**
   Advanced constructs (RL, EBMs, mock theta, deep models, GPU kernels) begin as feature generators, filters, or rankers—never as uncontrolled execution drivers.

4) **Everything Is an Event, Everything Is Correlated**
   Tick → Features → Decision → Risk → Order → Fill must be recorded with correlation IDs and replayable.

5) **Multi-Venue, Single Canonical Truth**
   Binance and NSE/Zerodha are different sources, but the system has one canonical event model and one WAL contract.

6) **The Backbone Comes Before Alpha Complexity**
   A profitable system requires truth, safety, and observability before it requires sophistication.

---

## 4.0 Scope and Non-Scope

### 4.1 In Scope

- Multi-venue ingestion: Binance (SBE), Zerodha (WS/REST), extension-ready to other venues
- Canonical event model for all venues
- WAL capture, session/run manifests, deterministic replay
- Promotion gates and CI enforcement
- Execution engine (order lifecycle) with risk policy enforcement
- Paper trading and exchange simulation (latency, slippage, queue realism)
- Feature engines for microstructure, options surface, carry/funding
- Research quarantine environment (Python/C++) and promotion pipeline
- Regime Router v1 (deterministic) and upgrade path to EARNHFT (v2)

### 4.2 Explicitly Out of Scope (initially)

- Full L3 order book reconstruction for all venues (optional later)
- Ultra-optimized zero-copy everywhere as a hard requirement in Phase 1 (targeted optimization later)
- Unbounded "all models at once" production deployment (must be gated, incremental)
- A single monolithic "do everything AI" controller in early phases

---

## 5.0 High-Level Architecture: Backbone + Edge Factory

QuantLaxmi follows a strict separation.

### 5.1 Platform Backbone (Production Spine – Rust)

**Purpose:** deterministic ingestion, canonicalization, WAL+manifests, replay, gating, risk enforcement, execution, observability.

**Key modules:**

- **Connectivity & parsing:** Binance SBE decoder, Zerodha adapters
- **Canonical event normalization:** fixed-point and schema enforcement
- **Event bus:** controlled concurrency, stable ordering rules
- **WAL writer/reader:** immutable, schema-stable
- **Replay engine:** invariant trace reproduction
- **Risk policy engine:** deterministic "circuit breaker" with escalation states
- **Execution/OMS:** order lifecycle, reconciliation hooks
- **Observability:** tracing + metrics + correlation IDs end-to-end

### 5.2 Edge Factory (Research Spine – Python/C++)

**Purpose:** alpha research, feature discovery, model training, simulation, and promotion.

**Key modules:**

- **Indic Math Engine:** mock theta / q-series / Ramanujan transforms (as feature generators)
- **Feature engineering:** microstructure + options surface + carry/funding
- **Simulation:** exchange simulator with queue realism
- **Model training:** regime gating models; rankers; RL training pipelines
- **Q-Teacher:** offline teacher/training data generator for RL (optional; gated)
- **Promotion pipeline:** generate artifacts + gate reports + promotion decisions

### 5.3 Router (Regime Router v1 → EARNHFT Router v2)

**Regime Router v1 (mandatory early):**

- Deterministic module classifying market regime and selecting execution/strategy profile.

**EARNHFT Router v2 (optional later):**

- Hierarchical RL router selecting among RL agents; only after Phase 1–2 backbone maturity and gate coverage.

---

## 6.0 Canonical Data Model (Single Truth Across Venues)

### 6.1 Canonical Event Taxonomy

All venues must be normalized into the same types. All events are deterministic and serializable with stable schema.

#### Market Events

- **QuoteEvent (L1):** bid_px, ask_px, bid_qty, ask_qty (fixed-point)
- **DepthEvent (L2 Snapshot/Delta):** levels or deltas (optional but supported)
- **TradeEvent:** last trade events
- **FundingEvent:** rate, timestamps (crypto)
- **InstrumentEvent:** expiry, strike, right, multiplier, tick, lot, symbol mapping
- **SessionEvent:** connect/disconnect/gaps/resync/health

#### Trading Pipeline Events

- **FeatureEvent (optional):** feature vector hash + summary stats (not raw vector by default)
- **DecisionEvent:** strategy_id, decision_id, action, confidence, rationale hash
- **RiskEvent:** risk_decision_id, allowed/blocked, rule outcomes
- **OrderEvent:** order intent + ack/reject + cancel/replace
- **FillEvent:** fill qty/px, fees, venue fields, timestamps

### 6.2 Required Correlation Identifiers (Mandatory)

- `session_id`
- `run_id`
- `venue`
- `symbol`
- `strategy_id`
- `decision_id`
- `risk_decision_id`
- `order_id`
- `fill_id`

### 6.3 Deterministic Numeric Contract

- Canonical layer uses fixed-point mantissa/exponent representation.
- No float intermediates in canonical event parsing.
- Where floating-point inference is unavoidable, inference must be deterministic with explicit rounding rules and a defined nondeterminism budget.

### 6.4 Schema Evolution Contract

- Events are versioned (e.g., `schema_version`).
- Backward-compatible changes only; breaking changes require a new version and migration tooling.
- Canonical serialization must be stable for hashing.

---

## 7.0 WAL + Manifests + Replay Contract

### 7.1 WAL Contract

WAL is immutable and append-only. It records:
- canonical market events
- all decisions and risk evaluations
- all order lifecycle transitions
- all fills and fee attribution events
- system health and session events

**Phase 1 format:** JSONL acceptable.
**Later:** binary framed WAL for throughput.

### 7.2 Manifests (Cryptographic Binding)

Each run is bound to:
- code hash / strategy hash
- config snapshot hash
- WAL checksums
- gate report checksums
- inventory/universe definitions

### 7.3 Replay Contract (Invariant)

`Replay(WAL, Config, CodeHash) -> DecisionTrace`

Invariant: decision trace hash must match the original run.

### 7.4 Deterministic Hashing Rules

- Stable canonical serialization
- Stable sorting keys for event streams
- SHA-256 computed over DecisionEvent (minimum), optionally extended

### 7.5 Nondeterminism Budget (Explicit Policy)

- Event ordering stable by `(venue_ts, capture_ts, seq_no, symbol)`
- Randomness forbidden unless seeded and recorded
- Concurrency must not change effective order
- Inference must be deterministic or bounded and validated

---

## 8.0 Technology Mandate and Language Roles

### 8.1 Rust (Production Spine)

Rust is default for:
- control plane
- network I/O
- protocol parsing
- canonicalization
- WAL + replay
- risk policy engine
- execution/OMS
- observability

Constraints:
- panic-free in production paths
- bounded allocations on hot paths where practical
- stable structured logs and metrics

### 8.2 C++ (Compute Muscle; gated)

Reserved for:
- compute-heavy Indic math transforms when proven bottlenecks
- optional CUDA kernels
- optional LibTorch/ONNX bridging if CPU inference bottlenecks proven

Rules:
- deterministic interfaces
- fixed-point IO boundaries
- full tests and replay safety

### 8.3 Python (Research Spine)

Default for:
- rapid feature iteration
- backtest prototypes
- training pipelines
- RL loops
- simulation experiments

Promotion rule:
- Python outputs artifacts; production consumes validated artifacts only.

---

## 9.0 Signal Generation and Alpha Philosophy

### 9.1 Feature Domains

1) Microstructure
2) Options Surface (India priority)
3) Carry / Funding / Calendar

### 9.2 Indic Mathematics Engine

Indic math treated as feature transforms:
- Ramanujan periodic dictionary features
- Mock theta / q-series transforms
- Continued fraction approximants
- Kuttaka-like constraint solving

Promotion governed by gates.

### 9.3 Execution Feasibility

Trade candidates must include:
- expected slippage
- liquidity feasibility
- adverse-selection proxy
- do-not-trade filters

---

## 10.0 Strategy Optimization and Adaptation: ML and RL

### 10.1 Minimum Safe ML Use

ML introduced first as:
- regime classifier
- trade ranker/filter
- anomaly do-not-trade detector
- parameter adapter

### 10.2 RL (EARNHFT) Integration Plan

RL is late-phase upgrade.

Roles:
- Router
- Agent
- Teacher/Q-Teacher

Safety posture:
- bounded control outputs first
- full gate compliance mandatory

---

## 11.0 Reliability and Validation: Promotion Gates (G0–G4)

### G0 – Data Truth
### G1 – Replay Parity
### G2 – Anti-Overfit Suite
### G3 – Robustness
### G4 – Deployability

(See `docs/PRODUCTION_CONTRACT.md` for implementable definitions and thresholds.)

---

## 12.0 Execution and Risk: Production Behavior Contract

### 12.1 Risk Policy Engine

Deterministic rule enforcement:
- max positions
- max notional
- max loss/day
- spread/quote-age constraints
- liquidity constraints
- rate limiting

### 12.2 Escalation Ladder

- WARN
- THROTTLE
- HALT

### 12.3 Audit Requirements

All risk decisions logged and written to WAL with IDs.

---

## 13.0 Observability and Operations

### 13.1 Tracing
### 13.2 Metrics
### 13.3 Runbooks

---

## 14.0 Repository Structure Mapping (QuantLaxmi Implementation Reality)

Production crates:
- quantlaxmi-core
- quantlaxmi-runner-common
- quantlaxmi-connectors-binance
- quantlaxmi-connectors-zerodha
- quantlaxmi-executor
- quantlaxmi-risk
- quantlaxmi-models
- quantlaxmi-data
- apps/quantlaxmi-crypto
- apps/quantlaxmi-india

Required Phase 1 crates:
- quantlaxmi-events
- quantlaxmi-wal
- quantlaxmi-gates

---

## 15.0 Security, Safety, and Compliance

- keys never in repo
- trading disabled by default
- WAL retention and audit policy defined

---

## 16.0 Phased Development Roadmap (Dependency-Correct)

Phase 1: Platform Spine
Phase 2: Edge Protector
Phase 3: Edge Builders
Phase 4: Adaptive Intelligence

---

## 17.0 Formal Definitions and Glossary

- WAL
- Replay parity
- Gate
- Router
- Ghost alpha

---

## 18.0 Appendix: Mandatory Acceptance Checklists

### 18.1 Run validity
### 18.2 Promotion readiness
