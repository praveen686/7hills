# QuantLaxmi Architecture Diagrams
## Event Flow, WAL/Replay Loop, Gates Pipeline, Capital Lifecycle, and Research Promotion

**Status:** Visual architecture reference aligned to Charter v2.2
**Last Updated:** 2026-01-27
**Format:** Mermaid diagrams + ASCII diagrams

---

## 1) System Overview: Backbone + Edge Factory

```mermaid
flowchart LR
  subgraph Backbone[Production Backbone - Rust]
    Conn[Connectors\nBinance SBE, Zerodha WS/REST]
    Canon[Canonical Events\nFixed-Point, Versioned]
    Bus[Event Bus\nOrdered, Deterministic]
    WAL[WAL Writer\nJSONL v1]
    Risk[Risk Policy Engine\nWARN/THROTTLE/HALT]
    Exec[Execution / OMS\nOrder Lifecycle]
    Obs[Observability\nTracing + Metrics]
    Man[Manifests\nSession + Run]
  end

  subgraph Edge[Edge Factory - Research - Python/C++]
    FE[Feature Engineering\nMicrostructure/Options/Carry]
    Indic[Indic Math Engine\nMock Theta/Q-series/Ramanujan]
    Sim[Simulator / Paper Env\nLatency/Slippage/Queue]
    ML[ML Models\nRegime/Ranker/Anomaly]
    RL[RL Training\nEARNHFT optional]
    Art[Artifacts\nModels/Feature Specs]
  end

  Conn --> Canon --> Bus --> WAL
  Bus --> Risk --> Exec
  WAL --> Man
  Risk --> WAL
  Exec --> WAL
  Obs --- Conn
  Obs --- Risk
  Obs --- Exec

  WAL -->|Offline read| Edge
  Edge --> Art -->|Promote via Gates| Backbone
```

---

## 2) System Layer Architecture (Current State)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CAPITAL LAYER (Phase 13+)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Portfolio       │    │ Capital         │    │ Capital         │         │
│  │ Selector        │───►│ Allocation      │───►│ Execution       │         │
│  │ (13.2b NEXT)    │    │ (13.3 PENDING)  │    │ (FUTURE)        │         │
│  └────────▲────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │              BUCKET LAYER (Phase 13.2a) ✅                      │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │                                                                 │
│  ┌────────┴────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Bucket          │    │ Capital         │    │ Bucket          │         │
│  │ Registry        │───►│ Bucket          │───►│ Snapshot        │         │
│  │                 │    │                 │    │                 │         │
│  └────────▲────────┘    └─────────────────┘    └─────────────────┘         │
│           │                      │                                          │
│           │         ┌────────────┴────────────┐                            │
│           │         │ BucketEligibilityBinding │                            │
│           │         └────────────▲────────────┘                            │
│           │                      │                                          │
├───────────┼──────────────────────┼──────────────────────────────────────────┤
│           │              ELIGIBILITY LAYER (Phase 13.1) ✅                  │
├───────────┼──────────────────────┼──────────────────────────────────────────┤
│           │                      │                                          │
│  ┌────────┴────────┐    ┌────────┴────────┐    ┌─────────────────┐         │
│  │ Eligibility     │    │ Eligibility     │    │ Eligibility     │         │
│  │ Validator       │───►│ Decision        │───►│ Constraints     │         │
│  │                 │    │                 │    │                 │         │
│  └────────▲────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │              PROMOTION LAYER (Phase 12.3) ✅                    │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │                                                                 │
│  ┌────────┴────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Promotion       │    │ Promotion       │    │ Paper           │         │
│  │ Validator       │───►│ Decision        │◄───│ Evidence        │         │
│  │                 │    │                 │    │                 │         │
│  └────────▲────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │              TOURNAMENT LAYER (Phase 12.2) ✅                   │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │                                                                 │
│  ┌────────┴────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Tournament      │    │ Leaderboard     │    │ Attribution     │         │
│  │ Runner          │───►│ V1              │───►│ Summary         │         │
│  │                 │    │                 │    │                 │         │
│  └────────▲────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │              STRATEGY LAYER (Phase 12.1) ✅                     │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │                                                                 │
│  ┌────────┴────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Strategy        │    │ Strategy        │    │ Strategy        │         │
│  │ Registry        │───►│ Context         │───►│ Decision        │         │
│  │                 │    │                 │    │                 │         │
│  └────────▲────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │              GATES LAYER (Phase 2) ✅                           │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           │                                                                 │
│  ┌────────┴────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ G0 DataTruth    │    │ G2 Backtest     │    │ G4 Deploy       │         │
│  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤         │
│  │ G1 Replay       │    │ G3 Robustness   │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           CORE LAYER (Phase 1) ✅                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ WAL             │    │ Replay          │    │ Events          │         │
│  │                 │    │ Engine          │    │ (Canonical)     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Models          │    │ Connectors      │    │ Executor        │         │
│  │                 │    │ (Binance/Zerodha)│    │                 │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3) Capital Lifecycle Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CAPITAL LIFECYCLE                                  │
└──────────────────────────────────────────────────────────────────────────────┘

  Tournament          Paper             Promotion         Eligibility
  Selection           Evidence          Decision          Decision
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  ┌───────┐         ┌───────┐         ┌───────┐         ┌───────┐
  │Rank by│         │Collect│         │Enforce│         │Check  │
  │Alpha  │────────►│Paper  │────────►│"No G3 │────────►│Metrics│
  │Score  │         │Trades │         │w/o    │         │Thresh-│
  │       │         │       │         │Paper" │         │olds   │
  └───────┘         └───────┘         └───────┘         └───────┘
      │                  │                  │                  │
      │ LeaderboardV1    │ PaperEvidence    │ PromotionDecision│ EligibilityDecision
      ▼                  ▼                  ▼                  ▼
  ┌───────┐         ┌───────┐         ┌───────┐         ┌───────┐
  │SHA-256│         │SHA-256│         │SHA-256│         │SHA-256│
  │Digest │         │Digest │         │Digest │         │Digest │
  └───────┘         └───────┘         └───────┘         └───────┘
                                                              │
                                                              ▼
  ┌───────┐         ┌───────┐         ┌───────┐         ┌───────┐
  │Capital│         │Bucket │         │Bucket │         │Bucket │
  │Bucket │◄────────│Binding│◄────────│Binding│◄────────│Snapshot│
  │       │         │       │         │Decision│         │       │
  └───────┘         └───────┘         └───────┘         └───────┘
      │                  │                  │                  │
      │ CapitalBucket    │ BucketEligibility│ BucketBinding   │ BucketSnapshot
      ▼                  ▼    Binding       ▼   Decision      ▼
  ┌───────┐         ┌───────┐         ┌───────┐         ┌───────┐
  │Venue  │         │Strategy│         │SHA-256│         │SHA-256│
  │Isolated│         │Bound  │         │Digest │         │Digest │
  └───────┘         └───────┘         └───────┘         └───────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │ PORTFOLIO       │
                                                    │ SELECTOR        │
                                                    │ (Phase 13.2b)   │
                                                    └─────────────────┘
```

---

## 4) Venue Isolation Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VENUE ISOLATION                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │         BucketRegistry               │
                    └──────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │   CRYPTO      │         │   INDIA       │         │   PAPER       │
  │   VENUE       │         │   VENUE       │         │   VENUE       │
  └───────────────┘         └───────────────┘         └───────────────┘
          │                           │                           │
    ┌─────┴─────┐               ┌─────┴─────┐                     │
    │           │               │           │                     │
    ▼           ▼               ▼           ▼                     ▼
┌───────┐   ┌───────┐       ┌───────┐   ┌───────┐           ┌───────┐
│Binance│   │Binance│       │  NSE  │   │  NSE  │           │ Any   │
│ Perp  │   │ Spot  │       │Futures│   │Options│           │Simula-│
│(USDT) │   │(USDT) │       │ (INR) │   │ (INR) │           │ tion  │
└───────┘   └───────┘       └───────┘   └───────┘           └───────┘
    │           │               │           │                   │
    │    ██████████████████████████████████████████████████     │
    │    █                                                █     │
    │    █  ISOLATION BOUNDARY: No cross-venue capital    █     │
    │    █  flow without explicit bridge (future phase)   █     │
    │    █                                                █     │
    │    ██████████████████████████████████████████████████     │
    │                                                           │
    ▼                                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Each bucket enforces:                                          │
│  - Single venue                                                 │
│  - Single currency                                              │
│  - Explicit constraints (max notional, max strategies, etc.)    │
│  - Risk class declaration                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5) Canonical Event Pipeline (Tick → Decision → Order → Fill)

```mermaid
sequenceDiagram
  autonumber
  participant V as Venue Feed
  participant C as Connector
  participant E as Canonicalizer
  participant B as Event Bus
  participant S as Strategy
  participant R as Risk Engine
  participant O as OMS/Executor
  participant W as WAL

  V->>C: raw message (SBE/WS)
  C->>E: parsed venue event
  E->>B: Canonical Quote/Trade/Depth (fixed-point)
  B->>W: append MarketEvent
  B->>S: deliver MarketEvent (ordered)
  S->>W: append DecisionEvent (decision_id)
  S->>R: propose OrderIntent (decision_id)
  R->>W: append RiskEvent (risk_decision_id)
  R->>O: approve/reject intent
  O->>W: append OrderEvent (order_id)
  O->>W: append FillEvent (fill_id, fees)
```

---

## 6) WAL + Manifest Binding

```mermaid
flowchart TB
  WAL[(WAL\nimmutable log)]
  CFG[(Config Snapshot)]
  CODE[(Code Hash\ncommit/build)]
  GATE[(Gate Report)]
  RUN[Run Manifest\nbinds all hashes]
  SES[Session Manifest\nuniverse + schema + sources]

  SES --> RUN
  WAL --> RUN
  CFG --> RUN
  CODE --> RUN
  GATE --> RUN
```

---

## 7) Replay Loop and Replay Parity Gate (G1)

```mermaid
flowchart LR
  WAL[(WAL)]
  CFG[(Config)]
  CODE[(Code Hash)]
  Replay[Replay Engine]
  Trace1[Original Decision Trace Hash]
  Trace2[Replay Decision Trace Hash]
  G1[G1 ReplayParity\npass/fail]

  WAL --> Replay
  CFG --> Replay
  CODE --> Replay
  Replay --> Trace2
  Trace1 --> G1
  Trace2 --> G1
```

---

## 8) Gates Pipeline (G0–G4) and Promotion States

```mermaid
flowchart LR
  subgraph States[Promotion States]
    R0[Research]
    R1[Candidate]
    R2[Shadow - Paper]
    R3[Limited Live]
    R4[Production]
  end

  subgraph Gates[Gates]
    G0[G0 DataTruth]
    G1[G1 ReplayParity]
    G2[G2 Anti-Overfit Suite]
    G3[G3 Robustness]
    G4[G4 Deployability]
  end

  R0 --> R1 --> G0 --> G1 --> G2 --> G3 --> G4 --> R2
  R2 -->|Evidence + Stability| R3 -->|Sustained| R4
```

---

## 9) Decision Artifact Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECISION ARTIFACT CHAIN (All SHA-256 Hashed)             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Tournament  │    │ Promotion   │    │ Eligibility │    │ Bucket      │
│ Manifest    │───►│ Decision    │───►│ Decision    │───►│ Binding     │
│             │    │             │    │             │    │ Decision    │
├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤
│strategy_id  │    │strategy_id  │    │strategy_id  │    │bucket_id    │
│rank         │    │accepted     │    │status       │    │strategy_id  │
│alpha_score  │    │reasons[]    │    │checks[]     │    │accepted     │
│digest ──────┼───►│digest ──────┼───►│digest ──────┼───►│digest       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
   Immutable          Immutable          Immutable          Immutable
   Artifact           Artifact           Artifact           Artifact

                              │
                              ▼
                    ┌─────────────────┐
                    │ Bucket          │
                    │ Snapshot        │
                    ├─────────────────┤
                    │ buckets[]       │
                    │ bindings[]      │
                    │ digest          │
                    │ taken_at        │
                    └─────────────────┘
                              │
                              ▼
                         Immutable
                         Audit Trail
```

---

## 10) Anti-Overfit Suite (G2) Concept Diagram

```mermaid
flowchart LR
  D[Real Data WAL] --> BT[Backtest / Eval]
  D --> TS[Time-Shifted]
  D --> PR[Permutation within day/regime]
  D --> RE[Random Entry Baseline]
  L[Labels] --> LS[Label Shuffle - supervised]

  TS --> BT
  PR --> BT
  RE --> BT
  LS --> BT

  BT --> PASS{Must FAIL\non controls}
  PASS -->|If profitable| STOP[Fail G2\nGhost Alpha]
  PASS -->|If collapses| OK[Pass G2\ncontinue]
```

---

## 11) Risk Escalation Ladder (WARN → THROTTLE → HALT)

```mermaid
stateDiagram-v2
  [*] --> NORMAL
  NORMAL --> WARN: threshold crossed
  WARN --> THROTTLE: persists / worsens
  THROTTLE --> HALT: critical breach
  WARN --> NORMAL: recovery
  THROTTLE --> NORMAL: recovery
  HALT --> NORMAL: manual + policy conditions
```

---

## 12) Test Coverage Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST COVERAGE BY LAYER                            │
└─────────────────────────────────────────────────────────────────────────────┘

Layer                    Tests    Coverage Focus
─────────────────────────────────────────────────────────────────────────────
Capital Buckets          13       Venue isolation, binding, snapshots
Capital Eligibility      13       Invariants, policy presets, digests
Promotion                8        Paper evidence requirement, gates
Tournament               10+      Ranking, determinism, manifests
Strategy SDK             39       Registry, context, decisions
Gates (G0-G4)            20+      Validation, replay parity
Models                   57       Schemas, serialization
Runner-Crypto            52       Backtest, paper, integration
─────────────────────────────────────────────────────────────────────────────
TOTAL                    280+     Full workspace passing
```

---

*End of Architecture Diagrams*
