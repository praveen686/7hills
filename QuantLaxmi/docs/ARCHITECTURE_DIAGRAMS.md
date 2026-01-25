# QuantLaxmi Architecture Diagrams
## Event Flow, WAL/Replay Loop, Gates Pipeline, and Research Promotion

**Status:** Visual architecture reference aligned to Charter v2.1 and Production Contract.
**Format:** Mermaid diagrams suitable for Markdown rendering.

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

## 2) Canonical Event Pipeline (Tick → Decision → Order → Fill)

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

## 3) WAL + Manifest Binding

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

## 4) Replay Loop and Replay Parity Gate (G1)

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

## 5) Gates Pipeline (G0–G4) and Promotion States

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

## 6) Regime Router v1 (Deterministic) → EARNHFT Router v2 Upgrade Path

```mermaid
flowchart TB
  ME[Market Events] --> F[Features\nvol/liquidity/trend/toxicity]
  F --> R1[Regime Router v1\ndeterministic]
  R1 --> P[Select Profile\nsizing/limits/behavior]
  P --> S[Strategy Execution]

  subgraph Later[Optional Phase 4]
    R2[EARNHFT Router v2\npolicy selector]
    A[Agent Pool\nONNX policies]
    T[Teacher/Q-Teacher\noffline]
  end

  F -.-> R2 -.-> A -.-> S
  T --> A
```

---

## 7) Anti-Overfit Suite (G2) Concept Diagram

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

## 8) Risk Escalation Ladder (WARN → THROTTLE → HALT)

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
