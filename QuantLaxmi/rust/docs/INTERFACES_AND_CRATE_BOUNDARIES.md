# QuantLaxmi Interfaces and Crate Boundaries
## Enforceable Module Contracts for Connectors, Events, WAL, Replay, Gates, Risk, Execution, and Strategy

**Status:** Implementable engineering spec.
**Purpose:** Prevent architectural drift and ensure deterministic, replayable, gate-governed evolution of QuantLaxmi.

---

## Table of Contents

1. [Design Rules](#1-design-rules)
2. [Crate Boundary Map](#2-crate-boundary-map)
3. [Canonical Events Contracts](#3-canonical-events-contracts)
4. [Connector Interfaces](#4-connector-interfaces)
5. [Event Bus and Ordering](#5-event-bus-and-ordering)
6. [WAL Interfaces](#6-wal-interfaces)
7. [Replay Interfaces](#7-replay-interfaces)
8. [Gate Interfaces](#8-gate-interfaces)
9. [Risk Policy Interfaces](#9-risk-policy-interfaces)
10. [Execution/OMS Interfaces](#10-executionoms-interfaces)
11. [Strategy Interfaces](#11-strategy-interfaces)
12. [Router Interfaces](#12-router-interfaces)
13. [Configuration and Hashing](#13-configuration-and-hashing)
14. [Observability Integration](#14-observability-integration)
15. [Testing Contracts](#15-testing-contracts)
16. [Anti-Patterns (Forbidden Couplings)](#16-anti-patterns-forbidden-couplings)
17. [Appendix: Suggested File Layout](#17-appendix-suggested-file-layout)

---

## 1) Design Rules

### 1.1 "Single Truth" rule
All venues must normalize into the **same canonical event structs** (fixed-point), and all downstream components consume only canonical events.

### 1.2 "Replay First" rule
Every module that transforms events must do so in a way that can be replayed deterministically from the WAL.

### 1.3 "Gate Governed" rule
Anything that influences execution (signals, models, parameters, policies) must be promotable only via the gates pipeline.

### 1.4 "No Float in Canonical Layer" rule
Canonical events and their serialization must never depend on floating-point parsing or formatting.

### 1.5 "Deterministic ordering" rule
Strategies must see events in a stable order. Concurrency must not alter effective order at the strategy boundary.

---

## 2) Crate Boundary Map

### 2.1 Required production crates (Backbone)
- **quantlaxmi-events**: canonical event types + fixed-point utilities + stable serialization
- **quantlaxmi-wal**: WAL record types + writer/reader + checksum helpers
- **quantlaxmi-replay** (optional crate, or module in core): replay engine and trace hashing
- **quantlaxmi-gates**: G0–G4 gate implementations and reports
- **quantlaxmi-core**: shared infrastructure (logging/tracing, time, common errors)
- **quantlaxmi-connectors-binance**: Binance ingestion/parsing (SBE or WS depending on mode)
- **quantlaxmi-connectors-zerodha**: Zerodha ingestion/parsing (WS/REST)
- **quantlaxmi-risk**: policy engine, escalation ladder, rule evaluation
- **quantlaxmi-executor**: OMS/order lifecycle/execution routes
- **quantlaxmi-runner-common**: manifest IO, shared runner CLI glue
- **apps/quantlaxmi-crypto**, **apps/quantlaxmi-india**: top-level binaries

### 2.2 Research boundary (Edge Factory)
- **research/** (directory or crate group):
  - indic math feature transforms
  - training pipelines (ML/RL)
  - simulation harness
  - artifact emitters (models, features, specs)

**Research outputs artifacts; production consumes artifacts only via governed loaders.**

---

## 3) Canonical Events Contracts

### 3.1 Canonical numeric type
All prices/qty in canonical events use:

- `mantissa: i64`
- `exponent: i32`

Canonical helper types (recommended):
- `FixedDecimal { mantissa: i64, exponent: i32 }`
- strict parsing function `parse_decimal_str_deterministic(s: &str) -> FixedDecimal`

### 3.2 Event enums
Recommended core enum:

- `enum CanonicalEvent { Market(MarketEvent), Trading(TradingEvent), Session(SessionEvent) }`

Where:
- `MarketEvent` includes Quote/Depth/Trade/Funding/Instrument
- `TradingEvent` includes Decision/Risk/Order/Fill/Feature (optional)

### 3.3 Stable serialization
Canonical serialization must be stable for hashing:
- fixed field order
- fixed timestamp formats
- fixed numeric formats (mantissa/exponent only)

Implement:
- `fn to_canonical_bytes(&self) -> Vec<u8>` (or writer-based)

---

## 4) Connector Interfaces

### 4.1 Purpose
Connectors translate venue-specific protocols into canonical events, and may include reconnection logic, sequence management, and health reporting.

### 4.2 Connector trait (recommended)

```rust
trait MarketDataConnector {
    type Error;

    fn venue(&self) -> Venue;
    fn start(&mut self, sink: impl CanonicalEventSink) -> Result<(), Self::Error>;
    fn stop(&mut self) -> Result<(), Self::Error>;

    // Optional: snapshot / resync hooks
    fn request_resync(&mut self) -> Result<(), Self::Error>;
}
```

### 4.3 CanonicalEventSink interface

```rust
trait CanonicalEventSink {
    fn on_event(&mut self, e: CanonicalEvent);
    fn on_health(&mut self, h: HealthEvent);
}
```

### 4.4 Connector responsibilities (must)

- convert to canonical fixed-point
- attach venue timestamps and capture timestamps
- include sequence numbers if available
- emit SessionEvents (connect/disconnect/gap/resync)
- never emit floats into canonical events

### 4.5 Connector responsibilities (must not)

- implement strategy logic
- compute features
- compute signals
- place orders directly

---

## 5) Event Bus and Ordering

### 5.1 Bus responsibilities

- merge multiple streams into one ordered stream (per symbol or per venue)
- ensure deterministic ordering
- forward events to:
  - WAL writer (always)
  - strategy engines
  - risk and execution pipelines (via strategy output)

### 5.2 Ordering key

Canonical ordering key must be defined:

1. venue timestamp (if present)
2. capture timestamp
3. sequence number (if present)
4. symbol
5. stable tie-breaker

### 5.3 Strategy boundary guarantee

Strategy receives events in the same order on replay as on live capture.

Violation → fails G1 ReplayParity.

---

## 6) WAL Interfaces

### 6.1 WAL record type

WAL should store records as:

`WalRecord { schema_version, record_type, payload, ids, checksums(optional) }`

Minimum record types:

- MarketEvent records
- DecisionEvent records
- RiskEvent records
- OrderEvent records
- FillEvent records
- Session/Health records

### 6.2 WAL writer

```rust
trait WalWriter {
    fn append(&mut self, record: WalRecord) -> Result<(), WalError>;
    fn flush(&mut self) -> Result<(), WalError>;
    fn rotate(&mut self) -> Result<(), WalError>; // optional
}
```

### 6.3 WAL reader

```rust
trait WalReader {
    fn next(&mut self) -> Result<Option<WalRecord>, WalError>;
    fn seek(&mut self, pos: WalPosition) -> Result<(), WalError>; // optional
}
```

### 6.4 WAL invariants

- append-only
- immutable after close
- checksum recorded in run manifest
- stable schema

---

## 7) Replay Interfaces

### 7.1 Replay engine responsibilities

- read WAL
- reconstruct event stream
- feed to strategy and policy layers
- compute decision trace hash
- verify parity

### 7.2 Replay API

```rust
struct ReplayInput {
    wal_paths: Vec<PathBuf>,
    config_snapshot: PathBuf,
    code_hash: String,
}

struct ReplayOutput {
    decision_trace_hash: String,
    metrics: serde_json::Value,
}

trait ReplayEngine {
    fn run(&self, input: ReplayInput) -> Result<ReplayOutput, ReplayError>;
}
```

### 7.3 Replay parity definition

- Hash computed over canonical serialization of all DecisionEvents (minimum)
- Recommended extension: include RiskEvents and Order intents

---

## 8) Gate Interfaces

### 8.1 Gate suite output format

```rust
struct GateResult {
    gate_name: String,
    pass: bool,
    metrics: serde_json::Value,
    notes: Vec<String>,
}

struct GateSummary {
    pass: bool,
    results: Vec<GateResult>,
    report_hash: String,
}
```

### 8.2 Gate input contract

```rust
struct GateInput {
    session_manifest: PathBuf,
    run_manifest: PathBuf,
    wal_paths: Vec<PathBuf>,
    config_snapshot: PathBuf,
    artifacts: Vec<PathBuf>,
}
```

### 8.3 Gate suite API

```rust
trait Gate {
    fn name(&self) -> &'static str;
    fn run(&self, input: &GateInput) -> GateResult;
}

trait GateSuite {
    fn run_all(&self, input: &GateInput) -> GateSummary;
}
```

### 8.4 Gate enforcement

- binaries must provide gates command
- CI must run gates on fixture runs
- promotion pipeline must block if mandatory gates fail

---

## 9) Risk Policy Interfaces

### 9.1 Policy engine responsibilities

- deterministic pre-trade checks
- escalation ladder
- write RiskEvents to WAL
- provide reason codes on reject/throttle/halt

### 9.2 Risk input/output

```rust
struct RiskInput {
    decision: DecisionEvent,
    market_snapshot_ref: Option<SnapshotRef>,
    positions: PositionState,
    limits: RiskLimits,
}

enum RiskVerdict {
    Allow { risk_decision_id: String, constraints: Constraints },
    Throttle { risk_decision_id: String, constraints: Constraints, reason: String },
    Reject { risk_decision_id: String, reason: String },
    Halt { risk_decision_id: String, scope: HaltScope, reason: String },
}
```

### 9.3 Escalation ladder contract

- **WARN**: log + metric (+ optional alert hook)
- **THROTTLE**: reduce size/frequency/aggressiveness
- **HALT**: stop symbol/venue/global; optional flatten under policy

No universal hard-coded thresholds in code; policies are versioned configs with hard safety caps.

---

## 10) Execution/OMS Interfaces

### 10.1 Execution responsibilities

- translate order intents into venue orders
- track lifecycle (new/ack/reject/partial/fill/cancel)
- write OrderEvents and FillEvents to WAL
- expose ack latency and slippage metrics

### 10.2 Order intent contract

```rust
struct OrderIntent {
    decision_id: String,
    risk_decision_id: String,
    symbol: String,
    side: Side,
    qty: FixedDecimal,
    price: Option<FixedDecimal>,
    order_type: OrderType,
    time_in_force: TimeInForce,
}
```

### 10.3 Execution API

```rust
trait Executor {
    fn submit(&mut self, intent: OrderIntent) -> Result<OrderAck, ExecError>;
    fn cancel(&mut self, order_id: String) -> Result<(), ExecError>;
    fn poll(&mut self) -> Result<Vec<OrderUpdate>, ExecError>; // or callback based
}
```

---

## 11) Strategy Interfaces

### 11.1 Strategy responsibilities

- consume canonical market events
- compute features (or call feature engine)
- propose decisions and order intents
- never bypass risk engine
- write DecisionEvents to WAL

### 11.2 Strategy API

```rust
trait Strategy {
    fn id(&self) -> &'static str;

    fn on_event(
        &mut self,
        e: &CanonicalEvent,
        ctx: &StrategyContext
    ) -> Vec<DecisionEvent>;
}
```

### 11.3 Strategy output discipline

Strategy emits DecisionEvents and OrderIntents through a controlled pipeline:

`Strategy → Risk → Executor`

Strategy must never call Executor directly.

---

## 12) Router Interfaces

### 12.1 Router v1 (deterministic)

Router selects a profile based on features.

```rust
trait RegimeRouter {
    fn classify(&self, features: &RouterFeatures) -> RegimeState;
    fn select_profile(&self, state: &RegimeState) -> ActiveProfile;
}
```

### 12.2 Router v2 (EARNHFT optional)

Router selects among agent profiles (ONNX policies). Still must:

- be deterministic
- be versioned and hashable
- pass gates before promotion

---

## 13) Configuration and Hashing

### 13.1 Config snapshot requirement

Every run must write:

- full config snapshot to disk
- config hash to run manifest

### 13.2 Hash construction

Hashes must bind:

- code hash (git + build metadata)
- config hash
- WAL checksum(s)
- gate report checksum
- decision trace hash

---

## 14) Observability Integration

### 14.1 Tracing spans

All spans must include:

- session_id, run_id, symbol, venue, strategy_id, decision_id, risk_decision_id, order_id

### 14.2 Metrics minimum set

- feed latency p50/p95/p99
- quote_age p50/p95/p99
- reconnect/gap counts
- order ack latency
- fill ratio
- slippage and fee attribution

---

## 15) Testing Contracts

### 15.1 Required test types

- unit tests (parsers, fixed-point, ordering)
- integration tests (fixture WAL replay parity)
- property tests (ordering invariants)
- fuzz tests for protocol parsers (recommended)
- CI gate tests (G0 and G4 minimum in Phase 1)

### 15.2 Fixtures

Maintain a small fixture set with:

- minimal WAL samples
- manifests
- expected decision hash outputs

---

## 16) Anti-Patterns (Forbidden Couplings)

**Forbidden:**

- connector calling strategy
- strategy calling executor directly
- WAL schema tied to venue-specific structs
- floating-point parsing in canonical event ingestion
- nondeterministic ordering at strategy boundary
- "temporary" bypass of gates for promotion

---

## 17) Appendix: Suggested File Layout

Recommended placements:

```
quantlaxmi-events/src/*
quantlaxmi-wal/src/*
quantlaxmi-gates/src/*
quantlaxmi-replay/src/* (optional)
research/indic/*
research/sim/*
docs/* (charter, contract, roadmap, diagrams, boundaries)
```

---

*End of document.*
