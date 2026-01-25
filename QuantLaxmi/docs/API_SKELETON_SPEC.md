# QuantLaxmi API Skeleton Spec
## Concrete Interfaces, Types, Error Taxonomy, and Invariants (Specification, Not Implementation)

**Status:** Implementable spec.
**Audience:** QuantLaxmi developers implementing the Backbone + Edge Factory architecture.
**Rule:** All production-facing components must conform to this API shape (or an equivalent adapter layer must exist).

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [Shared Types](#2-shared-types)
3. [Canonical Fixed-Point](#3-canonical-fixed-point)
4. [Canonical Events](#4-canonical-events)
5. [WAL Records and Storage](#5-wal-records-and-storage)
6. [Manifests](#6-manifests)
7. [Event Ordering and Deterministic Hashing](#7-event-ordering-and-deterministic-hashing)
8. [Connector API](#8-connector-api)
9. [Event Bus API](#9-event-bus-api)
10. [Strategy API](#10-strategy-api)
11. [Router API](#11-router-api)
12. [Risk API](#12-risk-api)
13. [Execution/OMS API](#13-executionoms-api)
14. [Replay API](#14-replay-api)
15. [Gates API](#15-gates-api)
16. [Observability API](#16-observability-api)
17. [Error Taxonomy](#17-error-taxonomy)
18. [Acceptance Invariants](#18-acceptance-invariants)
19. [Appendix: Suggested Rust Module Mapping](#19-appendix-suggested-rust-module-mapping)

---

## 1) Goals and Non-Goals

### 1.1 Goals
- Provide a consistent contract for all QuantLaxmi modules.
- Make replay parity, gates, and canonical events enforceable by design.
- Prevent connector/strategy/executor coupling and nondeterminism.

### 1.2 Non-Goals
- This document does not prescribe specific internal algorithms.
- This document does not define the UI or external APIs.
- This document is not a replacement for `docs/PRODUCTION_CONTRACT.md`; it operationalizes it.

---

## 2) Shared Types

### 2.1 Identifiers (required)
All IDs must be stable strings (UUID or deterministic hash). IDs must be emitted into WAL/logs.

```rust
type SessionId = String;
type RunId = String;
type StrategyId = String;
type DecisionId = String;
type RiskDecisionId = String;
type OrderId = String;
type FillId = String;
type Venue = enum { Binance, Zerodha, /* ... */ };
type Symbol = String;
```

### 2.2 Time types

Both timestamps should exist where feasible.

```rust
type VenueTsNanos = i64;   // from exchange/venue feed (if provided)
type CaptureTsNanos = i64; // local monotonic-corrected wall-clock capture time
```

### 2.3 Common enums

```rust
enum Side { Buy, Sell }
enum OrderType { Market, Limit, PostOnly, IOC, FOK }
enum TimeInForce { GTC, IOC, FOK, Day }
```

---

## 3) Canonical Fixed-Point

### 3.1 Canonical decimal

```rust
struct FixedDecimal {
    mantissa: i64,
    exponent: i32,
}
```

### 3.2 Required operations (deterministic)

- parse from string: `parse_decimal_str_deterministic(&str) -> Result<FixedDecimal, FixedParseError>`
- normalize: (optional) canonical exponent normalization rules
- comparison and safe scaling with overflow checks

### 3.3 Forbidden

- Parsing via floating-point intermediates.
- Formatting that depends on locale.
- Silent overflow.

---

## 4) Canonical Events

### 4.1 Event envelope (required metadata)

Every event must carry:

- session_id, run_id (run_id may be "unknown" during raw capture but must exist by WAL time)
- venue, symbol
- venue_ts (optional), capture_ts (required)
- seq_no (optional but recommended)

```rust
struct EventMeta {
    session_id: SessionId,
    run_id: RunId,
    venue: Venue,
    symbol: Symbol,
    venue_ts_nanos: Option<VenueTsNanos>,
    capture_ts_nanos: CaptureTsNanos,
    seq_no: Option<u64>,
    schema_version: u32,
}
```

### 4.2 Market events

```rust
struct QuoteEvent {
    meta: EventMeta,
    bid_px: FixedDecimal,
    ask_px: FixedDecimal,
    bid_qty: FixedDecimal,
    ask_qty: FixedDecimal,
}

struct DepthEvent {
    meta: EventMeta,
    // Implementation-defined representation (snapshot or delta)
    // Must be versioned and deterministic.
}

struct TradeEvent {
    meta: EventMeta,
    side: Side,
    px: FixedDecimal,
    qty: FixedDecimal,
}

struct FundingEvent {
    meta: EventMeta,
    funding_rate: FixedDecimal,
    next_funding_ts_nanos: Option<VenueTsNanos>,
}

struct InstrumentEvent {
    meta: EventMeta,
    // Must support India options/futures metadata + crypto symbol metadata
    // Required: tick size, lot size if applicable, multiplier if applicable
}
```

### 4.3 Trading pipeline events

```rust
struct DecisionEvent {
    meta: EventMeta,
    strategy_id: StrategyId,
    decision_id: DecisionId,
    action: String,            // e.g., "ENTER_LONG", "QUOTE_WIDEN", "NO_TRADE"
    confidence: Option<FixedDecimal>,
    rationale_hash: Option<String>,
    features_hash: Option<String>,
}

struct RiskEvent {
    meta: EventMeta,
    strategy_id: StrategyId,
    decision_id: DecisionId,
    risk_decision_id: RiskDecisionId,
    verdict: RiskVerdict,
    rule_outcomes: Vec<RiskRuleOutcome>,
}

enum RiskVerdict {
    Allow,
    Throttle,
    Reject,
    Halt,
}

struct RiskRuleOutcome {
    rule_name: String,
    pass: bool,
    value: Option<String>,
    threshold: Option<String>,
    reason: Option<String>,
}

struct OrderEvent {
    meta: EventMeta,
    strategy_id: StrategyId,
    decision_id: DecisionId,
    risk_decision_id: RiskDecisionId,
    order_id: OrderId,
    state: OrderState,
    intent: Option<OrderIntent>,   // on intent/new
    venue_ack: Option<String>,     // venue-specific ack id if any
}

enum OrderState {
    Intent,
    New,
    Ack,
    Reject,
    CancelRequested,
    CancelAck,
    Filled,
    PartialFill,
    Closed,
}

struct FillEvent {
    meta: EventMeta,
    strategy_id: StrategyId,
    decision_id: DecisionId,
    risk_decision_id: RiskDecisionId,
    order_id: OrderId,
    fill_id: FillId,
    qty: FixedDecimal,
    px: FixedDecimal,
    fee: Option<FixedDecimal>,
    fee_ccy: Option<String>,
}
```

### 4.4 Canonical event enum

```rust
enum CanonicalEvent {
    Quote(QuoteEvent),
    Depth(DepthEvent),
    Trade(TradeEvent),
    Funding(FundingEvent),
    Instrument(InstrumentEvent),
    Decision(DecisionEvent),
    Risk(RiskEvent),
    Order(OrderEvent),
    Fill(FillEvent),
    Session(SessionEvent),
}
```

### 4.5 Session and health events

```rust
struct SessionEvent {
    meta: EventMeta,
    kind: SessionEventKind,
    message: Option<String>,
}

enum SessionEventKind {
    Connect,
    Disconnect,
    GapDetected,
    ResyncStart,
    ResyncComplete,
    Heartbeat,
    HealthWarn,
    HealthCritical,
}
```

---

## 5) WAL Records and Storage

### 5.1 WAL record

WAL stores canonical events in a deterministic wrapper.

```rust
struct WalRecord {
    record_version: u32,
    record_type: WalRecordType,
    event: CanonicalEvent,
    record_checksum: Option<String>, // optional per-record checksum
}

enum WalRecordType {
    Market,
    Trading,
    Session,
}
```

### 5.2 WAL writer

```rust
trait WalWriter {
    fn append(&mut self, r: &WalRecord) -> Result<(), WalError>;
    fn flush(&mut self) -> Result<(), WalError>;
    fn close(&mut self) -> Result<(), WalError>;
}
```

### 5.3 WAL reader

```rust
trait WalReader {
    fn next(&mut self) -> Result<Option<WalRecord>, WalError>;
}
```

### 5.4 Format constraints

- Phase 1: JSONL
- Stable serialization required (see hashing section)
- WAL checksum stored in run manifest

---

## 6) Manifests

### 6.1 Session manifest (minimum)

```rust
struct SessionManifest {
    session_id: SessionId,
    created_ts_nanos: i64,
    venues: Vec<Venue>,
    symbols: Vec<Symbol>,
    schema_versions: Vec<(String, u32)>,
    capture_artifacts: Vec<ArtifactRef>,
}

struct ArtifactRef {
    name: String,
    path: String,
    sha256: String,
}
```

### 6.2 Run manifest (minimum)

```rust
struct RunManifest {
    run_id: RunId,
    session_id: SessionId,
    code_hash: String,
    config_sha256: String,
    wal_paths: Vec<String>,
    wal_sha256: Vec<String>,
    decision_trace_hash: String,
    gate_report_sha256: Option<String>,
    artifacts: Vec<ArtifactRef>,
}
```

---

## 7) Event Ordering and Deterministic Hashing

### 7.1 Ordering key (required)

Order events for replay and hashing by:

1. venue_ts_nanos (if present else MIN)
2. capture_ts_nanos
3. seq_no (if present else MIN)
4. symbol
5. stable tie-breaker (connector-defined, recorded)

### 7.2 Canonical serialization for hashing

All events must implement:

```rust
trait CanonicalSerialize {
    fn to_canonical_bytes(&self) -> Vec<u8>;
}
```

### 7.3 Decision trace hash

At minimum:

- hash all DecisionEvents' canonical bytes in ordered sequence
- SHA-256 over concatenation (or streaming SHA-256)

---

## 8) Connector API

### 8.1 Event sink

```rust
trait CanonicalEventSink {
    fn on_event(&mut self, e: CanonicalEvent);
    fn on_health(&mut self, h: HealthSignal);
}

struct HealthSignal {
    venue: Venue,
    severity: HealthSeverity,
    msg: String,
    ts_nanos: i64,
}

enum HealthSeverity { Info, Warn, Critical }
```

### 8.2 Connector trait

```rust
trait MarketDataConnector {
    type Error;

    fn venue(&self) -> Venue;
    fn start(&mut self, sink: &mut dyn CanonicalEventSink) -> Result<(), Self::Error>;
    fn stop(&mut self) -> Result<(), Self::Error>;

    fn request_resync(&mut self) -> Result<(), Self::Error>; // optional
}
```

Connector must not:

- place orders
- call strategies
- bypass canonical events

---

## 9) Event Bus API

### 9.1 Event bus trait

```rust
trait EventBus {
    fn publish(&mut self, e: CanonicalEvent) -> Result<(), BusError>;
    fn subscribe(&mut self, sub: SubscriberId, handler: Box<dyn EventHandler>) -> Result<(), BusError>;
}

type SubscriberId = String;

trait EventHandler {
    fn on_event(&mut self, e: &CanonicalEvent);
}
```

### 9.2 Ordering guarantee

EventBus must enforce deterministic ordering at subscription delivery boundary.

---

## 10) Strategy API

### 10.1 Strategy context

```rust
struct StrategyContext {
    session_id: SessionId,
    run_id: RunId,
    now_capture_ts_nanos: i64,
    // optional references: market snapshot, positions, router state
}
```

### 10.2 Strategy interface

```rust
trait Strategy {
    fn id(&self) -> StrategyId;

    fn on_event(&mut self, e: &CanonicalEvent, ctx: &StrategyContext) -> Vec<DecisionEvent>;

    fn on_end_of_run(&mut self) -> Vec<DecisionEvent>; // optional
}
```

Strategies must output decisions; they do not execute orders directly.

---

## 11) Router API

### 11.1 Router features/state

```rust
struct RouterFeatures {
    vol_bucket: String,
    liq_bucket: String,
    trend_bucket: String,
    toxicity_bucket: String,
}

struct RegimeState {
    regime_id: String,
    confidence: Option<FixedDecimal>,
}
```

### 11.2 Router interface

```rust
trait RegimeRouter {
    fn classify(&self, f: &RouterFeatures) -> RegimeState;
    fn select_profile(&self, state: &RegimeState) -> ActiveProfile;
}

struct ActiveProfile {
    profile_id: String,
    max_order_rate: u32,
    size_multiplier: FixedDecimal,
    allow_making: bool,
    allow_taking: bool,
}
```

---

## 12) Risk API

### 12.1 Risk input

```rust
struct RiskInput {
    decision: DecisionEvent,
    profile: ActiveProfile,
    // references to current positions and market snapshot are recommended
}
```

### 12.2 Risk output

```rust
enum RiskDisposition {
    Allow { risk_decision_id: RiskDecisionId, constraints: Constraints },
    Throttle { risk_decision_id: RiskDecisionId, constraints: Constraints, reason: String },
    Reject { risk_decision_id: RiskDecisionId, reason: String },
    Halt { risk_decision_id: RiskDecisionId, scope: HaltScope, reason: String },
}

struct Constraints {
    max_qty: Option<FixedDecimal>,
    max_notional: Option<FixedDecimal>,
    min_spread_ok: Option<bool>,
}

enum HaltScope { Symbol, Venue, Global }
```

### 12.3 Risk engine trait

```rust
trait RiskEngine {
    fn evaluate(&mut self, input: &RiskInput) -> RiskDisposition;
}
```

Risk engine must emit RiskEvent to WAL for every evaluation.

---

## 13) Execution/OMS API

### 13.1 Order intent

```rust
struct OrderIntent {
    strategy_id: StrategyId,
    decision_id: DecisionId,
    risk_decision_id: RiskDecisionId,
    symbol: Symbol,
    side: Side,
    qty: FixedDecimal,
    limit_px: Option<FixedDecimal>,
    order_type: OrderType,
    tif: TimeInForce,
}
```

### 13.2 Executor trait

```rust
trait Executor {
    fn submit(&mut self, intent: &OrderIntent) -> Result<OrderId, ExecError>;
    fn cancel(&mut self, order_id: &OrderId) -> Result<(), ExecError>;
    fn poll_updates(&mut self) -> Result<Vec<CanonicalEvent>, ExecError>; // emits OrderEvent/FillEvent
}
```

Executor must write OrderEvents/Fills to WAL and emit them into the event bus.

---

## 14) Replay API

### 14.1 Replay input/output

```rust
struct ReplayInput {
    wal_paths: Vec<String>,
    config_path: String,
    code_hash: String,
    expected_trace_hash: Option<String>,
}

struct ReplayOutput {
    trace_hash: String,
    stats: serde_json::Value,
}
```

### 14.2 Replay engine trait

```rust
trait ReplayEngine {
    fn run(&mut self, input: &ReplayInput) -> Result<ReplayOutput, ReplayError>;
}
```

---

## 15) Gates API

### 15.1 Gate input

```rust
struct GateInput {
    session_manifest_path: String,
    run_manifest_path: String,
    wal_paths: Vec<String>,
    config_path: String,
    artifacts: Vec<String>,
}
```

### 15.2 Gate result

```rust
struct GateResult {
    gate_name: String,
    pass: bool,
    metrics: serde_json::Value,
    notes: Vec<String>,
}
```

### 15.3 Gate trait

```rust
trait Gate {
    fn name(&self) -> &'static str;
    fn run(&self, input: &GateInput) -> GateResult;
}
```

### 15.4 Gate suite trait

```rust
struct GateSummary {
    pass: bool,
    results: Vec<GateResult>,
    report_sha256: String,
}

trait GateSuite {
    fn run_all(&self, input: &GateInput) -> GateSummary;
}
```

---

## 16) Observability API

### 16.1 Required fields on spans/logs

- session_id, run_id, venue, symbol, strategy_id, decision_id, risk_decision_id, order_id

### 16.2 Span helpers (recommended)

```rust
trait Obs {
    fn span_market(&self, meta: &EventMeta) -> SpanHandle;
    fn span_decision(&self, decision_id: &DecisionId) -> SpanHandle;
    fn span_order(&self, order_id: &OrderId) -> SpanHandle;
}
```

(Implementation may use tracing.)

---

## 17) Error Taxonomy

Errors must be typed and actionable.

```rust
enum QuantLaxmiError {
    Connector(ConnectorError),
    Canonicalize(CanonicalizeError),
    Wal(WalError),
    Replay(ReplayError),
    Gate(GateError),
    Risk(RiskError),
    Exec(ExecError),
    Config(ConfigError),
    Invariant(InvariantError),
}
```

All errors must:

- include structured context (venue, symbol, IDs)
- be logged with correlation IDs
- emit SessionEvent health warnings where appropriate

---

## 18) Acceptance Invariants

### 18.1 Invariant: Replay parity

For any run:

```
trace_hash_original == trace_hash_replay
```

### 18.2 Invariant: Event schema stability

- WAL produced by version N must be readable by N+1 (within a backward-compat window)
- schema_version increments only under explicit governance

### 18.3 Invariant: No bypass of risk

- All orders must reference a risk_decision_id
- RiskEvent must exist for every OrderIntent

### 18.4 Invariant: Deterministic ordering

- strategy boundary ordering matches replay ordering

### 18.5 Invariant: No float parsing in canonical ingestion

- any float parsing path is an automatic G0 failure

---

## 19) Appendix: Suggested Rust Module Mapping

### quantlaxmi-events

- `fixed_decimal.rs`
- `event_meta.rs`
- `market_events.rs`
- `trade_events.rs`
- `session_events.rs`
- `canonical_serialize.rs`

### quantlaxmi-wal

- `wal_record.rs`
- `writer.rs`
- `reader.rs`
- `checksum.rs`

### quantlaxmi-gates

- `gate.rs`
- `suite.rs`
- `g0_data_truth.rs`
- `g1_replay_parity.rs`
- `g2_anti_overfit.rs`
- `g3_robustness.rs`
- `g4_deployability.rs`

### quantlaxmi-replay (optional)

- `replay_engine.rs`
- `trace_hash.rs`
- `ordering.rs`

### quantlaxmi-risk

- `policy.rs`
- `ladder.rs`
- `rules/*.rs`

### quantlaxmi-executor

- `oms.rs`
- `venue_adapters/*.rs`

---

*End of specification.*
