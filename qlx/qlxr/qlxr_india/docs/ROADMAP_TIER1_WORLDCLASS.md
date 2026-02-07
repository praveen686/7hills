# BRAHMASTRA Tier-1 Worldclass Roadmap

**Date**: 2026-02-07
**Source**: Phase 0 Reconnaissance (Agents B, C, D, E)
**Status**: Ready for user approval

---

## Vision

Transform BRAHMASTRA from a backtesting research platform into a **production-grade, audit-ready, operator-facing trading system** with deterministic replay, full decision traceability, and real-time opportunity diagnostics.

---

## Current State Assessment

### What Works Well
- 4 Tier-1 strategies with honest Sharpe: S5 (4.29), S1 (5.59 RoR), S4 (3.07), S7 (2.37)
- SANOS pricing: 60+ math tests, validated calibration
- TimeGuard: causal offline features, T+1 lag
- Atomic JSON state: crash-safe portfolio persistence
- Frontend: 6 pages, WebSocket, React Query, dark theme
- DuckDB: 11 GB, 316 days, 27 data categories
- FastAPI: 15 routes returning real data

### What's Missing (Critical)

| Gap | Severity | Agent Source |
|-----|----------|-------------|
| Risk gates: 0% tested | CRITICAL | Agent E |
| No persistent event log | CRITICAL | Agent C |
| No execution audit trail | CRITICAL | Agent C |
| No deterministic replay | CRITICAL | Agent C |
| Determinism: <5% verified | CRITICAL | Agent E |
| No live causality tests | HIGH | Agent E |
| No Why Panel (signal causality) | HIGH | Agent D |
| No replay UI controls | HIGH | Agent D |
| No opportunity diagnostics | HIGH | Agent D |
| India cost model not validated | HIGH | Agent E |
| SANOS LP determinism unverified | MEDIUM | Agent E |
| No missingness handling | MEDIUM | Agent E |

---

## Phased Roadmap

### Phase 1: Safety Foundation — COMPLETE (2026-02-07)

**Goal**: Ensure capital protection mechanisms work before any live trading.
**Status**: DONE — 174 tests, 0 failures, 4.38s

| Task | Tests | Status |
|------|-------|--------|
| 1.1 Risk gate test suite | 84 tests (VPIN, DD, concentration, ordering, breakers) | DONE |
| 1.2 Determinism test suite | 29 tests (3x replay parity, FP tolerance) | DONE |
| 1.3 India cost validation | 38 tests (STT, per-leg index pts, funding, scenarios) | DONE |
| 1.4 Live causality tests | 23 tests (poison-future method, T+1 lag) | DONE |

**Exit Criteria**: MET — All risk gates 100% tested, determinism verified, costs validated, causality proven.

---

### Phase 2: Event Infrastructure — COMPLETE (2026-02-07)

**Goal**: Build persistent event logging for deterministic replay.
**Status**: DONE — 116 tests, 0 failures, 0.93s

| Task | Description | Status |
|------|-------------|--------|
| 2.1 EventLog persistence | Append-only JSONL, daily rotation, monotonic seq, crash recovery, optional hash chain | DONE |
| 2.2 ExecutionJournal | Order + fill lifecycle, typed payloads, linked by order_id | DONE |
| 2.3 SignalJournal | Pre-gate signal + gate decision events, linked by signal_seq | DONE |
| 2.4 SessionManifest | Atomic write via tempfile+rename, start/finalize lifecycle | DONE |
| 2.5 WAL Tap | Async bus subscriber, bounded queue, WalOverflowError (NO SILENT DROP) | DONE |
| 2.6 Canonical Serde | Sorted keys, NaN/Inf→null, roundtrip-stable, SHA-256 hash chain | DONE |
| 2.7 Orchestrator Wiring | Event emission at every decision boundary (signal, gate, risk, order, snapshot) | DONE |

**Files Created**: `qlx/events/` (5 files), `brahmastra/engine/` (4 files), `tests/events/` (6 files)

**Exit Criteria**: MET — All events persisted to JSONL. Monotonic seq, crash recovery, no silent drop, tamper-evident hash chain.

---

### Phase 3: Replay Engine — COMPLETE (2026-02-07)

**Goal**: Prove deterministic replay: same WAL + config = identical decisions.
**Status**: DONE — 95 tests, 0 failures, 0.16s

| Task | Tests | Status |
|------|-------|--------|
| 3.1 WAL Reader (JSONL stream, seq validation, hash chain verification) | 14 (WalReader) + 10 (hash basics) | DONE |
| 3.2 Replay Engine (deterministic runner, N-way parity, comparator, artifacts) | 20 (comparator) + engine impl | DONE |
| 3.3 Replay Parity Tests (3x for S5, S1, S4, S7) | 10 (parity) + 20 (idempotence) + 31 (hashchain) | DONE |
| 3.4 Cross-platform parity | Deferred (FP tolerance built in, rtol=1e-10) | DEFERRED |

**Files Created**: `brahmastra/replay/` (3 files: reader.py, comparator.py, engine.py), `tests/replay/` (3 test files)
**Bug Fixed**: `brahmastra/engine/event_log.py` — hash chain not reset to GENESIS on daily rotation

**Exit Criteria**: MET — `replay(WAL, config) == replay(WAL, config)` verified 3x for all 4 Tier-1 strategies, plus mixed and multi-day scenarios.

---

### Phase 4: Frontend — Why Panel

**Goal**: Operator can see WHY any signal was generated or trade was entered/exited.
**Owner**: Agent D (UI Temple Architect)
**Dependencies**: Phase 2 (SignalJournal, ExecutionJournal must exist)

| Task | Description |
|------|-------------|
| 4.1 Why Panel component | Side-panel showing signal causality chain |
| 4.2 Signal context API | `GET /api/signals/{id}/context` — features, regime, IV, components |
| 4.3 Gate decision API | `GET /api/gates/decisions/{signal_id}` — which gates passed/blocked |
| 4.4 Trade detail modal | Full entry/exit decision trace with market context |
| 4.5 Missing types | TradeDecisionLog, SignalOutcome, MarketContext TypeScript interfaces |

**Why Panel Fields** (per Tier-1 strategy):

| Strategy | Key Fields |
|----------|------------|
| S5 Hawkes | gex_regime, 5 component scores, ema_state, consecutive_count |
| S1 VRP | composite, sig_percentile, skew_premium, left_tail, atm_iv |
| S4 IV MR | atm_iv, iv_percentile, entry_threshold |
| S7 Regime | entropy, MI, VPIN, classified_regime, sub_strategy, z_score |

**Exit Criteria**: For every signal and trade, operator can see complete decision chain from data -> features -> decision -> risk check -> execution.

---

### Phase 5: Frontend — Replay Controls

**Goal**: Operator can time-travel to any historical date and inspect full market + portfolio state.
**Owner**: Agent D
**Dependencies**: Phase 3 (Replay engine must work)

| Task | Description |
|------|-------------|
| 5.1 Replay mode toggle | Sidebar toggle: "Live" vs "Replay" mode |
| 5.2 Date/time picker | Calendar + time selector for replay target |
| 5.3 Play/pause/step controls | Step through trades minute-by-minute |
| 5.4 Historical snapshot API | `GET /api/replay/snapshot/{timestamp}` — portfolio + market at time |
| 5.5 Trade timeline | Visual timeline of all trades with entry/exit markers |
| 5.6 Market context overlay | VIX, regime, IV at trade entry/exit |

**Exit Criteria**: Operator can pick any date in Oct 2024 - Feb 2026, see portfolio state, step through trades, and inspect market context at each decision point.

---

### Phase 6: Opportunity Diagnostics

**Goal**: Per-trade analysis showing missed upside, adverse moves, and efficiency.
**Owner**: Agent D (frontend) + Agent B (analytics API)
**Dependencies**: Phase 4 (decision trace must exist)

| Task | Description |
|------|-------------|
| 6.1 Trade analytics API | `GET /api/trades/{id}/analytics` — MFM, MDA, efficiency |
| 6.2 Opportunity card | Entry signal quality, max favorable/adverse moves, duration |
| 6.3 Missed opportunity tracker | Signals not taken, potential P&L if taken |
| 6.4 Signal-to-trade funnel | Conversion rate: signals -> admitted -> executed -> profitable |
| 6.5 ARS heatmap | 2D return surface: lookback x forward return, color-coded Sharpe |

**Diagnostic Metrics**:
- **MFM** (Max Favorable Move): Highest unrealized P&L during trade
- **MDA** (Max Adverse Drawdown): Deepest unrealized loss during trade
- **Efficiency**: (Actual P&L) / MFM — how much upside was captured
- **ARS** (Activation Readiness Score): How close was a strategy to activating

**Exit Criteria**: For every Tier-1 trade, operator sees MFM/MDA/efficiency. ARS heatmap shows regime-conditional return surfaces for all 4 strategies.

---

### Phase 7: Production Hardening

**Goal**: CI/CD gates, missingness handling, end-to-end determinism.
**Owner**: Agent E
**Dependencies**: Phases 1-3 (all test infra must be in place)

| Task | Description |
|------|-------------|
| 7.1 Missingness handling tests | 20-30 tests: stale data, market halts, missing strikes |
| 7.2 End-to-end backtest determinism | Full pipeline 3x replay with equity curve matching |
| 7.3 Regime gate tests | 10-15 tests: S7 blocking during choppy markets |
| 7.4 Position state machine tests | 15-20 tests: open -> hold -> exit -> archive |
| 7.5 CI/CD pipeline | Determinism gate: block deploy if replay parity fails |
| 7.6 SANOS LP determinism | Pin solver tolerance, verify same data -> same density |

**Exit Criteria**: CI/CD enforces determinism. All edge cases handled gracefully. System degrades safely when data is missing or stale.

---

## Dependency Graph

```
Phase 1 (Safety Tests)
   |
   v
Phase 2 (Event Infrastructure)
   |
   v
Phase 3 (Replay Engine)  -----> Phase 7 (Production Hardening)
   |
   v
Phase 4 (Why Panel)
   |
   v
Phase 5 (Replay UI)
   |
   v
Phase 6 (Opportunity Diagnostics)
```

Phases 1 and 2 can partially overlap. Phase 4 can start as soon as Phase 2 delivers SignalJournal.

---

## Per-Strategy Readiness Checklist

| Requirement | S5 Hawkes | S1 VRP | S4 IV MR | S7 Regime |
|-------------|-----------|--------|----------|-----------|
| Strategy Contract V1 | DONE | DONE | DONE | DONE |
| Data Contract verified | Pending | Pending | Pending | Pending |
| Feature determinism test | Pending | Pending | Pending | Pending |
| Decision replay parity | **DONE** (3x) | **DONE** (3x) | **DONE** (3x) | **DONE** (3x) |
| Risk gate integration test | Pending | Pending | Pending | Pending |
| Execution journal | Pending | Pending | Pending | Pending |
| Why Panel fields defined | DONE | DONE | DONE | DONE |
| Replay event log schema | **DONE** | **DONE** | **DONE** | **DONE** |
| Failure mode handling | Documented | Documented | Documented | Documented |

---

## New Backend Endpoints Required

| Endpoint | Phase | Description |
|----------|-------|-------------|
| `GET /api/signals/{id}/context` | 4 | Signal generation context (features, regime) |
| `GET /api/gates/decisions/{signal_id}` | 4 | Gate decisions for a signal |
| `GET /api/trades/{id}/decisions` | 4 | Trade entry/exit decision log |
| `GET /api/trades/{id}/analytics` | 6 | MFM, MDA, efficiency |
| `GET /api/replay/snapshots` | 5 | Available snapshot dates |
| `GET /api/replay/snapshot/{timestamp}` | 5 | Portfolio + market at time |
| `GET /api/replay/market-context/{timestamp}` | 5 | Market data at time |
| `GET /api/portfolio/missed-opportunities` | 6 | Signals not executed |
| `GET /api/research/ars-surface` | 6 | Return surface data |

---

## New Frontend Components Required

| Component | Phase | Description |
|-----------|-------|-------------|
| WhyPanel.tsx | 4 | Signal causality side-panel |
| DecisionLog.tsx | 4 | Timeline of gate decisions |
| TradeDetailModal.tsx | 4 | Deep dive into one trade |
| ReplayControls.tsx | 5 | Date picker, play/pause/step |
| ReplayTimeline.tsx | 5 | Visual timeline of trades |
| OpportunityCard.tsx | 6 | Per-trade MFM/MDA/efficiency |
| ARSHeatmap.tsx | 6 | 2D return surface heatmap |
| TradeOutcomeChart.tsx | 6 | Favorable/adverse move viz |

---

## New TypeScript Types Required

```typescript
TradeDecisionLog      // trade_id, signal_id, admission_checks, risk_checks, exit_reason
SignalOutcome         // signal_id, executed, outcome, unrealized_pnl
TradeDiagnostic       // MFM, MDA, efficiency_ratio, duration
ReplaySnapshot        // timestamp, portfolio, positions, market_data, active_signals
MarketContext         // timestamp, vix, regime, spot_prices
ARSSurfacePoint       // lookback_bucket, fwd_return_bucket, annualized_return, sharpe
```

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| LP solver non-determinism | Replay fails | Pin solver + tolerance in config |
| Floating-point drift | Cross-machine replay differs | Use rtol=1e-6 in assertions |
| EventLog I/O bottleneck | Live engine slows down | Async writes, buffered flush |
| JSONL file corruption | Replay impossible for session | Checksum per event, daily rotation |
| Frontend state explosion | Browser memory overflow | Virtual scrolling, paginated APIs |

---

**Phase 0**: COMPLETE (reconnaissance, contracts, roadmap)
**Phase 1**: COMPLETE (174 safety tests — risk gates, determinism, causality, costs)
**Phase 2**: COMPLETE (116 event infrastructure tests — serde, rotation, seq, recovery, WAL, hash chain)
**Phase 3**: COMPLETE (95 replay engine tests — parity, idempotence, hash chain verification)

**Cumulative**: 385 tests, 0 failures, 5.16s runtime

**Next step**: Phase 4 (Frontend — Why Panel).
