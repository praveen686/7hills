# BRAHMASTRA Progress Ledger

**Created**: 2026-02-07
**Phase**: 3 (Replay Engine) — COMPLETE
**Last Updated**: 2026-02-07

---

## Phase 0: Reconnaissance (READ-ONLY)

| Deliverable | Owner | Status | File |
|-------------|-------|--------|------|
| Audit Log | Agent A (Auditor) | DONE | docs/AUDIT_LOG.md |
| Progress Ledger | Agent A (Auditor) | DONE | docs/PROGRESS_LEDGER.md |
| Tier-1 Strategy Contracts V1 | Agent B (Strategy Contracts) | DONE | docs/STRATEGY_CONTRACTS_TIER1_V1.md |
| Events/WAL/Replay Audit | Agent C (Events Architect) | DONE | (embedded in ROADMAP) |
| Frontend UI Audit | Agent D (UI Temple) | DONE | (embedded in ROADMAP) |
| Gates & Tests Audit | Agent E (Gates Engineer) | DONE | (embedded in ROADMAP) |
| Tier-1 Worldclass Roadmap | All Agents | DONE | docs/ROADMAP_TIER1_WORLDCLASS.md |

---

## Phase 1: Safety Foundation — COMPLETE (2026-02-07)

| Deliverable | Tests | Status | File |
|-------------|-------|--------|------|
| Risk Gate Enforcement | **84 tests** | DONE | tests/risk/test_risk_gates.py |
| Determinism Test Suite (3x replay) | **29 tests** | DONE | tests/determinism/test_determinism.py |
| Live Causality Enforcement | **23 tests** | DONE | tests/causality/test_causality.py |
| India Cost Model Validation | **38 tests** | DONE | tests/costs/test_india_costs.py |
| **TOTAL** | **174 tests** | **ALL PASS** | 4.38s runtime |

### Coverage Summary
- **Risk Gates**: VPIN gate (12), Portfolio DD (11), Strategy DD (9), Concentration (18), Gate ordering (4), Circuit breaker (6), Portfolio state (9), Limits immutability (3), Multi-target (3), Reduce size (3), RiskCheckResult (2), Flat signals (4)
- **Determinism**: Entropy (4), MI (3), Rolling features (3), Regime detector (3), VPIN ticks (1), Tick entropy (1), Hawkes (1), Bar microstructure (1), Cost model (1), Risk manager (3), Portfolio state (3), Signal protocol (2), FP tolerance (3)
- **Causality**: Entropy (3), MI (2), Rolling (4), Regime (3), Bar VPIN (1), Tick VPIN (1), Tick entropy (1), Signal protocol (3), Signal state isolation (2), T+1 lag (3)
- **Costs**: Basic properties (7), Immutability (3), Validation (4), Funding (5), India FnO (5), Index point costs (5), Edge cases (5), Scenarios (4)

---

## Phase 2: Event Infrastructure — COMPLETE (2026-02-07)

| Deliverable | Tests | Status | File(s) |
|-------------|-------|--------|---------|
| Canonical Event Schemas (qlx/events/) | — | DONE | types.py, envelope.py, payloads.py, serde.py, hashing.py |
| WAL Persistence Layer | — | DONE | brahmastra/engine/event_log.py, wal_tap.py, journals.py, session_manifest.py |
| Orchestrator Event Wiring | — | DONE | brahmastra/orchestrator.py (rewritten with full event emission) |
| Serde Roundtrip + Canonical JSON | **56 tests** | DONE | tests/events/test_event_envelope_serde.py |
| Daily Rotation + Close | **15 tests** | DONE | tests/events/test_event_log_rotation.py |
| Seq Monotonicity + Recovery | **13 tests** | DONE | tests/events/test_seq_monotonicity.py |
| Atomic Append + Crash Recovery | **14 tests** | DONE | tests/events/test_atomic_append_recovery.py |
| WAL Tap Backpressure | **10 tests** | DONE | tests/events/test_wal_tap_backpressure.py |
| Hash Chain (Optional) | **8 tests** | DONE | tests/events/test_hash_chain_optional.py |
| **TOTAL** | **116 tests** | **ALL PASS** | 0.93s runtime |

### Phase 2 Coverage Summary
- **Serde**: Roundtrip stability (8), Canonical JSON (5), Sanitize NaN/Inf (7), Envelope immutability (3), Factory (4), All 8 payload types (8), Payload immutability (3), Deserialization errors (4), EventType coverage (10)
- **Rotation**: Daily file creation (4), File content validity (3), Event count tracking (3), Directory creation (1), Close/flush behavior (3)
- **Seq**: Monotonic increment (5), Recovery from existing files (3), Persisted seq matches returned (2), Allocator (3)
- **Recovery**: Truncated last line (3), read_event_log robustness (3), Append after recovery (2), SessionManifest atomic write (6)
- **WAL Tap**: Basic stats (2), Overflow error class (2), Persist loop (2), Shutdown drain (2), Idempotent stop (1)
- **Hash Chain**: Genesis stability (3), chain_hash determinism (5), verify_chain tamper detect (8), compute_chain (5), EventLog sidecar integration (5)

### Cumulative Test Counts
| Phase | Tests | Time |
|-------|-------|------|
| Phase 1 — Safety Foundation | 174 | 4.38s |
| Phase 2 — Event Infrastructure | 116 | 0.93s |
| Phase 3 — Replay Engine | 95 | 0.16s |
| **TOTAL** | **385** | **5.16s** |

---

## Phase 3: Replay Engine — COMPLETE (2026-02-07)

| Deliverable | Tests | Status | File(s) |
|-------------|-------|--------|---------|
| WAL Reader (JSONL stream + seq + hash chain verification) | — | DONE | brahmastra/replay/reader.py |
| Event Stream Comparator (field-level diff with FP tolerance) | — | DONE | brahmastra/replay/comparator.py |
| Replay Engine (deterministic runner, N-way parity, artifacts) | — | DONE | brahmastra/replay/engine.py |
| CLI replay subcommand (--date, --range, --times, --verify-hashes) | — | DONE | brahmastra/__main__.py (lines 76-134, 250-269) |
| Replay Parity Tests (3x for S5, S1, S4, S7) | **44 tests** | DONE | tests/replay/test_replay_parity_tier1.py |
| Replay Idempotence Tests (serde, write-read, comparator, hash chain) | **20 tests** | DONE | tests/replay/test_replay_idempotence.py |
| Replay Hash Chain Tests (basics, tamper, writer, reader, multi-day) | **31 tests** | DONE | tests/replay/test_replay_hashchain.py |
| **TOTAL** | **95 tests** | **ALL PASS** | 0.16s runtime |

### Bug Fix Discovered During Phase 3 Testing
- **File**: `brahmastra/engine/event_log.py` line 155-156
- **Issue**: `EventLogWriter._ensure_file()` did NOT reset `_last_hash` to GENESIS on daily rotation. When hash chain was enabled and the writer rotated to a new day, the second day's hash chain would start from the last hash of the previous day instead of GENESIS, causing per-day hash chain verification to fail.
- **Fix**: Added `self._last_hash = GENESIS` in `_ensure_file()` after opening the hash sidecar for a new day.
- **Impact**: Multi-day hash chain verification now correctly treats each day as an independent chain starting from GENESIS.

### Phase 3 Coverage Summary
- **WalReader** (14 tests): read_date, read_range, read_all, available_dates, corrupt line skip, seq validation (strict/lenient), filtering (type/strategy/symbol), stats
- **Comparator** (20 tests): identical streams, conviction/gate/equity/direction/strategy_id diffs, missing/extra events, event type counts, FP tolerance (pass/fail), NaN/Inf equality, summary format, to_dict serializable, selective event types, order/risk_alert comparison
- **3x Parity** (10 tests): S5 Hawkes, S1 VRP, S4 IV MR, S7 Regime, multi-day, mixed strategies, empty stream, field-exact-match (signal, gate, snapshot)
- **Serde Idempotence** (6 tests): single/all triple roundtrip, NaN/Inf/nested/empty payload
- **Write-Read Idempotence** (5 tests): single/double/triple cycle, EventLogWriter cycle, emit cycle
- **Comparator Idempotence** (5 tests): self-comparison, deterministic, commutative detection, to_dict stable, no mutation
- **Hash Chain Idempotence** (4 tests): write-read stable, 3x compute, writer chain stable, reader verification
- **Hash Chain Basics** (10 tests): genesis SHA-256, deterministic, depends-on-prev, depends-on-line, compute length, first-uses-genesis, chaining, verify valid/empty/single
- **Tamper Detection** (9 tests): first/middle/last line, tampered hash, deleted/inserted/reordered lines, serialized event tamper, single-byte tamper
- **Writer Hash Chain** (5 tests): creates sidecar, chain verifiable, no sidecar when disabled, matches recomputed, tamper detection after write
- **Reader Hash Verification** (4 tests): valid hashes, tampered logged, missing sidecar graceful, strict raises
- **Multi-Day Hash Chain** (3 tests): independent chains per day, writer rotation hashes, range read with verification

---

## Phase 4: Frontend — Why Panel (PLANNED)

| Deliverable | Owner | Status | Depends On |
|-------------|-------|--------|------------|
| Why Panel component | Agent D | PENDING | Signal Journal |
| Replay controls (date picker, play/pause) | Agent D | PENDING | Replay engine (Phase 3) |
| Trade detail modal | Agent D | PENDING | ExecutionJournal |
| Opportunity diagnostics page | Agent D | PENDING | Trade analytics API |
| ARS heatmap component | Agent D | PENDING | Research API |

---

## Phase 5: Production Hardening (PLANNED)

| Deliverable | Owner | Status | Depends On |
|-------------|-------|--------|------------|
| Missingness handling + tests | Agent E | PENDING | Phase 2 |
| End-to-end backtest determinism | Agent E | PENDING | Phase 3 |
| Regime gate tests | Agent E | PENDING | Phase 1 |
| Position state machine tests | Agent E | PENDING | Phase 1 |
| CI/CD pipeline with determinism gate | Agent E | PENDING | All tests |

---

## Existing Assets (Pre-Phase 0)

| Asset | Lines | Status |
|-------|-------|--------|
| Python codebase | ~8,800 | 57 files, syntax-clean |
| Frontend (Next.js) | ~4,000 | 27 files, TypeScript compiles |
| Research backtests | 11 strategies | Full scorecard with honest results |
| DuckDB data store | 11 GB | 27 categories, 316 days |
| FastAPI backend | 15 routes | All returning real data |
| SANOS pricing | 60+ tests | Math validated, no determinism guarantee |
| TimeGuard | 15 tests | Offline causality clean |

---

## Tier-1 Strategy Performance (Baseline)

| Strategy | Sharpe | Return | Trades | MaxDD | Status |
|----------|--------|--------|--------|-------|--------|
| S5 Hawkes | 4.29 | +3.62% | 12 | 0.56% | Contract written |
| S1 VRP Options | 5.59 RoR | +11.57% | 9 | 0.88% | Contract written |
| S4 IV MR | 3.07 | +8.91% | 7 | 2.50% | Contract written |
| S7 Regime | 2.37 | +4.10% | 2 | 1.30% | Contract written |
