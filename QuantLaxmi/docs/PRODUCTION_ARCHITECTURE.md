# QuantLaxmi Production Architecture Document

**Version**: 1.0.0
**Date**: 2026-01-30
**Classification**: Internal - Production Engineering

---

## Executive Summary

QuantLaxmi is a **production-grade, multi-market algorithmic trading platform** with complete isolation between India FNO (via Zerodha) and Crypto (via Binance). The architecture emphasizes:

- **Deterministic Execution**: Fixed-point mantissa+exponent arithmetic eliminates floating-point drift
- **Complete Audit Trail**: 7-level gate validation pipeline (G0-G7) with SHA-256 decision hashing
- **Connection Resilience**: Auto-reconnect WebSockets with exponential backoff and gap tracking
- **Market Isolation**: Zero cross-contamination between India and Crypto codepaths

**Production Readiness Assessment**:
| Component | Status | Rating |
|-----------|--------|--------|
| Crypto Capture | Production Ready | A |
| India Capture | Beta (needs auth hardening) | B+ |
| Backtest Engine | Production Ready | A |
| Live Execution | Requires Review | B- |
| Risk Management | Production Ready | A |

---

## 1. Component Architecture

### 1.1 Crate Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  apps/quantlaxmi-crypto    │    apps/quantlaxmi-india               │
│  (Crypto Entry Point)      │    (India Entry Point)                 │
└──────────────┬─────────────┴──────────────┬─────────────────────────┘
               │                             │
               ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RUNNER LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  quantlaxmi-runner-crypto  │  quantlaxmi-runner-india               │
│  - Binance perp capture    │  - Zerodha session capture             │
│  - SBE depth capture       │  - KiteSim backtest                    │
│  - Funding capture         │  - Multi-expiry discovery              │
│  - Tournament engine       │  - Equity curve tracking               │
└──────────────┬─────────────┴──────────────┬─────────────────────────┘
               │                             │
               ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CONNECTOR LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  quantlaxmi-connectors-    │  quantlaxmi-connectors-                │
│  binance                   │  zerodha                               │
│  - SBE binary decoder      │  - Kite WebSocket (184-byte)           │
│  - WebSocket streaming     │  - Auto-discovery (ATM±20)             │
│  - REST snapshot fetch     │  - TOTP authentication                 │
│  ✓ NO Zerodha imports      │  ✓ NO Binance imports                  │
└──────────────┬─────────────┴──────────────┬─────────────────────────┘
               │                             │
               ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SHARED INFRASTRUCTURE                           │
├─────────────────────────────────────────────────────────────────────┤
│  quantlaxmi-runner-common                                            │
│  - Circuit breakers, Rate limiters                                   │
│  - Session manifest, Web server                                      │
│  - TUI components, Artifact management                               │
└──────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                            │
├────────────────┬────────────────┬───────────────┬───────────────────┤
│ quantlaxmi-    │ quantlaxmi-    │ quantlaxmi-   │ quantlaxmi-       │
│ gates          │ executor       │ risk          │ strategy          │
│ - G0-G7 gates  │ - Simulated    │ - Pre-trade   │ - Strategy SDK    │
│ - Admission    │ - Live (Kite)  │ - Post-trade  │ - 3 built-in      │
│ - Capital alloc│ - Risk envelope│ - Circuit brk │ - Canonical hash  │
├────────────────┼────────────────┼───────────────┼───────────────────┤
│ quantlaxmi-    │ quantlaxmi-    │ quantlaxmi-   │ quantlaxmi-       │
│ options        │ eval           │ data          │ core              │
│ - Pricing      │ - Truth report │ - Bar aggr    │ - EventBus        │
│ - Greeks       │ - Strategy agg │ - L2 book     │ - Portfolio       │
│ - Margin       │               │ - VPIN        │ - Strategies      │
└────────────────┴────────────────┴───────────────┴───────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FOUNDATION LAYER                               │
├────────────────┬────────────────┬───────────────────────────────────┤
│ quantlaxmi-    │ quantlaxmi-    │ quantlaxmi-sbe                    │
│ models         │ wal            │ - SBE binary decoder              │
│ - DecisionEvent│ - JSONL writer │ - Template 10000 (trades)         │
│ - QuoteEvent   │ - Manifest     │ - Template 10003 (depth)          │
│ - SignalFrame  │ - Checksums    │                                   │
│ - Mantissa math│               │                                    │
│ ✓ NO DEPS      │               │                                    │
└────────────────┴────────────────┴───────────────────────────────────┘
```

### 1.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MARKET DATA INGESTION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Binance    │    │   Zerodha    │    │   Replay     │           │
│  │  WebSocket   │    │  WebSocket   │    │   (WAL)      │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │                    │
│         ▼                   ▼                   ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ ResilientWs  │    │ Binary Parse │    │  WAL Reader  │           │
│  │ (auto-recon) │    │ (184-byte)   │    │  (JSONL)     │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │                    │
│         └───────────────────┼───────────────────┘                    │
│                             ▼                                        │
│                    ┌────────────────┐                                │
│                    │   EventBus     │                                │
│                    │  (broadcast)   │                                │
│                    └────────┬───────┘                                │
└─────────────────────────────┼────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SIGNAL PROCESSING                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  SignalFrame │───▶│  Admission   │───▶│  Strategy    │           │
│  │  (L1 + bits) │    │  Controller  │    │  on_event()  │           │
│  └──────────────┘    └──────────────┘    └──────┬───────┘           │
│                                                  │                   │
│                                                  ▼                   │
│                                         ┌──────────────┐            │
│                                         │DecisionOutput│            │
│                                         │ + OrderIntent│            │
│                                         └──────┬───────┘            │
└────────────────────────────────────────────────┼────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EXECUTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ Order Intent │───▶│   Risk       │───▶│  Exchange    │           │
│  │   Record     │    │  Envelope    │    │  (Sim/Live)  │           │
│  └──────────────┘    └──────────────┘    └──────┬───────┘           │
│                                                  │                   │
│  ┌──────────────────────────────────────────────┼───────────────┐   │
│  │                        WAL PERSISTENCE       ▼               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │
│  │  │decisions │  │ orders   │  │  fills   │  │positions │     │   │
│  │  │ .jsonl   │  │ .jsonl   │  │ .jsonl   │  │ .jsonl   │     │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Gate Validation Pipeline (G0-G7)

### 2.1 Gate Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GATE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  G0 ────▶ G1 ────▶ G2 ────▶ G3 ────▶ G4 ────▶ G5 ────▶ G6 ────▶ G7 │
│  Data    Replay   Backtest Robust  Deploy  Order   Fill   Position │
│  Truth   Parity   Correct  ness    ability Intent  Deter  Deter    │
│                                                                      │
│  ✓ IMPL  ✓ IMPL   SCAFFOLD SCAFFOLD ✓ IMPL ✓ IMPL  ✓ IMPL ✓ IMPL   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Gate Specifications

| Gate | Purpose | Key Checks | Status |
|------|---------|------------|--------|
| **G0** | Data Truth | Manifest exists, SHA256 match, monotonic timestamps | Implemented |
| **G1** | Replay Parity | Decision trace hash match, excluded fields immunity | Implemented |
| **G2** | Backtest Correctness | No lookahead, fill realism, transaction costs | Scaffold |
| **G3** | Robustness | Connection loss, data gaps, extreme prices | Scaffold |
| **G4** | Deployability | Metrics, tracing, config snapshot, graceful shutdown | Implemented |
| **G5** | Order Intent | Monotonic seq, permission match, payload match | Implemented |
| **G6** | Execution Fill | Fill type, side, 12-field payload comparison | Implemented |
| **G7** | Position | 14-way mismatch taxonomy, cumulative state | Implemented |

### 2.3 Determinism Hash Specification (v2)

```
Domain Separator: b"quantlaxmi:decision_trace:v2\0"
Record Separator: 0x1E (ASCII RS)

Encoding Order (FROZEN):
1. ts_ns (i64 LE)
2. decision_id (16 bytes UUID)
3. strategy_id (u32 len + UTF-8)
4. symbol (u32 len + UTF-8)
5. decision_type (u32 len + UTF-8)
6. direction (i8)
7. target_qty_mantissa (i64 LE)
8. qty_exponent (i8)
9. reference_price_mantissa (i64 LE)
10. price_exponent (i8)
11. market_snapshot (all fields in order)
12. correlation_context (all fields)

EXCLUDED Fields (non-deterministic):
- confidence_mantissa
- metadata (JSON)
- spread_bps_mantissa
```

---

## 3. India FNO System

### 3.1 Capture Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INDIA FNO CAPTURE SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AUTO-DISCOVERY                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │  Spot Price  │─▶│  ATM Strike  │─▶│  Universe    │       │    │
│  │  │   (REST)     │  │  Calculation │  │  Manifest    │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  │                                                              │    │
│  │  Expiry Policy: T1 (nearest) + T2 (next) + T3 (monthly)     │    │
│  │  Strike Band: ATM ± 20 = 41 strikes × 2 types = 82/expiry   │    │
│  │  Total Instruments: ~250 per underlying                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    AUTHENTICATION                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │ Python       │─▶│  TOTP        │─▶│  Access      │       │    │
│  │  │ Sidecar      │  │  Generation  │  │  Token       │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  │  ⚠️ CRITICAL: Hardcoded path, no timeout, no refresh        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    WEBSOCKET STREAMING                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │  Full Mode   │─▶│  184-byte    │─▶│  5-Level     │       │    │
│  │  │  Subscribe   │  │  Binary Parse│  │  L2 Depth    │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  │                                                              │    │
│  │  Reconnect: Exponential backoff (1s → 60s max)              │    │
│  │  Heartbeat: 30s timeout triggers reconnect                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 KiteSim Execution Simulator

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KITESIM SIMULATOR                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Configuration:                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ latency_ms: 150        │ Order eligibility delay              │  │
│  │ slippage_bps: 0        │ Market order slippage                │  │
│  │ adverse_bps: 0         │ Adverse selection penalty            │  │
│  │ stale_quote_ms: 10000  │ Reject if quote older than this      │  │
│  │ timeout_ms: 5000       │ Atomic execution timeout             │  │
│  │ hedge_on_failure: true │ Rollback neutralization              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Execution Modes:                                                    │
│  ┌──────────────┐  ┌──────────────┐                                 │
│  │   L1Quote    │  │   L2Book     │                                 │
│  │  (Best B/A)  │  │ (Full Depth) │                                 │
│  └──────────────┘  └──────────────┘                                 │
│                                                                      │
│  Multi-Leg Semantics:                                               │
│  1. Place all legs sequentially                                     │
│  2. Poll for fill confirmation (15s timeout)                        │
│  3. On any leg failure → CANCEL all previous legs                   │
│  4. Atomic: all-or-nothing                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 India Fee Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ZERODHA FNO FEES (INR)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Component              │ Rate              │ Applied To            │
│  ──────────────────────────────────────────────────────────────────│
│  Brokerage              │ min(0.03%, ₹20)   │ Per order            │
│  Exchange (NSE)         │ 0.00325%          │ Turnover             │
│  Clearing (NSCCL)       │ 0.00075%          │ Turnover             │
│  GST                    │ 18%               │ Brokerage + Exchange │
│  STT (Sell only)        │ 0.05%             │ Sell premium         │
│  Stamp (Buy only)       │ 0.002%            │ Buy premium          │
│  SEBI                   │ 0.00005%          │ Turnover             │
│                                                                      │
│  Example: Buy 1 lot NIFTY 25300CE @ ₹150                            │
│  ─────────────────────────────────────────                          │
│  Premium: 65 × ₹150 = ₹9,750                                        │
│  Brokerage: ₹20 (flat)                                              │
│  Exchange: ₹0.32                                                    │
│  GST: ₹3.66                                                         │
│  Stamp: ₹0.20                                                       │
│  Total Fees: ₹24.18                                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Crypto Trading System

### 4.1 Capture Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CRYPTO CAPTURE SYSTEM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    RESILIENT WEBSOCKET                       │    │
│  │                                                              │    │
│  │  State Machine:                                              │    │
│  │  ┌──────────────┐    ┌──────────────┐                       │    │
│  │  │  Connected   │◀──▶│ Disconnected │                       │    │
│  │  │  (read/write)│    │  (backoff)   │                       │    │
│  │  └──────────────┘    └──────────────┘                       │    │
│  │                                                              │    │
│  │  Config:                                                     │    │
│  │  - initial_backoff: 1s                                       │    │
│  │  - max_backoff: 30s                                          │    │
│  │  - liveness_timeout: 30s                                     │    │
│  │  - max_reconnect_attempts: 100                               │    │
│  │  - ping_interval: 30s                                        │    │
│  │                                                              │    │
│  │  Gap Tracking:                                               │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │ { disconnect_ts, reconnect_ts, gap_ms, reason, atts } │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    CAPTURE STREAMS                           │    │
│  │                                                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │ spot_quotes  │  │ perp_depth   │  │   funding    │       │    │
│  │  │ @bookTicker  │  │ @depth@100ms │  │ @markPrice   │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  │       ↓                  ↓                  ↓                │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │  QuoteEvent  │  │  DepthEvent  │  │ FundingEvent │       │    │
│  │  │   JSONL      │  │   JSONL      │  │   JSONL      │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 SBE Depth Bootstrap Protocol

```
┌─────────────────────────────────────────────────────────────────────┐
│                  BINANCE SBE DEPTH SYNCHRONIZATION                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Connect to SBE WebSocket                                   │
│          └── Start buffering diffs immediately                      │
│                                                                      │
│  Step 2: Wait for subscription confirmation (2s window)            │
│          └── Buffer all incoming diffs                              │
│                                                                      │
│  Step 3: Fetch REST snapshot                                        │
│          └── GET /api/v3/depth?symbol={}&limit=1000                 │
│                                                                      │
│  Step 4: Find sync point in buffer                                  │
│          └── First diff where: U ≤ lastUpdateId+1 ≤ u               │
│                                                                      │
│  Step 5: Write snapshot (is_snapshot: true)                         │
│                                                                      │
│  Step 6: Apply buffered diffs (after sync point)                    │
│                                                                      │
│  Step 7: Continue with live stream                                  │
│          └── Check: U == prev_u + 1 (sequence validation)           │
│                                                                      │
│  On Reconnect:                                                       │
│  ─────────────                                                       │
│  1. Detect via ws.total_reconnects() counter change                 │
│  2. Re-fetch REST snapshot                                          │
│  3. Reset sync state                                                │
│  4. Resume from new snapshot                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Funding Rate Capture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FUNDING RATE TRACKING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stream: wss://fstream.binance.com/ws/{symbol}@markPrice            │
│  Update Interval: ~3 seconds                                        │
│  Settlement Times: 00:00, 08:00, 16:00 UTC (every 8 hours)          │
│                                                                      │
│  FundingEvent Fields:                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ mark_price_mantissa     │ Price at funding settlement         │  │
│  │ index_price_mantissa    │ Spot reference price               │  │
│  │ estimated_settle_mantissa│ Fair value for settlement          │  │
│  │ funding_rate_mantissa   │ Rate (-8 exponent, e.g., 100=-1bp) │  │
│  │ next_funding_time_ms    │ Next settlement timestamp          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Settlement Detection:                                               │
│  └── Count transitions in next_funding_time_ms                      │
│                                                                      │
│  Derived Metrics:                                                    │
│  - basis_bps = (mark - index) / index × 10000                       │
│  - funding_rate_bps = funding_rate_mantissa × 10^(-8) × 10000       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Strategy Runbook

### 5.1 Available Strategies

| Strategy | Version | Purpose | Market |
|----------|---------|---------|--------|
| funding_bias | v2.1.0 | Funding rate arbitrage | Crypto Perp |
| micro_breakout | v1.0.0 | Momentum breakouts | Any |
| spread_mean_revert | v1.0.0 | Mean reversion from EMA | Any |

### 5.2 Strategy: funding_bias

**Purpose**: Arbitrage perpetual-spot basis via funding rate payments.

**Entry Logic**:
```
SHORT when: funding_rate > +threshold (longs pay shorts)
LONG when:  funding_rate < -threshold (shorts pay longs)
```

**Exit Logic (Hysteresis)**:
```
Exit LONG when:  funding_rate >= exit_band (lost edge)
Exit SHORT when: funding_rate <= -exit_band (lost edge)
```

**Configuration**:
```toml
[strategy.funding_bias]
threshold_mantissa = 100       # 1 basis point @ -6
threshold_exponent = -6
exit_band_mantissa = 50        # 0.5 bps (threshold/2)
position_size_mantissa = 1000000  # 0.01 BTC @ -8
qty_exponent = -8
price_exponent = -2
```

**Backtest Command**:
```bash
./target/release/quantlaxmi-crypto backtest \
  --strategy funding_bias \
  --segment data/perp_sessions/perp_20260130_034709 \
  --out artifacts/funding_bias_backtest
```

### 5.3 Strategy: micro_breakout

**Purpose**: Capture momentum breakouts above/below rolling high/low.

**Entry Logic**:
```
LONG when:  mid > rolling_high × (1 + breakout_bps)
SHORT when: mid < rolling_low × (1 - breakout_bps)
```

**Exit Logic**:
```
- Time stop: Exit after time_stop_secs (default 60s)
- Stop-loss: Exit if loss > stop_loss_bps (default 30 bps)
```

**Configuration**:
```toml
[strategy.micro_breakout]
window_size = 20              # Rolling lookback
breakout_bps_mantissa = 15    # 15 bps threshold @ -2
max_spread_bps_mantissa = 50  # Filter wide spreads
time_stop_secs = 60           # Max hold time
stop_loss_bps_mantissa = 30   # 30 bps stop-loss
position_size_mantissa = 1000000
qty_exponent = -8
price_exponent = -2
```

### 5.4 Strategy: spread_mean_revert

**Purpose**: Trade deviations from exponential moving average.

**Entry Logic**:
```
LONG when:  mid < EMA × (1 - entry_bps)  (expect reversion up)
SHORT when: mid > EMA × (1 + entry_bps)  (expect reversion down)
```

**Exit Logic**:
```
Close when: |mid - EMA| / EMA < exit_bps (reversion complete)
```

**Configuration**:
```toml
[strategy.spread_mean_revert]
ema_alpha_mantissa = 500      # 5% weight on new value @ -4
ema_alpha_exponent = -4
entry_bps_mantissa = 25       # 25 bps entry threshold
exit_bps_mantissa = 5         # 5 bps exit band
max_spread_bps_mantissa = 30  # Filter wide spreads
warmup_ticks = 10             # EMA warmup
position_size_mantissa = 1000000
qty_exponent = -8
price_exponent = -2
```

---

## 6. Risk Management

### 6.1 Risk Envelope

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RISK ENVELOPE LIMITS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Limit                    │ Default (% of equity) │ Action          │
│  ──────────────────────────────────────────────────────────────────│
│  max_order_notional       │ 5%                    │ Reject/Clip     │
│  max_symbol_notional      │ 50%                   │ Reject/Clip     │
│  max_gross_notional       │ 200%                  │ Reject/Clip     │
│                                                                      │
│  Enforcement Priority:                                              │
│  1. Check max_order_notional FIRST                                  │
│  2. Check max_symbol_notional (post-fill projection)                │
│  3. Check max_gross_notional (sum of all positions)                 │
│                                                                      │
│  Risk Mode:                                                         │
│  ┌─────────────────┐  ┌─────────────────┐                          │
│  │     CLIP        │  │     REJECT      │                          │
│  │ Reduce quantity │  │ Fail entire order│                          │
│  │ (research mode) │  │ (production mode)│                          │
│  └─────────────────┘  └─────────────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Circuit Breakers

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CIRCUIT BREAKER TRIGGERS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Trigger                  │ Threshold     │ Action                  │
│  ──────────────────────────────────────────────────────────────────│
│  Consecutive Losses       │ 3             │ Suspend 5 min           │
│  Daily Loss               │ 2% of equity  │ Kill switch             │
│  Max Drawdown             │ 5%            │ Kill switch             │
│  Order Rate               │ 60/minute     │ Throttle                │
│                                                                      │
│  Post-Trade Monitor:                                                │
│  - Warn at 80% of drawdown limit                                    │
│  - Auto kill-switch at 100%                                         │
│  - Real-time equity tracking                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Production Readiness Issues

### 7.1 Critical (Must Fix)

| ID | Component | Issue | Impact |
|----|-----------|-------|--------|
| **C1** | India Auth | Hardcoded sidecar path | Fails if CWD differs |
| **C2** | India Auth | No process timeout | Can hang indefinitely |
| **C3** | India Auth | No token refresh | Session expires |
| **C4** | Live Exec | No pre-execution margin check | Could execute into margin call |
| **C5** | Multi-Leg | Partial fills not hedged | Leaves unhedged exposure |

### 7.2 High (Should Fix)

| ID | Component | Issue | Impact |
|----|-----------|-------|--------|
| **H1** | Reconnect | No jitter in backoff | Thundering herd risk |
| **H2** | Reconnect | No circuit breaker | Infinite loops on auth failure |
| **H3** | Capture | File flush only at deadline | Data loss on crash |
| **H4** | SBE Depth | Sequence gaps only logged | Corrupt book continues |
| **H5** | Funding | Settlement detection fragile | Miss settlement events |

### 7.3 Medium (Nice to Have)

| ID | Component | Issue | Impact |
|----|-----------|-------|--------|
| **M1** | Stats | No buffer size metrics | Silent drops |
| **M2** | Backtest | RealTime pacing not implemented | Can't simulate real-time |
| **M3** | Perp | JSON instead of SBE | Lower throughput |
| **M4** | Snapshot | f64 → i64 conversion | Potential rounding |

---

## 8. Operational Runbook

### 8.1 Starting Crypto Capture

```bash
# 48-hour resilient capture
nohup ./target/release/quantlaxmi-crypto capture-perp-session \
  --symbols BTCUSDT \
  --out-dir data/perp_sessions \
  --duration-secs 172800 \
  --include-spot \
  --include-depth \
  > /tmp/crypto_capture.log 2>&1 &

# Monitor
tail -f logs/quantlaxmi-crypto.log.$(date +%Y-%m-%d)

# Check progress
wc -l data/perp_sessions/perp_*/BTCUSDT/*.jsonl
```

### 8.2 Starting India FNO Capture

```bash
# 6-hour trading day capture
nohup ./target/release/quantlaxmi-india capture-session \
  --underlying NIFTY --underlying BANKNIFTY \
  --strike-band 20 \
  --expiry-policy t1t2t3 \
  --duration-secs 21600 \
  --out-dir data/india_sessions/fno_$(date +%Y%m%d) \
  > /tmp/india_capture.log 2>&1 &

# Monitor
tail -f logs/quantlaxmi-india.log.$(date +%Y-%m-%d)

# Check progress
find data/india_sessions/fno_$(date +%Y%m%d) -name "ticks.jsonl" -exec wc -l {} + | tail -1
```

### 8.3 Running Backtest

```bash
# India KiteSim backtest
./target/release/quantlaxmi-india backtest-kitesim \
  --strategy india_micro_mm \
  --replay data/india_sessions/fno_20260130/nifty/ticks.jsonl \
  --orders artifacts/orders.json \
  --out artifacts/kitesim_run \
  --latency-ms 150 \
  --slippage-bps 0 \
  --hedge-on-failure true

# Crypto backtest
./target/release/quantlaxmi-crypto backtest \
  --strategy funding_bias \
  --segment data/perp_sessions/perp_20260130_034709 \
  --out artifacts/crypto_backtest
```

### 8.4 Gate Validation

```bash
# Run G1 Replay Parity check
./target/release/quantlaxmi-crypto gate-check g1 \
  --live-session data/perp_sessions/perp_20260130_034709 \
  --replay-session artifacts/replay_20260130

# Run G4 Admission Determinism check
./target/release/quantlaxmi-crypto gate-check g4 \
  --live-wal sessions/live/wal \
  --replay-wal sessions/replay/wal
```

---

## 9. Monitoring & Observability

### 9.1 Key Metrics

| Metric | Location | Threshold |
|--------|----------|-----------|
| WebSocket reconnects | `logs/*.log` | < 5/hour |
| Depth sequence gaps | `stats.sequence_gaps` | 0 |
| Fill confirmation timeouts | `stats.timeouts` | < 1% |
| Parse errors | `stats.parse_errors` | 0 |
| Connection gaps | `connection_gaps[]` | Log all |

### 9.2 Log Locations

```
logs/
├── quantlaxmi-crypto.log.YYYY-MM-DD
├── quantlaxmi-india.log.YYYY-MM-DD
└── (rotated daily)

/tmp/
├── crypto_capture.log
├── india_capture.log
└── (session-specific)
```

### 9.3 Health Checks

```bash
# Check running processes
ps aux | grep quantlaxmi

# Check capture status
wc -l data/*/BTCUSDT/*.jsonl

# Check for errors in logs
grep -i "error\|panic\|fail" logs/*.log | tail -20
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-30 | Initial production document |

---

**Document End**
