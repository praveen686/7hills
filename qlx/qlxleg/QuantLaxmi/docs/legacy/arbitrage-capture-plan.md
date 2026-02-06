# Arbitrage-Ready Crypto Capture System

## Objective
Build a multi-symbol, multi-stream capture system for QuantLaxmi_Crypto that supports:
- Triangular arbitrage (needs 3+ legs)
- Cross-crypto statistical arbitrage (needs basket of majors)
- Basis/funding arbitrage (needs spot + perp pairs)
- Deterministic replay and VectorBT export

## Pre-requisite: Environment Setup
Copy `.env` from QuantKubera1 to QuantLaxmi:
```
cp /home/isoula/7hills/QuantKubera1/.env /home/isoula/7hills/QuantLaxmi/.env
```

---

## Symbol Universe (Profile 1 - Arbitrage-Ready Core)

| Category | Symbols |
|----------|---------|
| **Spot USDT** | BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT |
| **Cross pairs** | ETHBTC, BNBBTC, SOLBTC |
| **Perps** (if futures) | BTCUSDT_PERP, ETHUSDT_PERP |

---

## Implementation Tasks

### Phase 1: Trades Capture (Day 1)

**Gap**: Trade SBE decoder exists (template 10000) but no capture command.

**Files to modify:**
- `crates/quantlaxmi-runner-crypto/src/lib.rs` - Add `CaptureTrades` command
- `crates/quantlaxmi-runner-crypto/src/binance_trades_capture.rs` - NEW file

**Implementation:**
```rust
// New CLI command
CaptureTrades {
    #[arg(long)]
    symbol: String,
    #[arg(long)]
    out: String,
    #[arg(long, default_value = "300")]
    duration_secs: u64,
}
```

**Trade event structure** (JSONL output):
```json
{
  "ts": "2026-01-22T12:00:00.123Z",
  "tradingsymbol": "BTCUSDT",
  "trade_id": 123456789,
  "price": 9000012,
  "qty": 150000000,
  "price_exponent": -2,
  "qty_exponent": -8,
  "is_buyer_maker": false,
  "integrity_tier": "CERTIFIED"
}
```

---

### Phase 2: Multi-Symbol Session Capture (Day 1-2)

**Files to modify:**
- `crates/quantlaxmi-runner-crypto/src/lib.rs` - Add `CaptureSession` command
- `crates/quantlaxmi-runner-crypto/src/session_capture.rs` - NEW file

**CLI:**
```rust
CaptureSession {
    #[arg(long)]
    symbols: String,  // "BTCUSDT,ETHUSDT,BNBUSDT"

    #[arg(long)]
    out_dir: String,  // "data/sessions/2026-01-22_A"

    #[arg(long, default_value = "7200")]
    duration_secs: u64,

    #[arg(long)]
    include_trades: bool,

    #[arg(long)]
    strict: bool,
}
```

**Output structure:**
```
data/sessions/2026-01-22_A/
├── session_manifest.json       # Meta-manifest for entire session
├── BTCUSDT/
│   ├── depth.jsonl
│   ├── trades.jsonl
│   └── manifest.json
├── ETHUSDT/
│   ├── depth.jsonl
│   ├── trades.jsonl
│   └── manifest.json
└── BNBUSDT/
    ├── depth.jsonl
    ├── trades.jsonl
    └── manifest.json
```

**Architecture**: Spawn parallel tokio tasks per symbol, each with independent SBE WebSocket connection (current actor model). Aggregate stats at session level.

---

### Phase 3: Mark Price / Funding Capture (Day 2-3)

**For futures/perpetuals only.**

**Files to modify:**
- `crates/kubera-models/src/funding.rs` - NEW file for FundingEvent
- `crates/quantlaxmi-connectors-binance/src/sbe.rs` - Add mark price decoder (if SBE available)
- `crates/quantlaxmi-runner-crypto/src/funding_capture.rs` - NEW file

**Note**: Mark price may not be available via SBE. May need REST polling at fixed cadence (1s or 3s).

**FundingEvent structure:**
```rust
pub struct FundingEvent {
    pub ts: DateTime<Utc>,
    pub tradingsymbol: String,
    pub mark_price: i64,
    pub index_price: i64,
    pub funding_rate: i64,  // scaled by 1e8
    pub next_funding_time: DateTime<Utc>,
    pub price_exponent: i8,
}
```

---

### Phase 4: VectorBT Export (Day 2-3)

**Files to modify:**
- `crates/quantlaxmi-runner-crypto/src/vectorbt_export.rs` - NEW file
- `crates/quantlaxmi-runner-crypto/src/lib.rs` - Add `ExportVectorbt` command

**Output files:**
```
vectorbt/
├── market.csv      # OHLCV + features
├── fills.csv       # Paper/backtest fills
└── summary.json    # Session metadata
```

**market.csv schema:**
```csv
timestamp,symbol,open,high,low,close,volume,bid,ask,spread,mid
2026-01-22T12:00:00Z,BTCUSDT,90000.12,90005.50,...
```

**fills.csv schema:**
```csv
timestamp,symbol,side,price,qty,fee,pnl,position
2026-01-22T12:00:05Z,BTCUSDT,BUY,90001.00,0.01,...
```

---

### Phase 5: Session Orchestration Script (Day 3)

**File**: `scripts/run_crypto_session.sh`

```bash
#!/bin/bash
# Usage: ./run_crypto_session.sh --symbols BTCUSDT,ETHUSDT --duration 2h --tag session_A

SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT}"
DURATION="${DURATION:-7200}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="data/sessions/${TAG}"

# 1. Capture session (depth + trades)
cargo run --release --bin quantlaxmi-crypto -- \
  capture-session \
  --symbols "$SYMBOLS" \
  --out-dir "$OUT_DIR" \
  --duration-secs "$DURATION" \
  --include-trades \
  --strict

# 2. Validate determinism (replay short slice)
cargo run --release --bin quantlaxmi-crypto -- \
  validate-session --session-dir "$OUT_DIR"

# 3. Export VectorBT
cargo run --release --bin quantlaxmi-crypto -- \
  export-vectorbt --session-dir "$OUT_DIR" --out-dir "$OUT_DIR/vectorbt"

echo "Session complete: $OUT_DIR"
```

---

## Quality Gates (Enforced)

| Gate | Check | Failure Action |
|------|-------|----------------|
| **Bootstrap** | Snapshot applied, diffs sequenced | Hard fail |
| **Gap rate** | strict=true → 0 gaps | Abort capture |
| **Determinism** | Replay hash matches | Flag as non-certified |
| **Economics** | Spread p50/p95, crossed book check | Log warning |

---

## Critical Files Summary

| File | Purpose |
|------|---------|
| `lib.rs` | CLI commands, orchestration |
| `binance_sbe_depth_capture.rs` | Existing depth capture |
| `binance_trades_capture.rs` | NEW: trades capture |
| `session_capture.rs` | NEW: multi-symbol orchestration |
| `vectorbt_export.rs` | NEW: CSV export for research |
| `funding_capture.rs` | NEW: mark/funding (futures) |
| `paper_trading.rs` | Existing WAL + PaperVenue |

---

## Verification Plan

1. **Smoke test** (10 min):
   ```bash
   cargo run --release --bin quantlaxmi-crypto -- \
     capture-session --symbols BTCUSDT,ETHUSDT --duration-secs 600 \
     --out-dir data/smoke_test --include-trades --strict
   ```

2. **Determinism check**:
   ```bash
   cargo test -p quantlaxmi-runner-crypto determinism
   ```

3. **VectorBT export validation**:
   - Verify market.csv has correct columns
   - Verify fills.csv populates from paper session

4. **Full 2-hour session**:
   ```bash
   ./scripts/run_crypto_session.sh --symbols BTCUSDT,ETHUSDT,BNBUSDT \
     --duration 2h --tag first_arb_session
   ```

---

## First Week Deliverables

| Day | Deliverable |
|-----|-------------|
| 1 | Trades capture + 20-min smoke test |
| 1 | Multi-symbol session capture (depth + trades) |
| 2 | 2-hour BTC/ETH/BNB session captured |
| 2 | VectorBT export working |
| 3 | Mark/funding capture (if futures) |
| 3 | Session orchestration script |
| 4-7 | Daily captures, expand to Profile 1 symbols |
