# Crypto Calendar-Carry Design

**Status:** Design Phase
**Date:** Jan 24, 2026
**Author:** Claude + User

---

## 1. India Calendar-Carry Recap

The India strategy exploits term structure mispricing:

```
Position: Short Front Straddle + Long Back Straddle
Profit Source: Front decays faster than back (theta differential)
Edge Detection: SANOS calibration finds vol surface anomalies
```

| Component | India Implementation |
|-----------|---------------------|
| Front leg | Monthly expiry options (T1) |
| Back leg | Next monthly expiry (T2) |
| Instrument | ATM straddles (CE + PE) |
| Carry | Theta decay differential |
| Risk | Gamma, vega, pin risk |

---

## 2. Crypto Equivalents

Crypto has **different** but analogous term structure dynamics:

### 2.1 Available Instruments (Binance)

| Instrument | Expiry | Funding | Basis |
|------------|--------|---------|-------|
| Spot | None | None | Reference |
| Perpetual | Never | 8h intervals | vs Spot |
| Quarterly | Fixed dates | None | vs Spot/Perp |
| Bi-quarterly | Fixed dates | None | vs Spot/Perp |

### 2.2 Crypto "Calendar" Concepts

| India Concept | Crypto Analog | Mechanism |
|---------------|---------------|-----------|
| Front expiry | Perpetual | Continuous, funding-adjusted |
| Back expiry | Quarterly future | Fixed expiry, basis converges |
| Theta decay | Funding rate | Paid/received every 8h |
| Vol surface | Basis term structure | Spot-Perp-Quarterly spreads |
| Carry | Funding + Basis | Combined yield |

---

## 3. Crypto Calendar-Carry Strategy Options

### Option A: Perp vs Quarterly Basis Trade

```
Position: Short Perp + Long Quarterly (or vice versa)
Profit: Basis convergence + funding differential
```

**Mechanics:**
- Quarterly trades at premium/discount to perp
- As expiry approaches, basis converges to zero
- Funding adds/subtracts from P&L

**Pros:** Direct analog to India calendar spread
**Cons:** Requires quarterly futures data (not always liquid)

### Option B: Funding Rate Arbitrage (Spot-Perp)

```
Position: Long Spot + Short Perp (when funding positive)
          Short Spot + Long Perp (when funding negative)
Profit: Collect funding payments
```

**Mechanics:**
- When perp trades above spot, longs pay shorts (positive funding)
- Delta-neutral: Spot position hedges perp exposure
- Collect funding every 8 hours

**Pros:** Simpler, more liquid, continuous
**Cons:** Not a "calendar" spread per se, but economically similar

### Option C: Cross-Exchange Basis (Advanced)

```
Position: Long Binance Perp + Short OKX Perp (or similar)
Profit: Funding differential between exchanges
```

**Cons:** Requires multi-exchange, higher complexity

---

## 4. Recommended Strategy: Option B (Funding Arbitrage)

**Rationale:**
1. Most liquid (BTCUSDT perp is highest volume)
2. Simpler execution (2 legs vs 4 in India)
3. Continuous (no expiry gaps)
4. Abundant data (8h funding cycles = many samples fast)

### 4.1 Strategy Logic

```
Signal Generation:
1. Monitor funding rate (predicted next 8h)
2. If funding > threshold → Short Perp + Long Spot
3. If funding < -threshold → Long Perp + Short Spot
4. Hold through funding timestamp
5. Exit after funding collected

Entry Conditions:
- |funding_rate| > min_threshold (e.g., 0.01% = 1bp)
- Spread (perp-spot) consistent with funding direction
- Quote age within tolerance

Exit Conditions:
- Funding timestamp passed
- Basis reverted beyond threshold
- Stop loss on adverse move
```

### 4.2 P&L Components

```
P&L = Funding_Received
    - Spread_Cost (entry + exit)
    - Slippage
    - Fees
```

**Key insight:** Funding is known in advance (8h prediction), making this more deterministic than India vol trading.

---

## 5. Infrastructure Gaps

### 5.1 Current State (Spot-Only)

| Component | Status | Location |
|-----------|--------|----------|
| Spot depth capture | ✅ Done | binance_sbe_depth_capture.rs |
| Spot trades capture | ✅ Done | binance_trades_capture.rs |
| Spot bookTicker | ✅ Done | binance_capture.rs |
| Perp depth capture | ❌ Missing | - |
| Perp trades capture | ❌ Missing | - |
| Funding rate fetch | ❌ Missing | - |
| Quarterly futures | ❌ Missing | - |

### 5.2 Required New Components

#### 5.2.1 Binance Futures WebSocket Streams

**Endpoint:** `wss://fstream.binance.com/ws/<symbol>@depth`

Streams needed:
- `btcusdt@depth` (perp order book)
- `btcusdt@aggTrade` (perp trades)
- `btcusdt@markPrice` (funding rate + mark price)

#### 5.2.2 Funding Rate Data Structure

```rust
/// Funding rate event (every ~1 second from markPrice stream)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingEvent {
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub mark_price: f64,
    pub index_price: f64,        // Spot reference
    pub funding_rate: f64,       // Current predicted rate
    pub next_funding_time: DateTime<Utc>,  // When funding settles
}
```

#### 5.2.3 Basis Calculation

```rust
/// Basis = (Perp_Price - Spot_Price) / Spot_Price
/// Annualized = Basis * (365 * 3) for 8h funding
pub struct BasisEvent {
    pub ts: DateTime<Utc>,
    pub symbol: String,
    pub spot_mid: f64,
    pub perp_mid: f64,
    pub basis_bps: f64,           // In basis points
    pub basis_annualized: f64,    // Annualized yield
    pub funding_rate: f64,
    pub next_funding_time: DateTime<Utc>,
}
```

---

## 6. Implementation Phases

### Phase 1: Perp Data Capture (Week 1)

**Goal:** Capture perp depth + funding alongside spot

```
New files:
├── binance_perp_depth_capture.rs   # fstream.binance.com
├── binance_funding_capture.rs       # markPrice stream
└── binance_perp_session.rs          # Spot + Perp orchestrator
```

**Output format:** Same JSONL structure as spot, with `market: "perp"` field

### Phase 2: Basis Signal Generation (Week 2)

**Goal:** Generate funding arbitrage signals

```
New files:
├── run_funding_arb.rs              # Signal generator
└── funding_arb_signals.jsonl       # Output format
```

**Signal schema:**
```json
{
  "ts": "2026-01-24T12:00:00Z",
  "symbol": "BTCUSDT",
  "direction": "short_perp_long_spot",
  "funding_rate": 0.0003,
  "basis_bps": 5.2,
  "next_funding_time": "2026-01-24T16:00:00Z",
  "spot_mid": 100000.50,
  "perp_mid": 100052.30
}
```

### Phase 3: Scoring Engine (Week 3)

**Goal:** Backtest signals with realistic fills

```
Reuse from India:
- Quote staleness logic (find_quote_at_or_after)
- Bid/ask fill simulation
- Drop reason tracking
- fills.jsonl audit format

New:
- Funding P&L calculation
- Continuous time handling (no expiry)
- Multi-leg execution simulation
```

### Phase 4: Live Paper Trading (Week 4)

**Goal:** WAL-based paper trading with funding collection

```
Extend paper_trading.rs:
- Track funding accruals
- Mark-to-market with basis
- P&L attribution (funding vs basis vs slippage)
```

---

## 7. Key Differences from India

| Aspect | India | Crypto |
|--------|-------|--------|
| Legs | 4 (CE+PE × 2 expiries) | 2 (Spot + Perp) |
| Expiry | Monthly | Continuous |
| Carry source | Theta decay | Funding rate |
| Data frequency | Market hours only | 24/7 |
| Sample velocity | ~5 trades/day | ~50+ trades/day |
| Calibration | SANOS vol surface | Basis + funding threshold |

---

## 8. Success Criteria

### Minimum Viable Crypto Alpha

| Metric | Target |
|--------|--------|
| Trades | ≥500 |
| Days | ≥7 rolling |
| Net P&L | Positive after fees |
| Hit rate | >55% |
| Max DD | <30% of gross |
| Funding dominance | <80% (edge not purely funding) |

---

## 9. Immediate Next Steps

1. **Create `binance_perp_depth_capture.rs`** - Capture perp order book
2. **Create `binance_funding_capture.rs`** - Capture funding rate stream
3. **Create `BasisEvent` struct** - Unified spot+perp+funding view
4. **Test capture** - 24h rolling session on BTCUSDT
5. **Design signal logic** - Entry/exit thresholds for funding arb

---

## 10. Open Questions

1. **Funding threshold:** What minimum funding rate justifies entry? (0.01%? 0.03%?)
2. **Holding period:** Enter before funding, exit immediately after? Or hold longer?
3. **Position sizing:** Fixed notional or kelly-based?
4. **Multi-symbol:** Start with BTC only, or BTC+ETH from day 1?

These should be answered by data, not assumptions.

---

## Appendix: Binance Futures API Reference

### WebSocket Streams (fstream.binance.com)

```
# Order book depth (perp)
wss://fstream.binance.com/ws/btcusdt@depth@100ms

# Aggregated trades (perp)
wss://fstream.binance.com/ws/btcusdt@aggTrade

# Mark price + funding (every 3s)
wss://fstream.binance.com/ws/btcusdt@markPrice

# All mark prices (every 3s)
wss://fstream.binance.com/ws/!markPrice@arr
```

### REST Endpoints

```
# Current funding rate
GET /fapi/v1/premiumIndex?symbol=BTCUSDT

# Funding rate history
GET /fapi/v1/fundingRate?symbol=BTCUSDT&limit=100

# Order book snapshot
GET /fapi/v1/depth?symbol=BTCUSDT&limit=1000
```

### Funding Rate Mechanics

- Settles every 8 hours: 00:00, 08:00, 16:00 UTC
- Rate clamped: typically ±0.75% per 8h
- Paid by longs to shorts (positive) or shorts to longs (negative)
- Predicted rate updates every second via markPrice stream
