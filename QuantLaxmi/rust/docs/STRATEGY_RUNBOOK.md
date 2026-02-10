# QuantLaxmi Strategy Runbook

**Version**: 1.0.0
**Date**: 2026-01-30

---

## Overview

This runbook documents all available trading strategies in QuantLaxmi, their configurations, and operational procedures.

---

## 1. Strategy Architecture

### 1.1 Phase 2 Strategy SDK

Strategies in QuantLaxmi follow the **Phase 2 pattern** where strategies author both `DecisionEvent` AND `OrderIntent` together. The engine never infers intents from decisions.

```rust
pub trait Strategy: Send {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn config_hash(&self) -> String;           // SHA-256 of canonical bytes
    fn strategy_id(&self) -> String;           // {name}:{version}:{hash}
    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext)
        -> Vec<DecisionOutput>;
    fn on_fill(&mut self, fill: &FillNotification, ctx: &StrategyContext);
    fn required_signals(&self) -> Vec<SignalRequirements>;
}
```

### 1.2 Fixed-Point Arithmetic

All numeric values use **mantissa + exponent** format to eliminate floating-point drift:

| Exponent | Purpose | Example |
|----------|---------|---------|
| `-8` | Crypto quantities | 1,000,000 @ -8 = 0.01 BTC |
| `-2` | USD prices | 10001 @ -2 = $100.01 |
| `-4` | Confidence scores | 8500 @ -4 = 0.85 |
| `-6` | Funding rates | 100 @ -6 = 0.0001 (1 bps) |

### 1.3 Strategy Registration

```rust
let registry = StrategyRegistry::with_builtins();

// Available strategies:
// - "funding_bias" (v2.1.0)
// - "micro_breakout" (v1.0.0)
// - "spread_mean_revert" (v1.0.0)

let strategy = registry.create("funding_bias", Some(Path::new("config.toml")))?;
```

---

## 2. Strategy Catalog

### 2.1 funding_bias (v2.1.0)

**Purpose**: Arbitrage perpetual-spot basis via funding rate payments.

**Market**: Crypto Perpetuals (Binance Futures)

**Logic**:
- SHORT when funding rate > threshold (longs pay shorts, shorts profit)
- LONG when funding rate < -threshold (shorts pay longs, longs profit)
- Hysteresis band prevents flip-flop noise near threshold

**State Machine**:
```
            funding > +threshold
    FLAT ─────────────────────────▶ SHORT
      ▲                               │
      │                               │
      │ funding ≥ -exit_band          │ funding ≤ -exit_band
      │                               │
      ▼                               ▼
    LONG ◀─────────────────────────  FLAT
            funding < -threshold
```

**Configuration**:
```toml
[strategy.funding_bias]
# Entry threshold (mantissa @ exponent)
threshold_mantissa = 100       # 100 @ -6 = 0.0001 = 1 basis point
threshold_exponent = -6

# Exit hysteresis band (defaults to threshold/2)
exit_band_mantissa = 50        # 0.5 basis points
exit_band_exponent = -6        # (uses threshold_exponent if not set)

# Position sizing
position_size_mantissa = 1000000  # 0.01 BTC
qty_exponent = -8
price_exponent = -2
```

**Decision Events**:
| Tag | Direction | Description |
|-----|-----------|-------------|
| `entry_short` | -1 | Open short on positive funding |
| `entry_long` | +1 | Open long on negative funding |
| `exit_short` | 0 | Close short on funding reversion |
| `exit_long` | 0 | Close long on funding reversion |

**Metadata Fields**:
```json
{
  "tag": "entry_short",
  "funding_rate_mantissa": 150,
  "funding_rate_exponent": -6,
  "threshold_mantissa": 100,
  "exit_band_mantissa": 50,
  "policy": "funding_bias_v2.1"
}
```

**Usage**:
```bash
# Backtest
./target/release/quantlaxmi-crypto backtest \
  --strategy funding_bias \
  --config configs/funding_bias.toml \
  --segment data/perp_sessions/perp_20260130_034709 \
  --out artifacts/funding_bias_backtest

# Paper trading
./target/release/quantlaxmi-crypto paper \
  --strategy funding_bias \
  --config configs/funding_bias.toml \
  --symbol BTCUSDT
```

**Risk Considerations**:
- Funding payments occur every 8 hours (00:00, 08:00, 16:00 UTC)
- Basis can diverge significantly between funding periods
- Position should be sized for potential adverse basis movement

---

### 2.2 micro_breakout (v1.0.0)

**Purpose**: Capture momentum breakouts above/below rolling high/low.

**Market**: Any (Crypto or India FNO)

**Logic**:
- LONG when price breaks above rolling high by breakout_bps
- SHORT when price breaks below rolling low by breakout_bps
- Exit on time stop OR stop-loss

**State Machine**:
```
              mid > high × (1 + bps)
    FLAT ────────────────────────────▶ LONG
      │                                  │
      │                                  │ time_stop OR stop_loss
      │                                  ▼
      │                                FLAT
      │                                  ▲
      │                                  │ time_stop OR stop_loss
      │                                  │
      ▼                                  │
    SHORT ◀────────────────────────────┘
              mid < low × (1 - bps)
```

**Configuration**:
```toml
[strategy.micro_breakout]
# Rolling window for high/low
window_size = 20              # 20 ticks lookback

# Breakout threshold
breakout_bps_mantissa = 15    # 15 basis points
breakout_exponent = -2

# Spread filter (reject if spread too wide)
max_spread_bps_mantissa = 50  # 50 bps max spread
spread_exponent = -2

# Exit conditions
time_stop_secs = 60           # Max 60 seconds hold
stop_loss_bps_mantissa = 30   # 30 bps stop-loss
stop_loss_exponent = -2

# Position sizing
position_size_mantissa = 1000000
qty_exponent = -8
price_exponent = -2
```

**Entry Conditions**:
1. Full window populated (≥20 ticks)
2. Current spread ≤ max_spread_bps
3. Price crosses threshold level

**Exit Conditions** (checked first):
1. Time stop: held for > time_stop_secs
2. Stop-loss: unrealized loss > stop_loss_bps

**Usage**:
```bash
# Backtest on crypto
./target/release/quantlaxmi-crypto backtest \
  --strategy micro_breakout \
  --config configs/micro_breakout.toml \
  --segment data/perp_sessions/perp_20260130_034709 \
  --out artifacts/breakout_backtest

# Backtest on India FNO
./target/release/quantlaxmi-india backtest-kitesim \
  --strategy micro_breakout \
  --replay data/india_sessions/fno_20260130/ticks.jsonl \
  --orders artifacts/orders.json \
  --out artifacts/india_breakout
```

**Risk Considerations**:
- Momentum strategies suffer in range-bound markets
- Wide spreads can erode profitability
- Time stop prevents holding through reversals

---

### 2.3 spread_mean_revert (v1.0.0)

**Purpose**: Fade overshoots from exponential moving average.

**Market**: Any (Crypto or India FNO)

**Logic**:
- LONG when price is below EMA by entry_bps (expect reversion up)
- SHORT when price is above EMA by entry_bps (expect reversion down)
- Exit when deviation shrinks to exit_bps (reversion complete)

**EMA Update (Fixed-Point)**:
```
new_ema = (alpha × price + (scale - alpha) × old_ema) / scale
where scale = 10^(-exponent) = 10000 for alpha @ -4
```

**Configuration**:
```toml
[strategy.spread_mean_revert]
# EMA smoothing factor
ema_alpha_mantissa = 500      # 500 @ -4 = 0.05 (5% weight on new value)
ema_alpha_exponent = -4       # Effective ~20-tick window

# Entry threshold
entry_bps_mantissa = 25       # 25 bps deviation required
entry_exponent = -2

# Exit threshold (hysteresis)
exit_bps_mantissa = 5         # 5 bps = reversion complete
exit_exponent = -2

# Spread filter
max_spread_bps_mantissa = 30  # 30 bps max spread
spread_exponent = -2

# Warmup period
warmup_ticks = 10             # Wait 10 ticks for EMA initialization

# Position sizing
position_size_mantissa = 1000000
qty_exponent = -8
price_exponent = -2
```

**Entry Conditions**:
1. Warmup complete (≥10 ticks)
2. Current spread ≤ max_spread_bps
3. |deviation| ≥ entry_bps

**Exit Conditions**:
1. |deviation| < exit_bps (reversion complete)

**Usage**:
```bash
# Backtest
./target/release/quantlaxmi-crypto backtest \
  --strategy spread_mean_revert \
  --config configs/mean_revert.toml \
  --segment data/perp_sessions/perp_20260130_034709 \
  --out artifacts/mean_revert_backtest
```

**Risk Considerations**:
- Mean reversion fails in trending markets
- Narrow exit band may exit too early
- Wide entry band may miss opportunities

---

## 3. Strategy Development

### 3.1 Creating a New Strategy

```rust
use quantlaxmi_strategy::{
    Strategy, StrategyContext, DecisionOutput, ReplayEvent,
    FillNotification, SignalRequirements, OrderIntent, Side,
};
use quantlaxmi_models::DecisionEvent;

pub struct MyStrategy {
    config: MyStrategyConfig,
    state: MyState,
}

impl Strategy for MyStrategy {
    fn name(&self) -> &str { "my_strategy" }
    fn version(&self) -> &str { "1.0.0" }

    fn config_hash(&self) -> String {
        // Hash from canonical bytes (NOT JSON)
        let bytes = self.config.to_canonical_bytes();
        sha256_hex(&bytes)
    }

    fn strategy_id(&self) -> String {
        format!("{}:{}:{}", self.name(), self.version(), &self.config_hash()[..8])
    }

    fn on_event(&mut self, event: &ReplayEvent, ctx: &StrategyContext)
        -> Vec<DecisionOutput>
    {
        // Process event, potentially emit decisions + intents
        vec![]
    }

    fn on_fill(&mut self, fill: &FillNotification, ctx: &StrategyContext) {
        // Update position tracking
    }

    fn required_signals(&self) -> Vec<SignalRequirements> {
        // Declare required data fields for admission
        vec![]
    }
}
```

### 3.2 Configuration Canonical Encoding

All config fields must be encoded in a fixed, deterministic order:

```rust
impl CanonicalBytes for MyStrategyConfig {
    fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(CONFIG_ENCODING_VERSION);  // 0x01

        // Fields in struct definition order
        buf.extend_from_slice(&self.threshold_mantissa.to_le_bytes());
        buf.push(self.threshold_exponent as u8);

        // Option<T>: 0x00 for None, 0x01 + value for Some
        match self.exit_band_mantissa {
            Some(v) => {
                buf.push(0x01);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            None => buf.push(0x00),
        }

        buf
    }
}
```

### 3.3 Registering a Strategy

```rust
// In registry.rs
impl StrategyRegistry {
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();

        registry.register("funding_bias", Box::new(|config| {
            let cfg = load_config::<FundingBiasConfig>(config)?;
            Ok(Box::new(FundingBiasStrategy::new(cfg)))
        }));

        registry.register("my_strategy", Box::new(|config| {
            let cfg = load_config::<MyStrategyConfig>(config)?;
            Ok(Box::new(MyStrategy::new(cfg)))
        }));

        registry
    }
}
```

---

## 4. Backtest Procedures

### 4.1 Crypto Backtest

```bash
# Full backtest with all output artifacts
./target/release/quantlaxmi-crypto backtest \
  --strategy funding_bias \
  --config configs/funding_bias.toml \
  --segment data/perp_sessions/perp_20260130_034709 \
  --out artifacts/backtest_$(date +%Y%m%d_%H%M%S) \
  --initial-capital 10000

# Output files:
# - report.json        (summary metrics)
# - decisions.jsonl    (all decisions)
# - fills.jsonl        (all fills)
# - equity_curve.jsonl (time series)
# - pnl.json           (P&L breakdown)
```

### 4.2 India FNO Backtest (KiteSim)

```bash
# Step 1: Generate orders from replay
./target/release/quantlaxmi-india generate-orders \
  --strategy india_micro_mm \
  --replay data/india_sessions/fno_20260130/ticks.jsonl \
  --out artifacts/orders.json \
  --routing-log artifacts/routing_decisions.jsonl

# Step 2: Run KiteSim backtest
./target/release/quantlaxmi-india backtest-kitesim \
  --strategy india_micro_mm \
  --replay data/india_sessions/fno_20260130/ticks.jsonl \
  --orders artifacts/orders.json \
  --intents artifacts/intents.json \
  --out artifacts/kitesim_run \
  --latency-ms 150 \
  --slippage-bps 0 \
  --hedge-on-failure true

# Step 3: Generate labels (for ML training)
./target/release/quantlaxmi-india generate-labels \
  --routing-decisions artifacts/routing_decisions.jsonl \
  --fills artifacts/kitesim_run/fills.jsonl \
  --quotes data/india_sessions/fno_20260130/ticks.jsonl \
  --out artifacts/labels.jsonl
```

### 4.3 Backtest Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Sharpe Ratio | Risk-adjusted return (annualized) | > 1.5 |
| Sortino Ratio | Downside deviation only | > 2.0 |
| Max Drawdown | Peak-to-trough decline | < 10% |
| Win Rate | Profitable trades / total | > 55% |
| Profit Factor | Gross profit / |gross loss| | > 1.5 |

---

## 5. Live Trading Procedures

### 5.1 Pre-Flight Checklist

- [ ] Backtest shows positive expectancy
- [ ] G1 Replay Parity check passes
- [ ] Risk limits configured
- [ ] Kill switch accessible
- [ ] Credentials verified
- [ ] Paper trading validated

### 5.2 Paper Trading

```bash
# Crypto paper trading
./target/release/quantlaxmi-crypto paper \
  --strategy funding_bias \
  --config configs/funding_bias.toml \
  --symbol BTCUSDT \
  --initial-capital 10000

# India paper trading
./target/release/quantlaxmi-india paper \
  --config configs/paper.toml \
  --initial-capital 1000000
```

### 5.3 Live Trading

```bash
# Crypto live (CAUTION: REAL MONEY)
./target/release/quantlaxmi-crypto live \
  --strategy funding_bias \
  --config configs/live.toml \
  --symbol BTCUSDT

# India live (CAUTION: REAL MONEY)
./target/release/quantlaxmi-india live \
  --config configs/live.toml \
  --initial-capital 1000000
```

### 5.4 Emergency Procedures

**Kill Switch Activation**:
```bash
# Send SIGTERM to gracefully stop
kill -TERM <PID>

# Force stop if graceful fails
kill -9 <PID>
```

**Position Reconciliation**:
```bash
# Check Binance positions
# (manual via API or web UI)

# Check Zerodha positions
./target/release/quantlaxmi-india get-positions
```

---

## 6. Performance Tuning

### 6.1 Strategy Parameters

| Strategy | Parameter | Impact | Recommended Range |
|----------|-----------|--------|-------------------|
| funding_bias | threshold | Trade frequency | 50-200 @ -6 |
| funding_bias | exit_band | Hold time | 25-100 @ -6 |
| micro_breakout | breakout_bps | Entry sensitivity | 10-30 @ -2 |
| micro_breakout | time_stop_secs | Hold time | 30-120 |
| spread_mean_revert | ema_alpha | Responsiveness | 200-1000 @ -4 |
| spread_mean_revert | entry_bps | Trade frequency | 15-50 @ -2 |

### 6.2 Optimization Workflow

```bash
# Grid search example (pseudo-code)
for threshold in 50 100 150 200; do
  for exit_band in 25 50 75; do
    ./target/release/quantlaxmi-crypto backtest \
      --strategy funding_bias \
      --param threshold_mantissa=$threshold \
      --param exit_band_mantissa=$exit_band \
      --segment data/segment \
      --out artifacts/grid_${threshold}_${exit_band}
  done
done

# Analyze results
python scripts/analyze_grid.py artifacts/grid_*/report.json
```

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No signals generated | Warmup not complete | Wait for window_size ticks |
| Too many signals | Threshold too low | Increase threshold |
| Poor fill quality | Wide spreads | Tighten max_spread filter |
| Large drawdowns | No stop-loss | Add stop_loss_bps |
| Flip-flop trades | No hysteresis | Add exit_band different from threshold |

### 7.2 Diagnostic Commands

```bash
# Check strategy decisions
jq '.decision_type' artifacts/decisions.jsonl | sort | uniq -c

# Check fill quality
jq '.slippage_bps' artifacts/fills.jsonl | stats

# Check position changes
jq 'select(.direction != 0)' artifacts/decisions.jsonl | wc -l
```

---

## 8. India FNO Options Engine

### 8.1 Overview

The Options Engine provides automated strategy selection for India FNO markets using:
- **Grassmann Manifold Regime Detection** - Geometric analysis of market microstructure
- **PCR (Put-Call Ratio) Analysis** - Sentiment and positioning signals
- **IV Surface Tracking** - Volatility regime classification
- **Ramanujan Periodicity** - HFT detection and avoidance

### 8.2 Running the Options Engine TUI

```bash
# Basic usage with ATM options auto-discovery
cargo run --bin live-paper-tui -- --symbols NIFTY-ATM

# With custom parameters
cargo run --bin live-paper-tui -- \
  --symbols NIFTY-ATM \
  --capital 1000000 \
  --min-score 60 \
  --warmup-minutes 60
```

**TUI Controls:**
- `q` or `Esc` - Quit
- `Ctrl+C` - Force quit

### 8.3 Market Data Display

The TUI shows real-time market data:

| Field | Source | Description |
|-------|--------|-------------|
| Spot | NIFTY 50 Index | Current index price |
| Futures | NIFTY26XXXFUT | Current month futures price |
| Basis | Futures - Spot | Cash-futures spread |
| ATM IV | Option chain | At-the-money implied volatility |
| IV Percentile | Historical IV | Current IV vs 252-day history |
| Vol Regime | IV Percentile | LowVol / NormalVol / HighVol |
| PCR | Option chain | Put-Call ratio from OI |
| PCR Signal | PCR trend | Bullish / Bearish / Neutral |
| Regime | Grassmann | Quiet / TrendImpulse / MeanReversionChop / etc. |

### 8.4 Strategy Selection

The engine automatically selects from:

| Strategy | Best Regime | Vol Regime | Description |
|----------|-------------|------------|-------------|
| IronCondor | Quiet | NormalVol | Range-bound premium collection |
| IronButterfly | Quiet | HighVol | Pinned expiry play |
| Straddle | TrendImpulse | LowVol | Directional breakout |
| Strangle | TrendImpulse | NormalVol | Wide breakout |
| CalendarSpread | MeanReversionChop | Any | Time decay arbitrage |
| VerticalSpread | Any | Any | Directional bias |

**Strategy Score Components:**
- Regime Score (0-25): Regime alignment
- Vol Score (0-25): IV percentile alignment
- PCR Score (0-15): Sentiment alignment
- Risk Score (0-15): Position sizing
- Edge Score (0-20): Expected value

**Minimum Score**: Configurable via `--min-score` (default: 60)

### 8.5 Warmup Process

The engine requires warmup data before trading:

```bash
# Default 60-minute warmup
cargo run --bin live-paper-tui -- --symbols NIFTY-ATM --warmup-minutes 60

# Debug warmup process
cargo run --bin warmup-debug
```

**Warmup Requirements:**
- 32+ ticks for first Grassmann subspace
- 60+ ticks for prototype bank population
- NIFTY 50 historical data from Kite Connect API

### 8.6 Symbol Resolution

The connector automatically resolves:
- `NIFTY-ATM` → Discovers ATM strikes ±20 around spot
- `NIFTY 50` → Index token 256265 from NSE
- `NIFTY26JANFUT` → Current month futures from NFO

```bash
# Check resolved tokens in TUI stderr output:
[TUI] Subscribing to: NIFTY 50, NIFTY26JANFUT, NIFTY26FEBFUT
[TUI] Resolved 45 symbols
[TUI]   NIFTY 50 -> token 256265
[TUI]   NIFTY26JANFUT -> token 12345678
```

### 8.7 Configuration

```rust
EngineConfig {
    symbol: "NIFTY".into(),
    lot_size: 50,
    risk_free_rate: 0.065,      // 6.5% RBI rate
    dividend_yield: 0.012,       // ~1.2% NIFTY dividend yield
    max_positions: 3,
    max_loss_per_position: 5000.0,
    max_portfolio_delta: 500.0,
    min_iv_percentile_sell: 60.0,  // Sell premium when IV > 60th pctl
    max_iv_percentile_buy: 40.0,   // Buy premium when IV < 40th pctl
    min_strategy_score: 60.0,
    ramanujan_enabled: true,       // HFT detection
    block_on_hft: true,            // Block trades during HFT
    pcr_enabled: true,
    pcr_lookback: 100,
}
```

---

## 9. Data Capture

### 9.1 India FNO Capture

```bash
# Start capture session
./target/release/quantlaxmi-india capture-session \
  --underlying NIFTY,BANKNIFTY \
  --strike-band 20 \
  --expiry-policy t1t2t3 \
  --out-dir data/india_sessions/fno_$(date +%Y%m%d) \
  --duration-secs 21600

# Monitor progress
tail -f logs/quantlaxmi-india.log.$(date +%Y-%m-%d)

# Check tick count
find data/india_sessions/fno_* -name "ticks.jsonl" -exec wc -l {} + | tail -1
```

### 9.2 Crypto Perpetuals Capture

```bash
# Start capture session
./target/release/quantlaxmi-crypto capture-perp-session \
  --symbols BTCUSDT \
  --out-dir data/perp_sessions \
  --duration-secs 172800 \
  --include-spot \
  --include-depth

# Check running captures
ps aux | grep quantlaxmi
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-30 | Initial runbook |
| 1.1.0 | 2026-01-30 | Added Options Engine TUI, symbol resolution, data capture |

---

**Document End**
