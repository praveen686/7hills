# Intraday TFT Design — Multi-Timeframe Architecture

Design document for extending the TFT system to support both EOD (end-of-day) and
intraday trading. Captures the full architectural discussion from 2026-02-10.

---

## 1. Problem Statement

The current TFT operates at **daily frequency** — one signal per trading day, using
73 VSN-selected features (including tick-derived features aggregated to daily). This
serves EOD strategies (S1-S25) well but leaves intraday alpha on the table.

**Goal**: Add an intraday TFT that produces signals every 5-min bar, while keeping
the daily TFT for swing/EOD strategies. Both models must coexist on a single T4 GPU.

---

## 2. Three Architectural Approaches

### Approach 1: Two Separate TFTs (Simplest)

```
Daily TFT (current)          Intraday TFT (new)
├── 73 daily features         ├── ~50 minute-bar features
├── seq_len=63 days           ├── seq_len=120 bars (2 hrs)
├── target: next-day return   ├── target: next-30min return
├── 1 signal/day              ├── 1 signal/bar
└── → EOD strategies          └── → Intraday strategies
```

Completely independent. Each optimized for its frequency. No cross-frequency learning.

### Approach 2: Hierarchical (Recommended)

```
Daily TFT (current, already trained)
  │
  │  regime signal (bullish/bearish/neutral) + confidence
  │
  ▼
Intraday TFT
  ├── Minute-bar features (OHLCV, VWAP, order flow, spread)
  ├── Daily TFT output as STATIC covariate (regime context)
  ├── seq_len=120 bars
  ├── target: next-30min vol-scaled return
  └── Intraday signal conditioned on daily regime
```

Information flows naturally from higher → lower timeframe. The intraday model learns
that "long trades in bullish regime have higher Sharpe." TFT's architecture supports
this natively — static covariates feed into the Variable Selection Network.

### Approach 3: Multi-Scale Single TFT (Most Complex)

```
Single TFT
  ├── Static: daily regime features, VIX, FII flows
  ├── Time-varying (slow): hourly aggregates, VWAP bands
  ├── Time-varying (fast): 1-min OHLCV, tick features
  ├── Multi-resolution attention across scales
  └── Single output at minute frequency
```

Learns cross-frequency interactions end-to-end but much harder to train, debug,
and interpret.

### Why Approach 2 Wins

1. The daily TFT is already trained and validated (Sharpe ~1.88)
2. TFT's static covariate input is literally designed for this — daily regime
   becomes a static feature for the intraday model
3. We already have the minute-bar data infrastructure (Kite 1-min collector,
   tick features)
4. It keeps each model interpretable at its own timescale
5. The daily model doesn't need retraining — we just add a new intraday pipeline

**Note**: Approach 1 and Approach 2 converge in practice — even in Approach 1,
the daily TFT regime signal naturally becomes a static covariate for the intraday
model when combining signals for trading.

---

## 3. Key Differences: Daily vs Intraday TFT

| Dimension | Daily TFT (current) | Intraday TFT (new) |
|-----------|---------------------|---------------------|
| Bar size | 1 day | 5 min |
| seq_len | 63 bars (days) | 60-120 bars (1-2 hrs) |
| Target | Next-day return | Next-30min return |
| Features | FII flows, VIX, OI, breadth | VWAP dev, spread, tick imbalance, order flow |
| Training data | ~500 bars/asset/year | ~18,750 bars/asset/year (5-min) |
| Purge gap | 5 days | 30-60 bars (minutes) |
| Walk-forward window | train=150d, test=42d | train=20d, test=5d |
| Inference speed | <50ms (once/day) | <50ms (per bar) |

---

## 4. Resource Feasibility on EC2 (T4 GPU)

### EC2 Spec
- 32 vCPUs, 124GB RAM, Tesla T4 16GB VRAM, CUDA 12

### Training Time Estimates

| Bar Frequency | Episodes/fold | Batches/epoch (bs=32) | Time/epoch | Epochs | Time/fold | Folds | Total |
|--------------|--------------|----------------------|-----------|--------|----------|-------|-------|
| 1-min | ~30,000 | ~937 | ~17 min | ~30 | ~8.5 hrs | ~45 | **16 days** |
| **5-min** | **~6,000** | **~187** | **~3.5 min** | **~40** | **~2.3 hrs** | **~20** | **~46 hrs** |
| 15-min | ~2,000 | ~62 | ~1.2 min | ~50 | ~1 hr | ~15 | **~15 hrs** |

| Frequency | Feasible on T4? | Estimated Time |
|-----------|----------------|---------------|
| 1-min bars | No (16 days) | Needs multi-GPU |
| **5-min bars** | **Yes** | **~20-24 hrs** |
| 15-min bars | Comfortable | ~8 hrs |

**Decision**: Train the intraday TFT on **5-min bars**. Captures microstructure
signals (VWAP deviation, order flow imbalance, spread) and fits on T4 in ~1 day.
Industry standard for intraday systematic trading.

### Training Time Optimizations

1. **5-min bars instead of 1-min** — 5x reduction, still captures intraday dynamics
2. **Fewer epochs** — more data per epoch means faster convergence (30-40 vs 80)
3. **Larger step_size** — fewer walk-forward folds (step=10d cuts folds in half)
4. **Train sequentially** — daily first (done), then intraday. Not simultaneous.
5. **Phase 2 on subset** — use only NIFTY+BANKNIFTY for VSN weight extraction

### VRAM Usage

| Phase | Daily TFT | Intraday TFT |
|-------|-----------|-------------|
| Training | ~1.4 GB | ~1.2-1.5 GB |
| Inference | ~50 MB | ~40 MB |
| Both inference | — | **~90 MB (0.6% of T4)** |

---

## 5. Inference with Two Models on 1 GPU

### Why It's Trivial

| Metric | Daily TFT | Intraday TFT | Both |
|--------|-----------|-------------|------|
| Model size (params) | ~500K (~2 MB) | ~400K (~1.6 MB) | **~3.6 MB** |
| VRAM per forward pass | ~50 MB | ~40 MB | **~90 MB** |
| Time per inference | <50ms | <50ms | **<100ms** |
| Frequency | Once/day at 3:25 PM | Every 5 min (75/day) | — |

Both models loaded simultaneously use **~90 MB**. That's 0.6% of the 15.4 GB T4.

### Production Inference Flow

```
Market Open (9:15 AM)
  │
  ├─ Load both models into GPU memory (~90 MB total)
  │  daily_model = TFTInferencePipeline.from_checkpoint("daily")
  │  intra_model = TFTInferencePipeline.from_checkpoint("intraday")
  │
  ├─ 9:15 AM: Daily TFT runs once
  │   → regime = daily_model.predict(today)
  │   → {NIFTY: +0.22, BANKNIFTY: -0.15, ...}
  │
  ├─ 9:20 AM: Intraday TFT first bar
  │   → signal = intra_model.predict(bar, regime=regime)  # <50ms
  │   → position decision
  │
  ├─ 9:25 AM: Intraday TFT second bar
  │   → signal = intra_model.predict(bar, regime=regime)  # <50ms
  │
  │  ... (75 bars/day, each <50ms)
  │
  └─ 3:25 PM: Last bar + EOD daily book trade
```

### Time Budget per 5-min Bar

```
Available time between bars:                     300,000ms (5 min)

  ├── Feature computation (MegaFeatureBuilder):  ~200-500ms
  ├── GPU forward pass:                          ~30-50ms
  ├── Position sizing + risk check:              ~5ms
  ├── Order submission (Kite API):               ~100-200ms
  └── Total:                                     ~400-800ms

Utilization:                                     <0.3%
```

### When 1 GPU Becomes a Problem

| Scenario | Feasible? |
|----------|-----------|
| Inference: both models on T4 | Trivially yes (90 MB) |
| Train daily + infer intraday simultaneously | Yes (1.4 GB + 40 MB) |
| Train both simultaneously | No (train sequentially) |
| Train intraday while daily is in production | Yes (1.4 GB + 90 MB) |
| Inference for 10 models simultaneously | Yes (<1 GB total) |

**Bottom line**: GPU inference is embarrassingly cheap — it's training that's
expensive. Even 10 models would use <1 GB on the T4.

---

## 6. Signal Combination for Trading

### Signal Interaction Matrix

```
                    Intraday TFT
                    Long    Flat    Short
Daily TFT  Long  │  +++   │  +    │  ±    │
           Flat  │  +     │  —    │  -    │
           Short │  ±     │  -    │  ---  │
```

### Method 1: Agreement Amplification (Simplest)

```python
position = w_d * daily_signal + w_i * intra_signal

# When both agree: positions ADD (high conviction)
# When they disagree: positions partially cancel (low conviction)
# w_d = 0.6, w_i = 0.4 (daily gets more weight — longer horizon, more stable)
```

**Problem**: Treats them as same-horizon signals, which they're not.

### Method 2: Daily Filter + Intraday Trigger (Most Common)

```python
if daily_signal > threshold:
    # Only take LONG intraday signals
    if intra_signal > 0: ENTER LONG (sized by intra confidence)
    if intra_signal < 0: SKIP (don't fight the daily trend)

if daily_signal < -threshold:
    # Only take SHORT intraday signals
    ...

if abs(daily_signal) < threshold:
    # Neutral regime — trade both directions but half size
```

This is what most institutional desks do. Daily conviction gates intraday execution.
Avoids the worst scenario — taking an intraday long into a daily downtrend.

### Method 3: Separate Books + Risk Manager (Most Robust — Recommended)

```
┌─────────────────────────────────────────────┐
│              Risk Manager                    │
│  Max net exposure: 30% per asset             │
│  Max gross exposure: 50% per asset           │
│  Correlation penalty if signals conflict     │
├──────────────────┬──────────────────────────┤
│  Daily Book       │  Intraday Book           │
│  Budget: 20%      │  Budget: 15%             │
│  Hold: 1-5 days   │  Hold: 30min-2hrs        │
│  Daily TFT signal │  Intraday TFT signal     │
│  Trade at 3:25 PM │  Trade every 5-min bar   │
└──────────────────┴──────────────────────────┘
```

Each TFT has its own position budget. Risk manager enforces net/gross limits.
When both are long NIFTY, net exposure is 35% (20+15). When they conflict, net
exposure is 5% (20-15) — natural hedging.

### Method 4: Confidence-Weighted Regime (Most Sophisticated)

```python
daily_regime = classify(daily_signal, daily_confidence)
  # → STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR

intra_sizing_multiplier = {
    STRONG_BULL: {long: 1.5, short: 0.3},
    BULL:        {long: 1.2, short: 0.5},
    NEUTRAL:     {long: 1.0, short: 1.0},
    BEAR:        {long: 0.5, short: 1.2},
    STRONG_BEAR: {long: 0.3, short: 1.5},
}

intra_position = intra_signal * intra_sizing_multiplier[daily_regime][direction]
```

The daily TFT doesn't directly trade — it modulates intraday sizing. Strong daily
conviction amplifies aligned intraday trades and suppresses opposing ones.

### Recommended Production Setup

**Method 3 (Separate Books)** with **Method 2 logic inside the intraday book**:

- Daily book trades independently at EOD (current TFT, S1-S25)
- Intraday book uses daily TFT regime as a directional filter
- Risk manager caps combined exposure
- No double-counting of alpha — each book has its own P&L attribution

Maps to existing architecture:
- `TFTStrategy` (daily) → already built
- `TFTIntradayStrategy` (5-min) → new, with daily regime as static covariate
- `RiskManager` in `core/risk/` → already exists, needs multi-book support

---

## 7. Implementation Roadmap

### Phase 1: Intraday Feature Pipeline
- 5-min bar aggregation from 1-min Kite data
- Intraday features: VWAP deviation, order flow imbalance, bid-ask spread,
  volume profile, realized volatility, opening range metrics
- ~50 features at 5-min resolution

### Phase 2: Intraday TFT Training Pipeline
- Adapt `TrainingPipeline` for 5-min bars
- Shorter walk-forward windows (train=20d, test=5d)
- Purge gap in bars (30-60 bars) instead of days
- Daily TFT regime as static covariate input
- Estimated training time: ~20-24 hrs on T4

### Phase 3: Dual-Model Inference
- Load both models into GPU at market open
- Daily model runs once at 9:15 AM
- Intraday model runs every 5-min bar with regime context
- `TFTIntradayStrategy` adapter for strategy registry

### Phase 4: Multi-Book Risk Manager
- Separate position budgets for daily and intraday books
- Net/gross exposure limits per asset
- Directional filter (Method 2) inside intraday book
- P&L attribution per book

---

## 8. Open Questions

1. **5-min vs 15-min**: 5-min gives 3x more data but 3x longer training.
   Start with 5-min, can always downsample if too slow.

2. **Intraday feature set**: Which tick-derived features matter most at 5-min?
   VSN will tell us (2-pass approach), but initial candidates:
   - VWAP deviation, volume profile, order flow imbalance
   - Bid-ask spread (from Kite depth), tick count
   - Realized volatility (1-min), opening range break
   - Daily TFT regime (static covariate)

3. **Target definition**: Next-30min return? Next-bar (5-min) return?
   Longer targets are more stable but reduce trade frequency.

4. **Overnight handling**: Intraday model only runs 9:15-3:30 IST.
   Daily model handles overnight exposure decisions.

5. **Crypto extension**: Crypto markets are 24/7 — intraday TFT for BTC/ETH
   could run continuously. Different walk-forward calendar needed.
