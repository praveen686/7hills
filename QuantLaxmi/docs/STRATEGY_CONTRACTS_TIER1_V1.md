# BRAHMASTRA Tier-1 Strategy Contracts V1

**Agent B — Strategy Contracts Analyst**
**Date**: 2026-02-07
**Classification**: Production Reference
**Confidence**: High (all parameters extracted from source code)

---

## Executive Summary

This document defines **Strategy Contracts** for all 4 Tier-1 BRAHMASTRA strategies. Each contract has 8 sections: Data, Feature, Decision, Risk, Execution, Explainability, Replay, and Failure Modes.

| # | Strategy | Sharpe | Instrument | Warmup | Data Source |
|---|----------|--------|------------|--------|-------------|
| S5 | Hawkes Microstructure | 4.29 | FUT | 5d | nfo_1min (intraday) |
| S1 | VRP-RNDR | 1.67/5.59 | FUT/SPREAD | 55d | nse_fo_bhavcopy (EOD) |
| S4 | IV Mean-Reversion | 3.07 | FUT | 35d | nse_fo_bhavcopy (EOD) |
| S7 | Regime Switch | 2.37 | FUT | 120d | nse_index_close (EOD) |

---

# S5: HAWKES MICROSTRUCTURE STRATEGY

**File**: `strategies/s5_hawkes/strategy.py` (Sharpe 4.29)
**Entry Point**: `class S5HawkesStrategy(BaseStrategy)` at line 24

## 1. Data Contract

### Source Tables

| Table | Columns | Frequency | Staleness |
|-------|---------|-----------|-----------|
| `nfo_1min` | date, strike, instrument_type, open, high, low, close, volume, oi, symbol, expiry | Intraday (1-min) | Same-day |

**Query** (analytics.py:163-178): Filters by date, name (NIFTY/BANKNIFTY), instrument_type IN ('CE', 'PE'), orders by strike.

**Symbols**: `["NIFTY", "BANKNIFTY"]` (strategy.py:21)

## 2. Feature Contract

**Warmup**: 5 days (strategy.py:47) — EMA state initialization

### Components (signals.py weights at line 155-161)

| Component | Weight | Max Score | Null Handling |
|-----------|--------|-----------|---------------|
| GEX (Gamma Exposure) | 0.30 | +/-1.0 | Returns 0 if regime neutral |
| OI Flow | 0.25 | +/-1.0 | Returns 0 if snap.oi_flow is None |
| IV Term Structure | 0.20 | +/-1.0 | Returns 0 if slope normal |
| Basis | 0.15 | +/-1.0 | Returns 0 if zscore < 2 |
| PCR (Put-Call Ratio) | 0.10 | +/-1.0 | Returns 0 if neutral |

### EMA State (signals.py:57-61)

```
_ema_state: dict[str, float]  — symbol -> smoothed combined score
EMA_ALPHA = 0.3
MIN_CONSECUTIVE = 2  — require 2 consecutive raw signals before entry
```

**Computation** (signals.py:196-201):
```python
if sym not in _ema_state:
    smoothed = raw_combined  # first observation seeds EMA
else:
    smoothed = 0.3 * raw_combined + 0.7 * _ema_state[sym]
```

## 3. Decision Contract

**Entry** (signals.py:224-226):
- Condition: `abs(smoothed) >= entry_threshold` AND `consecutive >= MIN_CONSECUTIVE`
- Default entry_threshold: 0.25 (strategy.py:36)
- Direction: `"long" if smoothed > 0 else "short"`
- Conviction: `min(1.0, abs(smoothed) / 0.5)` — range [0, 1]

**Exit Reasons**:
- `"signal_flat"`: Smoothed score drops below EXIT_THRESHOLD (0.15) for 5 consecutive scans
- `"direction_flip"`: Direction reversal (exit as FLAT, re-enter next scan)

**Metadata** (strategy.py:132-138):
```
gex_regime: "mean_revert" | "momentum"
raw_score, smoothed_score: float
reasoning: human-readable string
```
Note: Component breakdown (gex, oi_flow, iv_term, basis, pcr) is not
currently emitted; optional future enhancement.

## 4. Risk Contract

- **GEX > 0** -> mean_revert regime (dealers long gamma, dampening)
- **GEX < 0** -> momentum regime (dealers short gamma, amplifying)
- Conviction [0, 1] directly = position size
- Kill Switches: VPIN > 0.70 blocks all entries, Portfolio DD / Strategy DD circuit breaker

## 5. Execution Contract

- **Instrument**: FUT only (strategy.py:90, 114, 130)
- **Cost**: ~5 bps round-trip
- **Settlement**: T+1 (Indian FnO standard)
- **Multi-leg**: None

## 6. Explainability Contract (Why Panel)

Minimum fields for operator display:
```
signal_date, symbol, direction, conviction
gex_regime, raw_score, smoothed_score, ema_alpha, consecutive_scans
reasoning: string
spot, futures, max_pain_strike, nearest_expiry
```
Note: Component breakdown (gex, oi_flow, iv_term, basis, pcr with weights)
is not currently emitted in the event stream. The emitter produces
`gex_regime`, `raw_score`, `smoothed_score`, and `reasoning` as the
explainability surface. Component-level attribution is an optional future
enhancement.

## 7. Replay Contract

Per-scan event log must record:
1. Option chain snapshot (per-strike OI, IV, delta, gamma)
2. GEX (net, call-side, put-side, flip-level)
3. OI flow (delta-weighted change)
4. Basis, PCR, IV term structure
5. EMA state (_ema_state, _direction_streak)
6. Entry/exit decision + conviction

**Determinism**: Requires persisted EMA state; call `reset_signal_state()` (signals.py:66-71) between CV folds.

## 8. Failure Modes

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Missing chain | signal=None | Data quality gate |
| Stale IV | IV locked previous day | Require same-day close |
| Illiquid OTM | GEX spike on tiny OI | Min OI filter |
| Fast vol spike | GEX flip jumps | EMA smoothing absorbs |
| VPIN > 0.70 | EXTREME regime | Kill-switch blocks entries |

---

# S1: VRP-RNDR (RISK-NEUTRAL DENSITY REGIME)

**File**: `strategies/s1_vrp/strategy.py` (Sharpe futures 1.67, options 5.59)
**Entry Point**: `class S1VRPStrategy(BaseStrategy)` at line 24

## 1. Data Contract

### Source Tables

| Table | Columns | Frequency | Staleness |
|-------|---------|-----------|-----------|
| `nse_fo_bhavcopy` | date, TckrSymb, FinInstrmTp, OptnTp, XpryDt, StrkPric, ClsPric, OpnIntrst, UndrlygPric | Daily close | T+0 EOD |

**Query** (density.py:108-152): `prepare_nifty_chain()` filters by symbol, FinInstrmTp="IDO", max_expiries=2.

**Symbols**: `["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]` (strategy.py:33)

## 2. Feature Contract

**Warmup**: `lookback + phys_window + 5 = 30 + 20 + 5 = 55` days (strategy.py:67)

### SANOS Calibration Parameters (sanos.py:154-364)
- eta = 0.50 (smoothness)
- n_model_strikes = 100
- K_min = 0.7, K_max = 1.5
- LP solver: highs, 60s timeout

### Composite Signal Components (density.py:274-320)

| Component | Weight | Computation | Missing Handling |
|-----------|--------|-------------|------------------|
| Skew Premium | 0.40 | phys_skew - rn_skew (percentile rank) | 0 if < 20 returns |
| Left Tail | 0.25 | P(K < mu - sigma) (percentile rank) | 0 if calibration fails |
| Entropy Delta | 0.20 | H_today - H_yesterday (z-score) | 0 if no previous |
| KL Direction | 0.15 | D_KL * sign(delta_skew) (z-score) | 0 if density mismatch |

### State Persistence (strategy.py:103-162)
```
density_history_{symbol}: last 200 days of [skew_premium, left_tail, entropy_change, kl_div, atm_iv]
spots_{symbol}: last 200 days of spot prices
composites_{symbol}: last 200 composite scores
prev_density_{symbol}: {q: list[float], entropy: float}
```

## 3. Decision Contract

**Entry** (strategy.py:243-265):
```python
composite = (
    0.40 * (2*skew_pctile - 1) + 0.25 * (2*lt_pctile - 1)
    - 0.20 * (ent_z / 3.0) + 0.15 * (kl_z * skew_dir / 3.0)
)
sig_pctile = percentile_rank(composites[-lookback:], composite)
if sig_pctile >= entry_pctile and composite > 0:
    conviction = min(1.0, abs(composite) / 0.5)
```
- entry_pctile = 0.75, exit_pctile = 0.40, hold_days = 5 (strategy.py:49-51)

**Exit**: max_hold OR sig_pctile < exit_pctile (strategy.py:226)

## 4. Risk Contract

**Options Variant** (options.py):
- SHORT_OFFSET = 0.03 (3% OTM), LONG_OFFSET = 0.06 (6% OTM)
- strike_width - net_credit = max_risk (defined-risk spread)
- MIN_OI = 100, MIN_PREMIUM = 0.50 Rs
- OPTION_COST_BPS = 20.0 (4-leg cost)

Kill Switches: VPIN > 0.70, Portfolio DD, Strategy DD

## 5. Execution Contract

- **Futures**: FUT only, 5 bps cost
- **Options**: SPREAD (bull put credit spreads), 20 bps cost
- **Settlement**: T+1

## 6. Explainability Contract (Why Panel)

Minimum fields:
```
signal_date, symbol, direction, conviction
lookback_days, components: {skew_pctile, left_tail_pctile, entropy_z, kl_z}
composite, composite_percentile, entry_threshold
physical_skewness, rn_skewness, skew_premium, left_tail, atm_iv
spot, nearest_expiry, forward
```

## 7. Replay Contract

Per-date event log:
1. F&O bhavcopy (per-strike closes, OI)
2. SANOS result (densities, fit errors, LP success/failure)
3. Extracted density (RN skew/kurt, entropy, tail weights)
4. Physical skewness (trailing 20+ days)
5. KL divergence vs yesterday, entropy change
6. Composite score, percentile rank
7. Entry/exit decision

**Determinism**: Requires state persistence; reset history between backtests. LP solver tolerance must be pinned.

## 8. Failure Modes

| Failure | Mitigation |
|---------|------------|
| SANOS LP timeout (>60s) | Skip day (density_ok=False) |
| LP infeasible | Use uniform density fallback (sanos.py:333-337) |
| < 5 OTM strikes | Skip symbol |
| Variance not monotonic | Enforce sigma^2*T increasing (sanos.py:205-207) |
| Spot gap > 5% | Use previous q as prior |

---

# S4: IV MEAN-REVERSION

**File**: `strategies/s4_iv_mr/strategy.py` (Sharpe 3.07)
**Entry Point**: `class S4IVMeanRevertStrategy(BaseStrategy)` at line 24

## 1. Data Contract

| Table | Columns | Frequency |
|-------|---------|-----------|
| `nse_fo_bhavcopy` | Same as S1 | Daily close |

**Symbols**: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY (strategy.py:25)

## 2. Feature Contract

**Warmup**: 30 + 5 = 35 days (strategy.py:56)

**Single Feature**: ATM IV from SANOS surface (engine.py:85-172)
```python
atm_strike = np.array([1.0])  # K/F = 1.0
T = (expiry_date - d).days / 365.0
iv_arr = result.iv(0, atm_strike, T)
atm_iv = float(iv_arr[0])
```

**Fallback** (engine.py:126): Brenner estimate = sqrt(atm_vars[0])

**State**: `iv_history_{symbol}`: last 200 days of {date, atm_iv, spot, forward}

## 3. Decision Contract

**Entry** (strategy.py:164-194):
```python
pctile = rolling_percentile(ivs, window=lookback)[-1]
if pctile >= entry_pctile:  # default 0.80
    size_weight = 0.25 + 0.75 * (pctile - 0.80) / 0.20
    conviction = size_weight  # range [0.25, 1.0]
```

**Exit**: hold >= hold_days (default 5) OR pctile < exit_pctile (default 0.50)

## 4. Risk Contract

- Conviction = size_weight directly scales position
- Kill Switches: VPIN, portfolio DD, strategy DD

## 5. Execution Contract

- **Instrument**: FUT, 5 bps cost, T+1 settlement

## 6. Explainability Contract (Why Panel)

```
signal_date, symbol, direction, conviction
iv_lookback, atm_iv, iv_percentile, entry_threshold, exit_threshold, hold_days
spot, forward, nearest_expiry
iv_history: trailing 30 days
```

## 7. Replay Contract

Per-date: F&O bhavcopy -> SANOS result -> ATM IV -> IV history -> rolling percentile -> decision

**Determinism**: IV history persisted; no randomness. But SANOS LP solver needs pinned tolerance.

## 8. Failure Modes

| Failure | Mitigation |
|---------|------------|
| SANOS fails | Use Brenner fallback |
| < 5 strikes | Skip symbol |
| IV locked | Require same-day close |
| IV > 1.0 | Clamp in bisection [0.001, 3.0] |

---

# S7: REGIME SWITCH

**File**: `strategies/s7_regime/strategy.py` (Sharpe 2.37)
**Entry Point**: `class S7RegimeSwitchStrategy(BaseStrategy)` at line 24

## 1. Data Contract

| Table | Columns | Frequency |
|-------|---------|-----------|
| `nse_index_close` | date, "Index Name", "Closing Index Value" | Daily close |

**Symbols**: `["NIFTY", "BANKNIFTY"]` (strategy.py:34)

## 2. Feature Contract

**Warmup**: 100 + 20 = 120 days (strategy.py:57)

### Regime Classes (detector.py:25-28)
```
TRENDING      — ent < 0.65 AND mi > 0.10
MEAN_REVERTING — 0.65 <= ent <= 0.85
RANDOM        — ent > 0.85 OR mi < 0.03 OR vpin > 0.70
```

### Thresholds (detector.py:43-47)
```
ENTROPY_LOW = 0.65, ENTROPY_HIGH = 0.85
MI_LOW = 0.03, MI_HIGH = 0.10
VPIN_TOXIC = 0.70
```

### Components
| Component | Range | Source |
|-----------|-------|--------|
| Shannon Entropy | [0, 1] | price_entropy() — binary word encoding |
| Mutual Info | nats | mutual_information() — lag-1 predictability |
| VPIN | [0, 1] | BVC (Bulk Volume Classification) |

## 3. Decision Contract

### TRENDING (strategy.py:178-257)
- SuperTrend (period=14, mult=3.0) + RSI (period=14) confirmation
- Long: price > lower_band AND rsi in [40, 70] -> conviction = confidence * 0.8
- Short: price < upper_band AND rsi in [30, 60] -> conviction = confidence * 0.8
- TTL: 10 bars

### MEAN_REVERTING (strategy.py:259-331)
- Bollinger Bands (window=20, 2-std)
- Long: z < -2.0 -> conviction = abs(z) / 3.0 * confidence
- Short: z > 2.0 -> conviction = abs(z) / 3.0 * confidence
- Exit: abs(z) < 0.5 (reverted to mean)
- TTL: 5 bars

### RANDOM (strategy.py:139-151)
- No directional edge -> exit any open position
- Block all new entries

### VPIN Kill-Switch (strategy.py:118-131)
- VPIN > 0.70 -> emergency exit + block entries

## 4. Risk Contract

- TRENDING confidence < 0.5 -> conviction scales down
- MEAN_REVERTING confidence < 0.3 -> no signal
- RANDOM -> always reject entries (after VPIN check)
- Kill Switches: VPIN > 0.70, Regime = RANDOM, Portfolio/Strategy DD

## 5. Execution Contract

- **Instrument**: FUT, 5 bps cost, T+1 settlement, no multi-leg

## 6. Explainability Contract (Why Panel)

```
signal_date, symbol, regime_analysis: {entropy, MI, VPIN, classified_regime, confidence}
sub_strategy: "trend_following" | "mean_reversion" | "random_exit"
direction, conviction, entry_reason
technical_indicators: {SuperTrend, RSI, Bollinger Bands}
z_score, pct_b, entry_price
```

## 7. Replay Contract

Per-date log:
1. Closing price, price history (100+ days)
2. Shannon entropy, mutual information, VPIN
3. Regime classification + confidence
4. Technical indicators (SuperTrend, RSI, Bollinger Bands)
5. Entry/exit decision

**Determinism**: No randomness; entropy/MI deterministic. Floating-point: use np.testing.assert_allclose with rtol=1e-6.

## 8. Failure Modes

| Failure | Mitigation |
|---------|------------|
| < 100 days history | Skip until warmup complete |
| Entropy NaN | Default RANDOM |
| RSI infinite | Clamp losses >= 1e-8 |
| ATR zero | Use 1% of price default |
| VPIN NaN | Return 0.0 |
| Regime oscillation | 100-bar window smooths |

---

# CROSS-STRATEGY COMPARISON

| Aspect | S5 Hawkes | S1 VRP | S4 IV MR | S7 Regime |
|--------|-----------|--------|----------|-----------|
| **Data Freshness** | Intraday | EOD | EOD | EOD |
| **Warmup Days** | 5 | 55 | 35 | 120 |
| **Instruments** | FUT | FUT/SPREAD | FUT | FUT |
| **Conviction Range** | [0, 1] | [0, 1] | [0.25, 1] | [0, 1] |
| **TTL Bars** | 5 | 5 | 5 | 10/5 |
| **Cost (bps)** | 5 | 5/20 | 5 | 5 |
| **Sharpe** | 4.29 | 1.67/5.59 | 3.07 | 2.37 |
| **Kill Switches** | VPIN, DD | VPIN, DD | VPIN, DD | VPIN, DD, Regime |

### Universal Kill-Switches
1. **VPIN > 0.70** (allocator/regime.py): Block all new entries
2. **Portfolio DD > max_dd**: Circuit breaker (risk/manager.py)
3. **Strategy DD > max_dd**: Per-strategy limit (risk/manager.py)
4. **Concentration limit**: Max single instrument (risk/manager.py)

---

**Prepared by**: Agent B (Strategy Contracts Analyst)
**All parameters extracted from source code with file:line citations.**
