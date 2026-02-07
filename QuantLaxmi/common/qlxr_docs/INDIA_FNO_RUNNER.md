# India F&O Strategy Runner

## Infrastructure

| Component | Spec | Strategic Edge |
|-----------|------|----------------|
| CPU | Xeon Platinum 8259CL, 16C/32T, AVX-512 | Vectorized Greeks across 1000s of strikes |
| GPU | Tesla T4, 15.7GB VRAM, Turing tensor cores | 65 TFLOPS FP16 — real-time vol surface fitting, ML inference |
| RAM | 124GB (119GB free) | Entire option chain history in-memory |
| Disk | 133GB free | Years of tick-level parquet storage |
| Network | **~1ms to NSE** (AWS ap-south-1) | Co-located with NSE infrastructure |
| ML Stack | PyTorch 2.10 CUDA, XGBoost 3.1, FinBERT loaded | GPU inference pipeline already warm |

---

## Existing Assets

- **NSE Bhavcopy downloader**: delivery, F&O, FII/DII — 8 months cached (2025-06-01 → 2026-02-03)
- **F&O OI data**: already parsing nearest-expiry stock futures OI
- **FinBERT pipeline**: running on T4 for crypto news, extensible to Indian news
- **Signal framework**: delivery spike, OI quadrant, FII flow — tested but directional edge is weak (Sharpe -3.2)
- **Paper trading framework**: state persistence, cost model (STT + brokerage + charges ≈ 25-30bps RT)
- **62 passing tests** for the India scanner module

---

## Strategy Pipeline

### S1: Volatility Risk Premium (VRP) Harvesting — PRIORITY #1
- **Status**: NOT STARTED
- **Edge**: IV systematically overprices RV by 3-8 vol points in NIFTY. Retail is net option buyer. We sell premium, delta-hedge, collect the spread.
- **Approach**:
  1. Download historical NIFTY option chain (all strikes, daily) from NSE archives
  2. Compute IV surface per day (GPU Black-Scholes inversion)
  3. Compute RV using HAR-RV model (1d, 5d, 22d components)
  4. Backtest: sell ATM straddle when IV-RV > threshold, delta-hedge with futures
  5. Paper trade with Zerodha/simulated fills
- **Data needed**: Historical option chain (strike-level OHLCV + OI) — check if F&O bhavcopy has this
- **GPU use**: SABR/SVI surface fitting, Greeks computation, Monte Carlo tail risk
- **Expected Sharpe**: 1.5–2.5

### S2: Intraday Microstructure (GEX + OI Flow + Basis + PCR + IV Term)
- **Status**: IN PROGRESS — collecting data + paper trading (2026-02-05)
- **Edge**: 5-signal microstructure composite — dealer gamma mechanics (GEX), institutional OI flow, panic/complacency (IV term structure), leverage positioning (basis), sentiment extremes (PCR)
- **Approach**:
  1. Collect full option chain snapshots every 3 min via Zerodha Kite API (NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY)
  2. GPU-accelerated bisection IV + BS Greeks across all strikes
  3. Compute GEX regime (mean-revert vs momentum), GEX flip level
  4. Track delta-weighted OI changes between snapshots (institutional flow)
  5. Combined weighted signal (GEX 30%, OI flow 25%, IV term 20%, basis 15%, PCR 10%)
  6. Paper trade NIFTY/BANKNIFTY futures, 2h max hold, 5bps cost
- **Data needed**: Live option chain (Kite API) — collecting since 2026-02-05 09:32 IST
- **GPU use**: Bisection IV + BS Greeks for 1400+ contracts per cycle
- **Files**: `apps/india_microstructure/` (auth.py, collector.py, analytics.py, signals.py, paper_state.py)
- **Expected Sharpe**: TBD (collecting data)

### S3: NLP Event Straddle (extends existing FinBERT)
- **Status**: PARTIAL — India news sentiment scanner DONE (Step 13), straddle buying NOT STARTED
- **Edge**: Detect corporate catalysts from Indian news before IV expands. Buy straddles cheap, sell into IV crush post-event.
- **What's done** (Step 13):
  1. ✅ 8 Indian RSS feeds (ET, Hindu BL, LiveMint, BS, Moneycontrol) — `apps/india_news/scraper.py`
  2. ✅ Entity extraction: 365 aliases → 155 F&O stocks — `apps/india_news/entity.py`
  3. ✅ FinBERT sentiment + event-type classifier (earnings, regulatory, macro, corporate)
  4. ✅ Stock futures paper trader (directional, 3d TTL, 30bps cost) — live with 5 positions
- **What's remaining** (straddle variant):
  - [ ] Buy ATM straddles when catalyst detected AND per-stock IV percentile < 50
  - [ ] Exit day after event (profit from IV crush OR directional move)
  - [ ] Backtest using historical earnings dates + F&O bhavcopy
  - [ ] Needs per-stock IV percentile rank (per-stock SANOS or per-stock IV tracking)
- **GPU use**: FinBERT batch inference (already running)
- **Expected Sharpe**: 0.8–1.5

### S4: FII Flow Momentum with Options Leverage
- **Status**: NOT STARTED (data already collected)
- **Edge**: FII flows persist for days. Trade NIFTY options for leveraged directional exposure.
- **Approach**:
  1. 3-day cumulative FII net flow (already have fii_stats in parquet)
  2. Flow > +5000 Cr → buy NIFTY OTM weekly calls
  3. Flow < -5000 Cr → buy NIFTY OTM weekly puts
  4. Fixed TTL (2 days) or flow reversal exit
  5. Small sizing (2-3% per trade), let option leverage do the work
- **Data needed**: Already have it
- **GPU use**: Light (XGBoost for threshold optimization)
- **Expected Sharpe**: 0.5–1.2

### S5: Volatility Surface Arbitrage
- **Status**: NOT STARTED
- **Edge**: Retail flow creates local IV dislocations. Options misprice vs smooth surface.
- **Approach**:
  1. Fit SVI parameterization to option chain on GPU
  2. Identify >1 vol point deviations from surface
  3. Sell overpriced, buy underpriced, delta-hedge
  4. Profit from intraday convergence
- **Data needed**: Tick-level option chain (harder to source)
- **GPU use**: Non-linear least-squares (Levenberg-Marquardt) every tick
- **Expected Sharpe**: 1.0–1.8

### S6: Cross-Expiry Calendar Theta
- **Status**: NOT STARTED
- **Edge**: Weekly options lose ~50% premium in last 2 days. Monthly decays linearly. Sell weekly, buy monthly.
- **Approach**:
  1. Sell weekly ATM strangle (Thursday expiry)
  2. Buy monthly ATM strangle as hedge
  3. Roll weekly — net theta positive, vega partially hedged
- **Data needed**: Multi-expiry option chain
- **GPU use**: Strike optimization via P&L simulation
- **Expected Sharpe**: 0.8–1.2

### S7: Intraday Vol Regime Classifier
- **Status**: NOT STARTED
- **Edge**: Small Transformer classifies NIFTY regime (trending/mean-reverting/breakout) in real-time.
- **Approach**:
  1. Train on 1-min NIFTY bars (returns, volume, OI, VIX, PCR)
  2. Trending → sell directional spreads; Mean-reverting → sell iron condors; Breakout → buy straddles
  3. Inference every 60s on T4
- **Data needed**: Historical 1-min bars (Zerodha Kite / NSE)
- **GPU use**: Training + real-time inference
- **Expected Sharpe**: 0.5–1.0

---

## Completed Work Log

| Date | What | Result |
|------|------|--------|
| 2026-02-04 | Backfilled NSE bhavcopy data (delivery, F&O, FII) | 167 days, 2025-06-01 → 2026-02-03 |
| 2026-02-04 | Ran institutional footprint backtest (8 months) | Sharpe -3.2, 42% win rate — directional signals don't work |
| 2026-02-04 | All 62 tests passing | bhavcopy, signals, scanner modules verified |
| 2026-02-04 | Built GPU IV engine (`apps/india_fno/iv_engine.py`) | 1,379/1,598 contracts solved in 1.8s, ATM IV 18.6%, put-call parity perfect |
| 2026-02-04 | VRP backtest v1: weekly ATM straddle selling | **Sharpe -8.4, 30% win rate** — spot moves eat the premium |
| 2026-02-04 | Premium vs move analysis | Weekly strangle premium (0.31%) << median 5d move (0.97%). Only 22% win rate for strangle buyers needing move > premium |
| 2026-02-04 | FII flow signal test | Contrarian not momentum. Heavy FII selling → market UP (+0.23%) |
| 2026-02-04 | Max pain pinning test | 0.17% distance on ALL days, 45% convergence — no signal |
| 2026-02-04 | Weekly strangle selling (all deltas) | 88% win rate but -2.09 to -6.96 Sharpe. Pennies in front of steamroller |
| 2026-02-04 | SANOS vol surface fitting (`apps/india_fno/sanos.py`) | 15/15 LP convergence, 0.62s avg, zero arb violations, 0.62% mean IV error |
| 2026-02-04 | IV Mean-Reversion strategy (`apps/india_fno/iv_mean_revert.py`) | **Sharpe 3.8-4.0**, 15% ann, 3.5% max DD, 16 trades, 75% win rate, all 36 configs profitable |
| 2026-02-04 | Entropy + MI features (`qlx/features/information.py`) | Ported from Timothy Masters' C++ library. Tested but useless for NIFTY daily regime filtering (entropy ~1.0, MI ~0 uniformly) |
| 2026-02-04 | IV Mean-Reversion paper trader (`apps/india_fno/paper_state.py`, `__main__.py`) | CLI with scan/backtest/paper/status/backfill. Backfill+simulate: 267 days, 22 signals, 21 trades, +12.31%, 57% win. Trades match standalone backtest exactly. 152 tests passing. |
| 2026-02-05 | Zerodha headless auto-login (`apps/india_microstructure/auth.py`) | Fully automated OAuth + TOTP login, session token caching, browser-free. Tested on EC2. |
| 2026-02-05 | Intraday option chain collector (`apps/india_microstructure/collector.py`) | 4 indices (NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY), 488+374+366+242 contracts per snapshot, 3-min interval, parquet storage |
| 2026-02-05 | Microstructure analytics engine (`apps/india_microstructure/analytics.py`) | GPU-accelerated: GEX regime + flip level, delta-weighted OI flow, max pain, futures basis (z-score), PCR, IV term structure |
| 2026-02-05 | Microstructure paper trader (`apps/india_microstructure/`) | 5-signal weighted composite (GEX/OI/IV/basis/PCR), paper trades NIFTY+BN futures, 2h max hold, 5bps cost, auto-close at market end. Collecting data since 09:32 IST. |
| 2026-02-05 | Multi-index backtest (NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY, NIFTYNXT50) | BANKNIFTY Sharpe 5.39 (+18.5%), MIDCPNIFTY 4.19 (+24.2%), NIFTY 4.07 (+14.8%), FINNIFTY 3.36 (+14.2%). NIFTYNXT50 excluded (Sharpe -0.66, illiquid options with 40-180% IV noise). |
| 2026-02-05 | Multi-index paper trader upgrade | `MultiIndexPaperState` trades 4 indices simultaneously. Equal-weight 1/N sizing. Parallelized backfill (4 ProcessPoolExecutor workers). Legacy state auto-migration. Per-index IV sparklines, trade breakdown, position dashboard. |
| 2026-02-05 | Microstructure v1 results: churning diagnosed | 74 scans, 14 entries, 7% win rate, -0.621% P&L. All exits via conviction_drop. OI flow flips ±1.0 between 3-min snapshots → noisy combined score → rapid entry/exit cycles. |
| 2026-02-05 | Microstructure v2: anti-churn tuning | EMA smoothing (α=0.3), entry threshold 0.25→0.40, consecutive≥2 raw signals, 15m min hold, conviction_drop→signal_decay (5 flat scans). Per-symbol basis history fix. |
| 2026-02-05 | Fixed multi-index backfill state overwrite | Previous `backfill` (no `--simulate`) had overwritten 76-trade state with migrated NIFTY-only state. Re-ran with `--simulate`: 76 trades, +16.54%, 63% WR confirmed across all 4 indices. |
| 2026-02-05 | India News Sentiment Scanner (`apps/india_news/`) | 8 RSS feeds, 365 entity aliases, FinBERT scoring, event-type weights, paper trader. First live scan: 202 headlines, 52 stock mentions, 5 signals, 5 positions entered. 53 tests passing. |
| 2026-02-05 | RNDR Density Strategy — futures variant (`density_strategy.py`) | Breeden-Litzenberger density from SANOS → 4-component composite signal (skew premium, left tail, entropy, KL). BANKNIFTY +5.4% Sharpe 1.89, FINNIFTY +6.6% Sharpe 2.62, MIDCPNIFTY +4.8% Sharpe 0.96. 49 tests (risk_neutral + density_strategy). |
| 2026-02-05 | RNDR Options Variant — bull put spreads (`density_options.py`) | Same signal → sell 3%/6% OTM put spread. BANKNIFTY +54% RoR Sharpe 5.34, FINNIFTY +76% Sharpe 10.95, MIDCPNIFTY +120% Sharpe 4.69. Aggregate +495% on risk capital. 25 tests. Total suite: 426 tests. |

---

## Key Findings

### NIFTY Volatility Surface (Jan 30, 2026)
- ATM IV (weekly): **18.6%**, put-call parity verified ±0.1%
- IV term structure: 18.6% (4d) → 15.0% (11d) → 12.0% (59d) → 11.5% (151d)
- Put skew: 19.6% at 25Δ put vs 18.4% at 25Δ call
- Forward computed from put-call parity: 25,343 vs spot 25,320

### VRP Analysis (Jul 2025 → Jan 2026)
- **Avg IV: 9.9%** | **Avg RV: 8.1%** | **Avg VRP: 1.8%**
- VRP is positive (IV > RV) most of the time → structural premium exists
- BUT: 1.8% annualized VRP ≈ 0.25% per week — not enough to cover costs + gamma
- Weekly straddle selling: premium collected 0.5-1.5% but spot moves 0.2-2.6%
- **Conclusion: blind ATM straddle selling is not profitable. Need conditional entry.**

### Premium vs Spot Move Analysis (THE critical finding)

| Metric | Weekly (1-7d) | Monthly (20-40d) |
|--------|--------------|------------------|
| Straddle premium | 0.82% of spot | 2.01% of spot |
| Strangle (25Δ) premium | 0.31% of spot | 0.95% of spot |
| Median 5d spot move | 0.97% | — |
| Mean 5d spot move | 1.02% | — |
| 90th pct 5d move | 2.10% | — |
| Strangle win rate (weekly) | **22%** | — |

**Key insight: NIFTY weekly option premiums are TOO SMALL relative to realized moves.**
- Weekly strangle (25Δ) at 0.31% gets breached 78% of the time
- Weekly straddle at 0.82% loses when move > 0.82% (happens ~55% of time)
- The VRP of 1.8% annualized = 0.25% per week → barely covers costs (30bps RT)
- **Selling weekly NIFTY options is a losing game in this regime**

**Monthly options are more promising**: 2.01% straddle and 0.95% strangle premiums
give more room for theta decay before gamma eats the position.

### Weekly Strangle Selling by Delta (8 weeks, Mon→Thu)

| Delta | Width | Prem% | Win% | Avg Net P&L | Sharpe |
|-------|-------|-------|------|-------------|--------|
| 10Δ | 3.0% | 0.12% | 88% | -0.28% | -6.96 |
| 15Δ | 2.4% | 0.18% | 88% | -0.25% | -4.81 |
| 20Δ | 1.8% | 0.25% | 88% | -0.20% | -3.14 |
| 25Δ | 1.5% | 0.32% | 88% | -0.16% | -2.09 |

**Classic "picking up pennies in front of a steamroller"**: 88% win rate but
the 1 losing week (-1.0% to -1.5%) wipes all gains. Costs (30bps RT including
STT at 0.0625% of notional) eat the premium even on winning trades.

### FII Flow Signal — DOES NOT PREDICT DIRECTION

| Threshold | Bullish Signals | 5d Return | Bear Signals | 5d Return |
|-----------|----------------|-----------|--------------|-----------|
| ±1000 Cr | 52 | +0.07% | 85 | +0.13% |
| ±3000 Cr | 31 | -0.05% | 68 | +0.09% |
| ±5000 Cr | 18 | **-0.41%** | 49 | **+0.23%** |

FII flow is actually **contrarian** — heavy FII selling → market recovers (DII/retail absorbs).
Not usable as momentum signal.

### Max Pain Pinning — NOT TRADEABLE
- Spot within 0.17% of max pain on BOTH expiry and non-expiry days
- Max pain simply tracks spot (OI concentrates at ATM)
- Convergence toward max pain: only 45% (no better than random)

### IV Mean-Reversion (SANOS ATM IV → Long Index Futures)

**Core insight**: When ATM IV spikes above its rolling 80th percentile, the index is in a fear regime. IV mean-reverts → spot recovers → go long futures.

#### Single-Index Parameter Sweep (NIFTY)

| Config | Lookback | Entry Pctl | Max Hold | Sharpe | Ann Return | Max DD |
|--------|----------|-----------|----------|--------|------------|--------|
| Best | 30d | 0.80 | 5d | **3.99** | 15.1% | 3.5% |
| Runner-up | 30d | 0.75 | 7d | **3.82** | 14.8% | 3.5% |
| Worst | 60d | 0.80 | 10d | 1.29 | 5.2% | 4.1% |

- **All 36 parameter combinations profitable** (Sharpe range: 1.29-3.99)
- 16 trades over 13 months, 75% win rate, +12.4% total return
- Best single trade: April 2025 India-Pakistan crisis, 46.9% IV spike → +5.22% in 4 days
- Even excluding the crisis outlier, remaining 15 trades are still net positive
- Cost assumption: 5bps round-trip (futures only, no STT on index futures)

#### Multi-Index Backtest (Jan 2025 → Feb 2026, lookback=30, entry=0.80, hold=5d)

| Symbol | Days | Trades | Return | Sharpe | Max DD | Win Rate |
|--------|------|--------|--------|--------|--------|----------|
| BANKNIFTY | 268 | 16 | **+18.55%** | **5.39** | 3.23% | 68.8% |
| MIDCPNIFTY | 268 | 21 | **+24.25%** | **4.19** | 4.48% | 76.2% |
| NIFTY | 268 | 22 | +14.77% | 4.07 | 3.61% | 59.1% |
| FINNIFTY | 268 | 20 | +14.17% | 3.36 | 3.64% | 55.0% |
| NIFTYNXT50 | 268 | 18 | -2.80% | -0.66 | 15.96% | 27.8% |

- **4 out of 5 indices profitable** with Sharpe > 3. Average +17.9% return across the 4 tradeable indices.
- NIFTYNXT50 excluded from live trading — illiquid options produce noisy IV (40-180% range), unreliable SANOS calibration.
- **MIDCPNIFTY** is the best performer (+24.2%) — higher base IV means bigger fear spikes to capture.
- **BANKNIFTY** has the best Sharpe (5.39) — liquid options, clean IV signal, 68.8% win rate.
- Strategy works across all liquid NSE index derivatives — not overfit to NIFTY alone.

**Entropy/MI regime filters**: Ported Shannon entropy and mutual information from Timothy Masters' C++ library (`qlx/features/information.py`). NIFTY daily returns have entropy uniformly ~1.0 and MI uniformly ~0 across all windows. Information-theoretic regime filters are useless at daily frequency on liquid equity indices — the signal is indistinguishable from noise at this timescale.

### Strategy Pivot Decision

**WHAT DOESN'T WORK with daily data:**
- Option selling (straddles, strangles, any delta) — premiums too small, costs too high
- Directional prediction (delivery, OI, FII) — Sharpe -3.2
- FII flow momentum — contrarian if anything
- Max pain pinning — not a signal

**WHAT MIGHT WORK (needs different approach):**
1. **NLP Event Straddle (S3)** — BUY options before corporate events detected by FinBERT
   - We're buying cheap premium (small known loss)
   - Profit from large event-driven moves (asymmetric payoff)
   - FinBERT + T4 GPU already running
   - India corporate news is structured (earnings calendar, SEBI filings)
   - **This leverages our GPU edge and is the natural next step**

2. **Monthly options + longer hold** — 2.01% straddle / 0.95% strangle premium
   - More room for theta decay before gamma eats the position
   - Needs proper backtest with rolling entry/exit

3. **Intraday strategies** — need broker API (Zerodha Kite) for real-time data
   - Expiry-day gamma scalping, intraday VRP, regime switching

---

## Current Sprint

### Step 1: Assess option chain data availability — DONE ✓
- [x] F&O bhavcopy has ALL strike-level data: StrkPric, OptnTp (CE/PE), CLOSE, OPEN_INT, UndrlygPric, EXPIRY_DT
- [x] ~1,598 NIFTY option rows per day, 18 expiries, strikes from 23300 to 28100
- [x] ATM OI is massive: 25300 CE = 5.16M OI, 25000 PE = 6.86M OI
- [x] 167 days of data already cached (2025-06-01 → 2026-02-03)
- [x] No additional data source needed — bhavcopy is sufficient for daily VRP backtest

### Step 2: Build IV computation engine (GPU) — DONE ✓
- [x] Black-Scholes IV via Newton-Raphson on CUDA (1.8s for 1598 contracts)
- [x] Implied forward from put-call parity (eliminates call/put IV asymmetry)
- [x] HAR-RV realized vol model (1d, 5d, 22d components)
- [x] Greeks: delta, gamma computed alongside IV
- [x] GEX (Gamma Exposure) per strike
- [ ] SVI surface parameterization (deferred — not needed for VRP v2)

### Step 3: VRP backtest v1 — DONE ✓ (result: doesn't work blind)
- [x] Historical IV-RV spread: avg VRP = 1.8%
- [x] ATM straddle selling: Sharpe -8.4, 30% win rate
- [x] Cost model: 30bps round-trip
- **Lesson: need conditional entry (GEX regime, higher threshold, strangle width)**

### Step 4: FII Flow Momentum — DONE ✓ (no signal)
- [x] 3-day cumulative FII flow analyzed at ±1000/2000/3000/5000 Cr thresholds
- [x] Result: FII flow is CONTRARIAN, not momentum (heavy selling → market recovers)
- [x] Not usable for directional option trades

### Step 5: Expiry Pinning — DONE ✓ (no signal)
- [x] Max pain computed for 32 expiry days
- [x] Spot within 0.17% of max pain on ALL days (not just expiry)
- [x] 45% convergence (random) — max pain simply tracks spot

### Step 6: Weekly Strangle Selling — DONE ✓ (doesn't work)
- [x] Tested 10Δ/15Δ/20Δ/25Δ strangles, Monday to Thursday
- [x] 88% win rate but losses >3× wins → net negative at every delta
- [x] STT (0.0625% of notional) + costs eat tiny premiums

### Step 7: SANOS Vol Surface Fitting — DONE ✓
- [x] LP-based arbitrage-free surface from arXiv:2601.11209v2 implemented (`apps/india_fno/sanos.py`)
- [x] 15/15 LP convergence across 15 trading days, avg 0.62s calibration
- [x] Zero butterfly and calendar arbitrage violations
- [x] Mean IV error vs Newton-Raphson: 0.62%, max 6.76% (expiry rollover outlier)
- [x] Optimal config: N=100 model strikes, K in [0.7, 1.5], eta=0.50
- [x] Tests passing: `tests/test_sanos.py`
- **Next**: Use SANOS surface for strategy signal generation (mispriced options, calendar spread signals)

### Step 8: NLP Event Straddle for India — PARTIAL (see Step 13 for news scanner)
- [x] Add Indian RSS feeds (8 sources: ET, Hindu BL, LiveMint, BS, Moneycontrol) — `apps/india_news/scraper.py`
- [x] Entity extraction (365 aliases → 155 F&O stocks) — `apps/india_news/entity.py`
- [x] FinBERT sentiment + event-type classification — `apps/india_news/strategy.py`
- [x] Directional stock futures paper trader (sentiment-based, 3d TTL) — `apps/india_news/`
- [ ] Build earnings calendar scraper for F&O stocks (~150 stocks)
- [ ] Buy ATM straddles on individual stocks 1-2 days before detected catalyst
- [ ] Exit day after event (profit from IV crush OR directional move)
- [ ] Backtest using historical earnings dates + F&O bhavcopy
- [ ] This BUYS options (max loss = premium) — opposite of selling strategies that failed

### Step 9: Monthly Option Strategies (FUTURE)
- [ ] Monthly strangle backtest (sell 25Δ, hold 10d, exit at 50% profit)
- [ ] Calendar spread (sell weekly, buy monthly)
- [ ] Need more than 8 months of data for statistical significance

### Step 10: IV Mean-Reversion Strategy — DONE ✓
- **Strategy**: When SANOS ATM IV spikes above rolling 80th percentile, go long index futures. Exit when IV drops below median or after max hold days.
- **Best config**: Lookback=30, entry percentile=0.75-0.80, max hold=5-7 days, cost=5bps → **Sharpe 3.8-4.0, ~15% annualized, max DD 3.5%**
- **Parameter stability**: All 36 configs tested are profitable (Sharpe 1.29-3.99). Shorter lookback (30 days) consistently best. Strategy is robust to parameter choice.
- **Trade statistics**: 16 trades in 13 months (Jan 2025 - Jan 2026), 75% win rate, +12.4% total return.
- **Standout trade**: April 2025 India-Pakistan crisis generated 46.9% IV spike → +5.22% return in 4 days (single best trade). Even excluding this outlier, remaining trades are still net positive.
- **Multi-index validation**: Extended to BANKNIFTY (Sharpe 5.39), MIDCPNIFTY (4.19), FINNIFTY (3.36). Strategy works across all liquid NSE index derivatives — not NIFTY-specific.
- **Information-theoretic regime filters tested (Entropy + MI)**: No effect on NIFTY daily. Daily prices have entropy ~1.0 and MI ~0 uniformly — information-theoretic filters are useless at daily frequency on liquid equity index. `qlx/features/information.py` ported from Timothy Masters' C++ library, fully tested (`tests/test_iv_mean_revert.py`) but provides no signal for NIFTY regime filtering.
- **Files**: `apps/india_fno/iv_mean_revert.py`, `qlx/features/information.py`
- **Tests**: `tests/test_iv_mean_revert.py`

### Step 11: Multi-Index IV Mean-Reversion Paper Trader — DONE ✓
- **Indices**: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY (NIFTYNXT50 excluded — illiquid, Sharpe -0.66)
- **CLI**: `python -m apps.india_fno {scan,backtest,paper,status,backfill}`
- **State**: `MultiIndexPaperState` — per-index IV histories, per-index positions, global equity tracking
- **Position sizing**: Equal weight 1/N per index (N=4), each trade affects 25% of portfolio
- **Persistence**: Atomic JSON (`data/iv_paper_state.json`), crash-safe tempfile+rename
- **Legacy migration**: Auto-detects and migrates single-index (NIFTY-only) state files to multi-index format
- **Backfill+simulate (multi-index, verified 2026-02-05)**: 268 days, 79 signals, **76 trades, +16.54% equity, 63% win rate** (4 indices combined, 1/4 weight each)
- **Per-index P&L**: MIDCPNIFTY +19.2% (15/20 wins, 75%), BANKNIFTY +17.3% (10/15, 67%), FINNIFTY +13.2% (11/20, 55%), NIFTY +11.9% (12/21, 57%)
- **Gotcha**: Running `backfill` without `--simulate` collects IV observations but does NOT generate trades. A previous run without `--simulate` overwrote a correct 76-trade state with a NIFTY-only migrated state, making it look like non-NIFTY indices never traded. Fix: `rm data/iv_paper_state.json && python -m apps.india_fno backfill --start 2025-01-01 --simulate`
- **Backfill**: Parallelized with `ProcessPoolExecutor` (4 workers, one per index) — 268 days in 179s (1.5 days/s)
- **Dashboard**: Per-index IV sparklines, percentile ranks, position status, trades by symbol breakdown
- **Cron**: `scripts/iv_paper_daily.sh` — runs at 10:15 UTC (3:45 PM IST, after market close)
- **Files**: `apps/india_fno/paper_state.py`, `apps/india_fno/__main__.py`, `scripts/iv_paper_daily.sh`
- **Tests**: `tests/test_iv_paper.py` (60 tests)

---

## Key Decisions & Lessons

- **Directional prediction doesn't work** — India scanner Sharpe -3.2, crypto XGBoost Sharpe -0.6
- **Structural edge works** — funding rate arb Sharpe 1.8, **IV mean-reversion Sharpe 3.4-5.4 across 4 indices**
- **India F&O structural edge = volatility risk premium** — retail pays us to take risk
- **IV mean-reversion is the best India strategy found** — works across NIFTY (4.07), BANKNIFTY (5.39), MIDCPNIFTY (4.19), FINNIFTY (3.36). Multi-index diversification reduces single-index concentration risk.
- **RNDR options variant (bull put spreads) massively outperforms futures** — same signal, direct skew premium harvesting. Sharpe 5-11 vs 1-3. Defined risk per trade. Options profit even if index goes nowhere, as long as it doesn't crash.
- **NIFTYNXT50 is not tradeable** — illiquid options produce 40-180% IV range, noisy SANOS calibration, Sharpe -0.66
- **Information-theoretic filters (entropy, MI) are useless on daily equity indices** — signal indistinguishable from noise at daily frequency
- Transaction costs matter enormously — minimize turnover
- **Intraday OI flow is extremely noisy at 3-min frequency** — must smooth signals (EMA) and enforce minimum hold times. Raw snapshot-to-snapshot OI changes are dominated by noise, not institutional flow.
- ~1ms to NSE is an asset — enables near-real-time option chain processing

---

---

## SANOS Paper (arXiv:2601.11209v2, Jan 2026)

**"Smooth strictly Arbitrage-free Non-parametric Option Surfaces"**
Buehler, Horvath, Kratsios, Limmer, Saqur — Oxford, McMaster, DRW

### Why This Matters for Us
Our current IV engine computes IV per-contract independently (Newton-Raphson).
SANOS replaces this with a **global arbitrage-free surface** that:
1. Guarantees no calendar or butterfly arbitrage
2. Is **smooth** (continuous density), not piecewise-linear
3. Calibrates via **linear programming** (LP) — extremely fast
4. Handles bid-ask spreads natively (fit within spreads, not to mid)
5. Has a discrete local vol (DLV) parameterization — only positivity constraints

### Core Idea
```
Ĉ(Tj, K) = Σ_i q_j^i × Call(K^i, K, ηV_j)
```
- Option prices are convex combinations of BS call payoffs at quoted strikes
- `q_j` are martingale densities (discrete transition probabilities)
- `V_j` are ATM variances, `η ∈ [0,1)` controls smoothness (default η=0.25)
- When η=0: reduces to standard linear interpolation
- Fitting: single global LP across all expiries — **instantaneous**

### How to Integrate
1. Replace our per-contract Newton-Raphson IV with SANOS surface fitting
2. Use LP calibration (scipy.optimize.linprog or GPU-accelerated LP)
3. The smooth surface gives us reliable Greeks (delta, gamma, vega) everywhere
4. The DLV parameterization enables forward local vol computation
5. **Enables Strategy S5 (Vol Surface Arb)**: find options mispriced vs the smooth surface

### Implementation — COMPLETED

**Implementation**: `apps/india_fno/sanos.py`
- LP-based arbitrage-free vol surface fitting from arXiv:2601.11209v2
- Calibrates across 6 NIFTY expiries simultaneously
- Uses scipy.optimize.linprog with HiGHS solver

**Key results (tested across 15 trading days, Jan-Feb 2026):**
- 15/15 LP convergence (100% success rate)
- Average calibration time: 0.62s
- Mean IV error vs Newton-Raphson: 0.62%
- Max IV error: 6.76% (expiry rollover day outlier)
- Price surface: perfectly convex (zero butterfly violations), zero calendar arbitrage
- IV smile: smooth U-shaped with natural put skew, avg 5 zigzags

**Optimal config**: N=100 model strikes, K in [0.7, 1.5], eta=0.50

**Bugs fixed during development:**
1. Brenner-Subrahmanyam formula: used sqrt(pi/2) instead of sqrt(2*pi) → 4x variance underestimate
2. IV inversion: Newton-Raphson diverged for short-dated OTM → switched to bisection
3. ATM variance from single OTM option → use ATM straddle (CE+PE)

**Tests**: `tests/test_sanos.py`

### Priority Notes
- HIGH for vol surface arb and improved Greeks
- MEDIUM for VRP (our IV engine works adequately for ATM)
- **Next step**: Use the SANOS surface for strategy signal generation (detect mispriced options, calendar spread signals, etc.)

### Step 12: Intraday Microstructure Paper Trader — IN PROGRESS (v2 tuned)

**Location**: `apps/india_microstructure/`

**Architecture**: Full-stack intraday options microstructure system — data collection + analytics + paper trading.

#### Modules

| File | Purpose |
|------|---------|
| `auth.py` | Headless Zerodha Kite login (TOTP-automated, token caching) |
| `collector.py` | Option chain snapshot collector (Kite API → parquet) |
| `analytics.py` | GPU-accelerated microstructure analytics (GEX, OI flow, max pain, basis, PCR, IV term structure) |
| `signals.py` | Combined signal generator (weighted multi-signal → trade direction) |
| `paper_state.py` | Paper trading state persistence (atomic JSON) |
| `__main__.py` | CLI: `collect`, `paper`, `analyze`, `status` |

#### Data Collection
- **Source**: Zerodha Kite API (headless OAuth + TOTP, auto-cached access_token)
- **Indices**: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY
- **Snapshot interval**: Every 3 minutes during market hours (9:15 AM - 3:30 PM IST)
- **Data per snapshot**: 488 contracts (NIFTY), 374 (BANKNIFTY), 366 (MIDCPNIFTY), 242 (FINNIFTY)
- **Schema**: timestamp, symbol, expiry, strike, option_type, ltp, oi, volume, bid_price, ask_price, bid_qty, ask_qty, underlying_price, futures_price, futures_oi
- **Storage**: `data/india/chain_snapshots/{date}/{SYMBOL}_{HHMMSS}.parquet`
- **Strike selection**: 30 strikes each side of ATM, nearest 4 expiries

#### Five Microstructure Signals

**1. GEX (Gamma Exposure) — weight 30%**
- Computes net dealer gamma across all strikes (GPU-accelerated bisection IV + BS Greeks)
- Convention: dealers short calls (−γ) and short puts (+γ) relative to retail
- Net GEX > 0 → dealers long gamma → **mean-reversion regime** (fade moves)
- Net GEX < 0 → dealers short gamma → **momentum regime** (ride breakouts)
- GEX flip level: strike where cumulative GEX changes sign — key support/resistance
- In mean-revert regime: distance from flip → directional signal

**2. OI Delta Flow — weight 25%**
- Delta-weighted OI change between consecutive snapshots
- Large put OI additions (negative delta flow) → institutional hedging → **contrarian bullish**
- Large call OI additions → retail speculation → **contrarian bearish**
- Score normalized by total OI for cross-index comparability

**3. IV Term Structure — weight 20%**
- ATM IV ratio: near-expiry / next-expiry (via Brenner-Subrahmanyam straddle approximation)
- Inverted (slope > 1.2) → near-term panic → **contrarian bullish**
- Steep (slope < 0.8) → complacency → **mild bearish**

**4. Futures Basis — weight 15%**
- Basis = futures − spot, annualized
- Z-score vs rolling 100-observation history
- |Z| > 2.0 → overleveraged positioning → **contrarian signal**

**5. Put-Call Ratio (PCR) — weight 10%**
- OI-weighted and volume-weighted PCR for nearest expiry
- PCR > 1.3 → extreme fear → contrarian bullish
- PCR < 0.7 → extreme greed → contrarian bearish

#### Trade Rules (v2 — tuned 2026-02-05)
- **Tradeable**: NIFTY and BANKNIFTY futures only (most liquid)
- **Signal smoothing**: EMA on combined score (α=0.3, ~3 scans to converge)
- **Entry**: EMA-smoothed score > 0.40 AND ≥2 consecutive raw signals in same direction
- **Exit**: Signal flip (EMA reverses direction, after 15m min hold), signal decay (flat for 5+ consecutive scans, after 15m min hold), or max hold (120 min)
- **Minimum hold**: 15 minutes — no exits allowed before this regardless of signal
- **Market close**: All positions auto-closed at 3:30 PM IST
- **Cost model**: 5 bps round-trip (index futures, no STT on futures)
- **Conviction scaling**: |smoothed_score| / 0.5, capped at 1.0
- **Basis history**: Per-symbol (was global, causing cross-index contamination)

#### First Observations (2026-02-05)
- All 4 indices in **momentum regime** (negative GEX) throughout the day
- NIFTY: GEX flip at 26,000-26,150 (spot ~25,650 → well below flip, bearish pressure)
- BANKNIFTY: GEX flip at 62,300 (spot ~60,100)
- Basis: NIFTY +86 pts (5.0% ann), BANKNIFTY +190 pts (4.4% ann) — elevated but not extreme
- PCR: NIFTY 0.61-0.79, BANKNIFTY 0.99-1.03 (both neutral)
- IV: near-term ~22-25% for NIFTY/BN, MIDCPNIFTY elevated at 35%, FINNIFTY 27%
- FINNIFTY/MIDCPNIFTY far_iv = 0% (only 2 expiries available, 2nd expiry = near expiry)

#### v1 Results & Churning Problem (2026-02-05, 09:32–13:30 IST)
- **74 scans, 14 entries, 14 exits, 7% win rate, -0.621% total P&L**
- **ALL 14 exits were `conviction_drop`** — signal goes flat after 1 scan, exits immediately
- Root cause: OI flow score (25% weight) flips from +0.91 to -1.00 between consecutive 3-min snapshots
- No signal smoothing → raw noise directly triggers entry/exit
- Low entry threshold (0.25) → OI flow alone can trigger (0.25 × ±1.0 = ±0.25)
- No minimum hold time → enter and exit in the same 3-min cycle
- Average position age: ~3 minutes (one scan cycle)
- Cost per round-trip: 5bps → 14 trades × 5bps = 70bps burned on commissions alone

#### v2 Tuning (2026-02-05, 13:38 IST)
- **EMA smoothing** (α=0.3): dampens OI flow noise, takes ~3 scans (9 min) for signal to converge
- **Entry threshold raised** 0.25 → 0.40: requires stronger multi-signal conviction
- **Consecutive signal requirement** (≥2): must see 2+ raw signals in same direction before entry
- **Minimum hold time** (15 min): prevents sub-cycle exits
- **conviction_drop exit removed**: replaced with `signal_decay` (flat for 5+ consecutive scans)
- **Basis history per-symbol**: was using global list mixing NIFTY/BANKNIFTY z-scores
- v2 first scans show correct behavior: EMA building slowly (0.012 → 0.067), no premature entries

#### CLI Usage
```bash
python -m apps.india_microstructure collect              # collect snapshots only
python -m apps.india_microstructure paper                # collect + paper trade
python -m apps.india_microstructure paper --reset        # clear state + start fresh
python -m apps.india_microstructure analyze              # analytics on latest snapshot
python -m apps.india_microstructure status               # show positions + P&L
```

#### Key Technical Decisions
- **NSE API blocked from EC2** (Akamai bot protection) → Zerodha Kite API for all live data
- **Headless OAuth**: POST `/api/login` → POST `/api/twofa` (TOTP) → follow `/connect/login` redirect chain → extract `request_token` → `generate_session()`. Token cached to disk, valid same IST calendar day.
- **GPU IV (bisection)**: More robust than Newton-Raphson for short-dated OTM options, ~0.6s per index chain
- **Parquet storage**: ~30KB per NIFTY snapshot (488 contracts), ~120KB total per 3-min cycle across 4 indices. Full day ≈ 15MB.
- **GEX scale**: Raw units are crores (OI × Gamma × S² × lot_size / 1e7). Sign is the key signal, not magnitude.

### Step 14: RNDR Options Variant (Bull Put Spreads) — DONE ✓ (2026-02-05)

**Location**: `apps/india_fno/density_options.py`

**Purpose**: When the RNDR density signal fires (fear overpriced), sell OTM put credit spreads to directly harvest the skew premium with defined risk per trade.

**Trade structure** (bull put spread):
- SELL ~3% OTM put (short leg — higher strike, higher premium)
- BUY ~6% OTM put (long leg — lower strike, lower premium)
- Net credit = short_premium − long_premium
- Max risk = strike_width − net_credit

**Reuses from density_strategy.py**: `compute_composite_signal()`, `build_density_series()`, `DensityDayObs`, `_rolling_percentile`. No code duplication.

**Strike selection**: Nearest expiry with DTE >= hold+2, OI >= 100, premium >= Rs 0.50

**Exit**: hold_days reached, signal decay (composite < exit pctile), max loss (spot <= long strike), or end-of-data

**Config**: short_offset=3%, long_offset=6%, slippage=20bps on total premium, min_dte_buffer=2

#### Backtest Results (Jan 2025 → Feb 2026, entry=0.85, exit=0.40, hold=5d)

| Index | Trades | RoR% | Sharpe | Max DD | Win Rate |
|-------|--------|------|--------|--------|----------|
| BANKNIFTY | 16 | +54.3% | 5.34 | 3.18% | 75% |
| MIDCPNIFTY | 14 | +119.5% | 4.69 | 18.40% | 79% |
| FINNIFTY | 11 | +75.6% | 10.95 | 0.00% | 100% |
| **Aggregate (RoR)** | **41** | **+494.6%** | — | — | — |

Returns are % of risk capital (max loss per spread), not portfolio %.

#### Comparison vs Futures Variant

| Metric | Futures | Options |
|--------|---------|---------|
| Aggregate return | +17.69% (spot %) | +494.6% (risk capital %) |
| At 5% risk sizing | ~17.7% portfolio | ~24.7% portfolio |
| Risk profile | Unlimited downside | Defined max loss |
| Signal utilization | Indirect (buy dip) | Direct (sell overpriced fear) |
| Avg Sharpe | 1.82 | 6.99 |

**Key insight**: The signal detects "fear is overpriced". Selling overpriced fear directly (put spreads) is more efficient than buying the dip (futures).

**Bug fixed during development**: Exit MTM was looking up option prices from wrong expiry (different week's options at same strike), causing phantom losses 2-3x the strike width. Fixed by filtering `_mark_to_market_spread()` to match the entry expiry date.

**Tests**: 25 in `tests/test_density_options.py` — MockBhavcopyCache with BS put-call parity pricing. Total suite: 426 tests passing.

### Step 13: India News Sentiment Scanner — DONE ✓ (2026-02-05)

**Location**: `apps/india_news/`

**Purpose**: Scrapes Indian business RSS feeds, extracts F&O stock mentions via entity mapping, scores sentiment with FinBERT, and paper-trades individual stock futures. Runs as a **separate strategy** alongside the IV mean-reversion paper trader (different instruments: stock futures vs index futures).

#### Architecture

| File | Purpose |
|------|---------|
| `entity.py` | 365 company aliases → ~155 F&O symbols, longest-match-first extraction |
| `scraper.py` | 8 India RSS feeds (ET, Hindu BL, LiveMint, BS, Moneycontrol), event classification |
| `strategy.py` | FinBERT scoring (reused from `news_momentum`), event-type weights, signal aggregation |
| `state.py` | Paper trading state — `IndiaNewsTradingState`, atomic JSON persistence |
| `__main__.py` | CLI: `scan`, `paper --once`, `status` |

#### Key Design Decisions
- **Entity extraction**: Longest-match-first to avoid "Tata Consultancy Services" matching "Tata" → TATAMOTORS. Short aliases (< 4 chars) use word-boundary regex to prevent false positives.
- **Event classification priority**: regulatory > earnings > macro > corporate > general (regulatory first because "margin rules" was matching "margin" in earnings)
- **Event-type weights**: earnings 1.5x, regulatory 1.2x, corporate 1.3x, macro 0.8x, general 1.0x
- **Reuses**: `SentimentClassifier` from `apps.news_momentum.sentiment`, `FNO_UNIVERSE` from `apps.india_scanner.universe`, Zerodha Kite API for prices
- **Separate state file**: `data/india_news_state.json` (does NOT touch IV paper trader state)

#### Config
- Confidence threshold: >= 0.70
- Score threshold: |score| >= 0.50
- TTL: 3 trading days
- Max positions: 5
- Cost: 30 bps round-trip (stock futures, includes STT)

#### First Live Run (2026-02-05)
- **202 headlines** fetched from 8 RSS feeds (24h window during Q3 results season)
- **52 headlines** mentioned F&O stocks
- **5 signals generated**: LONG JSWSTEEL (+1.39), SHORT TATAPOWER (-1.11), LONG HDFCBANK (+1.07), SHORT KOTAKBANK (-1.01), SHORT BEL (-0.96)
- **5 positions entered** (state reset — was using bhavcopy prices, now uses Kite API)
- FinBERT quirk noted: "Kotak hiring spree" scored negative — known limitation on India-specific positive phrasing

#### CLI Usage
```bash
python -m apps.india_news scan                    # fetch RSS, show FinBERT scores + signals
python -m apps.india_news scan --max-age 1440     # 24h window
python -m apps.india_news paper --once            # single paper trading cycle (for cron)
python -m apps.india_news paper --once --date 2026-02-04  # override date
python -m apps.india_news status                  # dashboard
```

#### Cron
- `scripts/india_news_daily.sh` at **10:30 UTC** (4:00 PM IST, after market close)
- Uses Kite API for prices (last traded price available after-hours)

#### Tests
- **54 tests** in `tests/test_india_news.py` — all passing
- Covers: entity extraction (12), event classification (5), scraper (4), strategy (10), state (11), integration (7)

---

## Risk Notes

- SEBI restricted weekly expiry to one benchmark per exchange (Nov 2024): NSE = NIFTY weekly (Thu), BSE = SENSEX weekly (Fri)
- STT on option selling is higher than buying (0.0625% on sell side for options)
- Margin requirements for short options are substantial (SPAN + exposure)
- NSE Akamai bot protection on main API — use archives for historical, broker API for live
