# India F&O Strategy Reference

## Strategy Inventory

| # | Strategy | Type | Sharpe | Return | Max DD | Win% | Status | Risk |
|---|----------|------|--------|--------|--------|------|--------|------|
| 1 | IV Mean-Reversion | Daily, index futures | 3.4–5.4 | +14–24% | 3.2–4.5% | 55–76% | **PRODUCTION** | Low |
| 1b | RNDR Density (Futures) | Daily, index futures | 1.9–2.6 | +5–7% | 2.4–5.8% | 53–62% | BACKTESTED | Low |
| 1c | RNDR Density (Options) | Daily, put spreads | 4.7–11.0 | +54–120% RoR | 0–18% | 75–100% | BACKTESTED | Medium |
| 2 | Intraday Microstructure | Intraday, index futures | TBD | TBD | TBD | TBD | TUNING (v2) | Medium |
| 3 | India News Sentiment | Daily, stock futures | TBD | TBD | TBD | TBD | COLLECTING | Medium |
| 4 | Option Selling (all variants) | Weekly options | -2 to -8 | negative | large | 88% | **FAILED** | High |
| 5 | Directional (Delivery/OI/FII) | Daily, stock futures | -3.2 | negative | large | 42% | **FAILED** | High |

---

## 1. IV Mean-Reversion (PRODUCTION)

**Signal**: ATM IV (from SANOS vol surface) spikes above rolling 80th percentile → go long index futures. Exit when IV drops below 50th percentile or after max hold days.

**Config**: lookback=30, entry=0.80, exit=0.50, hold=5d, cost=5bps RT

**Why it works**: When ATM IV spikes, the index is in a fear regime. IV mean-reverts within days, and spot recovers. Retail is net option buyer, pushing IV above fair value. We capture the spot recovery, not the vol collapse.

### Multi-Index Backtest (Jan 2025 → Feb 2026, 268 days)

| Index | Trades | Return | Sharpe | Max DD | Win Rate |
|-------|--------|--------|--------|--------|----------|
| BANKNIFTY | 15 | +18.1% | 5.39 | 3.23% | 66.7% |
| MIDCPNIFTY | 20 | +20.2% | 4.19 | 4.48% | 75.0% |
| NIFTY | 21 | +13.0% | 4.07 | 3.61% | 57.1% |
| FINNIFTY | 20 | +14.2% | 3.36 | 3.64% | 55.0% |
| **Combined (1/4 each)** | **76** | **+16.5%** | **—** | **—** | **63.2%** |

NIFTYNXT50 excluded — illiquid options produce 40–180% IV noise, Sharpe -0.66.

### Parameter Robustness
All 36 parameter combinations (lookback × entry pctl × hold) profitable. Sharpe range 1.29–3.99. Shorter lookback (30d) consistently best. Strategy is not overfit.

### Standout Trade
April 2025 India-Pakistan crisis: 46.9% IV spike → +5.22% in 4 days. Even excluding this outlier, remaining 15 trades are net positive.

### Live Paper State (as of 2026-02-05)
- **Equity**: 1.165x (+16.5%)
- **Open positions**: NIFTY (entry 25,321, Jan 30), BANKNIFTY (entry 59,958, Jan 29), MIDCPNIFTY (entry 13,424, Jan 29)
- **FINNIFTY**: flat
- **State file**: `data/iv_paper_state.json`

### Files
- `apps/india_fno/iv_mean_revert.py` — backtest engine
- `apps/india_fno/sanos.py` — SANOS vol surface (LP-based, arXiv:2601.11209v2)
- `apps/india_fno/iv_engine.py` — GPU IV computation (Newton-Raphson + bisection)
- `apps/india_fno/paper_state.py` — `MultiIndexPaperState`, atomic JSON
- `apps/india_fno/__main__.py` — CLI
- `tests/test_iv_paper.py` — 60 tests
- `tests/test_iv_mean_revert.py`, `tests/test_sanos.py`

### CLI
```bash
python -m apps.india_fno scan                          # show current IV + signals
python -m apps.india_fno backtest                      # run parameter sweep
python -m apps.india_fno paper --once                  # single scan cycle (for cron)
python -m apps.india_fno status                        # dashboard
python -m apps.india_fno backfill --start 2025-01-01 --simulate  # historical replay
```

### Cron
`scripts/iv_paper_daily.sh` at **10:15 UTC** (3:45 PM IST, after market close).
```
15 10 * * 1-5 /home/ubuntu/Desktop/7hills/qlx_python/scripts/iv_paper_daily.sh
```

---

## 1b/1c. RNDR — Risk-Neutral Density Regime (BACKTESTED)

**Signal**: 4-component composite from SANOS risk-neutral density — skew premium (40%), left tail weight (25%), entropy change (20%), KL divergence direction (15%). Fires when composite percentile >= 85th (fear is overpriced). Same signal drives both variants.

**Config**: lookback=30, entry=0.85, exit=0.40, hold=5d

**Edge**: Retail option buyers systematically overpay for OTM puts (Kahneman probability weighting), creating a gap between implied and realized skewness that mean-reverts predictably.

### Futures Variant (1b)

Long index futures when fear is overpriced. Cost: 5bps RT.

| Index | Trades | Return | Sharpe | Max DD | Win Rate |
|-------|--------|--------|--------|--------|----------|
| BANKNIFTY | 18 | +5.37% | 1.89 | 2.92% | 56% |
| MIDCPNIFTY | 17 | +4.78% | 0.96 | 5.75% | 53% |
| FINNIFTY | 16 | +6.59% | 2.62 | 2.38% | 62% |
| **Aggregate** | **51** | **+17.69%** | — | — | — |

NIFTY excluded (too efficient, negative across all configs).

### Options Variant (1c) — Bull Put Credit Spreads

Sell ~3% OTM put, buy ~6% OTM put (defined risk). Directly harvests the overpriced skew. Cost: 20bps slippage.

| Index | Trades | RoR% | Sharpe | Max DD | Win Rate |
|-------|--------|------|--------|--------|----------|
| BANKNIFTY | 16 | +54.3% | 5.34 | 3.18% | 75% |
| MIDCPNIFTY | 14 | +119.5% | 4.69 | 18.40% | 79% |
| FINNIFTY | 11 | +75.6% | 10.95 | 0.00% | 100% |
| **Aggregate** | **41** | **+494.6% RoR** | — | — | — |

Returns are % of risk capital (max loss per spread), not portfolio %. At 5% risk per trade, portfolio return ≈ 25%.

### Why Options >> Futures

The signal detects "fear is overpriced". Selling overpriced fear directly (put spreads) is more efficient than buying the dip (futures) and hoping for a bounce. Options profit even if the index goes nowhere, as long as it doesn't crash. Every trade has defined max loss.

### Caveats

- Options returns are on risk capital, not portfolio notional
- FINNIFTY 100% win rate is likely fragile (least liquid, 11 trades only)
- MIDCPNIFTY max DD 18.4% on risk capital — one bad trade
- P&L uses EOD bhavcopy settlement prices — intraday max-loss exits would be missed
- 20bps slippage may underestimate friction on illiquid OTM puts

### Files
- `apps/india_fno/density_strategy.py` — futures variant backtest + shared signal
- `apps/india_fno/density_options.py` — options variant backtest (bull put spreads)
- `apps/india_fno/risk_neutral.py` — Breeden-Litzenberger density extraction from SANOS
- `tests/test_density_strategy.py` — 24 tests (futures variant)
- `tests/test_density_options.py` — 25 tests (options variant)
- `tests/test_risk_neutral.py` — 25 tests (density extraction)

---

## 2. Intraday Microstructure (TUNING — v2)

**Signal**: 5-signal weighted composite from real-time option chain snapshots (3-min interval via Zerodha Kite API).

| Signal | Weight | Logic |
|--------|--------|-------|
| GEX (Gamma Exposure) | 30% | Net dealer gamma → mean-revert vs momentum regime + flip level |
| OI Delta Flow | 25% | Delta-weighted OI change between snapshots → institutional flow |
| IV Term Structure | 20% | Near/far IV ratio → panic (inverted) vs complacency (steep) |
| Futures Basis | 15% | Basis z-score → overleveraged positioning |
| Put-Call Ratio | 10% | PCR extremes → contrarian sentiment |

**Instruments**: NIFTY + BANKNIFTY futures only (most liquid)

**Trade rules (v2)**:
- EMA smoothing (alpha=0.3) on combined score
- Entry: smoothed score > 0.40 AND >= 2 consecutive raw signals in same direction
- Exit: signal decay (5 flat scans after 15m min hold), signal flip, or 120m max hold
- All positions auto-closed at 3:30 PM IST
- Cost: 5bps RT

### v1 Results (diagnosed failure)
74 scans, 14 trades, 7% win rate, -0.62% P&L. All exits via conviction_drop — OI flow flips +/-1.0 between 3-min snapshots. Average position age: ~3 minutes. Commissions (14 x 5bps = 70bps) exceeded any signal.

### v2 Fixes
EMA dampens noise (~3 scans to converge), raised threshold, consecutive signal requirement, 15m minimum hold, signal_decay exit (5 flat scans). Early v2 scans show correct slow buildup.

### Live State (2026-02-05)
- **Positions**: none (v2 correctly filtering noise)
- **Analytics**: all 4 indices in momentum regime (negative GEX), signals flat
- **State file**: `data/micro_paper_state.json`

### Files
- `apps/india_microstructure/auth.py` — headless Zerodha OAuth + TOTP
- `apps/india_microstructure/collector.py` — option chain snapshots → parquet
- `apps/india_microstructure/analytics.py` — GPU bisection IV + Greeks + all 5 signals
- `apps/india_microstructure/signals.py` — combined signal generator
- `apps/india_microstructure/paper_state.py` — paper trading state

### CLI
```bash
python -m apps.india_microstructure collect    # collect snapshots only
python -m apps.india_microstructure paper      # collect + paper trade
python -m apps.india_microstructure analyze    # analytics on latest snapshot
python -m apps.india_microstructure status     # positions + P&L
```

---

## 3. India News Sentiment (COLLECTING)

**Signal**: FinBERT sentiment on Indian business RSS feeds → F&O stock futures (directional).

**Pipeline**:
1. Fetch headlines from 8 RSS sources (ET, Hindu BL, LiveMint, BS, Moneycontrol, NDTV Profit, Financial Express, Outlook Business)
2. Extract F&O stock mentions (365 company aliases → ~155 symbols, longest-match-first)
3. Classify event type: regulatory > earnings > macro > corporate > general
4. Score with FinBERT (GPU, ~11ms per batch of 5)
5. Aggregate per-stock: weighted by event type (earnings 1.5x, corporate 1.3x, regulatory 1.2x, macro 0.8x)
6. Trade stock futures: long if score > +0.50, short if < -0.50 (conf >= 0.70)

**Config**: TTL=3 trading days, max_pos=5, cost=30bps RT (stock futures, includes STT)

### First Live Run (2026-02-05)
- 202 headlines from 8 feeds (Q3 results season)
- 52 headlines matched F&O stocks
- 5 signals: LONG JSWSTEEL (+1.39, earnings), SHORT TATAPOWER (-1.11, earnings), LONG HDFCBANK (+1.07, regulatory), SHORT KOTAKBANK (-1.01, regulatory), SHORT BEL (-0.96, general)
- 5 positions entered (state reset after price source fix — was using bhavcopy, now Kite)
- **Prices**: Zerodha Kite API (`last_price`) — no bhavcopy dependency
- **State file**: `data/india_news_state.json`

### Files
- `apps/india_news/entity.py` — 365 aliases → ~155 F&O symbols
- `apps/india_news/scraper.py` — 8 RSS feeds + event classification
- `apps/india_news/strategy.py` — FinBERT scoring + signal aggregation
- `apps/india_news/state.py` — `IndiaNewsTradingState`
- `apps/india_news/__main__.py` — CLI
- `tests/test_india_news.py` — 54 tests

### CLI
```bash
python -m apps.india_news scan                    # fetch + score headlines
python -m apps.india_news scan --max-age 1440     # 24h window
python -m apps.india_news paper --once            # single paper cycle (for cron)
python -m apps.india_news status                  # dashboard
```

### Cron
`scripts/india_news_daily.sh` at **10:30 UTC** (4:00 PM IST).
```
30 10 * * 1-5 /home/ubuntu/Desktop/7hills/qlx_python/scripts/india_news_daily.sh
```

---

## 4. Failed: Option Selling

### Weekly Strangle Selling (all deltas)

| Delta | Premium% | Win% | Avg Net P&L | Sharpe |
|-------|----------|------|-------------|--------|
| 10-delta | 0.12% | 88% | -0.28% | -6.96 |
| 15-delta | 0.18% | 88% | -0.25% | -4.81 |
| 20-delta | 0.25% | 88% | -0.20% | -3.14 |
| 25-delta | 0.32% | 88% | -0.16% | -2.09 |

**Root cause**: NIFTY weekly premiums (0.31% strangle, 0.82% straddle) are too small vs median 5d move (0.97%). The 1 losing week (-1.0% to -1.5%) wipes all gains. STT on option selling (0.0625% notional) + brokerage eat the premium even on wins.

### VRP Analysis
- Avg IV: 9.9%, avg RV: 8.1%, avg VRP: 1.8% annualized
- 1.8% / 52 = 0.035% per week — far less than 30bps RT costs
- Blind ATM straddle selling: Sharpe -8.4, 30% win rate

**Conclusion**: Volatility risk premium exists but is too small to harvest after costs in NIFTY weeklies. Monthly options (2.01% straddle premium) are more promising but untested.

---

## 5. Failed: Directional Signals (Delivery/OI/FII)

**Backtest**: 8-month institutional footprint scan. Sharpe -3.2, 42% win rate.

| Signal | Finding |
|--------|---------|
| FII flow | **Contrarian**, not momentum. Heavy selling → market UP (+0.23%). Not predictive. |
| Max pain pinning | Spot within 0.17% of max pain on ALL days (not just expiry). 45% convergence = random. |
| Delivery spike | Weak signal, overwhelmed by noise |
| OI quadrant | Moderate but insufficient alone |

**Also tested**: Entropy and mutual information regime filters (ported from Timothy Masters C++ library). NIFTY daily returns have entropy ~1.0 and MI ~0 uniformly — information-theoretic filters are useless on liquid equity index daily bars.

---

## Data Dependency Matrix

| Data Source | IV Paper | Microstructure | News | Frequency | Format |
|-------------|----------|----------------|------|-----------|--------|
| NSE F&O bhavcopy | **Required** | — | — | Daily (T+0) | parquet |
| Zerodha Kite quotes | — | — | **Required** (prices) | On-demand | API |
| NSE FII/DII data | — | — | — | Daily (T+0) | parquet |
| Zerodha Kite option chain | — | **Required** | — | 3-min intraday | parquet |
| RSS feeds (8 sources) | — | — | **Required** | On-demand | in-memory |
| SANOS vol surface | **Required** | — | — | Daily | computed |
| GPU IV engine | **Required** | **Required** | — | Per-scan | computed |
| FinBERT model | — | — | **Required** | Per-scan | GPU |
| F&O universe (150 stocks) | — | — | **Required** | Static | Python |

### Storage
- Bhavcopy cache: `data/india/` (~8 months, 2025-06-01 → 2026-02-05)
- Chain snapshots: `data/india/chain_snapshots/{date}/{SYMBOL}_{HHMMSS}.parquet` (~15MB/day)
- State files: `data/iv_paper_state.json`, `data/india_news_state.json`, `data/micro_paper_state.json`

---

## Infrastructure

### Hardware
- **CPU**: Xeon Platinum 8259CL, 16C/32T, AVX-512
- **GPU**: Tesla T4, 16GB VRAM, Turing tensor cores (65 TFLOPS FP16)
- **RAM**: 124GB (119GB free)
- **Network**: ~1ms to NSE (AWS ap-south-1, co-located with NSE infrastructure)

### Key Components

| Component | Speed | Used By |
|-----------|-------|---------|
| SANOS vol surface (LP) | 0.62s / day / index | IV paper |
| GPU IV engine (Newton-Raphson + bisection) | 1,379 contracts / 1.8s | IV paper, microstructure |
| FinBERT inference (T4) | ~11ms / batch of 5 | News sentiment |
| Headless Zerodha login (TOTP) | ~2s (cached same day) | Microstructure |

### Cost Models

| Instrument | Cost (RT) | Components |
|------------|-----------|------------|
| Index futures (NIFTY, BANKNIFTY, etc.) | **5 bps** | Brokerage (Rs 20 flat) + exchange + SEBI + GST. No STT on index futures. |
| Stock futures (F&O universe) | **30 bps** | STT (0.02%) + brokerage + exchange + SEBI + stamp + GST |
| Options (selling) | **~60+ bps** | STT (0.0625% on sell side) + brokerage + exchange + SEBI + stamp + GST |

---

## Execution & Monitoring

### State Files

| File | Strategy | Format |
|------|----------|--------|
| `data/iv_paper_state.json` | IV mean-reversion | JSON (MultiIndexPaperState: iv_histories, positions, closed_trades, equity) |
| `data/india_news_state.json` | News sentiment | JSON (active_trades, closed_trades, equity, config) |
| `data/micro_paper_state.json` | Microstructure | JSON (positions, closed_trades, analytics_log) |

All use atomic persistence (write to tmpfile, then rename) — crash-safe.

### Cron Schedule (weekdays only)

| Time (UTC) | Time (IST) | Script | Strategy |
|------------|------------|--------|----------|
| 10:15 | 3:45 PM | `scripts/iv_paper_daily.sh` | IV mean-reversion scan |
| 10:30 | 4:00 PM | `scripts/india_news_daily.sh` | News sentiment scan |
| Market hours | 9:15 AM – 3:30 PM | Manual / future cron | Microstructure collect+paper |

### Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `tests/test_iv_paper.py` | 60 | Multi-index paper state, backfill, simulate |
| `tests/test_sanos.py` | 15+ | SANOS vol surface, LP convergence, arb-free |
| `tests/test_iv_mean_revert.py` | 15+ | IV mean-reversion backtest |
| `tests/test_risk_neutral.py` | 25 | Breeden-Litzenberger density, RNDR signal |
| `tests/test_density_strategy.py` | 24 | RNDR futures variant backtest |
| `tests/test_density_options.py` | 25 | RNDR options variant (bull put spreads) |
| `tests/test_india_news.py` | 54 | Entity extraction, scraper, strategy, state |
| `tests/test_india_scanner.py` | 24 | Scanner, signals, bhavcopy |
| `tests/test_bhavcopy.py` | 22 | Bhavcopy download, parsing |
| `tests/test_signals.py` | 16 | Delivery, OI, FII signals |
| **Total** | **426** | |

---

## Capital Allocation Framework

### Allocation Principles
1. **Kelly-inspired sizing**: Allocate proportional to Sharpe ratio, inversely proportional to max DD
2. **No instrument overlap**: IV paper (index futures) and news (stock futures) trade different instruments — no conflict
3. **Priority rule**: When microstructure and IV paper both signal on the same index, IV paper gets priority (higher Sharpe, proven track record)

### Suggested Allocation (paper phase)

| Strategy | Allocation | Rationale |
|----------|------------|-----------|
| IV Mean-Reversion | 50% | Highest Sharpe (3.4–5.4), lowest DD (3–4.5%), proven over 268 days |
| News Sentiment | 25% | Different instruments (stock futures), uncorrelated signal, unproven |
| Microstructure | 25% | Highest potential but unproven, still tuning |

Within IV paper, equal weight 1/4 per index (current default).

---

## Correlation Analysis

### Cross-Strategy Correlation

| Pair | Expected Correlation | Reason |
|------|---------------------|--------|
| IV paper × News | **Low** | Different instruments (index vs stock futures), different signals (IV percentile vs NLP sentiment), different holding periods (5d vs 3d) |
| IV paper × Microstructure | **Medium** | Same instruments (NIFTY/BANKNIFTY futures) but different timeframes (daily vs intraday) and different signals. Potential overlap on same-index entries. |
| News × Microstructure | **Low** | Different instruments (stock vs index futures), different timeframes |

### Diversification Benefit
- **Daily + intraday + event-driven**: Three orthogonal signal types
- **Index + stock**: Two instrument classes with moderate correlation
- **Mean-reversion + microstructure + sentiment**: Different alpha sources
- IV paper and microstructure may conflict on same-index positions during high-IV regimes. Priority rule (IV paper wins) prevents double exposure.

---

## Regulatory Notes
- SEBI restricted weekly expiry to one benchmark per exchange (Nov 2024): NSE = NIFTY weekly (Thu)
- STT on option selling is higher than buying (0.0625% sell side)
- NSE main API blocked from EC2 (Akamai bot protection) — use bhavcopy archives for historical, Zerodha Kite for live
- Margin requirements for short options are substantial (SPAN + exposure) — not relevant for futures-only strategies
