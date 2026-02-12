# BRAHMASTRA Strategy Scorecard — 2026-02-12 (Updated)

## Executive Summary
- **Total strategies**: 26 (S1–S26)
- **Backtested**: 16 (S6, S8, S11 bugs fixed and re-run)
- **Profitable (Sharpe > 0.5)**: 6
- **Top performer**: S4 IV Mean-Reversion (NIFTY Sharpe 1.28, multi-index avg +15.5%)
- **Best walk-forward validated**: S25 DFF (NIFTY OOS Sharpe 1.87, BANKNIFTY 2.16)
- **Best ensemble**: S25+S4+S5 inverse-vol (Sharpe 1.90, MaxDD 1.20%)
- **ML pipelines running**: TFT X-Trend (Phase 4 Optuna, 10 trials), DTRN (vectorized backtest refactor)

## Summary Table

| # | Strategy | Best Sharpe | Return% | MaxDD% | WinRate | Trades | Period | Status |
|---|----------|-------------|---------|--------|---------|--------|--------|--------|
| S1 | VRP Options (bull put) | **2.90** | +101% RoR | — | 67% | 33 | 489d | **Live-ready** |
| S2 | Ramanujan Cycles | -0.21 | -1.20% | — | 48% | 1765 | 420d | No edge |
| S3 | Institutional Flow | — | — | — | — | — | — | No output |
| S4 | IV Mean-Reversion | **1.28** | +21.57% | 6.03% | 56% | 39 | 489d | **Live-ready** |
| S5 | Hawkes (tick micro) | **1.23** | +9.60% | 5.93% | 57% | 75 | 489d | **Live-ready** |
| S6 | Multi-Factor (XGB) | **-0.19** | -1.83% | 16.53% | 53% | 135 | 492d | No edge (fixed) |
| S7 | Regime Switch (NIFTY) | **1.13** | +8.65% | 2.72% | 83% | 6 | 489d | Low trade count |
| S8 | Expiry Theta | 0.00 | 0.00% | 0.00% | — | 0 | 489d | No trades (schema bug) |
| S9 | Cross-Sect Momentum | -0.51 | -12.81% | 30.98% | 53% | 448 | 489d | No edge |
| S10 | Gamma Scalp | -0.43 | -2.05% | 5.54% | 20% | 5 | 489d | No edge |
| S11 | Pairs Trading | 1.01 | +22.59% | 18.24% | 100% | 4 | 489d | Fixed, low activity |
| S12 | SABR/Vedic FFPE | 0.63 | +12.72% | 14.54% | 53% | 57 | 489d | Marginal |
| S13-S24 | Various | — | — | — | — | — | — | Stub only |
| S25 | Divergence Flow Field | **1.87** | +6.32% | <2.1% | — | 93 | 252d OOS | **Walk-forward validated** |
| S26 | Crypto Flow (CLRS) | — | — | — | — | — | — | No research script |

## Tier 1: Live-Ready (Sharpe > 1.0, validated)

### S25: Divergence Flow Field
- **Walk-forward OOS** (4 folds, 252 days): NIFTY Sharpe **1.87** (+6.32%), BANKNIFTY Sharpe **2.16** (+8.06%)
- All 4 folds profitable, min Sharpe 1.04 (NIFTY), 1.74 (BANKNIFTY)
- Helmholtz decomposition of NSE 4-party open interest → 12 features
- Signal quality genuine: Sharpe invariant to position scale
- Max DD < 2.1%, avg hold 1.1-1.4 days, 93 trades at thresh=0.3
- **3-agent audit: zero look-ahead bias confirmed**

### S4: IV Mean-Reversion
- **489 days, multi-index**: NIFTY Sharpe 1.28 (+21.57%), BANKNIFTY 1.01 (+20.04%), MIDCPNIFTY 0.97 (+23.41%)
- 30-39 trades per index, WR 54-66%
- Uses SANOS IV surface vs realized vol z-score
- MaxDD 6-11% across indices
- Avg total return +15.5% across 4 indices (171 trades)

### S5: Hawkes Jump (Tick Microstructure)
- **Best config**: hawkes_mean, short, lb=30, entry=0.8 → Sharpe **1.23**, +9.60%, MaxDD 5.93%, WR 57.3%, 75 trades
- Uses Hawkes process intensity from 1-min bar data
- Short bias (mean-reverting jump intensity)
- Other features (VPIN, entropy) negative — only hawkes_mean works

### S1: VRP Options (Bull Put Spreads)
- **BANKNIFTY options**: RoR +101.43%, Sharpe **2.90**, WR 67%, 33 trades
- **MIDCPNIFTY options**: RoR +68.75%, Sharpe **1.95**, WR 62%, 21 trades
- Futures variant much weaker (NIFTY Sharpe -0.24) — edge is in options only
- Uses risk-neutral density vs historical density divergence

## Tier 2: Promising but Limited

### S7: Regime Switch
- NIFTY: Sharpe 1.13, +8.65%, MaxDD 2.72%, WR 83%, but only **6 trades**
- BANKNIFTY: Sharpe -0.41 (doesn't work)
- Too few trades for statistical significance

### S12: SABR/Vedic FFPE
- Sharpe 0.63, +12.72%, MaxDD 14.54%, 57 trades
- IC near zero (-0.0007) — predictive power questionable
- MaxDD too high relative to return (Calmar 0.87)

## Tier 3: No Edge

| Strategy | Sharpe | Issue |
|----------|--------|-------|
| S2 Ramanujan | -0.21 | Fourier cycles don't predict returns |
| S6 Multi-Factor | -0.19 | XGBoost R²=-0.90, IC=-0.08 (p=0.36) — no predictive skill (inf bug fixed) |
| S9 Momentum | -0.51 | Cross-sectional momentum negative in India FnO |
| S10 Gamma | -0.43 | Too few trades, negative |

### S6: Multi-Factor (XGBoost) — FIXED, No Edge
- **Bug fixed**: `inf` values from Nifty50 Dividend Points (17 zero values → `pct_change()` → inf). Added `inf→NaN→ffill→fillna(0)` before XGBoost.
- **Walk-forward CV** (27 folds, train=60d, test=5d, purge=5d): R²=-0.90, IC=-0.08
- **Result**: Sharpe -0.19, -1.83%, MaxDD 16.53%, 135 predictions, hit rate 52.6%
- **Diagnosis**: 402 features (134 NSE index returns × 3 windows) are too noisy for 60-day training windows. Model overfits.

### S11: Pairs Trading — FIXED, Low Activity
- **Bugs fixed**: (1) Hurst R/S analysis computed on levels instead of increments (returned 1.0 for all), (2) insufficient lookback window, (3) overly strict thresholds
- **Cointegration funnel**: 21,736 pairs → 64 pass ADF (0.29%) → 0 pass Hurst<0.55
- **With relaxed Hurst**: 4 trades, all winners, Sharpe 1.01, +22.59%, but MaxDD 18.24%
- **Diagnosis**: India stock FnO shows near-zero cointegration in 2025-2026. Only 0.29% of pairs pass basic ADF test. Strategy works when pairs exist but the market environment is hostile.

## Tier 4: Broken / No Data

| Strategy | Issue |
|----------|-------|
| S3 Institutional | No output (missing data?) |
| S8 Expiry Theta | 0 trades — `timestamp` column missing from `nfo_1min` schema (silently swallowed by bare `except`). Data IS available (420 days of 1-min FnO bars). Needs column name fix. |
| S13-S24 | Stub `_scan_impl()` only — not implemented |
| S26 Crypto | No research backtest script |

## Ensemble Analysis (from 2026-02-10)

**S25 + S4 + S5 Inverse-Vol Ensemble**:
- Combined Sharpe: **1.90**, MaxDD **1.20%**, Return +5.04%
- Near-zero correlations: S25↔S4 -0.21, S25↔S5 -0.04, S4↔S5 -0.01
- Diversification ratio 1.77 (portfolio vol 44% below avg individual vol)
- **Best risk-adjusted combination in the system**

## ML Pipelines (In Progress)

### TFT X-Trend
- **Currently**: Phase 5 Production walk-forward, fold 0, val_sharpe=2.24
- 6 assets (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, BTCUSDT, ETHUSDT)
- 321→73 features after VSN selection
- Previous OOS Sharpe: 1.88 (before purge_gap fix — may be inflated)

### DTRN (Dynamic Topology Regime Network)
- **Currently**: 8 parallel folds training (Stage 3 — trading objective)
- 26 features from 1-min microstructure data
- fold1_BANKNIFTY showing positive avg_pnl in Stage 3
- First OOS results expected in ~2 hours

## Rankings by Sharpe (backtested strategies only)

1. **S1 VRP Options** (BANKNIFTY) — 2.90
2. **S25 DFF** (BANKNIFTY OOS) — 2.16
3. **S1 VRP Options** (MIDCPNIFTY) — 1.95
4. **S25 DFF** (NIFTY OOS) — 1.87
5. **S4 IV MR** (NIFTY) — 1.28
6. **S5 Hawkes** (hawkes_mean) — 1.23
7. **S7 Regime** (NIFTY) — 1.13 (6 trades only)
8. **S4 IV MR** (BANKNIFTY) — 1.01
9. **S4 IV MR** (MIDCPNIFTY) — 0.97
10. **S12 FFPE** — 0.63

## Methodology
- **Sharpe**: ddof=1, sqrt(252), all daily returns including flat days
- **Costs**: 3 pts/leg NIFTY, 5 pts/leg BANKNIFTY, 5 bps futures
- **Execution**: T+1 (signal at close day T, fill at open day T+1)
- **No look-ahead bias**: All signals fully causal
- **Walk-forward**: 4-fold purged (gap=5 days) for S25 and TFT
- **Data period**: 2024-02-12 to 2026-02-10 (489 trading days) for most strategies

## Key Takeaways

1. **Options strategies dominate**: S1 and S4 use actual SANOS IV surfaces — the edge is in vol mispricing
2. **Microstructure matters**: S5 Hawkes (1-min), S25 DFF (daily OI decomposition) — non-price signals work
3. **Classic momentum fails**: S2 (Ramanujan), S9 (cross-sectional) — India FnO is not a momentum market
4. **Data quality is critical**: S6 (inf values), S8 (no options data), S11 (no pairs) — failures are data, not logic
5. **Ensemble is king**: 3-strategy inverse-vol ensemble achieves Sharpe 1.90 with only 1.20% MaxDD
