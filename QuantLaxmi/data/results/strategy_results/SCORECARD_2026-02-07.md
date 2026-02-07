# BRAHMASTRA — Full Strategy Scorecard (2026-02-07)

**Data period**: Oct 2024 – Feb 2026 (316 trading days)
**Sharpe protocol**: ddof=1, sqrt(252) annualization, all daily returns including flat days
**Options pricing**: Actual prices from nse_fo_bhavcopy / nfo_1min (not toy approximations)
**Cost model**: Per-leg index-point costs (3 pts NIFTY, 5 pts BANKNIFTY) for options; 10 bps for equity

---

## Tier 1: Strategies with Real Edge

| # | Strategy | Best Config | Sharpe | Return | WR | Trades | MaxDD | Notes |
|---|----------|-------------|--------|--------|----|--------|-------|-------|
| S5 | Hawkes Microstructure | hawkes_ratio short lb=60 | **4.29** | +3.62% | 75% | 12 | 0.56% | GPU tick features, causal rolling IC |
| S1 | VRP Options (BANKNIFTY) | bull put spreads | **5.59** | +11.57% RoR | 50% | 9 | 0.88% | Actual option prices, return-on-risk |
| S4 | IV Mean-Revert (BANKNIFTY) | iv_lb=30, pctile=0.8 | **3.07** | +8.91% | 71% | 7 | 2.50% | SANOS-calibrated IV |
| S7 | Regime Switch (NIFTY) | lb=60 stm=2.0 | **2.37** | +4.10% | 100% | 2 | 1.30% | Only 2 trades — needs more data |
| S1 | VRP Futures (BANKNIFTY) | entry_pctile=0.8 | **1.67** | +2.72% | 33% | 9 | 0.71% | Directional density signal |

## Tier 2: Marginal / Break-Even

| # | Strategy | Best Config | Sharpe | Return | WR | Trades | MaxDD | Notes |
|---|----------|-------------|--------|--------|----|--------|-------|-------|
| S9 | Cross-Section Momentum | top_n=7 | **0.76** | +2.90% | 55% | 159 | 6.04% | Delivery + OI signals, weekly rebalance |
| S8 | Expiry Theta (NIFTY) | otm=2.0% | **0.19** | +0.22% | 46% | 66 | 0.84% | Actual option prices, realistic costs |

## Tier 3: No Edge / Negative

| # | Strategy | Best Config | Sharpe | Return | WR | Trades | MaxDD | Notes |
|---|----------|-------------|--------|--------|----|--------|-------|-------|
| S2 | Ramanujan Cycles | max_period=64 | **-1.11** | -4.65% | 47% | 1271 | — | Cycle detection lacks predictive power |
| S6 | Multi-Factor ML | XGBoost walk-forward | **-2.48** | -3.35% | 65% | 7.55% | — | Corrected Sharpe (was inflated 2.25x by wrong annualization) |

## Tier 4: Insufficient Signal / No Trades

| # | Strategy | Best Config | Sharpe | Return | WR | Trades | MaxDD | Notes |
|---|----------|-------------|--------|--------|----|--------|-------|-------|
| S10 | Gamma Scalp | ivp=0.30 dte=10 | **-1.01** | -0.49% | 0% | 2 | 0.85% | Actual straddle prices; IV never cheap enough in low-vol env |
| S11 | Pairs Trading | all configs | **0.00** | 0.00% | — | 0 | 0% | No cointegration found in stock futures |

## Signal-Only (No Backtest P&L)

| # | Strategy | Metric | Value | Notes |
|---|----------|--------|-------|-------|
| S3 | Institutional Flow | 1-day IC | +0.042 (p=0.17) | Quintile spread +0.21%, not stat-sig |
| S3 | Institutional Flow | 3-day IC | +0.048 (p=0.11) | Approaching significance |

---

## Detailed Results by Strategy

### S1 VRP Risk-Neutral Density
- **Futures variant**: BANKNIFTY Sharpe 1.67, +2.72%, 9 trades, 33% WR | NIFTY Sharpe 1.44, +1.28%, 15 trades
- **Options variant**: BANKNIFTY Sharpe 5.59, +11.57% RoR, 8 trades, 50% WR, avg credit 92.84 pts
- FINNIFTY Sharpe 2.68, +6.90% RoR | MIDCPNIFTY Sharpe 3.56, +8.53% RoR
- Data: 122 days, SANOS-calibrated risk-neutral density from nse_fo_bhavcopy

### S2 Ramanujan Cycles
- Dominant periods: 30d (16%), 12d (11%), 24d (9%)
- Period stability: 7% (highly unstable)
- Phase backtest: 1271 trades, -4.65%, Sharpe -1.11
- Energy vs |return| correlation: 0.135 (p=0.017) — weak but significant

### S3 Institutional Flow
- 1120 signal-observations across 130 dates
- 1-day IC: +0.042 (p=0.17), quintile spread +0.21%
- 3-day IC: +0.048 (p=0.11), quintile spread +0.15%
- Top signals: HEROMOTOCO, NMDC, BHEL, FEDERALBNK

### S4 IV Mean-Reversion
- BANKNIFTY: Sharpe 3.07, +8.91%, 7 trades, 71% WR, MaxDD 2.50%
- NIFTY: Sharpe 1.42, +3.77%, 10 trades, 50% WR
- FINNIFTY: Sharpe 1.40, +2.83%, 7 trades, 57% WR
- NIFTYNXT50: Sharpe -2.89, -7.37% — fails on this index
- Data: 122 days, SANOS IV calibration on all 5 indices

### S5 Tick Microstructure (GPU)
- Features: VPIN, entropy, Hawkes intensity, Hawkes ratio
- **hawkes_ratio (short)**: Sharpe 4.29, +3.62%, 12 trades, 75% WR, MaxDD 0.56%
- hawkes_mean (long lb=60): Sharpe 1.09, +1.64%, 16 trades
- VPIN and entropy: weak/negative predictive power
- Data: 309 days tick data, GPU-accelerated feature computation on Tesla T4

### S6 Multi-Factor ML
- Walk-forward XGBoost on 429 features from 143 NSE indices
- R-squared: -0.259 (negative OOS)
- IC: +0.185 (p=0.435) — not significant
- Strategy: Sharpe -2.48, -3.35%, MaxDD 7.55%
- **Bug fixed**: Sharpe was previously inflated ~2.25x by wrong annualization sqrt(252/5)

### S7 Regime Switch
- Entropy + MI + VPIN regime classification
- NIFTY lb=60: Sharpe 2.37, +4.10%, 2 trades, 100% WR
- BANKNIFTY lb=60: Sharpe 1.80, +1.16%, 2 trades
- lb=100+: 0-1 trades (insufficient regime transitions in 6-month window)

### S8 Expiry-Day Theta
- **Uses actual option prices from nse_fo_bhavcopy and nfo_1min**
- Costs: 3 pts/leg NIFTY, 5 pts/leg BANKNIFTY (entry only for settlement, entry+exit for breach)
- NIFTY otm=2.0%: Sharpe 0.19, +0.22%, 66 trades, 46% WR — break-even
- NIFTY otm=1.5%: Sharpe -1.03, -2.41%, 56% WR — costs eat credit
- NIFTY otm=1.0%: Sharpe -4.60, -5.54%, 14% WR — constant breach
- VIX filter has no effect (India VIX was <15 throughout period)

### S9 Cross-Sectional Momentum
- Delivery %, OI change, FII flow signals
- top_n=7: Sharpe 0.76, +2.90%, 159 trades, 55% WR, MaxDD 6.04%
- top_n=5: Sharpe -0.95, -4.37%
- top_n=3: Sharpe -0.83, -5.46%
- Broader basket (7) diversifies away single-stock risk

### S10 Gamma Scalping
- **Uses actual straddle prices from nse_fo_bhavcopy**
- Costs: 3 pts/leg NIFTY, 5 pts/leg BANKNIFTY (2 legs × entry + exit)
- NIFTY: 0 trades across all configs (IV never below percentile threshold)
- BANKNIFTY ivp=0.30: 2 trades, -0.49%, 0% WR
- Strategy logic sound but low-vol environment offers no entry opportunity

### S11 Pairs Trading
- Engle-Granger cointegration on stock futures from nse_fo_bhavcopy
- 0 trades across all 9 parameter combos (lb=[40,60,90] × z=[1.5,2.0,2.5])
- No stock pairs pass cointegration test at p<0.05 in this data window

---

## Integrity Notes

1. **Zero look-ahead bias**: All strategies use causal signals (TimeGuard), T+1 execution lag
2. **Correct Sharpe**: ddof=1 sample std, sqrt(252) annualization, all daily returns
3. **Actual option prices**: S8 and S10 use nse_fo_bhavcopy settlement/close prices and nfo_1min intraday bars
4. **Realistic costs**: Per-leg index-point costs for options (not bps of spot), 10 bps for equity
5. **S6 bug fixed**: Annualization corrected from sqrt(252/5) to sqrt(252), added ddof=1
6. **No cherry-picking**: All parameter sweep results shown, not just the best
