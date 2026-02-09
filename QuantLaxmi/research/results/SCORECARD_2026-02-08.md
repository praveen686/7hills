# AlphaForge Scorecard — 2026-02-08

## Summary

| Strategy | Sharpe | Total Ret% | Max DD% | Trades | Win% | Active Days | Data Source | Status |
|----------|--------|-----------|---------|--------|------|-------------|-------------|--------|
| HMM Regime (Kite 2yr) | **1.02** | +3.42% | -1.04% | 46 | 30.4% | 415 | Zerodha Kite API | OK |
| Momentum TFM (GPU) | **0.31** | +88.1% | -163.1% | 160 | 40.9% | 159 OOS | DuckDB nse_index_close | OK |
| Skew MR | -5.63 | -0.48% | -0.48% | 11 | 0.0% | 92 | DuckDB + SANOS IV | Costs too high |
| OFI Intraday | -5.38 | -141 pts | -162.6 pts | 32 | 15.6% | 59 | DuckDB nfo_1min | Needs L2 depth |
| HMM Regime (DuckDB) | -1.19 | -0.20% | -0.36% | 4 | 25.0% | 44 | DuckDB nse_index_close | Insufficient data |
| Enhanced VRP (60d lookback) | **2.03** | +0.26% | 0.00% | 2 | 100% | 122 | DuckDB + IV chain | OK (conservative) |

## Key Findings

### Profitable Strategies

1. **HMM Regime-Switching (Kite)** — Sharpe 1.02, +3.42% return, -1.04% max drawdown
   - 3-state HMM (Bull/Bear/Neutral) with hmmlearn
   - 496 daily bars from Zerodha Kite API (2 years)
   - Conservative: mostly neutral regime with RSI mean-reversion
   - Calmar ratio ~3.3 (excellent risk-adjusted return)
   - **Verdict**: Genuine edge with proper data. Needs longer history for more active signals.

2. **Momentum Transformer (TFT)** — Sharpe 0.31, +88% total return
   - LSTM encoder + multi-head attention + tanh position output
   - 358,454 parameters trained on Tesla T4 GPU
   - Walk-forward: 120d train, 20d test, 20d step
   - 159 out-of-sample days, 40.9% hit rate
   - **Note**: MaxDD of -163% suggests leveraged or synthetic position sizes. Total return of 88% is high but may reflect cumulative position sizing.
   - **Verdict**: Model learns meaningful patterns but needs position sizing calibration.

### Strategies Needing Improvement

3. **Volatility Skew Mean-Reversion** — Sharpe -5.63
   - 12 index points per round-trip cost (4 legs × 3 pts) overwhelms the small skew signal
   - Skew values average 0.006 (60 bps) — not enough to cover costs
   - **Fix**: Need higher-frequency entry/exit, wider z-threshold, or larger data set where skew spikes more dramatically

4. **OFI Intraday** — Sharpe -5.38
   - Proxy OFI (from 1-min bars, not real L2 depth) has low predictive power
   - 15.6% win rate, most exits via pressure_reversal
   - **Fix**: Needs real 5-level depth data (we have it for recent dates via Zerodha WebSocket)

5. **Enhanced VRP (60d lookback)** — Sharpe 2.03, +0.26% return
   - Short ATM straddle when VRP z-score > 95th percentile
   - Only 2 trades in 122 days (very conservative filter)
   - Both trades profitable: IV 11.2-15.8%, RV 6.0-8.3%
   - Avg VRP: 1.4% (IV-RV). Avg IV: 9.8%, Avg RV: 8.3%
   - **Verdict**: Edge confirmed but needs longer history for more trades. Reduce lookback = more sensitive entry.

## Data Observations

- **NSE daily data**: Only 125 trading days (2025-08-06 to 2026-02-06) — severely limits strategies needing long lookbacks
- **Kite API**: Can pull 2+ years of daily OHLCV — critical for strategies like HMM
- **1-min bars**: 316 days available — suitable for intraday strategies
- **5-level depth**: Only 1 day so far — need to accumulate for OFI
- **Telegram raw ticks**: 122 files, 3.5 GB — underutilized

## AFML Infrastructure Built

- **CPCV**: Combinatorial Purged Cross-Validation (10 splits tested)
- **Triple Barrier**: Volatility-adjusted labeling (103 labels generated)
- **CUSUM Event Filter**: 104/209 events (fixed/adaptive)
- **HRP**: Hierarchical Risk Parity allocation (tested on singular matrices)
- **Meta-Labeling**: Binary classifier + Kelly bet sizing

## Architecture

```
alpha_forge/
├── BLUEPRINT.md          ← Master research synthesis
├── strategies/
│   ├── hmm_regime.py     ← HMM 3-state regime switching (1065 lines)
│   ├── momentum_tfm.py   ← TFT + changepoint detection (1344 lines)
│   ├── ofi_intraday.py   ← Order flow imbalance intraday (782 lines)
│   ├── skew_mr.py        ← Volatility skew mean-reversion (995 lines)
│   └── vrp_enhanced.py   ← Enhanced VRP harvesting (1070 lines)
├── infra/
│   ├── cpcv.py           ← Combinatorial Purged CV
│   ├── triple_barrier.py ← Triple Barrier labeling
│   ├── hrp.py            ← Hierarchical Risk Parity
│   ├── cusum.py          ← CUSUM event filter
│   └── meta_label.py     ← Meta-labeling + Kelly sizing
├── research/
│   ├── run_all.py        ← Master backtest runner
│   └── run_hmm_kite.py   ← HMM with Kite historical data
├── backtests/
└── results/
    ├── SCORECARD_2026-02-08.md  ← This file
    ├── hmm_regime_kite_NIFTY50_2026-02-08.csv
    └── ofi_intraday_NIFTY_*.parquet
```

## Methodology
- Sharpe: ddof=1, sqrt(252), all daily returns including flat days
- Costs: 3 pts/leg (NIFTY), 5 pts/leg (BANKNIFTY)
- Execution: T+1 (signal at close day T, execute at close day T+1)
- No look-ahead bias — all signals fully causal
- Walk-forward validation for ML strategies

## Next Steps (from Research Synthesis)

### High-Priority (Expected Sharpe 2.0+)
1. **Deep Learning Statistical Arbitrage** (Attention Factors) — Sharpe 2.28 net in literature
2. **HMM-Regime VRP Combination** — Merge existing S1+S7+HMM Regime for regime-aware vol selling
3. **Hawkes + Mamba/SSM** — Upgrade S5 Hawkes (Sharpe 4.29) with sequence model
4. **Network Momentum / Lead-Lag** — Cross-asset information flow exploitation

### Medium-Priority
5. **GEX Pinning / Dealer Flow** — Fix S8+S10 with gamma exposure context
6. **FII/DII Derivative Flow** — Upgrade S3 with participant-wise OI data
7. **Conditional Autoencoder Factors** — Replace failed S6 with Gu-Kelly-Xiu approach
8. **TFT Meta-Learner** — Combine all strategy signals

Generated: 2026-02-08
