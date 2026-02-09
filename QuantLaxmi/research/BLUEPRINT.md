# AlphaForge — Next-Gen Trading System Blueprint

## Research Synthesis (8 agents, 50+ web searches, 2026-02-08)

### Mission
Build genuinely profitable strategies for India FnO (NIFTY, BANKNIFTY, stock F&O) using
state-of-the-art quant techniques. Every strategy must survive honest validation (CPCV,
walk-forward, placebo tests).

---

## Top Strategies to Implement (Ranked by Expected Edge)

### 1. HMM Regime-Switching Momentum/MR Strategy
- **Approach**: Hidden Markov Model detects 3 regimes (bull, bear, neutral). Switch between
  momentum (in bull) and mean-reversion (in bear/neutral). Proven on NIFTY50 with Sharpe 1.05.
- **Edge source**: Regime persistence — markets trend in regimes, MR between regimes.
- **Data**: Daily index close + 1-min bars for intraday confirmation.
- **Key reference**: "Regime-Aware Short-Term Trading Using HMM + Monte Carlo" (2025)

### 2. Momentum Transformer (TFT + Changepoint Detection)
- **Approach**: Temporal Fusion Transformer with variable selection, LSTM encoder, multi-head
  attention. Integrated CPD (changepoint detection via `ruptures`) improves Sharpe by 33-66%.
- **Edge source**: Attention mechanism peaks at momentum turning points; blends mom/MR automatically.
- **Data**: Daily OHLCV + technical indicators (40+ features).
- **Key reference**: "Trading Momentum Transformer" (kieranjwood, GitHub)

### 3. Order Flow Imbalance (OFI) Intraday Strategy
- **Approach**: Compute multi-level OFI from 5-level depth data. Predict short-term price
  moves (1-5 min). Combine with Hawkes process intensity for trade clustering detection.
- **Edge source**: Informed flow creates persistent imbalance before price moves.
- **Data**: 5-level depth (zerodha/5level/), tick data, 1-min bars.
- **Key reference**: "Order Book Filtration and Directional Signal Extraction at HF" (arXiv 2507)

### 4. Volatility Skew Mean-Reversion
- **Approach**: Track 25-delta put-call skew z-score. Enter risk reversal when skew spikes
  (z > 2.0). Exit when skew normalizes (z crosses 0).
- **Edge source**: Institutional hedging demand creates inelastic put-buying; skew overshoots
  and reverts as fear decays.
- **Data**: Option chain IV from SANOS calibration (existing infra).
- **Key reference**: Existing iv_engine.py `iv_skew_25d`

### 5. Enhanced VRP Harvesting (S1 Upgrade)
- **Approach**: Short straddle when VRP > 95th percentile, with HAR-RV forecast, density
  tail-gating (skip when left_tail > 0.20), and event-calendar overlay.
- **Edge source**: VRP is the most documented persistent risk premium. India VIX premium
  is 3-10 vol points on average.
- **Data**: SANOS density, HAR-RV, India VIX.

### 6. Graph-Mamba (SAMBA) Cross-Asset Prediction
- **Approach**: Mamba SSM for temporal patterns + GNN for cross-stock relationships.
  Bidirectional Mamba + Chebyshev convolutions. Captures NIFTY component correlations.
- **Edge source**: Inter-stock lead-lag relationships; O(n) complexity enables long sequences.
- **Data**: NIFTY 50 component daily returns + 82 technical features.
- **Key reference**: "Mamba Meets Financial Markets" (arXiv 2410.03707)

---

## Infrastructure to Build

### AFML Core (Lopez de Prado)
1. **Combinatorial Purged Cross-Validation (CPCV)** — honest validation for all strategies
2. **Triple Barrier Labeling** — volatility-adjusted labels for ML model training
3. **Hierarchical Risk Parity (HRP)** — multi-strategy capital allocation
4. **CUSUM Event Filter** — noise reduction for event sampling
5. **Fractional Differentiation** (enhance existing d=0.226 with auto-d selection)

### Execution Enhancement
- **Deep Hedging** (pfhedge) — NN-based optimal hedging overlay for options strategies
- **Kelly Criterion with Uncertainty** — bet sizing with confidence intervals

---

## Data Assets Available
- Tick data: 311 days (2024-10-29 to 2026-02-06), ~8GB
- 1-min bars (NFO + BFO): 316-317 days, ~1.6GB
- Daily NSE data: 125+ days (28 categories: bhavcopy, IV, delta, OI, FII/DII, delivery, etc.)
- 5-level depth: NIFTY/BANKNIFTY futures + options
- Option chains: Live snapshots (617 per day)
- Instruments: 311 daily masters

## Existing Infrastructure to Leverage
- SANOS vol surface calibration (core/pricing/sanos.py)
- IV engine with Greeks (core/pricing/iv_engine.py) — GPU-accelerated
- Microstructure features: VPIN, Kyle's Lambda, Amihud, Hawkes (core/features/microstructure.py)
- Information theory: entropy, mutual info (core/features/information.py)
- Fractional differentiation (core/features/fractional.py)
- Regime detector (strategies/s7_regime/detector.py)
- Risk-neutral density analysis (core/pricing/risk_neutral.py)
- DuckDB MarketDataStore with 28+ views
- FastAPI backend (19 routes) + Next.js frontend (8 pages)
- 992 passing tests

## Architecture
```
alpha_forge/
├── BLUEPRINT.md          ← this file
├── strategies/
│   ├── hmm_regime.py     ← HMM regime-switching strategy
│   ├── momentum_tfm.py   ← Momentum Transformer (TFT + CPD)
│   ├── ofi_intraday.py   ← Order Flow Imbalance intraday
│   ├── skew_mr.py        ← Volatility skew mean-reversion
│   ├── vrp_enhanced.py   ← Enhanced VRP harvesting
│   └── graph_mamba.py    ← Graph-Mamba cross-asset (future)
├── infra/
│   ├── cpcv.py           ← Combinatorial Purged CV
│   ├── triple_barrier.py ← Triple Barrier labeling
│   ├── hrp.py            ← Hierarchical Risk Parity
│   ├── cusum.py          ← CUSUM event filter
│   └── meta_label.py     ← Meta-labeling
├── research/
│   └── run_all.py        ← Master backtest runner
├── backtests/            ← Backtest output files
└── results/              ← Scorecard and analysis
```
