# QuantKubera Monolith v2 — Enhanced

## Institutional-Grade Momentum Transformer for Indian Derivatives

**Self-contained** | **Zero Look-Ahead Bias** | **Walk-Forward OOS Validation**

| Component | Reference | Features |
|-----------|-----------|----------|
| Temporal Fusion Transformer | Lim et al. (2021) | VSN, Interpretable MHA, GRN |
| AFML Event Pipeline | Lopez de Prado (2018) | CUSUM, Triple Barrier, Meta-Labeling |
| Fractional Differentiation | Hosking (1981) | Memory-preserving stationarity |
| Ramanujan Sum Filter Bank | Planat (2002) | Integer-period cycle detection |
| NIG Changepoint Detection | Adams & MacKay (2007) | Regime shift scoring |
| Market Microstructure | Easley et al. (2012), Kyle (1985) | VPIN, Lambda, Amihud |
| Information-Theoretic Entropy | Shannon, Masters | Predictability regime filter |
| Garman-Klass / Parkinson Vol | GK (1980), Parkinson (1980) | Efficient OHLC volatility |

### Pipeline
```
Zerodha Kite API → 31 Features (10 groups) → Variable Selection Network
GDELT News → FinBERT → 9 Sentiment Features ↗
India VIX → 3 VIX Features ↗
→ LSTM Encoder → Interpretable Multi-Head Attention → Momentum Signal
→ Walk-Forward OOS (purge gaps) → Meta-Labeling → Probit Bet Sizing
→ VectorBTPro Tearsheet
```

### Feature Groups (31 per-ticker + 12 cross-asset)
1. **Normalized Returns** (5): Vol-normalized multi-horizon returns
2. **MACD** (3): Z-scored momentum oscillators
3. **Volatility** (4): Realized, Garman-Klass, Parkinson estimators
4. **Changepoint Detection** (2): NIG Bayesian regime shifts
5. **Fractional Calculus** (3): Hosking differencing + Hurst exponent
6. **Ramanujan Periodogram** (4): Weekly/biweekly/monthly/quarterly cycles
7. **Microstructure** (4): VPIN, Kyle's λ, Amihud, HL spread
8. **Information Theory** (1): Shannon entropy of price patterns
9. **Momentum Quality** (3): Trend strength, consistency, mean-reversion
10. **Volume** (2): Volume z-score and momentum
11. **News Sentiment** (9): FinBERT scores from GDELT headlines (T+1 lag)
12. **India VIX** (3): Fear gauge z-score, 5d change, mean-reversion