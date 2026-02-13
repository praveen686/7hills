# QuantLaxmi Strategy Profitability & TFT/RL Integration Analysis (Feb 2026)

## 1. Strategy Landscape — 26 Strategies at a Glance

| Tier | Strategies | Performance | Status |
|---|---|---|---|
| **Tier 1: Live-Ready** | S1, S25, S4, S5 | Sharpe > 1.2 | Highest Priority |
| **Tier 2: Promising** | S7, S11, S12 | Sharpe 0.6 - 1.1 | Needs scaling/data fixes |
| **Tier 3: No Edge** | S2, S6, S9, S10 | Sharpe < 0 | Deprioritized |
| **Tier 4: Incomplete** | S3, S8, S13-S24, S26 | - | Stubs/Broken |

---

## 2. Profitable Strategies — Deep Analysis

### Tier 1: Live-Ready

| # | Strategy | Best Sharpe | Return% | MaxDD% | Edge Source |
|---|----------|-------------|---------|--------|-------------|
| **S1** | **VRP Options** | **2.90** | +101% | — | Risk-neutral density divergence; Bull Put Spreads |
| **S25**| **Div Flow Field** | **2.16** | +8.06% | 2.1% | Helmholtz decomposition of NSE OI flows (NIFTY/BNF) |
| **S4** | **IV Mean-Rev** | **1.28** | +21.6% | 6.0% | SANOS IV surface vs realized vol z-score |
| **S5** | **Hawkes Jump** | **1.23** | +9.60% | 5.9% | Hawkes intensity from 1-min bars; Short bias |

**Ensemble Alpha**: S25 + S4 + S5 inverse-vol → **Sharpe 1.90**, MaxDD **1.20%**.

---

## 3. TFT X-Trend Integration

The existing **X-Trend TFT** model (found in `quantlaxmi/models/ml/tft/`) is currently in **Phase 5 Production Training**.

- **Performance**: val_sharpe = 2.24 (current training status).
- **Features**: Consumes 292 features, pruned to 73 via VSN (Variable Selection Network).
- **Architecture**: VSN → LSTM → Cross-Attention.
- **Assets**: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, BTCUSDT, ETHUSDT.

### Roadmap for TFT
1. **Signal Filtering**: Feed DFF (S25) and IV (S4) features explicitly into TFT to learn regime-dependent scaling.
2. **Confidence Gating**: Use TFT's probability density output to gate entries for Tier 1 strategies.

---

## 4. Reinforcement Learning (RL) Roadmap

The project contains a comprehensive plan for RL integration (`docs/strategies/RL_INTEGRATION_PLAN.md`).

### Quick Wins (Existing Opt-in Flags)
These are already implemented in the code but set to `False` by default:
- **S1 VRP**: `use_kelly=True` (Kelly-Merton sizing).
- **S7 Regime**: `use_mdp=True` (MDP-based regime switching).
- **S10 Gamma**: `use_deep_hedge=True` (Deep Hedging agent for delta rebalancing).

### Strategic Integrations
1. **Thompson Sizing** (Pattern 2): Contextual bandits for ensemble weighting.
2. **Deep Hedging** (Pattern 3): Neural hedge for S1 options based on real IV paths.
3. **Execution Edge** (NEW-3): RL-based optimal execution calibrated from `kite_depth` L5 order book.

---

## 5. Development Roadmap (Priority Order)

1. **Verify Phase 1**: Run backtests with `use_kelly`, `use_mdp`, and `use_deep_hedge` enabled.
2. **Data Fixes**: Fix S8 (schema bug) and S6 (XGBoost data scaling) to unlock more alpha.
3. **Ensemble Weighting**: Implement dynamic RL-based weighting for S25+S4+S5.
4. **Intraday TFT**: Port X-Trend architecture to 1-min bar microstructure for S14 (OFI).
