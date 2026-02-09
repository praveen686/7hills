# RL for Finance — Analysis & Strategy Mapping

## Source: "Foundations of Reinforcement Learning with Applications in Finance"
**Authors**: Ashwin Rao & Tikhon Jelvis (Stanford CME 241)
**Evaluated**: 2026-02-08

---

## Book Quality Assessment: 8.5/10

Best single-volume treatment of RL applied to finance. Mathematically rigorous (full proofs
of Policy Gradient Theorem, HJB solutions, Bellman equations), includes production-quality
Python code, and covers the three domains we care about: **asset allocation, derivatives
pricing/hedging, and order-book trading**.

**Weakness**: Foundations text — models are simplified (GBM dynamics, exponential fill rates).
We extend every framework to handle India FnO + crypto realities.

---

## Chapter-by-Chapter Relevance

| Chapter | Topic | Relevance | Implementation |
|---------|-------|-----------|----------------|
| 2 | Markov Processes | Foundation | `core/markov_process.py` |
| 3 | MDP & Bellman | Foundation | `core/mdp.py` |
| 4-5 | Dynamic Programming | Foundation | `core/dynamic_programming.py` |
| 6 | Function Approximation | Critical | `core/function_approx.py` |
| 7 | Utility Theory | Reference | Used in sizing math |
| **8** | **Dynamic Asset Allocation** | **Direct** | `finance/asset_allocation.py` |
| **9** | **Derivatives Pricing/Hedging** | **Direct** | `finance/derivatives_pricing.py` |
| **10** | **Order-Book Trading** | **Direct** | `finance/optimal_execution.py`, `finance/market_making.py` |
| 11 | MC Prediction & Control | Foundation | `algorithms/monte_carlo.py` |
| 12 | TD Learning | Foundation | `algorithms/td_learning.py` |
| 13 | Q-Learning, DQN, LSPI | Critical | `algorithms/q_learning.py` |
| **14** | **Policy Gradient** | **Critical** | `algorithms/policy_gradient.py` |
| **15** | **Multi-Armed Bandits** | **Direct** | `algorithms/bandits.py` |
| 16 | Blending Learning & Planning | Reference | Future work |

---

## Part I: NEW Strategies from Book

### NEW-1: RL Dynamic Portfolio Allocation (Ch 8 + Ch 14)
- **Book**: `AssetAllocPG` — Merton's problem with REINFORCE/Actor-Critic
- **Our version**: Multi-asset allocation across S1-S12 + crypto strategies
- **State**: `(t, W_t, regime, VIX, skew, OFI)` enriched with 150+ mega features
- **Action**: Continuous allocation vector
- **Algorithm**: Actor-Critic with TD error (p433)
- **Closed-form benchmark**: π* = (μ-r)/(σ²γ) for CRRA utility

### NEW-2: Deep Hedging for Options (Ch 9)
- **Book**: `MaxExpUtility` — pricing/hedging in incomplete markets as MDP
- **Our version**: NN-based hedging for S1 VRP straddles, S4 IV MR, S10 Gamma Scalp
- **State**: `(spot, IV_surface, greeks, positions, time_to_expiry, VIX)`
- **Action**: Continuous delta-hedge adjustment
- **Algorithm**: DPG (p439) — deterministic policy for continuous hedge ratio
- **Expected**: +15-30% PnL vs Black-Scholes delta hedging (Buehler et al. 2018)

### NEW-3: RL Optimal Execution (Ch 10.2)
- **Book**: `OptimalOrderExecution` — backward-induction ADP
- **Bertsimas-Lo**: N*_t = R_t/(T-t) + h(t,β,θ,ρ)·X_t
- **Our version**: Optimal execution for large FnO orders + BTC/ETH
- **State**: `(mid_price, remaining_qty, spread, depth_imbalance, time_remaining)`
- **Impact model**: Calibrate β (temporary) and α (permanent) from tick data

### NEW-4: Avellaneda-Stoikov Market-Making (Ch 10.3)
- **Book**: Full HJB derivation of optimal bid/ask spreads
- **Key formulas**:
  - Pseudo-mid: Q_t^(m) = S_t - I_t·γ·σ²·(T-t)
  - Optimal spread: δ_bid* + δ_ask* = γ·σ²·(T-t) + (2/γ)·log(1+γ/k)
- **Primary target**: BTC/ETH perps on Binance (270ns via Rust SBE)
- **Parameters**: γ (risk aversion), σ (vol), k (fill rate decay)

### NEW-5: Thompson Sampling Strategy Selector (Ch 15)
- **Book**: Thompson Sampling with Gaussian-InvGamma posteriors (p462)
- **Our version**: Meta-strategy selecting among S1-S12 + crypto strategies
- **Context**: regime, VIX, expiry proximity, day-of-week
- **Achieves**: Lai-Robbins optimal regret bound (logarithmic)

---

## Part II: Augmenting Existing Strategies

| Strategy | Current | Book Upgrade | Chapter |
|----------|---------|-------------|---------|
| S7 Regime | Hardcoded mom/MR switch | MDP with learned switching policy | Ch 3-4 + Ch 8 |
| S1 VRP | Fixed position size | Kelly-Merton π* = (μ-r)/(σ²γ) | Ch 7-8 |
| S5 Hawkes | Fixed intensity threshold | Optimal stopping MDP (American option) | Ch 9 + Ch 14 |
| S6 XGBoost | Fixed size on prediction | Meta-labeling + Gradient Bandit sizing | Ch 15 |
| S10 Gamma | Fixed-interval hedging | Actor-Critic dynamic hedging | Ch 14.6 |

---

## Part III: Infrastructure (Rust Engine + Crypto)

### Rust Engine (`qlxleg/QuantLaxmi/`)
- 21 Rust crates, IEEE 1016-2009 compliant
- LMAX Disruptor event bus (~230ns latency)
- Mantissa-based fixed-point (deterministic replay)
- SANOS arbitrage-free surfaces (certified)
- HYDRA multi-expert ensemble (Exponentiated Gradient)
- 7 production gates (G0-G7)
- Zerodha WS + Binance SBE connectors

### Crypto (`qlxr_crypto/` + `alpha_forge/`)
- Binance REST + JSON WS + SBE binary connectors
- 145MB downloaded data (BTC, ETH, SOL; 1m-1d)
- Active: crypto lead-lag strategy (1090 lines)
- Archive: 4-signal engine (funding carry, cascade, MR, news)
- 24/7 trading = 5.6x more RL training episodes than India FnO

### Why Both Markets
| Factor | India FnO | Crypto |
|--------|-----------|--------|
| Hours | 6.25/day | 24/7 |
| RL episodes/year | ~250 | ~1400 |
| Execution latency | 50-100ms | 270ns (SBE) |
| Market-making | Marginal | Excellent |
| Funding carry | N/A | Persistent premium |

---

## Part IV: TFT + RL Integration

### Current TFT Status
- 209 mega features, 11 data groups
- NIFTY: Sharpe 1.41 (315 OOS days)
- BANKNIFTY: -0.95 (feature sparsity + NIFTY-centric features)
- Walk-forward: seq_len=20, train=60d, test=15d

### Integration Patterns

**Pattern 1: TFT as State Encoder**
- Pre-train TFT (supervised), freeze backbone
- Extract 256-dim hidden state h_t
- Attach Actor-Critic head (small MLP)
- RL trains on top of learned representation

**Pattern 2: TFT + Contextual Thompson Sampling**
- TFT outputs prediction + confidence
- Thompson Sampling learns when to trust TFT
- Arms: {flat, small/med/large × long/short}

**Pattern 3: TFT + Deep Hedging**
- TFT provides directional view
- RL decides optimal options structure
- Deep Hedging manages Greeks dynamically

**Pattern 4: Cross-Market TFT + RL**
- Separate TFTs for India and crypto
- RL allocation layer on top
- Learns cross-market correlation structure

**Pattern 5: TFT Attention as Reward Shaping**
- Attention weights spike at regime changes
- Use as auxiliary reward for faster RL convergence

---

## Priority Ranking

| # | Item | Market | Effort | Impact |
|---|------|--------|--------|--------|
| 1 | Thompson Sampling meta-allocator | Both | 3 days | +25% portfolio Sharpe |
| 2 | Avellaneda-Stoikov MM (Rust+RL) | Crypto | 1 week | 24/7 revenue |
| 3 | Deep Hedging for S1/S10 | India | 1 week | +20% options PnL |
| 4 | RL Optimal Execution | Both | 1 week | -30% slippage |
| 5 | Kelly-Merton sizing for S1 | India | 2 days | +15% Sharpe |
| 6 | TFT + Contextual Bandit | Both | 4 days | Better than fixed threshold |
| 7 | Funding Carry Actor-Critic | Crypto | 5 days | Proven premium |
| 8 | Cross-market regime RL | Both | 1 week | Correlation alpha |
| 9 | TFT backbone + AC head | Both | 2 weeks | Full integration |
| 10 | Deploy in Rust via ONNX | Both | 2 weeks | Production latency |
