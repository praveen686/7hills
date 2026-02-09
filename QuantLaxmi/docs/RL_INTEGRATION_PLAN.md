# Plan: Complete RL Integration — All ANALYSIS.md Patterns + Options Data from DuckDB

## Context

**Previous work (DONE)**: Pattern 1 (TFT as State Encoder) + NEW-1 (Actor-Critic) + NEW-5 (Thompson) + Kelly-Merton sizing — 1180 tests passing, backtest running.

**Problem**: 6 remaining patterns + 5 strategy augmentations from `docs/ANALYSIS.md` are unimplemented. Options data from DuckDB is not fed into TFT. All building blocks (5 agents, 5 environment files, 4 finance modules, all algorithms) already exist with 18,665 lines — they just need **integration pipelines** wiring them end-to-end with real data.

**What the user wants**:
1. ALL remaining ANALYSIS.md patterns implemented (Patterns 2-5, NEW-2/3/4)
2. Options chain data from DuckDB piped into TFT via new feature group
3. 5 strategy augmentations (S1 Kelly, S5 optimal stopping, S6 gradient bandit, S7 MDP regime, S10 deep hedging)
4. **ZERO stubs** — every function fully implemented, institutional grade
5. **Real data** — DuckDB option chains, kite_depth, Binance OHLCV — no synthetic
6. Training + backtest after implementation

---

## Architecture Overview

```
DuckDB (nse_fo_bhavcopy, nfo_1min, ticks, kite_depth, binance)
         ↓
OptionsFeatureBuilder (16 new options features: ATM IV, skew, PCR, VRP, gamma exposure...)
         ↓
MegaFeatureBuilder (~303 features = 287 existing + 16 options)
         ↓
XTrendBackbone (VSN → LSTM → CrossAttention → d_hidden)
         ↓
┌──────────────────────────────────────────────────────────────────────┐
│ 7 Integration Pipelines                                              │
│                                                                      │
│ P1 ✅ Actor-Critic (DONE)      P2 Contextual Thompson Sizing        │
│ P3 TFT + Deep Hedging          P4 Cross-Market India+Crypto         │
│ P5 Attention Reward Shaping     N2 Deep Hedging for S1/S10          │
│ N3 RL Optimal Execution         N4 Avellaneda-Stoikov MM            │
└──────────────────────────────────────────────────────────────────────┘
         ↓
5 Strategy Augmentations: S1 Kelly, S5 Stopping, S6 Bandit, S7 MDP, S10 DH
```

---

## Existing Code REUSED (NOT reimplemented)

| Component | File | Lines | What we use |
|-----------|------|-------|-------------|
| DeepHedgingAgent | `models/rl/agents/deep_hedger.py` | 521 | `train_on_paths()`, `get_hedge_action()`, `compare_vs_bs()` |
| OptimalExecutionAgent | `models/rl/agents/execution_agent.py` | 535 | `execute_order()`, `train()`, `benchmark_vs_twap()` |
| MarketMakingAgent | `models/rl/agents/market_maker.py` | 546 | `get_quotes()`, `train()`, `avellaneda_stoikov_quotes()` |
| KellySizer | `models/rl/agents/kelly_sizer.py` | 455 | `optimal_size()`, `drawdown_adjustment()` |
| ThompsonStrategyAllocator | `models/rl/agents/thompson_allocator.py` | 469 | `select_allocation()`, `update()`, `_NeuralContextualBandit` |
| OptionsEnv + GammaScalpEnv | `models/rl/environments/options_env.py` | 702 | Full Greeks in state, BS pricing |
| ExecutionEnv + LOBSimulator | `models/rl/environments/execution_env.py` | 583 | Market impact, order book |
| CryptoEnv + CryptoLeadLagEnv | `models/rl/environments/crypto_env.py` | 554 | BTC/ETH perps, funding, lead-lag |
| AssetAllocationMDP + Merton | `models/rl/finance/asset_allocation.py` | — | `MertonSolution`, `AssetAllocPG` |
| OrderExecutionMDP + A-C | `models/rl/finance/optimal_execution.py` | — | `AlmgrenChrissSolution`, `BertsimasLoSolution` |
| MarketMakingMDP + A-S | `models/rl/finance/market_making.py` | — | `AvellanedaStoikovSolution` |
| GradientBandit | `models/rl/algorithms/bandits.py` | — | `GradientBandit` for S6 |
| DynamicProgramming | `models/rl/core/dynamic_programming.py` | — | `value_iteration` for S7 MDP |
| MarketDataStore | `data/store.py` | — | `get_option_chain()`, DuckDB views |
| BinanceConnector | `data/connectors/binance_connector.py` | — | `fetch_klines()` |

---

## File Changes

### NEW Files (9)

| # | File | Purpose | ~Lines |
|---|------|---------|--------|
| 1 | `features/options_features.py` | Options features from DuckDB: ATM IV, skew, PCR, VRP, gamma exposure, theta rate, max pain, OI walls | ~450 |
| 2 | `models/rl/integration/thompson_sizing.py` | Pattern 2: TFT confidence → Contextual Thompson arms {flat, ±0.25, ±0.5, ±1.0} + GradientBanditSizer for S6 | ~500 |
| 3 | `models/rl/integration/deep_hedging_pipeline.py` | Pattern 3 + NEW-2: TFT directional view + Deep Hedging on real IV paths from DuckDB | ~600 |
| 4 | `models/rl/integration/cross_market.py` | Pattern 4: Dual India+Crypto backbones + RL allocation layer + Merton benchmark | ~550 |
| 5 | `models/rl/integration/attention_reward.py` | Pattern 5: Cross-attention weight spikes → auxiliary reward for faster RL convergence | ~350 |
| 6 | `models/rl/integration/execution_pipeline.py` | NEW-3: RL Optimal Execution calibrated from kite_depth + Hawkes optimal stopping for S5 | ~450 |
| 7 | `models/rl/integration/market_making_pipeline.py` | NEW-4: Avellaneda-Stoikov MM on BTC/ETH perps from Binance data | ~450 |
| 8 | `tests/test_rl_patterns.py` | 35 tests for all new patterns + strategy augmentations | ~900 |
| 9 | `tests/test_options_features.py` | 7 tests for options feature builder | ~200 |

### MODIFIED Files (7)

| # | File | Changes |
|---|------|---------|
| 10 | `features/mega.py` | Add `_build_extended_options_features()` calling OptionsFeatureBuilder, wire into builders list (+15 lines) |
| 11 | `models/ml/tft/x_trend.py` | Add `forward_with_weights()` to CrossAttentionBlock + `extract_hidden_with_attention()` to XTrendModel (+40 lines) |
| 12 | `models/rl/integration/integrated_env.py` | Add `set_reward_shaper()` hook + reward bonus in `step()` (+30 lines) |
| 13 | `models/rl/integration/__init__.py` | Export all new modules (+25 lines) |
| 14 | `strategies/s1_vrp/strategy.py` | Add Kelly-Merton sizing via KellySizer (opt-in flag `use_kelly=False`) (+45 lines) |
| 15 | `strategies/s7_regime/strategy.py` | Add MDP switching policy via value_iteration (opt-in flag `use_mdp=False`) (+55 lines) |
| 16 | `strategies/s10_gamma_scalp/strategy.py` | Add Actor-Critic dynamic hedging via DeepHedgingAgent (opt-in flag `use_deep_hedge=False`) (+50 lines) |

---

## Detailed Design per Pattern

### 1. `features/options_features.py` — Options Data from DuckDB

**OptionsFeatureBuilder**: Loads nse_fo_bhavcopy daily, computes 16 causal features:

| Feature | Description | Source |
|---------|-------------|--------|
| `optx_atm_iv` | ATM IV via Newton-Raphson BS inversion | ATM straddle price |
| `optx_iv_skew_25d` | 25-delta put IV / 25-delta call IV | Interpolated chain |
| `optx_pcr_vol` | Put/Call volume ratio | Daily volume |
| `optx_pcr_oi` | Put/Call OI ratio | Open interest |
| `optx_term_slope` | 2nd expiry ATM IV / 1st expiry ATM IV | Two nearest expiries |
| `optx_vrp` | Implied vol − Realized vol (20d) | ATM IV − close-to-close RV |
| `optx_net_gamma` | Net dealer gamma exposure (assumes short options) | OI × gamma per strike |
| `optx_theta_rate` | Portfolio theta / notional | Aggregate theta |
| `optx_oi_pcr_zscore_21d` | PCR OI z-score vs 21d rolling | Rolling stats |
| `optx_iv_rv_ratio` | IV / RV ratio | Ratio |
| `optx_skew_zscore_21d` | Skew z-score vs 21d rolling | Rolling stats |
| `optx_gamma_zscore_21d` | Gamma exposure z-score vs 21d | Rolling stats |
| `optx_put_wall` | Strike with max put OI (distance from spot) | OI distribution |
| `optx_call_wall` | Strike with max call OI (distance from spot) | OI distribution |
| `optx_max_pain_dist` | Max pain strike distance from spot | OI-weighted |
| `optx_iv_term_contango` | Binary: term structure in contango (1) or backwardation (0) | Term slope sign |

**Data flow**: `nse_fo_bhavcopy` (SQL via DuckDB) → per-date chain → BS Greeks → aggregate features → DataFrame

**Key methods**:
- `_load_chain()` — SQL: `SELECT * FROM nse_fo_bhavcopy WHERE FinInstrmTp IN ('IDO','STO') AND ...`
- `_compute_atm_iv()` — Newton-Raphson, 50 iter cap, fallback to straddle approx
- `_compute_net_gamma_exposure()` — `sum(OI_call * gamma - OI_put * gamma) * lot_size`
- `_compute_max_pain()` — strike minimizing total option buyer PnL

### 2. `models/rl/integration/thompson_sizing.py` — Pattern 2

**ThompsonSizingAgent**: 7 discrete arms = {0, ±0.25, ±0.5, ±1.0}
- Context (10-dim): [regime, vix, dte, dow_sin, dow_cos, tft_confidence, hidden_mean, hidden_std, recent_sharpe_5d, drawdown]
- Uses existing `ThompsonStrategyAllocator` with arm_names = sizing levels
- Uses `_NeuralContextualBandit` for context-dependent arm selection
- Posterior: NIG (Normal-Inverse-Gamma) per arm, updated daily with realized returns

**ThompsonSizingPipeline**: Walk-forward
1. Pretrain backbone (shared)
2. Extract `tft_position` (Gaussian mu) and `tft_confidence` (1/sigma) per day
3. Build context, select arm, observe return, update posterior
4. OOS evaluation: Thompson-sized vs uniform-sized Sharpe

**GradientBanditSizer** (for S6): Arms = {0.1, 0.25, 0.5, 0.75, 1.0}, uses `GradientBandit` from `algorithms/bandits.py`

### 3. `models/rl/integration/deep_hedging_pipeline.py` — Pattern 3 + NEW-2

**TFTDeepHedgingPipeline**: TFT says direction → DeepHedgingAgent picks structure + hedges
- `load_historical_iv_paths()` — real ATM IV from DuckDB nse_fo_bhavcopy (Newton-Raphson per date)
- `_build_iv_augmented_paths()` — bootstrap intraday from daily: 78 steps/day, GBM micro-steps conditioned on daily endpoints
- Training: `DeepHedgingAgent.train_on_paths(spot_paths, iv_paths)` with CVaR objective
- Walk-forward: TFT hidden → directional signal, DH manages Greeks

**StandaloneDeepHedger**: NEW-2 for S1 VRP straddles / S10 gamma scalp
- `train_on_historical()` — trains on real spot+IV paths from DuckDB
- `compare_vs_bs()` — OOS PnL variance comparison vs Black-Scholes delta hedging
- Environments: `GammaScalpEnv` (S10), `IVMeanReversionEnv` (S4), custom straddle env (S1)

### 4. `models/rl/integration/cross_market.py` — Pattern 4

**CryptoFeatureAdapter**: Loads Binance OHLCV → ~30 features per crypto asset
- Returns, vol, momentum, funding rate proxy, BTC dominance, overnight gap

**CrossMarketBackbone**: Two `XTrendBackbone` instances (India + Crypto)
- `extract_joint_hidden()` → concatenated (6 * d_hidden) vector

**CrossMarketAllocator**: Actor-Critic on `AssetAllocationMDP`
- 6 assets: [NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, BTC, ETH]
- Benchmark: `MertonSolution` analytical allocation π* = (1/γ) Σ^{-1} (μ − r·1)
- Learns cross-market correlation structure (BTC leads NIFTY by 6-12h)

**CrossMarketPipeline**: Walk-forward
1. Build India (MegaFeatureAdapter) + Crypto (CryptoFeatureAdapter) features
2. Pretrain both backbones independently
3. Train RL allocation (Actor-Critic, 200 episodes per fold)
4. OOS: compare RL vs Merton analytical

### 5. `models/rl/integration/attention_reward.py` — Pattern 5

**AttentionRewardShaper**: Extracts cross-attention weights from `XTrendModel`
- Requires `forward_with_weights()` added to `CrossAttentionBlock`
- Computes attention entropy per step, detects spikes (z-score > 2.0 below rolling mean)
- Spikes signal regime changes → auxiliary reward bonus

**AttentionShapedEnv**: Wrapper around `IntegratedTradingEnv`
- `step()` adds `bonus_scale * attention_spike` to base reward
- `precompute_bonuses()` pre-computes for entire fold (no step-level overhead)

### 6. `models/rl/integration/execution_pipeline.py` — NEW-3 + S5

**ExecutionCalibrator**: Calibrates impact params from kite_depth L5 order book
- `estimate_impact_params()` → {alpha_perm, beta_temp, sigma, spread, depth_mean, fill_rate_k}
- Regression: price_impact = alpha * cumulative_volume + beta * trade_size

**OptimalExecutionPipeline**: Train `OptimalExecutionAgent` on calibrated `ExecutionEnv`
- Benchmark vs TWAP and `AlmgrenChrissSolution`
- Walk-forward: calibrate per fold, train 5000 episodes, evaluate OOS

**HawkesOptimalStopping** (S5 augmentation): Finite-horizon MDP
- States: (intensity_bin, signal_strength_bin, days_held)
- Actions: {hold, exit}
- Solved via `value_iteration` from `core/dynamic_programming.py`

### 7. `models/rl/integration/market_making_pipeline.py` — NEW-4

**CryptoMMCalibrator**: Calibrates A-S params from Binance OHLCV
- `calibrate()` → {sigma, fill_rate_k, gamma, T_session, mean_spread}

**MarketMakingPipeline**: Train `MarketMakingAgent` on `CryptoEnv`
- A-S analytical benchmark: `avellaneda_stoikov_quotes()` from `market_maker.py`
- RL fine-tuning: warm-start from A-S, then Actor-Critic training
- Metrics: PnL, Sharpe, max inventory, fill rate, spread quality

---

## Strategy Augmentations (3 modified files)

### S1 VRP (`strategies/s1_vrp/strategy.py`)
- Add `use_kelly: bool = False` flag
- Import `KellySizer(mode="fractional_kelly", max_position_pct=0.25)`
- In signal generation: `kelly_size = self._kelly.optimal_size(expected_return, volatility, drawdown, heat)`
- Scale conviction by kelly_size

### S7 Regime (`strategies/s7_regime/strategy.py`)
- Add `use_mdp: bool = False` flag
- Define `RegimeMDP`: states={TRENDING, MEAN_REVERTING, RANDOM, TOXIC}, actions={TREND_FOLLOW, MEAN_REVERT, THETA_HARVEST, FLAT}
- Estimate transition probs from historical regime sequence
- Solve via `value_iteration` at startup
- Replace hardcoded `if regime == TRENDING:` with solved policy

### S10 Gamma Scalp (`strategies/s10_gamma_scalp/strategy.py`)
- Add `use_deep_hedge: bool = False` flag
- Import `DeepHedgingAgent`
- When rehedging: use `deep_hedger.get_hedge_action(spot, positions, greeks)` instead of simple delta-neutral

(S5 Hawkes and S6 XGBoost augmentations are composable imports from `execution_pipeline.py` and `thompson_sizing.py` respectively — no strategy file modifications needed.)

---

## Execution Order

### Phase 1: Foundation (sequential, ~4 files)
1. `features/options_features.py` — new file
2. `features/mega.py` — wire options features
3. `models/ml/tft/x_trend.py` — add `forward_with_weights()`
4. `models/rl/integration/integrated_env.py` — add reward shaping hook

### Phase 2: Core Patterns (5 parallel agents)
- Agent A: `thompson_sizing.py` (Pattern 2)
- Agent B: `deep_hedging_pipeline.py` (Pattern 3 + NEW-2)
- Agent C: `cross_market.py` (Pattern 4)
- Agent D: `attention_reward.py` (Pattern 5)
- Agent E: `execution_pipeline.py` + `market_making_pipeline.py` (NEW-3 + NEW-4)

### Phase 3: Strategy Augmentations (3 parallel agents)
- Agent F: `strategies/s1_vrp/strategy.py` (Kelly)
- Agent G: `strategies/s7_regime/strategy.py` (MDP)
- Agent H: `strategies/s10_gamma_scalp/strategy.py` (Deep Hedge)

### Phase 4: Tests + Exports (parallel)
- Agent I: `tests/test_options_features.py`
- Agent J: `tests/test_rl_patterns.py` (35 tests)
- Agent K: `models/rl/integration/__init__.py`

### Phase 5: Run tests + Audit (3 parallel agents)
- Run full test suite (expect 1220+ tests, 0 failures)
- Audit 1: Stub scan (no pass, NotImplementedError, TODO)
- Audit 2: Look-ahead bias check (all features causal)
- Audit 3: Math verification (Kelly f*, Merton π*, A-S spreads, NIG posteriors)

### Phase 6: Training + Backtest
- Run all pipelines on real data
- Report per-pattern Sharpe, compare vs benchmarks

---

## Verification

```bash
cd QuantLaxmi && source venv/bin/activate

# Options features test
python -m pytest tests/test_options_features.py -v

# All new pattern tests
python -m pytest tests/test_rl_patterns.py -v

# Full test suite (expect 1220+, 0 failures)
python -m pytest tests/ -v

# Run integrated backtest with options features
python -m models.ml.tft.momentum_tfm --model xtrend-rl --start 2024-01-01 --end 2026-02-06

# Run individual pipelines
python -c "from models.rl.integration.deep_hedging_pipeline import StandaloneDeepHedger; ..."
python -c "from models.rl.integration.market_making_pipeline import MarketMakingPipeline; ..."
```

### Success Criteria
- All 1220+ tests pass (0 failures)
- Options features: 16 new features visible in mega builder output
- Each pipeline produces non-degenerate results on real DuckDB data
- Deep Hedging: PnL variance < BS delta hedging variance (OOS)
- Execution: RL shortfall <= TWAP shortfall (after training)
- Market Making: RL Sharpe >= A-S analytical Sharpe
- Strategy augmentations: opt-in flags work, existing tests unchanged
- Zero stubs, zero TODOs, zero NotImplementedError in any new file
- 3-agent audit: CLEAN
