# QuantLaxmi RL — Reinforcement Learning for Finance

Implementation of RL algorithms from "Foundations of Reinforcement Learning with
Applications in Finance" (Rao & Jelvis, Stanford CME 241), adapted for the
BRAHMASTRA/QuantLaxmi trading system.

## Architecture

```
QuantLaxmi_RL/
├── core/                      # MDP framework (Ch 2-6)
│   ├── markov_process.py      # Markov Process, MRP, MDP
│   ├── dynamic_programming.py # Policy Evaluation, PI, VI
│   ├── function_approx.py     # Linear FA + DNN FA (PyTorch)
│   └── utils.py               # Returns, episodes, distributions
│
├── algorithms/                # RL algorithms (Ch 11-15)
│   ├── monte_carlo.py         # MC prediction & control
│   ├── td_learning.py         # TD(0), TD(λ), SARSA
│   ├── q_learning.py          # Q-Learning, DQN, LSPI
│   ├── policy_gradient.py     # REINFORCE, Actor-Critic, NPG, DPG
│   └── bandits.py             # UCB, Thompson Sampling, Gradient Bandits
│
├── finance/                   # Financial applications (Ch 8-10)
│   ├── asset_allocation.py    # Merton's problem, AssetAllocPG
│   ├── derivatives_pricing.py # American options MDP, Deep Hedging
│   ├── optimal_execution.py   # Bertsimas-Lo, Almgren-Chriss
│   └── market_making.py       # Avellaneda-Stoikov
│
├── environments/              # Trading MDP environments
│   ├── trading_env.py         # Generic trading MDP base
│   ├── india_fno_env.py       # India FnO (NIFTY, BANKNIFTY)
│   ├── crypto_env.py          # Binance BTC/ETH/SOL
│   ├── options_env.py         # Options trading
│   └── execution_env.py       # Order execution
│
├── agents/                    # RL agents for strategies
│   ├── thompson_allocator.py  # Strategy selector (Ch 15)
│   ├── deep_hedger.py         # Options hedging (Ch 9)
│   ├── execution_agent.py     # Optimal execution (Ch 10.2)
│   ├── market_maker.py        # Market-making (Ch 10.3)
│   └── kelly_sizer.py         # Position sizing (Ch 7-8)
│
├── integration/               # System integration (Phase 2)
│   ├── tft_backbone.py        # TFT + RL
│   ├── hydra_bridge.py        # Rust HYDRA bridge
│   └── feature_bridge.py      # Mega features connector
│
└── tests/                     # Comprehensive tests
```

## Design Principles

1. **Faithful to the book**: Core algorithms match Rao & Jelvis exactly
2. **PyTorch-native**: All neural network FA uses PyTorch (T4 GPU)
3. **Composable**: Algorithms, environments, and agents are independent
4. **Finance-first**: Every module has financial application examples
5. **No look-ahead bias**: All environments enforce causal data access
6. **Compatible**: Integrates with existing QuantLaxmi core/ and alpha_forge/

## Key Book Concepts → Code Mapping

| Book Concept | Chapter | Module |
|-------------|---------|--------|
| State, Action, Reward | Ch 2-3 | `core.markov_process` |
| Bellman Equations | Ch 3 | `core.dynamic_programming` |
| Function Approximation | Ch 6 | `core.function_approx` |
| Merton Allocation | Ch 8 | `finance.asset_allocation` |
| Deep Hedging | Ch 9 | `finance.derivatives_pricing` |
| Optimal Execution | Ch 10.2 | `finance.optimal_execution` |
| Avellaneda-Stoikov | Ch 10.3 | `finance.market_making` |
| REINFORCE | Ch 14.4 | `algorithms.policy_gradient` |
| Actor-Critic | Ch 14.6 | `algorithms.policy_gradient` |
| Thompson Sampling | Ch 15.5 | `algorithms.bandits` |
