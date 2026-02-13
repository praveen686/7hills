"""RL Algorithms — Monte Carlo, TD, Q-Learning, Policy Gradient, Bandits, Offline RL, Distributional.

Implements Chapters 11-15 of "Foundations of RL with Applications in Finance"
by Ashwin Rao & Tikhon Jelvis, plus modern offline RL and distributional RL methods.

Modules
-------
monte_carlo : Monte Carlo prediction and control (Ch 11)
td_learning : Temporal-Difference methods — TD(0), TD(lambda), SARSA (Ch 12)
q_learning  : Q-Learning, DQN, Double DQN, LSPI (Ch 13)
policy_gradient : REINFORCE, Actor-Critic, A2C, NPG, DDPG (Ch 14)
bandits     : Multi-Armed and Contextual Bandits (Ch 15)
offline_rl  : CQL, IQL, TD3+BC for offline/batch RL
distributional_rl : C51, QR-DQN, IQN, RiskAwareTrader (Bellemare+ 2017, Dabney+ 2018)
"""
from __future__ import annotations

# -- Monte Carlo (Ch 11) --------------------------------------------------
from quantlaxmi.models.rl.algorithms.monte_carlo import (
    mc_prediction,
    mc_control,
    glie_mc_control,
    importance_sampling_mc,
    TabularApprox,
)

# -- Temporal-Difference (Ch 12) -------------------------------------------
from quantlaxmi.models.rl.algorithms.td_learning import (
    td_prediction,
    td_lambda_prediction,
    sarsa,
    sarsa_lambda,
    TDExperienceReplay,
)

# -- Q-Learning & DQN (Ch 13) ----------------------------------------------
from quantlaxmi.models.rl.algorithms.q_learning import (
    q_learning,
    DQN,
    DoubleDQN,
    lspi,
    lspi_policy,
)

# -- Policy Gradient (Ch 14) -----------------------------------------------
from quantlaxmi.models.rl.algorithms.policy_gradient import (
    PolicyGradientBase,
    REINFORCE,
    ActorCritic,
    A2C,
    NaturalPolicyGradient,
    DeterministicPolicyGradient,
)

# -- PPO & SAC (Advanced Policy Gradient) ----------------------------------
from quantlaxmi.models.rl.algorithms.ppo_sac import (
    PPO,
    SAC,
)

# -- Offline RL (CQL, IQL, TD3+BC) -----------------------------------------
from quantlaxmi.models.rl.algorithms.offline_rl import (
    CQL,
    IQL,
    TD3BC,
    OfflineReplayBuffer,
)

# -- Distributional RL (C51, QR-DQN, IQN) ----------------------------------
from quantlaxmi.models.rl.algorithms.distributional_rl import (
    C51,
    QRDQN,
    IQN,
    RiskAwareTrader,
)

# -- Bandits (Ch 15) -------------------------------------------------------
from quantlaxmi.models.rl.algorithms.bandits import (
    BanditArm,
    GaussianArm,
    BernoulliArm,
    BanditAlgorithm,
    EpsilonGreedy,
    UCB1,
    ThompsonSampling,
    GradientBandit,
    ContextualBandit,
    LinUCB,
    NeuralThompsonSampling,
)

__all__ = [
    # Monte Carlo
    "mc_prediction",
    "mc_control",
    "glie_mc_control",
    "importance_sampling_mc",
    "TabularApprox",
    # Temporal-Difference
    "td_prediction",
    "td_lambda_prediction",
    "sarsa",
    "sarsa_lambda",
    "TDExperienceReplay",
    # Q-Learning
    "q_learning",
    "DQN",
    "DoubleDQN",
    "lspi",
    "lspi_policy",
    # Policy Gradient
    "PolicyGradientBase",
    "REINFORCE",
    "ActorCritic",
    "A2C",
    "NaturalPolicyGradient",
    "DeterministicPolicyGradient",
    # PPO & SAC
    "PPO",
    "SAC",
    # Offline RL
    "CQL",
    "IQL",
    "TD3BC",
    "OfflineReplayBuffer",
    # Distributional RL
    "C51",
    "QRDQN",
    "IQN",
    "RiskAwareTrader",
    # Bandits
    "BanditArm",
    "GaussianArm",
    "BernoulliArm",
    "BanditAlgorithm",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonSampling",
    "GradientBandit",
    "ContextualBandit",
    "LinUCB",
    "NeuralThompsonSampling",
]
