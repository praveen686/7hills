"""RL Algorithms — Monte Carlo, TD, Q-Learning, Policy Gradient, Bandits.

Implements Chapters 11-15 of "Foundations of RL with Applications in Finance"
by Ashwin Rao & Tikhon Jelvis.

Modules
-------
monte_carlo : Monte Carlo prediction and control (Ch 11)
td_learning : Temporal-Difference methods — TD(0), TD(lambda), SARSA (Ch 12)
q_learning  : Q-Learning, DQN, Double DQN, LSPI (Ch 13)
policy_gradient : REINFORCE, Actor-Critic, A2C, NPG, DDPG (Ch 14)
bandits     : Multi-Armed and Contextual Bandits (Ch 15)
"""
from __future__ import annotations

# -- Monte Carlo (Ch 11) --------------------------------------------------
from models.rl.algorithms.monte_carlo import (
    mc_prediction,
    mc_control,
    glie_mc_control,
    importance_sampling_mc,
    TabularApprox,
)

# -- Temporal-Difference (Ch 12) -------------------------------------------
from models.rl.algorithms.td_learning import (
    td_prediction,
    td_lambda_prediction,
    sarsa,
    sarsa_lambda,
    TDExperienceReplay,
)

# -- Q-Learning & DQN (Ch 13) ----------------------------------------------
from models.rl.algorithms.q_learning import (
    q_learning,
    DQN,
    DoubleDQN,
    lspi,
    lspi_policy,
)

# -- Policy Gradient (Ch 14) -----------------------------------------------
from models.rl.algorithms.policy_gradient import (
    PolicyGradientBase,
    REINFORCE,
    ActorCritic,
    A2C,
    NaturalPolicyGradient,
    DeterministicPolicyGradient,
)

# -- Bandits (Ch 15) -------------------------------------------------------
from models.rl.algorithms.bandits import (
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
