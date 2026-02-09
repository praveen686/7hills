"""Core MDP framework — Markov Processes, Dynamic Programming, Function Approximation.

Implements Chapters 2-6 of "Foundations of Reinforcement Learning with
Applications in Finance" (Rao & Jelvis, Stanford CME 241).

Modules:
  markov_process     — Distributions, States, MP, MRP, MDP, Policy (Ch 2-3)
  dynamic_programming — Policy Eval, PI, VI, Backward Induction (Ch 4-5)
  function_approx    — Tabular, Linear FA, DNN FA (Ch 6)
  utils              — Returns, episodes, exploration, utility theory
"""
from __future__ import annotations

# --- Ch 2-3: Markov Processes ---
from .markov_process import (
    # Distributions
    Distribution,
    Categorical,
    Gaussian,
    SampledDistribution,
    Constant,
    # States
    State,
    Terminal,
    NonTerminal,
    # Markov processes
    MarkovProcess,
    MarkovRewardProcess,
    FiniteMarkovProcess,
    FiniteMarkovRewardProcess,
    # MDPs
    MarkovDecisionProcess,
    FiniteMarkovDecisionProcess,
    # Policies
    Policy,
    DeterministicPolicy,
    TabularPolicy,
    # Composition
    apply_policy,
    apply_finite_policy,
)

# --- Ch 4-5: Dynamic Programming ---
from .dynamic_programming import (
    DEFAULT_TOLERANCE,
    ValueFunction,
    ActionValueFunction,
    policy_evaluation,
    policy_evaluation_direct,
    policy_improvement,
    policy_iteration,
    value_iteration,
    backward_induction,
    bellman_optimality_operator,
    bellman_policy_operator,
    lstd_prediction,
)

# --- Ch 6: Function Approximation ---
from .function_approx import (
    FunctionApprox,
    Tabular,
    LinearFunctionApprox,
    DNNSpec,
    DNNApprox,
    AdamGradient,
)

# --- Utilities ---
from .utils import (
    returns,
    returns_from_rewards,
    episodes_from_mdp,
    greedy_policy_from_qvf,
    epsilon_greedy_policy,
    softmax_policy,
    crra_utility,
    cara_utility,
    certainty_equivalent,
    iterate_converge,
    set_device,
    moving_average,
    td_target,
)

__all__ = [
    # Distributions
    "Distribution",
    "Categorical",
    "Gaussian",
    "SampledDistribution",
    "Constant",
    # States
    "State",
    "Terminal",
    "NonTerminal",
    # Markov processes
    "MarkovProcess",
    "MarkovRewardProcess",
    "FiniteMarkovProcess",
    "FiniteMarkovRewardProcess",
    # MDPs
    "MarkovDecisionProcess",
    "FiniteMarkovDecisionProcess",
    # Policies
    "Policy",
    "DeterministicPolicy",
    "TabularPolicy",
    # Composition
    "apply_policy",
    "apply_finite_policy",
    # DP
    "DEFAULT_TOLERANCE",
    "ValueFunction",
    "ActionValueFunction",
    "policy_evaluation",
    "policy_evaluation_direct",
    "policy_improvement",
    "policy_iteration",
    "value_iteration",
    "backward_induction",
    "bellman_optimality_operator",
    "bellman_policy_operator",
    "lstd_prediction",
    # Function Approximation
    "FunctionApprox",
    "Tabular",
    "LinearFunctionApprox",
    "DNNSpec",
    "DNNApprox",
    "AdamGradient",
    # Utilities
    "returns",
    "returns_from_rewards",
    "episodes_from_mdp",
    "greedy_policy_from_qvf",
    "epsilon_greedy_policy",
    "softmax_policy",
    "crra_utility",
    "cara_utility",
    "certainty_equivalent",
    "iterate_converge",
    "set_device",
    "moving_average",
    "td_target",
]
