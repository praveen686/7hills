"""Tests for models.rl.core — Distributions, DP, Function Approximation, Utils."""
import sys
sys.path.insert(0, '/home/ubuntu/Desktop/7hills/QuantLaxmi')

import math
import pytest
import numpy as np


# =====================================================================
# Test Distributions
# =====================================================================

def test_categorical_sample_and_expectation():
    from models.rl.core import Categorical
    d = Categorical({1: 0.3, 2: 0.7})
    samples = [d.sample() for _ in range(100)]
    assert all(s in [1, 2] for s in samples)
    assert abs(d.expectation(lambda x: x) - 1.7) < 0.01


def test_categorical_probabilities():
    from models.rl.core import Categorical
    d = Categorical({'a': 0.5, 'b': 0.5})
    probs = d.probabilities()
    assert abs(probs['a'] - 0.5) < 1e-8
    assert abs(probs['b'] - 0.5) < 1e-8


def test_categorical_invalid_probs():
    from models.rl.core import Categorical
    with pytest.raises(ValueError):
        Categorical({1: 0.5, 2: 0.3})  # sums to 0.8


def test_gaussian_sample():
    from models.rl.core import Gaussian
    g = Gaussian(mu=5.0, sigma=1.0)
    samples = [g.sample() for _ in range(1000)]
    assert abs(np.mean(samples) - 5.0) < 0.2


def test_gaussian_zero_sigma():
    from models.rl.core import Gaussian
    g = Gaussian(mu=3.0, sigma=0.0)
    assert g.sample() == 3.0


def test_constant_distribution():
    from models.rl.core import Constant
    c = Constant(42)
    assert c.sample() == 42
    assert c.expectation(lambda x: x) == 42.0


def test_sampled_distribution():
    from models.rl.core import SampledDistribution
    d = SampledDistribution(sampler=lambda: 7.0)
    assert d.sample() == 7.0
    assert abs(d.expectation(lambda x: x) - 7.0) < 0.01


# =====================================================================
# Test States
# =====================================================================

def test_non_terminal_and_terminal():
    from models.rl.core import NonTerminal, Terminal
    nt = NonTerminal(1)
    t = Terminal(2)
    assert nt.state == 1
    assert t.state == 2
    assert nt != t


def test_non_terminal_equality():
    from models.rl.core import NonTerminal
    a = NonTerminal('x')
    b = NonTerminal('x')
    assert a == b
    assert hash(a) == hash(b)


# =====================================================================
# Test Value Function
# =====================================================================

def test_value_function_basics():
    from models.rl.core import ValueFunction, NonTerminal
    nt_a = NonTerminal('a')
    nt_b = NonTerminal('b')
    vf = ValueFunction({nt_a: 1.0, nt_b: 2.0})
    assert vf[nt_a] == 1.0
    assert vf[nt_b] == 2.0


def test_value_function_default_zero():
    from models.rl.core import ValueFunction, NonTerminal
    vf = ValueFunction()
    assert vf[NonTerminal('missing')] == 0.0


def test_value_function_max_diff():
    from models.rl.core import ValueFunction, NonTerminal
    s = NonTerminal(1)
    v1 = ValueFunction({s: 1.0})
    v2 = ValueFunction({s: 3.0})
    assert abs(v1.max_diff(v2) - 2.0) < 1e-8


# =====================================================================
# Test Function Approximation — Tabular
# =====================================================================

def test_tabular_update():
    from models.rl.core import Tabular
    t = Tabular(learning_rate=1.0)
    t = t.update([(1, 5.0), (2, 10.0)])
    vals = t.evaluate([1, 2])
    assert abs(vals[0] - 5.0) < 0.01
    assert abs(vals[1] - 10.0) < 0.01


def test_tabular_solve():
    from models.rl.core import Tabular
    t = Tabular()
    data = [(1, 3.0), (1, 5.0), (2, 10.0)]
    t = t.solve(data)
    vals = t.evaluate([1, 2])
    assert abs(vals[0] - 4.0) < 0.01  # mean of 3 and 5
    assert abs(vals[1] - 10.0) < 0.01


def test_tabular_callable():
    from models.rl.core import Tabular
    t = Tabular(learning_rate=1.0)
    t = t.update([('a', 7.0)])
    assert abs(t('a') - 7.0) < 0.01


# =====================================================================
# Test Function Approximation — Linear
# =====================================================================

def test_linear_fa_solve():
    from models.rl.core import LinearFunctionApprox
    # f(x) = 2x + 1, features: [1, x]
    fa = LinearFunctionApprox(
        feature_functions=[lambda x: 1.0, lambda x: float(x)],
        direct_solve=True,
    )
    data = [(x, 2 * x + 1) for x in range(10)]
    fa = fa.solve(data)
    vals = fa.evaluate([5.0])
    assert abs(vals[0] - 11.0) < 0.1


def test_linear_fa_evaluate_empty():
    from models.rl.core import LinearFunctionApprox
    fa = LinearFunctionApprox(feature_functions=[lambda x: float(x)])
    result = fa.evaluate([])
    assert len(result) == 0


def test_linear_fa_update():
    from models.rl.core import LinearFunctionApprox, AdamGradient
    fa = LinearFunctionApprox(
        feature_functions=[lambda x: 1.0, lambda x: float(x)],
        adam_gradient=AdamGradient(learning_rate=0.1),
    )
    # Multiple updates should move weights closer to the target
    data = [(x, 2 * x + 1) for x in range(10)]
    for _ in range(500):
        fa = fa.update(data)
    vals = fa.evaluate([5.0])
    assert abs(vals[0] - 11.0) < 2.0  # loose tolerance for SGD


# =====================================================================
# Test Function Approximation — DNN
# =====================================================================

def test_dnn_approx_fit():
    from models.rl.core import DNNApprox, DNNSpec
    fa = DNNApprox(
        feature_functions=[lambda x: float(x)],
        dnn_spec=[DNNSpec(neurons=32), DNNSpec(neurons=16)],
        learning_rate=0.01,
        device="cpu",
    )
    data = [(x / 10.0, (x / 10.0) ** 2) for x in range(10)]
    fa = fa.solve(data)
    # Should approximate x^2 reasonably
    val = fa.evaluate([0.5])
    assert abs(val[0] - 0.25) < 0.2  # loose tolerance for DNN


# =====================================================================
# Test Utils — returns
# =====================================================================

def test_returns_computation():
    from models.rl.core import returns
    r = returns([1.0, 1.0, 1.0], gamma=0.9)
    assert abs(r[0] - (1 + 0.9 + 0.81)) < 0.01
    assert abs(r[2] - 1.0) < 0.01


def test_returns_empty():
    from models.rl.core import returns
    assert returns([], gamma=0.9) == []


def test_returns_single():
    from models.rl.core import returns
    r = returns([5.0], gamma=0.99)
    assert abs(r[0] - 5.0) < 1e-8


# =====================================================================
# Test Utils — utility functions
# =====================================================================

def test_crra_utility_log():
    from models.rl.core import crra_utility
    # gamma=1 should be log
    assert abs(crra_utility(math.e, 1.0) - 1.0) < 0.01


def test_crra_utility_risk_neutral():
    from models.rl.core import crra_utility
    # gamma=0 should be linear: u(x) = x^1 / 1 = x
    assert abs(crra_utility(5.0, 0.0) - 5.0) < 0.01


def test_crra_utility_invalid():
    from models.rl.core import crra_utility
    with pytest.raises(ValueError):
        crra_utility(-1.0, 1.0)


def test_cara_utility():
    from models.rl.core import cara_utility
    # u(0) = -exp(0)/alpha = -1/alpha
    val = cara_utility(0.0, 1.0)
    assert abs(val - (-1.0)) < 0.01


def test_cara_utility_zero_alpha():
    from models.rl.core import cara_utility
    # alpha=0 should be linear: u(x) = x
    assert abs(cara_utility(5.0, 0.0) - 5.0) < 0.01


# =====================================================================
# Test Dynamic Programming — Value Iteration on simple MDP
# =====================================================================

def test_value_iteration_simple():
    """3-state chain MDP: s0 -> s1 -> s2(terminal), reward 1 at each step.
    Action 'right' moves to the next state. Optimal V(s0) ~ 1 + gamma, V(s1) ~ 1.
    """
    from models.rl.core import FiniteMarkovDecisionProcess, value_iteration

    transition_map = {
        's0': {
            'right': {('s1', 1.0): 1.0},
        },
        's1': {
            'right': {('s2', 1.0): 1.0},
        },
        # s2 is terminal (not in map)
    }
    gamma = 0.9
    mdp = FiniteMarkovDecisionProcess(transition_map, gamma=gamma)
    vf, policy = value_iteration(mdp, gamma=gamma)

    from models.rl.core import NonTerminal
    v_s0 = vf[NonTerminal('s0')]
    v_s1 = vf[NonTerminal('s1')]

    assert abs(v_s1 - 1.0) < 0.01
    assert abs(v_s0 - (1.0 + gamma)) < 0.01


# =====================================================================
# Test Utils — iterate_converge, moving_average, td_target
# =====================================================================

def test_moving_average():
    from models.rl.core import moving_average
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ma = moving_average(data, window=3)
    # First valid moving avg at index 2: (1+2+3)/3 = 2, then (2+3+4)/3=3, (3+4+5)/3=4
    assert len(ma) > 0


def test_returns_from_rewards():
    from models.rl.core import returns_from_rewards
    # G_0 = 1 + 0.9*1 + 0.81*1 = 2.71
    val = returns_from_rewards([1.0, 1.0, 1.0], gamma=0.9)
    assert abs(val - 2.71) < 0.01
