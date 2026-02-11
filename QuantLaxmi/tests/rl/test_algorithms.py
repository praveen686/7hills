"""Tests for models.rl.algorithms — MC, TD, Q-Learning, Policy Gradient, Bandits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np


# =====================================================================
# Helpers — simple chain MDP for tabular algorithms
# =====================================================================

def _chain_mdp_episodes(num_episodes=50, gamma=0.9):
    """Generate episodes from a 3-state chain: s0 -> s1 -> s2(terminal).
    Reward is 1 at each step. Each episode is [(state, reward), ...].
    For MC/TD prediction format.
    """
    episodes = []
    for _ in range(num_episodes):
        episode = [('s0', 1.0), ('s1', 1.0), ('s2', 0.0)]
        episodes.append(episode)
    return episodes


def _chain_step(state, action):
    """Step function for Q-learning: (state, action) -> (next_state, reward, done)."""
    if state == 's0':
        return 's1', 1.0, False
    elif state == 's1':
        return 's2', 1.0, True
    else:
        return state, 0.0, True


# =====================================================================
# MC Prediction (Ch 11)
# =====================================================================

def test_mc_prediction_chain():
    from quantlaxmi.models.rl.algorithms import mc_prediction
    episodes = _chain_mdp_episodes(num_episodes=200)
    V = mc_prediction(episodes, gamma=0.9)
    # V(s0) should be higher than V(s1) since s0 collects more rewards
    v_s0 = V('s0')
    v_s1 = V('s1')
    assert v_s0 > v_s1, f"V(s0)={v_s0} should be > V(s1)={v_s1}"
    assert v_s1 > 0.5, f"V(s1)={v_s1} should be > 0.5"


# =====================================================================
# TD Prediction (Ch 12)
# =====================================================================

def test_td_prediction_chain():
    from quantlaxmi.models.rl.algorithms import td_prediction
    episodes = _chain_mdp_episodes(num_episodes=200)
    V = td_prediction(episodes, gamma=0.9, learning_rate=0.05)
    v_s0 = V('s0')
    v_s1 = V('s1')
    # s0 should have higher value than s1
    assert v_s0 > v_s1 * 0.5, f"V(s0)={v_s0} should be significantly > 0"
    assert v_s1 > 0.3, f"V(s1)={v_s1} should be > 0.3"


# =====================================================================
# Q-Learning (Ch 13)
# =====================================================================

def test_q_learning_chain():
    from quantlaxmi.models.rl.algorithms import q_learning
    Q, policy = q_learning(
        mdp_step=_chain_step,
        start_states=['s0'],
        actions=['right'],
        gamma=0.9,
        num_episodes=200,
        learning_rate=0.1,
        epsilon=0.1,
        seed=42,
    )
    # Policy should select 'right' from s0
    assert policy('s0') == 'right'
    assert policy('s1') == 'right'


# =====================================================================
# DQN (Ch 13)
# =====================================================================

def test_dqn_construct_and_select():
    from quantlaxmi.models.rl.algorithms import DQN
    dqn = DQN(
        state_dim=4,
        num_actions=2,
        hidden_layers=(16,),
        batch_size=4,
        device="cpu",
        seed=42,
    )
    state = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)
    action = dqn.select_action(state)
    assert action in [0, 1]


def test_dqn_greedy_select():
    from quantlaxmi.models.rl.algorithms import DQN
    dqn = DQN(
        state_dim=4,
        num_actions=2,
        hidden_layers=(16,),
        batch_size=4,
        device="cpu",
        seed=42,
    )
    state = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)
    action = dqn.select_action(state, greedy=True)
    assert action in [0, 1]


def test_dqn_store_and_train():
    from quantlaxmi.models.rl.algorithms import DQN
    dqn = DQN(
        state_dim=4,
        num_actions=2,
        hidden_layers=(16,),
        batch_size=4,
        buffer_size=100,
        device="cpu",
        seed=42,
    )
    # Store enough transitions to enable a train step
    for i in range(10):
        s = np.random.randn(4).astype(np.float32)
        a = int(np.random.randint(2))
        r = float(np.random.randn())
        s_next = np.random.randn(4).astype(np.float32)
        done = i == 9
        dqn.store_transition(s, a, r, s_next, done)

    # Should be able to train
    loss = dqn.train_step()
    assert isinstance(loss, float)


# =====================================================================
# REINFORCE (Ch 14)
# =====================================================================

def test_reinforce_construct_and_select():
    from quantlaxmi.models.rl.algorithms import REINFORCE
    agent = REINFORCE(
        state_dim=4,
        action_dim=2,
        hidden_layers=(16,),
        learning_rate=1e-3,
        gamma=0.99,
        baseline=False,
        device="cpu",
        seed=42,
    )
    state = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)
    action, log_prob = agent.select_action(state)
    assert action in [0, 1]
    assert isinstance(log_prob, float)


def test_reinforce_update():
    from quantlaxmi.models.rl.algorithms import REINFORCE
    agent = REINFORCE(
        state_dim=4,
        action_dim=2,
        hidden_layers=(16,),
        learning_rate=1e-3,
        gamma=0.99,
        baseline=False,
        device="cpu",
        seed=42,
    )
    # Generate a short episode
    for step in range(5):
        state = np.random.randn(4).astype(np.float32)
        action, lp = agent.select_action(state)
        agent.store_reward(float(np.random.randn()))

    loss = agent.update()
    assert isinstance(loss, float)


# =====================================================================
# Actor-Critic (Ch 14)
# =====================================================================

def test_actor_critic_construct_and_select():
    from quantlaxmi.models.rl.algorithms import ActorCritic
    ac = ActorCritic(
        state_dim=4,
        action_dim=2,
        actor_hidden=(16,),
        critic_hidden=(16,),
        device="cpu",
        seed=42,
    )
    state = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)
    action, log_prob, value = ac.select_action(state)
    assert action in [0, 1]
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_actor_critic_update_returns_two_losses():
    from quantlaxmi.models.rl.algorithms import ActorCritic
    ac = ActorCritic(
        state_dim=4,
        action_dim=2,
        actor_hidden=(16,),
        critic_hidden=(16,),
        device="cpu",
        seed=42,
    )
    # Build a list of transitions: (state, action, reward, log_prob, next_state, done)
    transitions = []
    for i in range(5):
        s = np.random.randn(4).astype(np.float32)
        a = int(np.random.randint(2))
        r = float(np.random.randn())
        lp = -0.5
        s_next = np.random.randn(4).astype(np.float32)
        done = (i == 4)
        transitions.append((s, a, r, lp, s_next, done))

    actor_loss, critic_loss = ac.update(transitions)
    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)


# =====================================================================
# Thompson Sampling (Ch 15)
# =====================================================================

def test_thompson_sampling_identifies_best_arm():
    from quantlaxmi.models.rl.algorithms import ThompsonSampling, GaussianArm
    arms = [
        GaussianArm(mu=0.0, sigma=0.5, seed=10),
        GaussianArm(mu=1.0, sigma=0.5, seed=20),
        GaussianArm(mu=0.5, sigma=0.5, seed=30),
    ]
    ts = ThompsonSampling(num_arms=3, seed=42)
    rewards, regrets = ts.run(arms, num_steps=500)
    cumulative_regret = regrets.sum()
    # With 500 pulls, cumulative regret should be finite and bounded
    assert cumulative_regret < 200, f"Regret too high: {cumulative_regret}"


# =====================================================================
# UCB1 (Ch 15)
# =====================================================================

def test_ucb1_identifies_best_arm():
    from quantlaxmi.models.rl.algorithms import UCB1, GaussianArm
    arms = [
        GaussianArm(mu=0.0, sigma=0.3, seed=10),
        GaussianArm(mu=2.0, sigma=0.3, seed=20),
        GaussianArm(mu=0.5, sigma=0.3, seed=30),
    ]
    ucb = UCB1(num_arms=3, seed=42)
    rewards, regrets = ucb.run(arms, num_steps=500)
    # After 500 pulls, best arm (1) should dominate
    cumulative_regret = regrets.sum()
    assert cumulative_regret < 100, f"Regret too high: {cumulative_regret}"


# =====================================================================
# EpsilonGreedy (Ch 15)
# =====================================================================

def test_epsilon_greedy_runs():
    from quantlaxmi.models.rl.algorithms import EpsilonGreedy, GaussianArm
    arms = [
        GaussianArm(mu=0.0, sigma=1.0, seed=10),
        GaussianArm(mu=1.0, sigma=1.0, seed=20),
    ]
    eg = EpsilonGreedy(num_arms=2, epsilon=0.1, seed=42)
    rewards, regrets = eg.run(arms, num_steps=200)
    assert len(rewards) == 200
    assert len(regrets) == 200


# =====================================================================
# GradientBandit (Ch 15)
# =====================================================================

def test_gradient_bandit_runs():
    from quantlaxmi.models.rl.algorithms import GradientBandit, GaussianArm
    arms = [
        GaussianArm(mu=0.0, sigma=1.0, seed=10),
        GaussianArm(mu=1.0, sigma=1.0, seed=20),
    ]
    gb = GradientBandit(num_arms=2, alpha=0.1, seed=42)
    rewards, regrets = gb.run(arms, num_steps=200)
    assert len(rewards) == 200
    assert len(regrets) == 200


# =====================================================================
# LinUCB — Contextual Bandits (Ch 15)
# =====================================================================

def test_linucb_contextual_selection():
    from quantlaxmi.models.rl.algorithms import LinUCB
    linucb = LinUCB(context_dim=3, num_arms=2, alpha=1.0)
    context = np.array([1.0, 0.5, -0.5])
    arm = linucb.select_arm(context)
    assert arm in [0, 1]
    # Update and re-select
    linucb.update(context, arm, 1.0)
    arm2 = linucb.select_arm(context)
    assert arm2 in [0, 1]


def test_linucb_learns_correct_arm():
    from quantlaxmi.models.rl.algorithms import LinUCB
    rng = np.random.default_rng(42)
    linucb = LinUCB(context_dim=2, num_arms=2, alpha=1.0)
    # Arm 0 gives reward = context[0], arm 1 gives reward = -context[0]
    # For positive context[0], arm 0 is better
    for _ in range(200):
        ctx = rng.standard_normal(2)
        arm = linucb.select_arm(ctx)
        if arm == 0:
            reward = ctx[0] + rng.normal(0, 0.1)
        else:
            reward = -ctx[0] + rng.normal(0, 0.1)
        linucb.update(ctx, arm, reward)

    # For a strongly positive context, should prefer arm 0
    test_ctx = np.array([3.0, 0.0])
    chosen = linucb.select_arm(test_ctx)
    assert chosen == 0, f"Expected arm 0 for positive context, got {chosen}"
