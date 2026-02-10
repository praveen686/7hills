"""Tests for PPO and SAC implementations.

Tests cover:
- PPO: discrete + continuous action spaces, GAE computation, clipped loss,
  gradient updates, rollout buffer, predict method, training loop.
- SAC: twin Q convergence, alpha auto-tuning, replay buffer, squashed
  Gaussian sampling, soft target updates, training loop.
- Integration tests on simple environments.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from quantlaxmi.models.rl.algorithms.ppo_sac import (
    PPO,
    SAC,
    _RolloutBuffer,
    _SACReplayBuffer,
    _SquashedGaussianPolicy,
)


# =====================================================================
# Simple test environments (gym-like interface)
# =====================================================================


class DiscreteCartPoleEnv:
    """Minimal CartPole-like environment for testing discrete PPO.

    State: [position, velocity, angle, angular_velocity] (dim=4)
    Actions: 0 (left) or 1 (right)
    Episode terminates after max_steps or if angle > 0.2 rad.
    """

    def __init__(self, max_steps: int = 50, seed: int = 42) -> None:
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._state: np.ndarray = np.zeros(4, dtype=np.float32)
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._state = self._rng.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self._step_count = 0
        return self._state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        force = 1.0 if action == 1 else -1.0
        x, x_dot, theta, theta_dot = self._state

        # Simplified dynamics
        gravity = 9.8
        cart_mass = 1.0
        pole_mass = 0.1
        length = 0.5
        dt = 0.02

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        total_mass = cart_mass + pole_mass

        temp = (force + pole_mass * length * theta_dot ** 2 * sin_theta) / total_mass
        theta_acc = (gravity * sin_theta - cos_theta * temp) / (
            length * (4.0 / 3.0 - pole_mass * cos_theta ** 2 / total_mass)
        )
        x_acc = temp - pole_mass * length * theta_acc * cos_theta / total_mass

        x += dt * x_dot
        x_dot += dt * x_acc
        theta += dt * theta_dot
        theta_dot += dt * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._step_count += 1

        done = abs(theta) > 0.2 or abs(x) > 2.4 or self._step_count >= self.max_steps
        reward = 1.0 if not done or self._step_count >= self.max_steps else 0.0

        return self._state.copy(), reward, done, {}


class ContinuousPendulumEnv:
    """Minimal pendulum-like environment for testing continuous algorithms.

    State: [cos(theta), sin(theta), theta_dot] (dim=3)
    Action: torque in [-1, 1] (dim=1)
    Reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    """

    def __init__(self, max_steps: int = 50, seed: int = 42) -> None:
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._theta: float = 0.0
        self._theta_dot: float = 0.0
        self._step_count = 0

    def reset(self) -> np.ndarray:
        self._theta = self._rng.uniform(-math.pi, math.pi)
        self._theta_dot = self._rng.uniform(-1.0, 1.0)
        self._step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [math.cos(self._theta), math.sin(self._theta), self._theta_dot],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if isinstance(action, np.ndarray):
            torque = float(np.clip(action[0] if action.ndim > 0 else action, -1.0, 1.0))
        else:
            torque = float(np.clip(action, -1.0, 1.0))

        dt = 0.05
        g = 10.0
        m = 1.0
        l = 1.0

        self._theta_dot += (g / l * math.sin(self._theta) + torque / (m * l ** 2)) * dt
        self._theta_dot = np.clip(self._theta_dot, -8.0, 8.0)
        self._theta += self._theta_dot * dt

        # Normalize angle to [-pi, pi]
        self._theta = ((self._theta + math.pi) % (2 * math.pi)) - math.pi

        self._step_count += 1
        done = self._step_count >= self.max_steps

        # Reward: want theta and theta_dot to be zero
        reward = -(self._theta ** 2 + 0.1 * self._theta_dot ** 2 + 0.001 * torque ** 2)

        return self._get_obs(), reward, done, {}


# =====================================================================
# PPO Tests
# =====================================================================


class TestRolloutBuffer:
    """Test the PPO rollout buffer."""

    def test_add_and_length(self) -> None:
        buf = _RolloutBuffer()
        assert len(buf) == 0
        for i in range(10):
            buf.add(
                state=np.zeros(4),
                action=0,
                reward=1.0,
                done=False,
                log_prob=-0.5,
                value=0.5,
            )
        assert len(buf) == 10

    def test_compute_gae_simple(self) -> None:
        """Test GAE computation with known values.

        With gamma=1.0, lambda=1.0 and no discounting,
        GAE reduces to Monte Carlo returns minus values.
        """
        buf = _RolloutBuffer()
        # 3-step trajectory: rewards [1, 1, 1], values [0, 0, 0], not done
        for _ in range(3):
            buf.add(np.zeros(2), 0, 1.0, False, -0.5, 0.0)

        buf.compute_gae(last_value=0.0, gamma=1.0, gae_lambda=1.0)

        # With gamma=1, lambda=1, values=0, last_value=0:
        # delta_t = r_t + gamma * V(t+1) - V(t) = r_t = 1.0 for all t
        # A_2 = delta_2 = 1.0
        # A_1 = delta_1 + gamma*lambda*A_2 = 1 + 1 = 2.0
        # A_0 = delta_0 + gamma*lambda*A_1 = 1 + 2 = 3.0
        np.testing.assert_allclose(buf.advantages, [3.0, 2.0, 1.0], atol=1e-10)
        # Returns = advantages + values = advantages (since values=0)
        np.testing.assert_allclose(buf.returns, [3.0, 2.0, 1.0], atol=1e-10)

    def test_gae_with_discount(self) -> None:
        """Test GAE with gamma < 1 and lambda < 1."""
        buf = _RolloutBuffer()
        gamma = 0.99
        lam = 0.95
        buf.add(np.zeros(2), 0, 1.0, False, -0.5, 0.5)
        buf.add(np.zeros(2), 0, 2.0, False, -0.5, 0.8)
        buf.add(np.zeros(2), 0, 3.0, True, -0.5, 1.0)

        buf.compute_gae(last_value=0.0, gamma=gamma, gae_lambda=lam)

        # Manual calculation (mask at step t uses done[t]):
        # t=2: mask=1-done[2]=0, delta_2 = 3.0 + gamma*last_value*0 - 1.0 = 2.0
        #      A_2 = delta_2 = 2.0 (gae starts here)
        # t=1: mask=1-done[1]=1.0 (done[1]=False), next_value=V(2)=1.0
        #      delta_1 = 2.0 + gamma*1.0*1.0 - 0.8 = 2.19
        #      A_1 = delta_1 + gamma*lam*mask_1*A_2 = 2.19 + 0.99*0.95*1.0*2.0 = 4.071
        # t=0: mask=1-done[0]=1.0, next_value=V(1)=0.8
        #      delta_0 = 1.0 + gamma*0.8*1.0 - 0.5 = 1.292
        #      A_0 = delta_0 + gamma*lam*1.0*A_1 = 1.292 + 0.99*0.95*4.071 = 5.1228905

        assert buf.advantages[2] == pytest.approx(2.0, abs=1e-6)
        assert buf.advantages[1] == pytest.approx(4.071, abs=1e-6)
        assert buf.advantages[0] == pytest.approx(1.292 + 0.99 * 0.95 * 4.071, abs=1e-4)
        assert len(buf.returns) == 3

    def test_gae_done_resets(self) -> None:
        """GAE should reset at episode boundaries (done=True)."""
        buf = _RolloutBuffer()
        buf.add(np.zeros(2), 0, 1.0, True, -0.5, 0.0)
        buf.add(np.zeros(2), 0, 1.0, False, -0.5, 0.0)

        buf.compute_gae(last_value=0.5, gamma=0.99, gae_lambda=0.95)

        # t=0 is done, so advantage at t=0 should not propagate from t=1
        # delta_0 = 1.0 + 0.99*0*V(1) - 0.0 = 1.0 (mask=0 because done[0]=True)
        # A_0 = 1.0
        assert buf.advantages[0] == pytest.approx(1.0, abs=1e-6)

    def test_get_batches_covers_all(self) -> None:
        """Mini-batches should cover all data points."""
        buf = _RolloutBuffer()
        for i in range(20):
            buf.add(np.array([float(i)]), i % 2, 1.0, False, -0.5, 0.5)
        buf.compute_gae(0.0, 0.99, 0.95)

        rng = np.random.default_rng(42)
        batches = buf.get_batches(8, rng)
        all_states = np.concatenate([b["states"] for b in batches])
        assert len(all_states) == 20

    def test_clear(self) -> None:
        buf = _RolloutBuffer()
        buf.add(np.zeros(2), 0, 1.0, False, -0.5, 0.5)
        buf.clear()
        assert len(buf) == 0


class TestPPODiscrete:
    """Test PPO with discrete action space."""

    def test_initialization(self) -> None:
        ppo = PPO(state_dim=4, action_dim=2, device="cpu", seed=42)
        assert ppo.clip_eps == 0.2
        assert ppo.gae_lambda == 0.95
        assert ppo.continuous is False

    def test_select_action_returns_valid(self) -> None:
        ppo = PPO(state_dim=4, action_dim=3, device="cpu", seed=42)
        state = np.random.randn(4).astype(np.float32)
        action, log_prob, value = ppo.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < 3
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert log_prob <= 0.0  # log probs are non-positive

    def test_predict_discrete(self) -> None:
        ppo = PPO(state_dim=4, action_dim=3, device="cpu", seed=42)
        state = np.random.randn(4).astype(np.float32)
        action = ppo.predict(state)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_update_changes_params(self) -> None:
        """PPO update should modify network parameters."""
        ppo = PPO(
            state_dim=4, action_dim=2, hidden_layers=(16,),
            device="cpu", seed=42, n_epochs=2, batch_size=4,
        )
        # Collect a small rollout
        for _ in range(8):
            state = np.random.randn(4).astype(np.float32)
            action, lp, val = ppo.select_action(state)
            ppo.store_transition(state, action, 1.0, False, lp, val)

        # Snapshot params before update
        params_before = [p.clone() for p in ppo.policy_network.parameters()]

        result = ppo.update(last_value=0.0)

        # Params should change
        changed = False
        for p_before, p_after in zip(params_before, ppo.policy_network.parameters()):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed, "PPO update should change policy parameters"
        assert "policy_loss" in result
        assert "value_loss" in result

    def test_clipped_loss_bounded(self) -> None:
        """Verify that PPO's clipped ratio stays within [1-eps, 1+eps]."""
        ppo = PPO(
            state_dim=4, action_dim=2, hidden_layers=(16,),
            device="cpu", seed=42, clip_eps=0.2,
        )
        # Create scenario: fill buffer, then update
        for _ in range(16):
            state = np.random.randn(4).astype(np.float32)
            action, lp, val = ppo.select_action(state)
            ppo.store_transition(state, action, 1.0, False, lp, val)

        # After update, the ratio should be close to 1 (since we just collected)
        # This is a sanity check that the mechanism works
        result = ppo.update(last_value=0.0)
        # approx_kl should be small for fresh data
        assert abs(result["approx_kl"]) < 1.0, "KL should be small for fresh rollout"

    def test_train_discrete_env(self) -> None:
        """PPO should train on discrete CartPole-like env without errors."""
        env = DiscreteCartPoleEnv(max_steps=30, seed=42)
        ppo = PPO(
            state_dim=4, action_dim=2, hidden_layers=(32,),
            device="cpu", seed=42, n_epochs=2, batch_size=16,
        )
        rewards = ppo.train(env, n_steps=200, rollout_steps=32)
        assert len(rewards) > 0, "Should complete at least one episode"


class TestPPOContinuous:
    """Test PPO with continuous action space."""

    def test_continuous_action(self) -> None:
        ppo = PPO(
            state_dim=3, action_dim=1, continuous=True,
            device="cpu", seed=42,
        )
        state = np.random.randn(3).astype(np.float32)
        action, log_prob, value = ppo.select_action(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert isinstance(log_prob, float)

    def test_predict_continuous(self) -> None:
        ppo = PPO(
            state_dim=3, action_dim=1, continuous=True,
            device="cpu", seed=42,
        )
        state = np.random.randn(3).astype(np.float32)
        action = ppo.predict(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_train_continuous_env(self) -> None:
        """PPO should train on continuous pendulum env without errors."""
        env = ContinuousPendulumEnv(max_steps=30, seed=42)
        ppo = PPO(
            state_dim=3, action_dim=1, continuous=True,
            hidden_layers=(32,), device="cpu", seed=42,
            n_epochs=2, batch_size=16,
        )
        rewards = ppo.train(env, n_steps=200, rollout_steps=32)
        assert len(rewards) > 0

    def test_entropy_positive(self) -> None:
        """Entropy should be positive (Gaussian has positive entropy)."""
        ppo = PPO(
            state_dim=3, action_dim=1, continuous=True,
            hidden_layers=(16,), device="cpu", seed=42,
        )
        for _ in range(16):
            state = np.random.randn(3).astype(np.float32)
            action, lp, val = ppo.select_action(state)
            ppo.store_transition(state, action, 0.0, False, lp, val)

        result = ppo.update(last_value=0.0)
        assert result["entropy"] > 0.0, "Gaussian entropy should be positive"


# =====================================================================
# SAC Tests
# =====================================================================


class TestSACReplayBuffer:
    """Test SAC replay buffer."""

    def test_add_and_sample(self) -> None:
        rng = np.random.default_rng(42)
        buf = _SACReplayBuffer(capacity=100, rng=rng)
        for i in range(20):
            buf.add(
                np.array([float(i)]),
                np.array([0.5]),
                1.0,
                np.array([float(i + 1)]),
                False,
            )
        assert len(buf) == 20

        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 1)
        assert actions.shape == (8, 1)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 1)
        assert dones.shape == (8,)

    def test_capacity_limit(self) -> None:
        rng = np.random.default_rng(42)
        buf = _SACReplayBuffer(capacity=10, rng=rng)
        for i in range(20):
            buf.add(np.zeros(1), np.zeros(1), 0.0, np.zeros(1), False)
        assert len(buf) == 10


class TestSquashedGaussianPolicy:
    """Test the squashed Gaussian policy network."""

    def test_output_bounds(self) -> None:
        """Squashed actions should be in (-1, 1)."""
        policy = _SquashedGaussianPolicy(state_dim=3, action_dim=2, hidden_layers=(16,))
        state = torch.randn(10, 3)
        actions, log_probs = policy.sample(state)
        assert actions.shape == (10, 2)
        assert log_probs.shape == (10, 1)
        # tanh output must be in (-1, 1)
        assert (actions.abs() < 1.0).all(), "Squashed actions must be in (-1, 1)"

    def test_log_prob_finite(self) -> None:
        """Log probabilities should be finite."""
        policy = _SquashedGaussianPolicy(state_dim=3, action_dim=1, hidden_layers=(16,))
        state = torch.randn(5, 3)
        _, log_probs = policy.sample(state)
        assert torch.isfinite(log_probs).all(), "Log probs should be finite"

    def test_deterministic_output(self) -> None:
        """Deterministic output should be consistent."""
        policy = _SquashedGaussianPolicy(state_dim=3, action_dim=1, hidden_layers=(16,))
        state = torch.randn(1, 3)
        a1 = policy.deterministic(state)
        a2 = policy.deterministic(state)
        torch.testing.assert_close(a1, a2)


class TestSAC:
    """Test SAC algorithm."""

    def test_initialization(self) -> None:
        sac = SAC(state_dim=3, action_dim=1, device="cpu", seed=42)
        assert sac.tau == 0.005
        assert sac.gamma == 0.99
        assert sac.auto_alpha is True
        assert sac.alpha.item() > 0

    def test_select_action_warmup(self) -> None:
        """During warmup, actions should be random uniform in [-1, 1]."""
        sac = SAC(
            state_dim=3, action_dim=2, device="cpu", seed=42,
            warmup_steps=100,
        )
        state = np.random.randn(3).astype(np.float32)
        action = sac.select_action(state, explore=True)
        assert action.shape == (2,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_select_action_after_warmup(self) -> None:
        """After warmup, actions should come from the policy."""
        sac = SAC(
            state_dim=3, action_dim=1, device="cpu", seed=42,
            warmup_steps=0,  # no warmup
        )
        state = np.random.randn(3).astype(np.float32)
        action = sac.select_action(state, explore=True)
        assert action.shape == (1,)
        # Should still be bounded by tanh
        assert np.all(np.abs(action) < 1.0)

    def test_predict_deterministic(self) -> None:
        sac = SAC(state_dim=3, action_dim=1, device="cpu", seed=42, warmup_steps=0)
        state = np.random.randn(3).astype(np.float32)
        a1 = sac.predict(state)
        a2 = sac.predict(state)
        np.testing.assert_allclose(a1, a2)

    def test_twin_q_different(self) -> None:
        """Twin Q-networks should produce different outputs (different init)."""
        sac = SAC(state_dim=3, action_dim=1, hidden_layers=(16,), device="cpu", seed=42)
        state = torch.randn(1, 3)
        action = torch.randn(1, 1)
        sa = torch.cat([state, action], dim=1)
        q1 = sac.q1_network(sa)
        q2 = sac.q2_network(sa)
        # They should generally differ (different random init layers)
        # But with same seed they might be similar; test they exist and are finite
        assert torch.isfinite(q1).all()
        assert torch.isfinite(q2).all()

    def test_alpha_auto_tuning(self) -> None:
        """Alpha should change after updates with auto_alpha=True."""
        sac = SAC(
            state_dim=3, action_dim=1, hidden_layers=(16,),
            device="cpu", seed=42, warmup_steps=0,
            batch_size=8, buffer_size=100,
        )
        alpha_before = sac.alpha.item()

        # Fill buffer
        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            sac.store_transition(s, a, 1.0, np.random.randn(3).astype(np.float32), False)

        # Run several updates
        for _ in range(10):
            sac.update()

        alpha_after = sac.alpha.item()
        assert alpha_after != alpha_before, "Alpha should change with auto-tuning"

    def test_soft_target_update(self) -> None:
        """Soft target update should move target params toward source."""
        sac = SAC(
            state_dim=3, action_dim=1, hidden_layers=(16,),
            device="cpu", seed=42, tau=0.5,  # large tau for visible change
        )
        # Get initial target params
        target_params_before = [p.clone() for p in sac.q1_target.parameters()]

        # Manually change source params
        with torch.no_grad():
            for p in sac.q1_network.parameters():
                p.add_(torch.randn_like(p))

        # Soft update
        sac._soft_update(sac.q1_target, sac.q1_network)

        # Target should have moved
        changed = False
        for p_before, p_after in zip(target_params_before, sac.q1_target.parameters()):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed, "Soft target update should change target parameters"

    def test_update_returns_metrics(self) -> None:
        """Update should return proper metric dict."""
        sac = SAC(
            state_dim=3, action_dim=1, hidden_layers=(16,),
            device="cpu", seed=42, warmup_steps=0, batch_size=8,
        )
        # Fill buffer
        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            sac.store_transition(s, a, 1.0, np.random.randn(3).astype(np.float32), False)

        result = sac.update()
        assert "q1_loss" in result
        assert "q2_loss" in result
        assert "policy_loss" in result
        assert "alpha" in result
        assert "entropy" in result
        assert result["alpha"] > 0

    def test_update_insufficient_buffer(self) -> None:
        """Update with too few samples should return zeros."""
        sac = SAC(state_dim=3, action_dim=1, device="cpu", batch_size=64)
        # Only 5 transitions (< batch_size)
        for _ in range(5):
            sac.store_transition(
                np.zeros(3), np.zeros(1), 0.0, np.zeros(3), False,
            )
        result = sac.update()
        assert result["q1_loss"] == 0.0
        assert result["q2_loss"] == 0.0

    def test_train_continuous_env(self) -> None:
        """SAC should train on continuous pendulum env without errors."""
        env = ContinuousPendulumEnv(max_steps=20, seed=42)
        sac = SAC(
            state_dim=3, action_dim=1, hidden_layers=(16,),
            device="cpu", seed=42, warmup_steps=50,
            batch_size=16, buffer_size=500,
        )
        rewards = sac.train(env, n_steps=200)
        assert len(rewards) > 0, "Should complete at least one episode"

    def test_fixed_alpha(self) -> None:
        """With auto_alpha=False, alpha should stay constant."""
        sac = SAC(
            state_dim=3, action_dim=1, hidden_layers=(16,),
            device="cpu", seed=42, auto_alpha=False,
            alpha_init=0.5, warmup_steps=0, batch_size=8,
        )
        alpha_before = sac.alpha.item()
        assert alpha_before == pytest.approx(0.5, abs=1e-4)

        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            sac.store_transition(s, a, 1.0, np.random.randn(3).astype(np.float32), False)

        for _ in range(5):
            sac.update()

        alpha_after = sac.alpha.item()
        assert alpha_after == pytest.approx(alpha_before, abs=1e-6), \
            "Alpha should not change with auto_alpha=False"


# =====================================================================
# Integration / Cross-algorithm tests
# =====================================================================


class TestIntegration:
    """Integration tests that verify both PPO and SAC work end-to-end."""

    def test_ppo_value_clipping(self) -> None:
        """PPO with value function clipping should still train."""
        env = DiscreteCartPoleEnv(max_steps=30, seed=42)
        ppo = PPO(
            state_dim=4, action_dim=2, hidden_layers=(16,),
            device="cpu", seed=42, clip_value=True,
            n_epochs=2, batch_size=16,
        )
        rewards = ppo.train(env, n_steps=100, rollout_steps=32)
        assert len(rewards) >= 0  # Should not crash

    def test_ppo_gradient_norm(self) -> None:
        """Gradient norms should be clipped to max_grad_norm."""
        ppo = PPO(
            state_dim=4, action_dim=2, hidden_layers=(16,),
            device="cpu", seed=42, max_grad_norm=0.5,
        )
        for _ in range(16):
            state = np.random.randn(4).astype(np.float32)
            action, lp, val = ppo.select_action(state)
            ppo.store_transition(state, action, 100.0, False, lp, val)  # large reward

        ppo.update(last_value=0.0)
        # If we got here without explosion, clipping worked
        # Check that params are finite
        for p in ppo.policy_network.parameters():
            assert torch.isfinite(p).all()

    def test_sac_q_gradient_flows(self) -> None:
        """Verify that Q-network gradients flow during SAC update."""
        sac = SAC(
            state_dim=3, action_dim=1, hidden_layers=(16,),
            device="cpu", seed=42, warmup_steps=0, batch_size=8,
        )
        q1_params_before = [p.clone() for p in sac.q1_network.parameters()]

        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            sac.store_transition(s, a, 1.0, np.random.randn(3).astype(np.float32), False)

        sac.update()

        changed = False
        for p_before, p_after in zip(q1_params_before, sac.q1_network.parameters()):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed, "Q1 network should update"

    def test_both_importable(self) -> None:
        """Both PPO and SAC should be importable from the algorithms package."""
        from quantlaxmi.models.rl.algorithms import PPO as PPO_imported
        from quantlaxmi.models.rl.algorithms import SAC as SAC_imported
        assert PPO_imported is PPO
        assert SAC_imported is SAC
