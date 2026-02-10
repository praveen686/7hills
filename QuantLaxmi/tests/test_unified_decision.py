"""Tests for UnifiedDecisionLayer, JointTrainingPipeline, HierarchicalDecisionLayer.

Covers:
- Forward pass shapes and value ranges
- Gradient flow through TFT encoder (frozen vs unfrozen)
- Parameter groups with correct LR scales
- JointTrainingPipeline phase ordering and metrics
- HierarchicalDecisionLayer high/low frequency and goal conditioning
- Edge cases: deterministic action, single-sample batch, epoch transitions
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from quantlaxmi.models.rl.integration.unified_decision import (
    UnifiedDecisionLayer,
    JointTrainingPipeline,
    HierarchicalDecisionLayer,
)


# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


class MockTFTEncoder(nn.Module):
    """Minimal TFT-like encoder for testing.

    Accepts (batch, seq_len, n_features) and outputs (batch, hidden_dim).
    Uses a simple linear + LSTM to have real gradients.
    """

    def __init__(self, n_features: int = 10, hidden_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        lstm_out, _ = self.lstm(h)
        return self.out(lstm_out[:, -1, :])


class MockScalarEncoder(nn.Module):
    """Encoder that returns scalar output (batch, 1) like MomentumTransformerModel."""

    def __init__(self, n_features: int = 10) -> None:
        super().__init__()
        self.hidden_dim = 1  # to be detected
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)


class MockEnvironment:
    """Simple environment for testing the pipeline.

    State: (seq_len, n_features) random market data.
    Action: (action_dim,) continuous.
    Reward: random but deterministic given action.
    Terminates after max_steps steps.
    """

    def __init__(
        self,
        seq_len: int = 20,
        n_features: int = 10,
        action_dim: int = 1,
        max_steps: int = 50,
    ) -> None:
        self.seq_len = seq_len
        self.n_features = n_features
        self.action_dim = action_dim
        self.max_steps = max_steps
        self._step = 0
        self._rng = np.random.default_rng(42)

    def reset(self) -> np.ndarray:
        self._step = 0
        return self._rng.standard_normal(
            (self.seq_len, self.n_features)
        ).astype(np.float32)

    def step(self, action: np.ndarray):
        self._step += 1
        done = self._step >= self.max_steps
        reward = float(np.sum(action) * 0.01 + self._rng.standard_normal() * 0.1)
        next_state = self._rng.standard_normal(
            (self.seq_len, self.n_features)
        ).astype(np.float32)
        info = {"step": self._step}
        return next_state, reward, done, info


class MockGymnasiumEnv:
    """Environment returning 5-tuple (Gymnasium API)."""

    def __init__(self, seq_len: int = 20, n_features: int = 10, max_steps: int = 50):
        self.seq_len = seq_len
        self.n_features = n_features
        self.max_steps = max_steps
        self._step = 0
        self._rng = np.random.default_rng(123)

    def reset(self):
        self._step = 0
        obs = self._rng.standard_normal(
            (self.seq_len, self.n_features)
        ).astype(np.float32)
        return obs, {"info": True}

    def step(self, action):
        self._step += 1
        terminated = self._step >= self.max_steps
        truncated = False
        reward = float(np.sum(action) * 0.01)
        obs = self._rng.standard_normal(
            (self.seq_len, self.n_features)
        ).astype(np.float32)
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encoder():
    return MockTFTEncoder(n_features=10, hidden_dim=32)


@pytest.fixture
def unified_model(encoder):
    return UnifiedDecisionLayer(
        tft_encoder=encoder,
        freeze_tft_epochs=5,
        tft_lr_scale=0.1,
        action_dim=1,
    )


@pytest.fixture
def mock_env():
    return MockEnvironment(seq_len=20, n_features=10, max_steps=50)


@pytest.fixture
def pipeline(unified_model, mock_env):
    return JointTrainingPipeline(
        unified_model=unified_model,
        env=mock_env,
        config={"base_lr": 1e-3, "batch_size": 16},
    )


# ============================================================================
# UnifiedDecisionLayer tests
# ============================================================================


class TestUnifiedDecisionLayer:
    """Tests for the UnifiedDecisionLayer module."""

    def test_forward_output_keys(self, unified_model):
        """Forward returns dict with all required keys."""
        x = torch.randn(4, 20, 10)
        out = unified_model(x)
        assert set(out.keys()) == {"action", "value", "features", "log_prob", "entropy"}

    def test_forward_action_shape(self, unified_model):
        """Action shape matches (batch, action_dim)."""
        x = torch.randn(8, 20, 10)
        out = unified_model(x)
        assert out["action"].shape == (8, 1)

    def test_forward_action_range(self, unified_model):
        """Actions are in [-1, 1] due to Tanh squashing."""
        x = torch.randn(16, 20, 10)
        out = unified_model(x)
        assert out["action"].min() >= -1.0
        assert out["action"].max() <= 1.0

    def test_forward_value_shape(self, unified_model):
        """Value estimate shape is (batch, 1)."""
        x = torch.randn(4, 20, 10)
        out = unified_model(x)
        assert out["value"].shape == (4, 1)

    def test_forward_log_prob_shape(self, unified_model):
        """Log probability shape is (batch,)."""
        x = torch.randn(4, 20, 10)
        out = unified_model(x)
        assert out["log_prob"].shape == (4,)

    def test_forward_entropy_shape(self, unified_model):
        """Entropy shape is (batch,)."""
        x = torch.randn(4, 20, 10)
        out = unified_model(x)
        assert out["entropy"].shape == (4,)

    def test_encode_output_shape(self, unified_model):
        """Encode returns (batch, feature_dim)."""
        x = torch.randn(4, 20, 10)
        features = unified_model.encode(x)
        assert features.shape == (4, 32)  # hidden_dim = 32

    def test_deterministic_action(self, unified_model):
        """act_deterministic returns consistent actions (no randomness)."""
        x = torch.randn(4, 20, 10)
        unified_model.eval()
        out1 = unified_model.act_deterministic(x)
        out2 = unified_model.act_deterministic(x)
        assert torch.allclose(out1["action"], out2["action"])

    def test_gradient_flows_when_unfrozen(self, encoder):
        """Gradients flow through TFT encoder when unfrozen."""
        model = UnifiedDecisionLayer(
            tft_encoder=encoder,
            freeze_tft_epochs=0,  # never frozen
            tft_lr_scale=0.1,
        )
        x = torch.randn(4, 20, 10)
        out = model(x)
        loss = out["action"].sum()
        loss.backward()

        # Check TFT encoder has gradients
        has_grad = False
        for p in encoder.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "TFT encoder should have gradients when unfrozen"

    def test_gradient_does_not_flow_when_frozen(self, encoder):
        """Gradients do NOT flow through TFT encoder when frozen."""
        model = UnifiedDecisionLayer(
            tft_encoder=encoder,
            freeze_tft_epochs=100,  # frozen for a long time
            tft_lr_scale=0.1,
        )
        model.set_epoch(0)  # epoch < 100 => frozen

        x = torch.randn(4, 20, 10)
        out = model(x)
        loss = out["action"].sum()
        loss.backward()

        # Check TFT encoder does NOT have gradients
        for p in encoder.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0, (
                "TFT encoder should not have gradients when frozen"
            )

    def test_freeze_unfreeze_epoch_transition(self, encoder):
        """TFT encoder transitions from frozen to unfrozen at the right epoch."""
        model = UnifiedDecisionLayer(
            tft_encoder=encoder,
            freeze_tft_epochs=3,
        )

        # Epochs 0, 1, 2 => frozen
        for epoch in range(3):
            model.set_epoch(epoch)
            assert model.is_tft_frozen(), f"Should be frozen at epoch {epoch}"

        # Epoch 3 => unfrozen
        model.set_epoch(3)
        assert not model.is_tft_frozen(), "Should be unfrozen at epoch 3"

    def test_parameter_groups_correct_lr_scales(self, unified_model):
        """Parameter groups have correct LR scales."""
        groups = unified_model.get_parameter_groups()

        # Should have at least 2 groups: tft_encoder and policy_network
        assert len(groups) >= 2

        tft_group = next(g for g in groups if g["name"] == "tft_encoder")
        policy_group = next(g for g in groups if g["name"] == "policy_network")

        assert tft_group["lr_scale"] == 0.1
        assert policy_group["lr_scale"] == 1.0

    def test_parameter_groups_cover_all_params(self, unified_model):
        """Parameter groups collectively cover all model parameters."""
        groups = unified_model.get_parameter_groups()
        group_param_ids = set()
        for g in groups:
            for p in g["params"]:
                group_param_ids.add(id(p))

        all_param_ids = {id(p) for p in unified_model.parameters()}
        assert group_param_ids == all_param_ids

    def test_build_optimizer_differential_lr(self, unified_model):
        """build_optimizer creates optimizer with differential LR."""
        optimizer = unified_model.build_optimizer(base_lr=1e-3)
        assert len(optimizer.param_groups) >= 2

        # Find TFT group (lower LR)
        lrs = sorted(set(g["lr"] for g in optimizer.param_groups))
        assert len(lrs) >= 2, "Should have at least 2 distinct learning rates"
        # TFT LR should be 0.1 * base_lr
        assert min(lrs) == pytest.approx(1e-4, rel=1e-6)

    def test_scalar_encoder_handling(self):
        """UnifiedDecisionLayer handles scalar encoder output (batch, 1)."""
        enc = MockScalarEncoder(n_features=10)
        model = UnifiedDecisionLayer(
            tft_encoder=enc,
            feature_dim=32,  # explicit since hidden_dim=1 is ambiguous
            action_dim=1,
        )
        x = torch.randn(4, 20, 10)
        out = model(x)
        assert out["action"].shape == (4, 1)

    def test_single_sample_batch(self, unified_model):
        """Works with batch size 1."""
        x = torch.randn(1, 20, 10)
        out = unified_model(x)
        assert out["action"].shape == (1, 1)
        assert out["log_prob"].shape == (1,)

    def test_feature_dim_property(self, unified_model):
        """feature_dim property returns correct value."""
        assert unified_model.feature_dim == 32


# ============================================================================
# JointTrainingPipeline tests
# ============================================================================


class TestJointTrainingPipeline:
    """Tests for the JointTrainingPipeline."""

    def test_phase1_returns_metrics(self, pipeline):
        """Phase 1 returns dict with losses, final_loss, best_epoch."""
        # Generate mock training data: (n_samples, seq_len, n_features+1)
        # Last feature is the return target.
        data = np.random.randn(32, 20, 11).astype(np.float32)
        metrics = pipeline.phase1_pretrain_tft(data, epochs=3)

        assert "losses" in metrics
        assert "final_loss" in metrics
        assert "best_epoch" in metrics
        assert len(metrics["losses"]) == 3

    def test_phase2_returns_metrics(self, pipeline):
        """Phase 2 returns dict with actor/critic losses."""
        metrics = pipeline.phase2_train_rl(n_steps=100)

        assert "actor_losses" in metrics
        assert "critic_losses" in metrics
        assert "entropies" in metrics
        assert "total_steps" in metrics
        assert metrics["total_steps"] >= 100

    def test_phase3_returns_metrics(self, pipeline):
        """Phase 3 returns dict with losses after joint fine-tuning."""
        metrics = pipeline.phase3_joint_finetune(n_steps=100)

        assert "actor_losses" in metrics
        assert "critic_losses" in metrics
        assert "total_steps" in metrics

    def test_phase4_returns_evaluation(self, pipeline):
        """Phase 4 returns evaluation metrics."""
        test_data = np.random.randn(20, 20, 11).astype(np.float32)
        metrics = pipeline.phase4_evaluate(test_data)

        assert "positions" in metrics
        assert "returns" in metrics
        assert "sharpe" in metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert metrics["n_samples"] == 20

    def test_phase_ordering_tracked(self, pipeline):
        """Phase history is tracked in order."""
        data = np.random.randn(32, 20, 11).astype(np.float32)
        test_data = np.random.randn(10, 20, 11).astype(np.float32)

        pipeline.phase1_pretrain_tft(data, epochs=2)
        pipeline.phase2_train_rl(n_steps=50)
        pipeline.phase3_joint_finetune(n_steps=50)
        pipeline.phase4_evaluate(test_data)

        assert pipeline.phase_history == [
            "phase1_pretrain_tft",
            "phase2_train_rl",
            "phase3_joint_finetune",
            "phase4_evaluate",
        ]

    def test_full_pipeline_runs(self, pipeline):
        """run_full_pipeline executes all 4 phases."""
        train_data = np.random.randn(32, 20, 11).astype(np.float32)
        test_data = np.random.randn(10, 20, 11).astype(np.float32)

        result = pipeline.run_full_pipeline(
            train_data,
            test_data,
            pretrain_epochs=2,
            rl_steps=50,
            finetune_steps=50,
        )

        assert "phase1" in result
        assert "phase2" in result
        assert "phase3" in result
        assert "phase4" in result
        assert result["phase_history"] == [
            "phase1_pretrain_tft",
            "phase2_train_rl",
            "phase3_joint_finetune",
            "phase4_evaluate",
        ]

    def test_tft_frozen_during_phase2(self, pipeline):
        """TFT encoder is frozen during phase 2."""
        pipeline.phase2_train_rl(n_steps=50)
        assert pipeline.model.is_tft_frozen()

    def test_tft_unfrozen_during_phase3(self, pipeline):
        """TFT encoder is unfrozen during phase 3."""
        pipeline.phase3_joint_finetune(n_steps=50)
        assert not pipeline.model.is_tft_frozen()

    def test_gymnasium_env_compatibility(self, unified_model):
        """Pipeline works with 5-tuple Gymnasium environments."""
        env = MockGymnasiumEnv(seq_len=20, n_features=10, max_steps=30)
        pipe = JointTrainingPipeline(
            unified_model=unified_model,
            env=env,
            config={"batch_size": 16},
        )
        metrics = pipe.phase2_train_rl(n_steps=60)
        assert metrics["total_steps"] >= 60

    def test_gae_computation(self, pipeline):
        """GAE advantages have correct shape and are finite."""
        rewards = np.array([0.1, -0.05, 0.2, 0.0, 0.1], dtype=np.float32)
        values = np.array([0.5, 0.4, 0.6, 0.3, 0.5], dtype=np.float32)
        dones = np.array([0, 0, 0, 0, 1], dtype=np.float32)

        advantages, returns = pipeline._compute_gae(rewards, values, dones)

        assert advantages.shape == (5,)
        assert returns.shape == (5,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))


# ============================================================================
# HierarchicalDecisionLayer tests
# ============================================================================


class TestHierarchicalDecisionLayer:
    """Tests for the HierarchicalDecisionLayer."""

    @pytest.fixture
    def hier_model(self):
        return HierarchicalDecisionLayer(
            high_freq=1,
            low_freq=375,
            high_state_dim=64,
            low_state_dim=32,
            n_regimes=3,
        )

    def test_high_level_step_output_keys(self, hier_model):
        """High-level step returns correct output keys."""
        state = torch.randn(2, 64)
        out = hier_model.high_level_step(state)
        assert "target_position" in out
        assert "regime_label" in out
        assert "confidence" in out
        assert "value" in out
        assert "regime_probs" in out

    def test_high_level_target_position_range(self, hier_model):
        """High-level target position is in [-1, 1]."""
        state = torch.randn(8, 64)
        out = hier_model.high_level_step(state)
        assert out["target_position"].min() >= -1.0
        assert out["target_position"].max() <= 1.0

    def test_high_level_regime_labels_valid(self, hier_model):
        """Regime labels are valid indices in [0, n_regimes)."""
        state = torch.randn(4, 64)
        out = hier_model.high_level_step(state)
        assert (out["regime_label"] >= 0).all()
        assert (out["regime_label"] < 3).all()

    def test_low_level_step_output_keys(self, hier_model):
        """Low-level step returns correct output keys."""
        # First set a goal via high-level
        daily_state = torch.randn(2, 64)
        hier_model.high_level_step(daily_state)

        intraday_state = torch.randn(2, 32)
        out = hier_model.low_level_step(intraday_state)
        assert "order_action" in out
        assert "execution_progress" in out
        assert "value" in out

    def test_low_level_requires_goal(self, hier_model):
        """Low-level step raises error if no goal is available."""
        intraday_state = torch.randn(2, 32)
        with pytest.raises(ValueError, match="No goal available"):
            hier_model.low_level_step(intraday_state)

    def test_goal_conditioning_changes_output(self, hier_model):
        """Low-level output changes with different goals."""
        intraday_state = torch.randn(4, 32)

        # Goal 1: strong long
        goal1 = torch.full((4, 1), 0.9)
        out1 = hier_model.low_level_step(intraday_state, goal=goal1)

        # Goal 2: strong short
        goal2 = torch.full((4, 1), -0.9)
        out2 = hier_model.low_level_step(intraday_state, goal=goal2)

        # Actions should differ given different goals
        assert not torch.allclose(
            out1["order_action"], out2["order_action"], atol=1e-4
        ), "Low-level actions should differ with different goals"

    def test_forward_combined(self, hier_model):
        """Combined forward pass produces both high and low outputs."""
        daily_state = torch.randn(2, 64)
        intraday_state = torch.randn(2, 32)

        out = hier_model(daily_state, intraday_state)
        assert "high" in out
        assert "low" in out
        assert "target_position" in out["high"]
        assert "order_action" in out["low"]

    def test_high_low_frequency_attributes(self, hier_model):
        """Frequency attributes are correctly set."""
        assert hier_model.high_freq == 1
        assert hier_model.low_freq == 375

    def test_confidence_in_0_1(self, hier_model):
        """Confidence output is in [0, 1] (sigmoid)."""
        state = torch.randn(4, 64)
        out = hier_model.high_level_step(state)
        assert out["confidence"].min() >= 0.0
        assert out["confidence"].max() <= 1.0

    def test_execution_progress_in_0_1(self, hier_model):
        """Execution progress is in [0, 1] (sigmoid)."""
        daily_state = torch.randn(2, 64)
        hier_model.high_level_step(daily_state)

        intraday_state = torch.randn(2, 32)
        out = hier_model.low_level_step(intraday_state)
        assert out["execution_progress"].min() >= 0.0
        assert out["execution_progress"].max() <= 1.0

    def test_regime_probs_sum_to_one(self, hier_model):
        """Regime probabilities sum to 1."""
        state = torch.randn(4, 64)
        out = hier_model.high_level_step(state)
        sums = out["regime_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows_hierarchical(self, hier_model):
        """Gradients flow through both high and low level policies."""
        daily_state = torch.randn(2, 64, requires_grad=True)
        intraday_state = torch.randn(2, 32, requires_grad=True)

        # Forward with gradient tracking (not using cached detached goal)
        high_out = hier_model.high_level(daily_state)
        goal = high_out["target_position"]  # not detached
        low_out = hier_model.low_level(intraday_state, goal)

        loss = low_out["action"].sum() + high_out["target_position"].sum()
        loss.backward()

        # High-level should have gradients
        has_high_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in hier_model.high_level.parameters()
        )
        assert has_high_grad, "High-level policy should have gradients"

        # Low-level should have gradients
        has_low_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in hier_model.low_level.parameters()
        )
        assert has_low_grad, "Low-level policy should have gradients"
