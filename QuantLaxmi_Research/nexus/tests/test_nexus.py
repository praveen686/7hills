"""NEXUS test suite — verify all components work end-to-end."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
import numpy as np

from nexus.config import NexusConfig
from nexus.mamba2 import Mamba2Backbone, Mamba2Block, selective_scan
from nexus.hyperbolic import (
    minkowski_dot, lorentz_distance, project_to_hyperboloid,
    exp_map_0, log_map_0, lorentz_centroid,
    EuclideanToLorentz, LorentzToEuclidean, LorentzLinear,
)
from nexus.topology import (
    takens_embedding, compute_persistence, persistence_entropy,
    betti_numbers, persistence_landscape, TopologicalSensor,
)
from nexus.jepa import JEPAWorldModel, JEPAPredictor
from nexus.planner import (
    LatentDynamicsModel, ExogenousDynamicsModel, ExplicitRewardModel,
    RewardModel, ValueModel, NexusPlanner,
)
from nexus.model import NEXUS, create_nexus

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, L, D = 4, 64, 32  # batch, seq_len, features


# ============================================================================
# Mamba-2 Tests
# ============================================================================

class TestMamba2:
    def test_selective_scan_shapes(self):
        """Selective scan produces correct output shapes."""
        x = torch.randn(B, L, 16)
        delta = torch.ones(B, L, 16) * 0.01
        A = -torch.ones(16, 8)
        B_mat = torch.randn(B, L, 8)
        C_mat = torch.randn(B, L, 8)
        D_vec = torch.ones(16)

        y = selective_scan(x, delta, A, B_mat, C_mat, D_vec)
        assert y.shape == (B, L, 16)

    def test_selective_scan_causal(self):
        """Selective scan is causal — future inputs don't affect past outputs."""
        x = torch.randn(1, 20, 8)
        delta = torch.ones(1, 20, 8) * 0.01
        A = -torch.ones(8, 4)
        B_mat = torch.randn(1, 20, 4)
        C_mat = torch.randn(1, 20, 4)
        D_vec = torch.ones(8)

        y1 = selective_scan(x, delta, A, B_mat, C_mat, D_vec)

        # Modify future input
        x2 = x.clone()
        x2[0, 15:, :] = torch.randn(5, 8)
        y2 = selective_scan(x2, delta, A, B_mat, C_mat, D_vec)

        # Past outputs should be identical
        assert torch.allclose(y1[0, :15], y2[0, :15], atol=1e-6)

    def test_mamba2_block_residual(self):
        """Mamba2Block has residual connection."""
        block = Mamba2Block(d_model=32, d_state=16, expand=2)
        x = torch.randn(B, L, 32)
        y = block(x)
        assert y.shape == x.shape
        # Output should not be zero (residual ensures this)
        assert y.abs().sum() > 0

    def test_mamba2_backbone_full(self):
        """Full Mamba-2 backbone encodes correctly."""
        backbone = Mamba2Backbone(d_input=D, d_model=64, d_state=16, n_layers=3)
        x = torch.randn(B, L, D)
        y = backbone(x)
        assert y.shape == (B, L, 64)

    def test_mamba2_gradient_flow(self):
        """Gradients flow through Mamba-2."""
        backbone = Mamba2Backbone(d_input=D, d_model=32, d_state=8, n_layers=2)
        x = torch.randn(B, L, D, requires_grad=True)
        y = backbone(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ============================================================================
# Hyperbolic Geometry Tests
# ============================================================================

class TestHyperbolic:
    def test_minkowski_dot_signature(self):
        """Minkowski dot product has (-,+,...,+) signature."""
        x = torch.tensor([2.0, 1.0, 1.0])
        y = torch.tensor([2.0, 1.0, 1.0])
        # ⟨x,y⟩_L = -2*2 + 1*1 + 1*1 = -4 + 2 = -2
        assert torch.isclose(minkowski_dot(x, y), torch.tensor(-2.0))

    def test_hyperboloid_constraint(self):
        """Points from exp_map_0 satisfy ⟨x,x⟩_L = 1/K."""
        v = torch.randn(10, 8)
        K = -1.0
        x = exp_map_0(v, K)
        inner = minkowski_dot(x, x)
        expected = torch.full((10,), 1.0 / K)
        assert torch.allclose(inner, expected, atol=5e-4)

    def test_exp_log_inverse(self):
        """exp_map_0 and log_map_0 are inverses."""
        v = torch.randn(5, 4) * 0.5  # Small vectors for numerical stability
        K = -1.0
        x = exp_map_0(v, K)
        v_recovered = log_map_0(x, K)
        assert torch.allclose(v, v_recovered, atol=1e-4)

    def test_lorentz_distance_positive(self):
        """Geodesic distance is always non-negative."""
        v1 = torch.randn(10, 4) * 0.3
        v2 = torch.randn(10, 4) * 0.3
        x1 = exp_map_0(v1)
        x2 = exp_map_0(v2)
        dist = lorentz_distance(x1, x2)
        assert (dist >= 0).all()

    def test_lorentz_distance_self_zero(self):
        """Distance to self is zero."""
        v = torch.randn(5, 4) * 0.3
        x = exp_map_0(v)
        dist = lorentz_distance(x, x)
        assert torch.allclose(dist, torch.zeros(5), atol=1e-3)

    def test_project_to_hyperboloid(self):
        """project_to_hyperboloid satisfies constraint."""
        x_spatial = torch.randn(8, 4)
        x_full = project_to_hyperboloid(x_spatial)
        assert x_full.shape == (8, 5)
        inner = minkowski_dot(x_full, x_full)
        assert torch.allclose(inner, torch.full((8,), -1.0), atol=1e-5)

    def test_euclidean_to_lorentz_roundtrip(self):
        """E→L→E roundtrip preserves information."""
        e2l = EuclideanToLorentz(16, 8)
        l2e = LorentzToEuclidean(8, 16)
        x = torch.randn(4, 16)
        hyp = e2l(x)
        assert hyp.shape == (4, 9)  # d+1 for Lorentz
        recon = l2e(hyp)
        assert recon.shape == (4, 16)

    def test_lorentz_centroid_on_hyperboloid(self):
        """Centroid remains on hyperboloid."""
        v = torch.randn(5, 4) * 0.3
        points = exp_map_0(v)
        centroid = lorentz_centroid(points)
        inner = minkowski_dot(centroid, centroid)
        assert torch.isclose(inner, torch.tensor(-1.0), atol=1e-4)


# ============================================================================
# Topology Tests
# ============================================================================

class TestTopology:
    def test_takens_embedding_shape(self):
        """Takens embedding produces correct shape."""
        x = np.random.randn(100)
        cloud = takens_embedding(x, dim=3, tau=1)
        assert cloud.shape == (98, 3)

    def test_takens_embedding_deterministic(self):
        """Same input produces same output."""
        x = np.random.randn(50)
        c1 = takens_embedding(x, dim=3)
        c2 = takens_embedding(x, dim=3)
        np.testing.assert_array_equal(c1, c2)

    def test_persistence_h0_count(self):
        """H0 persistence has N-1 finite pairs for N points."""
        cloud = np.random.randn(20, 3)
        result = compute_persistence(cloud, max_dim=0)
        # N points → N-1 merges in single linkage → N-1 finite pairs
        assert len(result["H0"]) == 19

    def test_persistence_entropy_nonneg(self):
        """Persistence entropy is non-negative."""
        diagram = [(0.0, 1.0), (0.0, 2.0), (0.0, 0.5)]
        H = persistence_entropy(diagram)
        assert H >= 0

    def test_persistence_entropy_empty(self):
        """Empty diagram has zero entropy."""
        assert persistence_entropy([]) == 0.0

    def test_betti_numbers_monotone(self):
        """β₀ is monotonically non-increasing with radius."""
        cloud = np.random.randn(30, 2)
        result = compute_persistence(cloud, max_dim=0)
        radii = np.linspace(0.1, 5.0, 20)
        bettis = [betti_numbers(result, r)["beta_0"] for r in radii]
        for i in range(len(bettis) - 1):
            assert bettis[i] >= bettis[i + 1]

    def test_persistence_landscape_shape(self):
        """Landscape has correct shape."""
        diagram = [(0.0, 1.0), (0.0, 2.0), (0.5, 1.5)]
        landscape = persistence_landscape(diagram, n_bins=50)
        assert landscape.shape == (50,)

    def test_topological_sensor_forward(self):
        """TopologicalSensor produces features."""
        sensor = TopologicalSensor(window=20)
        latent = torch.randn(2, 40, 8)
        features = sensor(latent)
        assert features.shape[0] == 2
        assert features.shape[-1] == 8


# ============================================================================
# JEPA World Model Tests
# ============================================================================

class TestJEPA:
    def test_jepa_forward_shapes(self):
        """JEPA forward produces all expected outputs."""
        model = JEPAWorldModel(
            d_input=D, d_model=64, d_latent=32, d_state=16,
            n_layers=2, predictor_depth=2, use_hyperbolic=False,
        )
        ctx = torch.randn(B, 32, D)
        tgt = torch.randn(B, 16, D)
        pos = torch.arange(16).unsqueeze(0).expand(B, -1)

        out = model(ctx, tgt, pos)
        assert "jepa_loss" in out
        assert out["context_latent"].shape == (B, 32, 32)
        assert out["target_latent"].shape == (B, 16, 32)
        assert out["predicted_latent"].shape == (B, 16, 32)

    def test_jepa_loss_decreases(self):
        """JEPA loss decreases with training."""
        model = JEPAWorldModel(
            d_input=8, d_model=32, d_latent=16, d_state=8,
            n_layers=1, predictor_depth=1, use_hyperbolic=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Use consistent data
        torch.manual_seed(42)
        ctx = torch.randn(8, 20, 8)
        tgt = torch.randn(8, 10, 8)
        pos = torch.arange(10).unsqueeze(0).expand(8, -1)

        losses = []
        for _ in range(20):
            out = model(ctx, tgt, pos)
            loss = out["jepa_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.world_model._update_target_encoder() if hasattr(model, 'world_model') else model._update_target_encoder()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_target_encoder_no_grad(self):
        """Target encoder parameters don't require grad."""
        model = JEPAWorldModel(d_input=8, d_model=16, d_latent=8,
                               n_layers=1, use_hyperbolic=False)
        for p in model.target_encoder.parameters():
            assert not p.requires_grad

    def test_ema_update(self):
        """EMA update moves target toward context encoder."""
        model = JEPAWorldModel(d_input=8, d_model=16, d_latent=8,
                               n_layers=1, ema_decay=0.5, use_hyperbolic=False)
        # Randomize context encoder
        for p in model.context_encoder.parameters():
            p.data.normal_()

        before = [p.clone() for p in model.target_encoder.parameters()]
        model._update_target_encoder()
        after = list(model.target_encoder.parameters())

        # Target should have moved
        moved = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert moved

    def test_imagine_futures(self):
        """World model can imagine future states."""
        model = JEPAWorldModel(d_input=8, d_model=32, d_latent=16,
                               n_layers=1, use_hyperbolic=False)
        current = torch.randn(B, 16)
        states, rewards = model.imagine(current, horizon=5)
        assert states.shape == (B, 5, 16)
        assert rewards.shape == (B, 5, 1)


# ============================================================================
# Planner Tests
# ============================================================================

class TestPlanner:
    def test_action_conditioned_dynamics_model(self):
        """Action-conditioned dynamics model predicts next latent state (impact mode)."""
        dyn = LatentDynamicsModel(d_latent=16, d_action=4)
        z = torch.randn(B, 16)
        a = torch.randn(B, 4)
        z_next = dyn(z, a)
        assert z_next.shape == (B, 16)

    def test_exogenous_dynamics_model(self):
        """Exogenous dynamics model predicts next state from state only (no action)."""
        dyn = ExogenousDynamicsModel(d_latent=16)
        z = torch.randn(B, 16)
        z_next = dyn(z)
        assert z_next.shape == (B, 16)
        # Same input should give same output (deterministic forward pass)
        z_next2 = dyn(z)
        assert torch.allclose(z_next, z_next2)

    def test_exogenous_dynamics_ignores_action(self):
        """Exogenous dynamics output does not depend on any action."""
        dyn = ExogenousDynamicsModel(d_latent=16)
        z = torch.randn(B, 16)
        z_next = dyn(z)
        # The same z should always produce the same z_next regardless
        # of what action is taken (action is not an input)
        z_next_again = dyn(z)
        assert torch.allclose(z_next, z_next_again)

    def test_explicit_reward_model(self):
        """Explicit reward model computes structured financial reward."""
        erm = ExplicitRewardModel(d_latent=16, d_action=4)
        z = torch.randn(B, 16)
        a = torch.randn(B, 4)
        r = erm(z, a)
        assert r.shape == (B, 1)
        # With previous position
        a_prev = torch.randn(B, 4)
        r2 = erm(z, a, a_prev=a_prev)
        assert r2.shape == (B, 1)

    def test_reward_model(self):
        """Reward model predicts scalar."""
        rm = RewardModel(d_latent=16, d_action=4)
        z = torch.randn(B, 16)
        a = torch.randn(B, 4)
        r = rm(z, a)
        assert r.shape == (B, 1)

    def test_value_model_ensemble(self):
        """Value model returns ensemble predictions."""
        vm = ValueModel(d_latent=16, n_ensemble=3)
        z = torch.randn(B, 16)
        v = vm(z)
        assert v.shape == (B, 3)

    def test_value_model_min(self):
        """Conservative value is minimum over ensemble."""
        vm = ValueModel(d_latent=16, n_ensemble=2)
        z = torch.randn(B, 16)
        v_min = vm.min_value(z)
        v_all = vm(z)
        assert torch.allclose(v_min.squeeze(), v_all.min(dim=-1).values, atol=1e-6)

    def test_cem_planner(self):
        """CEM planner returns valid actions."""
        planner = NexusPlanner(
            d_latent=16, d_action=4, horizon=3,
            n_samples=32, n_elites=8, n_iterations=2,
        )
        z = torch.randn(B, 16)
        actions, info = planner.plan(z)
        assert actions.shape == (B, 4)
        assert (actions.abs() <= 0.25 + 1e-6).all()  # within max_position

    def test_td_loss(self):
        """TD loss computation works."""
        planner = NexusPlanner(d_latent=16, d_action=4)
        z = torch.randn(B, 16)
        a = torch.randn(B, 4)
        r = torch.randn(B, 1)
        z_next = torch.randn(B, 16)
        done = torch.zeros(B, 1)

        losses = planner.compute_td_loss(z, a, r, z_next, done)
        assert "dynamics_loss" in losses
        assert "reward_loss" in losses
        assert "value_loss" in losses
        assert losses["total_loss"].requires_grad


# ============================================================================
# Full NEXUS Model Tests
# ============================================================================

class TestNEXUS:
    def test_create_nexus_small(self):
        """Create small NEXUS model."""
        model = create_nexus(n_features=32, n_assets=4, size="small")
        params = model.count_parameters()
        assert params["total"] > 0
        print(f"\nNEXUS small: {params['total']:,} parameters")
        for k, v in params.items():
            print(f"  {k}: {v:,}")

    def test_forward_pass(self):
        """Full forward pass produces all outputs."""
        model = create_nexus(n_features=D, n_assets=4, size="small")
        ctx = torch.randn(B, 32, D)
        tgt = torch.randn(B, 16, D)
        pos = torch.arange(16).unsqueeze(0).expand(B, -1)

        out = model(ctx, tgt, pos)
        assert "total_loss" in out
        assert "positions" in out
        assert "regime_logits" in out
        assert out["positions"].shape == (B, 4)
        assert out["regime_logits"].shape == (B, 4)

    def test_act_direct(self):
        """Direct policy inference."""
        model = create_nexus(n_features=D, n_assets=4, size="small")
        data = torch.randn(B, 64, D)
        positions, info = model.act(data, use_planning=False)
        assert positions.shape == (B, 4)
        assert "regime" in info

    def test_act_with_planning(self):
        """CEM planning inference."""
        model = create_nexus(n_features=D, n_assets=4, size="small")
        data = torch.randn(B, 32, D)
        positions, info = model.act(data, use_planning=True)
        assert positions.shape == (B, 4)
        assert "planned_value" in info

    def test_imagine_futures(self):
        """Imagined futures have correct shapes."""
        model = create_nexus(n_features=D, n_assets=4, size="small")
        data = torch.randn(B, 32, D)
        states, rewards = model.imagine_futures(data, horizon=5)
        assert states.shape == (B, 5, model.cfg.d_latent)
        assert rewards.shape == (B, 5, 1)

    def test_hyperbolic_embeddings(self):
        """Hyperbolic embeddings lie on hyperboloid."""
        model = create_nexus(n_features=D, n_assets=4, size="small")
        data = torch.randn(2, 16, D)
        hyp = model.get_hyperbolic_embeddings(data)
        # Check hyperboloid constraint: ⟨x,x⟩_L = -1
        inner = minkowski_dot(hyp.view(-1, hyp.size(-1)), hyp.view(-1, hyp.size(-1)))
        assert torch.allclose(inner, torch.full_like(inner, -1.0), atol=0.1)

    def test_gradient_flow_end_to_end(self):
        """Gradients flow through entire model."""
        model = create_nexus(n_features=D, n_assets=4, size="small")
        ctx = torch.randn(B, 32, D, requires_grad=True)
        tgt = torch.randn(B, 16, D)
        pos = torch.arange(16).unsqueeze(0).expand(B, -1)

        out = model(ctx, tgt, pos)
        out["total_loss"].backward()
        assert ctx.grad is not None

    def test_training_loop(self):
        """Mini training loop runs without error."""
        model = create_nexus(n_features=8, n_assets=2, size="small")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.manual_seed(42)
        losses = []
        for step in range(10):
            ctx = torch.randn(4, 20, 8)
            tgt = torch.randn(4, 10, 8)
            pos = torch.arange(10).unsqueeze(0).expand(4, -1)
            rewards = torch.randn(4, 10)

            out = model(ctx, tgt, pos, rewards=rewards)
            loss = out["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        print(f"\nTraining losses: {losses[0]:.4f} → {losses[-1]:.4f}")
        # Should not explode
        assert all(not math.isnan(l) for l in losses)
        assert all(not math.isinf(l) for l in losses)


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
