"""Tests for TFT architecture upgrades.

Tests cover:
1. Flash/SDPA Attention — output shape, causal masking
2. AMP — forward pass produces finite outputs
3. ALiBi — bias matrix shape and geometric slope values
4. Multi-horizon forecasting — output shapes, loss weighting
5. Cosine Annealing LR with warmup — schedule shape
6. Backward compatibility — multi_horizon=False matches original behavior
"""

from __future__ import annotations

import math
import pytest
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Use CPU for tests (deterministic, no GPU required)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 20


@pytest.fixture
def n_features():
    return 10


@pytest.fixture
def hidden_dim():
    return 32


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def sample_input(batch_size, seq_len, n_features, device):
    """Random input tensor for model tests."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, n_features, device=device)


# ============================================================================
# ALiBi Tests
# ============================================================================

class TestALiBi:
    """Tests for ALiBi (Attention with Linear Biases) positional encoding."""

    def test_alibi_import(self):
        """ALiBi class can be imported from momentum_tfm."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        assert ALiBi is not None

    def test_alibi_bias_shape(self, n_heads, seq_len):
        """ALiBi bias has shape (n_heads, seq_len, seq_len)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        bias = alibi(seq_len)
        assert bias.shape == (n_heads, seq_len, seq_len)

    def test_alibi_geometric_slopes(self, n_heads):
        """Slopes follow geometric sequence m_h = 2^(-8h/n_heads)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        expected_slopes = [2.0 ** (-8.0 * i / n_heads) for i in range(1, n_heads + 1)]
        actual_slopes = alibi.slopes.cpu().numpy()
        np.testing.assert_allclose(actual_slopes, expected_slopes, rtol=1e-6)

    def test_alibi_diagonal_is_zero(self, n_heads, seq_len):
        """Diagonal of ALiBi bias should be zero (distance 0)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        bias = alibi(seq_len)
        for h in range(n_heads):
            diag = torch.diagonal(bias[h])
            assert torch.allclose(diag, torch.zeros_like(diag)), \
                f"Head {h} diagonal is not zero"

    def test_alibi_values_negative(self, n_heads, seq_len):
        """ALiBi bias values should be <= 0 (negative biases for distant positions)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        bias = alibi(seq_len)
        assert (bias <= 0.0 + 1e-7).all(), "ALiBi bias should be non-positive"

    def test_alibi_symmetry(self, n_heads, seq_len):
        """ALiBi uses |i-j| so bias should be symmetric for each head."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        bias = alibi(seq_len)
        for h in range(n_heads):
            assert torch.allclose(bias[h], bias[h].T, atol=1e-6), \
                f"Head {h} bias is not symmetric"

    def test_alibi_variable_seq_len(self, n_heads):
        """ALiBi should handle different sequence lengths dynamically."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        for s in [5, 10, 50, 100]:
            bias = alibi(s)
            assert bias.shape == (n_heads, s, s)

    def test_alibi_no_learnable_params(self, n_heads):
        """ALiBi should have zero learnable parameters."""
        from quantlaxmi.models.ml.tft.momentum_tfm import ALiBi
        alibi = ALiBi(n_heads)
        n_params = sum(p.numel() for p in alibi.parameters() if p.requires_grad)
        assert n_params == 0, f"ALiBi has {n_params} learnable params, expected 0"


# ============================================================================
# SDPAttention Tests
# ============================================================================

class TestSDPAttention:
    """Tests for Scaled Dot-Product Attention with Flash/memory-efficient backends."""

    def test_sdpa_output_shape(self, batch_size, seq_len, hidden_dim, n_heads, device):
        """SDPA produces output of correct shape."""
        from quantlaxmi.models.ml.tft.momentum_tfm import SDPAttention
        attn = SDPAttention(hidden_dim, n_heads, dropout=0.0, use_alibi=True).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        out, weights = attn(x, is_causal=True)
        assert out.shape == (batch_size, seq_len, hidden_dim)

    def test_sdpa_without_alibi(self, batch_size, seq_len, hidden_dim, n_heads, device):
        """SDPA works without ALiBi (use_alibi=False)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import SDPAttention
        attn = SDPAttention(hidden_dim, n_heads, dropout=0.0, use_alibi=False).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        out, _ = attn(x, is_causal=True)
        assert out.shape == (batch_size, seq_len, hidden_dim)
        assert torch.isfinite(out).all()

    def test_sdpa_causal_masking(self, hidden_dim, n_heads, device):
        """Causal masking: output at position t should not depend on positions > t."""
        from quantlaxmi.models.ml.tft.momentum_tfm import SDPAttention
        attn = SDPAttention(hidden_dim, n_heads, dropout=0.0, use_alibi=False).to(device)
        attn.eval()

        # Create input where future positions have very different values
        torch.manual_seed(42)
        x = torch.randn(1, 10, hidden_dim, device=device)

        # Full sequence output
        out_full, _ = attn(x, is_causal=True)

        # Now change positions 5-9 (future relative to position 4)
        x_modified = x.clone()
        x_modified[:, 5:, :] = torch.randn(1, 5, hidden_dim, device=device) * 100.0

        out_modified, _ = attn(x_modified, is_causal=True)

        # Positions 0-4 should be identical (they only attend to 0-4)
        for t in range(5):
            assert torch.allclose(out_full[:, t, :], out_modified[:, t, :], atol=1e-5), \
                f"Position {t} output changed when future was modified -- causal mask broken"

    def test_sdpa_with_attention_weights(self, batch_size, seq_len, hidden_dim, n_heads, device):
        """When need_weights=True, attention weights are returned (fallback path)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import SDPAttention
        attn = SDPAttention(hidden_dim, n_heads, dropout=0.0, use_alibi=True).to(device)
        attn.eval()
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        out, weights = attn(x, is_causal=True, need_weights=True)
        assert out.shape == (batch_size, seq_len, hidden_dim)
        assert weights is not None
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_sdpa_finite_outputs(self, batch_size, seq_len, hidden_dim, n_heads, device):
        """SDPA output values should all be finite."""
        from quantlaxmi.models.ml.tft.momentum_tfm import SDPAttention
        attn = SDPAttention(hidden_dim, n_heads, dropout=0.0, use_alibi=True).to(device)
        attn.eval()
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        out, _ = attn(x)
        assert torch.isfinite(out).all(), "SDPA produced non-finite values"


# ============================================================================
# Full Model Tests (MomentumTransformerModel)
# ============================================================================

class TestMomentumTransformerModel:
    """Tests for the upgraded MomentumTransformerModel."""

    def test_model_sdpa_output_shape(self, sample_input, n_features, hidden_dim, n_heads, device):
        """Model with SDPA produces correct output shape."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, use_alibi=True, multi_horizon=False,
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert out.shape == (sample_input.shape[0], 1)

    def test_model_legacy_attn_output_shape(self, sample_input, n_features, hidden_dim, n_heads, device):
        """Model with legacy nn.MultiheadAttention produces correct output shape."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=False, use_alibi=False, multi_horizon=False,
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert out.shape == (sample_input.shape[0], 1)

    def test_backward_compat_single_output(self, sample_input, n_features, hidden_dim, n_heads, device):
        """multi_horizon=False returns a single tensor (not a tuple)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, multi_horizon=False,
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert isinstance(out, torch.Tensor), "Expected single tensor when multi_horizon=False"
        assert out.shape == (sample_input.shape[0], 1)

    def test_classify_works_with_sdpa(self, sample_input, n_features, hidden_dim, n_heads, device):
        """classify() method works with SDPA attention."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, use_alibi=True,
        ).to(device)
        model.eval()
        with torch.no_grad():
            logits = model.classify(sample_input)
        assert logits.shape == (sample_input.shape[0], 1)
        assert torch.isfinite(logits).all()

    def test_output_in_range(self, sample_input, n_features, hidden_dim, n_heads, device):
        """Model output should be in [-1, 1] due to Tanh activation."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, multi_horizon=False,
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert (out >= -1.0).all() and (out <= 1.0).all(), \
            f"Output out of [-1,1] range: min={out.min()}, max={out.max()}"


# ============================================================================
# Multi-Horizon Tests
# ============================================================================

class TestMultiHorizon:
    """Tests for multi-horizon forecasting."""

    def test_multi_horizon_output_count(self, sample_input, n_features, hidden_dim, n_heads, device):
        """Multi-horizon model returns 4 outputs (T+1, T+2, T+3, T+5)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, multi_horizon=True,
            aux_horizons=(2, 3, 5),
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input)
        assert isinstance(outputs, tuple), "Multi-horizon should return a tuple"
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

    def test_multi_horizon_shapes(self, sample_input, n_features, hidden_dim, n_heads, device):
        """Each multi-horizon output has shape (batch, 1)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, multi_horizon=True,
            aux_horizons=(2, 3, 5),
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input)
        for i, out in enumerate(outputs):
            assert out.shape == (sample_input.shape[0], 1), \
                f"Output {i} has shape {out.shape}, expected ({sample_input.shape[0]}, 1)"

    def test_multi_horizon_all_in_range(self, sample_input, n_features, hidden_dim, n_heads, device):
        """All multi-horizon outputs should be in [-1, 1]."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=True, multi_horizon=True,
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input)
        for i, out in enumerate(outputs):
            assert (out >= -1.0).all() and (out <= 1.0).all(), \
                f"Output {i} out of range"

    def test_multi_horizon_with_attention(self, sample_input, n_features, hidden_dim, n_heads, device):
        """return_attention=True appends weights as last element."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0,
            use_sdpa=False, use_alibi=False, multi_horizon=True,
            aux_horizons=(2, 3, 5),
        ).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input, return_attention=True)
        # 4 position outputs + 1 attention weights = 5
        assert len(outputs) == 5, f"Expected 5 elements, got {len(outputs)}"

    def test_multi_horizon_loss_function(self, device):
        """multi_horizon_loss computes weighted sum of per-horizon Sharpe losses."""
        from quantlaxmi.models.ml.tft.momentum_tfm import multi_horizon_loss

        torch.manual_seed(42)
        B = 32
        preds = tuple(torch.randn(B, 1, device=device) for _ in range(4))
        returns_dict = {
            1: torch.randn(B, device=device),
            2: torch.randn(B, device=device),
            3: torch.randn(B, device=device),
            5: torch.randn(B, device=device),
        }
        loss = multi_horizon_loss(preds, returns_dict)
        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_multi_horizon_loss_weights(self, device):
        """Auxiliary losses are weighted by the specified weights."""
        from quantlaxmi.models.ml.tft.momentum_tfm import multi_horizon_loss, sharpe_loss

        torch.manual_seed(42)
        B = 64
        preds = tuple(torch.randn(B, 1, device=device) for _ in range(4))
        returns_dict = {
            1: torch.randn(B, device=device),
            2: torch.randn(B, device=device),
            3: torch.randn(B, device=device),
            5: torch.randn(B, device=device),
        }

        # Compute manually
        l1 = sharpe_loss(preds[0].squeeze(-1), returns_dict[1])
        l2 = sharpe_loss(preds[1].squeeze(-1), returns_dict[2])
        l3 = sharpe_loss(preds[2].squeeze(-1), returns_dict[3])
        l5 = sharpe_loss(preds[3].squeeze(-1), returns_dict[5])
        expected = l1 + 0.3 * l2 + 0.2 * l3 + 0.1 * l5

        actual = multi_horizon_loss(preds, returns_dict, aux_weights=(0.3, 0.2, 0.1))
        assert torch.allclose(actual, expected, atol=1e-5), \
            f"Loss mismatch: {actual.item():.6f} vs {expected.item():.6f}"

    def test_multi_horizon_aux_heads_separate(self, n_features, hidden_dim, n_heads, device):
        """Auxiliary heads have separate parameters from the main output head."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, multi_horizon=True,
        ).to(device)
        # Main output_head parameters should not be in aux_heads
        main_params = set(id(p) for p in model.output_head.parameters())
        for key, head in model.aux_heads.items():
            aux_params = set(id(p) for p in head.parameters())
            assert main_params.isdisjoint(aux_params), \
                f"Aux head {key} shares parameters with main output head"


# ============================================================================
# AMP Tests
# ============================================================================

class TestAMP:
    """Tests for Automatic Mixed Precision support."""

    def test_amp_forward_cpu(self, sample_input, n_features, hidden_dim, n_heads, device):
        """AMP autocast on CPU should work (uses bfloat16 if supported, else noop)."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0, use_sdpa=True,
        ).to(device)
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cpu", enabled=True):
                out = model(sample_input)
        assert torch.isfinite(out).all(), "AMP forward produced non-finite outputs"
        assert out.shape == (sample_input.shape[0], 1)

    def test_amp_gradscaler_noop_on_cpu(self):
        """GradScaler should be a no-op on CPU (no crash)."""
        # On CPU, GradScaler is disabled but should not crash
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        x = torch.randn(4, requires_grad=True)
        loss = x.sum()
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD([x], lr=0.01))
        scaler.update()
        # No crash = pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_amp_forward_cuda(self, n_features, hidden_dim, n_heads):
        """AMP autocast on CUDA produces finite float16/float32 outputs."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        dev = torch.device("cuda")
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, dropout=0.0, use_sdpa=True,
        ).to(dev)
        model.eval()
        x = torch.randn(4, 20, n_features, device=dev)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                out = model(x)
        assert torch.isfinite(out).all()


# ============================================================================
# Cosine Annealing LR + Warmup Tests
# ============================================================================

class TestCosineWarmupLR:
    """Tests for cosine annealing LR scheduler with warmup."""

    def _build_scheduler(self, warmup_steps, total_steps, base_lr=0.001):
        """Helper to build the scheduler."""
        from quantlaxmi.models.ml.tft.production.training_pipeline import _build_cosine_warmup_scheduler
        param = torch.nn.Parameter(torch.randn(10))
        optimizer = torch.optim.Adam([param], lr=base_lr)
        scheduler = _build_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)
        # Wrap step to call optimizer.step() first (silences PyTorch warning;
        # we're only testing LR values, not training)
        _orig_step = scheduler.step
        def _step_with_dummy():
            optimizer.step()
            _orig_step()
        scheduler.step = _step_with_dummy
        return optimizer, scheduler

    def test_lr_starts_at_zero(self):
        """LR starts near 0 during warmup (step=0)."""
        opt, sched = self._build_scheduler(warmup_steps=10, total_steps=100, base_lr=0.001)
        # At step 0, lr_lambda returns 0/10 = 0
        lr = opt.param_groups[0]["lr"]
        # After scheduler init, lr is still base_lr * lr_lambda(0) = 0
        # But LambdaLR only modifies on step(), so check after step 0
        sched.step()  # step 1
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after < 0.001, f"LR after step 1 should be < base_lr, got {lr_after}"

    def test_lr_reaches_base_after_warmup(self):
        """LR should be approximately base_lr at the end of warmup."""
        base_lr = 0.001
        opt, sched = self._build_scheduler(warmup_steps=10, total_steps=100, base_lr=base_lr)
        for _ in range(10):
            sched.step()
        lr = opt.param_groups[0]["lr"]
        # At step=warmup_steps, lambda = warmup_steps/warmup_steps = 1.0
        # But cosine phase starts at 1.0 too, so should be ~base_lr
        assert abs(lr - base_lr) < base_lr * 0.15, \
            f"LR at warmup end should be ~{base_lr}, got {lr}"

    def test_lr_decreasing_after_warmup(self):
        """After warmup, LR should decrease (cosine decay)."""
        opt, sched = self._build_scheduler(warmup_steps=10, total_steps=100)
        # Advance past warmup
        for _ in range(15):
            sched.step()
        lr_at_15 = opt.param_groups[0]["lr"]
        for _ in range(30):
            sched.step()
        lr_at_45 = opt.param_groups[0]["lr"]
        assert lr_at_45 < lr_at_15, \
            f"LR should decrease after warmup: {lr_at_15} -> {lr_at_45}"

    def test_lr_near_zero_at_end(self):
        """LR should approach 0 at the end of the schedule."""
        opt, sched = self._build_scheduler(warmup_steps=10, total_steps=100)
        for _ in range(100):
            sched.step()
        lr = opt.param_groups[0]["lr"]
        assert lr < 1e-5, f"LR at end of schedule should be near 0, got {lr}"

    def test_warmup_is_linear(self):
        """During warmup, LR increases linearly."""
        base_lr = 0.01
        warmup_steps = 20
        opt, sched = self._build_scheduler(warmup_steps=warmup_steps, total_steps=200, base_lr=base_lr)
        lrs = []
        for step in range(warmup_steps):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # Each step should increase by ~base_lr/warmup_steps
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], \
                f"LR should increase during warmup: step {i}: {lrs[i-1]} -> {lrs[i]}"

    def test_cosine_shape(self):
        """The schedule after warmup follows a cosine curve (smooth, non-linear)."""
        base_lr = 0.01
        opt, sched = self._build_scheduler(warmup_steps=10, total_steps=110, base_lr=base_lr)
        # Skip warmup
        for _ in range(10):
            sched.step()
        # Collect cosine phase LRs
        lrs = []
        for _ in range(100):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # Check it's not purely linear (cosine is convex then concave)
        mid = len(lrs) // 2
        # The midpoint LR should be ~50% of base_lr (cosine property)
        assert abs(lrs[mid] - base_lr * 0.5) < base_lr * 0.15, \
            f"Midpoint LR {lrs[mid]} should be near {base_lr * 0.5}"


# ============================================================================
# Config Backward Compatibility Tests
# ============================================================================

class TestConfigBackwardCompat:
    """Tests ensuring new config defaults don't break existing behavior."""

    def test_momentum_config_defaults(self):
        """New config fields have backward-compatible defaults."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTFMConfig
        cfg = MomentumTFMConfig()
        assert cfg.multi_horizon is False
        assert cfg.use_alibi is True
        assert cfg.use_sdpa is True
        assert cfg.aux_horizons == (2, 3, 5)
        assert cfg.aux_weights == (0.3, 0.2, 0.1)

    def test_pipeline_config_defaults(self):
        """TrainingPipelineConfig has AMP and cosine LR defaults."""
        from quantlaxmi.models.ml.tft.production.training_pipeline import TrainingPipelineConfig
        cfg = TrainingPipelineConfig()
        assert cfg.use_amp is True
        assert cfg.use_cosine_lr is True
        assert cfg.warmup_fraction == 0.1

    def test_model_default_creates_sdpa(self, n_features, hidden_dim, n_heads, device):
        """Default MomentumTransformerModel uses SDPA path."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim, n_heads=n_heads,
        ).to(device)
        # use_sdpa defaults to True
        assert model.use_sdpa is True
        assert model._legacy_attn is False

    def test_model_legacy_path(self, n_features, hidden_dim, n_heads, device):
        """use_sdpa=False uses legacy nn.MultiheadAttention."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim, n_heads=n_heads,
            use_sdpa=False,
        ).to(device)
        assert model._legacy_attn is True


# ============================================================================
# Integration / Gradient Flow Tests
# ============================================================================

class TestGradientFlow:
    """Tests that gradients flow correctly through new components."""

    def test_sdpa_gradient_flow(self, n_features, hidden_dim, n_heads, device):
        """Gradients flow through SDPA attention during backprop."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, use_sdpa=True,
        ).to(device)
        model.train()
        x = torch.randn(2, 10, n_features, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist in SDPA projections
        for name, param in model.named_parameters():
            if "self_attn" in name and param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_multi_horizon_gradient_flow(self, n_features, hidden_dim, n_heads, device):
        """Gradients flow through auxiliary heads independently."""
        from quantlaxmi.models.ml.tft.momentum_tfm import MomentumTransformerModel
        model = MomentumTransformerModel(
            n_features=n_features, hidden_dim=hidden_dim,
            n_heads=n_heads, multi_horizon=True,
        ).to(device)
        model.train()
        x = torch.randn(2, 10, n_features, device=device)
        outputs = model(x)
        # Use only the T+2 loss
        loss = outputs[1].sum()
        loss.backward()
        # T+2 head should have gradients
        for name, param in model.aux_heads["T+2"].named_parameters():
            assert param.grad is not None, f"No gradient for aux T+2 {name}"
        # Main output_head should NOT have gradients (we only used T+2)
        for name, param in model.output_head.named_parameters():
            if param.grad is not None:
                assert (param.grad == 0).all(), \
                    f"Main head {name} should have zero grad when only T+2 loss used"
