"""Tests for Classic TFT (Lim et al. 2020) implementation.

Tests cover:
- Module shape correctness (GRN, attention, VSN, full model)
- Interpretable attention: shared V weights, causal masking
- VSN weight normalization (softmax -> sum to 1)
- Static covariate encoder: 4 distinct context vectors
- Quantile loss: symmetry at q=0.5, asymmetry at q!=0.5
- Data formatter: sliding windows, no look-ahead, scaler fitting
- End-to-end: loss decreases on synthetic data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    from classic_tft import ClassicTFTConfig

    return ClassicTFTConfig(
        d_hidden=32,
        n_heads=2,
        lstm_layers=1,
        stack_size=1,
        dropout=0.0,  # no dropout for deterministic tests
        n_static_cat=1,
        cat_cardinalities=[5],
        n_observed=4,
        n_known=3,
        encoder_steps=10,
        decoder_steps=3,
        quantiles=[0.1, 0.5, 0.9],
        output_size=1,
    )


@pytest.fixture
def small_model(small_config):
    from classic_tft import ClassicTFTModel

    model = ClassicTFTModel(small_config)
    model.eval()
    return model


@pytest.fixture
def sample_batch(small_config):
    """Create sample input tensors."""
    batch = 4
    cfg = small_config
    n_past = cfg.n_observed + cfg.n_known + cfg.output_size  # 4+3+1=8
    past = torch.randn(batch, cfg.encoder_steps, n_past)
    future = torch.randn(batch, cfg.decoder_steps, cfg.n_known)
    static_cat = torch.randint(0, cfg.cat_cardinalities[0], (batch, cfg.n_static_cat))
    targets = torch.randn(batch, cfg.decoder_steps, 1)
    return past, future, static_cat, targets


# ============================================================================
# Shape tests
# ============================================================================


class TestShapes:
    def test_forward_output_shape(self, small_model, sample_batch):
        """Full model forward produces correct output shape."""
        past, future, static_cat, _ = sample_batch
        with torch.no_grad():
            output, interp = small_model(past, future, static_cat)
        # output_size=1, n_quantiles=3 -> last dim = 1*3 = 3
        assert output.shape == (4, 3, 3)  # (batch, decoder_steps, output_size * n_quantiles)

    def test_gate_add_norm_shape(self):
        from classic_tft import GateAddNorm

        gate = GateAddNorm(32, 32, dropout=0.0)
        x = torch.randn(4, 10, 32)
        residual = torch.randn(4, 10, 32)
        out = gate(x, residual)
        assert out.shape == (4, 10, 32)

    def test_gate_add_norm_different_input_dim(self):
        """GateAddNorm with input_dim != hidden_dim should still work."""
        from classic_tft import GateAddNorm

        gate = GateAddNorm(64, 32, dropout=0.0)
        x = torch.randn(4, 10, 64)
        residual = torch.randn(4, 10, 32)
        out = gate(x, residual)
        assert out.shape == (4, 10, 32)

    def test_temporal_vsn_shape(self):
        from classic_tft import TemporalVSN

        vsn = TemporalVSN(n_variables=5, d_hidden=32, dropout=0.0)
        x = torch.randn(4, 10, 5)
        context = torch.randn(4, 32)
        output, weights = vsn(x, context)
        assert output.shape == (4, 10, 32)
        assert weights.shape == (4, 10, 5)

    def test_static_encoder_shape(self):
        from classic_tft import StaticCovariateEncoder

        enc = StaticCovariateEncoder(d_hidden=32, dropout=0.0)
        static_rep = torch.randn(4, 32)
        c_s, c_e, c_h, c_c = enc(static_rep)
        for c in [c_s, c_e, c_h, c_c]:
            assert c.shape == (4, 32)


# ============================================================================
# Attention tests
# ============================================================================


class TestInterpretableAttention:
    def test_shared_v_weights(self):
        """Verify V projection is a single shared layer, not per-head."""
        from classic_tft import InterpretableMultiHeadAttention

        attn = InterpretableMultiHeadAttention(n_heads=4, d_model=32, dropout=0.0)
        # Should have exactly 1 v_proj (shared), 4 q_projs, 4 k_projs
        assert isinstance(attn.v_proj, nn.Linear)
        assert len(attn.q_projs) == 4
        assert len(attn.k_projs) == 4

    def test_attention_output_shape(self):
        from classic_tft import InterpretableMultiHeadAttention

        attn = InterpretableMultiHeadAttention(n_heads=4, d_model=32, dropout=0.0)
        q = torch.randn(4, 10, 32)
        k = torch.randn(4, 10, 32)
        v = torch.randn(4, 10, 32)
        output, weights = attn(q, k, v)
        assert output.shape == (4, 10, 32)
        assert weights.shape == (4, 4, 10, 10)  # (batch, heads, T_q, T_k)

    def test_causal_mask_blocks_future(self):
        """With causal mask, position i cannot attend to positions > i."""
        from classic_tft import InterpretableMultiHeadAttention

        attn = InterpretableMultiHeadAttention(n_heads=1, d_model=16, dropout=0.0)
        T = 5
        x = torch.randn(1, T, 16)
        mask = torch.tril(torch.ones(T, T))  # causal mask
        _, weights = attn(x, x, x, mask=mask)
        # weights: (1, 1, T, T)
        w = weights[0, 0]  # (T, T)
        # Upper triangle (future positions) should be ~0
        for i in range(T):
            for j in range(i + 1, T):
                assert w[i, j].item() < 1e-6, (
                    f"Position {i} attends to future position {j}: {w[i, j].item()}"
                )

    def test_attention_weights_sum_to_one(self):
        from classic_tft import InterpretableMultiHeadAttention

        attn = InterpretableMultiHeadAttention(n_heads=2, d_model=16, dropout=0.0)
        x = torch.randn(2, 8, 16)
        _, weights = attn(x, x, x)
        # Each row should sum to 1 (softmax normalization)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_v_proj_dimension(self):
        """V projection should map d_model -> d_k (d_model // n_heads)."""
        from classic_tft import InterpretableMultiHeadAttention

        attn = InterpretableMultiHeadAttention(n_heads=4, d_model=32, dropout=0.0)
        assert attn.v_proj.in_features == 32
        assert attn.v_proj.out_features == 8  # 32 // 4 = 8


# ============================================================================
# VSN tests
# ============================================================================


class TestVSN:
    def test_vsn_weights_sum_to_one(self):
        """VSN selection weights must sum to 1 (softmax)."""
        from classic_tft import TemporalVSN

        vsn = TemporalVSN(n_variables=6, d_hidden=32, dropout=0.0)
        x = torch.randn(4, 10, 6)
        _, weights = vsn(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_vsn_with_context(self):
        """VSN should work with and without static context."""
        from classic_tft import TemporalVSN

        vsn = TemporalVSN(n_variables=4, d_hidden=16, dropout=0.0)
        x = torch.randn(2, 5, 4)

        out_no_ctx, w1 = vsn(x, context=None)
        ctx = torch.randn(2, 16)
        out_with_ctx, w2 = vsn(x, context=ctx)

        # Both should produce valid shapes
        assert out_no_ctx.shape == (2, 5, 16)
        assert out_with_ctx.shape == (2, 5, 16)
        # With context should differ from without (context modulates selection GRN)
        assert not torch.allclose(out_no_ctx, out_with_ctx)

    def test_vsn_single_variable(self):
        """VSN with 1 variable should still work and weight should be 1.0."""
        from classic_tft import TemporalVSN

        vsn = TemporalVSN(n_variables=1, d_hidden=16, dropout=0.0)
        x = torch.randn(2, 5, 1)
        _, weights = vsn(x)
        # With only 1 variable, softmax output is always 1.0
        assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


# ============================================================================
# Static encoder tests
# ============================================================================


class TestStaticEncoder:
    def test_four_distinct_contexts(self):
        """4 context vectors should be distinct (different GRNs)."""
        from classic_tft import StaticCovariateEncoder

        enc = StaticCovariateEncoder(d_hidden=32, dropout=0.0)
        static = torch.randn(4, 32)
        c_s, c_e, c_h, c_c = enc(static)
        # All 4 should be different (they use different GRNs with different weights)
        assert not torch.allclose(c_s, c_e)
        assert not torch.allclose(c_s, c_h)
        assert not torch.allclose(c_s, c_c)
        assert not torch.allclose(c_e, c_h)

    def test_batch_independence(self):
        """Each batch element should produce independent contexts."""
        from classic_tft import StaticCovariateEncoder

        enc = StaticCovariateEncoder(d_hidden=16, dropout=0.0)
        static = torch.randn(3, 16)
        c_s, _, _, _ = enc(static)
        # Different inputs should produce different outputs
        assert not torch.allclose(c_s[0], c_s[1])


# ============================================================================
# Loss tests
# ============================================================================


class TestQuantileLoss:
    def test_p50_is_half_mae(self):
        """At q=0.5, quantile loss = 0.5 * MAE (pinball at 0.5 = 0.5*|e|)."""
        from classic_tft import QuantileLoss

        loss_fn = QuantileLoss([0.5])
        preds = torch.tensor([[[1.0], [2.0], [3.0]]])
        targets = torch.tensor([[[1.5], [1.5], [1.5]]])
        loss = loss_fn(preds, targets)
        mae = torch.abs(preds - targets).mean()
        expected = mae * 0.5  # pinball at q=0.5 = 0.5 * |error|
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_asymmetric_penalty(self):
        """p90 penalizes under-prediction more; p10 penalizes over-prediction more."""
        from classic_tft import QuantileLoss

        # p90: penalizes under-prediction more
        loss_p90 = QuantileLoss([0.9])
        preds = torch.tensor([[[0.0]]])
        over = torch.tensor([[[-1.0]]])  # target < pred (over-predicted)
        under = torch.tensor([[[1.0]]])  # target > pred (under-predicted)
        loss_over = loss_p90(preds, over)
        loss_under = loss_p90(preds, under)
        assert loss_under > loss_over  # under-prediction costs more at p90

    def test_zero_loss_when_perfect(self):
        """Loss should be 0 when predictions exactly match targets."""
        from classic_tft import QuantileLoss

        loss_fn = QuantileLoss([0.1, 0.5, 0.9])
        targets = torch.tensor([[[2.0], [3.0]]])
        # All quantile predictions match target exactly
        preds = torch.tensor([[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]])
        loss = loss_fn(preds, targets)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_normalized_quantile_loss(self):
        """NQL = 2 * sum(pinball) / sum(|target|)."""
        from classic_tft import normalized_quantile_loss

        preds = torch.tensor([[[1.0, 2.0, 3.0]]])  # 3 quantiles
        targets = torch.tensor([[[2.0]]])
        nql = normalized_quantile_loss(preds, targets, 0.5, 1)  # p50 is index 1
        assert isinstance(nql, float)
        assert nql >= 0

    def test_normalized_quantile_loss_zero_target(self):
        """NQL should return 0.0 when targets are all zero."""
        from classic_tft import normalized_quantile_loss

        preds = torch.tensor([[[1.0, 2.0, 3.0]]])
        targets = torch.tensor([[[0.0]]])
        nql = normalized_quantile_loss(preds, targets, 0.5, 1)
        assert nql == 0.0

    def test_multi_quantile_loss_shape(self):
        """QuantileLoss with multiple quantiles should return a scalar."""
        from classic_tft import QuantileLoss

        loss_fn = QuantileLoss([0.1, 0.5, 0.9])
        preds = torch.randn(4, 5, 3)  # (batch, T, 3 quantiles)
        targets = torch.randn(4, 5, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0  # scalar


# ============================================================================
# Data formatter tests
# ============================================================================


class TestDataFormatter:
    def _make_synthetic_df(self, n: int = 30) -> pd.DataFrame:
        """Create a minimal synthetic DataFrame for formatter tests."""
        rng = np.random.RandomState(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {
                "Symbol": "TEST",
                "rv5_ss": rng.exponential(0.001, n),
                "open_to_close": rng.randn(n) * 0.01,
                "bv": rng.exponential(0.001, n),
                "medrv": rng.exponential(0.001, n),
                "rk_parzen": rng.exponential(0.001, n),
                "rv10": rng.exponential(0.001, n),
            },
            index=dates,
        )

    def test_for_volatility_factory(self):
        from data_formatter import TFTDataFormatter

        fmt = TFTDataFormatter.for_volatility(encoder_steps=20, decoder_steps=3)
        assert fmt.config.encoder_steps == 20
        assert fmt.config.decoder_steps == 3
        assert fmt.n_past_variables > 0
        assert fmt.n_future_variables > 0

    def test_sliding_window_shapes(self):
        """Windows should have correct encoder/decoder shapes."""
        from data_formatter import TFTDataFormatter

        fmt = TFTDataFormatter.for_volatility(encoder_steps=5, decoder_steps=2)
        df = self._make_synthetic_df(20)

        prepared = fmt.prepare_oxford_man(df)
        fmt.fit(prepared)
        windows = fmt.transform(prepared)

        assert windows["past_inputs"].shape[1] == 5  # encoder_steps
        assert windows["future_inputs"].shape[1] == 2  # decoder_steps
        assert windows["targets"].shape[1] == 2  # decoder_steps
        assert windows["targets"].shape[2] == 1  # output_size

    def test_sliding_window_count(self):
        """Number of windows = n - (encoder + decoder) + 1."""
        from data_formatter import TFTDataFormatter

        enc, dec = 5, 2
        fmt = TFTDataFormatter.for_volatility(encoder_steps=enc, decoder_steps=dec)
        df = self._make_synthetic_df(20)

        prepared = fmt.prepare_oxford_man(df)
        n = len(prepared)
        fmt.fit(prepared)
        windows = fmt.transform(prepared)

        expected_count = n - (enc + dec) + 1
        assert windows["past_inputs"].shape[0] == expected_count

    def test_scaler_fit_on_train_only(self):
        """Scalers should use training statistics, not test data."""
        from data_formatter import TFTDataFormatter

        fmt = TFTDataFormatter.for_volatility(encoder_steps=5, decoder_steps=2)

        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Symbol": "TEST",
                # Training portion has small rv5_ss, test has large
                "rv5_ss": np.concatenate([np.ones(15) * 0.001, np.ones(15) * 0.1]),
                "open_to_close": np.random.randn(n) * 0.01,
                "bv": np.random.exponential(0.001, n),
                "medrv": np.random.exponential(0.001, n),
                "rk_parzen": np.random.exponential(0.001, n),
                "rv10": np.random.exponential(0.001, n),
            },
            index=dates,
        )

        prepared = fmt.prepare_oxford_man(df)
        train = prepared.iloc[:15]

        fmt.fit(train)
        mean, std = fmt.target_scaler
        # log(0.001) ~ -6.9, so the mean should be very negative
        assert mean < -5

    def test_transform_before_fit_raises(self):
        """transform() without fit() should raise RuntimeError."""
        from data_formatter import TFTDataFormatter

        fmt = TFTDataFormatter.for_volatility(encoder_steps=5, decoder_steps=2)
        df = self._make_synthetic_df(20)
        prepared = fmt.prepare_oxford_man(df)

        with pytest.raises(RuntimeError, match="fit"):
            fmt.transform(prepared)

    def test_get_tft_config(self):
        """get_tft_config() should return correct input dimensions."""
        from data_formatter import TFTDataFormatter

        fmt = TFTDataFormatter.for_volatility(encoder_steps=10, decoder_steps=3)
        df = self._make_synthetic_df(30)
        prepared = fmt.prepare_oxford_man(df)
        fmt.fit(prepared)

        cfg = fmt.get_tft_config()
        assert cfg["n_observed"] == 6  # open_to_close, log_bv, log_medrv, log_rk_parzen, log_rv10, bv_ratio
        assert cfg["n_known"] == 6  # dow_0..dow_4, days_from_start
        assert cfg["n_static_cat"] == 1  # Symbol
        assert cfg["encoder_steps"] == 10
        assert cfg["decoder_steps"] == 3
        assert cfg["output_size"] == 1  # log_vol


# ============================================================================
# Training test
# ============================================================================


class TestTraining:
    def test_loss_decreases(self, small_config):
        """Training on synthetic data should decrease loss."""
        from classic_tft import ClassicTFTModel, QuantileLoss

        cfg = small_config
        model = ClassicTFTModel(cfg)
        criterion = QuantileLoss(cfg.quantiles)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        batch = 8
        n_past = cfg.n_observed + cfg.n_known + cfg.output_size
        torch.manual_seed(42)
        past = torch.randn(batch, cfg.encoder_steps, n_past)
        future = torch.randn(batch, cfg.decoder_steps, cfg.n_known)
        static_cat = torch.randint(0, cfg.cat_cardinalities[0], (batch, cfg.n_static_cat))
        targets = torch.randn(batch, cfg.decoder_steps, 1)

        losses = []
        for _ in range(30):
            model.train()
            optimizer.zero_grad()
            preds, _ = model(past, future, static_cat)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease from first to last
        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ============================================================================
# Interpretability test
# ============================================================================


class TestInterpretability:
    def test_interpretability_dict_keys(self, small_model, sample_batch):
        """Forward pass should return interpretability dict with expected keys."""
        past, future, static_cat, _ = sample_batch
        with torch.no_grad():
            _, interp = small_model(past, future, static_cat)

        assert "attention_weights" in interp
        assert "past_vsn_weights" in interp
        assert "future_vsn_weights" in interp
        assert "static_context" in interp

    def test_attention_weights_structure(self, small_model, sample_batch):
        """Attention weights should be a list of tensors (one per stack layer)."""
        past, future, static_cat, _ = sample_batch
        with torch.no_grad():
            _, interp = small_model(past, future, static_cat)

        attn_w = interp["attention_weights"]
        assert isinstance(attn_w, list)
        assert len(attn_w) == small_model.cfg.stack_size  # 1 stack layer
        total_time = small_model.cfg.encoder_steps + small_model.cfg.decoder_steps
        # Each attn weight: (batch, n_heads, total_time, total_time)
        assert attn_w[0].shape == (4, 2, total_time, total_time)

    def test_vsn_weights_shapes(self, small_model, sample_batch):
        """Past/future VSN weights should match input variable counts."""
        past, future, static_cat, _ = sample_batch
        cfg = small_model.cfg
        with torch.no_grad():
            _, interp = small_model(past, future, static_cat)

        past_w = interp["past_vsn_weights"]
        assert past_w.shape == (4, cfg.encoder_steps, small_model._n_past_vars)

        future_w = interp["future_vsn_weights"]
        assert future_w.shape == (4, cfg.decoder_steps, small_model._n_future_vars)

    def test_static_context_four_vectors(self, small_model, sample_batch):
        """Static context should contain 4 named context vectors."""
        past, future, static_cat, _ = sample_batch
        with torch.no_grad():
            _, interp = small_model(past, future, static_cat)

        sc = interp["static_context"]
        for key in ["c_s", "c_e", "c_h", "c_c"]:
            assert key in sc
            assert sc[key].shape == (4, small_model.cfg.d_hidden)


# ============================================================================
# Model configuration edge cases
# ============================================================================


class TestModelEdgeCases:
    def test_no_static_inputs(self):
        """Model should work with no static inputs (uses learnable default)."""
        from classic_tft import ClassicTFTConfig, ClassicTFTModel

        cfg = ClassicTFTConfig(
            d_hidden=16,
            n_heads=2,
            lstm_layers=1,
            stack_size=1,
            dropout=0.0,
            n_static_cat=0,
            cat_cardinalities=[],
            n_observed=3,
            n_known=2,
            encoder_steps=5,
            decoder_steps=2,
            quantiles=[0.1, 0.5, 0.9],
            output_size=1,
        )
        model = ClassicTFTModel(cfg)
        model.eval()

        batch = 2
        n_past = cfg.n_observed + cfg.n_known + cfg.output_size
        past = torch.randn(batch, cfg.encoder_steps, n_past)
        future = torch.randn(batch, cfg.decoder_steps, cfg.n_known)

        with torch.no_grad():
            output, interp = model(past, future)

        assert output.shape == (batch, cfg.decoder_steps, 3)

    def test_multiple_stack_layers(self):
        """Model with stack_size > 1 should produce multiple attention weight tensors."""
        from classic_tft import ClassicTFTConfig, ClassicTFTModel

        cfg = ClassicTFTConfig(
            d_hidden=16,
            n_heads=2,
            lstm_layers=1,
            stack_size=3,
            dropout=0.0,
            n_static_cat=1,
            cat_cardinalities=[3],
            n_observed=2,
            n_known=2,
            encoder_steps=5,
            decoder_steps=2,
            quantiles=[0.5],
            output_size=1,
        )
        model = ClassicTFTModel(cfg)
        model.eval()

        batch = 2
        n_past = cfg.n_observed + cfg.n_known + cfg.output_size
        past = torch.randn(batch, cfg.encoder_steps, n_past)
        future = torch.randn(batch, cfg.decoder_steps, cfg.n_known)
        static_cat = torch.randint(0, 3, (batch, 1))

        with torch.no_grad():
            output, interp = model(past, future, static_cat)

        assert output.shape == (batch, 2, 1)  # 1 quantile
        assert len(interp["attention_weights"]) == 3  # 3 stack layers
