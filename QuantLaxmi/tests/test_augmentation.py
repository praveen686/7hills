"""Tests for TFT data augmentation module."""

import numpy as np
import pytest

from quantlaxmi.models.ml.tft.production.augmentation import (
    AugmentationConfig,
    block_bootstrap,
    calibrated_noise,
    classify_features_into_groups,
    feature_group_dropout,
)

# Try torch imports
try:
    import torch
    from quantlaxmi.models.ml.tft.production.augmentation import AugmentedTFTDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_episode():
    """Synthetic episode: (seq_len=42, n_features=20), n_context=4."""
    rng = np.random.default_rng(123)
    seq_len, n_feat, n_ctx = 42, 20, 4
    x_tgt = rng.standard_normal((seq_len, n_feat)).astype(np.float32)
    x_ctx = rng.standard_normal((n_ctx, seq_len, n_feat)).astype(np.float32)
    return x_tgt, x_ctx


@pytest.fixture
def feature_names():
    """20 features across 4 groups."""
    return [
        "px_ret_1d", "px_vol_20d", "px_rsi_14", "px_macd", "px_bb_width",
        "opt_atm_iv", "opt_pcr", "opt_skew", "opt_iv_rank",
        "inst_fii_net", "inst_dii_net", "inst_client_net",
        "dff_d_hat", "dff_r_hat", "dff_energy", "dff_composite",
        "ca_nifty_bnf_spread", "ca_nifty_finnifty_corr",
        "ns_sent_mean", "ns_pos_ratio",
    ]


# ── Block Bootstrap Tests ───────────────────────────────────────────


class TestBlockBootstrap:

    def test_output_shape_matches_input(self, sample_episode, rng):
        x_tgt, x_ctx = sample_episode
        x_tgt_aug, x_ctx_aug = block_bootstrap(x_tgt, x_ctx, 7, rng)
        assert x_tgt_aug.shape == x_tgt.shape
        assert x_ctx_aug.shape == x_ctx.shape

    def test_values_come_from_original(self, sample_episode, rng):
        """Every row in augmented output must exist in original."""
        x_tgt, x_ctx = sample_episode
        x_tgt_aug, x_ctx_aug = block_bootstrap(x_tgt, x_ctx, 7, rng)

        for t in range(x_tgt_aug.shape[0]):
            # This row must match some row in original
            matches = np.any(
                np.all(np.isclose(x_tgt_aug[t], x_tgt, atol=1e-7), axis=1)
            )
            assert matches, f"Row {t} not found in original"

    def test_shared_time_index_across_tgt_and_ctx(self, rng):
        """Same time-index map must be applied to tgt and all ctx slices."""
        seq_len, n_feat, n_ctx = 42, 5, 3
        # Create data where each timestep has a unique marker
        x_tgt = np.arange(seq_len).reshape(-1, 1).repeat(n_feat, axis=1).astype(np.float32)
        x_ctx = np.zeros((n_ctx, seq_len, n_feat), dtype=np.float32)
        for c in range(n_ctx):
            x_ctx[c] = x_tgt * (c + 2)  # Different scale per context

        x_tgt_aug, x_ctx_aug = block_bootstrap(x_tgt, x_ctx, 7, rng)

        # For each timestep, the "day index" in tgt should match ctx
        tgt_day = x_tgt_aug[:, 0]  # marker column
        for c in range(n_ctx):
            ctx_day = x_ctx_aug[c, :, 0] / (c + 2)
            np.testing.assert_allclose(
                tgt_day, ctx_day, atol=1e-5,
                err_msg=f"Context {c} time indices don't match target",
            )

    def test_block_length_larger_than_seq_returns_copy(self, sample_episode, rng):
        x_tgt, x_ctx = sample_episode
        x_tgt_aug, x_ctx_aug = block_bootstrap(x_tgt, x_ctx, 100, rng)
        np.testing.assert_array_equal(x_tgt_aug, x_tgt)
        np.testing.assert_array_equal(x_ctx_aug, x_ctx)

    def test_different_seeds_different_results(self, sample_episode):
        x_tgt, x_ctx = sample_episode
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        aug1, _ = block_bootstrap(x_tgt, x_ctx, 7, rng1)
        aug2, _ = block_bootstrap(x_tgt, x_ctx, 7, rng2)
        assert not np.array_equal(aug1, aug2)

    def test_contiguous_blocks_preserved(self, rng):
        """Within each block, consecutive timesteps should be from consecutive original days."""
        seq_len, n_feat = 42, 3
        # Use day indices as feature values
        x_tgt = np.arange(seq_len).reshape(-1, 1).repeat(n_feat, axis=1).astype(np.float32)
        x_ctx = np.zeros((1, seq_len, n_feat), dtype=np.float32)

        block_len = 7
        x_tgt_aug, _ = block_bootstrap(x_tgt, x_ctx, block_len, rng)

        days = x_tgt_aug[:, 0].astype(int)
        # Check that within each block of 7, consecutive diffs are 1
        for block_start in range(0, seq_len - block_len, block_len):
            block = days[block_start : block_start + block_len]
            diffs = np.diff(block)
            np.testing.assert_array_equal(
                diffs, np.ones(block_len - 1),
                err_msg=f"Block at {block_start} is not contiguous: {block}",
            )


# ── Feature Group Dropout Tests ─────────────────────────────────────


class TestFeatureGroupDropout:

    def test_output_shape_preserved(self, sample_episode, rng, feature_names):
        x_tgt, _ = sample_episode
        groups = classify_features_into_groups(feature_names)
        result = feature_group_dropout(x_tgt, groups, 0.5, rng)
        assert result.shape == x_tgt.shape

    def test_protected_groups_never_dropped(self, rng, feature_names):
        """Price group (protected) should never be zeroed."""
        x = np.ones((10, len(feature_names)), dtype=np.float32)
        groups = classify_features_into_groups(feature_names)
        price_cols = groups.get("price", [])
        assert len(price_cols) > 0

        # Run 100 times with p_drop=1.0 — price should survive every time
        for _ in range(100):
            result = feature_group_dropout(x, groups, 1.0, rng, protected_groups={"price"})
            for col in price_cols:
                assert np.all(result[:, col] == 1.0), "Price column was zeroed!"

    def test_high_p_drops_most_groups(self, rng, feature_names):
        """With p_drop=1.0, non-protected groups should be zeroed."""
        x = np.ones((10, len(feature_names)), dtype=np.float32)
        groups = classify_features_into_groups(feature_names)
        result = feature_group_dropout(x, groups, 1.0, rng, protected_groups={"price"})

        # Non-price features should be zeroed
        non_price_cols = [
            i for i, name in enumerate(feature_names)
            if not name.startswith("px_")
        ]
        assert np.all(result[:, non_price_cols] == 0.0)

    def test_p_zero_changes_nothing(self, sample_episode, rng, feature_names):
        x_tgt, _ = sample_episode
        groups = classify_features_into_groups(feature_names)
        result = feature_group_dropout(x_tgt, groups, 0.0, rng)
        np.testing.assert_array_equal(result, x_tgt)


# ── Calibrated Noise Tests ──────────────────────────────────────────


class TestCalibratedNoise:

    def test_output_shape_preserved(self, sample_episode, rng):
        x_tgt, _ = sample_episode
        result = calibrated_noise(x_tgt, 0.02, rng)
        assert result.shape == x_tgt.shape

    def test_noise_magnitude(self, rng):
        """Noise should have approximately the requested std."""
        x = np.zeros((10000, 5), dtype=np.float32)
        result = calibrated_noise(x, 0.02, rng)
        # Result should be ~N(0, 0.02)
        empirical_std = np.std(result)
        assert 0.015 < empirical_std < 0.025, f"Noise std={empirical_std}, expected ~0.02"

    def test_zero_std_returns_original(self, sample_episode, rng):
        x_tgt, _ = sample_episode
        result = calibrated_noise(x_tgt, 0.0, rng)
        np.testing.assert_array_equal(result, x_tgt)

    def test_preserves_dtype(self, rng):
        x = np.ones((10, 5), dtype=np.float32)
        result = calibrated_noise(x, 0.01, rng)
        assert result.dtype == np.float32


# ── Feature Group Classification Tests ──────────────────────────────


class TestClassifyFeatures:

    def test_known_prefixes(self, feature_names):
        groups = classify_features_into_groups(feature_names)
        assert "price" in groups
        assert "options" in groups
        assert "divergence_flow" in groups
        assert "cross_asset" in groups
        assert "news_sentiment" in groups

    def test_all_features_classified(self, feature_names):
        groups = classify_features_into_groups(feature_names)
        total = sum(len(v) for v in groups.values())
        assert total == len(feature_names)

    def test_no_duplicate_indices(self, feature_names):
        groups = classify_features_into_groups(feature_names)
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert len(all_indices) == len(set(all_indices))

    def test_unknown_prefix_goes_to_other(self):
        names = ["unknown_feat1", "weird_thing"]
        groups = classify_features_into_groups(names)
        assert "_other" in groups
        assert len(groups["_other"]) == 2


# ── Augmented Dataset Tests ─────────────────────────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
class TestAugmentedDataset:

    def _make_dataset(self, N=100, seq_len=42, n_feat=20, n_ctx=4, factor=3):
        rng = np.random.default_rng(42)
        X_tgt = rng.standard_normal((N, seq_len, n_feat)).astype(np.float32)
        X_ctx = rng.standard_normal((N, n_ctx, seq_len, n_feat)).astype(np.float32)
        X_tid = rng.integers(0, 6, size=N).astype(np.int64)
        X_cid = rng.integers(0, 6, size=(N, n_ctx)).astype(np.int64)
        Y = rng.standard_normal(N).astype(np.float32)

        names = [f"px_f{i}" for i in range(5)] + [f"opt_f{i}" for i in range(5)] + \
                [f"inst_f{i}" for i in range(5)] + [f"dff_f{i}" for i in range(5)]

        cfg = AugmentationConfig(
            enabled=True,
            augmentation_factor=factor,
            noise_std=0.02,
            p_group_drop=0.10,
        )
        return AugmentedTFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y, cfg, names), Y

    def test_length_includes_augmented(self):
        ds, _ = self._make_dataset(N=100, factor=3)
        assert len(ds) == 400  # 100 real + 300 augmented

    def test_original_samples_unchanged(self):
        ds, Y = self._make_dataset(N=50, factor=2)
        for i in range(50):
            tgt, ctx, tid, cid, y = ds[i]
            assert isinstance(tgt, torch.Tensor)
            assert float(y) == pytest.approx(float(Y[i]), abs=1e-6)

    def test_augmented_samples_have_correct_target(self):
        """Y is NEVER modified by augmentation."""
        ds, Y = self._make_dataset(N=50, factor=2)
        for aug_idx in range(50, 150):
            _, _, _, _, y = ds[aug_idx]
            real_idx = aug_idx % 50
            assert float(y) == pytest.approx(float(Y[real_idx]), abs=1e-6)

    def test_augmented_samples_differ_from_original(self):
        ds, _ = self._make_dataset(N=50, factor=1)
        orig_tgt, _, _, _, _ = ds[0]
        aug_tgt, _, _, _, _ = ds[50]  # augmented copy of sample 0
        # Should NOT be identical (augmentation applied)
        assert not torch.allclose(orig_tgt, aug_tgt, atol=1e-6)

    def test_shapes_consistent(self):
        ds, _ = self._make_dataset(N=20, seq_len=42, n_feat=20, n_ctx=4)
        for i in [0, 10, 20, 60]:
            tgt, ctx, tid, cid, y = ds[i]
            assert tgt.shape == (42, 20)
            assert ctx.shape == (4, 42, 20)
            assert tid.shape == ()
            assert cid.shape == (4,)
            assert y.shape == ()

    def test_disabled_augmentation_returns_original_length(self):
        rng = np.random.default_rng(42)
        N, seq_len, n_feat, n_ctx = 50, 42, 10, 2
        cfg = AugmentationConfig(enabled=False)
        ds = AugmentedTFTDataset(
            rng.standard_normal((N, seq_len, n_feat)).astype(np.float32),
            rng.standard_normal((N, n_ctx, seq_len, n_feat)).astype(np.float32),
            np.zeros(N, dtype=np.int64),
            np.zeros((N, n_ctx), dtype=np.int64),
            np.zeros(N, dtype=np.float32),
            config=cfg,
        )
        assert len(ds) == N  # No augmentation

    def test_reseed_changes_augmented_output(self):
        ds, _ = self._make_dataset(N=20, factor=1)
        ds.reseed(0)
        tgt_e0, _, _, _, _ = ds[20]
        ds.reseed(1)
        tgt_e1, _, _, _, _ = ds[20]
        assert not torch.allclose(tgt_e0, tgt_e1, atol=1e-6)

    def test_dropout_mask_shared_between_tgt_and_ctx(self):
        """Feature group dropout must apply the SAME mask to tgt and all ctx slices."""
        rng = np.random.default_rng(42)
        N, seq_len, n_feat, n_ctx = 10, 42, 20, 4

        # All-ones data so we can see zeroing clearly
        X_tgt = np.ones((N, seq_len, n_feat), dtype=np.float32)
        X_ctx = np.ones((N, n_ctx, seq_len, n_feat), dtype=np.float32)
        X_tid = np.zeros(N, dtype=np.int64)
        X_cid = np.zeros((N, n_ctx), dtype=np.int64)
        Y = np.zeros(N, dtype=np.float32)

        names = [f"px_f{i}" for i in range(5)] + [f"opt_f{i}" for i in range(5)] + \
                [f"inst_f{i}" for i in range(5)] + [f"dff_f{i}" for i in range(5)]

        cfg = AugmentationConfig(
            enabled=True,
            augmentation_factor=1,
            block_bootstrap=False,  # isolate dropout effect
            calibrated_noise=False,
            p_group_drop=1.0,  # drop everything non-protected
        )
        ds = AugmentedTFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y, cfg, names)

        # Check augmented sample (idx=N is first augmented)
        tgt, ctx, _, _, _ = ds[N]
        tgt_np = tgt.numpy()
        ctx_np = ctx.numpy()

        for feat_idx in range(n_feat):
            tgt_zeroed = np.all(tgt_np[:, feat_idx] == 0.0)
            for c in range(n_ctx):
                ctx_zeroed = np.all(ctx_np[c, :, feat_idx] == 0.0)
                assert tgt_zeroed == ctx_zeroed, (
                    f"Feature {feat_idx}: tgt zeroed={tgt_zeroed}, ctx[{c}] zeroed={ctx_zeroed}"
                )

    def test_dataloader_integration(self):
        """Smoke test: DataLoader can iterate the augmented dataset."""
        ds, _ = self._make_dataset(N=32, factor=2)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
        batch = next(iter(loader))
        assert len(batch) == 5
        assert batch[0].shape[0] == 16  # batch dim
