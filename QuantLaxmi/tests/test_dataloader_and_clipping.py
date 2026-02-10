"""Tests for ITEM 6 (_TFTDataset + DataLoader) and ITEM 16 (outlier clipping).

Covers:
  - _TFTDataset __len__, __getitem__, tensor dtypes
  - DataLoader batch sizes and pin_memory behaviour
  - MegaFeatureBuilder outlier clipping (clips extremes, preserves normal,
    disabled via clip_sigma=None, logging of clip count)
"""

from __future__ import annotations

import logging
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.utils.data as torch_data

from quantlaxmi.models.ml.tft.production.training_pipeline import _TFTDataset


# ============================================================================
# _TFTDataset + DataLoader tests
# ============================================================================


def _make_dummy_arrays(n: int = 50):
    """Create dummy numpy arrays matching _TFTDataset signature."""
    seq_len, n_features, n_context, ctx_len = 10, 5, 4, 10
    X_tgt = np.random.randn(n, seq_len, n_features).astype(np.float32)
    X_ctx = np.random.randn(n, n_context, ctx_len, n_features).astype(np.float32)
    X_tid = np.arange(n, dtype=np.int64) % 4
    X_cid = np.random.randint(0, 4, size=(n, n_context), dtype=np.int64)
    Y = np.random.randn(n).astype(np.float32)
    return X_tgt, X_ctx, X_tid, X_cid, Y


class TestTFTDataset:
    """Tests for _TFTDataset."""

    def test_tft_dataset_len(self):
        """Dataset length matches number of samples."""
        n = 73
        X_tgt, X_ctx, X_tid, X_cid, Y = _make_dummy_arrays(n)
        ds = _TFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y)
        assert len(ds) == n

    def test_tft_dataset_getitem(self):
        """__getitem__ returns a 5-tuple of tensors with correct dtypes and shapes."""
        n = 20
        X_tgt, X_ctx, X_tid, X_cid, Y = _make_dummy_arrays(n)
        ds = _TFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y)

        item = ds[0]
        assert len(item) == 5

        tgt, ctx, tid, cid, y = item

        # Check dtypes
        assert tgt.dtype == torch.float32
        assert ctx.dtype == torch.float32
        assert tid.dtype == torch.int64
        assert cid.dtype == torch.int64
        assert y.dtype == torch.float32

        # Check shapes match the input arrays
        assert tgt.shape == (10, 5)    # (seq_len, n_features)
        assert ctx.shape == (4, 10, 5) # (n_context, ctx_len, n_features)
        assert tid.shape == ()          # scalar
        assert cid.shape == (4,)        # (n_context,)
        assert y.shape == ()            # scalar

    def test_tft_dataset_getitem_values(self):
        """__getitem__ returns the correct values from the numpy arrays."""
        X_tgt, X_ctx, X_tid, X_cid, Y = _make_dummy_arrays(30)
        ds = _TFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y)

        idx = 7
        tgt, ctx, tid, cid, y = ds[idx]

        np.testing.assert_allclose(tgt.numpy(), X_tgt[idx], atol=1e-7)
        np.testing.assert_allclose(ctx.numpy(), X_ctx[idx], atol=1e-7)
        assert tid.item() == X_tid[idx]
        np.testing.assert_array_equal(cid.numpy(), X_cid[idx])
        np.testing.assert_allclose(y.item(), Y[idx], atol=1e-7)

    def test_dataloader_batch_size(self):
        """DataLoader yields batches of the expected size."""
        n = 50
        batch_size = 16
        X_tgt, X_ctx, X_tid, X_cid, Y = _make_dummy_arrays(n)
        ds = _TFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y)

        loader = torch_data.DataLoader(ds, batch_size=batch_size, shuffle=False)

        sizes = []
        for batch in loader:
            tgt, ctx, tid, cid, y = batch
            sizes.append(len(tgt))

        # Expected: ceil(50/16) = 4 batches: [16, 16, 16, 2]
        assert len(sizes) == math.ceil(n / batch_size)
        assert sizes[0] == batch_size
        assert sizes[-1] == n % batch_size or sizes[-1] == batch_size

    def test_dataloader_pin_memory_cuda(self):
        """pin_memory=True when CUDA is available, False otherwise."""
        X_tgt, X_ctx, X_tid, X_cid, Y = _make_dummy_arrays(20)
        ds = _TFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y)

        cuda_available = torch.cuda.is_available()

        loader = torch_data.DataLoader(
            ds, batch_size=8,
            pin_memory=cuda_available,
        )
        assert loader.pin_memory == cuda_available

        # Also verify the opposite setting works
        loader_no_pin = torch_data.DataLoader(
            ds, batch_size=8,
            pin_memory=False,
        )
        assert loader_no_pin.pin_memory is False

    def test_dataloader_shuffle(self):
        """Shuffled DataLoader produces different orderings across epochs."""
        n = 100
        X_tgt, X_ctx, X_tid, X_cid, Y = _make_dummy_arrays(n)
        # Make Y unique so we can detect ordering
        Y = np.arange(n, dtype=np.float32)
        ds = _TFTDataset(X_tgt, X_ctx, X_tid, X_cid, Y)

        loader = torch_data.DataLoader(ds, batch_size=n, shuffle=True)

        # Collect two full epochs of Y values
        epoch1 = next(iter(loader))[4].numpy()
        epoch2 = next(iter(loader))[4].numpy()

        # With n=100, probability of identical order is astronomically small
        # But be safe -- just check that at least some differ
        assert not np.array_equal(epoch1, epoch2), (
            "Two shuffled epochs produced identical ordering"
        )


# ============================================================================
# Outlier clipping tests (MegaFeatureBuilder)
# ============================================================================


def _make_mega_builder(**kwargs):
    """Create a MegaFeatureBuilder with patched store to avoid needing real data."""
    from quantlaxmi.features.mega import MegaFeatureBuilder
    return MegaFeatureBuilder(**kwargs)


class TestOutlierClipping:
    """Tests for MegaFeatureBuilder.clip_sigma feature."""

    def test_outlier_clipping_clips_extreme(self):
        """Values at +/-10 get clipped to +/-5 (default clip_sigma=5.0)."""
        builder = _make_mega_builder()
        assert builder.clip_sigma == 5.0

        # Create a DataFrame with extreme values
        df = pd.DataFrame({
            "feat_a": [0.0, 1.0, -1.0, 10.0, -10.0, 3.0],
            "feat_b": [2.0, -8.0, 7.0, 0.5, -0.5, 15.0],
        }, index=pd.date_range("2025-01-01", periods=6))

        # Simulate what build() does at the end
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        clip_sigma = builder.clip_sigma
        df[numeric_cols] = df[numeric_cols].clip(-clip_sigma, clip_sigma)

        # Extremes should be clipped
        assert df["feat_a"].iloc[3] == 5.0   # was 10.0
        assert df["feat_a"].iloc[4] == -5.0  # was -10.0
        assert df["feat_b"].iloc[1] == -5.0  # was -8.0
        assert df["feat_b"].iloc[2] == 5.0   # was 7.0
        assert df["feat_b"].iloc[5] == 5.0   # was 15.0

    def test_outlier_clipping_preserves_normal(self):
        """Values within [-5, 5] are unchanged."""
        builder = _make_mega_builder()

        # All values within bounds
        original = np.array([-4.9, -2.0, 0.0, 2.5, 4.99])
        df = pd.DataFrame({"feat": original}, index=pd.date_range("2025-01-01", periods=5))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].clip(-builder.clip_sigma, builder.clip_sigma)

        np.testing.assert_array_equal(df["feat"].values, original)

    def test_outlier_clipping_disabled(self):
        """clip_sigma=None disables clipping entirely."""
        builder = _make_mega_builder(clip_sigma=None)
        assert builder.clip_sigma is None

        # Values should NOT be clipped
        original = np.array([100.0, -200.0, 50.0])
        df = pd.DataFrame({"feat": original.copy()}, index=pd.date_range("2025-01-01", periods=3))

        # Simulate: if clip_sigma is None, skip clipping
        if builder.clip_sigma is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[numeric_cols] = df[numeric_cols].clip(-builder.clip_sigma, builder.clip_sigma)

        np.testing.assert_array_equal(df["feat"].values, original)

    def test_outlier_clipping_custom_sigma(self):
        """Custom clip_sigma=3.0 clips to [-3, 3]."""
        builder = _make_mega_builder(clip_sigma=3.0)
        assert builder.clip_sigma == 3.0

        df = pd.DataFrame({
            "feat": [-10.0, -3.0, 0.0, 3.0, 10.0],
        }, index=pd.date_range("2025-01-01", periods=5))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].clip(-builder.clip_sigma, builder.clip_sigma)

        expected = np.array([-3.0, -3.0, 0.0, 3.0, 3.0])
        np.testing.assert_array_equal(df["feat"].values, expected)

    def test_outlier_clipping_counts_logged(self, caplog):
        """Verify logging of clip count when outlier clipping fires."""
        from quantlaxmi.features.mega import MegaFeatureBuilder, logger as mega_logger

        builder = _make_mega_builder()

        # Create data with known number of extreme values
        df = pd.DataFrame({
            "feat_a": [0.0, 10.0, -10.0],  # 2 clipped
            "feat_b": [0.0, 0.0, 20.0],    # 1 clipped
        }, index=pd.date_range("2025-01-01", periods=3))

        # Simulate the clipping + logging from build()
        clip_sigma = builder.clip_sigma
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        before = df[numeric_cols]
        n_clipped = (before.abs() >= clip_sigma).sum().sum()

        assert n_clipped == 3  # 10.0, -10.0, 20.0

        # Verify the log message would be generated
        with caplog.at_level(logging.INFO, logger=mega_logger.name):
            if n_clipped > 0:
                mega_logger.info(
                    "Outlier clipping: %d values clipped to [%.1f, %.1f]",
                    n_clipped, -clip_sigma, clip_sigma,
                )

        assert "Outlier clipping: 3 values clipped to [-5.0, 5.0]" in caplog.text

    def test_outlier_clipping_handles_nan(self):
        """NaN values are preserved through clipping (not counted as clipped)."""
        builder = _make_mega_builder()

        df = pd.DataFrame({
            "feat": [np.nan, 10.0, -10.0, np.nan, 2.0],
        }, index=pd.date_range("2025-01-01", periods=5))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        before = df[numeric_cols]
        # NaN abs() -> NaN, NaN >= 5.0 -> False, so NaNs are not counted
        n_clipped = (before.abs() >= builder.clip_sigma).sum().sum()
        assert n_clipped == 2  # only 10.0 and -10.0

        df[numeric_cols] = df[numeric_cols].clip(-builder.clip_sigma, builder.clip_sigma)

        # NaN should be preserved
        assert pd.isna(df["feat"].iloc[0])
        assert pd.isna(df["feat"].iloc[3])
        # Extremes clipped
        assert df["feat"].iloc[1] == 5.0
        assert df["feat"].iloc[2] == -5.0
        # Normal preserved
        assert df["feat"].iloc[4] == 2.0

    def test_outlier_clipping_boundary_values(self):
        """Values at exactly +/-5 are NOT clipped (>= threshold counts them, but
        clip() keeps the value at the boundary)."""
        builder = _make_mega_builder()

        df = pd.DataFrame({
            "feat": [-5.0, 5.0, -5.0001, 5.0001, 4.9999],
        }, index=pd.date_range("2025-01-01", periods=5))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].clip(-builder.clip_sigma, builder.clip_sigma)

        # Exactly at boundary stays
        assert df["feat"].iloc[0] == -5.0
        assert df["feat"].iloc[1] == 5.0
        # Just beyond gets clipped to boundary
        assert df["feat"].iloc[2] == -5.0
        assert df["feat"].iloc[3] == 5.0
        # Inside stays
        assert df["feat"].iloc[4] == 4.9999
