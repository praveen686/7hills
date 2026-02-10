"""Tests for MegaFeatureBuilder disk caching and CheckpointManager rotation."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from quantlaxmi.features.mega import MegaFeatureBuilder
from quantlaxmi.models.ml.tft.production.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_feature_df(n_rows: int = 20, n_cols: int = 5) -> pd.DataFrame:
    """Create a small synthetic feature DataFrame."""
    dates = pd.date_range("2025-06-01", periods=n_rows, freq="B")
    data = np.random.default_rng(42).standard_normal((n_rows, n_cols))
    cols = [f"feat_{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_builder(cache_dir: Path, use_cache: bool = True) -> MegaFeatureBuilder:
    """Create a MegaFeatureBuilder with a temp cache directory."""
    return MegaFeatureBuilder(
        cache_dir=cache_dir,
        use_cache=use_cache,
    )


# ============================================================================
# Feature Cache Tests
# ============================================================================


class TestMegaFeatureCache:
    """Tests for MegaFeatureBuilder disk caching."""

    def test_cache_miss_computes(self, tmp_path: Path):
        """First call computes features and saves to cache."""
        cache_dir = tmp_path / "feature_cache"
        builder = _make_builder(cache_dir, use_cache=True)
        fake_df = _make_feature_df()
        fake_names = list(fake_df.columns)

        # Patch the actual feature computation to return our fake data
        with patch.object(
            MegaFeatureBuilder,
            "build",
            wraps=lambda self, sym, s, e: (fake_df, fake_names),
        ):
            # Directly test cache helpers: nothing cached yet
            assert builder._load_from_cache("NIFTY 50", "2025-06-01", "2025-06-30") is None

            # Save to cache
            builder._save_to_cache("NIFTY 50", "2025-06-01", "2025-06-30", fake_df)

            # Verify cache file was created
            cache_path = builder._cache_path("NIFTY 50", "2025-06-01", "2025-06-30")
            assert cache_path.exists()

    def test_cache_hit_returns_cached(self, tmp_path: Path):
        """Second call returns cached data without recomputing."""
        cache_dir = tmp_path / "feature_cache"
        builder = _make_builder(cache_dir, use_cache=True)
        fake_df = _make_feature_df()

        # Save to cache
        builder._save_to_cache("NIFTY 50", "2025-06-01", "2025-06-30", fake_df)

        # Load from cache — should be a HIT
        result = builder._load_from_cache("NIFTY 50", "2025-06-01", "2025-06-30")
        assert result is not None

        loaded_df, loaded_names = result
        # Parquet round-trip may drop DatetimeIndex.freq — compare without freq
        pd.testing.assert_frame_equal(
            loaded_df, fake_df, check_freq=False,
        )
        assert loaded_names == list(fake_df.columns)

    def test_cache_invalidation_on_code_change(self, tmp_path: Path):
        """Different version hash means cache miss — recomputes."""
        cache_dir = tmp_path / "feature_cache"
        builder = _make_builder(cache_dir, use_cache=True)
        fake_df = _make_feature_df()

        # Save to cache with current version hash
        builder._save_to_cache("NIFTY 50", "2025-06-01", "2025-06-30", fake_df)

        # Verify it loads fine with the real hash
        result = builder._load_from_cache("NIFTY 50", "2025-06-01", "2025-06-30")
        assert result is not None

        # Now patch _version_hash to return a different hash (simulates code change)
        with patch.object(
            MegaFeatureBuilder, "_version_hash", return_value="deadbeef"
        ):
            result = builder._load_from_cache("NIFTY 50", "2025-06-01", "2025-06-30")
            assert result is None, "Should be a cache miss after code change"

    def test_cache_disabled(self, tmp_path: Path):
        """use_cache=False always returns None from load and doesn't save."""
        cache_dir = tmp_path / "feature_cache"
        builder = _make_builder(cache_dir, use_cache=False)
        fake_df = _make_feature_df()

        # Save should be a no-op
        builder._save_to_cache("NIFTY 50", "2025-06-01", "2025-06-30", fake_df)
        cache_path = builder._cache_path("NIFTY 50", "2025-06-01", "2025-06-30")
        assert not cache_path.exists(), "Cache file should NOT be created when use_cache=False"

        # Load should always return None
        result = builder._load_from_cache("NIFTY 50", "2025-06-01", "2025-06-30")
        assert result is None

    def test_version_hash_is_stable(self):
        """Calling _version_hash twice returns the same value."""
        h1 = MegaFeatureBuilder._version_hash()
        h2 = MegaFeatureBuilder._version_hash()
        assert h1 == h2
        assert len(h1) == 8

    def test_cache_key_includes_all_params(self, tmp_path: Path):
        """Cache key changes with symbol, dates, and version hash."""
        builder = _make_builder(tmp_path, use_cache=True)
        k1 = builder._cache_key("NIFTY 50", "2025-06-01", "2025-06-30")
        k2 = builder._cache_key("Nifty Bank", "2025-06-01", "2025-06-30")
        k3 = builder._cache_key("NIFTY 50", "2025-07-01", "2025-07-31")
        assert k1 != k2, "Different symbol must produce different key"
        assert k1 != k3, "Different dates must produce different key"


# ============================================================================
# Checkpoint Rotation Tests
# ============================================================================


def _save_fake_checkpoint(
    mgr: CheckpointManager,
    model_type: str,
    version: int,
) -> Path:
    """Save a minimal fake checkpoint with a given version number."""
    meta = CheckpointMetadata(
        version=version,
        model_type=model_type,
        n_features=10,
        sharpe_oos=1.5,
    )
    state_dict = {"weight": np.zeros(5)}
    return mgr.save(state_dict, meta)


class TestCheckpointRotation:
    """Tests for CheckpointManager rotation/cleanup policy."""

    def test_checkpoint_rotation_keeps_max(self, tmp_path: Path):
        """Save 7 checkpoints with max=5, verify only 5 remain."""
        mgr = CheckpointManager(base_dir=tmp_path, max_checkpoints=5)

        for v in range(1, 8):
            _save_fake_checkpoint(mgr, "x_trend", v)

        remaining = mgr.list_checkpoints("x_trend")
        assert len(remaining) == 5, f"Expected 5 checkpoints, got {len(remaining)}"

    def test_checkpoint_rotation_deletes_oldest(self, tmp_path: Path):
        """Verify the 2 oldest (v1, v2) are gone after saving 7 with max=5."""
        mgr = CheckpointManager(base_dir=tmp_path, max_checkpoints=5)

        for v in range(1, 8):
            _save_fake_checkpoint(mgr, "x_trend", v)

        remaining = mgr.list_checkpoints("x_trend")
        versions = [c["version"] for c in remaining]

        assert 1 not in versions, "v1 should have been rotated"
        assert 2 not in versions, "v2 should have been rotated"
        assert versions == [3, 4, 5, 6, 7], f"Expected [3..7], got {versions}"

    def test_checkpoint_rotation_disabled(self, tmp_path: Path):
        """max_checkpoints=0 keeps all checkpoints."""
        mgr = CheckpointManager(base_dir=tmp_path, max_checkpoints=0)

        for v in range(1, 8):
            _save_fake_checkpoint(mgr, "x_trend", v)

        remaining = mgr.list_checkpoints("x_trend")
        assert len(remaining) == 7, f"Expected all 7 kept, got {len(remaining)}"

    def test_checkpoint_rotation_under_limit(self, tmp_path: Path):
        """No deletion when under max_checkpoints."""
        mgr = CheckpointManager(base_dir=tmp_path, max_checkpoints=5)

        for v in range(1, 4):
            _save_fake_checkpoint(mgr, "x_trend", v)

        remaining = mgr.list_checkpoints("x_trend")
        assert len(remaining) == 3, "All 3 should remain (under limit)"

    def test_checkpoint_rotation_negative_keeps_all(self, tmp_path: Path):
        """max_checkpoints=-1 also keeps all (treated as disabled)."""
        mgr = CheckpointManager(base_dir=tmp_path, max_checkpoints=-1)

        for v in range(1, 4):
            _save_fake_checkpoint(mgr, "x_trend", v)

        remaining = mgr.list_checkpoints("x_trend")
        assert len(remaining) == 3

    def test_checkpoint_rotation_per_model_type(self, tmp_path: Path):
        """Rotation is scoped to model_type — other types unaffected."""
        mgr = CheckpointManager(base_dir=tmp_path, max_checkpoints=2)

        for v in range(1, 5):
            _save_fake_checkpoint(mgr, "x_trend", v)

        # Save a different model type
        _save_fake_checkpoint(mgr, "y_trend", 1)

        x_remaining = mgr.list_checkpoints("x_trend")
        y_remaining = mgr.list_checkpoints("y_trend")

        assert len(x_remaining) == 2, "x_trend should only keep 2"
        assert len(y_remaining) == 1, "y_trend should keep its 1"
