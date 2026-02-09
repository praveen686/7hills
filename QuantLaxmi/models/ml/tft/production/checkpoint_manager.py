"""Checkpoint Manager for Production TFT — versioned save/load.

Saves everything needed for production inference:
- model.pt (state_dict only — small, portable)
- metadata.json (config, features, normalization, metrics)
- feature_selector.pkl (optional — for reproducing feature selection)
- optuna_study.db (optional — for reviewing HP search)

Usage
-----
    mgr = CheckpointManager(base_dir="checkpoints")
    
    # Save
    mgr.save(model.state_dict(), metadata, feature_selector, optuna_study)
    
    # Load
    state_dict, metadata = mgr.load("checkpoints/x_trend/v3_20260209_143022")
    
    # Load latest
    state_dict, metadata = mgr.load_latest("x_trend")
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ============================================================================
# Metadata
# ============================================================================


@dataclass
class CheckpointMetadata:
    """All metadata needed to reconstruct a model for inference.

    This is serialized to metadata.json inside the checkpoint directory.
    """

    # Versioning
    version: int = 1
    model_type: str = "x_trend"
    created_at: str = ""
    git_commit: str = ""

    # Feature info
    feature_names: list[str] = field(default_factory=list)
    n_features: int = 0
    n_assets: int = 4
    asset_names: list[str] = field(default_factory=list)

    # Model config (full XTrendConfig as dict)
    config: dict[str, Any] = field(default_factory=dict)

    # Normalization stats (for applying saved normalization at inference)
    normalization: dict[str, Any] = field(default_factory=dict)

    # Training info
    training_info: dict[str, Any] = field(default_factory=dict)

    # Feature selection info
    feature_selection: dict[str, Any] = field(default_factory=dict)

    # Optuna results
    optuna_best_params: dict[str, Any] = field(default_factory=dict)

    # OOS performance
    sharpe_oos: float = 0.0
    total_return_oos: float = 0.0
    max_drawdown_oos: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        # Convert numpy types
        d = _convert_numpy(d)
        return json.dumps(d, indent=2, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CheckpointMetadata":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ============================================================================
# Checkpoint Manager
# ============================================================================


class CheckpointManager:
    """Manages versioned model checkpoints on disk.

    Directory layout::

        base_dir/
        └── {model_type}/
            ├── v1_20260209_140000/
            │   ├── model.pt
            │   ├── metadata.json
            │   ├── feature_selector.pkl  (optional)
            │   └── optuna_study.db       (optional)
            ├── v2_20260209_160000/
            │   └── ...
            └── ...

    Parameters
    ----------
    base_dir : str or Path
        Root directory for checkpoints. Defaults to "checkpoints".
    """

    def __init__(self, base_dir: str | Path = "checkpoints") -> None:
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        state_dict: dict,
        metadata: CheckpointMetadata,
        feature_selector: Optional[Any] = None,
        optuna_study: Optional[Any] = None,
    ) -> Path:
        """Save a checkpoint to disk.

        Parameters
        ----------
        state_dict : dict
            Model state_dict (from model.state_dict()).
        metadata : CheckpointMetadata
            Full metadata for this checkpoint.
        feature_selector : FeatureSelector, optional
            Saves as feature_selector.pkl.
        optuna_study : optuna.Study, optional
            Saves study DB for review.

        Returns
        -------
        Path
            The checkpoint directory path.
        """
        # Auto-fill metadata
        if not metadata.created_at:
            metadata.created_at = datetime.now().isoformat()
        if not metadata.git_commit:
            metadata.git_commit = self._get_git_commit()
        if metadata.version <= 0:
            metadata.version = self._next_version(metadata.model_type)

        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"v{metadata.version}_{timestamp}"
        ckpt_dir = self.base_dir / metadata.model_type / dir_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state_dict
        model_path = ckpt_dir / "model.pt"
        if _HAS_TORCH:
            torch.save(state_dict, model_path)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(state_dict, f)
        logger.info("Saved model state_dict to %s", model_path)

        # Save metadata
        meta_path = ckpt_dir / "metadata.json"
        meta_path.write_text(metadata.to_json())
        logger.info("Saved metadata to %s", meta_path)

        # Save feature selector (optional)
        if feature_selector is not None:
            fs_path = ckpt_dir / "feature_selector.pkl"
            if hasattr(feature_selector, "save"):
                feature_selector.save(fs_path)
            else:
                with open(fs_path, "wb") as f:
                    pickle.dump(feature_selector, f)
            logger.info("Saved feature selector to %s", fs_path)

        # Save optuna study (optional)
        if optuna_study is not None:
            study_path = ckpt_dir / "optuna_study.db"
            try:
                import optuna
                storage = optuna.storages.RDBStorage(
                    url=f"sqlite:///{study_path}"
                )
                # Copy study to new storage
                optuna.copy_study(
                    from_study_name=optuna_study.study_name,
                    from_storage=optuna_study._storage,
                    to_storage=storage,
                )
            except Exception as e:
                # Fallback: pickle the study
                pkl_path = ckpt_dir / "optuna_study.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(optuna_study, f)
                logger.warning("Could not save study DB, used pickle: %s", e)
            logger.info("Saved optuna study to %s", study_path)

        logger.info(
            "Checkpoint saved: %s (v%d, %d features, Sharpe=%.3f)",
            ckpt_dir, metadata.version, metadata.n_features, metadata.sharpe_oos,
        )
        return ckpt_dir

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_dir: str | Path,
    ) -> tuple[dict, CheckpointMetadata]:
        """Load a checkpoint from disk.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Path to the checkpoint directory.

        Returns
        -------
        state_dict : dict
            Model state_dict.
        metadata : CheckpointMetadata
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        # Load metadata
        meta_path = ckpt_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {ckpt_dir}")
        metadata = CheckpointMetadata.from_json(meta_path.read_text())

        # Load state_dict
        model_path = ckpt_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")

        if _HAS_TORCH:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        else:
            with open(model_path, "rb") as f:
                state_dict = pickle.load(f)

        logger.info(
            "Loaded checkpoint: %s (v%d, %d features)",
            ckpt_dir, metadata.version, metadata.n_features,
        )
        return state_dict, metadata

    def load_latest(
        self,
        model_type: str = "x_trend",
    ) -> tuple[dict, CheckpointMetadata]:
        """Load the most recent checkpoint for a model type.

        Parameters
        ----------
        model_type : str
            Model type directory name.

        Returns
        -------
        state_dict, metadata
        """
        checkpoints = self.list_checkpoints(model_type)
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for model_type={model_type!r} "
                f"in {self.base_dir}"
            )
        latest = checkpoints[-1]
        return self.load(latest["path"])

    def load_feature_selector(
        self,
        checkpoint_dir: str | Path,
    ) -> Any:
        """Load the feature selector from a checkpoint.

        Parameters
        ----------
        checkpoint_dir : str or Path

        Returns
        -------
        FeatureSelector or None
        """
        ckpt_dir = Path(checkpoint_dir)
        fs_path = ckpt_dir / "feature_selector.pkl"
        if not fs_path.exists():
            logger.info("No feature_selector.pkl in %s", ckpt_dir)
            return None

        from .feature_selection import FeatureSelector
        return FeatureSelector.load(fs_path)

    # ------------------------------------------------------------------
    # List / query
    # ------------------------------------------------------------------

    def list_checkpoints(
        self,
        model_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List all checkpoints, sorted by creation time.

        Parameters
        ----------
        model_type : str, optional
            Filter by model type. If None, list all.

        Returns
        -------
        list[dict]
            Each dict has keys: path, version, model_type, created_at,
            sharpe_oos, n_features.
        """
        results = []

        if model_type:
            type_dirs = [self.base_dir / model_type]
        else:
            if not self.base_dir.exists():
                return []
            type_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        for type_dir in type_dirs:
            if not type_dir.exists():
                continue
            for ckpt_dir in sorted(type_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                meta_path = ckpt_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    meta = CheckpointMetadata.from_json(meta_path.read_text())
                    results.append({
                        "path": ckpt_dir,
                        "version": meta.version,
                        "model_type": meta.model_type,
                        "created_at": meta.created_at,
                        "sharpe_oos": meta.sharpe_oos,
                        "n_features": meta.n_features,
                    })
                except Exception as e:
                    logger.warning("Could not read %s: %s", meta_path, e)

        # Sort by version
        results.sort(key=lambda x: (x["model_type"], x["version"]))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_version(self, model_type: str) -> int:
        """Compute next version number for a model type."""
        existing = self.list_checkpoints(model_type)
        if not existing:
            return 1
        return max(c["version"] for c in existing) + 1

    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash, or empty string."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""
