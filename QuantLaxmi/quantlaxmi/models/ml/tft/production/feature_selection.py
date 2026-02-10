"""Feature Selection for Production TFT — 2-tier VSN-native approach.

Tier 1: Cheap pre-filters (coverage, correlation) — run before training.
Tier 2: VSN weight extraction + stability analysis — run after training.

The key insight: TFT's Variable Selection Network (VSN) learns per-feature
softmax weights during training. This IS feature selection. External methods
(MI, permutation importance) measure what simpler models think is important,
which may disagree with what TFT actually uses.

Usage
-----
    selector = FeatureSelector(FeatureSelectionConfig())
    
    # Tier 1: cheap pre-filter
    kept = selector.prefilter(feature_df, min_coverage=0.3, corr_threshold=0.95)
    
    # Tier 2: after training, extract VSN weights per fold
    selector.add_fold_weights(model, fold_idx)
    report = selector.stability_report()
    final_features = report.recommended_features
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection pipeline."""

    # Tier 1: pre-filter thresholds
    min_coverage: float = 0.3
    correlation_threshold: float = 0.95

    # Tier 2: VSN-native thresholds
    vsn_weight_threshold: float = 0.003
    stability_min_folds: float = 0.25
    final_max_features: int = 80

    # Optional MI validation
    run_mi_validation: bool = False
    mi_n_neighbors: int = 5


# ============================================================================
# Stability Report
# ============================================================================


@dataclass
class StabilityReport:
    """Result of VSN stability analysis across walk-forward folds.

    Attributes
    ----------
    stable_features : list[str]
        Features passing both weight and stability thresholds.
    stability_scores : dict[str, float]
        Fraction of folds where each feature exceeded the weight threshold.
    mean_vsn_weights : dict[str, float]
        Mean VSN weight across all folds per feature.
    vsn_weight_per_fold : dict[str, list[float]]
        Raw VSN weight for each feature in each fold.
    recommended_features : list[str]
        Final recommended feature subset (top N stable features).
    feature_group_breakdown : dict[str, list[str]]
        Features grouped by their prefix (e.g., 'price_', 'vol_', 'fii_').
    n_folds : int
        Number of folds analyzed.
    """

    stable_features: list[str]
    stability_scores: dict[str, float]
    mean_vsn_weights: dict[str, float]
    vsn_weight_per_fold: dict[str, list[float]]
    recommended_features: list[str]
    feature_group_breakdown: dict[str, list[str]]
    n_folds: int


# ============================================================================
# Feature Selector
# ============================================================================


class FeatureSelector:
    """2-tier feature selection: cheap pre-filter + VSN-native selection.

    Parameters
    ----------
    config : FeatureSelectionConfig
        Thresholds and settings.
    """

    def __init__(self, config: Optional[FeatureSelectionConfig] = None) -> None:
        self.config = config or FeatureSelectionConfig()
        self._fold_weights: list[dict[str, float]] = []
        self._prefilter_kept: Optional[list[str]] = None
        self._prefilter_dropped_coverage: list[str] = []
        self._prefilter_dropped_corr: list[str] = []

    # ------------------------------------------------------------------
    # Tier 1: Cheap pre-filters
    # ------------------------------------------------------------------

    def prefilter(
        self,
        features: pd.DataFrame,
        min_coverage: Optional[float] = None,
        corr_threshold: Optional[float] = None,
    ) -> list[str]:
        """Apply coverage and correlation pre-filters.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix indexed by date, one column per feature.
        min_coverage : float, optional
            Minimum fraction of non-NaN values. Defaults to config.
        corr_threshold : float, optional
            Max absolute correlation between pairs. Defaults to config.

        Returns
        -------
        list[str]
            Names of features that passed both filters.
        """
        if min_coverage is None:
            min_coverage = self.config.min_coverage
        if corr_threshold is None:
            corr_threshold = self.config.correlation_threshold

        all_names = list(features.columns)
        n_start = len(all_names)

        # --- Coverage filter ---
        kept_coverage = self._filter_coverage(features, min_coverage)
        dropped_cov = [n for n in all_names if n not in kept_coverage]
        self._prefilter_dropped_coverage = dropped_cov
        logger.info(
            "Coverage filter (>%.0f%%): %d/%d kept, %d dropped",
            min_coverage * 100, len(kept_coverage), n_start, len(dropped_cov),
        )

        # --- Correlation filter ---
        features_kept = features[kept_coverage]
        kept_final = self._filter_correlation(features_kept, corr_threshold)
        dropped_corr = [n for n in kept_coverage if n not in kept_final]
        self._prefilter_dropped_corr = dropped_corr
        logger.info(
            "Correlation filter (|r|<%.2f): %d/%d kept, %d dropped",
            corr_threshold, len(kept_final), len(kept_coverage), len(dropped_corr),
        )

        self._prefilter_kept = kept_final
        logger.info(
            "Pre-filter: %d → %d features (%.0f%% reduction)",
            n_start, len(kept_final),
            (1.0 - len(kept_final) / max(n_start, 1)) * 100,
        )
        return kept_final

    def _filter_coverage(
        self, features: pd.DataFrame, min_coverage: float
    ) -> list[str]:
        """Drop features with less than min_coverage fraction non-NaN."""
        coverage = features.notna().mean()
        mask = coverage >= min_coverage
        return list(features.columns[mask])

    def _filter_correlation(
        self, features: pd.DataFrame, threshold: float
    ) -> list[str]:
        """Remove highly correlated features, keeping higher-variance one.

        For each pair with |corr| > threshold, drop the one with lower
        variance (it carries less information).
        """
        # Compute correlation matrix (use pairwise complete obs)
        corr_matrix = features.corr().abs()
        n_features = len(features.columns)
        to_drop: set[str] = set()
        variances = features.var()

        # Upper triangle scan
        cols = list(features.columns)
        for i in range(n_features):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, n_features):
                if cols[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > threshold:
                    # Drop the lower-variance feature
                    if variances[cols[i]] >= variances[cols[j]]:
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])

        return [c for c in cols if c not in to_drop]

    # ------------------------------------------------------------------
    # Tier 2: VSN-native selection
    # ------------------------------------------------------------------

    def extract_vsn_weights(
        self,
        model: "torch.nn.Module",
        feature_names: list[str],
        device: Optional["torch.device"] = None,
    ) -> dict[str, float]:
        """Extract per-feature importance from a trained model's VSN.

        Runs a unit input through the VSN weight_grn and reads the
        softmax output. This gives the model's learned feature weights.

        Parameters
        ----------
        model : nn.Module
            Trained XTrendModel or XTrendBackbone.
        feature_names : list[str]
            Feature names matching the model's n_features.
        device : torch.device, optional
            Device to use for inference.

        Returns
        -------
        dict[str, float]
            Feature name → VSN weight (sums to ~1.0).
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for VSN weight extraction")

        # Find the VSN module
        vsn = None
        if hasattr(model, 'vsn'):
            vsn = model.vsn
        elif hasattr(model, 'model') and hasattr(model.model, 'vsn'):
            vsn = model.model.vsn

        if vsn is None:
            raise ValueError("Could not find VSN in model")

        n_feat = vsn.n_features
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        model.eval()
        with torch.no_grad():
            # Run unit input through weight_grn
            dummy = torch.ones(1, 1, n_feat, device=device)
            raw_weights = vsn.weight_grn(dummy)
            weights = torch.softmax(raw_weights, dim=-1)
            w = weights.squeeze().cpu().numpy()

        result = {}
        for i, name in enumerate(feature_names):
            if i < len(w):
                result[name] = float(w[i])
        return result

    def add_fold_weights(
        self,
        model: "torch.nn.Module",
        feature_names: list[str],
        device: Optional["torch.device"] = None,
    ) -> dict[str, float]:
        """Extract and store VSN weights from one fold's trained model.

        Call this after training each walk-forward fold. The weights are
        accumulated for later stability analysis.

        Parameters
        ----------
        model : nn.Module
            Trained model for this fold.
        feature_names : list[str]
            Feature names.
        device : torch.device, optional

        Returns
        -------
        dict[str, float]
            The extracted weights for this fold.
        """
        weights = self.extract_vsn_weights(model, feature_names, device)
        self._fold_weights.append(weights)
        logger.info(
            "Added fold %d VSN weights (%d features, top-5: %s)",
            len(self._fold_weights),
            len(weights),
            ", ".join(f"{k}={v:.4f}" for k, v in
                      sorted(weights.items(), key=lambda x: -x[1])[:5]),
        )
        return weights

    def stability_report(
        self,
        feature_names: Optional[list[str]] = None,
    ) -> StabilityReport:
        """Analyze VSN weight stability across folds.

        For each feature, computes:
        - Mean weight across folds
        - Stability score: fraction of folds where weight > threshold
        - Whether it qualifies as "stable" (score >= min_folds AND
          mean weight > threshold)

        Parameters
        ----------
        feature_names : list[str], optional
            If not provided, uses union of all fold feature names.

        Returns
        -------
        StabilityReport
        """
        if not self._fold_weights:
            raise ValueError("No fold weights added. Call add_fold_weights() first.")

        n_folds = len(self._fold_weights)
        threshold = self.config.vsn_weight_threshold
        min_stability = self.config.stability_min_folds

        # Collect all feature names
        if feature_names is None:
            seen: set[str] = set()
            feature_names = []
            for fw in self._fold_weights:
                for name in fw:
                    if name not in seen:
                        seen.add(name)
                        feature_names.append(name)

        # Compute per-feature statistics
        vsn_weight_per_fold: dict[str, list[float]] = {}
        mean_vsn_weights: dict[str, float] = {}
        stability_scores: dict[str, float] = {}

        for name in feature_names:
            weights = []
            n_above = 0
            for fw in self._fold_weights:
                w = fw.get(name, 0.0)
                weights.append(w)
                if w > threshold:
                    n_above += 1

            vsn_weight_per_fold[name] = weights
            mean_vsn_weights[name] = float(np.mean(weights))
            stability_scores[name] = n_above / n_folds

        # Stable features: pass both thresholds
        stable_features = [
            name for name in feature_names
            if stability_scores[name] >= min_stability
            and mean_vsn_weights[name] > threshold
        ]

        # Sort by mean weight descending
        stable_features.sort(key=lambda n: -mean_vsn_weights[n])

        # Recommended: top N stable features
        max_feat = self.config.final_max_features
        recommended = stable_features[:max_feat]

        # Group breakdown by prefix
        group_breakdown: dict[str, list[str]] = {}
        for name in recommended:
            prefix = name.split("_")[0] if "_" in name else "other"
            group_breakdown.setdefault(prefix, []).append(name)

        logger.info(
            "Stability report: %d/%d features stable, %d recommended "
            "(threshold=%.4f, min_stability=%.1f%%)",
            len(stable_features), len(feature_names), len(recommended),
            threshold, min_stability * 100,
        )

        return StabilityReport(
            stable_features=stable_features,
            stability_scores=stability_scores,
            mean_vsn_weights=mean_vsn_weights,
            vsn_weight_per_fold=vsn_weight_per_fold,
            recommended_features=recommended,
            feature_group_breakdown=group_breakdown,
            n_folds=n_folds,
        )

    # ------------------------------------------------------------------
    # Optional: MI validation (sanity check, not for selection)
    # ------------------------------------------------------------------

    def mi_validation(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        vsn_ranking: dict[str, float],
    ) -> dict[str, float]:
        """Compare VSN ranking against mutual information ranking.

        This is a SANITY CHECK only — not used for feature selection.
        Large disagreements may indicate training issues.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (columns = feature names).
        targets : np.ndarray
            Target variable (next-day returns).
        vsn_ranking : dict[str, float]
            VSN weight per feature.

        Returns
        -------
        dict with keys: 'rank_correlation', 'top_disagreements'
        """
        from sklearn.feature_selection import mutual_info_regression

        # Clean data
        df = features.copy()
        df["_target"] = targets
        df = df.dropna()

        if len(df) < 30:
            logger.warning("MI validation: insufficient clean samples (%d)", len(df))
            return {"rank_correlation": float("nan"), "top_disagreements": []}

        X = df.drop(columns=["_target"]).values
        y = df["_target"].values
        col_names = list(df.drop(columns=["_target"]).columns)

        mi_scores = mutual_info_regression(
            X, y, n_neighbors=self.config.mi_n_neighbors, random_state=42
        )

        mi_ranking = {name: float(score) for name, score in zip(col_names, mi_scores)}

        # Rank correlation (Spearman)
        common = sorted(set(vsn_ranking) & set(mi_ranking))
        if len(common) < 3:
            return {"rank_correlation": float("nan"), "top_disagreements": []}

        vsn_ranks = np.argsort(np.argsort([-vsn_ranking[n] for n in common]))
        mi_ranks = np.argsort(np.argsort([-mi_ranking[n] for n in common]))

        from scipy.stats import spearmanr
        corr, _ = spearmanr(vsn_ranks, mi_ranks)

        # Top disagreements: features where rank difference is largest
        rank_diffs = [
            (common[i], abs(int(vsn_ranks[i]) - int(mi_ranks[i])))
            for i in range(len(common))
        ]
        rank_diffs.sort(key=lambda x: -x[1])
        top_disagree = rank_diffs[:10]

        logger.info(
            "MI validation: rank correlation=%.3f, top disagreement=%s (Δ=%d)",
            corr,
            top_disagree[0][0] if top_disagree else "N/A",
            top_disagree[0][1] if top_disagree else 0,
        )

        return {
            "rank_correlation": float(corr),
            "top_disagreements": top_disagree,
            "mi_ranking": mi_ranking,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save selector state (fold weights, config, prefilter results)."""
        path = Path(path)
        state = {
            "config": self.config,
            "fold_weights": self._fold_weights,
            "prefilter_kept": self._prefilter_kept,
            "prefilter_dropped_coverage": self._prefilter_dropped_coverage,
            "prefilter_dropped_corr": self._prefilter_dropped_corr,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved FeatureSelector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "FeatureSelector":
        """Load selector state from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        selector = cls(config=state["config"])
        selector._fold_weights = state["fold_weights"]
        selector._prefilter_kept = state["prefilter_kept"]
        selector._prefilter_dropped_coverage = state["prefilter_dropped_coverage"]
        selector._prefilter_dropped_corr = state["prefilter_dropped_corr"]
        logger.info("Loaded FeatureSelector from %s (%d folds)", path, len(selector._fold_weights))
        return selector

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def feature_importance_report(
        self,
        report: Optional[StabilityReport] = None,
    ) -> str:
        """Generate a human-readable feature importance report.

        Parameters
        ----------
        report : StabilityReport, optional
            If not provided, generates from current fold weights.

        Returns
        -------
        str
            Formatted report.
        """
        if report is None:
            report = self.stability_report()

        lines = [
            "=" * 70,
            "FEATURE IMPORTANCE REPORT",
            f"Folds analyzed: {report.n_folds}",
            f"Stable features: {len(report.stable_features)}",
            f"Recommended features: {len(report.recommended_features)}",
            "=" * 70,
            "",
            f"{'Rank':<5} {'Feature':<40} {'Mean Weight':>11} {'Stability':>10}",
            "-" * 70,
        ]

        # Sort by mean weight
        sorted_feats = sorted(
            report.mean_vsn_weights.items(),
            key=lambda x: -x[1],
        )

        for rank, (name, weight) in enumerate(sorted_feats[:50], 1):
            stability = report.stability_scores.get(name, 0.0)
            marker = "*" if name in report.recommended_features else " "
            lines.append(
                f"{rank:<5} {name:<40} {weight:>10.5f} {stability:>9.1%} {marker}"
            )

        lines.append("")
        lines.append("Group breakdown (recommended features):")
        lines.append("-" * 40)
        for group, feats in sorted(report.feature_group_breakdown.items()):
            lines.append(f"  {group}: {len(feats)} features")

        lines.append("")
        lines.append("* = recommended for production")

        return "\n".join(lines)

    @property
    def n_folds(self) -> int:
        """Number of fold weights accumulated."""
        return len(self._fold_weights)

    @property
    def prefilter_kept(self) -> Optional[list[str]]:
        """Features kept after pre-filtering, or None if not run."""
        return self._prefilter_kept
