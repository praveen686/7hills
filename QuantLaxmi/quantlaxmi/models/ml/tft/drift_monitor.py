"""Feature and prediction drift monitoring for TFT trading models.

Detects when feature distributions or model predictions have shifted from
training time, indicating model degradation. Uses two complementary tests:

1. **Population Stability Index (PSI)** — histogram-based divergence measure.
   PSI = sum( (p_i - q_i) * ln(p_i / q_i) )  where p=current, q=reference.
   Thresholds: <0.1 stable, 0.1–0.2 moderate, >0.2 significant drift.

2. **Kolmogorov–Smirnov test** — nonparametric two-sample test.
   Rejects H0 (same distribution) when p-value < alpha.

The monitor stores reference distributions (training-time histograms) and
compares live data windows against them. Alerts are rate-limited via a
cooldown mechanism to avoid log spam.

Usage
-----
    monitor = DriftMonitor()
    monitor.set_reference(train_features, train_predictions)

    # In production loop
    report = monitor.check_drift(live_features, live_predictions)
    if report.overall_status == "critical":
        logger.warning("Model drift detected: %s", report.drifted_features)

    # Persistence
    monitor.save_reference(Path("checkpoints/drift_ref.json"))
    monitor.load_reference(Path("checkpoints/drift_ref.json"))

References
----------
- Siddiqi (2006), "Credit Risk Scorecards" — PSI methodology
- Scipy docs, scipy.stats.ks_2samp — two-sample KS test
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.stats import ks_2samp

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logger.warning("scipy not available; KS tests will be skipped")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DriftConfig:
    """Configuration for drift detection thresholds.

    Parameters
    ----------
    psi_threshold : float
        PSI values above this indicate significant drift (default 0.2).
    ks_alpha : float
        Significance level for KS test (default 0.05).
    window_size : int
        Rolling window size for live monitoring (default 21 trading days).
    n_bins : int
        Number of histogram bins for PSI calculation (default 10).
    alert_cooldown : int
        Minimum days between repeated alerts for the same status (default 5).
    """

    psi_threshold: float = 0.2
    ks_alpha: float = 0.05
    window_size: int = 21
    n_bins: int = 10
    alert_cooldown: int = 5


# ============================================================================
# Drift Report
# ============================================================================


@dataclass
class DriftReport:
    """Results of a single drift check.

    Attributes
    ----------
    timestamp : str
        ISO-8601 timestamp of the check.
    feature_psi : dict[str, float]
        PSI value per feature.
    feature_ks_pvalue : dict[str, float]
        KS test p-value per feature.
    prediction_psi : float
        PSI of prediction distribution vs reference.
    drifted_features : list[str]
        Feature names with PSI > threshold.
    ks_rejected_features : list[str]
        Feature names where KS test rejects H0 (p < alpha).
    overall_status : str
        "ok" — no drift detected.
        "warning" — 1–3 features drifted.
        "critical" — >3 features drifted OR prediction drift.
    """

    timestamp: str = ""
    feature_psi: dict[str, float] = field(default_factory=dict)
    feature_ks_pvalue: dict[str, float] = field(default_factory=dict)
    prediction_psi: float = 0.0
    drifted_features: list[str] = field(default_factory=list)
    ks_rejected_features: list[str] = field(default_factory=list)
    overall_status: str = "ok"


# ============================================================================
# PSI Calculation
# ============================================================================


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index between two 1-D distributions.

    Uses the reference distribution to define bin edges, then computes the
    proportion of observations in each bin for both distributions.

    PSI = sum( (p_i - q_i) * ln(p_i / q_i) )

    where q_i = reference proportions, p_i = current proportions.

    Parameters
    ----------
    reference : np.ndarray
        Reference (training-time) data, 1-D.
    current : np.ndarray
        Current (live) data, 1-D.
    n_bins : int
        Number of histogram bins.
    eps : float
        Small constant added to proportions to avoid log(0).

    Returns
    -------
    float
        PSI value. 0 = identical distributions.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    current = np.asarray(current, dtype=np.float64).ravel()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Remove NaN/Inf
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Compute bin edges from reference distribution
    _, bin_edges = np.histogram(ref_clean, bins=n_bins)

    # Ensure current data falls within bins by extending edges
    bin_edges[0] = min(bin_edges[0], np.min(cur_clean))
    bin_edges[-1] = max(bin_edges[-1], np.max(cur_clean))

    # Compute proportions in each bin
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_props = ref_counts / len(ref_clean) + eps
    cur_props = cur_counts / len(cur_clean) + eps

    # PSI = sum( (p_i - q_i) * ln(p_i / q_i) )
    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return max(psi, 0.0)  # PSI is non-negative by construction


def compute_psi_from_edges(
    bin_edges: np.ndarray,
    ref_proportions: np.ndarray,
    current: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute PSI using pre-computed reference bin edges and proportions.

    Parameters
    ----------
    bin_edges : np.ndarray
        Bin edges from reference histogram.
    ref_proportions : np.ndarray
        Reference proportions per bin (with eps already added, or raw).
    current : np.ndarray
        Current data, 1-D.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        PSI value.
    """
    current = np.asarray(current, dtype=np.float64).ravel()
    cur_clean = current[np.isfinite(current)]

    if len(cur_clean) == 0:
        return 0.0

    # Extend edges to cover current data
    edges = np.array(bin_edges, dtype=np.float64)
    edges[0] = min(edges[0], np.min(cur_clean))
    edges[-1] = max(edges[-1], np.max(cur_clean))

    cur_counts, _ = np.histogram(cur_clean, bins=edges)
    cur_props = cur_counts / len(cur_clean) + eps

    ref_props = np.asarray(ref_proportions, dtype=np.float64) + eps

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return max(psi, 0.0)


# ============================================================================
# Drift Monitor
# ============================================================================


class DriftMonitor:
    """Monitor feature and prediction drift against training-time reference.

    The workflow is:
    1. ``set_reference()`` — store training distributions (histograms).
    2. ``check_drift()`` — compare live data against reference.
    3. Optionally ``save_reference()`` / ``load_reference()`` for persistence.

    Parameters
    ----------
    config : DriftConfig, optional
        Drift detection configuration. Defaults to ``DriftConfig()``.
    feature_names : list[str], optional
        Names for each feature column. If not provided, features are named
        ``feature_0``, ``feature_1``, etc.
    """

    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self.config = config or DriftConfig()
        self.feature_names = feature_names

        # Reference distributions: per-feature histograms
        self._ref_bin_edges: dict[str, np.ndarray] = {}
        self._ref_proportions: dict[str, np.ndarray] = {}
        self._ref_raw: dict[str, np.ndarray] = {}  # for KS test

        # Reference prediction distribution
        self._pred_bin_edges: Optional[np.ndarray] = None
        self._pred_proportions: Optional[np.ndarray] = None
        self._pred_raw: Optional[np.ndarray] = None

        self._is_reference_set = False

        # Alert cooldown tracking
        self._last_alert_time: Optional[datetime] = None
        self._last_alert_status: Optional[str] = None

    @property
    def is_reference_set(self) -> bool:
        """Whether reference distributions have been set."""
        return self._is_reference_set

    def _get_feature_names(self, n_features: int) -> list[str]:
        """Return feature names, auto-generating if needed."""
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                logger.warning(
                    "feature_names length (%d) != n_features (%d), auto-generating",
                    len(self.feature_names),
                    n_features,
                )
                return [f"feature_{i}" for i in range(n_features)]
            return self.feature_names
        return [f"feature_{i}" for i in range(n_features)]

    def set_reference(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Store training-time distributions as reference.

        Parameters
        ----------
        features : np.ndarray
            Training features, shape (n_samples, n_features).
        predictions : np.ndarray
            Training predictions, shape (n_samples,).
        """
        features = np.asarray(features, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64).ravel()

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_samples, n_features = features.shape
        names = self._get_feature_names(n_features)

        if n_samples == 0:
            logger.warning("Empty features passed to set_reference; skipping")
            return

        n_bins = self.config.n_bins

        self._ref_bin_edges = {}
        self._ref_proportions = {}
        self._ref_raw = {}

        for i, name in enumerate(names):
            col = features[:, i]
            col_clean = col[np.isfinite(col)]
            if len(col_clean) < 2:
                logger.debug("Feature %s has <2 finite values, skipping", name)
                continue

            counts, edges = np.histogram(col_clean, bins=n_bins)
            props = counts / len(col_clean)

            self._ref_bin_edges[name] = edges
            self._ref_proportions[name] = props
            self._ref_raw[name] = col_clean

        # Prediction reference
        pred_clean = predictions[np.isfinite(predictions)]
        if len(pred_clean) >= 2:
            counts, edges = np.histogram(pred_clean, bins=n_bins)
            self._pred_bin_edges = edges
            self._pred_proportions = counts / len(pred_clean)
            self._pred_raw = pred_clean

        self._is_reference_set = True
        logger.info(
            "Drift reference set: %d features, %d samples, %d prediction samples",
            len(self._ref_bin_edges),
            n_samples,
            len(pred_clean) if pred_clean is not None else 0,
        )

    def check_drift(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
    ) -> DriftReport:
        """Compare current data window against reference distributions.

        Parameters
        ----------
        features : np.ndarray
            Current features, shape (n_samples, n_features).
        predictions : np.ndarray
            Current predictions, shape (n_samples,).

        Returns
        -------
        DriftReport
            Detailed drift analysis results.
        """
        if not self._is_reference_set:
            logger.warning("check_drift called before set_reference; returning safe defaults")
            return DriftReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                overall_status="ok",
            )

        features = np.asarray(features, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64).ravel()

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        # Handle empty input
        if features.shape[0] == 0:
            return DriftReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                overall_status="ok",
            )

        n_features = features.shape[1]
        names = self._get_feature_names(n_features)

        feature_psi: dict[str, float] = {}
        feature_ks: dict[str, float] = {}
        drifted: list[str] = []
        ks_rejected: list[str] = []

        eps = 1e-6

        for i, name in enumerate(names):
            if name not in self._ref_bin_edges:
                continue

            col = features[:, i]
            col_clean = col[np.isfinite(col)]

            if len(col_clean) == 0:
                feature_psi[name] = 0.0
                feature_ks[name] = 1.0
                continue

            # PSI
            psi = compute_psi_from_edges(
                self._ref_bin_edges[name],
                self._ref_proportions[name],
                col_clean,
                eps=eps,
            )
            feature_psi[name] = psi

            if psi > self.config.psi_threshold:
                drifted.append(name)

            # KS test
            if _HAS_SCIPY and name in self._ref_raw:
                ref_data = self._ref_raw[name]
                if len(ref_data) >= 2 and len(col_clean) >= 2:
                    _, p_value = ks_2samp(ref_data, col_clean)
                    feature_ks[name] = float(p_value)
                    if p_value < self.config.ks_alpha:
                        ks_rejected.append(name)
                else:
                    feature_ks[name] = 1.0
            else:
                feature_ks[name] = 1.0  # no scipy → assume no rejection

        # Prediction PSI
        prediction_psi = 0.0
        if self._pred_bin_edges is not None and self._pred_proportions is not None:
            pred_clean = predictions[np.isfinite(predictions)]
            if len(pred_clean) > 0:
                prediction_psi = compute_psi_from_edges(
                    self._pred_bin_edges,
                    self._pred_proportions,
                    pred_clean,
                    eps=eps,
                )

        # Determine overall status
        n_drifted = len(drifted)
        pred_drifted = prediction_psi > self.config.psi_threshold

        if n_drifted > 3 or pred_drifted:
            status = "critical"
        elif n_drifted >= 1:
            status = "warning"
        else:
            status = "ok"

        now = datetime.now(timezone.utc)
        report = DriftReport(
            timestamp=now.isoformat(),
            feature_psi=feature_psi,
            feature_ks_pvalue=feature_ks,
            prediction_psi=prediction_psi,
            drifted_features=drifted,
            ks_rejected_features=ks_rejected,
            overall_status=status,
        )

        # Alert cooldown: suppress repeated alerts
        should_alert = True
        if self._last_alert_time is not None and self._last_alert_status == status:
            days_since = (now - self._last_alert_time).days
            if days_since < self.config.alert_cooldown:
                should_alert = False

        if should_alert and status != "ok":
            logger.warning(
                "Drift detected [%s]: %d features drifted (PSI>%.2f), "
                "prediction_psi=%.4f, ks_rejected=%d",
                status,
                n_drifted,
                self.config.psi_threshold,
                prediction_psi,
                len(ks_rejected),
            )
            self._last_alert_time = now
            self._last_alert_status = status
        elif status == "ok":
            # Reset cooldown on recovery
            self._last_alert_time = None
            self._last_alert_status = None

        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_reference(self, path: Path) -> None:
        """Save reference distributions to disk as JSON.

        Parameters
        ----------
        path : Path
            Output file path (JSON format).
        """
        if not self._is_reference_set:
            raise RuntimeError("No reference set; call set_reference() first")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "config": asdict(self.config),
            "feature_names": list(self._ref_bin_edges.keys()),
            "features": {},
            "predictions": None,
        }

        for name in self._ref_bin_edges:
            data["features"][name] = {
                "bin_edges": self._ref_bin_edges[name].tolist(),
                "proportions": self._ref_proportions[name].tolist(),
                "raw_sample": self._ref_raw[name][:1000].tolist()
                if name in self._ref_raw
                else [],
            }

        if self._pred_bin_edges is not None:
            data["predictions"] = {
                "bin_edges": self._pred_bin_edges.tolist(),
                "proportions": self._pred_proportions.tolist(),
                "raw_sample": self._pred_raw[:1000].tolist()
                if self._pred_raw is not None
                else [],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Drift reference saved to %s", path)

    def load_reference(self, path: Path) -> None:
        """Load reference distributions from disk.

        Parameters
        ----------
        path : Path
            Input file path (JSON format, as produced by ``save_reference``).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Drift reference file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Restore config
        if "config" in data:
            cfg = data["config"]
            self.config = DriftConfig(
                psi_threshold=cfg.get("psi_threshold", 0.2),
                ks_alpha=cfg.get("ks_alpha", 0.05),
                window_size=cfg.get("window_size", 21),
                n_bins=cfg.get("n_bins", 10),
                alert_cooldown=cfg.get("alert_cooldown", 5),
            )

        # Restore feature references
        self._ref_bin_edges = {}
        self._ref_proportions = {}
        self._ref_raw = {}

        for name, fdata in data.get("features", {}).items():
            self._ref_bin_edges[name] = np.array(fdata["bin_edges"], dtype=np.float64)
            self._ref_proportions[name] = np.array(
                fdata["proportions"], dtype=np.float64
            )
            if fdata.get("raw_sample"):
                self._ref_raw[name] = np.array(
                    fdata["raw_sample"], dtype=np.float64
                )

        # Restore prediction reference
        pdata = data.get("predictions")
        if pdata is not None:
            self._pred_bin_edges = np.array(pdata["bin_edges"], dtype=np.float64)
            self._pred_proportions = np.array(
                pdata["proportions"], dtype=np.float64
            )
            if pdata.get("raw_sample"):
                self._pred_raw = np.array(
                    pdata["raw_sample"], dtype=np.float64
                )
        else:
            self._pred_bin_edges = None
            self._pred_proportions = None
            self._pred_raw = None

        self._is_reference_set = True
        self.feature_names = data.get("feature_names", list(self._ref_bin_edges.keys()))

        logger.info(
            "Drift reference loaded from %s: %d features",
            path,
            len(self._ref_bin_edges),
        )
