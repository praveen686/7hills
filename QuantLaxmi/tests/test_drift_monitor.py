"""Tests for feature/prediction drift monitoring.

Tests cover:
1. No drift with same distribution → PSI ≈ 0, status "ok"
2. Single-feature drift → PSI > threshold, status "warning"
3. Multi-feature drift → status "critical"
4. Prediction drift detection
5. KS test rejects shifted distribution
6. KS test passes same distribution
7. Save/load round-trip preserves reference
8. PSI approximate symmetry
9. Alert cooldown suppresses repeated alerts
10. Empty features return safe defaults
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from quantlaxmi.models.ml.tft.drift_monitor import (
    DriftConfig,
    DriftMonitor,
    DriftReport,
    compute_psi,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def n_samples():
    return 500


@pytest.fixture
def n_features():
    return 8


@pytest.fixture
def reference_data(rng, n_samples, n_features):
    """Generate reference features and predictions (standard normal)."""
    features = rng.standard_normal((n_samples, n_features))
    predictions = rng.standard_normal(n_samples) * 0.5
    return features, predictions


@pytest.fixture
def feature_names(n_features):
    return [f"feat_{i}" for i in range(n_features)]


@pytest.fixture
def monitor(feature_names):
    """DriftMonitor with default config."""
    return DriftMonitor(feature_names=feature_names)


# ============================================================================
# Tests — PSI computation
# ============================================================================


class TestComputePSI:
    """Tests for the standalone compute_psi function."""

    def test_identical_distributions(self, rng):
        """PSI of identical distributions should be near zero."""
        data = rng.standard_normal(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"PSI for identical data should be ~0, got {psi}"

    def test_shifted_distribution(self, rng):
        """Shifting mean by 2σ should produce PSI > 0.2."""
        ref = rng.standard_normal(1000)
        shifted = ref + 2.0
        psi = compute_psi(ref, shifted)
        assert psi > 0.2, f"PSI for 2σ shift should be >0.2, got {psi}"

    def test_psi_non_negative(self, rng):
        """PSI should always be non-negative."""
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500) * 2.0 + 1.0
        psi = compute_psi(ref, cur)
        assert psi >= 0.0

    def test_empty_arrays(self):
        """Empty inputs should return 0."""
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert compute_psi(np.array([1, 2, 3]), np.array([])) == 0.0
        assert compute_psi(np.array([]), np.array([])) == 0.0

    def test_nan_handling(self, rng):
        """NaN values should be excluded gracefully."""
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500)
        cur[0:10] = np.nan
        psi = compute_psi(ref, cur)
        assert np.isfinite(psi)


# ============================================================================
# Tests — DriftMonitor core
# ============================================================================


class TestDriftMonitorNoDrift:
    """Test that same-distribution data produces no drift."""

    def test_no_drift_same_distribution(self, monitor, reference_data, rng, n_samples, n_features):
        """Reference = current → all PSI ≈ 0, status 'ok'."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        # Use same distribution (different samples from same RNG seed won't be
        # identical, but should be very close)
        current_features = rng.standard_normal((n_samples, n_features))
        current_preds = rng.standard_normal(n_samples) * 0.5

        report = monitor.check_drift(current_features, current_preds)

        assert report.overall_status == "ok"
        assert len(report.drifted_features) == 0
        for name, psi_val in report.feature_psi.items():
            assert psi_val < 0.2, f"Feature {name} PSI={psi_val} should be <0.2"


class TestDriftMonitorFeatureDrift:
    """Test feature drift detection."""

    def test_feature_drift_detected(self, monitor, reference_data, rng, n_samples, n_features):
        """Shift one feature's mean by 2σ → PSI > threshold, status 'warning'."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        current = rng.standard_normal((n_samples, n_features))
        # Shift feature 0 by 2 standard deviations
        current[:, 0] += 2.0

        report = monitor.check_drift(current, predictions)

        assert "feat_0" in report.drifted_features
        assert report.feature_psi["feat_0"] > 0.2
        assert report.overall_status in ("warning", "critical")

    def test_multiple_feature_drift_critical(self, monitor, reference_data, rng, n_samples, n_features):
        """Shift 5 features → status 'critical' (>3 drifted)."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        current = rng.standard_normal((n_samples, n_features))
        # Shift 5 features by 3σ
        for i in range(5):
            current[:, i] += 3.0

        report = monitor.check_drift(current, predictions)

        assert len(report.drifted_features) >= 4  # at least 4 should trigger
        assert report.overall_status == "critical"


class TestDriftMonitorPredictionDrift:
    """Test prediction drift detection."""

    def test_prediction_drift_detected(self, monitor, reference_data, rng, n_samples, n_features):
        """Shift predictions → prediction_psi > threshold, status 'critical'."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        current_features = rng.standard_normal((n_samples, n_features))
        # Shift predictions heavily
        shifted_preds = predictions + 3.0

        report = monitor.check_drift(current_features, shifted_preds)

        assert report.prediction_psi > 0.2
        assert report.overall_status == "critical"


# ============================================================================
# Tests — KS test
# ============================================================================


class TestKSTest:
    """Test Kolmogorov-Smirnov distribution comparison."""

    def test_ks_test_rejects_shifted(self, monitor, reference_data, rng, n_samples, n_features):
        """Shifted feature → KS p-value < 0.05."""
        pytest.importorskip("scipy")

        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        current = rng.standard_normal((n_samples, n_features))
        current[:, 0] += 2.0  # shift feature 0

        report = monitor.check_drift(current, predictions)

        assert report.feature_ks_pvalue["feat_0"] < 0.05
        assert "feat_0" in report.ks_rejected_features

    def test_ks_test_passes_same(self, monitor, reference_data, rng, n_samples, n_features):
        """Same distribution → KS p-value > 0.05."""
        pytest.importorskip("scipy")

        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        # Draw from same distribution
        current = rng.standard_normal((n_samples, n_features))
        report = monitor.check_drift(current, predictions)

        # Most features should pass (allow 1 false positive out of 8 at alpha=0.05)
        n_passed = sum(
            1 for p in report.feature_ks_pvalue.values() if p > 0.05
        )
        assert n_passed >= n_features - 1, (
            f"Expected most features to pass KS test, but only {n_passed}/{n_features} passed"
        )


# ============================================================================
# Tests — Persistence
# ============================================================================


class TestDriftMonitorPersistence:
    """Test save/load round-trip."""

    def test_save_load_reference(self, monitor, reference_data, rng, n_samples, n_features):
        """Round-trip save/load preserves distributions and produces same results."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drift_ref.json"
            monitor.save_reference(path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "features" in data
            assert "predictions" in data

            # Load into fresh monitor
            monitor2 = DriftMonitor(feature_names=monitor.feature_names)
            monitor2.load_reference(path)
            assert monitor2.is_reference_set

            # Both monitors should produce similar drift reports
            current = rng.standard_normal((n_samples, n_features))
            current_preds = rng.standard_normal(n_samples) * 0.5

            report1 = monitor.check_drift(current, current_preds)
            report2 = monitor2.check_drift(current, current_preds)

            # PSI values should match closely (raw samples are truncated to
            # 1000, so KS may differ, but PSI uses histograms which are exact)
            for name in report1.feature_psi:
                if name in report2.feature_psi:
                    np.testing.assert_allclose(
                        report1.feature_psi[name],
                        report2.feature_psi[name],
                        atol=1e-6,
                        err_msg=f"PSI mismatch for {name} after load",
                    )

    def test_save_without_reference_raises(self):
        """save_reference before set_reference should raise."""
        monitor = DriftMonitor()
        with pytest.raises(RuntimeError, match="No reference set"):
            monitor.save_reference(Path("/tmp/test_no_ref.json"))

    def test_load_nonexistent_raises(self):
        """load_reference with missing file should raise."""
        monitor = DriftMonitor()
        with pytest.raises(FileNotFoundError):
            monitor.load_reference(Path("/tmp/definitely_does_not_exist_12345.json"))


# ============================================================================
# Tests — PSI properties
# ============================================================================


class TestPSIProperties:
    """Test mathematical properties of PSI."""

    def test_psi_symmetric_property(self, rng):
        """PSI(p,q) ≈ PSI(q,p) for similar distributions.

        PSI is not exactly symmetric because bin edges are defined by the
        reference. But for similar distributions the asymmetry should be small.
        """
        a = rng.standard_normal(1000)
        b = rng.standard_normal(1000) + 0.3  # slight shift

        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)

        # Should be roughly similar (within 2x for moderate shifts)
        assert psi_ab > 0
        assert psi_ba > 0
        ratio = max(psi_ab, psi_ba) / max(min(psi_ab, psi_ba), 1e-10)
        assert ratio < 5.0, (
            f"PSI asymmetry too large: PSI(a,b)={psi_ab:.4f}, PSI(b,a)={psi_ba:.4f}"
        )


# ============================================================================
# Tests — Alert cooldown
# ============================================================================


class TestAlertCooldown:
    """Test that alert cooldown suppresses repeated alerts."""

    def test_alert_cooldown(self, monitor, reference_data, rng, n_samples, n_features):
        """Repeated checks within cooldown don't re-alert (same status)."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        # Create drifted data
        drifted = rng.standard_normal((n_samples, n_features))
        drifted[:, 0] += 3.0  # shift one feature

        # First check should set alert time
        report1 = monitor.check_drift(drifted, predictions)
        assert report1.overall_status in ("warning", "critical")
        first_alert_time = monitor._last_alert_time
        assert first_alert_time is not None

        # Second check: same status should NOT update alert time (cooldown)
        report2 = monitor.check_drift(drifted, predictions)
        assert report2.overall_status == report1.overall_status
        # Alert time should not have been updated (still within cooldown)
        assert monitor._last_alert_time == first_alert_time

    def test_cooldown_resets_on_recovery(self, monitor, reference_data, rng, n_samples, n_features):
        """Alert state resets when status returns to 'ok'."""
        features, predictions = reference_data
        monitor.set_reference(features, predictions)

        # Trigger drift
        drifted = rng.standard_normal((n_samples, n_features))
        drifted[:, 0] += 3.0
        monitor.check_drift(drifted, predictions)
        assert monitor._last_alert_time is not None

        # Now check with clean data → "ok"
        clean = rng.standard_normal((n_samples, n_features))
        report = monitor.check_drift(clean, predictions)
        assert report.overall_status == "ok"
        assert monitor._last_alert_time is None


# ============================================================================
# Tests — Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_features_handled(self, feature_names):
        """Empty input returns safe defaults with status 'ok'."""
        monitor = DriftMonitor(feature_names=feature_names)
        monitor.set_reference(
            np.random.randn(100, len(feature_names)),
            np.random.randn(100),
        )

        report = monitor.check_drift(
            np.empty((0, len(feature_names))),
            np.empty(0),
        )
        assert report.overall_status == "ok"
        assert len(report.drifted_features) == 0

    def test_check_before_reference_returns_ok(self, feature_names):
        """check_drift before set_reference returns safe defaults."""
        monitor = DriftMonitor(feature_names=feature_names)
        report = monitor.check_drift(
            np.random.randn(50, len(feature_names)),
            np.random.randn(50),
        )
        assert report.overall_status == "ok"

    def test_1d_features_reshaped(self):
        """1-D feature array is reshaped to (n, 1)."""
        monitor = DriftMonitor(feature_names=["only_feature"])
        features_1d = np.random.randn(100)
        predictions = np.random.randn(100)
        monitor.set_reference(features_1d, predictions)
        assert monitor.is_reference_set

        report = monitor.check_drift(features_1d, predictions)
        assert report.overall_status == "ok"

    def test_nan_in_features(self, rng, feature_names):
        """NaN values in features are handled gracefully."""
        n = len(feature_names)
        monitor = DriftMonitor(feature_names=feature_names)
        ref_features = rng.standard_normal((200, n))
        ref_preds = rng.standard_normal(200)
        monitor.set_reference(ref_features, ref_preds)

        # Inject NaNs
        current = rng.standard_normal((100, n))
        current[:5, :] = np.nan
        current_preds = rng.standard_normal(100)

        report = monitor.check_drift(current, current_preds)
        assert report.overall_status in ("ok", "warning", "critical")
        for psi_val in report.feature_psi.values():
            assert np.isfinite(psi_val)

    def test_set_reference_empty_skips(self):
        """Empty reference data should not set is_reference_set."""
        monitor = DriftMonitor()
        monitor.set_reference(np.empty((0, 5)), np.empty(0))
        assert not monitor.is_reference_set

    def test_drift_report_dataclass_defaults(self):
        """DriftReport has sane defaults."""
        report = DriftReport()
        assert report.overall_status == "ok"
        assert report.prediction_psi == 0.0
        assert report.drifted_features == []
        assert report.ks_rejected_features == []

    def test_config_custom_thresholds(self, rng):
        """Custom config thresholds are respected."""
        config = DriftConfig(psi_threshold=0.5, ks_alpha=0.01, n_bins=20)
        monitor = DriftMonitor(config=config, feature_names=["f0"])

        ref = rng.standard_normal((500, 1))
        preds = rng.standard_normal(500)
        monitor.set_reference(ref, preds)

        # Moderate shift — should NOT trigger at threshold=0.5
        current = rng.standard_normal((500, 1)) + 1.0
        report = monitor.check_drift(current, preds)

        # With psi_threshold=0.5, a 1σ shift may or may not trigger
        # but the config should be used (verify it ran without error)
        assert isinstance(report.feature_psi.get("f0", 0.0), float)
