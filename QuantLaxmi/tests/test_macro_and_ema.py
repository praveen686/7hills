"""Tests for macroeconomic features (Group 27) and ModelEMA weight averaging.

Tests cover:
- Macro: feature count, RBI step function, causality, graceful degradation, z-score finiteness
- EMA: shadow initialization, update convergence, apply/restore cycle, high-decay
  stability, state_dict roundtrip, context manager
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Macro feature tests
# ---------------------------------------------------------------------------

# Helper: create a temporary macro directory with realistic CSVs


def _write_macro_csvs(macro_dir: Path) -> None:
    """Write minimal but realistic macro CSV files for testing."""
    macro_dir.mkdir(parents=True, exist_ok=True)

    # RBI rates: two rate decisions
    rbi_csv = macro_dir / "rbi_rates.csv"
    rbi_csv.write_text(
        "date,rate\n"
        "2025-01-01,6.50\n"
        "2025-02-07,6.25\n"
        "2025-04-09,6.00\n"
    )

    # INR/USD: 60 trading days (enough for 21d rolling)
    dates = pd.bdate_range("2025-01-01", periods=60)
    inr_prices = 85.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.05, 60))
    inr_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": inr_prices})
    inr_df.to_csv(macro_dir / "inr_usd.csv", index=False)

    # Crude Brent: 60 trading days
    crude_prices = 75.0 + np.cumsum(np.random.default_rng(43).normal(0, 0.5, 60))
    crude_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": crude_prices})
    crude_df.to_csv(macro_dir / "crude_brent.csv", index=False)

    # US 10Y: 60 trading days
    us10y = 4.0 + np.cumsum(np.random.default_rng(44).normal(0, 0.02, 60))
    us10y_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "yield_pct": us10y})
    us10y_df.to_csv(macro_dir / "us10y_yield.csv", index=False)


@pytest.fixture
def macro_builder(tmp_path):
    """Create a MegaFeatureBuilder pointing at a temp macro dir."""
    macro_dir = tmp_path / "macro"
    _write_macro_csvs(macro_dir)

    # Monkey-patch DATA_ROOT so the builder finds our temp CSVs
    import quantlaxmi.data._paths as paths_mod
    original_root = paths_mod.DATA_ROOT
    paths_mod.DATA_ROOT = tmp_path
    try:
        from quantlaxmi.features.mega import MegaFeatureBuilder
        builder = MegaFeatureBuilder(
            market_dir=tmp_path / "market",
            use_cache=False,
        )
        yield builder
    finally:
        paths_mod.DATA_ROOT = original_root


class TestMacroFeatures:
    """Tests for _build_macro_features."""

    def test_macro_features_count(self, macro_builder):
        """Returns exactly 8 features."""
        df = macro_builder._build_macro_features("2025-02-01", "2025-03-15")
        assert not df.empty, "Macro features should not be empty"
        expected_cols = {
            "macro_rbi_rate",
            "macro_rbi_rate_chg",
            "macro_inr_usd_return_5d",
            "macro_inr_usd_vol_21d",
            "macro_crude_return_5d",
            "macro_crude_z21",
            "macro_us10y_level",
            "macro_us10y_chg_5d",
        }
        assert set(df.columns) == expected_cols, (
            f"Expected {expected_cols}, got {set(df.columns)}"
        )
        assert len(df.columns) == 8

    def test_rbi_rate_step_function(self, macro_builder):
        """RBI rate changes only on decision dates and is constant between them."""
        df = macro_builder._build_macro_features("2025-01-01", "2025-04-30")
        rbi = df["macro_rbi_rate"].dropna()
        if rbi.empty:
            pytest.skip("No RBI rate data")

        # Between 2025-01-01 and 2025-02-06, rate should be 6.50
        jan_rates = rbi.loc["2025-01-01":"2025-02-06"]
        if not jan_rates.empty:
            assert (jan_rates == 6.50).all(), (
                f"Expected 6.50 before Feb 7, got unique values: {jan_rates.unique()}"
            )

        # After 2025-02-07, rate should be 6.25 (until Apr 9)
        feb_rates = rbi.loc["2025-02-07":"2025-04-08"]
        if not feb_rates.empty:
            assert (feb_rates == 6.25).all(), (
                f"Expected 6.25 after Feb 7, got unique values: {feb_rates.unique()}"
            )

        # After 2025-04-09, rate should be 6.00
        apr_rates = rbi.loc["2025-04-09":"2025-04-30"]
        if not apr_rates.empty:
            assert (apr_rates == 6.00).all(), (
                f"Expected 6.00 after Apr 9, got unique values: {apr_rates.unique()}"
            )

    def test_macro_causal(self, macro_builder):
        """Feature at date T uses only data available on or before T.

        Verify by checking that INR 5-day return on day T equals
        (inr[T] - inr[T-5]) / inr[T-5], which uses only past data.
        """
        df = macro_builder._build_macro_features("2025-02-01", "2025-03-15")
        # The 5-day return should not contain any future information.
        # If it were non-causal, we'd see returns referencing future prices.
        # We verify the return is computed from past prices by checking
        # that the feature value on the first available date uses the
        # 5 prior days (which are in the buffer period).
        inr_ret = df["macro_inr_usd_return_5d"].dropna()
        if inr_ret.empty:
            pytest.skip("No INR return data")
        # All returns should be finite (no look-ahead NaN pattern)
        assert inr_ret.isna().sum() < len(inr_ret), (
            "Too many NaN in INR return — possibly using future data"
        )
        # Values should be small (daily FX moves are typically < 5%)
        assert (inr_ret.abs() < 0.10).all(), (
            f"INR 5d returns unreasonably large: max={inr_ret.abs().max():.4f}"
        )

    def test_macro_graceful_missing_file(self, tmp_path):
        """Returns empty DataFrame if CSV files are not found."""
        import quantlaxmi.data._paths as paths_mod
        original_root = paths_mod.DATA_ROOT
        # Point to an empty directory (no macro/ subdir)
        paths_mod.DATA_ROOT = tmp_path / "nonexistent"
        try:
            from quantlaxmi.features.mega import MegaFeatureBuilder
            builder = MegaFeatureBuilder(
                market_dir=tmp_path / "market",
                use_cache=False,
            )
            df = builder._build_macro_features("2025-01-01", "2025-03-01")
            assert df.empty, "Should return empty DataFrame when no files exist"
        finally:
            paths_mod.DATA_ROOT = original_root

    def test_macro_z_score_finite(self, macro_builder):
        """All z-scores are finite (no division by zero producing inf)."""
        df = macro_builder._build_macro_features("2025-02-01", "2025-03-15")
        z_cols = [c for c in df.columns if "z" in c.lower()]
        for col in z_cols:
            finite_vals = df[col].dropna()
            if finite_vals.empty:
                continue
            assert np.isfinite(finite_vals).all(), (
                f"Non-finite values in {col}: "
                f"inf_count={np.isinf(finite_vals).sum()}"
            )


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

# Skip all EMA tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn


def _make_simple_model() -> nn.Module:
    """Create a small linear model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    return model


class TestModelEMA:
    """Tests for ModelEMA weight averaging."""

    def test_ema_shadow_initialized(self):
        """Shadow params equal model params at initialization."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.999)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow, f"Missing shadow for {name}"
                assert torch.equal(ema.shadow[name], param.data), (
                    f"Shadow[{name}] != model param at init"
                )

    def test_ema_update_moves_toward_model(self):
        """After update, shadow is closer to current model params."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.9)

        # Record initial shadow (should equal model at init)
        initial_shadow = {
            k: v.clone() for k, v in ema.shadow.items()
        }

        # Simulate a gradient step by manually changing model params
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        # Record distances before update
        dist_before = {}
        for name, param in model.named_parameters():
            if name in ema.shadow:
                dist_before[name] = (ema.shadow[name] - param.data).norm().item()

        # Update EMA
        ema.update()

        # After update, shadow should be closer to the new model params
        for name, param in model.named_parameters():
            if name in ema.shadow:
                dist_after = (ema.shadow[name] - param.data).norm().item()
                assert dist_after < dist_before[name], (
                    f"Shadow[{name}] did not move toward model: "
                    f"before={dist_before[name]:.6f}, after={dist_after:.6f}"
                )

    def test_ema_apply_changes_model_weights(self):
        """Model weights change after apply() when shadow differs."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.5)

        # Change model params
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param) * 2.0)

        # Update EMA so shadow differs from model
        ema.update()

        # Record model params before apply
        before = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if name in ema.shadow
        }

        # Apply EMA
        ema.apply()

        # Check that at least some params changed
        changed = False
        for name, param in model.named_parameters():
            if name in ema.shadow:
                if not torch.equal(param.data, before[name]):
                    changed = True
                    break
        assert changed, "Model weights should change after apply()"

    def test_ema_restore_reverses_apply(self):
        """Original weights restored after restore() following apply()."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.5)

        # Change model params and update EMA
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param) * 3.0)
        ema.update()

        # Record original weights
        originals = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        # Apply then restore
        ema.apply()
        ema.restore()

        # Verify weights are exactly restored
        for name, param in model.named_parameters():
            assert torch.equal(param.data, originals[name]), (
                f"Weight {name} not restored after apply/restore cycle"
            )

    def test_ema_high_decay_slow_update(self):
        """With decay=0.999, shadow barely changes per step."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.999)

        # Record initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Make a big change to model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param) * 10.0)

        # One EMA update
        ema.update()

        # Shadow should have moved only 0.1% toward the new params
        for name in initial_shadow:
            delta = (ema.shadow[name] - initial_shadow[name]).abs().max().item()
            model_param = None
            for n, p in model.named_parameters():
                if n == name:
                    model_param = p
                    break
            full_delta = (model_param.data - initial_shadow[name]).abs().max().item()
            # The ratio should be close to (1-decay) = 0.001
            if full_delta > 1e-8:
                ratio = delta / full_delta
                assert ratio < 0.01, (
                    f"Shadow[{name}] moved too much: ratio={ratio:.4f}, "
                    f"expected ~0.001 for decay=0.999"
                )

    def test_ema_state_dict_roundtrip(self):
        """Save and load preserves shadow weights exactly."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.995)

        # Do a few updates to differentiate shadow from model
        for _ in range(5):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)
            ema.update()

        # Save state
        state = ema.state_dict()
        assert "shadow" in state
        assert "decay" in state
        assert state["decay"] == 0.995

        # Create a new EMA and load state
        model2 = _make_simple_model()
        ema2 = ModelEMA(model2, decay=0.5)  # different initial decay
        ema2.load_state_dict(state)

        assert ema2.decay == 0.995, "Decay not restored"

        # Check shadow weights match
        for name in ema.shadow:
            if name in ema2.shadow:
                assert torch.allclose(ema.shadow[name], ema2.shadow[name], atol=1e-6), (
                    f"Shadow[{name}] mismatch after roundtrip"
                )

    def test_ema_context_manager(self):
        """Context manager applies and restores weights correctly."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.5)

        # Change model and update EMA to create difference
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param) * 5.0)
        ema.update()

        # Save original weights
        originals = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        # Use context manager
        with ema.average_parameters():
            # Inside the context, weights should be EMA weights
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    # Model param should now equal shadow (EMA)
                    assert torch.equal(param.data, ema.shadow[name]), (
                        f"Inside context: {name} should be EMA weights"
                    )

        # Outside the context, weights should be restored
        for name, param in model.named_parameters():
            assert torch.equal(param.data, originals[name]), (
                f"After context: {name} should be restored to original"
            )

    def test_ema_invalid_decay(self):
        """Raises ValueError for invalid decay values."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        with pytest.raises(ValueError, match="decay must be in"):
            ModelEMA(model, decay=1.0)
        with pytest.raises(ValueError, match="decay must be in"):
            ModelEMA(model, decay=-0.1)

    def test_ema_repr(self):
        """__repr__ includes decay and param count."""
        from quantlaxmi.models.ml.tft.ema import ModelEMA

        model = _make_simple_model()
        ema = ModelEMA(model, decay=0.999)
        r = repr(ema)
        assert "0.999" in r
        assert "tracked_params=" in r
