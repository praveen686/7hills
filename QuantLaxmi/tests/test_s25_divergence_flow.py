"""Comprehensive tests for S25 Divergence Flow Field (DFF) strategy.

Tests cover:
- Conservation law verification (4 tests)
- Helmholtz decomposition properties (5 tests)
- Feature builder output (6 tests)
- DFFConfig behaviour (3 tests)
- S25DFFStrategy identity & lifecycle (5 tests)
- Signal generation (4 tests)
- Edge cases (3 tests)

Total: 30 tests.  All use synthetic data — NO DuckDB dependency.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from features.divergence_flow import DFFConfig, DivergenceFlowBuilder
from strategies.protocol import Signal
from strategies.s25_divergence_flow.strategy import S25DFFStrategy, create_strategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INSTRUMENT_LONG_SHORT = {
    "fut_idx": ("Future Index Long", "Future Index Short"),
    "fut_stk": ("Future Stock Long", "Future Stock Short"),
    "opt_idx_call": ("Option Index Call Long", "Option Index Call Short"),
    "opt_idx_put": ("Option Index Put Long", "Option Index Put Short"),
    "opt_stk_call": ("Option Stock Call Long", "Option Stock Call Short"),
    "opt_stk_put": ("Option Stock Put Long", "Option Stock Put Short"),
}

_PARTICIPANTS = ["FII", "DII", "CLIENT", "PRO"]


@pytest.fixture
def synthetic_participant_oi() -> pd.DataFrame:
    """Generate 100 business days of synthetic participant OI data.

    Conservation law is enforced: for each instrument class on each date,
    PRO = -(FII + DII + CLIENT) so that the four participants sum to zero.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2025-06-01", periods=100)

    rows: list[dict] = []
    for d in dates:
        date_rows: list[dict] = []
        for participant in _PARTICIPANTS:
            row: dict = {"date": d, "Client Type": participant}
            for ic, (long_col, short_col) in _INSTRUMENT_LONG_SHORT.items():
                if participant == "PRO":
                    # PRO = -(FII + DII + CLIENT) for this instrument
                    other_nets = [
                        prev[long_col] - prev[short_col]
                        for prev in date_rows  # only FII, DII, CLIENT so far
                    ]
                    net = -sum(other_nets)
                else:
                    net = np.random.randint(-50_000, 50_000)

                offset = np.random.randint(100_000, 500_000)
                row[long_col] = max(0, net) + offset
                row[short_col] = max(0, -net) + offset

            row["Total Long Contracts"] = sum(
                row[v[0]] for v in _INSTRUMENT_LONG_SHORT.values()
            )
            row["Total Short Contracts"] = sum(
                row[v[1]] for v in _INSTRUMENT_LONG_SHORT.values()
            )
            date_rows.append(row)
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def builder() -> DivergenceFlowBuilder:
    return DivergenceFlowBuilder()


@pytest.fixture
def net_positions(synthetic_participant_oi, builder):
    """Pre-computed net positions from synthetic data."""
    return builder._compute_net_positions(synthetic_participant_oi)


@pytest.fixture
def flows(net_positions, builder):
    """Pre-computed flows from net positions."""
    return builder._compute_flows(net_positions)


@pytest.fixture
def decomposition(flows, builder):
    """Pre-computed Helmholtz decomposition."""
    return builder._helmholtz_decompose(flows)


@pytest.fixture
def features_df(synthetic_participant_oi) -> pd.DataFrame:
    """Full feature DataFrame built from synthetic data."""
    return DivergenceFlowBuilder.build_from_dataframe(
        synthetic_participant_oi,
        start_date="2025-07-15",
        end_date="2025-10-17",
    )


# ---------------------------------------------------------------------------
# TestConservationLaw (4 tests)
# ---------------------------------------------------------------------------


class TestConservationLaw:
    """Verify the zero-sum conservation constraint on participant positions."""

    def test_conservation_law_holds(self, net_positions, builder):
        """Sum of net positions across all 4 participants should be ~0
        for every instrument class on every date."""
        check = builder.verify_conservation(net_positions, tolerance=1e-6)
        max_violation = check.abs().max().max()
        assert max_violation < 1e-6, f"Conservation violated: max |sum| = {max_violation}"

    def test_conservation_violation_logged(
        self, net_positions, builder, caplog
    ):
        """When data violates conservation, a warning should be logged."""
        # Perturb one cell to break the zero-sum
        perturbed = net_positions.copy()
        first_col = perturbed.columns[0]
        perturbed.iloc[0, 0] += 1e6  # large perturbation

        with caplog.at_level(logging.WARNING, logger="features.divergence_flow"):
            check = builder.verify_conservation(perturbed, tolerance=1e-6)

        assert any("Conservation violation" in m for m in caplog.messages), (
            "Expected warning about conservation violation"
        )
        assert check.abs().max().max() > 1e-6

    def test_flow_conservation(self, flows, builder):
        """Daily flows (diffs) should also satisfy conservation across participants."""
        # Re-derive net positions from flows is not directly possible,
        # but we can check that flows summed across participants per ic = 0.
        from features.divergence_flow import _INSTRUMENT_CLASSES, _PARTICIPANTS

        for ic in _INSTRUMENT_CLASSES:
            cols = [
                (p, ic) for p in _PARTICIPANTS if (p, ic) in flows.columns
            ]
            if cols:
                flow_sum = flows[cols].sum(axis=1)
                max_err = flow_sum.abs().max()
                assert max_err < 1e-6, (
                    f"Flow conservation violated for {ic}: max |sum| = {max_err}"
                )

    def test_informed_equals_negative_uninformed(self, flows):
        """For each instrument class, the informed aggregate flow should
        equal the negative of the uninformed aggregate flow, since all
        four participants sum to zero and informed = FII+PRO,
        uninformed = CLIENT+DII.

        D(t) per-ic = (informed - uninformed).  Since informed + uninformed = 0,
        informed = -uninformed, so D(t) = 2*informed.
        """
        from features.divergence_flow import (
            _INFORMED,
            _INSTRUMENT_CLASSES,
            _UNINFORMED,
        )

        for ic in _INSTRUMENT_CLASSES:
            informed = pd.Series(0.0, index=flows.index)
            uninformed = pd.Series(0.0, index=flows.index)
            for p in _INFORMED:
                if (p, ic) in flows.columns:
                    informed += flows[(p, ic)]
            for p in _UNINFORMED:
                if (p, ic) in flows.columns:
                    uninformed += flows[(p, ic)]
            diff = (informed + uninformed).abs().max()
            assert diff < 1e-6, (
                f"Informed + Uninformed != 0 for {ic}: max = {diff}"
            )


# ---------------------------------------------------------------------------
# TestHelmholtzDecomposition (5 tests)
# ---------------------------------------------------------------------------


class TestHelmholtzDecomposition:
    """Verify Helmholtz decomposition properties."""

    def test_divergence_sign_matches_informed_flow(
        self, synthetic_participant_oi
    ):
        """When FII+PRO are net buyers (all instruments), D should be
        positive on that date."""
        # Build controlled data: FII and PRO strongly long on day 1,
        # CLIENT and DII absorbing.
        np.random.seed(99)
        dates = pd.bdate_range("2025-08-01", periods=5)
        rows = []
        for d_idx, d in enumerate(dates):
            date_rows = []
            for p in _PARTICIPANTS:
                row = {"date": d, "Client Type": p}
                for ic, (lc, sc) in _INSTRUMENT_LONG_SHORT.items():
                    if p == "FII":
                        # FII is strongly long, increasing each day
                        net = 10_000 * (d_idx + 1)
                    elif p == "PRO":
                        # PRO also long
                        net_prev = [
                            prev[lc] - prev[sc] for prev in date_rows
                        ]
                        net = -sum(net_prev) + 5_000 * (d_idx + 1)
                    elif p == "DII":
                        net = -5_000 * (d_idx + 1)
                    else:  # CLIENT — absorbs remainder
                        net_prev = [
                            prev[lc] - prev[sc] for prev in date_rows
                        ]
                        net = -sum(net_prev)

                    offset = 200_000
                    row[lc] = max(0, net) + offset
                    row[sc] = max(0, -net) + offset
                row["Total Long Contracts"] = sum(
                    row[v[0]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                row["Total Short Contracts"] = sum(
                    row[v[1]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                date_rows.append(row)
                rows.append(row)

        df = pd.DataFrame(rows)
        builder = DivergenceFlowBuilder()
        net = builder._compute_net_positions(df)
        flows = builder._compute_flows(net)
        decomp = builder._helmholtz_decompose(flows)

        # After the first day (diff = 0), D should be > 0 because informed
        # (FII+PRO) are increasing their long positioning
        d_values = decomp["D"].iloc[1:]  # skip first (diff row is 0)
        assert (d_values > 0).all(), (
            f"Expected positive D when informed buys, got: {d_values.values}"
        )

    def test_rotation_detects_futures_preference(self):
        """When informed flow is concentrated in index futures (not options
        or stock derivatives), R should be positive."""
        np.random.seed(123)
        dates = pd.bdate_range("2025-09-01", periods=5)
        rows = []
        for d_idx, d in enumerate(dates):
            date_rows = []
            for p in _PARTICIPANTS:
                row = {"date": d, "Client Type": p}
                for ic, (lc, sc) in _INSTRUMENT_LONG_SHORT.items():
                    if p == "FII" and ic == "fut_idx":
                        net = 50_000 * (d_idx + 1)  # big in index futures
                    elif p == "FII":
                        net = 100  # tiny elsewhere
                    elif p == "PRO":
                        net_prev = [
                            prev[lc] - prev[sc] for prev in date_rows
                        ]
                        net = -sum(net_prev)
                    elif p == "DII":
                        net = -100
                    else:  # CLIENT
                        net_prev = [
                            prev[lc] - prev[sc] for prev in date_rows
                        ]
                        net = -sum(net_prev)

                    offset = 200_000
                    row[lc] = max(0, net) + offset
                    row[sc] = max(0, -net) + offset
                row["Total Long Contracts"] = sum(
                    row[v[0]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                row["Total Short Contracts"] = sum(
                    row[v[1]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                date_rows.append(row)
                rows.append(row)

        df = pd.DataFrame(rows)
        builder = DivergenceFlowBuilder()
        net = builder._compute_net_positions(df)
        flows = builder._compute_flows(net)
        decomp = builder._helmholtz_decompose(flows)

        # R should be positive when informed flow is in index futures
        r_values = decomp["R"].iloc[1:]
        assert (r_values > 0).all(), (
            f"Expected positive R for futures preference, got: {r_values.values}"
        )

    def test_energy_is_nonnegative(self, decomposition):
        """Energy E(t) = sum of absolute flows, must be >= 0."""
        assert (decomposition["E"] >= 0).all(), "E(t) must be non-negative"

    def test_normalized_produces_finite_values(self, decomposition):
        """d_hat and r_hat should be finite (no inf, no NaN after warmup)."""
        d_hat = decomposition["d_hat"]
        r_hat = decomposition["r_hat"]
        assert np.isfinite(d_hat).all(), "d_hat has non-finite values"
        assert np.isfinite(r_hat).all(), "r_hat has non-finite values"

    def test_zero_flow_produces_zero_divergence(self):
        """When all flows are zero (no change in positions), D=R=0."""
        # Create constant positions (no change day-to-day)
        dates = pd.bdate_range("2025-09-01", periods=5)
        rows = []
        for d in dates:
            date_rows = []
            for p in _PARTICIPANTS:
                row = {"date": d, "Client Type": p}
                for ic, (lc, sc) in _INSTRUMENT_LONG_SHORT.items():
                    if p == "PRO":
                        net_prev = [
                            prev[lc] - prev[sc] for prev in date_rows
                        ]
                        net = -sum(net_prev)
                    else:
                        net = 10_000  # constant across days
                    offset = 200_000
                    row[lc] = max(0, net) + offset
                    row[sc] = max(0, -net) + offset
                row["Total Long Contracts"] = sum(
                    row[v[0]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                row["Total Short Contracts"] = sum(
                    row[v[1]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                date_rows.append(row)
                rows.append(row)

        df = pd.DataFrame(rows)
        builder = DivergenceFlowBuilder()
        net = builder._compute_net_positions(df)
        flows = builder._compute_flows(net)
        decomp = builder._helmholtz_decompose(flows)

        # Flows are all zero after first row (diff of constant = 0)
        d_vals = decomp["D"].iloc[1:]
        r_vals = decomp["R"].iloc[1:]
        assert (d_vals.abs() < 1e-10).all(), f"D != 0 with zero flows: {d_vals.values}"
        assert (r_vals.abs() < 1e-10).all(), f"R != 0 with zero flows: {r_vals.values}"


# ---------------------------------------------------------------------------
# TestFeatureBuilder (6 tests)
# ---------------------------------------------------------------------------


class TestFeatureBuilder:
    """Verify DivergenceFlowBuilder produces correct feature DataFrame."""

    def test_build_from_dataframe_produces_12_features(self, features_df):
        """build_from_dataframe should return exactly 12 columns."""
        assert len(features_df.columns) == 12, (
            f"Expected 12 features, got {len(features_df.columns)}: "
            f"{features_df.columns.tolist()}"
        )

    def test_feature_names_all_prefixed(self, features_df):
        """Every column name should start with 'dff_'."""
        for col in features_df.columns:
            assert col.startswith("dff_"), f"Column {col!r} lacks 'dff_' prefix"

    def test_zscore_clipped(self, features_df):
        """Z-score columns must be clipped to [-4, 4]."""
        for col in ("dff_z_d", "dff_z_r", "dff_energy_z"):
            series = features_df[col].dropna()
            if series.empty:
                continue
            assert series.min() >= -4.0 - 1e-12, (
                f"{col} min = {series.min()}, expected >= -4"
            )
            assert series.max() <= 4.0 + 1e-12, (
                f"{col} max = {series.max()}, expected <= 4"
            )

    def test_composite_formula(self, features_df):
        """Verify composite = alpha * z_d + beta * z_r + gamma * z_d * z_r."""
        cfg = DFFConfig()
        z_d = features_df["dff_z_d"]
        z_r = features_df["dff_z_r"]
        expected = cfg.alpha * z_d + cfg.beta * z_r + cfg.gamma * z_d * z_r

        # Compare only rows where both z-scores are non-NaN
        valid = expected.notna() & features_df["dff_composite"].notna()
        pd.testing.assert_series_equal(
            features_df.loc[valid, "dff_composite"],
            expected.loc[valid],
            check_names=False,
            atol=1e-12,
        )

    def test_regime_values(self, features_df):
        """Regime indicator should only contain values in {0, 1, 2, 3, 4}."""
        regimes = features_df["dff_regime"].dropna().unique()
        allowed = {0, 1, 2, 3, 4}
        for val in regimes:
            assert int(val) in allowed, f"Unexpected regime value: {val}"

    def test_momentum_lag(self, features_df):
        """dff_momentum = d_hat(t) - d_hat(t-5)."""
        d_hat = features_df["dff_d_hat"]
        expected_momentum = d_hat - d_hat.shift(5)
        valid = expected_momentum.notna() & features_df["dff_momentum"].notna()
        pd.testing.assert_series_equal(
            features_df.loc[valid, "dff_momentum"],
            expected_momentum.loc[valid],
            check_names=False,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# TestDFFConfig (3 tests)
# ---------------------------------------------------------------------------


class TestDFFConfig:
    """Verify DFFConfig defaults and overrides."""

    def test_default_weights_sum_reasonably(self):
        """Default delta-equivalent weights should have reasonable magnitudes.
        Sum of absolute weights should be > 0 (they are used as scalars)."""
        cfg = DFFConfig()
        abs_sum = (
            abs(cfg.w_fut_idx)
            + abs(cfg.w_fut_stk)
            + abs(cfg.w_opt_idx_call)
            + abs(cfg.w_opt_idx_put)
            + abs(cfg.w_opt_stk_call)
            + abs(cfg.w_opt_stk_put)
        )
        assert abs_sum > 0, "Sum of |weights| should be positive"
        # Also check composite weights sum to 1.0
        assert abs(cfg.alpha + cfg.beta + cfg.gamma - 1.0) < 1e-12, (
            "Composite signal weights alpha+beta+gamma should equal 1.0"
        )

    def test_custom_config(self):
        """All parameters can be overridden."""
        cfg = DFFConfig(
            w_fut_idx=2.0,
            w_fut_stk=1.0,
            w_opt_idx_call=0.5,
            w_opt_idx_put=-0.5,
            w_opt_stk_call=0.3,
            w_opt_stk_put=-0.3,
            zscore_window=42,
            ema_span=10,
            momentum_lag=10,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            energy_floor=0.01,
        )
        assert cfg.zscore_window == 42
        assert cfg.ema_span == 10
        assert cfg.momentum_lag == 10
        assert cfg.alpha == 0.5
        assert cfg.energy_floor == 0.01

    def test_energy_floor_prevents_divzero(self):
        """With extremely tiny flows, d_hat and r_hat must stay finite
        because E is clamped to the energy floor."""
        cfg = DFFConfig(energy_floor=1.0)
        builder = DivergenceFlowBuilder(config=cfg)

        # Create data with tiny OI changes
        dates = pd.bdate_range("2025-09-01", periods=10)
        rows = []
        for d_idx, d in enumerate(dates):
            date_rows = []
            for p in _PARTICIPANTS:
                row = {"date": d, "Client Type": p}
                for ic, (lc, sc) in _INSTRUMENT_LONG_SHORT.items():
                    if p == "PRO":
                        net_prev = [
                            prev[lc] - prev[sc] for prev in date_rows
                        ]
                        net = -sum(net_prev)
                    else:
                        # Tiny random fluctuation around a fixed level
                        net = 1_000_000 + np.random.randint(-1, 2)
                    offset = 2_000_000
                    row[lc] = max(0, net) + offset
                    row[sc] = max(0, -net) + offset
                row["Total Long Contracts"] = sum(
                    row[v[0]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                row["Total Short Contracts"] = sum(
                    row[v[1]] for v in _INSTRUMENT_LONG_SHORT.values()
                )
                date_rows.append(row)
                rows.append(row)

        df = pd.DataFrame(rows)
        net = builder._compute_net_positions(df)
        flows = builder._compute_flows(net)
        decomp = builder._helmholtz_decompose(flows)

        assert np.isfinite(decomp["d_hat"]).all(), "d_hat not finite with tiny flows"
        assert np.isfinite(decomp["r_hat"]).all(), "r_hat not finite with tiny flows"


# ---------------------------------------------------------------------------
# TestS25Strategy (5 tests)
# ---------------------------------------------------------------------------


class TestS25Strategy:
    """Verify S25DFFStrategy identity, factory, and lifecycle."""

    def test_strategy_id(self, tmp_path):
        strat = S25DFFStrategy(state_dir=tmp_path)
        assert strat.strategy_id == "s25_dff"

    def test_warmup_days(self, tmp_path):
        strat = S25DFFStrategy(state_dir=tmp_path)
        assert strat.warmup_days() == 30

    def test_create_strategy_factory(self, tmp_path):
        """create_strategy() should return an S25DFFStrategy instance."""
        with patch.object(S25DFFStrategy, "__init__", return_value=None) as mock_init:
            # Use the real factory but verify the type
            pass
        strat = create_strategy()
        assert isinstance(strat, S25DFFStrategy)
        assert strat.strategy_id == "s25_dff"

    def test_signal_direction_matches_composite(
        self, synthetic_participant_oi, tmp_path
    ):
        """Positive composite -> long, negative composite -> short."""
        features = DivergenceFlowBuilder.build_from_dataframe(
            synthetic_participant_oi,
            start_date="2025-07-15",
            end_date="2025-10-17",
        )
        if features.empty:
            pytest.skip("No features produced")

        strat = S25DFFStrategy(
            entry_threshold=0.0001,  # very low so most signals pass
            state_dir=tmp_path,
        )

        # Pick a date where composite is clearly positive or negative
        for idx in features.index:
            composite = features.loc[idx, "dff_composite"]
            if pd.notna(composite) and abs(composite) > 0.01:
                break
        else:
            pytest.skip("No date with sufficiently large composite")

        latest = features.loc[idx]
        composite = latest["dff_composite"]

        # Directly test _scan_symbol
        scan_date = idx.date() + timedelta(days=1)
        sig = strat._scan_symbol(scan_date, "NIFTY", composite, latest)

        assert sig is not None, "Expected a signal"
        if composite > 0:
            assert sig.direction == "long", (
                f"composite={composite:.4f} but direction={sig.direction}"
            )
        else:
            assert sig.direction == "short", (
                f"composite={composite:.4f} but direction={sig.direction}"
            )

    def test_entry_threshold_filters_weak_signals(self, tmp_path):
        """Signals with |composite| < entry_threshold should be filtered out."""
        strat = S25DFFStrategy(
            entry_threshold=0.5,
            state_dir=tmp_path,
        )

        # Create a mock latest series with weak composite
        latest = pd.Series({
            "dff_composite": 0.2,  # below threshold of 0.5
            "dff_d_hat": 0.01,
            "dff_r_hat": 0.01,
            "dff_z_d": 0.1,
            "dff_z_r": 0.1,
            "dff_interaction": 0.01,
            "dff_energy": 10.0,
            "dff_regime": 1,
            "dff_momentum": 0.001,
        })

        sig = strat._scan_symbol(date(2025, 9, 1), "NIFTY", 0.2, latest)
        assert sig is None, "Weak signal should be filtered out"


# ---------------------------------------------------------------------------
# TestSignalGeneration (4 tests)
# ---------------------------------------------------------------------------


class TestSignalGeneration:
    """Verify signal properties and state transitions."""

    def test_conviction_bounded(self, tmp_path):
        """Conviction should be in [0, 1]."""
        strat = S25DFFStrategy(
            entry_threshold=0.1,
            signal_scale=1.0,
            max_conviction=0.8,
            state_dir=tmp_path,
        )

        # Very large composite — conviction should still be capped
        latest = pd.Series({
            "dff_composite": 5.0,
            "dff_d_hat": 0.5,
            "dff_r_hat": 0.1,
            "dff_z_d": 3.5,
            "dff_z_r": 1.0,
            "dff_interaction": 3.5,
            "dff_energy": 15.0,
            "dff_regime": 1,
            "dff_momentum": 0.1,
        })

        sig = strat._scan_symbol(date(2025, 9, 1), "NIFTY", 5.0, latest)
        assert sig is not None
        assert 0.0 <= sig.conviction <= 1.0, (
            f"Conviction {sig.conviction} out of [0,1]"
        )
        assert sig.conviction <= 0.8, (
            f"Conviction {sig.conviction} exceeds max_conviction=0.8"
        )

    def test_signal_metadata_contains_dff_fields(self, tmp_path):
        """Signal metadata should include composite, d_hat, r_hat, etc."""
        strat = S25DFFStrategy(
            entry_threshold=0.1,
            state_dir=tmp_path,
        )

        latest = pd.Series({
            "dff_composite": 1.5,
            "dff_d_hat": 0.3,
            "dff_r_hat": -0.1,
            "dff_z_d": 2.0,
            "dff_z_r": -0.5,
            "dff_interaction": -1.0,
            "dff_energy": 12.5,
            "dff_regime": 2,
            "dff_momentum": 0.05,
        })

        sig = strat._scan_symbol(date(2025, 9, 1), "BANKNIFTY", 1.5, latest)
        assert sig is not None
        expected_keys = {
            "composite", "d_hat", "r_hat", "z_d", "z_r",
            "interaction", "energy", "regime", "momentum",
        }
        assert expected_keys <= set(sig.metadata.keys()), (
            f"Missing metadata keys: {expected_keys - set(sig.metadata.keys())}"
        )

    def test_max_hold_exit(self, tmp_path):
        """After max_hold_days, strategy should emit a flat exit signal."""
        strat = S25DFFStrategy(
            entry_threshold=0.1,
            max_hold_days=5,
            state_dir=tmp_path,
        )

        # Simulate an existing position entered 6 days ago
        entry_date = date(2025, 9, 1)
        scan_date = entry_date + timedelta(days=6)  # > max_hold_days=5
        strat.set_state("position_NIFTY", {
            "entry_date": entry_date.isoformat(),
            "direction": "long",
            "entry_signal": 1.0,
        })

        latest = pd.Series({
            "dff_composite": 1.0,  # still strong, but held too long
            "dff_d_hat": 0.3,
            "dff_r_hat": 0.1,
            "dff_z_d": 1.5,
            "dff_z_r": 0.5,
            "dff_interaction": 0.75,
            "dff_energy": 11.0,
            "dff_regime": 1,
            "dff_momentum": 0.05,
        })

        sig = strat._scan_symbol(scan_date, "NIFTY", 1.0, latest)
        assert sig is not None, "Expected a flat exit signal"
        assert sig.direction == "flat"
        assert sig.metadata.get("exit_reason") == "max_hold"
        # State should be cleared
        assert strat.get_state("position_NIFTY") is None

    def test_exit_on_signal_decay(self, tmp_path):
        """When |composite| drops below exit_threshold, emit flat signal."""
        strat = S25DFFStrategy(
            entry_threshold=0.5,
            exit_threshold=0.3,
            max_hold_days=10,
            state_dir=tmp_path,
        )

        # Position entered 2 days ago with strong signal
        entry_date = date(2025, 9, 1)
        strat.set_state("position_NIFTY", {
            "entry_date": entry_date.isoformat(),
            "direction": "long",
            "entry_signal": 1.2,
        })

        # Now composite has decayed below exit threshold
        scan_date = entry_date + timedelta(days=2)
        latest = pd.Series({
            "dff_composite": 0.1,  # below exit_threshold=0.3
            "dff_d_hat": 0.01,
            "dff_r_hat": 0.005,
            "dff_z_d": 0.05,
            "dff_z_r": 0.02,
            "dff_interaction": 0.001,
            "dff_energy": 8.0,
            "dff_regime": 0,
            "dff_momentum": -0.01,
        })

        sig = strat._scan_symbol(scan_date, "NIFTY", 0.1, latest)
        assert sig is not None, "Expected a flat exit on signal decay"
        assert sig.direction == "flat"
        assert sig.metadata.get("exit_reason") == "signal_decay"
        assert strat.get_state("position_NIFTY") is None


# ---------------------------------------------------------------------------
# TestEdgeCases (3 tests)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty data, single day, NaN handling."""

    def test_empty_data_returns_empty_features(self):
        """Empty DataFrame should not crash and returns empty features."""
        empty_df = pd.DataFrame(columns=[
            "date", "Client Type",
            "Future Index Long", "Future Index Short",
            "Future Stock Long", "Future Stock Short",
            "Option Index Call Long", "Option Index Call Short",
            "Option Index Put Long", "Option Index Put Short",
            "Option Stock Call Long", "Option Stock Call Short",
            "Option Stock Put Long", "Option Stock Put Short",
            "Total Long Contracts", "Total Short Contracts",
        ])
        result = DivergenceFlowBuilder.build_from_dataframe(
            empty_df, "2025-08-01", "2025-08-31"
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_day_no_crash(self):
        """A single day of data should not crash (flows will be zero)."""
        dates = pd.bdate_range("2025-09-01", periods=1)
        rows = []
        date_rows = []
        for p in _PARTICIPANTS:
            row = {"date": dates[0], "Client Type": p}
            for ic, (lc, sc) in _INSTRUMENT_LONG_SHORT.items():
                if p == "PRO":
                    net_prev = [prev[lc] - prev[sc] for prev in date_rows]
                    net = -sum(net_prev)
                else:
                    net = np.random.randint(-10_000, 10_000)
                offset = 200_000
                row[lc] = max(0, net) + offset
                row[sc] = max(0, -net) + offset
            row["Total Long Contracts"] = sum(
                row[v[0]] for v in _INSTRUMENT_LONG_SHORT.values()
            )
            row["Total Short Contracts"] = sum(
                row[v[1]] for v in _INSTRUMENT_LONG_SHORT.values()
            )
            date_rows.append(row)
            rows.append(row)

        df = pd.DataFrame(rows)
        result = DivergenceFlowBuilder.build_from_dataframe(
            df, "2025-09-01", "2025-09-01"
        )
        assert isinstance(result, pd.DataFrame)
        # Single day may produce a row with many NaNs (z-scores need min_periods),
        # but it should NOT crash.

    def test_nan_handling(self, tmp_path):
        """NaN in features should not propagate into Signal objects.
        The strategy should skip producing a signal when composite is NaN."""
        strat = S25DFFStrategy(
            entry_threshold=0.1,
            state_dir=tmp_path,
        )

        latest = pd.Series({
            "dff_composite": np.nan,
            "dff_d_hat": np.nan,
            "dff_r_hat": np.nan,
            "dff_z_d": np.nan,
            "dff_z_r": np.nan,
            "dff_interaction": np.nan,
            "dff_energy": np.nan,
            "dff_regime": 0,
            "dff_momentum": np.nan,
        })

        # Mock the builder.build to return a DataFrame with NaN composite
        features_with_nan = pd.DataFrame(
            {"dff_composite": [np.nan]},
            index=pd.DatetimeIndex(["2025-08-31"]),
        )

        with patch.object(
            DivergenceFlowBuilder, "build", return_value=features_with_nan
        ):
            mock_store = MagicMock()
            signals = strat._scan_impl(date(2025, 9, 1), mock_store)

        assert signals == [], "NaN composite should produce no signals"
