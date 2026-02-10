"""
Phase 7 — SANOS LP Determinism Tests
=====================================
15 tests across 3 classes verifying that the SANOS volatility surface fitter
produces bit-identical results across repeated runs, obeys density invariants,
and responds to input perturbations in a bounded, stable manner.
"""

import numpy as np
import pytest

from quantlaxmi.core.pricing.sanos import fit_sanos, SANOSResult, bs_call


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_market_data(n_expiries=1, n_strikes=20, seed=42):
    """Create synthetic but realistic market data for SANOS testing."""
    np.random.seed(seed)
    market_strikes = []
    market_calls = []
    atm_variances = []

    for j in range(n_expiries):
        # Strikes from 0.85 to 1.15 (normalized by forward)
        k = np.linspace(0.85, 1.15, n_strikes)
        # BS prices with some noise
        v = 0.04 * (j + 1)  # increasing variance per expiry
        c = bs_call(np.ones_like(k), k, v) + np.random.normal(0, 0.001, n_strikes).clip(0)
        c = np.maximum(c, 0.001)  # ensure positive
        market_strikes.append(k)
        market_calls.append(c)
        atm_variances.append(v)

    return market_strikes, market_calls, np.array(atm_variances)


# =========================================================================
# Class 1 — LP determinism: repeated fits must yield bit-identical outputs
# =========================================================================

class TestSANOSLPDeterminism:
    """Verify that fit_sanos is fully deterministic across repeated calls."""

    def test_single_expiry_3x_density(self):
        """Fit 3 times with single expiry — densities must match exactly."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        results = [
            fit_sanos(ms, mc, atm_variances=av)
            for _ in range(3)
        ]
        for i in range(1, 3):
            for j in range(len(ms)):
                np.testing.assert_allclose(
                    results[i].densities[j],
                    results[0].densities[j],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Run {i} density diverged from run 0 (expiry {j})",
                )

    def test_single_expiry_10x_density(self):
        """Fit 10 times with single expiry — all densities identical."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=99)
        results = [
            fit_sanos(ms, mc, atm_variances=av)
            for _ in range(10)
        ]
        for i in range(1, 10):
            for j in range(len(ms)):
                np.testing.assert_allclose(
                    results[i].densities[j],
                    results[0].densities[j],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Run {i} density diverged from run 0 (expiry {j})",
                )

    def test_multi_expiry_3x_density(self):
        """Fit 3 times with 3 expiries — densities match per expiry."""
        ms, mc, av = _make_market_data(n_expiries=3, seed=42)
        results = [
            fit_sanos(ms, mc, atm_variances=av)
            for _ in range(3)
        ]
        for i in range(1, 3):
            for j in range(3):
                np.testing.assert_allclose(
                    results[i].densities[j],
                    results[0].densities[j],
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"Run {i} density diverged from run 0 (expiry {j})",
                )

    def test_multi_expiry_fit_error(self):
        """Fit 3 times with 3 expiries — max_fit_error matches exactly."""
        ms, mc, av = _make_market_data(n_expiries=3, seed=42)
        results = [
            fit_sanos(ms, mc, atm_variances=av)
            for _ in range(3)
        ]
        for i in range(1, 3):
            np.testing.assert_allclose(
                results[i].max_fit_error,
                results[0].max_fit_error,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Run {i} max_fit_error diverged from run 0",
            )

    def test_density_sum_invariant(self):
        """Density weights sum to 1.0 for each expiry (LP constraint: Σ q_i = 1)."""
        ms, mc, av = _make_market_data(n_expiries=3, seed=42)
        result = fit_sanos(ms, mc, atm_variances=av)
        for j in range(3):
            density_sum = np.sum(result.densities[j])
            np.testing.assert_allclose(
                density_sum,
                1.0,
                atol=1e-4,
                err_msg=f"Density for expiry {j} does not sum to 1.0",
            )

    def test_density_mean_invariant(self):
        """Density-weighted mean of model_strikes approx 1.0 (martingale: K' q = 1)."""
        ms, mc, av = _make_market_data(n_expiries=2, seed=42)
        result = fit_sanos(ms, mc, atm_variances=av)
        for j in range(2):
            mean = np.dot(result.model_strikes, result.densities[j])
            np.testing.assert_allclose(
                mean,
                1.0,
                atol=1e-3,
                err_msg=f"Martingale condition violated for expiry {j}: mean={mean}",
            )

    def test_price_3x(self):
        """price() on same strikes gives identical results across 3 fits."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        query_strikes = np.linspace(0.90, 1.10, 50)
        results = [
            fit_sanos(ms, mc, atm_variances=av)
            for _ in range(3)
        ]
        prices_0 = results[0].price(0, query_strikes)
        for i in range(1, 3):
            prices_i = results[i].price(0, query_strikes)
            np.testing.assert_allclose(
                prices_i,
                prices_0,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Run {i} prices diverged from run 0",
            )

    def test_iv_3x(self):
        """iv() on same strikes gives identical results across 3 fits."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        query_strikes = np.linspace(0.92, 1.08, 30)
        T = 30.0 / 365.0  # 30 days to expiry
        results = [
            fit_sanos(ms, mc, atm_variances=av)
            for _ in range(3)
        ]
        iv_0 = results[0].iv(0, query_strikes, T)
        for i in range(1, 3):
            iv_i = results[i].iv(0, query_strikes, T)
            np.testing.assert_allclose(
                iv_i,
                iv_0,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Run {i} IVs diverged from run 0",
            )


# =========================================================================
# Class 2 — Input sensitivity: small perturbations produce bounded changes
# =========================================================================

class TestSANOSInputSensitivity:
    """Verify that SANOS output changes are bounded under small input perturbations."""

    def test_price_perturbation_bounded(self):
        """Perturb one market call by 1% — output prices change by < 10%."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        result_base = fit_sanos(ms, mc, atm_variances=av)

        mc_perturbed = [c.copy() for c in mc]
        mid_idx = len(mc_perturbed[0]) // 2
        mc_perturbed[0][mid_idx] *= 1.01  # +1% perturbation

        result_pert = fit_sanos(ms, mc_perturbed, atm_variances=av)

        query_strikes = np.linspace(0.90, 1.10, 50)
        prices_base = result_base.price(0, query_strikes)
        prices_pert = result_pert.price(0, query_strikes)

        # Relative change should be bounded
        rel_change = np.abs(prices_pert - prices_base) / np.maximum(np.abs(prices_base), 1e-8)
        assert np.all(rel_change < 0.10), (
            f"Price change exceeded 10% bound: max relative change = {rel_change.max():.4f}"
        )

    def test_strike_perturbation_bounded(self):
        """Perturb one market strike by 0.1% — output density change is bounded."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        result_base = fit_sanos(ms, mc, atm_variances=av)

        ms_perturbed = [k.copy() for k in ms]
        mid_idx = len(ms_perturbed[0]) // 2
        ms_perturbed[0][mid_idx] *= 1.001  # +0.1% perturbation

        result_pert = fit_sanos(ms_perturbed, mc, atm_variances=av)

        for j in range(len(ms)):
            diff = np.abs(result_pert.densities[j] - result_base.densities[j])
            max_diff = diff.max()
            base_max = np.abs(result_base.densities[j]).max()
            # Density change should not exceed 2x the base max
            assert max_diff <= 2.0 * base_max, (
                f"Density change exceeded 2x base max: {max_diff:.6f} > {2.0 * base_max:.6f}"
            )

    def test_variance_perturbation_bounded(self):
        """Perturb atm_variance by 1% — output densities change by bounded amount."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        result_base = fit_sanos(ms, mc, atm_variances=av)

        av_perturbed = av.copy()
        av_perturbed[0] *= 1.01  # +1% perturbation

        result_pert = fit_sanos(ms, mc, atm_variances=av_perturbed)

        for j in range(len(ms)):
            diff = np.abs(result_pert.densities[j] - result_base.densities[j])
            max_diff = diff.max()
            base_max = np.abs(result_base.densities[j]).max()
            assert max_diff < base_max, (
                f"Density change exceeded base max: {max_diff:.6f} >= {base_max:.6f}"
            )

    def test_eta_perturbation_bounded(self):
        """Change eta from 0.50 to 0.51 — output prices change by < 5%."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        result_base = fit_sanos(ms, mc, atm_variances=av, eta=0.50)
        result_pert = fit_sanos(ms, mc, atm_variances=av, eta=0.51)

        query_strikes = np.linspace(0.90, 1.10, 50)
        prices_base = result_base.price(0, query_strikes)
        prices_pert = result_pert.price(0, query_strikes)

        rel_change = np.abs(prices_pert - prices_base) / np.maximum(np.abs(prices_base), 1e-8)
        assert np.all(rel_change < 0.05), (
            f"Price change exceeded 5% bound: max relative change = {rel_change.max():.4f}"
        )


# =========================================================================
# Class 3 — Deterministic flag on SANOSResult
# =========================================================================

class TestSANOSDeterministicFlag:
    """Verify the deterministic flag on SANOSResult is correctly managed."""

    def test_default_true(self):
        """SANOSResult.deterministic defaults to True when constructed."""
        result = SANOSResult(
            densities=[np.array([1.0])],
            model_strikes=np.linspace(0.7, 1.5, 100),
            variances=np.array([0.04]),
            eta=0.50,
            expiry_labels=None,
            market_strikes=[np.array([1.0])],
            market_mids=[np.array([0.05])],
            fit_errors=[np.array([0.0])],
            max_fit_error=0.0,
            lp_success=True,
        )
        assert result.deterministic is True

    def test_result_carries_flag(self):
        """After fit_sanos, result.deterministic is True (LP succeeded)."""
        ms, mc, av = _make_market_data(n_expiries=1, seed=42)
        result = fit_sanos(ms, mc, atm_variances=av)
        assert result.lp_success is True
        assert result.deterministic is True

    def test_non_deterministic_flag_preserved(self):
        """SANOSResult constructed with deterministic=False preserves the flag."""
        result = SANOSResult(
            densities=[np.array([1.0])],
            model_strikes=np.linspace(0.7, 1.5, 100),
            variances=np.array([0.04]),
            eta=0.50,
            expiry_labels=None,
            market_strikes=[np.array([1.0])],
            market_mids=[np.array([0.05])],
            fit_errors=[np.array([0.0])],
            max_fit_error=0.0,
            lp_success=False,
            deterministic=False,
        )
        assert result.deterministic is False
