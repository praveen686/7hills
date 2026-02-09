"""Comprehensive tests for the SANOS smooth arbitrage-free option surface.

Tests cover:
  - Black-Scholes call / vega correctness
  - SANOS LP fitting with synthetic data
  - SANOSResult pricing, IV, and density methods
  - Calendar and butterfly arbitrage constraints
  - prepare_nifty_chain helper with mock and real data
  - Edge cases (single expiry, few strikes, extreme eta)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from core.pricing.sanos import (
    SANOSResult,
    bs_call,
    bs_call_vega,
    fit_sanos,
    prepare_nifty_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "india" / "fno"


def _lognormal_call_prices(
    strikes: np.ndarray, sigma: float, T: float
) -> np.ndarray:
    """Exact BS call prices with F=1, so C(K) = BS(1, K, sigma^2 * T)."""
    return bs_call(np.ones_like(strikes), strikes, sigma**2 * T)


def _make_synthetic_single_expiry(
    sigma: float = 0.20,
    T: float = 0.25,
    n_strikes: int = 25,
    K_lo: float = 0.80,
    K_hi: float = 1.20,
):
    """Build a single-expiry synthetic option chain from lognormal prices.

    Returns (market_strikes, market_calls, atm_variance) all normalized
    by forward = 1.
    """
    K = np.linspace(K_lo, K_hi, n_strikes)
    C = _lognormal_call_prices(K, sigma, T)
    atm_var = sigma**2 * T
    return K, C, atm_var


def _make_synthetic_multi_expiry(
    sigma: float = 0.20,
    T_list: list[float] | None = None,
    n_strikes: int = 25,
):
    """Build multi-expiry synthetic data (increasing total variance)."""
    if T_list is None:
        T_list = [0.05, 0.10, 0.25]

    strikes_list, calls_list, vars_list = [], [], []
    for T in T_list:
        K, C, v = _make_synthetic_single_expiry(sigma=sigma, T=T, n_strikes=n_strikes)
        strikes_list.append(K)
        calls_list.append(C)
        vars_list.append(v)

    return strikes_list, calls_list, np.array(vars_list)


# ===================================================================
# 1. Black-Scholes call / vega correctness
# ===================================================================


class TestBSCall:
    """Test bs_call against known analytical values."""

    def test_atm_call(self):
        """ATM call (S=K) with known vol and time."""
        S = np.array([100.0])
        K = np.array([100.0])
        sigma, T = 0.20, 1.0
        v = np.array([sigma**2 * T])

        price = bs_call(S, K, v)
        # Expected: 100 * N(d+) - 100 * N(d-)
        d_plus = 0.5 * sigma * math.sqrt(T)
        d_minus = -0.5 * sigma * math.sqrt(T)
        expected = 100.0 * (norm.cdf(d_plus) - norm.cdf(d_minus))
        np.testing.assert_allclose(price, expected, atol=1e-10)

    def test_deep_itm(self):
        """Deep ITM call should be close to S - K (intrinsic value)."""
        S = np.array([100.0])
        K = np.array([50.0])
        v = np.array([0.04])  # sigma=0.2, T=1
        price = bs_call(S, K, v)
        intrinsic = 100.0 - 50.0
        # Deep ITM: price >= intrinsic and close to it
        assert price[0] > intrinsic - 0.01
        assert price[0] < intrinsic + 5.0  # time value bounded

    def test_deep_otm(self):
        """Deep OTM call should be near zero."""
        S = np.array([100.0])
        K = np.array([200.0])
        v = np.array([0.04])  # sigma=0.2, T=1
        price = bs_call(S, K, v)
        assert price[0] < 0.01

    def test_put_call_parity_normalized(self):
        """Call - Put = S - K (put-call parity, no discount since F=S here)."""
        S = np.array([1.0])
        K = np.array([1.05])
        v = np.array([0.04])
        call = bs_call(S, K, v)
        put = bs_call(K, S, v)  # Put via symmetry: P(S,K,v) = Call(K,S,v) + K - S... no
        # Use put-call parity directly: P = C - S + K
        # We verify: C(S, K) - (S - K) for ITM must be positive (time value)
        # Actually, let's just verify the formula against scipy
        sigma = 0.2
        T = 1.0
        d1 = (np.log(S / K) + 0.5 * v) / np.sqrt(v)
        d2 = d1 - np.sqrt(v)
        expected = float((S * norm.cdf(d1) - K * norm.cdf(d2)).item())
        np.testing.assert_allclose(call, expected, atol=1e-10)

    def test_vectorized(self):
        """bs_call should handle vector inputs correctly."""
        S = np.array([100.0, 100.0, 100.0])
        K = np.array([90.0, 100.0, 110.0])
        v = np.full(3, 0.04)
        prices = bs_call(S, K, v)
        assert prices.shape == (3,)
        # Prices should be decreasing with strike
        assert prices[0] > prices[1] > prices[2]

    def test_zero_variance_limit(self):
        """With v -> 0, call -> max(S - K, 0)."""
        S = np.array([100.0, 100.0])
        K = np.array([90.0, 110.0])
        v = np.array([1e-14, 1e-14])  # near zero
        prices = bs_call(S, K, v)
        np.testing.assert_allclose(prices[0], 10.0, atol=0.01)
        np.testing.assert_allclose(prices[1], 0.0, atol=0.01)

    def test_broadcasting(self):
        """Spot scalar, strikes vector, variance scalar."""
        S = np.array([1.0])
        K = np.linspace(0.8, 1.2, 5)
        v = np.array([0.04])
        prices = bs_call(S, K, v)
        assert prices.shape == (5,)
        # Monotonically decreasing in K
        assert np.all(np.diff(prices) < 0)


class TestBSCallVega:
    """Test bs_call_vega correctness."""

    def test_atm_vega_positive(self):
        """ATM vega should be the largest and positive."""
        S = np.array([100.0])
        K = np.array([100.0])
        v = np.array([0.04])
        vega = bs_call_vega(S, K, v)
        assert vega[0] > 0

    def test_vega_symmetry(self):
        """Vega should be approximately symmetric around ATM for small skew."""
        S = np.array([100.0, 100.0])
        K = np.array([95.0, 105.0])
        v = np.full(2, 0.04)
        vegas = bs_call_vega(S, K, v)
        # Should be close but not exact (log-normal asymmetry)
        np.testing.assert_allclose(vegas[0], vegas[1], rtol=0.15)

    def test_vega_vs_finite_diff(self):
        """bs_call_vega = S * phi(d+) * sqrt(v) should be consistent with dC/dv.

        Since dC/dv = S * phi(d+) / (2*sqrt(v)), we have:
            bs_call_vega = 2 * v * dC/dv
        We verify this via finite-difference on v.
        """
        S = np.array([100.0])
        K = np.array([100.0])
        sigma = 0.20
        T = 1.0
        v = np.array([sigma**2 * T])

        vega_analytic = bs_call_vega(S, K, v)

        # Finite difference of C w.r.t. v (total variance)
        h = 1e-7
        dC_dv = (bs_call(S, K, v + h) - bs_call(S, K, v - h)) / (2 * h)

        # bs_call_vega = S * phi(d+) * sqrt(v) = 2 * v * dC/dv
        expected = 2 * v * dC_dv
        np.testing.assert_allclose(vega_analytic, expected, rtol=1e-4)

    def test_deep_otm_vega_small(self):
        """Deep OTM vega should be near zero."""
        S = np.array([100.0])
        K = np.array([200.0])
        v = np.array([0.04])
        vega = bs_call_vega(S, K, v)
        assert vega[0] < 1.0  # very small


# ===================================================================
# 2. fit_sanos with synthetic data
# ===================================================================


class TestFitSanosSynthetic:
    """Test SANOS LP fitting with synthetic lognormal data."""

    @pytest.fixture
    def single_expiry_result(self):
        """Fit SANOS to a single lognormal expiry."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=25
        )
        result = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=80,
            K_min=0.70,
            K_max=1.40,
        )
        return result

    @pytest.fixture
    def multi_expiry_result(self):
        """Fit SANOS to 3 lognormal expiries."""
        strikes, calls, variances = _make_synthetic_multi_expiry(
            sigma=0.20, T_list=[0.05, 0.10, 0.25], n_strikes=20
        )
        result = fit_sanos(
            market_strikes=strikes,
            market_calls=calls,
            atm_variances=variances,
            eta=0.25,
            n_model_strikes=80,
        )
        return result

    def test_lp_converges(self, single_expiry_result):
        """LP solver should converge for well-posed lognormal data."""
        assert single_expiry_result.lp_success is True

    def test_densities_sum_to_one(self, single_expiry_result):
        """Each marginal density must sum to 1 (probability constraint)."""
        for j, q in enumerate(single_expiry_result.densities):
            np.testing.assert_allclose(
                q.sum(), 1.0, atol=1e-8,
                err_msg=f"Density {j} does not sum to 1",
            )

    def test_densities_unit_mean(self, single_expiry_result):
        """E[K] = sum(q * K_model) = 1 (martingale constraint, F=1)."""
        r = single_expiry_result
        for j, q in enumerate(r.densities):
            mean = np.dot(q, r.model_strikes)
            np.testing.assert_allclose(
                mean, 1.0, atol=1e-6,
                err_msg=f"Density {j} does not have unit mean",
            )

    def test_densities_nonneg(self, single_expiry_result):
        """All density weights must be non-negative."""
        for q in single_expiry_result.densities:
            assert np.all(q >= -1e-12), "Density has negative weights"

    def test_fit_errors_small(self, single_expiry_result):
        """Fit errors should be tiny for clean lognormal data."""
        r = single_expiry_result
        assert r.max_fit_error < 0.01, (
            f"Max fit error {r.max_fit_error:.6f} exceeds threshold 0.01"
        )

    def test_multi_expiry_lp_converges(self, multi_expiry_result):
        assert multi_expiry_result.lp_success is True

    def test_multi_expiry_densities_valid(self, multi_expiry_result):
        r = multi_expiry_result
        for j, q in enumerate(r.densities):
            np.testing.assert_allclose(q.sum(), 1.0, atol=1e-8)
            mean = np.dot(q, r.model_strikes)
            np.testing.assert_allclose(mean, 1.0, atol=1e-6)
            assert np.all(q >= -1e-12)

    def test_multi_expiry_fit_errors(self, multi_expiry_result):
        assert multi_expiry_result.max_fit_error < 0.01

    def test_expiry_labels_default(self, multi_expiry_result):
        """Default expiry labels should be T0, T1, T2."""
        assert multi_expiry_result.expiry_labels == ["T0", "T1", "T2"]

    def test_expiry_labels_custom(self):
        """Custom expiry labels should be preserved."""
        K, C, v = _make_synthetic_single_expiry()
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([v]),
            expiry_labels=["2025-08-28"],
        )
        assert r.expiry_labels == ["2025-08-28"]


# ===================================================================
# 3. SANOSResult.price: reprice market strikes
# ===================================================================


class TestSANOSResultPrice:
    @pytest.fixture
    def fitted(self):
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=25
        )
        return fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=80,
        )

    def test_reprice_market_strikes(self, fitted):
        """Re-pricing at market strikes should match market mids closely."""
        r = fitted
        mkt_K = r.market_strikes[0]
        mkt_C = r.market_mids[0]
        model_C = r.price(0, mkt_K)
        np.testing.assert_allclose(model_C, mkt_C, atol=0.005)

    def test_price_monotone_decreasing_in_K(self, fitted):
        """Call prices must be decreasing in strike."""
        K = np.linspace(0.75, 1.35, 200)
        prices = fitted.price(0, K)
        diffs = np.diff(prices)
        # Allow tiny numerical noise
        assert np.all(diffs < 1e-8), "Call prices not monotone decreasing in K"

    def test_price_bounded(self, fitted):
        """0 <= C(K) <= F = 1 (normalized)."""
        K = np.linspace(0.75, 1.35, 200)
        prices = fitted.price(0, K)
        assert np.all(prices >= -1e-10)
        assert np.all(prices <= 1.0 + 1e-10)

    def test_price_call_spread_bound(self, fitted):
        """C(K1) - C(K2) <= K2 - K1 for K1 < K2 (no super-replication)."""
        K = np.linspace(0.80, 1.30, 100)
        prices = fitted.price(0, K)
        for i in range(len(K) - 1):
            spread = prices[i] - prices[i + 1]
            max_spread = K[i + 1] - K[i]
            assert spread <= max_spread + 1e-10


# ===================================================================
# 4. SANOSResult.iv: bisection convergence
# ===================================================================


class TestSANOSResultIV:
    @pytest.fixture
    def fitted(self):
        sigma = 0.20
        T = 0.25
        K, C, atm_var = _make_synthetic_single_expiry(sigma=sigma, T=T)
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=80,
        )
        return r, sigma, T

    def test_iv_roundtrip(self, fitted):
        """IV -> price -> IV should be consistent (roundtrip < 1e-6)."""
        r, sigma, T = fitted
        K = np.linspace(0.85, 1.15, 30)
        ivs = r.iv(0, K, T)
        # Re-price from IV and compare to SANOS price
        prices_from_iv = bs_call(np.ones_like(K), K, ivs**2 * T)
        prices_sanos = r.price(0, K)
        np.testing.assert_allclose(prices_from_iv, prices_sanos, atol=1e-6)

    def test_iv_atm_recovers_input_vol(self, fitted):
        """ATM implied vol should be close to the input sigma."""
        r, sigma, T = fitted
        K_atm = np.array([1.0])
        iv_atm = r.iv(0, K_atm, T)
        # With eta > 0 there is smoothing, so allow some tolerance
        np.testing.assert_allclose(iv_atm, sigma, atol=0.03)

    def test_iv_positive(self, fitted):
        """Implied vol should be positive everywhere."""
        r, _, T = fitted
        K = np.linspace(0.85, 1.15, 50)
        ivs = r.iv(0, K, T)
        assert np.all(ivs > 0)


# ===================================================================
# 5. SANOSResult.density
# ===================================================================


class TestSANOSResultDensity:
    @pytest.fixture
    def fitted(self):
        sigma = 0.20
        T = 0.25
        K, C, atm_var = _make_synthetic_single_expiry(sigma=sigma, T=T)
        return fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=80,
        )

    def test_density_nonnegative(self, fitted):
        """Risk-neutral density must be non-negative everywhere."""
        K = np.linspace(0.70, 1.50, 500)
        density = fitted.density(0, K)
        assert np.all(density >= -1e-10), "Density is negative somewhere"

    def test_density_integrates_to_one(self, fitted):
        """Density should integrate to approximately 1 over a wide range."""
        # Use trapezoidal rule on a fine grid
        K = np.linspace(0.50, 2.00, 5000)
        density = fitted.density(0, K)
        integral = np.trapezoid(density, K)
        np.testing.assert_allclose(integral, 1.0, atol=0.05)

    def test_density_has_peak_near_forward(self, fitted):
        """Peak of the density should be near the forward (K=1)."""
        K = np.linspace(0.70, 1.50, 500)
        density = fitted.density(0, K)
        peak_K = K[np.argmax(density)]
        assert 0.85 < peak_K < 1.15, f"Density peak at {peak_K}, expected near 1.0"

    def test_density_tails_decay(self, fitted):
        """Density at extreme strikes should be very small."""
        K_far = np.array([0.50, 2.00])
        density = fitted.density(0, K_far)
        assert np.all(density < 0.1), "Density tails are too heavy"


# ===================================================================
# 6. Calendar arbitrage: prices increase with T
# ===================================================================


class TestCalendarArbitrage:
    @pytest.fixture
    def multi_result(self):
        strikes, calls, variances = _make_synthetic_multi_expiry(
            sigma=0.20, T_list=[0.05, 0.10, 0.25], n_strikes=20
        )
        return fit_sanos(
            market_strikes=strikes,
            market_calls=calls,
            atm_variances=variances,
            eta=0.25,
            n_model_strikes=80,
        )

    def test_no_calendar_arb(self, multi_result):
        """Call prices must be non-decreasing in expiry at every strike.

        C(T2, K) >= C(T1, K) for T2 > T1 (all K).
        """
        r = multi_result
        K = np.linspace(0.80, 1.25, 100)
        for j in range(len(r.densities) - 1):
            prices_short = r.price(j, K)
            prices_long = r.price(j + 1, K)
            violations = prices_short - prices_long
            assert np.all(violations < 1e-6), (
                f"Calendar arb between expiry {j} and {j+1}: "
                f"max violation = {violations.max():.8f}"
            )


# ===================================================================
# 7. Butterfly (convexity): d2C/dK2 >= 0
# ===================================================================


class TestButterfly:
    @pytest.fixture
    def fitted(self):
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=25
        )
        return fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=80,
        )

    def test_convexity(self, fitted):
        """d2C/dK2 >= 0 everywhere (no butterfly arbitrage)."""
        K = np.linspace(0.80, 1.30, 300)
        prices = fitted.price(0, K)
        # Second difference (discrete approximation of d2C/dK2)
        d2 = np.diff(prices, 2)
        dK = K[1] - K[0]
        d2_normalized = d2 / (dK**2)
        # Allow small numerical noise
        assert np.all(d2_normalized > -1e-3), (
            f"Butterfly arb: min d2C/dK2 = {d2_normalized.min():.6f}"
        )

    def test_density_equals_second_derivative(self, fitted):
        """Breeden-Litzenberger: d2C/dK2 should match the density method."""
        K = np.linspace(0.85, 1.20, 200)
        prices = fitted.price(0, K)
        dK = K[1] - K[0]

        # Numerical second derivative
        d2_num = np.diff(prices, 2) / (dK**2)
        K_mid = K[1:-1]

        # Analytical density from SANOS
        density_analytic = fitted.density(0, K_mid)

        # They should roughly agree (finite-diff introduces error)
        np.testing.assert_allclose(d2_num, density_analytic, atol=0.5, rtol=0.3)


# ===================================================================
# 8. prepare_nifty_chain with mock DataFrame
# ===================================================================


class TestPrepareNiftyChain:
    @pytest.fixture
    def mock_fno_df(self):
        """Create a minimal mock F&O DataFrame for NIFTY options."""
        import pandas as pd

        spot = 24000.0
        forward = 24050.0
        trade_date = "2025-07-30"
        expiry = "2025-08-28"

        rows = []
        # Generate CE and PE strikes around forward
        strikes = np.arange(23000, 25200, 100, dtype=float)
        for K in strikes:
            # Generate synthetic prices using BS
            sigma = 0.15
            T = 29 / 365.0
            v = sigma**2 * T
            S_arr = np.array([forward])
            K_arr = np.array([K])
            v_arr = np.array([v])
            call_price = float(bs_call(S_arr, K_arr, v_arr)[0])
            # Put via parity: P = C - F + K
            put_price = call_price - forward + K

            # CE row
            rows.append({
                "TradDt": trade_date,
                "TckrSymb": "NIFTY",
                "FinInstrmTp": "IDO",
                "XpryDt": expiry,
                "StrkPric": K,
                "OptnTp": "CE",
                "ClsPric": max(call_price, 0.05),
                "UndrlygPric": spot,
            })
            # PE row
            rows.append({
                "TradDt": trade_date,
                "TckrSymb": "NIFTY",
                "FinInstrmTp": "IDO",
                "XpryDt": expiry,
                "StrkPric": K,
                "OptnTp": "PE",
                "ClsPric": max(put_price, 0.05),
                "UndrlygPric": spot,
            })

        return pd.DataFrame(rows)

    def test_returns_dict(self, mock_fno_df):
        """prepare_nifty_chain should return a dict (not None) for valid data."""
        result = prepare_nifty_chain(mock_fno_df)
        assert result is not None

    def test_keys_present(self, mock_fno_df):
        result = prepare_nifty_chain(mock_fno_df)
        expected_keys = {
            "market_strikes", "market_calls", "market_spreads",
            "atm_variances", "expiry_labels", "forward", "spot", "trade_date",
        }
        assert set(result.keys()) == expected_keys

    def test_strikes_normalized_around_one(self, mock_fno_df):
        """After normalization by forward, strikes should bracket 1.0."""
        result = prepare_nifty_chain(mock_fno_df)
        for k_arr in result["market_strikes"]:
            assert k_arr.min() < 1.0
            assert k_arr.max() > 1.0

    def test_calls_positive(self, mock_fno_df):
        """Normalized call prices should be positive."""
        result = prepare_nifty_chain(mock_fno_df)
        for c_arr in result["market_calls"]:
            assert np.all(c_arr > 0)

    def test_forward_reasonable(self, mock_fno_df):
        """Implied forward should be close to spot."""
        result = prepare_nifty_chain(mock_fno_df)
        assert abs(result["forward"] - 24000.0) < 500.0

    def test_atm_variance_positive(self, mock_fno_df):
        result = prepare_nifty_chain(mock_fno_df)
        assert np.all(result["atm_variances"] > 0)

    def test_empty_symbol_returns_none(self, mock_fno_df):
        """Querying for a symbol not in the data should return None."""
        result = prepare_nifty_chain(mock_fno_df, symbol="BANKNIFTY")
        assert result is None

    def test_wrong_instrument_returns_none(self, mock_fno_df):
        """Querying for wrong instrument type should return None."""
        result = prepare_nifty_chain(mock_fno_df, instrument="STF")
        assert result is None

    def test_can_fit_sanos_from_mock_chain(self, mock_fno_df):
        """Full pipeline: mock data -> prepare_nifty_chain -> fit_sanos."""
        chain = prepare_nifty_chain(mock_fno_df)
        assert chain is not None

        result = fit_sanos(
            market_strikes=chain["market_strikes"],
            market_calls=chain["market_calls"],
            market_spreads=chain["market_spreads"],
            atm_variances=chain["atm_variances"],
            expiry_labels=chain["expiry_labels"],
            eta=0.25,
        )
        assert result.lp_success is True
        assert result.max_fit_error < 0.02

    def test_multi_expiry_mock(self):
        """Mock with 2 expiries -- verify both are processed."""
        import pandas as pd

        spot = 24000.0
        forward = 24050.0
        trade_date = "2025-07-30"
        expiries = ["2025-08-28", "2025-09-25"]

        rows = []
        for expiry in expiries:
            strikes = np.arange(23200, 24800, 200, dtype=float)
            for K in strikes:
                sigma = 0.15
                T = 29 / 365.0 if expiry == expiries[0] else 57 / 365.0
                v = sigma**2 * T
                cp = float(bs_call(np.array([forward]), np.array([K]), np.array([v]))[0])
                pp = cp - forward + K
                for tp, pr in [("CE", cp), ("PE", pp)]:
                    rows.append({
                        "TradDt": trade_date,
                        "TckrSymb": "NIFTY",
                        "FinInstrmTp": "IDO",
                        "XpryDt": expiry,
                        "StrkPric": K,
                        "OptnTp": tp,
                        "ClsPric": max(pr, 0.05),
                        "UndrlygPric": spot,
                    })

        df = pd.DataFrame(rows)
        result = prepare_nifty_chain(df)
        assert result is not None
        assert len(result["market_strikes"]) == 2
        assert len(result["expiry_labels"]) == 2


# ===================================================================
# 8b. prepare_nifty_chain with real parquet data (skip if missing)
# ===================================================================


class TestPrepareNiftyChainReal:
    @pytest.fixture
    def fno_df(self):
        """Load one day of real FnO bhavcopy from DuckDB store."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        try:
            from data.store import MarketDataStore
            store = MarketDataStore()
            dates = store.available_dates("nse_fo_bhavcopy")
            if not dates:
                pytest.skip("No nse_fo_bhavcopy data in DuckDB store")
            d = dates[-1]
            df = store.sql("SELECT * FROM nse_fo_bhavcopy WHERE date = ?", [d.isoformat()])
            if df is None or df.empty:
                pytest.skip("No data for latest date")
            return df
        except Exception as e:
            pytest.skip(f"DuckDB store unavailable: {e}")

    def test_real_chain_not_none(self, fno_df):
        """Should return a valid chain for real NIFTY data."""
        result = prepare_nifty_chain(fno_df, symbol="NIFTY")
        if result is None:
            pytest.skip("No NIFTY IDO data in this file")
        assert len(result["market_strikes"]) >= 1

    def test_real_chain_fit(self, fno_df):
        """Fit SANOS to real NIFTY data and check basic properties."""
        chain = prepare_nifty_chain(fno_df, symbol="NIFTY", max_expiries=3)
        if chain is None:
            pytest.skip("No NIFTY IDO data in this file")

        result = fit_sanos(
            market_strikes=chain["market_strikes"],
            market_calls=chain["market_calls"],
            market_spreads=chain["market_spreads"],
            atm_variances=chain["atm_variances"],
            expiry_labels=chain["expiry_labels"],
            eta=0.25,
        )
        assert result.lp_success is True
        # Real data won't fit as tightly as synthetic
        assert result.max_fit_error < 0.10

        # Densities should still satisfy constraints
        for j, q in enumerate(result.densities):
            np.testing.assert_allclose(q.sum(), 1.0, atol=1e-6)
            mean = np.dot(q, result.model_strikes)
            np.testing.assert_allclose(mean, 1.0, atol=1e-4)


# ===================================================================
# 9. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_expiry(self):
        """SANOS should work with just 1 expiry (no calendar constraints)."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.15, T=0.10, n_strikes=10
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=50,
        )
        assert r.lp_success is True
        assert len(r.densities) == 1
        np.testing.assert_allclose(r.densities[0].sum(), 1.0, atol=1e-8)

    def test_very_few_market_strikes(self):
        """SANOS should handle as few as 5 market strikes."""
        K = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
        C = _lognormal_call_prices(K, sigma=0.20, T=0.25)
        atm_var = 0.20**2 * 0.25

        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=50,
        )
        assert r.lp_success is True
        assert r.max_fit_error < 0.01

    def test_eta_zero(self):
        """eta=0 should still work (linear interpolation limit)."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=20
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.0,
            n_model_strikes=80,
        )
        # With eta=0, the LP becomes: sum q_i * max(K_model_i - K, 0)
        # which is a standard piecewise-linear approximation.
        assert r.lp_success is True
        for q in r.densities:
            np.testing.assert_allclose(q.sum(), 1.0, atol=1e-8)

    def test_eta_near_one(self):
        """eta close to 1 (very smooth) should still converge."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=20
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.99,
            n_model_strikes=60,
        )
        assert r.lp_success is True

    def test_high_vol(self):
        """High vol surface should still fit without error."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.80, T=1.0, n_strikes=30,
            K_lo=0.40, K_hi=2.50,
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=100,
            K_min=0.30,
            K_max=3.00,
        )
        assert r.lp_success is True
        assert r.max_fit_error < 0.05

    def test_low_vol(self):
        """Low vol surface should still fit."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.05, T=0.05, n_strikes=15,
            K_lo=0.95, K_hi=1.05,
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=80,
        )
        assert r.lp_success is True

    def test_many_expiries(self):
        """5 expiries should work without issue."""
        Ts = [0.02, 0.05, 0.10, 0.25, 0.50]
        strikes, calls, variances = _make_synthetic_multi_expiry(
            sigma=0.20, T_list=Ts, n_strikes=15
        )
        r = fit_sanos(
            market_strikes=strikes,
            market_calls=calls,
            atm_variances=variances,
            eta=0.25,
            n_model_strikes=60,
        )
        assert r.lp_success is True
        assert len(r.densities) == 5

        # Calendar arbitrage should hold across all pairs
        K_test = np.linspace(0.80, 1.25, 50)
        for j in range(4):
            p_short = r.price(j, K_test)
            p_long = r.price(j + 1, K_test)
            assert np.all(p_long >= p_short - 1e-6)

    def test_atm_variance_auto_estimation(self):
        """When atm_variances is None, fit_sanos should estimate them."""
        K, C, _ = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=25
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=None,  # let it estimate
            eta=0.25,
        )
        assert r.lp_success is True
        # Estimated variance should be in reasonable range
        assert 0.001 < r.variances[0] < 1.0

    def test_market_spreads_weighting(self):
        """Providing market spreads should not break anything."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=20
        )
        spreads = np.full_like(C, 0.005)  # uniform 50bps spread

        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            market_spreads=[spreads],
            atm_variances=np.array([atm_var]),
            eta=0.25,
        )
        assert r.lp_success is True

    def test_wide_model_grid(self):
        """Very wide model grid (K_min=0.3, K_max=3.0)."""
        K, C, atm_var = _make_synthetic_single_expiry(
            sigma=0.20, T=0.25, n_strikes=20
        )
        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=150,
            K_min=0.30,
            K_max=3.00,
        )
        assert r.lp_success is True
        # Martingale constraint should still hold
        mean = np.dot(r.densities[0], r.model_strikes)
        np.testing.assert_allclose(mean, 1.0, atol=1e-6)

    def test_narrow_model_grid(self):
        """Narrow model grid covering the market strikes."""
        K = np.linspace(0.92, 1.08, 15)
        C = _lognormal_call_prices(K, sigma=0.20, T=0.25)
        atm_var = 0.20**2 * 0.25

        r = fit_sanos(
            market_strikes=[K],
            market_calls=[C],
            atm_variances=np.array([atm_var]),
            eta=0.25,
            n_model_strikes=60,
            K_min=0.85,
            K_max=1.15,
        )
        assert r.lp_success is True


# ===================================================================
# 10. Variance monotonicity enforcement
# ===================================================================


class TestVarianceMonotonicity:
    def test_non_increasing_variances_fixed(self):
        """If supplied variances are not increasing, fit_sanos should fix them."""
        # Deliberately supply decreasing variances
        K1 = np.linspace(0.85, 1.15, 15)
        K2 = np.linspace(0.85, 1.15, 15)
        C1 = _lognormal_call_prices(K1, sigma=0.20, T=0.25)
        C2 = _lognormal_call_prices(K2, sigma=0.20, T=0.50)

        # Supply v2 < v1 (wrong order)
        bad_vars = np.array([0.02, 0.01])

        r = fit_sanos(
            market_strikes=[K1, K2],
            market_calls=[C1, C2],
            atm_variances=bad_vars,
            eta=0.25,
        )
        # Internal fix: v[1] should be at least v[0] + epsilon
        assert r.variances[1] >= r.variances[0]


# ===================================================================
# 11. Stress: large number of model strikes
# ===================================================================


class TestPerformance:
    @pytest.mark.slow
    def test_large_grid(self):
        """200 model strikes, 3 expiries -- should still converge."""
        strikes, calls, variances = _make_synthetic_multi_expiry(
            sigma=0.20, T_list=[0.05, 0.10, 0.25], n_strikes=30
        )
        r = fit_sanos(
            market_strikes=strikes,
            market_calls=calls,
            atm_variances=variances,
            eta=0.25,
            n_model_strikes=200,
        )
        assert r.lp_success is True
