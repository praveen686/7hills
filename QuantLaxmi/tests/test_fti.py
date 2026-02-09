"""Tests for FTI (Follow Through Index) Numba implementation.

Validates against the C++ reference (FTI.CPP) and pure-Python port (masters.py).
"""

import math
import numpy as np
import pandas as pd
import pytest

from features.masters import _fti_coefs, fti_single
from features.fti import (
    _quickselect_nb,
    fti_coefs_nb,
    fti_process_window_scratch_nb,
    fti_process_window_nb,
    fti_1d_nb,
    fti_nb,
    fti_alloc_scratch_nb,
    fti_stream_process,
    FollowThroughIndex,
    FTI_VBT,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_trend(n=300, slope=0.001, noise=0.0, base=100.0):
    """Generate a linear trend in price space."""
    rng = np.random.RandomState(42)
    t = np.arange(n, dtype=np.float64)
    log_prices = np.log(base) + slope * t
    if noise > 0:
        log_prices += rng.randn(n) * noise
    return np.exp(log_prices)


def _make_sine(n=500, period=20, amplitude=0.02, base=100.0):
    """Generate a sine-wave oscillation in log-price space."""
    t = np.arange(n, dtype=np.float64)
    log_prices = np.log(base) + amplitude * np.sin(2 * np.pi * t / period)
    return np.exp(log_prices)


def _make_noise(n=300, vol=0.01, base=100.0):
    """Generate pure random walk (no trend)."""
    rng = np.random.RandomState(123)
    log_ret = rng.randn(n) * vol
    log_prices = np.log(base) + np.cumsum(log_ret)
    return np.exp(log_prices)


# ── Section 1: Filter Coefficients ──────────────────────────────────────────


class TestFtiCoefs:
    """Validate fti_coefs_nb against the pure-Python _fti_coefs."""

    def test_matches_python_reference(self):
        """Numba coefs must match the pure-Python version exactly."""
        H = 32
        for period in [5, 10, 20, 40, 64]:
            py_coefs = _fti_coefs(period, H)
            nb_coefs = fti_coefs_nb(period, period, H)  # single-period slice
            np.testing.assert_allclose(
                nb_coefs[0], py_coefs, rtol=1e-12,
                err_msg=f"Coefs mismatch at period={period}",
            )

    def test_dc_gain_unity(self):
        """Sum of symmetric filter weights should equal 1 (unit DC gain)."""
        H = 33
        coefs = fti_coefs_nb(5, 65, H)
        for ip in range(coefs.shape[0]):
            # DC gain = c[0] + 2*sum(c[1:H+1])
            dc = coefs[ip, 0] + 2.0 * np.sum(coefs[ip, 1:])
            assert abs(dc - 1.0) < 1e-12, f"DC gain {dc} != 1.0 at ip={ip}"

    def test_multi_period_shape(self):
        """Coefs array has correct shape."""
        coefs = fti_coefs_nb(5, 65, 33)
        assert coefs.shape == (61, 34)  # 61 periods, 34 coefs each


# ── Section 2: LS Extrapolation Fix ─────────────────────────────────────────


class TestExtrapolation:
    """Verify the LS extrapolation matches C++ (fixes masters.py bug)."""

    def test_linear_trend_continues_upward(self):
        """For a linear uptrend, extrapolation must continue UP, not reverse."""
        # Simple linear: y = 1, 2, 3, ..., 10
        prices = np.arange(1.0, 11.0)
        H = 3
        coefs = fti_coefs_nb(5, 6, H)

        # use_log=False so we can reason about linear extrapolation directly
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 6, 0.95, 0.20, False,
        )

        # Filtered value at the current bar should be close to the current price
        # (a linear trend passes through a lowpass filter approximately unchanged)
        assert f > 8.0, (
            f"Filtered value {f} too low for linear trend ending at 10 — "
            f"suggests extrapolation is reversed (masters.py bug)"
        )

    def test_flat_prices_stay_flat(self):
        """Constant prices should produce filtered value near that constant."""
        prices = np.full(50, 100.0)
        H = 10
        coefs = fti_coefs_nb(5, 20, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 20, 0.95, 0.20, False,
        )
        assert abs(f - 100.0) < 0.01, f"Flat prices: filtered={f}, expected ~100"

    def test_downtrend_continues_downward(self):
        """For a linear downtrend, extrapolation must continue DOWN."""
        prices = np.arange(100.0, 80.0, -1.0)  # 100, 99, ..., 81
        H = 5
        coefs = fti_coefs_nb(5, 10, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 10, 0.95, 0.20, False,
        )
        # Filtered should be near the latest price (81), not pulled back up
        assert f < 85.0, f"Filtered={f} too high for downtrend ending at 81"


# ── Section 3: Process Window ────────────────────────────────────────────────


class TestProcessWindow:
    """Test fti_process_window_nb core algorithm."""

    def test_too_short_returns_nan(self):
        """Window shorter than half_length+2 should return NaN."""
        prices = np.array([100.0, 101.0, 102.0])
        H = 10
        coefs = fti_coefs_nb(5, 10, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 10, 0.95, 0.20, True,
        )
        assert np.isnan(f) and np.isnan(fti_val)

    def test_strong_trend_high_fti(self):
        """Clean linear trend should yield a high FTI."""
        prices = _make_trend(200, slope=0.003, noise=0.0)
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 64, 0.95, 0.20, True,
        )
        assert fti_val > 1.0, f"Clean trend should give FTI > 1, got {fti_val}"

    def test_noise_low_fti(self):
        """Pure noise should yield a low FTI."""
        prices = _make_noise(200, vol=0.02)
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 64, 0.95, 0.20, True,
        )
        assert fti_val < 6.0, f"Noise should give low FTI, got {fti_val}"

    def test_best_period_in_range(self):
        """Best period must be within [min_period, max_period]."""
        prices = _make_sine(200, period=20)
        H = 32
        min_p, max_p = 5, 64
        coefs = fti_coefs_nb(min_p, max_p, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, min_p, max_p, 0.95, 0.20, True,
        )
        assert min_p <= p <= max_p, f"Best period {p} outside [{min_p}, {max_p}]"

    def test_width_positive(self):
        """Channel width must be non-negative."""
        prices = _make_trend(200, slope=0.002, noise=0.005)
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 64, 0.95, 0.20, True,
        )
        assert w >= 0.0, f"Width should be >= 0, got {w}"

    def test_log_mode_filtered_positive(self):
        """In log mode, filtered price must be positive."""
        prices = _make_trend(200)
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        f, p, w, fti_val = fti_process_window_nb(
            prices, coefs, H, 5, 64, 0.95, 0.20, True,
        )
        assert f > 0.0, f"Filtered price should be positive in log mode, got {f}"


# ── Section 4: Rolling 1D ───────────────────────────────────────────────────


class TestFti1d:
    """Test fti_1d_nb rolling computation."""

    def test_warmup_nans(self):
        """First lookback-1 values must be NaN."""
        prices = _make_trend(300)
        lb = 100
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        f, p, w, fv = fti_1d_nb(prices, coefs, lb, H, 5, 64, 0.95, 0.20, True)
        assert np.all(np.isnan(fv[:lb - 1])), "Warmup period should be NaN"
        assert np.isfinite(fv[lb - 1]), "First valid bar should be finite"

    def test_output_shape(self):
        """Output arrays must match input length."""
        n = 250
        prices = _make_trend(n)
        lb = 100
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        f, p, w, fv = fti_1d_nb(prices, coefs, lb, H, 5, 64, 0.95, 0.20, True)
        assert f.shape == (n,) and p.shape == (n,)
        assert w.shape == (n,) and fv.shape == (n,)

    def test_trend_higher_fti_than_noise(self):
        """Trending data should have higher mean FTI than noise."""
        trend = _make_trend(300, slope=0.003)
        noise = _make_noise(300, vol=0.01)
        lb, H = 100, 32
        coefs = fti_coefs_nb(5, 64, H)

        _, _, _, fv_trend = fti_1d_nb(
            trend, coefs, lb, H, 5, 64, 0.95, 0.20, True,
        )
        _, _, _, fv_noise = fti_1d_nb(
            noise, coefs, lb, H, 5, 64, 0.95, 0.20, True,
        )

        mean_trend = np.nanmean(fv_trend)
        mean_noise = np.nanmean(fv_noise)
        assert mean_trend > mean_noise, (
            f"Trend FTI {mean_trend:.2f} should exceed noise FTI {mean_noise:.2f}"
        )


# ── Section 5: 2D Wrapper ───────────────────────────────────────────────────


class TestFtiNb:
    """Test fti_nb 2D wrapper."""

    def test_2d_matches_1d(self):
        """2D result must match column-wise 1D results."""
        prices = _make_trend(250, slope=0.002)
        lb, H, minp, maxp = 100, 32, 5, 64

        close_2d = prices.reshape(-1, 1)
        f2, p2, w2, fv2 = fti_nb(close_2d, lb, H, minp, maxp, 0.95, 0.20, True)

        coefs = fti_coefs_nb(minp, min(maxp, 2 * H), H)
        f1, p1, w1, fv1 = fti_1d_nb(
            prices, coefs, lb, H, minp, min(maxp, 2 * H), 0.95, 0.20, True,
        )

        np.testing.assert_allclose(f2[:, 0], f1, rtol=1e-12)
        np.testing.assert_allclose(fv2[:, 0], fv1, rtol=1e-12)

    def test_multi_column(self):
        """Multiple columns should be processed independently."""
        trend = _make_trend(250, slope=0.003)
        noise = _make_noise(250, vol=0.01)
        close_2d = np.column_stack([trend, noise])
        lb, H = 100, 32

        f, p, w, fv = fti_nb(close_2d, lb, H, 5, 64, 0.95, 0.20, True)
        assert fv.shape == (250, 2)
        # Trend column should have higher mean FTI
        assert np.nanmean(fv[:, 0]) > np.nanmean(fv[:, 1])


# ── Section 6: Feature Class ────────────────────────────────────────────────


class TestFollowThroughIndex:
    """Test the QuantLaxmi Feature subclass."""

    def test_feature_interface(self):
        """Feature must satisfy name/lookback/lookforward protocol."""
        feat = FollowThroughIndex(lookback_window=200)
        assert feat.name == "fti_200"
        assert feat.lookback == 200
        assert feat.lookforward == 0

    def test_output_columns(self):
        """Feature._compute must return 4 columns."""
        prices = _make_trend(300, slope=0.002)
        df = pd.DataFrame({
            "Open": prices, "High": prices * 1.01,
            "Low": prices * 0.99, "Close": prices,
            "Volume": np.ones(300) * 1000,
        })
        feat = FollowThroughIndex(lookback_window=100, half_length=20, max_period=40)
        result = feat._compute(df)
        assert set(result.columns) == {"filtered", "best_period", "best_width", "best_fti"}
        assert len(result) == 300


# ── Section 7: IndicatorFactory ──────────────────────────────────────────────


class TestFtiVbt:
    """Test the vectorbtpro IndicatorFactory wrapper."""

    def test_basic_run(self):
        """FTI_VBT.run() must produce the expected output structure."""
        prices = pd.Series(_make_trend(300, slope=0.002), name="Close")
        result = FTI_VBT.run(prices, lookback=100, min_period=5, max_period=40,
                             half_length=20)
        assert hasattr(result, "best_fti")
        assert hasattr(result, "filtered")
        assert hasattr(result, "best_period")
        assert hasattr(result, "best_width")
        # Should have valid values after warmup
        fti_vals = result.best_fti.values.flatten()
        assert np.any(np.isfinite(fti_vals)), "Should have some finite FTI values"


# ── Section 8: Streaming ────────────────────────────────────────────────────


class TestStreamProcess:
    """Test fti_stream_process convenience wrapper."""

    def test_returns_dict(self):
        """Streaming wrapper must return a dict with expected keys."""
        prices = _make_trend(200)
        H = 32
        coefs = fti_coefs_nb(5, 64, H)
        result = fti_stream_process(prices, coefs, H, 5, 64)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"filtered", "best_period", "best_width", "best_fti"}
        assert np.isfinite(result["best_fti"])


# ── Section 9: Coefs vs Python Reference (detailed) ─────────────────────────


class TestCoefsDetailed:
    """More detailed coefficient validation."""

    def test_all_periods_match(self):
        """Every period in [5, 65] should match the Python reference."""
        H = 32
        nb_coefs = fti_coefs_nb(5, 65, H)
        for ip, period in enumerate(range(5, 66)):
            py_coefs = _fti_coefs(period, H)
            np.testing.assert_allclose(
                nb_coefs[ip], py_coefs, rtol=1e-12,
                err_msg=f"Mismatch at period={period}",
            )

    def test_symmetry_property(self):
        """Filter coefficients should be non-negative near center for large periods."""
        H = 32
        coefs = fti_coefs_nb(30, 30, H)
        # Center coefficient should be positive
        assert coefs[0, 0] > 0


# ── Section 10: Quickselect ──────────────────────────────────────────────────


class TestQuickselect:
    """Validate _quickselect_nb against np.sort."""

    def test_matches_sort_simple(self):
        """Quickselect must return same value as sorted[k] for simple array."""
        arr = np.array([9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0])
        for k in range(len(arr)):
            work = arr.copy()
            result = _quickselect_nb(work, len(work), k)
            expected = np.sort(arr)[k]
            assert result == expected, f"k={k}: got {result}, expected {expected}"

    def test_matches_sort_random(self):
        """Quickselect must match sort for random arrays across many seeds."""
        for seed in range(20):
            rng = np.random.RandomState(seed)
            n = rng.randint(10, 200)
            arr = rng.randn(n)
            k = rng.randint(0, n)
            work = arr.copy()
            result = _quickselect_nb(work, n, k)
            expected = np.sort(arr)[k]
            np.testing.assert_allclose(
                result, expected, rtol=1e-14,
                err_msg=f"seed={seed}, n={n}, k={k}",
            )

    def test_all_equal(self):
        """Quickselect on constant array should return that constant."""
        arr = np.full(50, 42.0)
        for k in [0, 25, 49]:
            work = arr.copy()
            assert _quickselect_nb(work, 50, k) == 42.0

    def test_single_element(self):
        """Single-element array."""
        arr = np.array([7.0])
        assert _quickselect_nb(arr, 1, 0) == 7.0

    def test_two_elements(self):
        """Two-element array, both k values."""
        arr = np.array([5.0, 3.0])
        assert _quickselect_nb(arr.copy(), 2, 0) == 3.0
        assert _quickselect_nb(arr.copy(), 2, 1) == 5.0


# ── Section 11: Scratch Buffer API ──────────────────────────────────────────


class TestScratchBuffers:
    """Test scratch-buffer API for streaming FTI."""

    def test_alloc_scratch_shapes(self):
        """fti_alloc_scratch_nb returns correct shapes."""
        lb, H, minp, maxp = 200, 33, 5, 65
        y, fv, wv, ftv, dw, lw = fti_alloc_scratch_nb(lb, H, minp, maxp)
        maxp_c = min(maxp, 2 * H)
        n_periods = maxp_c - minp + 1
        channel_len = lb - H
        assert y.shape == (lb + H,)
        assert fv.shape == (n_periods,)
        assert wv.shape == (n_periods,)
        assert ftv.shape == (n_periods,)
        assert dw.shape == (channel_len,)
        assert lw.shape == (channel_len,)

    def test_scratch_matches_non_scratch(self):
        """Scratch version must produce identical results to allocating version."""
        prices = _make_trend(200, slope=0.003, noise=0.001)
        H, minp, maxp = 32, 5, 64
        coefs = fti_coefs_nb(minp, maxp, H)

        # Non-scratch
        f1, p1, w1, fv1 = fti_process_window_nb(
            prices, coefs, H, minp, maxp, 0.95, 0.20, True,
        )

        # Scratch
        scratch = fti_alloc_scratch_nb(len(prices), H, minp, maxp)
        y, fvs, wvs, ftvs, dws, lws = scratch
        f2, p2, w2, fv2 = fti_process_window_scratch_nb(
            prices, coefs, H, minp, maxp, 0.95, 0.20, True,
            y, fvs, wvs, ftvs, dws, lws,
        )

        np.testing.assert_allclose(f1, f2, rtol=1e-14)
        np.testing.assert_allclose(p1, p2, rtol=1e-14)
        np.testing.assert_allclose(w1, w2, rtol=1e-14)
        np.testing.assert_allclose(fv1, fv2, rtol=1e-14)

    def test_scratch_reuse_across_bars(self):
        """Scratch reused across bars must match fresh allocation each bar."""
        prices = _make_trend(300, slope=0.002, noise=0.001)
        lb, H, minp, maxp = 100, 32, 5, 64
        coefs = fti_coefs_nb(minp, maxp, H)
        scratch = fti_alloc_scratch_nb(lb, H, minp, maxp)
        y, fvs, wvs, ftvs, dws, lws = scratch

        for bar_idx in [lb - 1, 150, 200, 299]:
            window = prices[bar_idx - lb + 1: bar_idx + 1]

            # Fresh alloc
            f1, p1, w1, fv1 = fti_process_window_nb(
                window, coefs, H, minp, maxp, 0.95, 0.20, True,
            )

            # Reused scratch
            f2, p2, w2, fv2 = fti_process_window_scratch_nb(
                window, coefs, H, minp, maxp, 0.95, 0.20, True,
                y, fvs, wvs, ftvs, dws, lws,
            )

            np.testing.assert_allclose(f1, f2, rtol=1e-14,
                                       err_msg=f"bar={bar_idx}")
            np.testing.assert_allclose(fv1, fv2, rtol=1e-14,
                                       err_msg=f"bar={bar_idx}")

    def test_stream_process_with_scratch(self):
        """fti_stream_process with scratch must match without scratch."""
        prices = _make_trend(200)
        H, minp, maxp = 32, 5, 64
        coefs = fti_coefs_nb(minp, min(maxp, 2 * H), H)

        r1 = fti_stream_process(prices, coefs, H, minp, maxp)
        scratch = fti_alloc_scratch_nb(len(prices), H, minp, maxp)
        r2 = fti_stream_process(prices, coefs, H, minp, maxp, scratch=scratch)

        for key in ["filtered", "best_period", "best_width", "best_fti"]:
            np.testing.assert_allclose(
                r1[key], r2[key], rtol=1e-14,
                err_msg=f"Mismatch in {key}",
            )


# ── Section 12: Streaming Bar-by-Bar Parity ─────────────────────────────────


class TestStreamingParity:
    """Verify bar-by-bar streaming matches batch rolling."""

    def test_streaming_matches_batch(self):
        """Bar-by-bar ring buffer + fti_process_window_scratch_nb must match
        fti_1d_nb batch output exactly."""
        prices = _make_trend(300, slope=0.002, noise=0.001)
        lb, H, minp, maxp = 100, 32, 5, 64
        maxp_c = min(maxp, 2 * H)
        coefs = fti_coefs_nb(minp, maxp_c, H)

        # Batch
        f_batch, p_batch, w_batch, fv_batch = fti_1d_nb(
            prices, coefs, lb, H, minp, maxp_c, 0.95, 0.20, True,
        )

        # Streaming: simulate ring buffer
        scratch = fti_alloc_scratch_nb(lb, H, minp, maxp_c)
        y, fvs, wvs, ftvs, dws, lws = scratch

        f_stream = np.full(len(prices), np.nan)
        fv_stream = np.full(len(prices), np.nan)

        for i in range(lb - 1, len(prices)):
            window = prices[i - lb + 1: i + 1]
            f, p, w, fval = fti_process_window_scratch_nb(
                window, coefs, H, minp, maxp_c, 0.95, 0.20, True,
                y, fvs, wvs, ftvs, dws, lws,
            )
            f_stream[i] = f
            fv_stream[i] = fval

        # Must match exactly (same code path, same scratch reuse)
        np.testing.assert_allclose(
            f_stream[lb - 1:], f_batch[lb - 1:], rtol=1e-14,
            err_msg="Filtered values: streaming vs batch mismatch",
        )
        np.testing.assert_allclose(
            fv_stream[lb - 1:], fv_batch[lb - 1:], rtol=1e-14,
            err_msg="FTI values: streaming vs batch mismatch",
        )
