"""Microstructure feature computations.

All functions are pure numpy/pandas — no network calls.
Designed for both real-time (incremental) and batch (backtest) use.

References:
  - VPIN: Easley, Lopez de Prado, O'Hara (2012)
  - OFI: Cont, Kukanov, Stoikov (2014)
  - Hawkes: Bacry et al. (2015), "Hawkes processes in finance"
  - Kyle's Lambda: Kyle (1985), continuous auction model
  - Amihud: Amihud (2002), illiquidity measure
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# VPIN — Volume-synchronized Probability of Informed Trading
# ---------------------------------------------------------------------------

@dataclass
class VPINState:
    """Incremental VPIN accumulator.

    Volume is partitioned into equal-sized buckets.  Within each bucket,
    trades are classified as buy or sell using the Bulk Volume Classification
    (BVC) approach: fraction of volume assigned to buy side =
    CDF(ΔP / σ_ΔP), where ΔP is the log-price change over the bar.

    VPIN = mean(|V_buy - V_sell|) / bucket_size  over the last N buckets.
    """

    bucket_size: float = 50_000.0       # USD volume per bucket
    n_buckets: int = 50                  # rolling window of buckets

    # Internal state
    _current_buy: float = 0.0
    _current_sell: float = 0.0
    _current_vol: float = 0.0
    _completed: list[float] = field(default_factory=list)  # |Vb - Vs| per bucket

    def update(self, price: float, volume_usd: float, prev_price: float,
               sigma: float = 0.01) -> float | None:
        """Feed one trade/bar.  Returns VPIN when a bucket completes, else None.

        BVC: buy_frac = Phi(log(price/prev_price) / sigma)
        """
        if prev_price <= 0 or sigma <= 0:
            return None

        dp = math.log(price / prev_price)
        z = dp / sigma
        buy_frac = norm.cdf(z)

        buy_vol = volume_usd * buy_frac
        sell_vol = volume_usd * (1 - buy_frac)

        self._current_buy += buy_vol
        self._current_sell += sell_vol
        self._current_vol += volume_usd

        result = None
        # Fill buckets (a single bar can span multiple buckets)
        while self._current_vol >= self.bucket_size:
            overflow = self._current_vol - self.bucket_size
            frac_in = 1 - overflow / max(volume_usd, 1e-12)

            # Proportionally split the overflow
            bucket_buy = self._current_buy - buy_vol * (1 - frac_in)
            bucket_sell = self._current_sell - sell_vol * (1 - frac_in)
            imbalance = abs(bucket_buy - bucket_sell)
            self._completed.append(imbalance)

            if len(self._completed) > self.n_buckets:
                self._completed = self._completed[-self.n_buckets:]

            # Start new bucket with overflow
            self._current_buy = buy_vol * (1 - frac_in)
            self._current_sell = sell_vol * (1 - frac_in)
            self._current_vol = overflow

            if len(self._completed) >= self.n_buckets:
                result = sum(self._completed) / (self.n_buckets * self.bucket_size)

        return result


def compute_vpin_series(
    prices: np.ndarray,
    volumes_usd: np.ndarray,
    bucket_size: float = 50_000.0,
    n_buckets: int = 50,
    sigma_window: int = 20,
) -> np.ndarray:
    """Compute VPIN for an entire price/volume series (backtest mode).

    Returns array of same length as prices, NaN where VPIN not yet available.
    """
    n = len(prices)
    vpin = np.full(n, np.nan)

    # Rolling volatility for BVC
    log_ret = np.log(prices[1:] / prices[:-1])
    sigma = np.full(n, 0.01)
    for i in range(sigma_window, n):
        sigma[i] = max(np.std(log_ret[i - sigma_window:i]), 1e-8)

    state = VPINState(bucket_size=bucket_size, n_buckets=n_buckets)
    for i in range(1, n):
        v = state.update(prices[i], volumes_usd[i], prices[i - 1], sigma[i])
        if v is not None:
            vpin[i] = v

    # Forward-fill VPIN (updates only on bucket completion)
    last = np.nan
    for i in range(n):
        if np.isnan(vpin[i]):
            vpin[i] = last
        else:
            last = vpin[i]

    return vpin


# ---------------------------------------------------------------------------
# OFI — Order Flow Imbalance (Cont-Kukanov-Stoikov 2014)
# ---------------------------------------------------------------------------

@dataclass
class OFIState:
    """Incremental OFI computation from bookTicker updates.

    OFI_t = (bid_qty * I(bid >= prev_bid) - prev_bid_qty * I(bid <= prev_bid))
          + (prev_ask_qty * I(ask >= prev_ask) - ask_qty * I(ask <= prev_ask))

    Accumulates tick-level OFI, emits smoothed OFI on request.
    """

    ema_alpha: float = 0.05  # EMA smoothing for OFI signal

    _prev_bid: float = 0.0
    _prev_ask: float = 0.0
    _prev_bid_qty: float = 0.0
    _prev_ask_qty: float = 0.0
    _ema_ofi: float = 0.0
    _n_ticks: int = 0

    def update(self, bid: float, ask: float,
               bid_qty: float, ask_qty: float) -> float:
        """Process one bookTicker tick.  Returns smoothed OFI."""
        if self._n_ticks == 0:
            self._prev_bid = bid
            self._prev_ask = ask
            self._prev_bid_qty = bid_qty
            self._prev_ask_qty = ask_qty
            self._n_ticks = 1
            return 0.0

        # Bid-side
        ofi_bid = 0.0
        if bid >= self._prev_bid:
            ofi_bid += bid_qty
        if bid <= self._prev_bid:
            ofi_bid -= self._prev_bid_qty

        # Ask-side
        ofi_ask = 0.0
        if ask >= self._prev_ask:
            ofi_ask += self._prev_ask_qty
        if ask <= self._prev_ask:
            ofi_ask -= ask_qty

        raw_ofi = ofi_bid + ofi_ask

        # EMA smoothing
        self._ema_ofi = self.ema_alpha * raw_ofi + (1 - self.ema_alpha) * self._ema_ofi

        self._prev_bid = bid
        self._prev_ask = ask
        self._prev_bid_qty = bid_qty
        self._prev_ask_qty = ask_qty
        self._n_ticks += 1

        return self._ema_ofi


def compute_ofi_series(
    bids: np.ndarray,
    asks: np.ndarray,
    bid_qtys: np.ndarray,
    ask_qtys: np.ndarray,
    ema_alpha: float = 0.05,
) -> np.ndarray:
    """Compute smoothed OFI for a time series (backtest mode)."""
    n = len(bids)
    ofi = np.zeros(n)
    state = OFIState(ema_alpha=ema_alpha)

    for i in range(n):
        ofi[i] = state.update(bids[i], asks[i], bid_qtys[i], ask_qtys[i])

    return ofi


# ---------------------------------------------------------------------------
# Hawkes Process — Self-exciting intensity
# ---------------------------------------------------------------------------

@dataclass
class HawkesState:
    """Exponential-decay Hawkes intensity estimator.

    lambda(t) = mu + sum_i alpha * exp(-beta * (t - t_i))

    We track the decayed sum incrementally:
        S(t) = sum_i exp(-beta * (t - t_i))
        S(t_new) = S(t_prev) * exp(-beta * dt) + 1   (when event at t_new)
        lambda(t) = mu + alpha * S(t)

    The branching ratio n = alpha / beta governs criticality:
        n < 1: stationary (subcritical), n -> 1: explosive cascades
    """

    mu: float = 1.0         # baseline intensity (events/sec)
    alpha: float = 0.8      # excitation amplitude
    beta: float = 1.5       # decay rate

    _sum_kernel: float = 0.0
    _last_time: float = 0.0
    _n_events: int = 0

    @property
    def branching_ratio(self) -> float:
        """n = alpha/beta. Close to 1 = near critical (cascade)."""
        return self.alpha / self.beta if self.beta > 0 else 0.0

    def intensity(self, t: float) -> float:
        """Current intensity at time t (between events)."""
        dt = t - self._last_time if t > self._last_time else 0.0
        decayed = self._sum_kernel * math.exp(-self.beta * dt)
        return self.mu + self.alpha * decayed

    def event(self, t: float) -> float:
        """Record an event at time t. Returns new intensity."""
        dt = t - self._last_time if t > self._last_time else 0.0
        self._sum_kernel = self._sum_kernel * math.exp(-self.beta * dt) + 1.0
        self._last_time = t
        self._n_events += 1
        return self.mu + self.alpha * self._sum_kernel

    def intensity_ratio(self, t: float) -> float:
        """lambda(t) / mu — how excited vs baseline. >3 = cascade territory."""
        lam = self.intensity(t)
        return lam / self.mu if self.mu > 0 else 0.0


def calibrate_hawkes(
    timestamps: np.ndarray,
    window_sec: float = 300.0,
) -> tuple[float, float, float]:
    """Quick MoM calibration of Hawkes parameters from event times.

    Uses method of moments (Bacry et al. 2015):
        mu = (1 - n) * lambda_bar
        alpha/beta = n (branching ratio)
        beta estimated from autocorrelation half-life

    Returns (mu, alpha, beta).
    """
    if len(timestamps) < 10:
        return 1.0, 0.5, 1.5

    # Inter-event durations
    dt = np.diff(timestamps)
    dt = dt[dt > 0]
    if len(dt) < 5:
        return 1.0, 0.5, 1.5

    lambda_bar = len(timestamps) / max(timestamps[-1] - timestamps[0], 1e-6)

    # Estimate branching ratio from variance of counts
    # Var(N(T)) = lambda_bar * T / (1 - n)^2 for Hawkes
    bin_size = 10.0  # 10-second bins
    t0, t1 = timestamps[0], timestamps[-1]
    n_bins = int((t1 - t0) / bin_size)
    if n_bins < 5:
        return lambda_bar, 0.5, 1.5

    counts = np.histogram(timestamps, bins=n_bins)[0].astype(float)
    var_counts = np.var(counts)
    mean_counts = np.mean(counts)

    if mean_counts > 0:
        ratio = var_counts / mean_counts  # = 1/(1-n)^2 for Hawkes
        n_est = max(0.01, min(0.95, 1 - 1 / math.sqrt(max(ratio, 1.01))))
    else:
        n_est = 0.3

    # Beta from autocorrelation decay
    if len(counts) > 2:
        acf1 = np.corrcoef(counts[:-1], counts[1:])[0, 1]
        acf1 = max(0.01, min(0.99, abs(acf1)))
        beta = -math.log(acf1) / bin_size
        beta = max(0.1, min(10.0, beta))
    else:
        beta = 1.5

    mu = lambda_bar * (1 - n_est)
    alpha = n_est * beta

    return mu, alpha, beta


def compute_hawkes_intensity(
    timestamps: np.ndarray,
    eval_times: np.ndarray,
    mu: float = 1.0,
    alpha: float = 0.8,
    beta: float = 1.5,
) -> np.ndarray:
    """Compute Hawkes intensity at specified evaluation times (backtest)."""
    intensity = np.full(len(eval_times), mu)
    state = HawkesState(mu=mu, alpha=alpha, beta=beta)

    event_idx = 0
    for i, t in enumerate(eval_times):
        # Process all events up to time t
        while event_idx < len(timestamps) and timestamps[event_idx] <= t:
            state.event(timestamps[event_idx])
            event_idx += 1
        intensity[i] = state.intensity(t)

    return intensity


# ---------------------------------------------------------------------------
# Kyle's Lambda — Price impact regression
# ---------------------------------------------------------------------------

def kyles_lambda(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 100,
) -> np.ndarray:
    """Rolling Kyle's Lambda: ΔP = λ * signed_volume + ε.

    Signed volume = volume * sign(ΔP).
    λ = Cov(ΔP, V_signed) / Var(V_signed).

    High λ → fragile market, large price impact per unit of flow.
    """
    n = len(prices)
    lam = np.full(n, np.nan)

    dp = np.diff(prices) / prices[:-1]  # returns
    signed_vol = volumes[1:] * np.sign(dp)

    for i in range(window, len(dp)):
        sv = signed_vol[i - window:i]
        ret = dp[i - window:i]
        var_sv = np.var(sv)
        if var_sv > 1e-20:
            lam[i + 1] = np.cov(ret, sv)[0, 1] / var_sv

    return lam


# ---------------------------------------------------------------------------
# Amihud Illiquidity — |return| / dollar_volume
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    prices: np.ndarray,
    volumes_usd: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Rolling Amihud illiquidity measure.

    ILLIQ = mean(|r_t| / DolVol_t)

    Higher = less liquid = larger price impact.
    """
    n = len(prices)
    illiq = np.full(n, np.nan)

    abs_ret = np.abs(np.diff(np.log(prices)))
    dvol = volumes_usd[1:]

    ratio = np.where(dvol > 0, abs_ret / dvol, 0.0)

    for i in range(window, len(ratio)):
        illiq[i + 1] = np.mean(ratio[i - window:i])

    return illiq


# ---------------------------------------------------------------------------
# Funding Rate PCA — Cross-sectional decomposition
# ---------------------------------------------------------------------------

def funding_pca(
    funding_matrix: pd.DataFrame,
    n_components: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA on cross-sectional funding rates.

    Parameters
    ----------
    funding_matrix : DataFrame
        Rows = time, columns = symbols, values = raw funding rates.
        NaN-filled for missing observations.

    Returns
    -------
    loadings : (n_components, n_symbols) — PC loadings
    scores : (n_times, n_components) — PC scores
    residuals : (n_times, n_symbols) — residual funding (alpha)

    The residual for symbol j at time t is:
        r_jt = funding_jt - sum_k(loading_jk * score_tk)
    Positive residual = symbol pays MORE than explained by common factors.
    """
    mat = funding_matrix.fillna(0).values
    n_times, n_syms = mat.shape

    if n_times < 5 or n_syms < 3:
        return np.zeros((n_components, n_syms)), np.zeros((n_times, n_components)), mat

    # Center
    means = mat.mean(axis=0, keepdims=True)
    centered = mat - means

    # SVD (more numerically stable than eig on covariance)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((n_components, n_syms)), np.zeros((n_times, n_components)), mat

    k = min(n_components, len(S))
    loadings = Vt[:k]                          # (k, n_syms)
    scores = U[:, :k] * S[:k][np.newaxis, :]   # (n_times, k)

    # Reconstruct and compute residuals
    reconstructed = scores @ loadings + means
    residuals = mat - reconstructed

    return loadings, scores, residuals


def rolling_funding_residual(
    funding_matrix: pd.DataFrame,
    symbol: str,
    window: int = 30,
    n_components: int = 3,
) -> np.ndarray:
    """Compute rolling PCA residual funding for one symbol.

    Uses an expanding window (min `window` observations).
    """
    if symbol not in funding_matrix.columns:
        return np.full(len(funding_matrix), np.nan)

    n = len(funding_matrix)
    residuals = np.full(n, np.nan)
    col_idx = list(funding_matrix.columns).index(symbol)

    for i in range(window, n):
        sub = funding_matrix.iloc[max(0, i - window):i]
        _, _, res = funding_pca(sub, n_components=n_components)
        residuals[i] = res[-1, col_idx]

    return residuals


# ---------------------------------------------------------------------------
# Composite Regime Classifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeSnapshot:
    """Current regime assessment for a symbol."""

    symbol: str
    vpin: float              # 0-1, higher = more toxic
    ofi: float               # signed, positive = buy pressure
    hawkes_ratio: float      # intensity/mu, >3 = cascade
    kyles_lambda: float      # price impact, higher = fragile
    funding_residual: float  # PCA residual, positive = overpaying
    source: str = "kline"    # "kline" or "tick" — tick VPIN is real, kline is noise

    @property
    def is_cascade(self) -> bool:
        return self.hawkes_ratio > 3.0

    @property
    def is_safe_for_carry(self) -> bool:
        """Low toxicity + moderate OFI = safe to collect funding."""
        return self.vpin < 0.4 and abs(self.ofi) < 0.5

    @property
    def cascade_direction(self) -> str:
        """During cascade, which way is the flush?"""
        if not self.is_cascade:
            return "neutral"
        return "long" if self.ofi > 0 else "short"

    @property
    def post_cascade_reversion(self) -> str:
        """After cascade dissipates, fade the move."""
        if self.hawkes_ratio < 1.5 and self.vpin > 0.3:
            # Still elevated VPIN but intensity dropping — reversion
            return "long" if self.ofi < -0.3 else "short" if self.ofi > 0.3 else "neutral"
        return "neutral"
