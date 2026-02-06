"""Microstructure features for Indian equity derivatives.

Bar-level features (OHLCV-compatible via Feature protocol):
  - VPIN: Bulk Volume Classification approximation
  - Kyle's Lambda: price impact regression
  - Amihud illiquidity
  - Corwin-Schultz high-low spread estimator
  - Volume imbalance

Tick-level functions (for use with raw tick data from MarketDataStore):
  - vpin_from_ticks(): volume-synchronized VPIN
  - tick_entropy(): Shannon entropy of tick return distribution
  - trade_arrival_hawkes(): self-exciting intensity from trade timestamps

References:
  - Easley, Lopez de Prado, O'Hara (2012) — VPIN
  - Kyle (1985) — Lambda
  - Amihud (2002) — Illiquidity
  - Corwin & Schultz (2012) — High-low spread estimator
  - Bacry et al. (2015) — Hawkes processes in finance

Ported and adapted from qlxr_crypto/apps/crypto_flow/features.py
for the Indian market tick data format (ltp, volume, oi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from qlx.features.base import Feature


# ---------------------------------------------------------------------------
# Bar-Level Feature: Market Microstructure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Microstructure(Feature):
    """Bar-level microstructure features from OHLCV data.

    Produces: vpin, kyles_lambda, amihud, hl_spread, volume_imbalance
    """

    window: int = 50

    @property
    def name(self) -> str:
        return f"micro_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window + 1

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["Close"].values.astype(np.float64)
        v = df["Volume"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        l = df["Low"].values.astype(np.float64)
        n = len(df)
        out = pd.DataFrame(index=df.index)

        # --- Log returns ---
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(
            np.maximum(c[1:], 1e-8) / np.maximum(c[:-1], 1e-8)
        )

        # --- VPIN (Bulk Volume Classification) ---
        # Rolling sigma for BVC
        sigma = np.full(n, 0.01)
        for i in range(20, n):
            s = np.std(log_ret[i - 20 : i])
            if s > 1e-8:
                sigma[i] = s

        buy_frac = norm.cdf(log_ret / sigma)
        buy_vol = v * buy_frac
        sell_vol = v * (1 - buy_frac)
        imbalance = np.abs(buy_vol - sell_vol)
        total_vol = np.maximum(v, 1e-8)

        vpin = np.full(n, np.nan)
        for i in range(self.window, n):
            vpin[i] = (
                np.sum(imbalance[i - self.window : i])
                / np.sum(total_vol[i - self.window : i])
            )
        out["vpin"] = vpin

        # --- Kyle's Lambda: cov(return, signed_vol) / var(signed_vol) ---
        signed_vol = v * np.sign(log_ret)

        kl = np.full(n, np.nan)
        for i in range(self.window, n):
            sv = signed_vol[i - self.window : i]
            ret = log_ret[i - self.window : i]
            var_sv = np.var(sv)
            if var_sv > 1e-20:
                kl[i] = np.cov(ret, sv)[0, 1] / var_sv
        out["kyles_lambda"] = kl

        # --- Amihud illiquidity: mean(|return| / dollar_volume) ---
        abs_ret = np.abs(log_ret)
        dollar_vol = c * v
        ratio = np.where(dollar_vol > 0, abs_ret / dollar_vol, 0.0)

        amihud = np.full(n, np.nan)
        for i in range(self.window, n):
            amihud[i] = np.mean(ratio[i - self.window : i])
        out["amihud"] = amihud * 1e6  # scale for readability

        # --- High-Low Spread (Corwin-Schultz 2012) ---
        ln_hl = np.log(np.maximum(h, 1e-8) / np.maximum(l, 1e-8))
        hl_spread = np.full(n, np.nan)

        for i in range(2, n):
            beta = ln_hl[i - 1] ** 2 + ln_hl[i] ** 2
            h2 = max(h[i - 1], h[i])
            l2 = min(l[i - 1], l[i])
            gamma = np.log(max(h2, 1e-8) / max(l2, 1e-8)) ** 2

            denom = 3 - 2 * math.sqrt(2)
            if denom > 0 and beta > 0:
                alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / denom
                alpha -= math.sqrt(max(gamma / denom, 0))
                alpha = max(alpha, 0)
                hl_spread[i] = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))

        spread_series = pd.Series(hl_spread, index=df.index)
        out["hl_spread"] = spread_series.rolling(self.window).mean().values

        # --- Volume imbalance (signed volume change normalized) ---
        vol_change = np.zeros(n)
        vol_change[1:] = v[1:] - v[:-1]
        signed_vol_change = vol_change * np.sign(log_ret)

        vol_imb = np.full(n, np.nan)
        for i in range(self.window, n):
            total = np.sum(np.abs(signed_vol_change[i - self.window : i]))
            if total > 0:
                vol_imb[i] = (
                    np.sum(signed_vol_change[i - self.window : i]) / total
                )
        out["volume_imbalance"] = vol_imb

        return out


# ---------------------------------------------------------------------------
# Tick-Level: VPIN from raw ticks
# ---------------------------------------------------------------------------

def vpin_from_ticks(
    prices: np.ndarray,
    volumes: np.ndarray,
    bucket_size: float = 1_000_000.0,
    n_buckets: int = 50,
    sigma_window: int = 100,
) -> np.ndarray:
    """Compute VPIN from tick-level data.

    Parameters
    ----------
    prices : array of last traded prices.
    volumes : array of traded volumes.
    bucket_size : INR volume per bucket (default 10 lakh).
    n_buckets : rolling window of buckets for VPIN.
    sigma_window : number of ticks for rolling sigma.

    Returns
    -------
    Array of VPIN values (NaN where not yet available, forward-filled).
    """
    n = len(prices)
    vpin = np.full(n, np.nan)

    log_ret = np.zeros(n)
    log_ret[1:] = np.log(
        np.maximum(prices[1:], 1e-8) / np.maximum(prices[:-1], 1e-8)
    )

    sigma = np.full(n, 0.01)
    for i in range(sigma_window, n):
        s = np.std(log_ret[i - sigma_window : i])
        if s > 1e-8:
            sigma[i] = s

    dvol = prices * volumes
    buy_frac = norm.cdf(log_ret / sigma)
    buy_vol = dvol * buy_frac
    sell_vol = dvol * (1 - buy_frac)

    current_buy = 0.0
    current_sell = 0.0
    current_vol = 0.0
    completed: list[float] = []

    for i in range(1, n):
        bv = buy_vol[i]
        sv = sell_vol[i]
        dv = dvol[i]

        current_buy += bv
        current_sell += sv
        current_vol += dv

        while current_vol >= bucket_size:
            overflow = current_vol - bucket_size
            frac = 1.0 - overflow / max(dv, 1e-12)

            bucket_imb = abs(
                current_buy - bv * (1 - frac) - (current_sell - sv * (1 - frac))
            )
            completed.append(bucket_imb)

            if len(completed) > n_buckets:
                completed = completed[-n_buckets:]

            current_buy = bv * (1 - frac)
            current_sell = sv * (1 - frac)
            current_vol = overflow

            if len(completed) >= n_buckets:
                vpin[i] = sum(completed) / (n_buckets * bucket_size)

    # Forward-fill
    last = np.nan
    for i in range(n):
        if np.isnan(vpin[i]):
            vpin[i] = last
        else:
            last = vpin[i]

    return vpin


# ---------------------------------------------------------------------------
# Tick-Level: Shannon entropy of returns
# ---------------------------------------------------------------------------

def tick_entropy(
    prices: np.ndarray,
    window: int = 100,
    n_bins: int = 10,
) -> np.ndarray:
    """Shannon entropy of tick return distribution.

    High entropy = uniform/random market.
    Low entropy = concentrated/trending moves.
    """
    n = len(prices)
    entropy = np.full(n, np.nan)

    log_ret = np.zeros(n)
    log_ret[1:] = np.log(
        np.maximum(prices[1:], 1e-8) / np.maximum(prices[:-1], 1e-8)
    )

    for i in range(window, n):
        rets = log_ret[i - window : i]
        counts, _ = np.histogram(rets, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy[i] = -np.sum(probs * np.log2(probs))

    return entropy


# ---------------------------------------------------------------------------
# Tick-Level: Hawkes process for trade clustering
# ---------------------------------------------------------------------------

def trade_arrival_hawkes(
    timestamps: np.ndarray,
    eval_interval: float = 1.0,
) -> tuple[np.ndarray, float, float, float]:
    """Estimate Hawkes process parameters from trade arrival times.

    Returns (intensity_series, mu, alpha, beta).

    Uses Method of Moments calibration then computes the self-exciting
    intensity at regular intervals.  High intensity_ratio = cascade
    territory (informed trading / momentum ignition).
    """
    if len(timestamps) < 20:
        return np.full(1, np.nan), 1.0, 0.5, 1.5

    ts = timestamps.astype(np.float64)
    # Normalize to seconds
    if ts[0] > 1e15:
        ts = ts / 1e9
    elif ts[0] > 1e12:
        ts = ts / 1e3
    ts = ts - ts[0]

    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt) < 10:
        return np.full(1, np.nan), 1.0, 0.5, 1.5

    lambda_bar = len(timestamps) / max(ts[-1], 1e-6)

    # Branching ratio from variance of binned counts
    bin_size = 10.0
    n_bins = max(int(ts[-1] / bin_size), 5)
    counts = np.histogram(ts, bins=n_bins)[0].astype(float)
    mean_c = np.mean(counts)
    var_c = np.var(counts)

    if mean_c > 0:
        ratio = var_c / mean_c
        n_est = max(0.01, min(0.95, 1 - 1 / math.sqrt(max(ratio, 1.01))))
    else:
        n_est = 0.3

    # Beta from autocorrelation decay
    if len(counts) > 2:
        acf1 = abs(np.corrcoef(counts[:-1], counts[1:])[0, 1])
        acf1 = max(0.01, min(0.99, acf1))
        beta = -math.log(acf1) / bin_size
        beta = max(0.1, min(10.0, beta))
    else:
        beta = 1.5

    mu = lambda_bar * (1 - n_est)
    alpha = n_est * beta

    # Compute intensity at regular intervals
    eval_times = np.arange(0, ts[-1], eval_interval)
    intensity = np.full(len(eval_times), mu)

    sum_kernel = 0.0
    last_time = 0.0
    event_idx = 0

    for i, t in enumerate(eval_times):
        while event_idx < len(ts) and ts[event_idx] <= t:
            dt_e = ts[event_idx] - last_time
            sum_kernel = sum_kernel * math.exp(-beta * dt_e) + 1.0
            last_time = ts[event_idx]
            event_idx += 1

        dt_e = t - last_time
        decayed = sum_kernel * math.exp(-beta * dt_e)
        intensity[i] = mu + alpha * decayed

    return intensity, mu, alpha, beta
