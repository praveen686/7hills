"""HMM Regime-Switching Strategy for India FnO.

Uses a 3-state Gaussian Hidden Markov Model to detect market regimes
(Bull, Bear, Neutral) from daily price data and switches between
momentum and mean-reversion sub-strategies accordingly.

Architecture:
    1. Feature extraction: log returns, realized vol (Yang-Zhang),
       RSI(14), normalized volume, India VIX
    2. HMM fitting: GaussianHMM with 3 states on expanding window
       (min 120 days, refit every 20 days)
    3. Signal generation: momentum in bull, short/flat in bear,
       RSI mean-reversion in neutral
    4. Risk management: 2% daily stop, regime-flip exit,
       confidence threshold 0.6

Fully causal: at time t, only data up to t-1 is used for fitting.
No look-ahead bias.

Uses hmmlearn if available; otherwise implements Gaussian HMM from
scratch via EM (Baum-Welch) with numpy/scipy.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_HISTORY = 60         # Minimum days before first HMM fit
REFIT_INTERVAL = 20      # Refit every N trading days
N_STATES = 3             # Bull, Bear, Neutral
N_EM_ITER = 100          # EM iterations for custom HMM
EM_TOL = 1e-4            # EM convergence tolerance
CONFIDENCE_THRESHOLD = 0.6
DAILY_STOP_LOSS = -0.02  # -2% daily stop
MAX_POSITION = 1.0
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0


class Regime(IntEnum):
    BULL = 0
    BEAR = 1
    NEUTRAL = 2


@dataclass
class HMMResult:
    """Result of HMM fitting and decoding."""
    states: np.ndarray          # (T,) most likely state sequence
    posteriors: np.ndarray      # (T, N_STATES) posterior probabilities
    means: np.ndarray           # (N_STATES, n_features)
    covariances: np.ndarray     # (N_STATES, n_features, n_features)
    transmat: np.ndarray        # (N_STATES, N_STATES) transition matrix
    log_likelihood: float


# ===================================================================
# Feature Extraction (fully causal)
# ===================================================================

def _log_returns(close: np.ndarray, lag: int = 1) -> np.ndarray:
    """Log returns with given lag. First `lag` values are NaN."""
    ret = np.full_like(close, np.nan, dtype=np.float64)
    ret[lag:] = np.log(close[lag:] / close[:-lag])
    return ret


def _yang_zhang_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """Yang-Zhang realized volatility estimator (annualized).

    Combines overnight (close-to-open), Rogers-Satchell, and
    open-to-close components.  More efficient than close-to-close.

    Returns NaN for first `window` values.
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)

    log_oc = np.log(open_[1:] / close[:-1])       # overnight return
    log_co = np.log(close[1:] / open_[1:])         # open-to-close
    log_ho = np.log(high[1:] / open_[1:])
    log_lo = np.log(low[1:] / open_[1:])
    log_hc = np.log(high[1:] / close[1:])
    log_lc = np.log(low[1:] / close[1:])

    # Rogers-Satchell component (per-bar)
    rs = log_ho * log_hc + log_lo * log_lc

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    for i in range(window, n):
        # Slice indices for the arrays (which are 1-shorter than close)
        s, e = i - window, i
        oc_slice = log_oc[s:e]
        co_slice = log_co[s:e]
        rs_slice = rs[s:e]

        sigma_o = np.var(oc_slice, ddof=1)
        sigma_c = np.var(co_slice, ddof=1)
        sigma_rs = np.mean(rs_slice)

        yz_var = sigma_o + k * sigma_c + (1 - k) * sigma_rs
        # Clamp to avoid sqrt of negative due to floating-point
        result[i] = np.sqrt(max(yz_var, 0.0) * 252)

    return result


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI. Returns NaN for first `period` values."""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    delta = np.diff(close)

    if len(delta) < period:
        return result

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Seed with SMA
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return result


def _normalized_volume(volume: np.ndarray, window: int = 20) -> np.ndarray:
    """Volume / 20-day SMA volume. NaN for first `window - 1` values."""
    n = len(volume)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        sma = np.mean(volume[i - window + 1: i + 1])
        result[i] = volume[i] / sma if sma > 0 else np.nan
    return result


def extract_features(
    df: pd.DataFrame,
    vix_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Extract HMM input features from daily OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date, open, high, low, close.
        Optionally: volume.
    vix_series : pd.Series, optional
        India VIX indexed by date.

    Returns
    -------
    pd.DataFrame with columns: date, ret_1d, ret_5d, ret_21d,
        vol_5d, vol_21d, rsi_14, norm_vol, vix.
        Rows with any NaN are kept (caller handles masking).
    """
    close = df["close"].values.astype(np.float64)
    has_ohlc = all(c in df.columns for c in ("open", "high", "low"))
    has_vol = "volume" in df.columns

    feat = pd.DataFrame({"date": df["date"].values})
    feat["ret_1d"] = _log_returns(close, 1)
    feat["ret_5d"] = _log_returns(close, 5)
    feat["ret_21d"] = _log_returns(close, 21)

    if has_ohlc:
        open_ = df["open"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        feat["vol_5d"] = _yang_zhang_vol(open_, high, low, close, 5)
        feat["vol_21d"] = _yang_zhang_vol(open_, high, low, close, 21)
    else:
        # Fallback: close-to-close realized vol
        ret = _log_returns(close, 1)
        feat["vol_5d"] = pd.Series(ret).rolling(5).std() * np.sqrt(252)
        feat["vol_21d"] = pd.Series(ret).rolling(21).std() * np.sqrt(252)

    feat["rsi_14"] = _rsi(close, 14)

    if has_vol:
        feat["norm_vol"] = _normalized_volume(
            df["volume"].values.astype(np.float64), 20
        )
    else:
        feat["norm_vol"] = 1.0  # Default if no volume data

    # India VIX
    if vix_series is not None:
        feat["vix"] = feat["date"].map(
            vix_series.to_dict()
        ).astype(np.float64)
    else:
        feat["vix"] = np.nan

    return feat


# ===================================================================
# Gaussian HMM (from scratch, EM / Baum-Welch)
# ===================================================================

class GaussianHMM:
    """Multivariate Gaussian Hidden Markov Model.

    Implements Baum-Welch (EM) for parameter estimation and Viterbi
    for decoding. Diagonal covariance by default for numerical stability.

    If hmmlearn is available, delegates to it for better performance.
    Otherwise uses a pure numpy/scipy implementation.
    """

    def __init__(
        self,
        n_states: int = N_STATES,
        n_iter: int = N_EM_ITER,
        tol: float = EM_TOL,
        covariance_type: str = "diag",
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._fitted = False

        # Model parameters (set during fit)
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.log_likelihood_: float = -np.inf

        # Try to use hmmlearn
        self._use_hmmlearn = False
        self._hmmlearn_model = None
        try:
            from hmmlearn.hmm import GaussianHMM as _HMM
            self._hmmlearn_model = _HMM(
                n_components=n_states,
                covariance_type=covariance_type if covariance_type != "diag" else "diag",
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            self._use_hmmlearn = True
            logger.info("Using hmmlearn backend for GaussianHMM")
        except ImportError:
            logger.info("hmmlearn not available; using numpy/scipy EM backend")

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Fit the HMM to observation sequence X.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence.

        Returns
        -------
        self
        """
        if self._use_hmmlearn:
            return self._fit_hmmlearn(X)
        return self._fit_em(X)

    def _fit_hmmlearn(self, X: np.ndarray) -> "GaussianHMM":
        """Fit using hmmlearn backend."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._hmmlearn_model.fit(X)
        self.startprob_ = self._hmmlearn_model.startprob_
        self.transmat_ = self._hmmlearn_model.transmat_
        self.means_ = self._hmmlearn_model.means_
        if self.covariance_type == "diag":
            self.covars_ = self._hmmlearn_model.covars_
        else:
            self.covars_ = self._hmmlearn_model.covars_
        self.log_likelihood_ = self._hmmlearn_model.score(X)
        self._fitted = True
        return self

    def _fit_em(self, X: np.ndarray) -> "GaussianHMM":
        """Fit using custom EM (Baum-Welch) algorithm."""
        T, D = X.shape
        K = self.n_states
        rng = np.random.RandomState(self.random_state)

        # --- Initialization via K-means++ seeding ---
        # Pick initial means by K-means++ style
        means = np.zeros((K, D))
        idx0 = rng.randint(T)
        means[0] = X[idx0]
        for k in range(1, K):
            dists = np.min(
                [np.sum((X - means[j]) ** 2, axis=1) for j in range(k)],
                axis=0,
            )
            probs = dists / dists.sum()
            means[k] = X[rng.choice(T, p=probs)]
        self.means_ = means

        # Diagonal covariance initialized from data variance
        data_var = np.var(X, axis=0) + 1e-6
        self.covars_ = np.tile(data_var, (K, 1))

        # Uniform start and transition probabilities
        self.startprob_ = np.ones(K) / K
        self.transmat_ = np.ones((K, K)) / K

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # --- E-step: forward-backward ---
            log_B = self._compute_log_emission(X)  # (T, K)
            log_alpha, log_ll = self._forward(log_B)
            log_beta = self._backward(log_B)

            # Posteriors gamma(t, k) = P(z_t = k | X)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_axis1(log_gamma)[:, None]
            gamma = np.exp(log_gamma)

            # Xi(t, i, j) = P(z_t = i, z_{t+1} = j | X)
            log_xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        log_xi[t, i, j] = (
                            log_alpha[t, i]
                            + np.log(self.transmat_[i, j] + 1e-300)
                            + log_B[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                # Normalize
                log_norm = _logsumexp_2d(log_xi[t])
                log_xi[t] -= log_norm

            xi = np.exp(log_xi)

            # --- M-step ---
            # Start probabilities
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()

            # Transition matrix
            xi_sum = xi.sum(axis=0)       # (K, K)
            gamma_sum = gamma[:-1].sum(axis=0)  # (K,)
            self.transmat_ = xi_sum / (gamma_sum[:, None] + 1e-10)
            # Normalize rows
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

            # Means and covariances
            gamma_k = gamma.sum(axis=0)   # (K,)
            for k in range(K):
                wk = gamma[:, k]          # (T,)
                wk_sum = wk.sum() + 1e-10
                self.means_[k] = (wk[:, None] * X).sum(axis=0) / wk_sum
                diff = X - self.means_[k]
                self.covars_[k] = (
                    (wk[:, None] * diff * diff).sum(axis=0) / wk_sum
                )
                # Floor covariance to prevent degeneracy
                self.covars_[k] = np.maximum(self.covars_[k], 1e-6)

            self.log_likelihood_ = log_ll

            # Convergence check
            if abs(log_ll - prev_ll) < self.tol:
                logger.debug("EM converged at iteration %d (LL=%.4f)", iteration, log_ll)
                break
            prev_ll = log_ll

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior state probabilities P(z_t | X_{1:T}).

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T, n_states)
        """
        if self._use_hmmlearn and self._hmmlearn_model is not None:
            return self._hmmlearn_model.predict_proba(X)

        log_B = self._compute_log_emission(X)
        log_alpha, _ = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp_axis1(log_gamma)[:, None]
        return np.exp(log_gamma)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding: most likely state sequence.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T,)  — state indices
        """
        if self._use_hmmlearn and self._hmmlearn_model is not None:
            return self._hmmlearn_model.predict(X)

        return self._viterbi(X)

    def decode(self, X: np.ndarray) -> HMMResult:
        """Full decode: states, posteriors, parameters."""
        states = self.predict(X)
        posteriors = self.predict_proba(X)
        # Build full covariance matrices for result
        K, D = self.means_.shape
        if self.covariance_type == "diag":
            full_covs = np.zeros((K, D, D))
            for k in range(K):
                np.fill_diagonal(full_covs[k], self.covars_[k])
        else:
            full_covs = self.covars_.copy()

        return HMMResult(
            states=states,
            posteriors=posteriors,
            means=self.means_.copy(),
            covariances=full_covs,
            transmat=self.transmat_.copy(),
            log_likelihood=self.log_likelihood_,
        )

    # --- Internal methods ---

    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log emission probabilities log P(x_t | z_t = k).

        Returns shape (T, K).
        """
        T, D = X.shape
        K = self.n_states
        log_B = np.zeros((T, K))
        for k in range(K):
            if self.covariance_type == "diag":
                # Diagonal Gaussian
                var_k = self.covars_[k]  # (D,)
                diff = X - self.means_[k]  # (T, D)
                log_B[:, k] = (
                    -0.5 * D * np.log(2 * np.pi)
                    - 0.5 * np.sum(np.log(var_k))
                    - 0.5 * np.sum(diff ** 2 / var_k, axis=1)
                )
            else:
                # Full covariance
                rv = sp_stats.multivariate_normal(
                    mean=self.means_[k],
                    cov=self.covars_[k],
                    allow_singular=True,
                )
                log_B[:, k] = rv.logpdf(X)
        return log_B

    def _forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward algorithm (log-space). Returns (log_alpha, log_likelihood)."""
        T, K = log_B.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_B[0]

        log_A = np.log(self.transmat_ + 1e-300)

        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + log_A[:, j])
                    + log_B[t, j]
                )

        log_ll = _logsumexp(log_alpha[-1])
        return log_alpha, log_ll

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """Backward algorithm (log-space)."""
        T, K = log_B.shape
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0  # log(1) = 0

        log_A = np.log(self.transmat_ + 1e-300)

        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = _logsumexp(
                    log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :]
                )

        return log_beta

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for MAP state sequence."""
        log_B = self._compute_log_emission(X)
        T, K = log_B.shape

        log_delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=np.int64)

        log_delta[0] = np.log(self.startprob_ + 1e-300) + log_B[0]
        log_A = np.log(self.transmat_ + 1e-300)

        for t in range(1, T):
            for j in range(K):
                candidates = log_delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        # Backtrack
        states = np.zeros(T, dtype=np.int64)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


# --- Log-space utilities ---

def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp for 1-D array."""
    c = np.max(x)
    if not np.isfinite(c):
        return -np.inf
    return c + np.log(np.sum(np.exp(x - c)))


def _logsumexp_axis1(x: np.ndarray) -> np.ndarray:
    """Log-sum-exp along axis 1 for 2-D array."""
    c = np.max(x, axis=1)
    mask = np.isfinite(c)
    result = np.full(len(c), -np.inf)
    result[mask] = c[mask] + np.log(
        np.sum(np.exp(x[mask] - c[mask, None]), axis=1)
    )
    return result


def _logsumexp_2d(x: np.ndarray) -> float:
    """Log-sum-exp over entire 2-D array."""
    return _logsumexp(x.ravel())


# ===================================================================
# State Labeling — identify which HMM state is Bull/Bear/Neutral
# ===================================================================

def _label_states(means: np.ndarray) -> dict[int, Regime]:
    """Map HMM state indices to Regime labels based on learned means.

    Convention: feature 0 = ret_1d (1-day log return).
    - Bull: highest mean return
    - Bear: lowest mean return
    - Neutral: middle

    Parameters
    ----------
    means : np.ndarray, shape (N_STATES, n_features)

    Returns
    -------
    dict mapping HMM state index -> Regime
    """
    ret_means = means[:, 0]  # 1-day return is feature 0
    order = np.argsort(ret_means)  # ascending
    # order[0] = lowest return (Bear), order[-1] = highest (Bull)
    mapping = {}
    mapping[order[0]] = Regime.BEAR
    mapping[order[-1]] = Regime.BULL
    for k in range(len(order)):
        if k not in mapping:
            mapping[k] = Regime.NEUTRAL
    # Handle N_STATES > 3 gracefully (assign extras to NEUTRAL)
    if len(order) == 3:
        mapping[order[1]] = Regime.NEUTRAL
    return mapping


# ===================================================================
# Signal Generation
# ===================================================================

def _generate_signal(
    regime: Regime,
    confidence: float,
    rsi: float,
    prev_regime: Regime | None,
) -> tuple[float, str]:
    """Generate position signal from regime and indicators.

    Parameters
    ----------
    regime : Regime
        Current regime label.
    confidence : float
        Posterior probability of current regime.
    rsi : float
        Current RSI(14) value.
    prev_regime : Regime or None
        Previous bar's regime (for flip detection).

    Returns
    -------
    (position, signal_name)
        position in [-1.0, 1.0], signal_name as descriptive string.
    """
    # No trading when confidence is low
    if confidence < CONFIDENCE_THRESHOLD:
        return 0.0, "low_confidence"

    # Regime flip -> immediate exit (flat for this bar)
    if prev_regime is not None and regime != prev_regime:
        return 0.0, "regime_flip"

    if regime == Regime.BULL:
        # Momentum: long, sized by confidence
        pos = min(MAX_POSITION, confidence)
        return pos, "bull_momentum"

    elif regime == Regime.BEAR:
        # Short or flat depending on confidence
        if confidence >= 0.75:
            pos = -min(MAX_POSITION, confidence * 0.5)
            return pos, "bear_short"
        else:
            return 0.0, "bear_flat"

    else:  # NEUTRAL
        # Mean-reversion via RSI
        if not np.isfinite(rsi):
            return 0.0, "neutral_no_rsi"
        if rsi < RSI_OVERSOLD:
            pos = min(MAX_POSITION, confidence * 0.7)
            return pos, "neutral_mr_long"
        elif rsi > RSI_OVERBOUGHT:
            pos = -min(MAX_POSITION, confidence * 0.5)
            return pos, "neutral_mr_short"
        else:
            return 0.0, "neutral_flat"


# ===================================================================
# Backtest Engine
# ===================================================================

def _standardize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize features using pre-computed mean/std."""
    return (X - mean) / np.maximum(std, 1e-10)


def run_backtest(
    symbol: str,
    start_date: date,
    end_date: date,
    store: object,
    vix_store: object | None = None,
) -> pd.DataFrame:
    """Run the HMM Regime-Switching backtest.

    Fully causal: at each bar t, the HMM is fitted on data [0, t-1]
    and the signal for bar t is generated from the regime decoded at t-1.
    Daily return at t uses the position decided at close of t-1 applied
    to the return realized on day t.

    Parameters
    ----------
    symbol : str
        Index name (e.g., "NIFTY", "BANKNIFTY").
    start_date : date
        Backtest start date (inclusive). Need at least MIN_HISTORY days
        of data before this to warm up the HMM.
    end_date : date
        Backtest end date (inclusive).
    store : MarketDataStore
        DuckDB-backed data store with nse_index_close view.
    vix_store : optional
        If provided, used to load VIX data.

    Returns
    -------
    pd.DataFrame with columns:
        date, position, signal, regime, confidence, daily_return,
        cumulative_return
    """
    # --- Load data (with warm-up buffer) ---
    warmup_start = start_date - timedelta(days=int(MIN_HISTORY * 2))

    df = store.sql(
        'SELECT date, "Closing Index Value" as close '
        'FROM nse_index_close '
        'WHERE "Index Name" = ? AND date BETWEEN ? AND ? '
        'ORDER BY date',
        [symbol, warmup_start.isoformat(), end_date.isoformat()],
    )

    if df.empty:
        logger.error("No data returned for %s between %s and %s", symbol, warmup_start, end_date)
        return pd.DataFrame()

    # Ensure numeric close
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows for %s (%s to %s)",
        len(df), symbol, df["date"].iloc[0], df["date"].iloc[-1],
    )

    # --- Try to load OHLCV (may not exist for index close) ---
    # nse_index_close typically only has close. We'll use close-to-close vol.
    # Build a minimal DataFrame for feature extraction.
    feat_df = pd.DataFrame({
        "date": df["date"],
        "close": df["close"].values,
    })

    # --- Extract features ---
    features = extract_features(feat_df)

    # HMM feature columns (excluding date, vix if all NaN)
    feat_cols = ["ret_1d", "ret_5d", "ret_21d", "vol_5d", "vol_21d", "rsi_14"]
    if features["norm_vol"].notna().any() and (features["norm_vol"] != 1.0).any():
        feat_cols.append("norm_vol")
    if features["vix"].notna().any():
        feat_cols.append("vix")

    # --- Identify the first valid row where all features are non-NaN ---
    valid_mask = features[feat_cols].notna().all(axis=1)
    first_valid = valid_mask.idxmax() if valid_mask.any() else len(features)

    # --- Find the index of the backtest start date ---
    bt_start_idx = None
    for i, d in enumerate(features["date"]):
        if d >= start_date:
            bt_start_idx = i
            break

    if bt_start_idx is None:
        logger.error("No data on or after start_date %s", start_date)
        return pd.DataFrame()

    # Ensure enough warmup
    available_warmup = bt_start_idx - first_valid
    if available_warmup < MIN_HISTORY:
        logger.warning(
            "Only %d warmup days available (need %d). "
            "Extending backtest start to allow warmup.",
            available_warmup, MIN_HISTORY,
        )
        bt_start_idx = min(first_valid + MIN_HISTORY, len(features) - 1)

    # --- Run day-by-day ---
    records = []
    hmm = None
    last_fit_idx = -REFIT_INTERVAL  # Force fit on first eligible bar
    state_map: dict[int, Regime] = {}
    prev_regime: Regime | None = None
    cum_ret = 1.0
    position = 0.0
    fit_mean: np.ndarray | None = None
    fit_std: np.ndarray | None = None

    close_arr = df["close"].values.astype(np.float64)

    for t in range(bt_start_idx, len(features)):
        today_date = features["date"].iloc[t]
        today_close = close_arr[t]
        yesterday_close = close_arr[t - 1] if t > 0 else today_close
        day_return = np.log(today_close / yesterday_close) if yesterday_close > 0 else 0.0

        # --- Check if we have valid features up to t-1 ---
        if t - 1 < first_valid:
            records.append({
                "date": today_date,
                "position": 0.0,
                "signal": "warmup",
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "daily_return": 0.0,
                "cumulative_return": cum_ret,
            })
            continue

        # Build feature matrix up to t-1 (causal: never include today)
        feat_slice = features.iloc[first_valid:t]
        X_raw = feat_slice[feat_cols].values.astype(np.float64)

        # Drop any rows with NaN in the feature matrix
        valid_rows = np.all(np.isfinite(X_raw), axis=1)
        X_clean = X_raw[valid_rows]

        if len(X_clean) < MIN_HISTORY:
            records.append({
                "date": today_date,
                "position": 0.0,
                "signal": "insufficient_data",
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "daily_return": 0.0,
                "cumulative_return": cum_ret,
            })
            continue

        # --- Refit HMM if needed ---
        if hmm is None or (t - last_fit_idx) >= REFIT_INTERVAL:
            # Standardize features for numerical stability
            fit_mean = X_clean.mean(axis=0)
            fit_std = X_clean.std(axis=0, ddof=1)
            X_std = _standardize_features(X_clean, fit_mean, fit_std)

            hmm = GaussianHMM(
                n_states=N_STATES,
                n_iter=N_EM_ITER,
                tol=EM_TOL,
                covariance_type="diag",
                random_state=42,
            )
            try:
                hmm.fit(X_std)
                # Decode to identify state labels
                result = hmm.decode(X_std)
                # Un-standardize means for labeling
                raw_means = result.means * fit_std + fit_mean
                state_map = _label_states(raw_means)
                last_fit_idx = t
                logger.debug(
                    "HMM refit at %s (T=%d, LL=%.2f)",
                    today_date, len(X_std), result.log_likelihood,
                )
            except Exception as e:
                logger.warning("HMM fit failed at %s: %s", today_date, e)
                records.append({
                    "date": today_date,
                    "position": 0.0,
                    "signal": "fit_error",
                    "regime": "UNKNOWN",
                    "confidence": 0.0,
                    "daily_return": 0.0,
                    "cumulative_return": cum_ret,
                })
                continue

        # --- Decode today's regime (using features up to t-1) ---
        X_std = _standardize_features(X_clean, fit_mean, fit_std)

        try:
            posteriors = hmm.predict_proba(X_std)
        except Exception as e:
            logger.warning("HMM predict failed at %s: %s", today_date, e)
            records.append({
                "date": today_date,
                "position": 0.0,
                "signal": "predict_error",
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "daily_return": 0.0,
                "cumulative_return": cum_ret,
            })
            continue

        # Use the last row's posterior (most recent known regime)
        last_posterior = posteriors[-1]
        hmm_state = np.argmax(last_posterior)
        confidence = float(last_posterior[hmm_state])
        regime = state_map.get(hmm_state, Regime.NEUTRAL)

        # Current RSI
        rsi_val = float(features["rsi_14"].iloc[t - 1])

        # --- Generate signal ---
        position, signal_name = _generate_signal(
            regime, confidence, rsi_val, prev_regime
        )

        # --- Apply daily stop-loss ---
        realized_return = position * day_return
        if realized_return < DAILY_STOP_LOSS:
            realized_return = DAILY_STOP_LOSS
            signal_name = f"{signal_name}|stopped"
            position = 0.0  # Flatten after stop

        cum_ret *= (1.0 + realized_return)

        records.append({
            "date": today_date,
            "position": position,
            "signal": signal_name,
            "regime": regime.name,
            "confidence": round(confidence, 4),
            "daily_return": round(realized_return, 6),
            "cumulative_return": round(cum_ret, 6),
        })

        prev_regime = regime

    result_df = pd.DataFrame(records)
    if result_df.empty:
        logger.error("Backtest produced no records")
        return result_df

    # --- Compute summary statistics ---
    _print_statistics(result_df, symbol)
    return result_df


def _print_statistics(df: pd.DataFrame, symbol: str) -> dict:
    """Compute and print backtest statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Backtest result with daily_return column.
    symbol : str
        Symbol name for display.

    Returns
    -------
    dict of statistics.
    """
    rets = df["daily_return"].values
    all_rets = rets  # Include flat days for honest Sharpe

    # Sharpe ratio: ddof=1, sqrt(252), all daily returns
    mean_ret = np.mean(all_rets)
    std_ret = np.std(all_rets, ddof=1)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Win rate and trade count
    active_rets = rets[df["position"].values != 0]
    n_trades = len(active_rets)
    win_rate = float(np.mean(active_rets > 0)) if n_trades > 0 else 0.0
    avg_trade = float(np.mean(active_rets)) if n_trades > 0 else 0.0

    # Regime distribution
    regime_counts = df["regime"].value_counts().to_dict()

    # Total return
    total_ret = float(df["cumulative_return"].iloc[-1] - 1.0)

    stats = {
        "symbol": symbol,
        "start_date": str(df["date"].iloc[0]),
        "end_date": str(df["date"].iloc[-1]),
        "total_days": len(df),
        "total_return": f"{total_ret:.2%}",
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": f"{max_dd:.2%}",
        "win_rate": f"{win_rate:.2%}",
        "avg_trade_return": f"{avg_trade:.4%}",
        "num_active_days": n_trades,
        "regime_distribution": regime_counts,
    }

    print("\n" + "=" * 60)
    print(f"  HMM Regime-Switching Backtest: {symbol}")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:>22s}: {v}")
    print("=" * 60 + "\n")

    return stats


# ===================================================================
# Main — Run backtest
# ===================================================================

def main() -> None:
    """Run HMM regime strategy backtest for NIFTY."""
    # Add project root to path
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from quantlaxmi.data.store import MarketDataStore

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbol = "Nifty 50"
    start = date(2025, 8, 6)
    end = date(2026, 2, 6)

    logger.info("Starting HMM Regime-Switching backtest: %s %s -> %s", symbol, start, end)

    with MarketDataStore() as store:
        result_df = run_backtest(
            symbol=symbol,
            start_date=start,
            end_date=end,
            store=store,
        )

    if not result_df.empty:
        # Save results
        out_dir = Path(__file__).resolve().parent.parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"hmm_regime_{symbol}_{end.isoformat()}.csv"
        result_df.to_csv(out_file, index=False)
        logger.info("Results saved to %s", out_file)

        # Show final equity curve stats
        print(f"\nFinal cumulative return: {result_df['cumulative_return'].iloc[-1]:.4f}")
        print(f"Regime distribution:\n{result_df['regime'].value_counts().to_string()}")
    else:
        logger.error("Backtest returned empty results")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# BaseStrategy wrapper for registry integration
# ---------------------------------------------------------------------------

from quantlaxmi.strategies.base import BaseStrategy
from quantlaxmi.strategies.protocol import Signal


class S13HMMRegimeStrategy(BaseStrategy):
    """HMM regime detection strategy — BaseStrategy wrapper for registry."""

    @property
    def strategy_id(self) -> str:
        return "s13_hmm_regime"

    def warmup_days(self) -> int:
        return 120

    def _scan_impl(self, d, store) -> list[Signal]:
        """Research-only strategy — no live signals yet."""
        return []


def create_strategy() -> S13HMMRegimeStrategy:
    """Factory for registry auto-discovery."""
    return S13HMMRegimeStrategy()
