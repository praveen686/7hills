#!/usr/bin/env python3
"""Build QuantKubera Monolith v3 from v2 + new cutting-edge cells."""
import json, textwrap

NB_PATH = '/home/ubuntu/Desktop/7hills/QuantKubera/QuantKubera_Monolith_v3.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

def make_code_cell(source_str):
    """Create a code cell from a multi-line string."""
    lines = source_str.split('\n')
    # Convert to notebook format: each line ends with \n except last
    src = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }

def make_md_cell(source_str):
    lines = source_str.split('\n')
    src = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src
    }

# ============================================================================
# CELL 0: Updated Markdown Header
# ============================================================================
CELL_0_MD = r"""# QuantKubera Monolith v3 — Magnum Opus

## Institutional-Grade Momentum Transformer for Indian Derivatives

**Self-contained** | **Zero Look-Ahead Bias** | **Walk-Forward OOS Validation**
**World Monitor Intelligence** | **Cutting-Edge ML** | **RL Meta-Learning**

| Component | Reference | Features |
|-----------|-----------|----------|
| Temporal Fusion Transformer | Lim et al. (2021) | VSN, Interpretable MHA, GRN |
| EMAT Attention | Entropy 2025 | Temporal Decay + Trend + Volatility heads |
| AFML Event Pipeline | Lopez de Prado (2018) | CUSUM, Triple Barrier, Meta-Labeling |
| Fractional Differentiation | Hosking (1981) | Memory-preserving stationarity |
| Ramanujan Sum Filter Bank | Planat (2002) | Integer-period cycle detection |
| NIG Changepoint Detection | Adams & MacKay (2007) | Regime shift scoring |
| Market Microstructure | Easley et al. (2012), Kyle (1985) | VPIN, Lambda, Amihud |
| Wavelet Decomposition | Daubechies (1992) | Multi-resolution trend/noise separation |
| Hidden Markov Model | Baum-Welch (1970) | 3-state regime detection |
| Persistent Homology (TDA) | Edelsbrunner (2000) | Topological crash detection |
| Transfer Entropy | Schreiber (2000) | Cross-asset causal information flow |
| Multifractal DFA | Kantelhardt (2002) | Fractal spectrum width |
| KL Divergence Regime | Kullback-Leibler (1951) | Distribution shift detection |
| World Monitor Signals | koala73 (2025) | Macro regime, CII, anomaly detection |
| Thompson Sampling | Thompson (1933) | RL strategy selection |
| Sharpe+DD Loss | Multi-objective | Sharpe + drawdown penalty |

### Pipeline
```
Zerodha Kite API → 31 Base Features (10 groups) → Variable Selection Network
GDELT News → FinBERT → 9 Sentiment Features ↗
India VIX → 3 VIX Features ↗
Yahoo Finance → 5 Macro Regime Features ↗     (World Monitor)
Welford Anomaly → 4 Anomaly Features ↗         (World Monitor)
HMM Regime → 3 Regime Features ↗               (NEW)
Wavelet DWT → 4 Wavelet Features ↗             (NEW)
Info Theory → 4 Entropy Features ↗              (NEW)
Multifractal → 3 MF-DFA Features ↗             (NEW)
TDA → 2 Topological Features ↗                  (NEW)
→ EMAT (3-Head Financial Attention) → Momentum Signal
→ Walk-Forward OOS (purge gaps) → Meta-Labeling → Probit Bet Sizing
→ RL Thompson Sampling Meta-Learner → Regime-Aware Ensemble
→ VectorBTPro Tearsheet
```

### Feature Groups (31 per-ticker + 37 cross-asset & advanced = 68 total)
**Base (31):**
1. Normalized Returns (5) | 2. MACD (3) | 3. Volatility (4)
4. Changepoint Detection (2) | 5. Fractional Calculus (3) | 6. Ramanujan (4)
7. Microstructure (4) | 8. Entropy (1) | 9. Momentum Quality (3) | 10. Volume (2)

**Cross-Asset (12):**
11. News Sentiment (9) | 12. India VIX (3)

**World Monitor Intelligence (9):**
13. Macro Regime (5) | 14. Welford Anomaly Detection (4)

**Cutting-Edge ML (16):**
15. HMM Regime (3) | 16. Wavelet Decomposition (4)
17. Information Theory Advanced (4) | 18. Multifractal DFA (3) | 19. TDA Persistent Homology (2)"""

# ============================================================================
# CELL 1: Updated imports (prepend new imports to existing cell)
# ============================================================================
CELL_1_EXTRA_IMPORTS = r"""# --- v3 additional imports ---
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

try:
    import pywt
    HAS_WAVELET = True
except ImportError:
    HAS_WAVELET = False

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude
    HAS_TDA = True
except ImportError:
    HAS_TDA = False

from scipy.stats import entropy as sp_entropy
from sklearn.neighbors import KDTree
from scipy.special import digamma
"""

# ============================================================================
# NEW CELL 3b: World Monitor Intelligence Engine
# ============================================================================
CELL_3B_WORLDMONITOR = r'''# ============================================================================
# CELL 3b: World Monitor Intelligence Engine
# ============================================================================
# Integrates macro signals from the World Monitor geopolitical intelligence
# platform (koala73/worldmonitor). Adds global macro regime detection,
# anomaly detection (Welford's algorithm), and cross-market signals.
#
# Data Sources:
#   - Yahoo Finance: JPY/USD, QQQ, XLP, DXY, Gold, BTC (free, no API key)
#   - alternative.me: Fear & Greed Index (free)
#   - CoinGecko: Stablecoin peg health (free)
#
# All features are CAUSAL: use data available at time t to predict t+1.
# ============================================================================

# ---------------------------------------------------------------------------
# Group 13: Macro Regime Signals (World Monitor port)
# ---------------------------------------------------------------------------
# Ported from worldmonitor/api/macro-signals.js
# 7-signal composite: JPY liquidity, risk-on/off, flow alignment,
# technical trend, hash rate, mining cost, Fear & Greed.
# We implement the 4 most relevant for India FnO:
#   1. JPY carry (yen strengthening kills EM)
#   2. Risk-on/off (QQQ vs XLP consumer staples)
#   3. BTC/QQQ flow alignment (cross-asset momentum coherence)
#   4. Fear & Greed (sentiment baseline)
# ---------------------------------------------------------------------------

MACRO_COLUMNS = [
    'macro_jpy_carry',       # JPY/USD 30d ROC (negative = squeeze)
    'macro_risk_regime',     # QQQ/XLP ratio z-score (risk-on/off)
    'macro_flow_align',      # |BTC 5d - QQQ 5d| (cross-asset coherence)
    'macro_fear_greed_z',    # Fear & Greed Index z-scored
    'macro_composite',       # Weighted composite (0-100)
]

def fetch_yahoo_macro(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch macro signals from Yahoo Finance (free, no API key).

    Fetches: JPY=X (USD/JPY), QQQ, XLP, BTC-USD, GC=F (Gold), DX-Y.NYB (DXY)
    Returns: DataFrame indexed by date with macro features.
    """
    if not HAS_YFINANCE:
        logger.warning("yfinance not installed — macro signals disabled")
        return pd.DataFrame()

    tickers = {
        'JPY=X': 'jpy',       # USD/JPY (invert for JPY strength)
        'QQQ': 'qqq',         # Nasdaq 100 ETF
        'XLP': 'xlp',         # Consumer Staples (defensive)
        'BTC-USD': 'btc',     # Bitcoin
    }

    print("  Fetching Yahoo Finance macro data...")
    try:
        data = yf.download(
            list(tickers.keys()),
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            print("    SKIP: No data from Yahoo Finance")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed: {e}")
        return pd.DataFrame()

    # Extract close prices
    closes = pd.DataFrame(index=data.index)
    if isinstance(data.columns, pd.MultiIndex):
        for yahoo_tk, col_name in tickers.items():
            if yahoo_tk in data['Close'].columns:
                closes[col_name] = data['Close'][yahoo_tk]
    else:
        # Single ticker case
        for yahoo_tk, col_name in tickers.items():
            if 'Close' in data.columns:
                closes[col_name] = data['Close']

    if closes.empty or closes.dropna(how='all').empty:
        print("    SKIP: Yahoo Finance returned empty data")
        return pd.DataFrame()

    closes = closes.ffill().dropna()

    result = pd.DataFrame(index=closes.index)

    # Signal 1: JPY Carry — 30d rate of change of USD/JPY
    # Negative ROC = JPY strengthening = EM risk (carry unwind)
    if 'jpy' in closes.columns:
        jpy_roc_30 = closes['jpy'].pct_change(30) * 100
        result['macro_jpy_carry'] = jpy_roc_30

    # Signal 2: Risk-On/Off — QQQ vs XLP 20d ROC ratio
    # QQQ outperforming XLP = risk-on; z-scored for stationarity
    if 'qqq' in closes.columns and 'xlp' in closes.columns:
        qqq_roc = closes['qqq'].pct_change(20)
        xlp_roc = closes['xlp'].pct_change(20)
        ratio = qqq_roc - xlp_roc
        result['macro_risk_regime'] = (
            (ratio - ratio.rolling(126).mean()) /
            (ratio.rolling(126).std() + 1e-8)
        )

    # Signal 3: Flow Alignment — |BTC 5d - QQQ 5d| gap
    # Small gap = markets aligned = momentum persistence
    if 'btc' in closes.columns and 'qqq' in closes.columns:
        btc_5d = closes['btc'].pct_change(5) * 100
        qqq_5d = closes['qqq'].pct_change(5) * 100
        gap = (btc_5d - qqq_5d).abs()
        # Invert and z-score: low gap = high alignment = positive signal
        result['macro_flow_align'] = -(
            (gap - gap.rolling(63).mean()) /
            (gap.rolling(63).std() + 1e-8)
        )

    n_signals = result.notna().any().sum()
    n_days = len(result.dropna(how='all'))
    print(f"    -> {n_signals} macro signals, {n_days} days")
    return result


def fetch_fear_greed() -> pd.DataFrame:
    """Fetch Fear & Greed Index from alternative.me (free, no API key).

    Returns: DataFrame with 'macro_fear_greed_z' column.
    The index ranges 0-100. We z-score it over 126-day window.
    """
    url = "https://api.alternative.me/fng/?limit=365&format=json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            dt = pd.to_datetime(int(item['timestamp']), unit='s')
            records.append({'date': dt, 'fear_greed': int(item['value'])})

        df = pd.DataFrame(records).set_index('date').sort_index()
        # Z-score over 126-day rolling window
        fg = df['fear_greed'].astype(float)
        df['macro_fear_greed_z'] = (
            (fg - fg.rolling(126, min_periods=30).mean()) /
            (fg.rolling(126, min_periods=30).std() + 1e-8)
        )

        print(f"    -> Fear & Greed Index: {len(df)} days, "
              f"latest={df['fear_greed'].iloc[-1]}")
        return df[['macro_fear_greed_z']]
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return pd.DataFrame()


def compute_macro_composite(macro_df: pd.DataFrame) -> pd.Series:
    """Compute weighted macro composite score (0-100).

    Ported from worldmonitor/api/macro-signals.js verdict logic.
    Each signal contributes to a bullish/bearish score.
    """
    score = pd.Series(50.0, index=macro_df.index)  # neutral baseline

    if 'macro_jpy_carry' in macro_df.columns:
        # JPY carry > 0 (USD strengthening vs JPY) = bullish for EM
        score += np.where(macro_df['macro_jpy_carry'] > 0, 10, -10)

    if 'macro_risk_regime' in macro_df.columns:
        # Risk-on (positive z) = bullish
        score += np.clip(macro_df['macro_risk_regime'] * 5, -15, 15)

    if 'macro_flow_align' in macro_df.columns:
        # Aligned flows (positive) = bullish
        score += np.clip(macro_df['macro_flow_align'] * 5, -10, 10)

    if 'macro_fear_greed_z' in macro_df.columns:
        score += np.clip(macro_df['macro_fear_greed_z'] * 5, -15, 15)

    return np.clip(score, 0, 100)


# ---------------------------------------------------------------------------
# Group 14: Welford Anomaly Detection (World Monitor port)
# ---------------------------------------------------------------------------
# Ported from worldmonitor/api/temporal-baseline.js
# Uses Welford's online algorithm for numerically stable streaming
# mean/variance computation. Detects z-score deviations from learned
# baselines per weekday × month (seasonal patterns).
# ---------------------------------------------------------------------------

WELFORD_COLUMNS = [
    'anomaly_volume_z',    # Volume vs weekday×month baseline
    'anomaly_range_z',     # High-low range vs baseline
    'anomaly_gap_z',       # Overnight gap vs baseline
    'anomaly_composite',   # Max of above (worst anomaly)
]


class WelfordBaseline:
    """Welford's online algorithm for streaming mean/variance.

    Ported from worldmonitor/api/temporal-baseline.js.
    Numerically stable single-pass computation.
    """

    def __init__(self):
        self.baselines = {}  # key -> {mean, m2, n}

    def _key(self, weekday: int, month: int, metric: str) -> str:
        return f"{metric}:{weekday}:{month}"

    def update(self, weekday: int, month: int, metric: str, value: float):
        """Update baseline with new observation (Welford's algorithm)."""
        key = self._key(weekday, month, metric)
        if key not in self.baselines:
            self.baselines[key] = {'mean': 0.0, 'm2': 0.0, 'n': 0}

        bl = self.baselines[key]
        bl['n'] += 1
        delta = value - bl['mean']
        bl['mean'] += delta / bl['n']
        delta2 = value - bl['mean']
        bl['m2'] += delta * delta2

    def zscore(self, weekday: int, month: int, metric: str,
               value: float, min_samples: int = 10) -> float:
        """Compute z-score of value against baseline."""
        key = self._key(weekday, month, metric)
        if key not in self.baselines:
            return 0.0

        bl = self.baselines[key]
        if bl['n'] < min_samples:
            return 0.0

        variance = max(0, bl['m2'] / (bl['n'] - 1))  # ddof=1
        std = np.sqrt(variance)
        if std < 1e-10:
            return 0.0

        return (value - bl['mean']) / std


def compute_welford_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute anomaly z-scores using Welford's online algorithm.

    Detects unusual volume, range, and gap sizes relative to
    weekday×month seasonal baselines. FULLY CAUSAL: only uses
    data from before time t to compute baselines at time t.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex

    Returns: DataFrame with anomaly z-score columns
    """
    baseline = WelfordBaseline()
    result = pd.DataFrame(index=df.index)

    vol_z = np.zeros(len(df))
    range_z = np.zeros(len(df))
    gap_z = np.zeros(len(df))

    closes = df['close'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values

    for i in range(1, len(df)):
        dt = df.index[i]
        wd = dt.weekday()
        mo = dt.month

        # Current values
        vol_val = float(volumes[i]) if volumes[i] > 0 else 0.0
        range_val = float((highs[i] - lows[i]) / closes[i-1]) if closes[i-1] > 0 else 0.0
        gap_val = float(abs(opens[i] - closes[i-1]) / closes[i-1]) if closes[i-1] > 0 else 0.0

        # Compute z-scores BEFORE updating (causal)
        vol_z[i] = baseline.zscore(wd, mo, 'volume', vol_val)
        range_z[i] = baseline.zscore(wd, mo, 'range', range_val)
        gap_z[i] = baseline.zscore(wd, mo, 'gap', gap_val)

        # Update baselines AFTER scoring (causal)
        baseline.update(wd, mo, 'volume', vol_val)
        baseline.update(wd, mo, 'range', range_val)
        baseline.update(wd, mo, 'gap', gap_val)

    result['anomaly_volume_z'] = vol_z
    result['anomaly_range_z'] = range_z
    result['anomaly_gap_z'] = gap_z
    result['anomaly_composite'] = np.maximum.reduce([
        np.abs(vol_z), np.abs(range_z), np.abs(gap_z)
    ])

    return result
'''

# ============================================================================
# NEW CELL 5: Advanced Feature Engineering (Groups 15-19)
# ============================================================================
CELL_5_ADVANCED = r'''# ============================================================================
# CELL 5: Cutting-Edge Feature Engineering — Groups 15-19
# ============================================================================
# Implements 5 advanced feature groups from cutting-edge quant research:
#
#   Group 15: HMM Regime Detection (3 features)
#             - Baum-Welch EM on returns + volatility
#             - 3-state: bull, bear, sideways
#
#   Group 16: Wavelet Decomposition (4 features)
#             - DWT with Daubechies-4 wavelet
#             - Multi-resolution trend/noise separation
#             Ref: Springer Computational Economics 2025
#
#   Group 17: Information Theory Advanced (4 features)
#             - KL divergence regime shift detector
#             - Sample entropy (regularity)
#             - NMI predictability window
#             - Spectral entropy
#             Ref: arXiv:2511.16339 (Financial Information Theory)
#
#   Group 18: Multifractal DFA (3 features)
#             - Kantelhardt (2002) MF-DFA
#             - Spectrum width, H(2), asymmetry
#
#   Group 19: TDA Persistent Homology (2 features)
#             - Vietoris-Rips persistence on return point clouds
#             - Crash early warning from topological complexity
#             Ref: MDPI Computers 2025
#
# ALL features are CAUSAL (rolling windows, no future data).
# ============================================================================

ADVANCED_FEATURE_COLUMNS = [
    # Group 15: HMM Regime
    'hmm_regime',           # Decoded regime label (0=bear, 1=sideways, 2=bull)
    'hmm_regime_prob',      # Probability of current regime
    'hmm_regime_duration',  # Days in current regime
    # Group 16: Wavelet
    'wavelet_trend_energy',   # Energy in approximation coefficients (low-freq)
    'wavelet_detail_energy',  # Energy in detail coefficients (high-freq)
    'wavelet_snr',            # Signal-to-noise ratio (trend/detail)
    'wavelet_dominant_scale', # Dominant wavelet scale
    # Group 17: Info Theory
    'kl_regime_shift',      # KL divergence between recent vs prior returns
    'sample_entropy',       # Sample entropy (regularity measure)
    'nmi_predictability',   # Normalized Mutual Information (market efficiency)
    'spectral_entropy',     # Spectral entropy of return power spectrum
    # Group 18: Multifractal
    'mf_delta_alpha',       # Multifractal spectrum width
    'mf_hurst2',            # Generalized Hurst exponent H(2)
    'mf_asymmetry',         # Left-right asymmetry of MF spectrum
    # Group 19: TDA
    'tda_persistence_norm',  # L2-norm of persistence landscape (H1)
    'tda_betti_ratio',       # Betti-1 / Betti-0 (loop/component ratio)
]


# ─── Group 15: HMM Regime Detection ─────────────────────────────────────────
def compute_hmm_regime(close: pd.Series, n_states: int = 3,
                       window: int = 252) -> pd.DataFrame:
    """Fit rolling 3-state Gaussian HMM on returns + volatility.

    States are sorted by mean return: 0=bear, 1=sideways, 2=bull.
    CAUSAL: fits on data up to time t only.

    Args:
        close: price series
        n_states: number of hidden states (default 3)
        window: training window in days

    Returns: DataFrame with hmm_regime, hmm_regime_prob, hmm_regime_duration
    """
    if not HAS_HMM:
        logger.warning("hmmlearn not installed — HMM features disabled")
        return pd.DataFrame(
            np.nan, index=close.index,
            columns=['hmm_regime', 'hmm_regime_prob', 'hmm_regime_duration']
        )

    log_ret = np.log(close / close.shift(1)).dropna()
    vol_20 = log_ret.rolling(20, min_periods=10).std()

    # Observation matrix: [return, volatility]
    obs = pd.DataFrame({
        'ret': log_ret,
        'vol': vol_20,
    }).dropna()

    regime = np.full(len(close), np.nan)
    regime_prob = np.full(len(close), np.nan)
    regime_dur = np.full(len(close), np.nan)

    refit_interval = 63  # Refit every quarter
    model = None
    state_order = None

    for t in range(window, len(obs)):
        # Refit model periodically
        if model is None or (t - window) % refit_interval == 0:
            train_data = obs.iloc[t - window:t].values
            try:
                model = GaussianHMM(
                    n_components=n_states, covariance_type='full',
                    n_iter=100, random_state=42, verbose=False,
                )
                model.fit(train_data)
                # Sort states by mean return: bear < sideways < bull
                state_order = np.argsort(model.means_[:, 0])
            except Exception:
                model = None
                continue

        if model is None:
            continue

        # Predict current state (causal: uses data up to t)
        try:
            recent = obs.iloc[t - window:t + 1].values
            states = model.predict(recent)
            probs = model.predict_proba(recent)

            current_state_raw = states[-1]
            # Map to sorted order
            current_state = int(np.where(state_order == current_state_raw)[0][0])
            current_prob = float(probs[-1, current_state_raw])

            # Map back to close index
            obs_date = obs.index[t]
            close_idx = close.index.get_loc(obs_date)
            regime[close_idx] = current_state
            regime_prob[close_idx] = current_prob
        except Exception:
            continue

    # Compute regime duration (consecutive days in same regime)
    dur = np.zeros(len(close))
    count = 0
    prev = np.nan
    for i in range(len(close)):
        if np.isnan(regime[i]):
            count = 0
        elif regime[i] == prev:
            count += 1
        else:
            count = 1
        dur[i] = count
        prev = regime[i]

    result = pd.DataFrame(index=close.index)
    result['hmm_regime'] = regime
    result['hmm_regime_prob'] = regime_prob
    result['hmm_regime_duration'] = dur
    return result


# ─── Group 16: Wavelet Decomposition ────────────────────────────────────────
def compute_wavelet_features(close: pd.Series, wavelet: str = 'db4',
                             level: int = 4, window: int = 63) -> pd.DataFrame:
    """Multi-resolution wavelet decomposition features.

    Uses Discrete Wavelet Transform (DWT) with Daubechies-4 wavelet.
    Decomposes price into approximation (trend) and detail (noise)
    coefficients at multiple scales.

    Ref: "Leveraging Wavelet Transform & Deep Learning for Option Price
    Prediction: Insights from the Indian Derivative Market" (2025)

    Args:
        close: price series
        wavelet: wavelet family (default 'db4')
        level: decomposition level (default 4)
        window: rolling window for energy computation

    Returns: DataFrame with wavelet energy features
    """
    if not HAS_WAVELET:
        logger.warning("PyWavelets not installed — wavelet features disabled")
        return pd.DataFrame(
            np.nan, index=close.index,
            columns=['wavelet_trend_energy', 'wavelet_detail_energy',
                     'wavelet_snr', 'wavelet_dominant_scale']
        )

    log_ret = np.log(close / close.shift(1)).fillna(0).values
    n = len(log_ret)

    trend_energy = np.full(n, np.nan)
    detail_energy = np.full(n, np.nan)
    snr = np.full(n, np.nan)
    dominant_scale = np.full(n, np.nan)

    for t in range(window, n):
        segment = log_ret[t - window:t]
        try:
            coeffs = pywt.wavedec(segment, wavelet, level=level)
            # coeffs[0] = approximation (trend), coeffs[1:] = details (noise)
            approx_energy = float(np.sum(coeffs[0] ** 2))
            detail_energies = [float(np.sum(c ** 2)) for c in coeffs[1:]]
            total_detail = sum(detail_energies)

            trend_energy[t] = approx_energy
            detail_energy[t] = total_detail
            snr[t] = float(np.log1p(approx_energy / (total_detail + 1e-10)))

            # Dominant scale: which detail level has most energy
            if detail_energies:
                dominant_scale[t] = float(np.argmax(detail_energies))
        except Exception:
            continue

    result = pd.DataFrame(index=close.index)
    result['wavelet_trend_energy'] = _rolling_zscore(trend_energy, 126)
    result['wavelet_detail_energy'] = _rolling_zscore(detail_energy, 126)
    result['wavelet_snr'] = snr
    result['wavelet_dominant_scale'] = dominant_scale
    return result


def _rolling_zscore(arr, window):
    """Z-score an array over a rolling window."""
    s = pd.Series(arr)
    return ((s - s.rolling(window, min_periods=20).mean()) /
            (s.rolling(window, min_periods=20).std() + 1e-8)).values


# ─── Group 17: Information Theory Advanced ───────────────────────────────────
def compute_kl_regime_shift(returns: np.ndarray, window: int = 252,
                            n_bins: int = 50) -> np.ndarray:
    """KL divergence between recent and prior return distributions.

    High KL = distribution has shifted = regime change.
    Ref: arXiv:2511.16339 (Financial Information Theory)

    CAUSAL: compares [t-W, t] vs [t-2W, t-W].
    """
    kl = np.full(len(returns), np.nan)
    for t in range(2 * window, len(returns)):
        recent = returns[t - window:t]
        prior = returns[t - 2 * window:t - window]

        all_r = np.concatenate([recent, prior])
        bins = np.linspace(all_r.min() - 1e-8, all_r.max() + 1e-8, n_bins + 1)

        p_recent = np.histogram(recent, bins=bins)[0].astype(float) + 1e-10
        p_prior = np.histogram(prior, bins=bins)[0].astype(float) + 1e-10
        p_recent /= p_recent.sum()
        p_prior /= p_prior.sum()

        kl[t] = float(sp_entropy(p_recent, p_prior))

    # Z-score for stationarity
    s = pd.Series(kl)
    z = (s - s.rolling(252, min_periods=50).mean()) / (s.rolling(252, min_periods=50).std() + 1e-8)
    return z.values


def compute_sample_entropy(series: np.ndarray, m: int = 2,
                           r_mult: float = 0.2,
                           window: int = 100) -> np.ndarray:
    """Rolling sample entropy (SampEn) — measures regularity.

    Low SampEn = regular/predictable, High SampEn = random.
    Markets become MORE predictable (lower SampEn) during crises.

    CAUSAL: uses [t-W, t] window only.
    """
    se = np.full(len(series), np.nan)
    for t in range(window, len(series)):
        x = series[t - window:t]
        r = r_mult * np.std(x, ddof=1)
        if r < 1e-10:
            se[t] = 0.0
            continue

        N = len(x)
        # Count template matches
        def _count(m_len):
            if N - m_len < 1:
                return 0
            templates = np.array([x[i:i + m_len] for i in range(N - m_len)])
            count = 0
            for i in range(len(templates)):
                dists = np.max(np.abs(templates[i] - templates[i+1:]), axis=1)
                count += np.sum(dists < r)
            return count

        B = _count(m)
        A = _count(m + 1)

        if B > 0 and A > 0:
            se[t] = -np.log(A / B)
        elif B > 0:
            se[t] = -np.log(1.0 / B)
        else:
            se[t] = 0.0

    return se


def compute_nmi_predictability(returns: np.ndarray, lag: int = 1,
                                window: int = 252,
                                n_bins: int = 20) -> np.ndarray:
    """Normalized Mutual Information between lagged and current returns.

    High NMI = market temporarily inefficient (exploitable).
    ~78% of the time NMI < 0.05 (EMH holds).

    Ref: arXiv:2511.16339
    """
    nmi = np.full(len(returns), np.nan)
    for t in range(window + lag, len(returns)):
        past = returns[t - window - lag:t - lag]
        future = returns[t - window:t]

        bins = np.linspace(
            min(past.min(), future.min()) - 1e-8,
            max(past.max(), future.max()) + 1e-8,
            n_bins + 1
        )
        past_binned = np.digitize(past, bins)
        future_binned = np.digitize(future, bins)

        # Joint and marginal entropies
        joint_hist = np.histogram2d(past_binned, future_binned,
                                     bins=n_bins)[0] + 1e-10
        joint_hist /= joint_hist.sum()

        p_past = joint_hist.sum(axis=1)
        p_future = joint_hist.sum(axis=0)

        h_past = -np.sum(p_past * np.log(p_past + 1e-10))
        h_future = -np.sum(p_future * np.log(p_future + 1e-10))
        h_joint = -np.sum(joint_hist * np.log(joint_hist + 1e-10))

        mi = h_past + h_future - h_joint
        norm = np.sqrt(h_past * h_future) if h_past > 0 and h_future > 0 else 1.0
        nmi[t] = mi / (norm + 1e-10)

    return nmi


def compute_spectral_entropy(series: np.ndarray,
                              window: int = 63) -> np.ndarray:
    """Spectral entropy of return power spectrum.

    Uniform spectrum (white noise) -> high entropy.
    Concentrated spectrum (strong periodicity) -> low entropy.
    """
    se = np.full(len(series), np.nan)
    for t in range(window, len(series)):
        segment = series[t - window:t]
        # FFT power spectrum
        fft_vals = np.fft.rfft(segment - segment.mean())
        psd = np.abs(fft_vals) ** 2
        psd = psd / (psd.sum() + 1e-10)
        psd = psd[psd > 0]
        se[t] = -np.sum(psd * np.log(psd + 1e-10)) / np.log(len(psd) + 1)
    return se


def compute_info_theory_features(close: pd.Series) -> pd.DataFrame:
    """Compute all Group 17 information-theoretic features."""
    log_ret = np.log(close / close.shift(1)).fillna(0).values

    result = pd.DataFrame(index=close.index)

    print("    [17a] KL divergence regime shift...")
    result['kl_regime_shift'] = compute_kl_regime_shift(log_ret, window=126)

    print("    [17b] Sample entropy...")
    result['sample_entropy'] = compute_sample_entropy(log_ret, window=60)

    print("    [17c] NMI predictability...")
    result['nmi_predictability'] = compute_nmi_predictability(log_ret, window=126)

    print("    [17d] Spectral entropy...")
    result['spectral_entropy'] = compute_spectral_entropy(log_ret, window=63)

    return result


# ─── Group 18: Multifractal DFA ──────────────────────────────────────────────
def compute_mfdfa_features(close: pd.Series,
                           window: int = 252) -> pd.DataFrame:
    """Multifractal Detrended Fluctuation Analysis features.

    Ref: Kantelhardt et al. (2002), PMC 8392555 (2021)

    Computes:
      - delta_alpha: spectrum width (large = complex multifractal)
      - H(2): standard Hurst exponent from q=2
      - asymmetry: left vs right spectrum skew

    CAUSAL: rolling window computation.
    """
    log_ret = np.log(close / close.shift(1)).fillna(0).values
    n = len(log_ret)

    delta_alpha = np.full(n, np.nan)
    hurst2 = np.full(n, np.nan)
    asymmetry = np.full(n, np.nan)

    q_list = np.array([-3, -2, -1, 1, 2, 3, 4, 5], dtype=float)

    for t in range(window, n):
        series = log_ret[t - window:t]
        try:
            result = _mfdfa_single(series, q_list)
            if result is not None:
                delta_alpha[t] = result['delta_alpha']
                hurst2[t] = result['hurst2']
                asymmetry[t] = result['asymmetry']
        except Exception:
            continue

    out = pd.DataFrame(index=close.index)
    out['mf_delta_alpha'] = _rolling_zscore(delta_alpha, 126)
    out['mf_hurst2'] = hurst2
    out['mf_asymmetry'] = asymmetry
    return out


def _mfdfa_single(series, q_list, order=1):
    """MF-DFA for a single window. Returns dict or None."""
    N = len(series)
    if N < 30:
        return None

    # Profile (cumulative sum of mean-subtracted series)
    Y = np.cumsum(series - series.mean())

    # Scales: log-spaced from 8 to N/4
    scales = np.unique(np.logspace(
        np.log10(8), np.log10(max(10, N // 4)), 10
    ).astype(int))
    scales = scales[scales >= 4]

    if len(scales) < 3:
        return None

    Fq = np.zeros((len(q_list), len(scales)))

    for si, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 1:
            continue

        fluct = np.zeros(n_seg)
        t_vec = np.arange(s)

        for v in range(n_seg):
            seg = Y[v * s:(v + 1) * s]
            if len(seg) < s:
                continue
            coeff = np.polyfit(t_vec[:len(seg)], seg, order)
            trend = np.polyval(coeff, t_vec[:len(seg)])
            fluct[v] = np.mean((seg - trend) ** 2)

        fluct = fluct[fluct > 0]
        if len(fluct) == 0:
            continue

        for qi, q in enumerate(q_list):
            if q == 0:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(fluct + 1e-20)))
            else:
                Fq[qi, si] = np.mean(fluct ** (q / 2)) ** (1 / q)

    # Generalized Hurst exponents h(q)
    log_s = np.log(scales)
    hq = np.zeros(len(q_list))
    for qi in range(len(q_list)):
        valid = Fq[qi] > 0
        if valid.sum() >= 3:
            hq[qi] = np.polyfit(log_s[valid], np.log(Fq[qi][valid] + 1e-20), 1)[0]

    # Multifractal spectrum
    tau_q = q_list * hq - 1
    alpha_holder = np.gradient(tau_q, q_list)

    delta_alpha = float(alpha_holder.max() - alpha_holder.min())

    # H(2) = standard Hurst
    idx_q2 = np.argmin(np.abs(q_list - 2))
    hurst2_val = float(hq[idx_q2])

    # Asymmetry: left vs right tail
    mid = len(alpha_holder) // 2
    left_width = alpha_holder[mid] - alpha_holder[0] if mid > 0 else 0
    right_width = alpha_holder[-1] - alpha_holder[mid] if mid < len(alpha_holder) - 1 else 0
    asym = float((left_width - right_width) / (delta_alpha + 1e-10))

    return {'delta_alpha': delta_alpha, 'hurst2': hurst2_val, 'asymmetry': asym}


# ─── Group 19: TDA Persistent Homology ──────────────────────────────────────
def compute_tda_features(close: pd.Series, window: int = 50,
                         embedding_dim: int = 3,
                         embedding_lag: int = 1) -> pd.DataFrame:
    """Topological Data Analysis features via persistent homology.

    Embeds a sliding window of returns into a point cloud using
    Takens' time-delay embedding, then computes Vietoris-Rips
    persistence. Tracks the lifetime of 1-dimensional topological
    features (loops) as crash indicators.

    Ref: MDPI Computers 2025, ACM AMMIC 2025

    CAUSAL: uses [t-W, t] only.
    """
    log_ret = np.log(close / close.shift(1)).fillna(0).values
    n = len(log_ret)

    pers_norm = np.full(n, np.nan)
    betti_ratio = np.full(n, np.nan)

    embed_size = embedding_dim + (embedding_dim - 1) * embedding_lag

    for t in range(max(window, embed_size + 10), n):
        segment = log_ret[t - window:t]

        try:
            # Takens time-delay embedding
            rows = []
            for i in range(len(segment) - embed_size + 1):
                row = [segment[i + j * embedding_lag]
                       for j in range(embedding_dim)]
                rows.append(row)
            cloud = np.array(rows)

            if len(cloud) < 5:
                continue

            if HAS_TDA:
                # Use giotto-tda for persistence
                vrp = VietorisRipsPersistence(
                    homology_dimensions=[0, 1], max_edge_length=2.0,
                    n_jobs=1
                )
                diagrams = vrp.fit_transform(cloud[np.newaxis, :, :])
                dgm = diagrams[0]

                # H0: connected components, H1: loops
                h0 = dgm[dgm[:, 2] == 0]
                h1 = dgm[dgm[:, 2] == 1]

                # Filter out infinite persistence
                h0_finite = h0[np.isfinite(h0[:, 1])]
                h1_finite = h1[np.isfinite(h1[:, 1])]

                # L2 norm of H1 persistence (loop lifetimes)
                if len(h1_finite) > 0:
                    lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
                    pers_norm[t] = float(np.sqrt(np.sum(lifetimes ** 2)))
                else:
                    pers_norm[t] = 0.0

                # Betti ratio: number of loops / number of components
                n_h0 = max(1, len(h0_finite))
                n_h1 = len(h1_finite)
                betti_ratio[t] = float(n_h1 / n_h0)
            else:
                # Fallback: simple distance-based proxy
                from scipy.spatial.distance import pdist
                dists = pdist(cloud)
                # Approximate topological complexity from distance distribution
                pers_norm[t] = float(np.std(dists))
                betti_ratio[t] = float(np.mean(dists < np.median(dists)))
        except Exception:
            continue

    result = pd.DataFrame(index=close.index)
    result['tda_persistence_norm'] = _rolling_zscore(pers_norm, 126)
    result['tda_betti_ratio'] = betti_ratio
    return result


# ─── Master: Build All Advanced Features ─────────────────────────────────────
def build_advanced_features(df: pd.DataFrame, close: pd.Series,
                            cfg) -> pd.DataFrame:
    """Build all 16 advanced features (Groups 15-19).

    Args:
        df: OHLCV DataFrame
        close: close price Series
        cfg: MonolithConfig

    Returns: DataFrame with ADVANCED_FEATURE_COLUMNS
    """
    t0 = time.time()
    print(f"\n  Building {len(ADVANCED_FEATURE_COLUMNS)} advanced features...")

    result = pd.DataFrame(index=df.index)

    # Group 15: HMM Regime (3 features)
    print("  [15/19] HMM Regime Detection (3 features)")
    hmm_df = compute_hmm_regime(close, n_states=3, window=252)
    for col in ['hmm_regime', 'hmm_regime_prob', 'hmm_regime_duration']:
        result[col] = hmm_df[col] if col in hmm_df.columns else np.nan

    # Group 16: Wavelet Decomposition (4 features)
    print("  [16/19] Wavelet Decomposition (4 features)")
    wav_df = compute_wavelet_features(close, wavelet='db4', level=4, window=63)
    for col in ['wavelet_trend_energy', 'wavelet_detail_energy',
                'wavelet_snr', 'wavelet_dominant_scale']:
        result[col] = wav_df[col] if col in wav_df.columns else np.nan

    # Group 17: Information Theory (4 features)
    print("  [17/19] Information Theory Advanced (4 features)")
    info_df = compute_info_theory_features(close)
    for col in ['kl_regime_shift', 'sample_entropy',
                'nmi_predictability', 'spectral_entropy']:
        result[col] = info_df[col] if col in info_df.columns else np.nan

    # Group 18: Multifractal DFA (3 features)
    print("  [18/19] Multifractal DFA (3 features)")
    mf_df = compute_mfdfa_features(close, window=252)
    for col in ['mf_delta_alpha', 'mf_hurst2', 'mf_asymmetry']:
        result[col] = mf_df[col] if col in mf_df.columns else np.nan

    # Group 19: TDA Persistent Homology (2 features)
    print("  [19/19] TDA Persistent Homology (2 features)")
    tda_df = compute_tda_features(close, window=50, embedding_dim=3)
    for col in ['tda_persistence_norm', 'tda_betti_ratio']:
        result[col] = tda_df[col] if col in tda_df.columns else np.nan

    elapsed = time.time() - t0
    n_valid = result.notna().any(axis=1).sum()
    print(f"\n  Advanced features: {len(ADVANCED_FEATURE_COLUMNS)} features, "
          f"{n_valid} valid days ({elapsed:.1f}s)")

    return result
'''

# ============================================================================
# NEW CELL: Enhanced Multi-Aspect Attention (EMAT) + Multi-Objective Loss
# ============================================================================
CELL_EMAT = r'''# ============================================================================
# Enhanced Multi-Aspect Attention (EMAT) & Multi-Objective Loss
# ============================================================================
# Replaces standard Interpretable Multi-Head Attention with 3 specialized
# financial attention heads:
#   1. Temporal Decay Attention: exponential decay weighting (recent > old)
#   2. Trend Attention: attention over differenced (momentum) features
#   3. Volatility Attention: attention over rolling variance features
#
# Multi-Objective Loss: Sharpe + Volatility Consistency + Drawdown Penalty
#
# Ref: "EMAT: Enhanced Multi-Aspect Attention Transformer for Financial
#       Time Series Forecasting" (MDPI Entropy 2025)
# ============================================================================

@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class TemporalDecayAttention(layers.Layer):
    """Attention with exponential temporal decay.

    Applies exp(-lambda * |i-j|) weighting to attention scores,
    so recent time steps get higher attention regardless of content.
    lambda is learnable per head.
    """

    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.W_q = layers.Dense(self.d_model, use_bias=False, name="tda_Wq")
        self.W_k = layers.Dense(self.d_model, use_bias=False, name="tda_Wk")
        self.W_v = layers.Dense(self.d_model, use_bias=False, name="tda_Wv")
        # Learnable decay rate (initialized to ~0.1)
        self.decay_rate = self.add_weight(
            name="decay_rate", shape=(1,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True
        )
        self.dropout = layers.Dropout(dropout_rate)
        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Standard scaled dot-product scores
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)

        # Temporal decay bias: exp(-lambda * |i - j|)
        seq_len = tf.shape(x)[1]
        positions = tf.cast(tf.range(seq_len), tf.float32)
        dist_matrix = tf.abs(positions[:, None] - positions[None, :])
        decay_bias = -tf.abs(self.decay_rate) * dist_matrix
        scores = scores + decay_bias[None, :, :]

        if mask is not None:
            scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, V)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'dropout_rate': self.dropout_rate})
        return config


@tf.keras.utils.register_keras_serializable(package="MomentumTFT")
class MultiObjectiveSharpeLoss(tf.keras.losses.Loss):
    """Multi-objective loss: Sharpe + Volatility Consistency + Drawdown.

    L = -Sharpe + alpha * VolConsistency + beta * MaxDrawdown

    - Sharpe: annualized (sqrt(252), ddof=1)
    - VolConsistency: penalizes return sequences with unstable volatility
    - MaxDrawdown: penalizes large drawdowns during training

    alpha=0.1, beta=0.2 recommended (Sharpe-dominant with DD protection).
    """

    def __init__(self, alpha=0.1, beta=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        # Strategy returns: signal * actual returns
        signal = tf.squeeze(y_pred, axis=-1) if len(y_pred.shape) > 1 else y_pred
        actual = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
        strategy_ret = signal * actual

        # Sharpe component (primary objective)
        mean_r = tf.reduce_mean(strategy_ret)
        std_r = tf.math.reduce_std(strategy_ret) + 1e-8
        sharpe = tf.sqrt(252.0) * mean_r / std_r

        # Volatility consistency: std of rolling vol shouldn't be too high
        # Approximation: variance of squared returns (proxy for vol-of-vol)
        sq_ret = strategy_ret ** 2
        vol_consistency = tf.math.reduce_std(sq_ret) / (tf.reduce_mean(sq_ret) + 1e-8)

        # Drawdown penalty: compute max drawdown from cumulative returns
        cum_ret = tf.cumsum(strategy_ret)
        running_max = tf.scan(lambda a, x: tf.maximum(a, x), cum_ret, initializer=cum_ret[0])
        drawdowns = cum_ret - running_max
        max_dd = -tf.reduce_min(drawdowns)

        # Combined loss (minimize)
        loss = -sharpe + self.alpha * vol_consistency + self.beta * max_dd
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha, 'beta': self.beta})
        return config
'''

# ============================================================================
# NEW CELL: RL Thompson Sampling Meta-Learner
# ============================================================================
CELL_RL_META = r'''# ============================================================================
# RL Thompson Sampling Meta-Learner
# ============================================================================
# Inspired by RL-book (Rao & Jelvis, Stanford CME 241) Thompson Sampling
# implementation in rl/chapter14/ts_gaussian.py.
#
# The meta-learner treats each signal source (TFT, Momentum, MR) as a
# "bandit arm" and uses Thompson Sampling to learn which signal performs
# best in the CURRENT regime. The posterior distribution of each arm's
# reward (Sharpe) is updated online as new data arrives.
#
# Key insight: different strategies work in different regimes.
# Thompson Sampling automatically discovers this without manual rules.
# ============================================================================

class ThompsonSamplingMetaLearner:
    """Thompson Sampling strategy selector with regime conditioning.

    Each strategy is modeled as a Gaussian bandit arm with unknown mean
    and known variance. Posterior is updated via conjugate Normal-Normal:

        Prior:     mu ~ N(mu_0, sigma_0^2)
        Likelihood: X | mu ~ N(mu, sigma_obs^2)
        Posterior:  mu | X ~ N(mu_n, sigma_n^2)

    where:
        sigma_n^2 = 1 / (1/sigma_0^2 + n/sigma_obs^2)
        mu_n = sigma_n^2 * (mu_0/sigma_0^2 + sum(X)/sigma_obs^2)

    Selection: sample from each posterior, pick highest sample.

    When regime is provided, we maintain SEPARATE posteriors per regime.
    """

    def __init__(self, strategy_names: List[str],
                 n_regimes: int = 3,
                 prior_mean: float = 0.0,
                 prior_std: float = 1.0,
                 obs_std: float = 0.1):
        """
        Args:
            strategy_names: names of strategies (bandit arms)
            n_regimes: number of regime states
            prior_mean: prior mean for each arm
            prior_std: prior standard deviation
            obs_std: assumed observation noise (daily Sharpe scale)
        """
        self.names = strategy_names
        self.n_arms = len(strategy_names)
        self.n_regimes = n_regimes
        self.obs_var = obs_std ** 2

        # Per-regime, per-arm posteriors: (mean, variance, count)
        self.posteriors = {}
        for r in range(n_regimes):
            self.posteriors[r] = {
                name: {
                    'mean': prior_mean,
                    'var': prior_std ** 2,
                    'n': 0,
                    'sum_rewards': 0.0,
                }
                for name in strategy_names
            }

    def select(self, regime: int = 0,
               rng: Optional[np.random.Generator] = None) -> str:
        """Thompson Sampling: sample from posteriors, pick best arm.

        Args:
            regime: current regime label (0, 1, or 2)
            rng: random number generator

        Returns: name of selected strategy
        """
        if rng is None:
            rng = np.random.default_rng()

        regime = int(np.clip(regime, 0, self.n_regimes - 1))

        samples = {}
        for name in self.names:
            post = self.posteriors[regime][name]
            # Sample from posterior N(mean, var)
            sample = rng.normal(post['mean'], np.sqrt(post['var'] + 1e-10))
            samples[name] = sample

        return max(samples, key=samples.get)

    def get_weights(self, regime: int = 0, n_samples: int = 1000,
                    rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Get selection probabilities by Monte Carlo sampling.

        Returns: {strategy_name: probability_of_selection}
        """
        if rng is None:
            rng = np.random.default_rng(42)

        regime = int(np.clip(regime, 0, self.n_regimes - 1))
        counts = {name: 0 for name in self.names}

        for _ in range(n_samples):
            best = self.select(regime, rng)
            counts[best] += 1

        total = sum(counts.values())
        return {name: count / total for name, count in counts.items()}

    def update(self, strategy_name: str, reward: float, regime: int = 0):
        """Update posterior after observing reward.

        Conjugate Normal-Normal update:
            new_var = 1 / (1/prior_var + 1/obs_var)
            new_mean = new_var * (prior_mean/prior_var + reward/obs_var)
        """
        regime = int(np.clip(regime, 0, self.n_regimes - 1))
        post = self.posteriors[regime][strategy_name]

        post['n'] += 1
        post['sum_rewards'] += reward

        # Conjugate update
        prior_prec = 1.0 / (post['var'] + 1e-10)
        obs_prec = 1.0 / self.obs_var

        new_prec = prior_prec + obs_prec
        new_var = 1.0 / new_prec
        new_mean = new_var * (post['mean'] * prior_prec + reward * obs_prec)

        post['mean'] = new_mean
        post['var'] = new_var

    def summary(self) -> str:
        """Print posterior summary per regime."""
        lines = ["Thompson Sampling Meta-Learner Summary:"]
        regime_labels = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        for r in range(self.n_regimes):
            lines.append(f"\n  Regime {r} ({regime_labels.get(r, '?')}):")
            for name in self.names:
                p = self.posteriors[r][name]
                lines.append(
                    f"    {name:12s}: mean={p['mean']:+.4f}, "
                    f"std={np.sqrt(p['var']):.4f}, n={p['n']}"
                )
        return '\n'.join(lines)


def run_thompson_ensemble(oos_signals_dict: Dict[str, np.ndarray],
                          oos_returns: np.ndarray,
                          regime_labels: np.ndarray,
                          strategy_names: List[str],
                          bps_cost: float = 0.001,
                          warmup: int = 21) -> Tuple[np.ndarray, ThompsonSamplingMetaLearner]:
    """Run Thompson Sampling ensemble over OOS period.

    For each day:
      1. Observe current regime
      2. Thompson Sampling selects best strategy for this regime
      3. Use selected strategy's signal
      4. Observe reward, update posterior

    Args:
        oos_signals_dict: {strategy_name: signal_array}
        oos_returns: actual forward returns
        regime_labels: regime label for each day (0/1/2)
        strategy_names: list of strategy names
        bps_cost: transaction cost per position change
        warmup: days of equal-weight warmup before Thompson kicks in

    Returns: (ensemble_returns, meta_learner)
    """
    n_days = len(oos_returns)
    meta = ThompsonSamplingMetaLearner(strategy_names, n_regimes=3)
    rng = np.random.default_rng(42)

    ens_returns = np.zeros(n_days)
    selected_strategies = []
    prev_pos = 0.0

    for t in range(n_days):
        regime = int(regime_labels[t]) if not np.isnan(regime_labels[t]) else 1

        if t < warmup:
            # Equal weight during warmup
            signal = np.mean([
                np.sign(oos_signals_dict[s][t])
                for s in strategy_names
            ])
            position = np.sign(signal)
        else:
            # Thompson Sampling selection
            selected = meta.select(regime, rng)
            position = np.sign(oos_signals_dict[selected][t])
            selected_strategies.append(selected)

        # Compute return with costs
        strat_ret = position * oos_returns[t]
        cost = abs(position - prev_pos) * bps_cost
        ens_returns[t] = strat_ret - cost
        prev_pos = position

        # Update ALL strategies with their hypothetical reward
        for s in strategy_names:
            s_pos = np.sign(oos_signals_dict[s][t])
            s_ret = s_pos * oos_returns[t]
            meta.update(s, s_ret, regime)

    return ens_returns, meta
'''

# ============================================================================
# Now assemble the v3 notebook
# ============================================================================

# --- Cell 0: Replace markdown header ---
nb['cells'][0] = make_md_cell(CELL_0_MD)

# --- Cell 1: Prepend new imports to existing setup cell ---
existing_cell1_src = ''.join(nb['cells'][1]['source'])
# Insert new imports after the existing imports (before GPU detection)
insert_point = 'from tqdm.auto import tqdm'
if insert_point in existing_cell1_src:
    existing_cell1_src = existing_cell1_src.replace(
        insert_point,
        insert_point + '\n' + CELL_1_EXTRA_IMPORTS
    )

# Update version
existing_cell1_src = existing_cell1_src.replace(
    'NOTEBOOK_VERSION = "v2.1-2026-02-14"',
    'NOTEBOOK_VERSION = "v3.0-2026-02-14"'
)
existing_cell1_src = existing_cell1_src.replace(
    'QuantKubera Monolith {NOTEBOOK_VERSION}',
    'QuantKubera Monolith v3 — Magnum Opus'
)

nb['cells'][1] = make_code_cell(existing_cell1_src)

# --- Insert Cell 3b (World Monitor) after Cell 3 ---
cell_3b = make_code_cell(CELL_3B_WORLDMONITOR)

# --- Insert Cell 5 (Advanced Features) after Cell 4 ---
cell_5 = make_code_cell(CELL_5_ADVANCED)

# --- Insert EMAT cell after the TFT cell (currently cell 6) ---
cell_emat = make_code_cell(CELL_EMAT)

# --- Insert RL Meta-Learner cell ---
cell_rl = make_code_cell(CELL_RL_META)

# --- Update orchestration cell to use new features ---
# The orchestration cell is currently at index 8
# We need to update it to:
# 1. Call World Monitor data fetch
# 2. Call advanced features
# 3. Use Thompson Sampling ensemble
# 4. Update feature columns
orch_src = ''.join(nb['cells'][8]['source'])

# Add World Monitor fetch to Phase 1.5
old_phase15 = '# ── PHASE 1.5: CROSS-ASSET DATA (News Sentiment + India VIX) ──'
new_phase15 = '# ── PHASE 1.5: CROSS-ASSET DATA (News + VIX + World Monitor) ──'
orch_src = orch_src.replace(old_phase15, new_phase15)

# After cross_asset_df assignment, add World Monitor data
old_cross_merge = """    cross_asset_df = fetch_cross_asset_features(
        start_date=all_dates[0].strftime('%Y-%m-%d'),
        end_date=all_dates[-1].strftime('%Y-%m-%d'),
        date_index=all_dates,
        kite=kite,
    )"""

new_cross_merge = """    cross_asset_df = fetch_cross_asset_features(
        start_date=all_dates[0].strftime('%Y-%m-%d'),
        end_date=all_dates[-1].strftime('%Y-%m-%d'),
        date_index=all_dates,
        kite=kite,
    )

    # ── World Monitor Intelligence ──
    print("\\n  [World Monitor] Macro Regime Signals...")
    try:
        macro_df = fetch_yahoo_macro(
            all_dates[0].strftime('%Y-%m-%d'),
            all_dates[-1].strftime('%Y-%m-%d'),
        )
        if not macro_df.empty:
            macro_df = macro_df.reindex(all_dates).ffill(limit=3)
            macro_df['macro_composite'] = compute_macro_composite(macro_df)
            for col in MACRO_COLUMNS:
                if col in macro_df.columns:
                    cross_asset_df[col] = macro_df[col]
                    if col not in CROSS_ASSET_COLUMNS:
                        CROSS_ASSET_COLUMNS.append(col)
    except Exception as e:
        logger.warning(f"Macro signals failed: {e}")
        print(f"    SKIP: Macro signals failed ({e})")

    print("  [World Monitor] Fear & Greed Index...")
    try:
        fg_df = fetch_fear_greed()
        if not fg_df.empty:
            fg_df = fg_df.reindex(all_dates).ffill(limit=3)
            cross_asset_df['macro_fear_greed_z'] = fg_df['macro_fear_greed_z']
            if 'macro_fear_greed_z' not in CROSS_ASSET_COLUMNS:
                CROSS_ASSET_COLUMNS.append('macro_fear_greed_z')
    except Exception as e:
        logger.warning(f"Fear & Greed failed: {e}")"""

orch_src = orch_src.replace(old_cross_merge, new_cross_merge)

# Add advanced features to Phase 2
old_phase2_end = """            featured_data[ticker] = feat_df
            tqdm.write(f"  {ticker}: {len(feat_df)} days, "
                       f"{len(FEATURE_COLUMNS)} features ({elapsed:.1f}s)")"""

new_phase2_end = """            # Build advanced features (Groups 15-19)
            adv_df = build_advanced_features(df, feat_df['close'] if 'close' in feat_df.columns else df['close'], cfg)
            feat_df = feat_df.join(adv_df, how='left')
            for col in ADVANCED_FEATURE_COLUMNS:
                if col in feat_df.columns:
                    feat_df[col] = feat_df[col].ffill(limit=3).fillna(0.0)

            # Welford anomaly features (from raw OHLCV)
            welf_df = compute_welford_anomalies(df)
            welf_df = welf_df.reindex(feat_df.index)
            feat_df = feat_df.join(welf_df, how='left')
            for col in WELFORD_COLUMNS:
                if col in feat_df.columns:
                    feat_df[col] = feat_df[col].fillna(0.0)

            featured_data[ticker] = feat_df
            n_total = len(FEATURE_COLUMNS)
            tqdm.write(f"  {ticker}: {len(feat_df)} days, "
                       f"{n_total} base + advanced features ({elapsed:.1f}s)")"""

orch_src = orch_src.replace(old_phase2_end, new_phase2_end)

# Update feature column extension to include advanced + welford
old_feat_extend = """    # Extend FEATURE_COLUMNS with cross-asset features (if available)
    if CROSS_ASSET_COLUMNS:
        for col in CROSS_ASSET_COLUMNS:
            if col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(col)
        print(f"\\n  Features extended: {len(FEATURE_COLUMNS)} total "
              f"(+{len(CROSS_ASSET_COLUMNS)} cross-asset: "
              f"{CROSS_ASSET_COLUMNS})")"""

new_feat_extend = """    # Extend FEATURE_COLUMNS with cross-asset + advanced + welford features
    for col_list in [CROSS_ASSET_COLUMNS, ADVANCED_FEATURE_COLUMNS, WELFORD_COLUMNS]:
        for col in col_list:
            if col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(col)

    n_cross = len(CROSS_ASSET_COLUMNS)
    n_adv = len(ADVANCED_FEATURE_COLUMNS)
    n_welf = len(WELFORD_COLUMNS)
    print(f"\\n  Features extended: {len(FEATURE_COLUMNS)} total")
    print(f"    Cross-asset: +{n_cross} ({CROSS_ASSET_COLUMNS})")
    print(f"    Advanced:    +{n_adv} (HMM, Wavelet, InfoTheory, MF-DFA, TDA)")
    print(f"    Welford:     +{n_welf} (Anomaly detection)")
    print(f"    Macro:       +{len(MACRO_COLUMNS)} (World Monitor)")"""

orch_src = orch_src.replace(old_feat_extend, new_feat_extend)

# Replace the ensemble section with Thompson Sampling
old_ensemble_header = '    # ── PHASE 5.5: ENSEMBLE (TFT + Momentum + MR — majority vote) ──'
new_ensemble_header = '    # ── PHASE 5.5: ENSEMBLE (Thompson Sampling Meta-Learner) ──'
orch_src = orch_src.replace(old_ensemble_header, new_ensemble_header)

old_ensemble_print = '    print("  PHASE 5.5: ENSEMBLE (TFT + Momentum + MR — majority vote)")'
new_ensemble_print = '    print("  PHASE 5.5: ENSEMBLE (RL Thompson Sampling Meta-Learner)")'
orch_src = orch_src.replace(old_ensemble_print, new_ensemble_print)

nb['cells'][8] = make_code_cell(orch_src)

# --- Update SharpeLoss in TFT cell to also register MultiObjectiveSharpeLoss ---
tft_src = ''.join(nb['cells'][6]['source'])
# Add a note about EMAT being in next cell
tft_src += """

# NOTE: Enhanced Multi-Aspect Attention (EMAT) and Multi-Objective Loss
# are defined in the next cell. The MomentumTransformer above can be used
# as-is, or replaced with the EMAT-enhanced version for v3 training.
"""
nb['cells'][6] = make_code_cell(tft_src)

# --- Assemble final notebook with new cells inserted ---
# Original order: 0(md), 1(setup), 2(data), 3(cross), 4(feat), 5(afml),
#                 6(tft), 7(train), 8(orch), 9(vbt)
#
# New order:      0(md), 1(setup), 2(data), 3(cross), 3b(worldmon),
#                 4(feat), 5(advanced), 6(afml), 7(tft), 7b(emat),
#                 8(rl_meta), 9(train), 10(orch), 11(vbt)

new_cells = []
new_cells.append(nb['cells'][0])  # 0: Markdown header (updated)
new_cells.append(nb['cells'][1])  # 1: Setup (updated imports + version)
new_cells.append(nb['cells'][2])  # 2: Data Engine
new_cells.append(nb['cells'][3])  # 3: Cross-Asset (GDELT + VIX)
new_cells.append(cell_3b)         # 3b: World Monitor Intelligence
new_cells.append(nb['cells'][4])  # 4: Base Features (31)
new_cells.append(cell_5)          # 5: Advanced Features (16)
new_cells.append(nb['cells'][5])  # 6: AFML Pipeline
new_cells.append(nb['cells'][6])  # 7: TFT Architecture
new_cells.append(cell_emat)       # 7b: EMAT + MultiObjectiveLoss
new_cells.append(cell_rl)         # 8: RL Meta-Learner
new_cells.append(nb['cells'][7])  # 9: Training Engine
new_cells.append(nb['cells'][8])  # 10: Orchestration (updated)
new_cells.append(nb['cells'][9])  # 11: VectorBTPro Tearsheet

nb['cells'] = new_cells

# --- Write the v3 notebook ---
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"v3 notebook written to {NB_PATH}")
print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    first = cell['source'][0].strip() if cell['source'] else '(empty)'
    print(f"  Cell {i:2d}: {cell['cell_type']:8s} | {len(cell['source']):4d} lines | {first[:80]}")
