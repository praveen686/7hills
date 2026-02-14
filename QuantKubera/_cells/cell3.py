# ============================================================================
# CELL 3: Advanced Feature Engineering — 31 Features in 10 Groups
# ============================================================================

FEATURE_COLUMNS = [
    'norm_ret_1d', 'norm_ret_21d', 'norm_ret_63d', 'norm_ret_126d', 'norm_ret_252d',
    'macd_8_24', 'macd_16_48', 'macd_32_96',
    'rvol_20d', 'rvol_60d', 'gk_vol_20d', 'parkinson_vol_20d',
    'cp_rl_21', 'cp_score_21',
    'frac_diff_03', 'frac_diff_05', 'hurst_exp',
    'ram_5', 'ram_10', 'ram_21', 'ram_63',
    'vpin', 'kyles_lambda', 'amihud_illiq', 'hl_spread',
    'entropy',
    'trend_strength', 'momentum_consistency', 'mr_zscore',
    'vol_zscore', 'vol_momentum',
]


# ============================================================================
# Group 1: Normalized Returns (5 features)
# ============================================================================
def compute_returns(close: pd.Series, horizons: List[int] = [1, 21, 63, 126, 252]) -> pd.DataFrame:
    """
    Compute horizon returns normalized by EWM volatility.
    norm_ret_h = log(close / close.shift(h)) / (vol * sqrt(h))
    where vol = ewm(span=60).std() of daily log returns.
    """
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.ewm(span=60, min_periods=20).std()

    result = pd.DataFrame(index=close.index)
    for h in horizons:
        raw_ret = np.log(close / close.shift(h))
        denom = vol * np.sqrt(h)
        # Avoid division by zero
        denom = denom.replace(0, np.nan)
        result[f'norm_ret_{h}d'] = raw_ret / denom

    return result


# ============================================================================
# Group 2: MACD (3 features)
# ============================================================================
def compute_macd(close: pd.Series, pairs: List[Tuple[int, int]] = [(8, 24), (16, 48), (32, 96)]) -> pd.DataFrame:
    """
    Compute MACD z-scores for stationarity.
    Raw MACD = EWM(fast) - EWM(slow), then z-scored over 126-day rolling window.
    """
    result = pd.DataFrame(index=close.index)
    for fast, slow in pairs:
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        raw_macd = ema_fast - ema_slow

        roll_mean = raw_macd.rolling(window=126, min_periods=63).mean()
        roll_std = raw_macd.rolling(window=126, min_periods=63).std(ddof=1)
        roll_std = roll_std.replace(0, np.nan)

        result[f'macd_{fast}_{slow}'] = (raw_macd - roll_mean) / roll_std

    return result


# ============================================================================
# Group 3: Volatility (4 features)
# ============================================================================
def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4 volatility estimators:
      - rvol_20d, rvol_60d: realized vol (rolling std of log returns, annualized)
      - gk_vol_20d: Garman-Klass volatility
      - parkinson_vol_20d: Parkinson high-low volatility
    """
    close = df['close']
    high = df['high']
    low = df['low']
    opn = df['open']

    log_ret = np.log(close / close.shift(1))

    result = pd.DataFrame(index=df.index)

    # Realized vol
    result['rvol_20d'] = log_ret.rolling(window=20, min_periods=15).std(ddof=1) * np.sqrt(252)
    result['rvol_60d'] = log_ret.rolling(window=60, min_periods=40).std(ddof=1) * np.sqrt(252)

    # Garman-Klass: sqrt(mean(0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2) * 252)
    log_hl = np.log(high / low)
    log_co = np.log(close / opn)
    gk_term = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    gk_mean = gk_term.rolling(window=20, min_periods=15).mean()
    # Clamp to non-negative before sqrt
    gk_mean = gk_mean.clip(lower=0.0)
    result['gk_vol_20d'] = np.sqrt(gk_mean * 252)

    # Parkinson: sqrt(mean(ln(H/L)^2 / (4*ln2)) * 252)
    park_term = log_hl ** 2 / (4.0 * np.log(2.0))
    park_mean = park_term.rolling(window=20, min_periods=15).mean()
    park_mean = park_mean.clip(lower=0.0)
    result['parkinson_vol_20d'] = np.sqrt(park_mean * 252)

    return result


# ============================================================================
# Group 4: Changepoint Detection (2 features)
# ============================================================================
def nig_log_marginal(x: np.ndarray) -> float:
    """
    Normal-Inverse-Gamma log marginal likelihood P(x | NIG prior).
    Prior: mu0=0, kappa0=1, alpha0=1, beta0=1
    Posterior update:
      n = len(x), x_bar = mean(x), s2 = var(x)
      kappa_n = kappa0 + n
      alpha_n = alpha0 + n/2
      beta_n = beta0 + 0.5*n*s2 + 0.5*kappa0*n*x_bar^2 / kappa_n
    Log marginal = gammaln(alpha_n) - gammaln(alpha0) + alpha0*log(beta0) - alpha_n*log(beta_n)
                   + 0.5*log(kappa0/kappa_n) - (n/2)*log(2*pi)
    """
    n = len(x)
    if n < 2:
        return -np.inf

    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0

    x_bar = np.mean(x)
    s2 = np.var(x, ddof=1) if n > 1 else 0.0

    kappa_n = kappa0 + n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * (n - 1) * s2 + 0.5 * kappa0 * n * x_bar ** 2 / kappa_n

    # Protect against non-positive beta_n
    if beta_n <= 0:
        beta_n = 1e-300

    log_ml = (
        gammaln(alpha_n) - gammaln(alpha0)
        + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n)
        + 0.5 * np.log(kappa0 / kappa_n)
        - (n / 2.0) * np.log(2.0 * np.pi)
    )
    return log_ml


def compute_cpd(close: pd.Series, lookback: int = 21, min_seg: int = 5) -> pd.DataFrame:
    """
    Changepoint detection via NIG Bayesian model comparison.
    For each position, take lookback-window of log returns.
    Try all split points; best split maximizes sum of two-segment likelihoods.
    cp_rl = best_split_position / lookback (relative location in [0, 1])
    cp_score = sigmoid(best_split_ll - full_ll) (severity in [0, 1])
    """
    log_ret = np.log(close / close.shift(1)).values
    n = len(log_ret)

    cp_rl = np.full(n, np.nan)
    cp_score = np.full(n, np.nan)

    for i in range(lookback, n):
        window = log_ret[i - lookback + 1: i + 1]  # lookback values ending at i

        # Skip if any NaN
        if np.any(np.isnan(window)):
            continue

        full_ll = nig_log_marginal(window)

        best_split_ll = -np.inf
        best_split_pos = lookback // 2  # default to middle

        for s in range(min_seg, lookback - min_seg + 1):
            left = window[:s]
            right = window[s:]
            split_ll = nig_log_marginal(left) + nig_log_marginal(right)

            if split_ll > best_split_ll:
                best_split_ll = split_ll
                best_split_pos = s

        cp_rl[i] = best_split_pos / lookback

        # Severity: sigmoid of log-likelihood ratio
        delta = best_split_ll - full_ll
        # Clamp to avoid overflow in exp
        delta_clamped = np.clip(delta, -500, 500)
        cp_score[i] = 1.0 / (1.0 + np.exp(-delta_clamped))

    result = pd.DataFrame(index=close.index)
    result['cp_rl_21'] = cp_rl
    result['cp_score_21'] = cp_score
    return result


# ============================================================================
# Group 5: Fractional Calculus (3 features)
# ============================================================================
def frac_diff_weights(d: float, thresh: float = 1e-5, max_width: int = 500) -> np.ndarray:
    """
    Hosking (1981) fractional differencing weights.
    w[0] = 1, w[k] = -w[k-1] * (d - k + 1) / k
    Iterate until |w[k]| < thresh or max_width reached.

    Note: For d=0.3 with thresh=1e-5, weights decay as k^{-1.3} requiring
    ~7000 terms. max_width caps this to keep the warmup period practical
    while preserving >99.9% of the filter energy.

    Returns weights array from oldest (index 0) to newest (index -1).
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        weights.append(w_k)
        k += 1
        if k >= max_width:
            break
    # Reverse so index 0 = oldest weight
    return np.array(weights[::-1])


def compute_frac_diff(close: pd.Series, d: float) -> pd.Series:
    """
    Apply fractional differencing of order d to log(close).
    Convolve log prices with frac_diff weights.
    """
    log_price = np.log(close.values)
    w = frac_diff_weights(d)
    width = len(w)

    n = len(log_price)
    result = np.full(n, np.nan)

    for i in range(width - 1, n):
        segment = log_price[i - width + 1: i + 1]
        result[i] = np.dot(w, segment)

    return pd.Series(result, index=close.index)


def _compute_hurst_single(returns: np.ndarray, max_lag: int = 50) -> float:
    """
    Compute Hurst exponent from returns using MSD (Mean Squared Displacement).
    MSD(tau) = E[(X(t+tau) - X(t))^2] where X = cumsum(returns)
    Regress log(MSD) on log(tau) -> slope / 2 = H
    """
    if len(returns) < max_lag + 10:
        return np.nan

    X = np.cumsum(returns)
    taus = np.arange(1, max_lag + 1)
    msd = np.full(max_lag, np.nan)

    for idx, tau in enumerate(taus):
        diffs = X[tau:] - X[:-tau]
        if len(diffs) < 5:
            continue
        msd[idx] = np.mean(diffs ** 2)

    # Filter valid MSD values (positive and finite)
    valid = np.isfinite(msd) & (msd > 0)
    if valid.sum() < 5:
        return np.nan

    log_tau = np.log(taus[valid])
    log_msd = np.log(msd[valid])

    # Linear regression: log_msd = slope * log_tau + intercept
    n_valid = len(log_tau)
    sum_x = np.sum(log_tau)
    sum_y = np.sum(log_msd)
    sum_xy = np.sum(log_tau * log_msd)
    sum_xx = np.sum(log_tau ** 2)

    denom = n_valid * sum_xx - sum_x ** 2
    if abs(denom) < 1e-15:
        return np.nan

    slope = (n_valid * sum_xy - sum_x * sum_y) / denom
    hurst = slope / 2.0

    # Clamp to reasonable range
    return np.clip(hurst, 0.0, 1.0)


def compute_hurst(close: pd.Series, window: int = 252, max_lag: int = 50) -> pd.Series:
    """
    Rolling Hurst exponent computed over a window of returns.
    """
    log_ret = np.log(close / close.shift(1)).values
    n = len(log_ret)
    hurst_vals = np.full(n, np.nan)

    for i in range(window, n):
        segment = log_ret[i - window + 1: i + 1]
        if np.any(np.isnan(segment)):
            continue
        hurst_vals[i] = _compute_hurst_single(segment, max_lag=max_lag)

    return pd.Series(hurst_vals, index=close.index, name='hurst_exp')


def compute_fractional_features(close: pd.Series) -> pd.DataFrame:
    """Compute all fractional calculus features."""
    result = pd.DataFrame(index=close.index)
    result['frac_diff_03'] = compute_frac_diff(close, d=0.3)
    result['frac_diff_05'] = compute_frac_diff(close, d=0.5)
    result['hurst_exp'] = compute_hurst(close)
    return result


# ============================================================================
# Group 6: Ramanujan Sum Filter Bank (4 features)
# ============================================================================
def euler_phi(n: int) -> int:
    """Euler's totient function: count of integers in [1, n] coprime to n."""
    if n <= 0:
        return 0
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def mobius(n: int) -> int:
    """
    Mobius function:
      mu(1) = 1
      mu(n) = 0 if n has a squared prime factor
      mu(n) = (-1)^k if n is a product of k distinct primes
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    num_factors = 0
    temp = n
    p = 2

    while p * p <= temp:
        if temp % p == 0:
            temp //= p
            num_factors += 1
            if temp % p == 0:
                return 0  # Squared factor
        p += 1

    if temp > 1:
        num_factors += 1

    return 1 if num_factors % 2 == 0 else -1


def ramanujan_sum(q: int, n: int) -> float:
    """
    Ramanujan sum c_q(n) = sum over d|gcd(n,q) of mu(q/d) * phi(q) / phi(q/d)
    Simplified: c_q(n) = mu(q/g) * phi(q) / phi(q/g) where g = gcd(n, q)
    Actually the full definition sums over all d dividing gcd(n,q).
    """
    g = math.gcd(n, q)
    # Sum over divisors d of g: mu(q/d) * d * (phi(q/d) != 0 check)
    # More standard: c_q(n) = sum_{d | gcd(n,q)} d * mu(q/d)
    total = 0.0
    for d in range(1, g + 1):
        if g % d == 0:
            qd = q // d
            total += d * mobius(qd)
    return total


def compute_ramanujan(close: pd.Series, periods: List[int] = [5, 10, 21, 63],
                      window: int = 252) -> pd.DataFrame:
    """
    Ramanujan Sum Filter Bank: convolve log-returns with Ramanujan sum kernels
    to extract energy at specific trading cycle periods.
    """
    log_ret = np.log(close / close.shift(1)).values
    n = len(log_ret)
    result = pd.DataFrame(index=close.index)

    for q in periods:
        # Pre-compute kernel: kernel[j] = c_q(j+1) for j in [0, window)
        kernel = np.array([ramanujan_sum(q, j + 1) for j in range(window)])
        kernel = kernel / window  # Normalize

        # Convolve (causal: only use past data)
        filtered = np.full(n, np.nan)
        for i in range(window, n):
            segment = log_ret[i - window + 1: i + 1]
            if np.any(np.isnan(segment)):
                continue
            # Kernel is applied: newest data * kernel[0], oldest * kernel[-1]
            # Reverse kernel for convolution alignment (kernel[0] applies to most recent)
            filtered[i] = np.dot(segment, kernel[::-1][:len(segment)])

        result[f'ram_{q}'] = filtered

    return result


# ============================================================================
# Group 7: Microstructure (4 features)
# ============================================================================
def compute_microstructure(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Compute 4 microstructure features:
      - VPIN: Volume-Synchronized Probability of Informed Trading
      - Kyle's Lambda: price impact coefficient
      - Amihud Illiquidity: |return| / dollar volume
      - HL Spread: high-low spread proxy (Corwin-Schultz simplified)
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'].astype(float)

    log_ret = np.log(close / close.shift(1))

    result = pd.DataFrame(index=df.index)

    # --- VPIN ---
    sigma = log_ret.rolling(window=20, min_periods=10).std()
    sigma = sigma.replace(0, np.nan)
    # Bulk volume classification: buy probability = Phi(ret / sigma)
    z = log_ret / sigma
    buy_prob = pd.Series(norm.cdf(z.values), index=close.index)
    buy_vol = volume * buy_prob
    sell_vol = volume * (1.0 - buy_prob)
    abs_imbalance = (buy_vol - sell_vol).abs()
    total_vol = volume.rolling(window=window, min_periods=window // 2).sum()
    total_vol = total_vol.replace(0, np.nan)
    vpin = abs_imbalance.rolling(window=window, min_periods=window // 2).sum() / total_vol
    result['vpin'] = vpin

    # --- Kyle's Lambda ---
    abs_ret = log_ret.abs()
    signed_vol = np.sign(log_ret) * volume
    abs_signed_vol = signed_vol.abs()

    # Rolling covariance / variance
    cov_rv = abs_ret.rolling(window=window, min_periods=window // 2).cov(abs_signed_vol)
    var_sv = abs_signed_vol.rolling(window=window, min_periods=window // 2).var(ddof=1)
    var_sv = var_sv.replace(0, np.nan)
    result['kyles_lambda'] = cov_rv / var_sv

    # --- Amihud Illiquidity ---
    dollar_vol = close * volume
    dollar_vol = dollar_vol.replace(0, np.nan)
    daily_illiq = abs_ret / dollar_vol
    result['amihud_illiq'] = daily_illiq.rolling(window=window, min_periods=window // 2).mean()

    # --- HL Spread (Corwin-Schultz simplified) ---
    # Use rolling average of log(H/L) as spread proxy
    log_hl = np.log(high / low)
    # Corwin-Schultz: alpha derived from 2-day high-low ratio
    # Simplified version: spread = 2*(exp(alpha) - 1) / (1 + exp(alpha))
    # where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
    # beta = E[ln(H_t/L_t)^2]
    beta = (log_hl ** 2).rolling(window=window, min_periods=window // 2).mean()
    # Also compute gamma from 2-day range
    high_2d = high.rolling(window=2).max()
    low_2d = low.rolling(window=2).min()
    gamma = np.log(high_2d / low_2d) ** 2

    # alpha = (sqrt(2) - 1) * sqrt(beta) / (3 - 2*sqrt(2)) when gamma term is small
    # Full: alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    sqrt2 = np.sqrt(2.0)
    denom_cs = 3.0 - 2.0 * sqrt2

    # Ensure beta is non-negative
    beta_safe = beta.clip(lower=0.0)
    gamma_safe = gamma.clip(lower=0.0)

    alpha = (np.sqrt(2.0 * beta_safe) - np.sqrt(beta_safe)) / denom_cs
    # Adjust with gamma correction
    gamma_correction = np.sqrt(gamma_safe / denom_cs)
    alpha = alpha - gamma_correction
    # Clamp alpha to reasonable range to avoid extreme spread values
    alpha = alpha.clip(lower=-1.0, upper=2.0)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    # Clamp negative spreads to 0
    spread = spread.clip(lower=0.0)
    result['hl_spread'] = spread

    return result


# ============================================================================
# Group 8: Information Theory (1 feature)
# ============================================================================
def compute_entropy(close: pd.Series, word_len: int = 3, window: int = 252) -> pd.Series:
    """
    Shannon entropy of binary price movement patterns.
    Encode: 1 if price up, 0 if down.
    Form words of word_len consecutive bits.
    Compute normalized Shannon entropy over rolling window.
    """
    # Binary encoding: 1 if close > prev_close, 0 otherwise
    direction = (close.diff() > 0).astype(int).values
    n = len(direction)
    n_words = 2 ** word_len
    max_entropy = np.log2(n_words) if n_words > 1 else 1.0

    entropy_vals = np.full(n, np.nan)

    for i in range(window + word_len - 1, n):
        # Extract the window of directions
        seg = direction[i - window + 1: i + 1]

        # Build words
        words = []
        for j in range(word_len - 1, len(seg)):
            word = 0
            for k in range(word_len):
                word = (word << 1) | seg[j - word_len + 1 + k]
            words.append(word)

        if len(words) == 0:
            continue

        # Histogram
        counts = np.bincount(words, minlength=n_words).astype(float)
        probs = counts / counts.sum()

        # Shannon entropy (base 2)
        probs_pos = probs[probs > 0]
        H = -np.sum(probs_pos * np.log2(probs_pos))

        # Normalize to [0, 1]
        entropy_vals[i] = H / max_entropy if max_entropy > 0 else 0.0

    return pd.Series(entropy_vals, index=close.index, name='entropy')


# ============================================================================
# Group 9: Momentum Quality (3 features)
# ============================================================================
def compute_momentum_quality(close: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Compute momentum quality metrics:
      - trend_strength: |avg_up - avg_down| / (avg_up + avg_down)
      - momentum_consistency: fraction of positive returns in rolling window
      - mr_zscore: (close - EMA) / rolling_std  (mean-reversion z-score)
    """
    ret = close.pct_change()
    result = pd.DataFrame(index=close.index)

    # Trend strength
    up_ret = ret.clip(lower=0)
    down_ret = (-ret).clip(lower=0)  # magnitude of down moves

    avg_up = up_ret.rolling(window=window, min_periods=window // 2).mean()
    avg_down = down_ret.rolling(window=window, min_periods=window // 2).mean()

    denom_ts = avg_up + avg_down
    denom_ts = denom_ts.replace(0, np.nan)
    result['trend_strength'] = (avg_up - avg_down).abs() / denom_ts

    # Momentum consistency: fraction of positive returns
    pos_indicator = (ret > 0).astype(float)
    result['momentum_consistency'] = pos_indicator.rolling(
        window=window, min_periods=window // 2
    ).mean()

    # Mean reversion z-score: (close - EMA) / rolling_std
    ema = close.ewm(span=window, min_periods=window // 2).mean()
    rolling_std = close.rolling(window=window, min_periods=window // 2).std(ddof=1)
    rolling_std = rolling_std.replace(0, np.nan)
    result['mr_zscore'] = (close - ema) / rolling_std

    return result


# ============================================================================
# Group 10: Volume Features (2 features)
# ============================================================================
def compute_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute volume-based features:
      - vol_zscore: (volume - rolling_mean) / rolling_std
      - vol_momentum: volume.pct_change(5)  (5-day volume momentum)
    """
    volume = df['volume'].astype(float)
    result = pd.DataFrame(index=df.index)

    roll_mean = volume.rolling(window=window, min_periods=window // 2).mean()
    roll_std = volume.rolling(window=window, min_periods=window // 2).std(ddof=1)
    roll_std = roll_std.replace(0, np.nan)

    result['vol_zscore'] = (volume - roll_mean) / roll_std
    result['vol_momentum'] = volume.pct_change(periods=5)

    return result


# ============================================================================
# Master Function: Build All Features
# ============================================================================
def build_all_features(df: pd.DataFrame, cfg: Optional[MonolithConfig] = None) -> pd.DataFrame:
    """
    Build all 31 engineered features from 10 research groups.
    Adds forward target: target_ret = close.pct_change(1).shift(-1)
    Drops NaN warmup rows.
    Returns df with FEATURE_COLUMNS + 'target_ret' + original OHLCV.
    """
    if cfg is None:
        cfg = MonolithConfig()

    close = df['close']
    t0 = time.time()

    print("Building features...")

    # Group 1: Normalized Returns
    print("  [1/10] Normalized Returns (5 features)")
    feat_ret = compute_returns(close)

    # Group 2: MACD
    print("  [2/10] MACD Z-scores (3 features)")
    feat_macd = compute_macd(close)

    # Group 3: Volatility
    print("  [3/10] Volatility Estimators (4 features)")
    feat_vol = compute_volatility(df)

    # Group 4: Changepoint Detection
    print("  [4/10] NIG Changepoint Detection (2 features)")
    feat_cpd = compute_cpd(close)

    # Group 5: Fractional Calculus
    print("  [5/10] Fractional Differentiation + Hurst (3 features)")
    feat_frac = compute_fractional_features(close)

    # Group 6: Ramanujan Filter Bank
    print("  [6/10] Ramanujan Sum Filter Bank (4 features)")
    feat_ram = compute_ramanujan(close)

    # Group 7: Microstructure
    print("  [7/10] Market Microstructure (4 features)")
    feat_micro = compute_microstructure(df)

    # Group 8: Entropy
    print("  [8/10] Information-Theoretic Entropy (1 feature)")
    feat_entropy = compute_entropy(close)

    # Group 9: Momentum Quality
    print("  [9/10] Momentum Quality (3 features)")
    feat_mq = compute_momentum_quality(close)

    # Group 10: Volume Features
    print("  [10/10] Volume Features (2 features)")
    feat_vf = compute_volume_features(df)

    # Assemble all features into the dataframe
    out = df.copy()

    for col in feat_ret.columns:
        out[col] = feat_ret[col]
    for col in feat_macd.columns:
        out[col] = feat_macd[col]
    for col in feat_vol.columns:
        out[col] = feat_vol[col]
    for col in feat_cpd.columns:
        out[col] = feat_cpd[col]
    for col in feat_frac.columns:
        out[col] = feat_frac[col]
    for col in feat_ram.columns:
        out[col] = feat_ram[col]
    for col in feat_micro.columns:
        out[col] = feat_micro[col]
    out['entropy'] = feat_entropy
    for col in feat_mq.columns:
        out[col] = feat_mq[col]
    for col in feat_vf.columns:
        out[col] = feat_vf[col]

    # Target: next-day return (shift -1 is the ONLY forward-looking value, used as label)
    out['target_ret'] = close.pct_change(1).shift(-1)

    # Vol-scaled training target: raw_fwd_return / realized_vol
    # Vol computed on CURRENT (unshifted) returns — no leakage
    log_ret = np.log(close / close.shift(1))
    vol_20 = log_ret.rolling(20, min_periods=10).std()
    vol_20 = vol_20.replace(0, np.nan)
    out['target_train'] = out['target_ret'] / vol_20

    # Verify all expected feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Drop rows where ANY feature or target is NaN (warmup period)
    n_before = len(out)
    out_pre_drop = out  # keep reference for diagnostics
    out = out.dropna(subset=FEATURE_COLUMNS + ['target_ret', 'target_train'])
    n_after = len(out)

    elapsed = time.time() - t0
    print(f"\nFeature engineering complete in {elapsed:.1f}s")
    print(f"  Rows: {n_before} -> {n_after} (dropped {n_before - n_after} warmup rows)")
    print(f"  Features: {len(FEATURE_COLUMNS)}")

    if n_after == 0:
        # Diagnose which features are all-NaN
        nan_cols = [c for c in FEATURE_COLUMNS if out_pre_drop[c].isna().all()]
        raise ValueError(
            f"All rows dropped after NaN removal. "
            f"Features that are entirely NaN: {nan_cols}. "
            f"Data has {n_before} rows but longest warmup exceeds this."
        )

    print(f"  Date range: {out.index[0].date()} to {out.index[-1].date()}")
    print(f"  Columns: {list(out.columns)}")

    return out