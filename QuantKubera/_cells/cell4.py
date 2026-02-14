# ============================================================================
# CELL 4: AFML EVENT-DRIVEN PIPELINE (Lopez de Prado, 2018)
# ============================================================================
# Implements: CUSUM filter, Triple Barrier Labels, Meta-Labeling, Bet Sizing
# Reference: Advances in Financial Machine Learning, Chapters 2-3, 5
# ============================================================================

from scipy.stats import norm

def get_daily_vol(close, span=50):
    """EWM standard deviation of log returns.
    
    Args:
        close: pd.Series of prices indexed by datetime.
        span: int, span for exponential weighted moving average.
    
    Returns:
        pd.Series of daily volatility estimates.
    """
    log_ret = np.log(close).diff().dropna()
    return log_ret.ewm(span=span, min_periods=max(1, span // 2)).std()


def cusum_filter(close, threshold):
    """Symmetric CUSUM filter for event detection.
    
    Detects structural breaks by tracking positive and negative cumulative
    sums of log returns. When either sum exceeds the threshold, an event
    is recorded and the cumulative sum resets to zero.
    
    Args:
        close: pd.Series of prices indexed by datetime.
        threshold: float or pd.Series. If Series, must share close's index.
                   Typical usage: daily_vol * multiplier (e.g., 2.0).
    
    Returns:
        pd.DatetimeIndex of event timestamps where structural breaks detected.
    """
    log_ret = np.log(close).diff().dropna()
    events = []
    s_pos = 0.0
    s_neg = 0.0
    
    # Convert scalar threshold to Series for uniform handling
    if isinstance(threshold, (int, float)):
        thresh_series = pd.Series(threshold, index=log_ret.index)
    else:
        thresh_series = threshold.reindex(log_ret.index, method='ffill')
    
    for t, r in log_ret.items():
        h = thresh_series.loc[t]
        if np.isnan(h) or np.isnan(r):
            continue
        
        # Update cumulative sums (reset floor at zero)
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        
        # Check if either sum breaches threshold
        if s_pos > h:
            events.append(t)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -h:
            events.append(t)
            s_pos = 0.0
            s_neg = 0.0
    
    return pd.DatetimeIndex(events)


def triple_barrier_labels(close, events, pt_sl=(2.0, 1.0), num_days=10, min_ret=0.0):
    """Triple barrier labeling with volatility-scaled barriers.
    
    For each event, sets three barriers:
      - Upper (profit-take): pt_sl[0] * daily_vol above entry
      - Lower (stop-loss):  -pt_sl[1] * daily_vol below entry
      - Vertical:            num_days forward (max holding period)
    
    The label is determined by which barrier is touched first.
    
    Args:
        close: pd.Series of prices indexed by datetime.
        events: pd.DatetimeIndex of event timestamps (from cusum_filter).
        pt_sl: tuple (profit_take_mult, stop_loss_mult) of daily vol.
               Set either to 0 to disable that barrier.
        num_days: int, maximum holding period in trading days.
        min_ret: float, minimum absolute log return for a non-zero label
                 when the vertical barrier is hit first.
    
    Returns:
        pd.DataFrame indexed by event timestamps with columns:
            'ret':   realized log return at exit
            'bin':   label (+1 profit-take, -1 stop-loss, 0 vertical/below min_ret)
            't_end': timestamp when position was closed
    """
    daily_vol = get_daily_vol(close, span=50)
    
    results = []
    
    for t0 in events:
        if t0 not in close.index or t0 not in daily_vol.index:
            continue
        
        vol = daily_vol.loc[t0]
        if np.isnan(vol) or vol <= 0:
            continue
        
        p0 = close.loc[t0]
        
        # Define barrier levels
        upper_barrier = pt_sl[0] * vol if pt_sl[0] > 0 else np.inf
        lower_barrier = -pt_sl[1] * vol if pt_sl[1] > 0 else -np.inf
        
        # Get the forward price path from t0 up to t0 + num_days bars
        t0_idx = close.index.get_loc(t0)
        t_end_idx = min(t0_idx + num_days, len(close) - 1)
        path = close.iloc[t0_idx: t_end_idx + 1]
        
        if len(path) < 2:
            continue
        
        # Log returns relative to entry
        log_returns = np.log(path / p0)
        
        # Find first crossing of each barrier
        hit_upper = log_returns[log_returns >= upper_barrier].index
        hit_lower = log_returns[log_returns <= lower_barrier].index
        
        # Determine first touch times (use NaT for unhit barriers)
        t_upper = hit_upper[0] if len(hit_upper) > 0 else pd.NaT
        t_lower = hit_lower[0] if len(hit_lower) > 0 else pd.NaT
        t_vert = path.index[-1]  # vertical barrier always exists
        
        # Find earliest barrier touch
        candidates = {}
        if not pd.isna(t_upper):
            candidates[t_upper] = 'upper'
        if not pd.isna(t_lower):
            candidates[t_lower] = 'lower'
        candidates[t_vert] = 'vertical'
        
        t_end = min(candidates.keys())
        barrier_type = candidates[t_end]
        
        # Realized log return at exit
        ret = np.log(close.loc[t_end] / p0)
        
        # Assign label
        if barrier_type == 'upper':
            label = 1
        elif barrier_type == 'lower':
            label = -1
        else:
            # Vertical barrier: label based on return direction if above min_ret
            if abs(ret) > min_ret:
                label = int(np.sign(ret))
            else:
                label = 0
        
        results.append({'t0': t0, 'ret': ret, 'bin': label, 't_end': t_end})
    
    if len(results) == 0:
        return pd.DataFrame(columns=['ret', 'bin', 't_end'])
    
    df = pd.DataFrame(results).set_index('t0')
    df.index.name = None
    return df


def meta_labeling(primary_preds, true_labels):
    """Generate meta-labels: 1 if primary model correctly predicted direction.
    
    Meta-labeling separates side (direction) from size (confidence).
    The primary model decides direction; the meta-model decides whether
    to take the trade and how large to size it.
    
    Args:
        primary_preds: pd.Series of predicted directions (+1/-1).
        true_labels: pd.Series of actual directions (+1/-1/0).
    
    Returns:
        pd.Series of 0/1 meta-labels. 1 = primary model was correct.
    """
    # Align indices
    common_idx = primary_preds.index.intersection(true_labels.index)
    preds = primary_preds.loc[common_idx]
    actuals = true_labels.loc[common_idx]
    
    # Meta-label: 1 if signs agree (correct prediction), 0 otherwise
    # A zero true label means the trade was unprofitable, so meta-label = 0
    meta = (np.sign(preds) == np.sign(actuals)).astype(int)
    
    # If true label is 0, the trade had no edge -> meta = 0
    meta[actuals == 0] = 0
    
    return meta


def bet_size(meta_probs, max_leverage=1.0):
    """Probit-based position sizing from meta-model probabilities.
    
    Converts P(correct) from the meta-model into a continuous position
    size using the inverse normal CDF (probit function). This maps
    probabilities smoothly to position sizes with natural concavity
    near 0 and 1.
    
    Args:
        meta_probs: pd.Series of P(correct) from meta-model, in [0, 1].
        max_leverage: float, maximum absolute position size.
    
    Returns:
        pd.Series of position sizes in [-max_leverage, max_leverage].
        Values near 0.5 produce small sizes; values near 0 or 1 produce
        sizes approaching max_leverage.
    """
    # Clip to avoid infinities from norm.ppf at 0 and 1
    clipped = meta_probs.clip(1e-5, 1.0 - 1e-5)
    
    # Probit transform: map probability to z-score
    z = pd.Series(norm.ppf(clipped.values), index=clipped.index)
    
    # Convert z-score to position size via CDF
    # size = (2 * Phi(z) - 1) maps z to [-1, 1]
    # z > 0 (prob > 0.5) -> positive size, z < 0 (prob < 0.5) -> negative size
    size = (2.0 * norm.cdf(z) - 1.0) * max_leverage
    
    return size


print("=" * 70)
print("AFML EVENT-DRIVEN PIPELINE")
print("  Functions defined: get_daily_vol, cusum_filter, triple_barrier_labels,")
print("                     meta_labeling, bet_size")
print("  These will be applied during walk-forward training in Cell 6.")
print("=" * 70)