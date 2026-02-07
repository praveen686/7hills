"""Information-theoretic indicators: Entropy and Mutual Information.

Ported from Timothy Masters' C++ implementation.
These measure the predictability / randomness of price patterns.

- **Entropy**: Shannon entropy of binary price-change word patterns.
  Low entropy = trending / predictable, high entropy = random walk.
- **Mutual Information**: MI between current price move and recent history.
  High MI = history predicts future (trend-follow), low MI = no memory.

Both are useful as **regime filters**: gate strategy entries based on
market regime (trending vs random, predictable vs unpredictable).
"""

from __future__ import annotations

import numpy as np


def price_entropy(
    prices: np.ndarray,
    word_length: int = 2,
    window: int | None = None,
) -> float:
    """Shannon entropy of binary price-change word patterns.

    Encodes each bar as 1 (up) or 0 (down), then forms words of
    `word_length` consecutive bits. Entropy is computed from the
    histogram of these words, normalized to [0, 1].

    Args:
        prices: Close prices, chronological order (oldest first).
        word_length: Number of consecutive up/down bits per word (1-4).
        window: Number of bars to use. If None, uses all.

    Returns:
        Normalized entropy in [0, 1]. 1.0 = perfectly random (uniform
        distribution over all word patterns). 0.0 = perfectly predictable.
    """
    if window is not None:
        prices = prices[-window:]

    n = len(prices)
    if n <= word_length:
        return 0.0

    n_bins = 2 ** word_length
    bins = np.zeros(n_bins, dtype=int)

    # Build binary words from consecutive price changes
    # prices[i] > prices[i+1] → bit = 1 (down in Masters' reversed-time convention)
    # We use chronological: prices[i] > prices[i-1] → 1 (up)
    for i in range(word_length, n):
        k = 1 if prices[i] > prices[i - 1] else 0
        for j in range(1, word_length):
            k *= 2
            if prices[i - j] > prices[i - j - 1]:
                k += 1
        bins[k] += 1

    # Compute entropy
    total = n - word_length
    if total <= 0:
        return 0.0

    ent = 0.0
    for count in bins:
        if count > 0:
            p = count / total
            ent -= p * np.log(p)

    # Normalize by log(n_bins) so result is in [0, 1]
    max_ent = np.log(n_bins)
    return float(ent / max_ent) if max_ent > 0 else 0.0


def mutual_information(
    prices: np.ndarray,
    word_length: int = 2,
    window: int | None = None,
) -> float:
    """Mutual information between current price move and history.

    Encodes the current bar as 1 (up) or 0 (down), and the preceding
    `word_length` bars as a binary word. MI measures how much the
    history word tells us about the current direction.

    Args:
        prices: Close prices, chronological order (oldest first).
        word_length: Number of history bits (1-4).
        window: Number of bars to use. If None, uses all.

    Returns:
        Mutual information in nats. Higher = more predictable.
        Zero = current move is independent of recent history.
    """
    if window is not None:
        prices = prices[-window:]

    n = len(prices)
    if n < word_length + 2:
        return 0.0

    n_history = 2 ** word_length  # number of history categories
    n_bins = 2 * n_history        # current (0/1) × history
    bins = np.zeros(n_bins, dtype=int)
    dep_marg = np.zeros(2, dtype=float)  # marginal of current move

    n_cases = n - word_length - 1

    # Chronological convention: index 0 = oldest
    # Current move: prices[i+1] > prices[i] → 1, else 0
    # History: preceding word_length moves
    for i in range(word_length, word_length + n_cases):
        # Current move (dependent variable)
        current = 1 if prices[i + 1] > prices[i] else 0
        dep_marg[current] += 1

        # History word
        k = current
        for j in range(1, word_length + 1):
            k *= 2
            if prices[i - j + 1] > prices[i - j]:
                k += 1
        bins[k] += 1

    if n_cases <= 0:
        return 0.0

    # Normalize marginals
    dep_marg /= n_cases

    # Compute MI
    MI = 0.0
    for i in range(n_history):
        # hist_marg = P(history = i)
        hist_marg = (bins[i] + bins[i + n_history]) / n_cases
        if hist_marg <= 0:
            continue

        # P(current=0, history=i)
        p0 = bins[i] / n_cases
        if p0 > 0 and dep_marg[0] > 0:
            MI += p0 * np.log(p0 / (hist_marg * dep_marg[0]))

        # P(current=1, history=i)
        p1 = bins[i + n_history] / n_cases
        if p1 > 0 and dep_marg[1] > 0:
            MI += p1 * np.log(p1 / (hist_marg * dep_marg[1]))

    return float(MI)


# ---------------------------------------------------------------------------
# Rolling / vectorized versions for time-series
# ---------------------------------------------------------------------------

def rolling_entropy(
    prices: np.ndarray,
    word_length: int = 2,
    window: int = 100,
) -> np.ndarray:
    """Compute entropy in a rolling window over the price series.

    Returns array of same length as prices, with NaN for the first
    `window - 1` elements.
    """
    n = len(prices)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = price_entropy(prices[i - window + 1:i + 1], word_length)
    return out


def rolling_mutual_info(
    prices: np.ndarray,
    word_length: int = 2,
    window: int = 100,
) -> np.ndarray:
    """Compute mutual information in a rolling window.

    Returns array of same length as prices, with NaN for the first
    `window - 1` elements.
    """
    n = len(prices)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = mutual_information(prices[i - window + 1:i + 1], word_length)
    return out
