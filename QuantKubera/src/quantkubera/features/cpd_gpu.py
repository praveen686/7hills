import tensorflow as tf
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass(frozen=True)
class NIGPrior:
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0

@tf.function
def nig_segment_cost_from_prefix(
    S1: tf.Tensor, S2: tf.Tensor,
    t: int,
    s_idx: tf.Tensor,
    prior: NIGPrior,
    eps: float = 1e-12
) -> tf.Tensor:
    """
    Returns cost tensor [B, K] where cost[b,k] = -log p(x_{s+1:t}) under NIG.
    S1, S2: [B, T+1] prefix sums
    t: scalar int32 (current end)
    s_idx: [K] int32 (candidate starts)
    """
    # S1_t: [B, 1], S1_s: [B, K]
    S1_t = tf.gather(S1, t, axis=1)[:, None]
    S2_t = tf.gather(S2, t, axis=1)[:, None]

    S1_s = tf.gather(S1, s_idx, axis=1)
    S2_s = tf.gather(S2, s_idx, axis=1)

    sum_x = S1_t - S1_s
    sum_x2 = S2_t - S2_s

    n = tf.cast(t - s_idx, tf.float32)
    n = tf.maximum(n, 1.0)
    n_bk = n[None, :]

    mu0 = tf.cast(prior.mu0, tf.float32)
    kappa0 = tf.cast(prior.kappa0, tf.float32)
    alpha0 = tf.cast(prior.alpha0, tf.float32)
    beta0 = tf.cast(prior.beta0, tf.float32)

    xbar = sum_x / n_bk
    sse = tf.maximum(sum_x2 - tf.square(sum_x) / n_bk, 0.0)

    kappa_n = kappa0 + n_bk
    alpha_n = alpha0 + 0.5 * n_bk

    mean_term = (kappa0 * n_bk * tf.square(xbar - mu0)) / (2.0 * kappa_n)
    beta_n = tf.maximum(beta0 + 0.5 * sse + mean_term, eps)

    logp = (
        tf.math.lgamma(alpha_n) - tf.math.lgamma(alpha0)
        + 0.5 * (tf.math.log(kappa0) - tf.math.log(kappa_n))
        + alpha0 * tf.math.log(tf.maximum(beta0, eps))
        - alpha_n * tf.math.log(beta_n)
        - 0.5 * n_bk * tf.cast(math.log(math.pi), tf.float32)
    )

    return -logp


@tf.function
def batched_optimal_partition_nig(
    x: tf.Tensor,
    beta: float,
    prior: NIGPrior = NIGPrior(),
    min_seg: int = 10,
    lookback: int = 512,
    eps: float = 1e-12
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Batched Optimal Partitioning (Banded lookback) in TF.
    x: [B, T]
    Returns:
        F: [B, T+1] (DP objective)
        prev: [B, T+1] (predecessor indices)
    """
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    beta = tf.cast(beta, tf.float32)

    zeros = tf.zeros([B, 1], dtype=tf.float32)
    S1 = tf.concat([zeros, tf.math.cumsum(x, axis=1)], axis=1)
    S2 = tf.concat([zeros, tf.math.cumsum(tf.square(x), axis=1)], axis=1)

    F_ta = tf.TensorArray(tf.float32, size=T+1, clear_after_read=False)
    P_ta = tf.TensorArray(tf.int32, size=T+1, clear_after_read=False)

    F0 = -beta * tf.ones([B], tf.float32)
    P0 = -tf.ones([B], tf.int32)

    F_ta = F_ta.write(0, F0)
    P_ta = P_ta.write(0, P0)

    pos_inf = tf.cast(1e30, tf.float32)

    for t in tf.range(1, T + 1):
        s_start = tf.maximum(0, t - lookback)
        s_end = t - min_seg

        def candidate_case():
            s_idx = tf.range(s_start, s_end + 1, dtype=tf.int32)
            F_s = tf.transpose(F_ta.gather(s_idx), perm=[1, 0])
            C_s = nig_segment_cost_from_prefix(S1, S2, t, s_idx, prior, eps=eps)
            obj = F_s + C_s + beta
            arg = tf.argmin(obj, axis=1, output_type=tf.int32)
            best = tf.reduce_min(obj, axis=1)
            best_s = tf.gather(s_idx, arg)
            return best, best_s

        def no_candidate_case():
            return pos_inf * tf.ones([B], tf.float32), -tf.ones([B], tf.int32)

        Ft, Pt = tf.cond(s_end >= s_start, candidate_case, no_candidate_case)
        F_ta = F_ta.write(t, Ft)
        P_ta = P_ta.write(t, Pt)

    F = tf.transpose(F_ta.stack(), perm=[1, 0])
    prev = tf.transpose(P_ta.stack(), perm=[1, 0])

    return F, prev

@tf.function
def compute_severity_score(
    S1: tf.Tensor, S2: tf.Tensor,
    t: int,
    last_cp: tf.Tensor,
    prior: NIGPrior,
    eps: float = 1e-12
) -> tf.Tensor:
    """
    Computes a severity score for a CP at time t.
    Score = -log(p(segment_after_cp)) - (-log(p(entire_window_as_one)))
    Higher score = more significant change.
    """
    K = 21 # TMT context window
    s_idx = tf.maximum(0, t - K)
    
    # Cost with CP (just the latest segment)
    cost_seg = nig_segment_cost_from_prefix(S1, S2, t, tf.expand_dims(last_cp, 0), prior, eps=eps)
    
    # Cost without CP (entire window)
    cost_base = nig_segment_cost_from_prefix(S1, S2, t, tf.constant([s_idx], dtype=tf.int32), prior, eps=eps)
    
    # Score is the log-likelihood improvement
    score = tf.squeeze(cost_base - cost_seg)
    return tf.maximum(score, 0.0)

def extract_features_from_prev(
    prev_np: np.ndarray, 
    lookback_window: int = 21,
    F_np: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts cp_rl (Relative Location) and cp_score (Severity) from DP results.
    """
    B, Tp1 = prev_np.shape
    T = Tp1 - 1
    
    cp_rl = np.zeros((B, T), dtype=np.float32)
    cp_score = np.zeros((B, T), dtype=np.float32)
    
    for b in range(B):
        for t in range(1, T + 1):
            last_cp = prev_np[b, t]
            if last_cp <= 0:
                cp_rl[b, t-1] = 0.0
            else:
                dist = t - last_cp
                cp_rl[b, t-1] = np.clip(dist / lookback_window, 0, 1)
                
            # For simplicity, score can be derived from the F-change or we'll compute it in a separate pass
            # For now, let's use the distance as a proxy if we don't have F
            if F_np is not None:
                # Severity is essentially the jump in the objective compared to a no-cp path
                # Use a clipped exp to avoid overflows
                delta_f = F_np[b, t] - F_np[b, t-1]
                cp_score[b, t-1] = 1.0 / (1.0 + np.exp(np.clip(-0.1 * delta_f, -50, 50)))
            else:
                cp_score[b, t-1] = 0.5 # Neutral
                
    return cp_rl, cp_score
