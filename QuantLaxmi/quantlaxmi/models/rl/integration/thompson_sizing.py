"""Pattern 2: Contextual Thompson Sizing + Gradient Bandit Sizer.

Implements two complementary position-sizing approaches using bandits:

1. **ThompsonSizingAgent**: 7 discrete arms = {0, +/-0.25, +/-0.5, +/-1.0}.
   Each arm has a Normal-Inverse-Gamma (NIG) posterior, updated daily with
   realised returns.  Context (10-dim) enables arm selection to depend on
   market regime, VIX level, TFT confidence, drawdown, etc.

   Uses ThompsonStrategyAllocator from the agents module (where each "strategy"
   is actually a sizing level treated as a bandit arm) and the
   _NeuralContextualBandit for optional context-dependent arm selection.

2. **GradientBanditSizer** (for S6): Arms = {0.1, 0.25, 0.5, 0.75, 1.0}.
   Uses the GradientBandit algorithm from the bandits module with softmax
   policy over discrete size levels.  Simpler and faster than Thompson —
   suited for strategies with stable return distributions (e.g. S6 HMM Regime).

3. **ThompsonSizingPipeline**: Walk-forward pipeline that wires pretrained
   backbone hidden states into the ThompsonSizingAgent.  Extracts TFT position
   (Gaussian mu) and confidence (1/sigma) per day, builds the 10-dim context
   vector, selects arm via Thompson sampling, observes return, updates posterior.
   OOS comparison: Thompson-sized vs uniform-sized Sharpe.

Context vector (10-dim):
    [regime, vix, dte, dow_sin, dow_cos, tft_confidence,
     hidden_mean, hidden_std, recent_sharpe_5d, drawdown]

Math references:
    - NIG conjugate update: Ch 15.5 of Rao & Jelvis
    - Gradient Bandit (softmax policy gradient): Ch 15.4
    - Kelly fraction as a sizing baseline: Ch 8.1
    - Walk-forward protocol: no look-ahead, train-then-test
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

THOMPSON_ARM_LEVELS: List[float] = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
THOMPSON_ARM_NAMES: List[str] = [
    "neg_1.0", "neg_0.5", "neg_0.25", "zero", "pos_0.25", "pos_0.5", "pos_1.0"
]

GRADIENT_ARM_LEVELS: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0]

# Context vector dimension
CONTEXT_DIM: int = 10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ThompsonSizingConfig:
    """Configuration for ThompsonSizingAgent."""

    arm_levels: List[float] = field(default_factory=lambda: list(THOMPSON_ARM_LEVELS))
    arm_names: List[str] = field(default_factory=lambda: list(THOMPSON_ARM_NAMES))
    context_dim: int = CONTEXT_DIM
    prior_mu: float = 0.0
    prior_lambda: float = 1.0
    prior_alpha: float = 2.0
    prior_beta: float = 1.0
    use_neural: bool = False
    min_allocation: float = 0.0
    max_allocation: float = 1.0
    seed: int = 42


@dataclass
class GradientSizingConfig:
    """Configuration for GradientBanditSizer."""

    arm_levels: List[float] = field(default_factory=lambda: list(GRADIENT_ARM_LEVELS))
    alpha: float = 0.1
    use_baseline: bool = True
    seed: int = 42


@dataclass
class PipelineConfig:
    """Configuration for ThompsonSizingPipeline."""

    train_window: int = 252
    test_window: int = 63
    step_size: int = 21
    warmup_days: int = 20
    baseline_size: float = 0.25
    sharpe_lookback: int = 5
    vix_feature_idx: int = 1
    dte_feature_idx: int = 2
    regime_feature_idx: int = 0


# ============================================================================
# ThompsonSizingAgent
# ============================================================================


class ThompsonSizingAgent:
    """Contextual Thompson Sampling over discrete position sizing levels.

    Treats each sizing level as an arm in a multi-armed bandit. The 10-dim
    context vector allows the allocator to learn regime-dependent sizing
    preferences via the NIG posterior.

    Parameters
    ----------
    cfg : ThompsonSizingConfig
        Agent configuration (arm levels, prior params, etc.).
    """

    def __init__(self, cfg: Optional[ThompsonSizingConfig] = None) -> None:
        self.cfg = cfg or ThompsonSizingConfig()
        self._arm_levels = np.array(self.cfg.arm_levels, dtype=np.float64)
        self._arm_names = list(self.cfg.arm_names)
        self._n_arms = len(self._arm_levels)
        self._name_to_level: Dict[str, float] = dict(
            zip(self._arm_names, self._arm_levels)
        )
        self._rng = np.random.default_rng(self.cfg.seed)

        # Build the Thompson allocator (one "strategy" per arm)
        from quantlaxmi.models.rl.agents.thompson_allocator import ThompsonStrategyAllocator

        self._allocator = ThompsonStrategyAllocator(
            strategy_names=self._arm_names,
            context_dim=self.cfg.context_dim,
            prior_mu=self.cfg.prior_mu,
            prior_lambda=self.cfg.prior_lambda,
            prior_alpha=self.cfg.prior_alpha,
            prior_beta=self.cfg.prior_beta,
            min_allocation=self.cfg.min_allocation,
            max_allocation=self.cfg.max_allocation,
            use_neural=self.cfg.use_neural,
            seed=self.cfg.seed,
        )

        # History tracking for diagnostics
        self._selection_history: List[Dict] = []
        self._cumulative_return: float = 0.0
        self._n_selections: int = 0

        logger.info(
            "ThompsonSizingAgent: %d arms %s, context_dim=%d, neural=%s",
            self._n_arms, self._arm_levels.tolist(),
            self.cfg.context_dim, self.cfg.use_neural,
        )

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def arm_levels(self) -> np.ndarray:
        return self._arm_levels.copy()

    def select_size(self, context: np.ndarray) -> Tuple[float, str]:
        """Select a position size level given a context vector.

        Uses Thompson Sampling: sample from each arm's NIG posterior,
        allocate proportionally, then select the arm with highest weight.

        Parameters
        ----------
        context : (context_dim,) numpy array
            Market context features.

        Returns
        -------
        size : float
            Selected position size level (one of arm_levels).
        arm_name : str
            Name of the selected arm.
        """
        context = np.asarray(context, dtype=np.float64)
        if len(context) != self.cfg.context_dim:
            # Pad or truncate to expected dim
            padded = np.zeros(self.cfg.context_dim, dtype=np.float64)
            n = min(len(context), self.cfg.context_dim)
            padded[:n] = context[:n]
            context = padded

        # Get allocation weights from Thompson posterior sampling
        alloc_weights = self._allocator.select_allocation(context)

        # Select the arm with the highest allocation weight
        best_arm_name = max(alloc_weights, key=alloc_weights.get)
        best_size = self._name_to_level[best_arm_name]

        self._n_selections += 1
        logger.debug(
            "Thompson select #%d: arm=%s (size=%.2f), weights=%s",
            self._n_selections, best_arm_name, best_size,
            {k: f"{v:.3f}" for k, v in alloc_weights.items()},
        )

        return best_size, best_arm_name

    def select_size_stochastic(self, context: np.ndarray) -> Tuple[float, str]:
        """Select a position size by sampling proportionally to allocation weights.

        Unlike select_size() which picks argmax, this samples from the
        categorical distribution defined by the allocation weights —
        preserving exploration.

        Parameters
        ----------
        context : (context_dim,) numpy array

        Returns
        -------
        size : float
        arm_name : str
        """
        context = np.asarray(context, dtype=np.float64)
        if len(context) != self.cfg.context_dim:
            padded = np.zeros(self.cfg.context_dim, dtype=np.float64)
            n = min(len(context), self.cfg.context_dim)
            padded[:n] = context[:n]
            context = padded

        alloc_weights = self._allocator.select_allocation(context)
        names = list(alloc_weights.keys())
        probs = np.array([alloc_weights[n] for n in names], dtype=np.float64)

        # Ensure valid probability distribution
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(names)) / len(names)

        chosen_idx = self._rng.choice(len(names), p=probs)
        chosen_name = names[chosen_idx]
        chosen_size = self._name_to_level[chosen_name]

        return chosen_size, chosen_name

    def observe(
        self,
        arm_name: str,
        realised_return: float,
        context: np.ndarray,
    ) -> None:
        """Update the NIG posterior for the selected arm after observing reward.

        Parameters
        ----------
        arm_name : str
            Which arm was played.
        realised_return : float
            Observed daily return when using this sizing level.
        context : (context_dim,) numpy array
            Context at time of selection.
        """
        context = np.asarray(context, dtype=np.float64)
        if len(context) != self.cfg.context_dim:
            padded = np.zeros(self.cfg.context_dim, dtype=np.float64)
            n = min(len(context), self.cfg.context_dim)
            padded[:n] = context[:n]
            context = padded

        self._allocator.update(arm_name, realised_return, context)
        self._cumulative_return += realised_return

        self._selection_history.append({
            "arm": arm_name,
            "size": self._name_to_level[arm_name],
            "return": realised_return,
            "cum_return": self._cumulative_return,
        })

        logger.debug(
            "Thompson observe: arm=%s, return=%.6f, cum=%.6f",
            arm_name, realised_return, self._cumulative_return,
        )

    def get_posteriors(self) -> Dict[str, Dict]:
        """Return NIG posterior parameters for each arm.

        Returns
        -------
        dict mapping arm_name -> {"mu", "lambda", "alpha", "beta",
        "n_updates", "mean", "variance"}.
        """
        return self._allocator.get_posteriors()

    def get_arm_ranking(self) -> List[Tuple[str, float]]:
        """Return arms ranked by posterior mean (descending).

        Returns
        -------
        list of (arm_name, posterior_mean) tuples sorted descending.
        """
        return self._allocator.ranking()

    def get_history(self) -> List[Dict]:
        """Return selection history for diagnostics."""
        return list(self._selection_history)

    def train_neural(
        self,
        contexts: np.ndarray,
        arms: np.ndarray,
        returns: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Dict:
        """Train the optional neural contextual bandit on historical data.

        Parameters
        ----------
        contexts : (N, context_dim) — historical context vectors.
        arms : (N,) — integer arm indices selected.
        returns : (N,) — observed returns.
        epochs : int
        batch_size : int

        Returns
        -------
        dict with "final_loss" and "losses".
        """
        return self._allocator.train_neural(
            contexts, arms, returns, epochs=epochs, batch_size=batch_size
        )


# ============================================================================
# GradientBanditSizer (for S6)
# ============================================================================


class GradientBanditSizer:
    """Gradient Bandit over discrete position sizes for S6 HMM Regime strategy.

    Uses softmax policy gradient (Ch 15.4) to learn preferences over
    5 discrete sizing levels: {0.1, 0.25, 0.5, 0.75, 1.0}.

    Simpler than Thompson Sampling — no Bayesian posterior, just preference
    parameters updated via the REINFORCE-style gradient:
        H(a) += alpha * (R - R_bar) * (1{a=A} - pi(a))

    Parameters
    ----------
    cfg : GradientSizingConfig
    """

    def __init__(self, cfg: Optional[GradientSizingConfig] = None) -> None:
        self.cfg = cfg or GradientSizingConfig()
        self._arm_levels = np.array(self.cfg.arm_levels, dtype=np.float64)
        self._n_arms = len(self._arm_levels)

        from quantlaxmi.models.rl.algorithms.bandits import GradientBandit

        self._bandit = GradientBandit(
            num_arms=self._n_arms,
            alpha=self.cfg.alpha,
            use_baseline=self.cfg.use_baseline,
            seed=self.cfg.seed,
        )

        # History
        self._selection_history: List[Dict] = []
        self._cumulative_return: float = 0.0
        self._n_selections: int = 0

        logger.info(
            "GradientBanditSizer: %d arms %s, alpha=%.3f",
            self._n_arms, self._arm_levels.tolist(), self.cfg.alpha,
        )

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def arm_levels(self) -> np.ndarray:
        return self._arm_levels.copy()

    def select_size(self) -> Tuple[float, int]:
        """Select a position size via softmax policy sampling.

        Returns
        -------
        size : float
            Selected size level (one of arm_levels).
        arm_idx : int
            Index of the selected arm.
        """
        arm_idx = self._bandit.select_arm()
        size = float(self._arm_levels[arm_idx])
        self._n_selections += 1

        logger.debug(
            "GradientBandit select #%d: arm=%d (size=%.2f), prefs=%s",
            self._n_selections, arm_idx, size,
            np.round(self._bandit._preferences, 3).tolist(),
        )

        return size, arm_idx

    def observe(self, arm_idx: int, reward: float) -> None:
        """Update gradient bandit preferences after observing reward.

        Parameters
        ----------
        arm_idx : int
            Which arm was selected.
        reward : float
            Observed reward (daily return scaled by sizing level).
        """
        self._bandit.update(arm_idx, reward)
        self._cumulative_return += reward

        self._selection_history.append({
            "arm_idx": arm_idx,
            "size": float(self._arm_levels[arm_idx]),
            "reward": reward,
            "cum_return": self._cumulative_return,
        })

        logger.debug(
            "GradientBandit observe: arm=%d, reward=%.6f, cum=%.6f",
            arm_idx, reward, self._cumulative_return,
        )

    def get_probabilities(self) -> np.ndarray:
        """Return current softmax policy probabilities over arms.

        Returns
        -------
        probs : (n_arms,) numpy array summing to 1.
        """
        return self._bandit._softmax()

    def get_preferences(self) -> np.ndarray:
        """Return raw preference parameters H(a).

        Returns
        -------
        (n_arms,) numpy array
        """
        return self._bandit._preferences.copy()

    def get_history(self) -> List[Dict]:
        """Return selection history for diagnostics."""
        return list(self._selection_history)


# ============================================================================
# Context Builder
# ============================================================================


def build_context_vector(
    regime: float,
    vix: float,
    dte: float,
    day_of_week: int,
    tft_confidence: float,
    hidden_mean: float,
    hidden_std: float,
    recent_sharpe_5d: float,
    drawdown: float,
    extra: float = 0.0,
) -> np.ndarray:
    """Build the 10-dimensional context vector for Thompson sizing.

    All inputs are scalar and should be pre-normalised or naturally bounded.

    Parameters
    ----------
    regime : float
        Regime indicator (e.g. 0=calm, 1=trending, 2=volatile, 3=mean-reverting).
        Normalised to [0, 1] by dividing by 3.
    vix : float
        VIX or India VIX level. Normalised by /100.
    dte : float
        Days to expiry of nearest contract. Normalised by /30.
    day_of_week : int
        0=Monday ... 4=Friday. Encoded as sin and cos of weekly cycle.
    tft_confidence : float
        1 / sigma from TFT Gaussian output. Higher = more confident.
        Clipped to [0, 10].
    hidden_mean : float
        Mean of backbone hidden state vector for the current day.
    hidden_std : float
        Std of backbone hidden state vector for the current day.
    recent_sharpe_5d : float
        Rolling 5-day realised Sharpe ratio. Clipped to [-3, 3].
    drawdown : float
        Current portfolio drawdown fraction in [0, 1].
    extra : float
        Reserved slot for future features (default 0).

    Returns
    -------
    context : (10,) numpy array
    """
    dow_sin = math.sin(2.0 * math.pi * day_of_week / 5.0)
    dow_cos = math.cos(2.0 * math.pi * day_of_week / 5.0)

    context = np.array([
        np.clip(regime / 3.0, 0.0, 1.0),
        np.clip(vix / 100.0, 0.0, 1.0),
        np.clip(dte / 30.0, 0.0, 2.0),
        dow_sin,
        dow_cos,
        np.clip(tft_confidence, 0.0, 10.0),
        np.clip(hidden_mean, -5.0, 5.0),
        np.clip(hidden_std, 0.0, 5.0),
        np.clip(recent_sharpe_5d, -3.0, 3.0),
        np.clip(drawdown, 0.0, 1.0),
    ], dtype=np.float64)

    return context


def _compute_rolling_sharpe(returns: np.ndarray, lookback: int) -> float:
    """Compute realised Sharpe ratio over the last `lookback` returns.

    Uses ddof=1 for unbiased std estimate and annualises by sqrt(252).

    Parameters
    ----------
    returns : (N,) array of daily returns.
    lookback : int

    Returns
    -------
    sharpe : float (annualised)
    """
    if len(returns) < max(2, lookback):
        return 0.0
    window = returns[-lookback:]
    valid = window[~np.isnan(window)]
    if len(valid) < 2:
        return 0.0
    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return (mu / sigma) * math.sqrt(252)


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown from an equity curve.

    Parameters
    ----------
    equity_curve : (N,) cumulative equity values.

    Returns
    -------
    max_dd : float in [0, 1]
    """
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = np.where(peak > 0, (peak - equity_curve) / peak, 0.0)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


# ============================================================================
# ThompsonSizingPipeline
# ============================================================================


class ThompsonSizingPipeline:
    """Walk-forward pipeline for Thompson-based position sizing.

    Takes pretrained backbone hidden states, extracts TFT position (Gaussian
    mu) and confidence (1/sigma) per day, builds context vectors, selects
    sizing via Thompson Sampling, observes returns, and updates posteriors.

    Produces an OOS comparison: Thompson-sized returns vs uniform-sized returns.

    Parameters
    ----------
    sizing_cfg : ThompsonSizingConfig
    pipeline_cfg : PipelineConfig
    """

    def __init__(
        self,
        sizing_cfg: Optional[ThompsonSizingConfig] = None,
        pipeline_cfg: Optional[PipelineConfig] = None,
    ) -> None:
        self.sizing_cfg = sizing_cfg or ThompsonSizingConfig()
        self.pipeline_cfg = pipeline_cfg or PipelineConfig()
        self._agent = ThompsonSizingAgent(self.sizing_cfg)

    @property
    def agent(self) -> ThompsonSizingAgent:
        """Access the underlying Thompson sizing agent."""
        return self._agent

    def run(
        self,
        hidden_states: np.ndarray,
        tft_mu: np.ndarray,
        tft_sigma: np.ndarray,
        raw_returns: np.ndarray,
        features: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict:
        """Run the full walk-forward pipeline.

        Parameters
        ----------
        hidden_states : (n_days, d_hidden)
            Precomputed backbone hidden states for a single asset.
        tft_mu : (n_days,)
            TFT Gaussian mean (predicted position direction/magnitude).
        tft_sigma : (n_days,)
            TFT Gaussian sigma (uncertainty). Confidence = 1/sigma.
        raw_returns : (n_days,)
            Actual realised daily returns (used for reward observation).
        features : (n_days, n_features) optional
            Raw features for extracting regime/VIX/DTE. If None, defaults used.
        dates : DatetimeIndex optional
            Trading dates for day-of-week extraction.

        Returns
        -------
        dict with keys:
            thompson_returns : (n_oos_days,) — OOS returns with Thompson sizing
            uniform_returns : (n_oos_days,) — OOS returns with uniform sizing
            thompson_sharpe : float
            uniform_sharpe : float
            sharpe_improvement : float
            fold_details : list[dict]
            posteriors : dict
            arm_ranking : list
        """
        n_days = len(raw_returns)
        pcfg = self.pipeline_cfg

        # Validate inputs
        if len(hidden_states) != n_days:
            raise ValueError(
                f"hidden_states length ({len(hidden_states)}) != "
                f"raw_returns length ({n_days})"
            )
        if len(tft_mu) != n_days:
            raise ValueError(
                f"tft_mu length ({len(tft_mu)}) != raw_returns length ({n_days})"
            )
        if len(tft_sigma) != n_days:
            raise ValueError(
                f"tft_sigma length ({len(tft_sigma)}) != raw_returns length ({n_days})"
            )

        all_thompson_returns: List[float] = []
        all_uniform_returns: List[float] = []
        fold_details: List[Dict] = []
        fold_idx = 0

        fold_start = pcfg.warmup_days
        while fold_start + pcfg.train_window + pcfg.test_window <= n_days:
            train_end = fold_start + pcfg.train_window
            test_end = min(train_end + pcfg.test_window, n_days)

            logger.info(
                "Thompson sizing fold %d: train=[%d:%d], test=[%d:%d]",
                fold_idx, fold_start, train_end, train_end, test_end,
            )

            # --- Training phase: update posteriors on train window ---
            train_thompson_rets = []
            for t in range(fold_start, train_end):
                if t + 1 >= n_days:
                    break

                context = self._build_context_for_day(
                    t, hidden_states, tft_mu, tft_sigma,
                    raw_returns, features, dates, train_thompson_rets,
                )

                # Select arm and observe next-day return scaled by size
                size, arm_name = self._agent.select_size_stochastic(context)
                next_day_return = raw_returns[t + 1] if not np.isnan(raw_returns[t + 1]) else 0.0
                sized_return = size * next_day_return

                # Observe the sized return as reward
                self._agent.observe(arm_name, sized_return, context)
                train_thompson_rets.append(sized_return)

            # --- Test phase: select arms OOS (no posterior updates) ---
            fold_thompson_rets = []
            fold_uniform_rets = []
            test_recent_rets: List[float] = []

            for t in range(train_end, test_end):
                if t + 1 >= n_days:
                    break

                context = self._build_context_for_day(
                    t, hidden_states, tft_mu, tft_sigma,
                    raw_returns, features, dates, test_recent_rets,
                )

                # Thompson: select arm (greedy for OOS — argmax posterior)
                size, arm_name = self._agent.select_size(context)
                next_day_return = raw_returns[t + 1] if not np.isnan(raw_returns[t + 1]) else 0.0

                # Thompson-sized return
                thompson_ret = size * next_day_return
                fold_thompson_rets.append(thompson_ret)
                all_thompson_returns.append(thompson_ret)

                # Uniform-sized return (baseline)
                uniform_ret = pcfg.baseline_size * next_day_return
                fold_uniform_rets.append(uniform_ret)
                all_uniform_returns.append(uniform_ret)

                test_recent_rets.append(thompson_ret)

                # Update posteriors with OOS observations (online learning)
                self._agent.observe(arm_name, thompson_ret, context)

            # Fold-level Sharpe
            fold_t_arr = np.array(fold_thompson_rets) if fold_thompson_rets else np.array([0.0])
            fold_u_arr = np.array(fold_uniform_rets) if fold_uniform_rets else np.array([0.0])
            fold_t_sharpe = self._sharpe(fold_t_arr)
            fold_u_sharpe = self._sharpe(fold_u_arr)

            fold_details.append({
                "fold": fold_idx,
                "train_range": (fold_start, train_end),
                "test_range": (train_end, test_end),
                "n_test_days": len(fold_thompson_rets),
                "thompson_sharpe": fold_t_sharpe,
                "uniform_sharpe": fold_u_sharpe,
                "thompson_total_ret": float(np.sum(fold_t_arr)),
                "uniform_total_ret": float(np.sum(fold_u_arr)),
            })

            logger.info(
                "Fold %d: Thompson Sharpe=%.3f, Uniform Sharpe=%.3f, "
                "n_test=%d",
                fold_idx, fold_t_sharpe, fold_u_sharpe, len(fold_thompson_rets),
            )

            fold_start += pcfg.step_size
            fold_idx += 1

        # Aggregate OOS results
        thompson_arr = np.array(all_thompson_returns) if all_thompson_returns else np.array([0.0])
        uniform_arr = np.array(all_uniform_returns) if all_uniform_returns else np.array([0.0])
        thompson_sharpe = self._sharpe(thompson_arr)
        uniform_sharpe = self._sharpe(uniform_arr)
        improvement = thompson_sharpe - uniform_sharpe

        logger.info(
            "Pipeline complete: %d folds, Thompson Sharpe=%.3f, "
            "Uniform Sharpe=%.3f, improvement=%.3f",
            fold_idx, thompson_sharpe, uniform_sharpe, improvement,
        )

        return {
            "thompson_returns": thompson_arr,
            "uniform_returns": uniform_arr,
            "thompson_sharpe": thompson_sharpe,
            "uniform_sharpe": uniform_sharpe,
            "sharpe_improvement": improvement,
            "fold_details": fold_details,
            "posteriors": self._agent.get_posteriors(),
            "arm_ranking": self._agent.get_arm_ranking(),
            "n_folds": fold_idx,
            "n_oos_days": len(all_thompson_returns),
        }

    def _build_context_for_day(
        self,
        t: int,
        hidden_states: np.ndarray,
        tft_mu: np.ndarray,
        tft_sigma: np.ndarray,
        raw_returns: np.ndarray,
        features: Optional[np.ndarray],
        dates: Optional[pd.DatetimeIndex],
        recent_rets: List[float],
    ) -> np.ndarray:
        """Build the 10-dim context vector for day t (fully causal).

        Parameters
        ----------
        t : int — day index (uses data up to and including t, never t+1)
        hidden_states : (n_days, d_hidden)
        tft_mu : (n_days,) — TFT predicted position
        tft_sigma : (n_days,) — TFT uncertainty
        raw_returns : (n_days,) — realised returns (only up to t used)
        features : optional (n_days, n_features) — for regime/VIX/DTE
        dates : optional DatetimeIndex — for day-of-week
        recent_rets : list — recent sized returns within this fold

        Returns
        -------
        context : (10,) numpy array
        """
        pcfg = self.pipeline_cfg

        # Regime (from features or default)
        regime = 0.0
        if features is not None and t < len(features):
            regime = float(features[t, pcfg.regime_feature_idx]) if features.shape[1] > pcfg.regime_feature_idx else 0.0

        # VIX (from features or default)
        vix = 15.0  # default India VIX
        if features is not None and t < len(features):
            if features.shape[1] > pcfg.vix_feature_idx:
                raw_vix = float(features[t, pcfg.vix_feature_idx])
                vix = raw_vix if 5.0 <= raw_vix <= 100.0 else 15.0

        # DTE (from features or default)
        dte = 15.0  # mid-expiry default
        if features is not None and t < len(features):
            if features.shape[1] > pcfg.dte_feature_idx:
                raw_dte = float(features[t, pcfg.dte_feature_idx])
                dte = raw_dte if 0.0 <= raw_dte <= 60.0 else 15.0

        # Day of week
        dow = 2  # Wednesday default
        if dates is not None and t < len(dates):
            dow = dates[t].dayofweek  # 0=Monday ... 4=Friday

        # TFT confidence: 1 / sigma (higher = more confident)
        sigma_t = float(tft_sigma[t]) if t < len(tft_sigma) else 1.0
        sigma_t = max(sigma_t, 1e-6)
        tft_confidence = 1.0 / sigma_t

        # Hidden state statistics
        h_t = hidden_states[t] if t < len(hidden_states) else np.zeros(1)
        hidden_mean = float(np.mean(h_t))
        hidden_std = float(np.std(h_t))

        # Recent Sharpe (from sized returns within this fold, causal)
        if len(recent_rets) >= 2:
            recent_arr = np.array(recent_rets[-pcfg.sharpe_lookback:])
            recent_sharpe = _compute_rolling_sharpe(recent_arr, pcfg.sharpe_lookback)
        else:
            recent_sharpe = 0.0

        # Drawdown (from raw returns up to t, causal)
        if t >= 2:
            equity = np.cumprod(1.0 + np.nan_to_num(raw_returns[:t + 1], nan=0.0))
            dd = _compute_max_drawdown(equity)
        else:
            dd = 0.0

        return build_context_vector(
            regime=regime,
            vix=vix,
            dte=dte,
            day_of_week=dow,
            tft_confidence=tft_confidence,
            hidden_mean=hidden_mean,
            hidden_std=hidden_std,
            recent_sharpe_5d=recent_sharpe,
            drawdown=dd,
        )

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Compute annualised Sharpe ratio with ddof=1.

        Parameters
        ----------
        returns : (N,) array

        Returns
        -------
        sharpe : float (annualised by sqrt(252))
        """
        valid = returns[~np.isnan(returns)]
        if len(valid) < 2:
            return 0.0
        mu = float(np.mean(valid))
        sigma = float(np.std(valid, ddof=1))
        if sigma < 1e-12:
            return 0.0
        return (mu / sigma) * math.sqrt(252)

    def report(self, results: Dict) -> str:
        """Generate a human-readable report of pipeline results.

        Parameters
        ----------
        results : dict from run()

        Returns
        -------
        report : str
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'THOMPSON SIZING PIPELINE RESULTS':^70}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  OOS days:           {results['n_oos_days']}")
        lines.append(f"  Folds:              {results['n_folds']}")
        lines.append(f"  Thompson Sharpe:    {results['thompson_sharpe']:.4f}")
        lines.append(f"  Uniform Sharpe:     {results['uniform_sharpe']:.4f}")
        lines.append(f"  Improvement:        {results['sharpe_improvement']:+.4f}")
        lines.append("")
        lines.append("  Arm Ranking (posterior mean):")
        for rank, (name, mean) in enumerate(results["arm_ranking"], 1):
            lines.append(f"    {rank}. {name:<20} mu={mean:+.6f}")
        lines.append("")
        lines.append("  Fold-level details:")
        for fd in results["fold_details"]:
            lines.append(
                f"    Fold {fd['fold']}: "
                f"test=[{fd['test_range'][0]}:{fd['test_range'][1]}] "
                f"Thompson={fd['thompson_sharpe']:+.3f} "
                f"Uniform={fd['uniform_sharpe']:+.3f} "
                f"n={fd['n_test_days']}"
            )
        lines.append("")
        lines.append("  Posteriors:")
        for arm_name, post in results["posteriors"].items():
            lines.append(
                f"    {arm_name:<20} mu={post['mu']:+.4f} "
                f"lam={post['lambda']:.1f} "
                f"alpha={post['alpha']:.1f} "
                f"beta={post['beta']:.4f} "
                f"n={post['n_updates']}"
            )
        lines.append("=" * 70)
        report_str = "\n".join(lines)
        logger.info("\n%s", report_str)
        return report_str


# ============================================================================
# GradientBanditSizingPipeline (simpler, for S6)
# ============================================================================


class GradientBanditSizingPipeline:
    """Walk-forward pipeline using GradientBanditSizer for position sizing.

    Simpler than the Thompson pipeline: no context vector, just gradient
    bandit preferences updated with realised rewards.  Suited for strategies
    like S6 HMM Regime where the return distribution is relatively stable.

    Parameters
    ----------
    sizing_cfg : GradientSizingConfig
    pipeline_cfg : PipelineConfig
    """

    def __init__(
        self,
        sizing_cfg: Optional[GradientSizingConfig] = None,
        pipeline_cfg: Optional[PipelineConfig] = None,
    ) -> None:
        self.sizing_cfg = sizing_cfg or GradientSizingConfig()
        self.pipeline_cfg = pipeline_cfg or PipelineConfig()
        self._sizer = GradientBanditSizer(self.sizing_cfg)

    @property
    def sizer(self) -> GradientBanditSizer:
        return self._sizer

    def run(
        self,
        raw_returns: np.ndarray,
        signal: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run the walk-forward pipeline.

        Parameters
        ----------
        raw_returns : (n_days,)
            Realised daily returns (e.g. from S6 strategy signal × market return).
        signal : (n_days,) optional
            Directional signal in [-1, 1] from S6. If None, assumed +1 (always long).
            The gradient bandit sizes the magnitude, while signal gives direction.

        Returns
        -------
        dict with keys:
            gradient_returns : (n_oos_days,)
            uniform_returns : (n_oos_days,)
            gradient_sharpe : float
            uniform_sharpe : float
            sharpe_improvement : float
            fold_details : list[dict]
            final_probs : (n_arms,)
            final_preferences : (n_arms,)
        """
        n_days = len(raw_returns)
        pcfg = self.pipeline_cfg

        if signal is None:
            signal = np.ones(n_days, dtype=np.float64)

        all_gradient_rets: List[float] = []
        all_uniform_rets: List[float] = []
        fold_details: List[Dict] = []
        fold_idx = 0

        fold_start = pcfg.warmup_days
        while fold_start + pcfg.train_window + pcfg.test_window <= n_days:
            train_end = fold_start + pcfg.train_window
            test_end = min(train_end + pcfg.test_window, n_days)

            logger.info(
                "GradientBandit fold %d: train=[%d:%d], test=[%d:%d]",
                fold_idx, fold_start, train_end, train_end, test_end,
            )

            # --- Training phase ---
            for t in range(fold_start, train_end):
                if t + 1 >= n_days:
                    break
                size, arm_idx = self._sizer.select_size()
                next_day_return = raw_returns[t + 1] if not np.isnan(raw_returns[t + 1]) else 0.0
                direction = float(signal[t]) if not np.isnan(signal[t]) else 1.0
                sized_return = direction * size * next_day_return
                self._sizer.observe(arm_idx, sized_return)

            # --- Test phase ---
            fold_gradient_rets = []
            fold_uniform_rets = []

            for t in range(train_end, test_end):
                if t + 1 >= n_days:
                    break
                size, arm_idx = self._sizer.select_size()
                next_day_return = raw_returns[t + 1] if not np.isnan(raw_returns[t + 1]) else 0.0
                direction = float(signal[t]) if not np.isnan(signal[t]) else 1.0

                gradient_ret = direction * size * next_day_return
                fold_gradient_rets.append(gradient_ret)
                all_gradient_rets.append(gradient_ret)

                uniform_ret = pcfg.baseline_size * direction * next_day_return
                fold_uniform_rets.append(uniform_ret)
                all_uniform_rets.append(uniform_ret)

                # Continue learning OOS (online)
                self._sizer.observe(arm_idx, gradient_ret)

            # Fold Sharpe
            fg = np.array(fold_gradient_rets) if fold_gradient_rets else np.array([0.0])
            fu = np.array(fold_uniform_rets) if fold_uniform_rets else np.array([0.0])

            fold_details.append({
                "fold": fold_idx,
                "train_range": (fold_start, train_end),
                "test_range": (train_end, test_end),
                "n_test_days": len(fold_gradient_rets),
                "gradient_sharpe": _sharpe_from_array(fg),
                "uniform_sharpe": _sharpe_from_array(fu),
                "gradient_total_ret": float(np.sum(fg)),
                "uniform_total_ret": float(np.sum(fu)),
                "arm_probs": self._sizer.get_probabilities().tolist(),
            })

            logger.info(
                "Fold %d: Gradient Sharpe=%.3f, Uniform Sharpe=%.3f, "
                "probs=%s",
                fold_idx,
                fold_details[-1]["gradient_sharpe"],
                fold_details[-1]["uniform_sharpe"],
                np.round(self._sizer.get_probabilities(), 3).tolist(),
            )

            fold_start += pcfg.step_size
            fold_idx += 1

        # Aggregate
        g_arr = np.array(all_gradient_rets) if all_gradient_rets else np.array([0.0])
        u_arr = np.array(all_uniform_rets) if all_uniform_rets else np.array([0.0])
        g_sharpe = _sharpe_from_array(g_arr)
        u_sharpe = _sharpe_from_array(u_arr)

        logger.info(
            "GradientBandit pipeline: %d folds, Gradient Sharpe=%.3f, "
            "Uniform Sharpe=%.3f",
            fold_idx, g_sharpe, u_sharpe,
        )

        return {
            "gradient_returns": g_arr,
            "uniform_returns": u_arr,
            "gradient_sharpe": g_sharpe,
            "uniform_sharpe": u_sharpe,
            "sharpe_improvement": g_sharpe - u_sharpe,
            "fold_details": fold_details,
            "final_probs": self._sizer.get_probabilities(),
            "final_preferences": self._sizer.get_preferences(),
            "n_folds": fold_idx,
            "n_oos_days": len(all_gradient_rets),
        }

    def report(self, results: Dict) -> str:
        """Generate a human-readable report.

        Parameters
        ----------
        results : dict from run()

        Returns
        -------
        report : str
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"{'GRADIENT BANDIT SIZING PIPELINE (S6)':^70}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  OOS days:           {results['n_oos_days']}")
        lines.append(f"  Folds:              {results['n_folds']}")
        lines.append(f"  Gradient Sharpe:    {results['gradient_sharpe']:.4f}")
        lines.append(f"  Uniform Sharpe:     {results['uniform_sharpe']:.4f}")
        lines.append(f"  Improvement:        {results['sharpe_improvement']:+.4f}")
        lines.append("")
        lines.append(f"  Final arm probabilities:")
        arm_levels = self.sizing_cfg.arm_levels
        probs = results["final_probs"]
        prefs = results["final_preferences"]
        for i, (level, p, h) in enumerate(zip(arm_levels, probs, prefs)):
            lines.append(f"    size={level:.2f}  prob={p:.4f}  pref={h:+.4f}")
        lines.append("")
        lines.append("  Fold-level details:")
        for fd in results["fold_details"]:
            lines.append(
                f"    Fold {fd['fold']}: "
                f"test=[{fd['test_range'][0]}:{fd['test_range'][1]}] "
                f"Gradient={fd['gradient_sharpe']:+.3f} "
                f"Uniform={fd['uniform_sharpe']:+.3f} "
                f"n={fd['n_test_days']}"
            )
        lines.append("=" * 70)
        report_str = "\n".join(lines)
        logger.info("\n%s", report_str)
        return report_str


# ============================================================================
# Helper
# ============================================================================


def _sharpe_from_array(returns: np.ndarray) -> float:
    """Compute annualised Sharpe from a returns array (ddof=1, sqrt(252))."""
    valid = returns[~np.isnan(returns)]
    if len(valid) < 2:
        return 0.0
    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return (mu / sigma) * math.sqrt(252)
