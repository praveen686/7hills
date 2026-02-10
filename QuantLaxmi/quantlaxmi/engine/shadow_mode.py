"""A/B Testing / Shadow Mode for Model Deployment.

Runs two models in parallel -- a "champion" (live) and "challenger" (shadow) --
to validate new checkpoints before promoting them.  Only the champion's signal
is used for actual trading; the challenger runs silently and its hypothetical
performance is tracked for comparison.

Usage
-----
    runner = ShadowRunner(champion_model, challenger_model, config)
    for features, actual_return in daily_stream:
        result = runner.step(features, actual_return)
        live_signal = result["champion_signal"]   # only this is traded
    report = runner.evaluate()
    if runner.should_promote():
        runner.promote_challenger()

The ``ShadowReport`` dataclass summarises both models' Sharpe, signal
correlation, agreement percentage, and a text recommendation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model protocol — anything with a .predict(features) -> float
# ---------------------------------------------------------------------------

class PredictModel(Protocol):
    """Minimal interface for models used in shadow mode."""

    def predict(self, features: np.ndarray) -> float: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ShadowConfig:
    """Configuration for shadow-mode A/B testing.

    Parameters
    ----------
    enabled : bool
        Master switch for shadow mode.
    challenger_checkpoint : str
        Path to the challenger model checkpoint (informational).
    min_evaluation_days : int
        Minimum days of parallel running before promotion is eligible.
    min_sharpe_improvement : float
        Challenger must beat champion Sharpe by at least this much.
    auto_promote : bool
        If True, automatically promote the challenger when it qualifies.
    """

    enabled: bool = False
    challenger_checkpoint: str = ""
    min_evaluation_days: int = 5
    min_sharpe_improvement: float = 0.1
    auto_promote: bool = False


# ---------------------------------------------------------------------------
# Shadow Report
# ---------------------------------------------------------------------------

@dataclass
class ShadowReport:
    """Comparison report between champion and challenger models.

    Attributes
    ----------
    champion_sharpe : float
        Annualized Sharpe ratio of the champion (ddof=1).
    challenger_sharpe : float
        Annualized Sharpe ratio of the challenger (ddof=1).
    sharpe_improvement : float
        challenger_sharpe - champion_sharpe.
    signal_correlation : float
        Pearson correlation between champion and challenger signals.
    agreement_pct : float
        Percentage of days where both models agree on direction (sign).
    n_days : int
        Number of evaluation days so far.
    champion_total_return : float
        Cumulative return of the champion.
    challenger_total_return : float
        Cumulative return of the challenger.
    recommendation : str
        One of "keep_champion", "promote_challenger", "insufficient_data".
    """

    champion_sharpe: float = 0.0
    challenger_sharpe: float = 0.0
    sharpe_improvement: float = 0.0
    signal_correlation: float = 0.0
    agreement_pct: float = 0.0
    n_days: int = 0
    champion_total_return: float = 0.0
    challenger_total_return: float = 0.0
    recommendation: str = "insufficient_data"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "champion_sharpe": round(self.champion_sharpe, 4),
            "challenger_sharpe": round(self.challenger_sharpe, 4),
            "sharpe_improvement": round(self.sharpe_improvement, 4),
            "signal_correlation": round(self.signal_correlation, 4),
            "agreement_pct": round(self.agreement_pct, 4),
            "n_days": self.n_days,
            "champion_total_return": round(self.champion_total_return, 6),
            "challenger_total_return": round(self.challenger_total_return, 6),
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Shadow Runner
# ---------------------------------------------------------------------------

class ShadowRunner:
    """Run champion and challenger models in parallel for A/B comparison.

    Parameters
    ----------
    champion_model : PredictModel
        The live (production) model whose signals are actually traded.
    challenger_model : PredictModel | None
        The shadow model being evaluated.  May be ``None`` (no shadow).
    config : ShadowConfig | None
        Configuration.  Uses defaults if not provided.
    """

    def __init__(
        self,
        champion_model: Any,
        challenger_model: Any | None = None,
        config: ShadowConfig | None = None,
    ) -> None:
        self.champion = champion_model
        self.challenger = challenger_model
        self.config = config or ShadowConfig()

        # Signal and return histories
        self.champion_signals: list[float] = []
        self.challenger_signals: list[float] = []
        self.champion_returns: list[float] = []
        self.challenger_returns: list[float] = []

        self.start_date: str | None = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def step(self, features: np.ndarray, actual_return: float) -> dict[str, Any]:
        """Run both models on the same input.

        Only the champion signal should be used for live trading.
        The challenger signal is recorded for offline comparison.

        Parameters
        ----------
        features : np.ndarray
            Feature vector for this timestep.
        actual_return : float
            The realised return for computing hypothetical P&L.

        Returns
        -------
        dict
            Contains ``champion_signal``, ``challenger_signal`` (or NaN),
            and ``agreement`` (bool).
        """
        if self.start_date is None:
            self.start_date = datetime.now(timezone.utc).isoformat()

        # Champion always runs
        champion_signal = float(self.champion.predict(features))
        self.champion_signals.append(champion_signal)
        self.champion_returns.append(champion_signal * actual_return)

        # Challenger may be absent
        if self.challenger is not None:
            challenger_signal = float(self.challenger.predict(features))
        else:
            challenger_signal = float("nan")

        self.challenger_signals.append(challenger_signal)
        if self.challenger is not None:
            self.challenger_returns.append(challenger_signal * actual_return)
        else:
            self.challenger_returns.append(float("nan"))

        self._step_count += 1

        # Agreement: True if both signals have the same sign (or both zero)
        if self.challenger is not None:
            agreement = bool(np.sign(champion_signal) == np.sign(challenger_signal))
        else:
            agreement = False

        result = {
            "champion_signal": champion_signal,
            "challenger_signal": challenger_signal,
            "agreement": agreement,
        }

        # Auto-promote check
        if (
            self.config.auto_promote
            and self.challenger is not None
            and self.should_promote()
        ):
            logger.info("Auto-promoting challenger after %d steps", self._step_count)
            self.promote_challenger()
            result["promoted"] = True

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> ShadowReport:
        """Compare champion vs challenger performance.

        Computes annualized Sharpe (sqrt(252), ddof=1), signal correlation,
        and direction agreement percentage.

        Returns
        -------
        ShadowReport
        """
        n = len(self.champion_returns)
        if n == 0:
            return ShadowReport(n_days=0, recommendation="insufficient_data")

        champ_arr = np.array(self.champion_returns, dtype=np.float64)
        champ_sharpe = _annualized_sharpe(champ_arr)
        champ_total = float(np.sum(champ_arr))

        # If no challenger, report only champion
        has_challenger = (
            len(self.challenger_returns) == n
            and not all(math.isnan(r) for r in self.challenger_returns)
        )

        if not has_challenger:
            return ShadowReport(
                champion_sharpe=champ_sharpe,
                challenger_sharpe=0.0,
                sharpe_improvement=0.0,
                signal_correlation=0.0,
                agreement_pct=0.0,
                n_days=n,
                champion_total_return=champ_total,
                challenger_total_return=0.0,
                recommendation="insufficient_data",
            )

        chall_arr = np.array(self.challenger_returns, dtype=np.float64)
        chall_sharpe = _annualized_sharpe(chall_arr)
        chall_total = float(np.sum(chall_arr))

        # Signal correlation
        champ_sig = np.array(self.champion_signals, dtype=np.float64)
        chall_sig = np.array(self.challenger_signals, dtype=np.float64)

        if np.std(champ_sig) > 0 and np.std(chall_sig) > 0:
            sig_corr = float(np.corrcoef(champ_sig, chall_sig)[0, 1])
        else:
            sig_corr = 0.0

        # Agreement percentage
        agree_count = int(np.sum(np.sign(champ_sig) == np.sign(chall_sig)))
        agreement_pct = agree_count / n if n > 0 else 0.0

        # Recommendation
        improvement = chall_sharpe - champ_sharpe
        if n < self.config.min_evaluation_days:
            recommendation = "insufficient_data"
        elif improvement >= self.config.min_sharpe_improvement:
            recommendation = "promote_challenger"
        else:
            recommendation = "keep_champion"

        return ShadowReport(
            champion_sharpe=champ_sharpe,
            challenger_sharpe=chall_sharpe,
            sharpe_improvement=improvement,
            signal_correlation=sig_corr,
            agreement_pct=agreement_pct,
            n_days=n,
            champion_total_return=champ_total,
            challenger_total_return=chall_total,
            recommendation=recommendation,
        )

    def should_promote(self) -> bool:
        """Check if the challenger should replace the champion.

        Requires:
          1. At least ``min_evaluation_days`` of data.
          2. Challenger Sharpe exceeds champion by ``min_sharpe_improvement``.

        Returns
        -------
        bool
        """
        if self.challenger is None:
            return False
        report = self.evaluate()
        return (
            report.n_days >= self.config.min_evaluation_days
            and report.sharpe_improvement >= self.config.min_sharpe_improvement
        )

    def promote_challenger(self) -> None:
        """Swap the challenger into the champion position.

        After promotion:
          - The old champion is discarded.
          - The challenger becomes the new champion.
          - The challenger slot is set to ``None``.
          - All histories are cleared to start fresh evaluation.
        """
        if self.challenger is None:
            logger.warning("promote_challenger called but no challenger is set")
            return

        logger.info("Promoting challenger to champion")
        self.champion = self.challenger
        self.challenger = None

        # Reset histories
        self.champion_signals.clear()
        self.challenger_signals.clear()
        self.champion_returns.clear()
        self.challenger_returns.clear()
        self.start_date = None
        self._step_count = 0

    def reset(self) -> None:
        """Clear all histories without changing models."""
        self.champion_signals.clear()
        self.challenger_signals.clear()
        self.champion_returns.clear()
        self.challenger_returns.clear()
        self.start_date = None
        self._step_count = 0

    def set_challenger(self, model: Any, checkpoint: str = "") -> None:
        """Set a new challenger model for evaluation.

        Clears existing challenger history.

        Parameters
        ----------
        model : PredictModel
            The new challenger model.
        checkpoint : str
            Optional checkpoint path (for logging/metadata).
        """
        self.challenger = model
        self.config.challenger_checkpoint = checkpoint
        # Clear challenger-specific histories
        self.challenger_signals.clear()
        self.challenger_returns.clear()
        # Also clear champion histories to ensure aligned comparison
        self.champion_signals.clear()
        self.champion_returns.clear()
        self._step_count = 0
        self.start_date = None
        logger.info("New challenger set (checkpoint=%s)", checkpoint)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annualized_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio.

    Uses ddof=1 for sample standard deviation, annualized with sqrt(252).
    Returns 0.0 if fewer than 2 observations or zero variance.
    """
    if len(returns) < 2:
        return 0.0
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    if std_r == 0.0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)
