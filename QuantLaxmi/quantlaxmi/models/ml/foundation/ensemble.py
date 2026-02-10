"""Foundation model ensemble — combines Chronos with TFT predictions.

Implements dynamic weighting based on recent out-of-sample performance
(exponential moving Sharpe ratio).

Usage
-----
    from quantlaxmi.models.ml.foundation import FoundationEnsemble

    ens = FoundationEnsemble(chronos_weight=0.4, tft_weight=0.6)
    signal = ens.combine(chronos_signal=0.7, tft_signal=0.3)

    # Dynamic weighting
    ens.update_performance("chronos", daily_return=0.005)
    ens.update_performance("tft", daily_return=-0.002)
    w = ens.get_dynamic_weights()

References
----------
- Timmermann (2006), "Forecast Combinations"
- Genre et al. (2013), "Combining expert forecasts: Can anything beat the
  simple average?"
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    """Configuration for :class:`FoundationEnsemble`.

    Parameters
    ----------
    chronos_weight : float
        Static weight for Chronos forecasts.  Must be in [0, 1].
    tft_weight : float
        Static weight for TFT forecasts.  Must be in [0, 1].
        ``chronos_weight + tft_weight`` need not equal 1; they are
        normalized internally.
    use_dynamic_weights : bool
        If ``True``, weights are re-estimated daily from recent Sharpe.
    ema_halflife : int
        Half-life (in days) for the exponential moving Sharpe window.
    performance_window : int
        Maximum lookback (days) for Sharpe estimation.
    min_history : int
        Minimum number of observations before switching to dynamic.
    """

    chronos_weight: float = 0.4
    tft_weight: float = 0.6
    use_dynamic_weights: bool = True
    ema_halflife: int = 21
    performance_window: int = 63
    min_history: int = 10


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FoundationEnsemble:
    """Combines Chronos and TFT signals via weighted average.

    Supports both static and dynamic (performance-adaptive) weighting.

    Parameters
    ----------
    chronos_weight : float
        Default Chronos weight.
    tft_weight : float
        Default TFT weight.
    use_dynamic_weights : bool
        Adapt weights based on recent Sharpe.
    config : EnsembleConfig, optional
        Full configuration.  Overrides individual kwargs.
    """

    def __init__(
        self,
        chronos_weight: float = 0.4,
        tft_weight: float = 0.6,
        use_dynamic_weights: bool = True,
        config: EnsembleConfig | None = None,
    ) -> None:
        if config is not None:
            self.cfg = config
        else:
            self.cfg = EnsembleConfig(
                chronos_weight=chronos_weight,
                tft_weight=tft_weight,
                use_dynamic_weights=use_dynamic_weights,
            )

        # Performance histories for dynamic weighting
        self._returns: dict[str, deque] = {
            "chronos": deque(maxlen=self.cfg.performance_window),
            "tft": deque(maxlen=self.cfg.performance_window),
        }

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def combine(
        self,
        chronos_signal: float,
        tft_signal: float,
        use_dynamic: bool | None = None,
    ) -> float:
        """Combine two model signals into a single position signal.

        Parameters
        ----------
        chronos_signal : float
            Chronos model signal in [-1, 1].
        tft_signal : float
            TFT model signal in [-1, 1].
        use_dynamic : bool, optional
            Override ``config.use_dynamic_weights``.

        Returns
        -------
        float
            Combined signal in [-1, 1].
        """
        dynamic = use_dynamic if use_dynamic is not None else self.cfg.use_dynamic_weights

        if dynamic:
            weights = self.get_dynamic_weights()
        else:
            weights = self._normalize_weights(
                self.cfg.chronos_weight, self.cfg.tft_weight
            )

        combined = weights["chronos"] * chronos_signal + weights["tft"] * tft_signal
        return float(np.clip(combined, -1.0, 1.0))

    def combine_multi(
        self,
        signals: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Combine signals from an arbitrary number of models.

        Parameters
        ----------
        signals : dict[str, float]
            Model name to signal value mapping.
        weights : dict[str, float], optional
            Model name to weight mapping.  If ``None``, uses equal weights.

        Returns
        -------
        float
            Combined signal in [-1, 1].
        """
        if not signals:
            return 0.0

        if weights is None:
            w = {k: 1.0 / len(signals) for k in signals}
        else:
            total = sum(abs(v) for v in weights.values()) or 1.0
            w = {k: abs(v) / total for k, v in weights.items()}

        combined = sum(w.get(k, 0.0) * v for k, v in signals.items())
        return float(np.clip(combined, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def update_performance(self, model_name: str, daily_return: float) -> None:
        """Record a daily return for a model, used for dynamic weighting.

        Parameters
        ----------
        model_name : str
            ``"chronos"`` or ``"tft"`` (or any registered name).
        daily_return : float
            The model's P&L return for the day.
        """
        if model_name not in self._returns:
            self._returns[model_name] = deque(maxlen=self.cfg.performance_window)
        self._returns[model_name].append(daily_return)

    def get_dynamic_weights(self) -> dict[str, float]:
        """Compute weights from exponential-moving Sharpe ratios.

        Falls back to static weights when insufficient history is available.

        Returns
        -------
        dict[str, float]
            Normalized weights summing to 1.0.
        """
        sharpes: dict[str, float] = {}

        for name, returns in self._returns.items():
            if len(returns) < self.cfg.min_history:
                # Not enough data — fall back to static
                return self._normalize_weights(
                    self.cfg.chronos_weight, self.cfg.tft_weight
                )

            arr = np.array(returns)

            # Exponentially weighted Sharpe
            alpha = 1.0 - math.exp(-math.log(2) / self.cfg.ema_halflife)
            n = len(arr)
            weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()

            mean_r = float(np.sum(weights * arr))
            var_r = float(np.sum(weights * (arr - mean_r) ** 2))
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-8

            sharpes[name] = mean_r / std_r * math.sqrt(252)

        if not sharpes:
            return self._normalize_weights(
                self.cfg.chronos_weight, self.cfg.tft_weight
            )

        # Convert Sharpe to weights via softmax with temperature
        temperature = 2.0
        vals = np.array(list(sharpes.values()))
        exp_vals = np.exp(vals / temperature - np.max(vals / temperature))
        softmax = exp_vals / exp_vals.sum()

        result = {}
        for i, name in enumerate(sharpes.keys()):
            result[name] = float(softmax[i])

        # Ensure both models are in result
        for name in ["chronos", "tft"]:
            if name not in result:
                result[name] = 0.0

        return result

    def get_static_weights(self) -> dict[str, float]:
        """Return the static (configured) weights, normalized.

        Returns
        -------
        dict[str, float]
        """
        return self._normalize_weights(
            self.cfg.chronos_weight, self.cfg.tft_weight
        )

    # ------------------------------------------------------------------
    # Model agreement / disagreement metrics
    # ------------------------------------------------------------------

    @staticmethod
    def agreement_score(chronos_signal: float, tft_signal: float) -> float:
        """Measure agreement between two model signals.

        Returns a score in [0, 1] where 1 = perfect agreement (same sign
        and magnitude) and 0 = maximum disagreement (opposite signs).

        Parameters
        ----------
        chronos_signal : float
            Signal in [-1, 1].
        tft_signal : float
            Signal in [-1, 1].

        Returns
        -------
        float
            Agreement score in [0, 1].
        """
        if chronos_signal == 0.0 and tft_signal == 0.0:
            return 1.0

        # Cosine similarity in 1D (sign agreement + magnitude similarity)
        dot = chronos_signal * tft_signal
        mag = max(abs(chronos_signal), abs(tft_signal)) ** 2
        if mag == 0:
            return 1.0
        return float(np.clip((dot / mag + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def confidence_from_agreement(
        chronos_signal: float,
        tft_signal: float,
        base_confidence: float = 0.5,
    ) -> float:
        """Derive confidence from model agreement.

        When both models agree, confidence is boosted.  When they disagree,
        confidence is reduced.

        Parameters
        ----------
        chronos_signal : float
        tft_signal : float
        base_confidence : float
            Baseline confidence before agreement adjustment.

        Returns
        -------
        float
            Adjusted confidence in [0, 1].
        """
        agreement = FoundationEnsemble.agreement_score(chronos_signal, tft_signal)
        # Boost: agree → up to 1.0, disagree → down to 0.0
        adjusted = base_confidence * (0.5 + agreement)
        return float(np.clip(adjusted, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weights(w_chronos: float, w_tft: float) -> dict[str, float]:
        total = abs(w_chronos) + abs(w_tft)
        if total == 0:
            return {"chronos": 0.5, "tft": 0.5}
        return {
            "chronos": abs(w_chronos) / total,
            "tft": abs(w_tft) / total,
        }

    def __repr__(self) -> str:
        w = self.get_static_weights()
        return (
            f"FoundationEnsemble(chronos={w['chronos']:.2f}, "
            f"tft={w['tft']:.2f}, dynamic={self.cfg.use_dynamic_weights})"
        )
