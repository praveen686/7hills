"""Strategy registry — discover and manage strategy instances.

Strategies register themselves via ``StrategyRegistry.register()`` or
are auto-discovered from known module paths.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantlaxmi.strategies.protocol import StrategyProtocol

logger = logging.getLogger(__name__)

# Well-known strategy modules — imported lazily during discover()
_KNOWN_MODULES = [
    # Original strategies
    "quantlaxmi.strategies.s1_vrp.strategy",
    "quantlaxmi.strategies.s2_ramanujan.strategy",
    "quantlaxmi.strategies.s3_institutional.strategy",
    "quantlaxmi.strategies.s4_iv_mr.strategy",
    "quantlaxmi.strategies.s5_hawkes.strategy",
    "quantlaxmi.strategies.s6_multi_factor.strategy",
    "quantlaxmi.strategies.s7_regime.strategy",
    "quantlaxmi.strategies.s8_expiry_theta.strategy",
    "quantlaxmi.strategies.s9_momentum.strategy",
    "quantlaxmi.strategies.s10_gamma_scalp.strategy",
    "quantlaxmi.strategies.s11_pairs.strategy",
    # Alpha-forge strategies (S12-S24)
    "quantlaxmi.strategies.s12_vedic_ffpe.strategy",
    "quantlaxmi.strategies.s13_hmm_regime.strategy",
    "quantlaxmi.strategies.s14_ofi_intraday.strategy",
    "quantlaxmi.strategies.s15_skew_mr.strategy",
    "quantlaxmi.strategies.s16_vrp_enhanced.strategy",
    "quantlaxmi.strategies.s17_intraday_breakout.strategy",
    "quantlaxmi.strategies.s18_straddle_mr.strategy",
    "quantlaxmi.strategies.s19_crypto_leadlag.strategy",
    "quantlaxmi.strategies.s20_tick_micro.strategy",
    "quantlaxmi.strategies.s21_fii_flow.strategy",
    "quantlaxmi.strategies.s22_gex_dealer.strategy",
    "quantlaxmi.strategies.s23_network_momentum.strategy",
    "quantlaxmi.strategies.s24_regime_vrp_combo.strategy",
    # Advanced strategies
    "quantlaxmi.strategies.s25_divergence_flow.strategy",
    "quantlaxmi.strategies.s26_crypto_flow.strategy",
    # ML strategies
    "quantlaxmi.models.ml.tft.production.strategy_adapter",
]


class StrategyRegistry:
    """Singleton-style registry for strategy instances."""

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyProtocol] = {}

    def register(self, strategy: StrategyProtocol) -> None:
        sid = strategy.strategy_id
        if sid in self._strategies:
            logger.warning("Strategy %s already registered, replacing", sid)
        self._strategies[sid] = strategy
        logger.info("Registered strategy: %s", sid)

    def get(self, strategy_id: str) -> StrategyProtocol | None:
        return self._strategies.get(strategy_id)

    def list_ids(self) -> list[str]:
        return sorted(self._strategies.keys())

    def all(self) -> list[StrategyProtocol]:
        return list(self._strategies.values())

    def discover(self, extra_modules: list[str] | None = None) -> None:
        """Import known strategy modules to trigger registration.

        Each module is expected to call ``register()`` on a global
        registry instance or export a factory function.
        """
        modules = _KNOWN_MODULES + (extra_modules or [])
        for mod_path in modules:
            try:
                mod = importlib.import_module(mod_path)
                # Convention: module exposes ``create_strategy()`` factory
                factory = getattr(mod, "create_strategy", None)
                if factory is not None:
                    strategy = factory()
                    self.register(strategy)
                else:
                    logger.debug("Module %s has no create_strategy(), skipping", mod_path)
            except ImportError as e:
                logger.debug("Could not import %s: %s", mod_path, e)
            except Exception as e:
                logger.warning("Error loading strategy from %s: %s", mod_path, e)

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies


# Module-level default registry
default_registry = StrategyRegistry()
