"""Tests for S13-S24 strategy BaseStrategy compliance.

Verifies each strategy:
1. Can be imported and instantiated
2. Has strategy_id property
3. Has scan() method (via BaseStrategy)
4. Has warmup_days() method
5. Has create_strategy() factory
6. Can be discovered by StrategyRegistry
"""

from __future__ import annotations

import importlib
import pytest

STRATEGY_MODULES = [
    ("quantlaxmi.strategies.s13_hmm_regime.strategy", "s13_hmm_regime"),
    ("quantlaxmi.strategies.s14_ofi_intraday.strategy", "s14_ofi_intraday"),
    ("quantlaxmi.strategies.s15_skew_mr.strategy", "s15_skew_mr"),
    ("quantlaxmi.strategies.s16_vrp_enhanced.strategy", "s16_vrp_enhanced"),
    ("quantlaxmi.strategies.s17_intraday_breakout.strategy", "s17_intraday_breakout"),
    ("quantlaxmi.strategies.s18_straddle_mr.strategy", "s18_straddle_mr"),
    ("quantlaxmi.strategies.s19_crypto_leadlag.strategy", "s19_crypto_leadlag"),
    ("quantlaxmi.strategies.s20_tick_micro.strategy", "s20_tick_micro"),
    ("quantlaxmi.strategies.s21_fii_flow.strategy", "s21_fii_flow"),
    ("quantlaxmi.strategies.s22_gex_dealer.strategy", "s22_gex_dealer"),
    ("quantlaxmi.strategies.s23_network_momentum.strategy", "s23_network_momentum"),
    ("quantlaxmi.strategies.s24_regime_vrp_combo.strategy", "s24_regime_vrp_combo"),
]


@pytest.mark.parametrize("mod_path,expected_id", STRATEGY_MODULES)
def test_create_strategy_factory_exists(mod_path, expected_id):
    """Each module exposes create_strategy() factory."""
    mod = importlib.import_module(mod_path)
    assert hasattr(mod, "create_strategy"), f"{mod_path} missing create_strategy()"
    assert callable(mod.create_strategy)


@pytest.mark.parametrize("mod_path,expected_id", STRATEGY_MODULES)
def test_instantiation(mod_path, expected_id):
    """Strategy can be instantiated via factory."""
    mod = importlib.import_module(mod_path)
    strategy = mod.create_strategy()
    assert strategy is not None


@pytest.mark.parametrize("mod_path,expected_id", STRATEGY_MODULES)
def test_strategy_id(mod_path, expected_id):
    """strategy_id matches expected value."""
    mod = importlib.import_module(mod_path)
    strategy = mod.create_strategy()
    assert strategy.strategy_id == expected_id


@pytest.mark.parametrize("mod_path,expected_id", STRATEGY_MODULES)
def test_has_scan_method(mod_path, expected_id):
    """Strategy has scan() method."""
    mod = importlib.import_module(mod_path)
    strategy = mod.create_strategy()
    assert hasattr(strategy, "scan")
    assert callable(strategy.scan)


@pytest.mark.parametrize("mod_path,expected_id", STRATEGY_MODULES)
def test_has_warmup_days(mod_path, expected_id):
    """Strategy has warmup_days() returning int >= 0."""
    mod = importlib.import_module(mod_path)
    strategy = mod.create_strategy()
    w = strategy.warmup_days()
    assert isinstance(w, int)
    assert w >= 0


@pytest.mark.parametrize("mod_path,expected_id", STRATEGY_MODULES)
def test_inherits_base_strategy(mod_path, expected_id):
    """Strategy inherits from BaseStrategy."""
    from quantlaxmi.strategies.base import BaseStrategy
    mod = importlib.import_module(mod_path)
    strategy = mod.create_strategy()
    assert isinstance(strategy, BaseStrategy)


def test_registry_discovers_all():
    """All S13-S24 strategies are discoverable by registry."""
    from quantlaxmi.strategies.registry import StrategyRegistry
    registry = StrategyRegistry()
    registry.discover()

    for _, expected_id in STRATEGY_MODULES:
        assert expected_id in registry, f"{expected_id} not in registry"
