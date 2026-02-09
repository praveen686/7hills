"""Financial applications of RL — Chapters 8-10 of Rao & Jelvis.

Ch 8:  Dynamic Asset Allocation (Merton's Portfolio Problem)
Ch 9:  Derivatives Pricing & Deep Hedging
Ch 10: Optimal Execution (Bertsimas-Lo, Almgren-Chriss)
       Market-Making (Avellaneda-Stoikov)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Ch 8: Asset Allocation
# ---------------------------------------------------------------------------
from models.rl.finance.asset_allocation import (
    AssetAllocationMDP,
    AssetAllocState,
    MertonSolution,
    AssetAllocPG,
    MultiStrategyAllocator,
)

# ---------------------------------------------------------------------------
# Ch 9: Derivatives Pricing & Hedging
# ---------------------------------------------------------------------------
from models.rl.finance.derivatives_pricing import (
    DerivativePricingMDP,
    HedgingState,
    AmericanOptionMDP,
    AmericanOptionState,
    BlackScholesHedger,
    DeepHedger,
    MaxExpUtility,
)

# ---------------------------------------------------------------------------
# Ch 10.2: Optimal Execution
# ---------------------------------------------------------------------------
from models.rl.finance.optimal_execution import (
    OrderExecutionMDP,
    ExecutionState,
    BertsimasLoSolution,
    AlmgrenChrissSolution,
    RLExecutionAgent,
)

# ---------------------------------------------------------------------------
# Ch 10.3: Market-Making
# ---------------------------------------------------------------------------
from models.rl.finance.market_making import (
    MarketMakingMDP,
    MarketMakingState,
    AvellanedaStoikovSolution,
    RLMarketMaker,
    InventoryRiskManager,
)

__all__ = [
    # Asset Allocation (Ch 8)
    "AssetAllocationMDP",
    "AssetAllocState",
    "MertonSolution",
    "AssetAllocPG",
    "MultiStrategyAllocator",
    # Derivatives Pricing (Ch 9)
    "DerivativePricingMDP",
    "HedgingState",
    "AmericanOptionMDP",
    "AmericanOptionState",
    "BlackScholesHedger",
    "DeepHedger",
    "MaxExpUtility",
    # Optimal Execution (Ch 10.2)
    "OrderExecutionMDP",
    "ExecutionState",
    "BertsimasLoSolution",
    "AlmgrenChrissSolution",
    "RLExecutionAgent",
    # Market-Making (Ch 10.3)
    "MarketMakingMDP",
    "MarketMakingState",
    "AvellanedaStoikovSolution",
    "RLMarketMaker",
    "InventoryRiskManager",
]
