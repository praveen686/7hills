"""RL agents for trading — Thompson Allocator, Deep Hedger, Execution, Market Maker, Kelly Sizer.

Production-ready RL agent wrappers that connect the book's MDP theory
to real India FnO + crypto markets.

Agents:
  - ThompsonStrategyAllocator: contextual Thompson Sampling for strategy selection
  - DeepHedgingAgent: neural hedging for options portfolios
  - OptimalExecutionAgent: RL-based optimal order execution
  - MarketMakingAgent: Avellaneda-Stoikov + RL market maker
  - KellySizer: Kelly-Merton dynamic position sizing
"""

from .thompson_allocator import ThompsonStrategyAllocator
from .deep_hedger import DeepHedgingAgent
from .execution_agent import OptimalExecutionAgent
from .market_maker import MarketMakingAgent
from .kelly_sizer import KellySizer

__all__ = [
    "ThompsonStrategyAllocator",
    "DeepHedgingAgent",
    "OptimalExecutionAgent",
    "MarketMakingAgent",
    "KellySizer",
]
