"""Trading MDP environments for India FnO, Crypto, Options, Execution.

Provides Gymnasium-like environments for RL agent training and evaluation.
All environments enforce NO look-ahead bias (strictly causal data access).

Environments:
  - TradingEnv: abstract base class
  - SimulatedPriceEnv: parametric price dynamics (GBM, Heston, jumps, OU)
  - IndiaFnOEnv: India Futures & Options (NIFTY, BANKNIFTY, etc.)
  - NiftySwingEnv: NIFTY swing trading specialisation
  - BankNiftyScalpEnv: BANKNIFTY intraday scalping specialisation
  - CryptoEnv: Binance crypto perps (BTC, ETH, SOL)
  - CryptoFundingEnv: funding rate carry trade
  - CryptoLeadLagEnv: cross-market lead-lag (BTC/ETH -> NIFTY)
  - OptionsEnv: options trading with Greeks
  - GammaScalpEnv: long gamma, hedge delta (S10)
  - ThetaDecayEnv: short iron condors (S8)
  - IVMeanReversionEnv: IV mean reversion (S4)
  - ExecutionEnv: order execution with LOB simulation
  - LOBSimulator: lightweight limit order book
"""

from .trading_env import (
    TradingState,
    TradingAction,
    StepResult,
    TradingEnv,
    SimulatedPriceEnv,
)
from .india_fno_env import (
    IndiaFnOEnv,
    NiftySwingEnv,
    BankNiftyScalpEnv,
)
from .crypto_env import (
    CryptoEnv,
    CryptoFundingEnv,
    CryptoLeadLagEnv,
)
from .options_env import (
    OptionsEnv,
    GammaScalpEnv,
    ThetaDecayEnv,
    IVMeanReversionEnv,
)
from .execution_env import (
    ExecutionEnv,
    LOBSimulator,
)

__all__ = [
    # Base
    "TradingState",
    "TradingAction",
    "StepResult",
    "TradingEnv",
    "SimulatedPriceEnv",
    # India FnO
    "IndiaFnOEnv",
    "NiftySwingEnv",
    "BankNiftyScalpEnv",
    # Crypto
    "CryptoEnv",
    "CryptoFundingEnv",
    "CryptoLeadLagEnv",
    # Options
    "OptionsEnv",
    "GammaScalpEnv",
    "ThetaDecayEnv",
    "IVMeanReversionEnv",
    # Execution
    "ExecutionEnv",
    "LOBSimulator",
]
