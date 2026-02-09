"""Chapter 9: Derivatives Pricing and Hedging

Key insight from Rao & Jelvis: pricing and hedging derivatives in INCOMPLETE
markets is fundamentally an MDP. The classical Black-Scholes framework assumes
complete markets (continuous hedging, no transaction costs, GBM dynamics).
When any of these assumptions fail, the market is incomplete and perfect
replication is impossible.

The RL approach:
1. Define MDP where agent hedges a derivative by trading the underlying
2. Reward = utility of terminal hedging PnL
3. Optimal policy = optimal hedge ratios at each time step
4. Price = certainty equivalent of optimal hedging PnL

This generalizes Black-Scholes to handle:
- Transaction costs (bid-ask spread, commissions)
- Discrete hedging (rebalance at most daily)
- Stochastic volatility (Heston, SABR)
- Jumps in underlying (Merton jump-diffusion)
- Path-dependent payoffs

The Maximum Expected Utility (MEU) pricing framework:
    For seller: p_seller = CE[-V*(0)]
    For buyer:  p_buyer = -CE[V*(0)]
    In complete markets: p_seller = p_buyer = BS price
    In incomplete markets: p_buyer <= p_complete <= p_seller

References:
    Rao & Jelvis, Ch 9 -- "Derivatives Pricing and Hedging"
    Buehler et al. (2019), "Deep Hedging"
    Hull (2022), "Options, Futures, and Other Derivatives"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Graceful imports from sibling packages
# ---------------------------------------------------------------------------
try:
    from models.rl.core.markov_process import (
        MarkovDecisionProcess,
        NonTerminal,
        Terminal,
        State,
        Distribution,
        SampledDistribution,
    )
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal as TorchNormal

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

__all__ = [
    "DerivativePricingMDP",
    "AmericanOptionMDP",
    "BlackScholesHedger",
    "DeepHedger",
    "MaxExpUtility",
]


def _get_device(device: str = "auto") -> str:
    if device == "auto":
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


# ============================================================================
# BlackScholesHedger -- Analytical benchmark
# ============================================================================


class BlackScholesHedger:
    """Black-Scholes option pricing and greeks (benchmark).

    Under the BS assumptions (complete market, GBM dynamics, continuous hedging,
    no transaction costs), we have closed-form solutions.

    BS PDE:  dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

    Call price:  C = S*N(d1) - K*exp(-r*tau)*N(d2)
    Put price:   P = K*exp(-r*tau)*N(-d2) - S*N(-d1)

    where:
        d1 = [log(S/K) + (r + sigma^2/2)*tau] / (sigma*sqrt(tau))
        d2 = d1 - sigma*sqrt(tau)
        N(.) = standard normal CDF

    Greeks:
        Delta_call = N(d1),  Delta_put = N(d1) - 1
        Gamma = n(d1) / (S * sigma * sqrt(tau))
        Vega  = S * n(d1) * sqrt(tau)
        Theta_call = -(S*n(d1)*sigma)/(2*sqrt(tau)) - r*K*exp(-r*tau)*N(d2)

    where n(.) = standard normal PDF.

    References:
        Black & Scholes (1973), "The Pricing of Options and Corporate Liabilities"
        Rao & Jelvis Ch 9, Section 9.1
    """

    @staticmethod
    def _d1d2(
        spot: float,
        strike: float,
        tau: float,
        sigma: float,
        r: float,
    ) -> Tuple[float, float]:
        """Compute d1 and d2 for Black-Scholes formula.

        d1 = [log(S/K) + (r + sigma^2/2)*tau] / (sigma*sqrt(tau))
        d2 = d1 - sigma*sqrt(tau)
        """
        if tau <= 0 or sigma <= 0:
            # At expiry or zero vol: intrinsic only
            d1 = float("inf") if spot > strike else float("-inf")
            d2 = d1
            return d1, d2
        sqrt_tau = math.sqrt(tau)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (
            sigma * sqrt_tau
        )
        d2 = d1 - sigma * sqrt_tau
        return d1, d2

    @staticmethod
    def price(
        spot: float,
        strike: float,
        tau: float,
        sigma: float,
        r: float,
        option_type: str = "call",
    ) -> float:
        """Black-Scholes option price.

        Parameters
        ----------
        spot : float
            Current underlying price S.
        strike : float
            Strike price K.
        tau : float
            Time to expiry in years.
        sigma : float
            Annualized volatility.
        r : float
            Risk-free rate.
        option_type : str
            "call" or "put".

        Returns
        -------
        float
            Option price.
        """
        if tau <= 0:
            # At expiry: intrinsic value
            if option_type == "call":
                return max(spot - strike, 0.0)
            else:
                return max(strike - spot, 0.0)

        d1, d2 = BlackScholesHedger._d1d2(spot, strike, tau, sigma, r)
        df = math.exp(-r * tau)

        if option_type == "call":
            return spot * stats.norm.cdf(d1) - strike * df * stats.norm.cdf(d2)
        elif option_type == "put":
            return strike * df * stats.norm.cdf(-d2) - spot * stats.norm.cdf(-d1)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    @staticmethod
    def delta(
        spot: float,
        strike: float,
        tau: float,
        sigma: float,
        r: float,
        option_type: str = "call",
    ) -> float:
        """Black-Scholes delta: dV/dS.

        Delta_call = N(d1)
        Delta_put  = N(d1) - 1
        """
        if tau <= 0:
            if option_type == "call":
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        d1, _ = BlackScholesHedger._d1d2(spot, strike, tau, sigma, r)
        if option_type == "call":
            return float(stats.norm.cdf(d1))
        elif option_type == "put":
            return float(stats.norm.cdf(d1) - 1.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    @staticmethod
    def gamma(
        spot: float,
        strike: float,
        tau: float,
        sigma: float,
        r: float,
    ) -> float:
        """Black-Scholes gamma: d2V/dS2.

        Gamma = n(d1) / (S * sigma * sqrt(tau))

        Same for calls and puts (put-call parity).
        """
        if tau <= 0 or sigma <= 0:
            return 0.0

        d1, _ = BlackScholesHedger._d1d2(spot, strike, tau, sigma, r)
        return float(stats.norm.pdf(d1) / (spot * sigma * math.sqrt(tau)))

    @staticmethod
    def vega(
        spot: float,
        strike: float,
        tau: float,
        sigma: float,
        r: float,
    ) -> float:
        """Black-Scholes vega: dV/dsigma.

        Vega = S * n(d1) * sqrt(tau)

        Same for calls and puts.
        """
        if tau <= 0:
            return 0.0

        d1, _ = BlackScholesHedger._d1d2(spot, strike, tau, sigma, r)
        return float(spot * stats.norm.pdf(d1) * math.sqrt(tau))

    @staticmethod
    def theta(
        spot: float,
        strike: float,
        tau: float,
        sigma: float,
        r: float,
        option_type: str = "call",
    ) -> float:
        """Black-Scholes theta: dV/dt (per year).

        Theta_call = -(S*n(d1)*sigma)/(2*sqrt(tau)) - r*K*exp(-r*tau)*N(d2)
        Theta_put  = -(S*n(d1)*sigma)/(2*sqrt(tau)) + r*K*exp(-r*tau)*N(-d2)
        """
        if tau <= 0:
            return 0.0

        d1, d2 = BlackScholesHedger._d1d2(spot, strike, tau, sigma, r)
        sqrt_tau = math.sqrt(tau)
        df = math.exp(-r * tau)
        common = -(spot * stats.norm.pdf(d1) * sigma) / (2.0 * sqrt_tau)

        if option_type == "call":
            return float(common - r * strike * df * stats.norm.cdf(d2))
        elif option_type == "put":
            return float(common + r * strike * df * stats.norm.cdf(-d2))
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    @staticmethod
    def implied_vol(
        market_price: float,
        spot: float,
        strike: float,
        tau: float,
        r: float,
        option_type: str = "call",
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """Implied volatility via Newton-Raphson.

        Find sigma such that BS(sigma) = market_price.
        Uses Vega as the derivative for Newton's method.
        """
        sigma = 0.3  # initial guess
        for _ in range(max_iter):
            price = BlackScholesHedger.price(spot, strike, tau, sigma, r, option_type)
            v = BlackScholesHedger.vega(spot, strike, tau, sigma, r)
            if abs(v) < 1e-12:
                break
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / v
            sigma = max(sigma, 0.001)  # keep sigma positive
        return sigma


# ============================================================================
# DerivativePricingMDP -- Chapter 9
# ============================================================================


@dataclass
class HedgingState:
    """State for the derivative hedging MDP.

    Attributes:
        time_step: current step t in [0, T]
        spot_price: current underlying price S_t
        position: current hedge position (units of underlying held)
        cash: accumulated cash from trading
    """
    time_step: int
    spot_price: float
    position: float
    cash: float

    def __hash__(self) -> int:
        return hash((
            self.time_step,
            round(self.spot_price, 4),
            round(self.position, 4),
        ))


class DerivativePricingMDP:
    """Chapter 9: Derivatives pricing as MDP.

    The agent is SHORT a derivative and must hedge by trading the underlying.

    State: (time, spot_price, position, cash)
    Action: target hedge ratio h_t (position as fraction of 1 unit of underlying)
            Trade size = h_t - current_position
    Transition: spot evolves by GBM, position updated to h_t
    Reward: At terminal step only:
        U(cash + position*S_T - payoff(S_T))
        = U(hedging_PnL)

    The MDP naturally handles:
    - Discrete hedging (can only trade at each time step)
    - Transaction costs (proportional to |trade_size|)
    - Any payoff function

    Parameters
    ----------
    spot_price : float
        Initial underlying price S_0.
    strike : float
        Strike price K.
    expiry_steps : int
        Number of time steps until expiry.
    mu : float
        Drift of underlying (annualized). Under risk-neutral measure, mu = r.
    sigma : float
        Volatility of underlying (annualized).
    risk_free_rate : float
        Risk-free rate.
    payoff_fn : Callable[[float], float]
        payoff(S_T) at expiry. E.g., max(S_T - K, 0) for call.
    num_hedge_actions : int
        Number of discretized hedge ratios (for tabular methods).
        Actions are linspace(-1, 1, num_hedge_actions).
    transaction_cost : float
        Proportional transaction cost (fraction of trade value).
    utility : str
        "exponential" for CARA: U(x) = -exp(-alpha*x)
    risk_aversion : float
        Risk aversion parameter alpha.
    dt : float
        Time step in years.
    """

    def __init__(
        self,
        spot_price: float = 100.0,
        strike: float = 100.0,
        expiry_steps: int = 63,
        mu: float = 0.05,
        sigma: float = 0.20,
        risk_free_rate: float = 0.05,
        payoff_fn: Optional[Callable[[float], float]] = None,
        num_hedge_actions: int = 21,
        transaction_cost: float = 0.001,
        utility: str = "exponential",
        risk_aversion: float = 1.0,
        dt: float = 1.0 / 252,
    ) -> None:
        self.spot_price = spot_price
        self.strike = strike
        self.expiry_steps = expiry_steps
        self.mu = mu
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.utility = utility
        self.risk_aversion = risk_aversion
        self.dt = dt
        self.num_hedge_actions = num_hedge_actions

        # Default payoff: European call
        if payoff_fn is None:
            self.payoff_fn = lambda s: max(s - strike, 0.0)
        else:
            self.payoff_fn = payoff_fn

        # Discretized hedge ratios for tabular methods
        self.hedge_actions = np.linspace(-1.0, 1.0, num_hedge_actions)

    def utility_fn(self, pnl: float) -> float:
        """Utility of terminal hedging PnL."""
        if self.utility == "exponential":
            # CARA utility: U(x) = -exp(-alpha * x)
            return -math.exp(-self.risk_aversion * pnl)
        elif self.utility == "quadratic":
            # Mean-variance: U(x) = x - 0.5 * alpha * x^2
            return pnl - 0.5 * self.risk_aversion * pnl ** 2
        else:
            raise ValueError(f"Unknown utility: {self.utility}")

    def step(
        self,
        state: HedgingState,
        target_hedge: float,
        rng: np.random.Generator,
    ) -> Tuple[HedgingState, float, bool]:
        """Execute one hedging step.

        Parameters
        ----------
        state : HedgingState
        target_hedge : float
            Target position (units of underlying).
        rng : np.random.Generator

        Returns
        -------
        next_state, reward, done
        """
        # Trade to reach target position
        trade_size = target_hedge - state.position
        trade_cost = abs(trade_size * state.spot_price) * self.transaction_cost
        cash_change = -trade_size * state.spot_price - trade_cost

        # Evolve spot price: GBM with drift mu
        # S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        z = rng.standard_normal()
        log_return = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * math.sqrt(self.dt) * z
        new_spot = state.spot_price * math.exp(log_return)

        # Interest on cash
        new_cash = (state.cash + cash_change) * math.exp(self.risk_free_rate * self.dt)

        next_time = state.time_step + 1
        done = next_time >= self.expiry_steps

        # Reward: 0 at intermediate, utility of PnL at terminal
        reward = 0.0
        if done:
            # Terminal PnL = cash + position * S_T - payoff(S_T)
            terminal_pnl = new_cash + target_hedge * new_spot - self.payoff_fn(new_spot)
            reward = self.utility_fn(terminal_pnl)

        next_state = HedgingState(
            time_step=next_time,
            spot_price=new_spot,
            position=target_hedge,
            cash=new_cash,
        )
        return next_state, reward, done

    def initial_state(self) -> HedgingState:
        """Create initial hedging state.

        The seller receives the option premium upfront.
        Initial cash = BS price (or 0 if pricing from scratch).
        """
        # Compute initial BS price as the premium received
        tau = self.expiry_steps * self.dt
        premium = BlackScholesHedger.price(
            self.spot_price, self.strike, tau, self.sigma, self.risk_free_rate
        )
        return HedgingState(
            time_step=0,
            spot_price=self.spot_price,
            position=0.0,
            cash=premium,
        )

    def state_to_features(self, state: HedgingState) -> np.ndarray:
        """Convert state to feature vector for neural network.

        Features:
            [0] tau = time_remaining / total_time
            [1] moneyness = log(S_t / K)
            [2] current_position
            [3] normalized_cash = cash / (K * initial_BS_price_fraction)
            [4] realized_vol_proxy (could be computed from history)
        """
        tau = (self.expiry_steps - state.time_step) / max(self.expiry_steps, 1)
        moneyness = math.log(max(state.spot_price, 1e-10) / self.strike)
        return np.array([
            tau,
            moneyness,
            state.position,
            state.cash / max(self.spot_price, 1e-10),
        ], dtype=np.float64)


# ============================================================================
# AmericanOptionMDP -- Chapter 9.3: Optimal Stopping
# ============================================================================


@dataclass
class AmericanOptionState:
    """State for American option pricing MDP."""
    time_step: int
    spot_price: float

    def __hash__(self) -> int:
        return hash((self.time_step, round(self.spot_price, 4)))


class AmericanOptionMDP:
    """Chapter 9.3: American option pricing as Optimal Stopping MDP.

    State: (time, spot_price)
    Action: {0 = continue, 1 = exercise}
    Transition: spot evolves by GBM if continue
    Reward: payoff(S_t) if exercise, 0 if continue

    The optimal exercise boundary is the continuation value:
        Exercise if payoff(S_t) >= E[V(t+1, S_{t+1}) | S_t]

    This is equivalent to the Longstaff-Schwartz (2001) approach but
    formulated as an MDP solved by backward induction or RL.

    The American option premium over European is:
        V_american - V_european >= 0 (early exercise premium)

    For calls on non-dividend-paying stocks: early exercise is never optimal
    (the premium is always zero), so only American puts are interesting
    in the standard GBM model.

    Parameters
    ----------
    spot_price : float
        Initial spot price.
    strike : float
        Strike price.
    expiry_steps : int
        Number of exercise opportunities.
    mu : float
        Drift (use risk_free_rate for risk-neutral pricing).
    sigma : float
        Volatility.
    risk_free_rate : float
        Risk-free rate for discounting.
    option_type : str
        "call" or "put".
    dt : float
        Time per step.
    num_price_levels : int
        For discretized solving, number of price grid points.
    """

    def __init__(
        self,
        spot_price: float = 100.0,
        strike: float = 100.0,
        expiry_steps: int = 50,
        mu: float = 0.05,
        sigma: float = 0.20,
        risk_free_rate: float = 0.05,
        option_type: str = "put",
        dt: float = 1.0 / 252,
        num_price_levels: int = 100,
    ) -> None:
        self.spot_price = spot_price
        self.strike = strike
        self.expiry_steps = expiry_steps
        self.mu = mu
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate
        self.option_type = option_type
        self.dt = dt
        self.num_price_levels = num_price_levels

    def payoff(self, spot: float) -> float:
        """Intrinsic value (exercise payoff)."""
        if self.option_type == "put":
            return max(self.strike - spot, 0.0)
        elif self.option_type == "call":
            return max(spot - self.strike, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {self.option_type}")

    def step(
        self,
        state: AmericanOptionState,
        action: int,
        rng: np.random.Generator,
    ) -> Tuple[AmericanOptionState, float, bool]:
        """Execute one step.

        Parameters
        ----------
        state : AmericanOptionState
        action : int
            0 = continue, 1 = exercise
        rng : np.random.Generator

        Returns
        -------
        next_state, reward, done
        """
        if action == 1:
            # Exercise: receive payoff, episode ends
            reward = self.payoff(state.spot_price)
            terminal = AmericanOptionState(
                time_step=state.time_step,
                spot_price=state.spot_price,
            )
            return terminal, reward, True

        # Continue: evolve price
        z = rng.standard_normal()
        log_ret = (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * math.sqrt(self.dt) * z
        new_spot = state.spot_price * math.exp(log_ret)

        next_time = state.time_step + 1
        done = next_time >= self.expiry_steps

        # At expiry, must exercise if ITM
        reward = self.payoff(new_spot) if done else 0.0

        next_state = AmericanOptionState(
            time_step=next_time,
            spot_price=new_spot,
        )
        return next_state, reward, done

    def price_by_simulation(
        self,
        num_paths: int = 100000,
        seed: int = 42,
    ) -> float:
        """Price American option via Longstaff-Schwartz Monte Carlo.

        The LS algorithm (Longstaff & Schwartz, 2001):
        1. Simulate paths forward
        2. At each time step (backward), regress continuation value on
           basis functions of spot price
        3. Exercise if intrinsic > continuation value estimate

        This is the standard benchmark for American option pricing
        and is equivalent to approximate backward induction with
        linear function approximation (Ch 9.3, Rao & Jelvis).

        Returns
        -------
        float
            Estimated American option price (risk-neutral).
        """
        rng = np.random.default_rng(seed)
        T = self.expiry_steps
        df = math.exp(-self.risk_free_rate * self.dt)

        # Simulate all paths under risk-neutral measure (mu = r)
        # S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        z = rng.standard_normal((num_paths, T))
        log_rets = (self.risk_free_rate - 0.5 * self.sigma ** 2) * self.dt + self.sigma * math.sqrt(self.dt) * z
        log_prices = np.log(self.spot_price) + np.cumsum(log_rets, axis=1)
        prices = np.exp(log_prices)
        # Prepend initial price
        prices = np.column_stack([np.full(num_paths, self.spot_price), prices])

        # Compute payoffs at each time
        if self.option_type == "put":
            payoffs = np.maximum(self.strike - prices, 0.0)
        else:
            payoffs = np.maximum(prices - self.strike, 0.0)

        # Cash flows at exercise (backward induction)
        cf = payoffs[:, T].copy()  # at expiry, always exercise if ITM

        for t in range(T - 1, 0, -1):
            itm = payoffs[:, t] > 0  # in-the-money paths
            if not np.any(itm):
                cf *= df
                continue

            # Regression: continuation value = E[df * cf | S_t]
            # Use Laguerre polynomial basis: 1, S, S^2
            S_itm = prices[itm, t]
            cf_itm = cf[itm] * df

            # Design matrix with polynomial basis
            X = np.column_stack([
                np.ones(S_itm.shape[0]),
                S_itm / self.spot_price,
                (S_itm / self.spot_price) ** 2,
            ])

            # Least squares regression
            try:
                beta = np.linalg.lstsq(X, cf_itm, rcond=None)[0]
                continuation = X @ beta
            except np.linalg.LinAlgError:
                continuation = cf_itm

            # Exercise if intrinsic > continuation
            exercise = payoffs[itm, t] >= continuation
            # Update cash flows
            cf[itm] = np.where(exercise, payoffs[itm, t], cf[itm] * df)
            cf[~itm] *= df

        # Discount to time 0
        price = float(np.mean(cf * df))
        return price


# ============================================================================
# DeepHedger -- Neural network hedging (Ch 9 + Ch 14)
# ============================================================================


class DeepHedger:
    """Deep Hedging: neural network learns optimal hedging strategy.

    Buehler et al. (2019) showed that a recurrent neural network can learn
    hedging strategies that significantly outperform Black-Scholes delta
    hedging in the presence of:
    - Transaction costs
    - Discrete rebalancing
    - Model misspecification (stochastic vol, jumps)

    Architecture:
        Input:  (tau, moneyness, current_position, IV_proxy)
        Hidden: configurable layers with ReLU activations
        Output: hedge ratio in [-1, 1]

    Training uses REINFORCE with exponential utility reward:
        R = -exp(-alpha * (portfolio_PnL - payoff))

    Expected improvement over BS: 15-30% reduction in hedging PnL variance
    with transaction costs of 0.1% or more.

    References:
        Buehler et al. (2019), "Deep Hedging"
        Rao & Jelvis Ch 9, Section 9.4
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 1e-4,
        gamma: float = 1.0,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.001,
        device: str = "auto",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for DeepHedger")

        self.state_dim = state_dim
        self.gamma = gamma
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.device = _get_device(device)

        # --- Hedging network ---
        # Maps state features to hedge ratio in [-1, 1]
        layers: list[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())  # output in [-1, 1]
        self.hedge_net = nn.Sequential(*layers).to(self.device)

        self.optimizer = optim.Adam(self.hedge_net.parameters(), lr=learning_rate)
        self._trained = False

    def _generate_gbm_paths(
        self,
        spot: float,
        sigma: float,
        r: float,
        T: int,
        num_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate GBM price paths.

        S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

        Returns shape (num_paths, T+1) including initial price.
        """
        dt = 1.0 / 252
        z = rng.standard_normal((num_paths, T))
        log_rets = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
        log_prices = np.log(spot) + np.cumsum(log_rets, axis=1)
        prices = np.exp(log_prices)
        return np.column_stack([np.full(num_paths, spot), prices])

    def train(
        self,
        spot_paths: np.ndarray,
        strike: float,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        num_epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
        sigma_for_features: float = 0.20,
    ) -> dict:
        """Train the deep hedger on simulated or historical paths.

        The training objective is to minimize the negative expected utility
        of the hedging PnL:
            min_theta E[-U(PnL)]   where PnL = cash + position*S_T - payoff(S_T)

        Parameters
        ----------
        spot_paths : np.ndarray, shape (num_paths, T+1)
            Price paths. paths[:, 0] = initial price, paths[:, -1] = terminal.
        strike : float
            Strike price.
        payoff_fn : callable
            payoff(S_T) -> payoff values. Takes array, returns array.
        num_epochs : int
            Training epochs (passes through all paths).
        batch_size : int
        verbose : bool
        sigma_for_features : float
            Volatility used to compute BS delta as a feature.

        Returns
        -------
        dict with training history.
        """
        num_paths, T_plus_1 = spot_paths.shape
        T = T_plus_1 - 1
        dt = 1.0 / 252

        # Precompute terminal payoffs
        terminal_prices = spot_paths[:, -1]
        payoffs = payoff_fn(terminal_prices)

        # Initial BS premium (risk-neutral price)
        r = 0.05  # reasonable default
        tau_total = T * dt
        # Average initial premium for normalization
        bs_premium = BlackScholesHedger.price(
            float(spot_paths[0, 0]), strike, tau_total, sigma_for_features, r
        )

        history: dict[str, list] = {
            "epoch_loss": [],
            "mean_pnl": [],
            "std_pnl": [],
        }

        paths_t = torch.FloatTensor(spot_paths).to(self.device)
        payoffs_t = torch.FloatTensor(payoffs).to(self.device)

        for epoch in range(1, num_epochs + 1):
            # Shuffle paths
            perm = np.random.permutation(num_paths)
            epoch_losses: list[float] = []
            epoch_pnls: list[float] = []

            for start in range(0, num_paths, batch_size):
                idx = perm[start: start + batch_size]
                batch_paths = paths_t[idx]  # (B, T+1)
                batch_payoffs = payoffs_t[idx]  # (B,)
                B = batch_paths.shape[0]

                # Simulate hedging for this batch
                position = torch.zeros(B, device=self.device)
                cash = torch.full((B,), bs_premium, device=self.device)

                for t in range(T):
                    tau = (T - t) / T
                    moneyness = torch.log(batch_paths[:, t] / strike)
                    # Feature vector
                    features = torch.stack([
                        torch.full((B,), tau, device=self.device),
                        moneyness,
                        position,
                        cash / batch_paths[:, t].clamp(min=1e-6),
                    ], dim=-1)  # (B, 4)

                    # Get hedge ratio
                    hedge_ratio = self.hedge_net(features).squeeze(-1)  # (B,)

                    # Trade to target position
                    trade = hedge_ratio - position
                    trade_cost = torch.abs(trade * batch_paths[:, t]) * self.transaction_cost

                    # Update cash: sell old, buy new, pay cost
                    cash = cash - trade * batch_paths[:, t] - trade_cost
                    # Accrue interest
                    cash = cash * math.exp(r * dt)

                    position = hedge_ratio

                # Terminal PnL
                pnl = cash + position * batch_paths[:, -1] - batch_payoffs

                # Loss: negative expected utility
                # U(x) = -exp(-alpha * x)  =>  -U(x) = exp(-alpha * x)
                loss = torch.mean(torch.exp(-self.risk_aversion * pnl))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hedge_net.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_pnls.extend(pnl.detach().cpu().numpy().tolist())

            mean_loss = np.mean(epoch_losses)
            mean_pnl = np.mean(epoch_pnls)
            std_pnl = np.std(epoch_pnls, ddof=1) if len(epoch_pnls) > 1 else 0.0

            history["epoch_loss"].append(mean_loss)
            history["mean_pnl"].append(mean_pnl)
            history["std_pnl"].append(std_pnl)

            if verbose and epoch % max(1, num_epochs // 10) == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Loss: {mean_loss:.6f} | "
                    f"PnL: {mean_pnl:.4f} +/- {std_pnl:.4f}"
                )

        self._trained = True
        return history

    def hedge(self, state_features: np.ndarray) -> float:
        """Get optimal hedge ratio for current state.

        Parameters
        ----------
        state_features : np.ndarray, shape (state_dim,)
            [tau, moneyness, current_position, normalized_cash]

        Returns
        -------
        float
            Hedge ratio in [-1, 1].
        """
        state_t = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ratio = self.hedge_net(state_t).item()
        return ratio

    def evaluate(
        self,
        spot_paths: np.ndarray,
        strike: float,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        bs_sigma: Optional[float] = None,
        risk_free_rate: float = 0.05,
    ) -> dict:
        """Compare deep hedging vs Black-Scholes delta hedging.

        Parameters
        ----------
        spot_paths : np.ndarray, shape (num_paths, T+1)
        strike : float
        payoff_fn : callable
        bs_sigma : float or None
            If provided, runs BS delta hedging as benchmark.
        risk_free_rate : float

        Returns
        -------
        dict with keys:
            deep_pnl_mean, deep_pnl_std, deep_pnl_var,
            bs_pnl_mean, bs_pnl_std, bs_pnl_var (if bs_sigma given),
            improvement_pct (variance reduction vs BS)
        """
        num_paths, T_plus_1 = spot_paths.shape
        T = T_plus_1 - 1
        dt = 1.0 / 252
        r = risk_free_rate
        tc = self.transaction_cost

        payoffs = payoff_fn(spot_paths[:, -1])
        tau_total = T * dt

        # --- Deep hedging PnL ---
        bs_premium = BlackScholesHedger.price(
            float(spot_paths[0, 0]), strike, tau_total,
            bs_sigma if bs_sigma else 0.20, r
        )

        deep_pnls = np.zeros(num_paths)
        for i in range(num_paths):
            position = 0.0
            cash = bs_premium
            for t in range(T):
                tau = (T - t) / T
                moneyness = math.log(max(spot_paths[i, t], 1e-10) / strike)
                features = np.array([tau, moneyness, position, cash / max(spot_paths[i, t], 1e-10)])
                hedge_ratio = self.hedge(features)
                trade = hedge_ratio - position
                cash -= trade * spot_paths[i, t] + abs(trade * spot_paths[i, t]) * tc
                cash *= math.exp(r * dt)
                position = hedge_ratio
            deep_pnls[i] = cash + position * spot_paths[i, -1] - payoffs[i]

        result: dict[str, Any] = {
            "deep_pnl_mean": float(np.mean(deep_pnls)),
            "deep_pnl_std": float(np.std(deep_pnls, ddof=1)),
            "deep_pnl_var": float(np.var(deep_pnls, ddof=1)),
        }

        # --- BS benchmark ---
        if bs_sigma is not None:
            bs_pnls = np.zeros(num_paths)
            for i in range(num_paths):
                position = 0.0
                cash = bs_premium
                for t in range(T):
                    tau_rem = (T - t) * dt
                    # BS delta hedge
                    delta = BlackScholesHedger.delta(
                        spot_paths[i, t], strike, tau_rem, bs_sigma, r
                    )
                    trade = delta - position
                    cash -= trade * spot_paths[i, t] + abs(trade * spot_paths[i, t]) * tc
                    cash *= math.exp(r * dt)
                    position = delta
                bs_pnls[i] = cash + position * spot_paths[i, -1] - payoffs[i]

            result["bs_pnl_mean"] = float(np.mean(bs_pnls))
            result["bs_pnl_std"] = float(np.std(bs_pnls, ddof=1))
            result["bs_pnl_var"] = float(np.var(bs_pnls, ddof=1))

            # Improvement: variance reduction
            if result["bs_pnl_var"] > 1e-12:
                result["improvement_pct"] = (
                    (result["bs_pnl_var"] - result["deep_pnl_var"])
                    / result["bs_pnl_var"]
                    * 100.0
                )
            else:
                result["improvement_pct"] = 0.0

        return result


# ============================================================================
# MaxExpUtility -- Maximum Expected Utility pricing
# ============================================================================


class MaxExpUtility:
    """Chapter 9: Maximum Expected Utility pricing framework.

    In incomplete markets, there is no unique price. The MEU framework
    gives bid and ask prices as certainty equivalents of optimal hedging.

    For the SELLER of a derivative:
        The seller receives premium p and hedges optimally.
        p_seller = CE of optimal hedging PnL without the derivative premium
        = inf{p : E[U(optimal_hedging_PnL + p)] >= E[U(0)]}

    For the BUYER:
        p_buyer = sup{p : E[U(-p + payoff + optimal_hedge_PnL)] >= E[U(0)]}

    In complete markets: p_buyer = p_seller = BS price.
    In incomplete markets: p_buyer < p_seller (bid-ask spread).

    The spread (p_seller - p_buyer) measures the degree of market incompleteness.

    References:
        Rao & Jelvis Ch 9, Section 9.5
        Henderson (2002), "Valuation of Claims on Nontraded Assets"
    """

    @staticmethod
    def compute_price(
        deep_hedger: DeepHedger,
        spot_paths: np.ndarray,
        strike: float,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        risk_aversion: float = 1.0,
        is_seller: bool = True,
        risk_free_rate: float = 0.05,
    ) -> float:
        """Compute MEU price using the trained deep hedger.

        Algorithm:
        1. Run optimal hedging on all paths
        2. Compute terminal PnL for each path (WITHOUT the premium)
        3. Certainty equivalent = -(1/alpha) * log(E[exp(-alpha * PnL)])
        4. Price = -CE for seller, CE for buyer

        Parameters
        ----------
        deep_hedger : DeepHedger
        spot_paths : np.ndarray, shape (num_paths, T+1)
        strike : float
        payoff_fn : callable
        risk_aversion : float
        is_seller : bool
        risk_free_rate : float

        Returns
        -------
        float
            MEU price.
        """
        num_paths, T_plus_1 = spot_paths.shape
        T = T_plus_1 - 1
        dt = 1.0 / 252
        r = risk_free_rate
        tc = deep_hedger.transaction_cost
        payoffs = payoff_fn(spot_paths[:, -1])

        # Run hedging WITHOUT premium (cash starts at 0)
        pnls = np.zeros(num_paths)
        for i in range(num_paths):
            position = 0.0
            cash = 0.0
            for t in range(T):
                tau = (T - t) / T
                moneyness = math.log(max(spot_paths[i, t], 1e-10) / strike)
                features = np.array([tau, moneyness, position, cash / max(spot_paths[i, t], 1e-10)])
                hedge_ratio = deep_hedger.hedge(features)
                trade = hedge_ratio - position
                cash -= trade * spot_paths[i, t] + abs(trade * spot_paths[i, t]) * tc
                cash *= math.exp(r * dt)
                position = hedge_ratio
            # PnL from hedging (without premium)
            pnls[i] = cash + position * spot_paths[i, -1] - payoffs[i]

        # Certainty equivalent under exponential utility:
        # CE = -(1/alpha) * log(E[exp(-alpha * PnL)])
        alpha = risk_aversion

        # Numerical stability: shift by mean
        pnl_mean = np.mean(pnls)
        shifted = -alpha * (pnls - pnl_mean)
        log_mean_exp = np.log(np.mean(np.exp(shifted))) + (-alpha * pnl_mean)
        ce = -log_mean_exp / alpha

        if is_seller:
            # Seller's price: they need compensation for the hedging cost
            return -ce
        else:
            # Buyer's price
            return ce

    @staticmethod
    def bid_ask_spread(
        deep_hedger: DeepHedger,
        spot_paths: np.ndarray,
        strike: float,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.05,
    ) -> Tuple[float, float, float]:
        """Compute bid-ask spread from MEU pricing.

        Returns
        -------
        (bid, ask, spread) where bid <= BS_price <= ask
        """
        ask = MaxExpUtility.compute_price(
            deep_hedger, spot_paths, strike, payoff_fn,
            risk_aversion, is_seller=True, risk_free_rate=risk_free_rate,
        )
        bid = MaxExpUtility.compute_price(
            deep_hedger, spot_paths, strike, payoff_fn,
            risk_aversion, is_seller=False, risk_free_rate=risk_free_rate,
        )
        return bid, ask, ask - bid
