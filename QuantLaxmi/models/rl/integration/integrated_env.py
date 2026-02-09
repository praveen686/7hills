"""Integrated Multi-Asset Trading Environment.

Wraps the X-Trend backbone + IndiaFnOEnv instances into a single
Gymnasium-like environment that:

1. Pre-computes backbone hidden states for the entire fold (frozen, no grad).
2. Steps each per-asset IndiaFnOEnv with position targets.
3. Computes portfolio-level state: concatenated hidden states, positions,
   PnL, drawdown, heat, time features.
4. Returns reward = risk-adjusted PnL − cost penalty − drawdown penalty.

State vector layout (dim = 4×d_hidden + 10):
    [backbone_hidden(asset_0, d_h), ..., backbone_hidden(asset_3, d_h),
     positions(4), norm_pnl(1), drawdown(1), heat(1), time_feats(3)]

Action: (n_assets,) continuous position targets ∈ [-1, 1].

References:
    - Ch 3 (MDP formulation), Ch 10 (Financial MDPs)
    - IndiaFnOEnv: realistic India FnO costs (3 pts NIFTY, 5 pts BANKNIFTY)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from models.rl.environments.india_fno_env import (
    IndiaFnOEnv,
    COST_PER_LEG,
    LOT_SIZES,
    INITIAL_SPOTS,
    ANNUALISED_VOLS,
)
from models.rl.environments.trading_env import TradingAction


class IntegratedTradingEnv:
    """Multi-asset walk-forward environment with backbone hidden states.

    Parameters
    ----------
    backbone : XTrendBackbone
        Frozen backbone for hidden state extraction.
    features : (n_days, n_assets, n_features)
        Normalized feature tensor.
    targets : (n_days, n_assets)
        Vol-scaled next-day returns (for reward computation).
    dates : DatetimeIndex
        Trading dates.
    symbols : list[str]
        Asset symbols (e.g. ["NIFTY", "BANKNIFTY", ...]).
    reward_lambda_risk : float
        Weight for drawdown penalty in reward.
    reward_lambda_cost : float
        Weight for cost penalty in reward.
    """

    def __init__(
        self,
        backbone,
        features: np.ndarray,
        targets: np.ndarray,
        dates,
        symbols: list[str],
        reward_lambda_risk: float = 0.5,
        reward_lambda_cost: float = 1.0,
    ) -> None:
        self.backbone = backbone
        self.features = features
        self.targets = targets
        self.dates = dates
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.d_hidden = backbone.d_hidden if hasattr(backbone, 'd_hidden') else 64
        self.reward_lambda_risk = reward_lambda_risk
        self.reward_lambda_cost = reward_lambda_cost

        # Per-asset IndiaFnOEnv instances (simulated mode, inject historical prices)
        self._envs: list[IndiaFnOEnv] = []
        for sym in symbols:
            env = IndiaFnOEnv(
                instrument=sym.upper(),
                mode="simulated",
                discrete_actions=False,
                num_steps=1000,  # large enough for any fold
            )
            self._envs.append(env)

        # Portfolio tracking state
        self._positions = np.zeros(self.n_assets)
        self._cum_pnl = 0.0
        self._peak_pnl = 0.0
        self._drawdown = 0.0
        self._step_idx = 0
        self._fold_start = 0
        self._fold_end = 0
        self._hidden_states: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(42)

    @property
    def state_dim(self) -> int:
        """Total state vector dimension: 4×d_hidden + 10."""
        return self.n_assets * self.d_hidden + 10

    @property
    def action_dim(self) -> int:
        """One continuous position target per asset."""
        return self.n_assets

    def reset(
        self, fold_start_idx: int, fold_end_idx: int
    ) -> np.ndarray:
        """Reset environment for a new fold.

        Pre-computes backbone hidden states for the entire fold
        (frozen, no gradient), then returns the initial state vector.

        Parameters
        ----------
        fold_start_idx : int
            Start index in the features array.
        fold_end_idx : int
            End index (exclusive).

        Returns
        -------
        state_vec : (state_dim,)
        """
        self._fold_start = fold_start_idx
        self._fold_end = fold_end_idx
        self._step_idx = 0
        self._positions = np.zeros(self.n_assets)
        self._cum_pnl = 0.0
        self._peak_pnl = 0.0
        self._drawdown = 0.0

        # Pre-compute backbone hidden states for the fold
        self._hidden_states = self.backbone.precompute_hidden_states(
            self.features, fold_start_idx, fold_end_idx, self._rng
        )
        # shape: (fold_len, n_assets, d_hidden)

        # Reset per-asset envs
        for env in self._envs:
            env.reset()

        return self._build_state_vector()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Take one step in the environment.

        Parameters
        ----------
        actions : (n_assets,) — position targets ∈ [-1, 1]

        Returns
        -------
        state_vec : (state_dim,)
        reward : float
        done : bool — True when fold end reached
        info : dict with per-asset details
        """
        actions = np.clip(actions, -1.0, 1.0)
        day_idx = self._fold_start + self._step_idx + 1
        fold_len = self._fold_end - self._fold_start

        if day_idx >= self._fold_end or self._step_idx + 1 >= fold_len:
            return self._build_state_vector(), 0.0, True, {"reason": "fold_end"}

        # Per-asset PnL and costs
        total_pnl_delta = 0.0
        total_cost = 0.0
        per_asset_info = {}

        for i, sym in enumerate(self.symbols):
            # Position change
            old_pos = self._positions[i]
            new_pos = actions[i]
            trade = new_pos - old_pos

            # Cost in index points per leg
            sym_upper = sym.upper()
            cost_per_leg = COST_PER_LEG.get(sym_upper, 3.0)
            spot = INITIAL_SPOTS.get(sym_upper, 20000.0)
            cost_fraction = abs(trade) * cost_per_leg / spot  # normalize to fraction

            # PnL: position × next-day return (from targets)
            if day_idx < len(self.targets):
                ret_i = self.targets[day_idx, i]
                if np.isnan(ret_i):
                    ret_i = 0.0
            else:
                ret_i = 0.0

            pnl_i = old_pos * ret_i  # position at t earns return at t+1
            total_pnl_delta += pnl_i
            total_cost += cost_fraction

            self._positions[i] = new_pos

            per_asset_info[sym] = {
                "position": new_pos,
                "return": ret_i,
                "pnl": pnl_i,
                "cost": cost_fraction,
            }

        # Update portfolio state
        net_pnl = total_pnl_delta - total_cost
        self._cum_pnl += net_pnl
        self._peak_pnl = max(self._peak_pnl, self._cum_pnl)

        if self._peak_pnl > 0:
            self._drawdown = (self._peak_pnl - self._cum_pnl) / self._peak_pnl
        else:
            self._drawdown = 0.0

        # Reward: risk-adjusted PnL
        dd_threshold = 0.05  # 5% drawdown threshold
        dd_penalty = self.reward_lambda_risk * max(0.0, self._drawdown - dd_threshold)
        reward = net_pnl - dd_penalty

        self._step_idx += 1
        done = (self._step_idx + 1 >= fold_len) or (day_idx + 1 >= self._fold_end)

        state = self._build_state_vector()
        info = {
            "per_asset": per_asset_info,
            "cum_pnl": self._cum_pnl,
            "drawdown": self._drawdown,
            "total_cost": total_cost,
        }
        return state, reward, done, info

    def _build_state_vector(self) -> np.ndarray:
        """Assemble the full state vector.

        Layout: [hidden(asset_0), ..., hidden(asset_n), positions(n),
                 norm_pnl(1), drawdown(1), heat(1), time_feats(3)]
        """
        parts = []

        # Backbone hidden states for current step
        if self._hidden_states is not None and self._step_idx < len(self._hidden_states):
            for a in range(self.n_assets):
                parts.append(self._hidden_states[self._step_idx, a, :])
        else:
            for a in range(self.n_assets):
                parts.append(np.zeros(self.d_hidden, dtype=np.float32))

        # Portfolio state features
        parts.append(self._positions.copy())

        # Normalized PnL (clip to [-5, 5] for stability)
        norm_pnl = np.clip(self._cum_pnl * 100.0, -5.0, 5.0)  # scale up small numbers
        parts.append(np.array([norm_pnl], dtype=np.float32))

        # Drawdown
        parts.append(np.array([self._drawdown], dtype=np.float32))

        # Portfolio heat (total absolute exposure)
        heat = float(np.sum(np.abs(self._positions)))
        parts.append(np.array([heat], dtype=np.float32))

        # Time features: fold progress, day-of-week proxy, month proxy
        fold_len = max(1, self._fold_end - self._fold_start)
        progress = self._step_idx / fold_len
        day_sin = math.sin(2.0 * math.pi * self._step_idx / 5.0)  # weekly cycle
        day_cos = math.cos(2.0 * math.pi * self._step_idx / 5.0)
        parts.append(np.array([progress, day_sin, day_cos], dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    def get_cost_per_leg(self, symbol: str) -> float:
        """Return cost per leg in index points for a given symbol."""
        return COST_PER_LEG.get(symbol.upper(), 3.0)
