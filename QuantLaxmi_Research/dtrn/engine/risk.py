"""Risk management for DTRN — hard limits and regime-based controls."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from ..config import DTRNConfig


@dataclass
class RiskState:
    """Mutable risk tracking state."""
    position: int = 0                 # current position (contracts, signed)
    entry_price: float = 0.0          # average entry price
    day_pnl: float = 0.0             # accumulated PnL today
    peak_equity: float = 0.0          # peak equity for drawdown
    equity: float = 0.0               # current equity
    trades_today: int = 0
    daily_returns: list = field(default_factory=list)

    def reset_day(self):
        self.day_pnl = 0.0
        self.trades_today = 0


class RiskManager:
    """Enforce risk limits on position targets.

    Hard limits (non-negotiable):
    1. Max contracts
    2. Max daily loss (kill switch)
    3. Max drawdown (rolling)
    4. Max position change per step (prevents flip-flop)

    Regime-based:
    5. Reduce size in high_vol / liq_stress regimes
    """

    def __init__(self, config: DTRNConfig = None):
        if config is None:
            config = DTRNConfig()
        self.config = config
        self.state = RiskState(equity=config.initial_capital, peak_equity=config.initial_capital)

    def check_and_clip(
        self,
        target_position: float,      # desired position in [-1, 1] fraction
        current_price: float,         # current instrument price
        regime_probs: np.ndarray,     # (K,) regime probabilities
        instrument: str = "NIFTY",
    ) -> int:
        """Apply risk limits, return allowed position in contracts (signed).

        target_position: model output in [-1, 1]
        current_price: last traded price
        regime_probs: regime posterior from DTRN
        instrument: for lot size lookup
        """
        lot_size = self.config.lot_sizes.get(instrument, 75)
        max_lots = self.config.max_lots

        # 1. Kill switch — max daily loss
        if self.state.equity > 0:
            daily_loss_pct = -self.state.day_pnl / self.state.equity
            if daily_loss_pct >= self.config.max_daily_loss:
                return 0  # flat — stop trading today

        # 2. Max drawdown
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - self.state.equity) / self.state.peak_equity
            if drawdown >= self.config.max_drawdown:
                # Reduce max position by 50%
                max_lots = max(max_lots // 2, 1)

        # 3. Regime-based scaling
        # regime_probs: [calm_mr, trend, high_vol, liq_stress]
        if len(regime_probs) >= 4:
            high_vol_prob = regime_probs[2]
            stress_prob = regime_probs[3]

            # Reduce in high vol
            if high_vol_prob > 0.5:
                scale = max(0.3, 1.0 - high_vol_prob)
                target_position *= scale

            # Aggressive reduction in liquidity stress
            if stress_prob > 0.4:
                target_position *= 0.25

        # 4. Convert to lots then contracts
        target_lots = int(round(target_position * max_lots))
        target_contracts = target_lots * lot_size

        # 5. Clip to max contracts
        max_contracts = max_lots * lot_size
        target_contracts = np.clip(target_contracts, -max_contracts, max_contracts)

        # 6. Max position change per step (in contracts)
        max_change_lots = max(1, int(self.config.max_position_change_per_step * max_lots))
        max_change = max_change_lots * lot_size
        delta = target_contracts - self.state.position
        if abs(delta) > max_change:
            delta = int(np.sign(delta) * max_change)
            target_contracts = self.state.position + delta

        # Round to lot size
        target_contracts = int(round(target_contracts / lot_size)) * lot_size

        return target_contracts

    def update_position(self, new_position: int, fill_price: float, instrument: str = "NIFTY"):
        """Update risk state after a fill."""
        delta = new_position - self.state.position

        if delta != 0:
            self.state.trades_today += 1

        # PnL from position change
        if self.state.position != 0:
            # Mark-to-market PnL on existing position
            # (handled in backtest, not here — this just tracks position)
            pass

        self.state.position = new_position
        if abs(new_position) > 0:
            self.state.entry_price = fill_price

    def mark_to_market(self, current_price: float):
        """Update equity with mark-to-market PnL."""
        if self.state.position != 0 and self.state.entry_price > 0:
            pnl = self.state.position * (current_price - self.state.entry_price)
            self.state.equity = self.state.peak_equity + pnl + self.state.day_pnl
            if self.state.equity > self.state.peak_equity:
                self.state.peak_equity = self.state.equity

    def reset_day(self):
        """Reset daily counters."""
        self.state.reset_day()
