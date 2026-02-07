"""Transaction cost model.

Every backtest in QLX requires an explicit cost model.  There is no default
of zero.  This forces the researcher to think about realistic execution costs
before looking at any PnL numbers.

The cost model is a frozen dataclass — it cannot be accidentally mutated
mid-backtest.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """All-in transaction cost model.

    Parameters
    ----------
    commission_bps : float
        Exchange commission per side in basis points.
        Example: Binance taker = 4 bps, maker = 2 bps.
    slippage_bps : float
        Estimated market impact / slippage per side in basis points.
        For BTC at reasonable size, 1-5 bps is typical.
    funding_annual_pct : float
        Annualised funding cost for holding perpetual positions.
        Used to debit the equity curve for positions held overnight.
        Default 0 (spot).
    """

    commission_bps: float
    slippage_bps: float
    funding_annual_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.commission_bps < 0:
            raise ValueError(f"commission_bps must be >= 0, got {self.commission_bps}")
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")

    @property
    def one_way_frac(self) -> float:
        """Total one-way cost as a decimal fraction."""
        return (self.commission_bps + self.slippage_bps) / 10_000

    @property
    def roundtrip_frac(self) -> float:
        """Total round-trip cost as a decimal fraction."""
        return 2 * self.one_way_frac

    @property
    def roundtrip_bps(self) -> float:
        """Total round-trip cost in basis points."""
        return 2 * (self.commission_bps + self.slippage_bps)

    def holding_cost_per_bar(self, bars_per_year: float) -> float:
        """Funding cost per bar as a decimal fraction.

        Parameters
        ----------
        bars_per_year : float
            Number of bars in a year (e.g., 365*24*60 for minute bars).
        """
        if bars_per_year <= 0:
            return 0.0
        return self.funding_annual_pct / 100.0 / bars_per_year
