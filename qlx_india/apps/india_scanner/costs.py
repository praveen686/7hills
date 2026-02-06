"""India equity transaction cost model.

Covers all NSE equity delivery charges:
  - STT (Securities Transaction Tax): 0.1% on buy + sell (delivery)
  - Brokerage: flat Rs 20 per order (discount broker)
  - Exchange charges: 0.00345% (NSE)
  - SEBI turnover fee: 0.0001%
  - Stamp duty: 0.015% (buy side only)
  - GST: 18% on (brokerage + exchange charges + SEBI fee)

For F&O (futures): STT is 0.02% on sell side only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndiaCostModel:
    """All-in transaction cost model for Indian equities.

    All rates are in decimal (0.001 = 0.1%).
    Brokerage is per-order in INR (flat fee model).
    """

    # STT: 0.1% on both buy and sell for delivery
    stt_delivery_rate: float = 0.001
    # STT: 0.02% on sell side only for futures
    stt_futures_sell_rate: float = 0.0002
    # Brokerage: Rs 20 per executed order (discount broker)
    brokerage_per_order_inr: float = 20.0
    # Exchange transaction charges (NSE)
    exchange_charge_rate: float = 0.0000345
    # SEBI turnover fee
    sebi_fee_rate: float = 0.000001
    # Stamp duty (buy side only)
    stamp_duty_rate: float = 0.00015
    # GST on brokerage + exchange + SEBI
    gst_rate: float = 0.18
    # Estimated slippage for liquid F&O stocks
    slippage_bps: float = 5.0

    def delivery_cost_frac(self, trade_value_inr: float) -> float:
        """Total one-way cost as fraction of trade value (delivery trade).

        Includes STT on both sides (averaged per side), brokerage,
        exchange charges, stamp duty (buy side averaged), SEBI fee, GST.
        """
        # STT: full rate on both buy and sell → per-side = stt_rate
        stt = self.stt_delivery_rate

        # Brokerage as fraction of trade value
        brokerage_frac = self.brokerage_per_order_inr / trade_value_inr if trade_value_inr > 0 else 0

        # Exchange + SEBI
        exchange = self.exchange_charge_rate
        sebi = self.sebi_fee_rate

        # Stamp duty: buy side only → averaged per side = half
        stamp = self.stamp_duty_rate / 2

        # GST on (brokerage + exchange + sebi)
        gst = self.gst_rate * (brokerage_frac + exchange + sebi)

        # Slippage
        slippage = self.slippage_bps / 10_000

        return stt + brokerage_frac + exchange + sebi + stamp + gst + slippage

    def roundtrip_cost_frac(self, trade_value_inr: float) -> float:
        """Round-trip cost as fraction of trade value."""
        return 2 * self.delivery_cost_frac(trade_value_inr)

    def roundtrip_bps(self, trade_value_inr: float = 100_000) -> float:
        """Round-trip cost in basis points for a given trade size."""
        return self.roundtrip_cost_frac(trade_value_inr) * 10_000


# Default cost model for typical Rs 1L trade
DEFAULT_COSTS = IndiaCostModel()
