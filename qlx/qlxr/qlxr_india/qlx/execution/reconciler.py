"""Position reconciler — compares broker vs local state.

Detects mismatches between what the orchestrator thinks positions are
and what the broker reports.  Critical for catching missed fills,
phantom positions, and state corruption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qlx.execution.adaptor import BrokerAdaptor

logger = logging.getLogger(__name__)


@dataclass
class Mismatch:
    """Position mismatch between local and broker state."""

    symbol: str
    exchange: str
    local_qty: int
    broker_qty: int
    diff: int

    @property
    def description(self) -> str:
        return (
            f"{self.symbol} ({self.exchange}): "
            f"local={self.local_qty} broker={self.broker_qty} diff={self.diff}"
        )


class Reconciler:
    """Compares broker positions against local state."""

    def __init__(self, adaptor: BrokerAdaptor):
        self.adaptor = adaptor

    def reconcile(
        self,
        local_positions: dict[str, int],
    ) -> list[Mismatch]:
        """Compare local positions against broker.

        Parameters
        ----------
        local_positions : dict
            key = "exchange:symbol", value = quantity (signed)

        Returns
        -------
        list[Mismatch]
            Any positions that don't match.
        """
        broker_pos = self.adaptor.positions()
        broker_map: dict[str, int] = {}
        for p in broker_pos:
            key = f"{p.exchange}:{p.symbol}"
            broker_map[key] = p.quantity

        all_keys = set(local_positions.keys()) | set(broker_map.keys())
        mismatches: list[Mismatch] = []

        for key in sorted(all_keys):
            local_qty = local_positions.get(key, 0)
            broker_qty = broker_map.get(key, 0)

            if local_qty != broker_qty:
                parts = key.split(":", 1)
                exchange = parts[0] if len(parts) > 1 else ""
                symbol = parts[1] if len(parts) > 1 else key

                m = Mismatch(
                    symbol=symbol,
                    exchange=exchange,
                    local_qty=local_qty,
                    broker_qty=broker_qty,
                    diff=broker_qty - local_qty,
                )
                mismatches.append(m)
                logger.warning("Position mismatch: %s", m.description)

        if not mismatches:
            logger.info("Reconciliation OK: all positions match")

        return mismatches
