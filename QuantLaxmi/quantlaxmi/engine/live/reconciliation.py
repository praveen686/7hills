"""Broker Position Reconciliation at Startup.

Compares internal state (JSON-persisted positions) with broker positions
(fetched from Kite Connect API) to detect discrepancies before trading begins.

Discrepancy types:
  - **Matched**: Internal and broker agree on symbol, quantity, and price.
  - **Mismatched**: Both have the symbol but qty or price differs.
  - **Missing internal**: Broker has a position not in internal state.
  - **Missing broker**: Internal state has a position not at the broker.

Usage
-----
    reconciler = PositionReconciler(kite_client=kite)
    result = reconciler.reconcile(internal_positions)
    if not result.is_clean:
        logger.warning("Reconciliation found discrepancies: %s", result.summary())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Relative tolerance for floating-point price comparison (0.01% = 1 bps).
_PRICE_REL_TOL = 1e-4


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationResult:
    """Outcome of comparing internal vs broker positions.

    Attributes
    ----------
    matched : list[str]
        Symbols where internal and broker agree (within tolerance).
    mismatched : list[dict]
        Positions present on both sides but with differing qty or price.
        Each dict: ``{"symbol", "internal_qty", "broker_qty",
        "internal_price", "broker_price", "qty_diff", "price_diff_pct"}``.
    missing_internal : list[dict]
        Positions found at the broker but absent from internal state.
        Each dict: ``{"symbol", "qty", "avg_price"}``.
    missing_broker : list[dict]
        Positions in internal state but absent from the broker.
        Each dict: ``{"symbol", "qty", "avg_price"}``.
    is_clean : bool
        ``True`` if there are no mismatches or missing positions.
    timestamp : str
        ISO-8601 timestamp of the reconciliation.
    """

    matched: list[str] = field(default_factory=list)
    mismatched: list[dict] = field(default_factory=list)
    missing_internal: list[dict] = field(default_factory=list)
    missing_broker: list[dict] = field(default_factory=list)
    is_clean: bool = True
    timestamp: str = ""

    def summary(self) -> str:
        """Human-readable summary of the reconciliation."""
        parts = [
            f"matched={len(self.matched)}",
            f"mismatched={len(self.mismatched)}",
            f"missing_internal={len(self.missing_internal)}",
            f"missing_broker={len(self.missing_broker)}",
            f"is_clean={self.is_clean}",
        ]
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "matched": self.matched,
            "mismatched": self.mismatched,
            "missing_internal": self.missing_internal,
            "missing_broker": self.missing_broker,
            "is_clean": self.is_clean,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class PositionReconciler:
    """Compares internal state positions against broker (Kite) positions.

    Parameters
    ----------
    kite_client : KiteConnect | None
        Authenticated Kite Connect client.  ``None`` for paper/test mode
        (reconcile() returns all-clean when no broker data is available).
    price_rel_tol : float
        Relative tolerance for price comparison (default 0.01%).
    """

    def __init__(
        self,
        kite_client: Any = None,
        price_rel_tol: float = _PRICE_REL_TOL,
    ) -> None:
        self.kite = kite_client
        self.price_rel_tol = price_rel_tol

    def reconcile(
        self,
        internal_positions: dict[str, dict],
        broker_positions: dict[str, dict] | None = None,
    ) -> ReconciliationResult:
        """Compare internal state vs broker positions.

        Parameters
        ----------
        internal_positions : dict[str, dict]
            Internal positions keyed by symbol.
            Each value must have at least ``"qty"`` (int) and ``"avg_price"``
            (float) keys.
        broker_positions : dict[str, dict] | None
            Broker positions in the same format.  If ``None``, positions are
            fetched from Kite.  If Kite is also ``None`` (paper mode), all
            internal positions are treated as matched.

        Returns
        -------
        ReconciliationResult
        """
        now = datetime.now(timezone.utc).isoformat()

        # Resolve broker positions
        if broker_positions is None and self.kite is not None:
            broker_positions = self._fetch_broker_positions()
        elif broker_positions is None:
            # Paper mode -- no broker to reconcile against
            logger.info("Paper mode: skipping broker reconciliation")
            return ReconciliationResult(
                matched=list(internal_positions.keys()),
                mismatched=[],
                missing_internal=[],
                missing_broker=[],
                is_clean=True,
                timestamp=now,
            )

        # Normalise keys to uppercase for comparison
        internal = {k.upper(): v for k, v in internal_positions.items()}
        broker = {k.upper(): v for k, v in broker_positions.items()}

        all_symbols = set(internal.keys()) | set(broker.keys())

        matched: list[str] = []
        mismatched: list[dict] = []
        missing_internal: list[dict] = []
        missing_broker: list[dict] = []

        for sym in sorted(all_symbols):
            in_internal = sym in internal
            in_broker = sym in broker

            if in_internal and in_broker:
                int_pos = internal[sym]
                brk_pos = broker[sym]

                int_qty = int(int_pos.get("qty", 0))
                brk_qty = int(brk_pos.get("qty", 0))

                int_price = float(int_pos.get("avg_price", 0.0))
                brk_price = float(brk_pos.get("avg_price", 0.0))

                qty_match = int_qty == brk_qty
                price_match = self._prices_match(int_price, brk_price)

                if qty_match and price_match:
                    matched.append(sym)
                else:
                    price_diff_pct = 0.0
                    if brk_price != 0.0:
                        price_diff_pct = (int_price - brk_price) / brk_price * 100
                    mismatched.append({
                        "symbol": sym,
                        "internal_qty": int_qty,
                        "broker_qty": brk_qty,
                        "internal_price": int_price,
                        "broker_price": brk_price,
                        "qty_diff": int_qty - brk_qty,
                        "price_diff_pct": round(price_diff_pct, 4),
                    })

            elif in_broker and not in_internal:
                brk_pos = broker[sym]
                missing_internal.append({
                    "symbol": sym,
                    "qty": int(brk_pos.get("qty", 0)),
                    "avg_price": float(brk_pos.get("avg_price", 0.0)),
                })

            else:  # in_internal and not in_broker
                int_pos = internal[sym]
                missing_broker.append({
                    "symbol": sym,
                    "qty": int(int_pos.get("qty", 0)),
                    "avg_price": float(int_pos.get("avg_price", 0.0)),
                })

        is_clean = len(mismatched) == 0 and len(missing_internal) == 0 and len(missing_broker) == 0

        result = ReconciliationResult(
            matched=matched,
            mismatched=mismatched,
            missing_internal=missing_internal,
            missing_broker=missing_broker,
            is_clean=is_clean,
            timestamp=now,
        )

        if is_clean:
            logger.info(
                "Position reconciliation CLEAN: %d positions matched", len(matched),
            )
        else:
            logger.warning(
                "Position reconciliation DIRTY: %s", result.summary(),
            )

        return result

    def _fetch_broker_positions(self) -> dict[str, dict]:
        """Fetch net positions from Kite Connect API.

        Returns
        -------
        dict[str, dict]
            Positions keyed by tradingsymbol with ``qty`` and ``avg_price``.
        """
        try:
            positions_data = self.kite.positions()
            net = positions_data.get("net", [])
            result: dict[str, dict] = {}
            for p in net:
                sym = p.get("tradingsymbol", "")
                qty = int(p.get("quantity", 0))
                # Skip zero-quantity positions (fully closed)
                if qty == 0:
                    continue
                avg_price = float(p.get("average_price", 0.0))
                result[sym] = {"qty": qty, "avg_price": avg_price}
            return result
        except Exception as e:
            logger.error("Failed to fetch broker positions: %s", e)
            raise

    def _prices_match(self, a: float, b: float) -> bool:
        """Compare two prices within relative tolerance.

        Uses ``math.isclose`` with the configured relative tolerance
        and an absolute tolerance of 0.01 (for near-zero prices).
        """
        return math.isclose(a, b, rel_tol=self.price_rel_tol, abs_tol=0.01)
