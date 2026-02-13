"""Instrument token resolution for live trading.

Maps well-known trading symbols to Kite instrument tokens.
Resolves current-month FUT contracts from the instrument master.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Well-known NSE index spot tokens (hardcoded, stable across sessions)
INDEX_TOKENS: dict[str, int] = {
    "NIFTY 50": 256265,
    "NIFTY BANK": 260105,
    "NIFTY FIN SERVICE": 257801,
    "NIFTY MID SELECT": 288009,
}

# Reverse map: token → symbol
TOKEN_SYMBOLS: dict[int, str] = {
    256265: "NIFTY",
    260105: "BANKNIFTY",
    257801: "FINNIFTY",
    288009: "MIDCPNIFTY",
}


def resolve_index_tokens() -> dict[str, int]:
    """Return static index spot tokens (no API call needed)."""
    return {
        "NIFTY": 256265,
        "BANKNIFTY": 260105,
        "FINNIFTY": 257801,
        "MIDCPNIFTY": 288009,
    }


def resolve_fut_tokens(kite: Any) -> dict[str, int]:
    """Fetch NFO instrument master and resolve current-month FUT tokens.

    Parameters
    ----------
    kite : KiteConnect
        Authenticated Kite client.

    Returns
    -------
    dict
        {symbol: token} for current-month futures.
        e.g. {"NIFTY_FUT": 12345, "BANKNIFTY_FUT": 67890}
    """
    today = datetime.now(IST).date()
    result: dict[str, int] = {}

    try:
        instruments = kite.instruments("NFO")
    except Exception as exc:
        logger.error("Failed to fetch NFO instruments: %s", exc)
        return result

    # Group FUT instruments by name
    fut_by_name: dict[str, list[dict]] = {}
    for inst in instruments:
        if inst.get("instrument_type") != "FUT":
            continue
        name = inst.get("name", "")
        if name in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            fut_by_name.setdefault(name, []).append(inst)

    # For each name, find the nearest expiry >= today
    for name, futs in fut_by_name.items():
        candidates = []
        for f in futs:
            exp = f.get("expiry")
            if exp is None:
                continue
            if isinstance(exp, str):
                exp = date.fromisoformat(exp)
            if isinstance(exp, datetime):
                exp = exp.date()
            if exp >= today:
                candidates.append((exp, f))

        if not candidates:
            continue

        # Pick nearest expiry
        candidates.sort(key=lambda x: x[0])
        nearest = candidates[0][1]
        token = nearest.get("instrument_token", 0)
        tradingsymbol = nearest.get("tradingsymbol", f"{name}FUT")
        result[f"{name}_FUT"] = token
        logger.info(
            "Resolved %s FUT → %s (token=%d, expiry=%s)",
            name, tradingsymbol, token, candidates[0][0],
        )

    return result


def resolve_all_tokens(kite: Any) -> dict[str, int]:
    """Resolve all tokens needed for live trading.

    Returns combined dict of index spot + current-month FUT tokens.
    """
    tokens = resolve_index_tokens()
    try:
        fut_tokens = resolve_fut_tokens(kite)
        tokens.update(fut_tokens)
    except Exception as exc:
        logger.warning("FUT token resolution failed: %s", exc)

    logger.info("Resolved %d instrument tokens: %s", len(tokens), list(tokens.keys()))
    return tokens


def token_to_symbol(token: int, token_map: dict[str, int] | None = None) -> str:
    """Reverse lookup: token → symbol name."""
    if token in TOKEN_SYMBOLS:
        return TOKEN_SYMBOLS[token]
    if token_map:
        for sym, tok in token_map.items():
            if tok == token:
                return sym
    return f"UNKNOWN_{token}"
