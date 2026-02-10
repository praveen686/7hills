"""NFO futures and options token resolution with dynamic recentering.

Resolves instrument tokens for:
- Nearest-expiry futures (NIFTY, BANKNIFTY)
- Near-ATM options (±N strikes, nearest K weekly expiries)

Provides rollover detection and ATM-recenter logic so the collector
can dynamically subscribe/unsubscribe as spot price moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Strike widths for recenter threshold computation
STRIKE_WIDTHS = {"NIFTY": 50, "BANKNIFTY": 100}

# Recenter when spot moves ≥ 3 strike widths from last center
RECENTER_STRIKE_MULTIPLES = 3

# Spot quote symbols (NSE index)
SPOT_SYMBOLS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
}


# ---------------------------------------------------------------------------
# Token dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FuturesToken:
    """Resolved NFO futures instrument."""

    index_name: str              # "NIFTY" / "BANKNIFTY"
    instrument_token: int
    tradingsymbol: str           # e.g. "NIFTY26FEBFUT"
    expiry: date
    storage_key: str             # "NIFTY_FUT"


@dataclass
class OptionToken:
    """Resolved NFO option instrument."""

    index_name: str              # "NIFTY" / "BANKNIFTY"
    instrument_token: int
    tradingsymbol: str           # e.g. "NIFTY2620623500CE"
    expiry: date
    strike: float
    option_type: str             # "CE" / "PE"
    storage_key: str             # "NIFTY_OPT"


# ---------------------------------------------------------------------------
# Futures resolution
# ---------------------------------------------------------------------------

def resolve_futures_tokens(
    kite: KiteConnect,
    indices: list[str] | None = None,
) -> dict[int, FuturesToken]:
    """Resolve nearest-expiry futures tokens for the given indices.

    Returns mapping of instrument_token → FuturesToken.
    """
    if indices is None:
        indices = ["NIFTY", "BANKNIFTY"]

    instruments = pd.DataFrame(kite.instruments("NFO"))
    today = datetime.now(IST).date()

    tokens: dict[int, FuturesToken] = {}

    for name in indices:
        futs = instruments[
            (instruments["name"] == name)
            & (instruments["instrument_type"] == "FUT")
        ]
        futs = futs[futs["expiry"] >= today].sort_values("expiry")

        if futs.empty:
            logger.warning("No futures found for %s, skipping", name)
            continue

        row = futs.iloc[0]
        ft = FuturesToken(
            index_name=name,
            instrument_token=int(row["instrument_token"]),
            tradingsymbol=row["tradingsymbol"],
            expiry=row["expiry"] if isinstance(row["expiry"], date) else row["expiry"].date(),
            storage_key=f"{name}_FUT",
        )
        tokens[ft.instrument_token] = ft
        logger.info(
            "%s FUT: token=%d, symbol=%s, expiry=%s",
            name, ft.instrument_token, ft.tradingsymbol, ft.expiry,
        )

    return tokens


def should_reroll(tokens: dict[int, FuturesToken], today: date | None = None) -> bool:
    """Check if any token's expiry has passed and needs re-resolution."""
    if today is None:
        today = datetime.now(IST).date()
    return any(ft.expiry < today for ft in tokens.values())


# ---------------------------------------------------------------------------
# Options resolution
# ---------------------------------------------------------------------------

def resolve_option_tokens(
    instruments: pd.DataFrame,
    index_name: str,
    spot_price: float,
    n_strikes: int = 15,
    n_expiries: int = 2,
) -> dict[int, OptionToken]:
    """Resolve near-ATM option tokens from a cached instrument DataFrame.

    Parameters
    ----------
    instruments : pd.DataFrame
        Full NFO instrument dump (from kite.instruments("NFO")).
    index_name : str
        "NIFTY" or "BANKNIFTY".
    spot_price : float
        Current spot/futures price for ATM computation.
    n_strikes : int
        Number of strikes each side of ATM (total = 2*n_strikes + 1).
    n_expiries : int
        Number of nearest expiries to include.

    Returns
    -------
    dict[int, OptionToken]
        Mapping of instrument_token → OptionToken.
    """
    today = datetime.now(IST).date()

    opts = instruments[
        (instruments["name"] == index_name)
        & (instruments["instrument_type"].isin(["CE", "PE"]))
        & (instruments["expiry"] >= today)
    ].copy()

    if opts.empty:
        logger.warning("No options found for %s", index_name)
        return {}

    # Nearest n_expiries
    expiries = sorted(opts["expiry"].unique())[:n_expiries]
    opts = opts[opts["expiry"].isin(expiries)]

    # Find strikes near ATM
    strikes = sorted(opts["strike"].unique())
    if not strikes:
        return {}

    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
    lo = max(0, atm_idx - n_strikes)
    hi = min(len(strikes), atm_idx + n_strikes + 1)
    near_strikes = set(strikes[lo:hi])

    opts = opts[opts["strike"].isin(near_strikes)]

    tokens: dict[int, OptionToken] = {}
    storage_key = f"{index_name}_OPT"

    for _, row in opts.iterrows():
        exp = row["expiry"]
        if not isinstance(exp, date):
            exp = exp.date()

        ot = OptionToken(
            index_name=index_name,
            instrument_token=int(row["instrument_token"]),
            tradingsymbol=row["tradingsymbol"],
            expiry=exp,
            strike=float(row["strike"]),
            option_type=row["instrument_type"],
            storage_key=storage_key,
        )
        tokens[ot.instrument_token] = ot

    logger.info(
        "%s OPT: %d tokens, %d strikes (%.0f–%.0f), %d expiries, ATM≈%.0f",
        index_name, len(tokens), len(near_strikes),
        min(near_strikes), max(near_strikes),
        len(expiries), spot_price,
    )
    return tokens


def fetch_spot_prices(
    kite: KiteConnect,
    indices: list[str],
) -> dict[str, float]:
    """Fetch current spot prices for indices via REST quote."""
    symbols = [SPOT_SYMBOLS[name] for name in indices if name in SPOT_SYMBOLS]
    if not symbols:
        return {}

    quotes = kite.quote(symbols)
    prices: dict[str, float] = {}
    for name in indices:
        sym = SPOT_SYMBOLS.get(name)
        if sym and sym in quotes:
            prices[name] = quotes[sym]["last_price"]
    return prices


def check_recenter_needed(
    option_tokens: dict[int, OptionToken],
    spot_prices: dict[str, float],
    recenter_centers: dict[str, float],
) -> list[str]:
    """Return list of index names that need ATM recentering.

    Triggers when spot has moved ≥ RECENTER_STRIKE_MULTIPLES × strike_width
    from the last recenter center price.
    """
    indices_needing_recenter = []

    index_names = {ot.index_name for ot in option_tokens.values()}
    for name in index_names:
        current = spot_prices.get(name)
        center = recenter_centers.get(name)
        if current is None or center is None:
            continue

        width = STRIKE_WIDTHS.get(name, 50)
        threshold = width * RECENTER_STRIKE_MULTIPLES
        if abs(current - center) >= threshold:
            indices_needing_recenter.append(name)

    return indices_needing_recenter
