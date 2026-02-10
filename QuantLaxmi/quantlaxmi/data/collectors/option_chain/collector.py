"""Option chain snapshot collector for NIFTY and BANKNIFTY.

Fetches full option chain from Zerodha Kite every N minutes and stores
as timestamped parquet files.  Also captures NIFTY/BANKNIFTY spot and
futures prices for basis computation.

Data schema per snapshot:
  timestamp, symbol, expiry, strike, option_type, ltp, oi, volume,
  bid_price, ask_price, bid_qty, ask_qty, underlying_price

Storage: data/india/chain_snapshots/{date}/{symbol}_{HHMMSS}.parquet
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Index instrument tokens (NSE)
INDEX_TOKENS = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "MIDCPNIFTY": 288009,
    "FINNIFTY": 257801,
}

# Lot sizes per index
LOT_SIZES = {
    "NIFTY": 75,
    "BANKNIFTY": 30,
    "MIDCPNIFTY": 75,
    "FINNIFTY": 65,
}

# NSE quote symbols for spot price
SPOT_SYMBOLS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
}

# Number of strikes around ATM to capture (each side)
STRIKES_EACH_SIDE = 30

SNAPSHOT_DIR = Path("data/india/chain_snapshots")


@dataclass
class IndexInstruments:
    """Options + futures instruments for a single index."""
    opts: pd.DataFrame          # CE + PE option instruments
    fut_token: int              # Near-month futures instrument_token
    fut_symbol: str             # e.g. "NIFTY26FEBFUT"


@dataclass
class InstrumentMap:
    """Cached mapping of option instruments for a trading day."""
    indices: dict[str, IndexInstruments]   # keyed by index name
    loaded_date: str


def load_instrument_map(
    kite: KiteConnect,
    symbols: list[str] | None = None,
) -> InstrumentMap:
    """Fetch NFO instrument dump and build maps for all requested indices."""
    if symbols is None:
        symbols = list(INDEX_TOKENS.keys())

    instruments = pd.DataFrame(kite.instruments("NFO"))
    today = datetime.now(IST).date()

    indices: dict[str, IndexInstruments] = {}

    for name in symbols:
        # Options
        opts = instruments[
            (instruments["name"] == name)
            & (instruments["instrument_type"].isin(["CE", "PE"]))
        ].copy()

        if opts.empty:
            logger.warning("No options found for %s, skipping", name)
            continue

        # Near-month futures (smallest expiry >= today)
        futs = instruments[
            (instruments["name"] == name) & (instruments["instrument_type"] == "FUT")
        ]
        futs = futs[futs["expiry"] >= today].sort_values("expiry")

        if futs.empty:
            logger.warning("No futures found for %s, skipping", name)
            continue

        fut_row = futs.iloc[0]

        indices[name] = IndexInstruments(
            opts=opts,
            fut_token=int(fut_row["instrument_token"]),
            fut_symbol=fut_row["tradingsymbol"],
        )

        logger.info(
            "%s: %d opts, fut=%s", name, len(opts), fut_row["tradingsymbol"],
        )

    return InstrumentMap(indices=indices, loaded_date=str(today))


def _select_near_strikes(
    opts: pd.DataFrame, spot: float, n_each_side: int = STRIKES_EACH_SIDE,
    max_expiries: int = 4,
) -> pd.DataFrame:
    """Select options within N strikes of ATM for the nearest expiries."""
    # Nearest expiries
    expiries = sorted(opts["expiry"].unique())[:max_expiries]
    subset = opts[opts["expiry"].isin(expiries)].copy()

    # Filter strikes near ATM
    strikes = sorted(subset["strike"].unique())
    # Find ATM strike
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    lo = max(0, atm_idx - n_each_side)
    hi = min(len(strikes), atm_idx + n_each_side + 1)
    near_strikes = set(strikes[lo:hi])

    return subset[subset["strike"].isin(near_strikes)]


def snapshot_chain(
    kite: KiteConnect,
    imap: InstrumentMap,
    symbol: str = "NIFTY",
) -> pd.DataFrame | None:
    """Take a single snapshot of the full option chain + spot + futures.

    Returns a DataFrame with one row per option contract, plus metadata.
    Returns None if the fetch fails.
    """
    idx = imap.indices.get(symbol)
    if idx is None:
        logger.error("No instruments loaded for %s", symbol)
        return None

    opts = idx.opts
    fut_symbol = idx.fut_symbol
    spot_sym = SPOT_SYMBOLS.get(symbol, f"NSE:{symbol}")

    now = datetime.now(IST)

    try:
        # Get spot price
        spot_quote = kite.quote([spot_sym])
        spot_price = list(spot_quote.values())[0]["last_price"]

        # Get futures price
        fut_quote = kite.quote([f"NFO:{fut_symbol}"])
        fut_price = list(fut_quote.values())[0]["last_price"]
        fut_oi = list(fut_quote.values())[0].get("oi", 0)
    except Exception as e:
        logger.error("Failed to fetch spot/futures for %s: %s", symbol, e)
        return None

    # Select relevant strikes (near ATM, first 4 expiries)
    near_opts = _select_near_strikes(opts, spot_price)

    if near_opts.empty:
        logger.warning("No strikes selected for %s (spot=%.1f)", symbol, spot_price)
        return None

    # Fetch quotes in batches (Kite allows ~500 per call)
    all_symbols = [f"NFO:{ts}" for ts in near_opts["tradingsymbol"]]
    rows = []

    batch_size = 450
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i : i + batch_size]
        try:
            quotes = kite.quote(batch)
        except Exception as e:
            logger.error("Quote fetch failed for batch %d: %s", i, e)
            continue

        for kite_sym, q in quotes.items():
            ts_name = kite_sym.replace("NFO:", "")
            inst_row = near_opts[near_opts["tradingsymbol"] == ts_name]
            if inst_row.empty:
                continue
            inst = inst_row.iloc[0]

            depth = q.get("depth", {})
            best_bid = depth.get("buy", [{}])[0] if depth.get("buy") else {}
            best_ask = depth.get("sell", [{}])[0] if depth.get("sell") else {}

            rows.append({
                "timestamp": now,
                "symbol": symbol,
                "expiry": str(inst["expiry"]),
                "strike": float(inst["strike"]),
                "option_type": inst["instrument_type"],
                "ltp": q.get("last_price", 0),
                "oi": q.get("oi", 0),
                "volume": q.get("volume", 0),
                "bid_price": best_bid.get("price", 0),
                "ask_price": best_ask.get("price", 0),
                "bid_qty": best_bid.get("quantity", 0),
                "ask_qty": best_ask.get("quantity", 0),
                "underlying_price": spot_price,
                "futures_price": fut_price,
                "futures_oi": fut_oi,
                "instrument_token": int(inst["instrument_token"]),
                "tradingsymbol": ts_name,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    logger.info(
        "%s snapshot: %d contracts, spot=%.1f, fut=%.1f, basis=%.2f",
        symbol, len(df), spot_price, fut_price, fut_price - spot_price,
    )
    return df


def save_snapshot(df: pd.DataFrame, symbol: str) -> Path:
    """Save snapshot as parquet file."""
    now = datetime.now(IST)
    day_dir = SNAPSHOT_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{now.strftime('%H%M%S')}.parquet"
    path = day_dir / filename
    df.to_parquet(path, index=False)
    return path


def list_snapshots(
    symbol: str | None = None,
    date: str | None = None,
) -> list[Path]:
    """List stored snapshot files, optionally filtered by symbol/date."""
    if date is None:
        date = datetime.now(IST).strftime("%Y-%m-%d")

    day_dir = SNAPSHOT_DIR / date
    if not day_dir.exists():
        return []

    files = sorted(day_dir.glob("*.parquet"))
    if symbol is not None:
        files = [f for f in files if f.name.startswith(f"{symbol}_")]
    return files


def run_collector(
    kite: KiteConnect,
    interval_seconds: int = 180,
    symbols: list[str] | None = None,
) -> None:
    """Run continuous collection loop.

    Snapshots NIFTY + BANKNIFTY chains every `interval_seconds`.
    Runs during market hours (9:15 AM - 3:30 PM IST).
    """
    if symbols is None:
        symbols = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

    imap = load_instrument_map(kite, symbols=symbols)

    logger.info(
        "Collector started: symbols=%s, interval=%ds",
        symbols, interval_seconds,
    )

    snap_count = 0

    while True:
        now = datetime.now(IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if now < market_open:
            wait = (market_open - now).total_seconds()
            logger.info("Market not open yet, waiting %.0f seconds", wait)
            time.sleep(min(wait, 60))
            continue

        if now > market_close:
            logger.info("Market closed for today")
            break

        # Reload instrument map if date changed
        today = str(now.date())
        if imap.loaded_date != today:
            imap = load_instrument_map(kite)

        for sym in symbols:
            try:
                df = snapshot_chain(kite, imap, sym)
                if df is not None:
                    path = save_snapshot(df, sym)
                    snap_count += 1
                    logger.info(
                        "Snap #%d saved: %s (%d rows)",
                        snap_count, path.name, len(df),
                    )
            except Exception as e:
                logger.error("Snapshot failed for %s: %s", sym, e, exc_info=True)

        time.sleep(interval_seconds)
