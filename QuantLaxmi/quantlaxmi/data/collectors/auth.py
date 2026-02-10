"""Headless Zerodha Kite login with TOTP.

Automates the browser-based OAuth flow:
  1. POST /api/login with user_id + password
  2. POST /api/twofa with TOTP
  3. Follow redirect to extract request_token
  4. generate_session() → access_token

Caches access_token to disk (valid until ~6:00 AM IST next day).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from dotenv import load_dotenv
from kiteconnect import KiteConnect

from quantlaxmi.data._paths import ZERODHA_SESSION_CACHE

logger = logging.getLogger(__name__)

TOKEN_CACHE = ZERODHA_SESSION_CACHE


def _load_env() -> dict[str, str]:
    """Load Zerodha credentials from .env."""
    for p in (Path(".env"), Path("../.env")):
        if p.exists():
            load_dotenv(p)
            break
    return {
        "user_id": os.getenv("ZERODHA_USER_ID", ""),
        "password": os.getenv("ZERODHA_PASSWORD", ""),
        "totp_secret": os.getenv("ZERODHA_TOTP_SECRET", ""),
        "api_key": os.getenv("ZERODHA_API_KEY", ""),
        "api_secret": os.getenv("ZERODHA_API_SECRET", ""),
    }


def _is_token_valid(cached: dict) -> bool:
    """Check if cached access_token is still valid (same calendar day IST)."""
    ts = cached.get("timestamp", 0)
    if not ts:
        return False
    # Kite tokens expire at ~6:00 AM IST next day.
    # Conservative: treat as valid if generated today (IST = UTC+5:30).
    from datetime import timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    cached_day = datetime.fromtimestamp(ts, tz=ist).date()
    today = datetime.now(ist).date()
    return cached_day == today


def _load_cached_token() -> str | None:
    """Load access_token from disk if still valid."""
    if not TOKEN_CACHE.exists():
        return None
    try:
        cached = json.loads(TOKEN_CACHE.read_text())
        if _is_token_valid(cached):
            logger.info("Using cached Zerodha access_token")
            return cached["access_token"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_token(access_token: str) -> None:
    """Persist access_token to disk."""
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "access_token": access_token,
        "timestamp": time.time(),
    }))
    tmp.rename(TOKEN_CACHE)


def headless_login(env: dict[str, str] | None = None) -> KiteConnect:
    """Perform headless Zerodha login, return authenticated KiteConnect.

    Tries cached token first, falls back to full login flow.
    """
    if env is None:
        env = _load_env()

    api_key = env["api_key"]
    api_secret = env["api_secret"]

    # Try cached token
    cached = _load_cached_token()
    if cached:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(cached)
        # Validate by calling profile
        try:
            kite.profile()
            return kite
        except Exception:
            logger.info("Cached token invalid, performing fresh login")

    # Full headless login
    user_id = env["user_id"]
    password = env["password"]
    totp_secret = env["totp_secret"]

    if not all([user_id, password, totp_secret, api_key, api_secret]):
        raise ValueError(
            "Missing Zerodha credentials. Set ZERODHA_USER_ID, ZERODHA_PASSWORD, "
            "ZERODHA_TOTP_SECRET, ZERODHA_API_KEY, ZERODHA_API_SECRET in .env"
        )

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "X-Kite-Version": "3",
    })

    # Step 1: Login with user_id + password
    r = session.post(
        "https://kite.zerodha.com/api/login",
        data={"user_id": user_id, "password": password},
        timeout=15,
    )
    r.raise_for_status()
    login_data = r.json()
    if login_data.get("status") != "success":
        raise RuntimeError(f"Zerodha login failed: {login_data}")

    request_id = login_data["data"]["request_id"]
    logger.info("Login step 1 OK (request_id=%s)", request_id)

    # Step 2: TOTP
    totp_value = pyotp.TOTP(totp_secret).now()
    r = session.post(
        "https://kite.zerodha.com/api/twofa",
        data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp_value,
            "twofa_type": "totp",
        },
        timeout=15,
    )
    r.raise_for_status()
    twofa_data = r.json()
    if twofa_data.get("status") != "success":
        raise RuntimeError(f"Zerodha 2FA failed: {twofa_data}")

    logger.info("Login step 2 OK (2FA passed)")

    # Step 3: Get request_token via redirect.
    # After login on kite.zerodha.com, the connect/login URL on the same
    # domain should auto-authorize and redirect to our redirect_url with
    # ?request_token=xxx&status=success.
    # We follow redirects step by step to catch the request_token in the
    # final redirect URL (which may go to our redirect_url that doesn't exist).
    connect_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"

    request_token = None
    url = connect_url

    # Follow up to 10 redirects, checking each Location for request_token
    for _ in range(10):
        r = session.get(url, allow_redirects=False, timeout=15)
        if r.status_code in (301, 302, 303, 307):
            location = r.headers.get("Location", "")
            parsed = urlparse(location)
            qs = parse_qs(parsed.query)
            if "request_token" in qs:
                request_token = qs["request_token"][0]
                break
            url = location
        else:
            break

    if not request_token:
        raise RuntimeError(
            "Could not extract request_token. "
            "The Kite app may need manual authorization first. "
            f"Visit: {connect_url}"
        )

    logger.info("Got request_token from redirect")

    # Step 4: Generate session
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    kite.set_access_token(access_token)

    # Cache for reuse
    _save_token(access_token)
    logger.info("Zerodha session created and cached")

    return kite
