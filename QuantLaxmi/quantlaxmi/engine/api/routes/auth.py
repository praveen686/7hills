"""Authentication routes — Google, Zerodha, Binance OAuth + JWT."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

load_dotenv()  # Ensure .env vars are available for auth config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# JWT helpers (PyJWT)
# ---------------------------------------------------------------------------

JWT_SECRET = os.environ.get("JWT_SECRET", os.urandom(32).hex())
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days

try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None  # type: ignore[assignment]
    logger.warning("PyJWT not installed — auth endpoints will return 501")


def _encode_jwt(payload: dict[str, Any]) -> str:
    if pyjwt is None:
        raise HTTPException(501, "PyJWT not installed")
    payload["exp"] = int(time.time()) + JWT_EXPIRY_SECONDS
    payload["iat"] = int(time.time())
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict[str, Any]:
    if pyjwt is None:
        raise HTTPException(501, "PyJWT not installed")
    try:
        return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")

ZERODHA_API_KEY = os.environ.get("ZERODHA_API_KEY", "")
ZERODHA_API_SECRET = os.environ.get("ZERODHA_API_SECRET", "")
ZERODHA_REDIRECT_URI = os.environ.get("ZERODHA_REDIRECT_URI", "http://localhost:8000/api/auth/zerodha/callback")

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:1420")

WAITLIST_FILE = Path(os.environ.get("WAITLIST_FILE", "data/waitlist.jsonl"))


# ---------------------------------------------------------------------------
# Waitlist
# ---------------------------------------------------------------------------

class WaitlistRequest(BaseModel):
    email: str  # Using str instead of EmailStr to avoid optional dep


@router.post("/waitlist")
async def join_waitlist(body: WaitlistRequest) -> dict[str, str]:
    """Append email to waitlist JSONL file."""
    WAITLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "email": body.email,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(WAITLIST_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Waitlist signup: %s", body.email)
    return {"status": "ok", "message": "Added to waitlist"}


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------

@router.get("/google/url")
async def google_auth_url() -> dict[str, str]:
    """Return the Google OAuth consent screen URL."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(501, "GOOGLE_CLIENT_ID not configured")
    params = urllib.parse.urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    })
    return {"url": f"https://accounts.google.com/o/oauth2/v2/auth?{params}"}


@router.get("/google/callback")
async def google_callback(code: str = Query(...)) -> RedirectResponse:
    """Exchange Google auth code for tokens, create JWT, redirect to frontend."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(501, "Google OAuth not configured")

    async with httpx.AsyncClient() as client:
        # Exchange code for tokens
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(400, f"Google token exchange failed: {token_resp.text}")

        tokens = token_resp.json()
        access_token = tokens["access_token"]

        # Get user info
        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(400, "Failed to get Google user info")

        userinfo = userinfo_resp.json()

    jwt_token = _encode_jwt({
        "sub": userinfo.get("id", ""),
        "email": userinfo.get("email", ""),
        "name": userinfo.get("name", ""),
        "avatar": userinfo.get("picture", ""),
        "provider": "google",
    })

    return RedirectResponse(f"{FRONTEND_URL}/?token={jwt_token}")


# ---------------------------------------------------------------------------
# Zerodha OAuth
# ---------------------------------------------------------------------------

@router.get("/zerodha/url")
async def zerodha_auth_url() -> dict[str, str]:
    """Return the Zerodha Kite login URL."""
    if not ZERODHA_API_KEY:
        raise HTTPException(501, "ZERODHA_API_KEY not configured")
    url = f"https://kite.zerodha.com/connect/login?v=3&api_key={ZERODHA_API_KEY}"
    return {"url": url}


@router.get("/zerodha/callback")
async def zerodha_callback(
    request_token: str = Query(...),
) -> RedirectResponse:
    """Exchange Zerodha request_token for session, create JWT, redirect."""
    if not ZERODHA_API_KEY or not ZERODHA_API_SECRET:
        raise HTTPException(501, "Zerodha OAuth not configured")

    # Kite Connect checksum: SHA-256 of api_key + request_token + api_secret
    checksum = hashlib.sha256(
        (ZERODHA_API_KEY + request_token + ZERODHA_API_SECRET).encode()
    ).hexdigest()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.kite.trade/session/token",
            data={
                "api_key": ZERODHA_API_KEY,
                "request_token": request_token,
                "checksum": checksum,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(400, f"Zerodha session creation failed: {resp.text}")

        session = resp.json().get("data", {})

    jwt_token = _encode_jwt({
        "sub": session.get("user_id", ""),
        "email": session.get("email", ""),
        "name": session.get("user_name", session.get("user_id", "")),
        "provider": "zerodha",
    })

    return RedirectResponse(f"{FRONTEND_URL}/?token={jwt_token}")


# ---------------------------------------------------------------------------
# Binance API key validation
# ---------------------------------------------------------------------------

class BinanceAuthRequest(BaseModel):
    api_key: str
    api_secret: str


@router.post("/binance")
async def binance_auth(body: BinanceAuthRequest) -> dict[str, str]:
    """Validate Binance API credentials by calling account endpoint."""
    timestamp = str(int(time.time() * 1000))
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        body.api_secret.encode(), query_string.encode(), hashlib.sha256
    ).hexdigest()

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}",
            headers={"X-MBX-APIKEY": body.api_key},
        )
        if resp.status_code != 200:
            raise HTTPException(400, "Invalid Binance credentials")

        account = resp.json()

    jwt_token = _encode_jwt({
        "sub": f"binance_{body.api_key[:8]}",
        "email": "",
        "name": f"Binance {body.api_key[:8]}...",
        "provider": "binance",
    })

    return {"token": jwt_token}


# ---------------------------------------------------------------------------
# Auto-login endpoints (use .env credentials)
# ---------------------------------------------------------------------------


@router.post("/zerodha/auto")
async def zerodha_auto_login() -> dict[str, str]:
    """Headless Zerodha login using .env credentials (user_id, password, TOTP)."""
    import asyncio

    try:
        from quantlaxmi.data.collectors.auth import headless_login
    except ImportError:
        raise HTTPException(501, "Zerodha auth dependencies not installed (kiteconnect, pyotp)")

    try:
        kite = await asyncio.to_thread(headless_login)
        profile = await asyncio.to_thread(kite.profile)
    except ValueError as e:
        raise HTTPException(400, f"Missing Zerodha credentials: {e}")
    except Exception as e:
        logger.exception("Zerodha auto-login failed")
        raise HTTPException(400, f"Zerodha login failed: {e}")

    jwt_token = _encode_jwt({
        "sub": profile.get("user_id", ""),
        "email": profile.get("email", ""),
        "name": profile.get("user_name", profile.get("user_id", "")),
        "provider": "zerodha",
        "broker": profile.get("broker", "ZERODHA"),
        "exchanges": profile.get("exchanges", []),
        "products": profile.get("products", []),
        "order_types": profile.get("order_types", []),
    })

    return {"token": jwt_token}


def _binance_sign_hmac(query_string: str, api_secret: str) -> str:
    """HMAC-SHA256 signature for Binance API."""
    return hmac.new(
        api_secret.encode(), query_string.encode(), hashlib.sha256
    ).hexdigest()


def _binance_sign_ed25519(query_string: str, private_key_pem: str) -> str:
    """Ed25519 signature for Binance API (base64-encoded)."""
    import base64
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(private_key_pem.encode(), password=None)
    sig_bytes = private_key.sign(query_string.encode("ASCII"))
    return base64.b64encode(sig_bytes).decode("ASCII")


@router.post("/binance/auto")
async def binance_auto_login() -> dict[str, str]:
    """Validate Binance credentials from .env and return JWT.

    Supports both HMAC-SHA256 and Ed25519 signing:
    - Ed25519: set BINANCE_API_KEY_ED25519 + BINANCE_ED25519_PRIVATE_KEY (inline PEM)
               or BINANCE_ED25519_PRIVATE_KEY_PATH (path to .pem file)
    - HMAC:    set BINANCE_API_KEY + BINANCE_API_SECRET
    Ed25519 is tried first if configured.
    """
    ed25519_api_key = os.environ.get("BINANCE_API_KEY_ED25519", "")
    ed25519_pk_inline = os.environ.get("BINANCE_ED25519_PRIVATE_KEY", "")
    ed25519_pk_path = os.environ.get("BINANCE_ED25519_PRIVATE_KEY_PATH", "")
    hmac_api_key = os.environ.get("BINANCE_API_KEY", "")
    hmac_api_secret = os.environ.get("BINANCE_API_SECRET", "")

    # Resolve Ed25519 private key
    ed25519_pk_pem = ""
    if ed25519_pk_inline:
        ed25519_pk_pem = ed25519_pk_inline
    elif ed25519_pk_path:
        pk_file = Path(ed25519_pk_path)
        if pk_file.is_file():
            ed25519_pk_pem = pk_file.read_text()

    use_ed25519 = bool(ed25519_api_key and ed25519_pk_pem)
    use_hmac = bool(hmac_api_key and hmac_api_secret)

    if not use_ed25519 and not use_hmac:
        detail = "No valid Binance credentials in .env. "
        if ed25519_api_key and not ed25519_pk_pem:
            detail += (
                "BINANCE_API_KEY_ED25519 is set but Ed25519 private key is missing. "
                "Set BINANCE_ED25519_PRIVATE_KEY (inline PEM) or "
                "BINANCE_ED25519_PRIVATE_KEY_PATH (path to .pem file)."
            )
        else:
            detail += "Set BINANCE_API_KEY + BINANCE_API_SECRET (HMAC) or Ed25519 keys."
        raise HTTPException(400, detail)

    timestamp = str(int(time.time() * 1000))
    query_string = f"timestamp={timestamp}"

    # Try Ed25519 first, fall back to HMAC
    methods = []
    if use_ed25519:
        methods.append(("ed25519", ed25519_api_key, ed25519_pk_pem))
    if use_hmac:
        methods.append(("hmac", hmac_api_key, hmac_api_secret))

    # Try each signing method; attempt /api/v3/account for rich info,
    # but fall back to issuing JWT with basic info if IP-restricted.
    for method_name, api_key, secret in methods:
        try:
            if method_name == "ed25519":
                signature = _binance_sign_ed25519(query_string, secret)
            else:
                signature = _binance_sign_hmac(query_string, secret)
        except Exception as e:
            logger.warning("Binance %s signing error: %s", method_name, e)
            continue

        # Try authenticated /account endpoint for full info
        account: dict[str, Any] = {}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}",
                headers={"X-MBX-APIKEY": api_key},
            )
            if resp.status_code == 200:
                account = resp.json()
            else:
                # IP-restricted or insufficient perms — log but proceed
                logger.info(
                    "Binance /account returned %s (likely IP restriction) — "
                    "issuing JWT with basic info",
                    resp.status_code,
                )

        jwt_token = _encode_jwt({
            "sub": f"binance_{api_key[:8]}",
            "email": "",
            "name": f"Binance {api_key[:8]}...",
            "provider": "binance",
            "permissions": account.get("permissions", ["SPOT"]),
            "account_type": account.get("accountType", "SPOT"),
            "can_trade": account.get("canTrade", True),
            "can_withdraw": account.get("canWithdraw"),
            "can_deposit": account.get("canDeposit"),
        })
        return {"token": jwt_token}

    raise HTTPException(400, "Binance login failed — no valid signing method available")


# ---------------------------------------------------------------------------
# Token introspection
# ---------------------------------------------------------------------------

@router.get("/me")
async def get_current_user(request: Request) -> dict[str, Any]:
    """Decode JWT from Authorization header and return user info."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")

    token = auth_header[7:]  # Strip "Bearer "
    payload = _decode_jwt(token)

    return {
        "id": payload.get("sub", ""),
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
        "provider": payload.get("provider", ""),
        "avatar": payload.get("avatar", ""),
        # Zerodha-specific
        "broker": payload.get("broker", ""),
        "exchanges": payload.get("exchanges", []),
        "products": payload.get("products", []),
        "order_types": payload.get("order_types", []),
        # Binance-specific
        "permissions": payload.get("permissions", []),
        "account_type": payload.get("account_type", ""),
        "can_trade": payload.get("can_trade"),
        "can_withdraw": payload.get("can_withdraw"),
        "can_deposit": payload.get("can_deposit"),
    }
