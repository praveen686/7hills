"""Settings management routes.

GET  /api/settings              -- Return all settings grouped by section, secrets masked
PUT  /api/settings              -- Update settings, persist to data/settings.json
POST /api/settings/test/{provider} -- Test connection for zerodha/binance/telegram
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

from quantlaxmi.data._paths import _PROJECT_ROOT

SETTINGS_FILE = _PROJECT_ROOT / "data" / "settings.json"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SettingsUpdate(BaseModel):
    """Partial settings update — keys are section names, values are dicts."""

    broker: dict[str, str] | None = None
    telegram: dict[str, str] | None = None
    system: dict[str, str] | None = None


class ConnectionTestResult(BaseModel):
    status: str  # "ok" | "error"
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map of settings keys to environment variable names
_ENV_MAP: dict[str, dict[str, str]] = {
    "broker": {
        "zerodha_user_id": "ZERODHA_USER_ID",
        "zerodha_api_key": "ZERODHA_API_KEY",
        "zerodha_api_secret": "ZERODHA_API_SECRET",
        "binance_api_key": "BINANCE_API_KEY",
        "binance_api_secret": "BINANCE_API_SECRET",
    },
    "telegram": {
        "telegram_api_id": "TELEGRAM_API_ID",
        "telegram_api_hash": "TELEGRAM_API_HASH",
        "telegram_phone": "TELEGRAM_PHONE",
    },
}

# Keys whose values should be masked in API responses
_SECRET_KEYS = {
    "zerodha_api_key",
    "zerodha_api_secret",
    "binance_api_key",
    "binance_api_secret",
    "telegram_api_hash",
}


def _mask(value: str) -> str:
    """Mask a secret value, showing only the last 4 characters."""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def _load_persisted() -> dict[str, dict[str, str]]:
    """Load persisted settings from settings.json, or empty dict."""
    if SETTINGS_FILE.is_file():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read settings.json: %s", exc)
    return {}


def _save_persisted(data: dict[str, dict[str, str]]) -> None:
    """Persist settings to settings.json."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def _resolve_value(section: str, key: str, persisted: dict[str, dict[str, str]]) -> str:
    """Resolve a setting value: persisted settings.json overrides env var."""
    # Check persisted first
    if section in persisted and key in persisted[section]:
        return persisted[section][key]
    # Fall back to env var
    env_name = _ENV_MAP.get(section, {}).get(key, "")
    if env_name:
        return os.environ.get(env_name, "")
    return ""


def _build_settings(masked: bool = True) -> dict[str, Any]:
    """Build the full settings response dict."""
    persisted = _load_persisted()
    result: dict[str, Any] = {}

    # Broker section
    broker: dict[str, Any] = {}
    for key in _ENV_MAP["broker"]:
        raw = _resolve_value("broker", key, persisted)
        broker[key] = _mask(raw) if (masked and key in _SECRET_KEYS and raw) else raw

    # Computed _configured flags
    broker["_zerodha_configured"] = all(
        bool(_resolve_value("broker", k, persisted))
        for k in ("zerodha_user_id", "zerodha_api_key", "zerodha_api_secret")
    )
    broker["_binance_configured"] = all(
        bool(_resolve_value("broker", k, persisted))
        for k in ("binance_api_key", "binance_api_secret")
    )
    result["broker"] = broker

    # Telegram section
    telegram: dict[str, Any] = {}
    for key in _ENV_MAP["telegram"]:
        raw = _resolve_value("telegram", key, persisted)
        telegram[key] = _mask(raw) if (masked and key in _SECRET_KEYS and raw) else raw

    telegram["_configured"] = all(
        bool(_resolve_value("telegram", k, persisted))
        for k in ("telegram_api_id", "telegram_api_hash", "telegram_phone")
    )
    result["telegram"] = telegram

    # System section (read-only)
    mode = _resolve_value("system", "mode", persisted) or os.environ.get("QUANTLAXMI_MODE", "paper")
    result["system"] = {
        "mode": mode,
        "data_dir": str(_PROJECT_ROOT / "common" / "data"),
    }

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("")
async def get_settings() -> dict[str, Any]:
    """Return all settings grouped by section, with secrets masked."""
    return _build_settings(masked=True)


@router.put("")
async def update_settings(body: SettingsUpdate) -> dict[str, Any]:
    """Update settings, persist to data/settings.json, return masked response."""
    persisted = _load_persisted()

    for section_name in ("broker", "telegram", "system"):
        incoming = getattr(body, section_name, None)
        if incoming is None:
            continue

        # System section is read-only except for mode
        if section_name == "system":
            if "mode" in incoming and incoming["mode"] in ("paper", "live"):
                persisted.setdefault("system", {})["mode"] = incoming["mode"]
            continue

        # Validate keys
        valid_keys = set(_ENV_MAP.get(section_name, {}).keys())
        for key, value in incoming.items():
            if key.startswith("_"):
                continue  # Skip computed fields
            if key not in valid_keys:
                raise HTTPException(
                    422,
                    f"Unknown setting key '{key}' in section '{section_name}'. "
                    f"Valid keys: {sorted(valid_keys)}",
                )
            # Don't persist masked values (asterisks)
            if value and not value.startswith("*"):
                persisted.setdefault(section_name, {})[key] = value

    _save_persisted(persisted)
    logger.info("Settings updated and persisted to %s", SETTINGS_FILE)
    return _build_settings(masked=True)


@router.post("/test/{provider}")
async def test_connection(provider: str) -> ConnectionTestResult:
    """Test connection for a given provider (zerodha, binance, telegram)."""
    persisted = _load_persisted()

    if provider == "zerodha":
        api_key = _resolve_value("broker", "zerodha_api_key", persisted)
        if not api_key:
            return ConnectionTestResult(
                status="error",
                message="Zerodha API key is not configured.",
            )
        try:
            from kiteconnect import KiteConnect  # type: ignore[import-untyped]

            kite = KiteConnect(api_key=api_key)
            return ConnectionTestResult(
                status="ok",
                message=f"KiteConnect SDK loaded. API key ending in ...{api_key[-4:]} is set. "
                "Full auth requires login flow.",
            )
        except ImportError:
            return ConnectionTestResult(
                status="error",
                message="kiteconnect package is not installed. Run: pip install kiteconnect",
            )
        except Exception as exc:
            return ConnectionTestResult(status="error", message=str(exc))

    elif provider == "binance":
        api_key = _resolve_value("broker", "binance_api_key", persisted)
        if not api_key:
            return ConnectionTestResult(
                status="error",
                message="Binance API key is not configured.",
            )
        try:
            from binance.client import Client  # type: ignore[import-untyped]

            return ConnectionTestResult(
                status="ok",
                message=f"Binance SDK loaded. API key ending in ...{api_key[-4:]} is set.",
            )
        except ImportError:
            return ConnectionTestResult(
                status="error",
                message="python-binance package is not installed. Run: pip install python-binance",
            )
        except Exception as exc:
            return ConnectionTestResult(status="error", message=str(exc))

    elif provider == "telegram":
        api_id = _resolve_value("telegram", "telegram_api_id", persisted)
        if not api_id:
            return ConnectionTestResult(
                status="error",
                message="Telegram API ID is not configured.",
            )
        return ConnectionTestResult(
            status="ok",
            message=f"Telegram API ID ({api_id}) is set. Session auth required separately.",
        )

    else:
        raise HTTPException(404, f"Unknown provider: {provider}. Use zerodha, binance, or telegram.")
