"""
Telegram notification utility.

Usage:
    from app.utils.telegram import notify

    notify("✅ Model promoted to Production")

Environment variables:
    TELEGRAM_TOKEN   — bot token from @BotFather
    TELEGRAM_CHAT_ID — target chat / group id

If either env var is not set, notify() is a no-op (safe for local dev / CI
environments that don't have Telegram configured).
"""

import os

import httpx

from core.logger import get_logger

log = get_logger("utils.telegram")

_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
_API_BASE = "https://api.telegram.org"


def notify(message: str) -> None:
    """
    Send a Markdown-formatted message to the configured Telegram chat.
    Silently swallows errors so a notification failure never crashes the caller.
    """
    if not _TOKEN or not _CHAT_ID:
        return

    try:
        r = httpx.post(
            f"{_API_BASE}/bot{_TOKEN}/sendMessage",
            json={
                "chat_id": _CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        r.raise_for_status()
    except Exception as exc:
        log.warning("telegram_notify_failed", error=str(exc))
