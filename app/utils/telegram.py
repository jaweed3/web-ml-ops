import os

import httpx

from core.logger import get_logger

log = get_logger("utils.telegram")

_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
_API_BASE = "https://api.telegram.org"


def notify(message: str) -> None:
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
