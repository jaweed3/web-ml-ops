import hmac
import os

from fastapi import Header, HTTPException, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.components.runner import ONNXRunner
from app.constant import MAX_FILE_SIZE_BYTES

limiter = Limiter(key_func=get_remote_address)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_API_KEY = os.environ.get("API_KEY", "")  # empty = auth disabled


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not _API_KEY:
        return
    if not hmac.compare_digest(x_api_key or "", _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def check_content_length(content_length: str | None = Header(default=None)) -> None:
    if content_length is not None:
        try:
            size = int(content_length)
        except ValueError:
            return
        if size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"File too large ({size // 1024} KB). "
                f"Max {MAX_FILE_SIZE_BYTES // 1_048_576} MB.",
            )


def get_runner(request: Request) -> ONNXRunner:
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return runner
