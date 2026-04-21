import os

from fastapi import Header, HTTPException
from fastapi.security import APIKeyHeader

from app.constant import MAX_FILE_SIZE_BYTES

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_API_KEY = os.environ.get("API_KEY", "")  # empty → auth disabled (dev/local)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    FastAPI dependency — validates X-API-Key header.

    Behaviour
    ---------
    - If API_KEY env var is not set (empty string): auth is disabled,
      all requests pass through. Safe for local dev and CI.
    - If API_KEY is set: the header must be present and match exactly.
      Missing or wrong key → 401.
    """
    if not _API_KEY:
        return  # auth disabled
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def check_content_length(content_length: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency — rejects oversized uploads before reading bytes.

    Checks the Content-Length request header. If the declared size exceeds
    MAX_FILE_SIZE_BYTES the request is rejected immediately (422) without
    buffering the body into memory.

    Note: Content-Length may be absent (chunked transfer). In that case
    the per-read size check in the router acts as the final guard.
    """
    if content_length is not None:
        try:
            size = int(content_length)
        except ValueError:
            return  # malformed header — let router handle it
        if size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"File too large ({size // 1024} KB). Max {MAX_FILE_SIZE_BYTES // 1_048_576} MB.",
            )
