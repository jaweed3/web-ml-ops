import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.components.metrics import REQUEST_COUNT, REQUEST_DURATION
from app.utils.common import new_request_id
from core.logger import get_logger

log = get_logger("middleware.request")

# Paths that are too noisy to record as labelled metrics
_SKIP_METRIC_PATHS = {"/metrics", "/health", "/ready"}


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Per-request structured logging + Prometheus metric recording.

    Adds ``X-Request-ID`` header to every response.

    Log events:
    - ``request_received``  — method, path, request_id
    - ``request_complete``  — status_code, duration_ms, request_id
    """

    async def dispatch(self, request: Request, call_next):
        request_id = new_request_id()
        t0 = time.perf_counter()
        path = request.url.path

        log.info(
            "request_received",
            request_id=request_id,
            method=request.method,
            path=path,
        )

        response = await call_next(request)
        duration_s = time.perf_counter() - t0
        duration_ms = round(duration_s * 1000, 2)

        response.headers["X-Request-ID"] = request_id

        log.info(
            "request_complete",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # Record Prometheus metrics (skip noisy health/metrics paths)
        if path not in _SKIP_METRIC_PATHS:
            REQUEST_COUNT.labels(
                method=request.method,
                path=path,
                status_code=str(response.status_code),
            ).inc()
            REQUEST_DURATION.labels(method=request.method, path=path).observe(duration_s)

        return response
