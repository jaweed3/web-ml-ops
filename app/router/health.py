from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.components.runner import get_runner
from app.schema.health import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe — returns 200 if the process is alive."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready", response_model=ReadyResponse, responses={503: {"description": "Model not ready"}}
)
def ready():
    """
    Readiness probe — returns 200 only if the model is loaded and ready.
    Returns 503 while the model is still downloading / initializing.
    """
    try:
        runner = get_runner()
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(status="not_initialized").model_dump(),
        )

    if not runner.is_ready:
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(status="loading").model_dump(),
        )

    return ReadyResponse(
        status="ready",
        model_version=runner.version,
        loaded_at=datetime.fromtimestamp(runner.loaded_at, tz=timezone.utc).isoformat(),
    )
