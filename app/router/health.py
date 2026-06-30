from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from app.components.runner import ONNXRunner
from app.dependencies import get_runner
from app.schema.health import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/ready", response_model=ReadyResponse, responses={503: {"description": "Model not ready"}}
)
def ready(runner: ONNXRunner = Depends(get_runner)) -> Response:
    if not runner.is_ready:
        return JSONResponse(
            status_code=503,
            content=ReadyResponse(status="loading").model_dump(),
        )
    loaded_at = runner.loaded_at
    assert loaded_at is not None
    return ReadyResponse(
        status="ready",
        model_version=runner.version,
        loaded_at=datetime.fromtimestamp(loaded_at, tz=timezone.utc).isoformat(),
    )
