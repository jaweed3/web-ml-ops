from fastapi import APIRouter, HTTPException, Request

from app.components.runner import get_runner
from app.schema.meta import ModelInfoResponse, ModelVersionResponse

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """Full metadata about the currently loaded model."""
    try:
        runner = get_runner()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return ModelInfoResponse(
        name=request.app.state.model_name,
        version=runner.version,
        format=request.app.state.model_format,
        input_shape=runner.input_shape,
        loaded_at=runner.loaded_at,
    )


@router.get("/model/version", response_model=ModelVersionResponse)
def model_version() -> ModelVersionResponse:
    """Current model version."""
    try:
        return ModelVersionResponse(version=get_runner().version)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
